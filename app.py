from flask import Flask, render_template, request, jsonify
import io, base64, json, random, re, difflib
from PIL import Image
import numpy as np
import easyocr
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

# ---------- Load Model & Data ----------
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

with open("medicine_data_cleaned.json", "r", encoding="utf-8") as f:
    RAW_DATA = json.load(f)

# easyocr reader
reader = easyocr.Reader(["en"], gpu=False)

# ---------- Helpers ----------
def normalize_string(s: str) -> str:
    """Lowercase, remove non-alphanum, collapse spaces."""
    if not s:
        return ""
    s = re.sub(r'[^a-z0-9]+', ' ', s.lower())
    return s.strip()

def normalize_dataset(raw_dataset):
    """Normalize keys (capitalization, spacing) for consistent search."""
    canonical_map = {
        "drug_name": ["drug name", "drug_name", "medicine", "name"],
        "company_name": ["company name", "manufacturer", "company"],
        "active_ingredient": ["active ingredient", "ingredient", "salt"],
        "indication": ["indication", "use", "uses"],
        "dosage_and_administration": ["dosage and administration", "dosage", "dose"],
        "side_effects": ["side effects", "adverse effects"],
        "use_in_pregnancy": ["use in pregnancy", "pregnancy safety"]
    }

    normalized = []
    for raw_item in raw_dataset:
        key_map = {re.sub(r'[^a-z0-9]+', '', k.lower()): v for k, v in raw_item.items()}
        item = {}
        for canonical, variants in canonical_map.items():
            val = ""
            for v in variants:
                key = re.sub(r'[^a-z0-9]+', '', v)
                if key in key_map and key_map[key]:
                    val = key_map[key]
                    break
            item[canonical] = val
        item["_raw"] = raw_item
        item["_norm_drug"] = normalize_string(item.get("drug_name", ""))
        item["_norm_active"] = normalize_string(item.get("active_ingredient", ""))
        normalized.append(item)
    return normalized

DATASET = normalize_dataset(RAW_DATA)

# ---------- Search ----------
def search_dataset(name, max_results=5):
    """Fuzzy + token subset search in normalized dataset."""
    if not name or len(name.strip()) < 2:
        return []
    name_norm = normalize_string(name)
    results = []

    for item in DATASET:
        dn = item.get("_norm_drug", "")
        ai = item.get("_norm_active", "")
        if not dn and not ai:
            continue
        if name_norm == dn or name_norm in dn or all(t in dn for t in name_norm.split()):
            results.append(item)
        elif name_norm in ai:
            results.append(item)
    if not results:
        all_drugs = [i["_norm_drug"] for i in DATASET if i.get("_norm_drug")]
        close = difflib.get_close_matches(name_norm, all_drugs, n=max_results, cutoff=0.6)
        for c in close:
            match = next((i for i in DATASET if i["_norm_drug"] == c), None)
            if match:
                results.append(match)
    return results[:max_results]

# ---------- OCR Extraction ----------
def parse_ocr_text(ocr_lines):
    if not ocr_lines:
        return []
    full_text = " ".join(ocr_lines)
    primary_pattern = re.compile(r'\bTAB\b\s+(.+?)\s+([0-9](?:[+\-][0-9]){2})', flags=re.I)
    matches = primary_pattern.findall(full_text)
    parsed = []
    if matches:
        for name, dose in matches:
            parsed.append({"drug_name": re.sub(r'\s+', ' ', name).strip(), "dosage": dose})
        return parsed
    fallback = re.findall(r'\b([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)*)\b', full_text)
    return [{"drug_name": f.strip()} for f in fallback if len(f.strip()) > 2]

def extract_meds_from_text(text):
    if isinstance(text, list):
        parsed = parse_ocr_text(text)
    else:
        parsed = parse_ocr_text([text])
    if parsed:
        return [p["drug_name"] for p in parsed], parsed
    return [], []

# ---------- Alternatives ----------
def get_alternatives_by_active_ingredient(main_doc, max_results=3):
    main_active = main_doc.get("active_ingredient", "").lower()
    main_drug = main_doc.get("drug_name", "").lower()
    if not main_active:
        return []
    alts = []
    for i in DATASET:
        act = i.get("active_ingredient", "").lower()
        if act == main_active and i.get("drug_name", "").lower() != main_drug:
            alts.append(i["_raw"])
        if len(alts) >= max_results:
            break
    if not alts:
        alts.append({"Drug Name": "🙏 No alternates available in my dataset"})
    return alts

# ---------- Summary Generation ----------
def build_prompt(item):
    """Optimized prompt set for BART (avoids instruction repetition)."""
    variants = [
        f"{item['Drug Name']} is a medicine by {item['Company Name']} containing {item['Active Ingredient']}. "
        f"It is used for {item['Indication']}. Recommended dosage: {item['Dosage and Administration']}. "
        f"Possible side effects include {item['Side Effects']}. Use in pregnancy: {item['Use in pregnancy']}.",
        
        f"Drug: {item['Drug Name']} | Company: {item['Company Name']} | Ingredient: {item['Active Ingredient']} | "
        f"Use: {item['Indication']} | Dosage: {item['Dosage and Administration']} | "
        f"Side Effects: {item['Side Effects']} | Pregnancy: {item['Use in pregnancy']}.",
        
        f"{item['Drug Name']} ({item['Active Ingredient']}) — manufactured by {item['Company Name']}. "
        f"Indication: {item['Indication']}. Dosage: {item['Dosage and Administration']}. "
        f"Side Effects: {item['Side Effects']}. Pregnancy: {item['Use in pregnancy']}.",
        
        #f"Medical summary of {item['Drug Name']}:\n"
        f"Company: {item['Company Name']}\n"
        f"Active Ingredient: {item['Active Ingredient']}\n"
        f"Indication: {item['Indication']}\n"
        f"Dosage: {item['Dosage and Administration']}\n"
        f"Side Effects: {item['Side Effects']}\n"
        f"Use in pregnancy: {item['Use in pregnancy']}\n",
        
        f"{item['Drug Name']} is used for {item['Indication']}. "
        f"Contains {item['Active Ingredient']} and is produced by {item['Company Name']}. "
        f"Dosage: {item['Dosage and Administration']}. Side Effects: {item['Side Effects']}. "
        f"Pregnancy use: {item['Use in pregnancy']}.",
        
        f"{item['Drug Name']} — made by {item['Company Name']} — contains {item['Active Ingredient']}. "
        f"Indicated for {item['Indication']}. Usual dose: {item['Dosage and Administration']}. "
        f"Side effects: {item['Side Effects']}. Pregnancy: {item['Use in pregnancy']}.",
        
        f"{item['Indication']}. "
        f"Manufacturer: {item['Company Name']}. Ingredient: {item['Active Ingredient']}. "
        f"Dosage: {item['Dosage and Administration']}. Side effects: {item['Side Effects']}. "
        f"Pregnancy info: {item['Use in pregnancy']}."
    ]
    return random.choice(variants)



def generate_summary_text(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    output_ids = model.generate(
        input_ids,
        max_length=250,
        num_beams=4,
        repetition_penalty=1.2,
        early_stopping=True
    )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    summary = re.sub(r'^(write|create|summarize|describe|provide|generate).*?:', '', summary, flags=re.I).strip()
    summary += "\n"

    return summary

# ---------- Flask Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    data = request.get_json()
    img_b64 = data.get("image", "")
    if img_b64.startswith("data:image"):
        img_b64 = img_b64.split(",")[1]

    img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    ocr_results = reader.readtext(np.array(img), detail=0, paragraph=True)

    med_names, _ = extract_meds_from_text(ocr_results)
    if not med_names:
        return jsonify({"ok": False, "message": "No medicines detected."})

    found_docs = []
    for med in med_names:
        docs = search_dataset(med)
        if docs:
            found_docs.append(docs[0])

    if not found_docs:
        return jsonify({"ok": False, "message": "No matching medicines found in dataset."})

    summaries = []
    retrieved_data = []

    for doc in found_docs:
        item = doc["_raw"]
        prompt = build_prompt(item)
        summary = generate_summary_text(prompt)
        alternatives = get_alternatives_by_active_ingredient(doc)
        item_copy = dict(item)
        item_copy["Alternative Medicines"] = alternatives
        summaries.append({"drug_name": item["Drug Name"], "summary": "\n" + summary})
        retrieved_data.append(item_copy)
    if summaries:
        summaries[0]["summary"] = summaries[0]["summary"].lstrip("\n")

    return jsonify({
        "ok": True,
        "detected": med_names,
        "summaries": summaries,
        "retrieved": retrieved_data
    })

@app.route("/analyze-name", methods=["POST"])
def analyze_name():
    data = request.get_json()
    q = data.get("name", "").strip()
    if not q:
        return jsonify({"ok": False, "error": "No medicine name provided"})
    names = [n.strip() for n in q.split(",") if n.strip()]
    summaries = []
    retrieved_data = []

    for name in names:
        docs = search_dataset(name)
        if not docs:
            continue
        doc = docs[0]
        item = doc["_raw"]
        prompt = build_prompt(item)
        summary = generate_summary_text(prompt)
        alternatives = get_alternatives_by_active_ingredient(doc)
        item_copy = dict(item)
        item_copy["Alternative Medicines"] = alternatives
        summaries.append({"drug_name": item["Drug Name"], "summary": summary})
        retrieved_data.append(item_copy)

    if not summaries:
        return jsonify({"ok": False, "message": "No valid medicine found"})
    return jsonify({"ok": True, "summaries": summaries, "retrieved": retrieved_data})

@app.route("/regenerate", methods=["POST"])
@app.route("/regenerate", methods=["POST"])
def regenerate():
    data = request.get_json()
    q = data.get("name", "").strip()
    docs = search_dataset(q)
    if not docs:
        return jsonify({"ok": False, "error": "Medicine not found"})

    doc = docs[0]
    item = doc["_raw"]
    alternatives = get_alternatives_by_active_ingredient(doc)
    item_copy = dict(item)
    item_copy["Alternative Medicines"] = alternatives

    previous_summary = data.get("previous_summary", "")
    new_summary = ""
    attempts = 0
    max_attempts = 5

    while attempts < max_attempts:
        prompt = build_prompt(item)
        candidate = generate_summary_text(prompt)
        # compare stripped summaries
        if candidate.strip() != previous_summary.strip():
            new_summary = candidate
            break
        attempts += 1

    # fallback if all attempts fail
    if not new_summary:
        new_summary = candidate

    return jsonify({
        "ok": True,
        "summaries": [{"drug_name": item["Drug Name"], "summary": "\n" + new_summary}],
        "retrieved": [item_copy]
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)
