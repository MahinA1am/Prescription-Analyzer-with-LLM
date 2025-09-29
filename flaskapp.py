import json
import random
import pandas as pd
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
import easyocr
import re
import tempfile
import os

# Streamlit config — must be first Streamlit call
st.set_page_config(page_title="Medical Info Chatbot 💊", layout="centered")

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# Load data with caching
@st.cache_data
def load_data():
    with open("medicine_data_cleaned.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.dropna(subset=["Drug Name"])
    return df

df = load_data()

def extract_medicines_from_image(img):
    reader = easyocr.Reader(['en'])

    # Step 1: Save image to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_path = tmp_file.name
        img.save(tmp_path)

    # Step 2: Run OCR after file is saved and closed
    result = reader.readtext(tmp_path, detail=0)

    # Step 3: Remove the temp file after OCR is done
    try:
        os.remove(tmp_path)
    except PermissionError:
        print(f"⚠️ Could not delete temp file: {tmp_path}. It's still in use.")

    # Step 4: Reconstruct lines like TAB XYZ, CAP XYZ, INJ XYZ, SYR XYZ, handling various formats
    grouped = []
    i = 0
    while i < len(result):
        # Check if current line starts with a valid medicine type (case-insensitive)
        if result[i].strip().upper() in ['TAB', 'CAP', 'INJ', 'SYR']:
            line = result[i].strip() + " "
            i += 1
            # Check if next line exists and is not a new medicine type
            if i < len(result) and result[i].strip().upper() not in ['TAB', 'CAP', 'INJ', 'SYR']:
                # Handle cases like "TAB - A", "TAB A", "TAB – A", or "TAB\nA"
                line += result[i].strip()
                i += 1
            grouped.append(line.strip())
        else:
            # Handle standalone medicine names or other text
            grouped.append(result[i].strip())
            i += 1

    # Step 5: Extract medicine names using regex
    # Regex to match medicine type (TAB, CAP, INJ, SYR) followed by optional separator (-, –, space) and name
    pattern = re.compile(r"^(TAB|CAP|INJ|SYR)\s*[-–]?\s*([A-Z\s]+)$", re.IGNORECASE)
    extracted = []
    for line in grouped:
        match = pattern.search(line)
        if match:
            _, name = match.groups()
            extracted.append(name.strip().title())
        else:
            # Handle cases where only medicine name appears (e.g., standalone "A")
            name_pattern = re.compile(r"^[A-Z\s]+$", re.IGNORECASE)
            if name_pattern.search(line):
                extracted.append(line.strip().title())

    return extracted


# 📸 Image Upload with "+"
with st.expander("➕ Upload Prescription Image (Optional)", expanded=False):
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file:
        img = Image.open(image_file)
        st.image(img, caption="Uploaded Prescription", use_column_width=True)
        extracted_meds = extract_medicines_from_image(img)
        if extracted_meds:
            st.success(f"🧠 Extracted Medicines: {', '.join(extracted_meds)}")
            st.session_state["ocr_meds"] = ", ".join(extracted_meds)
        else:
            st.warning("⚠️ No medicine names detected in the image.")

# UI Title and description
st.title("💊 Medical Info Chatbot")
st.markdown("Get a quick summary and structured details of medicine(s). Enter one or more names separated by commas.")

# Use OCR meds if available
default_text = st.session_state.get("ocr_meds", "")
user_input = st.text_input("Medicine Name(s)", default_text)

# Prompt builder
def build_prompt(item):
    variants = [
        (
            f"Write a medical summary for the following drug in one clear paragraph. "
            f"Be sure to include what it's used for and if it's safe during pregnancy:\n"
            f"- Drug Name: {item['Drug Name']}\n"
            f"- Company Name: {item['Company Name']}\n"
            f"- Active Ingredient: {item['Active Ingredient']}\n"
            f"- Indication: {item['Indication']}\n"
            f"- Dosage and Administration: {item['Dosage and Administration']}\n"
            f"- Side Effects: {item['Side Effects']}\n"
            f"- Use in pregnancy: {item['Use in pregnancy']}\n"
        ),
        (
            f"Summarize the medicine below in a paragraph. Be sure to mention its use and pregnancy safety:\n"
            f"Drug: {item['Drug Name']} | Company: {item['Company Name']} | Ingredient: {item['Active Ingredient']} | "
            f"Indication: {item['Indication']} | Dosage: {item['Dosage and Administration']} | "
            f"Side Effects: {item['Side Effects']} | Pregnancy: {item['Use in pregnancy']}."
        ),
        (
            f"Create a brief but complete medical description of this drug, and ensure that indication and pregnancy use are included:\n"
            f"{item['Drug Name']} by {item['Company Name']} contains {item['Active Ingredient']}. "
            f"It is used for {item['Indication']}. "
            f"Recommended dosage is {item['Dosage and Administration']}. "
            f"Possible side effects include {item['Side Effects']}. "
            f"Pregnancy safety: {item['Use in pregnancy']}."
        ),
    ]
    return random.choice(variants)

def generate_summary_text(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    output_ids = model.generate(
        input_ids,
        max_length=320,
        num_beams=5,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_summary_until_different(item, old_summary, max_attempts=5):
    for _ in range(max_attempts):
        prompt = build_prompt(item)
        new_summary = generate_summary_text(prompt)
        if (
            new_summary.strip() != old_summary.strip()
            and "pregnancy" in new_summary.lower()
            and "indication" in new_summary.lower()
        ):
            return new_summary

    final_summary = new_summary.strip()
    if "indication" not in final_summary.lower():
        final_summary += f" This drug is indicated for {item['Indication'].strip()}."
    if "pregnancy" not in final_summary.lower():
        final_summary += f" Use during pregnancy: {item['Use in pregnancy'].strip()}."
    return final_summary

def get_item(drug_name):
    row = df[df["Drug Name"].str.lower() == drug_name.lower()]
    if row.empty:
        return None, f"No information found for '{drug_name}'."
    return row.iloc[0], None

def find_alternates(active_ingredient, current_drug_name, df, max_alternates=2):
    filtered = df[
        (df["Active Ingredient"].str.lower() == active_ingredient.lower()) &
        (df["Drug Name"].str.lower() != current_drug_name.lower())
    ]
    alternates = filtered["Drug Name"].tolist()[:max_alternates]
    
    # If no alternates found, return a special message
    if not alternates:
        return ["🙏 No alternates available in my dataset"]
    return alternates


def generate_bullet_list(item, alternates=None):
    alt_str = ""
    if alternates:
        alt_list_str = ", ".join(alternates)
        alt_str = f'<li><strong>🔄 Alternate Medicines:</strong> {alt_list_str}</li>'
    else:
        alt_str = '<li><strong>🔄 Alternate Medicines:</strong> 🙏 No alternates available in my dataset</li>'

    return f"""
    <ul style="list-style: none; padding-left: 0;">
        <li><strong>💊 Medicine Name:</strong> {item['Drug Name']}</li>
        <li><strong>🏢 Company Name:</strong> {item['Company Name']}</li>
        <li><strong>🧪 Active Ingredient:</strong> {item['Active Ingredient']}</li>
        <li><strong>🎯 Indication:</strong> {item['Indication']}</li>
        <li><strong>💉 Dosage and Administration:</strong> {item['Dosage and Administration']}</li>
        <li><strong>⚠️ Side Effects:</strong> {item['Side Effects']}</li>
        <li><strong>🤰 Use in pregnancy:</strong> {item['Use in pregnancy']}</li>
        {alt_str}
    </ul>
    """



if user_input:
    drug_names = [name.strip() for name in user_input.split(",") if name.strip()]
    found = False

    for drug in drug_names:
        item, error = get_item(drug)
        if error:
            st.warning(error)
            continue

        # Find alternate medicines by active ingredient
        alternates = find_alternates(item['Active Ingredient'], item['Drug Name'], df)

        key = f"summary_{drug.lower()}"
        prev_key = f"prev_summary_{drug.lower()}"

        if prev_key not in st.session_state:
            st.session_state[prev_key] = ""

        if st.button(f"🔁 Regenerate summary for {item['Drug Name']}", key=f"regen_{key}"):
            prompt = build_prompt(item)
            summary = generate_summary_until_different(item, st.session_state[prev_key])
            st.session_state[key] = summary
            st.session_state[prev_key] = summary

        if key not in st.session_state:
            prompt = build_prompt(item)
            summary = generate_summary_text(prompt)
            st.session_state[key] = summary
            st.session_state[prev_key] = summary

        summary = st.session_state[key]
        bullet_list = generate_bullet_list(item, alternates)

        if len(summary.split()) < 30:
            st.warning(f"⚠️ Summary for '{item['Drug Name']}' may be incomplete. Try regenerating or editing input.")

        st.markdown(f"""
        <div style="background-color: #ffffff; padding: 20px; margin-top: 20px;
                    border-left: 6px solid #1a73e8; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
           <h3 style="color: #1a73e8;">🧾 Summary for: <span style="color: #34a853;">{item['Drug Name']}</span></h3>
           <p style="font-size: 16px; line-height: 1.6;">{summary}</p>
           <hr>
           <div style="font-size: 15px; line-height: 1.8;">{bullet_list}</div>
        </div>
        """, unsafe_allow_html=True)

        found = True

    if not found:
        st.info("No valid medicine names found. Please check your input.")


# Custom CSS
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
    font-family: 'Segoe UI', sans-serif;
}
h3 {
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)
