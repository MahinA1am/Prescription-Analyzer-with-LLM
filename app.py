import json
import random
import pandas as pd
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

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

# Build prompt variants for T5 input
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

# Generate summary text from prompt
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

# New Summary
def generate_summary_until_different(item, old_summary, max_attempts=5):
    for _ in range(max_attempts):
        prompt = build_prompt(item)
        new_summary = generate_summary_text(prompt)

        # Check if it's different and already contains required fields
        if (
            new_summary.strip() != old_summary.strip()
            and "pregnancy" in new_summary.lower()
            and "indication" in new_summary.lower()
        ):
            return new_summary

    # If retries exhausted or fields missing, append missing info
    final_summary = new_summary.strip()

    if "indication" not in final_summary.lower():
        final_summary += f" This drug is indicated for {item['Indication'].strip()}."

    if "pregnancy" not in final_summary.lower():
        final_summary += f" Use during pregnancy: {item['Use in pregnancy'].strip()}."

    return final_summary



# Find item by drug name
def get_item(drug_name):
    row = df[df["Drug Name"].str.lower() == drug_name.lower()]
    if row.empty:
        return None, f"No information found for '{drug_name}'."
    return row.iloc[0], None

# Generate markdown bullet list with proper emojis and bold titles
def generate_bullet_list(item):
    html = f"""
    <ul style="list-style: none; padding-left: 0;">
        <li><strong>💊 Medicine Name:</strong> {item['Drug Name']}</li>
        <li><strong>🏢 Company Name:</strong> {item['Company Name']}</li>
        <li><strong>🧪 Active Ingredient:</strong> {item['Active Ingredient']}</li>
        <li><strong>🎯 Indication:</strong> {item['Indication']}</li>
        <li><strong>💉 Dosage and Administration:</strong> {item['Dosage and Administration']}</li>
        <li><strong>⚠️ Side Effects:</strong> {item['Side Effects']}</li>
        <li><strong>🤰 Use in pregnancy:</strong> {item['Use in pregnancy']}</li>
    </ul>
    """
    return html

# UI Title and description
st.title("💊 Medical Info Chatbot")
st.markdown("Get a quick summary and structured details of medicine(s). Enter one or more names separated by commas.")

# User input
user_input = st.text_input("Medicine Name(s)", "")

if user_input:
    drug_names = [name.strip() for name in user_input.split(",") if name.strip()]
    found = False

    for drug in drug_names:
        item, error = get_item(drug)
        if error:
            st.warning(error)
            continue

        # Unique session keys
        key = f"summary_{drug.lower()}"
        prev_key = f"prev_summary_{drug.lower()}"

        # Initialize previous summary if not already set
        if prev_key not in st.session_state:
            st.session_state[prev_key] = ""

        # Regenerate button
        if st.button(f"🔁 Regenerate summary for {item['Drug Name']}", key=f"regen_{key}"):
            prompt = build_prompt(item)
            summary = generate_summary_until_different(item, st.session_state[prev_key])
            st.session_state[key] = summary
            st.session_state[prev_key] = summary

        # Generate summary only if not already done
        if key not in st.session_state:
            prompt = build_prompt(item)
            summary = generate_summary_text(prompt)
            st.session_state[key] = summary
            st.session_state[prev_key] = summary  # store as previous too

        # Final values
        summary = st.session_state[key]
        bullet_list = generate_bullet_list(item)

        # Warn for short/incomplete output
        if len(summary.split()) < 30:
            st.warning(f"⚠️ Summary for '{item['Drug Name']}' may be incomplete. Try regenerating or editing input.")

        # Display styled summary box
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

# Custom CSS for page styling
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
