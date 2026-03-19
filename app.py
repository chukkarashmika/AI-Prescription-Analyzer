import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import warnings
import difflib

warnings.filterwarnings("ignore")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Prescription Analyzer",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🩺 AI Prescription Analyzer")

st.sidebar.info(
"""
Upload a prescription image and the AI model
will predict the medicine and show details.
"""
)

st.sidebar.markdown("### Supported Formats")
st.sidebar.write("• JPG\n• JPEG\n• PNG")

st.sidebar.markdown("---")
st.sidebar.caption("AI Based Medical Assistant")

# -----------------------------
# Main Title
# -----------------------------
st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>🩺 AI Prescription Analyzer</h1>",
    unsafe_allow_html=True
)

st.markdown("Upload a **prescription image** and get medicine details.")
st.markdown("---")

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "model/prescription_model.keras"
CLASSES_PATH = "model/classes.npy"
CSV_PATH = "medicine_database.csv"

# -----------------------------
# Load Model
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Load Classes
# -----------------------------
if not os.path.exists(CLASSES_PATH):
    st.error("❌ Classes file not found")
    st.stop()

classes = np.load(CLASSES_PATH, allow_pickle=True)

# -----------------------------
# Load Medicine CSV
# -----------------------------
if not os.path.exists(CSV_PATH):
    st.error("❌ medicine_database.csv not found")
    st.stop()

medicine_df = pd.read_csv(CSV_PATH)
medicine_df["medicine"] = medicine_df["medicine"].str.lower()

# Convert CSV → Dictionary
PRESCRIPTION_DB = {}

for _, row in medicine_df.iterrows():
    PRESCRIPTION_DB[row["medicine"]] = {
        "disease": row["disease"],
        "description": row["description"],
        "side_effects": row["side_effects"],
        "dosage": row["dosage"]
    }

# -----------------------------
# Upload Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader(
        "📤 Upload Prescription Image",
        type=["jpg", "jpeg", "png"]
    )

with col2:
    st.markdown("### 📌 Instructions")
    st.write("1️⃣ Upload prescription image")
    st.write("2️⃣ Click **Predict Medicine**")
    st.write("3️⃣ View details")

st.markdown("---")

# -----------------------------
# Image Processing
# -----------------------------
if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    with col2:
        if st.button("🔍 Predict Medicine"):

            pred = model.predict(img)
            idx = int(np.argmax(pred))

            medicine = str(classes[idx])
            confidence = float(pred[0][idx]) * 100

            st.success(f"💊 {medicine}")
            st.info(f"📊 Confidence: {confidence:.2f}%")

            # -----------------------------
            # 🔥 SMART MATCHING LOGIC
            # -----------------------------
            details = None
            medicine_clean = medicine.lower().replace("-", " ").strip()

            best_match = None
            highest_score = 0

            for key in PRESCRIPTION_DB:
                key_clean = key.lower().replace("-", " ").strip()

                score = difflib.SequenceMatcher(None, medicine_clean, key_clean).ratio()

                if score > highest_score:
                    highest_score = score
                    best_match = key

            # Threshold
            if highest_score > 0.5:
                details = PRESCRIPTION_DB[best_match]

            st.markdown("---")

            if details:
                st.markdown("### 🩺 Disease")
                st.write(details["disease"])

                st.markdown("### 📄 Description")
                st.write(details["description"])

                st.markdown("### ⚠ Side Effects")
                st.write(details["side_effects"])

                st.markdown("### 💊 Dosage")
                st.write(details["dosage"])
            else:
                st.warning("⚠ No details found in database")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("### 🤖 Features")
st.write("✔ AI-based medicine prediction")
st.write("✔ CSV-based medicine database (80+ medicines)")
st.write("✔ Smart matching system")

st.warning("⚠ For educational purposes only. Consult a doctor.")