# 🩺 AI Prescription Analyzer

## 📌 Project Overview
AI Prescription Analyzer is a web application that uses Deep Learning to predict medicines from prescription images.  
After predicting the medicine, it displays additional details like disease, dosage, side effects, and description using a CSV-based database.

---

## 🚀 Features
- 📤 Upload prescription image (JPG, JPEG, PNG)
- 🤖 AI-based medicine prediction
- 💊 Displays:
  - Medicine Name
  - Disease
  - Description
  - Side Effects
  - Dosage
- 📊 Confidence score
- 🖥 User-friendly interface using Streamlit

---

## 🛠 Tech Stack
- Python
- Streamlit (for UI)
- TensorFlow / Keras (for model)
- NumPy
- Pandas
- Pillow (image processing)

---

## 📂 Project Structure

---

## 📊 Dataset Details

### 🔹 1. Training Dataset
- Used to train the AI model
- Contains prescription images and labels
- Already trained (no need to modify)

### 🔹 2. Medicine Database (CSV)
File: `medicine_database.csv`

Contains:
- Medicine name
- Disease
- Description
- Side effects
- Dosage

👉 This is used to display details after prediction

---

## ⚙️ Installation (Requirements)

Make sure Python is installed, then run:

```bash
python -m pip install streamlit tensorflow pandas numpy pillow
