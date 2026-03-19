import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(
    TRAIN_CSV,
    engine="python",
    on_bad_lines="skip"
)

df.columns = df.columns.str.lower()

print("CSV Loaded")
print("Columns:", df.columns.tolist())
print("Total rows:", len(df))


TEXT_COL = "description"
LABEL_COL = "prescription"

df = df[[TEXT_COL, LABEL_COL]].dropna()

le = LabelEncoder()
y = le.fit_transform(df[LABEL_COL])

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df[TEXT_COL]).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(256, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

model.save(os.path.join(MODEL_DIR, "prescription_model.keras"))

import pickle
'''pickle.dump(vectorizer, open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb"))
pickle.dump(le, open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb"))'''
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
pickle.dump(label_encoder, open("model/label_encoder.pkl", "wb"))

print("✅ Model training completed and saved")

