import os
import pandas as pd
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

CSV_PATH = "dataset/fitz_undersampled_train_final.csv"
IMAGE_FOLDER = "dataset/fairface/fairface/train"

# ambil data
df = pd.read_csv(CSV_PATH)

def map_skin_tone(phototype):
    if phototype == "I & II":
        return "putih"
    elif phototype == "III":
        return "kuning_langsat"
    elif phototype in ["IV", "V"]:
        return "sawo_matang"
    elif phototype == "VI":
        return "gelap"
    return None

df["skin_tone"] = df["phototype"].apply(map_skin_tone)
df = df.dropna()

# BATASIN DULU BIAR GA LAMA
df = df.sample(min(1000, len(df)), random_state=42)

X = []
y = []

print("Total baris CSV yang dipakai:", len(df))

for i, (_, row) in enumerate(df.iterrows(), start=1):
    img_path = os.path.join(IMAGE_FOLDER, row["file"])

    if not os.path.exists(img_path):
        print("File tidak ada:", img_path)
        continue

    img = cv2.imread(img_path)

    if img is None:
        print("Gagal baca:", img_path)
        continue

    img = cv2.resize(img, (64, 64))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    features = [
        np.mean(img[:, :, 0]),
        np.mean(img[:, :, 1]),
        np.mean(img[:, :, 2]),
        np.mean(lab[:, :, 0]),
        np.mean(lab[:, :, 1]),
        np.mean(lab[:, :, 2]),
        np.mean(hsv[:, :, 0]),
        np.mean(hsv[:, :, 1]),
        np.mean(hsv[:, :, 2]),
    ]

    X.append(features)
    y.append(row["skin_tone"])

    if i % 100 == 0:
        print(f"Sudah proses {i}/{len(df)} gambar")

X = np.array(X)
y = np.array(y)

print("Total data valid:", len(X))

if len(X) == 0:
    print("ERROR: Tidak ada gambar yang berhasil dibaca. Cek IMAGE_FOLDER atau isi kolom file di CSV.")
    exit()

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

joblib.dump(model, "skin_tone_model.pkl")

print("Model berhasil disimpan!")