from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import joblib
import os

from recommendation import recommended_colors_map

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "skin_tone_model.pkl"
model = joblib.load(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.get("/")
def home():
    return {"message": "Glow Match API aktif"}

def extract_features(face):
    face = cv2.resize(face, (64, 64))

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

    features = [[
        np.mean(face[:, :, 0]),
        np.mean(face[:, :, 1]),
        np.mean(face[:, :, 2]),
        np.mean(lab[:, :, 0]),
        np.mean(lab[:, :, 1]),
        np.mean(lab[:, :, 2]),
        np.mean(hsv[:, :, 0]),
        np.mean(hsv[:, :, 1]),
        np.mean(hsv[:, :, 2]),
    ]]

    return features

@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Gambar tidak valid"}

        img = cv2.resize(img, (400, 400))
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_full, 1.3, 5)

        if len(faces) == 0:
            return {
                "error": "Wajah tidak terdeteksi. Pastikan wajah terlihat jelas, tidak terlalu gelap, dan menghadap kamera."
            }

        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]

        features = extract_features(face)

        skin_tone = model.predict(features)[0]

        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray_face))

        lab_face = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        lab_l = float(np.mean(lab_face[:, :, 0]))

        return {
            "skin_tone": skin_tone,
            "brightness": brightness,
            "lab_l": lab_l,
            "recommended_colors": recommended_colors_map[skin_tone],
        }

    except Exception as e:
        return {"error": str(e)}