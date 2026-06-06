from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from recommendation import recommended_colors_map

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = os.path.join(
    "models",
    "glowmatch_mobilenetv2.keras"
)

model = load_model(MODEL_PATH)

CLASS_NAMES = [
    "putih",
    "kuning_langsat",
    "sawo_matang",
    "gelap"
]

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =========================
# HOME
# =========================
@app.get("/")
def home():
    return {
        "message": "Glow Match API aktif"
    }

# =========================
# ANALYZE FACE
# =========================
@app.post("/analyze-face")
async def analyze_face(
    file: UploadFile = File(...)
):
    try:

        contents = await file.read()

        np_arr = np.frombuffer(
            contents,
            np.uint8
        )

        img = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if img is None:
            return {
                "error": "Gambar tidak valid"
            }

        # resize agar deteksi lebih stabil
        img = cv2.resize(
            img,
            (400, 400)
        )

        gray_full = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray_full,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) == 0:
            return {
                "error": "Wajah tidak terdeteksi. Pastikan wajah terlihat jelas, menghadap kamera, dan pencahayaan cukup."
            }

        # ambil wajah pertama
        x, y, w, h = faces[0]

        face_original = img[
            y:y+h,
            x:x+w
        ]

        # =========================
        # PREPROCESS UNTUK MODEL
        # =========================
        face = cv2.resize(
            face_original,
            (160, 160)
        )

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        face = preprocess_input(
            face.astype(np.float32)
        )

        face = np.expand_dims(
            face,
            axis=0
        )

        # =========================
        # PREDIKSI
        # =========================
        prediction = model.predict(
            face,
            verbose=0
        )

        index = np.argmax(
            prediction
        )

        skin_tone = CLASS_NAMES[index]

        confidence = float(
            prediction[0][index]
        )

        # =========================
        # BRIGHTNESS
        # =========================
        gray_face = cv2.cvtColor(
            face_original,
            cv2.COLOR_BGR2GRAY
        )

        brightness = float(
            np.mean(gray_face)
        )

        # =========================
        # LAB VALUE
        # =========================
        lab_face = cv2.cvtColor(
            face_original,
            cv2.COLOR_BGR2LAB
        )

        lab_l = float(
            np.mean(
                lab_face[:, :, 0]
            )
        )

        # =========================
        # RESPONSE
        # =========================
        return {
            "skin_tone": skin_tone,
            "confidence": round(
                confidence * 100,
                2
            ),
            "brightness": round(
                brightness,
                2
            ),
            "lab_l": round(
                lab_l,
                2
            ),
            "recommended_colors":
                recommended_colors_map[
                    skin_tone
                ]
        }

    except Exception as e:
        return {
            "error": str(e)
        }