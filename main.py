from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.get("/")
def home():
    return {"message": "Glow Match API aktif"}

def classify_skin_tone(brightness):
    if brightness > 175:
        return "putih"
    elif brightness > 140:
        return "kuning_langsat"
    elif brightness > 105:
        return "sawo_matang"
    return "gelap"

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

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = img[y:y+h, x:x+w]
        else:
            face = img

        face = cv2.resize(face, (240, 240))

        # Area sampel:
        # dahi, pipi kiri, pipi kanan
        forehead = face[35:75, 90:150]
        left_cheek = face[105:155, 45:95]
        right_cheek = face[105:155, 145:195]

        sample_regions = [forehead, left_cheek, right_cheek]

        brightness_values = []
        lab_l_values = []

        for region in sample_regions:
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            brightness_values.append(np.mean(gray_region))

            lab_region = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            l_channel = lab_region[:, :, 0]
            lab_l_values.append(np.mean(l_channel))

        brightness = float(np.mean(brightness_values))
        lab_l = float(np.mean(lab_l_values))

        skin_tone = classify_skin_tone(brightness)

        recommended_colors_map = {
            "putih": ["soft pink", "peach", "nude", "baby blue"],
            "kuning_langsat": ["dusty pink", "olive", "cream", "terracotta"],
            "sawo_matang": ["mocha", "maroon", "mustard", "army green"],
            "gelap": ["emerald", "navy", "burgundy", "gold"],
        }

        return {
            "skin_tone": skin_tone,
            "brightness": brightness,
            "lab_l": lab_l,
            "recommended_colors": recommended_colors_map[skin_tone],
        }

    except Exception as e:
        return {"error": str(e)}