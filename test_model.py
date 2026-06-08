import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "models/glowmatch_mobilenetv2.keras"

CLASS_NAMES = [
    "putih",
    "kuning_langsat",
    "sawo_matang",
    "gelap"
]

model = load_model(MODEL_PATH)

img_path = "images/14.jpg" 

img = cv2.imread(img_path)

if img is None:
    raise ValueError("Gambar tidak ditemukan")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (160, 160))
img = preprocess_input(img.astype(np.float32))

img = np.expand_dims(img, axis=0)

prediction = model.predict(img, verbose=0)[0]

predicted_class = np.argmax(prediction)
confidence = prediction[predicted_class] * 100

print("Prediksi :", CLASS_NAMES[predicted_class])
print("Confidence :", f"{confidence:.2f}%")

print("\nSemua Probabilitas:")
for cls, prob in zip(CLASS_NAMES, prediction):
    print(f"{cls}: {prob*100:.2f}%")