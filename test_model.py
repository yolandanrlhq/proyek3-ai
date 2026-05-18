import cv2
import numpy as np
import joblib

model = joblib.load("skin_tone_model.pkl")

img_path = "dataset/fairface/fairface/train/1.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (64, 64))

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

features = [[
    np.mean(img[:, :, 0]),
    np.mean(img[:, :, 1]),
    np.mean(img[:, :, 2]),
    np.mean(lab[:, :, 0]),
    np.mean(lab[:, :, 1]),
    np.mean(lab[:, :, 2]),
    np.mean(hsv[:, :, 0]),
    np.mean(hsv[:, :, 1]),
    np.mean(hsv[:, :, 2]),
]]

prediction = model.predict(features)[0]

print("Prediksi warna kulit:", prediction)