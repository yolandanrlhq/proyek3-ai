import os
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VAL_CSV = os.path.join(
    BASE_DIR,
    "dataset",
    "fitz_undersampled_test_final.csv"
)

VAL_IMG_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "fairface",
    "fairface",
    "val"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "glowmatch_mobilenetv2.keras"
)

IMG_SIZE = 160

CLASS_NAMES = [
    "putih",
    "kuning_langsat",
    "sawo_matang",
    "gelap"
]

# =========================
# LABEL MAPPING
# =========================
def map_phototype_to_skin_tone(phototype):
    phototype = str(phototype).strip()

    if phototype in ["I & II"]:
        return "putih"
    elif phototype == "III":
        return "kuning_langsat"
    elif phototype in ["IV", "V"]:
        return "sawo_matang"
    elif phototype == "VI":
        return "gelap"

    return None


df = pd.read_csv(VAL_CSV)

paths = []
labels = []

for _, row in df.iterrows():

    skin_tone = map_phototype_to_skin_tone(
        row["phototype"]
    )

    if skin_tone is None:
        continue

    image_path = os.path.join(
        VAL_IMG_DIR,
        str(row["file"]).strip()
    )

    if not os.path.exists(image_path):
        continue

    paths.append(image_path)
    labels.append(
        CLASS_NAMES.index(skin_tone)
    )

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(
    MODEL_PATH
)

# =========================
# PREDICT
# =========================
y_true = []
y_pred = []

for path, label in zip(paths, labels):

    img = tf.keras.utils.load_img(
        path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img = tf.keras.utils.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    pred = model.predict(
        img[None, ...],
        verbose=0
    )

    pred_class = pred.argmax()

    y_true.append(label)
    y_pred.append(pred_class)

# =========================
# REPORT
# =========================
print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )
)