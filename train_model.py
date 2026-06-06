import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_CSV = os.path.join(BASE_DIR, "dataset", "fitz_undersampled_train_final.csv")
VAL_CSV = os.path.join(BASE_DIR, "dataset", "fitz_undersampled_test_final.csv")

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "dataset", "fairface", "fairface", "train")
VAL_IMG_DIR = os.path.join(BASE_DIR, "dataset", "fairface", "fairface", "val")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "glowmatch_mobilenetv2.keras")

IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 20

CLASS_NAMES = ["putih", "kuning_langsat", "sawo_matang", "gelap"]

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
    else:
        return None


def load_dataset(csv_path, image_dir):
    df = pd.read_csv(csv_path)    

    if len(df) > 1000:
        df = df.sample(n=1000, random_state=42)

    images = []
    labels = []

    for _, row in df.iterrows():
        filename = str(row["file"]).strip()
        phototype = row["phototype"]

        skin_tone = map_phototype_to_skin_tone(phototype)
        if skin_tone is None:
            continue

        image_path = os.path.join(image_dir, filename)

        if not os.path.exists(image_path):
            print(f"File tidak ditemukan: {image_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Gagal membaca gambar: {image_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(img.astype(np.float32))

        label_index = CLASS_NAMES.index(skin_tone)

        images.append(img)
        labels.append(label_index)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


# =========================
# LOAD DATA
# =========================
print("Loading train data...")
print("Jumlah data train:", len(pd.read_csv(TRAIN_CSV)))

X_train, y_train = load_dataset(TRAIN_CSV, TRAIN_IMG_DIR)

print("Loading validation data...")
print("Jumlah data val:", len(pd.read_csv(VAL_CSV)))
X_val, y_val = load_dataset(VAL_CSV, VAL_IMG_DIR)

print("Train data:", X_train.shape, y_train.shape)
print("Validation data:", X_val.shape, y_val.shape)

unique, counts = np.unique(y_train, return_counts=True)

print("Distribusi kelas:")
for u, c in zip(unique, counts):
    print(CLASS_NAMES[u], c)

if len(X_train) == 0:
    raise ValueError("Data training kosong. Cek path CSV dan folder gambar.")

if len(X_val) == 0:
    raise ValueError("Data validasi kosong. Cek path CSV dan folder gambar.")

# =========================
# CLASS WEIGHT
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# =========================
# DATA AUGMENTATION
# =========================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# =========================
# MODEL
# =========================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.35)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.25)(x)
outputs = Dense(len(CLASS_NAMES), activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS
# =========================
os.makedirs(MODEL_DIR, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# =========================
# TRAIN HEAD
# =========================
print("Training classifier head...")

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks
)

# =========================
# FINE TUNING
# =========================
print("Fine tuning MobileNetV2...")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save(MODEL_PATH)

print("Model berhasil disimpan di:")
print(MODEL_PATH)