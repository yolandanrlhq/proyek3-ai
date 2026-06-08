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


def create_dataframe(csv_path, image_dir):
    df = pd.read_csv(csv_path)

    data = []

    for _, row in df.iterrows():
        filename = str(row["file"]).strip()

        skin_tone = map_phototype_to_skin_tone(row["phototype"])

        if skin_tone is None:
            continue

        image_path = os.path.join(image_dir, filename)

        if not os.path.exists(image_path):
            continue

        data.append({
            "image_path": image_path,
            "label": CLASS_NAMES.index(skin_tone)
        })

    return pd.DataFrame(data)

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = preprocess_input(image)

    label = tf.one_hot(label, depth=len(CLASS_NAMES))

    return image, label

# =========================
# LOAD DATA
# =========================
print("Loading train data...")
print("Jumlah data train:", len(pd.read_csv(TRAIN_CSV)))

print("Loading train dataframe...")
train_df = create_dataframe(
    TRAIN_CSV,
    TRAIN_IMG_DIR
)

print("Loading validation dataframe...")
val_df = create_dataframe(
    VAL_CSV,
    VAL_IMG_DIR
)

print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))

print("\nDistribusi train:")
print(train_df["label"].value_counts())

print("\nDistribusi validation:")
print(val_df["label"].value_counts())

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_df["image_path"].values,
        train_df["label"].values
    )
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_df["image_path"].values,
        val_df["label"].values
    )
)

train_dataset = (
    train_dataset
    .shuffle(5000)
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print(train_df["label"].value_counts())

# =========================
# CLASS WEIGHT
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00002), 
    loss="categorical_crossentropy",
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
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,    
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,    
    class_weight=class_weights,
    callbacks=callbacks
)

model.save(MODEL_PATH)

print("Model berhasil disimpan di:")
print(MODEL_PATH)