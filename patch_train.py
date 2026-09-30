import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ELA = os.environ.get("ELA", "0") == "1"
PATCH_DIR = os.path.join(BASE_DIR, "patches_ela" if ELA else "patches")
MODEL_DIR = os.path.join(BASE_DIR, "models")
# FORENSIC=1 adds a fixed noise-residual filter in front of EfficientNetV2B0
FORENSIC = os.environ.get("FORENSIC", "0") == "1"
VERSION = "v4_ela" if ELA else ("v3" if FORENSIC else "v2")
MODEL_PATH = os.path.join(MODEL_DIR, f"efficientnetv2b0_patch_tampering_{VERSION}.keras")
HISTORY_PATH = os.path.join(MODEL_DIR, f"training_history_{VERSION}.json")

# Patches are already 224 x 224 crops at the document's original resolution,
# so they are fed to the network without any resizing.
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 4
FINE_TUNE_EPOCHS = 16
FINE_TUNE_LAYERS = 60


# =========================================================
# LOAD PATCH DATASETS
# =========================================================

def load(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(PATCH_DIR, split),
        labels="inferred",
        label_mode="binary",
        class_names=["genuine", "tampered"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42,
    )


print("\n==============================")
print("LOADING PATCH DATASETS")
print("==============================")

train_dataset = load("train", True)
val_dataset = load("val", False)
test_dataset = load("test", False)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)


# =========================================================
# DATA AUGMENTATION
# =========================================================

# No zoom and no rotation: both resample the image and blur the fine pixel
# traces left by editing. Only photometric changes and flips are used.
data_augmentation = tf.keras.Sequential(
    [layers.RandomFlip("horizontal")] if ELA else
    [layers.RandomFlip("horizontal"), layers.RandomContrast(0.10), layers.RandomBrightness(0.08)],
    name="data_augmentation")


# =========================================================
# BUILD MODEL (same architecture as before)
# =========================================================

base_model = EfficientNetV2B0(include_top=False, weights="imagenet",
                              input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base_model.trainable = False

def forensic_filter():
    """Fixed 5x5 filter producing three channels for EfficientNetV2B0:
    0 - grey-scale image (content),
    1 - grey minus its 3x3 mean (fine noise residual, amplified),
    2 - grey minus its 5x5 mean (coarser residual, amplified).
    Editing tools change these residuals even when the text looks identical."""
    kernel = np.zeros((5, 5, 3, 3), dtype=np.float32)
    grey = np.zeros((5, 5), dtype=np.float32); grey[2, 2] = 1.0
    mean3 = np.zeros((5, 5), dtype=np.float32); mean3[1:4, 1:4] = 1.0 / 9
    mean5 = np.full((5, 5), 1.0 / 25, dtype=np.float32)
    for c in range(3):
        kernel[:, :, c, 0] = grey / 3
        kernel[:, :, c, 1] = 4.0 * (grey - mean3) / 3
        kernel[:, :, c, 2] = 4.0 * (grey - mean5) / 3
    layer = layers.Conv2D(3, 5, padding="same", trainable=False, name="forensic_filter")
    layer.build((None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    layer.set_weights([kernel, np.array([0.0, 127.5, 127.5], dtype=np.float32)])
    return layer


inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = data_augmentation(inputs)
if FORENSIC:
    x = forensic_filter()(x)
    x = layers.ReLU(max_value=255.0, name="forensic_clip")(x)   # keep the 0-255 pixel range
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = Model(inputs, outputs)


def compile_model(learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


# =========================================================
# CLASS WEIGHTS (counted from the patch folders)
# =========================================================

genuine_count = len(os.listdir(os.path.join(PATCH_DIR, "train", "genuine")))
tampered_count = len(os.listdir(os.path.join(PATCH_DIR, "train", "tampered")))
total = genuine_count + tampered_count
class_weights = {0: total / (2 * genuine_count), 1: total / (2 * tampered_count)}

print("\n==============================")
print("CLASS WEIGHTS")
print("==============================")
print(f"genuine={genuine_count} tampered={tampered_count} -> {class_weights}")


# =========================================================
# CALLBACKS
# =========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


def callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                             min_lr=1e-7, verbose=1),
    ]


# =========================================================
# TRAIN - phase 1: classification head only
# =========================================================

print("\n==============================")
print("PHASE 1: TRAINING THE CLASSIFICATION HEAD")
print("==============================")

compile_model(1e-3)
history_1 = model.fit(train_dataset, validation_data=val_dataset, epochs=HEAD_EPOCHS,
                      class_weight=class_weights, callbacks=callbacks())


# =========================================================
# TRAIN - phase 2: fine-tune the top of EfficientNetV2B0
# =========================================================

print("\n==============================")
print("PHASE 2: FINE-TUNING EFFICIENTNETV2B0")
print("==============================")

base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

compile_model(1e-5)
history_2 = model.fit(train_dataset, validation_data=val_dataset, epochs=FINE_TUNE_EPOCHS,
                      class_weight=class_weights, callbacks=callbacks())


# =========================================================
# TEST (patch level)
# =========================================================

print("\n==============================")
print("PATCH-LEVEL TEST RESULTS")
print("==============================")

results = model.evaluate(test_dataset, return_dict=True)
for name, value in results.items():
    print(f"{name}: {value:.4f}")

history = {k: [float(v) for v in history_1.history[k] + history_2.history.get(k, [])]
           for k in history_1.history}
history["phase_1_epochs"] = len(history_1.history["loss"])
history["patch_test"] = {k: float(v) for k, v in results.items()}
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=2)

model.save(MODEL_PATH)
print(f"\nModel saved at:\n{MODEL_PATH}")
print("Run  python evaluate.py  for document-level results.")
