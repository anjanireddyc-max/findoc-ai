import os
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0
from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PATCH_DIR = os.path.join(
    BASE_DIR,
    "patches"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "efficientnetv2b0_patch_tampering.keras"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20


# =========================================================
# DATASET DIRECTORIES
# =========================================================

TRAIN_DIR = os.path.join(
    PATCH_DIR,
    "train"
)

VAL_DIR = os.path.join(
    PATCH_DIR,
    "val"
)

TEST_DIR = os.path.join(
    PATCH_DIR,
    "test"
)


# =========================================================
# LOAD PATCH DATASETS
# =========================================================

print("\n==============================")
print("LOADING PATCH DATASETS")
print("==============================")


train_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    labels="inferred",

    label_mode="binary",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True,

    seed=42
)


val_dataset = tf.keras.utils.image_dataset_from_directory(

    VAL_DIR,

    labels="inferred",

    label_mode="binary",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    labels="inferred",

    label_mode="binary",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


print("\nClass names:")

print(
    train_dataset.class_names
)


# =========================================================
# PERFORMANCE
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomRotation(
        0.02
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomTranslation(
        height_factor=0.02,
        width_factor=0.02
    ),

    layers.RandomContrast(
        0.10
    )

], name="data_augmentation")


# =========================================================
# EFFICIENTNETV2B0
# =========================================================

print("\n==============================")
print("LOADING EFFICIENTNETV2B0")
print("==============================")


base_model = EfficientNetV2B0(

    include_top=False,

    weights="imagenet",

    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


# Fine-tune only the last part initially
base_model.trainable = True


for layer in base_model.layers[:-30]:

    layer.trainable = False


# =========================================================
# BUILD MODEL
# =========================================================

inputs = layers.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


x = data_augmentation(
    inputs
)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(
    x
)


x = layers.BatchNormalization()(
    x
)


x = layers.Dropout(
    0.4
)(
    x
)


x = layers.Dense(
    128,
    activation="relu"
)(
    x
)


x = layers.Dropout(
    0.3
)(
    x
)


outputs = layers.Dense(
    1,
    activation="sigmoid"
)(
    x
)


model = Model(
    inputs,
    outputs
)


# =========================================================
# COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[

        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


# =========================================================
# CLASS WEIGHTS
# =========================================================

# Our patch counts are approximately:
#
# Genuine  = 1449
# Tampered = 273
#
# Therefore tampered patches need more weight.

genuine_count = 1449
tampered_count = 273

total = (
    genuine_count +
    tampered_count
)

class_weights = {

    0: total / (
        2 * genuine_count
    ),

    1: total / (
        2 * tampered_count
    )
}


print("\n==============================")
print("CLASS WEIGHTS")
print("==============================")

print(
    class_weights
)


# =========================================================
# CALLBACKS
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_auc",

        mode="max",

        save_best_only=True,

        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=5,

        restore_best_weights=True,

        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-7,

        verbose=1
    )
]


# =========================================================
# MODEL SUMMARY
# =========================================================

print("\n==============================")
print("MODEL SUMMARY")
print("==============================")


model.summary()


# =========================================================
# TRAIN
# =========================================================

print("\n==============================")
print("STARTING PATCH TRAINING")
print("==============================")


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# =========================================================
# TEST
# =========================================================

print("\n==============================")
print("TESTING PATCH MODEL")
print("==============================")


results = model.evaluate(

    test_dataset,

    return_dict=True
)


print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")


for name, value in results.items():

    print(
        f"{name}: {value:.4f}"
    )


# =========================================================
# SAVE
# =========================================================

model.save(
    MODEL_PATH
)


print("\n==============================")
print("PATCH TRAINING COMPLETE")
print("==============================")


print(
    f"Model saved at:\n{MODEL_PATH}"
)