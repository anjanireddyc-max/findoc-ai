import os
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0
from sklearn.utils.class_weight import compute_class_weight

from config import (
    BASE_DIR,
    DATASET_DIR,
    MODEL_DIR,
    MODEL_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS
)


# =========================================================
# 1. FIND ANNOTATION FILE
# =========================================================

def find_annotation(split):

    annotations_dir = os.path.join(BASE_DIR, "annotations")

    possible_files = [
        os.path.join(annotations_dir, f"{split}.csv"),
        os.path.join(annotations_dir, f"{split}.txt"),
        os.path.join(annotations_dir, f"{split}(1).csv"),
        os.path.join(annotations_dir, f"{split}(1).txt")
    ]

    for file_path in possible_files:
        if os.path.exists(file_path):
            return file_path

    raise FileNotFoundError(
        f"Could not find annotation file for {split}"
    )


# =========================================================
# 2. LOAD ANNOTATIONS
# =========================================================

def load_annotations(split):

    file_path = find_annotation(split)

    print(f"\nLoading {split} annotations:")
    print(file_path)

    df = pd.read_csv(file_path)

    print("Columns:", list(df.columns))
    print("Samples:", len(df))

    if "image" not in df.columns:
        raise ValueError("image column not found")

    if "forged" not in df.columns:
        raise ValueError("forged column not found")

    df = df[["image", "forged"]].copy()

    df["forged"] = pd.to_numeric(
        df["forged"],
        errors="coerce"
    )

    df = df.dropna(subset=["forged"])

    df["forged"] = df["forged"].astype(int)

    df = df[df["forged"].isin([0, 1])]

    return df


# =========================================================
# 3. CREATE IMAGE PATHS
# =========================================================

def prepare_dataframe(df, split):

    image_folder = os.path.join(
        DATASET_DIR,
        split
    )

    df["filepath"] = df["image"].apply(
        lambda x: os.path.join(
            image_folder,
            x
        )
    )

    df = df[
        df["filepath"].apply(os.path.exists)
    ]

    print(
        f"{split}: {len(df)} images found"
    )

    return df


# =========================================================
# 4. LOAD DATA
# =========================================================

train_df = prepare_dataframe(
    load_annotations("train"),
    "train"
)

val_df = prepare_dataframe(
    load_annotations("val"),
    "val"
)

test_df = prepare_dataframe(
    load_annotations("test"),
    "test"
)


print("\n==============================")
print("DATASET SUMMARY")
print("==============================")

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain labels:")
print(train_df["forged"].value_counts())

print("\nValidation labels:")
print(val_df["forged"].value_counts())

print("\nTest labels:")
print(test_df["forged"].value_counts())


# =========================================================
# 5. IMAGE LOADING
# =========================================================

def load_image(filepath, label):

    image = tf.io.read_file(filepath)

    image = tf.image.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# =========================================================
# 6. CREATE TF DATASET
# =========================================================

def create_dataset(df, shuffle=False):

    paths = df["filepath"].values

    labels = df["forged"].values.astype(
        "float32"
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(
            len(df),
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_dataset = create_dataset(
    train_df,
    shuffle=True
)

val_dataset = create_dataset(
    val_df
)

test_dataset = create_dataset(
    test_df
)


# =========================================================
# 7. DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomRotation(0.015),

    layers.RandomZoom(0.08),

    layers.RandomTranslation(
        height_factor=0.02,
        width_factor=0.02
    ),

    layers.RandomContrast(0.10)

], name="data_augmentation")


# =========================================================
# 8. EFFICIENTNETV2B0
# =========================================================

base_model = EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)

# Fine-tune the last 30 layers
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False


# =========================================================
# 9. BUILD MODEL
# =========================================================

inputs = layers.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(
    inputs,
    outputs
)


# =========================================================
# 10. COMPILE MODEL
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


print("\n==============================")
print("MODEL")
print("==============================")

model.summary()


# =========================================================
# 11. CLASS WEIGHTS
# =========================================================

classes = train_df["forged"].unique()

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["forged"]
)

class_weights = {
    int(c): float(w)
    for c, w in zip(classes, weights)
}

print("\nClass weights:")
print(class_weights)


# =========================================================
# 12. CALLBACKS
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
# 13. TRAIN
# =========================================================

print("\n==============================")
print("STARTING TRAINING")
print("==============================")

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# =========================================================
# 14. TEST
# =========================================================

print("\n==============================")
print("TESTING")
print("==============================")

results = model.evaluate(
    test_dataset,
    return_dict=True
)

print("\nTest Results:")

for name, value in results.items():

    print(
        f"{name}: {value:.4f}"
    )


# =========================================================
# 15. SAVE MODEL
# =========================================================

model.save(
    MODEL_PATH
)

print("\n==============================")
print("COMPLETE")
print("==============================")

print(
    "Model saved at:"
)

print(MODEL_PATH)