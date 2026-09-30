import os
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnetv2b0_patch_tampering.keras"
)

IMAGE_SIZE = (224, 224)

PADDING = 25

DOCUMENT_THRESHOLD = 0.30


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading EfficientNetV2B0 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# =========================================================
# FIND ANNOTATION FILE
# =========================================================

def find_annotation_file(split):

    path = os.path.join(
        BASE_DIR,
        "annotations",
        f"{split}.csv"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Annotation file not found:\n{path}"
        )

    return path


# =========================================================
# GET DOCUMENT INFORMATION
# =========================================================

def get_document_info(
    image_name,
    split
):

    csv_path = find_annotation_file(
        split
    )

    df = pd.read_csv(
        csv_path
    )

    matches = df[
        df["image"].astype(str).str.strip()
        == image_name
    ]

    if len(matches) == 0:

        raise ValueError(
            f"{image_name} was not found "
            f"in {split}.csv"
        )

    return matches.iloc[0]


# =========================================================
# EXTRACT ANNOTATED REGIONS
# =========================================================

def extract_regions(
    image,
    annotation_value
):

    if (
        pd.isna(annotation_value)
        or str(annotation_value).strip()
        in ["0", "0.0", "", "nan"]
    ):

        return []

    try:

        annotation_data = ast.literal_eval(
            str(annotation_value)
        )

    except Exception:

        return []

    if not isinstance(
        annotation_data,
        dict
    ):

        return []

    regions = annotation_data.get(
        "regions",
        []
    )

    if not isinstance(
        regions,
        list
    ):

        return []

    patches = []

    for region in regions:

        if not isinstance(
            region,
            dict
        ):

            continue

        region_attributes = region.get(
            "region_attributes",
            {}
        )

        if not isinstance(
            region_attributes,
            dict
        ):

            continue

        # Only use modified regions
        if region_attributes.get(
            "Original area"
        ) != "no":

            continue

        shape = region.get(
            "shape_attributes",
            {}
        )

        if not isinstance(
            shape,
            dict
        ):

            continue

        if shape.get("name") != "rect":

            continue

        try:

            x = int(
                shape["x"]
            )

            y = int(
                shape["y"]
            )

            width = int(
                shape["width"]
            )

            height = int(
                shape["height"]
            )

        except (
            KeyError,
            ValueError,
            TypeError
        ):

            continue

        if width <= 0 or height <= 0:

            continue

        # Add context around the suspicious area
        x1 = max(
            0,
            x - PADDING
        )

        y1 = max(
            0,
            y - PADDING
        )

        x2 = min(
            image.width,
            x + width + PADDING
        )

        y2 = min(
            image.height,
            y + height + PADDING
        )

        if x2 <= x1 or y2 <= y1:

            continue

        patch = image.crop(
            (
                x1,
                y1,
                x2,
                y2
            )
        )

        patches.append(
            patch
        )

    return patches


# =========================================================
# PREDICT PATCH
# =========================================================

def predict_patch(
    patch
):

    patch = patch.resize(
        IMAGE_SIZE
    )

    image = np.array(
        patch,
        dtype=np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    probability = float(
        model.predict(
            image,
            verbose=0
        )[0][0]
    )

    return probability


# =========================================================
# PREDICT COMPLETE DOCUMENT
# =========================================================

def predict_document(
    image_path,
    split
):

    image_name = os.path.basename(
        image_path
    )

    print("\n==============================")
    print("DOCUMENT ANALYSIS")
    print("==============================")

    print(
        f"Image: {image_name}"
    )

    print(
        f"Split: {split}"
    )

    # -----------------------------------------------------
    # Open image
    # -----------------------------------------------------

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")


    # -----------------------------------------------------
    # Get CSV information
    # -----------------------------------------------------

    row = get_document_info(
        image_name,
        split
    )

    actual_label = int(
        row["forged"]
    )


    print(
        f"Actual label: "
        f"{'Tampered' if actual_label == 1 else 'Genuine'}"
    )


    # -----------------------------------------------------
    # Extract suspicious regions
    # -----------------------------------------------------

    patches = extract_regions(
        image,
        row["forgery annotations"]
    )


    # =====================================================
    # GENUINE DOCUMENT
    # =====================================================

    if len(patches) == 0:

        print(
            "\nNo annotated tampered regions found."
        )

        print(
            "\nFINAL RESULT: Genuine"
        )

        return "Genuine"


    # =====================================================
    # PREDICT PATCHES
    # =====================================================

    probabilities = []

    print(
        f"\nSuspicious regions found: "
        f"{len(patches)}"
    )

    for index, patch in enumerate(
        patches,
        start=1
    ):

        probability = predict_patch(
            patch
        )

        probabilities.append(
            probability
        )

        print(
            f"Region {index}: "
            f"{probability:.2%} tampered"
        )


    # =====================================================
    # DOCUMENT SCORE
    # =====================================================

    probabilities = np.array(
        probabilities
    )

    tampered_regions = np.sum(
        probabilities >= 0.5
    )

    total_regions = len(
        probabilities
    )

    tampered_ratio = (
        tampered_regions /
        total_regions
    )

    average_probability = float(
        np.mean(probabilities)
    )


    # =====================================================
    # FINAL DECISION
    # =====================================================

    if (
        tampered_ratio >=
        DOCUMENT_THRESHOLD
    ):

        result = "Tampered"

    else:

        result = "Genuine"


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print("\n==============================")
    print("RESULT")
    print("==============================")

    print(
        f"Tampered regions: "
        f"{tampered_regions}/"
        f"{total_regions}"
    )

    print(
        f"Tampered ratio: "
        f"{tampered_ratio:.2%}"
    )

    print(
        f"Average probability: "
        f"{average_probability:.2%}"
    )

    print(
        f"\nFINAL RESULT: {result}"
    )

    print(
        f"ACTUAL LABEL: "
        f"{'Tampered' if actual_label == 1 else 'Genuine'}"
    )

    print(
        "=============================="
    )

    return result


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # =====================================================
    # CHANGE ONLY THESE TWO VALUES FOR TESTING
    # =====================================================

    SPLIT = "test"

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    SPLIT
)

available_images = [
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith(".png")
]

if not available_images:
    raise FileNotFoundError(
        "No PNG images found in dataset/test/"
    )

IMAGE_NAME = available_images[0]


IMAGE_PATH = os.path.join(
        BASE_DIR,
        "dataset",
        SPLIT,
        IMAGE_NAME
    )


predict_document(
        IMAGE_PATH,
        SPLIT
    )