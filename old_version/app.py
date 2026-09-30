import os
import uuid
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR, "static", "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR, "static", "results"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnetv2b0_patch_tampering.keras"
)

IMAGE_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

TILE_SIZE = 256

TAMPERING_THRESHOLD = 0.50


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("Loading EfficientNetV2B0 model...")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# Get EfficientNet backbone
efficientnet = model.get_layer("efficientnetv2-b0")

print("EfficientNetV2B0 found.")
print("EfficientNet output shape:", efficientnet.output_shape)


# =========================================================
# FILE VALIDATION
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# CREATE DOCUMENT TILES
# =========================================================

def create_tiles(image):

    width, height = image.size

    tiles = []
    positions = []

    # Small image
    if width <= TILE_SIZE and height <= TILE_SIZE:

        tiles.append(image.copy())

        positions.append(
            (0, 0, width, height)
        )

        return tiles, positions


    x_positions = list(
        range(0, width, TILE_SIZE)
    )

    y_positions = list(
        range(0, height, TILE_SIZE)
    )


    for y in y_positions:

        for x in x_positions:

            x2 = min(
                x + TILE_SIZE,
                width
            )

            y2 = min(
                y + TILE_SIZE,
                height
            )

            tile = image.crop(
                (x, y, x2, y2)
            )

            if (
                tile.width < 32
                or tile.height < 32
            ):
                continue

            tiles.append(tile)

            positions.append(
                (x, y, x2, y2)
            )


    return tiles, positions


# =========================================================
# PREPARE IMAGE
# =========================================================

def prepare_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = np.expand_dims(
        array,
        axis=0
    )

    return tf.convert_to_tensor(
        array,
        dtype=tf.float32
    )


# =========================================================
# PREDICT TILE
# =========================================================

def predict_tile(tile):

    image = prepare_image(tile)

    prediction = model.predict(
        image,
        verbose=0
    )

    return float(
        prediction[0][0]
    )


# =========================================================
# GRAD-CAM
# =========================================================

def generate_gradcam(tile):

    image = prepare_image(tile)

    with tf.GradientTape() as tape:

        # Augmentation layer
        augmented = model.layers[1](
            image,
            training=False
        )

        # EfficientNet feature maps
        conv_outputs = efficientnet(
            augmented,
            training=False
        )

        # Classification head
        x = model.get_layer(
            "global_average_pooling2d"
        )(conv_outputs)

        x = model.get_layer(
            "batch_normalization"
        )(x)

        x = model.get_layer(
            "dropout"
        )(
            x,
            training=False
        )

        x = model.get_layer(
            "dense"
        )(x)

        x = model.get_layer(
            "dropout_1"
        )(
            x,
            training=False
        )

        predictions = model.get_layer(
            "dense_1"
        )(x)

        probability = predictions[:, 0]

    gradients = tape.gradient(
        probability,
        conv_outputs
    )

    if gradients is None:

        raise RuntimeError(
            "Grad-CAM gradients could not be generated."
        )


    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )


    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]


    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )


    heatmap = tf.maximum(
        heatmap,
        0
    )


    max_value = tf.reduce_max(
        heatmap
    )


    heatmap = heatmap / (
        max_value + 1e-8
    )


    heatmap = heatmap.numpy()


    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )


    heatmap_image = heatmap_image.resize(
        tile.size
    )


    heatmap = (
        np.asarray(
            heatmap_image,
            dtype=np.float32
        ) / 255.0
    )


    return heatmap


# =========================================================
# CREATE GRAD-CAM IMAGE
# =========================================================

def create_gradcam_image(tile, heatmap):

    tile_array = (
        np.asarray(tile)
        / 255.0
    )


    fig = plt.figure(
        figsize=(8, 8)
    )


    plt.imshow(
        tile_array
    )


    plt.imshow(
        heatmap,
        alpha=0.45,
        cmap="jet"
    )


    plt.axis("off")


    filename = (
        "gradcam_"
        + uuid.uuid4().hex
        + ".png"
    )


    output_path = os.path.join(
        RESULT_FOLDER,
        filename
    )


    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=180
    )


    plt.close(fig)


    return filename


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# DETECT
# =========================================================

@app.route("/detect")
def detect():

    return render_template(
        "detect.html"
    )


# =========================================================
# HOW IT WORKS
# =========================================================

@app.route("/how-it-works")
def how_it_works():

    return render_template(
        "how_it_works.html"
    )


# =========================================================
# PERFORMANCE
# =========================================================

@app.route("/performance")
def performance():

    return render_template(
        "performance.html"
    )


# =========================================================
# ABOUT
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# PREDICT
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "document" not in request.files:

        return render_template(
            "detect.html",
            error="Please upload or capture a document."
        )


    file = request.files["document"]


    if file.filename == "":

        return render_template(
            "detect.html",
            error="Please select or capture an image."
        )


    if not allowed_file(file.filename):

        return render_template(
            "detect.html",
            error="Only PNG, JPG and JPEG images are supported."
        )


    # -----------------------------------------------------
    # SAVE FILE
    # -----------------------------------------------------

    original_name = secure_filename(
        file.filename
    )


    unique_name = (
        uuid.uuid4().hex
        + "_"
        + original_name
    )


    upload_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )


    file.save(upload_path)


    # -----------------------------------------------------
    # OPEN IMAGE
    # -----------------------------------------------------

    try:

        document = Image.open(
            upload_path
        ).convert("RGB")

    except Exception:

        return render_template(
            "detect.html",
            error="The uploaded image could not be read."
        )


    # -----------------------------------------------------
    # CREATE TILES
    # -----------------------------------------------------

    tiles, positions = create_tiles(
        document
    )


    if len(tiles) == 0:

        return render_template(
            "detect.html",
            error="No usable document regions were found."
        )


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    predictions = []


    for tile in tiles:

        probability = predict_tile(
            tile
        )

        predictions.append(
            probability
        )


    predictions = np.asarray(
        predictions
    )


    # -----------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------

    highest_index = int(
        np.argmax(predictions)
    )


    highest_probability = float(
        predictions[highest_index]
    )


    average_probability = float(
        np.mean(predictions)
    )


    flagged_regions = int(
        np.sum(
            predictions >= TAMPERING_THRESHOLD
        )
    )


    # -----------------------------------------------------
    # FINAL RESULT
    # -----------------------------------------------------

    if highest_probability >= TAMPERING_THRESHOLD:

        result = "Tampered"

        confidence = highest_probability

    else:

        result = "Genuine"

        confidence = 1.0 - highest_probability


    # -----------------------------------------------------
    # GRAD-CAM
    # -----------------------------------------------------

    selected_tile = tiles[
        highest_index
    ]


    try:

        heatmap = generate_gradcam(
            selected_tile
        )

        gradcam_filename = create_gradcam_image(
            selected_tile,
            heatmap
        )

        gradcam_url = (
            "/static/results/"
            + gradcam_filename
        )

    except Exception as e:

        print(
            "Grad-CAM error:",
            e
        )

        gradcam_url = None


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    return render_template(
        "result.html",

        result=result,

        confidence=round(
            confidence * 100,
            2
        ),

        highest_score=round(
            highest_probability * 100,
            2
        ),

        average_score=round(
            average_probability * 100,
            2
        ),

        regions=len(tiles),

        flagged_regions=flagged_regions,

        original_image=(
            "/static/uploads/"
            + unique_name
        ),

        gradcam_image=gradcam_url
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )