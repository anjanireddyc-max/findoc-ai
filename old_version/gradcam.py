import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

TARGET_LAYER = "efficientnetv2-b0"


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded.")


# =========================================================
# GET EFFICIENTNET MODEL
# =========================================================

efficientnet = model.get_layer(
    TARGET_LAYER
)

print(
    "EfficientNetV2B0 found."
)


# =========================================================
# GET LAST CONVOLUTIONAL OUTPUT
# =========================================================

# EfficientNetV2B0 outputs a 7x7x1280 feature map.
# This is the spatial feature map used by Grad-CAM.

print(
    "EfficientNet output shape:",
    efficientnet.output.shape
)


# =========================================================
# GRAD-CAM
# =========================================================

def generate_gradcam(
    image_path
):

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    original = Image.open(
        image_path
    ).convert("RGB")

    resized = original.resize(
        IMAGE_SIZE
    )

    image = np.array(
        resized,
        dtype=np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )
    image = tf.convert_to_tensor(
    image,
    dtype=tf.float32
)

    # -----------------------------------------------------
    # Forward pass through complete model
    # -----------------------------------------------------

    with tf.GradientTape(
        persistent=True
    ) as tape:

        # Watch the input
        tape.watch(image)

        # Pass through augmentation
        augmented = model.layers[1](
            image,
            training=False
        )

        # Pass through EfficientNet
        conv_outputs = efficientnet(
            augmented,
            training=False
        )

        # Continue through classifier
        x = model.get_layer(
            "global_average_pooling2d"
        )(conv_outputs)

        x = model.get_layer(
            "batch_normalization"
        )(x)

        x = model.get_layer(
            "dropout"
        )(x, training=False)

        x = model.get_layer(
            "dense"
        )(x)

        x = model.get_layer(
            "dropout_1"
        )(x, training=False)

        predictions = model.get_layer(
            "dense_1"
        )(x)


        # Probability of tampered class
        probability = predictions[:, 0]


    # -----------------------------------------------------
    # Calculate gradients
    # -----------------------------------------------------

    gradients = tape.gradient(
        probability,
        conv_outputs
    )


    # -----------------------------------------------------
    # Global average pooling
    # -----------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )


    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]


    # -----------------------------------------------------
    # Weight feature maps
    # -----------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs *
        pooled_gradients,
        axis=-1
    )


    # -----------------------------------------------------
    # ReLU
    # -----------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )


    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + 1e-8
    )

    heatmap = heatmap.numpy()


    # -----------------------------------------------------
    # Resize heatmap
    # -----------------------------------------------------

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        original.size
    )

    heatmap = np.array(
        heatmap_image
    ) / 255.0


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    probability_value = float(
        probability.numpy()[0]
    )

    if probability_value >= 0.5:

        prediction = "Tampered"

    else:

        prediction = "Genuine"


    return (
        original,
        heatmap,
        prediction,
        probability_value
    )


# =========================================================
# DISPLAY GRAD-CAM
# =========================================================

def show_gradcam(
    image_path
):

    (
        original,
        heatmap,
        prediction,
        probability
    ) = generate_gradcam(
        image_path
    )


    original_array = (
        np.array(original)
        / 255.0
    )


    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.imshow(
        original_array
    )

    plt.imshow(
        heatmap,
        alpha=0.45,
        cmap="jet"
    )

    plt.axis(
        "off"
    )

    plt.title(
        f"Prediction: {prediction} "
        f"({probability:.2%})"
    )


    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_dir = os.path.join(
        BASE_DIR,
        "gradcam_results"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    filename = os.path.basename(
        image_path
    )

    output_path = os.path.join(
        output_dir,
        f"gradcam_{filename}"
    )


    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=200
    )

    plt.show()


    print("\n==============================")
    print("GRAD-CAM RESULT")
    print("==============================")

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Tampering probability: "
        f"{probability:.2%}"
    )

    print(
        f"Saved to:\n{output_path}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    test_folder = os.path.join(
        BASE_DIR,
        "patches",
        "test",
        "tampered"
    )


    files = [

        f for f in os.listdir(
            test_folder
        )

        if f.lower().endswith(
            ".png"
        )
    ]


    if not files:

        raise FileNotFoundError(
            "No tampered patches found."
        )


    image_path = os.path.join(
        test_folder,
        files[0]
    )


    print(
        f"Testing image:\n{image_path}"
    )


    show_gradcam(
        image_path
    )