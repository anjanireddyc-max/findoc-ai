import os
import tensorflow as tf


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnetv2b0_patch_tampering.keras"
)


print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nMODEL LAYERS")
print("=" * 60)

for i, layer in enumerate(model.layers):

    print(
        i,
        layer.name,
        layer.__class__.__name__,
        layer.output_shape
        if hasattr(layer, "output_shape")
        else ""
    )