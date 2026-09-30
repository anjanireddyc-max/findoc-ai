import os
import tensorflow as tf


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnetv2b0_patch_tampering_v2.keras"
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
        getattr(layer, "output", None).shape
        if getattr(layer, "output", None) is not None
        else ""
    )