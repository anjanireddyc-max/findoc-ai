"""Exports the trained EfficientNetV2B0-ELA model for the web application:
the backbone to ONNX (models/ela_backbone.onnx) and the head weights to models/ela_head.npz.
Also checks that the exported version gives the same probability as Keras."""
import os
import sys

import numpy as np
import tensorflow as tf
import tf2onnx

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "models", "efficientnetv2b0_patch_tampering_v4_ela.keras")
model = tf.keras.models.load_model(SRC)
backbone = model.get_layer("efficientnetv2-b0")

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
tf2onnx.convert.from_keras(backbone, input_signature=spec, opset=17, output_path=os.path.join(BASE, "models", "ela_backbone.onnx"))

bn = model.get_layer("batch_normalization")
gamma, beta, mean, var = [w.numpy() for w in bn.weights]
w1, b1 = [w.numpy() for w in model.get_layer("dense").weights]
w2, b2 = [w.numpy() for w in model.get_layer("dense_1").weights]
np.savez(os.path.join(BASE, "models", "ela_head.npz"), gamma=gamma, beta=beta, mean=mean, var=var,
         eps=np.float32(bn.epsilon), w1=w1, b1=b1, w2=w2, b2=b2)

# consistency check against Keras on a random ELA-like window
from ela_cnn import ELACNN
x = np.repeat(np.random.RandomState(0).randint(0, 60, (1, 224, 224, 1)), 3, axis=3).astype(np.float32)
keras_p = float(model.predict(x, verbose=0)[0][0])
e = ELACNN()
onnx_p, _ = e.predict_window(x[0, :, :, 0].astype(np.uint8), (0, 0, 224, 224))
print(f"keras {keras_p:.5f}  onnx+numpy {onnx_p:.5f}")
