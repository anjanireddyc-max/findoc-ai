"""EfficientNetV2B0 trained on ELA maps, run with ONNX Runtime (no TensorFlow needed).

models/ela_backbone.onnx  - EfficientNetV2B0 backbone: 224x224x3 ELA window -> 7x7x1280 features
models/ela_head.npz       - weights of the classification head (GAP, BatchNorm, Dense 128, Dense 1)

Because the head is small, the prediction and the Grad-CAM gradients are computed exactly
with NumPy from the backbone features, as in Selvaraju et al. (2017).
"""
import os

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKBONE = os.path.join(BASE_DIR, "models", "ela_backbone.onnx")
HEAD = os.path.join(BASE_DIR, "models", "ela_head.npz")
SIZE = 224


class ELACNN:
    def __init__(self):
        import onnxruntime as ort
        if not (os.path.exists(BACKBONE) and os.path.exists(HEAD)):
            raise FileNotFoundError("ELA model files not found")
        self.session = ort.InferenceSession(BACKBONE, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        w = np.load(HEAD)
        self.bn_scale = w["gamma"] / np.sqrt(w["var"] + w["eps"])
        self.bn_shift = w["beta"] - w["mean"] * self.bn_scale
        self.w1, self.b1, self.w2, self.b2 = w["w1"], w["b1"], w["w2"][:, 0], float(w["b2"][0])

    @staticmethod
    def window_around(box, size):
        """224 x 224 window centred on a (smaller) box, kept inside the page."""
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        x1 = int(min(max(0, cx - SIZE // 2), max(0, size[0] - SIZE)))
        y1 = int(min(max(0, cy - SIZE // 2), max(0, size[1] - SIZE)))
        return x1, y1, x1 + SIZE, y1 + SIZE

    def predict_window(self, ela_uint8, box):
        """ela_uint8: amplified ELA map of the whole page (H x W). Returns (probability, Grad-CAM 224x224)."""
        x1, y1, x2, y2 = box
        crop = np.zeros((SIZE, SIZE), np.float32)
        part = ela_uint8[y1:y2, x1:x2].astype(np.float32)
        crop[:part.shape[0], :part.shape[1]] = part
        batch = np.repeat(crop[None, :, :, None], 3, axis=3)
        feats = self.session.run(None, {self.input_name: batch})[0][0]          # 7 x 7 x 1280
        pooled = feats.mean(axis=(0, 1))
        bn = pooled * self.bn_scale + self.bn_shift
        pre = bn @ self.w1 + self.b1
        hidden = np.maximum(pre, 0)
        logit = hidden @ self.w2 + self.b2
        prob = float(1 / (1 + np.exp(-logit)))
        # d logit / d pooled  (the sigmoid factor and 1/49 only rescale the map)
        grad = self.bn_scale * (self.w1 @ (self.w2 * (pre > 0)))
        cam = np.maximum((feats * grad).sum(axis=2), 0)
        cam = cam / (cam.max() + 1e-8)
        cam = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((SIZE, SIZE), Image.BICUBIC), np.float32) / 255
        return prob, cam
