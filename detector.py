"""Document-level tampering detection shared by app.py and evaluate.py.

A document is scanned with overlapping WINDOW x WINDOW windows at its original
resolution - exactly how the training patches were made - and every window is
scored by the EfficientNetV2B0 patch model. The document score is the score of
its most suspicious window. Grad-CAM is computed for the most suspicious windows
and pasted back onto the full page.
"""
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientnetv2b0_patch_tampering_v2.keras")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "document_threshold.json")

WINDOW = 224
STRIDE = 112
BATCH = 32
GRADCAM_WINDOWS = 3
MAX_SIDE = 2400          # very large photos are scaled down so analysis stays fast
MIN_PIXEL_STD = 12.0     # blank windows (empty paper) are not scored
# Training receipts are almost colourless (99 % of their windows have a mean HSV
# saturation below 11). Strongly coloured windows - logos, colour graphics - are
# outside what the model has learned, so they are not scored.
MAX_WINDOW_SATURATION = 20.0
OUT_OF_DOMAIN_FRACTION = 0.05


class TamperDetector:
    def __init__(self, model_path=None):
        # settings chosen by evaluate.py: model file, decision threshold, score aggregation
        saved = {}
        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH) as f:
                saved = json.load(f)
        if model_path is None:
            model_path = os.path.join(BASE_DIR, "models", saved["model"]) if "model" in saved else MODEL_PATH
        self.model = tf.keras.models.load_model(model_path)
        same_model = os.path.basename(model_path) == saved.get("model", os.path.basename(MODEL_PATH))
        self.threshold = float(saved.get("threshold", 0.5)) if same_model else 0.5
        self.aggregation = saved.get("aggregation", "max") if same_model else "max"
        self.backbone = self.model.get_layer("efficientnetv2-b0")
        names = [layer.name for layer in self.model.layers]
        split = names.index("efficientnetv2-b0")
        # layers before the backbone (augmentation, forensic filter) and after it (classifier)
        self.pre = [l for l in self.model.layers[1:split] if l.name != "data_augmentation"]
        self.head = self.model.layers[split + 1:]

    # -----------------------------------------------------
    # windows
    # -----------------------------------------------------
    @staticmethod
    def prepare(image):
        image = image.convert("RGB")
        scale = min(1.0, MAX_SIDE / max(image.size))
        if scale < 1.0:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        if image.width < WINDOW or image.height < WINDOW:
            canvas = Image.new("RGB", (max(image.width, WINDOW), max(image.height, WINDOW)), (255, 255, 255))
            canvas.paste(image, (0, 0))
            image = canvas
        return image

    @staticmethod
    def positions(length):
        starts = list(range(0, length - WINDOW + 1, STRIDE))
        if starts[-1] != length - WINDOW:
            starts.append(length - WINDOW)
        return starts

    @staticmethod
    def model_input(image):
        """The training receipts are black-and-white scans, so every document is
        analysed in grey-scale (3 identical channels) to match them."""
        grey = np.asarray(image.convert("L"), dtype=np.float32)
        return np.repeat(grey[:, :, None], 3, axis=2)

    def windows(self, image):
        array = self.model_input(image)
        grey = array[:, :, 0]
        saturation = np.asarray(image.convert("HSV"), dtype=np.float32)[:, :, 1]
        boxes, crops = [], []
        content = colour = 0
        for y in self.positions(image.height):
            for x in self.positions(image.width):
                if grey[y:y + WINDOW, x:x + WINDOW].std() < MIN_PIXEL_STD:
                    continue                                 # blank paper
                content += 1
                if saturation[y:y + WINDOW, x:x + WINDOW].mean() > MAX_WINDOW_SATURATION:
                    colour += 1                              # coloured graphic, never seen in training
                    continue
                boxes.append((x, y, x + WINDOW, y + WINDOW))
                crops.append(array[y:y + WINDOW, x:x + WINDOW])
        if not boxes:                                        # blank or fully graphic image
            boxes.append((0, 0, WINDOW, WINDOW))
            crops.append(array[:WINDOW, :WINDOW])
        self.last_colour_fraction = colour / max(content, 1)
        return boxes, np.stack(crops)

    def score(self, image):
        """Returns the prepared image, window boxes and window probabilities."""
        image = self.prepare(image)
        boxes, crops = self.windows(image)
        probs = self.model.predict(crops, batch_size=BATCH, verbose=0)[:, 0]
        return image, boxes, probs

    @staticmethod
    def aggregate(probs, method):
        """Combines the window scores of one document into a document score."""
        ranked = np.sort(probs)[::-1]
        if method == "max":
            return float(ranked[0])
        if method.startswith("top"):
            return float(ranked[:int(method[3:])].mean())
        return float(ranked.mean())

    # -----------------------------------------------------
    # Grad-CAM
    # -----------------------------------------------------
    def gradcam(self, crop):
        batch = tf.convert_to_tensor(crop[None, ...], dtype=tf.float32)
        for layer in self.pre:
            batch = layer(batch, training=False)
        features = self.backbone(batch, training=False)       # 7 x 7 x 1280
        with tf.GradientTape() as tape:
            tape.watch(features)
            x = features
            for layer in self.head:
                x = layer(x, training=False)
            score = x[:, 0]
        grads = tape.gradient(score, features)
        weights = tf.reduce_mean(grads, axis=(1, 2))[0]
        cam = tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1)).numpy()
        cam = cam / (cam.max() + 1e-8)
        return np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((WINDOW, WINDOW), Image.BICUBIC),
                          dtype=np.float32) / 255.0

    # -----------------------------------------------------
    # full analysis used by the web application
    # -----------------------------------------------------
    def analyze(self, image):
        image, boxes, probs = self.score(image)
        best = int(np.argmax(probs))
        doc_prob = self.aggregate(probs, self.aggregation)
        label = "Tampered" if doc_prob >= self.threshold else "Genuine"

        # Grad-CAM for the most suspicious windows, weighted by their score
        array = self.model_input(image)
        cam_map = np.zeros(array.shape[:2], dtype=np.float32)
        order = np.argsort(probs)[::-1][:GRADCAM_WINDOWS]
        for i in order:
            if i != best and probs[i] < probs[best] * 0.8:
                continue
            x1, y1, x2, y2 = boxes[i]
            cam = self.gradcam(array[y1:y2, x1:x2]) * float(probs[i])
            cam_map[y1:y2, x1:x2] = np.maximum(cam_map[y1:y2, x1:x2], cam)
        if cam_map.max() > 0:
            # soften the square window borders so the map reads as one continuous heatmap
            blurred = Image.fromarray(np.uint8(cam_map / cam_map.max() * 255)).filter(ImageFilter.GaussianBlur(12))
            cam_map = np.asarray(blurred, dtype=np.float32)
            cam_map /= cam_map.max() + 1e-8

        return {
            "image": image,
            "label": label,
            "probability": doc_prob,
            "threshold": self.threshold,
            "windows": len(boxes),
            "flagged": int((probs >= self.threshold).sum()),
            "average": float(probs.mean()),
            "boxes": boxes,
            "probs": probs,
            "best_box": boxes[best],
            "cam": cam_map,
            "colour_fraction": self.last_colour_fraction,
            "out_of_domain": self.last_colour_fraction >= OUT_OF_DOMAIN_FRACTION,
        }
