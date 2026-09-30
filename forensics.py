"""ELA-based tampering detector (no TensorFlow needed).

The page is re-saved as JPEG (quality 70) and the re-compression error is measured.
Error levels are averaged over the ink (text) pixels of overlapping 64 x 64 windows.
A tampered region - pasted, re-typed or painted over - has a different error level
from the rest of the text, so its window stands out. The document score is how far the
most unusual window lies from the page's typical window, in standard deviations (z-score).

Settings were tuned on the validation receipts (quality 70, window 64: val AUC 0.819,
test AUC 0.822); the decision threshold is chosen on the validation receipts by
evaluate_ela.py and stored in models/ela_threshold.json.
"""
import json
import os

import numpy as np
from PIL import Image, ImageFilter

from ela import ela_map

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "ela_threshold.json")

QUALITY = 70
WINDOW = 64
STEP = 32
INK_LEVEL = 160          # grey value below which a pixel counts as ink / text
MIN_INK = 0.02           # windows with less ink than this are blank paper
MAX_SIDE = 3000          # larger images are processed in their original size up to this limit


def _box_sums(a, w, step):
    c = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = np.arange(0, a.shape[0] - w + 1, step)
    xs = np.arange(0, a.shape[1] - w + 1, step)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    return c[Y + w, X + w] - c[Y, X + w] - c[Y + w, X] + c[Y, X], Y, X


def load_thresholds():
    """Threshold for lossless scans (PNG, PDF pages) and for JPEG uploads, which were already compressed once."""
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            cfg = json.load(f)
        return float(cfg["threshold"]), float(cfg.get("jpeg_threshold", cfg["threshold"]))
    return 3.0, 3.0


class ELADetector:
    def __init__(self):
        self.threshold, self.jpeg_threshold = load_thresholds()

    @staticmethod
    def prepare(image):
        image = image.convert("RGB")
        if max(image.size) > MAX_SIDE:        # only very large photos are reduced
            s = MAX_SIDE / max(image.size)
            image = image.resize((round(image.width * s), round(image.height * s)), Image.LANCZOS)
        if min(image.size) < WINDOW:
            canvas = Image.new("RGB", (max(image.width, WINDOW), max(image.height, WINDOW)), (255, 255, 255))
            canvas.paste(image, (0, 0))
            image = canvas
        return image

    def score(self, image):
        """Returns z-score, window boxes, window z-scores and the raw ELA map."""
        image = self.prepare(image)
        grey = np.asarray(image.convert("L"), np.float32)
        ink = (grey < INK_LEVEL).astype(np.float32)
        e = ela_map(image, QUALITY)
        s_ink, Y, X = _box_sums(ink, WINDOW, STEP)
        s_e, _, _ = _box_sums(e * ink, WINDOW, STEP)
        keep = s_ink > MIN_INK * WINDOW * WINDOW
        if keep.sum() < 3:
            return image, 0.0, [], np.array([]), e
        v = s_e[keep] / s_ink[keep]
        med, sd = float(np.median(v)), float(v.std() + 1e-6)
        z = (v - med) / sd
        boxes = [(int(x), int(y), int(x) + WINDOW, int(y) + WINDOW) for x, y in zip(X[keep], Y[keep])]
        return image, float(z.max()), boxes, z, e

    def analyze(self, image, is_jpeg=False):
        image, z, boxes, zs, e = self.score(image)
        threshold = self.jpeg_threshold if is_jpeg else self.threshold
        label = "Tampered" if z >= threshold else "Genuine"
        order = np.argsort(zs)[::-1] if len(zs) else []
        flagged = [boxes[i] for i in order if zs[i] >= threshold][:8]
        best = boxes[order[0]] if len(boxes) else (0, 0, WINDOW, WINDOW)

        # heatmap: window z-scores spread over the page, then smoothed
        heat = np.zeros(e.shape, np.float32)
        count = np.zeros(e.shape, np.float32)
        for (x1, y1, x2, y2), v in zip(boxes, zs):
            heat[y1:y2, x1:x2] += max(v, 0.0)
            count[y1:y2, x1:x2] += 1
        heat = np.divide(heat, count, out=np.zeros_like(heat), where=count > 0)
        if heat.max() > 0:
            heat = heat / heat.max()
            heat = np.asarray(Image.fromarray(np.uint8(heat * 255)).filter(ImageFilter.GaussianBlur(10)), np.float32) / 255.0
        return {
            "image": image,
            "label": label,
            "z": z,
            "threshold": threshold,
            "windows": len(boxes),
            "flagged_boxes": flagged,
            "flagged": int((zs >= threshold).sum()) if len(zs) else 0,
            "best_box": best,
            "heat": heat,
            "ela": np.clip(e * 12, 0, 255).astype(np.uint8),
        }
