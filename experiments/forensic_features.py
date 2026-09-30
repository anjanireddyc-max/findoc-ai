"""Quick check: do classical forensic features separate tampered from genuine receipts?
Scores every validation and test document and prints the document-level AUC of each feature."""
import io
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = 64


def ela(img, q=90):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); buf.seek(0)
    re = Image.open(buf).convert("RGB")
    return np.abs(np.asarray(img, np.float32) - np.asarray(re, np.float32)).mean(axis=2)


def noise(grey):
    g = Image.fromarray(grey.astype(np.uint8))
    return np.abs(grey - np.asarray(g.filter(ImageFilter.MedianFilter(3)), np.float32))


def window_stats(m, ink):
    h, w = m.shape
    vals = []
    for y in range(0, h - W + 1, W // 2):
        for x in range(0, w - W + 1, W // 2):
            k = ink[y:y + W, x:x + W]
            if k.mean() < 0.02:
                continue
            vals.append(m[y:y + W, x:x + W][k].mean())
    return np.array(vals) if vals else np.array([0.0])


def features(path):
    img = Image.open(path).convert("RGB")
    grey = np.asarray(img.convert("L"), np.float32)
    ink = grey < 160
    f = {}
    for name, m in [("ela90", ela(img, 90)), ("ela75", ela(img, 75)), ("noise", noise(grey))]:
        v = window_stats(m, ink)
        med = np.median(v) + 1e-6
        f[name + "_maxratio"] = float(v.max() / med)
        f[name + "_p99ratio"] = float(np.percentile(v, 99) / med)
        f[name + "_zmax"] = float((v.max() - med) / (v.std() + 1e-6))
        f[name + "_minratio"] = float(med / (v.min() + 1e-6))
    return f


for split in sys.argv[1:] or ["val", "test"]:
    df = pd.read_csv(os.path.join(BASE, "annotations", f"{split}.csv"))
    rows, labels = [], []
    for r in df.itertuples():
        p = os.path.join(BASE, "dataset", split, r.image)
        if os.path.exists(p):
            rows.append(features(p)); labels.append(int(r.forged))
    X = pd.DataFrame(rows)
    print(f"== {split}: {len(labels)} docs")
    for c in X.columns:
        print(f"  {c:20s} AUC {roc_auc_score(labels, X[c]):.3f}")
