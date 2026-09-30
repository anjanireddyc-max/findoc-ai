# diagnose.py - checks how the current model behaves on whole documents (the way app.py uses it)
import ast
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientnetv2b0_patch_tampering.keras")
TILE = 256

model = tf.keras.models.load_model(MODEL_PATH)


def predict(images):
    batch = np.stack([np.asarray(i.convert("RGB").resize((224, 224)), dtype=np.float32) for i in images])
    return model.predict(batch, verbose=0)[:, 0]


def tiles(image):
    w, h = image.size
    out = []
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            t = image.crop((x, y, min(x + TILE, w), min(y + TILE, h)))
            if t.width >= 32 and t.height >= 32:
                out.append(t)
    return out


df = pd.read_csv(os.path.join(BASE_DIR, "annotations", "test.csv"))
random.seed(0)
rows = df.to_dict("records")

# 1) document-level result exactly as app.py computes it
scores, labels = [], []
for r in rows:
    img = Image.open(os.path.join(BASE_DIR, "dataset", "test", r["image"])).convert("RGB")
    scores.append(float(predict(tiles(img)).max()))
    labels.append(int(r["forged"]))
scores, labels = np.array(scores), np.array(labels)
pred = scores >= 0.5
tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
fn = int(((pred == 0) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
print(f"APP-STYLE DOCUMENT TEST: {len(rows)} docs ({labels.sum()} tampered)")
print(f"  accuracy {(tp + tn) / len(rows):.3f}  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
auc = tf.keras.metrics.AUC(); auc.update_state(labels, scores); print(f"  document AUC {auc.result().numpy():.3f}")

# 2) the scale shortcut: small crops from GENUINE documents
genuine_docs = [r for r in rows if r["forged"] == 0][:40]
small = []
for r in genuine_docs:
    img = Image.open(os.path.join(BASE_DIR, "dataset", "test", r["image"])).convert("RGB")
    x, y = random.randint(0, img.width - 80), random.randint(0, img.height - 85)
    small.append(img.crop((x, y, x + 80, y + 85)))
p = predict(small)
print(f"SMALL 80x85 CROPS FROM GENUINE DOCS: mean P(tampered) {p.mean():.3f}, flagged {int((p >= 0.5).sum())}/{len(p)}")

# 3) 256x256 windows centred on REAL tampered regions
big = []
for r in [r for r in rows if r["forged"] == 1]:
    img = Image.open(os.path.join(BASE_DIR, "dataset", "test", r["image"])).convert("RGB")
    try:
        regions = ast.literal_eval(str(r["forgery annotations"]))["regions"]
    except Exception:
        continue
    for reg in regions:
        if reg.get("region_attributes", {}).get("Original area") != "no":
            continue
        s = reg["shape_attributes"]
        cx, cy = s["x"] + s["width"] // 2, s["y"] + s["height"] // 2
        x1, y1 = max(0, min(cx - 128, img.width - 256)), max(0, min(cy - 128, img.height - 256))
        big.append(img.crop((x1, y1, x1 + 256, y1 + 256)))
p = predict(big)
print(f"256x256 WINDOWS CONTAINING REAL EDITS: mean P(tampered) {p.mean():.3f}, flagged {int((p >= 0.5).sum())}/{len(p)}")
