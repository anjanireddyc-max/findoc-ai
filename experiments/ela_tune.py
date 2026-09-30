"""Tunes the Error Level Analysis detector (JPEG quality, window size) on the validation
documents and reports the chosen setting on the test documents, plus localisation."""
import ast
import io
import itertools
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUALITIES = [50, 60, 70, 75, 80]
WINDOWS = [32, 48, 64, 96]


def ela(img, q):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); buf.seek(0)
    return np.abs(np.asarray(img, np.float32) - np.asarray(Image.open(buf).convert("RGB"), np.float32)).mean(axis=2)


def box_sum(a, w, step):
    c = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = np.arange(0, a.shape[0] - w + 1, step); xs = np.arange(0, a.shape[1] - w + 1, step)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    return c[Y + w, X + w] - c[Y, X + w] - c[Y + w, X] + c[Y, X], Y, X


def doc_scores(img, grey):
    ink = (grey < 160).astype(np.float32)
    out = {}
    for q in QUALITIES:
        e = ela(img, q)
        for w in WINDOWS:
            s_ink, Y, X = box_sum(ink, w, w // 2)
            s_e, _, _ = box_sum(e * ink, w, w // 2)
            keep = s_ink > 0.02 * w * w
            if keep.sum() < 3:
                out[(q, w)] = (0.0, None); continue
            v = s_e[keep] / s_ink[keep]
            med, sd = np.median(v), v.std() + 1e-6
            i = int(np.argmax(v))
            out[(q, w)] = (float((v[i] - med) / sd), (int(X[keep][i]), int(Y[keep][i]), w))
    return out


def load(split):
    df = pd.read_csv(os.path.join(BASE, "annotations", f"{split}.csv"))
    res, labels, boxes = [], [], []
    for r in df.itertuples():
        p = os.path.join(BASE, "dataset", split, r.image)
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")
        res.append(doc_scores(img, np.asarray(img.convert("L"), np.float32)))
        labels.append(int(r.forged))
        b = []
        if int(r.forged) == 1:
            try:
                for g in ast.literal_eval(str(r._5))["regions"]:
                    if g["region_attributes"].get("Original area") == "no":
                        s = g["shape_attributes"]; b.append((s["x"], s["y"], s["x"] + s["width"], s["y"] + s["height"]))
            except Exception:
                pass
        boxes.append(b)
    return res, np.array(labels), boxes


val = load("val"); test = load("test")
best = None
for k in itertools.product(QUALITIES, WINDOWS):
    a = roc_auc_score(val[1], [d[k][0] for d in val[0]])
    print(f"q={k[0]} w={k[1]}  val AUC {a:.3f}")
    if best is None or a > best[1]:
        best = (k, a)
k = best[0]
ts = np.array([d[k][0] for d in test[0]])
print(f"\nBEST q={k[0]} w={k[1]}: val AUC {best[1]:.3f}  test AUC {roc_auc_score(test[1], ts):.3f}")
hits = total = 0
for d, lab, bx in zip(test[0], test[1], test[2]):
    if lab == 1 and bx and d[k][1]:
        x, y, w = d[k][1]; total += 1
        hits += any(not (b[2] <= x or b[0] >= x + w or b[3] <= y or b[1] >= y + w) for b in bx)
print(f"localisation: most suspicious window overlaps a real edit in {hits}/{total} tampered test receipts")
np.save(os.path.join(BASE, "experiments", "ela_best.npy"), np.array([k[0], k[1]]))
