"""Chooses the final document-level decision rule on the VALIDATION receipts and measures it once on
the TEST receipts. Candidates:
  ela      - ELA z-score of the most unusual text region
  cnn_top  - EfficientNetV2B0-ELA probability, highest over the 5 most unusual ELA regions
  combined - average of the two after scaling each to 0..1 by its validation range
Writes models/final_metrics.json and models/final_rule.json, and figures in reports/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from ela_cnn import ELACNN
from forensics import BASE_DIR, ELADetector

REPORT = os.path.join(BASE_DIR, "reports")
ela, cnn = ELADetector(), ELACNN()


def doc_features(split):
    df = pd.read_csv(os.path.join(BASE_DIR, "annotations", f"{split}.csv"))
    F, y = [], []
    for r in df.itertuples():
        p = os.path.join(BASE_DIR, "dataset", split, r.image)
        if not os.path.exists(p):
            continue
        a = ela.analyze(Image.open(p))
        _, _, boxes, zs, _ = ela.score(a["image"])
        order = np.argsort(zs)[::-1][:5] if len(zs) else []
        probs = [cnn.predict_window(a["ela"], cnn.window_around(boxes[i], a["image"].size))[0] for i in order] or [0.0]
        F.append({"ela": a["z"], "cnn_top": float(max(probs))}); y.append(int(r.forged))
    return pd.DataFrame(F), np.array(y)


def metrics(y, s, t):
    p = (s >= t).astype(int)
    return {"accuracy": accuracy_score(y, p), "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0),
            "auc": roc_auc_score(y, s), "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist()}


V, vy = doc_features("val")
T, ty = doc_features("test")
lo = {c: float(V[c].min()) for c in V}; hi = {c: float(V[c].max()) for c in V}
norm = lambda D, c: (D[c] - lo[c]) / (hi[c] - lo[c] + 1e-9)
for D in (V, T):
    D["combined"] = (norm(D, "ela") + norm(D, "cnn_top")) / 2

val_auc = {c: roc_auc_score(vy, V[c]) for c in ["ela", "cnn_top", "combined"]}
rule = max(val_auc, key=val_auc.get)
cands = np.unique(np.round(V[rule], 4))
thr = float(max(cands, key=lambda c: (f1_score(vy, (V[rule] >= c).astype(int), zero_division=0), c)))
print("validation AUC:", {k: round(v, 3) for k, v in val_auc.items()}, "-> rule", rule, "threshold", round(thr, 4))
print("test AUC (for information):", {c: round(roc_auc_score(ty, T[c]), 3) for c in ["ela", "cnn_top", "combined"]})

names = {"ela": "Error Level Analysis", "cnn_top": "EfficientNetV2B0 on ELA maps",
         "combined": "ELA + EfficientNetV2B0 (combined)"}
res = {"method": names[rule], "rule": rule, "threshold": thr,
       "threshold_text": (f"z >= {thr:.2f}" if rule == "ela" else f"score >= {thr:.3f}"),
       "validation": metrics(vy, V[rule].values, thr), "test": metrics(ty, T[rule].values, thr),
       "test_documents": int(len(ty)), "test_tampered": int(ty.sum()),
       "all_test_auc": {c: float(roc_auc_score(ty, T[c])) for c in ["ela", "cnn_top", "combined"]},
       "all_val_auc": {k: float(v) for k, v in val_auc.items()}}
json.dump(res, open(os.path.join(BASE_DIR, "models", "final_metrics.json"), "w"), indent=2)
json.dump({"rule": rule, "threshold": thr, "lo": lo, "hi": hi},
          open(os.path.join(BASE_DIR, "models", "final_rule.json"), "w"), indent=2)
m = res["test"]
for k in ["accuracy", "precision", "recall", "f1", "auc"]:
    print(f"{k:>9}: {m[k] * 100:.2f}%")
print("confusion", m["confusion_matrix"])

cm = np.array(m["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.6, 4), dpi=200)
ax.imshow(cm, cmap="Oranges")
for (r, c), v in np.ndenumerate(cm):
    ax.text(c, r, str(v), ha="center", va="center", fontsize=16, color="white" if v > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["Genuine", "Tampered"]); ax.set_yticks([0, 1], ["Genuine", "Tampered"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix (Test Receipts)")
fig.tight_layout(); fig.savefig(os.path.join(REPORT, "final_confusion_matrix.png")); plt.close(fig)
fig, ax = plt.subplots(figsize=(4.6, 4), dpi=200)
for c, col in [("ela", "#c8553d"), ("cnn_top", "#2f6fd6"), ("combined", "#1f2a24")]:
    f_, t_, _ = roc_curve(ty, T[c]); ax.plot(f_, t_, color=col, lw=2, label=f"{names[c].split(' (')[0]} ({roc_auc_score(ty, T[c]):.3f})")
ax.plot([0, 1], [0, 1], "--", color="#999999")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curves (Test Receipts)")
ax.legend(loc="lower right", fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(REPORT, "final_roc_curve.png")); plt.close(fig)
