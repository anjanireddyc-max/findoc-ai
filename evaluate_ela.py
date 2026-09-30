"""Document-level evaluation of the ELA detector.

Chooses the z-score threshold with the best F1 on the validation receipts, measures the
test receipts, and saves models/ela_threshold.json, models/ela_metrics.json and figures
in reports/ (ela_confusion_matrix.png, ela_roc_curve.png).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from forensics import BASE_DIR, THRESHOLD_PATH, ELADetector

REPORT_DIR = os.path.join(BASE_DIR, "reports")
METRICS_PATH = os.path.join(BASE_DIR, "models", "ela_metrics.json")
os.makedirs(REPORT_DIR, exist_ok=True)
detector = ELADetector()


def scores(split):
    df = pd.read_csv(os.path.join(BASE_DIR, "annotations", f"{split}.csv"))
    s, y = [], []
    for r in df.itertuples():
        p = os.path.join(BASE_DIR, "dataset", split, r.image)
        if os.path.exists(p):
            s.append(detector.score(Image.open(p))[1]); y.append(int(r.forged))
    return np.array(s), np.array(y)


def metrics(y, s, t):
    p = (s >= t).astype(int)
    return {"accuracy": accuracy_score(y, p), "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0),
            "auc": roc_auc_score(y, s), "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist()}


vs, vy = scores("val")
cands = np.unique(np.round(vs, 3))
t = float(max(cands, key=lambda c: (f1_score(vy, (vs >= c).astype(int), zero_division=0), c)))
with open(THRESHOLD_PATH, "w") as f:
    json.dump({"threshold": t, "quality": 70, "window": 64}, f, indent=2)
ts, ty = scores("test")
res = {"threshold": t, "validation": metrics(vy, vs, t), "test": metrics(ty, ts, t),
       "test_documents": int(len(ty)), "test_tampered": int(ty.sum())}
with open(METRICS_PATH, "w") as f:
    json.dump(res, f, indent=2)

m = res["test"]
print(f"threshold z >= {t:.3f} (validation F1 {res['validation']['f1']:.3f}, AUC {res['validation']['auc']:.3f})")
for k in ["accuracy", "precision", "recall", "f1", "auc"]:
    print(f"{k:>9}: {m[k] * 100:.2f}%")
print("confusion", m["confusion_matrix"])

cm = np.array(m["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.6, 4), dpi=200)
ax.imshow(cm, cmap="Oranges")
for (r, c), v in np.ndenumerate(cm):
    ax.text(c, r, str(v), ha="center", va="center", fontsize=16, color="white" if v > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["Genuine", "Tampered"]); ax.set_yticks([0, 1], ["Genuine", "Tampered"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("ELA Detector - Confusion Matrix")
fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "ela_confusion_matrix.png")); plt.close(fig)
fpr, tpr, _ = roc_curve(ty, ts)
fig, ax = plt.subplots(figsize=(4.6, 4), dpi=200)
ax.plot(fpr, tpr, color="#c8553d", lw=2, label=f"AUC = {m['auc']:.3f}")
ax.plot([0, 1], [0, 1], "--", color="#999999")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ELA Detector - ROC Curve")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "ela_roc_curve.png")); plt.close(fig)
