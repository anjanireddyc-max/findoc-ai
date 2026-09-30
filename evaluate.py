"""Document-level evaluation of FinDoc AI.

1. Scores every validation document and picks the decision threshold that gives
   the best F1-score (saved to models/document_threshold.json, used by app.py).
2. Scores every test document with that threshold and reports accuracy,
   precision, recall, F1-score and ROC-AUC (saved to models/document_metrics.json).
3. Saves a confusion matrix, ROC curve and training curves to the reports folder.
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

from detector import BASE_DIR, THRESHOLD_PATH, TamperDetector

REPORT_DIR = os.path.join(BASE_DIR, "reports")
METRICS_PATH = os.path.join(BASE_DIR, "models", "document_metrics.json")
HISTORY_PATH = os.path.join(BASE_DIR, "models", "training_history_v2.json")
os.makedirs(REPORT_DIR, exist_ok=True)

import sys
from detector import MODEL_PATH
detector_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
detector = TamperDetector(detector_path)


AGGREGATIONS = ["max", "top3", "top5", "mean"]


def document_scores(split):
    """Window probabilities of every document (cached, so re-running is fast)."""
    cache = os.path.join(REPORT_DIR, f"window_scores_{split}_{os.path.basename(detector_path)}.npz")
    if os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        return list(data["windows"]), data["labels"]
    df = pd.read_csv(os.path.join(BASE_DIR, "annotations", f"{split}.csv"))
    windows, labels = [], []
    for i, row in enumerate(df.itertuples(), start=1):
        path = os.path.join(BASE_DIR, "dataset", split, row.image)
        if not os.path.exists(path):
            continue
        _, _, probs = detector.score(Image.open(path))
        windows.append(probs)
        labels.append(int(row.forged))
        if i % 50 == 0:
            print(f"  {split}: {i}/{len(df)} documents")
    labels = np.array(labels)
    np.savez(cache, windows=np.array(windows, dtype=object), labels=labels)
    return windows, labels


def metrics(labels, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
        "auc": roc_auc_score(labels, scores),
        "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
    }


# ---------------------------------------------------------
# 1. choose the threshold on the validation documents
# ---------------------------------------------------------
print("Scoring validation documents...")
val_windows, val_labels = document_scores("val")
aucs = {m: roc_auc_score(val_labels, [TamperDetector.aggregate(w, m) for w in val_windows]) for m in AGGREGATIONS}
aggregation = max(aucs, key=aucs.get)
print("Validation AUC per aggregation:", {m: round(a, 3) for m, a in aucs.items()}, "->", aggregation)
val_scores = np.array([TamperDetector.aggregate(w, aggregation) for w in val_windows])
candidates = np.unique(np.round(val_scores, 4))
best = max(candidates, key=lambda t: (f1_score(val_labels, (val_scores >= t).astype(int), zero_division=0), -t))
threshold = float(best)
with open(THRESHOLD_PATH, "w") as f:
    json.dump({"threshold": threshold, "aggregation": aggregation,
               "model": os.path.basename(detector_path)}, f, indent=2)
val_metrics = metrics(val_labels, val_scores, threshold)
print(f"Chosen threshold: {threshold:.4f}  (validation F1 {val_metrics['f1']:.3f})")

# ---------------------------------------------------------
# 2. evaluate on the test documents
# ---------------------------------------------------------
print("Scoring test documents...")
test_windows, test_labels = document_scores("test")
test_scores = np.array([TamperDetector.aggregate(w, aggregation) for w in test_windows])
test_metrics = metrics(test_labels, test_scores, threshold)

result = {"threshold": threshold, "aggregation": aggregation, "model": os.path.basename(detector_path), "validation": val_metrics, "test": test_metrics,
          "test_documents": int(len(test_labels)), "test_tampered": int(test_labels.sum())}
with open(METRICS_PATH, "w") as f:
    json.dump(result, f, indent=2)

print("\n==============================")
print("DOCUMENT-LEVEL TEST RESULTS")
print("==============================")
print(f"Documents: {len(test_labels)} ({int(test_labels.sum())} tampered)")
for k in ["accuracy", "precision", "recall", "f1", "auc"]:
    print(f"{k:>9}: {test_metrics[k] * 100:.2f}%")
(tn, fp), (fn, tp) = test_metrics["confusion_matrix"]
print(f"Confusion matrix: TN {tn}  FP {fp}  FN {fn}  TP {tp}")

# ---------------------------------------------------------
# 3. figures for the report
# ---------------------------------------------------------
cm = np.array(test_metrics["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.6, 4), dpi=200)
ax.imshow(cm, cmap="Blues")
for (r, c), v in np.ndenumerate(cm):
    ax.text(c, r, str(v), ha="center", va="center", fontsize=16,
            color="white" if v > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["Genuine", "Tampered"]); ax.set_yticks([0, 1], ["Genuine", "Tampered"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix (Test Documents)")
fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png")); plt.close(fig)

fpr, tpr, _ = roc_curve(test_labels, test_scores)
fig, ax = plt.subplots(figsize=(4.6, 4), dpi=200)
ax.plot(fpr, tpr, color="#2E3A87", lw=2, label=f"AUC = {test_metrics['auc']:.3f}")
ax.plot([0, 1], [0, 1], "--", color="#999999")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve (Test Documents)")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "roc_curve.png")); plt.close(fig)

if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH) as f:
        h = json.load(f)
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)
    for ax, key, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")]:
        ax.plot(epochs, h[key], color="#2E3A87", lw=2, label="Training")
        ax.plot(epochs, h["val_" + key], color="#EE7623", lw=2, label="Validation")
        ax.axvline(h["phase_1_epochs"] + 0.5, color="#999999", ls="--", lw=1)
        ax.set_xlabel("Epoch"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "training_curves.png")); plt.close(fig)

print(f"\nThreshold saved to {THRESHOLD_PATH}")
print(f"Metrics saved to {METRICS_PATH}")
print(f"Figures saved to {REPORT_DIR}")
