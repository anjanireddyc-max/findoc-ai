"""Chooses the ELA threshold for JPEG uploads on simulated 'online' samples built from the VALIDATION
receipts, then measures it on samples built from the TEST receipts. Saves it in models/ela_threshold.json."""
import json
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score

import forensics
from experiments.simulate_online import edit, jpeg

BASE = forensics.BASE_DIR
det = forensics.ELADetector()


def build(split, n, seed):
    random.seed(seed)
    df = pd.read_csv(os.path.join(BASE, "annotations", f"{split}.csv"))
    z, y = [], []
    for name in df[df.forged == 0].image.tolist()[:n]:
        online = jpeg(Image.open(os.path.join(BASE, "dataset", split, name)).convert("RGB"), random.choice([75, 85, 90, 95]))
        z.append(det.score(online)[1]); y.append(0)
        for kind in ("retype", "copymove"):
            z.append(det.score(jpeg(edit(online, kind), random.choice([80, 90, 95])))[1]); y.append(1)
    return np.array(z), np.array(y)


vz, vy = build("val", 80, 7)
# balanced choice: maximise the average of genuine and tampered accuracy
cands = np.unique(np.round(vz, 3))
bal = lambda t, z, y: (((z < t) & (y == 0)).sum() / (y == 0).sum() + ((z >= t) & (y == 1)).sum() / (y == 1).sum()) / 2
t = float(max(cands, key=lambda c: bal(c, vz, vy)))
tz, ty = build("test", 80, 11)
print(f"JPEG threshold z >= {t:.3f}  val balanced acc {bal(t, vz, vy):.3f}")
print(f"TEST: AUC {roc_auc_score(ty, tz):.3f}  genuine correct {((tz < t) & (ty == 0)).mean() / (ty == 0).mean():.1%}"
      f"  tampered caught {((tz >= t) & (ty == 1)).mean() / (ty == 1).mean():.1%}")
path = os.path.join(BASE, "models", "ela_threshold.json")
cfg = json.load(open(path)); cfg["jpeg_threshold"] = t
cfg["jpeg_test"] = {"genuine_correct": float(((tz < t) & (ty == 0)).sum() / (ty == 0).sum()),
                    "tampered_caught": float(((tz >= t) & (ty == 1)).sum() / (ty == 1).sum()), "auc": float(roc_auc_score(ty, tz))}
json.dump(cfg, open(path, "w"), indent=2)
