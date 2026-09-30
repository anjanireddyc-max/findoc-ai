"""For JPEG uploads: which ELA quality separates 'downloaded genuine JPEG' from 'edited and re-saved JPEG'?
Uses test receipts that are NOT the validation receipts used elsewhere; reports AUC per setting."""
import io
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

import forensics
from experiments.simulate_online import edit, jpeg

BASE = forensics.BASE_DIR
random.seed(7)
df = pd.read_csv(os.path.join(BASE, "annotations", "val.csv"))       # tune on VAL receipts
names = df[df.forged == 0].image.tolist()[:50]
samples = []
for name in names:
    src = Image.open(os.path.join(BASE, "dataset", "val", name)).convert("RGB")
    online = jpeg(src, random.choice([75, 85, 90, 95]))
    samples.append((online, 0))
    for kind in ("retype", "copymove"):
        samples.append((jpeg(edit(online, kind), random.choice([80, 90, 95])), 1))

det = forensics.ELADetector()
y = [s[1] for s in samples]
for q in [70, 80, 85, 90, 95, 98]:
    for w in [32, 64]:
        forensics.QUALITY, forensics.WINDOW, forensics.STEP = q, w, w // 2
        z = [det.score(img)[1] for img, _ in samples]
        print(f"q={q} w={w}: AUC {roc_auc_score(y, z):.3f}")
