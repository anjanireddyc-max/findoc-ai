"""Simulates the faculty test: a genuine receipt downloaded as JPEG, and the same receipt
edited (a price changed) and saved again. Reports how often the ELA detector is right."""
import io
import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from forensics import ELADetector

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "test_files", "online_sim")
os.makedirs(OUT, exist_ok=True)
det = ELADetector()
random.seed(1)


def jpeg(img, q):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")


def edit(img, kind):
    """Changes a small area the way a person would with Paint / an online editor."""
    img = img.copy(); w, h = img.size
    grey = np.asarray(img.convert("L"))
    rows = np.where((grey < 120).mean(axis=1) > 0.04)[0]
    y = int(random.choice(rows[len(rows) // 3: 2 * len(rows) // 3])) if len(rows) else h // 2
    x = int(w * 0.62)
    d = ImageDraw.Draw(img)
    if kind == "retype":                       # white-out a price and type a new one
        d.rectangle([x, y - 12, x + 90, y + 12], fill=(255, 255, 255))
        try:
            font = ImageFont.truetype("cour.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        d.text((x + 4, y - 11), "99.90", fill=(20, 20, 20), font=font)
    else:                                      # copy-move a block of text onto another line
        sy = int(random.choice(rows)) if len(rows) else h // 3
        patch = img.crop((int(w * 0.3), sy - 12, int(w * 0.3) + 90, sy + 12))
        img.paste(patch, (x, y - 12))
    return img


df = pd.read_csv(os.path.join(BASE, "annotations", "test.csv"))
genuine = df[df.forged == 0].image.tolist()[:60]
res = {"genuine": [], "retype": [], "copymove": []}
for i, name in enumerate(genuine):
    src = Image.open(os.path.join(BASE, "dataset", "test", name)).convert("RGB")
    online = jpeg(src, random.choice([80, 85, 90, 95]))            # "downloaded" genuine image
    res["genuine"].append(det.analyze(online)["label"] == "Genuine")
    for kind in ("retype", "copymove"):
        edited = jpeg(edit(online, kind), random.choice([85, 90, 95]))   # edited and re-saved
        res[kind].append(det.analyze(edited)["label"] == "Tampered")
        if i < 3:
            edited.save(os.path.join(OUT, f"edited_{kind}_{i}.jpg"), quality=95)
    if i < 3:
        online.save(os.path.join(OUT, f"genuine_{i}.jpg"), quality=95)
for k, v in res.items():
    print(f"{k:9s}: correct {sum(v)}/{len(v)}  ({100 * np.mean(v):.0f}%)")
