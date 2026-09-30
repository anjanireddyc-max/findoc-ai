import os
import ast
import random

import numpy as np
import pandas as pd
from PIL import Image


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
# ELA=1 writes Error Level Analysis maps of the windows to patches_ela/ instead of pixels
ELA = os.environ.get("ELA", "0") == "1"
ELA_QUALITY = int(os.environ.get("ELA_QUALITY", "75"))
PATCH_DIR = os.path.join(BASE_DIR, "patches_ela" if ELA else "patches")


# =========================================================
# SETTINGS
# =========================================================

# Every patch - genuine or tampered - is a WINDOW x WINDOW crop taken at the
# document's original resolution. Using one size and one scale for both classes
# stops the model from learning "zoomed-in crop = tampered" instead of real
# tampering traces, and it matches exactly what app.py analyses at run time.
WINDOW = 224

# Windows cut around every edited region (the region is always fully inside)
WINDOWS_PER_TAMPERED_REGION = {"train": 6, "val": 2, "test": 2}

# Windows cut from each genuine document
WINDOWS_PER_GENUINE_DOC = {"train": 5, "val": 3, "test": 3}

# Windows cut from the untouched parts of each forged document (hard negatives)
CLEAN_WINDOWS_PER_FORGED_DOC = {"train": 3, "val": 1, "test": 1}

# A window this uniform (e.g. blank paper) carries no information
MIN_PIXEL_STD = 12.0


# =========================================================
# HELPERS
# =========================================================

def pad_to_window(image):
    """Receipts narrower or shorter than the window are padded with white paper."""
    if image.width >= WINDOW and image.height >= WINDOW:
        return image
    canvas = Image.new("RGB", (max(image.width, WINDOW), max(image.height, WINDOW)), (255, 255, 255))
    canvas.paste(image, (0, 0))
    return canvas


def modified_regions(annotation_value):
    """Bounding boxes (x1, y1, x2, y2) of the edited regions of a forged document."""
    if pd.isna(annotation_value) or str(annotation_value).strip() in ["0", "0.0", "", "nan"]:
        return []
    try:
        data = ast.literal_eval(str(annotation_value))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []

    boxes = []
    for region in data.get("regions", []):
        if not isinstance(region, dict):
            continue
        attributes = region.get("region_attributes", {})
        shape = region.get("shape_attributes", {})
        if not isinstance(attributes, dict) or not isinstance(shape, dict):
            continue
        if attributes.get("Original area") != "no" or shape.get("name") != "rect":
            continue
        try:
            x, y = int(shape["x"]), int(shape["y"])
            w, h = int(shape["width"]), int(shape["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if w > 0 and h > 0:
            boxes.append((x, y, x + w, y + h))
    return boxes


def overlaps(window, box):
    x1, y1, x2, y2 = window
    return not (box[2] <= x1 or box[0] >= x2 or box[3] <= y1 or box[1] >= y2)


def has_content(patch):
    return float(np.asarray(patch.convert("L"), dtype=np.float32).std()) >= MIN_PIXEL_STD


def windows_around(box, image, count):
    """Random WINDOW x WINDOW windows that fully contain the edited box when it fits."""
    bx1, by1, bx2, by2 = box
    results = []
    for _ in range(count):
        # the window must start between (box end - WINDOW) and box start
        lo_x, hi_x = max(0, bx2 - WINDOW), min(bx1, image.width - WINDOW)
        lo_y, hi_y = max(0, by2 - WINDOW), min(by1, image.height - WINDOW)
        if lo_x > hi_x:            # edited box wider than the window: centre on it
            lo_x = hi_x = min(max(0, (bx1 + bx2) // 2 - WINDOW // 2), image.width - WINDOW)
        if lo_y > hi_y:
            lo_y = hi_y = min(max(0, (by1 + by2) // 2 - WINDOW // 2), image.height - WINDOW)
        x, y = random.randint(lo_x, hi_x), random.randint(lo_y, hi_y)
        results.append((x, y, x + WINDOW, y + WINDOW))
    return results


def random_clean_windows(image, avoid, count, tries=40):
    """Random windows with visible content that do not touch any edited region."""
    results = []
    for _ in range(count * tries):
        if len(results) == count:
            break
        x = random.randint(0, image.width - WINDOW)
        y = random.randint(0, image.height - WINDOW)
        window = (x, y, x + WINDOW, y + WINDOW)
        if any(overlaps(window, b) for b in avoid):
            continue
        if has_content(image.crop(window)):
            results.append(window)
    return results


# =========================================================
# PROCESS ONE DATASET SPLIT
# =========================================================

def process_split(split):
    df = pd.read_csv(os.path.join(ANNOTATIONS_DIR, f"{split}.csv"))
    image_dir = os.path.join(DATASET_DIR, split)
    for cls in ["genuine", "tampered"]:
        out_dir = os.path.join(PATCH_DIR, split, cls)
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):                  # start from a clean folder
            os.remove(os.path.join(out_dir, f))

    print("\n================================")
    print(f"PROCESSING: {split.upper()}  ({len(df)} documents)")
    print("================================")

    counts = {"genuine": 0, "tampered": 0}
    skipped = 0

    for index, row in df.iterrows():
        name = str(row["image"]).strip()
        path = os.path.join(image_dir, name)
        if not os.path.exists(path):
            skipped += 1
            continue
        image = pad_to_window(Image.open(path).convert("RGB"))
        source = image
        if ELA:
            from ela import ela_image
            image = ela_image(source, ELA_QUALITY)
        stem = os.path.splitext(name)[0]

        if int(row["forged"]) == 1:
            boxes = modified_regions(row["forgery annotations"])
            for r, box in enumerate(boxes):
                for k, window in enumerate(windows_around(box, image, WINDOWS_PER_TAMPERED_REGION[split])):
                    image.crop(window).save(os.path.join(PATCH_DIR, split, "tampered", f"{index}_{stem}_r{r}_{k}.png"))
                    counts["tampered"] += 1
            clean = random_clean_windows(source, boxes, CLEAN_WINDOWS_PER_FORGED_DOC[split])
        else:
            clean = random_clean_windows(source, [], WINDOWS_PER_GENUINE_DOC[split])

        for k, window in enumerate(clean):
            image.crop(window).save(os.path.join(PATCH_DIR, split, "genuine", f"{index}_{stem}_g{k}.png"))
            counts["genuine"] += 1

    print(f"Tampered patches: {counts['tampered']}")
    print(f"Genuine patches:  {counts['genuine']}")
    print(f"Skipped images:   {skipped}")
    return counts


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    random.seed(42)
    for split in ["train", "val", "test"]:
        process_split(split)
    print("\n================================")
    print("PATCH PREPARATION COMPLETE")
    print("================================")
    print(f"\nPatches saved in:\n{PATCH_DIR}")
