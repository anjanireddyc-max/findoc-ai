"""Save a full-document Grad-CAM visualisation for one image.

    python gradcam.py path/to/document.png

Without an argument, the first tampered document of the test set is used.
The picture is saved in the gradcam_results folder.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from detector import BASE_DIR, TamperDetector


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        df = pd.read_csv(os.path.join(BASE_DIR, "annotations", "test.csv"))
        name = df[df["forged"] == 1].iloc[0]["image"]
        image_path = os.path.join(BASE_DIR, "dataset", "test", name)

    result = TamperDetector().analyze(Image.open(image_path))
    image = np.asarray(result["image"], dtype=np.float32) / 255.0

    plt.figure(figsize=(6, 6 * image.shape[0] / image.shape[1]))
    plt.imshow(image)
    plt.imshow(result["cam"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"Prediction: {result['label']} ({result['probability']:.2%})")

    out_dir = os.path.join(BASE_DIR, "gradcam_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gradcam_" + os.path.splitext(os.path.basename(image_path))[0] + ".png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Prediction: {result['label']}  ({result['probability']:.2%})")
    print(f"Saved to:\n{out_path}")


if __name__ == "__main__":
    main()
