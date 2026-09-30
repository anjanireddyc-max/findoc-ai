"""Analyse any receipt or invoice image from the command line.

    python predict_document.py path/to/document.png

Without an argument, the first test document is used and its true label from
annotations/test.csv is printed for comparison (the label is NOT used to predict).
"""
import os
import sys

import pandas as pd
from PIL import Image

from detector import BASE_DIR, TamperDetector


def main():
    if len(sys.argv) > 1:
        image_path, actual = sys.argv[1], None
    else:
        df = pd.read_csv(os.path.join(BASE_DIR, "annotations", "test.csv"))
        row = df.iloc[0]
        image_path = os.path.join(BASE_DIR, "dataset", "test", row["image"])
        actual = "Tampered" if int(row["forged"]) == 1 else "Genuine"

    detector = TamperDetector()
    result = detector.analyze(Image.open(image_path))

    print("\n==============================")
    print("DOCUMENT ANALYSIS")
    print("==============================")
    print(f"Image:                 {os.path.basename(image_path)}")
    print(f"Regions analysed:      {result['windows']}")
    print(f"Regions flagged:       {result['flagged']}")
    print(f"Highest region score:  {result['probability']:.2%}")
    print(f"Decision threshold:    {result['threshold']:.2%}")
    print(f"\nFINAL RESULT: {result['label']}")
    if actual:
        print(f"ACTUAL LABEL: {actual}")


if __name__ == "__main__":
    main()
