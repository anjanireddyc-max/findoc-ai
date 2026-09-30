# FinDoc AI – Financial Document Tampering Detection

Explainable web application that checks receipts, invoices and other financial documents for tampering.
III B.Tech I Sem – Industry Oriented Internship, Deep Learning Techniques Development (MR24-1IT0163),
Malla Reddy University. Batch 13: M. Udayini (2411IT010142), C. Anjani (2411IT010063),
C. Sai Kiran (2411IT010067). Guide: Mrs. G. Devi Priya.

## What it does

| Uploaded file | How it is checked |
|---|---|
| Scanned or photographed document (PNG, JPG, BMP, TIFF, WEBP, scanned PDF, picture inside a DOCX/PPTX) | **Error Level Analysis (ELA)**: the page is re-saved as JPEG (quality 70); pasted, re-typed or painted-over areas have a different compression error from the rest of the text. The most unusual 64×64 text region decides the result. **EfficientNetV2B0**, trained on ELA maps, classifies that region and **Grad-CAM** shows the pixels it used. |
| Typed Word / PowerPoint file, digital PDF | **Metadata check**: creating program, author, last editor, dates, number of saves, PDF editing tools. |
| Images | Additionally, EXIF/XMP information about the editing software is reported. |

Every result lists the reasons for the decision and the analysis steps, with a heatmap and a zoomed view of the most suspicious region.

## Results (218 test receipts of the "Find it again!" dataset)

See `models/final_metrics.json` and the Performance page. The first CNN trained on raw pixels performed at
chance level on whole documents (AUC 0.47) because its training patches differed in scale; ELA brought
the document-level AUC to about 0.79.

## Run locally

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. On Windows with Microsoft Office installed, text-only Word and PowerPoint files
are also rendered for preview.

## Retrain / re-evaluate (needs the dataset in `dataset/` and `annotations/`)

```
pip install -r requirements-train.txt
python evaluate_ela.py                      # ELA detector: threshold + metrics
set ELA=1 && python prepare_patches.py      # 224x224 ELA training patches
set ELA=1 && python patch_train.py          # EfficientNetV2B0 on ELA maps
python export_onnx.py                       # ONNX backbone + head weights for the web app
```

## Deployment

`vercel.json` + `api/index.py` deploy the Flask app to Vercel. The web app needs only Flask, NumPy, Pillow,
PyMuPDF and ONNX Runtime (no TensorFlow). Uploads are limited to 4 MB on Vercel.

## Files

- `app.py` – Flask routes, reasons and heatmaps
- `forensics.py`, `ela.py` – ELA detector
- `ela_cnn.py` – EfficientNetV2B0-ELA inference and Grad-CAM with ONNX Runtime
- `metadata.py` – metadata checks for PDF, DOCX, PPTX and images
- `converter.py` – images, PDF, Word and PowerPoint to page images
- `prepare_patches.py`, `patch_train.py`, `evaluate.py`, `evaluate_ela.py`, `export_onnx.py` – training and evaluation
- `detector.py` – the first pixel-based CNN (kept for comparison)
- `test_files/` – sample genuine and tampered files for trying the app
