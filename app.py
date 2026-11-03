"""FinDoc AI web application (FastAPI).

Pages are rendered with Jinja2 templates; /predict receives the uploaded document as
multipart/form-data and returns the result page. /api/analyze returns the same analysis as
JSON, and FastAPI documents every API endpoint automatically at /docs.
"""
import base64
import io
import json
import os
import time

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw
from starlette.concurrency import run_in_threadpool

import admin_settings
import converter
from admin import init_admin
from converter import ALLOWED_EXTENSIONS, ConversionError, extension, load_pages
from forensics import ELADetector, WINDOW, STEP, QUALITY
from metadata import inspect as inspect_file

try:                                   # EfficientNetV2B0 on ELA maps (ONNX, optional)
    from ela_cnn import ELACNN
except Exception:                      # pragma: no cover - model or runtime not installed
    ELACNN = None


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, "models", "final_metrics.json")

app = FastAPI(
    title="FinDoc AI",
    description="Financial document tampering detection with Error Level Analysis, "
                "EfficientNetV2B0 and Grad-CAM.",
    version="2.0",
)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# =========================================================
# LOAD DETECTORS
# =========================================================

ela_detector = ELADetector()
cnn = None
if ELACNN is not None:
    try:
        cnn = ELACNN()
    except Exception as e:             # the ELA detector still works on its own
        print("ELA-CNN not loaded:", e)
print(f"ELA detector ready (threshold z >= {ela_detector.threshold:.2f}); "
      f"EfficientNetV2B0-ELA model: {'loaded' if cnn else 'not available'}")


def apply_settings():
    """Applies the admin settings (they can change while the app runs)."""
    cfg = admin_settings.load()
    ela_detector.threshold, ela_detector.jpeg_threshold = cfg["scan_threshold"], cfg["jpeg_threshold"]
    converter.MAX_PAGES = cfg["max_pages"]
    return cfg


@app.middleware("http")
async def settings_middleware(request: Request, call_next):
    """Runs before every request and applies the current admin settings."""
    apply_settings()
    return await call_next(request)


# =========================================================
# HELPERS
# =========================================================

def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        data = json.load(f)
    test = data["test"]
    (tn, fp), (fn, tp) = test["confusion_matrix"]
    return {
        "accuracy": round(test["accuracy"] * 100, 2), "precision": round(test["precision"] * 100, 2),
        "recall": round(test["recall"] * 100, 2), "f1": round(test["f1"] * 100, 2), "auc": round(test["auc"] * 100, 2),
        "documents": data["test_documents"], "tampered": data["test_tampered"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "threshold": data.get("threshold_text", ""),
        "method": data.get("method", "ELA"),
    }


def jet(values):
    """JET colour map (blue -> cyan -> yellow -> red) for values 0..1."""
    v = np.clip(values, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def data_uri(image, fmt="JPEG"):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, fmt, quality=88)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()


def limit(image, side=1400):
    s = min(1.0, side / max(image.size))
    return image if s == 1 else image.resize((round(image.width * s), round(image.height * s)), Image.LANCZOS)


def overlay(image, heat, boxes, alpha=0.45):
    """Heatmap over the page; flagged regions outlined in red."""
    base = np.asarray(image.convert("RGB"), np.float32)
    mix = (1 - alpha) * base + alpha * jet(heat).astype(np.float32)
    out = Image.fromarray(mix.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(230, 30, 30), width=3)
    return out


def where(box, size):
    width, height = size
    cx, cy = (box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height
    vertical = "top" if cy < 1 / 3 else "middle" if cy < 2 / 3 else "bottom"
    horizontal = "left" if cx < 1 / 3 else "centre" if cx < 2 / 3 else "right"
    return "centre" if (vertical, horizontal) == ("middle", "centre") else f"{vertical}-{horizontal}"


def explain(a, page_label, page_count, metrics, info):
    z, t = a["z"], a["threshold"]
    position = where(a["best_box"], a["image"].size)
    reasons = []
    if a["label"] == "Tampered":
        reasons.append(f"One region's compression error differs from the rest of the text by {z:.1f} standard "
                       f"deviations, which is at or above the decision threshold of {t:.1f}, so the document is "
                       "reported as tampered.")
        reasons.append(f"{a['flagged']} of the {a['windows']} text regions analysed crossed the threshold. "
                       "Pasted, re-typed or painted-over areas were saved a different number of times than the rest "
                       "of the page, so their error level stands out.")
    else:
        reasons.append(f"The most unusual region differs from the rest of the text by only {z:.1f} standard "
                       f"deviations, below the decision threshold of {t:.1f}, so the compression error is "
                       "consistent across the page.")
        reasons.append(f"None of the {a['windows']} text regions analysed crossed the threshold.")
    reasons.append(f"The most unusual region is in the {position} part of "
                   f"{'the page' if page_count == 1 else page_label.lower()} (outlined on the heatmap).")
    if a.get("cnn_prob") is not None:
        reasons.append(f"The EfficientNetV2B0 model trained on ELA maps gives this region a tampering "
                       f"probability of {a['cnn_prob'] * 100:.1f}%; its Grad-CAM map shows which pixels it used.")
    if abs(z - t) < 1.0:
        reasons.append("The score is close to the threshold, so this is a borderline result. Please check the "
                       "highlighted area manually.")
    if page_count > 1:
        reasons.append(f"The file has {page_count} pages; {page_label.lower()} had the most unusual region and is shown here.")
    reasons += [f"File check: {f}" for f in info["findings"]]
    if metrics:
        reasons.append(f"Reliability: on {metrics['documents']} test receipts the detector reached "
                       f"{metrics['accuracy']}% accuracy and caught {metrics['tp']} of {metrics['tampered']} tampered "
                       "receipts, so treat this result as a screening signal, not proof.")
    steps = [
        f"The {'page was' if page_count == 1 else f'{page_count} pages were'} re-saved as JPEG at quality {QUALITY} "
        "and compared with the original (Error Level Analysis).",
        f"The error was averaged over the text pixels of {a['windows']} overlapping {WINDOW} x {WINDOW} pixel "
        f"regions (every {STEP} pixels); blank paper was left out.",
        "Each region was compared with the page's typical region; the largest difference, in standard deviations, "
        "is the tampering score.",
        f"The score ({z:.1f}) was compared with the decision threshold ({t:.1f}) chosen on the validation receipts.",
    ]
    if a.get("cnn_prob") is not None:
        steps.append("The most unusual region was also classified by EfficientNetV2B0 (trained on ELA maps), and "
                     "Grad-CAM showed which pixels influenced it.")
    return reasons, steps


RULE_PATH = os.path.join(BASE_DIR, "models", "final_rule.json")
RULE = json.load(open(RULE_PATH)) if os.path.exists(RULE_PATH) else {"rule": "ela"}
if RULE["rule"] != "ela" and cnn is None:          # model files missing: fall back to ELA alone
    RULE = {"rule": "ela"}


def analyze_page(image, is_jpeg=False):
    a = ela_detector.analyze(image, is_jpeg)
    a["cnn_prob"], a["gradcam"] = None, None
    if cnn is not None:
        _, _, boxes, zs, _ = ela_detector.score(a["image"])
        best = None
        for i in (np.argsort(zs)[::-1][:5] if len(zs) else []):
            box = cnn.window_around(boxes[i], a["image"].size)
            prob, cam = cnn.predict_window(a["ela"], box)
            if best is None or prob > best[0]:
                best = (prob, cam, box)
        if best:
            a["cnn_prob"], a["gradcam"], a["cnn_box"] = best
    if RULE["rule"] != "ela":
        lo, hi = RULE["lo"], RULE["hi"]
        n = lambda v, c: (v - lo[c]) / (hi[c] - lo[c] + 1e-9)
        score = a["cnn_prob"] if RULE["rule"] == "cnn_top" else (n(a["z"], "ela") + n(a["cnn_prob"], "cnn_top")) / 2
        a["decision_score"], a["decision_threshold"] = float(score), float(RULE["threshold"])
        a["label"] = "Tampered" if score >= RULE["threshold"] else "Genuine"
    else:
        a["decision_score"], a["decision_threshold"] = a["z"], a["threshold"]
    return a


def metadata_result(info, pages, common):
    """Result context for typed Word / PowerPoint / digital PDF files (metadata check)."""
    page_label, first = pages[0]
    edited = info["signals"] > 0
    reasons = [f"This is a {info['kind']}: its pages were produced by software, not scanned or photographed, "
               "so there are no scanning or compression traces to analyse. The image-based tampering score is "
               "therefore not used for this file."] + info["findings"]
    reasons.append("Signs of editing were found in the file's metadata; compare the document with the original "
                   "from its issuer." if edited else
                   "No signs of editing were found in the file's metadata. Metadata can be removed or changed, "
                   "so this is not proof that the content is genuine.")
    steps = ["The file type and structure were checked (Word, PowerPoint or PDF with a text layer).",
             "The metadata was read: creating program, author, last editor, creation and modification dates, "
             "revision count and, for PDFs, the number of saves and the PDF tools used.",
             "Each finding that indicates editing after creation was counted as an editing signal."]
    return dict(
        digital=True, **common,
        result="Signs of editing found" if edited else "No signs of editing found",
        verdict_class="verdict-neutral" if edited else "verdict-genuine",
        verdict_text="The file's metadata shows that it was changed after it was created." if edited else
                     "The metadata does not show changes after creation.",
        reasons=reasons, steps=steps, page_label=page_label if len(pages) > 1 else None,
        original_image=data_uri(limit(first)) if first is not None else None, out_of_domain=False)


# =========================================================
# ANALYSIS (shared by the web page, the JSON API and the admin self-test)
# =========================================================

def run_analysis(data, filename):
    """Analyses one uploaded file. Returns (context for result.html, summary for the API and the log).
    Raises ConversionError when the file cannot be read."""
    pages = load_pages(data, filename)
    ext = extension(filename)
    used_pictures = any(label.startswith("Picture") for label, _ in pages)
    info = inspect_file(data, ext, used_pictures)
    common = dict(file_name=filename, page_count=len(pages), kind=info["kind"], signals=info["signals"])

    if info["digital"]:                    # typed Word / PowerPoint / digital PDF
        context = metadata_result(info, pages, common)
        summary = {"method": "metadata", "verdict": context["result"], "score": None, "threshold": None,
                   "outcome": "Edited" if info["signals"] > 0 else "Clean",
                   "detail": f"{info['signals']} metadata signal(s)"}
        return context, summary

    # ---------------- scanned / photographed documents: ELA ----------------
    is_jpeg = ext in ("jpg", "jpeg")
    results = [(label, analyze_page(image, is_jpeg)) for label, image in pages]
    page_label, a = max(results, key=lambda r: r[1]["decision_score"])
    if any(r["label"] == "Tampered" for _, r in results):
        a["label"] = "Tampered"
    tampered = a["label"] == "Tampered"

    page_image = a["image"]
    # colours are relative to the decision threshold: a genuine page stays blue / cyan and only
    # regions that reach the threshold turn red; boxes are drawn only around flagged regions
    strength = min(1.0, max(a["decision_score"], 0.0) / max(a["decision_threshold"], 1e-6))
    heat_img = overlay(page_image, a["heat"] * strength, a["flagged_boxes"] if tampered else [])
    bx = a.get("cnn_box") or a["best_box"]
    region = page_image.crop(bx)
    if a.get("gradcam") is not None:
        region = overlay(region, a["gradcam"] * strength, [])
    else:
        region = overlay(region, a["heat"][bx[1]:bx[3], bx[0]:bx[2]] * strength, [])
    region = region.resize((320, 320), Image.NEAREST)

    reasons, steps = explain(a, page_label, len(pages), load_metrics(), info)
    context = dict(
        digital=False, **common,
        result=a["label"],
        verdict_class="verdict-tampered" if tampered else "verdict-genuine",
        verdict_text="An area of the page was compressed differently from the rest of the text." if tampered else
                     "The compression error is consistent across the text of the page.",
        confidence=f"{a['decision_score']:.2f}", highest_score=f"{a['decision_score']:.2f}",
        average_score=f"{a['flagged']}", regions=a["windows"], flagged_regions=a["flagged"],
        threshold=f"{a['decision_threshold']:.2f}", out_of_domain=False,
        page_label=page_label if len(pages) > 1 else None,
        reasons=reasons, steps=steps,
        original_image=data_uri(limit(page_image)), gradcam_image=data_uri(limit(heat_img)),
        region_image=data_uri(region),
        region_label="MOST SUSPICIOUS REGION - GRAD-CAM (EFFICIENTNETV2B0-ELA)" if a.get("gradcam") is not None
                     else "MOST SUSPICIOUS REGION (ZOOMED)",
        heat_label="ELA TAMPERING HEATMAP" if tampered else "ELA HEATMAP (NO REGION ABOVE THRESHOLD)",
    )
    summary = {"method": "ela", "verdict": a["label"], "score": round(float(a["decision_score"]), 3),
               "threshold": round(float(a["decision_threshold"]), 3), "outcome": a["label"],
               "flagged_regions": int(a["flagged"]), "regions": int(a["windows"]),
               "cnn_probability": None if a.get("cnn_prob") is None else round(float(a["cnn_prob"]), 4),
               "detail": f"{a['flagged']} region(s) flagged"}
    return context, summary


def system_status():
    """Shown on the admin dashboard and by /api/health."""
    try:
        import onnxruntime
        ort_version = onnxruntime.__version__
    except Exception:
        ort_version = None
    return {"ela_ready": ela_detector is not None, "cnn_loaded": cnn is not None, "onnxruntime": ort_version,
            "trained_scan_threshold": admin_settings.defaults()["scan_threshold"],
            "trained_jpeg_threshold": admin_settings.defaults()["jpeg_threshold"], "metrics": load_metrics()}


init_admin(app, templates, system_status, run_analysis)


# =========================================================
# PAGES
# =========================================================

def page(request, name, status=200, **context):
    return templates.TemplateResponse(request, name, context, status_code=status)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request):
    return page(request, "index.html")


def detect_page(request, error=None, status=200):
    cfg = admin_settings.load()
    if cfg["maintenance"] and not error:
        error = cfg["maintenance_message"]
    accept = ",".join("." + e for e in sorted(admin_settings.allowed_extensions(cfg)))
    return page(request, "detect.html", status, error=error, max_mb=cfg["max_upload_mb"], accept=accept,
                maintenance=cfg["maintenance"])


@app.get("/detect", response_class=HTMLResponse, include_in_schema=False)
def detect(request: Request):
    return detect_page(request)


@app.get("/how-it-works", response_class=HTMLResponse, include_in_schema=False)
def how_it_works(request: Request):
    return page(request, "how_it_works.html")


@app.get("/performance", response_class=HTMLResponse, include_in_schema=False)
def performance(request: Request):
    return page(request, "performance.html", metrics=load_metrics())


@app.get("/about", response_class=HTMLResponse, include_in_schema=False)
def about(request: Request):
    return page(request, "about.html")


# =========================================================
# UPLOAD CHECKS
# =========================================================

class UploadRejected(Exception):
    def __init__(self, message, status=400, file_type="upload", detail=""):
        super().__init__(message)
        self.message, self.status, self.file_type, self.detail = message, status, file_type, detail


def check_request_size(request, cfg):
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > cfg["max_upload_mb"] * 1024 * 1024:
        raise UploadRejected(f"The file is larger than {cfg['max_upload_mb']} MB.", 413, detail="file too large")


def check_file(filename, data, cfg):
    if not filename:
        raise UploadRejected("Please upload or capture a document.")
    ext = extension(filename)
    file_type = admin_settings.group_of(ext)
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadRejected("Supported files: images (PNG, JPG, BMP, TIFF, WEBP), PDF, Word (DOCX, DOC) and "
                             "PowerPoint (PPTX, PPT).", 415, file_type, f".{ext} not supported")
    if ext not in admin_settings.allowed_extensions(cfg):
        raise UploadRejected(f"{file_type.capitalize()} files are currently disabled. Please upload another file type.",
                             415, file_type, f".{ext} disabled by admin")
    if len(data) > cfg["max_upload_mb"] * 1024 * 1024:
        raise UploadRejected(f"The file is larger than {cfg['max_upload_mb']} MB.", 413, file_type, "file too large")
    return file_type


async def receive(request, upload, cfg):
    """Shared by /predict and /api/analyze: checks, reads and analyses the upload."""
    started = time.time()
    if cfg["maintenance"]:
        raise UploadRejected(cfg["maintenance_message"], 503, detail="maintenance mode")
    check_request_size(request, cfg)
    filename = (upload.filename or "") if upload is not None else ""
    data = await upload.read() if upload is not None else b""
    file_type = check_file(filename, data, cfg)
    try:
        context, summary = await run_in_threadpool(run_analysis, data, filename)   # CPU work off the event loop
    except ConversionError as e:
        raise UploadRejected(str(e), 422, file_type, str(e)[:80])
    seconds = time.time() - started
    admin_settings.record(file_type, summary["outcome"], summary["score"], seconds, summary["detail"])
    return context, summary, seconds


# =========================================================
# PREDICT (web page)
# =========================================================

@app.post("/predict", response_class=HTMLResponse, include_in_schema=False)
async def predict(request: Request):
    cfg = admin_settings.load()
    try:
        upload = None
        if not cfg["maintenance"]:
            check_request_size(request, cfg)
            form = await request.form()
            upload = form.get("document")
            if isinstance(upload, str):          # empty file field
                upload = None
        context, _, _ = await receive(request, upload, cfg)
    except UploadRejected as e:
        if e.detail:
            admin_settings.record(e.file_type, "Rejected", detail=e.detail)
        return detect_page(request, e.message, e.status if e.status in (413, 503) else 200)
    return page(request, "result.html", **context)


# =========================================================
# JSON API (documented automatically at /docs)
# =========================================================

@app.post("/api/analyze", tags=["API"], summary="Analyse a document and return the result as JSON")
async def api_analyze(request: Request,
                      document: UploadFile = File(..., description="Image, PDF, Word or PowerPoint file")):
    cfg = admin_settings.load()
    try:
        context, summary, seconds = await receive(request, document, cfg)
    except UploadRejected as e:
        if e.detail:
            admin_settings.record(e.file_type, "Rejected", detail=e.detail)
        return JSONResponse({"error": e.message}, status_code=e.status)
    return {"file_name": context["file_name"], "kind": context["kind"], "pages": context["page_count"],
            **{k: v for k, v in summary.items() if k not in ("outcome", "detail")},
            "reasons": context["reasons"], "steps": context["steps"], "seconds": round(seconds, 2)}


@app.get("/api/health", tags=["API"], summary="Check that the detector and the model are loaded")
def api_health():
    s = system_status()
    cfg = admin_settings.load()
    return {"status": "ok", "ela_detector": s["ela_ready"], "efficientnet_model": s["cnn_loaded"],
            "onnxruntime": s["onnxruntime"], "maintenance": cfg["maintenance"],
            "scan_threshold": cfg["scan_threshold"], "jpeg_threshold": cfg["jpeg_threshold"],
            "max_upload_mb": cfg["max_upload_mb"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
