"""Settings that the admin page can change, and a small in-memory activity log.

Settings are stored as JSON. Locally the file is models/admin_settings.json. On Vercel the
project folder is read-only, so changes are written to /tmp: they apply while that server
instance is running and go back to the defaults when Vercel starts a new instance.

The activity log keeps only a summary of each analysis (time, file type, result, score) in
memory. Uploaded documents are never stored.
"""
import json
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime, timedelta, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ON_VERCEL = bool(os.environ.get("VERCEL"))
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "ela_threshold.json")
SETTINGS_PATH = "/tmp/findoc_admin_settings.json" if ON_VERCEL else os.path.join(BASE_DIR, "models", "admin_settings.json")

IST = timezone(timedelta(hours=5, minutes=30), "IST")
PLATFORM_MAX_MB = 4 if ON_VERCEL else 50          # Vercel rejects request bodies above ~4.5 MB

FILE_GROUPS = {
    "images": {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"},
    "pdf": {"pdf"},
    "word": {"docx", "doc"},
    "powerpoint": {"pptx", "ppt"},
}


def _trained_thresholds():
    try:
        with open(THRESHOLD_PATH) as f:
            cfg = json.load(f)
        return float(cfg["threshold"]), float(cfg.get("jpeg_threshold", cfg["threshold"]))
    except (OSError, KeyError, ValueError):
        return 3.0, 3.0


def defaults():
    scan, jpeg = _trained_thresholds()
    return {
        "scan_threshold": round(scan, 3),
        "jpeg_threshold": round(jpeg, 3),
        "max_upload_mb": int(os.environ.get("MAX_UPLOAD_MB", "4" if ON_VERCEL else "25")),
        "max_pages": 5,
        "allow_images": True,
        "allow_pdf": True,
        "allow_word": True,
        "allow_powerpoint": True,
        "maintenance": False,
        "maintenance_message": "The analysis service is under maintenance. Please try again later.",
    }


# (field, minimum, maximum) for numeric settings
LIMITS = {
    "scan_threshold": (1.0, 15.0),
    "jpeg_threshold": (1.0, 15.0),
    "max_upload_mb": (1, PLATFORM_MAX_MB),
    "max_pages": (1, 10),
}

_lock = threading.Lock()
_cache = {"mtime": None, "values": None}


def load():
    """Current settings: defaults overridden by the saved file (re-read only when it changes)."""
    try:
        mtime = os.path.getmtime(SETTINGS_PATH)
    except OSError:
        mtime = None
    with _lock:
        if _cache["values"] is not None and _cache["mtime"] == mtime:
            return dict(_cache["values"])
        values = defaults()
        if mtime is not None:
            try:
                with open(SETTINGS_PATH) as f:
                    saved = json.load(f)
                values.update({k: v for k, v in saved.items() if k in values})
            except (OSError, ValueError):
                pass
        _cache.update(mtime=mtime, values=values)
        return dict(values)


def validate(form):
    """Turns submitted form values into settings. Returns (settings, list of errors)."""
    values, errors = load(), []
    for field, (low, high) in LIMITS.items():
        raw = (form.get(field) or "").strip()
        try:
            number = float(raw) if "threshold" in field else int(raw)
        except ValueError:
            errors.append(f"{field.replace('_', ' ').capitalize()} must be a number.")
            continue
        if not low <= number <= high:
            errors.append(f"{field.replace('_', ' ').capitalize()} must be between {low} and {high}.")
            continue
        values[field] = round(number, 3) if isinstance(number, float) else number
    for group in FILE_GROUPS:
        values[f"allow_{group}"] = form.get(f"allow_{group}") == "on"
    if not any(values[f"allow_{g}"] for g in FILE_GROUPS):
        errors.append("At least one file type must stay enabled.")
    values["maintenance"] = form.get("maintenance") == "on"
    message = (form.get("maintenance_message") or "").strip()
    values["maintenance_message"] = message[:200] or defaults()["maintenance_message"]
    return values, errors


def save(values):
    with _lock:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(values, f, indent=2)
        _cache.update(mtime=None, values=None)


def reset():
    with _lock:
        try:
            os.remove(SETTINGS_PATH)
        except OSError:
            pass
        _cache.update(mtime=None, values=None)


def allowed_extensions(values):
    return set().union(*(exts for group, exts in FILE_GROUPS.items() if values[f"allow_{group}"]))


def group_of(ext):
    return next((group for group, exts in FILE_GROUPS.items() if ext in exts), "other")


# ------------------------------------------------------------------ activity log

STARTED = time.time()
_recent = deque(maxlen=50)
_counts = Counter()
_types = Counter()


def record(file_type, outcome, score=None, seconds=None, detail=""):
    """outcome: Tampered, Genuine, Edited (metadata), Clean (metadata) or Rejected."""
    with _lock:
        _counts["total"] += 1
        _counts[outcome] += 1
        _types[file_type] += 1
        _recent.appendleft({
            "time": datetime.now(IST).strftime("%d %b %Y, %H:%M:%S"),
            "type": file_type, "outcome": outcome,
            "score": None if score is None else f"{score:.2f}",
            "seconds": None if seconds is None else f"{seconds:.1f}",
            "detail": detail,
        })


def activity():
    with _lock:
        return {"counts": dict(_counts), "types": dict(_types), "recent": list(_recent),
                "uptime": _uptime(time.time() - STARTED)}


def clear_activity():
    with _lock:
        _recent.clear()
        _counts.clear()
        _types.clear()


def _uptime(seconds):
    minutes, _ = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes}m" if days else f"{hours}h {minutes}m"
