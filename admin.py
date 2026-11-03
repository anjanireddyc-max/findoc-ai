"""Admin pages: login, dashboard, settings and a self-test of the detection system.

The admin password comes from the ADMIN_PASSWORD environment variable (on Vercel it is set
in the project's environment variables). For a local run it can also be read from the file
admin_password.txt next to app.py, which is never committed or deployed.
"""
import hashlib
import hmac
import os
import platform
import re
import secrets
import time
from functools import wraps

from flask import Blueprint, abort, current_app, redirect, render_template, request, session, url_for

import admin_settings as settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "admin_samples")
SELF_TEST_HEADER = "X-FinDoc-Self-Test"

# sample file -> expected result (all taken from the test split or made for the demo)
SAMPLES = [
    ("tampered_receipt.png", "Tampered", "Scanned receipt with edited prices"),
    ("genuine_receipt.png", "Genuine", "Untouched scanned receipt"),
    ("edited_invoice.pdf", "Signs of editing found", "Typed PDF invoice changed with iLovePDF"),
    ("genuine_invoice.pdf", "No signs of editing found", "Typed PDF invoice saved once"),
]

bp = Blueprint("admin", __name__, url_prefix="/admin")

_failed = {}                  # ip -> (failures, first failure time)
MAX_FAILURES, LOCK_SECONDS = 5, 300


def admin_password():
    password = os.environ.get("ADMIN_PASSWORD", "")
    if not password and not settings.ON_VERCEL:
        path = os.path.join(BASE_DIR, "admin_password.txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                password = f.read().strip()
    return password


def init_admin(app, status):
    """status: function returning a dict about the loaded detectors (called by the dashboard)."""
    password = admin_password()
    # The session key must be the same on every Vercel instance, so it is derived from secrets
    # that all instances share instead of being random per instance.
    app.secret_key = os.environ.get("SECRET_KEY") or hashlib.sha256(
        ("findoc-admin-session:" + (password or secrets.token_hex(16))).encode()).hexdigest()
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                      SESSION_COOKIE_SECURE=settings.ON_VERCEL, PERMANENT_SESSION_LIFETIME=4 * 3600)
    app.config["SELF_TEST_TOKEN"] = secrets.token_hex(16)
    app.extensions["findoc_status"] = status
    app.register_blueprint(bp)


# ------------------------------------------------------------------ helpers

def _csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


def _check_csrf():
    sent = request.form.get("csrf", "")
    if not sent or not hmac.compare_digest(sent, session.get("csrf", "")):
        abort(400, "The form expired. Please go back and try again.")


@bp.app_context_processor
def _inject():
    return {"csrf_token": _csrf_token}


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("admin.login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


def _client_ip():
    return (request.headers.get("X-Forwarded-For", request.remote_addr or "") or "").split(",")[0].strip()


def _locked(ip):
    failures, first = _failed.get(ip, (0, 0))
    if failures >= MAX_FAILURES and time.time() - first < LOCK_SECONDS:
        return int(LOCK_SECONDS - (time.time() - first)) // 60 + 1
    if time.time() - first >= LOCK_SECONDS:
        _failed.pop(ip, None)
    return 0


# ------------------------------------------------------------------ routes

@bp.route("/login", methods=["GET", "POST"])
def login():
    password = admin_password()
    if not password:
        return render_template("admin_login.html", not_configured=True)
    if session.get("admin"):
        return redirect(url_for("admin.dashboard"))
    error = None
    ip = _client_ip()
    if request.method == "POST":
        _check_csrf()
        minutes = _locked(ip)
        if minutes:
            error = f"Too many wrong attempts. Try again in {minutes} minute(s)."
        elif hmac.compare_digest(request.form.get("password", "").encode(), password.encode()):
            _failed.pop(ip, None)
            session.clear()
            session.permanent = True
            session["admin"] = True
            target = request.args.get("next", "")
            return redirect(target if target.startswith("/admin") else url_for("admin.dashboard"))
        else:
            failures, first = _failed.get(ip, (0, time.time()))
            _failed[ip] = (failures + 1, first)
            error = "Wrong password."
    return render_template("admin_login.html", error=error)


@bp.route("/logout", methods=["POST"])
def logout():
    _check_csrf()
    session.clear()
    return redirect(url_for("admin.login"))


@bp.route("/")
@login_required
def dashboard():
    status = current_app.extensions["findoc_status"]()
    return render_template("admin_dashboard.html", active="dashboard", status=status,
                           activity=settings.activity(), values=settings.load(),
                           python=platform.python_version(), on_vercel=settings.ON_VERCEL)


@bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings_page():
    errors, saved = [], False
    values = settings.load()
    if request.method == "POST":
        _check_csrf()
        values, errors = settings.validate(request.form)
        if not errors:
            settings.save(values)
            saved = True
    return render_template("admin_settings.html", active="settings", values=values, errors=errors, saved=saved,
                           defaults=settings.defaults(), limits=settings.LIMITS, on_vercel=settings.ON_VERCEL)


@bp.route("/settings/reset", methods=["POST"])
@login_required
def settings_reset():
    _check_csrf()
    settings.reset()
    return redirect(url_for("admin.settings_page"))


@bp.route("/activity/clear", methods=["POST"])
@login_required
def activity_clear():
    _check_csrf()
    settings.clear_activity()
    return redirect(url_for("admin.dashboard"))


@bp.route("/self-test", methods=["GET", "POST"])
@login_required
def self_test():
    results = None
    if request.method == "POST":
        _check_csrf()
        results = []
        client = current_app.test_client()
        headers = {SELF_TEST_HEADER: current_app.config["SELF_TEST_TOKEN"]}
        for name, expected, description in SAMPLES:
            path = os.path.join(SAMPLES_DIR, name)
            if not os.path.exists(path):
                results.append(dict(name=name, description=description, expected=expected,
                                    got="Sample file missing", passed=False, seconds=None, score=None))
                continue
            start = time.time()
            with open(path, "rb") as f:
                response = client.post("/predict", data={"document": (f, name)}, headers=headers,
                                       content_type="multipart/form-data")
            html = response.get_data(as_text=True)
            verdict = re.search(r'data-verdict="([^"]*)"', html)
            score = re.search(r'data-score="([^"]*)"', html)
            got = verdict.group(1) if verdict else f"No result (HTTP {response.status_code})"
            results.append(dict(name=name, description=description, expected=expected, got=got,
                                passed=got == expected, seconds=f"{time.time() - start:.1f}",
                                score=score.group(1) if score and score.group(1) else None))
    return render_template("admin_selftest.html", active="selftest", results=results, samples=SAMPLES)
