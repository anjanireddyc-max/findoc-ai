"""Admin pages (FastAPI router): login, dashboard, settings and a self-test of the detector.

The admin password comes from the ADMIN_PASSWORD environment variable (on Vercel it is set
in the project's environment variables). For a local run it can also be read from the file
admin_password.txt next to app.py, which is never committed or deployed.
"""
import hashlib
import hmac
import os
import platform
import secrets
import time

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

import admin_settings as settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "admin_samples")

# sample file -> expected result (from the test split, or made for the demo)
SAMPLES = [
    ("tampered_receipt.png", "Tampered", "Scanned receipt with edited prices"),
    ("genuine_receipt.png", "Genuine", "Untouched scanned receipt"),
    ("edited_invoice.pdf", "Signs of editing found", "Typed PDF invoice changed with iLovePDF"),
    ("genuine_invoice.pdf", "No signs of editing found", "Typed PDF invoice saved once"),
]

router = APIRouter(prefix="/admin", include_in_schema=False)
_state = {}                   # templates, status function and analysis function, set by init_admin
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


def init_admin(app, templates, status, analyse):
    """Adds the signed-cookie session middleware and the /admin routes to the FastAPI app."""
    password = admin_password()
    # The session key must be the same on every Vercel instance, so it is derived from secrets
    # that all instances share instead of being random per instance.
    secret = os.environ.get("SECRET_KEY") or hashlib.sha256(
        ("findoc-admin-session:" + (password or secrets.token_hex(16))).encode()).hexdigest()
    app.add_middleware(SessionMiddleware, secret_key=secret, session_cookie="findoc_admin",
                       max_age=4 * 3600, same_site="lax", https_only=settings.ON_VERCEL)
    _state.update(templates=templates, status=status, analyse=analyse)
    app.include_router(router)


# ------------------------------------------------------------------ helpers

def _csrf(request):
    if "csrf" not in request.session:
        request.session["csrf"] = secrets.token_hex(16)
    return request.session["csrf"]


def _render(request, name, status_code=200, **context):
    token = _csrf(request)
    context.update(is_admin=bool(request.session.get("admin")), csrf_token=lambda: token)
    return _state["templates"].TemplateResponse(request, name, context, status_code=status_code)


async def _form(request):
    """Reads the submitted form and checks its CSRF token."""
    form = await request.form()
    sent = str(form.get("csrf", ""))
    if not sent or not hmac.compare_digest(sent, request.session.get("csrf", "")):
        return None
    return form


def _expired(request):
    return HTMLResponse("The form expired. Please go back, reload the page and try again.", status_code=400)


def _not_logged_in(request):
    if request.session.get("admin"):
        return None
    return RedirectResponse(f"/admin/login?next={request.url.path}", status_code=302)


def _client_ip(request):
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "")


def _locked(ip):
    failures, first = _failed.get(ip, (0, 0))
    if failures >= MAX_FAILURES and time.time() - first < LOCK_SECONDS:
        return int(LOCK_SECONDS - (time.time() - first)) // 60 + 1
    if time.time() - first >= LOCK_SECONDS:
        _failed.pop(ip, None)
    return 0


# ------------------------------------------------------------------ routes

@router.get("/login")
def login_page(request: Request):
    if not admin_password():
        return _render(request, "admin_login.html", not_configured=True)
    if request.session.get("admin"):
        return RedirectResponse("/admin/", status_code=302)
    return _render(request, "admin_login.html")


@router.post("/login")
async def login(request: Request):
    password = admin_password()
    if not password:
        return _render(request, "admin_login.html", not_configured=True)
    form = await _form(request)
    if form is None:
        return _expired(request)
    ip = _client_ip(request)
    minutes = _locked(ip)
    if minutes:
        return _render(request, "admin_login.html", error=f"Too many wrong attempts. Try again in {minutes} minute(s).")
    if hmac.compare_digest(str(form.get("password", "")).encode(), password.encode()):
        _failed.pop(ip, None)
        request.session.clear()
        request.session["admin"] = True
        target = request.query_params.get("next", "")
        return RedirectResponse(target if target.startswith("/admin") else "/admin/", status_code=302)
    failures, first = _failed.get(ip, (0, time.time()))
    _failed[ip] = (failures + 1, first)
    return _render(request, "admin_login.html", error="Wrong password.")


@router.post("/logout")
async def logout(request: Request):
    if await _form(request) is None:
        return _expired(request)
    request.session.clear()
    return RedirectResponse("/admin/login", status_code=302)


@router.get("/")
def dashboard(request: Request):
    if redirect := _not_logged_in(request):
        return redirect
    return _render(request, "admin_dashboard.html", active="dashboard", status=_state["status"](),
                   activity=settings.activity(), values=settings.load(),
                   python=platform.python_version(), on_vercel=settings.ON_VERCEL)


def _settings_page(request, values, errors=(), saved=False):
    return _render(request, "admin_settings.html", active="settings", values=values, errors=list(errors),
                   saved=saved, defaults=settings.defaults(), limits=settings.LIMITS, on_vercel=settings.ON_VERCEL)


@router.get("/settings")
def settings_page(request: Request):
    if redirect := _not_logged_in(request):
        return redirect
    return _settings_page(request, settings.load())


@router.post("/settings")
async def settings_save(request: Request):
    if redirect := _not_logged_in(request):
        return redirect
    form = await _form(request)
    if form is None:
        return _expired(request)
    values, errors = settings.validate(form)
    if not errors:
        settings.save(values)
    return _settings_page(request, values, errors, saved=not errors)


@router.post("/settings/reset")
async def settings_reset(request: Request):
    if redirect := _not_logged_in(request):
        return redirect
    if await _form(request) is None:
        return _expired(request)
    settings.reset()
    return RedirectResponse("/admin/settings", status_code=302)


@router.post("/activity/clear")
async def activity_clear(request: Request):
    if redirect := _not_logged_in(request):
        return redirect
    if await _form(request) is None:
        return _expired(request)
    settings.clear_activity()
    return RedirectResponse("/admin/", status_code=302)


def _run_samples():
    results = []
    for name, expected, description in SAMPLES:
        path = os.path.join(SAMPLES_DIR, name)
        row = dict(name=name, description=description, expected=expected, seconds=None, score=None)
        if not os.path.exists(path):
            results.append(dict(row, got="Sample file missing", passed=False))
            continue
        start = time.time()
        with open(path, "rb") as f:
            data = f.read()
        try:
            _, summary = _state["analyse"](data, name)
            got, score = summary["verdict"], summary["score"]
        except Exception as e:                 # report the problem instead of failing the page
            got, score = f"Error: {e}", None
        results.append(dict(row, got=got, passed=got == expected, seconds=f"{time.time() - start:.1f}",
                            score=None if score is None else f"{score:.2f}"))
    return results


@router.get("/self-test")
def self_test_page(request: Request):
    if redirect := _not_logged_in(request):
        return redirect
    return _render(request, "admin_selftest.html", active="selftest", results=None, samples=SAMPLES)


@router.post("/self-test")
async def self_test(request: Request):
    if redirect := _not_logged_in(request):
        return redirect
    if await _form(request) is None:
        return _expired(request)
    results = await run_in_threadpool(_run_samples)
    return _render(request, "admin_selftest.html", active="selftest", results=results, samples=SAMPLES)
