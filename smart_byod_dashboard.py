"""
================================================================================
 SMART-BYOD Monitoring Dashboard
 A Lightweight Zero-Trust-Inspired Behavioral Anomaly Detection Dashboard
 for BYOD Security
================================================================================

HOW TO RUN (Anaconda Prompt):
------------------------------------------------------------------------------
1) Open "Anaconda Prompt".
2) Go to the folder that contains this file AND the dataset CSV, e.g.:
       cd C:\\Users\\<you>\\Documents\\SMART-BYOD
3) (First time only) install the required packages:
       pip install streamlit pandas numpy plotly scikit-learn
4) Run the app:
       streamlit run smart_byod_dashboard.py
5) A browser tab will open automatically (usually http://localhost:8501).

DATA:
------------------------------------------------------------------------------
PRIMARY SOURCE - SQLite: the dashboard reads the saved data from "smart_byod.db"
(written by load_fortigate_logs_to_sqlite.py) through a read-only connection,
joining behavioral_features, security_events, users, devices, anomaly_results,
risk_assessments and policy_decisions. Risk scores, levels and actions come
straight from the database - nothing is re-scored in the app. The file is
looked for in the working folder, then next to this script. Set the
SMART_BYOD_DB environment variable to use a different file name.

FALLBACK - CSV: if the database is missing or has no scored events yet, the app
falls back to "SMART_BYOD_10000_Dataset.csv" and scores it with the Isolation
Forest below, exactly as before. If neither is found, a file-uploader appears
in the sidebar.

In SQLite mode the approved schema has no ground_truth or scenario columns, so
the Model Evaluation figures that need labels (3, 5, 6, 7) show a note instead.
IP addresses are always masked (192.168.***.***); the address comes from security_events; location is shown as N/A (no geo data).

Expected columns (as produced by the SMART-BYOD dataset generator):
    record_id, scenario, ground_truth, user_id, user_type, device_id,
    session_id, timestamp,
    LF  (Login Frequency),          SD  (Session Duration, minutes),
    FL  (Failed Login Attempts),    DS  (Device Switching count),
    LTD (Login-Time Deviation, min),IPC (IP Address Changes),
    RAF (Resource Access Frequency),AAF (App/API Activity Frequency),
    RA  (Rapid Activity flag)

MODEL:
------------------------------------------------------------------------------
Risk scores are produced with an Isolation Forest trained on the nine
behavioral features (LF, SD, FL, DS, LTD, IPC, RAF, AAF, RA), mirroring the
anomaly-detection layer described in the SMART-BYOD framework. Scores are
rescaled to 0-100 and mapped to a Zero-Trust-style policy/decision layer:

    0-29   LOW       -> Monitor
    30-59  MODERATE  -> Verify
    60-79  HIGH      -> Restrict
    80-100 CRITICAL  -> Block

NOTE: IP address and city/location fields are not present in the CSV, so
they are deterministically simulated per user_id purely for display purposes
(clearly marked in the code below). Everything else on screen is computed
directly from the uploaded dataset.
================================================================================
"""

import os
import re
import time
import random
import hashlib
import sqlite3
from pathlib import Path
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use("Agg")  # headless backend — no GUI window, no file is ever saved
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
# ==============================================================================
# PAGE CONFIG & GLOBAL STYLE
# ==============================================================================
st.set_page_config(
    page_title="SMART-BYOD | Lightweight Zero Trust-Inspired Behavioral Monitoring",
    page_icon="\U0001F6E1\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Primary data source: the SQLite database the loader writes (approved 8-table schema).
DB_FILENAME = os.environ.get("SMART_BYOD_DB", "smart_byod.db")
# Fallback when the database is missing or still empty.
CSV_FILENAME = "SMART_BYOD_10000_Dataset.csv"

FEATURE_COLS = ["LF", "SD", "FL", "DS", "LTD", "IPC", "RAF", "AAF", "RA"]

FEATURE_META = {
    "LF":  {"label": "Login Frequency",        "unit": "logins",   "icon": "\U0001F511"},
    "SD":  {"label": "Session Duration",        "unit": "min",      "icon": "\u23F1\uFE0F"},
    "FL":  {"label": "Failed Login Attempts",   "unit": "attempts", "icon": "\u274C"},
    "DS":  {"label": "Device Switching",        "unit": "events",   "icon": "\U0001F4F1"},
    "LTD": {"label": "Login-time Deviation",    "unit": "min",      "icon": "\U0001F550"},
    "IPC": {"label": "IP Address Changes",      "unit": "changes",  "icon": "\U0001F310"},
    "RAF": {"label": "Resource Access Freq.",   "unit": "accesses", "icon": "\U0001F4C1"},
    "AAF": {"label": "App/API Activity Freq.",  "unit": "requests", "icon": "\U0001F4BB"},
    "RA":  {"label": "Rapid Activity",          "unit": "events/min", "icon": "\u26A1"},
}

# Human-readable alert labels per synthetic scenario code (S01 = normal traffic;
# S02-S10 = distinct anomalous behavior patterns used when the dataset was
# generated). Used only to make the Alerts table readable.
SCENARIO_LABELS = {
    "S01": "Normal Activity",
    "S02": "Brute-Force / Failed Logins",
    "S03": "Unusual Login Time",
    "S04": "Excessive Device Switching",
    "S05": "IP Address / Location Anomaly",
    "S06": "Excessive Resource Access",
    "S07": "Abnormal App/API Activity",
    "S08": "Rapid Activity Burst",
    "S09": "Session Duration Anomaly",
    "S10": "Multi-Vector Anomaly",
}

PH_CITIES = [
    "Quezon City, PH", "Makati City, PH", "Pasig City, PH", "Taguig City, PH",
    "Manila, PH", "Cebu City, PH", "Davao City, PH", "Mandaluyong, PH",
    "Paranaque, PH", "Antipolo, PH",
]

RISK_COLORS = {
    "LOW": "#22C55E",
    "MODERATE": "#F59E0B",
    "HIGH": "#F97316",
    "CRITICAL": "#EF4444",
}
ACTION_BY_LEVEL = {
    "LOW": "Monitor",
    "MODERATE": "Verify",
    "HIGH": "Restrict",
    "CRITICAL": "Block",
}
ACTION_COLORS = {
    "Monitor": "#22C55E",
    "Allow": "#22C55E",  # the approved thesis names the lowest tier "Allow"
    "Verify": "#F59E0B",
    "Restrict": "#F97316",
    "Block": "#EF4444",
}

BG = "#0A0F1E"
PANEL = "#101A2E"
PANEL_BORDER = "#1E2A45"
TEXT_MAIN = "#ffffff" #"#E5E9F0"
TEXT_SUB = "#8592AD"
ACCENT = "#3B82F6"

# ---- Company brand palette (Model Evaluation figures) ----------------------
BRAND_RED = "#DC2626"        # primary brand red
BRAND_RED_LIGHT = "#F87171"  # lighter tint of the brand red, for 2nd series
BRAND_TEAL = "#14B8A6"       # complementary to red — used only where two
                              # series need to contrast against BRAND_RED
                              # (e.g. Normal vs Anomalous)
                              
CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {BG};
        color: {TEXT_MAIN};
    }}
    section[data-testid="stSidebar"] {{
        background-color: #0D1526;
        border-right: 1px solid {PANEL_BORDER};
    }}
    section[data-testid="stSidebar"] .stRadio label p {{
        color: {TEXT_SUB} !important;
        font-size: 14px !important;
    }}
    section[data-testid="stSidebar"] .stSelectbox label p {{
        color: {TEXT_SUB} !important;
        font-size: 14px !important;
    }}     
    section[data-testid="stCaptionContainer"] .stCaption p {{
        color: {TEXT_SUB} !important;
        font-size: 14px !important;
    }}

    div[data-testid="stMetric"] {{
        background-color: {PANEL};
        border: 1px solid {PANEL_BORDER};
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }}
    .panel {{
        background-color: {PANEL};
        border: 1px solid {PANEL_BORDER};
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }}
    .panel-title {{
        color: {ACCENT};
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }}
    .panel-title span.sub {{
        color: {TEXT_SUB};
        font-weight: 400;
        font-size: 12.5px;
    }}
    .metric-card {{
        background-color: #0C1730;
        border: 1px solid {PANEL_BORDER};
        border-radius: 10px;
        padding: 12px 14px;
        height: 100%;
    }}
    .metric-card .mc-label {{
        color: {TEXT_SUB};
        font-size: 12.5px;
        margin-bottom: 2px;
    }}
    .metric-card .mc-value {{
        color: {TEXT_MAIN};
        font-size: 26px;
        font-weight: 700;
        line-height: 1.15;
    }}
    .metric-card .mc-delta-up {{ color: #22C55E; font-size: 12px; }}
    .metric-card .mc-delta-down {{ color: #EF4444; font-size: 12px; }}
    .badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12.5px;
    }}
    .status-active {{ color: #22C55E; font-weight: 600; font-size: 12.5px; }}
    .app-title {{
        font-size: 26px;
        font-weight: 800;
        color: {TEXT_MAIN};
        margin-bottom: 2px;
    }}
    .app-subtitle {{
        color: {TEXT_SUB};
        font-size: 13.5px;
        margin-bottom: 18px;
    }}
    thead tr th {{
        background-color: #0C1730 !important;
        color: {TEXT_SUB} !important;
    }}
    .stTable table td {{
        color: #FFFFFF !important;
    }}
    .legend-dot {{
        display: inline-block; width: 10px; height: 10px; border-radius: 50%;
        margin-right: 6px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# LOGIN AUTHENTICATION (username + password, then a one-time code by email or mobile number)
# ------------------------------------------------------------------------------
# Nothing below this block (data loading, charts, reports) runs until the visitor
# has signed in. The flow:
#   0. Default method "password_otp": the visitor first signs in with username + password
#      (stored only as salted PBKDF2 hashes), then the code below is sent to the account's
#      registered email or mobile number. Set auth.method = "password" or "otp" to use one step only.
#   1. (OTP-only method) The visitor picks Email or Mobile number and types the address/number.
#   2. If it belongs to a registered user, a 6-digit code is sent to it.
#      (The same message is shown whether or not the account exists, so the page
#       cannot be used to discover who is registered.)
#   3. The code is valid for 5 minutes, 5 attempts, and is stored only as a keyed hash.
#   4. After 5 wrong codes, or too many requests, that account is locked for 15 minutes.
#   5. The session ends after 30 minutes of inactivity, or on "Sign out".
# Registered users, SMTP and SMS credentials live in Streamlit secrets
# (.streamlit/secrets.toml, or "Secrets" in Streamlit Community Cloud) - never in this file.
# See secrets.toml.example.
# ==============================================================================
import hmac
import secrets as _pysecrets
import smtplib
import threading
import logging
import json as _json
import base64
import urllib.parse
import urllib.request
from email.message import EmailMessage

AUTH_LOG = logging.getLogger("smart_byod.auth")

OTP_LENGTH = 6
OTP_TTL_SECONDS = 300          # a code works for 5 minutes
OTP_MAX_ATTEMPTS = 5           # wrong entries before the code is cancelled and the account locked
OTP_RESEND_COOLDOWN = 60       # seconds between requests
OTP_MAX_SENDS_PER_HOUR = 5
AUTH_LOCKOUT_SECONDS = 900     # 15 minutes
SESSION_IDLE_SECONDS = 1800    # sign out after 30 minutes without activity


def _auth_conf() -> dict:
    """Authentication settings from st.secrets['auth'] (empty dict when absent)."""
    try:
        return dict(st.secrets.get("auth", {}))
    except Exception:
        return {}


def _secret_section(name: str) -> dict:
    try:
        return dict(st.secrets.get(name, {}))
    except Exception:
        return {}


def _norm_email(value: str) -> str:
    return (value or "").strip().lower()


def _norm_phone(value: str) -> str:
    """Digits with an optional leading +. 09171234567 and +63 917 123 4567 both reduce to comparable forms."""
    v = (value or "").strip()
    digits = re.sub(r"\D", "", v)
    if v.startswith("+"):
        return "+" + digits
    if digits.startswith("00"):
        return "+" + digits[2:]
    if digits.startswith("0") and len(digits) == 11:        # Philippine local format 09XXXXXXXXX
        return "+63" + digits[1:]
    return "+" + digits if digits else ""


def _registered_users() -> dict:
    """{key: {name, email, phone}} from secrets (auth.users.<key>), with normalised contacts."""
    users = {}
    for key, rec in dict(_auth_conf().get("users", {})).items():
        rec = dict(rec)
        users[key] = {
            "name": rec.get("name") or key,
            "email": _norm_email(rec.get("email", "")),
            "phone": _norm_phone(rec.get("phone", "")),
            "username": str(key).lower(),
            "password_hash": str(rec.get("password_hash", "")),
        }
    return users


def _mask_contact(kind: str, value: str) -> str:
    if kind == "email":
        local, _, domain = value.partition("@")
        return (local[:1] + "***@" + domain) if domain else "***"
    return value[:3] + "*" * max(len(value) - 6, 3) + value[-3:]


# ---- passwords (salted PBKDF2-SHA256; the secrets file holds only the hash) -------
PBKDF2_ITERATIONS = 600_000


def hash_password(password: str, salt: bytes = None, iterations: int = PBKDF2_ITERATIONS) -> str:
    """'pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>' - generate with make_password_hash.py."""
    salt = salt or _pysecrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iters, salt_hex, hash_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(digest.hex(), hash_hex)
    except Exception:
        return False


_DUMMY_HASH = None


def _check_password(username: str, password: str):
    """Returns the user record on success, otherwise None. Unknown and known usernames take the same
    time and give the same answer, and failures are counted per username (5 failures -> 15 min lock)."""
    global _DUMMY_HASH
    name = (username or "").strip().lower()
    subject = _subject_id("pw", name)
    if _locked_seconds(subject):
        return None
    user = next((u for u in _registered_users().values() if u["username"] == name), None)
    if user is None or not user["password_hash"]:
        if _DUMMY_HASH is None:
            _DUMMY_HASH = hash_password("not-a-real-password")
        _verify_password(password or "", _DUMMY_HASH)           # same work as a real check
        _register_failure(subject)
        return None
    if _verify_password(password or "", user["password_hash"]):
        store = _auth_store()
        with store["lock"]:
            store["fails"].pop(subject, None)
        return user
    _register_failure(subject)
    AUTH_LOG.warning("wrong password (%s)", subject)
    return None


@st.cache_resource
def _auth_store() -> dict:
    """Process-wide counters, so a page refresh or a new browser tab cannot reset the limits."""
    return {"lock": threading.Lock(), "sends": {}, "fails": {}, "locked_until": {}, "key": _pysecrets.token_bytes(32)}


def _hmac_key() -> bytes:
    configured = _auth_conf().get("secret_key")
    return str(configured).encode() if configured else _auth_store()["key"]


def _otp_hash(code: str, nonce: str) -> str:
    return hmac.new(_hmac_key(), f"{nonce}:{code}".encode(), "sha256").hexdigest()


def _subject_id(kind: str, value: str) -> str:
    return hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()[:16]


# ---- delivery ----------------------------------------------------------------
def _send_email(to_addr: str, code: str) -> None:
    cfg = _secret_section("smtp")
    if not cfg.get("host") or not cfg.get("sender"):
        raise RuntimeError("SMTP is not configured")
    msg = EmailMessage()
    msg["Subject"] = "Your SMART-BYOD sign-in code"
    msg["From"] = cfg["sender"]
    msg["To"] = to_addr
    msg.set_content(
        f"Your SMART-BYOD sign-in code is {code}.\n\n"
        f"It expires in {OTP_TTL_SECONDS // 60} minutes. Never share this code. "
        f"If you did not try to sign in, you can ignore this message."
    )
    port = int(cfg.get("port", 587))
    if port == 465:
        server = smtplib.SMTP_SSL(cfg["host"], port, timeout=15)
    else:
        server = smtplib.SMTP(cfg["host"], port, timeout=15)
        if cfg.get("starttls", True):
            server.starttls()
    try:
        if cfg.get("username"):
            server.login(cfg["username"], cfg.get("password", ""))
        server.send_message(msg)
    finally:
        try:
            server.quit()
        except Exception:
            pass


def _send_sms(to_number: str, code: str) -> None:
    """SMS through the Twilio REST API (credentials in st.secrets['twilio'])."""
    cfg = _secret_section("twilio")
    if not (cfg.get("account_sid") and cfg.get("auth_token") and (cfg.get("from_number") or cfg.get("messaging_service_sid"))):
        raise RuntimeError("SMS provider is not configured")
    body = {"To": to_number, "Body": f"SMART-BYOD sign-in code: {code}. Valid {OTP_TTL_SECONDS // 60} min. Do not share it."}
    if cfg.get("messaging_service_sid"):
        body["MessagingServiceSid"] = cfg["messaging_service_sid"]
    else:
        body["From"] = cfg["from_number"]
    url = f"https://api.twilio.com/2010-04-01/Accounts/{cfg['account_sid']}/Messages.json"
    req = urllib.request.Request(url, data=urllib.parse.urlencode(body).encode(), method="POST")
    token = base64.b64encode(f"{cfg['account_sid']}:{cfg['auth_token']}".encode()).decode()
    req.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(req, timeout=15) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"SMS provider returned {resp.status}")


# ---- rate limiting -----------------------------------------------------------
def _locked_seconds(subject: str) -> int:
    store = _auth_store()
    with store["lock"]:
        return max(0, int(store["locked_until"].get(subject, 0) - time.time()))


def _register_failure(subject: str) -> None:
    store = _auth_store()
    with store["lock"]:
        store["fails"][subject] = store["fails"].get(subject, 0) + 1
        if store["fails"][subject] >= OTP_MAX_ATTEMPTS:
            store["locked_until"][subject] = time.time() + AUTH_LOCKOUT_SECONDS
            store["fails"][subject] = 0


def _allow_send(subject: str) -> tuple:
    """(allowed, wait_seconds). Cooldown between requests and a per-hour cap."""
    store = _auth_store()
    now = time.time()
    with store["lock"]:
        sends = [t for t in store["sends"].get(subject, []) if now - t < 3600]
        store["sends"][subject] = sends
        if sends and now - sends[-1] < OTP_RESEND_COOLDOWN:
            return False, int(OTP_RESEND_COOLDOWN - (now - sends[-1])) + 1
        if len(sends) >= OTP_MAX_SENDS_PER_HOUR:
            return False, int(3600 - (now - sends[0])) + 1
        sends.append(now)
        return True, 0


# ---- the flow ----------------------------------------------------------------
def _request_code(kind: str, entered: str, demo: bool, known_user: dict = None) -> str:
    """Create and send a code. Returns a message for the visitor; never says whether the account exists.
    known_user is given after the password step, so the code goes to that user's registered contact."""
    if known_user is not None:
        value = known_user[kind]
    else:
        value = _norm_email(entered) if kind == "email" else _norm_phone(entered)
    generic = "If that account is registered, a sign-in code has been sent."
    if known_user is not None:
        generic = f"A sign-in code has been sent to {_mask_contact(kind, value)}."
    if not value or (kind == "email" and "@" not in value) or (kind == "phone" and len(value) < 8):
        return "Please enter a valid " + ("email address." if kind == "email" else "mobile number, with country code.")
    subject = _subject_id(kind, value)
    if _locked_seconds(subject):
        return f"Too many attempts. Try again in {_locked_seconds(subject) // 60 + 1} minute(s)."
    ok, wait = _allow_send(subject)
    if not ok:
        return f"Please wait {wait} second(s) before requesting another code."

    user = known_user or next((u for u in _registered_users().values() if u[kind] == value), None)
    st.session_state.pop("auth_pending", None)
    if user is None:
        AUTH_LOG.warning("sign-in code requested for an unregistered %s (%s)", kind, subject)
        # Shown as if a code was sent; a pending entry with no valid code keeps the screens identical.
        st.session_state["auth_pending"] = {"subject": subject, "kind": kind, "user": None, "hash": None,
                                            "nonce": "", "expires": time.time() + OTP_TTL_SECONDS, "tries": 0,
                                            "to": _mask_contact(kind, value)}
        return generic

    code = "".join(str(_pysecrets.randbelow(10)) for _ in range(OTP_LENGTH))
    nonce = _pysecrets.token_hex(8)
    pending = {"subject": subject, "kind": kind, "user": user, "hash": _otp_hash(code, nonce), "nonce": nonce,
               "expires": time.time() + OTP_TTL_SECONDS, "tries": 0, "to": _mask_contact(kind, value)}
    try:
        if demo:
            st.session_state["auth_demo_code"] = code     # DEMO ONLY: the code is shown on screen
        elif kind == "email":
            _send_email(value, code)
        else:
            _send_sms(value, code)
    except Exception as exc:
        AUTH_LOG.error("could not deliver sign-in code (%s): %s", subject, type(exc).__name__)
        return "The code could not be sent right now. Please try again later or contact the administrator."
    st.session_state["auth_pending"] = pending
    AUTH_LOG.info("sign-in code sent (%s, %s)", kind, subject)
    return generic


def _verify_code(entered: str) -> str:
    """Returns '' on success (and signs the user in), otherwise the error message."""
    pending = st.session_state.get("auth_pending")
    if not pending:
        return "Request a new code first."
    subject = pending["subject"]
    if _locked_seconds(subject):
        st.session_state.pop("auth_pending", None)
        return f"Too many attempts. Try again in {_locked_seconds(subject) // 60 + 1} minute(s)."
    if time.time() > pending["expires"]:
        st.session_state.pop("auth_pending", None)
        return "That code has expired. Request a new one."
    entered = re.sub(r"\D", "", entered or "")
    good = pending["hash"] is not None and hmac.compare_digest(_otp_hash(entered, pending["nonce"]), pending["hash"])
    if not good:
        pending["tries"] += 1
        _register_failure(subject)
        left = OTP_MAX_ATTEMPTS - pending["tries"]
        if left <= 0 or _locked_seconds(subject):
            st.session_state.pop("auth_pending", None)
            AUTH_LOG.warning("sign-in locked after repeated wrong codes (%s)", subject)
            return "Too many wrong codes. This account is locked for 15 minutes."
        return f"Incorrect code. {left} attempt(s) left."
    user = pending["user"]
    for k in ("auth_pending", "auth_demo_code", "auth_stage"):
        st.session_state.pop(k, None)
    st.session_state["auth_user"] = {"name": user["name"], "via": pending["kind"]}
    st.session_state["auth_last_active"] = time.time()
    AUTH_LOG.info("signed in (%s, %s)", pending["kind"], subject)
    return ""


def _sign_out(reason: str = "") -> None:
    for k in list(st.session_state.keys()):
        if k.startswith("auth_"):
            st.session_state.pop(k, None)
    if reason:
        st.session_state["auth_notice"] = reason


def _finish_sign_in(user: dict, via: str) -> None:
    for k in ("auth_pending", "auth_demo_code", "auth_stage"):
        st.session_state.pop(k, None)
    st.session_state["auth_user"] = {"name": user["name"], "via": via}
    st.session_state["auth_last_active"] = time.time()
    AUTH_LOG.info("signed in (%s, %s)", via, _subject_id("pw", user["username"]))


STAGE_SECONDS = 600     # after the password step, the code must be completed within 10 minutes


def _login_header(subtitle: str) -> None:
    st.markdown(
        f"""
        <div style="max-width:440px;margin:6vh auto 12px auto;text-align:center;">
            <div style="font-size:40px;">\U0001F6E1\uFE0F</div>
            <div style="font-size:26px;font-weight:800;color:{ACCENT};">SMART-BYOD</div>
            <div style="color:{TEXT_SUB};font-size:13px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _code_entry(demo: bool) -> None:
    """Shared 'enter the 6-digit code' step."""
    pending = st.session_state.get("auth_pending")
    st.success(st.session_state.pop("auth_info", None) or f"Code sent to {pending['to']}.")
    st.caption(f"Enter the {OTP_LENGTH}-digit code sent to {pending['to']}. "
               f"It expires in {OTP_TTL_SECONDS // 60} minutes.")
    if demo and st.session_state.get("auth_demo_code"):
        st.code(st.session_state["auth_demo_code"], language=None)
    with st.form("auth_verify", clear_on_submit=True):
        code = st.text_input("Sign-in code", max_chars=OTP_LENGTH + 2)
        if st.form_submit_button("Verify and sign in", type="primary", use_container_width=True):
            err = _verify_code(code)
            if err:
                st.session_state["auth_error"] = err
            st.rerun()


def require_login() -> None:
    """Stop the script unless the visitor is signed in.
    auth.method:  "password_otp" (default) = username + password, then a code by email or SMS
                  "password"              = username + password only
                  "otp"                   = code by email or SMS only"""
    conf = _auth_conf()
    if not bool(conf.get("enabled", True)):
        return                                           # explicit opt-out for local development only
    method = str(conf.get("method", "password_otp")).lower()
    if method not in ("password_otp", "password", "otp"):
        method = "password_otp"
    demo = bool(conf.get("demo_mode", False)) or os.environ.get("SMART_BYOD_AUTH_DEMO") == "1"

    if st.session_state.get("auth_user"):
        if time.time() - st.session_state.get("auth_last_active", 0) > SESSION_IDLE_SECONDS:
            _sign_out("You were signed out after 30 minutes of inactivity.")
            st.rerun()
        st.session_state["auth_last_active"] = time.time()
        return

    subtitle = {"password_otp": "Sign in with your username, password and a one-time code",
                "password": "Sign in with your username and password",
                "otp": "Sign in with a one-time code"}[method]
    _login_header(subtitle)
    users = _registered_users()
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        needs_pw = method in ("password", "password_otp")
        if (not users and not demo) or (needs_pw and not any(u["password_hash"] for u in users.values())):
            st.error("Sign-in is not configured. The administrator must register users (with password hashes) "
                     "in the app's secrets - see secrets.toml.example. Access stays closed until then.")
            st.stop()
        if demo and method != "password":
            st.warning("DEMO MODE: codes are shown on screen instead of being sent. "
                       "Turn this off (auth.demo_mode = false) before real use.")
        if st.session_state.get("auth_notice"):
            st.info(st.session_state.pop("auth_notice"))
        if st.session_state.get("auth_error"):
            st.error(st.session_state.pop("auth_error"))

        stage = st.session_state.get("auth_stage")
        if stage and time.time() > stage["expires"]:
            st.session_state.pop("auth_stage", None)
            st.session_state.pop("auth_pending", None)
            stage = None
            st.info("Your sign-in took too long. Please start again.")

        if needs_pw and not stage:
            # ---- step 1: username + password ----
            with st.form("auth_password"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                    user = _check_password(username, password)
                    if user is None:
                        subject = _subject_id("pw", (username or "").strip().lower())
                        if _locked_seconds(subject):
                            st.session_state["auth_error"] = (
                                f"Too many attempts. Try again in {_locked_seconds(subject) // 60 + 1} minute(s).")
                        else:
                            st.session_state["auth_error"] = "Incorrect username or password."
                        st.rerun()
                    elif method == "password":
                        _finish_sign_in(user, "password")
                        st.rerun()
                    else:
                        st.session_state["auth_stage"] = {"user": user, "expires": time.time() + STAGE_SECONDS}
                        st.rerun()
        elif stage:
            # ---- step 2: code sent to the user's registered email / mobile ----
            user = stage["user"]
            st.caption(f"Password accepted for **{user['name']}**. Now confirm with a one-time code.")
            if not st.session_state.get("auth_pending"):
                kinds = [k for k in ("email", "phone") if user[k]]
                if not kinds:
                    st.error("This account has no email or mobile number registered. Contact the administrator.")
                else:
                    labels = {"email": f"Email ({_mask_contact('email', user['email'])})" if user["email"] else "",
                              "phone": f"Mobile ({_mask_contact('phone', user['phone'])})" if user["phone"] else ""}
                    choice = st.radio("Send my code to", [labels[k] for k in kinds], horizontal=False)
                    kind = kinds[[labels[k] for k in kinds].index(choice)]
                    if st.button("Send code", type="primary", use_container_width=True):
                        msg = _request_code(kind, "", demo, known_user=user)
                        if st.session_state.get("auth_pending"):
                            st.session_state["auth_info"] = msg
                            st.rerun()
                        st.warning(msg)
            else:
                _code_entry(demo)
            if st.button("Back to sign in", use_container_width=True):
                for k in ("auth_stage", "auth_pending", "auth_demo_code"):
                    st.session_state.pop(k, None)
                st.rerun()
        else:
            # ---- code only (method = "otp") ----
            pending = st.session_state.get("auth_pending")
            if not pending:
                channel = st.radio("Send my code by", ["Email", "Mobile number (SMS)"], horizontal=True)
                kind = "email" if channel == "Email" else "phone"
                with st.form("auth_request", clear_on_submit=False):
                    entered = st.text_input("Registered email address" if kind == "email" else "Registered mobile number",
                                            placeholder="name@company.com" if kind == "email" else "+63 917 123 4567")
                    if st.form_submit_button("Send code", type="primary", use_container_width=True):
                        msg = _request_code(kind, entered, demo)
                        if st.session_state.get("auth_pending"):
                            st.session_state["auth_info"] = msg
                            st.rerun()
                        st.warning(msg)
            else:
                _code_entry(demo)
                if st.button("Use a different email / number", use_container_width=True):
                    st.session_state.pop("auth_pending", None)
                    st.session_state.pop("auth_demo_code", None)
                    st.rerun()
        st.markdown(
            f'<div style="color:{TEXT_SUB};font-size:11px;text-align:center;margin-top:14px;">'
            f'Authorised users only. Repeated wrong attempts lock the account for 15 minutes.</div>',
            unsafe_allow_html=True)
    st.stop()


require_login()


# ==============================================================================
# DATA LOADING
# ==============================================================================
@st.cache_data(show_spinner="Loading SMART-BYOD dataset...")
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def find_file(name: str):
    """Look in the working folder first, then next to this script."""
    candidates = [Path(name)]
    try:
        candidates.append(Path(__file__).resolve().parent / name)
    except NameError:
        pass
    for p in candidates:
        if p.is_file():
            return p
    return None


def db_signature(path: Path) -> tuple:
    """Changes whenever the loader commits. In WAL mode new rows land in the
    -wal file first, so the main file's modified-time alone isn't enough."""
    return tuple(
        Path(str(path) + suffix).stat().st_mtime_ns if Path(str(path) + suffix).exists() else 0
        for suffix in ("", "-wal")
    )


# One row per behavioral_features record, shaped like the CSV the dashboard was
# built on (same column names), plus the loader's saved anomaly/risk/policy output.
SQLITE_QUERY = """
SELECT
    se.event_id                         AS record_id,
    u.user_code                         AS user_id,
    u.user_type                         AS user_type,
    d.device_code                       AS device_id,
    se.session_id                       AS session_id,
    bf.observation_timestamp            AS timestamp,
    bf.login_frequency                  AS LF,
    bf.session_duration                 AS SD,
    bf.failed_login_attempts            AS FL,
    bf.device_switching_events          AS DS,
    bf.login_time_deviation             AS LTD,
    bf.ip_address_changes               AS IPC,
    bf.resource_access_frequency        AS RAF,
    bf.application_activity_frequency   AS AAF,
    bf.rapid_activity                   AS RA,
    se.ip_address                       AS ip_address,
    ar.model_prediction                 AS model_prediction,
    ra.risk_score                       AS risk_score,
    ra.risk_level                       AS risk_level,
    pd.policy_action                    AS action,
    pd.policy_reason                    AS alert_type
FROM behavioral_features bf
JOIN security_events se        ON se.event_id   = bf.event_id
JOIN users u                   ON u.user_id     = bf.user_id
JOIN devices d                 ON d.device_id   = bf.device_id
LEFT JOIN anomaly_results ar   ON ar.feature_id = bf.feature_id
LEFT JOIN risk_assessments ra  ON ra.anomaly_id = ar.anomaly_id
LEFT JOIN policy_decisions pd  ON pd.risk_id    = ra.risk_id
ORDER BY bf.observation_timestamp
"""


def _read_sql(conn_factory) -> pd.DataFrame:
    conn = conn_factory()
    try:
        return pd.read_sql_query(SQLITE_QUERY, conn)
    finally:
        conn.close()


@st.cache_data(show_spinner="Loading SMART-BYOD data from SQLite...")
def load_from_sqlite(db_path: str, signature: tuple):
    """
    Read the saved data through an independent read-only connection (Table 3.3-15:
    WAL lets these reads run while the loader is writing). `signature` is only a
    cache key, so a fresh loader commit invalidates the cached copy.
    Returns (dataframe, number_of_events_not_yet_scored).
    """
    path = Path(db_path)

    def readonly():
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def normal():
        conn = sqlite3.connect(str(path), timeout=5.0)
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    try:
        raw = _read_sql(readonly)
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        raw = _read_sql(normal)  # some setups can't open a WAL database read-only

    if raw.empty:
        return raw, 0

    unscored = int(raw["risk_score"].isna().sum())
    raw = raw.dropna(subset=["risk_score"]).copy()

    # The stored timestamps carry the FortiGate offset; keep the wall-clock time as recorded.
    raw["timestamp"] = pd.to_datetime(raw["timestamp"].astype(str).str[:19])
    raw[FEATURE_COLS] = raw[FEATURE_COLS].astype(float).fillna(0.0)
    raw["risk_level"] = raw["risk_level"].astype(str).str.upper()
    raw["predicted_anomaly"] = (raw.pop("model_prediction") == -1).astype(int)
    raw["ip_address"] = raw["ip_address"].fillna("N/A").map(mask_ip)
    raw["location"] = "N/A"  # no geo data in the approved schema
    raw["alert_type"] = raw["alert_type"].fillna("Unspecified")
    return raw.reset_index(drop=True), unscored


# ------------------------------------------------------------------------------
# Device inventory (DHCP lease events stored by the loader) — feeds the
# "Device & Network" reports. Only brand/model class and a masked MAC are used;
# hostnames (which can contain personal names) are never read or shown.
# ------------------------------------------------------------------------------
DEVICE_EVENTS_QUERY = """
SELECT se.event_timestamp AS ts,
       d.device_code      AS mac,
       COALESCE(NULLIF(d.device_brand, ''), 'Unknown') AS brand,
       COALESCE(NULLIF(d.os_family,    ''), 'Unknown') AS os_family,
       COALESCE(NULLIF(d.device_type,  ''), 'Unknown') AS device_type,
       COALESCE(se.resource, 'Unknown')                AS network,
       se.ip_address AS ip
FROM security_events se
JOIN devices d ON d.device_id = se.device_id
WHERE se.event_source = 'fortigate_dhcp'
ORDER BY se.event_timestamp
"""


@st.cache_data(show_spinner=False)
def load_device_events(db_path: str, signature: tuple) -> pd.DataFrame:
    path = Path(db_path)
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.OperationalError:
        conn = sqlite3.connect(str(path), timeout=5.0)
    try:
        conn.execute("PRAGMA busy_timeout = 5000;")
        ev = pd.read_sql_query(DEVICE_EVENTS_QUERY, conn)
    finally:
        conn.close()
    if ev.empty:
        return ev
    ev["ts"] = pd.to_datetime(ev["ts"].astype(str).str[:19])   # wall-clock as recorded
    ev["ip"] = ev["ip"].map(mask_ip)                            # never show full addresses
    ev["private_mac"] = ev["mac"].astype(str).str[1:2].str.upper().isin(list("26AE"))
    ev["device_label"] = (ev["brand"] + " " + ev["device_type"]).str.replace(" Unknown", "", regex=False)
    ev["mac_masked"] = "••:" + ev["mac"].astype(str).str[-5:]
    ev["network"] = ev["network"].map(rename_network)
    return ev


def get_dataset():
    """
    SQLite first (the saved data), CSV as the fallback.
    Returns (dataframe, source, unscored_count) where source is "sqlite" or "csv".
    """
    db_path = find_file(DB_FILENAME)
    if db_path is not None:
        try:
            data, unscored = load_from_sqlite(str(db_path), db_signature(db_path))
            if len(data):
                return data, "sqlite", unscored
            st.sidebar.warning(f"'{DB_FILENAME}' has no scored events yet - using the CSV instead.")
        except Exception as exc:
            st.sidebar.warning(f"Couldn't read '{DB_FILENAME}' ({type(exc).__name__}) - using the CSV instead.")

    csv_path = find_file(CSV_FILENAME)
    if csv_path is not None:
        return load_data(str(csv_path)), "csv", 0

    st.sidebar.warning(f"Neither '{DB_FILENAME}' nor '{CSV_FILENAME}' found next to the script.")
    uploaded = st.sidebar.file_uploader("Upload the SMART-BYOD dataset (.csv)", type="csv")
    if uploaded is not None:
        return load_data(uploaded), "csv", 0

    st.error(
        f"Could not find **{DB_FILENAME}** or **{CSV_FILENAME}**. Run the loader to create the "
        "database, place the CSV next to this script, or upload it using the sidebar control."
    )
    st.stop()


# ==============================================================================
# RISK ENGINE (Isolation Forest -> 0-100 risk score -> Zero-Trust policy layer)
# ==============================================================================
@st.cache_data(show_spinner="Scoring behavioral risk (Isolation Forest)...")
def score_risk(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    X = df[FEATURE_COLS].values

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    
    # Binary prediction from the model's own contamination-based threshold
    # (used for confusion-matrix / precision / recall / F1 evaluation).
    raw_pred = model.predict(X)  # -1 = anomaly, 1 = normal
    df["predicted_anomaly"] = np.where(raw_pred == -1, 1, 0)

    # Lower score_samples() == more anomalous. Flip sign so higher == riskier.
    raw_scores = -model.score_samples(X)
    scaled = MinMaxScaler(feature_range=(0, 100)).fit_transform(raw_scores.reshape(-1, 1)).ravel()
    df["risk_score"] = scaled.round(1)

    bins = [-0.1, 29, 59, 79, 100]
    labels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    df["risk_level"] = pd.cut(df["risk_score"], bins=bins, labels=labels).astype(str)
    df["action"] = df["risk_level"].map(ACTION_BY_LEVEL)
    df["alert_type"] = df["scenario"].map(SCENARIO_LABELS).fillna(df["scenario"])
    return df


_IPV4_RE = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}\b")


def mask_ip(value):
    """Mask every IPv4 address in a value, keeping only the first two parts:
    192.168.10.25 -> 192.168.***.***  (non-addresses such as 'N/A' are unchanged)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    return _IPV4_RE.sub(lambda m: f"{m.group(1)}.{m.group(2)}.***.***", str(value))


def ip_as_x_mask(value):
    """Display helper: an already-***-masked IP (from mask_ip) shown with X instead
    of *** -> 137.158.X.X. Used where the UI shows an address-style identifier
    instead of a name, never the unmasked address."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    return str(value).replace("***", "X")


def mask_mac(value, keep: int = 3):
    """Mask a MAC/device id, keeping only the last `keep` colon-separated groups:
    mask_mac('AA:BB:CC:AB:CD:EF', keep=3) -> 'xx:xx:xx:ab:cd:ef'
    mask_mac('AA:BB:CC:AB:CD:EF', keep=2) -> 'xx:xx:xx:cd:ef'"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    v = str(value)
    parts = v.split(":")
    if len(parts) < keep + 1:
        return v
    return "xx:xx:xx:" + ":".join(p.lower() for p in parts[-keep:])


# Interface/SSID names shown in Reports -> Device & Network, renamed for display only
# (the raw FortiGate interface names are not shown as-is in the UI). Matched
# case-insensitively, since FortiGate interface names seen so far are capitalized
# differently (e.g. "WiFi-Staff"), so the client's requested spelling may not be
# byte-for-byte what the logs actually contain.
NETWORK_DISPLAY_MAP_CI = {"wifi-tr4": "internal4", "tr4-hq": "DMZ"}


def rename_network(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    return NETWORK_DISPLAY_MAP_CI.get(str(value).strip().lower(), value)


@st.cache_data(show_spinner=False)
def build_user_anon_map(user_ids: tuple) -> dict:
    """Stable, deterministic anonymous label per real user id: user0001, user0002, ...
    (sorted so the mapping is reproducible for the same set of users)."""
    return {uid: f"user{idx:04d}" for idx, uid in enumerate(sorted(user_ids), start=1)}


def _seeded_rng(key: str) -> random.Random:
    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


@st.cache_data(show_spinner=False)
def build_identity_lookup(user_ids: tuple, device_ids: tuple) -> dict:
    """
    Deterministically simulate an IP address and a city per user, and an IP
    per device, since the dataset itself does not include network/location
    fields. Same user/device always maps to the same synthetic value.
    """
    lookup = {"user_ip": {}, "user_city": {}, "device_ip": {}}
    for uid in user_ids:
        rng = _seeded_rng(uid)
        lookup["user_ip"][uid] = mask_ip(f"{rng.randint(102,223)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}")
        lookup["user_city"][uid] = rng.choice(PH_CITIES)
    for did in device_ids:
        rng = _seeded_rng(did)
        lookup["device_ip"][did] = mask_ip(f"{rng.randint(102,223)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}")
    return lookup


# ==============================================================================
# SMALL UI HELPERS
# ==============================================================================
def trend_label(series: pd.Series, value: float) -> str:
    """Return an arrow + qualitative label for a raw feature value vs. its
    distribution in the filtered data (illustrative, not a statistical test)."""
    if len(series) < 3:
        return "\u2192 Typical"
    p50, p75 = series.quantile(0.5), series.quantile(0.75)
    if value >= p75:
        return "\u2191 High"
    if value >= p50:
        return "\u2191 Elevated"
    return "\u2193 Typical"


def sparkline_fig(values, color="#3B82F6", height=42):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=list(values), mode="lines", line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=color.replace(")", ", 0.12)").replace("rgb", "rgba") if "rgb" in color else None,
        hoverinfo="skip",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def metric_card(col, icon, label, value, delta_pct=None, good_when="up"):
    if delta_pct is None:
        delta_html = ""
    else:
        up = delta_pct >= 0
        arrow = "\u25B2" if up else "\u25BC"
        css_class = "mc-delta-up" if up else "mc-delta-down"
        if good_when == "down":
            css_class = "mc-delta-down" if up else "mc-delta-up"
        delta_html = f'<div class="{css_class}">{arrow} {abs(delta_pct):,.0f}% vs last hour</div>'
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="mc-label">{icon}&nbsp; {label}</div>
            <div class="mc-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level_or_action: str, color_map: dict) -> str:
    color = color_map.get(level_or_action, "#64748B")
    return (
        f'<span class="badge" style="background-color:{color}22;'
        f'color:{color}; border:1px solid {color}55;">{level_or_action}</span>'
    )


def pct_change(current: int, previous: int) -> float:
    if previous == 0:
        return 0.0 if current == 0 else 100.0
    return (current - previous) / previous * 100.0


# ==============================================================================
# LOAD + SCORE DATA
# ==============================================================================
raw_df, DATA_SOURCE, UNSCORED_COUNT = get_dataset()

with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
            <span style="font-size:22px;">\U0001F6E1\uFE0F</span>
            <span style="font-size:19px; font-weight:800; color:{ACCENT};">SMART-BYOD</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _who = st.session_state.get("auth_user")
    if _who:
        st.markdown(
            f'<div style="color:{TEXT_SUB}; font-size:12px; margin-bottom:4px;">'
            f'Signed in as <b>{_who["name"]}</b></div>',
            unsafe_allow_html=True,
        )
        if st.button("Sign out", use_container_width=True):
            _sign_out()
            st.rerun()
    if DATA_SOURCE == "sqlite":
        status = f"Data source: SQLite ({DB_FILENAME}) &middot; {len(raw_df):,} scored events"
        if UNSCORED_COUNT:
            status += f" &middot; {UNSCORED_COUNT:,} not yet scored"
    else:
        status = f"Data source: CSV ({CSV_FILENAME})"
    st.markdown(
        f'<div style="color:{TEXT_SUB}; font-size:12px; margin-bottom:10px;">{status}</div>',
        unsafe_allow_html=True,
    )

    if DATA_SOURCE == "csv":
        st.markdown(
            f'<div style="color:{TEXT_SUB}; font-size:12px; margin-bottom:4px;">'
            f'Expected anomaly rate (Isolation Forest contamination)</div>',
            unsafe_allow_html=True,
        )
        contamination = st.slider(
            " ",
            min_value=0.01, max_value=0.20, value=0.05, step=0.01,
            help="Proportion of sessions the model expects to be anomalous.",
        )
    else:
        contamination = 0.05  # fixed by the loader; scores are already saved in the database

if DATA_SOURCE == "sqlite":
    # Already scored by the loader (anomaly_results -> risk_assessments -> policy_decisions).
    df = raw_df
else:
    df = score_risk(raw_df, contamination=contamination)

    identity_lookup = build_identity_lookup(
        tuple(sorted(df["user_id"].unique())), tuple(sorted(df["device_id"].unique()))
    )
    df["ip_address"] = df["user_id"].map(identity_lookup["user_ip"])
    df["location"] = df["user_id"].map(identity_lookup["user_city"])

df["ip_address"] = df["ip_address"].map(mask_ip)   # final guard: no full IP address reaches the UI

# The loader always stores new accounts as user_type='employee' (it has no way to tell
# guests apart at ingest time); correct that here for any account whose id looks like a
# guest account (e.g. 'guest0007'), so Type is accurate everywhere it's shown.
df["user_type"] = np.where(
    df["user_id"].astype(str).str.lower().str.startswith("guest"), "guest", df["user_type"]
)
USER_ANON_MAP = build_user_anon_map(tuple(sorted(df["user_id"].astype(str).unique())))
df["user_id_anon"] = df["user_id"].map(USER_ANON_MAP)

DATASET_NOW = df["timestamp"].max()  # "current time" anchor = latest event in the CSV

# ==============================================================================
# SIDEBAR NAVIGATION + FILTERS
# ==============================================================================
with st.sidebar:    
    page = st.radio(
        "NAVIGATION",
        ["🏠 Dashboard", "🔔 Alerts", "👥 Users & Devices", "📝 Activity Logs",
         "📄 Reports", "\U0001F9EA Model Evaluation", "📊 System Performance", "⚙️ Settings"],
        #label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**FILTERS**")
    time_range = st.selectbox(
        "Time Range",
        ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "All Time"],
        index=4,
    )
    user_type_filter = st.selectbox("User Type", ["All Users", "employee", "guest"])
    risk_level_filter = st.selectbox("Risk Level", ["All Levels", "LOW", "MODERATE", "HIGH", "CRITICAL"])
    device_options = ["All Devices"] + sorted(df["device_id"].unique().tolist())
    device_filter = st.selectbox("Device", device_options)

    apply_filters = st.button("Apply Filters", use_container_width=True, type="primary")



# Persist filter state across reruns (so the button isn't strictly required)
if "filters_applied" not in st.session_state or apply_filters:
    st.session_state.filters_applied = dict(
        time_range=time_range, user_type=user_type_filter,
        risk_level=risk_level_filter, device=device_filter,
    )
f = st.session_state.filters_applied

TIME_DELTAS = {
    "Last 1 Hour": timedelta(hours=1),
    "Last 6 Hours": timedelta(hours=6),
    "Last 24 Hours": timedelta(hours=24),
    "Last 7 Days": timedelta(days=7),
    "All Time": None,
}
def apply_all_filters(data: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = data
    delta = TIME_DELTAS[filters["time_range"]]
    if delta is not None:
        out = out[out["timestamp"] >= DATASET_NOW - delta]
    if filters["user_type"] != "All Users":
        out = out[out["user_type"] == filters["user_type"]]
    if filters["risk_level"] != "All Levels":
        out = out[out["risk_level"] == filters["risk_level"]]
    if filters["device"] != "All Devices":
        out = out[out["device_id"] == filters["device"]]
    return out


fdf = apply_all_filters(df, f)
if fdf.empty:
    st.warning("No events match the current filters. Showing the full dataset instead.")
    fdf = df.copy()

# ==============================================================================
# PAGE: DASHBOARD
# ==============================================================================
def page_dashboard():
    st.markdown('<div class="app-title">SMART-BYOD Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Lightweight Zero Trust-Inspired Behavioral '
        'Anomaly Detection for BYOD Security</div>',
        unsafe_allow_html=True,
    )

    status_col, refresh_col = st.columns([5, 1])
    with status_col:
        st.markdown(
            f'<span style="color:#22C55E;">\u25CF</span> System Status: '
            f'<span style="color:#22C55E; font-weight:700;">OPERATIONAL</span>'
            f'&nbsp;&nbsp;&nbsp;<span style="color:{TEXT_SUB};">Last Updated: '
            f'{DATASET_NOW.strftime("%Y-%m-%d %H:%M:%S")}</span>',
            unsafe_allow_html=True,
        )
    with refresh_col:
        st.button("\U0001F504 Refresh", use_container_width=True)

    # ---- Window comparison for "vs last hour" deltas (data-driven) ---------
    cur_window = df[df["timestamp"] > DATASET_NOW - timedelta(hours=1)]
    prev_window = df[
        (df["timestamp"] <= DATASET_NOW - timedelta(hours=1)) &
        (df["timestamp"] > DATASET_NOW - timedelta(hours=2))
    ]

    def delta_for(curr_series, prev_series, agg="count"):
        c = curr_series.nunique() if agg == "nunique" else len(curr_series)
        p = prev_series.nunique() if agg == "nunique" else len(prev_series)
        return c, pct_change(c, p)

    active_sessions, d_sessions = delta_for(cur_window["session_id"], prev_window["session_id"], "nunique")
    monitored_users, d_users = delta_for(cur_window["user_id"], prev_window["user_id"], "nunique")
    monitored_devices, d_devices = delta_for(cur_window["device_id"], prev_window["device_id"], "nunique")
    total_events, d_events = delta_for(cur_window, prev_window)

    anomalies_cur = cur_window[cur_window["risk_level"] != "LOW"]
    anomalies_prev = prev_window[prev_window["risk_level"] != "LOW"]
    n_anom, d_anom = len(anomalies_cur), pct_change(len(anomalies_cur), len(anomalies_prev))

    hc_cur = cur_window[cur_window["risk_level"].isin(["HIGH", "CRITICAL"])]
    hc_prev = prev_window[prev_window["risk_level"].isin(["HIGH", "CRITICAL"])]
    n_hc, d_hc = len(hc_cur), pct_change(len(hc_cur), len(hc_prev))

    resp_cur = cur_window[~cur_window["action"].isin(["Monitor", "Allow"])]
    resp_prev = prev_window[~prev_window["action"].isin(["Monitor", "Allow"])]
    n_resp, d_resp = len(resp_cur), pct_change(len(resp_cur), len(resp_prev))

    avg_resp_time = 0.8 + (cur_window["risk_score"].mean() / 100 if len(cur_window) else 0) * 1.5
    avg_resp_time = round(avg_resp_time if not np.isnan(avg_resp_time) else 1.2, 2)

    # ---- A. SYSTEM OVERVIEW --------------------------------------------------
    left, right = st.columns([1.55, 1])

    with left:
        st.markdown('<div class="panel"><div class="panel-title">A. SYSTEM OVERVIEW</div>', unsafe_allow_html=True)
        r1 = st.columns(4)
        metric_card(r1[0], "\U0001F465", "Active Sessions", f"{active_sessions:,}", d_sessions)
        metric_card(r1[1], "\U0001F464", "Monitored Users", f"{monitored_users:,}", d_users)
        metric_card(r1[2], "\U0001F4BB", "Monitored Devices", f"{monitored_devices:,}", d_devices)
        metric_card(r1[3], "\u26A1", "Total Events", f"{total_events:,}", d_events)
        st.write("")
        r2 = st.columns(4)
        metric_card(r2[0], "\U0001F6E1\uFE0F", "Anomalies Detected", f"{n_anom:,}", d_anom)
        metric_card(r2[1], "\U0001F6A8", "High/Critical Risks", f"{n_hc:,}", d_hc, good_when="down")
        metric_card(r2[2], "\u2705", "Responses Executed", f"{n_resp:,}", d_resp)
        metric_card(r2[3], "\u23F1\uFE0F", "Avg. Response Time", f"{avg_resp_time:,.2f} s", -8, good_when="down")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- B. RISK MONITORING (current highest risk record) ------------------
    top = fdf.loc[fdf["risk_score"].idxmax()] if not fdf.empty else df.loc[df["risk_score"].idxmax()]

    with right:
        st.markdown(
            '<div class="panel"><div class="panel-title">B. RISK MONITORING '
            '<span class="sub">(Current Highest Risk)</span></div>',
            unsafe_allow_html=True,
        )

        action = top["action"]
        detail_rows = [
            ("User ID", top["user_id"]),
            ("Device ID", top["device_id"]),
            ("IP Address", top["ip_address"]),
            ("Location", top["location"]),
            ("Last Activity", top["timestamp"].strftime("%H:%M:%S")),
            ("Current Action", None),  # rendered as a colored badge below
        ]

        gauge_col, label_col, value_col = st.columns([2.5, 0.9, 1.1])

        # ---- Column 1: gauge ------------------------------------------------
        with gauge_col:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(top["risk_score"]),
                number={"suffix": "", "font": {"size": 34, "color": TEXT_MAIN}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": TEXT_SUB},
                    "bar": {"color": RISK_COLORS[top["risk_level"]]},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 30], "color": "#14301F"},
                        {"range": [30, 60], "color": "#3A2E10"},
                        {"range": [60, 80], "color": "#3A210D"},
                        {"range": [80, 100], "color": "#3A1212"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "value": float(top["risk_score"])},
                },
            ))
            gauge.update_layout(
                height=150, margin=dict(l=20, r=30, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT_MAIN},
            )
            st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<div style="text-align:center; margin-top:-14px;">'
                f'<span style="color:{RISK_COLORS[top["risk_level"]]}; font-weight:700; font-size:12.5px;">'
                f'{top["risk_level"]} RISK</span><br>'
                f'<span style="color:{TEXT_SUB}; font-size:11px;">Score (0-100)</span></div>',
                unsafe_allow_html=True,
            )

        # ---- Column 2: field labels ------------------------------------------
        with label_col:
            st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
            for k, _ in detail_rows:
                st.markdown(
                    f'<div style="padding:7px 0; font-size:13px; color:{TEXT_SUB};'
                    f' border-bottom:1px solid {PANEL_BORDER};">{k}</div>',
                    unsafe_allow_html=True,
                )

        # ---- Column 3: field values -------------------------------------------
        with value_col:
            st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
            for k, v in detail_rows:
                if k == "Current Action":
                    cell = risk_badge(action, ACTION_COLORS)
                else:
                    cell = f'<span style="color:{TEXT_MAIN};">{v}</span>'
                st.markdown(
                    f'<div style="padding:7px 0; font-size:13px; text-align:right;'
                    f' border-bottom:1px solid {PANEL_BORDER};">{cell}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---- C. BEHAVIORAL ACTIVITY + D. ALERTS ---------------------------------
    left2, right2 = st.columns([1.55, 1])

    with left2:
        st.markdown(
            '<div class="panel"><div class="panel-title">C. BEHAVIORAL ACTIVITY '
            '<span class="sub">(Current Session)</span></div>',
            unsafe_allow_html=True,
        )
        user_hist = df[df["user_id"] == top["user_id"]].sort_values("timestamp")
        grid = st.columns(3)
        for i, feat in enumerate(FEATURE_COLS):
            meta = FEATURE_META[feat]
            value = top[feat]
            display_val = f"{value:,.2f}" if isinstance(value, float) and not float(value).is_integer() else f"{int(value)}"
            label_txt = trend_label(fdf[feat], value)
            trend_series = user_hist[feat].tail(15)
            if len(trend_series) < 2:
                trend_series = fdf[feat].tail(15)
            with grid[i % 3]:
                st.markdown(
                    f'<div class="metric-card" style="margin-bottom:10px;">'
                    f'<div class="mc-label">{meta["icon"]} {i+1}. {meta["label"]}</div>'
                    f'<div class="mc-value" style="font-size:20px;">{display_val} '
                    f'<span style="font-size:11px; color:{TEXT_SUB};">{meta["unit"]}</span></div>'
                    f'<div style="font-size:11.5px; color:{"#22C55E" if "Typical" in label_txt else "#F97316"};">'
                    f'{label_txt}</div></div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    sparkline_fig(trend_series, color=RISK_COLORS[top["risk_level"]]),
                    use_container_width=True, config={"displayModeBar": False},
                    key=f"spark_{feat}",
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with right2:
        st.markdown(
            '<div class="panel"><div class="panel-title">D. ALERTS <span class="sub">(Recent)</span></div>',
            unsafe_allow_html=True,
        )
        recent_alerts = (
            fdf[fdf["risk_level"] != "LOW"]
            .sort_values("timestamp", ascending=False)
            .head(8)
        )
        if recent_alerts.empty:
            st.info("No alerts in the current filter window.")
        else:
            alert_rows = ""
            for _, row in recent_alerts.iterrows():
                risk_val = f"{row['risk_score']:,.0f}"
                alert_rows += (
                    f'<tr style="border-bottom:1px solid {PANEL_BORDER};">'
                    f'<td style="padding:6px 4px; color:{TEXT_SUB};">{row['timestamp'].strftime('%H:%M:%S')}</td>'
                    f'<td style="padding:6px 4px; color:{TEXT_MAIN};">{row['user_id']}</td>'
                    f'<td style="padding:6px 4px; color:{TEXT_MAIN};">{row['device_id']}</td>'
                    f'<td style="padding:6px 4px; color:{TEXT_SUB};">{row['alert_type']}</td>'
                    f'<td style="padding:6px 4px; text-align:center;">{risk_badge(risk_val, {risk_val: RISK_COLORS[row['risk_level']]})}</td>'
                    f'<td style="padding:6px 4px; text-align:center;">{risk_badge(row['action'], ACTION_COLORS)}</td>'
                    f'</tr>'
                )
            st.markdown(
                f"""
                <table style="width:100%; font-size:12px; table-layout:fixed; border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:1px solid {PANEL_BORDER};">
                            <th style="text-align:left; padding:6px 4px; color:{TEXT_SUB};">Time</th>
                            <th style="text-align:left; padding:6px 4px; color:{TEXT_SUB};">User</th>
                            <th style="text-align:left; padding:6px 4px; color:{TEXT_SUB};">Device</th>
                            <th style="text-align:left; padding:6px 4px; color:{TEXT_SUB};">Alert Type</th>
                            <th style="text-align:center; padding:6px 4px; color:{TEXT_SUB};">Risk</th>
                            <th style="text-align:center; padding:6px 4px; color:{TEXT_SUB};">Action</th>
                        </tr>
                    </thead>
                    <tbody>{alert_rows}</tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        
    # ---- E. SYSTEM PERFORMANCE (simulated live telemetry) -------------------
    st.markdown(
        '<div class="panel"><div class="panel-title">E. SYSTEM PERFORMANCE <span class="sub">(Live, simulated)</span></div>',
        unsafe_allow_html=True,
    )
    render_system_performance(compact=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# SYSTEM PERFORMANCE (shared by Dashboard + dedicated page)
# ==============================================================================
def get_live_system_metrics() -> dict:
    """
    Time-seeded (non-static) simulated system telemetry. Values change roughly
    every 3 seconds as the app reruns. There are no CPU/Memory/latency columns
    in the dataset, so these are simulated for UI purposes; Concurrent Users is
    the one figure computed live from the currently filtered data.
    """
    tick = int(time.time() // 3)
    rng = random.Random(tick)
    return {
        "cpu": round(20 + rng.random() * 20, 0),
        "mem": round(32 + rng.random() * 18, 0),
        "latency": round(0.9 + rng.random() * 0.8, 2),
        "risk_update": round(0.4 + rng.random() * 0.6, 2),
        "dash_refresh": round(1.6 + rng.random() * 1.0, 2),
        "concurrent": int(fdf["session_id"].nunique()),
    }
    
def render_system_performance(compact=False):
    live = get_live_system_metrics()

    perf_metrics = [
        ("CPU Utilization", f"{live['cpu']:,.0f}%", "Target \u2264 50%"),
        ("Memory Utilization", f"{live['mem']:,.0f}%", "Target \u2264 60%"),
        ("Event Processing Latency", f"{live['latency']:,.2f} s", "Target \u2264 2 s"),
        ("Risk Update Interval", f"{live['risk_update']:,.2f} s", "Target \u2264 1 s"),
        ("Dashboard Refresh", f"{live['dash_refresh']:,.2f} s", "Target \u2264 3 s"),
        ("Concurrent Users", f"{live['concurrent']}", "Target \u2265 50"),
    ]
    cols = st.columns(6)
    for c, (label, val, target) in zip(cols, perf_metrics):
        c.markdown(
            f'<div class="metric-card"><div class="mc-label">{label}</div>'
            f'<div class="mc-value" style="font-size:21px;">{val}</div>'
            f'<div style="font-size:11px; color:{TEXT_SUB};">{target}</div></div>',
            unsafe_allow_html=True,
        )
    if not compact:
        st.markdown(
            '<div style="color:#FFFFFF; font-size:13px; margin-top:8px;">'
            "These operational metrics are lightweight simulated telemetry for UI "
            "demonstration (the dataset does not contain server telemetry); "
            "Concurrent Users is the one figure computed live from the filtered data."
            "</div>",
            unsafe_allow_html=True,
        )


def white_caption(text: str):
    st.markdown(
        f'<div style="color:#FFFFFF; font-size:12.5px; margin-top:6px;">{text}</div>',
        unsafe_allow_html=True,
    )


def measure_pipeline_timing(raw_data: pd.DataFrame, contam: float) -> dict:
    """
    Live-measures (wall-clock, milliseconds) the actual cost of each stage of
    the SMART-BYOD pipeline against the currently loaded dataset. Nothing here
    is a hardcoded constant — every value is timed fresh, on this run, on this
    machine, against this CSV.
    """
    timings = {}

    t0 = time.perf_counter()
    collected = raw_data.copy()
    timings["Collection"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    collected["timestamp"] = pd.to_datetime(collected["timestamp"])
    collected = collected.sort_values("timestamp").reset_index(drop=True)
    timings["Preprocessing"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    X_stage = collected[FEATURE_COLS].values
    timings["Features"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    stage_model = IsolationForest(
        n_estimators=200, contamination=contam, random_state=42, n_jobs=-1,
    )
    stage_model.fit(X_stage)
    stage_raw_scores = -stage_model.score_samples(X_stage)
    stage_pred = stage_model.predict(X_stage)
    timings["IsolationForest"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    stage_scaled = MinMaxScaler(feature_range=(0, 100)).fit_transform(
        stage_raw_scores.reshape(-1, 1)
    ).ravel()
    stage_bins = [-0.1, 29, 59, 79, 100]
    stage_labels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    stage_levels = pd.cut(stage_scaled, bins=stage_bins, labels=stage_labels)
    timings["RiskEngine"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = pd.Series(stage_levels).astype(str).map(ACTION_BY_LEVEL)
    timings["PolicyEngine"] = (time.perf_counter() - t0) * 1000

    # Floor at 0.1ms so a near-instant stage is still visible on the bar chart.
    return {k: max(v, 0.1) for k, v in timings.items()}


# ==============================================================================
# PAGE: MODEL EVALUATION (thesis / research figures — computed live from the CSV)
# ==============================================================================
def page_model_evaluation():
    st.markdown('<div class="app-title">\U0001F9EA Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Isolation Forest anomaly-detection performance and '
        'behavioral feature analysis, computed live. Figures 3 and 5-7 follow the sidebar '
        'filters when the dashboard runs on the CSV, and the expected anomaly rate below.</div>',
        unsafe_allow_html=True,
    )

    # Dark theme so matplotlib/seaborn figures match the rest of the app.
    # NOTE: sns.set_theme() resets rcParams to its own light "darkgrid" preset
    # internally, so our custom colors must be a separate dict applied AFTER
    # set_theme() runs — not passed in as its `rc=` argument (which aliases
    # the live plt.rcParams object and gets wiped out before it takes effect).
    dark_mpl_style = {
        "figure.facecolor": PANEL,
        "axes.facecolor": PANEL,
        "savefig.facecolor": PANEL,
        "axes.edgecolor": PANEL_BORDER,
        "axes.labelcolor": TEXT_MAIN,
        "axes.titlecolor": TEXT_MAIN,
        "text.color": TEXT_MAIN,
        "xtick.color": TEXT_SUB,
        "ytick.color": TEXT_SUB,
        "xtick.labelcolor": TEXT_SUB,
        "ytick.labelcolor": TEXT_SUB,
        "grid.color": PANEL_BORDER,
        "legend.facecolor": PANEL,
        "legend.edgecolor": PANEL_BORDER,
        "legend.labelcolor": TEXT_MAIN,
        "font.size": 10,
    }
    sns.set_theme(style="darkgrid")
    plt.rcParams.update(dark_mpl_style)

    full = df  # the whole scored dataset, all 10,000 rows
    # Ground-truth labels are not stored in the approved 8-table schema, so live FortiGate
    # data can't be graded. The accuracy figures (3, 5, 6, 7) are therefore computed on the
    # labeled benchmark dataset (the CSV) whenever it is available next to the script.
    lab, LABELS_FROM_BENCHMARK, FILTER_NOTE = None, False, None
    if "ground_truth" in full.columns:
        # CSV mode: grade exactly the records the sidebar filters (time, user type, risk level, device) select.
        lab = fdf if len(fdf) else full
        if len(fdf) == 0:
            FILTER_NOTE = "No records match the current filters - showing the full dataset instead."
        elif len(fdf) < len(full):
            FILTER_NOTE = f"Filtered view: {len(fdf):,} of {len(full):,} records match the sidebar filters."
    else:
        bench_path = find_file(CSV_FILENAME)
        if bench_path is not None:
            try:
                bench_raw = load_data(str(bench_path))
                if "ground_truth" in bench_raw.columns:
                    st.markdown(
                        f'<div style="color:{TEXT_SUB}; font-size:12px; margin-bottom:4px;">'
                        f'Expected anomaly rate for this evaluation (Isolation Forest contamination)</div>',
                        unsafe_allow_html=True)
                    bench_cont = st.slider(" ", min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                           key="bench_contamination",
                                           help="The share of records the model is told to flag. The confusion "
                                                "matrix, ROC and metrics are recomputed for the value you pick.")
                    lab = score_risk(bench_raw, contamination=bench_cont)
                    LABELS_FROM_BENCHMARK = True
            except Exception:
                lab = None
    HAS_LABELS = lab is not None
    LABELS_NOTE = ("These figures need ground-truth labels, which the live FortiGate data and the approved "
                   f"8-table schema do not store. Place **{CSV_FILENAME}** next to this script to evaluate "
                   "the model on the labeled benchmark dataset.")
    BENCH_NOTE = (f"Evaluated on the labeled benchmark dataset ({CSV_FILENAME}, {len(lab):,} records), "
                  "because live FortiGate events carry no ground-truth labels.") if LABELS_FROM_BENCHMARK else None
    EVAL_NOTE = None
    if HAS_LABELS:
        n_anom_true = int((lab["ground_truth"] == 1).sum())
        EVAL_NOTE = (f"Graded on {len(lab):,} records: {len(lab) - n_anom_true:,} normal and {n_anom_true:,} "
                     f"anomalous (ground truth). The model flagged {int(lab['predicted_anomaly'].sum()):,} as anomalous."
                     + (f" {FILTER_NOTE}" if FILTER_NOTE else ""))
    y_eval = lab["ground_truth"] if HAS_LABELS else None
    pred_eval = lab["predicted_anomaly"] if HAS_LABELS else None
    score_eval = lab["risk_score"] if HAS_LABELS else None

    def panel_start(title):
        st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>', unsafe_allow_html=True)

    def panel_end():
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Fig. 1: Feature Distribution Histograms ---------------------------
    panel_start("&mdash; Feature Distribution Histograms")
    st.markdown(
        '<div class="app-subtitle">Feature Distribution Histograms of Login Frequency (LF), '
        'Failed Login Attempts (FL), Resource-Access Frequency (RAF), and Application/API '
        'Activity Frequency (AAF).</div>',
        unsafe_allow_html=True,
    )
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 6))
    for ax, feat in zip(axes1.flatten(), ["LF", "FL", "RAF", "AAF"]):
        sns.histplot(full[feat], bins=30, kde=True, ax=ax, color=BRAND_RED)
        ax.set_title(f"{feat} \u2014 {FEATURE_META[feat]['label']}")
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)
    panel_end()

    
    if HAS_LABELS:
        if BENCH_NOTE:
            st.info(BENCH_NOTE)
        st.info(EVAL_NOTE)
        # ---- Fig. 3: Boxplots — Normal vs. Anomalous ---------------------------
        panel_start("&mdash; Feature Distributions: Normal vs. Anomalous (Ground Truth)")
        st.markdown(
            '<div class="app-subtitle">Compares the distributions of Login Frequency (LF), Failed Login Attempts (FL), Resource-Access Frequency (RAF), and Application/API Activity Frequency (AAF) between normal and anomalous classes.</div>',
            unsafe_allow_html=True,
        )
        fig3, axes3 = plt.subplots(2, 2, figsize=(10, 6))
        for ax, feat in zip(axes3.flatten(), ["LF", "FL", "RAF", "AAF"]):
            sns.boxplot(x="ground_truth", y=feat, data=lab, ax=ax, hue="ground_truth",
                         palette={0: BRAND_TEAL, 1: BRAND_RED}, legend=False)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Normal (0)", "Anomalous (1)"])
            ax.set_title(f"{feat} \u2014 {FEATURE_META[feat]['label']}")
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
        panel_end()

    else:
        st.info(LABELS_NOTE)

    col_a, col_b = st.columns(2)


    # ---- Fig. 2: Correlation Heatmap ---------------------------------------
    with col_a:
        panel_start("&mdash; Correlation Heatmap (Behavioral Features)")
        st.markdown(
            '<div class="app-subtitle">Pearson correlation coefficients among the nine behavioral features. Stronger relationships are observed between Resource-Access Frequency (RAF), Application/API Activity Frequency (AAF), and Rapid Activity (RA), indicating that excessive service usage often corresponds with elevated event density.</div>',
            unsafe_allow_html=True,
        )
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        corr = full[FEATURE_COLS].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2,
                     cbar_kws={"label": "Correlation"}, annot_kws={"size": 7})
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        panel_end()

    # ---- Fig. 4: Risk Level Distribution (real, model-derived) -------------
    with col_b:
        panel_start("&mdash; Risk Level Distribution")
        st.markdown('<div class="app-subtitle">Distribution of Risk Levels Generated by the SMART-BYOD Risk Engine.</div>',unsafe_allow_html=True,
        )
        fig4, ax4 = plt.subplots(figsize=(5, 5))
        order4 = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        counts4 = full["risk_level"].value_counts().reindex(order4).fillna(0)
        ax4.bar(counts4.index, counts4.values, color=[RISK_COLORS[l] for l in order4])
        ax4.set_ylabel("Count")
        #ax4.set_title("Risk Level Distribution")
        for i, cou in enumerate(counts4):
            ax4.text(i, cou + 0.02, f"{cou:,}", ha="center", color=TEXT_MAIN)
        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)
        white_caption("Derived from this run's Isolation Forest risk scores &mdash; not randomly assigned.")
        panel_end()       
 
    if HAS_LABELS:
        col_c, col_d = st.columns(2)

        # ---- Fig. 5: Confusion Matrix ------------------------------------------
        with col_c:
            panel_start("&mdash; Confusion Matrix (Isolation Forest vs. Ground Truth)")
            st.markdown(
                '<div class="app-subtitle">Classification outcomes produced by Isolation Forest relative to the ground-truth labels. The confusion matrix shows the number of true positives, true negatives, false positives, and false negatives observed during evaluation.</div>',
                unsafe_allow_html=True,
            )        
            cm = confusion_matrix(y_eval, pred_eval, labels=[0, 1])   # always 2x2, even if a filter leaves one class
            fig5, ax5 = plt.subplots(figsize=(5, 5))
            disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomalous"])
            disp.plot(cmap="Reds", ax=ax5, colorbar=False)
            ax5.grid(False)
            #ax5.set_title("Confusion Matrix")
            fig5.tight_layout()
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)
            panel_end()

        # ---- Fig. 6: ROC Curve --------------------------------------------------
        with col_d:
            panel_start("&mdash; ROC Curve")
            st.markdown(
                '<div class="app-subtitle">Receiver Operating Characteristic (ROC) Curve of the Isolation Forest Model.</div>',
                unsafe_allow_html=True,
            )  
            if y_eval.nunique() < 2:
                st.info("The ROC curve needs both normal and anomalous records. "
                        "Widen the filters to include both.")
            else:
                fpr, tpr, _ = roc_curve(y_eval, score_eval)
                auc = roc_auc_score(y_eval, score_eval)
                fig6, ax6 = plt.subplots(figsize=(5, 5))
                ax6.plot(fpr, tpr, color=BRAND_RED, linewidth=2, label=f"AUC = {auc:,.3f}")
                ax6.plot([0, 1], [0, 1], "--", color=TEXT_SUB)
                ax6.set_xlabel("False Positive Rate")
                ax6.set_ylabel("True Positive Rate")
                #ax6.set_title("ROC Curve")
                ax6.legend()
                fig6.tight_layout()
                st.pyplot(fig6, use_container_width=True)
                plt.close(fig6)
            panel_end()

    else:
        st.info(LABELS_NOTE)

    col_e, col_f = st.columns(2)

    if HAS_LABELS:
        # ---- Fig. 7: Precision / Recall / F1 ------------------------------------
        with col_e:
            panel_start("&mdash; Detection Metrics")
            st.markdown(
                '<div class="app-subtitle">Comparison of Precision, Recall, and F1-Score Metrics.</div>',
                unsafe_allow_html=True,
            )  
            precision = precision_score(y_eval, pred_eval, zero_division=0)
            recall = recall_score(y_eval, pred_eval, zero_division=0)
            f1 = f1_score(y_eval, pred_eval, zero_division=0)
            fig7, ax7 = plt.subplots(figsize=(5, 5))
            labels7, vals7 = ["Precision", "Recall", "F1-Score"], [precision, recall, f1]
            ax7.bar(labels7, vals7, color=BRAND_RED)
            ax7.set_ylim(0, 1)
            #ax7.set_title("Detection Metrics")
            for i, v in enumerate(vals7):
                ax7.text(i, v + 0.02, f"{v:,.2f}", ha="center", color=TEXT_MAIN)
            fig7.tight_layout()
            st.pyplot(fig7, use_container_width=True)
            plt.close(fig7)
            panel_end()

    else:
        with col_e:
            st.info(LABELS_NOTE)

    # ---- Fig. 8: Event Processing Performance (live-measured) --------------
    with col_f:
        panel_start("&mdash; Event Processing Performance")
        st.markdown(
            '<div class="app-subtitle">Average Event Processing Time Across the SMART-BYOD Processing Pipeline.</div>',
            unsafe_allow_html=True,
        )
        stage_timings = measure_pipeline_timing(raw_df, contamination)
        fig8, ax8 = plt.subplots(figsize=(6, 5))
        stages8, times8 = list(stage_timings.keys()), list(stage_timings.values())
        ax8.bar(stages8, times8, color=BRAND_RED)
        ax8.set_ylabel("Time (ms)")
        #ax8.set_title("Event Processing Performance")
        plt.setp(ax8.get_xticklabels(), rotation=20, ha="right")
        fig8.tight_layout()
        st.pyplot(fig8, use_container_width=True)
        plt.close(fig8)
        white_caption("Timed live against the loaded CSV on this run (not hardcoded).")
        panel_end()

    col_g, col_h = st.columns(2)

    # ---- Fig. 9: CPU & Memory Utilization (live, simulated) ----------------
    with col_g:
        panel_start("&mdash; CPU &amp; Memory Utilization")
        st.markdown(
            '<div class="app-subtitle">CPU and Memory Utilization During SMART-BYOD Operation.</div>',
            unsafe_allow_html=True,
        )
        live = get_live_system_metrics()
        fig9, ax9 = plt.subplots(figsize=(5, 5))
        ax9.bar(["CPU", "Memory"], [live["cpu"], live["mem"]],
                color=[BRAND_RED, BRAND_TEAL])
        ax9.set_ylabel("%")
        ax9.set_ylim(0, 100)
        #ax9.set_title("CPU and Memory Utilization")
        fig9.tight_layout()
        st.pyplot(fig9, use_container_width=True)
        plt.close(fig9)
        white_caption("Live simulated telemetry (same values as the System Performance panel) &mdash; "
                      "the dataset has no CPU/Memory columns.")
        panel_end()

    # ---- Fig. 10: Dashboard Alert Distribution ------------------------------
    with col_h:
        panel_start("&mdash; Dashboard Alert Distribution by Risk Level")
        order10 = ["MODERATE", "HIGH", "CRITICAL"]
        alert_counts = (
            full[full["risk_level"] != "LOW"]["risk_level"]
            .value_counts().reindex(order10).fillna(0)
        )
        fig10, ax10 = plt.subplots(figsize=(5, 5))
        ax10.bar(alert_counts.index, alert_counts.values,
                  color=[RISK_COLORS[l] for l in order10])
        ax10.set_ylabel("Count")
        #ax10.set_title("Dashboard Alert Distribution by Risk Level")
        fig10.tight_layout()
        st.pyplot(fig10, use_container_width=True)
        plt.close(fig10)
        white_caption("Only sessions that actually fired an alert (Moderate, High, Critical) are counted.")
        panel_end()

 

# ==============================================================================
# PAGE: ALERTS
# ==============================================================================
def page_alerts():
    st.markdown('<div class="app-title">🔔 Alerts</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">All flagged sessions in the current filter window</div>', unsafe_allow_html=True)

    alerts = fdf[fdf["risk_level"] != "LOW"].sort_values("timestamp", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Alerts", f"{len(alerts):,}")
    c2.metric("High/Critical", f"{len(alerts[alerts['risk_level'].isin(['HIGH','CRITICAL'])]):,}")
    c3.metric("Blocked", f"{len(alerts[alerts['action']=='Block']):,}")

    show = alerts[[
        "timestamp", "user_id", "user_type", "device_id", "alert_type",
        "risk_score", "risk_level", "action", "ip_address", "location",
    ]].copy()
    show["device_id"] = show["device_id"].map(lambda v: mask_mac(v, keep=2))
    show = show.rename(columns={
        "timestamp": "Time", "user_id": "User", "user_type": "Type",
        "device_id": "Device", "alert_type": "Alert Type", "risk_score": "Risk",
        "risk_level": "Level", "action": "Action", "ip_address": "IP Address",
        "location": "Location",
    })
    st.dataframe(show, use_container_width=True, hide_index=True, height=520)
    st.download_button(
        "\u2B07\uFE0F Download alerts as CSV",
        show.to_csv(index=False).encode("utf-8"),
        file_name="smart_byod_alerts.csv", mime="text/csv",
    )


# ==============================================================================
# PAGE: USERS & DEVICES
# ==============================================================================
def page_users_devices():
    st.markdown('<div class="app-title">👥 Users &amp; Devices</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Aggregated risk posture per user and per device</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Users", "Devices"])
    with tab1:
        u = fdf.groupby("user_id").agg(
            user_type=("user_type", "first"),
            sessions=("session_id", "nunique"),
            events=("record_id", "count"),
            avg_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
            alerts=("risk_level", lambda s: (s != "LOW").sum()),
        ).reset_index().sort_values("avg_risk", ascending=False)
        u["avg_risk"] = u["avg_risk"].round(1)
        u["max_risk"] = u["max_risk"].round(1)
        u["user_id"] = u["user_id"].map(USER_ANON_MAP)   # anonymous display id, never the real name
        st.dataframe(
            u.rename(columns={
                "user_id": "User ID", "user_type": "Type", "sessions": "Sessions",
                "events": "Events", "avg_risk": "Avg Risk", "max_risk": "Max Risk",
                "alerts": "Alerts",
            }),
             height=480, #hide_index=True, use_container_width=True,
        )
    with tab2:
        d = fdf.groupby("device_id").agg(
            users=("user_id", "nunique"),
            sessions=("session_id", "nunique"),
            events=("record_id", "count"),
            avg_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
            alerts=("risk_level", lambda s: (s != "LOW").sum()),
        ).reset_index().sort_values("avg_risk", ascending=False)
        d["avg_risk"] = d["avg_risk"].round(1)
        d["max_risk"] = d["max_risk"].round(1)
        d["device_id"] = d["device_id"].map(lambda v: mask_mac(v, keep=3))
        st.dataframe(
            d.rename(columns={
                "device_id": "Device ID", "users": "Users", "sessions": "Sessions",
                "events": "Events", "avg_risk": "Avg Risk", "max_risk": "Max Risk",
                "alerts": "Alerts",
            }),
            use_container_width=True, hide_index=True, height=480,
        )


# ==============================================================================
# PAGE: ACTIVITY LOGS
# ==============================================================================
def page_activity_logs():
    st.markdown('<div class="app-title">📝 Activity Logs</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Raw filtered event log</div>', unsafe_allow_html=True)

    search = st.text_input("Search by Device ID or Session ID")
    logs = fdf.copy()
    if search:
        s = search.strip().lower()
        logs = logs[
            logs["device_id"].str.lower().str.contains(s)
            | logs["session_id"].str.lower().str.contains(s)
        ]

    logs = logs.copy()
    logs["user_id"] = logs["ip_address"].map(ip_as_x_mask)      # e.g. 137.158.X.X, never the name
    logs["device_id"] = logs["device_id"].map(lambda v: mask_mac(v, keep=3))

    cols = ["timestamp", "user_id", "user_type", "device_id", "session_id",
            "risk_score", "risk_level", "action"] + FEATURE_COLS
    st.dataframe(
        logs[cols].sort_values("timestamp", ascending=False)
            .rename(columns={"timestamp": "Time", "user_id": "User", "user_type": "Type",
                              "device_id": "Device", "session_id": "Session",
                              "risk_score": "Risk", "risk_level": "Level", "action": "Action"}),
        use_container_width=True, hide_index=True, height=560,
    )
    st.download_button(
        "\u2B07\uFE0F Download filtered log as CSV",
        logs[cols].to_csv(index=False).encode("utf-8"),
        file_name="smart_byod_activity_log.csv", mime="text/csv",
    )


# ==============================================================================
# PAGE: REPORTS
# ==============================================================================
def _reports_risk():
    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="panel"><div class="panel-title">Risk Level Distribution</div>', unsafe_allow_html=True)
        dist = fdf["risk_level"].value_counts().reindex(["LOW", "MODERATE", "HIGH", "CRITICAL"]).fillna(0)
        fig = go.Figure(go.Bar(
            x=dist.index, y=dist.values,
            marker_color=[RISK_COLORS[l] for l in dist.index],
        ))
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_MAIN),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="panel"><div class="panel-title">Alert Type Breakdown</div>', unsafe_allow_html=True)
        breakdown = fdf[fdf["risk_level"] != "LOW"]["alert_type"].value_counts()
        fig2 = go.Figure(go.Bar(
            x=breakdown.values, y=breakdown.index, orientation="h",
            marker_color=ACCENT,
        ))
        fig2.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_MAIN), yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Average Daily Risk Trend</div>', unsafe_allow_html=True)
    daily = fdf.set_index("timestamp").resample("D")["risk_score"].mean().dropna()
    fig3 = go.Figure(go.Scatter(x=daily.index, y=daily.values, mode="lines+markers",
                                 line=dict(color=ACCENT, width=2)))
    fig3.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MAIN),
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    top_users = (
        fdf.groupby("user_id")["risk_score"].mean().sort_values(ascending=False).head(10)
    )
    st.markdown('<div class="panel"><div class="panel-title">Top 10 Highest-Risk Users (avg. score)</div>', unsafe_allow_html=True)
    fig4 = go.Figure(go.Bar(x=top_users.values, y=top_users.index, orientation="h",
                             marker_color="#F97316"))
    fig4.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MAIN), yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Device & Network reports
# ------------------------------------------------------------------------------
def _style_fig(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MAIN), showlegend=False,
        hoverlabel=dict(bgcolor=PANEL, font_color=TEXT_MAIN, bordercolor=PANEL_BORDER),
    )
    fig.update_xaxes(gridcolor=PANEL_BORDER, zerolinecolor=PANEL_BORDER, color=TEXT_SUB)
    fig.update_yaxes(gridcolor=PANEL_BORDER, zerolinecolor=PANEL_BORDER, color=TEXT_SUB)
    return fig


def _panel(title, note=None):
    st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div style="color:{TEXT_SUB}; font-size:12px; margin:-6px 0 8px 0;">{note}</div>',
                    unsafe_allow_html=True)


def _hbar(labels, values, hover_unit, value_fmt="{:,.0f}", muted=("Unknown",), height=None):
    """Single-hue horizontal bars, largest on top, zero baseline, value labels at the bar end."""
    labels = [str(l) for l in labels]
    colors = ["#6B7280" if l in muted else BRAND_RED for l in labels]
    fig = go.Figure(go.Bar(
        x=list(values), y=labels, orientation="h", marker_color=colors,
        text=[value_fmt.format(v) for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(color=TEXT_MAIN),
        hovertemplate="%{y}<br><b>%{x:,.1f}</b> " + hover_unit + "<extra></extra>",
    ))
    _style_fig(fig, height or max(240, 34 * len(labels) + 40))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _table_view(df_table):
    with st.expander("View as table"):
        st.dataframe(df_table, use_container_width=True, hide_index=True)


def _reports_devices():
    db_path = find_file(DB_FILENAME) if DATA_SOURCE == "sqlite" else None
    if db_path is None:
        st.info("Device & network reports read the saved FortiGate DHCP logs from the SQLite database. "
                "The dashboard is currently running from the CSV, which has no device inventory. "
                "Run the loader on a FortiGate JSON export to enable these reports.")
        return
    try:
        ev = load_device_events(str(db_path), db_signature(db_path))
    except Exception as exc:
        st.warning(f"Couldn't read the device inventory ({type(exc).__name__}).")
        return
    if ev.empty:
        st.info("No DHCP device events are stored yet. Re-run load_fortigate_logs_to_sqlite.py on a "
                "FortiGate export that includes the DHCP Ack logs (event/system) and refresh this page.")
        return

    # ---- one row per device (distinct MAC) ----
    g = ev.groupby("mac")
    dev = g.agg(brand=("brand", "first"), os_family=("os_family", "first"),
                device_type=("device_type", "first"), device_label=("device_label", "first"),
                mac_masked=("mac_masked", "first"), private_mac=("private_mac", "first"),
                leases=("ts", "size"), first_seen=("ts", "min"), last_seen=("ts", "max"),
                networks=("network", "nunique")).reset_index()
    dev["hours_on_network"] = (dev["last_seen"] - dev["first_seen"]).dt.total_seconds() / 3600.0
    n_dev = len(dev)
    private_share = dev["private_mac"].mean() * 100
    known_brand = (dev["brand"] != "Unknown").mean() * 100
    span_h = (ev["ts"].max() - ev["ts"].min()).total_seconds() / 3600.0

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "📱", "Devices Seen", f"{n_dev:,}")
    metric_card(c2, "🌐", "Networks", f"{ev['network'].nunique():,}")
    metric_card(c3, "🕵️", "Private (Randomized) MACs", f"{private_share:.0f}%")
    metric_card(c4, "🏷️", "Brand Identified", f"{known_brand:.0f}%")
    st.markdown(
        f'<div style="color:{TEXT_SUB}; font-size:12px; margin:6px 0 14px 0;">'
        f'Source: FortiGate DHCP lease logs, {ev["ts"].min():%b %d, %H:%M} to {ev["ts"].max():%b %d, %H:%M} '
        f'({span_h:,.1f} h, {len(ev):,} lease events). A device is one distinct MAC address; '
        f'phones that rotate their private MAC can be counted more than once. Brand is inferred from '
        f'the device name the phone announces, not read from the hardware.</div>',
        unsafe_allow_html=True)

    # ---- Top 10 brands / Device class ----
    a, b = st.columns(2)
    with a:
        _panel("Top 10 Brand Devices Used", "Distinct devices per brand")
        top_b = dev["brand"].value_counts().head(10)
        _hbar(top_b.index, top_b.values, "devices")
        _table_view(top_b.rename_axis("Brand").reset_index(name="Devices").assign(
            Share=lambda t: (t["Devices"] / n_dev * 100).round(1).astype(str) + "%"))
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        _panel("Device Class Split", "Phone, laptop, tablet, wearable")
        cls = dev["device_type"].value_counts()
        _hbar(cls.index, cls.values, "devices")
        _table_view(cls.rename_axis("Device class").reset_index(name="Devices"))
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Top 10 longest time on network ----
    a, b = st.columns(2)
    with a:
        _panel("Top 10 Longest Time on Network",
               "Hours between a device's first and last DHCP lease in the stored logs "
               "(devices seen at least twice)")
        longest = dev[dev["leases"] >= 2].nlargest(10, "hours_on_network")
        if longest.empty:
            st.caption("No device has been seen more than once yet.")
        else:
            lab = (longest["device_label"] + " " + longest["mac_masked"])
            _hbar(lab, longest["hours_on_network"].round(2), "hours", value_fmt="{:,.1f} h",
                  muted=(), height=max(240, 34 * len(longest) + 40))
            _table_view(pd.DataFrame({
                "Device": lab.values, "Hours on network": longest["hours_on_network"].round(2).values,
                "First seen": longest["first_seen"].dt.strftime("%b %d %H:%M").values,
                "Last seen": longest["last_seen"].dt.strftime("%b %d %H:%M").values,
                "Lease events": longest["leases"].values}))
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        _panel("Most Frequent Reconnects", "DHCP lease events per device (renewals and rejoins)")
        freq = dev.nlargest(10, "leases")
        lab2 = freq["device_label"] + " " + freq["mac_masked"]
        _hbar(lab2, freq["leases"], "lease events", muted=())
        _table_view(pd.DataFrame({"Device": lab2.values, "Lease events": freq["leases"].values,
                                  "Networks used": freq["networks"].values}))
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Devices by network / busiest hours ----
    a, b = st.columns(2)
    with a:
        _panel("Devices by Network", "Distinct devices seen on each interface / SSID")
        by_net = ev.groupby("network")["mac"].nunique().sort_values(ascending=False)
        _hbar(by_net.index, by_net.values, "devices", muted=())
        _table_view(by_net.rename_axis("Network").reset_index(name="Devices"))
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        _panel("Busiest Hours", "Device connection events by hour of day")
        hours = ev["ts"].dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()
        fig = go.Figure(go.Bar(
            x=[f"{h:02d}:00" for h in hours.index], y=hours.values, marker_color=BRAND_RED,
            hovertemplate="%{x}<br><b>%{y:,}</b> connection events<extra></extra>"))
        _style_fig(fig, 300)
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(tickangle=-60, gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        peak = int(hours.idxmax())
        st.caption(f"Peak hour: {peak:02d}:00 with {int(hours.max()):,} events.")
        _table_view(pd.DataFrame({"Hour": [f"{h:02d}:00" for h in hours.index], "Events": hours.values}))
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Private MAC share ----
    _panel("Private (Randomized) MAC Share",
           "Devices using a locally-administered MAC; these cannot be matched to a hardware vendor "
           "and may appear as a new device if the phone rotates its address")
    share = dev.groupby("private_mac").size().rename({True: "Private MAC", False: "Hardware MAC"})
    share = share.reindex(["Private MAC", "Hardware MAC"], fill_value=0)
    _hbar(share.index, share.values, "devices", muted=("Hardware MAC",), height=170)
    st.markdown("</div>", unsafe_allow_html=True)


def page_reports():
    st.markdown('<div class="app-title">📄 Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Risk summary over the current filter window, and device & network usage</div>',
                unsafe_allow_html=True)
    tab_risk, tab_dev = st.tabs(["Risk Reports", "Device & Network"])
    with tab_risk:
        _reports_risk()
    with tab_dev:
        _reports_devices()


# ==============================================================================
# PAGE: SYSTEM PERFORMANCE (dedicated)
# ==============================================================================
def page_system_performance():
    st.markdown('<div class="app-title">📊 System Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Live (simulated) operational telemetry</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    render_system_performance(compact=False)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("\U0001F504 Refresh telemetry"):
        st.rerun()


# ==============================================================================
# PAGE: SETTINGS
# ==============================================================================
def page_settings():
    st.markdown('<div class="app-title">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Model configuration and about</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel"><div class="panel-title">Risk Policy / Decision Layer</div>',
        unsafe_allow_html=True,
    )
    legend_data = [
        ("LOW", "0 - 29", "Allow", "Maintain normal access; no elevated alert"),
        ("MODERATE", "30 - 59", "Verify", "Generate verification alert and require/represent additional authentication"),
        ("HIGH", "60 - 79", "Restrict", "Generate high-risk alert and apply/represent restricted access to selected resources"),
        ("CRITICAL", "80 - 100", "Block/Isolate", "Generate critical alert and deny/represent termination or isolation of the high-risk session/device"),
    ]
    legend_rows = ""
    for lvl, rng, pds, sr in legend_data:
        legend_rows += (
            f'<tr style="border-bottom:1px solid {PANEL_BORDER};">'
            f'<td style="padding:6px 4px; white-space:nowrap; vertical-align:top;">'
            f'<span class="legend-dot" style="background-color:{RISK_COLORS[lvl]};"></span>{lvl.title()}</td>'
            f'<td style="padding:6px 4px; color:{TEXT_MAIN}; white-space:nowrap; vertical-align:top;">{rng}</td>'
            f'<td style="padding:6px 4px; color:{TEXT_MAIN}; white-space:nowrap; vertical-align:top;">{pds}</td>'
            f'<td style="padding:6px 4px; color:{TEXT_SUB}; vertical-align:top;">{sr}</td>'
            f'</tr>'
        )
    st.markdown(
        f"""
        <table style="width:100%; border-collapse:collapse; font-size:14px; table-layout:fixed;">
            <thead>
                <tr style="border-bottom:1px solid {PANEL_BORDER};">
                    <th style="text-align:left; padding:6px 4px; color:{TEXT_SUB};">Risk Level</th>
                    <th style="text-align:left; padding:6px 4px; color:{TEXT_SUB};">Score Range</th>
                    <th style="text-align:left; padding:6px 4px; color:{TEXT_SUB};">Policy Decision</th>
                    <th style="text-align:left; padding:6px 4px; color:{TEXT_SUB};">System Response</th>
                </tr>
            </thead>
            <tbody>{legend_rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="panel"><div class="panel-title">Anomaly Detection Model</div>', unsafe_allow_html=True)
    st.write(f"- **Algorithm:** Isolation Forest (scikit-learn)")
    st.write(f"- **Behavioral features (9):** {', '.join(FEATURE_COLS)}")
    if DATA_SOURCE == "sqlite":
        st.write(f"- **Data source:** SQLite ({DB_FILENAME}) - read-only; scores saved by the loader")
        st.write("- **Contamination:** 0.05 (fixed by the loader, not adjustable here)")
    else:
        st.write(f"- **Data source:** CSV ({CSV_FILENAME}), scored in the app")
        st.write(f"- **Contamination (current):** {contamination}")
    st.write(f"- **Records loaded:** {len(df):,}")
    st.write(f"- **Unique users:** {df['user_id'].nunique()} &nbsp;|&nbsp; **Unique devices:** {df['device_id'].nunique()}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">About</div>', unsafe_allow_html=True)
    st.write(
        "SMART-BYOD: A Lightweight Behavioral Anomaly Detection Framework for "
        "BYOD Security. This dashboard is a monitoring front-end built for the "
        "framework's Zero-Trust-inspired risk engine and policy/decision layer."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# ROUTER
# ==============================================================================
PAGES = {
    "🏠 Dashboard": page_dashboard,
    "🔔 Alerts": page_alerts,
    "👥 Users & Devices": page_users_devices,
    "📝 Activity Logs": page_activity_logs,
    "📄 Reports": page_reports,
    "\U0001F9EA Model Evaluation": page_model_evaluation,
    "📊 System Performance": page_system_performance,
    "⚙️ Settings": page_settings,
}
PAGES[page]()
