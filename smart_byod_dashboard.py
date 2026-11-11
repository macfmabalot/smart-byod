"""
================================================================================
 SMART-BYOD Monitoring Dashboard  (six-feature version)
 A Lightweight Zero Trust-Inspired Behavioral Anomaly Detection Dashboard
 for BYOD Security - consistent with the Capstone (20261108-MABALOT_Capstone.docx)
================================================================================

HOW TO RUN (Anaconda Prompt):
------------------------------------------------------------------------------
1) cd to the folder that holds this file and smart_byod.db (or the dataset CSV).
2) (First time only)  pip install streamlit pandas numpy plotly scikit-learn matplotlib seaborn psutil
3) streamlit run smart_byod_AA.py

DATA:
------------------------------------------------------------------------------
PRIMARY SOURCE - SQLite: "smart_byod.db", written by SMART-BYOD_Analyze.py, read through a
read-only connection. Features, anomaly flags, the four risk components (A, I, D, C), risk
scores, levels and policy decisions come straight from the database - nothing is re-scored
in the app. Generator-labeled records (dataset_records.label_source = 'generator') carry
ground truth; live FortiGate records do not and are never graded. A database built by the
nine-feature version must first be rebuilt:  python SMART-BYOD_Analyze.py --rebuild-only

FALLBACK - CSV: "SMART_BYOD_10000_Dataset.csv" (or an upload). The app scores it with the
same method as the analyzer: Isolation Forest fitted on the normal records of a session-level
60/20/20 training partition, A from the anomaly score (Table 3.5-6), and I, D, C derived
from FL, DS and LTD (the CSV has no device inventory or network context).

Expected columns:
    record_id, scenario, ground_truth, user_id, user_type, device_id, session_id, timestamp,
    LF  (Login Frequency, logins/h)          FL  (Failed Login Attempts, /15 min)
    DS  (New/unregistered devices, /h)       LTD (Login-Time Deviation, min)
    RAF (Distinct internal resources, /5 min) AAF (Application/API events, /5 min)
A CSV from the nine-feature version (with SD/IPC/RA columns) still loads: only the six
features are used and its two-digit scenario codes are mapped to S01-S08 (S06 IPC and
S09 RA are dropped). Its DS and RAF follow the old definitions - regenerate it.

SCENARIOS (Table 3.5-9) and expected risk bands (Table 3.5-14):
------------------------------------------------------------------------------
    S01 Normal employee activity        Low            S05 Unusual login time        Moderate
    S02 Excessive login activity        Low–Moderate   S06 Excessive resource access Low–Moderate
    S03 Repeated failed authentication  Moderate–High  S07 High application/API      Low–Moderate
    S04 New or unregistered device      Moderate       S08 Combined anomalous        High–Critical
Live events carry no scenario label; the dashboard shows an indicative scenario inferred
from the Table 3.4-2 normal ranges (2+ features outside them -> S08), never as ground truth.

RISK / POLICY (Tables 3.5-6 and 3.5-7):
------------------------------------------------------------------------------
    R = 100 (0.40 A + 0.25 I + 0.20 D + 0.15 C)
    0-29 LOW -> Allow   30-59 MODERATE -> Verify   60-79 HIGH -> Restrict   80-100 CRITICAL -> Block/Isolate

SYSTEM PERFORMANCE: measured values only - the analyzer's system_metrics rows, the host's
CPU/memory through psutil, and the dashboard's own page-render time. Nothing is simulated.
IP addresses are masked; locations are not in the schema (the ITAdmin view shows a
simulated city, clearly labeled, as before).
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

_T_START = time.perf_counter()   # page-render timer (dashboard refresh time)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit

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

try:
    import psutil
except ImportError:
    psutil = None
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

FEATURE_COLS = ["LF", "FL", "DS", "LTD", "RAF", "AAF"]   # F1-F6

FEATURE_META = {
    "LF":  {"label": "Login Frequency",            "unit": "logins/h",        "icon": "\U0001F511"},
    "FL":  {"label": "Failed Login Attempts",      "unit": "fails/15 min",    "icon": "❌"},
    "DS":  {"label": "New/Unregistered Devices",   "unit": "devices/h",       "icon": "\U0001F4F1"},
    "LTD": {"label": "Login-Time Deviation",       "unit": "min",             "icon": "\U0001F550"},
    "RAF": {"label": "Resource Access (distinct)", "unit": "resources/5 min", "icon": "\U0001F4C1"},
    "AAF": {"label": "App/API Activity",           "unit": "events/5 min",    "icon": "\U0001F4BB"},
}

# Table 3.4-2: upper end of the normal range of each feature
NORMAL_LIMITS = {"LF": 4, "FL": 1, "DS": 0, "LTD": 60, "RAF": 5, "AAF": 20}

# Table 3.5-6 / 3.5-7
W_A, W_I, W_D, W_C = 0.40, 0.25, 0.20, 0.15
CALIBRATION_TARGET_LOW = 0.95

# ------------------------------------------------------------------------------
# SCENARIO CATALOGUE (Table 3.5-9) with the expected risk band (Table 3.5-14)
# ------------------------------------------------------------------------------
SCENARIOS = {
    "S01": {"name": "Normal employee activity",       "feature": None,  "band": (0.0, 29.99),  "expected": "Low",
            "threat": "None (control scenario)"},
    "S02": {"name": "Excessive login activity",       "feature": "LF",  "band": (20.0, 40.0),  "expected": "Low–Moderate",
            "threat": "Account compromise / credential theft"},
    "S03": {"name": "Repeated failed authentication", "feature": "FL",  "band": (45.0, 65.0),  "expected": "Moderate–High",
            "threat": "Brute force / credential stuffing"},
    "S04": {"name": "New or unregistered device",     "feature": "DS",  "band": (32.0, 60.0),  "expected": "Moderate",
            "threat": "Compromised or unauthorized device"},
    "S05": {"name": "Unusual login time",             "feature": "LTD", "band": (35.0, 55.0),  "expected": "Moderate",
            "threat": "Insider misuse / stolen credentials"},
    "S06": {"name": "Excessive resource access",      "feature": "RAF", "band": (20.0, 40.0),  "expected": "Low–Moderate",
            "threat": "Data enumeration / exfiltration"},
    "S07": {"name": "High application/API activity",  "feature": "AAF", "band": (20.0, 40.0),  "expected": "Low–Moderate",
            "threat": "API abuse / automated activity"},
    "S08": {"name": "Combined anomalous behavior",    "feature": None,  "band": (72.0, 100.0), "expected": "High–Critical",
            "threat": "Multi-vector attack"},
}
SCENARIO_ORDER = list(SCENARIOS)
EVIDENCE_SCENARIOS = ["S02", "S03", "S04", "S05", "S06", "S07"]   # one evidence table per feature
FEATURE_TO_SCENARIO = {m["feature"]: c for c, m in SCENARIOS.items() if m["feature"]}
COMBINED_MIN_FEATURES = 2   # labeling rule 3: two or more anomalous indicators -> combined scenario (S08)

# Codes written by the nine-feature versions. None = scenario removed with SD / IPC / RA.
LEGACY_APP_CODES = {
    "N1": "S01", "S1": "S02", "S2": None, "S3": "S03", "S4": "S04", "S5": "S05",
    "S6": None, "S7": "S06", "S8": "S07", "S9": None, "S10": "S08",
}
LEGACY_NINE_FEATURE_CODES = {
    "N01": "S01", "S01": "S01", "S02": "S02", "S03": "S03", "S04": "S04", "S05": "S05",
    "S06": None, "S07": "S06", "S08": "S07", "S09": None, "S10": "S08",
}

SCENARIO_LABELS = {code: f"{code} – {meta['name']}" for code, meta in SCENARIOS.items()}


def normalize_scenario_code(value, nine_feature_source: bool = False):
    """Map a scenario label to S01-S08 (two-digit codes of a nine-feature source use the old numbering)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    code = str(value).strip().upper().replace(" ", "")
    if not code or code in ("NONE", "NAN"):
        return None
    if code in LEGACY_APP_CODES:
        return LEGACY_APP_CODES[code]
    if nine_feature_source and code in LEGACY_NINE_FEATURE_CODES:
        return LEGACY_NINE_FEATURE_CODES[code]
    if code in SCENARIOS:
        return code
    if re.fullmatch(r"N0*1", code):
        return "S01"
    return None


def is_nine_feature_frame(frame: pd.DataFrame) -> bool:
    return any(c in frame.columns for c in ("SD", "IPC", "RA"))


def scenario_codes(frame: pd.DataFrame) -> pd.Series:
    """S01-S08 code for every row of a labeled frame."""
    nine = is_nine_feature_frame(frame) or bool(frame.attrs.get("nine_feature", False))
    return frame["scenario"].map(lambda v: normalize_scenario_code(v, nine))


def band_text(code: str) -> str:
    m = SCENARIOS[code]
    lo, hi = m["band"]
    return f"{m['expected']} ({lo:g}–{hi:g})"


def expected_badge(code: str) -> str:
    """Badge for a scenario's expected band, colored by the band's upper level."""
    m = SCENARIOS.get(code, SCENARIOS["S01"])
    top_level = m["expected"].split("–")[-1].upper()
    return risk_badge(band_text(code), {band_text(code): RISK_COLORS.get(top_level, "#64748B")})


def wilson_interval(k: int, m: int, z: float = 1.96):
    if m == 0:
        return float("nan"), float("nan")
    p = k / m
    centre = p + z * z / (2 * m)
    half = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m))
    denom = 1 + z * z / m
    return (centre - half) / denom, (centre + half) / denom


# ------------------------------------------------------------------------------
# ALERTS (Table 3.3-17 and Figure 3.3-7)
# ------------------------------------------------------------------------------
ALERT_TYPE_BY_LEVEL = {"LOW": "Normal Activity", "MODERATE": "Verification Alert",
                       "HIGH": "Restriction Alert", "CRITICAL": "Critical Security Alert"}
ALERT_CARD = {   # level -> (icon, banner, recommended response)
    "LOW":      ("\U0001F7E2", "NORMAL ACTIVITY", "Allow – maintain normal access"),
    "MODERATE": ("\U0001F7E1", "VERIFICATION REQUIRED", "Additional Verification"),
    "HIGH":     ("\U0001F7E0", "HIGH-RISK ACTIVITY DETECTED", "Restrict Access"),
    "CRITICAL": ("\U0001F534", "CRITICAL SECURITY ALERT", "BLOCK/ISOLATE"),
}
FEATURE_INDICATORS = {
    "LF":  "Excessive login frequency (LF = {v:g} successful logins in 1 hour)",
    "FL":  "Multiple failed login attempts (FL = {v:g} in 15 minutes)",
    "DS":  "New or unregistered devices (DS = {v:g} in 1 hour)",
    "LTD": "Unusual login time (LTD = {v:.0f} minutes from the user's baseline)",
    "RAF": "High resource-access frequency (RAF = {v:g} distinct resources in 5 minutes)",
    "AAF": "High application/API activity (AAF = {v:g} events in 5 minutes)",
}


def indicators_from_features(row) -> str:
    """Alert indicators for rows without analyzer output (CSV mode): the Isolation Forest flag and every
    feature outside its Table 3.4-2 normal range, strongest first (same wording as the analyzer)."""
    out = ["Behavior flagged as anomalous by the Isolation Forest"] if int(row["predicted_anomaly"]) == 1 else []
    over = [f for f in FEATURE_COLS if float(row[f]) > NORMAL_LIMITS[f]]
    over.sort(key=lambda f: -float(row[f]) / max(NORMAL_LIMITS[f], 1))
    out += [FEATURE_INDICATORS[f].format(v=float(row[f])) for f in over]
    if float(row["DS"]) >= 1:
        out.append("New device detected (not in the inventory)")
    return "; ".join(out)


# Thresholds used by the Scenario Evidence tables (Dashboard, section C).
EVIDENCE_ROWS = 12                 # most recent events of the current user shown per table
FL_LOCKOUT_THRESHOLD = 5           # failed attempts that give I = 1 (Table 3.5-6) and a lockout record

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
ACTION_BY_LEVEL = {          # Table 3.5-7
    "LOW": "Allow",
    "MODERATE": "Verify",
    "HIGH": "Restrict",
    "CRITICAL": "Block/Isolate",
}
BLOCK_ACTIONS = ("Block/Isolate", "Block")      # "Block" = label written by the nine-feature analyzer
ACTION_COLORS = {
    "Allow": "#22C55E",
    "Monitor": "#22C55E",
    "Verify": "#F59E0B",
    "Restrict": "#F97316",
    "Block/Isolate": "#EF4444",
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

# Login usernames (as stored in auth.users.<key>, case-insensitive) that see
# unmasked device/user ids and a visible location. Everyone else always gets
# the masked/anonymous view, regardless of what the underlying data contains.
ADMIN_LOGIN_USERNAMES = {"itadmin"}

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
    st.session_state["auth_user"] = {"name": user["name"], "via": pending["kind"], "username": user.get("username", "")}
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
    st.session_state["auth_user"] = {"name": user["name"], "via": via, "username": user.get("username", "")}
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
        code = st.text_input("Sign-in code", max_chars=OTP_LENGTH + 2, autocomplete="one-time-code")
        if st.form_submit_button("Verify and sign in", type="primary", width='stretch'):
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
                username = st.text_input("Username", autocomplete="username")
                password = st.text_input("Password", type="password", autocomplete="current-password")
                if st.form_submit_button("Sign in", type="primary", width='stretch'):
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
                    if st.button("Send code", type="primary", width='stretch'):
                        msg = _request_code(kind, "", demo, known_user=user)
                        if st.session_state.get("auth_pending"):
                            st.session_state["auth_info"] = msg
                            st.rerun()
                        st.warning(msg)
            else:
                _code_entry(demo)
            if st.button("Back to sign in", width='stretch'):
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
                                            placeholder="name@company.com" if kind == "email" else "+63 917 123 4567",
                                            autocomplete="off")
                    if st.form_submit_button("Send code", type="primary", width='stretch'):
                        msg = _request_code(kind, entered, demo)
                        if st.session_state.get("auth_pending"):
                            st.session_state["auth_info"] = msg
                            st.rerun()
                        st.warning(msg)
            else:
                _code_entry(demo)
                if st.button("Use a different email / number", width='stretch'):
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
# built on (same column names), plus the analyzer's saved anomaly/risk/policy output
# and, for generator-labeled records, the ground truth from dataset_records.
SQLITE_QUERY = """
SELECT
    se.event_id                         AS record_id,
    u.user_code                         AS user_id,
    u.user_type                         AS user_type,
    d.device_code                       AS device_id,
    d.registration_status               AS device_status,
    se.session_id                       AS session_id,
    bf.observation_timestamp            AS timestamp,
    bf.login_frequency                  AS LF,
    bf.failed_login_attempts            AS FL,
    bf.device_switching_events          AS DS,
    bf.login_time_deviation             AS LTD,
    bf.resource_access_frequency        AS RAF,
    bf.application_activity_frequency   AS AAF,
    se.ip_address                       AS ip_address,
    se.resource                         AS resource,
    ar.model_version                    AS model_version,
    ar.anomaly_score                    AS anomaly_score,
    ar.model_prediction                 AS model_prediction,
    ar.anomaly_flag                     AS anomaly_flag,
    ar.top_features                     AS top_features,
    ra.anomaly_component                AS comp_A,
    ra.identity_risk                    AS comp_I,
    ra.device_risk                      AS comp_D,
    ra.contextual_risk                  AS comp_C,
    ra.risk_score                       AS risk_score,
    ra.risk_level                       AS risk_level,
    pd.policy_action                    AS action,
    pd.policy_reason                    AS alert_type,
    pd.response_time_ms                 AS response_time_ms,
    dr.scenario                         AS scenario,
    dr.ground_truth                     AS ground_truth,
    dr.label_source                     AS label_source
FROM behavioral_features bf
JOIN security_events se        ON se.event_id   = bf.event_id
JOIN users u                   ON u.user_id     = bf.user_id
JOIN devices d                 ON d.device_id   = bf.device_id
LEFT JOIN anomaly_results ar   ON ar.feature_id = bf.feature_id
LEFT JOIN risk_assessments ra  ON ra.anomaly_id = ar.anomaly_id
LEFT JOIN policy_decisions pd  ON pd.risk_id    = ra.risk_id
LEFT JOIN dataset_records dr   ON dr.feature_id = bf.feature_id
ORDER BY bf.observation_timestamp
"""


class NineFeatureDatabase(RuntimeError):
    pass


def _read_sql(conn_factory) -> pd.DataFrame:
    conn = conn_factory()
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(behavioral_features);")}
        if "session_duration" in cols or "ip_address_changes" in cols:
            raise NineFeatureDatabase("the database was built by the nine-feature analyzer - run "
                                      "'python SMART-BYOD_Analyze.py --rebuild-only' once")
        return pd.read_sql_query(SQLITE_QUERY, conn)
    finally:
        conn.close()


@st.cache_data(show_spinner="Loading SMART-BYOD data from SQLite...")
def load_from_sqlite(db_path: str, signature: tuple):
    """
    Read the saved data through an independent read-only connection (Table 3.3-15:
    WAL lets these reads run while the analyzer is writing). `signature` is only a
    cache key, so a fresh analyzer commit invalidates the cached copy.
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
    except NineFeatureDatabase:
        raise
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
    flag = raw.pop("anomaly_flag")
    pred = raw.pop("model_prediction")
    raw["predicted_anomaly"] = flag.fillna((pred == -1).astype(float)).astype(int)
    for c in ("comp_A", "comp_I", "comp_D", "comp_C"):
        raw[c] = raw[c].astype(float)
    raw["ip_address"] = raw["ip_address"].fillna("N/A")
    raw["location"] = "N/A"  # no geo data in the approved schema
    raw["driver"] = raw.pop("alert_type").fillna("Unspecified")            # policy_reason: dominant component
    raw["alert_type"] = raw["risk_level"].map(ALERT_TYPE_BY_LEVEL)           # Table 3.3-17
    raw["indicators"] = raw.pop("top_features").fillna("")
    raw["session_id"] = raw["session_id"].fillna("—").astype(str)
    # Ground truth exists only for generator-labeled records (labeling rule 1). Live-only data -> no labels.
    labeled = raw["label_source"].eq("generator")
    if labeled.any():
        raw.loc[~labeled, ["scenario", "ground_truth"]] = np.nan
    else:
        raw = raw.drop(columns=["scenario", "ground_truth"])
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
       COALESCE(NULLIF(d.registration_status, ''), 'unknown') AS registration_status,
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
    ev["ip"] = ev["ip"] #.map(mask_ip)                            # never show full addresses
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
        except NineFeatureDatabase as exc:
            st.sidebar.error(f"'{DB_FILENAME}': {exc}. Using the CSV instead.")
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
# RISK ENGINE for the CSV fallback - same method as SMART-BYOD_Analyze.py
#   Isolation Forest (t=100, seed 42) fitted on normal training records;
#   A from s(x) = -score_samples(x) with theta = -offset_ (Table 3.5-6);
#   I = min(FL/5, 1); D from DS (1 -> 0.6 new device, >=2 -> 1.0);
#   C = C_time from LTD (0 at 60 min, 1 at 180 min). The CSV has no device
#   inventory or source network, so the unclassified-device and C_net terms are 0.
# ==============================================================================
def anomaly_component(s, theta, s_min, s_max):
    """Vectorized Table 3.5-6 mapping: A > 0.5 exactly when s > theta (flagged)."""
    s = np.asarray(s, dtype=float)
    low = 0.5 * (s - s_min) / max(theta - s_min, 1e-12)
    high = 0.5 + 0.5 * (s - theta) / max(s_max - theta, 1e-12)
    return np.where(s <= theta, np.clip(low, 0.0, 0.5), np.clip(high, 0.5 + 1e-9, 1.0))


def components_from_features(frame: pd.DataFrame):
    fl, ds, ltd = (frame[c].astype(float).to_numpy() for c in ("FL", "DS", "LTD"))
    I = np.minimum(fl / FL_LOCKOUT_THRESHOLD, 1.0)
    D = np.where(ds >= 2, 1.0, np.where(ds >= 1, 0.6, 0.0))
    C = np.clip((ltd - 60.0) / 120.0, 0.0, 1.0)
    return I, D, C


def risk_level_of(R):
    return pd.cut(pd.Series(R), bins=[-np.inf, 30, 60, 80, np.inf], right=False,
                  labels=["LOW", "MODERATE", "HIGH", "CRITICAL"]).astype(str).to_numpy()


def policy_reason(A, I, D, C):
    parts = np.vstack([W_A * A, W_I * I, W_D * D, W_C * C])
    names = np.array(["Elevated anomaly score",
                      "Identity/authentication risk (failed logins, concurrent devices)",
                      "Device risk (new, unclassified or another user's device)",
                      "Contextual risk (login time, new external network)"])
    out = names[parts.argmax(axis=0)]
    return np.where(parts.max(axis=0) > 0, out, "Nominal behavioral risk")


def split_partitions(frame: pd.DataFrame) -> np.ndarray:
    """Session-level 60/20/20 split (seed 42), the same protocol as --evaluate-csv."""
    groups = frame["session_id"].astype(str) if "session_id" in frame.columns else pd.Series(range(len(frame)))
    idx = np.arange(len(frame))
    tr, rest = next(GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42).split(idx, groups=groups))
    va_rel, te_rel = next(GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
                          .split(rest, groups=groups.iloc[rest]))
    part = np.empty(len(frame), dtype=object)
    part[tr], part[rest[va_rel]], part[rest[te_rel]] = "train", "validation", "test"
    return part


@st.cache_data(show_spinner="Scoring behavioral risk (Isolation Forest + risk engine)...")
def score_risk(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    df.attrs["nine_feature"] = is_nine_feature_frame(df)
    X = df[FEATURE_COLS].astype(float).fillna(0.0).to_numpy()
    if "ground_truth" in df.columns and df["ground_truth"].notna().all():
        df["partition"] = split_partitions(df)
        fit_mask = (df["partition"].eq("train") & df["ground_truth"].astype(int).eq(0)).to_numpy()
    else:
        within = np.all([df[f].astype(float).to_numpy() <= NORMAL_LIMITS[f] for f in FEATURE_COLS], axis=0)
        fit_mask = within if within.sum() >= 20 else np.ones(len(df), dtype=bool)
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42).fit(X[fit_mask])
    s = -model.score_samples(X)
    s_fit = -model.score_samples(X[fit_mask])
    theta = float(-model.offset_)
    df["anomaly_score"] = s
    df["predicted_anomaly"] = (model.predict(X) == -1).astype(int)       # -1 = anomaly -> 1
    A = anomaly_component(s, theta, float(s_fit.min()), float(s_fit.max()))
    I, D, C = components_from_features(df)
    R = 100.0 * (W_A * A + W_I * I + W_D * D + W_C * C)
    df["comp_A"], df["comp_I"], df["comp_D"], df["comp_C"] = (np.round(100 * v, 2) for v in (A, I, D, C))
    df["risk_score"] = np.round(R, 1)
    df["risk_level"] = risk_level_of(R)
    df["action"] = df["risk_level"].map(ACTION_BY_LEVEL)
    df["driver"] = policy_reason(A, I, D, C)
    df["alert_type"] = df["risk_level"].map(ALERT_TYPE_BY_LEVEL)
    df["indicators"] = df.apply(indicators_from_features, axis=1)
    return df


# ------------------------------------------------------------------------------
# Indicative scenario (S01-S08) for events without a generator label
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_feature_baseline(frame: pd.DataFrame) -> dict:
    """Normal-behavior baseline per feature, from LOW-risk events (all events if
    there are too few). Used by the evidence tables."""
    normal, source = frame[frame["risk_level"] == "LOW"], "LOW-risk events"
    if len(normal) < 20:
        normal, source = frame, "all events"
    out = {}
    for feat in FEATURE_COLS:
        s = normal[feat].astype(float)
        q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        iqr = float(q[0.75] - q[0.25])
        std = float(s.std()) if len(s) > 1 else 0.0
        out[feat] = {
            "p05": float(q[0.05]), "median": float(q[0.5]), "p95": float(q[0.95]),
            "scale": iqr if iqr > 0 else (std if std > 0 else 1.0),
            "source": source, "n": int(len(s)),
        }
    return out


def infer_scenarios(frame: pd.DataFrame):
    """Indicative scenario from the Table 3.4-2 normal ranges (not ground truth):
         LOW risk                                   -> S01
         2+ features outside their normal range     -> S08 (labeling rule 3)
         1 feature outside its normal range         -> that feature's scenario
         none outside, but risk above Low           -> scenario of the relatively largest feature
    Returns (codes, number_of_features_outside_normal_range)."""
    X = frame[FEATURE_COLS].astype(float)
    lim = pd.Series(NORMAL_LIMITS)
    above = X.gt(lim, axis=1)
    rel = X.div(lim.replace(0, 1), axis=1)
    n_above = above.sum(axis=1)
    strongest_above = rel.where(above, -np.inf).idxmax(axis=1).map(FEATURE_TO_SCENARIO)
    most_relative = rel.idxmax(axis=1).map(FEATURE_TO_SCENARIO)
    is_low = frame["risk_level"].astype(str).eq("LOW")
    codes = np.select(
        [is_low, n_above >= COMBINED_MIN_FEATURES, n_above >= 1],
        [np.full(len(frame), "S01"), np.full(len(frame), "S08"), strongest_above],
        default=most_relative,
    )
    return pd.Series(codes, index=frame.index), n_above


def assign_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    """scenario_code / scenario_name / scenario_source. A generator label wins; otherwise the
    scenario is inferred and marked as indicative."""
    inferred, n_above = infer_scenarios(frame)
    if "scenario" in frame.columns:
        labelled = scenario_codes(frame)
        frame["scenario_code"] = labelled.fillna(inferred)
        frame["scenario_source"] = np.where(labelled.notna(), "Generator label (ground truth)",
                                            "Indicative - inferred from features")
    else:
        frame["scenario_code"] = inferred
        frame["scenario_source"] = "Indicative - inferred from features"
    frame["scenario_name"] = frame["scenario_code"].map(lambda c: SCENARIOS[c]["name"])
    frame["features_above_baseline"] = n_above.astype(int)
    return frame


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
def build_user_anon_map(employee_ids: tuple, guest_ids: tuple) -> dict:
    """Stable, deterministic anonymous label per real user id: user0001, user0002, ... for
    employees, guest001, guest002, ... for guests (each its own sequence, sorted so the
    mapping is reproducible for the same set of users). Used everywhere a User ID is shown
    anonymously, so the same real id always maps to the same label across every page."""
    m = {uid: f"user{idx:04d}" for idx, uid in enumerate(sorted(employee_ids), start=1)}
    m.update({uid: f"guest{idx:03d}" for idx, uid in enumerate(sorted(guest_ids), start=1)})
    return m


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
    IS_ADMIN = bool(_who) and str(_who.get("username", "")).strip().lower() in ADMIN_LOGIN_USERNAMES
    if _who:
        st.markdown(
            f'<div style="color:{TEXT_SUB}; font-size:12px; margin-bottom:4px;">'
            f'Signed in as <b>{_who["name"]}</b>{" (full access)" if IS_ADMIN else ""}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Sign out", width='stretch'):
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

# ITAdmin sees a location instead of "N/A". The approved schema has no real geo field,
# so this reuses the same deterministic simulated-city lookup CSV mode already uses
# elsewhere - never computed as, or labeled, a real GPS/geolocation value.
if IS_ADMIN and DATA_SOURCE == "sqlite" and df["location"].eq("N/A").all():
    _admin_lookup = build_identity_lookup(
        tuple(sorted(df["user_id"].unique())), tuple(sorted(df["device_id"].unique()))
    )
    df["location"] = df["user_id"].map(_admin_lookup["user_city"])

# The loader always stores new accounts as user_type='employee' (it has no way to tell
# guests apart at ingest time); correct that here for any account whose id looks like a
# guest account (e.g. 'guest0007'), so Type is accurate everywhere it's shown.
df["user_type"] = np.where(
    df["user_id"].astype(str).str.lower().str.startswith("guest"), "guest", df["user_type"]
)
_is_guest = df["user_type"].astype(str).str.lower().eq("guest")
USER_ANON_MAP = build_user_anon_map(
    tuple(sorted(df.loc[~_is_guest, "user_id"].astype(str).unique())),
    tuple(sorted(df.loc[_is_guest, "user_id"].astype(str).unique())),
)
df["user_id_anon"] = df["user_id"].map(USER_ANON_MAP)

# Scenario (S01-S08) for every event: the generator label when there is one, otherwise indicative.
FEATURE_BASELINE = compute_feature_baseline(df[FEATURE_COLS + ["risk_level"]])
df = assign_scenarios(df)

DATASET_NOW = df["timestamp"].max()  # "current time" anchor = latest event in the CSV

# ==============================================================================
# SIDEBAR NAVIGATION + FILTERS
# ==============================================================================
with st.sidebar:    
    NAV_PAGES = ["🏠 Dashboard", "🔔 Alerts", "👥 Users & Devices", "📝 Activity Logs",
                 "📄 Reports", "\U0001F9EA Model Evaluation", "📊 System Performance", "⚙️ Settings"]
    if not IS_ADMIN:
        NAV_PAGES = [p for p in NAV_PAGES if "Model Evaluation" not in p]
    page = st.radio(
        "NAVIGATION",
        NAV_PAGES,
        label_visibility="visible",
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
    if IS_ADMIN:
        device_options = ["All Devices"] + sorted(df["device_id"].unique().tolist())
        device_filter = st.selectbox("Device", device_options)
    else:
        device_filter = "All Devices"  # the Device filter is ITAdmin-only (it names real device ids)

    apply_filters = st.button("Apply Filters", width='stretch', type="primary")



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
# DASHBOARD SECTION C - SCENARIO EVIDENCE TABLES (S02-S07, one per feature)
# ------------------------------------------------------------------------------
# One tab per single-feature scenario, each showing the evidence for the user of
# the current highest-risk session. Identities follow the same masking rules as
# the rest of the dashboard (ITAdmin sees full ids).
# ==============================================================================
def _disp_user(u):
    return u if IS_ADMIN else USER_ANON_MAP.get(u, u)


def _disp_device(d):
    return d if IS_ADMIN else mask_mac(d, keep=3)


def _fmt_ts(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _num(v, nd=1):
    v = float(v)
    return f"{int(v):,}" if v.is_integer() else f"{v:,.{nd}f}"


def _user_baseline(user_rows: pd.DataFrame, feat: str) -> dict:
    """Personal baseline from the user's own LOW-risk history (5+ events), else the population baseline."""
    normal = user_rows.loc[user_rows["risk_level"] == "LOW", feat].astype(float)
    if len(normal) >= 5:
        q = normal.quantile([0.05, 0.5, 0.95])
        return {"p05": float(q[0.05]), "median": float(q[0.5]), "p95": float(q[0.95]),
                "source": f"user history ({len(normal)} normal events)"}
    b = FEATURE_BASELINE[feat]
    return {"p05": b["p05"], "median": b["median"], "p95": b["p95"],
            "source": f"population baseline ({b['source']})"}


def _vs_normal(value, feat: str) -> str:
    """Status against the Table 3.4-2 normal range."""
    return "▲ Outside normal range" if float(value) > NORMAL_LIMITS[feat] else "Within normal range"


def _baseline_text(base: dict, unit: str = "") -> str:
    u = f" {unit}" if unit else ""
    return f"{_num(base['median'])}{u} (observed normal {_num(base['p05'])}–{_num(base['p95'])})"


def _chips(items):
    html = "".join(
        f'<span style="display:inline-block; margin:0 16px 6px 0; font-size:12px; color:{TEXT_SUB};">'
        f'{k}: <b style="color:{TEXT_MAIN};">{v}</b></span>' for k, v in items
    )
    st.markdown(f'<div style="margin:4px 0 6px 0;">{html}</div>', unsafe_allow_html=True)


def _scenario_heading(code: str, note: str):
    m = SCENARIOS[code]
    feat = m["feature"]
    st.markdown(
        f'<div style="font-size:12.5px; color:{TEXT_SUB}; margin:4px 0 6px 0;">'
        f'<b style="color:{TEXT_MAIN};">{code} – {m["name"]}</b> &middot; Feature {feat} '
        f'(normal ≤ {NORMAL_LIMITS[feat]} {FEATURE_META[feat]["unit"]}) &middot; Threat: {m["threat"]} '
        f'&middot; Expected risk band: {expected_badge(code)}<br>{note}</div>',
        unsafe_allow_html=True,
    )


def _evidence_table(code: str, table: pd.DataFrame, footnote: str = None):
    st.dataframe(table, width='stretch', hide_index=True,
                 height=min(38 + 35 * len(table), 470))
    if footnote:
        st.markdown(f'<div style="color:{TEXT_SUB}; font-size:11.5px; margin-top:-4px;">{footnote}</div>',
                    unsafe_allow_html=True)
    st.download_button(
        f"⬇️ Download {code} evidence (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"smart_byod_{code}_evidence.csv", mime="text/csv", key=f"dl_evidence_{code}",
    )


def _evidence_lf(ev, hist, score):
    b = _user_baseline(hist, "LF")
    _scenario_heading("S02", "Successful logins of the current user per hour, against the normal range.")
    _chips([("Peak LF", _num(ev["LF"].max())), ("User baseline", _baseline_text(b, "logins/h")),
            ("Baseline source", b["source"])])
    _evidence_table("S02", pd.DataFrame({
        "User ID": ev["user_id"].map(_disp_user),
        "Event Timestamp": ev["timestamp"].map(_fmt_ts),
        "Logins in Last Hour (LF)": ev["LF"].map(_num),
        "Deviation from User Baseline": (ev["LF"] - b["median"]).map(lambda d: f"{d:+,.1f}"),
        "Risk Score (0–100)": score,
        "Status": ev["LF"].map(lambda v: _vs_normal(v, "LF")),
    }))


def _evidence_fl(ev, hist, score):
    _scenario_heading("S03", f"Failed logins in the last 15 minutes. FL ≥ {FL_LOCKOUT_THRESHOLD} gives full "
                             f"identity risk (I = 1) and a lockout record; a Block/Isolate decision also locks.")

    def lockout(r):
        if r.action in BLOCK_ACTIONS:
            return f"LOCKED – session blocked/isolated by policy at {_fmt_ts(r.timestamp)}"
        if float(r.FL) >= FL_LOCKOUT_THRESHOLD:
            return f"LOCKOUT THRESHOLD REACHED ({_num(r.FL)} ≥ {FL_LOCKOUT_THRESHOLD}) – policy action: {r.action}"
        return "No lockout"

    locks = [lockout(r) for r in ev.itertuples()]
    _chips([("Peak failed attempts (15 min)", _num(ev["FL"].max())),
            ("Events with failures", f"{int((ev['FL'] > 0).sum())}"),
            ("Lockout records", f"{sum(l != 'No lockout' for l in locks)}"),
            ("Identity risk I", f"{min(float(ev['FL'].max()) / FL_LOCKOUT_THRESHOLD, 1):.2f} at peak")])
    _evidence_table("S03", pd.DataFrame({
        "Failed Login Log": [f"{_fmt_ts(r.timestamp)} – {_num(r.FL)} failed login attempt(s) in 15 min, "
                             f"session {r.session_id}" for r in ev.itertuples()],
        "Source IP Address": ev["ip_address"].astype(str),
        "Device Information": [f"{_disp_device(r.device_id)} ({r.user_type})" for r in ev.itertuples()],
        "Failed Attempts (FL)": ev["FL"].map(_num),
        "Account Lockout Record": locks,
    }))


def _device_status_text(row) -> str:
    status = str(getattr(row, "device_status", "") or "").lower()
    if status == "registered":
        return "Registered to a user (inventory)"
    if status == "learned":
        return "Learned (used by the user > 24 h)"
    code = str(row.device_id)
    if code.startswith("UNKNOWN-") or (len(code) == 17 and code[1:2].upper() in list("26AE")):
        return "Unclassified (randomized MAC)"
    return "Not in the inventory"


def _evidence_ds(ev, hist, score):
    status = [_device_status_text(r) for r in ev.itertuples()]
    devices = ev["device_id"].dropna().unique().tolist()
    shown = ", ".join(_disp_device(d) for d in devices[:4]) + (" …" if len(devices) > 4 else "")
    _scenario_heading("S04", "Devices that authenticated under the current user's account. DS counts new or "
                             "unregistered devices per hour; a registered device with a randomized MAC or a "
                             "new DHCP lease is not counted.")
    _chips([("Distinct devices", f"{len(devices)} ({shown})"),
            ("Not registered / not learned", f"{sum(s == 'Not in the inventory' for s in status)} event(s)"),
            ("Peak DS", _num(ev["DS"].max()))])
    _evidence_table("S04", pd.DataFrame({
        "Event Timestamp": ev["timestamp"].map(_fmt_ts),
        "Device ID": ev["device_id"].map(_disp_device),
        "Device Status": status,
        "Session ID": ev["session_id"].astype(str),
        "New/Unregistered Devices in Last Hour (DS)": ev["DS"].map(_num),
        "Status": ev["DS"].map(lambda v: _vs_normal(v, "DS")),
    }), footnote="Device risk D (Table 3.5-6): 0 registered or learned, 0.2 unclassified, 0.6 not in the "
                 "inventory, 1.0 another user's device or DS ≥ 2.")


def _evidence_ltd(ev, hist, score):
    b = _user_baseline(hist, "LTD")
    normal_hist = hist[hist["risk_level"] == "LOW"]
    pattern_src = normal_hist if len(normal_hist) >= 5 else hist
    hours = pattern_src["timestamp"].dt.hour
    typical = (f"{int(hours.quantile(0.10)):02d}:00–{int(hours.quantile(0.90)):02d}:59"
               if len(hours) else "N/A")
    _scenario_heading("S05", "Login times of the current user against the user's baseline (organization working "
                             "hours until the user has 10 logins). Contextual risk C rises from 0 at 60 min to 1 at 180 min.")
    _chips([("Typical activity hours (approx.)", typical), ("Max deviation", f"{_num(ev['LTD'].max())} min"),
            ("Pattern built from", f"{len(pattern_src)} events")])
    _evidence_table("S05", pd.DataFrame({
        "Timestamp Record": ev["timestamp"].map(_fmt_ts),
        "Hour": ev["timestamp"].dt.strftime("%H:%M"),
        "Historical Pattern (typical hours)": typical,
        "Login-Time Deviation (min)": ev["LTD"].map(_num),
        "C_time": ev["LTD"].map(lambda v: f"{min(max((float(v) - 60) / 120, 0), 1):.2f}"),
        "Status": ev["LTD"].map(lambda v: _vs_normal(v, "LTD")),
    }))
    counts = hours.value_counts().reindex(range(24), fill_value=0)
    cur_hour = int(ev["timestamp"].iloc[-1].hour)
    colors = [BRAND_RED if h == cur_hour else "#334155" for h in range(24)]
    fig = go.Figure(go.Bar(x=[f"{h:02d}" for h in range(24)], y=counts.values, marker_color=colors,
                           hovertemplate="%{x}:00 – <b>%{y}</b> events<extra></extra>"))
    _style_fig(fig, 180)
    fig.update_xaxes(title_text="Hour of day (red = current event)", title_font=dict(size=11))
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False}, key="s05_login_pattern")


def _evidence_raf(ev, hist, score):
    b = _user_baseline(hist, "RAF")
    has_res = "resource" in ev.columns and ev["resource"].notna().any()
    res = ev["resource"].fillna("—").astype(str) if has_res else pd.Series("Not recorded in dataset", index=ev.index)
    _scenario_heading("S06", "Distinct internal resources (destination host and service) reached by the current "
                             "user in 5 minutes; repeat connections to the same resource count once.")
    _chips([("Peak distinct resources (5 min)", _num(ev["RAF"].max())),
            ("Distinct resources shown", f"{res[res != '—'].nunique()}" if has_res else "N/A"),
            ("User baseline", _baseline_text(b, "resources"))])
    _evidence_table("S06", pd.DataFrame({
        "Timestamp": ev["timestamp"].map(_fmt_ts),
        "Resource (host:service)": res.values,
        "Distinct Resources in 5 min (RAF)": ev["RAF"].map(_num),
        "Status": ev["RAF"].map(lambda v: _vs_normal(v, "RAF")),
        "User Activity Log": [
            f"{_fmt_ts(r.timestamp)} user={_disp_user(r.user_id)} device={_disp_device(r.device_id)} "
            f"resource={x} distinct_5min={_num(r.RAF)} risk={r.risk_level} action={r.action}"
            for r, x in zip(ev.itertuples(), res)
        ],
    }), footnote=None if has_res else "The CSV has no resource field; resource identifiers appear with live (SQLite) data.")


def _evidence_aaf(ev, hist, score):
    b = _user_baseline(hist, "AAF")
    _scenario_heading("S07", "Application/API (app-ctrl) events of the current user in 5 minutes.")
    _chips([("Peak AAF (5 min)", _num(ev["AAF"].max())), ("Mean per event", _num(ev["AAF"].mean())),
            ("User baseline", _baseline_text(b, "events"))])
    _evidence_table("S07", pd.DataFrame({
        "API Transaction Log": [f"{_fmt_ts(r.timestamp)} session={r.session_id} app_events_5min={_num(r.AAF)} "
                                f"risk={r.risk_level}" for r in ev.itertuples()],
        "Endpoint (Device)": ev["device_id"].map(_disp_device),
        "Application/API Events in 5 min (AAF)": ev["AAF"].map(_num),
        "Deviation from User Baseline": (ev["AAF"] - b["median"]).map(lambda d: f"{d:+,.1f}"),
        "Status": ev["AAF"].map(lambda v: _vs_normal(v, "AAF")),
    }))


EVIDENCE_RENDERERS = {
    "S02": _evidence_lf, "S03": _evidence_fl, "S04": _evidence_ds,
    "S05": _evidence_ltd, "S06": _evidence_raf, "S07": _evidence_aaf,
}


VAPT_USER_PREFIX = "vapt_"      # test accounts created by SMART-BYOD_Analyze.py --vapt-run


def vapt_records() -> pd.DataFrame:
    """Generator-labeled records of the VAPT test accounts (empty when none are stored)."""
    if "scenario_source" not in df.columns:
        return df.iloc[0:0]
    m = df["scenario_source"].astype(str).str.startswith("Generator") & \
        df["user_id"].astype(str).str.startswith(VAPT_USER_PREFIX)
    return df[m]


def vapt_verdict(code: str, sub: pd.DataFrame):
    """Same rule as SMART-BYOD_Analyze.py: an attack event (scenario feature outside its normal range) is
    flagged or raises an alert, and the peak risk of the attack events reaches the expected band (High or
    Critical for S08). S01 passes when no event raises an alert."""
    lo, hi = SCENARIOS[code]["band"]
    if code == "S01":
        alerts = int((sub["risk_level"] != "LOW").sum())
        return ("PASS" if alerts == 0 else "FAIL"), f"{alerts} false alert(s) on normal activity"
    feats = [SCENARIOS[code]["feature"]] if SCENARIOS[code]["feature"] else ["DS", "FL", "LTD", "RAF"]
    attack = sub[np.any([sub[f].astype(float) > NORMAL_LIMITS[f] for f in feats], axis=0)]
    if attack.empty:
        return "FAIL", "attack pattern not reproduced (no feature outside its normal range)"
    detected = attack[(attack["predicted_anomaly"] == 1) | (attack["risk_level"] != "LOW")]
    peak = attack.loc[attack["risk_score"].idxmax()]
    level_ok = peak["risk_level"] in ("HIGH", "CRITICAL") if code == "S08" else float(peak["risk_score"]) >= lo
    ok = len(detected) > 0 and level_ok
    return ("PASS" if ok else "FAIL"), (f"{len(detected)} of {len(attack)} attack event(s) detected; peak R "
                                        f"{float(peak['risk_score']):.1f} ({peak['risk_level']}, "
                                        f"{'flagged' if int(peak['predicted_anomaly']) == 1 else 'not flagged'}); "
                                        f"expected band {band_text(code)}")


def _vapt_banner(code: str, sub: pd.DataFrame):
    verdict, why = vapt_verdict(code, sub)
    color = "#22C55E" if verdict == "PASS" else "#EF4444"
    st.markdown(
        f'<div class="metric-card" style="margin:2px 0 10px 0; border-left:4px solid {color};">'
        f'<div class="mc-label">VAPT test case VAPT-{code} &middot; {SCENARIOS[code]["name"]} &middot; '
        f'threat: {SCENARIOS[code]["threat"]}</div>'
        f'<div style="font-size:15px; font-weight:800; color:{color};">{verdict}</div>'
        f'<div style="font-size:12px; color:{TEXT_MAIN};">{why}</div>'
        f'<div style="font-size:11.5px; color:{TEXT_SUB};">{len(sub):,} test record(s), '
        f'{int(sub["predicted_anomaly"].sum()):,} flagged, {int((sub["risk_level"] != "LOW").sum()):,} alert(s); '
        f'decisions: {", ".join(f"{k} {v}" for k, v in sub["action"].value_counts().items())}</div></div>',
        unsafe_allow_html=True,
    )


def _evidence_generic(code: str, ev: pd.DataFrame):
    """S01 / S08 evidence: all six features with the decision and its indicators."""
    _scenario_heading_generic(code)
    table = pd.DataFrame({"Timestamp": ev["timestamp"].map(_fmt_ts), "User": ev["user_id"].map(_disp_user),
                          "Device": ev["device_id"].map(_disp_device)})
    for f in FEATURE_COLS:
        table[f] = ev[f].map(_num)
    table["Risk"] = ev["risk_score"].map(lambda v: f"{float(v):,.1f}")
    table["Level"] = ev["risk_level"]
    table["Decision"] = ev["action"]
    table["Indicators"] = ev["indicators"] if "indicators" in ev.columns else ""
    _evidence_table(code, table)


def _scenario_heading_generic(code: str):
    m = SCENARIOS[code]
    st.markdown(
        f'<div style="font-size:12.5px; color:{TEXT_SUB}; margin:4px 0 6px 0;">'
        f'<b style="color:{TEXT_MAIN};">{code} – {m["name"]}</b> &middot; Threat: {m["threat"]} '
        f'&middot; Expected risk band: {expected_badge(code)}</div>',
        unsafe_allow_html=True,
    )


def render_scenario_evidence(top, user_hist: pd.DataFrame):
    """Section C evidence tabs: the user of the current highest-risk session, or the VAPT test records."""
    vapt = vapt_records()
    source = "Current highest-risk session"
    if not vapt.empty:
        source = st.radio("Evidence source", ["Current highest-risk session", "VAPT test records (S01–S08)"],
                          horizontal=True, key="evidence_source")
    if source.startswith("VAPT"):
        codes = [c for c in SCENARIO_ORDER if (vapt["scenario_code"] == c).any()]
        passed = sum(vapt_verdict(c, vapt[vapt["scenario_code"] == c])[0] == "PASS" for c in codes)
        st.markdown(
            f'<div style="color:{TEXT_SUB}; font-size:12px; margin:-6px 0 8px 0;">'
            f'VAPT test records injected by <code>SMART-BYOD_Analyze.py --vapt-run</code> '
            f'(accounts {VAPT_USER_PREFIX}s01 … {VAPT_USER_PREFIX}s08). '
            f'<b style="color:{TEXT_MAIN};">{passed} of {len(codes)} test case(s) passed.</b> '
            f'Each tab shows the last {EVIDENCE_ROWS} events of the case.</div>',
            unsafe_allow_html=True,
        )
        tabs = st.tabs([f"VAPT-{c}" + (f" · {SCENARIOS[c]['feature']}" if SCENARIOS[c]["feature"] else "") for c in codes])
        for tab, code in zip(tabs, codes):
            with tab:
                sub = vapt[vapt["scenario_code"] == code].sort_values("timestamp")
                _vapt_banner(code, sub)
                ev = sub.tail(EVIDENCE_ROWS).reset_index(drop=True)
                if code in EVIDENCE_RENDERERS:
                    hist = df[df["user_id"].isin(sub["user_id"].unique())].sort_values("timestamp")
                    EVIDENCE_RENDERERS[code](ev, hist, ev["risk_score"].map(lambda v: f"{float(v):,.1f}"))
                else:
                    _evidence_generic(code, ev)
        return

    ev = user_hist[user_hist["timestamp"] <= top["timestamp"]].tail(EVIDENCE_ROWS)
    if len(ev) < 2:
        ev = user_hist.tail(EVIDENCE_ROWS)
    ev = ev.sort_values("timestamp").reset_index(drop=True)
    score = ev["risk_score"].map(lambda v: f"{float(v):,.1f}")
    st.markdown(
        f'<div style="color:{TEXT_SUB}; font-size:12px; margin:-6px 0 8px 0;">'
        f'User {_disp_user(top["user_id"])} &middot; last {len(ev)} event(s) up to '
        f'{_fmt_ts(top["timestamp"])}. Status compares each value with the Table 3.4-2 normal range.</div>',
        unsafe_allow_html=True,
    )
    tabs = st.tabs([f"{c} · {SCENARIOS[c]['feature']}" for c in EVIDENCE_SCENARIOS])
    for tab, code in zip(tabs, EVIDENCE_SCENARIOS):
        with tab:
            EVIDENCE_RENDERERS[code](ev, user_hist, score)


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
        st.button("\U0001F504 Refresh", width='stretch')

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

    # measured per-event decision time (risk + policy) saved by the analyzer; not available for the CSV
    avg_resp_time = (float(cur_window["response_time_ms"].mean())
                     if "response_time_ms" in cur_window.columns and cur_window["response_time_ms"].notna().any() else None)

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
        metric_card(r2[3], "\u23F1\uFE0F", "Avg. Decision Time",
                    "—" if avg_resp_time is None else f"{avg_resp_time:,.2f} ms")
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
            ("User ID", top["user_id"] if IS_ADMIN else USER_ANON_MAP.get(top["user_id"], top["user_id"])),
            ("Device ID", top["device_id"] if IS_ADMIN else mask_mac(top["device_id"], keep=3)),
            ("IP Address", top["ip_address"] if IS_ADMIN else mask_ip(top["ip_address"])),
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
            st.plotly_chart(gauge, width='stretch', config={"displayModeBar": False})
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

        # ---- Risk components (Table 3.5-6): points each one adds to R ----------
        if "comp_A" in top.index and pd.notna(top.get("comp_A")):
            pts = [("A Anomaly", W_A * float(top["comp_A"])), ("I Identity", W_I * float(top["comp_I"])),
                   ("D Device", W_D * float(top["comp_D"])), ("C Context", W_C * float(top["comp_C"]))]
            chips = " &nbsp;·&nbsp; ".join(f'{n}: <b style="color:{TEXT_MAIN};">{v:,.1f}</b>' for n, v in pts)
            st.markdown(f'<div style="font-size:12px; color:{TEXT_SUB}; margin-top:6px;">Points added to R: '
                        f'{chips} &nbsp;=&nbsp; <b style="color:{TEXT_MAIN};">{sum(v for _, v in pts):,.1f}</b>'
                        f'<br>Main driver: {top.get("driver", "")}</div>', unsafe_allow_html=True)

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
        sc_code = top.get("scenario_code", "S01")
        sc_meta = SCENARIOS.get(sc_code, SCENARIOS["S01"])
        st.markdown(
            f'<div style="font-size:12.5px; color:{TEXT_SUB}; margin:-4px 0 10px 0;">'
            f'Detected scenario: <b style="color:{TEXT_MAIN};">{sc_code} – {sc_meta["name"]}</b> '
            f'<span style="font-size:11px;">({top.get("scenario_source", "Indicative - inferred from features")}, '
            f'{int(top.get("features_above_baseline", 0))} feature(s) outside the Table 3.4-2 normal range)</span><br>'
            f'Expected band {expected_badge(sc_code)} &nbsp;&middot;&nbsp; '
            f'Observed {risk_badge(top["risk_level"], RISK_COLORS)} '
            f'<span style="color:{TEXT_MAIN};">{float(top["risk_score"]):,.1f}</span></div>',
            unsafe_allow_html=True,
        )
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
                    width='stretch', config={"displayModeBar": False},
                    key=f"spark_{feat}",
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with right2:
        st.markdown(
            '<div class="panel"><div class="panel-title">D. ALERTS <span class="sub">(Recent)</span></div>',
            unsafe_allow_html=True,
        )
        if str(top["risk_level"]).upper() != "LOW":
            render_alert_card(top)          # current highest-risk alert, Figure 3.3-7 layout
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
                row_user = row["user_id"] if IS_ADMIN else USER_ANON_MAP.get(row["user_id"], row["user_id"])
                row_device = row["device_id"] if IS_ADMIN else mask_mac(row["device_id"], keep=3)
                alert_rows += (
                    f'<tr style="border-bottom:1px solid {PANEL_BORDER};">'
                    f'<td style="padding:6px 4px; color:{TEXT_SUB};">{row['timestamp'].strftime('%H:%M:%S')}</td>'
                    f'<td style="padding:6px 4px; color:{TEXT_MAIN};">{row_user}</td>'
                    f'<td style="padding:6px 4px; color:{TEXT_MAIN};">{row_device}</td>'
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

    # ---- C. BEHAVIORAL ACTIVITY - SCENARIO EVIDENCE (S02-S07) ---------------
    st.markdown(
        '<div class="panel"><div class="panel-title">C. BEHAVIORAL ACTIVITY – SCENARIO EVIDENCE '
        '<span class="sub">(Current Session S02–S07, or VAPT test records S01–S08)</span></div>',
        unsafe_allow_html=True,
    )
    render_scenario_evidence(top, user_hist)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- E. SYSTEM PERFORMANCE (measured) -------------------------------------
    st.markdown(
        '<div class="panel"><div class="panel-title">E. SYSTEM PERFORMANCE <span class="sub">(Measured)</span></div>',
        unsafe_allow_html=True,
    )
    render_system_performance(compact=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# SYSTEM PERFORMANCE (shared by Dashboard + dedicated page) - measured values only
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_system_metrics(db_path: str, signature: tuple) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.OperationalError:
        conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        return pd.read_sql_query("SELECT * FROM system_metrics ORDER BY metric_id DESC LIMIT 200", conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_system_metrics() -> dict:
    """Latest analyzer measurement (system_metrics), the host's CPU/memory through psutil, and this
    dashboard's own page-render time. A value that was not measured is None, never simulated."""
    out = {"cpu": None, "mem": None, "detect_ms": None, "resp_ms": None, "risk_update_s": None,
           "dash_refresh": st.session_state.get("last_refresh_s"),
           "concurrent": int(fdf["session_id"].nunique()), "source": "not measured"}
    db_path = find_file(DB_FILENAME) if DATA_SOURCE == "sqlite" else None
    sm = load_system_metrics(str(db_path), db_signature(db_path)) if db_path is not None else pd.DataFrame()
    if not sm.empty:
        last = sm.iloc[0]
        val = lambda c: (None if c not in sm.columns or pd.isna(last[c]) else float(last[c]))
        out.update(cpu=val("cpu_utilization"), mem=val("memory_utilization"), detect_ms=val("detection_latency_ms"),
                   resp_ms=val("response_time_ms"),
                   risk_update_s=(val("risk_update_interval_ms") / 1000 if val("risk_update_interval_ms") is not None else None),
                   source=f"analyzer run of {str(last['measurement_timestamp'])[:19]}")
    if (out["cpu"] is None or out["mem"] is None) and psutil is not None:
        try:
            out["cpu"] = out["cpu"] if out["cpu"] is not None else psutil.cpu_percent(interval=0.1)
            out["mem"] = out["mem"] if out["mem"] is not None else psutil.virtual_memory().percent
            out["source"] += " · CPU/memory: this host (psutil)"
        except Exception:
            pass
    return out


def _perf_note() -> str:
    """Alert response and risk-update interval come from the latest performance test (PT-01, PT-02)."""
    a, r = latest_perf_result("PT-01"), latest_perf_result("PT-02")
    if a is None and r is None:
        return "Alert response time and risk-update interval are measured by the performance test (--perf-test). "
    part = lambda name, v: f"{name}: {v[0]} ({v[1]})" if v else ""
    return "Latest performance test - " + "; ".join(x for x in (part("alert response", a), part("risk update", r)) if x) + ". "


def render_system_performance(compact=False):
    live = get_system_metrics()
    fmt = lambda v, f: "—" if v is None else f.format(v)
    perf_metrics = [
        ("CPU Utilization", fmt(live["cpu"], "{:,.0f}%"), "Target ≤ 50%"),
        ("Memory Utilization", fmt(live["mem"], "{:,.0f}%"), "Target ≤ 60%"),
        ("Detection Latency", fmt(live["detect_ms"], "{:,.3f} ms/event"), "Isolation Forest scoring"),
        ("Decision Time", fmt(live["resp_ms"], "{:,.2f} ms/event"), "Risk + policy per event"),
        ("Dashboard Refresh", fmt(live["dash_refresh"], "{:,.2f} s"), "Target ≤ 3 s"),
        ("Active Sessions", f"{live['concurrent']}", "In the filtered data"),
    ]
    cols = st.columns(6)
    for c, (label, val, target) in zip(cols, perf_metrics):
        c.markdown(
            f'<div class="metric-card"><div class="mc-label">{label}</div>'
            f'<div class="mc-value" style="font-size:21px;">{val}</div>'
            f'<div style="font-size:11px; color:{TEXT_SUB};">{target}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div style="color:{TEXT_SUB}; font-size:12px; margin-top:8px;">Source: {live["source"]}. '
        f'Dashboard refresh = render time of the previous page load. '
        f'{_perf_note()}'
        f'{"" if compact else " Values marked — were not measured; nothing on this panel is simulated."}</div>',
        unsafe_allow_html=True,
    )


def white_caption(text: str):
    st.markdown(
        f'<div style="color:#FFFFFF; font-size:12.5px; margin-top:6px;">{text}</div>',
        unsafe_allow_html=True,
    )


def measure_pipeline_timing(raw_data: pd.DataFrame, contam: float) -> dict:
    """Wall-clock cost (ms) of each pipeline stage on the loaded data, timed fresh on this run:
    collection, preprocessing, the six features, Isolation Forest, the risk engine (Table 3.5-6)
    and the policy engine (Table 3.5-7)."""
    timings = {}

    t0 = time.perf_counter()
    collected = raw_data.copy()
    timings["Collection"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    collected["timestamp"] = pd.to_datetime(collected["timestamp"])
    collected = collected.sort_values("timestamp").reset_index(drop=True)
    timings["Preprocessing"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    X_stage = collected[FEATURE_COLS].astype(float).fillna(0.0).values
    timings["Features"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    stage_model = IsolationForest(n_estimators=100, contamination=contam, random_state=42).fit(X_stage)
    s = -stage_model.score_samples(X_stage)
    _ = stage_model.predict(X_stage)
    timings["IsolationForest"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    A = anomaly_component(s, float(-stage_model.offset_), float(s.min()), float(s.max()))
    I, D, C = components_from_features(collected)
    R = 100.0 * (W_A * A + W_I * I + W_D * D + W_C * C)
    levels = risk_level_of(R)
    timings["RiskEngine"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = pd.Series(levels).map(ACTION_BY_LEVEL)
    timings["PolicyEngine"] = (time.perf_counter() - t0) * 1000

    # Floor at 0.1ms so a near-instant stage is still visible on the bar chart.
    return {k: max(v, 0.1) for k, v in timings.items()}

# ==============================================================================
# PAGE: MODEL EVALUATION (thesis figures — Chapter 4)
# ==============================================================================
def page_model_evaluation():
    st.markdown('<div class="app-title">\U0001F9EA Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Isolation Forest detection, risk-engine bands (Table 3.5-14), calibration '
        'and the six behavioral features. Only records with a generator label are graded; live FortiGate '
        'records carry no ground truth.</div>',
        unsafe_allow_html=True,
    )

    dark_mpl_style = {
        "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
        "axes.edgecolor": PANEL_BORDER, "axes.labelcolor": TEXT_MAIN, "axes.titlecolor": TEXT_MAIN,
        "text.color": TEXT_MAIN, "xtick.color": TEXT_SUB, "ytick.color": TEXT_SUB,
        "xtick.labelcolor": TEXT_SUB, "ytick.labelcolor": TEXT_SUB, "grid.color": PANEL_BORDER,
        "legend.facecolor": PANEL, "legend.edgecolor": PANEL_BORDER, "legend.labelcolor": TEXT_MAIN,
        "font.size": 10,
    }
    sns.set_theme(style="darkgrid")
    plt.rcParams.update(dark_mpl_style)

    full = df

    # ---- which labeled records are graded -----------------------------------
    lab, SOURCE_NOTE = None, None
    if "ground_truth" in full.columns and full["ground_truth"].notna().any():
        pool = fdf if fdf["ground_truth"].notna().any() else full
        lab = pool[pool["ground_truth"].notna()].copy()
        SOURCE_NOTE = ("Graded on the labeled CSV, scored in the app with the analyzer's method."
                       if DATA_SOURCE == "csv" else
                       "Graded on generator-labeled records scored by the analyzer. The live model is fitted on "
                       "normal-range records, so there is no held-out partition here; use "
                       "'SMART-BYOD_Analyze.py --evaluate-csv' for the held-out test metrics.")
    else:
        bench_path = find_file(CSV_FILENAME)
        if bench_path is not None:
            try:
                bench_raw = load_data(str(bench_path))
                if "ground_truth" in bench_raw.columns:
                    st.markdown(f'<div style="color:{TEXT_SUB}; font-size:12px; margin-bottom:4px;">'
                                f'Contamination for the benchmark evaluation</div>', unsafe_allow_html=True)
                    bench_cont = st.slider(" ", min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                           key="bench_contamination")
                    lab = score_risk(bench_raw, contamination=bench_cont)
                    SOURCE_NOTE = (f"The data shown elsewhere has no ground truth, so these figures use the labeled "
                                   f"benchmark {CSV_FILENAME} ({len(lab):,} records), scored with the analyzer's method.")
            except Exception:
                lab = None
    HAS_LABELS = lab is not None and len(lab) > 0
    if HAS_LABELS and "partition" in lab.columns:
        choice = st.radio("Evaluate on", ["Held-out test partition", "Full dataset (includes training records - optimistic)"],
                          horizontal=True, key="eval_partition")
        if choice.startswith("Held-out"):
            lab = lab[lab["partition"] == "test"]
    if HAS_LABELS:
        lab["ground_truth"] = lab["ground_truth"].astype(int)
        HAS_LABELS = len(lab) > 0
    LABELS_NOTE = ("No ground truth is available: live FortiGate records are never labeled (labeling rule 4). "
                   f"Load generator-labeled data through the analyzer, or place {CSV_FILENAME} next to this script.")
    nine_note = ("This dataset comes from the nine-feature version: only the six current features are used, but its "
                 "DS and RAF follow the old definitions. Regenerate it before reporting thesis results.")

    def panel_start(title):
        st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>', unsafe_allow_html=True)

    def panel_end():
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Figure 4.3-1: Feature Distribution Histograms -----------------------
    panel_start("&mdash; Feature Distribution Histograms")
    st.markdown('<div class="app-subtitle">Login Frequency (LF), Failed Login Attempts (FL), Resource-Access '
                'Frequency (RAF, distinct resources per 5 min) and Application/API Activity Frequency (AAF).</div>',
                unsafe_allow_html=True)
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 6))
    for ax, feat in zip(axes1.flatten(), ["LF", "FL", "RAF", "AAF"]):
        sns.histplot(full[feat], bins=30, kde=True, ax=ax, color=BRAND_RED)
        ax.set_title(f"{feat} — {FEATURE_META[feat]['label']}")
    fig1.tight_layout()
    st.pyplot(fig1, width='stretch')
    plt.close(fig1)
    panel_end()

    if HAS_LABELS:
        st.info(SOURCE_NOTE)
        if is_nine_feature_frame(lab):
            st.warning(nine_note)
        y_eval, pred_eval, score_eval = lab["ground_truth"], lab["predicted_anomaly"].astype(int), lab["anomaly_score"].astype(float)
        tp = int(((y_eval == 1) & (pred_eval == 1)).sum()); tn = int(((y_eval == 0) & (pred_eval == 0)).sum())
        fp = int(((y_eval == 0) & (pred_eval == 1)).sum()); fn = int(((y_eval == 1) & (pred_eval == 0)).sum())

        # ---- Tables 4.4-2 / 4.4-3: metrics with Wilson intervals and targets ----
        panel_start("&mdash; Detection Metrics against the Acceptance Criteria")
        P = tp / (tp + fp) if tp + fp else 0.0
        Rc = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * Rc / (P + Rc) if P + Rc else 0.0
        FPR = fp / (fp + tn) if fp + tn else 0.0
        ACC = (tp + tn) / max(len(lab), 1)
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        MCC = (tp * tn - fp * fn) / denom if denom else 0.0
        AUC = roc_auc_score(y_eval, score_eval) if y_eval.nunique() > 1 else float("nan")
        ci = lambda k, m: "—" if m == 0 else "[{:.3f}, {:.3f}]".format(*wilson_interval(k, m))
        rows = [("Accuracy", ACC, "≥ 0.85", ACC >= 0.85, ci(tp + tn, len(lab))),
                ("Precision", P, "≥ 0.80", P >= 0.80, ci(tp, tp + fp)),
                ("Recall", Rc, "≥ 0.80", Rc >= 0.80, ci(tp, tp + fn)),
                ("F1-score", F1, "≥ 0.80", F1 >= 0.80, "—"),
                ("ROC-AUC", AUC, "≥ 0.85", AUC >= 0.85, "—"),
                ("False-positive rate", FPR, "≤ 0.10", FPR <= 0.10, ci(fp, fp + tn)),
                ("Balanced accuracy", (Rc + 1 - FPR) / 2, "(reported)", None, "—"),
                ("MCC", MCC, "(reported)", None, "—")]
        st.dataframe(pd.DataFrame([{"Metric": n, "Value": round(float(v), 4), "Target": t,
                                    "Status": "—" if p is None else ("Met" if p else "Not met"), "95% Wilson CI": c}
                                   for n, v, t, p, c in rows]), width='stretch', hide_index=True)
        st.caption(f"{len(lab):,} records graded: {tn + fp:,} normal, {tp + fn:,} anomalous. "
                   f"TN {tn:,} · FP {fp:,} · FN {fn:,} · TP {tp:,}.")
        if (tp + fn) == 0 or (tn + fp) == 0:
            st.warning("This selection lacks one of the classes, so precision, recall and ROC-AUC are not meaningful.")
        panel_end()

        # ---- Scenario detection and Table 3.5-14 bands ------------------------
        if "scenario" in lab.columns:
            panel_start("&mdash; Scenario Results (S01–S08) and Expected Risk Bands (Table 3.5-14)")
            codes = scenario_codes(lab)
            det_rows = []
            for code in SCENARIO_ORDER:
                sub = lab[codes == code]
                if sub.empty:
                    continue
                n, flagged = len(sub), int(sub["predicted_anomaly"].sum())
                lo, hi = SCENARIOS[code]["band"]
                inband = float(((sub["risk_score"] >= lo) & (sub["risk_score"] <= hi)).mean())
                det_rows.append({
                    "Code": code, "Scenario": SCENARIOS[code]["name"], "Records": n,
                    "Flagged": flagged, "Rate": f"{flagged / n:.2%}",
                    "Metric": "False-positive rate" if code == "S01" else "Detection rate",
                    "Mean Risk": round(float(sub["risk_score"].mean()), 1),
                    "Expected Band": band_text(code), "Within Band": f"{inband:.1%}",
                    "Match": "Yes" if inband >= 0.5 else "No",
                })
            st.dataframe(pd.DataFrame(det_rows), width='stretch', hide_index=True)
            missing = [c for c in SCENARIO_ORDER if c not in set(codes.dropna())]
            dropped = int(codes.isna().sum())
            note = ("Detection rate = share of a scenario's records flagged by the Isolation Forest (for S01 it is the "
                    "false-positive rate). Within Band = share of records whose risk score falls in the expected band.")
            if missing:
                note += f" No records for: {', '.join(missing)}."
            if dropped:
                note += f" {dropped:,} record(s) belong to removed scenarios (SD/IPC/RA) or unknown codes."
            st.markdown(f'<div style="color:{TEXT_SUB}; font-size:12px;">{note}</div>', unsafe_allow_html=True)
            panel_end()

        # ---- Calibration check ------------------------------------------------
        panel_start("&mdash; Calibration Check (normal records in the Low band)")
        normal = lab[lab["ground_truth"] == 0]
        low_share = float((normal["risk_level"] == "LOW").mean()) if len(normal) else float("nan")
        ok = low_share >= CALIBRATION_TARGET_LOW
        st.markdown(f'<div style="font-size:14px;">{low_share:.1%} of {len(normal):,} normal records are Low '
                    f'(target ≥ {CALIBRATION_TARGET_LOW:.0%}) → <b style="color:{"#22C55E" if ok else "#EF4444"};">'
                    f'{"PASS" if ok else "FAIL"}</b></div>', unsafe_allow_html=True)
        panel_end()

        # ---- Figure 4.3-3: Boxplots — Normal vs. Anomalous ----------------------
        panel_start("&mdash; Feature Distributions: Normal vs. Anomalous (Ground Truth)")
        st.markdown('<div class="app-subtitle">LF, FL, RAF and AAF for normal and anomalous records.</div>',
                    unsafe_allow_html=True)
        fig3, axes3 = plt.subplots(2, 2, figsize=(10, 6))
        for ax, feat in zip(axes3.flatten(), ["LF", "FL", "RAF", "AAF"]):
            sns.boxplot(x="ground_truth", y=feat, data=lab, ax=ax, hue="ground_truth",
                        palette={0: BRAND_TEAL, 1: BRAND_RED}, legend=False)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Normal (0)", "Anomalous (1)"])
            ax.set_title(f"{feat} — {FEATURE_META[feat]['label']}")
        fig3.tight_layout()
        st.pyplot(fig3, width='stretch')
        plt.close(fig3)
        panel_end()
    else:
        st.info(LABELS_NOTE)

    col_a, col_b = st.columns(2)

    # ---- Figure 4.3-2: Correlation Heatmap (6 x 6) ----------------------------
    with col_a:
        panel_start("&mdash; Correlation Heatmap (Six Behavioral Features)")
        st.markdown('<div class="app-subtitle">Pearson correlation coefficients among LF, FL, DS, LTD, RAF and AAF.</div>',
                    unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        corr = full[FEATURE_COLS].astype(float).corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2, vmin=-1, vmax=1,
                    cbar_kws={"label": "Correlation"}, annot_kws={"size": 8})
        fig2.tight_layout()
        st.pyplot(fig2, width='stretch')
        plt.close(fig2)
        strongest = corr.where(~np.eye(len(FEATURE_COLS), dtype=bool)).abs().stack().idxmax()
        white_caption(f"Strongest pair in this data: {strongest[0]}–{strongest[1]} "
                      f"(r = {corr.loc[strongest]:.2f}).")
        panel_end()

    # ---- Figure 4.5-1: Risk Level Distribution --------------------------------
    with col_b:
        panel_start("&mdash; Risk Level Distribution")
        st.markdown('<div class="app-subtitle">Distribution of Risk Levels Generated by the SMART-BYOD Risk Engine.</div>',
                    unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(5, 5))
        order4 = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        counts4 = full["risk_level"].value_counts().reindex(order4).fillna(0)
        ax4.bar(counts4.index, counts4.values, color=[RISK_COLORS[l] for l in order4])
        ax4.set_ylabel("Count")
        for i, cou in enumerate(counts4):
            ax4.text(i, cou + 0.02, f"{int(cou):,}", ha="center", color=TEXT_MAIN)
        fig4.tight_layout()
        st.pyplot(fig4, width='stretch')
        plt.close(fig4)
        white_caption("R = 100 (0.40 A + 0.25 I + 0.20 D + 0.15 C), bands of Table 3.5-7.")
        panel_end()

    if HAS_LABELS:
        col_c, col_d = st.columns(2)
        with col_c:
            panel_start("&mdash; Confusion Matrix (Isolation Forest vs. Ground Truth)")
            cm = confusion_matrix(y_eval, pred_eval, labels=[0, 1])
            fig5, ax5 = plt.subplots(figsize=(5, 5))
            ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomalous"]).plot(cmap="Reds", ax=ax5, colorbar=False)
            ax5.grid(False)
            fig5.tight_layout()
            st.pyplot(fig5, width='stretch')
            plt.close(fig5)
            panel_end()
        with col_d:
            panel_start("&mdash; ROC Curve")
            st.markdown('<div class="app-subtitle">Computed from the continuous anomaly score s(x).</div>',
                        unsafe_allow_html=True)
            if y_eval.nunique() < 2:
                st.info("The ROC curve needs both normal and anomalous records.")
            else:
                fpr_c, tpr_c, _ = roc_curve(y_eval, score_eval)
                fig6, ax6 = plt.subplots(figsize=(5, 5))
                ax6.plot(fpr_c, tpr_c, color=BRAND_RED, linewidth=2, label=f"AUC = {AUC:,.3f}")
                ax6.plot([0, 1], [0, 1], "--", color=TEXT_SUB)
                ax6.set_xlabel("False Positive Rate")
                ax6.set_ylabel("True Positive Rate")
                ax6.legend()
                fig6.tight_layout()
                st.pyplot(fig6, width='stretch')
                plt.close(fig6)
            panel_end()

    col_e, col_f = st.columns(2)
    with col_e:
        if HAS_LABELS:
            panel_start("&mdash; Precision, Recall and F1-Score")
            fig7, ax7 = plt.subplots(figsize=(5, 5))
            vals7 = [P, Rc, F1]
            ax7.bar(["Precision", "Recall", "F1-Score"], vals7, color=BRAND_RED)
            ax7.set_ylim(0, 1)
            for i, v in enumerate(vals7):
                ax7.text(i, v + 0.02, f"{v:,.3f}", ha="center", color=TEXT_MAIN)
            fig7.tight_layout()
            st.pyplot(fig7, width='stretch')
            plt.close(fig7)
            panel_end()
        else:
            st.info(LABELS_NOTE)

    with col_f:
        panel_start("&mdash; Event Processing Performance")
        st.markdown('<div class="app-subtitle">Average Event Processing Time Across the SMART-BYOD Processing Pipeline.</div>',
                    unsafe_allow_html=True)
        stage_timings = measure_pipeline_timing(raw_df, contamination)
        fig8, ax8 = plt.subplots(figsize=(6, 5))
        ax8.bar(list(stage_timings.keys()), list(stage_timings.values()), color=BRAND_RED)
        ax8.set_ylabel("Time (ms)")
        plt.setp(ax8.get_xticklabels(), rotation=20, ha="right")
        fig8.tight_layout()
        st.pyplot(fig8, width='stretch')
        plt.close(fig8)
        white_caption(f"Timed live on {len(raw_df):,} loaded records on this run (not hardcoded).")
        panel_end()

    col_g, col_h = st.columns(2)
    with col_g:
        panel_start("&mdash; CPU &amp; Memory Utilization")
        live = get_system_metrics()
        if live["cpu"] is None or live["mem"] is None:
            st.info("No CPU/memory measurement is available (install psutil, or run the analyzer with psutil).")
        else:
            fig9, ax9 = plt.subplots(figsize=(5, 5))
            ax9.bar(["CPU", "Memory"], [live["cpu"], live["mem"]], color=[BRAND_RED, BRAND_TEAL])
            ax9.axhline(50, color=TEXT_SUB, linestyle="--", linewidth=1)
            ax9.axhline(60, color=TEXT_SUB, linestyle=":", linewidth=1)
            ax9.set_ylabel("%")
            ax9.set_ylim(0, 100)
            fig9.tight_layout()
            st.pyplot(fig9, width='stretch')
            plt.close(fig9)
            white_caption(f"Measured ({live['source']}). Dashed: CPU target 50%; dotted: memory target 60%.")
        panel_end()

    with col_h:
        panel_start("&mdash; Dashboard Alert Distribution by Risk Level")
        order10 = ["MODERATE", "HIGH", "CRITICAL"]
        alert_counts = full[full["risk_level"] != "LOW"]["risk_level"].value_counts().reindex(order10).fillna(0)
        fig10, ax10 = plt.subplots(figsize=(5, 5))
        ax10.bar(alert_counts.index, alert_counts.values, color=[RISK_COLORS[l] for l in order10])
        ax10.set_ylabel("Count")
        fig10.tight_layout()
        st.pyplot(fig10, width='stretch')
        plt.close(fig10)
        white_caption("Only events that raised an alert (Verify, Restrict, Block/Isolate) are counted.")
        panel_end()
 

# ==============================================================================
# PAGE: ALERTS
# ==============================================================================
def render_alert_card(row, key: str = ""):
    """One alert in the Figure 3.3-7 layout."""
    level = str(row["risk_level"]).upper()
    icon, banner, response = ALERT_CARD.get(level, ALERT_CARD["LOW"])
    color = RISK_COLORS.get(level, "#64748B")
    user = row["user_id"] if IS_ADMIN else USER_ANON_MAP.get(row["user_id"], row["user_id"])
    device = row["device_id"] if IS_ADMIN else mask_mac(row["device_id"], keep=3)
    inds = [i for i in str(row.get("indicators", "") or "").split("; ") if i]
    flagged = int(row.get("predicted_anomaly", 0)) == 1
    s_val = row.get("anomaly_score")
    s_txt = "—" if s_val is None or pd.isna(s_val) else f"{float(s_val):.3f} ({'flagged' if flagged else 'not flagged'})"
    comp = ""
    if "comp_A" in row.index and pd.notna(row.get("comp_A")):
        comp = (f"A {W_A * float(row['comp_A']):.1f} + I {W_I * float(row['comp_I']):.1f} + "
                f"D {W_D * float(row['comp_D']):.1f} + C {W_C * float(row['comp_C']):.1f}")
    ind_html = "".join(f"<li>{i}</li>" for i in inds) or "<li>No indicator outside its normal range</li>"
    st.markdown(
        f"""
        <div class="panel" style="border-left:4px solid {color};">
          <div style="font-size:16px; font-weight:800; color:{color};">{icon} {banner}</div>
          <div style="font-size:13px; color:{TEXT_MAIN}; margin:4px 0 2px 0;">{user} / {device} &middot;
               Risk Score: <b>{float(row['risk_score']):,.0f}</b> &middot; Recommended Response: <b>{response}</b></div>
          <table style="width:100%; font-size:12.5px; margin-top:8px; border-collapse:collapse;">
            <tr><td style="color:{TEXT_SUB}; width:150px;">Timestamp</td><td>{pd.Timestamp(row['timestamp']):%Y-%m-%d %H:%M:%S}</td></tr>
            <tr><td style="color:{TEXT_SUB};">User</td><td>{user}</td></tr>
            <tr><td style="color:{TEXT_SUB};">Device</td><td>{device}</td></tr>
            <tr><td style="color:{TEXT_SUB};">Anomaly Score s(x)</td><td>{s_txt}</td></tr>
            <tr><td style="color:{TEXT_SUB};">Risk Score</td><td>{float(row['risk_score']):,.1f}{(' &nbsp;(' + comp + ')') if comp else ''}</td></tr>
            <tr><td style="color:{TEXT_SUB};">Risk Level</td><td>{risk_badge(level, RISK_COLORS)}</td></tr>
            <tr><td style="color:{TEXT_SUB}; vertical-align:top;">Indicators</td><td><ul style="margin:0; padding-left:18px;">{ind_html}</ul></td></tr>
            <tr><td style="color:{TEXT_SUB};">Policy Decision</td><td>{risk_badge(str(row['action']).upper(), {str(row['action']).upper(): ACTION_COLORS.get(row['action'], color)})}</td></tr>
            <tr><td style="color:{TEXT_SUB};">Alert</td><td>{ALERT_TYPE_BY_LEVEL.get(level, '')}</td></tr>
            <tr><td style="color:{TEXT_SUB};">Main driver</td><td>{row.get('driver', '')}</td></tr>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_alerts():
    st.markdown('<div class="app-title">🔔 Alerts</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">All flagged sessions in the current filter window</div>', unsafe_allow_html=True)

    alerts = fdf[fdf["risk_level"] != "LOW"].sort_values("timestamp", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Alerts", f"{len(alerts):,}")
    c2.metric("High/Critical", f"{len(alerts[alerts['risk_level'].isin(['HIGH','CRITICAL'])]):,}")
    c3.metric("Blocked/Isolated", f"{int(alerts['action'].isin(BLOCK_ACTIONS).sum()):,}")

    if not alerts.empty:
        pick = alerts.head(300).reset_index(drop=True)
        labels = [f"{r.timestamp:%Y-%m-%d %H:%M:%S} · {r.risk_level} {r.risk_score:,.0f} · "
                  f"{(r.user_id if IS_ADMIN else USER_ANON_MAP.get(r.user_id, r.user_id))}" for r in pick.itertuples()]
        default = int(pick["risk_score"].idxmax())
        chosen = st.selectbox("Alert detail (Figure 3.3-7 layout)", range(len(pick)), index=default,
                              format_func=lambda i: labels[i])
        render_alert_card(pick.loc[chosen])

    show = alerts[[
        "timestamp", "user_id", "user_type", "device_id", "scenario_code", "alert_type", "driver",
        "risk_score", "risk_level", "action", "indicators", "ip_address", "location",
    ]].copy()
    show["scenario_code"] = show["scenario_code"].map(SCENARIO_LABELS).fillna(show["scenario_code"])
    if not IS_ADMIN:
        show["device_id"] = show["device_id"].map(lambda v: mask_mac(v, keep=2))
        
    show = show.rename(columns={
        "timestamp": "Time", "user_id": "User", "user_type": "Type",
        "device_id": "Device", "scenario_code": "Scenario", "alert_type": "Alert Type", "driver": "Main Driver",
        "indicators": "Indicators", "risk_score": "Risk",
        "risk_level": "Level", "action": "Action", "ip_address": "IP Address",
        "location": "Location",
    })
    st.dataframe(show, width='stretch', hide_index=True, height=520)
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
        if not IS_ADMIN:
            u["user_id"] = u["user_id"].map(USER_ANON_MAP)   # anonymous display id, never the real name
        st.dataframe(
            u.rename(columns={
                "user_id": "User ID", "user_type": "Type", "sessions": "Sessions",
                "events": "Events", "avg_risk": "Avg Risk", "max_risk": "Max Risk",
                "alerts": "Alerts",
            }),
            width='stretch', hide_index=True, height=480,
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
        if not IS_ADMIN:
            d["device_id"] = d["device_id"].map(lambda v: mask_mac(v, keep=3))
        st.dataframe(
            d.rename(columns={
                "device_id": "Device ID", "users": "Users", "sessions": "Sessions",
                "events": "Events", "avg_risk": "Avg Risk", "max_risk": "Max Risk",
                "alerts": "Alerts",
            }),
            width='stretch', hide_index=True, height=480,
        )


# ==============================================================================
# PAGE: ACTIVITY LOGS
# ==============================================================================
def page_activity_logs():
    st.markdown('<div class="app-title">📝 Activity Logs</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Raw filtered event log</div>', unsafe_allow_html=True)

    search = st.text_input(f"Search by {'User ID, ' if IS_ADMIN else ''}Device ID, or Session ID")
    logs = fdf.copy()
    if search:
        s = search.strip().lower()
        mask = logs["device_id"].str.lower().str.contains(s) | logs["session_id"].str.lower().str.contains(s)
        if IS_ADMIN:
            mask = mask | logs["user_id"].str.lower().str.contains(s)
        logs = logs[mask]

    logs = logs.copy()
    if not IS_ADMIN:
        logs["user_id"] = logs["ip_address"].map(mask_ip)  # e.g. 137.158.X.X, never the name
        logs["device_id"] = logs["device_id"].map(lambda v: mask_mac(v, keep=3))

    cols = ["timestamp", "user_id", "user_type", "device_id", "session_id",
            "risk_score", "risk_level", "action"] + FEATURE_COLS
    st.dataframe(
        logs[cols].sort_values("timestamp", ascending=False)
            .rename(columns={"timestamp": "Time", "user_id": "User", "user_type": "Type",
                              "device_id": "Device", "session_id": "Session",
                              "risk_score": "Risk", "risk_level": "Level", "action": "Action"}),
        width='stretch', hide_index=True, height=560,
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
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="panel"><div class="panel-title">Main Risk Driver of Alerts</div>', unsafe_allow_html=True)
        breakdown = fdf[fdf["risk_level"] != "LOW"]["driver"].value_counts()
        fig2 = go.Figure(go.Bar(
            x=breakdown.values, y=breakdown.index, orientation="h",
            marker_color=ACCENT,
        ))
        fig2.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_MAIN), yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})
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
    st.plotly_chart(fig3, width='stretch', config={"displayModeBar": False})
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
    st.plotly_chart(fig4, width='stretch', config={"displayModeBar": False})
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
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})


def _table_view(df_table):
    with st.expander("View as table"):
        st.dataframe(df_table, width='stretch', hide_index=True)


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
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        peak = int(hours.idxmax())
        st.caption(f"Peak hour: {peak:02d}:00 with {int(hours.max()):,} events.")
        _table_view(pd.DataFrame({"Hour": [f"{h:02d}:00" for h in hours.index], "Events": hours.values}))
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Device inventory status (Table 3.5-6, device component) ----
    _panel("Device Inventory Status",
           "Registered = in the device registry; learned = used by the same user for 24 h or more; "
           "unknown = neither (new or unclassified)")
    reg = ev.groupby("mac")["registration_status"].first().str.lower().value_counts()
    _hbar(reg.index, reg.values, "devices", muted=("unknown",), height=max(170, 34 * len(reg) + 40))
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
# ---- System performance test (analyzer --perf-test, PT-01 to PT-08) ----
PERF_STATUS_COLORS = {"Pass": "#22C55E", "Fail": "#EF4444", "Not measured": "#F59E0B"}


def ui_metrics_path(db_path: Path) -> Path:
    """Side database shared with the analyzer's performance test (same name rule as the analyzer):
    smart_byod.db -> smart_byod_ui_metrics.db. The main database stays read-only here."""
    return Path(db_path).resolve().with_name(Path(db_path).stem + "_ui_metrics.db")


@st.cache_data(show_spinner=False)
def load_perf_tables(db_path: str, signature: tuple) -> dict:
    """performance_runs / performance_results / performance_samples written by --perf-test."""
    out = {"runs": pd.DataFrame(), "results": pd.DataFrame(), "samples": pd.DataFrame()}
    try:
        conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.OperationalError:
        return out
    try:
        out["runs"] = pd.read_sql_query("SELECT * FROM performance_runs ORDER BY started_at DESC", conn)
        out["results"] = pd.read_sql_query("SELECT * FROM performance_results ORDER BY run_id, test_id", conn)
        out["samples"] = pd.read_sql_query("SELECT * FROM performance_samples ORDER BY run_id, t_offset_s", conn)
    except Exception:
        pass
    finally:
        conn.close()
    return out


def latest_perf_result(test_id: str):
    """(actual, status) of a test in the latest performance run, or None."""
    db_path = find_file(DB_FILENAME) if DATA_SOURCE == "sqlite" else None
    if db_path is None:
        return None
    t = load_perf_tables(str(db_path), db_signature(db_path))
    if t["runs"].empty or t["results"].empty:
        return None
    r = t["results"][(t["results"]["run_id"] == t["runs"].iloc[0]["run_id"]) & (t["results"]["test_id"] == test_id)]
    return None if r.empty else (r.iloc[0]["actual"], r.iloc[0]["status"])


def render_perf_results(db_path: Path):
    t = load_perf_tables(str(db_path), db_signature(db_path))
    runs, results, samples = t["runs"], t["results"], t["samples"]
    if runs.empty:
        st.info("No performance test has been run on this database yet. Run "
                "`python SMART-BYOD_Analyze.py --perf-test --db " + DB_FILENAME + "` and switch on the "
                "Live performance monitor below while it runs.")
        return
    labels = [f"{r.run_id} · {r.users} users · {r.duration_s:g} s active" for r in runs.itertuples()]
    pick = st.selectbox("Performance test run", labels, index=0, key="perf_run_pick")
    run = runs.iloc[labels.index(pick)]
    res = results[results["run_id"] == run["run_id"]]
    passed = int((res["status"] == "Pass").sum())
    st.markdown(
        f'<div style="color:{TEXT_SUB}; font-size:12.5px; margin:4px 0 8px 0;">{run["started_at"]} to '
        f'{run["finished_at"]} · idle {run["idle_s"]:g} s, active {run["duration_s"]:g} s, cycle '
        f'{run["interval_s"]:g} s · {int(run["events_generated"]):,} events generated · model {run["model_version"]} · '
        f'host {run["host_cpus"] if pd.notna(run["host_cpus"]) else "?"} CPUs, '
        f'{run["host_memory_gb"] if pd.notna(run["host_memory_gb"]) else "?"} GB · '
        f'<b style="color:{TEXT_MAIN};">{passed} of {len(res)} passed</b></div>',
        unsafe_allow_html=True)
    rows = ""
    for r in res.itertuples():
        color = PERF_STATUS_COLORS.get(r.status, TEXT_SUB)
        rows += (f'<tr style="border-bottom:1px solid {PANEL_BORDER};">'
                 f'<td style="padding:6px 4px;">{r.test_id}</td><td style="padding:6px 4px;">{r.metric}</td>'
                 f'<td style="padding:6px 4px;">{r.target}</td><td style="padding:6px 4px; color:{TEXT_SUB};">{r.method}</td>'
                 f'<td style="padding:6px 4px; font-weight:700;">{r.actual}</td>'
                 f'<td style="padding:6px 4px; color:{color}; font-weight:700;">{r.status}</td>'
                 f'<td style="padding:6px 4px; color:{TEXT_SUB}; font-size:12px;">{r.detail}</td></tr>')
    head = "".join(f'<th style="text-align:left; padding:6px 4px; color:{TEXT_SUB};">{h}</th>'
                   for h in ("Test ID", "Metric", "Target", "Test Method", "Actual Result", "Status", "Detail"))
    st.markdown(f'<table style="width:100%; border-collapse:collapse; font-size:13.5px;"><thead><tr '
                f'style="border-bottom:1px solid {PANEL_BORDER};">{head}</tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True)

    smp = samples[samples["run_id"] == run["run_id"]]
    if smp.empty:
        return
    st.markdown(f'<div class="panel-title" style="margin-top:18px;">CPU and Memory During the Test (PT-05, PT-06, PT-08)</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=smp["t_offset_s"], y=smp["cpu_pct"], name="CPU (host)", line=dict(color=BRAND_RED)))
    fig.add_trace(go.Scatter(x=smp["t_offset_s"], y=smp["mem_pct"], name="Memory (host)", line=dict(color=BRAND_TEAL)))
    fig.add_hline(y=50, line_dash="dash", line_color=BRAND_RED, annotation_text="CPU target 50%")
    fig.add_hline(y=60, line_dash="dot", line_color=BRAND_TEAL, annotation_text="Memory target 60%")
    active = smp[smp["phase"] == "active"]
    if not active.empty:
        fig.add_vrect(x0=float(active["t_offset_s"].min()), x1=float(active["t_offset_s"].max()),
                      fillcolor="#3B82F6", opacity=0.08, line_width=0, annotation_text="active monitoring",
                      annotation_position="top left")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MAIN), yaxis=dict(title="%", range=[0, 100]),
                      xaxis=dict(title="Seconds from the start of the test"), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False}, key="perf_cpu_mem")
    phases = smp.groupby("phase")[["cpu_pct", "mem_pct", "proc_rss_mb"]].mean().reindex(["idle", "active"])
    white_caption("Average idle → active: CPU " + " → ".join(f"{v:.1f}%" for v in phases["cpu_pct"].fillna(0)) +
                  ", memory " + " → ".join(f"{v:.1f}%" for v in phases["mem_pct"].fillna(0)) +
                  ", analyzer RSS " + " → ".join(f"{v:.0f} MB" for v in phases["proc_rss_mb"].fillna(0)) +
                  ". One sample per second, measured by the analyzer with psutil.")


def _live_monitor_body(db_path: Path):
    """PT-03: show the newest data update written by the running performance test and log how long it
    took to reach the screen (data-update-to-dashboard-display time)."""
    t_render = time.perf_counter()
    path = ui_metrics_path(db_path)
    if not path.exists():
        st.info("Waiting for a performance test: start `--perf-test` in the analyzer.")
        return
    try:
        conn = sqlite3.connect(str(path), timeout=2.0)
        conn.execute("CREATE TABLE IF NOT EXISTS dashboard_refresh_log (run_id TEXT, probe_id INTEGER, written_at REAL, "
                     "displayed_at REAL, delay_ms REAL, render_ms REAL, PRIMARY KEY (run_id, probe_id));")
        row = conn.execute("SELECT probe_id, run_id, written_at FROM perf_probe ORDER BY probe_id DESC LIMIT 1;").fetchone()
    except sqlite3.Error as exc:
        st.warning(f"Live monitor: {exc}")
        return
    if row is None:
        conn.close()
        st.info("Waiting for a performance test: start `--perf-test` in the analyzer.")
        return
    probe_id, run_id, written_at = row
    age = time.time() - written_at
    running = age < 10
    cpu = mem = None
    if psutil is not None:
        try:
            cpu, mem = psutil.cpu_percent(interval=None), psutil.virtual_memory().percent
        except Exception:
            pass
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test run", run_id, "running" if running else "finished", delta_color="off")
    c2.metric("Latest data update", datetime.fromtimestamp(written_at).strftime("%H:%M:%S"))
    c3.metric("Host CPU", "—" if cpu is None else f"{cpu:.0f}%")
    c4.metric("Host memory", "—" if mem is None else f"{mem:.0f}%")
    shown = st.session_state.setdefault("perf_probes_shown", set())
    if running and (run_id, probe_id) not in shown:
        displayed = time.time()
        conn.execute("INSERT OR IGNORE INTO dashboard_refresh_log VALUES (?, ?, ?, ?, ?, ?);",
                     (run_id, probe_id, written_at, displayed, round((displayed - written_at) * 1000, 1),
                      round((time.perf_counter() - t_render) * 1000, 1)))
        conn.commit()
        shown.add((run_id, probe_id))
    stats = conn.execute("SELECT COUNT(*), AVG(delay_ms), MAX(delay_ms) FROM dashboard_refresh_log WHERE run_id = ?;",
                         (run_id,)).fetchone()
    conn.close()
    if stats and stats[0]:
        st.markdown(f'<div style="color:{TEXT_SUB}; font-size:12.5px;">PT-03 dashboard refresh for {run_id}: '
                    f'<b style="color:{TEXT_MAIN};">{stats[1] / 1000:.2f} s</b> average over {stats[0]} updates '
                    f'(max {stats[2] / 1000:.2f} s; target ≤ 3 s). Measured as the time from the analyzer writing '
                    f'an update to this page showing it; analyzer and dashboard must run on the same computer.</div>',
                    unsafe_allow_html=True)


_live_monitor_fragment = (st.fragment(run_every=1.0)(_live_monitor_body)
                          if hasattr(st, "fragment") else None)


def page_system_performance():
    st.markdown('<div class="app-title">📊 System Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Measured operational telemetry (analyzer system_metrics, host psutil, page render time) '
                'and the system performance test PT-01 to PT-08</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    render_system_performance(compact=False)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("\U0001F504 Refresh telemetry"):
        st.rerun()

    db_path = find_file(DB_FILENAME) if DATA_SOURCE == "sqlite" else None
    st.markdown('<div class="panel"><div class="panel-title">System Performance Test (PT-01 to PT-08)</div>',
                unsafe_allow_html=True)
    if db_path is None:
        st.info("The performance test results are stored in the SQLite database; they are not available in CSV mode.")
    else:
        render_perf_results(db_path)
    st.markdown("</div>", unsafe_allow_html=True)

    if db_path is None:
        return
    st.markdown('<div class="panel"><div class="panel-title">Live Performance Monitor (measures PT-03)</div>',
                unsafe_allow_html=True)
    live_on = st.toggle("Live performance monitor (refresh every second while a --perf-test runs)", value=False,
                        key="perf_live_on")
    if live_on:
        if _live_monitor_fragment is None:
            st.warning("This Streamlit version has no st.fragment (needs 1.37 or later); the monitor refreshes only "
                       "when you press Refresh telemetry, so PT-03 values will be too high. Run: pip install -U streamlit")
            _live_monitor_body(db_path)
        else:
            _live_monitor_fragment(db_path)
    else:
        st.markdown(f'<div style="color:{TEXT_SUB}; font-size:12.5px;">Switch this on, then start '
                    f'<code>python SMART-BYOD_Analyze.py --perf-test --db {DB_FILENAME}</code>. Keep this page open '
                    f'until the analyzer prints its report; reload the page afterwards to see the results above.</div>',
                    unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


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
    st.write("- **Behavioral features (6):** " + ", ".join(f"{f} ({FEATURE_META[f]['label']}, {FEATURE_META[f]['unit']})" for f in FEATURE_COLS))
    st.write("- **Normal ranges (Table 3.4-2):** " + ", ".join(f"{f} ≤ {v}" for f, v in NORMAL_LIMITS.items()))
    if DATA_SOURCE == "sqlite":
        st.write(f"- **Data source:** SQLite ({DB_FILENAME}) - read-only; scores saved by SMART-BYOD_Analyze.py")
        mv = df["model_version"].dropna().unique() if "model_version" in df.columns else []
        st.write(f"- **Model version:** {', '.join(map(str, mv)) or 'n/a'} (contamination set in the analyzer)")
    else:
        st.write(f"- **Data source:** CSV ({CSV_FILENAME}), scored in the app")
        st.write(f"- **Contamination (current):** {contamination}")
    st.write(f"- **Records loaded:** {len(df):,}")
    st.write(f"- **Unique users:** {df['user_id'].nunique()} &nbsp;|&nbsp; **Unique devices:** {df['device_id'].nunique()}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Risk Components (Table 3.5-6)</div>', unsafe_allow_html=True)
    st.markdown(
        "R = 100 (0.40 A + 0.25 I + 0.20 D + 0.15 C)\n\n"
        "- **A (anomaly):** from s(x) = −score_samples(x); A > 0.5 exactly when the Isolation Forest flags the event\n"
        "- **I (identity):** min(FL/5, 1); at least 0.6 when the account logs in from two devices within 15 min\n"
        "- **D (device):** 0 registered or learned, 0.2 unclassified, 0.6 not in the inventory, 1.0 another user's device or DS ≥ 2\n"
        "- **C (context):** max(C_time, C_net); C_time 0 at LTD ≤ 60 min rising to 1 at 180 min; C_net 0.5 for a new external network\n\n"
        "Anomaly evidence alone adds at most 40 points (Verify); Critical needs A ≥ 0.5 and I ≥ 0.2 (Propositions 1-2)."
        + ("" if DATA_SOURCE == "sqlite" else "\n\n*CSV mode: I, D and C are derived from FL, DS and LTD only.*"))
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
st.session_state["last_refresh_s"] = time.perf_counter() - _T_START   # measured dashboard refresh time

