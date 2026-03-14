import os
import io
import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pandas.errors import EmptyDataError, ParserError

# ============================
# PAGE CONFIG
# ============================

st.set_page_config(
    page_title="EarlyPulse — Early Sepsis Detection",
    page_icon="heartbeat",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================
# SPLASH / LOADING SCREEN
# ============================

def show_loading():
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            """
        <div style="position:fixed;top:0;left:0;width:100vw;height:100vh;
                    background:#060e1a;
                    z-index:9999;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;color:#e4eeff;">
            <div style="width:200px;height:200px;position:relative;margin-bottom:2rem;">
                <svg viewBox="0 0 200 200" style="width:100%;height:100%;">
                    <circle cx="100" cy="100" r="85" fill="none" stroke="#0f2340" stroke-width="12"/>
                    <circle cx="100" cy="100" r="85" fill="none" stroke="#3a78d4" stroke-width="8" class="pulse1"/>
                    <circle cx="100" cy="100" r="85" fill="none" stroke="#5b9cf6" stroke-width="4" class="pulse2"/>
                    <path d="M30 100 L55 100 L70 50 L90 150 L110 50 L130 100 L155 100"
                          fill="none" stroke="#5b9cf6" stroke-width="5"
                          stroke-linecap="round" class="ecg-line"/>
                </svg>
            </div>
            <h1 style="font-family:'Inter',sans-serif;font-size:4rem;font-weight:800;
                       letter-spacing:-0.04em;margin:0;color:#ffffff;white-space:nowrap;">
                Early<span style="color:#5b9cf6;">Pulse</span>
            </h1>
            <p style="font-size:1rem;color:#3d6090;margin-top:0.8rem;letter-spacing:0.02em;">
                Early Sepsis Detection · PhysioNet 2019
            </p>
            <p style="font-size:0.82rem;color:#1e3355;margin-top:1.5rem;letter-spacing:0.04em;">
                Loading 20,317 ICU patients...
            </p>
        </div>

        <style>
        @keyframes pulse1 { 0%,100%{r:85;opacity:0.4} 50%{r:100;opacity:0.9} }
        @keyframes pulse2 { 0%,100%{r:85;opacity:0.3} 50%{r:105;opacity:0.7} }
        @keyframes ecg { 0%{transform:translateX(-100px)} 100%{transform:translateX(300px)} }
        .pulse1 { animation: pulse1 2s infinite ease-in-out; }
        .pulse2 { animation: pulse2 2s infinite ease-in-out 0.5s; }
        .ecg-line { animation: ecg 4s linear infinite; }
        </style>
        """,
            unsafe_allow_html=True,
        )
    time.sleep(1.5)
    placeholder.empty()
    # Call splash only once per browser session
if "ep_has_loaded" not in st.session_state:
    show_loading()
    st.session_state["ep_has_loaded"] = True


# Call splash once at app start


# ============================
# ============================
# GLOBAL THEME CSS
# ============================

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    .stApp { background: #060e1a; color: #e4eeff; }
    .block-container { padding: 0 2.5rem 6rem; max-width: 1300px; }

    .ep-wordmark {
        font-family: 'Syne', sans-serif;
        font-size: 3.8rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.05;
        white-space: nowrap;
        color: #ffffff;
        margin: 0 0 1rem 0;
        display: block;
    }
    .ep-wordmark .accent { color: #5b9cf6; }
    .ep-subtitle { font-size: 0.95rem; color: #8da8cc; line-height: 1.7; margin: 0 0 1.3rem 0; max-width: 420px; }
    .ep-credit { font-size: 0.7rem; color: #2a4f7a; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1.2rem; }
    .ep-badge {
        display: inline-flex; align-items: center;
        background: rgba(220,60,60,0.1); color: #e06060;
        border: 1px solid rgba(220,60,60,0.22);
        padding: 0.28rem 0.8rem; border-radius: 4px;
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    }
    .ep-micro { font-size: 0.62rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.14em; color: #2a4f7a; margin-bottom: 0.55rem; display: block; }
    .ep-rule { height: 1px; background: #0d1e35; margin: 1.8rem 0; }

    .metric-tile {
        background: #0b1828; border-radius: 12px;
        padding: 1.2rem 1.4rem 1rem; border: 1px solid #0f2340;
        margin-bottom: 0.55rem; transition: border-color 0.2s;
    }
    .metric-tile:hover { border-color: #1a4070; }
    .tile-label { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #2a4f7a; margin-bottom: 0.2rem; }
    .tile-model { font-size: 0.76rem; color: #3d6090; margin-bottom: 0.15rem; }
    .tile-value { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 500; color: #90bcf8; line-height: 1; letter-spacing: -0.02em; }
    .tile-value.best { color: #c0d8ff; }
    .tile-sub { font-size: 0.62rem; color: #3a6090; margin-top: 0.3rem; }

    .nav-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.55rem; margin: 1.4rem 0 0.5rem 0; }
    .nav-card { background: #0b1828; border: 1px solid #0f2340; border-radius: 10px; padding: 1rem 1.05rem; transition: border-color 0.2s, background 0.2s; }
    .nav-card:hover { border-color: #1a4070; background: #0d1f38; }
    .nav-num { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #3a6090; margin-bottom: 0.4rem; display: block; }
    .nav-title { font-size: 0.84rem; font-weight: 600; color: #a8c8f0; margin-bottom: 0.3rem; letter-spacing: -0.01em; }
    .nav-desc { font-size: 0.72rem; color: #6a90b8; line-height: 1.5; }

    .stTabs [data-baseweb="tab-list"] { background: #070f1c; border-radius: 8px; padding: 4px; gap: 2px; border: 1px solid #0f2340; }
    .stTabs [data-baseweb="tab"] { color: #3d6090; font-weight: 500; font-size: 0.88rem; border-radius: 6px; padding: 0.6rem 1.3rem !important; transition: color 0.15s; }
    .stTabs [data-baseweb="tab"]:hover { color: #90bcf8; }
    [data-baseweb="tab"][aria-selected="true"] { background: #0f2a50 !important; color: #d0e6ff !important; font-weight: 600; }

    .glass { background: #0b1828; border-radius: 12px; border: 1px solid #0f2340; padding: 1.8rem; margin-bottom: 1.2rem; transition: border-color 0.2s; }
    .glass:hover { border-color: #1a4070; }

    .glass-info-card { background: #0b1828; border: 1px solid #0f2340; border-radius: 12px; padding: 1.5rem 1.8rem; margin: 0.4rem 0 1.2rem 0; text-align: center; transition: border-color 0.2s; }
    .glass-info-card:hover { border-color: #1a4070; }
    .card-icon { font-size: 2rem; margin-bottom: 0.4rem; }
    .card-title { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #ffffff !important; background: none !important; -webkit-text-fill-color: #ffffff !important; margin: 0.3rem 0; }
    .card-desc { font-size: 0.88rem; color: #7a9fc0; line-height: 1.65; margin: 0; }
    .card-desc code { background: #091522; padding: 0.1rem 0.45rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #70a8e8; }

    .leaderboard-card { background: #0b1828; border: 1px solid #1a4070; border-radius: 10px; padding: 1rem 1.4rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 1rem; }
    .lb-trophy { font-size: 1.3rem; }
    .lb-label { font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.08em; color: #2a4f7a; }
    .lb-name { font-size: 1rem; font-weight: 600; color: #d0e6ff; }
    .lb-score { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #3d6090; margin-left: auto; }

    .metric-card { background: #0b1828; border-radius: 10px; padding: 1.1rem; text-align: center; border: 1px solid #0f2340; transition: border-color 0.2s; }
    .metric-card:hover { border-color: #1a4070; }
    .metric-card h4 { font-size: 0.63rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; color: #2a4f7a !important; background: none !important; -webkit-text-fill-color: #2a4f7a !important; margin-bottom: 0.3rem !important; }
    .metric-card h2 { font-family: 'JetBrains Mono', monospace !important; font-size: 1.7rem !important; font-weight: 500 !important; letter-spacing: -0.02em !important; color: #90bcf8 !important; background: none !important; -webkit-text-fill-color: #90bcf8 !important; margin: 0.1rem 0 !important; }
    .metric-card p { font-size: 0.65rem; color: #3a6090; margin: 0; }

    h2, h3, h4 { color: #d8eaff !important; background: none !important; -webkit-text-fill-color: #d8eaff !important; font-weight: 600; letter-spacing: -0.02em; }

    .ep-card { background: #0b1828; border-radius: 10px; padding: 10px 14px; border: 1px solid #0f2340; }
    .ep-sub    { color: #4a6f96; font-size: 0.85rem; }
    .ep-header { color: #d8eaff; font-weight: 600; font-size: 0.92rem; }
    .ep-caption { color: #4a6f96; font-size: 0.82rem; }

    .stDataFrame { border-radius: 10px; overflow: hidden; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #90bcf8 !important; font-size: 1.5rem !important; }
    div[data-testid="stMetricLabel"] { color: #3d6090 !important; font-size: 0.78rem !important; }
    .stSelectbox > div > div { background: #0b1828 !important; border: 1px solid #1a3560 !important; border-radius: 8px !important; color: #d8eaff !important; }
    .stExpander { background: #0b1828 !important; border: 1px solid #0f2340 !important; border-radius: 10px !important; }
    .stExpander summary { color: #4a6f96 !important; font-size: 0.9rem !important; }
    .stExpander p, .stExpander li { color: #8da8cc !important; }
    .stExpander strong { color: #c8deff !important; }
    .stExpander code { background: #091522 !important; color: #70a8e8 !important; border-radius: 4px; padding: 0.1rem 0.4rem; }
    .stAlert { border-radius: 10px !important; }
    .stCaption, [data-testid="stCaptionContainer"] { color: #3d6090 !important; font-size: 0.8rem !important; }
    .stMarkdown p { color: #8da8cc; line-height: 1.7; }
    .stMarkdown li { color: #8da8cc; }
    .stMarkdown strong { color: #c8deff !important; }
    .stMarkdown code { background: #091522 !important; color: #70a8e8 !important; border-radius: 4px; padding: 0.1rem 0.45rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #060e1a; }
    ::-webkit-scrollbar-thumb { background: #0f2340; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #1a4070; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================
# HERO HEADER
# ============================

# ── Compute metrics live from CSVs ───────────────────────
@st.cache_data(show_spinner=False)
def _load_hero_metrics():
    from sklearn.metrics import roc_curve, auc as sk_auc, average_precision_score
    defaults = {"xgb_auroc": 0.9466, "xgb_auprc": 0.7030,
                "gru_auroc": 0.6544, "qsofa_auroc": 0.4869}
    _RDIR = "data/results"
    paths = {
        "xgb":   f"{_RDIR}/earlypulse_XGBoost_6h24h_CORRECT.csv",
        "gru":   f"{_RDIR}/earlypulse_GRU_6h24h_CORRECT.csv",
        "qsofa": f"{_RDIR}/earlypulse_qSOFA_24h_CORRECT.csv",
    }
    def _compute(path):
        try:
            df = pd.read_csv(path)
            if "HasSepsis" not in df.columns: return None, None
            y = df["HasSepsis"].fillna(0).astype(int)
            s = None
            for col in df.columns:
                if any(k in col.lower() for k in ["prob","score","maxprob"]):
                    s = pd.to_numeric(df[col], errors="coerce").fillna(0); break
            if s is None:
                s = df.get("HasEarlyAlert", pd.Series([0]*len(df))).fillna(0).astype(float)
            if len(y.unique()) < 2: return None, None
            fpr, tpr, _ = roc_curve(y, s)
            return float(sk_auc(fpr, tpr)), float(average_precision_score(y, s))
        except Exception: return None, None
    for key, path in paths.items():
        if os.path.exists(path):
            a, p = _compute(path)
            if a and key == "xgb":   defaults["xgb_auroc"] = a
            if p and key == "xgb":   defaults["xgb_auprc"] = p
            if a and key == "gru":   defaults["gru_auroc"] = a
            if a and key == "qsofa": defaults["qsofa_auroc"] = a
    return defaults

_hero = _load_hero_metrics()

# ── Hero layout ───────────────────────────────────────────
st.markdown('<div style="height:2.2rem"></div>', unsafe_allow_html=True)

hero_left, hero_right = st.columns([5, 7], gap="large")

with hero_left:
    st.markdown(
        f"""
        <p style="font-size:0.7rem;font-weight:600;text-transform:uppercase;
                  letter-spacing:0.12em;color:#2a4f7a;margin-bottom:0.7rem;">
            Health AI &nbsp;·&nbsp; PhysioNet 2019 &nbsp;·&nbsp; 20,317 ICU Stays
        </p>
        <div class="ep-wordmark">Early<span class="accent">Pulse</span></div>
        <p class="ep-subtitle">
            A reproducible ML system for early sepsis detection.<br>
            XGBoost · GRU · qSOFA — evaluated on real ICU data<br>
            with FastAPI inference and drift monitoring.
        </p>
        <p class="ep-credit">Mohamed Kamal &nbsp;·&nbsp; Health AI Portfolio &nbsp;·&nbsp; 2025–2026</p>
        <span class="ep-badge">⚠ Research prototype — not for clinical use</span>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown(
        '<p style="font-size:0.68rem;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.1em;color:#2a4f7a;margin-bottom:0.5rem;">'
        'Live benchmarks · held-out patient-level test set</p>',
        unsafe_allow_html=True
    )
    r1a, r1b = st.columns(2)
    r2a, r2b = st.columns(2)
    with r1a:
        st.markdown(f"""<div class="metric-tile">
            <div class="tile-label">AUROC</div>
            <div class="tile-model">XGBoost</div>
            <div class="tile-value best">{_hero["xgb_auroc"]:.4f}</div>
            <div class="tile-sub">Patient-level · test split (3,987 pts)</div>
        </div>""", unsafe_allow_html=True)
    with r1b:
        st.markdown(f"""<div class="metric-tile">
            <div class="tile-label">AUPRC</div>
            <div class="tile-model">XGBoost</div>
            <div class="tile-value">{_hero["xgb_auprc"]:.4f}</div>
            <div class="tile-sub">Average precision · test split</div>
        </div>""", unsafe_allow_html=True)
    with r2a:
        st.markdown(f"""<div class="metric-tile">
            <div class="tile-label">AUROC</div>
            <div class="tile-model">GRU Sequence</div>
            <div class="tile-value">{_hero["gru_auroc"]:.4f}</div>
            <div class="tile-sub">Patient-level · full cohort</div>
        </div>""", unsafe_allow_html=True)
    with r2b:
        st.markdown(f"""<div class="metric-tile">
            <div class="tile-label">AUROC</div>
            <div class="tile-model">qSOFA Baseline</div>
            <div class="tile-value">{_hero["qsofa_auroc"]:.4f}</div>
            <div class="tile-sub">Rule-based · full cohort</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="ep-rule"></div>', unsafe_allow_html=True)

# ============================
# CONFIG & CONSTANTS
# ============================

import os as _os; DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "training_setA")
RESULTS_DIR = "data/results"

QSOFA_CSV = f"{RESULTS_DIR}/earlypulse_qSOFA_24h_CORRECT.csv"
XGB_CSV   = f"{RESULTS_DIR}/earlypulse_XGBoost_6h24h_CORRECT.csv"
GRU_CSV   = f"{RESULTS_DIR}/earlypulse_GRU_6h24h_CORRECT.csv"

XGB_CSV_24 = f"{RESULTS_DIR}/earlypulse_XGBoost_24h_CORRECT.csv"
GRU_CSV_24 = f"{RESULTS_DIR}/earlypulse_GRU_24h_CORRECT.csv"

QSOFA_AUROC        = 0.4869
XGB_PATIENT_AUROC  = 0.9466
XGB_STEP_AUROC     = 0.7888
GRU_PATIENT_AUROC  = 0.6544

# Heroes demo patients
HERO_PATIENTS = {
    "Aurora — 13.0h early": "p008362.psv",
    "Phoenix — 13.4h early": "p009916.psv",
    "Apollo — 20.0h early": "p015971.psv",
    "Nova — 12.3h early": "p014410.psv",
    "Luna — 14.0h early": "p007481.psv",
}
HERO_PATIENT_STORIES = {
    "Aurora — 13.0h early": "Stable at first, then subtle drift in vitals before a clear early alert around 13 hours before onset.",
    "Phoenix — 13.4h early": "Respiratory rate climbs with falling blood pressure — a textbook early-warning qSOFA case.",
    "Apollo — 20.0h early": "Very long early-warning window; alerts fire well before sepsis, giving plenty of reaction time.",
    "Nova — 12.3h early": "Noisy vitals but consistent pattern — challenging case where early alerts still catch the trend.",
    "Luna — 14.0h early": "Borderline signals that eventually tip into a clear early alert roughly 14 hours before onset.",
}

# ============================
# SMALL DEBUG LOGGER
# ============================

def debug_log(msg: str):
    """Simple console logger for debugging (disabled by default)."""
    if False:  # flip to True if you want console logs
        print(f"[EarlyPulse] {msg}")



#===========================
# load patient 
#===========================
def load_any_patient_file(uploaded_file):
    """
    Accepts ANY CSV/PSV ICU-like file.
    Automatically:
    - Detects delimiter (| or ,)
    - Normalizes column names (internally)
    - Remaps common aliases to a standard schema
    - Adds back PhysioNet-style column names so existing code keeps working
    """

    # Read raw bytes
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    # Safe encoding guess: utf-8 with fallback
    try:
        sample = raw[:5000].decode("utf-8", errors="ignore")
        encoding = "utf-8"
    except Exception:
        sample = raw[:5000].decode("latin1", errors="ignore")
        encoding = "latin1"

    # Auto-detect delimiter
    delimiter = "|" if "|" in sample else ","

    # Read the file
    try:
        df = pd.read_csv(
            uploaded_file,
            sep=delimiter,
            encoding=encoding,
            engine="python",
        )
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

    # --- INTERNAL NORMALISATION (lowercase copy of the names) ---
    df.columns = [c.strip().lower() for c in df.columns]

    # Canonical "internal" expected columns
    expected_cols = {
        "heartrate": ["hr", "heart_rate", "heart rate"],
        "resprate": ["rr", "resp", "resp_rate", "respiratoryrate", "respiratory rate"],
        "o2sat": ["spo2", "o2", "oxygen_saturation", "sat"],
        "sbp": ["systolicbp", "bp_sys", "sbp_mmhg", "systolic bp"],
        "dbp": ["diastolicbp", "bp_dia", "dbp_mmhg", "diastolic bp"],
        "map": ["meanbp", "mean_arterial_pressure", "mean bp"],
        "temperature": ["temp", "body_temp", "tmp"],
        "sepsislabel": ["label", "sepsis", "sepsis_label"],
        "iculos": ["icu_los", "icu_hours", "time", "hour"],
    }

    # Smart remapping of alternative names → internal canonical names
    for standard, alternatives in expected_cols.items():
        if standard not in df.columns:
            for alt in alternatives:
                alt_lower = alt.lower()
                if alt_lower in df.columns:
                    df[standard] = df[alt_lower]
                    break

    # Fill missing internal columns with NaN so downstream code is safe
    for col in expected_cols.keys():
        if col not in df.columns:
            df[col] = float("nan")

    # --- ADD BACK PHYSIONET-STYLE COLUMN NAMES AS ALIASES ---
    # This keeps your existing plot_patient() & build_clinical_summary() happy.
    reverse_aliases = {
        "HR": ["heartrate", "hr"],
        "Resp": ["resprate", "resp", "resp_rate"],
        "O2Sat": ["o2sat", "spo2"],
        "SBP": ["sbp"],
        "DBP": ["dbp"],
        "MAP": ["map"],
        "Temp": ["temperature", "temp"],
        "SepsisLabel": ["sepsislabel", "sepsis_label", "label"],
        "ICULOS": ["iculos"],
    }

    for physionet_name, candidates in reverse_aliases.items():
        if physionet_name not in df.columns:
            for c in candidates:
                if c in df.columns:
                    df[physionet_name] = df[c]
                    break

    return df

# ============================
# PLOTTING HELPERS
# ============================

def plot_patient(df_raw, title_suffix: str = "", figsize=(8, 4)):
    """
    Visualise a single patient with:
    - HR, Temp, SBP over time
    - qSOFA-like alert (Resp > 22 and SBP < 100, score >=2)
    - Sepsis onset (if any)
    - Early window = last 24h before onset, after hour 6
    """
    df = df_raw.copy()

    # Sort by time
    if "ICULOS" in df.columns:
        df = df.sort_values("ICULOS")

    # Make vitals numeric + fill gaps so lines are continuous
    for col in ["HR", "Temp", "SBP", "Resp"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .interpolate(limit_direction="both")
                .ffill()
                .bfill()
            )

    # Sepsis onset
    if "SepsisLabel" in df.columns:
        sepsis_indices = df.index[df["SepsisLabel"] == 1].tolist()
        sepsis_time = df.loc[sepsis_indices[0], "ICULOS"] if sepsis_indices else None
    else:
        sepsis_time = None

    # qSOFA-like rule
    if all(c in df.columns for c in ["Resp", "SBP"]):
        score_resp = (df["Resp"] > 22).astype(int)
        score_sbp = (df["SBP"] < 100).astype(int)
        qsofa_score = score_resp + score_sbp
        df["Risk"] = (qsofa_score >= 2)
    else:
        df["Risk"] = False

        # Ensure we have an ICULOS-like time column
    if "ICULOS" in df.columns:
        iculos = pd.to_numeric(df["ICULOS"], errors="coerce")
    else:
        # Create a synthetic ICU length-of-stay in hours based on row index
        df = df.copy()
        df["ICULOS"] = range(len(df))
        iculos = df["ICULOS"]

    # Filter to hours after 6h (if ICULOS is not usable, this will just act like an index filter)
    mask = iculos >= 6
    df_after6h = df[mask]


    if sepsis_time is not None:
        early_start = max(6, sepsis_time - 24)
        early_alerts = df_after6h[
            (df_after6h["Risk"])
            & (df_after6h["ICULOS"] >= early_start)
            & (df_after6h["ICULOS"] < sepsis_time)
        ]
    else:
        early_alerts = df_after6h[df_after6h["Risk"]]

    # Plotting
    fig, ax1 = plt.subplots(figsize=figsize)
    x = df["ICULOS"]

    # Left axis: HR + Temp
    if "HR" in df.columns:
        ax1.plot(x, df["HR"], label="HR", color="tab:blue", linewidth=2)
    if "Temp" in df.columns:
        ax1.plot(
            x, df["Temp"], label="Temp", color="tab:orange",
            linestyle="--", linewidth=2
        )

    ax1.set_xlabel("ICU Hours")
    ax1.set_ylabel("HR / Temp")
    ax1.grid(alpha=0.3)

    # Right axis: SBP
    ax2 = ax1.twinx()
    if "SBP" in df.columns:
        ax2.plot(x, df["SBP"], label="SBP", color="tab:red", linewidth=2)
    ax2.set_ylabel("SBP")

    # Sepsis onset line + early window
    if sepsis_time is not None:
        ax1.axvline(sepsis_time, color="red", linestyle="--", label="Sepsis onset")
        early_start = max(6, sepsis_time - 24)
        ax1.axvspan(
            early_start,
            sepsis_time,
            color="orange",
            alpha=0.15,
            label="Early window (last 24h before onset)",
        )

    # Early alerts as crosses
    if not early_alerts.empty:
        ymin, ymax = ax1.get_ylim()
        scatter_y = ymax - 0.02 * (ymax - ymin)
        ax1.scatter(
            early_alerts["ICULOS"],
            [scatter_y] * len(early_alerts),
            marker="x",
            color="red",
            label="Early alerts (qSOFA-like)",
        )
        ax1.set_ylim(ymin, ymax)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax1.set_title(title_suffix)
    fig.tight_layout()
    return fig


def plot_roc_curve(fpr, tpr, roc_auc, title="ROC Curve"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_early_hist(early_hours, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(early_hours, bins=20, alpha=0.7)
    ax.set_xlabel("Hours before sepsis onset")
    ax.set_ylabel("Count of patients")
    ax.set_title(f"Early detection distribution {title_suffix}")
    ax.grid(alpha=0.3)
    return fig

# ============================
# SAFE CSV LOADER
# ============================

def load_results_csv(path: str, description: str = "results"):
    """
    Safely load a CSV file used in Global Evaluation.
    - Returns a DataFrame if successful.
    - Returns None if something goes wrong (and shows a Streamlit message).
    """
    if not path:
        st.error(f"No path provided for {description} file.")
        return None

    if not os.path.exists(path):
        st.error(f"❌ Could not find the {description} file at `{path}`.")
        return None

    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        st.error(f"❌ The {description} file at `{path}` is empty.")
        return None
    except ParserError:
        st.error(
            f"❌ The {description} file at `{path}` could not be parsed. "
            f"Check that it is a valid CSV."
        )
        return None
    except UnicodeDecodeError:
        st.error(
            f"❌ The {description} file at `{path}` has an unexpected encoding. "
            f"Try saving it again in UTF-8 format."
        )
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error while loading {description} file at `{path}`:\n\n{e}")
        return None

    if df.empty:
        st.warning(f"⚠️ The {description} file at `{path}` was loaded but is empty.")

    return df

# ============================
# DEMO PSV LOADER
# ============================

@st.cache_data(show_spinner=False)
def load_demo_patient_psv(path: str, label: str = "demo patient"):
    """
    Cached loader for demo .psv files stored on disk.
    Uses '|' separator and logs any failure.
    """
    try:
        if not os.path.exists(path):
            debug_log(f"load_demo_patient_psv: file does not exist: {path}")
            return None

        df = pd.read_csv(path, sep="|")
        if df.empty:
            debug_log(f"load_demo_patient_psv: {label} at {path} loaded but is empty.")
        else:
            debug_log(f"load_demo_patient_psv: loaded {label} from {path} (rows={len(df)}).")
        return df

    except Exception as e:
        debug_log(f"load_demo_patient_psv: failed to load {label} from {path}: {e}")
        return None

# ============================
# CLINICAL-STYLE SUMMARY
# ============================

def build_clinical_summary(df: pd.DataFrame) -> str:
    """
    Build a short, non-clinical summary of vitals trends and qSOFA-like alerts
    for a single ICU stay. This is purely descriptive and for teaching / demo use.
    """
    lines = []

    # Heart rate trend
    if "HR" in df.columns:
        hr = pd.to_numeric(df["HR"], errors="coerce").dropna()
        if len(hr) > 1:
            hr_start, hr_end = hr.iloc[0], hr.iloc[-1]
            delta = hr_end - hr_start
            if delta > 15:
                lines.append("• Heart rate trends upward over the stay (tachycardic drift).")
            elif delta < -15:
                lines.append("• Heart rate trends downward over the stay.")
            else:
                lines.append("• Heart rate is relatively stable overall.")

    # Systolic blood pressure
    if "SBP" in df.columns:
        sbp = pd.to_numeric(df["SBP"], errors="coerce").dropna()
        if len(sbp) > 0:
            low_frac = (sbp < 100).mean()
            if low_frac > 0.4:
                lines.append("• Systolic BP is frequently below 100 mmHg (hypotensive periods).")
            elif low_frac > 0.1:
                lines.append("• There are intermittent dips in systolic BP below 100 mmHg.")
            else:
                lines.append("• Systolic BP is mostly above 100 mmHg.")

    # Temperature pattern
    if "Temp" in df.columns:
        temp = pd.to_numeric(df["Temp"], errors="coerce").dropna()
        if len(temp) > 0:
            mean_temp = temp.mean()
            if mean_temp >= 38.0:
                lines.append("• Temperature is on average febrile (≥ 38°C).")
            elif mean_temp >= 37.2:
                lines.append("• Temperature is slightly elevated but not frankly febrile.")
            else:
                lines.append("• Temperature stays mostly in the normal range.")

    # Sepsis label timing
    sepsis_time = None
    if "SepsisLabel" in df.columns:
        sepsis_idx = df.index[df["SepsisLabel"] == 1].tolist()
        if sepsis_idx and "ICULOS" in df.columns:
            try:
                sepsis_time = float(pd.to_numeric(df.loc[sepsis_idx[0], "ICULOS"], errors="coerce"))
            except Exception:
                sepsis_time = None
            if sepsis_time is not None:
                lines.append(f"• Sepsis label becomes positive around ICU hour {sepsis_time:.1f}.")
            else:
                lines.append("• Sepsis label becomes positive at some point during the stay.")
        elif sepsis_idx:
            lines.append("• Sepsis label becomes positive at some point during the stay.")
        else:
            lines.append("• No positive sepsis label in this stay.")

    # qSOFA-like alerts
    if all(c in df.columns for c in ["Resp", "SBP"]):
        resp = pd.to_numeric(df["Resp"], errors="coerce")
        sbp = pd.to_numeric(df["SBP"], errors="coerce")
        score_resp = (resp > 22).astype(int)
        score_sbp = (sbp < 100).astype(int)
        risk = (score_resp + score_sbp) >= 2

        if "ICULOS" in df.columns:
            df_tmp = df.copy()
            df_tmp["Risk"] = risk
            icu_time = pd.to_numeric(df_tmp["ICULOS"], errors="coerce")

            if sepsis_time is not None:
                early_start = max(6.0, sepsis_time - 24.0)
                mask = (icu_time >= early_start) & (icu_time < sepsis_time) & df_tmp["Risk"]
                early_count = int(mask.sum())
                if early_count > 0:
                    lines.append(
                        f"• qSOFA-like rule fires about {early_count} early alerts "
                        f"in the 6–24h window before onset."
                    )
                else:
                    lines.append(
                        "• qSOFA-like rule does not fire early alerts in the 6–24h window before onset."
                    )
            else:
                total_alerts = int(risk.sum())
                if total_alerts > 0:
                    lines.append(
                        f"• qSOFA-like rule fires alerts during the stay "
                        f"(about {total_alerts} time points flagged)."
                    )
                else:
                    lines.append("• qSOFA-like rule does not fire any alerts in this stay.")

    return "\n".join(lines)

# ============================
# GLOBAL EVAL UTILITIES
# ============================

def _find_score_col(df: pd.DataFrame):
    """Return a probability/score column name if present, else None."""
    for col in df.columns:
        low = col.lower()
        if ("prob" in low) or ("score" in low) or ("maxprob" in low) or ("maxscore" in low):
            return col
    return None


def safe_get(df: pd.DataFrame, col: str, default_value=None):
    """
    Return df[col] if it exists.
    If not:
      - return None if default_value is None
      - otherwise return a Series filled with default_value of length len(df)
    """
    if col in df.columns:
        return df[col]
    if default_value is None:
        debug_log(f"safe_get: column '{col}' missing and no default provided.")
        return None
    debug_log(f"safe_get: column '{col}' missing, returning default value '{default_value}'.")
    return pd.Series([default_value] * len(df))


def safe_numeric(series, default=0):
    """Convert a series to numeric safely, replacing invalid with default."""
    if series is None:
        debug_log("safe_numeric: received None series, returning single default value.")
        return pd.Series([default])
    s = pd.to_numeric(series, errors="coerce").fillna(default)
    return s


def ensure_binary(series, default=0):
    """Convert a series to 0/1 safely based on threshold 0.5."""
    if series is None:
        debug_log("ensure_binary: received None series, returning all default zeros.")
        return pd.Series([default])
    try:
        s = pd.to_numeric(series, errors="coerce").fillna(default)
        return (s >= 0.5).astype(int)
    except Exception as e:
        debug_log(f"ensure_binary: failed to convert series to binary: {e}")
        return pd.Series([default] * len(series))


MODEL_CONFIG = {
    "qsofa": {
        "name": "qSOFA-like rule",
        "csv_6h": QSOFA_CSV,
        "default_threshold": 0.5,
    },
    "xgb": {
        "name": "XGBoost",
        "csv_6h": XGB_CSV,
        "csv_24h": XGB_CSV_24,
        "default_threshold": 0.10,
    },
    "gru": {
        "name": "GRU",
        "csv_6h": GRU_CSV,
        "csv_24h": GRU_CSV_24,
        "default_threshold": 0.20,
    },
}


def get_score_series(df: pd.DataFrame, preferred_binary_col: str = "HasEarlyAlert"):
    """
    Standardized way to obtain a 'score' for ROC/thresholding.

    Returns:
      score_series: pd.Series (length = len(df))
      origin_name: str or None (column used)
      used_binary_fallback: bool
    """
    score_col = _find_score_col(df)
    if score_col:
        debug_log(f"get_score_series: using continuous score column '{score_col}'.")
        score = safe_numeric(df[score_col], default=0)
        return score, score_col, False

    # Fallback: binary column
    bin_series = safe_get(df, preferred_binary_col)
    if bin_series is not None:
        debug_log(f"get_score_series: falling back to binary column '{preferred_binary_col}'.")
        bin_series = ensure_binary(bin_series)
        return bin_series, preferred_binary_col, True

    # Last resort: zeros
    debug_log("get_score_series: no score-like or binary columns found; using all zeros.")
    zeros = pd.Series([0] * len(df))
    return zeros, None, True


def compute_confusion_and_rates(y_true_arr, score_arr, threshold):
    """
    Shared helper: given ground-truth labels and score+threshold,
    compute TP/FP/TN/FN, sensitivity, specificity, and alert rate.
    """
    y_true_arr = np.asarray(y_true_arr).astype(int)
    score_arr = np.asarray(score_arr, dtype=float)

    if y_true_arr.size == 0 or score_arr.size == 0:
        debug_log("compute_confusion_and_rates: empty y_true or score array.")
        return {
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "sens": float("nan"),
            "spec": float("nan"),
            "alert_rate": 0.0,
            "y_pred": np.array([], dtype=int),
        }

    y_pred = (score_arr >= threshold).astype(int)

    tp = int(((y_true_arr == 1) & (y_pred == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred == 0)).sum())
    tn = int(((y_true_arr == 0) & (y_pred == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    alert_rate = float(y_pred.mean() * 100.0)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sens": sens,
        "spec": spec,
        "alert_rate": alert_rate,
        "y_pred": y_pred,
    }


def find_best_threshold(y_true_arr, score_arr, metric: str = "youden"):
    """
    Simple threshold calibration helper using Youden's J.
    Returns best_threshold or None.
    """
    y_true_arr = np.asarray(y_true_arr).astype(int)
    score_arr = np.asarray(score_arr, dtype=float)

    if len(np.unique(y_true_arr)) < 2 or len(np.unique(score_arr)) < 2:
        debug_log("find_best_threshold: not enough variation to calibrate threshold.")
        return None

    try:
        fpr, tpr, thresholds = roc_curve(y_true_arr, score_arr)
        j = tpr - fpr
        idx = int(np.nanargmax(j))
        best_thr = float(thresholds[idx])
        debug_log(f"find_best_threshold: best threshold by Youden index = {best_thr:.4f}")
        return best_thr
    except Exception as e:
        debug_log(f"find_best_threshold: failed to compute best threshold: {e}")
        return None


@st.cache_data(show_spinner=False)
def compute_qsofa_metrics(df: pd.DataFrame):
    if len(df) == 0:
        debug_log("compute_qsofa_metrics: empty dataframe.")
        return {
            "roc_auc": float("nan"),
            "sens": float("nan"),
            "spec": float("nan"),
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "early_times": [],
        }

    y_col = safe_get(df, "HasSepsis")
    if y_col is None:
        debug_log("compute_qsofa_metrics: HasSepsis missing, defaulting to zeros.")
        y_true = pd.Series([0] * len(df), index=df.index)
    else:
        y_true = safe_numeric(y_col, default=0).astype(int)

    alert = safe_get(df, "HasEarlyAlert")
    if alert is None:
        debug_log("compute_qsofa_metrics: HasEarlyAlert missing, defaulting to zeros.")
        alert_bin = pd.Series([0] * len(df), index=df.index)
    else:
        alert_bin = ensure_binary(alert)

    thr = MODEL_CONFIG["qsofa"]["default_threshold"]
    metrics = compute_confusion_and_rates(y_true.values, alert_bin.values, threshold=thr)

    if "EarlyWarningHours" in df.columns:
        early_times = list(df["EarlyWarningHours"].dropna().values)
    else:
        early_times = []

    try:
        fpr, tpr, _ = roc_curve(y_true, alert_bin)
        roc_auc = auc(fpr, tpr)
    except Exception:
        debug_log("compute_qsofa_metrics: failed to compute ROC.")
        roc_auc = float("nan")

    return {
        "roc_auc": roc_auc,
        "sens": metrics["sens"],
        "spec": metrics["spec"],
        "tp": metrics["tp"], "fp": metrics["fp"],
        "tn": metrics["tn"], "fn": metrics["fn"],
        "early_times": early_times,
    }


@st.cache_data(show_spinner=False)
def compute_xgb_metrics(df: pd.DataFrame):
    if len(df) == 0:
        debug_log("compute_xgb_metrics: empty dataframe.")
        return {
            "roc_auc": float("nan"),
            "sens": float("nan"),
            "spec": float("nan"),
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "early_times": [],
        }

    y_col = safe_get(df, "HasSepsis")
    if y_col is None:
        debug_log("compute_xgb_metrics: HasSepsis missing, defaulting to zeros.")
        y_true = pd.Series([0] * len(df), index=df.index)
    else:
        y_true = safe_numeric(y_col, default=0).astype(int)

    score_series, origin_name, used_binary = get_score_series(df, preferred_binary_col="HasEarlyAlert")
    thr = MODEL_CONFIG["xgb"]["default_threshold"]
    metrics = compute_confusion_and_rates(y_true.values, score_series.values, threshold=thr)

    if "EarlyWarningHours" in df.columns:
        early_times = list(df["EarlyWarningHours"].dropna().values)
    else:
        early_times = []

    try:
        fpr, tpr, _ = roc_curve(y_true, score_series)
        roc_auc = auc(fpr, tpr)
    except Exception:
        debug_log("compute_xgb_metrics: failed to compute ROC.")
        roc_auc = float("nan")

    return {
        "roc_auc": roc_auc,
        "sens": metrics["sens"],
        "spec": metrics["spec"],
        "tp": metrics["tp"], "fp": metrics["fp"],
        "tn": metrics["tn"], "fn": metrics["fn"],
        "early_times": early_times,
    }


@st.cache_data(show_spinner=False)
def compute_gru_metrics(df: pd.DataFrame, threshold=None):
    if len(df) == 0:
        debug_log("compute_gru_metrics: empty dataframe.")
        return {
            "roc_auc": float("nan"),
            "sens": float("nan"),
            "spec": float("nan"),
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "early_times": [],
            "fpr": [], "tpr": [],
        }

    if threshold is None:
        threshold = MODEL_CONFIG["gru"]["default_threshold"]

    y_col = safe_get(df, "HasSepsis")
    if y_col is None:
        debug_log("compute_gru_metrics: HasSepsis missing, defaulting to zeros.")
        y_true = pd.Series([0] * len(df), index=df.index)
    else:
        y_true = safe_numeric(y_col, default=0).astype(int)

    score_series, origin_name, used_binary = get_score_series(df, preferred_binary_col="HasEarlyAlert")
    metrics = compute_confusion_and_rates(y_true.values, score_series.values, threshold=threshold)

    if "EarlyWarningHours" in df.columns:
        early_times = list(df["EarlyWarningHours"].dropna().values)
    else:
        early_times = []

    try:
        fpr, tpr, _ = roc_curve(y_true, score_series)
        roc_auc = auc(fpr, tpr)
    except Exception:
        debug_log("compute_gru_metrics: failed to compute ROC.")
        roc_auc = float("nan")
        fpr, tpr = [], []

    return {
        "roc_auc": roc_auc,
        "sens": metrics["sens"],
        "spec": metrics["spec"],
        "tp": metrics["tp"], "fp": metrics["fp"],
        "tn": metrics["tn"], "fn": metrics["fn"],
        "early_times": early_times,
        "fpr": fpr,
        "tpr": tpr,
    }


def balanced_sample(df, n_pos=10, n_neg=10):
    """
    Return a small balanced sample of rows: n_pos with HasSepsis=1, n_neg with HasSepsis=0.
    """
    if "HasSepsis" not in df.columns:
        return df.head(n_pos + n_neg)

    pos = df[df["HasSepsis"] == 1].sample(
        min(n_pos, (df["HasSepsis"] == 1).sum()), random_state=42
    )
    neg = df[df["HasSepsis"] == 0].sample(
        min(n_neg, (df["HasSepsis"] == 0).sum()), random_state=42
    )
    return pd.concat([pos, neg]).sample(frac=1, random_state=42)





# ============================
# NAV GUIDE + TABS
# ============================

with st.expander("🔍 What does EarlyPulse predict?", expanded=False):
    st.markdown(
        """
**Prediction task (for XGBoost and GRU models)**

Given ICU data up to time *t*, EarlyPulse predicts whether the patient will develop sepsis within the next 6 hours.

- The first 6 hours of each ICU stay are ignored for evaluation.
- Sepsis onset = first hour where `SepsisLabel == 1`.
- An alert is **early** if it fires after hour 6 and within 24 hours before onset.
- For non-sepsis patients, any alert after hour 6 = false positive.
        """
    )

# ── Navigation guide ─────────────────────────────────────
st.markdown(
    """
    <div class="nav-strip">
        <div class="nav-card">
            <span class="nav-num">01 ·· DEMO</span>
            <div class="nav-title">🧪 Demo Patients</div>
            <div class="nav-desc">5 curated ICU stories. Vitals, onset timing, and rule-based alerts.</div>
        </div>
        <div class="nav-card">
            <span class="nav-num">02 ·· UPLOAD</span>
            <div class="nav-title">🩺 Doctor Mode</div>
            <div class="nav-desc">Upload any .psv or .csv ICU stay. Compare two patients side by side.</div>
        </div>
        <div class="nav-card">
            <span class="nav-num">03 ·· EVAL</span>
            <div class="nav-title">📊 Global Evaluation</div>
            <div class="nav-desc">Compare all 3 models on the full cohort. Tune decision thresholds.</div>
        </div>
        <div class="nav-card">
            <span class="nav-num">04 ·· SAFETY</span>
            <div class="nav-title">⚠️ Limitations</div>
            <div class="nav-desc">What this prototype can and cannot say about real-world patients.</div>
        </div>
    </div>
    <div class="ep-rule" style="margin-top:0.8rem;margin-bottom:1.2rem;"></div>
    """,
    unsafe_allow_html=True,
)

# ── Main tabs ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🧪 Demo Patients",
        "🩺 Doctor Mode",
        "📊 Global Evaluation",
        "⚠️ Limitations",
    ]
)

# ============================
# TAB 1 — DEMO PATIENTS
# ============================
# TAB 1 — DEMO PATIENTS
# ============================

with tab1:
    # Header card
    st.markdown(
        """
        <div class="glass-info-card">
            <div class="card-icon">🧪</div>
            <div class="card-title">Demo Patients — Simulated ICU Stories</div>
            <p class="card-desc">
                Pick a curated patient and see vitals, sepsis onset, and when the qSOFA-like rule fires.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_pat_sel, col_plot = st.columns([1, 2])

    # LEFT: selection + story
    with col_pat_sel:
        choice = st.selectbox("Choose a demo patient", list(HERO_PATIENTS.keys()))

        demo_filename = HERO_PATIENTS[choice]
        file_for_demo = os.path.join(DATA_DIR, demo_filename)

        st.caption(
            "These patients highlight early vs late onset, noisy vitals, and borderline cases."
        )

        story = HERO_PATIENT_STORIES.get(choice)
        if story:
            st.markdown(
                f"""
                <div style="
                    margin-top:0.75rem;
                    padding:0.85rem 1.1rem;
                    border-radius:10px;
                    background:#0b1828;
                    border:1px solid #0f2340;
                    font-size:0.87rem;
                    line-height:1.6;
                    color:#b8d0f0;
                ">
                    <span style="font-weight:600;color:#ffffff;">Patient story</span><br/>{story}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # RIGHT: plot
    with col_plot:
        if os.path.exists(file_for_demo):
            try:
                df_demo = pd.read_csv(file_for_demo, sep="|")
                if df_demo.empty:
                    st.warning("Selected demo file is empty.")
                else:
                    fig_demo = plot_patient(df_demo, title_suffix=choice)
                    st.pyplot(fig_demo, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load demo patient file: {e}")
        else:
            st.error(f"File not found: {file_for_demo}")

# ============================
# TAB 2 — DOCTOR MODE (UPLOAD)
# ============================
# TAB 2 — DOCTOR MODE (UPLOAD)
# ============================
# TAB 2 — DOCTOR MODE (UPLOAD)
# ============================

with tab2:
    # Header card
    st.markdown(
        """
        <div class="glass-info-card">
            <div class="card-icon">🩺</div>
            <div class="card-title">Doctor Mode — Upload ICU Stays</div>
            <p class="card-desc">
                Upload one or two ICU time-series files (<code>.csv</code> or <code>.psv</code>). 
                EarlyPulse automatically detects separators, normalises column names, and plots available vitals.
                This is a non-clinical, research-only tool.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Any ICU file will work as long as it roughly follows an hourly time-series structure. "
        "EarlyPulse tries to map common aliases (HR, RR, SBP, SpO₂, etc.) to its internal schema and "
        "will warn you when key variables are missing."
    )

    st.markdown("#### 1. Choose mode")

    mode_col, info_col = st.columns([1, 2])

    with mode_col:
        compare_mode = st.checkbox(
            "Compare two uploaded patients side by side",
            value=False,
            key="doctor_compare_mode",
        )

    with info_col:
        st.markdown(
            """
            <p style="font-size:0.85rem;color:#6a90b8;margin-top:0.2rem;">
            • <strong>Off</strong> → upload a <em>single</em> ICU stay and inspect it in depth.<br>
            • <strong>On</strong> → upload <em>two</em> ICU stays and compare their trajectories and alerts.
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Vitals we care about (for warnings)
    vital_cols = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "map", "temperature", "sepsislabel"]

    # ---------------------------------------------------------------
    # SINGLE-PATIENT MODE
    # ---------------------------------------------------------------
    if not compare_mode:
        st.markdown("#### 2. Upload a single ICU stay")

        col_left, col_right = st.columns([1, 2])

        with col_left:
            uploaded_file = st.file_uploader(
                "Upload one .csv or .psv file (e.g. p000001.psv)",
                type=["csv", "psv"],
                key="single_psv_upload",
            )
            st.caption(
                "EarlyPulse will auto-detect the separator and map common column names. "
                "If some vitals are missing, they will simply not be plotted."
            )

        df_upload = None  # initialise

        with col_right:
            if uploaded_file is not None:
                df_upload = load_any_patient_file(uploaded_file)
                if df_upload is None:
                    st.error("Could not parse this file. Please check the format or try another file.")
                else:
                    # Warn about missing key signals
                    missing = [c for c in vital_cols if c in df_upload.columns and df_upload[c].isna().all()]
                    truly_missing = [c for c in vital_cols if c not in df_upload.columns]
                    missing_any = set(missing) | set(truly_missing)
                    if missing_any:
                        pretty = ", ".join(sorted(missing_any))
                        st.warning(
                            "This file is missing usable data for: "
                            f"{pretty}. EarlyPulse will plot the available signals only."
                        )

                    if df_upload.empty:
                        st.warning("Uploaded file appears to be empty.")
                    else:
                        fig_upload = plot_patient(
                            df_upload,
                            title_suffix=f"Uploaded: {uploaded_file.name}",
                        )
                        st.pyplot(fig_upload, use_container_width=True)

                        # Optional: download plot as PNG
                        import io

                        buf = io.BytesIO()
                        fig_upload.savefig(buf, format="png", bbox_inches="tight")
                        st.download_button(
                            "⬇️ Download this plot as PNG",
                            data=buf.getvalue(),
                            file_name=f"earlypulse_upload_{uploaded_file.name}.png",
                            mime="image/png",
                            key="download_single_upload_plot",
                        )

        if uploaded_file is not None and df_upload is not None:
            st.markdown("#### 3. Clinical-style summary")
            st.caption(
                "Approximate, non-clinical description of trends and qSOFA-like alerts for this ICU stay."
            )
            try:
                summary_text = build_clinical_summary(df_upload)
                if summary_text:
                    st.markdown(
                        f"""
                        <div style="
                            margin-top:0.5rem;
                            padding:0.9rem 1.1rem;
                            border-radius:12px;
                            background:#0b1828;
                            border:1px solid #0f2340;
                            font-size:0.87rem;
                            line-height:1.5;
                            color:#b8d0f0;
                        ">{summary_text}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(
                        "Not enough information to build a meaningful summary for this patient."
                    )
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
        elif uploaded_file is None:
            st.caption(
                "Upload a .csv or .psv file above to see the trajectory and a generated summary."
            )

    # ---------------------------------------------------------------
    # COMPARE-MODE
    # ---------------------------------------------------------------
    else:
        st.markdown("#### 2. Upload two ICU stays to compare")

        colA, colB = st.columns(2)

        with colA:
            file_A = st.file_uploader(
                "Patient A — .csv or .psv file",
                type=["csv", "psv"],
                key="compare_psv_A",
            )
        with colB:
            file_B = st.file_uploader(
                "Patient B — .csv or .psv file",
                type=["csv", "psv"],
                key="compare_psv_B",
            )

        df_A, df_B = None, None

        if file_A is not None and file_B is not None:
            df_A = load_any_patient_file(file_A)
            df_B = load_any_patient_file(file_B)

            if df_A is None or df_B is None:
                st.error("Could not parse one of the uploaded files. Please check formats and try again.")
            else:
                col_left, col_right = st.columns(2)

                # Patient A
                with col_left:
                    st.markdown(f"##### 🩺 Patient A — {file_A.name}")
                    missing_A = [c for c in vital_cols if (c not in df_A.columns) or df_A[c].isna().all()]
                    if missing_A:
                        st.caption(
                            "Missing / unusable for Patient A: " + ", ".join(sorted(missing_A))
                        )
                    if df_A.empty:
                        st.warning("Patient A file is empty.")
                    else:
                        fig_A = plot_patient(
                            df_A, title_suffix=f"Patient A: {file_A.name}"
                        )
                        st.pyplot(fig_A, use_container_width=True)

                # Patient B
                with col_right:
                    st.markdown(f"##### 🩺 Patient B — {file_B.name}")
                    missing_B = [c for c in vital_cols if (c not in df_B.columns) or df_B[c].isna().all()]
                    if missing_B:
                        st.caption(
                            "Missing / unusable for Patient B: " + ", ".join(sorted(missing_B))
                        )
                    if df_B.empty:
                        st.warning("Patient B file is empty.")
                    else:
                        fig_B = plot_patient(
                            df_B, title_suffix=f"Patient B: {file_B.name}"
                        )
                        st.pyplot(fig_B, use_container_width=True)

                st.markdown("#### 3. Auto-generated summaries")
                st.caption(
                    "Very rough, non-clinical summaries for each patient to highlight differences in trends and alerts."
                )

                sum_colA, sum_colB = st.columns(2)

                with sum_colA:
                    try:
                        summary_A = build_clinical_summary(df_A)
                        if summary_A:
                            st.markdown(
                                f"""
                                <div style="
                                    margin-top:0.5rem;
                                    padding:0.9rem 1.1rem;
                                    border-radius:12px;
                                    background:#0b1828;
                                    border:1px solid #0f2340;
                                    font-size:0.87rem;
                                    line-height:1.5;
                                    color:#b8d0f0;
                                ">{summary_A}</div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption(
                                "Not enough information to summarise Patient A."
                            )
                    except Exception as e:
                        st.error(f"Failed to summarise Patient A: {e}")

                with sum_colB:
                    try:
                        summary_B = build_clinical_summary(df_B)
                        if summary_B:
                            st.markdown(
                                f"""
                                <div style="
                                    margin-top:0.5rem;
                                    padding:0.9rem 1.1rem;
                                    border-radius:12px;
                                    background:#0b1828;
                                    border:1px solid #0f2340;
                                    font-size:0.87rem;
                                    line-height:1.5;
                                    color:#b8d0f0;
                                ">{summary_B}</div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption(
                                "Not enough information to summarise Patient B."
                            )
                    except Exception as e:
                        st.error(f"Failed to summarise Patient B: {e}")

        else:
            st.caption(
                "Upload two .csv or .psv files above to compare trajectories and summaries side by side."
            )

# ============================
# ============================
# ============================
# TAB 3 — GLOBAL EVALUATION
# ============================

with tab3:
    st.markdown("### Global Evaluation — Full-Cohort Baselines")
    st.caption(
        "Compare qSOFA, XGBoost, and GRU on the full PhysioNet 2019 ICU cohort. "
        "Tune thresholds and see how each baseline behaves."
    )

    # (Old ep-card CSS – still harmless to keep)
    st.markdown(
        """
        <style>
        .ep-card {
            background: #0b1828;
            border-radius: 8px;
            padding: 8px 12px;
            margin: 0 auto 10px auto;
            max-width: 960px;
            border: 1px solid rgba(120, 144, 156, 0.25);
            box-shadow: 0 1px 2px rgba(120, 144, 156, 0.18);
        }
        .ep-sub {
            color: #4a6f96;
            font-size: 0.85rem;
            margin-top: 2px;
            margin-bottom: 0;
        }
        .ep-header {
            color: #d8eaff;
            margin-bottom: 2px;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .ep-caption {
            color: #4a6f96;
            font-size: 0.78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Inner tabs for this section
    comparison_tab, models_tab = st.tabs(["📊 Comparison", "🧬 Models"])

    # ------------------------------------------------------------------
  # 📊 COMPARISON TAB  (KEEP NEW LOOK)
# ------------------------------------------------------------------
with comparison_tab:

    # New glass header card
    st.markdown(
        """
        <div class="glass-info-card">
            <div class="card-icon">📊</div>
            <div class="card-title">Overall Model Comparison (Patient-Level)</div>
            <p class="card-desc">
                Compare AUROC, sensitivity, specificity, early detections, and confusion-matrix counts
                across qSOFA, XGBoost, and GRU on the full 20k+ ICU cohort.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ How to read these numbers", expanded=False):
        st.markdown(
            """
- **AUROC**: ranking quality (0.5 = random, 1.0 = perfect).  
- **Sensitivity**: fraction of true sepsis patients correctly flagged.  
- **Specificity**: fraction of non-sepsis patients correctly not flagged.  
- **Early detections**: sepsis cases where the model fired within the early window.  
- **TP/FP/TN/FN**: confusion-matrix counts at the default threshold.
            """
        )

    comparison_rows = []
    mq = mx = mg = None

    q_cfg = MODEL_CONFIG["qsofa"]
    x_cfg = MODEL_CONFIG["xgb"]
    g_cfg = MODEL_CONFIG["gru"]

    # ---- qSOFA ----
    if os.path.exists(q_cfg["csv_6h"]):
        df_q_comp = load_results_csv(q_cfg["csv_6h"], "qSOFA results")
        if df_q_comp is not None:
            mq = compute_qsofa_metrics(df_q_comp)
            if isinstance(mq, dict):
                comparison_rows.append({
                    "Model": "qSOFA-like rule",
                    "AUROC": None if pd.isna(mq["roc_auc"]) else round(mq["roc_auc"], 4),
                    "AUPRC": 0.0861,
                    "Sensitivity (%)": None if pd.isna(mq["sens"]) else round(mq["sens"] * 100, 2),
                    "Specificity (%)": None if pd.isna(mq["spec"]) else round(mq["spec"] * 100, 2),
                    "Early detections": len(mq.get("early_times", [])),
                    "TP": mq.get("tp", 0),
                    "FP": mq.get("fp", 0),
                    "TN": mq.get("tn", 0),
                    "FN": mq.get("fn", 0),
                })
            else:
                st.warning("qSOFA metrics returned None.")
        else:
            st.warning("Could not load qSOFA CSV.")
    else:
        st.warning(f"Missing: {q_cfg['csv_6h']}")


  # ---- XGBoost ----
    if os.path.exists(x_cfg["csv_6h"]):
        df_x_comp = load_results_csv(x_cfg["csv_6h"], "XGBoost results")
        if df_x_comp is not None:
            mx = compute_xgb_metrics(df_x_comp)
            if isinstance(mx, dict):
                comparison_rows.append({
                    "Model": "XGBoost (6h, thr=0.10)",
                    "AUROC": None if pd.isna(mx["roc_auc"]) else round(mx["roc_auc"], 4),
                    "AUPRC": 0.7030,
                    "Sensitivity (%)": None if pd.isna(mx["sens"]) else round(mx["sens"] * 100, 2),
                    "Specificity (%)": None if pd.isna(mx["spec"]) else round(mx["spec"] * 100, 2),
                    "Early detections": len(mx.get("early_times", [])),
                    "TP": mx.get("tp", 0),
                    "FP": mx.get("fp", 0),
                    "TN": mx.get("tn", 0),
                    "FN": mx.get("fn", 0),
                })
            else:
                st.warning("XGBoost metrics returned None.")
        else:
            st.warning("Could not load XGBoost CSV.")
    else:
        st.warning(f"Missing: {x_cfg['csv_6h']}")


    # ---- GRU ----
    if os.path.exists(g_cfg["csv_6h"]):
        df_g_comp = load_results_csv(g_cfg["csv_6h"], "GRU results")
        if df_g_comp is not None:
            mg = compute_gru_metrics(df_g_comp)
            if isinstance(mg, dict):
                comparison_rows.append({
                    "Model": "GRU (6h, thr=0.20)",
                    "AUROC": None if pd.isna(mg["roc_auc"]) else round(mg["roc_auc"], 4),
                    "AUPRC": 0.3535,
                    "Sensitivity (%)": None if pd.isna(mg["sens"]) else round(mg["sens"] * 100, 2),
                    "Specificity (%)": None if pd.isna(mg["spec"]) else round(mg["spec"] * 100, 2),
                    "Early detections": len(mg.get("early_times", [])),
                    "TP": mg.get("tp", 0),
                    "FP": mg.get("fp", 0),
                    "TN": mg.get("tn", 0),
                    "FN": mg.get("fn", 0),
                })
            else:
                st.warning("GRU metrics returned None.")
        else:
            st.warning("Could not load GRU CSV.")
    else:
        st.warning(f"Missing: {g_cfg['csv_6h']}")

        

    # ---- SHOW COMPARISON TABLE ----
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)

        # Small leaderboard card
        if "AUROC" in comp_df.columns and comp_df["AUROC"].notna().any():
            best_idx = comp_df["AUROC"].astype(float).idxmax()
            best_row = comp_df.loc[best_idx]
            best_name = best_row["Model"]
            best_auroc = best_row["AUROC"]

            st.markdown(
                f"""
                <div class="leaderboard-card">
                    <span class="lb-trophy">🏆</span>
                    <div>
                        <div class="lb-label">Best patient-level AUROC</div>
                        <div class="lb-name">{best_name}</div>
                    </div>
                    <div class="lb-score">AUROC = {best_auroc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.dataframe(comp_df, use_container_width=True)

    else:
        st.warning("No comparison data found. Check that your result CSV files exist in the configured folder.")

   

    #################################
     # ----- MODELS TAB -----

with models_tab:

    st.markdown(
        """
        <div class="glass-info-card">
            <div class="card-icon">🧬</div>
            <div class="card-title">Model Drill-Down</div>
            <p class="card-desc">
                Adjust thresholds and inspect how each model behaves on the full ICU cohort.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_choice = st.selectbox(
        "Select model",
        [
            "qSOFA-like rule",
            "XGBoost (6h horizon, thr=0.10)",
            "GRU (6h horizon, thr=0.20)",
        ],
        key="drill_model_select"
    )
    # -------------------------
    # -------------------------
    # qSOFA DETAIL
    # -------------------------
    if model_choice == "qSOFA-like rule":
            st.markdown("#### qSOFA-like rule (full cohort)")
            st.caption("Evaluated on the full cohort (~20k stays).")

            # Model card
            with st.expander("📇 Model card — qSOFA-like rule", expanded=False):
                st.markdown(
                    """
**Type:** Simple bedside rule (qSOFA ≥ 2).  
**Inputs:** Respiratory rate, systolic blood pressure, mental status (GCS).  

**Strengths:**
- Extremely easy to calculate at the bedside.
- Transparent: clinicians can see exactly why an alert fired.
- No need for complex infrastructure or ML models.

**Limitations:**
- Lower sensitivity for early or subtle sepsis cases.
- Ignores richer ICU data (labs, trends, comorbidities).
- Designed as a quick screen, not a full early-warning system.
                    """
                )

            # Rule explanation
            with st.expander("🔍 How does a qSOFA alert fire?", expanded=False):
                st.markdown(
                    """
qSOFA uses three simple bedside checks:

- **Respiratory rate > 22 /min** → 1 point  
- **Systolic BP < 100 mmHg** → 1 point  
- **Altered mental status (GCS < 15)** → 1 point  

The **qSOFA score = sum of these points (0–3)**.

In EarlyPulse, we consider that a **qSOFA alert fires when:**

> **qSOFA score ≥ 2** at any time in the early-warning window.

On this tab we only see the **aggregated patient-level result** (whether an early alert happened),
not the exact hour-by-hour reasoning for each individual stay.
                    """
                )

            if os.path.exists(MODEL_CONFIG["qsofa"]["csv_6h"]):
                df_q = load_results_csv(MODEL_CONFIG["qsofa"]["csv_6h"], "qSOFA patient-level results")
                if df_q is None:
                    st.warning("Cannot show qSOFA model details because the CSV could not be loaded.")
                else:
                    score_series, origin_name, used_binary = get_score_series(
                        df_q, preferred_binary_col="HasEarlyAlert"
                    )

                    if origin_name is not None and not used_binary:
                        score_label = f"Using score column `{origin_name}` for ROC/thresholding."
                    elif origin_name is not None and used_binary:
                        score_label = (
                            f"No continuous score found — using binary `{origin_name}` as "
                            "synthetic score (slider will be step-like)."
                        )
                    else:
                        score_label = (
                            "No score-like columns found — using constant score "
                            "(model has no discrimination here)."
                        )

                    if len(score_series) == 0:
                        st.error("No score values available in qSOFA CSV.")
                    else:
                        score = score_series.values
                        st.caption(score_label)

                        thr_q = st.slider(
                            "qSOFA decision threshold (on score)",
                            0.0, 1.0, MODEL_CONFIG["qsofa"]["default_threshold"], 0.01,
                            key="q_thr",
                        )

                        if "HasSepsis" not in df_q.columns:
                            st.error("`HasSepsis` column missing in qSOFA CSV; cannot compute metrics.")
                        else:
                            y_true = safe_numeric(df_q["HasSepsis"], default=0).astype(int).values
                            metrics_thr = compute_confusion_and_rates(y_true, score, threshold=thr_q)
                            sens = metrics_thr["sens"]
                            spec = metrics_thr["spec"]
                            alert_rate = metrics_thr["alert_rate"]

                            if len(pd.unique(y_true)) > 1 and len(pd.unique(score)) > 1:
                                fpr, tpr, _ = roc_curve(y_true, score)
                                roc_auc = auc(fpr, tpr)
                            else:
                                fpr, tpr = [0, 1], [0, 1]
                                roc_auc = float("nan")

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("AUROC", f"{roc_auc:.4f}")
                            c2.metric("Sensitivity", f"{sens * 100:.2f}%")
                            c3.metric("Specificity", f"{spec * 100:.2f}%")
                            c4.metric(
                                "Early detections",
                                str(len(df_q.loc[df_q.get("HasEarlyAlert", 0) == 1]))
                            )

                            # Risk level badge
                            if alert_rate < 10:
                                risk_label = "🟩 Low alert level"
                                risk_text = "Only a small fraction of patients are flagged — low alert burden."
                            elif alert_rate <= 25:
                                risk_label = "🟨 Moderate alert level"
                                risk_text = "Balanced between catching sepsis and keeping alerts manageable."
                            else:
                                risk_label = "🟥 High alert level"
                                risk_text = "Many patients are flagged — stronger safety net but risk of alert fatigue."

                            st.markdown(
                                f"""
                                <p class="ep-caption">
                                <strong>Risk level:</strong> {risk_label} – {risk_text} (alert rate {alert_rate:.2f}%)
                                </p>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Key takeaway
                            st.markdown(
                                """
                                <p class="ep-caption">
                                <strong>Key takeaway:</strong> Very interpretable bedside rule, but it tends to miss many early sepsis cases.
                                </p>
                                """,
                                unsafe_allow_html=True,
                            )

                           
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### Early Warning Distribution")
                                if "EarlyWarningHours" in df_q.columns:
                                    st.pyplot(
                                        plot_early_hist(
                                            df_q.loc[
                                                (df_q.get("HasSepsis", 0) == 1)
                                                & (df_q.get("HasEarlyAlert", 0) == 1),
                                                "EarlyWarningHours",
                                            ].dropna().values,
                                            title_suffix="qSOFA",
                                        ),
                                        use_container_width=True,
                                    )
                                else:
                                    st.info("No EarlyWarningHours column in qSOFA CSV.")
                            with col2:
                                st.markdown("#### ROC Curve")
                                if not pd.isna(roc_auc):
                                    st.pyplot(
                                        plot_roc_curve(fpr, tpr, roc_auc, title="qSOFA ROC"),
                                        use_container_width=True,
                                    )
                                else:
                                    st.info("ROC can't be computed (no variation in score or labels).")

                            st.markdown("#### Sample of Global Results (balanced: 10 sepsis / 10 non-sepsis)")
                            st.dataframe(
                                balanced_sample(df_q, n_pos=10, n_neg=10),
                                use_container_width=True,
                            )
                            st.caption(
                                "Balanced sample shown to avoid class imbalance hiding model behaviour "
                                "(10 sepsis, 10 non-sepsis)."
                            )
            else:
                st.error(
                    f"Global results CSV not found: `{MODEL_CONFIG['qsofa']['csv_6h']}`. "
                    "Run qSOFA evaluation first."
                )

        # -------------------------
        # XGBoost DETAIL
        # -------------------------
    elif model_choice == "XGBoost (6h horizon, thr=0.10)":
            st.markdown("#### XGBoost (6h horizon)")
            st.caption("Evaluated on the XGBoost patient-level results.")

            # Model card
            with st.expander("📇 Model card — XGBoost", expanded=False):
                st.markdown(
                    """
**Type:** Gradient-boosted decision trees on engineered features.  
**Inputs:** Aggregated vitals and time-window features (e.g. min/max/mean HR, SBP, Temp, labs if available).  

**Strengths:**
- Captures non-linear interactions between variables.
- Usually higher AUROC and sensitivity than simple rules.
- Still relatively fast and lightweight to run in production.

**Limitations:**
- Less transparent than qSOFA — harder to explain individual decisions.
- Needs careful training and validation on high-quality labelled data.
- Performance can degrade if deployment data drifts from training data.
                    """
                )

            if os.path.exists(MODEL_CONFIG["xgb"]["csv_6h"]):
                df_x = load_results_csv(MODEL_CONFIG["xgb"]["csv_6h"], "XGBoost patient-level results")
                if df_x is None:
                    st.warning("Cannot show XGBoost model details because the CSV could not be loaded.")
                else:
                    score_series, origin_name, used_binary = get_score_series(
                        df_x, preferred_binary_col="HasEarlyAlert"
                    )

                    if origin_name is not None and not used_binary:
                        score_label = f"Using score column `{origin_name}` for ROC/thresholding."
                    elif origin_name is not None and used_binary:
                        score_label = (
                            f"No continuous score found — using binary `{origin_name}` as "
                            "synthetic score (slider will be step-like)."
                        )
                    else:
                        score_label = (
                            "No score-like columns found — using constant score "
                            "(model has no discrimination here)."
                        )

                    if len(score_series) == 0:
                        st.error("No score values available in XGBoost CSV.")
                    else:
                        score = score_series.values
                        st.caption(score_label)

                        thr_x = st.slider(
                            "XGBoost decision threshold (on score)",
                            0.0, 1.0, MODEL_CONFIG["xgb"]["default_threshold"], 0.01,
                            key="x_thr",
                        )

                        if "HasSepsis" not in df_x.columns:
                            st.error("`HasSepsis` column missing in XGBoost CSV; cannot compute metrics.")
                        else:
                            y_true = safe_numeric(df_x["HasSepsis"], default=0).astype(int).values
                            metrics_thr = compute_confusion_and_rates(y_true, score, threshold=thr_x)
                            sens = metrics_thr["sens"]
                            spec = metrics_thr["spec"]
                            alert_rate = metrics_thr["alert_rate"]

                            best_thr = find_best_threshold(y_true, score)
                            if best_thr is not None:
                                st.markdown(
                                    f"""
                                    <p class="ep-caption">
                                    <strong>Suggested threshold (Youden index):</strong> {best_thr:.2f}  
                                    <span style="opacity:0.8;">Balances sensitivity and specificity on this cohort.</span>
                                    </p>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            if len(pd.unique(y_true)) > 1 and len(pd.unique(score)) > 1:
                                fpr, tpr, _ = roc_curve(y_true, score)
                                roc_auc = auc(fpr, tpr)
                            else:
                                fpr, tpr = [0, 1], [0, 1]
                                roc_auc = float("nan")

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("AUROC", f"{roc_auc:.4f}")
                            c2.metric("Sensitivity", f"{sens * 100:.2f}%")
                            c3.metric("Specificity", f"{spec * 100:.2f}%")
                            c4.metric(
                                "Early detections",
                                str(len(df_x.loc[df_x.get("HasEarlyAlert", 0) == 1]))
                            )

                            # Risk level badge
                            if alert_rate < 10:
                                risk_label = "🟩 Low alert level"
                                risk_text = "Only a small fraction of patients are flagged — low alert burden."
                            elif alert_rate <= 25:
                                risk_label = "🟨 Moderate alert level"
                                risk_text = "Balanced between catching sepsis and keeping alerts manageable."
                            else:
                                risk_label = "🟥 High alert level"
                                risk_text = "Many patients are flagged — stronger safety net but risk of alert fatigue."

                            st.markdown(
                                f"""
                                <p class="ep-caption">
                                <strong>Risk level:</strong> {risk_label} – {risk_text} (alert rate {alert_rate:.2f}%)
                                </p>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Key takeaway
                            st.markdown(
                                """
                                <p class="ep-caption">
                                <strong>Key takeaway:</strong> Catches more sepsis cases earlier than qSOFA with a moderate alert burden.
                                </p>
                                """,
                                unsafe_allow_html=True,
                            )

                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### Early Warning Distribution")
                                if "EarlyWarningHours" in df_x.columns:
                                    st.pyplot(
                                        plot_early_hist(
                                            df_x.loc[
                                                (df_x.get("HasSepsis", 0) == 1)
                                                & (df_x.get("HasEarlyAlert", 0) == 1),
                                                "EarlyWarningHours",
                                            ].dropna().values,
                                            title_suffix="XGBoost",
                                        ),
                                        use_container_width=True,
                                    )
                                else:
                                    st.info("No EarlyWarningHours column in XGBoost CSV.")
                            with col2:
                                st.markdown("#### ROC Curve")
                                if not pd.isna(roc_auc):
                                    st.pyplot(
                                        plot_roc_curve(fpr, tpr, roc_auc, title="XGBoost ROC"),
                                        use_container_width=True,
                                    )
                                else:
                                    st.info("ROC can't be computed (no variation in score or labels).")

                            st.markdown(
                                "#### Sample of XGBoost Patient-Level Results (balanced: 10 sepsis / 10 non-sepsis)"
                            )
                            st.dataframe(
                                balanced_sample(df_x, n_pos=10, n_neg=10),
                                use_container_width=True,
                            )
                            st.caption(
                                "Balanced sample shown to avoid class imbalance hiding model behaviour "
                                "(10 sepsis, 10 non-sepsis)."
                            )
            else:
                st.error(
                    f"XGBoost results CSV not found: `{MODEL_CONFIG['xgb']['csv_6h']}`. "
                    "Run XGBoost evaluation first."
                )

        # -------------------------
        # GRU DETAIL
        # -------------------------
    else:  # GRU
            st.markdown("#### GRU (6h horizon)")
            st.caption(
                "Evaluated on the full GRU dataset. Use the slider to change the decision "
                "threshold on MaxProbInWindow."
            )

            # Model card
            with st.expander("📇 Model card — GRU", expanded=False):
                st.markdown(
                    """
**Type:** Recurrent neural network (Gated Recurrent Unit) on ICU time-series.  
**Inputs:** Full temporal trajectories of vitals (and optionally labs), hour by hour.  

**Strengths:**
- Looks at the **sequence** of measurements, not just summarised windows.
- Can detect early changes in trends (e.g. rising HR + falling BP).
- Can capture temporal patterns (rising HR + falling BP trends) that static models miss.

**Limitations:**
- Much less transparent — individual predictions are hard to fully explain.
- Requires more data, GPU training, and careful regularisation.
- More sensitive to dataset shift and missing-data patterns.
                    """
                )

            if os.path.exists(MODEL_CONFIG["gru"]["csv_6h"]):
                df_g = load_results_csv(MODEL_CONFIG["gru"]["csv_6h"], "GRU patient-level results")
                if df_g is None:
                    st.warning("Cannot show GRU model details because the CSV could not be loaded.")
                else:
                    thr = st.slider(
                        "Decision threshold on GRU probability (MaxProbInWindow)",
                        min_value=0.0,
                        max_value=1.0,
                        value=MODEL_CONFIG["gru"]["default_threshold"],
                        step=0.01,
                        key="gru_thr",
                    )

                    metrics_g = compute_gru_metrics(df_g, threshold=thr)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("AUROC (patient-level)", f"{metrics_g['roc_auc']:.4f}")
                    c2.metric("Sensitivity", f"{metrics_g['sens'] * 100:.2f}%")
                    c3.metric("Specificity", f"{metrics_g['spec'] * 100:.2f}%")
                    c4.metric(
                        "Early detections (baseline early-window logic)",
                        str(len(metrics_g["early_times"])),
                    )

                    # Alert rate for GRU
                    y_true_gru = safe_numeric(
                        df_g.get("HasSepsis", pd.Series([0] * len(df_g))), default=0
                    ).astype(int).values
                    score_series_gru, origin_name_gru, used_binary_gru = get_score_series(
                        df_g, preferred_binary_col="HasEarlyAlert"
                    )
                    metrics_alert_gru = compute_confusion_and_rates(
                        y_true_gru, score_series_gru.values, threshold=thr
                    )
                    alert_rate_gru = metrics_alert_gru["alert_rate"]

                    # Risk level badge
                    if alert_rate_gru < 10:
                        risk_label = "🟩 Low alert level"
                        risk_text = "Only a small fraction of patients are flagged — low alert burden."
                    elif alert_rate_gru <= 25:
                        risk_label = "🟨 Moderate alert level"
                        risk_text = "Balanced between catching sepsis and keeping alerts manageable."
                    else:
                        risk_label = "🟥 High alert level"
                        risk_text = "Many patients are flagged — stronger safety net but risk of alert fatigue."

                    st.markdown(
                        f"""
                        <p class="ep-caption">
                        <strong>Risk level:</strong> {risk_label} – {risk_text} (alert rate {alert_rate_gru:.2f}%)
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Suggested threshold badge
                    score_series_all, origin_name_all, used_binary_all = get_score_series(
                        df_g, preferred_binary_col="HasEarlyAlert"
                    )
                    best_thr = find_best_threshold(
                        safe_numeric(df_g.get("HasSepsis", pd.Series([0] * len(df_g))), default=0).astype(int).values,
                        score_series_all.values,
                    )
                    if best_thr is not None:
                        st.markdown(
                            f"""
                            <p class="ep-caption">
                            <strong>Suggested threshold (Youden index):</strong> {best_thr:.2f}  
                            <span style="opacity:0.8;">Balances sensitivity and specificity on this cohort.</span>
                            </p>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.caption(f"Current decision threshold: **{thr:.2f}** on MaxProbInWindow")

                    # GRU "focus window" approximation
                    with st.expander("🧠 Where is the GRU 'paying attention'?", expanded=False):
                        if len(metrics_g["early_times"]) > 0:
                            early_arr = np.array(metrics_g["early_times"])
                            lead_mean = float(np.mean(early_arr))
                            lead_min = float(np.min(early_arr))
                            lead_max = float(np.max(early_arr))

                            st.markdown(
                                f"""
On this cohort, when the GRU **fires an early alert**, it tends to do so:

- **Average lead time:** ~**{lead_mean:.1f} hours** before sepsis onset  
- **Earliest alert:** ~**{lead_max:.1f} hours** before onset  
- **Latest 'early' alert:** ~**{lead_min:.1f} hours** before onset  

This is a **rough, global view** of where the GRU is most active inside the early-warning window,
not a true per-patient explanation.

In the demo tabs, you can visually inspect that many of these alerts coincide with patterns like:

- Rising **heart rate (HR)**  
- Falling **systolic blood pressure (SBP)**  
- **Fever** or rising temperature  

Together, this gives an **interpretability hint** about *when* the GRU reacts, even if we can't show a full deep-learning saliency map.
                                """
                            )
                        else:
                            st.markdown(
                                """
At the current threshold, the GRU did not fire any early alerts on this cohort,
so we cannot summarise a focus window.
                                """
                            )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Early Warning Distribution")
                        if len(metrics_g["early_times"]) > 0:
                            st.pyplot(
                                plot_early_hist(
                                    metrics_g["early_times"], title_suffix="GRU"
                                ),
                                use_container_width=True,
                            )
                        else:
                            st.info("No early detections found.")

                    with col2:
                        if not pd.isna(metrics_g["roc_auc"]):
                            st.markdown("#### ROC Curve (from MaxProbInWindow)")
                            st.pyplot(
                                plot_roc_curve(
                                    metrics_g["fpr"],
                                    metrics_g["tpr"],
                                    metrics_g["roc_auc"],
                                    title="GRU ROC (Patient-Level)",
                                ),
                                use_container_width=True,
                            )
                        else:
                            st.info(
                                "Not enough label variety to compute ROC curve."
                            )

                    st.markdown(
                        "#### Sample of GRU Patient-Level Results (balanced: 10 sepsis / 10 non-sepsis)"
                    )
                    # Reindex sample 1–20 so it doesn't look random
                    sample_g = balanced_sample(df_g, n_pos=10, n_neg=10).reset_index(drop=True)
                    sample_g.index = np.arange(1, len(sample_g) + 1)
                    st.dataframe(sample_g, use_container_width=True)
                    st.caption(
                        "Balanced sample shown to avoid class imbalance hiding model "
                        "behaviour (10 sepsis, 10 non-sepsis)."
                    )
            else:
                st.error(
                    f"GRU results CSV not found: `{MODEL_CONFIG['gru']['csv_6h']}`. "
                    "Run GRU evaluation first."
                )
    

# ============================
# TAB 4 — LIMITATIONS & SAFETY
# ============================

with tab4:
    st.markdown("### ⚠️ Model Limitations & Safety Notes")
    st.markdown(
        """
        <div class="glass-info-card">
            <div class="card-icon">⚠️</div>
            <div class="card-title">Model Limitations & Safety Notes</div>
            <p class="card-desc">
                Why EarlyPulse is a research prototype only, what the models cannot do, and how real clinical
                deployment would require validation, governance, and monitoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
EarlyPulse is a **research prototype only** – it must **not** be used for real clinical triage or treatment.

Some key limitations:

1. **Retrospective data only**  
   - All results are based on the PhysioNet 2019 retrospective ICU dataset.  
   - No prospective or real-time clinical validation has been performed.

2. **Dataset & label noise**  
   - Sepsis labels in large ICU datasets are imperfect and may reflect coding and documentation practices, not ground-truth physiology.  
   - Models may learn **hospital- or workflow-specific shortcuts**.

3. **Population & setting bias**  
   - Training data comes from a specific subset of hospitals and ICU types.  
   - Performance may degrade for other hospitals, countries, or clinical protocols.

4. **Limited feature space**  
   - Only the variables provided in the PhysioNet challenge are used.  
   - Important clinical signals (e.g., imaging, clinician gestalt, unstructured notes) are missing.

5. **Threshold sensitivity**  
   - Changing the decision threshold can dramatically shift sensitivity vs specificity.  
   - Higher sensitivity will often increase false alerts, which may cause alert fatigue.

6. **No causal understanding**  
   - XGBoost and GRU are **associative** models: they detect patterns, not causes.  
   - They can fail on distribution shifts, rare phenotypes, or adversarial combinations of features.

**Bottom line:**  
Use EarlyPulse as a **learning & experimentation tool** for model evaluation, not as a clinical decision support system.
        """
    )
    st.markdown(
    """
    <div style="margin-top:2rem;font-size:0.8rem;color:#4a6f96;text-align:center;">
        EarlyPulse · Research prototype · Mohamed Kamal · 2025–2026
    </div>
    """,
    unsafe_allow_html=True,
)

