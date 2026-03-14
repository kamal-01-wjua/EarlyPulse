"""
src/io/loaders.py
─────────────────
All data-loading utilities for EarlyPulse.
Extracted from app.py so the dashboard stays thin.
"""

from __future__ import annotations
import io
import os
import chardet
import pandas as pd
import streamlit as st

# ── Column alias map ──────────────────────────────────────────────────────────
COLUMN_ALIASES: dict[str, str] = {
    # Heart-rate variants
    "heart_rate": "HR", "heartrate": "HR", "pulse": "HR",
    # SBP
    "systolic_bp": "SBP", "systolicbp": "SBP", "sbp": "SBP",
    "systolic blood pressure": "SBP", "sys_bp": "SBP",
    # Temperature
    "temperature": "Temp", "temp_c": "Temp", "temp": "Temp", "body_temp": "Temp",
    # Resp rate
    "resp_rate": "Resp", "rr": "Resp", "respiratory_rate": "Resp",
    # O2 sat
    "spo2": "O2Sat", "oxygen_sat": "O2Sat", "o2sat": "O2Sat",
    # MAP
    "mean_arterial_pressure": "MAP", "map": "MAP",
    # Sepsis label
    "sepsislabel": "SepsisLabel", "sepsis_label": "SepsisLabel", "sepsis": "SepsisLabel",
    "label": "SepsisLabel",
    # ICULOS
    "icu_los": "ICULOS", "iculos": "ICULOS", "icu_time": "ICULOS",
}

REQUIRED_VITALS = ["HR", "SBP", "Temp", "Resp", "O2Sat"]


@st.cache_data(show_spinner=False)
def load_results_csv(path: str, label: str = "data") -> pd.DataFrame | None:
    """Load a pre-computed result CSV with basic validation."""
    if not os.path.exists(path):
        st.warning(f"Result file not found: `{path}`")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            st.warning(f"{label} CSV is empty.")
            return None
        return df
    except Exception as exc:
        st.error(f"Failed to load {label}: {exc}")
        return None


def load_any_patient_file(uploaded_file) -> pd.DataFrame | None:
    """
    Accept any CSV/PSV ICU-like file from a Streamlit uploader.

    - Detects delimiter (| or ,)
    - Normalises column names
    - Remaps common aliases to the standard schema
    - Returns None on failure (shows st.error)
    """
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    # Encoding detection
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    # Delimiter detection
    first_line = text.split("\n")[0]
    delimiter = "|" if first_line.count("|") > first_line.count(",") else ","

    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except Exception as exc:
        st.error(f"Could not parse file: {exc}")
        return None

    if df.empty:
        st.error("Uploaded file appears to be empty.")
        return None

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        col: COLUMN_ALIASES[col.lower()]
        for col in df.columns
        if col.lower() in COLUMN_ALIASES
    }
    df.rename(columns=rename_map, inplace=True)

    # Add ICULOS if missing
    if "ICULOS" not in df.columns:
        df["ICULOS"] = range(1, len(df) + 1)

    # Coerce numeric columns
    for col in df.columns:
        if col not in ("PatientID",):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_demo_patient(data_dir: str, filename: str) -> pd.DataFrame | None:
    """Load a demo patient PSV file from the data directory."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep="|")
        if "ICULOS" not in df.columns:
            df["ICULOS"] = range(1, len(df) + 1)
        return df
    except Exception:
        return None
