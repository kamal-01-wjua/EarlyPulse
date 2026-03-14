"""
src/inference/predict_patient.py
─────────────────────────────────
CLI inference script — score any ICU patient file with the XGBoost model.

Usage:
    python src/inference/predict_patient.py --patient data/training_setA/p000001.psv
    python src/inference/predict_patient.py --patient my_icu_stay.csv --threshold 0.15

Output (JSON to stdout):
    {
      "patient_file": "p000001.psv",
      "sepsis_risk":  0.847,
      "alert":        true,
      "risk_level":   "HIGH",
      "threshold":    0.10,
      "n_hours":      48
    }
"""

from __future__ import annotations
import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ── Config (mirrors config.yaml) ─────────────────────────────────────────────
MODEL_PATH   = "earlypulse_xgb_6h24h_model.json"
FEATURE_PATH = "xgb_feature_names.txt"
DEFAULT_THRESHOLD = 0.10

VITALS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST",
    "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets",
    "Age", "Gender", "HospAdmTime",
]

STAT_FNS = {
    "mean": np.nanmean,
    "std":  lambda x: np.nanstd(x) if len(x) > 1 else 0.0,
    "min":  np.nanmin,
    "max":  np.nanmax,
    "last": lambda x: float(x[-1]) if len(x) > 0 else np.nan,
}


def load_patient(path: str) -> pd.DataFrame:
    """Load a PSV or CSV patient file, normalise column names."""
    sep = "|" if path.endswith(".psv") else ","
    df  = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    return df


def engineer_features(df: pd.DataFrame) -> np.ndarray:
    """Build the 185-feature vector from a patient dataframe."""
    row = []
    for vital in VITALS:
        if vital in df.columns:
            vals = pd.to_numeric(df[vital], errors="coerce").dropna().values
        else:
            vals = np.array([])
        for fname, fn in STAT_FNS.items():
            try:
                row.append(float(fn(vals)) if len(vals) > 0 else np.nan)
            except Exception:
                row.append(np.nan)
    return np.array(row, dtype=np.float32).reshape(1, -1)


def risk_label(prob: float, thresh: float) -> str:
    if prob < thresh:
        return "LOW"
    elif prob < thresh * 2:
        return "MODERATE"
    return "HIGH"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score an ICU patient file for sepsis risk (XGBoost)"
    )
    parser.add_argument("--patient",   required=True, help="Path to .psv or .csv patient file")
    parser.add_argument("--model",     default=MODEL_PATH, help="Path to XGBoost model .json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Decision threshold (default 0.10)")
    parser.add_argument("--json",      action="store_true", help="Output JSON only (for API use)")
    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}", file=sys.stderr)
        sys.exit(1)

    model = XGBClassifier()
    model.load_model(args.model)

    # Load patient
    if not os.path.exists(args.patient):
        print(f"ERROR: Patient file not found: {args.patient}", file=sys.stderr)
        sys.exit(1)

    df     = load_patient(args.patient)
    X      = engineer_features(df)

    # Handle NaN — use 0 (matches training pipeline)
    X_clean = np.nan_to_num(X, nan=0.0)

    prob  = float(model.predict_proba(X_clean)[0, 1])
    alert = prob >= args.threshold
    level = risk_label(prob, args.threshold)

    result = {
        "patient_file": os.path.basename(args.patient),
        "sepsis_risk":  round(prob, 4),
        "alert":        alert,
        "risk_level":   level,
        "threshold":    args.threshold,
        "n_hours":      len(df),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'─'*45}")
        print(f"  Patient      : {result['patient_file']}")
        print(f"  ICU hours    : {result['n_hours']}")
        print(f"  Sepsis risk  : {result['sepsis_risk']:.1%}")
        print(f"  Risk level   : {result['risk_level']}")
        print(f"  Alert fired  : {'⚠  YES' if result['alert'] else '✓  No'}")
        print(f"  Threshold    : {result['threshold']}")
        print(f"{'─'*45}\n")
        print("⚠  Research prototype — not for clinical use.")


if __name__ == "__main__":
    main()
