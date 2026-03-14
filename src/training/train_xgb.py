"""
src/training/train_xgb.py
──────────────────────────
Train the EarlyPulse XGBoost model.

Usage (from project root):
    python src/training/train_xgb.py

Inputs:
    X_xgboost.npy          — feature matrix (built by build_gru_tensors.py)
    y_xgboost.npy          — labels
    patient_ids.csv        — patient IDs

Outputs:
    earlypulse_xgb_6h24h_model.json
    xgb_feature_names.txt
"""

import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
THRESH        = 0.10
MODEL_OUT     = "earlypulse_xgb_6h24h_model.json"

# Default hyperparams — override with output from tune_xgb_optuna.py
XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "tree_method":      "hist",
    "eval_metric":      "aucpr",
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
    "verbosity":        1,
}

VITALS = [
    "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2",
    "BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST",
    "BUN","Alkalinephos","Calcium","Chloride","Creatinine",
    "Bilirubin_direct","Glucose","Lactate","Magnesium","Phosphate",
    "Potassium","Bilirubin_total","TroponinI","Hct","Hgb",
    "PTT","WBC","Fibrinogen","Platelets",
    "Age","Gender","HospAdmTime",
]
STATS = ["mean","std","min","max","last"]

# ── Load arrays ───────────────────────────────────────────────────────────────
print("Loading data arrays...")
X    = np.load("X_xgboost.npy")
y    = np.load("y_xgboost.npy")
pids = pd.read_csv("patient_ids.csv", header=None)[0].tolist()

split    = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

pos_weight = (1 - y_train.mean()) / max(y_train.mean(), 1e-6)
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Pos rate: {y_train.mean():.4f}")
print(f"scale_pos_weight: {pos_weight:.1f}")

XGB_PARAMS["scale_pos_weight"] = pos_weight

# ── Try loading Optuna best params ────────────────────────────────────────────
optuna_path = "experiments/optuna/best_params.json"
if os.path.exists(optuna_path):
    with open(optuna_path) as f:
        best = json.load(f)
    print(f"✅ Loading Optuna best params (CV AUPRC={best.get('best_cv_auprc','?'):.4f})")
    for k, v in best.items():
        if k != "best_cv_auprc" and k in XGB_PARAMS:
            XGB_PARAMS[k] = v
else:
    print("No Optuna results found — using default hyperparams.")
    print("Run src/training/tune_xgb_optuna.py first for best results.")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
model = XGBClassifier(**XGB_PARAMS)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ── Evaluate on test set ──────────────────────────────────────────────────────
probs = model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, probs)
auprc = average_precision_score(y_test, probs)

print(f"\n{'='*50}")
print(f"Test AUROC : {auroc:.4f}")
print(f"Test AUPRC : {auprc:.4f}")
print(f"{'='*50}")

# ── Save ──────────────────────────────────────────────────────────────────────
model.save_model(MODEL_OUT)
print(f"\nModel saved: {MODEL_OUT}")

feature_names = [f"{v}_{s}" for v in VITALS for s in STATS]
with open("xgb_feature_names.txt", "w") as f:
    f.write("\n".join(feature_names))
print("Feature names saved: xgb_feature_names.txt")
