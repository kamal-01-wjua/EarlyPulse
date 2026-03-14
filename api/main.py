"""
api/main.py
───────────
EarlyPulse FastAPI inference endpoint.

Endpoints:
  GET  /health       — liveness check
  GET  /model-info   — model metadata and benchmark metrics
  POST /predict      — score one patient (all 37 raw vitals as JSON)

Usage:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations
import os
import json
import logging
import numpy as np
from xgboost import XGBClassifier
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PatientFeatures, PredictionResponse, HealthResponse, ModelInfoResponse

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "earlypulse_xgb_6h24h_model.json"
METRICS_PATH = "experiments/xgb_eval/metrics.json"
THRESHOLD    = 0.10
MODEL_VER    = "earlypulse-xgb-v2.0"

VITALS = [
    "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2",
    "BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST",
    "BUN","Alkalinephos","Calcium","Chloride","Creatinine",
    "Bilirubin_direct","Glucose","Lactate","Magnesium","Phosphate",
    "Potassium","Bilirubin_total","TroponinI","Hct","Hgb",
    "PTT","WBC","Fibrinogen","Platelets","Age","Gender","HospAdmTime",
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("earlypulse")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EarlyPulse API",
    description="Early sepsis detection — XGBoost inference endpoint",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Load model at startup ─────────────────────────────────────────────────────
_model: XGBClassifier | None = None
_metrics: dict = {}

@app.on_event("startup")
def load_model() -> None:
    global _model, _metrics
    if os.path.exists(MODEL_PATH):
        _model = XGBClassifier()
        _model.load_model(MODEL_PATH)
        logger.info(f"Model loaded: {MODEL_PATH}")
    else:
        logger.warning(f"Model not found: {MODEL_PATH}")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)


def _features_to_array(data: PatientFeatures) -> np.ndarray:
    """Convert PatientFeatures → 37-element array (missing = 0.0)."""
    row = [getattr(data, v) or 0.0 for v in VITALS]
    return np.array(row, dtype=np.float32).reshape(1, -1)


def _risk_level(prob: float) -> str:
    if prob < THRESHOLD:
        return "LOW"
    elif prob < THRESHOLD * 2:
        return "MODERATE"
    return "HIGH"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    return ModelInfoResponse(
        model=MODEL_VER,
        auroc=_metrics.get("auroc", 0.9466),
        auprc=_metrics.get("auprc", 0.7030),
        brier=_metrics.get("brier", 0.031),
        threshold=THRESHOLD,
        n_features=len(VITALS),
        description="XGBoost trained on PhysioNet/CinC 2019 — patient-level test AUROC 0.9466",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(patient: PatientFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = _features_to_array(patient)
    prob  = float(_model.predict_proba(X)[0, 1])
    alert = prob >= THRESHOLD
    level = _risk_level(prob)

    logger.info(f"predict | risk={prob:.4f} | alert={alert} | level={level}")

    return PredictionResponse(
        sepsis_risk=round(prob, 4),
        alert=alert,
        risk_level=level,
        threshold=THRESHOLD,
        model_version=MODEL_VER,
    )
