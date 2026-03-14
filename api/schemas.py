"""
api/schemas.py
──────────────
Pydantic request/response models for the EarlyPulse FastAPI endpoint.
All 37 raw vitals are optional (missing = 0.0 as in training pipeline).
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """Raw ICU vitals — all optional, same as PhysioNet 2019 schema."""
    HR:               Optional[float] = Field(None, description="Heart rate (bpm)")
    O2Sat:            Optional[float] = Field(None, description="Oxygen saturation (%)")
    Temp:             Optional[float] = Field(None, description="Temperature (°C)")
    SBP:              Optional[float] = Field(None, description="Systolic BP (mmHg)")
    MAP:              Optional[float] = Field(None, description="Mean arterial pressure")
    DBP:              Optional[float] = Field(None, description="Diastolic BP (mmHg)")
    Resp:             Optional[float] = Field(None, description="Respiratory rate")
    EtCO2:            Optional[float] = None
    BaseExcess:       Optional[float] = None
    HCO3:             Optional[float] = None
    FiO2:             Optional[float] = None
    pH:               Optional[float] = None
    PaCO2:            Optional[float] = None
    SaO2:             Optional[float] = None
    AST:              Optional[float] = None
    BUN:              Optional[float] = None
    Alkalinephos:     Optional[float] = None
    Calcium:          Optional[float] = None
    Chloride:         Optional[float] = None
    Creatinine:       Optional[float] = None
    Bilirubin_direct: Optional[float] = None
    Glucose:          Optional[float] = None
    Lactate:          Optional[float] = None
    Magnesium:        Optional[float] = None
    Phosphate:        Optional[float] = None
    Potassium:        Optional[float] = None
    Bilirubin_total:  Optional[float] = None
    TroponinI:        Optional[float] = None
    Hct:              Optional[float] = None
    Hgb:              Optional[float] = None
    PTT:              Optional[float] = None
    WBC:              Optional[float] = None
    Fibrinogen:       Optional[float] = None
    Platelets:        Optional[float] = None
    Age:              Optional[float] = None
    Gender:           Optional[float] = None
    HospAdmTime:      Optional[float] = None

    model_config = {"json_schema_extra": {"example": {
        "HR": 102, "O2Sat": 94, "Temp": 38.4, "SBP": 88,
        "Resp": 24, "Age": 67, "Gender": 1
    }}}


class PredictionResponse(BaseModel):
    sepsis_risk:   float = Field(..., description="Probability [0–1] of sepsis within 6h")
    alert:         bool  = Field(..., description="True if risk >= threshold")
    risk_level:    str   = Field(..., description="LOW / MODERATE / HIGH")
    threshold:     float = Field(..., description="Decision threshold used")
    model_version: str   = Field(..., description="Model identifier")


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model:       str
    auroc:       float
    auprc:       float
    brier:       float
    threshold:   float
    n_features:  int
    description: str
