"""
src/evaluation/metrics.py
─────────────────────────
All metric computation for EarlyPulse models.
Extracted from app.py so logic is testable in isolation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


# ── Generic helpers ───────────────────────────────────────────────────────────

def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return AUROC or None if labels are degenerate."""
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc(fpr, tpr))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute TP/FP/TN/FN/sensitivity/specificity from binary predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                sens=sens, spec=spec)


# ── XGBoost ───────────────────────────────────────────────────────────────────

def compute_xgb_metrics(df: pd.DataFrame, threshold: float = 0.10) -> dict | None:
    """
    Compute patient-level XGBoost metrics from the results CSV.

    Expected columns: HasSepsis, MaxProb, [EarlyWarningHours]
    """
    required = {"HasSepsis", "MaxProb"}
    if not required.issubset(df.columns):
        return None

    y_true  = df["HasSepsis"].fillna(0).astype(int).values
    y_score = pd.to_numeric(df["MaxProb"], errors="coerce").fillna(0).values
    y_pred  = (y_score >= threshold).astype(int)

    roc_auc = safe_auroc(y_true, y_score)
    auprc   = safe_auprc(y_true, y_score)
    brier   = float(brier_score_loss(y_true, y_score))

    bin_m = binary_metrics(y_true, y_pred)

    early_times: list[float] = []
    if "EarlyWarningHours" in df.columns:
        early_times = (
            df.loc[
                (df["HasEarlyAlert"] == 1) & (df["HasSepsis"] == 1),
                "EarlyWarningHours",
            ]
            .dropna()
            .tolist()
        )

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    return dict(
        roc_auc=roc_auc,
        auprc=auprc,
        brier=brier,
        early_times=early_times,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        y_true=y_true,
        y_score=y_score,
        **bin_m,
    )


# ── GRU ──────────────────────────────────────────────────────────────────────

def compute_gru_metrics(df: pd.DataFrame, threshold: float = 0.20) -> dict | None:
    """Compute patient-level GRU metrics from the results CSV."""
    score_col = None
    for col in ("MaxProb", "MaxScore", "Score"):
        if col in df.columns:
            score_col = col
            break

    if score_col is None or "HasSepsis" not in df.columns:
        return None

    y_true  = df["HasSepsis"].fillna(0).astype(int).values
    y_score = pd.to_numeric(df[score_col], errors="coerce").fillna(0).values
    y_pred  = (y_score >= threshold).astype(int)

    roc_auc = safe_auroc(y_true, y_score)
    auprc   = safe_auprc(y_true, y_score)
    brier   = float(brier_score_loss(y_true, y_score))

    bin_m = binary_metrics(y_true, y_pred)

    early_times: list[float] = []
    if "EarlyWarningHours" in df.columns:
        early_times = (
            df.loc[
                (df["HasEarlyAlert"] == 1) & (df["HasSepsis"] == 1),
                "EarlyWarningHours",
            ]
            .dropna()
            .tolist()
        )

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    return dict(
        roc_auc=roc_auc,
        auprc=auprc,
        brier=brier,
        early_times=early_times,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        y_true=y_true,
        y_score=y_score,
        **bin_m,
    )


# ── qSOFA ────────────────────────────────────────────────────────────────────

def compute_qsofa_metrics(df: pd.DataFrame) -> dict | None:
    """Compute qSOFA metrics — binary alert, no probability score."""
    if "HasSepsis" not in df.columns or "HasEarlyAlert" not in df.columns:
        return None

    y_true = df["HasSepsis"].fillna(0).astype(int).values
    y_pred = df["HasEarlyAlert"].fillna(0).astype(int).values

    # Use alert as a binary score for AUROC
    y_score = y_pred.astype(float)

    roc_auc = safe_auroc(y_true, y_score)
    auprc   = safe_auprc(y_true, y_score)
    brier   = float(brier_score_loss(y_true, y_score))

    bin_m = binary_metrics(y_true, y_pred)

    early_times: list[float] = []
    if "EarlyWarningHours" in df.columns:
        early_times = (
            df.loc[
                (y_pred == 1) & (y_true == 1),
                "EarlyWarningHours",
            ]
            .dropna()
            .tolist()
        )

    fpr, tpr, _ = roc_curve(y_true, y_score)

    return dict(
        roc_auc=roc_auc,
        auprc=auprc,
        brier=brier,
        early_times=early_times,
        fpr=fpr,
        tpr=tpr,
        y_true=y_true,
        y_score=y_score,
        **bin_m,
    )


# ── Calibration ───────────────────────────────────────────────────────────────

def compute_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (fraction_of_positives, mean_predicted_value) for calibration curve.
    Wraps sklearn with uniform strategy.
    """
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="uniform"
    )
    return fraction_pos, mean_pred
