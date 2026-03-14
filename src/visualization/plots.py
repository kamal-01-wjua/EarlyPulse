"""
src/visualization/plots.py
──────────────────────────
All matplotlib/plotly figure builders for EarlyPulse.
Extracted from app.py so the dashboard stays thin.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Style constants ───────────────────────────────────────────────────────────
BG        = "#060e1a"
CARD_BG   = "#0b1828"
BORDER    = "#0f2340"
BLUE      = "#5b9cf6"
BLUE_LT   = "#90bcf8"
WHITE     = "#e4eeff"
GREY      = "#4a6f96"
RED_ALERT = "#e05050"
GREEN_OK  = "#3aaa70"


def _apply_dark_style(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply EarlyPulse dark theme to a matplotlib axes."""
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


# ── Patient trajectory ────────────────────────────────────────────────────────

def plot_patient_trajectory(
    df: pd.DataFrame,
    patient_name: str = "Patient",
    vitals: list[str] | None = None,
) -> plt.Figure:
    """
    Multi-panel vital signs plot with sepsis onset and early-warning window.
    Works for both demo and uploaded patients.
    """
    if vitals is None:
        vitals = [v for v in ["HR", "SBP", "Temp", "Resp", "O2Sat"] if v in df.columns]
    if not vitals:
        vitals = [c for c in df.columns if c not in ("ICULOS", "SepsisLabel", "PatientID")][:3]

    time_col = "ICULOS" if "ICULOS" in df.columns else df.columns[0]
    time = df[time_col].values

    sepsis_time: float | None = None
    if "SepsisLabel" in df.columns:
        onset_rows = df[df["SepsisLabel"] == 1]
        if not onset_rows.empty:
            sepsis_time = float(onset_rows[time_col].iloc[0])

    n = len(vitals)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(CARD_BG)
    fig.suptitle(patient_name, color=WHITE, fontsize=13, fontweight="bold", y=1.01)

    for ax, vital in zip(axes, vitals):
        _apply_dark_style(ax, fig)
        vals = pd.to_numeric(df[vital], errors="coerce")
        ax.plot(time, vals, color=BLUE, linewidth=1.8, alpha=0.9)
        ax.set_ylabel(vital, fontsize=9)

        if sepsis_time is not None:
            # Early-warning window shading (24 h before onset)
            window_start = max(time[0], sepsis_time - 24)
            ax.axvspan(window_start, sepsis_time,
                       alpha=0.12, color=RED_ALERT, label="Early-warning window")
            ax.axvline(sepsis_time, color=RED_ALERT, linewidth=1.5,
                       linestyle="--", label="Sepsis onset")

    axes[-1].set_xlabel("ICU hours", fontsize=9)

    if sepsis_time is not None:
        handles = [
            mpatches.Patch(color=RED_ALERT, alpha=0.4, label="Early-warning window"),
            plt.Line2D([0], [0], color=RED_ALERT, linewidth=1.5,
                       linestyle="--", label="Sepsis onset"),
        ]
        axes[0].legend(handles=handles, fontsize=7,
                       framealpha=0.15, labelcolor=WHITE)

    fig.tight_layout()
    return fig


# ── ROC curve ─────────────────────────────────────────────────────────────────

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    model_name: str = "Model",
    threshold_point: tuple[float, float] | None = None,
) -> plt.Figure:
    """Single-model ROC curve with optional operating-point marker."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    _apply_dark_style(ax, fig)

    ax.plot(fpr, tpr, color=BLUE_LT, linewidth=2.2, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], color=BORDER, linewidth=1, linestyle="--", label="Random")

    if threshold_point is not None:
        ax.scatter(*threshold_point, color=RED_ALERT, s=80, zorder=5,
                   label="Operating point")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE)
    fig.tight_layout()
    return fig


# ── Calibration curve ─────────────────────────────────────────────────────────

def plot_calibration_curve(
    fraction_pos: np.ndarray,
    mean_pred: np.ndarray,
    brier: float,
    model_name: str = "Model",
) -> plt.Figure:
    """
    Reliability diagram (calibration curve).
    A perfectly calibrated model follows the diagonal.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    _apply_dark_style(ax, fig)

    ax.plot([0, 1], [0, 1], color=BORDER, linewidth=1,
            linestyle="--", label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, color=BLUE_LT,
            linewidth=2.2, marker="o", markersize=5,
            label=f"Brier = {brier:.4f}")

    ax.fill_between(mean_pred, fraction_pos, mean_pred,
                    alpha=0.08, color=BLUE)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE)
    fig.tight_layout()
    return fig


# ── Early-warning histogram ───────────────────────────────────────────────────

def plot_early_warning_hist(
    early_times: list[float],
    model_name: str = "Model",
) -> plt.Figure | None:
    """Histogram of early-warning lead times (hours before sepsis onset)."""
    if not early_times:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _apply_dark_style(ax, fig)

    ax.hist(early_times, bins=20, color=BLUE, alpha=0.85, edgecolor=CARD_BG)
    ax.axvline(np.median(early_times), color=RED_ALERT, linewidth=1.5,
               linestyle="--", label=f"Median = {np.median(early_times):.1f} h")
    ax.set_xlabel("Hours before sepsis onset")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Early-Warning Lead Time — {model_name}")
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE)
    fig.tight_layout()
    return fig


# ── SHAP summary (static image fallback) ─────────────────────────────────────

def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    model_name: str = "XGBoost",
) -> plt.Figure:
    """Horizontal bar chart of mean absolute SHAP values."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals  = mean_abs[idx]

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
    _apply_dark_style(ax, fig)

    bars = ax.barh(range(len(names)), vals, color=BLUE, alpha=0.85,
                   edgecolor=CARD_BG, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance — {model_name} (SHAP)")
    fig.tight_layout()
    return fig
