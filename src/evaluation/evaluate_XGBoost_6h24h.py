"""
src/evaluation/evaluate_XGBoost_6h24h.py
─────────────────────────────────────────
Upgraded XGBoost evaluation:
  - Patient-level AUROC + AUPRC + Brier score
  - Calibration curve + reliability diagram
  - SHAP feature importance (top-20 bar chart)
  - Saves results CSV + all plots to experiments/

Usage (from project root):
    python src/evaluation/evaluate_XGBoost_6h24h.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠  shap not installed — skipping SHAP plot. Run: pip install shap")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "earlypulse_xgb_6h24h_model.json"
X_PATH       = "X_xgboost.npy"
Y_PATH       = "y_xgboost.npy"
PID_PATH     = "patient_ids.csv"
FEAT_PATH    = "xgb_feature_names.txt"   # one feature name per line
OUT_CSV      = "data/results/earlypulse_XGBoost_6h24h_CORRECT.csv"
OUT_DIR      = "experiments/xgb_eval"
THRESH       = 0.10

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("data/results", exist_ok=True)

BG       = "#0b1828"
BLUE_LT  = "#90bcf8"
BLUE     = "#5b9cf6"
BORDER   = "#0f2340"
WHITE    = "#d8eaff"
GREY     = "#4a6f96"
RED      = "#e05050"


def dark_axes(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.title.set_color(WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)


# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading model and arrays...")
model = XGBClassifier()
model.load_model(MODEL_PATH)

X   = np.load(X_PATH)
y   = np.load(Y_PATH)
pids = pd.read_csv(PID_PATH, header=None)[0].tolist()

split   = int(len(X) * 0.8)
X_test  = X[split:]
y_test  = y[split:]
p_test  = pids[split:]

# Feature names (optional)
if os.path.exists(FEAT_PATH):
    with open(FEAT_PATH) as f:
        feature_names = [l.strip() for l in f if l.strip()]
else:
    feature_names = [f"f{i}" for i in range(X_test.shape[1])]

print(f"Test patients: {len(X_test)}")

# ── Predictions ───────────────────────────────────────────────────────────────
print("Computing predictions...")
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= THRESH).astype(int)

# ── Core metrics ──────────────────────────────────────────────────────────────
auroc = roc_auc_score(y_test, probs)
auprc = average_precision_score(y_test, probs)
brier = brier_score_loss(y_test, probs)

tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
sens = tp / max(tp + fn, 1)
spec = tn / max(tn + fp, 1)

print(f"\n{'='*50}")
print(f"Test patients  : {len(X_test)}")
print(f"AUROC          : {auroc:.4f}")
print(f"AUPRC          : {auprc:.4f}")
print(f"Brier score    : {brier:.4f}  (lower = better; perfect = 0)")
print(f"Sensitivity    : {sens*100:.1f}%  (threshold = {THRESH})")
print(f"Specificity    : {spec*100:.1f}%")
print(f"TP/FP/TN/FN    : {tp}/{fp}/{tn}/{fn}")
print(f"{'='*50}\n")

# Save metrics JSON
metrics = dict(auroc=auroc, auprc=auprc, brier=brier,
               sensitivity=sens, specificity=spec,
               tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
               threshold=THRESH, n_test=len(X_test))
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ── Save result CSV ───────────────────────────────────────────────────────────
rows = []
for pid, label, prob, pred in zip(p_test, y_test, probs, preds):
    rows.append({
        "PatientID":        str(pid).replace(".psv", ""),
        "HasSepsis":        int(label),
        "HasEarlyAlert":    int(pred),
        "MaxProb":          float(prob),
        "EarlyWarningHours": None,
    })
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"Saved results CSV: {OUT_CSV}")

# ── ROC curve ──────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, probs)
fig, ax = plt.subplots(figsize=(5, 4.5))
dark_axes(ax, fig)
ax.plot(fpr, tpr, color=BLUE_LT, linewidth=2.2, label=f"AUROC = {auroc:.4f}")
ax.plot([0, 1], [0, 1], color=BORDER, linewidth=1, linestyle="--", label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — XGBoost")
ax.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=130, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: roc_curve.png")

# ── Calibration curve ─────────────────────────────────────────────────────────
frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
fig, ax = plt.subplots(figsize=(5, 4.5))
dark_axes(ax, fig)
ax.plot([0, 1], [0, 1], color=BORDER, linewidth=1, linestyle="--", label="Perfect")
ax.plot(mean_pred, frac_pos, color=BLUE_LT, linewidth=2.2, marker="o", markersize=5,
        label=f"XGBoost (Brier = {brier:.4f})")
ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.08, color=BLUE)
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve — XGBoost")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "calibration_curve.png"), dpi=130, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: calibration_curve.png")

# ── SHAP feature importance ───────────────────────────────────────────────────
if HAS_SHAP:
    print("\nComputing SHAP values (this may take 1–2 min)...")
    # Use a random 500-sample background for speed
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_test), size=min(500, len(X_test)), replace=False)
    explainer = shap.TreeExplainer(model, feature_names=feature_names)
    shap_values = explainer.shap_values(X_test[bg_idx])

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_n = 20
    idx = np.argsort(mean_abs)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals  = mean_abs[idx]

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.38)))
    dark_axes(ax, fig)
    ax.barh(range(len(names)), vals, color=BLUE, alpha=0.85, edgecolor=BG, height=0.7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance — XGBoost (SHAP, top {top_n})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_importance.png"), dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Saved: shap_importance.png")

    # Also save SHAP values for dashboard use
    np.save(os.path.join(OUT_DIR, "shap_values.npy"), shap_values)
    with open(os.path.join(OUT_DIR, "shap_feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))
    print("Saved: shap_values.npy + shap_feature_names.txt")

print("\n✅ XGBoost evaluation complete.")
print(f"   All outputs in: {OUT_DIR}/")
