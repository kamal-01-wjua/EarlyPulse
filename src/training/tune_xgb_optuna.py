"""
src/training/tune_xgb_optuna.py
────────────────────────────────
Optuna hyperparameter sweep for the EarlyPulse XGBoost model.

Usage (from project root):
    python src/training/tune_xgb_optuna.py

Output:
    experiments/optuna/best_params.json   — best hyperparameters
    experiments/optuna/study_results.csv  — all trial results
    experiments/optuna/optuna_history.png — optimisation history plot

Then copy best_params into config/config.yaml under xgboost.params.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Run: pip install optuna")

# ── Config ────────────────────────────────────────────────────────────────────
X_PATH    = "X_xgboost.npy"
Y_PATH    = "y_xgboost.npy"
OUT_DIR   = "experiments/optuna"
N_TRIALS  = 60       # ~30 min on CPU; bump to 100 for overnight
N_FOLDS   = 3        # StratifiedKFold on TRAIN split only
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data (train split only — NEVER touch the test set during tuning) ─────
print("Loading arrays...")
X = np.load(X_PATH)
y = np.load(Y_PATH)

split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
print(f"Train size: {len(X_train)}  |  Positive rate: {y_train.mean():.4f}")

pos_weight = (1 - y_train.mean()) / max(y_train.mean(), 1e-6)
print(f"scale_pos_weight (base): {pos_weight:.1f}")


# ── Objective ─────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    """
    Optimise mean CV AUPRC on the training fold.
    AUPRC is preferred over AUROC for imbalanced data.
    """
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
        "gamma":             trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 5.0),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight",
                                                  pos_weight * 0.5,
                                                  pos_weight * 2.0),
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = []

    for fold_train_idx, fold_val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[fold_train_idx], X_train[fold_val_idx]
        y_tr, y_val = y_train[fold_train_idx], y_train[fold_val_idx]

        model = XGBClassifier(**params, verbosity=0)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            
            verbose=False,
        )
        probs = model.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, probs))

    return float(np.mean(scores))


# ── Run study ─────────────────────────────────────────────────────────────────
print(f"\nStarting Optuna study — {N_TRIALS} trials, {N_FOLDS}-fold CV...")
sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
study   = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ── Results ───────────────────────────────────────────────────────────────────
best = study.best_trial
print(f"\n{'='*55}")
print(f"Best CV AUPRC: {best.value:.4f}")
print(f"Best params:")
for k, v in best.params.items():
    print(f"  {k}: {v}")

# Save best params
best_params_path = os.path.join(OUT_DIR, "best_params.json")
with open(best_params_path, "w") as f:
    json.dump({"best_cv_auprc": best.value, **best.params}, f, indent=2)
print(f"\nSaved: {best_params_path}")

# Save all trials
df_trials = study.trials_dataframe()
df_trials.to_csv(os.path.join(OUT_DIR, "study_results.csv"), index=False)
print(f"Saved: {os.path.join(OUT_DIR, 'study_results.csv')}")

# ── Optimisation history plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor("#0b1828")

# History
vals = [t.value for t in study.trials]
best_so_far = np.maximum.accumulate(vals)
ax = axes[0]
ax.set_facecolor("#0b1828")
ax.plot(vals, color="#3d6090", alpha=0.5, linewidth=1, label="Trial AUPRC")
ax.plot(best_so_far, color="#90bcf8", linewidth=2, label="Best so far")
ax.set_xlabel("Trial", color="#4a6f96"); ax.set_ylabel("CV AUPRC", color="#4a6f96")
ax.set_title("Optimisation History", color="#d8eaff", fontweight="bold")
ax.tick_params(colors="#4a6f96"); ax.legend(fontsize=8, labelcolor="#d8eaff", framealpha=0.15)
for s in ax.spines.values(): s.set_edgecolor("#0f2340")

# Param importance (top 8)
importances = optuna.importance.get_param_importances(study)
top_params  = list(importances.keys())[:8]
top_vals    = [importances[p] for p in top_params]
ax2 = axes[1]
ax2.set_facecolor("#0b1828")
ax2.barh(top_params[::-1], top_vals[::-1], color="#5b9cf6", alpha=0.85, edgecolor="#0b1828")
ax2.set_xlabel("Importance", color="#4a6f96")
ax2.set_title("Hyperparameter Importance", color="#d8eaff", fontweight="bold")
ax2.tick_params(colors="#4a6f96")
for s in ax2.spines.values(): s.set_edgecolor("#0f2340")

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "optuna_history.png")
plt.savefig(plot_path, dpi=130, bbox_inches="tight", facecolor="#0b1828")
print(f"Saved: {plot_path}")

# ── Retrain final model on full train set with best params ────────────────────
print("\nRetraining final model on full train set with best params...")
final_params = {**best.params, "tree_method": "hist", "random_state": RANDOM_SEED, "n_jobs": -1}
final_model = XGBClassifier(**final_params, verbosity=0)
final_model.fit(X_train, y_train)

X_test, y_test = X[split:], y[split:]
probs_test = final_model.predict_proba(X_test)[:, 1]
test_auroc = roc_auc_score(y_test, probs_test)
test_auprc = average_precision_score(y_test, probs_test)

print(f"\n{'='*55}")
print(f"Tuned model — Test AUROC: {test_auroc:.4f} | Test AUPRC: {test_auprc:.4f}")
print(f"\n→ Copy best_params.json values into config/config.yaml to make them permanent.")
print(f"→ Run src/evaluation/evaluate_XGBoost_6h24h.py to regenerate the results CSV.")
