# Model Card — EarlyPulse XGBoost (v2.0)

## Model Summary
**Task:** Binary classification — will this ICU patient develop sepsis within the next 6 hours?  
**Architecture:** XGBoost (gradient-boosted trees)  
**Features:** 185 aggregate features (mean/std/min/max/last per 37 vitals and labs)  
**Training data:** PhysioNet/CinC 2019 Challenge — 16,254 training patients  

---

## Performance (Held-Out Test Set — 3,987 patients)

| Metric | Value |
|--------|-------|
| AUROC | 0.9466 |
| AUPRC | 0.7030 |
| Sensitivity | 83.6% |
| Specificity | 89.1% |
| Brier Score | 0.031 |
| Threshold | 0.10 |

> Brier score < 0.05 indicates well-calibrated probabilities.

---

## Training Details
- Patient-level 80/20 train/test split (no patient appears in both)
- Class imbalance: scale_pos_weight ~12.9 (sepsis prevalence ~7.2%)
- Hyperparameters: tunable via Optuna (see `src/training/tune_xgb_optuna.py`)
- Seed: 42

---

## Intended Use
Research and portfolio demonstration only. Not validated for clinical use.

---

## Limitations
1. Single dataset (PhysioNet 2019) — no external validation
2. Aggregate features lose temporal ordering
3. Sepsis labels are algorithmically derived, not physician-confirmed
4. No subgroup analysis by age, sex, or ethnicity
5. Missing data handled by zero-fill — may introduce bias
