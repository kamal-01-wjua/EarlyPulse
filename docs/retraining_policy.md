# Retraining Policy — EarlyPulse

## When to Retrain

Retrain the model when **any** of the following are true:

| Trigger | Threshold |
|---------|-----------|
| PSI > 0.2 on 3+ features | Major population shift |
| KS p-value < 0.05 on 5+ features | Distribution drift |
| AUROC drops > 0.05 vs baseline | Performance degradation |
| New cohort > 6 months old | Temporal drift |

Run `python src/monitoring/drift_check.py --new_data <new_dir>` to check.

## Retraining Steps

1. `python src/training/tune_xgb_optuna.py` — re-run hyperparameter sweep
2. `python src/training/train_xgb.py` — train with new best params
3. `python src/evaluation/evaluate_XGBoost_6h24h.py` — evaluate and generate SHAP
4. Compare new metrics to `experiments/xgb_eval/metrics.json`
5. If AUROC improves or stays within 0.01: promote to production

## Versioning
Model versions follow `YYYY-MM-DD` date convention:
- `earlypulse_xgb_6h24h_model_2026-03-14.json`
