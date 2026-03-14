"""
src/monitoring/drift_check.py
──────────────────────────────
Feature drift detection using KS-test and Population Stability Index (PSI).

Usage (from project root):
    python src/monitoring/drift_check.py --new_data data/new_cohort/
    python src/monitoring/drift_check.py --new_data data/training_setA  # self-test

Outputs:
    experiments/drift_report_YYYY-MM-DD/drift_report.json
    experiments/drift_report_YYYY-MM-DD/drift_summary.png
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import date
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REFERENCE_DATA = "data/training_setA"
PSI_THRESHOLD  = 0.20   # >0.2 = major shift
KS_ALPHA       = 0.05   # p < 0.05 = significant drift

VITALS = [
    "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2",
    "BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST",
    "BUN","Alkalinephos","Calcium","Chloride","Creatinine",
    "Bilirubin_direct","Glucose","Lactate","Magnesium","Phosphate",
    "Potassium","Bilirubin_total","TroponinI","Hct","Hgb",
    "PTT","WBC","Fibrinogen","Platelets",
    "Age","Gender","HospAdmTime",
]


def load_cohort(data_dir: str) -> pd.DataFrame:
    """Load all PSV files from a directory into a single dataframe."""
    frames = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".psv"):
            try:
                df = pd.read_csv(os.path.join(data_dir, fname), sep="|")
                frames.append(df)
            except Exception:
                continue
    if not frames:
        raise ValueError(f"No PSV files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def compute_psi(ref: np.ndarray, new: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    ref_clean = ref[~np.isnan(ref)]
    new_clean = new[~np.isnan(new)]
    if len(ref_clean) < 10 or len(new_clean) < 10:
        return 0.0
    bins = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
    ref_counts = np.histogram(ref_clean, bins=bins)[0]
    new_counts = np.histogram(new_clean, bins=bins)[0]
    ref_pct = (ref_counts + 1e-6) / (len(ref_clean) + 1e-6 * len(ref_counts))
    new_pct = (new_counts + 1e-6) / (len(new_clean) + 1e-6 * len(new_counts))
    psi = np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct))
    return float(psi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_data", required=True, help="Directory of new PSV files")
    parser.add_argument("--reference", default=REFERENCE_DATA)
    args = parser.parse_args()

    out_dir = f"experiments/drift_report_{date.today().isoformat()}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading reference data from: {args.reference}")
    ref_df = load_cohort(args.reference)

    print(f"Loading new data from: {args.new_data}")
    new_df = load_cohort(args.new_data)

    print(f"Reference: {len(ref_df):,} rows | New: {len(new_df):,} rows")
    print(f"\nRunning drift checks on {len(VITALS)} features...\n")

    results = []
    drifted = []

    for feature in VITALS:
        if feature not in ref_df.columns or feature not in new_df.columns:
            results.append({"feature": feature, "status": "MISSING", "ks_pval": None, "psi": None})
            continue

        ref_vals = pd.to_numeric(ref_df[feature], errors="coerce").dropna().values
        new_vals = pd.to_numeric(new_df[feature], errors="coerce").dropna().values

        if len(ref_vals) < 5 or len(new_vals) < 5:
            results.append({"feature": feature, "status": "INSUFFICIENT", "ks_pval": None, "psi": None})
            continue

        ks_stat, ks_pval = stats.ks_2samp(ref_vals, new_vals)
        psi = compute_psi(ref_vals, new_vals)

        drifted_flag = (ks_pval < KS_ALPHA) or (psi > PSI_THRESHOLD)
        status = "DRIFT" if drifted_flag else "STABLE"

        results.append({"feature": feature, "status": status,
                        "ks_stat": round(ks_stat, 4),
                        "ks_pval": round(ks_pval, 4),
                        "psi": round(psi, 4)})

        if drifted_flag:
            drifted.append(feature)
            print(f"  ⚠  DRIFT  {feature:<25} KS p={ks_pval:.4f}  PSI={psi:.4f}")
        else:
            print(f"  ✓  STABLE {feature:<25} KS p={ks_pval:.4f}  PSI={psi:.4f}")

    # Summary
    n_stable = sum(1 for r in results if r["status"] == "STABLE")
    n_drift  = len(drifted)
    retrain  = n_drift > 3 or any(
        r["psi"] is not None and r["psi"] > 0.4 for r in results
    )

    report = {
        "date": date.today().isoformat(),
        "reference": args.reference,
        "new_data": args.new_data,
        "n_features_checked": len(VITALS),
        "n_stable": n_stable,
        "n_drifted": n_drift,
        "retraining_recommended": retrain,
        "drifted_features": drifted,
        "features": results,
    }

    json_path = os.path.join(out_dir, "drift_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Stable features  : {n_stable}")
    print(f"  Drifted features : {n_drift}")
    print(f"  Retraining needed: {'YES ⚠' if retrain else 'No ✓'}")
    print(f"{'='*55}")
    print(f"\nReport saved: {json_path}")

    # Plot
    psvals  = [r["psi"] or 0 for r in results if r["status"] in ("STABLE","DRIFT")]
    fnames  = [r["feature"] for r in results if r["status"] in ("STABLE","DRIFT")]
    colors  = ["#e05050" if r["status"]=="DRIFT" else "#5b9cf6"
               for r in results if r["status"] in ("STABLE","DRIFT")]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0b1828")
    ax.set_facecolor("#0b1828")
    bars = ax.bar(range(len(fnames)), psvals, color=colors, alpha=0.85, edgecolor="#0b1828")
    ax.axhline(PSI_THRESHOLD, color="#e0a050", linewidth=1.5,
               linestyle="--", label=f"PSI threshold ({PSI_THRESHOLD})")
    ax.set_xticks(range(len(fnames)))
    ax.set_xticklabels(fnames, rotation=45, ha="right", fontsize=7, color="#4a6f96")
    ax.set_ylabel("PSI", color="#4a6f96")
    ax.set_title("Feature Drift — Population Stability Index", color="#d8eaff", fontweight="bold")
    ax.tick_params(colors="#4a6f96")
    ax.legend(fontsize=8, framealpha=0.15, labelcolor="#d8eaff")
    for sp in ax.spines.values(): sp.set_edgecolor("#0f2340")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "drift_summary.png")
    plt.savefig(plot_path, dpi=130, bbox_inches="tight", facecolor="#0b1828")
    plt.close()
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
