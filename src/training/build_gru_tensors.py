"""
build_gru_tensors.py — Phase 3.25
Create GRU-ready sequences from PhysioNet 2019 training_setA.

For each patient:
- forward-fill + fillna(0)
- ignore first 6 hours (MIN_HOUR)
- for sepsis patients, only use hours before first SepsisLabel==1
- label each hour t: 1 if sepsis will occur in next 6h, else 0
- build variable-length sequences of features + labels
- compute global mean/std across all time steps
- normalize features and save everything into a single .pt file

Output:
    earlypulse_gru_sequences.pt
"""

import os
import glob

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------- CONFIG ----------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "training_setA")

OUTPUT_PATH = os.path.join(BASE_DIR, "earlypulse_gru_sequences.pt")

MIN_HOUR = 6          # ignore first 6 hours
HORIZON_HOURS = 6     # predict sepsis in next 6 hours

# Same candidates as XGBoost script
FEATURE_CANDIDATES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
    "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
    "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime"
]


def build_patient_sequence(file_path, feature_cols):
    """
    Build one patient's time sequence.

    Returns:
        X: np.ndarray shape (T_i, F)
        y: np.ndarray shape (T_i,)
        patient_id: str
    or
        None if no usable time steps.
    """
    patient_id = os.path.splitext(os.path.basename(file_path))[0]

    df = pd.read_csv(file_path, sep="|").ffill().fillna(0).reset_index(drop=True)

    # required columns
    for col in ["ICULOS", "SepsisLabel"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {file_path}")

    sepsis = df["SepsisLabel"].astype(int)
    has_sepsis = sepsis.max() == 1

    # time index = row index
    hour_idx = df.index.values

    first_sepsis_hour = None  # index (row number)
    if has_sepsis:
        first_sepsis_hour = int(sepsis.idxmax())

    # valid hours (where we create labels)
    if has_sepsis:
        valid_mask = (hour_idx >= MIN_HOUR) & (hour_idx < first_sepsis_hour)
    else:
        valid_mask = (hour_idx >= MIN_HOUR)

    if not valid_mask.any():
        return None

    df_valid = df.loc[valid_mask].copy()
    hour_valid = df_valid.index.values

    # labels per hour: 1 if sepsis in next HORIZON_HOURS, else 0
    if has_sepsis:
        delta = first_sepsis_hour - hour_valid
        labels = ((delta > 0) & (delta <= HORIZON_HOURS)).astype(np.float32)
    else:
        labels = np.zeros_like(hour_valid, dtype=np.float32)

    # features
    feat_cols = [c for c in feature_cols if c in df_valid.columns]
    X = df_valid[feat_cols].astype(np.float32).values

    if X.shape[0] == 0:
        return None

    return X, labels, patient_id


def main():
    psv_files = glob.glob(os.path.join(DATA_DIR, "*.psv"))
    if not psv_files:
        raise FileNotFoundError(f"No .psv files found in {DATA_DIR}")

    # detect available features from first file
    first_df = pd.read_csv(psv_files[0], sep="|")
    feature_cols = [c for c in FEATURE_CANDIDATES if c in first_df.columns]
    print(f"Using {len(feature_cols)} feature columns: {feature_cols}")

    sequences = []       # list of np.ndarray (T_i, F)
    labels = []          # list of np.ndarray (T_i,)
    patient_ids = []     # list of str

    # for normalization
    feat_sum = None      # np.ndarray (F,)
    feat_sumsq = None    # np.ndarray (F,)
    feat_count = 0

    print(f"Building sequences from {len(psv_files)} patients...")

    for file_path in tqdm(psv_files):
        try:
            result = build_patient_sequence(file_path, feature_cols)
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")
            continue

        if result is None:
            continue

        X, y, pid = result
        sequences.append(X)
        labels.append(y)
        patient_ids.append(pid)

        # update normalization stats
        if feat_sum is None:
            feat_sum = X.sum(axis=0)
            feat_sumsq = (X ** 2).sum(axis=0)
        else:
            feat_sum += X.sum(axis=0)
            feat_sumsq += (X ** 2).sum(axis=0)
        feat_count += X.shape[0]

    if not sequences:
        raise RuntimeError("No usable sequences were built. Check configuration.")

    # compute global mean/std
    feat_mean = feat_sum / feat_count
    feat_var = feat_sumsq / feat_count - feat_mean ** 2
    feat_std = np.sqrt(np.clip(feat_var, 1e-6, None))

    print("\nGlobal feature normalization stats:")
    print("  mean (first 5):", feat_mean[:5])
    print("  std  (first 5):", feat_std[:5])

    # normalize sequences
    norm_sequences = []
    for X in sequences:
        Xn = (X - feat_mean) / feat_std
        norm_sequences.append(Xn.astype(np.float32))

    # convert to tensors and save
    tensor_sequences = [torch.from_numpy(X) for X in norm_sequences]
    tensor_labels = [torch.from_numpy(y) for y in labels]

    data_dict = {
        "sequences": tensor_sequences,   # list[Tensor(T_i, F)]
        "labels": tensor_labels,         # list[Tensor(T_i,)]
        "patient_ids": patient_ids,
        "feature_names": feature_cols,
        "mean": feat_mean.astype(np.float32),
        "std": feat_std.astype(np.float32),
        "min_hour": MIN_HOUR,
        "horizon_hours": HORIZON_HOURS,
    }

    torch.save(data_dict, OUTPUT_PATH)

    total_patients = len(patient_ids)
    total_steps = int(sum(seq.shape[0] for seq in tensor_sequences))
    print("\n=== GRU Sequence Dataset Built ===")
    print(f"Patients with usable sequences : {total_patients}")
    print(f"Total time steps (all patients): {total_steps}")
    print(f"Num features per step          : {len(feature_cols)}")
    print(f"Saved to                       : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
