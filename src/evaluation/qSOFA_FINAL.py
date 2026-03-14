# qSOFA_FINAL.py — Full cohort evaluation (20,317 patients)
import os
import pandas as pd
from tqdm import tqdm

DATA_DIR = 'data/training_setA'
OUT_CSV = "earlypulse_qSOFA_24h_CORRECT.csv"

rows = []

for fname in tqdm(os.listdir(DATA_DIR)):
    if not fname.endswith(".psv"):
        continue

    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path, sep="|").ffill().fillna(0)

    # detect sepsis
    sepsis_times = df.loc[df["SepsisLabel"] == 1, "ICULOS"]
    sepsis_time = sepsis_times.min() if not sepsis_times.empty else None

    # qSOFA-like (Resp > 22 and SBP < 100)
    score = (df["Resp"] > 22).astype(int) + (df["SBP"] < 100).astype(int)
    df["Risk"] = (score >= 2)

    # only after hour 6
    df_after6 = df[df["ICULOS"] >= 6]

    # early alert definition
    if sepsis_time is not None:
        early_start = max(6, sepsis_time - 24)
        early_section = df_after6[
            (df_after6["Risk"]) &
            (df_after6["ICULOS"] >= early_start) &
            (df_after6["ICULOS"] < sepsis_time)
        ]
    else:
        early_section = df_after6[df_after6["Risk"]]

    if not early_section.empty:
        first_alert = early_section["ICULOS"].min()
        early_warning = None
        if sepsis_time is not None:
            early_warning = sepsis_time - first_alert
        has_early = 1
    else:
        first_alert = None
        early_warning = None
        has_early = 0

    rows.append({
        "PatientID": fname.replace(".psv", ""),
        "HasSepsis": 1 if sepsis_time is not None else 0,
        "HasEarlyAlert": has_early,
        "EarlyWarningHours": early_warning
    })

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
