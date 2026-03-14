# evaluate_gru_6h24h.py — full GRU patient-level evaluation (threshold=0.20)
import torch
import pandas as pd
import os

DATA_PATH = "earlypulse_gru_sequences.pt"
MODEL_PATH = "earlypulse_gru_6h_model.pt"
OUT_CSV = "earlypulse_GRU_6h24h_CORRECT.csv"

print("Loading GRU dataset...")
data = torch.load(DATA_PATH, weights_only=False)
sequences = data["sequences"]
labels = data["labels"]
pids = data["patient_ids"]

print("Loading GRU model...")
model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()

rows = []
THR = 0.20

print("Evaluating...")
with torch.no_grad():
    for pid, seq, label in zip(pids, sequences, labels):
        seq = seq.unsqueeze(0)
        prob = float(model(seq).item())
        has_alert = 1 if prob >= THR else 0

        # Early hours unknown — leave blank
        rows.append({
            "PatientID": pid,
            "HasSepsis": int(label.item()) if hasattr(label, "item") else int(label),
            "HasEarlyAlert": has_alert,
            "MaxProb": prob,
            "EarlyWarningHours": None
        })

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
