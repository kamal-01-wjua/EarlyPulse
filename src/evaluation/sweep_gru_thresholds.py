"""
sweep_gru_thresholds.py
Run a full threshold sweep for EarlyPulse GRU model.

Evaluates GRU patient-level performance for thresholds:
[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

Outputs:
- sweep_GRU_results.csv
- clean table of sensitivity, specificity, AUROC, and early hours
"""

import numpy as np
import pandas as pd
import torch
import os
import torch.serialization
from sklearn.metrics import roc_auc_score

# -------------------- CONFIG --------------------
BASE = r"C:\Users\sinuo\OneDrive\Desktop\EarlyPulse"
DATA_PATH = os.path.join(BASE, "earlypulse_gru_sequences.pt")
MODEL_PATH = os.path.join(BASE, "earlypulse_gru_6h_model.pt")
OUT_CSV = os.path.join(BASE, "sweep_GRU_results.csv")

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

MIN_HOUR = 6
EARLY_WINDOW = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])


# -------------------- GRU MODEL --------------------
class EarlyPulseGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, bidirectional=False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = torch.nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(out).squeeze(-1)
        return logits


# -------------------- LOAD --------------------
print("Loading GRU dataset...")
data = torch.load(DATA_PATH, weights_only=False)
seqs = data["sequences"]
labels = data["labels"]
pids = data["patient_ids"]

feature_dim = seqs[0].shape[1]

print("Loading model...")
model = EarlyPulseGRU(input_size=feature_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# -------------------- EVAL FUNCTION --------------------
def eval_threshold(thr):
    y_true_auc = []
    y_score_auc = []

    TP = FP = TN = FN = 0
    early_hours_list = []

    for seq, lbl in zip(seqs, labels):
        T = seq.shape[0]
        seq = seq.unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([T]).to(DEVICE)

        with torch.no_grad():
            logits = model(seq, lengths)
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # (T,)

        lbl_np = lbl.numpy()
        sepsis_pos = np.where(lbl_np == 1)[0]
        has_sepsis = len(sepsis_pos) > 0

        time_idx = np.arange(T)

        if has_sepsis:
            onset = sepsis_pos[0]
            early_start = max(MIN_HOUR, onset - EARLY_WINDOW)
            eval_mask = (time_idx >= early_start) & (time_idx < onset)
        else:
            eval_mask = (time_idx >= MIN_HOUR)

        eval_probs = probs[eval_mask]
        eval_times = time_idx[eval_mask]

        max_prob = eval_probs.max() if len(eval_probs) else 0.0

        y_true_auc.append(1 if has_sepsis else 0)
        y_score_auc.append(max_prob)

        has_alert = False
        early_hours = None

        if len(eval_probs) > 0:
            alert_mask = (eval_probs >= thr)
            if alert_mask.any():
                has_alert = True
                if has_sepsis:
                    alert_idx = eval_times[alert_mask][0]
                    early_hours = int(onset - alert_idx)
                    early_hours_list.append(early_hours)

        if has_sepsis:
            if has_alert: TP += 1
            else: FN += 1
        else:
            if has_alert: FP += 1
            else: TN += 1

    auroc = roc_auc_score(y_true_auc, y_score_auc)
    sens = TP / (TP + FN) if (TP + FN) else 0
    spec = TN / (TN + FP) if (TN + FP) else 0

    median_early = np.median(early_hours_list) if early_hours_list else 0
    mean_early = np.mean(early_hours_list) if early_hours_list else 0

    return {
        "Threshold": thr,
        "AUROC": auroc,
        "Sensitivity": sens,
        "Specificity": spec,
        "EarlyAlerts": len(early_hours_list),
        "MedianEarlyH": median_early,
        "MeanEarlyH": mean_early
    }


# -------------------- RUN SWEEP --------------------
print("Running threshold sweep...")

results = []
for thr in THRESHOLDS:
    print(f"Threshold {thr}...")
    res = eval_threshold(thr)
    results.append(res)

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)

print("\nSaved sweep results to:", OUT_CSV)
print(df)
