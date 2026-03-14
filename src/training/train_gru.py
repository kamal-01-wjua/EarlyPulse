"""
train_gru.py — Phase 3.3
Train a GRU model on earlypulse_gru_sequences.pt to predict 6h-ahead sepsis risk.

- Loads sequences from earlypulse_gru_sequences.pt
- Splits patients into train/val/test (80/10/10)
- Trains GRU with BCEWithLogitsLoss
- Reports step-level AUROC on test set
"""

import os
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# ---------------- CONFIG ----------------
BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(".", "earlypulse_gru_sequences.pt")
MODEL_PATH = os.path.join(".", "earlypulse_gru_6h_model.pt")

RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 1
BIDIRECTIONAL = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- UTILITIES ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ---------------- DATASET ----------------
class EarlyPulseGRUDataset(Dataset):
    """
    Wraps the sequences and labels from earlypulse_gru_sequences.pt
    Each item:
        - seq: Tensor(T_i, F)
        - y: Tensor(T_i,)
    """

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    """
    Collate variable-length sequences into padded batch:
        - inputs: (batch, max_T, F)
        - targets: (batch, max_T)
        - lengths: (batch,)
    """
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)

    # Pad sequences
    max_len = lengths.max().item()
    feat_dim = seqs[0].shape[1]

    padded_seqs = torch.zeros(len(seqs), max_len, feat_dim, dtype=torch.float32)
    padded_labels = torch.zeros(len(seqs), max_len, dtype=torch.float32)

    for i, (s, y) in enumerate(zip(seqs, labels)):
        T = s.shape[0]
        padded_seqs[i, :T, :] = s
        padded_labels[i, :T] = y

    return padded_seqs, padded_labels, lengths


# ---------------- MODEL ----------------
class EarlyPulseGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x, lengths):
        """
        x: (batch, seq_len, input_size)
        lengths: (batch,)
        """
        # Pack padded batch for GRU
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        logits = self.fc(out).squeeze(-1)  # (batch, seq_len)
        return logits


# ---------------- MAIN TRAINING LOGIC ----------------
def main():
    print("Loading GRU dataset from:", DATA_PATH)
    data = torch.load(DATA_PATH, weights_only=False)

    sequences = data["sequences"]          # list[Tensor(T_i, F)]
    labels = data["labels"]                # list[Tensor(T_i,)]
    feature_names = data["feature_names"]
    print(f"Loaded {len(sequences)} sequences with {len(feature_names)} features.")

    # Create patient index list and split train/val/test
    num_patients = len(sequences)
    indices = list(range(num_patients))
    random.shuffle(indices)

    n_train = int(0.8 * num_patients)
    n_val = int(0.1 * num_patients)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    print(f"Train patients: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_seqs = [sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_seqs = [sequences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    test_seqs = [sequences[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    train_dataset = EarlyPulseGRUDataset(train_seqs, train_labels)
    val_dataset = EarlyPulseGRUDataset(val_seqs, val_labels)
    test_dataset = EarlyPulseGRUDataset(test_seqs, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    input_size = train_seqs[0].shape[1]
    model = EarlyPulseGRU(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
    ).to(DEVICE)

    # Compute class imbalance for POS weight
    all_train_labels = torch.cat(train_labels)
    pos_fraction = all_train_labels.mean().item()
    pos_weight_value = (1.0 - pos_fraction) / max(pos_fraction, 1e-6)
    print(f"Positive label fraction (train): {pos_fraction:.6f}, pos_weight: {pos_weight_value:.1f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---------- TRAINING LOOP ----------
    best_val_auc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0

        for batch in train_loader:
            inputs, targets, lengths = batch
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(inputs, lengths)  # (batch, seq_len)

            # Mask out padded positions
            max_len = targets.size(1)
            mask = torch.arange(max_len, device=DEVICE)[None, :] < lengths[:, None]
            mask = mask.float()

            loss = criterion(logits[mask == 1], targets[mask == 1])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_steps += targets.size(0)

        avg_train_loss = total_loss / max(total_steps, 1)
        val_auc = evaluate_step_auc(model, val_loader, device=DEVICE)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train loss: {avg_train_loss:.4f} | Val step AUROC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> New best model saved (val AUROC = {best_val_auc:.4f})")

    # ---------- TEST EVAL ----------
    print("\nLoading best model and evaluating on test set...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    test_auc = evaluate_step_auc(model, test_loader, device=DEVICE)
    print(f"\n=== GRU Step-level Evaluation (6h prediction) ===")
    print(f"Step-level AUROC (test): {test_auc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


def evaluate_step_auc(model, data_loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets, lengths in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs, lengths)  # (batch, seq_len)
            probs = torch.sigmoid(logits)

            max_len = targets.size(1)
            mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]

            all_probs.append(probs[mask].cpu())
            all_labels.append(targets[mask].cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Edge case: if labels are all 0 or all 1, AUROC is undefined
    if len(np.unique(all_labels)) < 2:
        return float("nan")

    return roc_auc_score(all_labels, all_probs)


if __name__ == "__main__":
    main()
