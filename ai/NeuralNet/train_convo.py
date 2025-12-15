"""
Training script for a compact 1D CNN evaluator from FEN + centipawn labels.

Features:
- Sampling and warmup for stable training on large datasets
- AdamW with weight decay
- Simple RMSE validation on a held-out sample

Adjust `DATASET_PATH` and hyperparameters to your environment.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Chess import Chess

# ==============================
# CONFIGURATION
# ==============================
DATASET_PATH = r"C:\Users\maell\Documents\smart_chess_drive\chessData\smart-chess\chessData.csv"
CHECKPOINT_FILE = "chess_cnn_checkpoint.pt"
WEIGHTS_FILE = "chess_cnn_weights.npz"

BATCH_SIZE = 256
EPOCHS = 20
LR = 0.0005
WEIGHT_DECAY = 1e-5
MAX_SAMPLES = 400_000
EVAL_MAX = 6000
USE_SAMPLING = True
USE_WARMUP = True
WARMUP_EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"📌 Device: {DEVICE}")


# ==============================
# DATASET
# ==============================
class ChessDataset(Dataset):
    """Minimal dataset mapping FEN strings to encoded planes and target eval."""
    def __init__(self, fens, evals):
        self.fens = fens
        self.evals = evals
        self.chess = Chess()

    def __len__(self): 
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]
        val = self.evals[idx]
        self.chess.load_fen(fen)
        x = self.encode_board(self.chess)
        return torch.tensor(x, dtype=torch.float32), torch.tensor([val], dtype=torch.float32)

    @staticmethod
    def encode_board(chess):
        piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}
        v = np.zeros(768, dtype=np.float32)
        for p, bb in chess.bitboards.items():
            if bb == 0: continue
            offset = piece_map[p] * 64
            b = int(bb)
            while b:
                s = (b & -b).bit_length() - 1
                v[offset + s] = 1
                b &= b - 1
        return v


# ==============================
# CNN MODEL
# ==============================
class ChessCNN(nn.Module):
    """Tiny 1D CNN over 768-bit board encoding, outputs a single eval value."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)   # (B,1,768)
        x = self.conv(x)
        x = self.pool(x).squeeze(2)
        return self.fc(x)


# ==============================
# LOAD DATA (FIXED)
# ==============================
def load_data(path):
    """Load and clean CSV with columns [FEN, Evaluation].

    Converts mate scores to ±6 pawns and centipawns to pawns.
    Returns (fens, evals) numpy arrays.
    """

    def clean_eval(v):
        v = str(v).strip()
        if "#" in v:
            return 6.0 if "+" in v else -6.0   # convert mate to ±6 pawns
        try:
            return float(v) / 1000.0
        except:
            return None

    df = pd.read_csv(path, names=["FEN","Evaluation"], skiprows=1)
    df.dropna(inplace=True)
    
    df["Evaluation"] = df["Evaluation"].apply(clean_eval)
    df.dropna(inplace=True)

    print(f"Loaded {len(df):,} clean samples.")
    return df["FEN"].values, df["Evaluation"].values


# ==============================
# EVALUATION
# ==============================
def evaluate(model, loader):
    """Compute RMSE over a dataloader without gradient."""
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for x, t in loader:
            x = x.to(DEVICE)
            preds.extend(model(x).cpu().numpy().flatten())
            targs.extend(t.numpy().flatten())
    preds, targs = np.array(preds), np.array(targs)
    rmse = np.sqrt(np.mean((preds - targs)**2))
    return rmse


# ==============================
# TRAINING
# ==============================
def train():
    """Main training loop with optional sampling and LR warmup.

    Saves the best checkpoint (by RMSE) to `CHECKPOINT_FILE`.
    """
    fens, evals = load_data(DATASET_PATH)

    model = ChessCNN().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_rmse = float("inf")

    for epoch in range(EPOCHS):

        # Sampling
        if USE_SAMPLING and len(fens) > MAX_SAMPLES:
            idx = np.random.choice(len(fens), MAX_SAMPLES, replace=False)
            X, Y = fens[idx], evals[idx]
        else:
            X, Y = fens, evals

        # Warmup
        if USE_WARMUP and epoch < WARMUP_EPOCHS:
            warm_factor = (epoch+1) / WARMUP_EPOCHS
            lr_now = warm_factor * LR
            for pg in opt.param_groups: pg["lr"] = lr_now
            print(f"🔥 Warmup LR: {lr_now:.6f}")
        else:
            for pg in opt.param_groups: pg["lr"] = LR

        loader = DataLoader(ChessDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True)
        model.train()

        running_loss = 0
        for x, t in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, t = x.to(DEVICE), t.to(DEVICE)
            opt.zero_grad()
            y = model(x)
            loss = loss_fn(y, t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Loss={avg_loss:.6f}")

        # Validation
        val_idx = np.random.choice(len(fens), min(EVAL_MAX, len(fens)), replace=False)
        val_loader = DataLoader(ChessDataset(fens[val_idx], evals[val_idx]), batch_size=512)

        rmse = evaluate(model, val_loader)
        print(f"🔎 RMSE={rmse:.5f}")

        # Save best
        if rmse < best_rmse:
            best_rmse = rmse
            print(f"💾 Saving best model (RMSE={best_rmse:.5f})")
            torch.save({"model": model.state_dict(), "rmse": best_rmse}, CHECKPOINT_FILE)

    print("Training completed!")
    print(f"Best RMSE={best_rmse:.5f}")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    train()
