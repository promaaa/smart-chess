"""
Script d'entraînement PyTorch optimisé pour GPU
Compatible avec Google Colab et machines locales avec GPU
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

from Chess import Chess
from ai.NN.torch_nn_evaluator import TorchNNEvaluator, save_weights_npz, load_from_npz, torch_save_checkpoint, torch_load_checkpoint

# --- CONFIGURATION DE L'ENTRAÎNEMENT ---
DATASET_PATH = "C:\\Users\\gauti\\OneDrive\\Documents\\UE commande\\chessData.csv"  # Adapté pour Colab (fichier à la racine)
WEIGHTS_FILE = "chess_nn_weights.npz"
CHECKPOINT_FILE = "chess_model_checkpoint.pt"

# Architecture
HIDDEN_SIZE = 256
DROPOUT = 0.3
LEAKY_ALPHA = 0.01

# Hyperparamètres
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # L2 regularization (AdamW)
EPOCHS = 20
BATCH_SIZE = 128  # Plus grand pour GPU
MAX_SAMPLES = 500_000  # Plus de données avec GPU !
EVAL_MAX_SAMPLES = 5000

# Options
USE_SAMPLING = True
RESET_WEIGHTS = False
DEBUG_STATS = True

# LR Scheduler
USE_LR_SCHEDULER = True
LR_PATIENCE = 2
LR_FACTOR = 0.5

# LR Warmup
USE_LR_WARMUP = True
WARMUP_EPOCHS = 3
WARMUP_START_LR = 0.0001

# Device (auto-détection GPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class ChessDataset(Dataset):
    """Dataset PyTorch pour les positions d'échecs"""
    def __init__(self, fens, evaluations):
        self.fens = fens
        self.evaluations = evaluations
        self.chess = Chess()
        
        # Précalculer l'encodage pour accélérer (optionnel, consomme plus de RAM)
        # self.encoded = [self._encode_fen(fen) for fen in tqdm(fens, desc="Encoding positions")]
        
    def __len__(self):
        return len(self.fens)
    
    def __getitem__(self, idx):
        fen = self.fens[idx]
        target = self.evaluations[idx]
        
        # Encoder la position
        self.chess.load_fen(fen)
        encoded = self._encode_board(self.chess)
        
        return torch.from_numpy(encoded).float(), torch.tensor([target], dtype=torch.float32)
    
    def _encode_board(self, chess_instance):
        """Encode le plateau en vecteur 768D (identique à nn_evaluator.py)"""
        piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        vec = np.zeros(768, dtype=np.float32)
        for piece_char, bitboard in chess_instance.bitboards.items():
            if bitboard == 0:
                continue
            piece_index = piece_to_index[piece_char]
            temp_bb = int(bitboard)
            while temp_bb:
                square = (temp_bb & -temp_bb).bit_length() - 1
                vector_position = piece_index * 64 + square
                vec[vector_position] = 1.0
                temp_bb &= temp_bb - 1
        return vec


def load_data(filepath: str):
    """Charge le dataset FEN,Evaluation et le nettoie."""
    print(f"📂 Chargement du dataset depuis {filepath}...")
    
    df = pd.read_csv(
        filepath, 
        names=['FEN', 'Evaluation'], 
        skiprows=1,
        comment='#'
    )
    
    initial_count = len(df)
    df.dropna(inplace=True)
    cleaned_count = len(df)
    
    if initial_count > cleaned_count:
        print(f"🧹 Nettoyage : {initial_count - cleaned_count} lignes corrompues supprimées.")
    
    fens = df['FEN'].values
    EVAL_SCALE_FACTOR = 1000.0
    evaluations = (df['Evaluation'].astype(int).values) / EVAL_SCALE_FACTOR
    
    print(f"✅ {len(fens):,} positions valides chargées.")
    return fens, evaluations


def evaluate_model(model, dataloader, device):
    """Évalue le modèle sur un dataset"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(labels.numpy().flatten())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    mae = float(np.mean(np.abs(predictions - targets)))
    corr = float(np.corrcoef(predictions, targets)[0, 1]) if len(predictions) > 1 else 0.0
    
    return rmse, mae, corr, predictions, targets


def main():
    # 1. Charger les données
    all_fens, all_evaluations = load_data(DATASET_PATH)
    
    print(f"\n📊 Dataset complet: {len(all_fens):,} positions")
    
    eval_mean = float(np.mean(all_evaluations))
    
    # 2. Initialiser le modèle
    if RESET_WEIGHTS and os.path.exists(WEIGHTS_FILE):
        print(f"🗑️  Suppression des anciens poids: {WEIGHTS_FILE}")
        os.remove(WEIGHTS_FILE)
    
    if os.path.exists(CHECKPOINT_FILE):
        print(f"📥 Chargement du checkpoint PyTorch: {CHECKPOINT_FILE}")
        model = TorchNNEvaluator(hidden_size=HIDDEN_SIZE, dropout=DROPOUT, leaky_alpha=LEAKY_ALPHA)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model, _, start_step = torch_load_checkpoint(CHECKPOINT_FILE, model, optimizer, device=DEVICE)
        print(f"✅ Checkpoint chargé (step {start_step})")
    elif os.path.exists(WEIGHTS_FILE):
        print(f"📥 Chargement des poids NumPy: {WEIGHTS_FILE}")
        model, adam_moments = load_from_npz(WEIGHTS_FILE, device=DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # TODO: Restaurer les moments Adam si présents
        print(f"✅ Poids chargés depuis NumPy")
    else:
        print("🆕 Création d'un nouveau réseau...")
        model = TorchNNEvaluator(hidden_size=HIDDEN_SIZE, dropout=DROPOUT, leaky_alpha=LEAKY_ALPHA)
        # Initialisation He (PyTorch le fait déjà par défaut pour Linear + ReLU)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Warm-start du biais de sortie
        with torch.no_grad():
            model.l3.bias[0] = eval_mean
    
    model.to(DEVICE)
    
    # 3. Configuration de l'entraînement
    criterion = nn.MSELoss()
    
    # LR Scheduler
    if USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_FACTOR, 
            patience=LR_PATIENCE
        )
    
    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"  Dataset complet: {len(all_fens):,} positions")
    print(f"  Échantillon/epoch: {MAX_SAMPLES if USE_SAMPLING else len(all_fens):,} positions")
    print(f"  Architecture: 768 → {HIDDEN_SIZE} → {HIDDEN_SIZE} → 1")
    print(f"  Dropout: {DROPOUT}")
    print(f"  LeakyReLU alpha: {LEAKY_ALPHA}")
    print(f"  Learning rate: {LEARNING_RATE} (AdamW, weight decay: {WEIGHT_DECAY})")
    print(f"  LR Warmup: {USE_LR_WARMUP} ({WARMUP_START_LR if USE_LR_WARMUP else 'N/A'} → {LEARNING_RATE})")
    print(f"  LR Scheduler: {USE_LR_SCHEDULER} (patience: {LR_PATIENCE if USE_LR_SCHEDULER else 'N/A'})")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}\n")
    
    # 4. Boucle d'entraînement
    best_rmse = float('inf')
    
    for epoch in range(EPOCHS):
        # Échantillonnage à chaque epoch
        if USE_SAMPLING and len(all_fens) > MAX_SAMPLES:
            print(f"\n[Epoch {epoch+1}] 🎲 Échantillonnage: {MAX_SAMPLES:,} positions sur {len(all_fens):,}")
            idx = np.random.choice(len(all_fens), size=MAX_SAMPLES, replace=False)
            fens = all_fens[idx]
            evaluations = all_evaluations[idx]
        else:
            fens = all_fens
            evaluations = all_evaluations
        
        # LR Warmup
        if USE_LR_WARMUP and epoch < WARMUP_EPOCHS:
            warmup_progress = (epoch + 1) / WARMUP_EPOCHS
            lr = WARMUP_START_LR + (LEARNING_RATE - WARMUP_START_LR) * warmup_progress
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"🔥 Warmup epoch {epoch+1}/{WARMUP_EPOCHS}: LR = {lr:.6f}")
        
        # Créer le dataset et dataloader
        train_dataset = ChessDataset(fens, evaluations)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=0,  # Augmenter si CPU multi-core (ex: 4)
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Update
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Debug stats (premier batch)
            if DEBUG_STATS and epoch == 0 and batch_idx == 0:
                with torch.no_grad():
                    preds = outputs.cpu().numpy().flatten()
                    targs = targets.cpu().numpy().flatten()
                    batch_rmse = np.sqrt(np.mean((preds - targs) ** 2))
                    corr = np.corrcoef(preds, targs)[0, 1] if len(preds) > 1 else 0.0
                    print(f"\n[DEBUG batch 0] targets mean={targs.mean():.4f} std={targs.std():.4f}; "
                          f"preds mean={preds.mean():.4f} std={preds.std():.4f}; "
                          f"RMSE={batch_rmse:.4f}; corr={corr:.4f}")
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({"loss": f"{np.sqrt(avg_loss):.4f}"})
        
        # Évaluation fin d'époque
        print(f"\n🔍 Évaluation epoch {epoch+1}...")
        
        # Échantillon d'évaluation
        if EVAL_MAX_SAMPLES and len(all_fens) > EVAL_MAX_SAMPLES:
            eval_idx = np.random.choice(len(all_fens), size=EVAL_MAX_SAMPLES, replace=False)
            eval_fens = all_fens[eval_idx]
            eval_targets = all_evaluations[eval_idx]
        else:
            eval_fens = all_fens
            eval_targets = all_evaluations
        
        eval_dataset = ChessDataset(eval_fens, eval_targets)
        eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE*2, shuffle=False)
        
        rmse, mae, corr, preds, targets = evaluate_model(model, eval_loader, DEVICE)
        
        baseline_rmse = targets.std()
        improvement = 100 * (1 - rmse / baseline_rmse) if baseline_rmse > 0 else 0
        
        # Affichage
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{EPOCHS} - Évaluation sur {len(eval_fens):,} positions")
        print(f"{'='*70}")
        print(f"  RMSE:        {rmse:.4f}  (baseline: {baseline_rmse:.4f})")
        print(f"  MAE:         {mae:.4f}")
        print(f"  Amélioration: {improvement:+.1f}% vs baseline")
        print(f"  Corrélation: {corr:.4f}")
        print(f"  Std preds:   {preds.std():.4f}  (cible: {targets.std():.4f})")
        print(f"  Mean preds:  {preds.mean():.4f}  (cible: {targets.mean():.4f})")
        
        if improvement > 50:
            print(f"  ✓✓ Performance excellente!")
        elif improvement > 30:
            print(f"  ✓  Bon apprentissage!")
        elif improvement > 10:
            print(f"  →  Apprentissage en cours")
        else:
            print(f"  ⚠  Faible amélioration - vérifier hyperparamètres")
        print(f"{'='*70}\n")
        
        # LR Scheduler
        if USE_LR_SCHEDULER and (not USE_LR_WARMUP or epoch >= WARMUP_EPOCHS):
            scheduler.step(rmse)
        
        # Sauvegarder le meilleur modèle
        if rmse < best_rmse:
            best_rmse = rmse
            print(f"💾 Nouveau meilleur RMSE: {best_rmse:.4f} - Sauvegarde...")
            torch_save_checkpoint(CHECKPOINT_FILE, model, optimizer, epoch)
            save_weights_npz(model, WEIGHTS_FILE)
    
    print("\n🎉 Entraînement terminé!")
    print(f"📊 Meilleur RMSE: {best_rmse:.4f}")
    
    # Sauvegarde finale
    print(f"\n💾 Sauvegarde finale...")
    torch_save_checkpoint(CHECKPOINT_FILE, model, optimizer, EPOCHS)
    save_weights_npz(model, WEIGHTS_FILE)
    print(f"✅ Modèle sauvegardé dans {CHECKPOINT_FILE} et {WEIGHTS_FILE}")


if __name__ == "__main__":
    main()
