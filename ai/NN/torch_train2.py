import sys
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
DATASET_PATH = r"C:\Users\maell\Documents\smart_chess_drive\chessData\smart-chess\chessData.csv"
WEIGHTS_FILE = "chess_nn_weights.npz"
CHECKPOINT_FILE = "chess_model_checkpoint.pt"

# Architecture (NNUE-like: 768 → 4096 → 256 → 32 → 1)
HIDDEN1 = 4096
HIDDEN2 = 256
HIDDEN3 = 32
DROPOUT = 0.0  # NNUE typically doesn't use dropout

# Hyperparamètres
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005  # L2 regularization (AdamW)
EPOCHS = 10
BATCH_SIZE = 128  # Plus grand pour GPU
MAX_SAMPLES = 500_000  # Plus de données avec GPU !
EVAL_MAX_SAMPLES = 5000

# Gradient debugging / instrumentation
LOG_GRAD_NORM = True  # Imprimer périodiquement la norme des gradients
GRAD_LOG_EVERY = 100   # toutes les N mini-batches

# Options
USE_SAMPLING = True
RESET_WEIGHTS = False
DEBUG_STATS = True

# LR Scheduler
USE_LR_SCHEDULER = True
LR_PATIENCE = 3
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
        model = TorchNNEvaluator(hidden1=HIDDEN1, hidden2=HIDDEN2, hidden3=HIDDEN3, dropout=DROPOUT)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model, _, start_step = torch_load_checkpoint(CHECKPOINT_FILE, model, optimizer, device=DEVICE)
        # Also read checkpoint metadata (best_rmse) if present so we don't reset it
        try:
            ckpt_raw = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
            # Determine whether the checkpoint actually contains model weights.
            # We base the validity of `best_rmse` on that: if the checkpoint has no
            # 'model' entry (or it's empty), the stored best_rmse is considered
            # stale and will be removed.
            has_model_weights = 'model' in ckpt_raw and bool(ckpt_raw.get('model'))
            if 'best_rmse' in ckpt_raw and not has_model_weights:
                print(f"⚠️  Checkpoint contient 'best_rmse' mais ne contient pas de poids -> suppression du meilleur RMSE pour éviter métrique obsolète.")
                ckpt_copy = dict(ckpt_raw)
                ckpt_copy.pop('best_rmse', None)
                try:
                    torch.save(ckpt_copy, CHECKPOINT_FILE)
                    print(f"✅ best_rmse supprimé du checkpoint {CHECKPOINT_FILE}")
                except Exception as ex:
                    print(f"⚠️ Impossible de réécrire le checkpoint pour supprimer best_rmse: {ex}")
                best_rmse = float('inf')
            else:
                best_rmse = float(ckpt_raw.get('best_rmse', float('inf')))
        except Exception:
            best_rmse = float('inf')

        print(f"✅ Checkpoint chargé (step {start_step}), best_rmse={best_rmse}")
    elif os.path.exists(WEIGHTS_FILE):
        print(f"📥 Chargement des poids NumPy: {WEIGHTS_FILE}")
        model, adam_moments = load_from_npz(WEIGHTS_FILE, device=str(DEVICE))
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # If the saved .npz contains metadata (learning_rate / best_rmse), apply it
        if adam_moments is not None:
            # learning_rate saved as float in our save convention
            if 'learning_rate' in adam_moments:
                try:
                    lr_saved = float(adam_moments['learning_rate'])
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr_saved
                    print(f"ℹ️ LR restauré depuis {WEIGHTS_FILE}: {lr_saved:.6f}")
                except Exception:
                    pass
            if 'best_rmse' in adam_moments:
                try:
                    best_rmse = float(adam_moments['best_rmse'])
                    print(f"ℹ️ best_rmse restauré depuis {WEIGHTS_FILE}: {best_rmse:.4f}")
                except Exception:
                    best_rmse = float('inf')
        # TODO: Restaurer les moments Adam si présents (dépend de correspondance de forme)
        # If we have a checkpoint metadata file with a best_rmse and the npz exists,
        # allow loading that best_rmse so training can resume with the historical metric.
        try:
            if os.path.exists(CHECKPOINT_FILE):
                ckpt_raw = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
                # Only honor best_rmse from the checkpoint if the checkpoint
                # actually contains model weights.
                has_model_weights = 'model' in ckpt_raw and bool(ckpt_raw.get('model'))
                if has_model_weights:
                    best_rmse = float(ckpt_raw.get('best_rmse', float('inf')))
                else:
                    best_rmse = float('inf')
            else:
                best_rmse = float('inf')
        except Exception:
            best_rmse = float('inf')

        print(f"✅ Poids chargés depuis NumPy (best_rmse initial={best_rmse})")
    else:
        print("🆕 Création d'un nouveau réseau...")
        model = TorchNNEvaluator(hidden1=HIDDEN1, hidden2=HIDDEN2, hidden3=HIDDEN3, dropout=DROPOUT)
        # Initialisation He (PyTorch le fait déjà par défaut pour Linear + ReLU)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Warm-start du biais de sortie (l4 est maintenant la couche de sortie)
        with torch.no_grad():
            model.l4.bias[0] = eval_mean
    
    model.to(DEVICE)
    # --- Interactive LR selection at start ---
    # If a previous run saved an optimizer LR in the checkpoint or a metadata key
    # in the .npz, show it and allow the user to enter a LR to start with.
    recorded_lr = None
    try:
        if os.path.exists(CHECKPOINT_FILE):
            ck = torch.load(CHECKPOINT_FILE, map_location='cpu')
            optim_raw = ck.get('optim')
            if optim_raw and isinstance(optim_raw, dict):
                pgs = optim_raw.get('param_groups')
                if pgs and len(pgs) > 0:
                    recorded_lr = float(pgs[0].get('lr', recorded_lr))
    except Exception:
        recorded_lr = recorded_lr

    # fallback to .npz metadata if present
    try:
        if recorded_lr is None and os.path.exists(WEIGHTS_FILE):
            data = np.load(WEIGHTS_FILE)
            if 'learning_rate' in data:
                recorded_lr = float(data['learning_rate'])
    except Exception:
        recorded_lr = recorded_lr

    default_lr = recorded_lr if recorded_lr is not None else LEARNING_RATE
    if recorded_lr is not None:
        print(f"ℹ️ Learning rate enregistré détecté: {recorded_lr:.6f}")
    else:
        print(f"ℹ️ Aucun learning rate enregistré trouvé; valeur par défaut: {default_lr:.6f}")

    try:
        # interactive prompt: user can press Enter to accept default
        ui = input(f"Saisir le learning rate de départ [{default_lr:.6f}]: ").strip()
        if ui != "":
            chosen_lr = float(ui)
        else:
            chosen_lr = float(default_lr)
    except Exception:
        print("⚠️ Entrée invalide — utilisation de la valeur par défaut")
        chosen_lr = float(default_lr)

    # Apply chosen LR to optimizer param groups
    try:
        for pg in optimizer.param_groups:
            pg['lr'] = chosen_lr
        print(f"➡️ Learning rate initial utilisé pour l'entraînement: {chosen_lr:.6f}")
    except Exception:
        print("⚠️ Impossible d'appliquer le learning rate à l'optimizer; vérifiez l'état de l'optimizer")
    
    # 3. Configuration de l'entraînement
    criterion = nn.MSELoss()
    
    # LR Scheduler
    if USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_FACTOR, 
            patience=LR_PATIENCE
        )
    
    print(f"\n{'='*70}")
    # --- Sanity check: display actual model layer sizes vs configured architecture ---
    try:
        in_f = getattr(model.l1, 'in_features', None)
        out1 = getattr(model.l1, 'out_features', None)
        out2 = getattr(model.l2, 'out_features', None)
        out3 = getattr(model.l3, 'out_features', None)
        out4 = getattr(model.l4, 'out_features', None)
        if None not in (in_f, out1, out2, out3, out4):
            print(f"🔎 Modèle détecté: {in_f} → {out1} → {out2} → {out3} → {out4}")
            if (out1, out2, out3) != (HIDDEN1, HIDDEN2, HIDDEN3):
                print(f"⚠️ Les constantes HIDDEN1={HIDDEN1}, HIDDEN2={HIDDEN2}, HIDDEN3={HIDDEN3} diffèrent des tailles effectives du modèle ({out1}, {out2}, {out3}). Ceci peut venir d'un chargement de poids sauvegardés avec une architecture différente.")
    except Exception:
        pass

    print(f"Configuration:")
    print(f"  Dataset complet: {len(all_fens):,} positions")
    print(f"  Échantillon/epoch: {MAX_SAMPLES if USE_SAMPLING else len(all_fens):,} positions")
    print(f"  Architecture (NNUE-like): 768 → {HIDDEN1} → {HIDDEN2} → {HIDDEN3} → 1")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Learning rate: {LEARNING_RATE} (AdamW, weight decay: {WEIGHT_DECAY})")
    print(f"  LR Warmup: {USE_LR_WARMUP} ({WARMUP_START_LR if USE_LR_WARMUP else 'N/A'} → {LEARNING_RATE})")
    print(f"  LR Scheduler: {USE_LR_SCHEDULER} (patience: {LR_PATIENCE if USE_LR_SCHEDULER else 'N/A'})")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}\n")
    # 4. Boucle d'entraînement
    # If a checkpoint or weights file existed at startup we treat this run as
    # a resume. In that case we disable the LR warmup and reuse the optimizer
    # LR (if restored from checkpoint) or the metadata saved in the .npz.
    resumed = False
    effective_use_lr_warmup = USE_LR_WARMUP
    if os.path.exists(CHECKPOINT_FILE) or os.path.exists(WEIGHTS_FILE):
        resumed = True
        effective_use_lr_warmup = False
        print("ℹ️ Reprise détectée: désactivation du LR warmup pour utiliser le LR sauvegardé/précédent.")

    # best_rmse may have been set during checkpoint/weights loading above; default to +inf
    best_rmse = locals().get('best_rmse', float('inf'))

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
        if effective_use_lr_warmup and epoch < WARMUP_EPOCHS:
            warmup_progress = (epoch + 1) / WARMUP_EPOCHS
            lr = WARMUP_START_LR + (LEARNING_RATE - WARMUP_START_LR) * warmup_progress
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"🔥 Warmup epoch {epoch+1}/{WARMUP_EPOCHS}: LR = {lr:.6f}")

        # Afficher le LR courant utilisé pour cette époque (après warmup si applicable)
        try:
            lrs = [pg.get('lr', None) for pg in optimizer.param_groups]
            if len(lrs) == 1:
                print(f"➡️ Learning rate courant: {lrs[0]:.6f}")
            else:
                print("➡️ Learning rates courants: " + ", ".join(f"{x:.6f}" for x in lrs))
        except Exception:
            # Défensif: si optimizer absent ou structure inattendue, afficher l'hyperparamètre initial
            print(f"➡️ Learning rate (approx): {LEARNING_RATE}")
        
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
            
            # Check for NaNs/Infs before moving to device
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print(f"⚠️ WARNING: NaN or Inf found in inputs for batch {batch_idx}, epoch {epoch+1}. Skipping batch.")
                # Optional: Add more info about the problematic data
                # print("Problematic inputs:", inputs[torch.isnan(inputs) | torch.isinf(inputs)])
                continue # Skip this batch

            if torch.isnan(targets).any() or torch.isinf(targets).any():
                print(f"⚠️ WARNING: NaN or Inf found in targets for batch {batch_idx}, epoch {epoch+1}. Skipping batch.")
                # Optional: Add more info about the problematic data
                # print("Problematic targets:", targets[torch.isnan(targets) | torch.isinf(targets)])
                continue # Skip this batch

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward
            loss.backward()

            # --- Instrumentation gradients (avant clipping) ---
            if LOG_GRAD_NORM and (batch_idx % GRAD_LOG_EVERY == 0 or batch_idx == 0):
                try:
                    total_grad_norm_sq = 0.0
                    max_abs_grad = 0.0
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        gn = float(g.norm(2).cpu().item())
                        total_grad_norm_sq += gn * gn
                        try:
                            max_abs_grad = max(max_abs_grad, float(g.abs().max().cpu().item()))
                        except Exception:
                            pass
                    total_grad_norm = total_grad_norm_sq ** 0.5
                    total_param_norm_sq = 0.0
                    for p in model.parameters():
                        try:
                            total_param_norm_sq += float(p.data.norm(2).cpu().item()) ** 2
                        except Exception:
                            pass
                    total_param_norm = total_param_norm_sq ** 0.5
                    print(f"[GRAD] epoch={epoch+1} batch={batch_idx} grad_norm={total_grad_norm:.6f} max_abs_grad={max_abs_grad:.6f} param_norm={total_param_norm:.6f}")
                except Exception:
                    pass
            
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
        if USE_LR_SCHEDULER and (not effective_use_lr_warmup or epoch >= WARMUP_EPOCHS):
            scheduler.step(rmse)
        
        # Sauvegarder le meilleur modèle
        if rmse < best_rmse:
            best_rmse = rmse
            print(f"💾 Nouveau meilleur RMSE: {best_rmse:.4f} - Sauvegarde...")
            torch_save_checkpoint(CHECKPOINT_FILE, model, optimizer, epoch, best_rmse=best_rmse)
            # Save weights + metadata (learning_rate and best_rmse) so next runs can resume
            try:
                current_lr = float(optimizer.param_groups[0].get('lr', LEARNING_RATE))
            except Exception:
                current_lr = LEARNING_RATE
            meta = {'learning_rate': current_lr, 'best_rmse': float(best_rmse)}
            save_weights_npz(model, WEIGHTS_FILE, adam_moments=meta)
    
    print("\n🎉 Entraînement terminé!")
    print(f"📊 Meilleur RMSE: {best_rmse:.4f}")
    
    # Sauvegarde finale
    print(f"\n💾 Sauvegarde finale...")
    torch_save_checkpoint(CHECKPOINT_FILE, model, optimizer, EPOCHS, best_rmse=best_rmse)
    try:
        current_lr = float(optimizer.param_groups[0].get('lr', LEARNING_RATE))
    except Exception:
        current_lr = LEARNING_RATE
    meta = {'learning_rate': current_lr, 'best_rmse': float(best_rmse)}
    save_weights_npz(model, WEIGHTS_FILE, adam_moments=meta)
    print(f"✅ Modèle sauvegardé dans {CHECKPOINT_FILE} et {WEIGHTS_FILE}")


if __name__ == "__main__":
    main()
