"""
Test de généralisation : compare la performance sur l'échantillon d'entraînement
vs un nouvel échantillon du même dataset
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from Chess import Chess
from nn_evaluator import load_evaluator_from_file

DATASET_PATH = "C:\\Users\\gauti\\OneDrive\\Documents\\UE commande\\chessData.csv"
WEIGHTS_FILE = "chess_nn_weights.npz"
TEST_SAMPLES = 5000  # Plus petit échantillon pour test rapide

def load_data(filepath: str):
    """Charge le dataset."""
    print(f"Chargement du dataset depuis {filepath}...")
    df = pd.read_csv(
        filepath, 
        names=['FEN', 'Evaluation'], 
        skiprows=1,
        comment='#'
    )
    df.dropna(inplace=True)
    fens = df['FEN'].values
    EVAL_SCALE_FACTOR = 1000.0
    evaluations = (df['Evaluation'].astype(int).values) / EVAL_SCALE_FACTOR
    print(f"{len(fens)} positions valides chargées.")
    return fens, evaluations

def evaluate_sample(evaluator, fens, evaluations, name="Test"):
    """Évalue le modèle sur un échantillon."""
    print(f"\n{'='*70}")
    print(f"Évaluation sur {name} ({len(fens)} positions)")
    print(f"{'='*70}")
    
    chess = Chess()
    predictions = []
    targets = []
    
    for fen, target in tqdm(zip(fens, evaluations), total=len(fens), desc=name):
        chess.load_fen(fen)
        pred = evaluator.evaluate_position(chess) / 1000.0  # Rescale to match training
        predictions.append(pred)
        targets.append(target)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Métriques
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    corr = np.corrcoef(predictions, targets)[0, 1]
    
    baseline_rmse = targets.std()
    improvement = 100 * (1 - rmse / baseline_rmse) if baseline_rmse > 0 else 0
    
    print(f"\nRésultats {name}:")
    print(f"  RMSE:         {rmse:.4f}  (baseline: {baseline_rmse:.4f})")
    print(f"  MAE:          {mae:.4f}")
    print(f"  Corrélation:  {corr:.4f}")
    print(f"  Amélioration: {improvement:+.1f}% vs baseline")
    print(f"  Std preds:    {predictions.std():.4f}  (cible: {targets.std():.4f})")
    print(f"  Mean preds:   {predictions.mean():.4f}  (cible: {targets.mean():.4f})")
    
    # Diagnostic overfitting
    if improvement < 10:
        print(f"  ⚠️  OVERFITTING SÉVÈRE - Le modèle ne généralise pas!")
    elif improvement < 30:
        print(f"  ⚠️  Overfitting modéré - Besoin de régularisation")
    elif improvement < 50:
        print(f"  ✓  Généralisation acceptable")
    else:
        print(f"  ✓✓ Excellente généralisation!")
    
    return rmse, corr

def main():
    print("Test de généralisation du modèle")
    print("="*70)
    
    # Charger le modèle
    print(f"\nChargement du modèle depuis {WEIGHTS_FILE}...")
    evaluator = load_evaluator_from_file(WEIGHTS_FILE)
    
    # Charger tout le dataset
    all_fens, all_evals = load_data(DATASET_PATH)
    
    # Fixer la graine pour reproductibilité
    np.random.seed(42)
    
    # Échantillon 1 (simule l'entraînement)
    print(f"\n{'='*70}")
    print("ÉCHANTILLON 1 (seed=42)")
    print(f"{'='*70}")
    idx1 = np.random.choice(len(all_fens), size=TEST_SAMPLES, replace=False)
    fens1 = all_fens[idx1]
    evals1 = all_evals[idx1]
    rmse1, corr1 = evaluate_sample(evaluator, fens1, evals1, "Échantillon 1")
    
    # Échantillon 2 (nouveau, seed différent)
    print(f"\n{'='*70}")
    print("ÉCHANTILLON 2 (seed=123)")
    print(f"{'='*70}")
    np.random.seed(123)
    idx2 = np.random.choice(len(all_fens), size=TEST_SAMPLES, replace=False)
    fens2 = all_fens[idx2]
    evals2 = all_evals[idx2]
    rmse2, corr2 = evaluate_sample(evaluator, fens2, evals2, "Échantillon 2")
    
    # Comparaison
    print(f"\n{'='*70}")
    print("COMPARAISON FINALE")
    print(f"{'='*70}")
    print(f"Échantillon 1:  RMSE={rmse1:.4f}  Corr={corr1:.4f}")
    print(f"Échantillon 2:  RMSE={rmse2:.4f}  Corr={corr2:.4f}")
    
    rmse_diff = abs(rmse2 - rmse1)
    rmse_ratio = rmse2 / rmse1 if rmse1 > 0 else float('inf')
    
    print(f"\nDifférence RMSE: {rmse_diff:.4f}  (ratio: {rmse_ratio:.2f}x)")
    
    if rmse_ratio > 3.0:
        print("❌ OVERFITTING CRITIQUE - Le modèle a complètement mémorisé!")
        print("   → Recommandation: Augmenter la régularisation drastiquement")
    elif rmse_ratio > 1.5:
        print("⚠️  Overfitting important - Performance instable entre échantillons")
        print("   → Recommandation: Ajouter dropout, weight decay, ou plus de données")
    elif rmse_ratio > 1.2:
        print("⚠️  Léger overfitting - Généralisation moyenne")
        print("   → Recommandation: Régularisation modérée")
    else:
        print("✓  Bonne généralisation - Performance stable!")

if __name__ == "__main__":
    main()
