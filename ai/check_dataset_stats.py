import pandas as pd
import numpy as np

# Charger le dataset
df = pd.read_csv(
    r'C:\Users\gauti\OneDrive\Documents\UE commande\chessData.csv',
    names=['FEN', 'Evaluation'],
    skiprows=1,
    comment='#'
)

# Nettoyer
initial = len(df)
df.dropna(inplace=True)
print(f"Dataset: {len(df)} positions valides (supprimé {initial - len(df)} NaN)")

# Normaliser comme dans train.py
evals = df['Evaluation'].astype(int).values / 1000.0

print(f"\n=== Statistiques des évaluations normalisées (divisées par 1000) ===")
print(f"Moyenne: {evals.mean():.4f}")
print(f"Écart-type: {evals.std():.4f}")
print(f"Min: {evals.min():.2f}")
print(f"Max: {evals.max():.2f}")

# Baseline: toujours prédire la moyenne
baseline_rmse = evals.std()
print(f"\n=== Analyse de la loss ===")
print(f"Baseline RMSE (toujours prédire la moyenne): {baseline_rmse:.4f}")
print(f"Ta loss actuelle (RMSE): 0.60")

if baseline_rmse > 0.6:
    improvement = 100 * (1 - 0.6 / baseline_rmse)
    print(f"✓ Amélioration de {improvement:.1f}% par rapport à la baseline")
    print(f"  → C'est NORMAL et BON après 1 époque!")
else:
    print(f"✗ La loss est supérieure à la baseline - le modèle n'apprend pas encore")

# R² approximatif
r2 = 1 - (0.6**2 / baseline_rmse**2)
print(f"\nR² approximatif: {r2:.3f}")
print(f"(0 = aucune prédiction, 1 = prédiction parfaite)")

print(f"\n=== Objectifs raisonnables ===")
print(f"Après plusieurs époques:")
print(f"  - RMSE ~0.3-0.4 serait excellent (amélioration ~50-60%)")
print(f"  - RMSE ~0.4-0.5 serait très bon (amélioration ~35-50%)")
print(f"  - RMSE ~0.5-0.6 serait correct pour début d'entraînement")
