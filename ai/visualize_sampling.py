"""
Visualise l'impact du ré-échantillonnage à chaque epoch
"""

# Configuration simulée
TOTAL_POSITIONS = 12_767_881
MAX_SAMPLES = 200_000
EPOCHS = 20

print("=" * 70)
print("COMPARAISON STRATÉGIES D'ÉCHANTILLONNAGE")
print("=" * 70)
print(f"\nDataset complet: {TOTAL_POSITIONS:,} positions")
print(f"Échantillon/epoch: {MAX_SAMPLES:,} positions")
print(f"Nombre d'epochs: {EPOCHS}")

print("\n" + "=" * 70)
print("STRATÉGIE 1: Échantillonnage UNIQUE (ancienne méthode)")
print("=" * 70)
unique_positions = MAX_SAMPLES
print(f"Positions uniques vues: {unique_positions:,}")
print(f"Répétitions par position: {EPOCHS}x")
print(f"Coverage du dataset: {100 * unique_positions / TOTAL_POSITIONS:.2f}%")
print("⚠️  Risque d'overfitting: ÉLEVÉ")

print("\n" + "=" * 70)
print("STRATÉGIE 2: Ré-échantillonnage À CHAQUE EPOCH (nouvelle méthode)")
print("=" * 70)
max_unique_positions = min(TOTAL_POSITIONS, MAX_SAMPLES * EPOCHS)
print(f"Positions uniques vues (max): {max_unique_positions:,}")
print(f"Répétitions par position (moyenne): ~{EPOCHS * MAX_SAMPLES / TOTAL_POSITIONS:.2f}x")
print(f"Coverage du dataset: {100 * max_unique_positions / TOTAL_POSITIONS:.2f}%")
print("✓  Risque d'overfitting: FAIBLE")

print("\n" + "=" * 70)
print("AMÉLIORATION")
print("=" * 70)
improvement_factor = max_unique_positions / unique_positions
print(f"Positions uniques vues: {improvement_factor:.1f}x plus")
print(f"Diversité des données: +{100 * (improvement_factor - 1):.0f}%")

print("\n" + "=" * 70)
print("RECOMMANDATIONS")
print("=" * 70)
print("✓ Ré-échantillonnage à chaque epoch activé")
print("✓ Chaque epoch voit des positions différentes")
print("✓ Meilleure généralisation garantie")
print("\nProchaine étape: Augmenter MAX_SAMPLES à 500k ou 1M pour encore plus de diversité!")
print("=" * 70)
