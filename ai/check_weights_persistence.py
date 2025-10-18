"""
Vérifie si les poids sont bien persistés entre les exécutions
"""
import numpy as np
from nn_evaluator import load_evaluator_from_file, save_weights
import os

WEIGHTS_FILE = "chess_nn_weights.npz"

print("=" * 70)
print("DIAGNOSTIC: Persistance des poids")
print("=" * 70)

# 1. Vérifier que le fichier existe
if not os.path.exists(WEIGHTS_FILE):
    print(f"❌ ERREUR: {WEIGHTS_FILE} n'existe pas!")
    exit(1)

print(f"✓ Fichier trouvé: {WEIGHTS_FILE}")

# 2. Obtenir les infos du fichier
file_stats = os.stat(WEIGHTS_FILE)
print(f"  Taille: {file_stats.st_size:,} bytes")
print(f"  Dernière modification: {file_stats.st_mtime}")

# 3. Charger les poids
print("\nChargement des poids...")
evaluator = load_evaluator_from_file(WEIGHTS_FILE)

# 4. Afficher des statistiques sur les poids
print("\nStatistiques des poids actuels:")
print(f"  weights1: shape={evaluator.weights1.shape}, mean={evaluator.weights1.mean():.6f}, std={evaluator.weights1.std():.6f}")
print(f"  biases1:  shape={evaluator.biases1.shape}, mean={evaluator.biases1.mean():.6f}, std={evaluator.biases1.std():.6f}")
print(f"  weights2: shape={evaluator.weights2.shape}, mean={evaluator.weights2.mean():.6f}, std={evaluator.weights2.std():.6f}")
print(f"  biases2:  shape={evaluator.biases2.shape}, mean={evaluator.biases2.mean():.6f}, std={evaluator.biases2.std():.6f}")
print(f"  weights3: shape={evaluator.weights3.shape}, mean={evaluator.weights3.mean():.6f}, std={evaluator.weights3.std():.6f}")
print(f"  biases3:  shape={evaluator.biases3.shape}, mean={evaluator.biases3.mean():.6f}, std={evaluator.biases3.std():.6f}")

# 5. Calculer une signature unique (somme de tous les poids)
signature = (evaluator.weights1.sum() + evaluator.biases1.sum() + 
             evaluator.weights2.sum() + evaluator.biases2.sum() + 
             evaluator.weights3.sum() + evaluator.biases3.sum())
print(f"\nSignature des poids (somme totale): {signature:.10f}")

# 6. Sauvegarder la signature dans un fichier pour comparaison
signature_file = "weights_signature.txt"
if os.path.exists(signature_file):
    with open(signature_file, 'r') as f:
        old_signature = float(f.read().strip())
    print(f"Signature précédente:             {old_signature:.10f}")
    
    if abs(signature - old_signature) < 1e-6:
        print("❌ PROBLÈME: Les poids n'ont PAS changé depuis la dernière exécution!")
        print("   → Les modifications ne sont pas persistées")
    else:
        diff_pct = 100 * abs(signature - old_signature) / (abs(old_signature) + 1e-10)
        print(f"✓ Les poids ont changé! (différence: {diff_pct:.2f}%)")
else:
    print("(Première exécution, pas de signature précédente)")

# 7. Sauvegarder la signature actuelle
with open(signature_file, 'w') as f:
    f.write(f"{signature:.10f}")
print(f"\n✓ Signature sauvegardée dans {signature_file}")

# 8. Test: modifier légèrement les poids et re-sauvegarder
print("\n" + "=" * 70)
print("TEST: Modification et sauvegarde des poids")
print("=" * 70)

old_w1_mean = evaluator.weights1.mean()
print(f"weights1 mean avant modification: {old_w1_mean:.10f}")

# Modifier très légèrement
evaluator.weights1 += 0.0001

new_w1_mean = evaluator.weights1.mean()
print(f"weights1 mean après modification: {new_w1_mean:.10f}")

# Sauvegarder
test_file = "test_weights.npz"
save_weights(evaluator, test_file)
print(f"✓ Poids modifiés sauvegardés dans {test_file}")

# Recharger et vérifier
evaluator_reloaded = load_evaluator_from_file(test_file)
reloaded_w1_mean = evaluator_reloaded.weights1.mean()
print(f"weights1 mean après rechargement: {reloaded_w1_mean:.10f}")

if abs(reloaded_w1_mean - new_w1_mean) < 1e-9:
    print("✓✓ SAUVEGARDE/RECHARGEMENT FONCTIONNE CORRECTEMENT")
else:
    print("❌❌ PROBLÈME: Les poids rechargés sont différents!")
    print(f"   Différence: {abs(reloaded_w1_mean - new_w1_mean)}")

# Nettoyer
os.remove(test_file)

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("Relance ce script après un entraînement pour vérifier si les poids changent.")
print("=" * 70)
