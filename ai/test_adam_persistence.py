"""
Test de la persistance des moments Adam entre les exécutions
"""
import numpy as np
from nn_evaluator import NeuralNetworkEvaluator, save_weights, load_evaluator_from_file
import os

WEIGHTS_FILE = "test_adam_persistence.npz"

print("=" * 70)
print("TEST: Persistance des moments Adam")
print("=" * 70)

# 1. Créer un réseau et des moments Adam simulés
print("\n1. Création d'un réseau et simulation de moments Adam...")
evaluator = NeuralNetworkEvaluator.create_untrained_network()

# Simuler des moments Adam après quelques steps
m_w1 = np.random.randn(*evaluator.weights1.shape) * 0.01
v_w1 = np.random.randn(*evaluator.weights1.shape) * 0.001
m_b1 = np.random.randn(*evaluator.biases1.shape) * 0.01
v_b1 = np.random.randn(*evaluator.biases1.shape) * 0.001
m_w2 = np.random.randn(*evaluator.weights2.shape) * 0.01
v_w2 = np.random.randn(*evaluator.weights2.shape) * 0.001
m_b2 = np.random.randn(*evaluator.biases2.shape) * 0.01
v_b2 = np.random.randn(*evaluator.biases2.shape) * 0.001
m_w3 = np.random.randn(*evaluator.weights3.shape) * 0.01
v_w3 = np.random.randn(*evaluator.weights3.shape) * 0.001
m_b3 = np.random.randn(*evaluator.biases3.shape) * 0.01
v_b3 = np.random.randn(*evaluator.biases3.shape) * 0.001
adam_step = 1234

print(f"  Step Adam simulé: {adam_step}")
print(f"  m_w1 mean: {m_w1.mean():.6f}")
print(f"  v_w1 mean: {v_w1.mean():.6f}")

# 2. Sauvegarder avec moments Adam
print("\n2. Sauvegarde avec moments Adam...")
adam_dict = {
    'm_w1': m_w1, 'v_w1': v_w1,
    'm_b1': m_b1, 'v_b1': v_b1,
    'm_w2': m_w2, 'v_w2': v_w2,
    'm_b2': m_b2, 'v_b2': v_b2,
    'm_w3': m_w3, 'v_w3': v_w3,
    'm_b3': m_b3, 'v_b3': v_b3,
    'adam_step': np.array(adam_step)
}
save_weights(evaluator, WEIGHTS_FILE, adam_moments=adam_dict)

# 3. Recharger
print("\n3. Rechargement...")
evaluator_loaded, adam_moments_loaded = load_evaluator_from_file(WEIGHTS_FILE)

if adam_moments_loaded is None:
    print("❌ ERREUR: Moments Adam non chargés!")
    exit(1)

print("✓ Moments Adam chargés")

# 4. Vérifier que les valeurs sont identiques
print("\n4. Vérification de l'intégrité...")
checks = [
    ('adam_step', adam_step, int(adam_moments_loaded['adam_step'])),
    ('m_w1', m_w1.mean(), adam_moments_loaded['m_w1'].mean()),
    ('v_w1', v_w1.mean(), adam_moments_loaded['v_w1'].mean()),
    ('m_b1', m_b1.mean(), adam_moments_loaded['m_b1'].mean()),
]

all_ok = True
for name, original, loaded in checks:
    if isinstance(original, (int, np.integer)):
        match = original == loaded
    else:
        match = np.allclose(original, loaded)
    
    status = "✓" if match else "❌"
    print(f"  {status} {name}: original={original}, loaded={loaded}")
    if not match:
        all_ok = False

# 5. Nettoyer
os.remove(WEIGHTS_FILE)

print("\n" + "=" * 70)
if all_ok:
    print("✓✓ SUCCÈS: Les moments Adam sont correctement persistés!")
else:
    print("❌❌ ÉCHEC: Les moments Adam ne sont pas correctement persistés")
print("=" * 70)
