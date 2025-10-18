import pandas as pd
import time
import numpy as np
import psutil
import platform

print("="*70)
print("DIAGNOSTIC DE PERFORMANCE")
print("="*70)

# 1. Info système
print("\n1. Configuration matérielle:")
print(f"  CPU: {platform.processor()}")
print(f"  Cores physiques: {psutil.cpu_count(logical=False)}")
print(f"  Cores logiques: {psutil.cpu_count(logical=True)}")
print(f"  RAM totale: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"  RAM disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB")

# 2. Taille du dataset
print("\n2. Dataset:")
try:
    df = pd.read_csv(
        r'C:\Users\gauti\OneDrive\Documents\UE commande\chessData.csv',
        names=['FEN', 'Evaluation'],
        skiprows=1,
        comment='#'
    )
    df.dropna(inplace=True)
    n_positions = len(df)
    print(f"  Positions totales: {n_positions:,}")
    
    # Estimation du temps
    batch_size = 64
    n_batches = n_positions // batch_size
    print(f"  Batches par epoch: {n_batches:,}")
    
except Exception as e:
    print(f"  Erreur: {e}")
    n_positions = 0
    n_batches = 0

# 3. Test de vitesse forward pass
print("\n3. Benchmark forward pass:")
try:
    from nn_evaluator import NeuralNetworkEvaluator
    from Chess import Chess
    
    evaluator = NeuralNetworkEvaluator.create_untrained_network()
    chess = Chess()
    chess.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Warmup
    for _ in range(10):
        x = evaluator._encode_board(chess).reshape(1, -1)
        h1 = np.maximum(0, np.dot(x, evaluator.weights1) + evaluator.biases1)
        h2 = np.maximum(0, np.dot(h1, evaluator.weights2) + evaluator.biases2)
        y = np.dot(h2, evaluator.weights3) + evaluator.biases3
    
    # Benchmark
    n_iter = 1000
    start = time.time()
    for _ in range(n_iter):
        x = evaluator._encode_board(chess).reshape(1, -1)
        h1 = np.maximum(0, np.dot(x, evaluator.weights1) + evaluator.biases1)
        h2 = np.maximum(0, np.dot(h1, evaluator.weights2) + evaluator.biases2)
        y = np.dot(h2, evaluator.weights3) + evaluator.biases3
    
    elapsed = time.time() - start
    per_position = elapsed / n_iter * 1000  # en ms
    
    print(f"  {n_iter} forward pass en {elapsed:.2f}s")
    print(f"  Vitesse: {per_position:.3f} ms/position")
    print(f"  Throughput: {n_iter/elapsed:.0f} positions/sec")
    
    # Estimation temps epoch
    if n_positions > 0:
        # Forward + backward = ~3x le temps du forward seul
        time_per_position_train = per_position * 3 / 1000  # en secondes
        estimated_epoch_time = time_per_position_train * n_positions / 60  # en minutes
        
        print(f"\n4. Estimation temps d'entraînement:")
        print(f"  Temps estimé par epoch: {estimated_epoch_time:.1f} minutes")
        print(f"  Ton temps actuel: 240 minutes")
        print(f"  Ratio: {240 / estimated_epoch_time:.1f}x trop lent")
        
        if estimated_epoch_time < 10:
            print(f"  ✓ Performance attendue: BONNE")
        elif estimated_epoch_time < 30:
            print(f"  → Performance attendue: ACCEPTABLE")
        else:
            print(f"  ⚠ Performance attendue: LENTE (dataset énorme)")
            
except Exception as e:
    print(f"  Erreur: {e}")

# 5. Vérifier si NumPy utilise BLAS optimisé
print("\n5. Configuration NumPy:")
try:
    import numpy as np
    config = np.__config__.show()
    print("  (voir sortie ci-dessus pour détails BLAS/LAPACK)")
except:
    print("  Impossible de vérifier")

print("\n" + "="*70)
print("RECOMMANDATIONS:")
print("="*70)

if n_positions > 0:
    if n_positions < 50000:
        print("✓ Dataset raisonnable pour CPU")
    elif n_positions < 200000:
        print("→ Dataset moyen, envisager GPU si disponible")
    else:
        print("⚠ Dataset volumineux, GPU fortement recommandé")

print("\nPour accélérer l'entraînement:")
print("1. Vectoriser les batchs (traiter 64 positions en une fois)")
print("2. Utiliser un GPU si disponible (50-100x plus rapide)")
print("3. Réduire temporairement le dataset pour itérer plus vite")
print("4. Vérifier que NumPy utilise un BLAS optimisé (MKL, OpenBLAS)")
