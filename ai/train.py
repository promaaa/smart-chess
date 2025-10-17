import numpy as np
import pandas as pd
from tqdm import tqdm  # Pour une belle barre de progression

# Assurez-vous que ces fichiers sont dans le même dossier
from Chess import Chess
import os
from nn_evaluator import NeuralNetworkEvaluator, save_weights, load_evaluator_from_file

# --- CONFIGURATION DE L'ENTRAÎNEMENT ---
DATASET_PATH = "C:\\Users\\gauti\\OneDrive\\Documents\\UE commande\\chessData.csv"       # Chemin vers votre fichier de données
WEIGHTS_FILE = "chess_nn_weights.npz" # Fichier où les poids entraînés seront sauvegardés
LEARNING_RATE = 0.0005            # Learning rate plus stable pour 2 couches
EPOCHS = 10                        # Nombre de fois où l'on parcourt tout le dataset
BATCH_SIZE = 64                    # Batch plus petit pour stabiliser les updates
# Redémarre proprement en supprimant d'anciens poids incompatibles (architecture changée)
RESET_WEIGHTS = True

def load_data(filepath: str):
    """Charge le dataset FEN,Evaluation et le nettoie."""
    print(f"Chargement du dataset depuis {filepath}...")
    
    df = pd.read_csv(
        filepath, 
        names=['FEN', 'Evaluation'], 
        skiprows=1,
        comment='#'
    )
    
    # --- AJOUT DE L'ÉTAPE DE NETTOYAGE ---
    initial_count = len(df)
    # Supprime toutes les lignes où au moins une valeur est manquante (NaN)
    df.dropna(inplace=True) 
    
    cleaned_count = len(df)
    if initial_count > cleaned_count:
        print(f"Nettoyage : {initial_count - cleaned_count} lignes corrompues (avec des valeurs manquantes) ont été supprimées.")
    # ------------------------------------

    # Le reste du code fonctionnera maintenant car il n'y a plus de NaN
    fens = df['FEN'].values

    # Normaliser les évaluations pour stabiliser l'entraînement
    EVAL_SCALE_FACTOR = 1000.0
    evaluations = (df['Evaluation'].astype(int).values) / EVAL_SCALE_FACTOR
    
    print(f"{len(fens)} positions valides chargées.")
    return fens, evaluations

def main():
    # 1. Charger les données
    fens, evaluations = load_data(DATASET_PATH)
    
    # 2. Mélanger le jeu de données de manière globale avant de commencer
    # Cela garantit que même les exécutions courtes utilisent des données variées.
    print("Mélange initial du jeu de données...")
    permutation = np.random.permutation(len(fens))
    fens = fens[permutation]
    evaluations = evaluations[permutation]
    # ---------------------------
    
    # 3. Initialiser le réseau de neurones
    if RESET_WEIGHTS and os.path.exists(WEIGHTS_FILE):
        print(f"Suppression des anciens poids incompatibles: {WEIGHTS_FILE}")
        try:
            os.remove(WEIGHTS_FILE)
        except Exception as e:
            print(f"Impossible de supprimer {WEIGHTS_FILE}: {e}")

    if os.path.exists(WEIGHTS_FILE):
        print(f"Chargement des poids existants depuis {WEIGHTS_FILE} pour continuer l'entraînement...")
        evaluator = load_evaluator_from_file(WEIGHTS_FILE)
    else:
        print("Création d'un nouveau réseau (poids aléatoires)...")
        evaluator = NeuralNetworkEvaluator.create_untrained_network()
    
    # Créer une seule instance de l'échiquier pour la réutiliser
    chess_game = Chess()

    print("Début de l'entraînement...")
    # 3. Boucle d'entraînement principale
    for epoch in range(EPOCHS):
        total_loss = 0
        
        # Mélanger les données à chaque époque pour éviter que le réseau apprenne l'ordre
        permutation = np.random.permutation(len(fens))
        fens_shuffled = fens[permutation]
        evaluations_shuffled = evaluations[permutation]
        
        # Utiliser tqdm pour la barre de progression
        progress_bar = tqdm(range(0, len(fens), BATCH_SIZE), desc=f"Epoch {epoch + 1}/{EPOCHS}")
        
        for i in progress_bar:
            # Créer le mini-batch
            batch_fens = fens_shuffled[i:i + BATCH_SIZE]
            batch_evals = evaluations_shuffled[i:i + BATCH_SIZE]
            
            # Initialiser les gradients pour ce batch (2 hidden layers)
            dw1, db1 = np.zeros_like(evaluator.weights1), np.zeros_like(evaluator.biases1)
            dw2, db2 = np.zeros_like(evaluator.weights2), np.zeros_like(evaluator.biases2)
            dw3, db3 = np.zeros_like(evaluator.weights3), np.zeros_like(evaluator.biases3)

            # --- Forward Pass & Backpropagation sur le batch ---
            for fen, actual_eval in zip(batch_fens, batch_evals):
                # Charger la position
                chess_game.load_fen(fen)
                
                # --- A. Forward Pass (Prédiction) ---
                input_vector = evaluator._encode_board(chess_game).reshape(1, -1)
                h1_in = np.dot(input_vector, evaluator.weights1) + evaluator.biases1
                h1 = np.maximum(0, h1_in)
                h2_in = np.dot(h1, evaluator.weights2) + evaluator.biases2
                h2 = np.maximum(0, h2_in)
                final_output = np.dot(h2, evaluator.weights3) + evaluator.biases3
                predicted_eval = final_output[0][0]

                # --- B. Calcul de l'erreur (Loss) ---
                error = predicted_eval - actual_eval
                total_loss += error**2

                # --- C. Backpropagation (Calcul des gradients) ---
                grad_output = 2 * error
                # Output -> h2
                dw3 += h2.T * grad_output
                db3 += grad_output
                grad_h2 = grad_output * evaluator.weights3.T
                grad_h2[h2_in <= 0] = 0

                # h2 -> h1
                dw2 += h1.T.dot(grad_h2)
                db2 += grad_h2.flatten()
                grad_h1 = grad_h2.dot(evaluator.weights2.T)
                grad_h1[h1_in <= 0] = 0

                # h1 -> input
                dw1 += input_vector.T.dot(grad_h1)
                db1 += grad_h1.flatten()

            n_samples = len(batch_fens)

            # Gradient clipping global (stabilisation)
            CLIP = 5.0
            # Norme globale des gradients (Frobenius)
            g1 = dw1; g1b = db1
            g2 = dw2; g2b = db2
            g3 = dw3; g3b = db3
            total_norm = 0.0
            for g in [g1, g1b, g2, g2b, g3, g3b]:
                total_norm += float(np.linalg.norm(g / max(n_samples, 1))**2)
            total_norm = np.sqrt(total_norm)
            if total_norm > CLIP and total_norm > 0:
                scale = CLIP / total_norm
                dw1 *= scale; db1 *= scale
                dw2 *= scale; db2 *= scale
                dw3 *= scale; db3 *= scale

            # 4. Mettre à jour les poids (après chaque batch)
            evaluator.weights1 -= LEARNING_RATE * (dw1 / n_samples)
            evaluator.biases1 -= LEARNING_RATE * (db1 / n_samples)
            evaluator.weights2 -= LEARNING_RATE * (dw2 / n_samples)
            evaluator.biases2 -= LEARNING_RATE * (db2 / n_samples)
            evaluator.weights3 -= LEARNING_RATE * (dw3 / n_samples)
            evaluator.biases3 -= LEARNING_RATE * (db3 / n_samples)
            evaluator.weights3 -= LEARNING_RATE * (dw3 / n_samples)
            evaluator.biases3 -= LEARNING_RATE * (db3 / n_samples)

            # Mettre à jour la description de la barre de progression
            avg_loss = np.sqrt(total_loss / ((i // BATCH_SIZE + 1) * BATCH_SIZE))
            progress_bar.set_postfix({"loss": f"{avg_loss:.2f}"})

        print(f"\nFin de l'époque {epoch + 1}. Sauvegarde des poids intermédiaires...")
        save_weights(evaluator, WEIGHTS_FILE)

    print("Entraînement terminé.")
    
    # 5. Sauvegarder les poids entraînés
    save_weights(evaluator, WEIGHTS_FILE)

if __name__ == "__main__":
    main()