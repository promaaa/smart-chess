import numpy as np
import pandas as pd
from tqdm import tqdm  # Pour une belle barre de progression

# Assurez-vous que ces fichiers sont dans le même dossier
from Chess import Chess
from nn_evaluator import NeuralNetworkEvaluator, save_weights

# --- CONFIGURATION DE L'ENTRAÎNEMENT ---
DATASET_PATH = "dataset.csv"       # Chemin vers votre fichier de données
WEIGHTS_FILE = "chess_nn_weights.npz" # Fichier où les poids entraînés seront sauvegardés
LEARNING_RATE = 0.001              # À quelle vitesse le réseau apprend (petit = plus stable)
EPOCHS = 10                        # Nombre de fois où l'on parcourt tout le dataset
BATCH_SIZE = 256                   # Nombre de positions traitées avant de mettre à jour les poids

def load_data(filepath: str):
    """Charge le dataset FEN,Evaluation depuis un fichier CSV."""
    print(f"Chargement du dataset depuis {filepath}...")
    df = pd.read_csv(filepath)
    
    # Extraire les colonnes FEN et Evaluation
    fens = df['FEN,Evaluation'].apply(lambda x: ' '.join(x.split(' ')[:-1])).values
    evaluations = df['FEN,Evaluation'].apply(lambda x: int(x.split(' ')[-1])).values
    
    print(f"{len(fens)} positions chargées.")
    return fens, evaluations

def main():
    """Fonction principale pour l'entraînement."""
    # 1. Charger les données
    fens, evaluations = load_data(DATASET_PATH)
    
    # 2. Initialiser le réseau de neurones
    # Crée un réseau avec des poids aléatoires, prêt à être entraîné
    evaluator = NeuralNetworkEvaluator.create_untrained_network()
    
    # Créer une seule instance de l'échiquier pour la réutiliser (plus efficace)
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
            
            # Initialiser les gradients pour ce batch
            dw1, db1 = np.zeros_like(evaluator.weights1), np.zeros_like(evaluator.biases1)
            dw2, db2 = np.zeros_like(evaluator.weights2), np.zeros_like(evaluator.biases2)

            # --- Forward Pass & Backpropagation sur le batch ---
            for fen, actual_eval in zip(batch_fens, batch_evals):
                # Charger la position
                chess_game.load_fen(fen)
                
                # --- A. Forward Pass (Prédiction) ---
                input_vector = evaluator._encode_board(chess_game).reshape(1, -1)
                hidden_input = np.dot(input_vector, evaluator.weights1) + evaluator.biases1
                hidden_output = np.maximum(0, hidden_input) # ReLU
                final_output = np.dot(hidden_output, evaluator.weights2) + evaluator.biases2
                predicted_eval = final_output[0][0] * 100 # Multiplier par 100 comme dans la fonction d'éval
                
                # --- B. Calcul de l'erreur (Loss) ---
                error = predicted_eval - actual_eval
                total_loss += error**2
                
                # --- C. Backpropagation (Calcul des gradients) ---
                # Gradient pour la couche de sortie
                grad_output = 2 * error / 100 # Diviser par 100 pour annuler la mise à l'échelle
                
                # Gradients pour W2 et B2
                dw2 += hidden_output.T * grad_output
                db2 += grad_output
                
                # Gradient propagé à la couche cachée
                grad_hidden_output = grad_output * evaluator.weights2.T
                
                # Gradient à travers l'activation ReLU
                grad_hidden_input = grad_hidden_output * (hidden_input > 0)
                
                # Gradients pour W1 et B1
                dw1 += input_vector.T.dot(grad_hidden_input)
                db1 += grad_hidden_input.flatten()

            # 4. Mettre à jour les poids (après chaque batch)
            n_samples = len(batch_fens)
            evaluator.weights1 -= LEARNING_RATE * (dw1 / n_samples)
            evaluator.biases1 -= LEARNING_RATE * (db1 / n_samples)
            evaluator.weights2 -= LEARNING_RATE * (dw2 / n_samples)
            evaluator.biases2 -= LEARNING_RATE * (db2 / n_samples)

            # Mettre à jour la description de la barre de progression
            avg_loss = np.sqrt(total_loss / ((i // BATCH_SIZE + 1) * BATCH_SIZE))
            progress_bar.set_postfix({"loss": f"{avg_loss:.2f}"})

    print("Entraînement terminé.")
    
    # 5. Sauvegarder les poids entraînés
    save_weights(evaluator, WEIGHTS_FILE)

if __name__ == "__main__":
    main()