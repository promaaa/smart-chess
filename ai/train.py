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
EPOCHS = 10                      # Nombre de fois où l'on parcourt tout le dataset
BATCH_SIZE = 64                    # Batch plus petit pour stabiliser les updates
# Redémarre proprement en supprimant d'anciens poids incompatibles (architecture changée)
RESET_WEIGHTS = False
# Mode debug: overfit sur un très petit lot pour valider l'apprentissage
OVERFIT_TINY = False
OVERFIT_N = 32
# Afficher des stats de debug sur le premier batch (moyennes, écarts-types, RMSE, corrélation)
DEBUG_STATS = False
# En mode overfit, on permet un meilleur flux de gradient et des updates plus fortes
USE_LEAKY_RELU = True if OVERFIT_TINY else False
LEAKY_ALPHA = 0.01
# Optimiseur Adam (pas adaptatifs)
USE_ADAM = True
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Limiter l'évaluation de fin d'époque sur grands jeux de données (None pour désactiver)
EVAL_MAX_SAMPLES = 2000

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
    # Mode overfit tiny: réduire à quelques exemples pour vérifier que la loss descend
    if OVERFIT_TINY:
        n = min(OVERFIT_N, len(fens))
        fens = fens[:n]
        evaluations = evaluations[:n]
        print(f"Mode overfit tiny activé: {n} exemples")
    # Définir le batch_size effectif
    batch_size = min(BATCH_SIZE, len(fens))
    # Biais de sortie initialisé vers la moyenne des labels pour accélérer la convergence
    eval_mean = float(np.mean(evaluations)) if len(evaluations) > 0 else 0.0
    
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
    # Warm-start du biais de sortie pour rapprocher la moyenne des prédictions de celle des cibles
    try:
        evaluator.biases3[0, 0] = eval_mean
    except Exception:
        pass
    # En overfit tiny, augmenter légèrement la sensibilité de la couche de sortie
    if OVERFIT_TINY:
        try:
            evaluator.weights3 *= 1.5
        except Exception:
            pass
    
    # Créer une seule instance de l'échiquier pour la réutiliser
    chess_game = Chess()

    # Moments pour Adam (initialisation)
    if USE_ADAM:
        m_w1 = np.zeros_like(evaluator.weights1); v_w1 = np.zeros_like(evaluator.weights1)
        m_b1 = np.zeros_like(evaluator.biases1);  v_b1 = np.zeros_like(evaluator.biases1)
        m_w2 = np.zeros_like(evaluator.weights2); v_w2 = np.zeros_like(evaluator.weights2)
        m_b2 = np.zeros_like(evaluator.biases2);  v_b2 = np.zeros_like(evaluator.biases2)
        m_w3 = np.zeros_like(evaluator.weights3); v_w3 = np.zeros_like(evaluator.weights3)
        m_b3 = np.zeros_like(evaluator.biases3);  v_b3 = np.zeros_like(evaluator.biases3)
        adam_step = 0

    print("Début de l'entraînement...")
    # Learning rate effectif (plus grand en overfit tiny pour accélérer la convergence)
    lr = LEARNING_RATE * (10.0 if OVERFIT_TINY else 1.0)
    # Accélérer la montée en amplitude des prédictions en overfit
    out_lr = lr * 20.0 if OVERFIT_TINY else lr
    # 3. Boucle d'entraînement principale
    for epoch in range(EPOCHS):
        total_loss = 0
        
        # Mélanger les données (désactivé en overfit tiny pour stabilité et reproductibilité)
        if OVERFIT_TINY:
            fens_shuffled = fens
            evaluations_shuffled = evaluations
        else:
            permutation = np.random.permutation(len(fens))
            fens_shuffled = fens[permutation]
            evaluations_shuffled = evaluations[permutation]
        
        # Utiliser tqdm pour la barre de progression
        progress_bar = tqdm(range(0, len(fens), batch_size), desc=f"Epoch {epoch + 1}/{EPOCHS}")

        # En overfit tiny, répéter plusieurs pas sur le même batch pour accélérer l'ajustement
        repeat_steps = 10 if OVERFIT_TINY else 1
        for i in progress_bar:
            # Créer le mini-batch
            batch_fens = fens_shuffled[i:i + batch_size]
            batch_evals = evaluations_shuffled[i:i + batch_size]
            
            # Initialiser les gradients pour ce batch (2 hidden layers)
            dw1, db1 = np.zeros_like(evaluator.weights1), np.zeros_like(evaluator.biases1)
            dw2, db2 = np.zeros_like(evaluator.weights2), np.zeros_like(evaluator.biases2)
            dw3, db3 = np.zeros_like(evaluator.weights3), np.zeros_like(evaluator.biases3)
            # Debug: collecter les prédictions/cibles du 1er batch de la 1ère époque
            collect_debug = DEBUG_STATS and epoch == 0 and i == 0
            if collect_debug:
                dbg_preds = []
                dbg_targets = []

            # --- Forward Pass & Backpropagation sur le batch ---
            for _step in range(repeat_steps):
                for fen, actual_eval in zip(batch_fens, batch_evals):
                    # Charger la position
                    chess_game.load_fen(fen)
                    
                    # --- A. Forward Pass (Prédiction) ---
                    input_vector = evaluator._encode_board(chess_game).reshape(1, -1)
                    h1_in = np.dot(input_vector, evaluator.weights1) + evaluator.biases1
                    if USE_LEAKY_RELU:
                        h1 = np.where(h1_in > 0, h1_in, LEAKY_ALPHA * h1_in)
                        dh1 = np.where(h1_in > 0, 1.0, LEAKY_ALPHA)
                    else:
                        h1 = np.maximum(0, h1_in)
                        dh1 = (h1_in > 0).astype(h1_in.dtype)

                    h2_in = np.dot(h1, evaluator.weights2) + evaluator.biases2
                    if USE_LEAKY_RELU:
                        h2 = np.where(h2_in > 0, h2_in, LEAKY_ALPHA * h2_in)
                        dh2 = np.where(h2_in > 0, 1.0, LEAKY_ALPHA)
                    else:
                        h2 = np.maximum(0, h2_in)
                        dh2 = (h2_in > 0).astype(h2_in.dtype)
                    final_output = np.dot(h2, evaluator.weights3) + evaluator.biases3
                    predicted_eval = final_output[0][0]
                    if collect_debug:
                        dbg_preds.append(float(predicted_eval))
                        dbg_targets.append(float(actual_eval))

                    # --- B. Calcul de l'erreur (Loss) ---
                    error = predicted_eval - actual_eval
                    total_loss += error**2

                    # --- C. Backpropagation (Calcul des gradients) ---
                    grad_output = 2 * error
                    # Output -> h2
                    dw3 += h2.T * grad_output
                    db3 += grad_output
                    grad_h2 = grad_output * evaluator.weights3.T
                    # Appliquer la dérivée de l'activation
                    grad_h2 *= dh2

                    # h2 -> h1
                    dw2 += h1.T.dot(grad_h2)
                    db2 += grad_h2.flatten()
                    grad_h1 = grad_h2.dot(evaluator.weights2.T)
                    grad_h1 *= dh1

                    # h1 -> input
                    dw1 += input_vector.T.dot(grad_h1)
                    db1 += grad_h1.flatten()

            n_samples = len(batch_fens) * repeat_steps

            # Gradient clipping global (stabilisation)
            CLIP = None if OVERFIT_TINY else 5.0
            # Norme globale des gradients (Frobenius)
            g1 = dw1; g1b = db1
            g2 = dw2; g2b = db2
            g3 = dw3; g3b = db3
            total_norm = 0.0
            for g in [g1, g1b, g2, g2b, g3, g3b]:
                total_norm += float(np.linalg.norm(g / max(n_samples, 1))**2)
            total_norm = np.sqrt(total_norm)
            if CLIP is not None and total_norm > CLIP and total_norm > 0:
                scale = CLIP / total_norm
                dw1 *= scale; db1 *= scale
                dw2 *= scale; db2 *= scale
                dw3 *= scale; db3 *= scale

            # 4. Mettre à jour les poids (après chaque batch)
            if USE_ADAM:
                adam_step += 1
                # Gradients moyens
                g_w1 = dw1 / n_samples; g_b1 = db1 / n_samples
                g_w2 = dw2 / n_samples; g_b2 = db2 / n_samples
                g_w3 = dw3 / n_samples; g_b3 = db3 / n_samples

                # Moments
                m_w1 = ADAM_BETA1 * m_w1 + (1 - ADAM_BETA1) * g_w1
                v_w1 = ADAM_BETA2 * v_w1 + (1 - ADAM_BETA2) * (g_w1 * g_w1)
                m_b1 = ADAM_BETA1 * m_b1 + (1 - ADAM_BETA1) * g_b1
                v_b1 = ADAM_BETA2 * v_b1 + (1 - ADAM_BETA2) * (g_b1 * g_b1)

                m_w2 = ADAM_BETA1 * m_w2 + (1 - ADAM_BETA1) * g_w2
                v_w2 = ADAM_BETA2 * v_w2 + (1 - ADAM_BETA2) * (g_w2 * g_w2)
                m_b2 = ADAM_BETA1 * m_b2 + (1 - ADAM_BETA1) * g_b2
                v_b2 = ADAM_BETA2 * v_b2 + (1 - ADAM_BETA2) * (g_b2 * g_b2)

                m_w3 = ADAM_BETA1 * m_w3 + (1 - ADAM_BETA1) * g_w3
                v_w3 = ADAM_BETA2 * v_w3 + (1 - ADAM_BETA2) * (g_w3 * g_w3)
                m_b3 = ADAM_BETA1 * m_b3 + (1 - ADAM_BETA1) * g_b3
                v_b3 = ADAM_BETA2 * v_b3 + (1 - ADAM_BETA2) * (g_b3 * g_b3)

                # Corrections de biais
                mhat_w1 = m_w1 / (1 - (ADAM_BETA1 ** adam_step)); vhat_w1 = v_w1 / (1 - (ADAM_BETA2 ** adam_step))
                mhat_b1 = m_b1 / (1 - (ADAM_BETA1 ** adam_step)); vhat_b1 = v_b1 / (1 - (ADAM_BETA2 ** adam_step))
                mhat_w2 = m_w2 / (1 - (ADAM_BETA1 ** adam_step)); vhat_w2 = v_w2 / (1 - (ADAM_BETA2 ** adam_step))
                mhat_b2 = m_b2 / (1 - (ADAM_BETA1 ** adam_step)); vhat_b2 = v_b2 / (1 - (ADAM_BETA2 ** adam_step))
                mhat_w3 = m_w3 / (1 - (ADAM_BETA1 ** adam_step)); vhat_w3 = v_w3 / (1 - (ADAM_BETA2 ** adam_step))
                mhat_b3 = m_b3 / (1 - (ADAM_BETA1 ** adam_step)); vhat_b3 = v_b3 / (1 - (ADAM_BETA2 ** adam_step))

                evaluator.weights1 -= lr * mhat_w1 / (np.sqrt(vhat_w1) + ADAM_EPS)
                evaluator.biases1  -= lr * mhat_b1 / (np.sqrt(vhat_b1) + ADAM_EPS)
                evaluator.weights2 -= lr * mhat_w2 / (np.sqrt(vhat_w2) + ADAM_EPS)
                evaluator.biases2  -= lr * mhat_b2 / (np.sqrt(vhat_b2) + ADAM_EPS)
                evaluator.weights3 -= out_lr * mhat_w3 / (np.sqrt(vhat_w3) + ADAM_EPS)
                evaluator.biases3  -= out_lr * mhat_b3 / (np.sqrt(vhat_b3) + ADAM_EPS)
            else:
                evaluator.weights1 -= lr * (dw1 / n_samples)
                evaluator.biases1 -= lr * (db1 / n_samples)
                evaluator.weights2 -= lr * (dw2 / n_samples)
                evaluator.biases2 -= lr * (db2 / n_samples)
                evaluator.weights3 -= out_lr * (dw3 / n_samples)
                evaluator.biases3  -= out_lr * (db3 / n_samples)

            # Stats de debug pour le premier batch
            if collect_debug and len(dbg_preds) > 0:
                preds = np.array(dbg_preds)
                targets = np.array(dbg_targets)
                batch_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
                try:
                    corr = float(np.corrcoef(preds, targets)[0, 1])
                except Exception:
                    corr = float('nan')
                print("\n[DEBUG batch 0]",
                      f"targets mean={targets.mean():.4f} std={targets.std():.4f};",
                      f"preds mean={preds.mean():.4f} std={preds.std():.4f};",
                      f"RMSE={batch_rmse:.4f}; corr={corr:.4f}")

            # Mettre à jour la description de la barre de progression
            avg_loss = np.sqrt(total_loss / ((i // batch_size + 1) * batch_size))
            progress_bar.set_postfix({"loss": f"{avg_loss:.2f}"})

        # Évaluation fin d'époque (échantillon si dataset volumineux et pas en overfit tiny)
        try:
            if not OVERFIT_TINY and EVAL_MAX_SAMPLES is not None and len(fens) > EVAL_MAX_SAMPLES:
                idx = np.random.choice(len(fens), size=EVAL_MAX_SAMPLES, replace=False)
                eval_fens = fens[idx]
                eval_targets = evaluations[idx]
            else:
                eval_fens = fens
                eval_targets = evaluations

            preds_all = []
            targets_all = []
            for fen, target in zip(eval_fens, eval_targets):
                chess_game.load_fen(fen)
                x = evaluator._encode_board(chess_game).reshape(1, -1)
                h1_in = np.dot(x, evaluator.weights1) + evaluator.biases1
                if USE_LEAKY_RELU:
                    h1 = np.where(h1_in > 0, h1_in, LEAKY_ALPHA * h1_in)
                else:
                    h1 = np.maximum(0, h1_in)
                h2_in = np.dot(h1, evaluator.weights2) + evaluator.biases2
                if USE_LEAKY_RELU:
                    h2 = np.where(h2_in > 0, h2_in, LEAKY_ALPHA * h2_in)
                else:
                    h2 = np.maximum(0, h2_in)
                y = np.dot(h2, evaluator.weights3) + evaluator.biases3
                preds_all.append(float(y[0, 0]))
                targets_all.append(float(target))
            preds_all = np.array(preds_all)
            targets_all = np.array(targets_all)
            rmse_all = float(np.sqrt(np.mean((preds_all - targets_all) ** 2)))
            corr_all = float(np.corrcoef(preds_all, targets_all)[0, 1]) if len(preds_all) > 1 else float('nan')
            print(f"[EPOCH {epoch+1}] FULLSET RMSE={rmse_all:.4f}; corr={corr_all:.4f}; preds std={preds_all.std():.4f}; targets std={targets_all.std():.4f}")
        except Exception as e:
            print(f"[EPOCH {epoch+1}] eval error: {e}")

        print(f"\nFin de l'époque {epoch + 1}. Sauvegarde des poids intermédiaires...")
        save_weights(evaluator, WEIGHTS_FILE)

    print("Entraînement terminé.")
    
    # 5. Sauvegarder les poids entraînés
    save_weights(evaluator, WEIGHTS_FILE)

if __name__ == "__main__":
    main()