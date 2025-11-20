"""
Phase 5 : Texel Tuning - Guide d'Implémentation
================================================

Le Texel Tuning optimise automatiquement les poids de la fonction d'évaluation
en utilisant un grand dataset de positions avec résultats connus.

Gain estimé : +50-100 ELO
Temps requis : 10-20 heures (incluant génération de dataset)
"""

# ============================================================================
# ÉTAPE 1 : GÉNÉRATION DU DATASET
# ============================================================================

"""
Objectif : Collecter 10,000+ positions depuis des parties jouées

Méthode 1 : Self-play (Recommandé pour IA-Marc)
-----------------------------------------------
Faire jouer l'IA contre elle-même et extraire des positions.

Script : tuning/generate_dataset_selfplay.py
"""

import chess
import chess.engine
import json
import random
from pathlib import Path

def generate_selfplay_dataset(num_games=100, max_positions_per_game=50):
    """
    Génère un dataset depuis self-play.
    
    Args:
        num_games: Nombre de parties à jouer
        max_positions_per_game: Positions max par partie
        
    Returns:
        Liste de (FEN, résultat) où résultat = 1.0 (blanc gagne), 
        0.5 (nulle), 0.0 (noir gagne)
    """
    from engine_main import ChessEngine
    
    dataset = []
    engine = ChessEngine()
    engine.set_level("Club")  # Niveau moyen pour variété
    
    print(f"Génération de {num_games} parties...")
    
    for game_num in range(num_games):
        board = chess.Board()
        positions = []
        
        # Jouer une partie complète
        moves_count = 0
        while not board.is_game_over() and moves_count < 200:
            try:
                move = engine.get_move(board, time_limit=0.5)
                if move is None:
                    break
                
                # Sauvegarder la position avant le coup (tous les 2 coups)
                if moves_count % 2 == 0 and moves_count > 10:  # Skip opening
                    positions.append(board.fen())
                
                board.push(move)
                moves_count += 1
            except Exception as e:
                print(f"Erreur game {game_num}: {e}")
                break
        
        # Déterminer le résultat
        result = board.result()
        if result == "1-0":
            game_result = 1.0
        elif result == "0-1":
            game_result = 0.0
        else:
            game_result = 0.5
        
        # Échantillonner des positions
        sampled = random.sample(positions, min(max_positions_per_game, len(positions)))
        
        for fen in sampled:
            dataset.append({
                'fen': fen,
                'result': game_result,
                'game_id': game_num
            })
        
        if (game_num + 1) % 10 == 0:
            print(f"  Parties complétées: {game_num + 1}/{num_games} - Positions: {len(dataset)}")
    
    return dataset


"""
Méthode 2 : Utiliser un dataset public (Plus rapide)
-----------------------------------------------------
Télécharger des datasets existants depuis :
- Lichess Database : https://database.lichess.org/
- Zurichess quiet-labeled dataset
- CCRL datasets

Format attendu : FEN + résultat (1/0.5/0)
"""


# ============================================================================
# ÉTAPE 2 : EXTRACTION DES FEATURES
# ============================================================================

"""
Objectif : Pour chaque position, extraire les features que nous voulons optimiser

Features typiques à optimiser :
- Valeurs des pièces (Pion, Cavalier, Fou, Tour, Dame)
- PST (Piece-Square Tables) par phase (MG/EG)
- Poids de mobilité par pièce
- Bonus/malus structure de pions
- Poids de sécurité du roi
"""

def extract_features(board_fen):
    """
    Extrait les features d'une position.
    
    Returns:
        Dict de features avec noms et valeurs
    """
    board = chess.Board(board_fen)
    features = {}
    
    # Exemple : compter les pièces de chaque type
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        for piece_type in range(1, 7):  # PAWN à KING
            count = len(board.pieces(piece_type, color))
            features[f'{piece_type}_{color}'] = count * sign
    
    # Extraire mobilité, structure pions, etc.
    # ...
    
    return features


# ============================================================================
# ÉTAPE 3 : TUNING (Descente de Gradient)
# ============================================================================

"""
Algorithme de Texel :
1. Calculer la "prédiction" de l'évaluation pour chaque position
2. Comparer avec le résultat réel de la partie
3. Ajuster les poids pour minimiser l'erreur (Mean Squared Error)
4. Répéter jusqu'à convergence

Formule :
    eval_pred = sigmoid(eval(position))
    error = sum((eval_pred - actual_result)^2)
    
On minimise l'erreur en ajustant les poids.
"""

import numpy as np

def sigmoid(x, k=1.2):
    """
    Convertit un score centipawn en probabilité de gagner.
    
    Args:
        x: Score en centipawns
        k: Facteur de scaling (typiquement 1.0-2.0)
        
    Returns:
        Probabilité entre 0 et 1
    """
    return 1.0 / (1.0 + 10**(-k * x / 400))


def compute_mse(dataset, weights):
    """
    Calcule l'erreur MSE sur tout le dataset.
    
    Args:
        dataset: Liste de (features, result)
        weights: Poids actuels
        
    Returns:
        Mean Squared Error
    """
    total_error = 0.0
    
    for features, actual_result in dataset:
        # Calculer l'évaluation avec les poids actuels
        eval_score = sum(features[f] * weights[f] for f in features)
        
        # Convertir en probabilité
        predicted = sigmoid(eval_score)
        
        # Erreur quadratique
        error = (predicted - actual_result) ** 2
        total_error += error
    
    return total_error / len(dataset)


def tune_weights(dataset, initial_weights, learning_rate=0.01, epochs=100):
    """
    Optimise les poids par descente de gradient.
    
    Args:
        dataset: Dataset de positions
        initial_weights: Poids initiaux
        learning_rate: Taux d'apprentissage
        epochs: Nombre d'itérations
        
    Returns:
        Poids optimisés
    """
    weights = initial_weights.copy()
    
    print(f"Démarrage du tuning sur {len(dataset)} positions...")
    print(f"MSE initial: {compute_mse(dataset, weights):.6f}")
    
    for epoch in range(epochs):
        # Calcul du gradient
        gradients = {k: 0.0 for k in weights}
        
        for features, actual_result in dataset:
            eval_score = sum(features[f] * weights[f] for f in features)
            predicted = sigmoid(eval_score)
            
            # Gradient de l'erreur par rapport à eval_score
            error = predicted - actual_result
            dsigmoid = predicted * (1 - predicted) * np.log(10) / 400
            
            # Gradient pour chaque poids
            for f in features:
                gradients[f] += 2 * error * dsigmoid * features[f]
        
        # Moyenner les gradients
        for f in gradients:
            gradients[f] /= len(dataset)
        
        # Mise à jour des poids
        for f in weights:
            weights[f] -= learning_rate * gradients[f]
        
        # Afficher progression
        if (epoch + 1) % 10 == 0:
            mse = compute_mse(dataset, weights)
            print(f"Epoch {epoch + 1}/{epochs} - MSE: {mse:.6f}")
    
    print(f"MSE final: {compute_mse(dataset, weights):.6f}")
    return weights


# ============================================================================
# ÉTAPE 4 : APPLIQUER LES POIDS OPTIMISÉS
# ============================================================================

"""
Une fois les poids optimisés, les appliquer dans engine_brain.py :

1. Sauvegarder les nouveaux poids dans un fichier JSON
2. Modifier les constantes dans engine_brain.py
3. Tester la force du moteur (jouer contre version précédente)
"""

def save_tuned_weights(weights, filepath='tuning/tuned_weights.json'):
    """Sauvegarde les poids optimisés."""
    with open(filepath, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"Poids sauvegardés dans {filepath}")


def apply_weights_to_engine(weights_file):
    """
    Instructions pour appliquer les poids dans engine_brain.py
    """
    print("""
    Pour appliquer les poids optimisés :
    
    1. Ouvrir engine_brain.py
    
    2. Modifier les constantes :
       MG_VALS = [82, 337, 365, 477, 1025, 0]  # Anciennes valeurs
       MG_VALS = [85, 340, 368, 480, 1030, 0]  # Nouvelles valeurs optimisées
       
    3. Modifier les PST si optimisés
    
    4. Modifier les poids de mobilité, structure pions, etc.
    
    5. Tester avec :
       python ai_comparison/compare_ais.py
       
    6. Si meilleur, commit les changements !
    """)


# ============================================================================
# SCRIPT COMPLET DE TUNING
# ============================================================================

def main_tuning_workflow():
    """
    Workflow complet du tuning.
    """
    print("="*60)
    print("TEXEL TUNING - IA-MARC V2")
    print("="*60)
    
    # ÉTAPE 1 : Génération du dataset
    print("\n[1/4] Génération du dataset...")
    dataset_raw = generate_selfplay_dataset(num_games=100)
    
    # Sauvegarder le dataset
    Path('tuning').mkdir(exist_ok=True)
    with open('tuning/dataset.json', 'w') as f:
        json.dump(dataset_raw, f, indent=2)
    print(f"Dataset sauvegardé: {len(dataset_raw)} positions")
    
    # ÉTAPE 2 : Extraction des features
    print("\n[2/4] Extraction des features...")
    dataset_features = []
    for entry in dataset_raw:
        features = extract_features(entry['fen'])
        dataset_features.append((features, entry['result']))
    
    # ÉTAPE 3 : Tuning
    print("\n[3/4] Optimisation des poids...")
    initial_weights = {
        # Valeurs des pièces initiales
        'pawn_value': 100,
        'knight_value': 320,
        'bishop_value': 330,
        'rook_value': 500,
        'queen_value': 900,
        # Ajouter tous les poids à optimiser
    }
    
    tuned_weights = tune_weights(
        dataset_features, 
        initial_weights, 
        learning_rate=0.01,
        epochs=100
    )
    
    # ÉTAPE 4 : Sauvegarde
    print("\n[4/4] Sauvegarde des résultats...")
    save_tuned_weights(tuned_weights)
    
    print("\n" + "="*60)
    print("TUNING TERMINÉ !")
    print("="*60)
    apply_weights_to_engine('tuning/tuned_weights.json')


# ============================================================================
# ALTERNATIVE : UTILISER UN TUNER EXISTANT
# ============================================================================

"""
Pour gagner du temps, utiliser un tuner existant :

1. Gedas' Texel Tuner (Recommandé)
   https://github.com/GediminasMasaitis/texel-tuner
   - Support Python
   - Facile à utiliser
   - Bien documenté
   
   Usage :
   ```
   git clone https://github.com/GediminasMasaitis/texel-tuner
   cd texel-tuner
   python tune.py --input dataset.epd --output tuned.txt
   ```

2. Chess Tuning Tools
   https://github.com/official-stockfish/tuning-tools
   - Outil officiel de Stockfish
   - Très performant
   - Plus complexe

3. Utiliser Stockfish pour générer le dataset
   - Jouer des parties avec Stockfish
   - Extraire les positions avec annotations
   - Tuner sur ces positions
"""


if __name__ == "__main__":
    print(__doc__)
    print("\nPour lancer le tuning complet :")
    print("  python tuning_guide.py --full")
    print("\nPour générer seulement le dataset :")
    print("  python tuning_guide.py --dataset")
    print("\nPour plus d'infos, voir les commentaires dans ce fichier.")
