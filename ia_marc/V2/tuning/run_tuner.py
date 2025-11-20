#!/usr/bin/env python3
"""
Tuner Simple Python - Sans dépendances externes
================================================

Implémente le Texel Tuning directement en Python.
Plus simple que d'installer un tuner externe.

Basé sur l'algorithme de Texel (2010).
"""

import json
import chess
import numpy as np
from pathlib import Path

# Constantes
SIGMOID_K = 1.2  # Facteur de scaling pour sigmoid


def sigmoid(score_cp, k=SIGMOID_K):
    """
    Convertit un score centipawn en probabilité de victoire.
    
    Args:
        score_cp: Score en centipawns
        k: Scaling factor
        
    Returns:
        Probabilité entre 0 et 1
    """
    return 1.0 / (1.0 + 10 ** (-k * score_cp / 400.0))


def extract_simple_features(fen):
    """
    Extrait des features simples d'une position.
    On va optimiser principalement les valeurs des pièces.
    
    Returns:
        Dict {feature_name: feature_value}
    """
    board = chess.Board(fen)
    features = {}
    
    # Compter les pièces de chaque type pour chaque camp
    piece_types = {
        chess.PAWN: 'pawn',
        chess.KNIGHT: 'knight',
        chess.BISHOP: 'bishop',
        chess.ROOK: 'rook',
        chess.QUEEN: 'queen'
    }
    
    for piece_type, name in piece_types.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        features[f'{name}_diff'] = white_count - black_count
    
    # Feature pour le tempo (qui a le trait)
    features['tempo'] = 1 if board.turn == chess.WHITE else -1
    
    return features


def evaluate_position(features, weights):
    """
    Évalue une position avec les poids donnés.
    
    Args:
        features: Dict des features
        weights: Dict des poids
        
    Returns:
        Score en centipawns
    """
    score = 0
    for feature, value in features.items():
        if feature in weights:
            score += value * weights[feature]
    return score


def compute_error(dataset, weights):
    """
    Calcule l'erreur MSE sur tout le dataset.
    
    Args:
        dataset: Liste de (features, actual_result)
        weights: Poids actuels
        
    Returns:
        Mean Squared Error
    """
    total_error = 0.0
    
    for features, actual_result in dataset:
        eval_score = evaluate_position(features, weights)
        predicted = sigmoid(eval_score)
        error = (predicted - actual_result) ** 2
        total_error += error
    
    return total_error / len(dataset)


def tune_weights(dataset, initial_weights, learning_rate=0.001, epochs=200, verbose=True):
    """
    Optimise les poids par descente de gradient.
    
    Args:
        dataset: Dataset de positions (features, résultat)
        initial_weights: Poids initiaux
        learning_rate: Taux d'apprentissage
        epochs: Nombre d'itérations
        verbose: Afficher la progression
        
    Returns:
        Poids optimisés
    """
    weights = initial_weights.copy()
    best_error = float('inf')
    best_weights = weights.copy()
    
    if verbose:
        print(f"\n{'='*60}")
        print("OPTIMISATION DES POIDS (Texel Tuning)")
        print(f"{'='*60}")
        print(f"Dataset: {len(dataset)} positions")
        print(f"Learning rate: {learning_rate}")
        print(f"Epochs: {epochs}\n")
    
    initial_error = compute_error(dataset, weights)
    if verbose:
        print(f"Erreur initiale: {initial_error:.6f}\n")
    
    for epoch in range(epochs):
        # Calcul des gradients
        gradients = {k: 0.0 for k in weights}
        
        for features, actual_result in dataset:
            eval_score = evaluate_position(features, weights)
            predicted = sigmoid(eval_score)
            
            # Gradient: d(MSE)/d(weight)
            error_term = predicted - actual_result
            sigmoid_derivative = predicted * (1 - predicted) * np.log(10) * SIGMOID_K / 400.0
            
            for feature, value in features.items():
                if feature in weights:
                    gradients[feature] += 2 * error_term * sigmoid_derivative * value
        
        # Moyenner les gradients
        for feature in gradients:
            gradients[feature] /= len(dataset)
        
        # Mise à jour des poids
        for feature in weights:
            weights[feature] -= learning_rate * gradients[feature]
        
        # Calculer l'erreur
        current_error = compute_error(dataset, weights)
        
        # Sauvegarder le meilleur
        if current_error < best_error:
            best_error = current_error
            best_weights = weights.copy()
        
        # Afficher la progression
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | MSE: {current_error:.6f} | Best: {best_error:.6f}")
    
    if verbose:
        print(f"\n✅ Optimisation terminée !")
        print(f"Erreur finale: {best_error:.6f}")
        print(f"Amélioration: {(1 - best_error/initial_error)*100:.2f}%\n")
    
    return best_weights


def load_dataset_from_epd(epd_file):
    """
    Charge le dataset depuis un fichier EPD.
    
    Returns:
        Liste de (features, result)
    """
    print(f"Chargement du dataset: {epd_file}")
    
    dataset = []
    with open(epd_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            # Parser EPD: "fen c0 "result";" ou "fen c9 "result";"
            parts = None
            if ' c0 ' in line:
                parts = line.split(' c0 ')
            elif ' c9 ' in line:
                parts = line.split(' c9 ')
            
            if not parts or len(parts) != 2:
                continue
            
            fen = parts[0].strip()
            result_str = parts[1].strip().strip('";')
            
            # Convertir le résultat
            if result_str in ['1-0', '1.0', '1']:
                result = 1.0
            elif result_str in ['0-1', '0.0', '0']:
                result = 0.0
            elif result_str in ['1/2-1/2', '0.5']:
                result = 0.5
            else:
                try:
                    result = float(result_str)
                except ValueError: # Changed from generic 'except' to 'except ValueError' for better practice
                    continue
            
            # Extraire features
            try:
                features = extract_simple_features(fen)
                dataset.append((features, result))
            except Exception: # Catch any exception during feature extraction
                continue
    
    print(f"✅ Dataset chargé: {len(dataset)} positions\n")
    return dataset


def main():
    """Workflow principal du tuning."""
    
    # Vérifier que le dataset existe
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    epd_file = os.path.join(script_dir, 'dataset.epd')
    
    if not os.path.exists(epd_file):
        print("❌ Erreur: dataset.epd non trouvé !")
        print(f"Cherché dans: {epd_file}")
        print("Lancez d'abord: python generate_dataset_quick.py")
        return
    
    # Charger le dataset
    dataset = load_dataset_from_epd(epd_file)
    
    # Poids initiaux (valeurs actuelles de PeSTO en MG)
    initial_weights = {
        'pawn_diff': 82,
        'knight_diff': 337,
        'bishop_diff': 365,
        'rook_diff': 477,
        'queen_diff': 1025,
        'tempo': 10,  # Bonus au trait
    }
    
    print("Poids initiaux:")
    for feature, value in initial_weights.items():
        print(f"  {feature:15s}: {value:6.1f}")
    
    # Tuning
    optimized_weights = tune_weights(
        dataset,
        initial_weights,
        learning_rate=0.001,
        epochs=200,
        verbose=True
    )
    
    # Afficher les résultats
    print(f"{'='*60}")
    print("POIDS OPTIMISÉS")
    print(f"{'='*60}")
    
    for feature in initial_weights.keys():
        old = initial_weights[feature]
        new = optimized_weights[feature]
        diff = new - old
        sign = '+' if diff >= 0 else ''
        print(f"{feature:15s}: {old:6.1f} → {new:6.1f}  ({sign}{diff:+6.1f})")
    
    # Sauvegarder
    output_file = Path('optimized_weights.json')
    with open(output_file, 'w') as f:
        json.dump(optimized_weights, f, indent=2)
    
    print(f"\n✅ Poids sauvegardés: {output_file}")
    
    print("\n" + "="*60)
    print("PROCHAINE ÉTAPE")
    print("="*60)
    print("Appliquer les poids dans engine_brain.py:")
    print(f"  MG_VALS = [")
    print(f"    {optimized_weights['pawn_diff']:.0f},   # Pawn")
    print(f"    {optimized_weights['knight_diff']:.0f},  # Knight")
    print(f"    {optimized_weights['bishop_diff']:.0f},  # Bishop")
    print(f"    {optimized_weights['rook_diff']:.0f},   # Rook")
    print(f"    {optimized_weights['queen_diff']:.0f},  # Queen")
    print(f"    0      # King")
    print(f"  ]")


if __name__ == "__main__":
    main()
