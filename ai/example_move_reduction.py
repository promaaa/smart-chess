#!/usr/bin/env python3
"""
Exemple simple d'utilisation de la réduction sélective des coups.
"""

import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Chess import Chess
from evaluator import ChessEvaluator
from iterative_deepening_engine_TT_rdcut import IterativeDeepeningAlphaBeta

def example_usage():
    """Exemple d'utilisation avec différentes configurations"""
    
    print("=== EXEMPLE D'UTILISATION DE LA RÉDUCTION DES COUPS ===\n")
    
    # Créer une position de test
    chess = Chess()
    evaluator = ChessEvaluator()
    
    # 1. Configuration standard avec réduction
    print("1. Configuration avec réduction exponentielle (par défaut)")
    print("-" * 50)
    
    engine = IterativeDeepeningAlphaBeta(
        max_time=10,
        max_depth=4,
        evaluator=evaluator,
        move_reduction_enabled=True,  # Réduction activée
        reduction_seed=42  # Graine fixe pour reproductibilité
    )
    
    best_move = engine.get_best_move_with_time_limit(chess)
    print(f"Meilleur coup trouvé: {engine._format_move(best_move)}")
    engine.print_final_stats()
    
    # 2. Configuration sans réduction (pour comparaison)
    print("\n2. Configuration sans réduction (pour comparaison)")
    print("-" * 50)
    
    engine_no_reduction = IterativeDeepeningAlphaBeta(
        max_time=10,
        max_depth=4,
        evaluator=evaluator,
        move_reduction_enabled=False  # Réduction désactivée
    )
    
    best_move_no_reduction = engine_no_reduction.get_best_move_with_time_limit(chess)
    print(f"Meilleur coup trouvé: {engine_no_reduction._format_move(best_move_no_reduction)}")
    engine_no_reduction.print_final_stats()
    
    # 3. Changer la stratégie de réduction en cours d'exécution
    print("\n3. Changement de stratégie en cours d'exécution")
    print("-" * 50)
    
    engine_custom = IterativeDeepeningAlphaBeta(
        max_time=5,
        max_depth=3,
        evaluator=evaluator,
        move_reduction_enabled=True
    )
    
    # Modifier pour utiliser la réduction linéaire
    def linear_strategy(self, move_index, total_moves, depth):
        return self._linear_reduction(move_index, total_moves, depth)
    
    engine_custom.get_reduction_probability = linear_strategy.__get__(engine_custom, engine_custom.__class__)
    
    best_move_custom = engine_custom.get_best_move_with_time_limit(chess)
    print(f"Meilleur coup avec réduction linéaire: {engine_custom._format_move(best_move_custom)}")
    engine_custom.print_final_stats()
    
    # 4. Test avec différentes graines
    print("\n4. Test avec différentes graines aléatoires")
    print("-" * 50)
    
    for seed in [1, 42, 123]:
        engine_seed = IterativeDeepeningAlphaBeta(
            max_time=3,
            max_depth=3,
            evaluator=evaluator,
            move_reduction_enabled=True,
            reduction_seed=seed
        )
        
        best_move_seed = engine_seed.get_best_move_with_time_limit(chess)
        print(f"Graine {seed}: {engine_seed._format_move(best_move_seed)}, "
              f"Nœuds: {engine_seed.nodes_evaluated}, Sautés: {engine_seed.moves_skipped}")

def show_probability_curves():
    """Affiche les courbes de probabilité des différentes stratégies"""
    
    print("\n=== VISUALISATION DES COURBES DE PROBABILITÉ ===\n")
    
    # Créer une engine temporaire pour accéder aux méthodes
    temp_engine = IterativeDeepeningAlphaBeta()
    
    total_moves = 20
    depth = 4
    
    strategies = [
        ('Exponentielle', temp_engine._exponential_reduction),
        ('Linéaire', temp_engine._linear_reduction),
        ('Quadratique', temp_engine._quadratic_reduction),
        ('Adaptative', temp_engine._depth_adaptive_reduction),
    ]
    
    print(f"Probabilités de réduction pour {total_moves} coups à profondeur {depth}:")
    print("-" * 70)
    print(f"{'Position':<8}", end="")
    for name, _ in strategies:
        print(f"{name:<15}", end="")
    print()
    
    for move_index in range(0, total_moves, 2):  # Afficher tous les 2 coups
        print(f"{move_index:2d}/20    ", end="")
        for name, strategy_func in strategies:
            prob = strategy_func(move_index, total_moves, depth)
            print(f"{prob:.2f} ({prob*100:.0f}%)   ", end="")
        print()
    
    print("\nLégende:")
    print("- Position 0/20 = Premier coup (jamais supprimé)")
    print("- Position 18/20 = Avant-dernier coup (forte probabilité)")
    print("- Les probabilités augmentent vers la fin de la liste")

if __name__ == "__main__":
    try:
        example_usage()
        show_probability_curves()
        
    except KeyboardInterrupt:
        print("\n\nExemple interrompu par l'utilisateur")
    except Exception as e:
        print(f"\nErreur pendant l'exemple: {e}")
        import traceback
        traceback.print_exc()