#!/usr/bin/env python3
"""
Test de performance de la réduction sélective des coups.
Compare les performances avec différentes stratégies de réduction.
"""

import time
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chess import Chess
from evaluator import ChessEvaluator
from AI_reduction.iterative_deepening_engine_TT_rdcut import IterativeDeepeningAlphaBeta

def test_move_reduction_strategies():
    """Test de différentes stratégies de réduction"""
    
    print("=== TEST DES STRATÉGIES DE RÉDUCTION DES COUPS ===\n")
    
    # Configuration du test
    test_time = 30  # secondes par test (sera modifié pour certains tests)
    test_depth = 4
    
    # Position de test (milieu de partie avec beaucoup de coups possibles)
    chess = Chess()
    # Quelques coups pour arriver à une position intéressante
    # Convertir les coordonnées (row, col) en indices de case (0-63)
    def pos_to_square(row, col):
        return row * 8 + col
    
    moves = [
        (pos_to_square(1, 4), pos_to_square(3, 4), None),  # e2-e4
        (pos_to_square(6, 4), pos_to_square(4, 4), None),  # e7-e5
        (pos_to_square(0, 6), pos_to_square(2, 5), None),  # Ng1-f3
        (pos_to_square(7, 1), pos_to_square(5, 2), None),  # Nb8-c6
        (pos_to_square(0, 5), pos_to_square(3, 2), None),  # Bf1-c4
        (pos_to_square(7, 5), pos_to_square(4, 2), None),  # Bf8-c5
    ]
    
    for move in moves:
        chess.move_piece(move[0], move[1], promotion=move[2])
    
    print(f"Position de test configurée (après {len(moves)} coups)")
    print(f"Temps limite: {test_time}s, Profondeur cible: {test_depth}")
    print(f"Au tour de: {'Blancs' if chess.white_to_move else 'Noirs'}\n")
    
    evaluator = ChessEvaluator()
    results = []
    
    # Test 1: Sans réduction (référence)
    print("1. TEST SANS RÉDUCTION (référence)")
    print("-" * 40)
    
    engine_ref = IterativeDeepeningAlphaBeta(
        max_time=test_time, 
        max_depth=test_depth,
        evaluator=evaluator,
        move_reduction_enabled=False
    )
    
    start_time = time.time()
    best_move_ref = engine_ref.get_best_move_with_time_limit(chess)
    ref_time = time.time() - start_time
    
    engine_ref.print_final_stats()
    
    results.append({
        'name': 'Sans réduction',
        'time': ref_time,
        'nodes': engine_ref.nodes_evaluated,
        'move': engine_ref._format_move(best_move_ref) if best_move_ref else "None",
        'moves_skipped': 0
    })
    
    # Stratégies de réduction à tester
    strategies = [
        ('Réduction exponentielle', '_exponential_reduction'),
        ('Réduction linéaire', '_linear_reduction'),
        ('Réduction quadratique', '_quadratic_reduction'),
        ('Réduction adaptative', '_depth_adaptive_reduction'),
    ]
    
    test_num = 2
    for strategy_name, strategy_method in strategies:
        print(f"\n{test_num}. TEST {strategy_name.upper()}")
        print("-" * 40)
        
        # Temps spécial pour la réduction exponentielle
        current_test_time = 90 if strategy_name == 'Réduction exponentielle' else test_time
        if strategy_name == 'Réduction exponentielle':
            print(f"⏱️  Temps étendu à {current_test_time}s pour ce test")
        
        # Créer une engine avec réduction activée
        engine = IterativeDeepeningAlphaBeta(
            max_time=current_test_time,
            max_depth=test_depth,
            evaluator=evaluator,
            move_reduction_enabled=True,
            reduction_seed=42  # Seed fixe pour reproductibilité
        )
        
        # Modifier la stratégie de réduction utilisée
        def get_custom_probability(self, move_index, total_moves, depth):
            """Utilise la stratégie sélectionnée"""
            if not self.move_reduction_enabled or total_moves <= 3:
                return 0.0
            
            # Appeler la méthode de stratégie appropriée
            strategy_func = getattr(self, strategy_method)
            return strategy_func(move_index, total_moves, depth)
        
        # Remplacer temporairement la méthode
        original_method = engine.get_reduction_probability
        engine.get_reduction_probability = get_custom_probability.__get__(engine, engine.__class__)
        
        start_time = time.time()
        best_move = engine.get_best_move_with_time_limit(chess)
        test_time_taken = time.time() - start_time
        
        engine.print_final_stats()
        
        # Restaurer la méthode originale
        engine.get_reduction_probability = original_method
        
        # Calculer les gains de performance
        speed_gain = ((ref_time - test_time_taken) / ref_time) * 100 if ref_time > 0 else 0
        node_reduction = ((engine_ref.nodes_evaluated - engine.nodes_evaluated) / engine_ref.nodes_evaluated) * 100 if engine_ref.nodes_evaluated > 0 else 0
        
        results.append({
            'name': strategy_name,
            'time': test_time_taken,
            'nodes': engine.nodes_evaluated,
            'move': engine._format_move(best_move) if best_move else "None",
            'moves_skipped': engine.moves_skipped,
            'speed_gain': speed_gain,
            'node_reduction': node_reduction
        })
        
        print(f"Gain de vitesse: {speed_gain:+.1f}%")
        print(f"Réduction de nœuds: {node_reduction:+.1f}%")
        
        test_num += 1
    
    # Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ COMPARATIF")
    print("="*60)
    
    print(f"{'Stratégie':<25} {'Temps (s)':<10} {'Nœuds':<8} {'Sautés':<8} {'Gain %':<8}")
    print("-" * 60)
    
    for result in results:
        gain_str = f"{result.get('speed_gain', 0):+.1f}%" if 'speed_gain' in result else "  ref"
        print(f"{result['name']:<25} {result['time']:<10.2f} {result['nodes']:<8} {result['moves_skipped']:<8} {gain_str:<8}")
    
    print("\n" + "="*60)
    
    # Recommandations
    print("RECOMMANDATIONS:")
    best_strategy = max([r for r in results if 'speed_gain' in r], 
                       key=lambda x: x['speed_gain'], default=None)
    
    if best_strategy and best_strategy['speed_gain'] > 0:
        print(f"✓ Meilleure stratégie: {best_strategy['name']}")
        print(f"  Gain de vitesse: {best_strategy['speed_gain']:.1f}%")
        print(f"  Réduction de nœuds: {best_strategy.get('node_reduction', 0):.1f}%")
        print(f"  Coups sautés: {best_strategy['moves_skipped']}")
    else:
        print("⚠ Aucune stratégie n'améliore les performances sur ce test")
        
    print(f"\nTous les moteurs ont trouvé {'le même coup' if len(set(r['move'] for r in results)) == 1 else 'des coups différents'}")

def test_seed_reproducibility():
    """Test la reproductibilité avec différentes graines"""
    
    print("\n" + "="*60)
    print("TEST DE REPRODUCTIBILITÉ DES GRAINES")
    print("="*60)
    
    chess = Chess()
    evaluator = ChessEvaluator()
    
    seeds = [42, 123, 999]
    
    for seed in seeds:
        print(f"\nTest avec graine {seed}:")
        
        engine = IterativeDeepeningAlphaBeta(
            max_time=5,
            max_depth=3,
            evaluator=evaluator,
            move_reduction_enabled=True,
            reduction_seed=seed
        )
        
        start_time = time.time()
        best_move = engine.get_best_move_with_time_limit(chess)
        test_time = time.time() - start_time
        
        print(f"  Temps: {test_time:.2f}s, Nœuds: {engine.nodes_evaluated}, "
              f"Sautés: {engine.moves_skipped}, Coup: {engine._format_move(best_move)}")

if __name__ == "__main__":
    try:
        test_move_reduction_strategies()
        test_seed_reproducibility()
        
    except KeyboardInterrupt:
        print("\n\nTest interrompu par l'utilisateur")
    except Exception as e:
        print(f"\nErreur pendant le test: {e}")
        import traceback
        traceback.print_exc()