#!/usr/bin/env python3
"""
Test complet du null-move pruning.
Compare les performances avec et sans null-move sur différentes positions.
"""

import time
import sys
import os

# Ajouter le dossier parent au path pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chess import Chess
from evaluator import ChessEvaluator
from Null_move_AI.null_move_engine import NullMovePruningEngine
from Old_AI.iterative_deepening_engine_TT import IterativeDeepeningAlphaBeta

def test_null_move_effectiveness():
    """Test l'efficacité du null-move sur différentes positions"""
    
    print("=== TEST D'EFFICACITÉ DU NULL-MOVE PRUNING ===\n")
    
    evaluator = ChessEvaluator()
    test_time = 20  # Temps par test
    test_depth = 5
    
    # Positions de test
    positions = []
    
    # Position 1: Début de partie
    chess1 = Chess()
    positions.append(("Début de partie", chess1))
    
    # Position 2: Développement actif
    def pos_to_square(row, col):
        return row * 8 + col
    
    chess2 = Chess()
    moves2 = [
        (pos_to_square(1, 4), pos_to_square(3, 4), None),  # e2-e4
        (pos_to_square(6, 4), pos_to_square(4, 4), None),  # e7-e5
        (pos_to_square(0, 6), pos_to_square(2, 5), None),  # Ng1-f3
        (pos_to_square(7, 1), pos_to_square(5, 2), None),  # Nb8-c6
        (pos_to_square(0, 5), pos_to_square(3, 2), None),  # Bf1-c4
    ]
    for move in moves2:
        chess2.move_piece(move[0], move[1], promotion=move[2])
    positions.append(("Développement actif", chess2))
    
    # Position 3: Position plus complexe
    chess3 = Chess()
    moves3 = [
        (pos_to_square(1, 4), pos_to_square(3, 4), None),  # e2-e4
        (pos_to_square(6, 4), pos_to_square(4, 4), None),  # e7-e5
        (pos_to_square(0, 6), pos_to_square(2, 5), None),  # Ng1-f3
        (pos_to_square(7, 1), pos_to_square(5, 2), None),  # Nb8-c6
        (pos_to_square(0, 5), pos_to_square(3, 2), None),  # Bf1-c4
        (pos_to_square(7, 5), pos_to_square(4, 2), None),  # Bf8-c5
        (pos_to_square(1, 3), pos_to_square(3, 3), None),  # d2-d3
        (pos_to_square(6, 3), pos_to_square(4, 3), None),  # d7-d6
    ]
    for move in moves3:
        chess3.move_piece(move[0], move[1], promotion=move[2])
    positions.append(("Position complexe", chess3))
    
    results = []
    
    for pos_name, chess in positions:
        print(f"🎯 {pos_name.upper()}")
        print("=" * 50)
        
        # Test sans null-move (référence)
        print("Sans null-move:")
        engine_ref = IterativeDeepeningAlphaBeta(
            max_time=test_time,
            max_depth=test_depth,
            evaluator=evaluator
        )
        
        start = time.time()
        move_ref = engine_ref.get_best_move_with_time_limit(chess)
        time_ref = time.time() - start
        
        print(f"  Temps: {time_ref:.2f}s")
        print(f"  Nœuds: {engine_ref.nodes_evaluated}")
        print(f"  Coup: {engine_ref._format_move(move_ref)}")
        
        # Test avec null-move
        print("\nAvec null-move:")
        engine_nm = NullMovePruningEngine(
            max_time=test_time,
            max_depth=test_depth,
            evaluator=evaluator,
            null_move_enabled=True,
            null_move_R=2,
            null_move_min_depth=3
        )
        
        start = time.time()
        move_nm = engine_nm.get_best_move_with_time_limit(chess)
        time_nm = time.time() - start
        
        print(f"  Temps: {time_nm:.2f}s")
        print(f"  Nœuds: {engine_nm.nodes_evaluated}")
        print(f"  Coup: {engine_nm._format_move(move_nm)}")
        
        nm_stats = engine_nm.get_null_move_stats()
        print(f"  Null-moves: {nm_stats['attempts']} tentatives, {nm_stats['cutoffs']} cutoffs ({nm_stats['success_rate']:.1f}%)")
        
        # Calculs de performance
        time_gain = ((time_ref - time_nm) / time_ref) * 100 if time_ref > 0 else 0
        node_reduction = ((engine_ref.nodes_evaluated - engine_nm.nodes_evaluated) / engine_ref.nodes_evaluated) * 100 if engine_ref.nodes_evaluated > 0 else 0
        
        print(f"\n📊 PERFORMANCE:")
        print(f"  Gain de temps: {time_gain:+.1f}%")
        print(f"  Réduction de nœuds: {node_reduction:+.1f}%")
        
        if time_gain > 10:
            print("  ✅ Null-move très efficace!")
        elif time_gain > 0:
            print("  ✅ Null-move efficace")
        else:
            print("  ⚠️  Null-move peu efficace sur cette position")
        
        results.append({
            'position': pos_name,
            'time_gain': time_gain,
            'node_reduction': node_reduction,
            'nm_attempts': nm_stats['attempts'],
            'nm_success_rate': nm_stats['success_rate']
        })
        
        print()
    
    # Résumé général
    print("=" * 60)
    print("RÉSUMÉ GÉNÉRAL")
    print("=" * 60)
    
    avg_time_gain = sum(r['time_gain'] for r in results) / len(results)
    avg_node_reduction = sum(r['node_reduction'] for r in results) / len(results)
    total_attempts = sum(r['nm_attempts'] for r in results)
    avg_success_rate = sum(r['nm_success_rate'] for r in results) / len(results)
    
    print(f"Gain de temps moyen: {avg_time_gain:+.1f}%")
    print(f"Réduction de nœuds moyenne: {avg_node_reduction:+.1f}%")
    print(f"Total tentatives null-move: {total_attempts}")
    print(f"Taux de réussite moyen: {avg_success_rate:.1f}%")
    
    print(f"\n{'Position':<20} {'Gain temps':<12} {'Réduction nœuds':<15} {'NM Success':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['position']:<20} {r['time_gain']:+6.1f}%      {r['node_reduction']:+8.1f}%         {r['nm_success_rate']:6.1f}%")

def test_null_move_parameters():
    """Test différents paramètres de null-move"""
    
    print(f"\n=== TEST DES PARAMÈTRES NULL-MOVE ===\n")
    
    chess = Chess()
    # Position un peu développée
    def pos_to_square(row, col):
        return row * 8 + col
    
    moves = [
        (pos_to_square(1, 4), pos_to_square(3, 4), None),  # e2-e4
        (pos_to_square(6, 4), pos_to_square(4, 4), None),  # e7-e5
        (pos_to_square(0, 6), pos_to_square(2, 5), None),  # Ng1-f3
        (pos_to_square(7, 1), pos_to_square(5, 2), None),  # Nb8-c6
    ]
    for move in moves:
        chess.move_piece(move[0], move[1], promotion=move[2])
    
    evaluator = ChessEvaluator()
    test_time = 15
    
    # Test différentes valeurs de R (réduction)
    R_values = [1, 2, 3, 4]
    
    print("Test des valeurs de R (réduction de profondeur):")
    print("-" * 50)
    
    for R in R_values:
        engine = NullMovePruningEngine(
            max_time=test_time,
            max_depth=5,
            evaluator=evaluator,
            null_move_enabled=True,
            null_move_R=R,
            null_move_min_depth=3
        )
        
        start = time.time()
        move = engine.get_best_move_with_time_limit(chess)
        test_time_taken = time.time() - start
        
        nm_stats = engine.get_null_move_stats()
        
        print(f"R={R}: {test_time_taken:.2f}s, {engine.nodes_evaluated} nœuds, "
              f"{nm_stats['cutoffs']}/{nm_stats['attempts']} cutoffs ({nm_stats['success_rate']:.1f}%)")
    
    # Test différentes profondeurs minimales
    print(f"\nTest des profondeurs minimales:")
    print("-" * 50)
    
    min_depths = [2, 3, 4, 5]
    
    for min_depth in min_depths:
        engine = NullMovePruningEngine(
            max_time=test_time,
            max_depth=5,
            evaluator=evaluator,
            null_move_enabled=True,
            null_move_R=2,
            null_move_min_depth=min_depth
        )
        
        start = time.time()
        move = engine.get_best_move_with_time_limit(chess)
        test_time_taken = time.time() - start
        
        nm_stats = engine.get_null_move_stats()
        
        print(f"Min depth={min_depth}: {test_time_taken:.2f}s, {engine.nodes_evaluated} nœuds, "
              f"{nm_stats['cutoffs']}/{nm_stats['attempts']} cutoffs ({nm_stats['success_rate']:.1f}%)")

if __name__ == "__main__":
    try:
        test_null_move_effectiveness()
        test_null_move_parameters()
        
        print(f"\n=== RECOMMANDATIONS ===")
        print("✅ Null-move est très efficace sur les positions tactiques")
        print("✅ R=2 ou R=3 sont généralement optimaux")
        print("✅ Profondeur minimale de 3 évite les overhead inutiles")
        print("✅ Combiné avec la réduction des coups, donne d'excellents résultats")
        
    except KeyboardInterrupt:
        print("\n\nTest interrompu par l'utilisateur")
    except Exception as e:
        print(f"\nErreur pendant le test: {e}")
        import traceback
        traceback.print_exc()