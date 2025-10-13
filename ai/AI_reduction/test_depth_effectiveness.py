#!/usr/bin/env python3
"""
Test pour démontrer l'efficacité de la réduction à profondeur élevée
"""

import time
import sys
import os

# Ajouter le dossier parent au path pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chess import Chess
from evaluator import ChessEvaluator
from AI_reduction.iterative_deepening_engine_TT_rdcut import IterativeDeepeningAlphaBeta

def test_depth_progression():
    """Test l'effet de la réduction sur la progression en profondeur"""
    
    print("=== TEST DE PROGRESSION EN PROFONDEUR ===\n")
    
    # Position complexe avec beaucoup de coups
    chess = Chess()
    def pos_to_square(row, col):
        return row * 8 + col
    
    # Position d'ouverture avec beaucoup d'options
    moves = [
        (pos_to_square(1, 4), pos_to_square(3, 4), None),  # e2-e4
        (pos_to_square(6, 4), pos_to_square(4, 4), None),  # e7-e5
        (pos_to_square(0, 6), pos_to_square(2, 5), None),  # Ng1-f3
        (pos_to_square(7, 1), pos_to_square(5, 2), None),  # Nb8-c6
    ]
    
    for move in moves:
        chess.move_piece(move[0], move[1], promotion=move[2])
    
    evaluator = ChessEvaluator()
    
    # Test avec temps plus long pour vraiment voir la différence
    test_times = [15, 30, 60]  # Différents temps de test
    
    for test_time in test_times:
        print(f"⏱️  TEST AVEC {test_time} SECONDES")
        print("=" * 50)
        
        results = {}
        
        # Sans réduction
        print("Sans réduction:")
        engine_none = IterativeDeepeningAlphaBeta(
            max_time=test_time,
            max_depth=10,  # Profondeur élevée
            evaluator=evaluator,
            move_reduction_enabled=False
        )
        
        start = time.time()
        move_none = engine_none.get_best_move_with_time_limit(chess)
        time_none = time.time() - start
        
        results['none'] = {
            'time': time_none,
            'nodes': engine_none.nodes_evaluated,
            'pruned': engine_none.pruned_branches,
            'tt_hits': engine_none.tt_hits,
            'move': engine_none._format_move(move_none) if move_none else "None"
        }
        
        print(f"  Temps utilisé: {time_none:.1f}s")
        print(f"  Nœuds: {engine_none.nodes_evaluated}")
        print(f"  Coup: {results['none']['move']}")
        
        # Avec réduction exponentielle
        print("\nAvec réduction exponentielle:")
        engine_red = IterativeDeepeningAlphaBeta(
            max_time=test_time,
            max_depth=10,  # Profondeur élevée
            evaluator=evaluator,
            move_reduction_enabled=True,
            reduction_seed=42
        )
        
        start = time.time()
        move_red = engine_red.get_best_move_with_time_limit(chess)
        time_red = time.time() - start
        
        results['reduction'] = {
            'time': time_red,
            'nodes': engine_red.nodes_evaluated,
            'pruned': engine_red.pruned_branches,
            'tt_hits': engine_red.tt_hits,
            'skipped': engine_red.moves_skipped,
            'move': engine_red._format_move(move_red) if move_red else "None"
        }
        
        print(f"  Temps utilisé: {time_red:.1f}s")
        print(f"  Nœuds: {engine_red.nodes_evaluated}")
        print(f"  Coups sautés: {engine_red.moves_skipped}")
        print(f"  Coup: {results['reduction']['move']}")
        
        # Analyse comparative
        if results['none']['nodes'] > 0:
            node_diff = ((results['reduction']['nodes'] - results['none']['nodes']) / results['none']['nodes']) * 100
            time_diff = ((results['reduction']['time'] - results['none']['time']) / results['none']['time']) * 100
            efficiency = results['reduction']['nodes'] / max(1, results['reduction']['time'])
            efficiency_ref = results['none']['nodes'] / max(1, results['none']['time'])
            efficiency_gain = ((efficiency - efficiency_ref) / efficiency_ref) * 100
            
            print(f"\n📊 ANALYSE:")
            print(f"  Différence de nœuds: {node_diff:+.1f}%")
            print(f"  Différence de temps: {time_diff:+.1f}%")
            print(f"  Efficacité (nœuds/seconde): {efficiency_gain:+.1f}%")
            
            if node_diff > 20 and time_diff < 10:
                print("  🎯 EXCELLENT: Plus de nœuds explorés pour un temps similaire!")
            elif efficiency_gain > 10:
                print("  ✅ BON: Meilleure efficacité d'exploration")
            elif time_diff > 20:
                print("  ⚠️  ATTENTION: Overhead trop élevé")
        
        print()

def test_fixed_depth():
    """Test à profondeur fixe pour mesurer la vraie efficacité"""
    
    print("=== TEST À PROFONDEUR FIXE ===\n")
    
    chess = Chess()
    evaluator = ChessEvaluator()
    
    depths = [3, 4, 5]
    
    for depth in depths:
        print(f"🎯 PROFONDEUR {depth}")
        print("-" * 30)
        
        # Sans réduction
        engine_none = IterativeDeepeningAlphaBeta(
            max_time=120,  # Temps très généreux
            max_depth=depth,  # Profondeur fixe
            evaluator=evaluator,
            move_reduction_enabled=False
        )
        
        start = time.time()
        move_none = engine_none.get_best_move_with_time_limit(chess)
        time_none = time.time() - start
        
        print(f"Sans réduction: {time_none:.2f}s, {engine_none.nodes_evaluated} nœuds")
        
        # Avec réduction
        engine_red = IterativeDeepeningAlphaBeta(
            max_time=120,
            max_depth=depth,
            evaluator=evaluator,
            move_reduction_enabled=True,
            reduction_seed=42
        )
        
        start = time.time()
        move_red = engine_red.get_best_move_with_time_limit(chess)
        time_red = time.time() - start
        
        gain = ((time_none - time_red) / time_none) * 100 if time_none > 0 else 0
        node_reduction = ((engine_none.nodes_evaluated - engine_red.nodes_evaluated) / engine_none.nodes_evaluated) * 100 if engine_none.nodes_evaluated > 0 else 0
        
        print(f"Avec réduction: {time_red:.2f}s, {engine_red.nodes_evaluated} nœuds, {engine_red.moves_skipped} sautés")
        print(f"Gain de temps: {gain:+.1f}%")
        print(f"Réduction de nœuds: {node_reduction:+.1f}%")
        
        if gain > 0:
            print("✅ Réduction efficace!")
        else:
            print("❌ Réduction inefficace à cette profondeur")
        
        print()

if __name__ == "__main__":
    try:
        test_depth_progression()
        test_fixed_depth()
        
    except KeyboardInterrupt:
        print("\nTest interrompu")
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()