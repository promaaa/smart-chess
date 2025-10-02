#!/usr/bin/env python3
"""
Analyse détaillée du problème de performance avec la réduction.
Ce script va analyser pourquoi la réduction peut être plus lente.
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Chess import Chess
from evaluator import ChessEvaluator
from iterative_deepening_engine_TT_rdcut import IterativeDeepeningAlphaBeta

def analyze_reduction_overhead():
    """Analyse l'overhead de la réduction des coups"""
    
    print("=== ANALYSE DE L'OVERHEAD DE RÉDUCTION ===\n")
    
    # Position simple pour analyse
    chess = Chess()
    evaluator = ChessEvaluator()
    
    # Test court pour mesurer l'overhead pur
    test_time = 10
    test_depth = 3
    
    print(f"Test rapide: {test_time}s, profondeur {test_depth}")
    print("-" * 50)
    
    # 1. Sans réduction
    print("1. SANS RÉDUCTION")
    engine_none = IterativeDeepeningAlphaBeta(
        max_time=test_time,
        max_depth=test_depth,
        evaluator=evaluator,
        move_reduction_enabled=False
    )
    
    start = time.time()
    move_none = engine_none.get_best_move_with_time_limit(chess)
    time_none = time.time() - start
    
    print(f"   Temps: {time_none:.2f}s")
    print(f"   Nœuds: {engine_none.nodes_evaluated}")
    print(f"   Branches élaguées: {engine_none.pruned_branches}")
    print(f"   TT hits: {engine_none.tt_hits}")
    
    # 2. Avec réduction mais AUCUNE probabilité (overhead pur)
    print("\n2. AVEC RÉDUCTION MAIS 0% DE PROBABILITÉ (overhead pur)")
    engine_zero = IterativeDeepeningAlphaBeta(
        max_time=test_time,
        max_depth=test_depth,
        evaluator=evaluator,
        move_reduction_enabled=True,
        reduction_seed=42
    )
    
    # Modifier pour avoir 0% de probabilité partout
    def zero_probability(self, move_index, total_moves, depth):
        return 0.0  # Aucune réduction
    
    engine_zero.get_reduction_probability = zero_probability.__get__(engine_zero, engine_zero.__class__)
    
    start = time.time()
    move_zero = engine_zero.get_best_move_with_time_limit(chess)
    time_zero = time.time() - start
    
    print(f"   Temps: {time_zero:.2f}s")
    print(f"   Nœuds: {engine_zero.nodes_evaluated}")
    print(f"   Branches élaguées: {engine_zero.pruned_branches}")
    print(f"   TT hits: {engine_zero.tt_hits}")
    print(f"   Coups sautés: {engine_zero.moves_skipped}")
    
    overhead_pur = ((time_zero - time_none) / time_none) * 100
    print(f"   🔍 Overhead pur de la réduction: {overhead_pur:+.1f}%")
    
    # 3. Avec réduction agressive
    print("\n3. AVEC RÉDUCTION AGRESSIVE")
    engine_aggressive = IterativeDeepeningAlphaBeta(
        max_time=test_time,
        max_depth=test_depth,
        evaluator=evaluator,
        move_reduction_enabled=True,
        reduction_seed=42
    )
    
    # Réduction très agressive pour voir l'effet
    def aggressive_reduction(self, move_index, total_moves, depth):
        if move_index < 2:  # Protéger les 2 premiers
            return 0.0
        return min(0.8, move_index / total_moves)  # Jusqu'à 80%
    
    engine_aggressive.get_reduction_probability = aggressive_reduction.__get__(engine_aggressive, engine_aggressive.__class__)
    
    start = time.time()
    move_aggressive = engine_aggressive.get_best_move_with_time_limit(chess)
    time_aggressive = time.time() - start
    
    print(f"   Temps: {time_aggressive:.2f}s")
    print(f"   Nœuds: {engine_aggressive.nodes_evaluated}")
    print(f"   Branches élaguées: {engine_aggressive.pruned_branches}")
    print(f"   TT hits: {engine_aggressive.tt_hits}")
    print(f"   Coups sautés: {engine_aggressive.moves_skipped}")
    
    gain_aggressive = ((time_none - time_aggressive) / time_none) * 100
    reduction_rate = (engine_aggressive.moves_skipped / max(1, engine_aggressive.nodes_evaluated)) * 100
    print(f"   🎯 Gain de performance: {gain_aggressive:+.1f}%")
    print(f"   📊 Taux de réduction: {reduction_rate:.1f}%")
    
    # Analyse détaillée
    print(f"\n=== ANALYSE ===")
    print(f"Overhead pur (calculs de probabilité): {overhead_pur:+.1f}%")
    print(f"Gain net avec réduction agressive: {gain_aggressive:+.1f}%")
    
    if overhead_pur > 5:
        print("⚠️  PROBLÈME: L'overhead de calcul est trop élevé!")
        print("   Causes possibles:")
        print("   - Calculs de probabilité trop complexes")
        print("   - Appels de fonction trop fréquents")
        print("   - Filtrage de listes coûteux")
    
    if gain_aggressive < 0:
        print("⚠️  PROBLÈME: Même avec réduction agressive, pas de gain!")
        print("   Causes possibles:")
        print("   - Position trop simple (peu de coups)")
        print("   - Algorithme d'ordonnancement déjà très efficace")
        print("   - Overhead supérieur aux gains")
    
    return {
        'overhead_pur': overhead_pur,
        'gain_aggressive': gain_aggressive,
        'reduction_rate': reduction_rate
    }

def test_different_positions():
    """Test sur différentes positions pour voir l'impact"""
    
    print(f"\n=== TEST SUR DIFFÉRENTES POSITIONS ===\n")
    
    evaluator = ChessEvaluator()
    positions = []
    
    # Position 1: Début de partie (peu de coups)
    chess1 = Chess()
    positions.append(("Début de partie", chess1))
    
    # Position 2: Milieu de partie (plus de coups)
    chess2 = Chess()
    moves = [12, 28, 6, 21, 5, 26, 1, 18]  # Quelques coups d'ouverture
    for i in range(0, len(moves), 2):
        if i+1 < len(moves):
            chess2.move_piece(moves[i], moves[i+1])
    positions.append(("Milieu de partie", chess2))
    
    for pos_name, chess in positions:
        print(f"📍 {pos_name}")
        
        # Compter les coups légaux
        from base_engine import BaseChessEngine
        temp_engine = BaseChessEngine()
        legal_moves = temp_engine._get_all_legal_moves(chess)
        print(f"   Coups légaux: {len(legal_moves)}")
        
        if len(legal_moves) < 10:
            print("   ⚠️  Peu de coups - réduction peu utile")
        elif len(legal_moves) > 30:
            print("   ✅ Beaucoup de coups - réduction potentiellement utile")
        
        # Test rapide
        engine_test = IterativeDeepeningAlphaBeta(
            max_time=5,
            max_depth=2,
            evaluator=evaluator,
            move_reduction_enabled=False
        )
        
        start = time.time()
        move = engine_test.get_best_move_with_time_limit(chess)
        test_time = time.time() - start
        
        print(f"   Temps sans réduction: {test_time:.2f}s")
        print(f"   Nœuds: {engine_test.nodes_evaluated}")
        print()

if __name__ == "__main__":
    try:
        results = analyze_reduction_overhead()
        test_different_positions()
        
        print("=== RECOMMANDATIONS ===")
        if results['overhead_pur'] > 10:
            print("🔧 Optimiser les calculs de probabilité")
            print("🔧 Utiliser des lookup tables plutôt que des calculs")
            print("🔧 Réduire la fréquence des appels de fonction")
        
        if results['gain_aggressive'] < 5:
            print("🎯 La réduction n'est efficace que sur des positions complexes")
            print("🎯 Considérer une activation adaptive selon le nombre de coups")
        
        print("✅ Tester avec des profondeurs plus élevées (5-6)")
        
    except KeyboardInterrupt:
        print("\n\nAnalyse interrompue")
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()