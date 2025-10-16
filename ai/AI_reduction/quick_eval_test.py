#!/usr/bin/env python3
"""
Test rapide pour vérifier la performance en situation réelle
"""

import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Appliquer les optimisations bitboard immédiatement pour que toute instance
# de `Chess()` créée dans ce module utilise les versions optimisées.
try:
    from optimized_chess import patch_chess_class_globally
    patch_chess_class_globally()
    print("✅ Optimisations bitboard appliquées (import-time)")
except Exception:
    # Ne pas échouer l'import si optimized_chess manque — on continue sans optimisations
    print("⚠️  optimized_chess non disponible au moment de l'import")

from Chess import Chess
from evaluator import ChessEvaluator as OriginalEvaluator
from fast_evaluator import FastChessEvaluator, SuperFastChessEvaluator
from Null_move_AI.null_move_engine import NullMovePruningEngine

def quick_test():
    """Test rapide pour voir l'impact réel"""
    
    # Position de test
    chess = Chess()
    moves = [(12, 28, None), (52, 36, None), (6, 21, None), (57, 42, None)]
    for move in moves:
        chess.move_piece(move[0], move[1], move[2])
    
    print("🔍 Test rapide - 10 secondes par évaluateur")
    print("=" * 50)
    
    evaluators = [
        ("Original", OriginalEvaluator()),
        ("FastChess", FastChessEvaluator()),
        ("SuperFast", SuperFastChessEvaluator())
    ]
    
    results = []
    
    for name, evaluator in evaluators:
        print(f"\n🔍 Test {name}...")
        
        engine = NullMovePruningEngine(
            max_time=10,  # 10 secondes seulement
            max_depth=10,
            evaluator=evaluator,
            tt_size=100000,
            null_move_enabled=True,
            null_move_R=2
        )
        
        start = time.time()
        try:
            best_move = engine.get_best_move_with_time_limit(chess)
            elapsed = time.time() - start
            
            print(f"   ⏱️  Temps: {elapsed:.2f}s")
            print(f"   📈 Nœuds: {engine.nodes_evaluated:,}")
            print(f"   🚀 Vitesse: {engine.nodes_evaluated/elapsed:.0f} nœuds/sec")
            
            results.append({
                'name': name,
                'time': elapsed,
                'nodes': engine.nodes_evaluated,
                'speed': engine.nodes_evaluated/elapsed
            })
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    print(f"\n📊 COMPARAISON RAPIDE:")
    print("-" * 40)
    for r in results:
        print(f"{r['name']:<10}: {r['speed']:.0f} nœuds/sec")
    
    if len(results) >= 2:
        ratio = results[1]['speed'] / results[0]['speed']
        print(f"\nAmélioração FastChess: {ratio:.1f}x")

if __name__ == "__main__":
    # Appliquer optimisations bitboard
    try:
        from optimized_chess import patch_chess_class_globally
        patch_chess_class_globally()
        print("✅ Optimisations bitboard appliquées")
    except:
        print("⚠️  Sans optimisations bitboard")
    
    quick_test()