#!/usr/bin/env python3
"""
Test comparatif simple: Null-Move Engine efficacité
Chess.py vs optimized_chess.py - 60s chacun
"""

import sys
import os
import time

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Profile'))

from Chess import Chess
from Null_move_AI.null_move_engine import NullMovePruningEngine
from optimized_chess import patch_chess_class

def run_single_test(use_optimizations=False, time_limit=60):
    """Lance un test unique"""
    
    # Créer le board
    chess = Chess()
    
    # Appliquer optimisations si demandé
    if use_optimizations:
        patch_chess_class(chess)
        test_type = "OPTIMISÉ"
    else:
        test_type = "ORIGINAL"
    
    print(f"\n🔍 === TEST {test_type} ===")
    
    # Créer l'engine
    engine = NullMovePruningEngine(
        max_time=time_limit,
        max_depth=20,
        null_move_enabled=True,
        null_move_R=2,
        null_move_min_depth=3,
        tt_size=1000000
    )
    
    print(f"🚀 Recherche en cours ({time_limit}s max)...")
    start = time.time()
    
    try:
        best_move = engine.get_best_move_with_time_limit(chess)
        elapsed = time.time() - start
        
        # Résultats
        speed = engine.nodes_evaluated / elapsed if elapsed > 0 else 0
        null_eff = (engine.null_move_cutoffs / engine.null_move_attempts * 100) if engine.null_move_attempts > 0 else 0
        
        print(f"✅ Terminé en {elapsed:.2f}s")
        print(f"🎯 Coup: {engine._format_move(best_move) if best_move else 'None'}")
        print(f"📊 Nœuds: {engine.nodes_evaluated:,}")
        print(f"🚀 Vitesse: {speed:,.0f} nœuds/sec")
        print(f"✂️  Null-move: {engine.null_move_cutoffs:,}/{engine.null_move_attempts:,} ({null_eff:.1f}%)")
        print(f"💾 TT hits: {engine.tt_hits:,}")
        
        return {
            'success': True,
            'time': elapsed,
            'nodes': engine.nodes_evaluated,
            'speed': speed,
            'move': best_move,
            'null_cutoffs': engine.null_move_cutoffs,
            'null_attempts': engine.null_move_attempts,
            'tt_hits': engine.tt_hits,
            'engine': engine
        }
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ Erreur après {elapsed:.2f}s: {e}")
        return {'success': False, 'time': elapsed, 'error': str(e)}

def main():
    print("🔥 === TEST COMPARATIF NULL-MOVE === 🔥")
    print("📊 Efficacité Chess.py vs optimized_chess.py")
    print("⏱️  60 secondes par test")
    print("=" * 50)
    
    # Test 1: Original
    print("\n📊 Test version originale...")
    result_orig = run_single_test(use_optimizations=False, time_limit=60)
    
    # Test 2: Optimisé
    print("\n⚡ Test version optimisée...")
    result_opt = run_single_test(use_optimizations=True, time_limit=60)
    
    # Comparaison
    if result_orig['success'] and result_opt['success']:
        print(f"\n🏆 === COMPARAISON ===")
        print(f"-" * 40)
        
        speed_gain = result_opt['speed'] / result_orig['speed'] if result_orig['speed'] > 0 else 0
        node_gain = result_opt['nodes'] / result_orig['nodes'] if result_orig['nodes'] > 0 else 0
        
        print(f"⏱️  Temps:")
        print(f"   Original: {result_orig['time']:.2f}s")
        print(f"   Optimisé: {result_opt['time']:.2f}s")
        
        print(f"\n🚀 Vitesse:")
        print(f"   Original: {result_orig['speed']:,.0f} nœuds/sec")
        print(f"   Optimisé: {result_opt['speed']:,.0f} nœuds/sec")
        print(f"   📈 Gain: {speed_gain:.2f}x")
        
        print(f"\n📊 Exploration:")
        print(f"   Original: {result_orig['nodes']:,} nœuds")
        print(f"   Optimisé: {result_opt['nodes']:,} nœuds")
        print(f"   📈 Gain: {node_gain:.2f}x")
        
        # Validation
        move1 = result_orig['move']
        move2 = result_opt['move']
        same_move = (move1 == move2) if (move1 and move2) else False
        
        print(f"\n✅ Validation:")
        print(f"   Original: {result_orig['engine']._format_move(move1) if move1 else 'None'}")
        print(f"   Optimisé: {result_opt['engine']._format_move(move2) if move2 else 'None'}")
        print(f"   Identique: {'✅' if same_move else '❌'}")
        
        # Conclusion
        print(f"\n🎯 === RÉSULTAT ===")
        if speed_gain >= 2.5:
            print(f"🔥 EXCELLENT! {speed_gain:.1f}x plus rapide")
        elif speed_gain >= 2.0:
            print(f"🔥 TRÈS BON! {speed_gain:.1f}x plus rapide")
        elif speed_gain >= 1.5:
            print(f"👍 BON! {speed_gain:.1f}x plus rapide")
        else:
            print(f"🤔 MOYEN. {speed_gain:.1f}x plus rapide")
        
        print(f"🧠 Exploration {node_gain:.1f}x plus profonde")
    
    print(f"\n🏁 Test terminé!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Interrompu")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")