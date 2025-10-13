#!/usr/bin/env python3
"""
Test comparatif rapide de l'engine Null-Move avec timeout strict
Temps maximum: 60 secondes STRICT par test
"""

import sys
import os
import time
import signal
from contextlib import contextmanager

# Ajouter les paths nécessaires
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chess import Chess
from Null_move_AI.null_move_engine import NullMovePruningEngine

# Import des optimisations depuis le dossier Profile
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Profile'))
from optimized_chess import patch_chess_class, unpatch_chess_class

class TimeoutError(Exception):
    pass

@contextmanager
def timeout_context(seconds):
    """Context manager pour timeout strict"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Timeout après {seconds}s")
    
    # Sauvegarder l'ancien handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restaurer l'ancien handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def test_engine_with_timeout(chess, use_optimizations=False, timeout_seconds=60):
    """
    Test un engine avec timeout strict
    """
    if use_optimizations:
        patch_chess_class(chess)
    
    engine = NullMovePruningEngine(
        max_time=timeout_seconds - 5,  # 5s de marge pour éviter les problèmes
        max_depth=20,
        null_move_enabled=True,
        null_move_R=2,
        null_move_min_depth=3,
        tt_size=1000000
    )
    
    start_time = time.time()
    best_move = None
    
    try:
        # Utiliser un timeout système strict
        with timeout_context(timeout_seconds):
            best_move = engine.get_best_move_with_time_limit(chess)
    except TimeoutError:
        print(f"⏰ TIMEOUT STRICT après {timeout_seconds}s")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    actual_time = time.time() - start_time
    
    return {
        'best_move': best_move,
        'time': actual_time,
        'nodes': engine.nodes_evaluated,
        'pruned': engine.pruned_branches,
        'null_attempts': engine.null_move_attempts,
        'null_cutoffs': engine.null_move_cutoffs,
        'null_failures': engine.null_move_failures,
        'tt_entries': len(engine.transposition_table),
        'tt_hits': engine.tt_hits,
        'tt_misses': engine.tt_misses,
        'engine': engine
    }

def quick_null_move_test():
    """
    Test rapide et contrôlé de l'engine null-move
    """
    print("⚡ === TEST RAPIDE NULL-MOVE (TIMEOUT STRICT) === ⚡")
    print("⏱️  Limite STRICTE: 60 secondes par test")
    print("🛡️  Protection contre les boucles infinies")
    print()
    
    timeout_seconds = 60
    
    # ========== TEST 1: VERSION ORIGINALE ==========
    print("🏃 === TEST 1: VERSION ORIGINALE ===")
    
    chess_original = Chess()
    print("🚀 Démarrage (timeout 60s)...")
    
    results_original = test_engine_with_timeout(
        chess_original, 
        use_optimizations=False, 
        timeout_seconds=timeout_seconds
    )
    
    print(f"✅ Terminé en {results_original['time']:.2f}s")
    print(f"🎯 Coup: {results_original['engine']._format_move(results_original['best_move']) if results_original['best_move'] else 'None'}")
    print(f"📊 Nœuds: {results_original['nodes']:,}")
    print(f"🚀 Vitesse: {results_original['nodes']/results_original['time']:,.0f} nœuds/sec")
    print(f"✂️  Null cutoffs: {results_original['null_cutoffs']:,}/{results_original['null_attempts']:,}")
    print(f"💾 TT: {results_original['tt_hits']:,} hits")
    print()
    
    # ========== TEST 2: VERSION OPTIMISÉE ==========
    print("⚡ === TEST 2: VERSION OPTIMISÉE ===")
    
    chess_optimized = Chess()
    print("🚀 Démarrage (timeout 60s)...")
    
    results_optimized = test_engine_with_timeout(
        chess_optimized, 
        use_optimizations=True, 
        timeout_seconds=timeout_seconds
    )
    
    print(f"✅ Terminé en {results_optimized['time']:.2f}s")
    print(f"🎯 Coup: {results_optimized['engine']._format_move(results_optimized['best_move']) if results_optimized['best_move'] else 'None'}")
    print(f"📊 Nœuds: {results_optimized['nodes']:,}")
    print(f"🚀 Vitesse: {results_optimized['nodes']/results_optimized['time']:,.0f} nœuds/sec")
    print(f"✂️  Null cutoffs: {results_optimized['null_cutoffs']:,}/{results_optimized['null_attempts']:,}")
    print(f"💾 TT: {results_optimized['tt_hits']:,} hits")
    print()
    
    # ========== COMPARAISON ==========
    print("🏆 === COMPARAISON ===")
    print("-" * 50)
    
    if results_original['time'] > 0 and results_optimized['time'] > 0:
        time_ratio = results_original['time'] / results_optimized['time']
        speed_orig = results_original['nodes'] / results_original['time']
        speed_opt = results_optimized['nodes'] / results_optimized['time']
        speed_ratio = speed_opt / speed_orig if speed_orig > 0 else 0
        
        print(f"⏱️  Temps:")
        print(f"   Original:  {results_original['time']:.2f}s")
        print(f"   Optimisé:  {results_optimized['time']:.2f}s")
        print(f"   Speedup:   {time_ratio:.2f}x")
        print()
        
        print(f"🚀 Vitesse:")
        print(f"   Original:  {speed_orig:,.0f} nœuds/sec")
        print(f"   Optimisé:  {speed_opt:,.0f} nœuds/sec")
        print(f"   Speedup:   {speed_ratio:.2f}x")
        print()
        
        print(f"📊 Exploration:")
        print(f"   Original:  {results_original['nodes']:,} nœuds")
        print(f"   Optimisé:  {results_optimized['nodes']:,} nœuds")
        if results_original['nodes'] > 0:
            node_ratio = results_optimized['nodes'] / results_original['nodes']
            print(f"   Ratio:     {node_ratio:.2f}x")
        print()
        
        # Validation
        move1 = results_original['best_move']
        move2 = results_optimized['best_move']
        moves_match = (move1 == move2) if (move1 and move2) else False
        
        print(f"✅ Validation:")
        print(f"   Même coup: {'✅' if moves_match else '❌'}")
        print()
        
        # Conclusion
        if speed_ratio > 1.5:
            print(f"🎉 === SUCCÈS ===")
            print(f"✅ Optimisations très efficaces!")
            print(f"⚡ {speed_ratio:.1f}x plus rapide")
            print(f"💪 Peut explorer {node_ratio:.1f}x plus de nœuds")
        elif speed_ratio > 1.1:
            print(f"👍 === BON RÉSULTAT ===")
            print(f"✅ Optimisations efficaces")
            print(f"⚡ {speed_ratio:.1f}x plus rapide")
        else:
            print(f"🤔 === RÉSULTAT MITIGÉ ===")
            print(f"⚠️  Peu d'amélioration visible")

def main():
    """Fonction principale avec gestion d'erreurs"""
    try:
        quick_null_move_test()
    except KeyboardInterrupt:
        print("\n⚠️  Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
    finally:
        print("\n🏁 Test terminé")

if __name__ == "__main__":
    print("🔥 === TEST NULL-MOVE AVEC TIMEOUT STRICT === 🔥")
    print("🎯 Comparaison bitboards optimisés vs originaux")
    print("⏱️  60 secondes MAX par test (garanti)")
    print("=" * 60)
    print()
    
    main()