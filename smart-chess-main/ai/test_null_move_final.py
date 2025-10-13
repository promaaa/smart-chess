#!/usr/bin/env python3
"""
Test comparatif simple de l'engine Null-Move 
Temps limité à 30s pour éviter les débordements
"""

import sys
import os
import time

# Ajouter les paths nécessaires
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chess import Chess
from Null_move_AI.null_move_engine import NullMovePruningEngine

# Import des optimisations depuis le dossier Profile
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Profile'))
from optimized_chess import patch_chess_class

def test_null_move_simple():
    """
    Test simple et rapide de comparaison
    """
    print("⚡ === TEST SIMPLE NULL-MOVE === ⚡")
    print("⏱️  Limite: 30 secondes par test")
    print("🎯 Position: Début de partie")
    print()
    
    max_time = 30.0  # 30 secondes pour éviter les débordements
    
    # ========== TEST 1: VERSION ORIGINALE ==========
    print("🏃 === TEST 1: VERSION ORIGINALE ===")
    
    chess_original = Chess()
    engine_original = NullMovePruningEngine(
        max_time=max_time,
        max_depth=15,  # Profondeur raisonnable
        null_move_enabled=True,
        null_move_R=2,
        null_move_min_depth=3,
        tt_size=500000  # Table plus petite
    )
    
    print("🚀 Démarrage recherche originale...")
    start_time = time.time()
    
    try:
        best_move_original = engine_original.get_best_move_with_time_limit(chess_original)
        original_time = time.time() - start_time
        
        print(f"✅ Recherche terminée en {original_time:.2f}s")
        print(f"🎯 Coup: {engine_original._format_move(best_move_original) if best_move_original else 'None'}")
        print(f"📊 Nœuds: {engine_original.nodes_evaluated:,}")
        print(f"🚀 Vitesse: {engine_original.nodes_evaluated/original_time:,.0f} nœuds/sec")
        print(f"✂️  Null: {engine_original.null_move_cutoffs:,}/{engine_original.null_move_attempts:,} cutoffs")
        print()
        
    except Exception as e:
        print(f"❌ Erreur originale: {e}")
        return
    
    # ========== TEST 2: VERSION OPTIMISÉE ==========
    print("⚡ === TEST 2: VERSION OPTIMISÉE ===")
    
    chess_optimized = Chess()
    patch_chess_class(chess_optimized)  # Appliquer optimisations
    
    engine_optimized = NullMovePruningEngine(
        max_time=max_time,
        max_depth=15,
        null_move_enabled=True,
        null_move_R=2,
        null_move_min_depth=3,
        tt_size=500000
    )
    
    print("🚀 Démarrage recherche optimisée...")
    start_time = time.time()
    
    try:
        best_move_optimized = engine_optimized.get_best_move_with_time_limit(chess_optimized)
        optimized_time = time.time() - start_time
        
        print(f"✅ Recherche terminée en {optimized_time:.2f}s")
        print(f"🎯 Coup: {engine_optimized._format_move(best_move_optimized) if best_move_optimized else 'None'}")
        print(f"📊 Nœuds: {engine_optimized.nodes_evaluated:,}")
        print(f"🚀 Vitesse: {engine_optimized.nodes_evaluated/optimized_time:,.0f} nœuds/sec")
        print(f"✂️  Null: {engine_optimized.null_move_cutoffs:,}/{engine_optimized.null_move_attempts:,} cutoffs")
        print()
        
    except Exception as e:
        print(f"❌ Erreur optimisée: {e}")
        return
    
    # ========== COMPARAISON ==========
    print("🏆 === RÉSULTATS ===")
    print("-" * 50)
    
    if original_time > 0 and optimized_time > 0:
        time_speedup = original_time / optimized_time
        speed_orig = engine_original.nodes_evaluated / original_time
        speed_opt = engine_optimized.nodes_evaluated / optimized_time
        speed_speedup = speed_opt / speed_orig if speed_orig > 0 else 1
        
        print(f"⏱️  Temps de calcul:")
        print(f"   Original:  {original_time:.2f}s")
        print(f"   Optimisé:  {optimized_time:.2f}s")
        print(f"   Speedup:   {time_speedup:.2f}x")
        print()
        
        print(f"🚀 Vitesse (nœuds/sec):")
        print(f"   Original:  {speed_orig:,.0f}")
        print(f"   Optimisé:  {speed_opt:,.0f}")
        print(f"   Speedup:   {speed_speedup:.2f}x")
        print()
        
        print(f"📊 Exploration:")
        print(f"   Original:  {engine_original.nodes_evaluated:,} nœuds")
        print(f"   Optimisé:  {engine_optimized.nodes_evaluated:,} nœuds")
        node_ratio = engine_optimized.nodes_evaluated / engine_original.nodes_evaluated if engine_original.nodes_evaluated > 0 else 1
        print(f"   Ratio:     {node_ratio:.2f}x")
        print()
        
        # Validation des coups
        moves_match = (best_move_original == best_move_optimized) if (best_move_original and best_move_optimized) else False
        print(f"✅ Validation:")
        print(f"   Même coup optimal: {'✅' if moves_match else '❌'}")
        print(f"   Original:  {engine_original._format_move(best_move_original) if best_move_original else 'None'}")
        print(f"   Optimisé:  {engine_optimized._format_move(best_move_optimized) if best_move_optimized else 'None'}")
        print()
        
        # Conclusion
        print("🎯 === CONCLUSION ===")
        if speed_speedup >= 2.0:
            print(f"🔥 EXCELLENT! Speedup de {speed_speedup:.1f}x")
            print(f"✅ Les optimisations bitboard sont très efficaces!")
        elif speed_speedup >= 1.5:
            print(f"👍 BON! Speedup de {speed_speedup:.1f}x")
            print(f"✅ Les optimisations apportent un gain significatif")
        elif speed_speedup >= 1.2:
            print(f"🤔 MOYEN. Speedup de {speed_speedup:.1f}x")
            print(f"⚡ Amélioration modeste mais positive")
        else:
            print(f"😐 LIMITÉ. Speedup de {speed_speedup:.1f}x")
            print(f"🔍 Les optimisations n'ont pas beaucoup d'impact ici")
        
        if moves_match:
            print(f"💎 Qualité de jeu préservée (même coup trouvé)")
        else:
            print(f"⚠️  Coups différents - exploration différente")

if __name__ == "__main__":
    print("🏁 === TEST NULL-MOVE: OPTIMISATIONS BITBOARD === 🏁")
    print("🎯 Comparaison performance avec/sans optimisations")
    print("⏱️  Limite: 30s par test pour éviter les débordements")
    print("=" * 60)
    print()
    
    try:
        test_null_move_simple()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Fin du test")