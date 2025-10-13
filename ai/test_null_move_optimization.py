#!/usr/bin/env python3
"""
Test comparatif de l'engine Null-Move avec et sans optimisations bitboard
Temps maximum: 60 secondes par test
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
from optimized_chess import patch_chess_class, unpatch_chess_class

def test_null_move_performance():
    """
    Test comparatif de l'engine null-move avec/sans optimisations bitboard
    """
    print("🔍 === TEST COMPARATIF NULL-MOVE ENGINE === 🔍")
    print("⏱️  Temps maximum par test: 60 secondes")
    print("🎯 Position: Début de partie")
    print()
    
    # Configuration
    max_time = 60.0  # 60 secondes
    
    # Position de test (début de partie)
    chess_original = Chess()
    chess_optimized = Chess()
    
    print("📋 === CONFIGURATION ===")
    print(f"⏱️  Temps limite: {max_time}s")
    print(f"🔧 Null-move R: 2 (réduction de profondeur)")
    print(f"📊 Null-move min depth: 3")
    print(f"💾 Table de transposition: 1M entrées")
    print()
    
    # ========== TEST 1: VERSION ORIGINALE ==========
    print("🏃 === TEST 1: VERSION ORIGINALE (bitboards normaux) ===")
    
    engine_original = NullMovePruningEngine(
        max_time=max_time,
        max_depth=20,  # Profondeur élevée pour utiliser le temps
        null_move_enabled=True,
        null_move_R=2,
        null_move_min_depth=3,
        tt_size=1000000
    )
    
    start_time = time.time()
    print(f"🚀 Démarrage recherche originale...")
    
    best_move_original = engine_original.get_best_move_with_time_limit(chess_original)
    
    original_time = time.time() - start_time
    
    # Statistiques originales
    print(f"✅ Recherche terminée!")
    print(f"⏱️  Temps total: {original_time:.2f}s")
    print(f"🎯 Meilleur coup: {engine_original._format_move(best_move_original) if best_move_original else 'None'}")
    print(f"📊 Nœuds évalués: {engine_original.nodes_evaluated:,}")
    print(f"✂️  Branches élaguées: {engine_original.pruned_branches:,}")
    print(f"🔄 Null-move tentatives: {engine_original.null_move_attempts:,}")
    print(f"✂️  Null-move cutoffs: {engine_original.null_move_cutoffs:,}")
    print(f"❌ Null-move échecs: {engine_original.null_move_failures:,}")
    print(f"💾 Entrées TT: {len(engine_original.transposition_table):,}")
    print(f"📈 TT hits: {engine_original.tt_hits:,}")
    print(f"📉 TT misses: {engine_original.tt_misses:,}")
    
    if engine_original.tt_hits + engine_original.tt_misses > 0:
        tt_efficiency_orig = engine_original.tt_hits / (engine_original.tt_hits + engine_original.tt_misses) * 100
        print(f"⚡ TT efficacité: {tt_efficiency_orig:.1f}%")
    
    if engine_original.null_move_attempts > 0:
        null_efficiency_orig = engine_original.null_move_cutoffs / engine_original.null_move_attempts * 100
        print(f"🎯 Null-move efficacité: {null_efficiency_orig:.1f}%")
    
    nodes_per_sec_orig = engine_original.nodes_evaluated / original_time if original_time > 0 else 0
    print(f"🚀 Vitesse: {nodes_per_sec_orig:,.0f} nœuds/sec")
    print()
    
    # ========== TEST 2: VERSION OPTIMISÉE ==========
    print("⚡ === TEST 2: VERSION OPTIMISÉE (bitboards optimisés) ===")
    
    engine_optimized = NullMovePruningEngine(
        max_time=max_time,
        max_depth=20,  # Même configuration
        null_move_enabled=True,
        null_move_R=2,
        null_move_min_depth=3,
        tt_size=1000000
    )
    
    # Appliquer les optimisations bitboard
    patch_chess_class(chess_optimized)
    
    start_time = time.time()
    print(f"🚀 Démarrage recherche optimisée...")
    
    best_move_optimized = engine_optimized.get_best_move_with_time_limit(chess_optimized)
    
    optimized_time = time.time() - start_time
    
    # Statistiques optimisées
    print(f"✅ Recherche terminée!")
    print(f"⏱️  Temps total: {optimized_time:.2f}s")
    print(f"🎯 Meilleur coup: {engine_optimized._format_move(best_move_optimized) if best_move_optimized else 'None'}")
    print(f"📊 Nœuds évalués: {engine_optimized.nodes_evaluated:,}")
    print(f"✂️  Branches élaguées: {engine_optimized.pruned_branches:,}")
    print(f"🔄 Null-move tentatives: {engine_optimized.null_move_attempts:,}")
    print(f"✂️  Null-move cutoffs: {engine_optimized.null_move_cutoffs:,}")
    print(f"❌ Null-move échecs: {engine_optimized.null_move_failures:,}")
    print(f"💾 Entrées TT: {len(engine_optimized.transposition_table):,}")
    print(f"📈 TT hits: {engine_optimized.tt_hits:,}")
    print(f"📉 TT misses: {engine_optimized.tt_misses:,}")
    
    if engine_optimized.tt_hits + engine_optimized.tt_misses > 0:
        tt_efficiency_opt = engine_optimized.tt_hits / (engine_optimized.tt_hits + engine_optimized.tt_misses) * 100
        print(f"⚡ TT efficacité: {tt_efficiency_opt:.1f}%")
    
    if engine_optimized.null_move_attempts > 0:
        null_efficiency_opt = engine_optimized.null_move_cutoffs / engine_optimized.null_move_attempts * 100
        print(f"🎯 Null-move efficacité: {null_efficiency_opt:.1f}%")
    
    nodes_per_sec_opt = engine_optimized.nodes_evaluated / optimized_time if optimized_time > 0 else 0
    print(f"🚀 Vitesse: {nodes_per_sec_opt:,.0f} nœuds/sec")
    print()
    
    # ========== COMPARAISON FINALE ==========
    print("🏆 === COMPARAISON FINALE ===")
    print("-" * 60)
    
    if original_time > 0 and optimized_time > 0:
        time_ratio = original_time / optimized_time
        print(f"⏱️  Temps de calcul:")
        print(f"   📊 Original:  {original_time:.2f}s")
        print(f"   ⚡ Optimisé:  {optimized_time:.2f}s")
        print(f"   📈 Speedup:   {time_ratio:.2f}x")
        print()
        
        print(f"🚀 Vitesse (nœuds/sec):")
        print(f"   📊 Original:  {nodes_per_sec_orig:,.0f}")
        print(f"   ⚡ Optimisé:  {nodes_per_sec_opt:.0f}")
        if nodes_per_sec_orig > 0:
            speed_ratio = nodes_per_sec_opt / nodes_per_sec_orig
            print(f"   📈 Amélioration: {speed_ratio:.2f}x")
        print()
        
        print(f"📊 Exploration:")
        print(f"   📊 Original:  {engine_original.nodes_evaluated:,} nœuds")
        print(f"   ⚡ Optimisé:  {engine_optimized.nodes_evaluated:,} nœuds")
        
        if engine_original.nodes_evaluated > 0:
            node_ratio = engine_optimized.nodes_evaluated / engine_original.nodes_evaluated
            print(f"   📈 Ratio nœuds: {node_ratio:.2f}x")
        print()
        
        print(f"✂️  Null-move pruning:")
        print(f"   📊 Original:  {engine_original.null_move_cutoffs:,} cutoffs / {engine_original.null_move_attempts:,} tentatives")
        print(f"   ⚡ Optimisé:  {engine_optimized.null_move_cutoffs:,} cutoffs / {engine_optimized.null_move_attempts:,} tentatives")
        print()
        
        print(f"💾 Table de transposition:")
        print(f"   📊 Original:  {engine_original.tt_hits:,} hits, {tt_efficiency_orig:.1f}% efficacité")
        print(f"   ⚡ Optimisé:  {engine_optimized.tt_hits:,} hits, {tt_efficiency_opt:.1f}% efficacité")
        print()
        
        # Validation
        print(f"✅ Validation:")
        moves_match = (best_move_original == best_move_optimized) if (best_move_original and best_move_optimized) else False
        print(f"   🎯 Même meilleur coup: {'✅' if moves_match else '❌'}")
        print(f"   📊 Original:  {engine_original._format_move(best_move_original) if best_move_original else 'None'}")
        print(f"   ⚡ Optimisé:  {engine_optimized._format_move(best_move_optimized) if best_move_optimized else 'None'}")
        print()
        
        # Conclusion
        if time_ratio > 1.1:
            print(f"🎉 === CONCLUSION ===")
            print(f"✅ Les optimisations bitboard améliorent les performances!")
            print(f"⚡ Gain de temps: {((time_ratio - 1) * 100):.1f}%")
            print(f"🚀 Vitesse multipliée par {speed_ratio:.1f}x")
            if moves_match:
                print(f"💎 Qualité de jeu préservée (même coup optimal)")
            else:
                print(f"⚠️  Coup différent - possiblement due à l'exploration plus profonde")
        else:
            print(f"📊 === CONCLUSION ===")
            print(f"🤔 Gain marginal ou pas de différence significative")
            print(f"⚡ Les optimisations n'impactent peut-être pas assez à cette profondeur")

def format_time(seconds):
    """Formatte le temps en format lisible"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.1f}s"

if __name__ == "__main__":
    print("🔥 === TEST NULL-MOVE ENGINE: BITBOARDS OPTIMISÉS === 🔥")
    print("🎯 Objectif: Mesurer l'impact des optimisations sur le null-move pruning")
    print("⏱️  Durée: Maximum 60s par test")
    print("=" * 70)
    print()
    
    test_null_move_performance()
    
    print()
    print("=" * 70)
    print("🏁 Test terminé!")