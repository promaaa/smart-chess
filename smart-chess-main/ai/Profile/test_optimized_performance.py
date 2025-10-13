#!/usr/bin/env python3
"""
Test de performance avec Chess.py optimisé vs original
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Chess import Chess
from alphabeta_engine import AlphaBetaEngine  
from ai.Profile.optimized_chess import patch_chess_class, unpatch_chess_class
import time
import cProfile
import pstats

def test_engine_performance():
    """
    Compare les performances de l'engine avec et sans optimisations
    """
    print("🏁 === TEST PERFORMANCE ENGINE OPTIMISÉ === 🏁")
    
    # Configuration
    depth = 3
    chess = Chess()
    engine = AlphaBetaEngine(max_depth=depth)
    
    print(f"⚙️  Configuration: Profondeur {depth}, Position initiale")
    print("🔄 Test en cours...\n")
    
    # Test 1: Version originale
    print("📊 === TEST 1: VERSION ORIGINALE ===")
    chess = Chess()  # Reset
    engine = AlphaBetaEngine(max_depth=depth)
    
    start_time = time.time()
    profiler_orig = cProfile.Profile()
    profiler_orig.enable()
    
    best_move = engine.get_best_move(chess)
    nodes_orig = engine.nodes_evaluated
    
    profiler_orig.disable()
    orig_time = time.time() - start_time
    
    print(f"⏱️  Temps: {orig_time:.3f}s")
    print(f"📊 Nœuds: {nodes_orig}")
    print(f"🎯 Coup: {best_move}")
    print(f"⚡ Nœuds/sec: {nodes_orig/orig_time:.0f}")
    
    # Test 2: Version optimisée  
    print("\n📊 === TEST 2: VERSION OPTIMISÉE ===")
    chess = Chess()  # Reset
    engine = AlphaBetaEngine(max_depth=depth)
    patch_chess_class(chess)  # Appliquer les optimisations
    
    start_time = time.time()
    profiler_opt = cProfile.Profile()
    profiler_opt.enable()
    
    best_move = engine.get_best_move(chess)
    nodes_opt = engine.nodes_evaluated
    
    profiler_opt.disable()
    opt_time = time.time() - start_time
    
    print(f"⏱️  Temps: {opt_time:.3f}s")
    print(f"📊 Nœuds: {nodes_opt}")
    print(f"🎯 Coup: {best_move}")
    print(f"⚡ Nœuds/sec: {nodes_opt/opt_time:.0f}")
    
    # Comparaison
    print(f"\n🏆 === RÉSULTATS COMPARATIFS ===")
    print(f"📈 Speedup temps: {orig_time/opt_time:.1f}x")
    print(f"📈 Speedup nœuds/sec: {(nodes_opt/opt_time)/(nodes_orig/orig_time):.1f}x")
    print(f"✅ Nœuds identiques: {nodes_orig == nodes_opt}")
    
    # Analyse détaillée des hotspots optimisés
    print(f"\n🔍 === ANALYSE DES HOTSPOTS OPTIMISÉS ===")
    
    # Statistiques originales
    ps_orig = pstats.Stats(profiler_orig)
    ps_orig.sort_stats('cumulative')
    
    # Statistiques optimisées
    ps_opt = pstats.Stats(profiler_opt)
    ps_opt.sort_stats('cumulative')
    
    # Analyse des fonctions clés
    key_functions = ['square_mask', 'pieces_of_color', 'occupancy', 'is_in_check']
    
    print("Fonction                Appels Orig    Temps Orig   Appels Opt   Temps Opt   Gain")
    print("-" * 85)
    
    for func_name in key_functions:
        orig_stats = find_function_stats(ps_orig.stats, func_name)
        opt_stats = find_function_stats(ps_opt.stats, func_name)
        
        if orig_stats and opt_stats:
            orig_calls, orig_time = orig_stats
            opt_calls, opt_time = opt_stats
            gain = orig_time / opt_time if opt_time > 0 else float('inf')
            
            print(f"{func_name:20} {orig_calls:10} {orig_time:10.3f}s {opt_calls:10} {opt_time:10.3f}s {gain:8.1f}x")
    
    return orig_time, opt_time, nodes_orig, nodes_opt

def find_function_stats(stats_dict, func_name):
    """
    Trouve les statistiques d'une fonction dans le profiler
    """
    for key, value in stats_dict.items():
        if func_name in str(key):
            calls = value[0]  # cc (call count)
            tottime = value[1]  # tt (total time excluding subcalls)  
            return calls, tottime
    return None

def profile_optimized_hotspots():
    """
    Profile spécifiquement les fonctions optimisées
    """
    print("\n🎯 === PROFILING DES FONCTIONS OPTIMISÉES ===")
    
    chess = Chess()
    patch_chess_class(chess)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Simulation d'utilisation intensive des fonctions optimisées
    for _ in range(1000):
        for sq in range(64):
            chess.square_mask(sq)
        
        chess.pieces_of_color(True)
        chess.pieces_of_color(False)
        chess.occupancy()
    
    profiler.disable()
    
    ps = pstats.Stats(profiler)
    ps.sort_stats('cumulative')
    
    print("Top 10 fonctions par temps cumulé:")
    ps.print_stats(10)

if __name__ == "__main__":
    orig_time, opt_time, nodes_orig, nodes_opt = test_engine_performance()
    
    # Test supplémentaire de profiling
    profile_optimized_hotspots()
    
    print(f"\n🎉 === RÉSUMÉ FINAL ===")
    print(f"⚡ Gain de performance: {orig_time/opt_time:.1f}x plus rapide")
    print(f"✅ Résultats identiques: {nodes_orig == nodes_opt}")
    print(f"🏆 Optimisations réussies!")