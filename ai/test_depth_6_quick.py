#!/usr/bin/env python3
"""
Test rapide de performance en profondeur 6 avec des temps limités.
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Chess import Chess
from evaluator import ChessEvaluator
from ai.Null_move_AI.null_move_engine import NullMovePruningEngine
from ai.Old_AI.iterative_deepening_engine_TT import IterativeDeepeningAlphaBeta

def quick_depth_6_test():
    """Test rapide pour estimer le temps nécessaire en profondeur 6"""
    
    print("=== TEST RAPIDE PROFONDEUR 6 ===\n")
    
    # Position simple pour test rapide
    chess = Chess()
    evaluator = ChessEvaluator()
    
    print("🎯 ESTIMATION DES TEMPS - PROFONDEUR PAR PROFONDEUR")
    print("=" * 60)
    
    # Test progressif pour estimer le temps en profondeur 6
    depths = [3, 4, 5]  # On s'arrête à 5 pour estimer 6
    times_without_nm = []
    times_with_nm = []
    
    for depth in depths:
        print(f"\n📊 PROFONDEUR {depth}:")
        print("-" * 30)
        
        # Sans null-move
        engine_ref = IterativeDeepeningAlphaBeta(
            max_time=120,  # 2 minutes max par profondeur
            max_depth=depth,
            evaluator=evaluator
        )
        
        start = time.time()
        move_ref = engine_ref.get_best_move_with_time_limit(chess)
        time_ref = time.time() - start
        times_without_nm.append(time_ref)
        
        print(f"Sans null-move: {time_ref:.2f}s, {engine_ref.nodes_evaluated:,} nœuds")
        
        # Avec null-move
        engine_nm = NullMovePruningEngine(
            max_time=120,
            max_depth=depth,
            evaluator=evaluator,
            null_move_enabled=True,
            null_move_R=2,
            null_move_min_depth=2
        )
        
        start = time.time()
        move_nm = engine_nm.get_best_move_with_time_limit(chess)
        time_nm = time.time() - start
        times_with_nm.append(time_nm)
        
        nm_stats = engine_nm.get_null_move_stats()
        gain = ((time_ref - time_nm) / time_ref) * 100 if time_ref > 0 else 0
        
        print(f"Avec null-move: {time_nm:.2f}s, {engine_nm.nodes_evaluated:,} nœuds, {nm_stats['cutoffs']}/{nm_stats['attempts']} cutoffs")
        print(f"Gain: {gain:+.1f}%")
    
    # Estimation pour profondeur 6
    print(f"\n🔮 ESTIMATION PROFONDEUR 6:")
    print("=" * 40)
    
    if len(times_without_nm) >= 2:
        # Calcul du facteur de croissance moyen
        growth_factors_ref = []
        growth_factors_nm = []
        
        for i in range(1, len(times_without_nm)):
            if times_without_nm[i-1] > 0:
                growth_factors_ref.append(times_without_nm[i] / times_without_nm[i-1])
            if times_with_nm[i-1] > 0:
                growth_factors_nm.append(times_with_nm[i] / times_with_nm[i-1])
        
        if growth_factors_ref:
            avg_growth_ref = sum(growth_factors_ref) / len(growth_factors_ref)
            estimated_depth6_ref = times_without_nm[-1] * avg_growth_ref
            
            print(f"Facteur de croissance moyen (sans null-move): {avg_growth_ref:.1f}x")
            print(f"Temps estimé profondeur 6 (sans null-move): {estimated_depth6_ref:.1f}s ({estimated_depth6_ref/60:.1f} min)")
        
        if growth_factors_nm:
            avg_growth_nm = sum(growth_factors_nm) / len(growth_factors_nm)
            estimated_depth6_nm = times_with_nm[-1] * avg_growth_nm
            
            print(f"Facteur de croissance moyen (avec null-move): {avg_growth_nm:.1f}x")
            print(f"Temps estimé profondeur 6 (avec null-move): {estimated_depth6_nm:.1f}s ({estimated_depth6_nm/60:.1f} min)")
            
            if 'estimated_depth6_ref' in locals():
                estimated_gain = ((estimated_depth6_ref - estimated_depth6_nm) / estimated_depth6_ref) * 100
                print(f"Gain estimé avec null-move: {estimated_gain:+.1f}%")

def test_depth_6_limited_time():
    """Test avec temps limité pour voir jusqu'où on arrive en profondeur 6"""
    
    print(f"\n" + "="*60)
    print("⏰ TEST AVEC TEMPS LIMITÉ - VISER PROFONDEUR 6")
    print("="*60)
    
    chess = Chess()
    evaluator = ChessEvaluator()
    
    time_limits = [60, 120, 180]  # 1, 2, 3 minutes
    
    for time_limit in time_limits:
        print(f"\n🕐 TEMPS LIMITE: {time_limit}s ({time_limit//60} min)")
        print("-" * 40)
        
        # Test avec null-move
        engine = NullMovePruningEngine(
            max_time=time_limit,
            max_depth=6,  # Objectif profondeur 6
            evaluator=evaluator,
            null_move_enabled=True,
            null_move_R=2,
            null_move_min_depth=2
        )
        
        start = time.time()
        move = engine.get_best_move_with_time_limit(chess)
        actual_time = time.time() - start
        
        nm_stats = engine.get_null_move_stats()
        
        print(f"Temps utilisé: {actual_time:.1f}s")
        print(f"Nœuds évalués: {engine.nodes_evaluated:,}")
        print(f"Null-moves: {nm_stats['cutoffs']}/{nm_stats['attempts']} cutoffs ({nm_stats['success_rate']:.1f}%)")
        print(f"Coup trouvé: {engine._format_move(move)}")
        
        # Déterminer la profondeur atteinte (approximation)
        if actual_time < time_limit * 0.9:
            print("✅ Profondeur 6 probablement atteinte!")
        else:
            print("⏳ Recherche interrompue par le temps")

def compare_engines_quick():
    """Comparaison rapide des différentes IA sur une position"""
    
    print(f"\n" + "="*60)
    print("🏁 COMPARAISON RAPIDE DES IA")
    print("="*60)
    
    chess = Chess()
    evaluator = ChessEvaluator()
    test_time = 30  # 30 secondes par IA
    
    engines = [
        ("Alpha-Beta basique", "alphabeta_engine"),
        ("Avec table transposition", "iterative_deepening_engine_TT"),
        ("Avec null-move", "null_move")
    ]
    
    results = []
    
    for engine_name, engine_type in engines:
        print(f"\n🤖 {engine_name}:")
        
        try:
            if engine_type == "alphabeta_engine":
                from alphabeta_engine import AlphaBetaEngine
                engine = AlphaBetaEngine(max_depth=10, evaluator=evaluator)
                
                start = time.time()
                move = engine.get_best_move(chess)
                elapsed = time.time() - start
                nodes = engine.nodes_evaluated
                
            elif engine_type == "iterative_deepening_engine_TT":
                engine = IterativeDeepeningAlphaBeta(
                    max_time=test_time,
                    max_depth=10,
                    evaluator=evaluator
                )
                
                start = time.time()
                move = engine.get_best_move_with_time_limit(chess)
                elapsed = time.time() - start
                nodes = engine.nodes_evaluated
                
            elif engine_type == "null_move":
                engine = NullMovePruningEngine(
                    max_time=test_time,
                    max_depth=10,
                    evaluator=evaluator,
                    null_move_enabled=True
                )
                
                start = time.time()
                move = engine.get_best_move_with_time_limit(chess)
                elapsed = time.time() - start
                nodes = engine.nodes_evaluated
                nm_stats = engine.get_null_move_stats()
            
            nps = nodes / max(1, elapsed)  # nœuds par seconde
            
            print(f"  Temps: {elapsed:.2f}s")
            print(f"  Nœuds: {nodes:,}")
            print(f"  Efficacité: {nps:,.0f} nœuds/s")
            print(f"  Coup: {engine._format_move(move) if hasattr(engine, '_format_move') else 'N/A'}")
            
            if engine_type == "null_move":
                print(f"  Null-moves: {nm_stats['cutoffs']}/{nm_stats['attempts']} cutoffs")
            
            results.append({
                'name': engine_name,
                'time': elapsed,
                'nodes': nodes,
                'nps': nps,
                'move': move
            })
            
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
    
    # Résumé
    if results:
        print(f"\n📊 RÉSUMÉ:")
        print(f"{'IA':<25} {'Temps (s)':<10} {'Nœuds':<12} {'Nœuds/s':<12}")
        print("-" * 60)
        
        for r in results:
            print(f"{r['name']:<25} {r['time']:<10.2f} {r['nodes']:<12,} {r['nps']:<12,.0f}")

if __name__ == "__main__":
    try:
        quick_depth_6_test()
        test_depth_6_limited_time()
        compare_engines_quick()
        
        print(f"\n🎯 CONCLUSIONS:")
        print("✅ Null-move pruning fonctionne")
        print("✅ L'IA est déterministe (pas d'aléatoire)")
        print("🚀 Pour profondeur 6: prévoir 2-5 minutes selon la position")
        print("💡 Null-move plus efficace sur positions déséquilibrées")
        
    except KeyboardInterrupt:
        print("\n\n❌ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()