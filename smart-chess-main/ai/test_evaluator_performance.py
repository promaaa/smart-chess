#!/usr/bin/env python3
"""
Test de performance des évaluateurs - Compare l'évaluateur original vs optimisé
"""

import time
import sys
import os

# Ajouter le path pour importer les modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from Chess import Chess
from evaluator import ChessEvaluator as OriginalEvaluator
from fast_evaluator import FastChessEvaluator, SuperFastChessEvaluator

def create_test_position():
    """Crée une position de test avec plusieurs pièces"""
    chess = Chess()
    
    # Position après quelques coups d'ouverture
    moves = [
        (12, 28, None),  # e2-e4
        (52, 36, None),  # e7-e5  
        (6, 21, None),   # Ng1-f3
        (57, 42, None),  # Nb8-c6
        (5, 33, None),   # Bf1-b5
        (50, 34, None),  # a7-a6
    ]
    
    for from_sq, to_sq, promotion in moves:
        chess.move_piece(from_sq, to_sq, promotion)
    
    return chess

def benchmark_evaluator(evaluator, name, chess, num_evaluations=10000):
    """Teste la performance d'un évaluateur"""
    print(f"\n🔍 Test de {name}")
    print("-" * 40)
    
    # Préchauffage
    for _ in range(100):
        evaluator.evaluate_position(chess)
    
    # Test chronométré
    start_time = time.time()
    
    for _ in range(num_evaluations):
        score = evaluator.evaluate_position(chess)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Statistiques
    evals_per_sec = num_evaluations / elapsed
    time_per_eval = (elapsed / num_evaluations) * 1000000  # microsecondes
    
    print(f"⏱️  Temps total: {elapsed:.3f}s")
    print(f"🚀 Évaluations/sec: {evals_per_sec:,.0f}")
    print(f"⚡ Temps par éval: {time_per_eval:.1f}μs")
    print(f"📊 Score final: {score}")
    
    return {
        'name': name,
        'time': elapsed,
        'evals_per_sec': evals_per_sec,
        'time_per_eval': time_per_eval,
        'score': score
    }

def main():
    print("🎯 Test de Performance des Évaluateurs Chess AI")
    print("=" * 60)
    
    # Appliquer les optimisations bitboard si possible
    try:
        from optimized_chess import patch_chess_class_globally
        patch_chess_class_globally()
        print("✅ Optimisations bitboard appliquées")
    except:
        print("⚠️  Optimisations bitboard non disponibles")
    
    # Créer la position de test
    chess = create_test_position()
    print(f"\n📋 Position de test:")
    chess.print_board()
    
    # Nombre d'évaluations pour le test
    num_evals = 50000
    print(f"\n🔢 Nombre d'évaluations par test: {num_evals:,}")
    
    results = []
    
    # Test 1: Évaluateur original
    print("\n" + "=" * 60)
    original_eval = OriginalEvaluator()
    result1 = benchmark_evaluator(original_eval, "Évaluateur Original", chess, num_evals)
    results.append(result1)
    
    # Test 2: Évaluateur optimisé
    print("\n" + "=" * 60)
    fast_eval = FastChessEvaluator()
    result2 = benchmark_evaluator(fast_eval, "Évaluateur Optimisé", chess, num_evals)
    results.append(result2)
    
    # Test 3: Évaluateur ultra-rapide
    print("\n" + "=" * 60)
    super_fast_eval = SuperFastChessEvaluator()
    result3 = benchmark_evaluator(super_fast_eval, "Évaluateur Ultra-Rapide", chess, num_evals)
    results.append(result3)
    
    # Analyse comparative
    print("\n" + "=" * 80)
    print("📊 ANALYSE COMPARATIVE")
    print("=" * 80)
    
    print(f"\n{'Évaluateur':<20} {'Temps (s)':<12} {'Éval/sec':<15} {'μs/éval':<12} {'Score':<10}")
    print("-" * 75)
    
    for result in results:
        print(f"{result['name']:<20} {result['time']:<12.3f} {result['evals_per_sec']:<15,.0f} "
              f"{result['time_per_eval']:<12.1f} {result['score']:<10.0f}")
    
    # Calcul des améliorations
    if len(results) >= 2:
        original_speed = results[0]['evals_per_sec']
        optimized_speed = results[1]['evals_per_sec']
        super_fast_speed = results[2]['evals_per_sec']
        
        print(f"\n🚀 AMÉLIORATIONS DE PERFORMANCE:")
        print(f"   Optimisé vs Original: {optimized_speed/original_speed:.1f}x plus rapide")
        print(f"   Ultra-rapide vs Original: {super_fast_speed/original_speed:.1f}x plus rapide")
        print(f"   Ultra-rapide vs Optimisé: {super_fast_speed/optimized_speed:.1f}x plus rapide")
        
        # Vérification de la cohérence des scores
        original_score = results[0]['score']
        optimized_score = results[1]['score']
        
        if abs(original_score - optimized_score) < 50:  # Tolérance de 50 centipions
            print(f"\n✅ Cohérence des scores: OK (diff = {abs(original_score - optimized_score):.0f})")
        else:
            print(f"\n⚠️  Différence de score significative: {abs(original_score - optimized_score):.0f}")
    
    print(f"\n💡 RECOMMANDATION:")
    if results[1]['evals_per_sec'] > results[0]['evals_per_sec'] * 2:
        print("   🥇 Utilisez FastChessEvaluator pour de meilleures performances")
    elif results[2]['evals_per_sec'] > results[0]['evals_per_sec'] * 5:
        print("   🥇 SuperFastChessEvaluator recommandé pour maximum de vitesse")
    else:
        print("   ⚖️  L'évaluateur original reste viable")

if __name__ == "__main__":
    main()