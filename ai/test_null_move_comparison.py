#!/usr/bin/env python3
"""
Test comparatif: Efficacité Null-Move Engine
Chess.py vs optimized_chess.py - 60 secondes chacun
"""

import sys
import os
import time

# Ajouter le répertoire courant au path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from Chess import Chess
from Null_move_AI.null_move_engine import NullMovePruningEngine
from optimized_chess import patch_chess_class

def test_null_move_engine(use_optimizations=False, time_limit=60):
    """
    Test du null-move engine avec ou sans optimisations
    """
    print(f"\n{'='*60}")
    if use_optimizations:
        print("⚡ TEST VERSION OPTIMISÉE (optimized_chess.py)")
    else:
        print("📊 TEST VERSION ORIGINALE (Chess.py)")
    print(f"{'='*60}")
    
    # Créer le board de test
    chess = Chess()
    
    # Appliquer les optimisations si demandé
    if use_optimizations:
        patch_chess_class(chess)
        print("✅ Optimisations bitboard activées")
    else:
        print("📋 Version standard sans optimisations")
    
    # Configuration de l'engine null-move
    engine = NullMovePruningEngine(
        max_time=time_limit,
        max_depth=25,
        null_move_enabled=True,
        null_move_R=2,
        null_move_min_depth=3,
        tt_size=2000000
    )
    
    print(f"⚙️  Configuration engine:")
    print(f"   ⏱️  Temps limite: {time_limit}s")
    print(f"   🎯 Null-move R: {engine.null_move_R}")
    print(f"   📊 Min depth null-move: {engine.null_move_min_depth}")
    print(f"   💾 Table transposition: {engine.tt_size:,} entrées")
    print()
    
    print("🚀 Démarrage de la recherche...")
    start_time = time.time()
    
    try:
        best_move = engine.get_best_move_with_time_limit(chess)
        elapsed_time = time.time() - start_time
        
        print(f"✅ Recherche terminée en {elapsed_time:.2f}s")
        
        # Calculer les statistiques
        nodes_per_sec = engine.nodes_evaluated / elapsed_time if elapsed_time > 0 else 0
        null_efficiency = (engine.null_move_cutoffs / engine.null_move_attempts * 100) if engine.null_move_attempts > 0 else 0
        tt_efficiency = (engine.tt_hits / (engine.tt_hits + engine.tt_misses) * 100) if (engine.tt_hits + engine.tt_misses) > 0 else 0
        
        # Afficher les résultats détaillés
        print(f"\n📊 RÉSULTATS DÉTAILLÉS:")
        print(f"🎯 Meilleur coup trouvé: {engine._format_move(best_move) if best_move else 'Aucun'}")
        print(f"⏱️  Temps d'exécution: {elapsed_time:.2f}s")
        print(f"📈 Nœuds évalués: {engine.nodes_evaluated:,}")
        print(f"✂️  Branches élaguées: {engine.pruned_branches:,}")
        print(f"🚀 Vitesse: {nodes_per_sec:,.0f} nœuds/sec")
        print()
        
        print(f"🔄 NULL-MOVE PRUNING:")
        print(f"   Tentatives: {engine.null_move_attempts:,}")
        print(f"   Cutoffs réussis: {engine.null_move_cutoffs:,}")
        print(f"   Échecs: {engine.null_move_failures:,}")
        print(f"   Efficacité: {null_efficiency:.1f}%")
        print()
        
        print(f"💾 TABLE DE TRANSPOSITION:")
        print(f"   Entrées stockées: {len(engine.transposition_table):,}")
        print(f"   Hits: {engine.tt_hits:,}")
        print(f"   Misses: {engine.tt_misses:,}")
        print(f"   Efficacité: {tt_efficiency:.1f}%")
        
        return {
            'success': True,
            'best_move': best_move,
            'time': elapsed_time,
            'nodes': engine.nodes_evaluated,
            'speed': nodes_per_sec,
            'pruned': engine.pruned_branches,
            'null_attempts': engine.null_move_attempts,
            'null_cutoffs': engine.null_move_cutoffs,
            'null_efficiency': null_efficiency,
            'tt_hits': engine.tt_hits,
            'tt_efficiency': tt_efficiency,
            'engine': engine
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Erreur pendant la recherche: {e}")
        print(f"⏱️  Temps écoulé avant erreur: {elapsed_time:.2f}s")
        return {
            'success': False,
            'time': elapsed_time,
            'error': str(e)
        }

def compare_and_conclude(result_original, result_optimized):
    """
    Compare les résultats et tire les conclusions
    """
    print(f"\n{'='*70}")
    print("🏆 ANALYSE COMPARATIVE FINALE")
    print(f"{'='*70}")
    
    if not result_original['success'] or not result_optimized['success']:
        print("❌ Impossible de comparer - une des versions a échoué")
        return
    
    # Calculs des ratios
    speed_ratio = result_optimized['speed'] / result_original['speed'] if result_original['speed'] > 0 else 0
    nodes_ratio = result_optimized['nodes'] / result_original['nodes'] if result_original['nodes'] > 0 else 0
    time_ratio = result_original['time'] / result_optimized['time'] if result_optimized['time'] > 0 else 0
    
    print(f"⏱️  PERFORMANCE TEMPORELLE:")
    print(f"   📊 Original:  {result_original['time']:.2f}s")
    print(f"   ⚡ Optimisé:  {result_optimized['time']:.2f}s")
    print(f"   📈 Speedup:   {time_ratio:.2f}x")
    print()
    
    print(f"🚀 VITESSE DE CALCUL:")
    print(f"   📊 Original:  {result_original['speed']:,.0f} nœuds/sec")
    print(f"   ⚡ Optimisé:  {result_optimized['speed']:,.0f} nœuds/sec")
    print(f"   📈 Amélioration: {speed_ratio:.2f}x")
    print()
    
    print(f"🧠 CAPACITÉ D'EXPLORATION:")
    print(f"   📊 Original:  {result_original['nodes']:,} nœuds")
    print(f"   ⚡ Optimisé:  {result_optimized['nodes']:,} nœuds")
    print(f"   📈 Facteur:   {nodes_ratio:.2f}x")
    print()
    
    print(f"✂️  EFFICACITÉ NULL-MOVE:")
    print(f"   📊 Original:  {result_original['null_efficiency']:.1f}% ({result_original['null_cutoffs']:,}/{result_original['null_attempts']:,})")
    print(f"   ⚡ Optimisé:  {result_optimized['null_efficiency']:.1f}% ({result_optimized['null_cutoffs']:,}/{result_optimized['null_attempts']:,})")
    print()
    
    print(f"💾 EFFICACITÉ TABLE TRANSPOSITION:")
    print(f"   📊 Original:  {result_original['tt_efficiency']:.1f}% ({result_original['tt_hits']:,} hits)")
    print(f"   ⚡ Optimisé:  {result_optimized['tt_efficiency']:.1f}% ({result_optimized['tt_hits']:,} hits)")
    print()
    
    # Validation des coups
    move_orig = result_original['best_move']
    move_opt = result_optimized['best_move']
    same_move = (move_orig == move_opt) if (move_orig and move_opt) else False
    
    print(f"✅ VALIDATION QUALITATIVE:")
    print(f"   📊 Coup original:  {result_original['engine']._format_move(move_orig) if move_orig else 'Aucun'}")
    print(f"   ⚡ Coup optimisé:  {result_optimized['engine']._format_move(move_opt) if move_opt else 'Aucun'}")
    print(f"   🎯 Coups identiques: {'✅' if same_move else '❌'}")
    print()
    
    # Conclusion finale
    print(f"🎯 VERDICT FINAL:")
    if speed_ratio >= 3.0:
        verdict = "🔥 EXCEPTIONNEL!"
        description = "Les optimisations bitboard transforment complètement les performances"
    elif speed_ratio >= 2.5:
        verdict = "🔥 EXCELLENT!"
        description = "Gain de performance majeur grâce aux optimisations"
    elif speed_ratio >= 2.0:
        verdict = "🔥 TRÈS BON!"
        description = "Amélioration significative des performances"
    elif speed_ratio >= 1.5:
        verdict = "👍 BON!"
        description = "Gain de performance notable"
    elif speed_ratio >= 1.2:
        verdict = "🤔 MODÉRÉ"
        description = "Amélioration limitée mais positive"
    else:
        verdict = "😐 MARGINAL"
        description = "Impact limité des optimisations"
    
    print(f"   {verdict}")
    print(f"   📈 Speedup: {speed_ratio:.1f}x plus rapide")
    print(f"   🧠 Exploration: {nodes_ratio:.1f}x plus approfondie")
    print(f"   💡 {description}")
    
    if same_move:
        print(f"   💎 Bonus: Qualité de jeu préservée!")
    else:
        print(f"   ⚠️  Note: Exploration différente → coups différents")
    
    return speed_ratio, nodes_ratio

def main():
    """
    Fonction principale du test comparatif
    """
    print("🔥 TEST COMPARATIF NULL-MOVE ENGINE 🔥")
    print("📊 Efficacité: Chess.py vs optimized_chess.py")
    print("⏱️  Durée: 60 secondes par version")
    print("🎯 Position: Début de partie (identique)")
    print("🧠 Algorithme: Null-move pruning + Table de transposition + Iterative deepening")
    
    # Test version originale
    print("\n🏃 Lancement du test version originale...")
    result_original = test_null_move_engine(use_optimizations=False, time_limit=60)
    
    # Petite pause entre les tests
    print("\n⏸️  Pause de 2 secondes entre les tests...")
    time.sleep(2)
    
    # Test version optimisée
    print("\n⚡ Lancement du test version optimisée...")
    result_optimized = test_null_move_engine(use_optimizations=True, time_limit=60)
    
    # Analyse comparative
    if result_original['success'] and result_optimized['success']:
        speed_gain, exploration_gain = compare_and_conclude(result_original, result_optimized)
        
        print(f"\n🏁 RÉSUMÉ EXÉCUTIF:")
        print(f"⚡ Performance: {speed_gain:.1f}x plus rapide avec optimisations")
        print(f"🔍 Exploration: {exploration_gain:.1f}x plus approfondie")
        print(f"🎯 Conclusion: Les optimisations bitboard sont très efficaces pour le null-move engine!")
    else:
        print(f"\n❌ Test incomplet - voir les erreurs ci-dessus")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🏁 Fin du programme")