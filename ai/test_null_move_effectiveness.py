#!/usr/bin/env python3
"""
Test comparatif de l'efficacité du Null-Move Engine
Chess.py original vs optimized_chess.py
Temps: 60 secondes par test sur une position identique
"""

import sys
import os
import time
import copy

# Ajouter les paths nécessaires
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from Chess import Chess
from Null_move_AI.null_move_engine import NullMovePruningEngine

# Import des optimisations depuis le dossier Profile
profile_dir = os.path.join(current_dir, 'Profile')
sys.path.append(profile_dir)
from optimized_chess import patch_chess_class, unpatch_chess_class

def create_test_position():
    """
    Crée une position de test intéressante pour le null-move
    Position d'ouverture après quelques coups
    """
    chess = Chess()
    
    # Jouer quelques coups pour avoir une position plus complexe
    # 1. e4 e5 2. Nf3 Nc6 3. Bb5 (Ruy Lopez)
    moves = [
        (12, 28, None),  # e2-e4
        (52, 36, None),  # e7-e5  
        (6, 21, None),   # Ng1-f3
        (57, 42, None),  # Nb8-c6
        (5, 33, None),   # Bf1-b5
    ]
    
    print("🎯 Position de test: Ruy Lopez après 3. Bb5")
    print("Coups joués: 1.e4 e5 2.Nf3 Nc6 3.Bb5")
    
    for from_sq, to_sq, promotion in moves:
        try:
            chess.move_piece(from_sq, to_sq, promotion)
        except Exception as e:
            print(f"⚠️  Erreur dans la séquence: {e}")
            return Chess()  # Retourner position initiale si erreur
    
    return chess

def run_null_move_test(chess_board, test_name, use_optimizations=False, time_limit=60):
    """
    Lance un test du null-move engine
    """
    print(f"\n{'='*60}")
    print(f"🔍 {test_name}")
    print(f"{'='*60}")
    
    # Créer une copie indépendante du board
    if use_optimizations:
        # Appliquer les optimisations
        patch_chess_class(chess_board)
        print("⚡ Optimisations bitboard activées")
    else:
        print("📊 Version originale (sans optimisations)")
    
    # Configuration de l'engine
    engine = NullMovePruningEngine(
        max_time=time_limit,
        max_depth=25,  # Profondeur élevée pour utiliser tout le temps
        null_move_enabled=True,
        null_move_R=2,
        null_move_min_depth=3,
        tt_size=2000000  # Grande table de transposition
    )
    
    print(f"⚙️  Configuration:")
    print(f"   ⏱️  Temps limite: {time_limit}s")
    print(f"   🎯 Null-move R: {engine.null_move_R}")
    print(f"   📊 Profondeur max: {engine.max_depth}")
    print(f"   💾 Table TT: {engine.tt_size:,} entrées")
    print()
    
    # Lancement du test
    print("🚀 Démarrage de la recherche...")
    start_time = time.time()
    
    try:
        best_move = engine.get_best_move_with_time_limit(chess_board)
        actual_time = time.time() - start_time
        
        print(f"✅ Recherche terminée!")
        
    except Exception as e:
        actual_time = time.time() - start_time
        print(f"❌ Erreur pendant la recherche: {e}")
        best_move = None
    
    # Collecte des statistiques
    stats = {
        'best_move': best_move,
        'time': actual_time,
        'nodes_evaluated': engine.nodes_evaluated,
        'pruned_branches': engine.pruned_branches,
        'null_move_attempts': engine.null_move_attempts,
        'null_move_cutoffs': engine.null_move_cutoffs,
        'null_move_failures': engine.null_move_failures,
        'tt_entries': len(engine.transposition_table),
        'tt_hits': engine.tt_hits,
        'tt_misses': engine.tt_misses,
        'engine': engine
    }
    
    # Affichage des résultats
    print(f"\n📊 STATISTIQUES {test_name.upper()}:")
    print(f"⏱️  Temps réel: {actual_time:.2f}s")
    print(f"🎯 Meilleur coup: {engine._format_move(best_move) if best_move else 'Aucun'}")
    print(f"📈 Nœuds évalués: {stats['nodes_evaluated']:,}")
    print(f"✂️  Branches élaguées: {stats['pruned_branches']:,}")
    print(f"🚀 Vitesse: {stats['nodes_evaluated']/actual_time:,.0f} nœuds/sec")
    print()
    
    # Null-move statistics
    if stats['null_move_attempts'] > 0:
        null_efficiency = (stats['null_move_cutoffs'] / stats['null_move_attempts']) * 100
        print(f"🔄 NULL-MOVE PRUNING:")
        print(f"   Tentatives: {stats['null_move_attempts']:,}")
        print(f"   Cutoffs: {stats['null_move_cutoffs']:,}")
        print(f"   Échecs: {stats['null_move_failures']:,}")
        print(f"   Efficacité: {null_efficiency:.1f}%")
    else:
        print(f"🔄 NULL-MOVE: Aucune tentative")
    
    print()
    
    # Table de transposition
    if stats['tt_hits'] + stats['tt_misses'] > 0:
        tt_efficiency = (stats['tt_hits'] / (stats['tt_hits'] + stats['tt_misses'])) * 100
        print(f"💾 TABLE DE TRANSPOSITION:")
        print(f"   Entrées stockées: {stats['tt_entries']:,}")
        print(f"   Hits: {stats['tt_hits']:,}")
        print(f"   Misses: {stats['tt_misses']:,}")
        print(f"   Efficacité: {tt_efficiency:.1f}%")
    else:
        print(f"💾 TABLE DE TRANSPOSITION: Pas d'accès")
    
    return stats

def compare_results(stats_original, stats_optimized):
    """
    Compare et affiche les résultats des deux tests
    """
    print(f"\n{'='*70}")
    print(f"🏆 COMPARAISON FINALE")
    print(f"{'='*70}")
    
    # Temps de calcul
    time_speedup = stats_original['time'] / stats_optimized['time'] if stats_optimized['time'] > 0 else 0
    print(f"⏱️  TEMPS DE CALCUL:")
    print(f"   📊 Original:  {stats_original['time']:.2f}s")
    print(f"   ⚡ Optimisé:  {stats_optimized['time']:.2f}s")
    print(f"   📈 Speedup:   {time_speedup:.2f}x")
    print()
    
    # Vitesse de recherche
    speed_orig = stats_original['nodes_evaluated'] / stats_original['time'] if stats_original['time'] > 0 else 0
    speed_opt = stats_optimized['nodes_evaluated'] / stats_optimized['time'] if stats_optimized['time'] > 0 else 0
    speed_speedup = speed_opt / speed_orig if speed_orig > 0 else 0
    
    print(f"🚀 VITESSE DE RECHERCHE:")
    print(f"   📊 Original:  {speed_orig:,.0f} nœuds/sec")
    print(f"   ⚡ Optimisé:  {speed_opt:,.0f} nœuds/sec")
    print(f"   📈 Speedup:   {speed_speedup:.2f}x")
    print()
    
    # Exploration
    node_ratio = stats_optimized['nodes_evaluated'] / stats_original['nodes_evaluated'] if stats_original['nodes_evaluated'] > 0 else 0
    print(f"📊 CAPACITÉ D'EXPLORATION:")
    print(f"   📊 Original:  {stats_original['nodes_evaluated']:,} nœuds")
    print(f"   ⚡ Optimisé:  {stats_optimized['nodes_evaluated']:,} nœuds")
    print(f"   📈 Ratio:     {node_ratio:.2f}x")
    print()
    
    # Efficacité du null-move
    null_eff_orig = (stats_original['null_move_cutoffs'] / stats_original['null_move_attempts'] * 100) if stats_original['null_move_attempts'] > 0 else 0
    null_eff_opt = (stats_optimized['null_move_cutoffs'] / stats_optimized['null_move_attempts'] * 100) if stats_optimized['null_move_attempts'] > 0 else 0
    
    print(f"✂️  EFFICACITÉ NULL-MOVE:")
    print(f"   📊 Original:  {null_eff_orig:.1f}% ({stats_original['null_move_cutoffs']:,}/{stats_original['null_move_attempts']:,})")
    print(f"   ⚡ Optimisé:  {null_eff_opt:.1f}% ({stats_optimized['null_move_cutoffs']:,}/{stats_optimized['null_move_attempts']:,})")
    print()
    
    # Validation des coups
    move_orig = stats_original['best_move']
    move_opt = stats_optimized['best_move']
    moves_match = (move_orig == move_opt) if (move_orig and move_opt) else False
    
    print(f"✅ VALIDATION:")
    print(f"   📊 Coup original:  {stats_original['engine']._format_move(move_orig) if move_orig else 'Aucun'}")
    print(f"   ⚡ Coup optimisé:  {stats_optimized['engine']._format_move(move_opt) if move_opt else 'Aucun'}")
    print(f"   🎯 Coups identiques: {'✅' if moves_match else '❌'}")
    print()
    
    # Conclusion
    print(f"🎯 CONCLUSION:")
    if speed_speedup >= 3.0:
        print(f"🔥 EXCELLENT! Speedup de {speed_speedup:.1f}x")
        print(f"✅ Les optimisations bitboard sont extrêmement efficaces!")
        print(f"🚀 L'IA peut explorer {node_ratio:.1f}x plus de positions")
    elif speed_speedup >= 2.0:
        print(f"🔥 TRÈS BON! Speedup de {speed_speedup:.1f}x")
        print(f"✅ Les optimisations apportent un gain majeur")
        print(f"🚀 L'IA peut explorer {node_ratio:.1f}x plus de positions")
    elif speed_speedup >= 1.5:
        print(f"👍 BON! Speedup de {speed_speedup:.1f}x")
        print(f"✅ Les optimisations sont bénéfiques")
    else:
        print(f"🤔 LIMITÉ. Speedup de {speed_speedup:.1f}x")
        print(f"⚠️  Impact limité des optimisations")
    
    if moves_match:
        print(f"💎 Qualité de jeu préservée (même coup optimal)")
    else:
        print(f"⚠️  Exploration différente → coups différents")
    
    return speed_speedup, node_ratio

def main():
    """
    Fonction principale du test comparatif
    """
    print("🔥 TEST COMPARATIF NULL-MOVE ENGINE 🔥")
    print("📊 Chess.py original vs optimized_chess.py")
    print("⏱️  60 secondes par test")
    print("🎯 Position identique pour les deux tests")
    print()
    
    # Créer la position de test
    print("🎲 Préparation de la position de test...")
    test_position = create_test_position()
    
    # Créer deux copies indépendantes
    chess_original = Chess()
    chess_optimized = Chess()
    
    # Reproduire la même position sur les deux boards
    for move in test_position.history:
        try:
            chess_original.move_piece(move['from'], move['to'], move.get('promotion'))
            chess_optimized.move_piece(move['from'], move['to'], move.get('promotion'))
        except:
            # Si l'historique n'est pas disponible, utiliser la position initiale
            chess_original = Chess()
            chess_optimized = Chess()
            break
    
    print("✅ Position de test préparée")
    
    # Test 1: Version originale
    stats_original = run_null_move_test(
        chess_original, 
        "VERSION ORIGINALE (Chess.py)", 
        use_optimizations=False, 
        time_limit=60
    )
    
    # Test 2: Version optimisée
    stats_optimized = run_null_move_test(
        chess_optimized, 
        "VERSION OPTIMISÉE (optimized_chess.py)", 
        use_optimizations=True, 
        time_limit=60
    )
    
    # Comparaison finale
    speed_gain, exploration_gain = compare_results(stats_original, stats_optimized)
    
    print(f"\n🏁 TEST TERMINÉ")
    print(f"📈 Gain de vitesse: {speed_gain:.1f}x")
    print(f"🔍 Gain d'exploration: {exploration_gain:.1f}x")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🏁 Fin du programme")