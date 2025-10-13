#!/usr/bin/env python3
"""
Test de performance en profondeur 6 avec l'IA null-move.
Mesure le temps nécessaire pour atteindre exactement la profondeur 6.
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Chess import Chess
from evaluator import ChessEvaluator
from ai.Null_move_AI.null_move_engine import NullMovePruningEngine
from ai.Old_AI.iterative_deepening_engine_TT import IterativeDeepeningAlphaBeta

def test_depth_6_performance():
    """Test de performance à profondeur 6 exacte"""
    
    print("=== TEST PROFONDEUR 6 - PERFORMANCES ===")
    print("🔍 Comparaison équitable:")
    print("   - Base: Iterative Deepening + Table de Transposition")
    print("   - Test: Base + Null-Move Pruning")
    print("   - Même temps limite, même profondeur cible\n")
    
    evaluator = ChessEvaluator()
    
    # Positions de test variées
    positions = []
    
    # Position 1: Début de partie
    chess1 = Chess()
    positions.append(("Début de partie", chess1))
    
    # Position 2: Milieu de partie développée
    def pos_to_square(row, col):
        return row * 8 + col
    
    chess2 = Chess()
    moves2 = [
        (pos_to_square(1, 4), pos_to_square(3, 4), None),  # e2-e4
        (pos_to_square(6, 4), pos_to_square(4, 4), None),  # e7-e5
        (pos_to_square(0, 6), pos_to_square(2, 5), None),  # Ng1-f3
        (pos_to_square(7, 1), pos_to_square(5, 2), None),  # Nb8-c6
        (pos_to_square(0, 5), pos_to_square(3, 2), None),  # Bf1-c4
        (pos_to_square(7, 5), pos_to_square(4, 2), None),  # Bf8-c5
    ]
    for move in moves2:
        chess2.move_piece(move[0], move[1], promotion=move[2])
    positions.append(("Milieu de partie", chess2))
    
    # Position 3: Position tactique (plus complexe)
    chess3 = Chess()
    moves3 = [
        (pos_to_square(1, 4), pos_to_square(3, 4), None),  # e2-e4
        (pos_to_square(6, 4), pos_to_square(4, 4), None),  # e7-e5
        (pos_to_square(0, 6), pos_to_square(2, 5), None),  # Ng1-f3
        (pos_to_square(7, 1), pos_to_square(5, 2), None),  # Nb8-c6
        (pos_to_square(0, 5), pos_to_square(3, 2), None),  # Bf1-c4
        (pos_to_square(7, 5), pos_to_square(4, 2), None),  # Bf8-c5
        (pos_to_square(1, 3), pos_to_square(3, 3), None),  # d2-d3
        (pos_to_square(6, 3), pos_to_square(4, 3), None),  # d7-d6
        (pos_to_square(0, 2), pos_to_square(1, 3), None),  # Bc1-d2
        (pos_to_square(7, 2), pos_to_square(6, 3), None),  # Bc8-d7
    ]
    for move in moves3:
        chess3.move_piece(move[0], move[1], promotion=move[2])
    positions.append(("Position tactique", chess3))
    
    results = []
    
    for pos_name, chess in positions:
        print(f"🎯 {pos_name.upper()}")
        print("=" * 60)
        
        # Compter les coups légaux pour avoir une idée de la complexité
        legal_moves = chess._get_all_legal_moves if hasattr(chess, '_get_all_legal_moves') else []
        try:
            temp_engine = NullMovePruningEngine()
            legal_moves = temp_engine._get_all_legal_moves(chess)
            print(f"Coups légaux: {len(legal_moves)}")
        except:
            print("Coups légaux: N/A")
        
        print()
        
        # Test 1: Sans null-move (référence) - AVEC table de transposition
        print("📊 SANS NULL-MOVE (avec table de transposition):")
        engine_ref = IterativeDeepeningAlphaBeta(
            max_time=120,  # 2 minutes max pour test plus rapide
            max_depth=6,   # Profondeur fixe à 6
            evaluator=evaluator
        )
        
        start = time.time()
        move_ref = engine_ref.get_best_move_with_time_limit(chess)
        time_ref = time.time() - start
        
        print(f"  ⏱️  Temps total: {time_ref:.2f}s ({time_ref/60:.1f} min)")
        print(f"  🧠 Nœuds évalués: {engine_ref.nodes_evaluated:,}")
        print(f"  ✂️  Branches élaguées: {engine_ref.pruned_branches:,}")
        print(f"  💾 TT hits: {engine_ref.tt_hits:,} ({engine_ref.tt_hits/(engine_ref.tt_hits + engine_ref.tt_misses)*100:.1f}%)")
        print(f"  🎯 Meilleur coup: {engine_ref._format_move(move_ref)}")
        print()
        
        # Test 2: Avec null-move (+ table de transposition)
        print("🚀 AVEC NULL-MOVE (+ table de transposition):")
        engine_nm = NullMovePruningEngine(
            max_time=120,  # 2 minutes max pour test plus rapide
            max_depth=6,   # Profondeur fixe à 6
            evaluator=evaluator,
            null_move_enabled=True,
            null_move_R=2,
            null_move_min_depth=2  # Plus aggressif
        )
        
        start = time.time()
        move_nm = engine_nm.get_best_move_with_time_limit(chess)
        time_nm = time.time() - start
        
        nm_stats = engine_nm.get_null_move_stats()
        
        print(f"  ⏱️  Temps total: {time_nm:.2f}s ({time_nm/60:.1f} min)")
        print(f"  🧠 Nœuds évalués: {engine_nm.nodes_evaluated:,}")
        print(f"  ✂️  Branches élaguées: {engine_nm.pruned_branches:,}")
        print(f"  💾 TT hits: {engine_nm.tt_hits:,} ({engine_nm.tt_hits/(engine_nm.tt_hits + engine_nm.tt_misses)*100:.1f}%)")
        print(f"  🎯 Meilleur coup: {engine_nm._format_move(move_nm)}")
        print(f"  🔄 Null-moves: {nm_stats['attempts']} tentatives, {nm_stats['cutoffs']} cutoffs ({nm_stats['success_rate']:.1f}%)")
        print()
        
        # Analyse comparative
        if time_ref > 0:
            time_gain = ((time_ref - time_nm) / time_ref) * 100
            node_reduction = ((engine_ref.nodes_evaluated - engine_nm.nodes_evaluated) / engine_ref.nodes_evaluated) * 100 if engine_ref.nodes_evaluated > 0 else 0
            pruning_improvement = ((engine_nm.pruned_branches - engine_ref.pruned_branches) / max(1, engine_ref.pruned_branches)) * 100
            
            print("📈 ANALYSE COMPARATIVE:")
            print(f"  💨 Gain de temps: {time_gain:+.1f}%")
            print(f"  🔢 Réduction de nœuds: {node_reduction:+.1f}%")
            print(f"  ✂️  Amélioration élagage: {pruning_improvement:+.1f}%")
            
            if time_gain > 20:
                print("  🏆 EXCELLENT gain de performance!")
            elif time_gain > 5:
                print("  ✅ Bon gain de performance")
            elif time_gain > 0:
                print("  👍 Léger gain de performance")
            else:
                print("  ⚠️  Pas d'amélioration significative")
            
            # Efficacité par minute
            nps_ref = engine_ref.nodes_evaluated / max(1, time_ref)  # nœuds par seconde
            nps_nm = engine_nm.nodes_evaluated / max(1, time_nm)
            nps_improvement = ((nps_nm - nps_ref) / nps_ref) * 100 if nps_ref > 0 else 0
            
            print(f"  ⚡ Nœuds/seconde référence: {nps_ref:,.0f}")
            print(f"  ⚡ Nœuds/seconde null-move: {nps_nm:,.0f}")
            print(f"  ⚡ Amélioration efficacité: {nps_improvement:+.1f}%")
        
        results.append({
            'position': pos_name,
            'time_ref': time_ref,
            'time_nm': time_nm,
            'nodes_ref': engine_ref.nodes_evaluated,
            'nodes_nm': engine_nm.nodes_evaluated,
            'nm_attempts': nm_stats['attempts'],
            'nm_cutoffs': nm_stats['cutoffs'],
            'nm_success_rate': nm_stats['success_rate'],
            'time_gain': time_gain if 'time_gain' in locals() else 0
        })
        
        print("\n" + "="*60 + "\n")
    
    # Résumé global
    print("🏁 RÉSUMÉ GLOBAL - PROFONDEUR 6")
    print("=" * 60)
    
    total_time_ref = sum(r['time_ref'] for r in results)
    total_time_nm = sum(r['time_nm'] for r in results)
    total_nodes_ref = sum(r['nodes_ref'] for r in results)
    total_nodes_nm = sum(r['nodes_nm'] for r in results)
    total_nm_attempts = sum(r['nm_attempts'] for r in results)
    total_nm_cutoffs = sum(r['nm_cutoffs'] for r in results)
    
    global_time_gain = ((total_time_ref - total_time_nm) / total_time_ref) * 100 if total_time_ref > 0 else 0
    global_node_reduction = ((total_nodes_ref - total_nodes_nm) / total_nodes_ref) * 100 if total_nodes_ref > 0 else 0
    global_nm_success = (total_nm_cutoffs / max(1, total_nm_attempts)) * 100
    
    print(f"⏱️  Temps total référence: {total_time_ref:.1f}s ({total_time_ref/60:.1f} min)")
    print(f"⏱️  Temps total null-move: {total_time_nm:.1f}s ({total_time_nm/60:.1f} min)")
    print(f"💨 Gain de temps global: {global_time_gain:+.1f}%")
    print(f"🧠 Nœuds total référence: {total_nodes_ref:,}")
    print(f"🧠 Nœuds total null-move: {total_nodes_nm:,}")
    print(f"🔢 Réduction de nœuds globale: {global_node_reduction:+.1f}%")
    print(f"🔄 Null-moves total: {total_nm_attempts} tentatives, {total_nm_cutoffs} cutoffs ({global_nm_success:.1f}%)")
    
    print(f"\n📋 DÉTAIL PAR POSITION:")
    print(f"{'Position':<20} {'Temps (s)':<12} {'Gain %':<8} {'NM Success':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['position']:<20} {r['time_nm']:8.1f}     {r['time_gain']:+6.1f}%    {r['nm_success_rate']:8.1f}%")
    
    # Recommandations
    print(f"\n🎯 RECOMMANDATIONS:")
    if global_time_gain > 10:
        print("✅ Null-move très efficace à profondeur 6!")
        print("✅ Utiliser cette IA pour les recherches profondes")
    elif global_time_gain > 0:
        print("👍 Null-move modérément efficace")
        print("👍 Bénéfice visible sur certaines positions")
    else:
        print("⚠️  Null-move peu efficace sur ces positions")
        print("💡 Essayer avec des positions plus déséquilibrées")
    
    if global_nm_success > 20:
        print("🔥 Excellent taux de réussite null-move!")
    elif global_nm_success > 10:
        print("👌 Bon taux de réussite null-move")
    
    avg_time_per_position = total_time_nm / len(results)
    print(f"📊 Temps moyen par position (profondeur 6): {avg_time_per_position:.1f}s")
    
    if avg_time_per_position < 30:
        print("⚡ Très rapide pour la profondeur 6!")
    elif avg_time_per_position < 120:
        print("👍 Temps raisonnable pour la profondeur 6")
    else:
        print("⏳ Temps élevé - profondeur 6 est exigeante")

def test_depth_6_with_different_parameters():
    """Test avec différents paramètres null-move à profondeur 6"""
    
    print("\n" + "="*60)
    print("🔧 TEST PARAMÈTRES NULL-MOVE - PROFONDEUR 6")
    print("="*60)
    
    # Position de test (milieu de partie)
    chess = Chess()
    def pos_to_square(row, col):
        return row * 8 + col
    
    moves = [
        (pos_to_square(1, 4), pos_to_square(3, 4), None),  # e2-e4
        (pos_to_square(6, 4), pos_to_square(4, 4), None),  # e7-e5
        (pos_to_square(0, 6), pos_to_square(2, 5), None),  # Ng1-f3
        (pos_to_square(7, 1), pos_to_square(5, 2), None),  # Nb8-c6
    ]
    for move in moves:
        chess.move_piece(move[0], move[1], promotion=move[2])
    
    evaluator = ChessEvaluator()
    
    # Test différents paramètres
    configs = [
        ("R=2, min_depth=2", 2, 2),
        ("R=2, min_depth=3", 2, 3),
        ("R=3, min_depth=2", 3, 2),
        ("R=3, min_depth=3", 3, 3),
    ]
    
    print("Configuration des paramètres null-move:")
    print(f"{'Config':<20} {'Temps (s)':<12} {'Nœuds':<10} {'NM Success':<12}")
    print("-" * 60)
    
    for config_name, R, min_depth in configs:
        engine = NullMovePruningEngine(
            max_time=180,  # 3 minutes max par test
            max_depth=6,
            evaluator=evaluator,
            null_move_enabled=True,
            null_move_R=R,
            null_move_min_depth=min_depth
        )
        
        start = time.time()
        move = engine.get_best_move_with_time_limit(chess)
        test_time = time.time() - start
        
        nm_stats = engine.get_null_move_stats()
        
        print(f"{config_name:<20} {test_time:8.1f}     {engine.nodes_evaluated:<10,} {nm_stats['success_rate']:8.1f}%")

if __name__ == "__main__":
    try:
        test_depth_6_performance()
        test_depth_6_with_different_parameters()
        
    except KeyboardInterrupt:
        print("\n\n❌ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc()