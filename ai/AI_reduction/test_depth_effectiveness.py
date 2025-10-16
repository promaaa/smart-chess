#!/usr/bin/env python3
"""
Test de comparaison: Null-Move Pruning vs Iterative Deepening avec Réduction Exponentielle
Compare les performances des deux algorithmes avec optimisations bitboard
"""

import time
import sys
import os

# Ajouter le dossier parent au path pour importer les modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Chess import Chess
from evaluator import ChessEvaluator  # Évaluateur ORIGINAL pour comparaison !
from AI_reduction.iterative_deepening_engine_TT_rdcut import IterativeDeepeningAlphaBeta
from Null_move_AI.null_move_engine import NullMovePruningEngine

def create_test_position():
    """Crée une position de test stratégique"""
    chess = Chess()
    
    # Position d'ouverture Ruy Lopez après quelques coups
    moves = [
        (12, 28, None),  # e2-e4
        (52, 36, None),  # e7-e5  
        (6, 21, None),   # Ng1-f3
        (57, 42, None),  # Nb8-c6
        (5, 33, None),   # Bf1-b5
    ]
    
    print("🎯 Position de test: Ruy Lopez - 1.e4 e5 2.Nf3 Nc6 3.Bb5")
    
    for from_sq, to_sq, promotion in moves:
        try:
            chess.move_piece(from_sq, to_sq, promotion)
        except Exception as e:
            print(f"⚠️  Erreur dans la séquence: {e}")
            return Chess()
    
    return chess

def test_engine(engine, engine_name, chess_board, time_limit=60):
    """Test un engine pendant une durée limitée"""
    print(f"\n{'='*60}")
    print(f"🔍 TEST: {engine_name}")
    print(f"{'='*60}")
    
    # Configuration
    print(f"⚙️  Configuration:")
    print(f"   ⏱️  Temps limite: {time_limit}s")
    if hasattr(engine, 'null_move_enabled'):
        print(f"   🎯 Null-move: {'✅ Activé' if engine.null_move_enabled else '❌ Désactivé'}")
        if engine.null_move_enabled:
            print(f"   📊 Null-move R: {engine.null_move_R}")
    if hasattr(engine, 'move_reduction_enabled'):
        print(f"   ✂️  Réduction: {'✅ Activée' if engine.move_reduction_enabled else '❌ Désactivée'}")
        if hasattr(engine, 'exponential_reduction'):
            print(f"   📈 Réduction expo: {'✅ Oui' if engine.exponential_reduction else '❌ Non'}")
    print(f"   💾 Table TT: {engine.tt_size:,} entrées")
    print()
    
    print("🚀 Démarrage de la recherche...")
    start_time = time.time()
    
    try:
        best_move = engine.get_best_move_with_time_limit(chess_board)
        elapsed_time = time.time() - start_time
        
        print(f"✅ Recherche terminée en {elapsed_time:.2f}s")
        
        # Calculer statistiques
        nodes_per_sec = engine.nodes_evaluated / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n📊 RÉSULTATS:")
        print(f"🎯 Meilleur coup: {engine._format_move(best_move) if best_move else 'Aucun'}")
        print(f"⏱️  Temps: {elapsed_time:.2f}s")
        print(f"📈 Nœuds évalués: {engine.nodes_evaluated:,}")
        print(f"✂️  Branches élaguées: {engine.pruned_branches:,}")
        print(f"🚀 Vitesse: {nodes_per_sec:,.0f} nœuds/sec")
        print(f"💾 TT hits: {engine.tt_hits:,}")
        
        # Statistiques spécifiques
        if hasattr(engine, 'null_move_attempts') and engine.null_move_attempts > 0:
            null_eff = (engine.null_move_cutoffs / engine.null_move_attempts * 100)
            print(f"🔄 Null-move efficacité: {engine.null_move_cutoffs:,}/{engine.null_move_attempts:,} ({null_eff:.1f}%)")
        
        if hasattr(engine, 'moves_skipped'):
            print(f"✂️  Coups sautés: {engine.moves_skipped:,}")
        
        return {
            'success': True,
            'engine_name': engine_name,
            'best_move': best_move,
            'time': elapsed_time,
            'nodes': engine.nodes_evaluated,
            'speed': nodes_per_sec,
            'pruned': engine.pruned_branches,
            'tt_hits': engine.tt_hits,
            'engine': engine
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Erreur après {elapsed_time:.2f}s: {e}")
        return {
            'success': False,
            'engine_name': engine_name,
            'time': elapsed_time,
            'error': str(e)
        }

def compare_engines():
    """Compare les deux algorithmes d'IA"""
    print("🎯 COMPARAISON D'ALGORITHMES CHESS AI")
    print("=" * 80)
    print("🔄 Null-Move Pruning vs ✂️  Iterative Deepening + Réduction Exponentielle")
    print("⏱️  Limite de temps: 60 secondes par algorithme")
    print("=" * 80)
    
    # Créer la position de test
    evaluator = ChessEvaluator()
    chess_board = create_test_position()
    time_limit = 60
    
    print(f"\n📋 Position de test:")
    chess_board.print_board()
    print()
    
    results = []
    
    # =================== TEST 1: NULL-MOVE PRUNING ===================
    print("🔄 ÉTAPE 1/2: Test du Null-Move Pruning Engine")
    
    null_move_engine = NullMovePruningEngine(
        max_time=time_limit,
        max_depth=20,
        evaluator=evaluator,
        tt_size=500000,
        null_move_enabled=True,
        null_move_R=2
    )
    
    result1 = test_engine(null_move_engine, "Null-Move Pruning", chess_board, time_limit)
    results.append(result1)
    
    # Pause entre les tests
    print("\n⏳ Pause de 2 secondes...")
    time.sleep(2)
    
    # ================= TEST 2: RÉDUCTION EXPONENTIELLE =================
    print("\n✂️  ÉTAPE 2/2: Test du Réduction Engine")
    
    reduction_engine = IterativeDeepeningAlphaBeta(
        max_time=time_limit,
        max_depth=20,
        evaluator=evaluator,
        tt_size=500000,
        move_reduction_enabled=True,
        reduction_seed=42
    )
    # Ajouter attribut pour l'affichage
    reduction_engine.exponential_reduction = True
    
    result2 = test_engine(reduction_engine, "Réduction Exponentielle", chess_board, time_limit)
    results.append(result2)
    
    # =================== ANALYSE COMPARATIVE ===================
    print(f"\n{'='*80}")
    print("📊 ANALYSE COMPARATIVE FINALE")
    print(f"{'='*80}")
    
    if result1['success'] and result2['success']:
        print(f"\n🏆 TABLEAU COMPARATIF:")
        print(f"{'Métrique':<25} {'Null-Move':<15} {'Réduction':<15} {'Gagnant':<15}")
        print("-" * 70)
        
        # Temps d'exécution
        print(f"{'⏱️  Temps (s)':<25} {result1['time']:<15.2f} {result2['time']:<15.2f} ", end="")
        if result1['time'] < result2['time']:
            print("🔄 Null-Move")
        elif result2['time'] < result1['time']:
            print("✂️  Réduction")
        else:
            print("➡️ Égalité")
        
        # Nœuds explorés
        print(f"{'📈 Nœuds':<25} {result1['nodes']:<15,} {result2['nodes']:<15,} ", end="")
        if result1['nodes'] > result2['nodes']:
            print("🔄 Null-Move")
        elif result2['nodes'] > result1['nodes']:
            print("✂️  Réduction")
        else:
            print("➡️ Égalité")
        
        # Vitesse de recherche
        print(f"{'🚀 Vitesse (N/s)':<25} {result1['speed']:<15,.0f} {result2['speed']:<15,.0f} ", end="")
        if result1['speed'] > result2['speed']:
            print("🔄 Null-Move")
        elif result2['speed'] > result1['speed']:
            print("✂️  Réduction")
        else:
            print("➡️ Égalité")
        
        # Élagage
        print(f"{'✂️  Élagage':<25} {result1['pruned']:<15,} {result2['pruned']:<15,} ", end="")
        if result1['pruned'] > result2['pruned']:
            print("🔄 Null-Move")
        elif result2['pruned'] > result1['pruned']:
            print("✂️  Réduction")
        else:
            print("➡️ Égalité")
        
        # Table de transposition
        print(f"{'💾 TT Hits':<25} {result1['tt_hits']:<15,} {result2['tt_hits']:<15,} ", end="")
        if result1['tt_hits'] > result2['tt_hits']:
            print("🔄 Null-Move")
        elif result2['tt_hits'] > result1['tt_hits']:
            print("✂️  Réduction")
        else:
            print("➡️ Égalité")
        
        # Même meilleur coup?
        same_move = result1['best_move'] == result2['best_move']
        print(f"\n🎯 Même meilleur coup: {'✅ OUI' if same_move else '❌ NON'}")
        
        # Ratios de performance
        node_ratio = result2['nodes'] / result1['nodes'] if result1['nodes'] > 0 else 0
        speed_ratio = result2['speed'] / result1['speed'] if result1['speed'] > 0 else 0
        time_ratio = result2['time'] / result1['time'] if result1['time'] > 0 else 0
        
        print(f"\n📊 RATIOS (Réduction / Null-Move):")
        print(f"   📈 Nœuds explorés: {node_ratio:.2f}x")
        print(f"   🚀 Vitesse: {speed_ratio:.2f}x")
        print(f"   ⏱️  Temps: {time_ratio:.2f}x")
        
        # Analyse et recommandation
        print(f"\n🎖️  ANALYSE:")
        if result1['nodes'] > result2['nodes'] * 1.15:  # 15% de marge
            print("   🔄 Null-Move Pruning explore significativement plus de nœuds")
            print("   💡 Avantage: Recherche plus exhaustive")
        elif result2['nodes'] > result1['nodes'] * 1.15:
            print("   ✂️  Réduction Exponentielle explore significativement plus de nœuds")
            print("   💡 Avantage: Meilleure exploration dans le temps imparti")
        else:
            print("   ⚖️  Exploration similaire entre les deux algorithmes")
        
        if result1['speed'] > result2['speed'] * 1.1:  # 10% de marge
            print("   🔄 Null-Move Pruning est plus rapide")
        elif result2['speed'] > result1['speed'] * 1.1:
            print("   ✂️  Réduction Exponentielle est plus rapide")
        else:
            print("   ⚖️  Vitesse similaire entre les deux algorithmes")
        
        print(f"\n🏆 RECOMMANDATION:")
        if result1['nodes'] > result2['nodes'] * 1.1 and result1['speed'] >= result2['speed'] * 0.9:
            print("   🥇 Null-Move Pruning semble plus efficace globalement")
        elif result2['nodes'] > result1['nodes'] * 1.1 and result2['speed'] >= result1['speed'] * 0.9:
            print("   🥇 Réduction Exponentielle semble plus efficace globalement")
        else:
            print("   🤝 Performance équilibrée - choix selon le contexte")
            
    else:
        print("❌ ÉCHEC DE LA COMPARAISON")
        print("Un ou plusieurs tests ont échoué:")
        for i, result in enumerate(results, 1):
            if not result['success']:
                print(f"   ❌ Test {i} ({result['engine_name']}): {result.get('error', 'Erreur inconnue')}")

def main():
    """Fonction principale"""
    print("🎯 Comparaison d'Algorithmes Chess AI")
    print("=" * 60)
    
    # Appliquer les optimisations bitboard
    try:
        from optimized_chess import patch_chess_class_globally
        patch_chess_class_globally()
        print("✅ Optimisations bitboard appliquées")
        print("   📈 Amélioration attendue: ~2.67x plus rapide")
    except ImportError:
        print("⚠️  Optimisations bitboard non disponibles")
        print("   💡 Conseil: Assurez-vous que optimized_chess.py est accessible")
    except Exception as e:
        print(f"⚠️  Erreur lors de l'application des optimisations: {e}")
        print("   ⏸️  Continuons sans optimisations...")
    
    print()
    
    try:
        compare_engines()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
