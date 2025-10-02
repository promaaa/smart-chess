"""
Test de comparaison entre l'IA avec et sans table de transposition
"""

import time
from Chess import Chess
from evaluator import ChessEvaluator
from iterative_deepening_engine import IterativeDeepeningAlphaBeta
from iterative_deepening_engine_TT import IterativeDeepeningAlphaBeta as IterativeDeepeningTT


def test_transposition_table():
    """Teste l'efficacité de la table de transposition"""
    print("🧠 Test de la Table de Transposition")
    print("=" * 50)
    
    # Créer une position de test
    chess = Chess()
    
    # Jouer quelques coups pour avoir une position intéressante
    test_moves = [(12, 28), (52, 36), (6, 21), (57, 42)]  # e2e4, e7e5, g1f3, b8c6
    for from_sq, to_sq in test_moves:
        try:
            chess.move_piece(from_sq, to_sq)
        except:
            break
    
    print("Position de test:")
    chess.print_board()
    print()
    
    # Test sans table de transposition - Profondeur 4 exactement
    print("🔍 Sans Table de Transposition (Profondeur 4):")
    engine_normal = IterativeDeepeningAlphaBeta(max_time=300.0, max_depth=4)  # 5 minutes max
    
    start_time = time.time()
    best_move_normal = engine_normal.get_best_move_with_time_limit(chess)
    time_normal = time.time() - start_time
    
    print(f"Meilleur coup: {engine_normal._format_move(best_move_normal)}")
    print(f"Temps pour profondeur 4: {time_normal:.2f}s")
    print(f"Nœuds évalués: {engine_normal.nodes_evaluated}")
    print(f"Branches élaguées: {engine_normal.pruned_branches}")
    print()
    
    # Test avec table de transposition - Profondeur 4 exactement
    print("⚡ Avec Table de Transposition (Profondeur 4):")
    engine_tt = IterativeDeepeningTT(max_time=300.0, max_depth=4, tt_size=500000)  # 5 minutes max
    
    start_time = time.time()
    best_move_tt = engine_tt.get_best_move_with_time_limit(chess)
    time_tt = time.time() - start_time
    
    print(f"Meilleur coup: {engine_tt._format_move(best_move_tt)}")
    print(f"Temps pour profondeur 4: {time_tt:.2f}s")
    print(f"Nœuds évalués: {engine_tt.nodes_evaluated}")
    print(f"Branches élaguées: {engine_tt.pruned_branches}")
    print(f"Hits TT: {engine_tt.tt_hits}")
    print(f"Misses TT: {engine_tt.tt_misses}")
    
    # Calcul des améliorations pour la profondeur 4
    if time_normal > 0:
        speed_improvement = (time_normal - time_tt) / time_normal * 100
        nodes_reduction = (engine_normal.nodes_evaluated - engine_tt.nodes_evaluated) / engine_normal.nodes_evaluated * 100
        
        print("\n📊 Améliorations à Profondeur 4:")
        print(f"⏱️ Gain de temps: {speed_improvement:.1f}% ({time_normal - time_tt:.2f}s économisées)")
        print(f"🧠 Réduction des nœuds: {nodes_reduction:.1f}% ({engine_normal.nodes_evaluated - engine_tt.nodes_evaluated} nœuds en moins)")
        
        if engine_tt.tt_hits + engine_tt.tt_misses > 0:
            tt_hit_rate = engine_tt.tt_hits / (engine_tt.tt_hits + engine_tt.tt_misses) * 100
            print(f"Taux de hit TT: {tt_hit_rate:.1f}%")


def test_move_ordering():
    """Teste l'amélioration de l'ordonnancement des coups"""
    print("\n🎯 Test de l'Ordonnancement des Coups")
    print("=" * 40)
    
    chess = Chess()
    engine = IterativeDeepeningTT(max_time=60.0, max_depth=4)
    
    # Première recherche
    print("Première recherche (TT vide):")
    best_move1 = engine.get_best_move_with_time_limit(chess)
    nodes1 = engine.nodes_evaluated
    
    # Deuxième recherche (TT remplie)
    print("\nDeuxième recherche (TT remplie):")
    engine.nodes_evaluated = 0
    engine.pruned_branches = 0
    engine.tt_hits = 0
    engine.tt_misses = 0
    
    best_move2 = engine.get_best_move_with_time_limit(chess)
    nodes2 = engine.nodes_evaluated
    
    if nodes1 > 0:
        improvement = (nodes1 - nodes2) / nodes1 * 100
        print(f"\nAmélioration: {improvement:.1f}% moins de nœuds évalués")
        print(f"Nœuds 1ère recherche: {nodes1}")
        print(f"Nœuds 2ème recherche: {nodes2}")


def test_different_positions():
    """Teste sur différentes positions"""
    print("\n🏰 Test sur Différentes Positions")
    print("=" * 35)
    
    positions = [
        {
            'name': 'Position initiale',
            'moves': []
        },
        {
            'name': 'Milieu de partie',
            'moves': [(12, 28), (52, 36), (6, 21), (57, 42), (5, 26), (58, 42)]
        },
        {
            'name': 'Position tactique',
            'moves': [(12, 28), (52, 36), (28, 36), (59, 36), (6, 21)]
        }
    ]
    
    for pos in positions:
        print(f"\n--- {pos['name']} ---")
        chess = Chess()
        
        # Jouer les coups
        for from_sq, to_sq in pos['moves']:
            try:
                chess.move_piece(from_sq, to_sq)
            except:
                break
        
        engine = IterativeDeepeningTT(max_time=60.0, max_depth=4)
        start_time = time.time()
        best_move = engine.get_best_move_with_time_limit(chess)
        elapsed = time.time() - start_time
        
        print(f"Meilleur coup: {engine._format_move(best_move)}")
        print(f"Temps: {elapsed:.2f}s, Nœuds: {engine.nodes_evaluated}")
        
        if engine.tt_hits + engine.tt_misses > 0:
            hit_rate = engine.tt_hits / (engine.tt_hits + engine.tt_misses) * 100
            print(f"TT hit rate: {hit_rate:.1f}%")


if __name__ == "__main__":
    test_transposition_table()
    test_move_ordering()
    test_different_positions()
    
    print("\n✅ Tests terminés !")