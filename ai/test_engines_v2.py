import time
from Chess import Chess
from evaluator import ChessEvaluator
from alphabeta_engine import AlphaBetaEngine


def test_alphabeta():
    """Test basique du moteur alpha-beta"""
    chess = Chess()
    
    print("=== Test Alpha-Beta vs Minimax ===")
    print("Position initiale:")
    chess.print_board()
    
    # Test avec l'alpha-beta
    ab_engine = AlphaBetaEngine(max_depth=3, evaluator=ChessEvaluator())
    
    start_time = time.time()
    best_move_ab = ab_engine.get_best_move(chess)
    ab_time = time.time() - start_time
    
    if best_move_ab:
        from_sq, to_sq, promo = best_move_ab
        from_alg = f"{chr(ord('a') + from_sq % 8)}{from_sq // 8 + 1}"
        to_alg = f"{chr(ord('a') + to_sq % 8)}{to_sq // 8 + 1}"
        promo_str = promo if promo else ""
        
        print(f"Alpha-Beta: {from_alg}{to_alg}{promo_str}")
        print(f"Temps: {ab_time:.3f}s")
        print(f"Efficacité élagage: {ab_engine.pruned_branches / (ab_engine.nodes_evaluated + ab_engine.pruned_branches) * 100:.1f}%")



def compare_engines():
    """Compare les performances entre minimax et alpha-beta"""
    chess = Chess()
    
    print("\n=== Comparaison des moteurs ===")
    
    # Jouer quelques coups pour avoir une position plus intéressante
    test_moves = [(12, 28), (52, 36), (6, 21), (57, 42)]  # e2e4, e7e5, g1f3, b8c6
    for from_sq, to_sq in test_moves:
        try:
            chess.move_piece(from_sq, to_sq)
        except:
            break
    
    print("Position de test:")
    chess.print_board()
    
    # Test Alpha-Beta à différentes profondeurs
    for depth in [3, 4]:
        print(f"\n--- Profondeur {depth} ---")
        
        ab_engine = AlphaBetaEngine(max_depth=depth)
        
        start_time = time.time()
        best_move = ab_engine.get_best_move(chess)
        elapsed = time.time() - start_time
        
        if best_move:
            move_str = ab_engine._format_move(best_move)
            efficiency = ab_engine.pruned_branches / (ab_engine.nodes_evaluated + ab_engine.pruned_branches) * 100
            print(f"Alpha-Beta: {move_str}, Temps: {elapsed:.3f}s, Efficacité: {efficiency:.1f}%")


def test_all_engines():
    """Test complet de tous les moteurs"""
    print("🏁 Tests des moteurs d'échecs")
    print("=" * 50)
    
    test_alphabeta()
    """test_iterative_deepening()"""
    compare_engines()
    
    print("\n✅ Tests terminés !")


if __name__ == "__main__":
    test_all_engines()