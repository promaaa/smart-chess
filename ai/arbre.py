from Chess import Chess
from evaluator import ChessEvaluator, AdvancedChessEvaluator
import random

class MinimaxEngine:
    """Moteur d'échecs utilisant l'algorithme minimax"""
    
    def __init__(self, max_depth=3, evaluator=None):
        self.max_depth = max_depth
        self.evaluator = evaluator if evaluator else ChessEvaluator()
        self.nodes_evaluated = 0
    
    def get_best_move(self, chess):
        """
        Trouve le meilleur coup pour la position actuelle.
        Retourne un tuple (from_square, to_square, promotion)
        """
        self.nodes_evaluated = 0
        legal_moves = self._get_all_legal_moves(chess)
        
        if not legal_moves:
            return None
            
        best_move = None
        best_score = float('-inf') if chess.white_to_move else float('inf')
        
        for move in legal_moves:
            # Jouer le coup
            chess.move_piece(move[0], move[1], promotion=move[2])
            
            # Évaluer la position résultante
            score = self.minimax(chess, self.max_depth - 1, not chess.white_to_move)
            
            # Annuler le coup
            chess.undo_move()
            
            # Mettre à jour le meilleur coup
            if chess.white_to_move:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        
        print(f"Nodes evaluated: {self.nodes_evaluated}, Best score: {best_score}")
        return best_move
    
    def minimax(self, chess, depth, is_maximizing):
        """
        Algorithme minimax récursif.
        
        Args:
            chess: L'état du jeu
            depth: Profondeur restante à explorer
            is_maximizing: True si c'est au tour du joueur maximisant
        
        Returns:
            Le score de la meilleure position trouvée
        """
        self.nodes_evaluated += 1
        
        # Condition d'arrêt: profondeur 0 ou fin de partie
        if depth == 0:
            return self.evaluator.evaluate_position(chess)
        
        legal_moves = self._get_all_legal_moves(chess)

        # Fin de partie (mat ou pat) 
        # à modifier par la suite car cette façon 
        # d'évaluer la fin de partie peut être problématique
        if not legal_moves:
            if chess.is_in_check(chess.white_to_move):
                # Échec et mat - ajuster le score en fonction de la profondeur
                # Plus le mat est proche, plus le score est extrême
                mate_score = 10000 + depth
                return -mate_score if is_maximizing else mate_score
            else:
                # Pat
                return 0
        
        if is_maximizing:
            max_score = float('-inf')
            for move in legal_moves:
                chess.move_piece(move[0], move[1], promotion=move[2])
                score = self.minimax(chess, depth - 1, False)
                chess.undo_move()
                max_score = max(max_score, score)
            return max_score
        else:
            min_score = float('inf')
            for move in legal_moves:
                chess.move_piece(move[0], move[1], promotion=move[2])
                score = self.minimax(chess, depth - 1, True)
                chess.undo_move()
                min_score = min(min_score, score)
            return min_score
    
    def _get_all_legal_moves(self, chess):
        """Génère tous les coups légaux pour la position actuelle"""
        moves = []
        
        for from_square in range(64):
            piece = self._get_piece_at(chess, from_square)
            if not piece:
                continue
                
            # Vérifier si c'est une pièce du joueur actuel
            if piece.isupper() != chess.white_to_move:
                continue
                
            # Obtenir les coups pseudo-légaux
            move_mask = chess.get_all_moves(from_square)
            
            for to_square in range(64):
                if not (move_mask & chess.square_mask(to_square)):
                    continue
                
                # Gérer les promotions
                promotions = [None]
                if piece.lower() == 'p':
                    if (piece == 'P' and to_square // 8 == 7) or (piece == 'p' and to_square // 8 == 0):
                        promotions = ['Q', 'R', 'B', 'N'] if piece.isupper() else ['q', 'r', 'b', 'n']
                
                for promotion in promotions:
                    try:
                        # Tester si le coup est légal
                        chess.move_piece(from_square, to_square, promotion=promotion)
                        moves.append((from_square, to_square, promotion))
                        chess.undo_move()
                    except ValueError:
                        # Coup illégal (laisse le roi en échec)
                        pass
        
        return moves
    
    def _get_piece_at(self, chess, square):
        """Retourne la pièce à une case donnée"""
        mask = chess.square_mask(square)
        for piece, bitboard in chess.bitboards.items():
            if bitboard & mask:
                return piece
        return None

# Fonction de test
def test_minimax():
    """Test basique du moteur minimax"""
    chess = Chess()
    
    # Test avec l'évaluateur basique
    print("=== Test avec évaluateur basique ===")
    engine_basic = MinimaxEngine(max_depth=3, evaluator=ChessEvaluator())
    
    print("Position initiale:")
    chess.print_board()
    
    best_move = engine_basic.get_best_move(chess)
    if best_move:
        from_sq, to_sq, promo = best_move
        from_alg = f"{chr(ord('a') + from_sq % 8)}{from_sq // 8 + 1}"
        to_alg = f"{chr(ord('a') + to_sq % 8)}{to_sq // 8 + 1}"
        promo_str = promo if promo else ""
        
        print(f"Meilleur coup (évaluateur basique): {from_alg}{to_alg}{promo_str}")
    
    # Test avec l'évaluateur avancé
    print("\n=== Test avec évaluateur avancé ===")
    engine_advanced = MinimaxEngine(max_depth=3, evaluator=AdvancedChessEvaluator())
    
    best_move_advanced = engine_advanced.get_best_move(chess)
    if best_move_advanced:
        from_sq, to_sq, promo = best_move_advanced
        from_alg = f"{chr(ord('a') + from_sq % 8)}{from_sq // 8 + 1}"
        to_alg = f"{chr(ord('a') + to_sq % 8)}{to_sq // 8 + 1}"
        promo_str = promo if promo else ""
        
        print(f"Meilleur coup (évaluateur avancé): {from_alg}{to_alg}{promo_str}")

def test_game_with_different_evaluators():
    """Jouer une partie entre deux évaluateurs différents"""
    chess = Chess()
    engine1 = MinimaxEngine(max_depth=3, evaluator=ChessEvaluator())
    engine2 = MinimaxEngine(max_depth=3, evaluator=AdvancedChessEvaluator())
    
    print("=== Partie: Évaluateur basique (Blancs) vs Évaluateur avancé (Noirs) ===")
    chess.print_board()
    
    for move_count in range(10):  # Jouer 10 coups
        engine = engine1 if chess.white_to_move else engine2
        evaluator_name = "Basique" if chess.white_to_move else "Avancé"
        
        print(f"\nTour {move_count + 1} - {evaluator_name} ({'Blancs' if chess.white_to_move else 'Noirs'})")
        
        best_move = engine.get_best_move(chess)
        if best_move:
            from_sq, to_sq, promo = best_move
            from_alg = f"{chr(ord('a') + from_sq % 8)}{from_sq // 8 + 1}"
            to_alg = f"{chr(ord('a') + to_sq % 8)}{to_sq // 8 + 1}"
            promo_str = promo if promo else ""
            
            print(f"Coup joué: {from_alg}{to_alg}{promo_str}")
            chess.move_piece(from_sq, to_sq, promotion=promo)
            chess.print_board()
        else:
            print("Aucun coup légal trouvé!")
            break

if __name__ == "__main__":
    test_minimax()
    print("\n" + "="*50 + "\n")
    test_game_with_different_evaluators()
            from_sq, to_sq, promo = best_move
            from_alg = f"{chr(ord('a') + from_sq % 8)}{from_sq // 8 + 1}"
            to_alg = f"{chr(ord('a') + to_sq % 8)}{to_sq // 8 + 1}"
            promo_str = promo if promo else ""
            
            print(f"Meilleur coup: {from_alg}{to_alg}{promo_str}")
            chess.move_piece(from_sq, to_sq, promotion=promo)
            chess.print_board()
        else:
            print("Aucun coup légal trouvé!")
            break

if __name__ == "__main__":
    test_minimax()
