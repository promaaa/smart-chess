from Chess import Chess
from evaluator import ChessEvaluator


class BaseChessEngine:
    """
    Classe de base pour tous les moteurs d'échecs.
    Contient les méthodes communes pour la génération de coups et l'évaluation.
    """
    
    def __init__(self, evaluator=None):
        self.evaluator = evaluator if evaluator else ChessEvaluator()
        self.nodes_evaluated = 0
        self.pruned_branches = 0
    
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
    
    def _order_moves(self, chess, moves):
        """
        Ordonne les coups pour améliorer l'efficacité de l'élagage alpha-beta.
        Les bons coups sont évalués en premier pour maximiser les coupures.
        """
        def move_priority(move):
            from_sq, to_sq, promotion = move
            score = 0
            
            # Priorité aux captures
            captured_piece = self._get_piece_at(chess, to_sq)
            if captured_piece:
                # MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
                moving_piece = self._get_piece_at(chess, from_sq)
                victim_value = abs(self.evaluator.piece_values.get(captured_piece, 0))
                aggressor_value = abs(self.evaluator.piece_values.get(moving_piece, 0))
                score += victim_value * 10 - aggressor_value
            
            # Priorité aux promotions
            if promotion:
                promo_value = abs(self.evaluator.piece_values.get(promotion, 0))
                score += promo_value
            
            # Priorité aux coups vers le centre
            center_squares = [27, 28, 35, 36]  # d4, e4, d5, e5
            if to_sq in center_squares:
                score += 50
            
            # Priorité aux échecs (approximation simple)
            moving_piece = self._get_piece_at(chess, from_sq)
            if moving_piece and moving_piece.lower() in ['q', 'r', 'b', 'n']:
                # Bonus si la pièce peut attaquer le roi adverse
                enemy_king = 'k' if chess.white_to_move else 'K'
                enemy_king_square = None
                for sq in range(64):
                    if chess.bitboards.get(enemy_king, 0) & chess.square_mask(sq):
                        enemy_king_square = sq
                        break
                
                if enemy_king_square is not None:
                    # Vérifier si le coup donne échec (approximation)
                    chess.move_piece(from_sq, to_sq, promotion=promotion)
                    gives_check = chess.is_in_check(not chess.white_to_move)
                    chess.undo_move()
                    if gives_check:
                        score += 100
            
            return score
        
        # Trier par score décroissant (meilleurs coups en premier)
        return sorted(moves, key=move_priority, reverse=True)
    
    def _format_move(self, move):
        """Formate un coup pour l'affichage"""
        if not move:
            return "None"
        from_sq, to_sq, promo = move
        from_alg = f"{chr(ord('a') + from_sq % 8)}{from_sq // 8 + 1}"
        to_alg = f"{chr(ord('a') + to_sq % 8)}{to_sq // 8 + 1}"
        promo_str = promo.upper() if promo else ""
        return f"{from_alg}{to_alg}{promo_str}"
    
    def get_best_move(self, chess):
        """
        Méthode abstraite à implémenter par les classes filles.
        Retourne le meilleur coup pour la position actuelle.
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les classes filles")