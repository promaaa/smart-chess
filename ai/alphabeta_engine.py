from base_engine import BaseChessEngine
from evaluator import ChessEvaluator


class AlphaBetaEngine(BaseChessEngine):
    """Moteur d'échecs utilisant l'algorithme alpha-beta avec élagage"""
    
    def __init__(self, max_depth=4, evaluator=None):
        super().__init__(evaluator)
        self.max_depth = max_depth
    
    def get_best_move(self, chess):
        """
        Trouve le meilleur coup pour la position actuelle.
        Retourne un tuple (from_square, to_square, promotion)
        """
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        legal_moves = self._get_all_legal_moves(chess)
        
        if not legal_moves:
            return None
        
        # Ordonner les coups pour améliorer l'efficacité de l'élagage
        ordered_moves = self._order_moves(chess, legal_moves)
        
        best_move = None
        # Sauvegarder qui doit jouer AVANT de commencer
        current_player_is_white = chess.white_to_move
        best_score = float('-inf') if current_player_is_white else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in ordered_moves:
            # Jouer le coup
            chess.move_piece(move[0], move[1], promotion=move[2])
            
            # Évaluer la position résultante avec alpha-beta
            # Après le coup, c'est à l'autre joueur de jouer
            if current_player_is_white:  # Le joueur actuel était blanc
                score = self.alphabeta(chess, self.max_depth - 1, alpha, beta, False)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:  # Le joueur actuel était noir
                score = self.alphabeta(chess, self.max_depth - 1, alpha, beta, True)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
            
            # Annuler le coup
            chess.undo_move()
            
            # Élagage au niveau racine (optionnel)
            if beta <= alpha:
                self.pruned_branches += len(ordered_moves) - ordered_moves.index(move) - 1
                break
        
        print(f"Nodes evaluated: {self.nodes_evaluated}, Pruned branches: {self.pruned_branches}, Best score: {best_score}")
        return best_move
    
    def alphabeta(self, chess, depth, alpha, beta, is_maximizing):
        """
        Algorithme alpha-beta récursif avec élagage.
        
        Args:
            chess: L'état du jeu
            depth: Profondeur restante à explorer
            alpha: Meilleure valeur garantie pour le joueur maximisant
            beta: Meilleure valeur garantie pour le joueur minimisant
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
        if not legal_moves:
            if chess.is_in_check(chess.white_to_move):
                # Échec et mat - ajuster le score en fonction de la profondeur
                mate_score = 20000 + depth  # Plus le mat est proche, plus le score est extrême
                return -mate_score if is_maximizing else mate_score
            else:
                # Pat
                return 0
        
        # Ordonner les coups pour améliorer l'élagage
        ordered_moves = self._order_moves(chess, legal_moves)
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                chess.move_piece(move[0], move[1], promotion=move[2])
                eval_score = self.alphabeta(chess, depth - 1, alpha, beta, False)
                chess.undo_move()
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                # Élagage beta (beta cutoff)
                if beta <= alpha:
                    self.pruned_branches += len(ordered_moves) - ordered_moves.index(move) - 1
                    break
                    
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                chess.move_piece(move[0], move[1], promotion=move[2])
                eval_score = self.alphabeta(chess, depth - 1, alpha, beta, True)
                chess.undo_move()
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                # Élagage alpha (alpha cutoff)
                if beta <= alpha:
                    self.pruned_branches += len(ordered_moves) - ordered_moves.index(move) - 1
                    break
                    
            return min_eval