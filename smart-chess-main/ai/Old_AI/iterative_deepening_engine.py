import time
from alphabeta_engine import AlphaBetaEngine


class IterativeDeepeningAlphaBeta(AlphaBetaEngine):
    """Version avec approfondissement itératif pour une meilleure gestion du temps"""
    
    def __init__(self, max_time=5.0, max_depth=10, evaluator=None):
        super().__init__(max_depth, evaluator)
        self.max_time = max_time
        self.start_time = None
        
    def get_best_move_with_time_limit(self, chess):
        """
        Trouve le meilleur coup avec une limite de temps.
        Utilise l'approfondissement itératif.
        """
        self.start_time = time.time()
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        
        legal_moves = self._get_all_legal_moves(chess)
        if not legal_moves:
            return None
        
        best_move = legal_moves[0]  # Coup de sécurité
        
        # Approfondissement itératif
        for depth in range(1, self.max_depth + 1):
            if time.time() - self.start_time > self.max_time:
                print(f"Temps limite atteint à la profondeur {depth-1}")
                break
                
            try:
                current_best = self._search_at_depth(chess, depth)
                if current_best:
                    best_move = current_best
                    elapsed = time.time() - self.start_time
                    print(f"Depth {depth}: Best move = {self._format_move(best_move)}, "
                          f"Time: {elapsed:.2f}s, Nodes: {self.nodes_evaluated}")
            except TimeoutError:
                print(f"Recherche interrompue à la profondeur {depth}")
                break
        
        return best_move
    
    def _search_at_depth(self, chess, target_depth):
        """Recherche à une profondeur donnée avec vérification du temps"""
        legal_moves = self._get_all_legal_moves(chess)
        ordered_moves = self._order_moves(chess, legal_moves)
        
        best_move = None
        # Même correction ici
        current_player_is_white = chess.white_to_move
        best_score = float('-inf') if current_player_is_white else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in ordered_moves:
            # Vérifier le temps restant
            if time.time() - self.start_time > self.max_time:
                raise TimeoutError("Temps limite atteint")
            
            chess.move_piece(move[0], move[1], promotion=move[2])
            
            if current_player_is_white:  # Le joueur actuel était blanc
                score = self.alphabeta_with_timeout(chess, target_depth - 1, alpha, beta, False)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:  # Le joueur actuel était noir
                score = self.alphabeta_with_timeout(chess, target_depth - 1, alpha, beta, True)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
            
            chess.undo_move()
            
            if beta <= alpha:
                break
        
        return best_move
    
    def alphabeta_with_timeout(self, chess, depth, alpha, beta, is_maximizing):
        """Version d'alpha-beta avec vérification du temps"""
        # Vérifier le temps restant
        if time.time() - self.start_time > self.max_time:
            raise TimeoutError("Temps limite atteint")
        
        return self.alphabeta(chess, depth, alpha, beta, is_maximizing)