from Chess import Chess
from evaluator import ChessEvaluator, AdvancedChessEvaluator
import random

class AlphaBetaEngine:
    """Moteur d'échecs utilisant l'algorithme alpha-beta avec élagage"""
    
    def __init__(self, max_depth=4, evaluator=None):
        self.max_depth = max_depth
        self.evaluator = evaluator if evaluator else ChessEvaluator()
        self.nodes_evaluated = 0
        self.pruned_branches = 0
    
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
    '''Peut-être à supprimer ou à modifier'''
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
        import time
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
        import time
        
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
        import time
        
        # Vérifier le temps restant
        if time.time() - self.start_time > self.max_time:
            raise TimeoutError("Temps limite atteint")
        
        return self.alphabeta(chess, depth, alpha, beta, is_maximizing)
    
    def _format_move(self, move):
        """Formate un coup pour l'affichage"""
        if not move:
            return "None"
        from_sq, to_sq, promo = move
        from_alg = f"{chr(ord('a') + from_sq % 8)}{from_sq // 8 + 1}"
        to_alg = f"{chr(ord('a') + to_sq % 8)}{to_sq // 8 + 1}"
        promo_str = promo.upper() if promo else ""
        return f"{from_alg}{to_alg}{promo_str}"

# Fonctions de test et comparaison
def test_alphabeta():
    """Test basique du moteur alpha-beta"""
    chess = Chess()
    
    print("=== Test Alpha-Beta vs Minimax ===")
    print("Position initiale:")
    chess.print_board()
    
    # Test avec l'alpha-beta
    ab_engine = AlphaBetaEngine(max_depth=3, evaluator=ChessEvaluator())
    
    import time
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

def test_iterative_deepening():
    """Test de l'approfondissement itératif"""
    chess = Chess()
    
    print("\n=== Test Approfondissement Itératif ===")
    id_engine = IterativeDeepeningAlphaBeta(max_time=3.0, max_depth=8)
    
    best_move = id_engine.get_best_move_with_time_limit(chess)
    if best_move:
        print(f"Meilleur coup trouvé: {id_engine._format_move(best_move)}")

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
    for depth in [3, 4, 5]:
        print(f"\n--- Profondeur {depth} ---")
        
        ab_engine = AlphaBetaEngine(max_depth=depth)
        
        import time
        start_time = time.time()
        best_move = ab_engine.get_best_move(chess)
        elapsed = time.time() - start_time
        
        if best_move:
            move_str = ab_engine._format_move(best_move)
            efficiency = ab_engine.pruned_branches / (ab_engine.nodes_evaluated + ab_engine.pruned_branches) * 100
            print(f"Alpha-Beta: {move_str}, Temps: {elapsed:.3f}s, Efficacité: {efficiency:.1f}%")

if __name__ == "__main__":
    test_alphabeta()
    #test_iterative_deepening()
    #compare_engines()
