import random
from base_engine import BaseChessEngine

class AlphaBetaEngine(BaseChessEngine):
    """Moteur Alpha-Beta optimisé avec Zobrist hashing, killer moves et history heuristic.

    .Transposition Table (Zobrist Hashing)
        Mémoire qui stocke les positions de l’échiquier sous forme de codes uniques (hash). 
        Quand une position revient, on réutilise son évaluation déjà calculée

    . Killer Moves
        Les coups qui entraînent des coupures beta significatives sont mémorisés par profondeur (ply).
        coups  priorisés dans l’ordre des coups

    . History Heuristic
        Chaque coup est évalué selon sa performance historique lors des recherches précédentes.
        bonus de priorité (position) pour les coups qui ont de bonnes évaluations 
    """

    def __init__(self, max_depth=4, evaluator=None):
        super().__init__(evaluator)
        self.max_depth = max_depth
        self.ttable = {}
        self.killer_moves = [{} for _ in range(max_depth)]
        self.history_heuristic = {}
        # Initialisation Zobrist
        self.pieces = 'PRNBQKprnbqk'
        self.zobrist_table = [[random.getrandbits(64) for _ in range(64)] for _ in range(12)]
        self.zobrist_black_to_move = random.getrandbits(64)

    def get_best_move(self, chess):
        self.nodes_evaluated = 0
        self.pruned_branches = 0

        legal_moves = self._get_all_legal_moves(chess)
        if not legal_moves:
            return None

        # Move ordering à la racine
        ordered_moves = self._order_moves_root(chess, legal_moves)
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        maximizing = chess.white_to_move
        best_score = float('-inf') if maximizing else float('inf')

        for move in ordered_moves:
            chess.move_piece(move[0], move[1], promotion=move[2])
            score = self.alphabeta(chess, self.max_depth - 1, alpha, beta, not maximizing, 1)
            chess.undo_move()

            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)

        print(f"Nodes evaluated: {self.nodes_evaluated}, Pruned branches: {self.pruned_branches}, Best score: {best_score}")
        return best_move

    def alphabeta(self, chess, depth, alpha, beta, maximizing, ply):
        self.nodes_evaluated += 1
        pos_hash = self._hash_position(chess)
        if pos_hash in self.ttable:
            entry = self.ttable[pos_hash]
            if entry['depth'] >= depth:
                return entry['score']

        # Stop condition
        if depth == 0:
            score = self.evaluator.evaluate_position(chess)
            self.ttable[pos_hash] = {'score': score, 'depth': depth}
            return score

        legal_moves = self._get_all_legal_moves(chess)
        if not legal_moves:
            # Checkmate / stalemate
            if chess.is_in_check(chess.white_to_move):
                mate_score = 20000 + depth
                score = -mate_score if maximizing else mate_score
            else:
                score = 0
            self.ttable[pos_hash] = {'score': score, 'depth': depth}
            return score

        # Move ordering avec killer moves et history heuristic
        ordered_moves = self._order_moves(chess, legal_moves, ply)

        if maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                chess.move_piece(move[0], move[1], promotion=move[2])
                eval_score = self.alphabeta(chess, depth - 1, alpha, beta, False, ply + 1)
                chess.undo_move()

                if eval_score > max_eval:
                    max_eval = eval_score
                    if eval_score >= beta:
                        self.killer_moves[ply-1][(move[0], move[1])] = True
                        self.pruned_branches += len(ordered_moves) - ordered_moves.index(move) - 1
                        break
                alpha = max(alpha, eval_score)
                self.history_heuristic[(move[0], move[1])] = self.history_heuristic.get((move[0], move[1]), 0) + depth**2
            self.ttable[pos_hash] = {'score': max_eval, 'depth': depth}
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                chess.move_piece(move[0], move[1], promotion=move[2])
                eval_score = self.alphabeta(chess, depth - 1, alpha, beta, True, ply + 1)
                chess.undo_move()

                if eval_score < min_eval:
                    min_eval = eval_score
                    if eval_score <= alpha:
                        self.killer_moves[ply-1][(move[0], move[1])] = True
                        self.pruned_branches += len(ordered_moves) - ordered_moves.index(move) - 1
                        break
                beta = min(beta, eval_score)
                self.history_heuristic[(move[0], move[1])] = self.history_heuristic.get((move[0], move[1]), 0) + depth**2
            self.ttable[pos_hash] = {'score': min_eval, 'depth': depth}
            return min_eval

    def _hash_position(self, chess):
        h = 0
        for idx, piece in enumerate(self.pieces):
            bitboard = chess.bitboards.get(piece, 0)
            for sq in range(64):
                if bitboard & (1 << sq):
                    h ^= self.zobrist_table[idx][sq]
        if not chess.white_to_move:
            h ^= self.zobrist_black_to_move
        return h

    def _order_moves_root(self, chess, moves):
        return sorted(moves, key=lambda m: self._move_score(chess, m) + self.history_heuristic.get((m[0], m[1]), 0), reverse=True)

    def _order_moves(self, chess, moves, ply):
        scored_moves = []
        for move in moves:
            score = self._move_score(chess, move)
            # Killer moves bonus
            if ply-1 < len(self.killer_moves) and (move[0], move[1]) in self.killer_moves[ply-1]:
                score += 10000
            # History heuristic bonus
            score += self.history_heuristic.get((move[0], move[1]), 0)
            scored_moves.append((score, move))
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [m for s, m in scored_moves]

    def _move_score(self, chess, move):
        """
        Score rapide des coups pour favoriser un jeu plus humain :
        - Contrôle du centre
        - Développement des pièces mineures
        - Captures et promotions
        - Attaques sur le roi adverse
        """
        from_sq, to_sq, promotion = move
        score = 0
        moving = self._get_piece_at(chess, from_sq)
        captured = self._get_piece_at(chess, to_sq)

        # --- Captures / promotions ---
        if captured:
            # MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
            score += abs(self.evaluator.piece_values.get(captured, 0)) * 10
            score -= abs(self.evaluator.piece_values.get(moving, 0))
        if promotion:
            score += abs(self.evaluator.piece_values.get(promotion, 0)) * 10

        # --- Centre ---
        central_squares = [27, 28, 35, 36]  # d4, e4, d5, e5
        if to_sq in central_squares:
            score += 50
        # Bonus léger pour contrôle du centre avec toutes les pièces
        center_bonus_squares = [19, 20, 21, 22, 27, 28, 35, 36, 43, 44, 45, 46]  # étendu autour du centre
        if to_sq in center_bonus_squares:
            score += 20

        # --- Développement des pièces mineures ---
        if moving.lower() in ['n', 'b']:
            # Bonus si sortie des cases initiales
            start_squares = [1,6,57,62]  # Nb1, Ng1, Nb8, Ng8
            if from_sq in start_squares:
                score += 30

        # --- Échec / attaque sur roi adverse ---
        enemy_king_sq = self._find_king(chess, not chess.white_to_move)
        if enemy_king_sq is not None and self._attacks_square(chess, from_sq, enemy_king_sq, moving):
            score += 100

        # --- Pions avancés pour l’ouverture ---
        if moving.lower() == 'p':
            row = to_sq // 8
            if chess.white_to_move:
                score += row * 5  # bonus si pion avance
            else:
                score += (7 - row) * 5

        return score


    def _find_king(self, chess, white):
        king = 'K' if white else 'k'
        bitboard = chess.bitboards.get(king, 0)
        if bitboard == 0:
            return None
        for sq in range(64):
            if bitboard & (1 << sq):
                return sq
        return None

    def _attacks_square(self, chess, from_sq, target_sq, piece):
        """Approximation rapide pour savoir si une pièce attaque une case"""
        # Simple approximation pour pièces longues (Q, R, B)
        if piece.lower() in ['q', 'r', 'b']:
            # Ligne / diagonale
            from_row, from_col = divmod(from_sq, 8)
            tgt_row, tgt_col = divmod(target_sq, 8)
            if piece.lower() == 'r' and (from_row == tgt_row or from_col == tgt_col):
                return True
            if piece.lower() == 'b' and abs(from_row - tgt_row) == abs(from_col - tgt_col):
                return True
            if piece.lower() == 'q' and (from_row == tgt_row or from_col == tgt_col or abs(from_row - tgt_row) == abs(from_col - tgt_col)):
                return True
        # Cavalier
        if piece.lower() == 'n':
            dr, dc = divmod(target_sq,8)
            fr, fc = divmod(from_sq,8)
            if (abs(dr-fr), abs(dc-fc)) in [(2,1),(1,2)]:
                return True
        # Roi et pion ignorés (moins critique pour scoring rapide)
        return False
