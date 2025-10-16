import time
import hashlib
from alphabeta_engine import AlphaBetaEngine


class TranspositionEntry:
    """Entrée de la table de transposition"""
    def __init__(self, depth, score, flag, best_move=None, age=0):
        self.depth = depth          # Profondeur de recherche
        self.score = score          # Score évalué
        self.flag = flag           # Type: 'exact', 'lower', 'upper'
        self.best_move = best_move  # Meilleur coup trouvé
        self.age = age             # Âge de l'entrée pour le remplacement


class IterativeDeepeningAlphaBeta(AlphaBetaEngine):
    """Version avec approfondissement itératif, gestion du temps et table de transposition"""
    
    def __init__(self, max_time=5.0, max_depth=10, evaluator=None, tt_size=1000000):
        super().__init__(max_depth, evaluator)
        self.max_time = max_time
        self.start_time = None
        # Fixed-size transposition table: array of slots. Each slot holds a tuple
        # (key:int, depth:int, score:int/float, flag:str, best_move, age:int)
        # or None if empty.
        self.tt_size = tt_size         # Taille (nombre de slots)
        self.transposition_table = [None] * self.tt_size
        self.search_age = 0            # Âge de la recherche courante
        self.tt_hits = 0              # Statistiques TT
        self.tt_misses = 0
        
    def get_best_move_with_time_limit(self, chess):
        """
        Trouve le meilleur coup avec une limite de temps.
        Utilise l'approfondissement itératif avec table de transposition.
        """
        self.start_time = time.time()
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        self.tt_hits = 0
        self.tt_misses = 0
        self.search_age += 1  # Nouvelle recherche
        
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
                    # record last depth reached by search
                    try:
                        self.last_depth = int(depth)
                    except Exception:
                        self.last_depth = depth
                    elapsed = time.time() - self.start_time
                    tt_efficiency = self.tt_hits / (self.tt_hits + self.tt_misses) * 100 if (self.tt_hits + self.tt_misses) > 0 else 0
                    print(f"Depth {depth}: Best move = {self._format_move(best_move)}, "
                          f"Time: {elapsed:.2f}s, Nodes: {self.nodes_evaluated}, "
                          f"TT hits: {self.tt_hits} ({tt_efficiency:.1f}%)")
            except TimeoutError:
                print(f"Recherche interrompue à la profondeur {depth}")
                break
        
        # Count non-empty slots
        filled = sum(1 for e in self.transposition_table if e is not None)
        print(f"Table de transposition: {filled} entrées (slots: {self.tt_size})")
        # Ensure we return a canonical tuple (from_sq:int, to_sq:int, promotion)
        if best_move is None:
            return None

        try:
            # best_move can be tuple-like or an object; try to canonicalize
            if isinstance(best_move, tuple) and len(best_move) >= 2:
                fm = (int(best_move[0]), int(best_move[1]), best_move[2] if len(best_move) >= 3 else None)
            elif hasattr(best_move, 'from_sq') and hasattr(best_move, 'to_sq'):
                fm = (int(best_move.from_sq), int(best_move.to_sq), getattr(best_move, 'promotion', None))
            else:
                # As a last resort, try to parse formatted string like e2e4
                s = self._format_move(best_move)
                from_sq = ord(s[0]) - ord('a') + (int(s[1]) - 1) * 8
                to_sq = ord(s[2]) - ord('a') + (int(s[3]) - 1) * 8
                promo = s[4] if len(s) > 4 else None
                fm = (from_sq, to_sq, promo)
        except Exception:
            # If canonicalization fails, return the raw best_move (caller must handle)
            return best_move

        # store formatted last move for debugging/consumption
        try:
            self.last_move_formatted = self._format_move(fm)
        except Exception:
            self.last_move_formatted = None

        return fm
    
    def _search_at_depth(self, chess, target_depth):
        """Recherche à une profondeur donnée avec table de transposition"""
        # Consulter la table de transposition pour le meilleur coup
        position_hash = self._get_position_hash(chess)
        _, tt_best_move = self._probe_tt(position_hash, target_depth, float('-inf'), float('inf'))
        
        legal_moves = self._get_all_legal_moves(chess)
        ordered_moves = self._order_moves_with_tt(chess, legal_moves, tt_best_move)
        
        best_move = None
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
        
        # Stocker le résultat dans la table de transposition
        flag = 'exact'  # Au niveau racine, c'est toujours exact
        self._store_tt_entry(position_hash, target_depth, best_score, flag, best_move)
        
        return best_move
    
    def alphabeta_with_timeout(self, chess, depth, alpha, beta, is_maximizing):
        """Version d'alpha-beta avec vérification du temps et table de transposition"""
        # Vérifier le temps restant
        if time.time() - self.start_time > self.max_time:
            raise TimeoutError("Temps limite atteint")
        
        return self.alphabeta_with_tt(chess, depth, alpha, beta, is_maximizing)
    
    def _get_position_hash(self, chess):
        """Génère un hash unique pour la position actuelle.

        Essaie d'utiliser Zobrist (int rapide) si `compute_zobrist` est disponible
        dans `optimized_chess`. Sinon, retombe sur l'ancien MD5 string.
        """
        try:
            # Import local pour éviter la dépendance circulaire au top-level
            from optimized_chess import compute_zobrist
            key = compute_zobrist(chess)
            # Utiliser un int convertible en clé de dict directement
            return key
        except Exception:
            # Fallback: ancienne méthode MD5 (string)
            position_str = ""
            for square in range(64):
                piece = self._get_piece_at(chess, square)
                position_str += piece if piece else '.'

            position_str += 'W' if chess.white_to_move else 'B'

            if hasattr(chess, 'castling_rights'):
                for right in ['K', 'Q', 'k', 'q']:
                    position_str += right if chess.castling_rights.get(right, False) else '-'

            if hasattr(chess, 'en_passant_square') and chess.en_passant_square is not None:
                position_str += f"ep{chess.en_passant_square}"

            return hashlib.md5(position_str.encode()).hexdigest()
    
    def _store_tt_entry(self, position_hash, depth, score, flag, best_move=None):
        """Stocke une entrée dans la table de transposition"""
        # For fixed-size table we use index = position_hash % tt_size
        idx = int(position_hash) % self.tt_size

        # Each slot stores a tuple: (key, depth, score, flag, best_move, age)
        slot = self.transposition_table[idx]

        # Replace if slot is empty or new depth is greater or older (heuristic)
        if slot is None:
            self.transposition_table[idx] = (position_hash, depth, score, flag, best_move, self.search_age)
            return

        stored_key, stored_depth, _, _, _, stored_age = slot

        # Heuristic: prefer deeper entries; if equal, prefer newer entries
        if depth > stored_depth or (depth == stored_depth and self.search_age >= stored_age):
            self.transposition_table[idx] = (position_hash, depth, score, flag, best_move, self.search_age)
        # otherwise keep existing
    
    def _probe_tt(self, position_hash, depth, alpha, beta):
        """Interroge la table de transposition"""
        idx = int(position_hash) % self.tt_size
        slot = self.transposition_table[idx]
        if slot is None:
            self.tt_misses += 1
            return None, None

        stored_key, stored_depth, stored_score, stored_flag, stored_best_move, stored_age = slot

        # If key doesn't match (collision), treat as miss
        if stored_key != position_hash:
            self.tt_misses += 1
            return None, None

        self.tt_hits += 1

        # Vérifier si la profondeur est suffisante
        if stored_depth >= depth:
            if stored_flag == 'exact':
                return stored_score, stored_best_move
            elif stored_flag == 'lower' and stored_score >= beta:
                return stored_score, stored_best_move
            elif stored_flag == 'upper' and stored_score <= alpha:
                return stored_score, stored_best_move

        # Retourner au moins le meilleur coup même si le score n'est pas utilisable
        return None, stored_best_move
    
    def alphabeta_with_tt(self, chess, depth, alpha, beta, is_maximizing):
        """
        Algorithme alpha-beta avec table de transposition.
        """
        self.nodes_evaluated += 1
        original_alpha = alpha
        
        # Générer le hash de la position
        position_hash = self._get_position_hash(chess)
        
        # Consulter la table de transposition
        tt_score, tt_best_move = self._probe_tt(position_hash, depth, alpha, beta)
        if tt_score is not None:
            return tt_score
        
        # Condition d'arrêt: profondeur 0 ou fin de partie
        if depth == 0:
            score = self.evaluator.evaluate_position(chess)
            self._store_tt_entry(position_hash, depth, score, 'exact')
            return score
        
        legal_moves = self._get_all_legal_moves(chess)
        
        # Fin de partie (mat ou pat)
        if not legal_moves:
            if chess.is_in_check(chess.white_to_move):
                # Échec et mat
                mate_score = 20000 + depth
                score = -mate_score if is_maximizing else mate_score
                self._store_tt_entry(position_hash, depth, score, 'exact')
                return score
            else:
                # Pat
                self._store_tt_entry(position_hash, depth, 0, 'exact')
                return 0
        
        # Ordonner les coups avec priorité au meilleur coup TT
        ordered_moves = self._order_moves_with_tt(chess, legal_moves, tt_best_move)
        
        best_move = None
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                chess.move_piece(move[0], move[1], promotion=move[2])
                eval_score = self.alphabeta_with_tt(chess, depth - 1, alpha, beta, False)
                chess.undo_move()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                
                # Élagage beta
                if beta <= alpha:
                    self.pruned_branches += len(ordered_moves) - ordered_moves.index(move) - 1
                    break
            
            # Stocker dans la table de transposition
            if max_eval <= original_alpha:
                flag = 'upper'
            elif max_eval >= beta:
                flag = 'lower'
            else:
                flag = 'exact'
            
            self._store_tt_entry(position_hash, depth, max_eval, flag, best_move)
            return max_eval
            
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                chess.move_piece(move[0], move[1], promotion=move[2])
                eval_score = self.alphabeta_with_tt(chess, depth - 1, alpha, beta, True)
                chess.undo_move()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                
                # Élagage alpha
                if beta <= alpha:
                    self.pruned_branches += len(ordered_moves) - ordered_moves.index(move) - 1
                    break
            
            # Stocker dans la table de transposition
            if min_eval <= original_alpha:
                flag = 'upper'
            elif min_eval >= beta:
                flag = 'lower'
            else:
                flag = 'exact'
            
            self._store_tt_entry(position_hash, depth, min_eval, flag, best_move)
            return min_eval
    
    def _order_moves_with_tt(self, chess, moves, tt_best_move):
        """
        Ordonne les coups en donnant la priorité au meilleur coup de la table de transposition.
        """
        if not tt_best_move:
            return self._order_moves(chess, moves)
        
        # Séparer le meilleur coup TT des autres
        tt_move_found = False
        other_moves = []
        
        for move in moves:
            if (move[0] == tt_best_move[0] and 
                move[1] == tt_best_move[1] and 
                move[2] == tt_best_move[2]):
                tt_move_found = True
            else:
                other_moves.append(move)
        
        # Ordonner les autres coups normalement
        ordered_others = self._order_moves(chess, other_moves)
        
        # Mettre le coup TT en premier s'il existe
        if tt_move_found:
            return [tt_best_move] + ordered_others
        else:
            return ordered_others
    
    def clear_tt(self):
        """Vide la table de transposition"""
        # Reset fixed-size table
        self.transposition_table = [None] * self.tt_size
        self.search_age = 0
        print("Table de transposition vidée")