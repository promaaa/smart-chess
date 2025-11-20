import random
import time

import chess

# Valeurs infinies pour l'élagage
INFINITY = 999999
MATE_SCORE = 90000


class Searcher:
    def __init__(self, engine_brain):
        self.brain = engine_brain
        self.nodes = 0
        self.start_time = 0
        self.time_limit = 0
        self.stop_flag = False
        self.elo = 2000  # Niveau par défaut

    def set_elo(self, elo):
        """Permet de régler le niveau de difficulté dynamiquement."""
        self.elo = elo
        print(f"--- Niveau IA réglé sur ELO {self.elo} ---")

    def order_moves(self, board, moves):
        """
        Trie les coups pour optimiser Alpha-Beta (Move Ordering).
        Priorité : Captures > Promotions > Autres.
        """

        def score_move(move):
            if board.is_capture(move):
                # MVV-LVA simplifié : on priorise toute capture
                return 10000
            if move.promotion:
                return 9000
            return 0

        moves.sort(key=score_move, reverse=True)
        return moves

    def get_best_move(self, board, time_limit=None):
        """
        Méthode intelligente qui choisit la stratégie selon l'ELO.
        """
        # 1. Configuration selon l'ELO
        if self.elo < 800:
            # Niveau Débutant : Profondeur 1, 20% d'aléatoire pur
            if random.random() < 0.20:
                legal_moves = list(board.legal_moves)
                return random.choice(legal_moves) if legal_moves else None
            depth_limit = 1
            search_time = 0.5

        elif self.elo < 1200:
            # Niveau Amateur : Profondeur 2 fixée
            depth_limit = 2
            search_time = 1.0

        elif self.elo < 1600:
            # Niveau Club : Profondeur 3-4
            depth_limit = 4
            search_time = 3.0

        else:
            # Niveau Maître/Expert (Max RPi 5)
            # On laisse le temps dicter la profondeur
            depth_limit = 20
            search_time = time_limit if time_limit else 5.0

        return self._search_iterative(board, search_time, depth_limit)

    def _search_iterative(self, board, time_limit, max_depth_limit):
        """
        Recherche avec approfondissement itératif (Iterative Deepening).
        """
        self.start_time = time.time()
        self.time_limit = time_limit
        self.nodes = 0
        self.stop_flag = False

        best_move = None
        best_score = -INFINITY

        # On commence toujours prof 1 pour avoir au moins un coup à jouer
        current_depth = 1

        print(
            f"--- Recherche ELO {self.elo} (Max {time_limit}s / Prof {max_depth_limit}) ---"
        )

        while current_depth <= max_depth_limit:
            try:
                # Recherche Alpha-Beta
                # Note: Pour les bas ELOs (ex: prof 1 ou 2), on pourrait ajouter du bruit ici
                # mais limiter la profondeur est souvent suffisant pour simuler l'erreur humaine.

                best_move_this_depth, score = self.search_root(board, current_depth)

                # Si le temps a coupé la recherche en plein milieu, on garde le résultat précédent
                if self.stop_flag:
                    break

                best_move = best_move_this_depth
                best_score = score

                # Stats pour le debug
                duration = time.time() - self.start_time
                nps = int(self.nodes / (duration + 0.001))
                info = f"Prof: {current_depth} | Score: {best_score} | Coup: {best_move} | Noeuds: {self.nodes} ({nps} n/s)"
                print(info)

                # Arrêt si mat trouvé
                if abs(best_score) > MATE_SCORE - 100:
                    break

                current_depth += 1

                # Si on a dépassé le temps après une itération complète
                if time.time() - self.start_time > self.time_limit:
                    break

            except TimeoutError:
                break

        return best_move

    def search_root(self, board, depth):
        """Gère la racine de l'arbre."""
        best_val = -INFINITY
        best_move = None

        moves = list(board.legal_moves)
        if not moves:
            return None, 0

        self.order_moves(board, moves)

        alpha = -INFINITY
        beta = INFINITY

        for move in moves:
            board.push(move)
            value = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if self.check_time():
                return best_move, best_val

            if value > best_val:
                best_val = value
                best_move = move

            alpha = max(alpha, value)

        return best_move, best_val

    def negamax(self, board, depth, alpha, beta):
        self.nodes += 1

        # Check temps tous les 2048 noeuds
        if (self.nodes & 2047) == 0:
            self.check_time()

        if self.stop_flag:
            return 0

        if depth == 0:
            return self.quiescence(board, alpha, beta)

        if board.is_game_over():
            return self.brain.evaluate(board)

        max_score = -INFINITY
        moves = list(board.legal_moves)

        # Optimisation : Si pas de coups légaux (Mat ou Pat déjà géré par is_game_over normalement,
        # mais sécurité pour quiescence nodes qui deviennent des feuilles)
        if not moves:
            return self.brain.evaluate(board)

        self.order_moves(board, moves)

        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > max_score:
                max_score = score

            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Beta Cutoff

        return max_score

    def quiescence(self, board, alpha, beta):
        self.nodes += 1
        if (self.nodes & 2047) == 0:
            self.check_time()
        if self.stop_flag:
            return 0

        stand_pat = self.brain.evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        for move in board.generate_legal_captures():
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def check_time(self):
        if self.time_limit > 0 and (time.time() - self.start_time > self.time_limit):
            self.stop_flag = True
            return True
        return False
