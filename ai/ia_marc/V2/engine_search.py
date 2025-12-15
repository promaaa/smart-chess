"""
IA-Marc V2 - Advanced Search Engine
====================================

Moteur de recherche avancé intégrant:
- Iterative Deepening
- Principal Variation Search (PVS) / NegaScout
- Transposition Table (Zobrist Hashing)
- Advanced Move Ordering (Killer, History, MVV-LVA)
- Quiescence Search
- Time Management
- Opening Book (Cerebellum.bin / JSON)

Optimisé pour Raspberry Pi 5.
"""

import time
from typing import List, Optional, Tuple

import chess
from engine_config import EngineConfig
from engine_opening import OpeningBook
from engine_ordering import MoveOrderer
from engine_tt import TranspositionTable, TTEntryType
from engine_see import see  # SEE pour pruning en Q-search

# Constantes de recherche
INFINITY = 999999
MATE_SCORE = 90000
MAX_PLY = 64


class IterativeDeepeningSearch:
    """
    Moteur de recherche principal utilisant l'approfondissement itératif.
    """

    def __init__(self, brain, config: EngineConfig = None):
        self.brain = brain
        self.config = config or EngineConfig()

        # Composants
        self.tt = TranspositionTable(size_mb=self.config.tt_size_mb)
        self.orderer = MoveOrderer()
        self.opening_book = OpeningBook(
            book_path=self.config.opening_book_path, book_type="auto"
        )

        # Charger le livre d'ouvertures si activé
        if self.config.use_opening_book:
            self.opening_book.load()

        # État de la recherche
        self.nodes = 0
        self.start_time = 0
        self.time_limit = 0
        self.stop_flag = False
        self.seldepth = 0

        # Statistiques
        self.last_info = {}

    def clear(self):
        """Réinitialise le moteur."""
        self.tt.clear()
        self.orderer.clear()
        # On ne recharge pas le livre pour gagner du temps

    def search(
        self, board: chess.Board, time_limit: float = 5.0, depth_limit: int = 20
    ) -> Optional[chess.Move]:
        """
        Lance une recherche itérative avec gestion du temps et fenêtres d'aspiration.

        TECHNIQUES IMPLÉMENTÉES ICI :
        1.  Approfondissement Itératif (Iterative Deepening): La recherche ne commence pas
            directement à la profondeur maximale, mais par une recherche à profondeur 1,
            puis 2, puis 3, et ainsi de suite. Avantages :
            - Fournit un coup utilisable à tout moment si le temps est écoulé.
            - Améliore grandement l'ordonnancement des coups (move ordering) pour les
              recherches plus profondes, car le meilleur coup d'une itération est
              susceptible d'être le meilleur de la suivante (PV-move).

        2.  Fenêtres d'Aspiration (Aspiration Windows): Au lieu de rechercher dans une
            fenêtre de score infinie (-inf, +inf), on "aspire" à ce que le score
            de la nouvelle recherche soit proche de celui de la recherche précédente.
            On lance donc la recherche avec une fenêtre très étroite (ex: [score-50, score+50]).
            Si le score réel tombe dans cette fenêtre, la recherche est beaucoup plus rapide
            car plus d'élagages alpha-bêta se produisent. Si le score tombe en dehors,
            la recherche a échoué ("fail-high" ou "fail-low") et doit être relancée
            avec une fenêtre plus large ou infinie.
        """
        self.start_time = time.time()
        self.time_limit = time_limit
        self.nodes = 0
        self.stop_flag = False
        self.tt.new_search()
        self.orderer.age_history()

        # 1. Consulter le livre d'ouvertures
        if self.config.use_opening_book:
            book_move = self.opening_book.probe(
                board,
                elo_level=self.config.difficulty_level.elo,
                variety=(self.config.opening_variety > 1),
            )
            if book_move:
                print(f"info string Book move: {book_move}")
                return book_move

        best_move = None
        
        # Configuration des fenêtres d'aspiration
        use_aspiration = self.config.use_aspiration_windows
        aspiration_window = self.config.aspiration_window_size
        alpha, beta = -INFINITY, INFINITY

        # Time Management: Soft et Hard bounds
        # Soft bound: peut être dépassé si le score s'améliore
        # Hard bound: arrêt forcé
        soft_bound = self.time_limit * 0.4  # 40% du temps
        hard_bound = self.time_limit * 0.85  # 85% du temps
        
        # Boucle d'approfondissement itératif
        prev_score = 0
        for current_depth in range(1, depth_limit + 1):
            self.seldepth = current_depth

            # Appliquer la fenêtre d'aspiration à partir de la profondeur 5
            if use_aspiration and current_depth >= 5:
                alpha = prev_score - aspiration_window
                beta = prev_score + aspiration_window
            else:
                alpha = -INFINITY
                beta = INFINITY

            # Lancer la recherche pour la profondeur actuelle
            score = self.negamax(board, current_depth, alpha, beta, 0, do_null=True)

            # Si la recherche a échoué (hors de la fenêtre), relancer avec une fenêtre plus large
            if use_aspiration and (score <= alpha or score >= beta):
                alpha, beta = -INFINITY, INFINITY
                score = self.negamax(board, current_depth, alpha, beta, 0, do_null=True)

            if self.stop_flag:
                break

            # Récupérer le meilleur coup de la TT
            pv_move = self.tt.get_pv_move(board)
            if pv_move:
                best_move = pv_move

            # Afficher info UCI
            elapsed = time.time() - self.start_time
            nps = int(self.nodes / (elapsed + 0.001))
            score_str = f"cp {score}"
            if abs(score) > MATE_SCORE - 1000:
                mate_in = (MATE_SCORE - abs(score) + 1) // 2
                if score < 0: mate_in = -mate_in
                score_str = f"mate {mate_in}"
            print(
                f"info depth {current_depth} seldepth {self.seldepth} score {score_str} nodes {self.nodes} nps {nps} time {int(elapsed * 1000)} pv {best_move.uci() if best_move else ''}"
            )

            # Gestion du temps sophistiquée
            # Si mat détecté, arrêter
            if abs(score) > MATE_SCORE - 100: 
                break
            
            # Hard bound: toujours respecté
            if self.time_limit > 0 and elapsed > hard_bound:
                break
            
            # Soft bound: peut être dépassé si le score s'améliore
            if self.time_limit > 0 and elapsed > soft_bound:
                # Si le score baisse ou stagne, arrêter
                score_improvement = score - prev_score
                if score_improvement < 20:  # Moins de 20cp d'amélioration
                    break
                # Sinon, continuer jusqu'au hard bound

            prev_score = score

        return best_move

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        do_null: bool = True,
    ) -> int:
        """
        Recherche Alpha-Beta NegaMax avec PVS, élagage par coup nul et LMR.

        TECHNIQUES IMPLÉMENTÉES ICI :
        1.  Recherche de Variation Principale (PVS - Principal Variation Search):
            C'est une optimisation d'Alpha-Bêta. L'idée est que le premier coup
            (le mieux ordonné) est probablement le meilleur. On le recherche donc 
            avec une fenêtre alpha-bêta complète. Pour les autres coups, on les
            recherche avec une "fenêtre nulle" (zero-width window) de la forme
            (alpha, alpha+1), ce qui est beaucoup plus rapide. Si un coup dans cette 
            fenêtre nulle donne un score > alpha, une recherche complète est nécessaire.

        2.  Élagage par Coup Nul (Null Move Pruning): Une heuristique agressive.
            On donne le trait à l'adversaire ("passe son tour") et on lance
            une recherche à profondeur réduite (R). Si le score obtenu est toujours
            supérieur à bêta, on suppose que la position est si forte qu'on peut
            élaguer la branche. Ne s'applique pas en fin de partie (zugzwang).
        
        3.  Late Move Reduction (LMR): Les coups qui apparaissent tard dans la liste
            de coups triée sont probablement mauvais. On les recherche donc à une
            profondeur réduite pour gagner du temps. Si le coup s'avère prometteur,
            on le recherche à nouveau à la profondeur normale.
        """
        if (self.nodes & 2047) == 0: self.check_time()
        if self.stop_flag: return 0
        self.nodes += 1
        self.seldepth = max(self.seldepth, ply)

        tt_hit, tt_score, tt_move = self.tt.probe(board, depth, alpha, beta, ply)
        if tt_hit: return tt_score

        # Internal Iterative Reduction (IIR)
        # Réduit la profondeur si pas de coup TT pour forcer une recherche rapide
        if not tt_hit and depth >= 4:
            depth -= 1

        if depth <= 0: return self.quiescence(board, alpha, beta, ply)
        if board.is_game_over(): return self.finish_game_score(board, ply)

        in_check = board.is_check()

        # Null Move Pruning
        if do_null and depth >= 3 and not in_check and any(board.pieces(pt, board.turn) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]):
            R = 3 if depth >= 6 else 2
            board.push(chess.Move.null())
            null_score = -self.negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, do_null=False)
            board.pop()
            if null_score >= beta:
                return beta

        # Reverse Futility Pruning (RFP)
        # Si éval statique >> beta, position trop bonne, on peut couper
        if depth <= 4 and not in_check and abs(beta) < MATE_SCORE - 1000:
            eval_score = self.brain.evaluate(board)
            margin = 120 * depth  # Marge conservative: 120cp par profondeur
            if eval_score - margin >= beta:
                return eval_score

        moves = list(board.legal_moves)
        if not moves: return self.finish_game_score(board, ply)
        moves = self.orderer.order_moves(board, moves, depth, pv_move=tt_move)

        best_score = -INFINITY
        best_move = None
        tt_flag = TTEntryType.UPPER
        
        # Évaluation statique pour Futility Pruning
        static_eval = self.brain.evaluate(board) if not in_check else 0
        
        # Compteur de coups calmes pour Late Move Pruning
        quiet_count = 0

        for i, move in enumerate(moves):
            is_capture = board.is_capture(move)
            is_promotion = move.promotion is not None
            is_quiet = not is_capture and not is_promotion
            
            # Futility Pruning: élague les coups calmes qui ne peuvent améliorer alpha
            # Formule de BoyChesser: eval + margin < alpha
            if depth <= 3 and not in_check and is_quiet and best_score > -MATE_SCORE + 1000:
                futility_margin = 200 * depth  # 200cp par profondeur
                if static_eval + futility_margin < alpha:
                    continue  # Skip ce coup
            
            # Late Move Pruning: arrête après N coups calmes
            # Formule de BoyChesser: moveIdx > threshold(depth)
            if depth <= 5 and not in_check and is_quiet and best_score > -MATE_SCORE + 1000:
                # Seuil: plus on est profond, plus on peut chercher de coups
                # depth 1: 7 coups, depth 2: 11, depth 3: 15, depth 4: 19, depth 5: 23
                lmp_threshold = 3 + depth * depth
                if quiet_count >= lmp_threshold:
                    break  # Arrête la boucle de coups
            
            if is_quiet:
                quiet_count += 1
            
            # Check Extension: étend si en échec (in_check calculé AVANT le coup)
            # Passed Pawn Extension: étend si pion atteint 7ème/2ème rangée
            extension = 0
            if in_check:
                extension = 1
            else:
                # Vérifier si c'est un coup de pion
                piece = board.piece_at(move.from_square)
                if piece and piece.piece_type == chess.PAWN:
                    rank = chess.square_rank(move.to_square)
                    # Blanc: pion sur rank 6 (7ème rangée, 0-indexed)
                    # Noir: pion sur rank 1 (2ème rangée, 0-indexed)
                    if (piece.color == chess.WHITE and rank == 6) or \
                       (piece.color == chess.BLACK and rank == 1):
                        extension = 1
            
            board.push(move)
            
            # Calcul de la nouvelle profondeur avec extension
            new_depth = depth - 1 + extension
            
            # Late Move Reduction (LMR)
            # Ne pas réduire si extension active
            reduction = 0
            if self.config.use_late_move_reduction and i >= self.config.lmr_threshold and depth >= 3 and extension == 0 and not is_capture and not is_promotion:
                reduction = 1
                if i >= 6 and depth >= 4: reduction = 2

            # Principal Variation Search (PVS)
            if i == 0:
                score = -self.negamax(board, new_depth, -beta, -alpha, ply + 1, True)
            else:
                # Null Window Search avec réduction LMR
                score = -self.negamax(board, new_depth - reduction, -alpha - 1, -alpha, ply + 1, True)
                # Re-recherche si LMR a échoué
                if score > alpha and reduction > 0:
                    score = -self.negamax(board, new_depth, -alpha - 1, -alpha, ply + 1, True)
                # Re-recherche complète si dans la fenêtre
                if score > alpha and score < beta:
                    score = -self.negamax(board, new_depth, -beta, -alpha, ply + 1, True)
            
            board.pop()
            if self.stop_flag: return 0

            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    tt_flag = TTEntryType.EXACT
                    if score >= beta:
                        self.tt.store(board, depth, score, TTEntryType.LOWER, move, ply)
                        if not board.is_capture(move): self.orderer.add_killer(move, depth)
                        self.orderer.update_history(move, depth, True)
                        return beta
                    else:
                        self.orderer.update_history(move, depth, False)
        
        self.tt.store(board, depth, best_score, tt_flag, best_move, ply)
        return best_score

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """
        Recherche de quiescence pour stabiliser l'évaluation.

        TECHNIQUES IMPLÉMENTÉES ICI :
        1.  Recherche de Quiescence : Étend la recherche en explorant uniquement
            les captures et promotions pour éviter de mal évaluer une position
            tactiquement instable (effet d'horizon).

        2.  Élagage Delta (Delta Pruning) : Si l'évaluation statique plus la
            valeur de la plus grosse pièce prenable (dame) n'atteint pas alpha,
            on élague, car aucune capture ne pourra améliorer le score suffisamment.
        """
        if (self.nodes & 2047) == 0: self.check_time()
        if self.stop_flag: return 0
        self.nodes += 1
        
        if ply > MAX_PLY - 1: return self.brain.evaluate(board)
        
        stand_pat = self.brain.evaluate(board)
        if stand_pat >= beta: return beta

        # Delta Pruning
        if stand_pat < alpha - 975: # 975 = valeur d'une dame
            return alpha
        
        if stand_pat > alpha: alpha = stand_pat

        moves = list(board.generate_legal_captures())
        moves = self.orderer.order_moves(board, moves, 0)

        for move in moves:
            # SEE Pruning: élague les captures avec échange négatif
            # TinyHugeBot: "SEE forward pruning"
            if not see(board, move, threshold=-100):
                # Capture perdante (perd plus de 100cp), skip
                continue
            
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if self.stop_flag: return 0

            if score >= beta: return beta
            if score > alpha: alpha = score

        return alpha

    def finish_game_score(self, board: chess.Board, ply: int) -> int:
        """Retourne le score de fin de partie (Mat ou Pat)."""
        if board.is_checkmate():
            return -MATE_SCORE + ply
        # Pour les nulles (pat, matériel insuffisant), le score est 0.
        # Le mépris est géré au niveau de la recherche, pas dans l'évaluation d'une nulle forcée.
        return 0

    def check_time(self):
        """Vérifie si le temps est écoulé."""
        if self.time_limit > 0 and time.time() - self.start_time > self.time_limit:
            self.stop_flag = True

    def stop(self):
        """Arrête la recherche."""
        self.stop_flag = True

    def get_stats(self) -> dict:
        """Retourne les statistiques de la dernière recherche."""
        elapsed = time.time() - self.start_time
        nps = int(self.nodes / (elapsed + 0.001))
        
        stats = {
            "nodes": self.nodes,
            "time": elapsed,
            "nps": nps,
            "depth": self.seldepth, # Using seldepth as max depth reached
            "score": 0 # Placeholder, score is local to search
        }
        
        if hasattr(self, 'searcher') and self.searcher.tt:
            stats['tt_stats'] = self.searcher.tt.get_stats()
        elif self.tt:
            stats['tt_stats'] = self.tt.get_stats()
        return stats

# Alias pour compatibilité
SearchEngine = IterativeDeepeningSearch