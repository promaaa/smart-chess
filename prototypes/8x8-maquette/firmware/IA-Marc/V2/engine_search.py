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
- Opening Book (Cerebellum / JSON)

Optimisé pour Raspberry Pi 5.
"""

import time
import chess
from typing import Optional, Tuple, List

from engine_tt import TranspositionTable, TTEntryType
from engine_ordering import MoveOrderer
from engine_config import EngineConfig
from engine_opening import OpeningBook

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
            book_path=self.config.opening_book_path, 
            book_type="auto"
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

    def search(self, board: chess.Board, time_limit: float = 5.0, depth_limit: int = 20) -> Optional[chess.Move]:
        """
        Lance une recherche itérative.
        """
        self.start_time = time.time()
        self.time_limit = time_limit
        self.nodes = 0
        self.stop_flag = False
        self.tt.new_search()
        self.orderer.age_history() # Vieillir l'historique périodiquement

        # 1. Consulter le livre d'ouvertures
        if self.config.use_opening_book:
            book_move = self.opening_book.probe(
                board, 
                elo_level=self.config.difficulty_level.elo,
                variety=(self.config.opening_variety > 1)
            )
            if book_move:
                print(f"info string Book move: {book_move}")
                return book_move

        best_move = None
        alpha = -INFINITY
        beta = INFINITY

        # Iterative Deepening
        for current_depth in range(1, depth_limit + 1):
            self.seldepth = 0
            
            # Aspiration Windows (optionnel, pour l'instant simple PVS)
            # Si on avait un score précédent, on pourrait réduire la fenêtre alpha-beta
            
            score = self.negamax(board, current_depth, alpha, beta, 0, do_null=True)
            
            if self.stop_flag:
                break
            
            # Récupérer le meilleur coup de la TT pour cette profondeur
            pv_move = self.tt.get_pv_move(board)
            if pv_move:
                best_move = pv_move
            
            # Afficher info UCI
            elapsed = time.time() - self.start_time
            nps = int(self.nodes / (elapsed + 0.001))
            
            # Formater le score
            score_str = f"cp {score}"
            if abs(score) > MATE_SCORE - 1000:
                mate_in = (MATE_SCORE - abs(score) + 1) // 2
                if score < 0: mate_in = -mate_in
                score_str = f"mate {mate_in}"
            
            print(f"info depth {current_depth} seldepth {self.seldepth} score {score_str} nodes {self.nodes} nps {nps} time {int(elapsed * 1000)} pv {best_move.uci() if best_move else ''}")
            
            # Arrêt si mat trouvé
            if abs(score) > MATE_SCORE - 100:
                break
                
            # Gestion du temps
            if self.time_limit > 0 and elapsed > self.time_limit * 0.6:
                # Si on a utilisé plus de 60% du temps, on ne lance pas une nouvelle profondeur
                break

        return best_move

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, do_null: bool = True) -> int:
        """
        Recherche Alpha-Beta avec PVS (Principal Variation Search).
        """
        # 1. Vérification rapide (Temps & Noeuds)
        if self.nodes & 2047 == 0:
            self.check_time()
        
        if self.stop_flag:
            return 0
        
        self.nodes += 1
        self.seldepth = max(self.seldepth, ply)

        # 2. Transposition Table Lookup
        # On cherche si la position est déjà connue
        tt_hit, tt_score, tt_move = self.tt.probe(board, depth, alpha, beta, ply)
        if tt_hit:
            return tt_score

        # 3. Conditions terminales
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)
            
        if board.is_game_over():
            return self.finish_game_score(board, ply)

        # 4. Null Move Pruning (Élagage coup nul)
        # Si on a une bonne position, on passe notre tour pour voir si on est toujours bien
        # Ne pas faire en fin de partie (zugzwang) ou si en échec
        if do_null and depth >= 3 and not board.is_check() and ply > 0:
            board.push(chess.Move.null())
            null_score = -self.negamax(board, depth - 3, -beta, -beta + 1, ply + 1, do_null=False)
            board.pop()
            if self.stop_flag: return 0
            if null_score >= beta:
                return beta

        # 5. Génération et tri des coups
        moves = list(board.legal_moves)
        if not moves:
            return self.finish_game_score(board, ply)
            
        # Utiliser le coup de la TT comme PV move s'il existe
        moves = self.orderer.order_moves(board, moves, depth, pv_move=tt_move)
        
        # 6. Boucle principale PVS
        best_score = -INFINITY
        best_move = None
        tt_flag = TTEntryType.UPPER # Par défaut, on a pas dépassé alpha

        for i, move in enumerate(moves):
            board.push(move)
            
            # PVS (Principal Variation Search)
            if i == 0:
                # Premier coup (PV): recherche complète
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, do_null=True)
            else:
                # Autres coups: recherche avec fenêtre nulle (Null Window Search)
                # On suppose que le coup ne va pas améliorer alpha
                score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, do_null=True)
                
                # Si la recherche échoue (score > alpha), on doit refaire une recherche complète
                if alpha < score < beta:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, do_null=True)
            
            board.pop()
            
            if self.stop_flag:
                return 0

            if score > best_score:
                best_score = score
                best_move = move
                
                if score > alpha:
                    alpha = score
                    tt_flag = TTEntryType.EXACT # On a trouvé un score exact (PV)
                    
                    if score >= beta:
                        # Beta Cutoff
                        self.tt.store(board, depth, score, TTEntryType.LOWER, move, ply)
                        self.orderer.update_history(move, depth, caused_cutoff=True)
                        if not board.is_capture(move):
                            self.orderer.add_killer(move, depth)
                        return beta
                    else:
                        # Bon coup mais pas cutoff
                        self.orderer.update_history(move, depth, caused_cutoff=False)

        # 7. Stockage TT
        self.tt.store(board, depth, best_score, tt_flag, best_move, ply)
        
        return best_score

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """
        Recherche de quiescence pour stabiliser l'évaluation aux feuilles.
        Ne regarde que les captures.
        """
        if self.nodes & 2047 == 0:
            self.check_time()
        if self.stop_flag:
            return 0
            
        self.nodes += 1
        
        # 1. Stand-pat (évaluation statique)
        stand_pat = self.brain.evaluate(board)
        
        if stand_pat >= beta:
            return beta
            
        if stand_pat > alpha:
            alpha = stand_pat
            
        # 2. Générer les captures
        # Note: python-chess generate_legal_captures inclut les promotions
        moves = list(board.generate_legal_captures())
        moves = self.orderer.order_moves(board, moves, 0) # Tri simple MVV-LVA
        
        for move in moves:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            
            if self.stop_flag: return 0
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha

    def finish_game_score(self, board: chess.Board, ply: int) -> int:
        """Retourne le score de fin de partie (Mat ou Pat).
        
        Applique le facteur de mépris (contempt) pour les nulles.
        Un contempt > 0 rend le moteur plus ambitieux (évite les nulles).
        """
        if board.is_checkmate():
            return -MATE_SCORE + ply # Mat plus rapide est meilleur
        
        # Appliquer le contempt pour les nulles (pat, matériel insuffisant, etc.)
        # Score négatif = le moteur considère la nulle comme désavantageuse
        contempt = self.config.difficulty_level.contempt
        return -contempt # Pat ou match nul

    def check_time(self):
        """Vérifie si le temps est écoulé."""
        if self.time_limit > 0:
            if time.time() - self.start_time > self.time_limit:
                self.stop_flag = True
    
    def stop(self):
        """Arrête la recherche."""
        self.stop_flag = True


# Alias pour compatibilité avec engine_main.py
SearchEngine = IterativeDeepeningSearch
