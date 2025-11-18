#!/usr/bin/env python3
"""
IA-Marc V2 - UCI Chess Engine Wrapper
======================================

Wrapper UCI (Universal Chess Interface) pour le moteur d'échecs IA-Marc V2.
Ce processus communique via stdin/stdout et ne contient AUCUN code matériel.

Compatible PyPy3 pour performances maximales (2-5x speedup).

Usage:
    python3 chess_engine_uci.py
    pypy3 chess_engine_uci.py

Communication:
    Interface → Moteur : stdin (commandes UCI)
    Moteur → Interface : stdout (réponses UCI)

Commandes UCI supportées:
    - uci          : Identification du moteur
    - isready      : Vérification disponibilité
    - position     : Définir position (startpos ou fen) + moves
    - go           : Lancer recherche (movetime, depth, infinite)
    - stop         : Arrêter recherche en cours
    - quit         : Terminer proprement
    - setoption    : Configuration (niveau, personnalité)

Auteur: Smart Chess Team
Version: 2.0
"""

import sys
import time
from typing import List, Optional

import chess

# Import des modules du moteur V2
try:
    from engine_brain import EvaluationEngine
    from engine_config import DIFFICULTY_LEVELS, EngineConfig
    from engine_search import IterativeDeepeningSearch
except ImportError:
    # Fallback vers V1 si V2 pas complètement implémenté
    try:
        sys.path.insert(0, "../V1")
        from engine_brain import Engine as EvaluationEngine
        # Fallback searcher if engine_search is missing (should not happen in prod)
        from chess_engine_uci import SimpleSearcher as IterativeDeepeningSearch 
    except ImportError:
        print("info string ERROR: Cannot import engine modules", file=sys.stderr)
        sys.exit(1)


# ============================================================================
# MOTEUR DE RECHERCHE SIMPLIFIÉ (Temporaire - En attendant V2 complète)
# ============================================================================

INFINITY = 999999
MATE_SCORE = 90000


class SimpleSearcher:
    """
    Moteur de recherche simplifié basé sur V1.
    Sera remplacé par la version V2 complète avec TT, Killer Moves, etc.
    """

    def __init__(self, engine_brain, config=None):
        self.brain = engine_brain
        self.config = config or EngineConfig()
        self.nodes = 0
        self.start_time = 0
        self.time_limit = 0
        self.stop_flag = False
        self.depth_limit = 20

        # Statistiques
        self.last_depth = 0
        self.last_score = 0
        self.last_pv = []

    def order_moves(self, board, moves):
        """Move ordering basique : Captures > Promotions > Autres."""

        def score_move(move):
            if board.is_capture(move):
                return 10000
            if move.promotion:
                return 9000
            return 0

        moves.sort(key=score_move, reverse=True)
        return moves

    def search(
        self, board: chess.Board, time_limit: float = 5.0, depth_limit: int = 20
    ) -> Optional[chess.Move]:
        """
        Recherche itérative avec time management.

        Args:
            board: Position à analyser
            time_limit: Temps max en secondes
            depth_limit: Profondeur max

        Returns:
            Meilleur coup trouvé
        """
        self.start_time = time.time()
        self.time_limit = time_limit
        self.depth_limit = depth_limit
        self.nodes = 0
        self.stop_flag = False

        best_move = None
        best_score = -INFINITY

        # Iterative Deepening
        for current_depth in range(1, depth_limit + 1):
            if self.stop_flag:
                break

            move, score = self.search_root(board, current_depth)

            if self.stop_flag:
                break

            if move:
                best_move = move
                best_score = score
                self.last_depth = current_depth
                self.last_score = score

            # Afficher info UCI
            elapsed = time.time() - self.start_time
            nps = int(self.nodes / (elapsed + 0.001))
            print(
                f"info depth {current_depth} score cp {score} nodes {self.nodes} nps {nps} time {int(elapsed * 1000)}"
            )

            # Arrêt anticipé si mat trouvé
            if abs(score) > MATE_SCORE - 100:
                break

            # Vérifier le temps
            if time.time() - self.start_time > self.time_limit * 0.9:
                break

        return best_move

    def search_root(self, board, depth):
        """Recherche à la racine."""
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
        """Recherche NegaMax avec Alpha-Beta."""
        self.nodes += 1

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
                break  # Beta cutoff

        return max_score

    def quiescence(self, board, alpha, beta):
        """Quiescence search pour stabiliser l'évaluation."""
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
        """Vérifie si le temps est écoulé."""
        if self.time_limit > 0:
            if time.time() - self.start_time > self.time_limit:
                self.stop_flag = True
                return True
        return False

    def stop(self):
        """Arrête la recherche en cours."""
        self.stop_flag = True


# ============================================================================
# MOTEUR UCI PRINCIPAL
# ============================================================================


class UCIEngine:
    """
    Implémentation du protocole UCI pour IA-Marc V2.
    """

    def __init__(self):
        """Initialise le moteur UCI."""
        self.board = chess.Board()
        self.config = EngineConfig()
        self.brain = EvaluationEngine()
        self.searcher = IterativeDeepeningSearch(self.brain, self.config)

        # État
        self.debug = False
        self.searching = False

        # Info moteur
        self.name = "IA-Marc V2"
        self.author = "Smart Chess Team"
        self.version = "2.0"

    def log_debug(self, message: str):
        """Log un message de debug."""
        if self.debug:
            print(f"info string DEBUG: {message}")

    # ========================================================================
    # COMMANDES UCI
    # ========================================================================

    def cmd_uci(self):
        """Commande: uci - Identification du moteur."""
        print(f"id name {self.name}")
        print(f"id author {self.author}")

        # Options configurables
        print(
            "option name Level type combo default Club var Enfant var Debutant var Amateur var Club var Competition var Expert var Maitre var Maximum"
        )
        print(
            "option name Personality type combo default Equilibre var Equilibre var Agressif var Defensif var Positionnel var Tactique var Materialiste"
        )
        print("option name Hash type spin default 256 min 16 max 2048")
        print("option name Threads type spin default 1 min 1 max 4")

        print("uciok")

    def cmd_debug(self, args: List[str]):
        """Commande: debug [on|off] - Active/désactive le mode debug."""
        if args and args[0] == "on":
            self.debug = True
            self.log_debug("Debug mode enabled")
        elif args and args[0] == "off":
            self.debug = False

    def cmd_isready(self):
        """Commande: isready - Indique que le moteur est prêt."""
        print("readyok")

    def cmd_setoption(self, args: List[str]):
        """
        Commande: setoption name <id> [value <x>]
        Configure une option du moteur.
        """
        if len(args) < 2:
            return

        try:
            # Parser "name X value Y"
            name_idx = args.index("name")
            option_name = args[name_idx + 1]

            value = None
            if "value" in args:
                value_idx = args.index("value")
                value = " ".join(args[value_idx + 1 :])

            # Appliquer l'option
            if option_name == "Level":
                self.config.set_level(value.upper())
                self.log_debug(f"Level set to {value}")

            elif option_name == "Personality":
                self.config.set_personality(value.upper())
                self.log_debug(f"Personality set to {value}")

            elif option_name == "Hash":
                self.config.tt_size_mb = int(value)
                self.log_debug(f"Hash size set to {value} MB")

            elif option_name == "Threads":
                self.config.threads = int(value)
                self.log_debug(f"Threads set to {value}")

        except (ValueError, IndexError) as e:
            self.log_debug(f"Error parsing setoption: {e}")

    def cmd_ucinewgame(self):
        """Commande: ucinewgame - Prépare une nouvelle partie."""
        self.board = chess.Board()
        # Réinitialiser les caches
        self.searcher.clear()
        self.log_debug("New game initialized")

    def cmd_position(self, args: List[str]):
        """
        Commande: position [fen <fenstring> | startpos] moves <move1> ... <movei>
        Définit la position actuelle.
        """
        if not args:
            return

        try:
            # Parser la position
            if args[0] == "startpos":
                self.board = chess.Board()
                moves_start = 1

            elif args[0] == "fen":
                # Trouver où commencent les moves
                if "moves" in args:
                    moves_idx = args.index("moves")
                    fen_parts = args[1:moves_idx]
                    moves_start = moves_idx
                else:
                    fen_parts = args[1:]
                    moves_start = len(args)

                fen = " ".join(fen_parts)
                self.board = chess.Board(fen)
            else:
                return

            # Appliquer les moves si présents
            if moves_start < len(args) and args[moves_start] == "moves":
                for move_str in args[moves_start + 1 :]:
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move in self.board.legal_moves:
                            self.board.push(move)
                        else:
                            self.log_debug(f"Illegal move: {move_str}")
                            break
                    except ValueError:
                        self.log_debug(f"Invalid move format: {move_str}")
                        break

            self.log_debug(f"Position set: {self.board.fen()}")

        except Exception as e:
            self.log_debug(f"Error parsing position: {e}")

    def cmd_go(self, args: List[str]):
        """
        Commande: go [searchmoves <move1> ... <movei>] [ponder] [wtime <x>] [btime <x>]
                     [winc <x>] [binc <x>] [movestogo <x>] [depth <x>] [nodes <x>]
                     [mate <x>] [movetime <x>] [infinite]
        Lance la recherche.
        """
        if self.searching:
            return

        self.searching = True

        # Parser les paramètres
        movetime = None
        depth = None
        infinite = False

        i = 0
        while i < len(args):
            if args[i] == "movetime" and i + 1 < len(args):
                movetime = int(args[i + 1]) / 1000.0  # Convertir ms en secondes
                i += 2
            elif args[i] == "depth" and i + 1 < len(args):
                depth = int(args[i + 1])
                i += 2
            elif args[i] == "infinite":
                infinite = True
                i += 1
            else:
                i += 1

        # Déterminer les paramètres de recherche selon le niveau
        if movetime is None and depth is None and not infinite:
            # Utiliser les paramètres du niveau configuré
            level = self.config.difficulty_level
            movetime = level.time_limit
            depth = level.depth_limit

        if depth is None:
            depth = 20

        if movetime is None:
            movetime = 5.0

        # Lancer la recherche
        try:
            best_move = self.searcher.search(
                self.board, time_limit=movetime, depth_limit=depth
            )

            if best_move:
                print(f"bestmove {best_move.uci()}")
            else:
                # Pas de coup trouvé (position terminale?)
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    print(f"bestmove {legal_moves[0].uci()}")
                else:
                    print("bestmove 0000")

        except Exception as e:
            self.log_debug(f"Error during search: {e}")
            print("bestmove 0000")

        finally:
            self.searching = False

    def cmd_stop(self):
        """Commande: stop - Arrête la recherche en cours."""
        if self.searching:
            self.searcher.stop()
            self.log_debug("Search stopped")

    def cmd_quit(self):
        """Commande: quit - Termine le moteur proprement."""
        self.log_debug("Exiting engine")
        sys.exit(0)

    # ========================================================================
    # BOUCLE PRINCIPALE
    # ========================================================================

    def run(self):
        """
        Boucle principale du moteur UCI.
        Lit les commandes depuis stdin et répond sur stdout.
        """
        self.log_debug(f"{self.name} started")

        while True:
            try:
                # Lire une commande
                line = sys.stdin.readline().strip()

                if not line:
                    continue

                # Parser la commande
                parts = line.split()
                if not parts:
                    continue

                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []

                # Dispatcher les commandes
                if command == "uci":
                    self.cmd_uci()

                elif command == "debug":
                    self.cmd_debug(args)

                elif command == "isready":
                    self.cmd_isready()

                elif command == "setoption":
                    self.cmd_setoption(args)

                elif command == "ucinewgame":
                    self.cmd_ucinewgame()

                elif command == "position":
                    self.cmd_position(args)

                elif command == "go":
                    self.cmd_go(args)

                elif command == "stop":
                    self.cmd_stop()

                elif command == "quit":
                    self.cmd_quit()

                else:
                    self.log_debug(f"Unknown command: {command}")

            except EOFError:
                # stdin fermé
                break

            except KeyboardInterrupt:
                break

            except Exception as e:
                self.log_debug(f"Error processing command: {e}")
                import traceback

                traceback.print_exc(file=sys.stderr)


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================


def main():
    """Point d'entrée principal."""
    # Désactiver le buffering pour stdin/stdout
    sys.stdin.reconfigure(line_buffering=True)
    sys.stdout.reconfigure(line_buffering=True)

    # Créer et lancer le moteur
    engine = UCIEngine()
    engine.run()


if __name__ == "__main__":
    main()
