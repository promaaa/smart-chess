#!/usr/bin/env python3
"""
IA-Marc V2 - UCI Chess Engine Wrapper
======================================

Wrapper UCI (Universal Chess Interface) pour le moteur d'échecs IA-Marc V2.
Ce processus communique via stdin/stdout et ne contient AUCUN code matériel.

Compatible PyPy3 pour performances maximales (2-5x speedup).
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

import chess
from engine_brain import EvaluationEngine
from engine_config import EngineConfig
from engine_search import SearchEngine # Utilise l'alias pour la clarté

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

        # --- Find best available opening book ---
        polyglot_paths = [
            "../book/Cerebellum.bin", 
            "book/Cerebellum.bin", 
            "../../IA-Marc/book/Cerebellum.bin"
        ]
        book_path_found = None
        for path in polyglot_paths:
            if Path(path).exists():
                book_path_found = path
                break
        
        if book_path_found:
            self.config.opening_book_path = book_path_found
        # If not found, it will use the default "data/openings.json" from EngineConfig

        self.brain = EvaluationEngine()
        self.searcher = SearchEngine(self.brain, self.config)

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
            print(f"info string DEBUG: {message}", file=sys.stderr)

    # ========================================================================
    # COMMANDES UCI
    # ========================================================================

    def cmd_uci(self):
        """Commande: uci - Identification du moteur."""
        print(f"id name {self.name}")
        print(f"id author {self.author}")
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
        """
        if len(args) < 2: return
        try:
            name_idx = args.index("name")
            option_name = args[name_idx + 1]
            value = None
            if "value" in args:
                value_idx = args.index("value")
                value = " ".join(args[value_idx + 1 :])

            if option_name == "Level": self.config.set_level(value.upper())
            elif option_name == "Personality": self.config.set_personality(value.upper())
            elif option_name == "Hash": self.config.tt_size_mb = int(value)
            elif option_name == "Threads": self.config.threads = int(value)
            
            # Re-init searcher if needed
            self.searcher = SearchEngine(self.brain, self.config)
            self.log_debug(f"Option {option_name} set to {value}")

        except (ValueError, IndexError) as e:
            self.log_debug(f"Error parsing setoption: {e}")

    def cmd_ucinewgame(self):
        """Commande: ucinewgame - Prépare une nouvelle partie."""
        self.board = chess.Board()
        self.searcher.clear()
        self.log_debug("New game initialized")

    def cmd_position(self, args: List[str]):
        """
        Commande: position [fen <fenstring> | startpos] moves <move1> ...
        """
        if not args: return
        try:
            if args[0] == "startpos":
                self.board = chess.Board()
                moves_start = 1
            elif args[0] == "fen":
                moves_idx = args.index("moves") if "moves" in args else len(args)
                fen = " ".join(args[1:moves_idx])
                self.board = chess.Board(fen)
                moves_start = moves_idx
            else:
                return

            if moves_start < len(args) and args[moves_start] == "moves":
                for move_str in args[moves_start + 1 :]:
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move in self.board.legal_moves: self.board.push(move)
                        else:
                            self.log_debug(f"Illegal move: {move_str}"); break
                    except ValueError:
                        self.log_debug(f"Invalid move format: {move_str}"); break
            self.log_debug(f"Position set: {self.board.fen()}")
        except Exception as e:
            self.log_debug(f"Error parsing position: {e}")

    def cmd_go(self, args: List[str]):
        """
        Commande: go [movetime <x>] [depth <x>] ...
        """
        if self.searching: return
        self.searching = True
        
        movetime, depth, infinite = None, None, False
        i = 0
        while i < len(args):
            if args[i] == "movetime" and i + 1 < len(args): movetime = int(args[i+1])/1000.0; i+=2
            elif args[i] == "depth" and i + 1 < len(args): depth = int(args[i+1]); i+=2
            elif args[i] == "infinite": infinite = True; i+=1
            else: i += 1

        level = self.config.difficulty_level
        time_limit = movetime or level.time_limit
        depth_limit = depth or level.depth_limit

        try:
            best_move = self.searcher.search(self.board, time_limit=time_limit, depth_limit=depth_limit)
            if best_move:
                print(f"bestmove {best_move.uci()}")
            else:
                print("bestmove 0000")
        except Exception as e:
            self.log_debug(f"Error during search: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
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
        """
        self.log_debug(f"{self.name} started")
        while True:
            try:
                line = sys.stdin.readline().strip()
                if not line: continue
                parts = line.split()
                if not parts: continue
                command, args = parts[0].lower(), parts[1:]

                if command == "uci": self.cmd_uci()
                elif command == "debug": self.cmd_debug(args)
                elif command == "isready": self.cmd_isready()
                elif command == "setoption": self.cmd_setoption(args)
                elif command == "ucinewgame": self.cmd_ucinewgame()
                elif command == "position": self.cmd_position(args)
                elif command == "go": self.cmd_go(args)
                elif command == "stop": self.cmd_stop()
                elif command == "quit": self.cmd_quit()
                else: self.log_debug(f"Unknown command: {command}")
            except (EOFError, KeyboardInterrupt):
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
    sys.stdin.reconfigure(line_buffering=True)
    sys.stdout.reconfigure(line_buffering=True)
    engine = UCIEngine()
    engine.run()


if __name__ == "__main__":
    main()
