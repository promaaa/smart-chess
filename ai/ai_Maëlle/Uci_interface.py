# ======= fichier : uci_interface.py =======
#!/usr/bin/env python3

import sys
import chess
import chess.engine
import time
import random
from ChessInteractif14 import find_best_move, TranspositionTable, MAX_PLIES, TIME_LIMIT



#---------------------------  Fonction UCI---------------------

TT_GLOBAL = TranspositionTable()


def uci_loop():
    """
    Boucle principale de communication UCI.
    Cette fonction permet à un GUI (Arena, CuteChess, etc.)
    de dialoguer avec le moteur via le protocole UCI.
    """

    # Identification du moteur 
    print("id name MyPythonEngine", flush=True)
    print("id author Maelle", flush=True)
    print("uciok", flush=True)

    # Réinitialisation de la table de transposition au lancement
    global TT_GLOBAL
    TT_GLOBAL = TranspositionTable()

    # Initialisation de l'échiquier
    board = chess.Board()

    while True:
        line = sys.stdin.readline()
        if not line:
            continue
        line = line.strip()

   
        if line == "isready":
            print("readyok", flush=True)


        # Initialise la position de départ standard
        elif line.startswith("position startpos"):
            board = chess.Board()
            moves_idx = line.find("moves")
            if moves_idx != -1:
                moves = line[moves_idx + 6:].split()
                for move_uci in moves:
                    board.push(chess.Move.from_uci(move_uci))

        # Initialise une position à partir d'une FEN arbitraire
        elif line.startswith("position fen "):
            fen = line[13:].split(" moves")[0]
            board = chess.Board(fen)
            if "moves" in line:
                moves = line.split("moves")[1].strip().split()
                for move_uci in moves:
                    board.push(chess.Move.from_uci(move_uci))


        # Lancement du calcul du meilleur coup
        elif line.startswith("go"):
            try:
                # Recherche du meilleur coup via le moteur
                move, score, depth = find_best_move(
                    board,
                    MAX_PLIES,
                    TIME_LIMIT,
                    tt=TT_GLOBAL
                )

                # Envoi du coup au format UCI
                if move:
                    print(f"bestmove {move.uci()}", flush=True)
                else:
                    # Coup nul ou erreur de calcul
                    print("bestmove 0000", flush=True)

            except Exception as e:
                # Gestion d'erreur pour éviter le crash du moteur
                print(f"[error] {e}", file=sys.stderr, flush=True)
                print("bestmove 0000", flush=True)

        elif line == "quit":
            break


# Point d'entrée du programme
if __name__ == "__main__":
    uci_loop()
