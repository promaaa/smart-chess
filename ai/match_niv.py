import chess
import time
import os
from pathlib import Path
import defniveau
import ChessInteractif16

# ============================================================
# ======== CONFIGURATION =====================================
# ============================================================

Doc_PATH = Path(r"C:\Users\maell\Documents\smart_chess_drive\chessClaire")
Doc_PATH.mkdir(exist_ok=True)
PARTIES_FILE = Doc_PATH / "parties_Level.txt"

RESET = "\033[0m"
WHITE_PIECE = "\033[97m"
BLACK_PIECE = "\033[94m"


# ============================================================
# ============  AFFICHAGE ÉCHIQUIER  =========================
# ============================================================

def afficher_echiquier_statique(board, recent_moves=None, numero=None):
    if recent_moves is None:
        recent_moves = []

    os.system('cls' if os.name == 'nt' else 'clear')

    print("    a  b  c  d  e  f  g  h")
    print("  ┌────────────────────────┐")
    for rank in range(8, 0, -1):
        row = f"{rank} │"
        for file in range(8):
            sq = chess.square(file, rank - 1)
            piece = board.piece_at(sq)
            if piece:
                sym = piece.symbol()
                color_code = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE
                row += f" {color_code}{sym.upper() if piece.color else sym.lower()}{RESET} "
            else:
                row += " . "
        row += f"│ {rank}"
        print(row)
    print("  └────────────────────────┘")
    print("    a  b  c  d  e  f  g  h\n")

    if recent_moves:
        print(f"{numero} - Derniers coups :")
        for mv in recent_moves[-2:]:
            print(mv)
    print("\n")


# ============================================================
# ============ ENREGISTREMENT PARTIE =========================
# ============================================================

def enregistrer_partie(board, moves_list, result, niveau_blanc, niveau_noir, numero=None):
    from datetime import datetime
    with open(PARTIES_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "*" * 60 + "\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if numero:
            f.write(f"### Partie {numero} ###\n")
        f.write(f"Blancs : Moteur Niveau {niveau_blanc}\n")
        f.write(f"Noirs  : Moteur Niveau {niveau_noir}\n")
        f.write(f"Résultat : {result}\n\n")

        for mv in moves_list:
            f.write(mv + "\n")

        f.write("\n" + "*" * 60 + "\n\n")
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# ============  MATCH ENTRE NIVEAUX  =========================
# ============================================================

def play_game(lvl_white, lvl_black, numero):
    board = chess.Board()
    recent = []
    log_moves = []
    ply = 1  # numéro de coup blanc

    afficher_echiquier_statique(board, recent, numero)
    time.sleep(0.8)

    while not board.is_game_over():
        lvl = lvl_white if board.turn == chess.WHITE else lvl_black
        defniveau.set_level(lvl)

        move, _, _ = ChessInteractif16.find_best_move(board, ChessInteractif16.MAX_PLIES, ChessInteractif16.TIME_LIMIT)
        if move is None:
            break

        # Format PGN-like simple
        if board.turn == chess.WHITE:
            text = f"{ply}. {move.uci()}"
        else:
            text = f"... {move.uci()}"
            ply += 1

        board.push(move)
        recent.append(text)
        log_moves.append(text)

        afficher_echiquier_statique(board, recent, numero)
        time.sleep(0.3)

    result = board.outcome().result() if board.outcome() else "1/2-1/2"
    enregistrer_partie(board, log_moves, result, lvl_white, lvl_black, numero)
    return result


if __name__ == "__main__":
    print("Ce fichier ne doit pas être exécuté seul. Utilisez tournoi_niveaux.py")
