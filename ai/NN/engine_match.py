import chess
import chess.engine
import os
import time
from pathlib import Path

# --- CONFIGURATION ---

MY_ENGINE_CMD = ["python", "uci_interface.py"]

STOCKFISH_PATH = Path(r"C:\Program Files (x86)\stockfish\stockfish-windows-x86-64-avx2.exe")
chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))

NB_GAMES = 5

RESET = "\033[0m"
WHITE_PIECE = "\033[97m"
BLACK_PIECE = "\033[94m"

# ------------------------------------------------------------
# 📂 Emplacement du fichier dans "Mes Documents"
# ------------------------------------------------------------
# Fonctionne sur Windows, Linux et macOS
DOCUMENTS_DIR = Path.home() / "Documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)
PARTIES_FILE = DOCUMENTS_DIR / "parties.txt"


# ==========================================================
# ============  AFFICHAGE ÉCHIQUIER ========================
# ==========================================================
def afficher_echiquier_statique(board: chess.Board, recent_moves=None, numero=None):
    if recent_moves is None:
        recent_moves = []

    os.system('cls' if os.name == 'nt' else 'clear')

    print("    a  b  c  d  e  f  g  h")
    print("  ┌────────────────────────┐")
    for rank in range(8, 0, -1):
        row = f"{rank} │"
        for file in range(8):
            square = chess.square(file, rank - 1)
            piece = board.piece_at(square)
            if piece:
                symbol = piece.symbol()
                color_code = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE
                row += f" {color_code}{symbol.upper() if piece.color else symbol.lower()}{RESET} "
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


# ==========================================================
# ============  ENREGISTREMENT DES PARTIES =================
# ==========================================================
def enregistrer_partie_texte(board: chess.Board, moves_list, result, moteur_blanc, moteur_noir, fichier=PARTIES_FILE, numero=None):
    """
    Sauvegarde la partie complète dans le fichier Mes Documents\parties.txt
    """
    from datetime import datetime
    with open(fichier, "a", encoding="utf-8") as f:
        f.write("\n" + "*" * 60 + "\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if numero:
            f.write(f"### Partie {numero} ###\n")
        f.write(f"Blancs : {moteur_blanc}\n")
        f.write(f"Noirs  : {moteur_noir}\n")
        f.write(f"Résultat : {result}\n\n")

        # Liste des coups
        for i, mv in enumerate(moves_list, 1):
            f.write(f"{i:3d}. {mv}\n")

        f.write("\n" + "*" * 60 + "\n\n")

        # Force l'écriture immédiate
        f.flush()           # Vide le tampon Python
        os.fsync(f.fileno())  # Force l’écriture sur disque (Windows/Linux/macOS)

# ==========================================================
# ============  MATCH ENTRE MOTEURS =========================
# ==========================================================
def play_game(engine_white, engine_black, nom_blanc, nom_noir, numero=None):
    board = chess.Board()
    engines = {True: engine_white, False: engine_black}
    move_number = 1
    recent_moves = []
    moves_for_log = []

    afficher_echiquier_statique(board, recent_moves, numero)
    time.sleep(1)

    while not board.is_game_over():
        engine = engines[board.turn]

        # Temps par moteur
        if engine == engine_white:
            # Si c'est mon moteur, il gère son temps (max : 10s)
            time_limit = 10.0
        else:
            # Si c'est Stockfish : max 3 secondes
            time_limit = 3.0

        result = engine.play(board, chess.engine.Limit(time=time_limit))

        if result.move is None:
            break

        board.push(result.move)
        turn = "Blancs" if board.turn == chess.BLACK else "Noirs"
        move_text = f"{turn} joue {result.move.uci()}"
        recent_moves.append(f"{move_number}. {move_text}")
        moves_for_log.append(move_text)

        afficher_echiquier_statique(board, recent_moves, numero)
        move_number += 1
        time.sleep(0.5)

    outcome = board.outcome()
    result = outcome.result() if outcome else "1/2-1/2"

    # Enregistrer la partie complète en texte
    enregistrer_partie_texte(
        board,
        moves_for_log,
        result,
        moteur_blanc=nom_blanc,
        moteur_noir=nom_noir,
        numero=numero
    )

    return result


# ==========================================================
# ============  MAIN : TOURNOI AUTOMATIQUE ==================
# ==========================================================
def main():
    print("=== MATCH TEST ===")

    with chess.engine.SimpleEngine.popen_uci(MY_ENGINE_CMD) as my_engine, \
         chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH)) as sf_engine:

        sf_engine.configure({
            "Skill Level": 2,
            "Move Overhead": 30,
            "Threads": 1,
            "Hash": 32
        })

        my_score = 0
        sf_score = 0

        for i in range(1, NB_GAMES + 1):
            print(f"\n--- Partie {i}/{NB_GAMES} ---")

            if i % 2 == 1:
                result = play_game(my_engine, sf_engine, "Mon moteur", "Stockfish", numero=i)
            else:
                result = play_game(sf_engine, my_engine, "Stockfish", "Mon moteur", numero=i)

            print(f"Résultat : {result}")

            if i % 2 == 1:  # ton moteur joue Blanc
                if result == "1-0":
                    my_score += 1
                elif result == "0-1":
                    sf_score += 1
                else:
                    my_score += 0.5
                    sf_score += 0.5
            else:  # ton moteur joue Noir
                if result == "0-1":
                    my_score += 1
                elif result == "1-0":
                    sf_score += 1
                else:
                    my_score += 0.5
                    sf_score += 0.5

        print("\n=== SCORE FINAL ===")
        print(f"Ton moteur : {my_score}")
        print(f"Stockfish  : {sf_score}")
        print(f"Score global : {100 * my_score / NB_GAMES:.1f}%")

        print(f"\n📄 Les parties ont été enregistrées dans :\n{PARTIES_FILE}")

if __name__ == "__main__":
    main()
