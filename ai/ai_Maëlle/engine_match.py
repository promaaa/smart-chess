import chess.engine
import os
import time
from pathlib import Path

#----------------------------Initialisation des paramètres----------------------------

# Dossier principal du projet (logs, parties sauvegardées, etc.)
Doc_PATH = Path(r"C:\Users\maell\Documents\smart_chess_drive\chessClaire")

# Commande permettant de lancer ton moteur via l’interface UCI
MY_ENGINE_CMD = [
    "python",
    str(Path(r"C:\Users\maell\Documents\smart_chess_drive\chessClaire") / "uci_interface.py")
]

# Chemin vers l’exécutable Stockfish
STOCKFISH_PATH = Path(
    r"C:\Users\maell\Documents\smart_chess_drive\chessClaire\stockfish\stockfish-windows-x86-64-avx2.exe"
)

# Test d’ouverture du moteur Stockfish (sécurité)
chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))

# Nombre total de parties jouées dans le match
NB_GAMES = 10

# Paramètres de comparaison
sf_level = 2     # Niveau de Stockfish
my_motorv = 14   # Version de ton moteur (pour traçabilité)

# Codes ANSI pour l'affichage couleur dans le terminal
RESET = "\033[0m"
WHITE_PIECE = "\033[97m"
BLACK_PIECE = "\033[94m"


Doc_PATH.mkdir(exist_ok=True)
PARTIES_FILE = Doc_PATH / "parties.txt"


#----------------------------Affichage échéquier----------------------------

def afficher_echiquier_statique(board: chess.Board, recent_moves=None, numero=None):
    """
    Affiche l’échiquier ASCII dans le terminal avec les pièces colorées
    et les derniers coups joués.
    """
    if recent_moves is None:
        recent_moves = []

    # Nettoyage de l’écran
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

    # Affichage des derniers coups
    if recent_moves:
        print(f"{numero} - Derniers coups :")
        for mv in recent_moves[-2:]:
            print(mv)
    print("\n")


#----------------------------Enregistrement des partis----------------------------

def enregistrer_partie_texte(board: chess.Board, moves_list, result,
                              moteur_blanc, moteur_noir,
                              fichier=PARTIES_FILE, numero=None):
    """
    Sauvegarde l'intégralité d'une partie dans un fichier texte.
    Permet une analyse a posteriori (débogage, statistiques, rapport).
    """
    from datetime import datetime
    with open(fichier, "a", encoding="utf-8") as f:
        f.write("\n" + "*" * 60 + "\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Niveau : {sf_level}\n")
        f.write(f"Version : {my_motorv}\n")
        if numero:
            f.write(f"### Partie {numero} ###\n")
        f.write(f"Blancs : {moteur_blanc}\n")
        f.write(f"Noirs  : {moteur_noir}\n")
        f.write(f"Résultat : {result}\n\n")

        # Liste complète des coups joués
        for i, mv in enumerate(moves_list, 1):
            f.write(f"{i:3d}. {mv}\n")

        f.write("\n" + "*" * 60 + "\n\n")

        # Forçage de l’écriture sur disque
        f.flush()
        os.fsync(f.fileno())


#----------------------------Corps du match----------------------------

def play_game(engine_white, engine_black, nom_blanc, nom_noir, numero=None):
    """
    Lance une partie complète entre deux moteurs UCI.
    Gère l'affichage, le temps de réflexion et l'enregistrement.
    """
    board = chess.Board()
    engines = {True: engine_white, False: engine_black}
    move_number = 1
    recent_moves = []
    moves_for_log = []

    afficher_echiquier_statique(board, recent_moves, numero)
    time.sleep(1)

    while not board.is_game_over():
        engine = engines[board.turn]

        # Gestion du temps de réflexion
        if engine == engine_white:
            time_limit = 7.0   # Ton moteur
        else:
            time_limit = 3.0   # Stockfish

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

    # Sauvegarde de la partie
    enregistrer_partie_texte(
        board,
        moves_for_log,
        result,
        moteur_blanc=nom_blanc,
        moteur_noir=nom_noir,
        numero=numero
    )

    return result



#  ----------------------------MATCH AUTOMATIQUE CONTRE STOCKFISH----------------------------
def main():
    print("=== MATCH TEST ===")

    # Lancement des deux moteurs en mode UCI
    with chess.engine.SimpleEngine.popen_uci(MY_ENGINE_CMD) as my_engine, \
         chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH)) as sf_engine:

        # Configuration de Stockfish
        sf_engine.configure({
            "Skill Level": sf_level,
            "Move Overhead": 30,
            "Threads": 1,
            "Hash": 32
        })

        my_score = 0
        sf_score = 0

        for i in range(1, NB_GAMES + 1):
            print(f"\n--- Partie {i}/{NB_GAMES} ---")

            # Alternance des couleurs
            if i % 2 == 1:
                result = play_game(my_engine, sf_engine, "Mon moteur", "Stockfish", numero=i)
            else:
                result = play_game(sf_engine, my_engine, "Stockfish", "Mon moteur", numero=i)

            print(f"Résultat : {result}")

            # Attribution des points selon la couleur jouée
            if i % 2 == 1:
                if result == "1-0":
                    my_score += 1
                elif result == "0-1":
                    sf_score += 1
                else:
                    my_score += 0.5
                    sf_score += 0.5
            else:
                if result == "0-1":
                    my_score += 1
                elif result == "1-0":
                    sf_score += 1
                else:
                    my_score += 0.5
                    sf_score += 0.5

        # Résultats finaux
        print("\n=== SCORE FINAL ===")
        print(f"Ton moteur : {my_score}")
        print(f"Stockfish  : {sf_score}")
        print(f"Score global : {100 * my_score / NB_GAMES:.1f}%")

        print(f"\n Les parties ont été enregistrées dans :\n{PARTIES_FILE}")


if __name__ == "__main__":
    main()
