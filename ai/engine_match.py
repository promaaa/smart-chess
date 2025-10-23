import chess
import chess.engine
import subprocess
from pathlib import Path

# --- CONFIGURATION ---

# Ton moteur
MY_ENGINE_CMD = ["python", "uci_interface.py"]

# L’autre moteur (Stockfish par exemple)
STOCKFISH_PATH = Path(r"C:\Program Files (x86)\stockfish\stockfish-windows-x86-64-avx2.exe")
chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))

# Nombre de parties à jouer
NB_GAMES = 5

# Temps maximum par coup (en secondes)
TIME_PER_MOVE = 5.0


def play_game(engine_white, engine_black):
    board = chess.Board()
    engines = {True: engine_white, False: engine_black}

    move_number = 1

    while not board.is_game_over():
        engine = engines[board.turn]
        result = engine.play(board, chess.engine.Limit(time=TIME_PER_MOVE))
        if result.move is None:
            break
        board.push(result.move)

        # Affichage du coup
        turn = "Blancs" if board.turn == chess.BLACK else "Noirs"
        print(f"{move_number}. {turn} joue {result.move.uci()}")
        move_number += 1

    outcome = board.outcome()
    return outcome.result() if outcome else "1/2-1/2"


def main():
    print("=== MATCH TEST ===")

    # Lancer les moteurs
    with chess.engine.SimpleEngine.popen_uci(MY_ENGINE_CMD) as my_engine, \
         chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH)) as sf_engine:

        my_score = 0
        sf_score = 0

        for i in range(1, NB_GAMES + 1):
            print(f"\n--- Partie {i}/{NB_GAMES} ---")
            if i % 2 == 1:
                result = play_game(my_engine, sf_engine)
            else:
                result = play_game(sf_engine, my_engine)

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

if __name__ == "__main__":
    main()
