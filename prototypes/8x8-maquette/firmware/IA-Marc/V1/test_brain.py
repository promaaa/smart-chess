import chess
from engine_brain import Engine
from engine_search import Searcher

# Initialisation
board = chess.Board()
brain = Engine()
searcher = Searcher(brain)

# Test : L'IA joue les BLANCS
print("--- Début de la recherche ---")
best_move = searcher.get_best_move(
    board, depth=3
)  # Profondeur 3 est bien pour le Pi au début

print(f"\nL'IA a choisi : {best_move}")
