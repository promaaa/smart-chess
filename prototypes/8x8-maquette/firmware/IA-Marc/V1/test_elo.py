import chess
from engine_brain import Engine
from engine_search import Searcher

# Initialisation
board = chess.Board()
brain = Engine()
searcher = Searcher(brain)

print("=== TEST DES NIVEAUX D'ELO ===\n")

# 1. Test niveau DÉBUTANT (ELO 600)
# Il doit répondre très vite et jouer superficiellement
searcher.set_elo(600)
move = searcher.get_best_move(board)
print(f"--> Coup Débutant : {move}\n")

# 2. Test niveau CLUB (ELO 1500)
# Il doit réfléchir un peu plus (Profondeur 3-4)
searcher.set_elo(1500)
move = searcher.get_best_move(board)
print(f"--> Coup Club : {move}\n")

# 3. Test niveau EXPERT (ELO 2000+)
# Il doit utiliser tout le temps imparti (ex: 2 sec)
searcher.set_elo(2200)
move = searcher.get_best_move(board, time_limit=2.0)
print(f"--> Coup Expert : {move}\n")
