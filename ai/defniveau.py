import random
import ChessInteractif16
import chess

moteur = ChessInteractif16

# ===============================
#    DÉSACTIVER LE LIVRE D’OUVERTURE
# ===============================
###########  BLOQUAGE DÉFINITIF DU LIVRE POLYGLOT  ###########
import chess.polyglot

class FakeBookFinal:
    def __enter__(self):      # support "with .. as"
        return self
    def __exit__(self, *args):
        pass
    def find_all(self, *args, **kwargs):
        return []             # aucun coup
    def __iter__(self):
        return iter([])       # aucune ligne
    def close(self):
        pass

def truly_fake_book(*args, **kwargs):
    return FakeBookFinal()

# PATCH GLOBAL (FORCE ABSOLUE)
chess.polyglot.open_reader = truly_fake_book
###############################################################


# ===============================
#   OUVERTURES VARIÉES
# ===============================

orig_find = moteur.find_best_move

def find_best_varied(board, max_plies=None, time_limit=None, tt=None):
    if board.fullmove_number <= 2:
        moves = list(board.legal_moves)
        if len(moves) > 5:
            random.shuffle(moves)
            return moves[0], 0, 0
    return orig_find(board, max_plies, time_limit, tt)

moteur.find_best_move = find_best_varied

# ===============================
#   PÉNALITÉ ANTI-RÉPÉTITION
# ===============================

if hasattr(moteur, "repetition_penalty"):
    orig_rep = moteur.repetition_penalty

    def rep_patch(board):
        r = orig_rep(board)
        if r < 0:
            return r * 12
        return r

    moteur.repetition_penalty = rep_patch

# ===============================
#    DIFFÉRENCES DE NIVEAU
# ===============================

LEVELS = {
    1: {"noise": 250, "depth": 2, "time": 0.6},
    2: {"noise": 120, "depth": 4, "time": 1.0},
    3: {"noise": 40,  "depth": 6, "time": 1.8},
    4: {"noise": 10,  "depth": 8, "time": 3.0},
    5: {"noise": 3,   "depth": 9, "time": 5.0},
}

_current = 5

orig_eval = moteur.evaluate
def eval_patch(board):
    score = orig_eval(board)
    noise = LEVELS[_current]["noise"]
    if noise > 0:
        score += random.randint(-noise, noise)
    return score

moteur.evaluate = eval_patch

def set_level(L: int):
    global _current
    _current = max(1, min(5, L))
    moteur.MAX_PLIES = LEVELS[_current]["depth"]
    moteur.TIME_LIMIT = LEVELS[_current]["time"]
