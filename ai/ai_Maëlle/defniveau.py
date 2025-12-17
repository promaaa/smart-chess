import random
import ChessInteractif16
import chess

moteur = ChessInteractif16


#---------------------------  BLOQUAGE DÉFINITIF DU LIVRE POLYGLOT pour éviter les égalité----------------------
import chess.polyglot

class FakeBookFinal:
    """
    Faux livre d'ouverture Polyglot.
    simule un livre vide
    """
    def __enter__(self):      # Support de l'instruction "with"
        return self

    def __exit__(self, *args):
        pass

    def find_all(self, *args, **kwargs):
        return []             # Aucun coup disponible

    def __iter__(self):
        return iter([])       # Aucune ligne d'ouverture

    def close(self):
        pass

def truly_fake_book(*args, **kwargs):
    """
    Remplace ouverture par un livre vide 
    """
    return FakeBookFinal()

chess.polyglot.open_reader = truly_fake_book



# ------------------ éviter les débuts de partie identiques du à la phase initi identique -----------

orig_find = moteur.find_best_move

def find_best_varied(board, max_plies=None, time_limit=None, tt=None):
    """
    Part d'aléatoire uniquement dans les tout premiers coups

    """
    if board.fullmove_number <= 2:
        moves = list(board.legal_moves)
        if len(moves) > 5:
            random.shuffle(moves)
            return moves[0], 0, 0
    return orig_find(board, max_plies, time_limit, tt)

# Remplacement de la fonction de recherche originale
moteur.find_best_move = find_best_varied


# PÉNALITÉ ANTI-RÉPÉTITION

if hasattr(moteur, "repetition_penalty"):
    orig_rep = moteur.repetition_penalty

    def rep_patch(board):
        r = orig_rep(board)
        if r < 0:
            return r * 12
        return r

    moteur.repetition_penalty = rep_patch

# -----------------------------définir les niveaux-----------------------------------------

LEVELS = {
    1: {"noise": 250, "depth": 2, "time": 0.6},   # Niveau débutant
    2: {"noise": 120, "depth": 4, "time": 1.0},
    3: {"noise": 40,  "depth": 5, "time": 1.8},
    4: {"noise": 10,  "depth": 7, "time": 3.0},
    5: {"noise": 3,   "depth": 9, "time": 5.0},   # Niveau maximal
}

_current = 5

orig_eval = moteur.evaluate
def eval_patch(board):
    """
    Ajoute un bruit aléatoire à l'évaluation statique du moteur.
    Le bruit diminue lorsque le niveau augmente.
    """
    score = orig_eval(board)
    noise = LEVELS[_current]["noise"]
    if noise > 0:
        score += random.randint(-noise, noise)
    return score

moteur.evaluate = eval_patch

#  -----------------------FONCTION DE CHANGEMENT DE NIVEAU-------------------------------
#  Mise à jour paramètre
def set_level(L: int):
    """
    Définit le niveau de difficulté du moteur.
    Met à jour la profondeur maximale et le temps de recherche.
    """
    global _current
    _current = max(1, min(5, L))
    moteur.MAX_PLIES = LEVELS[_current]["depth"]
    moteur.TIME_LIMIT = LEVELS[_current]["time"]
