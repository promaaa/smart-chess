#!/usr/bin/env python3
"""
Interactive chess engine (v16).

Successor to v14 with refinements and additional heuristics.
Includes TT, Zobrist hashing, and opening book integration.
"""
import chess
import chess.polyglot
import time
import random
import sys
from eval_fastB import eval_core 
import os 

DEBUG_PRINT = False  

# ---- Table de hachage Zobrist ----
ZOBRIST_TABLE = [
    [random.getrandbits(64) for _ in range(12)] for _ in range(64)
]

PIECE_INDEX = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

# Table de transposition
TRANSPOSITION_TABLE = {}

# ----- Bonus développement rapide -----
DEVELOPMENT_BONUS = {
    chess.KNIGHT: 10,
    chess.BISHOP: 10,
    chess.ROOK: 5,
    chess.QUEEN: 0,
    chess.KING: 0,
    chess.PAWN: 0
}

# Historique développement
piece_development_history = {}
HISTORY_TABLE = {}

#--------------------------------------------------PST---------------------------------------------------------------------

# Droits Zobrist additionnels
ZOBRIST_CASTLING = {
    'K': random.getrandbits(64),
    'Q': random.getrandbits(64),
    'k': random.getrandbits(64),
    'q': random.getrandbits(64)
}
ZOBRIST_EP_FILE = [random.getrandbits(64) for _ in range(8)]

#----------------------------------------------------Paramètres--------------------------------------------------------------

TIME_LIMIT = 5.0
MAX_PLIES = 9
USE_TT = True
TT_GLOBAL = None

LAST_POSITIONS_LIMIT = 6
position_history = []


class SearchTimeout(Exception):
    """Timeout interne du moteur."""
    pass


#----------------------------------------Zobrist hashing--------------------------------------------------------------

def zobrist_hash(board: chess.Board) -> int:
    h = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_idx = PIECE_INDEX[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            h ^= ZOBRIST_TABLE[square][piece_idx]

    # trait
    if board.turn == chess.WHITE:
        h ^= 0xABCDEF1234567890

    # castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        h ^= ZOBRIST_CASTLING['K']
    if board.has_queenside_castling_rights(chess.WHITE):
        h ^= ZOBRIST_CASTLING['Q']
    if board.has_kingside_castling_rights(chess.BLACK):
        h ^= ZOBRIST_CASTLING['k']
    if board.has_queenside_castling_rights(chess.BLACK):
        h ^= ZOBRIST_CASTLING['q']

    ep = board.ep_square
    if ep is not None:
        file = chess.square_file(ep)
        if 0 <= file < 8:
            h ^= ZOBRIST_EP_FILE[file]

    return h

#----------------------------------------TT--------------------------------------------------------------

class TranspositionTable:
    def __init__(self, max_size=500000):
        self.table = {}
        self.max_size = max_size

    def get(self, key):
        return self.table.get(key, None)

    def store(self, key, depth, value, flag, best_move):
        if len(self.table) > self.max_size:
            for _ in range(self.max_size // 2):
                self.table.pop(next(iter(self.table)))
        self.table[key] = (depth, value, flag, best_move)

#----------------------------------------MVV-LVA--------------------------------------------------------------

def score_move(board: chess.Board, move: chess.Move):
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            victim_val = {chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900}.get(captured.piece_type,0)
            mover = board.piece_at(move.from_square)
            mover_val = 0 if mover is None else {chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900}.get(mover.piece_type,0)
            return 100000 + (victim_val*10 - mover_val)
    return 0

#----------------------------------------Pénalités répétition--------------------------------------------------------------

def repetition_penalty(board: chess.Board) -> int:
    if len(position_history) < 6:
        return 0
    repetitions = sum(1 for past in position_history if past.board_fen() == board.board_fen())
    if repetitions >= 3:
        return -150
    return 0

def endgame_repetition_penalty(board: chess.Board) -> int:
    if len(position_history) < 6:
        return 0
    last = position_history[-1].king(chess.WHITE if board.turn == chess.WHITE else chess.BLACK)
    current = board.king(board.turn)
    if last is not None and current is not None and last == current:
        return -100
    return 0

#----------------------------------------Fin de partie : refinements--------------------------------------------------------------

def endgame_refinement(board: chess.Board, base_score: int) -> int:
    score = base_score

    # Roi central : calcul entier
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq:
            file, rank = chess.square_file(king_sq), chess.square_rank(king_sq)
            file_dist = abs(file - 3)
            rank_dist = abs(rank - 3)
            center = (3 - file_dist) + (3 - rank_dist)

            if color == chess.WHITE:
                score += 30 * center
            else:
                score -= 30 * center

    # Pions avancés / passés
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        for sq in pawns:
            rank = chess.square_rank(sq)
            advance = rank if color == chess.WHITE else 7 - rank
            score += 10 * advance if color == chess.WHITE else -10 * advance

    score += repetition_penalty(board) + endgame_repetition_penalty(board)
    score += stagnation_penalty(position_history)

    position_history.append(board.copy())
    if len(position_history) > LAST_POSITIONS_LIMIT:
        position_history.pop(0)

    score += random.randint(-5, 5)

    return score

#----------------------------------------Quiescence--------------------------------------------------------------

def quiescence(board, alpha, beta, start_time, time_limit):
    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    for move in sorted(board.legal_moves, key=lambda m: score_move(board,m), reverse=True):
        if board.is_capture(move):
            board.push(move)
            if time.time() - start_time > time_limit:
                board.pop()
                raise SearchTimeout()
            score = -quiescence(board, -beta, -alpha, start_time, time_limit)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    return alpha

#----------------------------------------Move ordering--------------------------------------------------------------

def order_moves(board, moves, tt_move=None, killer_moves=None):
    scored_moves = []
    for move in moves:
        score = 0
        if tt_move and move == tt_move:
            score += 1000000
        if board.is_capture(move):
            score += score_move(board, move)
        if killer_moves and move in killer_moves:
            score += 5000
        if board.gives_check(move):
            score += 50
        if move in HISTORY_TABLE:
            score += HISTORY_TABLE[move]

        scored_moves.append((score, move))

    scored_moves.sort(reverse=True, key=lambda x: x[0])
    return [m for s, m in scored_moves]

#--------------------------------------------------Evaluation position--------------------------------------------------------------
#   1) matériel + tableaux PST
#   2) contrôle du centre
#   3) structure de pions (isolés, doublés, passés)
#   4) développement (ouverture)
#   5) activité des tours
#   6) sécurité du roi
#   7) mobilité
#   8) pénalités diverses
#   9) fin de partie (refinement)

def evaluate(board: chess.Board) -> int:
    # Conversion python-chess -> bitboards (uint64)
    wp = int(board.pieces(chess.PAWN,   chess.WHITE))
    wn = int(board.pieces(chess.KNIGHT, chess.WHITE))
    wb = int(board.pieces(chess.BISHOP, chess.WHITE))
    wr = int(board.pieces(chess.ROOK,   chess.WHITE))
    wq = int(board.pieces(chess.QUEEN,  chess.WHITE))
    wk = int(board.pieces(chess.KING,   chess.WHITE))

    bp = int(board.pieces(chess.PAWN,   chess.BLACK))
    bn = int(board.pieces(chess.KNIGHT, chess.BLACK))
    bb = int(board.pieces(chess.BISHOP, chess.BLACK))
    br = int(board.pieces(chess.ROOK,   chess.BLACK))
    bq = int(board.pieces(chess.QUEEN,  chess.BLACK))
    bk = int(board.pieces(chess.KING,   chess.BLACK))

    stm_white = (board.turn == chess.WHITE)
    halfmove_clock = board.halfmove_clock

    score = eval_core(wp, wn, wb, wr, wq, wk,
                      bp, bn, bb, br, bq, bk,
                      stm_white, halfmove_clock)

    # Tu peux garder ici ta logique de nulle par répétition
    if board.is_repetition(3) or board.can_claim_draw():
        return 0

    return score

#--------------------------------------------------Eviter répétition--------------------------------------------------------------
# Pénalise les positions qui stagnent (6 dernières positions identiques)

def stagnation_penalty(history: list[chess.Board]) -> int:
    if len(history) < 6:
        return 0
    last = history[-1].board_fen()
    if all(h.board_fen() == last for h in history[-6:]):
        return -80
    return 0

#--------------------------------------------------nombre de cases que les pièces peuvent contrôler--------------------------------------------------------------
# Plus une pièce a de mobilité, plus elle est active

def piece_mobility(board: chess.Board, color: chess.Color) -> int:
    mobility = 0
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board.pieces(piece_type, color):
            mobility += len(board.attacks(sq))
    return mobility

#--------------------------------------------------Evaluation sécurité du roi (défense)--------------------------------------------------------------
# (utilisée éventuellement ailleurs, mais pas dans evaluate)

def king_safety_score(board: chess.Board, color: chess.Color) -> int:
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    score = 0
    rank = chess.square_rank(king_sq)
    file = chess.square_file(king_sq)

    # Roi au centre : dangereux
    if rank in [3, 4] and file in [3, 4]:
        score -= 50

    # Bouclier de pions manquant devant le roi
    direction = 1 if color == chess.WHITE else -1
    front_pawns = [
        chess.square(file + df, rank + direction)
        for df in [-1, 0, 1]
        if 0 <= file + df < 8 and 0 <= rank + direction < 8
    ]
    for sq in front_pawns:
        if sq not in board.pieces(chess.PAWN, color):
            score -= 15

    # Bonus si roqué
    if color == chess.WHITE and rank == 0 and file in [6, 2]:
        score += 40
    elif color == chess.BLACK and rank == 7 and file in [6, 2]:
        score += 40

    # Roi attaqué directement
    if board.is_attacked_by(not color, king_sq):
        score -= 40

    return score

#-----------------------------------------------Pions passés (fonction optionnelle)--------------------------------------------------------------
# Pas utilisée directement dans evaluate, mais conservée si tu veux t’en servir ailleurs

def passed_pawn_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    pawns = board.pieces(chess.PAWN, color)
    direction = 1 if color == chess.WHITE else -1

    for sq in pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)

        is_passed = True
        for df in [-1, 0, 1]:
            f = file + df
            if 0 <= f < 8:
                for r in range(rank + direction, 8 if color == chess.WHITE else -1, direction):
                    if chess.square(f, r) in board.pieces(chess.PAWN, not color):
                        is_passed = False
                        break
            if not is_passed:
                break

        if is_passed:
            score += 10 + 5 * (rank if color == chess.WHITE else (7 - rank))

    return score

#--------------------------------------------------Recherche des mates en 1 ou 2 coups--------------------------------------------------------------

def find_mate_in_one_or_two(board: chess.Board):
    """
    Recherche mat en 1 ou 2 coups.
    Retourne le coup à jouer s'il existe, sinon None.
    """
    for move in board.legal_moves:
        board.push(move)
        # Mat en 1
        if board.is_checkmate():
            board.pop()
            return move

        # Mat en 2
        mate_in_two_found = True
        for reply in board.legal_moves:
            board.push(reply)
            mate_in_two = False
            for next_move in board.legal_moves:
                board.push(next_move)
                if board.is_checkmate():
                    mate_in_two = True
                board.pop()
            board.pop()
            if not mate_in_two:
                mate_in_two_found = False
                break

        board.pop()
        if mate_in_two_found:
            return move

    return None

#--------------------------------------------------Négamax--------------------------------------------------------------

def negamax(board, depth, alpha, beta, tt, start_time, time_limit,
            allow_null=True, killer_moves=None):

    if time.time() - start_time > time_limit:
        raise SearchTimeout()

    key = zobrist_hash(board)
    tt_move = None

    # --- Table de transposition ---
    if USE_TT:
        entry = tt.get(key)
        if entry:
            d, v, flag, bm = entry
            tt_move = bm
            if d >= depth:
                if flag == "EXACT":
                    return v, tt_move
                elif flag == "LOWERBOUND":
                    alpha = max(alpha, v)
                elif flag == "UPPERBOUND":
                    beta = min(beta, v)
                if alpha >= beta:
                    return v, tt_move

    # --- Feuille / quiescence ---
    if depth <= 0 or board.is_game_over():
        return quiescence(board, alpha, beta, start_time, time_limit), None

    alpha_orig = alpha
    best_val = -10**9
    best_move_local = None

    # --- Null move pruning ---
    if allow_null and depth >= 3 and not board.is_check():
        board.push(chess.Move.null())
        try:
            val, _ = negamax(board, depth - 3, -beta, -beta + 1, tt,
                             start_time, time_limit, allow_null=False)
        finally:
            board.pop()

        if -val >= beta:
            return -val, None

    # --- Move ordering ---
    moves = list(board.legal_moves)
    if killer_moves is None:
        killer_moves = {}

    moves = order_moves(board, moves, tt_move=tt_move,
                        killer_moves=killer_moves.get(depth, []))

    first_move = True
    move_index = 0

    # --- Boucle principale ---
    for move in moves:
        board.push(move)

        # --- LMR (Late Move Reductions) ---
        search_depth = depth - 1
        if (depth >= 3 and move_index >= 3 and
            not board.is_capture(move) and not board.gives_check(move)):
            search_depth = depth - 2

        # --- PVS ---
        if first_move:
            try:
                score, _ = negamax(board, search_depth,
                                   -beta, -alpha, tt,
                                   start_time, time_limit,
                                   allow_null=True,
                                   killer_moves=killer_moves)
                score = -score
            except SearchTimeout:
                board.pop()
                raise

            first_move = False

        else:
            # fenêtre étroite
            try:
                score, _ = negamax(board, search_depth,
                                   -alpha - 1, -alpha, tt,
                                   start_time, time_limit,
                                   allow_null=True,
                                   killer_moves=killer_moves)
                score = -score
            except SearchTimeout:
                board.pop()
                raise

            # si fail-high → vraie recherche
            if alpha < score < beta:
                try:
                    score, _ = negamax(board, search_depth,
                                       -beta, -alpha, tt,
                                       start_time, time_limit,
                                       allow_null=True,
                                       killer_moves=killer_moves)
                    score = -score
                except SearchTimeout:
                    board.pop()
                    raise

        board.pop()

        # --- Meilleur score ---
        if score > best_val:
            best_val = score
            best_move_local = move

        # --- Killer moves ---
        if not board.is_capture(move) and depth > 1:
            km = killer_moves.get(depth, [])
            if move not in km:
                km.append(move)
                if len(km) > 2:
                    km.pop(0)
                killer_moves[depth] = km

        # --- History Heuristic ---
        alpha = max(alpha, score)
        if alpha >= beta:
            if not board.is_capture(move):
                HISTORY_TABLE[move] = HISTORY_TABLE.get(move, 0) + depth * depth
            break

        move_index += 1

    # --- Stockage TT ---
    if USE_TT:
        if best_val <= alpha_orig:
            flag = "UPPERBOUND"
        elif best_val >= beta:
            flag = "LOWERBOUND"
        else:
            flag = "EXACT"
        tt.store(key, depth, best_val, flag, best_move_local)

    return best_val, best_move_local

#--------------------------------------------------Recherche du meilleur coup (iterative deepening)--------------------------------------------------------------

def find_best_move(board, max_plies=MAX_PLIES, time_limit=TIME_LIMIT, tt=None):
    """
    Recherche adaptative utilisant negamax + iterative deepening.
    """
    global TT_GLOBAL
    if tt is None:
        tt = TT_GLOBAL
    if tt is None:
        TT_GLOBAL = TranspositionTable()
        tt = TT_GLOBAL

    start_time = time.time()
    best_move = None
    best_score = -10**9
    best_depth = 0

    # --- Livre d'ouvertures ---
    # --- Livre d'ouverture Polyglot ---
    script_dir = os.path.dirname(os.path.abspath(__file__))  # dossier du script
    book_file = os.path.join(script_dir, "Perfect2023.bin")       # fichier dans le même dossier

    # Vérifie que le fichier existe
    if os.path.exists(book_file): 

        with chess.polyglot.open_reader(book_file) as reader:

            book_moves = list(reader.find_all(board))

            if book_moves:
                # Choisir aléatoirement mais pondéré par le poids dans le livre
                chosen_entry = random.choices(book_moves, weights=[entry.weight for entry in book_moves])[0]
                best_move = chosen_entry.move

                if DEBUG_PRINT:
                    with open("trace_moteur.log", "a") as f:
                        f.write(f"Book best_move {best_move}\n")

                return best_move, 500, 0
    # --- Déterminer la phase et adapter profondeur/temps ---
    num_pieces = len(board.piece_map())
    if num_pieces > 24:
        phase = "opening"
    elif num_pieces > 12:
        phase = "middlegame"
    else:
        phase = "endgame"

    if phase == "opening":
        adaptive_max = max_plies
        adaptive_time = time_limit
    elif phase == "middlegame":
        adaptive_max = max_plies + 1
        adaptive_time = time_limit * 1.2
    else:
        adaptive_max = max_plies + 2
        adaptive_time = time_limit * 1.6

    capture_moves = sum(1 for m in board.legal_moves
                        if board.is_capture(m) or board.gives_check(m))
    if capture_moves > 5:
        adaptive_max = min(adaptive_max + 1, 11)
        adaptive_time *= 1.3

    if DEBUG_PRINT:
        with open("trace_moteur.log", "a") as f:
            f.write(f"[INFO] Phase={phase} pieces={num_pieces} adaptive_max={adaptive_max} adaptive_time={adaptive_time:.2f}\n")


    # Vérification tactique mat en 1 ou 2 coups
    #tactical_move = find_mate_in_one_or_two(board)
    tactical_move = find_mate_in_one_or_two(board)
    if tactical_move:
        if DEBUG_PRINT:
            with open("trace_moteur.log", "a") as f:
                f.write(f"[TACTIC] Mat détecté avec {tactical_move.uci()}\n")
        return tactical_move, 10**7, 0

    # Iterative deepening
    last_score = 0  # Pour Aspiration Windows

    for depth in range(1, adaptive_max + 1):

        elapsed = time.time() - start_time
        if elapsed >= adaptive_time:
            if DEBUG_PRINT:
                with open("trace_moteur.log", "a") as f:
                    f.write("[time] temps epuisé avant profondeur\n")
            break

        remaining_time = adaptive_time - elapsed
        if remaining_time <= 0:
            break

        # --- Définition aspiration window ---
        if depth > 1:
            window = 50
            alpha = last_score - window
            beta  = last_score + window
        else:
            alpha = -10**7
            beta  = 10**7

        try:
            search_board = board.copy()
            val, move = negamax(search_board, depth, alpha, beta,
                                tt, start_time, remaining_time)

            # --- Fail-low / fail-high : relancer recherche large ---
            if val <= alpha or val >= beta:

                search_board = board.copy()
                val, move = negamax(search_board, depth,
                                    -10**7, 10**7,
                                    tt, start_time, remaining_time)

            # --- Mise à jour du meilleur résultat ---
            if move:
                best_move = move
                best_score = val
                best_depth = depth
                last_score = val  # ⬅️ Très important

            # Stop si valeur extrême (mat très proche)
            if abs(val) > 9000000:
                break

        except (TimeoutError, SearchTimeout):
            if DEBUG_PRINT:
                with open("trace_moteur.log", "a") as f:
                    f.write(f"[time] Timeout pendant profondeur {depth}\n")
            break

        except Exception as e:
            if DEBUG_PRINT:
                with open("trace_moteur.log", "a") as f:
                    f.write(f"[err] exception pendant negamax depth={depth}: {e}\n")
            break

    # fallback
    if best_move is None:
        if DEBUG_PRINT:
            with open("trace_moteur.log", "a") as f:     
                f.write(f"Random choice\n")
        for m in board.legal_moves:
            if not board.gives_check(m):
                best_move = m
                break
        if best_move is None:
            best_move = random.choice(list(board.legal_moves))
    else:
        if DEBUG_PRINT:
            with open("trace_moteur.log", "a") as f:
                f.write(f" depth={best_depth} move={best_move} score={best_score} \n")

    return best_move, best_score, best_depth


# ----- Affichage clair -----

def print_board(board):
    RESET = "\033[0m"
    WHITE = "\033[97m"
    BLACK = "\033[90m"

    cols = "A  B  C  D  E  F  G  H"

    print()
    print(f"   {cols}")
    for rank in range(8, 0, -1):
        row = f"{rank}  "
        for file in range(8):
            square = chess.square(file, rank - 1)
            piece = board.piece_at(square)

            if piece:
                symbol = piece.symbol().upper()
                if piece.color == chess.WHITE:
                    row += f"{WHITE}{symbol}{RESET}  "
                else:
                    row += f"{BLACK}{symbol}{RESET}  "
            else:
                row += ".  "
        print(f"{row}{rank}")
    print(f"   {cols}\n")

# ----- Programme interactif -----

def main():
    global TT_GLOBAL
    TT_GLOBAL = TranspositionTable()

    board = chess.Board()

    user_move = ""
    print(" ")
    if DEBUG_PRINT:
        print(" ")
    print("Début de partie ! Vous jouez les Blancs.")
    print("Entrez 'a' pour annuler votre dernier coup")
    print("       'f' pour interrompre la partie.")
    print("----- Bonne partie! -----")

    print_board(board)

    while not board.is_game_over():

        # Tour joueur
        if board.is_check():
            print("⚠️ Votre roi est en échec !")

        while True:
            try:
                user_move = input("Votre coup (UCI, ex: e2e4) ? ").strip().lower()

                if user_move == "f":
                    break

                if user_move == "a":
                    if len(board.move_stack) >= 2:
                        board.pop()
                        board.pop()
                        print("Dernier tour annulé.")
                        print_board(board)
                        continue
                    elif len(board.move_stack) == 1:
                        board.pop()
                        print("Dernier coup annulé (seul votre coup).")
                        print_board(board)
                        continue
                    else:
                        print("Aucun coup à annuler.")
                        continue

                move = chess.Move.from_uci(user_move)
                if move in board.legal_moves:
                    board.push(move)
                    break
                else:
                    print("Coup illégal, réessayez.")
            except Exception:
                print("Format invalide, réessayez (ex: e2e4).")

        if user_move == "f":
            break

        print_board(board)
        if board.is_game_over():
            break

        # Tour moteur
        print("Moteur réfléchit...")
        move, score, depth = find_best_move(board, MAX_PLIES, TIME_LIMIT)

        if move is not None:
            board.push(move)
            print(f"Moteur joue : {move.uci()} (depth {depth}, score {score})")
        else:
            print("⚠️ Aucun coup trouvé par le moteur (position finale ou erreur).")

        print_board(board)

    print("\nPartie terminée :", board.result())

if __name__ == "__main__":
    main() 