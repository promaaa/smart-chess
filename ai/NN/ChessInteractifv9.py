#!/usr/bin/env python3
import chess
import chess.polyglot
import time
import random
import sys

DEBUG_PRINT = False

# ---- Table de hachage Zobrist ----
ZOBRIST_TABLE = [
    [random.getrandbits(64) for _ in range(12)] for _ in range(64)
]

PIECE_INDEX = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

# Table de transposition (mémoire des positions)
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

# Historique redéploiement
piece_development_history = {}

# --- Livre d’ouvertures (simplifié mais réaliste, avec centre et développement rapide) ---
OPENING_BOOK = {
    # --- Ouverture espagnole (Ruy Lopez) ---
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": ("Ruy Lopez", ["e7e5"]),
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2": ("Ruy Lopez", ["g1f3"]),
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": ("Ruy Lopez", ["b8c6", "f8c5"]),
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3": ("Ruy Lopez", ["a7a6", "g8f6"]),

    # --- Ouverture italienne ---
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": ("Italienne", ["e7e5", "c7c5"]),
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": ("Italienne", ["b8c6", "f8c5"]),
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3": ("Italienne", ["a7a6", "g8f6"]),
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 3 4": ("Italienne", ["f8c5", "b8d7"]),

    # --- Défense sicilienne ---
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": ("Sicilienne", ["c7c5"]),
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2": ("Sicilienne", ["g1f3", "d2d4"]),
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": ("Sicilienne", ["d7d6", "g8f6"]),
    "rnbqkb1r/pp1ppppp/2p2n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": ("Sicilienne", ["d2d4", "Nc3"]),

    # --- Défense française ---
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": ("Française", ["e7e6"]),
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2": ("Française", ["d2d4", "Nc3"]),
    "rnbqkbnr/pppp1ppp/4p3/4p3/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 1 3": ("Française", ["d7d5", "Ng8f6"]),

    # --- Défense Caro-Kann ---
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": ("Caro-Kann", ["c7c6"]),
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ("Caro-Kann", ["d2d4", "Nc3"]),
    "rnbqkbnr/pp1ppppp/2p5/8/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 1 2": ("Caro-Kann", ["d7d5", "Ng8f6"]),

    # --- Défense Scandinave ---
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": ("Scandinave", ["d7d5"]),
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2": ("Scandinave", ["e4d5", "Nc3"]),

    # --- Ouvertures centrées sur le développement rapide et contrôle du centre ---
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": ("Ouverture centrale", ["e2e4", "d2d4", "g1f3", "c2c4"]),
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": ("Réponse classique", ["e7e5", "d7d5", "g8f6", "c7c5"]),
}

# ----- Tables Piece-Square Tables (PST) -----
PST = {
    chess.PAWN: [
         0,  5,  5,-10,-10,  5,  5,  0,
        10, 10, 10, 10, 10, 10, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10,  5, 10, 10,  5, 10,-10,
        -10,  0,  0, 10, 10,  0,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    chess.ROOK: [
         0,  0,  0,  5,  5,  0,  0,  0,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         5, 10, 10, 10, 10, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0,  0,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-30,-30,-40,-40,-30,-30,-30,
        -30,-30,-30,-30,-30,-30,-30,-30,
        -20,-20,-20,-20,-20,-20,-20,-20,
        -10,-10,-10,-10,-10,-10,-10,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ]
}

KING_MIDGAME = PST[chess.KING]
KING_ENDGAME = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

# Zobrist additionnel : droits de roque et en-passant par fichier
ZOBRIST_CASTLING = {
    'K': random.getrandbits(64),  # blanc roque côté roi
    'Q': random.getrandbits(64),  # blanc roque côté dame
    'k': random.getrandbits(64),  # noir roque côté roi
    'q': random.getrandbits(64)   # noir roque côté dame
}
ZOBRIST_EP_FILE = [random.getrandbits(64) for _ in range(8)]  # a..h file

# ----- Configuration -----
TIME_LIMIT = 5.0
MAX_PLIES = 8
USE_TT = True
TT_GLOBAL = None

# ----- Fonction de hachage Zobrist -----
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

    # en-passant : on ne stocke que le fichier (comme c'est classique)
    ep = board.ep_square
    if ep is not None:
        file = chess.square_file(ep)
        if 0 <= file < 8:
            h ^= ZOBRIST_EP_FILE[file]

    return h


# ----- Table de transposition -----
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

# ----- Score coup pour tri -----
def score_move(board: chess.Board, move: chess.Move):
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            victim_val = {chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900}.get(captured.piece_type,0)
            mover = board.piece_at(move.from_square)
            mover_val = 0 if mover is None else {chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900}.get(mover.piece_type,0)
            return 100000 + (victim_val*10 - mover_val)
    return 0


def endgame_refinement(board: chess.Board, base_score: int) -> int:
    """Ajuste le score de base selon les critères spécifiques à la fin de partie."""
    score = base_score
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)

    # Activation du roi en finale
    if white_king and black_king:
        wk_file, wk_rank = chess.square_file(white_king), chess.square_rank(white_king)
        bk_file, bk_rank = chess.square_file(black_king), chess.square_rank(black_king)

        # Plus le roi est centré, mieux c'est
        white_center_bonus = 14 - (abs(wk_file - 3.5) + abs(wk_rank - 3.5)) * 6
        black_center_bonus = 14 - (abs(bk_file - 3.5) + abs(bk_rank - 3.5)) * 6
        score += int(white_center_bonus - black_center_bonus)

    # Pions passés plus valorisés s’ils sont proches de la promotion
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            if color == chess.WHITE:
                advance = rank
                direction = range(rank+1, 8)
                blockers = [r for r in direction if chess.square(file, r) in board.pieces(chess.PAWN, chess.BLACK)]
            else:
                advance = 7 - rank
                direction = range(rank-1, -1, -1)
                blockers = [r for r in direction if chess.square(file, r) in board.pieces(chess.PAWN, chess.WHITE)]

            if not blockers:  # pion passé
                bonus = 15 + advance * 10  # plus il est avancé, plus le bonus est grand
                score += bonus if color == chess.WHITE else -bonus

    return score

# ----- Quiescence améliorée -----
def quiescence(board, alpha, beta, start_time, time_limit):
    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    for move in sorted(board.legal_moves, key=lambda m: score_move(board,m), reverse=True):
        if board.is_capture(move) or board.gives_check(move):
            board.push(move)
            if time.time() - start_time > time_limit:
                board.pop()
                raise TimeoutError()
            score = -quiescence(board, -beta, -alpha, start_time, time_limit)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    return alpha

# ----- Tri coups -----
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
        scored_moves.append((score, move))
    scored_moves.sort(reverse=True, key=lambda x:x[0])
    return [m for s,m in scored_moves]

def negamax(board, depth, alpha, beta, tt, start_time, time_limit, allow_null=True, killer_moves=None):
    if time.time() - start_time > time_limit:
        raise TimeoutError()

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
    if depth == 0 or board.is_game_over():
        return quiescence(board, alpha, beta, start_time, time_limit), None

    alpha_orig = alpha
    best_val = -10**9
    best_move_local = None

    # --- Null move pruning (sécurisé) ---
    if allow_null and depth >= 3 and not board.is_check():
        board.push(chess.Move.null())
        val, _ = negamax(board, depth - 1 - 2, -beta, -beta + 1, tt, start_time, time_limit, allow_null=False)
        board.pop()
        if -val >= beta:
            return -val, None

    # --- Move ordering ---
    moves = list(board.legal_moves)
    moves = order_moves(board, moves, tt_move=tt_move, killer_moves=killer_moves)

    if killer_moves is None:
        killer_moves = {}

    # --- Boucle principale ---
    for move in moves:
        board.push(move)
        try:
            score, _ = negamax(board, depth - 1, -beta, -alpha, tt, start_time, time_limit,
                               allow_null=allow_null, killer_moves=killer_moves)
            score = -score
        except TimeoutError:
            board.pop()
            raise
        board.pop()

        if score > best_val:
            best_val = score
            best_move_local = move

        # Killer moves : deux par profondeur max
        if not board.is_capture(move) and depth > 1:
            km = killer_moves.get(depth, [])
            if move not in km:
                km.append(move)
                if len(km) > 2:
                    km.pop(0)
                killer_moves[depth] = km

        alpha = max(alpha, score)
        if alpha >= beta:
            break

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


# --- Evaluate ---
def evaluate(board: chess.Board) -> int:
    """Évaluation hybride : solide, mais agressive pour prendre l'initiative contre moteur faible."""
    piece_values = {
        chess.PAWN: 100, chess.KNIGHT: 325, chess.BISHOP: 335,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
    }

    CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
    EXTENDED_CENTER = [
        chess.C3, chess.D3, chess.E3, chess.F3,
        chess.C4, chess.F4, chess.C5, chess.F5,
        chess.C6, chess.D6, chess.E6, chess.F6
    ]

    score = 0
    total_material = sum(
        piece_values[p.piece_type] for p in board.piece_map().values() if p.piece_type != chess.KING
    )
    phase = min(1.0, total_material / 5200)
    endgame = phase < 0.3

    # --- 1️⃣ Matériel + PST ---
    for piece_type in piece_values:
        for sq in board.pieces(piece_type, chess.WHITE):
            pst = KING_MIDGAME if piece_type == chess.KING else PST.get(piece_type, [0]*64)
            pst_end = KING_ENDGAME if piece_type == chess.KING else pst
            val = piece_values[piece_type] + int(pst[sq] * phase + pst_end[sq] * (1 - phase))
            score += val
        for sq in board.pieces(piece_type, chess.BLACK):
            pst = KING_MIDGAME if piece_type == chess.KING else PST.get(piece_type, [0]*64)
            pst_end = KING_ENDGAME if piece_type == chess.KING else pst
            val = piece_values[piece_type] + int(pst[chess.square_mirror(sq)] * phase +
                                                 pst_end[chess.square_mirror(sq)] * (1 - phase))
            score -= val

    # --- 2️⃣ Contrôle du centre ---
    for square in CENTER_SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            score += 35
        if board.is_attacked_by(chess.BLACK, square):
            score -= 35
    for square in EXTENDED_CENTER:
        if board.is_attacked_by(chess.WHITE, square):
            score += 12
        if board.is_attacked_by(chess.BLACK, square):
            score -= 12

    # --- 3️⃣ Coordination des pions/pièces ---
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        for sq in pawns:
            # Pions centraux soutenus
            if sq in [chess.D4, chess.E4, chess.D5, chess.E5]:
                supporters = sum(1 for piece_sq in board.piece_map()
                                 if board.color_at(piece_sq) == color and sq in board.attacks(piece_sq))
                score += supporters * (25 if color == chess.WHITE else -25)

            # Pions doublés
            file = chess.square_file(sq)
            if sum(1 for p in pawns if chess.square_file(p) == file) > 1:
                score += -15 if color == chess.WHITE else 15

            # Pions isolés
            isolated = all(not any(chess.square_file(p) == f for p in pawns)
                           for f in [file - 1, file + 1] if 0 <= f <= 7)
            if isolated:
                score += -10 if color == chess.WHITE else 10

            # Pions passés
            rank = chess.square_rank(sq)
            is_passed = True
            for f in [file - 1, file, file + 1]:
                if not 0 <= f <= 7:
                    continue
                for r in (range(rank + 1, 8) if color == chess.WHITE else range(0, rank)):
                    target = chess.square(f, r)
                    if target in board.pieces(chess.PAWN, not color):
                        is_passed = False
            if is_passed:
                advance = rank if color == chess.WHITE else 7 - rank
                base_bonus = 30 + advance * (25 if endgame else 12)
                score += base_bonus if color == chess.WHITE else -base_bonus

    # --- 4️⃣ Principes d'ouverture + développement agressif ---
    if board.fullmove_number <= 12:
        for color in [chess.WHITE, chess.BLACK]:
            # Pénalité dame sortie trop tôt
            queen_sq = next(iter(board.pieces(chess.QUEEN, color)), None)
            if queen_sq:
                q_rank = chess.square_rank(queen_sq)
                if (color == chess.WHITE and q_rank > 1) or (color == chess.BLACK and q_rank < 6):
                    score += -50 if color == chess.WHITE else 50

            # Bonus développement rapide cavaliers/fous
            for sq in board.pieces(chess.KNIGHT, color):
                r = chess.square_rank(sq)
                score += 20 if (color == chess.WHITE and r > 1) or (color == chess.BLACK and r < 6) else 0
            for sq in board.pieces(chess.BISHOP, color):
                r = chess.square_rank(sq)
                score += 12 if (color == chess.WHITE and r > 1) or (color == chess.BLACK and r < 6) else 0

            # Pénalité si roi non-roqué
            king_sq = board.king(color)
            if king_sq and board.fullmove_number >= 10:
                k_file = chess.square_file(king_sq)
                if k_file in [3,4]:
                    score += -30 if color == chess.WHITE else 30

    # --- 5️⃣ Tours actives / colonnes ouvertes ---
    for color in [chess.WHITE, chess.BLACK]:
        for sq in board.pieces(chess.ROOK, color):
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            # Bonus 7e rangée
            if (color == chess.WHITE and rank == 6) or (color == chess.BLACK and rank == 1):
                score += 20 if color == chess.WHITE else -20
            # Bonus colonnes ouvertes
            if not any(chess.square_file(p) == file for p in board.pieces(chess.PAWN, color)):
                score += 15 if color == chess.WHITE else -15

    # --- 6️⃣ Roi actif en finale ---
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk and bk:
        wk_file, wk_rank = chess.square_file(wk), chess.square_rank(wk)
        bk_file, bk_rank = chess.square_file(bk), chess.square_rank(bk)
        wk_center = 7 - (abs(wk_file - 3.5) + abs(wk_rank - 3.5))
        bk_center = 7 - (abs(bk_file - 3.5) + abs(bk_rank - 3.5))
        score += wk_center * (40 if endgame else 10)
        score -= bk_center * (40 if endgame else 10)

    # --- 7️⃣ Roi en échec ---
    if board.is_check():
        score += -40 if board.turn == chess.WHITE else 40

    # --- 8️⃣ Bonus progression pions finale ---
    if endgame:
        for color in [chess.WHITE, chess.BLACK]:
            for sq in board.pieces(chess.PAWN, color):
                rank = chess.square_rank(sq)
                bonus = (rank if color == chess.WHITE else 7 - rank) * 4
                score += bonus if color == chess.WHITE else -bonus

    # --- 9️⃣ Ajustement final ---
    score = endgame_refinement(board, score)

    return score if board.turn == chess.WHITE else -score


def find_best_move(board, max_plies=MAX_PLIES, time_limit=TIME_LIMIT, tt=None):
    """
    Recherche adaptative utilisant negamax + iterative deepening.
    Remplace l'appel erroné à `minimax`.
    """
    global TT_GLOBAL
    if tt is None:
        tt = TT_GLOBAL
    if tt is None:
        # initialise si nécessaire
        TT_GLOBAL = TranspositionTable()
        tt = TT_GLOBAL

    start_time = time.time()
    best_move = None
    best_score = -10**9
    best_depth = 0

    # --- Livre d'ouvertures ---
    fen_key = board.fen()
    if fen_key in OPENING_BOOK:
        opening_name, possible_moves = OPENING_BOOK[fen_key]
        legal_moves = [chess.Move.from_uci(m) for m in possible_moves if chess.Move.from_uci(m) in board.legal_moves]
        if legal_moves:
            chosen = random.choice(legal_moves)
            if DEBUG_PRINT:
                print(f"[opening] {opening_name} -> {chosen.uci()}", file=sys.stderr)
            return chosen, evaluate(board), 0

    # --- Déterminer la phase et adapter profondeur/temps ---
    num_pieces = len(board.piece_map())
    if num_pieces > 24:
        phase = "opening"
    elif num_pieces > 12:
        phase = "middlegame"
    else:
        phase = "endgame"

    # ⚙️ Nouveau réglage équilibré
    if phase == "opening":
        adaptive_max = max_plies
        adaptive_time = time_limit
    elif phase == "middlegame":
        adaptive_max = max_plies + 1
        adaptive_time = time_limit * 1.2
    else:  # endgame
        adaptive_max = max_plies + 2
        adaptive_time = time_limit * 1.6  # plus de temps pour finales critiques

    # Si position très tactique (captures ou échecs multiples)
    capture_moves = sum(1 for m in board.legal_moves if board.is_capture(m) or board.gives_check(m))
    if capture_moves > 5:
        adaptive_max = min(adaptive_max + 1, 11)
        adaptive_time *= 1.3


    if DEBUG_PRINT:
        print(f"[INFO] Phase={phase} pieces={num_pieces} adaptive_max={adaptive_max} adaptive_time={adaptive_time:.2f}s", file=sys.stderr)

    # Iterative deepening
    for depth in range(1, adaptive_max + 1):
        elapsed = time.time() - start_time
        if elapsed >= adaptive_time:
            if DEBUG_PRINT:
                print("[time] temps epuisé avant profondeur", depth, file=sys.stderr)
            break

        try:
            # appel à negamax avec les bornes larges
            val, move = negamax(board, depth, -10**7, 10**7, tt, start_time, adaptive_time)
            if move:
                best_move = move
                best_score = val
                best_depth = depth
                if DEBUG_PRINT:
                    print(f"[ID] depth={depth} move={move.uci()} score={val}", file=sys.stderr)
            # si valeur extrême, on peut sortir tôt
            if abs(val) > 9000000:
                break
        except TimeoutError:
            if DEBUG_PRINT:
                print(f"[time] Timeout pendant profondeur {depth}", file=sys.stderr)
            break
        except Exception as e:
            # sécurité : ne pas laisser tomber la boucle pour une autre erreur
            if DEBUG_PRINT:
                print(f"[err] exception pendant negamax depth={depth}: {e}", file=sys.stderr)
            break

    # fallback safety: si aucun coup déterminé, choisir coup légal aléatoire/sûr
    if best_move is None:
        # préférer coup qui ne prend pas risque immédiat (ne met pas le roi en échec)
        for m in board.legal_moves:
            if not board.gives_check(m):
                best_move = m
                break
        if best_move is None:
            best_move = random.choice(list(board.legal_moves))

    return best_move, best_score, best_depth

# ----- Affichage clair -----
def print_board(board):
    """
    Affiche le plateau avec alignement parfait et colonnes en majuscules.
    Blancs = lettres blanches
    Noirs  = lettres grises
    """
    # Séquences ANSI
    RESET = "\033[0m"
    WHITE = "\033[97m"   # blanc brillant
    BLACK = "\033[90m"   # gris clair
    # Si tu veux désactiver les couleurs :
    # RESET = WHITE = BLACK = ""

    cols = "A  B  C  D  E  F  G  H"

    print()
    print(f"   {cols}")
    for rank in range(8, 0, -1):
        row = f"{rank}  "
        for file in range(8):
            square = chess.square(file, rank - 1)
            piece = board.piece_at(square)

            if piece:
                symbol = piece.symbol().upper()  # tout en majuscule pour garder la taille uniforme
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
    # Ouverture classique
#    for move_uci in ["e2e4","e7e5","g1f3","b8c6"]:
#        board.push_uci(move_uci)

    user_move = ""
    print(" ")
    if DEBUG_PRINT: print(" ")
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
                # Commande d'annulation
                if user_move == "a":
                    if len(board.move_stack) >= 2:
                        board.pop()  # Annule le coup du moteur
                        board.pop()  # Annule le coup du joueur
                        print("Dernier tour annulé.")
                        print_board(board)
                        continue
                    elif len(board.move_stack) == 1:
                        board.pop()  # Annule seulement le coup du joueur
                        print("Dernier coup annulé (seul votre coup).")
                        print_board(board)
                        continue
                    else:
                        print("Aucun coup à annuler.")
                        continue

                # Vérification du coup normal
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
