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

# ----- Centre -----
CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]

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

# ----- Configuration -----
TIME_LIMIT = 3.0
MAX_PLIES = 5
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
    if board.turn == chess.WHITE:
        h ^= 0xABCDEF1234567890
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

# ----- Évaluation -----
def evaluate(board: chess.Board) -> int:
    piece_values = {
        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
    }
    score = 0
    for piece_type in piece_values:
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

    # Structure pions
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        for sq in pawns:
            file = chess.square_file(sq)
            if sum(1 for p in pawns if chess.square_file(p) == file) > 1:
                score += -10 if color == chess.WHITE else 10
            # pions passés
            is_passed = True
            rank = chess.square_rank(sq)
            for f in [file-1, file, file+1]:
                for r in (range(rank+1,8) if color==chess.WHITE else range(0,rank)):
                    if 0<=f<=7:
                        target = chess.square(f,r)
                        if target in board.pieces(chess.PAWN, not color):
                            is_passed = False
            if is_passed:
                score += (30 if color==chess.WHITE else -30)
    # Tours sur colonnes ouvertes
    for color in [chess.WHITE, chess.BLACK]:
        rooks = board.pieces(chess.ROOK, color)
        for sq in rooks:
            file = chess.square_file(sq)
            pawns_in_file = [p for p in board.pieces(chess.PAWN, chess.WHITE if color==chess.BLACK else chess.BLACK) if chess.square_file(p)==file]
            if len(pawns_in_file)==0:
                score += (25 if color==chess.WHITE else -25)
    # Roi milieu/finale
    total_material = sum([piece_values[p.piece_type] for p in board.piece_map().values()])
    phase = min(1.0, total_material / 52.0)
    for sq in board.pieces(chess.KING, chess.WHITE):
        mid = KING_MIDGAME[sq]
        end = KING_ENDGAME[sq]
        score += int(mid*phase + end*(1-phase))
        if board.is_check():
            score -= 30
    for sq in board.pieces(chess.KING, chess.BLACK):
        mid = KING_MIDGAME[sq]
        end = KING_ENDGAME[sq]
        score -= int(mid*phase + end*(1-phase))
        if board.is_check():
            score += 30
    return score if board.turn==chess.WHITE else -score

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

# ----- Negamax α–β + Null Move -----
def negamax(board, depth, alpha, beta, tt, start_time, time_limit, allow_null=True, killer_moves=None):
    if time.time()-start_time>time_limit: raise TimeoutError()
    key = zobrist_hash(board)
    tt_move = None
    if USE_TT:
        entry = tt.get(key)
        if entry:
            d,v,flag,bm = entry
            tt_move = bm
            if d>=depth:
                if flag=="EXACT": return v, tt_move
                elif flag=="LOWERBOUND": alpha = max(alpha,v)
                elif flag=="UPPERBOUND": beta = min(beta,v)
                if alpha>=beta: return v, tt_move
    if depth==0 or board.is_game_over():
        return quiescence(board, alpha, beta, start_time, time_limit), None
    # Null move
    if allow_null and depth>=3 and not board.is_check():
        board.turn = not board.turn
        val,_ = negamax(board, depth-1-2, -beta, -beta+1, tt, start_time, time_limit, allow_null=False)
        board.turn = not board.turn
        if -val>=beta: return -val, None
    best_val = -9999999
    best_move_local = None
    moves = list(board.legal_moves)
    moves = order_moves(board, moves, tt_move=tt_move, killer_moves=killer_moves)
    for move in moves:
        board.push(move)
        try:
            score,_ = negamax(board, depth-1, -beta, -alpha, tt, start_time, time_limit, killer_moves=killer_moves)
            score = -score
        except TimeoutError:
            board.pop()
            raise
        board.pop()
        if score>best_val:
            best_val = score
            best_move_local = move
            if killer_moves is not None and not board.is_capture(move):
                killer_moves.append(move)
        alpha = max(alpha, score)
        if alpha>=beta: break
    # TT store
    if USE_TT:
        if best_val<=alpha: flag="UPPERBOUND"
        elif best_val>=beta: flag="LOWERBOUND"
        else: flag="EXACT"
        tt.store(key, depth, best_val, flag, best_move_local)
    return best_val, best_move_local

# ----- Iterative Deepening + Aspiration -----
# ----- Recherche adaptative avec sécurité du roi -----
def find_best_move(board, max_plies=MAX_PLIES, time_limit=TIME_LIMIT, tt=None):
    global TT_GLOBAL
    if tt is None:
        tt = TT_GLOBAL
    if tt is None:
        raise ValueError("Table de transposition non initialisée (tt=None)")

    start_time = time.time()
    best_move = None
    best_score = -10**9
    best_depth = 0

    fen_key = board.fen()
    # 1️⃣ Vérification du livre d’ouvertures
    if fen_key in OPENING_BOOK:
        opening_name, possible_moves = OPENING_BOOK[fen_key]
        # Choisir un coup légal dans le livre
        legal_moves = [chess.Move.from_uci(m) for m in possible_moves if chess.Move.from_uci(m) in board.legal_moves]
        if legal_moves:
            chosen_move = random.choice(legal_moves)
            if DEBUG_PRINT:
                print(f"[opening] {opening_name} -> coup joué : {chosen_move.uci()}")
            return chosen_move, evaluate(board), 0

    # 2️⃣ Priorité au roque si disponible
    for move in board.legal_moves:
        if board.is_castling(move):
            return move, evaluate(board), 0

    # 3️⃣ Priorité au développement des pièces mineures
    minor_squares_white = [chess.B1, chess.G1, chess.C1, chess.F1]
    minor_squares_black = [chess.B8, chess.G8, chess.C8, chess.F8]
    minor_moves = []
    for move in board.legal_moves:
        if board.turn == chess.WHITE and move.from_square in minor_squares_white:
            minor_moves.append(move)
        elif board.turn == chess.BLACK and move.from_square in minor_squares_black:
            minor_moves.append(move)
    if minor_moves:
        return random.choice(minor_moves), evaluate(board), 0

    # 4️⃣ Recherche adaptative classique (Negamax)
    num_pieces = sum(len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
                     for pt in range(1, 7))
    adaptive_max = max_plies
    if num_pieces >= 24:
        adaptive_max = 3
    elif num_pieces >= 16:
        adaptive_max = 4

    for depth in range(1, adaptive_max + 1):
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break
        if DEBUG_PRINT:
            print(f"[info] Recherche à profondeur {depth}...", file=sys.stderr)
        try:
            score, move = negamax(board, depth, -10000000, 10000000, tt, start_time, time_limit)
            if move and not board.gives_check(move):
                # Ne pas jouer un coup qui met notre roi en échec
                best_move, best_score, best_depth = move, score, depth
        except TimeoutError:
            break

    # Sécurité finale : si aucun coup sûr, choisir un coup légal minimal
    if best_move is None:
        for move in board.legal_moves:
            if not board.gives_check(move):
                best_move = move
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
        if move is None:
            print("Moteur ne trouve pas de coup, partie terminée.")
            break
        board.push(move)
        print(f"Moteur joue : {move.uci()} (depth {depth}, score {score})")
        print_board(board)

    print("\nPartie terminée :", board.result())

if __name__ == "__main__":
    main()
