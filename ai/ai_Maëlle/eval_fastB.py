"""
Lightweight handcrafted evaluator used by interactive engines.

Provides piece-square tables (PST), simple material values,
and a fast `eval_core(...)` function operating on bitboards.

All scores are in centipawns; positive favors White.
"""
# ---------- Tables PST ----------

PAWN_PST = [
     0,  5,  5,-10,-10,  5,  5,  0,
    10, 10, 10, 10, 10, 10, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10,  5, 10, 10,  5, 10,-10,
    -10,  0,  0, 10, 10,  0,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_PST = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0,  0,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_MID = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-30,-30,-40,-40,-30,-30,-30,
    -30,-30,-30,-30,-30,-30,-30,-30,
    -20,-20,-20,-20,-20,-20,-20,-20,
    -10,-10,-10,-10,-10,-10,-10,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

KING_END = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

FILE_MASKS = [0] * 8
for f in range(8):
    m = 0
    for r in range(8):
        m |= 1 << (r * 8 + f)
    FILE_MASKS[f] = m


def _popcount(bb):
    c = 0
    while bb:
        bb &= bb - 1
        c += 1
    return c


def eval_core(wp, wn, wb, wr, wq, wk,
              bp, bn, bb, br, bq, bk,
              stm_white, halfmove_clock):
    """Core evaluation on bitboards.

    Parameters are 64-bit bitboards for each piece set, the side to move
    (`stm_white` as bool/int), and the halfmove clock. Returns an integer
    centipawn score (positive for White).
    """

    PAWN  = 100
    KNIGHT = 325
    BISHOP = 335
    ROOK  = 500
    QUEEN = 900
    KING  = 20000

    # ---------- Phase ----------
    material_sum = 0
    material_sum += _popcount(wp | bp) * PAWN
    material_sum += _popcount(wn | bn) * KNIGHT
    material_sum += _popcount(wb | bb) * BISHOP
    material_sum += _popcount(wr | br) * ROOK
    material_sum += _popcount(wq | bq) * QUEEN

    phase = material_sum * 1024 // 5200
    if phase > 1024:
        phase = 1024
    inv_phase = 1024 - phase
    endgame = phase < 300

    score = 0

    # ---------- Matériel + PST Blancs ----------
    # Pions
    bb_tmp = wp
    while bb_tmp:
        lsb = bb_tmp & -bb_tmp
        sq = lsb.bit_length() - 1
        mid = PAWN_PST[sq]
        pst = (mid * phase + mid * inv_phase) >> 10
        score += PAWN + pst
        bb_tmp ^= lsb

    # Cavaliers
    bb_tmp = wn
    while bb_tmp:
        lsb = bb_tmp & -bb_tmp
        sq = lsb.bit_length() - 1
        mid = KNIGHT_PST[sq]
        pst = (mid * phase + mid * inv_phase) >> 10
        score += KNIGHT + pst
        bb_tmp ^= lsb

    # Fous
    bb_tmp = wb
    while bb_tmp:
        lsb = bb_tmp & -bb_tmp
        sq = lsb.bit_length() - 1
        mid = BISHOP_PST[sq]
        pst = (mid * phase + mid * inv_phase) >> 10
        score += BISHOP + pst
        bb_tmp ^= lsb

    # Tours
    bb_tmp = wr
    while bb_tmp:
        lsb = bb_tmp & -bb_tmp
        sq = lsb.bit_length() - 1
        mid = ROOK_PST[sq]
        pst = (mid * phase + mid * inv_phase) >> 10
        score += ROOK + pst
        bb_tmp ^= lsb

    # Dames
    bb_tmp = wq
    while bb_tmp:
        lsb = bb_tmp & -bb_tmp
        sq = lsb.bit_length() - 1
        mid = QUEEN_PST[sq]
        pst = (mid * phase + mid * inv_phase) >> 10
        score += QUEEN + pst
        bb_tmp ^= lsb

    # Roi blanc
    if wk:
        lsb = wk & -wk
        sq = lsb.bit_length() - 1
        mid = KING_MID[sq]
        end = KING_END[sq]
        pst = (mid * phase + end * inv_phase) >> 10
        score += KING + pst

    # ---------- Matériel + PST Noirs ----------
    # on miroir les cases (flip vertical) : sq ^ 56
    # Pions noirs
    bb_tmp = bp
    while bb_tmp:
        lsb = bb_tmp & -bb_tmp
        sq = lsb.bit_length() - 1
        msq = sq ^ 56
        mid = PAWN_PST[msq]
        pst = (mid * phase + mid * inv_phase) >> 10
        score -= PAWN + pst
        bb_tmp ^= lsb

    # Cavaliers noirs
    bb_tmp = bn
    while bb_tmp:
        lsb = bb_tmp & -bb_tmp
        sq = lsb.bit_length() - 1
        msq = sq ^ 56
        mid = KNIGHT_PST[msq]
        pst = (mid * phase + mid * inv_phase) >> 10
        score -= KNIGHT + pst
        bb_tmp ^= lsb

    # Fous noirs
    bb_tmp = bb
    while bb_tmp:
        lsb = bb_tmp & -bb_tmp
        sq = lsb.bit_length() - 1
        msq = sq ^ 56
        mid = BISHOP_PST[msq]
        pst = (mid * phase + mid * inv_phase) >> 10
        score -= BISHOP + pst
        bb_tmp ^= lsb

    # Tours noires
    bb_tmp = br
    while bb_tmp:
        lsb = bb_tmp & -bb_tmp
        sq = lsb.bit_length() - 1
        msq = sq ^ 56
        mid = ROOK_PST[msq]
        pst = (mid * phase + mid * inv_phase) >> 10
        score -= ROOK + pst
        bb_tmp ^= lsb

    # Dames noires
    bb_tmp = bq
    while bb_tmp:
        lsb = bb_tmp & -bb_tmp
        sq = lsb.bit_length() - 1
        msq = sq ^ 56
        mid = QUEEN_PST[msq]
        pst = (mid * phase + mid * inv_phase) >> 10
        score -= QUEEN + pst
        bb_tmp ^= lsb

    # Roi noir
    if bk:
        lsb = bk & -bk
        sq = lsb.bit_length() - 1
        msq = sq ^ 56
        mid = KING_MID[msq]
        end = KING_END[msq]
        pst = (mid * phase + end * inv_phase) >> 10
        score -= KING + pst

    # ---------- Pions doublés + colonnes ouvertes pour les tours ----------
    # (simple mais rentable en Elo)

    # pions doublés / isolés & activité des tours
    for color in (1, 0):  # 1 = blanc, 0 = noir
        pawns = wp if color == 1 else bp
        rooks = wr if color == 1 else br
        sign = 1 if color == 1 else -1

        for f in range(8):
            file_mask = FILE_MASKS[f]
            pawns_on_file = pawns & file_mask
            n = _popcount(pawns_on_file)

            if n > 1:
                # pions doublés
                score -= 15 * sign

            # tours sur colonnes ouvertes / semi-ouvertes
            rooks_on_file = rooks & file_mask
            if rooks_on_file:
                own_pawns = pawns_on_file
                opp_pawns = (bp if color == 1 else wp) & file_mask

                if own_pawns == 0 and opp_pawns != 0:
                    score += 40 * sign    # semi-ouverte
                elif own_pawns == 0 and opp_pawns == 0:
                    score += 60 * sign    # ouverte

    # ---------- Pions passés (simple mais efficace) ----------
    for color in (1, 0):
        pawns = wp if color == 1 else bp
        opp_pawns = bp if color == 1 else wp
        sign = 1 if color == 1 else -1

        bb_tmp = pawns
        while bb_tmp:
            lsb = bb_tmp & -bb_tmp
            sq = lsb.bit_length() - 1
            file = sq & 7
            rank = sq >> 3

            is_passed = True
            for df in (-1, 0, 1):
                f2 = file + df
                if f2 < 0 or f2 > 7:
                    continue
                if color == 1:  # blanc
                    r2 = rank + 1
                    while r2 < 8:
                        sq2 = (r2 << 3) + f2
                        if (opp_pawns >> sq2) & 1:
                            is_passed = False
                            break
                        r2 += 1
                else:  # noir
                    r2 = rank - 1
                    while r2 >= 0:
                        sq2 = (r2 << 3) + f2
                        if (opp_pawns >> sq2) & 1:
                            is_passed = False
                            break
                        r2 -= 1
                if not is_passed:
                    break

            if is_passed:
                advance = rank if color == 1 else (7 - rank)
                base = 30 + advance * (25 if endgame else 12)
                score += base * sign

            bb_tmp ^= lsb

    # ---------- Roi central en finale ----------
    if endgame:
        # roi blanc
        if wk:
            lsb = wk & -wk
            sq = lsb.bit_length() - 1
            file = sq & 7
            rank = sq >> 3
            cf = 3 if file <= 3 else 4
            cr = 3 if rank <= 3 else 4
            center = (3 - abs(file - cf)) + (3 - abs(rank - cr))
            score += 10 * center

        # roi noir
        if bk:
            lsb = bk & -bk
            sq = lsb.bit_length() - 1
            file = sq & 7
            rank = sq >> 3
            cf = 3 if file <= 3 else 4
            cr = 3 if rank <= 3 else 4
            center = (3 - abs(file - cf)) + (3 - abs(rank - cr))
            score -= 10 * center

    # ---------- 50 coups ----------
    if halfmove_clock > 50:
        score -= (halfmove_clock - 50) * 2

    # Score du point de vue du camp au trait
    return score if stm_white else -score
