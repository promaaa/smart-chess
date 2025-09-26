import numpy as np

class Chess:
    def __init__(self):
        self.bitboards = {
            'P': np.uint64(0x000000000000FF00),
            'N': np.uint64(0x0000000000000042),
            'B': np.uint64(0x0000000000000024),
            'R': np.uint64(0x0000000000000081),
            'Q': np.uint64(0x0000000000000008),
            'K': np.uint64(0x0000000000000010),
            'p': np.uint64(0x00FF000000000000),
            'n': np.uint64(0x4200000000000000),
            'b': np.uint64(0x2400000000000000),
            'r': np.uint64(0x8100000000000000),
            'q': np.uint64(0x0800000000000000),
            'k': np.uint64(0x1000000000000000)
        }
        self.white_to_move = True
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_target = None
        self.check = False
        self.history = []

    # safe square_mask: do Python int shift then cast to np.uint64
    def square_mask(self, sq):
        return np.uint64(1 << int(sq))

    def occupancy(self):
        occ = np.uint64(0)
        for bb in self.bitboards.values():
            occ |= bb
        return occ

    def pieces_of_color(self, white):
        mask = np.uint64(0)
        for p, bb in self.bitboards.items():
            if (p.isupper() and white) or (p.islower() and not white):
                mask |= bb
        return mask

    def color_of_piece_char(self, p):
        return p.isupper()

    def print_board(self):
        board = ['.' for _ in range(64)]
        for piece, bitboard in self.bitboards.items():
            for i in range(64):
                if bool(bitboard & self.square_mask(i)):
                    board[i] = piece
        for rank in range(7, -1, -1):
            print(' '.join(board[rank*8:(rank+1)*8]))
        print()

    # --- compute_* et helpers (inchangés logiquement) ---
    def compute_king_moves(self, square, piece=None):
        king_moves = np.uint64(0)
        directions = [1, -1, 8, -8, 7, -7, 9, -9]
        occ = self.occupancy()
        own = np.uint64(0)
        if piece is not None:
            own = self.pieces_of_color(self.color_of_piece_char(piece))

        for direction in directions:
            target_square = square + direction
            if 0 <= target_square < 64:
                if abs((square % 8) - (target_square % 8)) <= 1:
                    if not (own & self.square_mask(target_square)):
                        king_moves |= self.square_mask(target_square)

        # castling handled as before (uses is_square_attacked)
        if piece == 'K':
            if self.castling_rights.get('K', False):
                f1 = 5; g1 = 6; e1 = 4
                if not (self.occupancy() & (self.square_mask(f1) | self.square_mask(g1))):
                    if (not self.is_square_attacked(e1, by_white=False) and
                        not self.is_square_attacked(f1, by_white=False) and
                        not self.is_square_attacked(g1, by_white=False)):
                        king_moves |= self.square_mask(g1)
            if self.castling_rights.get('Q', False):
                d1 = 3; c1 = 2; b1 = 1; e1 = 4
                if not (self.occupancy() & (self.square_mask(d1) | self.square_mask(c1) | self.square_mask(b1))):
                    if (not self.is_square_attacked(e1, by_white=False) and
                        not self.is_square_attacked(d1, by_white=False) and
                        not self.is_square_attacked(c1, by_white=False)):
                        king_moves |= self.square_mask(c1)
        elif piece == 'k':
            if self.castling_rights.get('k', False):
                f8 = 61; g8 = 62; e8 = 60
                if not (self.occupancy() & (self.square_mask(f8) | self.square_mask(g8))):
                    if (not self.is_square_attacked(e8, by_white=True) and
                        not self.is_square_attacked(f8, by_white=True) and
                        not self.is_square_attacked(g8, by_white=True)):
                        king_moves |= self.square_mask(g8)
            if self.castling_rights.get('q', False):
                d8 = 59; c8 = 58; b8 = 57; e8 = 60
                if not (self.occupancy() & (self.square_mask(d8) | self.square_mask(c8) | self.square_mask(b8))):
                    if (not self.is_square_attacked(e8, by_white=True) and
                        not self.is_square_attacked(d8, by_white=True) and
                        not self.is_square_attacked(c8, by_white=True)):
                        king_moves |= self.square_mask(c8)

        return king_moves

    def compute_knight_moves(self, square, piece=None):
        knight_moves = np.uint64(0)
        directions = [15, 17, 10, 6, -15, -17, -10, -6]
        own = np.uint64(0)
        if piece is not None:
            own = self.pieces_of_color(self.color_of_piece_char(piece))
        for direction in directions:
            target_square = square + direction
            if 0 <= target_square < 64:
                if abs((square % 8) - (target_square % 8)) <= 2:
                    if not (own & self.square_mask(target_square)):
                        knight_moves |= self.square_mask(target_square)
        return knight_moves

    def compute_pawn_moves(self, square, is_white):
        pawn_moves = np.uint64(0)
        occ = self.occupancy()
        own = self.pieces_of_color(is_white)
        enemy = self.occupancy() & ~own

        if is_white:
            one_forward = square + 8
            two_forward = square + 16
            if one_forward < 64 and not (occ & self.square_mask(one_forward)):
                pawn_moves |= self.square_mask(one_forward)
                if (square // 8) == 1 and two_forward < 64 and not (occ & self.square_mask(two_forward)):
                    pawn_moves |= self.square_mask(two_forward)
            if square % 8 > 0:
                left = square + 7
                if left < 64 and (enemy & self.square_mask(left)):
                    pawn_moves |= self.square_mask(left)
            if square % 8 < 7:
                right = square + 9
                if right < 64 and (enemy & self.square_mask(right)):
                    pawn_moves |= self.square_mask(right)
            if self.en_passant_target is not None and (square // 8) == 4:
                if square % 8 > 0 and self.en_passant_target == square + 7:
                    pawn_moves |= self.square_mask(self.en_passant_target)
                if square % 8 < 7 and self.en_passant_target == square + 9:
                    pawn_moves |= self.square_mask(self.en_passant_target)
        else:
            one_forward = square - 8
            two_forward = square - 16
            if one_forward >= 0 and not (occ & self.square_mask(one_forward)):
                pawn_moves |= self.square_mask(one_forward)
                if (square // 8) == 6 and two_forward >= 0 and not (occ & self.square_mask(two_forward)):
                    pawn_moves |= self.square_mask(two_forward)
            if square % 8 > 0:
                left = square - 9
                if left >= 0 and (enemy & self.square_mask(left)):
                    pawn_moves |= self.square_mask(left)
            if square % 8 < 7:
                right = square - 7
                if right >= 0 and (enemy & self.square_mask(right)):
                    pawn_moves |= self.square_mask(right)
            if self.en_passant_target is not None and (square // 8) == 3:
                if square % 8 > 0 and self.en_passant_target == square - 9:
                    pawn_moves |= self.square_mask(self.en_passant_target)
                if square % 8 < 7 and self.en_passant_target == square - 7:
                    pawn_moves |= self.square_mask(self.en_passant_target)

        return pawn_moves

    def compute_rook_moves(self, square, piece=None):
        rook_moves = np.uint64(0)
        occ = self.occupancy()
        own = np.uint64(0)
        if piece is not None:
            own = self.pieces_of_color(self.color_of_piece_char(piece))
        directions = [1, -1, 8, -8]
        for direction in directions:
            target_square = square
            while True:
                target_square += direction
                if not (0 <= target_square < 64):
                    break
                if direction in [1, -1] and (target_square // 8) != (square // 8):
                    break
                mask = self.square_mask(target_square)
                if own & mask:
                    break
                rook_moves |= mask
                if occ & mask:
                    break
        return rook_moves

    def compute_bishop_moves(self, square, piece=None):
        bishop_moves = np.uint64(0)
        occ = self.occupancy()
        own = np.uint64(0)
        if piece is not None:
            own = self.pieces_of_color(self.color_of_piece_char(piece))
        directions = [7, -7, 9, -9]
        for direction in directions:
            target_square = square
            while True:
                target_square += direction
                if not (0 <= target_square < 64):
                    break
                if abs((target_square % 8) - (square % 8)) != abs((target_square // 8) - (square // 8)):
                    break
                mask = self.square_mask(target_square)
                if own & mask:
                    break
                bishop_moves |= mask
                if occ & mask:
                    break
        return bishop_moves

    def compute_queen_moves(self, square, piece=None):
        return self.compute_rook_moves(square, piece) | self.compute_bishop_moves(square, piece)

    def get_all_moves(self, square):
        piece = None
        from_mask = self.square_mask(square)
        for p, bitboard in self.bitboards.items():
            if bitboard & from_mask:
                piece = p
                break
        if piece is None:
            return np.uint64(0)

        if piece in ['K', 'k']:
            return self.compute_king_moves(square, piece)
        elif piece in ['N', 'n']:
            return self.compute_knight_moves(square, piece)
        elif piece in ['P']:
            return self.compute_pawn_moves(square, True)
        elif piece in ['p']:
            return self.compute_pawn_moves(square, False)
        elif piece in ['R', 'r']:
            return self.compute_rook_moves(square, piece)
        elif piece in ['B', 'b']:
            return self.compute_bishop_moves(square, piece)
        elif piece in ['Q', 'q']:
            return self.compute_queen_moves(square, piece)
        else:
            return np.uint64(0)

    def ray_attacks_from(self, square, directions):
        occ = self.occupancy()
        attacks = np.uint64(0)
        for direction in directions:
            target = square
            while True:
                target += direction
                if not (0 <= target < 64):
                    break
                if direction in [1, -1] and (target // 8) != (square // 8):
                    break
                if direction in [7, -7, 9, -9]:
                    if abs((target % 8) - (square % 8)) != abs((target // 8) - (square // 8)):
                        break
                attacks |= self.square_mask(target)
                if occ & self.square_mask(target):
                    break
        return attacks

    def is_square_attacked(self, square, by_white):
        mask = self.square_mask(square)
        # pawns
        pawn_bb = self.bitboards['P'] if by_white else self.bitboards['p']
        for i in range(64):
            if bool(pawn_bb & self.square_mask(i)):
                if by_white:
                    if i % 8 > 0 and i + 7 == square:
                        return True
                    if i % 8 < 7 and i + 9 == square:
                        return True
                else:
                    if i % 8 < 7 and i - 7 == square:
                        return True
                    if i % 8 > 0 and i - 9 == square:
                        return True

        # knights
        knight_bb = self.bitboards['N'] if by_white else self.bitboards['n']
        for i in range(64):
            if bool(knight_bb & self.square_mask(i)):
                if self.compute_knight_moves(i, 'N' if by_white else 'n') & mask:
                    return True

        # bishops & queens
        bishop_bb = self.bitboards['B'] if by_white else self.bitboards['b']
        queen_bb = self.bitboards['Q'] if by_white else self.bitboards['q']
        for i in range(64):
            if bool((bishop_bb | queen_bb) & self.square_mask(i)):
                if self.ray_attacks_from(i, [7, -7, 9, -9]) & mask:
                    return True

        # rooks & queens
        rook_bb = self.bitboards['R'] if by_white else self.bitboards['r']
        for i in range(64):
            if bool((rook_bb | queen_bb) & self.square_mask(i)):
                if self.ray_attacks_from(i, [1, -1, 8, -8]) & mask:
                    return True

        # king
        king_bb = self.bitboards['K'] if by_white else self.bitboards['k']
        for i in range(64):
            if bool(king_bb & self.square_mask(i)):
                if self.compute_king_moves(i, 'K' if by_white else 'k') & mask:
                    return True

        return False

    def is_in_check(self, white_color):
        king_piece = 'K' if white_color else 'k'
        king_bb = self.bitboards.get(king_piece, np.uint64(0))
        if king_bb == 0:
            return False
        king_square = None
        for i in range(64):
            if bool(king_bb & self.square_mask(i)):
                king_square = i
                break
        if king_square is None:
            return False
        return self.is_square_attacked(king_square, by_white=not white_color)

    def move_piece(self, from_square, to_square, promotion=None):
        from_mask = self.square_mask(from_square)
        to_mask = self.square_mask(to_square)

        moving_piece = None
        for p, bb in self.bitboards.items():
            if bb & from_mask:
                moving_piece = p
                break
        if moving_piece is None:
            print("No piece at the source square.")
            return

        legal_moves = self.get_all_moves(from_square)
        if not (legal_moves & to_mask):
            raise ValueError("Illegal move (not in pseudo-legal moves)")

        prev_bitboards = {k: np.uint64(v) for k, v in self.bitboards.items()}
        prev_en_passant = self.en_passant_target
        prev_castling = dict(self.castling_rights)
        prev_white_to_move = self.white_to_move

        moving_side_white = self.white_to_move

        captured_piece = None
        en_passant_capture_square = None
        rook_from = None
        rook_to = None
        promotion_applied = None

        # normal capture
        for p, bb in self.bitboards.items():
            if p != moving_piece and (bb & to_mask):
                captured_piece = p
                self.bitboards[p] &= ~to_mask
                break

        # en-passant capture
        if moving_piece in ['P', 'p'] and self.en_passant_target is not None and to_square == self.en_passant_target:
            if moving_piece == 'P':
                en_passant_capture_square = to_square - 8
            else:
                en_passant_capture_square = to_square + 8
            mask_ep = self.square_mask(en_passant_capture_square)
            for p, bb in self.bitboards.items():
                if p.lower() == 'p' and (bb & mask_ep):
                    captured_piece = p
                    self.bitboards[p] &= ~mask_ep
                    break

        # move
        self.bitboards[moving_piece] &= ~from_mask

        is_pawn = moving_piece in ['P', 'p']
        promotion_rank_reached = False
        if is_pawn:
            if moving_piece == 'P' and (to_square // 8) == 7:
                promotion_rank_reached = True
            if moving_piece == 'p' and (to_square // 8) == 0:
                promotion_rank_reached = True

        if promotion_rank_reached:
            if promotion is None:
                promo_char = 'Q' if moving_piece.isupper() else 'q'
            else:
                promo_char = promotion if (promotion.isupper() == moving_piece.isupper()) else (promotion.upper() if moving_piece.isupper() else promotion.lower())
            if promo_char not in self.bitboards:
                self.bitboards[promo_char] = np.uint64(0)
            self.bitboards[promo_char] |= to_mask
            promotion_applied = promo_char
        else:
            self.bitboards[moving_piece] |= to_mask

        # castling rook move
        if moving_piece in ['K', 'k']:
            diff = to_square - from_square
            if abs(diff) == 2:
                if diff > 0:
                    rook_from = from_square + 3
                    rook_to = from_square + 1
                else:
                    rook_from = from_square - 4
                    rook_to = from_square - 1
                rf_mask = self.square_mask(rook_from)
                rt_mask = self.square_mask(rook_to)
                for p, bb in self.bitboards.items():
                    if p.lower() == 'r' and (bb & rf_mask):
                        self.bitboards[p] &= ~rf_mask
                        self.bitboards[p] |= rt_mask
                        break

        # update castling rights
        if moving_piece == 'K':
            self.castling_rights['K'] = False
            self.castling_rights['Q'] = False
        if moving_piece == 'k':
            self.castling_rights['k'] = False
            self.castling_rights['q'] = False
        if moving_piece in ['R', 'r']:
            if from_square == 7:
                self.castling_rights['K'] = False
            if from_square == 0:
                self.castling_rights['Q'] = False
            if from_square == 63:
                self.castling_rights['k'] = False
            if from_square == 56:
                self.castling_rights['q'] = False
        if captured_piece in ['R', 'r']:
            if to_square == 7:
                self.castling_rights['K'] = False
            if to_square == 0:
                self.castling_rights['Q'] = False
            if to_square == 63:
                self.castling_rights['k'] = False
            if to_square == 56:
                self.castling_rights['q'] = False

        # update en-passant target (double push)
        self.en_passant_target = None
        if moving_piece == 'P' and (from_square // 8) == 1 and (to_square // 8) == 3:
            self.en_passant_target = from_square + 8
        elif moving_piece == 'p' and (from_square // 8) == 6 and (to_square // 8) == 4:
            self.en_passant_target = from_square - 8

        # --- safety: verify king hasn't disappeared and that move doesn't leave king in check ---
        # If illegal: rollback and raise
        if self.is_in_check(moving_side_white):
            # restore snapshot
            for k, v in prev_bitboards.items():
                self.bitboards[k] = np.uint64(v)
            self.en_passant_target = prev_en_passant
            self.castling_rights = prev_castling
            self.white_to_move = prev_white_to_move
            raise ValueError("Illegal move: leaves king in check")

        # push history
        move_record = {
            'from': from_square,
            'to': to_square,
            'moving_piece': moving_piece,
            'captured_piece': captured_piece,
            'en_passant_capture_square': en_passant_capture_square,
            'promotion': promotion_applied,
            'rook_from': rook_from,
            'rook_to': rook_to,
            'prev_bitboards': prev_bitboards,
            'prev_en_passant': prev_en_passant,
            'prev_castling': prev_castling,
            'prev_white_to_move': prev_white_to_move
        }
        self.history.append(move_record)

        # flip side and update check flag
        self.white_to_move = not self.white_to_move
        self.check = self.is_in_check(self.white_to_move)

    def undo_move(self):
        if not self.history:
            print("No move to undo.")
            return
        record = self.history.pop()

        prev_bitboards = record['prev_bitboards']
        for k, v in prev_bitboards.items():
            self.bitboards[k] = np.uint64(v)

        self.en_passant_target = record['prev_en_passant']
        self.castling_rights = dict(record['prev_castling'])
        self.white_to_move = record['prev_white_to_move']

        # CORRECTION: recompute check for the side to move (previous trait restored)
        self.check = self.is_in_check(self.white_to_move)
        return

