import numpy as np

class Chess:
    # implementing a chess board representation with bitboards

    def __init__(self):
        # Initialize bitboards for each piece type and color
        self.bitboards = {
            'P': np.uint64(0x000000000000FF00),  # White Pawns
            'N': np.uint64(0x0000000000000042),  # White Knights
            'B': np.uint64(0x0000000000000024),  # White Bishops
            'R': np.uint64(0x0000000000000081),  # White Rooks
            'Q': np.uint64(0x0000000000000008),  # White Queen
            'K': np.uint64(0x0000000000000010),  # White King
            'p': np.uint64(0x00FF000000000000),  # Black Pawns
            'n': np.uint64(0x4200000000000000),  # Black Knights
            'b': np.uint64(0x2400000000000000),  # Black Bishops
            'r': np.uint64(0x8100000000000000),  # Black Rooks
            'q': np.uint64(0x0800000000000000),  # Black Queen
            'k': np.uint64(0x1000000000000000)   # Black King
        }
        self.white_to_move = True
        # castling_rights: 'K' white kingside, 'Q' white queenside, 'k' black kingside, 'q' black queenside
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_target = None   # index de case ou None
        self.check = False

    # -------------------- utilitaires --------------------
    def square_mask(self, sq):
        return np.uint64(1) << np.uint64(sq)

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
        # True if white (uppercase), False if black (lowercase)
        return p.isupper()

    # -------------------- affichage --------------------
    def print_board(self):
        # Print the board in a human-readable format
        board = ['.' for _ in range(64)]
        for piece, bitboard in self.bitboards.items():
            for i in range(64):
                if (bitboard >> i) & 1:
                    board[i] = piece
        for rank in range(7, -1, -1):
            print(' '.join(board[rank*8:(rank+1)*8]))
        print()

    # -------------------- déplacements de base (avec occupation & couleur) --------------------
    def compute_king_moves(self, square, piece=None):
        # Compute possible king moves from a given square.
        # Exclut les cases occupées par ses propres pièces.
        king_moves = np.uint64(0)
        directions = [1, -1, 8, -8, 7, -7, 9, -9]
        occ = self.occupancy()
        own = np.uint64(0)
        if piece is not None:
            own = self.pieces_of_color(self.color_of_piece_char(piece))

        for direction in directions:
            target_square = square + direction
            if 0 <= target_square < 64:
                # wrap-around checks
                if abs((square % 8) - (target_square % 8)) <= 1:
                    if not (own & self.square_mask(target_square)):
                        king_moves |= self.square_mask(target_square)

        # Castling (ajout seulement si piece fournie)
        if piece == 'K':  # white
            # kingside
            if self.castling_rights.get('K', False):
                f1 = 5; g1 = 6; e1 = 4
                if not (self.occupancy() & (self.square_mask(f1) | self.square_mask(g1))):
                    # e1, f1, g1 must not be attaqués par noir
                    if (not self.is_square_attacked(e1, by_white=False) and
                        not self.is_square_attacked(f1, by_white=False) and
                        not self.is_square_attacked(g1, by_white=False)):
                        king_moves |= self.square_mask(g1)
            # queenside
            if self.castling_rights.get('Q', False):
                d1 = 3; c1 = 2; b1 = 1; e1 = 4
                if not (self.occupancy() & (self.square_mask(d1) | self.square_mask(c1) | self.square_mask(b1))):
                    if (not self.is_square_attacked(e1, by_white=False) and
                        not self.is_square_attacked(d1, by_white=False) and
                        not self.is_square_attacked(c1, by_white=False)):
                        king_moves |= self.square_mask(c1)
        elif piece == 'k':  # black
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
        # Compute possible knight moves from a given square, excluding own pieces
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
        # Compute possible pawn moves including captures and en-passant, taking occupation into account.
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
            # captures
            if square % 8 > 0:
                left = square + 7
                if left < 64 and (enemy & self.square_mask(left)):
                    pawn_moves |= self.square_mask(left)
            if square % 8 < 7:
                right = square + 9
                if right < 64 and (enemy & self.square_mask(right)):
                    pawn_moves |= self.square_mask(right)
            # en-passant
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
            # captures
            if square % 8 > 0:
                left = square - 9
                if left >= 0 and (enemy & self.square_mask(left)):
                    pawn_moves |= self.square_mask(left)
            if square % 8 < 7:
                right = square - 7
                if right >= 0 and (enemy & self.square_mask(right)):
                    pawn_moves |= self.square_mask(right)
            # en-passant
            if self.en_passant_target is not None and (square // 8) == 3:
                if square % 8 > 0 and self.en_passant_target == square - 9:
                    pawn_moves |= self.square_mask(self.en_passant_target)
                if square % 8 < 7 and self.en_passant_target == square - 7:
                    pawn_moves |= self.square_mask(self.en_passant_target)

        return pawn_moves

    def compute_rook_moves(self, square, piece=None):
        # Compute rook moves blocked by occupancy and excluding own pieces on capture destinations
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
                # horizontal wrap-around guard
                if direction in [1, -1] and (target_square // 8) != (square // 8):
                    break
                mask = self.square_mask(target_square)
                if own & mask:
                    break
                rook_moves |= mask
                if occ & mask:  # blocked by some piece (capture possible if not own) then stop
                    break
        return rook_moves

    def compute_bishop_moves(self, square, piece=None):
        # Compute bishop moves blocked by occupancy and excluding own pieces on capture destinations
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
                # diagonal wrap-around guard relative to original square
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
        # Get all possible moves for the piece on the given square (pseudo-légal: supprime les moves sur ses propres pièces)
        piece = None
        from_mask = np.uint64(1) << np.uint64(square)
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

    # -------------------- détection d'attaque / échec (implémentation non-récursive) --------------------
    def ray_attacks_from(self, square, directions):
        occ = self.occupancy()
        attacks = np.uint64(0)
        for direction in directions:
            target = square
            while True:
                target += direction
                if not (0 <= target < 64):
                    break
                # wrap-around checks (horizontal/diagonal)
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
        """Retourne True si la case `square` est attaquée par la couleur `by_white`."""
        mask = self.square_mask(square)
        # Pawns
        pawn_bb = self.bitboards['P'] if by_white else self.bitboards['p']
        for i in range(64):
            if (pawn_bb >> i) & 1:
                if by_white:
                    # white pawn attacks i+7 (left) and i+9 (right)
                    if i % 8 > 0 and i + 7 == square:
                        return True
                    if i % 8 < 7 and i + 9 == square:
                        return True
                else:
                    # black pawn attacks i-7 and i-9
                    if i % 8 < 7 and i - 7 == square:
                        return True
                    if i % 8 > 0 and i - 9 == square:
                        return True

        # Knights
        knight_bb = self.bitboards['N'] if by_white else self.bitboards['n']
        for i in range(64):
            if (knight_bb >> i) & 1:
                if self.compute_knight_moves(i, 'N' if by_white else 'n') & mask:
                    return True

        # Bishops & Queens (diagonals)
        bishop_bb = self.bitboards['B'] if by_white else self.bitboards['b']
        queen_bb = self.bitboards['Q'] if by_white else self.bitboards['q']
        for i in range(64):
            if ((bishop_bb | queen_bb) >> i) & 1:
                if self.ray_attacks_from(i, [7, -7, 9, -9]) & mask:
                    return True

        # Rooks & Queens (orthogonals)
        rook_bb = self.bitboards['R'] if by_white else self.bitboards['r']
        for i in range(64):
            if ((rook_bb | queen_bb) >> i) & 1:
                if self.ray_attacks_from(i, [1, -1, 8, -8]) & mask:
                    return True

        # King (adjacent)
        king_bb = self.bitboards['K'] if by_white else self.bitboards['k']
        for i in range(64):
            if (king_bb >> i) & 1:
                if self.compute_king_moves(i, 'K' if by_white else 'k') & mask:
                    return True

        return False

    def is_in_check(self, white_color):
        """Retourne True si le roi de la couleur `white_color` est en échec."""
        king_piece = 'K' if white_color else 'k'
        king_bb = self.bitboards.get(king_piece, np.uint64(0))
        if king_bb == 0:
            # pas de roi (position illégale), on retourne False pour éviter crash
            return False
        king_square = None
        for i in range(64):
            if (king_bb >> i) & 1:
                king_square = i
                break
        if king_square is None:
            return False
        return self.is_square_attacked(king_square, by_white=not white_color)

    # -------------------- mouvement (avec prise en passant, roque, promotion, échec) --------------------
    def move_piece(self, from_square, to_square, promotion=None):
        """
        Déplace la pièce de from_square à to_square.
        promotion : 'Q','R','B','N' pour blanc, ou 'q','r','b','n' pour noir. Si None -> automatique reine.
        Vérifie que le coup est pseudo-légal via get_all_moves, puis refuse si il expose le roi.
        """
        from_mask = self.square_mask(from_square)
        to_mask = self.square_mask(to_square)

        # trouver la pièce qui bouge
        moving_piece = None
        for p, bb in self.bitboards.items():
            if bb & from_mask:
                moving_piece = p
                break
        if moving_piece is None:
            print("No piece at the source square.")
            return

        # pseudo-légalité
        legal_moves = self.get_all_moves(from_square)
        if not (legal_moves & to_mask):
            raise ValueError("Illegal move (not in pseudo-legal moves)")

        # sauvegarde d'état pour undo si nécessaire
        prev_bitboards = {k: np.uint64(v) for k, v in self.bitboards.items()}
        prev_en_passant = self.en_passant_target
        prev_castling = dict(self.castling_rights)

        moving_side_white = self.white_to_move

        captured_piece = None
        en_passant_capture_square = None

        # --- handle captures and en-passant BEFORE placing the piece ---
        # capture normale (si une pièce adverse est sur to_square)
        for p, bb in self.bitboards.items():
            if p != moving_piece and (bb & to_mask):
                captured_piece = p
                # retirer la pièce capturée (sera réappliqué en cas d'undo)
                self.bitboards[p] &= ~to_mask
                break

        # en-passant capture : si le mouvement est un pion se déplaçant vers en_passant_target
        if moving_piece in ['P', 'p'] and self.en_passant_target is not None and to_square == self.en_passant_target:
            # la prise est sur la case derrière la cible
            if moving_piece == 'P':
                en_passant_capture_square = to_square - 8
            else:
                en_passant_capture_square = to_square + 8
            mask_ep = self.square_mask(en_passant_capture_square)
            # trouver quel pion est capturé
            for p, bb in self.bitboards.items():
                if p.lower() == 'p' and (bb & mask_ep):
                    captured_piece = p
                    self.bitboards[p] &= ~mask_ep
                    break

        # --- effectuer le déplacement (retirer de from, ajouter à to) ---
        self.bitboards[moving_piece] &= ~from_mask  # remove from origin

        # Promotion
        is_pawn = moving_piece in ['P', 'p']
        promotion_rank_reached = False
        if is_pawn:
            if moving_piece == 'P' and (to_square // 8) == 7:
                promotion_rank_reached = True
            if moving_piece == 'p' and (to_square // 8) == 0:
                promotion_rank_reached = True

        if promotion_rank_reached:
            # choisir la pièce promue
            if promotion is None:
                promo_char = 'Q' if moving_piece.isupper() else 'q'
            else:
                # si utilisateur a passé 'Q'/'R'... on accepte en uppercase ou lowercase
                promo_char = promotion if (promotion.isupper() == moving_piece.isupper()) else (promotion.upper() if moving_piece.isupper() else promotion.lower())
            # ajouter la pièce promue
            if promo_char not in self.bitboards:
                # créer la clé si absente (rare) et mettre à 0
                self.bitboards[promo_char] = np.uint64(0)
            self.bitboards[promo_char] |= to_mask
        else:
            # mouvement normal : placer la même pièce sur to_square
            self.bitboards[moving_piece] |= to_mask

        # --- Cas spécial : roque (déplacement de la tour) ---
        if moving_piece in ['K', 'k']:
            diff = to_square - from_square
            if abs(diff) == 2:
                # petit roque
                if diff > 0:
                    rook_from = from_square + 3
                    rook_to = from_square + 1
                else:  # grand roque
                    rook_from = from_square - 4
                    rook_to = from_square - 1
                rf_mask = self.square_mask(rook_from)
                rt_mask = self.square_mask(rook_to)
                # trouver la tour correspondante (sécurité)
                for p, bb in self.bitboards.items():
                    if p.lower() == 'r' and (bb & rf_mask):
                        self.bitboards[p] &= ~rf_mask
                        self.bitboards[p] |= rt_mask
                        break

        # --- Mettre à jour les droits de roque si nécessaire ---
        # Si le roi a bougé : désactiver ses droits
        if moving_piece == 'K':
            self.castling_rights['K'] = False
            self.castling_rights['Q'] = False
        if moving_piece == 'k':
            self.castling_rights['k'] = False
            self.castling_rights['q'] = False
        # Si une tour initiale a bougé, désactiver le droit correspondant
        if moving_piece in ['R', 'r']:
            if from_square == 7:  # h1
                self.castling_rights['K'] = False
            if from_square == 0:  # a1
                self.castling_rights['Q'] = False
            if from_square == 63:  # h8
                self.castling_rights['k'] = False
            if from_square == 56:  # a8
                self.castling_rights['q'] = False
        # Si on a capturé une tour initiale, désactiver les droits de l'adversaire
        if captured_piece in ['R', 'r']:
            if to_square == 7:
                self.castling_rights['K'] = False
            if to_square == 0:
                self.castling_rights['Q'] = False
            if to_square == 63:
                self.castling_rights['k'] = False
            if to_square == 56:
                self.castling_rights['q'] = False

        # --- Mettre à jour en_passant_target (seulement si double-pas d'un pion) ---
        self.en_passant_target = None
        if moving_piece == 'P' and (from_square // 8) == 1 and (to_square // 8) == 3:
            # double pas blanc
            self.en_passant_target = from_square + 8
        elif moving_piece == 'p' and (from_square // 8) == 6 and (to_square // 8) == 4:
            # double pas noir
            self.en_passant_target = from_square - 8

        # --- Vérifier que le coup ne laisse pas son roi en échec ---
        if self.is_in_check(moving_side_white):
            # undo
            for k, v in prev_bitboards.items():
                self.bitboards[k] = np.uint64(v)
            self.en_passant_target = prev_en_passant
            self.castling_rights = prev_castling
            raise ValueError("Illegal move: leaves king in check")

        # --- Si tout est ok, on bascule le trait et on signale l'échec si présent pour le camp au trait ---
        self.white_to_move = not self.white_to_move
        self.check = self.is_in_check(self.white_to_move)

    # fin de la classe

        
# Example usage
chess = Chess()
chess.print_board()
chess.move_piece(52, 36)  # Move white pawn from e2 to e4
chess.print_board()

