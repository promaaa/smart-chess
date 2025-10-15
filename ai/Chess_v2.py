'''''''''
Ce code implémente un moteur d’échecs simplifié en Python, basé sur la représentation **bitboard** (chaque case de l’échiquier correspond à un bit d’un entier 64 bits).

Voici les grandes lignes de son fonctionnement :

* **Initialisation (`__init__`)** : crée un échiquier standard avec les pièces placées à leur position de départ sous forme de bitboards.
* **Fonctions de base** : `square_mask`, `occupancy`, `pieces_of_color`, etc., servent à manipuler les bitboards (déterminer quelles cases sont occupées, par quelle couleur, etc.).
* **Affichage (`print_board`)** : affiche l’échiquier sous forme textuelle.
* **Calcul des coups (`compute_*_moves`)** : calcule les mouvements possibles pour chaque type de pièce (roi, cavalier, pion, tour, fou, dame), en prenant en compte les règles comme le roque, la prise en passant ou les promotions.
* **Détection d’attaque et d’échec (`is_square_attacked`, `is_in_check`)** : vérifie si une case (ou le roi) est attaquée par une pièce adverse.
* **Gestion des coups (`move_piece`, `undo_move`)** : applique un coup sur le plateau (en mettant à jour les bitboards, les droits de roque, la prise en passant, etc.), vérifie la légalité du coup (notamment si le roi reste protégé), et permet de revenir en arrière.
'''''''''

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
    def compute_king_moves_basic(self, square, piece=None):
        """
        Mouvements basiques du roi (sans roque) pour éviter la récursion
        """
        king_moves = np.uint64(0)
        directions = [1, -1, 8, -8, 7, -7, 9, -9]
        own = np.uint64(0)
        if piece is not None:
            own = self.pieces_of_color(self.color_of_piece_char(piece))

        for direction in directions:
            target_square = square + direction
            if 0 <= target_square < 64:
                if abs((square % 8) - (target_square % 8)) <= 1:
                    if not (own & self.square_mask(target_square)):
                        king_moves |= self.square_mask(target_square)

        return king_moves
    # --- compute_* et helpers (inchangés logiquement) ---
    def compute_king_moves(self, square, piece=None):
        """
        Mouvements complets du roi (avec roque)
        """
        # Commencer par les mouvements basiques
        king_moves = self.compute_king_moves_basic(square, piece)

        # Ajouter le roque seulement si une pièce est spécifiée
        if piece is None:
            return king_moves

        # Gestion du roque
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
    def compute_pawn_attacks(self, square, piece):
        """
        Attaques du pion (utilisé par is_square_attacked)
        """
        attacks = np.uint64(0)
        is_white = piece.isupper()
        
        if is_white:
            # Pion blanc attaque vers le haut
            if square % 8 > 0 and square + 7 < 64:  # Attaque diagonale gauche
                attacks |= self.square_mask(square + 7)
            if square % 8 < 7 and square + 9 < 64:  # Attaque diagonale droite
                attacks |= self.square_mask(square + 9)
        else:
            # Pion noir attaque vers le bas
            if square % 8 < 7 and square - 7 >= 0:  # Attaque diagonale droite
                attacks |= self.square_mask(square - 7)
            if square % 8 > 0 and square - 9 >= 0:  # Attaque diagonale gauche
                attacks |= self.square_mask(square - 9)
        
        return attacks
    def is_square_attacked(self, square, by_white):
        """
        Version corrigée qui utilise compute_king_moves_basic pour éviter la récursion
        """
        mask = self.square_mask(square)
        
        # Vérifier les attaques de pions
        pawn_bb = self.bitboards['P'] if by_white else self.bitboards['p']
        pawn_piece = 'P' if by_white else 'p'
        for i in range(64):
            if bool(pawn_bb & self.square_mask(i)):
                if self.compute_pawn_attacks(i, pawn_piece) & mask:
                    return True

        # Vérifier les attaques de cavaliers
        knight_bb = self.bitboards['N'] if by_white else self.bitboards['n']
        knight_piece = 'N' if by_white else 'n'
        for i in range(64):
            if bool(knight_bb & self.square_mask(i)):
                if self.compute_knight_moves(i, knight_piece) & mask:
                    return True
        # Vérifier les attaques de fous et dames (diagonales)
        bishop_bb = self.bitboards['B'] if by_white else self.bitboards['b']
        queen_bb = self.bitboards['Q'] if by_white else self.bitboards['q']
        for i in range(64):
            if bool((bishop_bb | queen_bb) & self.square_mask(i)):
                if self.ray_attacks_from(i, [7, -7, 9, -9]) & mask:
                    return True

        # Vérifier les attaques de tours et dames (lignes/colonnes)
        rook_bb = self.bitboards['R'] if by_white else self.bitboards['r']
        for i in range(64):
            if bool((rook_bb | queen_bb) & self.square_mask(i)):
                if self.ray_attacks_from(i, [1, -1, 8, -8]) & mask:
                    return True

        # Vérifier les attaques du roi (UTILISER LA VERSION BASIQUE)
        king_bb = self.bitboards['K'] if by_white else self.bitboards['k']
        king_piece = 'K' if by_white else 'k'
        for i in range(64):
            if bool(king_bb & self.square_mask(i)):
                # ✅ UTILISER compute_king_moves_basic au lieu de compute_king_moves
                if self.compute_king_moves_basic(i, king_piece) & mask:
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

    def move_piece(self, from_sq: int, to_sq: int, promotion: str = None):
        """
        Déplace une pièce de from_sq vers to_sq, avec option de promotion.
        Enregistre uniquement les deltas nécessaires pour un undo rapide.
        """
        from_mask = self.square_mask(from_sq)
        to_mask = self.square_mask(to_sq)

        # sauvegarder un snapshot complet au cas où on doive restaurer
        prev_bitboards = {k: int(v) for k, v in self.bitboards.items()}
        prev_castling = dict(self.castling_rights)
        prev_en_passant = self.en_passant_target
        prev_white_to_move = self.white_to_move

        moving_piece = None
        captured_piece = None
        captured_square = None

        # Trouver la pièce qui bouge
        for piece, bb in self.bitboards.items():
            if bb & from_mask:
                moving_piece = piece
                break
        if moving_piece is None:
            raise RuntimeError(f"Aucune pièce trouvée sur la case {from_sq}")

        # Gérer capture normale sur la case de destination
        for piece, bb in self.bitboards.items():
            if bb & to_mask:
                captured_piece = piece
                captured_square = to_sq
                # retirer la pièce capturée (on modifiera l'état, on restaurera si nécessaire)
                self.bitboards[piece] &= ~to_mask
                break

        # Gérer capture en passant
        if moving_piece in ('P', 'p') and self.en_passant_target is not None and to_sq == self.en_passant_target:
            if moving_piece == 'P':  # blanc capture vers le haut
                captured_square = to_sq - 8
                captured_piece = 'p'
            else:  # noir
                captured_square = to_sq + 8
                captured_piece = 'P'
            # retirer la pièce capturée en passant
            self.bitboards[captured_piece] &= ~self.square_mask(captured_square)

        # Déplacer la pièce
        self.bitboards[moving_piece] &= ~from_mask
        self.bitboards[moving_piece] |= to_mask

        # Gestion des promotions : si un pion arrive sur la dernière rangée sans argument, promouvoir en dame
        promotion_needed = False
        if moving_piece == 'P' and (to_sq // 8) == 7:
            promotion_needed = True
        if moving_piece == 'p' and (to_sq // 8) == 0:
            promotion_needed = True

        if promotion_needed:
            promoted_piece = promotion if promotion else ('Q' if moving_piece == 'P' else 'q')
            # retirer le pion promu
            self.bitboards[moving_piece] &= ~to_mask
            # ajouter la pièce promue
            self.bitboards[promoted_piece] |= to_mask
        else:
            # si une promotion explicite est fournie
            if promotion and moving_piece in ('P', 'p'):
                promoted_piece = promotion if moving_piece == 'P' else promotion.lower()
                self.bitboards[moving_piece] &= ~to_mask
                self.bitboards[promoted_piece] |= to_mask

        # Roque: déplacer la tour si nécessaire
        if moving_piece in ('K', 'k') and abs(from_sq - to_sq) == 2:
            if moving_piece == 'K':
                if to_sq == 6:   # petit roque
                    self.bitboards['R'] &= ~self.square_mask(7)
                    self.bitboards['R'] |= self.square_mask(5)
                elif to_sq == 2:  # grand roque
                    self.bitboards['R'] &= ~self.square_mask(0)
                    self.bitboards['R'] |= self.square_mask(3)
            else:
                if to_sq == 62:
                    self.bitboards['r'] &= ~self.square_mask(63)
                    self.bitboards['r'] |= self.square_mask(61)
                elif to_sq == 58:
                    self.bitboards['r'] &= ~self.square_mask(56)
                    self.bitboards['r'] |= self.square_mask(59)

        # Mettre à jour les droits de roque et l'en passant (utilise l'état courant)
        self.update_castling_rights(moving_piece, from_sq)
        self.update_en_passant(moving_piece, from_sq, to_sq)

        # Vérifier la légalité : le camp qui a joué ne doit pas être en échec après le coup
        mover_is_white = moving_piece.isupper()
        try:
            illegal = self.is_in_check(mover_is_white)
        except Exception:
            # en cas d'erreur lors du test d'échec, restaurer et remonter
            illegal = True

        if illegal:
            # restaurer l'état complet précédent
            for k, v in prev_bitboards.items():
                self.bitboards[k] = np.uint64(v)
            self.castling_rights = dict(prev_castling)
            self.en_passant_target = prev_en_passant
            self.white_to_move = prev_white_to_move
            raise ValueError("Illegal move: leaves king in check")

        # Si légal, enregistrer un historique minimal (pour undo)
        self.history.append({
            'from': from_sq,
            'to': to_sq,
            'moving_piece': moving_piece,
            'captured_piece': captured_piece,
            'captured_square': captured_square,
            'promotion': promotion if promotion else (promoted_piece if promotion_needed else None),
            'prev_castling': prev_castling,
            'prev_en_passant': prev_en_passant,
            'prev_white_to_move': prev_white_to_move,
        })

        # Changer le trait
        self.white_to_move = not self.white_to_move
        # Recalculer l'état d'échec pour le côté à jouer
        self.check = self.is_in_check(self.white_to_move)

    def undo_move(self):
        """
        Annule le dernier coup enregistré dans self.history.

        Compatibilité:
        - Si l'entrée d'historique contient 'prev_bitboards', on restaure l'ancien comportement (copie complète).
        - Sinon, on suppose que l'entrée est un enregistrement minimal (delta) créé par move_piece:
        keys attendues: 'from', 'to', 'moving_piece', 'captured_piece' (ou None),
                        'captured_square' (case de capture, utile pour en-passant),
                        'promotion' (None ou 'Q'/'R'... uppercase letter for white),
                        'prev_castling', 'prev_en_passant', 'prev_white_to_move'
        """
        if not hasattr(self, "history") or not self.history:
            # Pas d'historique
            # (Garder le même message qu'avant pour compatibilité ; tu peux le supprimer en prod.)
            print("No move to undo.")
            return

        record = self.history.pop()

        # --- Old format: full bitboards saved ---
        if isinstance(record, dict) and "prev_bitboards" in record:
            prev_bitboards = record['prev_bitboards']
            # restaurer chaque bitboard (on remet le type numpy.uint64 pour rester consistant)
            for k, v in prev_bitboards.items():
                self.bitboards[k] = np.uint64(v)
            self.en_passant_target = record.get('prev_en_passant')
            self.castling_rights = dict(record.get('prev_castling', {}))
            self.white_to_move = record.get('prev_white_to_move', self.white_to_move)
            # recompute check for side to move
            self.check = self.is_in_check(self.white_to_move)
            return

        # --- New / minimal format: apply inverse du delta ---
        # Sécurité: vérifier clés minimales
        required = {'from', 'to', 'moving_piece', 'prev_castling', 'prev_en_passant', 'prev_white_to_move'}
        if not required.issubset(record.keys()):
            # Format inattendu : essayer la restauration prudente (rappel au développeur)
            # pour éviter laisser l'état corrompu.
            raise RuntimeError("undo_move: historique dans un format inattendu: {}".format(record.keys()))

        from_sq = int(record['from'])
        to_sq = int(record['to'])
        from_mask = self.square_mask(from_sq)
        to_mask = self.square_mask(to_sq)

        moving_piece = record['moving_piece']          # pièce qui a bougé (ex: 'P' ou 'k')
        captured_piece = record.get('captured_piece') # None ou lettre pièce capturée
        captured_square = record.get('captured_square')  # case où la capture a eu lieu (utile pour en-passant)
        promotion = record.get('promotion')           # None ou 'Q','R',...

        # Si une promotion a eu lieu, la pièce sur 'to' est le promoted_piece (ex 'Q' ou 'q')
        if promotion:
            # déterminer le caractère de la pièce promue selon le trait précédent
            # record['prev_white_to_move'] contient le trait **avant** le coup (donc l'auteur du coup)
            # si prev_white_to_move == True => move was by White, promotion letter uppercase
            promoted_piece = promotion if record['prev_white_to_move'] else promotion.lower()
            # enlever la pièce promue de la case 'to'
            # (il est possible que le code d'origine ait mis la promotion différemment ; ce comportement est standard)
            self.bitboards[promoted_piece] &= ~to_mask
            # remettre le pion d'origine sur la case 'from'
            pawn_piece = 'P' if record['prev_white_to_move'] else 'p'
            self.bitboards[pawn_piece] |= from_mask
        else:
            # pièce normale : déplacer la pièce de 'to' vers 'from'
            # retirer de 'to'
            self.bitboards[moving_piece] &= ~to_mask
            # remettre sur 'from'
            self.bitboards[moving_piece] |= from_mask

        # Restaurer la pièce capturée s'il y en a (capture normale ou en-passant)
        if captured_piece:
            if captured_square is None:
                # cas improbable : utiliser 'to' comme case de capture
                cap_sq = to_sq
            else:
                cap_sq = int(captured_square)
            self.bitboards[captured_piece] |= self.square_mask(cap_sq)

        # Spécial : roque — si le coup était une roque, il faut remettre la tour à sa case d'origine.
        # On peut détecter la roque par l'éloignement du roi (de 2 cases horizontales).
        # Ici on répare au cas où move_piece avait déplacé la tour en conséquence.
        # (Si move_piece n'a pas ajusté la tour, cette étape est inoffensive.)
        if moving_piece in ('K', 'k') and abs(from_sq - to_sq) == 2:
            # cas standard:
            if moving_piece == 'K':  # blanc
                if to_sq == 6:   # O-O (e1 -> g1)
                    # rook h1 (7) -> f1 (5) a été déplacée ; on remet
                    self.bitboards['R'] &= ~self.square_mask(5)
                    self.bitboards['R'] |= self.square_mask(7)
                elif to_sq == 2: # O-O-O (e1 -> c1)
                    self.bitboards['R'] &= ~self.square_mask(3)
                    self.bitboards['R'] |= self.square_mask(0)
            else:  # 'k' noir
                if to_sq == 62:  # e8 -> g8
                    self.bitboards['r'] &= ~self.square_mask(61)
                    self.bitboards['r'] |= self.square_mask(63)
                elif to_sq == 58: # e8 -> c8
                    self.bitboards['r'] &= ~self.square_mask(59)
                    self.bitboards['r'] |= self.square_mask(56)

        # Restaurer les flags
        self.castling_rights = dict(record.get('prev_castling', self.castling_rights))
        self.en_passant_target = record.get('prev_en_passant', self.en_passant_target)
        self.white_to_move = bool(record.get('prev_white_to_move', self.white_to_move))

        # Recalculer l'état d'échec pour le côté à jouer (simple vérif)
        self.check = self.is_in_check(self.white_to_move)

        return
    
    def update_castling_rights(self, moving_piece: str, from_sq: int):
        """
        Met à jour les droits de roque après un déplacement ou une capture.
        """
        # Supprime les droits de roque si le roi bouge
        if moving_piece == 'K':
            self.castling_rights['K'] = False
            self.castling_rights['Q'] = False
        elif moving_piece == 'k':
            self.castling_rights['k'] = False
            self.castling_rights['q'] = False

        # Supprime les droits si une tour bouge depuis sa case d’origine
        elif moving_piece == 'R':
            if from_sq == 0:
                self.castling_rights['Q'] = False  # Tour a1
            elif from_sq == 7:
                self.castling_rights['K'] = False  # Tour h1
        elif moving_piece == 'r':
            if from_sq == 56:
                self.castling_rights['q'] = False  # Tour a8
            elif from_sq == 63:
                self.castling_rights['k'] = False  # Tour h8

        # Supprime les droits si une tour d’origine est capturée
        # (utile si update_castling_rights est appelée après la capture)
        if self.bitboards['R'] & self.square_mask(0) == 0:
            self.castling_rights['Q'] = False
        if self.bitboards['R'] & self.square_mask(7) == 0:
            self.castling_rights['K'] = False
        if self.bitboards['r'] & self.square_mask(56) == 0:
            self.castling_rights['q'] = False
        if self.bitboards['r'] & self.square_mask(63) == 0:
            self.castling_rights['k'] = False

    def update_en_passant(self, moving_piece: str, from_sq: int, to_sq: int):
        """
        Met à jour la cible en passant (ou la désactive).
        """
        self.en_passant_target = None  # par défaut : désactivé

        # Pion blanc avance de 2 cases
        if moving_piece == 'P' and from_sq // 8 == 1 and to_sq // 8 == 3:
            self.en_passant_target = from_sq + 8

        # Pion noir avance de 2 cases
        elif moving_piece == 'p' and from_sq // 8 == 6 and to_sq // 8 == 4:
            self.en_passant_target = from_sq - 8