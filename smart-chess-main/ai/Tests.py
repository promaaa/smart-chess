from Chess import Chess
import numpy as np

# Helper to convert file/rank to square index
def sq(file, rank):
    return (rank - 1) * 8 + (ord(file) - ord('a'))
    

# --- Test suite class ---
class ChessTests:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def assert_true(self, cond, msg):
        if cond:
            print("OK:", msg)
            self.passed += 1
        else:
            print("FAIL:", msg)
            self.failed += 1

    def test_castling_kingside(self):
        c = Chess()
        # Clear board and set minimal pieces for white kingside castling
        for k in list(c.bitboards.keys()):
            c.bitboards[k] = 0
        # place white king on e1 and rook on h1, no black pieces
        c.bitboards['K'] = c.square_mask(sq('e',1))
        c.bitboards['R'] = c.square_mask(sq('h',1))
        c.castling_rights = {'K': True, 'Q': False, 'k': False, 'q': False}
        c.white_to_move = True
        # ensure squares f1 and g1 are empty and not attacked (no black pieces present)
        # perform castling: king e1->g1 (4->6)
        try:
            c.move_piece(sq('e',1), sq('g',1))
            # verify king on g1 and rook on f1
            king_on_g1 = bool(c.bitboards['K'] & c.square_mask(sq('g',1)))
            rook_on_f1 = bool(c.bitboards['R'] & c.square_mask(sq('f',1)))
            self.assert_true(king_on_g1 and rook_on_f1, "White kingside castling moved king to g1 and rook to f1")
        except Exception as e:
            self.assert_true(False, f"Castling raised exception: {e}")

    def test_en_passant(self):
        c = Chess()
        # clear board
        for k in list(c.bitboards.keys()):
            c.bitboards[k] = 0
        # white pawn on d5, black pawn on e7
        c.bitboards['P'] = c.square_mask(sq('d',5))
        c.bitboards['p'] = c.square_mask(sq('e',7))
        c.white_to_move = False  # let black move first
        # black plays e7->e5 (double push)
        try:
            c.move_piece(sq('e',7), sq('e',5))
            # en_passant_target should be e6
            self.assert_true(c.en_passant_target == sq('e',6), "En passant target set to e6 after black double-push")
            # now white plays d5xe6 en passant (d5->e6)
            c.move_piece(sq('d',5), sq('e',6))
            # verify black pawn at e5 was removed
            pawn_removed = not bool(c.bitboards['p'] & c.square_mask(sq('e',5)))
            pawn_on_e6 = bool(c.bitboards['P'] & c.square_mask(sq('e',6)))
            self.assert_true(pawn_removed and pawn_on_e6, "En passant captured pawn removed from e5 and white pawn on e6")
        except Exception as e:
            self.assert_true(False, f"En passant sequence failed: {e}")

    def test_promotion(self):
        c = Chess()
        # clear board
        for k in list(c.bitboards.keys()):
            c.bitboards[k] = 0
        # white pawn on g7 ready to promote to g8
        c.bitboards['P'] = c.square_mask(sq('g',7))
        # ensure black has no pieces to interfere
        c.white_to_move = True
        try:
            c.move_piece(sq('g',7), sq('g',8))  # promotion, default to Queen
            # check that a white queen exists on g8
            has_queen_g8 = 'Q' in c.bitboards and bool(c.bitboards['Q'] & c.square_mask(sq('g',8)))
            self.assert_true(has_queen_g8, "Pawn promoted to Queen on g8 by default")
        except Exception as e:
            self.assert_true(False, f"Promotion failed: {e}")

    def test_check_and_undo(self):
        c = Chess()
        # clear board
        for k in list(c.bitboards.keys()):
            c.bitboards[k] = 0
        # place white king on e1, black rook on e8 (checking down the file)
        c.bitboards['K'] = c.square_mask(sq('e',1))
        c.bitboards['r'] = c.square_mask(sq('e',8))
        c.white_to_move = True
        # Is white in check? should be False initially (rook not attacking because pieces between? actually no pieces between)
        # rook at e8 attacks e1 along file -> white in check
        in_check = c.is_in_check(True)
        self.assert_true(in_check, "White is in check from black rook on e8")
        # Now try to make a move that doesn't block the check: move a non-related piece
        c.bitboards['R'] = c.square_mask(sq('a',3))  # irrelevant rook somewhere else
        try:
            # attempt illegal move that doesn't block and see that it's allowed but leaves king in check
            # moving rook a1->a2 (non-related) should be allowed; but we want to test illegal move detection,
            # so try moving the rook from a1 to a2 which doesn't affect check: should raise ValueError if it leaves king in check
            from_sq = sq('a',3)
            to_sq = sq('a',4)
            try:
                c.move_piece(from_sq, to_sq)
                self.assert_true(False, "Move that leaves king in check should have raised ValueError")
            except ValueError:
                self.assert_true(True, "Illegal move leaving king in check correctly rejected")
        except Exception as e:
            self.assert_true(False, f"Check/illegal move test raised unexpected exception: {e}")

        # Now make a legal blocking move: move rook from a3 to e3 to block (e3 = sq('e',3))
        try:
            c.move_piece(sq('a',3), sq('e',3))
            # after this move, white should no longer be in check
            self.assert_true(not c.is_in_check(True), "Blocking move removed check")
            # test undo: undo last move and verify we are back in check
            c.undo_move()
            self.assert_true(c.is_in_check(True), "Undo restored previous check state")
        except Exception as e:
            self.assert_true(False, f"Blocking/undo sequence failed: {e}")

    def run_all(self):
        print("Running Chess tests...\n")
        self.test_castling_kingside()
        print()
        self.test_en_passant()
        print()
        self.test_promotion()
        print()
        self.test_check_and_undo()
        print()
        print(f"Tests passed: {self.passed}, failed: {self.failed}")

# Run tests
tests = ChessTests()
tests.run_all()
