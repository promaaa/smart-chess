"""Quick regression test for incremental Zobrist updates.
Plays random pseudo-legal moves (no deep legality checks beyond existing Chess API) and
compares the incremental zobrist key maintained by optimized_chess.patch_chess_class_globally
against full compute_zobrist() after each move and undo.
"""
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimized_chess import patch_chess_class_globally, compute_zobrist
from Chess import Chess

# Apply global patch to use incremental zobrist
patch_chess_class_globally()

ch = Chess()
# compute initial full key
try:
    ch.zobrist_key = compute_zobrist(ch)
except Exception:
    ch.zobrist_key = None

# collect all pseudo-legal moves generator: we'll reuse get_all_moves per square

def list_moves(board):
    moves = []
    for sq in range(64):
        bb = board.square_mask(sq)
        for p, b in board.bitboards.items():
            if b & bb:
                # generate targets
                targets = board.get_all_moves(sq)
                temp = int(targets)
                while temp:
                    t = (temp & -temp).bit_length() - 1
                    moves.append((sq, t))
                    temp &= temp - 1
                break
    return moves

random.seed(0xC0FFEE)

# Try random moves and validate zobrist
for i in range(200):
    moves = list_moves(ch)
    if not moves:
        print('No moves available, stopping')
        break
    m = random.choice(moves)
    from_sq, to_sq = m
    # Try move
    try:
        # store pre-key
        prev_key = getattr(ch, 'zobrist_key', None)
        ch.move_piece(from_sq, to_sq)
    except Exception as e:
        # illegal move or other: skip
        # print('move failed', e)
        continue

    # compute full key and compare
    full = compute_zobrist(ch)
    inc = getattr(ch, 'zobrist_key', None)
    if full != inc:
        print(f'Mismatch after move {i}: full={full} inc={inc}')
        print('Move:', from_sq, to_sq)
        # try undo and check again
        ch.undo_move()
        full2 = compute_zobrist(ch)
        inc2 = getattr(ch, 'zobrist_key', None)
        print('After undo: full=', full2, 'inc=', inc2)
        raise SystemExit('Zobrist incremental mismatch')

    # Now undo and check restored matches
    ch.undo_move()
    full_after_undo = compute_zobrist(ch)
    inc_after_undo = getattr(ch, 'zobrist_key', None)
    if full_after_undo != inc_after_undo:
        print(f'Mismatch after undo {i}: full={full_after_undo} inc={inc_after_undo}')
        raise SystemExit('Zobrist mismatch after undo')

print('Zobrist regression test passed')
