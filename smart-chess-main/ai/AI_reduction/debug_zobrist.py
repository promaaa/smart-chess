import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimized_chess import patch_chess_class_globally, compute_zobrist
from Chess import Chess

patch_chess_class_globally()
ch = Chess()
try:
    ch.zobrist_key = compute_zobrist(ch)
except Exception:
    ch.zobrist_key = None
print('initial full key:', ch.zobrist_key)

# list moves
moves = []
for sq in range(64):
    bb = ch.square_mask(sq)
    for p,b in ch.bitboards.items():
        if b & bb:
            targets = ch.get_all_moves(sq)
            temp = int(targets)
            while temp:
                t = (temp & -temp).bit_length() - 1
                moves.append((sq,t))
                temp &= temp - 1
            break

print('total moves:', len(moves))
# choose move 11->27 if available
if (11,27) not in moves:
    print('move (11,27) not available; printing some moves and exiting')
    print(moves[:20])
    sys.exit(0)

from_sq, to_sq = 11, 27
print('About to play move', from_sq, to_sq)
prev_key = getattr(ch, 'zobrist_key', None)
print('prev_key backup:', prev_key)
try:
    ch.move_piece(from_sq, to_sq)
    print('move applied')
except Exception as e:
    print('move failed', e)
    sys.exit(1)

print('history last entry:', ch.history[-1])
print('ch.zobrist_key (inc):', ch.zobrist_key)
full = compute_zobrist(ch)
print('compute_zobrist (full):', full)

# show some zobrist tables for moving piece and squares
from optimized_chess import _ZOBRIST_PIECES, _ZOBRIST_SIDE, _ZOBRIST_CASTLING, _ZOBRIST_EP_FILE
print('ZOBRIST_SIDE:', _ZOBRIST_SIDE)
print('moving piece char in history:', ch.history[-1].get('moving_piece'))
print('from,to:', ch.history[-1].get('from'), ch.history[-1].get('to'))

# Undo and show restored
ch.undo_move()
print('after undo, ch.zobrist_key:', ch.zobrist_key)
print('compute full after undo:', compute_zobrist(ch))
print('done')
