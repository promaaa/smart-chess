import sys
from ai.Chess import Chess
import ai.optimized_chess as oc

print('module optimized_chess loaded:', oc.__file__)
print('Zobrist pieces table initialized?', oc._ZOBRIST_PIECES is not None)
print('Zobrist side present?', hasattr(oc, '_ZOBRIST_SIDE'))

c = Chess()
print('Chess instance created; has zobrist?', hasattr(c, 'zobrist_key'))
# Patch this instance (should add zobrist_key and wrappers)
try:
    oc.patch_chess_class(c)
    patched = True
except Exception as e:
    print('patch_chess_class failed:', e)
    patched = False

print('Patched:', patched)
print('zobrist_key after patch:', getattr(c, 'zobrist_key', None))

# Compute via compute_zobrist
try:
    computed = oc.compute_zobrist(c)
    print('compute_zobrist() ->', computed)
except Exception as e:
    print('compute_zobrist failed:', e)
    computed = None

# If move_piece exists, make a simple pawn move from 12->28 (e2->e4)
if hasattr(c, 'move_piece'):
    try:
        before = getattr(c, 'zobrist_key', None)
        c.move_piece(12, 28)
        after = getattr(c, 'zobrist_key', None)
        print('move applied; history last:', c.history[-1] if c.history else None)
        print('zobrist before:', before)
        print('zobrist after :', after)
        print('computed equals initial?', computed == before)
    except Exception as e:
        print('move_piece failed:', e)
else:
    print('no move_piece on Chess')

print('\nDone')
