import sys, importlib.util
sys.path.insert(0, 'ai')
# Import Chess module from ai/Chess.py as Chess
spec = importlib.util.spec_from_file_location('Chess', 'ai/Chess.py')
Chess_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Chess_mod)
Chess = getattr(Chess_mod, 'Chess')
# Import optimized_chess
spec2 = importlib.util.spec_from_file_location('optimized_chess', 'ai/optimized_chess.py')
opt = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(opt)
print('optimized_chess loaded from', opt.__file__)
print('Zobrist table present?', opt._ZOBRIST_PIECES is not None)

c = Chess()
print('Chess instance created; has zobrist?', hasattr(c,'zobrist_key'))
opt.patch_chess_class(c)
print('after patch zobrist_key=', getattr(c,'zobrist_key'))
print('compute_zobrist ->', opt.compute_zobrist(c))
before = getattr(c,'zobrist_key')
# apply move
c.move_piece(12,28)
after = getattr(c,'zobrist_key')
print('history last:', c.history[-1])
print('before==compute?', before == opt.compute_zobrist(c))
print('before -> after', before, '->', after)
