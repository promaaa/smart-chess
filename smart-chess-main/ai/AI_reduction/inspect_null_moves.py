from match_two_ai import make_engine
from Chess import Chess

# create null engine
eng = make_engine('null', 1.0)
print('engine type', type(eng))
ch = Chess()
best = eng.get_best_move_with_time_limit(ch)
print('best (raw):', best)
try:
    print('best formatted:', eng._format_move(best))
except Exception as e:
    print('format error', e)

legal = eng._get_all_legal_moves(ch)
print('legal count', len(legal))
for i,m in enumerate(legal):
    try:
        fm = eng._format_move(m)
    except Exception:
        fm = str(m)
    print(i, m, '->', fm)

matches = [m for m in legal if isinstance(best, tuple) and isinstance(m, tuple) and int(m[0])==int(best[0]) and int(m[1])==int(best[1])]
print('numeric matches:', matches)

san_matches = [m for m in legal if eng._format_move(m) == eng._format_move(best)]
print('san matches:', san_matches)
