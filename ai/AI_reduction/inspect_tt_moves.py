from match_two_ai import make_engine
from Chess import Chess

e = make_engine('tt', 1.0)
print('engine', type(e))
ch = Chess()
best = e.get_best_move_with_time_limit(ch)
print('best_move (engine):', best)
legal = e._get_all_legal_moves(ch)
print('legal_count:', len(legal))
for i, m in enumerate(legal[:40]):
    print(i, m)

found = any(isinstance(m, tuple) and int(m[0]) == int(best[0]) and int(m[1]) == int(best[1]) for m in legal if isinstance(best, tuple))
print('found_match?', found)
