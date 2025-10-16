import os,sys,difflib
base = os.getcwd()
paths = [
    ('ai/optimized_chess.py', 'smart-chess-main/ai/optimized_chess.py'),
    ('ai/Chess.py', 'smart-chess-main/ai/Chess.py')
]
for a,b in paths:
    a_path = os.path.join(base,a)
    b_path = os.path.join(base,b)
    print('\n## Comparing', a, 'vs', b)
    a_exists = os.path.exists(a_path)
    b_exists = os.path.exists(b_path)
    print('  exists:', a, a_exists, b, b_exists)
    if a_exists and b_exists:
        with open(a_path,'r',encoding='utf-8',errors='ignore') as fa:
            A = fa.read().splitlines()
        with open(b_path,'r',encoding='utf-8',errors='ignore') as fb:
            B = fb.read().splitlines()
        if A==B:
            print('  -> IDENTICAL')
        else:
            print('  -> DIFFER')
            diff = list(difflib.unified_diff(A,B,fromfile=a, tofile=b, lineterm=''))
            for line in diff[:200]:
                print(line)
            if len(diff)>200:
                print('... diff truncated,', len(diff), 'lines total')
    elif a_exists:
        print('  -> Only', a, 'exists')
    elif b_exists:
        print('  -> Only', b, 'exists')
    else:
        print('  -> Neither exists')
