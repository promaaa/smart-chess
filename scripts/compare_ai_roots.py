import os, sys, filecmp
rootA = os.path.join(os.getcwd(),'ai')
rootB = os.path.join(os.getcwd(),'smart-chess-main','ai')
missing = []
identical = []
different = []
for dirpath,dirnames,filenames in os.walk(rootB):
    for fn in filenames:
        if not fn.endswith('.py'):
            continue
        rel = os.path.relpath(os.path.join(dirpath,fn), rootB)
        a_path = os.path.join(rootA, rel)
        b_path = os.path.join(rootB, rel)
        if not os.path.exists(a_path):
            missing.append(rel)
        else:
            try:
                if filecmp.cmp(a_path,b_path,shallow=False):
                    identical.append(rel)
                else:
                    different.append(rel)
            except Exception as e:
                different.append(rel)
print('MISSING:', len(missing))
for m in missing: print('  ',m)
print('\nIDENTICAL:', len(identical))
for i in identical[:50]: print('  ',i)
print('\nDIFFERENT:', len(different))
for d in different[:200]: print('  ',d)
