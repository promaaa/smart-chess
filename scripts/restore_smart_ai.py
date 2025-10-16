import os
import shutil
root = os.path.join(os.getcwd(), 'smart-chess-main', 'ai')
if not os.path.isdir(root):
    print('source dir not found:', root)
    raise SystemExit(1)
count_copied = 0
count_skipped = 0
for dirpath, dirnames, filenames in os.walk(root):
    for fn in filenames:
        if not fn.endswith('.py'):
            continue
        src = os.path.join(dirpath, fn)
        rel = os.path.relpath(src, root)
        dest = os.path.join(os.getcwd(), 'ai', rel)
        dest_dir = os.path.dirname(dest)
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(dest):
            print('SKIPPED:', rel)
            count_skipped += 1
        else:
            shutil.copy2(src, dest)
            print('COPIED :', rel)
            count_copied += 1
print('\nDone. Copied:', count_copied, 'Skipped:', count_skipped)
