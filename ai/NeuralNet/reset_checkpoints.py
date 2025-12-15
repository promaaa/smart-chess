"""Utility to backup and remove model checkpoints and weight files.

Use this from a Colab notebook or locally to reinitialize checkpoints when
architecture changes cause incompatibilities.

Examples:
    # from Python
    from ai.NN import reset_checkpoints as rc
    rc.clear_checkpoints(paths=["/content/drive/MyDrive/smart_chess_drive/smart-chess/ai/checkpoints"], backup=True, delete=False)

    # from shell
    python ai/NN/reset_checkpoints.py --paths /content/drive/MyDrive/smart_chess_drive/smart-chess/ai/checkpoints --force

This script by default moves files to a timestamped backup folder. Use
--delete to permanently remove them (use with caution).
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from typing import List


def _iter_files(paths: List[str]):
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    yield os.path.join(root, f)
        elif os.path.isfile(p):
            yield p


def clear_checkpoints(paths: List[str], backup: bool = True, delete: bool = False, dry_run: bool = False):
    """Backup then remove checkpoint and weight files.

    - paths: list of files or directories to process
    - backup: move files to a backup folder instead of deleting (recommended)
    - delete: permanently delete files (overrides backup)
    - dry_run: only print actions without performing them
    """
    if not paths:
        print("No paths provided to clear_checkpoints(). Nothing to do.")
        return

    files = list(_iter_files(paths))
    if not files:
        print("No files found in the specified paths.")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = None
    if backup and not delete:
        backup_dir = os.path.join(os.path.expanduser("~"), f"checkpoint_backups_{ts}")
        print(f"Backing up {len(files)} files to: {backup_dir}")
        if not dry_run:
            os.makedirs(backup_dir, exist_ok=True)

    for f in files:
        try:
            if backup and not delete:
                dest = os.path.join(backup_dir, os.path.basename(f))
                print(f"Moving {f} -> {dest}")
                if not dry_run:
                    shutil.move(f, dest)
            else:
                print(f"Deleting {f}")
                if not dry_run:
                    os.remove(f)
        except Exception as e:
            print(f"Failed to handle {f}: {e}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Backup and/or remove checkpoint and weight files.")
    parser.add_argument("--paths", nargs='+', help="Files or directories to clear", required=True)
    parser.add_argument("--delete", action='store_true', help="Permanently delete files instead of backing up")
    parser.add_argument("--dry-run", action='store_true', help="Only print actions, don't perform them")
    parser.add_argument("--yes", action='store_true', help="Skip confirmation prompt")

    args = parser.parse_args()

    if not args.yes:
        print("This will remove or move checkpoint files. Make sure you are targeting the correct paths:")
        for p in args.paths:
            print(f"  - {p}")
        resp = input("Proceed? [y/N]: ")
        if resp.lower() not in ('y', 'yes'):
            print("Aborted.")
            return

    clear_checkpoints(paths=args.paths, backup=not args.delete, delete=args.delete, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
