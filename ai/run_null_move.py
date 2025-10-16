#!/usr/bin/env python3
"""
Compare Null-move ON vs OFF on the same position for a fixed time budget.
Prints time, nodes, nps and null-move stats.
"""
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# Ensure optimized bitboards are applied if available
try:
    from optimized_chess import patch_chess_class_globally
    patch_chess_class_globally()
    print("✅ Optimisations bitboard appliquées (import-time)")
except Exception:
    print("⚠️  optimized_chess non disponible au moment de l'import")

from Chess import Chess
from fast_evaluator import SuperFastChessEvaluator
from Null_move_AI.null_move_engine import NullMovePruningEngine


def prepare_position():
    chess = Chess()
    # play a few moves to get a non-trivial midgame-ish position
    moves = [(12, 28, None), (52, 36, None), (6, 21, None), (57, 42, None)]
    for m in moves:
        chess.move_piece(m[0], m[1], m[2])
    return chess


def run_once(chess, time_budget, null_move_enabled=True):
    engine = NullMovePruningEngine(
        max_time=time_budget,
        max_depth=20,
        evaluator=SuperFastChessEvaluator(),
        tt_size=200000,
        null_move_enabled=null_move_enabled,
        null_move_R=2,
        null_move_min_depth=3
    )

    start = time.time()
    best = None
    try:
        best = engine.get_best_move_with_time_limit(chess)
    except Exception as e:
        print(f"Run error: {e}")
    elapsed = time.time() - start

    stats = {
        'enabled': null_move_enabled,
        'time': elapsed,
        'nodes': engine.nodes_evaluated,
        'nps': engine.nodes_evaluated / max(1e-6, elapsed),
        'null_attempts': getattr(engine, 'null_move_attempts', 0),
        'null_cutoffs': getattr(engine, 'null_move_cutoffs', 0),
        'null_failures': getattr(engine, 'null_move_failures', 0),
        'best_move': engine._format_move(best) if best else None
    }

    return stats


def compare(time_budget=30):
    chess = prepare_position()

    print(f"\n== Running comparison on same position — {time_budget}s runs ==\n")

    # Run with null-move enabled
    print("-- Null-move ENABLED --")
    stats_on = run_once(chess, time_budget, True)
    print(f"Time: {stats_on['time']:.2f}s, Nodes: {stats_on['nodes']:,}, NPS: {stats_on['nps']:.0f}")
    print(f"Null attempts: {stats_on['null_attempts']}, cutoffs: {stats_on['null_cutoffs']}, failures: {stats_on['null_failures']}")
    print(f"Best move: {stats_on['best_move']}\n")

    # Run with null-move disabled (start from same position)
    chess2 = prepare_position()
    print("-- Null-move DISABLED --")
    stats_off = run_once(chess2, time_budget, False)
    print(f"Time: {stats_off['time']:.2f}s, Nodes: {stats_off['nodes']:,}, NPS: {stats_off['nps']:.0f}")
    print(f"Best move: {stats_off['best_move']}\n")

    # Summary
    print("== SUMMARY ==")
    print(f"Enabled NPS: {stats_on['nps']:.0f}, Disabled NPS: {stats_off['nps']:.0f}")
    nodes_diff = stats_on['nodes'] - stats_off['nodes']
    nodes_pct = (nodes_diff / max(1, stats_off['nodes'])) * 100
    print(f"Nodes diff (on - off): {nodes_diff:,} ({nodes_pct:+.1f}%)")
    attempts = stats_on['null_attempts']
    if attempts:
        rate = stats_on['null_cutoffs'] / attempts * 100
        print(f"Null success rate: {rate:.1f}% ({stats_on['null_cutoffs']}/{attempts})")

    return stats_on, stats_off


if __name__ == '__main__':
    compare(time_budget=30)
