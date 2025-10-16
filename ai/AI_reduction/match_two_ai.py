#!/usr/bin/env python3
"""
Play a match between two AI engines, giving each side a fixed time per move.
Prints the depth reached for each move.

Usage: run from ai/AI_reduction folder or from repo root with python -m ai.AI_reduction.match_two_ai
"""
import time
import sys
import os
from typing import Tuple
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Ensure optimized bitboards if available
try:
    from optimized_chess import patch_chess_class_globally
    patch_chess_class_globally()
except Exception:
    pass

from Chess import Chess
from fast_evaluator import FastChessEvaluator, SuperFastChessEvaluator
# Engines
try:
    from Null_move_AI.null_move_engine import NullMovePruningEngine
except Exception:
    NullMovePruningEngine = None
try:
    from Old_AI.iterative_deepening_engine_TT import IterativeDeepeningAlphaBeta
except Exception:
    IterativeDeepeningAlphaBeta = None


def make_engine(name: str, time_per_move: float):
    """Factory to create engines by name."""
    if name == 'null' and NullMovePruningEngine is not None:
        return NullMovePruningEngine(max_time=time_per_move, max_depth=20, evaluator=SuperFastChessEvaluator(), tt_size=200000)
    if name == 'tt' and IterativeDeepeningAlphaBeta is not None:
        return IterativeDeepeningAlphaBeta(max_time=time_per_move, max_depth=20, evaluator=FastChessEvaluator(), tt_size=200000)
    # fallback: use NullMove if available
    if NullMovePruningEngine is not None:
        return NullMovePruningEngine(max_time=time_per_move, max_depth=20, evaluator=SuperFastChessEvaluator(), tt_size=200000)
    raise RuntimeError('No suitable engine available')


def move_to_san(chess, move_from, move_to):
    # Very small helper: return from-to like e2e4
    files = 'abcdefgh'
    def sq_to_str(sq):
        return files[sq % 8] + str(sq // 8 + 1)
    return f"{sq_to_str(move_from)}{sq_to_str(move_to)}"


def play_match(engine_white_name='null', engine_black_name='tt', time_per_move=20.0, max_moves=200, time_pattern=None):
    chess = Chess()
    engines = {
        True: make_engine(engine_white_name, time_per_move),
        False: make_engine(engine_black_name, time_per_move)
    }

    move_history = []
    side_white = True
    move_number = 1

    print(f"Starting match: White={engine_white_name} vs Black={engine_black_name}, {time_per_move}s per move")

    # time_pattern is a repeating sequence of per-ply times (seconds). Default: [45,10,10]
    if time_pattern is None:
        time_pattern = [45.0, 10.0, 10.0]

    # We'll apply the pattern per full move: both white and black plies of the same full move
    # use the same time budget. move_number starts at 1 and is incremented after Black's move.
    while move_number <= max_moves:
        engine = engines[side_white]
        # Provide a fresh copy of position (engines are allowed to read position but should not mutate it)
        # We'll call engine.get_best_move_with_time_limit(chess) or similar API. Different engines may use different method names.
        # choose time for this full move from repeating pattern
        chosen_time = float(time_pattern[(move_number - 1) % len(time_pattern)])
        # set engine max_time attribute if present so engine uses desired time
        try:
            engine.max_time = chosen_time
        except Exception:
            pass

        start = time.time()
        best_move = None
        depth_reached = None
        try:
            # Run search on a deepcopy so engines that mutate the board (and may leave it mutated on timeout)
            # do not affect the real game state. Many engines implemented get_best_move_with_time_limit
            # returning a move (from,to) tuple and exposing last_search_depth or nodes.
            chess_copy = copy.deepcopy(chess)
            best_move = engine.get_best_move_with_time_limit(chess_copy)
            # Try to read depth info from engine (common attributes used in this repo)
            depth_reached = getattr(engine, 'last_depth', None)
            if depth_reached is None:
                depth_reached = getattr(engine, 'depth_reached', None)
            if depth_reached is None:
                depth_reached = getattr(engine, 'current_depth', None)
        except Exception as e:
            print(f"Engine error: {e}")
            break
        elapsed = time.time() - start

        # Format move
        if best_move is None:
            print("No move returned by engine; aborting")
            break

        # Normalize engine return to (from,to,promotion) and validate
        def piece_on_square(chess_obj, sq):
            mask = chess_obj.square_mask(int(sq))
            for p, bb in chess_obj.bitboards.items():
                if bb & mask:
                    return True
            return False

        mv = None
        promo = None

        # Case 1: tuple-like (from, to, [promotion])
        if isinstance(best_move, tuple) and len(best_move) >= 2:
            try:
                from_sq = int(best_move[0])
                to_sq = int(best_move[1])
                promo = best_move[2] if len(best_move) >= 3 else None
                # Accept tuple moves as-is; we'll attempt to apply and recover if needed
                mv = (from_sq, to_sq, promo)
            except Exception:
                mv = None

        # Case 2: object with attributes
        if mv is None and hasattr(best_move, 'from_sq') and hasattr(best_move, 'to_sq'):
            try:
                from_sq = int(best_move.from_sq)
                to_sq = int(best_move.to_sq)
                promo = getattr(best_move, 'promotion', None)
                if piece_on_square(chess, from_sq):
                    mv = (from_sq, to_sq, promo)
            except Exception:
                mv = None

        # Case 3: formatted move string from engine._format_move or direct string
        if mv is None:
            formatted = None
            if hasattr(engine, '_format_move'):
                try:
                    formatted = engine._format_move(best_move)
                except Exception:
                    formatted = None
            if formatted is None and isinstance(best_move, str):
                formatted = best_move

            if formatted:
                # expected format like e2e4 or e7e8Q
                try:
                    from_alg = formatted[0:2]
                    to_alg = formatted[2:4]
                    promo_ch = formatted[4] if len(formatted) > 4 else None
                    file_to_index = lambda c: ord(c) - ord('a')
                    from_sq = file_to_index(from_alg[0]) + (int(from_alg[1]) - 1) * 8
                    to_sq = file_to_index(to_alg[0]) + (int(to_alg[1]) - 1) * 8
                    promo = promo_ch if promo_ch else None
                    # set mv even if piece_on_square check fails; attempt to apply later
                    mv = (from_sq, to_sq, promo)
                except Exception:
                    mv = None

        # Case 4: try to find matching legal move by formatting each legal move
        if mv is None:
            try:
                legal = engine._get_all_legal_moves(chess)
            except Exception:
                legal = []
            found = None
            for cand in legal:
                try:
                    if hasattr(engine, '_format_move') and formatted:
                        if engine._format_move(cand) == formatted:
                            found = cand
                            break
                    elif isinstance(cand, tuple) and len(cand) >= 2 and mv is not None:
                        # match by from/to squares
                        if int(cand[0]) == int(mv[0]) and int(cand[1]) == int(mv[1]):
                            found = cand
                            break
                except Exception:
                    continue
            if found:
                mv = (int(found[0]), int(found[1]), found[2] if len(found) >= 3 else None)

        if mv is None:
            print(f"Unrecognized or invalid move from engine: {best_move} (formatted={formatted})")
            break

        # Apply move on chess: prefer to find the corresponding legal move and apply it with promotion
        applied = False
        try:
            # Get legal moves from the engine (or fallback to chess._get_all_legal_moves if engine doesn't provide)
            try:
                legal = engine._get_all_legal_moves(chess)
            except Exception:
                # Some engines expose a different API; try to call chess.generate_legal_moves or similar
                try:
                    legal = chess.generate_legal_moves()
                except Exception:
                    legal = []

            # Try to match the canonical tuple mv to one of the legal moves
            matched = None
            # prepare target SAN like e2e4 to match against candidate moves
            try:
                target_san = move_to_san(chess, int(mv[0]), int(mv[1]))
            except Exception:
                target_san = None

            # utility to extract from/to/promo from a candidate move using several heuristics
            def extract_move_parts(c):
                # tuple-like
                if isinstance(c, tuple) and len(c) >= 2:
                    return (int(c[0]), int(c[1]), (c[2] if len(c) >= 3 else None))
                # try common attribute names
                for a_from in ('from_sq', 'from_square', 'from', 'src', 'fr'):
                    if hasattr(c, a_from):
                        for a_to in ('to_sq', 'to_square', 'to', 'dst', 'to_sq'):
                            if hasattr(c, a_to):
                                try:
                                    return (int(getattr(c, a_from)), int(getattr(c, a_to)), getattr(c, 'promotion', None))
                                except Exception:
                                    break
                # no standard shape
                return (None, None, None)

            for cand in legal:
                try:
                    cand_from, cand_to, cand_prom = extract_move_parts(cand)
                    # if we could extract numeric squares, compare directly
                    if cand_from is not None and cand_to is not None:
                        if int(cand_from) == int(mv[0]) and int(cand_to) == int(mv[1]):
                            # promotion check (if engine specified promotion)
                            if mv[2] is None or cand_prom is None or str(mv[2]) == str(cand_prom):
                                matched = cand
                                break
                    # else, try SAN/formatted comparison as a fallback
                    if target_san is not None:
                        # try engine formatting first
                        try:
                            cand_str = engine._format_move(cand) if hasattr(engine, '_format_move') else None
                        except Exception:
                            cand_str = None
                        if cand_str is None and cand_from is not None and cand_to is not None:
                            cand_str = move_to_san(chess, int(cand_from), int(cand_to))
                        if cand_str == target_san:
                            matched = cand
                            break
                except Exception:
                    continue

            if matched is not None:
                # apply matched legal move using its promotion if provided
                try:
                    if isinstance(matched, tuple):
                        prom = matched[2] if len(matched) >= 3 else None
                        chess.move_piece(matched[0], matched[1], promotion=prom)
                        mv = (matched[0], matched[1], prom)
                    else:
                        prom = getattr(matched, 'promotion', None)
                        chess.move_piece(int(getattr(matched, 'from_sq')), int(getattr(matched, 'to_sq')), promotion=prom)
                        mv = (int(getattr(matched, 'from_sq')), int(getattr(matched, 'to_sq')), prom)
                    applied = True
                except Exception as e:
                    print(f"Move application error when applying matched legal move: {e}")
                    applied = False
                chess.print_board()
            else:
                # No matching legal move found: fall back to formatted matching
                raise ValueError("No matching legal move found for canonical tuple")

        except Exception:
            # Recovery: try to match by engine's formatted string or by SAN/our formatting
            formatted = None
            try:
                formatted = engine._format_move(best_move)
            except Exception:
                formatted = None

            if formatted:
                # search legal moves by formatted representation
                try:
                    legal = engine._get_all_legal_moves(chess)
                except Exception:
                    legal = []
                found = None
                for m in legal:
                    try:
                        if hasattr(engine, '_format_move') and engine._format_move(m) == formatted:
                            found = m
                            break
                    except Exception:
                        continue

                if found is not None:
                    try:
                        if isinstance(found, tuple):
                            chess.move_piece(found[0], found[1], promotion=(found[2] if len(found) >= 3 else None))
                            mv = (found[0], found[1], (found[2] if len(found) >= 3 else None))
                        else:
                            chess.move_piece(int(getattr(found, 'from_sq')), int(getattr(found, 'to_sq')), promotion=getattr(found, 'promotion', None))
                            mv = (int(getattr(found, 'from_sq')), int(getattr(found, 'to_sq')), getattr(found, 'promotion', None))
                        applied = True
                    except Exception as e2:
                        print(f"Move application error after matching legal moves: {e2}")
                else:
                    # Try our own SAN-style matching
                    try:
                        legal = engine._get_all_legal_moves(chess)
                    except Exception:
                        legal = []
                    found = None
                    target = formatted
                    for m in legal:
                        try:
                            cand_str = move_to_san(chess, int(m[0]), int(m[1]))
                            if target is not None and cand_str == target:
                                found = m
                                break
                            # numeric match fallback
                            if isinstance(best_move, tuple) and int(m[0]) == int(best_move[0]) and int(m[1]) == int(best_move[1]):
                                found = m
                                break
                        except Exception:
                            continue

                    if found is not None:
                        chess.move_piece(found[0], found[1], promotion=(found[2] if len(found) >= 3 else None))
                        applied = True
                        mv = (found[0], found[1], (found[2] if len(found) >= 3 else None))
                        try:
                            print(f"Applied matched legal move: {move_to_san(chess, found[0], found[1])}")
                        except Exception:
                            pass
                    else:
                        print("Could not find a matching legal move for engine's best move. Aborting match to avoid playing a random move.")
                        break
            else:
                print(f"Move application error or no formatted move available for recovery. Best_move={best_move}")

            if not applied:
                break

        san = move_to_san(chess, mv[0], mv[1])
        side = 'White' if side_white else 'Black'
        depth_str = f"depth={depth_reached}" if depth_reached is not None else "depth=?"
        print(f"{move_number}. {side} {san}  ({elapsed:.2f}s, {depth_str})")

        move_history.append((move_number, side, san, elapsed, depth_reached))

        # swap side
        if not side_white:
            move_number += 1
        side_white = not side_white
    # move_number is incremented after Black's ply (below), so both plies of the same
    # full move use the same chosen_time computed above.

    print("Match finished. Moves:")
    for rec in move_history:
        print(rec)

    return move_history


if __name__ == '__main__':
    # defaults: null (white) vs tt (black), 20s per move
    play_match('null', 'tt', time_per_move=20.0, max_moves=200)
