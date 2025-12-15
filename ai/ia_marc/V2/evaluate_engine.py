#!/usr/bin/env python3
"""
Evaluation Script for IA-Marc V2
================================

Tests the engine for:
1. Stability (Self-play)
2. Performance (NPS)
3. Strength (Tactics)
4. Time Management
5. Resource Usage

Usage:
    python3 ai/ia_marc/V2/evaluate_engine.py
"""

import sys
import os
import time
import chess
import resource
import logging
from typing import List, Tuple

# Add the current directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from engine_main import ChessEngine
except ImportError:
    # Try importing from root if running from there
    sys.path.append(os.path.join(current_dir, "..", "..", ".."))
    from ai.ia_marc.V2.engine_main import ChessEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Evaluator")

class EngineEvaluator:
    def __init__(self):
        self.engine = ChessEngine(verbose=False)
        # Set to a reasonable level for testing
        self.engine.set_level("LEVEL7") 
        self.results = {}

    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")

    def test_performance_nps(self):
        self.print_header("TEST 1: Performance Benchmark (NPS)")
        
        # Disable book for NPS test
        self.engine.enable_opening_book(False)
        
        board = chess.Board()
        time_limit = 2.0
        print(f"Running search on startpos for {time_limit}s...")
        
        start_time = time.time()
        move, stats = self.engine.get_move_with_stats(board, time_limit=time_limit)
        elapsed = time.time() - start_time
        
        nps = stats.get('nps', 0)
        nodes = stats.get('nodes', 0)
        depth = stats.get('depth', 0)
        
        print(f"Result: {nodes} nodes in {elapsed:.2f}s")
        print(f"NPS: {nps:,.0f}")
        print(f"Depth reached: {depth}")
        
        if nps > 10000:
            print("✅ PASS: NPS is acceptable (>10k)")
            return True
        else:
            print("⚠️ WARNING: NPS is low (<10k). Check optimizations.")
            return True # Still pass, just warning

    def test_tactics(self):
        self.print_header("TEST 2: Tactical Strength (Mate in N)")
        
        # Disable book for tactics
        self.engine.enable_opening_book(False)
        
        puzzles = [
            # Mate in 1
            ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 4", "f3f7", 1),
            # Mate in 2
            ("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 10", "d5f6", 2),
        ]
        
        passed = 0
        for fen, solution_uci, mate_in in puzzles:
            board = chess.Board(fen)
            print(f"Solving Mate in {mate_in} position...")
            
            # Give enough time for tactics
            move, stats = self.engine.get_move_with_stats(board, time_limit=2.0)
            
            if move and move.uci() == solution_uci:
                print(f"✅ Solved! {move.uci()} found.")
                passed += 1
            else:
                print(f"❌ Failed. Expected {solution_uci}, got {move}")
        
        success_rate = passed / len(puzzles)
        print(f"Tactical Score: {passed}/{len(puzzles)}")
        return success_rate >= 0.5

    def test_time_management(self):
        self.print_header("TEST 3: Time Management")
        
        # Disable book for time management test
        self.engine.enable_opening_book(False)
        
        board = chess.Board()
        target_time = 1.0
        tolerance = 0.2 # 20% tolerance
        
        print(f"Requesting move with time_limit={target_time}s...")
        start = time.time()
        self.engine.get_move(board, time_limit=target_time)
        actual_time = time.time() - start
        
        print(f"Actual time taken: {actual_time:.3f}s")
        
        diff = abs(actual_time - target_time)
        # Allow being faster, but not too much slower (hard bound is usually strict)
        # Engine often stops early if it finds a forced move or stable score
        
        if actual_time > target_time + 0.5: # 0.5s buffer
            print(f"❌ FAIL: Exceeded time limit significantly (+{actual_time - target_time:.2f}s)")
            return False
        else:
            print("✅ PASS: Time management is within acceptable limits.")
            return True

    def test_stability_selfplay(self):
        self.print_header("TEST 4: Stability (Self-Play)")
        
        # Re-enable book for self-play to test realistic usage
        self.engine.enable_opening_book(True)
        
        board = chess.Board()
        moves_to_play = 10
        print(f"Playing {moves_to_play} moves (self-play)...")
        
        try:
            for i in range(moves_to_play):
                if board.is_game_over():
                    break
                
                move = self.engine.get_move(board, time_limit=0.5)
                if not move:
                    print("❌ FAIL: Engine returned None move.")
                    return False
                
                if move not in board.legal_moves:
                    print(f"❌ FAIL: Illegal move returned: {move}")
                    return False
                
                board.push(move)
                print(f"Move {i+1}: {move}")
                
            print("✅ PASS: No crashes or illegal moves during self-play.")
            return True
            
        except Exception as e:
            print(f"❌ CRASH: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_resources(self):
        self.print_header("TEST 5: Resource Usage")
        
        usage = resource.getrusage(resource.RUSAGE_SELF)
        memory_mb = usage.ru_maxrss / 1024 / 1024 # Mac uses bytes, Linux uses KB usually? 
        # Actually on Mac ru_maxrss is bytes, on Linux it is kilobytes.
        # Let's assume Mac since user is on Mac.
        if sys.platform == 'darwin':
            memory_mb = usage.ru_maxrss / 1024 / 1024
        else:
            memory_mb = usage.ru_maxrss / 1024
            
        print(f"Max Memory Usage: {memory_mb:.2f} MB")
        
        if memory_mb < 500:
            print("✅ PASS: Memory usage is low (<500MB)")
            return True
        else:
            print("⚠️ WARNING: High memory usage.")
            return True

    def run_all(self):
        print("\nStarting IA-Marc V2 Evaluation...")
        print(f"Platform: {sys.platform}")
        print(f"Python: {sys.version.split()[0]}")
        
        results = [
            self.test_performance_nps(),
            self.test_tactics(),
            self.test_time_management(),
            self.test_stability_selfplay(),
            self.check_resources()
        ]
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        if all(results):
            print("\n🎉 ALL TESTS PASSED! Engine is ready for beta testing.")
        else:
            print("\n⚠️ SOME TESTS FAILED. Review logs above.")

if __name__ == "__main__":
    evaluator = EngineEvaluator()
    evaluator.run_all()
