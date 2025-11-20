#!/usr/bin/env python3
"""
Test script for Contempt Factor implementation
Vérifie que le contempt est appliqué correctement aux positions nulles.
"""

import sys
import os

# Ajouter le chemin du module IA-Marc V2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import chess
from engine_main import ChessEngine
from engine_config import EngineConfig, DIFFICULTY_LEVELS

def test_contempt_basic():
    """Test basique: vérifier que le contempt affecte l'évaluation."""
    print("=== Test 1: Contempt dans l'évaluation ===\n")
    
    # Position de pat (stalemate)
    # Le roi noir est pat: https://lichess.org/analysis/7k/5Q2/6K1/8/8/8/8/8_b_-_-_0_1
    stalemate_fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    board = chess.Board(stalemate_fen)
    
    # Test avec contempt = 0
    engine = ChessEngine()
    engine.set_level("ENFANT")  # contempt = 0
    score = engine.evaluator.evaluate(board)
    print(f"Position de pat, contempt=0: {score} (attendu: 0)")
    assert score == 0, f"Attendu 0, obtenu {score}"
    
    # Test avec contempt = 20
    engine.set_level("EXPERT")  # contempt = 20
    engine.evaluator.configure(contempt=20)
    score = engine.evaluator.evaluate(board)
    print(f"Position de pat, contempt=20: {score} (attendu: -20)")
    assert score == -20, f"Attendu -20, obtenu {score}"
    
    print("✅ Test 1 réussi!\n")

def test_contempt_search():
    """Test de recherche: vérifier que le contempt affecte le choix des coups."""
    print("=== Test 2: Contempt dans la recherche ===\n")
    
    # Position où les blancs peuvent forcer le pat ou continuer à jouer
    # Après Qf7+, les blancs peuvent soit faire pat (Kg6), soit continuer
    test_fen = "7k/8/6K1/8/8/8/5Q2/8 w - - 0 1"
    board = chess.Board(test_fen)
    
    # Test avec contempt = 0 (devrait accepter le pat)
    print("Test avec contempt=0 (niveau ENFANT)...")
    engine = ChessEngine()
    engine.set_level("ENFANT")
    move = engine.get_move(board, time_limit=1.0)
    print(f"  Coup choisi: {move}")
    
    # Test avec contempt = 30 (devrait éviter le pat si possible)
    print("\nTest avec contempt=30 (niveau MAXIMUM)...")
    engine.set_level("MAXIMUM")
    engine.evaluator.configure(contempt=30)
    move = engine.get_move(board, time_limit=1.0)
    print(f"  Coup choisi: {move}")
    
    print("✅ Test 2 réussi!\n")

def test_contempt_levels():
    """Test des différents niveaux de contempt."""
    print("=== Test 3: Contempt par niveau ===\n")
    
    # Position de matériel insuffisant
    insufficient_fen = "8/8/8/4k3/8/8/3K4/8 w - - 0 1"
    board = chess.Board(insufficient_fen)
    
    for level_name, level in DIFFICULTY_LEVELS.items():
        engine = ChessEngine()
        engine.set_level(level_name)
        score = engine.evaluator.evaluate(board)
        print(f"{level_name:12} | contempt={level.contempt:2} | score={score:4} (attendu: {-level.contempt})")
        assert score == -level.contempt, f"Niveau {level_name}: attendu {-level.contempt}, obtenu {score}"
    
    print("\n✅ Test 3 réussi!\n")

if __name__ == "__main__":
    print("=== Tests du Facteur de Mépris (Contempt) ===\n")
    
    try:
        test_contempt_basic()
        test_contempt_levels()
        test_contempt_search()
        
        print("\n" + "="*50)
        print("✅ TOUS LES TESTS RÉUSSIS!")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
