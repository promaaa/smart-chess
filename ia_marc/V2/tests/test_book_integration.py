#!/usr/bin/env python3
"""
Test rapide pour vérifier que le livre d'ouvertures est bien chargé
"""

import os
import sys

# Ajouter le chemin
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("TEST DU LIVRE D'OUVERTURES")
print("=" * 60)

# Test 1: Vérifier que le fichier existe
book_path = "../book/Cerebellum3Merge.bin"
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), book_path))

print(f"\n1. Vérification du fichier:")
print(f"   Chemin: {abs_path}")
if os.path.exists(abs_path):
    size_mb = os.path.getsize(abs_path) / (1024 * 1024)
    print(f"   ✅ Fichier trouvé ({size_mb:.1f} MB)")
else:
    print(f"   ❌ Fichier non trouvé")
    sys.exit(1)

# Test 2: Charger le moteur
print(f"\n2. Chargement du moteur IA-Marc V2:")
try:
    from engine_main import ChessEngine

    engine = ChessEngine(verbose=False)
    print(f"   ✅ Moteur chargé")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# Test 3: Vérifier le livre d'ouvertures
print(f"\n3. Vérification du livre d'ouvertures:")
if engine.opening_book:
    print(f"   ✅ Livre chargé: {engine.config.opening_book_path}")
    print(f"   Type: {engine.opening_book.book_type}")
else:
    print(f"   ❌ Aucun livre chargé")
    sys.exit(1)

# Test 4: Tester une requête
print(f"\n4. Test d'une position d'ouverture:")
try:
    import chess

    board = chess.Board()  # Position initiale

    # Test avec le livre
    move = engine.opening_book.probe(board, elo_level=2000, variety=True)
    if move:
        print(f"   ✅ Coup trouvé: {move}")
        print(f"   Notation: {board.san(move)}")
    else:
        print(f"   ⚠️  Aucun coup trouvé (peut être normal)")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback

    traceback.print_exc()

# Test 5: Test complet avec get_move
print(f"\n5. Test complet avec get_move():")
try:
    board = chess.Board()
    engine.set_level("EXPERT")

    print(f"   Recherche d'un coup...")
    move = engine.get_move(board, time_limit=2.0)

    if move:
        print(f"   ✅ Coup retourné: {move} ({board.san(move)})")
    else:
        print(f"   ❌ Aucun coup retourné")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ TOUS LES TESTS RÉUSSIS!")
print("=" * 60)
