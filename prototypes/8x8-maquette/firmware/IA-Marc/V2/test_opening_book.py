#!/usr/bin/env python3
"""
Test de l'intégration du livre d'ouvertures
============================================

Script de test pour vérifier que le livre d'ouvertures Polyglot
fonctionne correctement avec le moteur IA-Marc V2.

Usage:
    python test_opening_book.py
"""

import logging
import sys
import time

import chess

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def test_polyglot_book():
    """Test du livre Polyglot directement."""
    print("\n" + "=" * 70)
    print("TEST 1: Livre d'ouvertures Polyglot (Cerebellum_Light.bin)")
    print("=" * 70)

    try:
        from engine_polyglot import PolyglotBook

        # Chemins possibles
        paths = [
            "../book/Cerebellum_Light.bin",
            "book/Cerebellum_Light.bin",
            "../IA-Marc/book/Cerebellum_Light.bin",
        ]

        book = None
        for path in paths:
            book = PolyglotBook(path)
            if book.load():
                print(f"✓ Livre chargé depuis: {path}")
                print(f"  Entrées: {len(book.entries):,}")
                break

        if not book or not book.loaded:
            print("[ERREUR] Fichier Cerebellum_Light.bin non trouvé")
            print("        Veuillez placer le fichier dans ../book/")
            return False

        # Test sur position initiale
        print("\nTest position initiale:")
        board = chess.Board()

        moves = book.probe(board, max_moves=10)
        print(f"  Coups disponibles: {len(moves)}")

        for i, (move, weight) in enumerate(moves[:5], 1):
            print(f"    {i}. {move.uci():6} (poids: {weight})")

        # Test variété
        print("\nTest variété (10 essais):")
        move_counts = {}

        for _ in range(10):
            move = book.get_weighted_move(board, variety=True)
            move_str = move.uci() if move else "None"
            move_counts[move_str] = move_counts.get(move_str, 0) + 1

        for move, count in sorted(
            move_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    {move}: {count}/10")

        # Test après 1.e4
        print("\nTest après 1.e4:")
        board.push(chess.Move.from_uci("e2e4"))

        moves = book.probe(board, max_moves=5)
        print(f"  Coups disponibles: {len(moves)}")

        for i, (move, weight) in enumerate(moves[:5], 1):
            print(f"    {i}. {move.uci():6} (poids: {weight})")

        # Statistiques
        stats = book.get_stats()
        print(f"\nStatistiques:")
        print(f"  Probes: {stats['probes']}")
        print(f"  Hits: {stats['hits']}")
        print(f"  Hit rate: {stats['hit_rate']:.1f}%")

        print("\n[OK] Test Polyglot reussi!\n")
        return True

    except ImportError as e:
        print(f"[ERREUR] Erreur d'import: {e}")
        return False
    except Exception as e:
        print(f"[ERREUR] Erreur: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_opening_book_wrapper():
    """Test du wrapper OpeningBook."""
    print("=" * 70)
    print("TEST 2: Wrapper OpeningBook (support JSON et Polyglot)")
    print("=" * 70)

    try:
        from engine_opening import OpeningBook

        # Test Polyglot
        print("\nChargement Polyglot...")
        paths = [
            "../book/Cerebellum_Light.bin",
            "book/Cerebellum_Light.bin",
        ]

        book = None
        for path in paths:
            book = OpeningBook(path, book_type="polyglot")
            if book.load():
                print(f"✓ Livre Polyglot chargé: {path}")
                break

        if not book or not book.loaded:
            print("[ERREUR] Livre Polyglot non charge")
            return False

        # Test probe
        board = chess.Board()
        move = book.probe(board, elo_level=1800, variety=True)

        print(f"  Coup suggéré: {move}")

        # Statistiques
        stats = book.get_stats()
        print(f"  Type: {stats['book_type']}")
        print(f"  Positions: {stats['positions']:,}")

        print("\n[OK] Test OpeningBook reussi!\n")
        return True

    except Exception as e:
        print(f"[ERREUR] Erreur: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_engine_integration():
    """Test de l'intégration dans le moteur."""
    print("=" * 70)
    print("TEST 3: Intégration dans le moteur IA-Marc V2")
    print("=" * 70)

    try:
        from engine_main import ChessEngine

        # Créer le moteur
        print("\nCréation du moteur...")
        engine = ChessEngine(verbose=False)

        # Vérifier que le livre est chargé
        if engine.opening_book is None:
            print("[WARN] Aucun livre d'ouvertures charge dans le moteur")
            print("       Le moteur fonctionnera quand meme, mais sans opening book")
            return False

        print(f"✓ Moteur créé avec livre d'ouvertures")

        # Test avec différents niveaux
        levels = ["Debutant", "Amateur", "Club", "Competition"]

        board = chess.Board()

        print("\nTest avec différents niveaux:")
        for level_name in levels:
            engine.set_level(level_name)

            start = time.time()
            move = engine.get_move(board, time_limit=2.0)
            elapsed = time.time() - start

            stats = engine.get_stats()
            elo = stats["elo"]

            # Vérifier si le coup vient du livre
            if "opening_book" in stats:
                book_stats = stats["opening_book"]
                from_book = book_stats["hits"] > 0
            else:
                from_book = False

            source = "[Book]" if from_book and elapsed < 0.1 else "[Calcul]"

            print(f"  {level_name:12} (ELO {elo:4}): {move} {source} ({elapsed:.3f}s)")

        # Test d'une partie complète (premiers coups)
        print("\nTest d'une partie complète (12 premiers coups):")
        board = chess.Board()
        engine.set_level("Club")

        move_count = 0
        book_count = 0

        for i in range(12):
            stats_before = engine.opening_book.get_stats()
            hits_before = stats_before["hits"]

            move = engine.get_move(board, time_limit=1.0)

            if not move:
                break

            stats_after = engine.opening_book.get_stats()
            hits_after = stats_after["hits"]

            from_book = hits_after > hits_before

            move_count += 1
            if from_book:
                book_count += 1

            print(f"  {move_count:2}. {move} {'[Book]' if from_book else '[Calcul]'}")

            board.push(move)

            if board.is_game_over():
                break

        print(
            f"\nRésultat: {book_count}/{move_count} coups du livre ({book_count * 100 / move_count:.0f}%)"
        )

        # Statistiques finales
        print("\nStatistiques du livre d'ouvertures:")
        book_stats = engine.opening_book.get_stats()
        print(f"  Type: {book_stats['book_type']}")
        print(f"  Positions: {book_stats.get('positions', 'N/A'):,}")
        print(f"  Probes: {book_stats['probes']}")
        print(f"  Hits: {book_stats['hits']}")
        print(f"  Hit rate: {book_stats['hit_rate']:.1f}%")

        print("\n[OK] Test d'integration reussi!\n")
        return True

    except Exception as e:
        print(f"[ERREUR] Erreur: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_disable_opening_book():
    """Test de désactivation du livre."""
    print("=" * 70)
    print("TEST 4: Désactivation du livre d'ouvertures")
    print("=" * 70)

    try:
        from engine_main import ChessEngine

        engine = ChessEngine(verbose=False)
        board = chess.Board()

        # Avec le livre
        print("\nAvec le livre d'ouvertures:")
        engine.enable_opening_book(True)

        start = time.time()
        move1 = engine.get_move(board, time_limit=2.0)
        time1 = time.time() - start

        print(f"  Coup: {move1}")
        print(f"  Temps: {time1:.3f}s")

        # Sans le livre
        print("\nSans le livre d'ouvertures:")
        engine.enable_opening_book(False)
        engine.reset()

        board = chess.Board()

        start = time.time()
        move2 = engine.get_move(board, time_limit=2.0)
        time2 = time.time() - start

        print(f"  Coup: {move2}")
        print(f"  Temps: {time2:.3f}s")

        # Comparaison
        print(f"\nGain de temps avec le livre: {time2 - time1:.3f}s")

        if time1 < 0.1:
            print("[OK] Le livre permet un coup instantane!")
        else:
            print("[WARN] Le coup devrait etre instantane avec le livre")

        print("\n[OK] Test de desactivation reussi!\n")
        return True

    except Exception as e:
        print(f"[ERREUR] Erreur: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Lance tous les tests."""
    print("\n" + "=" * 70)
    print("TESTS DU LIVRE D'OUVERTURES - IA-Marc V2")
    print("=" * 70)

    results = []

    # Test 1: Polyglot direct
    results.append(("Polyglot Book", test_polyglot_book()))

    # Test 2: Wrapper OpeningBook
    results.append(("OpeningBook Wrapper", test_opening_book_wrapper()))

    # Test 3: Intégration moteur
    results.append(("Intégration Moteur", test_engine_integration()))

    # Test 4: Désactivation
    results.append(("Désactivation", test_disable_opening_book()))

    # Résumé
    print("=" * 70)
    print("RÉSUMÉ DES TESTS")
    print("=" * 70)

    success_count = 0
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status:8} - {test_name}")
        if success:
            success_count += 1

    print("-" * 70)
    print(
        f"Résultat: {success_count}/{len(results)} tests réussis ({success_count * 100 // len(results)}%)"
    )
    print("=" * 70)

    # Recommandations
    if success_count == len(results):
        print("\nTous les tests sont reussis!")
        print("\nLe livre d'ouvertures est correctement integre.")
        print(
            "Le moteur jouera instantanement les coups theoriques en debut de partie."
        )
    elif success_count > 0:
        print("\nCertains tests ont echoue.")
        print("\nVerifiez que Cerebellum_Light.bin est bien present dans:")
        print("  - prototypes/8x8-maquette/firmware/IA-Marc/book/")
        print("  - ou prototypes/8x8-maquette/firmware/IA-Marc/V2/book/")
    else:
        print("\nTous les tests ont echoue.")
        print("\nAssurez-vous que:")
        print("  1. Le fichier Cerebellum_Light.bin est present")
        print("  2. Les modules engine_polyglot.py et engine_opening.py sont corrects")
        print("  3. Les dependances (chess) sont installees")

    return success_count == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
