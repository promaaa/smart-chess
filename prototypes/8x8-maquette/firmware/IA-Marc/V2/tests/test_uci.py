#!/usr/bin/env python3
"""
IA-Marc V2 - Tests du Moteur UCI
=================================

Tests automatisés pour valider le protocole UCI et le moteur d'échecs.

Usage:
    python3 tests/test_uci.py
    pytest tests/test_uci.py -v
"""

import subprocess
import sys
import time
from typing import List, Optional, Tuple


class UCITester:
    """Classe pour tester le moteur UCI via subprocess."""

    def __init__(
        self, engine_path: str = "chess_engine_uci.py", use_pypy: bool = False
    ):
        """
        Initialise le testeur UCI.

        Args:
            engine_path: Chemin vers le moteur UCI
            use_pypy: Utiliser PyPy3 au lieu de Python3
        """
        self.engine_path = engine_path
        self.use_pypy = use_pypy
        self.process = None

    def start(self):
        """Démarre le processus du moteur."""
        cmd = ["pypy3" if self.use_pypy else "python3", self.engine_path]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        print(f"✓ Moteur démarré: {' '.join(cmd)}")

    def stop(self):
        """Arrête le processus du moteur."""
        if self.process:
            self.send("quit")
            self.process.wait(timeout=2)
            self.process = None
            print("✓ Moteur arrêté")

    def send(self, command: str):
        """Envoie une commande au moteur."""
        if not self.process:
            raise RuntimeError("Moteur non démarré")

        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        print(f"→ {command}")

    def read_line(self, timeout: float = 1.0) -> Optional[str]:
        """Lit une ligne de réponse du moteur."""
        if not self.process:
            return None

        # Simple timeout simulation
        start = time.time()
        while time.time() - start < timeout:
            line = self.process.stdout.readline()
            if line:
                line = line.strip()
                print(f"← {line}")
                return line
        return None

    def read_until(self, keyword: str, timeout: float = 5.0) -> List[str]:
        """
        Lit les lignes jusqu'à trouver le mot-clé.

        Args:
            keyword: Mot-clé à chercher
            timeout: Timeout en secondes

        Returns:
            Liste de toutes les lignes lues
        """
        lines = []
        start = time.time()

        while time.time() - start < timeout:
            line = self.read_line(timeout=0.1)
            if line:
                lines.append(line)
                if keyword in line:
                    return lines

        return lines

    def expect(self, expected: str, timeout: float = 1.0) -> bool:
        """
        Attend une réponse spécifique.

        Args:
            expected: Texte attendu
            timeout: Timeout en secondes

        Returns:
            True si trouvé, False sinon
        """
        lines = self.read_until(expected, timeout)
        return any(expected in line for line in lines)


# ============================================================================
# TESTS
# ============================================================================


def test_01_uci_handshake():
    """Test 1: Handshake UCI de base."""
    print("\n" + "=" * 60)
    print("TEST 1: UCI Handshake")
    print("=" * 60)

    tester = UCITester()
    tester.start()

    try:
        # Envoyer "uci"
        tester.send("uci")

        # Attendre "uciok"
        if tester.expect("uciok", timeout=2.0):
            print("✅ UCI handshake réussi")
            return True
        else:
            print("❌ Pas de réponse 'uciok'")
            return False

    finally:
        tester.stop()


def test_02_isready():
    """Test 2: Commande isready."""
    print("\n" + "=" * 60)
    print("TEST 2: isready")
    print("=" * 60)

    tester = UCITester()
    tester.start()

    try:
        tester.send("uci")
        tester.expect("uciok", timeout=2.0)

        # Test isready
        tester.send("isready")

        if tester.expect("readyok", timeout=2.0):
            print("✅ isready fonctionne")
            return True
        else:
            print("❌ Pas de réponse 'readyok'")
            return False

    finally:
        tester.stop()


def test_03_position_startpos():
    """Test 3: Position initiale."""
    print("\n" + "=" * 60)
    print("TEST 3: position startpos")
    print("=" * 60)

    tester = UCITester()
    tester.start()

    try:
        tester.send("uci")
        tester.expect("uciok", timeout=2.0)

        # Définir position initiale
        tester.send("position startpos")

        # Vérifier que le moteur répond bien après
        tester.send("isready")

        if tester.expect("readyok", timeout=2.0):
            print("✅ position startpos acceptée")
            return True
        else:
            print("❌ Moteur ne répond plus après position")
            return False

    finally:
        tester.stop()


def test_04_position_with_moves():
    """Test 4: Position avec coups."""
    print("\n" + "=" * 60)
    print("TEST 4: position startpos moves e2e4 e7e5")
    print("=" * 60)

    tester = UCITester()
    tester.start()

    try:
        tester.send("uci")
        tester.expect("uciok", timeout=2.0)

        # Position avec coups
        tester.send("position startpos moves e2e4 e7e5")

        tester.send("isready")

        if tester.expect("readyok", timeout=2.0):
            print("✅ position avec moves acceptée")
            return True
        else:
            print("❌ Moteur ne répond plus")
            return False

    finally:
        tester.stop()


def test_05_go_movetime():
    """Test 5: Recherche avec movetime."""
    print("\n" + "=" * 60)
    print("TEST 5: go movetime 1000")
    print("=" * 60)

    tester = UCITester()
    tester.start()

    try:
        tester.send("uci")
        tester.expect("uciok", timeout=2.0)

        tester.send("position startpos")
        tester.send("go movetime 1000")

        # Attendre bestmove (max 5 secondes)
        lines = tester.read_until("bestmove", timeout=5.0)

        bestmove_found = False
        info_found = False

        for line in lines:
            if "bestmove" in line:
                bestmove_found = True
                print(f"✓ Coup trouvé: {line}")
            if "info depth" in line:
                info_found = True

        if bestmove_found:
            print("✅ Recherche movetime réussie")
            if info_found:
                print("✓ Bonus: Info depth envoyé")
            return True
        else:
            print("❌ Pas de bestmove reçu")
            return False

    finally:
        tester.stop()


def test_06_go_depth():
    """Test 6: Recherche avec profondeur fixe."""
    print("\n" + "=" * 60)
    print("TEST 6: go depth 3")
    print("=" * 60)

    tester = UCITester()
    tester.start()

    try:
        tester.send("uci")
        tester.expect("uciok", timeout=2.0)

        tester.send("position startpos")
        tester.send("go depth 3")

        lines = tester.read_until("bestmove", timeout=10.0)

        bestmove_found = False
        depth_3_found = False

        for line in lines:
            if "bestmove" in line:
                bestmove_found = True
            if "info depth 3" in line:
                depth_3_found = True

        if bestmove_found:
            print("✅ Recherche depth réussie")
            if depth_3_found:
                print("✓ Bonus: Profondeur 3 atteinte")
            return True
        else:
            print("❌ Pas de bestmove reçu")
            return False

    finally:
        tester.stop()


def test_07_setoption_level():
    """Test 7: Configuration du niveau."""
    print("\n" + "=" * 60)
    print("TEST 7: setoption name Level value Amateur")
    print("=" * 60)

    tester = UCITester()
    tester.start()

    try:
        tester.send("uci")
        tester.expect("uciok", timeout=2.0)

        # Changer le niveau
        tester.send("setoption name Level value Amateur")

        tester.send("isready")

        if tester.expect("readyok", timeout=2.0):
            print("✅ setoption Level accepté")

            # Tester que le niveau est appliqué
            tester.send("position startpos")
            tester.send("go movetime 500")

            if tester.expect("bestmove", timeout=3.0):
                print("✓ Moteur fonctionne avec nouveau niveau")
                return True
            else:
                print("⚠️ Niveau changé mais recherche échouée")
                return False
        else:
            print("❌ setoption non accepté")
            return False

    finally:
        tester.stop()


def test_08_multiple_searches():
    """Test 8: Recherches multiples (stabilité)."""
    print("\n" + "=" * 60)
    print("TEST 8: Multiple searches (stabilité)")
    print("=" * 60)

    tester = UCITester()
    tester.start()

    try:
        tester.send("uci")
        tester.expect("uciok", timeout=2.0)

        success_count = 0

        for i in range(5):
            print(f"\n--- Recherche {i + 1}/5 ---")
            tester.send("position startpos")
            tester.send("go movetime 200")

            if tester.expect("bestmove", timeout=3.0):
                success_count += 1
                print(f"✓ Recherche {i + 1} OK")
            else:
                print(f"✗ Recherche {i + 1} ÉCHEC")

        if success_count == 5:
            print(f"\n✅ Stabilité parfaite: {success_count}/5 recherches réussies")
            return True
        elif success_count >= 3:
            print(f"\n⚠️ Stabilité acceptable: {success_count}/5 recherches réussies")
            return True
        else:
            print(f"\n❌ Instable: seulement {success_count}/5 recherches réussies")
            return False

    finally:
        tester.stop()


def test_09_performance_benchmark():
    """Test 9: Benchmark de performance."""
    print("\n" + "=" * 60)
    print("TEST 9: Performance Benchmark")
    print("=" * 60)

    tester = UCITester()
    tester.start()

    try:
        tester.send("uci")
        tester.expect("uciok", timeout=2.0)

        tester.send("position startpos")

        # Recherche de 3 secondes
        print("\nRecherche de 3 secondes...")
        tester.send("go movetime 3000")

        lines = tester.read_until("bestmove", timeout=10.0)

        # Analyser les stats
        max_depth = 0
        total_nodes = 0
        nps = 0

        for line in lines:
            if "info depth" in line:
                parts = line.split()
                try:
                    depth_idx = parts.index("depth")
                    depth = int(parts[depth_idx + 1])
                    max_depth = max(max_depth, depth)

                    if "nodes" in parts:
                        nodes_idx = parts.index("nodes")
                        total_nodes = int(parts[nodes_idx + 1])

                    if "nps" in parts:
                        nps_idx = parts.index("nps")
                        nps = int(parts[nps_idx + 1])
                except (ValueError, IndexError):
                    pass

        print(f"\n📊 Résultats:")
        print(f"  Profondeur max: {max_depth}")
        print(f"  Nœuds totaux: {total_nodes:,}")
        print(f"  NPS (Nœuds/sec): {nps:,}")

        # Critères de succès
        if max_depth >= 4:
            print("\n✅ Performance acceptable (depth >= 4)")
            if nps >= 10000:
                print(f"✓ Bonus: Excellent NPS ({nps:,})")
            return True
        else:
            print(f"\n⚠️ Performance faible (depth = {max_depth})")
            return False

    finally:
        tester.stop()


def test_10_pypy_comparison():
    """Test 10: Comparaison CPython vs PyPy."""
    print("\n" + "=" * 60)
    print("TEST 10: CPython vs PyPy Performance")
    print("=" * 60)

    results = {}

    # Test avec CPython
    print("\n--- Test CPython ---")
    tester_python = UCITester(use_pypy=False)
    try:
        tester_python.start()
        tester_python.send("uci")
        tester_python.expect("uciok", timeout=2.0)
        tester_python.send("position startpos")
        tester_python.send("go movetime 2000")

        lines = tester_python.read_until("bestmove", timeout=10.0)

        for line in lines:
            if "nps" in line:
                parts = line.split()
                try:
                    nps_idx = parts.index("nps")
                    results["python"] = int(parts[nps_idx + 1])
                except (ValueError, IndexError):
                    pass
    finally:
        tester_python.stop()

    # Test avec PyPy
    print("\n--- Test PyPy ---")
    tester_pypy = UCITester(use_pypy=True)
    try:
        tester_pypy.start()
        tester_pypy.send("uci")
        tester_pypy.expect("uciok", timeout=2.0)
        tester_pypy.send("position startpos")
        tester_pypy.send("go movetime 2000")

        lines = tester_pypy.read_until("bestmove", timeout=10.0)

        for line in lines:
            if "nps" in line:
                parts = line.split()
                try:
                    nps_idx = parts.index("nps")
                    results["pypy"] = int(parts[nps_idx + 1])
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        print(f"⚠️ PyPy non disponible ou erreur: {e}")
        results["pypy"] = None
    finally:
        tester_pypy.stop()

    # Comparaison
    print(f"\n📊 Comparaison:")
    if "python" in results:
        print(f"  CPython: {results['python']:,} NPS")
    if results.get("pypy"):
        print(f"  PyPy:    {results['pypy']:,} NPS")

        if "python" in results and results["python"] > 0:
            speedup = results["pypy"] / results["python"]
            print(f"  Speedup: {speedup:.2f}x")

            if speedup >= 2.0:
                print("\n✅ PyPy apporte un gain significatif (>2x)")
            elif speedup >= 1.2:
                print("\n✓ PyPy apporte un gain modéré (>1.2x)")
            else:
                print("\n⚠️ PyPy n'apporte pas de gain majeur")
    else:
        print("  PyPy: Non disponible")

    return True


# ============================================================================
# MAIN
# ============================================================================


def run_all_tests():
    """Exécute tous les tests."""
    print("\n" + "=" * 60)
    print("IA-MARC V2 - SUITE DE TESTS UCI")
    print("=" * 60)

    tests = [
        test_01_uci_handshake,
        test_02_isready,
        test_03_position_startpos,
        test_04_position_with_moves,
        test_05_go_movetime,
        test_06_go_depth,
        test_07_setoption_level,
        test_08_multiple_searches,
        test_09_performance_benchmark,
        test_10_pypy_comparison,
    ]

    results = []

    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n❌ ERREUR dans {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            results.append((test.__name__, False))

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\n{passed}/{total} tests réussis ({passed * 100 // total}%)")

    if passed == total:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS!")
    elif passed >= total * 0.8:
        print("\n✓ La plupart des tests passent")
    else:
        print("\n⚠️ Plusieurs tests ont échoué")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
