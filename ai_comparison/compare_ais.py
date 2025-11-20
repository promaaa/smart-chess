#!/usr/bin/env python3
"""
Comparaison des deux moteurs d'échecs IA
=========================================

Compare l'ancienne IA (alphabeta_engine) avec la nouvelle IA (IA-Marc V2)
en mesurant les temps de réflexion et en estimant les performances sur RPi5.

Usage:
    python3 compare_ais.py
"""

import os
import platform
import sys
import time
import traceback

import chess

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai"))
sys.path.insert(
    0, os.path.join(PROJECT_ROOT, "prototypes/8x8-maquette/firmware/IA-Marc/V2")
)

# Import Old AI
try:
    from alphabeta_engine import AlphaBetaEngine
    from Chess import Chess as OldChessBoard
    from evaluator import AdvancedChessEvaluator

    OLD_AI_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Old AI non disponible: {e}")
    OLD_AI_AVAILABLE = False

# Import New AI
try:
    from engine_main import ChessEngine

    NEW_AI_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] New AI non disponible: {e}")
    NEW_AI_AVAILABLE = False


# Facteurs de performance estimés (benchmark relatif)
# Basé sur des benchmarks multi-coeurs typiques
PERFORMANCE_FACTORS = {
    "M4": 1.0,  # Apple M4 (référence)
    "M3": 0.85,  # Apple M3
    "M2": 0.75,  # Apple M2
    "M1": 0.65,  # Apple M1
    "RPi5": 0.20,  # Raspberry Pi 5 (4 coeurs Cortex-A76 @ 2.4GHz)
    "RPi4": 0.10,  # Raspberry Pi 4 (pour référence)
}


def detect_platform():
    """Détecte la plateforme actuelle."""
    system = platform.system()
    machine = platform.machine()
    processor = platform.processor()

    # Détection Apple Silicon
    if system == "Darwin" and machine == "arm64":
        # Essayer de détecter le modèle exact
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            brand = result.stdout.strip().lower()
            if "m4" in brand:
                return "M4"
            elif "m3" in brand:
                return "M3"
            elif "m2" in brand:
                return "M2"
            elif "m1" in brand:
                return "M1"
        except:
            pass
        return "M1"  # Par défaut pour Apple Silicon

    return "Unknown"


def estimate_rpi5_time(measured_time, current_platform="M4"):
    """Estime le temps sur RPi5 basé sur le temps mesuré."""
    if current_platform not in PERFORMANCE_FACTORS:
        current_platform = "M4"

    current_factor = PERFORMANCE_FACTORS[current_platform]
    rpi5_factor = PERFORMANCE_FACTORS["RPi5"]

    # Temps RPi5 = Temps mesuré * (facteur RPi5 / facteur actuel)
    estimated = measured_time * (current_factor / rpi5_factor)
    return estimated


def get_old_ai_move(engine, fen):
    """Obtient un coup de l'ancienne IA."""
    old_board = OldChessBoard()
    old_board.load_fen(fen)

    # Parser le FEN complet
    parts = fen.split(" ")
    old_board.white_to_move = parts[1] == "w"

    # Droits de roque
    castling = parts[2]
    old_board.castling_rights = {
        "K": "K" in castling,
        "Q": "Q" in castling,
        "k": "k" in castling,
        "q": "q" in castling,
    }

    # En passant
    ep_square = parts[3]
    if ep_square != "-":
        file = ord(ep_square[0]) - ord("a")
        rank = int(ep_square[1]) - 1
        old_board.en_passant_target = rank * 8 + file
    else:
        old_board.en_passant_target = None

    # Obtenir le coup
    move_tuple = engine.get_best_move(old_board)

    if move_tuple is None:
        return None

    from_sq, to_sq, promotion = move_tuple

    # Conversion de la promotion
    promotion_val = None
    if promotion:
        promotion = promotion.lower()
        if promotion == "q":
            promotion_val = chess.QUEEN
        elif promotion == "r":
            promotion_val = chess.ROOK
        elif promotion == "b":
            promotion_val = chess.BISHOP
        elif promotion == "n":
            promotion_val = chess.KNIGHT

    return chess.Move(from_sq, to_sq, promotion=promotion_val)


class PerformanceStats:
    """Classe pour suivre les statistiques de performance."""

    def __init__(self, name):
        self.name = name
        self.times = []
        self.moves = 0
        self.errors = 0

    def add_time(self, duration):
        self.times.append(duration)
        self.moves += 1

    def add_error(self):
        self.errors += 1

    def get_average(self):
        if not self.times:
            return 0
        return sum(self.times) / len(self.times)

    def get_total(self):
        return sum(self.times)

    def get_min(self):
        return min(self.times) if self.times else 0

    def get_max(self):
        return max(self.times) if self.times else 0


def play_match(white_player="new", black_player="old", max_moves=200):
    """Joue une partie entre les deux IA avec statistiques."""

    if not OLD_AI_AVAILABLE or not NEW_AI_AVAILABLE:
        print("[ERREUR] Les deux IA doivent être disponibles")
        return None, None

    board = chess.Board()

    # Initialiser les moteurs
    print("Initialisation des moteurs...")
    old_ai = AlphaBetaEngine(max_depth=4, evaluator=AdvancedChessEvaluator())
    new_ai = ChessEngine()
    new_ai.set_level("Club")  # Niveau équilibré

    # Statistiques
    old_stats = PerformanceStats("Old AI")
    new_stats = PerformanceStats("New AI (IA-Marc V2)")

    # Détection plateforme
    current_platform = detect_platform()
    print(f"Plateforme detectee: {current_platform}")
    print(f"\nPartie: Blancs={white_player} vs Noirs={black_player}")
    print("=" * 70)

    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        move_count += 1
        current_player = white_player if board.turn == chess.WHITE else black_player
        current_stats = new_stats if current_player == "new" else old_stats

        color = "Blancs" if board.turn == chess.WHITE else "Noirs"
        print(f"\nCoup {board.fullmove_number} - {color} ({current_player})")

        start_time = time.time()
        move = None

        try:
            if current_player == "old":
                move = get_old_ai_move(old_ai, board.fen())
            else:
                move = new_ai.get_move(board, time_limit=2.0)

            duration = time.time() - start_time

            if move is None or move not in board.legal_moves:
                print(f"[ERREUR] Coup invalide de {current_player}")
                current_stats.add_error()
                break

            current_stats.add_time(duration)
            print(f"  Joue: {move.uci()} ({duration:.3f}s)")
            board.push(move)

        except Exception as e:
            duration = time.time() - start_time
            print(f"[ERREUR] {current_player}: {e}")
            traceback.print_exc()
            current_stats.add_error()
            break

    print("\n" + "=" * 70)
    print("PARTIE TERMINEE")
    print(f"Resultat: {board.result()}")

    return old_stats, new_stats, board.result()


def print_performance_report(old_stats, new_stats, current_platform):
    """Affiche le rapport de performance détaillé."""

    print("\n" + "=" * 70)
    print("RAPPORT DE PERFORMANCE")
    print("=" * 70)

    # Statistiques Old AI
    print(f"\n{old_stats.name}:")
    print(f"  Coups joues     : {old_stats.moves}")
    print(f"  Temps total     : {old_stats.get_total():.2f}s")
    print(f"  Temps moyen     : {old_stats.get_average():.3f}s")
    print(
        f"  Temps min/max   : {old_stats.get_min():.3f}s / {old_stats.get_max():.3f}s"
    )
    print(f"  Erreurs         : {old_stats.errors}")

    # Statistiques New AI
    print(f"\n{new_stats.name}:")
    print(f"  Coups joues     : {new_stats.moves}")
    print(f"  Temps total     : {new_stats.get_total():.2f}s")
    print(f"  Temps moyen     : {new_stats.get_average():.3f}s")
    print(
        f"  Temps min/max   : {new_stats.get_min():.3f}s / {new_stats.get_max():.3f}s"
    )
    print(f"  Erreurs         : {new_stats.errors}")

    # Comparaison
    print(f"\nComparaison:")
    if old_stats.get_average() > 0 and new_stats.get_average() > 0:
        ratio = old_stats.get_average() / new_stats.get_average()
        if ratio > 1:
            print(f"  New AI est {ratio:.2f}x plus rapide que Old AI")
        else:
            print(f"  Old AI est {1 / ratio:.2f}x plus rapide que New AI")

    # Estimations RPi5
    print(f"\n" + "=" * 70)
    print(f"ESTIMATION RASPBERRY PI 5")
    print(f"=" * 70)
    print(f"Plateforme actuelle: {current_platform}")

    old_rpi5 = estimate_rpi5_time(old_stats.get_average(), current_platform)
    new_rpi5 = estimate_rpi5_time(new_stats.get_average(), current_platform)

    print(f"\nTemps moyen estime sur RPi5:")
    print(f"  {old_stats.name:25} : {old_rpi5:.3f}s")
    print(f"  {new_stats.name:25} : {new_rpi5:.3f}s")

    # Vérifier si dans les limites acceptables
    print(f"\nAnalyse pour RPi5:")
    if new_rpi5 < 3.0:
        print(f"  [OK] New AI reste rapide (< 3s) sur RPi5")
    elif new_rpi5 < 5.0:
        print(f"  [WARN] New AI acceptable (< 5s) sur RPi5")
    else:
        print(f"  [ALERT] New AI pourrait etre lent (> 5s) sur RPi5")

    if old_rpi5 < 3.0:
        print(f"  [OK] Old AI reste rapide (< 3s) sur RPi5")
    elif old_rpi5 < 5.0:
        print(f"  [WARN] Old AI acceptable (< 5s) sur RPi5")
    else:
        print(f"  [ALERT] Old AI pourrait etre lent (> 5s) sur RPi5")


def main():
    """Point d'entrée principal."""

    print("=" * 70)
    print("COMPARAISON DES MOTEURS D'ECHECS")
    print("=" * 70)

    if not OLD_AI_AVAILABLE:
        print("[ERREUR] Old AI non disponible")
        return

    if not NEW_AI_AVAILABLE:
        print("[ERREUR] New AI non disponible")
        return

    # Détecter la plateforme
    current_platform = detect_platform()

    # Configuration du match
    num_games = 4
    wins_new = 0
    wins_old = 0
    draws = 0
    
    total_old_stats = PerformanceStats("Old AI (Total)")
    total_new_stats = PerformanceStats("New AI (Total)")

    print(f"Lancement de {num_games} parties pour une comparaison fiable...")

    for i in range(num_games):
        # Alternance des couleurs
        if i % 2 == 0:
            white = "new"
            black = "old"
        else:
            white = "old"
            black = "new"
            
        print(f"\n--- Partie {i+1}/{num_games} ({white} vs {black}) ---")
        
        old_stats, new_stats, result = play_match(
            white_player=white, black_player=black, max_moves=200
        )

        if old_stats and new_stats:
            # Agrégation des stats
            total_old_stats.times.extend(old_stats.times)
            total_old_stats.moves += old_stats.moves
            total_old_stats.errors += old_stats.errors
            
            total_new_stats.times.extend(new_stats.times)
            total_new_stats.moves += new_stats.moves
            total_new_stats.errors += new_stats.errors
            
            # Suivi des victoires
            if result == "1-0":
                if white == "new":
                    wins_new += 1
                else:
                    wins_old += 1
            elif result == "0-1":
                if black == "new":
                    wins_new += 1
                else:
                    wins_old += 1
            else:
                draws += 1

    # Affichage du rapport global
    print("\n" + "=" * 70)
    print("RÉSULTATS AGRÉGÉS")
    print("=" * 70)
    print(f"Score Final sur {num_games} parties:")
    print(f"  New AI (IA-Marc V2) : {wins_new} victoires")
    print(f"  Old AI              : {wins_old} victoires")
    print(f"  Nuls                : {draws}")
    
    print_performance_report(total_old_stats, total_new_stats, current_platform)

    print("\n" + "=" * 70)
    print("Comparaison terminée")
    print("=" * 70)


if __name__ == "__main__":
    main()
