#!/usr/bin/env python3
"""
ai_integration_example.py - Exemple d'intégration entre l'IA et l'échiquier électronique

Ce programme démontre comment utiliser l'interface ChessboardLED pour afficher
les coups suggérés par une IA d'échecs.

Scénarios démontrés:
1. Affichage du meilleur coup de l'IA
2. Affichage de suggestions multiples
3. Mode assistance au joueur
4. Animation de coups
5. Communication via JSON

Auteur: Smart Chess Project
"""

import sys
import time
import json
from pathlib import Path

# Ajouter le répertoire parent au path pour importer chess_interface
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_interface import ChessboardLED, ChessMove


class MockChessEngine:
    """Simulateur d'IA d'échecs pour la démonstration."""

    def __init__(self):
        """Initialise le moteur simulé."""
        self.position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Position initiale

    def get_best_move(self, position: str = None) -> str:
        """
        Retourne le meilleur coup pour une position donnée.
        (Version simulée - retourne des coups prédéfinis)

        Args:
            position: Position FEN (non utilisé dans cette simulation)

        Returns:
            Coup UCI
        """
        # Coups d'ouverture classiques
        opening_moves = ["e2e4", "d2d4", "g1f3", "c2c4", "e2e3"]
        import random

        return random.choice(opening_moves)

    def get_top_moves(self, position: str = None, count: int = 3) -> list:
        """
        Retourne les N meilleurs coups avec évaluation.

        Args:
            position: Position FEN
            count: Nombre de coups à retourner

        Returns:
            Liste de dicts avec 'uci' et 'evaluation'
        """
        moves = [
            {"uci": "e2e4", "evaluation": 0.35, "rank": 1},
            {"uci": "d2d4", "evaluation": 0.28, "rank": 2},
            {"uci": "g1f3", "evaluation": 0.20, "rank": 3},
            {"uci": "c2c4", "evaluation": 0.15, "rank": 4},
            {"uci": "e2e3", "evaluation": 0.10, "rank": 5},
        ]
        return moves[:count]

    def analyze_position(self, position: str = None) -> dict:
        """
        Analyse complète d'une position.

        Returns:
            Dict avec le meilleur coup et les variantes
        """
        return {
            "best_move": "e2e4",
            "evaluation": 0.35,
            "depth": 20,
            "best_line": ["e2e4", "e7e5", "g1f3", "b8c6"],
            "alternatives": self.get_top_moves(position, 3),
        }


def demo_simple_move_display():
    """Démo 1: Affichage simple d'un coup suggéré par l'IA."""
    print("\n" + "=" * 60)
    print("DÉMO 1: Affichage simple d'un coup")
    print("=" * 60)

    board = ChessboardLED(verbose=True)
    engine = MockChessEngine()

    # L'IA calcule le meilleur coup
    print("\n[IA] Calcul du meilleur coup...")
    best_move = engine.get_best_move()
    print(f"[IA] Meilleur coup trouvé: {best_move}")

    # Afficher sur l'échiquier
    print("\n[Échiquier] Affichage du coup...")
    board.display_move(best_move)

    # Attendre
    print("\n[Info] Le coup reste affiché pendant 5 secondes...")
    time.sleep(5)

    # Nettoyer
    board.clear()
    print("\n✓ Démo 1 terminée\n")


def demo_animated_move():
    """Démo 2: Animation d'un coup."""
    print("\n" + "=" * 60)
    print("DÉMO 2: Animation d'un coup")
    print("=" * 60)

    board = ChessboardLED(verbose=True)
    engine = MockChessEngine()

    moves = ["e2e4", "g1f3", "f1c4", "e1g1"]  # Ouverture italienne

    for move in moves:
        print(f"\n[IA] Coup suggéré: {move}")
        board.animate_move(move, duration=2.0)
        time.sleep(1)

    board.clear()
    print("\n✓ Démo 2 terminée\n")


def demo_multiple_suggestions():
    """Démo 3: Affichage de plusieurs suggestions."""
    print("\n" + "=" * 60)
    print("DÉMO 3: Suggestions multiples")
    print("=" * 60)

    board = ChessboardLED(verbose=True)
    engine = MockChessEngine()

    # L'IA propose plusieurs coups
    print("\n[IA] Analyse de la position...")
    suggestions = engine.get_top_moves(count=3)

    print("\n[IA] Suggestions trouvées:")
    for i, move in enumerate(suggestions, 1):
        print(f"  {i}. {move['uci']} (éval: {move['evaluation']:+.2f})")

    # Afficher avec le meilleur en clignotant
    print("\n[Échiquier] Affichage des suggestions...")
    print("  - Meilleur coup: clignotant")
    print("  - Alternatives: fixes (faible intensité)")

    move_uci_list = [m["uci"] for m in suggestions]
    board.display_suggestions(move_uci_list, highlight_best=True, duration=8.0)

    print("\n✓ Démo 3 terminée\n")


def demo_player_assistance():
    """Démo 4: Mode assistance au joueur."""
    print("\n" + "=" * 60)
    print("DÉMO 4: Mode assistance au joueur")
    print("=" * 60)

    board = ChessboardLED(verbose=True)
    engine = MockChessEngine()

    # Simuler le coup du joueur
    player_move = "d2d4"
    print(f"\n[Joueur] Coup joué: {player_move}")
    board.display_move(player_move)
    time.sleep(2)

    # L'IA analyse
    print("\n[IA] Analyse du coup...")
    ai_best = engine.get_best_move()

    if player_move == ai_best:
        print("[IA] ✓ Excellent coup ! C'était le meilleur choix.")
        board.blink_move(player_move, duration=2.0, frequency=3.0)
    else:
        print(f"[IA] Le meilleur coup était: {ai_best}")
        print("[IA] Votre coup est aussi jouable.")

        # Montrer le meilleur coup en vert (simulation)
        board.clear()
        time.sleep(0.5)
        board.display_move(ai_best)
        time.sleep(3)

    board.clear()
    print("\n✓ Démo 4 terminée\n")


def demo_json_communication():
    """Démo 5: Communication via JSON."""
    print("\n" + "=" * 60)
    print("DÉMO 5: Communication JSON (simulation)")
    print("=" * 60)

    board = ChessboardLED(verbose=True)

    # Simuler la réception d'un message JSON de l'IA
    json_message = {
        "type": "suggestion",
        "uci": "e2e4",
        "from": "e2",
        "to": "e4",
        "evaluation": 0.35,
        "depth": 20,
        "timestamp": "2024-01-15T10:30:00Z",
    }

    print("\n[Réseau] Message JSON reçu:")
    print(json.dumps(json_message, indent=2))

    # Parser et afficher
    if json_message["type"] == "suggestion":
        uci = json_message["uci"]
        print(f"\n[Échiquier] Affichage du coup: {uci}")
        board.display_move(uci)
        time.sleep(3)

    # Envoyer un acquittement
    response = {
        "status": "ok",
        "displayed": True,
        "timestamp": "2024-01-15T10:30:01Z",
    }

    print("\n[Échiquier] Réponse envoyée:")
    print(json.dumps(response, indent=2))

    board.clear()
    print("\n✓ Démo 5 terminée\n")


def demo_special_moves():
    """Démo 6: Coups spéciaux (roque, promotion)."""
    print("\n" + "=" * 60)
    print("DÉMO 6: Coups spéciaux")
    print("=" * 60)

    board = ChessboardLED(verbose=True)

    special_moves = [
        ("e1g1", "Petit roque blanc"),
        ("e1c1", "Grand roque blanc"),
        ("e8g8", "Petit roque noir"),
        ("e7e8q", "Promotion en dame"),
        ("a7a8n", "Promotion en cavalier"),
    ]

    for uci, description in special_moves:
        print(f"\n[Test] {description}: {uci}")
        board.animate_move(uci, duration=2.0)
        time.sleep(1)

    board.clear()
    print("\n✓ Démo 6 terminée\n")


def demo_brightness_adjustment():
    """Démo 7: Ajustement de la luminosité."""
    print("\n" + "=" * 60)
    print("DÉMO 7: Ajustement de luminosité")
    print("=" * 60)

    board = ChessboardLED(verbose=True)

    board.display_move("e2e4")

    print("\n[Test] Variation de luminosité...")
    for brightness in [1.0, 0.7, 0.4, 0.1, 0.4, 0.7, 1.0]:
        print(f"  Luminosité: {brightness * 100:.0f}%")
        board.set_brightness(brightness)
        time.sleep(0.8)

    board.clear()
    print("\n✓ Démo 7 terminée\n")


def demo_continuous_analysis():
    """Démo 8: Analyse continue pendant la partie."""
    print("\n" + "=" * 60)
    print("DÉMO 8: Analyse continue (simulation)")
    print("=" * 60)

    board = ChessboardLED(verbose=True)
    engine = MockChessEngine()

    # Simuler une séquence de coups avec analyse IA
    moves_sequence = [
        ("e2e4", "Joueur Blanc"),
        ("e7e5", "Joueur Noir"),
        ("g1f3", "Joueur Blanc"),
        ("b8c6", "Joueur Noir"),
    ]

    print("\n[Info] Simulation d'une partie avec analyse IA\n")

    for move_uci, player in moves_sequence:
        # Afficher le coup joué
        print(f"[{player}] Joue: {move_uci}")
        board.display_move(move_uci)
        time.sleep(2)

        # L'IA analyse après chaque coup
        print(f"[IA] Analyse...")
        analysis = engine.analyze_position()
        print(f"[IA] Meilleur coup suivant: {analysis['best_move']}")
        print(f"[IA] Évaluation: {analysis['evaluation']:+.2f}")

        # Brève pause
        board.clear()
        time.sleep(0.5)

    print("\n✓ Démo 8 terminée\n")


def interactive_mode():
    """Mode interactif: entrer des coups manuellement."""
    print("\n" + "=" * 60)
    print("MODE INTERACTIF")
    print("=" * 60)
    print("\nEntrez des coups UCI (ex: e2e4) ou 'q' pour quitter\n")

    board = ChessboardLED(verbose=True)

    while True:
        try:
            user_input = input("[Vous] Coup UCI: ").strip().lower()

            if user_input in ["q", "quit", "exit"]:
                break

            if user_input == "clear":
                board.clear()
                continue

            # Valider et afficher
            try:
                move = ChessMove(user_input)
                print(f"[Échiquier] Affichage de {move.uci}")
                board.display_move(user_input)
            except ValueError as e:
                print(f"[Erreur] {e}")

        except KeyboardInterrupt:
            break

    board.clear()
    print("\n✓ Mode interactif terminé\n")


def main():
    """Programme principal."""
    print("\n" + "=" * 60)
    print(" Démonstration d'intégration IA - Échiquier Électronique")
    print("=" * 60)

    demos = {
        "1": ("Affichage simple", demo_simple_move_display),
        "2": ("Animation de coup", demo_animated_move),
        "3": ("Suggestions multiples", demo_multiple_suggestions),
        "4": ("Assistance joueur", demo_player_assistance),
        "5": ("Communication JSON", demo_json_communication),
        "6": ("Coups spéciaux", demo_special_moves),
        "7": ("Luminosité", demo_brightness_adjustment),
        "8": ("Analyse continue", demo_continuous_analysis),
        "9": ("Mode interactif", interactive_mode),
        "all": ("Toutes les démos", None),
    }

    print("\nChoisissez une démonstration:\n")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")
    print("\n  q. Quitter")

    try:
        choice = input("\n[Choix]: ").strip().lower()

        if choice == "q":
            print("Au revoir!")
            return

        if choice == "all":
            # Exécuter toutes les démos sauf interactive
            for key in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                demos[key][1]()
                time.sleep(1)
        elif choice in demos and demos[choice][1] is not None:
            demos[choice][1]()
        else:
            print("Choix invalide!")

    except KeyboardInterrupt:
        print("\n\nInterrompu par l'utilisateur")
    except Exception as e:
        print(f"\n[Erreur] {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nProgramme terminé.")


if __name__ == "__main__":
    main()
