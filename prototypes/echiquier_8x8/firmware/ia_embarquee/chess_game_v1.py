#!/usr/bin/env python3
"""
Script principal de l'Échiquier Intelligent (Smart Chess Board).
Intègre la gestion matérielle (I2C/Reed/LEDs) et l'IA maison (Engine + Searcher).

Auteur: Projet Smart Chess
Date: 18/11/2025
Plateforme: Raspberry Pi 5
"""

import sys
import time

import chess  # pip install chess

# --- Importation de l'IA Maison ---
try:
    import os
    # Chemin relatif depuis firmware/ia_embarquee/ vers la racine du projet
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../ia_marc/V1"))
    from engine_brain import Engine
    from engine_search import Searcher

    AI_AVAILABLE = True
except ImportError:
    # Fallback: Essayer le chemin local si le dossier a été déplacé
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "../ia_marc/V1"))
        from engine_brain import Engine
        from engine_search import Searcher
        AI_AVAILABLE = True
    except ImportError:
        print(
            "ERREUR CRITIQUE: Les fichiers 'engine_brain.py' ou 'engine_search.py' sont manquants."
        )
        sys.exit(1)

# --- Importation Matériel ---
import adafruit_tca9548a
import board
import busio
import digitalio
from adafruit_ht16k33.matrix import Matrix16x8
from adafruit_mcp230xx.mcp23017 import MCP23017

# --- CONFIGURATION ---
DEFAULT_AI_TIME = 3.0  # Temps de réflexion max pour l'IA (secondes)

# --- DÉBUT DE LA TABLE DE TRADUCTION (LED_MAP) ---
# (Copié depuis votre fichier original)
LED_MAP = {
    (8, 0): ("C4", 0, 0),
    (7, 0): ("C4", 0, 1),
    (6, 0): ("C4", 0, 2),
    (5, 0): ("C4", 0, 3),
    (4, 0): ("C4", 0, 4),
    (3, 0): ("C4", 0, 5),
    (2, 0): ("C4", 0, 6),
    (1, 0): ("C4", 0, 7),
    (8, 1): ("C4", 1, 0),
    (7, 1): ("C4", 1, 1),
    (6, 1): ("C4", 1, 2),
    (5, 1): ("C4", 1, 3),
    (4, 1): ("C4", 1, 4),
    (3, 1): ("C4", 1, 5),
    (2, 1): ("C4", 1, 6),
    (1, 1): ("C4", 1, 7),
    (0, 1): ("C4", 8, 1),
    (8, 2): ("C4", 2, 0),
    (7, 2): ("C4", 2, 1),
    (6, 2): ("C4", 2, 2),
    (5, 2): ("C4", 2, 3),
    (4, 2): ("C4", 2, 4),
    (3, 2): ("C4", 2, 5),
    (2, 2): ("C4", 2, 6),
    (1, 2): ("C4", 2, 7),
    (8, 3): ("C4", 3, 0),
    (7, 3): ("C4", 3, 1),
    (6, 3): ("C4", 3, 2),
    (5, 3): ("C4", 3, 3),
    (4, 3): ("C4", 3, 4),
    (3, 3): ("C4", 3, 5),
    (2, 3): ("C4", 3, 6),
    (1, 3): ("C4", 3, 7),
    (8, 4): ("C4", 4, 0),
    (7, 4): ("C4", 4, 1),
    (6, 4): ("C4", 4, 2),
    (5, 4): ("C4", 4, 3),
    (4, 4): ("C4", 4, 4),
    (3, 4): ("C4", 4, 5),
    (2, 4): ("C4", 4, 6),
    (1, 4): ("C4", 4, 7),
    (8, 5): ("C4", 5, 0),
    (7, 5): ("C4", 5, 1),
    (6, 5): ("C4", 5, 2),
    (5, 5): ("C4", 5, 3),
    (4, 5): ("C4", 5, 4),
    (3, 5): ("C4", 5, 5),
    (2, 5): ("C4", 5, 6),
    (1, 5): ("C4", 5, 7),
    (8, 6): ("C4", 6, 0),
    (7, 6): ("C4", 6, 1),
    (6, 6): ("C4", 6, 2),
    (5, 6): ("C4", 6, 3),
    (4, 6): ("C4", 6, 4),
    (3, 6): ("C4", 6, 5),
    (2, 6): ("C4", 6, 6),
    (1, 6): ("C4", 6, 7),
    (8, 7): ("C4", 7, 0),
    (7, 7): ("C4", 7, 1),
    (6, 7): ("C4", 7, 2),
    (5, 7): ("C4", 7, 3),
    (4, 7): ("C4", 7, 4),
    (3, 7): ("C4", 7, 5),
    (2, 7): ("C4", 7, 6),
    (1, 7): ("C4", 7, 7),
    # Données du Canal 5
    (8, 8): ("C5", 0, 0),
    (7, 8): ("C5", 0, 1),
    (6, 8): ("C5", 0, 2),
    (5, 8): ("C5", 0, 3),
    (4, 8): ("C5", 0, 4),
    (3, 8): ("C5", 0, 5),
    (2, 8): ("C5", 0, 6),
    (1, 8): ("C5", 0, 7),
    (0, 8): ("C5", 2, 0),
    (0, 7): ("C5", 1, 7),
    (0, 6): ("C5", 1, 6),
    (0, 5): ("C5", 1, 5),
    (0, 4): ("C5", 1, 4),
    (0, 3): ("C5", 1, 3),
    (0, 2): ("C5", 1, 2),
    (0, 1): ("C5", 1, 1),
    (0, 0): ("C5", 1, 0),
}
# --- FIN DE LA TABLE DE TRADUCTION ---


def setup_hardware():
    """Initialise tout le matériel I2C et renvoie les objets clés."""
    print("=== Initialisation du matériel de l'échiquier ===")
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        tca = adafruit_tca9548a.TCA9548A(i2c, address=0x72)

        print("   - Init LEDs C4 (0x70) et C5 (0x71)...")
        display_matrix = Matrix16x8(tca[4], address=0x70)
        display_row = Matrix16x8(tca[5], address=0x71)
        display_matrix.brightness = 0.2
        display_row.brightness = 0.2
        display_matrix.fill(0)
        display_row.fill(0)
        display_matrix.show()
        display_row.show()

        print("   - Init Capteurs C0, C1, C2, C3 (0x20)...")
        mcps = [
            MCP23017(tca[0], address=0x20),
            MCP23017(tca[1], address=0x20),
            MCP23017(tca[2], address=0x20),
            MCP23017(tca[3], address=0x20),
        ]

        sensor_pins = [[None for _ in range(8)] for _ in range(8)]
        pin_map_a = [0, 8, 1, 9, 2, 10, 3, 11]
        pin_map_b = [15, 7, 14, 6, 13, 5, 12, 4]

        for row in range(4):
            for col in range(8):
                sensor_pins[row * 2][col] = mcps[row].get_pin(pin_map_a[col])
                sensor_pins[row * 2 + 1][col] = mcps[row].get_pin(pin_map_b[col])

        for row in range(8):
            for col in range(8):
                pin = sensor_pins[row][col]
                pin.direction = digitalio.Direction.INPUT
                pin.pull = digitalio.Pull.UP

        print("=== Matériel OK ===")
        return display_matrix, display_row, sensor_pins

    except Exception as e:
        print(f"\nERREUR MATÉRIELLE FATALE: {e}")
        if "[Errno 121]" in str(e):
            print(">>> CONSEIL: Vérifiez les connexions I2C (SDA/SCL) et l'alimentation des périphériques.")
            print(">>> Vérifiez que l'adresse 0x70 (Matrix) est bien détectée (i2cdetect -y 1).")
        sys.exit(1)


def setup_ai():
    """Configure l'IA maison avec le niveau choisi."""
    print("\n--- Configuration de l'IA ---")
    try:
        elo_input = input(
            "Choisissez le niveau ELO (Débutant=600, Club=1500, Expert=2000) : "
        )
        elo = int(elo_input)
    except ValueError:
        elo = 1500
        print("Valeur invalide, défaut réglé sur 1500.")

    print(f"Chargement du moteur IA (ELO {elo})...")
    brain = Engine()
    searcher = Searcher(brain)
    searcher.set_elo(elo)
    return searcher


def get_state_from_sensors(sensor_pins):
    """Lit l'état physique (True = Pièce présente)."""
    state = [[False for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for c in range(8):
            state[r][c] = not sensor_pins[r][c].value
    return state


def get_initial_state(sensor_pins):
    """Attend la position de départ (avec gestion capteur d7 défectueux)."""
    print("\033[H\033[J", end="")
    print("MISE EN PLACE : Placez les pièces.")

    FAULTY_SQUARE = (6, 3)  # d7

    while True:
        state = get_state_from_sensors(sensor_pins)
        count = 0
        is_pos_correct = True
        errors_found = []

        # Affichage ASCII basique
        print("\033[H", end="")
        print(f"En attente... (Pièces: {count})")

        for r in range(8):
            for c in range(8):
                if state[r][c]:
                    count += 1
                # Vérification basique (rangées 0,1,6,7 pleines)
                expected = r in [0, 1, 6, 7]
                if state[r][c] != expected:
                    if (r, c) == FAULTY_SQUARE and expected and not state[r][c]:
                        pass  # On ignore d7 manquant
                    else:
                        is_pos_correct = False

        # Conditions de validation
        # 1. Tout est parfait (32 pièces)
        if count == 32 and is_pos_correct:
            print("\nPosition OK !")
            return state

        # 2. Dérogation d7 (31 pièces, seul d7 manque)
        if count == 31 and not state[6][3]:
            # On vérifie que le reste est bon
            temp_correct = True
            for r in range(8):
                for c in range(8):
                    if (r, c) != FAULTY_SQUARE:
                        expected = r in [0, 1, 6, 7]
                        if state[r][c] != expected:
                            temp_correct = False

            if temp_correct:
                print("\nPosition OK (Dérogation d7 active) !")
                state[6][3] = True  # On force l'état logique à True
                return state

        time.sleep(0.5)


def wait_for_state(sensor_pins, target_state):
    """Pause tant que l'échiquier ne correspond pas à l'état cible."""
    print(">> Veuillez corriger la position des pièces <<")
    FAULTY_SQUARE = (6, 3)

    while True:
        current = get_state_from_sensors(sensor_pins)
        if target_state[6][3]:
            current[6][3] = True  # Patch d7

        if current == target_state:
            return
        time.sleep(0.5)


def find_diffs(old_state, new_state):
    lifted, placed = [], []
    for r in range(8):
        for c in range(8):
            if old_state[r][c] and not new_state[r][c]:
                lifted.append((r, c))
            elif not old_state[r][c] and new_state[r][c]:
                placed.append((r, c))
    return lifted, placed


def set_square_leds(r, c, state, display_matrix, display_row):
    """Contrôle LED pour une case donnée."""
    # Conversion coords (0,0=a1) vers mapping physique
    target_col = 7 - c
    target_row = 7 - r

    corners = [
        (target_col, target_row),
        (target_col + 1, target_row),
        (target_col, target_row + 1),
        (target_col + 1, target_row + 1),
    ]

    for px, py in corners:
        if (px, py) in LED_MAP:
            canal, sw_col, sw_row = LED_MAP[(px, py)]
            if canal == "C4":
                display_matrix.pixel(sw_col, sw_row, state)
            elif canal == "C5":
                display_row.pixel(sw_col, sw_row, state)


def highlight_move(display_matrix, display_row, move):
    """Allume les LEDs pour un coup donné (départ et arrivée)."""
    display_matrix.fill(0)
    display_row.fill(0)

    if move:
        # Case Départ
        r1, c1 = move.from_square // 8, move.from_square % 8
        set_square_leds(r1, c1, 1, display_matrix, display_row)

        # Case Arrivée
        r2, c2 = move.to_square // 8, move.to_square % 8
        set_square_leds(r2, c2, 1, display_matrix, display_row)

    display_matrix.show()
    display_row.show()


def build_move_from_coords(from_coords, to_coords, board_logic):
    """Convertit coords physiques en chess.Move."""
    from_sq = chess.square(from_coords[1], from_coords[0])
    to_sq = chess.square(to_coords[1], to_coords[0])
    move = chess.Move(from_sq, to_sq)

    # Promotion automatique en Dame
    p = board_logic.piece_at(from_sq)
    if p and p.piece_type == chess.PAWN:
        if (p.color == chess.WHITE and to_coords[0] == 7) or (
            p.color == chess.BLACK and to_coords[0] == 0
        ):
            move.promotion = chess.QUEEN
    return move


def main():
    display_matrix, display_row, sensor_pins = setup_hardware()
    ai_engine = setup_ai()

    game_board = chess.Board()

    # État initial
    last_known_state = get_initial_state(sensor_pins)
    FAULTY_SQUARE = (6, 3)

    print("\n=== LA PARTIE COMMENCE ===")
    print("Vous jouez les Blancs.")

    try:
        while not game_board.is_game_over():
            print("-" * 30)
            print(game_board)

            # --- TOUR DE L'IA (NOIRS) ---
            if game_board.turn == chess.BLACK:
                print("\n>> L'IA réfléchit...")

                # Calcul du coup
                ai_move = ai_engine.get_best_move(
                    game_board, time_limit=DEFAULT_AI_TIME
                )

                if ai_move:
                    print(f">> L'IA joue : {ai_move.uci()}")
                    print(">> Veuillez déplacer la pièce noire indiquée par les LEDs.")

                    # Indication visuelle
                    highlight_move(display_matrix, display_row, ai_move)
                else:
                    print("ERREUR IA: Aucun coup trouvé (Pat/Mat ?)")
                    break

            # --- ATTENTE D'ACTION PHYSIQUE (Humain ou Exécution IA) ---
            if game_board.turn == chess.WHITE:
                print("\n>> À vous de jouer (Blancs)...")

            # Boucle de détection de mouvement
            move_detected = None

            while not move_detected:
                current_state = get_state_from_sensors(sensor_pins)

                # Patch d7
                if last_known_state[6][3]:
                    current_state[6][3] = True

                if current_state == last_known_state:
                    time.sleep(0.1)
                    continue

                # Stabilisation
                time.sleep(0.5)
                stable_state = get_state_from_sensors(sensor_pins)

                lifted, placed = find_diffs(last_known_state, stable_state)

                # Patch d7 (ignorer soulèvement fantôme)
                if FAULTY_SQUARE in lifted and len(lifted) > 1:
                    lifted.remove(FAULTY_SQUARE)

                # Analyse mouvement
                if len(lifted) == 1 and len(placed) == 1:
                    move = build_move_from_coords(lifted[0], placed[0], game_board)

                    # Validation
                    if move in game_board.legal_moves:
                        game_board.push(move)
                        print(f"Coup validé: {move.uci()}")

                        # Mise à jour LEDs pour confirmer le coup joué
                        highlight_move(display_matrix, display_row, move)

                        # Mise à jour état de référence
                        last_known_state = stable_state
                        # Patch d7 logique
                        last_known_state[6][3] = (
                            True if game_board.piece_at(chess.D7) else False
                        )

                        move_detected = True
                    else:
                        print(f"Coup illégal détecté ! ({move.uci()})")
                        wait_for_state(sensor_pins, last_known_state)

                # Gestion Roque, Prise en passant, Capture (Simplifié ici)
                elif len(lifted) == 2 and len(placed) == 1:  # Capture probable
                    # Logique simplifiée : on assume que la pièce placée est l'attaquant
                    attacker = placed[0]
                    if attacker in lifted:  # Erreur logic
                        pass
                    else:
                        # Trouver d'où vient l'attaquant
                        # Dans (lifted A, lifted B), l'un est 'placed'. Non, placed est l'arrivée.
                        # Lifted contient [Depart_Attaquant, Case_Capturée].
                        # Placed contient [Case_Capturée].
                        if placed[0] in lifted:
                            # C'est une capture standard
                            capturor_start = [x for x in lifted if x != placed[0]][0]
                            move = build_move_from_coords(
                                capturor_start, placed[0], game_board
                            )
                            if move in game_board.legal_moves:
                                game_board.push(move)
                                print(f"Capture validée: {move.uci()}")
                                highlight_move(display_matrix, display_row, move)
                                last_known_state = stable_state
                                last_known_state[6][3] = (
                                    True if game_board.piece_at(chess.D7) else False
                                )
                                move_detected = True
                            else:
                                print("Capture illégale.")
                                wait_for_state(sensor_pins, last_known_state)
                else:
                    # Mouvement complexe ou Roque
                    # Pour le roque, python-chess gère la logique, mais la détection physique est 2 lift/2 place
                    # À ce stade, pour le RPi, on demande de remettre en place si c'est trop confus
                    if len(lifted) == 2 and len(placed) == 2:
                        # Roque ?
                        # On vérifie si un coup de roque légal correspond
                        # C'est complexe à coder en détection pure sans ambiguité
                        # Pour l'instant on tente de resynchroniser
                        print("Mouvement complexe (Roque ?). Vérification...")
                        # On triche un peu : on regarde si l'état correspond à un coup légal possible
                        found_legal = False
                        for leg_move in game_board.legal_moves:
                            if game_board.is_castling(leg_move):
                                # On simule
                                game_board.push(leg_move)
                                # On regarde si ça colle (Note: c'est une approx, faudrait lier state->fen)
                                # Ici on accepte si le user a fait le mouvement correct du roi
                                # Simplification : on demande juste de valider le coup via Roi
                                pass

                        # Si trop compliqué, reset
                        print("Action non reconnue. Reset.")
                        wait_for_state(sensor_pins, last_known_state)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nArrêt de la partie.")
        display_matrix.fill(0)
        display_matrix.show()
        display_row.fill(0)
        display_row.show()


if __name__ == "__main__":
    main()
