#!/usr/bin/env python3
"""
Script de jeu d'échecs avec l'IA-Marc V2 sur matériel réel.
===========================================================

Basé sur chess_game_SF.py, mais utilise le moteur IA-Marc V2
au lieu de Stockfish.

Fonctionnalités :
- Détection des mouvements via capteurs Hall (MCP23017)
- Affichage des coups via LEDs (HT16K33)
- Intégration de l'IA-Marc V2 pour les Noirs
- Gestion du capteur défectueux d7
"""

import os
import sys
import time
import traceback

# Ajout du chemin courant pour les imports locaux si nécessaire
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import adafruit_tca9548a
import board
import busio
import chess
import digitalio
from adafruit_ht16k33.matrix import Matrix16x8
from adafruit_mcp230xx.mcp23017 import MCP23017

# Import de la nouvelle IA
try:
    from engine_main import ChessEngine
    NEW_AI_AVAILABLE = True
except ImportError as e:
    print(f"ERREUR CRITIQUE: Impossible d'importer l'IA-Marc V2: {e}")
    NEW_AI_AVAILABLE = False


# --- DÉBUT DE LA TABLE DE TRADUCTION (LED_MAP) ---
# Copié depuis chess_game_SF.py
LED_MAP = {
    # Données du Canal 4 (Matrice 8x8)
    (8, 0): ("C4", 0, 0), (7, 0): ("C4", 0, 1), (6, 0): ("C4", 0, 2), (5, 0): ("C4", 0, 3),
    (4, 0): ("C4", 0, 4), (3, 0): ("C4", 0, 5), (2, 0): ("C4", 0, 6), (1, 0): ("C4", 0, 7),
    (8, 1): ("C4", 1, 0), (7, 1): ("C4", 1, 1), (6, 1): ("C4", 1, 2), (5, 1): ("C4", 1, 3),
    (4, 1): ("C4", 1, 4), (3, 1): ("C4", 1, 5), (2, 1): ("C4", 1, 6), (1, 1): ("C4", 1, 7),
    (0, 1): ("C4", 8, 1),
    (8, 2): ("C4", 2, 0), (7, 2): ("C4", 2, 1), (6, 2): ("C4", 2, 2), (5, 2): ("C4", 2, 3),
    (4, 2): ("C4", 2, 4), (3, 2): ("C4", 2, 5), (2, 2): ("C4", 2, 6), (1, 2): ("C4", 2, 7),
    (8, 3): ("C4", 3, 0), (7, 3): ("C4", 3, 1), (6, 3): ("C4", 3, 2), (5, 3): ("C4", 3, 3),
    (4, 3): ("C4", 3, 4), (3, 3): ("C4", 3, 5), (2, 3): ("C4", 3, 6), (1, 3): ("C4", 3, 7),
    (8, 4): ("C4", 4, 0), (7, 4): ("C4", 4, 1), (6, 4): ("C4", 4, 2), (5, 4): ("C4", 4, 3),
    (4, 4): ("C4", 4, 4), (3, 4): ("C4", 4, 5), (2, 4): ("C4", 4, 6), (1, 4): ("C4", 4, 7),
    (8, 5): ("C4", 5, 0), (7, 5): ("C4", 5, 1), (6, 5): ("C4", 5, 2), (5, 5): ("C4", 5, 3),
    (4, 5): ("C4", 5, 4), (3, 5): ("C4", 5, 5), (2, 5): ("C4", 5, 6), (1, 5): ("C4", 5, 7),
    (8, 6): ("C4", 6, 0), (7, 6): ("C4", 6, 1), (6, 6): ("C4", 6, 2), (5, 6): ("C4", 6, 3),
    (4, 6): ("C4", 6, 4), (3, 6): ("C4", 6, 5), (2, 6): ("C4", 6, 6), (1, 6): ("C4", 6, 7),
    (8, 7): ("C4", 7, 0), (7, 7): ("C4", 7, 1), (6, 7): ("C4", 7, 2), (5, 7): ("C4", 7, 3),
    (4, 7): ("C4", 7, 4), (3, 7): ("C4", 7, 5), (2, 7): ("C4", 7, 6), (1, 7): ("C4", 7, 7),
    # Données du Canal 5
    (8, 8): ("C5", 0, 0), (7, 8): ("C5", 0, 1), (6, 8): ("C5", 0, 2), (5, 8): ("C5", 0, 3),
    (4, 8): ("C5", 0, 4), (3, 8): ("C5", 0, 5), (2, 8): ("C5", 0, 6), (1, 8): ("C5", 0, 7),
    (0, 8): ("C5", 2, 0),
    (0, 7): ("C5", 1, 7), (0, 6): ("C5", 1, 6), (0, 5): ("C5", 1, 5), (0, 4): ("C5", 1, 4),
    (0, 3): ("C5", 1, 3), (0, 2): ("C5", 1, 2), (0, 1): ("C5", 1, 1), (0, 0): ("C5", 1, 0),
}
# --- FIN DE LA TABLE DE TRADUCTION ---


def setup_hardware():
    """Initialise tout le matériel I2C et renvoie les objets clés."""
    print("=== Initialisation du matériel de l'échiquier ===")

    try:
        # 1. Bus I2C
        i2c = busio.I2C(board.SCL, board.SDA)

        # 2. Multiplexeur
        tca = adafruit_tca9548a.TCA9548A(i2c, address=0x72)

        # 3. Contrôleurs LED
        print("   - Init LEDs C4 (0x70) et C5 (0x71)...")
        display_matrix = Matrix16x8(tca[4], address=0x70)
        display_row = Matrix16x8(tca[5], address=0x71)
        display_matrix.brightness = 0.2
        display_row.brightness = 0.2
        display_matrix.fill(0)
        display_row.fill(0)
        display_matrix.show()
        display_row.show()

        # 4. Contrôleurs MCP23017
        print("   - Init Capteurs C0, C1, C2, C3 (0x20)...")
        mcps = [
            MCP23017(tca[0], address=0x20),  # Rangées 1-2
            MCP23017(tca[1], address=0x20),  # Rangées 3-4
            MCP23017(tca[2], address=0x20),  # Rangées 5-6
            MCP23017(tca[3], address=0x20),  # Rangées 7-8
        ]

        # 5. Matrice des pins capteurs
        sensor_pins = [[None for _ in range(8)] for _ in range(8)]
        pin_map_a = [0, 8, 1, 9, 2, 10, 3, 11]
        pin_map_b = [15, 7, 14, 6, 13, 5, 12, 4]

        for row in range(4):  # 4 MCPs
            for col in range(8):
                sensor_pins[row * 2][col] = mcps[row].get_pin(pin_map_a[col])
                sensor_pins[row * 2 + 1][col] = mcps[row].get_pin(pin_map_b[col])

        # 6. Configuration des pins
        for row in range(8):
            for col in range(8):
                pin = sensor_pins[row][col]
                pin.direction = digitalio.Direction.INPUT
                pin.pull = digitalio.Pull.UP

        print("=== Matériel initialisé avec succès ===")
        return display_matrix, display_row, sensor_pins

    except Exception as e:
        print(f"\nERREUR MATÉRIELLE: {e}")
        traceback.print_exc()
        raise


def setup_new_ai():
    """Initialise l'IA-Marc V2."""
    if not NEW_AI_AVAILABLE:
        return None

    try:
        print("Initialisation de l'IA-Marc V2...")
        ai = ChessEngine()
        ai.set_level("Club")  # Niveau par défaut
        print(f"IA-Marc V2 prête (Niveau: Club)")
        return ai
    except Exception as e:
        print(f"Erreur lors de l'initialisation de l'IA: {e}")
        traceback.print_exc()
        return None


def get_state_from_sensors(sensor_pins):
    """
    Lit les 64 capteurs et renvoie un état 8x8 (True = pièce présente).
    (row, col)
    """
    state = [[False for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for c in range(8):
            # Lecture sécurisée
            try:
                state[r][c] = not sensor_pins[r][c].value
            except OSError:
                # En cas d'erreur I2C transitoire, on suppose l'état précédent ou False
                pass
    return state


def get_initial_state(sensor_pins):
    """
    Attend que l'utilisateur place les 32 pièces de départ.
    """
    print("\033[H\033[J", end="")
    print("Veuillez placer les pièces en position de départ.")
    print("Le script démarrera lorsque les 32 pièces seront détectées.")
    
    FAULTY_SQUARE = (6, 3)  # d7

    while True:
        state = get_state_from_sensors(sensor_pins)
        count = 0
        is_pos_correct = True
        errors_found = []

        board_display = ""
        for r in range(7, -1, -1):
            board_display += f" {r + 1} |"
            for c in range(8):
                is_present = state[r][c]
                is_expected = r in [0, 1, 6, 7]

                if is_present:
                    count += 1
                    if is_expected:
                        board_display += "[X]"
                    else:
                        board_display += "[!]"
                        is_pos_correct = False
                        errors_found.append(((r, c), "extra"))
                else:
                    if is_expected:
                        board_display += "[?]"
                        is_pos_correct = False
                        errors_found.append(((r, c), "missing"))
                    else:
                        board_display += "[ ]"
            board_display += "\n"

        board_display += "   " + "-" * 33 + "\n"
        board_display += "     a  b  c  d  e  f  g  h\n"

        print("\033[H", end="")
        print("Veuillez placer les pièces en position de départ.")
        print(board_display, end="")
        print(f"Pièces détectées: {count}/32")

        # Logique de dérogation d7
        faulty_square_is_the_only_error = False
        if not is_pos_correct:
            if (
                len(errors_found) == 1
                and errors_found[0][0] == FAULTY_SQUARE
                and errors_found[0][1] == "missing"
            ):
                print("ATTENTION: Capteur d7 défectueux détecté (ignoré).")
                faulty_square_is_the_only_error = True

        if count == 32 and is_pos_correct:
            print("\nPosition de départ confirmée!")
            return state

        if count == 31 and faulty_square_is_the_only_error:
            print("\nPosition de départ confirmée (avec dérogation d7)!")
            state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]] = True
            return state

        time.sleep(0.5)


def wait_for_state(sensor_pins, target_state):
    """Met le jeu en pause jusqu'à ce que l'échiquier physique corresponde à target_state."""
    print("\nERREUR: Coup illégal ou état invalide.")
    print("Veuillez remettre les pièces à la position précédente.")

    FAULTY_SQUARE = (6, 3)

    while True:
        current_state = get_state_from_sensors(sensor_pins)
        
        if target_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]]:
            current_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]] = True

        if current_state == target_state:
            print("Échiquier réinitialisé. Reprise de la partie.")
            return
        time.sleep(0.5)


def find_diffs(old_state, new_state):
    """Compare deux états et renvoie les cases soulevées et posées."""
    lifted = []
    placed = []
    for r in range(8):
        for c in range(8):
            if old_state[r][c] and not new_state[r][c]:
                lifted.append((r, c))
            elif not old_state[r][c] and new_state[r][c]:
                placed.append((r, c))
    return lifted, placed


def set_square_leds(sensor_row, sensor_col, state, display_matrix, display_row):
    """Allume ou éteint les LEDs d'une case."""
    target_col = 7 - sensor_col
    target_row = 7 - sensor_row
    physical_corners = [
        (target_col, target_row), (target_col + 1, target_row),
        (target_col, target_row + 1), (target_col + 1, target_row + 1),
    ]
    for px, py in physical_corners:
        if (px, py) in LED_MAP:
            canal, sw_col, sw_row = LED_MAP[(px, py)]
            if canal == "C4":
                display_matrix.pixel(sw_col, sw_row, state)
            elif canal == "C5":
                display_row.pixel(sw_col, sw_row, state)


def update_leds(display_matrix, display_row, from_coords, to_coords):
    """Met à jour les LEDs pour afficher le coup."""
    display_matrix.fill(0)
    display_row.fill(0)
    if from_coords:
        set_square_leds(from_coords[0], from_coords[1], 1, display_matrix, display_row)
    if to_coords:
        set_square_leds(to_coords[0], to_coords[1], 1, display_matrix, display_row)
    display_matrix.show()
    display_row.show()


def build_move(from_coords, to_coords, game_board):
    """Construit un objet chess.Move."""
    from_sq = chess.square(from_coords[1], from_coords[0])
    to_sq = chess.square(to_coords[1], to_coords[0])
    move = chess.Move(from_sq, to_sq)

    # Promotion automatique en Dame
    piece = game_board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN:
        if (piece.color == chess.WHITE and from_coords[0] == 6 and to_coords[0] == 7) or \
           (piece.color == chess.BLACK and from_coords[0] == 1 and to_coords[0] == 0):
            move.promotion = chess.QUEEN
    return move


def main():
    print("=== Démarrage de Chess Game IA-Marc V2 ===")
    
    # Init Matériel
    try:
        display_matrix, display_row, sensor_pins = setup_hardware()
    except Exception as e:
        print(f"Erreur fatale matériel: {e}")
        return

    # Init IA
    ai_engine = setup_new_ai()
    if not ai_engine:
        print("Attention: IA non disponible, mode 2 joueurs uniquement.")

    game_board = chess.Board()
    last_known_physical_state = get_initial_state(sensor_pins)
    last_move_coords = (None, None)
    FAULTY_SQUARE = (6, 3)

    try:
        while not game_board.is_game_over():
            # Affichage
            print("\033[H\033[J", end="")
            print("--- Échiquier Logique (IA-Marc V2) ---")
            print(game_board)
            print("--------------------------------------")
            turn_color = "Blancs" if game_board.turn == chess.WHITE else "Noirs"
            print(f"Tour: {turn_color}")

            # Si c'est le tour de l'IA (Noirs)
            if ai_engine and game_board.turn == chess.BLACK:
                print("\n>>> L'IA réfléchit...")
                start_time = time.time()
                
                # Recherche du meilleur coup
                # On utilise 2.0 secondes comme limite pour être réactif sur RPi5
                best_move = ai_engine.get_move(game_board, time_limit=2.0)
                
                duration = time.time() - start_time
                print(f">>> Coup trouvé: {best_move.uci()} en {duration:.2f}s")
                
                # Afficher le coup suggéré sur les LEDs pour que l'humain le joue
                from_sq = best_move.from_square
                to_sq = best_move.to_square
                
                ai_from_coords = (chess.square_rank(from_sq), chess.square_file(from_sq))
                ai_to_coords = (chess.square_rank(to_sq), chess.square_file(to_sq))
                
                print(f"JOUEZ CE COUP: {best_move.uci()}")
                update_leds(display_matrix, display_row, ai_from_coords, ai_to_coords)
            else:
                print("En attente de votre coup...")

            # Boucle de détection physique
            while True:
                current_state = get_state_from_sensors(sensor_pins)
                
                # Patch d7
                if last_known_physical_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]]:
                    current_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]] = True

                if current_state != last_known_physical_state:
                    print("Mouvement détecté...")
                    time.sleep(0.75) # Debounce
                    stable_state = get_state_from_sensors(sensor_pins)
                    
                    lifted, placed = find_diffs(last_known_physical_state, stable_state)
                    
                    # Patch d7 fantôme
                    if FAULTY_SQUARE in lifted and len(lifted) > 1:
                         if last_known_physical_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]]:
                             lifted.remove(FAULTY_SQUARE)

                    # Analyse du coup
                    move = None
                    from_coords, to_coords = None, None

                    # Mouvement simple
                    if len(lifted) == 1 and len(placed) == 1:
                        from_coords = lifted[0]
                        to_coords = placed[0]
                        move = build_move(from_coords, to_coords, game_board)

                    # Capture
                    elif len(lifted) == 2 and len(placed) == 1:
                        to_coords = placed[0]
                        if to_coords in lifted:
                            from_coords = [sq for sq in lifted if sq != to_coords][0]
                            move = build_move(from_coords, to_coords, game_board)

                    # Roque
                    elif len(lifted) == 2 and len(placed) == 2:
                         king_sq = game_board.king(game_board.turn)
                         king_coords = (chess.square_rank(king_sq), chess.square_file(king_sq))
                         if king_coords in lifted:
                             # Trouver la destination du roi (colonne 2 ou 6)
                             for sq in placed:
                                 if sq[1] in [2, 6]:
                                     from_coords = king_coords
                                     to_coords = sq
                                     move = build_move(from_coords, to_coords, game_board)
                                     break

                    # Validation
                    if move and move in game_board.legal_moves:
                        print(f"Coup joué: {move.uci()}")
                        game_board.push(move)
                        
                        # Mise à jour état physique de référence
                        last_known_physical_state = stable_state
                        
                        # Mise à jour patch d7
                        if game_board.piece_at(chess.D7):
                            last_known_physical_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]] = True
                        else:
                            last_known_physical_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]] = False
                            
                        last_move_coords = (from_coords, to_coords)
                        update_leds(display_matrix, display_row, from_coords, to_coords)
                        break # Sortir de la boucle de détection pour passer au tour suivant
                    
                    elif move:
                        print(f"Coup illégal: {move.uci()}")
                        wait_for_state(sensor_pins, last_known_physical_state)
                    else:
                        print("Mouvement non reconnu.")
                        wait_for_state(sensor_pins, last_known_physical_state)
                
                time.sleep(0.1)

        print(f"Partie terminée! Résultat: {game_board.result()}")

    except KeyboardInterrupt:
        print("\nArrêt utilisateur.")
    finally:
        print("Extinction.")
        if 'display_matrix' in locals(): display_matrix.fill(0); display_matrix.show()
        if 'display_row' in locals(): display_row.fill(0); display_row.show()

if __name__ == "__main__":
    main()
