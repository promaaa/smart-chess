#!/usr/bin/env python3
"""
Script de suivi de partie d'échecs en temps réel.
Utilise le matériel de l'échiquier intelligent I2C
et la bibliothèque python-chess pour valider les mouvements.

Illumine le dernier coup joué (case de départ et case d'arrivée).

MODIFICATION (17/11/2025): Ajout d'une dérogation pour un capteur
défectueux connu sur la case d7 (r=6, c=3) afin de permettre
le démarrage de la partie.

MODIFICATION (17/11/2025): Ajout de l'intégration de Stockfish
pour suggérer les coups des Noirs.

MODIFICATION (17/11/2025): Correction du bug "Mouvement non reconnu"
causé par la dérogation du capteur d7.
"""

import os
import time

import adafruit_tca9548a
import board
import busio
import chess  # Nécessite 'pip install chess'
import digitalio
from adafruit_ht16k33.matrix import Matrix16x8
from adafruit_mcp230xx.mcp23017 import MCP23017

try:
    from stockfish import Stockfish

    STOCKFISH_AVAILABLE = True
except ImportError:
    print("ATTENTION: La bibliothèque 'stockfish' n'est pas installée.")
    print("Veuillez l'installer avec : pip install stockfish")
    STOCKFISH_AVAILABLE = False


# --- DÉBUT DE LA TABLE DE TRADUCTION (LED_MAP) ---
LED_MAP = {
    # Données du Canal 4 (Matrice 8x8)
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

    # 1. Bus I2C
    i2c = busio.I2C(board.SCL, board.SDA)

    # 2. Multiplexeur
    tca = adafruit_tca9548a.TCA9548A(i2c, address=0x72)

    # 3. Contrôleurs LED
    print("   - Init LEDs C4 (0x70) et C5 (0x71)...")
    # L'ERREUR TRACEBACK SE PRODUIT ICI (PROBLÈME DE CONNEXION)
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


def setup_stockfish():
    """Tente d'initialiser Stockfish."""
    if not STOCKFISH_AVAILABLE:
        print("Stockfish non disponible (bibliothèque manquante).")
        return None

    stockfish_path = "/usr/games/stockfish"

    if not os.path.exists(stockfish_path):
        print(f"ERREUR: Exécutable Stockfish non trouvé à {stockfish_path}")
        print("Veuillez l'installer avec : sudo apt install stockfish")
        return None

    try:
        stockfish = Stockfish(path=stockfish_path)
        stockfish.set_skill_level(5)  # Niveau 0-20
        print(f"Stockfish initialisé (Niveau: {stockfish.get_skill_level()})")
        return stockfish
    except Exception as e:
        print(f"Erreur lors de l'initialisation de Stockfish: {e}")
        return None


def get_state_from_sensors(sensor_pins):
    """
    Lit les 64 capteurs et renvoie un état 8x8 (True = pièce présente).
    (row, col)
    """
    state = [[False for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for c in range(8):
            state[r][c] = not sensor_pins[r][c].value
    return state


def get_initial_state(sensor_pins):
    """
    Attend que l'utilisateur place les 32 pièces de départ.
    Affiche un mini-échiquier de détection.
    Accepte une dérogation pour le capteur d7 défectueux.
    """

    print("\033[H\033[J", end="")
    print("Veuillez placer les pièces en position de départ.")
    print("Le script démarrera lorsque les 32 pièces seront détectées.")
    print("\nLégende:")
    print(" [X] = Pièce détectée (Correct)")
    print(" [?] = Pièce manquante (Incorrect)")
    print(" [ ] = Vide (Correct)")
    print(" [!] = Pièce extra (Incorrect)")
    print("-" * 40)

    # Coordonnées du capteur défectueux (rangée 6, colonne 3) = d7
    FAULTY_SQUARE = (6, 3)

    while True:
        state = get_state_from_sensors(sensor_pins)
        count = 0
        is_pos_correct = True

        board_display = ""
        errors_found = []

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

        # --- Impression de la sortie ---

        print("\033[H", end="")
        print("Veuillez placer les pièces en position de départ.")
        print("Le script démarrera lorsque les 32 pièces seront détectées.")
        print("\nLégende:")
        print(" [X] = Pièce détectée (Correct)")
        print(" [?] = Pièce manquante (Incorrect)")
        print(" [ ] = Vide (Correct)")
        print(" [!] = Pièce extra (Incorrect)")
        print("-" * 40)
        print(board_display, end="")
        print(f"Pièces détectées: {count}/32")

        # Logique de dérogation
        faulty_square_is_the_only_error = False
        if not is_pos_correct:
            if (
                len(errors_found) == 1
                and errors_found[0][0] == FAULTY_SQUARE
                and errors_found[0][1] == "missing"
            ):
                print("ATTENTION: Capteur d7 défectueux détecté (ignoré).")
                faulty_square_is_the_only_error = True
            else:
                print("ATTENTION: Des pièces sont mal placées ou manquantes!")
        else:
            print("                                                         ")

        # Condition de sortie (Normale)
        if count == 32 and is_pos_correct:
            print("\nPosition de départ confirmée!")
            print("                                                         ")
            return state

        # Condition de sortie (Dérogation capteur d7)
        if count == 31 and faulty_square_is_the_only_error:
            print("\nPosition de départ confirmée (avec dérogation d7)!")
            print("                                                         ")
            state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]] = True
            return state

        time.sleep(0.5)


def wait_for_state(sensor_pins, target_state):
    """Met le jeu en pause jusqu'à ce que l'échiquier physique corresponde à target_state."""
    print("\nERREUR: Coup illégal ou état invalide.")
    print("Veuillez remettre les pièces à la position précédente.")

    while True:
        current_state = get_state_from_sensors(sensor_pins)

        FAULTY_SQUARE = (6, 3)  # d7
        if target_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]] == True:
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
                lifted.append((r, c))  # (row, col)
            elif not old_state[r][c] and new_state[r][c]:
                placed.append((r, c))  # (row, col)
    return lifted, placed


def set_square_leds(sensor_row, sensor_col, state, display_matrix, display_row):
    """Allume (state=1) ou éteint (state=0) les 4 LEDs d'une case capteur."""
    target_col = 7 - sensor_col
    target_row = 7 - sensor_row

    physical_corners = [
        (target_col, target_row),
        (target_col + 1, target_row),
        (target_col, target_row + 1),
        (target_col + 1, target_row + 1),
    ]

    for px, py in physical_corners:
        if (px, py) in LED_MAP:
            canal, sw_col, sw_row = LED_MAP[(px, py)]

            if canal == "C4":
                display_matrix.pixel(sw_col, sw_row, state)
            elif canal == "C5":
                display_row.pixel(sw_col, sw_row, state)


def update_leds(display_matrix, display_row, from_coords, to_coords):
    """Éteint tout et allume les cases du dernier coup."""
    display_matrix.fill(0)
    display_row.fill(0)

    if from_coords:
        set_square_leds(from_coords[0], from_coords[1], 1, display_matrix, display_row)
    if to_coords:
        set_square_leds(to_coords[0], to_coords[1], 1, display_matrix, display_row)

    display_matrix.show()
    display_row.show()


def build_move(from_coords, to_coords, game_board):
    """Construit un objet chess.Move à partir des coordonnées (row, col)."""
    from_sq = chess.square(from_coords[1], from_coords[0])
    to_sq = chess.square(to_coords[1], to_coords[0])

    move = chess.Move(from_sq, to_sq)

    # Gérer la promotion (on suppose une Dame par défaut)
    piece = game_board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN:
        if (
            piece.color == chess.WHITE and from_coords[0] == 6 and to_coords[0] == 7
        ) or (piece.color == chess.BLACK and from_coords[0] == 1 and to_coords[0] == 0):
            move.promotion = chess.QUEEN

    return move


def main():
    """Fonction principale du jeu."""
    # Gestion de l'erreur matérielle I2C
    try:
        display_matrix, display_row, sensor_pins = setup_hardware()
    except Exception as e:
        print("\n\nERREUR FATALE LORS DE L'INITIALISATION MATÉRIELLE")
        print("Veuillez vérifier vos connexions I2C.")
        print(
            "Concerne probablement le multiplexeur (TCA) ou un contrôleur LED (HT16K33)."
        )
        print(f"Détail de l'erreur: {e}")
        return  # Arrête le script

    stockfish = setup_stockfish()

    game_board = chess.Board()

    last_known_physical_state = get_initial_state(sensor_pins)

    last_move_coords = (None, None)

    FAULTY_SQUARE = (6, 3)

    try:
        while not game_board.is_game_over():
            # 1. Afficher l'état actuel
            print("\033[H\033[J", end="")
            print("--- Échiquier Logique ---")
            print(game_board)
            print("-------------------------")
            turn_color = "Blancs" if game_board.turn == chess.WHITE else "Noirs"
            print(f"Tour: {turn_color}. En attente du coup...")

            # 2. Attendre un changement d'état
            current_state = get_state_from_sensors(sensor_pins)

            # Patch anti-dérive pour le capteur d7
            if last_known_physical_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]] == True:
                current_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]] = True

            if current_state == last_known_physical_state:
                time.sleep(0.1)
                continue

            # 3. Changement détecté !
            print("Changement détecté. Stabilisation...")
            time.sleep(0.75)
            stable_state = get_state_from_sensors(sensor_pins)

            # 4. Analyser le changement
            lifted, placed = find_diffs(last_known_physical_state, stable_state)

            # --- DÉBUT DE LA CORRECTION ---
            # Si le capteur d7 apparaît comme "soulevé" (ce qu'il fait toujours),
            # on le retire de la liste des pièces soulevées,
            # SAUF si c'est la SEULE pièce soulevée.
            if (
                FAULTY_SQUARE in lifted
                and last_known_physical_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]]
                == True
            ):
                # S'il y a d'autres pièces soulevées (ex: [e2, d7]),
                # on retire d7 car c'est un fantôme.
                if len(lifted) > 1:
                    print(f"Note: Soulèvement fantôme de {FAULTY_SQUARE} (d7) ignoré.")
                    lifted.remove(FAULTY_SQUARE)

            # --- FIN DE LA CORRECTION ---

            move = None
            from_coords, to_coords = None, None

            # --- Logique de détection de coup ---

            # Cas 1: Mouvement simple
            if len(lifted) == 1 and len(placed) == 1:
                from_coords = lifted[0]
                to_coords = placed[0]
                move = build_move(from_coords, to_coords, game_board)
                print(
                    f"Détecté: Mouvement simple {chess.SQUARE_NAMES[move.from_square]} -> {chess.SQUARE_NAMES[move.to_square]}"
                )

            # Cas 2: Capture
            elif len(lifted) == 2 and len(placed) == 1:
                to_coords = placed[0]
                if to_coords in lifted:
                    from_coords = [sq for sq in lifted if sq != to_coords][0]
                    move = build_move(from_coords, to_coords, game_board)
                    print(
                        f"Détecté: Capture {chess.SQUARE_NAMES[move.from_square]} -> {chess.SQUARE_NAMES[move.to_square]}"
                    )
                else:
                    print(
                        "Erreur de capture: la pièce n'a pas atterri sur la case capturée."
                    )

            # Cas 3: Roque
            elif len(lifted) == 2 and len(placed) == 2:
                king_sq = game_board.king(game_board.turn)
                king_start_coords = (
                    chess.square_rank(king_sq),
                    chess.square_file(king_sq),
                )

                if king_start_coords in lifted:
                    king_end_coords = None
                    for sq in placed:
                        if sq[1] == 2 or sq[1] == 6:
                            king_end_coords = sq
                            break

                    if king_end_coords:
                        from_coords = king_start_coords
                        to_coords = king_end_coords
                        move = build_move(from_coords, to_coords, game_board)
                        print(
                            f"Détecté: Mouvement du Roi (Roque?) {chess.SQUARE_NAMES[move.from_square]} -> {chess.SQUARE_NAMES[move.to_square]}"
                        )
                else:
                    print("Erreur de roque: Roi non déplacé.")

            # --- Validation et mise à jour ---

            if move and move in game_board.legal_moves:
                print(f"COUP VALIDE: {move.uci()}")
                game_board.push(move)

                # Gestion de l'état avec capteur défectueux
                last_known_physical_state = stable_state
                d7_square = chess.D7
                d7_coords = (chess.square_rank(d7_square), chess.square_file(d7_square))

                if game_board.piece_at(d7_square):
                    last_known_physical_state[d7_coords[0]][d7_coords[1]] = True
                else:
                    last_known_physical_state[d7_coords[0]][d7_coords[1]] = False

                last_move_coords = (from_coords, to_coords)

                # Suggestion Stockfish
                if (
                    stockfish
                    and game_board.turn == chess.BLACK
                    and not game_board.is_game_over()
                ):
                    print("\nStockfish analyse...")
                    stockfish.set_fen_position(game_board.fen())
                    best_move_uci = stockfish.get_best_move()
                    print(f"**************************************************")
                    print(
                        f"*** Suggestion Stockfish pour les Noirs: {best_move_uci} ***"
                    )
                    print(f"**************************************************")

            elif move:
                print(f"COUP ILLÉGAL: {move.uci()}")
                wait_for_state(sensor_pins, last_known_physical_state)

            else:
                print(
                    f"Mouvement non reconnu (lifted: {len(lifted)}, placed: {len(placed)}). Réinitialisation de l'état..."
                )
                wait_for_state(sensor_pins, last_known_physical_state)

            # 5. Mettre à jour les LEDs
            update_leds(
                display_matrix, display_row, last_move_coords[0], last_move_coords[1]
            )

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n=== Arrêt du programme ===")

    finally:
        # Éteindre toutes les LEDs avant de quitter
        print("Extinction des LEDs...")
        # On vérifie si les objets existent au cas où le setup aurait crashé
        if "display_matrix" in locals():
            display_matrix.fill(0)
            display_matrix.show()
        if "display_row" in locals():
            display_row.fill(0)
            display_row.show()
        print("Terminé.")


if __name__ == "__main__":
    main()
