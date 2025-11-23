
#!/usr/bin/env python3
"""
Script de jeu d'échecs avec l'IA-Marc V2 sur matériel réel + Menu sur écran 2.8".
=================================================================================
Mise à jour Finale : Alignement strict sur le code de test "screen_28.py" validé.

Fonctionnalités :
- Menu graphique pour choisir le niveau de l'IA.
- Affichage en temps réel du statut de la partie (Tour, Coup, Echec).
- Navigation via encodeur rotatif.
- Détection plateau + LEDs.
"""

import os
import sys
import time
import traceback
import threading

# Ajout du chemin courant pour les imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Imports Matériel Échiquier ---
import adafruit_tca9548a
import board
import busio
import digitalio
import chess
from adafruit_ht16k33.matrix import Matrix16x8
from adafruit_mcp230xx.mcp23017 import MCP23017

# --- Imports Matériel Menu (Écran + Encodeur) ---
try:
    from PIL import Image, ImageDraw, ImageFont
    import adafruit_rgb_display.ili9341 as ili9341
    from gpiozero import RotaryEncoder, Button
    MENU_HARDWARE_AVAILABLE = True
except ImportError as e:
    print(f"ATTENTION: Bibliothèques menu manquantes ({e}). Mode sans écran.")
    MENU_HARDWARE_AVAILABLE = False

# --- Import de l'IA ---
try:
    # Ajout du chemin vers le moteur IA (Racine du projet)
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../ia_marc/V2"))
    from engine_main import ChessEngine
    from engine_config import DIFFICULTY_LEVELS
    NEW_AI_AVAILABLE = True
except ImportError as e:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "../ia_marc/V2"))
        from engine_main import ChessEngine
        from engine_config import DIFFICULTY_LEVELS
        NEW_AI_AVAILABLE = True
    except ImportError as e2:
        print(f"ERREUR CRITIQUE: Impossible d'importer l'IA-Marc V2: {e} / {e2}")
        NEW_AI_AVAILABLE = False


# --- CONFIGURATION PINS (Confirmé fonctionnel) ---
# Écran SPI ILI9341
CS_PIN = digitalio.DigitalInOut(board.CE0)   # Pin 24
DC_PIN = digitalio.DigitalInOut(board.D25)   # Pin 22
RST_PIN = digitalio.DigitalInOut(board.D24)  # Pin 18 (Mapping regroupé)

# Vitesse standard (24MHz) validée par le test
BAUDRATE = 24000000

# Encodeur Rotatif (GPIO BCM)
ENC_CLK = 17
ENC_DT = 27
ENC_SW = 22

# --- FONCTION COULEUR (CORRECTION BGR) ---
# Votre écran est BGR, mais Pillow travaille en RGB.
# Cette fonction convertit (R, G, B) -> (B, G, R) pour l'affichage.
def color(r, g, b):
    return (b, g, r)

# Palette (Couleurs corrigées via la fonction)
C_BLACK  = color(0, 0, 0)
C_WHITE  = color(255, 255, 255)
C_RED    = color(255, 0, 0)
C_GREEN  = color(0, 255, 0)
C_BLUE   = color(0, 0, 255)
C_GRAY   = color(100, 100, 100)
C_ORANGE = color(255, 165, 0)


# --- DÉBUT DE LA TABLE DE TRADUCTION (LED_MAP) ---
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


class SmartDisplay:
    """Gère l'affichage complet (Menu + Jeu)."""
    
    def __init__(self):
        if not MENU_HARDWARE_AVAILABLE:
            print("Mode simulation écran (console)")
            return

        # Init SPI avec MISO (conforme au test)
        spi = busio.SPI(clock=board.SCK, MOSI=board.MOSI, MISO=board.MISO)
        
        # --- HARD RESET (Copie conforme du code de test) ---
        # Séquence indispensable pour démarrer l'écran à froid
        RST_PIN.direction = digitalio.Direction.OUTPUT
        RST_PIN.value = True
        time.sleep(0.1)
        RST_PIN.value = False
        time.sleep(0.1)
        RST_PIN.value = True
        time.sleep(0.1)
        
        # --- INITIALISATION ÉCRAN ---
        # Nous utilisons width=320, height=240 pour forcer le mode paysage
        # sans utiliser 'rotation=90' dans le constructeur (source de l'erreur précédente).
        self.disp = ili9341.ILI9341(
            spi, 
            cs=CS_PIN, 
            dc=DC_PIN, 
            rst=RST_PIN, 
            baudrate=BAUDRATE,
            width=320,
            height=240
        )
        
        # Init Encodeur
        self.encoder = RotaryEncoder(ENC_CLK, ENC_DT, max_steps=0)
        self.button = Button(ENC_SW)
        
        # Init Buffer Image
        # On utilise les dimensions natives rapportées par l'initialisation
        if self.disp.rotation % 180 == 90:
            self.width = self.disp.width
            self.height = self.disp.height
        else:
            self.width = self.disp.width
            self.height = self.disp.height
            
        print(f"Ecran initialisé : {self.width}x{self.height}")
        self.image = Image.new("RGB", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image)
        
        # Fonts
        try:
            self.font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            self.font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except IOError:
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

        # Menu Data
        self.levels = list(DIFFICULTY_LEVELS.keys())
        self.levels.sort()
        self.selected_index = 0
        
    def draw_menu(self):
        """Dessine le menu de sélection."""
        if not MENU_HARDWARE_AVAILABLE: return

        # Fond Noir
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=C_BLACK)
        
        # En-tête
        self.draw.rectangle((0, 0, self.width, 50), fill=C_BLUE)
        text_title = "SMART CHESS"
        bbox = self.draw.textbbox((0, 0), text_title, font=self.font_large)
        tw = bbox[2] - bbox[0]
        self.draw.text(((self.width - tw) // 2, 10), text_title, font=self.font_large, fill=C_WHITE)
        
        # Liste déroulante
        start_y = 60
        item_height = 30
        display_range = 5
        start_index = max(0, min(self.selected_index - 2, len(self.levels) - display_range))
        
        for i in range(start_index, min(start_index + display_range, len(self.levels))):
            level_key = self.levels[i]
            level_data = DIFFICULTY_LEVELS[level_key]
            y = start_y + (i - start_index) * item_height
            
            if i == self.selected_index:
                # Barre de sélection Verte
                self.draw.rectangle((10, y, self.width - 10, y + item_height), fill=C_GREEN)
                text_color = C_BLACK
                prefix = "> "
            else:
                text_color = C_WHITE
                prefix = "  "
                
            text = f"{prefix}{level_data.name} ({level_data.elo})"
            self.draw.text((20, y + 5), text, font=self.font_small, fill=text_color)
            
        self.disp.image(self.image)

    def run_menu(self):
        """Exécute la boucle du menu."""
        if not MENU_HARDWARE_AVAILABLE:
            # Fallback console
            for i, l in enumerate(self.levels): print(f"{i}: {l}")
            try: return self.levels[int(input("Choix: "))]
            except: return "LEVEL7"

        print("Affichage du menu...")
        self.draw_menu()
        last_steps = 0
        
        while True:
            steps = self.encoder.steps
            if steps != last_steps:
                diff = steps - last_steps
                self.selected_index = (self.selected_index + diff) % len(self.levels)
                last_steps = steps
                self.draw_menu()
            
            if self.button.is_pressed:
                # Feedback visuel
                self.draw.rectangle((0, 0, self.width, self.height), fill=C_GREEN)
                self.draw.text((80, 100), "NIVEAU OK", font=self.font_large, fill=C_BLACK)
                self.disp.image(self.image)
                time.sleep(1)
                return self.levels[self.selected_index]
            time.sleep(0.05)

    def update_game_status(self, board, level_key, last_move_uci=None, ai_thinking=False):
        """Met à jour l'écran pendant la partie."""
        if not MENU_HARDWARE_AVAILABLE: return
        
        # 1. Fond et Structure
        self.draw.rectangle((0, 0, self.width, self.height), fill=C_BLACK)
        
        # Barre du haut (Niveau)
        level_name = DIFFICULTY_LEVELS[level_key].name
        self.draw.rectangle((0, 0, self.width, 30), fill=C_GRAY)
        self.draw.text((10, 5), f"Niveau: {level_name}", font=self.font_small, fill=C_WHITE)
        
        # 2. Qui joue ?
        turn_text = "TRAIT: BLANCS" if board.turn == chess.WHITE else "TRAIT: NOIRS"
        color_bg = C_WHITE if board.turn == chess.WHITE else C_BLUE # Bleu pour noir (plus lisible)
        color_txt = C_BLACK if board.turn == chess.WHITE else C_WHITE
        
        self.draw.rectangle((0, 40, self.width, 80), fill=color_bg)
        self.draw.text((10, 50), turn_text, font=self.font_medium, fill=color_txt)
        
        # 3. Dernier coup
        if last_move_uci:
            self.draw.text((10, 90), f"Dernier coup: {last_move_uci}", font=self.font_medium, fill=C_WHITE)
            
        # 4. Statut spécial (Echec, Mat, Réflexion)
        status_y = 130
        if ai_thinking:
            self.draw.text((10, status_y), ">>> IA Réfléchit...", font=self.font_small, fill=C_ORANGE)
        elif board.is_checkmate():
            self.draw.rectangle((0, status_y, self.width, status_y+40), fill=C_RED)
            self.draw.text((10, status_y+5), "ECHEC ET MAT !", font=self.font_medium, fill=C_WHITE)
        elif board.is_check():
            self.draw.text((10, status_y), "ATTENTION: ECHEC", font=self.font_medium, fill=C_RED)
        elif board.is_stalemate():
            self.draw.text((10, status_y), "PAT (Egalité)", font=self.font_medium, fill=C_BLUE)
            
        self.disp.image(self.image)


def setup_hardware():
    """Initialise tout le matériel I2C et renvoie les objets clés."""
    print("=== Initialisation du matériel de l'échiquier ===")
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        tca = adafruit_tca9548a.TCA9548A(i2c, address=0x72)
        print("   - Init LEDs...")
        display_matrix = Matrix16x8(tca[4], address=0x70)
        display_row = Matrix16x8(tca[5], address=0x71)
        display_matrix.brightness = 0.2
        display_row.brightness = 0.2
        display_matrix.fill(0); display_row.fill(0); display_matrix.show(); display_row.show()

        print("   - Init Capteurs...")
        mcps = [MCP23017(tca[i], address=0x20) for i in range(4)]
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

        return display_matrix, display_row, sensor_pins
    except Exception as e:
        print(f"\nERREUR MATÉRIELLE: {e}")
        traceback.print_exc()
        raise

def setup_new_ai(level_name="LEVEL7"):
    if not NEW_AI_AVAILABLE: return None
    try:
        ai = ChessEngine()
        ai.set_level(level_name)
        return ai
    except Exception as e:
        print(f"Erreur init IA: {e}")
        return None

def get_state_from_sensors(sensor_pins):
    state = [[False for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for c in range(8):
            try: state[r][c] = not sensor_pins[r][c].value
            except OSError: pass
    return state

def get_initial_state(sensor_pins):
    print("\033[H\033[JPlacer les pièces...")
    FAULTY_SQUARE = (6, 3)
    while True:
        state = get_state_from_sensors(sensor_pins)
        count = sum(sum(row) for row in state)
        if count >= 31: 
             is_d7_missing = not state[6][3]
             if count == 32 or (count == 31 and is_d7_missing):
                 if is_d7_missing: state[6][3] = True
                 print("Position OK.")
                 return state
        time.sleep(0.5)

def wait_for_state(sensor_pins, target_state):
    print(">>> REMETTRE LES PIÈCES EN PLACE <<<")
    FAULTY_SQUARE = (6, 3)
    while True:
        current = get_state_from_sensors(sensor_pins)
        if target_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]]:
            current[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]] = True
        if current == target_state: return
        time.sleep(0.5)

def find_diffs(old_state, new_state):
    lifted, placed = [], []
    for r in range(8):
        for c in range(8):
            if old_state[r][c] and not new_state[r][c]: lifted.append((r, c))
            elif not old_state[r][c] and new_state[r][c]: placed.append((r, c))
    return lifted, placed

def set_square_leds(sensor_row, sensor_col, state, display_matrix, display_row):
    target_col, target_row = 7 - sensor_col, 7 - sensor_row
    corners = [(target_col, target_row), (target_col + 1, target_row), (target_col, target_row + 1), (target_col + 1, target_row + 1)]
    for px, py in corners:
        if (px, py) in LED_MAP:
            canal, sw_col, sw_row = LED_MAP[(px, py)]
            if canal == "C4": display_matrix.pixel(sw_col, sw_row, state)
            elif canal == "C5": display_row.pixel(sw_col, sw_row, state)

def update_leds(display_matrix, display_row, from_coords, to_coords):
    display_matrix.fill(0); display_row.fill(0)
    if from_coords: set_square_leds(from_coords[0], from_coords[1], 1, display_matrix, display_row)
    if to_coords: set_square_leds(to_coords[0], to_coords[1], 1, display_matrix, display_row)
    display_matrix.show(); display_row.show()

def build_move(from_coords, to_coords, game_board):
    from_sq = chess.square(from_coords[1], from_coords[0])
    to_sq = chess.square(to_coords[1], to_coords[0])
    move = chess.Move(from_sq, to_sq)
    p = game_board.piece_at(from_sq)
    if p and p.piece_type == chess.PAWN:
        if (p.color == chess.WHITE and from_coords[0] == 6 and to_coords[0] == 7) or \
           (p.color == chess.BLACK and from_coords[0] == 1 and to_coords[0] == 0):
            move.promotion = chess.QUEEN
    return move

def main():
    print("=== SMART CHESS V2 - Démarrage ===")
    
    # 1. Initialisation Ecran
    display = SmartDisplay()
    selected_level = display.run_menu()
    print(f"Niveau choisi : {selected_level}")

    # 2. Matériel
    try:
        display_matrix, display_row, sensor_pins = setup_hardware()
    except Exception: return

    # 3. IA
    ai_engine = setup_new_ai(selected_level)
    
    game_board = chess.Board()
    # Affichage initial du jeu
    display.update_game_status(game_board, selected_level)
    
    last_known_physical_state = get_initial_state(sensor_pins)
    FAULTY_SQUARE = (6, 3)

    try:
        while not game_board.is_game_over():
            # Mise à jour écran début de tour
            display.update_game_status(game_board, selected_level, 
                                     game_board.peek().uci() if game_board.move_stack else None)

            # Tour IA
            if ai_engine and game_board.turn == chess.BLACK:
                display.update_game_status(game_board, selected_level, ai_thinking=True)
                
                best_move = ai_engine.get_move(game_board, time_limit=2.0)
                
                # Feedback coup IA
                display.update_game_status(game_board, selected_level, f"IA JOUE: {best_move.uci()}")
                
                from_sq, to_sq = best_move.from_square, best_move.to_square
                ai_from = (chess.square_rank(from_sq), chess.square_file(from_sq))
                ai_to = (chess.square_rank(to_sq), chess.square_file(to_sq))
                update_leds(display_matrix, display_row, ai_from, ai_to)
            
            # Boucle détection physique
            while True:
                current_state = get_state_from_sensors(sensor_pins)
                if last_known_physical_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]]:
                    current_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]] = True

                if current_state != last_known_physical_state:
                    time.sleep(0.5) # Debounce
                    stable_state = get_state_from_sensors(sensor_pins)
                    lifted, placed = find_diffs(last_known_physical_state, stable_state)
                    
                    if FAULTY_SQUARE in lifted and len(lifted) > 1 and last_known_physical_state[FAULTY_SQUARE[0]][FAULTY_SQUARE[1]]:
                         lifted.remove(FAULTY_SQUARE)

                    # Analyse coup
                    move = None
                    if len(lifted) == 1 and len(placed) == 1:
                        move = build_move(lifted[0], placed[0], game_board)
                    elif len(lifted) == 2 and len(placed) == 1:
                        dest = placed[0]
                        if dest in lifted:
                            orig = [sq for sq in lifted if sq != dest][0]
                            move = build_move(orig, dest, game_board)
                    elif len(lifted) == 2 and len(placed) == 2:
                         king_sq = game_board.king(game_board.turn)
                         k_coords = (chess.square_rank(king_sq), chess.square_file(king_sq))
                         if k_coords in lifted:
                             for sq in placed:
                                 if sq[1] in [2, 6]:
                                     move = build_move(k_coords, sq, game_board); break

                    if move and move in game_board.legal_moves:
                        print(f"Coup joué: {move.uci()}")
                        game_board.push(move)
                        
                        last_known_physical_state = stable_state
                        if game_board.piece_at(chess.D7): last_known_physical_state[6][3] = True
                        else: last_known_physical_state[6][3] = False
                        
                        update_leds(display_matrix, display_row, lifted[0] if lifted else None, placed[0] if placed else None)
                        display.update_game_status(game_board, selected_level, move.uci())
                        break
                    
                    elif move:
                        print("Coup illégal")
                        wait_for_state(sensor_pins, last_known_physical_state)
                time.sleep(0.1)

        print("Fin de partie.")
        display.update_game_status(game_board, selected_level) # Afficher mat final

    except KeyboardInterrupt:
        print("\nArrêt.")
    finally:
        if 'display_matrix' in locals(): display_matrix.fill(0); display_matrix.show()
        if 'display_row' in locals(): display_row.fill(0); display_row.show()

if __name__ == "__main__":
    main()