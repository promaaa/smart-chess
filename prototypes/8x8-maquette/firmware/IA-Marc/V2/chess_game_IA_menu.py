#!/usr/bin/env python3
"""
Script de jeu d'échecs avec l'IA-Marc V2 sur matériel réel.
====================================================================
VERSION AVEC MENU DE SÉLECTION DE NIVEAU
====================================================================

Cette version intègre un menu pour permettre à l'utilisateur de
choisir le niveau de difficulté de l'IA avant de commencer la partie.

Fonctionnalités :
- Menu de sélection du niveau de l'IA (de 1 à 9)
- Simulation de l'interaction via un potentiomètre et un écran
- Intégration de l'IA-Marc V2 avec le niveau sélectionné
"""

import os
import sys
import time
import traceback

# Ajout du chemin courant pour les imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Imports du matériel (à décommenter sur le RPi) ---
# import adafruit_tca9548a
# import board
# import busio
# import digitalio
# from adafruit_ht16k33.matrix import Matrix16x8
# from adafruit_mcp230xx.mcp23017 import MCP23017

# --- Imports de l'IA ---
try:
    from engine_main import ChessEngine
    from engine_config import DIFFICULTY_LEVELS
    NEW_AI_AVAILABLE = True
except ImportError as e:
    print(f"ERREUR CRITIQUE: Impossible d'importer l'IA-Marc V2: {e}")
    NEW_AI_AVAILABLE = False


# ============================================================================
# SECTION POUR L'INTÉGRATION MATÉRIELLE (ÉCRAN + POTENTIOMÈTRE)
# ============================================================================
#
# IMPORTANT: Les fonctions ci-dessous sont des SIMULATIONS.
# Vous devez remplacer leur contenu par le code de vos propres pilotes
# pour l'écran et le potentiomètre.
#
# ----------------------------------------------------------------------------

def setup_menu_hardware():
    """
    Initialise le matériel du menu (écran, potentiomètre).
    Retourne les objets nécessaires pour interagir avec.
    """
    print("[SIMULATION] Initialisation de l'écran et du potentiomètre...")
    # TODO: Remplacez ceci par l'initialisation de votre écran (ex: afficheur OLED)
    # et de votre potentiomètre (ex: via GPIO ou ADC).
    #
    # screen = ... (votre objet écran)
    # potentiometer = ... (votre objet potentiomètre)
    #
    # return screen, potentiometer
    return None, None

def display_on_screen(screen, line1: str, line2: str = ""):
    """
    Affiche deux lignes de texte sur l'écran.
    """
    # TODO: Remplacez cette fonction pour qu'elle affiche le texte sur votre écran.
    # Exemple pour un écran OLED avec la librairie 'luma.oled':
    #
    # from luma.core.render import canvas
    # with canvas(screen) as draw:
    #     draw.text((0, 0), line1, fill="white")
    #     draw.text((0, 10), line2, fill="white")
    #
    print(f"[ÉCRAN] L1: {line1}")
    if line2:
        print(f"[ÉCRAN] L2: {line2}")
    print("-" * 20)

def get_potentiometer_input(potentiometer):
    """
    Lit l'entrée du potentiomètre.
    Retourne "UP", "DOWN", "CLICK" ou None.
    """
    # TODO: Remplacez cette fonction pour lire votre potentiomètre.
    # Vous devrez probablement garder en mémoire la dernière valeur lue
    # pour détecter un changement (rotation) ou un appui sur le bouton.
    #
    # Exemple de logique :
    #
    # last_value = potentiometer.value
    # time.sleep(0.1)
    # new_value = potentiometer.value
    # button_pressed = potentiometer.button.is_pressed
    #
    # if button_pressed: return "CLICK"
    # if new_value > last_value + THRESHOLD: return "UP"
    # if new_value < last_value - THRESHOLD: return "DOWN"
    # return None
    #
    # Pour la simulation, on utilise l'entrée clavier.
    try:
        key = input("[INPUT] 'z' pour monter, 's' pour descendre, 'entrée' pour choisir: ")
        if key == 'z': return "UP"
        if key == 's': return "DOWN"
        if key == '': return "CLICK"
    except KeyboardInterrupt:
        raise
    except Exception:
        return None
    return None

def select_ai_level(screen, potentiometer):
    """
    Affiche le menu de sélection et attend que l'utilisateur choisisse un niveau.
    """
    # La liste des niveaux sur lesquels l'utilisateur peut itérer.
    # On trie par ELO pour un ordre logique.
    levels = sorted(DIFFICULTY_LEVELS.items(), key=lambda item: item[1].elo)
    level_keys = [item[0] for item in levels]
    level_names = [f"{item[1].name} ({item[1].elo} ELO)" for item in levels]
    
    current_selection_index = 6  # Démarrer sur "LEVEL7 (Club)" par défaut

    display_on_screen(screen, "Choisir le niveau", level_names[current_selection_index])

    while True:
        action = get_potentiometer_input(potentiometer)

        if action == "UP":
            current_selection_index = (current_selection_index + 1) % len(level_keys)
        elif action == "DOWN":
            current_selection_index = (current_selection_index - 1 + len(level_keys)) % len(level_keys)
        elif action == "CLICK":
            selected_level_key = level_keys[current_selection_index]
            display_on_screen(screen, "Niveau choisi:", level_names[current_selection_index])
            time.sleep(2)
            return selected_level_key

        if action in ["UP", "DOWN"]:
            display_on_screen(screen, "Choisir le niveau", level_names[current_selection_index])
        
        time.sleep(0.1)

# ============================================================================
# LE RESTE DU SCRIPT EST IDENTIQUE À chess_game_IA.py
# SAUF POUR L'INITIALISATION DE L'IA DANS main()
# ============================================================================

def setup_chessboard_hardware():
    # ... (le code reste identique à l'original)
    return None, None, None # Mock pour l'instant

def setup_new_ai(selected_level: str):
    """Initialise l'IA-Marc V2 avec le niveau choisi."""
    if not NEW_AI_AVAILABLE:
        return None
    try:
        print("Initialisation de l'IA-Marc V2...")
        ai = ChessEngine()
        ai.set_level(selected_level)
        level_details = DIFFICULTY_LEVELS[selected_level]
        print(f"IA-Marc V2 prête (Niveau: {level_details.name}, ELO: {level_details.elo})")
        return ai
    except Exception as e:
        print(f"Erreur lors de l'initialisation de l'IA: {e}")
        traceback.print_exc()
        return None

# Toutes les autres fonctions (get_state_from_sensors, get_initial_state, etc.)
# restent les mêmes que dans chess_game_IA.py. Pour la simplicité de cet exemple,
# elles ne sont pas toutes recopiées ici. On assume qu'elles existent.
def get_initial_state(sensor_pins): return [[False]*8]*8
def update_leds(display_matrix, display_row, from_coords, to_coords): pass
def find_diffs(old, new): return [],[]
def build_move(f,t,b): return None
def wait_for_state(s, t): pass


def main():
    print("=== Démarrage de Chess Game IA-Marc V2 (Avec Menu) ===")
    
    # 1. Initialisation du matériel (échiquier, écran, potentiomètre)
    try:
        # NOTE: Les objets retournés ici seront 'None' car nous simulons
        display_matrix, display_row, sensor_pins = setup_chessboard_hardware()
        screen, potentiometer = setup_menu_hardware()
    except Exception as e:
        print(f"Erreur fatale matériel: {e}")
        return

    # 2. Afficher le menu et attendre la sélection de l'utilisateur
    if NEW_AI_AVAILABLE:
        chosen_level = select_ai_level(screen, potentiometer)
    else:
        print("IA non disponible, démarrage en mode 2 joueurs.")
        chosen_level = "LEVEL7" # Fallback

    # 3. Initialiser le moteur d'IA avec le niveau choisi
    ai_engine = setup_new_ai(chosen_level)
    if not ai_engine:
        print("Impossible de continuer sans IA.")
        return

    # Effacer l'écran du menu et préparer le jeu
    # TODO: Ajoutez ici le code pour effacer votre écran
    # display_on_screen(screen, "", "")
    print("\n--- Début de la partie ---")
    
    game_board = chess.Board()
    # Le reste de la boucle de jeu est identique à la version originale...
    # Par souci de clarté, la boucle de jeu n'est pas entièrement reproduite ici.
    # Le point important est que `ai_engine` est maintenant configuré avec le bon niveau.

    print("\n[SIMULATION] La boucle de jeu commencerait ici.")
    print(f"[SIMULATION] L'IA est configurée au niveau: {chosen_level}")
    
    # Exemple de comment l'IA serait appelée dans la boucle de jeu
    if ai_engine and game_board.turn == chess.BLACK:
        print("\n[SIMULATION] L'IA réfléchit...")
        best_move = ai_engine.get_move(game_board, time_limit=2.0)
        print(f"[SIMULATION] Coup trouvé: {best_move.uci()}")
        # ... le reste de la logique pour afficher le coup sur les LEDs, etc.
    
    print("\n=== Fin du script de simulation ===")


if __name__ == "__main__":
    main()