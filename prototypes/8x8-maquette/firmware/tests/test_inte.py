#!/usr/bin/env python3
"""
Script de test d'intégration final pour échiquier intelligent I2C
Mappage 81-LED personnalisé ET mappage 64-Capteurs personnalisé.
CORRECTION: Ajout de la transformation 180° (symétrie)
MODIFIÉ: Suppression du clignotement (gestion de l'état)
"""

import board
import busio
import time
import adafruit_tca9548a
from adafruit_mcp230xx.mcp23017 import MCP23017
from adafruit_ht16k33.matrix import Matrix16x8 
import digitalio

# --- DÉBUT DE LA TABLE DE TRADUCTION (LED_MAP) ---
# Construit à partir de vos 81 tests de scan.
# Format: (phys_x, phys_y) -> (canal, sw_col, sw_row)
LED_MAP = {
    # Données du Canal 4 (Matrice 8x8)
    (8, 0): ('C4', 0, 0),
    (7, 0): ('C4', 0, 1),
    (6, 0): ('C4', 0, 2),
    (5, 0): ('C4', 0, 3),
    (4, 0): ('C4', 0, 4),
    (3, 0): ('C4', 0, 5),
    (2, 0): ('C4', 0, 6),
    (1, 0): ('C4', 0, 7),
    
    (8, 1): ('C4', 1, 0),
    (7, 1): ('C4', 1, 1),
    (6, 1): ('C4', 1, 2),
    (5, 1): ('C4', 1, 3),
    (4, 1): ('C4', 1, 4),
    (3, 1): ('C4', 1, 5),
    (2, 1): ('C4', 1, 6),
    (1, 1): ('C4', 1, 7),
    (0, 1): ('C4', 8, 1), # Corrigé d'après votre LED_MAP (était 1,8)

    (8, 2): ('C4', 2, 0),
    (7, 2): ('C4', 2, 1),
    (6, 2): ('C4', 2, 2),
    (5, 2): ('C4', 2, 3),
    (4, 2): ('C4', 2, 4),
    (3, 2): ('C4', 2, 5),
    (2, 2): ('C4', 2, 6),
    (1, 2): ('C4', 2, 7),
    
    (8, 3): ('C4', 3, 0),
    (7, 3): ('C4', 3, 1),
    (6, 3): ('C4', 3, 2),
    (5, 3): ('C4', 3, 3),
    (4, 3): ('C4', 3, 4),
    (3, 3): ('C4', 3, 5),
    (2, 3): ('C4', 3, 6),
    (1, 3): ('C4', 3, 7),
    
    (8, 4): ('C4', 4, 0),
    (7, 4): ('C4', 4, 1),
    (6, 4): ('C4', 4, 2),
    (5, 4): ('C4', 4, 3),
    (4, 4): ('C4', 4, 4),
    (3, 4): ('C4', 4, 5),
    (2, 4): ('C4', 4, 6),
    (1, 4): ('C4', 4, 7),
    
    (8, 5): ('C4', 5, 0),
    (7, 5): ('C4', 5, 1),
    (6, 5): ('C4', 5, 2),
    (5, 5): ('C4', 5, 3),
    (4, 5): ('C4', 5, 4),
    (3, 5): ('C4', 5, 5),
    (2, 5): ('C4', 5, 6),
    (1, 5): ('C4', 5, 7),

    
    (8, 6): ('C4', 6, 0),
    (7, 6): ('C4', 6, 1),
    (6, 6): ('C4', 6, 2),
    (5, 6): ('C4', 6, 3),
    (4, 6): ('C4', 6, 4),
    (3, 6): ('C4', 6, 5),
    (2, 6): ('C4', 6, 6),
    (1, 6): ('C4', 6, 7),
    
    (8, 7): ('C4', 7, 0),
    (7, 7): ('C4', 7, 1),
    (6, 7): ('C4', 7, 2),
    (5, 7): ('C4', 7, 3),
    (4, 7): ('C4', 7, 4),
    (3, 7): ('C4', 7, 5),
    (2, 7): ('C4', 7, 6),
    (1, 7): ('C4', 7, 7),

    
    # Données du Canal 5
    (8, 8): ('C5', 0, 0),
    (7, 8): ('C5', 0, 1),
    (6, 8): ('C5', 0, 2),
    (5, 8): ('C5', 0, 3),
    (4, 8): ('C5', 0, 4),
    (3, 8): ('C5', 0, 5),
    (2, 8): ('C5', 0, 6),
    (1, 8): ('C5', 0, 7),
    
    
    (0, 8): ('C5', 2, 0),
    (0, 7): ('C5', 1, 7),
    (0, 6): ('C5', 1, 6),
    (0, 5): ('C5', 1, 5),
    (0, 4): ('C5', 1, 4),
    (0, 3): ('C5', 1, 3),
    (0, 2): ('C5', 1, 2),
    (0, 1): ('C5', 1, 1),
    (0, 0): ('C5', 1, 0),
}
# --- FIN DE LA TABLE DE TRADUCTION ---


def main():
    """
    Fonction principale du test d'intégration
    """
    print("=== Test d'intégration capteurs-LEDs de l'échiquier intelligent ===")
    
    # Étape 1: Initialisation du bus I2C principal
    print("1. Initialisation du bus I2C principal...")
    i2c = busio.I2C(board.SCL, board.SDA)
    
    # Étape 2: Initialisation du multiplexeur TCA9548A
    print("2. Initialisation du multiplexeur TCA9548A...")
    tca = adafruit_tca9548a.TCA9548A(i2c, address=0x72)
    
    # Étape 3: Initialisation des DEUX contrôleurs LED
    print("3. Initialisation de la matrice LED 9x8 (Canal 4, 0x70)...")
    display_matrix = Matrix16x8(tca[4], address=0x70)
    display_matrix.brightness = 0.2
    
    print("4. Initialisation de la rangée LED 9x1 (Canal 5, 0x71)...")
    display_row = Matrix16x8(tca[5], address=0x71)
    display_row.brightness = 0.2
    
    # --- MODIFICATION : Éteindre les LEDs UNE SEULE FOIS ---
    print("   - Extinction initiale des LEDs...")
    display_matrix.fill(0)
    display_matrix.show()
    display_row.fill(0)
    display_row.show()
    
    # Étape 5: Initialisation des 4 contrôleurs MCP23017
    print("5. Initialisation des contrôleurs MCP23017...")
    print("   - CM0: Canal 0, adresse 0x20 (rangées 1-2)")
    mcp0 = MCP23017(tca[0], address=0x20)
    
    print("   - CM1: Canal 1, adresse 0x20 (rangées 3-4)")
    mcp1 = MCP23017(tca[1], address=0x20)
    
    print("   - CM2: Canal 2, adresse 0x20 (rangées 5-6)")
    mcp2 = MCP23017(tca[2], address=0x20)
    
    print("   - CM3: Canal 3, adresse 0x20 (rangées 7-8)")
    mcp3 = MCP23017(tca[3], address=0x20)
    
    # Liste des contrôleurs MCP23017
    mcps = [mcp0, mcp1, mcp2, mcp3]
    
    # Étape 6: Création de la matrice des pins capteurs
    print("6. Configuration des 64 capteurs Reed...")
    sensor_pins = [[None for _ in range(8)] for _ in range(8)]
    
    # --- MAPPAGE CAPTEURS (selon votre câblage) ---
    # Rangée 1 (row=0)
    sensor_pins[0][0] = mcps[0].get_pin(0)  # A1
    sensor_pins[0][1] = mcps[0].get_pin(8)  # B1
    sensor_pins[0][2] = mcps[0].get_pin(1)  # C1
    sensor_pins[0][3] = mcps[0].get_pin(9)  # D1
    sensor_pins[0][4] = mcps[0].get_pin(2)  # E1
    sensor_pins[0][5] = mcps[0].get_pin(10) # F1
    sensor_pins[0][6] = mcps[0].get_pin(3)  # G1
    sensor_pins[0][7] = mcps[0].get_pin(11) # H1
    
    # Rangée 2 (row=1)
    sensor_pins[1][0] = mcps[0].get_pin(15) # A2
    sensor_pins[1][1] = mcps[0].get_pin(7)  # B2
    sensor_pins[1][2] = mcps[0].get_pin(14) # C2
    sensor_pins[1][3] = mcps[0].get_pin(6)  # D2
    sensor_pins[1][4] = mcps[0].get_pin(13) # E2
    sensor_pins[1][5] = mcps[0].get_pin(5)  # F2
    sensor_pins[1][6] = mcps[0].get_pin(12) # G2
    sensor_pins[1][7] = mcps[0].get_pin(4)  # H2
    
    # Rangée 3 (row=2)
    sensor_pins[2][0] = mcps[1].get_pin(0)  # A3
    sensor_pins[2][1] = mcps[1].get_pin(8)  # B3
    sensor_pins[2][2] = mcps[1].get_pin(1)  # C3
    sensor_pins[2][3] = mcps[1].get_pin(9)  # D3
    sensor_pins[2][4] = mcps[1].get_pin(2)  # E3
    sensor_pins[2][5] = mcps[1].get_pin(10) # F3
    sensor_pins[2][6] = mcps[1].get_pin(3)  # G3
    sensor_pins[2][7] = mcps[1].get_pin(11) # H3
    
    # Rangée 4 (row=3)
    sensor_pins[3][0] = mcps[1].get_pin(15) # A4
    sensor_pins[3][1] = mcps[1].get_pin(7)  # B4
    sensor_pins[3][2] = mcps[1].get_pin(14) # C4
    sensor_pins[3][3] = mcps[1].get_pin(6)  # D4
    sensor_pins[3][4] = mcps[1].get_pin(13) # E4
    sensor_pins[3][5] = mcps[1].get_pin(5)  # F4
    sensor_pins[3][6] = mcps[1].get_pin(12) # G4
    sensor_pins[3][7] = mcps[1].get_pin(4)  # H4
    
    # Rangée 5 (row=4)
    sensor_pins[4][0] = mcps[2].get_pin(0)  # A5
    sensor_pins[4][1] = mcps[2].get_pin(8)  # B5
    sensor_pins[4][2] = mcps[2].get_pin(1)  # C5
    sensor_pins[4][3] = mcps[2].get_pin(9)  # D5
    sensor_pins[4][4] = mcps[2].get_pin(2)  # E5
    sensor_pins[4][5] = mcps[2].get_pin(10) # F5
    sensor_pins[4][6] = mcps[2].get_pin(3)  # G5
    sensor_pins[4][7] = mcps[2].get_pin(11) # H5
    
    # Rangée 6 (row=5)
    sensor_pins[5][0] = mcps[2].get_pin(15) # A6
    sensor_pins[5][1] = mcps[2].get_pin(7)  # B6
    sensor_pins[5][2] = mcps[2].get_pin(14) # C6
    sensor_pins[5][3] = mcps[2].get_pin(6)  # D6
    sensor_pins[5][4] = mcps[2].get_pin(13) # E6
    sensor_pins[5][5] = mcps[2].get_pin(5)  # F6
    sensor_pins[5][6] = mcps[2].get_pin(12) # G6
    sensor_pins[5][7] = mcps[2].get_pin(4)  # H6
    
    # Rangée 7 (row=6)
    sensor_pins[6][0] = mcps[3].get_pin(0)  # A7
    sensor_pins[6][1] = mcps[3].get_pin(8)  # B7
    sensor_pins[6][2] = mcps[3].get_pin(1)  # C7
    sensor_pins[6][3] = mcps[3].get_pin(9)  # D7
    sensor_pins[6][4] = mcps[3].get_pin(2)  # E7
    sensor_pins[6][5] = mcps[3].get_pin(10) # F7
    sensor_pins[6][6] = mcps[3].get_pin(3)  # G7
    sensor_pins[6][7] = mcps[3].get_pin(11) # H7
    
    # Rangée 8 (row=7)
    sensor_pins[7][0] = mcps[3].get_pin(15) # A8
    sensor_pins[7][1] = mcps[3].get_pin(7)  # B8
    sensor_pins[7][2] = mcps[3].get_pin(14) # C8
    sensor_pins[7][3] = mcps[3].get_pin(6)  # D8
    sensor_pins[7][4] = mcps[3].get_pin(13) # E8
    sensor_pins[7][5] = mcps[3].get_pin(5)  # F8
    sensor_pins[7][6] = mcps[3].get_pin(12) # G8
    sensor_pins[7][7] = mcps[3].get_pin(4)  # H8
    
    # Configuration de tous les pins
    for row in range(8):
        for col in range(8):
            pin = sensor_pins[row][col]
            pin.direction = digitalio.Direction.INPUT
            pin.pull = digitalio.Pull.UP
    
    print("   - Configuration terminée")
    print("\n=== Détection des pièces en cours (Ctrl+C pour quitter) ===")
    
    # --- MODIFICATION : Ajout d'une variable pour mémoriser l'état ---
    last_board_state = [[False for _ in range(8)] for _ in range(8)]
    
    try:
        # Boucle principale de détection
        while True:
            something_changed = False
            
            # Parcours des 64 capteurs
            for row in range(8):
                for col in range(8):
                    
                    pin = sensor_pins[row][col] # Récupère le BON pin mappé
                    
                    # Lecture du capteur (actif LOW = pièce détectée)
                    # is_piece_present est VRAI (True) si une pièce est là
                    is_piece_present = not pin.value
                    
                    # --- DÉBUT DE LA LOGIQUE ANTI-CLIGNOTEMENT ---
                    # On compare l'état actuel à l'état mémorisé
                    if is_piece_present != last_board_state[row][col]:
                        # L'état a changé !
                        something_changed = True
                        
                        # 1. Définir la valeur LED (1 pour allumer, 0 pour éteindre)
                        led_value_to_set = 1 if is_piece_present else 0
                        
                        # 2. Calculer la case cible (opposée)
                        target_col = 7 - col
                        target_row = 7 - row

                        # 3. Définir les 4 coins physiques
                        physical_corners = [
                            (target_col, target_row),       # Coin A
                            (target_col + 1, target_row),   # Coin B
                            (target_col, target_row + 1),   # Coin C
                            (target_col + 1, target_row + 1)  # Coin D
                        ]

                        # 4. Appliquer la nouvelle valeur (Allumer ou Éteindre)
                        for (px, py) in physical_corners:
                            if (px, py) in LED_MAP:
                                canal, sw_col, sw_row = LED_MAP[(px, py)]
                                
                                if canal == 'C4':
                                    display_matrix.pixel(sw_col, sw_row, led_value_to_set)
                                elif canal == 'C5':
                                    display_row.pixel(sw_col, sw_row, led_value_to_set)
                        
                        # 5. Mettre à jour l'état mémorisé pour cette case
                        last_board_state[row][col] = is_piece_present
                        
                    # --- FIN DE LA LOGIQUE ANTI-CLIGNOTEMENT ---
            
            # Si quelque chose a changé, on met à jour les LEDs
            if something_changed:
                display_matrix.show()
                display_row.show()
            
            # Délai court pour éviter de surcharger le CPU
            time.sleep(0.05) 
            
    except KeyboardInterrupt:
        print("\n=== Arrêt du programme ===")
        # Éteindre toutes les LEDs avant de quitter
        display_matrix.fill(0)
        display_row.fill(0)
        display_matrix.show()
        display_row.show()

if __name__ == "__main__":
    main()
