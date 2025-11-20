#!/usr/bin/env python3
"""
Script de test d'éclairage pour échiquier intelligent I2C
Ce script teste les deux contrôleurs LED (LED_A et LED_B) via le multiplexeur TCA9548A
"""

import board
import busio
import time
import adafruit_tca9548a
# MODIFICATION: Importation de Matrix16x8 pour gérer 9 colonnes ou plus
from adafruit_ht16k33.matrix import Matrix16x8

def main():
    """
    Fonction principale du test d'éclairage
    """
    print("=== Test d'éclairage de l'échiquier intelligent ===")
    
    # Étape 1: Initialisation du bus I2C principal
    print("1. Initialisation du bus I2C principal...")
    i2c = busio.I2C(board.SCL, board.SDA)
    
    # Étape 2: Initialisation du multiplexeur TCA9548A
    print("2. Initialisation du multiplexeur TCA9548A...")
    
    # LIGNE MODIFIÉE :
    # On spécifie l'adresse physique du multiplexeur (ex: 0x72).
    # Elle DOIT être différente des adresses des appareils qu'il contrôle
    # (qui sont 0x70 et 0x71).
    tca = adafruit_tca9548a.TCA9548A(i2c, address=0x72)
    
    # Étape 3: Initialisation de la matrice 8x8 principale (LED_A)
    print("3. Initialisation de la matrice 9x8 principale (LED_A)...") # Info mise à jour
    print("   - Canal 4 du multiplexeur")
    print("   - Adresse I2C: 0x70")
    # MODIFICATION: Utilisation de Matrix16x8
    display_matrix = Matrix16x8(tca[4], address=0x70)
    
    # Étape 4: Initialisation de la rangée 1x8 (LED_B)
    print("4. Initialisation de la rangée 9x1 (LED_B)...") # Info mise à jour
    print("   - Canal 5 du multiplexeur")
    print("   - Adresse I2C: 0x71")
    # MODIFICATION: Utilisation de Matrix16x8
    display_row = Matrix16x8(tca[5], address=0x71)
    
    # Étape 5: Réglage de la luminosité
    print("5. Réglage de la luminosité...")
    brightness = 0.1
    display_matrix.brightness = brightness
    display_row.brightness = brightness
    print(f"   - Luminosité réglée à {brightness}")
    
    # Étape 6: Test de balayage sur la matrice 8x8
    print("6. Test de balayage sur la matrice 9x8...") # Info mise à jour
    print("   - Allumage pixel par pixel")
    # La boucle 'range(9)' (index 0-8) est maintenant correcte
    for x in range(8):
        for y in range(8):
            display_matrix.pixel(x, y, 1)
            display_matrix.show()
            time.sleep(0.1)
            display_matrix.pixel(x, y, 0)
    
    # Étape 7: Test de balayage sur la rangée 1x8
    print("7. Test de balayage sur la rangée 9x1...") # Info mise à jour
    print("   - Allumage pixel par pixel")
    # La boucle 'range(9)' (index 0-8) est maintenant correcte
    for x in range(8):
        for y in range(8):
            display_row.pixel(x, y, 1)
            display_row.show()
            time.sleep(0.1)
            display_row.pixel(x, y, 0)

    
    # Étape 8: Test d'allumage complet
    print("8. Test d'allumage complet...")
    print("   - Allumage de toutes les LEDs pendant 2 secondes")
    display_matrix.fill(1)
    display_row.fill(1)
    display_matrix.show()
    display_row.show()
    time.sleep(2)
    
    # Éteindre toutes les LEDs
    display_matrix.fill(0)
    display_row.fill(0)
    display_matrix.show()
    display_row.show()
    
    print("\n=== Test terminé avec succès ===")

if __name__ == "__main__":
    main()
