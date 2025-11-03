#!/usr/bin/env python3
"""
test_led_ht16k33.py - Test des LEDs avec deux HT16K33

Ce programme teste la matrice LED 9x9 contrôlée par deux HT16K33 :
- HT16K33 #1 : contrôle les 9 lignes (B1-B9)
- HT16K33 #2 : contrôle les 9 colonnes (A1-A9)

Séquences de test :
1. Allumage de chaque LED individuellement
2. Balayage par lignes
3. Balayage par colonnes
4. Test de la diagonale
5. Motifs d'échecs
6. Variation d'intensité
"""

import time
import board
import busio
from adafruit_ht16k33 import ht16k33

# --- Configuration ---
# Les deux HT16K33 sont sur le TCA9548A
HT16K33_ROWS_ADDR = 0x70  # Adresse du HT16K33 pour les lignes
HT16K33_COLS_ADDR = 0x70  # Adresse du HT16K33 pour les colonnes (même adresse car sur canal différent)

# Configuration TCA9548A
TCA_ROWS_CHANNEL = 4  # Canal TCA pour HT16K33 lignes
TCA_COLS_CHANNEL = 5  # Canal TCA pour HT16K33 colonnes


class HT16K33Matrix:
    """Contrôleur pour matrice LED 9x9 avec deux HT16K33 via TCA9548A."""

    def __init__(self, tca, rows_channel=TCA_ROWS_CHANNEL, cols_channel=TCA_COLS_CHANNEL,
                 rows_addr=HT16K33_ROWS_ADDR, cols_addr=HT16K33_COLS_ADDR):
        """
        Initialise les deux HT16K33 sur le TCA9548A.

        Args:
            tca: Instance du TCA9548A
            rows_channel: Canal TCA pour le HT16K33 des lignes
            cols_channel: Canal TCA pour le HT16K33 des colonnes
            rows_addr: Adresse I2C du HT16K33 lignes
            cols_addr: Adresse I2C du HT16K33 colonnes
        """
        print(f"Initialisation HT16K33 lignes (canal TCA {rows_channel}, addr 0x{rows_addr:02X})...")
        i2c_rows = tca[rows_channel]
        self.ht_rows = ht16k33.HT16K33(i2c_rows, address=rows_addr)
        self.ht_rows.blink_rate = 0
        self.ht_rows.brightness = 1.0

        print(f"Initialisation HT16K33 colonnes (canal TCA {cols_channel}, addr 0x{cols_addr:02X})...")
        i2c_cols = tca[cols_channel]
        self.ht_cols = ht16k33.HT16K33(i2c_cols, address=cols_addr)
        self.ht_cols.blink_rate = 0
        self.ht_cols.brightness = 1.0

        self.led_state = set()  # Ensemble de tuples (row, col)
        self.clear()</parameter>
</text>

<old_text line=320>
    # Initialisation
    i2c = busio.I2C(board.SCL, board.SDA)

    if USE_TCA:
        from adafruit_tca9548a import TCA9548A

        print(f"Utilisation du TCA9548A, canal {TCA_CHANNEL}")
        tca = TCA9548A(i2c)
        i2c_device = tca[TCA_CHANNEL]
    else:
        i2c_device = i2c

    matrix = HT16K33Matrix(i2c_device)</parameter>
        self.clear()

    def clear(self):
        """Éteint toutes les LEDs."""
        for i in range(16):
            self.ht_rows[i] = 0
            self.ht_cols[i] = 0
        self.led_state.clear()

    def set_led(self, row, col, state=True):
        """
        Active ou désactive une LED.

        Args:
            row: Ligne (1-9 pour B1-B9)
            col: Colonne (1-9 pour A1-A9)
            state: True pour allumer, False pour éteindre
        """
        if row < 1 or row > 9 or col < 1 or col > 9:
            return

        if state:
            self.led_state.add((row, col))
        else:
            self.led_state.discard((row, col))

    def update(self):
        """Met à jour l'affichage physique des HT16K33."""
        # Calculer les lignes et colonnes actives
        active_rows = set()
        active_cols = set()

        for row, col in self.led_state:
            active_rows.add(row)
            active_cols.add(col)

        # Mise à jour des lignes (B1-B9)
        row_bits = 0
        for row in active_rows:
            if 1 <= row <= 9:
                row_bits |= 1 << (row - 1)

        self.ht_rows[0] = row_bits & 0xFF  # B1-B8
        self.ht_rows[1] = (row_bits >> 8) & 0xFF  # B9

        # Mise à jour des colonnes (A1-A9)
        col_bits = 0
        for col in active_cols:
            if 1 <= col <= 9:
                col_bits |= 1 << (col - 1)

        self.ht_cols[0] = col_bits & 0xFF  # A1-A8
        self.ht_cols[1] = (col_bits >> 8) & 0xFF  # A9

    def set_brightness(self, brightness):
        """
        Ajuste la luminosité.

        Args:
            brightness: Valeur entre 0.0 et 1.0
        """
        self.ht_rows.brightness = brightness
        self.ht_cols.brightness = brightness

    def fill(self, state=True):
        """Allume ou éteint toutes les LEDs."""
        if state:
            for row in range(1, 10):
                for col in range(1, 10):
                    self.led_state.add((row, col))
        else:
            self.clear()


def test_individual_leds(matrix):
    """Test 1 : Allume chaque LED individuellement."""
    print("\n=== Test 1 : LEDs individuelles ===")
    matrix.clear()

    for row in range(1, 10):
        for col in range(1, 10):
            matrix.clear()
            matrix.set_led(row, col, True)
            matrix.update()
            print(f"LED B{row},A{col}", end="\r", flush=True)
            time.sleep(0.05)

    matrix.clear()
    matrix.update()
    print("\n✓ Test terminé")


def test_row_scan(matrix):
    """Test 2 : Balayage par lignes."""
    print("\n=== Test 2 : Balayage par lignes (B1-B9) ===")

    for row in range(1, 10):
        matrix.clear()
        for col in range(1, 10):
            matrix.set_led(row, col, True)
        matrix.update()
        print(f"Ligne B{row}", end="\r", flush=True)
        time.sleep(0.3)

    matrix.clear()
    matrix.update()
    print("\n✓ Test terminé")


def test_column_scan(matrix):
    """Test 3 : Balayage par colonnes."""
    print("\n=== Test 3 : Balayage par colonnes (A1-A9) ===")

    for col in range(1, 10):
        matrix.clear()
        for row in range(1, 10):
            matrix.set_led(row, col, True)
        matrix.update()
        print(f"Colonne A{col}", end="\r", flush=True)
        time.sleep(0.3)

    matrix.clear()
    matrix.update()
    print("\n✓ Test terminé")


def test_diagonals(matrix):
    """Test 4 : Diagonales."""
    print("\n=== Test 4 : Diagonales ===")

    # Diagonale principale (haut-gauche à bas-droite)
    print("Diagonale principale...")
    matrix.clear()
    for i in range(1, 10):
        matrix.set_led(i, 10 - i, True)
    matrix.update()
    time.sleep(1.5)

    # Diagonale secondaire (haut-droite à bas-gauche)
    print("Diagonale secondaire...")
    matrix.clear()
    for i in range(1, 10):
        matrix.set_led(i, i, True)
    matrix.update()
    time.sleep(1.5)

    # Les deux ensemble
    print("Croix diagonale...")
    for i in range(1, 10):
        matrix.set_led(i, 10 - i, True)
    matrix.update()
    time.sleep(1.5)

    matrix.clear()
    matrix.update()
    print("✓ Test terminé")


def test_checkerboard(matrix):
    """Test 5 : Motif damier."""
    print("\n=== Test 5 : Motif damier (échiquier) ===")

    # Damier classique
    print("Damier classique...")
    matrix.clear()
    for row in range(1, 10):
        for col in range(1, 10):
            if (row + col) % 2 == 0:
                matrix.set_led(row, col, True)
    matrix.update()
    time.sleep(2)

    # Damier inversé
    print("Damier inversé...")
    matrix.clear()
    for row in range(1, 10):
        for col in range(1, 10):
            if (row + col) % 2 == 1:
                matrix.set_led(row, col, True)
    matrix.update()
    time.sleep(2)

    matrix.clear()
    matrix.update()
    print("✓ Test terminé")


def test_borders(matrix):
    """Test 6 : Bordures et cadres."""
    print("\n=== Test 6 : Bordures ===")

    # Bordure extérieure
    print("Bordure extérieure...")
    matrix.clear()
    for i in range(1, 10):
        matrix.set_led(1, i, True)  # Ligne du haut
        matrix.set_led(9, i, True)  # Ligne du bas
        matrix.set_led(i, 1, True)  # Colonne de gauche
        matrix.set_led(i, 9, True)  # Colonne de droite
    matrix.update()
    time.sleep(2)

    # Cadres concentriques
    print("Cadres concentriques...")
    for offset in range(4):
        matrix.clear()
        for i in range(1 + offset, 10 - offset):
            matrix.set_led(1 + offset, i, True)
            matrix.set_led(9 - offset, i, True)
            matrix.set_led(i, 1 + offset, True)
            matrix.set_led(i, 9 - offset, True)
        matrix.update()
        time.sleep(0.5)

    matrix.clear()
    matrix.update()
    print("✓ Test terminé")


def test_brightness(matrix):
    """Test 7 : Variation de luminosité."""
    print("\n=== Test 7 : Variation de luminosité ===")

    # Remplir toute la matrice
    matrix.fill(True)

    # Monter progressivement
    print("Augmentation de la luminosité...")
    for brightness in range(0, 11):
        matrix.set_brightness(brightness / 10.0)
        matrix.update()
        print(f"Luminosité: {brightness * 10}%", end="\r", flush=True)
        time.sleep(0.2)

    print("\nDiminution de la luminosité...")
    for brightness in range(10, -1, -1):
        matrix.set_brightness(brightness / 10.0)
        matrix.update()
        print(f"Luminosité: {brightness * 10}%", end="\r", flush=True)
        time.sleep(0.2)

    # Retour à luminosité maximale
    matrix.set_brightness(1.0)
    matrix.clear()
    matrix.update()
    print("\n✓ Test terminé")


def test_animation_wave(matrix):
    """Test 8 : Animation vague."""
    print("\n=== Test 8 : Animation vague ===")

    for cycle in range(3):
        for offset in range(18):
            matrix.clear()
            for row in range(1, 10):
                col = ((row + offset) % 9) + 1
                if 1 <= col <= 9:
                    matrix.set_led(row, col, True)
            matrix.update()
            time.sleep(0.05)

    matrix.clear()
    matrix.update()
    print("✓ Test terminé")


def main():
    """Programme principal de test."""
    print("=" * 60)
    print("Test de la matrice LED HT16K33 9x9")
    print("=" * 60)

    # Initialisation
    i2c = busio.I2C(board.SCL, board.SDA)

    from adafruit_tca9548a import TCA9548A

    print(f"Utilisation du TCA9548A")
    print(f"  - Canal {TCA_ROWS_CHANNEL} : HT16K33 lignes")
    print(f"  - Canal {TCA_COLS_CHANNEL} : HT16K33 colonnes")
    tca = TCA9548A(i2c)

    matrix = HT16K33Matrix(tca)

    print("\nDémarrage des tests...")
    time.sleep(1)

    try:
        # Exécution des tests
        test_individual_leds(matrix)
        time.sleep(0.5)

        test_row_scan(matrix)
        time.sleep(0.5)

        test_column_scan(matrix)
        time.sleep(0.5)

        test_diagonals(matrix)
        time.sleep(0.5)

        test_checkerboard(matrix)
        time.sleep(0.5)

        test_borders(matrix)
        time.sleep(0.5)

        test_brightness(matrix)
        time.sleep(0.5)

        test_animation_wave(matrix)

        print("\n" + "=" * 60)
        print("Tous les tests sont terminés !")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTests interrompus par l'utilisateur")
    finally:
        matrix.clear()
        matrix.update()
        print("Programme terminé.")


if __name__ == "__main__":
    main()
