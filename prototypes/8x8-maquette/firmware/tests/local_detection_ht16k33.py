#!/usr/bin/env python3
"""
local_detection_ht16k33.py - Détection locale avec deux HT16K33

Ce programme surveille les 64 reed sensors de l'échiquier et allume
les LEDs correspondantes en utilisant deux HT16K33 :
- HT16K33 #1 : contrôle les 9 lignes (B1-B9)
- HT16K33 #2 : contrôle les 9 colonnes (A1-A9)

Pour allumer une LED à la position (Bx, Ay), on active:
- La ligne x sur HT16K33_rows
- La colonne y sur HT16K33_cols

Note: Une diagonale de LEDs (B9) peut ne pas fonctionner.
"""

import time
import board
import busio
from smbus2 import SMBus
from adafruit_ht16k33 import ht16k33

# --- Constantes ---
TCA_I2C_ADDR = 0x70
MCP_BASE_ADDR = 0x20
I2C_BUSNUM = 1

# Adresses I2C des HT16K33 (même adresse car sur canaux TCA différents)
HT16K33_ROWS_ADDR = 0x70  # HT16K33 pour les lignes
HT16K33_COLS_ADDR = 0x70  # HT16K33 pour les colonnes

# Canaux TCA9548A pour les HT16K33
TCA_ROWS_CHANNEL = 4  # Canal TCA pour HT16K33 lignes
TCA_COLS_CHANNEL = 5  # Canal TCA pour HT16K33 colonnes

# Registres MCP23017
GPIOA, GPIOB = 0x12, 0x13
IODIRA, IODIRB = 0x00, 0x01
GPPUA, GPPUB = 0x0C, 0x0D

# Mapping des canaux TCA vers les MCP23017
REED_TCA_CHANNELS = {
    0: 0,  # MCP0 (rows 1-2)
    1: 1,  # MCP1 (rows 3-4)
    2: 2,  # MCP2 (rows 5-6)
    3: 3,  # MCP3 (rows 7-8)
}

# Mapping complet: case -> (mcp_id, pin, led_positions)
CHESS_MAPPING = {
    # Rangée 1
    "a1": (0, "A0", [(1, 9), (1, 8), (2, 9), (2, 8)]),
    "b1": (0, "B0", [(1, 8), (1, 7), (2, 8), (2, 7)]),
    "c1": (0, "A1", [(1, 7), (1, 6), (2, 7), (2, 6)]),
    "d1": (0, "B1", [(1, 6), (1, 5), (2, 6), (2, 5)]),
    "e1": (0, "A2", [(1, 5), (1, 4), (2, 5), (2, 4)]),
    "f1": (0, "B2", [(1, 4), (1, 3), (2, 4), (2, 3)]),
    "g1": (0, "A3", [(1, 3), (1, 2), (2, 3), (2, 2)]),
    "h1": (0, "B3", [(1, 2), (1, 1), (2, 2), (2, 1)]),
    # Rangée 2
    "a2": (0, "B7", [(2, 9), (2, 8), (3, 9), (3, 8)]),
    "b2": (0, "A7", [(2, 8), (2, 7), (3, 8), (3, 7)]),
    "c2": (0, "B6", [(2, 7), (2, 6), (3, 7), (3, 6)]),
    "d2": (0, "A6", [(2, 6), (2, 5), (3, 6), (3, 5)]),
    "e2": (0, "B5", [(2, 5), (2, 4), (3, 5), (3, 4)]),
    "f2": (0, "A5", [(2, 4), (2, 3), (3, 4), (3, 3)]),
    "g2": (0, "B4", [(2, 3), (2, 2), (3, 3), (3, 2)]),
    "h2": (0, "A4", [(2, 2), (2, 1), (3, 2), (3, 1)]),
    # Rangée 3
    "a3": (1, "A0", [(3, 9), (3, 8), (4, 9), (4, 8)]),
    "b3": (1, "B0", [(3, 8), (3, 7), (4, 8), (4, 7)]),
    "c3": (1, "A1", [(3, 7), (3, 6), (4, 7), (4, 6)]),
    "d3": (1, "B1", [(3, 6), (3, 5), (4, 6), (4, 5)]),
    "e3": (1, "A2", [(3, 5), (3, 4), (4, 5), (4, 4)]),
    "f3": (1, "B2", [(3, 4), (3, 3), (4, 4), (4, 3)]),
    "g3": (1, "A3", [(3, 3), (3, 2), (4, 3), (4, 2)]),
    "h3": (1, "B3", [(3, 2), (3, 1), (4, 2), (4, 1)]),
    # Rangée 4
    "a4": (1, "B7", [(4, 9), (4, 8), (5, 9), (5, 8)]),
    "b4": (1, "A7", [(4, 8), (4, 7), (5, 8), (5, 7)]),
    "c4": (1, "B6", [(4, 7), (4, 6), (5, 7), (5, 6)]),
    "d4": (1, "A6", [(4, 6), (4, 5), (5, 6), (5, 5)]),
    "e4": (1, "B5", [(4, 5), (4, 4), (5, 5), (5, 4)]),
    "f4": (1, "A5", [(4, 4), (4, 3), (5, 4), (5, 3)]),
    "g4": (1, "B4", [(4, 3), (4, 2), (5, 3), (5, 2)]),
    "h4": (1, "A4", [(4, 2), (4, 1), (5, 2), (5, 1)]),
    # Rangée 5
    "a5": (2, "A0", [(5, 9), (5, 8), (6, 9), (6, 8)]),
    "b5": (2, "B0", [(5, 8), (5, 7), (6, 8), (6, 7)]),
    "c5": (2, "A1", [(5, 7), (5, 6), (6, 7), (6, 6)]),
    "d5": (2, "B1", [(5, 6), (5, 5), (6, 6), (6, 5)]),
    "e5": (2, "A2", [(5, 5), (5, 4), (6, 5), (6, 4)]),
    "f5": (2, "B2", [(5, 4), (5, 3), (6, 4), (6, 3)]),
    "g5": (2, "A3", [(5, 3), (5, 2), (6, 3), (6, 2)]),
    "h5": (2, "B3", [(5, 2), (5, 1), (6, 2), (6, 1)]),
    # Rangée 6
    "a6": (2, "B7", [(6, 9), (6, 8), (7, 9), (7, 8)]),
    "b6": (2, "A7", [(6, 8), (6, 7), (7, 8), (7, 7)]),
    "c6": (2, "B6", [(6, 7), (6, 6), (7, 7), (7, 6)]),
    "d6": (2, "A6", [(6, 6), (6, 5), (7, 6), (7, 5)]),
    "e6": (2, "B5", [(6, 5), (6, 4), (7, 5), (7, 4)]),
    "f6": (2, "A5", [(6, 4), (6, 3), (7, 4), (7, 3)]),
    "g6": (2, "B4", [(6, 3), (6, 2), (7, 3), (7, 2)]),
    "h6": (2, "A4", [(6, 2), (6, 1), (7, 2), (7, 1)]),
    # Rangée 7
    "a7": (3, "A0", [(7, 9), (7, 8), (8, 9), (8, 8)]),
    "b7": (3, "B0", [(7, 8), (7, 7), (8, 8), (8, 7)]),
    "c7": (3, "A1", [(7, 7), (7, 6), (8, 7), (8, 6)]),
    "d7": (3, "B1", [(7, 6), (7, 5), (8, 6), (8, 5)]),
    "e7": (3, "A2", [(7, 5), (7, 4), (8, 5), (8, 4)]),
    "f7": (3, "B2", [(7, 4), (7, 3), (8, 4), (8, 3)]),
    "g7": (3, "A3", [(7, 3), (7, 2), (8, 3), (8, 2)]),
    "h7": (3, "B3", [(7, 2), (7, 1), (8, 2), (8, 1)]),
    # Rangée 8
    "a8": (3, "B7", [(8, 9), (8, 8), (9, 9), (9, 8)]),
    "b8": (3, "A7", [(8, 8), (8, 7), (9, 8), (9, 7)]),
    "c8": (3, "B6", [(8, 7), (8, 6), (9, 7), (9, 6)]),
    "d8": (3, "A6", [(8, 6), (8, 5), (9, 6), (9, 5)]),
    "e8": (3, "B5", [(8, 5), (8, 4), (9, 5), (9, 4)]),
    "f8": (3, "A5", [(8, 4), (8, 3), (9, 4), (9, 3)]),
    "g8": (3, "B4", [(8, 3), (8, 2), (9, 3), (9, 2)]),
    "h8": (3, "A4", [(8, 2), (8, 1), (9, 2), (9, 1)]),
}

# Diagonale cassée (ligne B9)
BROKEN_LEDS = [(9, i) for i in range(1, 10)]


class HT16K33Matrix:
    """Contrôleur pour matrice LED avec deux HT16K33 via TCA9548A."""

    def __init__(
        self,
        tca,
        rows_channel=TCA_ROWS_CHANNEL,
        cols_channel=TCA_COLS_CHANNEL,
        rows_addr=HT16K33_ROWS_ADDR,
        cols_addr=HT16K33_COLS_ADDR,
    ):
        """
        Initialise les deux HT16K33 sur le TCA9548A.

        Args:
            tca: Instance du TCA9548A
            rows_channel: Canal TCA pour le HT16K33 des lignes
            cols_channel: Canal TCA pour le HT16K33 des colonnes
            rows_addr: Adresse I2C du HT16K33 lignes
            cols_addr: Adresse I2C du HT16K33 colonnes
        """
        print(
            f"  Initialisation HT16K33 lignes (canal TCA {rows_channel}, addr 0x{rows_addr:02X})..."
        )
        i2c_rows = tca[rows_channel]
        self.ht_rows = ht16k33.HT16K33(i2c_rows, address=rows_addr)
        self.ht_rows.blink_rate = 0
        self.ht_rows.brightness = 1.0

        print(
            f"  Initialisation HT16K33 colonnes (canal TCA {cols_channel}, addr 0x{cols_addr:02X})..."
        )
        i2c_cols = tca[cols_channel]
        self.ht_cols = ht16k33.HT16K33(i2c_cols, address=cols_addr)
        self.ht_cols.blink_rate = 0
        self.ht_cols.brightness = 1.0

        # État actuel de la matrice
        self.led_state = {}  # {(row, col): True/False}

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
        # Filtrer la ligne B9 si cassée
        if (row, col) in BROKEN_LEDS:
            return

        if row < 1 or row > 9 or col < 1 or col > 9:
            return

        if state:
            self.led_state[(row, col)] = True
        else:
            self.led_state.pop((row, col), None)

    def update(self):
        """
        Met à jour l'affichage physique des HT16K33.

        Le HT16K33 utilise un système de 16 lignes de 8 bits.
        Pour notre matrice 9x9, nous utilisons:
        - HT_rows: contrôle quelle ligne est active (B1-B9)
        - HT_cols: contrôle quelle colonne est active (A1-A9)

        Chaque LED nécessite que sa ligne ET sa colonne soient activées.
        """
        # Calculer quelles lignes et colonnes doivent être actives
        active_rows = set()
        active_cols = set()

        for row, col in self.led_state.keys():
            active_rows.add(row)
            active_cols.add(col)

        # Mise à jour du HT16K33 des lignes
        # On utilise les 9 premiers bits du premier registre
        row_bits = 0
        for row in active_rows:
            if 1 <= row <= 9:
                row_bits |= 1 << (row - 1)

        self.ht_rows[0] = row_bits & 0xFF  # Bits 0-7 (B1-B8)
        self.ht_rows[1] = (row_bits >> 8) & 0xFF  # Bit 8 (B9)

        # Mise à jour du HT16K33 des colonnes
        col_bits = 0
        for col in active_cols:
            if 1 <= col <= 9:
                col_bits |= 1 << (col - 1)

        self.ht_cols[0] = col_bits & 0xFF  # Bits 0-7 (A1-A8)
        self.ht_cols[1] = (col_bits >> 8) & 0xFF  # Bit 8 (A9)

    def set_leds_batch(self, led_positions, state=True):
        """
        Active/désactive plusieurs LEDs en une fois.

        Args:
            led_positions: Liste de tuples (row, col)
            state: True pour allumer, False pour éteindre
        """
        for row, col in led_positions:
            self.set_led(row, col, state)


class ChessDetectorHT16K33:
    """Détecteur de pièces d'échecs avec HT16K33."""

    def __init__(self):
        """Initialise les composants matériels."""
        print("Initialisation du détecteur d'échecs (HT16K33)...")

        # Initialisation du bus I²C
        self.i2c = busio.I2C(board.SCL, board.SDA)

        # Initialisation du TCA9548A
        from adafruit_tca9548a import TCA9548A

        self.tca = TCA9548A(self.i2c)

        # Initialisation de la matrice HT16K33 via TCA
        self.matrix = HT16K33Matrix(self.tca)
        print(f"  ✓ Matrice LED HT16K33 initialisée")

        # Initialisation du bus I2C pour les reed sensors (SMBus)
        self.bus = SMBus(I2C_BUSNUM)

        # Initialisation des MCP23017
        self.mcp_initialized = {}
        for mcp_id in range(4):
            channel = REED_TCA_CHANNELS[mcp_id]
            if self._init_mcp(mcp_id, channel):
                self.mcp_initialized[mcp_id] = True
                print(
                    f"  ✓ MCP{mcp_id} initialisé (canal TCA {channel}, addr 0x{MCP_BASE_ADDR + mcp_id:02X})"
                )
            else:
                self.mcp_initialized[mcp_id] = False
                print(f"  ✗ MCP{mcp_id} non détecté")

        # État précédent des cases actives
        self.previous_active_squares = set()

        print("Initialisation terminée!\n")

    def _tca_select(self, channel):
        """Sélectionne un canal sur le multiplexeur TCA9548A."""
        if channel is None:
            self.bus.write_byte(TCA_I2C_ADDR, 0x00)
        else:
            self.bus.write_byte(TCA_I2C_ADDR, 1 << channel)

    def _init_mcp(self, mcp_id, tca_channel):
        """Initialise un MCP23017."""
        try:
            self._tca_select(tca_channel)
            time.sleep(0.01)

            addr = MCP_BASE_ADDR + mcp_id
            self.bus.write_byte_data(addr, IODIRA, 0xFF)
            self.bus.write_byte_data(addr, IODIRB, 0xFF)
            self.bus.write_byte_data(addr, GPPUA, 0xFF)
            self.bus.write_byte_data(addr, GPPUB, 0xFF)
            return True
        except OSError:
            return False

    def _read_mcp_gpio(self, mcp_id, tca_channel):
        """Lit les états GPIO d'un MCP23017."""
        try:
            self._tca_select(tca_channel)
            time.sleep(0.005)

            addr = MCP_BASE_ADDR + mcp_id
            gpio_a = self.bus.read_byte_data(addr, GPIOA)
            gpio_b = self.bus.read_byte_data(addr, GPIOB)
            return gpio_a, gpio_b
        except OSError:
            return None, None

    def _is_pin_active(self, gpio_value, pin_name):
        """Vérifie si un pin est actif (bit à 0 = contact fermé)."""
        if gpio_value is None:
            return False

        port = pin_name[0]
        pin_num = int(pin_name[1])

        return not bool(gpio_value & (1 << pin_num))

    def scan_active_squares(self):
        """Scanne tous les reed sensors et retourne les cases actives."""
        active_squares = set()

        for square, (mcp_id, pin_name, _) in CHESS_MAPPING.items():
            if not self.mcp_initialized.get(mcp_id, False):
                continue

            channel = REED_TCA_CHANNELS[mcp_id]
            gpio_a, gpio_b = self._read_mcp_gpio(mcp_id, channel)

            if gpio_a is None:
                continue

            gpio_value = gpio_a if pin_name[0] == "A" else gpio_b

            if self._is_pin_active(gpio_value, pin_name):
                active_squares.add(square)

        return active_squares

    def update_leds(self, active_squares):
        """Met à jour l'affichage LED en fonction des cases actives."""
        if active_squares == self.previous_active_squares:
            return

        # Éteindre toutes les LEDs
        self.matrix.clear()

        # Allumer les LEDs pour chaque case active
        for square in active_squares:
            _, _, led_positions = CHESS_MAPPING[square]
            self.matrix.set_leds_batch(led_positions, state=True)

        # Mettre à jour l'affichage physique
        self.matrix.update()

        self.previous_active_squares = active_squares.copy()

    def run(self):
        """Boucle principale de détection."""
        print("Détection en cours... (Ctrl+C pour arrêter)")
        print("-" * 60)

        try:
            while True:
                # Scanner les cases actives
                active_squares = self.scan_active_squares()

                # Mettre à jour les LEDs
                self.update_leds(active_squares)

                # Afficher l'état si des pièces sont détectées
                if active_squares:
                    squares_list = sorted(active_squares)
                    print(
                        f"\rPièces détectées: {', '.join(squares_list)}",
                        end="",
                        flush=True,
                    )
                elif self.previous_active_squares:
                    print("\r" + " " * 60 + "\r", end="", flush=True)

                time.sleep(0.1)  # 10 Hz de rafraîchissement

        except KeyboardInterrupt:
            print("\n\nArrêt du programme...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Nettoie les ressources."""
        print("Nettoyage...")
        self.matrix.clear()
        self.matrix.update()
        self._tca_select(None)
        self.bus.close()
        print("Programme terminé.")


def main():
    """Point d'entrée principal."""
    detector = ChessDetectorHT16K33()
    detector.run()


if __name__ == "__main__":
    main()
