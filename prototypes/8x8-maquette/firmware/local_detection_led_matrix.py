#!/usr/bin/env python3
"""
local_detection.py - Détection locale des pièces d'échecs et allumage des LEDs

Ce programme surveille en continu les 64 reed sensors de l'échiquier et allume
les 4 LEDs autour de chaque case où un aimant est détecté.

Note: Une diagonale de LEDs ne fonctionne pas (B9), donc certaines cases
n'auront que 2 ou 3 LEDs allumées au lieu de 4.
"""

import time
import board
import busio
from smbus2 import SMBus
from adafruit_tca9548a import TCA9548A
from adafruit_is31fl3731.matrix import Matrix

# --- Constantes ---
TCA_I2C_ADDR = 0x70
MCP_BASE_ADDR = 0x20
I2C_BUSNUM = 1
LED_TCA_CHANNEL = 4  # Canal TCA pour la matrice LED
LED_BRIGHTNESS = 200  # Intensité des LEDs (0-255)

# Registres MCP23017
GPIOA, GPIOB = 0x12, 0x13
IODIRA, IODIRB = 0x00, 0x01
GPPUA, GPPUB = 0x0C, 0x0D

# Mapping des canaux TCA vers les MCP23017
# Canal 0 = MCP0 (rows 1-2)
# Canal 1 = MCP1 (rows 3-4)
# Canal 2 = MCP2 (rows 5-6)
# Canal 3 = MCP3 (rows 7-8)
REED_TCA_CHANNELS = {
    0: 0,  # MCP0
    1: 1,  # MCP1
    2: 2,  # MCP2
    3: 3,  # MCP3
}

# Mapping complet: case -> (mcp_id, pin, led_positions)
# Format: 'a1': (mcp_id, 'A0', [(1, 9), (1, 8), (2, 9), (2, 8)])
# Les coordonnées LED sont en (row_B, col_A) où B1-B9, A1-A9
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

# Diagonale cassée (B9) - filtrer ces LEDs
BROKEN_LEDS = [(9, i) for i in range(1, 10)]  # Toute la ligne B9


class ChessDetector:
    """Détecteur de pièces d'échecs avec affichage LED."""

    def __init__(self):
        """Initialise les composants matériels."""
        print("Initialisation du détecteur d'échecs...")

        # Initialisation du bus I2C pour la matrice LED
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.tca = TCA9548A(self.i2c)

        # Initialisation de la matrice LED
        i2c_led = self.tca[LED_TCA_CHANNEL]
        self.display = Matrix(i2c_led)
        self.display.fill(0)
        print(f"  ✓ Matrice LED initialisée (canal TCA {LED_TCA_CHANNEL})")

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

        # État précédent des cases actives pour optimisation
        self.previous_active_squares = set()

        print("Initialisation terminée!\n")

    def _tca_select(self, channel):
        """Sélectionne un canal sur le multiplexeur TCA9548A (via SMBus)."""
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
            # Configuration: tous les pins en entrée avec pull-up
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

        # Extraire le numéro de pin (A0-A7 ou B0-B7)
        port = pin_name[0]
        pin_num = int(pin_name[1])

        # Le bit est à 0 quand le reed est activé
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

            # Déterminer quelle valeur GPIO utiliser
            gpio_value = gpio_a if pin_name[0] == "A" else gpio_b

            if self._is_pin_active(gpio_value, pin_name):
                active_squares.add(square)

        return active_squares

    def update_leds(self, active_squares):
        """Met à jour l'affichage LED en fonction des cases actives."""
        # Optimisation: ne mettre à jour que si changement
        if active_squares == self.previous_active_squares:
            return

        # Éteindre toutes les LEDs
        self.display.fill(0)

        # Allumer les LEDs pour chaque case active
        for square in active_squares:
            _, _, led_positions = CHESS_MAPPING[square]

            for row, col in led_positions:
                # Filtrer la diagonale cassée
                if (row, col) in BROKEN_LEDS:
                    continue

                # Convertir les coordonnées (B row, A col) en (x, y) pour la matrice
                # La matrice utilise (x=col-1, y=row-1) car indexé de 0
                x = row - 1  # B1-B9 -> 0-8
                y = col - 1  # A1-A9 -> 0-8

                try:
                    self.display.pixel(x, y, LED_BRIGHTNESS)
                except (ValueError, IndexError):
                    # Ignorer les coordonnées hors limites
                    pass

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
                    # Effacer la ligne quand toutes les pièces sont retirées
                    print("\r" + " " * 60 + "\r", end="", flush=True)

                time.sleep(0.1)  # 10 Hz de rafraîchissement

        except KeyboardInterrupt:
            print("\n\nArrêt du programme...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Nettoie les ressources."""
        print("Nettoyage...")
        self.display.fill(0)
        self._tca_select(None)
        self.bus.close()
        print("Programme terminé.")


def main():
    """Point d'entrée principal."""
    detector = ChessDetector()
    detector.run()


if __name__ == "__main__":
    main()
