#!/usr/bin/env python3
"""
Interface PvP Remote - Hardware Bridge
Abstraction du matériel I2C pour le plateau d'échecs intelligent.
Supporte un mode simulation pour le développement sur Mac.
"""

import os
import time
import threading
from typing import Optional, Tuple, List, Dict, Callable

# Détection du mode hardware
HARDWARE_AVAILABLE = False
try:
    import board
    import busio
    import digitalio
    import adafruit_tca9548a
    from adafruit_ht16k33.matrix import Matrix16x8
    from adafruit_mcp230xx.mcp23017 import MCP23017
    HARDWARE_AVAILABLE = True
except ImportError:
    pass


# LED Map - Correspondance (row, col) -> (canal, sw_col, sw_row)
LED_MAP = {
    # Canal 4 (Matrice 8x8)
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
    # Canal 5 (Rangée supplémentaire)
    (8, 8): ("C5", 0, 0), (7, 8): ("C5", 0, 1), (6, 8): ("C5", 0, 2), (5, 8): ("C5", 0, 3),
    (4, 8): ("C5", 0, 4), (3, 8): ("C5", 0, 5), (2, 8): ("C5", 0, 6), (1, 8): ("C5", 0, 7),
    (0, 8): ("C5", 2, 0), (0, 7): ("C5", 1, 7), (0, 6): ("C5", 1, 6), (0, 5): ("C5", 1, 5),
    (0, 4): ("C5", 1, 4), (0, 3): ("C5", 1, 3), (0, 2): ("C5", 1, 2), (0, 1): ("C5", 1, 1),
    (0, 0): ("C5", 1, 0),
}


class HardwareBridge:
    """Interface d'abstraction pour le matériel I2C du plateau."""
    
    def __init__(self, simulation: bool = False):
        self.simulation = simulation or not HARDWARE_AVAILABLE
        self.display_matrix = None
        self.display_row = None
        self.sensor_pins = None
        self.mcps = None
        
        # État simulé (8x8 - True = pièce présente)
        self._simulated_state = [[False for _ in range(8)] for _ in range(8)]
        self._simulated_leds = [[False for _ in range(8)] for _ in range(8)]
        
        # Callbacks
        self._on_state_change: Optional[Callable] = None
        
        # Thread de polling
        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None
        self._last_state = None
        
        if self.simulation:
            print("[HardwareBridge] Mode SIMULATION activé")
            self._init_simulated_state()
        else:
            print("[HardwareBridge] Mode HARDWARE activé")
            self._init_hardware()
    
    def _init_simulated_state(self):
        """Initialise l'état simulé avec les pièces en position de départ."""
        # Rangées 0, 1 (blancs) et 6, 7 (noirs) ont des pièces
        for col in range(8):
            self._simulated_state[0][col] = True  # Rangée 1 (pièces blanches)
            self._simulated_state[1][col] = True  # Rangée 2 (pions blancs)
            self._simulated_state[6][col] = True  # Rangée 7 (pions noirs)
            self._simulated_state[7][col] = True  # Rangée 8 (pièces noires)
        self._last_state = [row[:] for row in self._simulated_state]
    
    def _init_hardware(self):
        """Initialise le matériel I2C."""
        try:
            # Bus I2C
            i2c = busio.I2C(board.SCL, board.SDA)
            
            # Multiplexeur
            tca = adafruit_tca9548a.TCA9548A(i2c, address=0x72)
            
            # Contrôleurs LED
            self.display_matrix = Matrix16x8(tca[4], address=0x70)
            self.display_row = Matrix16x8(tca[5], address=0x71)
            self.display_matrix.brightness = 0.3
            self.display_row.brightness = 0.3
            self.display_matrix.fill(0)
            self.display_row.fill(0)
            self.display_matrix.show()
            self.display_row.show()
            
            # Contrôleurs MCP23017
            self.mcps = [
                MCP23017(tca[0], address=0x20),
                MCP23017(tca[1], address=0x20),
                MCP23017(tca[2], address=0x20),
                MCP23017(tca[3], address=0x20),
            ]
            
            # Matrice des pins capteurs
            self.sensor_pins = [[None for _ in range(8)] for _ in range(8)]
            pin_map_a = [0, 8, 1, 9, 2, 10, 3, 11]
            pin_map_b = [15, 7, 14, 6, 13, 5, 12, 4]
            
            for row in range(4):
                for col in range(8):
                    self.sensor_pins[row * 2][col] = self.mcps[row].get_pin(pin_map_a[col])
                    self.sensor_pins[row * 2 + 1][col] = self.mcps[row].get_pin(pin_map_b[col])
            
            # Configuration des pins
            for row in range(8):
                for col in range(8):
                    pin = self.sensor_pins[row][col]
                    pin.direction = digitalio.Direction.INPUT
                    pin.pull = digitalio.Pull.UP
            
            print("[HardwareBridge] Matériel initialisé avec succès")
            self._last_state = self.get_board_state()
            
        except Exception as e:
            print(f"[HardwareBridge] Erreur initialisation hardware: {e}")
            print("[HardwareBridge] Basculement en mode simulation")
            self.simulation = True
            self._init_simulated_state()
    
    def get_board_state(self) -> List[List[bool]]:
        """Retourne l'état actuel du plateau (8x8, True = pièce présente)."""
        if self.simulation:
            return [row[:] for row in self._simulated_state]
        
        state = [[False for _ in range(8)] for _ in range(8)]
        for r in range(8):
            for c in range(8):
                state[r][c] = not self.sensor_pins[r][c].value
        return state
    
    def set_square_leds(self, row: int, col: int, state: int):
        """Allume (1) ou éteint (0) les 4 LEDs d'une case."""
        if self.simulation:
            self._simulated_leds[row][col] = bool(state)
            led_char = "●" if state else "○"
            print(f"[SIM LED] Case ({row},{col}) = {led_char}")
            return
        
        # Conversion coordonnées capteur -> LED
        target_col = 7 - col
        target_row = 7 - row
        
        physical_corners = [
            (target_col, target_row),
            (target_col + 1, target_row),
            (target_col, target_row + 1),
            (target_col + 1, target_row + 1),
        ]
        
        for px, py in physical_corners:
            if (px, py) in LED_MAP:
                canal, sw_col, sw_row = LED_MAP[(px, py)]
                if canal == "C4" and self.display_matrix:
                    self.display_matrix.pixel(sw_col, sw_row, state)
                elif canal == "C5" and self.display_row:
                    self.display_row.pixel(sw_col, sw_row, state)
    
    def clear_all_leds(self):
        """Éteint toutes les LEDs."""
        if self.simulation:
            self._simulated_leds = [[False for _ in range(8)] for _ in range(8)]
            print("[SIM LED] Toutes les LEDs éteintes")
            return
        
        if self.display_matrix:
            self.display_matrix.fill(0)
            self.display_matrix.show()
        if self.display_row:
            self.display_row.fill(0)
            self.display_row.show()
    
    def highlight_move(self, from_square: str, to_square: str):
        """
        Illumine les cases de départ et d'arrivée d'un coup.
        Les carrés sont en notation algébrique (ex: 'e2', 'e4').
        """
        self.clear_all_leds()
        
        # Conversion notation algébrique -> (row, col)
        from_col = ord(from_square[0]) - ord('a')
        from_row = int(from_square[1]) - 1
        to_col = ord(to_square[0]) - ord('a')
        to_row = int(to_square[1]) - 1
        
        self.set_square_leds(from_row, from_col, 1)
        self.set_square_leds(to_row, to_col, 1)
        
        if not self.simulation:
            if self.display_matrix:
                self.display_matrix.show()
            if self.display_row:
                self.display_row.show()
        
        print(f"[LED] Illumination: {from_square} -> {to_square}")
    
    def simulate_move(self, from_square: str, to_square: str):
        """
        Simule un déplacement de pièce sur le plateau (mode simulation uniquement).
        Utile pour tester sans hardware.
        """
        if not self.simulation:
            return
        
        from_col = ord(from_square[0]) - ord('a')
        from_row = int(from_square[1]) - 1
        to_col = ord(to_square[0]) - ord('a')
        to_row = int(to_square[1]) - 1
        
        self._simulated_state[from_row][from_col] = False
        self._simulated_state[to_row][to_col] = True
        print(f"[SIM BOARD] Pièce déplacée: {from_square} -> {to_square}")
    
    def set_state_change_callback(self, callback: Callable[[List[Tuple[int, int]], List[Tuple[int, int]]], None]):
        """Définit le callback appelé lors d'un changement d'état du plateau."""
        self._on_state_change = callback
    
    def start_polling(self, interval: float = 0.1):
        """Démarre le polling des capteurs."""
        if self._polling:
            return
        
        self._polling = True
        self._poll_thread = threading.Thread(target=self._poll_loop, args=(interval,), daemon=True)
        self._poll_thread.start()
        print(f"[HardwareBridge] Polling démarré (intervalle: {interval}s)")
    
    def stop_polling(self):
        """Arrête le polling des capteurs."""
        self._polling = False
        if self._poll_thread:
            self._poll_thread.join(timeout=1)
        print("[HardwareBridge] Polling arrêté")
    
    def _poll_loop(self, interval: float):
        """Boucle de polling des capteurs."""
        while self._polling:
            current_state = self.get_board_state()
            
            if self._last_state and current_state != self._last_state:
                lifted, placed = self._find_diffs(self._last_state, current_state)
                
                if (lifted or placed) and self._on_state_change:
                    self._on_state_change(lifted, placed)
                
                self._last_state = current_state
            
            time.sleep(interval)
    
    def _find_diffs(self, old_state: List[List[bool]], new_state: List[List[bool]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Compare deux états et retourne les cases soulevées et posées."""
        lifted = []
        placed = []
        
        for r in range(8):
            for c in range(8):
                if old_state[r][c] and not new_state[r][c]:
                    lifted.append((r, c))
                elif not old_state[r][c] and new_state[r][c]:
                    placed.append((r, c))
        
        return lifted, placed
    
    def cleanup(self):
        """Nettoie les ressources."""
        self.stop_polling()
        self.clear_all_leds()
        print("[HardwareBridge] Cleanup terminé")


if __name__ == "__main__":
    # Test rapide
    bridge = HardwareBridge(simulation=True)
    
    print("\n--- Test LED ---")
    bridge.highlight_move("e2", "e4")
    
    print("\n--- État du plateau ---")
    state = bridge.get_board_state()
    for row in range(7, -1, -1):
        line = f"{row + 1} |"
        for col in range(8):
            line += " X" if state[row][col] else " ."
        print(line)
    print("   ----------------")
    print("    a b c d e f g h")
    
    bridge.cleanup()
