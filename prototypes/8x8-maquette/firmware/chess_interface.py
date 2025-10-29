#!/usr/bin/env python3
"""
chess_interface.py - Interface de communication entre l'IA et l'échiquier électronique

Ce module fournit une API simple pour afficher des coups d'échecs sur l'échiquier
électronique en utilisant le format UCI (Universal Chess Interface).

Usage:
    from chess_interface import ChessboardLED

    board = ChessboardLED()
    board.display_move("e2e4")  # Affiche le coup e2-e4
    board.clear()  # Éteint toutes les LEDs

Auteur: Smart Chess Project
Version: 1.0
"""

import re
import time
import board
import busio
from smbus2 import SMBus
from adafruit_tca9548a import TCA9548A
from adafruit_ht16k33 import ht16k33
from typing import Optional, List, Tuple, Dict
from enum import Enum


# --- Constants ---
TCA_I2C_ADDR = 0x70
HT16K33_ROWS_ADDR = 0x70
HT16K33_COLS_ADDR = 0x70
TCA_ROWS_CHANNEL = 4
TCA_COLS_CHANNEL = 5

# Validation UCI
UCI_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


class DisplayMode(Enum):
    """Modes d'affichage pour les coups."""

    STATIC = "static"  # Affichage fixe
    BLINK_SLOW = "blink_slow"  # Clignotement lent (1 Hz)
    BLINK_FAST = "blink_fast"  # Clignotement rapide (2 Hz)
    FADE = "fade"  # Fondu progressif
    ANIMATION = "animation"  # Animation de déplacement


class ChessMove:
    """Représente un coup d'échecs au format UCI."""

    def __init__(self, uci: str):
        """
        Initialise un coup à partir d'une chaîne UCI.

        Args:
            uci: Coup au format UCI (ex: "e2e4", "e7e8q")

        Raises:
            ValueError: Si le format UCI est invalide
        """
        if not self.validate_uci(uci):
            raise ValueError(f"Invalid UCI format: {uci}")

        self.uci = uci.lower()
        self.from_square = uci[0:2]
        self.to_square = uci[2:4]
        self.promotion = uci[4] if len(uci) == 5 else None

    @staticmethod
    def validate_uci(uci: str) -> bool:
        """Valide le format d'un coup UCI."""
        return bool(UCI_PATTERN.match(uci.lower()))

    def __str__(self):
        return self.uci

    def __repr__(self):
        return f"ChessMove('{self.uci}')"


class HT16K33Matrix:
    """Contrôleur pour matrice LED 9x9 avec deux HT16K33 via TCA9548A."""

    def __init__(
        self,
        tca,
        rows_channel=TCA_ROWS_CHANNEL,
        cols_channel=TCA_COLS_CHANNEL,
        rows_addr=HT16K33_ROWS_ADDR,
        cols_addr=HT16K33_COLS_ADDR,
    ):
        """Initialise les deux HT16K33."""
        i2c_rows = tca[rows_channel]
        self.ht_rows = ht16k33.HT16K33(i2c_rows, address=rows_addr)
        self.ht_rows.blink_rate = 0
        self.ht_rows.brightness = 1.0

        i2c_cols = tca[cols_channel]
        self.ht_cols = ht16k33.HT16K33(i2c_cols, address=cols_addr)
        self.ht_cols.blink_rate = 0
        self.ht_cols.brightness = 1.0

        self.led_state = set()
        self.clear()

    def clear(self):
        """Éteint toutes les LEDs."""
        for i in range(16):
            self.ht_rows[i] = 0
            self.ht_cols[i] = 0
        self.led_state.clear()

    def set_led(self, row: int, col: int, state: bool = True):
        """Active ou désactive une LED."""
        if row < 1 or row > 9 or col < 1 or col > 9:
            return

        if state:
            self.led_state.add((row, col))
        else:
            self.led_state.discard((row, col))

    def update(self):
        """Met à jour l'affichage physique des HT16K33."""
        active_rows = set()
        active_cols = set()

        for row, col in self.led_state:
            active_rows.add(row)
            active_cols.add(col)

        # Mise à jour des lignes
        row_bits = 0
        for row in active_rows:
            if 1 <= row <= 9:
                row_bits |= 1 << (row - 1)

        self.ht_rows[0] = row_bits & 0xFF
        self.ht_rows[1] = (row_bits >> 8) & 0xFF

        # Mise à jour des colonnes
        col_bits = 0
        for col in active_cols:
            if 1 <= col <= 9:
                col_bits |= 1 << (col - 1)

        self.ht_cols[0] = col_bits & 0xFF
        self.ht_cols[1] = (col_bits >> 8) & 0xFF

    def set_brightness(self, brightness: float):
        """Ajuste la luminosité (0.0 - 1.0)."""
        self.ht_rows.brightness = brightness
        self.ht_cols.brightness = brightness


class ChessboardLED:
    """
    Interface principale pour contrôler l'affichage LED de l'échiquier.

    Cette classe gère la conversion des coups UCI en illumination de LEDs
    et fournit diverses méthodes d'affichage.
    """

    # Mapping: case -> (mcp_id, pin, led_positions)
    SQUARE_TO_LEDS = {
        # Rangée 1
        "a1": [(1, 9), (1, 8), (2, 9), (2, 8)],
        "b1": [(1, 8), (1, 7), (2, 8), (2, 7)],
        "c1": [(1, 7), (1, 6), (2, 7), (2, 6)],
        "d1": [(1, 6), (1, 5), (2, 6), (2, 5)],
        "e1": [(1, 5), (1, 4), (2, 5), (2, 4)],
        "f1": [(1, 4), (1, 3), (2, 4), (2, 3)],
        "g1": [(1, 3), (1, 2), (2, 3), (2, 2)],
        "h1": [(1, 2), (1, 1), (2, 2), (2, 1)],
        # Rangée 2
        "a2": [(2, 9), (2, 8), (3, 9), (3, 8)],
        "b2": [(2, 8), (2, 7), (3, 8), (3, 7)],
        "c2": [(2, 7), (2, 6), (3, 7), (3, 6)],
        "d2": [(2, 6), (2, 5), (3, 6), (3, 5)],
        "e2": [(2, 5), (2, 4), (3, 5), (3, 4)],
        "f2": [(2, 4), (2, 3), (3, 4), (3, 3)],
        "g2": [(2, 3), (2, 2), (3, 3), (3, 2)],
        "h2": [(2, 2), (2, 1), (3, 2), (3, 1)],
        # Rangée 3
        "a3": [(3, 9), (3, 8), (4, 9), (4, 8)],
        "b3": [(3, 8), (3, 7), (4, 8), (4, 7)],
        "c3": [(3, 7), (3, 6), (4, 7), (4, 6)],
        "d3": [(3, 6), (3, 5), (4, 6), (4, 5)],
        "e3": [(3, 5), (3, 4), (4, 5), (4, 4)],
        "f3": [(3, 4), (3, 3), (4, 4), (4, 3)],
        "g3": [(3, 3), (3, 2), (4, 3), (4, 2)],
        "h3": [(3, 2), (3, 1), (4, 2), (4, 1)],
        # Rangée 4
        "a4": [(4, 9), (4, 8), (5, 9), (5, 8)],
        "b4": [(4, 8), (4, 7), (5, 8), (5, 7)],
        "c4": [(4, 7), (4, 6), (5, 7), (5, 6)],
        "d4": [(4, 6), (4, 5), (5, 6), (5, 5)],
        "e4": [(4, 5), (4, 4), (5, 5), (5, 4)],
        "f4": [(4, 4), (4, 3), (5, 4), (5, 3)],
        "g4": [(4, 3), (4, 2), (5, 3), (5, 2)],
        "h4": [(4, 2), (4, 1), (5, 2), (5, 1)],
        # Rangée 5
        "a5": [(5, 9), (5, 8), (6, 9), (6, 8)],
        "b5": [(5, 8), (5, 7), (6, 8), (6, 7)],
        "c5": [(5, 7), (5, 6), (6, 7), (6, 6)],
        "d5": [(5, 6), (5, 5), (6, 6), (6, 5)],
        "e5": [(5, 5), (5, 4), (6, 5), (6, 4)],
        "f5": [(5, 4), (5, 3), (6, 4), (6, 3)],
        "g5": [(5, 3), (5, 2), (6, 3), (6, 2)],
        "h5": [(5, 2), (5, 1), (6, 2), (6, 1)],
        # Rangée 6
        "a6": [(6, 9), (6, 8), (7, 9), (7, 8)],
        "b6": [(6, 8), (6, 7), (7, 8), (7, 7)],
        "c6": [(6, 7), (6, 6), (7, 7), (7, 6)],
        "d6": [(6, 6), (6, 5), (7, 6), (7, 5)],
        "e6": [(6, 5), (6, 4), (7, 5), (7, 4)],
        "f6": [(6, 4), (6, 3), (7, 4), (7, 3)],
        "g6": [(6, 3), (6, 2), (7, 3), (7, 2)],
        "h6": [(6, 2), (6, 1), (7, 2), (7, 1)],
        # Rangée 7
        "a7": [(7, 9), (7, 8), (8, 9), (8, 8)],
        "b7": [(7, 8), (7, 7), (8, 8), (8, 7)],
        "c7": [(7, 7), (7, 6), (8, 7), (8, 6)],
        "d7": [(7, 6), (7, 5), (8, 6), (8, 5)],
        "e7": [(7, 5), (7, 4), (8, 5), (8, 4)],
        "f7": [(7, 4), (7, 3), (8, 4), (8, 3)],
        "g7": [(7, 3), (7, 2), (8, 3), (8, 2)],
        "h7": [(7, 2), (7, 1), (8, 2), (8, 1)],
        # Rangée 8
        "a8": [(8, 9), (8, 8), (9, 9), (9, 8)],
        "b8": [(8, 8), (8, 7), (9, 8), (9, 7)],
        "c8": [(8, 7), (8, 6), (9, 7), (9, 6)],
        "d8": [(8, 6), (8, 5), (9, 6), (9, 5)],
        "e8": [(8, 5), (8, 4), (9, 5), (9, 4)],
        "f8": [(8, 4), (8, 3), (9, 4), (9, 3)],
        "g8": [(8, 3), (8, 2), (9, 3), (9, 2)],
        "h8": [(8, 2), (8, 1), (9, 2), (9, 1)],
    }

    # LEDs cassées (ligne B9)
    BROKEN_LEDS = {(9, i) for i in range(1, 10)}

    def __init__(self, verbose: bool = False):
        """
        Initialise l'interface de l'échiquier.

        Args:
            verbose: Affiche des messages de debug si True
        """
        self.verbose = verbose
        self._log("Initialisation de l'interface échiquier...")

        # Initialisation I2C et TCA
        i2c = busio.I2C(board.SCL, board.SDA)
        tca = TCA9548A(i2c)

        # Initialisation de la matrice LED
        self.matrix = HT16K33Matrix(tca)
        self._log("✓ Matrice LED initialisée")

        self.clear()

    def _log(self, message: str):
        """Affiche un message si verbose est activé."""
        if self.verbose:
            print(f"[ChessboardLED] {message}")

    def _filter_broken_leds(
        self, led_positions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Filtre les LEDs cassées."""
        return [led for led in led_positions if led not in self.BROKEN_LEDS]

    def clear(self):
        """Éteint toutes les LEDs."""
        self._log("Effacement de l'échiquier")
        self.matrix.clear()
        self.matrix.update()

    def display_square(self, square: str, brightness: float = 1.0):
        """
        Illumine une case spécifique.

        Args:
            square: Case au format algébrique (ex: "e4")
            brightness: Luminosité (0.0 - 1.0)
        """
        if square not in self.SQUARE_TO_LEDS:
            raise ValueError(f"Invalid square: {square}")

        led_positions = self._filter_broken_leds(self.SQUARE_TO_LEDS[square])

        for row, col in led_positions:
            self.matrix.set_led(row, col, True)

        self.matrix.set_brightness(brightness)
        self.matrix.update()
        self._log(f"Case {square} illuminée")

    def display_squares(self, squares: List[str], brightness: float = 1.0):
        """
        Illumine plusieurs cases.

        Args:
            squares: Liste de cases au format algébrique
            brightness: Luminosité (0.0 - 1.0)
        """
        for square in squares:
            if square not in self.SQUARE_TO_LEDS:
                continue

            led_positions = self._filter_broken_leds(self.SQUARE_TO_LEDS[square])

            for row, col in led_positions:
                self.matrix.set_led(row, col, True)

        self.matrix.set_brightness(brightness)
        self.matrix.update()
        self._log(f"Cases illuminées: {', '.join(squares)}")

    def display_move(self, uci: str, brightness: float = 1.0):
        """
        Affiche un coup au format UCI.

        Args:
            uci: Coup UCI (ex: "e2e4")
            brightness: Luminosité (0.0 - 1.0)
        """
        move = ChessMove(uci)
        self.clear()
        self.display_squares([move.from_square, move.to_square], brightness)
        self._log(f"Coup affiché: {uci} ({move.from_square} → {move.to_square})")

    def blink(self, squares: List[str], duration: float = 5.0, frequency: float = 2.0):
        """
        Fait clignoter des cases.

        Args:
            squares: Liste de cases à faire clignoter
            duration: Durée totale en secondes
            frequency: Fréquence de clignotement en Hz
        """
        period = 1.0 / frequency
        half_period = period / 2.0
        start_time = time.time()

        self._log(
            f"Clignotement de {', '.join(squares)} pendant {duration}s à {frequency}Hz"
        )

        while time.time() - start_time < duration:
            # Allumer
            self.display_squares(squares)
            time.sleep(half_period)

            # Éteindre
            self.clear()
            time.sleep(half_period)

        self.clear()

    def blink_move(self, uci: str, duration: float = 5.0, frequency: float = 2.0):
        """
        Fait clignoter un coup.

        Args:
            uci: Coup UCI
            duration: Durée totale en secondes
            frequency: Fréquence de clignotement en Hz
        """
        move = ChessMove(uci)
        self.blink([move.from_square, move.to_square], duration, frequency)

    def animate_move(self, uci: str, duration: float = 1.5):
        """
        Animation de déplacement d'une pièce.

        Args:
            uci: Coup UCI
            duration: Durée de l'animation en secondes
        """
        move = ChessMove(uci)

        # Phase 1: Montrer la case de départ
        self._log(f"Animation: {move.from_square} → {move.to_square}")
        self.clear()
        self.display_square(move.from_square)
        time.sleep(duration * 0.3)

        # Phase 2: Clignotement rapide (transition)
        for _ in range(3):
            self.clear()
            time.sleep(duration * 0.05)
            self.display_square(move.from_square)
            time.sleep(duration * 0.05)

        # Phase 3: Montrer la case d'arrivée
        self.clear()
        self.display_square(move.to_square)
        time.sleep(duration * 0.3)

        # Phase 4: Afficher les deux cases ensemble
        self.display_squares([move.from_square, move.to_square])
        time.sleep(duration * 0.3)

    def display_suggestions(
        self, moves: List[str], highlight_best: bool = True, duration: float = 5.0
    ):
        """
        Affiche plusieurs suggestions de coups.

        Args:
            moves: Liste de coups UCI triés par ordre de préférence
            highlight_best: Fait clignoter le meilleur coup
            duration: Durée d'affichage
        """
        if not moves:
            return

        if highlight_best and len(moves) > 0:
            # Meilleur coup clignote
            best_move = ChessMove(moves[0])

            # Afficher les autres en fixe
            other_squares = []
            for uci in moves[1:]:
                m = ChessMove(uci)
                other_squares.extend([m.from_square, m.to_square])

            if other_squares:
                self.display_squares(other_squares, brightness=0.3)

            # Faire clignoter le meilleur
            self.blink([best_move.from_square, best_move.to_square], duration, 2.0)
        else:
            # Afficher tous les coups en fixe
            all_squares = []
            for uci in moves:
                m = ChessMove(uci)
                all_squares.extend([m.from_square, m.to_square])
            self.display_squares(all_squares)
            time.sleep(duration)

        self.clear()

    def set_brightness(self, brightness: float):
        """
        Ajuste la luminosité globale.

        Args:
            brightness: Luminosité (0.0 - 1.0)
        """
        self.matrix.set_brightness(brightness)
        self._log(f"Luminosité: {brightness * 100}%")


# Fonctions utilitaires


def square_to_coords(square: str) -> Tuple[int, int]:
    """
    Convertit une case algébrique en coordonnées (row, col).

    Args:
        square: Case au format algébrique (ex: "e4")

    Returns:
        Tuple (row, col) où row et col sont entre 0 et 7
    """
    col = ord(square[0]) - ord("a")
    row = int(square[1]) - 1
    return (row, col)


def coords_to_square(row: int, col: int) -> str:
    """
    Convertit des coordonnées en case algébrique.

    Args:
        row: Rangée (0-7)
        col: Colonne (0-7)

    Returns:
        Case au format algébrique (ex: "e4")
    """
    return chr(ord("a") + col) + str(row + 1)


def parse_uci(uci: str) -> Dict[str, Optional[str]]:
    """
    Parse un coup UCI en ses composants.

    Args:
        uci: Coup UCI (ex: "e2e4", "e7e8q")

    Returns:
        Dict avec 'from', 'to', et 'promotion'
    """
    return {
        "from": uci[0:2],
        "to": uci[2:4],
        "promotion": uci[4] if len(uci) == 5 else None,
    }


# Programme de démonstration
if __name__ == "__main__":
    print("=== Démonstration de l'interface Chess LED ===\n")

    board = ChessboardLED(verbose=True)

    try:
        # Test 1: Afficher un coup simple
        print("\n1. Affichage d'un coup simple: e2-e4")
        board.display_move("e2e4")
        time.sleep(2)

        # Test 2: Animation de coup
        print("\n2. Animation d'un coup: g1-f3")
        board.animate_move("g1f3", duration=2.0)
        time.sleep(1)

        # Test 3: Clignotement
        print("\n3. Clignotement d'un coup: d2-d4")
        board.blink_move("d2d4", duration=3.0, frequency=2.0)

        # Test 4: Suggestions multiples
        print("\n4. Suggestions multiples")
        suggestions = ["e2e4", "d2d4", "g1f3"]
        board.display_suggestions(suggestions, duration=5.0)

        # Test 5: Affichage de cases spécifiques
        print("\n5. Illumination de cases spécifiques")
        board.display_squares(["a1", "h1", "a8", "h8"])
        time.sleep(2)

        print("\n✓ Démonstration terminée!")

    except KeyboardInterrupt:
        print("\n\nInterrompu par l'utilisateur")
    finally:
        board.clear()
        print("Échiquier éteint.")
