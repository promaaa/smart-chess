#!/usr/bin/env python3
"""
IA-Marc V2 - Polyglot Opening Book Reader
==========================================

Lecteur de livres d'ouvertures au format Polyglot (.bin).
Format standard utilisé par de nombreux moteurs d'échecs.

Le format Polyglot:
- Fichier binaire avec entrées de 16 bytes
- Clé Zobrist 64-bit (big-endian)
- Coup encodé sur 16-bit
- Poids sur 16-bit
- Autres données sur 32-bit

Compatible avec Cerebellum_Light.bin et autres books Polyglot.

Optimisé pour Raspberry Pi 5.
"""

import logging
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import chess
import chess.polyglot

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES POLYGLOT
# ============================================================================

# Taille d'une entrée dans le fichier
ENTRY_SIZE = 16

# Tables Zobrist supprimées car on utilise chess.polyglot



# ============================================================================
# FONCTIONS ZOBRIST POLYGLOT
# ============================================================================


def compute_polyglot_key(board: chess.Board) -> int:
    """
    Calcule la clé Zobrist Polyglot pour une position.
    Utilise l'implémentation officielle de python-chess.

    Args:
        board: Position d'échecs

    Returns:
        Clé Zobrist 64-bit
    """
    return chess.polyglot.zobrist_hash(board)


# ============================================================================
# DÉCODAGE DES COUPS POLYGLOT
# ============================================================================


def decode_polyglot_move(encoded_move: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Décode un coup au format Polyglot.

    Format Polyglot (16 bits):
    - 6 bits: case de départ (0-63)
    - 6 bits: case d'arrivée (0-63)
    - 4 bits: type de promotion (0=aucune, 1=cavalier, 2=fou, 3=tour, 4=dame)

    Args:
        encoded_move: Coup encodé (16-bit)
        board: Position actuelle

    Returns:
        chess.Move ou None si invalide
    """
    # Extraire les composants
    from_square = (encoded_move >> 6) & 0x3F
    to_square = encoded_move & 0x3F
    promotion_bits = (encoded_move >> 12) & 0x0F

    # Mapper la promotion
    promotion_map = {
        0: None,
        1: chess.KNIGHT,
        2: chess.BISHOP,
        3: chess.ROOK,
        4: chess.QUEEN,
    }

    promotion = promotion_map.get(promotion_bits)

    try:
        move = chess.Move(from_square, to_square, promotion=promotion)

        # Vérifier que le coup est légal
        if move in board.legal_moves:
            return move
        else:
            return None

    except ValueError:
        return None


# ============================================================================
# CLASSE POLYGLOT BOOK
# ============================================================================


class PolyglotBook:
    """
    Lecteur de livre d'ouvertures au format Polyglot.

    Permet de lire des fichiers .bin comme Cerebellum_Light.bin.
    """

    def __init__(self, filepath: Optional[str] = None):
        """
        Initialise le lecteur.

        Args:
            filepath: Chemin vers le fichier .bin (optionnel)
        """
        self.filepath = filepath
        self.entries = []
        self.loaded = False

        # Statistiques
        self.probes = 0
        self.hits = 0

    def load(self, filepath: Optional[str] = None) -> bool:
        """
        Charge le livre d'ouvertures depuis un fichier .bin.

        Args:
            filepath: Chemin du fichier (optionnel)

        Returns:
            True si chargé avec succès, False sinon
        """
        if filepath:
            self.filepath = filepath

        if not self.filepath:
            logger.error("Aucun chemin de fichier spécifié")
            return False

        try:
            path = Path(self.filepath)

            if not path.exists():
                logger.warning(f"Fichier Polyglot non trouvé: {self.filepath}")
                return False

            # Lire toutes les entrées
            file_size = path.stat().st_size
            num_entries = file_size // ENTRY_SIZE

            logger.info(f"Chargement du book Polyglot: {num_entries} entrées...")

            with open(path, "rb") as f:
                self.entries = []

                for _ in range(num_entries):
                    data = f.read(ENTRY_SIZE)
                    if len(data) < ENTRY_SIZE:
                        break

                    # Décoder l'entrée (big-endian)
                    key = struct.unpack(">Q", data[0:8])[0]
                    move = struct.unpack(">H", data[8:10])[0]
                    weight = struct.unpack(">H", data[10:12])[0]
                    learn = struct.unpack(">I", data[12:16])[0]

                    self.entries.append(
                        {
                            "key": key,
                            "move": move,
                            "weight": weight,
                            "learn": learn,
                        }
                    )

            # Trier par clé pour recherche binaire
            self.entries.sort(key=lambda x: x["key"])

            self.loaded = True
            logger.info(f"Book Polyglot chargé: {len(self.entries)} entrées")

            return True

        except Exception as e:
            logger.error(f"Erreur lors du chargement du book Polyglot: {e}")
            return False

    def probe(
        self, board: chess.Board, max_moves: int = 10
    ) -> List[Tuple[chess.Move, int]]:
        """
        Cherche les coups disponibles pour une position.

        Args:
            board: Position actuelle
            max_moves: Nombre maximum de coups à retourner

        Returns:
            Liste de tuples (coup, poids) triée par poids décroissant
        """
        self.probes += 1

        if not self.loaded:
            return []

        # Calculer la clé Polyglot
        key = compute_polyglot_key(board)

        # Recherche binaire de la première entrée
        left, right = 0, len(self.entries)
        first_index = -1

        while left < right:
            mid = (left + right) // 2
            if self.entries[mid]["key"] < key:
                left = mid + 1
            elif self.entries[mid]["key"] > key:
                right = mid
            else:
                first_index = mid
                right = mid

        if first_index == -1:
            return []

        # Collecter toutes les entrées avec cette clé
        moves = []
        index = first_index

        while index < len(self.entries) and self.entries[index]["key"] == key:
            entry = self.entries[index]

            # Décoder le coup
            move = decode_polyglot_move(entry["move"], board)

            if move and move in board.legal_moves:
                moves.append((move, entry["weight"]))
                self.hits += 1

            index += 1

            if len(moves) >= max_moves:
                break

        # Trier par poids décroissant
        moves.sort(key=lambda x: x[1], reverse=True)

        return moves

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Retourne le meilleur coup (poids le plus élevé).

        Args:
            board: Position actuelle

        Returns:
            Meilleur coup ou None
        """
        moves = self.probe(board, max_moves=1)

        if moves:
            return moves[0][0]

        return None

    def get_weighted_move(
        self, board: chess.Board, variety: bool = True
    ) -> Optional[chess.Move]:
        """
        Retourne un coup avec sélection pondérée aléatoire.

        Args:
            board: Position actuelle
            variety: Si True, utilise sélection pondérée aléatoire
                    Si False, retourne toujours le meilleur

        Returns:
            Coup sélectionné ou None
        """
        moves = self.probe(board, max_moves=20)

        if not moves:
            return None

        if not variety:
            return moves[0][0]

        # Sélection pondérée
        import random

        total_weight = sum(weight for _, weight in moves)

        if total_weight == 0:
            return moves[0][0]

        r = random.random() * total_weight
        cumulative = 0

        for move, weight in moves:
            cumulative += weight
            if r <= cumulative:
                return move

        return moves[0][0]

    def get_stats(self) -> dict:
        """
        Retourne les statistiques du livre.

        Returns:
            Dictionnaire de statistiques
        """
        hit_rate = (self.hits / self.probes * 100) if self.probes > 0 else 0

        return {
            "loaded": self.loaded,
            "entries": len(self.entries),
            "probes": self.probes,
            "hits": self.hits,
            "hit_rate": hit_rate,
        }

    def clear_stats(self):
        """Réinitialise les statistiques."""
        self.probes = 0
        self.hits = 0


# ============================================================================
# TESTS
# ============================================================================


if __name__ == "__main__":
    print("=== Test du Polyglot Book Reader ===\n")

    # Configuration logging
    logging.basicConfig(level=logging.DEBUG)

    # Test avec Cerebellum_Light.bin
    book_path = "../book/Cerebellum_Light.bin"

    print(f"Test 1: Chargement de {book_path}")
    book = PolyglotBook(book_path)

    if not book.load():
        print("❌ Fichier non trouvé")
        print("Note: Placez Cerebellum_Light.bin dans le dossier ../book/")
        exit(1)

    print(f"✓ Chargé: {len(book.entries):,} entrées\n")

    # Test 2: Position initiale
    print("Test 2: Position initiale")
    board = chess.Board()

    moves = book.probe(board, max_moves=5)
    print(f"  Coups trouvés: {len(moves)}")

    for move, weight in moves[:5]:
        print(f"    {move.uci()}: poids {weight}")

    print()

    # Test 3: Meilleur coup
    print("Test 3: Meilleur coup")
    best_move = book.get_best_move(board)
    print(f"  Meilleur coup: {best_move}\n")

    # Test 4: Variété (10 essais)
    print("Test 4: Variété (10 essais)")
    move_counts = {}

    for _ in range(10):
        move = book.get_weighted_move(board, variety=True)
        move_str = move.uci() if move else "None"
        move_counts[move_str] = move_counts.get(move_str, 0) + 1

    for move, count in sorted(move_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {move}: {count}/10")

    print()

    # Test 5: Après 1.e4
    print("Test 5: Après 1.e4")
    board.push(chess.Move.from_uci("e2e4"))

    moves = book.probe(board, max_moves=5)
    print(f"  Coups trouvés: {len(moves)}")

    for move, weight in moves[:5]:
        print(f"    {move.uci()}: poids {weight}")

    print()

    # Test 6: Statistiques
    print("Test 6: Statistiques")
    stats = book.get_stats()
    print(f"  Entrées: {stats['entries']:,}")
    print(f"  Probes: {stats['probes']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Hit rate: {stats['hit_rate']:.1f}%")

    print("\n✅ Tests du Polyglot Book terminés!")
