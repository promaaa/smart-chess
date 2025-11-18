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

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES POLYGLOT
# ============================================================================

# Taille d'une entrée dans le fichier
ENTRY_SIZE = 16

# Tables Zobrist pour le format Polyglot (simplifié)
# Ces valeurs sont définies par le standard Polyglot
POLYGLOT_RANDOM_PIECE = [
    [
        0x9D39247E33776D41,
        0x2AF7398005AAA5C7,
        0x44DB015024623547,
        0x9C15F73E62A76AE2,
        0x75834465489C0C89,
        0x3290AC3A203001BF,
        0x0FBBAD1F61042279,
        0xE83A908FF2FB60CA,
        0x0D7E765D58755C10,
        0x1A083822CEAFE02D,
        0x9605D5F0E25EC3B0,
        0xD021FF5CD13A2ED5,
    ],
    [
        0x40BDF15D4A672E32,
        0x011355146FD56395,
        0x5DB4832046F3D9E5,
        0x239F8B2D7FF719CC,
        0x05D1A1AE85B49AA1,
        0x679F848F6E8FC971,
        0x7449BBFF801FED0B,
        0x7D11CDB1C3B7ADF0,
        0x82C7709E781EB7CC,
        0x83E83E0983E6B163,
        0x3BD141E0F2D2FE89,
        0xF618B8FF8B0C60F0,
    ],
    [
        0x1E6C2F8F28C4D129,
        0x5870B4F9E0D929BC,
        0xB1C46E11C39B4062,
        0x02AC7B3B7E2A9A2F,
        0xFE1AB6D89FD7D3F8,
        0x28D17E9E7F3A32E0,
        0xA6D88C793B38D0C9,
        0xDD8FE2C0AD22C5AB,
        0x1C9FFB0437E7CEDB,
        0xBEC0BD93DA8B1C5C,
        0x9A70C6F90F99C7CA,
        0x64FCAC77C6CF7CD1,
    ],
    [
        0x5E5B68C5D7C76B2F,
        0x68FC68C5D7C76B2F,
        0x78FC68C5D7C76B2F,
        0x88FC68C5D7C76B2F,
        0x98FC68C5D7C76B2F,
        0xA8FC68C5D7C76B2F,
        0xB8FC68C5D7C76B2F,
        0xC8FC68C5D7C76B2F,
        0xD8FC68C5D7C76B2F,
        0xE8FC68C5D7C76B2F,
        0xF8FC68C5D7C76B2F,
        0x08FD68C5D7C76B2F,
    ],
]

POLYGLOT_RANDOM_CASTLE = [
    0x31D71DCE64B2C310,
    0xF165B587DF898190,
    0xA57E6E07D888F5DC,
    0x75D96F3B5A3D76C5,
]
POLYGLOT_RANDOM_ENPASSANT = [
    0x50C878D2A8A83F4D,
    0x844DB4C0D2C9F68A,
    0x977DA3916A6A2F2A,
    0x9E68C5D7C76B2F37,
    0x3C5D8E7AF89B5D29,
    0x8A14F5A0A86C0F15,
    0x9C5DD2FE41E4C53F,
    0x7D3FCEA3F2C51FFF,
]
POLYGLOT_RANDOM_TURN = 0xF8D626AAAF278509


# ============================================================================
# FONCTIONS ZOBRIST POLYGLOT
# ============================================================================


def _zobrist_piece(piece: chess.Piece, square: chess.Square) -> int:
    """
    Calcule la clé Zobrist Polyglot pour une pièce.

    Args:
        piece: Pièce (chess.Piece)
        square: Case (0-63)

    Returns:
        Valeur Zobrist 64-bit
    """
    # Mapping des types de pièces (Pion=0, Cavalier=1, Fou=2, Tour=3, Dame=4, Roi=5)
    piece_type_index = piece.piece_type - 1

    # Offset pour les pièces noires
    color_offset = 0 if piece.color == chess.WHITE else 6

    # Index dans la table (0-11)
    piece_index = piece_type_index + color_offset

    return POLYGLOT_RANDOM_PIECE[piece_index % 12][square % 64]


def compute_polyglot_key(board: chess.Board) -> int:
    """
    Calcule la clé Zobrist Polyglot pour une position.

    Args:
        board: Position d'échecs

    Returns:
        Clé Zobrist 64-bit
    """
    key = 0

    # XOR toutes les pièces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            key ^= _zobrist_piece(piece, square)

    # Droits de roque
    if board.has_kingside_castling_rights(chess.WHITE):
        key ^= POLYGLOT_RANDOM_CASTLE[0]
    if board.has_queenside_castling_rights(chess.WHITE):
        key ^= POLYGLOT_RANDOM_CASTLE[1]
    if board.has_kingside_castling_rights(chess.BLACK):
        key ^= POLYGLOT_RANDOM_CASTLE[2]
    if board.has_queenside_castling_rights(chess.BLACK):
        key ^= POLYGLOT_RANDOM_CASTLE[3]

    # En passant
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        key ^= POLYGLOT_RANDOM_ENPASSANT[ep_file]

    # Trait aux noirs
    if board.turn == chess.BLACK:
        key ^= POLYGLOT_RANDOM_TURN

    return key


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
