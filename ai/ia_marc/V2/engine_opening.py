#!/usr/bin/env python3
"""
IA-Marc V2 - Opening Book Management
=====================================

Gestionnaire de bibliothèque d'ouvertures pour varier les débuts de partie
et renforcer le jeu aux niveaux supérieurs.

Fonctionnalités:
- Chargement depuis JSON ou Polyglot (.bin)
- Sélection pondérée par poids
- Filtrage par niveau ELO
- Variété dans les choix
- Support de Cerebellum_Light.bin et autres books Polyglot
- Extension facile

Optimisé pour Raspberry Pi 5.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess

logger = logging.getLogger(__name__)

# Import du module Polyglot
try:
    from engine_polyglot import PolyglotBook

    POLYGLOT_AVAILABLE = True
except ImportError:
    POLYGLOT_AVAILABLE = False
    logger.warning("Module Polyglot non disponible")


# ============================================================================
# CLASSE OPENING BOOK
# ============================================================================


class OpeningBook:
    """
    Bibliothèque d'ouvertures pour le moteur d'échecs.

    Permet de jouer des coups d'ouverture prédéfinis pour:
    - Éviter les coups faibles en début de partie
    - Varier les parties
    - Renforcer le jeu aux niveaux élevés

    Supporte les formats JSON et Polyglot (.bin).
    """

    def __init__(self, book_path: Optional[str] = None, book_type: str = "auto"):
        """
        Initialise le livre d'ouvertures.

        Args:
            book_path: Chemin vers le fichier JSON ou .bin (optionnel)
            book_type: Type de livre ("auto", "json", "polyglot")
        """
        self.book_path = book_path or "ai/ia_marc/V2/data/openings.json"
        self.book_type = book_type
        self.openings = {}
        self.metadata = {}
        self.loaded = False
        self.polyglot_book = None

        # Statistiques
        self.probes = 0
        self.hits = 0

    def load(self, filepath: Optional[str] = None) -> bool:
        """
        Charge le livre d'ouvertures depuis un fichier (JSON ou Polyglot).

        Args:
            filepath: Chemin vers le fichier (optionnel)

        Returns:
            True si chargé avec succès, False sinon
        """
        if filepath:
            self.book_path = filepath

        try:
            path = Path(self.book_path)

            if not path.exists():
                logger.warning(f"Fichier opening book non trouvé: {self.book_path}")
                return False

            # Déterminer le type de livre
            detected_type = self._detect_book_type(path)

            if self.book_type == "auto":
                self.book_type = detected_type

            # Charger selon le type
            if self.book_type == "polyglot":
                return self._load_polyglot(path)
            else:
                return self._load_json(path)

        except Exception as e:
            logger.error(f"Erreur lors du chargement du opening book: {e}")
            return False

    def _detect_book_type(self, path: Path) -> str:
        """
        Détecte automatiquement le type de livre.

        Args:
            path: Chemin du fichier

        Returns:
            "json" ou "polyglot"
        """
        if path.suffix.lower() == ".bin":
            return "polyglot"
        elif path.suffix.lower() == ".json":
            return "json"
        else:
            # Essayer de lire comme JSON
            try:
                with open(path, "r", encoding="utf-8") as f:
                    json.load(f)
                return "json"
            except:
                return "polyglot"

    def _load_json(self, path: Path) -> bool:
        """
        Charge un livre JSON.

        Args:
            path: Chemin du fichier

        Returns:
            True si succès
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.metadata = data.get("metadata", {})
            self.openings = data.get("openings", {})

            self.loaded = True

            total = self.metadata.get("total_positions", len(self.openings))
            logger.info(f"Opening book JSON chargé: {total} positions")

            return True

        except json.JSONDecodeError as e:
            logger.error(f"Erreur JSON dans {self.book_path}: {e}")
            return False

    def _load_polyglot(self, path: Path) -> bool:
        """
        Charge un livre Polyglot (.bin).

        Args:
            path: Chemin du fichier

        Returns:
            True si succès
        """
        if not POLYGLOT_AVAILABLE:
            logger.error("Support Polyglot non disponible")
            return False

        try:
            self.polyglot_book = PolyglotBook(str(path))
            success = self.polyglot_book.load()

            if success:
                self.loaded = True
                logger.info(
                    f"Opening book Polyglot chargé: {len(self.polyglot_book.entries):,} entrées"
                )

            return success

        except Exception as e:
            logger.error(f"Erreur lors du chargement Polyglot: {e}")
            return False

    def probe(
        self, board: chess.Board, elo_level: int = 2000, variety: bool = True
    ) -> Optional[chess.Move]:
        """
        Cherche un coup dans le livre d'ouvertures.

        Args:
            board: Position actuelle
            elo_level: Niveau ELO du joueur (pour filtrage JSON)
            variety: Utiliser sélection aléatoire pondérée (True)
                    ou toujours le meilleur (False)

        Returns:
            Coup suggéré ou None si pas dans le livre
        """
        self.probes += 1

        if not self.loaded:
            return None

        # Déléguer au livre Polyglot si utilisé
        if self.book_type == "polyglot" and self.polyglot_book:
            return self._probe_polyglot(board, variety)

        # Sinon, utiliser le livre JSON
        return self._probe_json(board, elo_level, variety)

    def _probe_json(
        self, board: chess.Board, elo_level: int, variety: bool
    ) -> Optional[chess.Move]:
        """
        Cherche un coup dans le livre JSON.

        Args:
            board: Position actuelle
            elo_level: Niveau ELO
            variety: Variété dans le choix

        Returns:
            Coup ou None
        """
        # Obtenir le FEN de la position (sans compteur de coups)
        fen = board.fen()
        fen_key = self._normalize_fen(fen)

        # Chercher dans le livre
        entry = self.openings.get(fen_key)

        if not entry:
            # Essayer avec la clé simplifiée (position de base)
            fen_key = self._get_position_key(board)
            entry = self.openings.get(fen_key)

        if not entry:
            return None

        # Filtrer les coups selon l'ELO
        moves = entry.get("moves", [])
        valid_moves = [m for m in moves if m.get("min_elo", 0) <= elo_level]

        if not valid_moves:
            return None

        # Sélectionner un coup
        if variety:
            move_str = self._weighted_choice(valid_moves)
        else:
            # Prendre le premier (poids le plus élevé normalement)
            move_str = valid_moves[0]["move"]

        # Convertir en chess.Move
        try:
            move = chess.Move.from_uci(move_str)

            # Vérifier que le coup est légal
            if move in board.legal_moves:
                self.hits += 1
                logger.debug(f"Opening book hit (JSON): {move}")
                return move
            else:
                logger.warning(f"Coup du book illégal: {move_str}")
                return None

        except ValueError:
            logger.warning(f"Format UCI invalide dans le book: {move_str}")
            return None

    def _probe_polyglot(
        self, board: chess.Board, variety: bool
    ) -> Optional[chess.Move]:
        """
        Cherche un coup dans le livre Polyglot.

        Args:
            board: Position actuelle
            variety: Variété dans le choix

        Returns:
            Coup ou None
        """
        if not self.polyglot_book:
            return None

        move = self.polyglot_book.get_weighted_move(board, variety=variety)

        if move:
            self.hits += 1
            logger.debug(f"Opening book hit (Polyglot): {move}")

        return move

    def _normalize_fen(self, fen: str) -> str:
        """
        Normalise un FEN pour la recherche (enlève compteurs).

        Args:
            fen: FEN complet

        Returns:
            FEN normalisé
        """
        # Garder seulement les 4 premiers champs (position, trait, roque, en passant)
        parts = fen.split()
        return " ".join(parts[:4])

    def _get_position_key(self, board: chess.Board) -> str:
        """
        Génère une clé simplifiée pour la position.

        Utilisé pour chercher les positions "standard" dans le livre JSON.
        """
        # Position initiale
        if board.move_stack == []:
            return "initial"

        # Construire la clé depuis les coups joués
        if len(board.move_stack) == 1:
            first_move = board.move_stack[0].uci()
            return first_move[:2] + first_move[2:4]  # e2e4 -> e4

        # Pour positions plus complexes, essayer des patterns connus
        move_sequence = "_".join([m.uci() for m in board.move_stack[:4]])

        # Mapping vers clés du livre
        key_mapping = {
            "e2e4": "e4",
            "d2d4": "d4",
            "g1f3": "Nf3",
            "c2c4": "c4",
            "e2e4_e7e5": "e4_e5",
            "e2e4_c7c5": "e4_c5",
            "d2d4_d7d5": "d4_d5",
            "d2d4_g8f6": "d4_Nf6",
        }

        return key_mapping.get(move_sequence, None)

    def _weighted_choice(self, moves: List[Dict]) -> str:
        """
        Sélection aléatoire pondérée d'un coup.

        Args:
            moves: Liste de coups avec poids

        Returns:
            UCI string du coup choisi
        """
        # Extraire les poids
        weights = [m.get("weight", 1) for m in moves]
        total = sum(weights)

        if total == 0:
            return moves[0]["move"]

        # Sélection aléatoire
        r = random.random() * total
        cumulative = 0

        for move, weight in zip(moves, weights):
            cumulative += weight
            if r <= cumulative:
                return move["move"]

        # Fallback
        return moves[0]["move"]

    def add_position(self, fen: str, moves: List[Dict], key: Optional[str] = None):
        """
        Ajoute une position au livre (en mémoire, JSON seulement).

        Args:
            fen: FEN de la position
            moves: Liste de coups possibles avec poids et min_elo
            key: Clé personnalisée (optionnel)
        """
        fen_key = key or self._normalize_fen(fen)

        self.openings[fen_key] = {"fen": fen, "moves": moves}

        logger.debug(f"Position ajoutée: {fen_key}")

    def save(self, filepath: Optional[str] = None):
        """
        Sauvegarde le livre d'ouvertures dans un fichier JSON.

        Args:
            filepath: Chemin de destination (optionnel)
        """
        if self.book_type == "polyglot":
            logger.warning("Sauvegarde non supportée pour les livres Polyglot")
            return

        if filepath:
            self.book_path = filepath

        try:
            data = {
                "metadata": self.metadata,
                "openings": self.openings,
                "statistics": {"total_positions": len(self.openings)},
            }

            path = Path(self.book_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Opening book sauvegardé: {self.book_path}")

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")

    def get_stats(self) -> Dict:
        """
        Retourne les statistiques du livre.

        Returns:
            Dictionnaire de statistiques
        """
        hit_rate = (self.hits / self.probes * 100) if self.probes > 0 else 0

        stats = {
            "loaded": self.loaded,
            "book_type": self.book_type,
            "probes": self.probes,
            "hits": self.hits,
            "hit_rate": hit_rate,
        }

        # Ajouter les stats spécifiques au type
        if self.book_type == "polyglot" and self.polyglot_book:
            stats["positions"] = len(self.polyglot_book.entries)
        else:
            stats["positions"] = len(self.openings)

        return stats

    def clear_stats(self):
        """Réinitialise les statistiques."""
        self.probes = 0
        self.hits = 0

    def __len__(self):
        """Nombre de positions dans le livre."""
        if self.book_type == "polyglot" and self.polyglot_book:
            return len(self.polyglot_book.entries)
        return len(self.openings)

    def __contains__(self, board: chess.Board):
        """Vérifie si une position est dans le livre."""
        if self.book_type == "polyglot" and self.polyglot_book:
            moves = self.polyglot_book.probe(board, max_moves=1)
            return len(moves) > 0
        else:
            fen_key = self._normalize_fen(board.fen())
            return fen_key in self.openings


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================


def create_basic_book(filepath: str = "ai/ia_marc/V2/data/openings.json"):
    """
    Crée un livre d'ouvertures basique si aucun n'existe.

    Args:
        filepath: Chemin de destination
    """
    book = OpeningBook()

    # Position initiale
    book.add_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        [
            {"move": "e2e4", "name": "King's Pawn", "weight": 40, "min_elo": 400},
            {"move": "d2d4", "name": "Queen's Pawn", "weight": 35, "min_elo": 600},
            {"move": "g1f3", "name": "Réti", "weight": 15, "min_elo": 1200},
            {"move": "c2c4", "name": "English", "weight": 10, "min_elo": 1400},
        ],
        key="initial",
    )

    book.metadata = {
        "name": "IA-Marc Basic Opening Book",
        "version": "1.0",
        "total_positions": len(book),
    }

    book.save(filepath)
    logger.info(f"Livre d'ouvertures basique créé: {filepath}")


# ============================================================================
# TESTS
# ============================================================================


if __name__ == "__main__":
    print("=== Test du Opening Book ===\n")

    # Configuration logging
    logging.basicConfig(level=logging.DEBUG)

    # Test 1: Livre JSON
    print("Test 1: Chargement du livre JSON")
    book_json = OpeningBook("data/openings.json")

    if not book_json.load():
        print("Livre non trouvé, création d'un livre basique...")
        create_basic_book("data/openings.json")
        book_json.load()

    print(f"✓ Chargé: {len(book_json)} positions\n")

    # Test 2: Livre Polyglot
    print("Test 2: Chargement du livre Polyglot")
    book_poly = OpeningBook("ai/ia_marc/book/Cerebellum_Light.bin", book_type="polyglot")

    if book_poly.load():
        print(f"✓ Chargé: {len(book_poly):,} positions\n")

        # Test 3: Position initiale avec Polyglot
        print("Test 3: Position initiale (Polyglot)")
        board = chess.Board()

        move = book_poly.probe(board, variety=True)
        print(f"  Coup suggéré: {move}")

        # Test 4: Variété
        print("\nTest 4: Variété (10 essais)")
        moves_count = {}

        for _ in range(10):
            move = book_poly.probe(board, variety=True)
            move_str = str(move) if move else "None"
            moves_count[move_str] = moves_count.get(move_str, 0) + 1

        for move, count in moves_count.items():
            print(f"  {move}: {count}/10")

        # Test 5: Statistiques
        print("\nTest 5: Statistiques")
        stats = book_poly.get_stats()
        print(f"  Type: {stats['book_type']}")
        print(f"  Positions: {stats['positions']:,}")
        print(f"  Probes: {stats['probes']}")
        print(f"  Hits: {stats['hits']}")
        print(f"  Hit rate: {stats['hit_rate']:.1f}%")
    else:
        print("❌ Fichier Cerebellum_Light.bin non trouvé")
        print("   Placez-le dans ../book/\n")

    print("\n✅ Tests du Opening Book terminés!")
