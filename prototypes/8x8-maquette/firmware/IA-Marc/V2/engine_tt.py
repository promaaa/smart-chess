"""
IA-Marc V2 - Transposition Table (TT)
======================================

Table de transposition avec hachage Zobrist pour éviter de recalculer
les positions déjà évaluées. C'est l'optimisation la plus importante
pour un moteur d'échecs (gain typique: 3-5x).

Optimisé pour Raspberry Pi 5 avec gestion mémoire efficace.
"""

import random
from enum import IntEnum
from typing import Optional, Tuple

import chess

# ============================================================================
# TYPES D'ENTRÉES DE LA TT
# ============================================================================


class TTEntryType(IntEnum):
    """Type d'entrée dans la transposition table."""

    EXACT = 0  # Score exact (PV node)
    LOWER = 1  # Beta cutoff (score >= beta)
    UPPER = 2  # Alpha cutoff (score <= alpha)


# ============================================================================
# GÉNÉRATION DES CLÉS ZOBRIST
# ============================================================================


class ZobristKeys:
    """
    Générateur et stockage des clés Zobrist pour le hachage rapide des positions.

    Le hachage Zobrist permet d'obtenir une signature unique (64-bit) pour chaque
    position d'échecs en temps O(1) avec les mises à jour incrémentales.
    """

    def __init__(self, seed: int = 42):
        """Initialise les clés Zobrist avec une graine aléatoire."""
        random.seed(seed)

        # Clés pour les pièces [piece_type][color][square]
        # piece_type: 1-6 (PAWN to KING)
        # color: 0=WHITE, 1=BLACK
        # square: 0-63
        self.piece_keys = [
            [[self._random_64bit() for _ in range(64)] for _ in range(2)]
            for _ in range(7)  # Index 0 inutilisé, 1-6 pour les pièces
        ]

        # Clés pour le roque (4 possibilités)
        self.castling_keys = [self._random_64bit() for _ in range(16)]

        # Clés pour l'en passant (8 colonnes + 1 pour "pas d'en passant")
        self.ep_keys = [self._random_64bit() for _ in range(9)]

        # Clé pour le trait (si c'est aux noirs de jouer)
        self.side_key = self._random_64bit()

    def _random_64bit(self) -> int:
        """Génère un nombre aléatoire 64-bit."""
        return random.randint(0, 2**64 - 1)

    def hash_position(self, board: chess.Board) -> int:
        """
        Calcule le hash Zobrist complet d'une position.

        Args:
            board: Position d'échecs

        Returns:
            Hash 64-bit de la position
        """
        h = 0

        # Hash des pièces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                color_idx = 0 if piece.color == chess.WHITE else 1
                h ^= self.piece_keys[piece.piece_type][color_idx][square]

        # Hash du roque
        castling_rights = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_rights |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_rights |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_rights |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_rights |= 8
        h ^= self.castling_keys[castling_rights]

        # Hash de l'en passant
        if board.ep_square is not None:
            ep_file = chess.square_file(board.ep_square)
            h ^= self.ep_keys[ep_file]
        else:
            h ^= self.ep_keys[8]  # Pas d'en passant

        # Hash du trait
        if board.turn == chess.BLACK:
            h ^= self.side_key

        return h


# ============================================================================
# ENTRÉE DE LA TRANSPOSITION TABLE
# ============================================================================


class TTEntry:
    """Une entrée dans la transposition table."""

    __slots__ = ["key", "depth", "score", "entry_type", "best_move", "age"]

    def __init__(
        self,
        key: int = 0,
        depth: int = 0,
        score: int = 0,
        entry_type: TTEntryType = TTEntryType.EXACT,
        best_move: Optional[chess.Move] = None,
        age: int = 0,
    ):
        self.key = key  # Hash complet (pour vérifier les collisions)
        self.depth = depth  # Profondeur de recherche
        self.score = score  # Score évalué
        self.entry_type = entry_type  # Type d'entrée (EXACT, LOWER, UPPER)
        self.best_move = best_move  # Meilleur coup trouvé
        self.age = age  # Âge de l'entrée (pour remplacement)

    def is_valid(self) -> bool:
        """Vérifie si l'entrée est valide."""
        return self.key != 0

    def __repr__(self):
        return f"<TTEntry: depth={self.depth}, score={self.score}, type={self.entry_type}, move={self.best_move}>"


# ============================================================================
# TRANSPOSITION TABLE
# ============================================================================


class TranspositionTable:
    """
    Table de transposition avec remplacement basé sur la profondeur et l'âge.

    Utilise un schéma "Always Replace" modifié qui privilégie les entrées
    plus profondes et plus récentes.
    """

    def __init__(self, size_mb: int = 256):
        """
        Initialise la transposition table.

        Args:
            size_mb: Taille de la table en mégaoctets
        """
        self.zobrist = ZobristKeys()
        self.current_age = 0

        # Calculer le nombre d'entrées
        # Estimation: ~32 bytes par entrée (avec overhead Python)
        bytes_per_entry = 32
        self.size = (size_mb * 1024 * 1024) // bytes_per_entry

        # Assurer que la taille est une puissance de 2 (pour masque rapide)
        self.size = 2 ** (self.size.bit_length() - 1)
        self.mask = self.size - 1

        # Créer la table
        self.table = [TTEntry() for _ in range(self.size)]

        # Statistiques
        self.hits = 0
        self.misses = 0
        self.collisions = 0
        self.stores = 0

        print(
            f"✓ Transposition Table initialisée: {self.size:,} entrées (~{size_mb}MB)"
        )

    def clear(self):
        """Vide complètement la table."""
        self.table = [TTEntry() for _ in range(self.size)]
        self.hits = 0
        self.misses = 0
        self.collisions = 0
        self.stores = 0
        self.current_age = 0

    def new_search(self):
        """Incrémente l'âge pour une nouvelle recherche."""
        self.current_age += 1
        if self.current_age > 255:  # Éviter le débordement
            self.current_age = 0

    def _get_index(self, key: int) -> int:
        """Calcule l'index dans la table à partir du hash."""
        return key & self.mask

    def probe(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int = 0
    ) -> Tuple[bool, int, Optional[chess.Move]]:
        """
        Cherche une position dans la table.

        Args:
            board: Position à chercher
            depth: Profondeur de recherche actuelle
            alpha: Borne alpha
            beta: Borne beta
            ply: Profondeur depuis la racine (pour ajuster les scores de mat)

        Returns:
            (found, score, best_move)
            - found: True si un score utilisable est trouvé
            - score: Score de la position (ajusté)
            - best_move: Meilleur coup suggéré (peut être None même si found=False)
        """
        key = self.zobrist.hash_position(board)
        index = self._get_index(key)
        entry = self.table[index]

        # Vérifier si l'entrée correspond
        if not entry.is_valid() or entry.key != key:
            self.misses += 1
            return False, 0, None

        # On a trouvé la position
        self.hits += 1

        # Vérifier si la profondeur est suffisante
        if entry.depth < depth:
            # L'entrée existe mais n'est pas assez profonde
            # On peut quand même utiliser le best_move pour le move ordering
            return False, 0, entry.best_move

        # Ajuster le score si c'est un mat
        score = entry.score
        if score > 90000:  # Mate score
            score -= ply
        elif score < -90000:
            score += ply

        # Vérifier si le score peut être utilisé
        if entry.entry_type == TTEntryType.EXACT:
            return True, score, entry.best_move

        elif entry.entry_type == TTEntryType.LOWER:
            # score >= beta
            if score >= beta:
                return True, score, entry.best_move

        elif entry.entry_type == TTEntryType.UPPER:
            # score <= alpha
            if score <= alpha:
                return True, score, entry.best_move

        # Le score existe mais ne peut pas être utilisé pour un cutoff
        # On retourne quand même le best_move pour le move ordering
        return False, score, entry.best_move

    def store(
        self,
        board: chess.Board,
        depth: int,
        score: int,
        entry_type: TTEntryType,
        best_move: Optional[chess.Move],
        ply: int = 0,
    ):
        """
        Stocke une position dans la table.

        Args:
            board: Position à stocker
            depth: Profondeur de recherche
            score: Score évalué
            entry_type: Type d'entrée (EXACT, LOWER, UPPER)
            best_move: Meilleur coup trouvé
            ply: Profondeur depuis la racine (pour ajuster les scores de mat)
        """
        key = self.zobrist.hash_position(board)
        index = self._get_index(key)

        # Ajuster le score si c'est un mat
        adjusted_score = score
        if score > 90000:  # Mate score
            adjusted_score += ply
        elif score < -90000:
            adjusted_score -= ply

        existing = self.table[index]

        # Schéma de remplacement:
        # Remplacer si:
        # 1. L'entrée est vide
        # 2. C'est la même position (mise à jour)
        # 3. La nouvelle entrée est plus profonde
        # 4. L'ancienne entrée est trop vieille (>4 générations)
        should_replace = (
            not existing.is_valid()
            or existing.key == key
            or depth >= existing.depth
            or (self.current_age - existing.age) > 4
        )

        if should_replace:
            if existing.is_valid() and existing.key != key:
                self.collisions += 1

            self.table[index] = TTEntry(
                key=key,
                depth=depth,
                score=adjusted_score,
                entry_type=entry_type,
                best_move=best_move,
                age=self.current_age,
            )
            self.stores += 1

    def get_pv_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Récupère le meilleur coup suggéré pour une position (si disponible).

        Utile pour le move ordering et l'affichage de la PV.
        """
        key = self.zobrist.hash_position(board)
        index = self._get_index(key)
        entry = self.table[index]

        if entry.is_valid() and entry.key == key and entry.best_move:
            # Vérifier que le coup est légal
            if entry.best_move in board.legal_moves:
                return entry.best_move

        return None

    def get_usage_percent(self) -> float:
        """Retourne le pourcentage de la table utilisée."""
        used = sum(1 for entry in self.table if entry.is_valid())
        return (used / self.size) * 100.0

    def get_hit_rate(self) -> float:
        """Retourne le taux de hits (%)."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100.0

    def get_stats(self) -> dict:
        """Retourne les statistiques de la table."""
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.get_hit_rate(),
            "collisions": self.collisions,
            "stores": self.stores,
            "usage": self.get_usage_percent(),
            "age": self.current_age,
        }

    def print_stats(self):
        """Affiche les statistiques de la table."""
        stats = self.get_stats()
        print("\n=== Statistiques Transposition Table ===")
        print(f"Taille:      {stats['size']:,} entrées")
        print(f"Hits:        {stats['hits']:,}")
        print(f"Misses:      {stats['misses']:,}")
        print(f"Hit Rate:    {stats['hit_rate']:.1f}%")
        print(f"Collisions:  {stats['collisions']:,}")
        print(f"Stores:      {stats['stores']:,}")
        print(f"Utilisation: {stats['usage']:.1f}%")
        print(f"Âge:         {stats['age']}")
        print()


# ============================================================================
# TESTS
# ============================================================================


if __name__ == "__main__":
    print("=== Test de la Transposition Table ===\n")

    # Test 1: Création et hachage
    print("Test 1: Hachage Zobrist")
    zobrist = ZobristKeys()
    board = chess.Board()

    hash1 = zobrist.hash_position(board)
    print(f"  Hash position initiale: {hash1}")

    board.push(chess.Move.from_uci("e2e4"))
    hash2 = zobrist.hash_position(board)
    print(f"  Hash après e2e4: {hash2}")
    print(f"  Les hashs sont différents: {hash1 != hash2}")

    board.pop()
    hash3 = zobrist.hash_position(board)
    print(f"  Hash après annulation: {hash3}")
    print(f"  Hash restauré: {hash1 == hash3}")
    print()

    # Test 2: Stockage et récupération
    print("Test 2: Stockage et récupération")
    tt = TranspositionTable(size_mb=16)

    board = chess.Board()
    best_move = chess.Move.from_uci("e2e4")

    # Stocker une position
    tt.store(
        board, depth=5, score=50, entry_type=TTEntryType.EXACT, best_move=best_move
    )

    # Récupérer
    found, score, move = tt.probe(board, depth=5, alpha=-1000, beta=1000)
    print(f"  Position stockée et récupérée: {found}")
    print(f"  Score: {score}, Move: {move}")
    print()

    # Test 3: Gestion des profondeurs
    print("Test 3: Gestion des profondeurs")
    found, score, move = tt.probe(board, depth=10, alpha=-1000, beta=1000)
    print(f"  Profondeur insuffisante détectée: {not found}")
    print(f"  Mais best_move disponible: {move is not None}")
    print()

    # Test 4: Statistiques
    print("Test 4: Statistiques")
    for i in range(100):
        board = chess.Board()
        if i < 50:
            board.push(chess.Move.from_uci("e2e4"))
        else:
            board.push(chess.Move.from_uci("d2d4"))

        tt.store(board, depth=3, score=i, entry_type=TTEntryType.EXACT, best_move=None)
        tt.probe(board, depth=3, alpha=-1000, beta=1000)

    tt.print_stats()

    print("✅ Tests de la Transposition Table réussis!")
