"""
IA-Marc V2 - Transposition Table (TT)
======================================

Table de transposition avec hachage Zobrist pour éviter de recalculer
les positions déjà évaluées. Utilise la fonction de hachage Zobrist
de la bibliothèque python-chess pour la robustesse.
"""

import sys
from enum import IntEnum
from typing import Optional, Tuple

import chess
import chess.polyglot

# ============================================================================
# TYPES D'ENTRÉES DE LA TT
# ============================================================================


class TTEntryType(IntEnum):
    """Type d'entrée dans la transposition table."""
    EXACT = 0
    LOWER = 1
    UPPER = 2


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
        self.key = key
        self.depth = depth
        self.score = score
        self.entry_type = entry_type
        self.best_move = best_move
        self.age = age

    def is_valid(self) -> bool:
        return self.key != 0


# ============================================================================
# TRANSPOSITION TABLE
# ============================================================================


class TranspositionTable:
    """
    Table de transposition utilisant le hachage Zobrist de python-chess.
    """

    def __init__(self, size_mb: int = 256):
        self.current_age = 0
        bytes_per_entry = 32
        self.size = (size_mb * 1024 * 1024) // bytes_per_entry
        self.size = 2 ** (self.size.bit_length() - 1) if self.size > 0 else 0
        self.mask = self.size - 1 if self.size > 0 else 0
        self.table = [TTEntry() for _ in range(self.size)]
        self.hits = 0
        self.misses = 0
        self.collisions = 0
        self.stores = 0

    def clear(self):
        if self.size > 0:
            self.table = [TTEntry() for _ in range(self.size)]
        self.hits = 0
        self.misses = 0
        self.collisions = 0
        self.stores = 0
        self.current_age = 0

    def new_search(self):
        self.current_age = (self.current_age + 1) & 255

    def _get_index(self, key: int) -> int:
        return key & self.mask

    def probe(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int = 0
    ) -> Tuple[bool, int, Optional[chess.Move]]:
        if self.size == 0: return False, 0, None
        
        key = chess.polyglot.zobrist_hash(board)
        index = self._get_index(key)
        entry = self.table[index]

        if not entry.is_valid() or entry.key != key:
            self.misses += 1
            return False, 0, None

        self.hits += 1

        if entry.depth < depth:
            return False, 0, entry.best_move

        score = entry.score
        if score > 90000: score -= ply
        elif score < -90000: score += ply

        if entry.entry_type == TTEntryType.EXACT:
            return True, score, entry.best_move
        if entry.entry_type == TTEntryType.LOWER and score >= beta:
            return True, score, entry.best_move
        if entry.entry_type == TTEntryType.UPPER and score <= alpha:
            return True, score, entry.best_move

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
        if self.size == 0: return
        
        key = chess.polyglot.zobrist_hash(board)
        index = self._get_index(key)

        adjusted_score = score
        if score > 90000: adjusted_score += ply
        elif score < -90000: adjusted_score -= ply

        existing = self.table[index]

        if not existing.is_valid() or existing.key == key or depth >= existing.depth or (self.current_age - existing.age) > 4:
            if existing.is_valid() and existing.key != key:
                self.collisions += 1
            self.table[index] = TTEntry(key, depth, adjusted_score, entry_type, best_move, self.current_age)
            self.stores += 1

    def get_pv_move(self, board: chess.Board) -> Optional[chess.Move]:
        if self.size == 0: return None
        
        key = chess.polyglot.zobrist_hash(board)
        index = self._get_index(key)
        entry = self.table[index]
        if entry.is_valid() and entry.key == key and entry.best_move:
            if entry.best_move in board.legal_moves:
                return entry.best_move
        return None

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        usage = sum(1 for e in self.table if e.is_valid()) / self.size * 100 if self.size > 0 else 0
        return {
            "size": self.size, "hits": self.hits, "misses": self.misses,
            "hit_rate": hit_rate, "collisions": self.collisions,
            "stores": self.stores, "usage": usage, "age": self.current_age,
        }