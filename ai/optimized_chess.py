#!/usr/bin/env python3
"""
Optimizations pour Chess.py - Ciblage des goulots d'étranglement identifiés par le profiler

PROBLÈMES IDENTIFIÉS:
1. square_mask() : 6.3M appels (8000 par nœud évalué !)
2. pieces_of_color() : 68k appels avec boucles sur bitboards
3. occupancy() : 118k appels avec boucles sur bitboards  
4. move_piece()/undo_move() : copies coûteuses de bitboards

SOLUTIONS D'OPTIMISATION:
1. Cache pré-calculé pour square_mask
2. Cache pour pieces_of_color par couleur
3. Cache pour occupancy 
4. Références au lieu de copies quand possible
"""

import numpy as np
from typing import Dict, Optional, Tuple
import random

# ---------------------- ZOBRIST HASHING ----------------------
# Generate Zobrist keys at module import. We use Python ints (64-bit) and
# a deterministic seed can be set here if reproducibility is desired.
_ZOBRIST_PIECES = None  # dict piece -> [64 keys]
_ZOBRIST_SIDE = None
_ZOBRIST_CASTLING = None  # dict 'K','Q','k','q' -> key
_ZOBRIST_EP_FILE = None  # file 0..7 -> key


def _init_zobrist(seed: Optional[int] = 0xC0FFEE):
    """Initialise tables Zobrist (called once)."""
    global _ZOBRIST_PIECES, _ZOBRIST_SIDE, _ZOBRIST_CASTLING, _ZOBRIST_EP_FILE
    rnd = random.Random(seed)

    pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    _ZOBRIST_PIECES = {p: [rnd.getrandbits(64) for _ in range(64)] for p in pieces}
    _ZOBRIST_SIDE = rnd.getrandbits(64)
    _ZOBRIST_CASTLING = {k: rnd.getrandbits(64) for k in ['K', 'Q', 'k', 'q']}
    _ZOBRIST_EP_FILE = [rnd.getrandbits(64) for _ in range(8)]


# Initialize Zobrist tables on import
_init_zobrist()


def compute_zobrist(chess) -> int:
    """Compute a Zobrist key for the given chess position.

    This implementation is intentionally simple: it iterates piece bitboards
    and XORs the corresponding piece-square keys. It also XORs side-to-move,
    castling rights and en-passant file when present. It's significantly
    faster than building a string and calling md5.
    """
    key = 0

    # Piece bitboards
    for piece, bb in getattr(chess, 'bitboards', {}).items():
        if bb == 0:
            continue
        # iterate set bits
        temp = int(bb)
        piece_table = _ZOBRIST_PIECES.get(piece)
        if piece_table is None:
            continue
        while temp:
            sq = (temp & -temp).bit_length() - 1
            key ^= piece_table[sq]
            temp &= temp - 1

    # Side to move
    if getattr(chess, 'white_to_move', False):
        key ^= _ZOBRIST_SIDE

    # Castling rights
    cr = getattr(chess, 'castling_rights', {})
    for k, present in cr.items():
        if present and k in _ZOBRIST_CASTLING:
            key ^= _ZOBRIST_CASTLING[k]

    # En-passant file (if any) - use file index 0..7
    # Chess.py uses attribute name 'en_passant_target'; accept both for compatibility
    ep = getattr(chess, 'en_passant_target', None)
    if ep is None:
        ep = getattr(chess, 'en_passant_square', None)
    if ep is not None:
        # en_passant_square may be an int square 0..63 or None
        file_index = ep % 8
        key ^= _ZOBRIST_EP_FILE[file_index]

    return key


def _zobrist_xor_move(chess, record_prev=False, prev_state=None):
    """
    Apply incremental XOR updates to chess.zobrist_key based on the last move recorded in chess.history.
    This function assumes the move has already been applied to the board state when called.

    If record_prev=True and prev_state is provided, the function will store enough info
    to allow undo to reverse the XOR by saving the previous key in prev_state['prev_zobrist'].
    """
    # Safety: if zobrist tables missing, do nothing
    global _ZOBRIST_PIECES, _ZOBRIST_SIDE, _ZOBRIST_CASTLING, _ZOBRIST_EP_FILE
    if _ZOBRIST_PIECES is None:
        return

    if not hasattr(chess, 'history') or not chess.history:
        # nothing to do
        return

    last = chess.history[-1]

    # Ensure zobrist_key exists on the object
    prev_key = getattr(chess, 'zobrist_key', None)
    if prev_key is None:
        # fallback: compute full key and store
        try:
            new_key = compute_zobrist(chess)
            if record_prev and prev_state is not None:
                prev_state['prev_zobrist'] = None
            chess.zobrist_key = new_key
        except Exception:
            chess.zobrist_key = None
        return

    key = prev_key

    # We will XOR out the moved piece from its origin and XOR in at destination.
    try:
        moving = last.get('moving_piece')
        from_sq = last.get('from')
        to_sq = last.get('to')
        captured = last.get('captured_piece')
        captured_sq = last.get('captured_square')
        promotion = last.get('promotion')
    except Exception:
        # If history doesn't contain expected fields, fallback to full recompute
        chess.zobrist_key = compute_zobrist(chess)
        return

    # Record prev key for undo if requested
    if record_prev and prev_state is not None:
        prev_state['prev_zobrist'] = prev_key

    # Toggle side-to-move key (always toggled after a legal move)
    key ^= _ZOBRIST_SIDE

    # Handle castling rights changes: XOR any castling flags that differ from previous
    # We rely on last['prev_castling'] existing (move_piece ensures it)
    prev_cr = last.get('prev_castling', {})
    curr_cr = getattr(chess, 'castling_rights', {})
    for cr in ['K', 'Q', 'k', 'q']:
        prev_val = prev_cr.get(cr, False)
        curr_val = curr_cr.get(cr, False)
        if prev_val != curr_val:
            # XOR the corresponding castling key
            key ^= _ZOBRIST_CASTLING.get(cr, 0)

    # Handle en-passant file: prev_en_passant may be None or square
    prev_ep = last.get('prev_en_passant')
    curr_ep = getattr(chess, 'en_passant_target', None)
    if prev_ep is not None:
        prev_file = int(prev_ep) % 8
    else:
        prev_file = None
    if curr_ep is not None:
        curr_file = int(curr_ep) % 8
    else:
        curr_file = None

    # XOR out previous EP file if present
    if prev_file is not None:
        key ^= _ZOBRIST_EP_FILE[prev_file]
    # XOR in current EP file if present
    if curr_file is not None:
        key ^= _ZOBRIST_EP_FILE[curr_file]

    # Pieces: we need to XOR out the moving piece at from_sq using the piece identity
    # BEFORE move, and XOR in at to_sq using the piece identity AFTER move.
    # Because move has already been applied, for promotions the piece on to_sq may differ.
    # We'll reconstruct expected piece identities from the record.

    # XOR out moving piece at from_sq using the recorded moving_piece
    if moving is not None and from_sq is not None:
        piece_table = _ZOBRIST_PIECES.get(moving)
        if piece_table is not None:
            key ^= piece_table[int(from_sq)]

    # If there was a capture, XOR out the captured piece at its square (captured_square)
    if captured is not None and captured_sq is not None:
        cap_table = _ZOBRIST_PIECES.get(captured)
        if cap_table is not None:
            key ^= cap_table[int(captured_sq)]

    # Handle promotions: if promotion is not None then the pawn was replaced by promoted piece
    if promotion:
        # promotion variable is stored as uppercase letter for white promotions when move made
        # Determine which side moved using prev_white_to_move
        mover_was_white = bool(last.get('prev_white_to_move', True))
        # pawn char
        pawn_char = 'P' if mover_was_white else 'p'
        promoted_char = promotion if mover_was_white else promotion.lower()
        # XOR out pawn at to_sq (since pawn was moved to to_sq then removed)
        pawn_table = _ZOBRIST_PIECES.get(pawn_char)
        if pawn_table is not None:
            key ^= pawn_table[int(to_sq)]
        # XOR in promoted piece at to_sq
        prom_table = _ZOBRIST_PIECES.get(promoted_char)
        if prom_table is not None:
            key ^= prom_table[int(to_sq)]
    else:
        # No promotion: XOR in moving piece at to_sq (same piece char)
        if moving is not None and to_sq is not None:
            piece_table = _ZOBRIST_PIECES.get(moving)
            if piece_table is not None:
                key ^= piece_table[int(to_sq)]

    # Finally write back the updated key
    chess.zobrist_key = key


class OptimizedChessBitboards:
    """
    Optimisations ciblées pour les goulots d'étranglement Chess.py
    """
    
    def __init__(self):
        # OPTIMISATION 1: Cache pré-calculé pour square_mask (évite 6.3M calculs !)
        # Use Python ints for masks
        self.SQUARE_MASKS = {i: (1 << i) for i in range(64)}
        
        # OPTIMISATION 2: Cache pour pieces_of_color
        self._pieces_cache = {'white': 0, 'black': 0}
        self._cache_valid = False
        
        # OPTIMISATION 3: Cache pour occupancy
        self._occupancy_cache = 0
        self._occupancy_valid = False
        
        # OPTIMISATION 4: Éviter les copies dans bitboards
        self._bitboards_dirty = True

    def optimized_square_mask(self, sq: int) -> int:
        """Version optimisée de square_mask - O(1) au lieu de calcul"""
        return self.SQUARE_MASKS[sq]
    
    def optimized_pieces_of_color(self, bitboards: Dict[str, int], white: bool) -> int:
        """Version optimisée de pieces_of_color avec cache"""
        if not self._cache_valid or self._bitboards_dirty:
            self._update_pieces_cache(bitboards)
        
        return self._pieces_cache['white'] if white else self._pieces_cache['black']
    
    def optimized_occupancy(self, bitboards: Dict[str, int]) -> int:
        """Version optimisée d'occupancy avec cache"""
        if not self._occupancy_valid or self._bitboards_dirty:
            self._update_occupancy_cache(bitboards)
        
        return self._occupancy_cache
    
    def _update_pieces_cache(self, bitboards: Dict[str, int]):
        """Met à jour le cache des pièces par couleur"""
        white_mask = 0
        black_mask = 0
        
        for piece, bb in bitboards.items():
            if piece.isupper():  # Pièces blanches
                white_mask |= int(bb)
            else:  # Pièces noires
                black_mask |= int(bb)
        
        self._pieces_cache['white'] = white_mask
        self._pieces_cache['black'] = black_mask
        self._cache_valid = True
    
    def _update_occupancy_cache(self, bitboards: Dict[str, int]):
        """Met à jour le cache d'occupancy"""
        occ = 0
        for bb in bitboards.values():
            occ |= int(bb)
        
        self._occupancy_cache = occ
        self._occupancy_valid = True
    
    def invalidate_cache(self):
        """Invalide tous les caches quand les bitboards changent"""
        self._cache_valid = False
        self._occupancy_valid = False
        self._bitboards_dirty = True
    
    def validate_cache(self):
        """Marque les caches comme valides après mise à jour"""
        self._bitboards_dirty = False

# Instance globale pour éviter la re-création
_optimizer = OptimizedChessBitboards()

def patch_chess_class(chess_instance):
    """
    Patch une instance de Chess avec les optimisations
    """
    # Sauvegarde des méthodes originales
    chess_instance._original_square_mask = chess_instance.square_mask
    chess_instance._original_pieces_of_color = chess_instance.pieces_of_color
    chess_instance._original_occupancy = chess_instance.occupancy
    chess_instance._original_move_piece = chess_instance.move_piece
    chess_instance._original_undo_move = chess_instance.undo_move
    
    # Injection des versions optimisées
    chess_instance.square_mask = lambda sq: _optimizer.optimized_square_mask(sq)
    chess_instance.pieces_of_color = lambda white: _optimizer.optimized_pieces_of_color(chess_instance.bitboards, white)
    chess_instance.occupancy = lambda: _optimizer.optimized_occupancy(chess_instance.bitboards)
    
    # Wrapper pour move_piece qui invalide le cache
    def optimized_move_piece(from_sq, to_sq, promotion=None):
        # Maintain zobrist key: compute previous key, call original move, store prev in history and recompute key
        try:
            from optimized_chess import compute_zobrist
        except Exception:
            compute_zobrist = None

        prev_key = None
        if compute_zobrist is not None:
            try:
                prev_key = compute_zobrist(chess_instance)
            except Exception:
                prev_key = None

        _optimizer.invalidate_cache()
        result = chess_instance._original_move_piece(from_sq, to_sq, promotion)
        _optimizer.validate_cache()

        # Store prev_zobrist in last history record if present and try incremental XOR update
        try:
            if prev_key is not None and hasattr(chess_instance, 'history') and chess_instance.history:
                # save prev key for undo safety
                chess_instance.history[-1]['prev_zobrist'] = prev_key
                # attempt incremental xor update
                try:
                    _zobrist_xor_move(chess_instance, record_prev=False)
                except Exception:
                    # fallback: full recompute
                    try:
                        chess_instance.zobrist_key = compute_zobrist(chess_instance)
                    except Exception:
                        chess_instance.zobrist_key = None
        except Exception:
            pass

        return result
    
    # Wrapper pour undo_move qui invalide le cache
    def optimized_undo_move():
        # Restore zobrist from history if available
        prev_key = None
        try:
            if hasattr(chess_instance, 'history') and chess_instance.history:
                prev_key = chess_instance.history[-1].get('prev_zobrist')
        except Exception:
            prev_key = None

        _optimizer.invalidate_cache()
        result = chess_instance._original_undo_move()
        _optimizer.validate_cache()

        try:
            if prev_key is not None:
                chess_instance.zobrist_key = prev_key
            else:
                # fallback: recompute if possible
                try:
                    from optimized_chess import compute_zobrist
                    chess_instance.zobrist_key = compute_zobrist(chess_instance)
                except Exception:
                    chess_instance.zobrist_key = None
        except Exception:
            pass

        return result
    
    chess_instance.move_piece = optimized_move_piece
    chess_instance.undo_move = optimized_undo_move
    
    # Initialisation du cache
    _optimizer.invalidate_cache()
    _optimizer.validate_cache()

    # Initialize zobrist key on patched instance if possible
    try:
        from optimized_chess import compute_zobrist
        chess_instance.zobrist_key = compute_zobrist(chess_instance)
    except Exception:
        chess_instance.zobrist_key = None
    
    return chess_instance

def unpatch_chess_class(chess_instance):
    """
    Restaure les méthodes originales de Chess
    """
    if hasattr(chess_instance, '_original_square_mask'):
        chess_instance.square_mask = chess_instance._original_square_mask
        chess_instance.pieces_of_color = chess_instance._original_pieces_of_color  
        chess_instance.occupancy = chess_instance._original_occupancy
        chess_instance.move_piece = chess_instance._original_move_piece
        chess_instance.undo_move = chess_instance._original_undo_move

class OptimizedChessBoard:
    """
    Version complètement optimisée du ChessBoard avec intégration native des optimisations
    """
    
    def __init__(self):
        # Bitboards standard
        self.bitboards = {
            'P': 0x000000000000FF00,  # White pawns
            'R': 0x0000000000000081,  # White rooks
            'N': 0x0000000000000042,  # White knights
            'B': 0x0000000000000024,  # White bishops
            'Q': 0x0000000000000008,  # White queen
            'K': 0x0000000000000010,  # White king
            'p': 0x00FF000000000000,  # Black pawns
            'r': 0x8100000000000000,  # Black rooks
            'n': 0x4200000000000000,  # Black knights
            'b': 0x2400000000000000,  # Black bishops
            'q': 0x0800000000000000,  # Black queen
            'k': 0x1000000000000000,  # Black king
        }
        
        # Intégration de l'optimiseur
        self.optimizer = OptimizedChessBitboards()
        
        # État du jeu
        self.white_to_move = True
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_target = None
        self.check = False
        self.history = []
    
    def square_mask(self, sq: int) -> int:
        """Version optimisée native"""
        return self.optimizer.optimized_square_mask(sq)
    
    def pieces_of_color(self, white: bool) -> int:
        """Version optimisée native"""
        return self.optimizer.optimized_pieces_of_color(self.bitboards, white)
    
    def occupancy(self) -> int:
        """Version optimisée native"""
        return self.optimizer.optimized_occupancy(self.bitboards)
    
    def _invalidate_caches(self):
        """Invalide les caches après modification des bitboards"""
        self.optimizer.invalidate_cache()
    
    def _validate_caches(self):
        """Valide les caches après modification des bitboards"""
        self.optimizer.validate_cache()

def benchmark_optimizations():
    """
    Benchmark des optimisations vs version originale
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from Chess import Chess
    import time
    
    print("🚀 === BENCHMARK DES OPTIMISATIONS === 🚀")
    
    # Test 1: square_mask
    chess = Chess()
    
    print("\n📊 Test 1: square_mask (1M appels)")
    
    # Version originale
    start = time.time()
    for _ in range(1000000):
        for sq in range(8):  # 8M appels total
            chess.square_mask(sq)
    original_time = time.time() - start
    
    # Version optimisée
    patch_chess_class(chess)
    start = time.time()
    for _ in range(1000000):
        for sq in range(8):  # 8M appels total
            chess.square_mask(sq)
    optimized_time = time.time() - start
    
    print(f"⏱️  Original: {original_time:.3f}s")
    print(f"⚡ Optimized: {optimized_time:.3f}s") 
    print(f"📈 Speedup: {original_time/optimized_time:.1f}x")
    
    # Test 2: pieces_of_color
    print("\n📊 Test 2: pieces_of_color (100k appels)")
    
    unpatch_chess_class(chess)
    start = time.time()
    for _ in range(100000):
        chess.pieces_of_color(True)
        chess.pieces_of_color(False)
    original_time = time.time() - start
    
    patch_chess_class(chess)
    start = time.time()
    for _ in range(100000):
        chess.pieces_of_color(True)
        chess.pieces_of_color(False)
    optimized_time = time.time() - start
    
    print(f"⏱️  Original: {original_time:.3f}s")
    print(f"⚡ Optimized: {optimized_time:.3f}s")
    print(f"📈 Speedup: {original_time/optimized_time:.1f}x")
    
    print("\n✅ Benchmark terminé!")

def patch_chess_class_globally():
    """
    Patch la classe Chess globalement (monkey patching)
    """
    from Chess import Chess

    # Ne rien faire si déjà patché
    if getattr(Chess, '_optimized_patched', False):
        return

    # Sauvegarder les méthodes originales de la classe (si pas déjà sauvegardées)
    if not hasattr(Chess, '_original_square_mask'):
        Chess._original_square_mask = Chess.square_mask
    if not hasattr(Chess, '_original_pieces_of_color'):
        Chess._original_pieces_of_color = Chess.pieces_of_color
    if not hasattr(Chess, '_original_occupancy'):
        Chess._original_occupancy = Chess.occupancy
    if not hasattr(Chess, '_original_move_piece'):
        Chess._original_move_piece = Chess.move_piece
    if not hasattr(Chess, '_original_undo_move'):
        Chess._original_undo_move = Chess.undo_move

    # Remplacer les méthodes de la classe
    def optimized_square_mask(self, sq):
        return _optimizer.optimized_square_mask(sq)

    def optimized_pieces_of_color(self, white):
        return _optimizer.optimized_pieces_of_color(self.bitboards, white)

    def optimized_occupancy(self):
        return _optimizer.optimized_occupancy(self.bitboards)

    # Les wrappers appellent les méthodes originales de la CLASSE (_original_move_piece)
    # afin d'éviter la récursion si on patch plusieurs fois
    def optimized_move_piece(self, from_sq, to_sq, promotion=None):
        _optimizer.invalidate_cache()
        # Maintain zobrist: store previous key if compute_zobrist available
        try:
            from optimized_chess import compute_zobrist
            prev_key = compute_zobrist(self)
        except Exception:
            prev_key = None

        result = Chess._original_move_piece(self, from_sq, to_sq, promotion)
        _optimizer.validate_cache()

        try:
            if prev_key is not None and hasattr(self, 'history') and self.history:
                self.history[-1]['prev_zobrist'] = prev_key
                try:
                    # Try incremental XOR update; on failure fallback to full recompute
                    try:
                        _zobrist_xor_move(self, record_prev=False)
                    except Exception:
                        self.zobrist_key = compute_zobrist(self)
                except Exception:
                    self.zobrist_key = None
        except Exception:
            pass

        return result

    def optimized_undo_move(self):
        _optimizer.invalidate_cache()
        # Try to read prev_zobrist from history (last entry)
        prev_key = None
        try:
            if hasattr(self, 'history') and self.history:
                prev_key = self.history[-1].get('prev_zobrist')
        except Exception:
            prev_key = None

        result = Chess._original_undo_move(self)
        _optimizer.validate_cache()

        try:
            if prev_key is not None:
                # restore stored previous key
                self.zobrist_key = prev_key
            else:
                # fallback: recompute if possible
                try:
                    from optimized_chess import compute_zobrist
                    self.zobrist_key = compute_zobrist(self)
                except Exception:
                    self.zobrist_key = None
        except Exception:
            pass

        return result

    # Appliquer les patches à la classe
    Chess.square_mask = optimized_square_mask
    Chess.pieces_of_color = optimized_pieces_of_color
    Chess.occupancy = optimized_occupancy
    Chess.move_piece = optimized_move_piece
    Chess.undo_move = optimized_undo_move

    # Marquer comme patché
    Chess._optimized_patched = True

    # Initialize zobrist_key on existing instances? Not necessary — when a Chess is created
    # and patched later, its zobrist_key will be computed on first move/patch usage.

if __name__ == "__main__":
    benchmark_optimizations()