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

class OptimizedChessBitboards:
    """
    Optimisations ciblées pour les goulots d'étranglement Chess.py
    """
    
    def __init__(self):
        # OPTIMISATION 1: Cache pré-calculé pour square_mask (évite 6.3M calculs !)
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
        _optimizer.invalidate_cache()
        result = chess_instance._original_move_piece(from_sq, to_sq, promotion)
        _optimizer.validate_cache()
        return result
    
    # Wrapper pour undo_move qui invalide le cache
    def optimized_undo_move():
        _optimizer.invalidate_cache()
        result = chess_instance._original_undo_move()
        _optimizer.validate_cache()
        return result
    
    chess_instance.move_piece = optimized_move_piece
    chess_instance.undo_move = optimized_undo_move
    
    # Initialisation du cache
    _optimizer.invalidate_cache()
    _optimizer.validate_cache()
    
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

if __name__ == "__main__":
    benchmark_optimizations()