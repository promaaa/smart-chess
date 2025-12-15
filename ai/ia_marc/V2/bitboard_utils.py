"""
Bitboard Utilities
==================

Fonctions utilitaires pour la manipulation de bitboards, y compris la
génération d'attaques pour les pièces coulissantes (tours, fous, dames).
Cette implémentation utilise une approche "classique" sans magic bitboards,
ce qui est un bon compromis entre simplicité et performance pour du pur Python.
"""

import chess

# ============================================================================
# CONSTANTES ET TABLES PRÉ-CALCULÉES
# ============================================================================

# Masques pour les fichiers et les rangées
FILE_A = chess.BB_FILE_A
FILE_H = chess.BB_FILE_H
RANK_1 = chess.BB_RANK_1
RANK_8 = chess.BB_RANK_8

# ============================================================================
# GÉNÉRATION D'ATTAQUES
# ============================================================================

def get_rook_attacks(square: int, occupied: int) -> int:
    """
    Calcule le bitboard des attaques pour une tour sur une case donnée.
    """
    attacks = 0
    
    # Directions: Nord, Sud, Est, Ouest
    directions = [8, -8, 1, -1]
    
    for direction in directions:
        sq = square
        while True:
            sq += direction
            # Vérifier les bords de l'échiquier
            if not (0 <= sq < 64): break
            if (direction == 1 or direction == -1) and chess.square_distance(sq, sq - direction) > 1: break

            attacks |= (1 << sq)
            if (1 << sq) & occupied:
                break # Arrêter après avoir inclus la pièce bloquante
                
    return attacks

def get_bishop_attacks(square: int, occupied: int) -> int:
    """
    Calcule le bitboard des attaques pour un fou sur une case donnée.
    """
    attacks = 0
    
    # Directions: Nord-Est, Nord-Ouest, Sud-Est, Sud-Ouest
    directions = [7, 9, -7, -9]

    for direction in directions:
        sq = square
        while True:
            sq += direction
            # Vérifier les bords de l'échiquier
            if not (0 <= sq < 64): break
            if chess.square_distance(sq, sq - direction) > 2: break
                
            attacks |= (1 << sq)
            if (1 << sq) & occupied:
                break # Arrêter après avoir inclus la pièce bloquante

    return attacks

def get_queen_attacks(square: int, occupied: int) -> int:
    """
    Calcule le bitboard des attaques pour une dame.
    """
    return get_rook_attacks(square, occupied) | get_bishop_attacks(square, occupied)