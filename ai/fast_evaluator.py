#!/usr/bin/env python3
"""
Évaluateur optimisé pour Chess AI - Corrige les goulots d'étranglement de performance

PROBLÈMES CORRIGÉS:
1. Boucles multiples sur 64 cases (768+ appels square_mask par évaluation)
2. Génération complète des coups pour la mobilité  
3. Calculs redondants

OPTIMISATIONS APPLIQUÉES:
1. Itération directe sur les bitboards avec manipulation bits
2. Approximation de mobilité sans génération de coups
3. Tables de position pré-calculées et optimisées
4. Cache des résultats intermédiaires
"""

import numpy as np

class FastChessEvaluator:
    """Évaluateur optimisé pour de meilleures performances"""
    
    def __init__(self):
        # Valeurs des pièces
        self.piece_values = {
            'P': 100, 'N': 300, 'B': 300, 'R': 500, 'Q': 900, 'K': 10000,
            'p': -100, 'n': -300, 'b': -300, 'r': -500, 'q': -900, 'k': -10000
        }
        # Pré-calculer les masques de cases pour éviter les calculs répétitifs
        # Use Python ints to avoid numpy.uint64 overhead on hot paths
        self.square_masks = [1 << i for i in range(64)]
        
        # Tables de position optimisées (valeurs en centipions)
        self.pawn_table = np.array([
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10,-20,-20, 10, 10,  5,
            5, -5,-10,  0,  0,-10, -5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5,  5, 10, 25, 25, 10,  5,  5,
           10, 10, 20, 30, 30, 20, 10, 10,
           50, 50, 50, 50, 50, 50, 50, 50,
            0,  0,  0,  0,  0,  0,  0,  0
        ], dtype=np.int16)
        
        self.knight_table = np.array([
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ], dtype=np.int16)
        
        self.bishop_table = np.array([
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ], dtype=np.int16)
        
        self.rook_table = np.array([
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ], dtype=np.int16)
        
        self.queen_table = np.array([
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ], dtype=np.int16)
        
        self.king_table = np.array([
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ], dtype=np.int16)

    def evaluate_position(self, chess):
        """Évaluation optimisée de la position"""
        score = 0
        
        # 1. Évaluation matérielle ET positionnelle combinées (une seule passe)
        score += self._fast_material_and_position_eval(chess)
        
        # 2. Mobilité approximative (très rapide)
        score += self._fast_mobility_eval(chess)
        
        # 3. Évaluation des échecs (déjà optimisé)
        score += self._evaluate_check_status(chess)
        
        return score
    
    def _fast_material_and_position_eval(self, chess):
        """Évaluation matérielle et positionnelle en une seule passe"""
        score = 0
        
        # Traitement de chaque type de pièce
        piece_configs = [
            ('P', 'p', self.piece_values['P'], self.pawn_table),
            ('N', 'n', self.piece_values['N'], self.knight_table),
            ('B', 'b', self.piece_values['B'], self.bishop_table),
            ('R', 'r', self.piece_values['R'], self.rook_table),
            ('Q', 'q', self.piece_values['Q'], self.queen_table),
            ('K', 'k', self.piece_values['K'], self.king_table)
        ]
        
        for white_piece, black_piece, piece_value, position_table in piece_configs:
            # Pièces blanches
            white_bitboard = chess.bitboards.get(white_piece, 0)
            score += self._eval_pieces_on_bitboard(white_bitboard, piece_value, position_table, False)
            
            # Pièces noires (table inversée)
            black_bitboard = chess.bitboards.get(black_piece, 0)
            score += self._eval_pieces_on_bitboard(black_bitboard, -piece_value, position_table, True)
        
        return score
    
    def _eval_pieces_on_bitboard(self, bitboard, piece_value, position_table, invert_table):
        """Évalue toutes les pièces d'un bitboard donné"""
        if bitboard == 0:
            return 0
            
        score = 0
        temp_bitboard = int(bitboard)
        
        # Parcourir tous les bits à 1 (positions des pièces)
        while temp_bitboard:
            # Trouver la position du bit le plus bas à 1
            square = (temp_bitboard & -temp_bitboard).bit_length() - 1
            
            # Ajouter la valeur matérielle
            score += piece_value
            
            # Ajouter la valeur positionnelle
            table_index = (63 - square) if invert_table else square
            score += position_table[table_index]
            
            # Supprimer ce bit du bitboard temporaire
            temp_bitboard &= temp_bitboard - 1
        
        return score
    
    def _fast_mobility_eval(self, chess):
        """Évaluation approximative de la mobilité sans génération de coups"""
        score = 0
        
        # Approximation basée sur les pièces développées
        # Au lieu de compter tous les coups, on récompense les pièces actives
        
        # Cavaliers développés (hors cases de départ)
        white_knights = chess.bitboards.get('N', 0)
        black_knights = chess.bitboards.get('n', 0)
        
        # Cases de départ des cavaliers
        knight_start_white = self.square_masks[1] | self.square_masks[6]  # b1, g1
        knight_start_black = self.square_masks[57] | self.square_masks[62]  # b8, g8
        
        # Bonus pour cavaliers développés
        developed_white_knights = white_knights & ~knight_start_white
        developed_black_knights = black_knights & ~knight_start_black
        
        score += int(developed_white_knights).bit_count() * 10
        score -= int(developed_black_knights).bit_count() * 10
        
        # Fous développés (approximation basée sur la diagonale)
        white_bishops = chess.bitboards.get('B', 0)
        black_bishops = chess.bitboards.get('b', 0)
        
        # Bonus pour fous sortis de la première rangée
        white_bishop_developed = white_bishops & ~(self.square_masks[2] | self.square_masks[5])  # c1, f1
        black_bishop_developed = black_bishops & ~(self.square_masks[58] | self.square_masks[61])  # c8, f8
        
        score += int(white_bishop_developed).bit_count() * 8
        score -= int(black_bishop_developed).bit_count() * 8

        return score
    
    def _evaluate_check_status(self, chess):
        """Pénalité pour être en échec (déjà optimisé)"""
        score = 0
        if chess.is_in_check(True):
            score -= 50
        if chess.is_in_check(False):
            score += 50
        return score

class SuperFastChessEvaluator:
    """Évaluateur ultra-rapide - seulement matériel + position basique"""
    
    def __init__(self):
        self.piece_values = {
            'P': 100, 'N': 300, 'B': 300, 'R': 500, 'Q': 900, 'K': 10000,
            'p': -100, 'n': -300, 'b': -300, 'r': -500, 'q': -900, 'k': -10000
        }
        
        # Bonus de position très simplifié (seulement centre)
        self.center_squares = (
            (1 << 27) | (1 << 28) | (1 << 35) | (1 << 36)  # d4, e4, d5, e5
        )
        
    def evaluate_position(self, chess):
        """Évaluation ultra-rapide - matériel + contrôle du centre"""
        score = 0
        
        # Matériel pur
        for piece, bitboard in chess.bitboards.items():
            piece_count = int(bitboard).bit_count()
            score += self.piece_values.get(piece, 0) * piece_count
        
        # Bonus simple pour contrôle du centre
        white_pieces = chess.pieces_of_color(True)
        black_pieces = chess.pieces_of_color(False)
        
        white_center = white_pieces & self.center_squares
        black_center = black_pieces & self.center_squares
        
        score += int(white_center).bit_count() * 20
        score -= int(black_center).bit_count() * 20

        return score

# Pour compatibilité avec le code existant
ChessEvaluator = FastChessEvaluator