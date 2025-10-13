'''complètement généré par IA pour tester le parcours d'arbre minimax'''

class ChessEvaluator:
    """Évaluateur de position d'échecs basique"""
    
    def __init__(self):
        # Valeurs des pièces
        self.piece_values = {
            'P': 100, 'N': 300, 'B': 300, 'R': 500, 'Q': 900, 'K': 10000,
            'p': -100, 'n': -300, 'b': -300, 'r': -500, 'q': -900, 'k': -10000
        }
    
    def evaluate_position(self, chess):
        """
        Évalue une position d'échecs.
        Retourne un score positif si les blancs sont avantagés,
        négatif si les noirs sont avantagés.
        """
        score = 0
        
        # Évaluation matérielle
        score += self._evaluate_material(chess)
        
        # Bonus pour mobilité (nombre de coups légaux)
        score += self._evaluate_mobility(chess)
        
        # Pénalité pour être en échec
        score += self._evaluate_check_status(chess)
        
        # Position des pièces (tables de position)
        score += self._evaluate_piece_positions(chess)
            
        return score
    
    def _evaluate_material(self, chess):
        """Évaluation matérielle basique"""
        score = 0
        for piece, bitboard in chess.bitboards.items():
            piece_count = bin(int(bitboard)).count('1')
            score += self.piece_values.get(piece, 0) * piece_count
        return score
    
    def _evaluate_mobility(self, chess):
        """Évalue la mobilité (nombre de coups possibles)"""
        white_moves = self._count_pseudo_legal_moves(chess, True)
        black_moves = self._count_pseudo_legal_moves(chess, False)
        return (white_moves - black_moves) * 2
    
    def _evaluate_check_status(self, chess):
        """Pénalité pour être en échec"""
        score = 0
        if chess.is_in_check(True):
            score -= 50
        if chess.is_in_check(False):
            score += 50
        return score
    
    def _evaluate_piece_positions(self, chess):
        """Évaluation basée sur la position des pièces (tables de position simplifiées)"""
        score = 0
        
        # Tables de position pour les pions (encourage le développement central)
        pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10,-20,-20, 10, 10,  5,
            5, -5,-10,  0,  0,-10, -5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5,  5, 10, 25, 25, 10,  5,  5,
           10, 10, 20, 30, 30, 20, 10, 10,
           50, 50, 50, 50, 50, 50, 50, 50,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        # Évaluer les pions blancs
        for square in range(64):
            if chess.bitboards.get('P', 0) & chess.square_mask(square):
                score += pawn_table[square] / 100
        
        # Évaluer les pions noirs (table inversée)
        for square in range(64):
            if chess.bitboards.get('p', 0) & chess.square_mask(square):
                score -= pawn_table[63 - square] / 100
                
        knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        # Évaluer les cavaliers blancs
        for square in range(64):
            if chess.bitboards.get('N', 0) & chess.square_mask(square):
                score += knight_table[square] / 100
        
        # Évaluer les cavaliers noirs (table inversée)
        for square in range(64):
            if chess.bitboards.get('n', 0) & chess.square_mask(square):
                score -= knight_table[63 - square] / 100

        bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]

        # Évaluer les fous blancs
        for square in range(64):
            if chess.bitboards.get('B', 0) & chess.square_mask(square):
                score += bishop_table[square] / 100

        # Évaluer les fous noirs (table inversée)
        for square in range(64):
            if chess.bitboards.get('b', 0) & chess.square_mask(square):
                score -= bishop_table[63 - square] / 100

        rook_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
            ]
        
        # Évaluer les tours blanches
        for square in range(64):
            if chess.bitboards.get('R', 0) & chess.square_mask(square):
                score += rook_table[square] / 100
        
        # Évaluer les tours noires (table inversée)
        for square in range(64):
            if chess.bitboards.get('r', 0) & chess.square_mask(square):
                score -= rook_table[63 - square] / 100
    
        queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]

        # Évaluer la reine blanche
        for square in range(64):
            if chess.bitboards.get('Q', 0) & chess.square_mask(square):
                score += queen_table[square] / 100
        
        # Évaluer les reines noires (table inversée)
        for square in range(64):
            if chess.bitboards.get('q', 0) & chess.square_mask(square):
                score -= queen_table[63 - square] / 100

        knight_table = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]

        # Évaluer le roi blanc (milieu de partie)
        for square in range(64):
            if chess.bitboards.get('K', 0) & chess.square_mask(square):
                score += knight_table[square] / 100
        # Évaluer le roi noir (milieu de partie, table inversée)
        for square in range(64):
            if chess.bitboards.get('k', 0) & chess.square_mask(square):
                score -= knight_table[63 - square] / 100
        
        return score
    
    def _count_pseudo_legal_moves(self, chess, is_white):
        """Compte le nombre de coups pseudo-légaux pour une couleur"""
        count = 0
        for square in range(64):
            piece = self._get_piece_at(chess, square)
            if piece and (piece.isupper() == is_white):
                moves = chess.get_all_moves(square)
                # Compter les bits à 1 dans moves
                count += bin(int(moves)).count('1')
        return count
    
    def _get_piece_at(self, chess, square):
        """Retourne la pièce à une case donnée"""
        mask = chess.square_mask(square)
        for piece, bitboard in chess.bitboards.items():
            if bitboard & mask:
                return piece
        return None

class AdvancedChessEvaluator(ChessEvaluator):
    """Évaluateur avancé avec plus de critères d'évaluation"""
    
    def __init__(self):
        super().__init__()
        # Ajustement des valeurs pour un jeu plus positionnel
        self.piece_values = {
            'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
            'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000
        }
    
    def evaluate_position(self, chess):
        """Évaluation plus sophistiquée"""
        score = super().evaluate_position(chess)
        
        # Critères additionnels
        score += self._evaluate_king_safety(chess)
        score += self._evaluate_pawn_structure(chess)
        score += self._evaluate_piece_coordination(chess)
        
        return score
    
    def _evaluate_king_safety(self, chess):
        """Évalue la sécurité du roi"""
        # Implémentation simplifiée - bonus si le roi est roqué
        score = 0
        
        # Vérifier si les rois ont roqué (positions typiques après roque)
        white_king_safe_squares = [1, 2, 6]  # b1, c1, g1
        black_king_safe_squares = [57, 58, 62]  # b8, c8, g8
        
        for square in white_king_safe_squares:
            if chess.bitboards.get('K', 0) & chess.square_mask(square):
                score += 30
                break
                
        for square in black_king_safe_squares:
            if chess.bitboards.get('k', 0) & chess.square_mask(square):
                score -= 30
                break
                
        return score
    
    def _evaluate_pawn_structure(self, chess):
        """Évalue la structure de pions"""
        # Implémentation basique - pénalise les pions doublés
        score = 0
        
        # Compter les pions par colonne
        for file in range(8):
            white_pawns_in_file = 0
            black_pawns_in_file = 0
            
            for rank in range(8):
                square = rank * 8 + file
                if chess.bitboards.get('P', 0) & chess.square_mask(square):
                    white_pawns_in_file += 1
                if chess.bitboards.get('p', 0) & chess.square_mask(square):
                    black_pawns_in_file += 1
            
            # Pénaliser les pions doublés
            if white_pawns_in_file > 1:
                score -= (white_pawns_in_file - 1) * 10
            if black_pawns_in_file > 1:
                score += (black_pawns_in_file - 1) * 10
                
        return score
    
    def _evaluate_piece_coordination(self, chess):
        """Évalue la coordination des pièces"""
        # Bonus simple pour le développement des pièces
        score = 0
        
        # Bonus si les cavaliers ne sont pas sur leur case de départ
        starting_knight_squares_white = [1, 6]  # b1, g1
        starting_knight_squares_black = [57, 62]  # b8, g8
        
        for square in starting_knight_squares_white:
            if not (chess.bitboards.get('N', 0) & chess.square_mask(square)):
                score += 10  # Bonus pour développement
                
        for square in starting_knight_squares_black:
            if not (chess.bitboards.get('n', 0) & chess.square_mask(square)):
                score -= 10
                
        return score
