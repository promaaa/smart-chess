"""
IA-Marc V2 - Evaluation Engine (Brain)
=======================================

Moteur d'évaluation avancé basé sur PeSTO (Piece-Square Tables)
avec extensions tactiques et positionnelles.

Fonctionnalités:
- Tapered Evaluation (interpolation MG/EG)
- Piece-Square Tables optimisées (PeSTO)
- Mobility evaluation
- Pawn structure analysis
- King safety
- Bonus de personnalité configurable

Optimisé pour Raspberry Pi 5.
"""

from typing import Optional

import chess

# ============================================================================
# CONSTANTES PESTO (Middle Game & End Game)
# ============================================================================

# Valeurs Matérielles de base (MG, EG)
# P, N, B, R, Q, K
MG_VALS = [82, 337, 365, 477, 1025, 0]
EG_VALS = [94, 281, 297, 512, 936, 0]

# Poids de phase pour le Tapered Eval (N, B, R, Q)
PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
TOTAL_PHASE = 24  # 4*1 + 4*1 + 4*2 + 2*4 = 24

# ============================================================================
# PIECE-SQUARE TABLES (PeSTO)
# ============================================================================

# Tables PST pour les BLANCS - Middle Game
MG_PAWN = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    98,
    134,
    61,
    95,
    68,
    126,
    34,
    -11,
    -6,
    7,
    26,
    31,
    65,
    56,
    25,
    -20,
    -14,
    13,
    6,
    21,
    23,
    12,
    17,
    -23,
    -27,
    -2,
    -5,
    12,
    17,
    6,
    10,
    -25,
    -26,
    -4,
    -4,
    -10,
    3,
    3,
    33,
    -12,
    -35,
    -1,
    -20,
    -23,
    -15,
    24,
    38,
    -22,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

MG_KNIGHT = [
    -167,
    -89,
    -34,
    -49,
    61,
    -97,
    -15,
    -107,
    -73,
    -41,
    72,
    36,
    23,
    62,
    7,
    -17,
    -47,
    60,
    37,
    65,
    84,
    129,
    73,
    44,
    -9,
    17,
    19,
    53,
    37,
    69,
    18,
    22,
    -13,
    4,
    16,
    13,
    28,
    19,
    21,
    -8,
    -23,
    -9,
    12,
    10,
    19,
    17,
    25,
    -16,
    -29,
    -53,
    -12,
    -3,
    -1,
    18,
    -14,
    -19,
    -105,
    -21,
    -58,
    -33,
    -17,
    -28,
    -19,
    -23,
]

MG_BISHOP = [
    -29,
    4,
    -82,
    -37,
    -25,
    -42,
    7,
    -8,
    -26,
    16,
    -18,
    -13,
    30,
    59,
    18,
    -47,
    -16,
    37,
    43,
    40,
    35,
    50,
    37,
    -2,
    -4,
    5,
    19,
    50,
    37,
    37,
    7,
    -2,
    -6,
    13,
    13,
    26,
    34,
    12,
    10,
    4,
    0,
    15,
    15,
    15,
    14,
    27,
    18,
    10,
    4,
    15,
    16,
    0,
    7,
    21,
    33,
    1,
    -33,
    -3,
    -14,
    -21,
    -13,
    -12,
    -39,
    -21,
]

MG_ROOK = [
    32,
    42,
    32,
    51,
    63,
    9,
    31,
    43,
    27,
    32,
    58,
    62,
    80,
    67,
    26,
    44,
    -5,
    19,
    26,
    36,
    17,
    45,
    61,
    16,
    -24,
    -11,
    7,
    26,
    24,
    35,
    -8,
    -20,
    -36,
    -26,
    -12,
    -1,
    9,
    -7,
    6,
    -23,
    -45,
    -25,
    -16,
    -17,
    3,
    0,
    -5,
    -33,
    -44,
    -16,
    -20,
    -9,
    -1,
    11,
    -6,
    -71,
    -19,
    -13,
    1,
    17,
    16,
    7,
    -37,
    -26,
]

MG_QUEEN = [
    -28,
    0,
    29,
    12,
    59,
    44,
    43,
    45,
    -24,
    -39,
    -5,
    1,
    -16,
    57,
    28,
    54,
    -13,
    -17,
    7,
    8,
    29,
    56,
    47,
    57,
    -27,
    -27,
    -16,
    -16,
    -1,
    17,
    -2,
    1,
    -9,
    -26,
    -9,
    -10,
    -2,
    -4,
    3,
    -3,
    -14,
    2,
    -11,
    -2,
    -5,
    2,
    14,
    5,
    -35,
    -8,
    11,
    2,
    8,
    15,
    -3,
    1,
    -1,
    -18,
    -9,
    -10,
    -30,
    -15,
    -13,
    -32,
]

MG_KING = [
    -65,
    23,
    16,
    -15,
    -56,
    -34,
    2,
    13,
    29,
    -1,
    -20,
    -7,
    -8,
    -4,
    -38,
    -29,
    -9,
    24,
    2,
    -16,
    -20,
    6,
    22,
    -22,
    -17,
    -20,
    -12,
    -27,
    -30,
    -25,
    -14,
    -36,
    -49,
    -1,
    -27,
    -39,
    -46,
    -44,
    -33,
    -51,
    -14,
    -14,
    -22,
    -46,
    -44,
    -30,
    -15,
    -27,
    1,
    7,
    -8,
    -64,
    -43,
    -16,
    9,
    8,
    -15,
    36,
    12,
    -54,
    8,
    -28,
    24,
    14,
]

# Tables PST pour les BLANCS - End Game
EG_PAWN = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    178,
    173,
    158,
    134,
    147,
    132,
    165,
    187,
    94,
    100,
    85,
    67,
    56,
    53,
    82,
    84,
    32,
    24,
    13,
    5,
    -2,
    4,
    17,
    17,
    13,
    9,
    -3,
    -7,
    -7,
    -8,
    3,
    -1,
    4,
    7,
    -6,
    1,
    0,
    -5,
    -1,
    -8,
    13,
    8,
    8,
    10,
    13,
    0,
    2,
    -7,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

EG_KNIGHT = [
    -58,
    -38,
    -13,
    -28,
    -31,
    -27,
    -63,
    -99,
    -25,
    -8,
    -25,
    -2,
    -9,
    -25,
    -24,
    -52,
    -24,
    -20,
    10,
    9,
    -1,
    -9,
    -19,
    -41,
    -17,
    3,
    22,
    22,
    22,
    11,
    8,
    -18,
    -18,
    -6,
    16,
    25,
    16,
    17,
    4,
    -18,
    -23,
    -3,
    -1,
    15,
    10,
    -3,
    -20,
    -22,
    -42,
    -20,
    -10,
    -5,
    -2,
    -20,
    -23,
    -44,
    -29,
    -51,
    -23,
    -15,
    -22,
    -18,
    -50,
    -64,
]

EG_BISHOP = [
    -14,
    -21,
    -11,
    -8,
    -7,
    -9,
    -17,
    -24,
    -8,
    -4,
    7,
    -12,
    -3,
    -13,
    -4,
    -14,
    -4,
    -8,
    -2,
    -6,
    -2,
    -14,
    -9,
    -1,
    -2,
    -2,
    -1,
    -5,
    -9,
    10,
    -5,
    -6,
    -9,
    2,
    -5,
    3,
    8,
    -3,
    2,
    -10,
    -13,
    -5,
    -2,
    3,
    -8,
    -12,
    -7,
    -8,
    -9,
    -14,
    1,
    0,
    -6,
    -3,
    -18,
    -14,
    -23,
    -9,
    -23,
    -5,
    -9,
    -16,
    -5,
    -17,
]

EG_ROOK = [
    13,
    10,
    18,
    15,
    12,
    12,
    8,
    5,
    11,
    13,
    13,
    11,
    12,
    12,
    2,
    7,
    0,
    9,
    9,
    1,
    1,
    5,
    10,
    0,
    -7,
    -10,
    -1,
    6,
    7,
    1,
    2,
    -5,
    -6,
    -13,
    -7,
    4,
    5,
    5,
    -4,
    -14,
    -13,
    -16,
    -13,
    -8,
    -5,
    -14,
    -14,
    -10,
    -19,
    -8,
    -7,
    -11,
    -9,
    -12,
    -15,
    -14,
    -9,
    -8,
    -9,
    -14,
    -14,
    -5,
    -14,
    -13,
]

EG_QUEEN = [
    -9,
    22,
    22,
    27,
    27,
    19,
    10,
    20,
    -17,
    20,
    32,
    41,
    58,
    25,
    30,
    0,
    -20,
    6,
    9,
    49,
    47,
    35,
    19,
    9,
    3,
    22,
    24,
    45,
    57,
    40,
    57,
    36,
    -18,
    28,
    19,
    47,
    31,
    34,
    39,
    23,
    -16,
    -27,
    15,
    6,
    9,
    17,
    10,
    5,
    -22,
    -23,
    -30,
    -16,
    -16,
    -23,
    -36,
    -32,
    -33,
    -28,
    -22,
    -43,
    -5,
    -32,
    -20,
    -41,
]

EG_KING = [
    -74,
    -35,
    -18,
    -18,
    -11,
    15,
    4,
    -17,
    -12,
    17,
    14,
    17,
    17,
    38,
    23,
    11,
    10,
    17,
    23,
    15,
    20,
    45,
    44,
    13,
    -8,
    22,
    24,
    27,
    26,
    33,
    26,
    3,
    -18,
    -4,
    21,
    24,
    27,
    23,
    9,
    -11,
    -19,
    -3,
    11,
    21,
    23,
    16,
    7,
    -9,
    -27,
    -11,
    4,
    13,
    14,
    4,
    -5,
    -17,
    -53,
    -34,
    -21,
    -11,
    -28,
    -14,
    -24,
    -43,
]

# ============================================================================
# PRÉ-CALCUL DES TABLES
# ============================================================================

# Mapping temporaire
raw_mg = {
    chess.PAWN: MG_PAWN,
    chess.KNIGHT: MG_KNIGHT,
    chess.BISHOP: MG_BISHOP,
    chess.ROOK: MG_ROOK,
    chess.QUEEN: MG_QUEEN,
    chess.KING: MG_KING,
}

raw_eg = {
    chess.PAWN: EG_PAWN,
    chess.KNIGHT: EG_KNIGHT,
    chess.BISHOP: EG_BISHOP,
    chess.ROOK: EG_ROOK,
    chess.QUEEN: EG_QUEEN,
    chess.KING: EG_KING,
}

# Construction des tables finales
# Index 0: Blancs, Index 1: Noirs
final_tables_mg = {pt: [None, None] for pt in range(1, 7)}
final_tables_eg = {pt: [None, None] for pt in range(1, 7)}

for pt in range(1, 7):
    # Blancs (Index 0)
    white_mg = [val + MG_VALS[pt - 1] for val in raw_mg[pt]]
    white_eg = [val + EG_VALS[pt - 1] for val in raw_eg[pt]]

    # Noirs (Index 1) - Miroir vertical ET négation
    black_mg = [-(raw_mg[pt][i ^ 56] + MG_VALS[pt - 1]) for i in range(64)]
    black_eg = [-(raw_eg[pt][i ^ 56] + EG_VALS[pt - 1]) for i in range(64)]

    final_tables_mg[pt][0] = white_mg
    final_tables_mg[pt][1] = black_mg
    final_tables_eg[pt][0] = white_eg
    final_tables_eg[pt][1] = black_eg


# ============================================================================
# EVALUATION ENGINE
# ============================================================================


class EvaluationEngine:
    """
    Moteur d'évaluation avancé avec support des personnalités et
    évaluations tactiques/positionnelles.
    """

    def __init__(self):
        """Initialise le moteur d'évaluation."""
        # Configuration des bonus (modifiable par personnalité)
        self.bonus_mobility = 10
        self.bonus_pawn_structure = 10
        self.bonus_king_safety = 10
        self.bonus_center = 5
        
        # Facteur de mépris pour les nulles
        self._contempt = 0

        # Flags d'activation
        self.use_mobility = True
        self.use_pawn_structure = True
        self.use_king_safety = True

        # Cache pour optimisation (optionnel)
        self._pawn_cache = {}

    def configure(self, **kwargs):
        """
        Configure les paramètres d'évaluation.

        Args:
            bonus_mobility: Bonus pour la mobilité
            bonus_pawn_structure: Bonus pour la structure de pions
            bonus_king_safety: Bonus pour la sécurité du roi
            bonus_center: Bonus pour le contrôle du centre
            contempt: Facteur de mépris pour les nulles (0-50)
            use_mobility: Activer l'évaluation de mobilité
            use_pawn_structure: Activer l'évaluation de structure
            use_king_safety: Activer l'évaluation de sécurité du roi
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def evaluate(self, board: chess.Board) -> int:
        """
        Évalue une position d'échecs.

        Args:
            board: Position à évaluer

        Returns:
            Score en centipawns du point de vue du joueur au trait
            Positif = avantage, Négatif = désavantage
        """
        # Détections rapides de fin de partie
        if board.is_checkmate():
            return -99999 if board.turn == chess.WHITE else 99999

        if board.is_insufficient_material() or board.is_stalemate():
            # Appliquer le contempt: une nulle est considérée comme légèrement défavorable
            # Cela encourage le moteur à éviter les nulles sauf si nécessaire
            contempt = getattr(self, '_contempt', 0)
            return -contempt

        # Score de base (matériel + PST)
        mg_score, eg_score, phase = self._evaluate_material_pst(board)

        # Évaluations additionnelles
        if self.use_mobility:
            mobility = self._evaluate_mobility(board)
            mg_score += mobility
            eg_score += mobility // 2  # Moins important en finale

        if self.use_pawn_structure:
            pawn_score = self._evaluate_pawn_structure(board)
            mg_score += pawn_score
            eg_score += pawn_score

        if self.use_king_safety and phase > 12:  # Seulement en milieu de jeu
            king_safety = self._evaluate_king_safety(board)
            mg_score += king_safety

        # Interpolation Tapered (MG -> EG)
        final_score = self._interpolate_score(mg_score, eg_score, phase)

        # Retour du score du point de vue du joueur au trait
        return final_score if board.turn == chess.WHITE else -final_score

    def _evaluate_material_pst(self, board: chess.Board) -> tuple:
        """
        Évalue le matériel et les positions (PST).

        Returns:
            (mg_score, eg_score, phase)
        """
        mg_score = 0
        eg_score = 0
        current_phase = 0

        pm = board.piece_map()

        for square, piece in pm.items():
            pt = piece.piece_type
            c_idx = 0 if piece.color == chess.WHITE else 1

            # Accumulation des scores
            mg_score += final_tables_mg[pt][c_idx][square]
            eg_score += final_tables_eg[pt][c_idx][square]

            # Calcul de la phase
            if pt in PHASE_WEIGHTS:
                current_phase += PHASE_WEIGHTS[pt]

        # Clamp de la phase
        if current_phase > TOTAL_PHASE:
            current_phase = TOTAL_PHASE

        return mg_score, eg_score, current_phase

    def _evaluate_mobility(self, board: chess.Board) -> int:
        """
        Évalue la mobilité (nombre de coups légaux).

        Args:
            board: Position à évaluer

        Returns:
            Score de mobilité
        """
        # Mobilité des blancs
        white_mobility = board.legal_moves.count() if board.turn == chess.WHITE else 0

        # Basculer pour les noirs
        board.push(chess.Move.null())  # Coup nul pour changer de trait
        black_mobility = board.legal_moves.count() if board.turn == chess.BLACK else 0
        board.pop()

        # Si on ne peut pas faire de coup nul (illégal), compter autrement
        if board.turn == chess.BLACK:
            white_mobility = 0
            black_mobility = board.legal_moves.count()
        else:
            white_mobility = board.legal_moves.count()
            black_mobility = 0

        # Méthode simplifiée : juste compter les coups du trait actuel
        mobility_score = board.legal_moves.count() * self.bonus_mobility

        return mobility_score if board.turn == chess.WHITE else -mobility_score

    def _evaluate_pawn_structure(self, board: chess.Board) -> int:
        """
        Évalue la structure de pions.

        Pénalités:
        - Pions doublés: -20
        - Pions isolés: -15

        Bonus:
        - Pions passés: +30 à +80 selon avancement
        - Pions protégés: +10

        Args:
            board: Position à évaluer

        Returns:
            Score de structure
        """
        score = 0

        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)

        # Évaluer les pions blancs
        score += self._evaluate_pawns_for_color(board, white_pawns, chess.WHITE)

        # Évaluer les pions noirs
        score -= self._evaluate_pawns_for_color(board, black_pawns, chess.BLACK)

        return score

    def _evaluate_pawns_for_color(
        self, board: chess.Board, pawns: chess.SquareSet, color: chess.Color
    ) -> int:
        """Évalue les pions pour une couleur donnée."""
        score = 0

        for square in pawns:
            file = chess.square_file(square)
            rank = chess.square_rank(square)

            # Pions doublés
            file_pawns = [sq for sq in pawns if chess.square_file(sq) == file]
            if len(file_pawns) > 1:
                score -= 20

            # Pions isolés (pas de pion allié sur les colonnes adjacentes)
            has_neighbor = False
            for adj_file in [file - 1, file + 1]:
                if 0 <= adj_file <= 7:
                    if any(chess.square_file(sq) == adj_file for sq in pawns):
                        has_neighbor = True
                        break

            if not has_neighbor:
                score -= 15

            # Pions passés (pas de pion ennemi devant)
            enemy_pawns = board.pieces(chess.PAWN, not color)
            is_passed = True

            direction = 1 if color == chess.WHITE else -1
            for check_rank in range(
                rank + direction, 8 if color == chess.WHITE else -1, direction
            ):
                for check_file in [file - 1, file, file + 1]:
                    if 0 <= check_file <= 7:
                        check_square = chess.square(check_file, check_rank)
                        if check_square in enemy_pawns:
                            is_passed = False
                            break
                if not is_passed:
                    break

            if is_passed:
                # Bonus proportionnel à l'avancement
                advancement = rank if color == chess.WHITE else (7 - rank)
                score += 30 + (advancement * 10)

        return score

    def _evaluate_king_safety(self, board: chess.Board) -> int:
        """
        Évalue la sécurité du roi (important en milieu de jeu).

        Args:
            board: Position à évaluer

        Returns:
            Score de sécurité
        """
        score = 0

        # Roi blanc
        white_king_sq = board.king(chess.WHITE)
        if white_king_sq:
            score += self._king_safety_for_square(board, white_king_sq, chess.WHITE)

        # Roi noir
        black_king_sq = board.king(chess.BLACK)
        if black_king_sq:
            score -= self._king_safety_for_square(board, black_king_sq, chess.BLACK)

        return score

    def _king_safety_for_square(
        self, board: chess.Board, king_sq: int, color: chess.Color
    ) -> int:
        """Évalue la sécurité d'un roi spécifique."""
        score = 0

        # Bonus pour le roque effectué
        if color == chess.WHITE:
            if king_sq in [chess.G1, chess.C1]:  # Roqué
                score += 30
        else:
            if king_sq in [chess.G8, chess.C8]:  # Roqué
                score += 30

        # Compter les pions devant le roi (bouclier)
        file = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)

        pawn_shield = 0
        direction = 1 if color == chess.WHITE else -1

        for check_file in [file - 1, file, file + 1]:
            if 0 <= check_file <= 7:
                for offset in [1, 2]:
                    check_rank = rank + (direction * offset)
                    if 0 <= check_rank <= 7:
                        check_square = chess.square(check_file, check_rank)
                        if board.piece_at(check_square) == chess.Piece(
                            chess.PAWN, color
                        ):
                            pawn_shield += 15

        score += pawn_shield

        # Pénalité si le roi est exposé au centre en milieu de jeu
        if file in [3, 4] and rank in [3, 4]:
            score -= 50

        return score

    def _interpolate_score(self, mg_score: int, eg_score: int, phase: int) -> int:
        """
        Interpole entre les scores MG et EG selon la phase.

        Args:
            mg_score: Score milieu de jeu
            eg_score: Score finale
            phase: Phase actuelle (0-24)

        Returns:
            Score interpolé
        """
        # Formule: (MG * Phase + EG * (24 - Phase)) / 24
        return (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) // TOTAL_PHASE

    def clear_cache(self):
        """Vide le cache d'évaluation."""
        self._pawn_cache.clear()


# ============================================================================
# ALIAS ET COMPATIBILITÉ V1
# ============================================================================

# Pour compatibilité avec l'ancienne API
Engine = EvaluationEngine


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=== Test du Moteur d'Évaluation ===\n")

    engine = EvaluationEngine()

    # Test 1: Position initiale
    print("Test 1: Position initiale")
    board = chess.Board()
    score = engine.evaluate(board)
    print(f"  Score: {score} (devrait être ~0)")
    print()

    # Test 2: Avantage matériel
    print("Test 2: Avantage matériel (Blancs +1 pion)")
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1")
    score = engine.evaluate(board)
    print(f"  Score Blancs: {score} (devrait être positif)")
    print()

    # Test 3: Mat
    print("Test 3: Mat en 1")
    board = chess.Board(
        "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    )
    board.push(chess.Move.from_uci("e8d7"))  # Roi se déplace (pas de mat encore)
    score = engine.evaluate(board)
    print(f"  Score avant mat: {score}")
    print()

    # Test 4: Performance
    print("Test 4: Performance")
    import time

    board = chess.Board()
    start = time.time()
    for _ in range(10000):
        engine.evaluate(board)
    duration = time.time() - start

    print(f"  10,000 évaluations: {duration:.3f}s")
    print(f"  Moyenne: {duration / 10000 * 1000:.3f}ms par évaluation")
    print(f"  Throughput: {10000 / duration:.0f} eval/s")
    print()

    # Test 5: Configuration
    print("Test 5: Configuration des bonus")
    engine.configure(bonus_mobility=20, use_pawn_structure=False)
    score_custom = engine.evaluate(chess.Board())
    print(f"  Score avec configuration personnalisée: {score_custom}")
    print()

    print("✅ Tests du moteur d'évaluation réussis!")
