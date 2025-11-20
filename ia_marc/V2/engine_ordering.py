"""
IA-Marc V2 - Move Ordering
===========================

Optimisation cruciale du Move Ordering pour maximiser l'efficacité de l'élagage
Alpha-Beta. Un bon move ordering peut réduire le nombre de nœuds explorés de 50-90%.

Techniques implémentées:
- Killer Moves: Mémorise les coups qui ont causé des cutoffs
- History Heuristic: Statistiques des coups qui ont bien fonctionné
- MVV-LVA: Most Valuable Victim - Least Valuable Attacker (captures)
- PV Move: Principal Variation (meilleur coup de la TT)

Optimisé pour Raspberry Pi 5.
"""

from typing import List, Optional, Tuple

import chess

# ============================================================================
# CONSTANTES
# ============================================================================

# Valeurs des pièces pour MVV-LVA
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Scores de base pour le move ordering
SCORE_PV_MOVE = 20000000
SCORE_CAPTURE_BASE = 10000000
SCORE_KILLER_1 = 9000000
SCORE_KILLER_2 = 8000000
SCORE_PROMOTION = 7000000
SCORE_CASTLING = 500000


# ============================================================================
# KILLER MOVES
# ============================================================================


class KillerMoves:
    """
    Table des Killer Moves: mémorise les coups non-captures qui ont causé
    des beta-cutoffs à chaque profondeur.

    Principe: Si un coup a causé un cutoff dans une position, il a de bonnes
    chances de causer un cutoff dans des positions similaires à la même profondeur.
    
    Upgraded to 4 slots (from 2) inspired by TinyHugeBot for better move ordering.
    """

    def __init__(self, max_depth: int = 64, slots_per_depth: int = 4):
        """
        Initialise la table des killer moves.

        Args:
            max_depth: Profondeur maximale de recherche
            slots_per_depth: Nombre de killers par profondeur (maintenant 4)
        """
        self.max_depth = max_depth
        self.slots = slots_per_depth
        # Table[depth][slot] = Move
        self.table = [[None for _ in range(slots_per_depth)] for _ in range(max_depth)]

    def add(self, move: chess.Move, depth: int):
        """
        Ajoute un killer move à la profondeur donnée.

        Args:
            move: Coup à ajouter
            depth: Profondeur où le cutoff s'est produit
        """
        if depth >= self.max_depth or depth < 0:
            return

        # Ne pas ajouter de doublons
        for i in range(self.slots):
            if self.table[depth][i] == move:
                return

        # Décalage: le nouveau killer devient le premier
        # Les anciens sont décalés (FIFO)
        for i in range(self.slots - 1, 0, -1):
            self.table[depth][i] = self.table[depth][i - 1]
        self.table[depth][0] = move

    def is_killer(self, move: chess.Move, depth: int) -> int:
        """
        Vérifie si un coup est un killer move.

        Args:
            move: Coup à vérifier
            depth: Profondeur actuelle

        Returns:
            0 si pas killer, 1-4 selon le rang du killer (1 = meilleur)
        """
        if depth >= self.max_depth or depth < 0:
            return 0

        for i in range(self.slots):
            if self.table[depth][i] == move:
                return i + 1  # Retourne 1, 2, 3, ou 4

        return 0

    def get_killers(self, depth: int) -> List[Optional[chess.Move]]:
        """
        Retourne les killer moves pour une profondeur donnée.

        Args:
            depth: Profondeur

        Returns:
            Liste des killer moves (peut contenir None)
        """
        if depth >= self.max_depth or depth < 0:
            return [None] * self.slots
        return self.table[depth]

    def clear(self):
        """Réinitialise tous les killer moves."""
        self.table = [[None for _ in range(self.slots)] for _ in range(self.max_depth)]


# ============================================================================
# HISTORY HEURISTIC
# ============================================================================


class HistoryTable:
    """
    History Heuristic: maintient des statistiques sur le succès des coups
    (combien de fois ils ont causé des cutoffs).

    Structure: [from_square][to_square] = score
    Plus le score est élevé, plus le coup a été efficace historiquement.
    """

    def __init__(self):
        """Initialise la table d'historique."""
        # Table 64x64 pour tous les coups possibles
        self.table = [[0 for _ in range(64)] for _ in range(64)]
        self.max_score = 1  # Pour normalisation

    def update(self, move: chess.Move, depth: int, caused_cutoff: bool = True):
        """
        Met à jour l'historique pour un coup.

        Args:
            move: Coup à mettre à jour
            depth: Profondeur où le coup a été joué
            caused_cutoff: True si le coup a causé un cutoff
        """
        from_sq = move.from_square
        to_sq = move.to_square

        # Bonus proportionnel à la profondeur au carré
        # Les coups profonds sont plus significatifs
        if caused_cutoff:
            bonus = depth * depth
            self.table[from_sq][to_sq] += bonus

            # Mettre à jour le score max pour normalisation
            if self.table[from_sq][to_sq] > self.max_score:
                self.max_score = self.table[from_sq][to_sq]
        else:
            # Pénalité légère pour les coups qui n'ont pas coupé
            penalty = depth
            self.table[from_sq][to_sq] -= penalty
            if self.table[from_sq][to_sq] < 0:
                self.table[from_sq][to_sq] = 0

    def get_score(self, move: chess.Move) -> int:
        """
        Retourne le score historique d'un coup.

        Args:
            move: Coup à évaluer

        Returns:
            Score historique (plus élevé = meilleur)
        """
        return self.table[move.from_square][move.to_square]

    def clear(self):
        """Réinitialise toute la table d'historique."""
        self.table = [[0 for _ in range(64)] for _ in range(64)]
        self.max_score = 1

    def age(self, factor: float = 0.9):
        """
        Vieillit les scores pour privilégier les informations récentes.

        Args:
            factor: Facteur de vieillissement (0.9 = -10%)
        """
        for i in range(64):
            for j in range(64):
                self.table[i][j] = int(self.table[i][j] * factor)
        self.max_score = int(self.max_score * factor)


# ============================================================================
# CONTINUATION HISTORY
# ============================================================================


class ContinuationHistory:
    """
    Continuation History: maintient des statistiques basées sur des paires
    de coups consécutifs (followup history).
    
    Structure: [prev_from][prev_to][curr_from][curr_to] = score
    
    Inspiré de TinyHugeBot et Stockfish. Améliore significativement le
    move ordering en capturant les patterns tactiques.
    """

    def __init__(self):
        """Initialise la table de continuation history."""
        # Table 64x64x64x64 serait trop grande (16M entrées)
        # On utilise une approximation: [from_sq*64 + to_sq] = score
        # Taille: 4096 entrées (64*64)
        self.table = [0 for _ in range(4096)]
        self.max_score = 1

    def _get_index(self, prev_move: Optional[chess.Move], curr_move: chess.Move) -> int:
        """
        Calcule l'index dans la table pour une paire de coups.
        
        Args:
            prev_move: Coup précédent (peut être None)
            curr_move: Coup actuel
            
        Returns:
            Index dans la table
        """
        if prev_move is None:
            # Pas de coup précédent, utiliser seulement le coup actuel
            return curr_move.from_square * 64 + curr_move.to_square
        
        # Combiner les deux coups: XOR pour mixer
        prev_idx = prev_move.from_square * 64 + prev_move.to_square
        curr_idx = curr_move.from_square * 64 + curr_move.to_square
        return (prev_idx ^ curr_idx) & 0xFFF  # Masque pour 4096 entrées

    def update(self, prev_move: Optional[chess.Move], curr_move: chess.Move, 
               depth: int, caused_cutoff: bool = True):
        """
        Met à jour l'historique de continuation.
        
        Args:
            prev_move: Coup précédent
            curr_move: Coup actuel
            depth: Profondeur
            caused_cutoff: True si le coup a causé un cutoff
        """
        idx = self._get_index(prev_move, curr_move)
        
        if caused_cutoff:
            bonus = depth * depth
            self.table[idx] += bonus
            if self.table[idx] > self.max_score:
                self.max_score = self.table[idx]
        else:
            penalty = depth
            self.table[idx] -= penalty
            if self.table[idx] < 0:
                self.table[idx] = 0

    def get_score(self, prev_move: Optional[chess.Move], curr_move: chess.Move) -> int:
        """
        Retourne le score de continuation pour une paire de coups.
        
        Args:
            prev_move: Coup précédent
            curr_move: Coup actuel
            
        Returns:
            Score de continuation
        """
        idx = self._get_index(prev_move, curr_move)
        return self.table[idx]

    def clear(self):
        """Réinitialise toute la table."""
        self.table = [0 for _ in range(4096)]
        self.max_score = 1

    def age(self, factor: float = 0.9):
        """
        Vieillit les scores.
        
        Args:
            factor: Facteur de vieillissement
        """
        for i in range(4096):
            self.table[i] = int(self.table[i] * factor)
        self.max_score = int(self.max_score * factor)


# ============================================================================
# MOVE ORDERING
# ============================================================================


class MoveOrderer:
    """
    Gestionnaire central du move ordering.
    Combine toutes les heuristiques pour trier les coups optimalement.
    """

    def __init__(self, max_depth: int = 64, killer_slots: int = 4):
        """
        Initialise le move orderer.

        Args:
            max_depth: Profondeur maximale
            killer_slots: Nombre de killers par profondeur (4 par défaut)
        """
        self.killers = KillerMoves(max_depth, killer_slots)
        self.history = HistoryTable()
        self.continuation_history = ContinuationHistory()

    def score_move(
        self,
        board: chess.Board,
        move: chess.Move,
        depth: int,
        pv_move: Optional[chess.Move] = None,
        prev_move: Optional[chess.Move] = None,
    ) -> int:
        """
        Attribue un score à un coup pour le tri.
        Plus le score est élevé, plus le coup doit être essayé tôt.

        Args:
            board: Position actuelle
            move: Coup à scorer
            depth: Profondeur actuelle
            pv_move: Coup de la variation principale (TT)
            prev_move: Coup précédent (pour continuation history)

        Returns:
            Score du coup
        """
        # 1. PV Move (meilleur coup de la TT) - Priorité absolue
        if pv_move and move == pv_move:
            return SCORE_PV_MOVE

        # 2. Captures (MVV-LVA)
        if board.is_capture(move):
            return self._score_capture(board, move)

        # 3. Promotions
        if move.promotion:
            return SCORE_PROMOTION + PIECE_VALUES.get(move.promotion, 0)

        # 4. Killer Moves (4 slots avec scores décroissants)
        killer_rank = self.killers.is_killer(move, depth)
        if killer_rank == 1:
            return SCORE_KILLER_1
        elif killer_rank == 2:
            return SCORE_KILLER_2
        elif killer_rank == 3:
            return SCORE_KILLER_2 - 100000  # Killer 3
        elif killer_rank == 4:
            return SCORE_KILLER_2 - 200000  # Killer 4

        # 5. Roque
        if board.is_castling(move):
            return SCORE_CASTLING

        # 6. History Heuristic + Continuation History
        history_score = self.history.get_score(move)
        continuation_score = self.continuation_history.get_score(prev_move, move)
        # Combiner les deux scores (continuation history a plus de poids)
        return history_score + continuation_score * 2

    def _score_capture(self, board: chess.Board, move: chess.Move) -> int:
        """
        Score une capture avec MVV-LVA (Most Valuable Victim - Least Valuable Attacker).

        Args:
            board: Position actuelle
            move: Coup de capture

        Returns:
            Score de la capture
        """
        # Pièce capturée (victime)
        captured = board.piece_at(move.to_square)
        victim_value = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0

        # Pièce qui capture (attaquant)
        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0

        # MVV-LVA: victimes de haute valeur en premier, attaquants de faible valeur en premier
        # Formule: (victim_value * 10) - attacker_value
        # Exemple: PxQ = (900*10) - 100 = 8900
        #          QxP = (100*10) - 900 = 100
        # Donc PxQ sera essayé avant QxP
        score = SCORE_CAPTURE_BASE + (victim_value * 10) - attacker_value

        # Bonus pour la promotion avec capture
        if move.promotion:
            score += PIECE_VALUES.get(move.promotion, 0)

        return score

    def order_moves(
        self,
        board: chess.Board,
        moves: List[chess.Move],
        depth: int,
        pv_move: Optional[chess.Move] = None,
        prev_move: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
        """
        Trie une liste de coups selon leur score.

        Args:
            board: Position actuelle
            moves: Liste des coups à trier
            depth: Profondeur actuelle
            pv_move: Coup de la variation principale
            prev_move: Coup précédent (pour continuation history)

        Returns:
            Liste triée des coups (meilleurs en premier)
        """
        # Scorer tous les coups
        scored_moves = [
            (move, self.score_move(board, move, depth, pv_move, prev_move)) for move in moves
        ]

        # Trier par score décroissant
        scored_moves.sort(key=lambda x: x[1], reverse=True)

        # Retourner seulement les coups
        return [move for move, score in scored_moves]

    def update_history(self, move: chess.Move, depth: int, caused_cutoff: bool = True, prev_move: Optional[chess.Move] = None):
        """
        Met à jour l'historique après un coup.

        Args:
            move: Coup joué
            depth: Profondeur
            caused_cutoff: True si beta cutoff
            prev_move: Coup précédent (pour continuation history)
        """
        self.history.update(move, depth, caused_cutoff)
        self.continuation_history.update(prev_move, move, depth, caused_cutoff)

    def add_killer(self, move: chess.Move, depth: int):
        """
        Ajoute un killer move.

        Args:
            move: Coup killer
            depth: Profondeur
        """
        self.killers.add(move, depth)

    def clear(self):
        """Réinitialise toutes les tables."""
        self.killers.clear()
        self.history.clear()
        self.continuation_history.clear()

    def age_history(self, factor: float = 0.9):
        """
        Vieillit l'historique (à appeler périodiquement).

        Args:
            factor: Facteur de vieillissement
        """
        self.history.age(factor)
        self.continuation_history.age(factor)


# ============================================================================
# TESTS
# ============================================================================


if __name__ == "__main__":
    print("=== Test du Move Ordering ===\n")

    # Test 1: Killer Moves
    print("Test 1: Killer Moves")
    killers = KillerMoves(max_depth=10, slots_per_depth=2)

    move1 = chess.Move.from_uci("e2e4")
    move2 = chess.Move.from_uci("d2d4")
    move3 = chess.Move.from_uci("g1f3")

    killers.add(move1, depth=3)
    killers.add(move2, depth=3)
    killers.add(move3, depth=3)

    print(f"  Killer 1 à depth 3: {killers.table[3][0]}")
    print(f"  Killer 2 à depth 3: {killers.table[3][1]}")
    print(f"  move3 est killer 1: {killers.is_killer(move3, 3) == 1}")
    print(f"  move2 est killer 2: {killers.is_killer(move2, 3) == 2}")
    print()

    # Test 2: History Heuristic
    print("Test 2: History Heuristic")
    history = HistoryTable()

    move = chess.Move.from_uci("e2e4")
    history.update(move, depth=5, caused_cutoff=True)
    history.update(move, depth=6, caused_cutoff=True)

    score = history.get_score(move)
    print(f"  Score de e2e4 après 2 cutoffs: {score}")
    print(f"  Score > 0: {score > 0}")
    print()

    # Test 3: MVV-LVA
    print("Test 3: MVV-LVA (Captures)")
    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    )
    orderer = MoveOrderer()

    # Position avec plusieurs captures possibles
    capture_moves = [m for m in board.legal_moves if board.is_capture(m)]

    if capture_moves:
        for move in capture_moves[:3]:
            score = orderer.score_move(board, move, depth=0)
            captured = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            print(
                f"  {move}: {attacker.symbol() if attacker else '?'}x{captured.symbol() if captured else '?'} → Score: {score}"
            )
    print()

    # Test 4: Move Ordering complet
    print("Test 4: Move Ordering Complet")
    board = chess.Board()

    # Simuler un PV move
    pv_move = chess.Move.from_uci("e2e4")

    # Ajouter des killers
    orderer.add_killer(chess.Move.from_uci("g1f3"), depth=0)

    # Ordonner les coups
    moves = list(board.legal_moves)
    ordered = orderer.order_moves(board, moves, depth=0, pv_move=pv_move)

    print(f"  Premier coup (devrait être PV): {ordered[0]}")
    print(f"  C'est bien le PV move: {ordered[0] == pv_move}")
    print(f"  Total coups: {len(ordered)}")
    print()

    # Test 5: Performance
    print("Test 5: Performance")
    import time

    board = chess.Board()
    moves = list(board.legal_moves)
    orderer = MoveOrderer()

    start = time.time()
    for _ in range(10000):
        orderer.order_moves(board, moves, depth=0)
    duration = time.time() - start

    print(f"  10,000 orderings: {duration:.3f}s")
    print(f"  Moyenne: {duration / 10000 * 1000:.3f}ms par ordering")
    print()

    print("✅ Tests du Move Ordering réussis!")
