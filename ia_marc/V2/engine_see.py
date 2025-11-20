"""
IA-Marc V2 - Static Exchange Evaluation (SEE)
==============================================

Évalue statiquement la valeur nette d'un échange de pièces.
Utilisé pour le pruning des captures négatives en Q-search.

Inspiré de TinyHugeBot et Stockfish.
Optimisé pour Raspberry Pi 5.
"""

import chess
from typing import Optional

# Valeurs des pièces pour SEE
SEE_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def get_smallest_attacker(board: chess.Board, square: chess.Square, color: chess.Color) -> Optional[chess.PieceType]:
    """
    Trouve le plus petit attaquant d'une couleur donnée sur une case.
    
    Args:
        board: Position actuelle
        square: Case attaquée
        color: Couleur de l'attaquant
        
    Returns:
        Type de pièce du plus petit attaquant, ou None si aucun
    """
    # Ordre de recherche : Pion, Cavalier, Fou, Tour, Dame, Roi
    piece_order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    
    for piece_type in piece_order:
        attackers = board.attackers(color, square) & board.pieces(piece_type, color)
        if attackers:
            # Retourner le premier attaquant de ce type
            return piece_type
    
    return None


def see(board: chess.Board, move: chess.Move, threshold: int = 0) -> bool:
    """
    Static Exchange Evaluation : évalue si un échange est bénéfique.
    
    Algorithme :
    1. Simuler l'échange pièce par pièce
    2. Calculer la valeur nette
    3. Retourner True si >= threshold
    
    Args:
        board: Position actuelle
        move: Coup à évaluer (doit être une capture)
        threshold: Seuil minimum (0 = au moins égal)
        
    Returns:
        True si l'échange est >= threshold, False sinon
    """
    # Si ce n'est pas une capture, retourner True (pas de perte)
    if not board.is_capture(move):
        return True
    
    # Valeur de la pièce capturée
    captured_piece = board.piece_at(move.to_square)
    if captured_piece is None:
        # En passant ou erreur
        if board.is_en_passant(move):
            gain = [SEE_PIECE_VALUES[chess.PAWN]]
        else:
            return True
    else:
        gain = [SEE_PIECE_VALUES[captured_piece.piece_type]]
    
    # Pièce qui attaque
    attacker_piece = board.piece_at(move.from_square)
    if attacker_piece is None:
        return True
    
    attacker_value = SEE_PIECE_VALUES[attacker_piece.piece_type]
    
    # Faire le coup virtuellement
    board_copy = board.copy()
    board_copy.push(move)
    
    # Simuler les échanges successifs
    target_square = move.to_square
    side_to_move = board_copy.turn  # Côté qui doit répondre
    current_attacker_value = attacker_value
    
    # Maximum 32 échanges (impossible en pratique)
    for _ in range(32):
        # Trouver le plus petit attaquant du côté au trait
        smallest_attacker = get_smallest_attacker(board_copy, target_square, side_to_move)
        
        if smallest_attacker is None:
            # Plus d'attaquants, l'échange s'arrête
            break
        
        # Valeur capturée = pièce attaquante précédente
        gain.append(current_attacker_value)
        current_attacker_value = SEE_PIECE_VALUES[smallest_attacker]
        
        # "Retirer" l'attaquant (sans vraiment jouer le coup pour la performance)
        # On suppose qu'il capture et change de côté
        side_to_move = not side_to_move
    
    # Calculer la valeur nette avec negamax
    # On part de la fin et on remonte
    while len(gain) > 1:
        # Le joueur choisit le meilleur : max(stand_pat, gain[-1] - gain[-2])
        # Si l'échange final est négatif, ne pas l'accepter
        gain[-2] = max(0, gain[-1] - gain[-2])
        gain.pop()
    
    # Comparer avec le seuil
    return gain[0] >= threshold


def see_capture_value(board: chess.Board, move: chess.Move) -> int:
    """
    Retourne la valeur estimée d'une capture via SEE.
    
    Args:
        board: Position actuelle
        move: Coup de capture
        
    Returns:
        Valeur estimée de la capture (peut être négative)
    """
    if not board.is_capture(move):
        return 0
    
    # Version simplifiée : juste la valeur de la victime
    # Pour une SEE complète, il faudrait simuler tous les échanges
    captured = board.piece_at(move.to_square)
    if captured:
        return SEE_PIECE_VALUES[captured.piece_type]
    elif board.is_en_passant(move):
        return SEE_PIECE_VALUES[chess.PAWN]
    
    return 0


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=== Test SEE (Static Exchange Evaluation) ===\n")
    
    # Test 1 : Capture simple
    print("Test 1: Capture simple (PxP)")
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("e4d5")  # Pion prend pion
    result = see(board, move)
    print(f"  e4xd5: {result} (attendu: True)")
    print()
    
    # Test 2 : Échange défavorable
    print("Test 2: Échange défavorable (PxQ mais défendu par P)")
    board = chess.Board("4k3/3p4/8/3Q4/4P3/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("e4d5")  # Pion prend Dame (défendue)
    result = see(board, move)
    print(f"  e4xd5: {result}")
    print(f"  (Gagne Dame=900 mais perd Pion=100, net=+800, donc True)")
    print()
    
    # Test 3 : Capture avec threshold
    print("Test 3: Capture avec threshold")
    board = chess.Board("4k3/8/8/3n4/4N3/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("e4d5")  # Cavalier prend Cavalier
    result_0 = see(board, move, threshold=0)
    result_100 = see(board, move, threshold=100)
    print(f"  e4xd5 (threshold=0): {result_0} (attendu: True, échange égal)")
    print(f"  e4xd5 (threshold=100): {result_100} (attendu: False, pas assez bon)")
    print()
    
    print("✅ Tests SEE terminés!")
