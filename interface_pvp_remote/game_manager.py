#!/usr/bin/env python3
"""
Interface PvP Remote - Game Manager
Gestion de l'état de la partie d'échecs.
"""

import chess
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum


class PlayerType(Enum):
    """Type de joueur."""
    WEB = "web"
    PHYSICAL = "physical"


class GameManager:
    """Gestionnaire de l'état de la partie."""
    
    def __init__(self):
        self.board = chess.Board()
        self.move_history: List[Dict[str, Any]] = []
        
        # Configuration des joueurs
        self.web_color = chess.WHITE  # Le joueur web joue les blancs
        self.physical_color = chess.BLACK  # Le joueur physique joue les noirs
        
        # État
        self.waiting_for_physical_confirmation = False
        self.pending_move: Optional[chess.Move] = None
    
    def reset(self):
        """Réinitialise la partie."""
        self.board = chess.Board()
        self.move_history = []
        self.waiting_for_physical_confirmation = False
        self.pending_move = None
    
    def get_current_player(self) -> PlayerType:
        """Retourne le type de joueur dont c'est le tour."""
        if self.board.turn == self.web_color:
            return PlayerType.WEB
        return PlayerType.PHYSICAL
    
    def get_fen(self) -> str:
        """Retourne le FEN actuel."""
        return self.board.fen()
    
    def get_legal_moves(self) -> List[str]:
        """Retourne la liste des coups légaux en UCI."""
        return [move.uci() for move in self.board.legal_moves]
    
    def is_legal_move(self, from_sq: str, to_sq: str, promotion: str = None) -> bool:
        """Vérifie si un coup est légal."""
        try:
            uci = from_sq + to_sq
            if promotion:
                uci += promotion
            move = chess.Move.from_uci(uci)
            return move in self.board.legal_moves
        except:
            return False
    
    def make_move(self, from_sq: str, to_sq: str, promotion: str = None) -> Optional[Dict[str, Any]]:
        """
        Effectue un coup et retourne les informations du coup.
        Retourne None si le coup est illégal.
        """
        try:
            uci = from_sq + to_sq
            if promotion:
                uci += promotion
            
            move = chess.Move.from_uci(uci)
            
            if move not in self.board.legal_moves:
                # Vérifier si c'est une promotion manquante
                for legal in self.board.legal_moves:
                    if legal.from_square == move.from_square and legal.to_square == move.to_square:
                        if legal.promotion:
                            move = legal  # Utiliser la promotion par défaut (dame)
                            break
                else:
                    return None
            
            # Informations avant le coup
            san = self.board.san(move)
            is_capture = self.board.is_capture(move)
            is_check = False
            is_checkmate = False
            is_castling = self.board.is_castling(move)
            
            # Jouer le coup
            self.board.push(move)
            
            # Informations après le coup
            is_check = self.board.is_check()
            is_checkmate = self.board.is_checkmate()
            is_stalemate = self.board.is_stalemate()
            is_game_over = self.board.is_game_over()
            
            move_info = {
                "from": from_sq,
                "to": to_sq,
                "san": san,
                "uci": move.uci(),
                "promotion": chess.piece_symbol(move.promotion).lower() if move.promotion else None,
                "is_capture": is_capture,
                "is_check": is_check,
                "is_checkmate": is_checkmate,
                "is_castling": is_castling,
                "is_stalemate": is_stalemate,
                "is_game_over": is_game_over,
                "fen": self.board.fen(),
                "move_number": len(self.move_history) + 1
            }
            
            self.move_history.append(move_info)
            
            return move_info
            
        except Exception as e:
            print(f"[GameManager] Erreur make_move: {e}")
            return None
    
    def detect_move_from_state_change(
        self, 
        lifted: List[Tuple[int, int]], 
        placed: List[Tuple[int, int]]
    ) -> Optional[Tuple[str, str]]:
        """
        Détecte un coup à partir des changements d'état du plateau.
        Retourne (from_square, to_square) ou None si invalide.
        """
        # Mouvement simple: 1 pièce soulevée, 1 pièce posée
        if len(lifted) == 1 and len(placed) == 1:
            from_row, from_col = lifted[0]
            to_row, to_col = placed[0]
            from_sq = self._coords_to_square(from_row, from_col)
            to_sq = self._coords_to_square(to_row, to_col)
            return (from_sq, to_sq)
        
        # Capture: 2 pièces soulevées, 1 pièce posée
        if len(lifted) == 2 and len(placed) == 1:
            to_row, to_col = placed[0]
            to_sq = self._coords_to_square(to_row, to_col)
            
            # Trouver la case de départ (celle qui n'est pas la case d'arrivée)
            for row, col in lifted:
                if (row, col) != (to_row, to_col):
                    from_sq = self._coords_to_square(row, col)
                    return (from_sq, to_sq)
        
        # Roque: 2 pièces soulevées, 2 pièces posées
        if len(lifted) == 2 and len(placed) == 2:
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                king_row = chess.square_rank(king_sq)
                king_col = chess.square_file(king_sq)
                
                for row, col in lifted:
                    if row == king_row and col == king_col:
                        # Le roi a été soulevé, chercher sa destination
                        for pr, pc in placed:
                            if pr == king_row and (pc == 2 or pc == 6):  # c ou g file
                                from_sq = self._coords_to_square(king_row, king_col)
                                to_sq = self._coords_to_square(pr, pc)
                                return (from_sq, to_sq)
        
        return None
    
    def _coords_to_square(self, row: int, col: int) -> str:
        """Convertit des coordonnées (row, col) en notation algébrique."""
        file_char = chr(ord('a') + col)
        rank_char = str(row + 1)
        return file_char + rank_char
    
    def _square_to_coords(self, square: str) -> Tuple[int, int]:
        """Convertit une notation algébrique en coordonnées (row, col)."""
        col = ord(square[0]) - ord('a')
        row = int(square[1]) - 1
        return (row, col)
    
    def get_expected_board_state(self) -> List[List[bool]]:
        """
        Retourne l'état attendu du plateau physique basé sur la position actuelle.
        """
        state = [[False for _ in range(8)] for _ in range(8)]
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = chess.square_rank(square)
                col = chess.square_file(square)
                state[row][col] = True
        
        return state
    
    def get_game_status(self) -> Dict[str, Any]:
        """Retourne le statut complet de la partie."""
        result = None
        if self.board.is_game_over():
            outcome = self.board.outcome()
            if outcome:
                if outcome.winner == chess.WHITE:
                    result = "white"
                elif outcome.winner == chess.BLACK:
                    result = "black"
                else:
                    result = "draw"
        
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "current_player": self.get_current_player().value,
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_game_over": self.board.is_game_over(),
            "result": result,
            "move_count": len(self.move_history),
            "legal_moves": self.get_legal_moves(),
            "last_move": self.move_history[-1] if self.move_history else None
        }
    
    def get_move_history_pgn(self) -> str:
        """Retourne l'historique des coups en format PGN simplifié."""
        moves = []
        for i, move in enumerate(self.move_history):
            if i % 2 == 0:
                moves.append(f"{i // 2 + 1}.")
            moves.append(move["san"])
        return " ".join(moves)


if __name__ == "__main__":
    # Test rapide
    gm = GameManager()
    
    print("État initial:")
    print(gm.get_game_status())
    
    print("\nCoup e2-e4:")
    result = gm.make_move("e2", "e4")
    print(result)
    
    print("\nCoup e7-e5:")
    result = gm.make_move("e7", "e5")
    print(result)
    
    print("\nHistorique PGN:", gm.get_move_history_pgn())
