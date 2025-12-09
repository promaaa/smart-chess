"""
Integration module bridging vision system with reed sensor hardware.

Provides APIs for comparing vision-detected board state with reed sensor
readings to detect discrepancies, illegal moves, or position verification.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SquareStatus(Enum):
    """Status of a chessboard square."""
    EMPTY = 0
    OCCUPIED = 1
    UNKNOWN = 2


@dataclass
class BoardState:
    """Complete board state from a sensor source."""
    squares: np.ndarray  # 8x8 array of SquareStatus values
    source: str          # "vision" or "reed"
    timestamp: float     # Timestamp of reading
    confidence: float    # Confidence in the reading [0, 1]


@dataclass
class ComparisonResult:
    """Result of comparing vision and reed sensor states."""
    matches: bool                      # Whether states match
    discrepancies: List[Tuple[int, int]]  # List of (row, col) with mismatches
    confidence: float                  # Confidence in comparison
    message: str                       # Human-readable description


class VisionReedBridge:
    """
    Bridge between vision system and reed sensor system.
    
    Used to:
    1. Compare vision detections with reed sensor readings
    2. Provide visual confirmation of piece positions
    3. Detect potential illegal moves or sensor errors
    """
    
    def __init__(self, chessboard_corners: Optional[np.ndarray] = None):
        """
        Initialize the bridge.
        
        Args:
            chessboard_corners: Initial corner positions (4, 2) from vision
        """
        self._corners = chessboard_corners
        self._last_vision_state: Optional[BoardState] = None
        self._last_reed_state: Optional[BoardState] = None
        
    def set_corners(self, corners: np.ndarray) -> None:
        """Update chessboard corner positions from vision system."""
        self._corners = corners
    
    def pixel_to_square(
        self,
        pixel: Tuple[float, float]
    ) -> Optional[Tuple[int, int]]:
        """
        Convert pixel coordinates to board square.
        
        Args:
            pixel: (x, y) pixel coordinates
            
        Returns:
            (row, col) square indices (0-7), or None if outside board
        """
        if self._corners is None:
            return None
        
        # Use perspective transform to map pixel to normalized board coords
        src_pts = self._corners.astype(np.float32)
        dst_pts = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        
        import cv2
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Transform the point
        pt = np.array([[pixel]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, M)[0, 0]
        
        # Check if within board bounds
        if 0 <= transformed[0] <= 1 and 0 <= transformed[1] <= 1:
            col = int(transformed[0] * 8)
            row = int(transformed[1] * 8)
            col = min(col, 7)
            row = min(row, 7)
            return (row, col)
        
        return None
    
    def square_to_pixel(
        self,
        row: int,
        col: int
    ) -> Optional[Tuple[float, float]]:
        """
        Convert board square to pixel coordinates (center of square).
        
        Args:
            row: Row index (0-7)
            col: Column index (0-7)
            
        Returns:
            (x, y) pixel coordinates of square center, or None if corners not set
        """
        if self._corners is None:
            return None
        
        # Normalized position of square center
        norm_x = (col + 0.5) / 8
        norm_y = (row + 0.5) / 8
        
        # Bilinear interpolation between corners
        tl, tr, br, bl = self._corners
        
        top = tl + norm_x * (tr - tl)
        bottom = bl + norm_x * (br - bl)
        pixel = top + norm_y * (bottom - top)
        
        return (float(pixel[0]), float(pixel[1]))
    
    def update_reed_state(
        self,
        reed_matrix: np.ndarray,
        timestamp: float
    ) -> None:
        """
        Update the last known reed sensor state.
        
        Args:
            reed_matrix: 8x8 boolean array (True = piece present)
            timestamp: Time of reading
        """
        squares = np.where(reed_matrix, SquareStatus.OCCUPIED.value, 
                          SquareStatus.EMPTY.value)
        self._last_reed_state = BoardState(
            squares=squares,
            source="reed",
            timestamp=timestamp,
            confidence=0.99  # Reed sensors are highly reliable
        )
    
    def update_vision_state(
        self,
        vision_squares: np.ndarray,
        timestamp: float,
        confidence: float
    ) -> None:
        """
        Update the last known vision state.
        
        Args:
            vision_squares: 8x8 array of SquareStatus values
            timestamp: Time of detection
            confidence: Overall vision confidence
        """
        self._last_vision_state = BoardState(
            squares=vision_squares,
            source="vision",
            timestamp=timestamp,
            confidence=confidence
        )
    
    def compare_states(self) -> ComparisonResult:
        """
        Compare vision and reed sensor states.
        
        Returns:
            ComparisonResult with match status and discrepancies
        """
        if self._last_vision_state is None:
            return ComparisonResult(
                matches=False,
                discrepancies=[],
                confidence=0.0,
                message="Vision state not available"
            )
        
        if self._last_reed_state is None:
            return ComparisonResult(
                matches=False,
                discrepancies=[],
                confidence=0.0,
                message="Reed sensor state not available"
            )
        
        # Find discrepancies
        discrepancies = []
        vision = self._last_vision_state.squares
        reed = self._last_reed_state.squares
        
        for row in range(8):
            for col in range(8):
                v_status = vision[row, col]
                r_status = reed[row, col]
                
                # Skip unknown squares
                if v_status == SquareStatus.UNKNOWN.value:
                    continue
                
                if v_status != r_status:
                    discrepancies.append((row, col))
        
        matches = len(discrepancies) == 0
        confidence = self._last_vision_state.confidence * self._last_reed_state.confidence
        
        if matches:
            message = "Vision and reed sensors agree on board state"
        else:
            message = f"Found {len(discrepancies)} discrepancies between vision and reed sensors"
        
        return ComparisonResult(
            matches=matches,
            discrepancies=discrepancies,
            confidence=confidence,
            message=message
        )
    
    def get_square_visual_confirmation(
        self,
        row: int,
        col: int
    ) -> Dict:
        """
        Get visual confirmation status for a specific square.
        
        Useful for verifying a move or checking a suspicious sensor reading.
        
        Args:
            row: Row index (0-7)
            col: Column index (0-7)
            
        Returns:
            Dict with vision and reed status for the square
        """
        result = {
            "row": row,
            "col": col,
            "vision_status": None,
            "reed_status": None,
            "match": None,
            "pixel_center": self.square_to_pixel(row, col)
        }
        
        if self._last_vision_state is not None:
            v_val = self._last_vision_state.squares[row, col]
            result["vision_status"] = SquareStatus(v_val).name
        
        if self._last_reed_state is not None:
            r_val = self._last_reed_state.squares[row, col]
            result["reed_status"] = SquareStatus(r_val).name
        
        if result["vision_status"] and result["reed_status"]:
            result["match"] = result["vision_status"] == result["reed_status"]
        
        return result


def square_name(row: int, col: int) -> str:
    """Convert row/col to algebraic notation (e.g., 'e4')."""
    return f"{chr(ord('a') + col)}{row + 1}"


def parse_square_name(name: str) -> Tuple[int, int]:
    """Convert algebraic notation to row/col."""
    col = ord(name[0].lower()) - ord('a')
    row = int(name[1]) - 1
    return (row, col)
