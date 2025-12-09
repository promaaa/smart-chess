"""
Visualization utilities for debugging the chessboard detection system.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List


# Color scheme (BGR format)
COLORS = {
    'corner': (0, 255, 0),       # Green for detected corners
    'edge': (0, 255, 255),       # Yellow for edge points
    'lost': (0, 0, 255),         # Red for lost tracking
    'detecting': (255, 128, 0),  # Orange for detection mode
    'tracking': (0, 255, 0),     # Green for tracking mode
    'board': (255, 0, 255),      # Magenta for board outline
    'text': (255, 255, 255),     # White text
    'background': (0, 0, 0)      # Black background
}


def draw_corners(
    frame: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = COLORS['corner'],
    radius: int = 8,
    thickness: int = 2,
    draw_labels: bool = True
) -> np.ndarray:
    """
    Draw corner points on frame.
    
    Args:
        frame: Input BGR frame
        corners: Corner coordinates (4, 2)
        color: Point color in BGR
        radius: Circle radius
        thickness: Line thickness (-1 for filled)
        draw_labels: Whether to draw corner labels (TL, TR, BR, BL)
        
    Returns:
        Frame with corners drawn
    """
    result = frame.copy()
    labels = ['TL', 'TR', 'BR', 'BL']
    
    for i, (x, y) in enumerate(corners):
        x, y = int(x), int(y)
        
        # Draw corner circle
        cv2.circle(result, (x, y), radius, color, thickness)
        
        # Draw corner label
        if draw_labels:
            cv2.putText(
                result, labels[i],
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA
            )
    
    return result


def draw_board_outline(
    frame: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = COLORS['board'],
    thickness: int = 2
) -> np.ndarray:
    """
    Draw lines connecting the 4 corners to show board outline.
    
    Args:
        frame: Input BGR frame
        corners: Corner coordinates (4, 2)
        color: Line color in BGR
        thickness: Line thickness
        
    Returns:
        Frame with board outline drawn
    """
    result = frame.copy()
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(result, [pts], isClosed=True, color=color, thickness=thickness)
    return result


def draw_tracking_points(
    frame: np.ndarray,
    points: np.ndarray,
    valid: Optional[np.ndarray] = None,
    color_valid: Tuple[int, int, int] = COLORS['edge'],
    color_invalid: Tuple[int, int, int] = COLORS['lost'],
    radius: int = 4
) -> np.ndarray:
    """
    Draw all tracking points (corners + edge points).
    
    Args:
        frame: Input BGR frame
        points: All tracked point coordinates (N, 2)
        valid: Boolean mask for valid points
        color_valid: Color for valid points
        color_invalid: Color for invalid points
        radius: Point radius
        
    Returns:
        Frame with tracking points drawn
    """
    result = frame.copy()
    
    if valid is None:
        valid = np.ones(len(points), dtype=bool)
    
    for i, (x, y) in enumerate(points):
        x, y = int(x), int(y)
        color = color_valid if valid[i] else color_invalid
        cv2.circle(result, (x, y), radius, color, -1)
    
    return result


def draw_debug_overlay(
    frame: np.ndarray,
    corners: Optional[np.ndarray],
    state: str,
    confidence: float,
    fps: float,
    inference_time_ms: float = 0.0,
    extra_info: Optional[dict] = None
) -> np.ndarray:
    """
    Draw complete debug overlay with state info and performance metrics.
    
    Args:
        frame: Input BGR frame
        corners: Detected corners (4, 2) or None if not detected
        state: Current tracker state ('detecting', 'tracking', 'lost')
        confidence: Current confidence score [0, 1]
        fps: Current frames per second
        inference_time_ms: Last inference time in milliseconds
        extra_info: Additional key-value pairs to display
        
    Returns:
        Frame with full debug overlay
    """
    result = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw corners and board if available
    if corners is not None:
        result = draw_board_outline(result, corners)
        result = draw_corners(result, corners)
    
    # State indicator colors
    state_colors = {
        'detecting': COLORS['detecting'],
        'tracking': COLORS['tracking'],
        'lost': COLORS['lost']
    }
    state_color = state_colors.get(state.lower(), COLORS['text'])
    
    # Draw state indicator bar at top
    bar_height = 40
    cv2.rectangle(result, (0, 0), (w, bar_height), (40, 40, 40), -1)
    
    # State text
    state_text = f"State: {state.upper()}"
    cv2.putText(result, state_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2, cv2.LINE_AA)
    
    # Confidence bar
    conf_bar_x = 200
    conf_bar_w = 100
    conf_bar_h = 20
    
    cv2.rectangle(result, (conf_bar_x, 10), (conf_bar_x + conf_bar_w, 10 + conf_bar_h),
                  (100, 100, 100), -1)
    filled_w = int(conf_bar_w * confidence)
    cv2.rectangle(result, (conf_bar_x, 10), (conf_bar_x + filled_w, 10 + conf_bar_h),
                  state_color, -1)
    cv2.putText(result, f"{confidence:.0%}", (conf_bar_x + conf_bar_w + 10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1, cv2.LINE_AA)
    
    # FPS counter
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(result, fps_text, (w - 100, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 1, cv2.LINE_AA)
    
    # Performance info at bottom
    info_y = h - 60
    cv2.rectangle(result, (0, info_y), (w, h), (40, 40, 40), -1)
    
    if inference_time_ms > 0:
        time_text = f"Inference: {inference_time_ms:.1f}ms"
        cv2.putText(result, time_text, (10, info_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1, cv2.LINE_AA)
    
    # Extra info
    if extra_info:
        x_offset = 200
        for key, value in extra_info.items():
            text = f"{key}: {value}"
            cv2.putText(result, text, (x_offset, info_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1, cv2.LINE_AA)
            x_offset += len(text) * 10 + 20
    
    return result


def draw_grid_overlay(
    frame: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (128, 128, 128),
    thickness: int = 1,
    divisions: int = 8
) -> np.ndarray:
    """
    Draw 8x8 grid overlay for chessboard visualization.
    
    Args:
        frame: Input BGR frame
        corners: 4 corner coordinates
        color: Grid line color
        thickness: Line thickness
        divisions: Number of divisions (8 for chessboard)
        
    Returns:
        Frame with grid overlay
    """
    result = frame.copy()
    
    # Corners: TL, TR, BR, BL
    tl, tr, br, bl = corners
    
    # Draw horizontal lines
    for i in range(divisions + 1):
        t = i / divisions
        left_point = tl + t * (bl - tl)
        right_point = tr + t * (br - tr)
        cv2.line(result,
                 tuple(left_point.astype(int)),
                 tuple(right_point.astype(int)),
                 color, thickness)
    
    # Draw vertical lines
    for i in range(divisions + 1):
        t = i / divisions
        top_point = tl + t * (tr - tl)
        bottom_point = bl + t * (br - bl)
        cv2.line(result,
                 tuple(top_point.astype(int)),
                 tuple(bottom_point.astype(int)),
                 color, thickness)
    
    return result


def create_side_by_side(
    original: np.ndarray,
    processed: np.ndarray,
    title_left: str = "Original",
    title_right: str = "Processed"
) -> np.ndarray:
    """
    Create side-by-side comparison of two frames.
    
    Args:
        original: Original frame
        processed: Processed frame with overlays
        title_left: Title for left image
        title_right: Title for right image
        
    Returns:
        Combined side-by-side frame
    """
    h, w = original.shape[:2]
    
    # Resize if needed
    if processed.shape[:2] != (h, w):
        processed = cv2.resize(processed, (w, h))
    
    # Add titles
    left = original.copy()
    right = processed.copy()
    
    cv2.putText(left, title_left, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text'], 2, cv2.LINE_AA)
    cv2.putText(right, title_right, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text'], 2, cv2.LINE_AA)
    
    # Combine
    return np.hstack([left, right])
