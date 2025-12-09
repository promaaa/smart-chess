"""
Lucas-Kanade optical flow tracker for chessboard corner tracking.

This module implements fast frame-to-frame tracking of chessboard corners
using the LK algorithm, which is significantly faster than CNN detection.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class TrackingResult:
    """Result of a tracking operation."""
    corners: np.ndarray           # Updated corner positions (4, 2)
    confidence: float             # Overall tracking confidence [0, 1]
    individual_status: np.ndarray # Status for each tracked point
    tracking_valid: bool          # Whether tracking is considered valid


class LKTracker:
    """
    Lucas-Kanade optical flow tracker for chessboard corners.
    
    Tracks 4 corner points plus additional edge points for robustness.
    Uses forward-backward consistency check to validate tracking quality.
    """
    
    def __init__(
        self,
        win_size: Tuple[int, int] = (21, 21),
        max_level: int = 3,
        min_eigval_threshold: float = 0.001,
        forward_backward_threshold: float = 1.0,
        edge_points_per_side: int = 4
    ):
        """
        Initialize the tracker.
        
        Args:
            win_size: Size of the search window for LK algorithm
            max_level: Number of pyramid levels to use
            min_eigval_threshold: Minimum eigenvalue threshold for good features
            forward_backward_threshold: Maximum allowed forward-backward error
            edge_points_per_side: Number of additional edge points per side
        """
        self.win_size = win_size
        self.max_level = max_level
        self.min_eigval_threshold = min_eigval_threshold
        self.fb_threshold = forward_backward_threshold
        self.edge_points_per_side = edge_points_per_side
        
        # LK parameters
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            minEigThreshold=min_eigval_threshold
        )
        
        # State
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.corner_indices: Optional[np.ndarray] = None
        
    def initialize(self, frame: np.ndarray, corners: np.ndarray) -> None:
        """
        Initialize tracker with detected chessboard corners.
        
        Args:
            frame: Current BGR frame
            corners: Detected corner positions (4, 2) in pixel coordinates
        """
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Generate additional edge points for more robust tracking
        all_points = self._generate_tracking_points(corners)
        
        self.prev_points = all_points.astype(np.float32)
        self.corner_indices = np.arange(4)  # First 4 points are corners
        
    def _generate_tracking_points(self, corners: np.ndarray) -> np.ndarray:
        """
        Generate additional tracking points along board edges.
        
        Args:
            corners: 4 corner points (4, 2)
            
        Returns:
            Array of all tracking points including edges
        """
        points = [corners]
        
        if self.edge_points_per_side > 0:
            # Generate points along each edge
            for i in range(4):
                start = corners[i]
                end = corners[(i + 1) % 4]
                
                # Interpolate points along edge
                for j in range(1, self.edge_points_per_side + 1):
                    t = j / (self.edge_points_per_side + 1)
                    point = start + t * (end - start)
                    points.append(point.reshape(1, 2))
        
        return np.vstack(points)
    
    def track(self, frame: np.ndarray) -> TrackingResult:
        """
        Track corners from previous frame to current frame.
        
        Args:
            frame: Current BGR frame
            
        Returns:
            TrackingResult with updated corner positions and validity info
        """
        if self.prev_gray is None or self.prev_points is None:
            return TrackingResult(
                corners=np.zeros((4, 2)),
                confidence=0.0,
                individual_status=np.zeros(4),
                tracking_valid=False
            )
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Forward tracking: prev -> curr
        next_points, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            self.prev_points,
            None,
            **self.lk_params
        )
        
        # Backward tracking: curr -> prev (for forward-backward consistency)
        back_points, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray,
            self.prev_gray,
            next_points,
            None,
            **self.lk_params
        )
        
        # Compute forward-backward error
        fb_error = np.linalg.norm(self.prev_points - back_points, axis=1)
        
        # Determine which points are valid
        valid = (status_fwd.flatten() == 1) & \
                (status_bwd.flatten() == 1) & \
                (fb_error < self.fb_threshold)
        
        # Extract corner status
        corner_valid = valid[self.corner_indices]
        corner_count = np.sum(corner_valid)
        
        # Only update if we have at least 3 valid corners
        tracking_valid = corner_count >= 3
        
        if tracking_valid:
            # Update corners
            updated_corners = next_points[self.corner_indices]
            
            # If a corner is lost, estimate from valid points
            if corner_count < 4:
                updated_corners = self._estimate_missing_corners(
                    updated_corners, corner_valid, next_points, valid
                )
            
            # Calculate overall confidence
            confidence = corner_count / 4.0 * np.mean(1.0 - fb_error[self.corner_indices] / self.fb_threshold)
            confidence = max(0.0, min(1.0, confidence))
            
            # Update state for next frame
            self.prev_gray = curr_gray
            self.prev_points = self._generate_tracking_points(updated_corners).astype(np.float32)
            
            return TrackingResult(
                corners=updated_corners,
                confidence=confidence,
                individual_status=corner_valid.astype(np.float32),
                tracking_valid=True
            )
        else:
            return TrackingResult(
                corners=next_points[self.corner_indices] if next_points is not None else np.zeros((4, 2)),
                confidence=0.0,
                individual_status=corner_valid.astype(np.float32),
                tracking_valid=False
            )
    
    def _estimate_missing_corners(
        self,
        corners: np.ndarray,
        valid: np.ndarray,
        all_points: np.ndarray,
        all_valid: np.ndarray
    ) -> np.ndarray:
        """
        Estimate missing corners using homography from valid points.
        
        Args:
            corners: Current corner positions (may have invalid ones)
            valid: Boolean mask for valid corners
            all_points: All tracked points
            all_valid: Boolean mask for all valid points
            
        Returns:
            Updated corners with missing ones estimated
        """
        if np.sum(valid) < 3:
            return corners
        
        estimated = corners.copy()
        
        # Use simple geometric estimation for single missing corner
        if np.sum(valid) == 3:
            missing_idx = np.where(~valid)[0][0]
            
            # Opposite corner method: estimate from parallelogram assumption
            # Corner order: 0=TL, 1=TR, 2=BR, 3=BL
            opposite = (missing_idx + 2) % 4
            adjacent1 = (missing_idx + 1) % 4
            adjacent2 = (missing_idx - 1) % 4
            
            if valid[opposite] and valid[adjacent1] and valid[adjacent2]:
                # P_missing = P_adj1 + P_adj2 - P_opposite
                estimated[missing_idx] = corners[adjacent1] + corners[adjacent2] - corners[opposite]
        
        return estimated
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.prev_gray = None
        self.prev_points = None
        self.corner_indices = None
    
    def is_initialized(self) -> bool:
        """Check if tracker is initialized."""
        return self.prev_gray is not None


def track_corners_single_frame(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_corners: np.ndarray,
    lk_params: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simple one-shot corner tracking between two frames.
    
    Args:
        prev_gray: Previous grayscale frame
        curr_gray: Current grayscale frame
        prev_corners: Previous corner positions (4, 2)
        lk_params: LK algorithm parameters
        
    Returns:
        Tuple of (new_corners, valid_mask, mean_error)
    """
    if lk_params is None:
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
    
    points = prev_corners.reshape(-1, 1, 2).astype(np.float32)
    
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, points, None, **lk_params
    )
    
    # Forward-backward check
    back_points, status_back, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, next_points, None, **lk_params
    )
    
    error = np.linalg.norm(points - back_points, axis=2).flatten()
    valid = (status.flatten() == 1) & (status_back.flatten() == 1) & (error < 1.0)
    
    return next_points.reshape(-1, 2), valid, np.mean(error)
