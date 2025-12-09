"""
Unified chessboard detector combining CNN detection and LK tracking.

This is the main entry point for the vision system.
"""
import cv2
import numpy as np
import torch
import time
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

from .config import Config, TrackerState
from .detection.model import load_model
from .detection.preprocessing import preprocess_frame, denormalize_corners
from .tracking.lk_tracker import LKTracker, TrackingResult
from .tracking.state import TrackerStateManager


@dataclass
class DetectionResult:
    """Result from the chessboard detector."""
    corners: Optional[np.ndarray]    # Corner positions (4, 2) in pixel coords, or None
    state: TrackerState              # Current state (DETECTING, TRACKING, LOST)
    confidence: float                # Confidence score [0, 1]
    inference_time_ms: float         # Time for this frame's processing
    detected: bool                   # Whether board was successfully located
    
    def get_corners_as_list(self) -> Optional[list]:
        """Get corners as a list of [x, y] pairs."""
        if self.corners is None:
            return None
        return self.corners.tolist()


class ChessboardDetector:
    """
    Main chessboard detector combining CNN detection and LK tracking.
    
    Usage:
        detector = ChessboardDetector(config)
        detector.load_model("path/to/model.pt")
        
        while True:
            frame = camera.read()
            result = detector.process_frame(frame)
            if result.detected:
                print(f"Corners: {result.corners}")
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize detector.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or Config()
        
        # CNN model (loaded separately)
        self.model: Optional[torch.nn.Module] = None
        self.device = "cpu"
        
        # LK Tracker
        self.tracker = LKTracker(
            win_size=self.config.tracking.win_size,
            max_level=self.config.tracking.max_level,
            min_eigval_threshold=self.config.tracking.min_eigval_threshold,
            forward_backward_threshold=self.config.tracking.forward_backward_threshold,
            edge_points_per_side=self.config.tracking.edge_points_per_side
        )
        
        # State manager
        self.state_manager = TrackerStateManager(
            min_detection_confidence=self.config.detection.confidence_threshold,
            consecutive_lost_threshold=self.config.tracking.consecutive_lost_frames
        )
        
        # Cache
        self._last_corners: Optional[np.ndarray] = None
        self._last_frame_size: Optional[Tuple[int, int]] = None
        
    def load_model(self, model_path: str, use_lite: bool = False) -> None:
        """
        Load the CNN model for detection.
        
        Args:
            model_path: Path to model checkpoint
            use_lite: Whether to use lightweight model
        """
        self.device = "cuda" if (self.config.detection.use_gpu and 
                                  torch.cuda.is_available()) else "cpu"
        
        self.model = load_model(
            model_path,
            input_size=self.config.detection.input_size,
            use_lite=use_lite,
            device=self.device
        )
        print(f"Loaded model from {model_path} on {self.device}")
    
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Process a video frame and return chessboard corners.
        
        Automatically switches between CNN detection and LK tracking based
        on the current state.
        
        Args:
            frame: BGR frame from camera
            
        Returns:
            DetectionResult with corners and metadata
        """
        start_time = time.perf_counter()
        
        # Store frame size for coordinate denormalization
        h, w = frame.shape[:2]
        self._last_frame_size = (h, w)
        
        if self.state_manager.is_detecting or self.state_manager.is_lost:
            # Use CNN detection
            result = self._detect(frame)
        else:
            # Use LK tracking
            result = self._track(frame)
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        return DetectionResult(
            corners=result[0],
            state=self.state_manager.state,
            confidence=result[1],
            inference_time_ms=inference_time,
            detected=result[0] is not None
        )
    
    def _detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Run CNN detection on frame.
        
        Returns:
            Tuple of (corners, confidence) or (None, 0) if detection failed
        """
        if self.model is None:
            # Model not loaded - can't detect
            self.state_manager.on_detection_result(False, 0.0)
            return None, 0.0
        
        # Preprocess
        processed = preprocess_frame(frame, self.config.detection.input_size)
        input_tensor = torch.from_numpy(processed).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)  # Shape: (1, 8)
        
        # Get corners and denormalize
        corners_normalized = output.cpu().numpy().reshape(4, 2)
        h, w = frame.shape[:2]
        corners = denormalize_corners(corners_normalized, (h, w))
        
        # Validate detection (simple heuristic: corners should form valid quadrilateral)
        confidence, is_valid = self._validate_corners(corners, frame.shape[:2])
        
        # Update state
        self.state_manager.on_detection_result(is_valid, confidence)
        
        if is_valid:
            self._last_corners = corners
            # Initialize tracker with detected corners
            self.tracker.initialize(frame, corners)
            return corners, confidence
        else:
            return None, confidence
    
    def _track(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Run LK tracking on frame.
        
        Returns:
            Tuple of (corners, confidence) or (None, 0) if tracking failed
        """
        result = self.tracker.track(frame)
        
        # Update state
        self.state_manager.on_tracking_result(result.tracking_valid, result.confidence)
        
        if result.tracking_valid:
            self._last_corners = result.corners
            return result.corners, result.confidence
        else:
            return None, result.confidence
    
    def _validate_corners(
        self,
        corners: np.ndarray,
        frame_size: Tuple[int, int]
    ) -> Tuple[float, bool]:
        """
        Validate detected corners form a reasonable quadrilateral.
        
        Args:
            corners: Detected corners (4, 2)
            frame_size: (height, width) of frame
            
        Returns:
            Tuple of (confidence, is_valid)
        """
        h, w = frame_size
        
        # Check corners are within frame
        in_bounds = np.all((corners >= 0) & (corners[:, 0] < w) & (corners[:, 1] < h))
        if not in_bounds:
            return 0.0, False
        
        # Check corners form convex quadrilateral
        def cross_product_sign(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        signs = []
        for i in range(4):
            o = corners[i]
            a = corners[(i + 1) % 4]
            b = corners[(i + 2) % 4]
            signs.append(np.sign(cross_product_sign(o, a, b)))
        
        is_convex = len(set(signs)) == 1 and signs[0] != 0
        if not is_convex:
            return 0.2, False
        
        # Check area is reasonable (not too small or too large)
        x = corners[:, 0]
        y = corners[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        frame_area = h * w
        area_ratio = area / frame_area
        
        if area_ratio < 0.01:  # Too small
            return 0.3, False
        if area_ratio > 0.95:  # Too large
            return 0.3, False
        
        # Compute confidence based on area ratio (optimal around 0.3-0.7)
        if 0.1 < area_ratio < 0.9:
            confidence = 0.9
        else:
            confidence = 0.6
        
        return confidence, True
    
    def get_state(self) -> TrackerState:
        """Get current detector state."""
        return self.state_manager.state
    
    def get_last_corners(self) -> Optional[np.ndarray]:
        """Get last known corner positions."""
        return self._last_corners
    
    def reset(self) -> None:
        """Reset detector to initial state (force re-detection)."""
        self.state_manager.reset()
        self.tracker.reset()
        self._last_corners = None
    
    def force_detection(self) -> None:
        """Force return to detection mode."""
        self.state_manager.force_detection()
        self.tracker.reset()


def create_detector(config_preset: str = "development") -> ChessboardDetector:
    """
    Create a detector with a preset configuration.
    
    Args:
        config_preset: One of 'development' or 'raspberry_pi'
        
    Returns:
        Configured ChessboardDetector instance
    """
    if config_preset == "raspberry_pi":
        config = Config.for_raspberry_pi()
    else:
        config = Config.for_development()
    
    return ChessboardDetector(config)
