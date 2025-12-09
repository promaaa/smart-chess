"""
Configuration module for the vision system.
Contains all configurable parameters for detection and tracking.
"""
from dataclasses import dataclass, field
from typing import Tuple
from enum import Enum


class TrackerState(Enum):
    """State of the chessboard tracker."""
    DETECTING = "detecting"   # Running CNN to find board
    TRACKING = "tracking"     # Using LK for fast updates
    LOST = "lost"             # Tracking failed, need to re-detect


@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    device_id: int = 0                    # Camera device ID or path
    width: int = 640                      # Frame width
    height: int = 480                     # Frame height
    fps: int = 30                         # Target frames per second
    use_picamera: bool = False            # Use PiCamera2 instead of OpenCV


@dataclass
class DetectionConfig:
    """CNN detection configuration."""
    input_size: Tuple[int, int] = (224, 224)  # Model input dimensions
    model_path: str = "checkpoints/chessboard_cnn.pt"
    confidence_threshold: float = 0.8         # Min confidence for valid detection
    use_gpu: bool = False                     # Use GPU if available
    batch_size: int = 1                       # Batch size for inference


@dataclass
class TrackingConfig:
    """Lucas-Kanade tracking configuration."""
    # LK optical flow parameters
    win_size: Tuple[int, int] = (21, 21)     # Search window size
    max_level: int = 3                        # Pyramid levels
    
    # Tracking quality thresholds
    min_eigval_threshold: float = 0.001       # Minimum eigenvalue for good features
    forward_backward_threshold: float = 1.0   # Max FB error for valid tracking
    min_tracked_points: int = 3               # Min points to consider tracking valid
    
    # Additional feature points on board edges
    edge_points_per_side: int = 4             # Extra points to track per edge
    
    # State transition
    consecutive_lost_frames: int = 5          # Frames before switching to DETECTING


@dataclass
class Config:
    """Main configuration container."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    
    # Debug options
    debug_visualization: bool = True          # Show debug overlay
    save_debug_frames: bool = False           # Save frames for debugging
    debug_output_dir: str = "debug_output"
    
    @classmethod
    def for_raspberry_pi(cls) -> "Config":
        """Optimized configuration for Raspberry Pi."""
        return cls(
            camera=CameraConfig(
                width=640,
                height=480,
                fps=30,
                use_picamera=True
            ),
            detection=DetectionConfig(
                input_size=(160, 160),        # Smaller for faster inference
                use_gpu=False
            ),
            tracking=TrackingConfig(
                win_size=(15, 15),            # Smaller window for speed
                max_level=2
            ),
            debug_visualization=False         # Disable for production
        )
    
    @classmethod
    def for_development(cls) -> "Config":
        """Configuration for development/testing on desktop."""
        return cls(
            camera=CameraConfig(
                width=1280,
                height=720,
                fps=30,
                use_picamera=False
            ),
            detection=DetectionConfig(
                input_size=(224, 224),
                use_gpu=True
            ),
            debug_visualization=True
        )
