"""
State management for the detection/tracking pipeline.

Handles transitions between DETECTING, TRACKING, and LOST states
with hysteresis to prevent rapid state oscillation.
"""
from enum import Enum
from typing import Optional
from dataclasses import dataclass
import time


class TrackerState(Enum):
    """Current state of the chessboard tracker."""
    DETECTING = "detecting"   # Running CNN to find board
    TRACKING = "tracking"     # Using LK for fast updates  
    LOST = "lost"             # Tracking failed, transitioning to detection


@dataclass
class StateInfo:
    """Information about the current state."""
    state: TrackerState
    frames_in_state: int
    last_transition_time: float
    confidence: float
    message: str


class TrackerStateManager:
    """
    Manages state transitions for the detection/tracking pipeline.
    
    State machine:
        DETECTING -> TRACKING: When CNN successfully detects board
        TRACKING -> LOST: When tracking confidence drops below threshold
        LOST -> DETECTING: After cooldown period
        TRACKING -> DETECTING: Manual reset or extended tracking failure
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.8,
        min_tracking_confidence: float = 0.3,
        consecutive_lost_threshold: int = 5,
        cooldown_frames: int = 3
    ):
        """
        Initialize state manager.
        
        Args:
            min_detection_confidence: Minimum CNN confidence to transition to TRACKING
            min_tracking_confidence: Minimum LK confidence before LOST
            consecutive_lost_threshold: Frames of low confidence before LOST
            cooldown_frames: Frames to wait before re-detecting after LOST
        """
        self.min_detection_conf = min_detection_confidence
        self.min_tracking_conf = min_tracking_confidence
        self.lost_threshold = consecutive_lost_threshold
        self.cooldown_frames = cooldown_frames
        
        # State
        self._state = TrackerState.DETECTING
        self._frames_in_state = 0
        self._low_confidence_streak = 0
        self._last_transition = time.time()
        self._last_confidence = 0.0
        
    @property
    def state(self) -> TrackerState:
        """Get current state."""
        return self._state
    
    @property
    def is_detecting(self) -> bool:
        return self._state == TrackerState.DETECTING
    
    @property
    def is_tracking(self) -> bool:
        return self._state == TrackerState.TRACKING
    
    @property
    def is_lost(self) -> bool:
        return self._state == TrackerState.LOST
    
    def get_info(self) -> StateInfo:
        """Get detailed state information."""
        messages = {
            TrackerState.DETECTING: "Searching for chessboard...",
            TrackerState.TRACKING: f"Tracking (confidence: {self._last_confidence:.2f})",
            TrackerState.LOST: f"Lost tracking, re-detecting in {self.cooldown_frames - self._frames_in_state} frames..."
        }
        
        return StateInfo(
            state=self._state,
            frames_in_state=self._frames_in_state,
            last_transition_time=self._last_transition,
            confidence=self._last_confidence,
            message=messages[self._state]
        )
    
    def on_detection_result(self, detected: bool, confidence: float) -> TrackerState:
        """
        Update state after CNN detection attempt.
        
        Args:
            detected: Whether board was detected
            confidence: Detection confidence
            
        Returns:
            New state after update
        """
        self._last_confidence = confidence
        
        if self._state == TrackerState.DETECTING:
            if detected and confidence >= self.min_detection_conf:
                self._transition_to(TrackerState.TRACKING)
            else:
                self._frames_in_state += 1
                
        elif self._state == TrackerState.LOST:
            self._frames_in_state += 1
            if self._frames_in_state >= self.cooldown_frames:
                self._transition_to(TrackerState.DETECTING)
        
        return self._state
    
    def on_tracking_result(self, valid: bool, confidence: float) -> TrackerState:
        """
        Update state after LK tracking attempt.
        
        Args:
            valid: Whether tracking was valid
            confidence: Tracking confidence
            
        Returns:
            New state after update
        """
        self._last_confidence = confidence
        
        if self._state != TrackerState.TRACKING:
            return self._state
        
        if valid and confidence >= self.min_tracking_conf:
            # Good tracking
            self._low_confidence_streak = 0
            self._frames_in_state += 1
        else:
            # Poor tracking
            self._low_confidence_streak += 1
            
            if self._low_confidence_streak >= self.lost_threshold:
                self._transition_to(TrackerState.LOST)
        
        return self._state
    
    def force_detection(self) -> None:
        """Force transition to DETECTING state (manual reset)."""
        self._transition_to(TrackerState.DETECTING)
    
    def reset(self) -> None:
        """Full reset to initial state."""
        self._state = TrackerState.DETECTING
        self._frames_in_state = 0
        self._low_confidence_streak = 0
        self._last_transition = time.time()
        self._last_confidence = 0.0
    
    def _transition_to(self, new_state: TrackerState) -> None:
        """Perform state transition."""
        if new_state != self._state:
            self._state = new_state
            self._frames_in_state = 0
            self._low_confidence_streak = 0
            self._last_transition = time.time()


class AdaptiveStateManager(TrackerStateManager):
    """
    Extended state manager with adaptive thresholds.
    
    Adjusts thresholds based on historical tracking quality.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tracking_history: list = []
        self._history_window = 30
        
    def on_tracking_result(self, valid: bool, confidence: float) -> TrackerState:
        """Track history and adapt thresholds."""
        # Record history
        self._tracking_history.append(confidence)
        if len(self._tracking_history) > self._history_window:
            self._tracking_history.pop(0)
        
        # Adapt lost threshold based on stability
        if len(self._tracking_history) >= 10:
            avg_conf = sum(self._tracking_history[-10:]) / 10
            if avg_conf > 0.8:
                # Very stable tracking - be more tolerant
                self.lost_threshold = min(10, self.lost_threshold + 1)
            elif avg_conf < 0.5:
                # Unstable - be more aggressive about re-detecting
                self.lost_threshold = max(3, self.lost_threshold - 1)
        
        return super().on_tracking_result(valid, confidence)
