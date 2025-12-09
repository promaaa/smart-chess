#!/usr/bin/env python3
"""
Demo script for testing the chessboard detection and tracking system.

Usage:
    python demo_live.py --camera 0
    python demo_live.py --video path/to/video.mp4
    python demo_live.py --test  # Run with synthetic test pattern
"""
import cv2
import numpy as np
import argparse
import time
from pathlib import Path

from vision.config import Config
from vision.chessboard_detector import ChessboardDetector, create_detector
from vision.utils.visualization import draw_debug_overlay, draw_grid_overlay


def generate_test_chessboard(size: tuple = (640, 480)) -> np.ndarray:
    """Generate a synthetic chessboard image for testing."""
    h, w = size
    img = np.ones((h, w, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Draw chessboard pattern
    board_size = min(h, w) - 100
    start_x = (w - board_size) // 2
    start_y = (h - board_size) // 2
    square_size = board_size // 8
    
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                color = (240, 240, 240)  # Light squares
            else:
                color = (80, 80, 80)  # Dark squares
            
            x = start_x + col * square_size
            y = start_y + row * square_size
            cv2.rectangle(img, (x, y), (x + square_size, y + square_size), color, -1)
    
    # Draw border
    cv2.rectangle(img, (start_x, start_y), 
                  (start_x + board_size, start_y + board_size), 
                  (0, 0, 0), 2)
    
    return img


def run_demo_camera(camera_id: int = 0, config: Config = None) -> None:
    """Run live demo with camera."""
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width if config else 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height if config else 480)
    
    run_demo_loop(cap, config)
    cap.release()


def run_demo_video(video_path: str, config: Config = None) -> None:
    """Run demo with video file."""
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    run_demo_loop(cap, config)
    cap.release()


def run_demo_test(config: Config = None) -> None:
    """Run demo with synthetic test pattern."""
    print("Running with synthetic test pattern...")
    print("Press 'q' to quit, 'r' to reset, 'm' to move board")
    
    # Create detector
    detector = create_detector("development")
    
    # Generate initial test image
    frame = generate_test_chessboard()
    offset_x, offset_y = 0, 0
    
    # FPS tracking
    fps_counter = []
    
    while True:
        start_time = time.perf_counter()
        
        # Shift the chessboard slightly to simulate movement
        shifted = np.roll(frame, offset_x, axis=1)
        shifted = np.roll(shifted, offset_y, axis=0)
        
        # Process frame
        result = detector.process_frame(shifted)
        
        # Calculate FPS
        elapsed = time.perf_counter() - start_time
        fps_counter.append(1.0 / elapsed)
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        fps = np.mean(fps_counter)
        
        # Draw debug overlay
        display = draw_debug_overlay(
            shifted,
            result.corners,
            result.state.value,
            result.confidence,
            fps,
            result.inference_time_ms
        )
        
        # Draw grid if corners detected
        if result.corners is not None:
            display = draw_grid_overlay(display, result.corners)
        
        # Show
        cv2.imshow("Chessboard Detection Demo (Test Mode)", display)
        
        # Handle input
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            offset_x, offset_y = 0, 0
            print("Reset detector")
        elif key == ord('m'):
            offset_x += np.random.randint(-20, 20)
            offset_y += np.random.randint(-20, 20)
            print(f"Moved board to offset ({offset_x}, {offset_y})")
    
    cv2.destroyAllWindows()


def run_demo_loop(cap: cv2.VideoCapture, config: Config = None) -> None:
    """Main demo processing loop."""
    print("Starting demo loop...")
    print("Press 'q' to quit, 'r' to reset detector")
    
    # Create detector
    detector = create_detector("development" if config is None else 
                               ("raspberry_pi" if config.camera.use_picamera else "development"))
    
    # Check for model file
    model_path = Path("vision/checkpoints/chessboard_cnn.pt")
    if model_path.exists():
        print(f"Loading model from {model_path}")
        detector.load_model(str(model_path))
    else:
        print(f"Warning: Model not found at {model_path}")
        print("Running in tracking-only mode (will need manual corner initialization)")
    
    # FPS tracking
    fps_counter = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or camera error")
            break
        
        start_time = time.perf_counter()
        
        # Process frame
        result = detector.process_frame(frame)
        
        # Calculate FPS
        elapsed = time.perf_counter() - start_time
        fps_counter.append(1.0 / elapsed)
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        fps = np.mean(fps_counter)
        
        # Draw debug overlay
        display = draw_debug_overlay(
            frame,
            result.corners,
            result.state.value,
            result.confidence,
            fps,
            result.inference_time_ms
        )
        
        # Draw grid if corners detected
        if result.corners is not None:
            display = draw_grid_overlay(display, result.corners)
        
        # Show
        cv2.imshow("Chessboard Detection Demo", display)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            print("Reset detector")
    
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Demo for chessboard detection and tracking system"
    )
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera device ID (default: 0)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file")
    parser.add_argument("--test", action="store_true",
                        help="Run with synthetic test pattern")
    parser.add_argument("--pi", action="store_true",
                        help="Use Raspberry Pi optimized settings")
    
    args = parser.parse_args()
    
    # Select config
    config = Config.for_raspberry_pi() if args.pi else Config.for_development()
    
    if args.test:
        run_demo_test(config)
    elif args.video:
        run_demo_video(args.video, config)
    else:
        camera_id = args.camera if args.camera is not None else 0
        run_demo_camera(camera_id, config)


if __name__ == "__main__":
    main()
