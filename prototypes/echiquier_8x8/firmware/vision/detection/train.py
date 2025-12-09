"""
Training script for the chessboard corner detection CNN.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json
import time

from .model import ChessboardCNN, ChessboardCNNLite, create_model
from .preprocessing import preprocess_frame, augment_image, normalize_corners


class ChessboardDataset(Dataset):
    """
    Dataset for chessboard corner detection.
    
    Expected data format:
        annotations.json: {
            "images": [
                {
                    "filename": "image_001.jpg",
                    "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                },
                ...
            ]
        }
        
    Corners should be in order: top-left, top-right, bottom-right, bottom-left
    """
    
    def __init__(
        self,
        data_dir: str,
        input_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
        augment_config: Optional[dict] = None
    ):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.augment = augment
        self.augment_config = augment_config
        
        # Load annotations
        annotations_path = self.data_dir / "annotations.json"
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        self.samples = data['images']
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        img_path = self.data_dir / "images" / sample['filename']
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Get corners
        corners = np.array(sample['corners'], dtype=np.float32)
        
        # Apply augmentation if enabled
        if self.augment:
            image, corners = augment_image(image, corners, self.augment_config)
        
        # Normalize corners to [0, 1]
        h, w = image.shape[:2]
        corners = normalize_corners(corners, (h, w))
        
        # Preprocess image
        processed = preprocess_frame(image, self.input_size)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(processed)
        corners_tensor = torch.from_numpy(corners.flatten())  # Shape: (8,)
        
        return image_tensor, corners_tensor


def corner_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.05
) -> float:
    """
    Calculate accuracy as percentage of corners within threshold distance.
    
    Args:
        pred: Predicted corners (batch, 8)
        target: Ground truth corners (batch, 8)
        threshold: Distance threshold in normalized coordinates
        
    Returns:
        Accuracy as float between 0 and 1
    """
    pred = pred.view(-1, 4, 2)
    target = target.view(-1, 4, 2)
    
    distances = torch.norm(pred - target, dim=2)  # (batch, 4)
    within_threshold = (distances < threshold).float()
    
    return within_threshold.mean().item()


def train_model(
    data_dir: str,
    output_dir: str,
    input_size: Tuple[int, int] = (224, 224),
    use_lite: bool = False,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    device: str = "auto"
) -> Dict:
    """
    Train the chessboard corner detection model.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path to save checkpoints and logs
        input_size: Model input size
        use_lite: Use lightweight model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        validation_split: Fraction of data for validation
        device: Device to train on ('auto', 'cpu', or 'cuda')
        
    Returns:
        Training history dict
    """
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    full_dataset = ChessboardDataset(data_dir, input_size, augment=True)
    
    # Split into train/val
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Disable augmentation for validation
    val_dataset.dataset.augment = False
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Create model
    model = create_model(input_size, use_lite)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for images, corners in train_loader:
            images = images.to(device)
            corners = corners.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, corners)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += corner_accuracy(outputs, corners)
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for images, corners in val_loader:
                images = images.to(device)
                corners = corners.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, corners)
                
                val_loss += loss.item()
                val_acc += corner_accuracy(outputs, corners)
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'input_size': input_size,
                'use_lite': use_lite
            }, output_path / "best_model.pt")
            print(f"  -> Saved new best model")
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, output_path / f"checkpoint_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'use_lite': use_lite,
        'history': history
    }, output_path / "final_model.pt")
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return history


def benchmark_inference(
    model_path: str,
    input_size: Tuple[int, int] = (224, 224),
    use_lite: bool = False,
    device: str = "cpu",
    num_iterations: int = 100
) -> Dict:
    """
    Benchmark model inference speed.
    
    Args:
        model_path: Path to trained model
        input_size: Input size
        use_lite: Whether to use lite model
        device: Device to benchmark on
        num_iterations: Number of inference iterations
        
    Returns:
        Dict with timing statistics
    """
    from .model import load_model
    
    model = load_model(model_path, input_size, use_lite, device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    results = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99))
    }
    
    print(f"Inference Benchmark ({device})")
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Std:  {results['std_ms']:.2f} ms")
    print(f"  P95:  {results['p95_ms']:.2f} ms")
    print(f"  P99:  {results['p99_ms']:.2f} ms")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train chessboard corner detection model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lite", action="store_true", help="Use lightweight model")
    parser.add_argument("--input_size", type=int, default=224)
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        input_size=(args.input_size, args.input_size),
        use_lite=args.lite,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
