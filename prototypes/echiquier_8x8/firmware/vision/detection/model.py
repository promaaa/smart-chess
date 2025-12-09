"""
CNN model architecture for chessboard corner detection.

Architecture: Classic CNN pyramid with Max Pooling
- Input: RGB image (configurable, default 224x224)
- Output: 8 values representing 4 corner coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
- Target: ~96% accuracy, ~30ms inference on Raspberry Pi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from pathlib import Path


class ChessboardCNN(nn.Module):
    """
    CNN for detecting chessboard corners in an image.
    
    Architecture:
        Conv2D(32) -> ReLU -> MaxPool(2)
        Conv2D(64) -> ReLU -> MaxPool(2)
        Conv2D(128) -> ReLU -> MaxPool(2)
        Conv2D(256) -> ReLU -> MaxPool(2)
        Flatten -> Dense(512) -> Dense(256) -> Dense(8)
    
    Output represents 4 corner points: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    Coordinates are normalized to [0, 1] range.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.input_size = input_size
        
        # Convolutional layers with BatchNorm for stable training
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened size after conv layers
        # After 4 max pools: size / 16
        h, w = input_size[0] // 16, input_size[1] // 16
        flatten_size = 256 * h * w
        
        # Fully connected layers for corner regression
        self.fc1 = nn.Linear(flatten_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 8)  # 4 corners * 2 coordinates
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, height, width)
            
        Returns:
            Tensor of shape (batch, 8) with normalized corner coordinates
        """
        # Convolutional feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))  # Output in [0, 1] range
        
        return x
    
    def predict_corners(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict corner coordinates and reshape to (batch, 4, 2).
        
        Args:
            x: Input tensor of shape (batch, 3, height, width)
            
        Returns:
            Tensor of shape (batch, 4, 2) with corner coordinates
        """
        coords = self.forward(x)
        return coords.view(-1, 4, 2)


class ChessboardCNNLite(nn.Module):
    """
    Lightweight version of ChessboardCNN for faster inference on Raspberry Pi.
    Uses fewer filters and smaller fully connected layers.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (160, 160)):
        super().__init__()
        self.input_size = input_size
        
        # Smaller convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 3 max pools: size / 8
        h, w = input_size[0] // 8, input_size[1] // 8
        flatten_size = 64 * h * w
        
        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, 8)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        return x
    
    def predict_corners(self, x: torch.Tensor) -> torch.Tensor:
        coords = self.forward(x)
        return coords.view(-1, 4, 2)


def load_model(
    model_path: str,
    input_size: Tuple[int, int] = (224, 224),
    use_lite: bool = False,
    device: str = "cpu"
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the saved model weights
        input_size: Expected input size for the model
        use_lite: Whether to use the lightweight model
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Loaded model ready for inference
    """
    if use_lite:
        model = ChessboardCNNLite(input_size)
    else:
        model = ChessboardCNN(input_size)
    
    path = Path(model_path)
    if path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def create_model(
    input_size: Tuple[int, int] = (224, 224),
    use_lite: bool = False
) -> nn.Module:
    """
    Create a new model instance for training.
    
    Args:
        input_size: Expected input size for the model
        use_lite: Whether to use the lightweight model
        
    Returns:
        New model instance
    """
    if use_lite:
        return ChessboardCNNLite(input_size)
    else:
        return ChessboardCNN(input_size)
