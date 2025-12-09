# Detection module for CNN-based chessboard corner detection

from .model import ChessboardCNN, load_model
from .preprocessing import preprocess_frame, augment_image

__all__ = ['ChessboardCNN', 'load_model', 'preprocess_frame', 'augment_image']
