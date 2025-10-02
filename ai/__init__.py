"""
Moteurs d'échecs - Package AI

Ce package contient différents types de moteurs d'échecs :
- BaseChessEngine : Classe de base avec méthodes communes
- AlphaBetaEngine : Moteur alpha-beta classique avec élagage
- IterativeDeepeningAlphaBeta : Moteur avec approfondissement itératif et gestion du temps

Usage:
    from ai.alphabeta_engine import AlphaBetaEngine
    from ai.iterative_deepening_engine import IterativeDeepeningAlphaBeta
    from ai.base_engine import BaseChessEngine
    
    # Ou pour importer tous les moteurs :
    from ai import *

Exemples:
    # Moteur alpha-beta simple
    engine = AlphaBetaEngine(max_depth=4)
    best_move = engine.get_best_move(chess_position)
    
    # Moteur avec limite de temps
    time_engine = IterativeDeepeningAlphaBeta(max_time=5.0, max_depth=8)
    best_move = time_engine.get_best_move_with_time_limit(chess_position)
"""

from .base_engine import BaseChessEngine
from .alphabeta_engine import AlphaBetaEngine
from .iterative_deepening_engine import IterativeDeepeningAlphaBeta

__all__ = [
    'BaseChessEngine',
    'AlphaBetaEngine', 
    'IterativeDeepeningAlphaBeta'
]

__version__ = '1.0.0'