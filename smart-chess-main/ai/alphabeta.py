"""
FICHIER DE COMPATIBILITÉ - DÉPRÉCIÉ

Ce fichier est maintenant déprécié. Les classes ont été séparées dans des fichiers distincts :
- BaseChessEngine -> base_engine.py
- AlphaBetaEngine -> alphabeta_engine.py  
- IterativeDeepeningAlphaBeta -> iterative_deepening_engine.py
- Tests -> test_engines.py

Pour la compatibilité, ce fichier importe toutes les classes depuis leurs nouveaux emplacements.
Il est recommandé d'utiliser directement les nouveaux modules.
"""

# Imports pour la compatibilité
from alphabeta_engine import AlphaBetaEngine
from ai.Old_AI.iterative_deepening_engine import IterativeDeepeningAlphaBeta
from ai.Old_AI.test_engines import test_alphabeta, test_iterative_deepening, compare_engines

# Réexporter pour la compatibilité
__all__ = ['AlphaBetaEngine', 'IterativeDeepeningAlphaBeta']

if __name__ == "__main__":
    test_alphabeta()
    #test_iterative_deepening()
    #compare_engines()
