import time
import hashlib
from iterative_deepening_engine_TT import IterativeDeepeningAlphaBeta as BaseIterativeEngine
from iterative_deepening_engine_TT import TranspositionEntry


class NullMovePruningEngine(BaseIterativeEngine):
    """
    Version avec null-move pruning en plus de toutes les optimisations précédentes.
    
    Le null-move pruning consiste à "passer son tour" et faire une recherche réduite :
    - Si même en donnant l'avantage à l'adversaire on obtient beta cutoff, on peut élaguer
    - Très efficace dans les positions où on a un avantage significatif
    """
    
    def __init__(self, max_time=5.0, max_depth=10, evaluator=None, tt_size=1000000, 
                 null_move_enabled=True, null_move_R=2, null_move_min_depth=3):
        """
        Args:
            null_move_enabled: Active/désactive le null-move pruning
            null_move_R: Réduction de profondeur pour le null-move (généralement 2-3)
            null_move_min_depth: Profondeur minimale pour appliquer null-move
        """
        super().__init__(max_time, max_depth, evaluator, tt_size)
        
        # Paramètres null-move
        self.null_move_enabled = null_move_enabled
        self.null_move_R = null_move_R  # Réduction de profondeur
        self.null_move_min_depth = null_move_min_depth  # Profondeur minimale
        
        # Statistiques null-move
        self.null_move_attempts = 0
        self.null_move_cutoffs = 0
        self.null_move_failures = 0
    
    def get_best_move_with_time_limit(self, chess):
        """Override pour reset les statistiques null-move"""
        self.null_move_attempts = 0
        self.null_move_cutoffs = 0
        self.null_move_failures = 0
        
        return super().get_best_move_with_time_limit(chess)
    
    def alphabeta_with_tt(self, chess, depth, alpha, beta, is_maximizing):
        """
        Version alpha-beta avec null-move pruning intégré
        """
        self.nodes_evaluated += 1
        original_alpha = alpha
        
        # Générer le hash de la position
        position_hash = self._get_position_hash(chess)
        
        # Consulter la table de transposition
        tt_score, tt_best_move = self._probe_tt(position_hash, depth, alpha, beta)
        if tt_score is not None:
            return tt_score
        
        # Condition d'arrêt: profondeur 0 ou fin de partie
        if depth == 0:
            score = self.evaluator.evaluate_position(chess)
            self._store_tt_entry(position_hash, depth, score, 'exact')
            return score
        
        legal_moves = self._get_all_legal_moves(chess)
        
        # Fin de partie (mat ou pat)
        if not legal_moves:
            if chess.is_in_check(chess.white_to_move):
                # Échec et mat
                mate_score = 20000 + depth
                score = -mate_score if is_maximizing else mate_score
                self._store_tt_entry(position_hash, depth, score, 'exact')
                return score
            else:
                # Pat
                self._store_tt_entry(position_hash, depth, 0, 'exact')
                return 0
        
        # ========== NULL-MOVE PRUNING ==========
        # Conditions pour appliquer null-move:
        # 1. Activé
        # 2. Profondeur suffisante
        # 3. Pas en échec (ne peut pas passer si on est attaqué)
        # 4. Fenêtre de recherche valide (pas à la racine avec alpha/beta infinis)
        
        if (self.null_move_enabled and 
            depth >= self.null_move_min_depth and 
            not chess.is_in_check(chess.white_to_move)):  # Conditions simplifiées
            
            self.null_move_attempts += 1
            
            # Faire le null-move: changer simplement le tour sans bouger
            chess.white_to_move = not chess.white_to_move
            
            try:
                # Recherche avec profondeur réduite
                null_depth = max(0, depth - 1 - self.null_move_R)
                
                # Recherche null-move avec fenêtre réduite [beta-1, beta]
                if is_maximizing:
                    null_score = self.alphabeta_with_tt(chess, null_depth, beta - 1, beta, False)
                else:
                    null_score = self.alphabeta_with_tt(chess, null_depth, alpha, alpha + 1, True)
                
                # Annuler le null-move
                chess.white_to_move = not chess.white_to_move
                
                # Si le null-move donne un cutoff, on peut élaguer
                if ((is_maximizing and null_score >= beta) or 
                    (not is_maximizing and null_score <= alpha)):
                    
                    self.null_move_cutoffs += 1
                    
                    # Stocker le cutoff dans la TT
                    flag = 'lower' if is_maximizing else 'upper'
                    cutoff_score = beta if is_maximizing else alpha
                    self._store_tt_entry(position_hash, depth, cutoff_score, flag)
                    
                    return cutoff_score
                else:
                    self.null_move_failures += 1
                    
            except Exception:
                # En cas d'erreur, annuler le null-move et continuer normalement
                chess.white_to_move = not chess.white_to_move
                self.null_move_failures += 1
        
        # ========== RECHERCHE NORMALE ==========
        
        # Ordonner les coups avec priorité au meilleur coup TT
        ordered_moves = self._order_moves_with_tt(chess, legal_moves, tt_best_move)
        
        best_move = None
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                chess.move_piece(move[0], move[1], promotion=move[2])
                eval_score = self.alphabeta_with_tt(chess, depth - 1, alpha, beta, False)
                chess.undo_move()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                
                # Élagage beta
                if beta <= alpha:
                    self.pruned_branches += len(ordered_moves) - ordered_moves.index(move) - 1
                    break
            
            # Stocker dans la table de transposition
            if max_eval <= original_alpha:
                flag = 'upper'
            elif max_eval >= beta:
                flag = 'lower'
            else:
                flag = 'exact'
            
            self._store_tt_entry(position_hash, depth, max_eval, flag, best_move)
            return max_eval
            
        else:  # is_minimizing
            min_eval = float('inf')
            for move in ordered_moves:
                chess.move_piece(move[0], move[1], promotion=move[2])
                eval_score = self.alphabeta_with_tt(chess, depth - 1, alpha, beta, True)
                chess.undo_move()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                
                # Élagage alpha
                if beta <= alpha:
                    self.pruned_branches += len(ordered_moves) - ordered_moves.index(move) - 1
                    break
            
            # Stocker dans la table de transposition
            if min_eval <= original_alpha:
                flag = 'upper'
            elif min_eval >= beta:
                flag = 'lower'
            else:
                flag = 'exact'
            
            self._store_tt_entry(position_hash, depth, min_eval, flag, best_move)
            return min_eval
    
    # ========== MÉTHODES DE CONFIGURATION NULL-MOVE ==========
    
    def enable_null_move(self, enabled=True):
        """Active ou désactive le null-move pruning"""
        self.null_move_enabled = enabled
        status = "activé" if enabled else "désactivé"
        print(f"Null-move pruning {status}")
    
    def set_null_move_params(self, R=2, min_depth=3):
        """Configure les paramètres du null-move"""
        self.null_move_R = R
        self.null_move_min_depth = min_depth
        print(f"Null-move: R={R}, profondeur min={min_depth}")
    
    def get_null_move_stats(self):
        """Retourne les statistiques null-move"""
        success_rate = (self.null_move_cutoffs / max(1, self.null_move_attempts)) * 100
        return {
            'attempts': self.null_move_attempts,
            'cutoffs': self.null_move_cutoffs,
            'failures': self.null_move_failures,
            'success_rate': success_rate
        }
    
    def print_final_stats(self):
        """Affiche les statistiques finales complètes avec null-move"""
        tt_efficiency = self.tt_hits / (self.tt_hits + self.tt_misses) * 100 if (self.tt_hits + self.tt_misses) > 0 else 0
        
        print(f"\n=== STATISTIQUES FINALES ===")
        print(f"Nœuds évalués: {self.nodes_evaluated}")
        print(f"Branches élaguées: {self.pruned_branches}")
        print(f"TT hits: {self.tt_hits} ({tt_efficiency:.1f}%)")
        print(f"TT misses: {self.tt_misses}")
        print(f"Entrées TT: {len(self.transposition_table)}")
        
        # Statistiques null-move
        if self.null_move_enabled and self.null_move_attempts > 0:
            success_rate = (self.null_move_cutoffs / self.null_move_attempts) * 100
            efficiency = (self.null_move_cutoffs / max(1, self.nodes_evaluated)) * 100
            
            print(f"=== STATISTIQUES NULL-MOVE ===")
            print(f"Tentatives: {self.null_move_attempts}")
            print(f"Cutoffs: {self.null_move_cutoffs}")
            print(f"Échecs: {self.null_move_failures}")
            print(f"Taux de réussite: {success_rate:.1f}%")
            print(f"Efficacité générale: {efficiency:.1f}%")
        elif self.null_move_enabled:
            print("Null-move: activé mais aucune tentative")
        else:
            print("Null-move: désactivé")
        
        print("================================\n")


# ========== FONCTIONS UTILITAIRES ==========

def create_null_move_engine(time_limit=30, depth_limit=8, enable_null_move=True):
    """
    Factory function pour créer une engine avec null-move pruning
    """
    from evaluator import ChessEvaluator
    
    engine = NullMovePruningEngine(
        max_time=time_limit,
        max_depth=depth_limit,
        evaluator=ChessEvaluator(),
        null_move_enabled=enable_null_move,
        null_move_R=2,  # Réduction standard
        null_move_min_depth=3  # Profondeur minimale standard
    )
    
    return engine


if __name__ == "__main__":
    """Test rapide du null-move pruning"""
    
    print("=== TEST RAPIDE NULL-MOVE PRUNING ===\n")
    
    from Chess import Chess
    from evaluator import ChessEvaluator
    
    # Position de test
    chess = Chess()
    
    # Test avec null-move
    print("Avec null-move pruning:")
    engine_nm = create_null_move_engine(time_limit=10, depth_limit=5)
    
    start = time.time()
    best_move = engine_nm.get_best_move_with_time_limit(chess)
    nm_time = time.time() - start
    
    print(f"Temps: {nm_time:.2f}s")
    print(f"Meilleur coup: {engine_nm._format_move(best_move)}")
    engine_nm.print_final_stats()
    
    # Test sans null-move pour comparaison
    print("Sans null-move pruning:")
    engine_no_nm = NullMovePruningEngine(
        max_time=10,
        max_depth=5,
        evaluator=ChessEvaluator(),
        null_move_enabled=False
    )
    
    start = time.time()
    best_move_no_nm = engine_no_nm.get_best_move_with_time_limit(chess)
    no_nm_time = time.time() - start
    
    print(f"Temps: {no_nm_time:.2f}s")
    print(f"Meilleur coup: {engine_no_nm._format_move(best_move_no_nm)}")
    engine_no_nm.print_final_stats()
    
    # Comparaison
    if nm_time < no_nm_time:
        gain = ((no_nm_time - nm_time) / no_nm_time) * 100
        print(f"🎯 Gain avec null-move: {gain:.1f}%")
    else:
        loss = ((nm_time - no_nm_time) / no_nm_time) * 100
        print(f"⚠️  Perte avec null-move: {loss:.1f}%")