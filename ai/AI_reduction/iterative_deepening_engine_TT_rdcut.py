import time
import hashlib
import random
import math
import sys
import os

# Ajouter le dossier parent au path pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphabeta_engine import AlphaBetaEngine


class TranspositionEntry:
    """Entrée de la table de transposition"""
    def __init__(self, depth, score, flag, best_move=None, age=0):
        self.depth = depth          # Profondeur de recherche
        self.score = score          # Score évalué
        self.flag = flag           # Type: 'exact', 'lower', 'upper'
        self.best_move = best_move  # Meilleur coup trouvé
        self.age = age             # Âge de l'entrée pour le remplacement


class IterativeDeepeningAlphaBeta(AlphaBetaEngine):
    """Version avec approfondissement itératif, gestion du temps, table de transposition et réduction sélective"""
    
    def __init__(self, max_time=5.0, max_depth=10, evaluator=None, tt_size=1000000, 
                 move_reduction_enabled=True, reduction_seed=None):
        super().__init__(max_depth, evaluator)
        self.max_time = max_time
        self.start_time = None
        self.transposition_table = {}  # Hash -> TranspositionEntry
        self.tt_size = tt_size         # Taille max de la table
        self.search_age = 0            # Âge de la recherche courante
        self.tt_hits = 0              # Statistiques TT
        self.tt_misses = 0
        
        # Paramètres de réduction des coups
        self.move_reduction_enabled = move_reduction_enabled
        self.moves_skipped = 0         # Statistiques des coups sautés
        
        # Initialiser le générateur aléatoire
        if reduction_seed is not None:
            random.seed(reduction_seed)
        else:
            random.seed()  # Seed basé sur le temps système
    
    # ========== FONCTIONS DE PROBABILITÉ DE RÉDUCTION (FACILEMENT MODIFIABLES) ==========
    
    def get_reduction_probability(self, move_index, total_moves, depth):
        """
        FONCTION PRINCIPALE DE PROBABILITÉ DE RÉDUCTION - MODIFIEZ ICI POUR TESTER
        
        Args:
            move_index: Position du coup dans la liste ordonnée (0 = premier coup)
            total_moves: Nombre total de coups légaux
            depth: Profondeur actuelle de recherche
            
        Returns:
            float: Probabilité de supprimer ce coup (0.0 = jamais, 1.0 = toujours)
        """
        if not self.move_reduction_enabled or total_moves <= 3:
            return 0.0  # Ne pas réduire si moins de 4 coups
        
        # Choisir la stratégie de réduction (changez ici pour tester)
        return self._exponential_reduction(move_index, total_moves, depth)
        # return self._linear_reduction(move_index, total_moves, depth)
        # return self._quadratic_reduction(move_index, total_moves, depth)
        # return self._depth_adaptive_reduction(move_index, total_moves, depth)
    
    def _exponential_reduction(self, move_index, total_moves, depth):
        """Réduction exponentielle: très peu de chances au début, forte augmentation à la fin"""
        if move_index < 2:  # Jamais supprimer les 2 premiers coups
            return 0.0
        
        # Calculer la position relative (0.0 = début, 1.0 = fin)
        relative_position = move_index / (total_moves - 1)
        
        # Fonction exponentielle: prob = (pos^3) * 0.8
        probability = (relative_position ** 3) * 0.8
        
        # Ajustement selon la profondeur (plus agressif en profondeur)
        depth_factor = min(1.0 + (depth * 0.1), 2.0)
        
        return min(probability * depth_factor, 0.9)  # Max 90%
    
    def _linear_reduction(self, move_index, total_moves, depth):
        """Réduction linéaire: augmentation constante"""
        if move_index < 2:
            return 0.0
        
        relative_position = move_index / (total_moves - 1)
        return min(relative_position * 0.7, 0.8)
    
    def _quadratic_reduction(self, move_index, total_moves, depth):
        """Réduction quadratique: courbe modérée"""
        if move_index < 2:
            return 0.0
        
        relative_position = move_index / (total_moves - 1)
        return min((relative_position ** 2) * 0.75, 0.85)
    
    def _depth_adaptive_reduction(self, move_index, total_moves, depth):
        """Réduction adaptative selon la profondeur"""
        if move_index < 2:
            return 0.0
        
        relative_position = move_index / (total_moves - 1)
        
        # Plus agressif aux grandes profondeurs
        if depth >= 6:
            return min((relative_position ** 2) * 0.9, 0.9)
        elif depth >= 4:
            return min((relative_position ** 1.5) * 0.7, 0.8)
        else:
            return min(relative_position * 0.5, 0.6)
    
    def should_skip_move(self, move_index, total_moves, depth):
        """Détermine si un coup doit être sauté selon la probabilité"""
        probability = self.get_reduction_probability(move_index, total_moves, depth)
        return random.random() < probability
    
    # ========== FIN DES FONCTIONS DE RÉDUCTION ==========
        
    def get_best_move_with_time_limit(self, chess):
        """
        Trouve le meilleur coup avec une limite de temps.
        Utilise l'approfondissement itératif avec table de transposition.
        """
        self.start_time = time.time()
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        self.tt_hits = 0
        self.tt_misses = 0
        self.moves_skipped = 0  # Reset statistiques de réduction
        self.search_age += 1  # Nouvelle recherche
        
        legal_moves = self._get_all_legal_moves(chess)
        if not legal_moves:
            return None
        
        best_move = legal_moves[0]  # Coup de sécurité
        
        # Approfondissement itératif
        for depth in range(1, self.max_depth + 1):
            if time.time() - self.start_time > self.max_time:
                print(f"Temps limite atteint à la profondeur {depth-1}")
                break
                
            try:
                current_best = self._search_at_depth(chess, depth)
                if current_best:
                    best_move = current_best
                    elapsed = time.time() - self.start_time
                    tt_efficiency = self.tt_hits / (self.tt_hits + self.tt_misses) * 100 if (self.tt_hits + self.tt_misses) > 0 else 0
                    reduction_info = f", Moves skipped: {self.moves_skipped}" if self.move_reduction_enabled else ""
                    print(f"Depth {depth}: Best move = {self._format_move(best_move)}, "
                          f"Time: {elapsed:.2f}s, Nodes: {self.nodes_evaluated}, "
                          f"TT hits: {self.tt_hits} ({tt_efficiency:.1f}%){reduction_info}")
            except TimeoutError:
                print(f"Recherche interrompue à la profondeur {depth}")
                break
        
        print(f"Table de transposition: {len(self.transposition_table)} entrées")
        return best_move
    
    def _search_at_depth(self, chess, target_depth):
        """Recherche à une profondeur donnée avec table de transposition"""
        # Consulter la table de transposition pour le meilleur coup
        position_hash = self._get_position_hash(chess)
        _, tt_best_move = self._probe_tt(position_hash, target_depth, float('-inf'), float('inf'))
        
        legal_moves = self._get_all_legal_moves(chess)
        ordered_moves = self._order_moves_with_tt(chess, legal_moves, tt_best_move)
        
        best_move = None
        current_player_is_white = chess.white_to_move
        best_score = float('-inf') if current_player_is_white else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in ordered_moves:
            # Vérifier le temps restant
            if time.time() - self.start_time > self.max_time:
                raise TimeoutError("Temps limite atteint")
            
            chess.move_piece(move[0], move[1], promotion=move[2])
            
            if current_player_is_white:  # Le joueur actuel était blanc
                score = self.alphabeta_with_timeout(chess, target_depth - 1, alpha, beta, False)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:  # Le joueur actuel était noir
                score = self.alphabeta_with_timeout(chess, target_depth - 1, alpha, beta, True)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
            
            chess.undo_move()
            
            if beta <= alpha:
                break
        
        # Stocker le résultat dans la table de transposition
        flag = 'exact'  # Au niveau racine, c'est toujours exact
        self._store_tt_entry(position_hash, target_depth, best_score, flag, best_move)
        
        return best_move
    
    def alphabeta_with_timeout(self, chess, depth, alpha, beta, is_maximizing):
        """Version d'alpha-beta avec vérification du temps et table de transposition"""
        # Vérifier le temps restant
        if time.time() - self.start_time > self.max_time:
            raise TimeoutError("Temps limite atteint")
        
        return self.alphabeta_with_tt(chess, depth, alpha, beta, is_maximizing)
    
    def _get_position_hash(self, chess):
        """Génère un hash unique pour la position actuelle"""
        # Créer une représentation de la position
        position_str = ""
        
        # Ajouter l'état de chaque case
        for square in range(64):
            piece = self._get_piece_at(chess, square)
            position_str += piece if piece else '.'
        
        # Ajouter qui doit jouer
        position_str += 'W' if chess.white_to_move else 'B'
        
        # Ajouter les droits de roque
        if hasattr(chess, 'castling_rights'):
            for right in ['K', 'Q', 'k', 'q']:
                position_str += right if chess.castling_rights.get(right, False) else '-'
        
        # Ajouter en passant si disponible
        if hasattr(chess, 'en_passant_square') and chess.en_passant_square is not None:
            position_str += f"ep{chess.en_passant_square}"
        
        # Générer le hash
        return hashlib.md5(position_str.encode()).hexdigest()
    
    def _store_tt_entry(self, position_hash, depth, score, flag, best_move=None):
        """Stocke une entrée dans la table de transposition"""
        # Gestion de la taille de la table
        if len(self.transposition_table) >= self.tt_size:
            # Remplacer l'entrée la plus ancienne
            oldest_hash = min(self.transposition_table.keys(), 
                             key=lambda h: self.transposition_table[h].age)
            del self.transposition_table[oldest_hash]
        
        # Stocker la nouvelle entrée
        entry = TranspositionEntry(depth, score, flag, best_move, self.search_age)
        self.transposition_table[position_hash] = entry
    
    def _probe_tt(self, position_hash, depth, alpha, beta):
        """Interroge la table de transposition"""
        if position_hash not in self.transposition_table:
            self.tt_misses += 1
            return None, None
        
        entry = self.transposition_table[position_hash]
        self.tt_hits += 1
        
        # Vérifier si la profondeur est suffisante
        if entry.depth >= depth:
            if entry.flag == 'exact':
                return entry.score, entry.best_move
            elif entry.flag == 'lower' and entry.score >= beta:
                return entry.score, entry.best_move
            elif entry.flag == 'upper' and entry.score <= alpha:
                return entry.score, entry.best_move
        
        # Retourner au moins le meilleur coup même si le score n'est pas utilisable
        return None, entry.best_move
    
    def alphabeta_with_tt(self, chess, depth, alpha, beta, is_maximizing):
        """
        Algorithme alpha-beta avec table de transposition.
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
        
        # Ordonner les coups avec priorité au meilleur coup TT
        ordered_moves = self._order_moves_with_tt(chess, legal_moves, tt_best_move)
        
        # Appliquer la réduction sélective des coups
        if self.move_reduction_enabled and depth > 1:  # Ne pas réduire à la profondeur 1
            filtered_moves = []
            for i, move in enumerate(ordered_moves):
                if not self.should_skip_move(i, len(ordered_moves), depth):
                    filtered_moves.append(move)
                else:
                    self.moves_skipped += 1
            ordered_moves = filtered_moves
            
            # S'assurer qu'il reste au moins un coup
            if not ordered_moves and legal_moves:
                ordered_moves = [legal_moves[0]]  # Garder au moins le premier coup
        
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
    
    def _order_moves_with_tt(self, chess, moves, tt_best_move):
        """
        Ordonne les coups en donnant la priorité au meilleur coup de la table de transposition.
        """
        if not tt_best_move:
            return self._order_moves(chess, moves)
        
        # Séparer le meilleur coup TT des autres
        tt_move_found = False
        other_moves = []
        
        for move in moves:
            if (move[0] == tt_best_move[0] and 
                move[1] == tt_best_move[1] and 
                move[2] == tt_best_move[2]):
                tt_move_found = True
            else:
                other_moves.append(move)
        
        # Ordonner les autres coups normalement
        ordered_others = self._order_moves(chess, other_moves)
        
        # Mettre le coup TT en premier s'il existe
        if tt_move_found:
            return [tt_best_move] + ordered_others
        else:
            return ordered_others
    
    def clear_tt(self):
        """Vide la table de transposition"""
        self.transposition_table.clear()
        self.search_age = 0
        print("Table de transposition vidée")
    
    # ========== MÉTHODES DE CONFIGURATION DE LA RÉDUCTION ==========
    
    def enable_move_reduction(self, enabled=True):
        """Active ou désactive la réduction des coups"""
        self.move_reduction_enabled = enabled
        status = "activée" if enabled else "désactivée"
        print(f"Réduction des coups {status}")
    
    def set_reduction_seed(self, seed):
        """Change la graine aléatoire pour des tests reproductibles"""
        random.seed(seed)
        print(f"Graine aléatoire définie à: {seed}")
    
    def get_reduction_stats(self):
        """Retourne les statistiques de réduction"""
        return {
            'moves_skipped': self.moves_skipped,
            'reduction_enabled': self.move_reduction_enabled
        }
    
    def print_final_stats(self):
        """Affiche les statistiques finales complètes"""
        tt_efficiency = self.tt_hits / (self.tt_hits + self.tt_misses) * 100 if (self.tt_hits + self.tt_misses) > 0 else 0
        
        print(f"\n=== STATISTIQUES FINALES ===")
        print(f"Nœuds évalués: {self.nodes_evaluated}")
        print(f"Branches élaguées: {self.pruned_branches}")
        print(f"TT hits: {self.tt_hits} ({tt_efficiency:.1f}%)")
        print(f"TT misses: {self.tt_misses}")
        print(f"Entrées TT: {len(self.transposition_table)}")
        
        if self.move_reduction_enabled:
            print(f"Coups sautés: {self.moves_skipped}")
            reduction_rate = (self.moves_skipped / max(1, self.nodes_evaluated)) * 100
            print(f"Taux de réduction: {reduction_rate:.1f}%")
        else:
            print("Réduction des coups: désactivée")
        
        print("================================\n")