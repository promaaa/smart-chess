#!/usr/bin/env python3
"""
IA-Marc V2 - Main Chess Engine API
===================================

API principale qui intègre tous les modules du moteur d'échecs.
Interface simple et intuitive pour utilisation dans le projet Smart Chess.

Usage:
    from engine_main import ChessEngine

    engine = ChessEngine()
    engine.set_level("Club")
    move = engine.get_move(board, time_limit=3.0)

Auteur: Smart Chess Team
Version: 2.0
"""

import logging
import time
from typing import Dict, Optional, Tuple

import chess
from engine_brain import EvaluationEngine
from engine_config import EngineConfig, get_default_config
from engine_opening import OpeningBook
from engine_search import SearchEngine

# Configuration du logging
logger = logging.getLogger(__name__)


# ============================================================================
# API PRINCIPALE
# ============================================================================


class ChessEngine:
    """
    Moteur d'échecs IA-Marc V2.

    Interface unifiée pour accéder à toutes les fonctionnalités du moteur.
    """

    def __init__(self, config: Optional[EngineConfig] = None, verbose: bool = False):
        """
        Initialise le moteur d'échecs.

        Args:
            config: Configuration personnalisée (optionnel)
            verbose: Mode verbeux pour le debugging
        """
        self.config = config or get_default_config()
        self.verbose = verbose

        # Configuration du logging
        if verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        # Initialisation des modules
        self._init_modules()

        # État
        self.ready = True
        self.searching = False
        self.use_opening_book = True

        logger.info(
            f"IA-Marc V2 initialisé: {self.config.difficulty_level.name} "
            f"({self.config.difficulty_level.elo} ELO)"
        )

    def _init_modules(self):
        """Initialise les modules du moteur."""
        try:
            # Moteur d'évaluation
            self.evaluator = EvaluationEngine()

            # Configurer l'évaluateur selon la personnalité
            personality = self.config.personality
            self.evaluator.configure(
                bonus_mobility=personality.bonus_mobility,
                bonus_pawn_structure=personality.bonus_pawn_structure,
                bonus_king_safety=personality.bonus_king_safety,
                bonus_center=personality.bonus_center,
                contempt=self.config.difficulty_level.contempt,
                use_mobility=self.config.eval_mobility,
                use_pawn_structure=self.config.eval_pawn_structure,
                use_king_safety=self.config.eval_king_safety,
            )

            # Moteur de recherche
            self.searcher = SearchEngine(self.evaluator, self.config)

            # Livre d'ouvertures
            self.opening_book = self._init_opening_book()

            logger.debug("Modules initialisés avec succès")

        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des modules: {e}")
            raise

    def _init_opening_book(self) -> Optional[OpeningBook]:
        """
        Initialise le livre d'ouvertures.
        Essaie d'abord Polyglot, puis JSON par défaut.

        Returns:
            OpeningBook ou None
        """
        # Essayer d'abord Cerebellum (full) pour la meilleure qualité, puis Light
        polyglot_paths = [
            # Cerebellum (full) - 800MB, 2M+ positions - BEST
            "ia_marc/book/Cerebellum.bin",
            # Cerebellum3Merge - 170MB, excellent quality
            "ia_marc/book/Cerebellum3Merge.bin",
            # Cerebellum Light - 157MB, 500k+ positions - EXCELLENT
            "ia_marc/book/Cerebellum_Light.bin",
        ]

        for path in polyglot_paths:
            book = OpeningBook(path, book_type="polyglot")
            if book.load():
                logger.info(f"Livre d'ouvertures Polyglot chargé: {path}")
                self.config.opening_book_path = path
                return book

        # Fallback sur le livre JSON
        json_path = "ia_marc/V2/data/openings.json"
        book = OpeningBook(json_path, book_type="json")
        if book.load():
            logger.info(f"Livre d'ouvertures JSON chargé: {json_path}")
            self.config.opening_book_path = json_path
            return book

        logger.warning("Aucun livre d'ouvertures disponible")
        return None

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def set_level(self, level_name: str):
        """
        Change le niveau de difficulté.

        Args:
            level_name: Nom du niveau (Enfant, Debutant, Amateur, Club,
                       Competition, Expert, Maitre, Maximum)
        """
        try:
            self.config.set_level(level_name)
            logger.info(
                f"Niveau changé: {self.config.difficulty_level.name} "
                f"({self.config.difficulty_level.elo} ELO)"
            )
        except ValueError as e:
            logger.error(f"Niveau invalide: {e}")
            raise

    def set_elo(self, elo: int):
        """
        Change le niveau selon un ELO approximatif.

        Args:
            elo: ELO cible (400-2400)
        """
        self.config.set_elo(elo)
        logger.info(f"ELO configuré: {self.config.difficulty_level.elo}")

    def set_personality(self, personality_name: str):
        """
        Change la personnalité de jeu.

        Args:
            personality_name: Nom de la personnalité (Equilibre, Agressif,
                            Defensif, Positionnel, Tactique, Materialiste)
        """
        try:
            self.config.set_personality(personality_name)

            # Reconfigurer l'évaluateur
            personality = self.config.personality
            self.evaluator.configure(
                bonus_mobility=personality.bonus_mobility,
                bonus_pawn_structure=personality.bonus_pawn_structure,
                bonus_king_safety=personality.bonus_king_safety,
                bonus_center=personality.bonus_center,
                contempt=self.config.difficulty_level.contempt,
            )

            logger.info(f"Personnalité changée: {self.config.personality.name}")

        except ValueError as e:
            logger.error(f"Personnalité invalide: {e}")
            raise

    def configure(self, **kwargs):
        """
        Configure le moteur avec des paramètres personnalisés.

        Args:
            **kwargs: Paramètres de configuration
                     (tt_size_mb, threads, use_parallel, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Configuration mise à jour: {key} = {value}")
            else:
                logger.warning(f"Paramètre de configuration inconnu: {key}")

    # ========================================================================
    # RECHERCHE
    # ========================================================================

    def get_move(
        self, board: chess.Board, time_limit: Optional[float] = None
    ) -> Optional[chess.Move]:
        """
        Obtient le meilleur coup pour une position donnée.

        Args:
            board: Position d'échecs (chess.Board)
            time_limit: Temps maximum en secondes (optionnel)
                       Si None, utilise le temps du niveau configuré

        Returns:
            Meilleur coup trouvé (chess.Move) ou None
        """
        if not self.ready:
            logger.error("Moteur non prêt")
            return None

        if self.searching:
            logger.warning("Recherche déjà en cours")
            return None

        # Vérifier que la position est valide
        if board.is_game_over():
            logger.info("Position terminale, pas de coup à jouer")
            return None

        # Déterminer les paramètres de recherche
        level = self.config.difficulty_level

        if time_limit is None:
            time_limit = level.time_limit

        depth_limit = level.depth_limit

        # Consulter le livre d'ouvertures en premier
        if self.use_opening_book and self.opening_book and len(board.move_stack) < 15:
            book_move = self.opening_book.probe(
                board, elo_level=level.elo, variety=True
            )
            if book_move:
                logger.info(f"Coup du livre d'ouvertures: {book_move}")
                return book_move

        # Appliquer les erreurs intentionnelles pour les bas niveaux
        if level.random_move_chance > 0:
            import random

            if random.random() < level.random_move_chance:
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    logger.debug(f"Coup aléatoire choisi (niveau bas): {move}")
                    return move

        # Lancer la recherche
        self.searching = True
        start_time = time.time()

        try:
            logger.debug(
                f"Début de la recherche: depth={depth_limit}, time={time_limit}s"
            )

            move = self.searcher.search(
                board, time_limit=time_limit, depth_limit=depth_limit
            )

            elapsed = time.time() - start_time
            logger.info(f"Recherche terminée en {elapsed:.2f}s: {move}")

            return move

        except Exception as e:
            logger.error(f"Erreur pendant la recherche: {e}")
            import traceback

            traceback.print_exc()

            # Fallback: retourner un coup légal aléatoire
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return legal_moves[0]
            return None

        finally:
            self.searching = False

    def get_move_with_stats(
        self, board: chess.Board, time_limit: Optional[float] = None
    ) -> Tuple[Optional[chess.Move], Dict]:
        """
        Obtient le meilleur coup avec statistiques détaillées.

        Args:
            board: Position d'échecs
            time_limit: Temps maximum en secondes (optionnel)

        Returns:
            (meilleur_coup, statistiques)
            statistiques contient: depth, score, nodes, nps, time, etc.
        """
        move = self.get_move(board, time_limit)
        stats = self.get_stats()

        return move, stats

    # ========================================================================
    # INFORMATIONS
    # ========================================================================

    def get_stats(self) -> Dict:
        """
        Retourne les statistiques de la dernière recherche.

        Returns:
            Dictionnaire avec les statistiques
        """
        if not hasattr(self, "searcher"):
            return {}

        stats = self.searcher.get_stats()

        # Ajouter les infos de configuration
        stats["level"] = self.config.difficulty_level.name
        stats["elo"] = self.config.difficulty_level.elo
        stats["personality"] = self.config.personality.name

        # Ajouter les stats du livre d'ouvertures
        if self.opening_book:
            stats["opening_book"] = self.opening_book.get_stats()

        return stats

    def is_ready(self) -> bool:
        """
        Vérifie si le moteur est prêt.

        Returns:
            True si prêt, False sinon
        """
        return self.ready and not self.searching

    def get_config(self) -> Dict:
        """
        Retourne la configuration actuelle.

        Returns:
            Dictionnaire de configuration
        """
        return self.config.get_config_dict()

    # ========================================================================
    # CONTRÔLE
    # ========================================================================

    def stop(self):
        """Arrête la recherche en cours."""
        if self.searching and hasattr(self, "searcher"):
            self.searcher.stop()
            logger.info("Recherche arrêtée par l'utilisateur")

    def reset(self):
        """Réinitialise le moteur (vide les caches, etc.)."""
        try:
            # Réinitialiser les modules
            self._init_modules()

            # Vider les caches
            if hasattr(self.evaluator, "clear_cache"):
                self.evaluator.clear_cache()

            if hasattr(self.searcher, "tt") and self.searcher.tt:
                self.searcher.tt.clear()

            # Réinitialiser les stats du livre d'ouvertures
            if self.opening_book:
                self.opening_book.clear_stats()

            logger.info("Moteur réinitialisé")

        except Exception as e:
            logger.error(f"Erreur lors de la réinitialisation: {e}")

    # ========================================================================
    # LIVRE D'OUVERTURES
    # ========================================================================

    def enable_opening_book(self, enable: bool = True):
        """
        Active ou désactive le livre d'ouvertures.

        Args:
            enable: True pour activer, False pour désactiver
        """
        self.use_opening_book = enable
        logger.info(f"Livre d'ouvertures: {'activé' if enable else 'désactivé'}")

    def set_opening_book(self, book_path: str, book_type: str = "auto"):
        """
        Change le livre d'ouvertures.

        Args:
            book_path: Chemin vers le fichier
            book_type: Type de livre ("auto", "json", "polyglot")
        """
        book = OpeningBook(book_path, book_type=book_type)
        if book.load():
            self.opening_book = book
            logger.info(f"Livre d'ouvertures changé: {book_path}")
        else:
            logger.error(f"Impossible de charger le livre: {book_path}")

    # ========================================================================
    # OPTIMISATIONS
    # ========================================================================

    def optimize_for_rpi5(self):
        """Optimise la configuration pour Raspberry Pi 5."""
        self.config.optimize_for_rpi5()
        self._init_modules()
        logger.info("Configuration optimisée pour Raspberry Pi 5")

    def optimize_for_speed(self):
        """Optimise pour la vitesse maximale (peut réduire la force)."""
        self.config.optimize_for_speed()
        self._init_modules()
        logger.info("Configuration optimisée pour vitesse")

    def optimize_for_strength(self):
        """Optimise pour la force maximale (peut être plus lent)."""
        self.config.optimize_for_strength()
        self._init_modules()
        logger.info("Configuration optimisée pour force")


# ============================================================================
# EXEMPLES D'UTILISATION
# ============================================================================


def example_basic():
    """Exemple basique d'utilisation."""
    print("=== Exemple Basique ===\n")

    # Créer le moteur
    engine = ChessEngine()

    # Configurer le niveau
    engine.set_level("LEVEL4")

    # Position initiale
    board = chess.Board()

    # Obtenir un coup
    print("Recherche en cours...")
    move = engine.get_move(board, time_limit=3.0)

    print(f"Meilleur coup: {move}")

    # Afficher les stats
    stats = engine.get_stats()
    print(f"Profondeur: {stats['depth']}")
    print(f"Nœuds: {stats['nodes']:,}")
    print(f"NPS: {stats['nps']:,}")


def example_advanced():
    """Exemple avancé avec configuration personnalisée."""
    print("\n=== Exemple Avancé ===\n")

    # Créer le moteur avec configuration
    engine = ChessEngine(verbose=True)

    # Optimiser pour RPi 5
    engine.optimize_for_rpi5()

    # Configurer le niveau et la personnalité
    engine.set_elo(1800)
    engine.set_personality("Agressif")

    # Position tactique
    board = chess.Board(
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    )

    # Recherche avec statistiques
    move, stats = engine.get_move_with_stats(board, time_limit=5.0)

    print(f"\nRésultat:")
    print(f"  Coup: {move}")
    print(f"  Score: {stats['score']}")
    print(f"  Profondeur: {stats['depth']}")
    print(f"  Temps: {stats['time']:.2f}s")
    print(f"  NPS: {stats['nps']:,}")


def example_all_levels():
    """Exemple testant tous les niveaux."""
    print("\n=== Test de Tous les Niveaux ===\n")

    engine = ChessEngine()
    board = chess.Board()

    levels = [
        "LEVEL1",
        "LEVEL2",
        "LEVEL3",
        "LEVEL4",
        "LEVEL5",
        "LEVEL6",
        "LEVEL7",
    ]

    for level_name in levels:
        engine.set_level(level_name)

        start = time.time()
        move = engine.get_move(board)
        elapsed = time.time() - start

        stats = engine.get_stats()

        print(
            f"{level_name:12} | ELO {stats['elo']:4} | "
            f"Depth {stats['depth']:2} | Time {elapsed:.2f}s | "
            f"Move {move}"
        )


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=== Tests du Moteur IA-Marc V2 ===\n")

    # Test 1: Exemple basique
    example_basic()

    # Test 2: Exemple avancé
    example_advanced()

    # Test 3: Tous les niveaux
    example_all_levels()

    print("\n✅ Tests terminés!")
