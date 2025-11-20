"""
IA-Marc V2 - Configuration et Niveaux de Difficulté
====================================================

Ce module définit tous les niveaux de difficulté, personnalités de jeu,
et paramètres de configuration du moteur d'échecs.

Optimisé pour Raspberry Pi 5 (4 cores, 8GB RAM)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# NIVEAUX DE DIFFICULTÉ (ELO)
# ============================================================================


@dataclass
class DifficultyLevel:
    """Définit un niveau de difficulté complet."""

    name: str
    elo: int
    depth_limit: int  # Profondeur maximale de recherche
    time_limit: float  # Temps maximum par coup (secondes)
    error_rate: float  # Probabilité d'erreur (0.0 = parfait, 0.5 = 50% erreurs)
    use_opening_book: bool  # Utiliser le livre d'ouvertures
    contempt: int  # Facteur de mépris (éviter les nulles)
    reduction_factor: float  # Facteur pour LMR (Late Move Reduction)

    # Paramètres avancés
    random_move_chance: float = 0.0  # Chance de jouer un coup aléatoire
    blunder_threshold: int = 200  # Seuil de gaffe (centipawns)
    time_variance: float = 0.1  # Variance du temps de réflexion

    def __repr__(self):
        return f"<Level {self.name}: ELO {self.elo}, Depth {self.depth_limit}>"


# Définition de tous les niveaux
DIFFICULTY_LEVELS = {
    # NIVEAU 1: ELO 200
    "LEVEL1": DifficultyLevel(
        name="Niveau 1 (Novice)", elo=200, depth_limit=1, time_limit=0.1,
        error_rate=0.8, use_opening_book=False, contempt=0, reduction_factor=0.5,
        random_move_chance=0.5, blunder_threshold=800, time_variance=0.5
    ),
    # NIVEAU 2: ELO 400
    "LEVEL2": DifficultyLevel(
        name="Niveau 2 (Débutant)", elo=400, depth_limit=1, time_limit=0.2,
        error_rate=0.6, use_opening_book=False, contempt=0, reduction_factor=0.5,
        random_move_chance=0.3, blunder_threshold=600, time_variance=0.4
    ),
    # NIVEAU 3: ELO 600
    "LEVEL3": DifficultyLevel(
        name="Niveau 3", elo=600, depth_limit=2, time_limit=0.3,
        error_rate=0.4, use_opening_book=False, contempt=0, reduction_factor=0.6,
        random_move_chance=0.2, blunder_threshold=500, time_variance=0.3
    ),
    # NIVEAU 4: ELO 800
    "LEVEL4": DifficultyLevel(
        name="Niveau 4", elo=800, depth_limit=2, time_limit=0.5,
        error_rate=0.3, use_opening_book=True, contempt=0, reduction_factor=0.6,
        random_move_chance=0.1, blunder_threshold=400, time_variance=0.2
    ),
    # NIVEAU 5: ELO 1000
    "LEVEL5": DifficultyLevel(
        name="Niveau 5 (Amateur)", elo=1000, depth_limit=3, time_limit=1.0,
        error_rate=0.2, use_opening_book=True, contempt=5, reduction_factor=0.7,
        random_move_chance=0.05, blunder_threshold=300, time_variance=0.15
    ),
    # NIVEAU 6: ELO 1200
    "LEVEL6": DifficultyLevel(
        name="Niveau 6", elo=1200, depth_limit=3, time_limit=1.5,
        error_rate=0.1, use_opening_book=True, contempt=10, reduction_factor=0.7,
        random_move_chance=0.02, blunder_threshold=200, time_variance=0.1
    ),
    # NIVEAU 7: ELO 1400
    "LEVEL7": DifficultyLevel(
        name="Niveau 7 (Club)", elo=1400, depth_limit=4, time_limit=2.0,
        error_rate=0.05, use_opening_book=True, contempt=15, reduction_factor=0.8,
        random_move_chance=0.0, blunder_threshold=150, time_variance=0.05
    ),
    # NIVEAU 8: ELO 1600
    "LEVEL8": DifficultyLevel(
        name="Niveau 8", elo=1600, depth_limit=5, time_limit=3.0,
        error_rate=0.02, use_opening_book=True, contempt=20, reduction_factor=0.9,
        random_move_chance=0.0, blunder_threshold=100, time_variance=0.02
    ),
    # NIVEAU 9: ELO 1800
    "LEVEL9": DifficultyLevel(
        name="Niveau 9 (Expert)", elo=1800, depth_limit=6, time_limit=4.0,
        error_rate=0.0, use_opening_book=True, contempt=25, reduction_factor=1.0,
        random_move_chance=0.0, blunder_threshold=50, time_variance=0.0
    ),
}


# ============================================================================
# PERSONNALITÉS DE JEU
# ============================================================================


@dataclass
class Personality:
    """Définit un style de jeu spécifique."""

    name: str
    description: str

    # Bonus d'évaluation (en centipawns)
    bonus_attack: int = 0  # Bonus pour les coups agressifs
    bonus_defense: int = 0  # Bonus pour la défense
    bonus_center: int = 0  # Bonus pour le contrôle du centre
    bonus_mobility: int = 0  # Bonus pour la mobilité
    bonus_king_safety: int = 0  # Bonus pour la sécurité du roi
    bonus_pawn_structure: int = 0  # Bonus pour la structure de pions
    bonus_material: int = 0  # Bonus pour le matériel

    # Modificateurs de recherche
    capture_preference: float = 1.0  # Préférence pour les captures
    check_preference: float = 1.0  # Préférence pour les échecs
    castle_preference: float = 1.0  # Préférence pour le roque

    def __repr__(self):
        return f"<Personality: {self.name}>"


PERSONALITIES = {
    "EQUILIBRE": Personality(
        name="Équilibré",
        description="Joue de manière équilibrée, sans préférence particulière",
        bonus_attack=0,
        bonus_defense=0,
        bonus_center=10,
        bonus_mobility=5,
        bonus_king_safety=10,
        bonus_pawn_structure=10,
        bonus_material=0,
    ),
    "AGRESSIF": Personality(
        name="Agressif",
        description="Attaque constamment, cherche les complications tactiques",
        bonus_attack=50,
        bonus_defense=-20,
        bonus_center=15,
        bonus_mobility=20,
        bonus_king_safety=-10,
        bonus_pawn_structure=-5,
        bonus_material=-10,
        capture_preference=1.3,
        check_preference=1.5,
        castle_preference=0.7,
    ),
    "DEFENSIF": Personality(
        name="Défensif",
        description="Joue solidement, évite les risques",
        bonus_attack=-20,
        bonus_defense=50,
        bonus_center=5,
        bonus_mobility=0,
        bonus_king_safety=40,
        bonus_pawn_structure=20,
        bonus_material=10,
        capture_preference=0.8,
        check_preference=0.7,
        castle_preference=1.5,
    ),
    "POSITIONNEL": Personality(
        name="Positionnel",
        description="Mise sur le contrôle de l'espace et la structure",
        bonus_attack=0,
        bonus_defense=10,
        bonus_center=30,
        bonus_mobility=25,
        bonus_king_safety=15,
        bonus_pawn_structure=30,
        bonus_material=0,
        capture_preference=0.9,
        check_preference=0.8,
        castle_preference=1.2,
    ),
    "TACTIQUE": Personality(
        name="Tactique",
        description="Cherche les combinaisons et les coups brillants",
        bonus_attack=40,
        bonus_defense=0,
        bonus_center=10,
        bonus_mobility=30,
        bonus_king_safety=5,
        bonus_pawn_structure=0,
        bonus_material=-5,
        capture_preference=1.5,
        check_preference=1.8,
        castle_preference=0.8,
    ),
    "MATERIALISTE": Personality(
        name="Matérialiste",
        description="Privilégie le gain de matériel avant tout",
        bonus_attack=10,
        bonus_defense=10,
        bonus_center=5,
        bonus_mobility=5,
        bonus_king_safety=10,
        bonus_pawn_structure=5,
        bonus_material=50,
        capture_preference=1.8,
        check_preference=0.9,
        castle_preference=1.0,
    ),
}


# ============================================================================
# CONFIGURATION DU MOTEUR
# ============================================================================


@dataclass
class EngineConfig:
    """Configuration complète du moteur d'échecs."""

    # Niveau et personnalité
    difficulty_level: DifficultyLevel = field(
        default_factory=lambda: DIFFICULTY_LEVELS["LEVEL7"]
    )
    personality: Personality = field(default_factory=lambda: PERSONALITIES["EQUILIBRE"])

    # Transposition Table
    tt_size_mb: int = 256  # Taille du cache (MB)
    tt_enabled: bool = True

    # Parallélisation
    threads: int = 4  # Nombre de threads (Lazy SMP)
    use_parallel: bool = True

    # Opening Book
    use_opening_book: bool = True
    opening_book_path: str = "ia_marc/V2/data/openings.json"
    opening_variety: int = 3  # Nombre de variantes à considérer

    # Recherche
    use_iterative_deepening: bool = True
    use_aspiration_windows: bool = True
    aspiration_window_size: int = 50

    # Élagage
    use_null_move_pruning: bool = True
    null_move_reduction: int = 2  # R factor for null move
    use_late_move_reduction: bool = True
    lmr_threshold: int = 4  # Commence LMR après N coups

    # Move Ordering
    use_killer_moves: bool = True
    killer_slots: int = 2  # Nombre de killers par profondeur
    use_history_heuristic: bool = True

    # Quiescence
    use_quiescence: bool = True
    quiescence_depth_limit: int = 10

    # Évaluation
    use_advanced_eval: bool = True
    eval_mobility: bool = True
    eval_pawn_structure: bool = True
    eval_king_safety: bool = True
    eval_piece_coordination: bool = False  # Coûteux, désactivé par défaut

    # Debug et logging
    verbose: bool = False
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    show_pv: bool = True  # Afficher la principale variation
    show_stats: bool = True  # Afficher les statistiques

    # Limites de sécurité
    max_time_per_move: float = 60.0  # Temps max absolu (secondes)
    max_nodes: int = 10_000_000  # Nombre max de nœuds
    max_depth: int = 30  # Profondeur max absolue

    # Raspberry Pi optimizations
    use_pypy: bool = False  # Détecté automatiquement
    memory_limit_mb: int = 1024  # Limite mémoire globale

    def __post_init__(self):
        """Validation et ajustements post-initialisation."""
        # Détection de PyPy
        import sys

        self.use_pypy = hasattr(sys, "pypy_version_info")

        # Ajuster les threads selon les capacités
        import os

        cpu_count = os.cpu_count() or 4
        if self.threads > cpu_count:
            logger.warning(
                f"Threads ({self.threads}) > CPU cores ({cpu_count}). Ajustement à {cpu_count}."
            )
            self.threads = cpu_count

        # Validation des limites
        if self.tt_size_mb > 2048:
            logger.warning(
                f"TT size très élevée ({self.tt_size_mb}MB). Limité à 2048MB."
            )
            self.tt_size_mb = 2048

        if self.tt_size_mb < 16:
            logger.warning(
                f"TT size très faible ({self.tt_size_mb}MB). Augmenté à 16MB."
            )
            self.tt_size_mb = 16

    def set_level(self, level_name: str) -> None:
        """Change le niveau de difficulté."""
        level_upper = level_name.upper()
        if level_upper in DIFFICULTY_LEVELS:
            self.difficulty_level = DIFFICULTY_LEVELS[level_upper]
            logger.info(f"Niveau changé: {self.difficulty_level}")
        else:
            available = ", ".join(DIFFICULTY_LEVELS.keys())
            raise ValueError(f"Niveau inconnu: {level_name}. Disponibles: {available}")

    def set_elo(self, elo: int) -> None:
        """Change le niveau selon un ELO approximatif."""
        # Trouver le niveau le plus proche
        closest = min(DIFFICULTY_LEVELS.values(), key=lambda x: abs(x.elo - elo))
        self.difficulty_level = closest
        logger.info(f"ELO {elo} → Niveau: {closest.name} (ELO {closest.elo})")

    def set_personality(self, personality_name: str) -> None:
        """Change la personnalité de jeu."""
        personality_upper = personality_name.upper()
        if personality_upper in PERSONALITIES:
            self.personality = PERSONALITIES[personality_upper]
            logger.info(f"Personnalité changée: {self.personality.name}")
        else:
            available = ", ".join(PERSONALITIES.keys())
            raise ValueError(
                f"Personnalité inconnue: {personality_name}. Disponibles: {available}"
            )

    def optimize_for_rpi5(self) -> None:
        """Optimise les paramètres pour Raspberry Pi 5."""
        logger.info("Optimisation pour Raspberry Pi 5...")
        self.threads = 4
        self.tt_size_mb = 512  # Équilibre performance/mémoire
        self.use_parallel = True
        self.use_null_move_pruning = True
        self.use_late_move_reduction = True
        self.use_aspiration_windows = True
        logger.info("Configuration optimisée pour RPi 5 (4 cores, 512MB TT)")

    def optimize_for_speed(self) -> None:
        """Optimise pour la vitesse maximale (peut réduire la force)."""
        logger.info("Optimisation pour vitesse maximale...")
        self.use_opening_book = False
        self.use_advanced_eval = False
        self.eval_pawn_structure = False
        self.eval_king_safety = False
        self.quiescence_depth_limit = 5
        logger.info("Configuration optimisée pour vitesse")

    def optimize_for_strength(self) -> None:
        """Optimise pour la force maximale (peut être plus lent)."""
        logger.info("Optimisation pour force maximale...")
        self.use_opening_book = True
        self.use_advanced_eval = True
        self.eval_mobility = True
        self.eval_pawn_structure = True
        self.eval_king_safety = True
        self.quiescence_depth_limit = 10
        self.tt_size_mb = 1024  # Plus de cache
        logger.info("Configuration optimisée pour force")

    def get_config_dict(self) -> Dict[str, Any]:
        """Retourne la configuration sous forme de dictionnaire."""
        return {
            "level": self.difficulty_level.name,
            "elo": self.difficulty_level.elo,
            "personality": self.personality.name,
            "tt_size_mb": self.tt_size_mb,
            "threads": self.threads,
            "use_parallel": self.use_parallel,
            "use_opening_book": self.use_opening_book,
            "use_pypy": self.use_pypy,
        }

    def __repr__(self):
        return f"<EngineConfig: {self.difficulty_level.name} ({self.difficulty_level.elo} ELO), {self.personality.name}>"


# ============================================================================
# CONFIGURATION PAR DÉFAUT
# ============================================================================


def get_default_config() -> EngineConfig:
    """Retourne une configuration par défaut optimisée pour RPi 5."""
    config = EngineConfig()
    config.optimize_for_rpi5()
    return config


def get_config_for_level(level_name: str) -> EngineConfig:
    """Retourne une configuration pré-configurée pour un niveau."""
    config = get_default_config()
    config.set_level(level_name)
    return config


# ============================================================================
# UTILITAIRES
# ============================================================================


def list_levels() -> None:
    """Affiche tous les niveaux disponibles."""
    print("\n=== Niveaux de Difficulté Disponibles ===\n")
    for key, level in DIFFICULTY_LEVELS.items():
        print(
            f"{level.name:15} - ELO {level.elo:4} - Depth {level.depth_limit:2} - Time {level.time_limit:4.1f}s"
        )
    print()


def list_personalities() -> None:
    """Affiche toutes les personnalités disponibles."""
    print("\n=== Personnalités de Jeu Disponibles ===\n")
    for key, personality in PERSONALITIES.items():
        print(f"{personality.name:15} - {personality.description}")
    print()


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=== Test de la Configuration ===\n")

    # Lister les niveaux
    list_levels()

    # Lister les personnalités
    list_personalities()

    # Créer une configuration par défaut
    print("Configuration par défaut:")
    config = get_default_config()
    print(config)
    print(config.get_config_dict())
    print()

    # Tester différents niveaux
    print("Test des niveaux:")
    config.set_level("DEBUTANT")
    print(f"  → {config.difficulty_level}")

    config.set_elo(1800)
    print(f"  → {config.difficulty_level}")
    print()

    # Tester les personnalités
    print("Test des personnalités:")
    config.set_personality("AGRESSIF")
    print(f"  → {config.personality}")

    config.set_personality("DEFENSIF")
    print(f"  → {config.personality}")
    print()

    # Optimisations
    print("Optimisations:")
    config.optimize_for_speed()
    print(f"  → Vitesse: TT={config.tt_size_mb}MB, Advanced={config.use_advanced_eval}")

    config.optimize_for_strength()
    print(f"  → Force: TT={config.tt_size_mb}MB, Advanced={config.use_advanced_eval}")

    print("\n✅ Tests de configuration réussis!")
