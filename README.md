# SmartChess

Un échiquier électronique intelligent qui détecte automatiquement la position des pièces et vous permet de jouer contre un adversaire IA puissant.

## Structure du Projet

Le dépôt a été restructuré pour plus de clarté :

- **`ai/`** : Recherche et développement originaux pour l'IA d'échecs.
- **`ia_marc/`** : Moteurs d'IA (V1 et V2), avec livre d'ouvertures.
- **`prototypes/`** : Conception matérielle et firmware.
  - **`echiquier_8x8/`** : Prototype principal 8x8.
    - `firmware/` : Code embarqué.
      - `ia_embarquee/` : Scripts de jeu principaux (`chess_game_v1.py`, `chess_game_v2.py`).
    - `hardware/` : Schémas et PCB KiCad.
  - **`echiquier_2x2/`** : Prototype de test 2x2.


## Démarrage Rapide

Ce guide explique comment lancer l'application principale d'échecs sur un Raspberry Pi ou une machine de développement.

### 1. Installation

Naviguez vers le répertoire de l'IA embarquée (V2) et installez les dépendances :

```bash
# Aller dans le répertoire de l'IA V2
cd ia_marc/V2/

# Créer un environnement virtuel
python3 -m venv venv

# Activer l'environnement virtuel
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Lancer le Jeu

Le script principal vous permet de jouer contre l'IA avec un menu de sélection de difficulté.

```bash
# Aller dans le répertoire des scripts embarqués
cd prototypes/echiquier_8x8/firmware/ia_embarquee/
    - `hardware/` : Schémas et PCB KiCad.

# Lancer le jeu (V2)
python3 chess_game_v2.py
```

### Niveaux de Difficulté IA

L'IA propose 9 niveaux de difficulté, allant de ELO 200 (Novice) à 1800 (Expert).

### Bibliothèque d'Ouvertures (Optionnel)

Pour une IA plus forte dans les ouvertures, téléchargez un fichier de livre Polyglot (ex: `Cerebellum_Light.bin`) et placez-le dans le répertoire `ia_marc/book/`.
