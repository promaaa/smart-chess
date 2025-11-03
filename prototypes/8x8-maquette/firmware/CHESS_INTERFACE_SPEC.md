# Spécification de l'Interface IA-Échiquier

## Vue d'ensemble

Ce document définit le format de communication standardisé entre le moteur d'IA d'échecs et le système électronique de l'échiquier pour l'affichage visuel des coups.

---

## Format de Communication : UCI (Universal Chess Interface)

### Pourquoi UCI ?

Le format **UCI** est le standard universel utilisé par tous les moteurs d'échecs modernes (Stockfish, Leela Chess Zero, etc.). Il est simple, non-ambigu et facile à parser.

### Structure d'un coup UCI

Un coup UCI est représenté par une chaîne de **4 ou 5 caractères** :

```
[case_départ][case_arrivée][promotion?]
```

#### Exemples

| Coup UCI | Description | Cases à illuminer |
|----------|-------------|-------------------|
| `e2e4` | Pion e2 vers e4 | e2 (départ) + e4 (arrivée) |
| `e7e5` | Pion e7 vers e5 | e7 (départ) + e5 (arrivée) |
| `g1f3` | Cavalier g1 vers f3 | g1 (départ) + f3 (arrivée) |
| `e1g1` | Petit roque blanc | e1 (départ) + g1 (arrivée) |
| `e1c1` | Grand roque blanc | e1 (départ) + c1 (arrivée) |
| `e7e8q` | Promotion en dame | e7 (départ) + e8 (arrivée) |
| `e7e8n` | Promotion en cavalier | e7 (départ) + e8 (arrivée) |
| `e7e8r` | Promotion en tour | e7 (départ) + e8 (arrivée) |
| `e7e8b` | Promotion en fou | e7 (départ) + e8 (arrivée) |

#### Lettres de promotion

- `q` : Queen (Dame)
- `r` : Rook (Tour)
- `b` : Bishop (Fou)
- `n` : Knight (Cavalier)

---

## Notation des cases

Les cases utilisent la **notation algébrique standard** :

- **Colonnes** : `a` à `h` (de gauche à droite, côté blancs)
- **Rangées** : `1` à `8` (de bas en haut, côté blancs)

```
8  a8 b8 c8 d8 e8 f8 g8 h8
7  a7 b7 c7 d7 e7 f7 g7 h7
6  a6 b6 c6 d6 e6 f6 g6 h6
5  a5 b5 c5 d5 e5 f5 g5 h5
4  a4 b4 c4 d4 e4 f4 g4 h4
3  a3 b3 c3 d3 e3 f3 g3 h3
2  a2 b2 c2 d2 e2 f2 g2 h2
1  a1 b1 c1 d1 e1 f1 g1 h1
   a  b  c  d  e  f  g  h
```

---

## Format d'échange de données

### JSON (Recommandé)

Format structuré pour la communication entre l'IA et l'échiquier.

#### Coup simple

```json
{
  "type": "move",
  "uci": "e2e4",
  "from": "e2",
  "to": "e4",
  "piece": "pawn",
  "player": "white",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Coup avec promotion

```json
{
  "type": "move",
  "uci": "e7e8q",
  "from": "e7",
  "to": "e8",
  "piece": "pawn",
  "promotion": "queen",
  "player": "black",
  "timestamp": "2024-01-15T10:30:05Z"
}
```

#### Suggestion de coup (IA)

```json
{
  "type": "suggestion",
  "uci": "e2e4",
  "from": "e2",
  "to": "e4",
  "evaluation": 0.35,
  "depth": 20,
  "best_line": ["e2e4", "e7e5", "g1f3"],
  "timestamp": "2024-01-15T10:30:10Z"
}
```

#### Coups multiples (variantes)

```json
{
  "type": "suggestions",
  "moves": [
    {
      "uci": "e2e4",
      "from": "e2",
      "to": "e4",
      "evaluation": 0.35,
      "rank": 1
    },
    {
      "uci": "d2d4",
      "from": "d2",
      "to": "d4",
      "evaluation": 0.28,
      "rank": 2
    },
    {
      "uci": "g1f3",
      "from": "g1",
      "to": "f3",
      "evaluation": 0.20,
      "rank": 3
    }
  ],
  "timestamp": "2024-01-15T10:30:15Z"
}
```

### Texte simple (Alternatif)

Pour une communication simple via fichier ou pipe :

```
MOVE e2e4
MOVE g1f3
SUGGESTION e2e4
CLEAR
```

---

## API de l'Échiquier Électronique

### Commandes supportées

| Commande | Paramètres | Description |
|----------|-----------|-------------|
| `display_move(uci)` | uci: string | Affiche un coup UCI |
| `display_squares(squares)` | squares: list | Illumine des cases spécifiques |
| `clear()` | - | Éteint toutes les LEDs |
| `blink(squares, duration)` | squares: list, duration: float | Fait clignoter des cases |
| `set_color(square, color)` | square: string, color: tuple | Change la couleur d'une case (si RGB) |
| `animate_move(from, to)` | from: string, to: string | Animation de déplacement |

### Exemples Python

```python
from chess_interface import ChessboardLED

board = ChessboardLED()

# Afficher un coup simple
board.display_move("e2e4")

# Afficher plusieurs suggestions
board.display_squares(["e2", "d2", "g1"], color="blue")
board.blink("e2")  # Meilleur coup clignote

# Animation de coup
board.animate_move("e2", "e4", duration=1.0)

# Nettoyer
board.clear()
```

---

## Modes d'affichage

### Mode 1 : Affichage simple

- **Case de départ** : LEDs allumées fixes
- **Case d'arrivée** : LEDs allumées fixes

### Mode 2 : Affichage avec distinction

- **Case de départ** : LEDs clignotantes lentes (1 Hz)
- **Case d'arrivée** : LEDs fixes

### Mode 3 : Animation de déplacement

1. Allumer case de départ
2. Animation progressive (déplacement visuel)
3. Allumer case d'arrivée
4. Éteindre case de départ

### Mode 4 : Suggestions multiples

- **Meilleur coup** : LEDs clignotantes rapides (2 Hz)
- **Deuxième choix** : LEDs clignotantes lentes (1 Hz)
- **Autres coups** : LEDs fixes faibles

---

## Temporisation et timing

### Durées recommandées

| Action | Durée | Notes |
|--------|-------|-------|
| Affichage statique | ∞ | Jusqu'à action utilisateur |
| Suggestion temporaire | 5-10 s | Auto-effacement |
| Clignotement rapide | 0.5 s (2 Hz) | Meilleur coup |
| Clignotement lent | 1 s (1 Hz) | Alternative |
| Animation | 1-2 s | Déplacement visuel |
| Délai entre coups | 0.5 s | Éviter la confusion |

---

## Cas spéciaux

### Roque

Le roque implique deux pièces mais un seul coup UCI :

- **Petit roque blanc** : `e1g1`
  - Afficher : e1 (roi) → g1
  - Note : La tour h1→f1 n'est pas affichée
  
- **Grand roque blanc** : `e1c1`
  - Afficher : e1 (roi) → c1
  
- **Petit roque noir** : `e8g8`
- **Grand roque noir** : `e8c8`

**Option avancée** : Afficher aussi le déplacement de la tour (4 cases illuminées)

### Prise en passant

La prise en passant est un coup normal en UCI :

- Exemple : `e5d6` (pion blanc e5 prend en passant en d6)
- Afficher : e5 → d6 (comme un coup normal)
- Le pion capturé en d5 n'est pas indiqué visuellement

### Promotion

Le 5ème caractère indique la pièce de promotion :

- `e7e8q` : Afficher e7 → e8
- Optionnel : Indiquer la pièce de promotion par un motif LED spécifique

---

## Protocole de communication

### Via socket TCP

```
Serveur : Raspberry Pi (échiquier)
Port : 5000
Format : JSON ou texte

Client (IA) envoie :
{"type": "move", "uci": "e2e4"}

Serveur répond :
{"status": "ok", "displayed": true}
```

### Via fichier partagé

```
Fichier : /tmp/chess_moves.json
Format : JSON

L'IA écrit le coup
L'échiquier lit et affiche
L'échiquier écrit un acquittement
```

### Via MQTT (IoT)

```
Topic : chess/moves
Payload : {"uci": "e2e4"}

Topic : chess/status
Payload : {"ready": true}
```

---

## Validation des coups

### Format UCI valide

Un coup UCI valide doit respecter :

1. Longueur : 4 ou 5 caractères
2. Cases : `[a-h][1-8][a-h][1-8][qrbn]?`
3. Les deux cases doivent être différentes
4. Respect des règles d'échecs (optionnel, selon implémentation)

### Regex de validation

```python
import re

UCI_PATTERN = r'^[a-h][1-8][a-h][1-8][qrbn]?$'

def is_valid_uci(move: str) -> bool:
    return bool(re.match(UCI_PATTERN, move))
```

---

## Messages d'erreur

### Codes d'erreur standardisés

| Code | Message | Description |
|------|---------|-------------|
| `E001` | Invalid UCI format | Format du coup incorrect |
| `E002` | Square out of bounds | Case hors de l'échiquier |
| `E003` | Hardware error | Problème matériel (LED/I2C) |
| `E004` | Timeout | Délai d'attente dépassé |
| `E005` | Invalid promotion | Pièce de promotion invalide |

### Exemple de réponse d'erreur

```json
{
  "status": "error",
  "code": "E001",
  "message": "Invalid UCI format",
  "received": "e2x4",
  "timestamp": "2024-01-15T10:30:20Z"
}
```

---

## Exemples d'utilisation

### Scénario 1 : Jouer un coup

```python
# L'IA calcule le meilleur coup
best_move = engine.get_best_move()  # Retourne "e2e4"

# Afficher sur l'échiquier
board.display_move(best_move)

# Attendre que le joueur joue
time.sleep(5)

# Effacer
board.clear()
```

### Scénario 2 : Montrer plusieurs options

```python
# L'IA propose 3 coups
suggestions = [
    {"uci": "e2e4", "eval": 0.35},
    {"uci": "d2d4", "eval": 0.28},
    {"uci": "g1f3", "eval": 0.20}
]

# Afficher le meilleur en clignotant
board.blink_move(suggestions[0]["uci"], frequency=2)

# Afficher les alternatives en fixe
for move in suggestions[1:]:
    board.display_move(move["uci"], intensity=0.5)
```

### Scénario 3 : Mode entraînement

```python
# Afficher le coup du joueur
board.display_move(player_move, color="green")
time.sleep(2)

# Comparer avec le meilleur coup de l'IA
if player_move != best_move:
    board.clear()
    board.display_move(best_move, color="blue")
    # Afficher "Voici le meilleur coup"
```

---

## Référence rapide

### Conversion case → coordonnées

```python
def square_to_coords(square: str) -> tuple:
    """Convertit e4 en (4, 4)"""
    col = ord(square[0]) - ord('a')  # 0-7
    row = int(square[1]) - 1          # 0-7
    return (row, col)

def coords_to_square(row: int, col: int) -> str:
    """Convertit (4, 4) en e4"""
    return chr(ord('a') + col) + str(row + 1)
```

### Parsing d'un coup UCI

```python
def parse_uci(uci: str) -> dict:
    """Parse un coup UCI"""
    return {
        "from": uci[0:2],
        "to": uci[2:4],
        "promotion": uci[4] if len(uci) == 5 else None
    }
```

---

## Annexes

### Bibliothèques Python recommandées

- `python-chess` : Validation et gestion des règles d'échecs
- `stockfish` : Interface avec le moteur Stockfish
- `chess.engine` : Support UCI générique

### Exemple d'intégration complète

Voir le fichier `chess_interface.py` pour l'implémentation de référence.

---

**Version** : 1.0  
**Date** : 2024  
**Auteur** : Smart Chess Project