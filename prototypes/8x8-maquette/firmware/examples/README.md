# Exemples d'utilisation de l'interface IA-Échiquier

Ce dossier contient des exemples pratiques d'intégration entre un moteur d'IA d'échecs et l'échiquier électronique.

## Fichiers

### `ai_integration_example.py`

Programme de démonstration complet montrant différents scénarios d'utilisation de l'interface ChessboardLED.

#### Démos incluses

1. **Affichage simple** - Montrer un coup suggéré par l'IA
2. **Animation de coup** - Animation visuelle du déplacement
3. **Suggestions multiples** - Afficher plusieurs options avec priorités
4. **Assistance joueur** - Comparer le coup du joueur avec l'IA
5. **Communication JSON** - Format d'échange de données
6. **Coups spéciaux** - Roque, promotion, etc.
7. **Luminosité** - Ajustement dynamique de l'intensité
8. **Analyse continue** - Suivi d'une partie complète
9. **Mode interactif** - Tester manuellement des coups

## Installation

### Prérequis

```bash
pip3 install adafruit-circuitpython-ht16k33
pip3 install adafruit-circuitpython-tca9548a
```

### Optionnel (pour IA réelle)

```bash
pip3 install python-chess
pip3 install stockfish
```

## Utilisation

### Lancer une démonstration

```bash
cd /path/to/firmware/examples
sudo python3 ai_integration_example.py
```

Le programme affiche un menu interactif :

```
Choisissez une démonstration:

  1. Affichage simple
  2. Animation de coup
  3. Suggestions multiples
  4. Assistance joueur
  5. Communication JSON
  6. Coups spéciaux
  7. Luminosité
  8. Analyse continue
  9. Mode interactif

  all. Toutes les démos
  q. Quitter
```

### Mode interactif

Choisissez l'option `9` pour tester manuellement des coups UCI :

```
[Vous] Coup UCI: e2e4
[Échiquier] Affichage de e2e4

[Vous] Coup UCI: g1f3
[Échiquier] Affichage de g1f3

[Vous] Coup UCI: clear
[Échiquier] Effacement de l'échiquier

[Vous] Coup UCI: q
```

## Intégrer avec une vraie IA

### Exemple avec Stockfish

```python
import chess
import chess.engine
from chess_interface import ChessboardLED

# Initialiser l'échiquier
board_led = ChessboardLED(verbose=True)

# Initialiser Stockfish
engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

# Position de départ
board = chess.Board()

# Obtenir le meilleur coup
result = engine.play(board, chess.engine.Limit(time=2.0))
best_move_uci = result.move.uci()

# Afficher sur l'échiquier
board_led.display_move(best_move_uci)

# Nettoyer
engine.quit()
board_led.clear()
```

### Exemple avec python-chess

```python
import chess
from chess_interface import ChessboardLED

board_led = ChessboardLED()
board = chess.Board()

# Obtenir tous les coups légaux
legal_moves = [move.uci() for move in board.legal_moves]

# Afficher les 3 premiers coups possibles
board_led.display_suggestions(legal_moves[:3], duration=10.0)
```

## Format UCI

Tous les exemples utilisent le format **UCI** (Universal Chess Interface) :

| Format | Description | Exemple |
|--------|-------------|---------|
| `e2e4` | Coup normal | Pion e2 vers e4 |
| `g1f3` | Coup de pièce | Cavalier g1 vers f3 |
| `e1g1` | Petit roque | Roi e1 vers g1 |
| `e7e8q` | Promotion | Pion e7 vers e8, promotion en dame |

## API Rapide

```python
from chess_interface import ChessboardLED

board = ChessboardLED()

# Afficher un coup
board.display_move("e2e4")

# Afficher plusieurs cases
board.display_squares(["e2", "e4", "d2", "d4"])

# Animation
board.animate_move("g1f3", duration=2.0)

# Clignotement
board.blink_move("d2d4", duration=5.0, frequency=2.0)

# Suggestions (meilleur coup clignote)
board.display_suggestions(["e2e4", "d2d4", "g1f3"], duration=8.0)

# Ajuster luminosité
board.set_brightness(0.5)  # 50%

# Éteindre
board.clear()
```

## Communication réseau (avancé)

### Serveur TCP simple

```python
import socket
import json
from chess_interface import ChessboardLED

board = ChessboardLED()
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 5000))
server.listen(1)

print("Serveur échiquier en écoute sur le port 5000...")

while True:
    client, addr = server.accept()
    data = client.recv(1024).decode()
    
    try:
        message = json.loads(data)
        if message['type'] == 'move':
            board.display_move(message['uci'])
            response = {'status': 'ok', 'displayed': True}
        else:
            response = {'status': 'error', 'message': 'Unknown type'}
    except Exception as e:
        response = {'status': 'error', 'message': str(e)}
    
    client.send(json.dumps(response).encode())
    client.close()
```

### Client (IA)

```python
import socket
import json

def send_move_to_board(uci_move):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 5000))
    
    message = {'type': 'move', 'uci': uci_move}
    client.send(json.dumps(message).encode())
    
    response = json.loads(client.recv(1024).decode())
    client.close()
    
    return response

# Envoyer un coup
result = send_move_to_board('e2e4')
print(result)  # {'status': 'ok', 'displayed': True}
```

## Dépannage

### Problème : "No module named 'chess_interface'"

**Solution** : Assurez-vous que `chess_interface.py` est dans le répertoire parent :

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/firmware"
```

### Problème : "Permission denied" sur I2C

**Solution** : Exécutez avec `sudo` :

```bash
sudo python3 ai_integration_example.py
```

### Problème : LEDs ne s'allument pas

**Vérifications** :
1. TCA9548A correctement branché (canaux 4 et 5)
2. HT16K33 alimentés (3.3V ou 5V)
3. Adresses I2C correctes (0x70 par défaut)
4. Connexions SDA/SCL du Raspberry Pi

Tester avec :
```bash
sudo i2cdetect -y 1
```

## Documentation complète

- **Spécification** : `../CHESS_INTERFACE_SPEC.md`
- **Module principal** : `../chess_interface.py`
- **Tests matériels** : `../tests/`

## Auteur

Smart Chess Project - 2024