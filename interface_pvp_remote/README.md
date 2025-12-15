# Interface PvP Remote

Interface web permettant à un joueur en ligne de jouer aux échecs contre un adversaire utilisant le plateau physique intelligent.

## Fonctionnement

```
┌──────────────────┐          ┌──────────────┐          ┌─────────────────┐
│   Joueur Web     │◄────────►│   Serveur    │◄────────►│ Plateau Physique│
│   (Blancs)       │ WebSocket│   Python     │   I2C    │    (Noirs)      │
└──────────────────┘          └──────────────┘          └─────────────────┘
```

### Flux de jeu
1. **Coup du joueur web** → Le serveur allume les LEDs des cases de départ/arrivée sur le plateau
2. **Le joueur physique** voit les LEDs, déplace la pièce correspondante
3. **Le joueur physique joue** → Les capteurs reed détectent le mouvement
4. **Le coup s'affiche** sur l'interface web du joueur en ligne

## Lancement

### Mode simulation (pour développement/test)
```bash
./start.sh --simulation
```
→ Interface: http://localhost:8081

### Mode hardware (sur Raspberry Pi)
```bash
./start.sh
```
→ Interface: http://<IP_PI>:8081

## Structure

| Fichier | Description |
|---------|-------------|
| `server.py` | Serveur WebSocket + HTTP |
| `hardware_bridge.py` | Interface I2C (LEDs + capteurs) |
| `game_manager.py` | Logique d'échecs (python-chess) |
| `index.html` | Interface web |
| `style.css` | Styles |
| `app.js` | Logique frontend |

## Ports

- **8081**: Interface web (HTTP)
- **8082**: Communication temps réel (WebSocket)

## Configuration des joueurs

Par défaut:
- **Joueur web** = Blancs (joue en premier)
- **Joueur physique** = Noirs
