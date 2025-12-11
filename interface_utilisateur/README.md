# Smart Chess - Interface Utilisateur

Simulateur d'interface pour jouer aux échecs contre l'IA Marc V2.

## Lancement rapide

```bash
./start.sh
```

Ou manuellement :
```bash
python3 server.py
# Ouvrir http://localhost:8080
```

## Fichiers

| Fichier | Description |
|---------|-------------|
| `index.html` | Interface HTML |
| `style.css` | Styles |
| `app.js` | Logique de l'application |
| `chess-ui.js` | Gestion de l'échiquier |
| `ai-engine.js` | Connexion à l'IA Marc V2 |
| `server.py` | Serveur backend Python |
| `start.sh` | Script de lancement |

## Niveaux

| Niveau | ELO | Description |
|--------|-----|-------------|
| 1 | 400 | Débutant |
| 2 | 600 | Novice |
| 3 | 800 | Occasionnel |
| 4 | 1000 | Amateur |
| 5 | 1200 | Club débutant |
| 6 | 1400 | Club intermédiaire |
| 7 | 1600 | Club avancé |
| 8 | 1700 | Maximum |

## Dépendances

- Python 3
- python-chess (`pip install python-chess`)
