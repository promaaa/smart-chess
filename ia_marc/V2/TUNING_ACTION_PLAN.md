# Phase 5 : Texel Tuning - Plan d'Action Simplifié

## Option A : Tuning Complet (10-20 heures)

### Prérequis
- ✅ Moteur fonctionnel (déjà fait !)
- ⏱️ Temps disponible : 10-20 heures
- 💾 Espace disque : ~1GB pour dataset

### Étapes

#### 1. Génération du Dataset (5-8 heures)
```bash
cd /Users/promaa/Documents/code/smart-chess/ia_marc/V2
mkdir -p tuning

# Créer le script de génération
cat > tuning/generate_dataset.py << 'EOF'
import sys
sys.path.insert(0, '..')
from engine_main import ChessEngine
import chess
import json
import random

engine = ChessEngine()
engine.set_level("Club")

dataset = []
for game_num in range(200):  # 200 parties
    board = chess.Board()
    positions = []
    
    while not board.is_game_over():
        if board.fullmove_number > 100:  # Limit moves
            break
        
        try:
            move = engine.get_move(board, time_limit=0.3)
            if move is None: break
            
            # Save position every 2 moves (skip opening)
            if board.fullmove_number > 5 and board.fullmove_number % 2 == 0:
                positions.append(board.fen())
            
            board.push(move)
        except:
            break
    
    # Game result
    result = board.result()
    game_result = 1.0 if result == "1-0" else 0.0 if result == "0-1" else 0.5
    
    # Sample 30 positions per game
    for fen in random.sample(positions, min(30, len(positions))):
        dataset.append({'fen': fen, 'result': game_result})
    
    if (game_num + 1) % 10 == 0:
        print(f"Games: {game_num + 1}/200, Positions: {len(dataset)}")

# Save dataset
with open('dataset.json', 'w') as f:
    json.dump(dataset, f)
print(f"Dataset saved: {len(dataset)} positions")
EOF

# Lancer la génération (prend 5-8 heures)
python tuning/generate_dataset.py
```

#### 2. Utiliser un Tuner Externe (2-4 heures)

**Option recommandée : Gedas' Texel Tuner**

```bash
# Installer le tuner
cd /Users/promaa/Documents/code/smart-chess
git clone https://github.com/GediminasMasaitis/texel-tuner
cd texel-tuner

# Convertir notre dataset au format EPD
cd /Users/promaa/Documents/code/smart-chess/ia_marc/V2/tuning
python << 'EOF'
import json
import chess

with open('dataset.json') as f:
    dataset = json.load(f)

with open('dataset.epd', 'w') as f:
    for entry in dataset:
        board = chess.Board(entry['fen'])
        result_str = str(entry['result'])
        f.write(f"{board.epd()} c0 \"{result_str}\";\n")
print("Dataset converted to EPD format")
EOF

# Lancer le tuning
python tune.py --input tuning/dataset.epd --output tuning/optimized_weights.txt
```

#### 3. Appliquer les Poids Optimisés (1 heure)

```bash
# Éditer manuellement engine_brain.py avec les nouveaux poids
# Les poids seront dans tuning/optimized_weights.txt

# Tester la différence
python ai_comparison/compare_ais.py
```

---

## Option B : Tuning Rapide avec Dataset Public (2-3 heures)

### Utiliser un dataset existant

```bash
cd /Users/promaa/Documents/code/smart-chess/ia_marc/V2/tuning

# Télécharger un dataset public
wget https://github.com/official-stockfish/books/raw/master/epd/quiet-labeled.epd

# Ou utiliser Lichess database (plus petit échantillon)
# 1. Aller sur https://database.lichess.org/
# 2. Télécharger un mois récent (format PGN)
# 3. Extraire ~10K positions

# Lancer le tuning directement
python tune.py --input quiet-labeled.epd --output optimized_weights.txt
```

---

## Option C : Tuning Manuel Simplifié (1-2 heures)

### Ajuster manuellement quelques paramètres clés

```python
# Tester différentes valeurs de pièces
# Dans engine_brain.py, lignes 27-28

# Test 1 : Valeurs PeSTO (actuelles)
MG_VALS = [82, 337, 365, 477, 1025, 0]
EG_VALS = [94, 281, 297, 512, 936, 0]

# Test 2 : Valeurs Stockfish
MG_VALS = [124, 781, 825, 1276, 2538, 0]
EG_VALS = [206, 854, 915, 1380, 2682, 0]

# Test 3 : Valeurs moyennes
MG_VALS = [100, 320, 330, 500, 900, 0]
EG_VALS = [100, 320, 330, 500, 900, 0]

# Pour chaque test :
# 1. Jouer 20 parties contre Stockfish niveau 5
# 2. Mesurer le score (win/draw/loss)
# 3. Garder la meilleure configuration
```

**Script de test rapide :**

```bash
cd /Users/promaa/Documents/code/smart-chess

# Tester config actuelle
python ai_comparison/compare_v2_stockfish.py --games 20 --stockfish-level 5

# Noter le score (ex: 8/20 = 40%)

# Modifier MG_VALS dans engine_brain.py

# Re-tester
python ai_comparison/compare_v2_stockfish.py --games 20 --stockfish-level 5

# Noter le nouveau score (ex: 10/20 = 50%)
# Si meilleur, garder !
```

---

## 🎯 Recommandation

**Pour démarrer rapidement :**
1. ✅ **Option C (Manuel)** : 1-2 heures, gain +10-30 ELO
   - Tester 3-4 configurations de valeurs de pièces
   - Mesurer avec parties contre Stockfish
   - Garder la meilleure

**Pour un gain optimal :**
2. 🚀 **Option B (Dataset Public)** : 2-3 heures, gain +50-100 ELO
   - Utiliser un dataset existant (quiet-labeled.epd)
   - Lancer Gedas' tuner
   - Appliquer les résultats

**Pour un tuning personnalisé :**
3. 🏆 **Option A (Complet)** : 10-20 heures, gain +60-120 ELO
   - Générer dataset via self-play
   - Tuner spécifiquement pour IA-Marc
   - Meilleur résultat mais plus long

---

## 📝 Notes

- Le tuning n'est PAS obligatoire - le moteur est déjà très fort !
- Gain estimé : +10 à +120 ELO selon méthode
- Peut être fait plus tard
- Nécessite validation par tests

**Le moteur actuel (~2100-2400 ELO) est déjà excellent pour le RPi 5 !**
