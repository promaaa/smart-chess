# Guide d'entraînement du réseau de neurones d'évaluation

## 🎯 Métriques à surveiller

### 1. RMSE (Root Mean Square Error)
- **Définition**: Erreur quadratique moyenne entre prédictions et évaluations réelles
- **Échelle**: Normalisé (divisé par 1000), donc 1.0 = 1000 centipawns d'erreur
- **Baseline**: ~0.9-1.0 (toujours prédire la moyenne)
- **Objectifs**:
  - Epoch 1: ~0.5-0.7 ✓
  - Epoch 5: ~0.4-0.5 ✓
  - Epoch 10: ~0.3-0.4 ✓ (excellent)

### 2. Corrélation
- **Définition**: Corrélation de Pearson entre prédictions et cibles
- **Échelle**: -1 à +1 (1 = prédiction parfaite, 0 = aléatoire)
- **Objectifs**:
  - Epoch 1-2: >0.1-0.2 (début d'apprentissage)
  - Epoch 5: >0.3-0.4 (bon)
  - Epoch 10: >0.5-0.7 (excellent)

### 3. Amélioration vs Baseline
- **Définition**: % de réduction d'erreur par rapport à toujours prédire la moyenne
- **Objectifs**:
  - Epoch 1: >20-30% ✓
  - Epoch 5: >40-50% ✓
  - Epoch 10: >55-65% ✓ (excellent)

### 4. Std(preds) vs Std(targets)
- **Définition**: Écart-type des prédictions comparé aux cibles
- **Cible**: ~0.8-1.0 (même ordre de grandeur que les évaluations)
- **Problème**: Si std(preds) << std(targets), le modèle manque de variance (prédictions trop "plates")

## 🚀 Configuration actuelle

```python
Architecture: 768 → 128 (ReLU) → 128 (ReLU) → 1
Optimiseur: Adam (β₁=0.9, β₂=0.999, ε=1e-8)
Learning rate: 0.0005
Batch size: 64
Gradient clipping: 5.0 (norme globale)
Normalisation: Évals / 1000.0
```

## 📊 Interprétation des résultats

### ✅ Signes de bon apprentissage
- RMSE descend progressivement entre epochs
- Corrélation monte progressivement
- Amélioration vs baseline > 30% après quelques epochs
- Std(preds) se rapproche de std(targets) (~0.8-1.0)

### ⚠️ Signes de problèmes
- RMSE stagne ou augmente
- Corrélation < 0.1 après 5+ epochs
- Amélioration vs baseline < 10%
- Std(preds) reste très faible (~0.1-0.2)

### 🔧 Actions correctives

**Si loss stagne haut (>0.7):**
- Augmenter le learning rate à 0.001
- Augmenter la taille des couches cachées à 256
- Vérifier que les données sont bien chargées

**Si corrélation reste faible (<0.2):**
- Vérifier la normalisation des données
- Augmenter le nombre d'epochs
- Essayer d'augmenter la capacité du réseau

**Si std(preds) trop faible:**
- Déjà géré par warm-start du biais de sortie
- Adam aide à adapter les learning rates par paramètre

## 📈 Commandes utiles

### Lancer l'entraînement
```bash
cd "c:\Users\gauti\OneDrive\Documents\UE commande\smart-chess\ai"
python train.py
```

### Vérifier les statistiques du dataset
```bash
python check_dataset_stats.py
```

### Tester le modèle entraîné
```python
from nn_evaluator import load_evaluator_from_file
from Chess import Chess

evaluator = load_evaluator_from_file("chess_nn_weights.npz")
chess = Chess()
chess.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
score = evaluator.evaluate_position(chess)
print(f"Évaluation: {score:.0f} centipawns")
```

## 🎓 Conseils

1. **Patience**: L'apprentissage sur un gros dataset prend du temps
2. **Surveillance**: Regarde les métriques de fin d'epoch pour suivre la progression
3. **Sauvegarde**: Les poids sont sauvegardés après chaque epoch
4. **Continuité**: Si interrompu, l'entraînement reprend depuis les derniers poids
5. **Validation**: Teste ton modèle sur des positions connues pour vérifier qu'il est sensé

## 📝 Notes sur l'échelle

- **Stockage**: Évaluations en centipawns (ex: +500 = avantage d'un pion)
- **Entraînement**: Normalisé ÷1000 (ex: +500 → 0.5)
- **Inférence**: Re-multiplié ×1000 (ex: 0.5 → +500 centipawns)

Cette normalisation stabilise l'entraînement en gardant les gradients dans une plage raisonnable.
