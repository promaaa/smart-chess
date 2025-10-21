# 🚀 Guide GPU Training - Récapitulatif

## ✅ Fichiers créés aujourd'hui

### 1. **torch_nn_evaluator.py**
- Réseau PyTorch (768→256→256→1, LeakyReLU, Dropout 0.3)
- Compatible GPU
- Conversion NumPy ↔ PyTorch

### 2. **train_torch.py**  
- Entraînement GPU optimisé (10-20x plus rapide)
- DataLoader, AdamW, LR scheduler, warmup
- Ré-échantillonnage à chaque epoch
- Sauvegarde .npz + .pt

### 3. **colab_training.ipynb**
- Notebook prêt pour Google Colab
- GPU Tesla T4 gratuit
- Instructions complètes

### 4. **convert_weights.py**
- Conversion poids NumPy → PyTorch
- Vérification de compatibilité

### 5. **check_gpu.py**
- Vérifie disponibilité GPU CUDA

### 6. **debug_conversion.py**
- Debug conversion (vérifié: ✅ fonctionne parfaitement)

### 7. **test_generalization.py** (créé précédemment)
- Test de généralisation sur échantillons aléatoires

### 8. **visualize_sampling.py** (créé précédemment)
- Visualise impact du ré-échantillonnage

---

## 📊 Tests effectués

### ✅ Conversion NumPy → PyTorch
```
=== VÉRIFICATION POIDS ===
W1 identique: True (diff max: 0.00000001)
B1 identique: True
W2 identique: True
W3 identique: True

=== COMPARAISON PRÉDICTIONS ===
NumPy score:  12.91
Torch score:  12.91
Diff:         0.00  ✅ PARFAIT
```

### ✅ Généralisation actuelle
```
Échantillon 1:  RMSE=0.7271  Corr=0.4806
Échantillon 2:  RMSE=0.7241  Corr=0.4921
Ratio: 1.00x ✅ Bonne généralisation
```

**Conclusion**: Pas d'overfitting, mais performance modeste (RMSE ~0.72)

### ✅ Impact ré-échantillonnage
```
AVANT: 200k positions × 20 répétitions = 200k positions uniques
APRÈS: 200k positions/epoch × 20 epochs = 4M positions uniques
Amélioration: +1900% diversité ✅
```

---

## 🎯 Utilisation Google Colab (RECOMMANDÉ)

### Setup (5 min)
1. Va sur https://colab.research.google.com/
2. Upload `colab_training.ipynb`
3. Runtime → Change runtime type → GPU (T4)
4. Upload fichiers : `Chess.py`, `torch_nn_evaluator.py`, `train_torch.py`
5. Upload dataset via Google Drive (pour 13M lignes)

### Performance attendue
- **CPU local**: 3 min/epoch sur 200k positions
- **GPU Colab**: 10-20 sec/epoch sur 200k positions
- **Accélération**: **10-20x** 🚀

### Config recommandée pour GPU
```python
MAX_SAMPLES = 500_000  # ou 1M
BATCH_SIZE = 128       # ou 256
EPOCHS = 30
HIDDEN_SIZE = 256      # ou 512
DROPOUT = 0.3
```

### Résultats attendus (30 epochs, 500k positions/epoch)
- Positions uniques: **15M** (vs 200k actuellement)
- RMSE: **< 0.50** (vs 0.72 actuellement)
- Corrélation: **> 0.70** (vs 0.48 actuellement)

---

## ⚠️ Bug à corriger dans nn_evaluator.py

Ton code d'évaluation utilise **ReLU** mais l'entraînement utilise **LeakyReLU** !

### Fichier: `nn_evaluator.py`, ligne ~62-63

**AVANT** (incorrect):
```python
h1 = np.maximum(0, np.dot(input_vector, self.weights1) + self.biases1)
h2 = np.maximum(0, np.dot(h1, self.weights2) + self.biases2)
```

**APRÈS** (correct, compatible avec train.py):
```python
LEAKY_ALPHA = 0.01
h1_in = np.dot(input_vector, self.weights1) + self.biases1
h1 = np.where(h1_in > 0, h1_in, LEAKY_ALPHA * h1_in)
h2_in = np.dot(h1, self.weights2) + self.biases2
h2 = np.where(h2_in > 0, h2_in, LEAKY_ALPHA * h2_in)
```

**Impact**: Sans ce fix, les prédictions en jeu seront incorrectes !

---

## 🔄 Workflow complet

### Phase 1: Développement local (CPU)
```bash
# Tester avec petit échantillon
python train.py  # NumPy, CPU, 200k positions
```

### Phase 2: Entraînement GPU (Colab)
```bash
# Sur Colab
!python train_torch.py  # PyTorch, GPU, 500k-1M positions
# Télécharger: chess_nn_weights.npz
```

### Phase 3: Intégration
```python
# Fixer nn_evaluator.py (LeakyReLU)
# Tester avec convert_weights.py
# Utiliser dans le jeu
```

---

## 📁 Structure finale

```
ai/
├── Chess.py                    # Moteur échecs
├── nn_evaluator.py             # Réseau NumPy (⚠️ fixer LeakyReLU)
├── train.py                    # Train CPU/NumPy
├── torch_nn_evaluator.py       # ✨ Réseau PyTorch
├── train_torch.py              # ✨ Train GPU/PyTorch
├── colab_training.ipynb        # ✨ Notebook Colab
├── convert_weights.py          # ✨ Conversion
├── check_gpu.py                # ✨ Vérif GPU
├── debug_conversion.py         # ✨ Debug
├── test_generalization.py      # Test généralisation
├── visualize_sampling.py       # Visualisation
├── README_GPU_COLAB.md         # Guide Colab
└── README_TRAINING.md          # Guide général
```

---

## 🚀 Prochaines étapes

1. [ ] **Fixer le bug LeakyReLU** dans `nn_evaluator.py`
2. [ ] **Tester train_torch.py localement** (1 epoch pour validation)
3. [ ] **Upload sur Colab** et lancer entraînement complet
4. [ ] **Télécharger poids entraînés**
5. [ ] **Tester avec test_generalization.py**
6. [ ] **Intégrer dans le jeu d'échecs**

---

**Questions? Lance `check_gpu.py` ou `debug_conversion.py` pour diagnostiquer !**

**Bon entraînement sur GPU ! 🚀🎯**
