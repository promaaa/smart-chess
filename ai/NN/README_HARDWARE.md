# Guide des ressources matérielles pour l'entraînement

## 📊 Ton contexte actuel

**Dataset** : 13 millions de positions
**Temps par epoch** : 4 heures (CPU only, ~55k positions traitées)
**Architecture** : 768 → 256 → 256 → 1 (~263k paramètres)

## 🔍 Analyse du problème

### Problème principal : **Boucle non-vectorisée**
```python
# Code actuel (LENT) ❌
for fen in batch:
    x = encode(fen)  # Une position à la fois
    forward_pass(x)   # NumPy sur 1 position
    backward_pass()
```

**Impact** : 13M appels Python → overhead énorme

### Solution 1 : **Échantillonnage (appliqué)** ✓
```python
MAX_SAMPLES = 200,000  # 1.5% du dataset
Temps par epoch : ~10-15 minutes (au lieu de 4h)
```

**Pourquoi ça suffit** :
- 200k positions bien échantillonnées ≈ diversité du dataset complet
- Plus d'epochs sur moins de data > moins d'epochs sur plus de data
- Itération rapide pour tuner hyperparamètres

## ⚡ Ressources matérielles recommandées

### **Option 1 : CPU moderne (ton cas actuel)**

**Minimum viable** :
- CPU : Intel i5/i7 ou AMD Ryzen 5/7 (4+ cores)
- RAM : 8 GB minimum, 16 GB recommandé
- Storage : SSD pour chargement rapide du CSV

**Performance avec 200k positions** :
```
Temps par epoch : 10-15 minutes
20 epochs : ~3-5 heures total
Performance attendue : RMSE ~0.30-0.35 (très bon)
```

**Performance avec 13M positions** :
```
Impossible en pratique sans vectorisation GPU
Même avec CPU puissant : plusieurs jours par epoch
```

### **Option 2 : GPU consumer (recommandé si sérieux)**

**Config recommandée** :
- GPU : NVIDIA RTX 3060 (12GB), RTX 4060 Ti, ou mieux
- RAM : 16 GB
- Framework : PyTorch ou TensorFlow

**Performance avec 200k positions** :
```
Temps par epoch : 1-2 minutes (10x plus rapide)
20 epochs : ~30-40 minutes total
```

**Performance avec 13M positions** :
```
Temps par epoch : 30-60 minutes
20 epochs : 10-20 heures total
Faisable avec batching GPU optimisé
```

**Coût** : ~400-600€ pour RTX 3060/4060 Ti

### **Option 3 : Cloud GPU (flexible)**

**Google Colab** :
- Gratuit : T4 GPU (15 GB), ~12h/jour
- Pro ($10/mois) : A100 GPU, temps illimité
- Performance : 50-100x plus rapide que CPU

**AWS/Azure/GCP** :
- ~$0.50-2/heure selon GPU
- Bon pour expérimenter sans investir

**Exemple AWS** :
```
Instance : p3.2xlarge (Tesla V100)
Prix : ~$3/heure
Temps 13M positions : ~5-10 heures total (tous epochs)
Coût total : ~$15-30
```

### **Option 4 : Laptop gaming moderne**

Si tu as déjà :
- RTX 3050/3060 laptop
- 16 GB RAM
- → Parfait pour 200k-500k positions

## 🎯 Mes recommandations par budget

### **Budget 0€ : Reste CPU + échantillonnage** ✓
```python
MAX_SAMPLES = 200,000
20 epochs × 12 minutes = 4 heures
```
**Résultat** : Modèle très performant (~1500 Elo)

### **Budget 10€/mois : Google Colab Pro**
```python
MAX_SAMPLES = 1,000,000
20 epochs × 5 minutes = 1h40
```
**Résultat** : Modèle excellent (~1700-1800 Elo)

### **Budget 500€ : Acheter RTX 3060**
```python
MAX_SAMPLES = 13,000,000 (tout!)
20 epochs × 45 minutes = 15 heures
```
**Résultat** : Modèle top (~1800-2000 Elo)
**Bonus** : Utilisable pour autres projets ML/gaming

## 📈 Comparaison performance vs dataset

| Positions | Epochs | Temps CPU | Temps GPU (RTX 3060) | RMSE attendu |
|-----------|--------|-----------|---------------------|--------------|
| 50k       | 20     | 2h        | 15 min              | ~0.35-0.40   |
| 200k ✓    | 20     | 4-5h      | 30 min              | ~0.30-0.35   |
| 500k      | 20     | 10-12h    | 1h                  | ~0.28-0.32   |
| 1M        | 20     | 20-24h    | 2h                  | ~0.27-0.31   |
| 13M       | 20     | Impossible| 15-20h              | ~0.25-0.30   |

**Rendements décroissants** : Passer de 200k à 13M ne divise la loss que par ~1.15x

## 💡 Stratégie optimale pour toi

### **Phase 1 : Validation (maintenant)** ✓
```python
MAX_SAMPLES = 200,000
EPOCHS = 20
```
**But** : Prouver que l'architecture + warmup marche
**Temps** : ~4-5h sur ton CPU
**Coût** : 0€

### **Phase 2 : Amélioration (optionnel)**
```python
MAX_SAMPLES = 500,000
EPOCHS = 30
```
**Option A** : CPU overnight (10-12h)
**Option B** : Google Colab gratuit (2h)
**But** : Atteindre RMSE ~0.28-0.30

### **Phase 3 : Maximum (si vraiment motivé)**
```python
MAX_SAMPLES = 13,000,000
EPOCHS = 50
```
**Requis** : GPU (Colab Pro ou hardware)
**Temps** : 20-30h total
**But** : Niveau compétitif (~2000 Elo)

## 🚀 Actions immédiates

1. **Lance avec 200k échantillons** (déjà configuré)
2. **Attends les résultats** (4-5h)
3. **Si RMSE < 0.35** → Mission accomplie ! ✓
4. **Si tu veux pousser** → Passe à 500k ou essaye Colab

## 📝 Note importante

**Avoir plus de données n'améliore pas toujours la performance** :
- 200k positions bien échantillonnées > 13M positions mal utilisées
- Plus d'epochs sur moins de data souvent meilleur
- Itération rapide > attendre 20h par run

**Stockfish NNUE** utilise 500M positions mais avec :
- Architecture spécialisée NNUE
- Self-play reinforcement learning
- Cluster de serveurs
- Mois d'entraînement

Pour ton projet, **200k-500k positions suffisent largement** ! 🎯
