# Guide: Entraîner ton modèle Chess sur GPU avec Google Colab (GRATUIT)

## 🚀 Pourquoi Colab ?
- ✅ GPU **Tesla T4 gratuit** (15-20x plus rapide que CPU)
- ✅ Aucune installation nécessaire
- ✅ 12 GB de RAM GPU
- ✅ Environnement Python déjà configuré

## 📝 Étapes pour utiliser Colab :

### 1. Ouvre Google Colab
👉 Va sur https://colab.research.google.com/

### 2. Active le GPU
- Menu : **Runtime** → **Change runtime type**
- Hardware accelerator : Sélectionne **GPU (T4)**
- Clique **Save**

### 3. Upload ton code
Crée un nouveau notebook et exécute :

```python
# Cellule 1: Installation des dépendances
!pip install pandas tqdm

# Cellule 2: Upload ton dataset
from google.colab import files
uploaded = files.upload()  # Upload chessData.csv (peut prendre du temps pour 13M lignes)

# Ou utilise Google Drive (plus rapide pour gros fichiers)
from google.colab import drive
drive.mount('/content/drive')
# Copie ton dataset dans Google Drive d'abord
```

### 4. Copie ton code d'entraînement
Je vais créer une version PyTorch optimisée pour GPU que tu pourras copier-coller dans Colab.

## ⚡ Performance attendue
- **CPU (ton PC)** : ~3 min/epoch sur 200k positions
- **GPU Colab T4** : ~10-20 secondes/epoch sur 200k positions
- **Accélération** : 10-20x plus rapide !

## 📊 Alternative: Utiliser un échantillon plus large
Avec un GPU, tu peux facilement entraîner sur 1-2M positions/epoch au lieu de 200k !

## 💾 Sauvegarder les poids
À la fin de l'entraînement dans Colab :
```python
from google.colab import files
files.download('chess_nn_weights.npz')  # Télécharge les poids entraînés
```

## 🔄 Workflow recommandé
1. Développe et teste ton code localement (CPU, petit échantillon)
2. Entraîne sur GPU Colab (gros échantillon, plus d'epochs)
3. Télécharge les poids entraînés
4. Utilise-les dans ton jeu d'échecs local

---

**Veux-tu que je crée la version PyTorch + GPU de ton entraînement ?**
