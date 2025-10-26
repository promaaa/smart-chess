# Fix: Persistance des poids et moments Adam

## 🐛 Problème identifié

L'entraînement semblait réinitialiser les poids entre les exécutions. Le problème était en fait **la réinitialisation des moments Adam** qui rendait l'optimisation inefficace.

### Causes :

1. **Moments Adam réinitialisés à zéro** à chaque exécution
   - Adam utilise des moyennes exponentielles (momentum + variance adaptative)
   - Ces moments s'accumulent au fil de l'entraînement
   - Les réinitialiser = perdre tout l'historique d'optimisation

2. **Warm-start du biais de sortie** appliqué même sur un réseau déjà entraîné
   - Écrasait le biais appris

3. **Pas de différenciation** entre nouveau réseau et réseau chargé

## ✅ Solution implémentée

### 1. Sauvegarde des moments Adam

**Fichier : `nn_evaluator.py`**

```python
def save_weights(evaluator, filename, adam_moments=None):
    """
    Sauvegarde les poids + moments Adam (optionnel)
    """
    save_dict = {
        'w1': evaluator.weights1, 'b1': evaluator.biases1,
        'w2': evaluator.weights2, 'b2': evaluator.biases2,
        'w3': evaluator.weights3, 'b3': evaluator.biases3
    }
    
    if adam_moments is not None:
        save_dict.update(adam_moments)  # Ajouter m_w1, v_w1, m_b1, v_b1, ..., adam_step
    
    np.savez(filename, **save_dict)
```

### 2. Chargement des moments Adam

```python
def load_evaluator_from_file(filename):
    """
    Charge poids + moments Adam (si disponibles)
    Returns: (evaluator, adam_moments)
    """
    data = np.load(filename)
    evaluator = NeuralNetworkEvaluator(...)
    
    # Charger les moments Adam s'ils existent
    adam_keys = ['m_w1', 'v_w1', 'm_b1', 'v_b1', ..., 'adam_step']
    if all(key in data for key in adam_keys):
        adam_moments = {key: data[key] for key in adam_keys}
    else:
        adam_moments = None
    
    return evaluator, adam_moments
```

### 3. Gestion dans train.py

**Chargement intelligent :**

```python
if os.path.exists(WEIGHTS_FILE):
    evaluator, adam_moments_loaded = load_evaluator_from_file(WEIGHTS_FILE)
    is_new_network = False
else:
    evaluator = NeuralNetworkEvaluator.create_untrained_network()
    adam_moments_loaded = None
    is_new_network = True

# Warm-start SEULEMENT pour nouveau réseau
if is_new_network:
    evaluator.biases3[0, 0] = eval_mean
```

**Initialisation des moments :**

```python
if USE_ADAM:
    if adam_moments_loaded:
        # Charger les moments existants
        m_w1 = adam_moments_loaded['m_w1']
        v_w1 = adam_moments_loaded['v_w1']
        # ...
        adam_step = int(adam_moments_loaded['adam_step'])
    else:
        # Créer de nouveaux moments
        m_w1 = np.zeros_like(evaluator.weights1)
        # ...
        adam_step = 0
```

**Sauvegarde avec moments :**

```python
if USE_ADAM:
    adam_dict = {
        'm_w1': m_w1, 'v_w1': v_w1,
        # ... tous les moments
        'adam_step': np.array(adam_step)
    }
    save_weights(evaluator, WEIGHTS_FILE, adam_moments=adam_dict)
else:
    save_weights(evaluator, WEIGHTS_FILE)
```

## 📊 Impact attendu

### Avant (avec bug) :
- Moments Adam réinitialisés → Learning rate adaptatif inefficace
- Warm-start écrasait le biais appris
- Optimisation repartait de zéro à chaque exécution

### Après (fix) :
- ✓ Moments Adam persistés → Optimisation continue correctement
- ✓ Warm-start seulement pour nouveaux réseaux
- ✓ Vrai entraînement incrémental possible

## 🧪 Test de validation

Script : `test_adam_persistence.py`

```bash
python test_adam_persistence.py
```

Résultat : **✓✓ SUCCÈS - Les moments Adam sont correctement persistés!**

## 🚀 Prochaines étapes

1. **Supprimer l'ancien fichier** (sans moments Adam) :
   ```bash
   Remove-Item chess_nn_weights.npz -Force
   ```

2. **Relancer l'entraînement** :
   ```bash
   python train.py
   ```

3. **Vérifier la continuité** :
   - Premier run : "Initialisation de nouveaux moments Adam"
   - Runs suivants : "Moments Adam chargés (step=XXX)"

4. **Monitorer les améliorations** :
   - RMSE devrait descendre progressivement
   - Corrélation devrait augmenter
   - Pas de régression entre les exécutions

## 📝 Notes techniques

- Fichier de poids maintenant plus gros (~4MB au lieu de 2MB)
- Backward compatible : anciens fichiers sans moments = création de nouveaux moments
- Adam step global : permet de reprendre bias correction au bon endroit
- Format .npz conserve tous les arrays NumPy efficacement

## ✅ Validation finale

Pour vérifier que tout fonctionne :

```bash
# 1. Première exécution (quelques epochs)
python train.py  # → "Initialisation de nouveaux moments Adam"

# 2. Vérifier signature des poids
python check_weights_persistence.py  # → Signature sauvegardée

# 3. Deuxième exécution
python train.py  # → "Moments Adam chargés (step=XXX)"

# 4. Re-vérifier signature
python check_weights_persistence.py  # → "✓ Les poids ont changé!"
```

---

**Problème résolu !** 🎉
