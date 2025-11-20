# Opening Book

## Cerebellum_Light.bin

Ce dossier doit contenir le fichier `Cerebellum_Light.bin` pour que le moteur d'échecs puisse utiliser le livre d'ouvertures.

### Pourquoi le fichier n'est pas inclus ?

Le fichier `Cerebellum_Light.bin` fait 157 MB, ce qui dépasse la limite de 100 MB de GitHub. Il n'est donc pas inclus dans le repository.

### Comment l'obtenir ?

#### Option 1 : Téléchargement direct (recommandé)

Téléchargez le fichier depuis l'une de ces sources :

- **Source principale** : https://zipproth.de/Brainfish/download/

Placez le fichier téléchargé dans ce dossier :
```
prototypes/8x8-maquette/firmware/IA-Marc/book/Cerebellum_Light.bin
```

#### Option 2 : Utiliser un autre livre Polyglot

N'importe quel livre d'ouvertures au format Polyglot (.bin) peut être utilisé. Renommez-le en `Cerebellum_Light.bin` ou modifiez le chemin dans le code.

### Vérification

Pour vérifier que le fichier est correctement installé :

```bash
cd prototypes/8x8-maquette/firmware/IA-Marc/V2
python3 -c "import os; print('OK' if os.path.exists('../book/Cerebellum_Light.bin') else 'Fichier manquant')"
```

Ou testez directement :

```bash
python3 test_opening_book.py
```

### Fonctionnement sans le fichier

Le moteur fonctionne sans le livre d'ouvertures, mais :
- Les coups d'ouverture seront calculés (1-2 secondes) au lieu d'être instantanés (< 1 ms)
- Fallback automatique sur `data/openings.json` si disponible

### Caractéristiques de Cerebellum_Light.bin

- **Taille** : 157 MB
- **Format** : Polyglot (binaire)
- **Positions** : 500,000+
- **Profondeur** : Jusqu'à 15-20 coups
- **Source** : Parties de maîtres et analyses d'ordinateurs