# Interface Utilisateur - Smart Chess

## Description

Ce dossier contient une simulation de l'interface utilisateur qui sera affichée sur l'écran 2.8" du plateau d'échecs intelligent Smart Chess. Cette interface web permet de :

- **Sélectionner un niveau de difficulté** parmi les 8 niveaux de l'IA Marc (400-1700 ELO)
- **Choisir une personnalité de jeu** (Équilibré, Agressif, Défensif, Positionnel, Tactique, Matérialiste)
- **Jouer aux échecs** contre l'IA Marc directement dans le navigateur
- **Visualiser l'échiquier** avec un design moderne simulant l'écran physique

## Structure des fichiers

```
interface_utilisateur/
├── index.html          # Page principale
├── style.css           # Styles CSS (design moderne, animations)
├── chess-ui.js         # Module de gestion de l'échiquier
├── ai-engine.js        # Simulation de l'IA Marc V2
├── app.js              # Application principale
└── README.md           # Cette documentation
```

## Niveaux de Difficulté

| Niveau | Nom | ELO | Description |
|--------|-----|-----|-------------|
| 1 | Débutant | 400 | Idéal pour les enfants et débutants |
| 2 | Novice | 600 | Pour ceux qui apprennent les bases |
| 3 | Occasionnel | 800 | Joueur occasionnel |
| 4 | Amateur | 1000 | Joueur amateur régulier |
| 5 | Club débutant | 1200 | Niveau club débutant |
| 6 | Club intermédiaire | 1400 | Niveau club intermédiaire |
| 7 | Club avancé | 1600 | Niveau club avancé |
| 8 | Maximum | 1700 | Force maximale de l'IA Marc |

## Personnalités de Jeu

- **Équilibré** : Joue de manière équilibrée, sans préférence particulière
- **Agressif** : Attaque constamment, cherche les complications tactiques
- **Défensif** : Joue solidement, évite les risques
- **Positionnel** : Mise sur le contrôle de l'espace et la structure
- **Tactique** : Cherche les combinaisons et les coups brillants
- **Matérialiste** : Privilégie le gain de matériel avant tout

## Utilisation

### Lancement rapide

1. Ouvrez un terminal dans ce dossier
2. Lancez un serveur HTTP local :

```bash
# Avec Python 3
python3 -m http.server 8080

# Ou avec Python 2
python -m SimpleHTTPServer 8080
```

3. Ouvrez votre navigateur à l'adresse : `http://localhost:8080`

### Comment jouer

1. **Sélectionnez un niveau** en cliquant sur une des 8 cartes de niveau
2. **Choisissez une personnalité** (optionnel, Équilibré par défaut)
3. **Cliquez sur "Commencer la partie"**
4. **Jouez** en cliquant sur une pièce puis sur une case valide
5. Attendez que l'IA joue son coup

### Raccourcis clavier

| Touche | Action |
|--------|--------|
| `N` | Nouvelle partie |
| `U` | Annuler le dernier coup |
| `Esc` | Retour au menu |
| `F` | Retourner l'échiquier |

## Caractéristiques techniques

### Design

- **Simulation d'écran 2.8"** : L'interface simule l'aspect d'un petit écran embarqué
- **Design moderne** : Glassmorphism, animations fluides, mode sombre
- **LEDs simulées** : La barre de LEDs en bas simule les 8 LEDs physiques du plateau
- **Responsive** : S'adapte aux différentes tailles d'écran

### IA Marc (simulation)

L'IA implémentée dans `ai-engine.js` est une simulation simplifiée de l'IA Marc V2 :

- **Algorithme Minimax** avec élagage Alpha-Beta
- **Recherche de quiescence** pour stabiliser l'évaluation
- **Tables de positionnement** (Piece-Square Tables)
- **Ordonnancement des coups** (MVV-LVA)
- **Niveaux de difficulté** avec taux d'erreur configurable

> **Note** : Cette simulation côté client est moins puissante que la vraie IA Marc V2 qui tourne sur le Raspberry Pi avec des optimisations avancées.

## Intégration avec le plateau physique

Cette interface est conçue pour :

1. **Prototypage** : Tester l'ergonomie de l'interface avant l'implémentation sur le vrai écran
2. **Démonstration** : Montrer le concept du projet Smart Chess
3. **Développement** : Développer l'interface en parallèle du hardware

Pour l'intégration finale sur l'écran 2.8" du Raspberry Pi, le code devra être adapté pour :
- Utiliser `pygame` ou `tkinter` au lieu du navigateur web
- Communiquer avec l'IA Marc V2 native (`ia_marc/V2/`)
- Recevoir les entrées des capteurs physiques

## Dépendances

- **chess.js** : Bibliothèque JavaScript pour la logique des échecs (chargée via CDN)
- **Google Fonts** : Polices Outfit et JetBrains Mono

## Compatibilité

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Auteur

Projet Smart Chess - Échiquier Intelligent
