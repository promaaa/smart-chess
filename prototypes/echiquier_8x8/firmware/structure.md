# Documentation Projet: Échiquier Intelligent

Ce document détaille l'architecture matérielle et logicielle d'un projet d'échiquier intelligent personnalisé. L'objectif est de fournir une référence technique complète pour le débogage et le développement.

## 1. Vue d'ensemble

Le projet consiste en un échiquier 8x8 capable de :
1.  **Détecter** la position des pièces en temps réel à l'aide d'une matrice de 64 capteurs Reed (un par case).
2.  **Afficher** des informations (coups, menaces, etc.) à l'aide d'une matrice de 64 LEDs (une par case) et d'une rangée supplémentaire de 8 LEDs, pilotées par deux contrôleurs `HT16K33`.

Tous les composants sont gérés par un microcontrôleur principal (ex: Raspberry Pi) via un bus I2C unique, organisé à l'aide d'un multiplexeur.

## 2. Composants Matériels

* **1x Processeur Principal** (ex: Raspberry Pi, gérant `board.SCL`/`board.SDA`).
* **1x Multiplexeur I2C** : `TCA9548A` (Sert de hub central).
* **4x Contrôleurs I/O 16-Pin** : `MCP23017` (Utilisés pour lire les capteurs).
* **64x Capteurs Reed** (Matrice 8x8, un par case).
* **2x Contrôleurs LED 16x8** : `HT16K33` (Pilotes pour les LEDs).
* **1x Matrice LED 8x8** (64 LEDs pour l'échiquier).
* **1x Rangée LED 1x8** (8 LEDs supplémentaires).

## 3. Architecture & Câblage

L'architecture est centralisée autour du bus I2C et du multiplexeur `TCA9548A`.

### 3.1. Bus I2C & Multiplexeur (TCA9548A)

Le processeur principal communique avec le `TCA9548A`. Le multiplexeur distribue ensuite les commandes I2C aux différents composants sur des canaux dédiés pour éviter les conflits d'adresse.

* `Processeur` -> `TCA9548A` (SCL/SDA)
    * **Canal 0** -> `MCP23017` (Nommé **CM0**) @ Adresse `0x20`
    * **Canal 1** -> `MCP23017` (Nommé **CM1**) @ Adresse `0x20`
    * **Canal 2** -> `MCP23017` (Nommé **CM2**) @ Adresse `0x20`
    * **Canal 3** -> `MCP23017` (Nommé **CM3**) @ Adresse `0x20`
    * **Canal 4** -> `HT16K33` (Nommé **LED_A**) @ Adresse `0x70` (Gère matrice 8x8)
    * **Canal 5** -> `HT16K33` (Nommé **LED_B**) @ Adresse `0x71` (Gère rangée 1x8)
    * Canaux 6, 7 -> Libres

> **Note :** L'adresse du second HT16K33 (LED_B) doit être changée (ex: `0x71`) en soudant le pontet d'adresse A0 pour éviter un conflit avec LED_A (`0x70`).

### 3.2. Sous-système Capteurs (64 Reed / 4x MCP23017)

Ce sous-système est inchangé. Les 64 capteurs (8 rangées x 8 colonnes) sont lus par les 4 puces MCP23017.

* Les capteurs sont configurés en `INPUT` avec `PULL-UP`. Un capteur est **actif (pièce présente) quand le pin est lu à `LOW` (0)**.
* **CM0 (Canal 0)** : Gère les 16 capteurs des **Rangées 1 et 2** (cases `a1` à `h2`).
* **CM1 (Canal 1)** : Gère les 16 capteurs des **Rangées 3 et 4** (cases `a3` à `h4`).
* **CM2 (Canal 2)** : Gère les 16 capteurs des **Rangées 5 et 6** (cases `a5` à `h6`).
* **CM3 (Canal 3)** : Gère les 16 capteurs des **Rangées 7 et 8** (cases `a7` à `h8`).

### 3.3. Sous-système Affichage (64+8 LEDs / 2x HT16K33)

C'est le composant qui a été modifié.
* **Contrôleurs :** 2x `HT16K33`.
* **Matrice Physique :** Le système est composé d'une matrice principale 8x8 (64 LEDs) et d'une rangée supplémentaire 1x8 (8 LEDs).
* **Logique de Câblage (Confirmée) :**
    * **LED_A (Canal 4 @ 0x70) :** Gère la matrice 8x8 principale (cases `a1` à `h8`).
    * **LED_B (Canal 5 @ 0x71) :** Gère la 9ème rangée supplémentaire (8 LEDs, connectées aux 8 colonnes).
* **Bibliothèque :** Le projet utilise `adafruit_ht16k33.matrix` pour contrôler les LEDs.

## 4. Mappings Logiciels (Synthèse)

La correspondance entre le matériel et le logiciel est définie par les fichiers de mapping.

### 4.1. Mapping Capteurs (Reed -> Case)

Ce mapping est défini dans `mapping` et `local_detection.py` (Inchangé).

| Case Échiquier (row, col) | Contrôleur | Pin |
| :--- | :--- | :--- |
| `a1` (0,0) | CM0 (Canal 0) | `A0` |
| ... | ... | ... |
| `h8` (7,7) | CM3 (Canal 3) | `A4` |

### 4.2. Mapping Affichage (Case -> LEDs)

Ce mapping est redéfini en fonction du câblage final des `HT16K33`.

* **Principe :** Le mapping est divisé en deux parties : la matrice 8x8 (échiquier) et la rangée 1x8 (extra).
* **Logique de Mapping :**
    * Une case `(row, col)` (index 0-7) de l'échiquier : La LED est contrôlée par **LED_A** (Canal 4).
    * Une LED `(col)` (index 0-7) de la 9ème rangée : La LED est contrôlée par **LED_B** (Canal 5).
* **Exemple d'appel logiciel :**
    * Pour allumer `c2` (row=1, col=2) sur l'échiquier :
        * Utilise `LED_A` (ex: `display_A`).
        * Appel : `display_A.pixel(col, row, 1)` -> `display_A.pixel(2, 1, 1)`
    * Pour allumer la 3ème LED (col=2) de la rangée supplémentaire :
        * Utilise `LED_B` (ex: `display_B`).
        * Appel (en supposant que la rangée est mappée sur la ligne 0 du driver) :
        * `display_B.pixel(col, 0, 1)` -> `display_B.pixel(2, 0, 1)`

> **Note :** Le mapping exact `(col, row)` ou `(row, col)` dépend de l'orientation de la matrice lors du câblage sur les broches COM/ROW du HT16K33. L'exemple ci-dessus suppose `(col, row)`.

## 5. Stratégie de Débogage (Post-Migration HT16K33)

* **Problème Précédent (Résolu) :** Le diagnostic pointait vers une puce `IS31FL3731` défectueuse. Ce composant a été retiré.
* **Stratégie Logicielle :**
    1.  Le projet **doit** utiliser la bibliothèque `adafruit_ht16k33`.
    2.  **Initialisation :** Deux objets (ex: `display_A`, `display_B`) doivent être créés, un pour chaque `HT16K33`, en ciblant les bons canaux du multiplexeur (`TCA9548A`).
    3.  **Bibliothèques Recommandées :**
        * `adafruit_tca9548a` pour le multiplexeur.
        * `adafruit_ht16k33.matrix.Matrix8x8` pour `display_A`.
        * `adafruit_ht16k33.matrix.Matrix8x8` (ou `Matrix16x8`) pour `display_B`, même si seule la première rangée est utilisée.
    4.  **Test de Connexion :** Vérifier que les deux puces sont bien détectées sur le bus I2C (sur les canaux 4 et 5) aux adresses `0x70` et `0x71` respectives.
    5.  **Validation du Mapping :** Écrire un script de balayage qui allume d'abord tous les pixels de `display_A` (la matrice 8x8), puis tous les pixels de la première rangée de `display_B` (la rangée 1x8), pour confirmer le câblage et l'adressage.
