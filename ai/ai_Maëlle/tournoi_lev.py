# ======= fichier : tournoi_niveaux.py =======

import time
import match_niv

# Nombre de parties jouées pour chaque duel entre deux niveaux
NB_PARTIES_PAR_MATCH = 10

# Liste des niveaux du moteur à comparer
NIVEAUX = [1, 2, 3, 4, 5]


def duel(lvlA, lvlB):
    """
    Lance un duel complet entre deux niveaux du moteur.
    """
    print(f"\n DUEL : Niveau {lvlA} vs Niveau {lvlB} ")

    # Scores cumulés pour chaque niveau
    scoreA = 0
    scoreB = 0

    for i in range(1, NB_PARTIES_PAR_MATCH + 1):
        print(f"\n--- Partie {i}/{NB_PARTIES_PAR_MATCH} ---")

        # Lancement d'une partie entre les deux niveaux
        res = match_niv.play_game(lvlA, lvlB, i)

        # Attribution des points selon le résultat
        if res == "1-0":
            scoreA += 1
        elif res == "0-1":
            scoreB += 1
        else:
            # Match nul : demi-point pour chaque niveau
            scoreA += 0.5
            scoreB += 0.5

    # Affichage du score final du duel
    print(f"\n Score final : L{lvlA} = {scoreA}  |  L{lvlB} = {scoreB}")

    return scoreA, scoreB


def main():
    """
    Chaque niveau affronte tous les autres niveaux une seule fois.
    """
    print("\n TOURNOI INTÉGRAL DES NIVEAUX ")

    # Dictionnaire stockant le score global de chaque niveau
    global_score = {lvl: 0 for lvl in NIVEAUX}

    # Boucle sur tous les duels possibles avec A < B
    for i in range(len(NIVEAUX)):
        for j in range(i + 1, len(NIVEAUX)):
            A = NIVEAUX[i]
            B = NIVEAUX[j]

            # Lancement du duel entre les deux niveaux
            scoreA, scoreB = duel(A, B)

            # Ajout des scores au classement global
            global_score[A] += scoreA
            global_score[B] += scoreB

            # Pause volontaire pour lisibilité et stabilité
            time.sleep(1)

    # Affichage du classement final
    print("\n RÉSULTATS GLOBAUX DU TOURNOI ")
    for lvl, score in sorted(global_score.items()):
        print(f"Niveau {lvl} : {score} points")

    print("\n Tournoi terminé. Toutes les parties sont enregistrées !")


# Point d'entrée du script
if __name__ == "__main__":
    main()

 