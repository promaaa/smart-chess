# ======= fichier : tournoi_niveaux.py =======

import time
import match_niv

NB_PARTIES_PAR_MATCH = 10  # nombre de parties par duel
NIVEAUX = [1, 2, 3, 4, 5]


def duel(lvlA, lvlB):
    print(f"\n🔷 DUEL : Niveau {lvlA} vs Niveau {lvlB} 🔶")
    scoreA = 0
    scoreB = 0

    for i in range(1, NB_PARTIES_PAR_MATCH + 1):
        print(f"\n--- Partie {i}/{NB_PARTIES_PAR_MATCH} ---")
        res = match_niv.play_game(lvlA, lvlB, i)

        # score
        if res == "1-0":
            scoreA += 1
        elif res == "0-1":
            scoreB += 1
        else:
            scoreA += 0.5
            scoreB += 0.5

    print(f"\n🎯 Score final : L{lvlA} = {scoreA}  |  L{lvlB} = {scoreB}")
    return scoreA, scoreB


def main():
    print("\n🏁 TOURNOI INTÉGRAL DES NIVEAUX 🏁")
    global_score = {lvl: 0 for lvl in NIVEAUX}

    # Tous les duels possibles niveau A < niveau B
    for i in range(len(NIVEAUX)):
        for j in range(i + 1, len(NIVEAUX)):
            A = NIVEAUX[i]
            B = NIVEAUX[j]

            scoreA, scoreB = duel(A, B)
            global_score[A] += scoreA
            global_score[B] += scoreB

            time.sleep(1)

    print("\n🏆 RÉSULTATS GLOBAUX DU TOURNOI 🏆")
    for lvl, score in sorted(global_score.items()):
        print(f"Niveau {lvl} : {score} points")

    print("\n📌 Tournoi terminé. Toutes les parties sont enregistrées !")


if __name__ == "__main__":
    main()
 