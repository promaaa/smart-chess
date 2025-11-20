#!/usr/bin/env python3
"""
Comparaison de l'IA Marc V2 avec Stockfish.
Ce script exécute un nombre configurable de parties (par défaut 10) où le niveau
`Skill Level` de Stockfish varie de 0 à 20.  Il collecte :
- temps moyen, total, min/max pour chaque moteur
- nombre de coups joués
- nombre de victoires / défaites / nuls
- graphiques (matplotlib) pour visualiser les performances et les taux de victoire.
"""

import os
import sys
import time
import traceback
import chess
import chess.engine
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration du chemin vers le répertoire du projet
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Ajoute le répertoire contenant l'IA V2 au PYTHONPATH
sys.path.insert(0, os.path.join(PROJECT_ROOT, "prototypes/8x8-maquette/firmware/IA-Marc/V2"))

# ---------------------------------------------------------------------------
# Import de l'IA V2
# ---------------------------------------------------------------------------
try:
    from engine_main import ChessEngine  # classe principale du moteur V2
    NEW_AI_AVAILABLE = True
except Exception as e:
    print(f"[WARN] IA‑Marc V2 non disponible : {e}")
    NEW_AI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helper : lancement de Stockfish via python‑chess
# ---------------------------------------------------------------------------
def get_stockfish_engine(level: int = 20):
    """Retourne une instance `SimpleEngine` de Stockfish configurée au `Skill Level` donné.
    Le binaire est recherché dans le PATH ; vous pouvez forcer un chemin via la variable
    d'environnement `STOCKFISH_PATH`.
    """
    stockfish_path = os.getenv("STOCKFISH_PATH", "stockfish")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({
            "Skill Level": level,
            "Hash": 64,
            "Threads": 2,
        })
        return engine
    except Exception as exc:
        print(f"[ERROR] Impossible de lancer Stockfish (niveau {level}) : {exc}")
        raise

# ---------------------------------------------------------------------------
# Classe utilitaire pour les statistiques de performance
# ---------------------------------------------------------------------------
class PerformanceStats:
    """Collecte les temps de réflexion et le nombre de coups joués."""

    def __init__(self, name: str):
        self.name = name
        self.times: list[float] = []
        self.moves = 0
        self.errors = 0

    def add_time(self, duration: float):
        self.times.append(duration)
        self.moves += 1

    def add_error(self):
        self.errors += 1

    def get_average(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0

    def get_total(self) -> float:
        return sum(self.times)

    def get_min(self) -> float:
        return min(self.times) if self.times else 0.0

    def get_max(self) -> float:
        return max(self.times) if self.times else 0.0

# ---------------------------------------------------------------------------
# Fonctions de jeu
# ---------------------------------------------------------------------------
def get_new_ai_move(engine: ChessEngine, board: chess.Board, time_limit: float = 2.0):
    """Demande un coup à l'IA‑Marc V2."""
    try:
        return engine.get_move(board, time_limit=time_limit)
    except Exception as exc:
        print(f"[ERROR] IA‑Marc V2 : {exc}")
        return None


def play_match(
    white_player: str = "new",  # "new" ou "stockfish"
    black_player: str = "stockfish",
    max_moves: int = 200,
    stockfish_level: int = 20,
) -> tuple[PerformanceStats | None, PerformanceStats | None, str | None]:
    """Joue une partie entre IA‑Marc V2 et Stockfish.
    Retourne les statistiques de chaque moteur ainsi que le résultat du jeu
    ("1-0", "0-1" ou "1/2-1/2").
    """
    if not NEW_AI_AVAILABLE:
        print("[ERREUR] IA‑Marc V2 non disponible – abort.")
        return None, None, None

    board = chess.Board()
    # Initialise les moteurs
    new_ai = ChessEngine()
    new_ai.set_level("Club")  # niveau équilibré
    
    # Initialisation de Stockfish (une seule fois par partie)
    stockfish_engine = None
    if white_player == "stockfish" or black_player == "stockfish":
        try:
            stockfish_engine = get_stockfish_engine(level=stockfish_level)
        except Exception:
            return None, None, None

    new_stats = PerformanceStats("IA‑Marc V2")
    stockfish_stats = PerformanceStats("Stockfish")

    try:
        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            move_count += 1
            current_player = white_player if board.turn == chess.WHITE else black_player
            current_stats = new_stats if current_player == "new" else stockfish_stats
            
            # Affichage compact pour éviter de spammer la console
            if move_count % 10 == 0:
                print(f"  ... Coup {board.fullmove_number}")

            start = time.time()
            move = None
            try:
                if current_player == "new":
                    move = get_new_ai_move(new_ai, board, time_limit=2.0)
                elif current_player == "stockfish":
                    if stockfish_engine:
                        result = stockfish_engine.play(board, chess.engine.Limit(time=2.0))
                        move = result.move
                    else:
                         raise RuntimeError("Moteur Stockfish non initialisé")
                else:
                    raise ValueError(f"Joueur inconnu : {current_player}")
                
                duration = time.time() - start
                
                if move is None or move not in board.legal_moves:
                    print(f"[ERREUR] Coup illégal ou nul de {current_player}")
                    current_stats.add_error()
                    break
                
                current_stats.add_time(duration)
                # print(f"  Joue : {move.uci()} ({duration:.3f}s)") # Trop verbeux
                board.push(move)
            except Exception as exc:
                duration = time.time() - start
                print(f"[ERREUR] Exception pendant le tour de {current_player}: {exc}")
                traceback.print_exc()
                current_stats.add_error()
                break
    finally:
        # Nettoyage garanti
        if stockfish_engine:
            stockfish_engine.quit()

    # Retour du résultat du jeu
    return new_stats, stockfish_stats, board.result()

# ---------------------------------------------------------------------------
# Rapport de performance (texte)
# ---------------------------------------------------------------------------
def print_report(new_stats: PerformanceStats, stockfish_stats: PerformanceStats):
    separator = "=" * 70
    print(f"\n{separator}\nRAPPORT DE PERFORMANCE\n{separator}\n")
    for stats in (new_stats, stockfish_stats):
        print(f"{stats.name}:")
        print(f"  Coups joués   : {stats.moves}")
        print(f"  Temps total   : {stats.get_total():.2f}s")
        print(f"  Temps moyen   : {stats.get_average():.3f}s")
        print(f"  Temps min/max : {stats.get_min():.3f}s / {stats.get_max():.3f}s")
        print(f"  Erreurs       : {stats.errors}\n")
    if new_stats.get_average() and stockfish_stats.get_average():
        ratio = new_stats.get_average() / stockfish_stats.get_average()
        if ratio < 1:
            print(f"IA‑Marc V2 est {1/ratio:.2f}× plus rapide que Stockfish.")
        else:
            print(f"Stockfish est {ratio:.2f}× plus rapide que IA‑Marc V2.")
    print(separator)

# ---------------------------------------------------------------------------
# Visualisation graphique
# ---------------------------------------------------------------------------
def plot_results(levels, avg_time_new, avg_time_stock, win_counts):
    """Génère deux graphiques :
    1️⃣ Temps moyen par coup en fonction du niveau de Stockfish.
    2️⃣ Pourcentage de victoires (IA‑Marc, Stockfish, nuls) par niveau.
    Les figures sont sauvegardées dans le répertoire `ai_comparison`.
    """
    # --- Temps moyen ---
    plt.figure(figsize=(10, 6))
    plt.plot(levels, avg_time_new, marker='o', label='IA‑Marc V2')
    plt.plot(levels, avg_time_stock, marker='s', label='Stockfish')
    plt.title('Temps moyen par coup selon le niveau de Stockfish')
    plt.xlabel('Niveau Stockfish (Skill Level)')
    plt.ylabel('Temps moyen (s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    time_plot_path = os.path.join(PROJECT_ROOT, 'ai_comparison', 'time_vs_level.png')
    plt.savefig(time_plot_path)
    plt.close()

    # --- Taux de victoire ---
    win_new = [win_counts[l].get('new_wins', 0) for l in levels]
    win_stock = [win_counts[l].get('stock_wins', 0) for l in levels]
    draws = [win_counts[l].get('draws', 0) for l in levels]
    total = [win_new[i] + win_stock[i] + draws[i] for i in range(len(levels))]
    pct_new = [100 * w / t if t else 0 for w, t in zip(win_new, total)]
    pct_stock = [100 * w / t if t else 0 for w, t in zip(win_stock, total)]
    pct_draw = [100 * d / t if t else 0 for d, t in zip(draws, total)]

    plt.figure(figsize=(10, 6))
    plt.stackplot(levels, pct_new, pct_stock, pct_draw,
                 labels=['IA‑Marc V2', 'Stockfish', 'Nuls'],
                 colors=['#4caf50', '#f44336', '#9e9e9e'])
    plt.title('Pourcentage de résultats par niveau de Stockfish')
    plt.xlabel('Niveau Stockfish (Skill Level)')
    plt.ylabel('Pourcentage du nombre de parties')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    win_plot_path = os.path.join(PROJECT_ROOT, 'ai_comparison', 'winrate_vs_level.png')
    plt.savefig(win_plot_path)
    plt.close()

    print(f"\nGraphiques générés :\n  • {time_plot_path}\n  • {win_plot_path}\n")

# ---------------------------------------------------------------------------
# Expérimentation ciblée : 5 niveaux (0,5,10,15,20) – 10 parties chacun
# ---------------------------------------------------------------------------

def run_targeted_experiments():
    """Joue 10 parties pour chaque niveau de Stockfish spécifié."""
    levels = [0, 5, 10, 15, 20]
    repetitions = 10
    results: dict[int, list[tuple[PerformanceStats, PerformanceStats]]] = {}
    win_counts: dict[int, dict[str, int]] = {}

    for lvl in levels:
        print(f"\n=== Niveau Stockfish {lvl} – {repetitions} parties ===")
        for i in range(repetitions):
            print(f"--- Partie {i+1}/{repetitions} (niveau {lvl}) ---")
            white = os.getenv("WHITE_PLAYER", "new")
            black = os.getenv("BLACK_PLAYER", "stockfish")
            new_stats, stockfish_stats, result = play_match(
                white_player=white,
                black_player=black,
                max_moves=200,
                stockfish_level=lvl,
            )
            if new_stats is None or stockfish_stats is None:
                print("[ERREUR] Match interrompu, on continue avec le suivant.")
                continue
            
            print(f"Résultat partie: {result}")
            results.setdefault(lvl, []).append((new_stats, stockfish_stats))
            if result:
                win_counts.setdefault(lvl, {"new_wins": 0, "stock_wins": 0, "draws": 0})
                if (result == "1-0" and white == "new") or (result == "0-1" and black == "new"):
                    win_counts[lvl]["new_wins"] += 1
                elif (result == "0-1" and white == "new") or (result == "1-0" and black == "new"):
                    win_counts[lvl]["stock_wins"] += 1
                else:
                    win_counts[lvl]["draws"] += 1

    # Rapport agrégé
    print("\n" + "=" * 70)
    print("RAPPORT AGRÉGÉ PAR NIVEAU DE STOCKFISH")
    print("=" * 70)
    avg_time_new = []
    avg_time_stock = []
    for lvl in levels:
        agg_new = PerformanceStats(f"IA‑Marc V2 (lvl {lvl})")
        agg_stock = PerformanceStats(f"Stockfish (lvl {lvl})")
        for ns, ss in results.get(lvl, []):
            agg_new.times.extend(ns.times)
            agg_new.moves += ns.moves
            agg_new.errors += ns.errors
            agg_stock.times.extend(ss.times)
            agg_stock.moves += ss.moves
            agg_stock.errors += ss.errors
        print(f"\nNiveau Stockfish : {lvl}")
        print_report(agg_new, agg_stock)
        if lvl in win_counts:
            wc = win_counts[lvl]
            total = wc["new_wins"] + wc["stock_wins"] + wc["draws"]
            print(f"  Résultats sur {total} parties : IA‑Marc V2 {wc['new_wins']} victoires, "
                  f"Stockfish {wc['stock_wins']} victoires, {wc['draws']} nuls")
        avg_time_new.append(agg_new.get_average())
        avg_time_stock.append(agg_stock.get_average())

    # Génération des graphiques
    plot_results(levels, avg_time_new, avg_time_stock, win_counts)

def main():
    print("=" * 70)
    print("COMPARAISON IA‑Marc V2 vs Stockfish (tests ciblés)")
    print("=" * 70)
    run_targeted_experiments()

if __name__ == "__main__":
    main()
