#!/usr/bin/env python3
"""
Profiler simple et efficace pour identifier les goulots d'étranglement.
"""

import cProfile
import pstats
import io
import sys
import os
import time

# Ajouter le dossier parent au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Chess import Chess
from evaluator import ChessEvaluator
from alphabeta_engine import AlphaBetaEngine


def run_performance_profile():
    """Lance le profiling sur le moteur d'échecs"""
    print("🔍 === PROFILER MOTEUR D'ÉCHECS === 🔍\n")
    
    # Position de test simple
    chess = Chess()
    engine = AlphaBetaEngine(max_depth=3, evaluator=ChessEvaluator())
    
    print("Configuration: Position initiale, Profondeur 3")
    print("Démarrage du profiling...\n")
    
    # Profiling
    pr = cProfile.Profile()
    
    start_time = time.time()
    pr.enable()
    
    # Exécuter le moteur
    best_move = engine.get_best_move(chess)
    
    pr.disable()
    elapsed_time = time.time() - start_time
    
    print(f"✅ Profiling terminé en {elapsed_time:.2f}s")
    print(f"🎯 Coup: {engine._format_move(best_move)}")
    print(f"📊 Nœuds: {engine.nodes_evaluated:,}\n")
    
    # Analyser les résultats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats(pstats.SortKey.CUMULATIVE)
    
    print("🏆 === TOP 20 FONCTIONS LES PLUS COÛTEUSES ===")
    print("=" * 80)
    
    # Capturer et afficher le top 20
    ps.print_stats(20)
    
    # Obtenir les statistiques détaillées
    stats_output = s.getvalue()
    
    print("\n🔍 === ANALYSE DÉTAILLÉE ===")
    
    # Analyser spécifiquement les fonctions Chess.py
    chess_functions = []
    copy_functions = []
    
    # Parser les lignes de statistiques
    lines = stats_output.split('\n')
    current_data = []
    
    for line in lines:
        if 'Chess.py' in line or 'copy' in line.lower():
            current_data.append(line.strip())
    
    if current_data:
        print("\n🎯 FONCTIONS CHESS.PY ET COPIES IDENTIFIÉES:")
        print("-" * 60)
        for i, line in enumerate(current_data[:10]):
            print(f"{i+1:2d}. {line}")
    
    # Statistiques brutes pour analyse manuelle
    all_stats = ps.stats
    
    print(f"\n📊 === STATISTIQUES GLOBALES ===")
    print(f"Total fonctions analysées: {len(all_stats)}")
    
    # Top des fonctions par temps total
    sorted_by_time = sorted(all_stats.items(), 
                           key=lambda x: x[1][1], reverse=True)  # x[1][1] = tottime
    
    print(f"\n⏱️  TOP 10 PAR TEMPS D'EXÉCUTION:")
    print("-" * 70)
    print(f"{'Rang':<4} {'Temps (s)':<10} {'Appels':<10} {'Fonction'}")
    print("-" * 70)
    
    for i, (func_key, stats_tuple) in enumerate(sorted_by_time[:10]):
        filename, line_num, func_name = func_key
        ncalls, tottime, cumtime = stats_tuple[:3]
        
        # Nettoyer le nom de fichier
        clean_filename = filename.split('\\')[-1] if '\\' in filename else filename.split('/')[-1]
        
        print(f"{i+1:<4} {tottime:<10.4f} {ncalls:<10} {clean_filename}::{func_name}")
    
    # Recherche spécifique des hotspots
    print(f"\n🔥 HOTSPOTS CRITIQUES:")
    print("-" * 50)
    
    hotspots = []
    for func_key, stats_tuple in sorted_by_time:
        filename, line_num, func_name = func_key
        ncalls, tottime, cumtime = stats_tuple[:3]
        
        # Identifier les fonctions problématiques
        if (('Chess.py' in filename or 'copy' in func_name.lower()) and 
            tottime > 0.001):  # Plus de 1ms
            
            clean_filename = filename.split('\\')[-1] if '\\' in filename else filename.split('/')[-1]
            hotspots.append({
                'file': clean_filename,
                'function': func_name,
                'time': tottime,
                'calls': ncalls,
                'avg_time': tottime / ncalls if ncalls > 0 else 0
            })
    
    for i, hotspot in enumerate(hotspots[:5]):
        print(f"{i+1}. {hotspot['file']}::{hotspot['function']}")
        print(f"   ⏱️  {hotspot['time']:.4f}s total ({hotspot['calls']} appels)")
        print(f"   📊 {hotspot['avg_time']*1000:.2f}ms par appel")
        
        # Recommandations
        if 'copy' in hotspot['function'].lower():
            print("   💡 PROBLÈME: Trop de copies d'objets")
        elif 'move' in hotspot['function'].lower():
            print("   💡 PROBLÈME: Logique de mouvement coûteuse") 
        elif 'undo' in hotspot['function'].lower():
            print("   💡 PROBLÈME: Annulation de coups lente")
        elif 'check' in hotspot['function'].lower():
            print("   💡 PROBLÈME: Vérifications répétitives")
        
        print()
    
    # Recommandations finales
    print("💡 === RECOMMANDATIONS D'OPTIMISATION ===")
    print("1. 🎯 Réduire les copies de bitboards")
    print("2. ⚡ Optimiser move_piece() et undo_move()")  
    print("3. 🧠 Ajouter du cache pour les évaluations")
    print("4. 🔍 Utiliser des références au lieu de copies")
    print("5. 📊 Pré-calculer les patterns fréquents")
    
    # Sauvegarde
    report_name = f"profile_report_{int(time.time())}.txt"
    with open(report_name, 'w') as f:
        f.write("=== RAPPORT DE PROFILING ===\n\n")
        f.write(f"Durée: {elapsed_time:.2f}s\n")
        f.write(f"Nœuds: {engine.nodes_evaluated:,}\n")
        f.write(f"Coup: {engine._format_move(best_move)}\n\n")
        f.write(stats_output)
    
    print(f"\n📄 Rapport détaillé: {report_name}")


if __name__ == "__main__":
    try:
        run_performance_profile()
    except KeyboardInterrupt:
        print("\n⏹️  Profiling interrompu")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()