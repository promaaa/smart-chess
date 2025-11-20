#!/usr/bin/env python3
"""
Dataset Generator Rapide - Option B
====================================

Génère un petit dataset de 5,000 positions via self-play rapide.
Plus rapide que télécharger et parser un gros dataset public.

Temps estimé: 30-60 minutes
"""

import sys
import os
# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_main import ChessEngine
import chess
import json
import random
from pathlib import Path

def generate_quick_dataset(num_games=50, time_per_move=0.2):
    """
    Génère rapidement un dataset via self-play.
    
    Args:
        num_games: Nombre de parties (50 = ~5000 positions)
        time_per_move: Temps par coup en secondes (0.2 = rapide)
    """
    print("="*60)
    print("GÉNÉRATION RAPIDE DE DATASET")
    print("="*60)
    print(f"Parties: {num_games}")
    print(f"Temps/coup: {time_per_move}s")
    print(f"Durée estimée: {num_games * 40 * time_per_move / 60:.1f} minutes\n")
    
    engine = ChessEngine()
    engine.set_level("LEVEL6")  # Niveau moyen (6/12)
    
    dataset = []
    total_positions = 0
    
    for game_num in range(num_games):
        board = chess.Board()
        positions = []
        moves_count = 0
        
        # Jouer une partie
        while not board.is_game_over() and moves_count < 150:
            try:
                move = engine.get_move(board, time_limit=time_per_move)
                if move is None:
                    break
                
                # Sauvegarder position (skip opening/endgame)
                if 10 < moves_count < 80 and moves_count % 2 == 0:
                    positions.append(board.fen())
                
                board.push(move)
                moves_count += 1
                
            except Exception as e:
                print(f"  ⚠️ Erreur game {game_num}: {e}")
                break
        
        # Résultat de la partie
        result = board.result()
        if result == "1-0":
            game_result = 1.0
        elif result == "0-1":
            game_result = 0.0
        else:
            game_result = 0.5
        
        # Échantillonner 100 positions max par partie
        sampled = random.sample(positions, min(100, len(positions)))
        
        for fen in sampled:
            dataset.append({
                'fen': fen,
                'result': game_result,
                'game_id': game_num
            })
        
        total_positions += len(sampled)
        
        # Afficher progression
        if (game_num + 1) % 5 == 0:
            print(f"  ✓ Parties: {game_num + 1}/{num_games} | Positions: {total_positions}")
    
    print(f"\n✅ Dataset généré: {len(dataset)} positions")
    return dataset


def convert_to_epd(dataset, output_file='tuning/dataset.epd'):
    """
    Convertit le dataset JSON en format EPD pour les tuners.
    """
    print(f"\nConversion en format EPD...")
    
    with open(output_file, 'w') as f:
        for entry in dataset:
            board = chess.Board(entry['fen'])
            result = entry['result']
            # Format EPD: position c0 "result";
            f.write(f'{board.epd()} c0 "{result}";\n')
    
    print(f"✅ Fichier EPD créé: {output_file}")
    return output_file


def save_dataset_json(dataset, output_file='tuning/dataset.json'):
    """Sauvegarde aussi en JSON pour backup."""
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Backup JSON: {output_file}")


if __name__ == "__main__":
    # Créer le dossier tuning
    Path('tuning').mkdir(exist_ok=True)
    
    # Générer le dataset
    print("\n🚀 Démarrage de la génération...\n")
    dataset = generate_quick_dataset(num_games=50, time_per_move=0.2)
    
    # Sauvegarder
    save_dataset_json(dataset)
    epd_file = convert_to_epd(dataset)
    
    print("\n" + "="*60)
    print("✅ DATASET PRÊT !")
    print("="*60)
    print(f"Fichier EPD: {epd_file}")
    print(f"Positions: {len(dataset)}")
    print("\nProchaine étape: Lancer le tuner")
    print("  python tuning/run_tuner.py")
