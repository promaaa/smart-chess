"""
Script pour convertir les poids NumPy en PyTorch et vice-versa
Assure la compatibilité entre les deux implémentations
"""
import numpy as np
import torch
import os
from torch_nn_evaluator import TorchNNEvaluator, save_weights_npz, load_from_npz
from nn_evaluator import NeuralNetworkEvaluator, load_evaluator_from_file


def numpy_to_torch(numpy_weights_file, torch_checkpoint_file=None, device='cpu'):
    """Convertit les poids NumPy en modèle PyTorch
    
    Args:
        numpy_weights_file: Fichier .npz avec les poids NumPy
        torch_checkpoint_file: Fichier .pt pour sauvegarder (optionnel)
        device: 'cpu' ou 'cuda'
    
    Returns:
        model: TorchNNEvaluator avec les poids chargés
    """
    print(f"📥 Chargement des poids NumPy depuis {numpy_weights_file}...")
    
    if not os.path.exists(numpy_weights_file):
        raise FileNotFoundError(f"Fichier {numpy_weights_file} introuvable!")
    
    # Charger avec la fonction PyTorch
    model, adam_moments = load_from_npz(numpy_weights_file, device=device)
    
    print(f"✅ Modèle PyTorch créé avec succès")
    print(f"   Architecture: 768 → {model.l1.out_features} → {model.l2.out_features} → 1")
    
    if adam_moments:
        print(f"   Moments Adam chargés (step {adam_moments['adam_step']})")
    
    # Sauvegarder en checkpoint PyTorch si demandé
    if torch_checkpoint_file:
        checkpoint = {
            'model': model.state_dict(),
            'step': adam_moments['adam_step'] if adam_moments else 0
        }
        torch.save(checkpoint, torch_checkpoint_file)
        print(f"💾 Checkpoint PyTorch sauvegardé dans {torch_checkpoint_file}")
    
    return model


def torch_to_numpy(model, numpy_weights_file):
    """Convertit un modèle PyTorch en poids NumPy
    
    Args:
        model: TorchNNEvaluator
        numpy_weights_file: Fichier .npz de sortie
    """
    print(f"💾 Sauvegarde des poids PyTorch en format NumPy...")
    save_weights_npz(model, numpy_weights_file)
    print(f"✅ Poids sauvegardés dans {numpy_weights_file}")


def verify_conversion(numpy_file, torch_model=None):
    """Vérifie que la conversion est correcte en comparant les prédictions
    
    Args:
        numpy_file: Fichier .npz avec poids NumPy
        torch_model: Modèle PyTorch (ou None pour charger depuis numpy_file)
    """
    from Chess import Chess
    
    print(f"\n🔍 Vérification de la conversion...")
    
    # Charger version NumPy
    print("  Chargement version NumPy...")
    numpy_evaluator, _ = load_evaluator_from_file(numpy_file)
    
    # Charger/créer version PyTorch
    if torch_model is None:
        print("  Chargement version PyTorch...")
        torch_model, _ = load_from_npz(numpy_file, device='cpu')
    
    # Tester sur plusieurs positions
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Initial
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # Symétrique
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1PP2/5N2/PPPP2PP/RNBQK2R b KQkq - 0 4",  # Italienne
    ]
    
    chess = Chess()
    max_diff = 0.0
    
    print(f"\n  {'Position':<20} {'NumPy':>10} {'PyTorch':>10} {'Diff':>10}")
    print(f"  {'-'*54}")
    
    for fen in test_fens:
        chess.load_fen(fen)
        
        # Évaluation NumPy
        numpy_score = numpy_evaluator.evaluate_position(chess)
        
        # Évaluation PyTorch
        torch_model.eval()
        with torch.no_grad():
            torch_score = torch_model.evaluate_position(chess, device='cpu')
        
        diff = abs(numpy_score - torch_score)
        max_diff = max(max_diff, diff)
        
        pos_name = fen.split()[0][:20]
        print(f"  {pos_name:<20} {numpy_score:>10.2f} {torch_score:>10.2f} {diff:>10.4f}")
    
    print(f"  {'-'*54}")
    print(f"  Différence maximale: {max_diff:.4f} centipawns")
    
    if max_diff < 0.01:
        print(f"  ✅ Conversion parfaite!")
    elif max_diff < 1.0:
        print(f"  ✅ Conversion excellente (différences minimes dues à la précision float)")
    elif max_diff < 10.0:
        print(f"  ⚠️  Différences notables - vérifier la conversion")
    else:
        print(f"  ❌ Conversion incorrecte - différences importantes!")
    
    return max_diff


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convertir entre poids NumPy et PyTorch")
    parser.add_argument('--numpy-file', default='chess_nn_weights.npz', 
                       help='Fichier .npz avec poids NumPy')
    parser.add_argument('--torch-file', default='chess_model_checkpoint.pt',
                       help='Fichier .pt pour checkpoint PyTorch')
    parser.add_argument('--verify', action='store_true',
                       help='Vérifier la conversion')
    parser.add_argument('--to-torch', action='store_true',
                       help='Convertir NumPy → PyTorch')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device pour PyTorch')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CONVERSION POIDS NUMPY ↔ PYTORCH")
    print("="*70)
    
    if args.to_torch:
        model = numpy_to_torch(args.numpy_file, args.torch_file, device=args.device)
        if args.verify:
            verify_conversion(args.numpy_file, model)
    elif args.verify:
        verify_conversion(args.numpy_file)
    else:
        print("Utilise --to-torch pour convertir ou --verify pour vérifier")
        print(f"\nExemple:")
        print(f"  python convert_weights.py --to-torch --verify")
        print(f"  python convert_weights.py --verify")


if __name__ == "__main__":
    # Si lancé sans arguments, faire une conversion + vérification par défaut
    import sys
    if len(sys.argv) == 1:
        print("="*70)
        print("CONVERSION ET VÉRIFICATION AUTOMATIQUE")
        print("="*70)
        
        numpy_file = 'chess_nn_weights.npz'
        torch_file = 'chess_model_checkpoint.pt'
        
        if os.path.exists(numpy_file):
            model = numpy_to_torch(numpy_file, torch_file, device='cpu')
            verify_conversion(numpy_file, model)
        else:
            print(f"❌ Fichier {numpy_file} introuvable!")
            print(f"   Lance d'abord train.py pour créer des poids.")
    else:
        main()
