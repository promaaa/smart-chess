"""Debug de la conversion NumPy <-> PyTorch"""
import numpy as np
import torch
from Chess import Chess
from nn_evaluator import load_evaluator_from_file
from torch_nn_evaluator import load_from_npz

# Charger les deux versions
numpy_eval, _ = load_evaluator_from_file('chess_nn_weights.npz')
torch_model, _ = load_from_npz('chess_nn_weights.npz', device='cpu')
torch_model.eval()

# Position de test
chess = Chess()
chess.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

# Encode board (doit être identique)
print("=== ENCODAGE ===")
numpy_encoded = numpy_eval._encode_board(chess)
torch_encoded = torch_model.encode_board(chess, device='cpu').numpy().flatten()

print(f"NumPy shape: {numpy_encoded.shape}")
print(f"Torch shape: {torch_encoded.shape}")
print(f"Encodage identique: {np.allclose(numpy_encoded, torch_encoded)}")
print(f"Diff max encodage: {np.abs(numpy_encoded - torch_encoded).max()}")

# Forward pass détaillé - NumPy
print("\n=== FORWARD NUMPY ===")
x_np = numpy_encoded.reshape(1, -1)
print(f"Input shape: {x_np.shape}")

h1_in_np = np.dot(x_np, numpy_eval.weights1) + numpy_eval.biases1
print(f"h1_in shape: {h1_in_np.shape}, mean: {h1_in_np.mean():.4f}, std: {h1_in_np.std():.4f}")

# LeakyReLU
LEAKY_ALPHA = 0.01
h1_np = np.where(h1_in_np > 0, h1_in_np, LEAKY_ALPHA * h1_in_np)
print(f"h1 (after leaky) mean: {h1_np.mean():.4f}, std: {h1_np.std():.4f}")

h2_in_np = np.dot(h1_np, numpy_eval.weights2) + numpy_eval.biases2
print(f"h2_in shape: {h2_in_np.shape}, mean: {h2_in_np.mean():.4f}, std: {h2_in_np.std():.4f}")

h2_np = np.where(h2_in_np > 0, h2_in_np, LEAKY_ALPHA * h2_in_np)
print(f"h2 (after leaky) mean: {h2_np.mean():.4f}, std: {h2_np.std():.4f}")

out_np = np.dot(h2_np, numpy_eval.weights3) + numpy_eval.biases3
numpy_score_raw = out_np[0][0]
print(f"Output (raw): {numpy_score_raw:.4f}")
numpy_score = numpy_score_raw * 1000.0
print(f"Output (scaled): {numpy_score:.2f}")

# Forward pass détaillé - PyTorch
print("\n=== FORWARD PYTORCH ===")
with torch.no_grad():
    x_torch = torch.from_numpy(numpy_encoded.reshape(1, -1)).float()
    print(f"Input shape: {x_torch.shape}")
    
    # Layer 1
    h1_in_torch = torch_model.l1(x_torch)
    print(f"h1_in shape: {h1_in_torch.shape}, mean: {h1_in_torch.mean():.4f}, std: {h1_in_torch.std():.4f}")
    
    h1_torch = torch_model.leaky_relu(h1_in_torch)
    print(f"h1 (after leaky) mean: {h1_torch.mean():.4f}, std: {h1_torch.std():.4f}")
    
    # NO DROPOUT in eval mode
    
    # Layer 2
    h2_in_torch = torch_model.l2(h1_torch)
    print(f"h2_in shape: {h2_in_torch.shape}, mean: {h2_in_torch.mean():.4f}, std: {h2_in_torch.std():.4f}")
    
    h2_torch = torch_model.leaky_relu(h2_in_torch)
    print(f"h2 (after leaky) mean: {h2_torch.mean():.4f}, std: {h2_torch.std():.4f}")
    
    # Layer 3
    out_torch = torch_model.l3(h2_torch)
    torch_score_raw = out_torch[0, 0].item()
    print(f"Output (raw): {torch_score_raw:.4f}")
    torch_score = torch_score_raw * 1000.0
    print(f"Output (scaled): {torch_score:.2f}")

print("\n=== COMPARAISON ===")
print(f"NumPy score:  {numpy_score:.2f}")
print(f"Torch score:  {torch_score:.2f}")
print(f"Diff:         {abs(numpy_score - torch_score):.2f}")

# Vérifier les poids layer par layer
print("\n=== VÉRIFICATION POIDS ===")
w1_np = numpy_eval.weights1
w1_torch = torch_model.l1.weight.detach().cpu().numpy().T
print(f"W1 identique: {np.allclose(w1_np, w1_torch, atol=1e-5)}")
print(f"W1 diff max: {np.abs(w1_np - w1_torch).max():.8f}")

b1_np = numpy_eval.biases1.flatten()
b1_torch = torch_model.l1.bias.detach().cpu().numpy()
print(f"B1 identique: {np.allclose(b1_np, b1_torch, atol=1e-5)}")
print(f"B1 diff max: {np.abs(b1_np - b1_torch).max():.8f}")

w2_np = numpy_eval.weights2
w2_torch = torch_model.l2.weight.detach().cpu().numpy().T
print(f"W2 identique: {np.allclose(w2_np, w2_torch, atol=1e-5)}")
print(f"W2 diff max: {np.abs(w2_np - w2_torch).max():.8f}")

w3_np = numpy_eval.weights3
w3_torch = torch_model.l3.weight.detach().cpu().numpy().T
print(f"W3 identique: {np.allclose(w3_np, w3_torch, atol=1e-5)}")
print(f"W3 diff max: {np.abs(w3_np - w3_torch).max():.8f}")
