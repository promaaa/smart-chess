import numpy as np
import torch
import torch.nn as nn
from Chess import Chess


class TorchNNEvaluator(nn.Module):
    """PyTorch implementation équivalente du `NeuralNetworkEvaluator` en NumPy.

    - architecture: Linear(input -> hidden) -> LeakyReLU -> Dropout -> Linear(hidden -> hidden) -> LeakyReLU -> Dropout -> Linear(hidden -> out)
    - fournit des helpers pour charger/sauver au format .npz (compatibilité avec l'ancien code NumPy)
    - fournit des helpers pour checkpoint/restore PyTorch (optimizer.state_dict)
    - Support GPU automatique
    """

    def __init__(self, input_size=768, hidden_size=256, output_size=1, dropout=0.3, leaky_alpha=0.01):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_alpha)

        self.piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(self.l1(x))
        x = self.dropout1(x)
        x = self.leaky_relu(self.l2(x))
        x = self.dropout2(x)
        return self.l3(x)

    def encode_board(self, chess_instance: Chess, device='cpu') -> torch.Tensor:
        vec = np.zeros(self.input_size, dtype=np.float32)
        for piece_char, bitboard in chess_instance.bitboards.items():
            if bitboard == 0:
                continue
            piece_index = self.piece_to_index[piece_char]
            temp_bb = int(bitboard)
            while temp_bb:
                square = (temp_bb & -temp_bb).bit_length() - 1
                vector_position = piece_index * 64 + square
                vec[vector_position] = 1.0
                temp_bb &= temp_bb - 1
        t = torch.from_numpy(vec).to(torch.float32).to(device)
        return t.unsqueeze(0)  # shape (1, input_size)

    def evaluate_position(self, chess_instance: Chess, device='cpu') -> float:
        x = self.encode_board(chess_instance, device=device)
        self.to(device)
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        normalized_score = out[0, 0].item()
        EVAL_SCALE_FACTOR = 1000.0
        return normalized_score * EVAL_SCALE_FACTOR


def save_weights_npz(model: TorchNNEvaluator, filename: str, adam_moments: dict = None):
    """Sauvegarde les poids du modèle dans un .npz compatible avec l'ancien format NumPy.

    Le format correspond aux clés attendues par `nn_evaluator.load_evaluator_from_file` :
    - w1: shape (input, hidden)
    - b1: shape (1, hidden)
    - w2, b2, w3, b3
    On convertit les poids PyTorch (weight shape: out, in) en (in, out).
    """
    sd = model.state_dict()
    save_dict = {
        'w1': sd['l1.weight'].cpu().numpy().T,
        'b1': sd['l1.bias'].cpu().numpy().reshape(1, -1),
        'w2': sd['l2.weight'].cpu().numpy().T,
        'b2': sd['l2.bias'].cpu().numpy().reshape(1, -1),
        'w3': sd['l3.weight'].cpu().numpy().T,
        'b3': sd['l3.bias'].cpu().numpy().reshape(1, -1),
    }
    if adam_moments is not None:
        # ensure numpy arrays
        for k, v in dict(adam_moments).items():
            save_dict[k] = np.array(v)

    np.savez(filename, **save_dict)
    if adam_moments is not None:
        print(f"Poids et moments Adam sauvegardés (npz) dans {filename}")
    else:
        print(f"Poids sauvegardés (npz) dans {filename}")


def load_from_npz(filename: str, device='cpu'):
    """Charge un .npz produit par la version NumPy et renvoie (model, adam_moments)

    - adam_moments (si présent) est renvoyé sous forme de dict de tensors (torch.float32)
    - si les moments Adam ne sont pas tous présents, on renvoie None pour adam_moments
    """
    data = np.load(filename)
    # infer sizes
    w1 = data['w1']
    b1 = data['b1']
    w2 = data['w2']
    b2 = data['b2']
    w3 = data['w3']
    b3 = data['b3']
    input_size = int(w1.shape[0])
    hidden_size = int(w1.shape[1])
    output_size = int(w3.shape[1]) if w3.ndim == 2 else 1

    model = TorchNNEvaluator(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    # copy weights (transpose to torch linear layout)
    model.l1.weight.data.copy_(torch.from_numpy(w1.T).to(torch.float32))
    model.l1.bias.data.copy_(torch.from_numpy(b1.reshape(-1)).to(torch.float32))
    model.l2.weight.data.copy_(torch.from_numpy(w2.T).to(torch.float32))
    model.l2.bias.data.copy_(torch.from_numpy(b2.reshape(-1)).to(torch.float32))
    model.l3.weight.data.copy_(torch.from_numpy(w3.T).to(torch.float32))
    model.l3.bias.data.copy_(torch.from_numpy(b3.reshape(-1)).to(torch.float32))

    # collect adam moments if all present
    adam_moments = None
    adam_keys = ['m_w1', 'v_w1', 'm_b1', 'v_b1', 'm_w2', 'v_w2',
                 'm_b2', 'v_b2', 'm_w3', 'v_w3', 'm_b3', 'v_b3', 'adam_step']
    if all(key in data for key in adam_keys):
        adam_moments = {}
        for k in adam_keys:
            val = data[k]
            # convert scalar 0-d to Python int for adam_step
            if k == 'adam_step':
                adam_moments[k] = int(val)
            else:
                adam_moments[k] = torch.from_numpy(np.array(val)).to(torch.float32)

    model.to(device)
    return model, adam_moments


def torch_save_checkpoint(path: str, model: TorchNNEvaluator, optimizer=None, step: int = None):
    ckpt = {'model': model.state_dict()}
    if optimizer is not None:
        ckpt['optim'] = optimizer.state_dict()
    if step is not None:
        ckpt['step'] = int(step)
    torch.save(ckpt, path)
    print(f"Checkpoint PyTorch sauvegardé dans {path}")


def torch_load_checkpoint(path: str, model: TorchNNEvaluator = None, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    optim_state = ckpt.get('optim')
    step = ckpt.get('step')

    if model is None:
        # nothing to load into
        return None, optim_state, step

    # Try the normal strict load first
    try:
        model.load_state_dict(ckpt['model'])
        model.to(device)
        if optimizer is not None and optim_state is not None:
            optimizer.load_state_dict(optim_state)
        return model, optim_state, step
    except RuntimeError as e:
        # Likely a shape mismatch. We'll attempt a best-effort copy of matching
        # parameter slices from the checkpoint into the current model.
        msg = str(e)
        print("⚠️ Warning while loading checkpoint with strict=True:", msg)
        print("Attempting flexible parameter copy (will copy matching slices where possible)...")

    ckpt_sd = ckpt['model']
    model_sd = model.state_dict()
    flexible_copy_performed = False

    def _copy_tensor(dst: torch.Tensor, src: torch.Tensor):
        """Copy as much of src into dst as possible by slicing the leading dims.

        This works for linear layers where weight tensors are (out, in) and bias
        are 1-D. We copy the overlapping block along each dimension.
        """
        if dst.shape == src.shape:
            dst.copy_(src)
            return
        # Determine overlapping slice for each dim
        slices = tuple(slice(0, min(s, d)) for s, d in zip(src.shape, dst.shape))
        try:
            dst[slices].copy_(src[slices])
        except Exception as ex:
            # fallback: try to flatten-copy up to min elements
            n = min(dst.numel(), src.numel())
            dst.view(-1)[:n].copy_(src.view(-1)[:n])

    # Iterate and copy compatible tensors
    for k, dst_tensor in model_sd.items():
        if k in ckpt_sd:
            src_tensor = ckpt_sd[k]
            # Only attempt to copy tensors (skip non-tensor entries)
            if not isinstance(src_tensor, torch.Tensor):
                try:
                    src_tensor = torch.tensor(src_tensor)
                except Exception:
                    continue
            # Ensure same dtype where possible
            if src_tensor.dtype != dst_tensor.dtype:
                try:
                    src_tensor = src_tensor.to(dst_tensor.dtype)
                except Exception:
                    pass
            # Copy overlapping region
            try:
                # detect if shapes differ; mark that we performed a flexible copy
                if tuple(src_tensor.shape) != tuple(dst_tensor.shape):
                    flexible_copy_performed = True
                _copy_tensor(dst_tensor, src_tensor)
                print(f"  Copied params for '{k}' (src {tuple(src_tensor.shape)} -> dst {tuple(dst_tensor.shape)})")
            except Exception as ex:
                print(f"  Skipped '{k}' due to copy error: {ex}")
        else:
            print(f"  Key '{k}' not found in checkpoint; using model init for this param.")

    # Load the modified state dict into model (non-strict because some keys might be missing)
    try:
        model.load_state_dict(model_sd)
        model.to(device)
    except Exception as ex:
        print("Error loading adapted state_dict into model:", ex)

    # Try to restore optimizer state if provided. Restoring optimizer buffers when
    # the model architecture (and thus parameter shapes) changed can lead to
    # shape mismatches inside optimizer tensors and runtime errors during step().
    # Safer approach: only restore optimizer state when the checkpoint parameter
    # tensor shapes exactly match the current model's tensors. Otherwise skip
    # restoring optimizer state and start with a freshly initialized optimizer.
    if optimizer is not None and optim_state is not None:
        try:
            # If we performed any flexible parameter copy above (slicing/expanding),
            # that means the checkpoint and model shapes differed — in that case
            # it's unsafe to restore optimizer buffers. Skip optimizer restore.
            if flexible_copy_performed:
                print("⚠️ Flexible parameter copy was performed (checkpoint/model shape mismatch); skipping optimizer restore to avoid buffer shape errors.")
            else:
                # Quick compatibility check: ensure all parameter tensors that exist in
                # both checkpoint and current model have identical shapes.
                ckpt_model_sd = ckpt.get('model', {})
                shape_mismatch = False
                for k, dst_tensor in model_sd.items():
                    if k in ckpt_model_sd:
                        src_tensor = ckpt_model_sd[k]
                        # ensure tensor-like
                        if not isinstance(src_tensor, torch.Tensor):
                            try:
                                src_tensor = torch.tensor(src_tensor)
                            except Exception:
                                continue
                        if tuple(src_tensor.shape) != tuple(dst_tensor.shape):
                            shape_mismatch = True
                            break

                if shape_mismatch:
                    print("⚠️ Checkpoint parameter shapes differ from the current model; skipping optimizer restore to avoid buffer shape errors.")
                else:
                    # shapes match exactly: safe to restore optimizer state
                    optimizer.load_state_dict(optim_state)
                    print("Optimizer state restored")
        except Exception as ex:
            # Any unexpected error: do not crash — skip optimizer restore
            print("⚠️ Unexpected error while restoring optimizer state; skipping optimizer restore:", ex)

    return model, optim_state, step


if __name__ == '__main__':
    # Exemple d'utilisation similaire au fichier NumPy original
    WEIGHTS_FILE = 'chess_nn_weights_from_torch.npz'
    game = Chess()

    print('--- Création d un réseau PyTorch non entraîné ---')
    model = TorchNNEvaluator(input_size=768, hidden_size=256, output_size=1)
    # évaluation cpu
    score1 = model.evaluate_position(game)
    print(f'Score du réseau PyTorch vierge : {score1:.2f}')

    # sauvegarde en .npz (pour compatibilité)
    save_weights_npz(model, WEIGHTS_FILE)

    # rechargement depuis .npz
    print(f"\n--- Chargement du réseau depuis le fichier '{WEIGHTS_FILE}' ---")
    loaded_model, adam_moms = load_from_npz(WEIGHTS_FILE)
    score2 = loaded_model.evaluate_position(game)
    print(f"Score du réseau chargé : {score2:.2f}")
    if adam_moms is not None:
        print(f"Moments Adam chargés (step={adam_moms['adam_step']})")
    # la valeur peut légèrement différer en float32 mais doit être proche
    try:
        assert abs(score1 - score2) < 1e-3
    except AssertionError:
        print('Attention: score initial et score chargé diffèrent; vérifie dtypes/precision')
