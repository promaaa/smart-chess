import numpy as np
import torch
import torch.nn as nn
from Chess import Chess


class TorchNNEvaluator(nn.Module):
    """NNUE-like PyTorch evaluator with 4 hidden layers.

    Architecture: 768 → 4096 (ReLU) → 256 (ReLU) → 32 (ReLU) → 1 (Linear)
    """

    def __init__(self, input_size=768, hidden1=4096, hidden2=256, hidden3=32, output_size=1, dropout=0.0):
        super().__init__()
        # NNUE-style architecture with 4 hidden layers
        self.l1 = nn.Linear(input_size, hidden1)
        self.l2 = nn.Linear(hidden1, hidden2)
        self.l3 = nn.Linear(hidden2, hidden3)
        self.l4 = nn.Linear(hidden3, output_size)

        # Use ReLU (NNUE standard) instead of LeakyReLU
        self.act = nn.ReLU()
        
        # Optional dropout (NNUE typically doesn't use dropout, but we keep it configurable)
        self.dropout = dropout
        if dropout > 0:
            self.drop1 = nn.Dropout(p=dropout)
            self.drop2 = nn.Dropout(p=dropout)
            self.drop3 = nn.Dropout(p=dropout)

        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        # piece map used by encode_board
        self._piece_to_index = {c: i for i, c in enumerate('PNBRQKpnbrqk')}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.act(x)
        if self.dropout > 0:
            x = self.drop1(x)
        
        x = self.l2(x)
        x = self.act(x)
        if self.dropout > 0:
            x = self.drop2(x)
        
        x = self.l3(x)
        x = self.act(x)
        if self.dropout > 0:
            x = self.drop3(x)
        
        x = self.l4(x)  # Linear output (no activation)
        return x

    def encode_board(self, chess_instance: Chess, device='cpu') -> torch.Tensor:
        """Encode bitboards into a 768-d float tensor (fast, minimal Python).

        Keeps the original semantics (1.0 where piece present), returns shape (1,768).
        """
        vec = np.zeros(self.input_size, dtype=np.float32)
        for piece_char, bb in chess_instance.bitboards.items():
            if not bb:
                continue
            pi = self._piece_to_index.get(piece_char)
            if pi is None:
                continue
            tb = int(bb)
            while tb:
                lsb = tb & -tb
                square = (lsb.bit_length() - 1)
                vec[pi * 64 + square] = 1.0
                tb &= tb - 1
        t = torch.from_numpy(vec).to(torch.float32).to(device)
        return t.unsqueeze(0)

    def evaluate_position(self, chess_instance: Chess, device='cpu') -> float:
        self.to(device)
        self.eval()
        x = self.encode_board(chess_instance, device=device)
        with torch.no_grad():
            out = self.forward(x)
        return float(out[0, 0].item() * 1000.0)


def _linear_keys(layer_name: str):
    return f'{layer_name}.weight', f'{layer_name}.bias'


def save_weights_npz(model: TorchNNEvaluator, filename: str, adam_moments: dict = None):
    """Save model weights to .npz compatible with the NumPy loader.

    We store weights transposed (in, out) to match NumPy layout, and include any
    provided metadata (adam moments, learning_rate, best_rmse) into the archive.
    """
    sd = model.state_dict()
    save_dict = {}
    for lname in ('l1', 'l2', 'l3', 'l4'):
        wk, bk = _linear_keys(lname)
        w = sd[wk].cpu().numpy().T
        b = sd[bk].cpu().numpy().reshape(1, -1)
        save_dict[f'w{lname[-1]}'] = w
        save_dict[f'b{lname[-1]}'] = b

    if adam_moments:
        for k, v in dict(adam_moments).items():
            save_dict[k] = np.array(v)

    np.savez(filename, **save_dict)
    print(f"Poids sauvegardés (npz) dans {filename}")


def load_from_npz(filename: str, device='cpu'):
    """Load .npz saved by save_weights_npz and return (model, metadata).

    metadata may contain adam moments and optional keys: learning_rate, best_rmse.
    """
    data = np.load(filename)
    w1 = data['w1']; b1 = data['b1']
    w2 = data['w2']; b2 = data['b2']
    w3 = data['w3']; b3 = data['b3']
    w4 = data['w4']; b4 = data['b4']

    # Infer architecture from saved shapes
    input_size = int(w1.shape[0])
    hidden1 = int(w1.shape[1])
    hidden2 = int(w2.shape[1])
    hidden3 = int(w3.shape[1])
    output_size = int(w4.shape[1]) if w4.ndim == 2 else 1

    model = TorchNNEvaluator(
        input_size=input_size, 
        hidden1=hidden1, 
        hidden2=hidden2, 
        hidden3=hidden3, 
        output_size=output_size
    )
    
    model.l1.weight.data.copy_(torch.from_numpy(w1.T).to(torch.float32))
    model.l1.bias.data.copy_(torch.from_numpy(b1.reshape(-1)).to(torch.float32))
    model.l2.weight.data.copy_(torch.from_numpy(w2.T).to(torch.float32))
    model.l2.bias.data.copy_(torch.from_numpy(b2.reshape(-1)).to(torch.float32))
    model.l3.weight.data.copy_(torch.from_numpy(w3.T).to(torch.float32))
    model.l3.bias.data.copy_(torch.from_numpy(b3.reshape(-1)).to(torch.float32))
    model.l4.weight.data.copy_(torch.from_numpy(w4.T).to(torch.float32))
    model.l4.bias.data.copy_(torch.from_numpy(b4.reshape(-1)).to(torch.float32))

    # collect optional metadata
    metadata = {}
    for key in data.files:
        if key not in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4'):
            try:
                metadata[key] = data[key].tolist() if np.asarray(data[key]).shape == () else data[key]
            except Exception:
                metadata[key] = data[key]

    model.to(device)
    return model, (metadata if metadata else None)


def torch_save_checkpoint(path: str, model: TorchNNEvaluator, optimizer=None, step: int = None, best_rmse: float = None):
    ckpt = {'model': model.state_dict()}
    if optimizer is not None:
        ckpt['optim'] = optimizer.state_dict()
    if step is not None:
        ckpt['step'] = int(step)
    if best_rmse is not None:
        ckpt['best_rmse'] = float(best_rmse)
    torch.save(ckpt, path)
    print(f"Checkpoint PyTorch sauvegardé dans {path}")


def torch_load_checkpoint(path: str, model: TorchNNEvaluator = None, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    optim_state = ckpt.get('optim')
    step = ckpt.get('step')

    if model is None:
        return None, optim_state, step

    # strict load if possible
    try:
        model.load_state_dict(ckpt['model'])
        model.to(device)
        if optimizer is not None and optim_state is not None:
            optimizer.load_state_dict(optim_state)
        return model, optim_state, step
    except Exception:
        # fallback: try to copy overlapping params (useful when shapes changed)
        ckpt_sd = ckpt['model']
        model_sd = model.state_dict()
        for k, dst in model_sd.items():
            if k in ckpt_sd:
                src = ckpt_sd[k]
                try:
                    src_t = src if isinstance(src, torch.Tensor) else torch.tensor(src)
                    # copy overlapping slice
                    if src_t.shape == dst.shape:
                        dst.copy_(src_t)
                    else:
                        slices = tuple(slice(0, min(s, d)) for s, d in zip(src_t.shape, dst.shape))
                        dst[slices].copy_(src_t[slices])
                except Exception:
                    continue
        model.load_state_dict(model_sd)
        model.to(device)

    # restore optimizer only if safe (shapes compatible)
    if optimizer is not None and optim_state is not None:
        try:
            optimizer.load_state_dict(optim_state)
            print("Optimizer state restored")
        except Exception:
            print("⚠️ Skipping optimizer restore (incompatible shapes)")

    return model, optim_state, step


if __name__ == '__main__':
    WEIGHTS_FILE = 'chess_nn_weights_from_torch.npz'
    game = Chess()
    model = TorchNNEvaluator()
    save_weights_npz(model, WEIGHTS_FILE)
    loaded, meta = load_from_npz(WEIGHTS_FILE)
    print('Saved and loaded, meta keys:', (meta.keys() if meta else None))
