"""Vérifier la disponibilité du GPU"""
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Nombre de GPUs: {torch.cuda.device_count()}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Aucun GPU CUDA détecté")
        print("\nPour installer PyTorch avec CUDA:")
        print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    print("❌ PyTorch n'est pas installé")
    print("\nPour installer PyTorch avec CUDA:")
    print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
