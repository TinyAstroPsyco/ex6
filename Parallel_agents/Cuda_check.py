# Cuda check
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory Allocated:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
    print("GPU Memory Cached:", torch.cuda.memory_reserved(0) / 1024**2, "MB")
else:
    print("CUDA not available. You're on CPU.")
