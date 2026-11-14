"""
src/main.py
Entry point for the fine-tuning platform application.
"""

import os
# Prevent PyTorch CUDA memory fragmentation on memory-constrained GPUs (<= 6GB VRAM)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Disable experimental XetHub CAS protocol for model uploads (uses robust Git-LFS)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from src.presentation.cli import main

if __name__ == "__main__":
    main()
