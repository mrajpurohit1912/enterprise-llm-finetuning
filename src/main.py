"""
src/main.py
Entry point for the fine-tuning platform application.
"""

import os
# Prevent PyTorch CUDA memory fragmentation on memory-constrained GPUs (<= 6GB VRAM)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.presentation.cli import main

if __name__ == "__main__":
    main()
