"""
Reproducibility seed utility.

Kept from Phase 2 notebook ``set_seed()``.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic behaviour (may hurt perf; toggle if needed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
