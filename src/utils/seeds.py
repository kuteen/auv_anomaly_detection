"""Deterministic seeding for reproducibility.

Seeds every random source used in the benchmark from a single integer, the
``random`` and ``numpy`` generators, the ``PYTHONHASHSEED`` environment
variable, and torch on both CPU and any available CUDA devices.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for ``random``, ``numpy``, ``torch`` (CPU + CUDA)."""
    random.seed(seed)
    # Pin hash randomisation so set and dict ordering is reproducible.
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seed every visible CUDA device when one is present.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
