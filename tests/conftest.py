"""Shared fixtures for the smoke-test suite.

All fixtures are tiny and CPU-only. No real data, training, or network
access is required. This file also puts the flat ``src`` layout on the
import path, so bare ``pytest`` runs from the repository root with no
PYTHONPATH and the modules import as top-level names (``from data ...``).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 19 sensors, windows of length 64, processed tensor shape [N, 64, 19].
N_SENSORS = 19
WINDOW_LENGTH = 64

DEFAULTS_CONFIG = REPO_ROOT / "src" / "config" / "defaults.yaml"


@pytest.fixture
def window_tensor() -> np.ndarray:
    """A small synthetic processed window tensor shaped ``[N, 64, 19]``.

    Values are strictly positive so that a zero-fill dropout fault is
    always detectable as a change.
    """
    rng = np.random.default_rng(0)
    arr = rng.uniform(1.0, 5.0, size=(8, WINDOW_LENGTH, N_SENSORS))
    return arr.astype(np.float32)


@pytest.fixture
def correlated_series() -> np.ndarray:
    """A ``[T, 19]`` training matrix with some genuine cross-correlation."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal((300, N_SENSORS))
    # Couple a few channels so correlation edges actually appear.
    base[:, 1] = 0.9 * base[:, 0] + 0.1 * base[:, 1]
    base[:, 2] = 0.8 * base[:, 0] + 0.2 * base[:, 2]
    return base.astype(np.float32)
