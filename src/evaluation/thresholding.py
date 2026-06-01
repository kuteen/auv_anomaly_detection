"""Apply a threshold (scalar or array) to continuous scores."""

from __future__ import annotations

import numpy as np


def apply_threshold(
    scores: np.ndarray,
    threshold: float | np.ndarray,
) -> np.ndarray:
    """Return a boolean prediction array.

    Parameters
    ----------
    scores : np.ndarray  [T]
    threshold : scalar or array of same length

    Returns
    -------
    np.ndarray  bool [T]
    """
    return scores > threshold
