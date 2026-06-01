"""Threshold calibration strategies.

Thresholds are fitted exclusively on healthy validation scores so that
they reflect the expected distribution of nominal behaviour.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def quantile_threshold(scores: np.ndarray, q: float = 0.99) -> float:
    """Fixed threshold at the *q*-th quantile of healthy validation scores."""
    thr = float(np.quantile(scores, q))
    logger.info("Quantile threshold (q=%.3f): %.6f", q, thr)
    return thr


def rolling_threshold(
    scores: np.ndarray, window: int = 500, k: float = 3.0
) -> np.ndarray:
    """Adaptive threshold: rolling mean + *k* × rolling std.

    Returns an array of the same length as *scores*.
    """
    n = len(scores)
    thresholds = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window + 1)
        seg = scores[start : i + 1]
        thresholds[i] = seg.mean() + k * seg.std()
    # Any residual NaN (e.g. a zero-length window) falls back to the global stats.
    thresholds = np.nan_to_num(thresholds, nan=scores.mean() + k * scores.std())
    return thresholds


def evt_threshold(
    scores: np.ndarray, init_q: float = 0.95, risk: float = 1e-4
) -> float:
    """Peaks-Over-Threshold (POT) via Generalised Pareto Distribution.

    Requires ``scipy``.  Falls back to quantile if fitting fails.
    """
    try:
        from scipy.stats import genpareto
    except ImportError:
        logger.warning("scipy not available – falling back to quantile threshold")
        return quantile_threshold(scores, init_q)

    # POT models the tail above an initial high quantile t0 as a GPD over the
    # exceedances (score minus t0).
    t0 = float(np.quantile(scores, init_q))
    exceedances = scores[scores > t0] - t0
    if len(exceedances) < 10:
        logger.warning("Too few exceedances for EVT – falling back to quantile")
        return quantile_threshold(scores, init_q)

    try:
        # Fit the GPD shape (c) and scale with location pinned at 0.
        c, _, scale = genpareto.fit(exceedances, floc=0)
        n = len(scores)
        n_exc = len(exceedances)
        # Closed-form GPD return level for the target tail risk, anchored at t0.
        thr = t0 + (scale / c) * ((n / n_exc * risk) ** (-c) - 1)
        logger.info("EVT threshold: %.6f  (shape=%.4f, scale=%.4f)", thr, c, scale)
        return float(thr)
    except Exception as exc:
        logger.warning("EVT fit failed (%s) – falling back to quantile", exc)
        return quantile_threshold(scores, init_q)


def select_threshold(
    scores: np.ndarray, config: Dict[str, Any]
) -> float | np.ndarray:
    """Dispatch to the configured thresholding method.

    Returns a scalar for fixed methods or an array for rolling.
    """
    method = config["thresholding"]["method"]
    if method == "quantile":
        return quantile_threshold(scores, config["thresholding"].get("quantile", 0.99))
    elif method == "rolling":
        return rolling_threshold(
            scores,
            window=config["thresholding"].get("rolling_window", 500),
            k=config["thresholding"].get("rolling_k", 3.0),
        )
    elif method == "evt":
        return evt_threshold(
            scores,
            init_q=config["thresholding"].get("evt_threshold_quantile", 0.95),
        )
    else:
        raise ValueError(f"Unknown thresholding method: {method}")
