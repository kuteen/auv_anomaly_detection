"""Statistical helpers for seed-level benchmark aggregation.

Aggregates metrics across repeated seeds into mean, standard deviation and a
95% confidence interval. The interval uses a small lookup of two-sided
Student-t critical values rather than SciPy, keeping the dependency footprint
light, and falls back to the normal approximation for large samples.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
_T_CRITICAL_95_INFINITY = 1.960


def _coerce_finite_values(values: Iterable[Any]) -> List[float]:
    """Return a list of finite numeric values as floats."""
    clean: List[float] = []
    for value in values:
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if math.isfinite(numeric):
                clean.append(numeric)
    return clean


def t_critical_95(df: int) -> float:
    """Return the 95% two-sided Student-t critical value for ``df``.

    Args:
        df: Degrees of freedom, typically ``n - 1``.

    Returns:
        The tabulated critical value, or the normal-approximation value of
        1.960 for degrees of freedom beyond the lookup table.
    """
    if df <= 0:
        return 0.0
    if df in _T_CRITICAL_95:
        return _T_CRITICAL_95[df]
    # Above 30 df the t-distribution is close to normal, use the z value.
    return _T_CRITICAL_95_INFINITY


def summarize_numeric_values(values: Sequence[Any]) -> Dict[str, float | int]:
    """Summarize one numeric sample with mean, std, and 95% CI.

    Non-finite and non-numeric entries are dropped before summarising.

    Returns:
        A dict with ``n``, ``mean``, ``std``, ``sem``, ``ci95_halfwidth`` and
        the interval bounds ``ci95_low`` and ``ci95_high``, or an empty dict
        when no finite values remain.
    """
    clean = _coerce_finite_values(values)
    if not clean:
        return {}

    arr = np.asarray(clean, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean())
    # Sample standard deviation (ddof=1), undefined for a single value.
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    # Standard error of the mean, std scaled by 1/sqrt(n).
    sem = float(std / math.sqrt(n)) if n > 0 else 0.0
    # CI half-width = t_(0.975, n-1) * SEM, zero when the sample is too small.
    ci95_halfwidth = float(t_critical_95(n - 1) * sem) if n > 1 else 0.0

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_halfwidth": ci95_halfwidth,
        "ci95_low": mean - ci95_halfwidth,
        "ci95_high": mean + ci95_halfwidth,
    }


def aggregate_numeric_dicts(
    metrics: Sequence[Dict[str, Any]],
    numeric_keys: Sequence[str],
) -> Dict[str, float]:
    """Aggregate selected numeric keys over a sequence of metric dicts."""
    aggregated: Dict[str, float] = {}
    for key in numeric_keys:
        summary = summarize_numeric_values(metric.get(key) for metric in metrics)
        for suffix, value in summary.items():
            aggregated[f"{key}_{suffix}"] = value
    return aggregated
