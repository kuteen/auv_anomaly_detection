"""Core evaluation metrics for anomaly detection."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

logger = logging.getLogger(__name__)


def detection_latency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit: str = "timesteps",
    sampling_interval_s: float = 4.0,
) -> float:
    """Time from true fault onset to first detection.

    Parameters
    ----------
    y_true : boolean array [T]
    y_pred : boolean array [T]
    unit : ``"timesteps"`` or ``"seconds"``
    sampling_interval_s : float  – used when *unit* is ``"seconds"``.

    Returns
    -------
    float – latency in the chosen unit, or ``np.inf`` if never detected.
    """
    true_onset = np.argmax(y_true)  # first True
    # argmax returns 0 for an all-False array, so confirm an anomaly truly exists.
    if not y_true[true_onset]:
        return 0.0  # no anomaly in ground truth

    # Search only from the fault onset onwards, earlier positives are not detections.
    detected_after = np.where(y_pred[true_onset:])[0]
    if len(detected_after) == 0:
        return float("inf")  # fault never flagged

    lat = int(detected_after[0])
    if unit == "seconds":
        lat = lat * sampling_interval_s
    return float(lat)


def false_alarm_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_hours: Optional[float] = None,
) -> Dict[str, float]:
    """False-alarm rate per mission and (optionally) per hour.

    Returns
    -------
    dict with keys ``"fa_count"``, ``"fa_rate_mission"``, ``"fa_rate_per_hour"``.
    """
    normal_mask = ~y_true.astype(bool)
    fa = int(y_pred[normal_mask].sum())
    n_normal = int(normal_mask.sum())
    rate_mission = fa / max(n_normal, 1)
    result: Dict[str, float] = {
        "fa_count": fa,
        "fa_rate_mission": rate_mission,
    }
    if total_hours is not None and total_hours > 0:
        result["fa_rate_per_hour"] = fa / total_hours
    return result


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None,
    sampling_interval_s: float = 4.0,
    total_hours: Optional[float] = None,
    latency_unit: str = "timesteps",
) -> Dict[str, Any]:
    """Compute the full metric suite.

    Parameters
    ----------
    y_true : bool/int array [T]  – ground-truth labels.
    y_pred : bool/int array [T]  – binary predictions.
    scores : float array [T], optional – continuous anomaly scores
        (needed for ROC-AUC and PR-AUC).
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # ROC-AUC and PR-AUC are undefined when only one class is present, so guard
    # on both classes being represented and emit NaN otherwise.
    if scores is not None and y_true.sum() > 0 and (~y_true).sum() > 0:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, scores)
        metrics["pr_auc"] = float(auc(rec_arr, prec_arr))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    metrics["detection_latency"] = detection_latency(
        y_true, y_pred, unit=latency_unit, sampling_interval_s=sampling_interval_s,
    )

    fa = false_alarm_rate(y_true, y_pred, total_hours=total_hours)
    metrics.update(fa)

    return metrics
