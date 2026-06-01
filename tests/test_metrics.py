"""Evaluation metric suite on synthetic scores and labels."""

from __future__ import annotations

import math

import numpy as np

from evaluation.metrics import (
    compute_metrics,
    detection_latency,
    false_alarm_rate,
)

EXPECTED_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "detection_latency",
    "fa_count",
    "fa_rate_mission",
}


def _synthetic_case(n: int = 200, onset: int = 100):
    rng = np.random.default_rng(0)
    y_true = np.zeros(n, dtype=bool)
    y_true[onset:] = True
    scores = rng.standard_normal(n)
    scores[onset:] += 4.0  # clearly separable anomalies
    y_pred = scores > 1.5
    return y_true, y_pred, scores


def test_compute_metrics_keys_and_ranges() -> None:
    y_true, y_pred, scores = _synthetic_case()
    metrics = compute_metrics(y_true, y_pred, scores=scores)

    assert EXPECTED_KEYS.issubset(metrics.keys())
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        value = metrics[key]
        assert 0.0 <= value <= 1.0, f"{key}={value} out of [0, 1]"

    # A clearly separable case should score well.
    assert metrics["roc_auc"] > 0.9
    assert metrics["f1"] > 0.8
    assert metrics["fa_count"] >= 0


def test_roc_auc_nan_without_both_classes() -> None:
    y_true = np.zeros(50, dtype=bool)  # single class
    scores = np.linspace(0, 1, 50)
    y_pred = scores > 0.5
    metrics = compute_metrics(y_true, y_pred, scores=scores)
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["pr_auc"])


def test_detection_latency_units() -> None:
    y_true = np.zeros(20, dtype=bool)
    y_true[10:] = True
    y_pred = np.zeros(20, dtype=bool)
    y_pred[13:] = True  # detected three steps after onset

    lat_steps = detection_latency(y_true, y_pred, unit="timesteps")
    assert lat_steps == 3.0

    lat_secs = detection_latency(
        y_true, y_pred, unit="seconds", sampling_interval_s=4.0
    )
    assert lat_secs == 12.0


def test_detection_latency_infinite_when_missed() -> None:
    y_true = np.zeros(20, dtype=bool)
    y_true[10:] = True
    y_pred = np.zeros(20, dtype=bool)  # never fires
    assert math.isinf(detection_latency(y_true, y_pred))


def test_false_alarm_rate() -> None:
    y_true = np.zeros(10, dtype=bool)
    y_true[5:] = True
    y_pred = np.zeros(10, dtype=bool)
    y_pred[0] = True  # one false alarm among the five normal samples
    fa = false_alarm_rate(y_true, y_pred, total_hours=2.0)
    assert fa["fa_count"] == 1
    assert fa["fa_rate_mission"] == 1 / 5
    assert fa["fa_rate_per_hour"] == 0.5
