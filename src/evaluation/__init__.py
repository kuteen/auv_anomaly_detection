"""Evaluation sub-package: metrics, split logic, thresholding, and reporting."""

from evaluation.metrics import compute_metrics
from evaluation.protocols import build_protocol_splits
from evaluation.thresholding import apply_threshold
from evaluation.reporting import save_results

__all__ = [
    "compute_metrics",
    "build_protocol_splits",
    "apply_threshold",
    "save_results",
]
