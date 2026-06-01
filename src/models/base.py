"""Common interface for all anomaly-detection models.

Defines the :class:`ScoreResult` container and the abstract
:class:`AnomalyDetector` base class. Every classical and deep baseline
implements ``fit`` and ``score`` against this interface, so the benchmark
harness can train and evaluate models interchangeably. The score sign
convention is shared across the suite, higher values mean more anomalous.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class ScoreResult:
    """Container returned by :meth:`AnomalyDetector.score`.

    Attributes
    ----------
    scores_time_channel : np.ndarray  [T, m]
        Per-timestep, per-channel anomaly scores.
    scores_time_global : np.ndarray  [T]
        Aggregated anomaly score per timestep.
    reconstruction : np.ndarray or None  [T, m]
        Reconstructed time series, if the model supports it.
    """

    scores_time_channel: np.ndarray
    scores_time_global: np.ndarray
    reconstruction: Optional[np.ndarray] = None


class AnomalyDetector(abc.ABC):
    """Base class that every model must implement."""

    # Short identifier used in logs and result keys, overridden per model.
    name: str = "base"

    @abc.abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train the model.

        Returns a dict of training metrics (e.g. final loss, best epoch).
        """

    @abc.abstractmethod
    def score(
        self,
        test_loader: DataLoader,
        config: Dict[str, Any],
    ) -> ScoreResult:
        """Compute anomaly scores on test data.

        Returns a :class:`ScoreResult` whose scores follow the suite-wide
        convention, higher values mean more anomalous.
        """

    def save(self, path: str) -> None:
        """Persist model state.  Override for PyTorch models."""

    def load(self, path: str) -> None:
        """Restore model state."""
