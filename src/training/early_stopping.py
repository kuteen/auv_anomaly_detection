"""Early stopping callback."""

from __future__ import annotations


class EarlyStopping:
    """Stop training when the monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after the last improvement.
    min_delta : float
        Minimum decrease in the metric to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best: float = float("inf")
        self._counter: int = 0

    def step(self, metric: float) -> bool:
        """Return *True* if training should stop."""
        if metric < self._best - self.min_delta:
            self._best = metric
            self._counter = 0
            return False
        self._counter += 1
        return self._counter >= self.patience

    def reset(self) -> None:
        """Clear the best metric and patience counter for a fresh run."""
        self._best = float("inf")
        self._counter = 0
