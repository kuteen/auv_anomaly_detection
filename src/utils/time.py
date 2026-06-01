"""Simple wall-clock timing helpers for the project CLI.

Provides a context manager that measures elapsed wall-clock time for a block
of work and logs the duration on exit. Timing uses ``time.perf_counter`` so it
is monotonic and unaffected by system clock adjustments.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class Timer:
    """Context manager that logs elapsed time.

    Usage::

        with Timer("Training"):
            model.fit(...)
    """

    def __init__(self, label: str = "Block") -> None:
        """Create a timer.

        Args:
            label: Human-readable name used in the completion log line.
        """
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer and return ``self`` for ``as`` binding."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        """Record the elapsed time on ``self.elapsed`` and log it."""
        self.elapsed = time.perf_counter() - self._start
        logger.info("%s completed in %.2f s", self.label, self.elapsed)
