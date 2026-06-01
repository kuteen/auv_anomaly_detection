"""Logging configuration with timestamps.

Configures the root logger with a timestamped format on stdout. The level
comes from the explicit argument, then the ``AUVAD_LOG_LEVEL`` environment
variable, defaulting to ``INFO``. A few chatty third-party loggers are pinned
to ``WARNING`` to keep benchmark output readable.
"""

from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: int | None = None) -> None:
    """Configure the root logger with a timestamped format.

    Args:
        level: Explicit logging level, when ``None`` the level is taken from
            ``AUVAD_LOG_LEVEL`` and otherwise defaults to ``INFO``.
    """
    env_level = os.getenv("AUVAD_LOG_LEVEL", "").upper()
    resolved_level = level
    # Fall back to the env var, then INFO, when no explicit level is given.
    if resolved_level is None:
        resolved_level = getattr(logging, env_level, logging.INFO) if env_level else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=resolved_level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Silence verbose third-party loggers so benchmark output stays readable.
    for noisy_logger in ("matplotlib", "numexpr", "numexpr.utils", "PIL"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
