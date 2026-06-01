"""Shared utility helpers for the benchmark.

Bundles the small cross-cutting helpers used across the project, logging setup,
runtime and device configuration, deterministic seeding, IO, wall-clock timing,
seed-level statistics, and terminal UX. The most commonly used entry points are
re-exported here for convenient ``from utils import ...`` access.
"""

from utils.seeds import set_global_seed
from utils.logging import setup_logging
from utils.runtime import RuntimeContext, configure_runtime
from utils.stats import aggregate_numeric_dicts, summarize_numeric_values, t_critical_95
from utils.terminal import format_metric, print_banner, print_section, print_summary

__all__ = [
    "aggregate_numeric_dicts",
    "RuntimeContext",
    "configure_runtime",
    "format_metric",
    "print_banner",
    "print_section",
    "print_summary",
    "set_global_seed",
    "setup_logging",
    "summarize_numeric_values",
    "t_critical_95",
]
