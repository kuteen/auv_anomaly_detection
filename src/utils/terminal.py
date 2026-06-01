"""Small terminal UX helpers for the project CLI.

Groups together the presentation utilities used by the benchmark command line,
terminal width detection, metric formatting, boxed banners and section
headings, and a thin progress-bar wrapper around ``tqdm``. When ``tqdm`` is not
installed the progress helpers degrade gracefully to plain iteration and
printing, so the CLI remains usable without the optional dependency.
"""

from __future__ import annotations

import math
import shutil
from typing import Iterable, Iterator, Optional, Sequence, Tuple, TypeVar

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None
    HAS_TQDM = False

T = TypeVar("T")


# ── Terminal width and formatting ───────────────────────────────────────


def terminal_width(default: int = 100) -> int:
    """Return the current terminal width in columns.

    Args:
        default: Width used when the terminal size cannot be detected, for
            example when output is redirected to a file or pipe.
    """
    return shutil.get_terminal_size((default, 24)).columns


def format_metric(value: object, *, digits: int = 4, missing: str = "n/a") -> str:
    """Format a numeric metric for CLI display.

    Non-numeric values and NaN floats render as the ``missing`` placeholder so
    summary tables stay aligned.

    Args:
        value: Metric value to render, any non-numeric type is treated as missing.
        digits: Number of decimal places for numeric values.
        missing: Placeholder string for missing or NaN values.
    """
    if isinstance(value, (int, float)):
        # NaN is a float but should display as missing rather than "nan".
        if isinstance(value, float) and math.isnan(value):
            return missing
        return f"{float(value):.{digits}f}"
    return missing


# ── Banners and section headings ────────────────────────────────────────


def print_banner(title: str, rows: Optional[Sequence[Tuple[str, object]]] = None) -> None:
    """Print a boxed banner with optional key-value rows.

    Args:
        title: Heading shown inside the box.
        rows: Optional ``(label, value)`` pairs printed under the title.
    """
    # Clamp the rule width so banners stay readable on both narrow and very
    # wide terminals.
    width = max(72, min(terminal_width(), 120))
    line = "=" * width
    print()
    print(line)
    print(f"  {title}")
    print(line)
    for label, value in rows or ():
        print(f"  {label:<12} {value}")
    print(line)
    print()


def print_section(title: str) -> None:
    """Print a section heading styled for terminal output."""
    width = max(72, min(terminal_width(), 120))
    prefix = f"-- {title} "
    # Pad with rule characters out to the clamped width, never negative.
    remaining = max(0, width - len(prefix))
    print()
    print(prefix + ("-" * remaining))


def print_summary(title: str, rows: Sequence[Tuple[str, object]]) -> None:
    """Print a compact closing summary with key-value rows.

    Args:
        title: Heading shown above the rows.
        rows: ``(label, value)`` pairs printed under the title.
    """
    width = max(72, min(terminal_width(), 120))
    line = "-" * width
    print()
    print(line)
    print(f"  {title}")
    print(line)
    for label, value in rows:
        print(f"  {label:<12} {value}")
    print(line)
    print()


# ── Progress reporting ──────────────────────────────────────────────────


class ProgressBar:
    """Small wrapper around tqdm with a no-op fallback.

    When ``tqdm`` is unavailable the bar still tracks ``count`` and routes
    ``write`` through ``print``, so callers need no conditional logic.
    """

    def __init__(
        self,
        *,
        total: int,
        desc: str,
        unit: str = "item",
        leave: bool = False,
    ) -> None:
        """Create a progress bar.

        Args:
            total: Expected number of units of work.
            desc: Label shown to the left of the bar.
            unit: Noun for one unit of work, shown in the rate display.
            leave: Whether to keep the finished bar on screen.
        """
        self.total = total
        self.desc = desc
        self.count = 0
        self._bar = None
        # Only build a real tqdm bar when the dependency is present, otherwise
        # the instance acts as a lightweight counter.
        if HAS_TQDM:
            self._bar = tqdm(
                total=total,
                desc=f"  {desc}",
                unit=unit,
                dynamic_ncols=True,
                leave=leave,
            )

    def update(self, n: int = 1) -> None:
        """Advance the bar by ``n`` units and keep the local count in sync."""
        self.count += n
        if self._bar is not None:
            self._bar.update(n)

    def set_postfix_str(self, text: str) -> None:
        """Set the trailing status text shown after the bar."""
        if self._bar is not None:
            self._bar.set_postfix_str(text)

    def write(self, message: str) -> None:
        """Emit a message without corrupting the bar, falling back to ``print``."""
        if self._bar is not None:
            tqdm.write(message)
        else:
            print(message)

    def close(self) -> None:
        """Close the underlying bar, a no-op without tqdm."""
        if self._bar is not None:
            self._bar.close()


def iter_progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: Optional[int] = None,
    unit: str = "item",
    leave: bool = False,
) -> Iterator[T]:
    """Yield items from ``iterable``, wrapping in a tqdm bar when available.

    Args:
        iterable: Source items to iterate.
        desc: Label shown to the left of the bar.
        total: Expected item count, useful when ``iterable`` has no ``len``.
        unit: Noun for one item, shown in the rate display.
        leave: Whether to keep the finished bar on screen.

    Yields:
        Each item of ``iterable`` in order.
    """
    if HAS_TQDM:
        yield from tqdm(
            iterable,
            total=total,
            desc=f"  {desc}",
            unit=unit,
            dynamic_ncols=True,
            leave=leave,
        )
        return
    # Plain pass-through when tqdm is not installed.
    yield from iterable
