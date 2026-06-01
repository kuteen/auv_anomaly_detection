"""Result saving: CSV, LaTeX tables, summary markdown, plots."""

from __future__ import annotations

import csv
import logging
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_dir(path: pathlib.Path) -> None:
    """Create the output directory, including parents, if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_results(
    metrics_list: List[Dict[str, Any]],
    output_dir: str | pathlib.Path,
    model_names: Optional[List[str]] = None,
    save_csv: bool = True,
    save_latex: bool = True,
    save_md: bool = True,
) -> None:
    """Persist evaluation results in multiple formats.

    Parameters
    ----------
    metrics_list : list of dict
        One dict per model (or per run).  Each dict should contain at
        least ``model``, ``precision``, ``recall``, ``f1``, ``roc_auc``.
    output_dir : path
    """
    out = pathlib.Path(output_dir)
    _ensure_dir(out)

    if not metrics_list:
        logger.warning("No metrics to save")
        return

    keys = list(metrics_list[0].keys())

    # ── CSV ──────────────────────────────────────────────────────────
    if save_csv:
        csv_path = out / "results.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics_list)
        logger.info("Saved CSV: %s", csv_path)

    # ── LaTeX ────────────────────────────────────────────────────────
    if save_latex:
        tex_path = out / "results.tex"
        with open(tex_path, "w") as fh:
            fh.write("\\begin{table}[htbp]\n")
            fh.write("\\caption{Benchmark results}\n")
            fh.write("\\begin{center}\n")
            cols = " ".join(["c"] * len(keys))
            fh.write(f"\\begin{{tabular}}{{{cols}}}\n")
            fh.write("\\toprule\n")
            fh.write(" & ".join(f"\\textbf{{{k}}}" for k in keys) + " \\\\\n")
            fh.write("\\midrule\n")
            for row in metrics_list:
                vals = []
                for k in keys:
                    v = row[k]
                    if isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                fh.write(" & ".join(vals) + " \\\\\n")
            fh.write("\\bottomrule\n")
            fh.write("\\end{tabular}\n")
            fh.write("\\end{center}\n")
            fh.write("\\end{table}\n")
        logger.info("Saved LaTeX: %s", tex_path)

    # ── Markdown ─────────────────────────────────────────────────────
    if save_md:
        md_path = out / "results.md"
        with open(md_path, "w") as fh:
            fh.write("# Benchmark Results\n\n")
            fh.write("| " + " | ".join(keys) + " |\n")
            fh.write("| " + " | ".join(["---"] * len(keys)) + " |\n")
            for row in metrics_list:
                vals = []
                for k in keys:
                    v = row[k]
                    if isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                fh.write("| " + " | ".join(vals) + " |\n")
        logger.info("Saved Markdown: %s", md_path)
