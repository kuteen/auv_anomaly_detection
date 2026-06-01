#!/usr/bin/env python3
"""Audit raw parquet coverage and processed tensor readiness.

Produces the reproducibility tables that accompany the benchmark. For every
mission listed in the dataset manifest it reports raw row counts, per-sensor
missingness, the shape of the cached processed tensor, and how those windows
distribute across the configured protocol splits. Outputs are written as CSV
tables plus a short Markdown commentary under the audit report directory.

Run directly, ``python src/workflows/run_data_audit.py``, or import
``run_audit`` for programmatic use.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

# Allow direct script execution: python src/workflows/run_data_audit.py ...
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config import load_config, validate_config
from data.manifest import (
    load_sequence_index,
    mission_records,
    validate_manifest_files,
)
from evaluation.protocols import build_protocol_splits


# ── Coverage summaries ───────────────────────────────────────────────

@dataclass
class AuditPaths:
    """Filesystem locations for the audit run, resolved from CLI arguments."""

    config_path: pathlib.Path
    output_dir: pathlib.Path
    figure_dir: pathlib.Path


def _raw_sensor_missingness(df: pd.DataFrame, sensors: List[str]) -> Dict[str, float]:
    """Return the percentage of missing values per sensor in one raw frame.

    A sensor absent from the frame is treated as fully missing, 100 per cent.
    """
    metrics: Dict[str, float] = {}
    for sensor in sensors:
        if sensor not in df.columns:
            metrics[sensor] = 100.0
        else:
            metrics[sensor] = float(df[sensor].isna().mean() * 100.0)
    return metrics


def _mission_summary(records, sensors: List[str], index_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-mission coverage table.

    For each mission this pairs the raw parquet, read in full, with the cached
    processed tensor, read memory-mapped to avoid loading the whole array. The
    ``index_rows`` column cross-checks the tensor window count against the
    canonical sequence index.

    Returns
    -------
    pandas.DataFrame
        One row per mission, sorted by mission id.
    """
    rows = []
    # Number of indexed windows per mission, used to cross-check the tensors.
    indexed_counts = index_df.groupby("mission_id").size().to_dict()
    for record in records:
        raw_df = pd.read_parquet(record.raw_path)
        tensor = np.load(record.tensor_path, mmap_mode="r")
        rows.append(
            {
                "mission_id": record.mission_id,
                "region": record.region,
                "year": record.year,
                "raw_rows": int(len(raw_df)),
                "raw_missing_pct": float(raw_df[sensors].isna().mean().mean() * 100.0),
                "tensor_windows": int(tensor.shape[0]),
                "tensor_timesteps": int(tensor.shape[0] * tensor.shape[1]),
                "tensor_channels": int(tensor.shape[2]),
                "index_rows": int(indexed_counts.get(record.mission_id, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("mission_id").reset_index(drop=True)


def _sensor_missingness(records, sensors: List[str]) -> pd.DataFrame:
    """Build the per-mission, per-sensor missingness table in raw parquet.

    Returns
    -------
    pandas.DataFrame
        Rows are missions, columns are the sensor missingness percentages.
    """
    rows = []
    for record in records:
        raw_df = pd.read_parquet(record.raw_path)
        stats = _raw_sensor_missingness(raw_df, sensors)
        stats["mission_id"] = record.mission_id
        rows.append(stats)
    columns = ["mission_id", *sensors]
    return pd.DataFrame(rows)[columns].sort_values("mission_id").reset_index(drop=True)


def _split_coverage(cfg: Dict, mission_summary: pd.DataFrame) -> pd.DataFrame:
    """Tabulate how processed windows fall across each protocol split.

    Mission-keyed splits sum the windows of their member missions directly. The
    ``global_random_split`` mode instead has no fixed mission membership, so the
    train/val/test sizes are derived from the same fractions the protocol
    applies at runtime.

    Returns
    -------
    pandas.DataFrame
        One row per split with train, validation, and test window counts.
    """
    split_rows = []
    window_counts = mission_summary.set_index("mission_id")["tensor_windows"].to_dict()
    for split in build_protocol_splits(cfg):
        # Mission-keyed splits, sum the windows of their member missions.
        train_windows = sum(window_counts.get(mid, 0) for mid in split.get("train_ids", []))
        val_windows = sum(window_counts.get(mid, 0) for mid in split.get("val_ids", []))
        test_windows = sum(window_counts.get(mid, 0) for mid in split.get("test_ids", []))
        if split["mode"] == "global_random_split":
            # No fixed membership, recreate the protocol's fractional sizing.
            total = int(sum(window_counts.values()))
            n_test = max(1, int(round(total * 0.30)))
            n_test = min(n_test, total - 2)
            dev = total - n_test
            n_val = max(1, int(round(dev * cfg["splits"].get("val_fraction", 0.10))))
            n_val = min(n_val, dev - 1)
            train_windows = dev - n_val
            val_windows = n_val
            test_windows = n_test
        split_rows.append(
            {
                "split_name": split["name"],
                "train_windows": int(train_windows),
                "val_windows": int(val_windows),
                "test_windows": int(test_windows),
            }
        )
    return pd.DataFrame(split_rows)


# ── Report writing ───────────────────────────────────────────────────

def _write_outputs(
    paths: AuditPaths,
    mission_summary: pd.DataFrame,
    sensor_missingness: pd.DataFrame,
    split_coverage: pd.DataFrame,
) -> None:
    """Write the audit tables as CSV plus a Markdown commentary."""
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.figure_dir.mkdir(parents=True, exist_ok=True)

    mission_summary.to_csv(paths.output_dir / "per_mission_summary.csv", index=False)
    sensor_missingness.to_csv(paths.output_dir / "sensor_missingness_raw_pct.csv", index=False)
    split_coverage.to_csv(paths.output_dir / "split_coverage.csv", index=False)

    commentary = f"""# Data Audit Commentary

- Missions audited: **{len(mission_summary)}**
- Total raw rows: **{int(mission_summary['raw_rows'].sum()):,}**
- Total processed windows: **{int(mission_summary['tensor_windows'].sum()):,}**
- Processed tensor location: `data/processed/`
- Benchmark outputs location: `reports/`

This audit is now based on the supported public workflow only: raw parquet missions plus cached processed tensors. Intermediate CSV preprocessing stages are no longer part of the product surface.
"""
    (paths.output_dir / "DATA_AUDIT_COMMENTARY.md").write_text(commentary, encoding="utf-8")


# ── Audit orchestration ──────────────────────────────────────────────

def run_audit(paths: AuditPaths) -> None:
    """Run the full audit and write its reports.

    Loads and validates the config, confirms every manifest path resolves,
    then builds the mission, sensor, and split coverage tables and persists
    them under ``paths.output_dir``.
    """
    cfg = load_config(str(paths.config_path))
    validate_config(cfg)
    # Both raw parquet and processed tensors must exist for a complete audit.
    validate_manifest_files(cfg["data"]["dataset_manifest"], require_raw=True, require_tensors=True)

    records = mission_records(cfg["data"]["dataset_manifest"])
    sensors = list(cfg["data"]["sensors"])
    index_df = load_sequence_index(cfg["data"]["dataset_manifest"])

    mission_summary = _mission_summary(records, sensors, index_df)
    sensor_missingness = _sensor_missingness(records, sensors)
    split_coverage = _split_coverage(cfg, mission_summary)
    _write_outputs(paths, mission_summary, sensor_missingness, split_coverage)


# ── CLI entry point ──────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse audit command-line arguments."""
    parser = argparse.ArgumentParser(description="Run raw/parquet plus processed-tensor audit.")
    parser.add_argument(
        "--config-path",
        default="src/config/defaults.yaml",
        help="Benchmark config used to infer train/val/test splits.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/data_audit",
        help="Directory for CSV reports and Markdown commentary.",
    )
    parser.add_argument(
        "--figure-dir",
        default="reports/data_audit/charts",
        help="Directory reserved for audit charts.",
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments and run the audit."""
    args = parse_args()
    paths = AuditPaths(
        config_path=pathlib.Path(args.config_path),
        output_dir=pathlib.Path(args.output_dir),
        figure_dir=pathlib.Path(args.figure_dir),
    )
    run_audit(paths)


if __name__ == "__main__":
    main()
