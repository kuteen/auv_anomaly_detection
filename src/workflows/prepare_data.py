"""Build canonical processed tensors directly from raw parquet missions.

This is the single supported preprocessing entry-point for the public
workflow. It reads each raw parquet mission named in the dataset manifest,
interpolates and normalises the sensor channels, then slices them into
fixed-length windows cached as ``.npy`` tensors.

The pipeline runs in two passes. Pass one accumulates global min/max
normalisation statistics across every mission so a single shared scale is
applied. Pass two normalises each mission and writes its window tensor plus a
row in the canonical sequence index, which records the raw-to-processed
traceability for every window.

Run with ``python src/workflows/prepare_data.py --config <yaml>``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

# Allow direct script execution: python src/workflows/prepare_data.py ...
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config import load_config, validate_config
from data.manifest import mission_records, sequence_index_path, validate_manifest_files
from data.preprocessing import (
    accumulate_global_minmax,
    apply_global_minmax,
    prepare_raw_dataframe,
)
from utils.io import save_json
from utils.logging import setup_logging
from utils.terminal import ProgressBar, print_banner, print_section, print_summary


# ── Sequence windowing ───────────────────────────────────────────────

def _build_sequences(
    df: pd.DataFrame,
    mission_meta: Dict[str, str],
    sensors: List[str],
    group_keys: List[str],
    window_length: int,
    stride: int,
) -> tuple[np.ndarray, List[Dict[str, object]]]:
    """Create fixed-length windows and the processed-row traceability index.

    Windows never straddle a metadata group boundary, the frame is grouped by
    ``group_keys`` first and each group is sliced independently. Groups shorter
    than one window are skipped.

    Parameters
    ----------
    df : pandas.DataFrame
        Normalised mission frame carrying the sensor and metadata columns.
    mission_meta : dict
        Mission identity fields stamped onto every index row.
    group_keys : list[str]
        Columns that partition the frame into contiguous sequences.
    window_length : int
        Number of timesteps per window.
    stride : int
        Step between successive window start positions.

    Returns
    -------
    tuple[numpy.ndarray, list[dict]]
        The stacked window tensor of shape ``[N, window_length, len(sensors)]``
        and one traceability index row per window.

    Raises
    ------
    ValueError
        If required columns are missing, or the mission yields no full window.
    """
    required_columns = group_keys + ["time", *sensors]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for mission {mission_meta['mission_id']}: {missing}"
        )

    # Keep the original row position so each window records its raw provenance.
    work_df = df.reset_index(drop=False).rename(columns={"index": "__processed_row__"})
    sequences: List[np.ndarray] = []
    index_rows: List[Dict[str, object]] = []
    sequence_idx = 0

    grouped = work_df.groupby(group_keys, sort=True, dropna=False)
    for group_values, group in grouped:
        if len(group) < window_length:
            continue

        # A single group key collapses to a scalar, normalise to a tuple.
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        # Slide a fixed window across the group at the configured stride.
        for start in range(0, len(group) - window_length + 1, stride):
            stop = start + window_length
            window = group.iloc[start:stop]
            sequences.append(window[sensors].to_numpy(dtype=np.float32, copy=True))
            index_rows.append(
                {
                    "sequence_id": f"{mission_meta['mission_id']}_{sequence_idx:06d}",
                    "sequence_idx": sequence_idx,
                    "mission_id": mission_meta["mission_id"],
                    "region": mission_meta["region"],
                    "mission_year": mission_meta["year"],
                    "raw_path": str(mission_meta["raw_path"]),
                    "tensor_path": str(mission_meta["tensor_path"]),
                    "group_year": int(group_values[0]),
                    "group_julian_day": int(group_values[1]),
                    "group_dive_cycle": int(group_values[2]),
                    "processed_row_start": int(window["__processed_row__"].iloc[0]),
                    "processed_row_stop": int(window["__processed_row__"].iloc[-1]),
                    "time_start_s": float(window["time"].iloc[0]),
                    "time_end_s": float(window["time"].iloc[-1]),
                    "n_timesteps": window_length,
                    "n_channels": len(sensors),
                }
            )
            sequence_idx += 1

    if not sequences:
        raise ValueError(
            f"Mission {mission_meta['mission_id']} did not yield any full {window_length}-step windows"
        )

    return np.stack(sequences, axis=0).astype(np.float32, copy=False), index_rows


# ── Normalisation ────────────────────────────────────────────────────

def _save_normalisation_stats(
    stats: Dict[str, Dict[str, float]],
    *,
    cfg: Dict,
) -> None:
    """Persist the global normalisation statistics alongside their config."""
    payload = {
        "normalisation": cfg["data_processing"]["normalisation"],
        "normalisation_min": float(cfg["data_processing"]["normalisation_min"]),
        "normalisation_max": float(cfg["data_processing"]["normalisation_max"]),
        "raw_dir": cfg["data"]["raw_dir"],
        "processed_dir": cfg["data"]["processed_dir"],
        "sensors": list(cfg["data"]["sensors"]),
        "stats": stats,
    }
    save_json(payload, cfg["data"]["normalisation_stats_path"])


# ── Data preparation pipeline ────────────────────────────────────────

def run(config_path: str) -> None:
    """Build cached tensors, stats, and the sequence index from raw parquet.

    Drives the two-pass preparation, pass one fits global normalisation
    statistics, pass two writes per-mission window tensors and the canonical
    sequence index. The previous cache is invalidated first so an interrupted
    rebuild cannot leave a mixed-contract dataset behind.

    Parameters
    ----------
    config_path : str
        Path to the experiment YAML config naming the dataset manifest,
        sensors, windowing, and normalisation settings.
    """
    cfg = load_config(config_path)
    validate_config(cfg)

    validate_manifest_files(
        cfg["data"]["dataset_manifest"],
        require_raw=True,
        require_tensors=False,
    )

    manifest_index_path = sequence_index_path(cfg["data"]["dataset_manifest"])
    normalisation_stats_path = pathlib.Path(cfg["data"]["normalisation_stats_path"])

    sensors = list(cfg["data"]["sensors"])
    group_keys = list(cfg["data_processing"]["metadata_columns"])
    window_length = int(cfg["windowing"]["window_length"])
    stride = int(cfg["windowing"]["stride"])
    records = mission_records(cfg["data"]["dataset_manifest"])
    normalisation_min = float(cfg["data_processing"]["normalisation_min"])
    normalisation_max = float(cfg["data_processing"]["normalisation_max"])

    pathlib.Path(cfg["data"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    manifest_index_path.parent.mkdir(parents=True, exist_ok=True)
    normalisation_stats_path.parent.mkdir(parents=True, exist_ok=True)

    # Invalidate the previous cache up front so an interrupted rebuild cannot
    # accidentally leave behind a seemingly valid mixed-contract dataset.
    manifest_index_path.unlink(missing_ok=True)
    normalisation_stats_path.unlink(missing_ok=True)

    print_banner(
        "AUV Anomaly Detection - Prepare Data",
        [
            ("Config", config_path),
            ("Missions", len(records)),
            ("Sensors", len(sensors)),
            ("Window", f"{window_length} steps"),
            ("Stride", stride),
            ("Raw dir", cfg["data"]["raw_dir"]),
            ("Processed", cfg["data"]["processed_dir"]),
            ("Norm range", f"[{normalisation_min:g}, {normalisation_max:g}]"),
        ],
    )

    # Pass 1, fold every mission into a single shared min/max so all missions
    # are later scaled on the same global range.
    global_stats: Dict[str, Dict[str, float]] = {}
    print_section("Pass 1/2 - Global normalisation stats")
    pass1 = ProgressBar(total=len(records), desc="pass 1/2 missions", unit="mission", leave=True)
    for record in records:
        pass1.set_postfix_str(record.mission_id)
        raw_df = pd.read_parquet(record.raw_path)
        interpolated_df = prepare_raw_dataframe(raw_df, cfg)
        if interpolated_df.empty:
            raise ValueError(f"Mission {record.mission_id} produced no interpolated rows")
        global_stats = accumulate_global_minmax(global_stats, interpolated_df, sensors)
        pass1.update(1)
    pass1.close()

    missing_stats = [sensor for sensor in sensors if sensor not in global_stats]
    if missing_stats:
        raise ValueError(f"Missing global normalisation stats for sensors: {missing_stats}")
    _save_normalisation_stats(global_stats, cfg=cfg)

    # Pass 2, normalise each mission with the shared stats and cache its
    # window tensor, collecting all traceability rows for one combined index.
    all_index_rows: List[Dict[str, object]] = []
    print_section("Pass 2/2 - Mission tensors")
    pass2 = ProgressBar(total=len(records), desc="pass 2/2 missions", unit="mission", leave=True)
    for record in records:
        pass2.set_postfix_str(record.mission_id)
        raw_df = pd.read_parquet(record.raw_path)
        interpolated_df = prepare_raw_dataframe(raw_df, cfg)
        normalised_df = apply_global_minmax(
            interpolated_df,
            global_stats,
            sensors,
            range_min=normalisation_min,
            range_max=normalisation_max,
        )
        tensor, index_rows = _build_sequences(
            df=normalised_df,
            mission_meta={
                "mission_id": record.mission_id,
                "region": record.region,
                "year": record.year,
                "raw_path": record.raw_path,
                "tensor_path": record.tensor_path,
            },
            sensors=sensors,
            group_keys=group_keys,
            window_length=window_length,
            stride=stride,
        )
        pathlib.Path(record.tensor_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(record.tensor_path, tensor)
        pass2.write(f"  Saved {record.tensor_path.name} {tuple(tensor.shape)}")
        all_index_rows.extend(index_rows)
        pass2.update(1)
    pass2.close()

    # Write the combined index to a temp file then atomically swap it in, so a
    # partial write never leaves a corrupt index alongside good tensors.
    index_df = pd.DataFrame(all_index_rows)
    tmp_index_path = manifest_index_path.with_name(f".{manifest_index_path.name}.tmp")
    index_df.to_parquet(tmp_index_path, index=False)
    tmp_index_path.replace(manifest_index_path)
    print_summary(
        "Preparation complete",
        [
            ("Missions", len(records)),
            ("Sequences", len(index_df)),
            ("Stats", cfg["data"]["normalisation_stats_path"]),
            ("Index", manifest_index_path),
            ("Processed", cfg["data"]["processed_dir"]),
            ("Channels", len(sensors)),
        ],
    )


# ── CLI entry point ──────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse data-preparation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare cached ML tensors directly from raw parquet missions.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    """Configure logging, parse arguments, and run preparation."""
    args = parse_args()
    setup_logging()
    run(args.config)


if __name__ == "__main__":
    main()
