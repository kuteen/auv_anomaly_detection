"""Historical raw-parquet preprocessing pipeline, executed entirely in memory.

The pipeline turns a raw mission parquet frame into the interpolated,
normalised rows that later feed window construction and the benchmark
tensors. The historical processing order is preserved exactly so cached
tensors stay reproducible, the stages are:

1. Convert degree-decimal-minute (DDM) latitude and longitude to decimal
   degrees, and select the benchmark columns of interest.
2. Drop dive groups that are too sparse to model.
3. Trim outliers, first per fixed-size chunk with an IQR rule, then per
   dive with a percentile rule.
4. Synchronise the dual time streams onto a single monotone clock and
   bridge prolonged surface gaps.
5. Interpolate each dive onto the fixed regular time grid.
6. Accumulate and apply per-channel global min-max normalisation.

Most stages group by the mission metadata columns so each dive cycle is
processed independently. ``validate_cached_preprocessing_contract`` guards
against silently mixing tensors built under different normalisation or
sensor contracts.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Cache contract validation ──────────────────────────────────


def validate_cached_preprocessing_contract(cfg: Dict) -> None:
    """Fail when cached tensors were built under a different preprocessing contract."""
    stats_path = pathlib.Path(cfg["data"]["normalisation_stats_path"])
    if not stats_path.exists():
        raise ValueError(f"Missing normalisation stats file: {stats_path}")

    with open(stats_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict):
        raise ValueError(f"Normalisation stats file is malformed: {stats_path}")

    expected_norm = str(cfg["data_processing"]["normalisation"])
    cached_norm = str(payload.get("normalisation", ""))
    if cached_norm != expected_norm:
        raise ValueError(
            "Cached processed tensors were built with a different normalisation mode: "
            f"expected '{expected_norm}', found '{cached_norm or '<missing>'}'."
        )

    expected_min = float(cfg["data_processing"].get("normalisation_min", 0.0))
    expected_max = float(cfg["data_processing"].get("normalisation_max", 1.0))
    cached_min = payload.get("normalisation_min")
    cached_max = payload.get("normalisation_max")
    if cached_min is None or cached_max is None:
        raise ValueError(
            "Cached processed tensors predate the current preprocessing contract "
            f"({stats_path} is missing normalisation_min/max). Rebuild data."
        )
    if abs(float(cached_min) - expected_min) > 1e-9 or abs(float(cached_max) - expected_max) > 1e-9:
        raise ValueError(
            "Cached processed tensors were built with an incompatible normalisation range: "
            f"expected [{expected_min:g}, {expected_max:g}], "
            f"found [{float(cached_min):g}, {float(cached_max):g}]."
        )

    expected_sensors = list(cfg["data"]["sensors"])
    cached_sensors = list(payload.get("sensors", []))
    if cached_sensors != expected_sensors:
        raise ValueError(
            "Cached processed tensors were built with a different sensor contract. "
            "Rebuild data before training."
        )

    cached_stats = payload.get("stats", {})
    missing_stats = [sensor for sensor in expected_sensors if sensor not in cached_stats]
    if missing_stats:
        raise ValueError(
            "Normalisation stats are incomplete for the current sensor contract: "
            f"{missing_stats}"
        )


# ── Coordinate conversion ──────────────────────────────────────


def convert_ddm_to_dd(value: object) -> float | None:
    """Convert a degree-decimal-minute coordinate to decimal degrees.

    Glider position is logged in the ``DDDMM.mmm`` convention where the
    last two integer digits are whole minutes. Blank or non-numeric inputs
    return ``None`` so downstream interpolation can treat them as missing.
    """
    if pd.isna(value) or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    # Preserve hemisphere, then split the magnitude into whole degrees
    # (all but the final two integer digits) and the trailing minutes.
    sign = -1.0 if numeric < 0 else 1.0
    numeric = abs(numeric)
    degrees = int(numeric / 100)
    decimal_minutes = numeric % 100
    return sign * (degrees + decimal_minutes / 60.0)


# ── Column and group selection ─────────────────────────────────


def _ordered_intersection(columns: Iterable[str], available: Iterable[str]) -> List[str]:
    """Return requested columns that exist in ``available``, order preserved."""
    available_set = set(available)
    seen = set()
    ordered: List[str] = []
    for column in columns:
        if column in available_set and column not in seen:
            ordered.append(column)
            seen.add(column)
    return ordered


def _group_iter(df: pd.DataFrame, group_keys: List[str]):
    """Yield an independent copy of each dive-cycle group, sorted by key."""
    # Group by the mission metadata keys so every dive cycle is processed in
    # isolation, dropna=False keeps groups whose keys contain NaN.
    grouped = df.groupby(group_keys, sort=True, dropna=False)
    for _, group in grouped:
        yield group.copy()


def _filter_parameters_of_interest(
    df: pd.DataFrame,
    sensors: List[str],
    metadata_columns: List[str],
    time_columns: List[str],
    ddm_columns: List[str],
) -> pd.DataFrame:
    """Select benchmark columns, decode DDM coordinates, front the metadata.

    Keeps only the metadata, time, sensor, and DDM columns that exist,
    converts the DDM columns to decimal degrees in place, then reorders so
    the metadata columns lead.
    """
    benchmark_columns = metadata_columns + time_columns + sensors + ddm_columns
    selected_columns = _ordered_intersection(benchmark_columns, df.columns)
    result = df[selected_columns].copy()

    # Decode raw DDM latitude/longitude into decimal degrees in place.
    for column in ddm_columns:
        if column in result.columns:
            result.loc[:, column] = result[column].apply(convert_ddm_to_dd).astype(float)

    front_columns = [column for column in metadata_columns if column in result.columns]
    other_columns = [column for column in result.columns if column not in front_columns]
    return result[front_columns + other_columns]


def _filter_sparse_groups(
    df: pd.DataFrame,
    *,
    group_keys: List[str],
    min_datapoints_per_group: int,
) -> pd.DataFrame:
    """Drop dive groups where any column has too few non-null datapoints."""
    kept_groups: List[pd.DataFrame] = []
    for group in _group_iter(df, group_keys):
        # Keep a dive only if every column clears the minimum-density bar.
        if all(group[column].notna().sum() >= min_datapoints_per_group for column in group.columns):
            kept_groups.append(group)

    if not kept_groups:
        return df.iloc[0:0].copy()
    return pd.concat(kept_groups, ignore_index=True)


# ── Outlier removal ────────────────────────────────────────────


def _null_iqr_outliers(
    chunk: pd.DataFrame,
    columns: List[str],
    *,
    iqr_multiplier: float,
) -> pd.DataFrame:
    """Null values outside the Tukey IQR fence for each column in a chunk.

    Bounds are ``Q1 - k*IQR`` and ``Q3 + k*IQR``. Columns with fewer than
    two numeric values are left untouched.
    """
    result = chunk.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        numeric_values = values.dropna()
        if len(numeric_values) < 2:
            continue

        q1 = np.percentile(numeric_values, 25)
        q3 = np.percentile(numeric_values, 75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask = (values < lower) | (values > upper)
        result.loc[mask, column] = np.nan
    return result


def _null_percentile_outliers(
    group: pd.DataFrame,
    columns: List[str],
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> pd.DataFrame:
    """Null values outside the given lower/upper quantiles for each column.

    Quantiles are taken over the whole dive group, columns with fewer than
    two numeric values are left untouched.
    """
    result = group.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        numeric_values = values.dropna()
        if len(numeric_values) < 2:
            continue

        lower = np.percentile(numeric_values, lower_quantile * 100.0)
        upper = np.percentile(numeric_values, upper_quantile * 100.0)
        mask = (values < lower) | (values > upper)
        result.loc[mask, column] = np.nan
    return result


def remove_outliers(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Reproduce the original chunked-IQR + per-dive percentile outlier removal.

    For each dive group the time and sensor columns are first cleaned in
    fixed-size row chunks with an IQR fence, then cleaned once more across
    the whole group with a percentile fence. Group-key columns are never
    treated as outlier candidates. Removed values become ``NaN`` and are
    later filled by interpolation.
    """
    sensors = list(cfg["data"]["sensors"])
    proc_cfg = cfg["data_processing"]
    group_keys = list(proc_cfg["metadata_columns"])
    time_columns = list(proc_cfg["time_columns"])
    chunk_size = int(proc_cfg["outlier_chunk_size"])
    iqr_multiplier = float(proc_cfg["outlier_iqr_multiplier"])
    lower_q = float(proc_cfg["outlier_percentile_lower"])
    upper_q = float(proc_cfg["outlier_percentile_upper"])

    columns_for_outliers = [
        column
        for column in (time_columns + sensors)
        if column in df.columns and column not in group_keys
    ]

    cleaned_groups: List[pd.DataFrame] = []
    for group in _group_iter(df, group_keys):
        working = group.reset_index(drop=True).copy()
        chunked: List[pd.DataFrame] = []
        # First pass: localised IQR fence over consecutive row chunks, this
        # tracks slow drift in sensor baselines across a long dive.
        for start in range(0, len(working), chunk_size):
            stop = start + chunk_size
            chunk = working.iloc[start:stop].copy()
            chunked.append(
                _null_iqr_outliers(
                    chunk,
                    columns_for_outliers,
                    iqr_multiplier=iqr_multiplier,
                )
            )
        working = pd.concat(chunked, ignore_index=True)
        # Second pass: global percentile fence over the whole dive group.
        working = _null_percentile_outliers(
            working,
            columns_for_outliers,
            lower_quantile=lower_q,
            upper_quantile=upper_q,
        )
        cleaned_groups.append(working)

    return pd.concat(cleaned_groups, ignore_index=True) if cleaned_groups else df.iloc[0:0].copy()


# ── Time synchronisation ───────────────────────────────────────


def synchronise_time(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Synchronise the dual time streams and remove prolonged surface gaps.

    Combines the primary and secondary time columns into one monotone
    mission clock measured from the first observed sample, then caps any
    inter-sample gap longer than ``time_gap_threshold_s`` at the fixed
    ``time_gap_fill_s`` so long surface intervals do not stretch the grid.
    """
    proc_cfg = cfg["data_processing"]
    group_keys = list(proc_cfg["metadata_columns"])
    time_columns = list(proc_cfg["time_columns"])
    gap_threshold = float(proc_cfg["time_gap_threshold_s"])
    gap_fill = float(proc_cfg["time_gap_fill_s"])

    result = df.copy()
    result[time_columns] = result[time_columns].apply(pd.to_numeric, errors="coerce")
    # Prefer the primary clock, fall back to the secondary where it is null.
    result["combined_time"] = result[time_columns[0]].fillna(result[time_columns[1]])
    result = result.sort_values(by="combined_time").reset_index(drop=True)

    first_primary = result.groupby(group_keys, sort=True, dropna=False)[time_columns[0]].transform("first")
    first_secondary = result.groupby(group_keys, sort=True, dropna=False)[time_columns[1]].transform("first")
    result["start_time"] = pd.concat([first_primary, first_secondary], axis=1).max(axis=1)
    result["time"] = result["combined_time"] - result["start_time"]
    result = result[result["time"] >= 0].copy()
    result["time_diff"] = result.groupby(group_keys, sort=True, dropna=False)["time"].transform(
        lambda series: series.ffill().diff().fillna(0.0)
    )
    # Cap long surface gaps, then rebuild a monotone clock from the deltas.
    result.loc[result["time_diff"] > gap_threshold, "time_diff"] = gap_fill
    result["time"] = result.groupby(group_keys, sort=True, dropna=False)["time_diff"].cumsum()

    return result.drop(columns=[*time_columns, "start_time", "time_diff"])


# ── Interpolation to fixed grid ────────────────────────────────


def interpolate_data(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Interpolate each dive onto the historical fixed 5-second grid.

    For each dive the regular grid timestamps are unioned with the observed
    rows, sensor columns are linearly interpolated across the union, then
    only the exact grid timestamps are retained. Dives with no valid time
    are skipped. Grouping columns are constant within a dive and are carried
    onto the new grid rows.
    """
    proc_cfg = cfg["data_processing"]
    group_keys = list(proc_cfg["metadata_columns"])
    time_step = float(proc_cfg["interpolation_interval_s"])

    interpolated_groups: List[pd.DataFrame] = []
    for group in _group_iter(df, group_keys):
        working = group.copy()
        max_time = working["time"].max()
        if pd.isna(max_time):
            continue

        # Build the regular target grid and seed it with the constant
        # dive-key values so the new rows align with the original group.
        time_range = np.arange(0, max_time + time_step, time_step, dtype=float)
        new_rows = pd.DataFrame({"time": time_range})
        for column in group_keys:
            new_rows[column] = working[column].iloc[0]

        combined = (
            pd.concat([working, new_rows], ignore_index=True)
            .drop_duplicates(subset="time")
            .sort_values("time")
            .reset_index(drop=True)
        )

        combined = combined.dropna(subset=["time"]).set_index("time")
        columns_to_interpolate = [
            column
            for column in combined.columns
            if column not in (*group_keys, "time")
        ]
        for column in columns_to_interpolate:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
        combined[columns_to_interpolate] = combined[columns_to_interpolate].interpolate(
            method="linear",
            axis=0,
            limit_direction="both",
        )
        combined = combined.reset_index()
        # Keep only the exact grid timestamps, dropping the original samples.
        combined = combined[combined["time"] % time_step == 0].reset_index(drop=True)
        interpolated_groups.append(combined)

    if not interpolated_groups:
        return df.iloc[0:0].copy()
    return pd.concat(interpolated_groups, ignore_index=True)


# ── Pipeline orchestration ─────────────────────────────────────


def prepare_raw_dataframe(raw_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Run the exact historical preprocessing flow, ending at interpolated rows.

    Chains coordinate decoding and column selection, sparse-group removal,
    outlier trimming, time synchronisation, and fixed-grid interpolation.
    Returns an empty frame early when nothing survives the sparse-group
    filter. Normalisation is applied separately by the global min-max
    helpers below.
    """
    sensors = list(cfg["data"]["sensors"])
    proc_cfg = cfg["data_processing"]
    metadata_columns = list(proc_cfg["metadata_columns"])
    time_columns = list(proc_cfg["time_columns"])
    ddm_columns = list(proc_cfg["ddm_columns"])
    min_datapoints_per_group = int(proc_cfg["min_datapoints_per_group"])

    numeric_df = raw_df.apply(pd.to_numeric, errors="coerce")
    numeric_df["combined_time"] = numeric_df[time_columns[0]].fillna(numeric_df[time_columns[1]])
    numeric_df = numeric_df.sort_values(by="combined_time").reset_index(drop=True)

    selected = _filter_parameters_of_interest(
        numeric_df,
        sensors=sensors,
        metadata_columns=metadata_columns,
        time_columns=time_columns,
        ddm_columns=ddm_columns,
    )
    filtered = _filter_sparse_groups(
        selected,
        group_keys=metadata_columns,
        min_datapoints_per_group=min_datapoints_per_group,
    )
    if filtered.empty:
        return filtered

    outliers_removed = remove_outliers(filtered, cfg)
    synced = synchronise_time(outliers_removed, cfg)
    interpolated = interpolate_data(synced, cfg)
    logger.info("Prepared raw mission frame: %s -> %s rows", len(raw_df), len(interpolated))
    return interpolated


# ── Global normalisation ───────────────────────────────────────


def accumulate_global_minmax(
    stats: Dict[str, Dict[str, float]],
    df: pd.DataFrame,
    sensors: List[str],
) -> Dict[str, Dict[str, float]]:
    """Accumulate global min/max sensor statistics across interpolated missions.

    Folds one mission frame into the running ``stats`` mapping so the final
    range spans every mission. ``stats`` is mutated and returned. Sensors
    absent from a frame, or with no numeric values, are skipped.
    """
    for sensor in sensors:
        if sensor not in df.columns:
            continue
        values = pd.to_numeric(df[sensor], errors="coerce").dropna()
        if values.empty:
            continue
        current_min = float(values.min())
        current_max = float(values.max())
        if sensor not in stats:
            stats[sensor] = {"min": current_min, "max": current_max}
        else:
            stats[sensor]["min"] = min(stats[sensor]["min"], current_min)
            stats[sensor]["max"] = max(stats[sensor]["max"], current_max)
    return stats


def apply_global_minmax(
    df: pd.DataFrame,
    stats: Dict[str, Dict[str, float]],
    sensors: List[str],
    *,
    range_min: float = 0.0,
    range_max: float = 1.0,
) -> pd.DataFrame:
    """Apply per-channel global min-max normalisation into a target range.

    Scales each sensor with the accumulated global ``stats`` so values map
    into ``[range_min, range_max]``. Constant sensors are left unchanged
    with a warning, and the helper ``combined_time`` column is dropped.
    """
    normalised = df.copy()
    if "combined_time" in normalised.columns:
        normalised = normalised.drop(columns=["combined_time"])

    scale_span = range_max - range_min
    for sensor in sensors:
        if sensor not in normalised.columns or sensor not in stats:
            continue
        min_value = stats[sensor]["min"]
        max_value = stats[sensor]["max"]
        if min_value == max_value:
            logger.warning("Sensor %s has constant values; leaving unchanged.", sensor)
            continue
        scaled = (normalised[sensor] - min_value) / (max_value - min_value)
        normalised.loc[:, sensor] = range_min + (scaled * scale_span)

    return normalised
