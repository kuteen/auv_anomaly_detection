"""Configuration schema: load, merge, and validate YAML config files."""

from __future__ import annotations

import copy
import pathlib
from typing import Any, Dict, Optional

import yaml


_DEFAULTS_PATH = pathlib.Path(__file__).parent / "defaults.yaml"
_REPO_ROOT = _DEFAULTS_PATH.parents[2]


# ── Loading and merging ──────────────────────────────────────────────


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        # Recurse only when both sides are mappings, so nested sections merge
        # key by key. Any other value (scalar, list, or type mismatch) replaces
        # the base wholesale rather than being combined.
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_config_path(path_like: str | pathlib.Path, *, anchor: pathlib.Path | None = None) -> pathlib.Path:
    """Resolve a config path against the anchor directory, repo root, then cwd.

    Absolute paths are returned unchanged. Relative paths are tried in
    preference order and the first existing candidate wins. When none
    exist the highest-priority candidate is returned so the caller raises
    a sensible not-found error.
    """
    path = pathlib.Path(path_like)
    if path.is_absolute():
        return path

    # Build candidates in descending priority. The anchor (the directory of
    # the file referencing this path) takes precedence over the repo root,
    # which in turn takes precedence over the current working directory.
    candidates = []
    if anchor is not None:
        candidates.append((anchor / path).resolve())
    candidates.append((_REPO_ROOT / path).resolve())
    candidates.append(path.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_yaml_file(path: pathlib.Path) -> Dict[str, Any]:
    """Parse a YAML file, returning an empty dict when the file is empty."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load defaults and optionally merge an override file."""
    config = _load_yaml_file(_DEFAULTS_PATH)
    config_anchor = _DEFAULTS_PATH.parent

    if path is not None:
        # User overrides take precedence over the packaged defaults.
        config_path = _resolve_config_path(path)
        overrides = _load_yaml_file(config_path)
        config = _deep_merge(config, overrides)
        # Resolve later references relative to the override file's directory.
        config_anchor = config_path.parent

    data_processing_ref = config.get("data", {}).get("data_processing_config")
    if data_processing_ref:
        # Fold in the referenced data_processing config as a lower-priority
        # base so that anything already set in the main config still wins.
        data_processing_path = _resolve_config_path(
            data_processing_ref,
            anchor=config_anchor,
        )
        data_processing_cfg = _load_yaml_file(data_processing_path)
        config = _deep_merge(data_processing_cfg, config)

    return config


# ── Validation ───────────────────────────────────────────────────────


_REQUIRED_SECTIONS = [
    "data",
    "data_processing",
    "windowing",
    "splits",
    "model",
    "runtime",
    "training",
    "output",
]


def validate_config(cfg: Dict[str, Any]) -> None:
    """Raise ``ValueError`` if mandatory sections or keys are missing."""
    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            raise ValueError(f"Missing required config section: '{section}'")

    data_cfg = cfg["data"]
    required_data_keys = [
        "data_processing_config",
        "dataset_manifest",
        "raw_dir",
        "processed_dir",
        "normalisation_stats_path",
        "sensors",
    ]
    for key in required_data_keys:
        if not data_cfg.get(key):
            raise ValueError(f"data.{key} must be set")

    proc_cfg = cfg["data_processing"]
    required_processing_keys = [
        "metadata_columns",
        "time_columns",
        "ddm_columns",
        "min_datapoints_per_group",
        "outlier_chunk_size",
        "outlier_iqr_multiplier",
        "outlier_percentile_lower",
        "outlier_percentile_upper",
        "time_gap_threshold_s",
        "time_gap_fill_s",
        "interpolation_interval_s",
        "normalisation",
        "normalisation_min",
        "normalisation_max",
    ]
    for key in required_processing_keys:
        if proc_cfg.get(key) in (None, "", []):
            raise ValueError(f"data_processing.{key} must be set")

    # The published tensors were produced with a fixed global min-max scaling,
    # so only 'minmax' keeps downstream models consistent with the release.
    if proc_cfg["normalisation"] != "minmax":
        raise ValueError(
            "The public raw-to-tensor pipeline preserves the original global min-max "
            "normalisation. Set data_processing.normalisation to 'minmax'."
        )

    try:
        normalisation_min = float(proc_cfg["normalisation_min"])
        normalisation_max = float(proc_cfg["normalisation_max"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "data_processing.normalisation_min and data_processing.normalisation_max "
            "must be numeric."
        ) from exc
    if normalisation_min >= normalisation_max:
        raise ValueError(
            "data_processing.normalisation_min must be smaller than "
            "data_processing.normalisation_max."
        )

    if cfg["windowing"]["window_length"] < 2:
        raise ValueError("windowing.window_length must be >= 2")

    if cfg["windowing"]["stride"] < 1:
        raise ValueError("windowing.stride must be >= 1")

    valid_models = {
        "isolation_forest", "ocsvm", "elliptic_envelope",
        "lstm_ae", "cnn_ae",
        "tranad",
        "graph_stgnn",
    }
    if cfg["model"]["name"] not in valid_models:
        raise ValueError(
            f"Unknown model '{cfg['model']['name']}'. Choose from {sorted(valid_models)}"
        )

    splits_cfg = cfg["splits"]
    val_fraction = splits_cfg.get("val_fraction")
    if not isinstance(val_fraction, (int, float)) or not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("splits.val_fraction must be a number between 0 and 1")

    valid_faults = {"wing_loss", "sensor_dropout"}
    configured_faults = cfg.get("faults", {}).get("types", [])
    if not configured_faults:
        raise ValueError("faults.types must contain at least one fault type")
    invalid_faults = [fault for fault in configured_faults if fault not in valid_faults]
    if invalid_faults:
        raise ValueError(
            f"Unsupported fault types {invalid_faults}. Choose from {sorted(valid_faults)}"
        )

    valid_graph_builders = {"fixed", "correlation", "learned"}
    if cfg["graph"]["builder"] not in valid_graph_builders:
        raise ValueError(
            f"Unknown graph builder '{cfg['graph']['builder']}'. Choose from {sorted(valid_graph_builders)}"
        )

    if cfg["graph"]["builder"] == "fixed" and not cfg["graph"].get("adjacency_list"):
        raise ValueError("graph.adjacency_list must be set when graph.builder is 'fixed'")

    valid_graph_convs = {"gcn", "graphsage", "gat", "gatv2"}
    graph_conv = cfg.get("model", {}).get("graph_conv", "gcn")
    if graph_conv not in valid_graph_convs:
        raise ValueError(
            f"Unknown model.graph_conv '{graph_conv}'. Choose from {sorted(valid_graph_convs)}"
        )

    valid_temporal_modes = {"transformer", "rnn", "gru", "lstm"}
    temporal_mode = cfg.get("model", {}).get("temporal_mode", "transformer")
    if temporal_mode not in valid_temporal_modes:
        raise ValueError(
            f"Unknown model.temporal_mode '{temporal_mode}'. "
            f"Choose from {sorted(valid_temporal_modes)}"
        )

    runtime_cfg = cfg.get("runtime", {})
    valid_devices = {"auto", "cpu", "cuda"}
    if runtime_cfg.get("device", "auto") not in valid_devices:
        raise ValueError(
            f"Unknown runtime.device '{runtime_cfg.get('device')}'. "
            f"Choose from {sorted(valid_devices)}"
        )

    for key in ["cpu_threads", "interop_threads", "dataloader_workers"]:
        value = runtime_cfg.get(key, "auto")
        # 'auto' defers the count to runtime detection, so skip the numeric check.
        if value == "auto":
            continue
        try:
            parsed = int(value)
            # 0 workers (main-process loading) is valid, thread counts need >= 1.
            minimum = 0 if key == "dataloader_workers" else 1
            if parsed < minimum:
                raise ValueError
        except (TypeError, ValueError) as exc:
            requirement = "a non-negative integer" if key == "dataloader_workers" else "an integer >= 1"
            raise ValueError(f"runtime.{key} must be 'auto' or {requirement}") from exc

    matmul_precision = str(runtime_cfg.get("matmul_precision", "high")).lower()
    valid_precisions = {"highest", "high", "medium"}
    if matmul_precision not in valid_precisions:
        raise ValueError(
            f"Unknown runtime.matmul_precision '{matmul_precision}'. "
            f"Choose from {sorted(valid_precisions)}"
        )

    pin_memory = runtime_cfg.get("pin_memory", "auto")
    if pin_memory not in {"auto", True, False}:
        raise ValueError("runtime.pin_memory must be one of ['auto', true, false]")

    classical_cfg = cfg.get("classical", {})
    for key in [
        "isolation_forest_max_samples",
        "ocsvm_max_train_windows",
        "elliptic_envelope_max_train_windows",
    ]:
        value = classical_cfg.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"classical.{key} must be null or an integer >= 1")

    isolation_forest_n_estimators = classical_cfg.get("isolation_forest_n_estimators")
    if isolation_forest_n_estimators is not None:
        if not isinstance(isolation_forest_n_estimators, int) or isolation_forest_n_estimators < 1:
            raise ValueError(
                "classical.isolation_forest_n_estimators must be null or an integer >= 1"
            )

    training_cfg = cfg.get("training", {})
    batch_size = training_cfg.get("batch_size")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("training.batch_size must be an integer >= 1")

    seeds = training_cfg.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("training.seeds must be a non-empty list of integers")
    if not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("training.seeds must contain only integers")
    if not all(seed >= 0 for seed in seeds):
        raise ValueError("training.seeds must contain only integers >= 0")
