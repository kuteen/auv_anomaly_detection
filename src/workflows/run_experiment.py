"""Run one or more real-data experiments for a model and aggregate across seeds.

This workflow ties the benchmark together end to end. It loads cached processed
tensors, carves protocol splits, trains (or reloads) one model per split, calibrates
a detection threshold on healthy validation scores, injects synthetic faults into the
held-out test windows, scores them, and reports metrics. A single seed produces a
per-run summary, multiple seeds are aggregated with confidence intervals across seeds.

The file reads top to bottom as a pipeline. Section banners group it into phases,
configuration and setup, data preparation, model construction, training, fault
injection, scoring and metrics, aggregation, the ``run()`` entrypoint, and the
argparse and main block.
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import pathlib
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Allow direct script execution: python src/workflows/run_experiment.py ...
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config import load_config, validate_config
from data import validate_cached_preprocessing_contract
from data.faults import inject_fault
from data.graph_builders import (
    CorrelationGraphBuilder,
    FixedGraphBuilder,
    LearnedGraphBuilder,
)
from data.manifest import (
    manifest_by_mission,
    sequence_counts_by_mission,
    validate_manifest_files,
)
from evaluation.metrics import compute_metrics
from evaluation.protocols import build_protocol_splits
from evaluation.reporting import save_results
from evaluation.thresholding import apply_threshold
from training.calibration import select_threshold
from utils.io import save_json, save_yaml
from utils.logging import setup_logging
from utils.runtime import RuntimeContext, configure_runtime
from utils.seeds import set_global_seed
from utils.stats import aggregate_numeric_dicts
from utils.terminal import ProgressBar, format_metric, print_banner, print_section, print_summary
from utils.time import Timer

logger = logging.getLogger(__name__)
_LOADER_WORKER_OVERRIDE_LOGGED = False


# ── Configuration and setup ──────────────────────────────────────────

# Model registry. Each entry lazily imports its backend and wires the model
# constructor to the relevant config block, so unused backends never get imported.
_MODEL_CONSTRUCTORS = {
    "isolation_forest": lambda cfg: __import__(
        "models.baselines_classical",
        fromlist=["IsolationForestDetector"],
    ).IsolationForestDetector(
        n_estimators=int(cfg.get("classical", {}).get("isolation_forest_n_estimators", 300)),
        max_samples=cfg.get("classical", {}).get("isolation_forest_max_samples", 8192),
        seed=int(cfg.get("training", {}).get("seeds", [42])[0]),
    ),
    "ocsvm": lambda cfg: __import__(
        "models.baselines_classical",
        fromlist=["OneClassSVMDetector"],
    ).OneClassSVMDetector(),
    "elliptic_envelope": lambda cfg: __import__(
        "models.baselines_classical",
        fromlist=["EllipticEnvelopeDetector"],
    ).EllipticEnvelopeDetector(),
    "lstm_ae": lambda cfg: __import__(
        "models.baselines_deep",
        fromlist=["LSTMAutoencoder"],
    ).LSTMAutoencoder(
        n_channels=len(cfg["data"]["sensors"]),
        hidden_dim=cfg["model"]["hidden_dim"],
    ),
    "cnn_ae": lambda cfg: __import__(
        "models.baselines_deep",
        fromlist=["CNNAutoencoder"],
    ).CNNAutoencoder(
        n_channels=len(cfg["data"]["sensors"]),
        hidden_dim=cfg["model"]["hidden_dim"],
    ),
    "tranad": lambda cfg: __import__(
        "models.baselines_deep",
        fromlist=["TranADWrapper"],
    ).TranADWrapper(
        n_channels=len(cfg["data"]["sensors"]),
        hidden_dim=cfg["model"]["hidden_dim"],
        window_length=cfg["windowing"]["window_length"],
        output_min=float(cfg["data_processing"].get("normalisation_min", 0.0)),
        output_max=float(cfg["data_processing"].get("normalisation_max", 1.0)),
    ),
    "graph_stgnn": lambda cfg: __import__(
        "models.baselines_deep",
        fromlist=["SpatioTemporalGNN"],
    ).SpatioTemporalGNN(
        n_channels=len(cfg["data"]["sensors"]),
        window_length=cfg["windowing"]["window_length"],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_heads=cfg["model"].get("n_heads", 4),
        n_layers=cfg["model"].get("n_layers", 2),
        dropout=cfg["model"].get("dropout", 0.1),
        graph_conv=cfg["model"].get("graph_conv", "gcn"),
        temporal_mode=cfg["model"].get("temporal_mode", "transformer"),
    ),
}


def _is_cuda_capacity_error(exc: BaseException, runtime: RuntimeContext) -> bool:
    """Return True when a CUDA failure looks recoverable by shrinking batch size."""
    if runtime.device_type != "cuda":
        return False
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "invalid configuration argument" in message


def _release_runtime_memory(runtime: RuntimeContext) -> None:
    """Best-effort device cleanup between retry attempts."""
    gc.collect()
    if runtime.device_type == "cuda":
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


# ── Data preparation ─────────────────────────────────────────────────


def _ensure_processed_tensors(config_path: str, cfg: Dict[str, Any], *, rebuild_data: bool) -> None:
    """Ensure processed tensors exist, optionally rebuilding them from raw parquet."""
    if rebuild_data:
        # Forced rebuild path, regenerate the cached tensors from raw parquet first.
        print_section("Data preparation")
        from workflows.prepare_data import run as prepare_data_run

        prepare_data_run(config_path)
        return

    # Cache-reuse path, confirm the cached tensors are present, complete, and still
    # consistent with the current preprocessing contract before trusting them.
    try:
        validate_manifest_files(
            cfg["data"]["dataset_manifest"],
            require_raw=False,
            require_tensors=True,
        )
        sequence_counts_by_mission(cfg["data"]["dataset_manifest"])
        validate_cached_preprocessing_contract(cfg)
    except ValueError as exc:
        raise ValueError(
            "Processed tensors are missing, incomplete, or incompatible with the current preprocessing contract. "
            f"Run `python src/workflows/prepare_data.py --config {config_path}` first, "
            "or rerun this command with `--rebuild-data`.\n"
            f"Original error: {exc}"
        ) from exc


def _load_sequence_array(
    mission_id: str,
    manifest_map: Dict[str, Any],
    expected_window_length: int,
    expected_channels: int,
    index_counts: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Load and validate one mission tensor from the canonical sequence directory."""
    record = manifest_map[mission_id]
    array = np.load(record.tensor_path).astype(np.float32, copy=False)
    if array.ndim != 3:
        raise ValueError(
            f"Mission {mission_id} sequence tensor must be rank-3, got shape {array.shape}"
        )
    if array.shape[1] != expected_window_length:
        raise ValueError(
            f"Mission {mission_id} has window length {array.shape[1]}, "
            f"expected {expected_window_length}"
        )
    if array.shape[2] != expected_channels:
        raise ValueError(
            f"Mission {mission_id} has {array.shape[2]} channels, "
            f"expected {expected_channels}"
        )
    if index_counts is not None and mission_id in index_counts and len(array) != index_counts[mission_id]:
        raise ValueError(
            f"Mission {mission_id} tensor row count ({len(array)}) does not match "
            f"sequence_index.parquet ({index_counts[mission_id]})"
        )
    return array


def _chronological_dev_test_split(
    windows: np.ndarray,
    test_fraction: float = 0.30,
    val_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one mission into train, val, and test windows in chronological order."""
    n_windows = len(windows)
    if n_windows < 4:
        raise ValueError(
            f"Need at least 4 windows for a within-mission split, got {n_windows}"
        )

    n_test = max(1, int(round(n_windows * test_fraction)))
    n_test = min(n_test, n_windows - 2)
    dev = windows[: n_windows - n_test]
    test = windows[n_windows - n_test :]

    n_val = max(1, int(round(len(dev) * val_fraction)))
    n_val = min(n_val, len(dev) - 1)
    train = dev[: len(dev) - n_val]
    val = dev[len(dev) - n_val :]

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("Chronological split produced an empty train, val, or test block")

    return train, val, test


def _train_val_split(windows: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Chronologically split training missions into train and validation windows."""
    n_windows = len(windows)
    if n_windows < 2:
        raise ValueError(
            f"Need at least 2 windows to carve validation from a training mission, got {n_windows}"
        )

    n_val = max(1, int(round(n_windows * val_fraction)))
    n_val = min(n_val, n_windows - 1)
    train = windows[: n_windows - n_val]
    val = windows[n_windows - n_val :]
    return train, val


def _random_dev_test_split_indices(
    n_windows: int,
    *,
    seed: int,
    test_fraction: float = 0.30,
    val_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split pooled windows into train, validation, and test indices."""
    if n_windows < 4:
        raise ValueError(f"Need at least 4 windows for a pooled split, got {n_windows}")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_windows)

    n_test = max(1, int(round(n_windows * test_fraction)))
    n_test = min(n_test, n_windows - 2)
    dev_idx = permutation[: n_windows - n_test]
    test_idx = permutation[n_windows - n_test :]

    n_val = max(1, int(round(len(dev_idx) * val_fraction)))
    n_val = min(n_val, len(dev_idx) - 1)
    val_idx = dev_idx[:n_val]
    train_idx = dev_idx[n_val:]

    return train_idx, val_idx, test_idx


def _concat_windows(windows_list: List[np.ndarray], window_length: int, n_channels: int) -> np.ndarray:
    """Concatenate non-empty mission arrays while preserving the canonical shape."""
    arrays = [array for array in windows_list if len(array) > 0]
    if not arrays:
        return np.empty((0, window_length, n_channels), dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


def _prepare_split_data(
    cfg: Dict[str, Any],
    split: Dict[str, Any],
    manifest_map: Dict[str, Any],
    index_counts: Optional[Dict[str, int]] = None,
    split_seed: int = 42,
) -> Dict[str, Any]:
    """Resolve one split into train/val windows and per-mission test windows."""
    window_length = int(cfg["windowing"]["window_length"])
    n_channels = len(cfg["data"]["sensors"])
    val_fraction = float(cfg["splits"].get("val_fraction", 0.15))

    if split["mode"] == "global_random_split":
        # Global pooled split. Concatenate every mission's windows into one pool,
        # then split that pool randomly into train, val, and test. Mission ids and
        # per-mission local indices are tracked alongside so the test windows can be
        # regrouped back to their source mission afterwards.
        mission_ids = split.get("mission_ids", [])
        pooled_windows: List[np.ndarray] = []
        pooled_mission_ids: List[np.ndarray] = []
        pooled_local_indices: List[np.ndarray] = []

        for mission_id in mission_ids:
            windows = _load_sequence_array(
                mission_id,
                manifest_map,
                window_length,
                n_channels,
                index_counts=index_counts,
            )
            pooled_windows.append(windows)
            pooled_mission_ids.append(np.full(len(windows), mission_id, dtype=object))
            pooled_local_indices.append(np.arange(len(windows), dtype=np.int64))

        all_windows = _concat_windows(pooled_windows, window_length, n_channels)
        all_mission_ids = np.concatenate(pooled_mission_ids, axis=0)
        all_local_indices = np.concatenate(pooled_local_indices, axis=0)

        train_idx, val_idx, test_idx = _random_dev_test_split_indices(
            len(all_windows),
            seed=split_seed,
            test_fraction=0.30,
            val_fraction=val_fraction,
        )

        train_windows = all_windows[train_idx].astype(np.float32, copy=False)
        val_windows = all_windows[val_idx].astype(np.float32, copy=False)

        # Regroup the pooled test windows back per mission so faults can be injected
        # per mission. Restore chronological order within each mission using the
        # tracked local indices, so fault onset still falls at a sensible point.
        test_missions: Dict[str, np.ndarray] = {}
        test_mission_ids = all_mission_ids[test_idx]
        test_local_indices = all_local_indices[test_idx]
        test_windows = all_windows[test_idx]
        for mission_id in mission_ids:
            mask = test_mission_ids == mission_id
            if not np.any(mask):
                continue
            order = np.argsort(test_local_indices[mask], kind="stable")
            mission_test_windows = test_windows[mask][order].astype(np.float32, copy=False)
            if len(mission_test_windows) > 0:
                test_missions[mission_id] = mission_test_windows

        if len(train_windows) == 0 or len(val_windows) == 0 or not test_missions:
            raise ValueError(
                f"Split '{split['name']}' produced empty training, validation, or test data"
            )

        return {
            "name": split["name"],
            "train_windows": train_windows,
            "val_windows": val_windows,
            "test_missions": test_missions,
        }

    # Mission-held-out split. Train and validate on the listed training missions,
    # then test on entirely separate held-out missions. When validation missions are
    # named explicitly use them as is, otherwise carve a validation tail from each
    # training mission chronologically.
    train_arrays: List[np.ndarray] = []
    val_arrays: List[np.ndarray] = []
    explicit_val_ids = split.get("val_ids", [])

    for mission_id in split["train_ids"]:
        windows = _load_sequence_array(
            mission_id,
            manifest_map,
            window_length,
            n_channels,
            index_counts=index_counts,
        )
        if explicit_val_ids:
            train_arrays.append(windows)
        else:
            train_part, val_part = _train_val_split(windows, val_fraction)
            train_arrays.append(train_part)
            val_arrays.append(val_part)

    for mission_id in explicit_val_ids:
        windows = _load_sequence_array(
            mission_id,
            manifest_map,
            window_length,
            n_channels,
            index_counts=index_counts,
        )
        val_arrays.append(windows)

    test_missions = {
        mission_id: _load_sequence_array(
            mission_id,
            manifest_map,
            window_length,
            n_channels,
            index_counts=index_counts,
        )
        for mission_id in split["test_ids"]
    }

    train_windows = _concat_windows(train_arrays, window_length, n_channels)
    val_windows = _concat_windows(val_arrays, window_length, n_channels)
    if len(train_windows) == 0 or len(val_windows) == 0 or not test_missions:
        raise ValueError(
            f"Split '{split['name']}' produced empty training, validation, or test data"
        )

    return {
        "name": split["name"],
        "train_windows": train_windows,
        "val_windows": val_windows,
        "test_missions": test_missions,
    }


def _make_loader(
    windows: np.ndarray,
    batch_size: int,
    runtime: RuntimeContext,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap an in-memory window array in a single-process DataLoader.

    Benchmark tensors are already materialised in RAM, so worker processes only add
    overhead. Loader workers are forced to zero regardless of the runtime setting, the
    override is logged once.
    """
    global _LOADER_WORKER_OVERRIDE_LOGGED
    tensor = torch.from_numpy(windows).float()
    # Benchmark windows are already materialized in RAM, so worker processes
    # only add IPC/file-descriptor overhead. Single-process loaders are more
    # robust for long sequential benchmark runs on these cached tensors.
    loader_workers = 0
    if runtime.dataloader_workers > 0 and not _LOADER_WORKER_OVERRIDE_LOGGED:
        logger.info(
            "Benchmark tensors are loaded eagerly; overriding runtime.dataloader_workers=%d "
            "with single-process loaders to avoid worker churn and fd exhaustion.",
            runtime.dataloader_workers,
        )
        _LOADER_WORKER_OVERRIDE_LOGGED = True
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": loader_workers,
        "pin_memory": runtime.pin_memory,
    }
    if loader_workers > 0:
        kwargs["persistent_workers"] = runtime.persistent_workers
        if runtime.prefetch_factor is not None:
            kwargs["prefetch_factor"] = runtime.prefetch_factor
    return DataLoader(
        TensorDataset(tensor),
        **kwargs,
    )


# ── Model construction ───────────────────────────────────────────────


def _build_graph(cfg: Dict[str, Any], train_windows: np.ndarray) -> np.ndarray:
    """Build the graph adjacency used by graph-based models."""
    builder_cfg = cfg.get("graph", {})
    builder_type = builder_cfg.get("builder", "correlation")
    train_series = train_windows.reshape(-1, train_windows.shape[-1])

    if builder_type == "fixed":
        builder = FixedGraphBuilder(
            n_sensors=train_windows.shape[-1],
            adjacency_list=builder_cfg.get("adjacency_list", "domain_knowledge"),
        )
    elif builder_type == "correlation":
        builder = CorrelationGraphBuilder(
            threshold=builder_cfg.get("correlation_threshold", 0.3),
            top_k=builder_cfg.get("correlation_top_k", None),
        )
    elif builder_type == "learned":
        builder = LearnedGraphBuilder(
            n_sensors=train_windows.shape[-1],
            init_data=train_series,
            threshold=builder_cfg.get("correlation_threshold", 0.3),
        )
    else:
        raise ValueError(f"Unsupported graph builder '{builder_type}'")

    _, adjacency_norm, _ = builder.build(train_data=train_series)
    return adjacency_norm


def _checkpoint_path_for_split(
    checkpoint_dir: Optional[pathlib.Path],
    split_name: str,
) -> Optional[pathlib.Path]:
    """Return the canonical checkpoint path for one split, if enabled."""
    if checkpoint_dir is None:
        return None
    return checkpoint_dir / f"{split_name}.ckpt"


def _move_loaded_model_to_runtime(model: Any, runtime: RuntimeContext) -> None:
    """Best-effort device placement after loading a checkpoint."""
    module = getattr(model, "_module", None)
    if module is not None and hasattr(module, "to"):
        module.to(runtime.device)
    adjacency = getattr(model, "_A_norm", None)
    if isinstance(adjacency, torch.Tensor):
        model._A_norm = adjacency.to(runtime.device)


# ── Fault injection ──────────────────────────────────────────────────


def _inject_fault_for_mission(
    mission_windows: np.ndarray,
    cfg: Dict[str, Any],
    fault_type: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Inject a fault into one held-out mission and return window labels.

    The held-out test windows are healthy. A synthetic fault of the requested type is
    injected starting at the configured onset fraction, the returned boolean labels
    mark which windows are faulty so detection can be scored against ground truth.
    """
    # Assemble the per-fault keyword arguments. ``wing_loss`` needs a magnitude and the
    # normalisation range so the perturbation stays within the valid signal bounds.
    fault_kwargs: Dict[str, Any] = {
        "onset": cfg["faults"].get("onset_fraction", 0.5),
        "seed": seed,
        "channel_names": list(cfg["data"]["sensors"]),
    }
    if fault_type == "wing_loss":
        fault_kwargs["magnitude"] = cfg["faults"].get("magnitude", 2.0)
        fault_kwargs["range_min"] = cfg["data_processing"].get("normalisation_min", 0.0)
        fault_kwargs["range_max"] = cfg["data_processing"].get("normalisation_max", 1.0)

    faulty_windows, window_labels, metadata = inject_fault(
        mission_windows,
        fault_type,
        **fault_kwargs,
    )
    return (
        faulty_windows.astype(np.float32, copy=False),
        window_labels.astype(bool, copy=False),
        metadata,
    )


# ── Scoring and metrics ──────────────────────────────────────────────

# Numeric metric keys averaged when rolling mission, fault, split, or seed results up.
_NUMERIC_METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "detection_latency",
    "fa_count",
    "fa_rate_mission",
    "fa_rate_per_hour",
]


def _aggregate_numeric_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """Average numeric metrics across mission-level or split-level results."""
    aggregated: Dict[str, float] = {}
    for key in _NUMERIC_METRIC_KEYS:
        values = []
        for metric in metrics:
            value = metric.get(key)
            if isinstance(value, (int, float)) and not np.isnan(value):
                values.append(float(value))
        if values:
            aggregated[key] = float(np.mean(values))
    return aggregated


def _evaluate_fault_with_model(
    cfg: Dict[str, Any],
    split_data: Dict[str, Any],
    model: Any,
    *,
    batch_size: int,
    fault_type: str,
    threshold: float,
    runtime: RuntimeContext,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate one trained model against one injected fault type."""
    mission_metrics: List[Dict[str, Any]] = []
    # Each held-out test mission is faulted, scored, and metered independently. The per
    # mission seed offset keeps the injected faults distinct across missions.
    for mission_offset, (mission_id, mission_windows) in enumerate(split_data["test_missions"].items()):
        faulty_windows, labels, fault_metadata = _inject_fault_for_mission(
            mission_windows=mission_windows,
            cfg=cfg,
            fault_type=fault_type,
            seed=seed + mission_offset,
        )
        test_loader = _make_loader(
            faulty_windows,
            batch_size=batch_size,
            runtime=runtime,
        )
        # Apply the validation-calibrated threshold to the continuous anomaly scores to
        # obtain binary predictions, then meter them against the injected ground truth.
        result = model.score(test_loader, cfg)
        preds = apply_threshold(result.scores_time_global, threshold)
        metrics = compute_metrics(
            labels,
            preds,
            scores=result.scores_time_global,
            sampling_interval_s=cfg["data_processing"].get("interpolation_interval_s", 5),
            latency_unit=cfg["evaluation"].get("latency_unit", "timesteps"),
        )
        metrics["mission_id"] = mission_id
        metrics.update(fault_metadata)
        mission_metrics.append(metrics)

    fault_metrics = _aggregate_numeric_metrics(mission_metrics)
    fault_metrics.update(
        {
            "fault": fault_type,
            "mission_metrics": mission_metrics,
            "threshold_method": cfg["thresholding"]["method"],
            "threshold_value": float(threshold),
            "batch_size": batch_size,
        }
    )
    return fault_metrics


# ── Training ─────────────────────────────────────────────────────────


def _evaluate_split(
    cfg: Dict[str, Any],
    split_data: Dict[str, Any],
    model_name: str,
    seed: int,
    fault_types: List[str],
    runtime: RuntimeContext,
    checkpoint_dir: Optional[pathlib.Path] = None,
    eval_only: bool = False,
) -> Dict[str, Any]:
    """Train once on a split, or reuse a saved checkpoint, then evaluate."""
    requested_batch_size = int(cfg["training"]["batch_size"])
    batch_size = max(1, min(requested_batch_size, len(split_data["train_windows"])))
    checkpoint_path = _checkpoint_path_for_split(checkpoint_dir, split_data["name"])

    # Retry loop. On a recoverable CUDA capacity error the batch size is halved and the
    # whole split is retried, hence the loop wrapping construction, training, and eval.
    while True:
        train_loader = val_loader = test_loader = model = val_scores = result = None
        try:
            val_loader = _make_loader(
                split_data["val_windows"],
                batch_size=batch_size,
                runtime=runtime,
            )

            cfg["_n_channels"] = len(cfg["data"]["sensors"])
            model = _MODEL_CONSTRUCTORS[model_name](cfg)
            if eval_only:
                # Eval-only path. Skip training entirely and restore the model from the
                # split checkpoint, then place it on the active device. The checkpoint
                # must already exist, this path never writes one back.
                if checkpoint_path is None:
                    raise ValueError(
                        f"eval_only requires a checkpoint directory for split '{split_data['name']}'"
                    )
                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Missing checkpoint for eval-only run: {checkpoint_path}"
                    )
                with Timer(f"Loading checkpoint for {model_name} on {split_data['name']}"):
                    model.load(str(checkpoint_path))
                _move_loaded_model_to_runtime(model, runtime)
            else:
                # Training path. Build the loader, attach the graph adjacency for graph
                # models, then fit on the training windows with the validation loader.
                train_loader = _make_loader(
                    split_data["train_windows"],
                    batch_size=batch_size,
                    runtime=runtime,
                    shuffle=True,
                )
                adjacency = _build_graph(cfg, split_data["train_windows"]) if model_name.startswith("graph") else None
                if adjacency is not None:
                    model.set_adjacency(adjacency)
                with Timer(f"Training {model_name} on {split_data['name']}"):
                    model.fit(train_loader, val_loader, cfg)

            # Threshold calibration. Score the healthy validation windows and pick the
            # detection threshold from those clean scores alone, so calibration never
            # sees a fault. The same threshold is then reused for every fault type.
            val_scores = model.score(val_loader, cfg)
            threshold = select_threshold(val_scores.scores_time_global, cfg)

            # Evaluate each configured fault type against the one trained model. The
            # widely spaced per-fault seed offset keeps injected faults independent.
            fault_metrics: Dict[str, Dict[str, Any]] = {}
            for fault_index, fault_type in enumerate(fault_types):
                fault_seed = seed + (fault_index * 10_000)
                fault_metrics[fault_type] = _evaluate_fault_with_model(
                    cfg=cfg,
                    split_data=split_data,
                    model=model,
                    batch_size=batch_size,
                    fault_type=fault_type,
                    threshold=threshold,
                    runtime=runtime,
                    seed=fault_seed,
                )

            # Checkpoint save. Only freshly trained models are persisted. Write to a
            # temporary file then atomically rename, so a crash mid-write cannot leave a
            # corrupt checkpoint behind for a later eval-only run to pick up.
            if not eval_only and checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                tmp_checkpoint_path = checkpoint_dir / f".{split_data['name']}.ckpt.tmp"
                model.save(str(tmp_checkpoint_path))
                tmp_checkpoint_path.replace(checkpoint_path)

            split_metrics = _aggregate_numeric_metrics(list(fault_metrics.values()))
            split_metrics.update(
                {
                    "split_name": split_data["name"],
                    "faults": list(fault_types),
                    "fault_metrics": fault_metrics,
                    "threshold_method": cfg["thresholding"]["method"],
                    "threshold_value": float(threshold),
                    "batch_size": batch_size,
                }
            )
            if len(fault_types) == 1:
                only_fault = fault_types[0]
                split_metrics["fault"] = only_fault
                split_metrics["mission_metrics"] = fault_metrics[only_fault]["mission_metrics"]
            if checkpoint_path is not None:
                split_metrics["checkpoint_path"] = str(checkpoint_path)
            if batch_size != requested_batch_size:
                split_metrics["requested_batch_size"] = requested_batch_size
            return split_metrics
        except Exception as exc:
            # Only retry recoverable CUDA capacity failures, and only while there is
            # still room to shrink. Anything else propagates unchanged.
            if not _is_cuda_capacity_error(exc, runtime) or batch_size <= 1:
                raise
            next_batch_size = max(1, batch_size // 2)
            logger.warning(
                "CUDA batch size %d failed for %s on %s (%s). Retrying with %d.",
                batch_size,
                model_name,
                split_data["name"],
                exc.__class__.__name__,
                next_batch_size,
            )
            batch_size = next_batch_size
        finally:
            del train_loader, val_loader, test_loader, model, val_scores, result
            _release_runtime_memory(runtime)


# ── Aggregation ──────────────────────────────────────────────────────


def _configured_training_seeds(cfg: Dict[str, Any]) -> List[int]:
    """Return validated training seeds as integers."""
    configured_seeds = cfg.get("training", {}).get("seeds", [])
    if not configured_seeds:
        raise ValueError("training.seeds must contain at least one seed")
    try:
        seeds = [int(seed) for seed in configured_seeds]
    except (TypeError, ValueError) as exc:
        raise ValueError("training.seeds must contain only integers") from exc
    if not seeds:
        raise ValueError("training.seeds must contain at least one seed")
    if any(seed < 0 for seed in seeds):
        raise ValueError("training.seeds must contain only integers >= 0")
    return seeds


def _run_one_seed(
    *,
    base_cfg: Dict[str, Any],
    model_name: str,
    seed: int,
    selected_faults: List[str],
    manifest_map: Dict[str, Any],
    index_counts: Dict[str, int],
    protocol_splits: List[Dict[str, Any]],
    output_base_dir: pathlib.Path,
    eval_only: bool,
) -> tuple[Dict[str, Any], pathlib.Path, pathlib.Path]:
    """Execute one seed-specific training/evaluation pass."""
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["name"] = model_name
    cfg["training"]["seeds"] = [seed]

    set_global_seed(seed)
    runtime = configure_runtime(cfg)
    run_id = f"{model_name}_seed{seed}"
    out_dir = output_base_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "models"
    save_yaml(cfg, out_dir / "config.yaml")

    split_metrics: List[Dict[str, Any]] = []
    if len(protocol_splits) == 1:
        print_section(f"Seed {seed}: training and evaluation")
        split = protocol_splits[0]
        split_data = _prepare_split_data(
            cfg,
            split,
            manifest_map,
            index_counts=index_counts,
            split_seed=seed,
        )
        metrics = _evaluate_split(
            cfg=cfg,
            split_data=split_data,
            model_name=model_name,
            seed=seed,
            fault_types=selected_faults,
            runtime=runtime,
            checkpoint_dir=checkpoint_dir,
            eval_only=eval_only,
        )
        split_metrics.append(metrics)
        if len(selected_faults) == 1:
            print(
                "  "
                f"Acc={format_metric(metrics.get('accuracy'))} "
                f"Prec={format_metric(metrics.get('precision'))} "
                f"Rec={format_metric(metrics.get('recall'))} "
                f"F1={format_metric(metrics.get('f1'))}"
            )
        else:
            print(
                "  Overall "
                f"Acc={format_metric(metrics.get('accuracy'))} "
                f"Prec={format_metric(metrics.get('precision'))} "
                f"Rec={format_metric(metrics.get('recall'))} "
                f"F1={format_metric(metrics.get('f1'))}"
            )
            for fault_name in selected_faults:
                fault_metrics = metrics["fault_metrics"][fault_name]
                print(
                    "  "
                    f"{fault_name:<16} "
                    f"Acc={format_metric(fault_metrics.get('accuracy'))} "
                    f"Prec={format_metric(fault_metrics.get('precision'))} "
                    f"Rec={format_metric(fault_metrics.get('recall'))} "
                    f"F1={format_metric(fault_metrics.get('f1'))}"
                )
    else:
        print_section(f"Seed {seed}: split evaluation")
        split_progress = ProgressBar(total=len(protocol_splits), desc="splits", unit="split", leave=True)
        for split in protocol_splits:
            split_progress.set_postfix_str(split["name"])
            split_data = _prepare_split_data(
                cfg,
                split,
                manifest_map,
                index_counts=index_counts,
                split_seed=seed,
            )
            metrics = _evaluate_split(
                cfg=cfg,
                split_data=split_data,
                model_name=model_name,
                seed=seed,
                fault_types=selected_faults,
                runtime=runtime,
                checkpoint_dir=checkpoint_dir,
                eval_only=eval_only,
            )
            split_metrics.append(metrics)
            split_progress.write(
                "  "
                f"{split['name']}: "
                f"Acc={format_metric(metrics.get('accuracy'))} "
                f"Prec={format_metric(metrics.get('precision'))} "
                f"Rec={format_metric(metrics.get('recall'))} "
                f"F1={format_metric(metrics.get('f1'))}"
            )
            split_progress.update(1)
        split_progress.close()

    fault_metrics_flat = [
        fault_metrics
        for split_metric in split_metrics
        for fault_metrics in split_metric.get("fault_metrics", {}).values()
    ]
    aggregated = _aggregate_numeric_metrics(fault_metrics_flat)
    per_fault: Dict[str, Dict[str, Any]] = {}
    for fault_name in selected_faults:
        fault_split_metrics = [
            split_metric["fault_metrics"][fault_name]
            for split_metric in split_metrics
            if fault_name in split_metric.get("fault_metrics", {})
        ]
        fault_aggregate = _aggregate_numeric_metrics(fault_split_metrics)
        fault_aggregate.update(
            {
                "fault": fault_name,
                "split_count": len(fault_split_metrics),
            }
        )
        per_fault[fault_name] = fault_aggregate
    aggregated.update(
        {
            "model": model_name,
            "seed": seed,
            "faults": selected_faults,
            "fault_count": len(selected_faults),
            "split_count": len(split_metrics),
            "per_fault": per_fault,
            "splits": split_metrics,
        }
    )
    if len(selected_faults) == 1:
        aggregated["fault"] = selected_faults[0]

    summary_rows: List[Dict[str, Any]] = []
    summary_rows.append(
        {
            "model": model_name,
            "seed": seed,
            "fault": selected_faults[0] if len(selected_faults) == 1 else "all",
            "accuracy": aggregated.get("accuracy"),
            "precision": aggregated.get("precision"),
            "recall": aggregated.get("recall"),
            "f1": aggregated.get("f1"),
            "roc_auc": aggregated.get("roc_auc"),
            "pr_auc": aggregated.get("pr_auc"),
        }
    )
    if len(selected_faults) > 1:
        for fault_name in selected_faults:
            fault_summary = per_fault.get(fault_name, {})
            summary_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "fault": fault_name,
                    "accuracy": fault_summary.get("accuracy"),
                    "precision": fault_summary.get("precision"),
                    "recall": fault_summary.get("recall"),
                    "f1": fault_summary.get("f1"),
                    "roc_auc": fault_summary.get("roc_auc"),
                    "pr_auc": fault_summary.get("pr_auc"),
                }
            )

    save_json(aggregated, out_dir / "metrics.json")
    save_results(
        summary_rows,
        out_dir / "summary",
        save_csv=cfg["output"].get("save_csv", True),
        save_latex=cfg["output"].get("save_latex", True),
        save_md=cfg["output"].get("save_md", True),
    )
    logger.info(
        "Run %s → Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f over %d splits and %d fault evaluations",
        run_id,
        aggregated.get("accuracy", float("nan")),
        aggregated.get("precision", float("nan")),
        aggregated.get("recall", float("nan")),
        aggregated.get("f1", float("nan")),
        len(split_metrics),
        len(selected_faults),
    )
    return aggregated, out_dir, checkpoint_dir


def _aggregate_seed_runs(
    *,
    model_name: str,
    seeds: List[int],
    selected_faults: List[str],
    per_seed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate a model run across multiple configured seeds."""
    # Multi-seed aggregation. Combine the per-seed metric dicts into means with 95%
    # confidence interval half-widths across seeds, both overall and per fault type.
    aggregated = aggregate_numeric_dicts(per_seed_results, _NUMERIC_METRIC_KEYS)
    aggregated.update(
        {
            "model": model_name,
            "seeds": list(seeds),
            "seed_count": len(per_seed_results),
            "faults": list(selected_faults),
            "fault_count": len(selected_faults),
            "seed_results": per_seed_results,
        }
    )

    per_fault: Dict[str, Dict[str, Any]] = {}
    for fault_name in selected_faults:
        fault_seed_metrics = [
            seed_metrics["per_fault"][fault_name]
            for seed_metrics in per_seed_results
            if fault_name in seed_metrics.get("per_fault", {})
        ]
        fault_aggregate = aggregate_numeric_dicts(fault_seed_metrics, _NUMERIC_METRIC_KEYS)
        fault_aggregate.update(
            {
                "fault": fault_name,
                "seed_count": len(fault_seed_metrics),
            }
        )
        per_fault[fault_name] = fault_aggregate

    aggregated["per_fault"] = per_fault
    if len(selected_faults) == 1:
        aggregated["fault"] = selected_faults[0]
    return aggregated


# ── The run() entrypoint ─────────────────────────────────────────────


def run(
    config_path: str,
    model_override: Optional[str] = None,
    fault_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    rebuild_data: bool = False,
    eval_only: bool = False,
) -> None:
    """Run the full benchmark for one model over every configured seed.

    Load and validate the config, ensure processed tensors are available, then run each
    seed through training (or checkpoint reuse), fault injection, scoring, and metrics.
    A single seed writes a per-run summary, multiple seeds are aggregated with across
    seed confidence intervals into a multiseed output directory.

    Args:
        config_path: Path to the experiment YAML config.
        model_override: Optional model name overriding ``model.name`` from the config.
        fault_override: Optional single fault type, otherwise all configured faults run.
        output_dir_override: Optional output base directory overriding the config.
        rebuild_data: Rebuild processed tensors from raw parquet before running.
        eval_only: Skip training and evaluate existing split checkpoints only.
    """
    cfg = load_config(config_path)
    validate_config(cfg)
    _ensure_processed_tensors(config_path, cfg, rebuild_data=rebuild_data)

    model_name = model_override or cfg["model"]["name"]
    cfg["model"]["name"] = model_name
    seeds = _configured_training_seeds(cfg)
    # Resolve which faults to evaluate. A command-line override wins, otherwise use the
    # configured list, falling back to ``wing_loss`` when nothing is configured.
    configured_faults = list(cfg["faults"]["types"]) if cfg["faults"].get("types") else []
    selected_faults = [fault_override] if fault_override else configured_faults
    if not selected_faults:
        selected_faults = ["wing_loss"]

    manifest_map = manifest_by_mission(cfg["data"]["dataset_manifest"])
    index_counts = sequence_counts_by_mission(cfg["data"]["dataset_manifest"])
    protocol_splits = build_protocol_splits(cfg)
    output_base_dir = pathlib.Path(output_dir_override or cfg["output"]["base_dir"])
    runtime = configure_runtime(cfg)
    loader_workers = 0

    print_banner(
        "AUV Anomaly Detection - Run Model",
        [
            ("Config", config_path),
            ("Model", model_name),
            ("Mode", "evaluate existing checkpoint" if eval_only else "train + evaluate"),
            ("Fault" if len(selected_faults) == 1 else "Faults", ", ".join(selected_faults)),
            ("Splits", len(protocol_splits)),
            ("Seed" if len(seeds) == 1 else "Seeds", seeds[0] if len(seeds) == 1 else ", ".join(str(seed) for seed in seeds)),
            ("Device", runtime.device_label),
            ("CPU threads", runtime.cpu_threads),
            ("Loader workers", loader_workers),
            ("Data", "rebuild from raw parquet" if rebuild_data else "use cached processed tensors"),
            ("Output", output_base_dir),
        ],
    )

    per_seed_results: List[Dict[str, Any]] = []
    last_seed_dir: Optional[pathlib.Path] = None
    last_checkpoint_dir: Optional[pathlib.Path] = None

    # Run every configured seed in turn, collecting each seed's aggregated metrics.
    for seed in seeds:
        seed_metrics, seed_dir, checkpoint_dir = _run_one_seed(
            base_cfg=cfg,
            model_name=model_name,
            seed=seed,
            selected_faults=selected_faults,
            manifest_map=manifest_map,
            index_counts=index_counts,
            protocol_splits=protocol_splits,
            output_base_dir=output_base_dir,
            eval_only=eval_only,
        )
        per_seed_results.append(seed_metrics)
        last_seed_dir = seed_dir
        last_checkpoint_dir = checkpoint_dir

    # Single-seed path. Report that one seed's metrics directly, no across-seed
    # aggregation is meaningful so the per-run outputs are the final result.
    if len(seeds) == 1:
        single_result = per_seed_results[0]
        assert last_seed_dir is not None
        assert last_checkpoint_dir is not None
        summary_lines = [
            ("Accuracy", format_metric(single_result.get("accuracy"))),
            ("Precision", format_metric(single_result.get("precision"))),
            ("Recall", format_metric(single_result.get("recall"))),
            ("F1", format_metric(single_result.get("f1"))),
            ("ROC-AUC", format_metric(single_result.get("roc_auc"))),
            ("PR-AUC", format_metric(single_result.get("pr_auc"))),
        ]
        if len(selected_faults) > 1:
            for fault_name in selected_faults:
                summary_lines.append(
                    (
                        f"{fault_name} F1",
                        format_metric(single_result.get("per_fault", {}).get(fault_name, {}).get("f1")),
                    )
                )
        summary_lines.extend(
            [
                ("Metrics", last_seed_dir / "metrics.json"),
                ("Models", last_checkpoint_dir),
                ("Summary", last_seed_dir / "summary"),
            ]
        )
        print_summary("Model run complete", summary_lines)
        return

    # Multi-seed path. Aggregate every seed's metrics into means and confidence
    # intervals, then persist the combined config, metrics, and summary tables.
    aggregate_dir = output_base_dir / f"{model_name}_multiseed"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_metrics = _aggregate_seed_runs(
        model_name=model_name,
        seeds=seeds,
        selected_faults=selected_faults,
        per_seed_results=per_seed_results,
    )
    save_yaml(cfg, aggregate_dir / "config.yaml")
    save_json(aggregate_metrics, aggregate_dir / "metrics.json")

    aggregate_summary_rows: List[Dict[str, Any]] = [
        {
            "model": model_name,
            "seed_count": len(seeds),
            "fault": selected_faults[0] if len(selected_faults) == 1 else "all",
            "accuracy_mean": aggregate_metrics.get("accuracy_mean"),
            "accuracy_ci95": aggregate_metrics.get("accuracy_ci95_halfwidth"),
            "precision_mean": aggregate_metrics.get("precision_mean"),
            "precision_ci95": aggregate_metrics.get("precision_ci95_halfwidth"),
            "recall_mean": aggregate_metrics.get("recall_mean"),
            "recall_ci95": aggregate_metrics.get("recall_ci95_halfwidth"),
            "f1_mean": aggregate_metrics.get("f1_mean"),
            "f1_ci95": aggregate_metrics.get("f1_ci95_halfwidth"),
            "roc_auc_mean": aggregate_metrics.get("roc_auc_mean"),
            "roc_auc_ci95": aggregate_metrics.get("roc_auc_ci95_halfwidth"),
            "pr_auc_mean": aggregate_metrics.get("pr_auc_mean"),
            "pr_auc_ci95": aggregate_metrics.get("pr_auc_ci95_halfwidth"),
        }
    ]
    if len(selected_faults) > 1:
        for fault_name in selected_faults:
            fault_summary = aggregate_metrics.get("per_fault", {}).get(fault_name, {})
            aggregate_summary_rows.append(
                {
                    "model": model_name,
                    "seed_count": len(seeds),
                    "fault": fault_name,
                    "accuracy_mean": fault_summary.get("accuracy_mean"),
                    "accuracy_ci95": fault_summary.get("accuracy_ci95_halfwidth"),
                    "precision_mean": fault_summary.get("precision_mean"),
                    "precision_ci95": fault_summary.get("precision_ci95_halfwidth"),
                    "recall_mean": fault_summary.get("recall_mean"),
                    "recall_ci95": fault_summary.get("recall_ci95_halfwidth"),
                    "f1_mean": fault_summary.get("f1_mean"),
                    "f1_ci95": fault_summary.get("f1_ci95_halfwidth"),
                    "roc_auc_mean": fault_summary.get("roc_auc_mean"),
                    "roc_auc_ci95": fault_summary.get("roc_auc_ci95_halfwidth"),
                    "pr_auc_mean": fault_summary.get("pr_auc_mean"),
                    "pr_auc_ci95": fault_summary.get("pr_auc_ci95_halfwidth"),
                }
            )

    save_results(
        aggregate_summary_rows,
        aggregate_dir / "summary",
        save_csv=cfg["output"].get("save_csv", True),
        save_latex=cfg["output"].get("save_latex", True),
        save_md=cfg["output"].get("save_md", True),
    )

    summary_lines = [
        ("Seeds", ", ".join(str(seed) for seed in seeds)),
        (
            "Accuracy",
            f"{format_metric(aggregate_metrics.get('accuracy_mean'))} ± {format_metric(aggregate_metrics.get('accuracy_ci95_halfwidth'))}",
        ),
        (
            "Precision",
            f"{format_metric(aggregate_metrics.get('precision_mean'))} ± {format_metric(aggregate_metrics.get('precision_ci95_halfwidth'))}",
        ),
        (
            "Recall",
            f"{format_metric(aggregate_metrics.get('recall_mean'))} ± {format_metric(aggregate_metrics.get('recall_ci95_halfwidth'))}",
        ),
        (
            "F1",
            f"{format_metric(aggregate_metrics.get('f1_mean'))} ± {format_metric(aggregate_metrics.get('f1_ci95_halfwidth'))}",
        ),
        (
            "ROC-AUC",
            f"{format_metric(aggregate_metrics.get('roc_auc_mean'))} ± {format_metric(aggregate_metrics.get('roc_auc_ci95_halfwidth'))}",
        ),
        (
            "PR-AUC",
            f"{format_metric(aggregate_metrics.get('pr_auc_mean'))} ± {format_metric(aggregate_metrics.get('pr_auc_ci95_halfwidth'))}",
        ),
    ]
    if len(selected_faults) > 1:
        for fault_name in selected_faults:
            fault_summary = aggregate_metrics.get("per_fault", {}).get(fault_name, {})
            summary_lines.append(
                (
                    f"{fault_name} F1",
                    f"{format_metric(fault_summary.get('f1_mean'))} ± {format_metric(fault_summary.get('f1_ci95_halfwidth'))}",
                )
            )
    summary_lines.extend(
        [
            ("Metrics", aggregate_dir / "metrics.json"),
            ("Seed runs", output_base_dir),
            ("Summary", aggregate_dir / "summary"),
        ]
    )
    print_summary("Multi-seed model run complete", summary_lines)


# ── Argparse and main ────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments for a single benchmark run."""
    parser = argparse.ArgumentParser(description="Run one benchmark model.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override (e.g. graph_stgnn, tranad).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override for this run.",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild processed tensors from raw parquet before running.",
    )
    parser.add_argument(
        "--fault",
        default=None,
        help="Optional single-fault override. Omit to evaluate all configured injected faults from the same trained model.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, load the existing split checkpoint(s), and rerun evaluation only.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point, parse arguments, configure logging, and run."""
    args = parse_args()
    setup_logging()
    run(
        args.config,
        model_override=args.model,
        fault_override=args.fault,
        output_dir_override=args.output_dir,
        rebuild_data=args.rebuild_data,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
