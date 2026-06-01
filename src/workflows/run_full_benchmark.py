#!/usr/bin/env python3
"""Full benchmark runner. Orchestrates all experiments and aggregates results.

Usage:
    python src/workflows/run_full_benchmark.py --config src/config/defaults.yaml
    python src/workflows/run_full_benchmark.py --config src/config/defaults.yaml --models graph_stgnn tranad
    python src/workflows/run_full_benchmark.py --config src/config/defaults.yaml --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pathlib
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Ensure src is on the path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config import load_config, validate_config
from data import validate_cached_preprocessing_contract
from data.manifest import sequence_counts_by_mission, validate_manifest_files
from utils.io import save_json
from utils.logging import setup_logging
from utils.runtime import configure_runtime
from utils.stats import aggregate_numeric_dicts
from workflows.run_experiment import run as run_experiment

logger = logging.getLogger(__name__)


# ── CLI Progress Tracker ─────────────────────────────────────────────

class ProgressTracker:
    """Live CLI progress with tqdm bars and a status summary panel.

    Displays:
    - A tqdm progress bar per phase (main, ablation)
    - Running totals: completed / failed / skipped
    - Best metrics seen so far per model
    - Estimated time remaining
    """

    def __init__(self, total: int, phase: str = "benchmark", disable: bool = False):
        """Set up the tracker, building a tqdm bar unless disabled or absent."""
        self.total = total
        self.phase = phase
        self.completed = 0
        self.failed = 0
        self.best_f1: Dict[str, float] = {}
        self.start_time = time.time()
        self.disable = disable or not HAS_TQDM
        self._bar: Optional[tqdm] = None

        if not self.disable:
            cols = shutil.get_terminal_size((120, 24)).columns
            self._bar = tqdm(
                total=total,
                desc=f"  {phase}",
                unit="run",
                ncols=min(cols, 140),
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
                leave=True,
            )

    def update(self, model: str, fault: str, seed: int,
               result: Dict[str, Any]) -> None:
        """Record one completed run and refresh the display."""
        self.completed += 1
        success = "error" not in result

        if success:
            accuracy = result.get("accuracy", 0.0)
            f1 = result.get("f1", 0.0)
            display = MODEL_DISPLAY.get(model, model)
            prev_best = self.best_f1.get(model, 0.0)
            if f1 > prev_best:
                self.best_f1[model] = f1
        else:
            self.failed += 1
            accuracy, f1 = 0.0, 0.0

        if self._bar is not None:
            # Build a compact postfix string
            postfix_parts = []
            if success:
                postfix_parts.append(f"Acc={accuracy:.3f}")
                postfix_parts.append(f"F1={f1:.3f}")
            else:
                postfix_parts.append("FAILED")
            postfix_parts.append(f"fail={self.failed}")

            # Show best F1 per model seen so far
            top = sorted(self.best_f1.items(), key=lambda x: x[1], reverse=True)[:3]
            if top:
                top_str = " ".join(
                    f"{MODEL_DISPLAY.get(m, m)}={v:.3f}" for m, v in top
                )
                postfix_parts.append(f"best: {top_str}")

            self._bar.set_postfix_str(" | ".join(postfix_parts))
            self._bar.update(1)
        else:
            # Fallback plain logging
            elapsed = time.time() - self.start_time
            rate = elapsed / self.completed if self.completed > 0 else 0
            remaining = rate * (self.total - self.completed)
            status = f"Acc={accuracy:.4f} F1={f1:.4f}" if success else "FAILED"
            logger.info(
                "[%d/%d] %s/%s/seed=%d  %s  (ETA: %s)",
                self.completed, self.total, model, fault, seed,
                status, _format_eta(remaining),
            )

    def close(self) -> None:
        """Close the underlying progress bar, if any."""
        if self._bar is not None:
            self._bar.close()

    def summary(self) -> str:
        """Return a text summary of this phase."""
        elapsed = time.time() - self.start_time
        lines = [
            f"  {self.phase}: {self.completed} runs in {_format_eta(elapsed)}"
            f" ({self.failed} failed)",
        ]
        if self.best_f1:
            top = sorted(self.best_f1.items(), key=lambda x: x[1], reverse=True)
            for model, f1 in top:
                display = MODEL_DISPLAY.get(model, model)
                lines.append(f"    {display:>10s}  best F1 = {f1:.4f}")
        return "\n".join(lines)

# ── Model and fault roster ───────────────────────────────────────────

ALL_MODELS = [
    "isolation_forest",
    "ocsvm",
    "elliptic_envelope",
    "lstm_ae",
    "cnn_ae",
    "tranad",
    "graph_stgnn",
]

ALL_FAULTS = ["wing_loss", "sensor_dropout"]

PER_VEHICLE_IDS = ["454", "481", "494", "499", "517", "592", "605", "615"]
GRAPH_SPATIAL_MODES = ["gcn", "graphsage", "gat", "gatv2"]
GRAPH_TEMPORAL_MODES = ["transformer", "rnn", "gru", "lstm"]
# Full STGNN operator grid, every spatial conv paired with every temporal mode.
GRAPH_ABLATION_VARIANTS = [
    {
        "name": f"{graph_conv}_{temporal_mode}",
        "graph_conv": graph_conv,
        "temporal_mode": temporal_mode,
    }
    for graph_conv in GRAPH_SPATIAL_MODES
    for temporal_mode in GRAPH_TEMPORAL_MODES
]

# Display names for models (used in tables)
MODEL_DISPLAY = {
    "isolation_forest": "IF",
    "ocsvm": "OC-SVM",
    "elliptic_envelope": "EE",
    "lstm_ae": "LSTM-AE",
    "cnn_ae": "CNN-AE",
    "tranad": "TranAD",
    "graph_stgnn": "STGNN",
}


def _format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ── Run orchestration ────────────────────────────────────────────────

def _write_results_snapshot(
    output_dir: pathlib.Path,
    *,
    main_results: List[Dict[str, Any]],
    ablation_results: List[Dict[str, Any]],
    models: List[str],
    faults: List[str],
    seeds: List[int],
    device_label: str,
    elapsed_seconds: Optional[float] = None,
    status: str = "running",
    filename: str = "results.partial.json",
) -> pathlib.Path:
    """Persist a recoverable benchmark snapshot during long-running jobs."""
    payload = {
        "main": main_results,
        "ablation": ablation_results,
        "aggregated": aggregate_results(main_results, ablation_results),
        "config": {
            "models": models,
            "faults": faults,
            "seeds": seeds,
            "seed_count": len(seeds),
            "device": device_label,
            "elapsed_seconds": elapsed_seconds,
            "status": status,
        },
    }
    snapshot_path = output_dir / filename
    save_json(payload, snapshot_path)
    return snapshot_path


def _run_single(
    config_path: str,
    model: str,
    faults: List[str],
    seed: int,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    output_dir_override: Optional[str] = None,
    run_group: str = "main",
    variant_name: Optional[str] = None,
    eval_only: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment and return metrics dict.

    Parameters
    ----------
    config_path : str
        Path to base YAML config.
    model : str
        Model name (e.g. ``graph_stgnn``).
    faults : list[str]
        Fault types to inject during evaluation from the same trained model.
    seed : int
        Random seed for this run.
    cfg_overrides : dict, optional
        Additional config overrides (e.g. STGNN operator changes).

    Returns
    -------
    dict
        Metrics from the run, or error info on failure.
    """
    try:
        # Run the benchmark one seed at a time so raw results and checkpoints
        # remain seed-specific and can be aggregated explicitly afterward.
        cfg = load_config(config_path)
        if cfg_overrides:
            for key, val in cfg_overrides.items():
                keys = key.split(".")
                d = cfg
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = val

        cfg["training"]["seeds"] = [seed]
        # One model is trained once, then evaluated against every fault here.
        cfg["faults"]["types"] = list(faults)

        # Write temporary config
        import yaml
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
            tmp_config = f.name

        run_output_dir = pathlib.Path(output_dir_override or cfg["output"]["base_dir"])
        if run_group == "ablation":
            variant_slug = variant_name or model
            run_output_dir = run_output_dir / "ablation" / variant_slug
        else:
            run_output_dir = run_output_dir / "main" / model

        run_experiment(
            tmp_config,
            model_override=model,
            output_dir_override=str(run_output_dir),
            eval_only=eval_only,
        )

        # Read metrics from output
        run_id = f"{model}_seed{seed}"
        out_dir = run_output_dir / run_id
        metrics_file = out_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            return metrics

        return {"model": model, "seed": seed, "faults": list(faults), "error": "no metrics file"}

    except Exception as exc:
        logger.error("Run failed: model=%s faults=%s seed=%d: %s", model, faults, seed, exc)
        return {"model": model, "seed": seed, "faults": list(faults), "error": str(exc)}
    finally:
        # Clean up temp config
        try:
            pathlib.Path(tmp_config).unlink(missing_ok=True)
        except (NameError, OSError):
            pass


def _ensure_processed_tensors(config_path: str, cfg: Dict[str, Any]) -> None:
    """Fail fast when cached tensors are missing and rebuild was not requested."""
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
            "or rerun the benchmark with `--rebuild-data`.\n"
            f"Original error: {exc}"
        ) from exc


def run_main_benchmark(
    config_path: str,
    models: List[str],
    faults: List[str],
    seeds: List[int],
    output_dir_override: Optional[str] = None,
    dry_run: bool = False,
    eval_only: bool = False,
    snapshot_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> List[Dict[str, Any]]:
    """Run the main benchmark across all configured seeds."""
    total = len(models) * len(seeds)
    logger.info(
        "Main benchmark: %d models x %d seeds x 1 %s x %d fault evaluations = %d runs",
        len(models),
        len(seeds),
        "eval-only pass" if eval_only else "training run",
        len(faults),
        total,
    )

    if dry_run:
        # Plan only, print the roster of runs without training anything.
        for model in models:
            for seed in seeds:
                print(
                    f"  [DRY RUN] {MODEL_DISPLAY.get(model, model):>8s} / "
                    f"{len(faults)} faults ({', '.join(faults)}) / seed={seed}"
                )
        return []

    results = []
    tracker = ProgressTracker(total, phase="Main Benchmark")

    # One run per model and seed, each training once then scoring all faults.
    for model in models:
        for seed in seeds:
            result = _run_single(
                config_path,
                model,
                faults,
                seed,
                output_dir_override=output_dir_override,
                run_group="main",
                eval_only=eval_only,
            )
            results.append(result)
            if snapshot_callback is not None:
                snapshot_callback(results)
            tracker.update(model, f"{len(faults)} faults", seed, result)

    tracker.close()
    print(tracker.summary())
    return results


def run_graph_ablation(
    config_path: str,
    faults: List[str],
    seeds: List[int],
    output_dir_override: Optional[str] = None,
    dry_run: bool = False,
    eval_only: bool = False,
    snapshot_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> List[Dict[str, Any]]:
    """Run STGNN operator ablation across all configured seeds."""
    total = len(GRAPH_ABLATION_VARIANTS) * len(seeds)
    logger.info(
        "Graph ablation: %d operator variants x %d seeds x 1 %s x %d fault evaluations = %d runs",
        len(GRAPH_ABLATION_VARIANTS),
        len(seeds),
        "eval-only pass" if eval_only else "training run",
        len(faults),
        total,
    )

    if dry_run:
        # Plan only, list each operator variant and seed without training.
        for variant in GRAPH_ABLATION_VARIANTS:
            for seed in seeds:
                print(
                    "  [DRY RUN] "
                    f"{MODEL_DISPLAY['graph_stgnn']:>8s} / "
                    f"{variant['name']:<18s} / {len(faults)} faults ({', '.join(faults)}) / seed={seed}"
                )
        return []

    results = []
    tracker = ProgressTracker(total, phase="Graph Ablation")

    for variant in GRAPH_ABLATION_VARIANTS:
        for seed in seeds:
            # Pin the graph to the fixed domain-knowledge adjacency, then swap
            # in this variant's spatial conv and temporal operator.
            overrides = {
                "graph.builder": "fixed",
                "graph.adjacency_list": "domain_knowledge",
                "model.graph_conv": variant["graph_conv"],
                "model.temporal_mode": variant["temporal_mode"],
            }
            result = _run_single(
                config_path,
                "graph_stgnn",
                faults,
                seed,
                cfg_overrides=overrides,
                output_dir_override=output_dir_override,
                run_group="ablation",
                variant_name=variant["name"],
                eval_only=eval_only,
            )
            result["graph_variant"] = variant["name"]
            result["graph_conv"] = variant["graph_conv"]
            result["temporal_mode"] = variant["temporal_mode"]
            results.append(result)
            if snapshot_callback is not None:
                snapshot_callback(results)
            tracker.update("graph_stgnn", variant["name"], seed, result)

    tracker.close()
    print(tracker.summary())
    return results


# ── Result aggregation ───────────────────────────────────────────────

def aggregate_results(main_results: List[Dict], ablation_results: List[Dict]) -> Dict[str, Any]:
    """Aggregate raw results into structured summary.

    Failed runs, those carrying an ``error`` key, are excluded throughout.
    Produces nested summaries for the overall main benchmark, per-fault,
    per-vehicle mission F1, and the STGNN operator ablation, each reduced over
    seeds.
    """
    output = {
        "main_benchmark": {},
        "per_fault": {},
        "per_vehicle": {},
        "graph_ablation": {},
    }
    main_metric_keys = [
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

    # --- Main benchmark: aggregate over seeds for each model ---
    for model in ALL_MODELS:
        model_runs = [r for r in main_results if r.get("model") == model and "error" not in r]
        if not model_runs:
            continue
        agg = aggregate_numeric_dicts(model_runs, main_metric_keys)
        agg["seed_count"] = len(model_runs)
        output["main_benchmark"][model] = agg

    # --- Per-fault: aggregate each model/fault over seeds ---
    for model in ALL_MODELS:
        fault_agg = {}
        for fault in ALL_FAULTS:
            runs = [
                r
                for r in main_results
                if r.get("model") == model and "error" not in r
            ]
            if not runs:
                continue
            per_run_fault = [
                r.get("per_fault", {}).get(fault, {})
                for r in runs
                if fault in r.get("per_fault", {})
            ]
            if per_run_fault:
                stats = aggregate_numeric_dicts(per_run_fault, main_metric_keys)
                stats["seed_count"] = len(per_run_fault)
                fault_agg[fault] = stats
        if fault_agg:
            output["per_fault"][model] = fault_agg

    # --- Per-mission: mission-level F1 by mission id aggregated over seeds ---
    for model in ALL_MODELS:
        mission_agg = {}
        for mission_id in PER_VEHICLE_IDS:
            values = []
            for run in main_results:
                if run.get("model") != model or "error" in run:
                    continue
                for split in run.get("splits", []):
                    fault_metrics = split.get("fault_metrics", {}).get("wing_loss", {})
                    for mission_metrics in fault_metrics.get("mission_metrics", []):
                        if mission_metrics.get("mission_id") == mission_id:
                            f1 = mission_metrics.get("f1")
                            if f1 is not None:
                                values.append(f1)
            if values:
                stats = aggregate_numeric_dicts(
                    [{"f1": value} for value in values],
                    ["f1"],
                )
                stats["seed_count"] = len(values)
                mission_agg[mission_id] = stats
        if mission_agg:
            avg_values = [entry["f1_mean"] for entry in mission_agg.values()]
            avg_stats = aggregate_numeric_dicts(
                [{"f1": value} for value in avg_values],
                ["f1"],
            )
            avg_stats["seed_count"] = len(avg_values)
            mission_agg["avg"] = avg_stats
            output["per_vehicle"][model] = mission_agg

    # --- Graph ablation: aggregate operator variants over seeds ---
    stgnn_variants = {}
    for variant in GRAPH_ABLATION_VARIANTS:
        runs = [
            r
            for r in ablation_results
            if r.get("graph_variant") == variant["name"] and "error" not in r
        ]
        if not runs:
            continue
        stats = aggregate_numeric_dicts(runs, ["accuracy", "f1"])
        stgnn_variants[variant["name"]] = {
            "graph_conv": variant["graph_conv"],
            "temporal_mode": variant["temporal_mode"],
            "seed_count": len(runs),
            **stats,
        }
    if stgnn_variants:
        output["graph_ablation"]["graph_stgnn"] = stgnn_variants

    return output


# ── CLI entry point ──────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Full benchmark runner. Orchestrates all experiments and aggregates results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to experiment YAML config.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Run specific models only (e.g. --models graph_stgnn tranad).")
    parser.add_argument("--faults", nargs="+", default=None,
                        help="Run specific fault types only (e.g. --faults wing_loss sensor_dropout).")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip graph ablation study.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be run without executing.")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory for results JSON.")
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild processed tensors from raw parquet once before the benchmark.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, load the existing checkpoint(s), and rerun evaluation only.",
    )
    return parser.parse_args()


def run(
    config_path: str,
    models: Optional[List[str]] = None,
    faults: Optional[List[str]] = None,
    *,
    skip_ablation: bool = False,
    dry_run: bool = False,
    output_dir_override: Optional[str] = None,
    rebuild_data: bool = False,
    eval_only: bool = False,
) -> None:
    """Run the full benchmark end to end and write the results.

    Optionally rebuilds processed tensors first, runs the main model roster,
    then the STGNN operator ablation unless skipped, and finally aggregates and
    saves ``results.json``. Recoverable partial snapshots are written after each
    completed run so a long job can be resumed.

    Parameters
    ----------
    config_path : str
        Path to the base experiment YAML config.
    models : list[str], optional
        Subset of models to run, defaults to the full roster.
    faults : list[str], optional
        Subset of fault types to evaluate, defaults to all configured faults.
    skip_ablation : bool
        Skip the STGNN operator ablation phase.
    dry_run : bool
        Print the planned runs without training or saving anything.
    rebuild_data : bool
        Rebuild processed tensors from raw parquet once before benchmarking.
    eval_only : bool
        Reuse existing checkpoints and rerun evaluation only.
    """
    if rebuild_data and not dry_run:
        from workflows.prepare_data import run as prepare_data_run

        print("\n-- Data preparation ------------------------------------------------------------")
        prepare_data_run(config_path)
    elif rebuild_data and dry_run:
        logger.info("Ignoring --rebuild-data during dry-run; cached processed tensors will be assumed.")

    cfg = load_config(config_path)
    validate_config(cfg)
    runtime = configure_runtime(cfg)
    loader_workers = 0
    if not dry_run and not rebuild_data:
        # Fail fast if the cached tensors are missing or stale before training.
        _ensure_processed_tensors(config_path, cfg)

    models = models if models else ALL_MODELS
    faults = faults if faults else ALL_FAULTS
    configured_seeds = cfg["training"]["seeds"]
    if not configured_seeds:
        logger.error("training.seeds must contain at least one seed")
        sys.exit(1)
    benchmark_seeds = [int(seed) for seed in configured_seeds]

    # Validate model names
    for m in models:
        if m not in ALL_MODELS:
            logger.error("Unknown model: %s. Available: %s", m, ALL_MODELS)
            sys.exit(1)

    # Validate fault names
    for f in faults:
        if f not in ALL_FAULTS:
            logger.error("Unknown fault: %s. Available: %s", f, ALL_FAULTS)
            sys.exit(1)

    output_dir = pathlib.Path(output_dir_override or cfg["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────
    n_main = len(models) * len(benchmark_seeds)
    n_abl = len(GRAPH_ABLATION_VARIANTS) * len(benchmark_seeds) if not skip_ablation else 0
    n_total = n_main + n_abl

    print("\n" + "=" * 70)
    print("  AUV Anomaly Detection, Full Benchmark")
    print("=" * 70)
    print(f"  Device:      {runtime.device_label}")
    print(f"  CPU threads: {runtime.cpu_threads}")
    print(f"  Loader wkrs: {loader_workers}")
    print(f"  Mode:        {'evaluate existing checkpoints' if eval_only else 'train + evaluate'}")
    print(f"  Models:      {', '.join(MODEL_DISPLAY.get(m, m) for m in models)} ({len(models)})")
    print(f"  Faults:      {', '.join(faults)} ({len(faults)} per trained run)")
    print(f"  Seeds:       {', '.join(str(seed) for seed in benchmark_seeds)} ({len(benchmark_seeds)})")
    print(
        "  Data:        "
        + ("rebuild from raw parquet" if rebuild_data and not dry_run else "use cached processed tensors")
    )
    print(
        f"  Runs:        {n_main} main + {n_abl} ablation = {n_total} total "
        f"{'evaluation passes' if eval_only else 'training runs'}"
    )
    print(f"  Output:      {output_dir}")
    if dry_run:
        print(f"  Mode:        DRY RUN (no models will be trained)")
    print("=" * 70 + "\n")

    overall_start = time.time()
    partial_results_path = output_dir / "results.partial.json"

    def snapshot_main(current_main_results: List[Dict[str, Any]]) -> None:
        _write_results_snapshot(
            output_dir,
            main_results=current_main_results,
            ablation_results=[],
            models=models,
            faults=faults,
            seeds=benchmark_seeds,
            device_label=runtime.device_label,
            elapsed_seconds=time.time() - overall_start,
            status="running_main",
        )

    # 1. Main benchmark
    print("\n── Phase 1/2: Main Benchmark " + "─" * 42)
    main_results = run_main_benchmark(
        config_path,
        models,
        faults,
        benchmark_seeds,
        output_dir_override=str(output_dir),
        dry_run=dry_run,
        eval_only=eval_only,
        snapshot_callback=None if dry_run else snapshot_main,
    )

    # 2. Graph ablation
    ablation_results = []
    if not skip_ablation:
        def snapshot_ablation(current_ablation_results: List[Dict[str, Any]]) -> None:
            _write_results_snapshot(
                output_dir,
                main_results=main_results,
                ablation_results=current_ablation_results,
                models=models,
                faults=faults,
                seeds=benchmark_seeds,
                device_label=runtime.device_label,
                elapsed_seconds=time.time() - overall_start,
                status="running_ablation",
            )

        print("\n── Phase 2/2: Graph Ablation " + "─" * 42)
        ablation_results = run_graph_ablation(
            config_path,
            faults,
            benchmark_seeds,
            output_dir_override=str(output_dir),
            dry_run=dry_run,
            eval_only=eval_only,
            snapshot_callback=None if dry_run else snapshot_ablation,
        )
    else:
        print("\n── Phase 2/2: Graph Ablation (skipped) " + "─" * 31)

    elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print(f"  Benchmark complete in {_format_eta(elapsed)}")
    print("=" * 70)

    if dry_run:
        logger.info("Dry run complete, no results to save.")
        return

    # Write the final aggregated results.json and drop the partial snapshot.
    results_path = _write_results_snapshot(
        output_dir,
        main_results=main_results,
        ablation_results=ablation_results,
        models=models,
        faults=faults,
        seeds=benchmark_seeds,
        device_label=runtime.device_label,
        elapsed_seconds=elapsed,
        status="completed",
        filename="results.json",
    )
    partial_results_path.unlink(missing_ok=True)
    logger.info("Raw results saved to %s", results_path)


def main() -> None:
    """Configure logging, parse arguments, and run the benchmark."""
    args = parse_args()
    setup_logging()
    run(
        args.config,
        models=args.models,
        faults=args.faults,
        skip_ablation=args.skip_ablation,
        dry_run=args.dry_run,
        output_dir_override=args.output_dir,
        rebuild_data=args.rebuild_data,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
