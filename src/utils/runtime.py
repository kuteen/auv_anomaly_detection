"""Runtime configuration helpers for CPU threading and device selection.

Resolves a single :class:`RuntimeContext` from the ``runtime`` block of the
project config, covering compute device, intra- and inter-op thread counts,
and DataLoader settings such as worker count, pinned memory and prefetch.
Values default to ``"auto"`` and are derived from the available hardware, so a
config can stay minimal while still adapting to CPU-only or CUDA machines.
Thread-pool and interop limits are process-global side effects, applied once
and guarded so repeated calls stay safe.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional dependency
    threadpool_limits = None

logger = logging.getLogger(__name__)

_INTEROP_THREADS_SET = False
_THREADPOOL_CONTROLLER = None


@dataclass(frozen=True)
class RuntimeContext:
    """Resolved execution settings for the current process."""

    device: torch.device
    device_type: str
    device_label: str
    cpu_threads: int
    interop_threads: int
    dataloader_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: Optional[int]
    non_blocking: bool


# ── Thread and device resolution ────────────────────────────────────────


def _parse_thread_setting(value: Any, fallback: int) -> int:
    """Resolve a thread-count setting, mapping ``"auto"`` to ``fallback``.

    Args:
        value: Raw config value, ``None``, ``""`` or ``"auto"`` select the fallback.
        fallback: Auto-derived thread count to use when unset.

    Returns:
        A positive integer thread count.

    Raises:
        ValueError: If an explicit count is less than one.
    """
    if value in (None, "", "auto"):
        return fallback
    parsed = int(value)
    if parsed < 1:
        raise ValueError("Thread counts must be >= 1")
    return parsed


def _resolve_device(runtime_cfg: Dict[str, Any]) -> torch.device:
    """Resolve the compute device from the requested mode and CUDA availability.

    ``"auto"`` prefers CUDA when present and otherwise falls back to CPU. An
    explicit ``"cuda"`` request on a machine without CUDA warns and degrades to
    CPU rather than raising.

    Raises:
        ValueError: If ``runtime.device`` is not one of auto, cpu or cuda.
    """
    requested = str(runtime_cfg.get("device", "auto")).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("runtime.device must be one of ['auto', 'cpu', 'cuda']")

    has_cuda = torch.cuda.is_available()
    # Explicit CUDA request on a CPU-only host degrades gracefully.
    if requested == "cuda" and not has_cuda:
        logger.warning("runtime.device=cuda requested but CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    # "auto": take CUDA when present, otherwise CPU.
    if has_cuda:
        return torch.device("cuda")
    return torch.device("cpu")


# ── Runtime configuration ───────────────────────────────────────────────


def configure_runtime(cfg: Dict[str, Any]) -> RuntimeContext:
    """Configure thread pools and device defaults for the current process.

    Reads the ``runtime`` block of ``cfg``, applies process-global thread,
    precision and CUDA backend settings, and returns the resolved context. The
    interop thread limit can only be set once per process, so it is guarded and
    silently skipped on later calls.

    Args:
        cfg: Project config, the optional ``"runtime"`` key holds device and
            threading overrides, every field defaults to an auto-derived value.

    Returns:
        The fully resolved :class:`RuntimeContext` for the process.
    """
    global _INTEROP_THREADS_SET, _THREADPOOL_CONTROLLER

    runtime_cfg = cfg.get("runtime", {})
    cpu_count = max(1, os.cpu_count() or 1)
    device = _resolve_device(runtime_cfg)
    device_type = device.type
    device_label = "cpu"
    if device_type == "cuda":
        device_label = torch.cuda.get_device_name(0)

    # Intra-op threads default to all cores, interop to a small slice of them.
    cpu_threads = _parse_thread_setting(runtime_cfg.get("cpu_threads", "auto"), cpu_count)
    interop_threads = _parse_thread_setting(
        runtime_cfg.get("interop_threads", "auto"),
        min(cpu_threads, 4),
    )

    # GPU runs can feed more loader workers than CPU-only runs, capped to keep
    # oversubscription in check.
    workers_default = min(cpu_count, 8) if device_type == "cuda" else min(cpu_count, 4)
    dataloader_workers = max(
        0,
        int(runtime_cfg.get("dataloader_workers", workers_default))
        if runtime_cfg.get("dataloader_workers", "auto") != "auto"
        else workers_default,
    )
    # Pinned memory only helps host-to-device transfers, so default it on for CUDA.
    pin_memory_value = runtime_cfg.get("pin_memory", "auto")
    if pin_memory_value == "auto":
        pin_memory = device_type == "cuda"
    else:
        pin_memory = bool(pin_memory_value)
    # prefetch_factor and persistent_workers are only meaningful with workers > 0.
    prefetch_factor = int(runtime_cfg.get("prefetch_factor", 4)) if dataloader_workers > 0 else None
    persistent_workers = bool(runtime_cfg.get("persistent_workers", True)) and dataloader_workers > 0

    # Apply the global thread limits, BLAS pools via threadpoolctl plus torch.
    if threadpool_limits is not None:
        _THREADPOOL_CONTROLLER = threadpool_limits(limits=cpu_threads)
    torch.set_num_threads(cpu_threads)
    # Interop thread count is immutable after first use, set it at most once.
    if not _INTEROP_THREADS_SET:
        torch.set_num_interop_threads(interop_threads)
        _INTEROP_THREADS_SET = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(runtime_cfg.get("matmul_precision", "high")))

    # CUDA-only backend tuning, autotune kernels and enable TF32 where supported.
    if device_type == "cuda":
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = bool(runtime_cfg.get("allow_tf32", True))
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(runtime_cfg.get("allow_tf32", True))

    return RuntimeContext(
        device=device,
        device_type=device_type,
        device_label=device_label,
        cpu_threads=cpu_threads,
        interop_threads=interop_threads,
        dataloader_workers=dataloader_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        non_blocking=device_type == "cuda" and pin_memory,
    )
