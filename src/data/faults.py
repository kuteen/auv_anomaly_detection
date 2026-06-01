"""Synthetic fault injection for evaluation.

The public benchmark injects faults directly on processed test windows.
Each generator modifies a copy of the input array and returns a tuple
``(faulted_windows, labels, metadata)`` where:

- ``faulted_windows`` has shape ``[N, K, m]``
- ``labels`` has shape ``[N]`` and marks anomalous windows
- ``metadata`` captures the affected window/channel for traceability

Faults are persistent. Every generator resolves a single onset window
from the ``onset`` fraction, then labels that window and all later
windows as anomalous so the fault carries through to mission end. The
inputs are processed window tensors of shape ``[N, K, m]`` where ``N``
is the window count, ``K`` is the window length (the fixed grid is 64
timesteps, so windows are typically ``[N, 64, m]``), and ``m`` is the
sensor channel count.

The persistence convention is implemented as ``labels[window_idx:] =
True``. Once the onset window is resolved, every window from that index
to the final window is marked anomalous, matching real degradations that
do not self-heal within a mission.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np

FaultResult = Tuple[np.ndarray, np.ndarray, Dict[str, Any]]


# ── Fault generators ───────────────────────────────────────────


def wing_loss(
    data: np.ndarray,
    channel: int | str = 14,
    onset: float = 0.5,
    magnitude: float = 2.0,
    seed: int = 42,
    channel_names: list | None = None,
    range_min: float = 0.0,
    range_max: float = 1.0,
) -> FaultResult:
    """Inject a persistent roll fault from onset to mission end.

    Simulates a wing or roll-actuator failure by adding a constant offset
    plus light Gaussian jitter to a single channel, then renormalising the
    perturbed values back into the dataset range.

    Args:
        data: Processed window tensor of shape ``[N, K, m]``.
        channel: Affected sensor as an integer index or a name resolved
            against ``channel_names``. Defaults to the roll channel.
        onset: Fraction in ``[0, 1]`` locating the first faulted window.
        magnitude: Constant offset added to the channel before jitter.
        seed: Seed for the additive-noise generator, for reproducibility.
        channel_names: Optional channel-name list for string resolution.
        range_min: Lower bound of the post-fault renormalisation range.
        range_max: Upper bound of the post-fault renormalisation range.

    Returns:
        Tuple ``(faulted_windows, labels, metadata)``. ``labels`` has shape
        ``[N]`` and is ``True`` from the onset window onwards.
    """
    out = _validate_window_tensor(data)
    ch = _resolve_channel(channel, channel_names)
    rng = np.random.default_rng(seed)
    n_windows, window_length, _ = out.shape
    window_idx = _resolve_window_index(onset, n_windows)
    labels = np.zeros(n_windows, dtype=bool)
    # Persistent tail: onset window and every later window are anomalous.
    labels[window_idx:] = True

    # Perturb only the faulted tail, leaving pre-onset windows untouched.
    for idx in range(window_idx, n_windows):
        perturbed = out[idx, :, ch].copy()
        perturbed += magnitude + (0.05 * rng.standard_normal(window_length))
        out[idx, :, ch] = _renormalise_to_range(
            perturbed,
            range_min=range_min,
            range_max=range_max,
        )
    return out, labels, _fault_metadata(
        fault_type="wing_loss",
        window_idx=window_idx,
        fault_window_count=int(labels.sum()),
        channel_idx=ch,
        channel_names=channel_names,
    )


def sensor_dropout(
    data: np.ndarray,
    channel: int | str | None = None,
    onset: float = 0.5,
    duration: float = 0.3,
    seed: int = 42,
    channel_names: list | None = None,
    zero_value: float = 0.0,
) -> FaultResult:
    """Zero one sensor channel from onset to mission end.

    Simulates a dead sensor or telemetry dropout by overwriting a single
    channel with ``zero_value`` across the faulted tail. When ``channel``
    is ``None`` a channel is drawn at random from the seeded generator.

    Args:
        data: Processed window tensor of shape ``[N, K, m]``.
        channel: Affected sensor as an integer index or a name, or ``None``
            to pick a random channel.
        onset: Fraction in ``[0, 1]`` locating the first faulted window.
        duration: Accepted for backwards compatibility and ignored, the
            fault always persists from onset to the end of the mission.
        seed: Seed for random channel selection, for reproducibility.
        channel_names: Optional channel-name list for string resolution.
        zero_value: Value written into the dropped channel.

    Returns:
        Tuple ``(faulted_windows, labels, metadata)``. ``labels`` has shape
        ``[N]`` and is ``True`` from the onset window onwards.
    """
    del duration  # Fault duration is now defined by the onset-to-end tail.

    out = _validate_window_tensor(data)
    rng = np.random.default_rng(seed)
    n_windows, _, n_channels = out.shape
    window_idx = _resolve_window_index(onset, n_windows)
    if channel is None:
        ch = int(rng.integers(0, n_channels))
    else:
        ch = _resolve_channel(channel, channel_names)

    labels = np.zeros(n_windows, dtype=bool)
    # Persistent tail: onset window and every later window are anomalous.
    labels[window_idx:] = True
    # Overwrite the channel across the whole faulted tail in one slice.
    out[window_idx:, :, ch] = zero_value
    return out, labels, _fault_metadata(
        fault_type="sensor_dropout",
        window_idx=window_idx,
        fault_window_count=int(labels.sum()),
        channel_idx=ch,
        channel_names=channel_names,
    )


# ── Dispatch ───────────────────────────────────────────────────


FAULT_REGISTRY: Dict[str, Callable[..., FaultResult]] = {
    "wing_loss": wing_loss,
    "sensor_dropout": sensor_dropout,
}


def inject_fault(
    data: np.ndarray,
    fault_type: str,
    **kwargs,
) -> FaultResult:
    """Dispatch to a named fault generator on a processed window tensor."""
    if fault_type not in FAULT_REGISTRY:
        raise ValueError(
            f"Unknown fault type '{fault_type}'. Available: {list(FAULT_REGISTRY)}"
        )
    return FAULT_REGISTRY[fault_type](data, **kwargs)


# ── Helpers ────────────────────────────────────────────────────


def _validate_window_tensor(data: np.ndarray) -> np.ndarray:
    """Copy the input and assert a non-empty ``[N, K, m]`` window tensor."""
    out = np.array(data, copy=True)
    if out.ndim != 3:
        raise ValueError(
            "Fault injection now expects processed window tensors with shape [N, K, m]."
        )
    if out.shape[0] == 0:
        raise ValueError("Cannot inject a fault into an empty window tensor.")
    return out


def _resolve_window_index(onset: float, n_windows: int) -> int:
    """Map an onset fraction to a window index, clamped to ``[0, N-1]``."""
    # Floor the fractional position, then clamp so the onset always lands on
    # a valid window even for onset values of 0.0 or 1.0.
    idx = int(np.floor(n_windows * float(onset)))
    return max(0, min(n_windows - 1, idx))


def _resolve_channel(channel: int | str, names: list | None) -> int:
    """Resolve a channel index from an int, or a name via ``names``."""
    if isinstance(channel, int):
        return channel
    if names is None:
        raise ValueError("channel_names must be provided when channel is a string")
    if channel not in names:
        raise ValueError(f"Channel '{channel}' not in names list")
    return names.index(channel)


def _fault_metadata(
    *,
    fault_type: str,
    window_idx: int,
    fault_window_count: int,
    channel_idx: int,
    channel_names: list | None,
) -> Dict[str, Any]:
    """Assemble the traceability metadata returned by every generator."""
    metadata: Dict[str, Any] = {
        "fault_type": fault_type,
        "fault_window_idx": int(window_idx),
        "fault_window_count": int(fault_window_count),
        "fault_channel": int(channel_idx),
    }
    if channel_names is not None and 0 <= channel_idx < len(channel_names):
        metadata["fault_channel_name"] = str(channel_names[channel_idx])
    return metadata


def _renormalise_to_range(
    values: np.ndarray,
    *,
    range_min: float,
    range_max: float,
) -> np.ndarray:
    """Min-max rescale ``values`` into ``[range_min, range_max]``."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi == lo:
        return np.full_like(values, fill_value=range_max, dtype=np.float32)
    scaled = (values - lo) / (hi - lo)
    return (range_min + (scaled * (range_max - range_min))).astype(np.float32, copy=False)
