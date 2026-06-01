"""Synthetic fault injection on processed window tensors."""

from __future__ import annotations

import numpy as np
import pytest

from data import faults

from .conftest import N_SENSORS, WINDOW_LENGTH

FAULT_TYPES = ["wing_loss", "sensor_dropout"]


@pytest.mark.parametrize("fault_type", FAULT_TYPES)
def test_inject_fault_preserves_shape_and_dtype(window_tensor, fault_type) -> None:
    out, labels, meta = faults.inject_fault(window_tensor, fault_type, seed=1)
    assert out.shape == (window_tensor.shape[0], WINDOW_LENGTH, N_SENSORS)
    assert out.dtype == np.float32
    assert labels.shape == (window_tensor.shape[0],)
    assert labels.dtype == bool


@pytest.mark.parametrize("fault_type", FAULT_TYPES)
def test_inject_fault_changes_values(window_tensor, fault_type) -> None:
    out, _labels, _meta = faults.inject_fault(window_tensor, fault_type, seed=1)
    # The original array must be left untouched (works on a copy).
    assert not np.shares_memory(out, window_tensor)
    # And values must actually differ somewhere.
    assert not np.array_equal(out, window_tensor)


@pytest.mark.parametrize("fault_type", FAULT_TYPES)
def test_inject_fault_labels_mark_a_tail(window_tensor, fault_type) -> None:
    out, labels, meta = faults.inject_fault(window_tensor, fault_type, seed=1)
    assert labels.any(), "at least one window must be flagged anomalous"
    # Faults run from an onset window to the end, so the final window is True.
    assert bool(labels[-1])
    assert meta["fault_window_count"] == int(labels.sum())
    for key in ("fault_type", "fault_window_idx", "fault_window_count", "fault_channel"):
        assert key in meta
    assert meta["fault_type"] == fault_type


def test_sensor_dropout_zeros_target_channel(window_tensor) -> None:
    out, labels, meta = faults.inject_fault(
        window_tensor, "sensor_dropout", channel=3, seed=1
    )
    ch = meta["fault_channel"]
    assert ch == 3
    onset = meta["fault_window_idx"]
    # The faulted tail of the target channel must be exactly zero.
    assert np.all(out[onset:, :, ch] == 0.0)
    # Channels other than the target must be unchanged.
    other = [c for c in range(N_SENSORS) if c != ch]
    assert np.array_equal(out[:, :, other], window_tensor[:, :, other])


def test_unknown_fault_raises() -> None:
    arr = np.ones((4, WINDOW_LENGTH, N_SENSORS), dtype=np.float32)
    with pytest.raises(ValueError):
        faults.inject_fault(arr, "does_not_exist")


def test_wrong_rank_raises() -> None:
    flat = np.ones((10, N_SENSORS), dtype=np.float32)
    with pytest.raises(ValueError):
        faults.inject_fault(flat, "wing_loss")
