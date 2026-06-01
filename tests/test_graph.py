"""Graph construction on the 19-sensor contract."""

from __future__ import annotations

import numpy as np
import torch

from data.graph_builders import (
    CorrelationGraphBuilder,
    FixedGraphBuilder,
    LearnedGraphBuilder,
    adjacency_to_edge_index,
    normalise_adjacency,
)
from data.graph_topology import (
    SENSOR_TO_IDX,
    build_adjacency_matrix,
)

from .conftest import N_SENSORS


def _assert_valid_adjacency(A: np.ndarray) -> None:
    assert A.shape == (N_SENSORS, N_SENSORS)
    assert np.allclose(A, A.T), "adjacency must be symmetric (undirected)"
    assert np.allclose(np.diag(A), 0.0), "raw adjacency must have a zero diagonal"


def test_sensor_index_map_matches_contract() -> None:
    assert len(SENSOR_TO_IDX) == N_SENSORS
    assert sorted(SENSOR_TO_IDX.values()) == list(range(N_SENSORS))


def test_domain_adjacency_matrix() -> None:
    A = build_adjacency_matrix(N_SENSORS)
    _assert_valid_adjacency(A)
    assert np.count_nonzero(A) > 0
    # Self-loops are intentionally added only during normalisation.
    assert not np.any(np.diag(A))


def test_normalise_adjacency_is_symmetric_with_self_loops() -> None:
    A = build_adjacency_matrix(N_SENSORS)
    A_norm = normalise_adjacency(A)
    assert A_norm.shape == (N_SENSORS, N_SENSORS)
    assert np.allclose(A_norm, A_norm.T)
    # Self-loops mean the normalised diagonal is strictly positive.
    assert np.all(np.diag(A_norm) > 0.0)


def test_adjacency_to_edge_index() -> None:
    A = build_adjacency_matrix(N_SENSORS)
    edge_index = adjacency_to_edge_index(A)
    assert isinstance(edge_index, torch.Tensor)
    assert edge_index.dtype == torch.long
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == int(np.count_nonzero(A))


def test_fixed_builder_domain_knowledge() -> None:
    builder = FixedGraphBuilder(N_SENSORS, "domain_knowledge")
    A, A_norm, edge_index = builder.build()
    _assert_valid_adjacency(A)
    assert np.allclose(A_norm, A_norm.T)
    assert edge_index.shape[0] == 2
    # The domain builder must reproduce the canonical topology edge set.
    # Edges are undirected and a couple are specified reciprocally, so the
    # nonzero count is taken from the reference matrix rather than 2x the
    # raw edge list length.
    A_ref = build_adjacency_matrix(N_SENSORS)
    assert np.count_nonzero(A) == np.count_nonzero(A_ref)
    # The fixed builder uses binary edges, so the sparsity pattern matches.
    assert np.array_equal(A != 0, A_ref != 0)


def test_fixed_builder_identity_and_full_presets() -> None:
    A_id, _, _ = FixedGraphBuilder(N_SENSORS, "identity").build()
    assert np.count_nonzero(A_id) == 0

    A_full, _, _ = FixedGraphBuilder(N_SENSORS, "full").build()
    _assert_valid_adjacency(A_full)
    assert np.count_nonzero(A_full) == N_SENSORS * (N_SENSORS - 1)


def test_correlation_builder(correlated_series) -> None:
    builder = CorrelationGraphBuilder(threshold=0.3)
    A, A_norm, edge_index = builder.build(correlated_series)
    _assert_valid_adjacency(A)
    assert set(np.unique(A)).issubset({0.0, 1.0})
    assert np.allclose(A_norm, A_norm.T)


def test_learned_builder_is_differentiable(correlated_series) -> None:
    builder = LearnedGraphBuilder(N_SENSORS, init_data=correlated_series)
    soft = builder.forward()
    assert soft.shape == (N_SENSORS, N_SENSORS)
    assert soft.requires_grad
    # Soft adjacency is bounded by the sigmoid and symmetric with zero diagonal.
    assert torch.all(soft >= 0.0) and torch.all(soft <= 1.0)
    assert torch.allclose(soft, soft.T, atol=1e-6)
    assert torch.allclose(torch.diag(soft), torch.zeros(N_SENSORS), atol=1e-6)

    A_bin, A_norm, edge_index = builder.build()
    _assert_valid_adjacency(A_bin)
    assert np.allclose(A_norm, A_norm.T)
