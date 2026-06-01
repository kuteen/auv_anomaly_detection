"""Graph construction strategies for sensor networks.

Three builders are provided:

* **FixedGraphBuilder** – uses a hand-crafted adjacency list.
* **CorrelationGraphBuilder** – estimates edges from pairwise Pearson
  correlation of training data with a threshold and optional top-*k*.
* **LearnedGraphBuilder** – starts from a correlation graph and learns
  a continuous adjacency as a trainable parameter matrix.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def normalise_adjacency(A: np.ndarray) -> np.ndarray:
    """Symmetric normalisation: D^{-1/2} A D^{-1/2} with self-loops."""
    # Add self-loops, then scale by inverse-sqrt degree on both sides so the
    # operator is symmetric and its spectral radius stays bounded.
    A_hat = A + np.eye(A.shape[0])
    D = np.diag(A_hat.sum(axis=1))
    # Floor the degree before the reciprocal sqrt to avoid divide-by-zero.
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D.diagonal(), 1e-12)))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def adjacency_to_edge_index(A: np.ndarray) -> torch.LongTensor:
    """Convert a dense adjacency matrix to COO ``edge_index [2, E]``."""
    # Each non-zero entry becomes one directed edge, row 0 holds sources and
    # row 1 holds destinations, matching the PyG edge_index convention.
    src, dst = np.nonzero(A)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


# ────────────────────────────────────────────────────────────────────────


class FixedGraphBuilder:
    """Build an adjacency matrix from a manually specified adjacency list.

    Parameters
    ----------
    n_sensors : int
    adjacency_list : dict
        Mapping ``{sensor_idx: [neighbour_idx, ...], ...}``.
        May also be a path to a JSON file.
        Can also be "domain_knowledge" to use the built-in AUV topology.
    """

    def __init__(self, n_sensors: int, adjacency_list: Dict | str | pathlib.Path) -> None:
        self.m = n_sensors
        
        # Handle built-in graph presets first.
        if isinstance(adjacency_list, str) and adjacency_list == "identity":
            adjacency_list = {i: [] for i in range(self.m)}
        elif isinstance(adjacency_list, str) and adjacency_list == "full":
            adjacency_list = {
                i: [j for j in range(self.m) if j != i]
                for i in range(self.m)
            }
        elif isinstance(adjacency_list, str) and adjacency_list == "domain_knowledge":
            try:
                from data.graph_topology import build_adjacency_matrix, SENSOR_TO_IDX
                A = build_adjacency_matrix(n_sensors)
                # Convert to adjacency list format
                adjacency_list = {}
                for i in range(A.shape[0]):
                    neighbors = []
                    for j in range(A.shape[1]):
                        if A[i, j] != 0 and i != j:
                            neighbors.append(j)
                    if neighbors:
                        adjacency_list[i] = neighbors
            except ImportError:
                logger.warning("Could not load domain knowledge graph, using empty graph")
                adjacency_list = {}
        elif isinstance(adjacency_list, (str, pathlib.Path)):
            with open(adjacency_list) as fh:
                adjacency_list = json.load(fh)
        
        self._adj_list = {int(k): [int(v) for v in vs] for k, vs in adjacency_list.items()}

    def build(self, adjacency_weights: Dict = None, **_kwargs) -> Tuple[np.ndarray, np.ndarray, torch.LongTensor]:
        """Build the adjacency matrix.
        
        Parameters
        ----------
        adjacency_weights : dict, optional
            Optional weights for edges. If not provided, uses binary edges.
        """
        A = np.zeros((self.m, self.m), dtype=np.float32)
        for src, dsts in self._adj_list.items():
            for dst in dsts:
                # Use weight if provided, otherwise default to 1.0
                weight = 1.0
                if adjacency_weights is not None:
                    weight = adjacency_weights.get((src, dst), adjacency_weights.get((dst, src), 1.0))
                A[src, dst] = weight
                A[dst, src] = weight  # Undirected graph
        A_norm = normalise_adjacency(A)
        edge_index = adjacency_to_edge_index(A)
        return A, A_norm, edge_index


class CorrelationGraphBuilder:
    """Build adjacency from pairwise absolute Pearson correlation.

    Parameters
    ----------
    threshold : float
        Minimum absolute correlation to include an edge.
    top_k : int or None
        If set, keep only the *k* strongest neighbours per node
        (after thresholding).
    """

    def __init__(self, threshold: float = 0.3, top_k: Optional[int] = None) -> None:
        self.threshold = threshold
        self.top_k = top_k

    def build(
        self, train_data: np.ndarray, **_kwargs
    ) -> Tuple[np.ndarray, np.ndarray, torch.LongTensor]:
        """
        Parameters
        ----------
        train_data : np.ndarray  shape [T, m]
        """
        corr = np.abs(np.corrcoef(train_data.T))  # [m, m]
        np.fill_diagonal(corr, 0.0)
        A = (corr >= self.threshold).astype(np.float32)

        if self.top_k is not None:
            for i in range(A.shape[0]):
                row = corr[i].copy()
                row[A[i] == 0] = -1
                if (row >= 0).sum() > self.top_k:
                    cutoff = np.sort(row)[::-1][self.top_k]
                    A[i, row < cutoff] = 0.0
            # Symmetrise
            A = np.maximum(A, A.T)

        A_norm = normalise_adjacency(A)
        edge_index = adjacency_to_edge_index(A)
        logger.info(
            "CorrelationGraph: %d edges (threshold=%.2f, top_k=%s)",
            int(A.sum()),
            self.threshold,
            self.top_k,
        )
        return A, A_norm, edge_index


class LearnedGraphBuilder(nn.Module):
    """Trainable adjacency initialised from correlation estimates.

    The raw parameter matrix is passed through a sigmoid so values
    remain in [0, 1].  During training the adjacency is differentiable;
    for inference it can be binarised with a threshold.

    Parameters
    ----------
    n_sensors : int
    init_data : np.ndarray  [T, m] – used for correlation initialisation.
    threshold : float
        Correlation threshold for the initial estimate.
    """

    def __init__(
        self,
        n_sensors: int,
        init_data: Optional[np.ndarray] = None,
        threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.m = n_sensors

        if init_data is not None:
            corr = np.abs(np.corrcoef(init_data.T))
            np.fill_diagonal(corr, 0.0)
            # Initialise logits so sigmoid ≈ corr values
            logits = np.log(np.clip(corr, 1e-6, 1 - 1e-6) / (1 - np.clip(corr, 1e-6, 1 - 1e-6)))
        else:
            logits = np.zeros((n_sensors, n_sensors), dtype=np.float32)

        self._logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))
        self._threshold = threshold

    def forward(self) -> torch.Tensor:
        """Return a differentiable adjacency matrix ``[m, m]``."""
        A = torch.sigmoid(self._logits)
        A = (A + A.T) / 2  # symmetrise
        A = A - torch.diag(torch.diag(A))  # zero diagonal
        return A

    def build(self, **_kwargs) -> Tuple[np.ndarray, np.ndarray, torch.LongTensor]:
        """Return numpy arrays for compatibility with other builders."""
        with torch.no_grad():
            A_soft = self.forward().cpu().numpy()
        A_bin = (A_soft >= self._threshold).astype(np.float32)
        A_norm = normalise_adjacency(A_bin)
        edge_index = adjacency_to_edge_index(A_bin)
        return A_bin, A_norm, edge_index
