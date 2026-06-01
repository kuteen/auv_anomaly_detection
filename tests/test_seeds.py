"""Deterministic seeding reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from utils.seeds import set_global_seed


def test_torch_draw_is_reproducible() -> None:
    set_global_seed(123)
    a = torch.randn(8)
    set_global_seed(123)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_numpy_draw_is_reproducible() -> None:
    set_global_seed(7)
    a = np.random.rand(5)
    set_global_seed(7)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_python_random_is_reproducible() -> None:
    set_global_seed(99)
    a = [random.random() for _ in range(5)]
    set_global_seed(99)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_different_seeds_differ() -> None:
    set_global_seed(1)
    a = torch.randn(8)
    set_global_seed(2)
    b = torch.randn(8)
    assert not torch.equal(a, b)


def test_pythonhashseed_is_set() -> None:
    set_global_seed(2024)
    assert os.environ["PYTHONHASHSEED"] == "2024"
