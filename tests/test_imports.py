"""Import guard: every public src subpackage and module must import.

This catches rename and path regressions after the flat-``src`` refactor
(for example ``data_processing`` -> ``data``).
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    # config
    "config",
    "config.schema",
    # data (was data_processing)
    "data",
    "data.faults",
    "data.graph_builders",
    "data.graph_topology",
    "data.manifest",
    "data.preprocessing",
    # models
    "models",
    "models.base",
    "models.baselines_classical",
    "models.baselines_deep",
    "models.losses",
    # training
    "training",
    "training.calibration",
    "training.early_stopping",
    "training.train_loop",
    # evaluation
    "evaluation",
    "evaluation.metrics",
    "evaluation.protocols",
    "evaluation.reporting",
    "evaluation.thresholding",
    # utils
    "utils",
    "utils.io",
    "utils.logging",
    "utils.runtime",
    "utils.seeds",
    "utils.stats",
    "utils.terminal",
    "utils.time",
    # version
    "version",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


def test_old_data_processing_package_is_gone() -> None:
    """The legacy ``data_processing`` package must no longer be importable."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("data_processing")


def test_version_string() -> None:
    import version

    assert version.__version__ == "1.0.0"
