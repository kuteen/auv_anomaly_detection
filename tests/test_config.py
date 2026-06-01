"""Configuration loading and validation contract."""

from __future__ import annotations

import copy

import pytest

from config.schema import load_config, validate_config

from .conftest import DEFAULTS_CONFIG, N_SENSORS, WINDOW_LENGTH


@pytest.fixture
def cfg() -> dict:
    return load_config(str(DEFAULTS_CONFIG))


def test_defaults_validate(cfg: dict) -> None:
    # Should not raise.
    validate_config(cfg)


def test_expected_top_level_sections(cfg: dict) -> None:
    for section in ("data", "data_processing", "training", "evaluation", "output"):
        assert section in cfg, f"missing top-level section '{section}'"


def test_data_processing_key_is_preserved(cfg: dict) -> None:
    # The module is now 'data', but the config key 'data_processing' and the
    # merged data_processing.yaml contract must stay.
    proc = cfg["data_processing"]
    assert proc["normalisation"] == "minmax"
    assert proc["outlier_chunk_size"] == WINDOW_LENGTH


def test_sensor_contract(cfg: dict) -> None:
    sensors = cfg["data"]["sensors"]
    assert len(sensors) == N_SENSORS


def test_windowing_contract(cfg: dict) -> None:
    assert cfg["windowing"]["window_length"] == WINDOW_LENGTH
    assert cfg["windowing"]["stride"] >= 1


def test_output_base_dir_under_reports(cfg: dict) -> None:
    base_dir = cfg["output"]["base_dir"]
    assert base_dir.startswith("reports/"), base_dir
    assert not base_dir.startswith("data/outputs"), base_dir


def test_model_roster(cfg: dict) -> None:
    assert cfg["model"]["name"] == "graph_stgnn"


def test_validation_rejects_unknown_model(cfg: dict) -> None:
    broken = copy.deepcopy(cfg)
    broken["model"]["name"] = "not_a_model"
    with pytest.raises(ValueError):
        validate_config(broken)


def test_validation_rejects_unknown_fault(cfg: dict) -> None:
    broken = copy.deepcopy(cfg)
    broken["faults"]["types"] = ["teleport"]
    with pytest.raises(ValueError):
        validate_config(broken)
