"""Data sub-package: manifest handling, preprocessing, faults, and graph builders."""

from data.manifest import (
    MissionRecord,
    load_manifest,
    load_sequence_index,
    manifest_by_mission,
    mission_records,
    sequence_counts_by_mission,
    sequence_index_path,
    unique_regions,
    validate_manifest_files,
)
from data.preprocessing import (
    accumulate_global_minmax,
    apply_global_minmax,
    convert_ddm_to_dd,
    interpolate_data,
    prepare_raw_dataframe,
    remove_outliers,
    synchronise_time,
    validate_cached_preprocessing_contract,
)
from data.faults import inject_fault, FAULT_REGISTRY
from data.graph_builders import (
    FixedGraphBuilder,
    CorrelationGraphBuilder,
    LearnedGraphBuilder,
)

__all__ = [
    "MissionRecord",
    "load_manifest",
    "load_sequence_index",
    "manifest_by_mission",
    "mission_records",
    "sequence_counts_by_mission",
    "sequence_index_path",
    "unique_regions",
    "validate_manifest_files",
    "accumulate_global_minmax",
    "apply_global_minmax",
    "convert_ddm_to_dd",
    "interpolate_data",
    "prepare_raw_dataframe",
    "remove_outliers",
    "synchronise_time",
    "validate_cached_preprocessing_contract",
    "inject_fault",
    "FAULT_REGISTRY",
    "FixedGraphBuilder",
    "CorrelationGraphBuilder",
    "LearnedGraphBuilder",
]
