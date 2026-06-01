"""Dataset manifest helpers for the real-data benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml


REQUIRED_MISSION_KEYS = {
    "mission_id",
    "region",
    "year",
    "raw_path",
    "tensor_path",
}


@dataclass(frozen=True)
class MissionRecord:
    """Mission metadata resolved from the tracked dataset manifest."""

    mission_id: str
    region: str
    year: int
    raw_path: Path
    tensor_path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def load_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    """Load the dataset manifest and validate its top-level structure."""
    manifest_file = _resolve_repo_path(manifest_path)
    with open(manifest_file, "r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}

    missions = manifest.get("missions")
    if not isinstance(missions, list) or not missions:
        raise ValueError(f"Manifest '{manifest_file}' does not define any missions")
    return manifest


def mission_records(manifest_path: str | Path) -> List[MissionRecord]:
    """Return validated mission records with resolved filesystem paths."""
    manifest = load_manifest(manifest_path)
    records: List[MissionRecord] = []

    for entry in manifest["missions"]:
        missing = REQUIRED_MISSION_KEYS.difference(entry)
        if missing:
            raise ValueError(
                f"Manifest mission entry is missing required keys: {sorted(missing)}"
            )

        record = MissionRecord(
            mission_id=str(entry["mission_id"]),
            region=str(entry["region"]),
            year=int(entry["year"]),
            raw_path=_resolve_repo_path(entry["raw_path"]),
            tensor_path=_resolve_repo_path(entry["tensor_path"]),
        )
        records.append(record)

    return records


def manifest_by_mission(manifest_path: str | Path) -> Dict[str, MissionRecord]:
    """Return mission records keyed by mission id."""
    return {record.mission_id: record for record in mission_records(manifest_path)}


def sequence_index_path(manifest_path: str | Path) -> Path:
    """Resolve the canonical sequence index sidecar path."""
    manifest = load_manifest(manifest_path)
    path_like = manifest.get("sequence_index_path")
    if not path_like:
        raise ValueError("Manifest does not define sequence_index_path")
    return _resolve_repo_path(path_like)


def validate_manifest_files(
    manifest_path: str | Path,
    *,
    require_raw: bool = True,
    require_tensors: bool = True,
) -> None:
    """Raise ``ValueError`` when required manifest paths do not resolve to files."""
    for record in mission_records(manifest_path):
        checks = []
        if require_raw:
            checks.append(("raw_path", record.raw_path))
        if require_tensors:
            checks.append(("tensor_path", record.tensor_path))
        for label, path in checks:
            if not path.exists():
                raise ValueError(
                    f"Manifest entry '{record.mission_id}' points to missing {label}: {path}"
                )


def load_sequence_index(manifest_path: str | Path) -> pd.DataFrame:
    """Load the canonical processed-tensor traceability index."""
    index_path = sequence_index_path(manifest_path)
    if not index_path.exists():
        raise ValueError(f"Processed sequence index does not exist: {index_path}")
    index_df = pd.read_parquet(index_path)
    if "mission_id" not in index_df.columns:
        raise ValueError(f"Sequence index at {index_path} is missing the 'mission_id' column")
    return index_df


def sequence_counts_by_mission(manifest_path: str | Path) -> Dict[str, int]:
    """Return the indexed number of processed windows for each mission."""
    index_df = load_sequence_index(manifest_path)
    counts = index_df.groupby("mission_id").size().to_dict()
    return {str(mission_id): int(count) for mission_id, count in counts.items()}


def unique_regions(records: Iterable[MissionRecord]) -> List[str]:
    """Return sorted unique region labels."""
    return sorted({record.region for record in records})
