"""Split builders for the pooled mission-agnostic benchmark."""

from __future__ import annotations

from typing import Any, Dict, List

from data.manifest import manifest_by_mission, mission_records

SplitSpec = Dict[str, Any]


def _manual_override(config: Dict[str, Any]) -> List[SplitSpec]:
    """Return an explicit split override when mission ids are provided in config."""
    split_cfg = config["splits"]
    train_ids = [str(mid) for mid in split_cfg.get("train_ids", [])]
    val_ids = [str(mid) for mid in split_cfg.get("val_ids", [])]
    test_ids = [str(mid) for mid in split_cfg.get("test_ids", [])]

    if not (train_ids or val_ids or test_ids):
        return []
    if not train_ids or not test_ids:
        raise ValueError(
            "Explicit split override requires both splits.train_ids and splits.test_ids"
        )

    return [
        {
            "name": "manual_split",
            "mode": "held_out_missions",
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        }
    ]


def _global_split(config: Dict[str, Any]) -> List[SplitSpec]:
    """Build one pooled mission-agnostic split across all missions."""
    records = mission_records(config["data"]["dataset_manifest"])
    mission_ids = sorted(record.mission_id for record in records)
    if len(mission_ids) < 2:
        raise ValueError("The global benchmark split requires at least two missions.")

    # Pooled mission-agnostic protocol. Every mission feeds both train and test,
    # so the window-level random split happens downstream rather than by mission.
    # The empty val_ids signal that the validation set is carved out later.
    return [
        {
            "name": "global_split",
            "mode": "global_random_split",
            "mission_ids": mission_ids,
            "train_ids": mission_ids,
            "val_ids": [],
            "test_ids": mission_ids,
        }
    ]


def build_protocol_splits(config: Dict[str, Any]) -> List[SplitSpec]:
    """Resolve the configured split settings into executable split specifications."""
    manual = _manual_override(config)
    splits = manual if manual else _global_split(config)

    manifest_map = manifest_by_mission(config["data"]["dataset_manifest"])
    for split in splits:
        for key in ("train_ids", "val_ids", "test_ids"):
            for mission_id in split.get(key, []):
                if mission_id not in manifest_map:
                    raise ValueError(
                        f"Split '{split['name']}' references unknown mission id '{mission_id}'"
                    )

    return splits
