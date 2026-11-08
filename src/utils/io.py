"""IO helpers: JSON / YAML convenience wrappers.

Every writer goes through :func:`_atomic_write_bytes`, so a saved artefact is
either the complete new file or the untouched old one, never a partial write.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
from typing import Any

import yaml


def _atomic_write_bytes(payload: bytes, path: pathlib.Path, *, mode: str = "wb") -> None:
    """Write bytes to a sibling temp file then atomically replace the target.

    The payload is written to a temporary file in the destination directory and
    swapped in with ``os.replace``. A reader therefore never observes a
    half-written file, and an interrupted write leaves the original intact. The
    temp file is removed on any failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file in the SAME directory so os.replace is atomic (a
    # cross-filesystem rename would not be).
    fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = pathlib.Path(tmp_path_str)
    try:
        with os.fdopen(fd, mode) as fh:
            fh.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_json(obj: Any, path: str | pathlib.Path) -> None:
    """Serialise an object to indented JSON and write it atomically.

    Values that are not JSON-serialisable fall back to their string form via
    ``default=str``, so a save never fails on an unexpected type.
    """
    path = pathlib.Path(path)
    payload = json.dumps(obj, indent=2, default=str).encode("utf-8")
    _atomic_write_bytes(payload, path)


def save_yaml(obj: Any, path: str | pathlib.Path) -> None:
    """Serialise an object to block-style YAML and write it atomically."""
    path = pathlib.Path(path)
    payload = yaml.dump(obj, default_flow_style=False).encode("utf-8")
    _atomic_write_bytes(payload, path)
