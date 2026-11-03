"""Cheap content hashing used by the CircuitFlow artifact boundary.

This module deliberately has no analysis-package dependency.  Operational
artifact identity must remain available even when no scientific analysis
extras (pandas, torch, zarr) are installed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from contracts.runtime_receipts import recorded_runtime_hash_payload

_CHUNK_BYTES = 1 << 20


def _canonical(value: Any) -> Any:
    # Exact builtin types first.  Each reaches the same branch
    # below as before (a str/int/bool/None/float has no ``item`` attribute
    # and is not a Path, Mapping, list or tuple), without the ABC checks.
    kind = type(value)
    if kind is str or kind is int or kind is bool or value is None:
        return value
    if kind is float:
        return None if (value != value or value in {float("inf"), float("-inf")}) else value
    if kind is dict:
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if kind is list or kind is tuple:
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    if hasattr(value, "item"):
        try:
            return _canonical(value.item())
        except (TypeError, ValueError):
            pass
    return value


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        _canonical(recorded_runtime_hash_payload(payload)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(root)
    members: list[dict[str, Any]] = []
    for member in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = member.relative_to(root).as_posix()
        if member.is_symlink():
            members.append({"path": relative, "kind": "symlink", "target": str(member.readlink())})
        elif member.is_file():
            members.append(
                {
                    "path": relative,
                    "kind": "file",
                    "bytes": int(member.stat().st_size),
                    "sha256": sha256_file(member),
                }
            )
        elif member.is_dir() and not any(member.iterdir()):
            members.append({"path": relative, "kind": "empty_directory"})
    return stable_digest({"tree": members})


def sha256_path(path: str | Path) -> str:
    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    if source.is_dir():
        return sha256_tree(source)
    raise FileNotFoundError(source)


def hash_score_store(store_root: str | Path) -> str:
    """Pin an exact ScoreStore without reading every shard on routine paths."""

    root = Path(store_root)
    manifest = root / "manifest.json"
    if manifest.is_file():
        return stable_digest({"score_store_manifest": sha256_file(manifest)})
    members = {member.relative_to(root).as_posix(): sha256_file(member) for member in sorted(root.rglob("*.npz"))}
    index = root / "index.json"
    if index.is_file():
        members["index.json"] = sha256_file(index)
    if not members:
        raise FileNotFoundError(f"no ScoreStore manifest or shards under {root}")
    return stable_digest({"score_store": members})


__all__ = ["hash_score_store", "sha256_file", "sha256_path", "sha256_tree", "stable_digest"]
