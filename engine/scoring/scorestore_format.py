"""Canonical exact ScoreStore schema, writer, reader, and URI contract.

Scoring owns this durable execution schema. Engine and mechanistic-measurement
paths remain exact compatibility facades and must not redefine it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from scoring.score_heads import ROUTE_ABS_SCORE_HEAD


SCORESTORE_SCHEMA_NAME = "exact_score_store"
SCORESTORE_SCHEMA_VERSION = "exact_score_store_v2"
SCORESTORE_URI_PREFIX = "scorestore:"
SCORESTORE_CANONICAL_LAYOUT = "cell_target_root_v1"


@dataclass(frozen=True)
class ScoreStoreWriteReport:
    store_root: Path
    manifest_path: Path
    index_path: Path
    shard_paths: tuple[Path, ...]
    target_count: int
    tensor_count: int
    archive_bytes: int

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_name": "exact_score_store_write_report",
            "schema_version": "exact_score_store_write_report_v1",
            "store_root": str(self.store_root),
            "manifest_path": str(self.manifest_path),
            "index_path": str(self.index_path),
            "shard_paths": [str(path) for path in self.shard_paths],
            "target_count": int(self.target_count),
            "tensor_count": int(self.tensor_count),
            "archive_bytes": int(self.archive_bytes),
        }


def write_exact_score_store(
    *,
    output_dir: str | Path,
    target_payloads: Mapping[str, Mapping[str, Any]],
    cell_id: str,
    target_family: str,
    route_schema: Mapping[str, Any] | None = None,
    source_score_config: Mapping[str, Any] | None = None,
    include_payload_keys: tuple[str, ...] = (ROUTE_ABS_SCORE_HEAD,),
    structural_sign_payload_key: str | None = None,
    aliases: Mapping[str, str] | None = None,
    compression: bool = True,
) -> ScoreStoreWriteReport:
    """Write exact score tensors into a chunked, lossless ScoreStore.

    The writer intentionally does not quantize, top-k truncate, or rank-only
    encode any included tensor.  Durable compression is limited to NumPy's
    lossless npz compression and metadata deduplication.
    """

    if not target_payloads:
        raise ValueError("target_payloads must contain at least one target.")
    if not include_payload_keys:
        raise ValueError("include_payload_keys must be non-empty.")

    root = Path(output_dir)
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    aliases_dict = {str(k): str(v) for k, v in dict(aliases or {}).items()}
    for alias, source in aliases_dict.items():
        if alias == source:
            raise ValueError(f"ScoreStore alias {alias!r} points to itself.")
        if source not in include_payload_keys and source != "structural_sign":
            raise ValueError(
                f"ScoreStore alias {alias!r} points to {source!r}, which is not stored exactly."
            )

    manifest_targets: list[str] = []
    index: dict[str, Any] = {
        "schema_name": "exact_score_store_index",
        "schema_version": "exact_score_store_index_v2",
        "cell_id": str(cell_id),
        "target_family": str(target_family),
        "targets": {},
        "aliases": aliases_dict,
    }
    manifest: dict[str, Any] = {
        "schema_name": SCORESTORE_SCHEMA_NAME,
        "schema_version": SCORESTORE_SCHEMA_VERSION,
        "canonical_layout": SCORESTORE_CANONICAL_LAYOUT,
        "cell_id": str(cell_id),
        "target_family": str(target_family),
        "target_ids": [],
        "route_schema": dict(route_schema or {}),
        "source_score_config": dict(source_score_config or {}),
        "include_payload_keys": list(include_payload_keys),
        "structural_sign_payload_key": structural_sign_payload_key,
        "aliases": aliases_dict,
        "canonical_exact": True,
        "forbidden_canonical_encodings": [
            "lossy_quantization",
            "top_k_truncation",
            "rank_only_storage",
            "sampled_route_downsampling",
            "approximate_reconstruction",
        ],
        "shards": [],
    }

    shard_paths: list[Path] = []
    tensor_count = 0
    for target_id, payload in target_payloads.items():
        target_key = str(target_id)
        manifest_targets.append(target_key)
        target_index: dict[str, Any] = {}
        shard_arrays: dict[str, np.ndarray] = {}
        shard_entries: list[dict[str, Any]] = []

        for payload_key in include_payload_keys:
            if payload_key not in payload:
                raise KeyError(
                    f"target {target_key!r} missing required payload key {payload_key!r}."
                )
            layer_map = payload[payload_key]
            if not isinstance(layer_map, Mapping):
                raise TypeError(
                    f"target {target_key!r} payload {payload_key!r} must be a layer mapping; "
                    f"got {type(layer_map).__name__}."
                )
            target_index.setdefault(payload_key, {})
            for layer_name, tensor_value in layer_map.items():
                layer_key = str(layer_name)
                tensor_key = _tensor_key(target_key, payload_key, layer_key)
                tensor_for_validation = (
                    tensor_value.detach()
                    if torch.is_tensor(tensor_value)
                    else torch.as_tensor(tensor_value)
                )
                if tensor_for_validation.is_floating_point() and not bool(
                    torch.isfinite(tensor_for_validation).all().item()
                ):
                    raise ValueError(
                        "ScoreStore refuses to write non-finite canonical score tensor: "
                        f"target={target_key!r}, payload={payload_key!r}, layer={layer_key!r}."
                    )
                array, tensor_meta = _tensor_to_exact_array(tensor_value)
                shard_arrays[tensor_key] = array
                entry = _entry_for_tensor(
                    payload_key=payload_key,
                    layer=layer_key,
                    shard_name=f"{_safe_name(target_key)}.npz",
                    tensor_key=tensor_key,
                    array=array,
                    tensor_meta=tensor_meta,
                )
                target_index[payload_key][layer_key] = entry
                shard_entries.append(entry)
                tensor_count += 1

        if structural_sign_payload_key is not None:
            if structural_sign_payload_key not in payload:
                raise KeyError(
                    f"target {target_key!r} missing structural sign payload key "
                    f"{structural_sign_payload_key!r}."
                )
            sign_map = payload[structural_sign_payload_key]
            if not isinstance(sign_map, Mapping):
                raise TypeError("structural sign payload must be a layer mapping.")
            target_index.setdefault("structural_sign", {})
            for layer_name, tensor_value in sign_map.items():
                layer_key = str(layer_name)
                tensor_key = _tensor_key(target_key, "structural_sign", layer_key)
                array = _as_sign_array(tensor_value)
                shard_arrays[tensor_key] = array
                entry = _entry_for_tensor(
                    payload_key="structural_sign",
                    layer=layer_key,
                    shard_name=f"{_safe_name(target_key)}.npz",
                    tensor_key=tensor_key,
                    array=array,
                    tensor_meta={
                        "tensor_dtype": "torch.int8",
                        "storage_dtype": str(array.dtype),
                        "storage_encoding": "int8_sign_codes",
                    },
                )
                target_index["structural_sign"][layer_key] = entry
                shard_entries.append(entry)
                tensor_count += 1

        shard_path = shard_dir / f"{_safe_name(target_key)}.npz"
        if compression:
            np.savez_compressed(shard_path, **shard_arrays)
        else:
            np.savez(shard_path, **shard_arrays)
        shard_hash = sha256_file(shard_path)
        shard_paths.append(shard_path)
        manifest["shards"].append(
            {
                "target_id": target_key,
                "path": str(shard_path.relative_to(root)),
                "sha256": shard_hash,
                "bytes": int(shard_path.stat().st_size),
                "tensor_count": len(shard_entries),
            }
        )
        index["targets"][target_key] = target_index

    manifest["target_ids"] = manifest_targets
    manifest["target_count"] = len(manifest_targets)
    manifest["tensor_count"] = int(tensor_count)
    manifest["archive_bytes"] = int(sum(path.stat().st_size for path in shard_paths))

    manifest_path = root / "manifest.json"
    index_path = root / "index.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return ScoreStoreWriteReport(
        store_root=root,
        manifest_path=manifest_path,
        index_path=index_path,
        shard_paths=tuple(shard_paths),
        target_count=len(manifest_targets),
        tensor_count=tensor_count,
        archive_bytes=int(sum(path.stat().st_size for path in shard_paths)),
    )


def canonical_target_scorestore_dir(
    scorestore_root: str | Path, target_id: str | int
) -> Path:
    """Return the canonical durable ScoreStore directory for one target.

    Production score roots are organized as ``scorestore_v2/<cell>/<target_id>``.
    Each target directory is a complete exact ScoreStore containing its own
    ``manifest.json``, ``index.json``, and losslessly-compressed shards.  Legacy
    or compatibility views may exist elsewhere, but active consumers should use
    this layout whenever a cell-level scorestore root is available.
    """

    return Path(scorestore_root) / str(target_id)


def is_canonical_target_scorestore_dir(path: str | Path) -> bool:
    root = Path(path)
    if not (root / "manifest.json").exists() or not (root / "index.json").exists():
        return False
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except Exception:
        return False
    return (
        manifest.get("schema_version") == SCORESTORE_SCHEMA_VERSION
        and manifest.get("canonical_exact") is True
    )


def iter_canonical_target_scorestores(
    cell_scorestore_root: str | Path,
) -> Iterable[Path]:
    """Yield canonical target stores directly below one cell ScoreStore root."""

    root = Path(cell_scorestore_root)
    if not root.exists():
        return
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.is_dir() and is_canonical_target_scorestore_dir(child):
            yield child


def load_scorestore_tensor(
    reference: str | Path,
    *,
    base_path: str | Path | None = None,
    verify: bool = True,
) -> torch.Tensor:
    """Load one exact tensor from a ScoreStore reference."""

    parsed = parse_scorestore_reference(str(reference), base_path=base_path)
    root = parsed["store_root"]
    target_id = parsed["target"]
    payload_key = parsed["payload"]
    layer = parsed["layer"]

    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    aliases = index.get("aliases", {}) or {}
    payload_key = str(aliases.get(payload_key, payload_key))
    try:
        entry = index["targets"][target_id][payload_key][layer]
    except KeyError as exc:
        raise KeyError(
            f"ScoreStore tensor not found: target={target_id!r}, payload={payload_key!r}, "
            f"layer={layer!r}, store={root}"
        ) from exc

    shard_path = root / "shards" / entry["shard"]
    if not shard_path.exists():
        # Older manifests may store a path relative to the store root.
        shard_path = root / entry["shard"]
    if verify:
        actual_shard_hash = sha256_file(shard_path)
        shard_hash = _shard_hash_for(index_root=root, shard_path=shard_path)
        if shard_hash is not None and actual_shard_hash != shard_hash:
            raise ValueError(f"ScoreStore shard hash mismatch for {shard_path}.")

    with np.load(shard_path) as data:
        array = np.asarray(data[entry["tensor_key"]])
    if verify and sha256_array(array) != entry["sha256"]:
        raise ValueError(
            f"ScoreStore tensor hash mismatch for target={target_id!r}, "
            f"payload={payload_key!r}, layer={layer!r}."
        )
    return _exact_array_to_tensor(array, entry)


def load_scorestore_target_payload(
    reference: str | Path,
    *,
    target: str | int | None = None,
    payload: str | None = None,
    layers: Iterable[str] | None = None,
    base_path: str | Path | None = None,
    verify: bool = True,
) -> dict[str, torch.Tensor]:
    """Load all requested layers for one target/payload from a shard once.

    This is the production-friendly path for pruning: exact ScoreStore v2 writes
    one shard per target, so repeated one-layer loads would repeatedly decompress
    the same archive.  This helper verifies the shard once and then converts the
    requested tensors without changing dtype, values, or route shape.
    """

    parsed = parse_scorestore_locator(str(reference), base_path=base_path)
    root = parsed["store_root"]
    target_id = str(target if target is not None else parsed.get("target", ""))
    payload_key = str(payload if payload is not None else parsed.get("payload", ""))
    if not target_id:
        raise ValueError("ScoreStore target payload load requires a target id.")
    if not payload_key:
        raise ValueError("ScoreStore target payload load requires a payload key.")

    index = json.loads((root / "index.json").read_text())
    aliases = index.get("aliases", {}) or {}
    payload_key = str(aliases.get(payload_key, payload_key))
    try:
        layer_entries = index["targets"][target_id][payload_key]
    except KeyError as exc:
        raise KeyError(
            f"ScoreStore payload not found: target={target_id!r}, payload={payload_key!r}, store={root}"
        ) from exc

    requested_layers = (
        [str(layer) for layer in layers] if layers is not None else list(layer_entries)
    )
    missing_layers = [layer for layer in requested_layers if layer not in layer_entries]
    if missing_layers:
        raise KeyError(
            f"ScoreStore payload missing layers for target={target_id!r}, payload={payload_key!r}: "
            f"{missing_layers[:8]}" + ("..." if len(missing_layers) > 8 else "")
        )

    shard_to_layers: dict[Path, list[str]] = {}
    for layer in requested_layers:
        entry = layer_entries[layer]
        shard_path = root / "shards" / entry["shard"]
        if not shard_path.exists():
            shard_path = root / entry["shard"]
        shard_to_layers.setdefault(shard_path, []).append(layer)

    loaded: dict[str, torch.Tensor] = {}
    for shard_path, shard_layers in shard_to_layers.items():
        if verify:
            actual_shard_hash = sha256_file(shard_path)
            shard_hash = _shard_hash_for(index_root=root, shard_path=shard_path)
            if shard_hash is not None and actual_shard_hash != shard_hash:
                raise ValueError(f"ScoreStore shard hash mismatch for {shard_path}.")
        with np.load(shard_path) as data:
            for layer in shard_layers:
                entry = layer_entries[layer]
                array = np.asarray(data[entry["tensor_key"]])
                if verify and sha256_array(array) != entry["sha256"]:
                    raise ValueError(
                        f"ScoreStore tensor hash mismatch for target={target_id!r}, "
                        f"payload={payload_key!r}, layer={layer!r}."
                    )
                loaded[layer] = _exact_array_to_tensor(array, entry)
    return loaded


def parse_scorestore_reference(
    reference: str,
    *,
    base_path: str | Path | None = None,
) -> dict[str, Any]:
    parsed = parse_scorestore_locator(reference, base_path=base_path)
    missing = {"target", "payload", "layer"} - set(parsed)
    if missing:
        raise ValueError(
            f"ScoreStore reference missing query fields: {sorted(missing)}"
        )
    return parsed


def parse_scorestore_locator(
    reference: str,
    *,
    base_path: str | Path | None = None,
) -> dict[str, Any]:
    if not str(reference).startswith(SCORESTORE_URI_PREFIX):
        path_text = str(reference)
        params: dict[str, str] = {}
    else:
        body = str(reference)[len(SCORESTORE_URI_PREFIX) :]
        parsed = urlparse(body)
        if parsed.scheme or parsed.netloc:
            # Support references like scorestore:///abs/path/manifest.json?...
            path_text = parsed.path
            query = parsed.query
        else:
            path_text, _, query = body.partition("?")
        params = {
            key: values[-1]
            for key, values in parse_qs(query, keep_blank_values=True).items()
        }
    path = Path(path_text)
    if not path.is_absolute():
        root = Path(base_path) if base_path is not None else Path.cwd()
        path = (root / path).resolve()
    if path.name == "manifest.json":
        store_root = path.parent
    else:
        store_root = path
    result: dict[str, Any] = {
        "store_root": store_root,
    }
    for key in ("target", "payload", "layer"):
        if key in params:
            result[key] = str(params[key])
    return result


def scorestore_reference_exists(
    reference: str | Path, *, base_path: str | Path | None = None
) -> bool:
    try:
        parsed = parse_scorestore_reference(str(reference), base_path=base_path)
        root = parsed["store_root"]
        if not (root / "manifest.json").exists() or not (root / "index.json").exists():
            return False
        index = json.loads((root / "index.json").read_text())
        payload = str(
            (index.get("aliases") or {}).get(parsed["payload"], parsed["payload"])
        )
        return parsed["layer"] in index.get("targets", {}).get(
            parsed["target"], {}
        ).get(payload, {})
    except Exception:
        return False


def is_scorestore_reference(value: str | Path) -> bool:
    return str(value).startswith(SCORESTORE_URI_PREFIX)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(contiguous.dtype).encode("utf-8"))
    h.update(json.dumps(list(contiguous.shape)).encode("utf-8"))
    h.update(contiguous.tobytes(order="C"))
    return h.hexdigest()


def _tensor_to_exact_array(value: Any) -> tuple[np.ndarray, dict[str, str]]:
    tensor = (
        value.detach().cpu().contiguous()
        if torch.is_tensor(value)
        else torch.as_tensor(value).cpu().contiguous()
    )
    tensor_dtype = str(tensor.dtype)
    if tensor.dtype is torch.bfloat16:
        array = tensor.view(torch.uint16).numpy()
        return np.ascontiguousarray(array), {
            "tensor_dtype": tensor_dtype,
            "storage_dtype": str(array.dtype),
            "storage_encoding": "torch_bfloat16_raw_uint16",
        }
    array = tensor.numpy()
    return np.ascontiguousarray(array), {
        "tensor_dtype": tensor_dtype,
        "storage_dtype": str(array.dtype),
        "storage_encoding": "numpy_exact",
    }


def _exact_array_to_tensor(array: np.ndarray, entry: Mapping[str, Any]) -> torch.Tensor:
    tensor_dtype = str(entry.get("tensor_dtype", ""))
    if entry.get("storage_encoding") == "torch_bfloat16_raw_uint16":
        return torch.from_numpy(
            np.ascontiguousarray(array.astype(np.uint16, copy=False))
        ).view(torch.bfloat16)
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    expected = _torch_dtype_from_string(tensor_dtype)
    if expected is not None and tensor.dtype != expected:
        tensor = tensor.to(expected)
    return tensor


def _torch_dtype_from_string(value: str) -> torch.dtype | None:
    if not value.startswith("torch."):
        return None
    return getattr(torch, value.split(".", 1)[1], None)


def _as_sign_array(value: Any) -> np.ndarray:
    tensor = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
    sign = torch.sign(tensor.to(dtype=torch.float32)).to(dtype=torch.int8)
    return np.ascontiguousarray(sign.numpy())


def _entry_for_tensor(
    *,
    payload_key: str,
    layer: str,
    shard_name: str,
    tensor_key: str,
    array: np.ndarray,
    tensor_meta: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "payload_key": str(payload_key),
        "layer": str(layer),
        "shard": str(shard_name),
        "tensor_key": str(tensor_key),
        "shape": [int(dim) for dim in array.shape],
        "storage_dtype": str(array.dtype),
        "sha256": sha256_array(array),
        "uncompressed_bytes": int(np.ascontiguousarray(array).nbytes),
        **dict(tensor_meta),
    }


def _shard_hash_for(*, index_root: Path, shard_path: Path) -> str | None:
    manifest_path = index_root / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    for shard in manifest.get("shards", []):
        candidate = index_root / str(shard.get("path", ""))
        if candidate.resolve() == shard_path.resolve():
            return str(shard.get("sha256"))
    return None


def _tensor_key(target_id: str, payload_key: str, layer: str) -> str:
    return "__".join(_safe_name(part) for part in (target_id, payload_key, layer))


def _safe_name(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text or "unnamed"


__all__ = [
    "SCORESTORE_CANONICAL_LAYOUT",
    "SCORESTORE_SCHEMA_NAME",
    "SCORESTORE_SCHEMA_VERSION",
    "SCORESTORE_URI_PREFIX",
    "ScoreStoreWriteReport",
    "canonical_target_scorestore_dir",
    "is_canonical_target_scorestore_dir",
    "is_scorestore_reference",
    "iter_canonical_target_scorestores",
    "load_scorestore_target_payload",
    "load_scorestore_tensor",
    "parse_scorestore_locator",
    "parse_scorestore_reference",
    "scorestore_reference_exists",
    "sha256_array",
    "sha256_file",
    "write_exact_score_store",
]
