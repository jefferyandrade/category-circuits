"""Canonical content addressing for scientific and runtime contracts.

This low-layer owner preserves the byte identities historically emitted by the
IRG contract module while making hashing available without importing a workflow
namespace.  Prefix strings remain schema identities; they do not grant workflow
authority.
"""

from __future__ import annotations

from collections.abc import Mapping as _AbstractMapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np

from contracts.runtime_receipts import recorded_runtime_hash_payload

from mechanistic_systems.core.axis_identity import ConstructiveOrdinalRouteAxis

# Arrays at or above this size are hashed on a worker thread when a value
# carries two or more of them.  hashlib releases the GIL for buffers larger
# than ~2 KiB, so per-array digests run genuinely in parallel; each digest is
# computed by the very same chunked routine as the sequential path, so the
# bytes fed to SHA-256 -- and therefore the digests -- are identical.
CONCURRENT_ARRAY_HASH_MIN_BYTES = 8 * 1024 * 1024
CONCURRENT_ARRAY_HASH_MAX_WORKERS = 8


def _array_sha256(
    array: np.ndarray,
    *,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> str:
    """Hash an array without materializing a second full-size byte string."""

    value = np.asarray(array)
    digest = hashlib.sha256()
    if value.size == 0:
        return digest.hexdigest()
    if value.flags.c_contiguous:
        view = memoryview(value).cast("B")
        for start in range(0, len(view), int(chunk_bytes)):
            digest.update(view[start : start + int(chunk_bytes)])
        return digest.hexdigest()
    if value.ndim == 0:
        digest.update(np.ascontiguousarray(value).view(np.uint8))
        return digest.hexdigest()
    rows_per_chunk = max(1, int(chunk_bytes) // max(1, int(value[0:1].nbytes)))
    for start in range(0, value.shape[0], rows_per_chunk):
        slab = np.ascontiguousarray(value[start : start + rows_per_chunk])
        digest.update(memoryview(slab).cast("B"))
    return digest.hexdigest()


def _concurrent_hash_worker_count(array_count: int) -> int:
    return max(1, min(CONCURRENT_ARRAY_HASH_MAX_WORKERS, os.cpu_count() or 1, array_count))


# One module-level hashing pool (grown on demand) instead of a new
# ThreadPoolExecutor per ``canonical_json`` call.  Scheduling only: every
# digest is still one ``_array_sha256`` call over the same array.
_HASH_POOL: ThreadPoolExecutor | None = None
_HASH_POOL_WORKERS = 0
_HASH_POOL_LOCK = threading.Lock()


def _hash_pool(workers: int) -> ThreadPoolExecutor:
    global _HASH_POOL, _HASH_POOL_WORKERS
    with _HASH_POOL_LOCK:
        if _HASH_POOL is None or _HASH_POOL_WORKERS < workers:
            # A smaller pool is dropped, never shut down: another thread may
            # still be mapping on it; its idle workers exit once it is freed.
            _HASH_POOL = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="content-address-sha256"
            )
            _HASH_POOL_WORKERS = workers
        return _HASH_POOL


def _reset_hash_pool_after_fork() -> None:  # pragma: no cover - fork hook
    global _HASH_POOL, _HASH_POOL_WORKERS, _HASH_POOL_LOCK
    _HASH_POOL_LOCK = threading.Lock()
    _HASH_POOL = None
    _HASH_POOL_WORKERS = 0


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_hash_pool_after_fork)


def _digest_arrays(ordered: list[np.ndarray]) -> list[str]:
    if len(ordered) < 2:
        return [_array_sha256(array) for array in ordered]
    workers = _concurrent_hash_worker_count(len(ordered))
    if workers <= 1:
        return [_array_sha256(array) for array in ordered]
    return list(_hash_pool(workers).map(_array_sha256, ordered))


def _collect_large_arrays(value: Any, sink: list[np.ndarray]) -> None:
    """Gather every ndarray leaf ``_canonicalize`` will digest, in its order."""

    if isinstance(value, ConstructiveOrdinalRouteAxis):
        return
    if is_dataclass(value):
        for item in fields(value):
            _collect_large_arrays(getattr(value, item.name), sink)
        return
    if isinstance(value, np.ndarray):
        if value.nbytes >= CONCURRENT_ARRAY_HASH_MIN_BYTES:
            sink.append(value)
        return
    if issubclass(type(value), _AbstractMapping):
        for key in value:
            _collect_large_arrays(value[key], sink)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _collect_large_arrays(item, sink)


def _array_digests_concurrently(value: Any) -> dict[int, tuple[np.ndarray, str]]:
    """Digest the large ndarray leaves of ``value`` on worker threads.

    Returns ``{id(array): (array, sha256)}``.  The mapping holds a STRONG
    reference to every array it digested, and the lookup in
    :func:`_canonicalize` re-checks object identity (``cached is value``)
    before using a cached digest.  Both halves matter: ``id()`` is only
    unique among live objects, so a cache that stored ids alone could return
    a digest computed for a DIFFERENT array whose address had since been
    recycled -- which is exactly what happened for any container that
    materializes its leaves on access.  Fewer than two large arrays takes the
    sequential path.
    """

    arrays: list[np.ndarray] = []
    _collect_large_arrays(value, arrays)
    unique: dict[int, np.ndarray] = {}
    for array in arrays:
        unique.setdefault(id(array), array)
    if len(unique) < 2:
        return {}
    ordered = list(unique.values())
    digests = _digest_arrays(ordered)
    return {
        id(array): (array, digest)
        for array, digest in zip(ordered, digests, strict=True)
    }


def _canonicalize(
    value: Any,
    array_digests: Mapping[int, tuple[np.ndarray, str]] | None = None,
    pending: list[tuple[np.ndarray, dict[str, Any]]] | None = None,
) -> Any:
    # Fast paths.  An exact builtin type reaches the same branch of the
    # chain below as without them (none of them is a route axis, dataclass,
    # Enum, Path, ndarray, numpy scalar or bytes), so it is dispatched first
    # instead of after nine isinstance checks against ABC-backed types.
    kind = type(value)
    if kind is str or kind is int or kind is bool or value is None:
        return value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("content-addressed values must be finite")
        return value
    if kind is dict:
        return {
            str(key): _canonicalize(value[key], array_digests, pending)
            for key in sorted(value, key=str)
        }
    if kind is list or kind is tuple:
        return [_canonicalize(item, array_digests, pending) for item in value]
    if isinstance(value, ConstructiveOrdinalRouteAxis):
        # The route:<ordinal> grammar is injective and fully determined by its
        # count.  Bind the exact ordered-axis digest without expanding tens of
        # millions of generated identifiers into the control-plane JSON.
        return {
            "schema": "constructive_ordinal_route_axis_ref_v1",
            "count": len(value),
            "ordered_digest": value.ordered_digest,
        }
    if is_dataclass(value):
        return {
            item.name: _canonicalize(getattr(value, item.name), array_digests, pending)
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        cached = None if array_digests is None else array_digests.get(id(value))
        # Identity, not address: a recycled ``id`` must miss, not lie.
        digest = cached[1] if cached is not None and cached[0] is value else None
        node: dict[str, Any] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": digest,
        }
        if digest is None:
            if pending is not None and value.nbytes >= CONCURRENT_ARRAY_HASH_MIN_BYTES:
                # Digested after the pass (``_fill_pending_digests``); the
                # list holds the array, so its ``id`` stays unique meanwhile.
                pending.append((value, node))
            else:
                node["sha256"] = _array_sha256(array)
        return node
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {
            "bytes_sha256": hashlib.sha256(value).hexdigest(),
            "length": len(value),
        }
    # ``isinstance(x, typing.Mapping)`` is exactly
    # ``issubclass(type(x), collections.abc.Mapping)``; the direct form skips
    # the typing alias dispatch.
    if issubclass(kind, _AbstractMapping):
        return {
            str(key): _canonicalize(value[key], array_digests, pending)
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        sequence = list(value)
        if isinstance(value, (set, frozenset)):
            # Sort keys need complete digests: no deferred leaves in sets.
            sequence = sorted(
                sequence,
                key=lambda item: json.dumps(
                    _canonicalize(item, array_digests), sort_keys=True
                ),
            )
            return [_canonicalize(item, array_digests) for item in sequence]
        return [_canonicalize(item, array_digests, pending) for item in sequence]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("content-addressed values must be finite")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported content-addressed value type: {type(value)!r}")


def _fill_pending_digests(pending: list[tuple[np.ndarray, dict[str, Any]]]) -> None:
    """Digest the deferred large arrays (concurrently when two or more)."""

    if not pending:
        return
    unique: dict[int, np.ndarray] = {}
    for array, _node in pending:
        unique.setdefault(id(array), array)
    ordered = list(unique.values())
    digests = dict(
        zip((id(array) for array in ordered), _digest_arrays(ordered), strict=True)
    )
    for array, node in pending:
        node["sha256"] = digests[id(array)]


def canonical_json(value: Any) -> str:
    """Return the historical stable JSON representation.

    Large ndarray leaves are digested concurrently; the emitted JSON is
    byte-identical to the sequential form because every array digest is
    produced by the same chunked ``_array_sha256``.  One pass
    canonicalizes the value and defers its large leaves (instead of a
    collection pass followed by a canonicalization pass).
    """

    pending: list[tuple[np.ndarray, dict[str, Any]]] = []
    canonical = _canonicalize(value, None, pending)
    _fill_pending_digests(pending)
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def content_hash(value: Any, *, prefix: str = "") -> str:
    """Return a SHA-256 identity over :func:`canonical_json`."""

    digest = hashlib.sha256(canonical_json(recorded_runtime_hash_payload(value)).encode("utf-8")).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


__all__ = ["canonical_json", "content_hash"]
