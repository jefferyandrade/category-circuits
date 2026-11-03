"""Compact kind-3 walk products (decision, mask-delta, removal logs).

Production drops the pure-audit per-decision provenance
(outcome-integrity hashes, JSON decision receipts,
connection-delta receipts, progress payloads) and keeps the data products as
compact binary written asynchronously, plus ONE end-of-run integrity hash.
This module owns those products for the search:

* ``DecisionRecord`` rows (numpy structured, ~140 B each) and their per-look
  ``(target, delivered, mean, lower, upper)`` rows;
* the ``MaskDeltaLog``: one row per accepted (layer, partition) slice,
  ``(k_old, k_new, removed_count)``, from which every accepted mask is
  reproducible with the group's stable orders;
* the per-step REMOVAL LOG: the exact removed layer-local route indices of
  every accepted step, delta-encoded ``<u4`` and compressed at the walk tail.

The join key of every record is ``(query_id, layer_id, partition_id,
proposal_index)``; ``decision_key`` renders it as a string.
Nothing here feeds a stream id, a draw, a fold or a decision.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import threading
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from mechanistic_systems.measurement.run_products import effective_settings

V6_WALK_LOGS_SCHEMA = "irg_v6_walk_logs_v2"
V6_DECISION_RECORD_SCHEMA = "irg_v6_decision_record_v2"
V6_DECISION_REF_SCHEMA = "irg_v6_decision_record_ref_v1"
V6_REMOVAL_LOG_SCHEMA = "irg_v6_removal_log_v1"
V6_LEDGER_SCHEMA = "bce_precision_decision_ledger_v6_1"
V6_PANEL_RECEIPT_SCHEMA = "bce_oracle_sampled_panel_timing_receipt_v6_1"
V6_FORCED_ACTIONS_ENV = "IRG_V6_FORCED_ACTIONS"
_REMOVAL_MAGIC = b"IRGV6RL1"

ACTION_CODES = {
    "accept": 0,
    "reject": 1,
    "conservative_no_commit": 2,
    "unresolved": 3,
    "no_commit": 4,
}
PARTITION_CODES = {"positive": 0, "negative": 1}
FLAG_SUPPRESSED = 1
FLAG_FORCED = 2
FLAG_ACCEPT_REPLICATION = 4

DECISION_RECORD_DTYPE = np.dtype(
    [
        ("query_code", "<i4"),
        ("layer_code", "<i2"),
        ("partition_code", "i1"),
        ("action_code", "i1"),
        ("proposal_index", "<i4"),
        ("reason_code", "<i2"),
        ("looks", "<i2"),
        ("natural_action_code", "i1"),
        ("engine_reason_code", "<i2"),
        ("engine_forecast_target", "<i8"),
        ("flags", "u1"),
        ("panel_images", "<i8"),
        ("removed_count", "<i8"),
        ("k_old", "<i8"),
        ("k_new", "<i8"),
        ("previous_density", "<f8"),
        ("proposed_density", "<f8"),
        ("mean", "<f8"),
        ("lower", "<f8"),
        ("upper", "<f8"),
        ("threshold", "<f8"),
        ("alpha_allocated", "<f8"),
        ("candidate_mask_digest", "V32"),
    ]
)
DECISION_LOOK_DTYPE = np.dtype(
    [
        ("decision_ordinal", "<i4"),
        ("look_index", "<i2"),
        ("target_images", "<i8"),
        ("delivered_images", "<i8"),
        ("mean", "<f8"),
        ("lower", "<f8"),
        ("upper", "<f8"),
    ]
)
MASK_DELTA_DTYPE = np.dtype(
    [
        ("step_ordinal", "<i4"),
        ("decision_ordinal", "<i4"),
        ("query_code", "<i4"),
        ("layer_code", "<i2"),
        ("partition_code", "i1"),
        ("proposal_index", "<i4"),
        ("k_old", "<i8"),
        ("k_new", "<i8"),
        ("removed_count", "<i8"),
    ]
)
REMOVAL_STEP_DTYPE = np.dtype(
    MASK_DELTA_DTYPE.descr + [("index_offset", "<i8")]
)


def decision_key(
    query_id: str, layer_id: str, partition_id: str, proposal_index: int
) -> str:
    return f"{query_id}:{layer_id}:{partition_id}:{int(proposal_index)}"


def decision_record_hash(key: str) -> str:
    """The v6 value of ``decision_receipt_hash``: the decision-record join key."""

    return f"v6-decision-record:{key}"


# ---------------------------------------------------------------------------
# Codec.  zstandard when importable (the cluster env has 0.23.0), else zlib.
# ---------------------------------------------------------------------------


def _codec() -> str:
    forced = os.environ.get("IRG_V6_SA_CODEC", "").strip().lower()
    if forced in {"zlib", "raw"}:
        return forced
    import importlib.util

    return "zstd" if importlib.util.find_spec("zstandard") is not None else "zlib"


def _compressor(codec: str) -> Any:
    if codec == "zstd":
        import zstandard

        return zstandard.ZstdCompressor(level=3).compressobj()
    if codec == "zlib":
        return zlib.compressobj(1)
    return None


def _compress(codec: str, data: bytes) -> bytes:
    if codec == "raw":
        return bytes(data)
    stream = _compressor(codec)
    return stream.compress(data) + stream.flush()


def _decompress(codec: str, data: bytes) -> bytes:
    if codec == "raw":
        return bytes(data)
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "zstd":
        import zstandard

        return zstandard.ZstdDecompressor().decompressobj().decompress(data)
    raise ValueError(f"unknown v6 log codec {codec!r}")


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _npy_load(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def _suffix(codec: str) -> str:
    return {"zstd": ".npy.zst", "zlib": ".npy.zlib", "raw": ".npy"}[codec]


# ---------------------------------------------------------------------------
# Per-trace log (one route group), filled on the walk's critical path with
# plain tuples; removal arrays are delta-encoded and compressed off-path.
# ---------------------------------------------------------------------------


def delta_encode_u32(sorted_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(sorted_indices, dtype=np.int64).reshape(-1)
    if values.size and (int(values[0]) < 0 or int(values[-1]) > 0xFFFFFFFF):
        raise ValueError("removal-log indices exceed the uint32 layer axis")
    encoded = np.empty(values.size, dtype="<u4")
    if values.size:
        encoded[0] = values[0]
        encoded[1:] = np.diff(values)
    return encoded


def delta_decode_u32(encoded: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(encoded, dtype=np.int64))


class TraceWalkLog:
    """The kind-3 products of one ``trace_bce_precision_frontier`` call."""

    def __init__(self, *, query_id: str, layer_id: str) -> None:
        self.query_id = str(query_id)
        self.layer_id = str(layer_id)
        self.codec = _codec()
        self.records: list[dict[str, Any]] = []
        self.looks: list[tuple[int, int, int, int, float, float, float]] = []
        self.deltas: list[tuple[int, str, str, int, int, int, int]] = []
        self.accepted_mask_hashes: list[str] = []
        self._removed: list[np.ndarray] = []
        self._removal_index_counts: list[int] = []
        self.final_mask_hashes: dict[str, str] = {}
        self._closed_snapshot: Mapping[str, Any] | None = None

    def add_decision(self, record: Mapping[str, Any]) -> int:
        self.records.append(dict(record))
        ordinal = len(self.records) - 1
        for look in record.get("per_look", ()):
            self.looks.append((ordinal, *look))
        return ordinal

    def add_accept(self, *, decision_ordinal: int, view: Any) -> None:
        removed = view.removed_local_by_layer()
        for delta in view.mask_deltas():
            if delta.removed_count <= 0:
                continue
            self.deltas.append(
                (
                    int(decision_ordinal),
                    str(delta.layer),
                    str(delta.partition),
                    int(self.records[decision_ordinal]["proposal_index"]),
                    int(delta.k_old),
                    int(delta.k_new),
                    int(delta.removed_count),
                )
            )
            values = removed[delta.layer]
            self._removal_index_counts.append(int(values.size))
            self._removed.append(values)

    def close(self) -> dict[str, Any]:
        if self._closed_snapshot is not None:
            return _copy_log(self._closed_snapshot)
        if not self._removed:
            blob, raw_length, raw_sha = _compress(self.codec, b""), 0, hashlib.sha256().hexdigest()
        else:
            stream = _compressor(self.codec)
            chunks = []
            raw = hashlib.sha256()
            raw_length = 0
            for values in self._removed:
                data = delta_encode_u32(values).tobytes()
                raw.update(data)
                raw_length += len(data)
                chunks.append(data if stream is None else stream.compress(data))
            if stream is not None:
                chunks.append(stream.flush())
            blob, raw_sha = b"".join(chunks), raw.hexdigest()
            self._removed.clear()
        closed = {
            "query_id": self.query_id,
            "layer_id": self.layer_id,
            "codec": self.codec,
            "records": tuple(self.records),
            "looks": tuple(self.looks),
            "deltas": tuple(self.deltas),
            "removal_index_counts": tuple(self._removal_index_counts),
            "removal_blob": blob,
            "removal_raw_length": int(raw_length),
            "removal_raw_sha256": raw_sha,
            "final_mask_hashes": dict(self.final_mask_hashes),
        }
        self._closed_snapshot = _freeze_log(closed)
        return _copy_log(self._closed_snapshot)


def _freeze_log(value: Any) -> Any:
    """Snapshot a closed log recursively before it crosses a public boundary."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_log(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_log(item) for item in value)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_log(item) for item in value.tolist())
    return value


def _copy_log(value: Any) -> Any:
    """Return detached containers while retaining the closed log's tuple schema."""

    if isinstance(value, Mapping):
        return {key: _copy_log(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_log(item) for item in value)
    return value


class TraceWalkLogHandle:
    """The ledger's handle on a trace log (closed lazily at the walk tail)."""

    def __init__(self, log: TraceWalkLog) -> None:
        self._log = log
        self._closed: Mapping[str, Any] | None = None
        self.decision_count = len(log.records)
        self.accepted_step_count = len(log._removal_index_counts)

    def result(self) -> dict[str, Any]:
        if self._closed is None:
            self._closed = _freeze_log(self._log.close())
        return _copy_log(self._closed)

    def __repr__(self) -> str:  # the ledger is canonicalized by test dumps
        return (
            f"TraceWalkLogHandle(decisions={self.decision_count}, "
            f"accepted_steps={self.accepted_step_count})"
        )


# ---------------------------------------------------------------------------
# Category-level products: tables, arrays, sidecars, integrity hash.
# ---------------------------------------------------------------------------


def _mask_digest_bytes(mask_hash: str | None) -> bytes:
    if not mask_hash:
        return bytes(32)
    tail = str(mask_hash)[-64:]
    try:
        return bytes.fromhex(tail)
    except ValueError:
        return hashlib.sha256(str(mask_hash).encode("utf-8")).digest()


def _table_code(table: dict[str, int], value: str) -> int:
    code = table.get(value)
    if code is None:
        code = len(table)
        table[value] = code
    return code


def build_walk_arrays(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Encode closed trace logs into the compact category arrays."""

    queries: dict[str, int] = {}
    layers: dict[str, int] = {}
    reasons: dict[str, int] = {}
    records: list[tuple] = []
    looks: list[tuple] = []
    deltas: list[tuple] = []
    removal_steps: list[tuple] = []
    offset = 0
    step = 0
    base = 0
    for result in results:
        query_code = _table_code(queries, str(result["query_id"]))
        for record in result["records"]:
            records.append(
                (
                    query_code,
                    _table_code(layers, str(record["layer_id"])),
                    PARTITION_CODES[str(record["partition_id"])],
                    ACTION_CODES[str(record["action"])],
                    int(record["proposal_index"]),
                    _table_code(reasons, str(record["reason"])),
                    int(record["looks"]),
                    ACTION_CODES[str(record.get("natural_action", record["action"]))],
                    _table_code(reasons, str(record.get("engine_reason", record["reason"]))),
                    -1 if record.get("engine_forecast_target") is None else int(record["engine_forecast_target"]),
                    int(record.get("flags", 0)),
                    int(record["panel_images"]),
                    int(record["removed_count"]),
                    int(record.get("k_old", -1)),
                    int(record.get("k_new", -1)),
                    float(record["previous_density"]),
                    float(record["proposed_density"]),
                    float(record["mean"]),
                    float(record["lower"]),
                    float(record["upper"]),
                    float(record["threshold"]),
                    float(record["alpha_allocated"]),
                    _mask_digest_bytes(record.get("candidate_mask_hash")),
                )
            )
        for ordinal, *rest in result["looks"]:
            looks.append((base + int(ordinal), *rest))
        counts = result["removal_index_counts"]
        for index, (ordinal, layer, partition, proposal, k_old, k_new, removed) in enumerate(
            result["deltas"]
        ):
            row = (
                step,
                base + int(ordinal),
                query_code,
                _table_code(layers, str(layer)),
                PARTITION_CODES[str(partition)],
                int(proposal),
                int(k_old),
                int(k_new),
                int(removed),
            )
            deltas.append(row)
            removal_steps.append((*row, offset))
            offset += int(counts[index])
            step += 1
        base += len(result["records"])
    return {
        "tables": {
            "queries": sorted(queries, key=queries.get),
            "layers": sorted(layers, key=layers.get),
            "reasons": sorted(reasons, key=reasons.get),
            "actions": sorted(ACTION_CODES, key=ACTION_CODES.get),
            "partitions": sorted(PARTITION_CODES, key=PARTITION_CODES.get),
        },
        "decision_records": np.asarray(records, dtype=DECISION_RECORD_DTYPE),
        "decision_looks": np.asarray(looks, dtype=DECISION_LOOK_DTYPE),
        "mask_deltas": np.asarray(deltas, dtype=MASK_DELTA_DTYPE),
        "removal_steps": np.asarray(removal_steps, dtype=REMOVAL_STEP_DTYPE),
        "removal_index_count": int(offset),
    }


def _removal_log_bytes(
    *,
    codec: str,
    steps: np.ndarray,
    tables: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    total_indices: int,
) -> bytes:
    steps_blob = _compress(codec, _npy_bytes(steps))
    member_table = []
    cursor = len(steps_blob)
    for member in members:
        member_table.append(
            {
                "offset": int(cursor),
                "length": len(member["blob"]),
                "codec": str(member["codec"]),
                "raw_length": int(member["raw_length"]),
                "raw_sha256": str(member["raw_sha256"]),
            }
        )
        cursor += len(member["blob"])
    header = json.dumps(
        {
            "schema": V6_REMOVAL_LOG_SCHEMA,
            "index_space": "layer_local_flat_route_index",
            "index_encoding": "per_slice_delta_uint32_le",
            "steps_codec": codec,
            "steps_length": len(steps_blob),
            "steps_dtype": REMOVAL_STEP_DTYPE.descr,
            "tables": dict(tables),
            "members": member_table,
            "total_indices": int(total_indices),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return b"".join(
        [
            _REMOVAL_MAGIC,
            struct.pack("<Q", len(header)),
            header,
            steps_blob,
            *(member["blob"] for member in members),
        ]
    )


def read_removal_log_for_tests(data: bytes) -> tuple[dict[str, Any], np.ndarray, list[np.ndarray]]:
    """Decode a removal log, including containers written by earlier releases."""

    if data[: len(_REMOVAL_MAGIC)] != _REMOVAL_MAGIC:
        raise ValueError("not a v6 removal log")
    (length,) = struct.unpack("<Q", data[8:16])
    header = json.loads(data[16 : 16 + length].decode("utf-8"))
    body = data[16 + length :]
    steps = _npy_load(_decompress(header["steps_codec"], body[: header["steps_length"]]))
    flat_parts = []
    for member in header["members"]:
        raw = _decompress(
            member["codec"], body[member["offset"] : member["offset"] + member["length"]]
        )
        if len(raw) != member["raw_length"] or hashlib.sha256(raw).hexdigest() != member["raw_sha256"]:
            raise ValueError("v6 removal-log member failed its integrity check")
        flat_parts.append(np.frombuffer(raw, dtype="<u4"))
    flat = np.concatenate(flat_parts) if flat_parts else np.empty(0, dtype="<u4")
    removed: list[np.ndarray] = []
    for index, row in enumerate(steps):
        stop = (
            int(steps[index + 1]["index_offset"])
            if index + 1 < steps.size
            else int(header["total_indices"])
        )
        removed.append(delta_decode_u32(flat[int(row["index_offset"]) : stop]))
    return header, steps, removed


class V6WalkProducts(dict):
    """The category's walk logs: a JSON-safe descriptor plus sidecar bytes.

    The ``dict`` content is the descriptor.  It is fixed at construction, so
    it can sit in ``workflow_details`` and be hashed (``content_hash``) or
    serialized at any time with the same result.  The sidecar bytes are
    produced on a writer thread started at construction, overlapped with the
    terminal phase; ``sidecars()`` joins it and returns ``{name: bytes}`` for
    the receipt writer to persist next to ``receipt.json``, and
    ``manifest()`` returns the manifest (file digests, record counts and the
    ONE end-of-run integrity hash) for the receipt.  Both fail closed.
    """

    def __init__(self, results: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(
            {
                "schema": V6_WALK_LOGS_SCHEMA,
                "decision_count": int(sum(len(r["records"]) for r in results)),
                "accepted_step_count": int(sum(len(r["deltas"]) for r in results)),
                "join_key": ("query_id", "layer_id", "partition_id", "proposal_index"),
                "manifest_file": "v6_walk_logs.json",
            }
        )
        # The writer and arrays() consume only this one immutable handoff.
        self._results = tuple(_freeze_log(result) for result in results)
        self._sidecars: dict[str, bytes] | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._serialize, name="irg-v6-walk-logs", daemon=True
        )
        self._thread.start()

    def _reject_mutation(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("v6 walk-products descriptor is immutable")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation
    __ior__ = _reject_mutation

    def _serialize(self) -> None:
        try:
            codec = _codec()
            arrays = build_walk_arrays(self._results)
            names = {
                "decision_records": "v6_decision_log" + _suffix(codec),
                "decision_looks": "v6_decision_looks" + _suffix(codec),
                "mask_deltas": "v6_mask_delta_log" + _suffix(codec),
            }
            sidecars = {
                names[key]: _compress(codec, _npy_bytes(arrays[key]))
                for key in names
            }
            removal_name = "v6_removal_log.bin"
            sidecars[removal_name] = _removal_log_bytes(
                codec=codec,
                steps=arrays["removal_steps"],
                tables=arrays["tables"],
                members=[
                    {
                        "blob": result["removal_blob"],
                        "codec": result["codec"],
                        "raw_length": result["removal_raw_length"],
                        "raw_sha256": result["removal_raw_sha256"],
                    }
                    for result in self._results
                ],
                total_indices=arrays["removal_index_count"],
            )
            files = {
                name: {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
                for name, blob in sorted(sidecars.items())
            }
            manifest = {
                "schema": V6_WALK_LOGS_SCHEMA,
                "effective_settings": effective_settings(),
                "codec": codec,
                "tables": arrays["tables"],
                "dtypes": {
                    "decision_records": DECISION_RECORD_DTYPE.descr,
                    "decision_looks": DECISION_LOOK_DTYPE.descr,
                    "mask_deltas": MASK_DELTA_DTYPE.descr,
                },
                "join_key": ["query_id", "layer_id", "partition_id", "proposal_index"],
                "record_counts": {
                    "decisions": int(arrays["decision_records"].size),
                    "looks": int(arrays["decision_looks"].size),
                    "accepted_slices": int(arrays["mask_deltas"].size),
                    "removed_indices": int(arrays["removal_index_count"]),
                },
                "removal_log": {"file": removal_name},
                "files": files,
            }
            manifest["end_of_run_integrity_sha256"] = hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            sidecars["v6_walk_logs.json"] = (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            self._manifest = manifest
            self._sidecars = sidecars
        except BaseException as error:
            self._error = error

    def _join(self) -> None:
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("v6 walk-log serialization failed") from self._error
        assert self._sidecars is not None

    def sidecars(self) -> dict[str, bytes]:
        """Join the writer and return the sidecar bytes (fail closed)."""

        self._join()
        return dict(self._sidecars)

    def manifest(self) -> dict[str, Any]:
        """Join the writer and return the manifest, with the integrity hash."""

        self._join()
        return json.loads(json.dumps(self._manifest))

    def arrays(self) -> dict[str, Any]:
        """Decoded arrays (acceptance and tests)."""

        return build_walk_arrays(self._results)


def load_sidecar_array(name: str, data: bytes) -> np.ndarray:
    codec = "zstd" if name.endswith(".zst") else ("zlib" if name.endswith(".zlib") else "raw")
    return _npy_load(_decompress(codec, data))


# ---------------------------------------------------------------------------
# Forced-action hook (acceptance testing).  IRG_V6_FORCED_ACTIONS=<path>.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForcedAction:
    action: str
    reason: str | None


class ForcedActions:
    """Logged actions keyed by (layer_id, partition_id, proposal_index)."""

    def __init__(self, entries: Mapping[tuple[str, str, int], ForcedAction], *, source: str) -> None:
        self.entries = dict(entries)
        self.source = str(source)

    def lookup(
        self, *, query_id: str, layer_id: str, partition_id: str, proposal_index: int
    ) -> ForcedAction:
        key = (str(layer_id), str(partition_id), int(proposal_index))
        forced = self.entries.get(key)
        if forced is None:
            raise RuntimeError(
                "forced-action replay diverged: the log has no decision "
                f"{decision_key(query_id, layer_id, partition_id, proposal_index)!r}"
            )
        return forced


_FORCED_CACHE: dict[str, ForcedActions] = {}


def _entries_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int], ForcedAction]:
    entries: dict[tuple[str, str, int], ForcedAction] = {}
    per_layer: dict[str, int] = {}
    for row in rows:
        layer = str(row["layer_id"])
        if "proposal_index" in row and row["proposal_index"] is not None:
            index = int(row["proposal_index"])
        else:
            index = per_layer.get(layer, 0)
        per_layer[layer] = index + 1
        key = (layer, str(row["partition_id"]), index)
        if key in entries:
            raise ValueError(f"forced-action log repeats decision {key!r}")
        action = str(row["action"])
        if action not in ACTION_CODES:
            raise ValueError(f"forced-action log has an unknown action {action!r}")
        reason = row.get("reason")
        entries[key] = ForcedAction(action=action, reason=None if reason is None else str(reason))
    return entries


def load_forced_actions(path: str) -> ForcedActions:
    """Read a decision log: v6 manifest, JSONL progress, receipt or JSON list."""

    resolved = os.path.abspath(str(path))
    cached = _FORCED_CACHE.get(resolved)
    if cached is not None:
        return cached
    with open(resolved, "rb") as handle:
        raw = handle.read()
    rows: list[Mapping[str, Any]] = []
    text = raw.decode("utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "v0_proposal_decision_completed":
                rows.append(event)
    if isinstance(payload, Mapping) and payload.get("schema") == V6_WALK_LOGS_SCHEMA:
        directory = os.path.dirname(resolved)
        name = next(n for n in payload["files"] if n.startswith("v6_decision_log"))
        with open(os.path.join(directory, name), "rb") as handle:
            records = load_sidecar_array(name, handle.read())
        tables = payload["tables"]
        rows = [
            {
                "layer_id": tables["layers"][int(r["layer_code"])],
                "partition_id": tables["partitions"][int(r["partition_code"])],
                "proposal_index": int(r["proposal_index"]),
                "action": tables["actions"][int(r["action_code"])],
                "reason": tables["reasons"][int(r["reason_code"])],
            }
            for r in records
        ]
    elif isinstance(payload, Mapping) and isinstance(payload.get("workflow"), Mapping):
        rows = list(payload["workflow"].get("decision_summaries", ()))
    elif isinstance(payload, Mapping) and "decisions" in payload:
        rows = list(payload["decisions"])
    elif isinstance(payload, list):
        rows = list(payload)
    if not rows:
        raise ValueError(f"forced-action log {resolved!r} holds no decisions")
    forced = ForcedActions(_entries_from_rows(rows), source=resolved)
    _FORCED_CACHE[resolved] = forced
    return forced


def forced_actions_from_env() -> ForcedActions | None:
    path = os.environ.get(V6_FORCED_ACTIONS_ENV, "").strip()
    return load_forced_actions(path) if path else None


__all__ = [
    "ACTION_CODES",
    "DECISION_LOOK_DTYPE",
    "DECISION_RECORD_DTYPE",
    "ForcedActions",
    "MASK_DELTA_DTYPE",
    "REMOVAL_STEP_DTYPE",
    "TraceWalkLog",
    "TraceWalkLogHandle",
    "V6WalkProducts",
    "V6_FORCED_ACTIONS_ENV",
    "V6_LEDGER_SCHEMA",
    "build_walk_arrays",
    "decision_key",
    "decision_record_hash",
    "delta_decode_u32",
    "delta_encode_u32",
    "forced_actions_from_env",
    "load_forced_actions",
    "load_sidecar_array",
    "read_removal_log_for_tests",
]
