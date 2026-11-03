"""Compact run outputs bound by one integrity hash.

A run keeps the data products (step matrices, the final mask, the
category matrices, the decision logs) as compact binary written off the
critical path, with ONE end-of-run integrity hash, and bumps the receipt
layout.  This module owns the arm-output writer and the on-disk formats:

* ``RunProductWriter`` writes every product on one helper thread, in a fixed
  order, while the caller serializes the receipt; the helper is
  joined, re-raising any failure, before ``receipt.json`` is written, so the
  receipt is always written LAST and never exists without its products.
* Each product's sha256 is taken inline: from the bytes the caller holds, or,
  for an already-staged file (the multi-GB step archive), from the digest its
  own streaming writer computed.  ``receipt.json`` lists every product's
  sha256 and size under ``run_products``, and its ``receipt_hash`` (the
  canonical ``content_hash``, over the compact payload) is the
  single end-of-run integrity hash: it binds every product digest.
* The retained per-execution telemetry rows (``physical_execution_receipts``,
  1,024 per arm under the launcher's retention) leave ``receipt.json`` for a
  compressed JSON-lines trace sidecar.  In the receipt each moved value is
  replaced by a reference carrying its row count and the sha256 of the
  uncompressed trace, so no field is lost and nothing is silently dropped.
* Product sources that serialize on their own thread (the walk's decision,
  mask-delta and removal logs, ``workflow_details["v6_walk_logs"]``) are
  joined before anything is hashed, because joining completes their receipt
  descriptor, and their bytes are written as products.

Nothing here touches a draw, a statistic or a decision.  The two
seed-bearing digests (the candidate-mask digest and the removed-index digest)
are computed in the search and are not read or rewritten here.

"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

RECEIPT_LAYOUT_V6 = "irg_v6_compact_receipt_v1"
RUN_PRODUCTS_SCHEMA = "irg_v6_run_products_v1"
TRACE_SCHEMA = "irg_v6_execution_trace_v1"
TRACE_REFERENCE_SCHEMA = "irg_v6_trace_reference_v1"
TRACE_BASENAME = "trace_execution_receipts.jsonl"
CODEC_ENV = "IRG_V6_D_CODEC"
SOURCE_MANIFEST_ENV = "IRG_SOURCE_MANIFEST_VERIFICATION"
SOURCE_MANIFEST_SCHEMA = "irg_source_manifest_verification_v1"
SOURCE_MANIFEST_MODES = frozenset(
    {
        "verify_once_full_hash",
        "verify_once_reused_stamp",
        "sha256sum_check",
        "git_head_and_clean_status",
    }
)
# The receipt paths whose values are per-execution telemetry rows.  The
# unified execution receipt carries a shallow copy of the provider usage, so
# both paths normally hold the SAME row tuple; it is stored once.
TRACE_PATHS: tuple[tuple[str, ...], ...] = (
    ("decision_provider_resource_usage", "physical_execution_receipts"),
    (
        "unified_native_execution_receipt",
        "provider",
        "physical_execution_receipts",
    ),
)
# Keys of ``workflow_details`` holding a product source: an object with a
# ``sidecars() -> Mapping[str, bytes]`` method (the walk logs).
PRODUCT_SOURCE_KEYS: tuple[str, ...] = ("v6_walk_logs",)
RESERVED_PRODUCT_NAMES = frozenset({"receipt.json", "performance_gate.json"})


CALIBRATION_PENALTY_ENV = "IRG_CALIBRATION_PENALTY"
LOSS_WEIGHTING_ENV = "IRG_LOSS_WEIGHTING"
SUBSTITUTED_LOSS_WEIGHTING_ENV = "IRG_SUBSTITUTED_LOSS_WEIGHTING"
PRUNING_FUNCTIONAL_ENV = "IRG_PRUNING_FUNCTIONAL"


def loss_weighting(env: Mapping[str, str] | None = None) -> str:
    """Resolve the binary loss estimand; explicit blank values are invalid."""
    source = os.environ if env is None else env
    value = str(source.get(LOSS_WEIGHTING_ENV, "balanced"))
    if value not in {"balanced", "per_image"}:
        raise ValueError("IRG_LOSS_WEIGHTING must be balanced or per_image")
    return value


def substituted_loss_weighting(env: Mapping[str, str] | None = None) -> str:
    """Resolve the reporting-only substituted loss policy."""
    source = os.environ if env is None else env
    value = str(source.get(SUBSTITUTED_LOSS_WEIGHTING_ENV, "follow"))
    if value not in {"follow", "per_image"}:
        raise ValueError("IRG_SUBSTITUTED_LOSS_WEIGHTING must be follow or per_image")
    return value


def effective_substituted_loss_weighting(env: Mapping[str, str] | None = None) -> str:
    """Resolve the substituted estimand without changing the ordinary loss."""
    source = os.environ if env is None else env
    return loss_weighting(source) if substituted_loss_weighting(source) == "follow" else "per_image"


def calibration_penalty(env: Mapping[str, str] | None = None) -> str:
    """Resolve the calibration law; an explicit empty value is invalid."""
    source = os.environ if env is None else env
    value = str(source.get(CALIBRATION_PENALTY_ENV, "identity_anchor"))
    if value not in {"identity_anchor", "firth"}:
        raise ValueError("IRG_CALIBRATION_PENALTY must be identity_anchor or firth")
    return value


def pruning_functional(env: Mapping[str, str] | None = None) -> str:
    """Resolve the walk objective and require the coherent substituted arm."""
    source = os.environ if env is None else env
    value = str(source.get(PRUNING_FUNCTIONAL_ENV, "target_logit"))
    if value not in {"target_logit", "substituted_softmax"}:
        raise ValueError("IRG_PRUNING_FUNCTIONAL must be target_logit or substituted_softmax")
    if value == "substituted_softmax":
        if loss_weighting(source) != "per_image":
            raise ValueError("substituted_softmax requires IRG_LOSS_WEIGHTING=per_image")
        if calibration_penalty(source) != "firth":
            raise ValueError("substituted_softmax requires IRG_CALIBRATION_PENALTY=firth")
    return value


def effective_settings(
    env: Mapping[str, str] | None = None,
    *, resolved: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Disclose default policies and component-resolved runtime values.

    Image gathering and pinned storage belong to a data plane. Receipts for
    components that do not own those resources report zero; their owners
    supply the enabled state and allocated byte limit through ``resolved``.
    """
    from mechanistic_systems.measurement.run_products import (
        loss_weighting, LOSS_WEIGHTING_ENV,
        substituted_loss_weighting, SUBSTITUTED_LOSS_WEIGHTING_ENV,
        pruning_functional, PRUNING_FUNCTIONAL_ENV,
    )

    source = os.environ if env is None else env
    settings = {"CIRCUIT_FORWARD_PREFIX_REUSE": "1"}
    settings.update({name: str(source[name]) for name in source
                     if name.startswith(("IRG_", "CIRCUIT_"))})
    penalty = calibration_penalty(source)
    if pruning_functional(source) == "target_logit":
        settings.pop(PRUNING_FUNCTIONAL_ENV, None)
    if loss_weighting(source) == "balanced":
        settings.pop(LOSS_WEIGHTING_ENV, None)
    if substituted_loss_weighting(source) == "follow":
        settings.pop(SUBSTITUTED_LOSS_WEIGHTING_ENV, None)
    if penalty == "identity_anchor":
        settings.pop(CALIBRATION_PENALTY_ENV, None)
    prefix = str(source.get("IRG_PREFIX_REUSE", "") or
                 source.get("CIRCUIT_FORWARD_PREFIX_REUSE", "") or "1").strip().lower()
    settings["CIRCUIT_FORWARD_PREFIX_REUSE"] = (
        "0" if prefix in {"0", "false", "no", "off"} else "1")
    settings["IRG_DEVICE_GATHER_POLICY"] = str(source.get("IRG_DEVICE_GATHER", "auto"))
    settings["IRG_PINNED_POOL_MAX_BYTES_POLICY"] = str(source.get("IRG_PINNED_POOL_MAX_BYTES", "auto"))
    settings["IRG_DEVICE_GATHER"] = "0"
    settings["IRG_PINNED_POOL_MAX_BYTES"] = "0"
    if resolved is not None:
        settings.update({name: str(value) for name, value in resolved.items()})
    if calibration_penalty(settings) == "identity_anchor":
        settings.pop(CALIBRATION_PENALTY_ENV, None)
    if loss_weighting(settings) == "balanced":
        settings.pop(LOSS_WEIGHTING_ENV, None)
    if substituted_loss_weighting(settings) == "follow":
        settings.pop(SUBSTITUTED_LOSS_WEIGHTING_ENV, None)
    if pruning_functional(settings) == "target_logit":
        settings.pop(PRUNING_FUNCTIONAL_ENV, None)
    from mechanistic_systems.core.determinism import deterministic_backend_record

    backend = deterministic_backend_record()
    settings.update({
        "CUDNN_BENCHMARK": str(int(backend["cudnn_benchmark"])),
        "CUDNN_DETERMINISTIC": str(int(backend["cudnn_deterministic"])),
        "TORCH_DETERMINISTIC_ALGORITHMS": str(int(backend["deterministic_algorithms"])),
        "TORCH_DETERMINISTIC_WARN_ONLY": str(int(backend["deterministic_warn_only"])),
        "CUBLAS_WORKSPACE_CONFIG": str(backend["cublas_workspace_config"]),
        "TORCHINDUCTOR_MAX_AUTOTUNE": str(int(backend["inductor_max_autotune"])),
        "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE": str(int(backend["inductor_max_autotune_pointwise"])),
        "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": str(int(backend["inductor_max_autotune_gemm"])),
        "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": str(int(backend["inductor_coordinate_descent_tuning"])),
        "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK": str(int(backend["inductor_dynamic_scale_rblock"])),
        "TORCHINDUCTOR_SEARCH_AUTOTUNE_CACHE": str(int(backend["inductor_search_autotune_cache"])),
        "TORCHINDUCTOR_BENCHMARK_FUSION": str(int(backend["inductor_benchmark_fusion"])),
        "TRITON_FIXED_RECIPES": json.dumps(
            backend["fixed_triton_recipes"], sort_keys=True, separators=(",", ":")),
    })
    return dict(sorted(settings.items()))


# ---------------------------------------------------------------------------
# Codec: zstandard when importable (the cluster environment ships it), else
# zlib.  The suffix names the codec, so a reader never guesses.
# ---------------------------------------------------------------------------


def _zstandard_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("zstandard") is not None
    except (ImportError, ValueError):
        return False


def product_codec(env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    forced = str(source.get(CODEC_ENV, "")).strip().lower()
    if forced:
        if forced not in {"zstd", "zlib", "raw"}:
            raise ValueError(f"{CODEC_ENV} must be zstd, zlib or raw")
        if forced == "zstd" and not _zstandard_available():
            raise RuntimeError(f"{CODEC_ENV}=zstd but zstandard is not importable")
        return forced
    return "zstd" if _zstandard_available() else "zlib"


def codec_suffix(codec: str) -> str:
    return {"zstd": ".zst", "zlib": ".zlib", "raw": ""}[codec]


def codec_from_name(name: str) -> str:
    if name.endswith(".zst"):
        return "zstd"
    if name.endswith(".zlib"):
        return "zlib"
    return "raw"


def compress(codec: str, data: bytes) -> bytes:
    if codec == "raw":
        return bytes(data)
    if codec == "zlib":
        return zlib.compress(bytes(data), 6)
    if codec == "zstd":
        import zstandard

        return zstandard.ZstdCompressor(level=3).compress(bytes(data))
    raise ValueError(f"unknown product codec {codec!r}")


def decompress(codec: str, data: bytes) -> bytes:
    if codec == "raw":
        return bytes(data)
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "zstd":
        import zstandard

        return zstandard.ZstdDecompressor().decompressobj().decompress(data)
    raise ValueError(f"unknown product codec {codec!r}")


# ---------------------------------------------------------------------------
# The launcher's source-manifest outcome, recorded in the receipt.
# ---------------------------------------------------------------------------


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def source_manifest_verification(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The launcher's report of how it verified the source tree, validated.

    The sbatch exports ``IRG_SOURCE_MANIFEST_VERIFICATION`` as one JSON
    object after its check passed (a failed check never reaches Python).  Only
    registered fields are copied; an absent or malformed report is recorded as
    such rather than guessed.
    """

    source = os.environ if env is None else env
    raw = source.get(SOURCE_MANIFEST_ENV)
    if raw is None or not str(raw).strip():
        return {"schema": SOURCE_MANIFEST_SCHEMA, "mode": "not_reported_by_launcher"}
    try:
        report = json.loads(str(raw))
    except ValueError:
        report = None
    mode = report.get("mode") if isinstance(report, dict) else None
    if mode not in SOURCE_MANIFEST_MODES:
        return {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "mode": "unparseable_launcher_report",
            "report_sha256": hashlib.sha256(str(raw).encode("utf-8")).hexdigest(),
        }
    block: dict[str, Any] = {"schema": SOURCE_MANIFEST_SCHEMA, "mode": str(mode)}
    for key in ("file_count", "readers"):
        value = report.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            block[key] = int(value)
    for key in ("manifest_sha256", "stat_digest"):
        value = report.get(key)
        if _is_hex(value, 64):
            block[key] = str(value)
    seconds = report.get("seconds")
    if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
        if math.isfinite(float(seconds)) and float(seconds) >= 0.0:
            block["seconds"] = float(seconds)
    reason = report.get("fallback_reason")
    if isinstance(reason, str) and reason and len(reason) <= 200:
        block["fallback_reason"] = reason
    block["amortized"] = str(mode) == "verify_once_reused_stamp"
    return block


# ---------------------------------------------------------------------------
# Product sources (the walk logs)
# ---------------------------------------------------------------------------


def _check_product_name(name: Any) -> str:
    text = str(name)
    if (
        not isinstance(name, str)
        or not text
        or text in RESERVED_PRODUCT_NAMES
        or "/" in text
        or "\\" in text
        or text in {".", ".."}
        or text.startswith(".")
    ):
        raise ValueError(f"invalid run product name {name!r}")
    return text


def collect_run_product_sidecars(workflow_details: Any) -> dict[str, bytes]:
    """Join every product source in ``workflow_details``; return its bytes.

    Must run before the receipt is hashed or serialized: a source's
    ``sidecars()`` completes its descriptor in place (for the walk logs:
    ``status``, ``files`` and its own end-of-run digest).
    """

    sidecars: dict[str, bytes] = {}
    if not isinstance(workflow_details, Mapping):
        return sidecars
    for key in PRODUCT_SOURCE_KEYS:
        source = workflow_details.get(key)
        produce = getattr(source, "sidecars", None)
        if source is None or not callable(produce):
            continue
        for name, blob in dict(produce()).items():
            checked = _check_product_name(name)
            if checked in sidecars:
                raise ValueError(f"run product {checked!r} is produced twice")
            if not isinstance(blob, (bytes, bytearray, memoryview)):
                raise TypeError(f"run product {checked!r} is not bytes")
            sidecars[checked] = bytes(blob)
    return sidecars


# ---------------------------------------------------------------------------
# Execution trace sidecar
# ---------------------------------------------------------------------------


def split_trace(
    payload: Mapping[str, Any],
    paths: Sequence[Sequence[str]] = TRACE_PATHS,
) -> tuple[dict[str, Any], dict[str, tuple[Any, ...]], list[tuple[tuple[str, ...], str]]]:
    """Copy ``payload`` along ``paths`` without the trace rows.

    Returns ``(compact, members, placements)``: ``members`` maps a member name
    to its rows (each distinct row sequence once), ``placements`` lists
    ``(path, member)`` for every path that held rows.  ``payload`` itself and
    every object it references are left unmodified.
    """

    compact = dict(payload)
    members: dict[str, tuple[Any, ...]] = {}
    by_identity: dict[int, str] = {}
    placements: list[tuple[tuple[str, ...], str]] = []
    for raw_path in paths:
        path = tuple(str(part) for part in raw_path)
        parent: Any = compact
        chain: list[dict[str, Any]] = []
        for part in path[:-1]:
            child = parent.get(part) if isinstance(parent, Mapping) else None
            if not isinstance(child, Mapping):
                parent = None
                break
            copied = dict(child)
            parent[part] = copied
            chain.append(copied)
            parent = copied
        if not isinstance(parent, dict) or path[-1] not in parent:
            continue
        rows = parent[path[-1]]
        if not isinstance(rows, (list, tuple)):
            continue
        member = by_identity.get(id(rows))
        if member is None:
            candidate = tuple(rows)
            member = next(
                (name for name, existing in members.items() if existing == candidate),
                None,
            )
            if member is None:
                member = "/".join(path)
                members[member] = candidate
            by_identity[id(rows)] = member
        placements.append((path, member))
    return compact, members, placements


def encode_trace(members: Mapping[str, Sequence[Any]]) -> bytes:
    """Canonical JSON lines: a header, then one ``{member, index, row}`` per row.

    Non-finite floats raise ``ValueError`` (``allow_nan=False``): the check the
    receipt's non-finite scan used to apply to these rows.
    """

    lines = [
        json.dumps(
            {
                "schema": TRACE_SCHEMA,
                "members": {name: len(rows) for name, rows in members.items()},
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    ]
    for name, rows in members.items():
        for index, row in enumerate(rows):
            lines.append(
                json.dumps(
                    {"member": name, "index": index, "row": row},
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                    allow_nan=False,
                )
            )
    return ("\n".join(lines) + "\n").encode("utf-8")


def decode_trace(data: bytes) -> dict[str, list[Any]]:
    lines = data.decode("utf-8").splitlines()
    if not lines:
        raise ValueError("empty execution trace")
    header = json.loads(lines[0])
    if header.get("schema") != TRACE_SCHEMA:
        raise ValueError("not an execution trace")
    members: dict[str, list[Any]] = {name: [] for name in header["members"]}
    for line in lines[1:]:
        record = json.loads(line)
        rows = members[record["member"]]
        if record["index"] != len(rows):
            raise ValueError("execution trace rows are out of order")
        rows.append(record["row"])
    for name, count in header["members"].items():
        if len(members[name]) != int(count):
            raise ValueError("execution trace row count differs from its header")
    return members


def _set_path(root: dict[str, Any], path: Sequence[str], value: Any) -> None:
    node: Any = root
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_durable_product(
    destination: Path, value: Any, persist_staged: Callable[[Path, Any], None],
) -> None:
    """Write or move once, fsync, then rename into the run and fsync its directory."""

    with tempfile.TemporaryDirectory(prefix=".run-product-", dir=destination.parent) as staging:
        staged = Path(staging) / destination.name
        if isinstance(value, bytes):
            with staged.open("xb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            persist_staged(staged, value)
            with staged.open("rb") as handle:
                os.fsync(handle.fileno())
        os.replace(staged, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class RunProductWriter:
    """Writes run products on one helper thread, in the order they are added.

    ``add`` records each product's sha256 and size immediately (from the
    bytes, or for a staged file from its producer's digest), so the receipt
    can list them while the bytes are still being written.  ``join`` must be
    called before the receipt is written; it re-raises the first failure.
    One deferred slot can assemble the step archive at mask lock. The same
    thread waits for the caller to add the run's products and join it. It
    creates no destination while assembling, and holds at most one result.
    """

    def __init__(
        self,
        destination: str | os.PathLike[str],
        *,
        persist_staged: Callable[[Path, Any], None],
    ) -> None:
        self.destination = Path(destination)
        self._persist_staged = persist_staged
        self._entries: list[tuple[str, Any]] = []
        self._manifest: dict[str, dict[str, Any]] = {}
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._written: list[str] = []
        self._deferred: Callable[[], Any] | None = None
        self._deferred_value: Any = None
        self._deferred_done = threading.Event()
        self._write_ready = threading.Event()
        self._finished = False
        self.completion_transferred = False

    def transfer_completion(self) -> None:
        """Let the item writer own completion after archive assembly settles.

        Release the assembly closure and its provider before item GPU cleanup.
        The returned archive is already owned by the completion's products.
        """
        if self._deferred is not None and not self._deferred_done.is_set():
            raise RuntimeError("cannot transfer an unfinished archive assembly")
        self._raise_error()
        self._deferred = None
        self._deferred_value = None
        self.completion_transferred = True

    def defer(self, produce: Callable[[], Any]) -> None:
        """Reserve the single assembly slot before starting the writer."""

        if self._deferred is not None or self._thread is not None or self._finished:
            raise RuntimeError("run product writer already has a deferred slot or has started")
        if not callable(produce):
            raise TypeError("deferred run product must be callable")
        self._deferred = produce

    def deferred_result(self) -> Any:
        """Wait for assembly, keeping this writer available for the later writes."""

        if self._deferred is None or self._thread is None:
            raise RuntimeError("run product writer has no started deferred slot")
        self._deferred_done.wait()
        self._raise_error()
        return self._deferred_value

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("arm product write failed") from self._error

    def add(self, name: str, value: Any, *, sha256: str | None = None) -> None:
        if self._finished or (self._thread is not None and self._write_ready.is_set()):
            raise RuntimeError("run products are already being written")
        checked = _check_product_name(name)
        if checked in self._manifest:
            raise ValueError(f"run product {checked!r} is added twice")
        if isinstance(value, (bytes, bytearray, memoryview)):
            data = bytes(value)
            digest = hashlib.sha256(data).hexdigest()
            if sha256 is not None and str(sha256) != digest:
                raise ValueError(f"run product {checked!r} differs from its digest")
            size = len(data)
            stored: Any = data
        else:
            source = os.fspath(value)
            if not os.path.exists(source):
                raise RuntimeError(f"staged run product {source!r} is missing")
            size = int(os.path.getsize(source))
            # The step archive's streaming writer already hashed its bytes;
            # an unhashed staged file is hashed here, before it moves.
            digest = str(sha256) if sha256 is not None else _sha256_file(source)
            if not _is_hex(digest, 64):
                raise ValueError(f"staged run product {checked!r} has no sha256")
            stored = source
        self._entries.append((checked, stored))
        self._manifest[checked] = {"sha256": digest, "bytes": int(size)}

    def manifest(self) -> dict[str, dict[str, Any]]:
        return {name: dict(self._manifest[name]) for name in sorted(self._manifest)}

    @property
    def written(self) -> tuple[str, ...]:
        return tuple(self._written)

    def _run(self) -> None:
        try:
            if self._deferred is not None:
                self._deferred_value = self._deferred()
                self._deferred_done.set()
            self._write_ready.wait()
            for name, stored in self._entries:
                target = self.destination / name
                _write_durable_product(target, stored, self._persist_staged)
                self._written.append(name)
        except BaseException as error:  # re-raised by join()
            self._error = error
        finally:
            self._deferred_done.set()

    def start(self) -> None:
        if self._thread is not None or self._finished:
            raise RuntimeError("run products are already being written")
        if self._deferred is None:
            self._write_ready.set()
        self._thread = threading.Thread(
            target=self._run, name="arm-product-writer", daemon=True
        )
        self._thread.start()

    def join(self) -> None:
        if not self._finished:
            self._write_ready.set()
            if self._thread is None:
                self._run()
            else:
                self._thread.join()
            self._finished = True
        self._raise_error()

    def cancel(self) -> None:
        """Join assembly without starting uncommitted product writes on an error."""

        if not self._write_ready.is_set():
            self._entries.clear()
        self.join()


def write_compact_run_outputs(
    *,
    destination: str | os.PathLike[str],
    payload: Mapping[str, Any],
    products: Iterable[tuple[str, Any, str | None]],
    nonfinite_paths: Callable[[Any], Sequence[str]],
    receipt_hash: Callable[[Mapping[str, Any]], str],
    persist_staged: Callable[[Path, Any], None],
    codec: str | None = None,
    products_write_start: Callable[[], None] | None = None,
    products_write_end: Callable[[], None] | None = None,
    writer: RunProductWriter | None = None,
    before_receipt: Callable[[Mapping[str, Any]], None] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write the products and the compact receipt; return the receipt payload.

    Order: the trace is split off and encoded, the compact payload is scanned
    for non-finite values, the writer starts, the receipt is hashed and
    serialized on this thread, the writer is joined (re-raising any failure),
    and ``receipt.json`` is written last. When product-write timing is
    requested, the join precedes the receipt hash so its end row is included.
    """

    compact, members, placements = split_trace(payload)
    from mechanistic_systems.core.determinism import deterministic_backend_record

    compact["determinism"] = deterministic_backend_record()
    compact["effective_settings"] = effective_settings(
        env=environment,
        resolved=compact.get("effective_settings"))
    if PRUNING_FUNCTIONAL_ENV in compact["effective_settings"]:
        compact["pruning_functional"] = compact["effective_settings"][PRUNING_FUNCTIONAL_ENV]
    target = Path(destination)
    if writer is None:
        writer = RunProductWriter(target, persist_staged=persist_staged)
    elif writer.destination != target:
        raise ValueError("run product writer destination differs from the run")
    for name, value, sha256 in products:
        writer.add(name, value, sha256=sha256)
    resolved_codec = product_codec(environment) if codec is None else codec
    trace_block: dict[str, Any] | None = None
    if members:
        try:
            raw = encode_trace(members)
        except ValueError as error:
            raise ValueError(
                "pilot receipt trace contains accidental non-finite values"
            ) from error
        trace_name = TRACE_BASENAME + codec_suffix(resolved_codec)
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        writer.add(trace_name, compress(resolved_codec, raw))
        for path, member in placements:
            _set_path(
                compact,
                path,
                {
                    "schema": TRACE_REFERENCE_SCHEMA,
                    "file": trace_name,
                    "member": member,
                    "row_count": len(members[member]),
                    "trace_raw_sha256": raw_sha256,
                },
            )
        trace_block = {
            "file": trace_name,
            "codec": resolved_codec,
            "raw_sha256": raw_sha256,
            "raw_bytes": len(raw),
            "members": {name: len(rows) for name, rows in members.items()},
        }
    bad_paths = nonfinite_paths(compact)
    if bad_paths:
        raise ValueError(
            "pilot receipt contains accidental non-finite values at "
            + ", ".join(list(bad_paths)[:16])
        )
    target.mkdir(parents=True, exist_ok=True)
    if products_write_start is not None:
        products_write_start()
    if writer._thread is None:
        writer.start()
    else:
        writer._write_ready.set()
    try:
        compact["receipt_layout"] = RECEIPT_LAYOUT_V6
        compact["run_products"] = {
            "schema": RUN_PRODUCTS_SCHEMA,
            "files": writer.manifest(),
            "trace": trace_block,
            "integrity": "receipt_hash_binds_every_product_sha256",
            "receipt_written_last": True,
        }
        if compact.get("pruning_functional") == "substituted_softmax":
            compact["run_products"]["pruning_functional"] = "substituted_softmax"
            for descriptor in compact["run_products"]["files"].values():
                descriptor["pruning_functional"] = "substituted_softmax"
        if products_write_end is not None:
            writer.join()
            products_write_end()
        compact["receipt_hash"] = receipt_hash(compact)
        receipt_text = json.dumps(compact, indent=2, sort_keys=True, default=str) + "\n"
    finally:
        writer.join()
    if before_receipt is not None:
        before_receipt(compact)
    _write_durable_product(target / "receipt.json", receipt_text.encode("utf-8"), persist_staged)
    return compact


# ---------------------------------------------------------------------------
# Reader (tools and acceptance): verify products, optionally restore the trace
# ---------------------------------------------------------------------------


def load_run_receipt(
    directory: str | os.PathLike[str],
    *,
    verify_products: bool = True,
    restore_trace: bool = False,
) -> dict[str, Any]:
    """Load ``receipt.json``; for the v6 layout, check every product's sha256.

    ``restore_trace=True`` puts the trace rows back at their receipt paths, so
    a v5.2 reader of ``physical_execution_receipts`` sees the rows it expects.
    A v5.2-layout receipt is returned unchanged.
    """

    root = Path(directory)
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("receipt_layout") != RECEIPT_LAYOUT_V6:
        return receipt
    products = receipt.get("run_products") or {}
    files = dict(products.get("files") or {})
    if verify_products:
        for name, entry in files.items():
            path = root / _check_product_name(name)
            if int(os.path.getsize(path)) != int(entry["bytes"]):
                raise ValueError(f"run product {name!r} has the wrong size")
            if _sha256_file(path) != str(entry["sha256"]):
                raise ValueError(f"run product {name!r} differs from its sha256")
    trace = products.get("trace")
    if restore_trace and trace:
        blob = (root / _check_product_name(trace["file"])).read_bytes()
        raw = decompress(str(trace["codec"]), blob)
        if hashlib.sha256(raw).hexdigest() != str(trace["raw_sha256"]):
            raise ValueError("execution trace differs from its sha256")
        members = decode_trace(raw)
        for path in TRACE_PATHS:
            node: Any = receipt
            for part in path[:-1]:
                node = node.get(part) if isinstance(node, Mapping) else None
            reference = node.get(path[-1]) if isinstance(node, Mapping) else None
            if (
                isinstance(reference, Mapping)
                and reference.get("schema") == TRACE_REFERENCE_SCHEMA
            ):
                node[path[-1]] = list(members[str(reference["member"])])
    return receipt


__all__ = [
    "CODEC_ENV",
    "PRODUCT_SOURCE_KEYS",
    "RECEIPT_LAYOUT_V6",
    "RUN_PRODUCTS_SCHEMA",
    "RunProductWriter",
    "SOURCE_MANIFEST_ENV",
    "SOURCE_MANIFEST_SCHEMA",
    "TRACE_PATHS",
    "TRACE_REFERENCE_SCHEMA",
    "TRACE_SCHEMA",
    "codec_from_name",
    "collect_run_product_sidecars",
    "compress",
    "decode_trace",
    "decompress",
    "encode_trace",
    "effective_settings",
    "load_run_receipt",
    "product_codec",
    "source_manifest_verification",
    "split_trace",
    "write_compact_run_outputs",
]
