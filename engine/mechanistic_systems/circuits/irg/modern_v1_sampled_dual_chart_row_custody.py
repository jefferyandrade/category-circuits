"""Immutable sampled-D custody for corrected-A and literal-J row shards.

This module owns only proposal-input bytes.  A look appends exactly two raw,
signed float64 shards produced by the same registered chart capture:

* corrected Aumann--Shapley evidence rows (``A``); and
* literal relaxed-gate response-Jacobian rows (``J``).

The shards are deliberately estimand-free.  In particular, finite-population
stratum masses and normalized draw weights are *not* persisted.  They are
derived when a registered prefix is materialized, as ``pi_h / n_h``.  Thus a
change of statistical design changes a materialization receipt, never the raw
scientific observation or its identity.

Every durable receipt is path-free.  Physical paths and inode snapshots live
only in nonserializable process-local attachments.  One append performs an
aggregate scratch forecast before creating payload files, seals both roles,
constructs an append-only manifest linked to its predecessor, and publishes a
single HEAD pointer atomically.  Any failure before complete publication
removes both new-role shards and leaves the prior prefix published.

Nothing here certifies F/Q/R, opens C/K/O, authorizes a mask commit, or carries
terminal/Pareto authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat as stat_module
import tempfile
from typing import (
    Any,
    Final,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    NoReturn,
    Sequence,
)

import numpy as np

from mechanistic_systems.contracts.content_address import canonical_json, content_hash

try:  # pragma: no cover - availability is host dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SAMPLED_DUAL_CHART_ROW_FORECAST_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_forecast_v1"
)
SAMPLED_DUAL_CHART_ROW_REQUEST_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_request_v1"
)
SAMPLED_DUAL_CHART_ROW_SHARD_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_shard_receipt_v1"
)
SAMPLED_DUAL_CHART_ROW_LOOK_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_look_v1"
)
SAMPLED_DUAL_CHART_ROW_PREFIX_MANIFEST_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_prefix_manifest_v1"
)
SAMPLED_DUAL_CHART_ROW_MATERIALIZATION_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_materialization_v1"
)
SAMPLED_DUAL_CHART_ROW_VERIFIED_PACKET_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_verified_packet_v1"
)
SAMPLED_DUAL_CHART_ROW_GROUP_PACKET_PLAN_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_group_packet_plan_v1"
)
SAMPLED_DUAL_CHART_ROW_GROUP_FORECAST_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_group_forecast_v1"
)
SAMPLED_DUAL_CHART_ROW_GROUP_PLAN_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_group_plan_v1"
)
SAMPLED_DUAL_CHART_ROW_GROUP_PROOF_CLOSURE_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_group_proof_closure_v1"
)
SAMPLED_DUAL_CHART_ROW_GROUP_COMMIT_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_group_commit_receipt_v1"
)
SAMPLED_DUAL_CHART_ROW_PREPARED_GROUP_RECOVERY_RESULT_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_row_prepared_group_recovery_result_v1"
)
SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE: Final = (
    "proposal_input_custody_only_no_f_q_r_c_k_o_mask_commit_terminal_or_"
    "pareto_authority_v1"
)
SAMPLED_DUAL_CHART_ROW_LAYOUT: Final = (
    "registered_route_window_major_context_major_route_minor_little_endian_f64"
)

_FORECAST_PREFIX: Final = "partition-free-v1-sampled-dual-row-forecast-"
_REQUEST_PREFIX: Final = "partition-free-v1-sampled-dual-row-request-"
_LOGICAL_PREFIX: Final = "partition-free-v1-sampled-dual-row-logical-"
_RECEIPT_PREFIX: Final = "partition-free-v1-sampled-dual-row-receipt-"
_LOOK_PREFIX: Final = "partition-free-v1-sampled-dual-row-look-"
_MANIFEST_PREFIX: Final = "partition-free-v1-sampled-dual-row-manifest-"
_MATERIALIZATION_PREFIX: Final = "partition-free-v1-sampled-dual-row-weights-"
_GROUP_PACKET_PLAN_PREFIX: Final = (
    "partition-free-v1-sampled-dual-row-group-packet-plan-"
)
_GROUP_FORECAST_PREFIX: Final = "partition-free-v1-sampled-dual-row-group-forecast-"
_GROUP_PLAN_PREFIX: Final = "partition-free-v1-sampled-dual-row-group-plan-"
_GROUP_PROOF_PREFIX: Final = "partition-free-v1-sampled-dual-row-group-proof-"
_GROUP_COMMIT_PREFIX: Final = "partition-free-v1-sampled-dual-row-group-commit-"
_PAYLOAD_DOMAIN: Final = "partition-free-v1-sampled-dual-row-payload-v1"
_FLOAT64_BYTES: Final = 8
_HEAD_NAME: Final = ".modern-v1-sampled-dual-chart-row-prefix-head.json"
_LOCK_NAME: Final = ".modern-v1-sampled-dual-chart-row-prefix.lock"
_ISSUANCE_TOKEN: Final = object()
_STREAMING_ATTACHMENT_TOKEN: Final = object()
_GROUP_PROOF_ISSUANCE_TOKEN: Final = object()
_VERIFIED_PACKET_ISSUANCE_TOKEN: Final = object()
_COLD_ATTACHMENT_TOKEN: Final = object()
_SEALED_FILE_MODE: Final = stat_module.S_IRUSR


class V1SampledDualChartRowRole(str, Enum):
    """The two noninterchangeable row children of one registered D capture."""

    CORRECTED_AUMANN_SHAPLEY = "corrected_aumann_shapley_A"
    LITERAL_RELAXED_GATE_JACOBIAN = "literal_relaxed_gate_J"


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowChunk:
    """One same-capture route window; arrays remain ephemeral provider state."""

    start: int
    stop: int
    corrected_a_rows: Any = field(repr=False, compare=False)
    literal_j_rows: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        start = _nonnegative_int(self.start, name="dual-row chunk start")
        stop = _positive_int(self.stop, name="dual-row chunk stop")
        if stop <= start:
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "dual-row chunk must have positive width",
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)


class V1SampledDualChartRowRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    AXIS_MISMATCH = "axis_mismatch"
    STALE_IDENTITY = "stale_identity"
    ROLE_SWAP = "role_swap"
    SCRATCH_INADMISSIBLE = "scratch_inadmissible"
    NONFINITE_ROW = "nonfinite_row"
    CONTENT_MISMATCH = "content_mismatch"
    MISSING_SHARD = "missing_shard"
    CUSTODY_MUTATED = "custody_mutated"
    PUBLICATION_CONFLICT = "publication_conflict"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledDualChartRowRefusal(ValueError):
    """Typed fail-closed refusal for sampled dual-row custody."""

    def __init__(
        self,
        code: V1SampledDualChartRowRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1SampledDualChartRowRefusalCode, message: str) -> NoReturn:
    raise V1SampledDualChartRowRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _ordered_strings(
    values: object,
    *,
    name: str,
    unique: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered string sequence",
        )
    try:
        result = tuple(  # type: ignore[union-attr]
            _identity(item, name=f"{name} member") for item in values
        )
    except TypeError as error:
        raise V1SampledDualChartRowRefusal(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            f"{name} must be iterable",
        ) from error
    if not result:
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            f"{name} must be nonempty",
        )
    if unique and len(set(result)) != len(result):
        _refuse(
            V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
            f"{name} must not contain duplicates",
        )
    return result


def _sha256(value: object, *, name: str) -> str:
    result = _identity(value, name=name)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        _refuse(
            V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
            f"{name} must be one lowercase SHA-256",
        )
    return result


def _content_address_token(value: object, *, prefix: str, name: str) -> str:
    """Validate one filename-safe, domain-prefixed SHA-256 content address."""

    result = _identity(value, name=name)
    suffix = result[len(prefix) :] if result.startswith(prefix) else ""
    if (
        len(suffix) != 64
        or any(character not in "0123456789abcdef" for character in suffix)
        or "/" in result
        or "\\" in result
        or ".." in result
    ):
        _refuse(
            V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
            f"{name} is not one filename-safe content address",
        )
    return result


def _scratch_directory(value: str | os.PathLike[str]) -> Path:
    path = Path(os.fspath(value)).resolve()
    if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
        _refuse(
            V1SampledDualChartRowRefusalCode.SCRATCH_INADMISSIBLE,
            "node-local scratch must be an existing writable directory",
        )
    return path


def _scratch_capacity(path: Path) -> tuple[int, int]:
    try:
        stat = os.statvfs(path)
    except OSError as error:
        raise V1SampledDualChartRowRefusal(
            V1SampledDualChartRowRefusalCode.SCRATCH_INADMISSIBLE,
            "node-local scratch capacity could not be inspected",
        ) from error
    fragment = int(stat.f_frsize)
    if fragment <= 0:  # pragma: no cover - invalid filesystem metadata.
        _refuse(
            V1SampledDualChartRowRefusalCode.SCRATCH_INADMISSIBLE,
            "filesystem reported a nonpositive fragment size",
        )
    return int(stat.f_bavail) * fragment, fragment


def _route_windows(route_count: int, chunk_size: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, min(route_count, start + chunk_size))
        for start in range(0, route_count, chunk_size)
    )


def _registered_route_window_cover(
    route_count: int,
    chunk_size: int,
    route_windows: Iterable[tuple[int, int] | Sequence[int]],
    *,
    label: str,
) -> tuple[tuple[int, int], ...]:
    """Validate a provider-preserved contiguous route cover.

    Production positive-removal streaming is block-aware: legal chunks may be
    shorter than the registered maximum at route-block boundaries.  Custody
    therefore binds the exact observed ledger instead of imposing a synthetic
    fixed global tiling, while still requiring one ordered, complete, bounded
    cover of the route axis.
    """

    routes = _positive_int(route_count, name=f"{label} route count")
    chunk = _positive_int(chunk_size, name=f"{label} route chunk size")
    if chunk > routes:
        _refuse(
            V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
            f"{label} route chunk exceeds the route axis",
        )
    expected_start = 0
    normalized: list[tuple[int, int]] = []
    for pair in route_windows:
        window = tuple(pair)
        if len(window) != 2:
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                f"{label} route-window entry is not a start/stop pair",
            )
        start, stop = window
        begin = _nonnegative_int(start, name=f"{label} route-window start")
        end = _positive_int(stop, name=f"{label} route-window stop")
        if (
            begin != expected_start
            or end <= begin
            or end > routes
            or end - begin > chunk
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                f"{label} route windows are not the exact registered cover",
            )
        normalized.append((begin, end))
        expected_start = end
    if expected_start != routes or not normalized:
        _refuse(
            V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
            f"{label} route windows do not close the route axis",
        )
    return tuple(normalized)


def _new_payload_hasher(role: V1SampledDualChartRowRole) -> Any:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN.encode("ascii"))
    digest.update(b"\x00")
    digest.update(role.value.encode("ascii"))
    digest.update(b"\x00")
    return digest


def _file_digest(path: Path, role: V1SampledDualChartRowRole) -> str:
    digest = _new_payload_hasher(role)
    with path.open("rb", buffering=0) as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class _FileInvariant(NamedTuple):
    """Process-local immutable-file witness; never enters a durable receipt."""

    device: int
    inode: int
    size: int
    uid: int
    mode: int
    ctime_ns: int
    mtime_ns: int


def _invariant_from_stat(stat: os.stat_result) -> _FileInvariant:
    return _FileInvariant(
        device=int(stat.st_dev),
        inode=int(stat.st_ino),
        size=int(stat.st_size),
        uid=int(stat.st_uid),
        mode=int(stat.st_mode),
        ctime_ns=int(stat.st_ctime_ns),
        mtime_ns=int(stat.st_mtime_ns),
    )


def _file_invariant(path: Path) -> _FileInvariant:
    stat = path.stat()
    return _invariant_from_stat(stat)


def _descriptor_invariant(descriptor: int) -> _FileInvariant:
    stat = os.fstat(descriptor)
    return _invariant_from_stat(stat)


def _is_read_only_regular_file(invariant: _FileInvariant) -> bool:
    return stat_module.S_ISREG(invariant.mode) and not bool(
        stat_module.S_IMODE(invariant.mode) & 0o222
    )


def _seal_file_read_only(path: Path) -> _FileInvariant:
    """Remove every write bit and durably close the hot-lease metadata seam."""

    os.chmod(path, _SEALED_FILE_MODE)
    with path.open("rb", buffering=0) as stream:
        os.fsync(stream.fileno())
        invariant = _descriptor_invariant(stream.fileno())
    if not _is_read_only_regular_file(invariant):  # pragma: no cover - OS defect.
        _refuse(
            V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
            "sampled row shard could not be sealed read-only",
        )
    return invariant


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowScratchForecast:
    """Allocation-free aggregate forecast for both role shards of one look."""

    route_count: int
    context_count: int
    registered_route_chunk_size: int
    exact_bytes_per_role_shard: int
    role_shard_count: int
    aggregate_exact_payload_bytes: int
    maximum_owned_host_staging_bytes: int
    filesystem_fragment_size_bytes: int
    allocation_bytes_per_role_upper_bound: int
    aggregate_allocation_bytes_upper_bound: int
    filesystem_free_bytes_before: int
    admitted: bool
    forecast_constructed_without_file_creation: bool = True
    raw_shards_carry_normalized_weights: bool = False
    schema: str = SAMPLED_DUAL_CHART_ROW_FORECAST_SCHEMA
    _forecast_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        routes = _positive_int(self.route_count, name="forecast route count")
        contexts = _positive_int(self.context_count, name="forecast context count")
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="forecast route chunk size",
        )
        if chunk > routes:
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "route chunk size exceeds the route axis",
            )
        fragment = _positive_int(
            self.filesystem_fragment_size_bytes,
            name="filesystem fragment size",
        )
        per_role = routes * contexts * _FLOAT64_BYTES
        per_allocation = ((per_role + fragment - 1) // fragment) * fragment
        aggregate = 2 * per_role
        aggregate_allocation = 2 * per_allocation
        free = _nonnegative_int(
            self.filesystem_free_bytes_before,
            name="filesystem free bytes",
        )
        if (
            self.exact_bytes_per_role_shard != per_role
            or self.role_shard_count != 2
            or self.aggregate_exact_payload_bytes != aggregate
            or self.maximum_owned_host_staging_bytes
            != contexts * chunk * _FLOAT64_BYTES
            or self.allocation_bytes_per_role_upper_bound != per_allocation
            or self.aggregate_allocation_bytes_upper_bound != aggregate_allocation
            or self.admitted is not (aggregate_allocation <= free)
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "aggregate dual-role scratch forecast does not replay",
            )
        if (
            self.forecast_constructed_without_file_creation is not True
            or self.raw_shards_carry_normalized_weights is not False
            or self.schema != SAMPLED_DUAL_CHART_ROW_FORECAST_SCHEMA
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "scratch forecast changed the estimand-free two-shard contract",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "context_count", contexts)
        object.__setattr__(self, "registered_route_chunk_size", chunk)
        object.__setattr__(self, "filesystem_fragment_size_bytes", fragment)
        object.__setattr__(self, "filesystem_free_bytes_before", free)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "route_count",
                "context_count",
                "registered_route_chunk_size",
                "exact_bytes_per_role_shard",
                "role_shard_count",
                "aggregate_exact_payload_bytes",
                "maximum_owned_host_staging_bytes",
                "filesystem_fragment_size_bytes",
                "allocation_bytes_per_role_upper_bound",
                "aggregate_allocation_bytes_upper_bound",
                "filesystem_free_bytes_before",
                "admitted",
                "forecast_constructed_without_file_creation",
                "raw_shards_carry_normalized_weights",
            )
        }

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_FORECAST_PREFIX)
        if self._forecast_hash is not None and self._forecast_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "scratch forecast was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "scratch forecast hash differs",
            )
        object.__setattr__(self, "_forecast_hash", observed)
        return observed

    def assert_admissible(self) -> None:
        self.assert_integrity()
        if not self.admitted:
            _refuse(
                V1SampledDualChartRowRefusalCode.SCRATCH_INADMISSIBLE,
                "both raw role shards do not fit the pre-write scratch forecast",
            )


def forecast_v1_sampled_dual_chart_row_scratch(
    *,
    route_count: int,
    context_count: int,
    registered_route_chunk_size: int,
    scratch_directory: str | os.PathLike[str],
) -> V1SampledDualChartRowScratchForecast:
    """Forecast both immutable role shards without creating a file."""

    path = _scratch_directory(scratch_directory)
    routes = _positive_int(route_count, name="route count")
    contexts = _positive_int(context_count, name="context count")
    chunk = _positive_int(registered_route_chunk_size, name="route chunk size")
    if chunk > routes:
        _refuse(
            V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
            "route chunk size exceeds the route axis",
        )
    free, fragment = _scratch_capacity(path)
    payload = routes * contexts * _FLOAT64_BYTES
    allocation = ((payload + fragment - 1) // fragment) * fragment
    return V1SampledDualChartRowScratchForecast(
        route_count=routes,
        context_count=contexts,
        registered_route_chunk_size=chunk,
        exact_bytes_per_role_shard=payload,
        role_shard_count=2,
        aggregate_exact_payload_bytes=2 * payload,
        maximum_owned_host_staging_bytes=contexts * chunk * _FLOAT64_BYTES,
        filesystem_fragment_size_bytes=fragment,
        allocation_bytes_per_role_upper_bound=allocation,
        aggregate_allocation_bytes_upper_bound=2 * allocation,
        filesystem_free_bytes_before=free,
        admitted=2 * allocation <= free,
    )


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowGroupPacketPlan:
    """Path-free preregistration of one physical packet in an atomic look.

    ``registered_packet_plan_hash`` binds the owning sampled-D transaction's
    packet plan.  Provider and reducer *plan* identities are fixed here;
    their content-dependent completion receipts are supplied only after the
    packet has been staged.  No field may contain an operational path.
    """

    packet_ordinal: int
    registered_packet_plan_hash: str
    source_capture_identity_hash: str
    provider_plan_identity_hash: str
    reducer_plan_identity_hashes: tuple[str, ...]
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_labels: tuple[str, ...]
    schema: str = SAMPLED_DUAL_CHART_ROW_GROUP_PACKET_PLAN_SCHEMA
    _packet_plan_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        ordinal = _nonnegative_int(self.packet_ordinal, name="group packet ordinal")
        contexts = _ordered_strings(
            self.context_ids,
            name="group packet context IDs",
            unique=True,
        )
        draws = _ordered_strings(
            self.draw_ids,
            name="group packet draw IDs",
            unique=True,
        )
        strata = _ordered_strings(
            self.stratum_labels,
            name="group packet stratum labels",
            unique=False,
        )
        reducers = _ordered_strings(
            self.reducer_plan_identity_hashes,
            name="group packet reducer plan identities",
            unique=True,
        )
        if len(contexts) != len(draws) or len(contexts) != len(strata):
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "group packet context, draw, and stratum axes differ",
            )
        if len(set(strata)) != 1:
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "one registered sampled-D packet must remain in one stratum",
            )
        for name in (
            "registered_packet_plan_hash",
            "source_capture_identity_hash",
            "provider_plan_identity_hash",
        ):
            _identity(getattr(self, name), name=name.replace("_", " "))
        if self.provider_plan_identity_hash in reducers:
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "provider and reducer plan identities must remain distinct",
            )
        if self.schema != SAMPLED_DUAL_CHART_ROW_GROUP_PACKET_PLAN_SCHEMA:
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "group packet-plan schema changed",
            )
        object.__setattr__(self, "packet_ordinal", ordinal)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "stratum_labels", strata)
        object.__setattr__(self, "reducer_plan_identity_hashes", reducers)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "packet_ordinal": self.packet_ordinal,
            "registered_packet_plan_hash": self.registered_packet_plan_hash,
            "source_capture_identity_hash": self.source_capture_identity_hash,
            "provider_plan_identity_hash": self.provider_plan_identity_hash,
            "reducer_plan_identity_hashes": self.reducer_plan_identity_hashes,
            "context_ids": self.context_ids,
            "draw_ids": self.draw_ids,
            "stratum_labels": self.stratum_labels,
        }

    @property
    def packet_plan_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_GROUP_PACKET_PLAN_PREFIX)
        if self._packet_plan_hash is not None and self._packet_plan_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group packet plan was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group packet-plan hash differs",
            )
        object.__setattr__(self, "_packet_plan_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowGroupScratchForecast:
    """Path-free aggregate allocation forecast for every packet-role shard."""

    route_count: int
    packet_context_counts: tuple[int, ...]
    registered_route_chunk_size: int
    exact_payload_bytes_per_role_shard: tuple[int, ...]
    role_shard_count: int
    aggregate_exact_payload_bytes: int
    maximum_owned_host_staging_bytes: int
    filesystem_fragment_size_bytes: int
    allocation_bytes_per_role_shard_upper_bound: tuple[int, ...]
    aggregate_allocation_bytes_upper_bound: int
    filesystem_free_bytes_before: int
    admitted: bool
    forecast_constructed_without_file_creation: bool = True
    raw_shards_carry_normalized_weights: bool = False
    schema: str = SAMPLED_DUAL_CHART_ROW_GROUP_FORECAST_SCHEMA
    _forecast_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        routes = _positive_int(self.route_count, name="group forecast route count")
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="group forecast route chunk size",
        )
        if chunk > routes:
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "group route chunk exceeds its route axis",
            )
        counts = tuple(
            _positive_int(value, name="group packet context count")
            for value in self.packet_context_counts
        )
        if not counts:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "group forecast requires at least one packet",
            )
        fragment = _positive_int(
            self.filesystem_fragment_size_bytes,
            name="group filesystem fragment size",
        )
        exact = tuple(routes * count * _FLOAT64_BYTES for count in counts)
        allocations = tuple(
            ((value + fragment - 1) // fragment) * fragment for value in exact
        )
        free = _nonnegative_int(
            self.filesystem_free_bytes_before,
            name="group filesystem free bytes",
        )
        aggregate_exact = 2 * sum(exact)
        aggregate_allocation = 2 * sum(allocations)
        if (
            tuple(self.exact_payload_bytes_per_role_shard) != exact
            or self.role_shard_count != 2 * len(counts)
            or self.aggregate_exact_payload_bytes != aggregate_exact
            or self.maximum_owned_host_staging_bytes
            != max(counts) * chunk * _FLOAT64_BYTES
            or tuple(self.allocation_bytes_per_role_shard_upper_bound) != allocations
            or self.aggregate_allocation_bytes_upper_bound != aggregate_allocation
            or self.admitted is not (aggregate_allocation <= free)
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group aggregate scratch forecast does not replay",
            )
        if (
            self.forecast_constructed_without_file_creation is not True
            or self.raw_shards_carry_normalized_weights is not False
            or self.schema != SAMPLED_DUAL_CHART_ROW_GROUP_FORECAST_SCHEMA
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "group forecast changed raw-row custody semantics",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "packet_context_counts", counts)
        object.__setattr__(self, "registered_route_chunk_size", chunk)
        object.__setattr__(self, "filesystem_fragment_size_bytes", fragment)
        object.__setattr__(self, "filesystem_free_bytes_before", free)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "route_count",
                "packet_context_counts",
                "registered_route_chunk_size",
                "exact_payload_bytes_per_role_shard",
                "role_shard_count",
                "aggregate_exact_payload_bytes",
                "maximum_owned_host_staging_bytes",
                "filesystem_fragment_size_bytes",
                "allocation_bytes_per_role_shard_upper_bound",
                "aggregate_allocation_bytes_upper_bound",
                "filesystem_free_bytes_before",
                "admitted",
                "forecast_constructed_without_file_creation",
                "raw_shards_carry_normalized_weights",
            )
        }

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_GROUP_FORECAST_PREFIX)
        if self._forecast_hash is not None and self._forecast_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group scratch forecast was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group scratch forecast hash differs",
            )
        object.__setattr__(self, "_forecast_hash", observed)
        return observed

    def assert_admissible(self) -> None:
        self.assert_integrity()
        if not self.admitted:
            _refuse(
                V1SampledDualChartRowRefusalCode.SCRATCH_INADMISSIBLE,
                "all preregistered packet-role shards do not fit scratch",
            )


def forecast_v1_sampled_dual_chart_row_group_scratch(
    *,
    route_count: int,
    packet_context_counts: Sequence[int],
    registered_route_chunk_size: int,
    scratch_directory: str | os.PathLike[str],
) -> V1SampledDualChartRowGroupScratchForecast:
    """Forecast a complete packet group before creating any custody file."""

    directory = _scratch_directory(scratch_directory)
    routes = _positive_int(route_count, name="group route count")
    chunk = _positive_int(
        registered_route_chunk_size,
        name="group registered route chunk size",
    )
    counts = tuple(
        _positive_int(value, name="group packet context count")
        for value in packet_context_counts
    )
    if not counts:
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            "group forecast requires one or more packet context counts",
        )
    if chunk > routes:
        _refuse(
            V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
            "group route chunk exceeds the route axis",
        )
    free, fragment = _scratch_capacity(directory)
    exact = tuple(routes * count * _FLOAT64_BYTES for count in counts)
    allocations = tuple(
        ((value + fragment - 1) // fragment) * fragment for value in exact
    )
    aggregate_allocation = 2 * sum(allocations)
    return V1SampledDualChartRowGroupScratchForecast(
        route_count=routes,
        packet_context_counts=counts,
        registered_route_chunk_size=chunk,
        exact_payload_bytes_per_role_shard=exact,
        role_shard_count=2 * len(counts),
        aggregate_exact_payload_bytes=2 * sum(exact),
        maximum_owned_host_staging_bytes=max(counts) * chunk * _FLOAT64_BYTES,
        filesystem_fragment_size_bytes=fragment,
        allocation_bytes_per_role_shard_upper_bound=allocations,
        aggregate_allocation_bytes_upper_bound=aggregate_allocation,
        filesystem_free_bytes_before=free,
        admitted=aggregate_allocation <= free,
    )


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowGroupPlan:
    """Complete path-free, pre-outcome plan for one atomic sampled-D look."""

    registered_look_index: int
    expected_prior_manifest_hash: str | None
    first_prefix_look_ordinal: int
    registered_d_prefix_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    packet_plans: tuple[V1SampledDualChartRowGroupPacketPlan, ...]
    registered_route_chunk_size: int
    scratch_forecast: V1SampledDualChartRowGroupScratchForecast
    all_packets_preregistered: bool = True
    provider_and_reducer_proofs_required_before_publication: bool = True
    one_head_publication_boundary: bool = True
    raw_shards_carry_normalized_weights: bool = False
    authority_scope: str = SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
    schema: str = SAMPLED_DUAL_CHART_ROW_GROUP_PLAN_SCHEMA
    _group_plan_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        look = _positive_int(self.registered_look_index, name="registered look index")
        first = _nonnegative_int(
            self.first_prefix_look_ordinal,
            name="first prefix look ordinal",
        )
        generation = _nonnegative_int(
            self.proposal_generation,
            name="group proposal generation",
        )
        routes = _positive_int(self.route_count, name="group route count")
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="group route chunk size",
        )
        packets = tuple(self.packet_plans)
        if not packets or any(
            type(packet) is not V1SampledDualChartRowGroupPacketPlan
            for packet in packets
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "group plan requires one or more exact packet plans",
            )
        if tuple(packet.packet_ordinal for packet in packets) != tuple(
            range(len(packets))
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "group packet ordinals must be one exact contiguous cover",
            )
        if first == 0 and self.expected_prior_manifest_hash is not None:
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "a first prefix group cannot name a predecessor manifest",
            )
        if first > 0:
            _identity(
                self.expected_prior_manifest_hash,
                name="group predecessor manifest hash",
            )
        if type(self.scratch_forecast) is not V1SampledDualChartRowGroupScratchForecast:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "group plan requires its exact aggregate scratch forecast",
            )
        self.scratch_forecast.assert_integrity()
        if (
            self.scratch_forecast.route_count != routes
            or self.scratch_forecast.registered_route_chunk_size != chunk
            or self.scratch_forecast.packet_context_counts
            != tuple(len(packet.context_ids) for packet in packets)
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "group packet axes differ from the aggregate scratch forecast",
            )
        contexts = tuple(value for packet in packets for value in packet.context_ids)
        draws = tuple(value for packet in packets for value in packet.draw_ids)
        if len(set(contexts)) != len(contexts) or len(set(draws)) != len(draws):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "group packets reuse a context or draw identity",
            )
        for packet in packets:
            packet.assert_integrity()
        for name in (
            "registered_d_prefix_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "corrected_a_stream_identity_hash",
            "literal_j_stream_identity_hash",
        ):
            _identity(getattr(self, name), name=name.replace("_", " "))
        if self.corrected_a_stream_identity_hash == self.literal_j_stream_identity_hash:
            _refuse(
                V1SampledDualChartRowRefusalCode.ROLE_SWAP,
                "group corrected-A and literal-J streams collapsed",
            )
        if (
            self.all_packets_preregistered is not True
            or self.provider_and_reducer_proofs_required_before_publication is not True
            or self.one_head_publication_boundary is not True
            or self.raw_shards_carry_normalized_weights is not False
            or self.authority_scope != SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
            or self.schema != SAMPLED_DUAL_CHART_ROW_GROUP_PLAN_SCHEMA
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "group plan changed atomic custody or authority",
            )
        object.__setattr__(self, "registered_look_index", look)
        object.__setattr__(self, "first_prefix_look_ordinal", first)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "registered_route_chunk_size", chunk)
        object.__setattr__(self, "packet_plans", packets)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "registered_look_index": self.registered_look_index,
            "expected_prior_manifest_hash": self.expected_prior_manifest_hash,
            "first_prefix_look_ordinal": self.first_prefix_look_ordinal,
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_axis_hash": self.route_axis_hash,
            "route_id_axis_hash": self.route_id_axis_hash,
            "route_count": self.route_count,
            "corrected_a_stream_identity_hash": self.corrected_a_stream_identity_hash,
            "literal_j_stream_identity_hash": self.literal_j_stream_identity_hash,
            "packet_plan_hashes": tuple(
                packet.packet_plan_hash for packet in self.packet_plans
            ),
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "scratch_forecast_hash": self.scratch_forecast.forecast_hash,
            "all_packets_preregistered": True,
            "provider_and_reducer_proofs_required_before_publication": True,
            "one_head_publication_boundary": True,
            "raw_shards_carry_normalized_weights": False,
            "authority_scope": self.authority_scope,
        }

    @property
    def group_plan_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.scratch_forecast.assert_integrity()
        for packet in self.packet_plans:
            packet.assert_integrity()
        observed = content_hash(self._payload(), prefix=_GROUP_PLAN_PREFIX)
        if self._group_plan_hash is not None and self._group_plan_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group plan was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group plan hash differs",
            )
        object.__setattr__(self, "_group_plan_hash", observed)
        return observed


def prepare_v1_sampled_dual_chart_row_group_plan(
    *,
    registered_look_index: int,
    expected_prior_manifest_hash: str | None,
    first_prefix_look_ordinal: int,
    registered_d_prefix_hash: str,
    chart_identity_hash: str,
    current_mask_hash: str,
    proposal_generation: int,
    route_axis_hash: str,
    route_id_axis_hash: str,
    route_count: int,
    corrected_a_stream_identity_hash: str,
    literal_j_stream_identity_hash: str,
    packet_plans: Sequence[V1SampledDualChartRowGroupPacketPlan],
    registered_route_chunk_size: int,
    scratch_directory: str | os.PathLike[str],
) -> V1SampledDualChartRowGroupPlan:
    """Freeze a path-free group only after forecasting its entire write set."""

    packets = tuple(packet_plans)
    forecast = forecast_v1_sampled_dual_chart_row_group_scratch(
        route_count=route_count,
        packet_context_counts=tuple(len(packet.context_ids) for packet in packets),
        registered_route_chunk_size=registered_route_chunk_size,
        scratch_directory=scratch_directory,
    )
    return V1SampledDualChartRowGroupPlan(
        registered_look_index=registered_look_index,
        expected_prior_manifest_hash=expected_prior_manifest_hash,
        first_prefix_look_ordinal=first_prefix_look_ordinal,
        registered_d_prefix_hash=registered_d_prefix_hash,
        chart_identity_hash=chart_identity_hash,
        current_mask_hash=current_mask_hash,
        proposal_generation=proposal_generation,
        route_axis_hash=route_axis_hash,
        route_id_axis_hash=route_id_axis_hash,
        route_count=route_count,
        corrected_a_stream_identity_hash=corrected_a_stream_identity_hash,
        literal_j_stream_identity_hash=literal_j_stream_identity_hash,
        packet_plans=packets,
        registered_route_chunk_size=registered_route_chunk_size,
        scratch_forecast=forecast,
    )


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowGroupPacketProofClosure:
    """Path-free proof ledger issued after one packet's external work closes."""

    group_plan_hash: str
    packet_plan_hash: str
    packet_ordinal: int
    provider_plan_identity_hash: str
    reducer_plan_identity_hashes: tuple[str, ...]
    provider_proof_hash: str
    reducer_proof_hashes: tuple[str, ...]
    provider_closed: bool = True
    reducers_closed: bool = True
    authority_scope: str = SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
    schema: str = SAMPLED_DUAL_CHART_ROW_GROUP_PROOF_CLOSURE_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _closure_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _GROUP_PROOF_ISSUANCE_TOKEN:
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "group proof closures are issued only by the live transaction",
            )
        ordinal = _nonnegative_int(self.packet_ordinal, name="proof packet ordinal")
        reducer_plans = _ordered_strings(
            self.reducer_plan_identity_hashes,
            name="proof reducer plan identities",
            unique=True,
        )
        reducer_proofs = _ordered_strings(
            self.reducer_proof_hashes,
            name="proof reducer receipt hashes",
            unique=True,
        )
        if len(reducer_plans) != len(reducer_proofs):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "reducer plan and proof ledgers differ",
            )
        for name in (
            "group_plan_hash",
            "packet_plan_hash",
            "provider_plan_identity_hash",
            "provider_proof_hash",
        ):
            _identity(getattr(self, name), name=name.replace("_", " "))
        if (
            self.provider_closed is not True
            or self.reducers_closed is not True
            or self.authority_scope != SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
            or self.schema != SAMPLED_DUAL_CHART_ROW_GROUP_PROOF_CLOSURE_SCHEMA
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "group proof closure changed custody authority",
            )
        object.__setattr__(self, "packet_ordinal", ordinal)
        object.__setattr__(self, "reducer_plan_identity_hashes", reducer_plans)
        object.__setattr__(self, "reducer_proof_hashes", reducer_proofs)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "group_plan_hash": self.group_plan_hash,
            "packet_plan_hash": self.packet_plan_hash,
            "packet_ordinal": self.packet_ordinal,
            "provider_plan_identity_hash": self.provider_plan_identity_hash,
            "reducer_plan_identity_hashes": self.reducer_plan_identity_hashes,
            "provider_proof_hash": self.provider_proof_hash,
            "reducer_proof_hashes": self.reducer_proof_hashes,
            "provider_closed": True,
            "reducers_closed": True,
            "authority_scope": self.authority_scope,
        }

    @property
    def closure_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_GROUP_PROOF_PREFIX)
        if self._closure_hash is not None and self._closure_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group proof closure was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group proof-closure hash differs",
            )
        object.__setattr__(self, "_closure_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowLookRequest:
    """Exact causal identity of one dual-role append, plus an operational path."""

    look_ordinal: int
    expected_prior_manifest_hash: str | None
    registered_d_prefix_hash: str
    source_capture_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_labels: tuple[str, ...]
    registered_route_chunk_size: int
    scratch_forecast: V1SampledDualChartRowScratchForecast
    scratch_directory: str = field(repr=False, compare=False)
    raw_shards_carry_normalized_weights: bool = False
    authority_scope: str = SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
    schema: str = SAMPLED_DUAL_CHART_ROW_REQUEST_SCHEMA
    _request_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        look = _nonnegative_int(self.look_ordinal, name="look ordinal")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        routes = _positive_int(self.route_count, name="route count")
        contexts = _ordered_strings(self.context_ids, name="context IDs", unique=True)
        draws = _ordered_strings(self.draw_ids, name="draw IDs", unique=True)
        strata = _ordered_strings(
            self.stratum_labels,
            name="stratum labels",
            unique=False,
        )
        if len(contexts) != len(draws) or len(contexts) != len(strata):
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "context, draw, and stratum axes must have equal lengths",
            )
        if look == 0 and self.expected_prior_manifest_hash is not None:
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "the first look cannot name a predecessor manifest",
            )
        if look > 0:
            _identity(
                self.expected_prior_manifest_hash,
                name="prior manifest hash",
            )
        if self.corrected_a_stream_identity_hash == self.literal_j_stream_identity_hash:
            _refuse(
                V1SampledDualChartRowRefusalCode.ROLE_SWAP,
                "corrected-A and literal-J streams must remain distinct",
            )
        if type(self.scratch_forecast) is not V1SampledDualChartRowScratchForecast:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "request requires its exact aggregate scratch forecast",
            )
        self.scratch_forecast.assert_integrity()
        if (
            self.scratch_forecast.route_count != routes
            or self.scratch_forecast.context_count != len(contexts)
            or self.scratch_forecast.registered_route_chunk_size
            != self.registered_route_chunk_size
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "request axes differ from the aggregate scratch forecast",
            )
        _scratch_directory(self.scratch_directory)
        for name in (
            "registered_d_prefix_hash",
            "source_capture_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "corrected_a_stream_identity_hash",
            "literal_j_stream_identity_hash",
        ):
            _identity(getattr(self, name), name=name.replace("_", " "))
        if (
            self.raw_shards_carry_normalized_weights is not False
            or self.authority_scope != SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
            or self.schema != SAMPLED_DUAL_CHART_ROW_REQUEST_SCHEMA
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "request changed estimand or proposal-input authority",
            )
        object.__setattr__(self, "look_ordinal", look)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "stratum_labels", strata)
        object.__setattr__(
            self,
            "registered_route_chunk_size",
            _positive_int(
                self.registered_route_chunk_size,
                name="registered route chunk size",
            ),
        )
        object.__setattr__(
            self,
            "scratch_directory",
            str(_scratch_directory(self.scratch_directory)),
        )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "look_ordinal": self.look_ordinal,
            "expected_prior_manifest_hash": self.expected_prior_manifest_hash,
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "source_capture_identity_hash": self.source_capture_identity_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_axis_hash": self.route_axis_hash,
            "route_id_axis_hash": self.route_id_axis_hash,
            "route_count": self.route_count,
            "corrected_a_stream_identity_hash": (self.corrected_a_stream_identity_hash),
            "literal_j_stream_identity_hash": self.literal_j_stream_identity_hash,
            "context_ids": self.context_ids,
            "draw_ids": self.draw_ids,
            "stratum_labels": self.stratum_labels,
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "scratch_forecast_hash": self.scratch_forecast.forecast_hash,
            "raw_shards_carry_normalized_weights": (
                self.raw_shards_carry_normalized_weights
            ),
            "authority_scope": self.authority_scope,
        }

    @property
    def request_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.scratch_forecast.assert_integrity()
        observed = content_hash(self._payload(), prefix=_REQUEST_PREFIX)
        if self._request_hash is not None and self._request_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "look request was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "look request hash differs",
            )
        object.__setattr__(self, "_request_hash", observed)
        return observed


def prepare_v1_sampled_dual_chart_row_look(
    *,
    look_ordinal: int,
    expected_prior_manifest_hash: str | None,
    registered_d_prefix_hash: str,
    source_capture_identity_hash: str,
    chart_identity_hash: str,
    current_mask_hash: str,
    proposal_generation: int,
    route_axis_hash: str,
    route_id_axis_hash: str,
    route_count: int,
    corrected_a_stream_identity_hash: str,
    literal_j_stream_identity_hash: str,
    context_ids: Sequence[str],
    draw_ids: Sequence[str],
    stratum_labels: Sequence[str],
    registered_route_chunk_size: int,
    scratch_directory: str | os.PathLike[str],
) -> V1SampledDualChartRowLookRequest:
    """Freeze one append only after forecasting the aggregate two-shard write."""

    path = _scratch_directory(scratch_directory)
    contexts = tuple(context_ids)
    forecast = forecast_v1_sampled_dual_chart_row_scratch(
        route_count=route_count,
        context_count=len(contexts),
        registered_route_chunk_size=registered_route_chunk_size,
        scratch_directory=path,
    )
    return V1SampledDualChartRowLookRequest(
        look_ordinal=look_ordinal,
        expected_prior_manifest_hash=expected_prior_manifest_hash,
        registered_d_prefix_hash=registered_d_prefix_hash,
        source_capture_identity_hash=source_capture_identity_hash,
        chart_identity_hash=chart_identity_hash,
        current_mask_hash=current_mask_hash,
        proposal_generation=proposal_generation,
        route_axis_hash=route_axis_hash,
        route_id_axis_hash=route_id_axis_hash,
        route_count=route_count,
        corrected_a_stream_identity_hash=corrected_a_stream_identity_hash,
        literal_j_stream_identity_hash=literal_j_stream_identity_hash,
        context_ids=contexts,
        draw_ids=tuple(draw_ids),
        stratum_labels=tuple(stratum_labels),
        registered_route_chunk_size=registered_route_chunk_size,
        scratch_forecast=forecast,
        scratch_directory=str(path),
    )


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowShardReceipt:
    """Immutable path-free receipt for one raw signed role shard."""

    request_hash: str
    scratch_forecast_hash: str
    role: V1SampledDualChartRowRole
    look_ordinal: int
    registered_d_prefix_hash: str
    source_capture_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    stream_identity_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_labels: tuple[str, ...]
    registered_route_chunk_size: int
    route_windows: tuple[tuple[int, int], ...]
    exact_payload_file_bytes: int
    content_sha256: str
    source_array_kind: str
    source_device: str
    bounded_host_chunk_copy_count: int
    bounded_host_chunk_copy_bytes: int
    d2h_transfer_count: int
    d2h_transfer_bytes: int
    file_write_call_count: int
    file_write_bytes: int
    maximum_owned_host_staging_bytes: int
    preallocation_method: str
    file_layout: str = SAMPLED_DUAL_CHART_ROW_LAYOUT
    file_dtype: str = "<f8"
    every_context_route_cell_written_exactly_once: bool = True
    raw_shard_carries_normalized_weights: bool = False
    physical_scratch_path_bound: bool = False
    authority_scope: str = SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
    schema: str = SAMPLED_DUAL_CHART_ROW_SHARD_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _logical_identity_hash: str | None = field(init=False, default=None, repr=False)
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUANCE_TOKEN:
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "shard receipts are issued only by a sealed role writer",
            )
        if type(self.role) is not V1SampledDualChartRowRole:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "shard role must be exact",
            )
        routes = _positive_int(self.route_count, name="receipt route count")
        contexts = _ordered_strings(
            self.context_ids, name="receipt contexts", unique=True
        )
        draws = _ordered_strings(self.draw_ids, name="receipt draws", unique=True)
        strata = _ordered_strings(
            self.stratum_labels, name="receipt strata", unique=False
        )
        if len(contexts) != len(draws) or len(contexts) != len(strata):
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "receipt sample axes differ",
            )
        chunk_size = _positive_int(
            self.registered_route_chunk_size,
            name="receipt route chunk size",
        )
        if chunk_size > routes:
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "receipt route chunk exceeds the route axis",
            )
        registered_windows = _registered_route_window_cover(
            routes,
            chunk_size,
            self.route_windows,
            label="receipt",
        )
        exact_bytes = routes * len(contexts) * _FLOAT64_BYTES
        window_count = len(registered_windows)
        exact_payload_bytes = _nonnegative_int(
            self.exact_payload_file_bytes,
            name="receipt payload bytes",
        )
        host_copy_count = _nonnegative_int(
            self.bounded_host_chunk_copy_count,
            name="bounded host-copy count",
        )
        host_copy_bytes = _nonnegative_int(
            self.bounded_host_chunk_copy_bytes,
            name="bounded host-copy bytes",
        )
        file_write_count = _nonnegative_int(
            self.file_write_call_count,
            name="file-write call count",
        )
        file_write_bytes = _nonnegative_int(
            self.file_write_bytes,
            name="file-write bytes",
        )
        if (
            exact_payload_bytes != exact_bytes
            or host_copy_count != window_count
            or host_copy_bytes != exact_bytes
            or file_write_count < window_count
            or file_write_bytes != exact_bytes
            or self.maximum_owned_host_staging_bytes
            != len(contexts) * chunk_size * _FLOAT64_BYTES
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "shard byte or transfer telemetry does not replay",
            )
        if self.source_array_kind not in {"numpy_cpu", "torch_cpu", "torch_cuda"}:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "shard source array kind is not registered",
            )
        cuda = self.source_array_kind == "torch_cuda"
        if (
            self.source_array_kind in {"numpy_cpu", "torch_cpu"}
            and self.source_device != "cpu"
        ) or (cuda and not self.source_device.startswith("cuda")):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "source device differs from the source array kind",
            )
        if self.d2h_transfer_count != (
            window_count if cuda else 0
        ) or self.d2h_transfer_bytes != (exact_bytes if cuda else 0):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "D2H telemetry does not match the registered source device",
            )
        _sha256(self.content_sha256, name="shard content digest")
        if (
            self.file_layout != SAMPLED_DUAL_CHART_ROW_LAYOUT
            or self.file_dtype != "<f8"
            or self.every_context_route_cell_written_exactly_once is not True
            or self.raw_shard_carries_normalized_weights is not False
            or self.physical_scratch_path_bound is not False
            or self.authority_scope != SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
            or self.schema != SAMPLED_DUAL_CHART_ROW_SHARD_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "shard receipt changed raw-row custody or authority",
            )
        for name in (
            "request_hash",
            "scratch_forecast_hash",
            "registered_d_prefix_hash",
            "source_capture_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "stream_identity_hash",
            "source_array_kind",
            "source_device",
            "preallocation_method",
        ):
            _identity(getattr(self, name), name=name.replace("_", " "))
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "registered_route_chunk_size", chunk_size)
        object.__setattr__(self, "route_windows", registered_windows)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "stratum_labels", strata)
        self.assert_integrity()

    def _logical_payload(self) -> dict[str, object]:
        return {
            "schema": "partition_free_v1_sampled_dual_chart_raw_shard_v1",
            "role": self.role.value,
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "source_capture_identity_hash": self.source_capture_identity_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_axis_hash": self.route_axis_hash,
            "route_id_axis_hash": self.route_id_axis_hash,
            "route_count": self.route_count,
            "stream_identity_hash": self.stream_identity_hash,
            "context_ids": self.context_ids,
            "draw_ids": self.draw_ids,
            "stratum_labels": self.stratum_labels,
            "route_windows": self.route_windows,
            "file_layout": self.file_layout,
            "file_dtype": self.file_dtype,
            "exact_payload_file_bytes": self.exact_payload_file_bytes,
            "content_sha256": self.content_sha256,
            "raw_shard_carries_normalized_weights": False,
        }

    @property
    def logical_identity_hash(self) -> str:
        observed = content_hash(self._logical_payload(), prefix=_LOGICAL_PREFIX)
        if (
            self._logical_identity_hash is not None
            and self._logical_identity_hash != observed
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "logical shard identity was mutated",
            )
        object.__setattr__(self, "_logical_identity_hash", observed)
        return observed

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "logical_identity_hash": self.logical_identity_hash,
            "request_hash": self.request_hash,
            "scratch_forecast_hash": self.scratch_forecast_hash,
            "look_ordinal": self.look_ordinal,
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "source_array_kind": self.source_array_kind,
            "source_device": self.source_device,
            "bounded_host_chunk_copy_count": self.bounded_host_chunk_copy_count,
            "bounded_host_chunk_copy_bytes": self.bounded_host_chunk_copy_bytes,
            "d2h_transfer_count": self.d2h_transfer_count,
            "d2h_transfer_bytes": self.d2h_transfer_bytes,
            "file_write_call_count": self.file_write_call_count,
            "file_write_bytes": self.file_write_bytes,
            "maximum_owned_host_staging_bytes": self.maximum_owned_host_staging_bytes,
            "preallocation_method": self.preallocation_method,
            "every_context_route_cell_written_exactly_once": True,
            "physical_scratch_path_bound": False,
            "authority_scope": self.authority_scope,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "shard receipt was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "shard receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def to_dict(self) -> dict[str, object]:
        result = self._logical_payload() | self._payload()
        result["receipt_hash"] = self.receipt_hash
        return result


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowLookReceipt:
    look_ordinal: int
    prior_manifest_hash: str | None
    request_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_labels: tuple[str, ...]
    corrected_a_shard: V1SampledDualChartRowShardReceipt
    literal_j_shard: V1SampledDualChartRowShardReceipt
    schema: str = SAMPLED_DUAL_CHART_ROW_LOOK_SCHEMA
    _look_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.corrected_a_shard) is not V1SampledDualChartRowShardReceipt
            or type(self.literal_j_shard) is not V1SampledDualChartRowShardReceipt
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "look requires its exact two shard receipts",
            )
        a = self.corrected_a_shard
        j = self.literal_j_shard
        a.assert_integrity()
        j.assert_integrity()
        if (
            a.role is not V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY
            or j.role is not V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.ROLE_SWAP,
                "look corrected-A and literal-J shards were swapped",
            )
        shared_names = (
            "look_ordinal",
            "registered_d_prefix_hash",
            "source_capture_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "proposal_generation",
            "route_axis_hash",
            "route_id_axis_hash",
            "route_count",
            "context_ids",
            "draw_ids",
            "stratum_labels",
            "registered_route_chunk_size",
            "route_windows",
            "source_array_kind",
            "source_device",
        )
        if any(getattr(a, name) != getattr(j, name) for name in shared_names):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "dual-role shards do not share one exact capture identity",
            )
        if a.stream_identity_hash == j.stream_identity_hash:
            _refuse(
                V1SampledDualChartRowRefusalCode.ROLE_SWAP,
                "dual-role shard stream identities collapsed",
            )
        if (
            self.look_ordinal != a.look_ordinal
            or tuple(self.context_ids) != a.context_ids
            or tuple(self.draw_ids) != a.draw_ids
            or tuple(self.stratum_labels) != a.stratum_labels
            or self.request_hash != a.request_hash
            or self.request_hash != j.request_hash
            or self.schema != SAMPLED_DUAL_CHART_ROW_LOOK_SCHEMA
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "look envelope differs from its role shards",
            )
        if self.look_ordinal == 0 and self.prior_manifest_hash is not None:
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "first look cannot name a prior manifest",
            )
        if self.look_ordinal > 0:
            _identity(self.prior_manifest_hash, name="look prior manifest hash")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "look_ordinal": self.look_ordinal,
            "prior_manifest_hash": self.prior_manifest_hash,
            "request_hash": self.request_hash,
            "context_ids": self.context_ids,
            "draw_ids": self.draw_ids,
            "stratum_labels": self.stratum_labels,
            "corrected_a_shard_receipt_hash": self.corrected_a_shard.receipt_hash,
            "literal_j_shard_receipt_hash": self.literal_j_shard.receipt_hash,
        }

    @property
    def look_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.corrected_a_shard.assert_integrity()
        self.literal_j_shard.assert_integrity()
        observed = content_hash(self._payload(), prefix=_LOOK_PREFIX)
        if self._look_hash is not None and self._look_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "look receipt was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "look receipt hash differs",
            )
        object.__setattr__(self, "_look_hash", observed)
        return observed

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "look_hash": self.look_hash,
            "corrected_a_shard": self.corrected_a_shard.to_dict(),
            "literal_j_shard": self.literal_j_shard.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowPrefixManifest:
    """Append-only, predecessor-linked manifest for one sampled D prefix."""

    prior_manifest_hash: str | None
    registered_d_prefix_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    looks: tuple[V1SampledDualChartRowLookReceipt, ...]
    raw_shards_carry_normalized_weights: bool = False
    authority_scope: str = SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
    schema: str = SAMPLED_DUAL_CHART_ROW_PREFIX_MANIFEST_SCHEMA
    _manifest_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.looks or any(
            type(look) is not V1SampledDualChartRowLookReceipt for look in self.looks
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "prefix manifest requires one or more exact look receipts",
            )
        expected_ordinals = tuple(range(len(self.looks)))
        if tuple(look.look_ordinal for look in self.looks) != expected_ordinals:
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "look ordinals are not one append-only prefix",
            )
        final = self.looks[-1]
        if self.prior_manifest_hash != final.prior_manifest_hash:
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "manifest predecessor differs from the appended look",
            )
        if len(self.looks) == 1 and self.prior_manifest_hash is not None:
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "first prefix manifest cannot name a predecessor",
            )
        if len(self.looks) > 1:
            _identity(self.prior_manifest_hash, name="manifest predecessor hash")
        expected_shared = {
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_axis_hash": self.route_axis_hash,
            "route_id_axis_hash": self.route_id_axis_hash,
            "route_count": self.route_count,
        }
        for look in self.looks:
            look.assert_integrity()
            a = look.corrected_a_shard
            j = look.literal_j_shard
            if any(
                getattr(a, name) != value for name, value in expected_shared.items()
            ):
                _refuse(
                    V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                    "look departed from the prefix chart/mask/route identity",
                )
            if any(
                getattr(j, name) != value for name, value in expected_shared.items()
            ):
                _refuse(
                    V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                    "literal-J look departed from prefix identity",
                )
            if (
                a.stream_identity_hash != self.corrected_a_stream_identity_hash
                or j.stream_identity_hash != self.literal_j_stream_identity_hash
            ):
                _refuse(
                    V1SampledDualChartRowRefusalCode.ROLE_SWAP,
                    "a look changed the registered dual streams",
                )
        contexts = self.context_ids
        draws = self.draw_ids
        if len(set(contexts)) != len(contexts) or len(set(draws)) != len(draws):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "append-only prefix reused a context or draw identity",
            )
        if (
            self.corrected_a_stream_identity_hash == self.literal_j_stream_identity_hash
            or self.raw_shards_carry_normalized_weights is not False
            or self.authority_scope != SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
            or self.schema != SAMPLED_DUAL_CHART_ROW_PREFIX_MANIFEST_SCHEMA
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "manifest changed dual-stream or proposal-input authority",
            )
        expected_prior: str | None = None
        for stop, look in enumerate(self.looks, start=1):
            if look.prior_manifest_hash != expected_prior:
                _refuse(
                    V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                    "a look does not name the exact prior prefix manifest",
                )
            expected_prior = content_hash(
                self._payload_for(
                    looks=self.looks[:stop],
                    prior_manifest_hash=look.prior_manifest_hash,
                ),
                prefix=_MANIFEST_PREFIX,
            )
        self.assert_integrity()

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(value for look in self.looks for value in look.context_ids)

    @property
    def draw_ids(self) -> tuple[str, ...]:
        return tuple(value for look in self.looks for value in look.draw_ids)

    @property
    def stratum_labels(self) -> tuple[str, ...]:
        return tuple(value for look in self.looks for value in look.stratum_labels)

    @property
    def context_count(self) -> int:
        return len(self.context_ids)

    def _payload_for(
        self,
        *,
        looks: tuple[V1SampledDualChartRowLookReceipt, ...],
        prior_manifest_hash: str | None,
    ) -> dict[str, object]:
        contexts = tuple(value for look in looks for value in look.context_ids)
        draws = tuple(value for look in looks for value in look.draw_ids)
        strata = tuple(value for look in looks for value in look.stratum_labels)
        return {
            "schema": self.schema,
            "prior_manifest_hash": prior_manifest_hash,
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_axis_hash": self.route_axis_hash,
            "route_id_axis_hash": self.route_id_axis_hash,
            "route_count": self.route_count,
            "corrected_a_stream_identity_hash": self.corrected_a_stream_identity_hash,
            "literal_j_stream_identity_hash": self.literal_j_stream_identity_hash,
            "look_hashes": tuple(look.look_hash for look in looks),
            "context_ids": contexts,
            "draw_ids": draws,
            "stratum_labels": strata,
            "raw_shards_carry_normalized_weights": False,
            "authority_scope": self.authority_scope,
        }

    def _payload(self) -> dict[str, object]:
        return self._payload_for(
            looks=self.looks,
            prior_manifest_hash=self.prior_manifest_hash,
        )

    @property
    def manifest_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for look in self.looks:
            look.assert_integrity()
        observed = content_hash(self._payload(), prefix=_MANIFEST_PREFIX)
        if self._manifest_hash is not None and self._manifest_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "prefix manifest was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "prefix manifest hash differs",
            )
        object.__setattr__(self, "_manifest_hash", observed)
        return observed

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "manifest_hash": self.manifest_hash,
            "looks": tuple(look.to_dict() for look in self.looks),
        }


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowVerifiedPacket:
    """One path-free packet released after a complete one-pass custody read.

    The returned rows are a read-only little-endian fp64 copy.  They are not a
    persistence surface: the object deliberately refuses serialization and
    carries only the same inert proposal-input authority as its source shard.
    """

    shard_receipt_hash: str
    role: V1SampledDualChartRowRole
    rows: np.ndarray = field(repr=False, compare=False)
    context_count: int
    route_count: int
    route_window_count: int
    content_sha256: str
    descriptor_open_count: int
    sequential_read_pass_count: int
    descriptor_readinto_call_count: int
    sequential_payload_read_bytes: int
    digest_update_count: int
    digest_payload_bytes: int
    trailing_probe_count: int
    trailing_probe_bytes: int
    output_packet_bytes: int
    maximum_window_staging_bytes: int
    peak_owned_host_route_bytes: int
    stored_invariant_matched_before_open: bool = True
    descriptor_invariant_matched_before_read: bool = True
    descriptor_invariant_matched_after_read: bool = True
    path_invariant_matched_after_close: bool = True
    rows_released_only_after_complete_verification: bool = True
    physical_scratch_path_bound: bool = False
    authority_scope: str = SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
    schema: str = SAMPLED_DUAL_CHART_ROW_VERIFIED_PACKET_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _VERIFIED_PACKET_ISSUANCE_TOKEN:
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "verified packets are issued only by a sealed attachment read",
            )
        if type(self.role) is not V1SampledDualChartRowRole:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "verified packet role must be exact",
            )
        contexts = _positive_int(self.context_count, name="packet context count")
        routes = _positive_int(self.route_count, name="packet route count")
        windows = _positive_int(
            self.route_window_count,
            name="packet route-window count",
        )
        rows = np.asarray(self.rows)
        exact_bytes = contexts * routes * _FLOAT64_BYTES
        if (
            rows.dtype != np.dtype("<f8")
            or rows.shape != (contexts, routes)
            or bool(rows.flags.writeable)
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "verified packet rows are not read-only little-endian fp64",
            )
        for name in (
            "descriptor_open_count",
            "sequential_read_pass_count",
            "descriptor_readinto_call_count",
            "sequential_payload_read_bytes",
            "digest_update_count",
            "digest_payload_bytes",
            "trailing_probe_count",
            "trailing_probe_bytes",
            "output_packet_bytes",
            "maximum_window_staging_bytes",
            "peak_owned_host_route_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
            )
        if (
            self.descriptor_open_count != 1
            or self.sequential_read_pass_count != 1
            or self.descriptor_readinto_call_count < windows
            or self.sequential_payload_read_bytes != exact_bytes
            or self.digest_update_count != windows
            or self.digest_payload_bytes != exact_bytes
            or self.trailing_probe_count != 1
            or self.trailing_probe_bytes != 0
            or self.output_packet_bytes != exact_bytes
            or not 0 < self.maximum_window_staging_bytes <= exact_bytes
            or self.peak_owned_host_route_bytes
            != exact_bytes + self.maximum_window_staging_bytes
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "verified packet one-pass byte/window telemetry does not replay",
            )
        _identity(self.shard_receipt_hash, name="verified packet shard receipt hash")
        _sha256(self.content_sha256, name="verified packet content digest")
        if (
            self.stored_invariant_matched_before_open is not True
            or self.descriptor_invariant_matched_before_read is not True
            or self.descriptor_invariant_matched_after_read is not True
            or self.path_invariant_matched_after_close is not True
            or self.rows_released_only_after_complete_verification is not True
            or self.physical_scratch_path_bound is not False
            or self.authority_scope != SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
            or self.schema != SAMPLED_DUAL_CHART_ROW_VERIFIED_PACKET_SCHEMA
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "verified packet changed its custody or authority boundary",
            )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "context_count", contexts)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "route_window_count", windows)

    def __reduce__(self) -> NoReturn:
        raise TypeError("verified raw-row packets are live and nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("verified raw-row packets are live and nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowShardVerificationTelemetry:
    """Non-authoritative counters separating hot witnesses from byte reads."""

    issuing_process_id: int
    content_admission: str
    live_witness_verification_count: int
    live_witness_payload_read_count: int
    live_witness_payload_read_bytes: int
    cold_content_verification_count: int
    cold_content_payload_read_bytes: int
    one_pass_content_verification_count: int
    one_pass_payload_read_bytes: int
    poisoned: bool


class V1SampledDualChartRowShardAttachment:
    """Nonserializable process-local lease on one immutable sealed shard.

    A hot lease is admitted either by the writer's already-completed streaming
    digest or by one explicit cold role-domain digest.  After admission,
    :meth:`verify_live_witnesses` performs only receipt and filesystem metadata
    checks; it never opens or reads the payload.  The PID and complete immutable
    file witness make a fork, chmod, replacement, size, uid, ctime, or mtime
    drift fail closed and permanently poison the attachment.
    """

    __slots__ = (
        "_path",
        "_receipt",
        "_invariant",
        "_issuing_pid",
        "_integrity_failed",
        "_content_admission",
        "_live_witness_verification_count",
        "_cold_content_verification_count",
        "_cold_content_payload_read_bytes",
        "_one_pass_content_verification_count",
        "_one_pass_payload_read_bytes",
    )

    def __init__(
        self,
        *,
        path: Path,
        receipt: V1SampledDualChartRowShardReceipt,
        streaming_digest_verified: bool = False,
        _streaming_token: object | None = None,
        cold_content_digest_verified: bool = False,
        _cold_token: object | None = None,
    ) -> None:
        if type(receipt) is not V1SampledDualChartRowShardReceipt:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "live attachment requires an exact shard receipt",
            )
        receipt.assert_integrity()
        resolved = path.resolve()
        try:
            invariant = _file_invariant(resolved)
        except OSError as error:
            raise V1SampledDualChartRowRefusal(
                V1SampledDualChartRowRefusalCode.MISSING_SHARD,
                "sealed shard file is missing",
            ) from error
        if invariant.size != receipt.exact_payload_file_bytes:
            _refuse(
                V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                "sealed shard has the wrong byte size",
            )
        if streaming_digest_verified and cold_content_digest_verified:
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "one live attachment cannot claim two content-admission paths",
            )
        if (
            streaming_digest_verified
            and _streaming_token is not _STREAMING_ATTACHMENT_TOKEN
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "only the sealed writer may attest its streaming payload digest",
            )
        if cold_content_digest_verified and _cold_token is not _COLD_ATTACHMENT_TOKEN:
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "only an exact cold digest may attest reattached shard content",
            )

        # Legacy durable shards may predate the read-only seal.  Preserve
        # restart compatibility by hashing such a shard exactly once before
        # stripping every write bit; no hot lease exists before both steps.
        cold_verified_before_attachment = False
        if not _is_read_only_regular_file(invariant):
            if streaming_digest_verified or cold_content_digest_verified:
                _refuse(
                    V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                    "pre-attested shard is not one read-only regular file",
                )
            if _file_digest(resolved, receipt.role) != receipt.content_sha256:
                _refuse(
                    V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                    "legacy shard digest differs from its path-free receipt",
                )
            try:
                invariant = _seal_file_read_only(resolved)
            except OSError as error:
                raise V1SampledDualChartRowRefusal(
                    V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                    "legacy shard could not be sealed read-only after cold digest",
                ) from error
            cold_verified_before_attachment = True
        if not _is_read_only_regular_file(invariant):
            _refuse(
                V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                "sealed shard must be one read-only regular file",
            )
        self._path = resolved
        self._receipt = receipt
        self._invariant = invariant
        self._issuing_pid = int(os.getpid())
        self._integrity_failed = False
        self._live_witness_verification_count = 0
        self._cold_content_verification_count = 0
        self._cold_content_payload_read_bytes = 0
        self._one_pass_content_verification_count = 0
        self._one_pass_payload_read_bytes = 0
        if streaming_digest_verified:
            self._content_admission = "writer_streaming_digest"
        elif cold_content_digest_verified or cold_verified_before_attachment:
            self._content_admission = "cold_full_role_domain_sha256"
            self._cold_content_verification_count = 1
            self._cold_content_payload_read_bytes = receipt.exact_payload_file_bytes
        else:
            self._content_admission = "cold_full_role_domain_sha256"
            self.verify_cold_content()

    @property
    def receipt(self) -> V1SampledDualChartRowShardReceipt:
        self._receipt.assert_integrity()
        return self._receipt

    @property
    def role(self) -> V1SampledDualChartRowRole:
        return self._receipt.role

    @property
    def physical_path(self) -> Path:
        self.verify_live_witnesses()
        return self._path

    @property
    def verification_telemetry(
        self,
    ) -> V1SampledDualChartRowShardVerificationTelemetry:
        """Return deterministic process-local verification counters."""

        return V1SampledDualChartRowShardVerificationTelemetry(
            issuing_process_id=self._issuing_pid,
            content_admission=self._content_admission,
            live_witness_verification_count=self._live_witness_verification_count,
            live_witness_payload_read_count=0,
            live_witness_payload_read_bytes=0,
            cold_content_verification_count=self._cold_content_verification_count,
            cold_content_payload_read_bytes=self._cold_content_payload_read_bytes,
            one_pass_content_verification_count=(
                self._one_pass_content_verification_count
            ),
            one_pass_payload_read_bytes=self._one_pass_payload_read_bytes,
            poisoned=self._integrity_failed,
        )

    def verify_live_witnesses(
        self,
        *,
        expected_role: V1SampledDualChartRowRole | None = None,
        expected_chart_identity_hash: str | None = None,
        expected_current_mask_hash: str | None = None,
        expected_proposal_generation: int | None = None,
    ) -> None:
        """Verify the PID, receipt, role chain, and stat witness without reads."""

        if self._integrity_failed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                "shard attachment previously failed custody verification",
            )
        receipt = self._receipt
        receipt.assert_integrity()
        if expected_role is not None and receipt.role is not expected_role:
            _refuse(
                V1SampledDualChartRowRefusalCode.ROLE_SWAP,
                "corrected-A and literal-J live attachments are not interchangeable",
            )
        if (
            expected_chart_identity_hash is not None
            and receipt.chart_identity_hash != expected_chart_identity_hash
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "live shard belongs to a stale chart",
            )
        if (
            expected_current_mask_hash is not None
            and receipt.current_mask_hash != expected_current_mask_hash
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "live shard belongs to a stale mask",
            )
        if (
            expected_proposal_generation is not None
            and receipt.proposal_generation != expected_proposal_generation
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "live shard belongs to a stale proposal generation",
            )
        if int(os.getpid()) != self._issuing_pid:
            self._integrity_failed = True
            _refuse(
                V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                "live shard lease crossed its issuing process boundary",
            )
        try:
            invariant = _file_invariant(self._path)
        except OSError as error:
            self._integrity_failed = True
            raise V1SampledDualChartRowRefusal(
                V1SampledDualChartRowRefusalCode.MISSING_SHARD,
                "sealed shard disappeared",
            ) from error
        if invariant != self._invariant or not _is_read_only_regular_file(invariant):
            self._integrity_failed = True
            _refuse(
                V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                "sealed shard device/inode/size/uid/mode/ctime/mtime witness changed",
            )
        self._live_witness_verification_count += 1

    def verify_cold_content(
        self,
        *,
        expected_role: V1SampledDualChartRowRole | None = None,
        expected_chart_identity_hash: str | None = None,
        expected_current_mask_hash: str | None = None,
        expected_proposal_generation: int | None = None,
    ) -> None:
        """Explicitly perform one complete role-domain SHA-256 verification."""

        self.verify_live_witnesses(
            expected_role=expected_role,
            expected_chart_identity_hash=expected_chart_identity_hash,
            expected_current_mask_hash=expected_current_mask_hash,
            expected_proposal_generation=expected_proposal_generation,
        )
        receipt = self._receipt
        try:
            observed = _file_digest(self._path, receipt.role)
        except OSError as error:
            self._integrity_failed = True
            raise V1SampledDualChartRowRefusal(
                V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                "sealed shard could not complete cold content verification",
            ) from error
        if observed != receipt.content_sha256:
            self._integrity_failed = True
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "sealed shard bytes changed",
            )
        # Reading may alter atime, which is deliberately outside the witness;
        # every immutability-bearing field must remain identical.
        self.verify_live_witnesses(
            expected_role=expected_role,
            expected_chart_identity_hash=expected_chart_identity_hash,
            expected_current_mask_hash=expected_current_mask_hash,
            expected_proposal_generation=expected_proposal_generation,
        )
        self._cold_content_verification_count += 1
        self._cold_content_payload_read_bytes += receipt.exact_payload_file_bytes

    def verify(
        self,
        *,
        expected_role: V1SampledDualChartRowRole | None = None,
        expected_chart_identity_hash: str | None = None,
        expected_current_mask_hash: str | None = None,
        expected_proposal_generation: int | None = None,
        rehash_payload: bool = True,
    ) -> None:
        """Legacy compatibility shim for callers selecting hot versus cold."""

        verifier = (
            self.verify_cold_content if rehash_payload else self.verify_live_witnesses
        )
        verifier(
            expected_role=expected_role,
            expected_chart_identity_hash=expected_chart_identity_hash,
            expected_current_mask_hash=expected_current_mask_hash,
            expected_proposal_generation=expected_proposal_generation,
        )

    def materialize_verified_packet_one_pass(
        self,
    ) -> V1SampledDualChartRowVerifiedPacket:
        """Return one exact packet after a single sequential digesting read.

        Unlike ``verify`` followed by ``iter_route_windows``, this operation
        reads the payload once.  The complete output matrix remains private
        and writable until the descriptor/path invariants, exact byte cover,
        trailing-byte probe, and role-domain digest have all closed.  Digest
        equality re-establishes the exact bytes the sealed writer admitted
        only after its finite-fp64 check, without another packet-sized boolean
        workspace.  Any custody mismatch poisons this live attachment.
        """

        if self._integrity_failed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                "shard attachment previously failed custody verification",
            )
        receipt = self._receipt
        try:
            self.verify_live_witnesses()
            path_before = _file_invariant(self._path)
            if path_before != self._invariant:
                _refuse(
                    V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                    "sealed shard invariant changed before one-pass read",
                )
            contexts = len(receipt.context_ids)
            routes = receipt.route_count
            rows = np.empty((contexts, routes), dtype=np.dtype("<f8"))
            digest = _new_payload_hasher(receipt.role)
            readinto_calls = 0
            payload_bytes = 0
            maximum_window_bytes = 0
            expected_start = 0
            with self._path.open("rb", buffering=0) as stream:
                descriptor_before = _descriptor_invariant(stream.fileno())
                if descriptor_before != self._invariant:
                    _refuse(
                        V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                        "opened descriptor differs from sealed shard custody",
                    )
                for start, stop in receipt.route_windows:
                    if start != expected_start:
                        _refuse(
                            V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                            "sealed shard route windows are not sequential",
                        )
                    width = stop - start
                    expected_bytes = contexts * width * _FLOAT64_BYTES
                    raw = bytearray(expected_bytes)
                    target = memoryview(raw)
                    cursor = 0
                    while cursor < expected_bytes:
                        observed = stream.readinto(target[cursor:])
                        readinto_calls += 1
                        if observed is None or observed <= 0:
                            _refuse(
                                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                                "sealed shard ended before its registered packet cover",
                            )
                        cursor += int(observed)
                    digest.update(raw)
                    window = np.frombuffer(raw, dtype=np.dtype("<f8")).reshape(
                        contexts,
                        width,
                    )
                    rows[:, start:stop] = window
                    expected_start = stop
                    payload_bytes += expected_bytes
                    maximum_window_bytes = max(maximum_window_bytes, expected_bytes)
                trailing = stream.read(1)
                descriptor_after = _descriptor_invariant(stream.fileno())
            path_after = _file_invariant(self._path)
            if descriptor_after != descriptor_before:
                _refuse(
                    V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                    "sealed shard descriptor invariant changed during one-pass read",
                )
            if path_after != self._invariant:
                _refuse(
                    V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                    "sealed shard path invariant changed during one-pass read",
                )
            if (
                expected_start != routes
                or payload_bytes != receipt.exact_payload_file_bytes
            ):
                _refuse(
                    V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                    "one-pass packet read did not cover the exact registered payload",
                )
            if trailing:
                _refuse(
                    V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                    "sealed shard contains trailing bytes",
                )
            observed_digest = digest.hexdigest()
            if observed_digest != receipt.content_sha256:
                _refuse(
                    V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                    "one-pass role-domain digest differs from the sealed shard",
                )
        except V1SampledDualChartRowRefusal:
            self._integrity_failed = True
            raise
        except OSError as error:
            self._integrity_failed = True
            raise V1SampledDualChartRowRefusal(
                V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                "sealed shard could not complete one descriptor-bound read",
            ) from error

        rows.setflags(write=False)
        self._one_pass_content_verification_count += 1
        self._one_pass_payload_read_bytes += payload_bytes
        return V1SampledDualChartRowVerifiedPacket(
            shard_receipt_hash=receipt.receipt_hash,
            role=receipt.role,
            rows=rows,
            context_count=contexts,
            route_count=routes,
            route_window_count=len(receipt.route_windows),
            content_sha256=observed_digest,
            descriptor_open_count=1,
            sequential_read_pass_count=1,
            descriptor_readinto_call_count=readinto_calls,
            sequential_payload_read_bytes=payload_bytes,
            digest_update_count=len(receipt.route_windows),
            digest_payload_bytes=payload_bytes,
            trailing_probe_count=1,
            trailing_probe_bytes=len(trailing),
            output_packet_bytes=int(rows.nbytes),
            maximum_window_staging_bytes=maximum_window_bytes,
            peak_owned_host_route_bytes=int(rows.nbytes) + maximum_window_bytes,
            _issuance_token=_VERIFIED_PACKET_ISSUANCE_TOKEN,
        )

    def iter_route_windows(
        self,
        *,
        verify: bool = True,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield bounded read-only raw row windows in registered order."""

        if verify:
            self.verify_live_witnesses()
        receipt = self._receipt
        with self._path.open("rb", buffering=0) as stream:
            for start, stop in receipt.route_windows:
                width = stop - start
                cells = len(receipt.context_ids) * width
                expected_bytes = cells * _FLOAT64_BYTES
                raw = bytearray(expected_bytes)
                target = memoryview(raw)
                cursor = 0
                while cursor < expected_bytes:
                    observed = stream.readinto(target[cursor:])
                    if observed is None or observed <= 0:
                        _refuse(
                            V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                            "sealed shard ended before its registered route window",
                        )
                    cursor += int(observed)
                array = np.frombuffer(raw, dtype=np.dtype("<f8")).reshape(
                    len(receipt.context_ids),
                    width,
                )
                array.setflags(write=False)
                yield start, stop, array
            if stream.read(1):
                _refuse(
                    V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                    "sealed shard contains trailing bytes",
                )

    def materialize_raw_rows(self, *, verify: bool = True) -> np.ndarray:
        result = np.empty(
            (len(self._receipt.context_ids), self._receipt.route_count),
            dtype=np.float64,
        )
        for start, stop, window in self.iter_route_windows(verify=verify):
            result[:, start:stop] = window
        result.setflags(write=False)
        return result

    def __reduce__(self) -> NoReturn:
        raise TypeError("live shard attachments are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live shard attachments are nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowLivePrefixAttachment:
    """Live verified attachment for every role shard in one published prefix."""

    manifest: V1SampledDualChartRowPrefixManifest
    corrected_a_attachments: tuple[V1SampledDualChartRowShardAttachment, ...]
    literal_j_attachments: tuple[V1SampledDualChartRowShardAttachment, ...]

    def __post_init__(self) -> None:
        if type(self.manifest) is not V1SampledDualChartRowPrefixManifest:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "live prefix requires an exact manifest",
            )
        count = len(self.manifest.looks)
        if (
            len(self.corrected_a_attachments) != count
            or len(self.literal_j_attachments) != count
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.MISSING_SHARD,
                "live prefix does not attach both roles for every look",
            )
        self.verify_live_witnesses()

    def _verify_role_chain(self) -> None:
        self.manifest.assert_integrity()
        for index, look in enumerate(self.manifest.looks):
            a = self.corrected_a_attachments[index]
            j = self.literal_j_attachments[index]
            if (
                a.receipt.receipt_hash != look.corrected_a_shard.receipt_hash
                or j.receipt.receipt_hash != look.literal_j_shard.receipt_hash
            ):
                _refuse(
                    V1SampledDualChartRowRefusalCode.ROLE_SWAP,
                    "live prefix attachment order or roles differ from the manifest",
                )

    def verify_live_witnesses(self) -> None:
        """Verify the complete manifest/role/stat chain with zero payload reads."""

        self._verify_role_chain()
        for index, _look in enumerate(self.manifest.looks):
            a = self.corrected_a_attachments[index]
            j = self.literal_j_attachments[index]
            a.verify_live_witnesses(
                expected_role=V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY,
                expected_chart_identity_hash=self.manifest.chart_identity_hash,
                expected_current_mask_hash=self.manifest.current_mask_hash,
                expected_proposal_generation=self.manifest.proposal_generation,
            )
            j.verify_live_witnesses(
                expected_role=V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN,
                expected_chart_identity_hash=self.manifest.chart_identity_hash,
                expected_current_mask_hash=self.manifest.current_mask_hash,
                expected_proposal_generation=self.manifest.proposal_generation,
            )

    def verify_cold_content(self) -> None:
        """Explicitly SHA-256 every A/J role shard in manifest order."""

        self._verify_role_chain()
        for index, _look in enumerate(self.manifest.looks):
            a = self.corrected_a_attachments[index]
            j = self.literal_j_attachments[index]
            a.verify_cold_content(
                expected_role=V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY,
                expected_chart_identity_hash=self.manifest.chart_identity_hash,
                expected_current_mask_hash=self.manifest.current_mask_hash,
                expected_proposal_generation=self.manifest.proposal_generation,
            )
            j.verify_cold_content(
                expected_role=V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN,
                expected_chart_identity_hash=self.manifest.chart_identity_hash,
                expected_current_mask_hash=self.manifest.current_mask_hash,
                expected_proposal_generation=self.manifest.proposal_generation,
            )

    def verify(self, *, rehash_payload: bool = True) -> None:
        """Legacy compatibility shim retaining the historical switch."""

        if rehash_payload:
            self.verify_cold_content()
        else:
            self.verify_live_witnesses()

    def __reduce__(self) -> NoReturn:
        raise TypeError("live prefix attachments are nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledDualChartMaterializedRows:
    """One design-specific streaming view; raw shard identity stays unchanged."""

    attachment: V1SampledDualChartRowLivePrefixAttachment = field(
        repr=False,
        compare=False,
    )
    manifest_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_labels: tuple[str, ...]
    stratum_masses: tuple[tuple[str, float], ...]
    stratum_sample_counts: tuple[tuple[str, int], ...]
    normalized_context_weights: np.ndarray = field(repr=False, compare=False)
    raw_shards_mutated: bool = False
    authority_scope: str = SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
    schema: str = SAMPLED_DUAL_CHART_ROW_MATERIALIZATION_SCHEMA
    _materialization_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.attachment) is not V1SampledDualChartRowLivePrefixAttachment:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "materialized design requires one exact live prefix attachment",
            )
        self.attachment.verify_live_witnesses()
        manifest = self.attachment.manifest
        if (
            self.manifest_hash != manifest.manifest_hash
            or tuple(self.context_ids) != manifest.context_ids
            or tuple(self.draw_ids) != manifest.draw_ids
            or tuple(self.stratum_labels) != manifest.stratum_labels
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "materialized design differs from its live raw-row prefix",
            )
        count = len(self.context_ids)
        weights = np.asarray(self.normalized_context_weights)
        if (
            weights.dtype != np.float64
            or weights.shape != (count,)
            or not bool(np.isfinite(weights).all())
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "deterministic materialization weights are invalid",
            )
        if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "stratified materialization weights do not sum to one",
            )
        if (
            self.raw_shards_mutated is not False
            or self.authority_scope != SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
            or self.schema != SAMPLED_DUAL_CHART_ROW_MATERIALIZATION_SCHEMA
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "materialization claimed mutation or decision authority",
            )
        weights.setflags(write=False)
        object.__setattr__(self, "normalized_context_weights", weights)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "manifest_hash": self.manifest_hash,
            "context_ids": self.context_ids,
            "draw_ids": self.draw_ids,
            "stratum_labels": self.stratum_labels,
            "stratum_masses": self.stratum_masses,
            "stratum_sample_counts": self.stratum_sample_counts,
            "normalized_context_weights": self.normalized_context_weights,
            "raw_shards_mutated": False,
            "authority_scope": self.authority_scope,
        }

    @property
    def materialization_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_MATERIALIZATION_PREFIX)
        if (
            self._materialization_hash is not None
            and self._materialization_hash != observed
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "materialization identity was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "materialization hash differs",
            )
        object.__setattr__(self, "_materialization_hash", observed)
        return observed

    def dense_signed_work_value(self, action: Any) -> float:
        """Stream exact per-context signed cancellation, then absolute work.

        Only one projection scalar per sampled context is retained; no
        context-by-route row bank is reconstructed.
        """

        z = np.asarray(action, dtype=np.float64)
        route_count = self.attachment.manifest.route_count
        if z.shape != (route_count,) or not bool(np.isfinite(z).all()):
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "action must be one finite vector on the registered route axis",
            )
        self.attachment.verify_live_witnesses()
        projections = np.zeros(len(self.context_ids), dtype=np.float64)
        context_start = 0
        for shard in self.attachment.corrected_a_attachments:
            context_stop = context_start + len(shard.receipt.context_ids)
            for start, stop, rows in shard.iter_route_windows(verify=False):
                projections[context_start:context_stop] += rows @ z[start:stop]
            context_start = context_stop
        return float(np.dot(self.normalized_context_weights, np.abs(projections)))

    def materialize_raw_rows_for_reference(
        self,
        *,
        role: V1SampledDualChartRowRole,
        maximum_cells: int,
    ) -> np.ndarray:
        """Guarded small-domain reference reconstruction, never production fallback."""

        if type(role) is not V1SampledDualChartRowRole:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "reference materialization requires an exact row role",
            )
        maximum = _positive_int(maximum_cells, name="reference maximum cells")
        required = len(self.context_ids) * self.attachment.manifest.route_count
        if required > maximum:
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "full raw-row materialization exceeded its explicit reference bound",
            )
        self.attachment.verify_live_witnesses()
        shards = (
            self.attachment.corrected_a_attachments
            if role is V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY
            else self.attachment.literal_j_attachments
        )
        result = np.concatenate(
            [item.materialize_raw_rows(verify=False) for item in shards],
            axis=0,
        )
        result.setflags(write=False)
        return result

    def __reduce__(self) -> NoReturn:
        raise TypeError("live weighted prefix views are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live weighted prefix views are nonserializable")


def _source_descriptor_for_shape(
    rows: Any,
    *,
    shape: tuple[int, int],
) -> tuple[str, str]:
    if isinstance(rows, np.ndarray):
        if rows.shape != shape or rows.dtype != np.float64:
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "NumPy row source must be exact context-by-route float64",
            )
        return "numpy_cpu", "cpu"
    if torch is not None and torch.is_tensor(rows):
        if (
            tuple(rows.shape) != shape
            or rows.dtype != torch.float64
            or rows.requires_grad
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "Torch row source must be detached context-by-route float64",
            )
        if rows.device.type not in {"cpu", "cuda"}:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "Torch row sources are registered only on CPU or CUDA",
            )
        return ("torch_cuda" if rows.device.type == "cuda" else "torch_cpu"), str(
            rows.device
        )
    _refuse(
        V1SampledDualChartRowRefusalCode.INVALID_TYPE,
        "row source must be one NumPy array or Torch tensor",
    )


def _source_descriptor(
    rows: Any,
    request: V1SampledDualChartRowLookRequest,
) -> tuple[str, str]:
    return _source_descriptor_for_shape(
        rows,
        shape=(len(request.context_ids), request.route_count),
    )


def _host_rows(rows: Any) -> np.ndarray:
    if isinstance(rows, np.ndarray):
        # Always take one bounded owned snapshot so later provider mutation
        # cannot race the file write and the copy telemetry is literal.
        result = np.array(rows, dtype=np.dtype("<f8"), order="C", copy=True)
    else:
        tensor = rows.contiguous().detach()
        if tensor.device.type == "cuda":
            tensor = tensor.to(device="cpu", non_blocking=False)
            # The NumPy view retains the owned D2H tensor as its base.
            result = np.asarray(tensor.numpy(), dtype=np.dtype("<f8"), order="C")
        else:
            result = np.array(
                tensor.numpy(),
                dtype=np.dtype("<f8"),
                order="C",
                copy=True,
            )
    if not bool(np.isfinite(result).all()):
        _refuse(
            V1SampledDualChartRowRefusalCode.NONFINITE_ROW,
            "raw signed row shard contains a non-finite value",
        )
    return result


def _shard_path(directory: Path, receipt_hash: str) -> Path:
    token = _content_address_token(
        receipt_hash,
        prefix=_RECEIPT_PREFIX,
        name="sampled dual-row shard receipt hash",
    )
    return directory / f"sampled-dual-chart-shard-{token}.f64"


def _manifest_path(directory: Path, manifest_hash: str) -> Path:
    token = _content_address_token(
        manifest_hash,
        prefix=_MANIFEST_PREFIX,
        name="sampled dual-row manifest hash",
    )
    return directory / f"sampled-dual-chart-manifest-{token}.json"


class _OpenRoleShardWriter:
    """One bounded role writer owned by an atomic dual-role transaction."""

    __slots__ = (
        "request",
        "role",
        "_partial",
        "_sealed",
        "_stream",
        "_preallocation",
        "_hasher",
        "_source_descriptor",
        "_next_route",
        "_route_windows",
        "_write_calls",
        "_write_bytes",
        "_host_copy_count",
        "_host_copy_bytes",
        "_d2h_count",
        "_d2h_bytes",
        "_staged_receipt",
        "_closed",
    )

    def __init__(
        self,
        *,
        request: V1SampledDualChartRowLookRequest,
        role: V1SampledDualChartRowRole,
    ) -> None:
        directory = Path(request.scratch_directory)
        descriptor = -1
        partial: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(
                prefix=f".sampled-dual-chart-{role.name.lower()}-",
                suffix=".partial",
                dir=directory,
            )
            partial = Path(raw_path)
            exact_bytes = request.scratch_forecast.exact_bytes_per_role_shard
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(descriptor, 0, exact_bytes)  # type: ignore[attr-defined]
                preallocation = "posix_fallocate"
            else:  # pragma: no cover - Linux production has posix_fallocate.
                os.ftruncate(descriptor, exact_bytes)
                preallocation = "ftruncate_exact_size_fallback"
            os.lseek(descriptor, 0, os.SEEK_SET)
            stream = os.fdopen(descriptor, "r+b", buffering=0)
            descriptor = -1
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            if partial is not None:
                partial.unlink(missing_ok=True)
            raise
        self.request = request
        self.role = role
        self._partial = partial
        self._sealed: Path | None = None
        self._stream: Any | None = stream
        self._preallocation = preallocation
        self._hasher = _new_payload_hasher(role)
        self._source_descriptor: tuple[str, str] | None = None
        self._next_route = 0
        self._route_windows: list[tuple[int, int]] = []
        self._write_calls = 0
        self._write_bytes = 0
        self._host_copy_count = 0
        self._host_copy_bytes = 0
        self._d2h_count = 0
        self._d2h_bytes = 0
        self._staged_receipt: V1SampledDualChartRowShardReceipt | None = None
        self._closed = False

    def consume(self, *, start: int, stop: int, rows: Any) -> tuple[str, str]:
        if self._closed or self._stream is None:
            raise RuntimeError("sampled role writer is already closed")
        if (
            start != self._next_route
            or stop <= start
            or stop > self.request.route_count
            or stop - start > self.request.registered_route_chunk_size
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                "dual-row chunks must form a bounded contiguous route cover",
            )
        descriptor = _source_descriptor_for_shape(
            rows,
            shape=(len(self.request.context_ids), stop - start),
        )
        if self._source_descriptor is None:
            self._source_descriptor = descriptor
        elif self._source_descriptor != descriptor:
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "one raw role changed source device during its look",
            )
        host = _host_rows(rows)
        view = memoryview(host).cast("B")
        cursor = 0
        while cursor < len(view):
            written = self._stream.write(view[cursor:])
            self._write_calls += 1
            if written is None or written <= 0:
                raise OSError("short write while sealing sampled dual rows")
            cursor += int(written)
            self._write_bytes += int(written)
        self._hasher.update(view)
        self._host_copy_count += 1
        self._host_copy_bytes += len(view)
        if descriptor[0] == "torch_cuda":
            self._d2h_count += 1
            self._d2h_bytes += len(view)
        self._route_windows.append((start, stop))
        self._next_route = stop
        return descriptor

    def stage_hidden(self) -> V1SampledDualChartRowShardReceipt:
        """Seal bytes while retaining the random, HEAD-invisible stage path."""

        if self._closed or self._stream is None or self._partial is None:
            raise RuntimeError("sampled role writer cannot be sealed twice")
        exact_bytes = self.request.scratch_forecast.exact_bytes_per_role_shard
        if (
            self._next_route != self.request.route_count
            or self._source_descriptor is None
            or self._write_bytes != exact_bytes
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "cannot seal a partial sampled role shard",
            )
        self._stream.flush()
        os.fsync(self._stream.fileno())
        if self._stream.tell() != exact_bytes:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "sealed role write cursor differs from the exact forecast",
            )
        # The streaming role-domain digest is complete and the exact bytes are
        # durable before any hot lease can exist.  Strip every write bit while
        # the writer still owns the descriptor, fsync that metadata transition,
        # and only then permit atomic publication.
        os.fchmod(self._stream.fileno(), _SEALED_FILE_MODE)
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._stream = None
        source_kind, source_device = self._source_descriptor
        stream_hash = (
            self.request.corrected_a_stream_identity_hash
            if self.role is V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY
            else self.request.literal_j_stream_identity_hash
        )
        receipt = V1SampledDualChartRowShardReceipt(
            request_hash=self.request.request_hash,
            scratch_forecast_hash=self.request.scratch_forecast.forecast_hash,
            role=self.role,
            look_ordinal=self.request.look_ordinal,
            registered_d_prefix_hash=self.request.registered_d_prefix_hash,
            source_capture_identity_hash=self.request.source_capture_identity_hash,
            chart_identity_hash=self.request.chart_identity_hash,
            current_mask_hash=self.request.current_mask_hash,
            proposal_generation=self.request.proposal_generation,
            route_axis_hash=self.request.route_axis_hash,
            route_id_axis_hash=self.request.route_id_axis_hash,
            route_count=self.request.route_count,
            stream_identity_hash=stream_hash,
            context_ids=self.request.context_ids,
            draw_ids=self.request.draw_ids,
            stratum_labels=self.request.stratum_labels,
            registered_route_chunk_size=self.request.registered_route_chunk_size,
            route_windows=tuple(self._route_windows),
            exact_payload_file_bytes=exact_bytes,
            content_sha256=self._hasher.hexdigest(),
            source_array_kind=source_kind,
            source_device=source_device,
            bounded_host_chunk_copy_count=self._host_copy_count,
            bounded_host_chunk_copy_bytes=self._host_copy_bytes,
            d2h_transfer_count=self._d2h_count,
            d2h_transfer_bytes=self._d2h_bytes,
            file_write_call_count=self._write_calls,
            file_write_bytes=self._write_bytes,
            maximum_owned_host_staging_bytes=(
                self.request.scratch_forecast.maximum_owned_host_staging_bytes
            ),
            preallocation_method=self._preallocation,
            _issuance_token=_ISSUANCE_TOKEN,
        )
        self._staged_receipt = receipt
        self._closed = True
        return receipt

    def publish_staged(
        self,
    ) -> tuple[
        V1SampledDualChartRowShardReceipt,
        V1SampledDualChartRowShardAttachment,
        Path,
    ]:
        """Move one hidden shard to its content-addressed path.

        An exact orphan from a process crash is adopted only after its role-
        domain payload digest and byte count replay.  This makes an exact
        retry idempotent without allowing unrelated orphan bytes to acquire
        authority.
        """

        receipt = self._staged_receipt
        if not self._closed or receipt is None or self._partial is None:
            raise RuntimeError("sampled role writer is not hidden-staged")
        sealed = _shard_path(Path(self.request.scratch_directory), receipt.receipt_hash)
        adopted_existing = False
        if sealed.exists():
            try:
                invariant = _file_invariant(sealed)
            except OSError as error:  # pragma: no cover - stat race under lock.
                raise V1SampledDualChartRowRefusal(
                    V1SampledDualChartRowRefusalCode.PUBLICATION_CONFLICT,
                    "content-addressed orphan shard could not be inspected",
                ) from error
            if (
                invariant.size != receipt.exact_payload_file_bytes
                or _file_digest(sealed, receipt.role) != receipt.content_sha256
            ):
                _refuse(
                    V1SampledDualChartRowRefusalCode.PUBLICATION_CONFLICT,
                    "existing content-addressed shard is not the exact retry orphan",
                )
            if not _is_read_only_regular_file(invariant):
                try:
                    invariant = _seal_file_read_only(sealed)
                except OSError as error:
                    raise V1SampledDualChartRowRefusal(
                        V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                        "exact retry orphan could not be sealed read-only",
                    ) from error
            self._partial.unlink(missing_ok=True)
            adopted_existing = True
        else:
            os.replace(self._partial, sealed)
        self._partial = None
        self._sealed = sealed
        _fsync_directory(Path(self.request.scratch_directory))
        attachment = V1SampledDualChartRowShardAttachment(
            path=sealed,
            receipt=receipt,
            streaming_digest_verified=not adopted_existing,
            _streaming_token=(
                _STREAMING_ATTACHMENT_TOKEN if not adopted_existing else None
            ),
            cold_content_digest_verified=adopted_existing,
            _cold_token=_COLD_ATTACHMENT_TOKEN if adopted_existing else None,
        )
        return receipt, attachment, sealed

    def seal(
        self,
    ) -> tuple[
        V1SampledDualChartRowShardReceipt,
        V1SampledDualChartRowShardAttachment,
        Path,
    ]:
        """Compatibility one-look seal: hidden stage followed by publication."""

        self.stage_hidden()
        return self.publish_staged()

    def abort(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass
        self._stream = None
        for path in (self._partial, self._sealed):
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        self._partial = None
        self._sealed = None
        self._staged_receipt = None
        self._closed = True


def _head_path(directory: Path) -> Path:
    return directory / _HEAD_NAME


def _read_head_hash(directory: Path) -> str | None:
    path = _head_path(directory)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V1SampledDualChartRowRefusal(
            V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
            "published prefix HEAD is unreadable",
        ) from error
    if (
        set(payload) != {"schema", "manifest_hash"}
        or payload.get("schema") != "partition_free_v1_sampled_dual_chart_head_v1"
    ):
        _refuse(
            V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
            "published prefix HEAD schema changed",
        )
    return _content_address_token(
        payload.get("manifest_hash"),
        prefix=_MANIFEST_PREFIX,
        name="published manifest hash",
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
    )
    partial = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as stream:
            view = memoryview(payload)
            cursor = 0
            while cursor < len(view):
                written = stream.write(view[cursor:])
                if written is None or written <= 0:
                    raise OSError("short write while publishing sampled-row metadata")
                cursor += int(written)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _atomic_publish_head(directory: Path, manifest_hash: str) -> None:
    token = _content_address_token(
        manifest_hash,
        prefix=_MANIFEST_PREFIX,
        name="published manifest hash",
    )
    payload = canonical_json(
        {
            "schema": "partition_free_v1_sampled_dual_chart_head_v1",
            "manifest_hash": token,
        }
    ).encode("utf-8")
    _atomic_write(_head_path(directory), payload)


def _restore_head(directory: Path, prior_hash: str | None) -> None:
    if prior_hash is None:
        _head_path(directory).unlink(missing_ok=True)
        _fsync_directory(directory)
    else:
        _atomic_publish_head(directory, prior_hash)


def _write_manifest(
    directory: Path, manifest: V1SampledDualChartRowPrefixManifest
) -> Path:
    path = _manifest_path(directory, manifest.manifest_hash)
    payload = canonical_json(manifest.to_dict()).encode("utf-8")
    if path.exists():
        try:
            observed = path.read_bytes()
        except OSError as error:
            raise V1SampledDualChartRowRefusal(
                V1SampledDualChartRowRefusalCode.PUBLICATION_CONFLICT,
                "content-addressed orphan manifest could not be inspected",
            ) from error
        if observed != payload:
            _refuse(
                V1SampledDualChartRowRefusalCode.PUBLICATION_CONFLICT,
                "existing content-addressed manifest is not the exact retry orphan",
            )
        return path
    _atomic_write(path, payload)
    return path


def _remove_unpublished_partial_files(directory: Path) -> None:
    """Remove only suffix-marked files that can never be named by HEAD.

    Callers must hold ``_LOCK_NAME``.  A process crash may leave random
    writer or atomic-metadata partials.  Published manifests and shards never
    use this suffix, so replay cleanup cannot remove authoritative bytes.
    """

    for path in directory.glob(".*.partial"):
        if path.name in {_HEAD_NAME, _LOCK_NAME}:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            raise V1SampledDualChartRowRefusal(
                V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                "unpublished crash partial could not be removed under lock",
            ) from error
    _fsync_directory(directory)


def _validate_extension(
    request: V1SampledDualChartRowLookRequest,
    prior: V1SampledDualChartRowLivePrefixAttachment | None,
) -> None:
    expected_prior = None if prior is None else prior.manifest.manifest_hash
    expected_ordinal = 0 if prior is None else len(prior.manifest.looks)
    if (
        request.expected_prior_manifest_hash != expected_prior
        or request.look_ordinal != expected_ordinal
    ):
        _refuse(
            V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
            "look does not extend the exact published predecessor",
        )
    if prior is None:
        return
    # Content was admitted by the writer or one cold reattachment digest.  The
    # PID-bound immutable witness avoids rereading the accumulated prefix.
    prior.verify_live_witnesses()
    manifest = prior.manifest
    shared = {
        "registered_d_prefix_hash": request.registered_d_prefix_hash,
        "chart_identity_hash": request.chart_identity_hash,
        "current_mask_hash": request.current_mask_hash,
        "proposal_generation": request.proposal_generation,
        "route_axis_hash": request.route_axis_hash,
        "route_id_axis_hash": request.route_id_axis_hash,
        "route_count": request.route_count,
        "corrected_a_stream_identity_hash": request.corrected_a_stream_identity_hash,
        "literal_j_stream_identity_hash": request.literal_j_stream_identity_hash,
    }
    if any(getattr(manifest, name) != value for name, value in shared.items()):
        _refuse(
            V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
            "new look changed the prefix chart/mask/route/stream identity",
        )
    if set(request.context_ids).intersection(manifest.context_ids) or set(
        request.draw_ids
    ).intersection(manifest.draw_ids):
        _refuse(
            V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
            "new look reuses a context or draw already in the prefix",
        )


def _group_packet_row_forecast(
    *,
    plan: V1SampledDualChartRowGroupPlan,
    packet: V1SampledDualChartRowGroupPacketPlan,
) -> V1SampledDualChartRowScratchForecast:
    """Derive one internal packet forecast from the frozen group forecast."""

    forecast = plan.scratch_forecast
    index = packet.packet_ordinal
    exact = forecast.exact_payload_bytes_per_role_shard[index]
    allocation = forecast.allocation_bytes_per_role_shard_upper_bound[index]
    return V1SampledDualChartRowScratchForecast(
        route_count=plan.route_count,
        context_count=len(packet.context_ids),
        registered_route_chunk_size=plan.registered_route_chunk_size,
        exact_bytes_per_role_shard=exact,
        role_shard_count=2,
        aggregate_exact_payload_bytes=2 * exact,
        maximum_owned_host_staging_bytes=(
            len(packet.context_ids) * plan.registered_route_chunk_size * _FLOAT64_BYTES
        ),
        filesystem_fragment_size_bytes=forecast.filesystem_fragment_size_bytes,
        allocation_bytes_per_role_upper_bound=allocation,
        aggregate_allocation_bytes_upper_bound=2 * allocation,
        filesystem_free_bytes_before=forecast.filesystem_free_bytes_before,
        admitted=2 * allocation <= forecast.filesystem_free_bytes_before,
    )


def _group_packet_row_request(
    *,
    plan: V1SampledDualChartRowGroupPlan,
    packet: V1SampledDualChartRowGroupPacketPlan,
    expected_prior_manifest_hash: str | None,
    scratch_directory: Path,
) -> V1SampledDualChartRowLookRequest:
    """Reconstruct the exact path-independent internal packet request."""

    return V1SampledDualChartRowLookRequest(
        look_ordinal=plan.first_prefix_look_ordinal + packet.packet_ordinal,
        expected_prior_manifest_hash=expected_prior_manifest_hash,
        registered_d_prefix_hash=plan.registered_d_prefix_hash,
        source_capture_identity_hash=packet.source_capture_identity_hash,
        chart_identity_hash=plan.chart_identity_hash,
        current_mask_hash=plan.current_mask_hash,
        proposal_generation=plan.proposal_generation,
        route_axis_hash=plan.route_axis_hash,
        route_id_axis_hash=plan.route_id_axis_hash,
        route_count=plan.route_count,
        corrected_a_stream_identity_hash=plan.corrected_a_stream_identity_hash,
        literal_j_stream_identity_hash=plan.literal_j_stream_identity_hash,
        context_ids=packet.context_ids,
        draw_ids=packet.draw_ids,
        stratum_labels=packet.stratum_labels,
        registered_route_chunk_size=plan.registered_route_chunk_size,
        scratch_forecast=_group_packet_row_forecast(plan=plan, packet=packet),
        scratch_directory=str(scratch_directory),
    )


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowGroupCommitReceipt:
    """Path-free closure of a prepared atomic group.

    The same receipt is returned before and after HEAD publication.  It binds
    the complete prospective manifest, the internally derived intra-group
    prefix hashes, and every external proof closure.  Publication state lives
    only in the nonserializable prepared/final runtime wrappers below.
    """

    group_plan_hash: str
    registered_look_index: int
    prior_manifest_hash: str | None
    manifest_hash: str
    packet_plan_hashes: tuple[str, ...]
    packet_look_hashes: tuple[str, ...]
    intra_group_prefix_manifest_hashes: tuple[str, ...]
    proof_closure_hashes: tuple[str, ...]
    provider_proof_hashes: tuple[str, ...]
    reducer_proof_hashes: tuple[tuple[str, ...], ...]
    packet_count: int
    role_shard_count: int
    all_packets_preregistered: bool = True
    all_provider_and_reducer_proofs_closed: bool = True
    single_head_publication_boundary: bool = True
    physical_scratch_path_bound: bool = False
    restart_semantics: str = (
        "unpublished_partial_cleanup_and_exact_content_addressed_orphan_"
        "adoption_under_lock_full_group_replay_required_v1"
    )
    authority_scope: str = SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
    schema: str = SAMPLED_DUAL_CHART_ROW_GROUP_COMMIT_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        packet_count = _positive_int(self.packet_count, name="group packet count")
        role_count = _positive_int(self.role_shard_count, name="group role shard count")
        ledgers = (
            _ordered_strings(
                self.packet_plan_hashes,
                name="group packet-plan hashes",
                unique=True,
            ),
            _ordered_strings(
                self.packet_look_hashes,
                name="group packet look hashes",
                unique=True,
            ),
            _ordered_strings(
                self.intra_group_prefix_manifest_hashes,
                name="group intra-prefix hashes",
                unique=True,
            ),
            _ordered_strings(
                self.proof_closure_hashes,
                name="group proof-closure hashes",
                unique=True,
            ),
            _ordered_strings(
                self.provider_proof_hashes,
                name="group provider proof hashes",
                unique=True,
            ),
        )
        reducers = tuple(
            _ordered_strings(
                values,
                name="group reducer proof hashes",
                unique=True,
            )
            for values in self.reducer_proof_hashes
        )
        if (
            any(len(values) != packet_count for values in ledgers)
            or len(reducers) != packet_count
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group commit receipt packet ledgers are incomplete",
            )
        if role_count != 2 * packet_count:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group commit receipt does not bind two role shards per packet",
            )
        for name in ("group_plan_hash", "manifest_hash"):
            _identity(getattr(self, name), name=name.replace("_", " "))
        if self.prior_manifest_hash is not None:
            _identity(self.prior_manifest_hash, name="group prior manifest hash")
        if (
            self.all_packets_preregistered is not True
            or self.all_provider_and_reducer_proofs_closed is not True
            or self.single_head_publication_boundary is not True
            or self.physical_scratch_path_bound is not False
            or self.restart_semantics
            != (
                "unpublished_partial_cleanup_and_exact_content_addressed_orphan_"
                "adoption_under_lock_full_group_replay_required_v1"
            )
            or self.authority_scope != SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
            or self.schema != SAMPLED_DUAL_CHART_ROW_GROUP_COMMIT_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "group commit receipt changed atomic custody or authority",
            )
        object.__setattr__(self, "packet_count", packet_count)
        object.__setattr__(self, "role_shard_count", role_count)
        object.__setattr__(self, "packet_plan_hashes", ledgers[0])
        object.__setattr__(self, "packet_look_hashes", ledgers[1])
        object.__setattr__(self, "intra_group_prefix_manifest_hashes", ledgers[2])
        object.__setattr__(self, "proof_closure_hashes", ledgers[3])
        object.__setattr__(self, "provider_proof_hashes", ledgers[4])
        object.__setattr__(self, "reducer_proof_hashes", reducers)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "group_plan_hash",
                "registered_look_index",
                "prior_manifest_hash",
                "manifest_hash",
                "packet_plan_hashes",
                "packet_look_hashes",
                "intra_group_prefix_manifest_hashes",
                "proof_closure_hashes",
                "provider_proof_hashes",
                "reducer_proof_hashes",
                "packet_count",
                "role_shard_count",
                "all_packets_preregistered",
                "all_provider_and_reducer_proofs_closed",
                "single_head_publication_boundary",
                "physical_scratch_path_bound",
                "restart_semantics",
                "authority_scope",
            )
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_GROUP_COMMIT_PREFIX)
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group commit receipt was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "group commit receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowPreparedGroupCommit:
    """Runtime candidate whose files exist but whose HEAD is still predecessor."""

    receipt: V1SampledDualChartRowGroupCommitReceipt
    attachment: V1SampledDualChartRowLivePrefixAttachment = field(
        repr=False,
        compare=False,
    )
    head_published: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not V1SampledDualChartRowGroupCommitReceipt
            or type(self.attachment) is not V1SampledDualChartRowLivePrefixAttachment
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "prepared group commit requires exact receipt and attachment",
            )
        self.receipt.assert_integrity()
        if (
            self.attachment.manifest.manifest_hash != self.receipt.manifest_hash
            or self.head_published is not False
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "prepared group commit misreported its HEAD boundary",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("prepared group commits carry nonserializable live paths")


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowGroupCommit:
    """Runtime result after the one atomic group HEAD publication."""

    receipt: V1SampledDualChartRowGroupCommitReceipt
    attachment: V1SampledDualChartRowLivePrefixAttachment = field(
        repr=False,
        compare=False,
    )
    head_published: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not V1SampledDualChartRowGroupCommitReceipt
            or type(self.attachment) is not V1SampledDualChartRowLivePrefixAttachment
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "published group commit requires exact receipt and attachment",
            )
        self.receipt.assert_integrity()
        if (
            self.attachment.manifest.manifest_hash != self.receipt.manifest_hash
            or self.head_published is not True
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "published group commit misreported its HEAD boundary",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("published group commits carry nonserializable live paths")


class V1SampledDualChartRowGroupStagingConsumer:
    """Bounded route-window consumer for exactly one preregistered packet."""

    __slots__ = (
        "_transaction",
        "_packet_plan",
        "_request",
        "_a_writer",
        "_j_writer",
        "_next_route",
        "_sealed",
    )

    def __init__(
        self,
        *,
        transaction: V1SampledDualChartRowGroupTransaction,
        packet_plan: V1SampledDualChartRowGroupPacketPlan,
        request: V1SampledDualChartRowLookRequest,
    ) -> None:
        self._transaction = transaction
        self._packet_plan = packet_plan
        self._request = request
        try:
            self._a_writer = _OpenRoleShardWriter(
                request=request,
                role=V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY,
            )
            self._j_writer = _OpenRoleShardWriter(
                request=request,
                role=V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN,
            )
        except Exception:
            transaction.abort()
            raise
        transaction._register_writers(self._a_writer, self._j_writer)
        self._next_route = 0
        self._sealed = False

    @property
    def packet_ordinal(self) -> int:
        return self._packet_plan.packet_ordinal

    @property
    def packet_plan_hash(self) -> str:
        return self._packet_plan.packet_plan_hash

    def consume(self, chunk: V1SampledDualChartRowChunk) -> None:
        """Consume one same-capture A/J route window with bounded host staging."""

        try:
            if self._sealed:
                raise RuntimeError("group packet staging consumer is already sealed")
            if type(chunk) is not V1SampledDualChartRowChunk:
                _refuse(
                    V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                    "group staging requires exact dual-role chunk records",
                )
            if (
                chunk.start != self._next_route
                or chunk.stop <= chunk.start
                or chunk.stop > self._request.route_count
                or chunk.stop - chunk.start > self._request.registered_route_chunk_size
            ):
                _refuse(
                    V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                    "group packet changed its bounded contiguous route-window order",
                )
            a_source = self._a_writer.consume(
                start=chunk.start,
                stop=chunk.stop,
                rows=chunk.corrected_a_rows,
            )
            j_source = self._j_writer.consume(
                start=chunk.start,
                stop=chunk.stop,
                rows=chunk.literal_j_rows,
            )
            if a_source != j_source:
                _refuse(
                    V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                    "same-capture packet roles changed source device kind",
                )
            self._next_route = chunk.stop
        except Exception:
            self._transaction.abort()
            raise

    def seal(self) -> V1SampledDualChartRowLookReceipt:
        """Seal both packet-role shards hidden; never publish HEAD."""

        try:
            if self._sealed:
                raise RuntimeError("group packet staging consumer cannot seal twice")
            if self._next_route != self._request.route_count:
                _refuse(
                    V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                    "group packet ended before complete route coverage",
                )
            a_receipt = self._a_writer.stage_hidden()
            j_receipt = self._j_writer.stage_hidden()
            look = V1SampledDualChartRowLookReceipt(
                look_ordinal=self._request.look_ordinal,
                prior_manifest_hash=self._request.expected_prior_manifest_hash,
                request_hash=self._request.request_hash,
                context_ids=self._request.context_ids,
                draw_ids=self._request.draw_ids,
                stratum_labels=self._request.stratum_labels,
                corrected_a_shard=a_receipt,
                literal_j_shard=j_receipt,
            )
            self._transaction._accept_staged_packet(
                packet_plan=self._packet_plan,
                look=look,
            )
            self._sealed = True
            return look
        except Exception:
            self._transaction.abort()
            raise

    def __reduce__(self) -> NoReturn:
        raise TypeError("group staging consumers carry nonserializable live paths")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("group staging consumers carry nonserializable live paths")


class V1SampledDualChartRowGroupTransaction:
    """One-lock, two-phase atomic custody transaction for a sampled-D look.

    Integration protocol:

    1. preregister the complete :class:`V1SampledDualChartRowGroupPlan`;
    2. open and seal each packet consumer in packet-ordinal order;
    3. mark each packet's provider and reducer proofs closed;
    4. call :meth:`prepare_commit` and bind its path-free receipt into the
       owning V5 registered-look commit;
    5. after that outer commit validates, call :meth:`publish_prepared` with
       the exact receipt hash.

    The publication lock is held for the entire protocol.  Until step five,
    any exception or explicit abort removes every new file and restores the
    exact predecessor HEAD.  A process crash requires full group replay:
    suffix-marked hidden stages are safely removed under the lock, while
    exact content-addressed shard/manifest orphans are byte-verified and
    adopted.  This runtime object and every consumer are nonserializable.
    """

    __slots__ = (
        "plan",
        "_directory",
        "_prior",
        "_lock",
        "_state",
        "_next_packet",
        "_active_consumer",
        "_writers",
        "_staged_looks",
        "_virtual_manifests",
        "_proof_closures",
        "_manifest_path",
        "_prepared",
    )

    def __init__(
        self,
        *,
        plan: V1SampledDualChartRowGroupPlan,
        scratch_directory: str | os.PathLike[str],
        prior: V1SampledDualChartRowLivePrefixAttachment | None,
    ) -> None:
        if type(plan) is not V1SampledDualChartRowGroupPlan:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "group transaction requires its exact path-free plan",
            )
        if (
            prior is not None
            and type(prior) is not V1SampledDualChartRowLivePrefixAttachment
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "group transaction predecessor has the wrong type",
            )
        plan.assert_integrity()
        plan.scratch_forecast.assert_admissible()
        directory = _scratch_directory(scratch_directory)
        self.plan = plan
        self._directory = directory
        self._prior = prior
        self._lock: Any | None = None
        self._state = "constructing"
        self._next_packet = 0
        self._active_consumer: V1SampledDualChartRowGroupStagingConsumer | None = None
        self._writers: list[_OpenRoleShardWriter] = []
        self._staged_looks: list[V1SampledDualChartRowLookReceipt] = []
        self._virtual_manifests: list[V1SampledDualChartRowPrefixManifest] = []
        self._proof_closures: dict[
            int,
            V1SampledDualChartRowGroupPacketProofClosure,
        ] = {}
        self._manifest_path: Path | None = None
        self._prepared: V1SampledDualChartRowPreparedGroupCommit | None = None
        try:
            lock = (directory / _LOCK_NAME).open("a+b")
            self._lock = lock
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            _remove_unpublished_partial_files(directory)
            self._validate_open_state()
            free, _ = _scratch_capacity(directory)
            if free < plan.scratch_forecast.aggregate_allocation_bytes_upper_bound:
                _refuse(
                    V1SampledDualChartRowRefusalCode.SCRATCH_INADMISSIBLE,
                    "live scratch fell below the complete packet-group forecast",
                )
            self._state = "active"
        except Exception:
            self._release_lock()
            raise

    def _expected_predecessor_hash(self) -> str | None:
        return None if self._prior is None else self._prior.manifest.manifest_hash

    def _validate_open_state(self) -> None:
        expected = self._expected_predecessor_hash()
        if _read_head_hash(self._directory) != expected:
            _refuse(
                V1SampledDualChartRowRefusalCode.PUBLICATION_CONFLICT,
                "published prefix HEAD differs from the group predecessor",
            )
        if (
            self.plan.expected_prior_manifest_hash != expected
            or self.plan.first_prefix_look_ordinal
            != (0 if self._prior is None else len(self._prior.manifest.looks))
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "group plan does not extend the exact published predecessor",
            )
        if self._prior is None:
            return
        self._prior.verify_live_witnesses()
        manifest = self._prior.manifest
        shared = {
            "registered_d_prefix_hash": self.plan.registered_d_prefix_hash,
            "chart_identity_hash": self.plan.chart_identity_hash,
            "current_mask_hash": self.plan.current_mask_hash,
            "proposal_generation": self.plan.proposal_generation,
            "route_axis_hash": self.plan.route_axis_hash,
            "route_id_axis_hash": self.plan.route_id_axis_hash,
            "route_count": self.plan.route_count,
            "corrected_a_stream_identity_hash": (
                self.plan.corrected_a_stream_identity_hash
            ),
            "literal_j_stream_identity_hash": self.plan.literal_j_stream_identity_hash,
        }
        if any(getattr(manifest, name) != value for name, value in shared.items()):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "group changed predecessor chart/mask/route/stream identity",
            )
        contexts = {
            value for packet in self.plan.packet_plans for value in packet.context_ids
        }
        draws = {
            value for packet in self.plan.packet_plans for value in packet.draw_ids
        }
        if contexts.intersection(manifest.context_ids) or draws.intersection(
            manifest.draw_ids
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "group reuses predecessor context or draw identities",
            )

    def _assert_state(self, *states: str) -> None:
        if self._state not in states:
            raise RuntimeError(
                f"sampled dual-row group transaction is {self._state}, "
                f"expected one of {states}"
            )

    def _release_lock(self) -> None:
        lock = self._lock
        self._lock = None
        if lock is not None:
            try:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                except OSError:
                    # Closing the descriptor is the authoritative kernel lock
                    # release fallback.  Nothing scientific remains after a
                    # successful HEAD replace.
                    pass
            finally:
                try:
                    lock.close()
                except OSError:
                    pass

    def _register_writers(
        self,
        a_writer: _OpenRoleShardWriter,
        j_writer: _OpenRoleShardWriter,
    ) -> None:
        self._writers.extend((a_writer, j_writer))

    def _current_virtual_predecessor_hash(self) -> str | None:
        if self._virtual_manifests:
            return self._virtual_manifests[-1].manifest_hash
        return self._expected_predecessor_hash()

    def _request_for_packet(
        self,
        packet: V1SampledDualChartRowGroupPacketPlan,
    ) -> V1SampledDualChartRowLookRequest:
        return _group_packet_row_request(
            plan=self.plan,
            packet=packet,
            expected_prior_manifest_hash=self._current_virtual_predecessor_hash(),
            scratch_directory=self._directory,
        )

    def open_packet_staging_consumer(
        self,
        *,
        packet_ordinal: int,
    ) -> V1SampledDualChartRowGroupStagingConsumer:
        """Open the next preregistered packet; packet order is not negotiable."""

        try:
            self._assert_state("active")
            ordinal = _nonnegative_int(packet_ordinal, name="open packet ordinal")
            if self._active_consumer is not None:
                raise RuntimeError(
                    "one group packet staging consumer is already active"
                )
            if ordinal != self._next_packet or ordinal >= len(self.plan.packet_plans):
                _refuse(
                    V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                    "group packet staging departed from preregistered order",
                )
            packet = self.plan.packet_plans[ordinal]
            request = self._request_for_packet(packet)
            consumer = V1SampledDualChartRowGroupStagingConsumer(
                transaction=self,
                packet_plan=packet,
                request=request,
            )
            self._active_consumer = consumer
            return consumer
        except Exception:
            self.abort()
            raise

    def _accept_staged_packet(
        self,
        *,
        packet_plan: V1SampledDualChartRowGroupPacketPlan,
        look: V1SampledDualChartRowLookReceipt,
    ) -> None:
        if (
            self._active_consumer is None
            or packet_plan.packet_ordinal != self._next_packet
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "group accepted a packet outside its active preregistration",
            )
        prior_looks = () if self._prior is None else self._prior.manifest.looks
        looks = prior_looks + tuple(self._staged_looks) + (look,)
        manifest = V1SampledDualChartRowPrefixManifest(
            prior_manifest_hash=look.prior_manifest_hash,
            registered_d_prefix_hash=self.plan.registered_d_prefix_hash,
            chart_identity_hash=self.plan.chart_identity_hash,
            current_mask_hash=self.plan.current_mask_hash,
            proposal_generation=self.plan.proposal_generation,
            route_axis_hash=self.plan.route_axis_hash,
            route_id_axis_hash=self.plan.route_id_axis_hash,
            route_count=self.plan.route_count,
            corrected_a_stream_identity_hash=(
                self.plan.corrected_a_stream_identity_hash
            ),
            literal_j_stream_identity_hash=self.plan.literal_j_stream_identity_hash,
            looks=looks,
        )
        self._staged_looks.append(look)
        self._virtual_manifests.append(manifest)
        self._next_packet += 1
        self._active_consumer = None

    def mark_packet_proofs_closed(
        self,
        *,
        packet_ordinal: int,
        provider_proof_hash: str,
        reducer_proof_hashes: Sequence[str],
    ) -> V1SampledDualChartRowGroupPacketProofClosure:
        """Bind exact external completion proofs after the packet is staged."""

        try:
            self._assert_state("active")
            ordinal = _nonnegative_int(packet_ordinal, name="proof packet ordinal")
            if ordinal >= self._next_packet:
                _refuse(
                    V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                    "proof closure cannot precede hidden packet staging",
                )
            if ordinal in self._proof_closures:
                _refuse(
                    V1SampledDualChartRowRefusalCode.PUBLICATION_CONFLICT,
                    "packet proof closure is append-only",
                )
            packet = self.plan.packet_plans[ordinal]
            proofs = tuple(reducer_proof_hashes)
            if len(proofs) != len(packet.reducer_plan_identity_hashes):
                _refuse(
                    V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                    "packet reducer proof ledger differs from preregistration",
                )
            closure = V1SampledDualChartRowGroupPacketProofClosure(
                group_plan_hash=self.plan.group_plan_hash,
                packet_plan_hash=packet.packet_plan_hash,
                packet_ordinal=ordinal,
                provider_plan_identity_hash=packet.provider_plan_identity_hash,
                reducer_plan_identity_hashes=packet.reducer_plan_identity_hashes,
                provider_proof_hash=provider_proof_hash,
                reducer_proof_hashes=proofs,
                _issuance_token=_GROUP_PROOF_ISSUANCE_TOKEN,
            )
            self._proof_closures[ordinal] = closure
            return closure
        except Exception:
            self.abort()
            raise

    def prepare_commit(self) -> V1SampledDualChartRowPreparedGroupCommit:
        """Prepare all content-addressed files without changing HEAD."""

        try:
            self._assert_state("active", "prepared")
            if self._state == "prepared":
                assert self._prepared is not None
                return self._prepared
            packet_count = len(self.plan.packet_plans)
            if (
                self._active_consumer is not None
                or self._next_packet != packet_count
                or tuple(sorted(self._proof_closures)) != tuple(range(packet_count))
            ):
                _refuse(
                    V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                    "group cannot prepare before every packet and proof closes",
                )
            new_a: list[V1SampledDualChartRowShardAttachment] = []
            new_j: list[V1SampledDualChartRowShardAttachment] = []
            for index in range(0, len(self._writers), 2):
                _a_receipt, a_attachment, _a_path = self._writers[
                    index
                ].publish_staged()
                _j_receipt, j_attachment, _j_path = self._writers[
                    index + 1
                ].publish_staged()
                new_a.append(a_attachment)
                new_j.append(j_attachment)
            manifest = self._virtual_manifests[-1]
            self._manifest_path = _write_manifest(self._directory, manifest)
            prior_a = () if self._prior is None else self._prior.corrected_a_attachments
            prior_j = () if self._prior is None else self._prior.literal_j_attachments
            attachment = V1SampledDualChartRowLivePrefixAttachment(
                manifest=manifest,
                corrected_a_attachments=prior_a + tuple(new_a),
                literal_j_attachments=prior_j + tuple(new_j),
            )
            closures = tuple(
                self._proof_closures[index] for index in range(packet_count)
            )
            receipt = V1SampledDualChartRowGroupCommitReceipt(
                group_plan_hash=self.plan.group_plan_hash,
                registered_look_index=self.plan.registered_look_index,
                prior_manifest_hash=self.plan.expected_prior_manifest_hash,
                manifest_hash=manifest.manifest_hash,
                packet_plan_hashes=tuple(
                    packet.packet_plan_hash for packet in self.plan.packet_plans
                ),
                packet_look_hashes=tuple(look.look_hash for look in self._staged_looks),
                intra_group_prefix_manifest_hashes=tuple(
                    item.manifest_hash for item in self._virtual_manifests
                ),
                proof_closure_hashes=tuple(
                    closure.closure_hash for closure in closures
                ),
                provider_proof_hashes=tuple(
                    closure.provider_proof_hash for closure in closures
                ),
                reducer_proof_hashes=tuple(
                    closure.reducer_proof_hashes for closure in closures
                ),
                packet_count=packet_count,
                role_shard_count=2 * packet_count,
            )
            prepared = V1SampledDualChartRowPreparedGroupCommit(
                receipt=receipt,
                attachment=attachment,
            )
            self._prepared = prepared
            self._state = "prepared"
            return prepared
        except Exception:
            self.abort()
            raise

    def publish_prepared(
        self,
        *,
        expected_receipt_hash: str,
    ) -> V1SampledDualChartRowGroupCommit:
        """Perform the only HEAD replace after outer V5 validation closes."""

        try:
            self._assert_state("prepared")
            prepared = self._prepared
            assert prepared is not None
            prepared.receipt.assert_integrity(expected_receipt_hash)
            if _read_head_hash(self._directory) != self._expected_predecessor_hash():
                _refuse(
                    V1SampledDualChartRowRefusalCode.PUBLICATION_CONFLICT,
                    "group predecessor HEAD changed before publication",
                )
            result = V1SampledDualChartRowGroupCommit(
                receipt=prepared.receipt,
                attachment=prepared.attachment,
            )
            _atomic_publish_head(self._directory, prepared.receipt.manifest_hash)
        except Exception:
            self.abort()
            raise
        self._state = "published"
        self._release_lock()
        return result

    def commit(self) -> V1SampledDualChartRowGroupCommit:
        """Convenience wrapper; V5 integration should use the two-phase API."""

        prepared = self.prepare_commit()
        return self.publish_prepared(
            expected_receipt_hash=prepared.receipt.receipt_hash
        )

    def abort(self) -> None:
        """Rollback every unpublished group effect and release the one lock."""

        if self._state in {"aborted", "published"}:
            return
        predecessor = self._expected_predecessor_hash()
        first_failure: BaseException | None = None
        try:
            if self._lock is not None:
                # Restore unconditionally: an injected or system fault may
                # occur after HEAD replacement but before the publisher
                # returns, including with unreadable torn metadata.
                try:
                    _restore_head(self._directory, predecessor)
                except BaseException as error:  # pragma: no cover - I/O fault.
                    first_failure = error
            if self._manifest_path is not None:
                try:
                    self._manifest_path.unlink(missing_ok=True)
                except BaseException as error:  # pragma: no cover - I/O fault.
                    if first_failure is None:
                        first_failure = error
            for writer in self._writers:
                writer.abort()
            if self._lock is not None:
                try:
                    _remove_unpublished_partial_files(self._directory)
                    _fsync_directory(self._directory)
                except BaseException as error:  # pragma: no cover - I/O fault.
                    if first_failure is None:
                        first_failure = error
        finally:
            self._manifest_path = None
            self._prepared = None
            self._active_consumer = None
            self._state = "aborted"
            self._release_lock()
        if first_failure is not None:
            raise first_failure

    def __enter__(self) -> V1SampledDualChartRowGroupTransaction:
        self._assert_state("active")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        if self._state != "published":
            self.abort()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("group transactions carry nonserializable live paths")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("group transactions carry nonserializable live paths")


def begin_v1_sampled_dual_chart_row_group_transaction(
    *,
    plan: V1SampledDualChartRowGroupPlan,
    scratch_directory: str | os.PathLike[str],
    prior: V1SampledDualChartRowLivePrefixAttachment | None = None,
) -> V1SampledDualChartRowGroupTransaction:
    """Acquire the one publication lock for a fully preregistered group."""

    return V1SampledDualChartRowGroupTransaction(
        plan=plan,
        scratch_directory=scratch_directory,
        prior=prior,
    )


def _validate_prepared_group_recovery_bindings(
    *,
    plan: V1SampledDualChartRowGroupPlan,
    receipt: V1SampledDualChartRowGroupCommitReceipt,
    manifest: V1SampledDualChartRowPrefixManifest,
    scratch_directory: Path,
) -> None:
    """Replay every path-free plan/receipt/manifest edge before reattachment."""

    plan.assert_integrity()
    receipt.assert_integrity()
    manifest.assert_integrity()
    packet_count = len(plan.packet_plans)
    first = plan.first_prefix_look_ordinal
    if (
        receipt.group_plan_hash != plan.group_plan_hash
        or receipt.registered_look_index != plan.registered_look_index
        or receipt.prior_manifest_hash != plan.expected_prior_manifest_hash
        or receipt.manifest_hash != manifest.manifest_hash
        or receipt.packet_count != packet_count
        or receipt.role_shard_count != 2 * packet_count
        or receipt.packet_plan_hashes
        != tuple(packet.packet_plan_hash for packet in plan.packet_plans)
        or len(manifest.looks) != first + packet_count
    ):
        _refuse(
            V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
            "prepared recovery plan, receipt, and prospective manifest diverged",
        )
    shared = {
        "registered_d_prefix_hash": plan.registered_d_prefix_hash,
        "chart_identity_hash": plan.chart_identity_hash,
        "current_mask_hash": plan.current_mask_hash,
        "proposal_generation": plan.proposal_generation,
        "route_axis_hash": plan.route_axis_hash,
        "route_id_axis_hash": plan.route_id_axis_hash,
        "route_count": plan.route_count,
        "corrected_a_stream_identity_hash": plan.corrected_a_stream_identity_hash,
        "literal_j_stream_identity_hash": plan.literal_j_stream_identity_hash,
    }
    if any(getattr(manifest, name) != value for name, value in shared.items()):
        _refuse(
            V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
            "prepared prospective manifest changed group chart custody",
        )
    old_looks = manifest.looks[:first]
    if first == 0:
        if plan.expected_prior_manifest_hash is not None:
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "prepared first group unexpectedly names a predecessor",
            )
    else:
        prior_prefix = V1SampledDualChartRowPrefixManifest(
            prior_manifest_hash=old_looks[-1].prior_manifest_hash,
            registered_d_prefix_hash=plan.registered_d_prefix_hash,
            chart_identity_hash=plan.chart_identity_hash,
            current_mask_hash=plan.current_mask_hash,
            proposal_generation=plan.proposal_generation,
            route_axis_hash=plan.route_axis_hash,
            route_id_axis_hash=plan.route_id_axis_hash,
            route_count=plan.route_count,
            corrected_a_stream_identity_hash=plan.corrected_a_stream_identity_hash,
            literal_j_stream_identity_hash=plan.literal_j_stream_identity_hash,
            looks=old_looks,
        )
        if prior_prefix.manifest_hash != plan.expected_prior_manifest_hash:
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "prospective manifest does not contain the declared predecessor",
            )

    current_prior = plan.expected_prior_manifest_hash
    observed_virtual_hashes: list[str] = []
    new_looks = manifest.looks[first:]
    for packet, look in zip(plan.packet_plans, new_looks, strict=True):
        request = _group_packet_row_request(
            plan=plan,
            packet=packet,
            expected_prior_manifest_hash=current_prior,
            scratch_directory=scratch_directory,
        )
        a = look.corrected_a_shard
        j = look.literal_j_shard
        if (
            look.look_ordinal != request.look_ordinal
            or look.prior_manifest_hash != current_prior
            or look.request_hash != request.request_hash
            or look.context_ids != packet.context_ids
            or look.draw_ids != packet.draw_ids
            or look.stratum_labels != packet.stratum_labels
            or a.request_hash != request.request_hash
            or j.request_hash != request.request_hash
            or a.scratch_forecast_hash != request.scratch_forecast.forecast_hash
            or j.scratch_forecast_hash != request.scratch_forecast.forecast_hash
            or a.source_capture_identity_hash != packet.source_capture_identity_hash
            or j.source_capture_identity_hash != packet.source_capture_identity_hash
            or a.stream_identity_hash != plan.corrected_a_stream_identity_hash
            or j.stream_identity_hash != plan.literal_j_stream_identity_hash
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                "prepared packet receipt differs from its preregistered request",
            )
        virtual = V1SampledDualChartRowPrefixManifest(
            prior_manifest_hash=current_prior,
            registered_d_prefix_hash=plan.registered_d_prefix_hash,
            chart_identity_hash=plan.chart_identity_hash,
            current_mask_hash=plan.current_mask_hash,
            proposal_generation=plan.proposal_generation,
            route_axis_hash=plan.route_axis_hash,
            route_id_axis_hash=plan.route_id_axis_hash,
            route_count=plan.route_count,
            corrected_a_stream_identity_hash=plan.corrected_a_stream_identity_hash,
            literal_j_stream_identity_hash=plan.literal_j_stream_identity_hash,
            looks=old_looks + new_looks[: packet.packet_ordinal + 1],
        )
        current_prior = virtual.manifest_hash
        observed_virtual_hashes.append(current_prior)

    if (
        tuple(look.look_hash for look in new_looks) != receipt.packet_look_hashes
        or tuple(observed_virtual_hashes) != receipt.intra_group_prefix_manifest_hashes
        or not observed_virtual_hashes
        or observed_virtual_hashes[-1] != manifest.manifest_hash
    ):
        _refuse(
            V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
            "prepared recovery intra-group prefix ledger does not replay",
        )
    observed_closures: list[str] = []
    for index, packet in enumerate(plan.packet_plans):
        closure = V1SampledDualChartRowGroupPacketProofClosure(
            group_plan_hash=plan.group_plan_hash,
            packet_plan_hash=packet.packet_plan_hash,
            packet_ordinal=index,
            provider_plan_identity_hash=packet.provider_plan_identity_hash,
            reducer_plan_identity_hashes=packet.reducer_plan_identity_hashes,
            provider_proof_hash=receipt.provider_proof_hashes[index],
            reducer_proof_hashes=receipt.reducer_proof_hashes[index],
            _issuance_token=_GROUP_PROOF_ISSUANCE_TOKEN,
        )
        observed_closures.append(closure.closure_hash)
    if tuple(observed_closures) != receipt.proof_closure_hashes:
        _refuse(
            V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
            "prepared recovery external proof ledger does not replay",
        )


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowPreparedGroupRecoveryResult:
    """Runtime proof of one provider-free publish or successor acknowledgement."""

    commit: V1SampledDualChartRowGroupCommit = field(repr=False, compare=False)
    head_advance_performed: bool
    matching_successor_acknowledged: bool
    provider_execution_count: int = 0
    vjp_execution_count: int = 0
    raw_row_replay_count: int = 0
    authority_scope: str = SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
    schema: str = SAMPLED_DUAL_CHART_ROW_PREPARED_GROUP_RECOVERY_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.commit) is not V1SampledDualChartRowGroupCommit:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "prepared recovery result requires its exact live commit",
            )
        if (
            type(self.head_advance_performed) is not bool
            or type(self.matching_successor_acknowledged) is not bool
            or self.head_advance_performed is self.matching_successor_acknowledged
            or self.provider_execution_count != 0
            or self.vjp_execution_count != 0
            or self.raw_row_replay_count != 0
            or self.authority_scope != SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE
            or self.schema
            != SAMPLED_DUAL_CHART_ROW_PREPARED_GROUP_RECOVERY_RESULT_SCHEMA
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.AUTHORITY_VIOLATION,
                "prepared recovery result changed provider-free custody authority",
            )

    @property
    def receipt(self) -> V1SampledDualChartRowGroupCommitReceipt:
        return self.commit.receipt

    @property
    def attachment(self) -> V1SampledDualChartRowLivePrefixAttachment:
        return self.commit.attachment

    def __reduce__(self) -> NoReturn:
        raise TypeError("prepared recovery results carry nonserializable live paths")


class V1SampledDualChartRowPreparedGroupRecovery:
    """Locked provider-free recovery of one durable prepared group."""

    __slots__ = (
        "plan",
        "receipt",
        "prospective_manifest",
        "attachment",
        "head_already_published",
        "_directory",
        "_manifest_path",
        "_manifest_invariant",
        "_canonical_manifest_bytes",
        "_lock",
        "_closed",
    )

    def __init__(
        self,
        *,
        plan: V1SampledDualChartRowGroupPlan,
        receipt: V1SampledDualChartRowGroupCommitReceipt,
        prospective_manifest: V1SampledDualChartRowPrefixManifest,
        attachment: V1SampledDualChartRowLivePrefixAttachment,
        head_already_published: bool,
        directory: Path,
        manifest_path: Path,
        manifest_invariant: tuple[int, int, int, int],
        canonical_manifest_bytes: bytes,
        lock: Any,
    ) -> None:
        self.plan = plan
        self.receipt = receipt
        self.prospective_manifest = prospective_manifest
        self.attachment = attachment
        self.head_already_published = head_already_published
        self._directory = directory
        self._manifest_path = manifest_path
        self._manifest_invariant = manifest_invariant
        self._canonical_manifest_bytes = canonical_manifest_bytes
        self._lock: Any | None = lock
        self._closed = False

    def _close_lock(self) -> None:
        lock = self._lock
        self._lock = None
        if lock is not None:
            try:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            finally:
                try:
                    lock.close()
                except OSError:
                    pass
        self._closed = True

    def _verify_live_prepublication_custody(self) -> None:
        self.plan.assert_integrity()
        self.receipt.assert_integrity()
        self.prospective_manifest.assert_integrity()
        try:
            invariant = _file_invariant(self._manifest_path)
            payload = self._manifest_path.read_bytes()
        except OSError as error:
            raise V1SampledDualChartRowRefusal(
                V1SampledDualChartRowRefusalCode.MISSING_SHARD,
                "prepared prospective manifest disappeared during recovery",
            ) from error
        if (
            invariant != self._manifest_invariant
            or payload != self._canonical_manifest_bytes
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.CUSTODY_MUTATED,
                "prepared prospective manifest inode or canonical bytes changed",
            )
        self.attachment.verify_live_witnesses()

    def publish_or_acknowledge(
        self,
        *,
        expected_receipt_hash: str,
    ) -> V1SampledDualChartRowPreparedGroupRecoveryResult:
        """Publish predecessor HEAD once, or acknowledge the exact successor."""

        if self._closed or self._lock is None:
            raise RuntimeError("prepared group recovery is already closed")
        try:
            self.receipt.assert_integrity(expected_receipt_hash)
            self._verify_live_prepublication_custody()
            observed_head = _read_head_hash(self._directory)
            predecessor = self.plan.expected_prior_manifest_hash
            successor = self.receipt.manifest_hash
            if observed_head not in {predecessor, successor}:
                _refuse(
                    V1SampledDualChartRowRefusalCode.PUBLICATION_CONFLICT,
                    "prepared recovery HEAD is neither predecessor nor successor",
                )
            already = observed_head == successor
            commit = V1SampledDualChartRowGroupCommit(
                receipt=self.receipt,
                attachment=self.attachment,
            )
            result = V1SampledDualChartRowPreparedGroupRecoveryResult(
                commit=commit,
                head_advance_performed=not already,
                matching_successor_acknowledged=already,
            )
            if not already:
                _atomic_publish_head(self._directory, successor)
        except Exception:
            self._close_lock()
            raise
        self._close_lock()
        return result

    def close(self) -> None:
        """Release recovery custody without deleting durable prepared bytes."""

        self._close_lock()

    def __enter__(self) -> V1SampledDualChartRowPreparedGroupRecovery:
        if self._closed:
            raise RuntimeError("prepared group recovery is already closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("prepared group recovery carries nonserializable live paths")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("prepared group recovery carries nonserializable live paths")


def reattach_v1_sampled_dual_chart_row_prepared_group(
    *,
    plan: V1SampledDualChartRowGroupPlan,
    receipt: V1SampledDualChartRowGroupCommitReceipt,
    prospective_manifest: V1SampledDualChartRowPrefixManifest,
    scratch_directory: str | os.PathLike[str],
) -> V1SampledDualChartRowPreparedGroupRecovery:
    """Recover a durable prepared group without provider, VJP, or row replay."""

    if type(plan) is not V1SampledDualChartRowGroupPlan:
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            "prepared recovery requires its exact group plan",
        )
    if type(receipt) is not V1SampledDualChartRowGroupCommitReceipt:
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            "prepared recovery requires its exact group receipt",
        )
    if type(prospective_manifest) is not V1SampledDualChartRowPrefixManifest:
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            "prepared recovery requires its exact prospective manifest",
        )
    directory = _scratch_directory(scratch_directory)
    lock = (directory / _LOCK_NAME).open("a+b")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        _remove_unpublished_partial_files(directory)
        _validate_prepared_group_recovery_bindings(
            plan=plan,
            receipt=receipt,
            manifest=prospective_manifest,
            scratch_directory=directory,
        )
        head = _read_head_hash(directory)
        predecessor = plan.expected_prior_manifest_hash
        successor = receipt.manifest_hash
        if head not in {predecessor, successor}:
            _refuse(
                V1SampledDualChartRowRefusalCode.PUBLICATION_CONFLICT,
                "prepared recovery found a foreign published HEAD",
            )
        manifest_path = _manifest_path(directory, successor)
        canonical_manifest_bytes = canonical_json(
            prospective_manifest.to_dict()
        ).encode("utf-8")
        try:
            observed_manifest_bytes = manifest_path.read_bytes()
            manifest_invariant = _file_invariant(manifest_path)
        except OSError as error:
            raise V1SampledDualChartRowRefusal(
                V1SampledDualChartRowRefusalCode.MISSING_SHARD,
                "prepared prospective manifest is missing",
            ) from error
        if observed_manifest_bytes != canonical_manifest_bytes:
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "prepared prospective manifest bytes are not exact canonical JSON",
            )
        a_attachments: list[V1SampledDualChartRowShardAttachment] = []
        j_attachments: list[V1SampledDualChartRowShardAttachment] = []
        for look in prospective_manifest.looks:
            a_attachments.append(
                V1SampledDualChartRowShardAttachment(
                    path=_shard_path(
                        directory,
                        look.corrected_a_shard.receipt_hash,
                    ),
                    receipt=look.corrected_a_shard,
                )
            )
            j_attachments.append(
                V1SampledDualChartRowShardAttachment(
                    path=_shard_path(
                        directory,
                        look.literal_j_shard.receipt_hash,
                    ),
                    receipt=look.literal_j_shard,
                )
            )
        attachment = V1SampledDualChartRowLivePrefixAttachment(
            manifest=prospective_manifest,
            corrected_a_attachments=tuple(a_attachments),
            literal_j_attachments=tuple(j_attachments),
        )
        attachment.verify_live_witnesses()
        return V1SampledDualChartRowPreparedGroupRecovery(
            plan=plan,
            receipt=receipt,
            prospective_manifest=prospective_manifest,
            attachment=attachment,
            head_already_published=head == successor,
            directory=directory,
            manifest_path=manifest_path,
            manifest_invariant=manifest_invariant,
            canonical_manifest_bytes=canonical_manifest_bytes,
            lock=lock,
        )
    except Exception:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally:
            lock.close()
        raise


def append_v1_sampled_dual_chart_row_chunks(
    *,
    request: V1SampledDualChartRowLookRequest,
    chunks: Sequence[V1SampledDualChartRowChunk] | Iterator[V1SampledDualChartRowChunk],
    prior: V1SampledDualChartRowLivePrefixAttachment | None = None,
) -> V1SampledDualChartRowLivePrefixAttachment:
    """Atomically append one bounded same-capture A/J route stream."""

    if type(request) is not V1SampledDualChartRowLookRequest:
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            "append requires its exact look request",
        )
    request.assert_integrity()
    request.scratch_forecast.assert_admissible()
    directory = Path(request.scratch_directory)
    lock_path = directory / _LOCK_NAME
    manifest_path: Path | None = None
    head_published = False
    a_writer: _OpenRoleShardWriter | None = None
    j_writer: _OpenRoleShardWriter | None = None
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _remove_unpublished_partial_files(directory)
            live_head = _read_head_hash(directory)
            expected_head = None if prior is None else prior.manifest.manifest_hash
            if live_head != expected_head:
                _refuse(
                    V1SampledDualChartRowRefusalCode.PUBLICATION_CONFLICT,
                    "published prefix HEAD differs from the append predecessor",
                )
            _validate_extension(request, prior)
            free, _ = _scratch_capacity(directory)
            if free < request.scratch_forecast.aggregate_allocation_bytes_upper_bound:
                _refuse(
                    V1SampledDualChartRowRefusalCode.SCRATCH_INADMISSIBLE,
                    "live scratch fell below the aggregate two-shard forecast",
                )
            a_writer = _OpenRoleShardWriter(
                request=request,
                role=V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY,
            )
            j_writer = _OpenRoleShardWriter(
                request=request,
                role=V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN,
            )
            expected_windows = iter(
                _route_windows(
                    request.route_count,
                    request.registered_route_chunk_size,
                )
            )
            observed_count = 0
            for chunk in chunks:
                if type(chunk) is not V1SampledDualChartRowChunk:
                    _refuse(
                        V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                        "chunk append requires exact dual-role chunk records",
                    )
                try:
                    expected_start, expected_stop = next(expected_windows)
                except StopIteration:
                    _refuse(
                        V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                        "dual-row source emitted trailing route chunks",
                    )
                if (chunk.start, chunk.stop) != (expected_start, expected_stop):
                    _refuse(
                        V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                        "dual-row source changed registered route-window order",
                    )
                a_source = a_writer.consume(
                    start=chunk.start,
                    stop=chunk.stop,
                    rows=chunk.corrected_a_rows,
                )
                j_source = j_writer.consume(
                    start=chunk.start,
                    stop=chunk.stop,
                    rows=chunk.literal_j_rows,
                )
                if a_source != j_source:
                    _refuse(
                        V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
                        "same-capture corrected-A and literal-J chunks changed device kind",
                    )
                observed_count += 1
            try:
                next(expected_windows)
            except StopIteration:
                pass
            else:
                _refuse(
                    V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                    "dual-row source ended before complete route coverage",
                )
            if observed_count == 0:
                _refuse(
                    V1SampledDualChartRowRefusalCode.AXIS_MISMATCH,
                    "dual-row source emitted no registered route windows",
                )
            a_receipt, a_attachment, _ = a_writer.seal()
            j_receipt, j_attachment, _ = j_writer.seal()
            look = V1SampledDualChartRowLookReceipt(
                look_ordinal=request.look_ordinal,
                prior_manifest_hash=request.expected_prior_manifest_hash,
                request_hash=request.request_hash,
                context_ids=request.context_ids,
                draw_ids=request.draw_ids,
                stratum_labels=request.stratum_labels,
                corrected_a_shard=a_receipt,
                literal_j_shard=j_receipt,
            )
            old_looks = () if prior is None else prior.manifest.looks
            manifest = V1SampledDualChartRowPrefixManifest(
                prior_manifest_hash=request.expected_prior_manifest_hash,
                registered_d_prefix_hash=request.registered_d_prefix_hash,
                chart_identity_hash=request.chart_identity_hash,
                current_mask_hash=request.current_mask_hash,
                proposal_generation=request.proposal_generation,
                route_axis_hash=request.route_axis_hash,
                route_id_axis_hash=request.route_id_axis_hash,
                route_count=request.route_count,
                corrected_a_stream_identity_hash=(
                    request.corrected_a_stream_identity_hash
                ),
                literal_j_stream_identity_hash=request.literal_j_stream_identity_hash,
                looks=old_looks + (look,),
            )
            manifest_path = _write_manifest(directory, manifest)
            prior_a = () if prior is None else prior.corrected_a_attachments
            prior_j = () if prior is None else prior.literal_j_attachments
            result = V1SampledDualChartRowLivePrefixAttachment(
                manifest=manifest,
                corrected_a_attachments=prior_a + (a_attachment,),
                literal_j_attachments=prior_j + (j_attachment,),
            )
            # Publication is last: HEAD is the only visibility boundary.
            _atomic_publish_head(directory, manifest.manifest_hash)
            head_published = True
            return result
        except Exception:
            # If a fault occurred after the atomic replace, restore the exact
            # predecessor before deleting the unpublished look's storage.
            try:
                if head_published or _read_head_hash(directory) not in {
                    None if prior is None else prior.manifest.manifest_hash,
                }:
                    _restore_head(
                        directory,
                        None if prior is None else prior.manifest.manifest_hash,
                    )
            finally:
                if manifest_path is not None:
                    manifest_path.unlink(missing_ok=True)
                if a_writer is not None:
                    a_writer.abort()
                if j_writer is not None:
                    j_writer.abort()
                _fsync_directory(directory)
            raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def append_v1_sampled_dual_chart_row_look(
    *,
    request: V1SampledDualChartRowLookRequest,
    corrected_a_rows: Any,
    literal_j_rows: Any,
    prior: V1SampledDualChartRowLivePrefixAttachment | None = None,
) -> V1SampledDualChartRowLivePrefixAttachment:
    """Convenience append for an already-resident exact row pair.

    Production capture paths should prefer
    :func:`append_v1_sampled_dual_chart_row_chunks` so neither complete role
    bank needs to coexist in host or device memory.
    """

    if type(request) is not V1SampledDualChartRowLookRequest:
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            "append requires its exact look request",
        )
    corrected_source = _source_descriptor(corrected_a_rows, request)
    literal_j_source = _source_descriptor(literal_j_rows, request)
    if corrected_source != literal_j_source:
        _refuse(
            V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
            "same-capture corrected-A and literal-J rows changed source device kind",
        )

    def _chunks() -> Iterator[V1SampledDualChartRowChunk]:
        for start, stop in _route_windows(
            request.route_count,
            request.registered_route_chunk_size,
        ):
            yield V1SampledDualChartRowChunk(
                start=start,
                stop=stop,
                corrected_a_rows=corrected_a_rows[:, start:stop],
                literal_j_rows=literal_j_rows[:, start:stop],
            )

    return append_v1_sampled_dual_chart_row_chunks(
        request=request,
        chunks=_chunks(),
        prior=prior,
    )


def _receipt_from_dict(payload: Mapping[str, Any]) -> V1SampledDualChartRowShardReceipt:
    receipt = V1SampledDualChartRowShardReceipt(
        request_hash=payload["request_hash"],
        scratch_forecast_hash=payload["scratch_forecast_hash"],
        role=V1SampledDualChartRowRole(payload["role"]),
        look_ordinal=payload["look_ordinal"],
        registered_d_prefix_hash=payload["registered_d_prefix_hash"],
        source_capture_identity_hash=payload["source_capture_identity_hash"],
        chart_identity_hash=payload["chart_identity_hash"],
        current_mask_hash=payload["current_mask_hash"],
        proposal_generation=payload["proposal_generation"],
        route_axis_hash=payload["route_axis_hash"],
        route_id_axis_hash=payload["route_id_axis_hash"],
        route_count=payload["route_count"],
        stream_identity_hash=payload["stream_identity_hash"],
        context_ids=tuple(payload["context_ids"]),
        draw_ids=tuple(payload["draw_ids"]),
        stratum_labels=tuple(payload["stratum_labels"]),
        registered_route_chunk_size=payload["registered_route_chunk_size"],
        route_windows=tuple(tuple(pair) for pair in payload["route_windows"]),
        exact_payload_file_bytes=payload["exact_payload_file_bytes"],
        content_sha256=payload["content_sha256"],
        source_array_kind=payload["source_array_kind"],
        source_device=payload["source_device"],
        bounded_host_chunk_copy_count=payload["bounded_host_chunk_copy_count"],
        bounded_host_chunk_copy_bytes=payload["bounded_host_chunk_copy_bytes"],
        d2h_transfer_count=payload["d2h_transfer_count"],
        d2h_transfer_bytes=payload["d2h_transfer_bytes"],
        file_write_call_count=payload["file_write_call_count"],
        file_write_bytes=payload["file_write_bytes"],
        maximum_owned_host_staging_bytes=payload["maximum_owned_host_staging_bytes"],
        preallocation_method=payload["preallocation_method"],
        _issuance_token=_ISSUANCE_TOKEN,
    )
    receipt.assert_integrity(payload["receipt_hash"])
    if canonical_json(receipt.to_dict()) != canonical_json(dict(payload)):
        _refuse(
            V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
            "serialized shard receipt has omitted or extra fields",
        )
    return receipt


def _manifest_from_dict(
    payload: Mapping[str, Any]
) -> V1SampledDualChartRowPrefixManifest:
    looks = []
    for raw in payload["looks"]:
        a = _receipt_from_dict(raw["corrected_a_shard"])
        j = _receipt_from_dict(raw["literal_j_shard"])
        look = V1SampledDualChartRowLookReceipt(
            look_ordinal=raw["look_ordinal"],
            prior_manifest_hash=raw["prior_manifest_hash"],
            request_hash=raw["request_hash"],
            context_ids=tuple(raw["context_ids"]),
            draw_ids=tuple(raw["draw_ids"]),
            stratum_labels=tuple(raw["stratum_labels"]),
            corrected_a_shard=a,
            literal_j_shard=j,
        )
        look.assert_integrity(raw["look_hash"])
        looks.append(look)
    manifest = V1SampledDualChartRowPrefixManifest(
        prior_manifest_hash=payload["prior_manifest_hash"],
        registered_d_prefix_hash=payload["registered_d_prefix_hash"],
        chart_identity_hash=payload["chart_identity_hash"],
        current_mask_hash=payload["current_mask_hash"],
        proposal_generation=payload["proposal_generation"],
        route_axis_hash=payload["route_axis_hash"],
        route_id_axis_hash=payload["route_id_axis_hash"],
        route_count=payload["route_count"],
        corrected_a_stream_identity_hash=payload["corrected_a_stream_identity_hash"],
        literal_j_stream_identity_hash=payload["literal_j_stream_identity_hash"],
        looks=tuple(looks),
    )
    manifest.assert_integrity(payload["manifest_hash"])
    if canonical_json(manifest.to_dict()) != canonical_json(dict(payload)):
        _refuse(
            V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
            "serialized prefix manifest has omitted or extra fields",
        )
    return manifest


def reattach_v1_sampled_dual_chart_row_prefix(
    *,
    scratch_directory: str | os.PathLike[str],
    manifest: V1SampledDualChartRowPrefixManifest | None = None,
) -> V1SampledDualChartRowLivePrefixAttachment:
    """Reattach a published prefix after full size/digest verification."""

    directory = _scratch_directory(scratch_directory)
    head = _read_head_hash(directory)
    if head is None:
        _refuse(
            V1SampledDualChartRowRefusalCode.MISSING_SHARD,
            "sampled dual-row prefix HEAD is absent",
        )
    supplied_manifest = manifest
    if (
        supplied_manifest is not None
        and type(supplied_manifest) is not V1SampledDualChartRowPrefixManifest
    ):
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            "reattachment manifest must be exact",
        )
    path = _manifest_path(directory, head)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V1SampledDualChartRowRefusal(
            V1SampledDualChartRowRefusalCode.MISSING_SHARD,
            "published sampled dual-row manifest is absent or unreadable",
        ) from error
    manifest = _manifest_from_dict(raw)
    if supplied_manifest is not None:
        supplied_manifest.assert_integrity(head)
        if canonical_json(supplied_manifest.to_dict()) != canonical_json(
            manifest.to_dict()
        ):
            _refuse(
                V1SampledDualChartRowRefusalCode.CONTENT_MISMATCH,
                "supplied manifest differs from the atomically published manifest",
            )
    manifest.assert_integrity(head)
    a_attachments = []
    j_attachments = []
    for look in manifest.looks:
        a_attachments.append(
            V1SampledDualChartRowShardAttachment(
                path=_shard_path(directory, look.corrected_a_shard.receipt_hash),
                receipt=look.corrected_a_shard,
            )
        )
        j_attachments.append(
            V1SampledDualChartRowShardAttachment(
                path=_shard_path(directory, look.literal_j_shard.receipt_hash),
                receipt=look.literal_j_shard,
            )
        )
    result = V1SampledDualChartRowLivePrefixAttachment(
        manifest=manifest,
        corrected_a_attachments=tuple(a_attachments),
        literal_j_attachments=tuple(j_attachments),
    )
    # Every cold shard attachment already performed exactly one complete
    # role-domain digest; this prefix check must remain metadata-only.
    result.verify_live_witnesses()
    return result


def materialize_v1_sampled_dual_chart_rows(
    *,
    attachment: V1SampledDualChartRowLivePrefixAttachment,
    stratum_masses: Mapping[str, float],
) -> V1SampledDualChartMaterializedRows:
    """Apply deterministic ``pi_h / n_h`` weights without changing raw shards."""

    if type(attachment) is not V1SampledDualChartRowLivePrefixAttachment:
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            "materialization requires an exact live prefix attachment",
        )
    attachment.verify_live_witnesses()
    labels = attachment.manifest.stratum_labels
    required = set(labels)
    if set(stratum_masses) != required:
        _refuse(
            V1SampledDualChartRowRefusalCode.STALE_IDENTITY,
            "stratum masses must name exactly the sampled prefix strata",
        )
    masses: dict[str, float] = {}
    for label in sorted(required):
        value = float(stratum_masses[label])
        if not math.isfinite(value) or value < 0.0:
            _refuse(
                V1SampledDualChartRowRefusalCode.INVALID_TYPE,
                "stratum masses must be finite and nonnegative",
            )
        masses[label] = value
    if not math.isclose(sum(masses.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        _refuse(
            V1SampledDualChartRowRefusalCode.INVALID_TYPE,
            "stratum masses must sum to one",
        )
    counts = {label: labels.count(label) for label in sorted(required)}
    weights = np.asarray(
        [masses[label] / counts[label] for label in labels],
        dtype=np.float64,
    )
    return V1SampledDualChartMaterializedRows(
        attachment=attachment,
        manifest_hash=attachment.manifest.manifest_hash,
        context_ids=attachment.manifest.context_ids,
        draw_ids=attachment.manifest.draw_ids,
        stratum_labels=labels,
        stratum_masses=tuple(masses.items()),
        stratum_sample_counts=tuple(counts.items()),
        normalized_context_weights=weights,
    )


__all__ = [
    "SAMPLED_DUAL_CHART_ROW_AUTHORITY_SCOPE",
    "SAMPLED_DUAL_CHART_ROW_FORECAST_SCHEMA",
    "SAMPLED_DUAL_CHART_ROW_GROUP_COMMIT_RECEIPT_SCHEMA",
    "SAMPLED_DUAL_CHART_ROW_GROUP_FORECAST_SCHEMA",
    "SAMPLED_DUAL_CHART_ROW_GROUP_PACKET_PLAN_SCHEMA",
    "SAMPLED_DUAL_CHART_ROW_GROUP_PLAN_SCHEMA",
    "SAMPLED_DUAL_CHART_ROW_GROUP_PROOF_CLOSURE_SCHEMA",
    "SAMPLED_DUAL_CHART_ROW_LAYOUT",
    "SAMPLED_DUAL_CHART_ROW_LOOK_SCHEMA",
    "SAMPLED_DUAL_CHART_ROW_MATERIALIZATION_SCHEMA",
    "SAMPLED_DUAL_CHART_ROW_PREFIX_MANIFEST_SCHEMA",
    "SAMPLED_DUAL_CHART_ROW_PREPARED_GROUP_RECOVERY_RESULT_SCHEMA",
    "SAMPLED_DUAL_CHART_ROW_REQUEST_SCHEMA",
    "SAMPLED_DUAL_CHART_ROW_SHARD_RECEIPT_SCHEMA",
    "SAMPLED_DUAL_CHART_ROW_VERIFIED_PACKET_SCHEMA",
    "V1SampledDualChartMaterializedRows",
    "V1SampledDualChartRowChunk",
    "V1SampledDualChartRowGroupCommit",
    "V1SampledDualChartRowGroupCommitReceipt",
    "V1SampledDualChartRowGroupPacketPlan",
    "V1SampledDualChartRowGroupPacketProofClosure",
    "V1SampledDualChartRowGroupPlan",
    "V1SampledDualChartRowGroupScratchForecast",
    "V1SampledDualChartRowGroupStagingConsumer",
    "V1SampledDualChartRowGroupTransaction",
    "V1SampledDualChartRowLivePrefixAttachment",
    "V1SampledDualChartRowLookReceipt",
    "V1SampledDualChartRowLookRequest",
    "V1SampledDualChartRowPrefixManifest",
    "V1SampledDualChartRowPreparedGroupCommit",
    "V1SampledDualChartRowPreparedGroupRecovery",
    "V1SampledDualChartRowPreparedGroupRecoveryResult",
    "V1SampledDualChartRowRefusal",
    "V1SampledDualChartRowRefusalCode",
    "V1SampledDualChartRowRole",
    "V1SampledDualChartRowScratchForecast",
    "V1SampledDualChartRowShardAttachment",
    "V1SampledDualChartRowShardReceipt",
    "V1SampledDualChartRowShardVerificationTelemetry",
    "V1SampledDualChartRowVerifiedPacket",
    "append_v1_sampled_dual_chart_row_look",
    "append_v1_sampled_dual_chart_row_chunks",
    "begin_v1_sampled_dual_chart_row_group_transaction",
    "forecast_v1_sampled_dual_chart_row_group_scratch",
    "forecast_v1_sampled_dual_chart_row_scratch",
    "materialize_v1_sampled_dual_chart_rows",
    "prepare_v1_sampled_dual_chart_row_look",
    "prepare_v1_sampled_dual_chart_row_group_plan",
    "reattach_v1_sampled_dual_chart_row_prefix",
    "reattach_v1_sampled_dual_chart_row_prepared_group",
]
