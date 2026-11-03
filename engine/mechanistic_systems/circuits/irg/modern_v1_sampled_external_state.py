"""Typed external continuation state for sampled modern-v1 discovery.

One registered proposal look produces two coherent sufficient states:

* stratified corrected-attribution moments; and
* the packet-boundary stratified Frequent-Directions accumulator snapshot.

This module publishes them in *one* immutable storage transaction.  The
returned descriptor contains no filesystem root.  Moving the published state
directory to another node-local root therefore preserves the exact commit
identity.

The numerical arrays reopen as verified, read-only memory mappings.  The
existing domain constructors intentionally copy arrays into immutable
``bytes`` backing stores, so materializing the legacy in-memory objects cannot
be zero-copy.  That conversion is explicit and its conservative peak-copy
ledger is part of the descriptor; ordinary reopening does not perform it.

The route-ID axis is not duplicated in this artifact.  A caller restoring an
FD snapshot must restate the current :class:`V1StratifiedFDDesign`; its design
hash, route count, and ordered route-axis hash are checked before any domain
object is materialized.  This is the same authority boundary as the canonical
FD continuation API: checkpoint bytes may continue a declared design, but may
not redefine it.

No pickle or executable codec participates in publication or opening.  This
state is discovery-continuation evidence only.  It binds common proposal,
route, stratum, and sampled-prefix identities, but is not the raw-VJP tee's
execution receipt; the enclosing runtime transaction must retain that receipt
separately.  This state cannot certify F/Q, open a BCE oracle, or commit a mask.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import os
from types import MappingProxyType
from typing import Final, NoReturn, cast

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_external_state import (
    V1ExternalArrayIdentity,
    V1ExternalArrayPolicy,
    V1ExternalStateDescriptor,
    V1ExternalStateLiveAttachment,
    V1ExternalStateView,
    open_external_state,
    open_external_state_with_attachment,
    publish_external_state,
    publish_external_state_with_attachment,
)
from mechanistic_systems.circuits.irg.modern_v1_proposal_statistics import (
    CORRECTED_ATTRIBUTION_MOMENT_REPRESENTATION,
    STRATIFIED_CORRECTED_ATTRIBUTION_MOMENTS_SCHEMA,
    V1StratifiedCorrectedAttributionMoments,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_look_transaction import (
    V1SampledDualChartRegisteredLookReceipt,
)
from mechanistic_systems.circuits.irg.modern_v1_stratified_fd_gpu import (
    STRATIFIED_FD_ACCUMULATOR_SNAPSHOT_SCHEMA,
    V1StratifiedFDAccumulatorSnapshot,
    V1StratifiedFDDesign,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.measurement.device_route_moments import (
    BoundedOrderedFDSourceProofSnapshot,
    DeviceFrequentDirectionsTrustAccumulatorSnapshot,
)


SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA: Final = (
    "partition_free_v1_sampled_registered_look_external_descriptor_v1"
)
SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_dual_chart_registered_look_external_descriptor_v2"
)
SAMPLED_REGISTERED_LOOK_EXTERNAL_STATE_SCHEMA: Final = (
    "partition_free_v1_sampled_registered_look_external_state_v1"
)
SAMPLED_REGISTERED_LOOK_EXTERNAL_STATE_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_dual_chart_registered_look_external_state_v2"
)
SAMPLED_REGISTERED_LOOK_METADATA_SCHEMA: Final = (
    "partition_free_v1_sampled_registered_look_metadata_v1"
)
SAMPLED_REGISTERED_LOOK_METADATA_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_dual_chart_registered_look_metadata_v2"
)
SAMPLED_EXTERNAL_MEMORY_FORECAST_SCHEMA: Final = (
    "partition_free_v1_sampled_external_memory_forecast_v1"
)
SAMPLED_EXTERNAL_NAMESPACE: Final = "modern_v1_sampled_registered_look"
SAMPLED_EXTERNAL_AUTHORITY_SCOPE: Final = (
    "discovery_continuation_only_no_fq_bce_or_mask_authority_v1"
)
SAMPLED_EXTERNAL_MEMORY_SCOPE: Final = (
    "canonical_array_payload_and_bounded_codec_buffers_excluding_os_page_cache_"
    "filesystem_cache_allocator_metadata_and_caller_owned_route_axis_v1"
)

_COMMIT_PREFIX: Final = "partition-free-v1-sampled-registered-look-external-"
_MOMENTS_PREFIX: Final = "partition-free-v1-corrected-attribution-moments-"
_DEVICE_FD_PREFIX: Final = "irg-device-fd-accumulator-snapshot-"
_STRATIFIED_FD_PREFIX: Final = "partition-free-v1-stratified-fd-snapshot-"
_METADATA_NAME: Final = "metadata_utf8"
_MOMENT_ARRAY_NAMES: Final = (
    "moments_absolute_m2",
    "moments_absolute_mean",
    "moments_signed_m2",
    "moments_signed_mean",
)
_PUBLISH_BUFFER_ELEMENTS: Final = 1024 * 1024
_VERIFY_BUFFER_BYTES: Final = 8 * 1024 * 1024
_TYPED_VALIDATION_CHUNK_ELEMENTS: Final = 128 * 1024
_TYPED_VALIDATION_FLOAT_BUFFERS: Final = 2
_TYPED_VALIDATION_BOOLEAN_BUFFERS: Final = 1
_LEDGER_UNIQUENESS_FIXED_BYTES_UPPER_BOUND: Final = 1024
_LEDGER_UNIQUENESS_BYTES_PER_ENTRY_UPPER_BOUND: Final = 64
_LE_F64: Final = np.dtype("<f8")
_LE_U64: Final = np.dtype("<u8")
_U8: Final = np.dtype("u1")
_NEGATIVE_F64_ZERO_BITS: Final = np.uint64(1 << 63)


class V1SampledExternalStateRefusalCode(str, Enum):
    """Stable typed refusal categories above the generic external store."""

    INVALID_TYPE = "invalid_type"
    INVALID_SCHEMA = "invalid_schema"
    INCOHERENT_REGISTERED_LOOK = "incoherent_registered_look"
    TYPED_IDENTITY_MISMATCH = "typed_identity_mismatch"
    ROUTE_DESIGN_MISMATCH = "route_design_mismatch"
    CLOSED_VIEW = "closed_view"


class V1SampledExternalStateRefusal(ValueError):
    """Fail-closed typed-state error."""

    def __init__(self, code: V1SampledExternalStateRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1SampledExternalStateRefusalCode, message: str) -> NoReturn:
    raise V1SampledExternalStateRefusal(code, message)


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or cast(int, value) < 0:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must be an exact nonnegative integer",
        )
    return cast(int, value)


def _exact_positive_int(value: object, *, name: str) -> int:
    result = _exact_nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must be positive",
        )
    return result


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not cast(str, value):
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must be one nonempty string",
        )
    return cast(str, value)


def _exact_mapping(
    value: object,
    *,
    keys: set[str],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must be one exact object",
        )
    result = cast(dict[str, object], value)
    if set(result) != keys:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} fields are not exact",
        )
    return result


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {list, tuple}:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must be an ordered string sequence",
        )
    result = tuple(
        _identity(item, name=f"{name} item") for item in cast(Sequence[object], value)
    )
    return result


def _pair_of_strings(value: object, *, name: str) -> tuple[str, str]:
    result = _string_tuple(value, name=name)
    if len(result) != 2:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must contain exactly two values",
        )
    return cast(tuple[str, str], result)


def _pair_of_ints(value: object, *, name: str) -> tuple[int, int]:
    if type(value) not in {list, tuple}:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must contain exactly two integers",
        )
    result = tuple(
        _exact_nonnegative_int(item, name=f"{name} item")
        for item in cast(Sequence[object], value)
    )
    if len(result) != 2:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must contain exactly two integers",
        )
    return cast(tuple[int, int], result)


def _pair_of_positive_ints(value: object, *, name: str) -> tuple[int, int]:
    result = _pair_of_ints(value, name=name)
    if any(item <= 0 for item in result):
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must contain exactly two positive integers",
        )
    return result


def _nested_pair(
    value: object, *, name: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(value) not in {list, tuple} or len(cast(Sequence[object], value)) != 2:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            f"{name} must contain exactly two ledgers",
        )
    rows = cast(Sequence[object], value)
    return (
        _string_tuple(rows[0], name=f"{name} first ledger"),
        _string_tuple(rows[1], name=f"{name} second ledger"),
    )


def _fd_array_names(index: int) -> tuple[str, str, str]:
    prefix = f"fd_{index:08d}"
    return (
        f"{prefix}_active_buffer_rows",
        f"{prefix}_mean",
        f"{prefix}_weighted_second_moment_diagonal",
    )


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _strict_metadata(value: np.ndarray) -> dict[str, object]:
    if value.dtype != _U8 or value.ndim != 1 or value.flags.writeable:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            "metadata must be a read-only uint8 vector",
        )

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                _refuse(
                    V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                    f"metadata repeats field {key!r}",
                )
            result[key] = item
        return result

    raw = value.tobytes(order="C")
    try:
        decoded = json.loads(raw.decode("ascii"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            f"metadata JSON is invalid: {error}",
        )
    if type(decoded) is not dict:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            "registered-look metadata must be one exact object",
        )
    schema = decoded.get("schema")
    expected_keys = (
        {"schema", "moments", "fd", "dual_chart"}
        if schema == SAMPLED_REGISTERED_LOOK_METADATA_SCHEMA_V2
        else {"schema", "moments", "fd"}
    )
    metadata = _exact_mapping(
        decoded,
        keys=expected_keys,
        name="registered-look metadata",
    )
    if raw != _canonical_json_bytes(metadata):
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            "metadata is not canonical ASCII JSON",
        )
    if metadata["schema"] not in {
        SAMPLED_REGISTERED_LOOK_METADATA_SCHEMA,
        SAMPLED_REGISTERED_LOOK_METADATA_SCHEMA_V2,
    }:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            "registered-look metadata schema is invalid",
        )
    return metadata


@dataclass(frozen=True, slots=True)
class V1SampledExternalMemoryForecast:
    """Exact payload ledger and conservative codec-copy upper bounds.

    The ledger excludes OS page cache, filesystem cache, Python allocator
    bookkeeping, and the caller-owned route-ID axis.  ``materialization`` means
    explicitly rebuilding the legacy domain objects; opening the mapped view
    performs no route-scale heap copy.  Typed-open semantic validation carries
    separate deterministic bounds for its fixed-size vector scratch and the
    one-at-a-time metadata-ledger uniqueness set.
    """

    moments_numeric_payload_bytes: int
    fd_numeric_payload_bytes: int
    metadata_payload_bytes: int
    persisted_payload_bytes: int
    largest_moments_array_bytes: int
    largest_fd_array_bytes: int
    publication_stream_chunk_bytes_upper_bound: int
    publication_additional_peak_host_copy_bytes_upper_bound: int
    verification_stream_buffer_bytes_upper_bound: int
    metadata_decode_peak_host_bytes_upper_bound: int
    typed_open_invariant_chunk_elements: int
    typed_open_invariant_scratch_bytes_upper_bound: int
    typed_open_ledger_uniqueness_entry_count: int
    typed_open_ledger_uniqueness_scratch_bytes_upper_bound: int
    typed_open_additional_peak_host_scratch_bytes_upper_bound: int
    reopened_numeric_heap_copy_bytes: int
    moments_materialized_owned_bytes: int
    moments_materialization_peak_heap_bytes_upper_bound: int
    fd_materialized_owned_bytes: int
    fd_materialization_peak_heap_bytes_upper_bound: int
    combined_materialization_peak_heap_bytes_upper_bound: int
    route_axis_payload_stored_bytes: int = 0
    caller_restates_fd_design: bool = True
    pickle_codec_used: bool = False
    memory_accounting_scope: str = SAMPLED_EXTERNAL_MEMORY_SCOPE
    schema: str = SAMPLED_EXTERNAL_MEMORY_FORECAST_SCHEMA

    def __post_init__(self) -> None:
        numeric_fields = (
            "moments_numeric_payload_bytes",
            "fd_numeric_payload_bytes",
            "metadata_payload_bytes",
            "persisted_payload_bytes",
            "largest_moments_array_bytes",
            "largest_fd_array_bytes",
            "publication_stream_chunk_bytes_upper_bound",
            "publication_additional_peak_host_copy_bytes_upper_bound",
            "verification_stream_buffer_bytes_upper_bound",
            "metadata_decode_peak_host_bytes_upper_bound",
            "typed_open_invariant_chunk_elements",
            "typed_open_invariant_scratch_bytes_upper_bound",
            "typed_open_ledger_uniqueness_entry_count",
            "typed_open_ledger_uniqueness_scratch_bytes_upper_bound",
            "typed_open_additional_peak_host_scratch_bytes_upper_bound",
            "reopened_numeric_heap_copy_bytes",
            "moments_materialized_owned_bytes",
            "moments_materialization_peak_heap_bytes_upper_bound",
            "fd_materialized_owned_bytes",
            "fd_materialization_peak_heap_bytes_upper_bound",
            "combined_materialization_peak_heap_bytes_upper_bound",
            "route_axis_payload_stored_bytes",
        )
        for name in numeric_fields:
            object.__setattr__(
                self,
                name,
                _exact_nonnegative_int(getattr(self, name), name=name),
            )
        moments = self.moments_numeric_payload_bytes
        fd = self.fd_numeric_payload_bytes
        metadata = self.metadata_payload_bytes
        largest_moments = self.largest_moments_array_bytes
        largest_fd = self.largest_fd_array_bytes
        route_count = largest_moments // (2 * _LE_F64.itemsize)
        validation_elements = min(route_count, _TYPED_VALIDATION_CHUNK_ELEMENTS)
        validation_scratch = validation_elements * (
            _TYPED_VALIDATION_FLOAT_BUFFERS * _LE_F64.itemsize
            + _TYPED_VALIDATION_BOOLEAN_BUFFERS * _U8.itemsize
        )
        ledger_scratch = (
            _LEDGER_UNIQUENESS_FIXED_BYTES_UPPER_BOUND
            + self.typed_open_ledger_uniqueness_entry_count
            * _LEDGER_UNIQUENESS_BYTES_PER_ENTRY_UPPER_BOUND
        )
        expected = {
            "persisted_payload_bytes": moments + fd + metadata,
            "publication_additional_peak_host_copy_bytes_upper_bound": (
                2 * metadata + 2 * self.publication_stream_chunk_bytes_upper_bound
            ),
            "metadata_decode_peak_host_bytes_upper_bound": 2 * metadata,
            "typed_open_invariant_chunk_elements": validation_elements,
            "typed_open_invariant_scratch_bytes_upper_bound": validation_scratch,
            "typed_open_ledger_uniqueness_scratch_bytes_upper_bound": ledger_scratch,
            "typed_open_additional_peak_host_scratch_bytes_upper_bound": max(
                validation_scratch,
                ledger_scratch,
            ),
            "reopened_numeric_heap_copy_bytes": 0,
            "moments_materialized_owned_bytes": moments,
            "moments_materialization_peak_heap_bytes_upper_bound": (
                moments + largest_moments
            ),
            "fd_materialized_owned_bytes": fd,
            "fd_materialization_peak_heap_bytes_upper_bound": fd + largest_fd,
            "combined_materialization_peak_heap_bytes_upper_bound": (
                moments + fd + max(largest_moments, largest_fd)
            ),
            "route_axis_payload_stored_bytes": 0,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                _refuse(
                    V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                    f"{name} does not replay from the memory ledger",
                )
        if (
            self.caller_restates_fd_design is not True
            or self.pickle_codec_used is not False
            or self.memory_accounting_scope != SAMPLED_EXTERNAL_MEMORY_SCOPE
            or self.schema != SAMPLED_EXTERNAL_MEMORY_FORECAST_SCHEMA
        ):
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                "external-state memory/codec policy is invalid",
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "moments_numeric_payload_bytes": self.moments_numeric_payload_bytes,
            "fd_numeric_payload_bytes": self.fd_numeric_payload_bytes,
            "metadata_payload_bytes": self.metadata_payload_bytes,
            "persisted_payload_bytes": self.persisted_payload_bytes,
            "largest_moments_array_bytes": self.largest_moments_array_bytes,
            "largest_fd_array_bytes": self.largest_fd_array_bytes,
            "publication_stream_chunk_bytes_upper_bound": (
                self.publication_stream_chunk_bytes_upper_bound
            ),
            "publication_additional_peak_host_copy_bytes_upper_bound": (
                self.publication_additional_peak_host_copy_bytes_upper_bound
            ),
            "verification_stream_buffer_bytes_upper_bound": (
                self.verification_stream_buffer_bytes_upper_bound
            ),
            "metadata_decode_peak_host_bytes_upper_bound": (
                self.metadata_decode_peak_host_bytes_upper_bound
            ),
            "typed_open_invariant_chunk_elements": (
                self.typed_open_invariant_chunk_elements
            ),
            "typed_open_invariant_scratch_bytes_upper_bound": (
                self.typed_open_invariant_scratch_bytes_upper_bound
            ),
            "typed_open_ledger_uniqueness_entry_count": (
                self.typed_open_ledger_uniqueness_entry_count
            ),
            "typed_open_ledger_uniqueness_scratch_bytes_upper_bound": (
                self.typed_open_ledger_uniqueness_scratch_bytes_upper_bound
            ),
            "typed_open_additional_peak_host_scratch_bytes_upper_bound": (
                self.typed_open_additional_peak_host_scratch_bytes_upper_bound
            ),
            "reopened_numeric_heap_copy_bytes": self.reopened_numeric_heap_copy_bytes,
            "moments_materialized_owned_bytes": self.moments_materialized_owned_bytes,
            "moments_materialization_peak_heap_bytes_upper_bound": (
                self.moments_materialization_peak_heap_bytes_upper_bound
            ),
            "fd_materialized_owned_bytes": self.fd_materialized_owned_bytes,
            "fd_materialization_peak_heap_bytes_upper_bound": (
                self.fd_materialization_peak_heap_bytes_upper_bound
            ),
            "combined_materialization_peak_heap_bytes_upper_bound": (
                self.combined_materialization_peak_heap_bytes_upper_bound
            ),
            "route_axis_payload_stored_bytes": self.route_axis_payload_stored_bytes,
            "caller_restates_fd_design": self.caller_restates_fd_design,
            "pickle_codec_used": self.pickle_codec_used,
            "memory_accounting_scope": self.memory_accounting_scope,
        }

    @classmethod
    def from_json_dict(cls, value: object) -> "V1SampledExternalMemoryForecast":
        expected = {
            "schema",
            "moments_numeric_payload_bytes",
            "fd_numeric_payload_bytes",
            "metadata_payload_bytes",
            "persisted_payload_bytes",
            "largest_moments_array_bytes",
            "largest_fd_array_bytes",
            "publication_stream_chunk_bytes_upper_bound",
            "publication_additional_peak_host_copy_bytes_upper_bound",
            "verification_stream_buffer_bytes_upper_bound",
            "metadata_decode_peak_host_bytes_upper_bound",
            "typed_open_invariant_chunk_elements",
            "typed_open_invariant_scratch_bytes_upper_bound",
            "typed_open_ledger_uniqueness_entry_count",
            "typed_open_ledger_uniqueness_scratch_bytes_upper_bound",
            "typed_open_additional_peak_host_scratch_bytes_upper_bound",
            "reopened_numeric_heap_copy_bytes",
            "moments_materialized_owned_bytes",
            "moments_materialization_peak_heap_bytes_upper_bound",
            "fd_materialized_owned_bytes",
            "fd_materialization_peak_heap_bytes_upper_bound",
            "combined_materialization_peak_heap_bytes_upper_bound",
            "route_axis_payload_stored_bytes",
            "caller_restates_fd_design",
            "pickle_codec_used",
            "memory_accounting_scope",
        }
        payload = _exact_mapping(value, keys=expected, name="memory forecast")
        return cls(**payload)  # type: ignore[arg-type]


def _forecast_memory(
    *,
    moment_arrays: Sequence[np.ndarray],
    fd_arrays: Sequence[np.ndarray],
    metadata_bytes: int,
    ledger_uniqueness_entry_count: int,
) -> V1SampledExternalMemoryForecast:
    moments = sum(int(value.nbytes) for value in moment_arrays)
    fd = sum(int(value.nbytes) for value in fd_arrays)
    largest_moments = max((int(value.nbytes) for value in moment_arrays), default=0)
    largest_fd = max((int(value.nbytes) for value in fd_arrays), default=0)
    all_arrays = (
        tuple(moment_arrays) + tuple(fd_arrays) + (np.empty(metadata_bytes, dtype=_U8),)
    )
    publish_chunk = max(
        (
            min(int(value.nbytes), _PUBLISH_BUFFER_ELEMENTS * int(value.dtype.itemsize))
            for value in all_arrays
        ),
        default=0,
    )
    largest_persisted = max((int(value.nbytes) for value in all_arrays), default=0)
    route_count = largest_moments // (2 * _LE_F64.itemsize)
    validation_elements = min(route_count, _TYPED_VALIDATION_CHUNK_ELEMENTS)
    validation_scratch = validation_elements * (
        _TYPED_VALIDATION_FLOAT_BUFFERS * _LE_F64.itemsize
        + _TYPED_VALIDATION_BOOLEAN_BUFFERS * _U8.itemsize
    )
    ledger_scratch = (
        _LEDGER_UNIQUENESS_FIXED_BYTES_UPPER_BOUND
        + ledger_uniqueness_entry_count * _LEDGER_UNIQUENESS_BYTES_PER_ENTRY_UPPER_BOUND
    )
    return V1SampledExternalMemoryForecast(
        moments_numeric_payload_bytes=moments,
        fd_numeric_payload_bytes=fd,
        metadata_payload_bytes=metadata_bytes,
        persisted_payload_bytes=moments + fd + metadata_bytes,
        largest_moments_array_bytes=largest_moments,
        largest_fd_array_bytes=largest_fd,
        publication_stream_chunk_bytes_upper_bound=publish_chunk,
        publication_additional_peak_host_copy_bytes_upper_bound=(
            2 * metadata_bytes + 2 * publish_chunk
        ),
        verification_stream_buffer_bytes_upper_bound=min(
            largest_persisted,
            _VERIFY_BUFFER_BYTES,
        ),
        metadata_decode_peak_host_bytes_upper_bound=2 * metadata_bytes,
        typed_open_invariant_chunk_elements=validation_elements,
        typed_open_invariant_scratch_bytes_upper_bound=validation_scratch,
        typed_open_ledger_uniqueness_entry_count=ledger_uniqueness_entry_count,
        typed_open_ledger_uniqueness_scratch_bytes_upper_bound=ledger_scratch,
        typed_open_additional_peak_host_scratch_bytes_upper_bound=max(
            validation_scratch,
            ledger_scratch,
        ),
        reopened_numeric_heap_copy_bytes=0,
        moments_materialized_owned_bytes=moments,
        moments_materialization_peak_heap_bytes_upper_bound=(moments + largest_moments),
        fd_materialized_owned_bytes=fd,
        fd_materialization_peak_heap_bytes_upper_bound=fd + largest_fd,
        combined_materialization_peak_heap_bytes_upper_bound=(
            moments + fd + max(largest_moments, largest_fd)
        ),
    )


@dataclass(frozen=True, slots=True)
class V1SampledRegisteredLookExternalDescriptor:
    """Root-free atomic commit for one complete sampled registered look."""

    external_state: V1ExternalStateDescriptor
    moments_state_hash: str
    fd_snapshot_hash: str
    fd_design_hash: str
    proposal_generation: int
    look_index: int
    packet_boundary_index: int
    route_count: int
    route_axis_hash: str
    stratum_names: tuple[str, str]
    memory_forecast: V1SampledExternalMemoryForecast
    commit_hash: str
    corrected_evidence_stream_identity_hash: str | None = None
    literal_jacobian_stream_identity_hash: str | None = None
    dual_chart_receipt_hash: str | None = None
    same_capture_and_vjp: bool = False
    canonical_q_input_authority: bool = False
    authority_scope: str = SAMPLED_EXTERNAL_AUTHORITY_SCOPE
    proposal_discovery_input_authority: bool = True
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        if type(self.external_state) is not V1ExternalStateDescriptor:
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_TYPE,
                "external_state must have its exact descriptor type",
            )
        expected_state_schema = (
            SAMPLED_REGISTERED_LOOK_EXTERNAL_STATE_SCHEMA_V2
            if self.schema == SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA_V2
            else SAMPLED_REGISTERED_LOOK_EXTERNAL_STATE_SCHEMA
        )
        if (
            self.external_state.namespace != SAMPLED_EXTERNAL_NAMESPACE
            or self.external_state.state_schema != expected_state_schema
        ):
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                "generic external state has the wrong typed namespace/schema",
            )
        for name in (
            "moments_state_hash",
            "fd_snapshot_hash",
            "fd_design_hash",
            "route_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "proposal_generation",
            "look_index",
            "packet_boundary_index",
            "route_count",
        ):
            validator = (
                _exact_nonnegative_int
                if name == "proposal_generation"
                else _exact_positive_int
            )
            object.__setattr__(self, name, validator(getattr(self, name), name=name))
        names = _pair_of_strings(self.stratum_names, name="stratum names")
        if len(set(names)) != 2:
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                "stratum names must be unique",
            )
        object.__setattr__(self, "stratum_names", names)
        if type(self.memory_forecast) is not V1SampledExternalMemoryForecast:
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_TYPE,
                "descriptor requires its exact memory forecast",
            )
        if self.schema == SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA:
            if (
                self.corrected_evidence_stream_identity_hash is not None
                or self.literal_jacobian_stream_identity_hash is not None
                or self.dual_chart_receipt_hash is not None
                or self.same_capture_and_vjp is not False
                or self.canonical_q_input_authority is not False
            ):
                _refuse(
                    V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                    "legacy sampled external descriptors cannot claim dual-chart "
                    "or canonical-Q authority",
                )
        elif self.schema == SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA_V2:
            evidence = _identity(
                self.corrected_evidence_stream_identity_hash,
                name="corrected-evidence stream identity hash",
            )
            jacobian = _identity(
                self.literal_jacobian_stream_identity_hash,
                name="literal-Jacobian stream identity hash",
            )
            receipt = _identity(
                self.dual_chart_receipt_hash,
                name="dual-chart receipt hash",
            )
            if (
                evidence == jacobian
                or self.same_capture_and_vjp is not True
                or self.canonical_q_input_authority is not True
            ):
                _refuse(
                    V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                    "dual-chart sampled external descriptor collapsed its streams "
                    "or lost same-VJP authority",
                )
            object.__setattr__(
                self,
                "corrected_evidence_stream_identity_hash",
                evidence,
            )
            object.__setattr__(
                self,
                "literal_jacobian_stream_identity_hash",
                jacobian,
            )
            object.__setattr__(self, "dual_chart_receipt_hash", receipt)
        else:
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                "sampled external descriptor schema is invalid",
            )
        if (
            self.authority_scope != SAMPLED_EXTERNAL_AUTHORITY_SCOPE
            or self.proposal_discovery_input_authority is not True
            or self.fq_certification_authority is not False
            or self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
        ):
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                "sampled external descriptor exceeded discovery-continuation authority",
            )
        self._validate_array_manifest()
        expected = content_hash(self._commit_payload(), prefix=_COMMIT_PREFIX)
        if self.commit_hash != expected:
            _refuse(
                V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                "registered-look external commit hash does not replay",
            )

    def _validate_array_manifest(self) -> None:
        by_name = {value.logical_name: value for value in self.external_state.arrays}
        expected_names = {_METADATA_NAME, *_MOMENT_ARRAY_NAMES}
        for index in range(len(self.stratum_names)):
            expected_names.update(_fd_array_names(index))
        if set(by_name) != expected_names:
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                "typed external array names are incomplete or unexpected",
            )
        if any(identity.allow_nonfinite for identity in by_name.values()):
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                "typed sampled state cannot waive finite-array verification",
            )
        metadata = by_name[_METADATA_NAME]
        if metadata.dtype != _U8.str or len(metadata.shape) != 1:
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                "typed metadata array identity is invalid",
            )
        for name in _MOMENT_ARRAY_NAMES:
            identity = by_name[name]
            if identity.dtype != _LE_F64.str or identity.shape != (2, self.route_count):
                _refuse(
                    V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                    f"{name} identity is invalid",
                )
        for index in range(len(self.stratum_names)):
            active_name, mean_name, diagonal_name = _fd_array_names(index)
            active = by_name[active_name]
            mean = by_name[mean_name]
            diagonal = by_name[diagonal_name]
            if (
                active.dtype != _LE_F64.str
                or len(active.shape) != 2
                or active.shape[1] != self.route_count
                or mean.dtype != _LE_F64.str
                or len(mean.shape) != 2
                or mean.shape[1] != self.route_count
                or diagonal.dtype != _LE_F64.str
                or diagonal.shape != (self.route_count,)
            ):
                _refuse(
                    V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                    "FD external array identity is invalid",
                )
        observed_bytes = sum(identity.byte_count for identity in by_name.values())
        if observed_bytes != self.memory_forecast.persisted_payload_bytes:
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                "external array bytes differ from the memory forecast",
            )

    def _commit_payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "external_state_descriptor_hash": self.external_state.descriptor_hash,
            "moments_state_hash": self.moments_state_hash,
            "fd_snapshot_hash": self.fd_snapshot_hash,
            "fd_design_hash": self.fd_design_hash,
            "proposal_generation": self.proposal_generation,
            "look_index": self.look_index,
            "packet_boundary_index": self.packet_boundary_index,
            "route_count": self.route_count,
            "route_axis_hash": self.route_axis_hash,
            "stratum_names": self.stratum_names,
            "memory_forecast": self.memory_forecast.to_json_dict(),
            "authority_scope": self.authority_scope,
            "proposal_discovery_input_authority": (
                self.proposal_discovery_input_authority
            ),
            "fq_certification_authority": self.fq_certification_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }
        if self.schema == SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA_V2:
            payload.update(
                {
                    "corrected_evidence_stream_identity_hash": (
                        self.corrected_evidence_stream_identity_hash
                    ),
                    "literal_jacobian_stream_identity_hash": (
                        self.literal_jacobian_stream_identity_hash
                    ),
                    "dual_chart_receipt_hash": self.dual_chart_receipt_hash,
                    "same_capture_and_vjp": self.same_capture_and_vjp,
                    "canonical_q_input_authority": (
                        self.canonical_q_input_authority
                    ),
                }
            )
        return payload

    def to_json_dict(self) -> dict[str, object]:
        result = self._commit_payload()
        result.pop("external_state_descriptor_hash")
        result["external_state"] = self.external_state.to_json_dict()
        result["commit_hash"] = self.commit_hash
        return result

    @classmethod
    def from_json_dict(
        cls,
        value: object,
    ) -> "V1SampledRegisteredLookExternalDescriptor":
        if type(value) is not dict:
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                "typed external descriptor must be one exact object",
            )
        schema = value.get("schema")
        expected = {
            "schema",
            "external_state",
            "moments_state_hash",
            "fd_snapshot_hash",
            "fd_design_hash",
            "proposal_generation",
            "look_index",
            "packet_boundary_index",
            "route_count",
            "route_axis_hash",
            "stratum_names",
            "memory_forecast",
            "commit_hash",
            "authority_scope",
            "proposal_discovery_input_authority",
            "fq_certification_authority",
            "bce_oracle_authority",
            "mask_commit_authority",
        }
        if schema == SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA_V2:
            expected.update(
                {
                    "corrected_evidence_stream_identity_hash",
                    "literal_jacobian_stream_identity_hash",
                    "dual_chart_receipt_hash",
                    "same_capture_and_vjp",
                    "canonical_q_input_authority",
                }
            )
        payload = _exact_mapping(value, keys=expected, name="typed external descriptor")
        return cls(
            external_state=V1ExternalStateDescriptor.from_json_dict(
                payload["external_state"]
            ),
            moments_state_hash=cast(str, payload["moments_state_hash"]),
            fd_snapshot_hash=cast(str, payload["fd_snapshot_hash"]),
            fd_design_hash=cast(str, payload["fd_design_hash"]),
            proposal_generation=cast(int, payload["proposal_generation"]),
            look_index=cast(int, payload["look_index"]),
            packet_boundary_index=cast(int, payload["packet_boundary_index"]),
            route_count=cast(int, payload["route_count"]),
            route_axis_hash=cast(str, payload["route_axis_hash"]),
            stratum_names=cast(
                tuple[str, str],
                tuple(cast(Sequence[str], payload["stratum_names"])),
            ),
            memory_forecast=V1SampledExternalMemoryForecast.from_json_dict(
                payload["memory_forecast"]
            ),
            commit_hash=cast(str, payload["commit_hash"]),
            corrected_evidence_stream_identity_hash=cast(
                str | None,
                payload.get("corrected_evidence_stream_identity_hash"),
            ),
            literal_jacobian_stream_identity_hash=cast(
                str | None,
                payload.get("literal_jacobian_stream_identity_hash"),
            ),
            dual_chart_receipt_hash=cast(
                str | None,
                payload.get("dual_chart_receipt_hash"),
            ),
            same_capture_and_vjp=cast(
                bool,
                payload.get("same_capture_and_vjp", False),
            ),
            canonical_q_input_authority=cast(
                bool,
                payload.get("canonical_q_input_authority", False),
            ),
            authority_scope=cast(str, payload["authority_scope"]),
            proposal_discovery_input_authority=cast(
                bool, payload["proposal_discovery_input_authority"]
            ),
            fq_certification_authority=cast(
                bool, payload["fq_certification_authority"]
            ),
            bce_oracle_authority=cast(bool, payload["bce_oracle_authority"]),
            mask_commit_authority=cast(bool, payload["mask_commit_authority"]),
            schema=cast(str, payload["schema"]),
        )


def _proof_metadata(
    proof: BoundedOrderedFDSourceProofSnapshot | None,
) -> dict[str, object] | None:
    if proof is None:
        return None
    proof.validate_integrity()
    return {
        "schema": proof.schema,
        "entry_count": proof.entry_count,
        "source_row_chain_hash": proof.source_row_chain_hash,
        "stream_weight_chain_hash": proof.stream_weight_chain_hash,
        "peak_host_row_bytes": proof.peak_host_row_bytes,
        "total_host_transfer_bytes": proof.total_host_transfer_bytes,
        "snapshot_hash": proof.snapshot_hash,
    }


def _fd_child_metadata(
    *,
    name: str,
    snapshot: DeviceFrequentDirectionsTrustAccumulatorSnapshot,
    index: int,
) -> dict[str, object]:
    snapshot.validate_integrity()
    active_name, mean_name, diagonal_name = _fd_array_names(index)
    return {
        "stratum_name": name,
        "schema": snapshot.schema,
        "producer_algorithm_id": snapshot.producer_algorithm_id,
        "producer_implementation": snapshot.producer_implementation,
        "route_axis_count": len(snapshot.route_ids),
        "ordered_route_axis_hash": snapshot.ordered_route_axis_hash,
        "sketch_rank": snapshot.sketch_rank,
        "dtype_identity": snapshot.dtype_identity,
        "filled_count": snapshot.filled_count,
        "mean_present": snapshot.mean is not None,
        "cumulative_shrinkage": snapshot.cumulative_shrinkage,
        "total_weight": snapshot.total_weight,
        "context_count": snapshot.context_count,
        "context_ids": list(snapshot.context_ids),
        "bounded_source_proof": _proof_metadata(snapshot.bounded_source_proof),
        "latest_source_row_hashes": list(snapshot.latest_source_row_hashes),
        "snapshot_hash": snapshot.snapshot_hash,
        "active_array_name": active_name,
        "mean_array_name": mean_name,
        "diagonal_array_name": diagonal_name,
    }


def _moments_metadata(
    moments: V1StratifiedCorrectedAttributionMoments,
) -> dict[str, object]:
    return {
        "campaign_contract_hash": moments.campaign_contract_hash,
        "model_identity_hash": moments.model_identity_hash,
        "query_contract_hash": moments.query_contract_hash,
        "preprocessing_hash": moments.preprocessing_hash,
        "current_mask_hash": moments.current_mask_hash,
        "raw_logit_stream_source_hash": moments.raw_logit_stream_source_hash,
        "sampling_law_hash": moments.sampling_law_hash,
        "proposal_generation": moments.proposal_generation,
        "route_count": moments.route_count,
        "route_axis_hash": moments.route_axis_hash,
        "stratum_names": list(moments.stratum_names),
        "population_stratum_counts": list(moments.population_stratum_counts),
        "sealed_population_order_hashes": list(moments.sealed_population_order_hashes),
        "stratum_counts": list(moments.stratum_counts),
        "observed_context_ids_by_stratum": [
            list(row) for row in moments.observed_context_ids_by_stratum
        ],
        "observed_draw_ids_by_stratum": [
            list(row) for row in moments.observed_draw_ids_by_stratum
        ],
        "look_index": moments.look_index,
        "prior_state_hash": moments.prior_state_hash,
        "moment_representation": CORRECTED_ATTRIBUTION_MOMENT_REPRESENTATION,
        "schema": moments.schema,
        "state_hash": moments.state_hash,
    }


def _fd_metadata(
    snapshot: V1StratifiedFDAccumulatorSnapshot,
) -> dict[str, object]:
    return {
        "design_hash": snapshot.design_hash,
        "design_stratum_names": list(snapshot.design_stratum_names),
        "route_axis_hash": snapshot.route_axis_hash,
        "sketch_rank": snapshot.sketch_rank,
        "packet_boundary_index": snapshot.packet_boundary_index,
        "strata": [
            _fd_child_metadata(name=name, snapshot=child, index=index)
            for index, (name, child) in enumerate(snapshot.stratum_snapshots)
        ],
        "source_measurement_hashes_by_stratum": [
            [name, list(values)]
            for name, values in snapshot.source_measurement_hashes_by_stratum
        ],
        "schema": snapshot.schema,
        "snapshot_hash": snapshot.snapshot_hash,
    }


def _assert_registered_look_coherence(
    *,
    moments: V1StratifiedCorrectedAttributionMoments,
    fd_snapshot: V1StratifiedFDAccumulatorSnapshot,
    fd_design: V1StratifiedFDDesign,
    dual_chart_receipt: V1SampledDualChartRegisteredLookReceipt | None,
) -> None:
    if type(moments) is not V1StratifiedCorrectedAttributionMoments:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_TYPE,
            "moments require their exact canonical type",
        )
    if type(fd_snapshot) is not V1StratifiedFDAccumulatorSnapshot:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_TYPE,
            "FD continuation requires its exact canonical snapshot type",
        )
    if type(fd_design) is not V1StratifiedFDDesign:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_TYPE,
            "FD design requires its exact canonical type",
        )
    moments.assert_integrity()
    fd_snapshot.assert_integrity()
    fd_design.assert_integrity()
    common = (
        (moments.campaign_contract_hash, fd_design.campaign_contract_hash),
        (moments.model_identity_hash, fd_design.model_identity_hash),
        (moments.query_contract_hash, fd_design.query_contract_hash),
        (moments.preprocessing_hash, fd_design.preprocessing_hash),
        (moments.current_mask_hash, fd_design.current_mask_hash),
        (moments.current_mask_hash, fd_design.measured_mask_hash),
        (moments.sampling_law_hash, fd_design.sampling_law_hash),
        (moments.proposal_generation, fd_design.proposal_generation),
        (moments.route_count, len(fd_design.route_ids)),
        (moments.route_axis_hash, fd_design.route_axis_hash),
        (moments.stratum_names, fd_design.stratum_names),
        (fd_snapshot.design_hash, fd_design.design_hash),
        (fd_snapshot.design_stratum_names, fd_design.stratum_names),
        (fd_snapshot.route_axis_hash, fd_design.route_axis_hash),
    )
    if any(left != right for left, right in common):
        _refuse(
            V1SampledExternalStateRefusalCode.INCOHERENT_REGISTERED_LOOK,
            "moments, FD snapshot, and FD design do not bind one proposal state",
        )
    if dual_chart_receipt is None:
        if (
            moments.raw_logit_stream_source_hash
            != fd_design.raw_logit_stream_source_hash
        ):
            _refuse(
                V1SampledExternalStateRefusalCode.INCOHERENT_REGISTERED_LOOK,
                "legacy sampled external state requires one historical shared "
                "stream identity",
            )
    else:
        if type(dual_chart_receipt) is not V1SampledDualChartRegisteredLookReceipt:
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_TYPE,
                "dual-chart publication requires its exact registered-look receipt",
            )
        try:
            dual_chart_receipt.receipt_hash
        except (TypeError, ValueError) as error:
            _refuse(
                V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                f"dual-chart receipt is invalid: {error}",
            )
        if (
            moments.raw_logit_stream_source_hash
            != dual_chart_receipt.corrected_evidence_stream_identity_hash
            or fd_design.raw_logit_stream_source_hash
            != dual_chart_receipt.literal_jacobian_stream_identity_hash
            or moments.raw_logit_stream_source_hash
            == fd_design.raw_logit_stream_source_hash
            or dual_chart_receipt.same_capture_and_vjp is not True
            or dual_chart_receipt.canonical_q_input_authority is not True
        ):
            _refuse(
                V1SampledExternalStateRefusalCode.INCOHERENT_REGISTERED_LOOK,
                "dual-chart receipt does not bind corrected moments and literal-J "
                "FD state",
            )
    if fd_snapshot.sketch_rank <= 0 or fd_snapshot.packet_boundary_index <= 0:
        _refuse(
            V1SampledExternalStateRefusalCode.INCOHERENT_REGISTERED_LOOK,
            "registered-look FD state must be at a nonempty packet boundary",
        )
    if any(count <= 0 for count in moments.stratum_counts):
        _refuse(
            V1SampledExternalStateRefusalCode.INCOHERENT_REGISTERED_LOOK,
            "a complete registered look must observe both strata",
        )
    children = dict(fd_snapshot.stratum_snapshots)
    for index, name in enumerate(moments.stratum_names):
        child = children[name]
        expected_contexts = moments.observed_context_ids_by_stratum[index]
        if (
            child.context_ids != expected_contexts
            or child.context_count != moments.stratum_counts[index]
            or len(fd_snapshot.source_measurement_hashes_by_stratum[index][1])
            != moments.stratum_counts[index]
        ):
            _refuse(
                V1SampledExternalStateRefusalCode.INCOHERENT_REGISTERED_LOOK,
                "moments and FD state do not identify the same sampled prefixes",
            )


def _array_identity_by_name(
    descriptor: V1ExternalStateDescriptor,
) -> dict[str, V1ExternalArrayIdentity]:
    return {value.logical_name: value for value in descriptor.arrays}


def _array_hash_token(identity: V1ExternalArrayIdentity) -> dict[str, object]:
    """Replay ``content_hash``'s ndarray token without hashing a mapped file twice."""

    return {
        "dtype": str(np.dtype(identity.dtype)),
        "shape": list(identity.shape),
        "sha256": identity.sha256,
    }


def _moments_constructor_metadata(value: object) -> dict[str, object]:
    keys = {
        "campaign_contract_hash",
        "model_identity_hash",
        "query_contract_hash",
        "preprocessing_hash",
        "current_mask_hash",
        "raw_logit_stream_source_hash",
        "sampling_law_hash",
        "proposal_generation",
        "route_count",
        "route_axis_hash",
        "stratum_names",
        "population_stratum_counts",
        "sealed_population_order_hashes",
        "stratum_counts",
        "observed_context_ids_by_stratum",
        "observed_draw_ids_by_stratum",
        "look_index",
        "prior_state_hash",
        "moment_representation",
        "schema",
        "state_hash",
    }
    return _exact_mapping(value, keys=keys, name="moments metadata")


def _proof_from_metadata(value: object) -> BoundedOrderedFDSourceProofSnapshot | None:
    if value is None:
        return None
    keys = {
        "schema",
        "entry_count",
        "source_row_chain_hash",
        "stream_weight_chain_hash",
        "peak_host_row_bytes",
        "total_host_transfer_bytes",
        "snapshot_hash",
    }
    payload = _exact_mapping(value, keys=keys, name="bounded FD proof metadata")
    try:
        proof = BoundedOrderedFDSourceProofSnapshot(**payload)  # type: ignore[arg-type]
        proof.validate_integrity()
    except (TypeError, ValueError) as error:
        _refuse(
            V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
            f"bounded FD proof metadata is invalid: {error}",
        )
    return proof


def _fd_child_metadata_checked(value: object, *, index: int) -> dict[str, object]:
    keys = {
        "stratum_name",
        "schema",
        "producer_algorithm_id",
        "producer_implementation",
        "route_axis_count",
        "ordered_route_axis_hash",
        "sketch_rank",
        "dtype_identity",
        "filled_count",
        "mean_present",
        "cumulative_shrinkage",
        "total_weight",
        "context_count",
        "context_ids",
        "bounded_source_proof",
        "latest_source_row_hashes",
        "snapshot_hash",
        "active_array_name",
        "mean_array_name",
        "diagonal_array_name",
    }
    payload = _exact_mapping(value, keys=keys, name="FD stratum metadata")
    expected_names = _fd_array_names(index)
    if (
        payload["active_array_name"],
        payload["mean_array_name"],
        payload["diagonal_array_name"],
    ) != expected_names:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            "FD metadata array names are not canonical",
        )
    if type(payload["mean_present"]) is not bool:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            "FD mean-presence marker must be an exact boolean",
        )
    for name in ("cumulative_shrinkage", "total_weight"):
        scalar = payload[name]
        if type(scalar) not in {float, int} or not math.isfinite(float(scalar)):
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                f"{name} must be finite",
            )
    _proof_from_metadata(payload["bounded_source_proof"])
    return payload


def _fd_metadata_checked(value: object) -> dict[str, object]:
    keys = {
        "design_hash",
        "design_stratum_names",
        "route_axis_hash",
        "sketch_rank",
        "packet_boundary_index",
        "strata",
        "source_measurement_hashes_by_stratum",
        "schema",
        "snapshot_hash",
    }
    return _exact_mapping(value, keys=keys, name="stratified FD metadata")


def _assert_globally_unique_ledgers(
    ledgers: tuple[tuple[str, ...], tuple[str, ...]],
    *,
    name: str,
) -> None:
    """Check cross-stratum uniqueness with metadata-scale, not route-scale, state."""

    seen: set[str] = set()
    for ledger in ledgers:
        for identity in ledger:
            if identity in seen:
                _refuse(
                    V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                    f"{name} cannot repeat across or within strata",
                )
            seen.add(identity)


def _validate_mapped_moment_invariants(
    *,
    view: V1ExternalStateView,
    counts: tuple[int, int],
    route_count: int,
    expected_scratch_bytes: int,
) -> None:
    """Validate canonical moment semantics using bounded reusable scratch.

    Two float64 buffers and one boolean buffer are allocated at the registered
    chunk size.  No operation allocates a route-sized result: every NumPy ufunc
    writes into one of those buffers.  This pass happens before a typed view is
    returned to the caller.
    """

    arrays = {
        name: view.arrays[name]
        for name in (
            "moments_signed_mean",
            "moments_signed_m2",
            "moments_absolute_mean",
            "moments_absolute_m2",
        )
    }
    if any(
        value.dtype != _LE_F64
        or value.shape != (2, route_count)
        or value.flags.writeable
        for value in arrays.values()
    ):
        _refuse(
            V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
            "mapped moments do not preserve the canonical read-only route axis",
        )

    chunk_elements = min(route_count, _TYPED_VALIDATION_CHUNK_ELEMENTS)
    scratch_bytes = chunk_elements * (
        _TYPED_VALIDATION_FLOAT_BUFFERS * _LE_F64.itemsize
        + _TYPED_VALIDATION_BOOLEAN_BUFFERS * _U8.itemsize
    )
    if expected_scratch_bytes != scratch_bytes:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            "typed-open invariant scratch does not replay from the route axis",
        )
    left = np.empty(chunk_elements, dtype=_LE_F64)
    right = np.empty(chunk_elements, dtype=_LE_F64)
    boolean = np.empty(chunk_elements, dtype=np.bool_)
    jensen_scale = 128.0 * np.finfo(np.float64).eps

    for stratum, count in enumerate(counts):
        for start in range(0, route_count, chunk_elements):
            stop = min(route_count, start + chunk_elements)
            length = stop - start
            work_bool = boolean[:length]
            signed_mean = arrays["moments_signed_mean"][stratum, start:stop]
            signed_m2 = arrays["moments_signed_m2"][stratum, start:stop]
            absolute_mean = arrays["moments_absolute_mean"][stratum, start:stop]
            absolute_m2 = arrays["moments_absolute_m2"][stratum, start:stop]
            chunk_arrays = (signed_mean, signed_m2, absolute_mean, absolute_m2)

            for value in chunk_arrays:
                np.isfinite(value, out=work_bool)
                if not bool(np.all(work_bool)):
                    _refuse(
                        V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                        "mapped moments must be finite",
                    )
                np.equal(
                    value.view(_LE_U64),
                    _NEGATIVE_F64_ZERO_BITS,
                    out=work_bool,
                )
                if bool(np.any(work_bool)):
                    _refuse(
                        V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                        "mapped moments must use canonical positive zero",
                    )
            for value in (signed_m2, absolute_mean, absolute_m2):
                np.less(value, 0.0, out=work_bool)
                if bool(np.any(work_bool)):
                    _refuse(
                        V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                        "M2 and absolute moments must be nonnegative",
                    )
            if count == 0:
                for value in chunk_arrays:
                    np.not_equal(value, 0.0, out=work_bool)
                    if bool(np.any(work_bool)):
                        _refuse(
                            V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                            "an empty stratum must have zero central moments",
                        )
            if count < 2:
                for value in (signed_m2, absolute_m2):
                    np.not_equal(value, 0.0, out=work_bool)
                    if bool(np.any(work_bool)):
                        _refuse(
                            V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                            "M2 must vanish before a stratum has two rows",
                        )

            work_left = left[:length]
            work_right = right[:length]
            np.absolute(signed_mean, out=work_left)
            np.maximum(1.0, absolute_mean, out=work_right)
            np.multiply(work_right, jensen_scale, out=work_right)
            np.add(absolute_mean, work_right, out=work_right)
            np.greater(work_left, work_right, out=work_bool)
            if bool(np.any(work_bool)):
                _refuse(
                    V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                    "signed and absolute means violate Jensen's bound",
                )


def _validate_typed_open(
    *,
    descriptor: V1SampledRegisteredLookExternalDescriptor,
    view: V1ExternalStateView,
    metadata: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], tuple[dict[str, object], ...]]:
    dual_chart = (
        descriptor.schema
        == SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA_V2
    )
    if dual_chart:
        if metadata.get("schema") != SAMPLED_REGISTERED_LOOK_METADATA_SCHEMA_V2:
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
                "dual-chart descriptor requires dual-chart metadata",
            )
        dual = _exact_mapping(
            metadata.get("dual_chart"),
            keys={
                "corrected_evidence_stream_identity_hash",
                "literal_jacobian_stream_identity_hash",
                "dual_chart_receipt_hash",
                "same_capture_and_vjp",
                "corrected_rows_excluded_from_response_geometry",
                "canonical_q_input_authority",
                "fq_certification_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
            },
            name="dual-chart metadata",
        )
        if (
            dual["corrected_evidence_stream_identity_hash"]
            != descriptor.corrected_evidence_stream_identity_hash
            or dual["literal_jacobian_stream_identity_hash"]
            != descriptor.literal_jacobian_stream_identity_hash
            or dual["dual_chart_receipt_hash"]
            != descriptor.dual_chart_receipt_hash
            or dual["same_capture_and_vjp"] is not True
            or dual["corrected_rows_excluded_from_response_geometry"] is not True
            or dual["canonical_q_input_authority"] is not True
            or dual["fq_certification_authority"] is not False
            or dual["bce_oracle_authority"] is not False
            or dual["mask_commit_authority"] is not False
        ):
            _refuse(
                V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                "dual-chart metadata does not replay its authority receipt",
            )
    elif metadata.get("schema") != SAMPLED_REGISTERED_LOOK_METADATA_SCHEMA:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            "legacy descriptor cannot open dual-chart metadata",
        )
    identities = _array_identity_by_name(descriptor.external_state)
    moments = _moments_constructor_metadata(metadata["moments"])
    fd = _fd_metadata_checked(metadata["fd"])
    raw_strata = fd["strata"]
    if type(raw_strata) is not list or len(raw_strata) != len(descriptor.stratum_names):
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            "FD stratum metadata is incomplete",
        )
    strata = tuple(
        _fd_child_metadata_checked(item, index=index)
        for index, item in enumerate(cast(list[object], raw_strata))
    )

    for name in (
        "campaign_contract_hash",
        "model_identity_hash",
        "query_contract_hash",
        "preprocessing_hash",
        "current_mask_hash",
        "raw_logit_stream_source_hash",
        "sampling_law_hash",
    ):
        _identity(moments[name], name=name)
    proposal_generation = _exact_nonnegative_int(
        moments["proposal_generation"],
        name="proposal generation",
    )
    route_count = _exact_positive_int(moments["route_count"], name="route count")
    route_axis_hash = _identity(moments["route_axis_hash"], name="route axis hash")
    moment_names = _pair_of_strings(moments["stratum_names"], name="moments strata")
    if len(set(moment_names)) != 2:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            "moment stratum names must be distinct",
        )
    population_counts = _pair_of_positive_ints(
        moments["population_stratum_counts"],
        name="population stratum counts",
    )
    _pair_of_strings(
        moments["sealed_population_order_hashes"],
        name="sealed population order hashes",
    )
    counts = _pair_of_ints(moments["stratum_counts"], name="stratum counts")
    contexts = _nested_pair(
        moments["observed_context_ids_by_stratum"],
        name="observed context ledgers",
    )
    draws = _nested_pair(
        moments["observed_draw_ids_by_stratum"],
        name="observed draw ledgers",
    )
    look_index = _exact_positive_int(moments["look_index"], name="look index")
    prior_state_hash = moments["prior_state_hash"]
    if look_index == 1:
        if prior_state_hash is not None:
            _refuse(
                V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                "the first moments look cannot name a prior state",
            )
    else:
        _identity(prior_state_hash, name="prior state hash")
    state_hash = _identity(moments["state_hash"], name="moments state hash")
    if (
        moments["schema"] != STRATIFIED_CORRECTED_ATTRIBUTION_MOMENTS_SCHEMA
        or moments["moment_representation"]
        != CORRECTED_ATTRIBUTION_MOMENT_REPRESENTATION
        or state_hash != descriptor.moments_state_hash
        or proposal_generation != descriptor.proposal_generation
        or look_index != descriptor.look_index
        or route_count != descriptor.route_count
        or route_axis_hash != descriptor.route_axis_hash
        or moment_names != descriptor.stratum_names
        or (
            dual_chart
            and moments["raw_logit_stream_source_hash"]
            != descriptor.corrected_evidence_stream_identity_hash
        )
        or any(
            count > population for count, population in zip(counts, population_counts)
        )
        or any(
            len(contexts[i]) != counts[i] or len(draws[i]) != counts[i] for i in (0, 1)
        )
        or descriptor.memory_forecast.typed_open_ledger_uniqueness_entry_count
        != sum(counts)
    ):
        _refuse(
            V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
            "moments metadata differs from the registered-look descriptor",
        )
    _assert_globally_unique_ledgers(contexts, name="observed context ledgers")
    _assert_globally_unique_ledgers(draws, name="observed draw ledgers")
    _validate_mapped_moment_invariants(
        view=view,
        counts=counts,
        route_count=route_count,
        expected_scratch_bytes=(
            descriptor.memory_forecast.typed_open_invariant_scratch_bytes_upper_bound
        ),
    )
    if any(count <= 0 for count in counts):
        _refuse(
            V1SampledExternalStateRefusalCode.INCOHERENT_REGISTERED_LOOK,
            "a complete registered look must observe both strata",
        )

    moments_payload = {
        key: moments[key]
        for key in (
            "schema",
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "proposal_generation",
            "route_count",
            "route_axis_hash",
            "stratum_names",
            "population_stratum_counts",
            "sealed_population_order_hashes",
            "stratum_counts",
            "observed_context_ids_by_stratum",
            "observed_draw_ids_by_stratum",
        )
    }
    moments_payload.update(
        {
            "moment_representation": moments["moment_representation"],
            "signed_mean_by_stratum": _array_hash_token(
                identities["moments_signed_mean"]
            ),
            "signed_m2_by_stratum": _array_hash_token(identities["moments_signed_m2"]),
            "absolute_mean_by_stratum": _array_hash_token(
                identities["moments_absolute_mean"]
            ),
            "absolute_m2_by_stratum": _array_hash_token(
                identities["moments_absolute_m2"]
            ),
            "look_index": moments["look_index"],
            "prior_state_hash": moments["prior_state_hash"],
        }
    )
    observed_moments_hash = content_hash(moments_payload, prefix=_MOMENTS_PREFIX)
    if observed_moments_hash != descriptor.moments_state_hash:
        _refuse(
            V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
            "external arrays do not replay the canonical moments state hash",
        )

    fd_names = _string_tuple(fd["design_stratum_names"], name="FD stratum names")
    source_pairs_raw = fd["source_measurement_hashes_by_stratum"]
    if type(source_pairs_raw) is not list:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            "FD source ledgers must be an ordered list",
        )
    source_pairs: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
        (
            _identity(cast(Sequence[object], pair)[0], name="source-ledger stratum"),
            _string_tuple(cast(Sequence[object], pair)[1], name="source ledger"),
        )
        for pair in cast(list[object], source_pairs_raw)
        if type(pair) in {list, tuple} and len(cast(Sequence[object], pair)) == 2
    )
    if len(source_pairs) != len(cast(list[object], source_pairs_raw)):
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_SCHEMA,
            "FD source-ledger entries are malformed",
        )
    if (
        fd["schema"] != STRATIFIED_FD_ACCUMULATOR_SNAPSHOT_SCHEMA
        or fd["snapshot_hash"] != descriptor.fd_snapshot_hash
        or fd["design_hash"] != descriptor.fd_design_hash
        or fd["route_axis_hash"] != descriptor.route_axis_hash
        or fd["packet_boundary_index"] != descriptor.packet_boundary_index
        or fd_names != descriptor.stratum_names
        or tuple(name for name, _values in source_pairs) != fd_names
    ):
        _refuse(
            V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
            "FD metadata differs from the registered-look descriptor",
        )

    child_hashes: list[tuple[str, str]] = []
    for index, child in enumerate(strata):
        active_name, mean_name, diagonal_name = _fd_array_names(index)
        active_identity = identities[active_name]
        mean_identity = identities[mean_name]
        diagonal_identity = identities[diagonal_name]
        proof = _proof_from_metadata(child["bounded_source_proof"])
        child_name = _identity(child["stratum_name"], name="FD child stratum")
        child_contexts = _string_tuple(child["context_ids"], name="FD child contexts")
        latest = _string_tuple(
            child["latest_source_row_hashes"],
            name="FD latest source rows",
        )
        filled = _exact_nonnegative_int(child["filled_count"], name="FD filled count")
        context_count = _exact_nonnegative_int(
            child["context_count"],
            name="FD context count",
        )
        route_count = _exact_positive_int(
            child["route_axis_count"],
            name="FD route count",
        )
        rank = _exact_positive_int(child["sketch_rank"], name="FD sketch rank")
        mean_present = cast(bool, child["mean_present"])
        if (
            child_name != descriptor.stratum_names[index]
            or route_count != descriptor.route_count
            or child["ordered_route_axis_hash"] != descriptor.route_axis_hash
            or rank != fd["sketch_rank"]
            or active_identity.shape != (filled, descriptor.route_count)
            or diagonal_identity.shape != (descriptor.route_count,)
            or (
                mean_present
                and (len(mean_identity.shape) != 2 or mean_identity.shape[0] <= 0)
            )
            or (not mean_present and mean_identity.shape != (0, descriptor.route_count))
            or child_contexts != contexts[index]
            or context_count != counts[index]
            or len(source_pairs[index][1]) != counts[index]
        ):
            _refuse(
                V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                "FD stratum state does not match the registered sampled prefix",
            )
        child_payload = {
            "schema": child["schema"],
            "producer_algorithm_id": child["producer_algorithm_id"],
            "producer_implementation": child["producer_implementation"],
            "route_axis_count": route_count,
            "ordered_route_axis_hash": child["ordered_route_axis_hash"],
            "sketch_rank": rank,
            "dtype_identity": child["dtype_identity"],
            "active_buffer_rows": _array_hash_token(active_identity),
            "filled_count": filled,
            "mean": _array_hash_token(mean_identity) if mean_present else None,
            "weighted_second_moment_diagonal": _array_hash_token(diagonal_identity),
            "cumulative_shrinkage": child["cumulative_shrinkage"],
            "total_weight": child["total_weight"],
            "context_count": context_count,
            "context_ids": child_contexts,
            "bounded_source_proof_snapshot_hash": (
                None if proof is None else proof.snapshot_hash
            ),
            "latest_source_row_hashes": latest,
        }
        observed_child = content_hash(child_payload, prefix=_DEVICE_FD_PREFIX)
        if observed_child != child["snapshot_hash"]:
            _refuse(
                V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                "external arrays do not replay a canonical FD child hash",
            )
        child_hashes.append((child_name, observed_child))

    outer_payload = {
        "schema": fd["schema"],
        "design_hash": fd["design_hash"],
        "design_stratum_names": fd_names,
        "route_axis_hash": fd["route_axis_hash"],
        "sketch_rank": fd["sketch_rank"],
        "packet_boundary_index": fd["packet_boundary_index"],
        "stratum_snapshot_hashes": tuple(child_hashes),
        "source_measurement_hashes_by_stratum": source_pairs,
    }
    observed_fd_hash = content_hash(outer_payload, prefix=_STRATIFIED_FD_PREFIX)
    if observed_fd_hash != descriptor.fd_snapshot_hash:
        _refuse(
            V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
            "external arrays do not replay the stratified FD snapshot hash",
        )
    return moments, fd, strata


_SAMPLED_LIVE_ATTACHMENT_CONSTRUCTION_TOKEN: Final = object()


class V1SampledRegisteredLookLiveAttachment:
    """Nonserializable typed lease over one exactly admitted sampled state."""

    __slots__ = ("_descriptor", "_external_attachment", "_closed")

    def __init__(
        self,
        *,
        descriptor: V1SampledRegisteredLookExternalDescriptor,
        external_attachment: V1ExternalStateLiveAttachment,
        token: object,
    ) -> None:
        if token is not _SAMPLED_LIVE_ATTACHMENT_CONSTRUCTION_TOKEN:
            raise TypeError("sampled live attachments are runtime-issued only")
        if (
            type(descriptor) is not V1SampledRegisteredLookExternalDescriptor
            or type(external_attachment) is not V1ExternalStateLiveAttachment
            or external_attachment.descriptor_hash
            != descriptor.external_state.descriptor_hash
        ):
            raise TypeError("sampled live attachment owners are not exact")
        self._descriptor = descriptor
        self._external_attachment = external_attachment
        self._closed = False

    @property
    def commit_hash(self) -> str:
        return self._descriptor.commit_hash

    @property
    def pid(self) -> int:
        return self._external_attachment.pid

    @property
    def file_count(self) -> int:
        return self._external_attachment.file_count

    @property
    def closed(self) -> bool:
        return self._closed or self._external_attachment.closed

    def matches(self, descriptor: V1SampledRegisteredLookExternalDescriptor) -> bool:
        return bool(
            not self.closed
            and type(descriptor) is V1SampledRegisteredLookExternalDescriptor
            and descriptor == self._descriptor
        )

    def verify(self, descriptor: V1SampledRegisteredLookExternalDescriptor) -> None:
        if self._closed:
            _refuse(
                V1SampledExternalStateRefusalCode.CLOSED_VIEW,
                "sampled live attachment is closed",
            )
        if type(descriptor) is not V1SampledRegisteredLookExternalDescriptor:
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_TYPE,
                "sampled live attachment requires its exact descriptor type",
            )
        strict = V1SampledRegisteredLookExternalDescriptor.from_json_dict(
            descriptor.to_json_dict()
        )
        if strict != self._descriptor:
            _refuse(
                V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                "sampled live attachment names another typed commit",
            )
        self._external_attachment.verify(strict.external_state)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._external_attachment.close()

    def __enter__(self) -> "V1SampledRegisteredLookLiveAttachment":
        if self.closed:
            _refuse(
                V1SampledExternalStateRefusalCode.CLOSED_VIEW,
                "sampled live attachment is closed",
            )
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("sampled live attachments cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("sampled live attachments cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("sampled live attachments cannot be copied")

    def __del__(self) -> None:  # pragma: no cover - defensive interpreter cleanup.
        try:
            self.close()
        except BaseException:
            pass


def _sampled_live_attachment(
    descriptor: V1SampledRegisteredLookExternalDescriptor,
    external_attachment: V1ExternalStateLiveAttachment,
) -> V1SampledRegisteredLookLiveAttachment:
    return V1SampledRegisteredLookLiveAttachment(
        descriptor=descriptor,
        external_attachment=external_attachment,
        token=_SAMPLED_LIVE_ATTACHMENT_CONSTRUCTION_TOKEN,
    )


def _publish_sampled_registered_look_state(
    root: str | os.PathLike[str],
    *,
    moments: V1StratifiedCorrectedAttributionMoments,
    fd_snapshot: V1StratifiedFDAccumulatorSnapshot,
    fd_design: V1StratifiedFDDesign,
    dual_chart_receipt: V1SampledDualChartRegisteredLookReceipt | None,
    issue_live_attachment: bool,
) -> tuple[
    V1SampledRegisteredLookExternalDescriptor,
    V1SampledRegisteredLookLiveAttachment | None,
]:
    """Atomically publish one coherent complete registered-look continuation."""

    _assert_registered_look_coherence(
        moments=moments,
        fd_snapshot=fd_snapshot,
        fd_design=fd_design,
        dual_chart_receipt=dual_chart_receipt,
    )
    dual_chart = dual_chart_receipt is not None
    metadata = {
        "schema": (
            SAMPLED_REGISTERED_LOOK_METADATA_SCHEMA_V2
            if dual_chart
            else SAMPLED_REGISTERED_LOOK_METADATA_SCHEMA
        ),
        "moments": _moments_metadata(moments),
        "fd": _fd_metadata(fd_snapshot),
    }
    if dual_chart:
        assert dual_chart_receipt is not None
        metadata["dual_chart"] = {
            "corrected_evidence_stream_identity_hash": (
                dual_chart_receipt.corrected_evidence_stream_identity_hash
            ),
            "literal_jacobian_stream_identity_hash": (
                dual_chart_receipt.literal_jacobian_stream_identity_hash
            ),
            "dual_chart_receipt_hash": dual_chart_receipt.receipt_hash,
            "same_capture_and_vjp": True,
            "corrected_rows_excluded_from_response_geometry": True,
            "canonical_q_input_authority": True,
            "fq_certification_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
        }
    metadata_bytes = _canonical_json_bytes(metadata)
    metadata_array = np.frombuffer(metadata_bytes, dtype=_U8)

    arrays: dict[str, np.ndarray] = {
        _METADATA_NAME: metadata_array,
        "moments_signed_mean": moments.signed_mean_by_stratum,
        "moments_signed_m2": moments.signed_m2_by_stratum,
        "moments_absolute_mean": moments.absolute_mean_by_stratum,
        "moments_absolute_m2": moments.absolute_m2_by_stratum,
    }
    fd_arrays: list[np.ndarray] = []
    for index, (_name, child) in enumerate(fd_snapshot.stratum_snapshots):
        active_name, mean_name, diagonal_name = _fd_array_names(index)
        empty_mean = np.empty((0, moments.route_count), dtype=_LE_F64)
        active = np.asarray(child.active_buffer_rows)
        mean = empty_mean if child.mean is None else np.asarray(child.mean)
        diagonal = np.asarray(child.weighted_second_moment_diagonal)
        arrays[active_name] = active
        arrays[mean_name] = mean
        arrays[diagonal_name] = diagonal
        fd_arrays.extend((active, mean, diagonal))
    moment_arrays = [arrays[name] for name in _MOMENT_ARRAY_NAMES]
    memory = _forecast_memory(
        moment_arrays=moment_arrays,
        fd_arrays=fd_arrays,
        metadata_bytes=len(metadata_bytes),
        # Context and draw ledgers are checked sequentially, so only one
        # prefix-sized uniqueness set is resident at a time.
        ledger_uniqueness_entry_count=sum(moments.stratum_counts),
    )
    policies = tuple(
        V1ExternalArrayPolicy(
            logical_name=name,
            expected_dtype=value.dtype.str,
            expected_shape=tuple(int(item) for item in value.shape),
            allow_nonfinite=False,
        )
        for name, value in arrays.items()
    )
    external_attachment: V1ExternalStateLiveAttachment | None = None
    if issue_live_attachment:
        external, external_attachment = publish_external_state_with_attachment(
            root,
            namespace=SAMPLED_EXTERNAL_NAMESPACE,
            state_schema=(
                SAMPLED_REGISTERED_LOOK_EXTERNAL_STATE_SCHEMA_V2
                if dual_chart
                else SAMPLED_REGISTERED_LOOK_EXTERNAL_STATE_SCHEMA
            ),
            arrays=arrays,
            policies=policies,
        )
    else:
        external = publish_external_state(
            root,
            namespace=SAMPLED_EXTERNAL_NAMESPACE,
            state_schema=(
                SAMPLED_REGISTERED_LOOK_EXTERNAL_STATE_SCHEMA_V2
                if dual_chart
                else SAMPLED_REGISTERED_LOOK_EXTERNAL_STATE_SCHEMA
            ),
            arrays=arrays,
            policies=policies,
        )
    try:
        descriptor_schema = (
            SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA_V2
            if dual_chart
            else SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA
        )
        provisional = {
            "schema": descriptor_schema,
            "external_state_descriptor_hash": external.descriptor_hash,
            "moments_state_hash": moments.state_hash,
            "fd_snapshot_hash": fd_snapshot.snapshot_hash,
            "fd_design_hash": fd_design.design_hash,
            "proposal_generation": moments.proposal_generation,
            "look_index": moments.look_index,
            "packet_boundary_index": fd_snapshot.packet_boundary_index,
            "route_count": moments.route_count,
            "route_axis_hash": moments.route_axis_hash,
            "stratum_names": moments.stratum_names,
            "memory_forecast": memory.to_json_dict(),
            "authority_scope": SAMPLED_EXTERNAL_AUTHORITY_SCOPE,
            "proposal_discovery_input_authority": True,
            "fq_certification_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
        }
        if dual_chart:
            assert dual_chart_receipt is not None
            provisional.update(
                {
                    "corrected_evidence_stream_identity_hash": (
                        dual_chart_receipt.corrected_evidence_stream_identity_hash
                    ),
                    "literal_jacobian_stream_identity_hash": (
                        dual_chart_receipt.literal_jacobian_stream_identity_hash
                    ),
                    "dual_chart_receipt_hash": dual_chart_receipt.receipt_hash,
                    "same_capture_and_vjp": True,
                    "canonical_q_input_authority": True,
                }
            )
        commit_hash = content_hash(provisional, prefix=_COMMIT_PREFIX)
        descriptor = V1SampledRegisteredLookExternalDescriptor(
            external_state=external,
            moments_state_hash=moments.state_hash,
            fd_snapshot_hash=fd_snapshot.snapshot_hash,
            fd_design_hash=fd_design.design_hash,
            proposal_generation=moments.proposal_generation,
            look_index=moments.look_index,
            packet_boundary_index=fd_snapshot.packet_boundary_index,
            route_count=moments.route_count,
            route_axis_hash=moments.route_axis_hash,
            stratum_names=moments.stratum_names,
            memory_forecast=memory,
            commit_hash=commit_hash,
            corrected_evidence_stream_identity_hash=(
                None
                if dual_chart_receipt is None
                else dual_chart_receipt.corrected_evidence_stream_identity_hash
            ),
            literal_jacobian_stream_identity_hash=(
                None
                if dual_chart_receipt is None
                else dual_chart_receipt.literal_jacobian_stream_identity_hash
            ),
            dual_chart_receipt_hash=(
                None
                if dual_chart_receipt is None
                else dual_chart_receipt.receipt_hash
            ),
            same_capture_and_vjp=dual_chart,
            canonical_q_input_authority=dual_chart,
            schema=descriptor_schema,
        )
        attachment = (
            None
            if external_attachment is None
            else _sampled_live_attachment(descriptor, external_attachment)
        )
        external_attachment = None
        return descriptor, attachment
    finally:
        if external_attachment is not None:
            external_attachment.close()


def publish_sampled_registered_look_state(
    root: str | os.PathLike[str],
    *,
    moments: V1StratifiedCorrectedAttributionMoments,
    fd_snapshot: V1StratifiedFDAccumulatorSnapshot,
    fd_design: V1StratifiedFDDesign,
    dual_chart_receipt: V1SampledDualChartRegisteredLookReceipt | None = None,
) -> V1SampledRegisteredLookExternalDescriptor:
    descriptor, attachment = _publish_sampled_registered_look_state(
        root,
        moments=moments,
        fd_snapshot=fd_snapshot,
        fd_design=fd_design,
        dual_chart_receipt=dual_chart_receipt,
        issue_live_attachment=False,
    )
    assert attachment is None
    return descriptor


def publish_sampled_registered_look_state_with_attachment(
    root: str | os.PathLike[str],
    *,
    moments: V1StratifiedCorrectedAttributionMoments,
    fd_snapshot: V1StratifiedFDAccumulatorSnapshot,
    fd_design: V1StratifiedFDDesign,
    dual_chart_receipt: V1SampledDualChartRegisteredLookReceipt | None = None,
) -> tuple[
    V1SampledRegisteredLookExternalDescriptor,
    V1SampledRegisteredLookLiveAttachment,
]:
    descriptor, attachment = _publish_sampled_registered_look_state(
        root,
        moments=moments,
        fd_snapshot=fd_snapshot,
        fd_design=fd_design,
        dual_chart_receipt=dual_chart_receipt,
        issue_live_attachment=True,
    )
    assert attachment is not None
    return descriptor, attachment


@dataclass(slots=True)
class V1SampledRegisteredLookExternalView:
    """Verified typed mappings plus explicit legacy materialization methods."""

    descriptor: V1SampledRegisteredLookExternalDescriptor
    external_view: V1ExternalStateView
    _moments_metadata: dict[str, object] = field(repr=False)
    _fd_metadata: dict[str, object] = field(repr=False)
    _fd_strata_metadata: tuple[dict[str, object], ...] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def _assert_open(self) -> None:
        if self._closed:
            _refuse(
                V1SampledExternalStateRefusalCode.CLOSED_VIEW,
                "sampled external view is closed",
            )

    @property
    def arrays(self) -> Mapping[str, np.ndarray]:
        self._assert_open()
        return self.external_view.arrays

    @property
    def moment_arrays(self) -> Mapping[str, np.ndarray]:
        self._assert_open()
        return MappingProxyType(
            {name: self.external_view.arrays[name] for name in _MOMENT_ARRAY_NAMES}
        )

    @property
    def fd_arrays(self) -> Mapping[str, np.ndarray]:
        self._assert_open()
        return MappingProxyType(
            {
                name: value
                for name, value in self.external_view.arrays.items()
                if name.startswith("fd_")
            }
        )

    def materialize_moments(self) -> V1StratifiedCorrectedAttributionMoments:
        """Build the canonical in-memory object, incurring the receipted copy."""

        self._assert_open()
        metadata = self._moments_metadata
        try:
            result = V1StratifiedCorrectedAttributionMoments(
                campaign_contract_hash=cast(str, metadata["campaign_contract_hash"]),
                model_identity_hash=cast(str, metadata["model_identity_hash"]),
                query_contract_hash=cast(str, metadata["query_contract_hash"]),
                preprocessing_hash=cast(str, metadata["preprocessing_hash"]),
                current_mask_hash=cast(str, metadata["current_mask_hash"]),
                raw_logit_stream_source_hash=cast(
                    str, metadata["raw_logit_stream_source_hash"]
                ),
                sampling_law_hash=cast(str, metadata["sampling_law_hash"]),
                proposal_generation=cast(int, metadata["proposal_generation"]),
                route_count=cast(int, metadata["route_count"]),
                route_axis_hash=cast(str, metadata["route_axis_hash"]),
                stratum_names=_pair_of_strings(
                    metadata["stratum_names"], name="moments stratum names"
                ),
                population_stratum_counts=_pair_of_ints(
                    metadata["population_stratum_counts"],
                    name="population counts",
                ),
                sealed_population_order_hashes=_pair_of_strings(
                    metadata["sealed_population_order_hashes"],
                    name="sealed order hashes",
                ),
                stratum_counts=_pair_of_ints(
                    metadata["stratum_counts"], name="stratum counts"
                ),
                observed_context_ids_by_stratum=_nested_pair(
                    metadata["observed_context_ids_by_stratum"],
                    name="observed contexts",
                ),
                observed_draw_ids_by_stratum=_nested_pair(
                    metadata["observed_draw_ids_by_stratum"],
                    name="observed draws",
                ),
                signed_mean_by_stratum=self.external_view.arrays["moments_signed_mean"],
                signed_m2_by_stratum=self.external_view.arrays["moments_signed_m2"],
                absolute_mean_by_stratum=self.external_view.arrays[
                    "moments_absolute_mean"
                ],
                absolute_m2_by_stratum=self.external_view.arrays["moments_absolute_m2"],
                look_index=cast(int, metadata["look_index"]),
                prior_state_hash=cast(str | None, metadata["prior_state_hash"]),
                schema=cast(str, metadata["schema"]),
            )
            result.assert_integrity(self.descriptor.moments_state_hash)
        except (TypeError, ValueError) as error:
            _refuse(
                V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                f"moments materialization failed: {error}",
            )
        return result

    def materialize_fd_snapshot(
        self,
        *,
        design: V1StratifiedFDDesign,
    ) -> V1StratifiedFDAccumulatorSnapshot:
        """Rebuild the canonical FD snapshot against a restated current design."""

        self._assert_open()
        if type(design) is not V1StratifiedFDDesign:
            _refuse(
                V1SampledExternalStateRefusalCode.INVALID_TYPE,
                "FD restoration requires the exact canonical design type",
            )
        try:
            design.assert_integrity()
        except (TypeError, ValueError) as error:
            _refuse(
                V1SampledExternalStateRefusalCode.ROUTE_DESIGN_MISMATCH,
                f"restated FD design is invalid: {error}",
            )
        if (
            design.design_hash != self.descriptor.fd_design_hash
            or len(design.route_ids) != self.descriptor.route_count
            or design.route_axis_hash != self.descriptor.route_axis_hash
            or design.stratum_names != self.descriptor.stratum_names
            or (
                self.descriptor.schema
                == SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA_V2
                and design.raw_logit_stream_source_hash
                != self.descriptor.literal_jacobian_stream_identity_hash
            )
        ):
            _refuse(
                V1SampledExternalStateRefusalCode.ROUTE_DESIGN_MISMATCH,
                "restated FD design does not match the external continuation",
            )
        children: list[tuple[str, DeviceFrequentDirectionsTrustAccumulatorSnapshot]] = (
            []
        )
        try:
            for index, metadata in enumerate(self._fd_strata_metadata):
                active_name, mean_name, diagonal_name = _fd_array_names(index)
                proof = _proof_from_metadata(metadata["bounded_source_proof"])
                mean = (
                    self.external_view.arrays[mean_name]
                    if metadata["mean_present"] is True
                    else None
                )
                child = DeviceFrequentDirectionsTrustAccumulatorSnapshot(
                    schema=cast(str, metadata["schema"]),
                    producer_algorithm_id=cast(str, metadata["producer_algorithm_id"]),
                    producer_implementation=cast(
                        str, metadata["producer_implementation"]
                    ),
                    route_ids=design.route_ids,
                    ordered_route_axis_hash=cast(
                        str, metadata["ordered_route_axis_hash"]
                    ),
                    sketch_rank=cast(int, metadata["sketch_rank"]),
                    dtype_identity=cast(str, metadata["dtype_identity"]),
                    active_buffer_rows=self.external_view.arrays[active_name],
                    filled_count=cast(int, metadata["filled_count"]),
                    mean=mean,
                    weighted_second_moment_diagonal=self.external_view.arrays[
                        diagonal_name
                    ],
                    cumulative_shrinkage=float(metadata["cumulative_shrinkage"]),
                    total_weight=float(metadata["total_weight"]),
                    context_count=cast(int, metadata["context_count"]),
                    context_ids=_string_tuple(
                        metadata["context_ids"], name="FD child contexts"
                    ),
                    bounded_source_proof=proof,
                    latest_source_row_hashes=_string_tuple(
                        metadata["latest_source_row_hashes"],
                        name="latest source-row hashes",
                    ),
                    snapshot_hash=cast(str, metadata["snapshot_hash"]),
                )
                children.append((cast(str, metadata["stratum_name"]), child))
            source_pairs = tuple(
                (
                    _identity(pair[0], name="source-ledger stratum"),
                    _string_tuple(pair[1], name="source ledger"),
                )
                for pair in cast(
                    Sequence[Sequence[object]],
                    self._fd_metadata["source_measurement_hashes_by_stratum"],
                )
            )
            result = V1StratifiedFDAccumulatorSnapshot(
                design_hash=cast(str, self._fd_metadata["design_hash"]),
                design_stratum_names=_string_tuple(
                    self._fd_metadata["design_stratum_names"],
                    name="FD stratum names",
                ),
                route_axis_hash=cast(str, self._fd_metadata["route_axis_hash"]),
                sketch_rank=cast(int, self._fd_metadata["sketch_rank"]),
                packet_boundary_index=cast(
                    int, self._fd_metadata["packet_boundary_index"]
                ),
                stratum_snapshots=tuple(children),
                source_measurement_hashes_by_stratum=source_pairs,
                schema=cast(str, self._fd_metadata["schema"]),
            )
            result.assert_integrity(self.descriptor.fd_snapshot_hash)
        except (TypeError, ValueError) as error:
            _refuse(
                V1SampledExternalStateRefusalCode.TYPED_IDENTITY_MISMATCH,
                f"FD materialization failed: {error}",
            )
        return result

    def close(self) -> None:
        if not self._closed:
            self.external_view.close()
            self._closed = True

    def __enter__(self) -> "V1SampledRegisteredLookExternalView":
        self._assert_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _open_sampled_registered_look_state(
    root: str | os.PathLike[str],
    descriptor: V1SampledRegisteredLookExternalDescriptor,
    *,
    issue_live_attachment: bool,
) -> tuple[
    V1SampledRegisteredLookExternalView,
    V1SampledRegisteredLookLiveAttachment | None,
]:
    """Verify and reopen one atomic sampled registered-look continuation."""

    if type(descriptor) is not V1SampledRegisteredLookExternalDescriptor:
        _refuse(
            V1SampledExternalStateRefusalCode.INVALID_TYPE,
            "typed open requires its exact descriptor type",
        )
    descriptor = V1SampledRegisteredLookExternalDescriptor.from_json_dict(
        descriptor.to_json_dict()
    )
    external_attachment: V1ExternalStateLiveAttachment | None = None
    if issue_live_attachment:
        external_view, external_attachment = open_external_state_with_attachment(
            root,
            descriptor.external_state,
        )
    else:
        external_view = open_external_state(root, descriptor.external_state)
    try:
        metadata = _strict_metadata(external_view.arrays[_METADATA_NAME])
        moments, fd, strata = _validate_typed_open(
            descriptor=descriptor,
            view=external_view,
            metadata=metadata,
        )
        result_view = V1SampledRegisteredLookExternalView(
            descriptor=descriptor,
            external_view=external_view,
            _moments_metadata=moments,
            _fd_metadata=fd,
            _fd_strata_metadata=strata,
        )
        typed_attachment = (
            None
            if external_attachment is None
            else _sampled_live_attachment(descriptor, external_attachment)
        )
        external_attachment = None
        return result_view, typed_attachment
    except BaseException:
        if external_attachment is not None:
            external_attachment.close()
        external_view.close()
        raise


def open_sampled_registered_look_state(
    root: str | os.PathLike[str],
    descriptor: V1SampledRegisteredLookExternalDescriptor,
) -> V1SampledRegisteredLookExternalView:
    view, attachment = _open_sampled_registered_look_state(
        root,
        descriptor,
        issue_live_attachment=False,
    )
    assert attachment is None
    return view


def open_sampled_registered_look_state_with_attachment(
    root: str | os.PathLike[str],
    descriptor: V1SampledRegisteredLookExternalDescriptor,
) -> tuple[
    V1SampledRegisteredLookExternalView,
    V1SampledRegisteredLookLiveAttachment,
]:
    """Run the exact typed cold barrier and issue a PID-bound hot attachment."""

    view, attachment = _open_sampled_registered_look_state(
        root,
        descriptor,
        issue_live_attachment=True,
    )
    assert attachment is not None
    return view, attachment


__all__ = [
    "SAMPLED_EXTERNAL_AUTHORITY_SCOPE",
    "SAMPLED_EXTERNAL_MEMORY_SCOPE",
    "SAMPLED_EXTERNAL_NAMESPACE",
    "SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA_V2",
    "SAMPLED_REGISTERED_LOOK_EXTERNAL_DESCRIPTOR_SCHEMA",
    "SAMPLED_REGISTERED_LOOK_EXTERNAL_STATE_SCHEMA",
    "V1SampledExternalMemoryForecast",
    "V1SampledExternalStateRefusal",
    "V1SampledExternalStateRefusalCode",
    "V1SampledRegisteredLookExternalDescriptor",
    "V1SampledRegisteredLookExternalView",
    "V1SampledRegisteredLookLiveAttachment",
    "open_sampled_registered_look_state",
    "open_sampled_registered_look_state_with_attachment",
    "publish_sampled_registered_look_state",
    "publish_sampled_registered_look_state_with_attachment",
]
