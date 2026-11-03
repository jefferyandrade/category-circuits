"""Restore sealed sampled corrected-A rows into signed-work device custody.

The sampled V5 acquisition already owns the exact corrected Aumann--Shapley
rows needed by the signed-F proposal operator.  This module is the narrow
post-FD bridge from that immutable route-window-major custody to the
``V1SignedWorkRowBankDeviceHandle``.  It does not recapture a context, run a
VJP, change the registered D prefix, or acquire F/Q/R certification authority.

Production restoration is CUDA-only.  Every corrected-A shard is opened once,
read sequentially once, role-domain hashed while it is read, and copied into
its exact context slice of one bounded device route-window buffer.  The bridge
never constructs a host ``context x route`` bank.  A guarded CPU mode exists
only for small-domain reference tests and carries no production authority.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, BinaryIO, Final, Mapping, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from .modern_v1_neutral_runtime import DeclaredRouteAxis
from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartMaterializedRows,
    V1SampledDualChartRowRole,
    V1SampledDualChartRowShardAttachment,
    _descriptor_invariant,
    _file_invariant,
    _new_payload_hasher,
)
from .modern_v1_sampled_literal_j_fd_rank_replay import (
    V1SampledLiteralJOriginalCaptureBinding,
)
from .modern_v1_signed_work_row_spool_cuda import (
    V1SignedWorkRowBankDeviceHandle,
    V1SignedWorkRowSpoolCUDAConsumer,
    V1SignedWorkRowSpoolExecutionMode,
    V1SignedWorkRowSpoolRefusal,
    V1SignedWorkRowSpoolRefusalCode,
    V1SignedWorkRowSpoolRequest,
    forecast_v1_signed_work_row_spool_hbm,
)

try:  # pragma: no cover - availability is host dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SAMPLED_SIGNED_WORK_ROW_BANK_RESTORATION_SCHEMA: Final = (
    "partition_free_v1_sampled_corrected_A_signed_work_row_bank_restoration_v1"
)
SAMPLED_SIGNED_WORK_ROW_BANK_RESTORATION_AUTHORITY_SCOPE: Final = (
    "sampled_D_corrected_A_signed_work_proposal_input_restoration_only_no_FQR_"
    "certification_C_calibration_oracle_mask_commit_terminal_or_pareto_"
    "authority_v1"
)
SAMPLED_SIGNED_WORK_ROW_BANK_RESTORATION_MEMORY_SCOPE: Final = (
    "one_device_context_route_bank_plus_one_device_context_route_window_and_"
    "one_host_single_shard_route_window_no_host_context_route_bank_v1"
)

_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-signed-work-row-bank-restoration-receipt-"
)
_TRANSPORT_PREFIX: Final = (
    "partition-free-v1-sampled-signed-work-row-bank-restoration-transport-"
)
_SOURCE_AXIS_PREFIX: Final = (
    "partition-free-v1-sampled-signed-work-original-source-measurement-axis-"
)
_FLOAT64_BYTES: Final = 8
_RECEIPT_ISSUANCE_TOKEN: Final = object()
_RESULT_ISSUANCE_TOKEN: Final = object()


class V1SampledSignedWorkRowBankRestorationRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    AXIS_MISMATCH = "axis_mismatch"
    STALE_IDENTITY = "stale_identity"
    SOURCE_PROOF_MISMATCH = "source_proof_mismatch"
    DUPLICATE_OR_OUT_OF_ORDER_SHARD = "duplicate_or_out_of_order_shard"
    PAYLOAD_MISMATCH = "payload_mismatch"
    MEMORY_INADMISSIBLE = "memory_inadmissible"
    DEVICE_REFUSAL = "device_refusal"
    AUTHORITY_VIOLATION = "authority_violation"
    CUSTODY_MUTATED = "custody_mutated"


class V1SampledSignedWorkRowBankRestorationRefusal(RuntimeError):
    """Typed fail-closed boundary for the sampled-A restoration seam."""

    def __init__(
        self,
        code: V1SampledSignedWorkRowBankRestorationRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledSignedWorkRowBankRestorationRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledSignedWorkRowBankRestorationRefusal(code, message)


def _refuse_identity_mismatch(
    fields: tuple[tuple[str, object, object], ...],
    *,
    message: str,
) -> NoReturn:
    mismatched = tuple(name for name, left, right in fields if left != right)
    if not mismatched:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.STALE_IDENTITY,
            message,
        )
    rendered = ", ".join(mismatched[:8])
    if len(mismatched) > 8:
        rendered = f"{rendered}, ... (+{len(mismatched) - 8} more)"
    _refuse(
        V1SampledSignedWorkRowBankRestorationRefusalCode.STALE_IDENTITY,
        f"{message}; mismatched_fields=({rendered})",
    )


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _flatten(values: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    return tuple(item for packet in values for item in packet)


def _read_exact_bytes(stream: BinaryIO, expected_bytes: int) -> tuple[bytearray, int]:
    """Read one registered rectangle without a second payload-sized copy."""

    size = _nonnegative_int(expected_bytes, name="registered rectangle bytes")
    raw = bytearray(size)
    target = memoryview(raw)
    cursor = 0
    calls = 0
    while cursor < size:
        observed = stream.readinto(target[cursor:])
        calls += 1
        if observed is None or observed <= 0:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.PAYLOAD_MISMATCH,
                "sealed corrected-A shard ended before its two-dimensional cover",
            )
        cursor += int(observed)
    return raw, calls


def _poison(attachments: tuple[V1SampledDualChartRowShardAttachment, ...]) -> None:
    for attachment in attachments:
        try:
            setattr(attachment, "_integrity_failed", True)
        except (AttributeError, TypeError):  # pragma: no cover - defensive only.
            pass


def _translate_spool_refusal(error: V1SignedWorkRowSpoolRefusal) -> NoReturn:
    if error.code is V1SignedWorkRowSpoolRefusalCode.MEMORY_INADMISSIBLE:
        code = V1SampledSignedWorkRowBankRestorationRefusalCode.MEMORY_INADMISSIBLE
    elif error.code in {
        V1SignedWorkRowSpoolRefusalCode.DEVICE_MISMATCH,
        V1SignedWorkRowSpoolRefusalCode.NONFINITE_ROW,
    }:
        code = V1SampledSignedWorkRowBankRestorationRefusalCode.DEVICE_REFUSAL
    elif error.code is V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION:
        code = V1SampledSignedWorkRowBankRestorationRefusalCode.AUTHORITY_VIOLATION
    elif error.code in {
        V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
        V1SignedWorkRowSpoolRefusalCode.STALE_IDENTITY,
        V1SignedWorkRowSpoolRefusalCode.DUPLICATE_OR_OUT_OF_ORDER_CHUNK,
        V1SignedWorkRowSpoolRefusalCode.PARTIAL_PACKET,
    }:
        code = V1SampledSignedWorkRowBankRestorationRefusalCode.AXIS_MISMATCH
    else:
        code = V1SampledSignedWorkRowBankRestorationRefusalCode.CUSTODY_MUTATED
    raise V1SampledSignedWorkRowBankRestorationRefusal(
        code,
        "signed-work device custody refused the sampled-A restoration",
    ) from error


@dataclass(frozen=True, slots=True)
class V1SampledSignedWorkRowBankRestorationReceipt:
    """Path-free proof of one exact sampled-A to device-custody restoration."""

    materialization_hash: str
    row_prefix_manifest_hash: str
    original_capture_binding_hash: str
    registered_d_prefix_hash: str
    rank_invariant_raw_prefix_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    source_measurement_axis_hash: str
    corrected_a_shard_receipt_hashes: tuple[str, ...]
    corrected_a_shard_content_sha256s: tuple[str, ...]
    source_capture_identity_hashes: tuple[str, ...]
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    context_count: int
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    row_spool_request_hash: str
    row_spool_memory_forecast_hash: str
    row_spool_custody_receipt_hash: str
    destination_transport_hash: str
    execution_mode: V1SignedWorkRowSpoolExecutionMode
    declared_device: str
    corrected_a_shard_count: int
    maximum_shard_context_count: int
    registered_route_chunk_size: int
    route_window_count: int
    two_dimensional_rectangle_count: int
    descriptor_open_count: int
    sequential_descriptor_pass_count: int
    descriptor_readinto_call_count: int
    digest_update_count: int
    payload_read_bytes: int
    trailing_probe_count: int
    trailing_probe_bytes: int
    maximum_owned_host_staging_bytes: int
    device_route_window_staging_bytes: int
    row_bank_device_bytes: int
    complete_two_dimensional_cover: bool = True
    every_context_route_cell_written_exactly_once: bool = True
    every_source_descriptor_read_once: bool = True
    source_measurements_from_original_capture_only: bool = True
    source_host_context_route_bank_materialized: bool = False
    production_cuda_proposal_input_authority: bool = True
    cpu_reference_test_only: bool = False
    private_cpu_reference_device_bank_materialized: bool = False
    fqr_certification_authority: bool = False
    c_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    memory_scope: str = SAMPLED_SIGNED_WORK_ROW_BANK_RESTORATION_MEMORY_SCOPE
    authority_scope: str = SAMPLED_SIGNED_WORK_ROW_BANK_RESTORATION_AUTHORITY_SCOPE
    schema: str = SAMPLED_SIGNED_WORK_ROW_BANK_RESTORATION_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _RECEIPT_ISSUANCE_TOKEN:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.AUTHORITY_VIOLATION,
                "restoration receipts are issued only after exact source close",
            )
        for name in (
            "materialization_hash",
            "row_prefix_manifest_hash",
            "original_capture_binding_hash",
            "registered_d_prefix_hash",
            "rank_invariant_raw_prefix_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "corrected_a_stream_identity_hash",
            "literal_j_stream_identity_hash",
            "source_measurement_axis_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "row_spool_request_hash",
            "row_spool_memory_forecast_hash",
            "row_spool_custody_receipt_hash",
            "destination_transport_hash",
            "declared_device",
        ):
            _identity(getattr(self, name), name=name.replace("_", " "))
        routes = _positive_int(self.route_count, name="receipt route count")
        contexts = _positive_int(self.context_count, name="receipt context count")
        shards = _positive_int(
            self.corrected_a_shard_count,
            name="receipt corrected-A shard count",
        )
        maximum_shard_contexts = _positive_int(
            self.maximum_shard_context_count,
            name="receipt maximum shard context count",
        )
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="receipt registered route chunk size",
        )
        if chunk > routes:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.AXIS_MISMATCH,
                "receipt route chunk exceeds the route axis",
            )
        windows = _positive_int(
            self.route_window_count,
            name="receipt route-window count",
        )
        if windows != (routes + chunk - 1) // chunk:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.PAYLOAD_MISMATCH,
                "receipt route-window count does not replay from its chunk size",
            )
        rectangles = shards * windows
        if (
            len(self.corrected_a_shard_receipt_hashes) != shards
            or len(self.corrected_a_shard_content_sha256s) != shards
            or len(self.source_capture_identity_hashes) != shards
            or len(set(self.corrected_a_shard_receipt_hashes)) != shards
            or len(set(self.source_capture_identity_hashes)) != shards
        ):
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.DUPLICATE_OR_OUT_OF_ORDER_SHARD,
                "restoration receipt has a duplicate or incomplete shard axis",
            )
        expected_exact = {
            "two_dimensional_rectangle_count": rectangles,
            "descriptor_open_count": shards,
            "sequential_descriptor_pass_count": shards,
            "digest_update_count": rectangles,
            "payload_read_bytes": contexts * routes * _FLOAT64_BYTES,
            "trailing_probe_count": shards,
            "trailing_probe_bytes": 0,
            "maximum_owned_host_staging_bytes": (
                maximum_shard_contexts * min(routes, chunk) * _FLOAT64_BYTES
            ),
            "device_route_window_staging_bytes": (contexts * chunk * _FLOAT64_BYTES),
            "row_bank_device_bytes": contexts * routes * _FLOAT64_BYTES,
        }
        for name, expected in expected_exact.items():
            if _nonnegative_int(getattr(self, name), name=name) != expected:
                _refuse(
                    V1SampledSignedWorkRowBankRestorationRefusalCode.PAYLOAD_MISMATCH,
                    f"{name} does not replay from the registered two-dimensional cover",
                )
        if self.descriptor_readinto_call_count < rectangles:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.PAYLOAD_MISMATCH,
                "descriptor read telemetry is smaller than the rectangle cover",
            )
        exact_true = (
            self.complete_two_dimensional_cover,
            self.every_context_route_cell_written_exactly_once,
            self.every_source_descriptor_read_once,
            self.source_measurements_from_original_capture_only,
        )
        if any(type(value) is not bool or value is not True for value in exact_true):
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.PAYLOAD_MISMATCH,
                "restoration did not close its exact source/cover proof",
            )
        if self.source_host_context_route_bank_materialized is not False:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.AUTHORITY_VIOLATION,
                "sampled-A restoration may not reconstruct a source host row bank",
            )
        if type(self.execution_mode) is not V1SignedWorkRowSpoolExecutionMode:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.INVALID_TYPE,
                "restoration execution mode must be exact",
            )
        production = (
            self.execution_mode is V1SignedWorkRowSpoolExecutionMode.CUDA_PRODUCTION
        )
        if (
            self.production_cuda_proposal_input_authority is not production
            or self.cpu_reference_test_only is production
            or self.private_cpu_reference_device_bank_materialized is production
            or (production and not self.declared_device.startswith("cuda:"))
            or (not production and self.declared_device != "cpu")
        ):
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.AUTHORITY_VIOLATION,
                "restoration authority differs from its registered execution mode",
            )
        downstream = (
            self.fqr_certification_authority,
            self.c_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool or value is not False for value in downstream):
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.AUTHORITY_VIOLATION,
                "proposal-input restoration acquired downstream decision authority",
            )
        if (
            self.memory_scope != SAMPLED_SIGNED_WORK_ROW_BANK_RESTORATION_MEMORY_SCOPE
            or self.authority_scope
            != SAMPLED_SIGNED_WORK_ROW_BANK_RESTORATION_AUTHORITY_SCOPE
            or self.schema != SAMPLED_SIGNED_WORK_ROW_BANK_RESTORATION_SCHEMA
        ):
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.AUTHORITY_VIOLATION,
                "restoration schema, memory scope, or authority scope changed",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "context_count", contexts)
        object.__setattr__(self, "corrected_a_shard_count", shards)
        object.__setattr__(self, "maximum_shard_context_count", maximum_shard_contexts)
        object.__setattr__(self, "registered_route_chunk_size", chunk)
        object.__setattr__(self, "route_window_count", windows)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: (value.value if isinstance(value, Enum) else value)
            for name, value in (
                (field_.name, getattr(self, field_.name))
                for field_ in self.__dataclass_fields__.values()
                if not field_.name.startswith("_")
            )
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.CUSTODY_MUTATED,
                "sampled-A restoration receipt was mutated",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.CUSTODY_MUTATED,
                "sampled-A restoration receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class V1SampledSignedWorkRowBankRestorationResult:
    """Live device handle plus its path-free proposal-input-only proof."""

    receipt: V1SampledSignedWorkRowBankRestorationReceipt
    row_bank_handle: V1SignedWorkRowBankDeviceHandle = field(
        repr=False,
        compare=False,
    )
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _RESULT_ISSUANCE_TOKEN:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.AUTHORITY_VIOLATION,
                "live restoration results are issued only after exact close",
            )
        if type(self.receipt) is not V1SampledSignedWorkRowBankRestorationReceipt:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.INVALID_TYPE,
                "restoration result requires its exact receipt",
            )
        if type(self.row_bank_handle) is not V1SignedWorkRowBankDeviceHandle:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.INVALID_TYPE,
                "restoration result requires exact row-bank custody",
            )
        self.receipt.assert_integrity()
        custody = self.row_bank_handle.receipt
        if (
            custody.receipt_hash != self.receipt.row_spool_custody_receipt_hash
            or custody.registered_d_prefix_hash != self.receipt.registered_d_prefix_hash
            or custody.chart_identity_hash != self.receipt.chart_identity_hash
            or custody.route_axis_hash != self.receipt.route_axis_hash
            or custody.route_id_axis_hash != self.receipt.route_id_axis_hash
        ):
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.CUSTODY_MUTATED,
                "live row-bank custody differs from the restoration receipt",
            )

    def signed_work_operator_kwargs(self) -> Mapping[str, object]:
        """Return the exact identity arguments for ``from_row_bank_device_handle``."""

        custody = self.row_bank_handle.receipt
        return MappingProxyType(
            {
                "row_bank_handle": self.row_bank_handle,
                "expected_custody_receipt_hash": custody.receipt_hash,
                "expected_registered_d_prefix_hash": (
                    self.receipt.registered_d_prefix_hash
                ),
                "expected_chart_identity_hash": self.receipt.chart_identity_hash,
                "expected_raw_logit_stream_identity_hash": (
                    self.receipt.corrected_a_stream_identity_hash
                ),
                "expected_native_fiber_axis_receipt_hash": (
                    self.receipt.native_fiber_axis_receipt_hash
                ),
                "expected_route_axis_hash": self.receipt.route_axis_hash,
                "route_id_axis_hash": self.receipt.route_id_axis_hash,
                "source_geometry_hash": self.receipt.receipt_hash,
            }
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("live sampled-A restoration results are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live sampled-A restoration results are nonserializable")


def _validate_sources(
    *,
    materialized_rows: V1SampledDualChartMaterializedRows,
    original_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    compact_route_axis: DeclaredRouteAxis,
) -> tuple[
    tuple[V1SampledDualChartRowShardAttachment, ...],
    tuple[str, ...],
]:
    if type(materialized_rows) is not V1SampledDualChartMaterializedRows:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.INVALID_TYPE,
            "restoration requires exact sampled dual-chart materialization",
        )
    if type(original_capture_binding) is not V1SampledLiteralJOriginalCaptureBinding:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.INVALID_TYPE,
            "restoration requires the exact original-capture binding",
        )
    if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.INVALID_TYPE,
            "restoration requires exact native-fiber custody",
        )
    if type(compact_route_axis) is not DeclaredRouteAxis:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.INVALID_TYPE,
            "restoration requires an exact compact DeclaredRouteAxis",
        )
    materialized_rows.assert_integrity()
    original_capture_binding.assert_integrity()
    native_fiber_axis_receipt.assert_integrity()
    manifest = materialized_rows.attachment.manifest
    attachments = tuple(materialized_rows.attachment.corrected_a_attachments)
    expected_attachment_hashes = tuple(
        look.corrected_a_shard.receipt_hash for look in manifest.looks
    )
    observed_attachment_hashes = tuple(
        item.receipt.receipt_hash for item in attachments
    )
    if (
        len(attachments) != len(manifest.looks)
        or len(set(observed_attachment_hashes)) != len(attachments)
        or observed_attachment_hashes != expected_attachment_hashes
    ):
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.DUPLICATE_OR_OUT_OF_ORDER_SHARD,
            "corrected-A attachments repeat or differ from manifest order",
        )
    materialized_rows.attachment.verify_live_witnesses()
    shared = (
        (
            "materialized_rows.manifest_hash",
            materialized_rows.manifest_hash,
            manifest.manifest_hash,
        ),
        (
            "original_capture_binding.row_prefix_manifest_hash",
            original_capture_binding.row_prefix_manifest_hash,
            manifest.manifest_hash,
        ),
        (
            "original_capture_binding.registered_d_prefix_hash",
            original_capture_binding.registered_d_prefix_hash,
            manifest.registered_d_prefix_hash,
        ),
        (
            "original_capture_binding.chart_identity_hash",
            original_capture_binding.chart_identity_hash,
            manifest.chart_identity_hash,
        ),
        (
            "original_capture_binding.current_mask_hash",
            original_capture_binding.current_mask_hash,
            manifest.current_mask_hash,
        ),
        (
            "original_capture_binding.proposal_generation",
            original_capture_binding.proposal_generation,
            manifest.proposal_generation,
        ),
        (
            "original_capture_binding.route_axis_hash",
            original_capture_binding.route_axis_hash,
            manifest.route_axis_hash,
        ),
        (
            "original_capture_binding.route_id_axis_hash",
            original_capture_binding.route_id_axis_hash,
            manifest.route_id_axis_hash,
        ),
        (
            "original_capture_binding.route_count",
            original_capture_binding.route_count,
            manifest.route_count,
        ),
        (
            "original_capture_binding.corrected_a_stream_identity_hash",
            original_capture_binding.corrected_a_stream_identity_hash,
            manifest.corrected_a_stream_identity_hash,
        ),
        (
            "original_capture_binding.literal_j_stream_identity_hash",
            original_capture_binding.literal_j_stream_identity_hash,
            manifest.literal_j_stream_identity_hash,
        ),
        (
            "compact_route_axis.route_axis_hash",
            compact_route_axis.route_axis_hash,
            manifest.route_axis_hash,
        ),
        (
            "compact_route_axis.route_id_axis_hash",
            compact_route_axis.route_id_axis_hash,
            manifest.route_id_axis_hash,
        ),
        (
            "compact_route_axis.route_count",
            compact_route_axis.route_count,
            manifest.route_count,
        ),
        (
            "native_fiber_axis_receipt.route_axis_hash",
            native_fiber_axis_receipt.route_axis_hash,
            manifest.route_axis_hash,
        ),
        (
            "native_fiber_axis_receipt.route_id_axis_hash",
            native_fiber_axis_receipt.route_id_axis_hash,
            manifest.route_id_axis_hash,
        ),
        (
            "native_fiber_axis_receipt.route_count",
            native_fiber_axis_receipt.route_count,
            manifest.route_count,
        ),
    )
    if any(left != right for _, left, right in shared):
        _refuse_identity_mismatch(
            shared,
            message=(
                "manifest, original capture, compact route, or native-fiber "
                "identity differs"
            ),
        )
    looks = manifest.looks
    packet_count = len(original_capture_binding.packet_plan_hashes)
    if len(attachments) != len(looks) or len(attachments) != packet_count:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.SOURCE_PROOF_MISMATCH,
            "corrected-A shards and original capture packets have different counts",
        )
    if len({item.receipt.receipt_hash for item in attachments}) != len(attachments):
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.DUPLICATE_OR_OUT_OF_ORDER_SHARD,
            "corrected-A restoration cannot repeat a shard",
        )
    expected_contexts = tuple(look.context_ids for look in looks)
    expected_draws = tuple(look.draw_ids for look in looks)
    expected_strata = tuple(look.stratum_labels for look in looks)
    if (
        original_capture_binding.packet_context_ids != expected_contexts
        or original_capture_binding.packet_draw_ids != expected_draws
        or original_capture_binding.packet_stratum_labels != expected_strata
    ):
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.AXIS_MISMATCH,
            "original capture packet axes differ from corrected-A custody",
        )
    captures = tuple(
        look.corrected_a_shard.source_capture_identity_hash for look in looks
    )
    if original_capture_binding.packet_source_capture_identity_hashes != captures:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.SOURCE_PROOF_MISMATCH,
            "original capture identities differ from corrected-A shard custody",
        )
    for index, (attachment, look) in enumerate(zip(attachments, looks, strict=True)):
        receipt = attachment.receipt
        if (
            receipt.role is not V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY
            or receipt.receipt_hash != look.corrected_a_shard.receipt_hash
            or receipt.look_ordinal != index
            or receipt.context_ids != expected_contexts[index]
            or receipt.draw_ids != expected_draws[index]
            or receipt.stratum_labels != expected_strata[index]
        ):
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.DUPLICATE_OR_OUT_OF_ORDER_SHARD,
                "corrected-A attachment order differs from the append-only manifest",
            )
    measurements = _flatten(
        original_capture_binding.packet_fd_source_measurement_hashes
    )
    if len(measurements) != manifest.context_count or len(set(measurements)) != len(
        measurements
    ):
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.SOURCE_PROOF_MISMATCH,
            "original-capture source-measurement axis is incomplete or repeated",
        )
    if (
        _flatten(original_capture_binding.packet_context_ids)
        != materialized_rows.context_ids
        or _flatten(original_capture_binding.packet_draw_ids)
        != materialized_rows.draw_ids
        or _flatten(original_capture_binding.packet_stratum_labels)
        != materialized_rows.stratum_labels
    ):
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.AXIS_MISMATCH,
            "materialized D axes are not the original-capture prefix",
        )
    return attachments, measurements


def restore_v1_sampled_signed_work_row_bank_cuda(
    *,
    materialized_rows: V1SampledDualChartMaterializedRows,
    original_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
    rank_invariant_raw_prefix_hash: str,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    compact_route_axis: DeclaredRouteAxis,
    memory_budget_bytes: int,
    current_reserved_device_bytes: int,
    declared_device: str,
    execution_mode: V1SignedWorkRowSpoolExecutionMode,
    cpu_reference_maximum_cells: int | None = None,
) -> V1SampledSignedWorkRowBankRestorationResult:
    """Stream one sealed corrected-A prefix into exact signed-work custody."""

    if torch is None:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.DEVICE_REFUSAL,
            "sampled-A restoration requires PyTorch",
        )
    rank_prefix = _identity(
        rank_invariant_raw_prefix_hash,
        name="rank-invariant raw-prefix hash",
    )
    if type(execution_mode) is not V1SignedWorkRowSpoolExecutionMode:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.INVALID_TYPE,
            "restoration execution mode must be exact",
        )
    attachments, source_measurements = _validate_sources(
        materialized_rows=materialized_rows,
        original_capture_binding=original_capture_binding,
        native_fiber_axis_receipt=native_fiber_axis_receipt,
        compact_route_axis=compact_route_axis,
    )
    manifest = materialized_rows.attachment.manifest
    contexts = manifest.context_count
    routes = manifest.route_count
    chunk_size = manifest.looks[0].corrected_a_shard.registered_route_chunk_size
    cells = contexts * routes
    if execution_mode is V1SignedWorkRowSpoolExecutionMode.CPU_REFERENCE_TEST_ONLY:
        if cpu_reference_maximum_cells is None:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.AUTHORITY_VIOLATION,
                "CPU reference restoration requires an explicit small-domain bound",
            )
        maximum = _positive_int(
            cpu_reference_maximum_cells,
            name="CPU reference maximum cells",
        )
        if cells > maximum:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.AUTHORITY_VIOLATION,
                "sampled-A prefix exceeds the registered CPU reference bound",
            )
    elif cpu_reference_maximum_cells is not None:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.AUTHORITY_VIOLATION,
            "production CUDA restoration cannot enable the CPU reference path",
        )
    try:
        runtime_device = torch.device(
            _identity(declared_device, name="declared device")
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise V1SampledSignedWorkRowBankRestorationRefusal(
            V1SampledSignedWorkRowBankRestorationRefusalCode.DEVICE_REFUSAL,
            "declared restoration device is invalid",
        ) from error
    if execution_mode is V1SignedWorkRowSpoolExecutionMode.CUDA_PRODUCTION:
        if (
            runtime_device.type != "cuda"
            or runtime_device.index is None
            or not bool(torch.cuda.is_available())
        ):
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.DEVICE_REFUSAL,
                "production sampled-A restoration requires an available explicit CUDA device",
            )
    elif str(runtime_device) != "cpu":
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.AUTHORITY_VIOLATION,
            "private CPU reference restoration requires the exact CPU device",
        )
    budget = _nonnegative_int(memory_budget_bytes, name="fresh HBM budget bytes")
    reserved = _nonnegative_int(
        current_reserved_device_bytes,
        name="fresh post-FD reserved device bytes",
    )
    forecast = forecast_v1_signed_work_row_spool_hbm(
        route_count=routes,
        context_count=contexts,
        registered_route_chunk_size=chunk_size,
        memory_budget_bytes=budget,
        current_reserved_device_bytes=reserved,
    )
    if not forecast.admitted:
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.MEMORY_INADMISSIBLE,
            "exact sampled-A row bank and bounded staging exceed the fresh budget",
        )
    destination_transport_hash = content_hash(
        {
            "schema": "sampled_corrected_A_signed_work_restoration_transport_v1",
            "materialization_hash": materialized_rows.materialization_hash,
            "original_capture_binding_hash": original_capture_binding.binding_hash,
            "registered_d_prefix_hash": manifest.registered_d_prefix_hash,
            "rank_invariant_raw_prefix_hash": rank_prefix,
            "corrected_a_stream_identity_hash": (
                manifest.corrected_a_stream_identity_hash
            ),
        },
        prefix=_TRANSPORT_PREFIX,
    )
    request = V1SignedWorkRowSpoolRequest(
        route_ids=compact_route_axis.route_ids,
        route_axis_hash=compact_route_axis.route_axis_hash,
        context_ids=materialized_rows.context_ids,
        draw_ids=materialized_rows.draw_ids,
        stratum_labels=materialized_rows.stratum_labels,
        context_weights=tuple(
            float(value) for value in materialized_rows.normalized_context_weights
        ),
        source_measurement_hashes=source_measurements,
        registered_d_prefix_hash=manifest.registered_d_prefix_hash,
        chart_identity_hash=manifest.chart_identity_hash,
        current_mask_hash=manifest.current_mask_hash,
        proposal_generation=manifest.proposal_generation,
        destination_transport_hash=destination_transport_hash,
        raw_logit_stream_identity_hash=manifest.corrected_a_stream_identity_hash,
        native_fiber_axis_receipt=native_fiber_axis_receipt,
        registered_route_chunk_size=chunk_size,
        memory_budget_bytes=budget,
        declared_device=str(runtime_device),
        execution_mode=execution_mode,
        current_reserved_device_bytes=reserved,
    )
    try:
        consumer = V1SignedWorkRowSpoolCUDAConsumer(request)
        consumer.bind_route_chunk_size(chunk_size)
        consumer.begin_packet(
            materialized_rows.context_ids,
            route_ids=compact_route_axis.route_ids,
            destination_transport_hash=destination_transport_hash,
            loss_pullback_hash=manifest.corrected_a_stream_identity_hash,
        )
    except V1SignedWorkRowSpoolRefusal as error:
        _translate_spool_refusal(error)

    shard_states: list[dict[str, Any]] = []
    context_cursor = 0
    payload_read_bytes = 0
    readinto_calls = 0
    digest_updates = 0
    written_cells = 0
    maximum_host_staging = 0
    trailing_probe_bytes = 0
    route_windows = tuple(attachments[0].receipt.route_windows)
    if any(tuple(item.receipt.route_windows) != route_windows for item in attachments):
        _refuse(
            V1SampledSignedWorkRowBankRestorationRefusalCode.AXIS_MISMATCH,
            "corrected-A shards do not share one registered route-window cover",
        )
    try:
        # One flat allocation supports a contiguous [context,width] view for
        # every width, including the final short window.
        device_staging_storage = torch.empty(
            contexts * chunk_size,
            dtype=torch.float64,
            device=runtime_device,
        )
    except (RuntimeError, AssertionError) as error:
        raise V1SampledSignedWorkRowBankRestorationRefusal(
            V1SampledSignedWorkRowBankRestorationRefusalCode.DEVICE_REFUSAL,
            "bounded device route-window staging allocation failed",
        ) from error

    try:
        with ExitStack() as stack:
            for attachment in attachments:
                receipt = attachment.receipt
                path = getattr(attachment, "_path")
                sealed_invariant = getattr(attachment, "_invariant")
                if getattr(attachment, "_integrity_failed"):
                    _refuse(
                        V1SampledSignedWorkRowBankRestorationRefusalCode.CUSTODY_MUTATED,
                        "corrected-A attachment was already poisoned",
                    )
                if _file_invariant(path) != sealed_invariant:
                    _refuse(
                        V1SampledSignedWorkRowBankRestorationRefusalCode.CUSTODY_MUTATED,
                        "corrected-A shard invariant changed before restoration",
                    )
                stream = stack.enter_context(path.open("rb", buffering=0))
                descriptor_before = _descriptor_invariant(stream.fileno())
                if descriptor_before != sealed_invariant:
                    _refuse(
                        V1SampledSignedWorkRowBankRestorationRefusalCode.CUSTODY_MUTATED,
                        "opened corrected-A descriptor differs from sealed custody",
                    )
                context_stop = context_cursor + len(receipt.context_ids)
                if context_stop > contexts:
                    _refuse(
                        V1SampledSignedWorkRowBankRestorationRefusalCode.AXIS_MISMATCH,
                        "corrected-A context slices exceed the materialized prefix",
                    )
                shard_states.append(
                    {
                        "receipt": receipt,
                        "path": path,
                        "sealed_invariant": sealed_invariant,
                        "stream": stream,
                        "descriptor_before": descriptor_before,
                        "digest": _new_payload_hasher(receipt.role),
                        "context_start": context_cursor,
                        "context_stop": context_stop,
                        "route_cursor": 0,
                        "payload_bytes": 0,
                    }
                )
                context_cursor = context_stop
            if context_cursor != contexts:
                _refuse(
                    V1SampledSignedWorkRowBankRestorationRefusalCode.AXIS_MISMATCH,
                    "corrected-A context slices do not cover the materialized prefix",
                )

            for start, stop in route_windows:
                width = stop - start
                device_window = device_staging_storage[: contexts * width].view(
                    contexts,
                    width,
                )
                for state in shard_states:
                    if state["route_cursor"] != start:
                        _refuse(
                            V1SampledSignedWorkRowBankRestorationRefusalCode.DUPLICATE_OR_OUT_OF_ORDER_SHARD,
                            "corrected-A rectangle order repeated or skipped a route window",
                        )
                    local_contexts = state["context_stop"] - state["context_start"]
                    expected_bytes = local_contexts * width * _FLOAT64_BYTES
                    raw, calls = _read_exact_bytes(state["stream"], expected_bytes)
                    state["digest"].update(raw)
                    state["route_cursor"] = stop
                    state["payload_bytes"] += expected_bytes
                    payload_read_bytes += expected_bytes
                    readinto_calls += calls
                    digest_updates += 1
                    maximum_host_staging = max(maximum_host_staging, expected_bytes)
                    host_window = np.frombuffer(raw, dtype=np.dtype("<f8")).reshape(
                        local_contexts,
                        width,
                    )
                    with torch.no_grad():
                        device_window[
                            state["context_start"] : state["context_stop"]
                        ].copy_(torch.from_numpy(host_window), non_blocking=False)
                    written_cells += local_contexts * width
                consumer.consume_positive_removal_chunk(start, stop, device_window)

            for state in shard_states:
                trailing = state["stream"].read(1)
                trailing_probe_bytes += len(trailing)
                descriptor_after = _descriptor_invariant(state["stream"].fileno())
                if descriptor_after != state["descriptor_before"]:
                    _refuse(
                        V1SampledSignedWorkRowBankRestorationRefusalCode.CUSTODY_MUTATED,
                        "corrected-A descriptor invariant changed during restoration",
                    )
                if _file_invariant(state["path"]) != state["sealed_invariant"]:
                    _refuse(
                        V1SampledSignedWorkRowBankRestorationRefusalCode.CUSTODY_MUTATED,
                        "corrected-A path invariant changed during restoration",
                    )
                receipt = state["receipt"]
                if (
                    trailing
                    or state["route_cursor"] != routes
                    or state["payload_bytes"] != receipt.exact_payload_file_bytes
                    or state["digest"].hexdigest() != receipt.content_sha256
                ):
                    _refuse(
                        V1SampledSignedWorkRowBankRestorationRefusalCode.PAYLOAD_MISMATCH,
                        "corrected-A bytes, trailing probe, or role-domain digest differ",
                    )
        if written_cells != cells or payload_read_bytes != cells * _FLOAT64_BYTES:
            _refuse(
                V1SampledSignedWorkRowBankRestorationRefusalCode.PAYLOAD_MISMATCH,
                "restoration did not cover every context-route cell exactly once",
            )
        del device_staging_storage
        consumer.end_packet()
    except V1SignedWorkRowSpoolRefusal as error:
        _poison(attachments)
        _translate_spool_refusal(error)
    except V1SampledSignedWorkRowBankRestorationRefusal:
        _poison(attachments)
        raise
    except (OSError, RuntimeError, AssertionError) as error:
        _poison(attachments)
        raise V1SampledSignedWorkRowBankRestorationRefusal(
            V1SampledSignedWorkRowBankRestorationRefusalCode.CUSTODY_MUTATED,
            "sampled-A restoration could not close one-pass descriptor custody",
        ) from error

    custody = consumer.receipt
    production = execution_mode is V1SignedWorkRowSpoolExecutionMode.CUDA_PRODUCTION
    receipt = V1SampledSignedWorkRowBankRestorationReceipt(
        materialization_hash=materialized_rows.materialization_hash,
        row_prefix_manifest_hash=manifest.manifest_hash,
        original_capture_binding_hash=original_capture_binding.binding_hash,
        registered_d_prefix_hash=manifest.registered_d_prefix_hash,
        rank_invariant_raw_prefix_hash=rank_prefix,
        chart_identity_hash=manifest.chart_identity_hash,
        current_mask_hash=manifest.current_mask_hash,
        proposal_generation=manifest.proposal_generation,
        corrected_a_stream_identity_hash=manifest.corrected_a_stream_identity_hash,
        literal_j_stream_identity_hash=manifest.literal_j_stream_identity_hash,
        source_measurement_axis_hash=content_hash(
            source_measurements,
            prefix=_SOURCE_AXIS_PREFIX,
        ),
        corrected_a_shard_receipt_hashes=tuple(
            item.receipt.receipt_hash for item in attachments
        ),
        corrected_a_shard_content_sha256s=tuple(
            item.receipt.content_sha256 for item in attachments
        ),
        source_capture_identity_hashes=tuple(
            item.receipt.source_capture_identity_hash for item in attachments
        ),
        route_axis_hash=manifest.route_axis_hash,
        route_id_axis_hash=manifest.route_id_axis_hash,
        route_count=routes,
        context_count=contexts,
        native_fiber_axis_receipt_hash=native_fiber_axis_receipt.receipt_hash,
        native_fiber_measure_hash=native_fiber_axis_receipt.measure_hash,
        row_spool_request_hash=request.request_hash,
        row_spool_memory_forecast_hash=forecast.forecast_hash,
        row_spool_custody_receipt_hash=custody.receipt_hash,
        destination_transport_hash=destination_transport_hash,
        execution_mode=execution_mode,
        declared_device=str(runtime_device),
        corrected_a_shard_count=len(attachments),
        maximum_shard_context_count=max(
            len(item.receipt.context_ids) for item in attachments
        ),
        registered_route_chunk_size=chunk_size,
        route_window_count=len(route_windows),
        two_dimensional_rectangle_count=len(attachments) * len(route_windows),
        descriptor_open_count=len(attachments),
        sequential_descriptor_pass_count=len(attachments),
        descriptor_readinto_call_count=readinto_calls,
        digest_update_count=digest_updates,
        payload_read_bytes=payload_read_bytes,
        trailing_probe_count=len(attachments),
        trailing_probe_bytes=trailing_probe_bytes,
        maximum_owned_host_staging_bytes=maximum_host_staging,
        device_route_window_staging_bytes=contexts * chunk_size * _FLOAT64_BYTES,
        row_bank_device_bytes=contexts * routes * _FLOAT64_BYTES,
        production_cuda_proposal_input_authority=production,
        cpu_reference_test_only=not production,
        private_cpu_reference_device_bank_materialized=not production,
        _issuance_token=_RECEIPT_ISSUANCE_TOKEN,
    )
    return V1SampledSignedWorkRowBankRestorationResult(
        receipt=receipt,
        row_bank_handle=consumer.private_device_handle,
        _issuance_token=_RESULT_ISSUANCE_TOKEN,
    )


__all__ = [
    "SAMPLED_SIGNED_WORK_ROW_BANK_RESTORATION_AUTHORITY_SCOPE",
    "SAMPLED_SIGNED_WORK_ROW_BANK_RESTORATION_MEMORY_SCOPE",
    "SAMPLED_SIGNED_WORK_ROW_BANK_RESTORATION_SCHEMA",
    "V1SampledSignedWorkRowBankRestorationReceipt",
    "V1SampledSignedWorkRowBankRestorationRefusal",
    "V1SampledSignedWorkRowBankRestorationRefusalCode",
    "V1SampledSignedWorkRowBankRestorationResult",
    "restore_v1_sampled_signed_work_row_bank_cuda",
]
