"""Same-capture scorer adapter for label-native F/J projection.

The canonical scorer exposes corrected Aumann--Shapley evidence and the
literal relaxed-gate Jacobian as two nominal route-chunk consumers.  This
module binds those callbacks to one shared transaction.  It accepts exactly
one corrected chunk followed by the matching literal-J chunk for every
physical route window, retains at most the one unmatched corrected tensor,
and immediately forwards each pair to
``V1LabelNativeProjectionCUDAAccumulator.consume_dual_chart_chunk``.

The shared coordinator is the proof boundary: both children must name the
same ordered context packet, canonical route-axis object, destination
transport, registered capture/VJP proof, and exact physical windows.  The
streams keep distinct identities and distinct storage.  A reorder, drop,
duplicate, asymmetric finalization, or identity drift aborts the accumulator
and closes the projection geometry.  No raw route row survives a successful
pair, and only compact projection results and receipts leave the adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from numbers import Integral
from pathlib import Path
from threading import RLock
from typing import Any, Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.measurement.optimized_irg_providers import (
    _RouteChunkedLossOnlyConsumer,
)

from .modern_v1_label_native_projection_cuda import (
    V1LabelNativeProjectionGeometry,
    V1LabelNativeProjectionResult,
)


LABEL_NATIVE_PROJECTION_SCORER_ADAPTER_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_label_native_projection_scorer_adapter_receipt_v2"
)
LABEL_NATIVE_PROJECTION_SCORER_ADAPTER_RESULT_SCHEMA: Final = (
    "partition_free_v1_label_native_projection_scorer_adapter_result_v2"
)
LABEL_NATIVE_PROJECTION_SCORER_CHILD_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_label_native_projection_scorer_child_receipt_v2"
)
LABEL_NATIVE_PROJECTION_SCORER_EXECUTION_CLOSURE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_label_native_projection_scorer_execution_closure_receipt_v1"
)
LABEL_NATIVE_PROJECTION_SCORER_EXECUTION_CLOSURE_SCHEMA: Final = (
    "partition_free_v1_label_native_projection_scorer_execution_closure_v1"
)
LABEL_NATIVE_PROJECTION_SCORER_PAIRING_ID: Final = (
    "corrected_then_literal_same_packet_same_window_distinct_storage_v1"
)
LABEL_NATIVE_PROJECTION_SCORER_AUTHORITY: Final = (
    "same_capture_dual_chart_transport_into_compact_label_native_projection_"
    "only_no_candidate_C_K_O_commit_terminal_or_pareto_authority_v1"
)

_RECEIPT_PREFIX: Final = "modern-v1-label-native-scorer-adapter-receipt-"
_EXECUTION_CLOSURE_PREFIX: Final = (
    "modern-v1-label-native-scorer-execution-closure-"
)
_PAIRING_PREFIX: Final = "modern-v1-label-native-scorer-adapter-pairing-"
_CANONICAL_SCORER_RECEIPT_PREFIX: Final = (
    "partition-free-v1-raw-target-logit-stream-"
)
_CANONICAL_SCORER_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_raw_target_logit_stream_receipt_v1"
)
_RAW_TARGET_LOGIT_COORDINATE: Final = "raw_target_logit"
_F64_BYTES: Final = 8
_CHILD_TOKEN: Final = object()


class V1LabelNativeProjectionScorerAdapterRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    CUDA_REQUIRED = "cuda_required"
    CPU_REFERENCE_LIMIT_EXCEEDED = "cpu_reference_limit_exceeded"
    STALE_GEOMETRY = "stale_geometry"
    CHUNK_BIND_ORDER = "chunk_bind_order"
    CHUNK_SIZE_MISMATCH = "chunk_size_mismatch"
    PACKET_ORDER = "packet_order"
    CONTEXT_ORDER = "context_order"
    ROUTE_AXIS_MISMATCH = "route_axis_mismatch"
    STREAM_IDENTITY_MISMATCH = "stream_identity_mismatch"
    CAPTURE_VJP_MISMATCH = "capture_vjp_mismatch"
    CHUNK_ORDER = "chunk_order"
    DUPLICATE_CHUNK = "duplicate_chunk"
    ROW_CONTRACT = "row_contract"
    COLLAPSED_STREAM = "collapsed_stream"
    PARTIAL_FINALIZATION = "partial_finalization"
    FINALIZE_ASYMMETRY = "finalize_asymmetry"
    PROJECTION_FAILED = "projection_failed"
    FAILED = "failed"
    CLOSED = "closed"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LabelNativeProjectionScorerAdapterRefusal(RuntimeError):
    def __init__(
        self,
        code: V1LabelNativeProjectionScorerAdapterRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LabelNativeProjectionScorerAdapterRefusalCode:
            raise TypeError("scorer-adapter refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LabelNativeProjectionScorerAdapterRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LabelNativeProjectionScorerAdapterRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one exact integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _identity_axis(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be one identity sequence")
    result = tuple(_identity(value, name=f"{name} item") for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


def label_native_projection_scorer_adapter_source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class V1LabelNativeProjectionScorerAdapterReceipt:
    geometry_receipt_hash: str
    provider_receipt_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    expected_context_ids: tuple[str, ...]
    observed_context_packets: tuple[tuple[str, ...], ...]
    chart_identity_hash: str
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    same_capture_and_vjp_registration_hash: str
    destination_transport_hash: str
    selected_endpoint_labels: tuple[int, ...]
    selected_endpoint_action_hashes: tuple[str, ...]
    route_chunk_size: int
    packet_projection_receipt_hashes: tuple[str, ...]
    packet_route_window_ledger_hashes: tuple[str, ...]
    packet_count: int
    row_pair_count: int
    row_pairing_chain_hash: str
    production_cuda_execution: bool
    exact_projection_device_to_host_bytes: int
    compact_projection_result_bytes: int
    adapter_source_hash: str
    maximum_pending_raw_route_row_objects: int = 1
    raw_route_rows_retained_after_pairing: bool = False
    host_context_route_row_transfer_bytes: int = 0
    full_route_labels_transferred_to_host: bool = False
    endpoint_hard_masks_materialized: bool = False
    route_by_endpoint_hard_masks_materialized: bool = False
    corrected_then_literal_pairing_proved: bool = True
    distinct_row_object_and_storage_proved: bool = True
    exact_physical_window_pairing_proved: bool = True
    ordered_context_coverage_proved: bool = True
    geometry_cleanup_complete: bool = True
    registration_only_projection: bool = True
    same_capture_and_vjp_execution_proof_available: bool = False
    C_projection_row_derivation_authority: bool = False
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    pairing_id: str = LABEL_NATIVE_PROJECTION_SCORER_PAIRING_ID
    authority_scope: str = LABEL_NATIVE_PROJECTION_SCORER_AUTHORITY
    schema: str = LABEL_NATIVE_PROJECTION_SCORER_ADAPTER_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "geometry_receipt_hash",
            "provider_receipt_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "chart_identity_hash",
            "corrected_evidence_stream_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "same_capture_and_vjp_registration_hash",
            "destination_transport_hash",
            "row_pairing_chain_hash",
            "adapter_source_hash",
            "pairing_id",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        route_count = _positive_int(self.route_count, name="route count")
        route_chunk_size = _positive_int(
            self.route_chunk_size,
            name="route chunk size",
        )
        packet_count = _positive_int(self.packet_count, name="packet count")
        pair_count = _positive_int(self.row_pair_count, name="row pair count")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "route_chunk_size", route_chunk_size)
        object.__setattr__(self, "packet_count", packet_count)
        object.__setattr__(self, "row_pair_count", pair_count)
        contexts = _identity_axis(
            self.expected_context_ids,
            name="expected scorer context axis",
        )
        packets = tuple(
            _identity_axis(packet, name="observed scorer context packet")
            for packet in self.observed_context_packets
        )
        flattened = tuple(item for packet in packets for item in packet)
        projection_hashes = tuple(
            _identity(value, name="packet projection receipt hash")
            for value in self.packet_projection_receipt_hashes
        )
        window_hashes = tuple(
            _identity(value, name="packet route-window ledger hash")
            for value in self.packet_route_window_ledger_hashes
        )
        labels = tuple(
            _positive_int(value, name="selected endpoint label")
            for value in self.selected_endpoint_labels
        )
        actions = tuple(
            _identity(value, name="selected endpoint action hash")
            for value in self.selected_endpoint_action_hashes
        )
        if (
            flattened != contexts
            or len(packets) != packet_count
            or len(projection_hashes) != packet_count
            or len(window_hashes) != packet_count
            or not labels
            or tuple(sorted(set(labels))) != labels
            or len(labels) != len(actions)
            or self.corrected_evidence_stream_identity_hash
            == self.literal_jacobian_stream_identity_hash
        ):
            raise ValueError("scorer-adapter receipt axes do not replay")
        object.__setattr__(self, "expected_context_ids", contexts)
        object.__setattr__(self, "observed_context_packets", packets)
        object.__setattr__(self, "packet_projection_receipt_hashes", projection_hashes)
        object.__setattr__(
            self,
            "packet_route_window_ledger_hashes",
            window_hashes,
        )
        object.__setattr__(self, "selected_endpoint_labels", labels)
        object.__setattr__(self, "selected_endpoint_action_hashes", actions)
        for name in (
            "exact_projection_device_to_host_bytes",
            "compact_projection_result_bytes",
            "maximum_pending_raw_route_row_objects",
            "host_context_route_row_transfer_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be one exact integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))
        expected_compact = len(contexts) * len(labels) * 6 * _F64_BYTES
        if (
            self.compact_projection_result_bytes != expected_compact
            or self.maximum_pending_raw_route_row_objects != 1
        ):
            raise ValueError("scorer-adapter compact-memory accounting does not replay")
        required_true = (
            self.corrected_then_literal_pairing_proved,
            self.distinct_row_object_and_storage_proved,
            self.exact_physical_window_pairing_proved,
            self.ordered_context_coverage_proved,
            self.geometry_cleanup_complete,
            self.registration_only_projection,
        )
        required_false = (
            self.raw_route_rows_retained_after_pairing,
            self.full_route_labels_transferred_to_host,
            self.endpoint_hard_masks_materialized,
            self.route_by_endpoint_hard_masks_materialized,
            self.same_capture_and_vjp_execution_proof_available,
            self.C_projection_row_derivation_authority,
            self.candidate_selection_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            type(self.production_cuda_execution) is not bool
            or any(type(value) is not bool for value in required_true + required_false)
            or not all(required_true)
            or any(required_false)
            or self.host_context_route_row_transfer_bytes != 0
            or self.pairing_id != LABEL_NATIVE_PROJECTION_SCORER_PAIRING_ID
            or self.authority_scope != LABEL_NATIVE_PROJECTION_SCORER_AUTHORITY
            or self.schema != LABEL_NATIVE_PROJECTION_SCORER_ADAPTER_RECEIPT_SCHEMA
        ):
            _refuse(
                V1LabelNativeProjectionScorerAdapterRefusalCode.AUTHORITY_VIOLATION,
                "scorer-adapter receipt crossed its pairing-only authority",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in field_iter(type(self))
            if not item.name.startswith("_")
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("label-native scorer-adapter receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("label-native scorer-adapter receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


def field_iter(dataclass_type: type[object]) -> tuple[Any, ...]:
    """Return dataclass fields without importing a mutable serialization layer."""

    from dataclasses import fields

    return tuple(fields(dataclass_type))


@dataclass(frozen=True, slots=True)
class V1LabelNativeProjectionScorerAdapterResult:
    receipt: V1LabelNativeProjectionScorerAdapterReceipt
    packet_projection_results: tuple[V1LabelNativeProjectionResult, ...] = field(
        repr=False,
        compare=False,
    )
    schema: str = LABEL_NATIVE_PROJECTION_SCORER_ADAPTER_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1LabelNativeProjectionScorerAdapterReceipt:
            raise TypeError("scorer-adapter result requires its exact receipt")
        self.receipt.assert_integrity()
        results = tuple(self.packet_projection_results)
        if (
            len(results) != self.receipt.packet_count
            or any(type(value) is not V1LabelNativeProjectionResult for value in results)
            or tuple(value.receipt.receipt_hash for value in results)
            != self.receipt.packet_projection_receipt_hashes
            or tuple(value.receipt.context_ids for value in results)
            != self.receipt.observed_context_packets
            or tuple(value.receipt.route_window_ledger_hash for value in results)
            != self.receipt.packet_route_window_ledger_hashes
            or any(
                value.receipt.geometry_receipt_hash
                != self.receipt.geometry_receipt_hash
                or value.receipt.chart_identity_hash != self.receipt.chart_identity_hash
                or value.receipt.corrected_evidence_stream_identity_hash
                != self.receipt.corrected_evidence_stream_identity_hash
                or value.receipt.literal_jacobian_stream_identity_hash
                != self.receipt.literal_jacobian_stream_identity_hash
                or value.receipt.same_capture_and_vjp_registration_hash
                != self.receipt.same_capture_and_vjp_registration_hash
                for value in results
            )
            or self.schema != LABEL_NATIVE_PROJECTION_SCORER_ADAPTER_RESULT_SCHEMA
        ):
            raise ValueError("scorer-adapter compact result does not replay")
        object.__setattr__(self, "packet_projection_results", results)


@dataclass(frozen=True, slots=True)
class V1LabelNativeProjectionScorerExecutionClosureReceipt:
    """Post-scorer proof that the provisional paired rows were co-produced.

    Projection receipts bind the pre-outcome registration only.  This receipt
    can exist only after the canonical scorer has closed its provider and raw
    packet receipts and derived the execution proof from that observed
    material.  It authorizes use of the compact rows as C measurements, never
    a candidate decision or any later K/O/commit transition.
    """

    adapter_receipt_hash: str
    scorer_receipt_hash: str
    scorer_execution_material_hash: str
    same_capture_and_vjp_registration_hash: str
    same_capture_and_vjp_execution_proof_hash: str
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    route_id_axis_hash: str
    expected_context_ids: tuple[str, ...]
    route_count: int
    route_chunk_size: int
    packet_count: int
    qualified_lane_registration_hash: str
    selected_compact_provider_endpoint_indices: tuple[int, ...]
    selected_source_endpoint_labels: tuple[int, ...]
    selected_endpoint_counts: tuple[int, ...]
    selected_endpoint_action_hashes: tuple[str, ...]
    bound_child_stream_roles: tuple[str, ...]
    one_physical_scorer_invocation: bool = True
    same_capture_and_vjp_execution_closed: bool = True
    registration_not_used_as_execution_proof: bool = True
    C_projection_row_derivation_authority: bool = True
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = LABEL_NATIVE_PROJECTION_SCORER_AUTHORITY
    schema: str = LABEL_NATIVE_PROJECTION_SCORER_EXECUTION_CLOSURE_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "adapter_receipt_hash",
            "scorer_receipt_hash",
            "scorer_execution_material_hash",
            "same_capture_and_vjp_registration_hash",
            "same_capture_and_vjp_execution_proof_hash",
            "corrected_evidence_stream_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "route_id_axis_hash",
            "qualified_lane_registration_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        contexts = _identity_axis(
            self.expected_context_ids,
            name="post-scorer closure context axis",
        )
        compact_indices = tuple(
            _positive_int(value, name="compact provider endpoint index")
            for value in self.selected_compact_provider_endpoint_indices
        )
        source_labels = tuple(
            _positive_int(value, name="source endpoint label")
            for value in self.selected_source_endpoint_labels
        )
        counts = tuple(
            _positive_int(value, name="endpoint count")
            for value in self.selected_endpoint_counts
        )
        actions = tuple(
            _identity(value, name="endpoint action hash")
            for value in self.selected_endpoint_action_hashes
        )
        roles = tuple(
            _identity(value, name="bound scorer-child role")
            for value in self.bound_child_stream_roles
        )
        if (
            compact_indices != tuple(range(1, len(source_labels) + 1))
            or tuple(sorted(set(source_labels))) != source_labels
            or tuple(sorted(counts)) != counts
            or not source_labels
            or len(source_labels) != len(counts)
            or len(source_labels) != len(actions)
            or roles != (_StreamRole.CORRECTED.value, _StreamRole.JACOBIAN.value)
            or self.same_capture_and_vjp_registration_hash
            == self.same_capture_and_vjp_execution_proof_hash
            or self.corrected_evidence_stream_identity_hash
            == self.literal_jacobian_stream_identity_hash
        ):
            raise ValueError("post-scorer closure axes do not replay")
        object.__setattr__(self, "expected_context_ids", contexts)
        object.__setattr__(
            self,
            "selected_compact_provider_endpoint_indices",
            compact_indices,
        )
        object.__setattr__(self, "selected_source_endpoint_labels", source_labels)
        object.__setattr__(self, "selected_endpoint_counts", counts)
        object.__setattr__(self, "selected_endpoint_action_hashes", actions)
        object.__setattr__(self, "bound_child_stream_roles", roles)
        for name in ("route_count", "route_chunk_size", "packet_count"):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        required_true = (
            self.one_physical_scorer_invocation,
            self.same_capture_and_vjp_execution_closed,
            self.registration_not_used_as_execution_proof,
            self.C_projection_row_derivation_authority,
        )
        required_false = (
            self.candidate_selection_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool for value in required_true + required_false)
            or not all(required_true)
            or any(required_false)
            or self.authority_scope != LABEL_NATIVE_PROJECTION_SCORER_AUTHORITY
            or self.schema
            != LABEL_NATIVE_PROJECTION_SCORER_EXECUTION_CLOSURE_RECEIPT_SCHEMA
        ):
            _refuse(
                V1LabelNativeProjectionScorerAdapterRefusalCode.AUTHORITY_VIOLATION,
                "post-scorer closure crossed C-row measurement authority",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in field_iter(type(self))
            if not item.name.startswith("_")
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_EXECUTION_CLOSURE_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("label-native scorer execution closure was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("label-native scorer execution closure hash differs")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LabelNativeProjectionScorerExecutionClosure:
    receipt: V1LabelNativeProjectionScorerExecutionClosureReceipt
    projection_result: V1LabelNativeProjectionScorerAdapterResult = field(
        repr=False,
        compare=False,
    )
    schema: str = LABEL_NATIVE_PROJECTION_SCORER_EXECUTION_CLOSURE_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.receipt)
            is not V1LabelNativeProjectionScorerExecutionClosureReceipt
            or type(self.projection_result)
            is not V1LabelNativeProjectionScorerAdapterResult
            or self.receipt.adapter_receipt_hash
            != self.projection_result.receipt.receipt_hash
            or self.schema != LABEL_NATIVE_PROJECTION_SCORER_EXECUTION_CLOSURE_SCHEMA
        ):
            raise ValueError("post-scorer execution closure does not replay")
        self.receipt.assert_integrity()
        self.projection_result.receipt.assert_integrity()


def _bound_adapter_children(
    value: object,
    *,
    adapter_receipt_hash: str,
) -> tuple[Mapping[str, object], ...]:
    matches: dict[tuple[str, str], Mapping[str, object]] = {}
    visited: set[int] = set()

    def visit(current: object) -> None:
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in visited:
                return
            visited.add(marker)
            if (
                current.get("schema") == LABEL_NATIVE_PROJECTION_SCORER_CHILD_RECEIPT_SCHEMA
                and current.get("adapter_receipt_hash") == adapter_receipt_hash
            ):
                role = str(current.get("stream_role", ""))
                stream = str(current.get("stream_identity_hash", ""))
                matches[(role, stream)] = current
            for child in current.values():
                visit(child)
            return
        if isinstance(current, (tuple, list)):
            marker = id(current)
            if marker in visited:
                return
            visited.add(marker)
            for child in current:
                visit(child)

    visit(value)
    return tuple(matches[key] for key in sorted(matches))


def close_v1_label_native_projection_scorer_execution(
    *,
    projection_result: V1LabelNativeProjectionScorerAdapterResult,
    scorer_receipt: Mapping[str, object],
    expected_same_capture_and_vjp_execution_proof_hash: str,
) -> V1LabelNativeProjectionScorerExecutionClosure:
    """Close actual scorer evidence after provider/raw-packet validation.

    The execution proof must be the proof carried by the content-addressed
    canonical scorer receipt and must be the SHA-256 image of its registered
    execution-material hash.  Supplying the pre-run registration token in its
    place therefore fails closed.
    """

    if type(projection_result) is not V1LabelNativeProjectionScorerAdapterResult:
        raise TypeError("post-scorer closure requires the exact adapter result")
    projection_result.receipt.assert_integrity()
    if not isinstance(scorer_receipt, Mapping):
        raise TypeError("post-scorer closure requires one scorer receipt mapping")
    receipt_map = dict(scorer_receipt)
    scorer_hash = _identity(
        receipt_map.get("receipt_hash"),
        name="canonical scorer receipt hash",
    )
    scorer_payload = dict(receipt_map)
    scorer_payload.pop("receipt_hash", None)
    if (
        receipt_map.get("schema") != _CANONICAL_SCORER_RECEIPT_SCHEMA
        or content_hash(
            scorer_payload,
            prefix=_CANONICAL_SCORER_RECEIPT_PREFIX,
        )
        != scorer_hash
    ):
        _refuse(
            V1LabelNativeProjectionScorerAdapterRefusalCode.CAPTURE_VJP_MISMATCH,
            "canonical scorer receipt integrity did not close",
        )
    adapter = projection_result.receipt
    registration = _identity(
        receipt_map.get("same_capture_and_vjp_registration_hash"),
        name="scorer same-capture/VJP registration",
    )
    material = _identity(
        receipt_map.get("same_capture_and_vjp_execution_material_hash"),
        name="scorer execution-material hash",
    )
    proof = _identity(
        receipt_map.get("same_capture_and_vjp_execution_proof_hash"),
        name="scorer execution-proof hash",
    )
    expected_proof = _identity(
        expected_same_capture_and_vjp_execution_proof_hash,
        name="expected scorer execution-proof hash",
    )
    child_receipts = _bound_adapter_children(
        receipt_map.get("child_streaming_receipts", ()),
        adapter_receipt_hash=adapter.receipt_hash,
    )
    child_by_role = {
        str(value.get("stream_role", "")): value for value in child_receipts
    }
    expected_child_roles = (
        _StreamRole.CORRECTED.value,
        _StreamRole.JACOBIAN.value,
    )
    child_invalid = (
        tuple(role for role in expected_child_roles if role in child_by_role)
        != expected_child_roles
        or len(child_by_role) != 2
        or any(
            child_by_role[role].get("same_capture_and_vjp_registration_hash")
            != adapter.same_capture_and_vjp_registration_hash
            or child_by_role[role].get(
                "same_capture_and_vjp_execution_proof_available"
            )
            is not False
            or child_by_role[role].get("registration_only_projection") is not True
            or child_by_role[role].get("C_projection_row_derivation_authority")
            is not False
            for role in expected_child_roles
        )
    )
    if (
        registration != adapter.same_capture_and_vjp_registration_hash
        or proof != expected_proof
        or proof != hashlib.sha256(material.encode("utf-8")).hexdigest()
        or proof == registration
        or receipt_map.get("corrected_evidence_stream_identity_hash")
        != adapter.corrected_evidence_stream_identity_hash
        or receipt_map.get("literal_jacobian_stream_identity_hash")
        != adapter.literal_jacobian_stream_identity_hash
        or int(receipt_map.get("context_count", -1))
        != len(adapter.expected_context_ids)
        or int(receipt_map.get("context_packet_count", -1))
        != adapter.packet_count
        or int(receipt_map.get("route_count", -1)) != adapter.route_count
        or int(receipt_map.get("capture_batch_count", -1)) != adapter.packet_count
        or int(receipt_map.get("vjp_count", -1)) != adapter.packet_count
        or receipt_map.get("one_physical_scorer_invocation") is not True
        or receipt_map.get("dual_chart_streaming") is not True
        or receipt_map.get("same_capture_and_vjp_required") is not True
        or receipt_map.get("shared_positive_removal_rows") is not False
        or receipt_map.get("host_context_route_rows_transferred") is not False
        or int(receipt_map.get("host_context_route_row_transfer_count", -1)) != 0
        or int(receipt_map.get("host_context_route_row_transfer_bytes", -1)) != 0
        or child_invalid
    ):
        _refuse(
            V1LabelNativeProjectionScorerAdapterRefusalCode.CAPTURE_VJP_MISMATCH,
            "post-scorer receipt does not prove this registered dual-chart execution",
        )
    packet_receipts = projection_result.packet_projection_results
    first = packet_receipts[0].receipt
    if any(
        value.receipt.qualified_lane_registration_hash
        != first.qualified_lane_registration_hash
        or value.receipt.selected_compact_provider_endpoint_indices
        != first.selected_compact_provider_endpoint_indices
        or value.receipt.selected_source_endpoint_labels
        != first.selected_source_endpoint_labels
        or value.receipt.selected_endpoint_counts != first.selected_endpoint_counts
        or value.receipt.selected_endpoint_action_hashes
        != first.selected_endpoint_action_hashes
        for value in packet_receipts
    ):
        _refuse(
            V1LabelNativeProjectionScorerAdapterRefusalCode.CAPTURE_VJP_MISMATCH,
            "post-scorer projection packets changed the qualified endpoint axis",
        )
    closure_receipt = V1LabelNativeProjectionScorerExecutionClosureReceipt(
        adapter_receipt_hash=adapter.receipt_hash,
        scorer_receipt_hash=scorer_hash,
        scorer_execution_material_hash=material,
        same_capture_and_vjp_registration_hash=registration,
        same_capture_and_vjp_execution_proof_hash=proof,
        corrected_evidence_stream_identity_hash=(
            adapter.corrected_evidence_stream_identity_hash
        ),
        literal_jacobian_stream_identity_hash=(
            adapter.literal_jacobian_stream_identity_hash
        ),
        route_id_axis_hash=adapter.route_id_axis_hash,
        expected_context_ids=adapter.expected_context_ids,
        route_count=adapter.route_count,
        route_chunk_size=adapter.route_chunk_size,
        packet_count=adapter.packet_count,
        qualified_lane_registration_hash=first.qualified_lane_registration_hash,
        selected_compact_provider_endpoint_indices=(
            first.selected_compact_provider_endpoint_indices
        ),
        selected_source_endpoint_labels=first.selected_source_endpoint_labels,
        selected_endpoint_counts=first.selected_endpoint_counts,
        selected_endpoint_action_hashes=first.selected_endpoint_action_hashes,
        bound_child_stream_roles=expected_child_roles,
    )
    return V1LabelNativeProjectionScorerExecutionClosure(
        receipt=closure_receipt,
        projection_result=projection_result,
    )


class _AdapterState(str, Enum):
    OPEN = "open"
    SEALED = "sealed"
    FAILED = "failed"
    CLOSED = "closed"


class _StreamRole(str, Enum):
    CORRECTED = "corrected_evidence"
    JACOBIAN = "literal_jacobian"


class V1LabelNativeProjectionScorerAdapter:
    """Shared coordinator behind the two nominal scorer consumers."""

    __slots__ = (
        "_active_accumulator",
        "_begun_roles",
        "_bound_roles",
        "_chart_identity_hash",
        "_context_offset",
        "_corrected_ended",
        "_corrected_stream_hash",
        "_destination_transport_hash",
        "_expected_context_ids",
        "_expected_next_chunk_role",
        "_geometry",
        "_jacobian_stream_hash",
        "_lock",
        "_next_route",
        "_observed_context_packets",
        "_packet_context_ids",
        "_packet_results",
        "_pair_count",
        "_pairing_chain_hash",
        "_pending_corrected_rows",
        "_pending_window",
        "_route_chunk_size",
        "_route_id_axis_hash",
        "_route_ids",
        "_same_capture_hash",
        "_sealed_result",
        "_state",
        "corrected_evidence_consumer",
        "literal_jacobian_consumer",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        geometry: V1LabelNativeProjectionGeometry,
        route_ids: Sequence[str],
        route_id_axis_hash: str,
        expected_context_ids: Sequence[str],
        chart_identity_hash: str,
        corrected_evidence_stream_identity_hash: str,
        literal_jacobian_stream_identity_hash: str,
        same_capture_and_vjp_registration_hash: str,
        destination_transport_hash: str,
        production: bool,
    ) -> None:
        if type(geometry) is not V1LabelNativeProjectionGeometry:
            raise TypeError("scorer adapter requires the exact projection geometry")
        if type(production) is not bool:
            raise TypeError("scorer-adapter production flag must be exact")
        geometry._require_live()
        if geometry.has_active_accumulator:
            _refuse(
                V1LabelNativeProjectionScorerAdapterRefusalCode.STALE_GEOMETRY,
                "projection geometry already owns an accumulator",
            )
        if isinstance(route_ids, (str, bytes)) or len(route_ids) != geometry.receipt.route_count:
            _refuse(
                V1LabelNativeProjectionScorerAdapterRefusalCode.ROUTE_AXIS_MISMATCH,
                "scorer route IDs differ from the projection route count",
            )
        backend = geometry.receipt.backend_kind
        if production and backend != "torch_cuda":
            _refuse(
                V1LabelNativeProjectionScorerAdapterRefusalCode.CUDA_REQUIRED,
                "production scorer pairing requires CUDA projection geometry",
            )
        if not production and (
            backend != "numpy_reference"
            or geometry.receipt.route_count > geometry.design.private_cpu_route_limit
        ):
            _refuse(
                V1LabelNativeProjectionScorerAdapterRefusalCode.CPU_REFERENCE_LIMIT_EXCEEDED,
                "private CPU scorer pairing exceeds its registered reference boundary",
            )
        corrected = _identity(
            corrected_evidence_stream_identity_hash,
            name="corrected-evidence stream identity",
        )
        jacobian = _identity(
            literal_jacobian_stream_identity_hash,
            name="literal-J stream identity",
        )
        if corrected == jacobian:
            raise ValueError("corrected-evidence and literal-J identities must differ")
        self._geometry = geometry
        self._route_ids = route_ids
        self._route_id_axis_hash = _identity(
            route_id_axis_hash,
            name="route-ID axis hash",
        )
        self._expected_context_ids = _identity_axis(
            expected_context_ids,
            name="expected scorer context axis",
        )
        self._chart_identity_hash = _identity(
            chart_identity_hash,
            name="chart identity",
        )
        self._corrected_stream_hash = corrected
        self._jacobian_stream_hash = jacobian
        self._same_capture_hash = _identity(
            same_capture_and_vjp_registration_hash,
            name="same-capture/VJP registration",
        )
        self._destination_transport_hash = _identity(
            destination_transport_hash,
            name="destination transport hash",
        )
        self._state = _AdapterState.OPEN
        self._lock = RLock()
        self._bound_roles: list[_StreamRole] = []
        self._begun_roles: list[_StreamRole] = []
        self._route_chunk_size: int | None = None
        self._context_offset = 0
        self._packet_context_ids: tuple[str, ...] | None = None
        self._next_route = 0
        self._expected_next_chunk_role = _StreamRole.CORRECTED
        self._pending_corrected_rows: Any | None = None
        self._pending_window: tuple[int, int] | None = None
        self._corrected_ended = False
        self._active_accumulator: Any | None = None
        self._packet_results: list[V1LabelNativeProjectionResult] = []
        self._observed_context_packets: list[tuple[str, ...]] = []
        self._pair_count = 0
        self._pairing_chain_hash = content_hash(
            {
                "schema": "label_native_scorer_pairing_chain_seed_v1",
                "geometry_receipt_hash": geometry.receipt.receipt_hash,
                "route_id_axis_hash": self._route_id_axis_hash,
                "expected_context_ids": self._expected_context_ids,
                "corrected_stream_identity_hash": corrected,
                "literal_jacobian_stream_identity_hash": jacobian,
                "same_capture_and_vjp_registration_hash": self._same_capture_hash,
                "destination_transport_hash": self._destination_transport_hash,
            },
            prefix=_PAIRING_PREFIX + "seed-",
        )
        self._sealed_result: V1LabelNativeProjectionScorerAdapterResult | None = None
        self.corrected_evidence_consumer = _V1LabelNativeProjectionScorerChild(
            _construction_token=_CHILD_TOKEN,
            owner=self,
            role=_StreamRole.CORRECTED,
        )
        self.literal_jacobian_consumer = _V1LabelNativeProjectionScorerChild(
            _construction_token=_CHILD_TOKEN,
            owner=self,
            role=_StreamRole.JACOBIAN,
        )

    @classmethod
    def from_projection_geometry(
        cls,
        **kwargs: object,
    ) -> "V1LabelNativeProjectionScorerAdapter":
        return cls(**kwargs, production=True)  # type: ignore[arg-type]

    @classmethod
    def _from_projection_geometry_reference_for_tests(
        cls,
        **kwargs: object,
    ) -> "V1LabelNativeProjectionScorerAdapter":
        return cls(**kwargs, production=False)  # type: ignore[arg-type]

    @property
    def is_closed(self) -> bool:
        return self._state in {
            _AdapterState.SEALED,
            _AdapterState.FAILED,
            _AdapterState.CLOSED,
        }

    @property
    def is_failed(self) -> bool:
        return self._state is _AdapterState.FAILED

    @property
    def maximum_pending_raw_route_row_objects(self) -> int:
        return 1

    def _require_open(self) -> None:
        if self._state is _AdapterState.FAILED:
            _refuse(
                V1LabelNativeProjectionScorerAdapterRefusalCode.FAILED,
                "scorer adapter is failed and cannot be resumed",
            )
        if self._state is not _AdapterState.OPEN:
            _refuse(
                V1LabelNativeProjectionScorerAdapterRefusalCode.CLOSED,
                "scorer adapter is already sealed or closed",
            )
        self._geometry._require_live()

    def _cleanup_live_state(self) -> None:
        self._pending_corrected_rows = None
        self._pending_window = None
        accumulator = self._active_accumulator
        self._active_accumulator = None
        if accumulator is not None:
            try:
                accumulator.close()
            except BaseException:
                pass
        if not self._geometry.is_closed:
            try:
                self._geometry.close()
            except BaseException:
                pass

    def _abort(
        self,
        code: V1LabelNativeProjectionScorerAdapterRefusalCode,
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> NoReturn:
        self._cleanup_live_state()
        self._state = _AdapterState.FAILED
        error = V1LabelNativeProjectionScorerAdapterRefusal(code, message)
        if cause is None:
            raise error
        raise error from cause

    def _bind_route_chunk_size(self, role: _StreamRole, chunk_size: int) -> None:
        with self._lock:
            self._require_open()
            size = _positive_int(chunk_size, name="scorer route chunk size")
            expected = (
                _StreamRole.CORRECTED
                if not self._bound_roles
                else _StreamRole.JACOBIAN
                if self._bound_roles == [_StreamRole.CORRECTED]
                else None
            )
            if role is not expected:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.CHUNK_BIND_ORDER,
                    "dual-chart chunk geometry was rebound, duplicated, or reordered",
                )
            if size > self._geometry.design.route_chunk_size:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.CHUNK_SIZE_MISMATCH,
                    "scorer route chunks exceed the projection workspace design",
                )
            if self._route_chunk_size is None:
                self._route_chunk_size = size
            elif size != self._route_chunk_size:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.CHUNK_SIZE_MISMATCH,
                    "corrected and literal-J consumers bound different chunk sizes",
                )
            self._bound_roles.append(role)

    def _validate_packet_identity(
        self,
        *,
        role: _StreamRole,
        context_ids: Sequence[str],
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> tuple[str, ...]:
        if self._bound_roles != [_StreamRole.CORRECTED, _StreamRole.JACOBIAN]:
            self._abort(
                V1LabelNativeProjectionScorerAdapterRefusalCode.CHUNK_BIND_ORDER,
                "both scorer children must bind one chunk geometry before packets",
            )
        if route_ids is not self._route_ids:
            self._abort(
                V1LabelNativeProjectionScorerAdapterRefusalCode.ROUTE_AXIS_MISMATCH,
                "packet route IDs are not the registered canonical axis object",
            )
        expected_stream = (
            self._corrected_stream_hash
            if role is _StreamRole.CORRECTED
            else self._jacobian_stream_hash
        )
        if loss_pullback_hash != expected_stream:
            self._abort(
                V1LabelNativeProjectionScorerAdapterRefusalCode.STREAM_IDENTITY_MISMATCH,
                "scorer child named the other or an unregistered stream identity",
            )
        if destination_transport_hash != self._destination_transport_hash:
            self._abort(
                V1LabelNativeProjectionScorerAdapterRefusalCode.CAPTURE_VJP_MISMATCH,
                "dual-chart children name different capture transports",
            )
        try:
            contexts = _identity_axis(context_ids, name="scorer context packet")
        except (TypeError, ValueError) as error:
            self._abort(
                V1LabelNativeProjectionScorerAdapterRefusalCode.CONTEXT_ORDER,
                "scorer context packet is invalid",
                cause=error,
            )
        if role is _StreamRole.CORRECTED:
            expected = self._expected_context_ids[
                self._context_offset : self._context_offset + len(contexts)
            ]
            if contexts != expected:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.CONTEXT_ORDER,
                    "corrected stream reordered, skipped, or duplicated contexts",
                )
        elif contexts != self._packet_context_ids:
            self._abort(
                V1LabelNativeProjectionScorerAdapterRefusalCode.CAPTURE_VJP_MISMATCH,
                "dual-chart children name different context packets",
            )
        return contexts

    def _begin_packet(
        self,
        role: _StreamRole,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        with self._lock:
            self._require_open()
            expected_role = (
                _StreamRole.CORRECTED
                if not self._begun_roles
                else _StreamRole.JACOBIAN
                if self._begun_roles == [_StreamRole.CORRECTED]
                else None
            )
            if role is not expected_role:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.PACKET_ORDER,
                    "dual-chart packet begin was reordered, duplicated, or nested",
                )
            contexts = self._validate_packet_identity(
                role=role,
                context_ids=context_ids,
                route_ids=route_ids,
                destination_transport_hash=destination_transport_hash,
                loss_pullback_hash=loss_pullback_hash,
            )
            if role is _StreamRole.CORRECTED:
                self._packet_context_ids = contexts
                self._begun_roles.append(role)
                return
            try:
                self._active_accumulator = self._geometry.open_accumulator(
                    context_ids=contexts,
                    chart_identity_hash=self._chart_identity_hash,
                    corrected_evidence_stream_identity_hash=self._corrected_stream_hash,
                    literal_jacobian_stream_identity_hash=self._jacobian_stream_hash,
                    same_capture_and_vjp_registration_hash=self._same_capture_hash,
                )
            except BaseException as error:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.PROJECTION_FAILED,
                    "projection accumulator refused the registered scorer packet",
                    cause=error,
                )
            self._begun_roles.append(role)
            self._next_route = 0
            self._expected_next_chunk_role = _StreamRole.CORRECTED
            self._corrected_ended = False

    def _validate_rows(
        self,
        rows: Any,
        *,
        span: int,
    ) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - environment dependent.
            self._abort(
                V1LabelNativeProjectionScorerAdapterRefusalCode.ROW_CONTRACT,
                "PyTorch is required for scorer route rows",
                cause=error,
            )
        packet = self._packet_context_ids
        if (
            packet is None
            or not isinstance(rows, torch.Tensor)
            or tuple(rows.shape) != (len(packet), span)
            or rows.dtype is not torch.float64
            or rows.device != self._geometry._device
            or not rows.is_contiguous()
        ):
            self._abort(
                V1LabelNativeProjectionScorerAdapterRefusalCode.ROW_CONTRACT,
                "scorer rows must be contiguous FP64 [context, route] on the geometry device",
            )

    def _consume_chunk(
        self,
        role: _StreamRole,
        start: int,
        stop: int,
        rows: Any,
    ) -> None:
        with self._lock:
            self._require_open()
            if self._begun_roles != [_StreamRole.CORRECTED, _StreamRole.JACOBIAN]:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.PACKET_ORDER,
                    "route rows arrived outside one fully opened dual-chart packet",
                )
            if self._corrected_ended:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.FINALIZE_ASYMMETRY,
                    "route rows arrived after corrected-stream finalization",
                )
            if role is not self._expected_next_chunk_role:
                code = (
                    V1LabelNativeProjectionScorerAdapterRefusalCode.DUPLICATE_CHUNK
                    if role is _StreamRole.CORRECTED
                    and self._pending_corrected_rows is not None
                    else V1LabelNativeProjectionScorerAdapterRefusalCode.CHUNK_ORDER
                )
                self._abort(
                    code,
                    "corrected and literal-J route chunks were duplicated or reordered",
                )
            try:
                begin = int(start)
                end = int(stop)
            except (TypeError, ValueError) as error:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.CHUNK_ORDER,
                    "route-window coordinates are not exact integers",
                    cause=error,
                )
            if isinstance(start, bool) or isinstance(stop, bool):
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.CHUNK_ORDER,
                    "route-window booleans are forbidden",
                )
            span = end - begin
            assert self._route_chunk_size is not None
            if role is _StreamRole.CORRECTED:
                if (
                    begin != self._next_route
                    or span <= 0
                    or span > self._route_chunk_size
                    or end > self._geometry.receipt.route_count
                ):
                    self._abort(
                        V1LabelNativeProjectionScorerAdapterRefusalCode.CHUNK_ORDER,
                        "corrected chunks must cover the physical route axis in order",
                    )
                self._validate_rows(rows, span=span)
                self._pending_corrected_rows = rows
                self._pending_window = (begin, end)
                self._expected_next_chunk_role = _StreamRole.JACOBIAN
                return
            if self._pending_window != (begin, end):
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.CHUNK_ORDER,
                    "literal-J chunk does not match the pending corrected window",
                )
            self._validate_rows(rows, span=span)
            corrected_rows = self._pending_corrected_rows
            if corrected_rows is None:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.CHUNK_ORDER,
                    "literal-J chunk has no pending corrected partner",
                )
            same_object = rows is corrected_rows
            try:
                same_storage = (
                    rows.untyped_storage().data_ptr()
                    == corrected_rows.untyped_storage().data_ptr()
                )
            except (AttributeError, RuntimeError) as error:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.ROW_CONTRACT,
                    "scorer rows do not expose stable tensor storage identity",
                    cause=error,
                )
            if same_object or same_storage:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.COLLAPSED_STREAM,
                    "corrected evidence and literal-J reused one row object or storage",
                )
            accumulator = self._active_accumulator
            if accumulator is None:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.PACKET_ORDER,
                    "paired rows have no active projection accumulator",
                )
            try:
                accumulator.consume_dual_chart_chunk(
                    begin,
                    end,
                    corrected_evidence_rows=corrected_rows,
                    literal_jacobian_rows=rows,
                )
            except BaseException as error:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.PROJECTION_FAILED,
                    "paired route projection failed",
                    cause=error,
                )
            packet_index = len(self._packet_results)
            self._pair_count += 1
            self._pairing_chain_hash = content_hash(
                {
                    "schema": "label_native_scorer_pairing_chain_link_v1",
                    "prior_chain_hash": self._pairing_chain_hash,
                    "pair_index": self._pair_count,
                    "packet_index": packet_index,
                    "context_ids": self._packet_context_ids,
                    "route_window": (begin, end),
                    "corrected_then_literal": True,
                    "simultaneously_live_distinct_objects": True,
                    "distinct_tensor_storage": True,
                    "same_capture_and_vjp_registration_hash": self._same_capture_hash,
                },
                prefix=_PAIRING_PREFIX + "link-",
            )
            self._pending_corrected_rows = None
            self._pending_window = None
            self._next_route = end
            self._expected_next_chunk_role = _StreamRole.CORRECTED

    def _end_packet(self, role: _StreamRole) -> None:
        with self._lock:
            self._require_open()
            if self._begun_roles != [_StreamRole.CORRECTED, _StreamRole.JACOBIAN]:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.FINALIZE_ASYMMETRY,
                    "dual-chart packet finalization began before both children",
                )
            if role is _StreamRole.CORRECTED:
                if self._corrected_ended:
                    self._abort(
                        V1LabelNativeProjectionScorerAdapterRefusalCode.FINALIZE_ASYMMETRY,
                        "corrected stream finalized twice",
                    )
                if (
                    self._pending_corrected_rows is not None
                    or self._expected_next_chunk_role is not _StreamRole.CORRECTED
                    or self._next_route != self._geometry.receipt.route_count
                ):
                    self._abort(
                        V1LabelNativeProjectionScorerAdapterRefusalCode.PARTIAL_FINALIZATION,
                        "corrected stream finalized before every paired route window",
                    )
                self._corrected_ended = True
                return
            if not self._corrected_ended:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.FINALIZE_ASYMMETRY,
                    "literal-J stream finalized before corrected evidence",
                )
            accumulator = self._active_accumulator
            if accumulator is None:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.FINALIZE_ASYMMETRY,
                    "literal-J finalization has no paired projection accumulator",
                )
            try:
                result = accumulator.finalize()
            except BaseException as error:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.PROJECTION_FAILED,
                    "projection refused a nominally complete scorer packet",
                    cause=error,
                )
            contexts = self._packet_context_ids
            assert contexts is not None
            self._active_accumulator = None
            self._packet_results.append(result)
            self._observed_context_packets.append(contexts)
            self._context_offset += len(contexts)
            self._packet_context_ids = None
            self._begun_roles.clear()
            self._next_route = 0
            self._corrected_ended = False
            if self._context_offset == len(self._expected_context_ids):
                self._seal()

    def _seal(self) -> None:
        if (
            self._active_accumulator is not None
            or self._pending_corrected_rows is not None
            or self._begun_roles
            or self._context_offset != len(self._expected_context_ids)
            or not self._packet_results
        ):
            self._abort(
                V1LabelNativeProjectionScorerAdapterRefusalCode.PARTIAL_FINALIZATION,
                "scorer adapter cannot seal partial context or route coverage",
            )
        try:
            self._geometry.close()
        except BaseException as error:
            self._abort(
                V1LabelNativeProjectionScorerAdapterRefusalCode.PROJECTION_FAILED,
                "projection geometry cleanup failed after scorer completion",
                cause=error,
            )
        packet_results = tuple(self._packet_results)
        projection_hashes = tuple(
            value.receipt.receipt_hash for value in packet_results
        )
        window_hashes = tuple(
            value.receipt.route_window_ledger_hash for value in packet_results
        )
        exact_d2h = sum(
            value.receipt.exact_device_to_host_bytes for value in packet_results
        )
        receipt = V1LabelNativeProjectionScorerAdapterReceipt(
            geometry_receipt_hash=self._geometry.receipt.receipt_hash,
            provider_receipt_hash=self._geometry.receipt.provider_receipt_hash,
            route_axis_hash=self._geometry.receipt.route_axis_hash,
            route_id_axis_hash=self._route_id_axis_hash,
            route_count=self._geometry.receipt.route_count,
            expected_context_ids=self._expected_context_ids,
            observed_context_packets=tuple(self._observed_context_packets),
            chart_identity_hash=self._chart_identity_hash,
            corrected_evidence_stream_identity_hash=self._corrected_stream_hash,
            literal_jacobian_stream_identity_hash=self._jacobian_stream_hash,
            same_capture_and_vjp_registration_hash=self._same_capture_hash,
            destination_transport_hash=self._destination_transport_hash,
            selected_endpoint_labels=self._geometry.receipt.selected_endpoint_labels,
            selected_endpoint_action_hashes=(
                self._geometry.receipt.selected_endpoint_action_hashes
            ),
            route_chunk_size=self._route_chunk_size or 0,
            packet_projection_receipt_hashes=projection_hashes,
            packet_route_window_ledger_hashes=window_hashes,
            packet_count=len(packet_results),
            row_pair_count=self._pair_count,
            row_pairing_chain_hash=self._pairing_chain_hash,
            production_cuda_execution=(
                self._geometry.receipt.backend_kind == "torch_cuda"
            ),
            exact_projection_device_to_host_bytes=exact_d2h,
            compact_projection_result_bytes=(
                len(self._expected_context_ids)
                * len(self._geometry.receipt.selected_endpoint_labels)
                * 6
                * _F64_BYTES
            ),
            adapter_source_hash=label_native_projection_scorer_adapter_source_hash(),
        )
        self._sealed_result = V1LabelNativeProjectionScorerAdapterResult(
            receipt=receipt,
            packet_projection_results=packet_results,
        )
        self._state = _AdapterState.SEALED

    def finalize(self) -> V1LabelNativeProjectionScorerAdapterResult:
        with self._lock:
            if self._state is _AdapterState.SEALED:
                assert self._sealed_result is not None
                return self._sealed_result
            if self._state is _AdapterState.FAILED:
                _refuse(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.FAILED,
                    "failed scorer adapter has no result",
                )
            if self._state is _AdapterState.CLOSED:
                _refuse(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.CLOSED,
                    "closed scorer adapter has no result",
                )
            self._abort(
                V1LabelNativeProjectionScorerAdapterRefusalCode.PARTIAL_FINALIZATION,
                "scorer adapter cannot expose a result before complete dual-chart coverage",
            )

    def _child_receipt(self, role: _StreamRole) -> Mapping[str, object]:
        with self._lock:
            result = self.finalize()
            receipt = result.receipt
            stream_hash = (
                receipt.corrected_evidence_stream_identity_hash
                if role is _StreamRole.CORRECTED
                else receipt.literal_jacobian_stream_identity_hash
            )
            payload: dict[str, object] = {
                "schema": LABEL_NATIVE_PROJECTION_SCORER_CHILD_RECEIPT_SCHEMA,
                "adapter_receipt_hash": receipt.receipt_hash,
                "stream_role": role.value,
                "stream_identity_hash": stream_hash,
                "route_chunked_streaming": True,
                "route_chunk_size": receipt.route_chunk_size,
                "route_count": receipt.route_count,
                "context_count": len(receipt.expected_context_ids),
                "packet_count": receipt.packet_count,
                "route_chunk_count": receipt.row_pair_count,
                "positive_removal_stream_coordinate": _RAW_TARGET_LOGIT_COORDINATE,
                "positive_removal_stream_identity_hash": stream_hash,
                "physical_mask_axis_hash": (
                    self._geometry.receipt.route_axis_hash
                ),
                "ordered_score_route_axis_hash": receipt.route_id_axis_hash,
                "same_capture_and_vjp_registration_hash": (
                    receipt.same_capture_and_vjp_registration_hash
                ),
                "same_capture_and_vjp_execution_proof_available": False,
                "registration_only_projection": True,
                "C_projection_row_derivation_authority": False,
                "packet_projection_receipt_hashes": (
                    receipt.packet_projection_receipt_hashes
                ),
                "corrected_then_literal_pairing_proved": True,
                "shared_positive_removal_rows": False,
                "full_tripartite_rows_materialized": False,
                "host_context_route_rows_transferred": False,
                "host_context_route_row_transfer_count": 0,
                "host_context_route_row_transfer_bytes": 0,
                "host_context_route_transfer_role": "none",
                "raw_route_rows_retained": False,
                "endpoint_hard_masks_materialized": False,
                "candidate_selection_authority": False,
                "mask_commit_authority": False,
            }
            payload["projection_derivation_receipt_hash"] = content_hash(
                payload,
                prefix="modern-v1-label-native-scorer-child-derivation-",
            )
            return payload

    def _collapsed_stream(self) -> NoReturn:
        with self._lock:
            if self._state is _AdapterState.OPEN:
                self._abort(
                    V1LabelNativeProjectionScorerAdapterRefusalCode.COLLAPSED_STREAM,
                    "generic/fused consumption would collapse the dual-chart streams",
                )
            _refuse(
                V1LabelNativeProjectionScorerAdapterRefusalCode.CLOSED,
                "scorer adapter is already closed",
            )

    def close(self) -> None:
        with self._lock:
            if self._state in {
                _AdapterState.SEALED,
                _AdapterState.FAILED,
                _AdapterState.CLOSED,
            }:
                return
            self._cleanup_live_state()
            self._state = _AdapterState.CLOSED

    def __enter__(self) -> "V1LabelNativeProjectionScorerAdapter":
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("scorer-adapter transactions are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("scorer-adapter transactions are nonserializable")


class _V1LabelNativeProjectionScorerChild(_RouteChunkedLossOnlyConsumer):
    """One nominal scorer ABI child backed by the shared coordinator."""

    positive_removal_stream_coordinate = _RAW_TARGET_LOGIT_COORDINATE

    __slots__ = (
        "_owner",
        "_role",
        "chunk_size",
        "device",
        "head_ids",
        "loss_pullback_hash",
        "route_ids",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        _construction_token: object,
        owner: V1LabelNativeProjectionScorerAdapter,
        role: _StreamRole,
    ) -> None:
        if _construction_token is not _CHILD_TOKEN:
            raise PermissionError("scorer children are issued only by their coordinator")
        self._owner = owner
        self._role = role
        self.route_ids = owner._route_ids
        self.device = owner._geometry._device
        self.loss_pullback_hash = (
            owner._corrected_stream_hash
            if role is _StreamRole.CORRECTED
            else owner._jacobian_stream_hash
        )
        self.head_ids = (
            f"{role.value}:{_RAW_TARGET_LOGIT_COORDINATE}",
        )
        self.chunk_size = 0

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        self._owner._bind_route_chunk_size(self._role, chunk_size)
        self.chunk_size = int(chunk_size)

    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        self._owner._begin_packet(
            self._role,
            context_ids,
            route_ids=route_ids,
            destination_transport_hash=destination_transport_hash,
            loss_pullback_hash=loss_pullback_hash,
        )

    def consume_positive_removal_chunk(
        self,
        start: int,
        stop: int,
        rows: Any,
    ) -> None:
        self._owner._consume_chunk(self._role, start, stop, rows)

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        del start, stop, rows
        self._owner._collapsed_stream()

    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        global_raw: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        del start, stop, global_raw, gate, pullback, affine_scale, score_epilogue
        self._owner._collapsed_stream()

    def end_packet(self) -> None:
        self._owner._end_packet(self._role)

    def route_streaming_receipt(self) -> Mapping[str, object]:
        return self._owner._child_receipt(self._role)

    def __reduce__(self) -> NoReturn:
        raise TypeError("scorer-adapter children are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("scorer-adapter children are nonserializable")


__all__ = [
    "LABEL_NATIVE_PROJECTION_SCORER_ADAPTER_RECEIPT_SCHEMA",
    "LABEL_NATIVE_PROJECTION_SCORER_AUTHORITY",
    "LABEL_NATIVE_PROJECTION_SCORER_EXECUTION_CLOSURE_RECEIPT_SCHEMA",
    "LABEL_NATIVE_PROJECTION_SCORER_EXECUTION_CLOSURE_SCHEMA",
    "LABEL_NATIVE_PROJECTION_SCORER_PAIRING_ID",
    "V1LabelNativeProjectionScorerAdapter",
    "V1LabelNativeProjectionScorerAdapterReceipt",
    "V1LabelNativeProjectionScorerAdapterRefusal",
    "V1LabelNativeProjectionScorerAdapterRefusalCode",
    "V1LabelNativeProjectionScorerAdapterResult",
    "V1LabelNativeProjectionScorerExecutionClosure",
    "V1LabelNativeProjectionScorerExecutionClosureReceipt",
    "close_v1_label_native_projection_scorer_execution",
    "label_native_projection_scorer_adapter_source_hash",
]
