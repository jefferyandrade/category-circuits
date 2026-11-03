"""Provider-free physical-rank replay of sampled literal-J row custody.

Canonical sampled D captures corrected Aumann--Shapley evidence and the
literal relaxed-gate Jacobian once, then seals the two roles in distinct raw
row shards.  Raising the Frequent-Directions rank must replay those already
measured literal-J observations; it must not call the model, recapture a
batch, run another VJP, or silently substitute corrected evidence for the
response Jacobian.

This module owns that narrow replay seam.  It verifies a path-free binding to
the original same-capture/VJP acquisition, reconstructs one registered
literal-J packet at a time from route-window-major custody, and appends the
packet unweighted to a fresh stratified FD accumulator.  Corrected moments
remain caller owned and are returned by object identity.  The resulting
receipt has proposal-input authority only: it does not certify F/Q/R, open C,
K, or O, authorize a mask commit, or carry terminal/Pareto authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Final, Mapping, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_proposal_statistics import (
    V1StratifiedCorrectedAttributionMoments,
)
from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowLivePrefixAttachment,
    V1SampledDualChartRowRole,
)
from .modern_v1_stratified_fd_gpu import (
    V1StratifiedDeviceFDTrustAccumulator,
    V1StratifiedFDAccumulatorSnapshot,
    V1StratifiedFDDesign,
)

try:  # pragma: no cover - availability depends on the execution host.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SAMPLED_LITERAL_J_ORIGINAL_CAPTURE_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_literal_j_original_capture_binding_v1"
)
SAMPLED_LITERAL_J_FD_RANK_REPLAY_RECEIPT_SCHEMA_V1: Final = (
    "partition_free_v1_sampled_literal_j_fd_rank_replay_receipt_v1"
)
SAMPLED_LITERAL_J_FD_RANK_REPLAY_RECEIPT_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_literal_j_fd_rank_replay_receipt_v2"
)
SAMPLED_LITERAL_J_FD_RANK_REPLAY_RECEIPT_SCHEMA: Final = (
    SAMPLED_LITERAL_J_FD_RANK_REPLAY_RECEIPT_SCHEMA_V2
)
SAMPLED_LITERAL_J_FD_RANK_REPLAY_AUTHORITY_SCOPE: Final = (
    "provider_free_literal_j_rank_replay_for_proposal_q_geometry_only_no_"
    "f_q_r_certification_c_k_o_mask_commit_terminal_or_pareto_authority_v1"
)
SAMPLED_LITERAL_J_FD_RANK_REPLAY_ARITHMETIC: Final = (
    "registered_packet_order_unweighted_within_stratum_fp64_fd_replay_v1"
)

_BINDING_PREFIX: Final = "partition-free-v1-literal-j-original-capture-binding-"
_RECEIPT_PREFIX: Final = "partition-free-v1-literal-j-fd-rank-replay-one-pass-v2-"
_CONTEXT_AXIS_PREFIX: Final = "partition-free-v1-literal-j-replay-context-axis-"
_DRAW_AXIS_PREFIX: Final = "partition-free-v1-literal-j-replay-draw-axis-"
_STRATUM_AXIS_PREFIX: Final = "partition-free-v1-literal-j-replay-stratum-axis-"
_PACKET_AXIS_PREFIX: Final = "partition-free-v1-literal-j-replay-packet-axis-"
_SOURCE_MEASUREMENT_PREFIX: Final = (
    "partition-free-v1-literal-j-replay-source-measurement-"
)
_SOURCE_MEASUREMENT_AXIS_PREFIX: Final = (
    "partition-free-v1-literal-j-replay-source-measurement-axis-"
)
_FLOAT64_BYTES: Final = 8
_RECEIPT_ISSUANCE_TOKEN: Final = object()


class V1SampledLiteralJFDRankReplayMode(str, Enum):
    """Execution modes with deliberately unequal authority."""

    PRODUCTION_CUDA = "production_cuda"
    HIGH_PRECISION_CPU_REFERENCE = "high_precision_cpu_reference_only"


class V1SampledLiteralJFDRankReplayRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    AXIS_MISMATCH = "axis_mismatch"
    STALE_IDENTITY = "stale_identity"
    SOURCE_PROOF_MISMATCH = "source_proof_mismatch"
    ROLE_VIOLATION = "role_violation"
    PRODUCTION_CUDA_REQUIRED = "production_cuda_required"
    NONFINITE_ROW = "nonfinite_row"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledLiteralJFDRankReplayRefusal(ValueError):
    """Typed fail-closed refusal for literal-J physical-rank replay."""

    def __init__(
        self,
        code: V1SampledLiteralJFDRankReplayRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledLiteralJFDRankReplayRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledLiteralJFDRankReplayRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _ordered_identities(
    values: object,
    *,
    name: str,
    unique: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered string sequence",
        )
    try:
        result = tuple(
            _identity(value, name=f"{name} member")  # type: ignore[arg-type]
            for value in values  # type: ignore[union-attr]
        )
    except TypeError as error:
        raise V1SampledLiteralJFDRankReplayRefusal(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            f"{name} must be iterable",
        ) from error
    if not result:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            f"{name} must be nonempty",
        )
    if unique and len(set(result)) != len(result):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.STALE_IDENTITY,
            f"{name} must not contain duplicates",
        )
    return result


def _packet_axis(
    values: object,
    *,
    name: str,
    unique_within_packet: bool,
    unique_flattened: bool,
) -> tuple[tuple[str, ...], ...]:
    if isinstance(values, (str, bytes)):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            f"{name} must be one nested ordered axis",
        )
    try:
        result = tuple(
            _ordered_identities(
                packet,
                name=f"{name} packet",
                unique=unique_within_packet,
            )
            for packet in values  # type: ignore[union-attr]
        )
    except TypeError as error:
        raise V1SampledLiteralJFDRankReplayRefusal(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            f"{name} must be iterable",
        ) from error
    if not result:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            f"{name} must contain at least one packet",
        )
    flattened = tuple(value for packet in result for value in packet)
    if unique_flattened and len(set(flattened)) != len(flattened):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.STALE_IDENTITY,
            f"{name} must be globally unique",
        )
    return result


def _flatten(values: Sequence[Sequence[str]]) -> tuple[str, ...]:
    return tuple(value for packet in values for value in packet)


@dataclass(frozen=True, slots=True)
class V1SampledLiteralJOriginalCaptureBinding:
    """Path-free binding to the one original dual-row acquisition.

    A verified V5 transaction is expected to construct this object.  The
    replay module deliberately does not import the transaction receipt type:
    the rank ladder is coupled to scientific coordinates and proof hashes,
    not to an outer persistence implementation.

    ``original_outer_receipt_hash`` and ``row_group_plan_hash`` identify the
    latest V5 suffix whose publication advanced this cumulative prefix; they
    are preserved as singular compatibility fields. ``parent_binding_hash`` is
    the cumulative-chain commitment: first-look bindings set it to ``None``,
    and every later binding commits the exact verified predecessor binding.
    """

    original_outer_receipt_hash: str
    row_group_plan_hash: str
    row_prefix_manifest_hash: str
    packet_plan_hashes: tuple[str, ...]
    packet_source_capture_identity_hashes: tuple[str, ...]
    packet_provider_proof_hashes: tuple[str, ...]
    packet_raw_dual_proof_hashes: tuple[str, ...]
    packet_fd_source_measurement_hashes: tuple[tuple[str, ...], ...]
    packet_context_ids: tuple[tuple[str, ...], ...]
    packet_draw_ids: tuple[tuple[str, ...], ...]
    packet_stratum_labels: tuple[tuple[str, ...], ...]
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    original_capture_batch_count: int
    original_vjp_count: int
    registered_d_prefix_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    parent_binding_hash: str | None = None
    same_capture_and_vjp: bool = True
    corrected_rows_excluded_from_response_geometry: bool = True
    canonical_q_input_authority: bool = True
    schema: str = SAMPLED_LITERAL_J_ORIGINAL_CAPTURE_BINDING_SCHEMA
    _binding_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "original_outer_receipt_hash",
            "row_group_plan_hash",
            "row_prefix_manifest_hash",
            "corrected_a_stream_identity_hash",
            "literal_j_stream_identity_hash",
            "registered_d_prefix_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        parent = (
            None
            if self.parent_binding_hash is None
            else _identity(self.parent_binding_hash, name="parent binding hash")
        )
        packet_plans = _ordered_identities(
            self.packet_plan_hashes,
            name="capture packet-plan hashes",
            unique=True,
        )
        source_captures = _ordered_identities(
            self.packet_source_capture_identity_hashes,
            name="capture source identities",
            unique=True,
        )
        provider_proofs = _ordered_identities(
            self.packet_provider_proof_hashes,
            name="capture provider proofs",
            unique=True,
        )
        raw_dual_proofs = _ordered_identities(
            self.packet_raw_dual_proof_hashes,
            name="capture raw-dual proofs",
            unique=True,
        )
        fd_source_measurements = _packet_axis(
            self.packet_fd_source_measurement_hashes,
            name="capture FD source-measurement hashes",
            unique_within_packet=True,
            unique_flattened=True,
        )
        contexts = _packet_axis(
            self.packet_context_ids,
            name="capture context axis",
            unique_within_packet=True,
            unique_flattened=True,
        )
        draws = _packet_axis(
            self.packet_draw_ids,
            name="capture draw axis",
            unique_within_packet=True,
            unique_flattened=True,
        )
        strata = _packet_axis(
            self.packet_stratum_labels,
            name="capture stratum axis",
            unique_within_packet=False,
            unique_flattened=False,
        )
        packet_count = len(packet_plans)
        if any(
            len(values) != packet_count
            for values in (
                source_captures,
                provider_proofs,
                raw_dual_proofs,
                fd_source_measurements,
                contexts,
                draws,
                strata,
            )
        ):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
                "original capture proof ledgers have different packet counts",
            )
        if any(
            len(contexts[index]) != len(draws[index])
            or len(contexts[index]) != len(strata[index])
            or len(contexts[index]) != len(fd_source_measurements[index])
            for index in range(packet_count)
        ):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.AXIS_MISMATCH,
                "original per-packet measurement/context/draw/stratum axes differ",
            )
        capture_count = _positive_int(
            self.original_capture_batch_count,
            name="original capture-batch count",
        )
        vjp_count = _positive_int(self.original_vjp_count, name="original VJP count")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="original proposal generation",
        )
        routes = _positive_int(self.route_count, name="original route count")
        if capture_count != packet_count or vjp_count != packet_count:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
                "original acquisition must bind one capture and VJP per packet",
            )
        if self.corrected_a_stream_identity_hash == self.literal_j_stream_identity_hash:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.ROLE_VIOLATION,
                "corrected-A and literal-J stream identities collapsed",
            )
        if (
            self.same_capture_and_vjp is not True
            or self.corrected_rows_excluded_from_response_geometry is not True
            or self.canonical_q_input_authority is not True
            or self.schema != SAMPLED_LITERAL_J_ORIGINAL_CAPTURE_BINDING_SCHEMA
        ):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.AUTHORITY_VIOLATION,
                "original capture binding changed dual-row or Q-input authority",
            )
        object.__setattr__(self, "packet_plan_hashes", packet_plans)
        object.__setattr__(
            self, "packet_source_capture_identity_hashes", source_captures
        )
        object.__setattr__(self, "packet_provider_proof_hashes", provider_proofs)
        object.__setattr__(self, "packet_raw_dual_proof_hashes", raw_dual_proofs)
        object.__setattr__(
            self,
            "packet_fd_source_measurement_hashes",
            fd_source_measurements,
        )
        object.__setattr__(self, "packet_context_ids", contexts)
        object.__setattr__(self, "packet_draw_ids", draws)
        object.__setattr__(self, "packet_stratum_labels", strata)
        object.__setattr__(self, "original_capture_batch_count", capture_count)
        object.__setattr__(self, "original_vjp_count", vjp_count)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "parent_binding_hash", parent)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "original_outer_receipt_hash": self.original_outer_receipt_hash,
            "row_group_plan_hash": self.row_group_plan_hash,
            "row_prefix_manifest_hash": self.row_prefix_manifest_hash,
            "parent_binding_hash": self.parent_binding_hash,
            "packet_plan_hashes": self.packet_plan_hashes,
            "packet_source_capture_identity_hashes": (
                self.packet_source_capture_identity_hashes
            ),
            "packet_provider_proof_hashes": self.packet_provider_proof_hashes,
            "packet_raw_dual_proof_hashes": self.packet_raw_dual_proof_hashes,
            "packet_fd_source_measurement_hashes": (
                self.packet_fd_source_measurement_hashes
            ),
            "packet_context_ids": self.packet_context_ids,
            "packet_draw_ids": self.packet_draw_ids,
            "packet_stratum_labels": self.packet_stratum_labels,
            "corrected_a_stream_identity_hash": self.corrected_a_stream_identity_hash,
            "literal_j_stream_identity_hash": self.literal_j_stream_identity_hash,
            "original_capture_batch_count": self.original_capture_batch_count,
            "original_vjp_count": self.original_vjp_count,
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_axis_hash": self.route_axis_hash,
            "route_id_axis_hash": self.route_id_axis_hash,
            "route_count": self.route_count,
            "same_capture_and_vjp": True,
            "corrected_rows_excluded_from_response_geometry": True,
            "canonical_q_input_authority": True,
        }

    @property
    def binding_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_BINDING_PREFIX)
        if self._binding_hash is not None and self._binding_hash != observed:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
                "original capture binding was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
                "original capture binding hash differs",
            )
        object.__setattr__(self, "_binding_hash", observed)
        return observed

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "binding_hash": self.binding_hash}


@dataclass(frozen=True, slots=True)
class V1SampledLiteralJFDRankReplayReceipt:
    """Versioned, path-free receipt for one rank-specific literal-J replay."""

    original_capture_binding_hash: str
    original_outer_receipt_hash: str
    row_group_plan_hash: str
    row_prefix_manifest_hash: str
    literal_j_shard_receipt_hashes: tuple[str, ...]
    literal_j_shard_content_sha256s: tuple[str, ...]
    fd_design_hash: str
    fd_snapshot_hash: str
    corrected_moments_state_hash: str
    requested_physical_rank: int
    packet_count: int
    context_count: int
    original_capture_batch_count: int
    original_vjp_count: int
    context_axis_hash: str
    draw_axis_hash: str
    stratum_axis_hash: str
    packet_axis_hash: str
    source_measurement_axis_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    literal_j_stream_identity_hash: str
    corrected_a_stream_identity_hash: str
    execution_mode: V1SampledLiteralJFDRankReplayMode
    device_identity: str
    literal_j_verified_packet_read_count: int
    literal_j_route_window_read_count: int
    literal_j_descriptor_readinto_call_count: int
    literal_j_digest_update_count: int
    literal_j_payload_read_bytes: int
    separate_literal_j_integrity_scan_count: int
    separate_literal_j_integrity_scan_bytes: int
    aggregate_host_file_read_bytes: int
    packet_host_matrix_allocation_count: int
    aggregate_packet_host_matrix_bytes: int
    maximum_literal_j_window_staging_bytes: int
    maximum_owned_host_staging_bytes: int
    fd_append_input_tensor_allocation_count: int
    fd_append_input_tensor_bytes: int
    maximum_fd_append_input_tensor_bytes: int
    cpu_reference_fd_append_copy_bytes: int
    h2d_transfer_count: int
    h2d_transfer_bytes: int
    maximum_h2d_transfer_bytes: int
    fd_snapshot_host_array_count: int
    fd_snapshot_host_bytes: int
    d2h_snapshot_transfer_count: int
    d2h_snapshot_transfer_bytes: int
    maximum_d2h_snapshot_transfer_bytes: int
    provider_call_count: int = 0
    capture_batch_count: int = 0
    vjp_count: int = 0
    corrected_shard_read_count: int = 0
    weights_argument_was_none_for_every_packet: bool = True
    observations_replayed_unweighted_within_stratum: bool = True
    exact_registered_packet_order: bool = True
    corrected_moments_caller_owned: bool = True
    corrected_moments_recomputed: bool = False
    one_pass_verified_custody_materialization: bool = True
    source_literal_j_raw_row_custody_retained: bool = True
    replay_temporary_context_route_rows_retained: bool = False
    raw_context_route_rows_retained: bool = True
    fq_certification_authority: bool = False
    c_k_o_authority: bool = False
    mask_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    replay_arithmetic: str = SAMPLED_LITERAL_J_FD_RANK_REPLAY_ARITHMETIC
    authority_scope: str = SAMPLED_LITERAL_J_FD_RANK_REPLAY_AUTHORITY_SCOPE
    schema: str = SAMPLED_LITERAL_J_FD_RANK_REPLAY_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _RECEIPT_ISSUANCE_TOKEN:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.AUTHORITY_VIOLATION,
                "rank-replay receipts are issued only by a completed replay",
            )
        for name in (
            "original_capture_binding_hash",
            "original_outer_receipt_hash",
            "row_group_plan_hash",
            "row_prefix_manifest_hash",
            "fd_design_hash",
            "fd_snapshot_hash",
            "corrected_moments_state_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "stratum_axis_hash",
            "packet_axis_hash",
            "source_measurement_axis_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "literal_j_stream_identity_hash",
            "corrected_a_stream_identity_hash",
            "device_identity",
        ):
            _identity(getattr(self, name), name=name)
        shard_receipts = _ordered_identities(
            self.literal_j_shard_receipt_hashes,
            name="replayed literal-J shard receipts",
            unique=True,
        )
        shard_digests = _ordered_identities(
            self.literal_j_shard_content_sha256s,
            name="replayed literal-J shard content digests",
            unique=False,
        )
        rank = _positive_int(self.requested_physical_rank, name="physical FD rank")
        packet_count = _positive_int(self.packet_count, name="replay packet count")
        context_count = _positive_int(self.context_count, name="replay context count")
        original_capture_count = _positive_int(
            self.original_capture_batch_count,
            name="bound original capture-batch count",
        )
        original_vjp_count = _positive_int(
            self.original_vjp_count,
            name="bound original VJP count",
        )
        routes = _positive_int(self.route_count, name="replay route count")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="replay proposal generation",
        )
        if (
            len(shard_receipts) != packet_count
            or len(shard_digests) != packet_count
            or original_capture_count != packet_count
            or original_vjp_count != packet_count
        ):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
                "replay receipt does not bind every literal-J shard",
            )
        for name in (
            "literal_j_verified_packet_read_count",
            "literal_j_route_window_read_count",
            "literal_j_descriptor_readinto_call_count",
            "literal_j_digest_update_count",
            "literal_j_payload_read_bytes",
            "separate_literal_j_integrity_scan_count",
            "separate_literal_j_integrity_scan_bytes",
            "aggregate_host_file_read_bytes",
            "packet_host_matrix_allocation_count",
            "aggregate_packet_host_matrix_bytes",
            "maximum_literal_j_window_staging_bytes",
            "maximum_owned_host_staging_bytes",
            "fd_append_input_tensor_allocation_count",
            "fd_append_input_tensor_bytes",
            "maximum_fd_append_input_tensor_bytes",
            "cpu_reference_fd_append_copy_bytes",
            "h2d_transfer_count",
            "h2d_transfer_bytes",
            "maximum_h2d_transfer_bytes",
            "fd_snapshot_host_array_count",
            "fd_snapshot_host_bytes",
            "d2h_snapshot_transfer_count",
            "d2h_snapshot_transfer_bytes",
            "maximum_d2h_snapshot_transfer_bytes",
            "provider_call_count",
            "capture_batch_count",
            "vjp_count",
            "corrected_shard_read_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        payload_bytes = context_count * routes * _FLOAT64_BYTES
        if (
            self.literal_j_verified_packet_read_count != packet_count
            or self.literal_j_route_window_read_count <= 0
            or self.literal_j_descriptor_readinto_call_count
            < self.literal_j_route_window_read_count
            or self.literal_j_digest_update_count
            != self.literal_j_route_window_read_count
            or self.literal_j_payload_read_bytes != payload_bytes
            or self.separate_literal_j_integrity_scan_count != 0
            or self.separate_literal_j_integrity_scan_bytes != 0
            or self.aggregate_host_file_read_bytes != payload_bytes
            or self.packet_host_matrix_allocation_count != packet_count
            or self.aggregate_packet_host_matrix_bytes != payload_bytes
            or not 0 < self.maximum_literal_j_window_staging_bytes <= payload_bytes
            or self.maximum_owned_host_staging_bytes <= 0
            or self.maximum_owned_host_staging_bytes
            < self.maximum_literal_j_window_staging_bytes
            or self.fd_append_input_tensor_allocation_count != packet_count
            or self.fd_append_input_tensor_bytes != payload_bytes
            or not 0 < self.maximum_fd_append_input_tensor_bytes <= payload_bytes
            or self.fd_snapshot_host_array_count <= 0
            or self.fd_snapshot_host_bytes <= 0
            or self.maximum_owned_host_staging_bytes < self.fd_snapshot_host_bytes
            or self.provider_call_count != 0
            or self.capture_batch_count != 0
            or self.vjp_count != 0
            or self.corrected_shard_read_count != 0
        ):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.AUTHORITY_VIOLATION,
                "rank-replay execution or host-I/O telemetry does not replay",
            )
        cuda = self.execution_mode is V1SampledLiteralJFDRankReplayMode.PRODUCTION_CUDA
        if (
            cuda
            and (
                self.h2d_transfer_count != packet_count
                or self.h2d_transfer_bytes != payload_bytes
                or self.d2h_snapshot_transfer_count != self.fd_snapshot_host_array_count
                or self.d2h_snapshot_transfer_bytes != self.fd_snapshot_host_bytes
                or self.maximum_d2h_snapshot_transfer_bytes <= 0
                or self.cpu_reference_fd_append_copy_bytes != 0
            )
        ) or (
            not cuda
            and (
                self.h2d_transfer_count != 0
                or self.h2d_transfer_bytes != 0
                or self.maximum_h2d_transfer_bytes != 0
                or self.d2h_snapshot_transfer_count != 0
                or self.d2h_snapshot_transfer_bytes != 0
                or self.maximum_d2h_snapshot_transfer_bytes != 0
                or self.cpu_reference_fd_append_copy_bytes != payload_bytes
            )
        ):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.AUTHORITY_VIOLATION,
                "rank-replay H2D telemetry differs from its execution mode",
            )
        if (
            type(self.execution_mode) is not V1SampledLiteralJFDRankReplayMode
            or self.literal_j_stream_identity_hash
            == self.corrected_a_stream_identity_hash
            or self.weights_argument_was_none_for_every_packet is not True
            or self.observations_replayed_unweighted_within_stratum is not True
            or self.exact_registered_packet_order is not True
            or self.corrected_moments_caller_owned is not True
            or self.corrected_moments_recomputed is not False
            or self.one_pass_verified_custody_materialization is not True
            or self.source_literal_j_raw_row_custody_retained is not True
            or self.replay_temporary_context_route_rows_retained is not False
            or self.raw_context_route_rows_retained is not True
            or self.fq_certification_authority is not False
            or self.c_k_o_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_or_pareto_authority is not False
            or self.replay_arithmetic != SAMPLED_LITERAL_J_FD_RANK_REPLAY_ARITHMETIC
            or self.authority_scope != SAMPLED_LITERAL_J_FD_RANK_REPLAY_AUTHORITY_SCOPE
            or self.schema != SAMPLED_LITERAL_J_FD_RANK_REPLAY_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.AUTHORITY_VIOLATION,
                "rank-replay receipt changed its scientific or authority boundary",
            )
        object.__setattr__(self, "literal_j_shard_receipt_hashes", shard_receipts)
        object.__setattr__(self, "literal_j_shard_content_sha256s", shard_digests)
        object.__setattr__(self, "requested_physical_rank", rank)
        object.__setattr__(self, "packet_count", packet_count)
        object.__setattr__(self, "context_count", context_count)
        object.__setattr__(
            self,
            "original_capture_batch_count",
            original_capture_count,
        )
        object.__setattr__(self, "original_vjp_count", original_vjp_count)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "proposal_generation", generation)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        names = tuple(
            name for name in self.__dataclass_fields__ if not name.startswith("_")
        )
        return {name: getattr(self, name) for name in names}

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
                "rank-replay receipt was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
                "rank-replay receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class V1SampledLiteralJFDRankReplayResult:
    """Live result retaining the caller's exact corrected-moments object."""

    corrected_moments: V1StratifiedCorrectedAttributionMoments = field(
        repr=False,
        compare=False,
    )
    fd_snapshot: V1StratifiedFDAccumulatorSnapshot
    receipt: V1SampledLiteralJFDRankReplayReceipt

    def __post_init__(self) -> None:
        if type(self.corrected_moments) is not V1StratifiedCorrectedAttributionMoments:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
                "replay result requires exact corrected moments",
            )
        if type(self.fd_snapshot) is not V1StratifiedFDAccumulatorSnapshot:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
                "replay result requires an exact FD snapshot",
            )
        if type(self.receipt) is not V1SampledLiteralJFDRankReplayReceipt:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
                "replay result requires its exact receipt",
            )
        self.receipt.assert_integrity()
        if (
            self.corrected_moments.state_hash
            != self.receipt.corrected_moments_state_hash
            or self.fd_snapshot.snapshot_hash != self.receipt.fd_snapshot_hash
        ):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
                "live replay outputs differ from the just-issued receipt",
            )


def load_v1_sampled_literal_j_fd_rank_replay_receipt(
    payload: Mapping[str, object],
    *,
    expected_receipt_hash: str | None = None,
) -> V1SampledLiteralJFDRankReplayReceipt:
    """Replay a serialized rank-replay receipt through its owning module.

    The private receipt token remains private to this module.  Callers may
    recover an exact typed receipt from a path-free persisted payload, but
    they cannot mint one by importing the issuance token or by treating a
    hash-only projection as receipt authority.
    """

    if not isinstance(payload, Mapping):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            "rank-replay receipt payload must be one mapping",
        )
    values = dict(payload)
    declared_hash = values.pop("receipt_hash", None)
    values.pop("_receipt_hash", None)
    allowed = {
        name
        for name in V1SampledLiteralJFDRankReplayReceipt.__dataclass_fields__
        if not name.startswith("_")
    }
    if set(values) != allowed:
        missing = tuple(sorted(allowed - set(values)))
        extra = tuple(sorted(set(values) - allowed))
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            "rank-replay receipt payload fields do not match schema; "
            f"missing={missing}, extra={extra}",
        )
    mode = values["execution_mode"]
    if type(mode) is str:
        try:
            values["execution_mode"] = V1SampledLiteralJFDRankReplayMode(mode)
        except ValueError:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
                "rank-replay receipt payload has an unknown execution mode",
            )
    elif type(mode) is not V1SampledLiteralJFDRankReplayMode:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            "rank-replay receipt payload execution mode is invalid",
        )
    receipt = V1SampledLiteralJFDRankReplayReceipt(
        **values,
        _issuance_token=_RECEIPT_ISSUANCE_TOKEN,
    )
    expected = expected_receipt_hash
    if expected is None and declared_hash is not None:
        if type(declared_hash) is not str or not declared_hash:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
                "rank-replay receipt payload declared a noncanonical hash",
            )
        expected = declared_hash
    receipt.assert_integrity(expected)
    return receipt


def derive_v1_sampled_literal_j_original_capture_binding_from_v5_commit(
    *,
    commit: object,
    attachment: V1SampledDualChartRowLivePrefixAttachment,
) -> V1SampledLiteralJOriginalCaptureBinding:
    """Derive the public replay binding from the actual V5 commit boundary.

    This is the backend-facing constructor for higher-rank replay.  It rejects
    caller-authored ledgers by recomputing them from the V5 transaction
    request, dual-chart proof, row-custody proof, and published row prefix.
    The resulting binding is still path-free and serializable, but its packet
    identities are not a free-form caller assertion.
    """

    return (
        derive_v1_sampled_literal_j_cumulative_original_capture_binding_from_v5_commit(
            prior_original_capture_binding=None,
            commit=commit,
            attachment=attachment,
        )
    )


def _assert_no_duplicate_flat_identities(
    packets: Sequence[Sequence[str]],
    *,
    name: str,
) -> None:
    flat = _flatten(packets)
    if len(set(flat)) != len(flat):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.STALE_IDENTITY,
            f"{name} contains duplicate identities",
        )


def _assert_binding_matches_manifest_prefix(
    *,
    binding: V1SampledLiteralJOriginalCaptureBinding,
    attachment: V1SampledDualChartRowLivePrefixAttachment,
    stop: int,
) -> None:
    manifest = attachment.manifest
    binding.assert_integrity()
    if stop <= 0 or stop > len(manifest.looks):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.AXIS_MISMATCH,
            "prior binding does not name a nonempty manifest prefix",
        )
    prefix_looks = tuple(manifest.looks[:stop])
    if (
        binding.packet_source_capture_identity_hashes
        != tuple(
            look.literal_j_shard.source_capture_identity_hash for look in prefix_looks
        )
        or binding.packet_context_ids
        != tuple(look.context_ids for look in prefix_looks)
        or binding.packet_draw_ids != tuple(look.draw_ids for look in prefix_looks)
        or binding.packet_stratum_labels
        != tuple(look.stratum_labels for look in prefix_looks)
    ):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
            "prior binding does not match the cumulative attachment prefix",
        )
    shared = (
        (binding.registered_d_prefix_hash, manifest.registered_d_prefix_hash),
        (binding.chart_identity_hash, manifest.chart_identity_hash),
        (binding.current_mask_hash, manifest.current_mask_hash),
        (binding.proposal_generation, manifest.proposal_generation),
        (binding.route_axis_hash, manifest.route_axis_hash),
        (binding.route_id_axis_hash, manifest.route_id_axis_hash),
        (binding.route_count, manifest.route_count),
        (
            binding.corrected_a_stream_identity_hash,
            manifest.corrected_a_stream_identity_hash,
        ),
        (
            binding.literal_j_stream_identity_hash,
            manifest.literal_j_stream_identity_hash,
        ),
    )
    if any(left != right for left, right in shared):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
            "prior binding and cumulative attachment name different chart coordinates",
        )


def derive_v1_sampled_literal_j_cumulative_original_capture_binding_from_v5_commit(
    *,
    prior_original_capture_binding: V1SampledLiteralJOriginalCaptureBinding | None,
    commit: object,
    attachment: V1SampledDualChartRowLivePrefixAttachment,
) -> V1SampledLiteralJOriginalCaptureBinding:
    """Derive a cumulative replay binding from a newly published V5 suffix.

    For the first look, ``prior_original_capture_binding`` must be ``None`` and
    the current V5 group must cover the entire published prefix.  For later
    looks, the prior binding must exactly match the predecessor manifest named
    by the current group; this function then appends only the newly committed
    suffix ledgers and returns one binding for the cumulative manifest.  The
    singular ``original_outer_receipt_hash`` and ``row_group_plan_hash`` fields
    remain latest-suffix compatibility identities; ``parent_binding_hash`` is
    the cumulative chain commitment to the exact verified predecessor binding.
    """

    from .modern_v1_sampled_look_transaction import (
        SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5,
        V1SampledRegisteredLookCommit,
    )

    if (
        prior_original_capture_binding is not None
        and type(prior_original_capture_binding)
        is not V1SampledLiteralJOriginalCaptureBinding
    ):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            "prior original-capture binding must be exact when supplied",
        )
    if type(commit) is not V1SampledRegisteredLookCommit:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            "original capture binding requires an exact V5 sampled-look commit",
        )
    typed_commit = commit
    if typed_commit.request.schema != SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.AUTHORITY_VIOLATION,
            "original capture binding can only be derived from V5 row custody",
        )
    if type(attachment) is not V1SampledDualChartRowLivePrefixAttachment:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            "original capture binding requires an exact live row-prefix attachment",
        )
    # The writer or a cold reattachment already established content identity.
    # Binding derivation needs only the process-local immutable witness and
    # path-free role chain; it must not rescan the route-scale payload.
    attachment.verify_live_witnesses()
    typed_commit.commit_hash
    request = typed_commit.request
    group_plan = request.dual_chart_row_group_plan
    dual_receipt = typed_commit.dual_chart_receipt
    custody = typed_commit.dual_chart_row_custody_receipt
    if group_plan is None or dual_receipt is None or custody is None:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
            "V5 commit lacks the dual-chart or row-custody proof required for replay",
        )
    group_plan.assert_integrity()
    dual_receipt.receipt_hash
    custody.receipt_hash
    manifest = attachment.manifest
    group_packets = tuple(group_plan.packet_plans)
    suffix_start = group_plan.first_prefix_look_ordinal
    suffix_stop = suffix_start + len(group_packets)
    if (
        manifest.manifest_hash != custody.group_commit_receipt.manifest_hash
        or manifest.registered_d_prefix_hash != group_plan.registered_d_prefix_hash
        or manifest.chart_identity_hash != group_plan.chart_identity_hash
        or manifest.current_mask_hash != group_plan.current_mask_hash
        or manifest.proposal_generation != group_plan.proposal_generation
        or manifest.route_axis_hash != group_plan.route_axis_hash
        or manifest.route_id_axis_hash != group_plan.route_id_axis_hash
        or manifest.route_count != group_plan.route_count
        or manifest.corrected_a_stream_identity_hash
        != group_plan.corrected_a_stream_identity_hash
        or manifest.literal_j_stream_identity_hash
        != group_plan.literal_j_stream_identity_hash
    ):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
            "published row prefix differs from the V5 row-group plan",
        )
    if len(manifest.looks) != suffix_stop:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.AXIS_MISMATCH,
            "V5 suffix does not end at the published cumulative prefix",
        )
    if prior_original_capture_binding is None:
        if suffix_start != 0 or group_plan.expected_prior_manifest_hash is not None:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.STALE_IDENTITY,
                "non-initial V5 suffix requires the prior replay binding",
            )
        prior_packet_plan_hashes: tuple[str, ...] = ()
        prior_source_capture_identity_hashes: tuple[str, ...] = ()
        prior_provider_proof_hashes: tuple[str, ...] = ()
        prior_raw_dual_proof_hashes: tuple[str, ...] = ()
        prior_fd_source_measurement_hashes: tuple[tuple[str, ...], ...] = ()
        prior_context_ids: tuple[tuple[str, ...], ...] = ()
        prior_draw_ids: tuple[tuple[str, ...], ...] = ()
        prior_stratum_labels: tuple[tuple[str, ...], ...] = ()
    else:
        prior = prior_original_capture_binding
        prior.assert_integrity()
        if (
            suffix_start != len(prior.packet_plan_hashes)
            or group_plan.expected_prior_manifest_hash != prior.row_prefix_manifest_hash
            or custody.group_commit_receipt.prior_manifest_hash
            != prior.row_prefix_manifest_hash
        ):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.STALE_IDENTITY,
                "prior replay binding is not the current V5 predecessor",
            )
        _assert_binding_matches_manifest_prefix(
            binding=prior,
            attachment=attachment,
            stop=suffix_start,
        )
        shared = (
            (prior.registered_d_prefix_hash, group_plan.registered_d_prefix_hash),
            (prior.chart_identity_hash, group_plan.chart_identity_hash),
            (prior.current_mask_hash, group_plan.current_mask_hash),
            (prior.proposal_generation, group_plan.proposal_generation),
            (prior.route_axis_hash, group_plan.route_axis_hash),
            (prior.route_id_axis_hash, group_plan.route_id_axis_hash),
            (prior.route_count, group_plan.route_count),
            (
                prior.corrected_a_stream_identity_hash,
                group_plan.corrected_a_stream_identity_hash,
            ),
            (
                prior.literal_j_stream_identity_hash,
                group_plan.literal_j_stream_identity_hash,
            ),
        )
        if any(left != right for left, right in shared):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
                "prior replay binding and current V5 suffix name different chart coordinates",
            )
        prior_packet_plan_hashes = prior.packet_plan_hashes
        prior_source_capture_identity_hashes = (
            prior.packet_source_capture_identity_hashes
        )
        prior_provider_proof_hashes = prior.packet_provider_proof_hashes
        prior_raw_dual_proof_hashes = prior.packet_raw_dual_proof_hashes
        prior_fd_source_measurement_hashes = prior.packet_fd_source_measurement_hashes
        prior_context_ids = prior.packet_context_ids
        prior_draw_ids = prior.packet_draw_ids
        prior_stratum_labels = prior.packet_stratum_labels
    packet_plans = tuple(plan.plan_hash for plan in request.packet_plans)
    suffix_looks = tuple(manifest.looks[suffix_start:suffix_stop])
    if suffix_looks and suffix_looks[0].prior_manifest_hash != (
        None
        if prior_original_capture_binding is None
        else prior_original_capture_binding.row_prefix_manifest_hash
    ):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.STALE_IDENTITY,
            "V5 suffix does not continue the expected predecessor manifest",
        )
    expected_suffix_contexts = tuple(look.context_ids for look in suffix_looks)
    expected_suffix_draws = tuple(look.draw_ids for look in suffix_looks)
    expected_suffix_strata = tuple(look.stratum_labels for look in suffix_looks)
    suffix_source_captures = tuple(
        packet.source_capture_identity_hash for packet in group_packets
    )
    suffix_fd_source_measurements = tuple(
        plan.fd_request.source_measurement_hashes for plan in request.packet_plans
    )
    if packet_plans != tuple(
        custody.registered_packet_plan_hashes
    ) or packet_plans != tuple(
        packet.registered_packet_plan_hash for packet in group_packets
    ):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
            "V5 commit packet-plan ledgers do not agree",
        )
    if (
        suffix_source_captures
        != tuple(
            look.literal_j_shard.source_capture_identity_hash for look in suffix_looks
        )
        or expected_suffix_contexts
        != tuple(packet.context_ids for packet in group_packets)
        or expected_suffix_draws != tuple(packet.draw_ids for packet in group_packets)
        or expected_suffix_strata
        != tuple(packet.stratum_labels for packet in group_packets)
    ):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
            "V5 row-group suffix differs from the published row-prefix suffix",
        )
    if (
        tuple(dual_receipt.packet_plan_hashes) != packet_plans
        or tuple(dual_receipt.provider_receipt_hashes)
        != tuple(custody.provider_receipt_hashes)
        or tuple(dual_receipt.raw_dual_stream_receipt_hashes)
        == tuple(custody.provider_receipt_hashes)
    ):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
            "dual-chart provider/raw-stream proof ledgers are not replay-safe",
        )
    cumulative_packet_plan_hashes = prior_packet_plan_hashes + packet_plans
    cumulative_source_capture_identity_hashes = (
        prior_source_capture_identity_hashes + suffix_source_captures
    )
    cumulative_provider_proof_hashes = prior_provider_proof_hashes + tuple(
        custody.provider_receipt_hashes
    )
    cumulative_raw_dual_proof_hashes = prior_raw_dual_proof_hashes + tuple(
        dual_receipt.raw_dual_stream_receipt_hashes
    )
    cumulative_fd_source_measurement_hashes = (
        prior_fd_source_measurement_hashes + suffix_fd_source_measurements
    )
    cumulative_context_ids = prior_context_ids + expected_suffix_contexts
    cumulative_draw_ids = prior_draw_ids + expected_suffix_draws
    cumulative_stratum_labels = prior_stratum_labels + expected_suffix_strata
    for packets, name in (
        (cumulative_context_ids, "cumulative context axis"),
        (cumulative_draw_ids, "cumulative draw axis"),
        (
            cumulative_fd_source_measurement_hashes,
            "cumulative FD source-measurement axis",
        ),
    ):
        _assert_no_duplicate_flat_identities(packets, name=name)
    return V1SampledLiteralJOriginalCaptureBinding(
        original_outer_receipt_hash=typed_commit.commit_hash,
        row_group_plan_hash=group_plan.group_plan_hash,
        row_prefix_manifest_hash=manifest.manifest_hash,
        packet_plan_hashes=cumulative_packet_plan_hashes,
        packet_source_capture_identity_hashes=(
            cumulative_source_capture_identity_hashes
        ),
        packet_provider_proof_hashes=cumulative_provider_proof_hashes,
        packet_raw_dual_proof_hashes=cumulative_raw_dual_proof_hashes,
        packet_fd_source_measurement_hashes=(cumulative_fd_source_measurement_hashes),
        packet_context_ids=cumulative_context_ids,
        packet_draw_ids=cumulative_draw_ids,
        packet_stratum_labels=cumulative_stratum_labels,
        corrected_a_stream_identity_hash=(group_plan.corrected_a_stream_identity_hash),
        literal_j_stream_identity_hash=group_plan.literal_j_stream_identity_hash,
        original_capture_batch_count=len(cumulative_packet_plan_hashes),
        original_vjp_count=len(cumulative_packet_plan_hashes),
        registered_d_prefix_hash=group_plan.registered_d_prefix_hash,
        chart_identity_hash=group_plan.chart_identity_hash,
        current_mask_hash=group_plan.current_mask_hash,
        proposal_generation=group_plan.proposal_generation,
        route_axis_hash=group_plan.route_axis_hash,
        route_id_axis_hash=group_plan.route_id_axis_hash,
        route_count=group_plan.route_count,
        parent_binding_hash=(
            None
            if prior_original_capture_binding is None
            else prior_original_capture_binding.binding_hash
        ),
    )


def _assert_original_binding_matches_prefix(
    *,
    binding: V1SampledLiteralJOriginalCaptureBinding,
    prefix: V1SampledDualChartRowLivePrefixAttachment,
) -> None:
    manifest = prefix.manifest
    manifest.assert_integrity()
    binding.assert_integrity()
    looks = manifest.looks
    binding_shared = (
        (binding.row_prefix_manifest_hash, manifest.manifest_hash),
        (binding.registered_d_prefix_hash, manifest.registered_d_prefix_hash),
        (binding.chart_identity_hash, manifest.chart_identity_hash),
        (binding.current_mask_hash, manifest.current_mask_hash),
        (binding.proposal_generation, manifest.proposal_generation),
        (binding.route_axis_hash, manifest.route_axis_hash),
        (binding.route_id_axis_hash, manifest.route_id_axis_hash),
        (binding.route_count, manifest.route_count),
        (
            binding.corrected_a_stream_identity_hash,
            manifest.corrected_a_stream_identity_hash,
        ),
        (
            binding.literal_j_stream_identity_hash,
            manifest.literal_j_stream_identity_hash,
        ),
    )
    if any(left != right for left, right in binding_shared):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
            "original capture binding differs from the attached raw-row prefix",
        )
    if len(looks) != len(binding.packet_plan_hashes):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
            "original capture packet ledger differs from the row prefix",
        )
    if binding.packet_source_capture_identity_hashes != tuple(
        look.literal_j_shard.source_capture_identity_hash for look in looks
    ):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
            "original source-capture ledger differs from literal-J custody",
        )
    expected_contexts = tuple(look.context_ids for look in looks)
    expected_draws = tuple(look.draw_ids for look in looks)
    expected_strata = tuple(look.stratum_labels for look in looks)
    if (
        binding.packet_context_ids != expected_contexts
        or binding.packet_draw_ids != expected_draws
        or binding.packet_stratum_labels != expected_strata
    ):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.AXIS_MISMATCH,
            "original packet context/draw/stratum order differs from custody",
        )


def _assert_design_and_moments_match(
    *,
    design: V1StratifiedFDDesign,
    moments: V1StratifiedCorrectedAttributionMoments,
    binding: V1SampledLiteralJOriginalCaptureBinding,
) -> None:
    shared = (
        (moments.campaign_contract_hash, design.campaign_contract_hash),
        (moments.model_identity_hash, design.model_identity_hash),
        (moments.query_contract_hash, design.query_contract_hash),
        (moments.preprocessing_hash, design.preprocessing_hash),
        (moments.sampling_law_hash, design.sampling_law_hash),
        (moments.current_mask_hash, design.current_mask_hash),
        (moments.current_mask_hash, design.measured_mask_hash),
        (moments.current_mask_hash, binding.current_mask_hash),
        (moments.proposal_generation, design.proposal_generation),
        (moments.proposal_generation, binding.proposal_generation),
        (moments.route_count, len(design.route_ids)),
        (moments.route_count, binding.route_count),
        (moments.route_axis_hash, design.route_axis_hash),
        (moments.route_axis_hash, binding.route_id_axis_hash),
        (
            moments.raw_logit_stream_source_hash,
            binding.corrected_a_stream_identity_hash,
        ),
        (design.raw_logit_stream_source_hash, binding.literal_j_stream_identity_hash),
        (moments.stratum_names, design.stratum_names),
    )
    if any(left != right for left, right in shared):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.STALE_IDENTITY,
            "corrected moments, literal-J design, and custody name different coordinates",
        )
    names = design.stratum_names
    by_name = {name: index for index, name in enumerate(names)}
    contexts_by_stratum: list[list[str]] = [[] for _name in names]
    draws_by_stratum: list[list[str]] = [[] for _name in names]
    for contexts, draws, labels in zip(
        binding.packet_context_ids,
        binding.packet_draw_ids,
        binding.packet_stratum_labels,
        strict=True,
    ):
        if len(set(labels)) != 1:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.AXIS_MISMATCH,
                "one registered FD replay packet cannot mix strata",
            )
        label = labels[0]
        if label not in by_name:
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.AXIS_MISMATCH,
                "capture names an unregistered FD stratum",
            )
        index = by_name[label]
        contexts_by_stratum[index].extend(contexts)
        draws_by_stratum[index].extend(draws)
    observed_contexts = tuple(tuple(values) for values in contexts_by_stratum)
    observed_draws = tuple(tuple(values) for values in draws_by_stratum)
    observed_counts = tuple(len(values) for values in contexts_by_stratum)
    if (
        observed_contexts != moments.observed_context_ids_by_stratum
        or observed_draws != moments.observed_draw_ids_by_stratum
        or observed_counts != moments.stratum_counts
        or any(count <= 0 for count in observed_counts)
    ):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.AXIS_MISMATCH,
            "corrected moments do not describe the exact replayed context/draw prefixes",
        )


def _registered_source_measurements_for_packet(
    *,
    binding: V1SampledLiteralJOriginalCaptureBinding,
    packet_index: int,
    literal_j_shard_receipt_hash: str,
) -> tuple[str, ...]:
    del literal_j_shard_receipt_hash
    measurements = binding.packet_fd_source_measurement_hashes[packet_index]
    if len(measurements) != len(binding.packet_context_ids[packet_index]):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.AXIS_MISMATCH,
            "registered FD source-measurement packet length changed",
        )
    return measurements


def _device_for_mode(
    *,
    mode: V1SampledLiteralJFDRankReplayMode,
    device: object,
) -> Any:
    if torch is None:
        raise RuntimeError("literal-J FD rank replay requires PyTorch")
    if type(mode) is not V1SampledLiteralJFDRankReplayMode:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            "rank replay requires an exact execution mode",
        )
    runtime_device = torch.device(device)
    if mode is V1SampledLiteralJFDRankReplayMode.PRODUCTION_CUDA:
        if runtime_device.type != "cuda" or not bool(torch.cuda.is_available()):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.PRODUCTION_CUDA_REQUIRED,
                "production rank replay requires an available CUDA device",
            )
        if runtime_device.index is None:
            runtime_device = torch.device("cuda", int(torch.cuda.current_device()))
    elif runtime_device.type != "cpu":
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.PRODUCTION_CUDA_REQUIRED,
            "the reference replay mode is CPU-only and carries no production promotion",
        )
    return runtime_device


def _device_identity(device: Any) -> str:
    if device.type != "cuda":
        return "torch:cpu:fp64_reference"
    index = int(device.index)
    try:
        name = str(torch.cuda.get_device_name(device))
    except (AssertionError, RuntimeError):  # pragma: no cover - device failure.
        name = "cuda-device-name-unavailable"
    return f"torch:cuda:{index}:{name}:fp64"


def replay_v1_sampled_literal_j_fd_at_physical_rank(
    *,
    attachment: V1SampledDualChartRowLivePrefixAttachment,
    original_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
    fd_design: V1StratifiedFDDesign,
    requested_physical_rank: int,
    corrected_moments: V1StratifiedCorrectedAttributionMoments,
    mode: V1SampledLiteralJFDRankReplayMode = (
        V1SampledLiteralJFDRankReplayMode.PRODUCTION_CUDA
    ),
    device: object = "cuda",
) -> V1SampledLiteralJFDRankReplayResult:
    """Replay sealed literal-J packets into one fresh rank-specific FD state.

    The default is fail-closed production CUDA.  CPU execution must be
    requested explicitly through ``HIGH_PRECISION_CPU_REFERENCE`` and is
    suitable for exact small-domain parity only.
    """

    if type(attachment) is not V1SampledDualChartRowLivePrefixAttachment:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            "rank replay requires one exact live dual-row prefix attachment",
        )
    if type(original_capture_binding) is not V1SampledLiteralJOriginalCaptureBinding:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            "rank replay requires its exact original-capture binding",
        )
    if type(fd_design) is not V1StratifiedFDDesign:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            "rank replay requires one exact stratified FD design",
        )
    if type(corrected_moments) is not V1StratifiedCorrectedAttributionMoments:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.INVALID_TYPE,
            "rank replay requires exact caller-owned corrected moments",
        )
    rank = _positive_int(requested_physical_rank, name="requested physical rank")
    runtime_device = _device_for_mode(mode=mode, device=device)
    if rank > len(fd_design.route_ids) or rank > fd_design.spec.maximum_rank:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.AXIS_MISMATCH,
            "requested physical rank exceeds the registered route/design maximum",
        )

    moments_hash_before = corrected_moments.assert_integrity()
    # Keep the pre-consumption custody gate metadata-only.  Each literal-J
    # packet's one-pass materializer below is the authoritative consuming
    # role-domain SHA-256 verification and must be the only route-scale scan.
    attachment.verify_live_witnesses()
    _assert_original_binding_matches_prefix(
        binding=original_capture_binding,
        prefix=attachment,
    )
    _assert_design_and_moments_match(
        design=fd_design,
        moments=corrected_moments,
        binding=original_capture_binding,
    )
    manifest = attachment.manifest
    looks = manifest.looks
    j_attachments = attachment.literal_j_attachments
    if len(j_attachments) != len(looks):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.ROLE_VIOLATION,
            "literal-J live attachment ledger is incomplete",
        )

    accumulator = V1StratifiedDeviceFDTrustAccumulator(
        fd_design,
        sketch_rank=rank,
        device=runtime_device,
        emit_derivation_receipt=False,
    )
    source_measurements_by_packet: list[tuple[str, ...]] = []
    payload_bytes = 0
    route_window_reads = 0
    descriptor_readinto_calls = 0
    digest_update_count = 0
    aggregate_host_matrix_bytes = 0
    aggregate_fd_append_tensor_bytes = 0
    maximum_fd_append_tensor_bytes = 0
    maximum_window_staging_bytes = 0
    maximum_host_staging_bytes = 0
    maximum_h2d_bytes = 0

    for packet_index, (look, live_j) in enumerate(
        zip(looks, j_attachments, strict=True)
    ):
        receipt = live_j.receipt
        if (
            receipt.role is not V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN
            or receipt.receipt_hash != look.literal_j_shard.receipt_hash
            or receipt.stream_identity_hash
            != original_capture_binding.literal_j_stream_identity_hash
        ):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.ROLE_VIOLATION,
                "rank replay was not given the registered literal-J shard",
            )
        verified_packet = live_j.materialize_verified_packet_one_pass()
        if (
            verified_packet.role
            is not V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN
            or verified_packet.shard_receipt_hash != receipt.receipt_hash
            or verified_packet.content_sha256 != receipt.content_sha256
            or verified_packet.context_count != len(look.context_ids)
            or verified_packet.route_count != manifest.route_count
        ):
            _refuse(
                V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
                "verified literal-J packet differs from its registered shard axes",
            )
        rows = verified_packet.rows
        packet_bytes = verified_packet.output_packet_bytes
        source_measurements = _registered_source_measurements_for_packet(
            binding=original_capture_binding,
            packet_index=packet_index,
            literal_j_shard_receipt_hash=receipt.receipt_hash,
        )
        source_measurements_by_packet.append(source_measurements)
        fd_append_rows = torch.tensor(
            rows[:, None, :],
            dtype=torch.float64,
            device=runtime_device,
        )
        fd_append_tensor_bytes = int(fd_append_rows.numel()) * _FLOAT64_BYTES
        accumulator.append_labeled_rows(
            rows=fd_append_rows,
            context_ids=look.context_ids,
            stratum_labels=look.stratum_labels,
            source_measurement_hashes=source_measurements,
            weights=None,
        )
        payload_bytes += packet_bytes
        route_window_reads += verified_packet.route_window_count
        descriptor_readinto_calls += verified_packet.descriptor_readinto_call_count
        digest_update_count += verified_packet.digest_update_count
        aggregate_host_matrix_bytes += packet_bytes
        aggregate_fd_append_tensor_bytes += fd_append_tensor_bytes
        maximum_fd_append_tensor_bytes = max(
            maximum_fd_append_tensor_bytes,
            fd_append_tensor_bytes,
        )
        maximum_window_staging_bytes = max(
            maximum_window_staging_bytes,
            verified_packet.maximum_window_staging_bytes,
        )
        maximum_host_staging_bytes = max(
            maximum_host_staging_bytes,
            verified_packet.peak_owned_host_route_bytes,
        )
        if runtime_device.type == "cpu":
            maximum_host_staging_bytes = max(
                maximum_host_staging_bytes,
                packet_bytes + fd_append_tensor_bytes,
            )
        if runtime_device.type == "cuda":
            maximum_h2d_bytes = max(maximum_h2d_bytes, packet_bytes)
        del fd_append_rows, rows, verified_packet

    snapshot = accumulator.snapshot()
    snapshot_hash = snapshot.assert_integrity()
    snapshot_host_arrays = tuple(
        array
        for _name, child in snapshot.stratum_snapshots
        for array in (
            child.active_buffer_rows,
            child.mean,
            child.weighted_second_moment_diagonal,
        )
        if array is not None
    )
    snapshot_host_bytes = sum(int(array.nbytes) for array in snapshot_host_arrays)
    maximum_snapshot_array_bytes = max(
        (int(array.nbytes) for array in snapshot_host_arrays),
        default=0,
    )
    maximum_host_staging_bytes = max(
        maximum_host_staging_bytes,
        snapshot_host_bytes,
    )
    expected_context_count = len(manifest.context_ids)
    if (
        snapshot.sketch_rank != rank
        or snapshot.design_hash != fd_design.design_hash
        or snapshot.route_axis_hash != fd_design.route_axis_hash
        or snapshot.packet_boundary_index != len(looks)
        or sum(child.context_count for _name, child in snapshot.stratum_snapshots)
        != expected_context_count
        or any(
            not math.isclose(
                child.total_weight, child.context_count, rel_tol=0.0, abs_tol=0.0
            )
            for _name, child in snapshot.stratum_snapshots
        )
    ):
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.SOURCE_PROOF_MISMATCH,
            "fresh FD snapshot does not replay packet/rank/unweighted semantics",
        )
    # The replay never passes corrected moments to a reducer or adapter.  The
    # frozen, read-only object that was verified on entry is returned directly;
    # consulting its cached identity here avoids a second route-scale hash scan.
    if corrected_moments.state_hash != moments_hash_before:
        _refuse(
            V1SampledLiteralJFDRankReplayRefusalCode.AUTHORITY_VIOLATION,
            "caller-owned corrected moments changed during literal-J replay",
        )

    flat_contexts = _flatten(original_capture_binding.packet_context_ids)
    flat_draws = _flatten(original_capture_binding.packet_draw_ids)
    flat_strata = _flatten(original_capture_binding.packet_stratum_labels)
    flat_measurements = _flatten(source_measurements_by_packet)
    packet_axis_payload = tuple(
        (
            original_capture_binding.packet_plan_hashes[index],
            original_capture_binding.packet_source_capture_identity_hashes[index],
            original_capture_binding.packet_provider_proof_hashes[index],
            original_capture_binding.packet_raw_dual_proof_hashes[index],
            look.literal_j_shard.receipt_hash,
        )
        for index, look in enumerate(looks)
    )
    cuda = runtime_device.type == "cuda"
    receipt = V1SampledLiteralJFDRankReplayReceipt(
        original_capture_binding_hash=original_capture_binding.binding_hash,
        original_outer_receipt_hash=original_capture_binding.original_outer_receipt_hash,
        row_group_plan_hash=original_capture_binding.row_group_plan_hash,
        row_prefix_manifest_hash=manifest.manifest_hash,
        literal_j_shard_receipt_hashes=tuple(
            look.literal_j_shard.receipt_hash for look in looks
        ),
        literal_j_shard_content_sha256s=tuple(
            look.literal_j_shard.content_sha256 for look in looks
        ),
        fd_design_hash=fd_design.design_hash,
        fd_snapshot_hash=snapshot_hash,
        corrected_moments_state_hash=moments_hash_before,
        requested_physical_rank=rank,
        packet_count=len(looks),
        context_count=expected_context_count,
        original_capture_batch_count=(
            original_capture_binding.original_capture_batch_count
        ),
        original_vjp_count=original_capture_binding.original_vjp_count,
        context_axis_hash=content_hash(flat_contexts, prefix=_CONTEXT_AXIS_PREFIX),
        draw_axis_hash=content_hash(flat_draws, prefix=_DRAW_AXIS_PREFIX),
        stratum_axis_hash=content_hash(flat_strata, prefix=_STRATUM_AXIS_PREFIX),
        packet_axis_hash=content_hash(packet_axis_payload, prefix=_PACKET_AXIS_PREFIX),
        source_measurement_axis_hash=content_hash(
            flat_measurements,
            prefix=_SOURCE_MEASUREMENT_AXIS_PREFIX,
        ),
        chart_identity_hash=manifest.chart_identity_hash,
        current_mask_hash=manifest.current_mask_hash,
        proposal_generation=manifest.proposal_generation,
        route_axis_hash=manifest.route_axis_hash,
        route_id_axis_hash=manifest.route_id_axis_hash,
        route_count=manifest.route_count,
        literal_j_stream_identity_hash=manifest.literal_j_stream_identity_hash,
        corrected_a_stream_identity_hash=manifest.corrected_a_stream_identity_hash,
        execution_mode=mode,
        device_identity=_device_identity(runtime_device),
        literal_j_verified_packet_read_count=len(looks),
        literal_j_route_window_read_count=route_window_reads,
        literal_j_descriptor_readinto_call_count=descriptor_readinto_calls,
        literal_j_digest_update_count=digest_update_count,
        literal_j_payload_read_bytes=payload_bytes,
        separate_literal_j_integrity_scan_count=0,
        separate_literal_j_integrity_scan_bytes=0,
        aggregate_host_file_read_bytes=payload_bytes,
        packet_host_matrix_allocation_count=len(looks),
        aggregate_packet_host_matrix_bytes=aggregate_host_matrix_bytes,
        maximum_literal_j_window_staging_bytes=maximum_window_staging_bytes,
        maximum_owned_host_staging_bytes=maximum_host_staging_bytes,
        fd_append_input_tensor_allocation_count=len(looks),
        fd_append_input_tensor_bytes=aggregate_fd_append_tensor_bytes,
        maximum_fd_append_input_tensor_bytes=maximum_fd_append_tensor_bytes,
        cpu_reference_fd_append_copy_bytes=(
            aggregate_fd_append_tensor_bytes if not cuda else 0
        ),
        h2d_transfer_count=len(looks) if cuda else 0,
        h2d_transfer_bytes=payload_bytes if cuda else 0,
        maximum_h2d_transfer_bytes=maximum_h2d_bytes,
        fd_snapshot_host_array_count=len(snapshot_host_arrays),
        fd_snapshot_host_bytes=snapshot_host_bytes,
        d2h_snapshot_transfer_count=len(snapshot_host_arrays) if cuda else 0,
        d2h_snapshot_transfer_bytes=snapshot_host_bytes if cuda else 0,
        maximum_d2h_snapshot_transfer_bytes=(
            maximum_snapshot_array_bytes if cuda else 0
        ),
        _issuance_token=_RECEIPT_ISSUANCE_TOKEN,
    )
    return V1SampledLiteralJFDRankReplayResult(
        corrected_moments=corrected_moments,
        fd_snapshot=snapshot,
        receipt=receipt,
    )


__all__ = [
    "SAMPLED_LITERAL_J_FD_RANK_REPLAY_ARITHMETIC",
    "SAMPLED_LITERAL_J_FD_RANK_REPLAY_AUTHORITY_SCOPE",
    "SAMPLED_LITERAL_J_FD_RANK_REPLAY_RECEIPT_SCHEMA",
    "SAMPLED_LITERAL_J_FD_RANK_REPLAY_RECEIPT_SCHEMA_V1",
    "SAMPLED_LITERAL_J_FD_RANK_REPLAY_RECEIPT_SCHEMA_V2",
    "SAMPLED_LITERAL_J_ORIGINAL_CAPTURE_BINDING_SCHEMA",
    "V1SampledLiteralJFDRankReplayMode",
    "V1SampledLiteralJFDRankReplayReceipt",
    "V1SampledLiteralJFDRankReplayRefusal",
    "V1SampledLiteralJFDRankReplayRefusalCode",
    "V1SampledLiteralJFDRankReplayResult",
    "V1SampledLiteralJOriginalCaptureBinding",
    "derive_v1_sampled_literal_j_cumulative_original_capture_binding_from_v5_commit",
    "derive_v1_sampled_literal_j_original_capture_binding_from_v5_commit",
    "load_v1_sampled_literal_j_fd_rank_replay_receipt",
    "replay_v1_sampled_literal_j_fd_at_physical_rank",
]
