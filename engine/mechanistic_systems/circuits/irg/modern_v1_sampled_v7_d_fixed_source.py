"""Source-issued D1-only ``D_fixed`` custody for the first honest V7 packet.

V7 must measure a fresh ``D_W2`` packet without first pretending that packet
was part of the fitted anchor.  This module therefore exposes a deliberately
narrow D1-only source seam.  It derives the five canonical D1 axes from the
still-live dual-row prefix and its original-capture binding, freezes their
exact empirical ``D_fixed`` measure, and provides immutable bounded replay
slices.  No provider, restored row bank, fresh certification row, statistical
decision, C/K/O transition, or mask-commit authority is accepted here.

The numerical source/replay adapter below consumes the same canonical plan;
callers cannot supply a second axis rendering for the block-support and
literal-J projections.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral
from typing import Any, Final, Iterator, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .contracts import content_hash
from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_literal_j_projection_cuda import (
    V1LiteralJPrivatePrimalDirectionCUDAHandle,
    V1LiteralJPrivatePrimalDirectionCUDALease,
    V1LiteralJPrivatePrimalDirectionDescriptorAxis,
)
from .modern_v1_literal_j_two_pass_d_fixed_projection_cuda import (
    V1LiteralJTwoPassDFixedProjectionDesign,
    V1LiteralJTwoPassDFixedProjectionPacketDescriptor,
    V1LiteralJTwoPassDFixedProjectionRegistration,
)
from .modern_v1_literal_j_two_pass_d_fixed_shard_replay import (
    V1LiteralJConditionedD1ShardReplaySource,
    V1LiteralJTwoPassDFixedShardReplayResult,
    _replay_v1_literal_j_two_pass_d1_only_from_shards_cpu_reference,
    replay_v1_literal_j_two_pass_d1_only_from_shards,
)
from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from .modern_v1_neutral_runtime import DeclaredRouteAxis
from .modern_v1_sampled_conditioned_d1_compact_source import (
    V1SampledConditionedD1CompactSourceResult,
    assemble_v1_sampled_conditioned_d1_compact_source,
)
from .modern_v1_sampled_dual_chart_row_custody import (
    materialize_v1_sampled_dual_chart_rows,
    V1SampledDualChartRowLivePrefixAttachment,
)
from .modern_v1_sampled_literal_j_fd_rank_replay import (
    V1SampledLiteralJOriginalCaptureBinding,
)
from .modern_v1_sampled_residual_population_measure import (
    V1SampledResidualPopulationMeasureDesign,
    V1SampledResidualPopulationWeightReceipt,
)
from .modern_v1_sampled_variable_shell_source_bridge import (
    V1SampledVariableShellBoundChunkSinkBinding,
)
from .modern_v1_signed_work_block_support_source_cuda import (
    V1SignedWorkBlockSupportCUDAExecutionDesign,
    V1SignedWorkBlockSupportExecutionMode,
    V1SignedWorkBlockSupportPreferredActionBinding,
    V1SignedWorkBlockSupportSourceCUDAAccumulator,
    V1SignedWorkBlockSupportSourceDeviceHandle,
    V1SignedWorkBlockSupportSourceRole,
    bind_v1_signed_work_block_support_preferred_action,
    register_v1_signed_work_block_support_source,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellExecutionMode,
    V1SignedWorkVariableShellSourceCUDAAccumulator,
    V1SignedWorkVariableShellSourceCUDAExecutionDesign,
    V1SignedWorkVariableShellSourceRegistration,
)

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SAMPLED_V7_D1_CANONICAL_AXIS_PLAN_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D1_canonical_axis_plan_v1"
)
SAMPLED_V7_D1_CANONICAL_AXIS_AUTHORITY: Final = (
    "live_D1_axis_and_bounded_source_replay_custody_only_no_provider_D_W2_"
    "history_stability_C_K_O_commit_terminal_or_pareto_authority_v1"
)
SAMPLED_V7_D1_ONLY_ANCHOR_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D1_only_anchor_receipt_v1"
)
SAMPLED_V7_D1_ONLY_REPLAY_PLAN_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D1_only_replay_plan_v1"
)
SAMPLED_V7_D1_FIXED_SOURCE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D1_fixed_source_receipt_v1"
)
SAMPLED_V7_COMMON_TWO_BRANCH_KKT_ID: Final = (
    "spectral_plus_diagonal_l1_common_coordinate_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-D1-fixed-source-"
_D1_PREFIX: Final = "partition-free-v1-sampled-conditioned-D1-compact-source-"
_D_W2_PREFIX: Final = "partition-free-v1-sampled-variable-shell-D-W2-suffix-"
_PLAN_ISSUER: Final = object()
_ANCHOR_ISSUER: Final = object()
_REPLAY_ISSUER: Final = object()


class V1SampledV7DFixedSourceRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    AXIS_MISMATCH = "axis_mismatch"
    ORDER_MISMATCH = "order_mismatch"
    BRANCH_MISMATCH = "branch_mismatch"
    ACTION_MISMATCH = "action_mismatch"
    MEASURE_MISMATCH = "measure_mismatch"
    PROVIDER_FORBIDDEN = "provider_forbidden"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7DFixedSourceRefusal(RuntimeError):
    """Typed fail-closed boundary for the D1-only V7 source path."""

    def __init__(
        self,
        code: V1SampledV7DFixedSourceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7DFixedSourceRefusalCode:
            raise TypeError("V7 D_fixed refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7DFixedSourceRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7DFixedSourceRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_conditioned_D1_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_D1_PREFIX + role + "-axis-",
    )


def _d1_pair_axis_hash(*, target: Sequence[str], complement: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_variable_shell_D_W2_pair_axis_v1",
            "role": "D1",
            "target": tuple(target),
            "complement": tuple(complement),
        },
        prefix=_D_W2_PREFIX + "D1-pair-axis-",
    )


def _context_owner_token(
    *,
    role: str,
    source_owner_hash: str,
    context_id: str,
    measurement_id: str,
) -> str:
    """Replay the conditioned-D1 compact ABI's per-context owner token."""

    return content_hash(
        {
            "schema": "partition_free_v1_sampled_conditioned_D1_context_owner_v1",
            "role": role,
            "source_owner_hash": source_owner_hash,
            "context_id": context_id,
            "measurement_id": measurement_id,
        },
        prefix=_D1_PREFIX + role + "-context-owner-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7D1CanonicalAxisRecord:
    """One target-then-complement D1 row and all five canonical identities."""

    canonical_ordinal: int
    shard_index: int
    row_index: int
    context_id: str
    draw_id: str
    measurement_id: str
    capture_id: str
    source_packet_hash: str
    stratum: str

    def __post_init__(self) -> None:
        for name in ("canonical_ordinal", "shard_index", "row_index"):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        for name in (
            "context_id",
            "draw_id",
            "measurement_id",
            "capture_id",
            "source_packet_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.stratum not in {"target", "complement"}:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.ORDER_MISMATCH,
                "D1 records must belong to target or complement",
            )


@dataclass(frozen=True, slots=True)
class V1SampledV7D1CanonicalShardSlice:
    """Array-free bounded gather description for one immutable source shard."""

    stratum: str
    shard_index: int
    source_rows: tuple[int, ...]
    destination_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.stratum not in {"target", "complement"}:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.ORDER_MISMATCH,
                "replay slices must use the canonical two-stratum axis",
            )
        object.__setattr__(
            self,
            "shard_index",
            _nonnegative_int(self.shard_index, name="slice shard index"),
        )
        axes: list[tuple[int, ...]] = []
        for name in ("source_rows", "destination_rows"):
            value = getattr(self, name)
            if type(value) is not tuple or not value:
                _refuse(
                    V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
                    f"{name} must be one nonempty immutable axis",
                )
            axis = tuple(
                _nonnegative_int(item, name=f"{name} item") for item in value
            )
            if len(axis) != len(set(axis)):
                _refuse(
                    V1SampledV7DFixedSourceRefusalCode.AXIS_MISMATCH,
                    f"{name} must not repeat a row",
                )
            object.__setattr__(self, name, axis)
            axes.append(axis)
        if len(axes[0]) != len(axes[1]):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.AXIS_MISMATCH,
                "source and destination rows must align",
            )


class _V1SampledV7D1PlanLease:
    __slots__ = ("_active", "_active_slice", "_next_slice")

    def __init__(self) -> None:
        self._active = True
        self._active_slice: int | None = None
        self._next_slice = 0

    def assert_active(self) -> None:
        if not self._active:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.EVICTED,
                "D1 canonical source custody was evicted or its lease expired",
            )

    def expire(self) -> None:
        if self._active_slice is not None:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.AUTHORITY_VIOLATION,
                "cannot evict D1 custody while a replay slice is borrowed",
            )
        self._active = False

    def open_slice(self, ordinal: int) -> None:
        self.assert_active()
        if self._active_slice is not None or ordinal != self._next_slice:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.ORDER_MISMATCH,
                "D1 replay slices must be borrowed exactly once in canonical order",
            )
        self._active_slice = ordinal

    def assert_slice(self, ordinal: int) -> None:
        self.assert_active()
        if self._active_slice != ordinal:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.EVICTED,
                "D1 replay slice lease is closed, stale, or noncanonical",
            )

    def close_slice(self, ordinal: int) -> None:
        self.assert_slice(ordinal)
        self._active_slice = None
        self._next_slice += 1

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("V7 D1 source leases are nonserializable")


class V1SampledV7D1ReplaySliceLease:
    """One-shot source borrow allowing bounded paired A/J reads for one slice."""

    __slots__ = (
        "_plan",
        "_ordinal",
        "slice",
        "records",
        "corrected_a_attachment",
        "literal_j_attachment",
        "_closed",
    )

    def __init__(
        self,
        *,
        plan: "V1SampledV7D1CanonicalAxisPlan",
        ordinal: int,
        replay_slice: V1SampledV7D1CanonicalShardSlice,
    ) -> None:
        plan.assert_live()
        assert plan._lease is not None
        plan._lease.open_slice(ordinal)
        try:
            assert plan._row_prefix is not None
            corrected = plan._row_prefix.corrected_a_attachments[
                replay_slice.shard_index
            ]
            literal = plan._row_prefix.literal_j_attachments[
                replay_slice.shard_index
            ]
            records = tuple(
                plan.records[index] for index in replay_slice.destination_rows
            )
            if (
                tuple(item.row_index for item in records) != replay_slice.source_rows
                or any(item.shard_index != replay_slice.shard_index for item in records)
                or any(item.stratum != replay_slice.stratum for item in records)
                or corrected.receipt.context_ids != literal.receipt.context_ids
                or corrected.receipt.draw_ids != literal.receipt.draw_ids
                or corrected.receipt.route_windows != literal.receipt.route_windows
            ):
                _refuse(
                    V1SampledV7DFixedSourceRefusalCode.STALE_SOURCE,
                    "paired A/J shard custody differs from the canonical replay slice",
                )
            corrected.verify_live_witnesses()
            literal.verify_live_witnesses()
        except BaseException:
            # Opening failed before any source value could be released.  Burn
            # the attempted slot so a partially validated source is never
            # retried under the same causal coordinate.
            plan._lease.close_slice(ordinal)
            raise
        self._plan = plan
        self._ordinal = ordinal
        self.slice = replay_slice
        self.records = records
        self.corrected_a_attachment = corrected
        self.literal_j_attachment = literal
        self._closed = False

    def assert_live(self) -> None:
        if self._closed:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.EVICTED,
                "D1 replay slice was already burned",
            )
        self._plan.assert_live()
        assert self._plan._lease is not None
        self._plan._lease.assert_slice(self._ordinal)
        self.corrected_a_attachment.verify_live_witnesses()
        self.literal_j_attachment.verify_live_witnesses()

    def close(self) -> None:
        """Revalidate the source and irreversibly burn this slice.

        The burn happens even when the final source-witness check fails.  A
        shard that became stale after either consumer pass must never leave
        the canonical ordinal open for a retry under the same causal
        coordinate.
        """

        try:
            self.assert_live()
        finally:
            self.burn()

    def burn(self) -> None:
        """Unconditionally close the exact active ordinal without rereading rows.

        This is the fail-closed lifetime primitive for consumers whose replay
        body already failed.  It is deliberately idempotent and consults only
        the private one-shot plan lease; it cannot turn stale source custody
        into valid measurement evidence.
        """

        if self._closed:
            return
        assert self._plan._lease is not None
        self._plan._lease.close_slice(self._ordinal)
        self._closed = True

    def __enter__(self) -> "V1SampledV7D1ReplaySliceLease":
        self.assert_live()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc, traceback
        if exc_type is None:
            self.close()
        else:
            # Preserve the replay-body exception while still consuming the
            # one-shot ordinal.  No source reread is required for cleanup.
            self.burn()

    def __reduce__(self) -> NoReturn:
        raise TypeError("live V7 D1 replay-slice leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live V7 D1 replay-slice leases are nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledV7D1CanonicalAxisPlan:
    """Live, issuer-only canonical D1 axis and bounded replay description."""

    row_prefix_manifest_hash: str
    original_capture_binding_hash: str
    registered_d_prefix_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    records: tuple[V1SampledV7D1CanonicalAxisRecord, ...]
    replay_slices: tuple[V1SampledV7D1CanonicalShardSlice, ...]
    context_axis_hash: str
    draw_axis_hash: str
    measurement_axis_hash: str
    capture_axis_hash: str
    source_packet_axis_hash: str
    d1_pair_prefix_axis_hash: str
    d1_stratum_counts: tuple[int, int]
    exact_D_fixed_context_weight_fraction: tuple[int, int]
    exact_D_fixed_stratum_weight_fractions: tuple[tuple[int, int], tuple[int, int]]
    provider_call_count: int = 0
    full_row_bank_restoration_count: int = 0
    fresh_D_W2_context_count: int = 0
    target_then_complement: bool = True
    exact_uniform_D_fixed_measure: bool = True
    fresh_D_W2_excluded_from_anchor_history: bool = True
    source_projection_custody_only: bool = True
    stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_D1_CANONICAL_AXIS_AUTHORITY
    schema: str = SAMPLED_V7_D1_CANONICAL_AXIS_PLAN_SCHEMA
    _row_prefix: V1SampledDualChartRowLivePrefixAttachment = field(
        repr=False,
        compare=False,
        default=None,  # type: ignore[assignment]
    )
    _original_capture_binding: V1SampledLiteralJOriginalCaptureBinding = field(
        repr=False,
        compare=False,
        default=None,  # type: ignore[assignment]
    )
    _lease: _V1SampledV7D1PlanLease | None = field(
        repr=False,
        compare=False,
        default=None,
    )
    _issuer: object = field(repr=False, compare=False, default=None)
    _plan_hash: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._issuer is not _PLAN_ISSUER:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
                "D1 canonical axis plans are issued only from live source custody",
            )
        if (
            type(self._row_prefix) is not V1SampledDualChartRowLivePrefixAttachment
            or type(self._original_capture_binding)
            is not V1SampledLiteralJOriginalCaptureBinding
            or type(self._lease) is not _V1SampledV7D1PlanLease
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
                "D1 canonical axis plan lost its exact live source owners",
            )
        for name in (
            "row_prefix_manifest_hash",
            "original_capture_binding_hash",
            "registered_d_prefix_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "measurement_axis_hash",
            "capture_axis_hash",
            "source_packet_axis_hash",
            "d1_pair_prefix_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "proposal_generation",
            "route_count",
            "provider_call_count",
            "full_row_bank_restoration_count",
            "fresh_D_W2_context_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        records = tuple(self.records)
        slices = tuple(self.replay_slices)
        if (
            not records
            or not slices
            or any(type(item) is not V1SampledV7D1CanonicalAxisRecord for item in records)
            or any(type(item) is not V1SampledV7D1CanonicalShardSlice for item in slices)
            or tuple(item.canonical_ordinal for item in records)
            != tuple(range(len(records)))
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.AXIS_MISMATCH,
                "D1 records and replay slices must form one exact nonempty cover",
            )
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "replay_slices", slices)
        counts = self.d1_stratum_counts
        if (
            type(counts) is not tuple
            or len(counts) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, Integral) or int(item) <= 0
                for item in counts
            )
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.MEASURE_MISMATCH,
                "D1 must contain positive target and complement strata",
            )
        counts = (int(counts[0]), int(counts[1]))
        object.__setattr__(self, "d1_stratum_counts", counts)
        total = len(records)
        if (
            sum(counts) != total
            or tuple(item.stratum for item in records)
            != ("target",) * counts[0] + ("complement",) * counts[1]
            or self.exact_D_fixed_context_weight_fraction != (1, total)
            or self.exact_D_fixed_stratum_weight_fractions
            != tuple(
                (Fraction(count, total).numerator, Fraction(count, total).denominator)
                for count in counts
            )
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.MEASURE_MISMATCH,
                "D1 plan does not carry the exact uniform empirical measure",
            )
        axes = (
            tuple(item.context_id for item in records),
            tuple(item.draw_id for item in records),
            tuple(item.measurement_id for item in records),
            tuple(item.capture_id for item in records),
            tuple(item.source_packet_hash for item in records),
        )
        if any(len(axis) != len(set(axis)) for axis in axes):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.AXIS_MISMATCH,
                "all five canonical D1 axes must be unique",
            )
        expected_hashes = tuple(
            _axis_hash(role, axis)
            for role, axis in zip(
                ("context", "draw", "measurement", "capture", "source_packet"),
                axes,
                strict=True,
            )
        )
        if expected_hashes != (
            self.context_axis_hash,
            self.draw_axis_hash,
            self.measurement_axis_hash,
            self.capture_axis_hash,
            self.source_packet_axis_hash,
        ) or self.d1_pair_prefix_axis_hash != _d1_pair_axis_hash(
            target=axes[0][: counts[0]],
            complement=axes[0][counts[0] :],
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.STALE_SOURCE,
                "D1 canonical axis hashes do not replay",
            )
        destination_cover = tuple(
            row for item in slices for row in item.destination_rows
        )
        if sorted(destination_cover) != list(range(total)) or len(destination_cover) != len(
            set(destination_cover)
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.AXIS_MISMATCH,
                "bounded replay slices do not exactly partition the canonical axis",
            )
        required_true = (
            self.target_then_complement,
            self.exact_uniform_D_fixed_measure,
            self.fresh_D_W2_excluded_from_anchor_history,
            self.source_projection_custody_only,
        )
        required_false = (
            self.stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(item) is not bool for item in required_true + required_false)
            or any(item is not True for item in required_true)
            or any(item is not False for item in required_false)
            or self.provider_call_count != 0
            or self.full_row_bank_restoration_count != 0
            or self.fresh_D_W2_context_count != 0
            or self.authority_scope != SAMPLED_V7_D1_CANONICAL_AXIS_AUTHORITY
            or self.schema != SAMPLED_V7_D1_CANONICAL_AXIS_PLAN_SCHEMA
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.AUTHORITY_VIOLATION,
                "D1 canonical plan crossed its zero-provider source boundary",
            )
        self.assert_live()
        object.__setattr__(self, "_plan_hash", self._compute_plan_hash())

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(item.context_id for item in self.records)

    @property
    def draw_ids(self) -> tuple[str, ...]:
        return tuple(item.draw_id for item in self.records)

    @property
    def measurement_ids(self) -> tuple[str, ...]:
        return tuple(item.measurement_id for item in self.records)

    @property
    def capture_ids(self) -> tuple[str, ...]:
        return tuple(item.capture_id for item in self.records)

    @property
    def source_packet_hashes(self) -> tuple[str, ...]:
        return tuple(item.source_packet_hash for item in self.records)

    @property
    def context_count(self) -> int:
        return len(self.records)

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def _compute_plan_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "canonical-plan-")

    @property
    def plan_hash(self) -> str:
        self.assert_live()
        observed = self._compute_plan_hash()
        if observed != self._plan_hash:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.STALE_SOURCE,
                "D1 canonical plan changed after issuance",
            )
        return observed

    def assert_live(self) -> None:
        assert self._lease is not None
        self._lease.assert_active()
        assert self._row_prefix is not None
        assert self._original_capture_binding is not None
        self._row_prefix.verify_live_witnesses()
        manifest = self._row_prefix.manifest
        original = self._original_capture_binding
        if (
            manifest.assert_integrity() != self.row_prefix_manifest_hash
            or original.assert_integrity() != self.original_capture_binding_hash
            or manifest.registered_d_prefix_hash != self.registered_d_prefix_hash
            or manifest.chart_identity_hash != self.chart_identity_hash
            or manifest.current_mask_hash != self.current_mask_hash
            or manifest.proposal_generation != self.proposal_generation
            or manifest.route_axis_hash != self.route_axis_hash
            or manifest.route_id_axis_hash != self.route_id_axis_hash
            or manifest.route_count != self.route_count
            or manifest.corrected_a_stream_identity_hash
            != self.corrected_evidence_stream_source_hash
            or manifest.literal_j_stream_identity_hash
            != self.literal_j_stream_source_hash
            or original.row_prefix_manifest_hash != self.row_prefix_manifest_hash
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.STALE_SOURCE,
                "D1 canonical plan lost its live row-prefix/original-capture identity",
            )

    def iter_replay_slices(self) -> Iterator[V1SampledV7D1CanonicalShardSlice]:
        """Yield immutable array-free slices while the source lease is live."""

        self.assert_live()
        for item in self.replay_slices:
            self.assert_live()
            yield item

    def open_replay_slice(
        self,
        replay_slice: V1SampledV7D1CanonicalShardSlice,
    ) -> V1SampledV7D1ReplaySliceLease:
        """Borrow the exact next paired A/J shard slice once.

        A consumer may perform both of its bounded passes while this lease is
        open.  Closing the lease irreversibly burns that slice; the next call
        must name the next canonical descriptor.
        """

        self.assert_live()
        if type(replay_slice) is not V1SampledV7D1CanonicalShardSlice:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
                "replay borrow requires an exact canonical slice descriptor",
            )
        matches = tuple(
            index
            for index, item in enumerate(self.replay_slices)
            if item is replay_slice
        )
        if len(matches) != 1:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.STALE_SOURCE,
                "replay slice was not issued by this live canonical plan",
            )
        return V1SampledV7D1ReplaySliceLease(
            plan=self,
            ordinal=matches[0],
            replay_slice=replay_slice,
        )

    def evict(self) -> None:
        assert self._lease is not None
        self._lease.assert_active()
        self._lease.expire()

    def __reduce__(self) -> NoReturn:
        raise TypeError("live V7 D1 canonical plans are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live V7 D1 canonical plans are nonserializable")


def derive_v1_sampled_v7_d1_canonical_axis_plan(
    *,
    row_prefix: V1SampledDualChartRowLivePrefixAttachment,
    original_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
) -> V1SampledV7D1CanonicalAxisPlan:
    """Derive the only canonical D1 axis from live same-capture source owners."""

    if (
        type(row_prefix) is not V1SampledDualChartRowLivePrefixAttachment
        or type(original_capture_binding)
        is not V1SampledLiteralJOriginalCaptureBinding
    ):
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
            "canonical D1 derivation requires exact live source-owner types",
        )
    row_prefix.verify_live_witnesses()
    manifest = row_prefix.manifest
    manifest_hash = manifest.assert_integrity()
    binding_hash = original_capture_binding.assert_integrity()
    binding = original_capture_binding
    flattened = lambda values: tuple(item for packet in values for item in packet)
    if (
        binding.row_prefix_manifest_hash != manifest_hash
        or binding.registered_d_prefix_hash != manifest.registered_d_prefix_hash
        or binding.chart_identity_hash != manifest.chart_identity_hash
        or binding.current_mask_hash != manifest.current_mask_hash
        or binding.proposal_generation != manifest.proposal_generation
        or binding.route_axis_hash != manifest.route_axis_hash
        or binding.route_id_axis_hash != manifest.route_id_axis_hash
        or binding.route_count != manifest.route_count
        or binding.corrected_a_stream_identity_hash
        != manifest.corrected_a_stream_identity_hash
        or binding.literal_j_stream_identity_hash
        != manifest.literal_j_stream_identity_hash
        or flattened(binding.packet_context_ids) != manifest.context_ids
        or flattened(binding.packet_draw_ids) != manifest.draw_ids
        or flattened(binding.packet_stratum_labels) != manifest.stratum_labels
        or len(binding.packet_context_ids) != len(manifest.looks)
    ):
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.STALE_SOURCE,
            "row prefix and original-capture binding do not name one D1 source",
        )
    raw: list[tuple[int, int, str, str, str, str, str, str]] = []
    for shard_index, look in enumerate(manifest.looks):
        contexts = binding.packet_context_ids[shard_index]
        draws = binding.packet_draw_ids[shard_index]
        strata = binding.packet_stratum_labels[shard_index]
        measurements = binding.packet_fd_source_measurement_hashes[shard_index]
        capture_owner = binding.packet_source_capture_identity_hashes[shard_index]
        packet_owner = binding.packet_plan_hashes[shard_index]
        if (
            look.context_ids != contexts
            or look.draw_ids != draws
            or look.stratum_labels != strata
            or len(measurements) != len(contexts)
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.ORDER_MISMATCH,
                "one source shard is missing, duplicated, or reordered",
            )
        for row_index, (context, draw, measurement, stratum) in enumerate(
            zip(contexts, draws, measurements, strata, strict=True)
        ):
            if stratum not in {"target", "complement"}:
                _refuse(
                    V1SampledV7DFixedSourceRefusalCode.ORDER_MISMATCH,
                    "D1 source contains a noncanonical stratum",
                )
            raw.append(
                (
                    shard_index,
                    row_index,
                    context,
                    draw,
                    measurement,
                    _context_owner_token(
                        role="capture",
                        source_owner_hash=capture_owner,
                        context_id=context,
                        measurement_id=measurement,
                    ),
                    _context_owner_token(
                        role="source_packet",
                        source_owner_hash=packet_owner,
                        context_id=context,
                        measurement_id=measurement,
                    ),
                    stratum,
                )
            )
    ordered = tuple(
        item
        for stratum in ("target", "complement")
        for item in raw
        if item[-1] == stratum
    )
    records = tuple(
        V1SampledV7D1CanonicalAxisRecord(
            canonical_ordinal=ordinal,
            shard_index=item[0],
            row_index=item[1],
            context_id=item[2],
            draw_id=item[3],
            measurement_id=item[4],
            capture_id=item[5],
            source_packet_hash=item[6],
            stratum=item[7],
        )
        for ordinal, item in enumerate(ordered)
    )
    counts = tuple(
        sum(item.stratum == stratum for item in records)
        for stratum in ("target", "complement")
    )
    if min(counts, default=0) <= 0 or len(records) != manifest.context_count:
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.AXIS_MISMATCH,
            "D1 source does not close a positive target/complement cover",
        )
    slices: list[V1SampledV7D1CanonicalShardSlice] = []
    for stratum in ("target", "complement"):
        for shard_index in range(len(manifest.looks)):
            selected = tuple(
                item
                for item in records
                if item.stratum == stratum and item.shard_index == shard_index
            )
            if selected:
                slices.append(
                    V1SampledV7D1CanonicalShardSlice(
                        stratum=stratum,
                        shard_index=shard_index,
                        source_rows=tuple(item.row_index for item in selected),
                        destination_rows=tuple(
                            item.canonical_ordinal for item in selected
                        ),
                    )
                )
    axes = (
        tuple(item.context_id for item in records),
        tuple(item.draw_id for item in records),
        tuple(item.measurement_id for item in records),
        tuple(item.capture_id for item in records),
        tuple(item.source_packet_hash for item in records),
    )
    total = len(records)
    lease = _V1SampledV7D1PlanLease()
    return V1SampledV7D1CanonicalAxisPlan(
        row_prefix_manifest_hash=manifest_hash,
        original_capture_binding_hash=binding_hash,
        registered_d_prefix_hash=manifest.registered_d_prefix_hash,
        chart_identity_hash=manifest.chart_identity_hash,
        current_mask_hash=manifest.current_mask_hash,
        proposal_generation=manifest.proposal_generation,
        route_axis_hash=manifest.route_axis_hash,
        route_id_axis_hash=manifest.route_id_axis_hash,
        route_count=manifest.route_count,
        corrected_evidence_stream_source_hash=(
            manifest.corrected_a_stream_identity_hash
        ),
        literal_j_stream_source_hash=manifest.literal_j_stream_identity_hash,
        records=records,
        replay_slices=tuple(slices),
        context_axis_hash=_axis_hash("context", axes[0]),
        draw_axis_hash=_axis_hash("draw", axes[1]),
        measurement_axis_hash=_axis_hash("measurement", axes[2]),
        capture_axis_hash=_axis_hash("capture", axes[3]),
        source_packet_axis_hash=_axis_hash("source_packet", axes[4]),
        d1_pair_prefix_axis_hash=_d1_pair_axis_hash(
            target=axes[0][: counts[0]],
            complement=axes[0][counts[0] :],
        ),
        d1_stratum_counts=(counts[0], counts[1]),
        exact_D_fixed_context_weight_fraction=(1, total),
        exact_D_fixed_stratum_weight_fractions=tuple(
            (Fraction(count, total).numerator, Fraction(count, total).denominator)
            for count in counts
        ),  # type: ignore[arg-type]
        _row_prefix=row_prefix,
        _original_capture_binding=original_capture_binding,
        _lease=lease,
        _issuer=_PLAN_ISSUER,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7D1OnlyAnchorReceipt:
    """Array-free identity of the pre-packet D1-only fixed anchor."""

    canonical_axis_plan_hash: str
    row_prefix_manifest_hash: str
    original_capture_binding_hash: str
    source_template_registration_hash: str
    source_registration_hash: str
    preferred_action_binding_hash: str
    preferred_action_hash: str
    preferred_resulting_mask_hash: str
    preferred_removed_route_index_hash: str
    preferred_selected_count: int
    private_primal_descriptor_axis_hash: str
    private_primal_source_problem_hash: str
    private_primal_source_policy_hash: str
    private_primal_source_result_hash: str
    private_primal_source_execution_receipt_hash: str
    private_primal_source_execution_design_hash: str
    context_axis_hash: str
    draw_axis_hash: str
    measurement_axis_hash: str
    capture_axis_hash: str
    source_packet_axis_hash: str
    d1_pair_prefix_axis_hash: str
    d1_stratum_counts: tuple[int, int]
    exact_D_fixed_context_weight_fraction: tuple[int, int]
    context_count: int
    route_count: int
    conditioned_d1_segment_count: int = 1
    prior_residual_fit_segment_count: int = 0
    fresh_D_W2_context_count: int = 0
    provider_call_count: int = 0
    full_row_bank_restoration_count: int = 0
    common_two_branch_registration: bool = True
    action_frozen_before_packet_outcomes: bool = True
    exact_D_fixed_measure_bound: bool = True
    fresh_D_W2_excluded_from_anchor_history: bool = True
    source_projection_custody_only: bool = True
    stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_V7_D1_ONLY_ANCHOR_SCHEMA
    _issuer: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuer is not _ANCHOR_ISSUER:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
                "V7 D1-only anchor receipts are issuer-only",
            )
        for name in (
            "canonical_axis_plan_hash",
            "row_prefix_manifest_hash",
            "original_capture_binding_hash",
            "source_template_registration_hash",
            "source_registration_hash",
            "preferred_action_binding_hash",
            "preferred_action_hash",
            "preferred_resulting_mask_hash",
            "preferred_removed_route_index_hash",
            "private_primal_descriptor_axis_hash",
            "private_primal_source_problem_hash",
            "private_primal_source_policy_hash",
            "private_primal_source_result_hash",
            "private_primal_source_execution_receipt_hash",
            "private_primal_source_execution_design_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "measurement_axis_hash",
            "capture_axis_hash",
            "source_packet_axis_hash",
            "d1_pair_prefix_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "preferred_selected_count",
            "context_count",
            "route_count",
            "conditioned_d1_segment_count",
            "prior_residual_fit_segment_count",
            "fresh_D_W2_context_count",
            "provider_call_count",
            "full_row_bank_restoration_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if (
            self.preferred_selected_count <= 0
            or self.context_count <= 0
            or self.route_count <= 0
            or self.d1_stratum_counts[0] <= 0
            or self.d1_stratum_counts[1] <= 0
            or sum(self.d1_stratum_counts) != self.context_count
            or self.exact_D_fixed_context_weight_fraction != (1, self.context_count)
            or self.conditioned_d1_segment_count != 1
            or self.prior_residual_fit_segment_count != 0
            or self.fresh_D_W2_context_count != 0
            or self.provider_call_count != 0
            or self.full_row_bank_restoration_count != 0
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.MEASURE_MISMATCH,
                "V7 D1-only anchor counts or exact measure do not close",
            )
        required_true = (
            self.common_two_branch_registration,
            self.action_frozen_before_packet_outcomes,
            self.exact_D_fixed_measure_bound,
            self.fresh_D_W2_excluded_from_anchor_history,
            self.source_projection_custody_only,
        )
        required_false = (
            self.stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(item) is not bool for item in required_true + required_false)
            or any(item is not True for item in required_true)
            or any(item is not False for item in required_false)
            or self.schema != SAMPLED_V7_D1_ONLY_ANCHOR_SCHEMA
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.AUTHORITY_VIOLATION,
                "V7 D1-only anchor crossed source/projection custody",
            )
        object.__setattr__(self, "_receipt_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "anchor-receipt-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._receipt_hash:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.STALE_SOURCE,
                "V7 D1-only anchor receipt changed after issuance",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7D1OnlyReplayPlan:
    """Live replay plan binding canonical source, action, and KKT lineage."""

    canonical_axis_plan: V1SampledV7D1CanonicalAxisPlan = field(
        repr=False,
        compare=False,
    )
    source_registration: V1SignedWorkVariableShellSourceRegistration
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding = field(
        repr=False,
        compare=False,
    )
    direction_descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis
    receipt: V1SampledV7D1OnlyAnchorReceipt
    _parent_mask: HardMask = field(repr=False, compare=False)
    _issuer: object = field(default=None, repr=False, compare=False)
    schema: str = SAMPLED_V7_D1_ONLY_REPLAY_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if (
            self._issuer is not _REPLAY_ISSUER
            or type(self.canonical_axis_plan) is not V1SampledV7D1CanonicalAxisPlan
            or type(self.source_registration)
            is not V1SignedWorkVariableShellSourceRegistration
            or type(self.preferred_action_binding)
            is not V1SignedWorkBlockSupportPreferredActionBinding
            or type(self.direction_descriptor_axis)
            is not V1LiteralJPrivatePrimalDirectionDescriptorAxis
            or type(self.receipt) is not V1SampledV7D1OnlyAnchorReceipt
            or type(self._parent_mask) is not HardMask
            or self.schema != SAMPLED_V7_D1_ONLY_REPLAY_PLAN_SCHEMA
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
                "V7 D1-only replay plans require exact issuer-owned custody",
            )
        self.assert_live()

    def assert_live(self) -> None:
        canonical = self.canonical_axis_plan
        canonical.assert_live()
        self.preferred_action_binding.assert_live()
        self.receipt.assert_integrity()
        registration = self.source_registration
        direction = self.direction_descriptor_axis
        if (
            self.receipt.canonical_axis_plan_hash != canonical.plan_hash
            or self.receipt.source_registration_hash
            != registration.registration_hash
            or self.receipt.preferred_action_binding_hash
            != self.preferred_action_binding.binding_hash
            or self.receipt.private_primal_descriptor_axis_hash
            != direction.descriptor_axis_hash
            or registration.kkt_branch_id != SAMPLED_V7_COMMON_TWO_BRANCH_KKT_ID
            or registration.kkt_branch_source_receipt_hash
            != direction.source_result_hash
            or self._parent_mask.mask_hash != canonical.current_mask_hash
            or self._parent_mask.route_axis_hash != canonical.route_axis_hash
            or self._parent_mask.route_count != canonical.route_count
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.STALE_SOURCE,
                "V7 D1-only replay custody changed after source issuance",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("live V7 D1-only replay plans are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live V7 D1-only replay plans are nonserializable")

    def evict(self) -> None:
        self.assert_live()
        self.canonical_axis_plan.evict()
        self.preferred_action_binding.evict()


def issue_v1_sampled_v7_d1_only_replay_plan(
    *,
    row_prefix: V1SampledDualChartRowLivePrefixAttachment,
    original_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
    source_registration_template: V1SignedWorkVariableShellSourceRegistration,
    parent_mask: HardMask,
    preferred_action: V1PackedRemovalAction,
    direction_descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
) -> V1SampledV7D1OnlyReplayPlan:
    """Freeze the common-coordinate D1 anchor before any fresh V7 packet."""

    exact_types = (
        (source_registration_template, V1SignedWorkVariableShellSourceRegistration),
        (parent_mask, HardMask),
        (preferred_action, V1PackedRemovalAction),
        (direction_descriptor_axis, V1LiteralJPrivatePrimalDirectionDescriptorAxis),
    )
    if any(type(value) is not expected for value, expected in exact_types):
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
            "V7 replay-plan issuance requires exact source, action, and KKT types",
        )
    canonical = derive_v1_sampled_v7_d1_canonical_axis_plan(
        row_prefix=row_prefix,
        original_capture_binding=original_capture_binding,
    )
    template = source_registration_template
    axes = (
        template.registered_context_ids,
        template.registered_draw_ids,
        template.registered_measurement_ids,
        template.registered_capture_ids,
        template.registered_source_packet_hashes,
    )
    canonical_axes = (
        canonical.context_ids,
        canonical.draw_ids,
        canonical.measurement_ids,
        canonical.capture_ids,
        canonical.source_packet_hashes,
    )
    binding_hash = original_capture_binding.binding_hash
    expected_weight = float(Fraction(1, canonical.context_count))
    shared = (
        (template.chart_identity_hash, canonical.chart_identity_hash),
        (template.current_mask_hash, canonical.current_mask_hash),
        (template.proposal_generation, canonical.proposal_generation),
        (template.route_axis_hash, canonical.route_axis_hash),
        (template.route_id_axis_hash, canonical.route_id_axis_hash),
        (template.registered_d_prefix_hash, canonical.registered_d_prefix_hash),
        (template.registered_d_w2_prefix_hash, canonical.d1_pair_prefix_axis_hash),
        (template.parent_d_source_receipt_hash, binding_hash),
        (template.same_capture_vjp_source_hash, binding_hash),
        (
            template.corrected_evidence_stream_source_hash,
            canonical.corrected_evidence_stream_source_hash,
        ),
        (
            template.literal_j_stream_source_hash,
            canonical.literal_j_stream_source_hash,
        ),
    )
    if (
        axes != canonical_axes
        or any(left != right for left, right in shared)
        or template.context_weights
        != (expected_weight,) * canonical.context_count
        or template.kkt_branch_id != SAMPLED_V7_COMMON_TWO_BRANCH_KKT_ID
        or template.kkt_branch_source_receipt_hash
        != direction_descriptor_axis.source_result_hash
        or direction_descriptor_axis.route_axis_hash != canonical.route_axis_hash
        or direction_descriptor_axis.route_count != canonical.route_count
        or parent_mask.mask_hash != canonical.current_mask_hash
        or parent_mask.route_axis_hash != canonical.route_axis_hash
        or parent_mask.route_count != canonical.route_count
        or preferred_action.removed_count <= 0
    ):
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.BRANCH_MISMATCH,
            "common KKT branch, D1 axes, frame, measure, or action parent differ",
        )
    registration_id = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D1_source_registration_id_v1",
            "canonical_axis_plan_hash": canonical.plan_hash,
            "source_template_registration_hash": template.registration_hash,
            "private_primal_descriptor_axis_hash": (
                direction_descriptor_axis.descriptor_axis_hash
            ),
            "common_two_branch_kkt_id": SAMPLED_V7_COMMON_TWO_BRANCH_KKT_ID,
            "fresh_D_W2_in_anchor": False,
        },
        prefix=_PREFIX + "common-source-registration-",
    )
    source_registration = replace(template, registration_id=registration_id)
    try:
        binding = bind_v1_signed_work_block_support_preferred_action(
            source_registration=source_registration,
            preferred_action=preferred_action,
            parent_mask=parent_mask,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise V1SampledV7DFixedSourceRefusal(
            V1SampledV7DFixedSourceRefusalCode.ACTION_MISMATCH,
            "preferred packed action is not the frozen irreversible child",
        ) from error
    parent_snapshot = HardMask(
        route_axis_hash=parent_mask.route_axis_hash,
        route_count=parent_mask.route_count,
        packed_bits=bytes(parent_mask.packed_bits),
    )
    receipt = V1SampledV7D1OnlyAnchorReceipt(
        canonical_axis_plan_hash=canonical.plan_hash,
        row_prefix_manifest_hash=canonical.row_prefix_manifest_hash,
        original_capture_binding_hash=canonical.original_capture_binding_hash,
        source_template_registration_hash=template.registration_hash,
        source_registration_hash=source_registration.registration_hash,
        preferred_action_binding_hash=binding.binding_hash,
        preferred_action_hash=binding.preferred_action_hash,
        preferred_resulting_mask_hash=binding.preferred_resulting_mask_hash,
        preferred_removed_route_index_hash=(
            binding.preferred_removed_route_index_hash
        ),
        preferred_selected_count=binding.preferred_selected_count,
        private_primal_descriptor_axis_hash=(
            direction_descriptor_axis.descriptor_axis_hash
        ),
        private_primal_source_problem_hash=(
            direction_descriptor_axis.source_problem_hash
        ),
        private_primal_source_policy_hash=direction_descriptor_axis.source_policy_hash,
        private_primal_source_result_hash=direction_descriptor_axis.source_result_hash,
        private_primal_source_execution_receipt_hash=(
            direction_descriptor_axis.source_execution_receipt_hash
        ),
        private_primal_source_execution_design_hash=(
            direction_descriptor_axis.source_execution_design_hash
        ),
        context_axis_hash=canonical.context_axis_hash,
        draw_axis_hash=canonical.draw_axis_hash,
        measurement_axis_hash=canonical.measurement_axis_hash,
        capture_axis_hash=canonical.capture_axis_hash,
        source_packet_axis_hash=canonical.source_packet_axis_hash,
        d1_pair_prefix_axis_hash=canonical.d1_pair_prefix_axis_hash,
        d1_stratum_counts=canonical.d1_stratum_counts,
        exact_D_fixed_context_weight_fraction=(1, canonical.context_count),
        context_count=canonical.context_count,
        route_count=canonical.route_count,
        _issuer=_ANCHOR_ISSUER,
    )
    return V1SampledV7D1OnlyReplayPlan(
        canonical_axis_plan=canonical,
        source_registration=source_registration,
        preferred_action_binding=binding,
        direction_descriptor_axis=direction_descriptor_axis,
        receipt=receipt,
        _parent_mask=parent_snapshot,
        _issuer=_REPLAY_ISSUER,
    )


class _V1SampledV7D1BlockSupportSink:
    """Single-segment fanout into the existing block-support accumulator."""

    __slots__ = (
        "source_registration",
        "accumulator",
        "sink_identity_hash",
        "context_cursor",
        "active_count",
        "event_count",
        "completed",
    )

    def __init__(
        self,
        *,
        source_registration: V1SignedWorkVariableShellSourceRegistration,
        accumulator: V1SignedWorkBlockSupportSourceCUDAAccumulator,
        anchor_receipt_hash: str,
    ) -> None:
        self.source_registration = source_registration
        self.accumulator = accumulator
        self.sink_identity_hash = content_hash(
            {
                "schema": "partition_free_v1_sampled_v7_D1_block_sink_v1",
                "anchor_receipt_hash": anchor_receipt_hash,
                "source_registration_hash": source_registration.registration_hash,
                "block_registration_hash": accumulator.registration.registration_hash,
            },
            prefix=_PREFIX + "block-sink-",
        )
        self.context_cursor = 0
        self.active_count = 0
        self.event_count = 0
        self.completed = False

    def _source(self, source_registration: object) -> None:
        if (
            type(source_registration) is not V1SignedWorkVariableShellSourceRegistration
            or source_registration.registration_hash
            != self.source_registration.registration_hash
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.STALE_SOURCE,
                "D1 block sink received another source registration",
            )

    def begin_bound_packet(
        self,
        *,
        source_registration: V1SignedWorkVariableShellSourceRegistration,
        context_start_ordinal: int,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        measurement_ids: Sequence[str],
        capture_ids: Sequence[str],
        source_packet_hashes: Sequence[str],
    ) -> None:
        self._source(source_registration)
        if (
            self.completed
            or self.active_count
            or context_start_ordinal != self.context_cursor
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.ORDER_MISMATCH,
                "D1 block-support packet is stale, duplicate, or reordered",
            )
        axes = tuple(
            tuple(item)
            for item in (
                context_ids,
                draw_ids,
                measurement_ids,
                capture_ids,
                source_packet_hashes,
            )
        )
        count = len(axes[0])
        if len({len(item) for item in axes}) != 1 or count <= 0:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.AXIS_MISMATCH,
                "D1 block-support packet axes do not align",
            )
        self.accumulator.begin_packet(
            context_ids=axes[0],
            draw_ids=axes[1],
            measurement_ids=axes[2],
            capture_ids=axes[3],
            source_packet_hashes=axes[4],
        )
        self.active_count = count
        self.event_count += 1

    def consume_bound_route_chunk(
        self,
        *,
        source_registration: V1SignedWorkVariableShellSourceRegistration,
        route_start: int,
        corrected_evidence_rows: Any,
        literal_j_rows: Any,
        chunk_binding: Any,
    ) -> None:
        del literal_j_rows
        self._source(source_registration)
        if not self.active_count:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.ORDER_MISMATCH,
                "D1 block-support chunk lacks an active packet",
            )
        self.accumulator.consume_route_chunk(
            route_start=route_start,
            corrected_evidence_rows=corrected_evidence_rows,
            chunk_binding=chunk_binding,
        )
        self.event_count += 1

    def end_bound_packet(
        self,
        *,
        source_registration: V1SignedWorkVariableShellSourceRegistration,
    ) -> None:
        self._source(source_registration)
        if not self.active_count:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.ORDER_MISMATCH,
                "D1 block-support packet closed without opening",
            )
        count = self.active_count
        self.accumulator.end_packet()
        self.context_cursor += count
        self.active_count = 0
        self.event_count += 1
        if self.context_cursor == self.source_registration.context_count:
            self.completed = True
        elif self.context_cursor > self.source_registration.context_count:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.ORDER_MISMATCH,
                "D1 block-support source exceeded its canonical context axis",
            )


def _fraction_binary64_enclosure(value: Fraction) -> tuple[float, float]:
    center = float(value)
    exact_center = Fraction.from_float(center)
    lower = center if exact_center <= value else math.nextafter(center, -math.inf)
    upper = center if exact_center >= value else math.nextafter(center, math.inf)
    radius = math.nextafter(max(center - lower, upper - center), math.inf)
    return center, radius


def _parent_primal_tensor(
    plan: V1SampledV7D1OnlyReplayPlan,
    *,
    declared_device: str,
) -> Any:
    if torch is None:
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
            "V7 D1 fixed-source execution requires PyTorch",
        )
    bits = np.unpackbits(
        np.frombuffer(plan._parent_mask.packed_bits, dtype=np.uint8),
        bitorder="big",
    )[: plan.canonical_axis_plan.route_count]
    host = bits.astype(np.float64, copy=True)
    return torch.from_numpy(host).to(
        device=torch.device(declared_device),
        dtype=torch.float64,
    ).contiguous()


def _register_d1_only_literal_projection(
    *,
    plan: V1SampledV7D1OnlyReplayPlan,
    conditioned: V1SampledConditionedD1CompactSourceResult,
    block_support_source: V1SignedWorkBlockSupportSourceDeviceHandle,
) -> V1LiteralJTwoPassDFixedProjectionRegistration:
    canonical = plan.canonical_axis_plan
    source = plan.source_registration
    direction = plan.direction_descriptor_axis
    conditioned_receipt = conditioned.receipt
    block_receipt = block_support_source.receipt
    fraction = (1, canonical.context_count)
    center, radius = _fraction_binary64_enclosure(Fraction(*fraction))
    same_vjp_proof = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D1_same_VJP_proof_v1",
            "anchor_receipt_hash": plan.receipt.receipt_hash,
            "conditioned_D1_receipt_hash": conditioned_receipt.receipt_hash,
            "block_support_source_receipt_hash": block_receipt.receipt_hash,
            "corrected_shard_receipt_hashes": (
                conditioned_receipt.corrected_a_shard_receipt_hashes
            ),
            "literal_shard_receipt_hashes": (
                conditioned_receipt.literal_j_shard_receipt_hashes
            ),
            "fresh_D_W2_in_anchor": False,
            "provider_calls": 0,
        },
        prefix=_PREFIX + "same-vjp-proof-",
    )
    packet = V1LiteralJTwoPassDFixedProjectionPacketDescriptor(
        packet_index=0,
        segment_index=0,
        segment_role="conditioned_D1",
        context_start=0,
        context_count=canonical.context_count,
        physical_source_registration_hash=source.registration_hash,
        registered_d_prefix_hash=source.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=source.registered_d_w2_prefix_hash,
        witness_stream_source_hash=source.witness_stream_source_hash,
        same_capture_vjp_source_hash=source.same_capture_vjp_source_hash,
        literal_j_stream_source_hash=source.literal_j_stream_source_hash,
        context_ids=canonical.context_ids,
        draw_ids=canonical.draw_ids,
        measurement_ids=canonical.measurement_ids,
        capture_ids=canonical.capture_ids,
        source_packet_hashes=canonical.source_packet_hashes,
        physical_context_weight_fraction=fraction,
        structural_context_weight_fraction=fraction,
    )
    numeric_source_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D1_numeric_source_v1",
            "anchor_receipt_hash": plan.receipt.receipt_hash,
            "block_support_source_receipt_hash": block_receipt.receipt_hash,
            "conditioned_D1_receipt_hash": conditioned_receipt.receipt_hash,
            "exact_D_fixed_context_weight_fraction": fraction,
        },
        prefix=_PREFIX + "numeric-source-",
    )
    direction_axis_hash = content_hash(
        {
            "schema": "literal_j_two_pass_direction_axis_v1",
            "route_axis_hash": canonical.route_axis_hash,
            "route_id_axis_hash": canonical.route_id_axis_hash,
            "route_count": canonical.route_count,
            "direction_ids": direction.direction_ids,
            "direction_source_hash": direction.descriptor_axis_hash,
        },
        prefix=(
            "partition-free-v1-literal-j-two-pass-D-fixed-projection-"
            "direction-axis-"
        ),
    )
    source_replay_axis_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D1_source_replay_axis_v1",
            "anchor_receipt_hash": plan.receipt.receipt_hash,
            "numeric_source_receipt_hash": numeric_source_hash,
            "packet_descriptor_hash": packet.descriptor_hash,
            "five_canonical_axis_hashes": (
                canonical.context_axis_hash,
                canonical.draw_axis_hash,
                canonical.measurement_axis_hash,
                canonical.capture_axis_hash,
                canonical.source_packet_axis_hash,
            ),
        },
        prefix=_PREFIX + "source-replay-axis-",
    )
    registration_id = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D1_projection_registration_id_v1",
            "anchor_receipt_hash": plan.receipt.receipt_hash,
            "numeric_source_receipt_hash": numeric_source_hash,
            "direction_axis_hash": direction_axis_hash,
            "same_capture_and_vjp_execution_proof_hash": same_vjp_proof,
        },
        prefix=_PREFIX + "projection-registration-",
    )
    return V1LiteralJTwoPassDFixedProjectionRegistration(
        registration_id=registration_id,
        replay_registration_receipt_hash=plan.receipt.receipt_hash,
        numeric_source_receipt_hash=numeric_source_hash,
        block_support_source_receipt_hash=block_receipt.receipt_hash,
        full_population_compact_source_receipt_hash=(
            conditioned_receipt.compact_source_receipt_hash
        ),
        source_replay_axis_hash=source_replay_axis_hash,
        model_identity_hash=source.model_identity_hash,
        chart_identity_hash=canonical.chart_identity_hash,
        current_mask_hash=canonical.current_mask_hash,
        proposal_generation=canonical.proposal_generation,
        route_axis_hash=canonical.route_axis_hash,
        route_id_axis_hash=canonical.route_id_axis_hash,
        route_count=canonical.route_count,
        registered_d_prefix_hash=canonical.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=canonical.d1_pair_prefix_axis_hash,
        final_planned_dw2_prefix_hash=canonical.d1_pair_prefix_axis_hash,
        structural_source_registration_hash=source.registration_hash,
        structural_context_axis_hash=canonical.context_axis_hash,
        structural_draw_axis_hash=canonical.draw_axis_hash,
        structural_measurement_axis_hash=canonical.measurement_axis_hash,
        structural_capture_axis_hash=canonical.capture_axis_hash,
        structural_source_packet_axis_hash=canonical.source_packet_axis_hash,
        composite_witness_stream_source_hash=source.witness_stream_source_hash,
        composite_same_capture_vjp_source_hash=source.same_capture_vjp_source_hash,
        composite_literal_j_stream_source_hash=source.literal_j_stream_source_hash,
        same_capture_and_vjp_execution_proof_hash=same_vjp_proof,
        direction_ids=direction.direction_ids,
        direction_source_hash=direction.descriptor_axis_hash,
        direction_axis_hash=direction_axis_hash,
        segment_roles=("conditioned_D1",),
        segment_context_counts=(canonical.context_count,),
        segment_uniform_weight_fractions=(fraction,),
        structural_context_count=canonical.context_count,
        structural_uniform_weight_fraction=fraction,
        context_weight_binary64_centers=(center,) * canonical.context_count,
        context_weight_binary64_radii=(radius,) * canonical.context_count,
        packets=(packet,),
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7D1FixedSourceReceipt:
    anchor_receipt_hash: str
    source_registration_hash: str
    block_support_source_receipt_hash: str
    conditioned_d1_source_receipt_hash: str
    conditioned_d1_compact_source_receipt_hash: str
    literal_j_projection_registration_hash: str
    literal_j_projection_replay_receipt_hash: str
    literal_j_projection_receipt_hash: str
    private_primal_descriptor_axis_hash: str
    exact_D_fixed_context_weight_fraction: tuple[int, int]
    context_count: int
    route_count: int
    block_sink_event_count: int
    provider_call_count: int = 0
    additional_model_forward_count: int = 0
    additional_vjp_count: int = 0
    full_row_bank_restoration_count: int = 0
    prior_residual_fit_segment_count: int = 0
    fresh_D_W2_context_count: int = 0
    immutable_shards_window_streamed: bool = True
    source_projection_custody_only: bool = True
    stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_V7_D1_FIXED_SOURCE_RECEIPT_SCHEMA
    _issuer: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuer is not _REPLAY_ISSUER:
            raise TypeError("V7 D1 fixed-source receipts are issuer-only")
        for name in (
            "anchor_receipt_hash",
            "source_registration_hash",
            "block_support_source_receipt_hash",
            "conditioned_d1_source_receipt_hash",
            "conditioned_d1_compact_source_receipt_hash",
            "literal_j_projection_registration_hash",
            "literal_j_projection_replay_receipt_hash",
            "literal_j_projection_receipt_hash",
            "private_primal_descriptor_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "context_count",
            "route_count",
            "block_sink_event_count",
            "provider_call_count",
            "additional_model_forward_count",
            "additional_vjp_count",
            "full_row_bank_restoration_count",
            "prior_residual_fit_segment_count",
            "fresh_D_W2_context_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if (
            self.context_count <= 0
            or self.route_count <= 0
            or self.block_sink_event_count <= 0
            or self.exact_D_fixed_context_weight_fraction != (1, self.context_count)
            or any(
                getattr(self, name) != 0
                for name in (
                    "provider_call_count",
                    "additional_model_forward_count",
                    "additional_vjp_count",
                    "full_row_bank_restoration_count",
                    "prior_residual_fit_segment_count",
                    "fresh_D_W2_context_count",
                )
            )
            or self.immutable_shards_window_streamed is not True
            or self.source_projection_custody_only is not True
            or any(
                getattr(self, name) is not False
                for name in (
                    "stability_authority",
                    "c_stream_authority",
                    "calibration_authority",
                    "oracle_authority",
                    "commit_authority",
                    "terminal_authority",
                    "pareto_authority",
                )
            )
            or self.schema != SAMPLED_V7_D1_FIXED_SOURCE_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.AUTHORITY_VIOLATION,
                "V7 D1 fixed-source receipt crossed its zero-provider boundary",
            )
        object.__setattr__(self, "_receipt_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "source-receipt-")

    @property
    def receipt_hash(self) -> str:
        observed = self._compute_hash()
        if observed != self._receipt_hash:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.STALE_SOURCE,
                "V7 D1 fixed-source receipt changed after issuance",
            )
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledV7D1FixedSourceResult:
    block_support_source: V1SignedWorkBlockSupportSourceDeviceHandle = field(
        repr=False,
        compare=False,
    )
    conditioned_d1_source: V1SampledConditionedD1CompactSourceResult = field(
        repr=False,
        compare=False,
    )
    literal_j_projection: V1LiteralJTwoPassDFixedShardReplayResult = field(
        repr=False,
        compare=False,
    )
    receipt: V1SampledV7D1FixedSourceReceipt

    def __post_init__(self) -> None:
        if (
            type(self.block_support_source)
            is not V1SignedWorkBlockSupportSourceDeviceHandle
            or type(self.conditioned_d1_source)
            is not V1SampledConditionedD1CompactSourceResult
            or type(self.literal_j_projection)
            is not V1LiteralJTwoPassDFixedShardReplayResult
            or type(self.receipt) is not V1SampledV7D1FixedSourceReceipt
        ):
            raise TypeError("V7 D1 fixed-source result requires exact live owners")
        if (
            self.block_support_source.receipt.receipt_hash
            != self.receipt.block_support_source_receipt_hash
            or self.conditioned_d1_source.receipt.receipt_hash
            != self.receipt.conditioned_d1_source_receipt_hash
            or self.literal_j_projection.receipt.receipt_hash
            != self.receipt.literal_j_projection_replay_receipt_hash
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.STALE_SOURCE,
                "V7 D1 fixed-source result changed its live custody",
            )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live V7 D1 fixed-source results are nonserializable")


def _execute_v1_sampled_v7_d1_fixed_source(
    *,
    replay_plan: V1SampledV7D1OnlyReplayPlan,
    residual_measure_design: V1SampledResidualPopulationMeasureDesign,
    residual_weight_receipt: V1SampledResidualPopulationWeightReceipt,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    compact_route_axis: DeclaredRouteAxis,
    variable_shell_design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
    block_support_design: V1SignedWorkBlockSupportCUDAExecutionDesign,
    declared_device: str,
    direction_custody: (
        V1LiteralJPrivatePrimalDirectionCUDAHandle
        | V1LiteralJPrivatePrimalDirectionCUDALease
        | None
    ),
    private_descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis | None,
    private_directions: Any | None,
    literal_j_design: V1LiteralJTwoPassDFixedProjectionDesign | None,
    device_total_memory_bytes: int | None,
    device_free_memory_bytes: int | None,
    current_reserved_device_bytes: int | None,
) -> V1SampledV7D1FixedSourceResult:
    if type(replay_plan) is not V1SampledV7D1OnlyReplayPlan:
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
            "D1 fixed-source execution requires one exact live replay plan",
        )
    replay_plan.assert_live()
    exact_types = (
        (residual_measure_design, V1SampledResidualPopulationMeasureDesign),
        (residual_weight_receipt, V1SampledResidualPopulationWeightReceipt),
        (native_fiber_axis_receipt, V1NativeDestinationFiberAxisReceipt),
        (compact_route_axis, DeclaredRouteAxis),
        (variable_shell_design, V1SignedWorkVariableShellSourceCUDAExecutionDesign),
        (block_support_design, V1SignedWorkBlockSupportCUDAExecutionDesign),
    )
    if any(type(value) is not expected for value, expected in exact_types):
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
            "D1 fixed-source execution inputs must use exact runtime types",
        )
    canonical = replay_plan.canonical_axis_plan
    source = replay_plan.source_registration
    if (
        residual_measure_design.d1_pair_prefix_axis_hash
        != canonical.d1_pair_prefix_axis_hash
        or residual_measure_design.d1_stratum_counts != canonical.d1_stratum_counts
        or residual_weight_receipt.measure_design_hash
        != residual_measure_design.design_hash
        or source.population_count != residual_measure_design.population_count
        or native_fiber_axis_receipt.route_axis_hash != canonical.route_axis_hash
        or native_fiber_axis_receipt.route_id_axis_hash != canonical.route_id_axis_hash
        or native_fiber_axis_receipt.route_count != canonical.route_count
        or compact_route_axis.route_axis_hash != canonical.route_axis_hash
        or compact_route_axis.route_id_axis_hash != canonical.route_id_axis_hash
        or compact_route_axis.route_count != canonical.route_count
    ):
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.MEASURE_MISMATCH,
            "D1 source, finite-population design, or route geometry differs",
        )
    if torch is None:
        raise RuntimeError("V7 D1 fixed-source execution requires PyTorch")
    cpu_reference = (
        variable_shell_design.execution_mode
        is V1SignedWorkVariableShellExecutionMode.CPU_REFERENCE_TEST_ONLY
    )
    if cpu_reference != (
        block_support_design.execution_mode
        is V1SignedWorkBlockSupportExecutionMode.CPU_REFERENCE_TEST_ONLY
    ):
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.AUTHORITY_VIOLATION,
            "variable-shell and block-support execution modes must agree",
        )
    if cpu_reference:
        if any(
            value is None
            for value in (
                device_total_memory_bytes,
                device_free_memory_bytes,
                current_reserved_device_bytes,
            )
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
                "private CPU reference execution requires synthetic memory facts",
            )
        block_memory = (
            int(device_total_memory_bytes),
            int(device_free_memory_bytes),
            int(current_reserved_device_bytes),
        )
    else:
        if any(
            value is not None
            for value in (
                device_total_memory_bytes,
                device_free_memory_bytes,
                current_reserved_device_bytes,
            )
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.AUTHORITY_VIOLATION,
                "production CUDA refuses caller-authored memory facts",
            )
        device = torch.device(declared_device)
        if device.type != "cuda" or device.index is None:
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.INVALID_TYPE,
                "production D1 fixed-source execution requires explicit CUDA",
            )
        free, total = torch.cuda.mem_get_info(device)
        block_memory = (
            int(total),
            int(free),
            int(torch.cuda.memory_reserved(device)),
        )
    frozen_primal = _parent_primal_tensor(
        replay_plan,
        declared_device=declared_device,
    )
    variable_accumulator = V1SignedWorkVariableShellSourceCUDAAccumulator(
        registration=source,
        native_fiber_axis_receipt=native_fiber_axis_receipt,
        frozen_primal=frozen_primal,
        design=variable_shell_design,
        declared_device=declared_device,
        device_total_memory_bytes=(block_memory[0] if cpu_reference else None),
        device_free_memory_bytes=(block_memory[1] if cpu_reference else None),
        current_reserved_device_bytes=(block_memory[2] if cpu_reference else None),
    )
    block_partition_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D1_single_block_partition_v1",
            "anchor_receipt_hash": replay_plan.receipt.receipt_hash,
            "segment_roles": ("conditioned_D1",),
            "context_count": canonical.context_count,
        },
        prefix=_PREFIX + "block-partition-",
    )
    block_registration = register_v1_signed_work_block_support_source(
        source_registration=source,
        source_role=V1SignedWorkBlockSupportSourceRole.D_FIXED,
        source_context_start_ordinal=0,
        context_group_id="V7:D_fixed:conditioned_D1",
        block_partition_hash=block_partition_hash,
        block_ordinal=0,
        preferred_action_binding=replay_plan.preferred_action_binding,
        context_ids=canonical.context_ids,
        draw_ids=canonical.draw_ids,
        measurement_ids=canonical.measurement_ids,
        capture_ids=canonical.capture_ids,
        source_packet_hashes=canonical.source_packet_hashes,
    )
    block_accumulator = V1SignedWorkBlockSupportSourceCUDAAccumulator(
        registration=block_registration,
        source_registration=source,
        native_fiber_axis_receipt=native_fiber_axis_receipt,
        preferred_action_binding=replay_plan.preferred_action_binding,
        design=block_support_design,
        declared_device=declared_device,
        device_total_memory_bytes=block_memory[0],
        device_free_memory_bytes=block_memory[1],
        current_reserved_device_bytes=block_memory[2],
    )
    sink = _V1SampledV7D1BlockSupportSink(
        source_registration=source,
        accumulator=block_accumulator,
        anchor_receipt_hash=replay_plan.receipt.receipt_hash,
    )
    sink_binding = V1SampledVariableShellBoundChunkSinkBinding(
        sink_identity_hash=sink.sink_identity_hash,
        sink=sink,
    )
    masses = {
        "target": canonical.d1_stratum_counts[0] / canonical.context_count,
        "complement": canonical.d1_stratum_counts[1] / canonical.context_count,
    }
    assert canonical._row_prefix is not None
    assert canonical._original_capture_binding is not None
    materialized = materialize_v1_sampled_dual_chart_rows(
        attachment=canonical._row_prefix,
        stratum_masses=masses,
    )
    conditioned = assemble_v1_sampled_conditioned_d1_compact_source(
        materialized_rows=materialized,
        original_capture_binding=canonical._original_capture_binding,
        residual_measure_design=residual_measure_design,
        residual_weight_receipt=residual_weight_receipt,
        accumulator=variable_accumulator,
        compact_route_axis=compact_route_axis,
        bound_chunk_sinks=(sink_binding,),
    )
    if not sink.completed or sink.context_cursor != canonical.context_count:
        _refuse(
            V1SampledV7DFixedSourceRefusalCode.ORDER_MISMATCH,
            "D1 block-support fanout did not consume the complete canonical source",
        )
    block_source = block_accumulator.seal()
    conditioned_literal = V1LiteralJConditionedD1ShardReplaySource(
        replay_receipt=conditioned.receipt,
        compact_source_receipt=conditioned.source_handle.receipt,
        literal_j_shards=canonical._row_prefix.literal_j_attachments,
    )
    projection_registration = _register_d1_only_literal_projection(
        plan=replay_plan,
        conditioned=conditioned,
        block_support_source=block_source,
    )
    if cpu_reference:
        if (
            direction_custody is not None
            or type(private_descriptor_axis)
            is not V1LiteralJPrivatePrimalDirectionDescriptorAxis
            or private_descriptor_axis.descriptor_axis_hash
            != replay_plan.direction_descriptor_axis.descriptor_axis_hash
            or private_directions is None
            or type(literal_j_design) is not V1LiteralJTwoPassDFixedProjectionDesign
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.AUTHORITY_VIOLATION,
                "private reference direction path is incomplete or mismatched",
            )
        projection = _replay_v1_literal_j_two_pass_d1_only_from_shards_cpu_reference(
            registration=projection_registration,
            block_support_source=block_source,
            conditioned_d1=conditioned_literal,
            descriptor_axis=private_descriptor_axis,
            directions=private_directions,
            design=literal_j_design,
        )
    else:
        if (
            private_descriptor_axis is not None
            or private_directions is not None
            or type(direction_custody)
            not in {
                V1LiteralJPrivatePrimalDirectionCUDAHandle,
                V1LiteralJPrivatePrimalDirectionCUDALease,
            }
        ):
            _refuse(
                V1SampledV7DFixedSourceRefusalCode.AUTHORITY_VIOLATION,
                "production V7 replay accepts typed CUDA direction custody only",
            )
        projection = replay_v1_literal_j_two_pass_d1_only_from_shards(
            registration=projection_registration,
            block_support_source=block_source,
            conditioned_d1=conditioned_literal,
            direction_custody=direction_custody,
            design=literal_j_design,
        )
    receipt = V1SampledV7D1FixedSourceReceipt(
        anchor_receipt_hash=replay_plan.receipt.receipt_hash,
        source_registration_hash=source.registration_hash,
        block_support_source_receipt_hash=block_source.receipt.receipt_hash,
        conditioned_d1_source_receipt_hash=conditioned.receipt.receipt_hash,
        conditioned_d1_compact_source_receipt_hash=(
            conditioned.receipt.compact_source_receipt_hash
        ),
        literal_j_projection_registration_hash=(
            projection.registration.registration_hash
        ),
        literal_j_projection_replay_receipt_hash=projection.receipt.receipt_hash,
        literal_j_projection_receipt_hash=projection.projection.receipt.receipt_hash,
        private_primal_descriptor_axis_hash=(
            replay_plan.direction_descriptor_axis.descriptor_axis_hash
        ),
        exact_D_fixed_context_weight_fraction=(1, canonical.context_count),
        context_count=canonical.context_count,
        route_count=canonical.route_count,
        block_sink_event_count=sink.event_count,
        _issuer=_REPLAY_ISSUER,
    )
    return V1SampledV7D1FixedSourceResult(
        block_support_source=block_source,
        conditioned_d1_source=conditioned,
        literal_j_projection=projection,
        receipt=receipt,
    )


def execute_v1_sampled_v7_d1_fixed_source(
    *,
    replay_plan: V1SampledV7D1OnlyReplayPlan,
    residual_measure_design: V1SampledResidualPopulationMeasureDesign,
    residual_weight_receipt: V1SampledResidualPopulationWeightReceipt,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    compact_route_axis: DeclaredRouteAxis,
    variable_shell_design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
    block_support_design: V1SignedWorkBlockSupportCUDAExecutionDesign,
    declared_device: str,
    direction_custody: (
        V1LiteralJPrivatePrimalDirectionCUDAHandle
        | V1LiteralJPrivatePrimalDirectionCUDALease
    ),
    literal_j_design: V1LiteralJTwoPassDFixedProjectionDesign | None = None,
) -> V1SampledV7D1FixedSourceResult:
    """Production CUDA D1-only source and literal-J projection execution."""

    return _execute_v1_sampled_v7_d1_fixed_source(
        replay_plan=replay_plan,
        residual_measure_design=residual_measure_design,
        residual_weight_receipt=residual_weight_receipt,
        native_fiber_axis_receipt=native_fiber_axis_receipt,
        compact_route_axis=compact_route_axis,
        variable_shell_design=variable_shell_design,
        block_support_design=block_support_design,
        declared_device=declared_device,
        direction_custody=direction_custody,
        private_descriptor_axis=None,
        private_directions=None,
        literal_j_design=literal_j_design,
        device_total_memory_bytes=None,
        device_free_memory_bytes=None,
        current_reserved_device_bytes=None,
    )


def _execute_v1_sampled_v7_d1_fixed_source_cpu_reference(
    *,
    replay_plan: V1SampledV7D1OnlyReplayPlan,
    residual_measure_design: V1SampledResidualPopulationMeasureDesign,
    residual_weight_receipt: V1SampledResidualPopulationWeightReceipt,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    compact_route_axis: DeclaredRouteAxis,
    variable_shell_design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
    block_support_design: V1SignedWorkBlockSupportCUDAExecutionDesign,
    descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
    directions: Any,
    literal_j_design: V1LiteralJTwoPassDFixedProjectionDesign,
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    current_reserved_device_bytes: int,
) -> V1SampledV7D1FixedSourceResult:
    """Small-domain test oracle; unavailable as a production fallback."""

    return _execute_v1_sampled_v7_d1_fixed_source(
        replay_plan=replay_plan,
        residual_measure_design=residual_measure_design,
        residual_weight_receipt=residual_weight_receipt,
        native_fiber_axis_receipt=native_fiber_axis_receipt,
        compact_route_axis=compact_route_axis,
        variable_shell_design=variable_shell_design,
        block_support_design=block_support_design,
        declared_device="cpu",
        direction_custody=None,
        private_descriptor_axis=descriptor_axis,
        private_directions=directions,
        literal_j_design=literal_j_design,
        device_total_memory_bytes=device_total_memory_bytes,
        device_free_memory_bytes=device_free_memory_bytes,
        current_reserved_device_bytes=current_reserved_device_bytes,
    )


__all__ = [
    "SAMPLED_V7_COMMON_TWO_BRANCH_KKT_ID",
    "SAMPLED_V7_D1_CANONICAL_AXIS_AUTHORITY",
    "SAMPLED_V7_D1_CANONICAL_AXIS_PLAN_SCHEMA",
    "SAMPLED_V7_D1_FIXED_SOURCE_RECEIPT_SCHEMA",
    "SAMPLED_V7_D1_ONLY_ANCHOR_SCHEMA",
    "SAMPLED_V7_D1_ONLY_REPLAY_PLAN_SCHEMA",
    "V1SampledV7D1CanonicalAxisPlan",
    "V1SampledV7D1CanonicalAxisRecord",
    "V1SampledV7D1CanonicalShardSlice",
    "V1SampledV7D1FixedSourceReceipt",
    "V1SampledV7D1FixedSourceResult",
    "V1SampledV7D1OnlyAnchorReceipt",
    "V1SampledV7D1OnlyReplayPlan",
    "V1SampledV7D1ReplaySliceLease",
    "V1SampledV7DFixedSourceRefusal",
    "V1SampledV7DFixedSourceRefusalCode",
    "derive_v1_sampled_v7_d1_canonical_axis_plan",
    "execute_v1_sampled_v7_d1_fixed_source",
    "issue_v1_sampled_v7_d1_only_replay_plan",
]
