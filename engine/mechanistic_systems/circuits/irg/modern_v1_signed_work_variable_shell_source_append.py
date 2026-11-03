"""Authority-inert append joint for sealed variable-shell source packets.

``V1SignedWorkVariableShellSourceCUDAAccumulator`` deliberately registers a
fixed packet prefix before allocating its compact tensors.  Adaptive ``D_W2``
acquisition, however, must be able to stop after any sealed packet without
guessing the terminal prefix length.  This module supplies the missing seam:

* the next packet axes and exact measure quanta are frozen before its source
  handle is presented;
* every packet remains an independently issued compact source handle;
* cumulative F/Q values are the exact measure-weighted combination of the
  packet enclosures; and
* no context-by-route bank, route-scale host transfer, or replacement source
  receipt is created.

The joint is a process-local source owner only.  Its receipts cannot authorize
uniform confidence, action invariance, C/K/O, a mask commit, or a Pareto claim.
The caller must use one common integer measure unit across all packets.  This
preserves registered stratum weights exactly: packet-local normalized weights
are checked against ``measure_quanta / packet_measure_quanta``, while packet
evaluations are combined with the unnormalized packet measure.  No empirical
or packet-size renormalization is permitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral
from typing import Final, NoReturn, Sequence

from .contracts import content_hash
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceDeviceHandle,
    V1SignedWorkVariableShellSourceReceipt,
    V1SignedWorkVariableShellSourceRefusal,
    V1SignedWorkVariableShellSourceTensorLease,
    v1_variable_shell_population_axis_hash,
)

try:  # pragma: no cover - availability depends on the runtime image.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


VARIABLE_SHELL_SOURCE_APPEND_LINEAGE_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_append_lineage_v1"
)
VARIABLE_SHELL_SOURCE_APPEND_PACKET_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_append_packet_v2"
)
VARIABLE_SHELL_SOURCE_APPEND_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_append_receipt_v2"
)
VARIABLE_SHELL_SOURCE_APPEND_ABORT_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_append_abort_receipt_v1"
)
VARIABLE_SHELL_SOURCE_APPEND_EVALUATION_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_append_evaluation_v1"
)
VARIABLE_SHELL_SOURCE_APPEND_SEGMENT_LEASE_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_append_segment_lease_v2"
)
VARIABLE_SHELL_SOURCE_APPEND_SEGMENTED_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_append_segmented_receipt_v2"
)
VARIABLE_SHELL_SOURCE_APPEND_ALGORITHM: Final = (
    "predictable_packet_registration_then_compact_source_append_with_exact_"
    "measure_quanta_unique_custody_ids_repeatable_logical_strata_and_outward_"
    "F_Q_enclosure_v2"
)
VARIABLE_SHELL_SOURCE_APPEND_AUTHORITY_SCOPE: Final = (
    "adaptive_D_W2_compact_source_append_and_sample_function_evaluation_only_"
    "no_uniform_confidence_solution_sensitivity_action_invariance_C_K_O_"
    "commit_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-variable-shell-source-append-"
_SOURCE_AXIS_PREFIX: Final = "partition-free-v1-variable-shell-source-axis-"
_GENESIS_RECEIPT_HASH: Final = "partition-free-v1-variable-shell-append-genesis-v2"
_ISSUER_TOKEN: Final = object()


class V1SignedWorkVariableShellSourceAppendRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    STALE_IDENTITY = "stale_identity"
    AXIS_MISMATCH = "axis_mismatch"
    ORDER_MISMATCH = "order_mismatch"
    AXIS_OVERLAP = "axis_overlap"
    WEIGHT_MISMATCH = "weight_mismatch"
    DUPLICATE_PACKET = "duplicate_packet"
    ACTIVE_REGISTRATION = "active_registration"
    MUTATED_CUSTODY = "mutated_custody"
    EVICTED = "evicted"
    EMPTY_SOURCE = "empty_source"
    AUTHORITY_VIOLATION = "authority_violation"
    NUMERICAL_REFUSAL = "numerical_refusal"


class V1SignedWorkVariableShellSourceAppendRefusal(RuntimeError):
    """Typed fail-closed refusal for adaptive compact-source append."""

    def __init__(
        self,
        code: V1SignedWorkVariableShellSourceAppendRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkVariableShellSourceAppendRefusalCode:
            raise TypeError("append refusal code must have exact enum type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkVariableShellSourceAppendRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkVariableShellSourceAppendRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered string sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result:
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_MISMATCH,
            f"{name} cannot be empty",
        )
    if len(result) != len(set(result)):
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_OVERLAP,
            f"{name} must be unique within one packet",
        )
    return result


def _label_axis(value: object, *, name: str) -> tuple[str, ...]:
    """Validate an aligned categorical axis whose labels may repeat."""

    if type(value) not in {tuple, list}:
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered string sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result:
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_MISMATCH,
            f"{name} cannot be empty",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_DESIGN,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_DESIGN,
            f"{name} must be positive",
        )
    return result


def _measure_quanta(value: object, *, count: int) -> tuple[int, ...]:
    if type(value) not in {tuple, list} or len(value) != count:
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_MISMATCH,
            "measure quanta must align with the packet context axis",
        )
    result = tuple(
        _nonnegative_int(item, name="context measure quantum") for item in value
    )
    if sum(result) <= 0:
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_DESIGN,
            "every packet must carry positive total registered measure",
        )
    return result


def _source_axis_hash(role: str, values: Sequence[str]) -> str:
    """Replay the public source receipt's registered-axis content address."""

    return content_hash(
        {
            "schema": "partition_free_v1_variable_shell_registered_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_SOURCE_AXIS_PREFIX,
    )


def _fraction_down(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.NUMERICAL_REFUSAL,
            "cumulative evaluation overflowed binary64",
        )
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    return math.nextafter(result, -math.inf)


def _fraction_up(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.NUMERICAL_REFUSAL,
            "cumulative evaluation overflowed binary64",
        )
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return math.nextafter(result, math.inf)


def _fraction_nearest(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkVariableShellSourceAppendRefusalCode.NUMERICAL_REFUSAL,
            "cumulative evaluation overflowed binary64",
        )
    return result


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceAppendLineage:
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    registered_d_prefix_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    shell_native_fiber_indices: tuple[int, ...]
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    kkt_branch_id: str
    kkt_branch_source_receipt_hash: str
    resource_axis_hash: str
    resource_source_receipt_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    design_hash: str
    population_count: int
    population_context_axis_hash: str
    population_draw_axis_hash: str
    route_count: int
    fiber_count: int
    cut_count: int
    predictable_packet_order_source_hash: str
    stratum_measure_source_hash: str
    common_integer_measure_unit_hash: str
    schema: str = VARIABLE_SHELL_SOURCE_APPEND_LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "shell_freeze_receipt_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "kkt_branch_id",
            "kkt_branch_source_receipt_hash",
            "resource_axis_hash",
            "resource_source_receipt_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "design_hash",
            "population_context_axis_hash",
            "population_draw_axis_hash",
            "predictable_packet_order_source_hash",
            "stratum_measure_source_hash",
            "common_integer_measure_unit_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        for name in ("population_count", "route_count", "fiber_count", "cut_count"):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        if (
            type(self.shell_indices) is not tuple
            or type(self.shell_native_fiber_indices) is not tuple
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "shell index axes must be immutable tuples",
            )
        indices = tuple(
            _nonnegative_int(item, name="shell route index")
            for item in self.shell_indices
        )
        fibers = tuple(
            _nonnegative_int(item, name="shell native-fiber index")
            for item in self.shell_native_fiber_indices
        )
        route_ids = _identity_axis(self.shell_route_ids, name="shell route IDs")
        if not (
            1 <= len(indices) <= 8
            and len(indices) == len(route_ids) == len(fibers)
            and len(indices) == len(set(indices))
            and all(item < self.route_count for item in indices)
            and all(item < self.fiber_count for item in fibers)
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_MISMATCH,
                "bounded shell axes are misaligned or outside the registered geometry",
            )
        object.__setattr__(self, "shell_indices", indices)
        object.__setattr__(self, "shell_route_ids", route_ids)
        object.__setattr__(self, "shell_native_fiber_indices", fibers)
        if self.schema != VARIABLE_SHELL_SOURCE_APPEND_LINEAGE_SCHEMA:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_DESIGN,
                "unsupported append lineage schema",
            )

    @property
    def lineage_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "lineage-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceAppendPacketRegistration:
    lineage_hash: str
    parent_append_receipt_hash: str
    packet_ordinal: int
    context_start_ordinal: int
    registered_d_w2_prefix_hash: str
    registered_context_ids: tuple[str, ...]
    registered_draw_ids: tuple[str, ...]
    registered_measurement_ids: tuple[str, ...]
    registered_capture_ids: tuple[str, ...]
    registered_source_packet_hashes: tuple[str, ...]
    registered_stratum_ids: tuple[str, ...]
    context_measure_quanta: tuple[int, ...]
    registered_logical_stratum_ids: tuple[str, ...] = ()
    packet_registered_before_source_outcomes: bool = True
    predictable_order_source_bound: bool = True
    common_integer_measure_unit_bound: bool = True
    schema: str = VARIABLE_SHELL_SOURCE_APPEND_PACKET_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "append packet registrations must be issued by the live owner",
            )
        for name in (
            "lineage_hash",
            "parent_append_receipt_hash",
            "registered_d_w2_prefix_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "packet_ordinal",
            _nonnegative_int(self.packet_ordinal, name="packet ordinal"),
        )
        object.__setattr__(
            self,
            "context_start_ordinal",
            _nonnegative_int(self.context_start_ordinal, name="context start ordinal"),
        )
        axes: dict[str, tuple[str, ...]] = {}
        for name in (
            "registered_context_ids",
            "registered_draw_ids",
            "registered_measurement_ids",
            "registered_capture_ids",
            "registered_source_packet_hashes",
            "registered_stratum_ids",
        ):
            axis = _identity_axis(getattr(self, name), name=name)
            object.__setattr__(self, name, axis)
            axes[name] = axis
        logical = (
            self.registered_stratum_ids
            if not self.registered_logical_stratum_ids
            else _label_axis(
                self.registered_logical_stratum_ids,
                name="registered logical stratum IDs",
            )
        )
        object.__setattr__(self, "registered_logical_stratum_ids", logical)
        axes["registered_logical_stratum_ids"] = logical
        if len({len(axis) for axis in axes.values()}) != 1:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_MISMATCH,
                "all append packet axes must have one common length",
            )
        object.__setattr__(
            self,
            "context_measure_quanta",
            _measure_quanta(
                self.context_measure_quanta,
                count=len(self.registered_context_ids),
            ),
        )
        flags = (
            self.packet_registered_before_source_outcomes,
            self.predictable_order_source_bound,
            self.common_integer_measure_unit_bound,
        )
        if any(type(item) is not bool or item is not True for item in flags):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_DESIGN,
                "packet predictability and common measure-unit bindings are mandatory",
            )
        if self.schema != VARIABLE_SHELL_SOURCE_APPEND_PACKET_SCHEMA:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_DESIGN,
                "unsupported append packet schema",
            )

    @property
    def context_count(self) -> int:
        return len(self.registered_context_ids)

    @property
    def packet_measure_quanta(self) -> int:
        return sum(self.context_measure_quanta)

    @property
    def registration_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "packet-registration-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceAppendAbortReceipt:
    """Proof that one exact pending reservation was cancelled pre-outcome."""

    lineage_hash: str
    aborted_packet_registration_hash: str
    retained_segment_count: int
    retained_context_count: int
    source_outcomes_seen: bool = False
    alpha_coordinate_recycled: bool = False
    append_or_decision_authority: bool = False
    schema: str = VARIABLE_SHELL_SOURCE_APPEND_ABORT_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            raise PermissionError("append abort receipts are owner-issued only")
        _identity(self.lineage_hash, name="append abort lineage hash")
        _identity(
            self.aborted_packet_registration_hash,
            name="aborted packet registration hash",
        )
        object.__setattr__(
            self,
            "retained_segment_count",
            _nonnegative_int(self.retained_segment_count, name="retained segment count"),
        )
        object.__setattr__(
            self,
            "retained_context_count",
            _nonnegative_int(self.retained_context_count, name="retained context count"),
        )
        if (
            self.source_outcomes_seen is not False
            or self.alpha_coordinate_recycled is not False
            or self.append_or_decision_authority is not False
            or self.schema != VARIABLE_SHELL_SOURCE_APPEND_ABORT_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AUTHORITY_VIOLATION,
                "append abort receipt crossed its pre-outcome rollback boundary",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "abort-receipt-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceAppendReceipt:
    lineage_hash: str
    parent_append_receipt_hash: str
    appended_packet_registration_hash: str
    appended_source_receipt_hash: str
    packet_registration_axis_hash: str
    source_receipt_axis_hash: str
    context_axis_hash: str
    draw_axis_hash: str
    measurement_axis_hash: str
    capture_axis_hash: str
    source_packet_axis_hash: str
    stratum_axis_hash: str
    measure_axis_hash: str
    source_chain_hash: str
    segment_count: int
    context_count: int
    population_count: int
    cumulative_measure_quanta: int
    retained_tensor_bytes: int
    finite_census_axis_custody_complete: bool
    finite_census_complete: bool
    predictable_packet_prefix_complete: bool = True
    exact_common_measure_unit_preserved: bool = True
    source_handles_retained_without_tensor_merge: bool = True
    source_issued: bool = True
    route_scale_rows_retained: bool = False
    route_scale_d2h_performed: bool = False
    statistical_uniform_confidence_authority: bool = False
    solution_sensitivity_authority: bool = False
    normal_cone_authority: bool = False
    action_invariance_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = VARIABLE_SHELL_SOURCE_APPEND_AUTHORITY_SCOPE
    schema: str = VARIABLE_SHELL_SOURCE_APPEND_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "append receipts must be issued by the live owner",
            )
        for name in (
            "lineage_hash",
            "parent_append_receipt_hash",
            "appended_packet_registration_hash",
            "appended_source_receipt_hash",
            "packet_registration_axis_hash",
            "source_receipt_axis_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "measurement_axis_hash",
            "capture_axis_hash",
            "source_packet_axis_hash",
            "stratum_axis_hash",
            "measure_axis_hash",
            "source_chain_hash",
        ):
            _identity(getattr(self, name), name=name)
        for name in (
            "segment_count",
            "context_count",
            "population_count",
            "cumulative_measure_quanta",
            "retained_tensor_bytes",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        if self.context_count > self.population_count:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_MISMATCH,
                "appended prefix exceeds the registered population",
            )
        if self.finite_census_complete is not (
            self.context_count == self.population_count
            and self.finite_census_axis_custody_complete
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AUTHORITY_VIOLATION,
                "finite-census marker does not replay count and axis custody",
            )
        required_true = (
            self.predictable_packet_prefix_complete,
            self.exact_common_measure_unit_preserved,
            self.source_handles_retained_without_tensor_merge,
            self.source_issued,
        )
        closed = (
            self.route_scale_rows_retained,
            self.route_scale_d2h_performed,
            self.statistical_uniform_confidence_authority,
            self.solution_sensitivity_authority,
            self.normal_cone_authority,
            self.action_invariance_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(
            type(item) is not bool
            for item in required_true
            + closed
            + (
                self.finite_census_axis_custody_complete,
                self.finite_census_complete,
            )
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "append receipt authority markers must be exact booleans",
            )
        if any(item is not True for item in required_true) or any(
            item is not False for item in closed
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AUTHORITY_VIOLATION,
                "append receipt leaked route custody or downstream authority",
            )
        if (
            self.authority_scope != VARIABLE_SHELL_SOURCE_APPEND_AUTHORITY_SCOPE
            or self.schema != VARIABLE_SHELL_SOURCE_APPEND_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_DESIGN,
                "unsupported append receipt semantics",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "receipt-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceAppendEvaluationReceipt:
    append_receipt_hash: str
    source_chain_hash: str
    shell_gate_hash: str
    shell_gates: tuple[float, ...]
    segment_evaluation_receipt_hashes: tuple[str, ...]
    segment_measure_quanta: tuple[int, ...]
    cumulative_measure_quanta: int
    f_lower: float
    f_center: float
    f_upper: float
    q_lower: float
    q_center: float
    q_upper: float
    finite_census_complete: bool
    sample_function_evaluation_only: bool = True
    population_function_exact_except_numerical_enclosure: bool = False
    source_issued: bool = True
    statistical_uniform_confidence_authority: bool = False
    solution_sensitivity_authority: bool = False
    normal_cone_authority: bool = False
    action_invariance_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = VARIABLE_SHELL_SOURCE_APPEND_AUTHORITY_SCOPE
    schema: str = VARIABLE_SHELL_SOURCE_APPEND_EVALUATION_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "append evaluation receipts must be issued by the live owner",
            )
        for name in ("append_receipt_hash", "source_chain_hash", "shell_gate_hash"):
            _identity(getattr(self, name), name=name)
        if (
            type(self.shell_gates) is not tuple
            or not self.shell_gates
            or any(
                type(item) is not float or not math.isfinite(item)
                for item in self.shell_gates
            )
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "evaluation gates must be one finite immutable binary64 tuple",
            )
        if (
            type(self.segment_evaluation_receipt_hashes) is not tuple
            or type(self.segment_measure_quanta) is not tuple
            or not self.segment_evaluation_receipt_hashes
            or len(self.segment_evaluation_receipt_hashes)
            != len(self.segment_measure_quanta)
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_MISMATCH,
                "segment evaluation identities and measures must align",
            )
        for item in self.segment_evaluation_receipt_hashes:
            _identity(item, name="segment evaluation receipt hash")
        masses = tuple(
            _positive_int(item, name="segment measure quanta")
            for item in self.segment_measure_quanta
        )
        object.__setattr__(self, "segment_measure_quanta", masses)
        total = _positive_int(
            self.cumulative_measure_quanta, name="cumulative measure quanta"
        )
        object.__setattr__(self, "cumulative_measure_quanta", total)
        if sum(masses) != total:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.WEIGHT_MISMATCH,
                "segment measures do not replay cumulative measure",
            )
        for lower_name, center_name, upper_name in (
            ("f_lower", "f_center", "f_upper"),
            ("q_lower", "q_center", "q_upper"),
        ):
            lower = getattr(self, lower_name)
            center = getattr(self, center_name)
            upper = getattr(self, upper_name)
            if not (
                type(lower) is float
                and type(center) is float
                and type(upper) is float
                and math.isfinite(lower)
                and math.isfinite(center)
                and math.isfinite(upper)
                and 0.0 <= lower <= center <= upper
            ):
                _refuse(
                    V1SignedWorkVariableShellSourceAppendRefusalCode.NUMERICAL_REFUSAL,
                    "cumulative F/Q interval is nonfinite or not outward ordered",
                )
        required = (
            self.sample_function_evaluation_only,
            self.source_issued,
        )
        closed = (
            self.statistical_uniform_confidence_authority,
            self.solution_sensitivity_authority,
            self.normal_cone_authority,
            self.action_invariance_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        flags = (
            required
            + closed
            + (
                self.finite_census_complete,
                self.population_function_exact_except_numerical_enclosure,
            )
        )
        if any(type(item) is not bool for item in flags):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "evaluation authority markers must be exact booleans",
            )
        if (
            any(item is not True for item in required)
            or any(item is not False for item in closed)
            or self.population_function_exact_except_numerical_enclosure
            is not self.finite_census_complete
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AUTHORITY_VIOLATION,
                "append evaluation leaked statistical or downstream authority",
            )
        if (
            self.authority_scope != VARIABLE_SHELL_SOURCE_APPEND_AUTHORITY_SCOPE
            or self.schema != VARIABLE_SHELL_SOURCE_APPEND_EVALUATION_SCHEMA
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_DESIGN,
                "unsupported append evaluation semantics",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "evaluation-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceAppendSegmentedReceipt:
    """Source-issued compact protocol view over one sealed adaptive prefix.

    This is deliberately not a ``V1SignedWorkVariableShellSourceReceipt``.
    It commits the ordered source chain and exact common-measure axis while
    leaving the packet tensors segmented.  Consumers must use the owner-issued
    segment lease below; they may not reinterpret this receipt as one forged
    final-prefix source packet.
    """

    append_receipt_hash: str
    source_chain_hash: str
    packet_registration_axis_hash: str
    source_receipt_axis_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    shell_native_fiber_indices: tuple[int, ...]
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    kkt_branch_id: str
    kkt_branch_source_receipt_hash: str
    resource_axis_hash: str
    resource_source_receipt_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    design_hash: str
    population_context_axis_hash: str
    population_draw_axis_hash: str
    context_axis_hash: str
    draw_axis_hash: str
    measurement_axis_hash: str
    capture_axis_hash: str
    source_packet_axis_hash: str
    context_count: int
    population_count: int
    route_count: int
    fiber_count: int
    cut_count: int
    segment_count: int
    cumulative_measure_quanta: int
    retained_tensor_bytes: int
    source_replay_hash: str
    finite_census_axis_custody_complete: bool
    finite_census_complete: bool
    segmented_compact_source: bool = True
    common_integer_measure_unit_preserved: bool = True
    proposal_fit_source_only: bool = True
    exact_d1_join_required: bool = True
    full_population_action_witness_authority: bool = False
    final_prefix_packet_replay_performed: bool = False
    route_scale_rows_retained: bool = False
    route_scale_d2h_performed: bool = False
    sample_function_source_only: bool = True
    statistical_uniform_confidence_authority: bool = False
    action_invariance_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = VARIABLE_SHELL_SOURCE_APPEND_AUTHORITY_SCOPE
    schema: str = VARIABLE_SHELL_SOURCE_APPEND_SEGMENTED_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "segmented source receipts must be issued by the append owner",
            )
        for name in (
            "append_receipt_hash",
            "source_chain_hash",
            "packet_registration_axis_hash",
            "source_receipt_axis_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "kkt_branch_id",
            "kkt_branch_source_receipt_hash",
            "resource_axis_hash",
            "resource_source_receipt_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "design_hash",
            "population_context_axis_hash",
            "population_draw_axis_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "measurement_axis_hash",
            "capture_axis_hash",
            "source_packet_axis_hash",
            "source_replay_hash",
        ):
            _identity(getattr(self, name), name=name)
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        for name in (
            "context_count",
            "population_count",
            "route_count",
            "fiber_count",
            "cut_count",
            "segment_count",
            "cumulative_measure_quanta",
            "retained_tensor_bytes",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        indices = tuple(
            _nonnegative_int(item, name="shell route index")
            for item in self.shell_indices
        )
        fibers = tuple(
            _nonnegative_int(item, name="shell native-fiber index")
            for item in self.shell_native_fiber_indices
        )
        route_ids = _identity_axis(self.shell_route_ids, name="shell route IDs")
        if (
            not 1 <= len(indices) <= 8
            or len(indices) != len(route_ids)
            or len(indices) != len(fibers)
            or len(indices) != len(set(indices))
            or any(item >= self.route_count for item in indices)
            or any(item >= self.fiber_count for item in fibers)
            or self.context_count > self.population_count
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_MISMATCH,
                "segmented source axes do not close",
            )
        object.__setattr__(self, "shell_indices", indices)
        object.__setattr__(self, "shell_route_ids", route_ids)
        object.__setattr__(self, "shell_native_fiber_indices", fibers)
        required_true = (
            self.segmented_compact_source,
            self.common_integer_measure_unit_preserved,
            self.sample_function_source_only,
            self.proposal_fit_source_only,
            self.exact_d1_join_required,
        )
        closed = (
            self.final_prefix_packet_replay_performed,
            self.full_population_action_witness_authority,
            self.route_scale_rows_retained,
            self.route_scale_d2h_performed,
            self.statistical_uniform_confidence_authority,
            self.action_invariance_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        flags = (
            required_true
            + closed
            + (
                self.finite_census_axis_custody_complete,
                self.finite_census_complete,
            )
        )
        if any(type(item) is not bool for item in flags):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "segmented source markers must be exact booleans",
            )
        if any(item is not True for item in required_true) or any(
            item is not False for item in closed
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AUTHORITY_VIOLATION,
                "segmented source receipt leaked replay or downstream authority",
            )
        if self.finite_census_complete is not (
            self.context_count == self.population_count
            and self.finite_census_axis_custody_complete
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AUTHORITY_VIOLATION,
                "segmented finite-census marker does not replay axis custody",
            )
        if (
            self.authority_scope != VARIABLE_SHELL_SOURCE_APPEND_AUTHORITY_SCOPE
            or self.schema != VARIABLE_SHELL_SOURCE_APPEND_SEGMENTED_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_DESIGN,
                "unsupported segmented source semantics",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "segmented-receipt-",
        )


class V1SignedWorkVariableShellSourceAppendBorrowedSegment:
    """One immutable segment descriptor backed by a live tensor lease."""

    __slots__ = (
        "source_receipt",
        "segment_measure_quanta",
        "context_measure_quanta",
        "stratum_ids",
        "logical_stratum_ids",
        "tensor_lease",
        "_identity_hash",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        source_receipt: V1SignedWorkVariableShellSourceReceipt,
        segment_measure_quanta: int,
        context_measure_quanta: tuple[int, ...],
        stratum_ids: tuple[str, ...],
        logical_stratum_ids: tuple[str, ...],
        tensor_lease: V1SignedWorkVariableShellSourceTensorLease,
    ) -> None:
        if (
            type(source_receipt) is not V1SignedWorkVariableShellSourceReceipt
            or type(tensor_lease) is not V1SignedWorkVariableShellSourceTensorLease
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "borrowed segments require exact source receipts and tensor leases",
            )
        mass = _positive_int(segment_measure_quanta, name="segment measure quanta")
        quanta = tuple(
            _nonnegative_int(item, name="context measure quantum")
            for item in context_measure_quanta
        )
        strata = tuple(
            _identity(item, name="segment stratum id") for item in stratum_ids
        )
        logical_strata = _label_axis(
            logical_stratum_ids,
            name="segment logical stratum IDs",
        )
        if (
            len(quanta) != source_receipt.context_count
            or len(strata) != source_receipt.context_count
            or len(logical_strata) != source_receipt.context_count
            or sum(quanta) != mass
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.WEIGHT_MISMATCH,
                "borrowed segment context quanta do not replay its mass",
            )
        self.source_receipt = source_receipt
        self.segment_measure_quanta = mass
        self.context_measure_quanta = quanta
        self.stratum_ids = strata
        self.logical_stratum_ids = logical_strata
        self.tensor_lease = tensor_lease
        self._identity_hash = content_hash(
            {
                "source_receipt_hash": source_receipt.receipt_hash,
                "segment_measure_quanta": mass,
                "context_measure_quanta": quanta,
                "stratum_ids": strata,
                "logical_stratum_ids": logical_strata,
                "tensor_lease_hash": tensor_lease.lease_hash,
            },
            prefix=_PREFIX + "borrowed-segment-identity-",
        )

    def compact_tensors(self) -> tuple[object, ...]:
        current = content_hash(
            {
                "source_receipt_hash": self.source_receipt.receipt_hash,
                "segment_measure_quanta": self.segment_measure_quanta,
                "context_measure_quanta": self.context_measure_quanta,
                "stratum_ids": self.stratum_ids,
                "logical_stratum_ids": self.logical_stratum_ids,
                "tensor_lease_hash": self.tensor_lease.lease_hash,
            },
            prefix=_PREFIX + "borrowed-segment-identity-",
        )
        if current != self._identity_hash:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.MUTATED_CUSTODY,
                "borrowed segment measure or source identity changed",
            )
        return self.tensor_lease.compact_tensors()

    def __reduce__(self) -> NoReturn:
        raise TypeError("borrowed compact segments are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("borrowed compact segments are nonserializable")


class V1SignedWorkVariableShellSourceAppendSegmentLease:
    """Locked process-local borrow of every segment in one sealed prefix."""

    __slots__ = (
        "_owner",
        "_lease_token",
        "_closed",
        "_segments",
        "append_receipt_hash",
        "source_chain_hash",
        "borrow_purpose_hash",
        "lease_hash",
        "sample_function_source_only",
        "solver_authority",
        "action_invariance_authority",
        "c_stream_authority",
        "commit_authority",
        "schema",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        owner: "V1SignedWorkVariableShellSourceAppendOwner",
        lease_token: object,
        borrow_purpose_hash: str,
        segments: tuple[V1SignedWorkVariableShellSourceAppendBorrowedSegment, ...],
    ) -> None:
        purpose = _identity(borrow_purpose_hash, name="segment borrow purpose hash")
        receipt = owner.receipt
        if not segments:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.EMPTY_SOURCE,
                "segment lease cannot borrow an empty source",
            )
        self._owner = owner
        self._lease_token = lease_token
        self._closed = False
        self._segments = segments
        self.append_receipt_hash = receipt.receipt_hash
        self.source_chain_hash = receipt.source_chain_hash
        self.borrow_purpose_hash = purpose
        self.sample_function_source_only = True
        self.solver_authority = False
        self.action_invariance_authority = False
        self.c_stream_authority = False
        self.commit_authority = False
        self.schema = VARIABLE_SHELL_SOURCE_APPEND_SEGMENT_LEASE_SCHEMA
        self.lease_hash = content_hash(
            {
                "schema": self.schema,
                "append_receipt_hash": self.append_receipt_hash,
                "source_chain_hash": self.source_chain_hash,
                "borrow_purpose_hash": purpose,
                "segment_source_receipt_hashes": tuple(
                    item.source_receipt.receipt_hash for item in segments
                ),
                "segment_measure_quanta": tuple(
                    item.segment_measure_quanta for item in segments
                ),
                "segment_stratum_axes": tuple(item.stratum_ids for item in segments),
                "segment_logical_stratum_axes": tuple(
                    item.logical_stratum_ids for item in segments
                ),
                "sample_function_source_only": True,
                "all_downstream_authority": False,
            },
            prefix=_PREFIX + "segment-lease-",
        )

    def compact_segments(
        self,
    ) -> tuple[V1SignedWorkVariableShellSourceAppendBorrowedSegment, ...]:
        if self._closed:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.EVICTED,
                "segment lease was closed",
            )
        if self._owner.receipt.receipt_hash != self.append_receipt_hash:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.STALE_IDENTITY,
                "append source changed while its segments were borrowed",
            )
        for segment in self._segments:
            segment.compact_tensors()
        return self._segments

    def close(self) -> None:
        if self._closed:
            return
        self._owner._close_segment_lease(
            token=self._lease_token,
            expected_append_receipt_hash=self.append_receipt_hash,
            segments=self._segments,
        )
        self._closed = True

    def __enter__(self) -> "V1SignedWorkVariableShellSourceAppendSegmentLease":
        self.compact_segments()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("adaptive compact segment leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("adaptive compact segment leases are nonserializable")


class V1SignedWorkVariableShellSourceAppendOwner:
    """Process-local owner of a predictable adaptive prefix of compact handles."""

    __slots__ = (
        "lineage",
        "_context_ids",
        "_draw_ids",
        "_measurement_ids",
        "_capture_ids",
        "_source_packet_hashes",
        "_stratum_ids",
        "_logical_stratum_ids",
        "_measure_quanta",
        "_seen_context_ids",
        "_seen_draw_ids",
        "_seen_measurement_ids",
        "_seen_capture_ids",
        "_seen_source_packet_hashes",
        "_packet_registration_hashes",
        "_source_receipt_hashes",
        "_source_handles",
        "_segment_measure_quanta",
        "_pending",
        "_receipt",
        "_device",
        "_evicted",
        "_active_segment_lease_tokens",
    )

    def __init__(
        self, *, lineage: V1SignedWorkVariableShellSourceAppendLineage
    ) -> None:
        if type(lineage) is not V1SignedWorkVariableShellSourceAppendLineage:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "append owner requires one exact lineage",
            )
        self.lineage = lineage
        self._context_ids: list[str] = []
        self._draw_ids: list[str] = []
        self._measurement_ids: list[str] = []
        self._capture_ids: list[str] = []
        self._source_packet_hashes: list[str] = []
        self._stratum_ids: list[str] = []
        self._logical_stratum_ids: list[str] = []
        self._measure_quanta: list[int] = []
        self._seen_context_ids: set[str] = set()
        self._seen_draw_ids: set[str] = set()
        self._seen_measurement_ids: set[str] = set()
        self._seen_capture_ids: set[str] = set()
        self._seen_source_packet_hashes: set[str] = set()
        self._packet_registration_hashes: list[str] = []
        self._source_receipt_hashes: list[str] = []
        self._source_handles: list[V1SignedWorkVariableShellSourceDeviceHandle] = []
        self._segment_measure_quanta: list[int] = []
        self._pending: (
            V1SignedWorkVariableShellSourceAppendPacketRegistration | None
        ) = None
        self._receipt: V1SignedWorkVariableShellSourceAppendReceipt | None = None
        self._device: str | None = None
        self._evicted = False
        self._active_segment_lease_tokens: set[object] = set()

    def _require_live(self) -> None:
        if self._evicted:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.EVICTED,
                "adaptive source owner was evicted",
            )

    @property
    def context_count(self) -> int:
        self._require_live()
        return len(self._context_ids)

    @property
    def segment_count(self) -> int:
        self._require_live()
        return len(self._source_handles)

    @property
    def has_pending_packet(self) -> bool:
        self._require_live()
        return self._pending is not None

    @property
    def receipt(self) -> V1SignedWorkVariableShellSourceAppendReceipt:
        self._require_live()
        if self._receipt is None:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.EMPTY_SOURCE,
                "no compact source packet has been appended",
            )
        self._validate_all_sources_live()
        return self._receipt

    def register_next_packet(
        self,
        *,
        registered_d_w2_prefix_hash: str,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        measurement_ids: Sequence[str],
        capture_ids: Sequence[str],
        source_packet_hashes: Sequence[str],
        stratum_ids: Sequence[str],
        context_measure_quanta: Sequence[int],
        logical_stratum_ids: Sequence[str] | None = None,
    ) -> V1SignedWorkVariableShellSourceAppendPacketRegistration:
        self._require_live()
        if self._active_segment_lease_tokens:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ACTIVE_REGISTRATION,
                "cannot extend an adaptive source while its sealed segments are borrowed",
            )
        if self._pending is not None:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ACTIVE_REGISTRATION,
                "the preceding predictable packet registration is still active",
            )
        axes = tuple(
            _identity_axis(value, name=name)
            for name, value in (
                ("context IDs", context_ids),
                ("draw IDs", draw_ids),
                ("measurement IDs", measurement_ids),
                ("capture IDs", capture_ids),
                ("source packet hashes", source_packet_hashes),
                ("stratum IDs", stratum_ids),
            )
        )
        if len({len(axis) for axis in axes}) != 1:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_MISMATCH,
                "all predictable packet axes must have one common length",
            )
        if len(self._context_ids) + len(axes[0]) > self.lineage.population_count:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_MISMATCH,
                "predictable packet would exceed the registered population",
            )
        for axis, seen, name in (
            (axes[0], self._seen_context_ids, "context"),
            (axes[1], self._seen_draw_ids, "draw"),
            (axes[2], self._seen_measurement_ids, "measurement"),
            (axes[3], self._seen_capture_ids, "capture"),
            (axes[4], self._seen_source_packet_hashes, "source packet"),
        ):
            if any(item in seen for item in axis):
                _refuse(
                    V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_OVERLAP,
                    f"{name} axis overlaps an already appended prefix",
                )
        quanta = _measure_quanta(context_measure_quanta, count=len(axes[0]))
        logical_strata = (
            axes[5]
            if logical_stratum_ids is None
            else _label_axis(logical_stratum_ids, name="logical stratum IDs")
        )
        if len(logical_strata) != len(axes[0]):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.AXIS_MISMATCH,
                "logical stratum labels must align with the packet context axis",
            )
        registration = V1SignedWorkVariableShellSourceAppendPacketRegistration(
            lineage_hash=self.lineage.lineage_hash,
            parent_append_receipt_hash=(
                _GENESIS_RECEIPT_HASH
                if self._receipt is None
                else self._receipt.receipt_hash
            ),
            packet_ordinal=len(self._source_handles),
            context_start_ordinal=len(self._context_ids),
            registered_d_w2_prefix_hash=_identity(
                registered_d_w2_prefix_hash,
                name="registered D_W2 packet-prefix hash",
            ),
            registered_context_ids=axes[0],
            registered_draw_ids=axes[1],
            registered_measurement_ids=axes[2],
            registered_capture_ids=axes[3],
            registered_source_packet_hashes=axes[4],
            registered_stratum_ids=axes[5],
            context_measure_quanta=quanta,
            registered_logical_stratum_ids=logical_strata,
            _issuance_token=_ISSUER_TOKEN,
        )
        self._pending = registration
        return registration

    def abort_pending_packet(
        self,
        *,
        registration: V1SignedWorkVariableShellSourceAppendPacketRegistration,
    ) -> "V1SignedWorkVariableShellSourceAppendAbortReceipt":
        """Cancel exactly the still-pending preregistration without outcomes.

        This is a narrow failure-atomicity primitive for a multi-destination
        fan-out transaction.  It cannot undo an appended source, choose a
        different packet, or recycle any sample/alpha coordinate.
        """

        self._require_live()
        if self._active_segment_lease_tokens:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ACTIVE_REGISTRATION,
                "cannot abort a packet while compact segments are borrowed",
            )
        if type(registration) is not V1SignedWorkVariableShellSourceAppendPacketRegistration:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "pending abort requires the exact append registration",
            )
        if self._pending is None:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ORDER_MISMATCH,
                "there is no pending packet registration to abort",
            )
        if self._pending.registration_hash != registration.registration_hash:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ORDER_MISMATCH,
                "pending abort registration is stale or belongs to another owner",
            )
        if registration.packet_ordinal != len(self._source_handles):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ORDER_MISMATCH,
                "pending abort cannot cross the sealed append prefix",
            )
        self._pending = None
        return V1SignedWorkVariableShellSourceAppendAbortReceipt(
            lineage_hash=self.lineage.lineage_hash,
            aborted_packet_registration_hash=registration.registration_hash,
            retained_segment_count=len(self._source_handles),
            retained_context_count=len(self._context_ids),
            _issuance_token=_ISSUER_TOKEN,
        )

    def _validate_source_identity(
        self,
        registration: V1SignedWorkVariableShellSourceAppendPacketRegistration,
        source: V1SignedWorkVariableShellSourceDeviceHandle,
    ) -> str:
        try:
            receipt = source.receipt
        except V1SignedWorkVariableShellSourceRefusal as error:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.MUTATED_CUSTODY,
                f"packet source is not live: {error}",
            )
        expected_scalar_fields = {
            "model_identity_hash": self.lineage.model_identity_hash,
            "chart_identity_hash": self.lineage.chart_identity_hash,
            "current_mask_hash": self.lineage.current_mask_hash,
            "proposal_generation": self.lineage.proposal_generation,
            "route_axis_hash": self.lineage.route_axis_hash,
            "route_id_axis_hash": self.lineage.route_id_axis_hash,
            "registered_d_prefix_hash": self.lineage.registered_d_prefix_hash,
            "registered_d_w2_prefix_hash": registration.registered_d_w2_prefix_hash,
            "shell_freeze_receipt_hash": self.lineage.shell_freeze_receipt_hash,
            "native_fiber_axis_receipt_hash": self.lineage.native_fiber_axis_receipt_hash,
            "native_fiber_measure_hash": self.lineage.native_fiber_measure_hash,
            "kkt_branch_id": self.lineage.kkt_branch_id,
            "kkt_branch_source_receipt_hash": self.lineage.kkt_branch_source_receipt_hash,
            "resource_axis_hash": self.lineage.resource_axis_hash,
            "resource_source_receipt_hash": self.lineage.resource_source_receipt_hash,
            "witness_stream_source_hash": self.lineage.witness_stream_source_hash,
            "same_capture_vjp_source_hash": self.lineage.same_capture_vjp_source_hash,
            "corrected_evidence_stream_source_hash": self.lineage.corrected_evidence_stream_source_hash,
            "literal_j_stream_source_hash": self.lineage.literal_j_stream_source_hash,
            "alpha_ledger_id": self.lineage.alpha_ledger_id,
            "alpha_coordinate_id": self.lineage.alpha_coordinate_id,
            "alpha_lineage_hash": self.lineage.alpha_lineage_hash,
            "design_hash": self.lineage.design_hash,
            "population_count": self.lineage.population_count,
            "population_context_axis_hash": self.lineage.population_context_axis_hash,
            "population_draw_axis_hash": self.lineage.population_draw_axis_hash,
            "route_count": self.lineage.route_count,
            "fiber_count": self.lineage.fiber_count,
            "cut_count": self.lineage.cut_count,
            "context_count": registration.context_count,
            "context_axis_hash": _source_axis_hash(
                "context", registration.registered_context_ids
            ),
            "draw_axis_hash": _source_axis_hash(
                "draw", registration.registered_draw_ids
            ),
            "measurement_axis_hash": _source_axis_hash(
                "measurement", registration.registered_measurement_ids
            ),
            "capture_axis_hash": _source_axis_hash(
                "capture", registration.registered_capture_ids
            ),
            "source_packet_axis_hash": _source_axis_hash(
                "source_packet", registration.registered_source_packet_hashes
            ),
        }
        for name, expected in expected_scalar_fields.items():
            if getattr(receipt, name) != expected:
                _refuse(
                    V1SignedWorkVariableShellSourceAppendRefusalCode.STALE_IDENTITY,
                    f"packet source {name} differs from its predictable append registration",
                )
        expected_tuple_fields = {
            "shell_indices": self.lineage.shell_indices,
            "shell_route_ids": self.lineage.shell_route_ids,
            "shell_native_fiber_indices": self.lineage.shell_native_fiber_indices,
        }
        for name, expected in expected_tuple_fields.items():
            if getattr(receipt, name) != expected:
                _refuse(
                    V1SignedWorkVariableShellSourceAppendRefusalCode.STALE_IDENTITY,
                    f"packet source {name} differs from the frozen bounded shell",
                )
        if receipt.receipt_hash in self._source_receipt_hashes:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.DUPLICATE_PACKET,
                "the same sealed compact source was already appended",
            )
        purpose = content_hash(
            {
                "lineage_hash": self.lineage.lineage_hash,
                "packet_registration_hash": registration.registration_hash,
                "source_receipt_hash": receipt.receipt_hash,
                "validate_registered_measure_only": True,
            },
            prefix=_PREFIX + "weight-validation-",
        )
        try:
            with source.borrow_compact_tensors(borrow_purpose_hash=purpose) as lease:
                tensors = lease.compact_tensors()
                weights = tensors[6]
                if torch is None or type(weights) is not torch.Tensor:
                    _refuse(
                        V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                        "compact source weight vector is not a live tensor",
                    )
                expected_weights = torch.tensor(
                    tuple(
                        float(Fraction(item, registration.packet_measure_quanta))
                        for item in registration.context_measure_quanta
                    ),
                    dtype=torch.float64,
                    device=weights.device,
                )
                if not bool(torch.equal(weights, expected_weights)):
                    _refuse(
                        V1SignedWorkVariableShellSourceAppendRefusalCode.WEIGHT_MISMATCH,
                        "packet-local weights are not the exact registered measure-quanta normalization",
                    )
                device = str(weights.device)
        except V1SignedWorkVariableShellSourceRefusal as error:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.MUTATED_CUSTODY,
                f"packet source tensor custody failed validation: {error}",
            )
        if self._device is not None and device != self._device:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.STALE_IDENTITY,
                "all compact packet sources must remain on one execution device",
            )
        return device

    def append_registered_packet(
        self,
        *,
        registration: V1SignedWorkVariableShellSourceAppendPacketRegistration,
        source: V1SignedWorkVariableShellSourceDeviceHandle,
    ) -> V1SignedWorkVariableShellSourceAppendReceipt:
        self._require_live()
        if self._active_segment_lease_tokens:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ACTIVE_REGISTRATION,
                "cannot append a packet while sealed segments are borrowed",
            )
        if (
            type(registration)
            is not V1SignedWorkVariableShellSourceAppendPacketRegistration
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "append requires an exact source-issued packet registration",
            )
        if type(source) is not V1SignedWorkVariableShellSourceDeviceHandle:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "append requires an exact sealed compact source handle",
            )
        if self._pending is None:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ORDER_MISMATCH,
                "register_next_packet must precede source append",
            )
        if registration.registration_hash != self._pending.registration_hash:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ORDER_MISMATCH,
                "packet registration is stale, reordered, or not the active registration",
            )
        expected_parent = (
            _GENESIS_RECEIPT_HASH
            if self._receipt is None
            else self._receipt.receipt_hash
        )
        if (
            registration.lineage_hash != self.lineage.lineage_hash
            or registration.parent_append_receipt_hash != expected_parent
            or registration.packet_ordinal != len(self._source_handles)
            or registration.context_start_ordinal != len(self._context_ids)
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ORDER_MISMATCH,
                "packet registration does not extend the current append chain",
            )
        device = self._validate_source_identity(registration, source)

        self._context_ids.extend(registration.registered_context_ids)
        self._draw_ids.extend(registration.registered_draw_ids)
        self._measurement_ids.extend(registration.registered_measurement_ids)
        self._capture_ids.extend(registration.registered_capture_ids)
        self._source_packet_hashes.extend(registration.registered_source_packet_hashes)
        self._stratum_ids.extend(registration.registered_stratum_ids)
        self._logical_stratum_ids.extend(registration.registered_logical_stratum_ids)
        self._measure_quanta.extend(registration.context_measure_quanta)
        self._seen_context_ids.update(registration.registered_context_ids)
        self._seen_draw_ids.update(registration.registered_draw_ids)
        self._seen_measurement_ids.update(registration.registered_measurement_ids)
        self._seen_capture_ids.update(registration.registered_capture_ids)
        self._seen_source_packet_hashes.update(
            registration.registered_source_packet_hashes
        )
        self._packet_registration_hashes.append(registration.registration_hash)
        self._source_receipt_hashes.append(source.receipt.receipt_hash)
        self._source_handles.append(source)
        self._segment_measure_quanta.append(registration.packet_measure_quanta)
        self._device = device

        context_count = len(self._context_ids)
        finite_axes = (
            context_count == self.lineage.population_count
            and v1_variable_shell_population_axis_hash(
                "context", tuple(self._context_ids)
            )
            == self.lineage.population_context_axis_hash
            and v1_variable_shell_population_axis_hash("draw", tuple(self._draw_ids))
            == self.lineage.population_draw_axis_hash
        )
        packet_axis_hash = content_hash(
            {
                "schema": "partition_free_v1_variable_shell_append_packet_axis_v1",
                "lineage_hash": self.lineage.lineage_hash,
                "packet_registration_hashes": tuple(self._packet_registration_hashes),
            },
            prefix=_PREFIX + "packet-axis-",
        )
        source_axis_hash = content_hash(
            {
                "schema": "partition_free_v1_variable_shell_append_source_axis_v1",
                "lineage_hash": self.lineage.lineage_hash,
                "source_receipt_hashes": tuple(self._source_receipt_hashes),
            },
            prefix=_PREFIX + "source-axis-",
        )
        source_chain_hash = content_hash(
            {
                "schema": "partition_free_v1_variable_shell_append_source_chain_v1",
                "lineage_hash": self.lineage.lineage_hash,
                "parent_append_receipt_hash": expected_parent,
                "packet_registration_axis_hash": packet_axis_hash,
                "source_receipt_axis_hash": source_axis_hash,
                "algorithm_id": VARIABLE_SHELL_SOURCE_APPEND_ALGORITHM,
            },
            prefix=_PREFIX + "source-chain-",
        )
        receipt = V1SignedWorkVariableShellSourceAppendReceipt(
            lineage_hash=self.lineage.lineage_hash,
            parent_append_receipt_hash=expected_parent,
            appended_packet_registration_hash=registration.registration_hash,
            appended_source_receipt_hash=source.receipt.receipt_hash,
            packet_registration_axis_hash=packet_axis_hash,
            source_receipt_axis_hash=source_axis_hash,
            context_axis_hash=_source_axis_hash("context", self._context_ids),
            draw_axis_hash=_source_axis_hash("draw", self._draw_ids),
            measurement_axis_hash=_source_axis_hash(
                "measurement", self._measurement_ids
            ),
            capture_axis_hash=_source_axis_hash("capture", self._capture_ids),
            source_packet_axis_hash=_source_axis_hash(
                "source_packet", self._source_packet_hashes
            ),
            stratum_axis_hash=content_hash(
                {
                    "schema": "partition_free_v1_variable_shell_append_stratum_axis_v2",
                    "stratum_measure_source_hash": self.lineage.stratum_measure_source_hash,
                    "unique_item_ids": tuple(self._stratum_ids),
                    "logical_labels": tuple(self._logical_stratum_ids),
                },
                prefix=_PREFIX + "stratum-axis-",
            ),
            measure_axis_hash=content_hash(
                {
                    "schema": "partition_free_v1_variable_shell_append_measure_axis_v1",
                    "common_integer_measure_unit_hash": self.lineage.common_integer_measure_unit_hash,
                    "context_measure_quanta": tuple(self._measure_quanta),
                    "no_heuristic_renormalization": True,
                },
                prefix=_PREFIX + "measure-axis-",
            ),
            source_chain_hash=source_chain_hash,
            segment_count=len(self._source_handles),
            context_count=context_count,
            population_count=self.lineage.population_count,
            cumulative_measure_quanta=sum(self._measure_quanta),
            retained_tensor_bytes=sum(
                item.receipt.retained_tensor_bytes for item in self._source_handles
            ),
            finite_census_axis_custody_complete=finite_axes,
            finite_census_complete=finite_axes,
            _issuance_token=_ISSUER_TOKEN,
        )
        self._receipt = receipt
        self._pending = None
        return receipt

    def _validate_all_sources_live(self) -> None:
        for expected_hash, source in zip(
            self._source_receipt_hashes, self._source_handles, strict=True
        ):
            try:
                actual_hash = source.receipt.receipt_hash
            except V1SignedWorkVariableShellSourceRefusal as error:
                _refuse(
                    V1SignedWorkVariableShellSourceAppendRefusalCode.MUTATED_CUSTODY,
                    f"an appended compact source is no longer live: {error}",
                )
            if actual_hash != expected_hash:
                _refuse(
                    V1SignedWorkVariableShellSourceAppendRefusalCode.MUTATED_CUSTODY,
                    "an appended compact source identity changed",
                )

    def evaluate(
        self, shell_gates: object
    ) -> V1SignedWorkVariableShellSourceAppendEvaluationReceipt:
        receipt = self.receipt
        # A packet evaluation has already collapsed its contexts under that
        # packet's local normalization.  Such scalars can be recombined only
        # when every packet carries the same logical-stratum mass vector.
        # Unequal terminal tails remain fully supported by the segmented
        # tensor protocol, but must not be silently reduced through this
        # information-losing scalar compatibility surface.
        if self._logical_stratum_ids != self._stratum_ids:
            packet_mass_axes: list[tuple[tuple[str, Fraction], ...]] = []
            offset = 0
            for source, mass in zip(
                self._source_handles,
                self._segment_measure_quanta,
                strict=True,
            ):
                count = source.receipt.context_count
                logical = self._logical_stratum_ids[offset : offset + count]
                quanta = self._measure_quanta[offset : offset + count]
                offset += count
                masses: dict[str, Fraction] = {}
                for label, quantum in zip(logical, quanta, strict=True):
                    masses[label] = masses.get(label, Fraction(0)) + Fraction(
                        quantum, mass
                    )
                packet_mass_axes.append(tuple(sorted(masses.items())))
            if offset != len(self._measure_quanta) or len(set(packet_mass_axes)) != 1:
                _refuse(
                    V1SignedWorkVariableShellSourceAppendRefusalCode.WEIGHT_MISMATCH,
                    "scalar packet evaluations cannot compose unequal logical-stratum tails; use the segmented tensor protocol",
                )
        evaluations = []
        for source in self._source_handles:
            try:
                evaluations.append(source.evaluate(shell_gates))
            except V1SignedWorkVariableShellSourceRefusal as error:
                _refuse(
                    V1SignedWorkVariableShellSourceAppendRefusalCode.MUTATED_CUSTODY,
                    f"an appended source could not be evaluated: {error}",
                )
        gate_hashes = {item.shell_gate_hash for item in evaluations}
        gate_axes = {item.shell_gates for item in evaluations}
        if len(gate_hashes) != 1 or len(gate_axes) != 1:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.STALE_IDENTITY,
                "packet evaluations do not share one frozen shell-gate coordinate",
            )
        total = sum(self._segment_measure_quanta)

        def combine(name: str) -> Fraction:
            return (
                sum(
                    (
                        Fraction(mass) * Fraction.from_float(getattr(item, name))
                        for mass, item in zip(
                            self._segment_measure_quanta, evaluations, strict=True
                        )
                    ),
                    start=Fraction(0),
                )
                / total
            )

        f_lower_exact = combine("f_lower")
        f_center_exact = combine("f_center")
        f_upper_exact = combine("f_upper")
        q_lower_exact = combine("q_lower")
        q_center_exact = combine("q_center")
        q_upper_exact = combine("q_upper")
        f_lower = max(0.0, _fraction_down(f_lower_exact))
        f_center = _fraction_nearest(f_center_exact)
        f_upper = _fraction_up(f_upper_exact)
        q_lower = max(0.0, _fraction_down(q_lower_exact))
        q_center = _fraction_nearest(q_center_exact)
        q_upper = _fraction_up(q_upper_exact)
        # The exact center is enclosed by the exact endpoint combination;
        # these min/max guards cover the final nearest conversion itself.
        f_lower = min(f_lower, f_center)
        f_upper = max(f_upper, f_center)
        q_lower = min(q_lower, q_center)
        q_upper = max(q_upper, q_center)
        first = evaluations[0]
        return V1SignedWorkVariableShellSourceAppendEvaluationReceipt(
            append_receipt_hash=receipt.receipt_hash,
            source_chain_hash=receipt.source_chain_hash,
            shell_gate_hash=first.shell_gate_hash,
            shell_gates=first.shell_gates,
            segment_evaluation_receipt_hashes=tuple(
                item.receipt_hash for item in evaluations
            ),
            segment_measure_quanta=tuple(self._segment_measure_quanta),
            cumulative_measure_quanta=total,
            f_lower=f_lower,
            f_center=f_center,
            f_upper=f_upper,
            q_lower=q_lower,
            q_center=q_center,
            q_upper=q_upper,
            finite_census_complete=receipt.finite_census_complete,
            population_function_exact_except_numerical_enclosure=(
                receipt.finite_census_complete
            ),
            _issuance_token=_ISSUER_TOKEN,
        )

    def borrow_compact_segments(
        self,
        *,
        borrow_purpose_hash: str,
    ) -> V1SignedWorkVariableShellSourceAppendSegmentLease:
        """Borrow every sealed packet through source-owned tensor leases.

        The append prefix is locked until the returned lease closes.  The
        lease exposes exact context measure quanta, so a downstream compact
        quotient can build global weights directly from the common integer
        measure rather than renormalizing packet-local weights.
        """

        receipt = self.receipt
        if self._pending is not None:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ACTIVE_REGISTRATION,
                "cannot borrow a prefix with an unresolved packet registration",
            )
        purpose = _identity(borrow_purpose_hash, name="segment borrow purpose hash")
        token = object()
        self._active_segment_lease_tokens.add(token)
        tensor_leases: list[V1SignedWorkVariableShellSourceTensorLease] = []
        try:
            offset = 0
            borrowed: list[V1SignedWorkVariableShellSourceAppendBorrowedSegment] = []
            for ordinal, (source, mass) in enumerate(
                zip(
                    self._source_handles,
                    self._segment_measure_quanta,
                    strict=True,
                )
            ):
                source_receipt = source.receipt
                count = source_receipt.context_count
                quanta = tuple(self._measure_quanta[offset : offset + count])
                strata = tuple(self._stratum_ids[offset : offset + count])
                logical_strata = tuple(
                    self._logical_stratum_ids[offset : offset + count]
                )
                offset += count
                segment_purpose = content_hash(
                    {
                        "append_receipt_hash": receipt.receipt_hash,
                        "source_chain_hash": receipt.source_chain_hash,
                        "borrow_purpose_hash": purpose,
                        "segment_ordinal": ordinal,
                        "source_receipt_hash": source_receipt.receipt_hash,
                        "segment_measure_quanta": mass,
                    },
                    prefix=_PREFIX + "segment-tensor-purpose-",
                )
                tensor_lease = source.borrow_compact_tensors(
                    borrow_purpose_hash=segment_purpose
                )
                tensor_leases.append(tensor_lease)
                borrowed.append(
                    V1SignedWorkVariableShellSourceAppendBorrowedSegment(
                        source_receipt=source_receipt,
                        segment_measure_quanta=mass,
                        context_measure_quanta=quanta,
                        stratum_ids=strata,
                        logical_stratum_ids=logical_strata,
                        tensor_lease=tensor_lease,
                    )
                )
            if offset != len(self._measure_quanta):
                _refuse(
                    V1SignedWorkVariableShellSourceAppendRefusalCode.WEIGHT_MISMATCH,
                    "segment context counts do not partition the appended measure axis",
                )
            return V1SignedWorkVariableShellSourceAppendSegmentLease(
                owner=self,
                lease_token=token,
                borrow_purpose_hash=purpose,
                segments=tuple(borrowed),
            )
        except Exception:
            for lease in reversed(tensor_leases):
                try:
                    lease.close()
                except Exception:
                    pass
            self._active_segment_lease_tokens.discard(token)
            raise

    def _close_segment_lease(
        self,
        *,
        token: object,
        expected_append_receipt_hash: str,
        segments: tuple[V1SignedWorkVariableShellSourceAppendBorrowedSegment, ...],
    ) -> None:
        if token not in self._active_segment_lease_tokens:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.STALE_IDENTITY,
                "segment lease token is not active on this append owner",
            )
        if self.receipt.receipt_hash != expected_append_receipt_hash:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.STALE_IDENTITY,
                "append receipt changed during a segment borrow",
            )
        if tuple(item.source_receipt.receipt_hash for item in segments) != tuple(
            self._source_receipt_hashes
        ):
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.MUTATED_CUSTODY,
                "borrowed segment identity axis changed before lease close",
            )
        for segment in segments:
            segment.compact_tensors()
        for segment in reversed(segments):
            segment.tensor_lease.close()
        self._active_segment_lease_tokens.remove(token)

    def evict(self) -> None:
        self._require_live()
        if self._active_segment_lease_tokens:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ACTIVE_REGISTRATION,
                "cannot evict an adaptive source while its segments are borrowed",
            )
        if self._pending is not None:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ACTIVE_REGISTRATION,
                "cannot evict while a predictable packet registration is active",
            )
        for source in self._source_handles:
            try:
                source.evict()
            except V1SignedWorkVariableShellSourceRefusal as error:
                _refuse(
                    V1SignedWorkVariableShellSourceAppendRefusalCode.MUTATED_CUSTODY,
                    f"compact source eviction failed: {error}",
                )
        self._source_handles.clear()
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("adaptive variable-shell source owners are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("adaptive variable-shell source owners are nonserializable")


class V1SignedWorkVariableShellSourceAppendSegmentedHandle:
    """Typed compact-source protocol handle over one frozen append receipt.

    The handle never merges or replays the adaptive prefix.  It simply binds a
    stable receipt view to the owner-issued segment-lease protocol used by the
    Phase-B touched-fiber quotient.
    """

    __slots__ = ("_owner", "_receipt", "_owner_receipt_hash")
    __hash__ = None

    def __init__(
        self,
        *,
        owner: V1SignedWorkVariableShellSourceAppendOwner,
    ) -> None:
        if type(owner) is not V1SignedWorkVariableShellSourceAppendOwner:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.INVALID_TYPE,
                "segmented handle requires the exact append owner",
            )
        if owner.has_pending_packet:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.ACTIVE_REGISTRATION,
                "cannot freeze a segmented view with a pending packet",
            )
        append_receipt = owner.receipt
        lineage = owner.lineage
        registered_d_w2_prefix_hash = content_hash(
            {
                "schema": (
                    "partition_free_v1_variable_shell_append_registered_"
                    "d_w2_prefix_v1"
                ),
                "predictable_packet_order_source_hash": (
                    lineage.predictable_packet_order_source_hash
                ),
                "common_integer_measure_unit_hash": (
                    lineage.common_integer_measure_unit_hash
                ),
                "context_axis_hash": append_receipt.context_axis_hash,
                "draw_axis_hash": append_receipt.draw_axis_hash,
                "measurement_axis_hash": append_receipt.measurement_axis_hash,
                "capture_axis_hash": append_receipt.capture_axis_hash,
                "source_packet_axis_hash": append_receipt.source_packet_axis_hash,
                "stratum_axis_hash": append_receipt.stratum_axis_hash,
                "measure_axis_hash": append_receipt.measure_axis_hash,
                "context_count": append_receipt.context_count,
                "population_count": append_receipt.population_count,
            },
            prefix=_PREFIX + "registered-d-w2-prefix-",
        )
        self._owner = owner
        self._owner_receipt_hash = append_receipt.receipt_hash
        self._receipt = V1SignedWorkVariableShellSourceAppendSegmentedReceipt(
            append_receipt_hash=append_receipt.receipt_hash,
            source_chain_hash=append_receipt.source_chain_hash,
            packet_registration_axis_hash=(
                append_receipt.packet_registration_axis_hash
            ),
            source_receipt_axis_hash=append_receipt.source_receipt_axis_hash,
            model_identity_hash=lineage.model_identity_hash,
            chart_identity_hash=lineage.chart_identity_hash,
            current_mask_hash=lineage.current_mask_hash,
            proposal_generation=lineage.proposal_generation,
            route_axis_hash=lineage.route_axis_hash,
            route_id_axis_hash=lineage.route_id_axis_hash,
            registered_d_prefix_hash=lineage.registered_d_prefix_hash,
            registered_d_w2_prefix_hash=registered_d_w2_prefix_hash,
            shell_freeze_receipt_hash=lineage.shell_freeze_receipt_hash,
            shell_indices=lineage.shell_indices,
            shell_route_ids=lineage.shell_route_ids,
            shell_native_fiber_indices=lineage.shell_native_fiber_indices,
            native_fiber_axis_receipt_hash=(lineage.native_fiber_axis_receipt_hash),
            native_fiber_measure_hash=lineage.native_fiber_measure_hash,
            kkt_branch_id=lineage.kkt_branch_id,
            kkt_branch_source_receipt_hash=(lineage.kkt_branch_source_receipt_hash),
            resource_axis_hash=lineage.resource_axis_hash,
            resource_source_receipt_hash=lineage.resource_source_receipt_hash,
            witness_stream_source_hash=lineage.witness_stream_source_hash,
            same_capture_vjp_source_hash=lineage.same_capture_vjp_source_hash,
            corrected_evidence_stream_source_hash=(
                lineage.corrected_evidence_stream_source_hash
            ),
            literal_j_stream_source_hash=lineage.literal_j_stream_source_hash,
            alpha_ledger_id=lineage.alpha_ledger_id,
            alpha_coordinate_id=lineage.alpha_coordinate_id,
            alpha_lineage_hash=lineage.alpha_lineage_hash,
            design_hash=lineage.design_hash,
            population_context_axis_hash=lineage.population_context_axis_hash,
            population_draw_axis_hash=lineage.population_draw_axis_hash,
            context_axis_hash=append_receipt.context_axis_hash,
            draw_axis_hash=append_receipt.draw_axis_hash,
            measurement_axis_hash=append_receipt.measurement_axis_hash,
            capture_axis_hash=append_receipt.capture_axis_hash,
            source_packet_axis_hash=append_receipt.source_packet_axis_hash,
            context_count=append_receipt.context_count,
            population_count=append_receipt.population_count,
            route_count=lineage.route_count,
            fiber_count=lineage.fiber_count,
            cut_count=lineage.cut_count,
            segment_count=append_receipt.segment_count,
            cumulative_measure_quanta=(append_receipt.cumulative_measure_quanta),
            retained_tensor_bytes=append_receipt.retained_tensor_bytes,
            source_replay_hash=append_receipt.source_chain_hash,
            finite_census_axis_custody_complete=(
                append_receipt.finite_census_axis_custody_complete
            ),
            finite_census_complete=append_receipt.finite_census_complete,
            _issuance_token=_ISSUER_TOKEN,
        )

    @property
    def receipt(self) -> V1SignedWorkVariableShellSourceAppendSegmentedReceipt:
        if self._owner.receipt.receipt_hash != self._owner_receipt_hash:
            _refuse(
                V1SignedWorkVariableShellSourceAppendRefusalCode.STALE_IDENTITY,
                "adaptive prefix changed after its segmented handle was frozen",
            )
        return self._receipt

    @property
    def retained_tensor_bytes(self) -> int:
        return self.receipt.retained_tensor_bytes

    def borrow_compact_segments(
        self,
        *,
        borrow_purpose_hash: str,
    ) -> V1SignedWorkVariableShellSourceAppendSegmentLease:
        self.receipt
        return self._owner.borrow_compact_segments(
            borrow_purpose_hash=borrow_purpose_hash
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("segmented compact-source handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("segmented compact-source handles are nonserializable")


__all__ = [
    "VARIABLE_SHELL_SOURCE_APPEND_ALGORITHM",
    "VARIABLE_SHELL_SOURCE_APPEND_AUTHORITY_SCOPE",
    "VARIABLE_SHELL_SOURCE_APPEND_ABORT_RECEIPT_SCHEMA",
    "V1SignedWorkVariableShellSourceAppendEvaluationReceipt",
    "V1SignedWorkVariableShellSourceAppendBorrowedSegment",
    "V1SignedWorkVariableShellSourceAppendAbortReceipt",
    "V1SignedWorkVariableShellSourceAppendLineage",
    "V1SignedWorkVariableShellSourceAppendOwner",
    "V1SignedWorkVariableShellSourceAppendPacketRegistration",
    "V1SignedWorkVariableShellSourceAppendReceipt",
    "V1SignedWorkVariableShellSourceAppendRefusal",
    "V1SignedWorkVariableShellSourceAppendRefusalCode",
    "V1SignedWorkVariableShellSourceAppendSegmentLease",
    "V1SignedWorkVariableShellSourceAppendSegmentedHandle",
    "V1SignedWorkVariableShellSourceAppendSegmentedReceipt",
]
