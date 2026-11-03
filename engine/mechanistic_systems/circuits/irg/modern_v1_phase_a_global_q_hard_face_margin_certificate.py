"""Source-bound hard-mask outside-face certificate from a strict dual margin.

The population scanner has already frozen an outer-LP dual witness, consumed
the unique true-``Q`` support stream, and retained only compact sufficient
statistics.  This joint completes the next discrete theorem without ever
reconstructing a routewise coefficient or membership bank.

For each hard vertex of the registered ``B <= 8`` shell, it keeps the scanned
outside membership fixed and checks:

* coordinate legality and exact compact block capacities;
* every registered support plane, using the conservative outside upper sum
  and the *direct lower-endpoint* shell coefficients owned by the outer LP;
* a lower enclosure of the candidate resource.

Let ``L_L`` be the largest certified feasible resource lower bound, ``U_U``
the scanner's outward structural-dual upper bound, and ``delta_L`` the strict
lower bound on every incumbent-aligned outside reduced-advantage magnitude.
If ``U_U - L_L < delta_L``, every feasible hard action with resource at least
``L_L`` has the same outside membership.  A changed outside bit alone would
contribute at least ``delta_L`` to its dual gap, contradicting the strict
inequality.

This is deliberately not a full-D decision certificate.  It proves one
source-bound, resource-competitive hard outside-face statement only; signed-F
and local action-image composition remain separate downstream obligations.  It
grants no C/K/O, mask-commit, runtime, terminal, or Pareto authority.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import hashlib
import itertools
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_phase_a_global_q_outside_face_cuda import (
    V1PhaseAGlobalQOutsideFaceResult,
    V1PhaseAGlobalQOutsideSignStatus,
)
from .modern_v1_phase_a_global_q_outer_lp_dual_proposal import (
    V1PhaseAGlobalQOuterLPDualProposalResult,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTProblem
from .modern_v1_sampled_v7_action_pair_alias_source_join import (
    V1SampledV7ActionPairAliasSourceJoinReceipt,
)
from .modern_v1_sampled_v7_phase_a_robust_shell_decision import (
    V1SampledV7PhaseARobustDecisionStatus,
    V1SampledV7PhaseARobustShellDecisionReceipt,
)

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_DESIGN_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_true_q_hard_face_margin_design_v1"
)
PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_true_q_hard_face_margin_certificate_v1"
)
PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_ALGORITHM: Final = (
    "source_bound_outward_float64_outer_LP_dual_gap_below_strict_outside_bit_"
    "margin_with_complete_B_le_8_hard_shell_enumeration_v1"
)
PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_AUTHORITY: Final = (
    "resource_competitive_hard_outside_membership_theorem_only_pending_signed_"
    "F_and_local_action_image_no_D_C_K_O_commit_runtime_terminal_or_pareto_"
    "authority_v1"
)

_PREFIX: Final = "partition-free-v1-phase-a-global-q-hard-face-margin-"
_RECEIPT_TOKEN: Final = object()
_HANDLE_TOKEN: Final = object()
_MAXIMUM_SHELL_SIZE: Final = 8
_F64_BYTES: Final = 8


__all__ = (
    "PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_DESIGN_SCHEMA",
    "PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_SCHEMA",
    "PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_ALGORITHM",
    "PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_AUTHORITY",
    "V1PhaseAGlobalQHardFaceMarginCertificateDesign",
    "V1PhaseAGlobalQHardFaceMarginCertificateStatus",
    "V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode",
    "V1PhaseAGlobalQHardFaceMarginCertificateRefusal",
    "V1PhaseAGlobalQHardFaceMarginCertificateReceipt",
    "V1PhaseAGlobalQHardFaceMarginCertificateLease",
    "V1PhaseAGlobalQHardFaceMarginCertificateHandle",
    "V1PhaseAGlobalQHardFaceMarginCertificateResult",
    "certify_v1_phase_a_global_q_hard_face_margin_cuda",
)


class V1PhaseAGlobalQHardFaceMarginCertificateStatus(str, Enum):
    CERTIFIED_RESOURCE_COMPETITIVE_HARD_OUTSIDE_FACE = (
        "certified_resource_competitive_hard_outside_face"
    )
    NO_CERTIFIED_FEASIBLE_HARD_FACE_VERTEX = (
        "no_conservatively_certified_feasible_hard_face_vertex"
    )
    OUTSIDE_REDUCED_ADVANTAGE_AMBIGUOUS = (
        "outside_reduced_advantage_sign_ambiguous"
    )
    OUTSIDE_REDUCED_ADVANTAGE_MISALIGNED = (
        "outside_reduced_advantage_sign_misaligned"
    )
    DUAL_GAP_NOT_STRICTLY_BELOW_OUTSIDE_MARGIN = (
        "dual_gap_not_strictly_below_outside_bit_margin"
    )


class V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    SOURCE_MISMATCH = "source_mismatch"
    NO_STABLE_PHYSICAL_ACTION = "no_stable_physical_action"
    SHELL_AXIS_MISMATCH = "shell_axis_mismatch"
    DEVICE_REQUIRED = "device_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    DETERMINISM_DISABLED = "determinism_disabled"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    NUMERICAL_REFUSAL = "numerical_refusal"
    ACTIVE_LEASE = "active_lease"
    MUTATED_CUSTODY = "mutated_custody"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1PhaseAGlobalQHardFaceMarginCertificateRefusal(RuntimeError):
    def __init__(
        self,
        code: V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode:
            raise TypeError("hard-face margin refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode,
    message: str,
) -> NoReturn:
    raise V1PhaseAGlobalQHardFaceMarginCertificateRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return value


def _finite_optional(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.INVALID_TYPE,
            f"{name} must be one finite scalar or None",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be finite",
        )
    return result


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(dict(payload), prefix=_PREFIX + role + "-")


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value.data_ptr()),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        str(value.dtype),
        str(value.device),
        int(value.storage_offset()),
        int(value._version),
        bool(value.requires_grad),
    )


def _compact_tensor_digest(value: Any, *, role: str) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(role.encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(item) for item in array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return _payload_hash(
        {
            "role": role,
            "shape": tuple(int(item) for item in array.shape),
            "payload_sha256": digest.hexdigest(),
        },
        role="compact-tensor",
    )


def _down(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, -math.inf))


def _up(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _add_down(left: Any, right: Any) -> Any:
    raw = left + right
    return torch.where(right == 0.0, left, torch.where(left == 0.0, right, _down(raw)))


def _add_up(left: Any, right: Any) -> Any:
    raw = left + right
    return torch.where(right == 0.0, left, torch.where(left == 0.0, right, _up(raw)))


def _sub_down(left: Any, right: Any) -> Any:
    raw = left - right
    return torch.where(right == 0.0, left, _down(raw))


def _sub_up(left: Any, right: Any) -> Any:
    raw = left - right
    return torch.where(right == 0.0, left, _up(raw))


def _hard_mask_bit(mask: HardMask, index: int) -> bool:
    byte = mask.packed_bits[index // 8]
    return bool(byte & (1 << (7 - (index % 8))))


def _strict_gap_below_margin(*, gap_upper: float, margin_lower: float) -> bool:
    """Keep the theorem boundary explicit: equality never certifies."""

    return gap_upper < margin_lower


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQHardFaceMarginCertificateDesign:
    design_id: str = "canonical-global-q-hard-face-margin-v1"
    maximum_shell_size: int = _MAXIMUM_SHELL_SIZE
    hbm_ceiling_fraction: float = 0.85
    cpu_reference_route_limit: int = 4096
    private_reference_device_total_memory_bytes: int = 1 << 30
    require_deterministic_algorithms: bool = True
    schema: str = PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        _identity(self.design_id, name="design_id")
        if self.maximum_shell_size != _MAXIMUM_SHELL_SIZE:
            raise ValueError("canonical hard-face shell cap is exactly eight")
        if self.hbm_ceiling_fraction != 0.85:
            raise ValueError("canonical hard-face HBM ceiling is exactly 85%")
        _positive_int(self.cpu_reference_route_limit, name="cpu_reference_route_limit")
        _positive_int(
            self.private_reference_device_total_memory_bytes,
            name="private_reference_device_total_memory_bytes",
        )
        if self.require_deterministic_algorithms is not True:
            raise ValueError("hard-face certification requires deterministic algorithms")
        if self.schema != PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_DESIGN_SCHEMA:
            raise ValueError("unsupported hard-face margin design schema")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQHardFaceMarginCertificateReceipt:
    design_hash: str
    outside_face_receipt_hash: str
    dual_proposal_receipt_hash: str
    action_pair_alias_source_join_receipt_hash: str
    robust_shell_decision_receipt_hash: str
    relaxed_kkt_problem_hash: str
    parent_mask_hash: str
    action_pair_hash: str
    selected_physical_action_identity_hash: str
    selected_packed_action_hash: str
    selected_resulting_mask_hash: str
    route_axis_hash: str
    shell_axis_hash: str
    block_axis_hash: str
    route_count: int
    direction_count: int
    shell_size: int
    block_count: int
    shell_indices: tuple[int, ...]
    selected_outside_count: int
    excluded_outside_eligible_count: int
    selected_outside_block_counts: tuple[int, ...]
    remaining_shell_block_capacities: tuple[int, ...]
    enumerated_shell_vertex_count: int
    coordinate_legal_shell_vertex_count: int
    block_legal_shell_vertex_count: int
    certified_feasible_shell_vertex_count: int
    best_hard_shell_gate_bits: tuple[bool, ...] | None
    best_shell_selected_count: int | None
    best_hard_face_resource_lower: float | None
    best_hard_face_resource_upper: float | None
    structural_dual_resource_lower: float
    structural_dual_resource_upper: float
    dual_gap_lower: float | None
    dual_gap_upper: float | None
    strict_outside_bit_margin_lower: float | None
    strict_outside_bit_margin_upper: float | None
    source_outside_sign_status: V1PhaseAGlobalQOutsideSignStatus
    status: V1PhaseAGlobalQHardFaceMarginCertificateStatus
    best_shell_gate_bits_digest: str
    best_hard_face_resource_bounds_digest: str
    dual_gap_bounds_digest: str
    strict_outside_margin_bounds_digest: str
    component_bytes: tuple[tuple[str, int], ...]
    projected_incremental_peak_bytes: int
    current_reserved_device_bytes: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    hbm_ceiling_bytes: int
    measurement_device_identity: str
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    compact_result_device_to_host_scalar_count: int
    best_hard_face_candidate_present: bool
    strict_outside_signs: bool
    dual_gap_strictly_below_outside_bit_margin: bool
    all_resource_competitive_hard_actions_share_outside_membership: bool
    complete_frozen_shell_enumeration: bool = True
    conservative_outer_lp_face_feasibility: bool = True
    direct_lower_endpoint_shell_coefficients_used: bool = True
    outside_support_upper_enclosure_used: bool = True
    exact_compact_block_counts_used: bool = True
    structural_dual_upper_enclosure_used: bool = True
    strict_inequality_boundary_enforced: bool = True
    preallocation_hbm_forecast_admitted: bool = True
    bounded_compact_device_execution: bool = True
    route_scale_parent_mask_unpack_used: bool = False
    route_scale_d2h: bool = False
    route_scale_coefficient_or_membership_retained: bool = False
    relaxed_face_equality_claim: bool = False
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_ALGORITHM
    authority_scope: str = PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_AUTHORITY
    schema: str = PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _admission_memory: tuple[int, int] | None = field(default=None, repr=False, compare=False)
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("hard-face margin receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        available, allocated = self._admission_memory or (
            self.device_free_memory_bytes, self.current_reserved_device_bytes,
        )
        for name in (
            "design_hash",
            "outside_face_receipt_hash",
            "dual_proposal_receipt_hash",
            "action_pair_alias_source_join_receipt_hash",
            "robust_shell_decision_receipt_hash",
            "relaxed_kkt_problem_hash",
            "parent_mask_hash",
            "action_pair_hash",
            "selected_physical_action_identity_hash",
            "selected_packed_action_hash",
            "selected_resulting_mask_hash",
            "route_axis_hash",
            "shell_axis_hash",
            "block_axis_hash",
            "best_shell_gate_bits_digest",
            "best_hard_face_resource_bounds_digest",
            "dual_gap_bounds_digest",
            "strict_outside_margin_bounds_digest",
            "measurement_device_identity",
        ):
            _identity(getattr(self, name), name=name)
        for name in (
            "route_count",
            "direction_count",
            "shell_size",
            "device_total_memory_bytes",
        ):
            _positive_int(getattr(self, name), name=name)
        for name in (
            "block_count",
            "selected_outside_count",
            "excluded_outside_eligible_count",
            "coordinate_legal_shell_vertex_count",
            "block_legal_shell_vertex_count",
            "certified_feasible_shell_vertex_count",
            "projected_incremental_peak_bytes",
            "current_reserved_device_bytes",
            "device_free_memory_bytes",
            "hbm_ceiling_bytes",
            "compact_result_device_to_host_scalar_count",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        if (
            type(self.shell_indices) is not tuple
            or tuple(sorted(set(self.shell_indices))) != self.shell_indices
            or len(self.shell_indices) != self.shell_size
            or self.shell_size > _MAXIMUM_SHELL_SIZE
            or any(type(value) is not int or value < 0 or value >= self.route_count for value in self.shell_indices)
            or self.enumerated_shell_vertex_count != 2**self.shell_size
            or not 0 <= self.certified_feasible_shell_vertex_count <= self.block_legal_shell_vertex_count <= self.coordinate_legal_shell_vertex_count <= self.enumerated_shell_vertex_count
            or len(self.selected_outside_block_counts) != self.block_count
            or len(self.remaining_shell_block_capacities) != self.block_count
            or any(
                type(value) is not int or isinstance(value, bool) or value < 0
                for value in self.selected_outside_block_counts
                + self.remaining_shell_block_capacities
            )
        ):
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.SOURCE_MISMATCH,
                "hard-face compact axes or counts do not close",
            )
        if type(self.status) is not V1PhaseAGlobalQHardFaceMarginCertificateStatus:
            raise TypeError("hard-face margin status must be exact")
        if type(self.source_outside_sign_status) is not V1PhaseAGlobalQOutsideSignStatus:
            raise TypeError("source outside-sign status must be exact")
        for name in (
            "structural_dual_resource_lower",
            "structural_dual_resource_upper",
            "best_hard_face_resource_lower",
            "best_hard_face_resource_upper",
            "dual_gap_lower",
            "dual_gap_upper",
            "strict_outside_bit_margin_lower",
            "strict_outside_bit_margin_upper",
        ):
            _finite_optional(getattr(self, name), name=name)
        present = self.best_hard_face_candidate_present
        best_values = (
            self.best_hard_shell_gate_bits,
            self.best_shell_selected_count,
            self.best_hard_face_resource_lower,
            self.best_hard_face_resource_upper,
            self.dual_gap_lower,
            self.dual_gap_upper,
        )
        margin_present = self.strict_outside_bit_margin_lower is not None
        if (
            type(present) is not bool
            or type(self.strict_outside_signs) is not bool
            or type(self.dual_gap_strictly_below_outside_bit_margin) is not bool
            or type(
                self.all_resource_competitive_hard_actions_share_outside_membership
            )
            is not bool
            or type(self.production_cuda_execution) is not bool
            or type(self.private_cpu_reference_execution) is not bool
            or present != all(value is not None for value in best_values)
            or (not present and any(value is not None for value in best_values))
            or (present and len(self.best_hard_shell_gate_bits or ()) != self.shell_size)
            or (
                present
                and any(
                    type(bit) is not bool
                    for bit in self.best_hard_shell_gate_bits or ()
                )
            )
            or (
                present
                and (
                    type(self.best_shell_selected_count) is not int
                    or isinstance(self.best_shell_selected_count, bool)
                    or self.best_shell_selected_count
                    != sum(self.best_hard_shell_gate_bits or ())
                )
            )
            or (present and self.certified_feasible_shell_vertex_count <= 0)
            or (not present and self.certified_feasible_shell_vertex_count != 0)
            or self.strict_outside_signs != margin_present
            or margin_present != (self.strict_outside_bit_margin_upper is not None)
            or (
                present
                and (
                    self.best_hard_face_resource_lower > self.best_hard_face_resource_upper
                    or self.dual_gap_lower > self.dual_gap_upper
                )
            )
            or (
                margin_present
                and self.strict_outside_bit_margin_lower
                > self.strict_outside_bit_margin_upper
            )
            or self.structural_dual_resource_lower > self.structural_dual_resource_upper
            or self.structural_dual_resource_lower < 0.0
            or (
                present
                and (
                    self.best_hard_face_resource_lower < 0.0
                    or self.dual_gap_lower < 0.0
                )
            )
            or (
                margin_present
                and self.strict_outside_bit_margin_lower <= 0.0
            )
            or type(self.component_bytes) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or not item[0]
                or type(item[1]) is not int
                or item[1] < 0
                for item in self.component_bytes
            )
        ):
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.AUTHORITY_VIOLATION,
                "hard-face theorem bounds or presence flags diverged",
            )
        expected_gap = bool(
            present
            and margin_present
            and _strict_gap_below_margin(
                gap_upper=float(self.dual_gap_upper),
                margin_lower=float(self.strict_outside_bit_margin_lower),
            )
        )
        certified = (
            self.status
            is V1PhaseAGlobalQHardFaceMarginCertificateStatus.CERTIFIED_RESOURCE_COMPETITIVE_HARD_OUTSIDE_FACE
        )
        expected_status = (
            V1PhaseAGlobalQHardFaceMarginCertificateStatus.NO_CERTIFIED_FEASIBLE_HARD_FACE_VERTEX
            if not present
            else V1PhaseAGlobalQHardFaceMarginCertificateStatus.OUTSIDE_REDUCED_ADVANTAGE_MISALIGNED
            if self.source_outside_sign_status
            is V1PhaseAGlobalQOutsideSignStatus.MISALIGNED
            else V1PhaseAGlobalQHardFaceMarginCertificateStatus.OUTSIDE_REDUCED_ADVANTAGE_AMBIGUOUS
            if self.source_outside_sign_status
            is V1PhaseAGlobalQOutsideSignStatus.AMBIGUOUS
            else V1PhaseAGlobalQHardFaceMarginCertificateStatus.CERTIFIED_RESOURCE_COMPETITIVE_HARD_OUTSIDE_FACE
            if expected_gap
            else V1PhaseAGlobalQHardFaceMarginCertificateStatus.DUAL_GAP_NOT_STRICTLY_BELOW_OUTSIDE_MARGIN
        )
        positive = (
            self.complete_frozen_shell_enumeration,
            self.conservative_outer_lp_face_feasibility,
            self.direct_lower_endpoint_shell_coefficients_used,
            self.outside_support_upper_enclosure_used,
            self.exact_compact_block_counts_used,
            self.structural_dual_upper_enclosure_used,
            self.strict_inequality_boundary_enforced,
            self.preallocation_hbm_forecast_admitted,
            self.bounded_compact_device_execution,
        )
        closed = (
            self.route_scale_parent_mask_unpack_used,
            self.route_scale_d2h,
            self.route_scale_coefficient_or_membership_retained,
            self.relaxed_face_equality_claim,
            self.d_stability_authority,
            self.c_measurement_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.runtime_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            self.dual_gap_strictly_below_outside_bit_margin != expected_gap
            or self.status is not expected_status
            or self.all_resource_competitive_hard_actions_share_outside_membership != certified
            or certified != (present and self.strict_outside_signs and expected_gap)
            or any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or self.hbm_ceiling_bytes != math.floor(0.85 * self.device_total_memory_bytes)
            or self.projected_incremental_peak_bytes
            != sum(value for _name, value in self.component_bytes)
            or allocated + self.projected_incremental_peak_bytes
            > self.hbm_ceiling_bytes
            or self.projected_incremental_peak_bytes > available
            or self.algorithm != PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_ALGORITHM
            or self.authority_scope != PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_AUTHORITY
            or self.schema != PHASE_A_GLOBAL_Q_HARD_FACE_MARGIN_CERTIFICATE_SCHEMA
        ):
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.AUTHORITY_VIOLATION,
                "hard-face receipt crossed its discrete theorem boundary",
            )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if not item.name.startswith("_")
            }
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="receipt")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.MUTATED_CUSTODY,
                "hard-face margin receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


_COMPACT_NAMES: Final = (
    "best_shell_gate_bits",
    "best_hard_face_resource_bounds",
    "dual_gap_bounds",
    "strict_outside_margin_bounds",
)


class V1PhaseAGlobalQHardFaceMarginCertificateLease:
    __slots__ = ("_owner", "_token", "_closed") + _COMPACT_NAMES

    def __init__(
        self,
        owner: "V1PhaseAGlobalQHardFaceMarginCertificateHandle",
        token: object,
        values: tuple[Any, ...],
    ) -> None:
        self._owner = owner
        self._token = token
        self._closed = False
        for name, value in zip(_COMPACT_NAMES, values, strict=True):
            setattr(self, name, value)

    def __enter__(self) -> "V1PhaseAGlobalQHardFaceMarginCertificateLease":
        self._owner.assert_live()
        return self

    def close(self) -> None:
        if not self._closed:
            self._owner._close(self._token)
            self._closed = True

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("hard-face margin leases are nonserializable")


class V1PhaseAGlobalQHardFaceMarginCertificateHandle:
    __slots__ = ("receipt", "_values", "_guards", "_invariants", "_active_token", "_evicted")

    def __init__(
        self,
        *,
        receipt: V1PhaseAGlobalQHardFaceMarginCertificateReceipt,
        values: tuple[Any, ...],
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _HANDLE_TOKEN:
            raise PermissionError("hard-face margin handles require their issuer")
        if type(receipt) is not V1PhaseAGlobalQHardFaceMarginCertificateReceipt:
            raise TypeError("hard-face margin handle requires its exact receipt")
        expected_shapes = ((receipt.shell_size,), (2,), (2,), (2,))
        expected_dtypes = (torch.bool, torch.float64, torch.float64, torch.float64)
        if torch is None or len(values) != len(expected_shapes) or any(
            type(value) is not torch.Tensor
            or tuple(int(item) for item in value.shape) != shape
            or value.dtype is not dtype
            or not value.is_contiguous()
            or value.requires_grad
            or str(value.device) != receipt.measurement_device_identity
            or (dtype is torch.float64 and not bool(torch.isfinite(value).all().item()))
            for value, shape, dtype in zip(values, expected_shapes, expected_dtypes, strict=True)
        ):
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.MUTATED_CUSTODY,
                "hard-face compact tensors do not match their receipt",
            )
        expected_digests = (
            receipt.best_shell_gate_bits_digest,
            receipt.best_hard_face_resource_bounds_digest,
            receipt.dual_gap_bounds_digest,
            receipt.strict_outside_margin_bounds_digest,
        )
        if tuple(
            _compact_tensor_digest(value, role=name)
            for name, value in zip(_COMPACT_NAMES, values, strict=True)
        ) != expected_digests:
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.MUTATED_CUSTODY,
                "hard-face compact tensor digests diverged",
            )
        self.receipt = receipt
        self._values = values
        self._guards = tuple(value.detach().clone() for value in values)
        self._invariants = tuple(_tensor_invariant(value) for value in values)
        self._active_token: object | None = None
        self._evicted = False
        self.assert_live()

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.EVICTED,
                "hard-face margin handle was evicted",
            )
        self.receipt.assert_integrity()
        if any(
            _tensor_invariant(value) != invariant or not bool(torch.equal(value, guard))
            for value, invariant, guard in zip(self._values, self._invariants, self._guards, strict=True)
        ):
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.MUTATED_CUSTODY,
                "hard-face compact tensor custody mutated",
            )

    def borrow(self) -> V1PhaseAGlobalQHardFaceMarginCertificateLease:
        self.assert_live()
        if self._active_token is not None:
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.ACTIVE_LEASE,
                "only one hard-face margin lease may be live",
            )
        token = object()
        self._active_token = token
        return V1PhaseAGlobalQHardFaceMarginCertificateLease(self, token, self._values)

    def _close(self, token: object) -> None:
        if token is not self._active_token:
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.ACTIVE_LEASE,
                "hard-face margin lease token is stale",
            )
        try:
            self.assert_live()
        finally:
            self._active_token = None

    def evict(self) -> None:
        self.assert_live()
        if self._active_token is not None:
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.ACTIVE_LEASE,
                "cannot evict hard-face margin handle under an active lease",
            )
        self._values = ()
        self._guards = ()
        self._invariants = ()
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("hard-face margin handles are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("hard-face margin handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQHardFaceMarginCertificateResult:
    receipt: V1PhaseAGlobalQHardFaceMarginCertificateReceipt
    compact_handle: V1PhaseAGlobalQHardFaceMarginCertificateHandle = field(
        repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not V1PhaseAGlobalQHardFaceMarginCertificateReceipt
            or type(self.compact_handle) is not V1PhaseAGlobalQHardFaceMarginCertificateHandle
        ):
            raise TypeError("hard-face margin result requires exact components")
        self.compact_handle.assert_live()
        if self.compact_handle.receipt.receipt_hash != self.receipt.receipt_hash:
            raise ValueError("hard-face margin result receipt and handle diverged")


def _shell_block_ordinals(
    problem: V1RelaxedKKTProblem,
    shell_indices: tuple[int, ...],
) -> tuple[int, ...]:
    result: list[int] = []
    for coordinate in shell_indices:
        matches = tuple(
            ordinal
            for ordinal, block in enumerate(problem.block_capacities)
            if block.start <= coordinate < block.stop
        )
        if len(matches) != 1:
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.SOURCE_MISMATCH,
                "each shell coordinate must belong to exactly one compact block",
            )
        result.append(matches[0])
    return tuple(result)


def _forecast_memory(
    *,
    direction_count: int,
    shell_size: int,
    production: bool,
    device: Any,
    design: V1PhaseAGlobalQHardFaceMarginCertificateDesign,
) -> tuple[tuple[tuple[str, int], ...], int, int, int, int, int, tuple[int, int]]:
    components = (
        # The vectorized nextafter/where expressions have multiple live
        # temporaries.  Forecast a deliberately conservative envelope rather
        # than counting only the logical input and output arrays.
        ("candidate_support_interval_workspace", 16 * direction_count * _F64_BYTES),
        ("candidate_scalar_interval_workspace", 64 * _F64_BYTES),
        ("candidate_shell_bits", 4 * shell_size),
        ("result_tensors_digests_and_guards", 4 * (shell_size + 6 * _F64_BYTES)),
    )
    incremental = sum(value for _name, value in components)
    if production:
        properties = torch.cuda.get_device_properties(device)
        total = int(properties.total_memory)
        reserved = int(torch.cuda.memory_reserved(device))
        free, _observed_total = torch.cuda.mem_get_info(device)
        free = int(free)
        available = cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free)
        allocated = int(torch.cuda.memory_allocated(device))
    else:
        total = design.private_reference_device_total_memory_bytes
        reserved = 0
        free = total
        available, allocated = free, reserved
    ceiling = math.floor(0.85 * total)
    if allocated + incremental > ceiling or incremental > available:
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.HBM_FORECAST_EXCEEDED,
            "hard-face compact workspace exceeds the 85% device-memory ceiling",
        )
    return components, incremental, reserved, total, free, ceiling, (available, allocated)


def _certify(
    *,
    outside_face: V1PhaseAGlobalQOutsideFaceResult,
    dual_proposal: V1PhaseAGlobalQOuterLPDualProposalResult,
    action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    robust_shell_decision: V1SampledV7PhaseARobustShellDecisionReceipt,
    parent_mask: HardMask,
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    design: V1PhaseAGlobalQHardFaceMarginCertificateDesign,
    private_cpu_reference: bool,
) -> V1PhaseAGlobalQHardFaceMarginCertificateResult:
    exact_types = (
        (outside_face, V1PhaseAGlobalQOutsideFaceResult),
        (dual_proposal, V1PhaseAGlobalQOuterLPDualProposalResult),
        (action_pair_alias_source_join, V1SampledV7ActionPairAliasSourceJoinReceipt),
        (robust_shell_decision, V1SampledV7PhaseARobustShellDecisionReceipt),
        (parent_mask, HardMask),
        (relaxed_kkt_problem, V1RelaxedKKTProblem),
        (design, V1PhaseAGlobalQHardFaceMarginCertificateDesign),
    )
    if torch is None or any(type(value) is not expected for value, expected in exact_types):
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.INVALID_TYPE,
            "hard-face margin certification requires exact source owners",
        )
    try:
        outside_face.compact_handle.assert_live()
        outside_hash = outside_face.receipt.assert_integrity()
        dual_proposal.device_handle.assert_live()
        proposal_hash = dual_proposal.receipt.assert_integrity()
        source_join_hash = action_pair_alias_source_join.assert_integrity()
        robust_hash = robust_shell_decision.assert_integrity()
        pair = action_pair_alias_source_join.action_pair
        pair_hash = pair.assert_integrity()
    except V1PhaseAGlobalQHardFaceMarginCertificateRefusal:
        raise
    except Exception as error:
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.STALE_SOURCE,
            f"one hard-face source failed replay: {type(error).__name__}: {error}",
        )
    outside = outside_face.receipt
    proposal = dual_proposal.receipt
    robust = robust_shell_decision
    if robust.status is not V1SampledV7PhaseARobustDecisionStatus.STABLE_PHYSICAL_ACTION_PREMISE:
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.NO_STABLE_PHYSICAL_ACTION,
            "hard-face certification requires one stable physical action premise",
        )
    matches = tuple(
        action
        for action in pair.physical_actions
        if action.physical_action_identity_hash
        == robust.selected_physical_action_identity_hash
    )
    if len(matches) != 1:
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.NO_STABLE_PHYSICAL_ACTION,
            "stable action identity has no unique source-joined physical action",
        )
    selected = matches[0]
    selected.assert_integrity()
    child = selected.packed_action.resulting_mask
    shell = tuple(proposal.shell_indices)
    blocks = tuple(relaxed_kkt_problem.block_capacities)
    if (
        outside_hash != outside_face.compact_handle.receipt.receipt_hash
        or proposal_hash != dual_proposal.device_handle.receipt.receipt_hash
        or outside.dual_proposal_receipt_hash != proposal_hash
        or outside.action_pair_alias_source_join_receipt_hash != source_join_hash
        or outside.robust_shell_decision_receipt_hash != robust_hash
        or outside.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
        or outside.parent_mask_hash != parent_mask.mask_hash
        or outside.action_pair_hash != pair_hash
        or outside.selected_physical_action_identity_hash != selected.physical_action_identity_hash
        or outside.selected_packed_action_hash != selected.packed_action_hash
        or outside.selected_resulting_mask_hash != child.mask_hash
        or proposal.action_pair_alias_source_join_receipt_hash != source_join_hash
        or proposal.robust_shell_decision_receipt_hash != robust_hash
        or proposal.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
        or proposal.parent_mask_hash != parent_mask.mask_hash
        or proposal.selected_physical_action_identity_hash != selected.physical_action_identity_hash
        or proposal.selected_packed_action_hash != selected.packed_action_hash
        or proposal.selected_resulting_mask_hash != child.mask_hash
        or pair.parent_mask_hash != parent_mask.mask_hash
        or pair.route_axis_hash != relaxed_kkt_problem.route_axis_hash
        or pair.route_count != relaxed_kkt_problem.route_count
        or parent_mask.route_axis_hash != relaxed_kkt_problem.route_axis_hash
        or parent_mask.route_count != relaxed_kkt_problem.route_count
        or outside.route_axis_hash != relaxed_kkt_problem.route_axis_hash
        or outside.route_count != relaxed_kkt_problem.route_count
        or outside.shell_axis_hash != proposal.shell_axis_hash
        or outside.block_axis_hash != proposal.block_axis_hash
        or outside.shell_size != len(shell)
        or outside.block_count != len(blocks)
        or shell != tuple(robust.shell_indices)
        or proposal.block_ids != tuple(block.block_id for block in blocks)
        or proposal.block_hashes != tuple(block.block_hash for block in blocks)
        or outside.block_capacities
        != tuple(block.removal_capacity for block in blocks)
        or outside.remaining_shell_block_capacities
        != tuple(
            block.removal_capacity - used
            for block, used in zip(
                blocks, outside.selected_outside_block_counts, strict=True
            )
        )
    ):
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.SOURCE_MISMATCH,
            "hard-face source graph or compact axes diverged",
        )
    if (
        not 1 <= len(shell) <= design.maximum_shell_size
        or tuple(sorted(set(shell))) != shell
        or any(index < 0 or index >= outside.route_count for index in shell)
    ):
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.SHELL_AXIS_MISMATCH,
            "hard-face shell must be sorted, unique, in range, and B <= 8",
        )
    outside_eligible = outside.selected_outside_count + outside.excluded_outside_eligible_count
    if outside_eligible <= 0:
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.SHELL_AXIS_MISMATCH,
            "strict-margin theorem requires at least one eligible outside coordinate",
        )
    production = bool(outside.production_cuda_execution)
    if production == private_cpu_reference:
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.CPU_PRODUCTION_FALLBACK
            if production
            else V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.DEVICE_REQUIRED,
            "hard-face production/private execution mode does not match its source",
        )
    if (
        outside.private_cpu_reference_execution == production
        or proposal.production_source_authority != production
        or proposal.private_cpu_reference == production
        or outside.measurement_device_identity != proposal.measurement_device_identity
    ):
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.SOURCE_MISMATCH,
            "hard-face source execution modes or devices disagree",
        )
    device = torch.device(outside.measurement_device_identity)
    if production:
        if device.type != "cuda":
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.DEVICE_REQUIRED,
                "production hard-face certification requires CUDA",
            )
        if not torch.are_deterministic_algorithms_enabled():
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.DETERMINISM_DISABLED,
                "production hard-face certification requires deterministic torch algorithms",
            )
    elif outside.route_count > design.cpu_reference_route_limit:
        _refuse(
            V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.CPU_PRODUCTION_FALLBACK,
            "private hard-face CPU reference route cap exceeded",
        )
    components, incremental, reserved, total, free, ceiling, admission_memory = _forecast_memory(
        direction_count=outside.direction_count,
        shell_size=len(shell),
        production=production,
        device=device,
        design=design,
    )
    shell_blocks = _shell_block_ordinals(relaxed_kkt_problem, shell)
    shell_selectable = tuple(
        bool(relaxed_kkt_problem.eligible_mask[index])
        and _hard_mask_bit(parent_mask, index)
        for index in shell
    )
    enumerated = 2 ** len(shell)
    coordinate_legal = 0
    block_legal = 0
    feasible = 0
    best_bits: tuple[bool, ...] | None = None
    best_resource_lower: Any | None = None
    best_resource_upper: Any | None = None
    with outside_face.compact_handle.borrow() as face_lease, dual_proposal.device_handle.borrow() as proposal_lease:
        if (
            not bool(torch.equal(face_lease.support_multipliers, proposal_lease.support_multipliers))
            or not bool(torch.equal(face_lease.block_multipliers, proposal_lease.block_multipliers))
            or not bool((face_lease.support_coefficient_lower <= face_lease.support_coefficient_upper).all().item())
            or not bool((face_lease.outside_support_lower_sum <= face_lease.outside_support_upper_sum).all().item())
            or not bool((face_lease.outside_resource_bounds[0] <= face_lease.outside_resource_bounds[1]).item())
            or not bool((face_lease.full_dual_bounds[0] <= face_lease.full_dual_bounds[1]).item())
            or bool((face_lease.support_multipliers < 0.0).any().item())
            or bool((face_lease.block_multipliers < 0.0).any().item())
        ):
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.NUMERICAL_REFUSAL,
                "hard-face compact source intervals or frozen multipliers are invalid",
            )
        for bits in itertools.product((False, True), repeat=len(shell)):
            if any(bit and not selectable for bit, selectable in zip(bits, shell_selectable, strict=True)):
                continue
            coordinate_legal += 1
            counts = [0] * len(blocks)
            for bit, block_ordinal in zip(bits, shell_blocks, strict=True):
                if bit:
                    counts[block_ordinal] += 1
            if any(
                count > remaining
                for count, remaining in zip(
                    counts, outside.remaining_shell_block_capacities, strict=True
                )
            ):
                continue
            block_legal += 1
            lhs_upper = face_lease.outside_support_upper_sum.clone()
            for offset, bit in enumerate(bits):
                if bit:
                    lhs_upper = _add_up(
                        lhs_upper,
                        face_lease.support_coefficient_lower[:, offset],
                    )
            if not bool((lhs_upper <= face_lease.support_rhs_upper).all().item()):
                continue
            feasible += 1
            resource_lower = face_lease.outside_resource_bounds[0].clone()
            resource_upper = face_lease.outside_resource_bounds[1].clone()
            for offset, bit in enumerate(bits):
                if bit:
                    resource_lower = _add_down(resource_lower, face_lease.shell_resource[offset])
                    resource_upper = _add_up(resource_upper, face_lease.shell_resource[offset])
            lower_value = float(resource_lower.item())
            if (
                best_resource_lower is None
                or lower_value > float(best_resource_lower.item())
                or (lower_value == float(best_resource_lower.item()) and bits < best_bits)
            ):
                best_bits = bits
                best_resource_lower = resource_lower
                best_resource_upper = resource_upper
        dual_lower = face_lease.full_dual_bounds[0].clone()
        dual_upper = face_lease.full_dual_bounds[1].clone()
        selected_lower = outside.selected_reduced_advantage_lower_min
        selected_upper = outside.selected_reduced_advantage_upper_min
        excluded_lower = outside.excluded_reduced_advantage_lower_max
        excluded_upper = outside.excluded_reduced_advantage_upper_max
        if (
            (outside.selected_outside_count == 0) != (selected_lower is None or selected_upper is None)
            or (outside.excluded_outside_eligible_count == 0)
            != (excluded_lower is None or excluded_upper is None)
        ):
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.SOURCE_MISMATCH,
                "outside reduced-advantage category bounds do not match their counts",
            )
        independently_misaligned = (
            selected_upper is not None and selected_upper < 0.0
        ) or (excluded_lower is not None and excluded_lower > 0.0)
        independently_strict = (
            selected_lower is None or selected_lower > 0.0
        ) and (excluded_upper is None or excluded_upper < 0.0)
        expected_sign_status = (
            V1PhaseAGlobalQOutsideSignStatus.MISALIGNED
            if independently_misaligned
            else V1PhaseAGlobalQOutsideSignStatus.STRICT_INCUMBENT_ALIGNED
            if independently_strict
            else V1PhaseAGlobalQOutsideSignStatus.AMBIGUOUS
        )
        if outside.outside_sign_status is not expected_sign_status:
            _refuse(
                V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.SOURCE_MISMATCH,
                "outside sign receipt does not replay its outward bounds",
            )
        margin_lower_value: float | None = None
        margin_upper_value: float | None = None
        if independently_strict:
            lower_candidates = []
            upper_candidates = []
            if selected_lower is not None:
                lower_candidates.append(selected_lower)
                upper_candidates.append(float(selected_upper))
            if excluded_upper is not None:
                lower_candidates.append(-excluded_upper)
                upper_candidates.append(-float(excluded_lower))
            margin_lower_value = min(lower_candidates)
            margin_upper_value = min(upper_candidates)
            if margin_lower_value <= 0.0 or margin_upper_value < margin_lower_value:
                _refuse(
                    V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.NUMERICAL_REFUSAL,
                    "strict outside margin enclosure is invalid",
                )
        if best_resource_lower is None:
            best_bits_tensor = torch.zeros((len(shell),), dtype=torch.bool, device=device)
            resource_bounds = torch.zeros((2,), dtype=torch.float64, device=device)
            gap_bounds = torch.zeros((2,), dtype=torch.float64, device=device)
        else:
            assert best_bits is not None and best_resource_upper is not None
            if bool((dual_upper < best_resource_lower).item()):
                _refuse(
                    V1PhaseAGlobalQHardFaceMarginCertificateRefusalCode.NUMERICAL_REFUSAL,
                    "structural dual upper lies below a certified feasible resource lower bound",
                )
            gap_lower = torch.maximum(
                torch.zeros((), dtype=torch.float64, device=device),
                _sub_down(dual_lower, best_resource_upper),
            )
            gap_upper = _sub_up(dual_upper, best_resource_lower)
            best_bits_tensor = torch.tensor(best_bits, dtype=torch.bool, device=device)
            resource_bounds = torch.stack((best_resource_lower, best_resource_upper)).contiguous()
            gap_bounds = torch.stack((gap_lower, gap_upper)).contiguous()
        margin_bounds = (
            torch.zeros((2,), dtype=torch.float64, device=device)
            if margin_lower_value is None
            else torch.tensor(
                (margin_lower_value, margin_upper_value),
                dtype=torch.float64,
                device=device,
            )
        )
    present = best_bits is not None
    resource_values = tuple(float(value) for value in resource_bounds.detach().cpu().tolist())
    gap_values = tuple(float(value) for value in gap_bounds.detach().cpu().tolist())
    dual_values = (float(dual_lower.item()), float(dual_upper.item()))
    margin_values = tuple(float(value) for value in margin_bounds.detach().cpu().tolist())
    gap_strict = bool(
        present
        and independently_strict
        and _strict_gap_below_margin(
            gap_upper=gap_values[1], margin_lower=margin_values[0]
        )
    )
    if not present:
        status = V1PhaseAGlobalQHardFaceMarginCertificateStatus.NO_CERTIFIED_FEASIBLE_HARD_FACE_VERTEX
    elif outside.outside_sign_status is V1PhaseAGlobalQOutsideSignStatus.MISALIGNED:
        status = V1PhaseAGlobalQHardFaceMarginCertificateStatus.OUTSIDE_REDUCED_ADVANTAGE_MISALIGNED
    elif outside.outside_sign_status is V1PhaseAGlobalQOutsideSignStatus.AMBIGUOUS:
        status = V1PhaseAGlobalQHardFaceMarginCertificateStatus.OUTSIDE_REDUCED_ADVANTAGE_AMBIGUOUS
    elif gap_strict:
        status = V1PhaseAGlobalQHardFaceMarginCertificateStatus.CERTIFIED_RESOURCE_COMPETITIVE_HARD_OUTSIDE_FACE
    else:
        status = V1PhaseAGlobalQHardFaceMarginCertificateStatus.DUAL_GAP_NOT_STRICTLY_BELOW_OUTSIDE_MARGIN
    values = (
        best_bits_tensor.contiguous(),
        resource_bounds.contiguous(),
        gap_bounds.contiguous(),
        margin_bounds.contiguous(),
    )
    digests = tuple(
        _compact_tensor_digest(value, role=name)
        for name, value in zip(_COMPACT_NAMES, values, strict=True)
    )
    receipt = V1PhaseAGlobalQHardFaceMarginCertificateReceipt(
        design_hash=design.design_hash,
        outside_face_receipt_hash=outside_hash,
        dual_proposal_receipt_hash=proposal_hash,
        action_pair_alias_source_join_receipt_hash=source_join_hash,
        robust_shell_decision_receipt_hash=robust_hash,
        relaxed_kkt_problem_hash=relaxed_kkt_problem.problem_hash,
        parent_mask_hash=parent_mask.mask_hash,
        action_pair_hash=pair_hash,
        selected_physical_action_identity_hash=selected.physical_action_identity_hash,
        selected_packed_action_hash=selected.packed_action_hash,
        selected_resulting_mask_hash=child.mask_hash,
        route_axis_hash=outside.route_axis_hash,
        shell_axis_hash=outside.shell_axis_hash,
        block_axis_hash=outside.block_axis_hash,
        route_count=outside.route_count,
        direction_count=outside.direction_count,
        shell_size=len(shell),
        block_count=len(blocks),
        shell_indices=shell,
        selected_outside_count=outside.selected_outside_count,
        excluded_outside_eligible_count=outside.excluded_outside_eligible_count,
        selected_outside_block_counts=outside.selected_outside_block_counts,
        remaining_shell_block_capacities=outside.remaining_shell_block_capacities,
        enumerated_shell_vertex_count=enumerated,
        coordinate_legal_shell_vertex_count=coordinate_legal,
        block_legal_shell_vertex_count=block_legal,
        certified_feasible_shell_vertex_count=feasible,
        best_hard_shell_gate_bits=best_bits,
        best_shell_selected_count=None if best_bits is None else sum(best_bits),
        best_hard_face_resource_lower=None if not present else resource_values[0],
        best_hard_face_resource_upper=None if not present else resource_values[1],
        structural_dual_resource_lower=dual_values[0],
        structural_dual_resource_upper=dual_values[1],
        dual_gap_lower=None if not present else gap_values[0],
        dual_gap_upper=None if not present else gap_values[1],
        strict_outside_bit_margin_lower=(
            None if not independently_strict else margin_values[0]
        ),
        strict_outside_bit_margin_upper=(
            None if not independently_strict else margin_values[1]
        ),
        source_outside_sign_status=outside.outside_sign_status,
        status=status,
        best_shell_gate_bits_digest=digests[0],
        best_hard_face_resource_bounds_digest=digests[1],
        dual_gap_bounds_digest=digests[2],
        strict_outside_margin_bounds_digest=digests[3],
        component_bytes=components,
        projected_incremental_peak_bytes=incremental,
        current_reserved_device_bytes=reserved,
        device_total_memory_bytes=total,
        device_free_memory_bytes=free,
        hbm_ceiling_bytes=ceiling,
        measurement_device_identity=outside.measurement_device_identity,
        production_cuda_execution=production,
        private_cpu_reference_execution=not production,
        compact_result_device_to_host_scalar_count=len(shell) + 6,
        best_hard_face_candidate_present=present,
        strict_outside_signs=independently_strict,
        dual_gap_strictly_below_outside_bit_margin=gap_strict,
        all_resource_competitive_hard_actions_share_outside_membership=(
            status
            is V1PhaseAGlobalQHardFaceMarginCertificateStatus.CERTIFIED_RESOURCE_COMPETITIVE_HARD_OUTSIDE_FACE
        ),
        _issuance_token=_RECEIPT_TOKEN,
        _admission_memory=admission_memory,
    )
    handle = V1PhaseAGlobalQHardFaceMarginCertificateHandle(
        receipt=receipt,
        values=values,
        _issuance_token=_HANDLE_TOKEN,
    )
    return V1PhaseAGlobalQHardFaceMarginCertificateResult(
        receipt=receipt, compact_handle=handle
    )


def certify_v1_phase_a_global_q_hard_face_margin_cuda(
    *,
    outside_face: V1PhaseAGlobalQOutsideFaceResult,
    dual_proposal: V1PhaseAGlobalQOuterLPDualProposalResult,
    action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    robust_shell_decision: V1SampledV7PhaseARobustShellDecisionReceipt,
    parent_mask: HardMask,
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    design: V1PhaseAGlobalQHardFaceMarginCertificateDesign = V1PhaseAGlobalQHardFaceMarginCertificateDesign(),
) -> V1PhaseAGlobalQHardFaceMarginCertificateResult:
    """Run the bounded production CUDA hard-face theorem joint."""

    return _certify(
        outside_face=outside_face,
        dual_proposal=dual_proposal,
        action_pair_alias_source_join=action_pair_alias_source_join,
        robust_shell_decision=robust_shell_decision,
        parent_mask=parent_mask,
        relaxed_kkt_problem=relaxed_kkt_problem,
        design=design,
        private_cpu_reference=False,
    )


def _certify_v1_phase_a_global_q_hard_face_margin_cpu_reference(
    **kwargs: Any,
) -> V1PhaseAGlobalQHardFaceMarginCertificateResult:
    """Private bounded CPU oracle for tests; never a production fallback."""

    return _certify(**kwargs, private_cpu_reference=True)
