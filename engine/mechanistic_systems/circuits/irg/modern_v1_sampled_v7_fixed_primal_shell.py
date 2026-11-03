"""Live V7 fitting-D1-census projections and provisional shell union.

This is the zero-provider ``D_fixed`` half of one honest V7 decision packet.
It consumes the already sealed D1 corrected-A/literal-J shards exactly once,
projects both live KKT private primals from a shared bounded two-pass replay,
runs the two fixed-primal diagnostics in spectral--diagonal order, and freezes
their exact ``B <= 8`` union.  It does not measure ``D_W2`` and grants no
sampling-stability, C, K, O, commit, terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import hashlib
import math
from numbers import Integral, Real
from typing import Any, Callable, Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.core.axis_identity import ConstructiveOrdinalRouteAxis

from .contracts import content_hash
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_kkt_bridge import V1SampledKKTSourceProblemBundle
from .modern_v1_sampled_signed_work_pre_c_bridge import (
    V1SampledSignedWorkPreCRegistrationResult,
)
from .modern_v1_sampled_v7_d_fixed_source import (
    V1SampledV7D1CanonicalAxisPlan,
    V1SampledV7D1ReplaySliceLease,
)
from .modern_v1_signed_work_device_shell_cuda import (
    V1SignedWorkDeviceShellExecutionBundle,
)
from .modern_v1_signed_work_fixed_primal_projection_cuda import (
    V1SignedWorkFixedPrimalProjectionCUDAAccumulator,
    V1SignedWorkFixedPrimalProjectionCUDAExecution,
    V1SignedWorkFixedPrimalProjectionCUDAExecutionDesign,
    V1SignedWorkFixedPrimalProjectionRegistration,
    issue_v1_signed_work_fixed_primal_v7_d1_exact_census_registration,
    v1_signed_work_fixed_primal_census_replay_digest_family_hash,
)
from .modern_v1_signed_work_fixed_primal_shell_union import (
    V1SignedWorkFixedPrimalShellUnionReceipt,
    freeze_v1_signed_work_fixed_primal_shell_union,
)
from .modern_v1_signed_work_kkt_cuda import (
    V1SignedWorkLeastWorkCUDAPolicy,
    V1SignedWorkLeastWorkCUDAProblem,
    V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
)
from .modern_v1_signed_work_literal_j_private_primal_bridge_cuda import (
    V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecution,
)
from .modern_v1_signed_work_normal_cone_action_invariance_cuda import (
    V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult,
    V1SignedWorkNormalConeActionInvarianceCUDADesign,
    diagnose_v1_signed_work_fixed_primal_rounding_cuda,
)
from .modern_v1_signed_work_pre_c_coordinator import (
    V1SignedWorkPreCLiveCommonCoordinateContext,
)

try:  # pragma: no cover - availability is host-dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SAMPLED_V7_FIXED_PRIMAL_SHELL_DESIGN_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_frozen_D1_fitting_census_fixed_primal_shell_design_v1"
)
SAMPLED_V7_FIXED_PRIMAL_SHELL_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_frozen_D1_fitting_census_fixed_primal_shell_receipt_v1"
)
SAMPLED_V7_FIXED_PRIMAL_SHELL_AUTHORITY: Final = (
    "frozen_D1_fitting_set_exact_census_fixed_primal_projection_and_B_le_8_"
    "provisional_shell_requires_d_over_N_scaling_and_D_W2_residual_join_"
    "initializer_only_no_D_stability_C_K_O_commit_terminal_or_pareto_authority_v1"
)
_PREFIX: Final = "partition-free-v1-sampled-v7-D1-fixed-primal-shell-"
_ISSUER: Final = object()
_F64_BYTES: Final = 8

V1SampledV7FixedPrimalD1ProjectionConsumer = Callable[
    [
        tuple[
            V1SignedWorkFixedPrimalProjectionCUDAExecution,
            V1SignedWorkFixedPrimalProjectionCUDAExecution,
        ]
    ],
    None,
]


class V1SampledV7FixedPrimalShellRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_LINEAGE = "stale_lineage"
    BRANCH_ORDER_MISMATCH = "branch_order_mismatch"
    AXIS_MISMATCH = "axis_mismatch"
    NONCENSUS_SOURCE = "noncensus_source"
    SOURCE_REPLAY_MISMATCH = "source_replay_mismatch"
    CUDA_REQUIRED = "cuda_required"
    PRODUCTION_CPU_FALLBACK = "production_cpu_fallback"
    SHELL_OVERSIZED = "shell_oversized"
    REUSED_OR_EVICTED = "reused_or_evicted"
    CLEANUP_FAILURE = "cleanup_failure"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7FixedPrimalShellRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7FixedPrimalShellRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7FixedPrimalShellRefusalCode:
            raise TypeError("V7 fixed-primal-shell refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7FixedPrimalShellRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7FixedPrimalShellRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7FixedPrimalShellRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        _refuse(
            V1SampledV7FixedPrimalShellRefusalCode.INVALID_TYPE,
            f"{name} must be one exact positive integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        _refuse(
            V1SampledV7FixedPrimalShellRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    return int(value)


@dataclass(frozen=True, slots=True)
class V1SampledV7FixedPrimalShellDesign:
    route_tile_size: int = 131_072
    maximum_shell_routes: int = 8
    hbm_ceiling_fraction: float = 0.85
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    evict_projection_geometry_after_diagnostics: bool = True
    evict_d1_plan_after_replay: bool = True
    schema: str = SAMPLED_V7_FIXED_PRIMAL_SHELL_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_tile_size",
            _positive_int(self.route_tile_size, name="route tile size"),
        )
        maximum = _positive_int(
            self.maximum_shell_routes, name="maximum shell routes"
        )
        if maximum > 8:
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.SHELL_OVERSIZED,
                "canonical V7 provisional shell is bounded by B=8",
            )
        object.__setattr__(self, "maximum_shell_routes", maximum)
        if (
            isinstance(self.hbm_ceiling_fraction, bool)
            or not isinstance(self.hbm_ceiling_fraction, Real)
            or not math.isfinite(float(self.hbm_ceiling_fraction))
            or float(self.hbm_ceiling_fraction) != 0.85
        ):
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.INVALID_TYPE,
                "canonical HBM ceiling is exactly 0.85",
            )
        for name in (
            "production_cuda_required",
            "allow_private_cpu_reference",
            "evict_projection_geometry_after_diagnostics",
            "evict_d1_plan_after_replay",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.production_cuda_required == self.allow_private_cpu_reference:
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.PRODUCTION_CPU_FALLBACK,
                "exactly one production-CUDA/private-reference mode is required",
            )
        if (
            self.evict_projection_geometry_after_diagnostics is not True
            or self.evict_d1_plan_after_replay is not True
            or self.schema != SAMPLED_V7_FIXED_PRIMAL_SHELL_DESIGN_SCHEMA
        ):
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.AUTHORITY_VIOLATION,
                "V7 source and projection custody must be evicted at the natural seam",
            )

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class V1SampledV7FixedPrimalShellReceipt:
    design_hash: str
    d1_axis_plan_hash: str
    d1_row_prefix_manifest_hash: str
    d1_original_capture_binding_hash: str
    d1_pair_prefix_axis_hash: str
    problem_hash: str
    policy_hash: str
    parent_mask_hash: str
    chart_identity_hash: str
    model_identity_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    native_fiber_axis_receipt_hash: str
    route_count: int
    branch_axis: tuple[V1RelaxedKKTBranch, V1RelaxedKKTBranch]
    branch_execution_hashes: tuple[str, str]
    branch_shell_hashes: tuple[str, str]
    branch_receipt_hashes: tuple[str, str]
    private_primal_source_hash: str
    private_primal_descriptor_axis_hash: str
    fixed_primal_direction_ids: tuple[str, str]
    projection_receipt_hashes: tuple[str, str]
    diagnostic_hashes: tuple[str, str]
    shell_union_receipt_hash: str
    d1_stratum_counts: tuple[int, int]
    d1_stratum_weight_fractions: tuple[tuple[int, int], tuple[int, int]]
    d1_context_count: int
    replay_slice_count: int
    physical_paired_row_pass_count: int
    role_descriptor_sequential_pass_count: int
    sealed_source_payload_read_bytes: int
    host_to_device_payload_bytes: int
    shared_census_replay_digest_family_hash: str
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    provider_call_count: int = 0
    vjp_call_count: int = 0
    forward_call_count: int = 0
    restored_context_by_route_bank_count: int = 0
    route_scale_device_to_host_bytes: int = 0
    branch_duplicate_source_pass_count: int = 0
    exact_frozen_d1_fitting_census: bool = True
    full_population_census_authority: bool = False
    full_population_decision_stability_authority: bool = False
    d1_scaling_and_dw2_residual_join_required: bool = True
    sampling_radii_identically_zero: bool = True
    supports_derived_from_sealed_d1_rows: bool = True
    pass1_pass2_value_identity_verified: bool = True
    shared_two_branch_source_replay: bool = True
    target_then_complement_replay: bool = True
    projection_geometry_evicted: bool = True
    d1_plan_evicted: bool = True
    provisional_initializer_only: bool = True
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_FIXED_PRIMAL_SHELL_AUTHORITY
    schema: str = SAMPLED_V7_FIXED_PRIMAL_SHELL_RECEIPT_SCHEMA
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer is not _ISSUER:
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.INVALID_TYPE,
                "V7 fixed-primal shell receipts are source-issued only",
            )
        for name in (
            "design_hash", "d1_axis_plan_hash", "d1_row_prefix_manifest_hash",
            "d1_original_capture_binding_hash", "d1_pair_prefix_axis_hash",
            "problem_hash", "policy_hash", "parent_mask_hash",
            "chart_identity_hash", "model_identity_hash", "route_axis_hash",
            "route_id_axis_hash", "native_fiber_axis_receipt_hash",
            "private_primal_source_hash", "private_primal_descriptor_axis_hash",
            "shell_union_receipt_hash",
            "shared_census_replay_digest_family_hash",
        ):
            _identity(getattr(self, name), name=name)
        if self.branch_axis != (
            V1RelaxedKKTBranch.SPECTRAL,
            V1RelaxedKKTBranch.DIAGONAL_L1,
        ):
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.BRANCH_ORDER_MISMATCH,
                "receipt branch axis must be spectral then diagonal_l1",
            )
        for name in (
            "branch_execution_hashes", "branch_shell_hashes",
            "branch_receipt_hashes", "fixed_primal_direction_ids",
            "projection_receipt_hashes", "diagnostic_hashes",
        ):
            axis = tuple(_identity(item, name=f"{name} item") for item in getattr(self, name))
            if len(axis) != 2:
                raise ValueError(f"{name} must contain exactly two branches")
        for name in (
            "route_count", "d1_context_count", "replay_slice_count",
            "physical_paired_row_pass_count",
            "role_descriptor_sequential_pass_count",
        ):
            _positive_int(getattr(self, name), name=name)
        _nonnegative_int(self.proposal_generation, name="proposal generation")
        counts = tuple(
            _positive_int(value, name="D1 stratum count")
            for value in self.d1_stratum_counts
        )
        fractions = tuple(self.d1_stratum_weight_fractions)
        if (
            len(counts) != 2
            or len(fractions) != 2
            or any(
                type(value) is not tuple
                or len(value) != 2
                or any(
                    isinstance(part, bool) or not isinstance(part, Integral)
                    for part in value
                )
                for value in fractions
            )
            or sum(counts) != self.d1_context_count
            or fractions
            != tuple(
                (
                    Fraction(count, self.d1_context_count).numerator,
                    Fraction(count, self.d1_context_count).denominator,
                )
                for count in counts
            )
            or self.physical_paired_row_pass_count != 2 * self.replay_slice_count
            or self.role_descriptor_sequential_pass_count
            != 4 * self.replay_slice_count
        ):
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.NONCENSUS_SOURCE,
                "D1 census counts or exact replay-pass accounting do not close",
            )
        counter_names = (
            "provider_call_count",
            "vjp_call_count",
            "forward_call_count",
            "restored_context_by_route_bank_count",
            "route_scale_device_to_host_bytes",
            "branch_duplicate_source_pass_count",
        )
        counters = tuple(
            _nonnegative_int(getattr(self, name), name=name)
            for name in counter_names
        )
        if any(value != 0 for value in counters):
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.AUTHORITY_VIOLATION,
                "D1 fixed-primal source path must remain zero-provider and zero-D2H",
            )
        _positive_int(
            self.sealed_source_payload_read_bytes,
            name="sealed source payload read bytes",
        )
        _nonnegative_int(
            self.host_to_device_payload_bytes,
            name="host-to-device payload bytes",
        )
        if (
            type(self.production_cuda_execution) is not bool
            or type(self.private_cpu_reference_execution) is not bool
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or (
                self.production_cuda_execution
                and self.host_to_device_payload_bytes <= 0
            )
            or (
                self.private_cpu_reference_execution
                and self.host_to_device_payload_bytes != 0
            )
        ):
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.AUTHORITY_VIOLATION,
                "source payload transfer accounting does not match the execution mode",
            )
        open_flags = (
            self.exact_frozen_d1_fitting_census,
            self.d1_scaling_and_dw2_residual_join_required,
            self.sampling_radii_identically_zero,
            self.supports_derived_from_sealed_d1_rows,
            self.pass1_pass2_value_identity_verified,
            self.shared_two_branch_source_replay,
            self.target_then_complement_replay,
            self.projection_geometry_evicted, self.d1_plan_evicted,
            self.provisional_initializer_only,
        )
        closed_flags = (
            self.full_population_census_authority,
            self.full_population_decision_stability_authority,
            self.d_stability_authority, self.c_stream_authority,
            self.calibration_authority, self.oracle_authority,
            self.commit_authority, self.terminal_authority, self.pareto_authority,
        )
        if (
            any(type(value) is not bool for value in open_flags + closed_flags)
            or any(value is not True for value in open_flags)
            or any(value is not False for value in closed_flags)
            or self.authority_scope != SAMPLED_V7_FIXED_PRIMAL_SHELL_AUTHORITY
            or self.schema != SAMPLED_V7_FIXED_PRIMAL_SHELL_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.AUTHORITY_VIOLATION,
                "V7 fixed-primal receipt crossed its provisional authority ceiling",
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
class V1SampledV7FixedPrimalShellResult:
    receipt: V1SampledV7FixedPrimalShellReceipt
    branch_diagnostics: tuple[
        V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult,
        V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult,
    ]
    shell_union: V1SignedWorkFixedPrimalShellUnionReceipt

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1SampledV7FixedPrimalShellReceipt:
            raise TypeError("V7 fixed-primal result requires an exact receipt")
        diagnostics = tuple(self.branch_diagnostics)
        if (
            len(diagnostics) != 2
            or any(
                type(item) is not V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult
                for item in diagnostics
            )
            or type(self.shell_union) is not V1SignedWorkFixedPrimalShellUnionReceipt
        ):
            raise TypeError("V7 fixed-primal result requires exact diagnostics and union")
        if (
            tuple(item.diagnostic.branch for item in diagnostics)
            != self.receipt.branch_axis
            or tuple(item.diagnostic.diagnostic_hash for item in diagnostics)
            != self.receipt.diagnostic_hashes
            or self.shell_union.receipt_hash != self.receipt.shell_union_receipt_hash
        ):
            raise ValueError("V7 fixed-primal result identities do not close")


def _projection_registration(
    *,
    plan: V1SampledV7D1CanonicalAxisPlan,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    parent_mask: HardMask,
    model_identity_hash: str,
    same_capture_vjp_source_hash: str,
    causal_registration_receipt_hash: str,
    fixed_primal_id: str,
    fixed_primal_source_receipt_hash: str,
) -> V1SignedWorkFixedPrimalProjectionRegistration:
    plan_hash = plan.plan_hash
    support_a = content_hash(
        (plan_hash, "corrected_A", "exact_D1_census_maxima"),
        prefix=_PREFIX + "support-",
    )
    support_j = content_hash(
        (plan_hash, "literal_J", "exact_D1_census_maxima"),
        prefix=_PREFIX + "support-",
    )
    return issue_v1_signed_work_fixed_primal_v7_d1_exact_census_registration(
        d1_axis_plan=plan,
        registration_id=content_hash(
            (plan_hash, fixed_primal_id), prefix=_PREFIX + "projection-registration-"
        ),
        model_identity_hash=model_identity_hash,
        chart_identity_hash=problem.source_problem.source_geometry_hash,
        current_mask_hash=parent_mask.mask_hash,
        proposal_generation=plan.proposal_generation,
        route_axis_hash=plan.route_axis_hash,
        route_id_axis_hash=plan.route_id_axis_hash,
        witness_stream_source_hash=plan_hash,
        same_capture_vjp_source_hash=same_capture_vjp_source_hash,
        corrected_evidence_stream_source_hash=(
            plan.corrected_evidence_stream_source_hash
        ),
        literal_j_stream_source_hash=plan.literal_j_stream_source_hash,
        causal_registration_receipt_hash=causal_registration_receipt_hash,
        fixed_primal_id=fixed_primal_id,
        fixed_primal_source_receipt_hash=fixed_primal_source_receipt_hash,
        corrected_support_source_receipt_hash=support_a,
        literal_j_support_source_receipt_hash=support_j,
        alpha_ledger_id=content_hash(
            (plan_hash, "alpha_free_exact_census"), prefix=_PREFIX + "ledger-"
        ),
        alpha_coordinate_id=content_hash(
            (plan_hash, fixed_primal_id, "single_exact_census"),
            prefix=_PREFIX + "coordinate-",
        ),
        look_ordinal=1,
    )


def _projection_call_bindings(
    registration: V1SignedWorkFixedPrimalProjectionRegistration,
) -> dict[str, object]:
    return {
        "model_identity_hash": registration.model_identity_hash,
        "chart_identity_hash": registration.chart_identity_hash,
        "current_mask_hash": registration.current_mask_hash,
        "proposal_generation": registration.proposal_generation,
        "witness_stream_source_hash": registration.witness_stream_source_hash,
        "same_capture_vjp_source_hash": registration.same_capture_vjp_source_hash,
        "corrected_evidence_stream_source_hash": (
            registration.corrected_evidence_stream_source_hash
        ),
        "literal_j_stream_source_hash": registration.literal_j_stream_source_hash,
        "alpha_ledger_id": registration.alpha_ledger_id,
        "alpha_coordinate_id": registration.alpha_coordinate_id,
        "look_ordinal": registration.look_ordinal,
    }


def _validate_core_lineage(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    branch_executions: tuple[
        V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
        V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    ],
    branch_shells: tuple[
        V1SignedWorkDeviceShellExecutionBundle,
        V1SignedWorkDeviceShellExecutionBundle,
    ],
    parent_mask: HardMask,
    plan: V1SampledV7D1CanonicalAxisPlan,
    route_ids: ConstructiveOrdinalRouteAxis,
) -> None:
    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("V7 fixed-primal shell requires an exact least-work problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("V7 fixed-primal shell requires an exact least-work policy")
    executions = tuple(branch_executions)
    shells = tuple(branch_shells)
    expected = (V1RelaxedKKTBranch.SPECTRAL, V1RelaxedKKTBranch.DIAGONAL_L1)
    if (
        len(executions) != 2
        or any(
            type(value) is not V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle
            for value in executions
        )
        or tuple(value.branch for value in executions) != expected
        or len(shells) != 2
        or any(
            type(value) is not V1SignedWorkDeviceShellExecutionBundle
            for value in shells
        )
        or tuple(value.branch_identity.branch for value in shells) != expected
    ):
        _refuse(
            V1SampledV7FixedPrimalShellRefusalCode.BRANCH_ORDER_MISMATCH,
            "live KKT executions and shells must be spectral then diagonal_l1",
        )
    if type(parent_mask) is not HardMask:
        raise TypeError("V7 fixed-primal shell requires an exact parent mask")
    if type(plan) is not V1SampledV7D1CanonicalAxisPlan:
        raise TypeError("V7 fixed-primal shell requires an exact D1 axis plan")
    if type(route_ids) is not ConstructiveOrdinalRouteAxis:
        raise TypeError("V7 fixed-primal shell requires the constructive route-ID owner")
    try:
        plan.assert_live()
    except BaseException as error:
        raise V1SampledV7FixedPrimalShellRefusal(
            V1SampledV7FixedPrimalShellRefusalCode.REUSED_OR_EVICTED,
            "D1 source plan is stale, consumed, or evicted",
        ) from error
    problem_hash = problem.assert_integrity()
    source = problem.source_problem
    operator = problem.signed_work_operator
    native = operator.native_fiber_axis_receipt
    native.assert_integrity()
    if (
        parent_mask.mask_hash != shells[0].execution_receipt.parent_mask_hash
        or parent_mask.mask_hash != shells[1].execution_receipt.parent_mask_hash
        or parent_mask.mask_hash != plan.current_mask_hash
        or plan.route_axis_hash != source.route_axis_hash
        or plan.route_axis_hash != native.route_axis_hash
        or plan.route_id_axis_hash != native.route_id_axis_hash
        or plan.route_count != source.route_count
        or len(route_ids) != source.route_count
        or any(value.result.problem_hash != problem_hash for value in executions)
        or any(value.result.policy_hash != policy.policy_hash for value in executions)
        or any(value.execution_receipt.problem_hash != problem_hash for value in shells)
        or any(value.execution_receipt.policy_hash != policy.policy_hash for value in shells)
        or any(
            shell.execution_receipt.least_work_execution_hash != execution.execution_hash
            for shell, execution in zip(shells, executions, strict=True)
        )
    ):
        _refuse(
            V1SampledV7FixedPrimalShellRefusalCode.STALE_LINEAGE,
            "D1, parent, route, KKT execution, and live shell identities differ",
        )


def _paired_source_pass(
    *,
    lease: V1SampledV7D1ReplaySliceLease,
    accumulators: tuple[
        V1SignedWorkFixedPrimalProjectionCUDAAccumulator,
        V1SignedWorkFixedPrimalProjectionCUDAAccumulator,
    ],
    pass_ordinal: int,
    route_tile_size: int,
    device: Any,
) -> tuple[int, int, str]:
    if type(lease) is not V1SampledV7D1ReplaySliceLease:
        raise TypeError("paired D1 replay requires an exact one-shot slice lease")
    a_iterator = lease.corrected_a_attachment.iter_route_windows(verify=True)
    j_iterator = lease.literal_j_attachment.iter_route_windows(verify=True)
    h2d_bytes = 0
    route_cursor = 0
    replay_digest = hashlib.sha256()
    replay_digest.update(
        content_hash(
            {
                "schema": "partition_free_v1_sampled_v7_D1_paired_A_J_replay_digest_header_v1",
                "slice": lease.slice,
                "records": lease.records,
            },
            prefix=_PREFIX + "paired-replay-header-",
        ).encode("ascii")
    )
    for a_window, j_window in zip(a_iterator, j_iterator, strict=True):
        a_start, a_stop, a_values = a_window
        j_start, j_stop, j_values = j_window
        if (a_start, a_stop) != (j_start, j_stop) or a_start != route_cursor:
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.SOURCE_REPLAY_MISMATCH,
                "paired A/J shard windows differ or are not a canonical route cover",
            )
        source_rows = lease.slice.source_rows
        for start in range(a_start, a_stop, route_tile_size):
            stop = min(a_stop, start + route_tile_size)
            left = start - a_start
            right = stop - a_start
            a_host = np.ascontiguousarray(a_values[source_rows, left:right])
            j_host = np.ascontiguousarray(j_values[source_rows, left:right])
            if a_host.dtype != np.float64 or j_host.dtype != np.float64:
                _refuse(
                    V1SampledV7FixedPrimalShellRefusalCode.SOURCE_REPLAY_MISMATCH,
                    "sealed D1 source left its FP64 role contract",
                )
            replay_digest.update(start.to_bytes(8, "big", signed=False))
            replay_digest.update(stop.to_bytes(8, "big", signed=False))
            replay_digest.update(len(source_rows).to_bytes(8, "big", signed=False))
            replay_digest.update(memoryview(a_host).cast("B"))
            replay_digest.update(memoryview(j_host).cast("B"))
            a_device = torch.as_tensor(
                a_host, dtype=torch.float64, device=device
            ).contiguous()
            j_device = torch.as_tensor(
                j_host, dtype=torch.float64, device=device
            ).contiguous()
            if device.type == "cuda":
                h2d_bytes += int(a_device.numel() + j_device.numel()) * _F64_BYTES
            for accumulator in accumulators:
                if pass_ordinal == 1:
                    accumulator.consume_census_route_window_pass1(
                        route_start=start,
                        corrected_evidence_rows=a_device,
                        literal_j_rows=j_device,
                    )
                else:
                    accumulator.consume_census_route_window_pass2(
                        route_start=start,
                        corrected_evidence_rows=a_device,
                        literal_j_rows=j_device,
                    )
            del a_device, j_device, a_host, j_host
        route_cursor = a_stop
    if route_cursor != lease.corrected_a_attachment.receipt.route_count:
        _refuse(
            V1SampledV7FixedPrimalShellRefusalCode.SOURCE_REPLAY_MISMATCH,
            "paired source pass did not cover the complete route axis",
        )
    payload_read = (
        lease.corrected_a_attachment.receipt.exact_payload_file_bytes
        + lease.literal_j_attachment.receipt.exact_payload_file_bytes
    )
    return h2d_bytes, payload_read, replay_digest.hexdigest()


def _execute_core(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    branch_executions: tuple[
        V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
        V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    ],
    branch_shells: tuple[
        V1SignedWorkDeviceShellExecutionBundle,
        V1SignedWorkDeviceShellExecutionBundle,
    ],
    parent_mask: HardMask,
    d1_axis_plan: V1SampledV7D1CanonicalAxisPlan,
    route_ids: ConstructiveOrdinalRouteAxis,
    model_identity_hash: str,
    same_capture_vjp_source_hash: str,
    causal_registration_receipt_hash: str,
    fixed_primal_ids: tuple[str, str],
    private_primal_source_hash: str,
    private_primal_descriptor_axis_hash: str,
    directions: Any,
    design: V1SampledV7FixedPrimalShellDesign,
    d1_projection_consumer: V1SampledV7FixedPrimalD1ProjectionConsumer | None = None,
) -> V1SampledV7FixedPrimalShellResult:
    _validate_core_lineage(
        problem=problem,
        policy=policy,
        branch_executions=branch_executions,
        branch_shells=branch_shells,
        parent_mask=parent_mask,
        plan=d1_axis_plan,
        route_ids=route_ids,
    )
    if torch is None or type(directions) is not torch.Tensor:
        raise TypeError("V7 fixed-primal directions must be one exact torch tensor")
    if (
        directions.dtype != torch.float64
        or directions.ndim != 2
        or tuple(int(value) for value in directions.shape)
        != (2, problem.source_problem.route_count)
        or not directions.is_contiguous()
        or directions.requires_grad
    ):
        _refuse(
            V1SampledV7FixedPrimalShellRefusalCode.AXIS_MISMATCH,
            "private primals must be one contiguous FP64 canonical branch stack",
        )
    if design.production_cuda_required and directions.device.type != "cuda":
        _refuse(
            V1SampledV7FixedPrimalShellRefusalCode.PRODUCTION_CPU_FALLBACK,
            "production V7 fixed-primal projection refuses CPU fallback",
        )
    if design.allow_private_cpu_reference and directions.device.type != "cpu":
        _refuse(
            V1SampledV7FixedPrimalShellRefusalCode.INVALID_TYPE,
            "private-reference V7 projection requires CPU theorem state",
        )
    plan_hash = d1_axis_plan.plan_hash
    slice_count = len(d1_axis_plan.replay_slices)
    maximum_batch = max(len(item.source_rows) for item in d1_axis_plan.replay_slices)
    projection_design = V1SignedWorkFixedPrimalProjectionCUDAExecutionDesign(
        route_tile_size=design.route_tile_size,
        maximum_batch_size=maximum_batch,
        hbm_ceiling_fraction=design.hbm_ceiling_fraction,
        production_cuda_required=design.production_cuda_required,
        allow_private_cpu_reference=design.allow_private_cpu_reference,
        exact_census_windowed_source=True,
    )
    native = problem.signed_work_operator.native_fiber_axis_receipt
    registrations = tuple(
        _projection_registration(
            plan=d1_axis_plan,
            problem=problem,
            parent_mask=parent_mask,
            model_identity_hash=_identity(model_identity_hash, name="model identity hash"),
            same_capture_vjp_source_hash=_identity(
                same_capture_vjp_source_hash, name="same-capture/VJP source hash"
            ),
            causal_registration_receipt_hash=_identity(
                causal_registration_receipt_hash,
                name="causal registration receipt hash",
            ),
            fixed_primal_id=fixed_primal_ids[index],
            fixed_primal_source_receipt_hash=(
                branch_executions[index].branch_receipt.receipt_hash
            ),
        )
        for index in range(2)
    )
    accumulators = tuple(
        V1SignedWorkFixedPrimalProjectionCUDAAccumulator(
            registration=registration,
            native_fiber_axis_receipt=native,
            fixed_primal=directions[index],
            fixed_primal_id=registration.fixed_primal_id,
            fixed_primal_source_receipt_hash=(
                registration.fixed_primal_source_receipt_hash
            ),
            corrected_evidence_abs_support_by_stratum=None,
            literal_j_abs_support_by_stratum=None,
            corrected_support_source_receipt_hash=(
                registration.corrected_support_source_receipt_hash
            ),
            literal_j_support_source_receipt_hash=(
                registration.literal_j_support_source_receipt_hash
            ),
            design=projection_design,
            device_total_memory_bytes=(1 << 34 if design.allow_private_cpu_reference else None),
            device_free_memory_bytes=(1 << 34 if design.allow_private_cpu_reference else None),
        )
        for index, registration in enumerate(registrations)
    )
    projections: list[V1SignedWorkFixedPrimalProjectionCUDAExecution] = []
    payload_read = 0
    h2d_bytes = 0
    replay_digests: list[str] = []
    replay_started = False
    try:
        for replay_slice in d1_axis_plan.iter_replay_slices():
            replay_started = True
            lease = d1_axis_plan.open_replay_slice(replay_slice)
            try:
                records = lease.records
                axes = {
                    "context_ids": tuple(item.context_id for item in records),
                    "draw_ids": tuple(item.draw_id for item in records),
                    "measurement_ids": tuple(item.measurement_id for item in records),
                    "capture_ids": tuple(item.capture_id for item in records),
                }
                for accumulator, registration in zip(
                    accumulators, registrations, strict=True
                ):
                    accumulator.begin_census_same_capture_batch(
                        stratum_id=replay_slice.stratum,
                        **axes,
                        **_projection_call_bindings(registration),
                    )
                pass_h2d, pass_read, pass1_digest = _paired_source_pass(
                    lease=lease,
                    accumulators=accumulators,
                    pass_ordinal=1,
                    route_tile_size=design.route_tile_size,
                    device=directions.device,
                )
                h2d_bytes += pass_h2d
                payload_read += pass_read
                for accumulator in accumulators:
                    accumulator.finish_census_route_window_pass1(
                        streaming_replay_digest=pass1_digest,
                    )
                pass_h2d, pass_read, pass2_digest = _paired_source_pass(
                    lease=lease,
                    accumulators=accumulators,
                    pass_ordinal=2,
                    route_tile_size=design.route_tile_size,
                    device=directions.device,
                )
                h2d_bytes += pass_h2d
                payload_read += pass_read
                for accumulator in accumulators:
                    accumulator.finish_census_same_capture_batch(
                        streaming_replay_digest=pass2_digest,
                    )
                if pass1_digest != pass2_digest:
                    _refuse(
                        V1SampledV7FixedPrimalShellRefusalCode.SOURCE_REPLAY_MISMATCH,
                        "paired A/J pass-one/pass-two value digests differ",
                    )
                lease.assert_live()
                replay_digests.append(pass1_digest)
            finally:
                lease.burn()
        projections.extend(accumulator.finalize_fixed_look() for accumulator in accumulators)
        replay_digest_family_hash = (
            v1_signed_work_fixed_primal_census_replay_digest_family_hash(
                replay_digests
            )
        )
        for projection in projections:
            if (
                projection.receipt.exact_census_mode is not True
                or projection.receipt.sampling_alpha_spent is not False
                or projection.receipt.signed_f_sampling_radius_identically_zero is not True
                or projection.receipt.literal_j_q_sampling_radius_identically_zero is not True
                or projection.receipt.pass1_pass2_value_identity_verified is not True
                or projection.receipt.census_replay_digest_family_hash
                != replay_digest_family_hash
            ):
                _refuse(
                    V1SampledV7FixedPrimalShellRefusalCode.NONCENSUS_SOURCE,
                    "one branch projection did not retain exact D1 census semantics",
                )
        if d1_projection_consumer is not None:
            if not callable(d1_projection_consumer):
                raise TypeError("D1 projection consumer must be callable")
            d1_projection_consumer((projections[0], projections[1]))
            for projection in projections:
                projection.device_handle.assert_live_identity()
        diagnostic_design = V1SignedWorkNormalConeActionInvarianceCUDADesign(
            route_tile_size=design.route_tile_size,
            hard_replay_route_tile_size=design.route_tile_size,
            maximum_shell_routes=design.maximum_shell_routes,
            production_cuda_required=design.production_cuda_required,
            allow_private_cpu_reference=design.allow_private_cpu_reference,
            require_cublas_deterministic_workspace=design.production_cuda_required,
        )
        diagnostics = tuple(
            diagnose_v1_signed_work_fixed_primal_rounding_cuda(
                problem=problem,
                policy=policy,
                execution=execution,
                projection=projection,
                shell=shell,
                parent_mask=parent_mask,
                design=diagnostic_design,
                current_reserved_bytes=(0 if design.allow_private_cpu_reference else None),
                device_total_memory_bytes=((1 << 34) if design.allow_private_cpu_reference else None),
            )
            for execution, projection, shell in zip(
                branch_executions, projections, branch_shells, strict=True
            )
        )
        shell_union = freeze_v1_signed_work_fixed_primal_shell_union(
            branch_results=diagnostics,
            native_fiber_axis_receipt=native,
            route_ids=route_ids,
            maximum_shell_routes=design.maximum_shell_routes,
        )
        if len(shell_union.shell_indices) > design.maximum_shell_routes:
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.SHELL_OVERSIZED,
                "fixed-primal shell union exceeded B=8",
            )
        for projection in projections:
            projection.device_handle.evict()
        d1_axis_plan.evict()
        receipt = V1SampledV7FixedPrimalShellReceipt(
            design_hash=design.design_hash,
            d1_axis_plan_hash=plan_hash,
            d1_row_prefix_manifest_hash=d1_axis_plan.row_prefix_manifest_hash,
            d1_original_capture_binding_hash=d1_axis_plan.original_capture_binding_hash,
            d1_pair_prefix_axis_hash=d1_axis_plan.d1_pair_prefix_axis_hash,
            problem_hash=problem.problem_hash,
            policy_hash=policy.policy_hash,
            parent_mask_hash=parent_mask.mask_hash,
            chart_identity_hash=problem.source_problem.source_geometry_hash,
            model_identity_hash=model_identity_hash,
            proposal_generation=d1_axis_plan.proposal_generation,
            route_axis_hash=d1_axis_plan.route_axis_hash,
            route_id_axis_hash=d1_axis_plan.route_id_axis_hash,
            native_fiber_axis_receipt_hash=native.receipt_hash,
            route_count=d1_axis_plan.route_count,
            branch_axis=(
                V1RelaxedKKTBranch.SPECTRAL,
                V1RelaxedKKTBranch.DIAGONAL_L1,
            ),
            branch_execution_hashes=tuple(
                value.execution_hash for value in branch_executions
            ),
            branch_shell_hashes=tuple(value.execution_hash for value in branch_shells),
            branch_receipt_hashes=tuple(
                value.branch_receipt.receipt_hash for value in branch_executions
            ),
            private_primal_source_hash=private_primal_source_hash,
            private_primal_descriptor_axis_hash=private_primal_descriptor_axis_hash,
            fixed_primal_direction_ids=fixed_primal_ids,
            projection_receipt_hashes=tuple(
                value.receipt.receipt_hash for value in projections
            ),
            diagnostic_hashes=tuple(
                value.diagnostic.diagnostic_hash for value in diagnostics
            ),
            shell_union_receipt_hash=shell_union.receipt_hash,
            d1_stratum_counts=d1_axis_plan.d1_stratum_counts,
            d1_stratum_weight_fractions=(
                d1_axis_plan.exact_D_fixed_stratum_weight_fractions
            ),
            d1_context_count=d1_axis_plan.context_count,
            replay_slice_count=slice_count,
            physical_paired_row_pass_count=2 * slice_count,
            role_descriptor_sequential_pass_count=4 * slice_count,
            sealed_source_payload_read_bytes=payload_read,
            host_to_device_payload_bytes=h2d_bytes,
            shared_census_replay_digest_family_hash=replay_digest_family_hash,
            production_cuda_execution=design.production_cuda_required,
            private_cpu_reference_execution=design.allow_private_cpu_reference,
            _issuer=_ISSUER,
        )
        return V1SampledV7FixedPrimalShellResult(
            receipt=receipt,
            branch_diagnostics=diagnostics,
            shell_union=shell_union,
        )
    except BaseException as primary_error:
        cleanup_errors: list[BaseException] = []
        attempted_projection_owners = 0
        for projection in projections:
            attempted_projection_owners += 1
            try:
                if not projection.device_handle.is_evicted:
                    projection.device_handle.evict()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        attempted_plan_owners = 0
        if replay_started:
            attempted_plan_owners = 1
            try:
                d1_axis_plan.evict()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            refusal = V1SampledV7FixedPrimalShellRefusal(
                V1SampledV7FixedPrimalShellRefusalCode.CLEANUP_FAILURE,
                "fail-closed cleanup reported one or more failures after "
                f"attempting {attempted_projection_owners} projection owner(s) "
                f"and {attempted_plan_owners} D1 plan owner(s)",
            )
            raise refusal from BaseExceptionGroup(
                "V7 fixed-primal primary and cleanup failures",
                [primary_error, *cleanup_errors],
            )
        raise


def issue_v1_sampled_v7_fixed_primal_shell_cuda(
    *,
    prepared_source_problem: V1SampledKKTSourceProblemBundle,
    source_registration: V1SampledSignedWorkPreCRegistrationResult,
    common_coordinate: V1SignedWorkPreCLiveCommonCoordinateContext,
    private_primal_bridge: V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecution,
    d1_axis_plan: V1SampledV7D1CanonicalAxisPlan,
    route_ids: ConstructiveOrdinalRouteAxis,
    design: V1SampledV7FixedPrimalShellDesign | None = None,
    d1_projection_consumer: V1SampledV7FixedPrimalD1ProjectionConsumer | None = None,
) -> V1SampledV7FixedPrimalShellResult:
    """Consume one live production coordinate and exact D1 source once."""

    if design is None:
        design = V1SampledV7FixedPrimalShellDesign()
    if type(design) is not V1SampledV7FixedPrimalShellDesign:
        raise TypeError("V7 fixed-primal shell design must be exact")
    if not design.production_cuda_required:
        _refuse(
            V1SampledV7FixedPrimalShellRefusalCode.PRODUCTION_CPU_FALLBACK,
            "public V7 fixed-primal issuance is production CUDA only",
        )
    if type(prepared_source_problem) is not V1SampledKKTSourceProblemBundle:
        raise TypeError("public V7 issuance requires the exact sampled source problem")
    if type(source_registration) is not V1SampledSignedWorkPreCRegistrationResult:
        raise TypeError("public V7 issuance requires the exact live source registration")
    if type(common_coordinate) is not V1SignedWorkPreCLiveCommonCoordinateContext:
        raise TypeError("public V7 issuance requires the exact live common coordinate")
    if type(private_primal_bridge) is not V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecution:
        raise TypeError("public V7 issuance requires the exact private-primal bridge")
    prepared_hash = prepared_source_problem.assert_integrity()
    source_registration.assert_integrity()
    private_primal_bridge.assert_live_identity()
    registration_receipt = source_registration.registration_receipt
    admission = prepared_source_problem.sampled_geometry_source.admission
    source_receipt = admission.receipt
    dual = admission.dual_chart_stream_binding
    bridge_source = private_primal_bridge.source_receipt
    executions = common_coordinate.branch_executions
    if (
        dual is None
        or source_registration.source_bundle is not common_coordinate.source
        or common_coordinate.problem.source_problem is not prepared_source_problem.problem
        or registration_receipt.prepared_source_problem_hash != prepared_hash
        or registration_receipt.row_prefix_manifest_hash
        != d1_axis_plan.row_prefix_manifest_hash
        or registration_receipt.original_capture_binding_hash
        != d1_axis_plan.original_capture_binding_hash
        or registration_receipt.original_capture_chart_identity_hash
        != d1_axis_plan.chart_identity_hash
        or registration_receipt.registered_d_prefix_hash
        != d1_axis_plan.registered_d_prefix_hash
        or registration_receipt.current_mask_hash != d1_axis_plan.current_mask_hash
        or registration_receipt.proposal_generation != d1_axis_plan.proposal_generation
        or registration_receipt.route_axis_hash != d1_axis_plan.route_axis_hash
        or registration_receipt.route_id_axis_hash != d1_axis_plan.route_id_axis_hash
        or registration_receipt.observed_stratum_counts != d1_axis_plan.d1_stratum_counts
        or registration_receipt.corrected_a_stream_identity_hash
        != d1_axis_plan.corrected_evidence_stream_source_hash
        or registration_receipt.literal_j_stream_identity_hash
        != d1_axis_plan.literal_j_stream_source_hash
        or dual[0] != d1_axis_plan.corrected_evidence_stream_source_hash
        or dual[1] != d1_axis_plan.literal_j_stream_source_hash
        or bridge_source.problem_hash != common_coordinate.problem.problem_hash
        or bridge_source.policy_hash != common_coordinate.policies.phase_b.policy_hash
        or bridge_source.source_execution_hashes
        != tuple(value.execution_hash for value in executions)
        or private_primal_bridge.receipt.route_axis_hash != d1_axis_plan.route_axis_hash
        or private_primal_bridge.receipt.route_count != d1_axis_plan.route_count
    ):
        _refuse(
            V1SampledV7FixedPrimalShellRefusalCode.STALE_LINEAGE,
            "prepared/source/D1/common-coordinate/private-primal lineage does not close",
        )
    with private_primal_bridge.borrow_for_literal_j_projection() as lease:
        directions = lease.directions_for_literal_j_projection()
        return _execute_core(
            problem=common_coordinate.problem,
            policy=common_coordinate.policies.phase_b,
            branch_executions=executions,
            branch_shells=common_coordinate.branch_shells,
            parent_mask=common_coordinate.parent_mask,
            d1_axis_plan=d1_axis_plan,
            route_ids=route_ids,
            model_identity_hash=source_receipt.model_identity_hash,
            same_capture_vjp_source_hash=dual[2],
            causal_registration_receipt_hash=registration_receipt.receipt_hash,
            fixed_primal_ids=private_primal_bridge.descriptor_axis.direction_ids,
            private_primal_source_hash=bridge_source.source_hash,
            private_primal_descriptor_axis_hash=(
                private_primal_bridge.descriptor_axis.descriptor_axis_hash
            ),
            directions=directions,
            design=design,
            d1_projection_consumer=d1_projection_consumer,
        )


def _issue_v1_sampled_v7_fixed_primal_shell_private_reference_for_tests(
    *,
    common_coordinate: V1SignedWorkPreCLiveCommonCoordinateContext,
    d1_axis_plan: V1SampledV7D1CanonicalAxisPlan,
    route_ids: ConstructiveOrdinalRouteAxis,
    model_identity_hash: str = "private-reference:model",
    same_capture_vjp_source_hash: str = "private-reference:same-capture-vjp",
    causal_registration_receipt_hash: str = "private-reference:causal-registration",
    design: V1SampledV7FixedPrimalShellDesign | None = None,
    d1_projection_consumer: V1SampledV7FixedPrimalD1ProjectionConsumer | None = None,
) -> V1SampledV7FixedPrimalShellResult:
    """Tiny exact CPU sibling; unavailable to production callers."""

    if type(common_coordinate) is not V1SignedWorkPreCLiveCommonCoordinateContext:
        raise TypeError("private V7 reference requires the exact common coordinate")
    if not common_coordinate.private_cpu_reference_execution:
        _refuse(
            V1SampledV7FixedPrimalShellRefusalCode.PRODUCTION_CPU_FALLBACK,
            "private reference cannot consume a production coordinate",
        )
    if design is None:
        design = V1SampledV7FixedPrimalShellDesign(
            route_tile_size=min(131_072, common_coordinate.problem.source_problem.route_count),
            production_cuda_required=False,
            allow_private_cpu_reference=True,
        )
    if type(design) is not V1SampledV7FixedPrimalShellDesign or not design.allow_private_cpu_reference:
        raise TypeError("private V7 reference requires its exact private design")
    primals = []
    direction_ids = []
    for execution in common_coordinate.branch_executions:
        state = execution.state_handle
        if state is None:
            _refuse(
                V1SampledV7FixedPrimalShellRefusalCode.REUSED_OR_EVICTED,
                "private KKT theorem state is unavailable",
            )
        primal, _advantage = state.materialize_for_private_test(execution.branch)
        primals.append(primal)
        direction_ids.append(
            content_hash(
                (execution.execution_hash, execution.branch.value),
                prefix=_PREFIX + "private-direction-",
            )
        )
    directions = torch.tensor(np.stack(primals), dtype=torch.float64).contiguous()
    source_hash = content_hash(
        tuple(value.execution_hash for value in common_coordinate.branch_executions),
        prefix=_PREFIX + "private-source-",
    )
    descriptor_hash = content_hash(
        tuple(direction_ids), prefix=_PREFIX + "private-descriptor-axis-"
    )
    return _execute_core(
        problem=common_coordinate.problem,
        policy=common_coordinate.policies.phase_b,
        branch_executions=common_coordinate.branch_executions,
        branch_shells=common_coordinate.branch_shells,
        parent_mask=common_coordinate.parent_mask,
        d1_axis_plan=d1_axis_plan,
        route_ids=route_ids,
        model_identity_hash=model_identity_hash,
        same_capture_vjp_source_hash=same_capture_vjp_source_hash,
        causal_registration_receipt_hash=causal_registration_receipt_hash,
        fixed_primal_ids=tuple(direction_ids),  # type: ignore[arg-type]
        private_primal_source_hash=source_hash,
        private_primal_descriptor_axis_hash=descriptor_hash,
        directions=directions,
        design=design,
        d1_projection_consumer=d1_projection_consumer,
    )


__all__ = [
    "SAMPLED_V7_FIXED_PRIMAL_SHELL_AUTHORITY",
    "V1SampledV7FixedPrimalShellDesign",
    "V1SampledV7FixedPrimalShellReceipt",
    "V1SampledV7FixedPrimalShellRefusal",
    "V1SampledV7FixedPrimalShellRefusalCode",
    "V1SampledV7FixedPrimalShellResult",
    "V1SampledV7FixedPrimalD1ProjectionConsumer",
    "issue_v1_sampled_v7_fixed_primal_shell_cuda",
]
