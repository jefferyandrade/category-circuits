"""Reference blockwise Jensen witness for sampled signed-work separation.

This module owns one deliberately narrow statistical joint.  A preferred hard
action ``P`` and the *complete* shared alternate domain ``Z_alt`` are frozen by
the D-side proposal before a fresh, outcome-independent equal-block partition of
``U \\ D`` is opened.  For every fresh context the full-axis native-fiber
support plane obeys

``l_x(z) <= f_x(z)``.

For one equal-size residual block ``B`` the production solver supplies a
downward enclosure of the single shared-action *full-population* minimum

``G_B = inf_z [w_D mean_D l_x(z) + w_R mean_B l_x(z)]``

``          - [w_D mean_D f_x(P) + w_R mean_B f_x(P)]``,

where ``w_D=|D|/|U|`` and ``w_R=|U\\D|/|U|`` are exact registered rational
weights.  The D tangent is a typed fixed term reused in every block solve.

Concavity of the infimum of affine functions gives

``mean_B G_B <= inf_z F_U(z) - F_U(P)``.

Consequently, a positive lower confidence bound for the finite population of
block lower bounds proves strict full-population F separation.  This is a
scalar finite-population problem: it does not use a route-wise Bonferroni box,
per-stratum minimizers, coordinatewise substitutes, or caller-provided
e-values.  ``BoundedWoRBettingCS`` is reused conservatively on the sealed block
order; a full block census is compared exactly as a rational sum of binary64
observations.

The joint intentionally stops there.  Preferred-Q feasibility and the final
F/Q join are absent, so action invariance, C/K/O, mask commit, terminal, and
Pareto authority remain false even after strict sampled F separation.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from fractions import Fraction
import math
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.evidence.finite_population import BoundedWoRBettingCS

from .response_acquisition import SealedUniformPermutationReceipt


BLOCK_SUPPORT_ALIAS_SCHEMA: Final = (
    "partition_free_v1_block_support_branch_rank_resource_source_cardinality_alias_v1"
)
BLOCK_SUPPORT_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_equal_block_jensen_support_witness_registration_v1"
)
BLOCK_SUPPORT_SOLVER_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_block_shared_action_downward_solver_receipt_v1"
)
BLOCK_SUPPORT_PLANE_DEFICIT_SCHEMA: Final = (
    "partition_free_v1_block_support_preferred_plane_deficit_receipt_v1"
)
BLOCK_SUPPORT_D_FIXED_TERM_SCHEMA: Final = (
    "partition_free_v1_block_support_D_fixed_tangent_term_receipt_v1"
)
BLOCK_SUPPORT_OBSERVATION_SCHEMA: Final = (
    "partition_free_v1_block_support_margin_lower_bound_observation_v1"
)
BLOCK_SUPPORT_STATE_SCHEMA: Final = (
    "partition_free_v1_block_support_witness_compact_state_v1"
)
BLOCK_SUPPORT_DECISION_SCHEMA: Final = (
    "partition_free_v1_block_support_witness_decision_v1"
)
BLOCK_SUPPORT_JENSEN_REFERENCE_SCHEMA: Final = (
    "partition_free_v1_equal_block_jensen_reference_receipt_v1"
)

BLOCK_SUPPORT_PLANE_RULE: Final = (
    "per_context_full_axis_native_fiber_plane_is_affine_lower_support_with_"
    "certified_sign_or_y_zero_and_explicit_preferred_plane_deficit_v2"
)
BLOCK_SUPPORT_SHARED_MINIMUM_RULE: Final = (
    "one_shared_action_minimum_covers_all_constraints_within_shell_and_outside_pair_union_v1"
)
BLOCK_SUPPORT_JENSEN_RULE: Final = (
    "equal_residual_block_mean_of_exact_D_weight_plus_residual_weight_shared_"
    "action_affine_infima_lower_bounds_full_population_shared_action_infimum_v2"
)
BLOCK_SUPPORT_SOLVER_RULE: Final = (
    "global_shared_action_weighted_D_plus_residual_slope_lower_center_minus_"
    "registered_radius_minus_explicit_weighted_preferred_plane_deficits_v2"
)
BLOCK_SUPPORT_NUMERICAL_RULE: Final = (
    "binary64_nextafter_outward_difference_then_clip_up_to_preoutcome_analytic_lower_bound_v1"
)
BLOCK_SUPPORT_ALPHA_RULE: Final = (
    "trajectory_registered_preoutcome_nontransferable_nonrecycling_alpha_coordinate_v1"
)
BLOCK_SUPPORT_SAMPLING_RULE: Final = (
    "verified_SealedUniformPermutationReceipt_over_exact_U_minus_D_chunked_"
    "post_action_freeze_into_equal_blocks_v2"
)
BLOCK_SUPPORT_AUTHORITY_SCOPE: Final = (
    "strict_full_population_signed_F_separation_only_no_preferred_Q_full_join_or_downstream_authority_v2"
)
BLOCK_SUPPORT_SIGN_POLICY: Final = (
    "strict_sign_only_when_source_certified_otherwise_y_zero_with_explicit_"
    "preferred_plane_deficit_no_nominal_fp64_zero_sign_claim_v1"
)

_PREFIX: Final = "partition-free-v1-block-support-witness-v1-"
_REGISTRATION_TOKEN: Final = object()
_SOLVER_TOKEN: Final = object()
_DEFICIT_TOKEN: Final = object()
_D_FIXED_TOKEN: Final = object()
_OBSERVATION_TOKEN: Final = object()
_DECISION_TOKEN: Final = object()
_REFERENCE_TOKEN: Final = object()
_EMPTY_CHAIN_HASH: Final = content_hash(
    {"schema": "partition_free_v1_block_support_empty_chain_v1"},
    prefix=_PREFIX + "chain-",
)


class V1BlockSupportWitnessStatus(str, Enum):
    """Decision states of the scalar support-margin acquisition."""

    STABLE_STRICT_SEPARATION = "stable_strict_full_population_F_separation"
    ACQUIRE_MORE_BLOCKS = "acquire_more_sealed_blocks"
    FAIL_CLOSED = "failed_closed_no_strict_separation_authority"


class V1BlockSupportWitnessReason(str, Enum):
    """Stable machine-readable reasons for one decision state."""

    LOWER_CONFIDENCE_BOUND_STRICTLY_POSITIVE = (
        "lower_confidence_bound_strictly_positive"
    )
    EXACT_CENSUS_MEAN_STRICTLY_POSITIVE = "exact_census_mean_strictly_positive"
    UNRESOLVED_FINITE_POPULATION_INTERVAL = "unresolved_finite_population_interval"
    UPPER_CONFIDENCE_BOUND_NONPOSITIVE = "upper_confidence_bound_nonpositive"
    EXACT_CENSUS_MEAN_NONPOSITIVE = "exact_census_mean_nonpositive"
    NUMERICAL_OR_EMPTY_CONFIDENCE_SET = "numerical_or_empty_confidence_set"


class V1BlockSupportWitnessRefusalCode(str, Enum):
    """Typed structural failures that can never acquire authority."""

    INVALID_IDENTITY = "invalid_identity"
    INVALID_ALIAS_AXIS = "invalid_alias_axis"
    INVALID_POPULATION = "invalid_population"
    INVALID_PARTITION = "invalid_equal_block_partition"
    STALE_OR_NONFRESH_SOURCE = "stale_or_nonfresh_source"
    INVALID_RANGE = "invalid_analytic_observation_range"
    INVALID_ALPHA = "invalid_nonrecycling_alpha_allocation"
    SOURCE_MISMATCH = "source_identity_mismatch"
    INCOMPLETE_SHARED_MINIMUM = "incomplete_shared_action_minimum"
    INVALID_SOLVER_ENCLOSURE = "invalid_downward_solver_enclosure"
    OUT_OF_ORDER_BLOCK = "out_of_order_or_repeated_block"
    RECEIPT_TAMPERING = "receipt_integrity_failure"
    NUMERICAL_FAILURE = "numerical_failure"
    AUTHORITY_LEAKAGE = "authority_leakage"


class V1BlockSupportWitnessRefusal(RuntimeError):
    """Intentional fail-closed refusal at this statistical seam."""

    def __init__(self, code: V1BlockSupportWitnessRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1BlockSupportWitnessRefusalCode, message: str) -> NoReturn:
    raise V1BlockSupportWitnessRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_IDENTITY,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_IDENTITY,
            f"{name} must be one nonnegative integer",
        )
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_IDENTITY,
            f"{name} must be positive",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1BlockSupportWitnessRefusalCode.NUMERICAL_FAILURE,
            f"{name} must be finite",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_SOLVER_ENCLOSURE,
            f"{name} must be nonnegative",
        )
    return result


def _strict_false_authority(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value for value in values):
        _refuse(
            V1BlockSupportWitnessRefusalCode.AUTHORITY_LEAKAGE,
            f"{owner} attempted to open authority outside this joint",
        )


def _ordered_axis_hash(kind: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_ordered_identity_axis_v1",
            "kind": kind,
            "values": tuple(values),
        },
        prefix=_PREFIX + "axis-",
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportAliasBinding:
    """One member of the complete branch/rank/resource/source/cardinality axis."""

    branch_id: str
    fd_rank: int
    resource_alias_id: str
    source_receipt_hash: str
    source_replay_hash: str
    cardinality_alias_id: str
    cardinality: int
    constraint_cover_hash: str
    within_shell_cover_hash: str
    outside_pair_cover_hash: str
    schema: str = BLOCK_SUPPORT_ALIAS_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "branch_id",
            "resource_alias_id",
            "source_receipt_hash",
            "source_replay_hash",
            "cardinality_alias_id",
            "constraint_cover_hash",
            "within_shell_cover_hash",
            "outside_pair_cover_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(self, "fd_rank", _positive_int(self.fd_rank, name="FD rank"))
        object.__setattr__(
            self,
            "cardinality",
            _nonnegative_int(self.cardinality, name="cardinality"),
        )
        if self.schema != BLOCK_SUPPORT_ALIAS_SCHEMA:
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_ALIAS_AXIS,
                "unsupported alias schema",
            )

    @property
    def alias_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "alias-")


@dataclass(frozen=True, slots=True)
class V1BlockSupportPreferredPlaneDeficitReceipt:
    """Typed context-axis deficit for conservative numerical support planes.

    A source-certified nonzero anchor may use its strict sign.  Every
    sign-ambiguous cell instead uses ``y=0`` and contributes its enclosed
    preferred value to this nonnegative deficit.  In particular, nominal
    binary64 zero is never promoted to an exact mathematical sign claim.
    """

    support_plane_source_hash: str
    context_axis_hash: str
    context_count: int
    sign_policy_receipt_hash: str
    evaluated_context_fiber_cell_count: int
    source_certified_strict_sign_cell_count: int
    ambiguous_y_zero_cell_count: int
    preferred_plane_deficit_upper_bound: float
    deficit_numerical_radius: float
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    ambiguous_cells_use_y_zero: bool = True
    explicit_preferred_plane_deficit: bool = True
    nominal_fp64_zero_is_exact_sign: bool = False
    array_free_receipt: bool = True
    sign_policy: str = BLOCK_SUPPORT_SIGN_POLICY
    schema: str = BLOCK_SUPPORT_PLANE_DEFICIT_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _DEFICIT_TOKEN:
            raise TypeError("preferred-plane deficit receipts are issuer-only")
        for name in (
            "support_plane_source_hash",
            "context_axis_hash",
            "sign_policy_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "context_count",
            _positive_int(self.context_count, name="deficit context count"),
        )
        evaluated = _nonnegative_int(
            self.evaluated_context_fiber_cell_count,
            name="evaluated context-fiber cell count",
        )
        strict = _nonnegative_int(
            self.source_certified_strict_sign_cell_count,
            name="source-certified strict-sign cell count",
        )
        ambiguous = _nonnegative_int(
            self.ambiguous_y_zero_cell_count,
            name="ambiguous y-zero cell count",
        )
        if strict + ambiguous != evaluated:
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_SOLVER_ENCLOSURE,
                "strict-sign and y-zero cells must cover the evaluated support-plane cells",
            )
        object.__setattr__(self, "evaluated_context_fiber_cell_count", evaluated)
        object.__setattr__(self, "source_certified_strict_sign_cell_count", strict)
        object.__setattr__(self, "ambiguous_y_zero_cell_count", ambiguous)
        deficit = _finite_nonnegative(
            self.preferred_plane_deficit_upper_bound,
            name="preferred-plane deficit upper bound",
        )
        radius = _finite_nonnegative(
            self.deficit_numerical_radius,
            name="preferred-plane deficit numerical radius",
        )
        object.__setattr__(self, "preferred_plane_deficit_upper_bound", deficit)
        object.__setattr__(self, "deficit_numerical_radius", radius)
        if (
            self.ambiguous_cells_use_y_zero is not True
            or self.explicit_preferred_plane_deficit is not True
            or self.nominal_fp64_zero_is_exact_sign is not False
            or self.array_free_receipt is not True
            or self.sign_policy != BLOCK_SUPPORT_SIGN_POLICY
            or self.schema != BLOCK_SUPPORT_PLANE_DEFICIT_SCHEMA
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.AUTHORITY_LEAKAGE,
                "preferred-plane deficit receipt weakened the ambiguity policy",
            )
        object.__setattr__(self, "receipt_hash", self._compute_hash())

    @property
    def total_deficit_upper_outward(self) -> float:
        return math.nextafter(
            self.preferred_plane_deficit_upper_bound + self.deficit_numerical_radius,
            math.inf,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "support_plane_source_hash": self.support_plane_source_hash,
            "context_axis_hash": self.context_axis_hash,
            "context_count": self.context_count,
            "sign_policy_receipt_hash": self.sign_policy_receipt_hash,
            "cell_counts": (
                self.evaluated_context_fiber_cell_count,
                self.source_certified_strict_sign_cell_count,
                self.ambiguous_y_zero_cell_count,
            ),
            "preferred_plane_deficit_upper_bound": self.preferred_plane_deficit_upper_bound,
            "deficit_numerical_radius": self.deficit_numerical_radius,
            "ambiguous_cells_use_y_zero": self.ambiguous_cells_use_y_zero,
            "explicit_preferred_plane_deficit": self.explicit_preferred_plane_deficit,
            "nominal_fp64_zero_is_exact_sign": self.nominal_fp64_zero_is_exact_sign,
            "array_free_receipt": self.array_free_receipt,
            "sign_policy": self.sign_policy,
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "plane-deficit-")

    def assert_integrity(self) -> str:
        expected = self._compute_hash()
        if self.receipt_hash != expected:
            _refuse(
                V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                "preferred-plane deficit receipt hash no longer replays",
            )
        return expected


def issue_v1_block_support_preferred_plane_deficit(
    *,
    support_plane_source_hash: str,
    context_ids: Sequence[str],
    sign_policy_receipt_hash: str,
    evaluated_context_fiber_cell_count: int,
    source_certified_strict_sign_cell_count: int,
    ambiguous_y_zero_cell_count: int,
    preferred_plane_deficit_upper_bound: float,
    deficit_numerical_radius: float = 0.0,
) -> V1BlockSupportPreferredPlaneDeficitReceipt:
    contexts = tuple(_identity(value, name="deficit context") for value in context_ids)
    if not contexts or len(contexts) != len(set(contexts)):
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_POPULATION,
            "preferred-plane deficit context axis must be nonempty and unique",
        )
    return V1BlockSupportPreferredPlaneDeficitReceipt(
        support_plane_source_hash=support_plane_source_hash,
        context_axis_hash=_ordered_axis_hash(
            "support_plane_deficit_contexts", contexts
        ),
        context_count=len(contexts),
        sign_policy_receipt_hash=sign_policy_receipt_hash,
        evaluated_context_fiber_cell_count=evaluated_context_fiber_cell_count,
        source_certified_strict_sign_cell_count=source_certified_strict_sign_cell_count,
        ambiguous_y_zero_cell_count=ambiguous_y_zero_cell_count,
        preferred_plane_deficit_upper_bound=preferred_plane_deficit_upper_bound,
        deficit_numerical_radius=deficit_numerical_radius,
        _issuer_token=_DEFICIT_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportDFixedTermReceipt:
    """Typed frozen D tangent/deficit term reused by every residual block."""

    d_context_axis_hash: str
    d_context_count: int
    d_fixed_tangent_term_hash: str
    d_fixed_tangent_source_receipt_hash: str
    corrected_evidence_source_hash: str
    full_axis_support_plane_source_hash: str
    chart_hash: str
    current_mask_hash: str
    generation_index: int
    preferred_plane_deficit_receipt_hash: str
    preferred_plane_deficit_upper_outward: float
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    fixed_before_residual_block_outcomes: bool = True
    action_dependent_tangent_used_inside_shared_minimum: bool = True
    per_block_D_refit_forbidden: bool = True
    array_free_receipt: bool = True
    schema: str = BLOCK_SUPPORT_D_FIXED_TERM_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _D_FIXED_TOKEN:
            raise TypeError("D fixed-term receipts are issuer-only")
        for name in (
            "d_context_axis_hash",
            "d_fixed_tangent_term_hash",
            "d_fixed_tangent_source_receipt_hash",
            "corrected_evidence_source_hash",
            "full_axis_support_plane_source_hash",
            "chart_hash",
            "current_mask_hash",
            "preferred_plane_deficit_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "d_context_count",
            _positive_int(self.d_context_count, name="D context count"),
        )
        object.__setattr__(
            self,
            "generation_index",
            _nonnegative_int(self.generation_index, name="generation index"),
        )
        object.__setattr__(
            self,
            "preferred_plane_deficit_upper_outward",
            _finite_nonnegative(
                self.preferred_plane_deficit_upper_outward,
                name="D preferred-plane deficit upper",
            ),
        )
        if (
            self.fixed_before_residual_block_outcomes is not True
            or self.action_dependent_tangent_used_inside_shared_minimum is not True
            or self.per_block_D_refit_forbidden is not True
            or self.array_free_receipt is not True
            or self.schema != BLOCK_SUPPORT_D_FIXED_TERM_SCHEMA
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.AUTHORITY_LEAKAGE,
                "D fixed-term receipt weakened shared-minimum/freeze requirements",
            )
        object.__setattr__(self, "receipt_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "d_context_axis_hash": self.d_context_axis_hash,
            "d_context_count": self.d_context_count,
            "d_fixed_tangent_term_hash": self.d_fixed_tangent_term_hash,
            "d_fixed_tangent_source_receipt_hash": self.d_fixed_tangent_source_receipt_hash,
            "corrected_evidence_source_hash": self.corrected_evidence_source_hash,
            "full_axis_support_plane_source_hash": self.full_axis_support_plane_source_hash,
            "chart_hash": self.chart_hash,
            "current_mask_hash": self.current_mask_hash,
            "generation_index": self.generation_index,
            "preferred_plane_deficit_receipt_hash": self.preferred_plane_deficit_receipt_hash,
            "preferred_plane_deficit_upper_outward": self.preferred_plane_deficit_upper_outward,
            "fixed_before_residual_block_outcomes": self.fixed_before_residual_block_outcomes,
            "action_dependent_tangent_used_inside_shared_minimum": self.action_dependent_tangent_used_inside_shared_minimum,
            "per_block_D_refit_forbidden": self.per_block_D_refit_forbidden,
            "array_free_receipt": self.array_free_receipt,
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "D-fixed-")

    def assert_integrity(self) -> str:
        expected = self._compute_hash()
        if self.receipt_hash != expected:
            _refuse(
                V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                "D fixed-term receipt hash no longer replays",
            )
        return expected


def issue_v1_block_support_D_fixed_term(
    *,
    d_context_ids: Sequence[str],
    d_fixed_tangent_term_hash: str,
    d_fixed_tangent_source_receipt_hash: str,
    corrected_evidence_source_hash: str,
    full_axis_support_plane_source_hash: str,
    chart_hash: str,
    current_mask_hash: str,
    generation_index: int,
    preferred_plane_deficit_receipt: V1BlockSupportPreferredPlaneDeficitReceipt,
) -> V1BlockSupportDFixedTermReceipt:
    contexts = tuple(
        _identity(value, name="D fixed-term context") for value in d_context_ids
    )
    if not contexts or len(contexts) != len(set(contexts)):
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_POPULATION,
            "D fixed-term context axis must be nonempty and unique",
        )
    if (
        type(preferred_plane_deficit_receipt)
        is not V1BlockSupportPreferredPlaneDeficitReceipt
    ):
        raise TypeError("D fixed term requires a typed preferred-plane deficit receipt")
    preferred_plane_deficit_receipt.assert_integrity()
    expected_axis = _ordered_axis_hash("support_plane_deficit_contexts", contexts)
    if (
        preferred_plane_deficit_receipt.context_axis_hash != expected_axis
        or preferred_plane_deficit_receipt.context_count != len(contexts)
        or preferred_plane_deficit_receipt.support_plane_source_hash
        != full_axis_support_plane_source_hash
    ):
        _refuse(
            V1BlockSupportWitnessRefusalCode.SOURCE_MISMATCH,
            "D preferred-plane deficit differs from its fixed tangent source",
        )
    return V1BlockSupportDFixedTermReceipt(
        d_context_axis_hash=_ordered_axis_hash("D_union_in_U_order", contexts),
        d_context_count=len(contexts),
        d_fixed_tangent_term_hash=d_fixed_tangent_term_hash,
        d_fixed_tangent_source_receipt_hash=d_fixed_tangent_source_receipt_hash,
        corrected_evidence_source_hash=corrected_evidence_source_hash,
        full_axis_support_plane_source_hash=full_axis_support_plane_source_hash,
        chart_hash=chart_hash,
        current_mask_hash=current_mask_hash,
        generation_index=generation_index,
        preferred_plane_deficit_receipt_hash=preferred_plane_deficit_receipt.receipt_hash,
        preferred_plane_deficit_upper_outward=(
            preferred_plane_deficit_receipt.total_deficit_upper_outward
        ),
        _issuer_token=_D_FIXED_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportWitnessRegistration:
    """Compact, array-free pre-outcome registration for one witness attempt."""

    preferred_action_hash: str
    preferred_action_freeze_receipt_hash: str
    preferred_action_cardinality: int
    complete_alternate_domain_hash: str
    complete_alternate_domain_receipt_hash: str
    alternate_action_count: int
    complete_constraint_union_hash: str
    shared_action_minimum_domain_hash: str
    alias_bindings: tuple[V1BlockSupportAliasBinding, ...]
    alias_axis_hash: str
    corrected_evidence_source_hash: str
    full_axis_support_plane_source_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_source_hash: str
    chart_hash: str
    current_mask_hash: str
    route_axis_hash: str
    native_fiber_axis_hash: str
    generation_index: int
    full_population_axis_hash: str
    full_population_count: int
    d_fit_axis_hash: str
    d_fit_count: int
    d_cert_axis_hash: str
    d_cert_count: int
    d_union_axis_hash: str
    d_union_count: int
    d_fixed_term_receipt_hash: str
    d_fixed_tangent_term_hash: str
    d_preferred_plane_deficit_upper_outward: float
    residual_axis_hash: str
    residual_count: int
    d_weight_numerator: int
    residual_weight_numerator: int
    population_weight_denominator: int
    residual_permutation_receipt_hash: str
    residual_permutation_algorithm_id: str
    residual_generated_order_hash: str
    causal_partition_freeze_receipt_hash: str
    block_axis_hash: str
    betting_stratum_id: str
    sealed_block_order_hash: str
    block_count: int
    block_size: int
    action_freeze_sequence_index: int
    partition_freeze_sequence_index: int
    analytic_margin_lower_bound: float
    analytic_margin_upper_bound: float
    solver_lower_bound_policy_hash: str
    numerical_lower_bound_policy_hash: str
    alpha_ledger_hash: str
    alpha_coordinate_hash: str
    alpha_allocation_index: int
    family_alpha_budget: float
    family_alpha_spent_before: float
    attempt_alpha_allocation: float
    family_alpha_spent_after: float
    registration_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    complete_alternate_domain_registered: bool = True
    preferred_excluded_from_alternate_domain: bool = True
    complete_branch_rank_resource_source_cardinality_axis: bool = True
    complete_constraint_within_shell_outside_pair_union: bool = True
    shared_action_minimum_required: bool = True
    per_stratum_minimum_forbidden: bool = True
    coordinatewise_substitute_forbidden: bool = True
    full_axis_native_fiber_support_plane_required: bool = True
    exact_residual_is_U_minus_D: bool = True
    equal_block_partition: bool = True
    fresh_post_action_freeze_partition: bool = True
    no_D_context_reused_for_certification: bool = True
    alpha_allocation_nonrecycling: bool = True
    array_free_receipt: bool = True
    preferred_q_full_join_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    support_plane_rule: str = BLOCK_SUPPORT_PLANE_RULE
    shared_minimum_rule: str = BLOCK_SUPPORT_SHARED_MINIMUM_RULE
    jensen_rule: str = BLOCK_SUPPORT_JENSEN_RULE
    solver_rule: str = BLOCK_SUPPORT_SOLVER_RULE
    numerical_rule: str = BLOCK_SUPPORT_NUMERICAL_RULE
    alpha_rule: str = BLOCK_SUPPORT_ALPHA_RULE
    sampling_rule: str = BLOCK_SUPPORT_SAMPLING_RULE
    authority_scope: str = BLOCK_SUPPORT_AUTHORITY_SCOPE
    schema: str = BLOCK_SUPPORT_REGISTRATION_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _REGISTRATION_TOKEN:
            raise TypeError("block-support registrations are issuer-only")
        for name in (
            "preferred_action_hash",
            "preferred_action_freeze_receipt_hash",
            "complete_alternate_domain_hash",
            "complete_alternate_domain_receipt_hash",
            "complete_constraint_union_hash",
            "shared_action_minimum_domain_hash",
            "alias_axis_hash",
            "corrected_evidence_source_hash",
            "full_axis_support_plane_source_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_source_hash",
            "chart_hash",
            "current_mask_hash",
            "route_axis_hash",
            "native_fiber_axis_hash",
            "full_population_axis_hash",
            "d_fit_axis_hash",
            "d_cert_axis_hash",
            "d_union_axis_hash",
            "d_fixed_term_receipt_hash",
            "d_fixed_tangent_term_hash",
            "residual_axis_hash",
            "residual_permutation_receipt_hash",
            "residual_permutation_algorithm_id",
            "residual_generated_order_hash",
            "causal_partition_freeze_receipt_hash",
            "block_axis_hash",
            "betting_stratum_id",
            "sealed_block_order_hash",
            "solver_lower_bound_policy_hash",
            "numerical_lower_bound_policy_hash",
            "alpha_ledger_hash",
            "alpha_coordinate_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        aliases = tuple(self.alias_bindings)
        if (
            not aliases
            or any(type(value) is not V1BlockSupportAliasBinding for value in aliases)
            or len({value.alias_hash for value in aliases}) != len(aliases)
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_ALIAS_AXIS,
                "alias axis must be nonempty, typed, and unique",
            )
        expected_alias_hash = _ordered_axis_hash(
            "branch_rank_resource_source_cardinality_alias_hashes",
            tuple(value.alias_hash for value in aliases),
        )
        if self.alias_axis_hash != expected_alias_hash:
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_ALIAS_AXIS,
                "alias axis hash does not replay",
            )
        object.__setattr__(self, "alias_bindings", aliases)
        for name in (
            "preferred_action_cardinality",
            "generation_index",
            "d_fit_count",
            "d_cert_count",
            "d_union_count",
            "action_freeze_sequence_index",
            "alpha_allocation_index",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        for name in (
            "alternate_action_count",
            "full_population_count",
            "residual_count",
            "block_count",
            "block_size",
            "partition_freeze_sequence_index",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        if self.registered_d_prefix_hash == self.registered_d_w2_source_hash:
            _refuse(
                V1BlockSupportWitnessRefusalCode.STALE_OR_NONFRESH_SOURCE,
                "fresh block certification may not reuse the frozen D source",
            )
        if self.partition_freeze_sequence_index <= self.action_freeze_sequence_index:
            _refuse(
                V1BlockSupportWitnessRefusalCode.STALE_OR_NONFRESH_SOURCE,
                "equal-block partition must be frozen after the preferred action",
            )
        if self.block_count * self.block_size != self.residual_count:
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_PARTITION,
                "equal blocks must cover the residual population exactly",
            )
        for name in (
            "d_weight_numerator",
            "residual_weight_numerator",
            "population_weight_denominator",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        if (
            self.d_weight_numerator != self.d_union_count
            or self.residual_weight_numerator != self.residual_count
            or self.population_weight_denominator != self.full_population_count
            or self.d_weight_numerator + self.residual_weight_numerator
            != self.population_weight_denominator
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_POPULATION,
                "full-population D/residual weights must be exact context-count ratios",
            )
        object.__setattr__(
            self,
            "d_preferred_plane_deficit_upper_outward",
            _finite_nonnegative(
                self.d_preferred_plane_deficit_upper_outward,
                name="D preferred-plane deficit upper",
            ),
        )
        lower = _finite(self.analytic_margin_lower_bound, name="analytic margin lower")
        upper = _finite(self.analytic_margin_upper_bound, name="analytic margin upper")
        if not lower < upper:
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_RANGE,
                "analytic observation range must be finite and nondegenerate",
            )
        object.__setattr__(self, "analytic_margin_lower_bound", lower)
        object.__setattr__(self, "analytic_margin_upper_bound", upper)
        budget = _finite(self.family_alpha_budget, name="family alpha budget")
        before = _finite(self.family_alpha_spent_before, name="alpha spent before")
        attempt = _finite(self.attempt_alpha_allocation, name="attempt alpha")
        after = _finite(self.family_alpha_spent_after, name="alpha spent after")
        if (
            not 0.0 < budget < 1.0
            or before < 0.0
            or not 0.0 < attempt < 1.0
            or not math.isclose(after, before + attempt, rel_tol=0.0, abs_tol=1e-15)
            or after > budget + 1e-15
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_ALPHA,
                "alpha allocation is not one valid nonrecycling ledger coordinate",
            )
        object.__setattr__(self, "family_alpha_budget", budget)
        object.__setattr__(self, "family_alpha_spent_before", before)
        object.__setattr__(self, "attempt_alpha_allocation", attempt)
        object.__setattr__(self, "family_alpha_spent_after", after)
        required_true = (
            self.complete_alternate_domain_registered,
            self.preferred_excluded_from_alternate_domain,
            self.complete_branch_rank_resource_source_cardinality_axis,
            self.complete_constraint_within_shell_outside_pair_union,
            self.shared_action_minimum_required,
            self.per_stratum_minimum_forbidden,
            self.coordinatewise_substitute_forbidden,
            self.full_axis_native_fiber_support_plane_required,
            self.exact_residual_is_U_minus_D,
            self.equal_block_partition,
            self.fresh_post_action_freeze_partition,
            self.no_D_context_reused_for_certification,
            self.alpha_allocation_nonrecycling,
            self.array_free_receipt,
        )
        if any(value is not True for value in required_true):
            _refuse(
                V1BlockSupportWitnessRefusalCode.INCOMPLETE_SHARED_MINIMUM,
                "registration omitted a required theorem premise",
            )
        _strict_false_authority(
            (
                self.preferred_q_full_join_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="block-support registration",
        )
        if (
            self.support_plane_rule != BLOCK_SUPPORT_PLANE_RULE
            or self.shared_minimum_rule != BLOCK_SUPPORT_SHARED_MINIMUM_RULE
            or self.jensen_rule != BLOCK_SUPPORT_JENSEN_RULE
            or self.solver_rule != BLOCK_SUPPORT_SOLVER_RULE
            or self.numerical_rule != BLOCK_SUPPORT_NUMERICAL_RULE
            or self.alpha_rule != BLOCK_SUPPORT_ALPHA_RULE
            or self.sampling_rule != BLOCK_SUPPORT_SAMPLING_RULE
            or self.authority_scope != BLOCK_SUPPORT_AUTHORITY_SCOPE
            or self.schema != BLOCK_SUPPORT_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_IDENTITY,
                "registration rules or schema differ from the canonical joint",
            )
        object.__setattr__(self, "registration_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "preferred_action_hash": self.preferred_action_hash,
            "preferred_action_freeze_receipt_hash": self.preferred_action_freeze_receipt_hash,
            "preferred_action_cardinality": self.preferred_action_cardinality,
            "complete_alternate_domain_hash": self.complete_alternate_domain_hash,
            "complete_alternate_domain_receipt_hash": self.complete_alternate_domain_receipt_hash,
            "alternate_action_count": self.alternate_action_count,
            "complete_constraint_union_hash": self.complete_constraint_union_hash,
            "shared_action_minimum_domain_hash": self.shared_action_minimum_domain_hash,
            "alias_axis_hash": self.alias_axis_hash,
            "corrected_evidence_source_hash": self.corrected_evidence_source_hash,
            "full_axis_support_plane_source_hash": self.full_axis_support_plane_source_hash,
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "registered_d_w2_source_hash": self.registered_d_w2_source_hash,
            "chart_hash": self.chart_hash,
            "current_mask_hash": self.current_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "native_fiber_axis_hash": self.native_fiber_axis_hash,
            "generation_index": self.generation_index,
            "population": (
                self.full_population_axis_hash,
                self.full_population_count,
                self.d_fit_axis_hash,
                self.d_fit_count,
                self.d_cert_axis_hash,
                self.d_cert_count,
                self.d_union_axis_hash,
                self.d_union_count,
                self.d_fixed_term_receipt_hash,
                self.d_fixed_tangent_term_hash,
                self.d_preferred_plane_deficit_upper_outward,
                self.residual_axis_hash,
                self.residual_count,
                self.d_weight_numerator,
                self.residual_weight_numerator,
                self.population_weight_denominator,
            ),
            "partition": (
                self.residual_permutation_receipt_hash,
                self.residual_permutation_algorithm_id,
                self.residual_generated_order_hash,
                self.causal_partition_freeze_receipt_hash,
                self.block_axis_hash,
                self.betting_stratum_id,
                self.sealed_block_order_hash,
                self.block_count,
                self.block_size,
                self.action_freeze_sequence_index,
                self.partition_freeze_sequence_index,
            ),
            "analytic_range": (
                self.analytic_margin_lower_bound,
                self.analytic_margin_upper_bound,
            ),
            "solver_lower_bound_policy_hash": self.solver_lower_bound_policy_hash,
            "numerical_lower_bound_policy_hash": self.numerical_lower_bound_policy_hash,
            "alpha": (
                self.alpha_ledger_hash,
                self.alpha_coordinate_hash,
                self.alpha_allocation_index,
                self.family_alpha_budget,
                self.family_alpha_spent_before,
                self.attempt_alpha_allocation,
                self.family_alpha_spent_after,
            ),
            "rules": (
                self.support_plane_rule,
                self.shared_minimum_rule,
                self.jensen_rule,
                self.solver_rule,
                self.numerical_rule,
                self.alpha_rule,
                self.sampling_rule,
                self.authority_scope,
            ),
            "complete_theorem_premises": True,
            "all_downstream_authority": False,
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "registration-")

    def assert_integrity(self) -> str:
        expected = self._compute_hash()
        if self.registration_hash != expected:
            _refuse(
                V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                "registration hash no longer replays",
            )
        return expected


class V1BlockSupportWitnessSource:
    """Nonserializable live owner of the sealed context-to-block partition."""

    __slots__ = ("registration", "_blocks", "_block_ids", "_closed")

    def __init__(
        self,
        *,
        registration: V1BlockSupportWitnessRegistration,
        blocks: tuple[tuple[str, ...], ...],
        block_ids: tuple[str, ...],
    ) -> None:
        if type(registration) is not V1BlockSupportWitnessRegistration:
            raise TypeError("block-support source requires its canonical registration")
        registration.assert_integrity()
        self.registration = registration
        self._blocks = blocks
        self._block_ids = block_ids
        self._closed = False

    @property
    def block_ids(self) -> tuple[str, ...]:
        self._assert_live()
        return self._block_ids

    def block_context_ids(self, block_index: int) -> tuple[str, ...]:
        self._assert_live()
        index = _nonnegative_int(block_index, name="block index")
        if index >= len(self._blocks):
            _refuse(
                V1BlockSupportWitnessRefusalCode.OUT_OF_ORDER_BLOCK,
                "block index exceeds the sealed partition",
            )
        return self._blocks[index]

    def close(self) -> None:
        self._blocks = ()
        self._block_ids = ()
        self._closed = True

    def _assert_live(self) -> None:
        if self._closed:
            _refuse(
                V1BlockSupportWitnessRefusalCode.STALE_OR_NONFRESH_SOURCE,
                "sealed block source has been closed",
            )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live block-support sources are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("live block-support sources cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("live block-support sources cannot be copied")


def register_v1_signed_work_block_support_witness(
    *,
    preferred_action_hash: str,
    preferred_action_freeze_receipt_hash: str,
    preferred_action_cardinality: int,
    complete_alternate_domain_hash: str,
    complete_alternate_domain_receipt_hash: str,
    alternate_action_count: int,
    complete_constraint_union_hash: str,
    shared_action_minimum_domain_hash: str,
    alias_bindings: Sequence[V1BlockSupportAliasBinding],
    corrected_evidence_source_hash: str,
    full_axis_support_plane_source_hash: str,
    registered_d_prefix_hash: str,
    registered_d_w2_source_hash: str,
    chart_hash: str,
    current_mask_hash: str,
    route_axis_hash: str,
    native_fiber_axis_hash: str,
    generation_index: int,
    full_population_context_ids: Sequence[str],
    d_fit_context_ids: Sequence[str],
    d_cert_context_ids: Sequence[str],
    d_fixed_term_receipt: V1BlockSupportDFixedTermReceipt,
    residual_permutation_receipt: SealedUniformPermutationReceipt,
    block_size: int,
    causal_partition_freeze_receipt_hash: str,
    action_freeze_sequence_index: int,
    partition_freeze_sequence_index: int,
    analytic_margin_lower_bound: float,
    analytic_margin_upper_bound: float,
    solver_lower_bound_policy_hash: str,
    numerical_lower_bound_policy_hash: str,
    alpha_ledger_hash: str,
    alpha_allocation_index: int,
    family_alpha_budget: float,
    family_alpha_spent_before: float,
    attempt_alpha_allocation: float,
) -> V1BlockSupportWitnessSource:
    """Freeze the compact registration and its fresh equal-block live source."""

    aliases = tuple(alias_bindings)
    if (
        not aliases
        or any(type(value) is not V1BlockSupportAliasBinding for value in aliases)
        or len({value.alias_hash for value in aliases}) != len(aliases)
    ):
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_ALIAS_AXIS,
            "complete alias bindings must be typed and unique",
        )
    full = tuple(
        _identity(value, name="full-population context")
        for value in full_population_context_ids
    )
    d_fit = tuple(_identity(value, name="D-fit context") for value in d_fit_context_ids)
    d_cert = tuple(
        _identity(value, name="D-cert context") for value in d_cert_context_ids
    )
    if not full or len(full) != len(set(full)):
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_POPULATION,
            "full population must be a nonempty unique context axis",
        )
    if len(d_fit) != len(set(d_fit)) or len(d_cert) != len(set(d_cert)):
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_POPULATION,
            "D fit/cert axes may not repeat contexts",
        )
    if set(d_fit) & set(d_cert):
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_POPULATION,
            "D fit and D certification axes must be disjoint",
        )
    full_set = set(full)
    d_union_set = set(d_fit) | set(d_cert)
    if not d_union_set <= full_set:
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_POPULATION,
            "D axes must be subsets of the registered full population",
        )
    expected_residual = tuple(value for value in full if value not in d_union_set)
    if not expected_residual:
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_POPULATION,
            "U minus the complete D union must be nonempty",
        )
    residual = expected_residual
    d_union = tuple(value for value in full if value in d_union_set)
    if type(d_fixed_term_receipt) is not V1BlockSupportDFixedTermReceipt:
        raise TypeError(
            "block support registration requires a typed D fixed-term receipt"
        )
    d_fixed_term_receipt.assert_integrity()
    if (
        d_fixed_term_receipt.d_context_axis_hash
        != _ordered_axis_hash("D_union_in_U_order", d_union)
        or d_fixed_term_receipt.d_context_count != len(d_union)
        or d_fixed_term_receipt.corrected_evidence_source_hash
        != corrected_evidence_source_hash
        or d_fixed_term_receipt.full_axis_support_plane_source_hash
        != full_axis_support_plane_source_hash
        or d_fixed_term_receipt.chart_hash != chart_hash
        or d_fixed_term_receipt.current_mask_hash != current_mask_hash
        or d_fixed_term_receipt.generation_index != generation_index
    ):
        _refuse(
            V1BlockSupportWitnessRefusalCode.SOURCE_MISMATCH,
            "D fixed tangent/deficit receipt differs from the registered D/chart source",
        )
    if type(residual_permutation_receipt) is not SealedUniformPermutationReceipt:
        raise TypeError(
            "block support registration requires the exact sealed permutation receipt type"
        )
    try:
        residual_permutation_receipt.verify()
    except ValueError as error:
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_PARTITION,
            f"residual sealed permutation failed verification: {error}",
        )
    if residual_permutation_receipt.canonical_population != tuple(sorted(residual)):
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_PARTITION,
            "sealed permutation population differs from exact U minus D",
        )
    size = _positive_int(block_size, name="equal block size")
    if len(residual) % size != 0:
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_PARTITION,
            "verified residual permutation cannot be chunked into exact equal blocks",
        )
    generated_order = residual_permutation_receipt.generated_order
    blocks = tuple(
        tuple(generated_order[start : start + size])
        for start in range(0, len(generated_order), size)
    )
    if not blocks:
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_PARTITION,
            "sealed equal-block partition must be nonempty",
        )
    flattened = tuple(value for block in blocks for value in block)
    if len(flattened) != len(set(flattened)) or set(flattened) != set(residual):
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_PARTITION,
            "sealed blocks must cover every residual context exactly once",
        )
    if set(flattened) & d_union_set:
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_PARTITION,
            "fresh block evidence may not overlap D fit/cert contexts",
        )
    permutation_hash = residual_permutation_receipt.content_hash
    generated_order_hash = _ordered_axis_hash(
        "verified_residual_generated_order", generated_order
    )
    block_ids = tuple(
        content_hash(
            {
                "schema": "partition_free_v1_block_support_sealed_block_identity_v1",
                "residual_permutation_receipt_hash": permutation_hash,
                "sealed_block_index": index,
                "context_axis_hash": _ordered_axis_hash("block_contexts", block),
                "context_count": len(block),
            },
            prefix=_PREFIX + "block-",
        )
        for index, block in enumerate(blocks)
    )
    block_axis_hash = _ordered_axis_hash("sealed_block_ids", block_ids)
    betting_stratum_id = content_hash(
        {
            "schema": "partition_free_v1_block_support_betting_stratum_v1",
            "block_axis_hash": block_axis_hash,
            "residual_permutation_receipt_hash": permutation_hash,
        },
        prefix=_PREFIX + "stratum-",
    )
    sealed_order_hash = content_hash(
        {"stratum_id": betting_stratum_id, "context_ids": block_ids}
    )
    alias_axis_hash = _ordered_axis_hash(
        "branch_rank_resource_source_cardinality_alias_hashes",
        tuple(value.alias_hash for value in aliases),
    )
    budget = _finite(family_alpha_budget, name="family alpha budget")
    before = _finite(family_alpha_spent_before, name="family alpha spent before")
    attempt = _finite(attempt_alpha_allocation, name="attempt alpha allocation")
    after = before + attempt
    alpha_coordinate_hash = content_hash(
        {
            "schema": "partition_free_v1_block_support_alpha_coordinate_v1",
            "alpha_ledger_hash": _identity(alpha_ledger_hash, name="alpha ledger hash"),
            "allocation_index": _nonnegative_int(
                alpha_allocation_index, name="alpha allocation index"
            ),
            "family_alpha_spent_before": before,
            "attempt_alpha_allocation": attempt,
            "family_alpha_spent_after": after,
            "nonrecycling": True,
        },
        prefix=_PREFIX + "alpha-",
    )
    registration = V1BlockSupportWitnessRegistration(
        preferred_action_hash=preferred_action_hash,
        preferred_action_freeze_receipt_hash=preferred_action_freeze_receipt_hash,
        preferred_action_cardinality=preferred_action_cardinality,
        complete_alternate_domain_hash=complete_alternate_domain_hash,
        complete_alternate_domain_receipt_hash=complete_alternate_domain_receipt_hash,
        alternate_action_count=alternate_action_count,
        complete_constraint_union_hash=complete_constraint_union_hash,
        shared_action_minimum_domain_hash=shared_action_minimum_domain_hash,
        alias_bindings=aliases,
        alias_axis_hash=alias_axis_hash,
        corrected_evidence_source_hash=corrected_evidence_source_hash,
        full_axis_support_plane_source_hash=full_axis_support_plane_source_hash,
        registered_d_prefix_hash=registered_d_prefix_hash,
        registered_d_w2_source_hash=registered_d_w2_source_hash,
        chart_hash=chart_hash,
        current_mask_hash=current_mask_hash,
        route_axis_hash=route_axis_hash,
        native_fiber_axis_hash=native_fiber_axis_hash,
        generation_index=generation_index,
        full_population_axis_hash=_ordered_axis_hash("full_population", full),
        full_population_count=len(full),
        d_fit_axis_hash=_ordered_axis_hash("D_fit", d_fit),
        d_fit_count=len(d_fit),
        d_cert_axis_hash=_ordered_axis_hash("D_cert", d_cert),
        d_cert_count=len(d_cert),
        d_union_axis_hash=_ordered_axis_hash("D_union_in_U_order", d_union),
        d_union_count=len(d_union),
        d_fixed_term_receipt_hash=d_fixed_term_receipt.receipt_hash,
        d_fixed_tangent_term_hash=d_fixed_term_receipt.d_fixed_tangent_term_hash,
        d_preferred_plane_deficit_upper_outward=(
            d_fixed_term_receipt.preferred_plane_deficit_upper_outward
        ),
        residual_axis_hash=_ordered_axis_hash("U_minus_D_in_U_order", residual),
        residual_count=len(residual),
        d_weight_numerator=len(d_union),
        residual_weight_numerator=len(residual),
        population_weight_denominator=len(full),
        residual_permutation_receipt_hash=permutation_hash,
        residual_permutation_algorithm_id=residual_permutation_receipt.algorithm_id,
        residual_generated_order_hash=generated_order_hash,
        causal_partition_freeze_receipt_hash=causal_partition_freeze_receipt_hash,
        block_axis_hash=block_axis_hash,
        betting_stratum_id=betting_stratum_id,
        sealed_block_order_hash=sealed_order_hash,
        block_count=len(blocks),
        block_size=size,
        action_freeze_sequence_index=action_freeze_sequence_index,
        partition_freeze_sequence_index=partition_freeze_sequence_index,
        analytic_margin_lower_bound=analytic_margin_lower_bound,
        analytic_margin_upper_bound=analytic_margin_upper_bound,
        solver_lower_bound_policy_hash=solver_lower_bound_policy_hash,
        numerical_lower_bound_policy_hash=numerical_lower_bound_policy_hash,
        alpha_ledger_hash=alpha_ledger_hash,
        alpha_coordinate_hash=alpha_coordinate_hash,
        alpha_allocation_index=alpha_allocation_index,
        family_alpha_budget=budget,
        family_alpha_spent_before=before,
        attempt_alpha_allocation=attempt,
        family_alpha_spent_after=after,
        _issuer_token=_REGISTRATION_TOKEN,
    )
    return V1BlockSupportWitnessSource(
        registration=registration,
        blocks=blocks,
        block_ids=block_ids,
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportSharedMinimumSolverReceipt:
    """Issuer-only downward enclosure of one complete block shared minimum."""

    registration_hash: str
    block_index: int
    block_id: str
    block_context_axis_hash: str
    block_context_count: int
    preferred_action_hash: str
    complete_alternate_domain_hash: str
    complete_constraint_union_hash: str
    shared_action_minimum_domain_hash: str
    alias_axis_hash: str
    corrected_evidence_source_hash: str
    full_axis_support_plane_source_hash: str
    chart_hash: str
    current_mask_hash: str
    generation_index: int
    d_fixed_term_receipt_hash: str
    d_fixed_tangent_term_hash: str
    d_weight_numerator: int
    residual_weight_numerator: int
    population_weight_denominator: int
    residual_preferred_plane_deficit_receipt_hash: str
    solver_execution_receipt_hash: str
    support_plane_execution_receipt_hash: str
    weighted_shared_slope_minimum_lower_center: float
    solver_downward_numerical_radius: float
    d_preferred_plane_deficit_upper_outward: float
    residual_preferred_plane_deficit_upper_outward: float
    weighted_preferred_plane_deficit_upper_outward: float
    weighted_shared_slope_minimum_lower_outward: float
    raw_margin_lower_outward: float
    clipped_margin_lower_outward: float
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    complete_alias_axis_solved: bool = True
    complete_constraint_union_solved: bool = True
    within_shell_cover_solved: bool = True
    outside_pair_cover_solved: bool = True
    one_shared_action_minimum_solved: bool = True
    per_stratum_minimum_used: bool = False
    coordinatewise_substitute_used: bool = False
    global_lower_bound_certified: bool = True
    downward_numerical_enclosure: bool = True
    array_free_receipt: bool = True
    preferred_q_full_join_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    solver_rule: str = BLOCK_SUPPORT_SOLVER_RULE
    numerical_rule: str = BLOCK_SUPPORT_NUMERICAL_RULE
    schema: str = BLOCK_SUPPORT_SOLVER_RECEIPT_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _SOLVER_TOKEN:
            raise TypeError("block shared-minimum receipts are issuer-only")
        for name in (
            "registration_hash",
            "block_id",
            "block_context_axis_hash",
            "preferred_action_hash",
            "complete_alternate_domain_hash",
            "complete_constraint_union_hash",
            "shared_action_minimum_domain_hash",
            "alias_axis_hash",
            "corrected_evidence_source_hash",
            "full_axis_support_plane_source_hash",
            "chart_hash",
            "current_mask_hash",
            "d_fixed_term_receipt_hash",
            "d_fixed_tangent_term_hash",
            "residual_preferred_plane_deficit_receipt_hash",
            "solver_execution_receipt_hash",
            "support_plane_execution_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in ("block_index", "generation_index"):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "block_context_count",
            _positive_int(self.block_context_count, name="block context count"),
        )
        for name in (
            "d_weight_numerator",
            "residual_weight_numerator",
            "population_weight_denominator",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        if (
            self.d_weight_numerator + self.residual_weight_numerator
            != self.population_weight_denominator
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_POPULATION,
                "solver receipt full-population weights do not sum exactly",
            )
        for name in (
            "weighted_shared_slope_minimum_lower_center",
            "weighted_shared_slope_minimum_lower_outward",
            "raw_margin_lower_outward",
            "clipped_margin_lower_outward",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        for name in (
            "solver_downward_numerical_radius",
            "d_preferred_plane_deficit_upper_outward",
            "residual_preferred_plane_deficit_upper_outward",
            "weighted_preferred_plane_deficit_upper_outward",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        required_true = (
            self.complete_alias_axis_solved,
            self.complete_constraint_union_solved,
            self.within_shell_cover_solved,
            self.outside_pair_cover_solved,
            self.one_shared_action_minimum_solved,
            self.global_lower_bound_certified,
            self.downward_numerical_enclosure,
            self.array_free_receipt,
        )
        if (
            any(value is not True for value in required_true)
            or self.per_stratum_minimum_used is not False
            or self.coordinatewise_substitute_used is not False
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.INCOMPLETE_SHARED_MINIMUM,
                "solver receipt does not cover one complete shared-action minimum",
            )
        _strict_false_authority(
            (
                self.preferred_q_full_join_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="block shared-minimum solver receipt",
        )
        if (
            self.solver_rule != BLOCK_SUPPORT_SOLVER_RULE
            or self.numerical_rule != BLOCK_SUPPORT_NUMERICAL_RULE
            or self.schema != BLOCK_SUPPORT_SOLVER_RECEIPT_SCHEMA
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_SOLVER_ENCLOSURE,
                "solver receipt rules or schema differ",
            )
        object.__setattr__(self, "receipt_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "registration_hash": self.registration_hash,
            "block": (
                self.block_index,
                self.block_id,
                self.block_context_axis_hash,
                self.block_context_count,
            ),
            "source": (
                self.preferred_action_hash,
                self.complete_alternate_domain_hash,
                self.complete_constraint_union_hash,
                self.shared_action_minimum_domain_hash,
                self.alias_axis_hash,
                self.corrected_evidence_source_hash,
                self.full_axis_support_plane_source_hash,
                self.chart_hash,
                self.current_mask_hash,
                self.generation_index,
                self.d_fixed_term_receipt_hash,
                self.d_fixed_tangent_term_hash,
                self.d_weight_numerator,
                self.residual_weight_numerator,
                self.population_weight_denominator,
                self.residual_preferred_plane_deficit_receipt_hash,
            ),
            "execution": (
                self.solver_execution_receipt_hash,
                self.support_plane_execution_receipt_hash,
            ),
            "enclosure": (
                self.weighted_shared_slope_minimum_lower_center,
                self.solver_downward_numerical_radius,
                self.d_preferred_plane_deficit_upper_outward,
                self.residual_preferred_plane_deficit_upper_outward,
                self.weighted_preferred_plane_deficit_upper_outward,
                self.weighted_shared_slope_minimum_lower_outward,
                self.raw_margin_lower_outward,
                self.clipped_margin_lower_outward,
            ),
            "complete_shared_action_minimum": True,
            "substitutes_used": False,
            "rules": (self.solver_rule, self.numerical_rule),
            "all_downstream_authority": False,
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "solver-")

    def assert_integrity(self) -> str:
        expected = self._compute_hash()
        if self.receipt_hash != expected:
            _refuse(
                V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                "solver receipt hash no longer replays",
            )
        return expected


@dataclass(frozen=True, slots=True)
class V1BlockSupportMarginLowerBoundObservation:
    """Legacy/reference observation accepted by the reference block CS."""

    registration_hash: str
    solver_receipt_hash: str
    block_index: int
    block_id: str
    block_context_axis_hash: str
    block_context_count: int
    sealed_block_order_hash: str
    alias_axis_hash: str
    complete_alternate_domain_hash: str
    source_identity_hash: str
    margin_lower_bound: float
    observation_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    issuer_typed_downward_certified: bool = True
    shared_action_minimum: bool = True
    array_free_receipt: bool = True
    caller_supplied_e_factor: bool = False
    preferred_q_full_join_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = BLOCK_SUPPORT_OBSERVATION_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _OBSERVATION_TOKEN:
            raise TypeError("block support observations are issuer-only")
        for name in (
            "registration_hash",
            "solver_receipt_hash",
            "block_id",
            "block_context_axis_hash",
            "sealed_block_order_hash",
            "alias_axis_hash",
            "complete_alternate_domain_hash",
            "source_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self, "block_index", _nonnegative_int(self.block_index, name="block index")
        )
        object.__setattr__(
            self,
            "block_context_count",
            _positive_int(self.block_context_count, name="block context count"),
        )
        object.__setattr__(
            self,
            "margin_lower_bound",
            _finite(self.margin_lower_bound, name="margin lower bound"),
        )
        if (
            self.issuer_typed_downward_certified is not True
            or self.shared_action_minimum is not True
            or self.array_free_receipt is not True
            or self.caller_supplied_e_factor is not False
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.INCOMPLETE_SHARED_MINIMUM,
                "observation is not the canonical issued downward block bound",
            )
        _strict_false_authority(
            (
                self.preferred_q_full_join_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="block support observation",
        )
        if self.schema != BLOCK_SUPPORT_OBSERVATION_SCHEMA:
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_IDENTITY,
                "unsupported block observation schema",
            )
        object.__setattr__(self, "observation_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "registration_hash": self.registration_hash,
            "solver_receipt_hash": self.solver_receipt_hash,
            "block": (
                self.block_index,
                self.block_id,
                self.block_context_axis_hash,
                self.block_context_count,
            ),
            "sealed_block_order_hash": self.sealed_block_order_hash,
            "alias_axis_hash": self.alias_axis_hash,
            "complete_alternate_domain_hash": self.complete_alternate_domain_hash,
            "source_identity_hash": self.source_identity_hash,
            "margin_lower_bound": self.margin_lower_bound,
            "issuer_typed_downward_certified": True,
            "caller_supplied_e_factor": False,
            "all_downstream_authority": False,
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "observation-")

    def assert_integrity(self) -> str:
        expected = self._compute_hash()
        if self.observation_hash != expected:
            _refuse(
                V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                "block observation hash no longer replays",
            )
        return expected


def issue_v1_signed_work_block_support_lower_bound(
    *,
    source: V1BlockSupportWitnessSource,
    block_index: int,
    solver_execution_receipt_hash: str,
    support_plane_execution_receipt_hash: str,
    weighted_shared_slope_minimum_lower_center: float,
    solver_downward_numerical_radius: float,
    residual_preferred_plane_deficit_receipt: V1BlockSupportPreferredPlaneDeficitReceipt,
    alias_axis_hash: str | None = None,
    complete_alternate_domain_hash: str | None = None,
    full_axis_support_plane_source_hash: str | None = None,
    complete_alias_axis_solved: bool = True,
    complete_constraint_union_solved: bool = True,
    within_shell_cover_solved: bool = True,
    outside_pair_cover_solved: bool = True,
    one_shared_action_minimum_solved: bool = True,
    per_stratum_minimum_used: bool = False,
    coordinatewise_substitute_used: bool = False,
    global_lower_bound_certified: bool = True,
    downward_numerical_enclosure: bool = True,
) -> V1BlockSupportMarginLowerBoundObservation:
    """Issue one legacy/reference observation from supplied solver metadata.

    The solver center is the one shared-action minimum of the exact
    ``|D|/|U|`` D tangent plus ``|U\\D|/|U|`` block tangent.  It is *not* a sum
    of separately minimized D and residual objectives.  Registered numerical
    error and the explicit D/block preferred-plane deficits are subtracted
    outward.  Clipping *up* to the analytic lower bound remains downward-valid.

    This compatibility issuer does not validate a physical complete-cover
    receipt.  Consequently neither its observations nor its decisions are a
    production theorem boundary; the versioned production certifier exact-
    type-gates a source-bound complete-union envelope instead.
    """

    if type(source) is not V1BlockSupportWitnessSource:
        raise TypeError("block support issuance requires the live canonical source")
    registration = source.registration
    registration.assert_integrity()
    index = _nonnegative_int(block_index, name="block index")
    contexts = source.block_context_ids(index)
    block_id = source.block_ids[index]
    observed_alias_hash = (
        registration.alias_axis_hash
        if alias_axis_hash is None
        else _identity(alias_axis_hash, name="alias axis hash")
    )
    observed_domain_hash = (
        registration.complete_alternate_domain_hash
        if complete_alternate_domain_hash is None
        else _identity(complete_alternate_domain_hash, name="alternate domain hash")
    )
    observed_plane_hash = (
        registration.full_axis_support_plane_source_hash
        if full_axis_support_plane_source_hash is None
        else _identity(
            full_axis_support_plane_source_hash, name="support plane source hash"
        )
    )
    if (
        observed_alias_hash != registration.alias_axis_hash
        or observed_domain_hash != registration.complete_alternate_domain_hash
        or observed_plane_hash != registration.full_axis_support_plane_source_hash
    ):
        _refuse(
            V1BlockSupportWitnessRefusalCode.SOURCE_MISMATCH,
            "solver issuance names a different alias/domain/support-plane source",
        )
    if (
        type(residual_preferred_plane_deficit_receipt)
        is not V1BlockSupportPreferredPlaneDeficitReceipt
    ):
        raise TypeError(
            "block support issuance requires a typed residual preferred-plane deficit"
        )
    residual_preferred_plane_deficit_receipt.assert_integrity()
    block_context_axis_hash = _ordered_axis_hash("block_contexts", contexts)
    expected_deficit_axis = _ordered_axis_hash(
        "support_plane_deficit_contexts", contexts
    )
    if (
        residual_preferred_plane_deficit_receipt.context_axis_hash
        != expected_deficit_axis
        or residual_preferred_plane_deficit_receipt.context_count != len(contexts)
        or residual_preferred_plane_deficit_receipt.support_plane_source_hash
        != registration.full_axis_support_plane_source_hash
    ):
        _refuse(
            V1BlockSupportWitnessRefusalCode.SOURCE_MISMATCH,
            "residual preferred-plane deficit differs from the exact block/support source",
        )
    slope_center = _finite(
        weighted_shared_slope_minimum_lower_center,
        name="weighted shared slope minimum lower center",
    )
    solver_radius = _finite_nonnegative(
        solver_downward_numerical_radius,
        name="solver downward numerical radius",
    )
    slope_lower = math.nextafter(slope_center - solver_radius, -math.inf)
    d_deficit = registration.d_preferred_plane_deficit_upper_outward
    residual_deficit = (
        residual_preferred_plane_deficit_receipt.total_deficit_upper_outward
    )
    d_weight = Fraction(
        registration.d_weight_numerator,
        registration.population_weight_denominator,
    )
    residual_weight = Fraction(
        registration.residual_weight_numerator,
        registration.population_weight_denominator,
    )
    # Fraction multiplication prevents a rounded weight from silently changing
    # the registered population estimand; the final binary64 conversion is
    # explicitly rounded upward before subtraction.
    weighted_deficit_fraction = d_weight * Fraction.from_float(
        d_deficit
    ) + residual_weight * Fraction.from_float(residual_deficit)
    weighted_deficit = math.nextafter(float(weighted_deficit_fraction), math.inf)
    raw_margin = math.nextafter(slope_lower - weighted_deficit, -math.inf)
    if not all(
        math.isfinite(value) for value in (slope_lower, weighted_deficit, raw_margin)
    ):
        _refuse(
            V1BlockSupportWitnessRefusalCode.NUMERICAL_FAILURE,
            "outward solver arithmetic became nonfinite",
        )
    if raw_margin > registration.analytic_margin_upper_bound:
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_SOLVER_ENCLOSURE,
            "downward solver bound exceeds the analytic observation upper bound",
        )
    clipped_margin = max(raw_margin, registration.analytic_margin_lower_bound)
    solver_receipt = V1BlockSupportSharedMinimumSolverReceipt(
        registration_hash=registration.registration_hash,
        block_index=index,
        block_id=block_id,
        block_context_axis_hash=block_context_axis_hash,
        block_context_count=len(contexts),
        preferred_action_hash=registration.preferred_action_hash,
        complete_alternate_domain_hash=registration.complete_alternate_domain_hash,
        complete_constraint_union_hash=registration.complete_constraint_union_hash,
        shared_action_minimum_domain_hash=registration.shared_action_minimum_domain_hash,
        alias_axis_hash=registration.alias_axis_hash,
        corrected_evidence_source_hash=registration.corrected_evidence_source_hash,
        full_axis_support_plane_source_hash=registration.full_axis_support_plane_source_hash,
        chart_hash=registration.chart_hash,
        current_mask_hash=registration.current_mask_hash,
        generation_index=registration.generation_index,
        d_fixed_term_receipt_hash=registration.d_fixed_term_receipt_hash,
        d_fixed_tangent_term_hash=registration.d_fixed_tangent_term_hash,
        d_weight_numerator=registration.d_weight_numerator,
        residual_weight_numerator=registration.residual_weight_numerator,
        population_weight_denominator=registration.population_weight_denominator,
        residual_preferred_plane_deficit_receipt_hash=(
            residual_preferred_plane_deficit_receipt.receipt_hash
        ),
        solver_execution_receipt_hash=solver_execution_receipt_hash,
        support_plane_execution_receipt_hash=support_plane_execution_receipt_hash,
        weighted_shared_slope_minimum_lower_center=slope_center,
        solver_downward_numerical_radius=solver_radius,
        d_preferred_plane_deficit_upper_outward=d_deficit,
        residual_preferred_plane_deficit_upper_outward=residual_deficit,
        weighted_preferred_plane_deficit_upper_outward=weighted_deficit,
        weighted_shared_slope_minimum_lower_outward=slope_lower,
        raw_margin_lower_outward=raw_margin,
        clipped_margin_lower_outward=clipped_margin,
        complete_alias_axis_solved=complete_alias_axis_solved,
        complete_constraint_union_solved=complete_constraint_union_solved,
        within_shell_cover_solved=within_shell_cover_solved,
        outside_pair_cover_solved=outside_pair_cover_solved,
        one_shared_action_minimum_solved=one_shared_action_minimum_solved,
        per_stratum_minimum_used=per_stratum_minimum_used,
        coordinatewise_substitute_used=coordinatewise_substitute_used,
        global_lower_bound_certified=global_lower_bound_certified,
        downward_numerical_enclosure=downward_numerical_enclosure,
        _issuer_token=_SOLVER_TOKEN,
    )
    source_identity_hash = content_hash(
        {
            "registration_hash": registration.registration_hash,
            "corrected_evidence_source_hash": registration.corrected_evidence_source_hash,
            "full_axis_support_plane_source_hash": registration.full_axis_support_plane_source_hash,
            "registered_d_w2_source_hash": registration.registered_d_w2_source_hash,
            "chart_hash": registration.chart_hash,
            "current_mask_hash": registration.current_mask_hash,
            "generation_index": registration.generation_index,
        },
        prefix=_PREFIX + "source-identity-",
    )
    return V1BlockSupportMarginLowerBoundObservation(
        registration_hash=registration.registration_hash,
        solver_receipt_hash=solver_receipt.receipt_hash,
        block_index=index,
        block_id=block_id,
        block_context_axis_hash=block_context_axis_hash,
        block_context_count=len(contexts),
        sealed_block_order_hash=registration.sealed_block_order_hash,
        alias_axis_hash=registration.alias_axis_hash,
        complete_alternate_domain_hash=registration.complete_alternate_domain_hash,
        source_identity_hash=source_identity_hash,
        margin_lower_bound=clipped_margin,
        _issuer_token=_OBSERVATION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportWitnessState:
    """Compact append-only state; observation values remain in replay receipts."""

    registration_hash: str
    observed_block_count: int
    last_observation_hash: str | None
    receipt_chain_hash: str
    state_hash: str = field(init=False)
    schema: str = BLOCK_SUPPORT_STATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "registration_hash",
            _identity(self.registration_hash, name="registration hash"),
        )
        count = _nonnegative_int(self.observed_block_count, name="observed block count")
        object.__setattr__(self, "observed_block_count", count)
        if count == 0:
            if (
                self.last_observation_hash is not None
                or self.receipt_chain_hash != _EMPTY_CHAIN_HASH
            ):
                _refuse(
                    V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                    "empty state has a nonempty receipt chain",
                )
        else:
            object.__setattr__(
                self,
                "last_observation_hash",
                _identity(self.last_observation_hash, name="last observation hash"),
            )
            object.__setattr__(
                self,
                "receipt_chain_hash",
                _identity(self.receipt_chain_hash, name="receipt chain hash"),
            )
        if self.schema != BLOCK_SUPPORT_STATE_SCHEMA:
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_IDENTITY,
                "unsupported block witness state schema",
            )
        object.__setattr__(self, "state_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "registration_hash": self.registration_hash,
                "observed_block_count": self.observed_block_count,
                "last_observation_hash": self.last_observation_hash,
                "receipt_chain_hash": self.receipt_chain_hash,
            },
            prefix=_PREFIX + "state-",
        )

    def assert_integrity(self) -> str:
        expected = self._compute_hash()
        if self.state_hash != expected:
            _refuse(
                V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                "block witness state hash no longer replays",
            )
        return expected


@dataclass(frozen=True, slots=True)
class V1BlockSupportWitnessDecision:
    """Array-free scalar separation decision with deliberately closed authority."""

    registration_hash: str
    state_hash: str
    receipt_chain_hash: str
    status: V1BlockSupportWitnessStatus
    reason: V1BlockSupportWitnessReason
    observed_block_count: int
    block_population_count: int
    confidence_lower: float
    confidence_upper: float
    sample_mean_of_downward_bounds: float | None
    full_census: bool
    exact_census_sum_numerator: int | None
    exact_census_sum_denominator: int | None
    decision_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    sampled_full_population_F_strict_separation_authority: bool = False
    preferred_q_full_join_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    alpha_allocation_nonrecycling: bool = True
    array_free_receipt: bool = True
    authority_scope: str = BLOCK_SUPPORT_AUTHORITY_SCOPE
    schema: str = BLOCK_SUPPORT_DECISION_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _DECISION_TOKEN:
            raise TypeError("block support decisions are issuer-only")
        for name in ("registration_hash", "state_hash", "receipt_chain_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            type(self.status) is not V1BlockSupportWitnessStatus
            or type(self.reason) is not V1BlockSupportWitnessReason
        ):
            raise TypeError(
                "block support decision status/reason must be canonical enums"
            )
        observed = _nonnegative_int(
            self.observed_block_count, name="observed block count"
        )
        population = _positive_int(
            self.block_population_count, name="block population count"
        )
        if observed > population:
            _refuse(
                V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                "decision observes more blocks than its population",
            )
        object.__setattr__(self, "observed_block_count", observed)
        object.__setattr__(self, "block_population_count", population)
        lower = _finite(self.confidence_lower, name="confidence lower")
        upper = _finite(self.confidence_upper, name="confidence upper")
        if lower > upper:
            _refuse(
                V1BlockSupportWitnessRefusalCode.NUMERICAL_FAILURE,
                "decision confidence bounds are inverted",
            )
        object.__setattr__(self, "confidence_lower", lower)
        object.__setattr__(self, "confidence_upper", upper)
        if self.sample_mean_of_downward_bounds is not None:
            object.__setattr__(
                self,
                "sample_mean_of_downward_bounds",
                _finite(self.sample_mean_of_downward_bounds, name="sample mean"),
            )
        if type(self.full_census) is not bool or self.full_census != (
            observed == population
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                "decision census marker differs from counts",
            )
        if self.full_census:
            if (
                isinstance(self.exact_census_sum_numerator, bool)
                or not isinstance(self.exact_census_sum_numerator, int)
                or isinstance(self.exact_census_sum_denominator, bool)
                or not isinstance(self.exact_census_sum_denominator, int)
                or self.exact_census_sum_denominator <= 0
            ):
                _refuse(
                    V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                    "census decision must retain its exact rational sum",
                )
        elif (
            self.exact_census_sum_numerator is not None
            or self.exact_census_sum_denominator is not None
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                "partial decision cannot claim an exact census sum",
            )
        stable = self.status is V1BlockSupportWitnessStatus.STABLE_STRICT_SEPARATION
        if self.sampled_full_population_F_strict_separation_authority is not stable:
            _refuse(
                V1BlockSupportWitnessRefusalCode.AUTHORITY_LEAKAGE,
                "scalar F separation authority must equal the stable status",
            )
        if (
            stable
            and lower <= 0.0
            and not (
                self.full_census
                and self.exact_census_sum_numerator is not None
                and self.exact_census_sum_numerator > 0
            )
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.AUTHORITY_LEAKAGE,
                "strict separation lacks a positive confidence or exact census lower result",
            )
        _strict_false_authority(
            (
                self.preferred_q_full_join_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="block support decision",
        )
        if (
            self.alpha_allocation_nonrecycling is not True
            or self.array_free_receipt is not True
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.AUTHORITY_LEAKAGE,
                "decision weakened compact/nonrecycling guarantees",
            )
        if (
            self.authority_scope != BLOCK_SUPPORT_AUTHORITY_SCOPE
            or self.schema != BLOCK_SUPPORT_DECISION_SCHEMA
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_IDENTITY,
                "decision scope or schema differs",
            )
        object.__setattr__(self, "decision_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "registration_hash": self.registration_hash,
            "state_hash": self.state_hash,
            "receipt_chain_hash": self.receipt_chain_hash,
            "status": self.status.value,
            "reason": self.reason.value,
            "observed_block_count": self.observed_block_count,
            "block_population_count": self.block_population_count,
            "confidence_interval": (self.confidence_lower, self.confidence_upper),
            "sample_mean_of_downward_bounds": self.sample_mean_of_downward_bounds,
            "full_census": self.full_census,
            "exact_census_sum": (
                self.exact_census_sum_numerator,
                self.exact_census_sum_denominator,
            ),
            "sampled_full_population_F_strict_separation_authority": self.sampled_full_population_F_strict_separation_authority,
            "preferred_q_full_join_authority": False,
            "all_downstream_authority": False,
            "alpha_allocation_nonrecycling": True,
            "array_free_receipt": True,
            "authority_scope": self.authority_scope,
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "decision-")

    def assert_integrity(self) -> str:
        expected = self._compute_hash()
        if self.decision_hash != expected:
            _refuse(
                V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                "decision hash no longer replays",
            )
        return expected


class V1BlockSupportWitnessCertifier:
    """Mutable reference controller over issuer-typed sealed block receipts."""

    def __init__(self, source: V1BlockSupportWitnessSource) -> None:
        if type(source) is not V1BlockSupportWitnessSource:
            raise TypeError(
                "block support certifier requires its live canonical source"
            )
        source.registration.assert_integrity()
        self.source = source
        self._observations: list[V1BlockSupportMarginLowerBoundObservation] = []
        self._state = V1BlockSupportWitnessState(
            registration_hash=source.registration.registration_hash,
            observed_block_count=0,
            last_observation_hash=None,
            receipt_chain_hash=_EMPTY_CHAIN_HASH,
        )

    @property
    def state(self) -> V1BlockSupportWitnessState:
        self._state.assert_integrity()
        return self._state

    @property
    def observations(self) -> tuple[V1BlockSupportMarginLowerBoundObservation, ...]:
        return tuple(self._observations)

    def record(
        self, observation: V1BlockSupportMarginLowerBoundObservation
    ) -> V1BlockSupportWitnessState:
        if type(observation) is not V1BlockSupportMarginLowerBoundObservation:
            raise TypeError("block witness accepts only issuer-typed observations")
        observation.assert_integrity()
        registration = self.source.registration
        if observation.registration_hash != registration.registration_hash:
            _refuse(
                V1BlockSupportWitnessRefusalCode.SOURCE_MISMATCH,
                "observation names another registration",
            )
        expected_index = len(self._observations)
        if (
            observation.block_index != expected_index
            or observation.block_id != self.source.block_ids[expected_index]
            or observation.sealed_block_order_hash
            != registration.sealed_block_order_hash
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.OUT_OF_ORDER_BLOCK,
                "observation differs from the exact next sealed block",
            )
        contexts = self.source.block_context_ids(expected_index)
        if (
            observation.block_context_axis_hash
            != _ordered_axis_hash("block_contexts", contexts)
            or observation.block_context_count != len(contexts)
            or observation.alias_axis_hash != registration.alias_axis_hash
            or observation.complete_alternate_domain_hash
            != registration.complete_alternate_domain_hash
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.SOURCE_MISMATCH,
                "observation block/domain/alias identity differs from registration",
            )
        if not (
            registration.analytic_margin_lower_bound
            <= observation.margin_lower_bound
            <= registration.analytic_margin_upper_bound
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.INVALID_RANGE,
                "issued observation left the pre-outcome analytic range",
            )
        prior = self._state.receipt_chain_hash
        chain = content_hash(
            {
                "schema": "partition_free_v1_block_support_receipt_chain_link_v1",
                "prior_receipt_chain_hash": prior,
                "block_index": expected_index,
                "observation_hash": observation.observation_hash,
            },
            prefix=_PREFIX + "chain-",
        )
        self._observations.append(observation)
        self._state = V1BlockSupportWitnessState(
            registration_hash=registration.registration_hash,
            observed_block_count=len(self._observations),
            last_observation_hash=observation.observation_hash,
            receipt_chain_hash=chain,
        )
        return self._state

    def decision(self) -> V1BlockSupportWitnessDecision:
        registration = self.source.registration
        self._state.assert_integrity()
        sequence = BoundedWoRBettingCS(
            stratum_id=registration.betting_stratum_id,
            population_size=registration.block_count,
            alpha=registration.attempt_alpha_allocation,
            lower_bound=registration.analytic_margin_lower_bound,
            upper_bound=registration.analytic_margin_upper_bound,
            sealed_context_order=self.source.block_ids,
            sealed_order_hash=registration.sealed_block_order_hash,
            claim_bearing=True,
        )
        try:
            for observation in self._observations:
                sequence.record(
                    observation.margin_lower_bound,
                    context_id=observation.block_id,
                )
            snapshot = sequence.interval()
        except (ArithmeticError, FloatingPointError, ValueError) as error:
            _refuse(
                V1BlockSupportWitnessRefusalCode.NUMERICAL_FAILURE,
                f"bounded-WoR confidence sequence refused: {error}",
            )
        full_census = len(self._observations) == registration.block_count
        exact_sum: Fraction | None = None
        if full_census:
            exact_sum = sum(
                (
                    Fraction.from_float(value.margin_lower_bound)
                    for value in self._observations
                ),
                Fraction(0, 1),
            )
            if exact_sum > 0:
                status = V1BlockSupportWitnessStatus.STABLE_STRICT_SEPARATION
                reason = V1BlockSupportWitnessReason.EXACT_CENSUS_MEAN_STRICTLY_POSITIVE
            else:
                status = V1BlockSupportWitnessStatus.FAIL_CLOSED
                reason = V1BlockSupportWitnessReason.EXACT_CENSUS_MEAN_NONPOSITIVE
        elif (
            snapshot.empty
            or not math.isfinite(snapshot.lower)
            or not math.isfinite(snapshot.upper)
            or snapshot.lower > snapshot.upper
        ):
            status = V1BlockSupportWitnessStatus.FAIL_CLOSED
            reason = V1BlockSupportWitnessReason.NUMERICAL_OR_EMPTY_CONFIDENCE_SET
        elif snapshot.lower > 0.0:
            status = V1BlockSupportWitnessStatus.STABLE_STRICT_SEPARATION
            reason = (
                V1BlockSupportWitnessReason.LOWER_CONFIDENCE_BOUND_STRICTLY_POSITIVE
            )
        elif snapshot.upper <= 0.0:
            status = V1BlockSupportWitnessStatus.FAIL_CLOSED
            reason = V1BlockSupportWitnessReason.UPPER_CONFIDENCE_BOUND_NONPOSITIVE
        else:
            status = V1BlockSupportWitnessStatus.ACQUIRE_MORE_BLOCKS
            reason = V1BlockSupportWitnessReason.UNRESOLVED_FINITE_POPULATION_INTERVAL
        return V1BlockSupportWitnessDecision(
            registration_hash=registration.registration_hash,
            state_hash=self._state.state_hash,
            receipt_chain_hash=self._state.receipt_chain_hash,
            status=status,
            reason=reason,
            observed_block_count=len(self._observations),
            block_population_count=registration.block_count,
            confidence_lower=snapshot.lower,
            confidence_upper=snapshot.upper,
            sample_mean_of_downward_bounds=snapshot.sample_mean,
            full_census=full_census,
            exact_census_sum_numerator=(
                exact_sum.numerator if exact_sum is not None else None
            ),
            exact_census_sum_denominator=(
                exact_sum.denominator if exact_sum is not None else None
            ),
            sampled_full_population_F_strict_separation_authority=(
                status is V1BlockSupportWitnessStatus.STABLE_STRICT_SEPARATION
            ),
            _issuer_token=_DECISION_TOKEN,
        )


def require_v1_block_support_production_witness_decision(
    decision: V1BlockSupportWitnessDecision,
) -> NoReturn:
    """Reject the legacy/reference decision at every production boundary."""

    if type(decision) is not V1BlockSupportWitnessDecision:
        raise TypeError("legacy production refusal requires the exact decision type")
    decision.assert_integrity()
    _refuse(
        V1BlockSupportWitnessRefusalCode.AUTHORITY_LEAKAGE,
        "legacy caller-metadata block-support decisions cannot authorize "
        "production D/full-U/C; use the source-bound production decision schema",
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportJensenReferenceReceipt:
    """Array-free brute-force audit of the equal-block Jensen inequalities."""

    input_hash: str
    block_margin_lower_bounds: tuple[float, ...]
    mean_block_margin_lower_bound: float
    global_support_plane_gap: float
    global_exact_work_gap: float
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    mean_block_below_global_support_gap: bool = True
    support_gap_below_exact_gap: bool = True
    reference_only: bool = True
    statistical_authority: bool = False
    action_invariance_authority: bool = False
    schema: str = BLOCK_SUPPORT_JENSEN_REFERENCE_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _REFERENCE_TOKEN:
            raise TypeError("Jensen reference receipts are issuer-only")
        object.__setattr__(
            self, "input_hash", _identity(self.input_hash, name="reference input hash")
        )
        margins = tuple(
            _finite(value, name="block margin")
            for value in self.block_margin_lower_bounds
        )
        if not margins:
            raise ValueError("Jensen reference requires at least one block")
        object.__setattr__(self, "block_margin_lower_bounds", margins)
        for name in (
            "mean_block_margin_lower_bound",
            "global_support_plane_gap",
            "global_exact_work_gap",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if (
            self.mean_block_below_global_support_gap is not True
            or self.support_gap_below_exact_gap is not True
            or self.reference_only is not True
            or self.statistical_authority is not False
            or self.action_invariance_authority is not False
            or self.schema != BLOCK_SUPPORT_JENSEN_REFERENCE_SCHEMA
        ):
            _refuse(
                V1BlockSupportWitnessRefusalCode.AUTHORITY_LEAKAGE,
                "Jensen reference receipt changed theorem or authority",
            )
        object.__setattr__(self, "receipt_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "input_hash": self.input_hash,
                "block_margin_lower_bounds": self.block_margin_lower_bounds,
                "mean_block_margin_lower_bound": self.mean_block_margin_lower_bound,
                "global_support_plane_gap": self.global_support_plane_gap,
                "global_exact_work_gap": self.global_exact_work_gap,
                "theorem_holds": True,
                "reference_only": True,
                "all_authority": False,
            },
            prefix=_PREFIX + "jensen-reference-",
        )

    def assert_integrity(self) -> str:
        expected = self._compute_hash()
        if self.receipt_hash != expected:
            _refuse(
                V1BlockSupportWitnessRefusalCode.RECEIPT_TAMPERING,
                "Jensen reference receipt hash no longer replays",
            )
        return expected


def evaluate_v1_block_support_jensen_reference(
    *,
    d_preferred_work: Sequence[float],
    d_support_plane_work_by_alternate: Sequence[Sequence[float]],
    d_exact_work_by_alternate: Sequence[Sequence[float]],
    residual_preferred_work_by_block: Sequence[Sequence[float]],
    residual_support_plane_work_by_block_and_alternate: Sequence[
        Sequence[Sequence[float]]
    ],
    residual_exact_work_by_block_and_alternate: Sequence[Sequence[Sequence[float]]],
) -> V1BlockSupportJensenReferenceReceipt:
    """Brute-force the two deterministic inequalities on a tiny finite domain.

    The D axis is fixed in every block objective.  Residual axes are
    ``[block][context]`` and ``[block][alternate][context]``.  One alternate
    index is shared jointly across D and the residual block; separately
    minimizing the two strata is intentionally impossible here.
    """

    d_preferred = tuple(
        _finite(value, name="D preferred work") for value in d_preferred_work
    )
    d_support = tuple(
        tuple(_finite(value, name="D support-plane work") for value in alternate)
        for alternate in d_support_plane_work_by_alternate
    )
    d_exact = tuple(
        tuple(_finite(value, name="D exact alternate work") for value in alternate)
        for alternate in d_exact_work_by_alternate
    )
    preferred = tuple(
        tuple(_finite(value, name="residual preferred work") for value in block)
        for block in residual_preferred_work_by_block
    )
    support = tuple(
        tuple(
            tuple(
                _finite(value, name="residual support-plane work")
                for value in alternate
            )
            for alternate in block
        )
        for block in residual_support_plane_work_by_block_and_alternate
    )
    exact = tuple(
        tuple(
            tuple(
                _finite(value, name="residual exact alternate work")
                for value in alternate
            )
            for alternate in block
        )
        for block in residual_exact_work_by_block_and_alternate
    )
    if (
        not d_preferred
        or not preferred
        or len(preferred) != len(support)
        or len(preferred) != len(exact)
    ):
        raise ValueError("Jensen reference block axes differ or are empty")
    block_size = len(preferred[0])
    if block_size == 0 or any(len(block) != block_size for block in preferred):
        raise ValueError("Jensen reference requires nonempty equal-size blocks")
    alternate_count = len(support[0])
    if (
        alternate_count == 0
        or len(d_support) != alternate_count
        or len(d_exact) != alternate_count
        or any(len(values) != len(d_preferred) for values in d_support + d_exact)
    ):
        raise ValueError("Jensen reference D/residual alternate axes differ")
    for support_values, exact_values in zip(d_support, d_exact, strict=True):
        if any(
            left > right
            for left, right in zip(support_values, exact_values, strict=True)
        ):
            raise ValueError("support plane exceeds exact work on a D reference cell")
    for preferred_block, support_block, exact_block in zip(
        preferred, support, exact, strict=True
    ):
        if len(support_block) != alternate_count or len(exact_block) != alternate_count:
            raise ValueError("Jensen reference alternate axes differ")
        if any(
            len(values) != len(preferred_block)
            for values in support_block + exact_block
        ):
            raise ValueError("Jensen reference context axes differ")
        for support_values, exact_values in zip(
            support_block, exact_block, strict=True
        ):
            if any(
                left > right
                for left, right in zip(support_values, exact_values, strict=True)
            ):
                raise ValueError("support plane exceeds exact work on a reference cell")
    d_count = len(d_preferred)
    residual_count = len(preferred) * block_size
    full_count = d_count + residual_count
    d_weight = d_count / full_count
    residual_weight = residual_count / full_count
    d_preferred_mean = math.fsum(d_preferred) / d_count
    d_support_means = tuple(math.fsum(values) / d_count for values in d_support)
    d_exact_means = tuple(math.fsum(values) / d_count for values in d_exact)
    block_margins = []
    for preferred_block, support_block in zip(preferred, support, strict=True):
        preferred_mean = math.fsum(preferred_block) / block_size
        alternate_means = tuple(
            d_weight * d_support_means[index]
            + residual_weight * (math.fsum(values) / block_size)
            for index, values in enumerate(support_block)
        )
        preferred_full = d_weight * d_preferred_mean + residual_weight * preferred_mean
        block_margins.append(min(alternate_means) - preferred_full)
    global_residual_preferred = (
        math.fsum(value for block in preferred for value in block) / residual_count
    )
    global_preferred = (
        d_weight * d_preferred_mean + residual_weight * global_residual_preferred
    )
    global_residual_support_means = tuple(
        math.fsum(
            support[block_index][alternate_index][context_index]
            for block_index in range(len(preferred))
            for context_index in range(block_size)
        )
        / residual_count
        for alternate_index in range(alternate_count)
    )
    global_residual_exact_means = tuple(
        math.fsum(
            exact[block_index][alternate_index][context_index]
            for block_index in range(len(preferred))
            for context_index in range(block_size)
        )
        / residual_count
        for alternate_index in range(alternate_count)
    )
    global_support_means = tuple(
        d_weight * d_support_means[index]
        + residual_weight * global_residual_support_means[index]
        for index in range(alternate_count)
    )
    global_exact_means = tuple(
        d_weight * d_exact_means[index]
        + residual_weight * global_residual_exact_means[index]
        for index in range(alternate_count)
    )
    mean_block = math.fsum(block_margins) / len(block_margins)
    global_support_gap = min(global_support_means) - global_preferred
    global_exact_gap = min(global_exact_means) - global_preferred
    tolerance = 32.0 * math.ulp(
        max(1.0, abs(mean_block), abs(global_support_gap), abs(global_exact_gap))
    )
    if (
        mean_block > global_support_gap + tolerance
        or global_support_gap > global_exact_gap + tolerance
    ):
        _refuse(
            V1BlockSupportWitnessRefusalCode.INVALID_SOLVER_ENCLOSURE,
            "brute-force Jensen inequalities failed",
        )
    input_hash = content_hash(
        {
            "d_preferred_work": d_preferred,
            "d_support_plane_work_by_alternate": d_support,
            "d_exact_work_by_alternate": d_exact,
            "residual_preferred_work_by_block": preferred,
            "residual_support_plane_work_by_block_and_alternate": support,
            "residual_exact_work_by_block_and_alternate": exact,
        },
        prefix=_PREFIX + "jensen-input-",
    )
    return V1BlockSupportJensenReferenceReceipt(
        input_hash=input_hash,
        block_margin_lower_bounds=tuple(block_margins),
        mean_block_margin_lower_bound=mean_block,
        global_support_plane_gap=global_support_gap,
        global_exact_work_gap=global_exact_gap,
        _issuer_token=_REFERENCE_TOKEN,
    )


__all__ = [
    "BLOCK_SUPPORT_ALPHA_RULE",
    "BLOCK_SUPPORT_AUTHORITY_SCOPE",
    "BLOCK_SUPPORT_JENSEN_RULE",
    "BLOCK_SUPPORT_NUMERICAL_RULE",
    "BLOCK_SUPPORT_PLANE_RULE",
    "BLOCK_SUPPORT_SAMPLING_RULE",
    "BLOCK_SUPPORT_SHARED_MINIMUM_RULE",
    "BLOCK_SUPPORT_SIGN_POLICY",
    "BLOCK_SUPPORT_SOLVER_RULE",
    "V1BlockSupportAliasBinding",
    "V1BlockSupportDFixedTermReceipt",
    "V1BlockSupportJensenReferenceReceipt",
    "V1BlockSupportMarginLowerBoundObservation",
    "V1BlockSupportPreferredPlaneDeficitReceipt",
    "V1BlockSupportSharedMinimumSolverReceipt",
    "V1BlockSupportWitnessCertifier",
    "V1BlockSupportWitnessDecision",
    "V1BlockSupportWitnessReason",
    "V1BlockSupportWitnessRefusal",
    "V1BlockSupportWitnessRefusalCode",
    "V1BlockSupportWitnessRegistration",
    "V1BlockSupportWitnessSource",
    "V1BlockSupportWitnessState",
    "V1BlockSupportWitnessStatus",
    "evaluate_v1_block_support_jensen_reference",
    "issue_v1_block_support_D_fixed_term",
    "issue_v1_block_support_preferred_plane_deficit",
    "issue_v1_signed_work_block_support_lower_bound",
    "require_v1_block_support_production_witness_decision",
    "register_v1_signed_work_block_support_witness",
]
