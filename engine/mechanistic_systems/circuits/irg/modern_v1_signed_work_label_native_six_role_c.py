"""Label-native independent-C acquisition for signed-work modern v1.

This module is the representation-neutral successor to the materialized-mask
six-role C accumulator.  It binds each branch to the exact pre-C
``V1LabelNativePackedDirectionReceipt`` and its canonical device-shell action
axis.  No endpoint ``HardMask``, provider handle, label tensor, or compiled
gate is retained here.

One row closes six nonnegative observables for every registered action:

``F_native, Q_J, R_half, R_one, D_half, D_one``.

``F_native`` is copied from an outward native-fiber work interval.  ``Q_J`` is
derived from the literal signed relaxed-gate response interval.  ``R`` and
``D`` are derived from the same complete-action raw-logit secants.  The
derivation uses the exact binary64 interval arithmetic of the historical
fibered directional record.  Corrected-evidence, literal-J, and raw-logit
streams remain distinct identities while an explicit proof binds them to the
same capture/VJP execution.

The accumulator advances both KKT branches, every endpoint, and all six roles
in one copy-then-commit transaction.  Its snapshots contain only immutable
sufficient statistics and hash chains.  All objects are evidence-only: they
cannot select a candidate, open K/O, commit a mask, or claim terminal/Pareto
authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Final, Literal, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_device_shell_ladder import V1DeviceShellFrozenActionReceipt
from .modern_v1_fibered_directional_record import (
    _absolute_interval,
    _decimal,
    _outward_float,
)
from .modern_v1_fqr_sequential_acquisition import (
    _binary64_scaled_integer,
    _downward_fraction_to_probability,
    _downward_weighted_sum,
    _fraction_to_float_outward,
    _outward_weighted_sum,
    _response_burden,
    _response_burden_lower,
    _scaled_mean,
    _serfling_radius_upper,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_label_native_direction_provider import (
    V1LabelNativeCanonicalActionAxisRegistration,
    V1LabelNativePackedDirectionReceipt,
    V1LabelNativeQualifiedLaneRegistration,
)
from .modern_v1_signed_work_label_native_c_bridge import (
    V1SignedWorkLabelNativeCBranchReceipt,
    V1SignedWorkLabelNativeCBridgeFamilyBundle,
)
from .modern_v1_signed_work_six_role_c_acquisition import (
    D_HALF_ROLE,
    D_ONE_ROLE,
    F_NATIVE_ROLE,
    Q_J_ROLE,
    R_HALF_ROLE,
    R_ONE_ROLE,
    SIGNED_WORK_SIX_ROLE_DOSES,
    SIGNED_WORK_SIX_ROLES,
)


LABEL_NATIVE_SIX_ROLE_BRANCH_DESIGN_SCHEMA: Final = (
    "partition_free_v1_label_native_independent_C_six_role_branch_design_v1"
)
LABEL_NATIVE_SIX_ROLE_DESIGN_SCHEMA: Final = (
    "partition_free_v1_label_native_independent_C_six_role_design_v1"
)
LABEL_NATIVE_SIX_ROLE_ACTION_IDENTITY_SCHEMA: Final = (
    "partition_free_v1_label_native_six_role_action_identity_v1"
)
LABEL_NATIVE_SIX_ROLE_C_COORDINATE_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_label_native_six_role_C_coordinate_registration_v1"
)
LABEL_NATIVE_SIX_ROLE_C_SOURCE_EXECUTION_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_label_native_six_role_C_source_execution_registration_v1"
)
LABEL_NATIVE_SIX_ROLE_C_ELIGIBLE_SUBSET_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_label_native_six_role_C_eligible_subset_registration_v2"
)
LABEL_NATIVE_SIX_ROLE_ROW_SCHEMA: Final = (
    "partition_free_v1_label_native_independent_C_six_role_row_v1"
)
LABEL_NATIVE_SIX_ROLE_CELL_SCHEMA: Final = (
    "partition_free_v1_label_native_independent_C_six_role_cell_v1"
)
LABEL_NATIVE_SIX_ROLE_ENDPOINT_SCHEMA: Final = (
    "partition_free_v1_label_native_independent_C_six_role_endpoint_v1"
)
LABEL_NATIVE_SIX_ROLE_BRANCH_PROGRESS_SCHEMA: Final = (
    "partition_free_v1_label_native_independent_C_six_role_branch_progress_v1"
)
LABEL_NATIVE_SIX_ROLE_PROGRESS_SCHEMA: Final = (
    "partition_free_v1_label_native_independent_C_six_role_progress_v1"
)
LABEL_NATIVE_SIX_ROLE_SNAPSHOT_SCHEMA: Final = (
    "partition_free_v1_label_native_independent_C_six_role_snapshot_v1"
)
LABEL_NATIVE_SIX_ROLE_AUTHORITY: Final = (
    "fresh_independent_C_label_native_action_axis_native_F_literal_J_Q_"
    "directional_R_and_direct_D_evidence_only_no_selection_K_O_commit_"
    "terminal_or_pareto_authority_v1"
)
LABEL_NATIVE_SIX_ROLE_ALPHA_POLICY: Final = (
    "pre_C_branch_stratum_packed_action_role_two_tail_nonrecycling_all_prefix_v1"
)
LABEL_NATIVE_SIX_ROLE_SAMPLING_LAW: Final = (
    "sealed_without_replacement_prefix_of_complete_registered_train_C_v1"
)
LABEL_NATIVE_SIX_ROLE_METHOD: Final = (
    "separate_one_sided_lower_and_upper_hoeffding_serfling_v1"
)
LABEL_NATIVE_SIX_ROLE_CENSUS_METHOD: Final = (
    "exact_finite_population_census_of_outward_row_envelopes_v1"
)

_ACTION_PREFIX: Final = "modern-v1-label-native-six-role-action-"
_C_COORDINATE_REGISTRATION_PREFIX: Final = (
    "modern-v1-label-native-six-role-C-coordinate-registration-"
)
_C_SOURCE_EXECUTION_REGISTRATION_PREFIX: Final = (
    "modern-v1-label-native-six-role-C-source-execution-registration-"
)
_C_ELIGIBLE_SUBSET_REGISTRATION_PREFIX: Final = (
    "modern-v1-label-native-six-role-C-eligible-subset-registration-"
)
_BRANCH_DESIGN_PREFIX: Final = "modern-v1-label-native-six-role-C-branch-design-"
_DESIGN_PREFIX: Final = "modern-v1-label-native-six-role-C-design-"
_ROW_PREFIX: Final = "modern-v1-label-native-six-role-C-row-"
_CELL_PREFIX: Final = "modern-v1-label-native-six-role-C-cell-"
_ENDPOINT_PREFIX: Final = "modern-v1-label-native-six-role-C-endpoint-"
_BRANCH_PROGRESS_PREFIX: Final = "modern-v1-label-native-six-role-C-branch-progress-"
_PROGRESS_PREFIX: Final = "modern-v1-label-native-six-role-C-progress-"
_SNAPSHOT_PREFIX: Final = "modern-v1-label-native-six-role-C-snapshot-"
_CHAIN_PREFIX: Final = "modern-v1-label-native-six-role-C-chain-"
_LE_F64: Final = np.dtype("<f8")
_CANONICAL_BRANCHES: Final = (
    V1RelaxedKKTBranch.SPECTRAL,
    V1RelaxedKKTBranch.DIAGONAL_L1,
)

LabelNativeSixRoleSamplingStatus = Literal[
    "awaiting_C",
    "partial_C_evidence",
    "full_C_census",
]


class V1LabelNativeSixRoleCRefusalCode(str, Enum):
    """Stable refusal-code strings for the additive label-native seam."""

    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    STALE_COORDINATE = "stale_coordinate"
    ACTION_AXIS_MISMATCH = "action_axis_mismatch"
    CAPTURE_ALIAS = "capture_alias"
    STREAM_IDENTITY_MISMATCH = "stream_identity_mismatch"
    COORDINATE_PROOF_MISMATCH = "coordinate_proof_mismatch"
    EXECUTION_IDENTITY_MISMATCH = "execution_identity_mismatch"
    BRIDGE_IDENTITY_MISMATCH = "bridge_identity_mismatch"
    ELIGIBLE_SUBSET_MISMATCH = "eligible_subset_mismatch"
    INTERVAL_INVALID = "interval_invalid"
    SUPPORT_OVERFLOW = "support_overflow"
    BRANCH_MISMATCH = "branch_mismatch"
    SNAPSHOT_MISMATCH = "snapshot_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LabelNativeSixRoleCRefusal(ValueError):
    def __init__(
        self,
        code: V1LabelNativeSixRoleCRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LabelNativeSixRoleCRefusalCode:
            raise TypeError("label-native six-role refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LabelNativeSixRoleCRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LabelNativeSixRoleCRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
            f"{name} must be positive",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.INTERVAL_INVALID,
            f"{name} must be finite",
        )
    return 0.0 if result == 0.0 else result


def _finite_nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.INTERVAL_INVALID,
            f"{name} must be nonnegative",
        )
    return result


def _probability(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if not 0.0 < result <= 1.0:
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.INTERVAL_INVALID,
            f"{name} must lie in (0, 1]",
        )
    return result


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
            f"{name} must be nonempty and unique",
        )
    return result


def _axis_pairs(
    value: object,
    *,
    name: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
            f"{name} must be an ordered sequence",
        )
    result: list[tuple[str, tuple[str, ...]]] = []
    for raw in value:
        if type(raw) not in {tuple, list} or len(raw) != 2:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                f"{name} rows must be name/axis pairs",
            )
        axis_name = _identity(raw[0], name=f"{name} name")
        axis = _identity_axis(raw[1], name=f"{name} axis")
        result.append((axis_name, axis))
    rows = tuple(result)
    if not rows or len({item[0] for item in rows}) != len(rows):
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
            f"{name} names must be nonempty and unique",
        )
    return rows


def _immutable_f64(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
    nonnegative: bool = False,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=_LE_F64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
            f"{name} must be convertible to binary64: {error}",
        )
    if array.shape != shape or not bool(np.isfinite(array).all()):
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.INTERVAL_INVALID,
            f"{name} must be finite with shape {shape}",
        )
    if nonnegative and bool(np.any(array < 0.0)):
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.INTERVAL_INVALID,
            f"{name} must be nonnegative",
        )
    array[array == 0.0] = 0.0
    result = np.frombuffer(array.tobytes(order="C"), dtype=_LE_F64).reshape(shape)
    result.setflags(write=False)
    return result


def _action_identities(
    registration: V1LabelNativeCanonicalActionAxisRegistration,
    bridge_branch: V1SignedWorkLabelNativeCBranchReceipt,
    subset: "V1LabelNativeSixRoleCEligibleSubsetRegistration",
) -> tuple["V1LabelNativeSixRoleActionIdentity", ...]:
    source = registration.source_shell_ladder
    return tuple(
        V1LabelNativeSixRoleActionIdentity(
            endpoint_index=crosswalk.compact_C_endpoint_index,
            source_endpoint_label=crosswalk.source_endpoint_label,
            action_axis_registration_hash=registration.registration_hash,
            source_shell_ladder_receipt_hash=source.receipt_hash,
            compact_C_bridge_branch_receipt_hash=bridge_branch.receipt_hash,
            eligible_subset_registration_hash=subset.registration_hash,
            bridge_action_crosswalk_receipt_hash=crosswalk.receipt_hash,
            canonical_physical_identity_hash=(
                crosswalk.canonical_physical_identity_hash
            ),
            source_action_receipt=crosswalk.source_action_receipt,
        )
        for crosswalk in bridge_branch.eligible_action_crosswalks
    )


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleCCoordinateRegistration:
    """One pre-outcome C coordinate and its exact execution identities.

    These identities are intentionally registered before observations exist.
    A row carrying the right context/draw strings but an arbitrary proof hash
    therefore cannot enter the sufficient statistics.
    """

    stratum_id: str
    sealed_order_index: int
    context_id: str
    draw_id: str
    C_coordinate_proof_hash: str
    parent_execution_identity_hash: str
    same_capture_and_vjp_execution_proof_hash: str
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    schema: str = LABEL_NATIVE_SIX_ROLE_C_COORDINATE_REGISTRATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "stratum_id",
            "context_id",
            "draw_id",
            "C_coordinate_proof_hash",
            "parent_execution_identity_hash",
            "same_capture_and_vjp_execution_proof_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "sealed_order_index",
            _nonnegative_int(
                self.sealed_order_index,
                name="registered C sealed-order index",
            ),
        )
        if (
            type(self.issued_before_C_outcomes) is not bool
            or type(self.C_outcomes_seen) is not bool
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "C-coordinate registration flags must be exact booleans",
            )
        if not self.issued_before_C_outcomes or self.C_outcomes_seen:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "C-coordinate identity was not frozen before outcomes",
            )
        if self.schema != LABEL_NATIVE_SIX_ROLE_C_COORDINATE_REGISTRATION_SCHEMA:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "unsupported C-coordinate registration schema",
            )

    @property
    def registration_hash(self) -> str:
        return content_hash(self, prefix=_C_COORDINATE_REGISTRATION_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleCSourceExecutionRegistration:
    """Pre-outcome source authority for all label-native C rows.

    The registration binds the three scientifically distinct value streams,
    the code/source identities that issue coordinate and execution proofs,
    and the exact proof/execution identity expected at every sealed C
    coordinate.  It contains no outcomes.
    """

    registration_id: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    certification_context_population_hash: str
    certification_stream_commitment_hash: str
    joint_six_role_family_registration_hash: str
    same_capture_scorer_registration_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    raw_logit_stream_identity_hash: str
    C_coordinate_proof_source_registration_hash: str
    C_execution_registration_hash: str
    coordinates: tuple[V1LabelNativeSixRoleCCoordinateRegistration, ...]
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    observation_values_registered: bool = False
    schema: str = LABEL_NATIVE_SIX_ROLE_C_SOURCE_EXECUTION_REGISTRATION_SCHEMA
    _registration_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "registration_id",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "certification_context_population_hash",
            "certification_stream_commitment_hash",
            "joint_six_role_family_registration_hash",
            "same_capture_scorer_registration_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "corrected_evidence_stream_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "raw_logit_stream_identity_hash",
            "C_coordinate_proof_source_registration_hash",
            "C_execution_registration_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="registered C route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(
                self.proposal_generation,
                name="registered C proposal generation",
            ),
        )
        coordinates = tuple(self.coordinates)
        if not coordinates or any(
            type(value) is not V1LabelNativeSixRoleCCoordinateRegistration
            for value in coordinates
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "C source/execution registration requires exact coordinates",
            )
        coordinate_axis = tuple(
            (value.stratum_id, value.sealed_order_index)
            for value in coordinates
        )
        if len(coordinate_axis) != len(set(coordinate_axis)):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "C source/execution registration repeats a sealed coordinate",
            )
        proof_axes = (
            tuple(value.C_coordinate_proof_hash for value in coordinates),
            tuple(value.parent_execution_identity_hash for value in coordinates),
            tuple(
                value.same_capture_and_vjp_execution_proof_hash
                for value in coordinates
            ),
        )
        all_proof_identities = tuple(
            identity for axis in proof_axes for identity in axis
        )
        if (
            any(len(axis) != len(set(axis)) for axis in proof_axes)
            or len(all_proof_identities) != len(set(all_proof_identities))
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.CAPTURE_ALIAS,
                "C coordinates may not alias proof or execution identities",
            )
        streams = (
            self.corrected_evidence_stream_identity_hash,
            self.literal_jacobian_stream_identity_hash,
            self.raw_logit_stream_identity_hash,
        )
        if len(set(streams)) != 3:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.CAPTURE_ALIAS,
                "registered corrected, literal-J, and raw-logit streams must be distinct",
            )
        source_registrations = (
            self.C_coordinate_proof_source_registration_hash,
            self.C_execution_registration_hash,
        )
        if (
            len(set(source_registrations)) != 2
            or set(source_registrations).intersection(streams)
            or set(source_registrations).intersection(all_proof_identities)
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.CAPTURE_ALIAS,
                "C proof/execution source registrations must remain distinct",
            )
        flags = (
            self.issued_before_C_outcomes,
            self.C_outcomes_seen,
            self.observation_values_registered,
        )
        if any(type(value) is not bool for value in flags):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "C source/execution registration flags must be exact booleans",
            )
        if not self.issued_before_C_outcomes or any(flags[1:]):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "C source/execution registration contains post-outcome state",
            )
        if self.schema != LABEL_NATIVE_SIX_ROLE_C_SOURCE_EXECUTION_REGISTRATION_SCHEMA:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "unsupported C source/execution registration schema",
            )
        object.__setattr__(self, "coordinates", coordinates)
        self.assert_integrity()

    @property
    def stream_identity_hashes(self) -> tuple[str, str, str]:
        return (
            self.corrected_evidence_stream_identity_hash,
            self.literal_jacobian_stream_identity_hash,
            self.raw_logit_stream_identity_hash,
        )

    @property
    def coordinate_axis(
        self,
    ) -> tuple[tuple[str, int, str, str], ...]:
        return tuple(
            (
                value.stratum_id,
                value.sealed_order_index,
                value.context_id,
                value.draw_id,
            )
            for value in self.coordinates
        )

    def coordinate(
        self,
        *,
        stratum_id: str,
        sealed_order_index: int,
    ) -> V1LabelNativeSixRoleCCoordinateRegistration:
        stratum = _identity(stratum_id, name="registered C stratum lookup")
        position = _nonnegative_int(
            sealed_order_index,
            name="registered C coordinate lookup index",
        )
        for value in self.coordinates:
            if value.stratum_id == stratum and value.sealed_order_index == position:
                return value
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
            "C coordinate was not pre-registered",
        )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "registration_id",
                "chart_identity_hash",
                "current_mask_hash",
                "route_axis_hash",
                "route_count",
                "proposal_generation",
                "certification_context_population_hash",
                "certification_stream_commitment_hash",
                "joint_six_role_family_registration_hash",
                "same_capture_scorer_registration_hash",
                "native_fiber_axis_receipt_hash",
                "native_fiber_measure_hash",
                "corrected_evidence_stream_identity_hash",
                "literal_jacobian_stream_identity_hash",
                "raw_logit_stream_identity_hash",
                "C_coordinate_proof_source_registration_hash",
                "C_execution_registration_hash",
                "coordinates",
                "issued_before_C_outcomes",
                "C_outcomes_seen",
                "observation_values_registered",
            )
        }

    @property
    def registration_hash(self) -> str:
        cached = self._registration_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(),
            prefix=_C_SOURCE_EXECUTION_REGISTRATION_PREFIX,
        )
        if self._registration_hash is not None and observed != self._registration_hash:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "C source/execution registration was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "C source/execution registration hash does not match",
            )
        object.__setattr__(self, "_registration_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleCEligibleSubsetRegistration:
    """Exact pre-C map from the complete provider axis to Q-safe actions.

    The durable source identity always remains the complete shell action axis.
    A qualified provider executes only the Q-safe labels registered before C;
    the legacy/reference provider may execute the complete source and apply
    this same registration as a pre-outcome column filter.  Those two modes
    are explicit and mutually exclusive.
    """

    branch: V1RelaxedKKTBranch
    compact_C_bridge_branch_receipt_hash: str
    packed_direction_receipt_hash: str
    action_axis_registration_hash: str
    complete_source_endpoint_counts: tuple[int, ...]
    complete_source_action_receipt_hashes: tuple[str, ...]
    complete_source_action_hashes: tuple[str, ...]
    eligible_compact_C_endpoint_indices: tuple[int, ...]
    eligible_source_endpoint_labels: tuple[int, ...]
    eligible_endpoint_counts: tuple[int, ...]
    eligible_action_receipt_hashes: tuple[str, ...]
    eligible_action_hashes: tuple[str, ...]
    eligible_crosswalk_receipt_hashes: tuple[str, ...]
    eligible_canonical_physical_identity_hashes: tuple[str, ...]
    qualified_lane_registration_hash: str | None = None
    provider_action_axis_is_complete_source_axis: bool = True
    C_action_axis_is_preoutcome_eligible_subset: bool = True
    provider_native_subset_execution_available: bool = False
    preoutcome_registered_column_filter_required: bool = True
    ineligible_provider_columns_have_C_observation_authority: bool = False
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    schema: str = LABEL_NATIVE_SIX_ROLE_C_ELIGIBLE_SUBSET_REGISTRATION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "eligible subset branch must use the exact KKT enum",
            )
        for name in (
            "compact_C_bridge_branch_receipt_hash",
            "packed_direction_receipt_hash",
            "action_axis_registration_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        complete_counts = tuple(
            _positive_int(value, name="complete source endpoint count")
            for value in self.complete_source_endpoint_counts
        )
        complete_receipts = _identity_axis(
            self.complete_source_action_receipt_hashes,
            name="complete source action receipt axis",
        )
        complete_actions = _identity_axis(
            self.complete_source_action_hashes,
            name="complete source action axis",
        )
        compact = tuple(
            _positive_int(value, name="eligible compact C endpoint index")
            for value in self.eligible_compact_C_endpoint_indices
        )
        labels = tuple(
            _positive_int(value, name="eligible source endpoint label")
            for value in self.eligible_source_endpoint_labels
        )
        eligible_counts = tuple(
            _positive_int(value, name="eligible endpoint count")
            for value in self.eligible_endpoint_counts
        )
        eligible_receipts = _identity_axis(
            self.eligible_action_receipt_hashes,
            name="eligible action receipt axis",
        )
        eligible_actions = _identity_axis(
            self.eligible_action_hashes,
            name="eligible action axis",
        )
        crosswalks = _identity_axis(
            self.eligible_crosswalk_receipt_hashes,
            name="eligible crosswalk receipt axis",
        )
        physical = tuple(
            _identity(value, name="eligible canonical physical identity")
            for value in self.eligible_canonical_physical_identity_hashes
        )
        eligible_size = len(labels)
        if (
            not complete_counts
            or tuple(sorted(set(complete_counts))) != complete_counts
            or not (
                len(complete_counts)
                == len(complete_receipts)
                == len(complete_actions)
            )
            or eligible_size == 0
            or any(
                len(axis) != eligible_size
                for axis in (
                    compact,
                    eligible_counts,
                    eligible_receipts,
                    eligible_actions,
                    crosswalks,
                    physical,
                )
            )
            or compact != tuple(range(1, eligible_size + 1))
            or tuple(sorted(set(labels))) != labels
            or labels[-1] > len(complete_counts)
            or eligible_counts != tuple(complete_counts[label - 1] for label in labels)
            or eligible_receipts
            != tuple(complete_receipts[label - 1] for label in labels)
            or eligible_actions
            != tuple(complete_actions[label - 1] for label in labels)
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.ELIGIBLE_SUBSET_MISMATCH,
                "eligible C actions are not an exact source-labeled subset",
            )
        qualified_hash = self.qualified_lane_registration_hash
        if qualified_hash is not None:
            qualified_hash = _identity(
                qualified_hash,
                name="qualified lane registration hash",
            )
        structural_true = (
            self.provider_action_axis_is_complete_source_axis,
            self.C_action_axis_is_preoutcome_eligible_subset,
            self.issued_before_C_outcomes,
        )
        structural_false = (
            self.ineligible_provider_columns_have_C_observation_authority,
            self.C_outcomes_seen,
        )
        mode_flags = (
            self.provider_native_subset_execution_available,
            self.preoutcome_registered_column_filter_required,
        )
        if any(
            type(value) is not bool
            for value in structural_true + structural_false + mode_flags
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "eligible subset flags must be exact booleans",
            )
        qualified_mode = mode_flags == (True, False) and qualified_hash is not None
        reference_mode = mode_flags == (False, True) and qualified_hash is None
        if (
            not all(structural_true)
            or any(structural_false)
            or not (qualified_mode or reference_mode)
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "eligible subset crossed its qualified/reference pre-C authority",
            )
        if self.schema != LABEL_NATIVE_SIX_ROLE_C_ELIGIBLE_SUBSET_REGISTRATION_SCHEMA:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "unsupported eligible subset registration schema",
            )
        object.__setattr__(self, "complete_source_endpoint_counts", complete_counts)
        object.__setattr__(
            self,
            "complete_source_action_receipt_hashes",
            complete_receipts,
        )
        object.__setattr__(self, "complete_source_action_hashes", complete_actions)
        object.__setattr__(self, "eligible_compact_C_endpoint_indices", compact)
        object.__setattr__(self, "eligible_source_endpoint_labels", labels)
        object.__setattr__(self, "eligible_endpoint_counts", eligible_counts)
        object.__setattr__(self, "eligible_action_receipt_hashes", eligible_receipts)
        object.__setattr__(self, "eligible_action_hashes", eligible_actions)
        object.__setattr__(self, "eligible_crosswalk_receipt_hashes", crosswalks)
        object.__setattr__(
            self,
            "eligible_canonical_physical_identity_hashes",
            physical,
        )
        object.__setattr__(self, "qualified_lane_registration_hash", qualified_hash)

    @property
    def registration_hash(self) -> str:
        return content_hash(self, prefix=_C_ELIGIBLE_SUBSET_REGISTRATION_PREFIX)


def _eligible_subset_registration(
    *,
    branch: V1RelaxedKKTBranch,
    packed_direction_receipt: V1LabelNativePackedDirectionReceipt,
    action_axis_registration: V1LabelNativeCanonicalActionAxisRegistration,
    bridge_branch: V1SignedWorkLabelNativeCBranchReceipt,
) -> V1LabelNativeSixRoleCEligibleSubsetRegistration:
    bridge_branch.assert_integrity()
    source = action_axis_registration.source_shell_ladder
    if (
        bridge_branch.branch is not branch
        or packed_direction_receipt.action_axis_registration.registration_hash
        != action_axis_registration.registration_hash
        or packed_direction_receipt.endpoint_counts
        != bridge_branch.complete_source_endpoint_counts
        or packed_direction_receipt.endpoint_action_hashes
        != bridge_branch.complete_source_action_hashes
        or action_axis_registration.endpoint_counts
        != bridge_branch.complete_source_endpoint_counts
        or action_axis_registration.endpoint_action_hashes
        != bridge_branch.complete_source_action_hashes
        or action_axis_registration.source_action_receipt_hashes
        != bridge_branch.complete_source_action_receipt_hashes
        or source.receipt_hash != bridge_branch.shell_ladder_receipt_hash
        or source.route_axis_hash != bridge_branch.route_axis_hash
        or source.route_count != bridge_branch.route_count
        or source.parent_mask_hash != bridge_branch.parent_mask_hash
    ):
        _refuse(
            V1LabelNativeSixRoleCRefusalCode.BRIDGE_IDENTITY_MISMATCH,
            "provider full action axis differs from the compact C bridge source",
        )
    crosswalks = bridge_branch.eligible_action_crosswalks
    eligible_labels = tuple(value.source_endpoint_label for value in crosswalks)
    eligible_counts = tuple(value.cumulative_removed_count for value in crosswalks)
    eligible_receipts = tuple(
        value.source_action_receipt_hash for value in crosswalks
    )
    eligible_actions = tuple(value.packed_action_hash for value in crosswalks)
    eligible_crosswalks = tuple(value.receipt_hash for value in crosswalks)
    eligible_physical = tuple(
        value.canonical_physical_identity_hash for value in crosswalks
    )
    qualified = packed_direction_receipt.qualified_lane_registration
    qualified_hash: str | None = None
    native_subset_execution = False
    column_filter_required = True
    if qualified is not None:
        if type(qualified) is not V1LabelNativeQualifiedLaneRegistration:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "qualified provider receipt requires an exact lane registration",
            )
        qualified_hash = qualified.assert_integrity()
        if (
            qualified.branch_value != branch.value
            or qualified.action_axis_registration_hash
            != action_axis_registration.registration_hash
            or qualified.compact_C_bridge_branch_receipt_hash
            != bridge_branch.receipt_hash
            or qualified.complete_source_endpoint_counts
            != bridge_branch.complete_source_endpoint_counts
            or qualified.complete_source_action_receipt_hashes
            != bridge_branch.complete_source_action_receipt_hashes
            or qualified.complete_source_action_hashes
            != bridge_branch.complete_source_action_hashes
            or qualified.source_endpoint_labels != eligible_labels
            or qualified.endpoint_counts != eligible_counts
            or qualified.endpoint_action_receipt_hashes != eligible_receipts
            or qualified.endpoint_action_hashes != eligible_actions
            or qualified.action_crosswalk_receipt_hashes != eligible_crosswalks
            or qualified.canonical_physical_identity_hashes != eligible_physical
            or packed_direction_receipt.executed_source_endpoint_labels
            != eligible_labels
            or packed_direction_receipt.executed_endpoint_counts != eligible_counts
            or packed_direction_receipt.executed_endpoint_action_hashes
            != eligible_actions
            or packed_direction_receipt.compact_provider_to_source_endpoint_labels
            != eligible_labels
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.ELIGIBLE_SUBSET_MISMATCH,
                "qualified provider execution differs from the frozen Q-safe subset",
            )
        native_subset_execution = True
        column_filter_required = False
    return V1LabelNativeSixRoleCEligibleSubsetRegistration(
        branch=branch,
        compact_C_bridge_branch_receipt_hash=bridge_branch.receipt_hash,
        packed_direction_receipt_hash=packed_direction_receipt.receipt_hash,
        action_axis_registration_hash=action_axis_registration.registration_hash,
        complete_source_endpoint_counts=(
            bridge_branch.complete_source_endpoint_counts
        ),
        complete_source_action_receipt_hashes=(
            bridge_branch.complete_source_action_receipt_hashes
        ),
        complete_source_action_hashes=bridge_branch.complete_source_action_hashes,
        eligible_compact_C_endpoint_indices=tuple(
            value.compact_C_endpoint_index for value in crosswalks
        ),
        eligible_source_endpoint_labels=eligible_labels,
        eligible_endpoint_counts=eligible_counts,
        eligible_action_receipt_hashes=eligible_receipts,
        eligible_action_hashes=eligible_actions,
        eligible_crosswalk_receipt_hashes=eligible_crosswalks,
        eligible_canonical_physical_identity_hashes=eligible_physical,
        qualified_lane_registration_hash=qualified_hash,
        provider_native_subset_execution_available=native_subset_execution,
        preoutcome_registered_column_filter_required=column_filter_required,
    )


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleActionIdentity:
    """Compact exact provenance for one canonical packed hard action."""

    endpoint_index: int
    source_endpoint_label: int
    action_axis_registration_hash: str
    source_shell_ladder_receipt_hash: str
    compact_C_bridge_branch_receipt_hash: str
    eligible_subset_registration_hash: str
    bridge_action_crosswalk_receipt_hash: str
    canonical_physical_identity_hash: str
    source_action_receipt: V1DeviceShellFrozenActionReceipt
    schema: str = LABEL_NATIVE_SIX_ROLE_ACTION_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        endpoint = _positive_int(self.endpoint_index, name="endpoint index")
        source_endpoint = _positive_int(
            self.source_endpoint_label,
            name="source endpoint label",
        )
        object.__setattr__(self, "endpoint_index", endpoint)
        object.__setattr__(self, "source_endpoint_label", source_endpoint)
        for name in (
            "action_axis_registration_hash",
            "source_shell_ladder_receipt_hash",
            "compact_C_bridge_branch_receipt_hash",
            "eligible_subset_registration_hash",
            "bridge_action_crosswalk_receipt_hash",
            "canonical_physical_identity_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        if type(self.source_action_receipt) is not V1DeviceShellFrozenActionReceipt:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "action identity requires an exact device-shell action receipt",
            )
        if self.source_action_receipt.endpoint_label != source_endpoint:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.ACTION_AXIS_MISMATCH,
                "action source endpoint differs from its canonical shell label",
            )
        if self.schema != LABEL_NATIVE_SIX_ROLE_ACTION_IDENTITY_SCHEMA:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "unsupported label-native action identity schema",
            )

    @property
    def packed_action_hash(self) -> str:
        return self.source_action_receipt.packed_action_hash

    @property
    def cumulative_removed_count(self) -> int:
        return self.source_action_receipt.cumulative_removed_count

    @property
    def identity_hash(self) -> str:
        return content_hash(self, prefix=_ACTION_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleCBranchDesign:
    """One exact pre-C branch and its six analytic support families."""

    branch: V1RelaxedKKTBranch
    packed_direction_receipt: V1LabelNativePackedDirectionReceipt = field(
        repr=False,
        compare=False,
    )
    action_axis_registration: V1LabelNativeCanonicalActionAxisRegistration
    compact_C_bridge_branch_receipt: V1SignedWorkLabelNativeCBranchReceipt = field(
        repr=False,
        compare=False,
    )
    C_source_execution_registration: (
        V1LabelNativeSixRoleCSourceExecutionRegistration
    ) = field(repr=False, compare=False)
    chart_identity_hash: str
    proposal_generation: int
    certification_context_population_hash: str
    certification_stream_commitment_hash: str
    joint_six_role_family_registration_hash: str
    same_capture_scorer_registration_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    trajectory_alpha_ledger_hash: str
    trajectory_alpha_ledger_generation: int
    analytic_support_upper_by_stratum_endpoint_role: np.ndarray = field(
        repr=False,
        compare=False,
    )
    support_bound_source_hashes_by_role: tuple[tuple[str, str], ...]
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    D_values_reused_as_C_observations: bool = False
    endpoint_hard_masks_registered: bool = False
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = LABEL_NATIVE_SIX_ROLE_AUTHORITY
    schema: str = LABEL_NATIVE_SIX_ROLE_BRANCH_DESIGN_SCHEMA
    action_identities: tuple[V1LabelNativeSixRoleActionIdentity, ...] = field(
        init=False,
    )
    eligible_action_subset_registration: (
        V1LabelNativeSixRoleCEligibleSubsetRegistration
    ) = field(init=False)
    _design_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "branch design requires the exact KKT branch enum",
            )
        if type(self.packed_direction_receipt) is not V1LabelNativePackedDirectionReceipt:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "branch design requires an exact label-native provider receipt",
            )
        if type(self.action_axis_registration) is not (
            V1LabelNativeCanonicalActionAxisRegistration
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "branch design requires an exact canonical action registration",
            )
        if type(self.compact_C_bridge_branch_receipt) is not (
            V1SignedWorkLabelNativeCBranchReceipt
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "branch design requires an exact compact C-bridge branch receipt",
            )
        self.compact_C_bridge_branch_receipt.assert_integrity()
        if type(self.C_source_execution_registration) is not (
            V1LabelNativeSixRoleCSourceExecutionRegistration
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "branch design requires an exact pre-C source/execution registration",
            )
        self.C_source_execution_registration.assert_integrity()
        for name in (
            "chart_identity_hash",
            "certification_context_population_hash",
            "certification_stream_commitment_hash",
            "joint_six_role_family_registration_hash",
            "same_capture_scorer_registration_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "trajectory_alpha_ledger_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        generation = _nonnegative_int(
            self.proposal_generation,
            name="branch proposal generation",
        )
        ledger_generation = _nonnegative_int(
            self.trajectory_alpha_ledger_generation,
            name="branch trajectory alpha-ledger generation",
        )
        if ledger_generation != generation:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "branch alpha ledger belongs to another proposal generation",
            )
        source = self.C_source_execution_registration
        if (
            source.chart_identity_hash != self.chart_identity_hash
            or source.proposal_generation != generation
            or source.certification_context_population_hash
            != self.certification_context_population_hash
            or source.certification_stream_commitment_hash
            != self.certification_stream_commitment_hash
            or source.joint_six_role_family_registration_hash
            != self.joint_six_role_family_registration_hash
            or source.same_capture_scorer_registration_hash
            != self.same_capture_scorer_registration_hash
            or source.native_fiber_axis_receipt_hash
            != self.native_fiber_axis_receipt_hash
            or source.native_fiber_measure_hash != self.native_fiber_measure_hash
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "branch metadata differs from its pre-C source/execution registration",
            )
        subset = _eligible_subset_registration(
            branch=self.branch,
            packed_direction_receipt=self.packed_direction_receipt,
            action_axis_registration=self.action_axis_registration,
            bridge_branch=self.compact_C_bridge_branch_receipt,
        )
        object.__setattr__(
            self,
            "eligible_action_subset_registration",
            subset,
        )
        self._assert_action_source_binding()
        action_identities = _action_identities(
            self.action_axis_registration,
            self.compact_C_bridge_branch_receipt,
            subset,
        )
        endpoint_count = len(action_identities)
        try:
            raw_support = np.asarray(
                self.analytic_support_upper_by_stratum_endpoint_role
            )
            stratum_count = int(raw_support.shape[0]) if raw_support.ndim == 3 else 0
        except (AttributeError, TypeError, ValueError):
            stratum_count = 0
        if stratum_count <= 0:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INTERVAL_INVALID,
                "branch support must register at least one stratum",
            )
        support = _immutable_f64(
            self.analytic_support_upper_by_stratum_endpoint_role,
            shape=(stratum_count, endpoint_count, len(SIGNED_WORK_SIX_ROLES)),
            name="six-role analytic support",
            nonnegative=True,
        )
        sources = tuple(
            (
                _identity(role, name="support role"),
                _identity(source_hash, name="support source hash"),
            )
            for role, source_hash in self.support_bound_source_hashes_by_role
        )
        if tuple(role for role, _source in sources) != SIGNED_WORK_SIX_ROLES:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "support source axis must cover the canonical six roles in order",
            )
        required_true = (self.issued_before_C_outcomes,)
        required_false = (
            self.C_outcomes_seen,
            self.D_values_reused_as_C_observations,
            self.endpoint_hard_masks_registered,
            self.candidate_selection_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in required_true + required_false):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "branch authority flags must be exact booleans",
            )
        if not all(required_true) or any(required_false):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "branch design crossed its pre-outcome evidence boundary",
            )
        if (
            self.authority_scope != LABEL_NATIVE_SIX_ROLE_AUTHORITY
            or self.schema != LABEL_NATIVE_SIX_ROLE_BRANCH_DESIGN_SCHEMA
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "unsupported branch authority or schema",
            )
        object.__setattr__(
            self,
            "analytic_support_upper_by_stratum_endpoint_role",
            support,
        )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(
            self,
            "trajectory_alpha_ledger_generation",
            ledger_generation,
        )
        object.__setattr__(self, "support_bound_source_hashes_by_role", sources)
        object.__setattr__(
            self,
            "action_identities",
            action_identities,
        )
        self.assert_integrity()

    @property
    def endpoint_action_hashes(self) -> tuple[str, ...]:
        return tuple(value.packed_action_hash for value in self.action_identities)

    @property
    def endpoint_action_identity_hashes(self) -> tuple[str, ...]:
        return tuple(value.identity_hash for value in self.action_identities)

    @property
    def stratum_count(self) -> int:
        return int(self.analytic_support_upper_by_stratum_endpoint_role.shape[0])

    def _assert_action_source_binding(self) -> None:
        receipt = self.packed_direction_receipt
        registration = self.action_axis_registration
        shell_source = registration.source_shell_ladder
        C_source = self.C_source_execution_registration
        bridge = self.compact_C_bridge_branch_receipt
        expected_subset = _eligible_subset_registration(
            branch=self.branch,
            packed_direction_receipt=receipt,
            action_axis_registration=registration,
            bridge_branch=bridge,
        )
        if (
            receipt.action_axis_registration.registration_hash
            != registration.registration_hash
            or receipt.endpoint_action_hashes != registration.endpoint_action_hashes
            or receipt.endpoint_counts != registration.endpoint_counts
            or receipt.route_axis_hash != shell_source.route_axis_hash
            or receipt.route_count != shell_source.route_count
            or receipt.parent_mask_hash != shell_source.parent_mask_hash
            or receipt.route_axis_hash != C_source.route_axis_hash
            or receipt.route_count != C_source.route_count
            or receipt.parent_mask_hash != C_source.current_mask_hash
            or bridge.route_axis_hash != shell_source.route_axis_hash
            or bridge.route_count != shell_source.route_count
            or bridge.parent_mask_hash != shell_source.parent_mask_hash
            or expected_subset.registration_hash
            != self.eligible_action_subset_registration.registration_hash
            or receipt.endpoint_query_conditioned_on_c_outcomes
            or receipt.endpoint_hard_masks_materialized
            or receipt.candidate_composition_recomputed
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.ACTION_AXIS_MISMATCH,
                "branch provider receipt left its frozen canonical action axis",
            )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch": self.branch.value,
            "packed_direction_receipt_hash": self.packed_direction_receipt.receipt_hash,
            "action_axis_registration_hash": self.action_axis_registration.registration_hash,
            "compact_C_bridge_branch_receipt_hash": (
                self.compact_C_bridge_branch_receipt.receipt_hash
            ),
            "eligible_action_subset_registration": (
                self.eligible_action_subset_registration
            ),
            "C_source_execution_registration_hash": (
                self.C_source_execution_registration.registration_hash
            ),
            "source_shell_ladder_receipt_hash": (
                self.action_axis_registration.source_shell_ladder.receipt_hash
            ),
            "action_identities": self.action_identities,
            "chart_identity_hash": self.chart_identity_hash,
            "proposal_generation": self.proposal_generation,
            "certification_context_population_hash": (
                self.certification_context_population_hash
            ),
            "certification_stream_commitment_hash": (
                self.certification_stream_commitment_hash
            ),
            "joint_six_role_family_registration_hash": (
                self.joint_six_role_family_registration_hash
            ),
            "same_capture_scorer_registration_hash": (
                self.same_capture_scorer_registration_hash
            ),
            "native_fiber_axis_receipt_hash": self.native_fiber_axis_receipt_hash,
            "native_fiber_measure_hash": self.native_fiber_measure_hash,
            "trajectory_alpha_ledger_hash": self.trajectory_alpha_ledger_hash,
            "trajectory_alpha_ledger_generation": (
                self.trajectory_alpha_ledger_generation
            ),
            "analytic_support_upper_by_stratum_endpoint_role": (
                self.analytic_support_upper_by_stratum_endpoint_role
            ),
            "support_bound_source_hashes_by_role": (
                self.support_bound_source_hashes_by_role
            ),
            "issued_before_C_outcomes": self.issued_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
            "D_values_reused_as_C_observations": (
                self.D_values_reused_as_C_observations
            ),
            "endpoint_hard_masks_registered": self.endpoint_hard_masks_registered,
            "candidate_selection_authority": self.candidate_selection_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def design_hash(self) -> str:
        cached = self._design_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.compact_C_bridge_branch_receipt.assert_integrity()
        self.C_source_execution_registration.assert_integrity()
        self._assert_action_source_binding()
        if self.action_identities != _action_identities(
            self.action_axis_registration,
            self.compact_C_bridge_branch_receipt,
            self.eligible_action_subset_registration,
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.ACTION_AXIS_MISMATCH,
                "branch action provenance was reordered or replaced",
            )
        observed = content_hash(self._payload(), prefix=_BRANCH_DESIGN_PREFIX)
        if self._design_hash is not None and observed != self._design_hash:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "branch design was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "branch design hash does not match",
            )
        object.__setattr__(self, "_design_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleCDesign:
    """Pre-outcome family registration for both label-native KKT branches."""

    design_id: str
    branch_designs: tuple[V1LabelNativeSixRoleCBranchDesign, ...] = field(
        repr=False,
        compare=False,
    )
    compact_C_bridge_family_bundle: V1SignedWorkLabelNativeCBridgeFamilyBundle = field(
        repr=False,
        compare=False,
    )
    C_source_execution_registration: (
        V1LabelNativeSixRoleCSourceExecutionRegistration
    ) = field(repr=False, compare=False)
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    certification_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    sealed_draw_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    stratum_weights: tuple[tuple[str, float], ...]
    certification_context_population_hash: str
    certification_stream_commitment_hash: str
    joint_six_role_family_registration_hash: str
    same_capture_scorer_registration_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    trajectory_alpha_ledger_hash: str
    trajectory_alpha_ledger_generation: int
    trajectory_familywise_alpha: float
    trajectory_alpha_committed_before_C_six_role: float
    six_role_family_alpha_allocation: float
    alpha_policy: str = LABEL_NATIVE_SIX_ROLE_ALPHA_POLICY
    sampling_law: str = LABEL_NATIVE_SIX_ROLE_SAMPLING_LAW
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    validation_outcomes_seen: bool = False
    D_values_reused_as_C_observations: bool = False
    signed_F_is_objective_not_default_feasibility_constraint: bool = True
    label_native_actions_are_receipt_bound_not_mask_materialized: bool = True
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = LABEL_NATIVE_SIX_ROLE_AUTHORITY
    schema: str = LABEL_NATIVE_SIX_ROLE_DESIGN_SCHEMA
    cell_count: int = field(init=False)
    tail_count: int = field(init=False)
    tail_family_alpha: float = field(init=False)
    registered_tail_alpha_total: float = field(init=False)
    _design_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "design_id",
            _identity(self.design_id, name="label-native six-role design id"),
        )
        branches = tuple(self.branch_designs)
        if (
            len(branches) != 2
            or any(
                type(value) is not V1LabelNativeSixRoleCBranchDesign
                for value in branches
            )
            or tuple(value.branch for value in branches) != _CANONICAL_BRANCHES
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.BRANCH_MISMATCH,
                "family design requires spectral and diagonal-l1 branches in order",
            )
        if type(self.compact_C_bridge_family_bundle) is not (
            V1SignedWorkLabelNativeCBridgeFamilyBundle
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "family design requires an exact compact C-bridge family bundle",
            )
        bridge_bundle = self.compact_C_bridge_family_bundle
        bridge_bundle.assert_integrity()
        if type(self.C_source_execution_registration) is not (
            V1LabelNativeSixRoleCSourceExecutionRegistration
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "family design requires an exact pre-C source/execution registration",
            )
        source = self.C_source_execution_registration
        source.assert_integrity()
        bridge_branches = (bridge_bundle.spectral, bridge_bundle.diagonal_l1)
        for value, bridge_branch in zip(
            branches,
            bridge_branches,
            strict=True,
        ):
            value.assert_integrity()
            if (
                value.C_source_execution_registration.registration_hash
                != source.registration_hash
                or value.compact_C_bridge_branch_receipt.receipt_hash
                != bridge_branch.receipt_hash
            ):
                _refuse(
                    V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                    "branch source/bridge differs from the family registration",
                )
        contexts = _axis_pairs(
            self.certification_context_ids_by_stratum,
            name="certification C population",
        )
        draws = _axis_pairs(
            self.sealed_draw_ids_by_stratum,
            name="sealed C draw ledger",
        )
        if (
            tuple(name for name, _axis in contexts)
            != tuple(name for name, _axis in draws)
            or tuple(len(axis) for _name, axis in contexts)
            != tuple(len(axis) for _name, axis in draws)
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "C contexts and draw ledgers do not share one population axis",
            )
        expected_coordinate_axis = tuple(
            (stratum, index, context, draws[stratum_index][1][index])
            for stratum_index, (stratum, context_axis) in enumerate(contexts)
            for index, context in enumerate(context_axis)
        )
        if source.coordinate_axis != expected_coordinate_axis:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "pre-C source registration differs from the sealed C coordinate axis",
            )
        raw_weights = tuple(self.stratum_weights)
        if len(raw_weights) != len(contexts):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "stratum weights do not cover the C population",
            )
        weights: list[tuple[str, float]] = []
        for raw, context in zip(raw_weights, contexts, strict=True):
            if type(raw) not in {tuple, list} or len(raw) != 2:
                _refuse(
                    V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                    "stratum weights must be name/value pairs",
                )
            name = _identity(raw[0], name="stratum weight name")
            weight = _finite_nonnegative(raw[1], name="stratum weight")
            if name != context[0] or weight <= 0.0:
                _refuse(
                    V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                    "stratum weight axis or positivity differs from C",
                )
            weights.append((name, weight))
        if math.fsum(value for _name, value in weights) != 1.0:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INTERVAL_INVALID,
                "stratum weights must sum exactly to one",
            )

        for name in (
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "certification_context_population_hash",
            "certification_stream_commitment_hash",
            "joint_six_role_family_registration_hash",
            "same_capture_scorer_registration_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "trajectory_alpha_ledger_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        route_count = _positive_int(self.route_count, name="route count")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        ledger_generation = _nonnegative_int(
            self.trajectory_alpha_ledger_generation,
            name="trajectory alpha-ledger generation",
        )
        if ledger_generation != generation:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "alpha ledger belongs to another proposal generation",
            )
        if (
            source.chart_identity_hash != self.chart_identity_hash
            or source.current_mask_hash != self.current_mask_hash
            or source.route_axis_hash != self.route_axis_hash
            or source.route_count != route_count
            or source.proposal_generation != generation
            or source.certification_context_population_hash
            != self.certification_context_population_hash
            or source.certification_stream_commitment_hash
            != self.certification_stream_commitment_hash
            or source.joint_six_role_family_registration_hash
            != self.joint_six_role_family_registration_hash
            or source.same_capture_scorer_registration_hash
            != self.same_capture_scorer_registration_hash
            or source.native_fiber_axis_receipt_hash
            != self.native_fiber_axis_receipt_hash
            or source.native_fiber_measure_hash != self.native_fiber_measure_hash
            or bridge_bundle.receipt.route_axis_hash != self.route_axis_hash
            or bridge_bundle.receipt.route_count != route_count
            or bridge_bundle.receipt.parent_mask_hash != self.current_mask_hash
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "family metadata differs from its pre-C source/execution registration",
            )
        for branch in branches:
            receipt = branch.packed_direction_receipt
            support = branch.analytic_support_upper_by_stratum_endpoint_role
            if (
                receipt.route_axis_hash != self.route_axis_hash
                or receipt.route_count != route_count
                or receipt.parent_mask_hash != self.current_mask_hash
                or branch.stratum_count != len(contexts)
                or support.shape[1] != len(branch.action_identities)
                or support.shape[2] != len(SIGNED_WORK_SIX_ROLES)
                or branch.chart_identity_hash != self.chart_identity_hash
                or branch.proposal_generation != generation
                or branch.certification_context_population_hash
                != self.certification_context_population_hash
                or branch.certification_stream_commitment_hash
                != self.certification_stream_commitment_hash
                or branch.joint_six_role_family_registration_hash
                != self.joint_six_role_family_registration_hash
                or branch.same_capture_scorer_registration_hash
                != self.same_capture_scorer_registration_hash
                or branch.native_fiber_axis_receipt_hash
                != self.native_fiber_axis_receipt_hash
                or branch.native_fiber_measure_hash
                != self.native_fiber_measure_hash
                or branch.trajectory_alpha_ledger_hash
                != self.trajectory_alpha_ledger_hash
                or branch.trajectory_alpha_ledger_generation != ledger_generation
            ):
                _refuse(
                    V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                    "branch route, parent, stratum, or action axis differs from family",
                )

        familywise = _probability(
            self.trajectory_familywise_alpha,
            name="trajectory familywise alpha",
        )
        committed = _finite_nonnegative(
            self.trajectory_alpha_committed_before_C_six_role,
            name="alpha committed before six-role C",
        )
        allocation = _probability(
            self.six_role_family_alpha_allocation,
            name="six-role family alpha allocation",
        )
        if (
            Fraction.from_float(committed) + Fraction.from_float(allocation)
            > Fraction.from_float(familywise)
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "six-role C allocation overspends the trajectory alpha ledger",
            )
        cells = sum(
            len(contexts)
            * len(branch.endpoint_action_hashes)
            * len(SIGNED_WORK_SIX_ROLES)
            for branch in branches
        )
        tails = 2 * cells
        tail_alpha = _downward_fraction_to_probability(
            Fraction.from_float(allocation) / tails
        )
        registered = math.fsum(tail_alpha for _ in range(tails))
        if registered > allocation:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "rounded tail allocation overspends its registered family",
            )
        required_true = (
            self.issued_before_C_outcomes,
            self.signed_F_is_objective_not_default_feasibility_constraint,
            self.label_native_actions_are_receipt_bound_not_mask_materialized,
        )
        required_false = (
            self.C_outcomes_seen,
            self.validation_outcomes_seen,
            self.D_values_reused_as_C_observations,
            self.candidate_selection_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in required_true + required_false):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "family authority flags must be exact booleans",
            )
        if not all(required_true) or any(required_false):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "family design crossed its pre-outcome evidence boundary",
            )
        if (
            self.alpha_policy != LABEL_NATIVE_SIX_ROLE_ALPHA_POLICY
            or self.sampling_law != LABEL_NATIVE_SIX_ROLE_SAMPLING_LAW
            or self.authority_scope != LABEL_NATIVE_SIX_ROLE_AUTHORITY
            or self.schema != LABEL_NATIVE_SIX_ROLE_DESIGN_SCHEMA
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "unsupported sampling, authority, or schema semantics",
            )
        object.__setattr__(self, "branch_designs", branches)
        object.__setattr__(self, "certification_context_ids_by_stratum", contexts)
        object.__setattr__(self, "sealed_draw_ids_by_stratum", draws)
        object.__setattr__(self, "stratum_weights", tuple(weights))
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(
            self,
            "trajectory_alpha_ledger_generation",
            ledger_generation,
        )
        object.__setattr__(self, "trajectory_familywise_alpha", familywise)
        object.__setattr__(
            self,
            "trajectory_alpha_committed_before_C_six_role",
            committed,
        )
        object.__setattr__(self, "six_role_family_alpha_allocation", allocation)
        object.__setattr__(self, "cell_count", cells)
        object.__setattr__(self, "tail_count", tails)
        object.__setattr__(self, "tail_family_alpha", tail_alpha)
        object.__setattr__(self, "registered_tail_alpha_total", registered)
        self.assert_integrity()

    @property
    def branches(self) -> tuple[V1RelaxedKKTBranch, ...]:
        return tuple(value.branch for value in self.branch_designs)

    @property
    def stratum_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, _axis in self.certification_context_ids_by_stratum
        )

    @property
    def population_sizes(self) -> tuple[int, ...]:
        return tuple(
            len(axis) for _name, axis in self.certification_context_ids_by_stratum
        )

    @property
    def alpha_coordinates(self) -> tuple[tuple[str, str, str, str, str], ...]:
        return tuple(
            (branch.branch.value, stratum, action, role, tail)
            for branch in self.branch_designs
            for stratum in self.stratum_names
            for action in branch.endpoint_action_hashes
            for role in SIGNED_WORK_SIX_ROLES
            for tail in ("lower", "upper")
        )

    def branch_design(
        self,
        branch: V1RelaxedKKTBranch,
    ) -> V1LabelNativeSixRoleCBranchDesign:
        if type(branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "branch lookup requires the exact KKT branch enum",
            )
        try:
            return self.branch_designs[self.branches.index(branch)]
        except ValueError as error:
            raise V1LabelNativeSixRoleCRefusal(
                V1LabelNativeSixRoleCRefusalCode.BRANCH_MISMATCH,
                "branch is not registered for label-native six-role C",
            ) from error

    def look_alpha(
        self,
        *,
        branch: V1RelaxedKKTBranch,
        stratum_index: int,
        sample_size: int,
    ) -> float:
        self.branch_design(branch)
        index = _nonnegative_int(stratum_index, name="stratum index")
        if index >= len(self.population_sizes):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "stratum index left the registered C axis",
            )
        count = _positive_int(sample_size, name="sample size")
        population = self.population_sizes[index]
        if count > population:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "sample size exceeds its registered population",
            )
        if count == population:
            return 0.0
        schedule = Fraction(population, count * (count + 1) * (population - 1))
        return _downward_fraction_to_probability(
            Fraction.from_float(self.tail_family_alpha) * schedule
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_id": self.design_id,
            "branch_design_hashes": tuple(
                value.design_hash for value in self.branch_designs
            ),
            "compact_C_bridge_family_bundle_hash": (
                self.compact_C_bridge_family_bundle.bundle_hash
            ),
            "compact_C_bridge_family_receipt_hash": (
                self.compact_C_bridge_family_bundle.receipt.receipt_hash
            ),
            "C_source_execution_registration_hash": (
                self.C_source_execution_registration.registration_hash
            ),
            "branches": tuple(value.value for value in self.branches),
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "proposal_generation": self.proposal_generation,
            "certification_context_ids_by_stratum": (
                self.certification_context_ids_by_stratum
            ),
            "sealed_draw_ids_by_stratum": self.sealed_draw_ids_by_stratum,
            "stratum_weights": self.stratum_weights,
            "certification_context_population_hash": (
                self.certification_context_population_hash
            ),
            "certification_stream_commitment_hash": (
                self.certification_stream_commitment_hash
            ),
            "joint_six_role_family_registration_hash": (
                self.joint_six_role_family_registration_hash
            ),
            "same_capture_scorer_registration_hash": (
                self.same_capture_scorer_registration_hash
            ),
            "native_fiber_axis_receipt_hash": self.native_fiber_axis_receipt_hash,
            "native_fiber_measure_hash": self.native_fiber_measure_hash,
            "trajectory_alpha_ledger_hash": self.trajectory_alpha_ledger_hash,
            "trajectory_alpha_ledger_generation": (
                self.trajectory_alpha_ledger_generation
            ),
            "trajectory_familywise_alpha": self.trajectory_familywise_alpha,
            "trajectory_alpha_committed_before_C_six_role": (
                self.trajectory_alpha_committed_before_C_six_role
            ),
            "six_role_family_alpha_allocation": (
                self.six_role_family_alpha_allocation
            ),
            "roles": SIGNED_WORK_SIX_ROLES,
            "alpha_coordinates": self.alpha_coordinates,
            "cell_count": self.cell_count,
            "tail_count": self.tail_count,
            "tail_family_alpha": self.tail_family_alpha,
            "registered_tail_alpha_total": self.registered_tail_alpha_total,
            "alpha_policy": self.alpha_policy,
            "sampling_law": self.sampling_law,
            "issued_before_C_outcomes": self.issued_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
            "validation_outcomes_seen": self.validation_outcomes_seen,
            "D_values_reused_as_C_observations": (
                self.D_values_reused_as_C_observations
            ),
            "signed_F_is_objective_not_default_feasibility_constraint": (
                self.signed_F_is_objective_not_default_feasibility_constraint
            ),
            "label_native_actions_are_receipt_bound_not_mask_materialized": (
                self.label_native_actions_are_receipt_bound_not_mask_materialized
            ),
            "candidate_selection_authority": self.candidate_selection_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def design_hash(self) -> str:
        cached = self._design_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.compact_C_bridge_family_bundle.assert_integrity()
        self.C_source_execution_registration.assert_integrity()
        for value in self.branch_designs:
            value.assert_integrity()
        observed = content_hash(self._payload(), prefix=_DESIGN_PREFIX)
        if self._design_hash is not None and observed != self._design_hash:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "label-native six-role design was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "label-native six-role design hash does not match",
            )
        object.__setattr__(self, "_design_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleCRow:
    """One same-capture label-native C row for one frozen KKT branch."""

    design_hash: str
    branch_design_hash: str
    branch: V1RelaxedKKTBranch
    packed_direction_receipt: V1LabelNativePackedDirectionReceipt = field(
        repr=False,
        compare=False,
    )
    action_axis_registration: V1LabelNativeCanonicalActionAxisRegistration
    eligible_action_subset_registration: (
        V1LabelNativeSixRoleCEligibleSubsetRegistration
    )
    endpoint_action_hashes: tuple[str, ...]
    endpoint_action_identity_hashes: tuple[str, ...]
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    stratum_id: str
    sealed_order_index: int
    context_id: str
    draw_id: str
    C_coordinate_proof_hash: str
    C_source_execution_registration_hash: str
    C_coordinate_proof_source_registration_hash: str
    C_execution_registration_hash: str
    certification_context_population_hash: str
    certification_stream_commitment_hash: str
    joint_six_role_family_registration_hash: str
    same_capture_scorer_registration_hash: str
    parent_execution_identity_hash: str
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    raw_logit_stream_identity_hash: str
    same_capture_and_vjp_execution_proof_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    native_fiber_work_projection_receipt_hash: str
    literal_jacobian_projection_receipt_hash: str
    packed_direction_execution_proof_hash: str
    raw_logit_interval_proof_hash: str
    parent_raw_target_logit_center: float
    parent_raw_target_logit_radius: float
    fibered_work_lower_by_endpoint: np.ndarray = field(repr=False, compare=False)
    fibered_work_upper_by_endpoint: np.ndarray = field(repr=False, compare=False)
    global_literal_jacobian_center_by_endpoint: np.ndarray = field(
        repr=False,
        compare=False,
    )
    global_literal_jacobian_radius_by_endpoint: np.ndarray = field(
        repr=False,
        compare=False,
    )
    dosed_raw_target_logit_center_by_endpoint: np.ndarray = field(
        repr=False,
        compare=False,
    )
    dosed_raw_target_logit_radius_by_endpoint: np.ndarray = field(
        repr=False,
        compare=False,
    )
    doses: tuple[float, float] = (0.5, 1.0)
    corrected_evidence_and_literal_J_are_distinct_streams: bool = True
    raw_logit_is_a_distinct_stream: bool = True
    same_capture_and_vjp_certified: bool = True
    native_fiber_signed_cancellation_only: bool = True
    literal_J_is_global_signed_complete_action: bool = True
    secants_execute_complete_packed_actions: bool = True
    fibered_work_bounds_are_outward: bool = True
    literal_jacobian_bounds_are_outward: bool = True
    raw_logit_bounds_are_outward: bool = True
    endpoint_order_unchanged_after_C_outcomes: bool = True
    postselection_occurred: bool = False
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = LABEL_NATIVE_SIX_ROLE_AUTHORITY
    schema: str = LABEL_NATIVE_SIX_ROLE_ROW_SCHEMA
    global_literal_jacobian_lower_by_endpoint: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    global_literal_jacobian_upper_by_endpoint: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    signed_raw_logit_delta_lower_by_endpoint: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    signed_raw_logit_delta_upper_by_endpoint: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    signed_remainder_lower_by_endpoint: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    signed_remainder_upper_by_endpoint: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    observation_lower_by_endpoint_role: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    observation_upper_by_endpoint_role: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    _row_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "branch_design_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "stratum_id",
            "context_id",
            "draw_id",
            "C_coordinate_proof_hash",
            "C_source_execution_registration_hash",
            "C_coordinate_proof_source_registration_hash",
            "C_execution_registration_hash",
            "certification_context_population_hash",
            "certification_stream_commitment_hash",
            "joint_six_role_family_registration_hash",
            "same_capture_scorer_registration_hash",
            "parent_execution_identity_hash",
            "corrected_evidence_stream_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "raw_logit_stream_identity_hash",
            "same_capture_and_vjp_execution_proof_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "native_fiber_work_projection_receipt_hash",
            "literal_jacobian_projection_receipt_hash",
            "packed_direction_execution_proof_hash",
            "raw_logit_interval_proof_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "C row requires the exact KKT branch enum",
            )
        if type(self.packed_direction_receipt) is not V1LabelNativePackedDirectionReceipt:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "C row requires an exact packed-direction receipt",
            )
        if type(self.action_axis_registration) is not (
            V1LabelNativeCanonicalActionAxisRegistration
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "C row requires an exact action-axis registration",
            )
        if type(self.eligible_action_subset_registration) is not (
            V1LabelNativeSixRoleCEligibleSubsetRegistration
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "C row requires an exact pre-C eligible subset registration",
            )
        subset = self.eligible_action_subset_registration
        actions = _identity_axis(
            self.endpoint_action_hashes,
            name="C row endpoint actions",
        )
        action_identities = _identity_axis(
            self.endpoint_action_identity_hashes,
            name="C row endpoint action identities",
        )
        receipt = self.packed_direction_receipt
        if (
            receipt.action_axis_registration.registration_hash
            != self.action_axis_registration.registration_hash
            or subset.branch is not self.branch
            or subset.packed_direction_receipt_hash != receipt.receipt_hash
            or subset.action_axis_registration_hash
            != self.action_axis_registration.registration_hash
            or receipt.endpoint_counts != subset.complete_source_endpoint_counts
            or receipt.endpoint_action_hashes
            != subset.complete_source_action_hashes
            or self.action_axis_registration.endpoint_action_hashes
            != subset.complete_source_action_hashes
            or subset.eligible_action_hashes != actions
            or len(action_identities) != len(actions)
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.ACTION_AXIS_MISMATCH,
                "C row eligible actions left the registered provider subset",
            )
        endpoint_count = len(actions)
        vector_shape = (endpoint_count,)
        matrix_shape = (endpoint_count, 2)
        f_lower = _immutable_f64(
            self.fibered_work_lower_by_endpoint,
            shape=vector_shape,
            name="native-fiber work lower bounds",
            nonnegative=True,
        )
        f_upper = _immutable_f64(
            self.fibered_work_upper_by_endpoint,
            shape=vector_shape,
            name="native-fiber work upper bounds",
            nonnegative=True,
        )
        if bool(np.any(f_lower > f_upper)):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INTERVAL_INVALID,
                "native-fiber work lower bound exceeds its upper bound",
            )
        j_center = _immutable_f64(
            self.global_literal_jacobian_center_by_endpoint,
            shape=vector_shape,
            name="global literal-J centers",
        )
        j_radius = _immutable_f64(
            self.global_literal_jacobian_radius_by_endpoint,
            shape=vector_shape,
            name="global literal-J radii",
            nonnegative=True,
        )
        child_center = _immutable_f64(
            self.dosed_raw_target_logit_center_by_endpoint,
            shape=matrix_shape,
            name="dosed raw-logit centers",
        )
        child_radius = _immutable_f64(
            self.dosed_raw_target_logit_radius_by_endpoint,
            shape=matrix_shape,
            name="dosed raw-logit radii",
            nonnegative=True,
        )
        parent_center = _finite(
            self.parent_raw_target_logit_center,
            name="parent raw target-logit center",
        )
        parent_radius = _finite_nonnegative(
            self.parent_raw_target_logit_radius,
            name="parent raw target-logit radius",
        )
        route_count = _positive_int(self.route_count, name="route count")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        position = _nonnegative_int(
            self.sealed_order_index,
            name="sealed C order index",
        )
        doses = tuple(_finite_nonnegative(value, name="secant dose") for value in self.doses)
        if doses != (0.5, 1.0):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "C row requires canonical doses (0.5, 1.0)",
            )
        stream_ids = (
            self.corrected_evidence_stream_identity_hash,
            self.literal_jacobian_stream_identity_hash,
            self.raw_logit_stream_identity_hash,
        )
        required_true = (
            self.corrected_evidence_and_literal_J_are_distinct_streams,
            self.raw_logit_is_a_distinct_stream,
            self.same_capture_and_vjp_certified,
            self.native_fiber_signed_cancellation_only,
            self.literal_J_is_global_signed_complete_action,
            self.secants_execute_complete_packed_actions,
            self.fibered_work_bounds_are_outward,
            self.literal_jacobian_bounds_are_outward,
            self.raw_logit_bounds_are_outward,
            self.endpoint_order_unchanged_after_C_outcomes,
        )
        required_false = (
            self.postselection_occurred,
            self.candidate_selection_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in required_true + required_false):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "C row authority/capture flags must be exact booleans",
            )
        if len(set(stream_ids)) != 3 or not all(required_true):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.CAPTURE_ALIAS,
                "corrected, literal-J, and raw-logit streams must stay distinct",
            )
        if any(required_false):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "C row crossed its evidence-only authority boundary",
            )
        if (
            receipt.route_axis_hash != self.route_axis_hash
            or receipt.route_count != route_count
            or receipt.parent_mask_hash != self.current_mask_hash
            or self.authority_scope != LABEL_NATIVE_SIX_ROLE_AUTHORITY
            or self.schema != LABEL_NATIVE_SIX_ROLE_ROW_SCHEMA
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "C row route/parent/authority identity is stale",
            )

        j_lower = np.empty(vector_shape, dtype=np.float64)
        j_upper = np.empty(vector_shape, dtype=np.float64)
        delta_lower = np.empty(matrix_shape, dtype=np.float64)
        delta_upper = np.empty(matrix_shape, dtype=np.float64)
        remainder_lower = np.empty(matrix_shape, dtype=np.float64)
        remainder_upper = np.empty(matrix_shape, dtype=np.float64)
        observations_lower = np.empty(
            (endpoint_count, len(SIGNED_WORK_SIX_ROLES)),
            dtype=np.float64,
        )
        observations_upper = np.empty_like(observations_lower)
        observations_lower[:, 0] = f_lower
        observations_upper[:, 0] = f_upper

        # This is the same exact-over-binary64 Decimal derivation used by
        # V1FiberedDirectionalRecord.  Only final conversion rounds, outward.
        from decimal import localcontext

        with localcontext() as decimal_context:
            decimal_context.prec = 2500
            p_center = _decimal(parent_center)
            p_radius = _decimal(parent_radius)
            p_lower = p_center - p_radius
            p_upper = p_center + p_radius
            for endpoint in range(endpoint_count):
                jc = _decimal(float(j_center[endpoint]))
                jr = _decimal(float(j_radius[endpoint]))
                jl = jc - jr
                ju = jc + jr
                j_lower[endpoint] = _outward_float(
                    jl,
                    upward=False,
                    name="literal-J lower bound",
                )
                j_upper[endpoint] = _outward_float(
                    ju,
                    upward=True,
                    name="literal-J upper bound",
                )
                abs_j_lower, abs_j_upper = _absolute_interval(jl, ju)
                observations_lower[endpoint, 1] = _outward_float(
                    abs_j_lower * abs_j_lower,
                    upward=False,
                    name="Q lower bound",
                )
                observations_upper[endpoint, 1] = _outward_float(
                    abs_j_upper * abs_j_upper,
                    upward=True,
                    name="Q upper bound",
                )
                for dose_index, dose in enumerate(doses):
                    cc = _decimal(float(child_center[endpoint, dose_index]))
                    cr = _decimal(float(child_radius[endpoint, dose_index]))
                    raw_lower = (cc - cr) - p_upper
                    raw_upper = (cc + cr) - p_lower
                    eta = _decimal(dose)
                    tangent_lower = eta * jl
                    tangent_upper = eta * ju
                    rem_lower = raw_lower - tangent_upper
                    rem_upper = raw_upper - tangent_lower
                    delta_lower[endpoint, dose_index] = _outward_float(
                        raw_lower,
                        upward=False,
                        name="raw-logit delta lower bound",
                    )
                    delta_upper[endpoint, dose_index] = _outward_float(
                        raw_upper,
                        upward=True,
                        name="raw-logit delta upper bound",
                    )
                    remainder_lower[endpoint, dose_index] = _outward_float(
                        rem_lower,
                        upward=False,
                        name="signed remainder lower bound",
                    )
                    remainder_upper[endpoint, dose_index] = _outward_float(
                        rem_upper,
                        upward=True,
                        name="signed remainder upper bound",
                    )
                    abs_r_lower, abs_r_upper = _absolute_interval(
                        rem_lower,
                        rem_upper,
                    )
                    abs_d_lower, abs_d_upper = _absolute_interval(
                        raw_lower,
                        raw_upper,
                    )
                    r_role = 2 + dose_index
                    d_role = 4 + dose_index
                    observations_lower[endpoint, r_role] = _outward_float(
                        abs_r_lower,
                        upward=False,
                        name="absolute remainder lower bound",
                    )
                    observations_upper[endpoint, r_role] = _outward_float(
                        abs_r_upper,
                        upward=True,
                        name="absolute remainder upper bound",
                    )
                    observations_lower[endpoint, d_role] = _outward_float(
                        abs_d_lower,
                        upward=False,
                        name="absolute displacement lower bound",
                    )
                    observations_upper[endpoint, d_role] = _outward_float(
                        abs_d_upper,
                        upward=True,
                        name="absolute displacement upper bound",
                    )

        object.__setattr__(self, "endpoint_action_hashes", actions)
        object.__setattr__(
            self,
            "endpoint_action_identity_hashes",
            action_identities,
        )
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "sealed_order_index", position)
        object.__setattr__(self, "parent_raw_target_logit_center", parent_center)
        object.__setattr__(self, "parent_raw_target_logit_radius", parent_radius)
        object.__setattr__(self, "fibered_work_lower_by_endpoint", f_lower)
        object.__setattr__(self, "fibered_work_upper_by_endpoint", f_upper)
        object.__setattr__(
            self,
            "global_literal_jacobian_center_by_endpoint",
            j_center,
        )
        object.__setattr__(
            self,
            "global_literal_jacobian_radius_by_endpoint",
            j_radius,
        )
        object.__setattr__(
            self,
            "dosed_raw_target_logit_center_by_endpoint",
            child_center,
        )
        object.__setattr__(
            self,
            "dosed_raw_target_logit_radius_by_endpoint",
            child_radius,
        )
        object.__setattr__(self, "doses", doses)
        for name, value, nonnegative in (
            ("global_literal_jacobian_lower_by_endpoint", j_lower, False),
            ("global_literal_jacobian_upper_by_endpoint", j_upper, False),
            ("signed_raw_logit_delta_lower_by_endpoint", delta_lower, False),
            ("signed_raw_logit_delta_upper_by_endpoint", delta_upper, False),
            ("signed_remainder_lower_by_endpoint", remainder_lower, False),
            ("signed_remainder_upper_by_endpoint", remainder_upper, False),
            ("observation_lower_by_endpoint_role", observations_lower, True),
            ("observation_upper_by_endpoint_role", observations_upper, True),
        ):
            object.__setattr__(
                self,
                name,
                _immutable_f64(
                    value,
                    shape=value.shape,
                    name=name,
                    nonnegative=nonnegative,
                ),
            )
        if bool(
            np.any(
                self.observation_lower_by_endpoint_role
                > self.observation_upper_by_endpoint_role
            )
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INTERVAL_INVALID,
                "derived six-role observation interval is reversed",
            )
        self.assert_integrity()

    @property
    def source_action_receipt_hashes(self) -> tuple[str, ...]:
        return self.action_axis_registration.source_action_receipt_hashes

    def _assert_source_binding(self) -> None:
        receipt = self.packed_direction_receipt
        registration = self.action_axis_registration
        subset = self.eligible_action_subset_registration
        if (
            receipt.action_axis_registration.registration_hash
            != registration.registration_hash
            or subset.packed_direction_receipt_hash != receipt.receipt_hash
            or subset.action_axis_registration_hash != registration.registration_hash
            or receipt.endpoint_counts != subset.complete_source_endpoint_counts
            or receipt.endpoint_action_hashes
            != subset.complete_source_action_hashes
            or registration.endpoint_action_hashes
            != subset.complete_source_action_hashes
            or subset.eligible_action_hashes != self.endpoint_action_hashes
            or receipt.route_axis_hash != self.route_axis_hash
            or receipt.route_count != self.route_count
            or receipt.parent_mask_hash != self.current_mask_hash
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.ACTION_AXIS_MISMATCH,
                "C row no longer replays from its provider/action receipts",
            )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "design_hash",
                "branch_design_hash",
                "branch",
                "endpoint_action_hashes",
                "endpoint_action_identity_hashes",
                "chart_identity_hash",
                "current_mask_hash",
                "route_axis_hash",
                "route_count",
                "proposal_generation",
                "stratum_id",
                "sealed_order_index",
                "context_id",
                "draw_id",
                "C_coordinate_proof_hash",
                "C_source_execution_registration_hash",
                "C_coordinate_proof_source_registration_hash",
                "C_execution_registration_hash",
                "certification_context_population_hash",
                "certification_stream_commitment_hash",
                "joint_six_role_family_registration_hash",
                "same_capture_scorer_registration_hash",
                "parent_execution_identity_hash",
                "corrected_evidence_stream_identity_hash",
                "literal_jacobian_stream_identity_hash",
                "raw_logit_stream_identity_hash",
                "same_capture_and_vjp_execution_proof_hash",
                "native_fiber_axis_receipt_hash",
                "native_fiber_measure_hash",
                "native_fiber_work_projection_receipt_hash",
                "literal_jacobian_projection_receipt_hash",
                "packed_direction_execution_proof_hash",
                "raw_logit_interval_proof_hash",
                "parent_raw_target_logit_center",
                "parent_raw_target_logit_radius",
                "fibered_work_lower_by_endpoint",
                "fibered_work_upper_by_endpoint",
                "global_literal_jacobian_center_by_endpoint",
                "global_literal_jacobian_radius_by_endpoint",
                "dosed_raw_target_logit_center_by_endpoint",
                "dosed_raw_target_logit_radius_by_endpoint",
                "global_literal_jacobian_lower_by_endpoint",
                "global_literal_jacobian_upper_by_endpoint",
                "signed_raw_logit_delta_lower_by_endpoint",
                "signed_raw_logit_delta_upper_by_endpoint",
                "signed_remainder_lower_by_endpoint",
                "signed_remainder_upper_by_endpoint",
                "observation_lower_by_endpoint_role",
                "observation_upper_by_endpoint_role",
                "doses",
                "corrected_evidence_and_literal_J_are_distinct_streams",
                "raw_logit_is_a_distinct_stream",
                "same_capture_and_vjp_certified",
                "native_fiber_signed_cancellation_only",
                "literal_J_is_global_signed_complete_action",
                "secants_execute_complete_packed_actions",
                "fibered_work_bounds_are_outward",
                "literal_jacobian_bounds_are_outward",
                "raw_logit_bounds_are_outward",
                "endpoint_order_unchanged_after_C_outcomes",
                "postselection_occurred",
                "candidate_selection_authority",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
                "authority_scope",
            )
        } | {
            "packed_direction_receipt_hash": (
                self.packed_direction_receipt.receipt_hash
            ),
            "action_axis_registration_hash": (
                self.action_axis_registration.registration_hash
            ),
            "eligible_action_subset_registration": (
                self.eligible_action_subset_registration
            ),
            "source_shell_ladder_receipt_hash": (
                self.action_axis_registration.source_shell_ladder.receipt_hash
            ),
            "source_action_receipt_hashes": self.source_action_receipt_hashes,
        }

    @property
    def row_hash(self) -> str:
        cached = self._row_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._assert_source_binding()
        observed = content_hash(self._payload(), prefix=_ROW_PREFIX)
        if self._row_hash is not None and observed != self._row_hash:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "label-native six-role C row was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "label-native six-role C row hash does not match",
            )
        object.__setattr__(self, "_row_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleCellSummary:
    branch: V1RelaxedKKTBranch
    stratum_id: str
    action_identity: V1LabelNativeSixRoleActionIdentity
    role: str
    dose: float | None
    population_size: int
    sample_size: int
    lower: float
    upper: float
    value_upper_bound: float
    current_lower_look_alpha: float
    current_upper_look_alpha: float
    full_census: bool
    method: str
    branch_observation_chain_hash: str
    schema: str = LABEL_NATIVE_SIX_ROLE_CELL_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "cell branch must use the exact KKT enum",
            )
        object.__setattr__(
            self,
            "stratum_id",
            _identity(self.stratum_id, name="cell stratum"),
        )
        if type(self.action_identity) is not V1LabelNativeSixRoleActionIdentity:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "cell requires an exact source action identity",
            )
        if (
            self.role not in SIGNED_WORK_SIX_ROLES
            or self.dose != SIGNED_WORK_SIX_ROLE_DOSES[self.role]
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "cell role/dose is not canonical",
            )
        population = _positive_int(self.population_size, name="cell population")
        count = _nonnegative_int(self.sample_size, name="cell sample size")
        if count > population:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "cell sample exceeds its finite population",
            )
        lower = _finite_nonnegative(self.lower, name="cell lower bound")
        upper = _finite_nonnegative(self.upper, name="cell upper bound")
        support = _finite_nonnegative(
            self.value_upper_bound,
            name="cell analytic support",
        )
        if not lower <= upper <= support:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.SUPPORT_OVERFLOW,
                "cell interval exceeds its pre-outcome support",
            )
        lower_alpha = _finite_nonnegative(
            self.current_lower_look_alpha,
            name="cell lower look alpha",
        )
        upper_alpha = _finite_nonnegative(
            self.current_upper_look_alpha,
            name="cell upper look alpha",
        )
        if type(self.full_census) is not bool or self.full_census != (
            count == population
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "cell census flag differs from its sample count",
            )
        expected_method = LABEL_NATIVE_SIX_ROLE_METHOD
        if count == 0:
            expected_method = "unobserved_registered_support_v1"
        elif count == population:
            expected_method = LABEL_NATIVE_SIX_ROLE_CENSUS_METHOD
        if self.method != expected_method:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "cell method differs from its sampling state",
            )
        object.__setattr__(
            self,
            "branch_observation_chain_hash",
            _identity(
                self.branch_observation_chain_hash,
                name="cell branch observation chain",
            ),
        )
        if self.schema != LABEL_NATIVE_SIX_ROLE_CELL_SCHEMA:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "unsupported cell schema",
            )
        object.__setattr__(self, "population_size", population)
        object.__setattr__(self, "sample_size", count)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "value_upper_bound", support)
        object.__setattr__(self, "current_lower_look_alpha", lower_alpha)
        object.__setattr__(self, "current_upper_look_alpha", upper_alpha)

    @property
    def summary_hash(self) -> str:
        return content_hash(self, prefix=_CELL_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleEndpointSummary:
    branch: V1RelaxedKKTBranch
    action_identity: V1LabelNativeSixRoleActionIdentity
    f_native_lower: float
    f_native_upper: float
    q_j_lower: float
    q_j_upper: float
    r_half_lower: float
    r_half_upper: float
    r_one_lower: float
    r_one_upper: float
    d_half_lower: float
    d_half_upper: float
    d_one_lower: float
    d_one_upper: float
    chart_half_burden_lower: float
    chart_half_burden_upper: float
    chart_one_burden_lower: float
    chart_one_burden_upper: float
    signed_F_is_objective_evidence: bool = True
    Q_and_R_are_chart_trust_evidence: bool = True
    D_is_finite_response_admission_evidence: bool = True
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = LABEL_NATIVE_SIX_ROLE_ENDPOINT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "endpoint branch must use the exact KKT enum",
            )
        if type(self.action_identity) is not V1LabelNativeSixRoleActionIdentity:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "endpoint requires an exact source action identity",
            )
        for lower_name, upper_name in (
            ("f_native_lower", "f_native_upper"),
            ("q_j_lower", "q_j_upper"),
            ("r_half_lower", "r_half_upper"),
            ("r_one_lower", "r_one_upper"),
            ("d_half_lower", "d_half_upper"),
            ("d_one_lower", "d_one_upper"),
            ("chart_half_burden_lower", "chart_half_burden_upper"),
            ("chart_one_burden_lower", "chart_one_burden_upper"),
        ):
            lower = _finite_nonnegative(getattr(self, lower_name), name=lower_name)
            upper = _finite_nonnegative(getattr(self, upper_name), name=upper_name)
            if lower > upper:
                _refuse(
                    V1LabelNativeSixRoleCRefusalCode.INTERVAL_INVALID,
                    f"endpoint interval {lower_name}/{upper_name} is reversed",
                )
            object.__setattr__(self, lower_name, lower)
            object.__setattr__(self, upper_name, upper)
        required_true = (
            self.signed_F_is_objective_evidence,
            self.Q_and_R_are_chart_trust_evidence,
            self.D_is_finite_response_admission_evidence,
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
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "endpoint summary crossed its evidence-only role split",
            )
        if self.schema != LABEL_NATIVE_SIX_ROLE_ENDPOINT_SCHEMA:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "unsupported endpoint summary schema",
            )

    @property
    def summary_hash(self) -> str:
        return content_hash(self, prefix=_ENDPOINT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleBranchProgress:
    branch: V1RelaxedKKTBranch
    branch_design_hash: str
    packed_direction_receipt_hash: str
    action_axis_registration_hash: str
    source_shell_ladder_receipt_hash: str
    compact_C_bridge_branch_receipt_hash: str
    eligible_action_subset_registration_hash: str
    eligible_source_endpoint_labels: tuple[int, ...]
    observed_counts_by_stratum: tuple[int, ...]
    next_expected_context_by_stratum: tuple[str | None, ...]
    next_expected_draw_by_stratum: tuple[str | None, ...]
    cell_summaries: tuple[V1LabelNativeSixRoleCellSummary, ...]
    endpoint_summaries: tuple[V1LabelNativeSixRoleEndpointSummary, ...]
    branch_observation_chain_hash: str
    schema: str = LABEL_NATIVE_SIX_ROLE_BRANCH_PROGRESS_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "branch progress requires the exact KKT enum",
            )
        for name in (
            "branch_design_hash",
            "packed_direction_receipt_hash",
            "action_axis_registration_hash",
            "source_shell_ladder_receipt_hash",
            "compact_C_bridge_branch_receipt_hash",
            "eligible_action_subset_registration_hash",
            "branch_observation_chain_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        counts = tuple(
            _nonnegative_int(value, name="observed C count")
            for value in self.observed_counts_by_stratum
        )
        if not counts:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "branch progress must cover at least one stratum",
            )
        labels = tuple(
            _positive_int(value, name="progress eligible source endpoint label")
            for value in self.eligible_source_endpoint_labels
        )
        if not labels or tuple(sorted(set(labels))) != labels:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.ELIGIBLE_SUBSET_MISMATCH,
                "progress eligible source labels are empty or reordered",
            )
        for value in (
            self.next_expected_context_by_stratum
            + self.next_expected_draw_by_stratum
        ):
            if value is not None:
                _identity(value, name="next expected C coordinate")
        if (
            len(self.next_expected_context_by_stratum) != len(counts)
            or len(self.next_expected_draw_by_stratum) != len(counts)
            or any(
                type(value) is not V1LabelNativeSixRoleCellSummary
                for value in self.cell_summaries
            )
            or any(
                type(value) is not V1LabelNativeSixRoleEndpointSummary
                for value in self.endpoint_summaries
            )
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "branch progress axes or summary types are invalid",
            )
        if self.schema != LABEL_NATIVE_SIX_ROLE_BRANCH_PROGRESS_SCHEMA:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "unsupported branch progress schema",
            )
        object.__setattr__(self, "observed_counts_by_stratum", counts)
        object.__setattr__(self, "eligible_source_endpoint_labels", labels)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_BRANCH_PROGRESS_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleProgressReceipt:
    design_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    certification_context_population_hash: str
    certification_stream_commitment_hash: str
    joint_six_role_family_registration_hash: str
    same_capture_scorer_registration_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    trajectory_alpha_ledger_hash: str
    branch_receipts: tuple[V1LabelNativeSixRoleBranchProgress, ...]
    shared_record_count: int
    transaction_chain_hash: str
    sampling_status: LabelNativeSixRoleSamplingStatus
    tail_count: int
    registered_tail_alpha_total: float
    D_values_reused_as_C_observations: bool = False
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = LABEL_NATIVE_SIX_ROLE_AUTHORITY
    schema: str = LABEL_NATIVE_SIX_ROLE_PROGRESS_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "certification_context_population_hash",
            "certification_stream_commitment_hash",
            "joint_six_role_family_registration_hash",
            "same_capture_scorer_registration_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "trajectory_alpha_ledger_hash",
            "transaction_chain_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="progress route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(
                self.proposal_generation,
                name="progress proposal generation",
            ),
        )
        branches = tuple(self.branch_receipts)
        if (
            len(branches) != 2
            or any(type(value) is not V1LabelNativeSixRoleBranchProgress for value in branches)
            or tuple(value.branch for value in branches) != _CANONICAL_BRANCHES
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.BRANCH_MISMATCH,
                "progress must cover both canonical branches in order",
            )
        shared = _nonnegative_int(
            self.shared_record_count,
            name="shared C record count",
        )
        tails = _positive_int(self.tail_count, name="registered tail count")
        registered = _finite_nonnegative(
            self.registered_tail_alpha_total,
            name="registered tail alpha total",
        )
        if registered <= 0.0:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INTERVAL_INVALID,
                "registered tail alpha must be positive",
            )
        if self.sampling_status not in {
            "awaiting_C",
            "partial_C_evidence",
            "full_C_census",
        }:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "unknown label-native C sampling status",
            )
        authority = (
            self.D_values_reused_as_C_observations,
            self.candidate_selection_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in authority) or any(authority):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "progress receipt crossed its evidence-only boundary",
            )
        if (
            self.authority_scope != LABEL_NATIVE_SIX_ROLE_AUTHORITY
            or self.schema != LABEL_NATIVE_SIX_ROLE_PROGRESS_SCHEMA
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "unsupported progress authority or schema",
            )
        object.__setattr__(self, "branch_receipts", branches)
        object.__setattr__(self, "shared_record_count", shared)
        object.__setattr__(self, "tail_count", tails)
        object.__setattr__(self, "registered_tail_alpha_total", registered)

    def assert_replays(self, design: V1LabelNativeSixRoleCDesign) -> str:
        if type(design) is not V1LabelNativeSixRoleCDesign:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "progress replay requires the exact family design",
            )
        design.assert_integrity()
        expected = {
            "design_hash": design.design_hash,
            "chart_identity_hash": design.chart_identity_hash,
            "current_mask_hash": design.current_mask_hash,
            "route_axis_hash": design.route_axis_hash,
            "route_count": design.route_count,
            "proposal_generation": design.proposal_generation,
            "certification_context_population_hash": (
                design.certification_context_population_hash
            ),
            "certification_stream_commitment_hash": (
                design.certification_stream_commitment_hash
            ),
            "joint_six_role_family_registration_hash": (
                design.joint_six_role_family_registration_hash
            ),
            "same_capture_scorer_registration_hash": (
                design.same_capture_scorer_registration_hash
            ),
            "native_fiber_axis_receipt_hash": design.native_fiber_axis_receipt_hash,
            "native_fiber_measure_hash": design.native_fiber_measure_hash,
            "trajectory_alpha_ledger_hash": design.trajectory_alpha_ledger_hash,
            "tail_count": design.tail_count,
            "registered_tail_alpha_total": design.registered_tail_alpha_total,
        }
        stale = tuple(
            name for name, value in expected.items() if getattr(self, name) != value
        )
        if stale:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "progress receipt differs from design at: " + ", ".join(stale),
            )
        observed_count_axes: list[tuple[int, ...]] = []
        for branch_receipt, branch_design in zip(
            self.branch_receipts,
            design.branch_designs,
            strict=True,
        ):
            counts = branch_receipt.observed_counts_by_stratum
            observed_count_axes.append(counts)
            expected_next_context = tuple(
                None if count == len(design.certification_context_ids_by_stratum[index][1])
                else design.certification_context_ids_by_stratum[index][1][count]
                for index, count in enumerate(counts)
            )
            expected_next_draw = tuple(
                None if count == len(design.sealed_draw_ids_by_stratum[index][1])
                else design.sealed_draw_ids_by_stratum[index][1][count]
                for index, count in enumerate(counts)
            )
            expected_cell_axis = tuple(
                (
                    branch_design.branch,
                    stratum,
                    action.identity_hash,
                    role,
                    SIGNED_WORK_SIX_ROLE_DOSES[role],
                    design.population_sizes[stratum_index],
                    counts[stratum_index],
                    float(
                        branch_design.analytic_support_upper_by_stratum_endpoint_role[
                            stratum_index,
                            endpoint_index,
                            role_index,
                        ]
                    ),
                    branch_receipt.branch_observation_chain_hash,
                )
                for stratum_index, stratum in enumerate(design.stratum_names)
                for endpoint_index, action in enumerate(
                    branch_design.action_identities
                )
                for role_index, role in enumerate(SIGNED_WORK_SIX_ROLES)
            )
            observed_cell_axis = tuple(
                (
                    cell.branch,
                    cell.stratum_id,
                    cell.action_identity.identity_hash,
                    cell.role,
                    cell.dose,
                    cell.population_size,
                    cell.sample_size,
                    cell.value_upper_bound,
                    cell.branch_observation_chain_hash,
                )
                for cell in branch_receipt.cell_summaries
            )
            if (
                branch_receipt.branch_design_hash != branch_design.design_hash
                or branch_receipt.packed_direction_receipt_hash
                != branch_design.packed_direction_receipt.receipt_hash
                or branch_receipt.action_axis_registration_hash
                != branch_design.action_axis_registration.registration_hash
                or branch_receipt.source_shell_ladder_receipt_hash
                != branch_design.action_axis_registration.source_shell_ladder.receipt_hash
                or branch_receipt.compact_C_bridge_branch_receipt_hash
                != branch_design.compact_C_bridge_branch_receipt.receipt_hash
                or branch_receipt.eligible_action_subset_registration_hash
                != branch_design.eligible_action_subset_registration.registration_hash
                or branch_receipt.eligible_source_endpoint_labels
                != branch_design.eligible_action_subset_registration.eligible_source_endpoint_labels
                or tuple(
                    value.action_identity.identity_hash
                    for value in branch_receipt.endpoint_summaries
                )
                != branch_design.endpoint_action_identity_hashes
                or len(counts) != len(design.stratum_names)
                or any(
                    count > population
                    for count, population in zip(
                        counts,
                        design.population_sizes,
                        strict=True,
                    )
                )
                or branch_receipt.next_expected_context_by_stratum
                != expected_next_context
                or branch_receipt.next_expected_draw_by_stratum
                != expected_next_draw
                or observed_cell_axis != expected_cell_axis
            ):
                _refuse(
                    V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                    "progress branch no longer replays from its action source",
                )
            weights = tuple(value for _name, value in design.stratum_weights)
            for endpoint_index, endpoint in enumerate(
                branch_receipt.endpoint_summaries
            ):
                lowers: dict[str, float] = {}
                uppers: dict[str, float] = {}
                for role_index, role in enumerate(SIGNED_WORK_SIX_ROLES):
                    role_cells = tuple(
                        branch_receipt.cell_summaries[
                            (
                                stratum_index
                                * len(branch_design.action_identities)
                                + endpoint_index
                            )
                            * len(SIGNED_WORK_SIX_ROLES)
                            + role_index
                        ]
                        for stratum_index in range(len(design.stratum_names))
                    )
                    lowers[role] = _downward_weighted_sum(
                        weights,
                        tuple(value.lower for value in role_cells),
                    )
                    uppers[role] = _outward_weighted_sum(
                        weights,
                        tuple(value.upper for value in role_cells),
                    )
                expected_endpoint_values = (
                    lowers[F_NATIVE_ROLE],
                    uppers[F_NATIVE_ROLE],
                    lowers[Q_J_ROLE],
                    uppers[Q_J_ROLE],
                    lowers[R_HALF_ROLE],
                    uppers[R_HALF_ROLE],
                    lowers[R_ONE_ROLE],
                    uppers[R_ONE_ROLE],
                    lowers[D_HALF_ROLE],
                    uppers[D_HALF_ROLE],
                    lowers[D_ONE_ROLE],
                    uppers[D_ONE_ROLE],
                    _response_burden_lower(
                        eta=0.5,
                        q_lower=lowers[Q_J_ROLE],
                        r_lower=lowers[R_HALF_ROLE],
                    ),
                    _response_burden(
                        eta=0.5,
                        q_upper=uppers[Q_J_ROLE],
                        r_upper=uppers[R_HALF_ROLE],
                    ),
                    _response_burden_lower(
                        eta=1.0,
                        q_lower=lowers[Q_J_ROLE],
                        r_lower=lowers[R_ONE_ROLE],
                    ),
                    _response_burden(
                        eta=1.0,
                        q_upper=uppers[Q_J_ROLE],
                        r_upper=uppers[R_ONE_ROLE],
                    ),
                )
                observed_endpoint_values = (
                    endpoint.f_native_lower,
                    endpoint.f_native_upper,
                    endpoint.q_j_lower,
                    endpoint.q_j_upper,
                    endpoint.r_half_lower,
                    endpoint.r_half_upper,
                    endpoint.r_one_lower,
                    endpoint.r_one_upper,
                    endpoint.d_half_lower,
                    endpoint.d_half_upper,
                    endpoint.d_one_lower,
                    endpoint.d_one_upper,
                    endpoint.chart_half_burden_lower,
                    endpoint.chart_half_burden_upper,
                    endpoint.chart_one_burden_lower,
                    endpoint.chart_one_burden_upper,
                )
                if observed_endpoint_values != expected_endpoint_values:
                    _refuse(
                        V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                        "endpoint summary does not replay from its registered cells",
                    )
        if (
            any(axis != observed_count_axes[0] for axis in observed_count_axes[1:])
            or self.shared_record_count != sum(observed_count_axes[0])
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.BRANCH_MISMATCH,
                "progress violates atomic cross-branch count equality",
            )
        total_population = sum(design.population_sizes)
        expected_status: LabelNativeSixRoleSamplingStatus
        if self.shared_record_count == 0:
            expected_status = "awaiting_C"
        elif self.shared_record_count == total_population:
            expected_status = "full_C_census"
        else:
            expected_status = "partial_C_evidence"
        if self.sampling_status != expected_status:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "progress sampling status differs from its exact counts",
            )
        return self.receipt_hash

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PROGRESS_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleAccumulatorSnapshot:
    design_hash: str
    branches: tuple[V1RelaxedKKTBranch, ...]
    counts_by_branch_and_stratum: tuple[tuple[int, ...], ...]
    lower_scaled_sums_by_branch: tuple[tuple[int, ...], ...]
    upper_scaled_sums_by_branch: tuple[tuple[int, ...], ...]
    branch_observation_chain_hashes: tuple[str, ...]
    shared_record_count: int
    transaction_chain_hash: str
    schema: str = LABEL_NATIVE_SIX_ROLE_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "design_hash",
            _identity(self.design_hash, name="snapshot design hash"),
        )
        branches = tuple(self.branches)
        if branches != _CANONICAL_BRANCHES:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.BRANCH_MISMATCH,
                "snapshot does not cover both canonical branches",
            )
        if not (
            len(branches)
            == len(self.counts_by_branch_and_stratum)
            == len(self.lower_scaled_sums_by_branch)
            == len(self.upper_scaled_sums_by_branch)
            == len(self.branch_observation_chain_hashes)
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "snapshot branch axes are misaligned",
            )
        counts = tuple(
            tuple(
                _nonnegative_int(value, name="snapshot observed count")
                for value in row
            )
            for row in self.counts_by_branch_and_stratum
        )
        if not counts[0] or any(row != counts[0] for row in counts[1:]):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "snapshot violates atomic cross-branch count equality",
            )
        lower_rows = tuple(
            tuple(int(value) for value in row)
            for row in self.lower_scaled_sums_by_branch
        )
        upper_rows = tuple(
            tuple(int(value) for value in row)
            for row in self.upper_scaled_sums_by_branch
        )
        for row in lower_rows + upper_rows:
            if any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 0
                for value in row
            ):
                _refuse(
                    V1LabelNativeSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                    "snapshot sufficient statistics must be nonnegative integers",
                )
        if any(
            len(lower) != len(upper)
            or any(left > right for left, right in zip(lower, upper, strict=True))
            for lower, upper in zip(lower_rows, upper_rows, strict=True)
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "snapshot lower sufficient statistic exceeds its upper statistic",
            )
        chains = tuple(
            _identity(value, name="snapshot branch chain")
            for value in self.branch_observation_chain_hashes
        )
        shared = _nonnegative_int(
            self.shared_record_count,
            name="snapshot shared record count",
        )
        if shared != sum(counts[0]):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "snapshot shared count differs from its atomic branch counts",
            )
        object.__setattr__(
            self,
            "transaction_chain_hash",
            _identity(
                self.transaction_chain_hash,
                name="snapshot transaction chain",
            ),
        )
        if self.schema != LABEL_NATIVE_SIX_ROLE_SNAPSHOT_SCHEMA:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "unsupported snapshot schema",
            )
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "counts_by_branch_and_stratum", counts)
        object.__setattr__(self, "lower_scaled_sums_by_branch", lower_rows)
        object.__setattr__(self, "upper_scaled_sums_by_branch", upper_rows)
        object.__setattr__(self, "branch_observation_chain_hashes", chains)
        object.__setattr__(self, "shared_record_count", shared)

    @property
    def snapshot_hash(self) -> str:
        return content_hash(self, prefix=_SNAPSHOT_PREFIX)


class V1LabelNativeSixRoleCAccumulator:
    """Atomic sufficient-statistic owner for both label-native branches."""

    def __init__(self, design: V1LabelNativeSixRoleCDesign) -> None:
        if type(design) is not V1LabelNativeSixRoleCDesign:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "accumulator requires the exact label-native C design",
            )
        design_hash = design.assert_integrity()
        self.design = design
        self._counts = [
            np.zeros(len(design.stratum_names), dtype=np.int64)
            for _branch in design.branch_designs
        ]
        self._lower_sums = [
            np.zeros(
                (
                    len(design.stratum_names),
                    len(branch.endpoint_action_hashes),
                    len(SIGNED_WORK_SIX_ROLES),
                ),
                dtype=object,
            )
            for branch in design.branch_designs
        ]
        self._upper_sums = [value.copy() for value in self._lower_sums]
        self._branch_chains = [
            content_hash(
                {
                    "design_hash": design_hash,
                    "branch": branch.branch.value,
                    "packed_direction_receipt_hash": (
                        branch.packed_direction_receipt.receipt_hash
                    ),
                    "action_axis_registration_hash": (
                        branch.action_axis_registration.registration_hash
                    ),
                    "record_count": 0,
                },
                prefix=_CHAIN_PREFIX,
            )
            for branch in design.branch_designs
        ]
        self._shared_record_count = 0
        self._transaction_chain = content_hash(
            {
                "design_hash": design_hash,
                "branches": tuple(value.value for value in design.branches),
                "record_count": 0,
            },
            prefix=_CHAIN_PREFIX,
        )
        # This process-local provenance bit is deliberately absent from the
        # snapshot schema.  Only rows accepted by ``record_atomic`` can
        # preserve it; public snapshot reconstruction cannot mint it.
        self._live_row_authenticated_prefix = True

    @classmethod
    def from_snapshot(
        cls,
        design: V1LabelNativeSixRoleCDesign,
        snapshot: V1LabelNativeSixRoleAccumulatorSnapshot,
        *,
        expected_snapshot_hash: str | None = None,
    ) -> "V1LabelNativeSixRoleCAccumulator":
        if type(snapshot) is not V1LabelNativeSixRoleAccumulatorSnapshot:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "restore requires the exact label-native snapshot",
            )
        if snapshot.shared_record_count and expected_snapshot_hash is None:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "nonempty restore requires an expected snapshot hash",
            )
        if (
            expected_snapshot_hash is not None
            and snapshot.snapshot_hash != expected_snapshot_hash
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "snapshot hash does not match its checkpoint manifest",
            )
        if (
            type(design) is not V1LabelNativeSixRoleCDesign
            or snapshot.design_hash != design.design_hash
            or snapshot.branches != design.branches
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "snapshot belongs to another label-native C design",
            )
        result = cls(design)
        counts: list[np.ndarray] = []
        lower_sums: list[np.ndarray] = []
        upper_sums: list[np.ndarray] = []
        for index, branch in enumerate(design.branch_designs):
            shape = result._lower_sums[index].shape
            flat_size = math.prod(shape)
            raw_counts = snapshot.counts_by_branch_and_stratum[index]
            if (
                len(raw_counts) != len(design.stratum_names)
                or any(
                    count > population
                    for count, population in zip(
                        raw_counts,
                        design.population_sizes,
                        strict=True,
                    )
                )
                or len(snapshot.lower_scaled_sums_by_branch[index]) != flat_size
                or len(snapshot.upper_scaled_sums_by_branch[index]) != flat_size
                or shape[1] != len(branch.endpoint_action_hashes)
            ):
                _refuse(
                    V1LabelNativeSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                    "snapshot sufficient-statistic shape or population is stale",
                )
            counts.append(np.asarray(raw_counts, dtype=np.int64))
            lower_sums.append(
                np.asarray(
                    snapshot.lower_scaled_sums_by_branch[index],
                    dtype=object,
                ).reshape(shape)
            )
            upper_sums.append(
                np.asarray(
                    snapshot.upper_scaled_sums_by_branch[index],
                    dtype=object,
                ).reshape(shape)
            )
        # Build and source-replay a complete receipt before exposing restored
        # state.  A malformed snapshot therefore never becomes live state.
        result._build_progress(
            counts=counts,
            lower_sums=lower_sums,
            upper_sums=upper_sums,
            branch_chains=list(snapshot.branch_observation_chain_hashes),
            shared_record_count=snapshot.shared_record_count,
            transaction_chain=snapshot.transaction_chain_hash,
        )
        result._counts = counts
        result._lower_sums = lower_sums
        result._upper_sums = upper_sums
        result._branch_chains = list(snapshot.branch_observation_chain_hashes)
        result._shared_record_count = snapshot.shared_record_count
        result._transaction_chain = snapshot.transaction_chain_hash
        result._live_row_authenticated_prefix = False
        return result

    @property
    def shared_record_count(self) -> int:
        return self._shared_record_count

    @property
    def live_row_authenticated_prefix(self) -> bool:
        """Whether the complete live prefix entered through exact C rows.

        Snapshot restore intentionally returns ``False`` even when the
        snapshot is hash-bound: sufficient statistics do not reproduce the
        row-level source/execution checks that originally authenticated them.
        """

        return self._live_row_authenticated_prefix is True

    def require_live_row_authenticated_prefix(self) -> str:
        """Return the live transaction chain or fail closed for restored state."""

        if not self.live_row_authenticated_prefix:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "C prefix was reconstructed from a snapshot, not live authenticated rows",
            )
        return self._transaction_chain

    def _validate_row(
        self,
        *,
        branch_index: int,
        row: V1LabelNativeSixRoleCRow,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        if type(row) is not V1LabelNativeSixRoleCRow:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.INVALID_TYPE,
                "atomic C acquisition requires exact unified rows",
            )
        row.assert_integrity()
        branch = self.design.branch_designs[branch_index]
        source = self.design.C_source_execution_registration
        if (
            row.C_source_execution_registration_hash != source.registration_hash
            or row.C_coordinate_proof_source_registration_hash
            != source.C_coordinate_proof_source_registration_hash
            or row.C_execution_registration_hash
            != source.C_execution_registration_hash
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.EXECUTION_IDENTITY_MISMATCH,
                "C row does not replay from the registered proof/execution source",
            )
        if (
            row.corrected_evidence_stream_identity_hash,
            row.literal_jacobian_stream_identity_hash,
            row.raw_logit_stream_identity_hash,
        ) != source.stream_identity_hashes:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STREAM_IDENTITY_MISMATCH,
                "C row value streams differ from their pre-outcome registration",
            )
        expected = {
            "design_hash": self.design.design_hash,
            "branch_design_hash": branch.design_hash,
            "branch": branch.branch,
            "chart_identity_hash": self.design.chart_identity_hash,
            "current_mask_hash": self.design.current_mask_hash,
            "route_axis_hash": self.design.route_axis_hash,
            "route_count": self.design.route_count,
            "proposal_generation": self.design.proposal_generation,
            "certification_context_population_hash": (
                self.design.certification_context_population_hash
            ),
            "certification_stream_commitment_hash": (
                self.design.certification_stream_commitment_hash
            ),
            "joint_six_role_family_registration_hash": (
                self.design.joint_six_role_family_registration_hash
            ),
            "same_capture_scorer_registration_hash": (
                self.design.same_capture_scorer_registration_hash
            ),
            "native_fiber_axis_receipt_hash": (
                self.design.native_fiber_axis_receipt_hash
            ),
            "native_fiber_measure_hash": self.design.native_fiber_measure_hash,
            "endpoint_action_hashes": branch.endpoint_action_hashes,
            "endpoint_action_identity_hashes": (
                branch.endpoint_action_identity_hashes
            ),
        }
        stale = tuple(
            name for name, value in expected.items() if getattr(row, name) != value
        )
        if stale:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_SOURCE,
                "C row differs from its frozen family at: " + ", ".join(stale),
            )
        if (
            row.packed_direction_receipt.receipt_hash
            != branch.packed_direction_receipt.receipt_hash
            or row.action_axis_registration.registration_hash
            != branch.action_axis_registration.registration_hash
            or row.eligible_action_subset_registration.registration_hash
            != branch.eligible_action_subset_registration.registration_hash
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.ACTION_AXIS_MISMATCH,
                "C row belongs to another packed direction/eligible subset",
            )
        try:
            stratum = self.design.stratum_names.index(row.stratum_id)
        except ValueError as error:
            raise V1LabelNativeSixRoleCRefusal(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "C row names an unregistered stratum",
            ) from error
        count = int(self._counts[branch_index][stratum])
        contexts = self.design.certification_context_ids_by_stratum[stratum][1]
        draws = self.design.sealed_draw_ids_by_stratum[stratum][1]
        if count >= len(contexts):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "C stratum is already a complete census",
            )
        coordinate = source.coordinate(
            stratum_id=row.stratum_id,
            sealed_order_index=count,
        )
        if (
            row.sealed_order_index != count
            or row.context_id != contexts[count]
            or row.draw_id != draws[count]
            or row.context_id != coordinate.context_id
            or row.draw_id != coordinate.draw_id
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.STALE_COORDINATE,
                "C row is not the next sealed without-replacement coordinate",
            )
        if row.C_coordinate_proof_hash != coordinate.C_coordinate_proof_hash:
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.COORDINATE_PROOF_MISMATCH,
                "C row coordinate proof was not registered for this sealed position",
            )
        if (
            row.parent_execution_identity_hash
            != coordinate.parent_execution_identity_hash
            or row.same_capture_and_vjp_execution_proof_hash
            != coordinate.same_capture_and_vjp_execution_proof_hash
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.EXECUTION_IDENTITY_MISMATCH,
                "C row parent/VJP execution identity differs from registration",
            )
        lower = row.observation_lower_by_endpoint_role
        upper = row.observation_upper_by_endpoint_role
        support = branch.analytic_support_upper_by_stratum_endpoint_role[stratum]
        if bool(np.any(lower > upper)) or bool(np.any(upper > support)):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.SUPPORT_OVERFLOW,
                "C row exceeds a pre-outcome analytic six-role support",
            )
        return stratum, lower, upper

    def record_atomic(
        self,
        rows_by_branch: Sequence[
            tuple[V1RelaxedKKTBranch, V1LabelNativeSixRoleCRow]
        ],
    ) -> V1LabelNativeSixRoleProgressReceipt:
        """Advance both branches and all roles, or mutate no live state."""

        self.design.assert_integrity()
        rows = tuple(rows_by_branch)
        if (
            len(rows) != 2
            or any(type(value) not in {tuple, list} or len(value) != 2 for value in rows)
            or tuple(value[0] for value in rows) != self.design.branches
        ):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.BRANCH_MISMATCH,
                "atomic transaction must cover both branches in frozen order",
            )
        validated = tuple(
            self._validate_row(branch_index=index, row=pair[1])
            for index, pair in enumerate(rows)
        )
        shared_coordinates = tuple(
            (
                pair[1].stratum_id,
                pair[1].sealed_order_index,
                pair[1].context_id,
                pair[1].draw_id,
                pair[1].C_coordinate_proof_hash,
                pair[1].parent_raw_target_logit_center,
                pair[1].parent_raw_target_logit_radius,
                pair[1].parent_execution_identity_hash,
                pair[1].corrected_evidence_stream_identity_hash,
                pair[1].literal_jacobian_stream_identity_hash,
                pair[1].raw_logit_stream_identity_hash,
                pair[1].same_capture_and_vjp_execution_proof_hash,
            )
            for pair in rows
        )
        if any(value != shared_coordinates[0] for value in shared_coordinates[1:]):
            _refuse(
                V1LabelNativeSixRoleCRefusalCode.CAPTURE_ALIAS,
                "branches did not measure one exact same-capture C coordinate",
            )

        # Work only on private copies until validation, hashing, interval
        # construction, and receipt source replay all succeed.
        counts = [value.copy() for value in self._counts]
        lower_sums = [value.copy() for value in self._lower_sums]
        upper_sums = [value.copy() for value in self._upper_sums]
        branch_chains = list(self._branch_chains)
        for index, (pair, (stratum, lower, upper)) in enumerate(
            zip(rows, validated, strict=True)
        ):
            for endpoint in range(lower.shape[0]):
                for role in range(lower.shape[1]):
                    lower_sums[index][stratum, endpoint, role] += (
                        _binary64_scaled_integer(float(lower[endpoint, role]))
                    )
                    upper_sums[index][stratum, endpoint, role] += (
                        _binary64_scaled_integer(float(upper[endpoint, role]))
                    )
            counts[index][stratum] += 1
            branch_chains[index] = content_hash(
                {
                    "previous_chain_hash": branch_chains[index],
                    "branch": pair[0].value,
                    "row_hash": pair[1].row_hash,
                    "packed_direction_receipt_hash": (
                        pair[1].packed_direction_receipt.receipt_hash
                    ),
                    "action_axis_registration_hash": (
                        pair[1].action_axis_registration.registration_hash
                    ),
                    "sealed_C_coordinate": shared_coordinates[0][:5],
                    "six_roles_advanced": SIGNED_WORK_SIX_ROLES,
                },
                prefix=_CHAIN_PREFIX,
            )
        new_count = self._shared_record_count + 1
        transaction_chain = content_hash(
            {
                "previous_chain_hash": self._transaction_chain,
                "shared_record_count": new_count,
                "sealed_C_coordinate": shared_coordinates[0],
                "branch_rows": tuple(
                    (pair[0].value, pair[1].row_hash) for pair in rows
                ),
                "six_roles_advanced_atomically": SIGNED_WORK_SIX_ROLES,
            },
            prefix=_CHAIN_PREFIX,
        )
        receipt = self._build_progress(
            counts=counts,
            lower_sums=lower_sums,
            upper_sums=upper_sums,
            branch_chains=branch_chains,
            shared_record_count=new_count,
            transaction_chain=transaction_chain,
        )
        self._counts = counts
        self._lower_sums = lower_sums
        self._upper_sums = upper_sums
        self._branch_chains = branch_chains
        self._shared_record_count = new_count
        self._transaction_chain = transaction_chain
        return receipt

    def _cell(
        self,
        *,
        branch_index: int,
        stratum: int,
        endpoint: int,
        role: int,
        counts: list[np.ndarray],
        lower_sums: list[np.ndarray],
        upper_sums: list[np.ndarray],
        branch_chains: list[str],
    ) -> V1LabelNativeSixRoleCellSummary:
        branch = self.design.branch_designs[branch_index]
        count = int(counts[branch_index][stratum])
        population = self.design.population_sizes[stratum]
        bound = float(
            branch.analytic_support_upper_by_stratum_endpoint_role[
                stratum,
                endpoint,
                role,
            ]
        )
        lower_sum = int(lower_sums[branch_index][stratum, endpoint, role])
        upper_sum = int(upper_sums[branch_index][stratum, endpoint, role])
        if count == 0:
            lower, upper, look_alpha = 0.0, bound, 0.0
            method = "unobserved_registered_support_v1"
        elif count == population:
            lower = _fraction_to_float_outward(
                _scaled_mean(lower_sum, count),
                direction="lower",
            )
            upper = _fraction_to_float_outward(
                _scaled_mean(upper_sum, count),
                direction="upper",
            )
            look_alpha = 0.0
            method = LABEL_NATIVE_SIX_ROLE_CENSUS_METHOD
        else:
            lower_mean = _fraction_to_float_outward(
                _scaled_mean(lower_sum, count),
                direction="lower",
            )
            upper_mean = _fraction_to_float_outward(
                _scaled_mean(upper_sum, count),
                direction="upper",
            )
            look_alpha = self.design.look_alpha(
                branch=branch.branch,
                stratum_index=stratum,
                sample_size=count,
            )
            radius = _serfling_radius_upper(
                upper_bound=bound,
                count=count,
                population=population,
                look_alpha=look_alpha,
            )
            lower = max(
                0.0,
                math.nextafter(lower_mean - radius, -math.inf),
            )
            upper = min(
                bound,
                math.nextafter(upper_mean + radius, math.inf),
            )
            method = LABEL_NATIVE_SIX_ROLE_METHOD
        role_name = SIGNED_WORK_SIX_ROLES[role]
        return V1LabelNativeSixRoleCellSummary(
            branch=branch.branch,
            stratum_id=self.design.stratum_names[stratum],
            action_identity=branch.action_identities[endpoint],
            role=role_name,
            dose=SIGNED_WORK_SIX_ROLE_DOSES[role_name],
            population_size=population,
            sample_size=count,
            lower=lower,
            upper=upper,
            value_upper_bound=bound,
            current_lower_look_alpha=look_alpha,
            current_upper_look_alpha=look_alpha,
            full_census=count == population,
            method=method,
            branch_observation_chain_hash=branch_chains[branch_index],
        )

    def _build_progress(
        self,
        *,
        counts: list[np.ndarray],
        lower_sums: list[np.ndarray],
        upper_sums: list[np.ndarray],
        branch_chains: list[str],
        shared_record_count: int,
        transaction_chain: str,
    ) -> V1LabelNativeSixRoleProgressReceipt:
        branch_receipts: list[V1LabelNativeSixRoleBranchProgress] = []
        for branch_index, branch in enumerate(self.design.branch_designs):
            cells = tuple(
                self._cell(
                    branch_index=branch_index,
                    stratum=stratum,
                    endpoint=endpoint,
                    role=role,
                    counts=counts,
                    lower_sums=lower_sums,
                    upper_sums=upper_sums,
                    branch_chains=branch_chains,
                )
                for stratum in range(len(self.design.stratum_names))
                for endpoint in range(len(branch.endpoint_action_hashes))
                for role in range(len(SIGNED_WORK_SIX_ROLES))
            )
            weights = tuple(value for _name, value in self.design.stratum_weights)
            endpoints: list[V1LabelNativeSixRoleEndpointSummary] = []
            for endpoint, action_identity in enumerate(branch.action_identities):
                lowers: dict[str, float] = {}
                uppers: dict[str, float] = {}
                for role_index, role in enumerate(SIGNED_WORK_SIX_ROLES):
                    role_cells = tuple(
                        cells[
                            (
                                stratum * len(branch.endpoint_action_hashes)
                                + endpoint
                            )
                            * len(SIGNED_WORK_SIX_ROLES)
                            + role_index
                        ]
                        for stratum in range(len(self.design.stratum_names))
                    )
                    lowers[role] = _downward_weighted_sum(
                        weights,
                        tuple(value.lower for value in role_cells),
                    )
                    uppers[role] = _outward_weighted_sum(
                        weights,
                        tuple(value.upper for value in role_cells),
                    )
                endpoints.append(
                    V1LabelNativeSixRoleEndpointSummary(
                        branch=branch.branch,
                        action_identity=action_identity,
                        f_native_lower=lowers[F_NATIVE_ROLE],
                        f_native_upper=uppers[F_NATIVE_ROLE],
                        q_j_lower=lowers[Q_J_ROLE],
                        q_j_upper=uppers[Q_J_ROLE],
                        r_half_lower=lowers[R_HALF_ROLE],
                        r_half_upper=uppers[R_HALF_ROLE],
                        r_one_lower=lowers[R_ONE_ROLE],
                        r_one_upper=uppers[R_ONE_ROLE],
                        d_half_lower=lowers[D_HALF_ROLE],
                        d_half_upper=uppers[D_HALF_ROLE],
                        d_one_lower=lowers[D_ONE_ROLE],
                        d_one_upper=uppers[D_ONE_ROLE],
                        chart_half_burden_lower=_response_burden_lower(
                            eta=0.5,
                            q_lower=lowers[Q_J_ROLE],
                            r_lower=lowers[R_HALF_ROLE],
                        ),
                        chart_half_burden_upper=_response_burden(
                            eta=0.5,
                            q_upper=uppers[Q_J_ROLE],
                            r_upper=uppers[R_HALF_ROLE],
                        ),
                        chart_one_burden_lower=_response_burden_lower(
                            eta=1.0,
                            q_lower=lowers[Q_J_ROLE],
                            r_lower=lowers[R_ONE_ROLE],
                        ),
                        chart_one_burden_upper=_response_burden(
                            eta=1.0,
                            q_upper=uppers[Q_J_ROLE],
                            r_upper=uppers[R_ONE_ROLE],
                        ),
                    )
                )
            branch_counts = tuple(int(value) for value in counts[branch_index])
            contexts = self.design.certification_context_ids_by_stratum
            draws = self.design.sealed_draw_ids_by_stratum
            branch_receipts.append(
                V1LabelNativeSixRoleBranchProgress(
                    branch=branch.branch,
                    branch_design_hash=branch.design_hash,
                    packed_direction_receipt_hash=(
                        branch.packed_direction_receipt.receipt_hash
                    ),
                    action_axis_registration_hash=(
                        branch.action_axis_registration.registration_hash
                    ),
                    source_shell_ladder_receipt_hash=(
                        branch.action_axis_registration.source_shell_ladder.receipt_hash
                    ),
                    compact_C_bridge_branch_receipt_hash=(
                        branch.compact_C_bridge_branch_receipt.receipt_hash
                    ),
                    eligible_action_subset_registration_hash=(
                        branch.eligible_action_subset_registration.registration_hash
                    ),
                    eligible_source_endpoint_labels=(
                        branch.eligible_action_subset_registration.eligible_source_endpoint_labels
                    ),
                    observed_counts_by_stratum=branch_counts,
                    next_expected_context_by_stratum=tuple(
                        None if count == len(contexts[index][1]) else contexts[index][1][count]
                        for index, count in enumerate(branch_counts)
                    ),
                    next_expected_draw_by_stratum=tuple(
                        None if count == len(draws[index][1]) else draws[index][1][count]
                        for index, count in enumerate(branch_counts)
                    ),
                    cell_summaries=cells,
                    endpoint_summaries=tuple(endpoints),
                    branch_observation_chain_hash=branch_chains[branch_index],
                )
            )
        observed = sum(int(value) for value in counts[0])
        population = sum(self.design.population_sizes)
        status: LabelNativeSixRoleSamplingStatus
        if observed == 0:
            status = "awaiting_C"
        elif observed == population:
            status = "full_C_census"
        else:
            status = "partial_C_evidence"
        receipt = V1LabelNativeSixRoleProgressReceipt(
            design_hash=self.design.design_hash,
            chart_identity_hash=self.design.chart_identity_hash,
            current_mask_hash=self.design.current_mask_hash,
            route_axis_hash=self.design.route_axis_hash,
            route_count=self.design.route_count,
            proposal_generation=self.design.proposal_generation,
            certification_context_population_hash=(
                self.design.certification_context_population_hash
            ),
            certification_stream_commitment_hash=(
                self.design.certification_stream_commitment_hash
            ),
            joint_six_role_family_registration_hash=(
                self.design.joint_six_role_family_registration_hash
            ),
            same_capture_scorer_registration_hash=(
                self.design.same_capture_scorer_registration_hash
            ),
            native_fiber_axis_receipt_hash=(
                self.design.native_fiber_axis_receipt_hash
            ),
            native_fiber_measure_hash=self.design.native_fiber_measure_hash,
            trajectory_alpha_ledger_hash=self.design.trajectory_alpha_ledger_hash,
            branch_receipts=tuple(branch_receipts),
            shared_record_count=shared_record_count,
            transaction_chain_hash=transaction_chain,
            sampling_status=status,
            tail_count=self.design.tail_count,
            registered_tail_alpha_total=self.design.registered_tail_alpha_total,
        )
        receipt.assert_replays(self.design)
        return receipt

    def progress_receipt(self) -> V1LabelNativeSixRoleProgressReceipt:
        self.design.assert_integrity()
        return self._build_progress(
            counts=self._counts,
            lower_sums=self._lower_sums,
            upper_sums=self._upper_sums,
            branch_chains=self._branch_chains,
            shared_record_count=self._shared_record_count,
            transaction_chain=self._transaction_chain,
        )

    def export_snapshot(self) -> V1LabelNativeSixRoleAccumulatorSnapshot:
        return V1LabelNativeSixRoleAccumulatorSnapshot(
            design_hash=self.design.design_hash,
            branches=self.design.branches,
            counts_by_branch_and_stratum=tuple(
                tuple(int(value) for value in row) for row in self._counts
            ),
            lower_scaled_sums_by_branch=tuple(
                tuple(int(value) for value in row.reshape(-1))
                for row in self._lower_sums
            ),
            upper_scaled_sums_by_branch=tuple(
                tuple(int(value) for value in row.reshape(-1))
                for row in self._upper_sums
            ),
            branch_observation_chain_hashes=tuple(self._branch_chains),
            shared_record_count=self._shared_record_count,
            transaction_chain_hash=self._transaction_chain,
        )


__all__ = [
    "D_HALF_ROLE",
    "D_ONE_ROLE",
    "F_NATIVE_ROLE",
    "LABEL_NATIVE_SIX_ROLE_ALPHA_POLICY",
    "LABEL_NATIVE_SIX_ROLE_AUTHORITY",
    "Q_J_ROLE",
    "R_HALF_ROLE",
    "R_ONE_ROLE",
    "SIGNED_WORK_SIX_ROLES",
    "V1LabelNativeSixRoleActionIdentity",
    "V1LabelNativeSixRoleAccumulatorSnapshot",
    "V1LabelNativeSixRoleBranchProgress",
    "V1LabelNativeSixRoleCAccumulator",
    "V1LabelNativeSixRoleCBranchDesign",
    "V1LabelNativeSixRoleCDesign",
    "V1LabelNativeSixRoleCRefusal",
    "V1LabelNativeSixRoleCRefusalCode",
    "V1LabelNativeSixRoleCRow",
    "V1LabelNativeSixRoleCellSummary",
    "V1LabelNativeSixRoleEndpointSummary",
    "V1LabelNativeSixRoleProgressReceipt",
]
