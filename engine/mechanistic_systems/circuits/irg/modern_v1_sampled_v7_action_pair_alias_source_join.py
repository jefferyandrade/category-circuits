"""Exact live ancestry join for a sampled-V7 branch action pair.

``V1SampledV7BranchActionPair`` deliberately owns only the two branch aliases
and deduplicated packed children.  In particular, physical deduplication can
retain one action object for two branch crosswalks, and a no-action alias owns
no action object at all.  Consequently an action pair cannot later prove its
crosswalk ancestry by comparing hashes to an independently supplied pre-C
bundle.

This module closes that narrow provenance seam in two authority-inert stages:

* registration atomically issues the canonical pair from the exact live
  common-coordinate context and retains that context's two crosswalk objects;
* joining follows the packed-family builder's exact normalization edges
  (retained parent objects by ``is``, never digest equality), requires the
  normalized wrappers themselves to occur in one qualification source cell,
  then replays the coordinate, uncertainty, and action images.

Neither receipt certifies D stability or opens C, K, O, mask-commit, terminal,
performance, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from itertools import product
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_v7_branch_action_pair import (
    V1SampledV7BranchActionPair,
    issue_v1_sampled_v7_branch_action_pair,
)
from .modern_v1_sampled_v7_branch_decision_image_production import (
    V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
)
from .modern_v1_sampled_v7_complete_action_coordinate_packed_crosswalk import (
    V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt,
)
from .modern_v1_sampled_v7_d_source_image_binding import (
    V1SampledV7DSourceImageBindingReceipt,
)
from .modern_v1_signed_work_decision_witness import (
    V1SignedWorkDecisionUncertaintySet,
)
from .modern_v1_signed_work_decision_witness_cuda import (
    V1SignedWorkCUDADecisionWitnessQualification,
    V1SignedWorkCUDARegisteredUncertaintyCell,
)
from .modern_v1_signed_work_hard_fq_replay_cuda import (
    V1SignedWorkHardFQEndpointStatus,
)
from .modern_v1_signed_work_phase_a_alias_safe_margin_cuda import (
    V1PhaseACompleteRegisteredAliasMapReceipt,
)
from .modern_v1_signed_work_pre_c_coordinator import (
    V1SignedWorkPreCLiveCommonCoordinateContext,
)
from .modern_v1_signed_work_pre_c_packed_actions import (
    V1SignedWorkCoordinateCandidateCrosswalk,
    v1_signed_work_packed_action_dedup_identity,
)


SAMPLED_V7_ACTION_PAIR_SOURCE_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_action_pair_exact_pre_C_source_registration_v1"
)
SAMPLED_V7_ACTION_PAIR_ALIAS_SOURCE_JOIN_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_action_pair_alias_map_exact_source_graph_join_v1"
)
SAMPLED_V7_ACTION_PAIR_SOURCE_JOIN_AUTHORITY: Final = (
    "exact_live_pre_C_crosswalk_identity_and_alias_source_graph_ancestry_only_"
    "no_D_stability_C_K_O_commit_terminal_performance_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-action-pair-source-join-"
_REGISTRATION_TOKEN: Final = object()
_JOIN_TOKEN: Final = object()
_BRANCH_AXIS: Final = (
    V1RelaxedKKTBranch.SPECTRAL,
    V1RelaxedKKTBranch.DIAGONAL_L1,
)


class V1SampledV7ActionPairSourceJoinRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    PAIR_SOURCE_MISMATCH = "pair_source_mismatch"
    FOREIGN_CROSSWALK_IDENTITY = "foreign_crosswalk_identity"
    ALIAS_SOURCE_MISMATCH = "alias_source_mismatch"
    UNCERTAINTY_GRAPH_MISMATCH = "uncertainty_graph_mismatch"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    ACTION_AXIS_MISMATCH = "action_axis_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7ActionPairSourceJoinRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7ActionPairSourceJoinRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7ActionPairSourceJoinRefusalCode:
            raise TypeError("action-pair source-join refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7ActionPairSourceJoinRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7ActionPairSourceJoinRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _closed_authority(values: tuple[bool, ...], *, owner: str) -> None:
    if any(type(value) is not bool for value in values) or any(values):
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its exact-source-only authority boundary",
        )


def _crosswalk_axis(
    values: object,
) -> tuple[
    V1SignedWorkCoordinateCandidateCrosswalk,
    V1SignedWorkCoordinateCandidateCrosswalk,
]:
    if type(values) is not tuple:
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.INVALID_TYPE,
            "candidate crosswalk custody must be one immutable tuple",
        )
    axis = tuple(values)
    if (
        len(axis) != 2
        or any(type(value) is not V1SignedWorkCoordinateCandidateCrosswalk for value in axis)
        or tuple(value.q_branch for value in axis) != _BRANCH_AXIS
    ):
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.PAIR_SOURCE_MISMATCH,
            "registration lost the exact canonical two-branch crosswalk axis",
        )
    return axis  # type: ignore[return-value]


def _replay_pair_crosswalk_image(
    *,
    pair: V1SampledV7BranchActionPair,
    crosswalks: tuple[
        V1SignedWorkCoordinateCandidateCrosswalk,
        V1SignedWorkCoordinateCandidateCrosswalk,
    ],
) -> dict[str, object]:
    try:
        pair_hash = pair.assert_integrity()
        for value in crosswalks:
            value.__post_init__()
    except V1SampledV7ActionPairSourceJoinRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.STALE_SOURCE,
            f"pair or registered crosswalk failed replay: {type(error).__name__}: {error}",
        )
    if pair.branch_axis != _BRANCH_AXIS:
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.PAIR_SOURCE_MISMATCH,
            "pair lost the canonical branch axis",
        )

    for alias, crosswalk in zip(pair.branch_aliases, crosswalks, strict=True):
        common = (
            alias.branch is crosswalk.q_branch,
            alias.q_status is crosswalk.q_status,
            alias.coordinate_candidate_key_hash
            == crosswalk.coordinate_candidate_key_hash,
            alias.resource_coordinate_hash == crosswalk.resource_coordinate_hash,
            alias.coordinate_ordinal == crosswalk.coordinate_ordinal,
            alias.source_endpoint_label == crosswalk.source_endpoint_label,
            alias.candidate_crosswalk_hash == crosswalk.crosswalk_hash,
            alias.source_materialization_bundle_hash
            == crosswalk.source_materialization_bundle_hash,
            alias.endpoint_decision_receipt_hash
            == crosswalk.endpoint_decision.receipt_hash,
            alias.parent_mask_hash == crosswalk.parent_mask_hash,
            alias.route_axis_hash == crosswalk.route_axis_hash,
            alias.route_count == crosswalk.route_count,
        )
        if not all(common):
            _refuse(
                V1SampledV7ActionPairSourceJoinRefusalCode.PAIR_SOURCE_MISMATCH,
                "pair alias differs from its atomically retained source crosswalk",
            )
        safe = (
            crosswalk.q_status
            is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        )
        if safe:
            action = crosswalk.packed_action
            if action is None or alias.physical_action_ordinal is None:
                _refuse(
                    V1SampledV7ActionPairSourceJoinRefusalCode.ACTION_AXIS_MISMATCH,
                    "a Q-safe registered crosswalk lost its packed child",
                )
            custody = pair.physical_actions[alias.physical_action_ordinal - 1]
            if (
                v1_signed_work_packed_action_dedup_identity(action)
                != custody.dedup_identity
                or action.resulting_mask.packed_bits
                != custody.packed_action.resulting_mask.packed_bits
                or alias.packed_action_hash != action.action_hash
                or alias.resulting_mask_hash != action.resulting_mask.mask_hash
                or alias.removed_route_index_hash != action.removed_route_index_hash
                or alias.removed_count != action.removed_count
            ):
                _refuse(
                    V1SampledV7ActionPairSourceJoinRefusalCode.ACTION_AXIS_MISMATCH,
                    "registered branch action differs from pair physical custody",
                )
        elif alias.no_action is None or crosswalk.packed_action is not None:
            _refuse(
                V1SampledV7ActionPairSourceJoinRefusalCode.ACTION_AXIS_MISMATCH,
                "non-Q-safe registered branch did not remain an exact no-action alias",
            )

    first = crosswalks[0]
    if (
        any(
            value.resource_coordinate_hash != pair.resource_coordinate_hash
            for value in crosswalks
        )
        or any(value.coordinate_ordinal != pair.coordinate_ordinal for value in crosswalks)
        or any(value.parent_mask_hash != pair.parent_mask_hash for value in crosswalks)
        or any(value.route_axis_hash != pair.route_axis_hash for value in crosswalks)
        or any(value.route_count != pair.route_count for value in crosswalks)
        or first.parent_mask_hash != pair.parent_mask_hash
    ):
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.COORDINATE_MISMATCH,
            "registered crosswalks crossed pair coordinate/parent/route identity",
        )
    return {
        "pair_hash": pair_hash,
        "branch_crosswalk_hashes": tuple(value.crosswalk_hash for value in crosswalks),
        "source_bundle_hash": pair.source_bundle_hash,
        "source_registration_hash": pair.source_registration_hash,
        "registered_d_coordinate_hash": pair.registered_d_coordinate_hash,
        "response_geometry_problem_hash": pair.response_geometry_problem_hash,
        "response_geometry_source_hash": pair.response_geometry_source_hash,
        "least_work_problem_hash": pair.least_work_problem_hash,
        "resource_coordinate_hash": pair.resource_coordinate_hash,
        "coordinate_ordinal": pair.coordinate_ordinal,
        "parent_mask_hash": pair.parent_mask_hash,
        "route_axis_hash": pair.route_axis_hash,
        "route_count": pair.route_count,
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7ActionPairSourceRegistrationReceipt:
    """Atomic pair issuance plus exact two-crosswalk parent custody."""

    pair_hash: str
    branch_crosswalk_hashes: tuple[str, str]
    source_bundle_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    response_geometry_problem_hash: str
    response_geometry_source_hash: str
    least_work_problem_hash: str
    resource_coordinate_hash: str
    coordinate_ordinal: int
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    _action_pair: V1SampledV7BranchActionPair = field(repr=False, compare=False)
    _candidate_crosswalks: tuple[
        V1SignedWorkCoordinateCandidateCrosswalk,
        V1SignedWorkCoordinateCandidateCrosswalk,
    ] = field(repr=False, compare=False)
    _issuer_token: InitVar[object]
    _integrity_seal: str = field(init=False, repr=False, compare=False)
    exact_live_crosswalk_objects_retained: bool = True
    array_free_public_receipt: bool = True
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_ACTION_PAIR_SOURCE_JOIN_AUTHORITY
    schema: str = SAMPLED_V7_ACTION_PAIR_SOURCE_REGISTRATION_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _REGISTRATION_TOKEN:
            raise TypeError("action-pair source registrations are issuer-only")
        if type(self._action_pair) is not V1SampledV7BranchActionPair:
            _refuse(
                V1SampledV7ActionPairSourceJoinRefusalCode.INVALID_TYPE,
                "registration requires one exact action pair",
            )
        axis = _crosswalk_axis(self._candidate_crosswalks)
        object.__setattr__(self, "_candidate_crosswalks", axis)
        observed = _replay_pair_crosswalk_image(pair=self._action_pair, crosswalks=axis)
        if any(getattr(self, name) != value for name, value in observed.items()):
            _refuse(
                V1SampledV7ActionPairSourceJoinRefusalCode.PAIR_SOURCE_MISMATCH,
                "registration public image differs from atomic pair issuance",
            )
        _closed_authority(
            (
                self.d_stability_authority,
                self.c_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.performance_authority,
                self.pareto_authority,
            ),
            owner="action-pair source registration",
        )
        if (
            self.exact_live_crosswalk_objects_retained is not True
            or self.array_free_public_receipt is not True
            or self.authority_scope != SAMPLED_V7_ACTION_PAIR_SOURCE_JOIN_AUTHORITY
            or self.schema != SAMPLED_V7_ACTION_PAIR_SOURCE_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1SampledV7ActionPairSourceJoinRefusalCode.AUTHORITY_VIOLATION,
                "registration semantics changed",
            )
        object.__setattr__(self, "_integrity_seal", self.receipt_hash)

    @property
    def action_pair(self) -> V1SampledV7BranchActionPair:
        self.assert_integrity()
        return self._action_pair

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "pair_hash": self.pair_hash,
                "branch_crosswalk_hashes": self.branch_crosswalk_hashes,
                "source_bundle_hash": self.source_bundle_hash,
                "source_registration_hash": self.source_registration_hash,
                "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
                "response_geometry_problem_hash": self.response_geometry_problem_hash,
                "response_geometry_source_hash": self.response_geometry_source_hash,
                "least_work_problem_hash": self.least_work_problem_hash,
                "resource_coordinate_hash": self.resource_coordinate_hash,
                "coordinate_ordinal": self.coordinate_ordinal,
                "parent_mask_hash": self.parent_mask_hash,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "exact_live_crosswalk_objects_retained": True,
                "array_free_public_receipt": True,
                "all_downstream_authority": False,
                "authority_scope": self.authority_scope,
            },
            prefix=_PREFIX + "registration-",
        )

    def assert_integrity(self) -> str:
        observed = _replay_pair_crosswalk_image(
            pair=self._action_pair,
            crosswalks=_crosswalk_axis(self._candidate_crosswalks),
        )
        if (
            any(getattr(self, name) != value for name, value in observed.items())
            or self.receipt_hash != self._integrity_seal
        ):
            _refuse(
                V1SampledV7ActionPairSourceJoinRefusalCode.STALE_SOURCE,
                "action-pair source registration no longer replays",
            )
        return self.receipt_hash

    def __reduce__(self) -> NoReturn:
        raise TypeError("action-pair source registrations are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("action-pair source registrations are nonserializable")


def issue_v1_sampled_v7_action_pair_source_registration(
    *,
    source_context: V1SignedWorkPreCLiveCommonCoordinateContext,
) -> V1SampledV7ActionPairSourceRegistrationReceipt:
    """Atomically issue a pair and retain its exact live crosswalk parents."""

    if type(source_context) is not V1SignedWorkPreCLiveCommonCoordinateContext:
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.INVALID_TYPE,
            "source registration requires the exact live common-coordinate context",
        )
    pair = issue_v1_sampled_v7_branch_action_pair(source_context=source_context)
    crosswalks = _crosswalk_axis(tuple(source_context.candidate_crosswalks))
    public = _replay_pair_crosswalk_image(pair=pair, crosswalks=crosswalks)
    return V1SampledV7ActionPairSourceRegistrationReceipt(
        **public,
        _action_pair=pair,
        _candidate_crosswalks=crosswalks,
        _issuer_token=_REGISTRATION_TOKEN,
    )


def _owner_hashes(
    uncertainty: V1SignedWorkDecisionUncertaintySet,
) -> tuple[str, str, str, str]:
    return (
        uncertainty.signed_f_sampling.set_hash,
        uncertainty.literal_j_sampling.set_hash,
        uncertainty.fd_residual.set_hash,
        uncertainty.solver_rounding.set_hash,
    )


def _is_exact_canonical_normalization_descendant(
    *,
    observed: V1SignedWorkCoordinateCandidateCrosswalk,
    source: V1SignedWorkCoordinateCandidateCrosswalk,
) -> bool:
    """Recognize the packed-family builder's exact normalization edge.

    The builder keeps non-Q-safe crosswalks by identity.  For Q-safe aliases
    it uses ``dataclasses.replace(crosswalk, packed_action=canonical)``.  The
    replacement therefore has a new wrapper identity but retains the exact
    endpoint, source action, and materialization parent objects.  Those live
    object edges, together with complete public-field equality, are the typed
    ancestry proof; a matching digest alone is never sufficient.
    """

    if type(observed) is not V1SignedWorkCoordinateCandidateCrosswalk:
        return False
    if source.q_status is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE:
        return observed is source
    if (
        observed.endpoint_decision is not source.endpoint_decision
        or observed.source_packed_action is not source.source_packed_action
        or observed.materialization_receipt is not source.materialization_receipt
    ):
        return False
    names = (
        "coordinate_candidate_key_hash",
        "resource_coordinate_hash",
        "coordinate_ordinal",
        "q_branch",
        "source_endpoint_label",
        "source_materialization_bundle_hash",
        "parent_mask_hash",
        "route_axis_hash",
        "route_count",
        "source_retained_host_packed_payload_bytes",
        "source_peak_host_workspace_upper_bound_bytes",
        "source_cuda_logical_workspace_upper_bound_bytes",
        "source_packed_payload_d2h_bytes",
        "authority_scope",
        "schema",
    )
    if any(getattr(observed, name) != getattr(source, name) for name in names):
        return False
    if observed.packed_action is None or source.packed_action is None:
        return False
    return (
        v1_signed_work_packed_action_dedup_identity(observed.packed_action)
        == v1_signed_work_packed_action_dedup_identity(source.packed_action)
        and observed.packed_action.resulting_mask.packed_bits
        == source.packed_action.resulting_mask.packed_bits
        and observed.crosswalk_hash == source.crosswalk_hash
    )


def _matching_source_cells(
    *,
    registration: V1SampledV7ActionPairSourceRegistrationReceipt,
    qualification: V1SignedWorkCUDADecisionWitnessQualification,
) -> tuple[
    tuple[
        V1SignedWorkCUDARegisteredUncertaintyCell,
        tuple[
            V1SignedWorkCoordinateCandidateCrosswalk,
            V1SignedWorkCoordinateCandidateCrosswalk,
        ],
    ],
    ...,
]:
    pair = registration._action_pair
    source_axis = registration._candidate_crosswalks
    matches: list[
        tuple[
            V1SignedWorkCUDARegisteredUncertaintyCell,
            tuple[
                V1SignedWorkCoordinateCandidateCrosswalk,
                V1SignedWorkCoordinateCandidateCrosswalk,
            ],
        ]
    ] = []
    for cell in qualification.source_cells:
        candidate_axis = tuple(
            value
            for value in cell.packed_action_family.candidate_crosswalks
            if value.resource_coordinate_hash == pair.resource_coordinate_hash
            and value.coordinate_ordinal == pair.coordinate_ordinal
        )
        if len(candidate_axis) == 2 and all(
            _is_exact_canonical_normalization_descendant(
                observed=observed,
                source=expected,
            )
            for observed, expected in zip(candidate_axis, source_axis, strict=True)
        ):
            normalized = candidate_axis
            matches.append((cell, normalized))  # type: ignore[arg-type]
    if not matches:
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.FOREIGN_CROSSWALK_IDENTITY,
            "no qualification cell contains the exact canonical-normalization "
            "descendants of the registered crosswalks",
        )
    return tuple(matches)


def _replay_join(
    *,
    registration: V1SampledV7ActionPairSourceRegistrationReceipt,
    alias_map: V1PhaseACompleteRegisteredAliasMapReceipt,
) -> dict[str, object]:
    try:
        registration_hash = registration.assert_integrity()
        alias_map_hash = alias_map.assert_live_identity()
        source_image = alias_map.source_image
        source_image_hash = source_image.assert_integrity()
    except V1SampledV7ActionPairSourceJoinRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.STALE_SOURCE,
            f"registration or alias source failed replay: {type(error).__name__}: {error}",
        )
    if type(source_image) is not V1SampledV7DSourceImageBindingReceipt:
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.INVALID_TYPE,
            "alias map lost its exact D source image",
        )
    complete = source_image._complete_crosswalk
    qualification = source_image._cuda_qualification
    if (
        type(complete)
        is not V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt
        or type(qualification) is not V1SignedWorkCUDADecisionWitnessQualification
        or complete._cuda_qualification is not qualification
        or alias_map.source_image_receipt_hash != source_image_hash
        or alias_map.complete_crosswalk_receipt_hash != complete.receipt_hash
        or alias_map.qualification_hash != qualification.qualification_hash
    ):
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.ALIAS_SOURCE_MISMATCH,
            "alias map is not the exact complete-crosswalk/qualification source image",
        )

    pair = registration._action_pair
    if (
        source_image.parent_mask_hash != pair.parent_mask_hash
        or source_image.route_axis_hash != pair.route_axis_hash
        or source_image.route_count != pair.route_count
        or qualification.parent_mask_hash != pair.parent_mask_hash
        or qualification.route_axis_hash != pair.route_axis_hash
        or qualification.route_count != pair.route_count
    ):
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.COORDINATE_MISMATCH,
            "pair and alias source image use different parent or route axes",
        )

    branch_evidence = tuple(source_image._branch_evidence)
    if (
        len(branch_evidence) != 2
        or any(
            type(value) is not V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt
            for value in branch_evidence
        )
        or tuple(value.branch for value in branch_evidence) != _BRANCH_AXIS
    ):
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.UNCERTAINTY_GRAPH_MISMATCH,
            "source image lost its exact canonical branch evidence axis",
        )
    uncertainty = branch_evidence[0]._uncertainty
    if (
        type(uncertainty) is not V1SignedWorkDecisionUncertaintySet
        or branch_evidence[1]._uncertainty is not uncertainty
    ):
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.UNCERTAINTY_GRAPH_MISMATCH,
            "branch evidence does not share one exact uncertainty owner object",
        )
    owner_hashes = _owner_hashes(uncertainty)
    qualification_owner_hashes = (
        qualification.signed_f_uncertainty_set_hash,
        qualification.literal_j_uncertainty_set_hash,
        qualification.fd_residual_uncertainty_set_hash,
        qualification.solver_uncertainty_set_hash,
    )
    expected_grid = tuple(product(*uncertainty.realization_axes))
    if (
        uncertainty.uncertainty_hash != source_image.uncertainty_set_hash
        or uncertainty.uncertainty_hash != qualification.uncertainty_set_hash
        or uncertainty.route_axis_hash != pair.route_axis_hash
        or uncertainty.registered_d_w_prefix_hash != pair.source_registration_hash
        or qualification.registered_d_w_prefix_hash != pair.source_registration_hash
        or qualification.registered_d_coordinate_hash
        != pair.registered_d_coordinate_hash
        or owner_hashes != qualification_owner_hashes
        or tuple(value.grid_key for value in qualification.source_cells) != expected_grid
        or any(
            value.uncertainty_set_hash != uncertainty.uncertainty_hash
            or value.registered_d_w_prefix_hash != pair.source_registration_hash
            or value.registered_d_coordinate_hash
            != pair.registered_d_coordinate_hash
            or (
                value.signed_f_uncertainty_set_hash,
                value.literal_j_uncertainty_set_hash,
                value.fd_residual_uncertainty_set_hash,
                value.solver_uncertainty_set_hash,
            )
            != owner_hashes
            for value in qualification.source_cells
        )
    ):
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.UNCERTAINTY_GRAPH_MISMATCH,
            "pair source registration and alias source do not share one complete "
            "uncertainty graph",
        )

    matching_cells = _matching_source_cells(
        registration=registration,
        qualification=qualification,
    )
    row_hashes: list[str] = []
    for cell, normalized_crosswalks in matching_cells:
        for alias, source_crosswalk in zip(
            pair.branch_aliases,
            normalized_crosswalks,
            strict=True,
        ):
            bindings = tuple(
                value
                for value in cell.phase_b_state_bindings
                if value.resource_coordinate_hash == pair.resource_coordinate_hash
                and value.coordinate_ordinal == pair.coordinate_ordinal
                and alias.branch in value.q_branches
            )
            if len(bindings) != 1:
                _refuse(
                    V1SampledV7ActionPairSourceJoinRefusalCode.COORDINATE_MISMATCH,
                    "registered crosswalk has no unique same-coordinate Phase-B state",
                )
            problem = bindings[0].problem
            if (
                problem.problem_hash != pair.least_work_problem_hash
                or problem.source_registration_hash != pair.source_registration_hash
                or problem.registered_d_coordinate_hash
                != pair.registered_d_coordinate_hash
                or problem.source_problem.problem_hash
                != pair.response_geometry_problem_hash
                or problem.source_problem.source_geometry_hash
                != pair.response_geometry_source_hash
            ):
                _refuse(
                    V1SampledV7ActionPairSourceJoinRefusalCode.COORDINATE_MISMATCH,
                    "exact crosswalk cell descends from a different source "
                    "registration or geometry",
                )

            rows = tuple(
                value
                for value in complete.structural_rows
                if value.qualification_registered_cell_hash
                == cell.registered_cell_hash
                and value.qualification_grid_key == cell.grid_key
                and value.branch is alias.branch
                and value.resource_coordinate_hash
                == pair.resource_coordinate_hash
                and value.qualification_crosswalk_hash
                == source_crosswalk.crosswalk_hash
            )
            safe = (
                alias.q_status
                is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
            )
            if safe and not rows:
                _refuse(
                    V1SampledV7ActionPairSourceJoinRefusalCode.ACTION_AXIS_MISMATCH,
                    "Q-safe exact source crosswalk is absent from the structural action image",
                )
            if not safe and rows:
                _refuse(
                    V1SampledV7ActionPairSourceJoinRefusalCode.ACTION_AXIS_MISMATCH,
                    "a no-action source crosswalk acquired a structural action row",
                )
            if safe:
                assert alias.physical_action_ordinal is not None
                expected = pair.physical_actions[
                    alias.physical_action_ordinal - 1
                ].dedup_identity
                if any(
                    value.packed_dedup_identity != expected
                    or value.qualification_endpoint_decision_hash
                    != source_crosswalk.endpoint_decision.receipt_hash
                    for value in rows
                ):
                    _refuse(
                        V1SampledV7ActionPairSourceJoinRefusalCode.ACTION_AXIS_MISMATCH,
                        "structural row differs from the exact pair action/endpoint",
                    )
                row_hashes.extend(value.structural_row_hash for value in rows)

    expected_alias_rows = {
        value.structural_row_hash: value
        for value in complete.structural_rows
        if value.structural_row_hash in row_hashes
    }
    observed_alias_rows = {
        value.source_structural_row_hash: value
        for value in alias_map.alias_rows
        if value.source_structural_row_hash in expected_alias_rows
    }
    if (
        set(observed_alias_rows) != set(expected_alias_rows)
        or alias_map.branch_action_projection_domain
        != complete.branch_action_projection_domain
        or any(
            observed_alias_rows[key].branch is not row.branch
            or observed_alias_rows[key].resource_coordinate_hash
            != row.resource_coordinate_hash
            or observed_alias_rows[key].qualification_crosswalk_hash
            != row.qualification_crosswalk_hash
            or observed_alias_rows[key].packed_dedup_identity
            != row.packed_dedup_identity
            for key, row in expected_alias_rows.items()
        )
    ):
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.ALIAS_SOURCE_MISMATCH,
            "alias map changed the exact source structural-row/action image",
        )

    return {
        "registration_receipt_hash": registration_hash,
        "pair_hash": pair.pair_hash,
        "alias_map_receipt_hash": alias_map_hash,
        "source_image_receipt_hash": source_image_hash,
        "complete_crosswalk_receipt_hash": complete.receipt_hash,
        "qualification_hash": qualification.qualification_hash,
        "uncertainty_set_hash": uncertainty.uncertainty_hash,
        "source_registration_hash": pair.source_registration_hash,
        "registered_d_coordinate_hash": pair.registered_d_coordinate_hash,
        "resource_coordinate_hash": pair.resource_coordinate_hash,
        "coordinate_ordinal": pair.coordinate_ordinal,
        "branch_crosswalk_hashes": registration.branch_crosswalk_hashes,
        "matching_registered_cell_hashes": tuple(
            value.registered_cell_hash for value, _crosswalks in matching_cells
        ),
        "matching_uncertainty_grid_keys": tuple(
            value.grid_key for value, _crosswalks in matching_cells
        ),
        "matching_structural_row_hashes": tuple(dict.fromkeys(row_hashes)),
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7ActionPairAliasSourceJoinReceipt:
    """Exact identity proof from atomic pair issuance into alias-map ancestry."""

    registration_receipt_hash: str
    pair_hash: str
    alias_map_receipt_hash: str
    source_image_receipt_hash: str
    complete_crosswalk_receipt_hash: str
    qualification_hash: str
    uncertainty_set_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    resource_coordinate_hash: str
    coordinate_ordinal: int
    branch_crosswalk_hashes: tuple[str, str]
    matching_registered_cell_hashes: tuple[str, ...]
    matching_uncertainty_grid_keys: tuple[tuple[str, str, str, str], ...]
    matching_structural_row_hashes: tuple[str, ...]
    _registration: V1SampledV7ActionPairSourceRegistrationReceipt = field(
        repr=False, compare=False
    )
    _alias_map: V1PhaseACompleteRegisteredAliasMapReceipt = field(
        repr=False, compare=False
    )
    _issuer_token: InitVar[object]
    _integrity_seal: str = field(init=False, repr=False, compare=False)
    exact_crosswalk_object_identity_joined: bool = True
    complete_uncertainty_graph_replayed: bool = True
    array_free_public_receipt: bool = True
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_ACTION_PAIR_SOURCE_JOIN_AUTHORITY
    schema: str = SAMPLED_V7_ACTION_PAIR_ALIAS_SOURCE_JOIN_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _JOIN_TOKEN:
            raise TypeError("action-pair alias source joins are issuer-only")
        if (
            type(self._registration)
            is not V1SampledV7ActionPairSourceRegistrationReceipt
            or type(self._alias_map)
            is not V1PhaseACompleteRegisteredAliasMapReceipt
        ):
            _refuse(
                V1SampledV7ActionPairSourceJoinRefusalCode.INVALID_TYPE,
                "source join requires exact registration and alias-map parents",
            )
        observed = _replay_join(
            registration=self._registration,
            alias_map=self._alias_map,
        )
        if any(getattr(self, name) != value for name, value in observed.items()):
            _refuse(
                V1SampledV7ActionPairSourceJoinRefusalCode.STALE_SOURCE,
                "source-join public image differs from its exact parents",
            )
        _closed_authority(
            (
                self.d_stability_authority,
                self.c_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.performance_authority,
                self.pareto_authority,
            ),
            owner="action-pair alias source join",
        )
        if (
            self.exact_crosswalk_object_identity_joined is not True
            or self.complete_uncertainty_graph_replayed is not True
            or self.array_free_public_receipt is not True
            or self.authority_scope != SAMPLED_V7_ACTION_PAIR_SOURCE_JOIN_AUTHORITY
            or self.schema != SAMPLED_V7_ACTION_PAIR_ALIAS_SOURCE_JOIN_SCHEMA
        ):
            _refuse(
                V1SampledV7ActionPairSourceJoinRefusalCode.AUTHORITY_VIOLATION,
                "source-join semantics changed",
            )
        object.__setattr__(self, "_integrity_seal", self.receipt_hash)

    @property
    def action_pair(self) -> V1SampledV7BranchActionPair:
        self.assert_integrity()
        return self._registration._action_pair

    @property
    def alias_map(self) -> V1PhaseACompleteRegisteredAliasMapReceipt:
        self.assert_integrity()
        return self._alias_map

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "registration_receipt_hash": self.registration_receipt_hash,
                "pair_hash": self.pair_hash,
                "alias_map_receipt_hash": self.alias_map_receipt_hash,
                "source_image_receipt_hash": self.source_image_receipt_hash,
                "complete_crosswalk_receipt_hash": self.complete_crosswalk_receipt_hash,
                "qualification_hash": self.qualification_hash,
                "uncertainty_set_hash": self.uncertainty_set_hash,
                "source_registration_hash": self.source_registration_hash,
                "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
                "resource_coordinate_hash": self.resource_coordinate_hash,
                "coordinate_ordinal": self.coordinate_ordinal,
                "branch_crosswalk_hashes": self.branch_crosswalk_hashes,
                "matching_registered_cell_hashes": self.matching_registered_cell_hashes,
                "matching_uncertainty_grid_keys": self.matching_uncertainty_grid_keys,
                "matching_structural_row_hashes": self.matching_structural_row_hashes,
                "exact_crosswalk_object_identity_joined": True,
                "complete_uncertainty_graph_replayed": True,
                "array_free_public_receipt": True,
                "all_downstream_authority": False,
                "authority_scope": self.authority_scope,
            },
            prefix=_PREFIX + "joined-",
        )

    def assert_integrity(self) -> str:
        observed = _replay_join(
            registration=self._registration,
            alias_map=self._alias_map,
        )
        if (
            any(getattr(self, name) != value for name, value in observed.items())
            or self.receipt_hash != self._integrity_seal
        ):
            _refuse(
                V1SampledV7ActionPairSourceJoinRefusalCode.STALE_SOURCE,
                "action-pair alias source join no longer replays",
            )
        return self.receipt_hash

    def __reduce__(self) -> NoReturn:
        raise TypeError("action-pair alias source joins are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("action-pair alias source joins are nonserializable")


def bind_v1_sampled_v7_action_pair_alias_source_join(
    *,
    registration: V1SampledV7ActionPairSourceRegistrationReceipt,
    alias_map: V1PhaseACompleteRegisteredAliasMapReceipt,
) -> V1SampledV7ActionPairAliasSourceJoinReceipt:
    """Join an atomically registered pair to an exact alias-map source graph."""

    if (
        type(registration)
        is not V1SampledV7ActionPairSourceRegistrationReceipt
        or type(alias_map) is not V1PhaseACompleteRegisteredAliasMapReceipt
    ):
        _refuse(
            V1SampledV7ActionPairSourceJoinRefusalCode.INVALID_TYPE,
            "source join requires exact registration and alias-map receipts",
        )
    public = _replay_join(registration=registration, alias_map=alias_map)
    return V1SampledV7ActionPairAliasSourceJoinReceipt(
        **public,
        _registration=registration,
        _alias_map=alias_map,
        _issuer_token=_JOIN_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_ACTION_PAIR_ALIAS_SOURCE_JOIN_SCHEMA",
    "SAMPLED_V7_ACTION_PAIR_SOURCE_JOIN_AUTHORITY",
    "SAMPLED_V7_ACTION_PAIR_SOURCE_REGISTRATION_SCHEMA",
    "V1SampledV7ActionPairAliasSourceJoinReceipt",
    "V1SampledV7ActionPairSourceJoinRefusal",
    "V1SampledV7ActionPairSourceJoinRefusalCode",
    "V1SampledV7ActionPairSourceRegistrationReceipt",
    "bind_v1_sampled_v7_action_pair_alias_source_join",
    "issue_v1_sampled_v7_action_pair_source_registration",
]
