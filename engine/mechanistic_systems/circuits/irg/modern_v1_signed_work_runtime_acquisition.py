"""Controller carrier for canonical signed-work six-role v1 proposals.

The six-role C certifier is the authority seam for the modern signed-work
trajectory: native-fiber F is objective evidence, direct finite displacement D
admits the endpoint to the response budget, and Q/R labels old-chart transport
fidelity.  This module copies exactly one source-replayed, direct-D-certified,
pre-frozen packed action into the controller-facing acquisition interface.

It grants proposal authority only.  Candidate calibration, paired BCE oracle,
mask commit, terminal validation, and Pareto claims remain outside this type.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Final

from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .contracts import content_hash
from .modern_v1_campaign_contract import V1OptionACampaignBinding
from .modern_v1_candidate_composition import (
    V1PackedRemovalAction,
    V1RemovedRouteSetIdentity,
)
from .modern_v1_device_shell_ladder import V1DeviceShellFrozenActionReceipt
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_label_native_direction_provider import (
    V1LabelNativePackedDirectionReceipt,
)
from .modern_v1_signed_work_six_role_c_acquisition import (
    V1SignedWorkSixRoleCDesign,
    V1SignedWorkSixRoleProgressReceipt,
)
from .modern_v1_signed_work_six_role_certification import (
    SIGNED_WORK_SIX_ROLE_AUTHORITY_SCOPE,
    V1SignedWorkChartTransportStatus,
    V1SignedWorkIntervalStatus,
    V1SignedWorkSixRoleCertificationReceipt,
    V1SignedWorkSixRoleDecision,
    V1SignedWorkSixRoleDecisionDesign,
    V1SignedWorkSixRoleEndpointCertificate,
    V1SignedWorkSixRoleEndpointCoordinate,
    V1SignedWorkSixRoleFallbackQueueAuthorization,
    V1SignedWorkSixRoleProposalAuthorization,
    authorize_v1_signed_work_six_role_proposal,
    authorize_v1_signed_work_six_role_pre_O_fallback_queue,
)


SIGNED_WORK_ACTION_NUMERIC_STATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_six_role_action_numeric_state_v1"
)
SIGNED_WORK_CANDIDATE_PROPOSAL_SCHEMA: Final = (
    "partition_free_v1_signed_work_six_role_candidate_proposal_v1"
)
SIGNED_WORK_CANDIDATE_COMPOSITION_SCHEMA: Final = (
    "partition_free_v1_signed_work_six_role_candidate_composition_result_v1"
)
SIGNED_WORK_ACQUISITION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_six_role_acquisition_generation_receipt_v1"
)
SIGNED_WORK_AUTHORIZED_ACQUISITION_SCHEMA: Final = (
    "partition_free_v1_signed_work_six_role_authorized_acquisition_v1"
)
SIGNED_WORK_CANDIDATE_COMPOSITION_V2_SCHEMA: Final = (
    "partition_free_v1_signed_work_six_role_candidate_composition_result_v2"
)
SIGNED_WORK_ACQUISITION_RECEIPT_V2_SCHEMA: Final = (
    "partition_free_v1_signed_work_six_role_acquisition_generation_receipt_v2"
)
SIGNED_WORK_AUTHORIZED_ACQUISITION_V2_SCHEMA: Final = (
    "partition_free_v1_signed_work_six_role_authorized_acquisition_v2"
)
SIGNED_WORK_DEFERRED_BRANCH_SOURCE_SCHEMA: Final = (
    "partition_free_v1_signed_work_deferred_branch_source_v1"
)
SIGNED_WORK_COMPACT_FALLBACK_DESCRIPTOR_SCHEMA: Final = (
    "partition_free_v1_signed_work_compact_fallback_descriptor_v1"
)
SIGNED_WORK_DEFERRED_FALLBACK_PLAN_SCHEMA: Final = (
    "partition_free_v1_signed_work_deferred_fallback_plan_v1"
)
SIGNED_WORK_SINGLETON_COMPOSER_ID: Final = (
    "canonical_signed_work_six_role_direct_D_single_endpoint_queue_v1"
)
SIGNED_WORK_PRE_O_FALLBACK_COMPOSER_ID: Final = (
    "canonical_signed_work_six_role_direct_D_pre_O_fallback_queue_v1"
)
SIGNED_WORK_RUNTIME_AUTHORITY_SCOPE: Final = (
    "canonical_prefrozen_signed_work_six_role_direct_D_single_proposal_only_"
    "paired_bce_required_no_calibration_oracle_commit_terminal_or_pareto_"
    "authority_v1"
)
SIGNED_WORK_RUNTIME_FALLBACK_AUTHORITY_SCOPE: Final = (
    "canonical_prefrozen_signed_work_six_role_direct_D_pre_O_fallback_queue_"
    "paired_bce_required_no_calibration_oracle_commit_terminal_or_pareto_"
    "authority_v1"
)

_STATE_TOKEN: Final = object()
_PROPOSAL_TOKEN: Final = object()
_COMPOSITION_TOKEN: Final = object()
_COMPACT_DESCRIPTOR_TOKEN: Final = object()
_DEFERRED_PLAN_TOKEN: Final = object()


def _identity(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_positive(value: object, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _freeze(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{name} keys must be nonempty strings")
            frozen[key] = _freeze(item, f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, f"{name}[{index}]") for index, item in enumerate(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a nonfinite float")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{name} contains unsupported value {type(value)!r}")


def _fresh_mask_hash(mask: HardMask, name: str) -> str:
    if type(mask) is not HardMask:
        raise TypeError(f"{name} must be an exact HardMask")
    packed = mask.packed_bits
    if type(packed) is not bytes or len(packed) != (mask.route_count + 7) // 8:
        raise ValueError(f"{name} has an invalid packed representation")
    padding = len(packed) * 8 - mask.route_count
    if padding and packed[-1] & ((1 << padding) - 1):
        raise ValueError(f"{name} contains nonzero padding bits")
    return content_hash(
        {
            "route_axis_hash": mask.route_axis_hash,
            "route_count": mask.route_count,
            "packed_bits": packed,
        },
        prefix="irg-mask-",
    )


def _branch_value(value: object, name: str) -> str:
    raw = getattr(value, "value", value)
    if raw not in {"spectral", "diagonal_l1"}:
        raise ValueError(f"{name} is not a registered relaxed-KKT branch")
    return str(raw)


def _interval_pair(value: object, name: str) -> tuple[float, float]:
    if type(value) not in {tuple, list} or len(value) != 2:
        raise TypeError(f"{name} must be one lower/upper pair")
    lower = _finite_nonnegative(value[0], f"{name} lower")
    upper = _finite_nonnegative(value[1], f"{name} upper")
    if lower > upper:
        raise ValueError(f"{name} is inverted")
    return lower, upper


def _status_value(value: object, name: str) -> str:
    raw = getattr(value, "value", value)
    if raw not in {"certified", "refuted", "unresolved"}:
        raise ValueError(f"{name} is not a six-role interval status")
    return str(raw)


def _chart_status_value(value: object, name: str) -> str:
    raw = getattr(value, "value", value)
    if raw not in {"chart_certified", "chart_refuted", "chart_unresolved"}:
        raise ValueError(f"{name} is not a six-role chart status")
    return str(raw)


def _six_role_interval_payload(
    *,
    state: "V1SignedWorkActionNumericState | None" = None,
    values: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = values if values is not None else {}

    def item(name: str) -> object:
        return source[name] if values is not None else getattr(state, name)  # type: ignore[arg-type]

    return {
        "schema": "partition_free_v1_signed_work_six_role_interval_v1",
        "six_role_authorization_hash": item("six_role_authorization_hash"),
        "selected_coordinate_hash": item("selected_coordinate_hash"),
        "packed_action_hash": item("packed_action_hash"),
        "representative_branch": item("representative_branch"),
        "representative_endpoint_index": item("representative_endpoint_index"),
        "f_native_interval": item("f_native_interval"),
        "q_j_interval": item("q_j_interval"),
        "r_half_interval": item("r_half_interval"),
        "r_one_interval": item("r_one_interval"),
        "d_half_interval": item("d_half_interval"),
        "d_one_interval": item("d_one_interval"),
        "chart_half_interval": item("chart_half_interval"),
        "chart_one_interval": item("chart_one_interval"),
        "tau_l": item("tau_l"),
        "direct_D_status": item("direct_D_status"),
        "chart_transport_status": item("chart_transport_status"),
        "admission_basis": item("admission_basis"),
        "direct_D_owns_finite_response_admission": True,
        "QR_transport_is_label_not_hidden_C_veto": True,
        "signed_F_used_as_feasibility_veto": False,
    }


def _resource_payload(
    *,
    selected_coordinate_hash: str,
    packed_action_hash: str,
    parent_mask_hash: str,
    resulting_mask_hash: str,
    registered_resource: float,
    resource_units: str,
    resource_source_hash: str,
) -> dict[str, object]:
    return {
        "schema": "partition_free_v1_signed_work_six_role_resource_receipt_v1",
        "selected_coordinate_hash": selected_coordinate_hash,
        "packed_action_hash": packed_action_hash,
        "parent_mask_hash": parent_mask_hash,
        "resulting_mask_hash": resulting_mask_hash,
        "registered_resource": registered_resource,
        "resource_units": resource_units,
        "resource_source_hash": resource_source_hash,
    }


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeferredBranchSource:
    """Receipt-only source binding for one lazy fallback branch.

    This object deliberately accepts the immutable label-native provider
    *receipt*, never the provider, its label handle, or a direction packet.
    The receipt closes the prefix query to the canonical device-shell action
    axis while remaining small enough to persist as source provenance.
    """

    branch: V1RelaxedKKTBranch
    packed_direction_receipt: V1LabelNativePackedDirectionReceipt = field(
        repr=False,
        compare=False,
    )
    schema: str = SIGNED_WORK_DEFERRED_BRANCH_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("deferred branch source requires an exact KKT branch")
        receipt = self.packed_direction_receipt
        if type(receipt) is not V1LabelNativePackedDirectionReceipt:
            raise TypeError(
                "deferred branch source requires an exact label-native receipt"
            )
        source = receipt.action_axis_registration.source_shell_ladder
        if (
            receipt.endpoint_hard_masks_materialized
            or receipt.full_prefix_ranking_copied_to_host
            or receipt.candidate_composition_recomputed
            or receipt.fqr_authority
            or receipt.calibration_authority
            or receipt.oracle_authority
            or receipt.mask_commit_authority
            or receipt.terminal_authority
            or receipt.pareto_authority
            or receipt.action_axis_registration.endpoint_action_hashes
            != tuple(value.packed_action_hash for value in source.actions)
            or self.schema != SIGNED_WORK_DEFERRED_BRANCH_SOURCE_SCHEMA
        ):
            raise ValueError(
                "deferred branch source is materialized, post-selected, or overbroad"
            )

    @property
    def source_commitment_hash(self) -> str:
        receipt = self.packed_direction_receipt
        source = receipt.action_axis_registration.source_shell_ladder
        return content_hash(
            {
                "schema": self.schema,
                "branch": self.branch.value,
                "provider_receipt_hash": receipt.receipt_hash,
                "provider_design_hash": receipt.design_hash,
                "provider_policy_hash": receipt.policy_hash,
                "prefix_rank_receipt_hash": receipt.prefix_rank_receipt_hash,
                "endpoint_query_hash": receipt.endpoint_query_hash,
                "resulting_label_commitment": receipt.resulting_label_commitment,
                "action_axis_registration_hash": (
                    receipt.action_axis_registration.registration_hash
                ),
                "source_shell_ladder_receipt_hash": source.receipt_hash,
                "source_ladder_hash": source.source_ladder_hash,
                "source_action_receipt_hashes": tuple(
                    value.receipt_hash for value in source.actions
                ),
            },
            prefix="partition-free-v1-signed-work-deferred-branch-source-",
        )


@dataclass(frozen=True, slots=True, init=False)
class V1SignedWorkCompactFallbackDescriptor:
    """One authorized fallback rung without a resident or packed child mask."""

    queue_index: int
    coordinate_hash: str
    physical_candidate_identity_hash: str
    pre_C_priority: int
    branch: V1RelaxedKKTBranch
    endpoint_index: int
    endpoint_label: int
    certificate_hash: str
    direct_D_status: str
    admission_basis: str
    chart_transport_status: str
    old_chart_validity_claim: bool
    f_native_interval: tuple[float, float]
    q_j_interval: tuple[float, float]
    r_half_interval: tuple[float, float]
    r_one_interval: tuple[float, float]
    d_half_interval: tuple[float, float]
    d_one_interval: tuple[float, float]
    chart_half_interval: tuple[float, float]
    chart_one_interval: tuple[float, float]
    tau_l: float
    six_role_interval_hash: str
    frozen_action_hash: str
    crosswalk_receipt_hash: str
    crosswalk_materialization_receipt_hash: str
    canonical_candidate_occurrence: int
    cross_branch_exact_deduplication: bool
    hard_replay_endpoint_receipt_hash: str
    source_action_receipt: V1DeviceShellFrozenActionReceipt = field(
        repr=False,
        compare=False,
    )
    source_action_receipt_hash: str
    shell_count: int
    cumulative_removed_count: int
    packed_action_hash: str
    resulting_mask_packed_sha256: str
    resulting_mask_hash: str
    removed_route_index_hash: str
    branch_bridge_receipt_hash: str
    bridge_shell_execution_hash: str
    bridge_shell_execution_receipt_hash: str
    bridge_shell_branch_identity_receipt_hash: str
    bridge_hard_replay_branch_receipt_hash: str
    bridge_coordinates_hash: str
    bridge_design_hash: str
    provider_source_commitment_hash: str
    provider_receipt_hash: str
    provider_design_hash: str
    provider_policy_hash: str
    provider_forecast_hash: str
    provider_backend_kind: str
    provider_device_identity: str
    action_axis_registration_hash: str
    prefix_rank_receipt_hash: str
    endpoint_query_hash: str
    resulting_label_commitment: str
    source_shell_ladder_receipt_hash: str
    source_ladder_hash: str
    source_label_commitment: str
    source_eligibility_commitment: str
    source_block_capacity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    six_role_fallback_authorization_hash: str
    six_role_certification_receipt_hash: str
    six_role_decision_hash: str
    six_role_decision_design_hash: str
    six_role_C_design_hash: str
    six_role_C_progress_receipt_hash: str
    bridge_family_bundle_hash: str
    deferred_one_endpoint_materialization_required: bool
    complete_ladder_mask_materialization_forbidden: bool
    paired_bce_oracle_required: bool
    calibration_K_open_authority: bool
    paired_oracle_O_open_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    schema: str = SIGNED_WORK_COMPACT_FALLBACK_DESCRIPTOR_SCHEMA
    _descriptor_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _COMPACT_DESCRIPTOR_TOKEN:
            raise TypeError("compact fallback descriptors are issuer-only")
        expected = tuple(
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        if set(values) != set(expected):
            raise TypeError(
                "compact fallback descriptor fields differ: "
                f"missing={sorted(set(expected).difference(values))}, "
                f"extra={sorted(set(values).difference(expected))}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(
            self, "schema", SIGNED_WORK_COMPACT_FALLBACK_DESCRIPTOR_SCHEMA
        )
        object.__setattr__(self, "_descriptor_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("compact fallback descriptor requires an exact branch")
        if type(self.source_action_receipt) is not V1DeviceShellFrozenActionReceipt:
            raise TypeError("compact fallback descriptor requires an exact action receipt")
        for name in (
            "queue_index",
            "proposal_generation",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )
        for name in (
            "pre_C_priority",
            "endpoint_index",
            "endpoint_label",
            "canonical_candidate_occurrence",
            "shell_count",
            "cumulative_removed_count",
            "route_count",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in (
            "coordinate_hash",
            "physical_candidate_identity_hash",
            "certificate_hash",
            "six_role_interval_hash",
            "frozen_action_hash",
            "crosswalk_receipt_hash",
            "crosswalk_materialization_receipt_hash",
            "hard_replay_endpoint_receipt_hash",
            "source_action_receipt_hash",
            "packed_action_hash",
            "resulting_mask_packed_sha256",
            "resulting_mask_hash",
            "removed_route_index_hash",
            "branch_bridge_receipt_hash",
            "bridge_shell_execution_hash",
            "bridge_shell_execution_receipt_hash",
            "bridge_shell_branch_identity_receipt_hash",
            "bridge_hard_replay_branch_receipt_hash",
            "bridge_coordinates_hash",
            "bridge_design_hash",
            "provider_source_commitment_hash",
            "provider_receipt_hash",
            "provider_design_hash",
            "provider_policy_hash",
            "provider_forecast_hash",
            "provider_backend_kind",
            "provider_device_identity",
            "action_axis_registration_hash",
            "prefix_rank_receipt_hash",
            "endpoint_query_hash",
            "resulting_label_commitment",
            "source_shell_ladder_receipt_hash",
            "source_ladder_hash",
            "source_label_commitment",
            "source_eligibility_commitment",
            "source_block_capacity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "six_role_fallback_authorization_hash",
            "six_role_certification_receipt_hash",
            "six_role_decision_hash",
            "six_role_decision_design_hash",
            "six_role_C_design_hash",
            "six_role_C_progress_receipt_hash",
            "bridge_family_bundle_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        intervals = (
            "f_native_interval",
            "q_j_interval",
            "r_half_interval",
            "r_one_interval",
            "d_half_interval",
            "d_one_interval",
            "chart_half_interval",
            "chart_one_interval",
        )
        for name in intervals:
            object.__setattr__(self, name, _interval_pair(getattr(self, name), name))
        object.__setattr__(self, "tau_l", _finite_positive(self.tau_l, "tau_l"))
        object.__setattr__(
            self, "direct_D_status", _status_value(self.direct_D_status, "direct_D_status")
        )
        object.__setattr__(
            self,
            "chart_transport_status",
            _chart_status_value(self.chart_transport_status, "chart_transport_status"),
        )
        if self.direct_D_status != V1SignedWorkIntervalStatus.CERTIFIED.value:
            raise ValueError("compact fallback descriptor is not direct-D certified")
        if self.admission_basis not in {"chart_certified", "direct_D_exactified"}:
            raise ValueError("compact fallback descriptor lacks an admission basis")
        expected_chart_claim = self.admission_basis == "chart_certified"
        if (
            type(self.old_chart_validity_claim) is not bool
            or type(self.cross_branch_exact_deduplication) is not bool
            or self.old_chart_validity_claim is not expected_chart_claim
            or (
                expected_chart_claim
                and self.chart_transport_status
                != V1SignedWorkChartTransportStatus.CHART_CERTIFIED.value
            )
        ):
            raise ValueError("compact fallback descriptor overclaimed chart validity")
        action = self.source_action_receipt
        if (
            action.receipt_hash != self.source_action_receipt_hash
            or action.endpoint_label != self.endpoint_label
            or action.shell_count != self.shell_count
            or action.cumulative_removed_count != self.cumulative_removed_count
            or action.packed_action_hash != self.packed_action_hash
            or action.resulting_mask_packed_sha256
            != self.resulting_mask_packed_sha256
            or action.resulting_mask_hash != self.resulting_mask_hash
            or action.removed_route_index_hash != self.removed_route_index_hash
        ):
            raise ValueError("compact fallback action receipt was not copied exactly")
        expected_interval = content_hash(
            _six_role_interval_payload(
                values={
                    "six_role_authorization_hash": (
                        self.six_role_fallback_authorization_hash
                    ),
                    "selected_coordinate_hash": self.coordinate_hash,
                    "packed_action_hash": self.packed_action_hash,
                    "representative_branch": self.branch.value,
                    "representative_endpoint_index": self.endpoint_index,
                    "f_native_interval": self.f_native_interval,
                    "q_j_interval": self.q_j_interval,
                    "r_half_interval": self.r_half_interval,
                    "r_one_interval": self.r_one_interval,
                    "d_half_interval": self.d_half_interval,
                    "d_one_interval": self.d_one_interval,
                    "chart_half_interval": self.chart_half_interval,
                    "chart_one_interval": self.chart_one_interval,
                    "tau_l": self.tau_l,
                    "direct_D_status": self.direct_D_status,
                    "chart_transport_status": self.chart_transport_status,
                    "admission_basis": self.admission_basis,
                }
            ),
            prefix="partition-free-v1-signed-work-six-role-interval-",
        )
        if self.six_role_interval_hash != expected_interval:
            raise ValueError("compact fallback interval receipt is not reproducible")
        authority_flags = (
            self.deferred_one_endpoint_materialization_required,
            self.complete_ladder_mask_materialization_forbidden,
            self.paired_bce_oracle_required,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
            self.execution_authority,
        )
        if (
            any(type(value) is not bool for value in authority_flags)
            or not all(authority_flags[:3])
            or any(authority_flags[3:])
            or self.provider_backend_kind not in {"numpy_reference", "torch_cuda"}
        ):
            raise ValueError("compact fallback descriptor crossed its authority firewall")
        if self.schema != SIGNED_WORK_COMPACT_FALLBACK_DESCRIPTOR_SCHEMA:
            raise ValueError("unsupported compact fallback descriptor schema")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_descriptor_hash"
        }

    @property
    def pre_O_rung_registration_hash(self) -> str:
        return content_hash(
            {
                "schema": "partition_free_v1_signed_work_pre_O_rung_identity_v1",
                "coordinate_hash": self.coordinate_hash,
                "physical_candidate_identity_hash": (
                    self.physical_candidate_identity_hash
                ),
                "packed_action_hash": self.packed_action_hash,
                "resulting_mask_hash": self.resulting_mask_hash,
                "cumulative_removed_count": self.cumulative_removed_count,
                "certificate_hash": self.certificate_hash,
                "six_role_interval_hash": self.six_role_interval_hash,
                "current_mask_hash": self.current_mask_hash,
                "proposal_generation": self.proposal_generation,
            },
            prefix="partition-free-v1-signed-work-pre-O-rung-",
        )

    @property
    def descriptor_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(),
            prefix="partition-free-v1-signed-work-compact-fallback-descriptor-",
        )
        if self._descriptor_hash is not None and observed != self._descriptor_hash:
            raise ValueError("compact fallback descriptor was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("compact fallback descriptor hash does not match")
        object.__setattr__(self, "_descriptor_hash", observed)
        return observed


@dataclass(frozen=True, slots=True, init=False)
class V1SignedWorkDeferredFallbackPlan:
    """Frozen resolved-prefix plan whose hard endpoints remain deferred."""

    campaign_contract_hash: str
    campaign_binding_hash: str
    registration_hash: str
    six_role_fallback_authorization_hash: str
    six_role_certification_receipt_hash: str
    six_role_decision_hash: str
    six_role_decision_design_hash: str
    six_role_C_design_hash: str
    six_role_C_progress_receipt_hash: str
    bridge_family_bundle_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    selected_coordinate_hash: str
    descriptors: tuple[V1SignedWorkCompactFallbackDescriptor, ...]
    ordered_descriptor_hashes: tuple[str, ...]
    ordered_coordinate_hashes: tuple[str, ...]
    ordered_physical_candidate_identity_hashes: tuple[str, ...]
    ordered_pre_O_rung_registration_hashes: tuple[str, ...]
    source_branch_commitment_hashes: tuple[tuple[str, str], ...]
    blocked_unresolved_coordinate_hash: str | None
    fallback_exhaustion_status: str
    authorized_descriptor_count: int
    resolved_prefix_descriptors_frozen_before_K_O: bool
    source_ladder_decision_fully_resolved_before_K_O: bool
    blocked_on_more_C: bool
    fully_resolved_no_more_candidate: bool
    deferred_one_endpoint_materialization_required: bool
    complete_ladder_mask_materialization_forbidden: bool
    endpoint_hard_masks_embedded: bool
    packed_child_payloads_embedded: bool
    provider_or_label_handle_embedded: bool
    source_authorized_filtering_performed: bool
    reordering_performed: bool
    omitted_mask_optimality_authority: bool
    paired_bce_oracle_required: bool
    calibration_K_open_authority: bool
    paired_oracle_O_open_authority: bool
    irreversible_decision_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    schema: str = SIGNED_WORK_DEFERRED_FALLBACK_PLAN_SCHEMA
    _plan_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _DEFERRED_PLAN_TOKEN:
            raise TypeError("deferred fallback plans are issuer-only")
        expected = tuple(
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        if set(values) != set(expected):
            raise TypeError(
                "deferred fallback plan fields differ: "
                f"missing={sorted(set(expected).difference(values))}, "
                f"extra={sorted(set(values).difference(expected))}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", SIGNED_WORK_DEFERRED_FALLBACK_PLAN_SCHEMA)
        object.__setattr__(self, "_plan_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "registration_hash",
            "six_role_fallback_authorization_hash",
            "six_role_certification_receipt_hash",
            "six_role_decision_hash",
            "six_role_decision_design_hash",
            "six_role_C_design_hash",
            "six_role_C_progress_receipt_hash",
            "bridge_family_bundle_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "selected_coordinate_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        object.__setattr__(self, "route_count", _positive_int(self.route_count, "route_count"))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, "proposal_generation"),
        )
        descriptors = tuple(self.descriptors)
        if not descriptors or any(
            type(value) is not V1SignedWorkCompactFallbackDescriptor
            for value in descriptors
        ):
            raise TypeError("deferred fallback plan requires compact descriptors")
        for index, descriptor in enumerate(descriptors):
            descriptor.assert_integrity()
            if descriptor.queue_index != index:
                raise ValueError("deferred fallback descriptor indices are not contiguous")
            if (
                descriptor.six_role_fallback_authorization_hash
                != self.six_role_fallback_authorization_hash
                or descriptor.six_role_certification_receipt_hash
                != self.six_role_certification_receipt_hash
                or descriptor.six_role_decision_hash != self.six_role_decision_hash
                or descriptor.six_role_decision_design_hash
                != self.six_role_decision_design_hash
                or descriptor.six_role_C_design_hash != self.six_role_C_design_hash
                or descriptor.six_role_C_progress_receipt_hash
                != self.six_role_C_progress_receipt_hash
                or descriptor.bridge_family_bundle_hash
                != self.bridge_family_bundle_hash
                or descriptor.chart_identity_hash != self.chart_identity_hash
                or descriptor.current_mask_hash != self.current_mask_hash
                or descriptor.route_axis_hash != self.route_axis_hash
                or descriptor.route_count != self.route_count
                or descriptor.proposal_generation != self.proposal_generation
            ):
                raise ValueError("a deferred descriptor left the plan source frame")
        descriptor_hashes = tuple(value.descriptor_hash for value in descriptors)
        coordinates = tuple(value.coordinate_hash for value in descriptors)
        physical = tuple(
            value.physical_candidate_identity_hash for value in descriptors
        )
        rungs = tuple(value.pre_O_rung_registration_hash for value in descriptors)
        order = tuple(
            (-value.cumulative_removed_count, value.pre_C_priority)
            for value in descriptors
        )
        sources = tuple(
            (
                _branch_value(branch, "source branch"),
                _identity(source_hash, "source branch commitment hash"),
            )
            for branch, source_hash in self.source_branch_commitment_hashes
        )
        if (
            self.selected_coordinate_hash != coordinates[0]
            or self.ordered_descriptor_hashes != descriptor_hashes
            or self.ordered_coordinate_hashes != coordinates
            or self.ordered_physical_candidate_identity_hashes != physical
            or self.ordered_pre_O_rung_registration_hashes != rungs
            or len(set(coordinates)) != len(coordinates)
            or len(set(physical)) != len(physical)
            or order != tuple(sorted(order))
            or len(sources) != len(set(branch for branch, _value in sources))
        ):
            raise ValueError("deferred fallback plan order or identity axes differ")
        if self.blocked_unresolved_coordinate_hash is not None:
            object.__setattr__(
                self,
                "blocked_unresolved_coordinate_hash",
                _identity(
                    self.blocked_unresolved_coordinate_hash,
                    "blocked unresolved coordinate hash",
                ),
            )
        blocked = self.blocked_unresolved_coordinate_hash is not None
        fully_resolved = not blocked
        if (
            self.fallback_exhaustion_status
            not in {"blocked_on_more_C", "fully_resolved_no_more_candidate"}
            or self.fallback_exhaustion_status
            != (
                "blocked_on_more_C"
                if blocked
                else "fully_resolved_no_more_candidate"
            )
            or self.blocked_on_more_C is not blocked
            or self.fully_resolved_no_more_candidate is not fully_resolved
            or self.source_ladder_decision_fully_resolved_before_K_O
            is not fully_resolved
        ):
            raise ValueError("deferred fallback exhaustion status is contradictory")
        authority_flags = (
            self.resolved_prefix_descriptors_frozen_before_K_O,
            self.source_ladder_decision_fully_resolved_before_K_O,
            self.blocked_on_more_C,
            self.fully_resolved_no_more_candidate,
            self.deferred_one_endpoint_materialization_required,
            self.complete_ladder_mask_materialization_forbidden,
            self.endpoint_hard_masks_embedded,
            self.packed_child_payloads_embedded,
            self.provider_or_label_handle_embedded,
            self.source_authorized_filtering_performed,
            self.reordering_performed,
            self.omitted_mask_optimality_authority,
            self.paired_bce_oracle_required,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.irreversible_decision_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
            self.execution_authority,
        )
        if (
            any(type(value) is not bool for value in authority_flags)
            or not self.resolved_prefix_descriptors_frozen_before_K_O
            or not self.deferred_one_endpoint_materialization_required
            or not self.complete_ladder_mask_materialization_forbidden
            or self.endpoint_hard_masks_embedded
            or self.packed_child_payloads_embedded
            or self.provider_or_label_handle_embedded
            or not self.source_authorized_filtering_performed
            or self.reordering_performed
            or self.omitted_mask_optimality_authority
            or not self.paired_bce_oracle_required
            or self.calibration_K_open_authority
            or self.paired_oracle_O_open_authority
            or self.irreversible_decision_authority
            or self.mask_commit_authority
            or self.terminal_authority
            or self.pareto_authority
            or self.execution_authority
        ):
            raise ValueError("deferred fallback plan crossed its authority firewall")
        if (
            _positive_int(
                self.authorized_descriptor_count,
                "authorized_descriptor_count",
            )
            != len(descriptors)
            or self.schema != SIGNED_WORK_DEFERRED_FALLBACK_PLAN_SCHEMA
        ):
            raise ValueError("deferred fallback plan size or schema is invalid")
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "source_branch_commitment_hashes", sources)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: (
                tuple(value.descriptor_hash for value in self.descriptors)
                if name == "descriptors"
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
            if name != "_plan_hash"
        }

    @property
    def plan_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for descriptor in self.descriptors:
            descriptor.assert_integrity()
        observed = content_hash(
            self._payload(),
            prefix="partition-free-v1-signed-work-deferred-fallback-plan-",
        )
        if self._plan_hash is not None and observed != self._plan_hash:
            raise ValueError("deferred fallback plan was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("deferred fallback plan hash does not match")
        object.__setattr__(self, "_plan_hash", observed)
        return observed


@dataclass(frozen=True, slots=True, init=False)
class V1SignedWorkActionNumericState:
    """One direct-D-admitted packed action plus its six-role intervals."""

    queue_index: int
    representative_branch: str
    representative_endpoint_index: int
    representative_frozen_action_hash: str
    pre_C_priority: int
    endpoint_label: int
    selected_coordinate_hash: str
    source_crosswalk_receipt_hash: str
    source_hard_replay_endpoint_receipt_hash: str
    parent_mask: HardMask = field(repr=False, compare=False)
    packed_action: V1PackedRemovalAction = field(repr=False, compare=False)
    parent_mask_hash: str
    resulting_mask_hash: str
    packed_action_hash: str
    route_axis_hash: str
    route_count: int
    removed_count: int
    removed_route_index_hash: str
    registered_resource: float
    resource_units: str
    resource_source_hash: str
    resource_receipt_hash: str
    f_native_interval: tuple[float, float]
    q_j_interval: tuple[float, float]
    r_half_interval: tuple[float, float]
    r_one_interval: tuple[float, float]
    d_half_interval: tuple[float, float]
    d_one_interval: tuple[float, float]
    chart_half_interval: tuple[float, float]
    chart_one_interval: tuple[float, float]
    tau_l: float
    direct_D_status: str
    chart_transport_status: str
    admission_basis: str
    six_role_interval_hash: str
    six_role_authorization_hash: str
    six_role_certification_receipt_hash: str
    six_role_decision_hash: str
    six_role_decision_design_hash: str
    six_role_C_design_hash: str
    six_role_C_progress_receipt_hash: str
    bridge_family_bundle_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    prefrozen_ladder_hash: str
    generation_index: int
    single_endpoint_proposal_authority: bool
    source_replayed_six_role_authorization: bool
    direct_D_owns_finite_response_admission: bool
    QR_transport_is_label_not_hidden_C_veto: bool
    old_chart_validity_claim: bool
    relinearization_required_after_oracle_accept: bool
    proposal_only: bool
    paired_bce_oracle_required: bool
    calibration_K_open_authority: bool
    paired_oracle_O_open_authority: bool
    candidate_freeze_authority: bool
    bce_oracle_authority: bool
    irreversible_decision_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    schema: str = SIGNED_WORK_ACTION_NUMERIC_STATE_SCHEMA
    _state_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _STATE_TOKEN:
            raise TypeError("V1SignedWorkActionNumericState is issuer-only")
        expected = tuple(
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        if set(values) != set(expected):
            missing = sorted(set(expected).difference(values))
            extra = sorted(set(values).difference(expected))
            raise TypeError(
                "signed-work action-state fields differ: "
                f"missing={missing}, extra={extra}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", SIGNED_WORK_ACTION_NUMERIC_STATE_SCHEMA)
        object.__setattr__(self, "_state_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.parent_mask) is not HardMask:
            raise TypeError("signed-work state requires an exact parent mask")
        if type(self.packed_action) is not V1PackedRemovalAction:
            raise TypeError("signed-work state requires an exact packed action")
        parent_hash = _fresh_mask_hash(self.parent_mask, "signed-work parent mask")
        child = self.packed_action.resulting_mask
        child_hash = _fresh_mask_hash(child, "signed-work resulting mask")
        self.parent_mask.assert_same_axis(child)
        removed_count, removed_hash = child.removed_index_count_and_hash_from(
            self.parent_mask
        )
        object.__setattr__(
            self, "queue_index", _nonnegative_int(self.queue_index, "queue_index")
        )
        object.__setattr__(
            self,
            "representative_endpoint_index",
            _positive_int(
                self.representative_endpoint_index,
                "representative_endpoint_index",
            ),
        )
        object.__setattr__(
            self, "pre_C_priority", _positive_int(self.pre_C_priority, "pre_C_priority")
        )
        object.__setattr__(
            self, "endpoint_label", _positive_int(self.endpoint_label, "endpoint_label")
        )
        object.__setattr__(
            self, "generation_index", _nonnegative_int(self.generation_index, "generation_index")
        )
        object.__setattr__(self, "route_count", _positive_int(self.route_count, "route_count"))
        object.__setattr__(
            self, "removed_count", _positive_int(self.removed_count, "removed_count")
        )
        object.__setattr__(
            self,
            "representative_branch",
            _branch_value(self.representative_branch, "representative_branch"),
        )
        for name in (
            "representative_frozen_action_hash",
            "selected_coordinate_hash",
            "source_crosswalk_receipt_hash",
            "source_hard_replay_endpoint_receipt_hash",
            "parent_mask_hash",
            "resulting_mask_hash",
            "packed_action_hash",
            "route_axis_hash",
            "removed_route_index_hash",
            "resource_units",
            "resource_source_hash",
            "resource_receipt_hash",
            "six_role_interval_hash",
            "six_role_authorization_hash",
            "six_role_certification_receipt_hash",
            "six_role_decision_hash",
            "six_role_decision_design_hash",
            "six_role_C_design_hash",
            "six_role_C_progress_receipt_hash",
            "bridge_family_bundle_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "prefrozen_ladder_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        if (
            self.parent_mask_hash != parent_hash
            or self.resulting_mask_hash != child_hash
            or self.packed_action_hash != self.packed_action.action_hash
            or self.route_axis_hash != self.parent_mask.route_axis_hash
            or self.route_count != self.parent_mask.route_count
            or self.removed_count != removed_count
            or self.removed_count != self.packed_action.removed_count
            or self.removed_route_index_hash != removed_hash
            or self.removed_route_index_hash != self.packed_action.removed_route_index_hash
        ):
            raise ValueError("signed-work action identity differs from its exact masks")
        intervals = {
            name: _interval_pair(getattr(self, name), name)
            for name in (
                "f_native_interval",
                "q_j_interval",
                "r_half_interval",
                "r_one_interval",
                "d_half_interval",
                "d_one_interval",
                "chart_half_interval",
                "chart_one_interval",
            )
        }
        for name, value in intervals.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "tau_l", _finite_positive(self.tau_l, "tau_l"))
        object.__setattr__(
            self, "direct_D_status", _status_value(self.direct_D_status, "direct_D_status")
        )
        object.__setattr__(
            self,
            "chart_transport_status",
            _chart_status_value(self.chart_transport_status, "chart_transport_status"),
        )
        if self.direct_D_status != V1SignedWorkIntervalStatus.CERTIFIED.value:
            raise ValueError("signed-work runtime action must be direct-D certified")
        if self.admission_basis not in {"chart_certified", "direct_D_exactified"}:
            raise ValueError("signed-work runtime action lacks a valid admission basis")
        expected_chart_claim = self.admission_basis == "chart_certified"
        if (
            self.old_chart_validity_claim is not expected_chart_claim
            or (
                self.admission_basis == "chart_certified"
                and self.chart_transport_status
                != V1SignedWorkChartTransportStatus.CHART_CERTIFIED.value
            )
        ):
            raise ValueError("signed-work runtime action overclaimed chart validity")
        resource = _finite_nonnegative(self.registered_resource, "registered_resource")
        object.__setattr__(self, "registered_resource", resource)
        expected_interval = content_hash(
            _six_role_interval_payload(state=self),
            prefix="partition-free-v1-signed-work-six-role-interval-",
        )
        if self.six_role_interval_hash != expected_interval:
            raise ValueError("signed-work six-role interval hash is not reproducible")
        expected_resource = content_hash(
            _resource_payload(
                selected_coordinate_hash=self.selected_coordinate_hash,
                packed_action_hash=self.packed_action_hash,
                parent_mask_hash=self.parent_mask_hash,
                resulting_mask_hash=self.resulting_mask_hash,
                registered_resource=resource,
                resource_units=self.resource_units,
                resource_source_hash=self.resource_source_hash,
            ),
            prefix="partition-free-v1-signed-work-resource-",
        )
        if self.resource_receipt_hash != expected_resource:
            raise ValueError("signed-work resource receipt is not reproducible")
        firewall = (
            self.single_endpoint_proposal_authority,
            self.source_replayed_six_role_authorization,
            self.direct_D_owns_finite_response_admission,
            self.QR_transport_is_label_not_hidden_C_veto,
            self.relinearization_required_after_oracle_accept,
            self.proposal_only,
            self.paired_bce_oracle_required,
            not self.calibration_K_open_authority,
            not self.paired_oracle_O_open_authority,
            not self.candidate_freeze_authority,
            not self.bce_oracle_authority,
            not self.irreversible_decision_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.execution_authority,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("signed-work state crossed its proposal-only firewall")
        if self.schema != SIGNED_WORK_ACTION_NUMERIC_STATE_SCHEMA:
            raise ValueError("unsupported signed-work action-state schema")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "queue_index": self.queue_index,
            "representative_branch": self.representative_branch,
            "representative_endpoint_index": self.representative_endpoint_index,
            "representative_frozen_action_hash": self.representative_frozen_action_hash,
            "pre_C_priority": self.pre_C_priority,
            "endpoint_label": self.endpoint_label,
            "selected_coordinate_hash": self.selected_coordinate_hash,
            "source_crosswalk_receipt_hash": self.source_crosswalk_receipt_hash,
            "source_hard_replay_endpoint_receipt_hash": (
                self.source_hard_replay_endpoint_receipt_hash
            ),
            "parent_mask_hash": self.parent_mask_hash,
            "resulting_mask_hash": self.resulting_mask_hash,
            "packed_action_hash": self.packed_action_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "removed_count": self.removed_count,
            "removed_route_index_hash": self.removed_route_index_hash,
            "registered_resource": self.registered_resource,
            "resource_units": self.resource_units,
            "resource_source_hash": self.resource_source_hash,
            "resource_receipt_hash": self.resource_receipt_hash,
            "six_role_interval_hash": self.six_role_interval_hash,
            "six_role_intervals": _six_role_interval_payload(state=self),
            "six_role_authorization_hash": self.six_role_authorization_hash,
            "six_role_certification_receipt_hash": (
                self.six_role_certification_receipt_hash
            ),
            "six_role_decision_hash": self.six_role_decision_hash,
            "six_role_decision_design_hash": self.six_role_decision_design_hash,
            "six_role_C_design_hash": self.six_role_C_design_hash,
            "six_role_C_progress_receipt_hash": self.six_role_C_progress_receipt_hash,
            "bridge_family_bundle_hash": self.bridge_family_bundle_hash,
            "native_fiber_axis_receipt_hash": self.native_fiber_axis_receipt_hash,
            "native_fiber_measure_hash": self.native_fiber_measure_hash,
            "prefrozen_ladder_hash": self.prefrozen_ladder_hash,
            "generation_index": self.generation_index,
            "single_endpoint_proposal_authority": self.single_endpoint_proposal_authority,
            "source_replayed_six_role_authorization": (
                self.source_replayed_six_role_authorization
            ),
            "direct_D_owns_finite_response_admission": (
                self.direct_D_owns_finite_response_admission
            ),
            "QR_transport_is_label_not_hidden_C_veto": (
                self.QR_transport_is_label_not_hidden_C_veto
            ),
            "old_chart_validity_claim": self.old_chart_validity_claim,
            "relinearization_required_after_oracle_accept": (
                self.relinearization_required_after_oracle_accept
            ),
            "proposal_only": self.proposal_only,
            "paired_bce_oracle_required": self.paired_bce_oracle_required,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "execution_authority": self.execution_authority,
        }

    @property
    def state_hash(self) -> str:
        return self.assert_integrity()

    @property
    def numeric_state_hash(self) -> str:
        return self.state_hash

    @property
    def resulting_mask(self) -> HardMask:
        return self.packed_action.resulting_mask

    @property
    def removed_routes(self) -> V1RemovedRouteSetIdentity:
        return V1RemovedRouteSetIdentity(
            route_count=self.route_count,
            count=self.removed_count,
            index_hash=self.removed_route_index_hash,
        )

    @property
    def compact_removed_indices_available(self) -> bool:
        return False

    @property
    def f_native_lower(self) -> float:
        return self.f_native_interval[0]

    @property
    def f_native_upper(self) -> float:
        return self.f_native_interval[1]

    @property
    def q_j_lower(self) -> float:
        return self.q_j_interval[0]

    @property
    def q_j_upper(self) -> float:
        return self.q_j_interval[1]

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            _fresh_mask_hash(self.parent_mask, "signed-work parent mask")
            != self.parent_mask_hash
            or _fresh_mask_hash(self.resulting_mask, "signed-work resulting mask")
            != self.resulting_mask_hash
            or self.packed_action.action_hash != self.packed_action_hash
        ):
            raise ValueError("signed-work state masks or action were tampered")
        observed = content_hash(
            self._payload(),
            prefix="partition-free-v1-signed-work-action-state-",
        )
        if self._state_hash is not None and observed != self._state_hash:
            raise ValueError("signed-work action state was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("signed-work action-state hash does not match")
        object.__setattr__(self, "_state_hash", observed)
        return observed

    def assert_can_certify_decision(self) -> None:
        raise PermissionError("signed-work proposal states are not scientific certificates")

    def assert_can_commit_mask(self) -> None:
        raise PermissionError("signed-work proposal states cannot commit a hard mask")


@dataclass(frozen=True, slots=True, init=False)
class V1SignedWorkCandidateProposal:
    """One source-replayed six-role proposal in the immutable controller queue."""

    numeric_state: V1SignedWorkActionNumericState
    domain_hash: str
    certification_scope_hash: str
    evaluation_receipt_hash: str
    candidate_id: str
    proposal_only: bool
    proposal_witness_only: bool
    fresh_relinearization_required: bool
    paired_bce_oracle_required: bool
    candidate_freeze_authority: bool
    bce_oracle_authority: bool
    irreversible_decision_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    schema: str = SIGNED_WORK_CANDIDATE_PROPOSAL_SCHEMA

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _PROPOSAL_TOKEN:
            raise TypeError("V1SignedWorkCandidateProposal is issuer-only")
        expected = tuple(
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        if set(values) != set(expected):
            missing = sorted(set(expected).difference(values))
            extra = sorted(set(values).difference(expected))
            raise TypeError(
                "signed-work proposal fields differ: "
                f"missing={missing}, extra={extra}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", SIGNED_WORK_CANDIDATE_PROPOSAL_SCHEMA)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.numeric_state) is not V1SignedWorkActionNumericState:
            raise TypeError("signed-work proposal requires its issued numeric state")
        self.numeric_state.assert_integrity()
        for name in (
            "domain_hash",
            "certification_scope_hash",
            "evaluation_receipt_hash",
            "candidate_id",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        firewall = (
            self.proposal_only,
            self.proposal_witness_only,
            self.fresh_relinearization_required,
            self.paired_bce_oracle_required,
            not self.candidate_freeze_authority,
            not self.bce_oracle_authority,
            not self.irreversible_decision_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.execution_authority,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("signed-work proposal crossed its proposal-only firewall")
        if self.schema != SIGNED_WORK_CANDIDATE_PROPOSAL_SCHEMA:
            raise ValueError("unsupported signed-work proposal schema")
        expected_receipt = content_hash(
            self._evaluation_payload(),
            prefix="partition-free-v1-signed-work-candidate-evaluation-",
        )
        if self.evaluation_receipt_hash != expected_receipt:
            raise ValueError("signed-work evaluation receipt is not reproducible")
        expected_candidate = content_hash(
            {
                "evaluation_receipt_hash": expected_receipt,
                "resulting_mask_hash": self.resulting_mask_hash,
                "removed_route_index_hash": self.removed_route_index_hash,
                "six_role_interval_hash": self.six_role_interval_hash,
                "selected_coordinate_hash": self.selected_coordinate_hash,
            },
            prefix="partition-free-v1-signed-work-candidate-",
        )
        if self.candidate_id != expected_candidate:
            raise ValueError("signed-work candidate identity is not reproducible")

    def _evaluation_payload(self) -> dict[str, object]:
        return {
            "schema": "partition_free_v1_signed_work_candidate_evaluation_v1",
            "numeric_state_hash": self.numeric_state_hash,
            "domain_hash": self.domain_hash,
            "certification_scope_hash": self.certification_scope_hash,
            "queue_index": self.queue_index,
            "representative_branch": self.representative_branch,
            "representative_endpoint_index": self.representative_endpoint_index,
            "packed_action_hash": self.packed_action_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "resulting_mask_hash": self.resulting_mask_hash,
            "removed_count": self.removed_count,
            "removed_route_index_hash": self.removed_route_index_hash,
            "registered_resource": self.resource_gain,
            "resource_receipt_hash": self.resource_receipt_hash,
            "six_role_interval_hash": self.six_role_interval_hash,
            "six_role_authorization_hash": self.six_role_authorization_hash,
            "selected_coordinate_hash": self.selected_coordinate_hash,
            "direct_D_owns_finite_response_admission": True,
            "QR_transport_is_label_not_hidden_C_veto": True,
            "paired_bce_oracle_required": True,
            "proposal_only": True,
        }

    @property
    def proposal_id(self) -> str:
        return self.candidate_id

    @property
    def proposal_hash(self) -> str:
        return self.assert_integrity()

    @property
    def numeric_state_hash(self) -> str:
        return self.numeric_state.state_hash

    @property
    def queue_index(self) -> int:
        return self.numeric_state.queue_index

    @property
    def representative_branch(self) -> str:
        return self.numeric_state.representative_branch

    @property
    def representative_endpoint_index(self) -> int:
        return self.numeric_state.representative_endpoint_index

    @property
    def endpoint_index(self) -> int:
        return self.representative_endpoint_index

    @property
    def action_hash(self) -> str:
        return self.numeric_state.packed_action_hash

    @property
    def packed_action_hash(self) -> str:
        return self.numeric_state.packed_action_hash

    @property
    def parent_mask(self) -> HardMask:
        return self.numeric_state.parent_mask

    @property
    def parent_mask_hash(self) -> str:
        return self.numeric_state.parent_mask_hash

    @property
    def resulting_mask(self) -> HardMask:
        return self.numeric_state.resulting_mask

    @property
    def resulting_mask_hash(self) -> str:
        return self.numeric_state.resulting_mask_hash

    @property
    def removed_routes(self) -> V1RemovedRouteSetIdentity:
        return self.numeric_state.removed_routes

    @property
    def removed_count(self) -> int:
        return self.numeric_state.removed_count

    @property
    def removed_route_index_hash(self) -> str:
        return self.numeric_state.removed_route_index_hash

    @property
    def compact_removed_indices_available(self) -> bool:
        return False

    @property
    def resource_gain(self) -> float:
        return self.numeric_state.registered_resource

    @property
    def resource_receipt_hash(self) -> str:
        return self.numeric_state.resource_receipt_hash

    @property
    def six_role_interval_hash(self) -> str:
        return self.numeric_state.six_role_interval_hash

    @property
    def fqr_interval_hash(self) -> str:
        """Compatibility spelling for receipt consumers expecting interval identity."""

        return self.six_role_interval_hash

    @property
    def selected_coordinate_hash(self) -> str:
        return self.numeric_state.selected_coordinate_hash

    @property
    def six_role_authorization_hash(self) -> str:
        return self.numeric_state.six_role_authorization_hash

    @property
    def exact_intervention_damage_evaluated(self) -> bool:
        return False

    def resolve_removed_routes(self, parent_mask: HardMask):
        if type(parent_mask) is not HardMask or _fresh_mask_hash(
            parent_mask,
            "signed-work proposal parent",
        ) != self.parent_mask_hash:
            raise ValueError("signed-work proposal parent differs from its bound identity")
        resolved = self.resulting_mask.removed_index_set_from(parent_mask)
        if (
            resolved.count != self.removed_count
            or resolved.index_hash != self.removed_route_index_hash
        ):
            raise ValueError("resolved signed-work delta differs from its identity")
        return resolved

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.numeric_state.assert_integrity()
        self.__post_init__()
        if expected_hash is not None and self.candidate_id != expected_hash:
            raise ValueError("signed-work proposal hash does not match")
        return self.candidate_id

    def assert_can_certify_decision(self) -> None:
        raise PermissionError("signed-work proposals cannot certify a pruning decision")

    def assert_can_commit_mask(self) -> None:
        raise PermissionError("signed-work proposals cannot commit a hard mask")


def _domain_hash(states: Sequence[V1SignedWorkActionNumericState]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_signed_work_singleton_physical_domain_v1",
            "six_role_authorization_hash": states[0].six_role_authorization_hash,
            "parent_mask_hash": states[0].parent_mask_hash,
            "route_axis_hash": states[0].route_axis_hash,
            "route_count": states[0].route_count,
            "selected_coordinate_hashes": tuple(
                value.selected_coordinate_hash for value in states
            ),
            "queue_packed_action_hashes": tuple(value.packed_action_hash for value in states),
            "queue_resulting_mask_hashes": tuple(value.resulting_mask_hash for value in states),
            "single_endpoint_proposal_authority": True,
            "filtering_performed": False,
            "reordering_performed": False,
        },
        prefix="partition-free-v1-signed-work-domain-",
    )


def _certification_scope_hash(state: V1SignedWorkActionNumericState) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_signed_work_six_role_certification_scope_v1",
            "six_role_authorization_hash": state.six_role_authorization_hash,
            "six_role_certification_receipt_hash": state.six_role_certification_receipt_hash,
            "six_role_decision_hash": state.six_role_decision_hash,
            "six_role_decision_design_hash": state.six_role_decision_design_hash,
            "six_role_C_design_hash": state.six_role_C_design_hash,
            "six_role_C_progress_receipt_hash": state.six_role_C_progress_receipt_hash,
            "selected_coordinate_hash": state.selected_coordinate_hash,
            "direct_D_owns_finite_response_admission": True,
            "QR_transport_is_label_not_hidden_C_veto": True,
            "paired_bce_oracle_required": True,
        },
        prefix="partition-free-v1-signed-work-certification-scope-",
    )


def _issue_proposal(
    *,
    state: V1SignedWorkActionNumericState,
    domain_hash: str,
    certification_scope_hash: str,
) -> V1SignedWorkCandidateProposal:
    provisional = {
        "numeric_state": state,
        "domain_hash": domain_hash,
        "certification_scope_hash": certification_scope_hash,
        "proposal_only": True,
        "proposal_witness_only": True,
        "fresh_relinearization_required": True,
        "paired_bce_oracle_required": True,
        "candidate_freeze_authority": False,
        "bce_oracle_authority": False,
        "irreversible_decision_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
        "execution_authority": False,
    }
    shell = object.__new__(V1SignedWorkCandidateProposal)
    for name, value in provisional.items():
        object.__setattr__(shell, name, value)
    object.__setattr__(shell, "schema", SIGNED_WORK_CANDIDATE_PROPOSAL_SCHEMA)
    receipt_hash = content_hash(
        shell._evaluation_payload(),
        prefix="partition-free-v1-signed-work-candidate-evaluation-",
    )
    candidate_id = content_hash(
        {
            "evaluation_receipt_hash": receipt_hash,
            "resulting_mask_hash": state.resulting_mask_hash,
            "removed_route_index_hash": state.removed_route_index_hash,
            "six_role_interval_hash": state.six_role_interval_hash,
            "selected_coordinate_hash": state.selected_coordinate_hash,
        },
        prefix="partition-free-v1-signed-work-candidate-",
    )
    return V1SignedWorkCandidateProposal(
        _issuer_token=_PROPOSAL_TOKEN,
        **provisional,
        evaluation_receipt_hash=receipt_hash,
        candidate_id=candidate_id,
    )


@dataclass(frozen=True, slots=True, init=False)
class V1SignedWorkCandidateCompositionResult:
    """Controller-compatible singleton result from six-role direct-D authority."""

    action_numeric_states: tuple[V1SignedWorkActionNumericState, ...]
    proposals: tuple[V1SignedWorkCandidateProposal, ...]
    domain_hash: str
    certification_scope_hash: str
    six_role_authorization_hash: str
    six_role_certification_receipt_hash: str
    six_role_decision_hash: str
    six_role_decision_design_hash: str
    six_role_C_design_hash: str
    six_role_C_progress_receipt_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    queue_coordinate_hashes: tuple[str, ...]
    queue_branch_endpoint_coordinates: tuple[tuple[str, int], ...]
    queue_packed_action_hashes: tuple[str, ...]
    queue_resulting_mask_hashes: tuple[str, ...]
    queue_six_role_interval_hashes: tuple[str, ...]
    queue_resource_receipt_hashes: tuple[str, ...]
    validated_action_count: int
    proposal_count: int
    composer_id: str
    selection_receipt_hash: str
    single_endpoint_materialized_exactly: bool
    complete_queue_frozen_before_K_O: bool
    filtering_performed: bool
    reordering_performed: bool
    claims_hard_global_optimum: bool
    omitted_mask_optimality_authority: bool
    proposal_only: bool
    proposal_witness_only: bool
    fresh_relinearization_required: bool
    paired_bce_oracle_required: bool
    candidate_freeze_authority: bool
    bce_oracle_authority: bool
    irreversible_decision_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    schema: str = SIGNED_WORK_CANDIDATE_COMPOSITION_SCHEMA
    _result_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _COMPOSITION_TOKEN:
            raise TypeError("V1SignedWorkCandidateCompositionResult is issuer-only")
        expected = tuple(
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        if set(values) != set(expected):
            missing = sorted(set(expected).difference(values))
            extra = sorted(set(values).difference(expected))
            raise TypeError(
                "signed-work composition fields differ: "
                f"missing={missing}, extra={extra}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", SIGNED_WORK_CANDIDATE_COMPOSITION_SCHEMA)
        object.__setattr__(self, "_result_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        states = tuple(self.action_numeric_states)
        proposals = tuple(self.proposals)
        if len(states) != 1 or any(
            type(item) is not V1SignedWorkActionNumericState for item in states
        ):
            raise TypeError("signed-work composition requires one issued numeric state")
        if len(proposals) != 1 or any(
            type(item) is not V1SignedWorkCandidateProposal for item in proposals
        ):
            raise TypeError("signed-work composition requires one issued proposal")
        state = states[0]
        proposal = proposals[0]
        if proposal.numeric_state is not state:
            raise ValueError("signed-work proposal recontextualized its numeric state")
        if (
            proposal.domain_hash != self.domain_hash
            or proposal.certification_scope_hash != self.certification_scope_hash
        ):
            raise ValueError("signed-work proposal names another composition")
        shared = (
            ("six_role_authorization_hash", state.six_role_authorization_hash),
            (
                "six_role_certification_receipt_hash",
                state.six_role_certification_receipt_hash,
            ),
            ("six_role_decision_hash", state.six_role_decision_hash),
            ("six_role_decision_design_hash", state.six_role_decision_design_hash),
            ("six_role_C_design_hash", state.six_role_C_design_hash),
            ("six_role_C_progress_receipt_hash", state.six_role_C_progress_receipt_hash),
            ("parent_mask_hash", state.parent_mask_hash),
            ("route_axis_hash", state.route_axis_hash),
        )
        if any(getattr(self, name) != expected for name, expected in shared):
            raise ValueError("signed-work composition changed a source identity")
        if (
            self.route_count != state.route_count
            or self.queue_coordinate_hashes != (state.selected_coordinate_hash,)
            or self.queue_branch_endpoint_coordinates
            != ((state.representative_branch, state.representative_endpoint_index),)
            or self.queue_packed_action_hashes != (state.packed_action_hash,)
            or self.queue_resulting_mask_hashes != (state.resulting_mask_hash,)
            or self.queue_six_role_interval_hashes != (state.six_role_interval_hash,)
            or self.queue_resource_receipt_hashes != (state.resource_receipt_hash,)
        ):
            raise ValueError("signed-work composition silently changed its queue axis")
        if self.domain_hash != _domain_hash(states):
            raise ValueError("signed-work domain hash is not reproducible")
        if self.certification_scope_hash != _certification_scope_hash(state):
            raise ValueError("signed-work certification scope hash is not reproducible")
        if (
            _positive_int(self.validated_action_count, "validated_action_count") != 1
            or _positive_int(self.proposal_count, "proposal_count") != 1
            or self.composer_id != SIGNED_WORK_SINGLETON_COMPOSER_ID
        ):
            raise ValueError("signed-work composition is not a singleton direct-D queue")
        firewall = (
            self.single_endpoint_materialized_exactly,
            self.complete_queue_frozen_before_K_O,
            not self.filtering_performed,
            not self.reordering_performed,
            not self.claims_hard_global_optimum,
            not self.omitted_mask_optimality_authority,
            self.proposal_only,
            self.proposal_witness_only,
            self.fresh_relinearization_required,
            self.paired_bce_oracle_required,
            not self.candidate_freeze_authority,
            not self.bce_oracle_authority,
            not self.irreversible_decision_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.execution_authority,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("signed-work composition crossed its proposal-only firewall")
        if self.schema != SIGNED_WORK_CANDIDATE_COMPOSITION_SCHEMA:
            raise ValueError("unsupported signed-work composition schema")
        expected_selection = content_hash(
            self._selection_payload(),
            prefix="partition-free-v1-signed-work-candidate-selection-",
        )
        if self.selection_receipt_hash != expected_selection:
            raise ValueError("signed-work selection receipt is not reproducible")
        object.__setattr__(self, "action_numeric_states", states)
        object.__setattr__(self, "proposals", proposals)
        self.assert_integrity()

    @property
    def constraints_hash(self) -> str:
        return self.certification_scope_hash

    @property
    def ranked_proposals(self) -> tuple[V1SignedWorkCandidateProposal, ...]:
        return self.proposals

    @property
    def selected_proposal(self) -> V1SignedWorkCandidateProposal:
        return self.proposals[0]

    @property
    def result_hash(self) -> str:
        return self.assert_integrity()

    def _selection_payload(self) -> dict[str, object]:
        return {
            "schema": "partition_free_v1_signed_work_candidate_selection_v1",
            "composer_id": self.composer_id,
            "domain_hash": self.domain_hash,
            "certification_scope_hash": self.certification_scope_hash,
            "six_role_authorization_hash": self.six_role_authorization_hash,
            "queue_coordinate_hashes": self.queue_coordinate_hashes,
            "queue_candidate_ids": tuple(value.candidate_id for value in self.proposals),
            "queue_branch_endpoint_coordinates": self.queue_branch_endpoint_coordinates,
            "queue_packed_action_hashes": self.queue_packed_action_hashes,
            "validated_action_count": self.validated_action_count,
            "proposal_count": self.proposal_count,
            "single_endpoint_materialized_exactly": True,
            "complete_queue_frozen_before_K_O": True,
            "filtering_performed": False,
            "reordering_performed": False,
            "claims_hard_global_optimum": False,
            "omitted_mask_optimality_authority": False,
            "proposal_only": True,
            "paired_bce_oracle_required": True,
        }

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "action_numeric_state_hashes": tuple(
                value.state_hash for value in self.action_numeric_states
            ),
            "candidate_ids": tuple(value.candidate_id for value in self.proposals),
            "domain_hash": self.domain_hash,
            "certification_scope_hash": self.certification_scope_hash,
            "six_role_authorization_hash": self.six_role_authorization_hash,
            "six_role_certification_receipt_hash": self.six_role_certification_receipt_hash,
            "six_role_decision_hash": self.six_role_decision_hash,
            "six_role_decision_design_hash": self.six_role_decision_design_hash,
            "six_role_C_design_hash": self.six_role_C_design_hash,
            "six_role_C_progress_receipt_hash": self.six_role_C_progress_receipt_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "queue_coordinate_hashes": self.queue_coordinate_hashes,
            "queue_branch_endpoint_coordinates": self.queue_branch_endpoint_coordinates,
            "queue_packed_action_hashes": self.queue_packed_action_hashes,
            "queue_resulting_mask_hashes": self.queue_resulting_mask_hashes,
            "queue_six_role_interval_hashes": self.queue_six_role_interval_hashes,
            "queue_resource_receipt_hashes": self.queue_resource_receipt_hashes,
            "validated_action_count": self.validated_action_count,
            "proposal_count": self.proposal_count,
            "composer_id": self.composer_id,
            "selection_receipt_hash": self.selection_receipt_hash,
            "single_endpoint_materialized_exactly": self.single_endpoint_materialized_exactly,
            "complete_queue_frozen_before_K_O": self.complete_queue_frozen_before_K_O,
            "filtering_performed": self.filtering_performed,
            "reordering_performed": self.reordering_performed,
            "claims_hard_global_optimum": self.claims_hard_global_optimum,
            "omitted_mask_optimality_authority": self.omitted_mask_optimality_authority,
            "proposal_only": self.proposal_only,
            "proposal_witness_only": self.proposal_witness_only,
            "fresh_relinearization_required": self.fresh_relinearization_required,
            "paired_bce_oracle_required": self.paired_bce_oracle_required,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "execution_authority": self.execution_authority,
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for state, proposal in zip(self.action_numeric_states, self.proposals, strict=True):
            state.assert_integrity()
            proposal.assert_integrity()
        observed = content_hash(
            self._payload(),
            prefix="partition-free-v1-signed-work-composition-",
        )
        if self._result_hash is not None and observed != self._result_hash:
            raise ValueError("signed-work composition was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("signed-work composition hash does not match")
        object.__setattr__(self, "_result_hash", observed)
        return observed


def _invalidation_chart_hash(
    *,
    generation_index: int,
    current_mask: HardMask,
    composition: V1SignedWorkCandidateCompositionResult,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_signed_work_invalidation_chart_v1",
            "generation_index": generation_index,
            "current_mask_hash": current_mask.mask_hash,
            "route_axis_hash": current_mask.route_axis_hash,
            "six_role_authorization_hash": composition.six_role_authorization_hash,
            "six_role_decision_design_hash": composition.six_role_decision_design_hash,
            "six_role_C_design_hash": composition.six_role_C_design_hash,
            "queue_coordinate_hashes": composition.queue_coordinate_hashes,
            "old_chart_validity_claim": (
                composition.action_numeric_states[0].old_chart_validity_claim
            ),
        },
        prefix="partition-free-v1-signed-work-invalidation-chart-",
    )


def _acquisition_hash(
    *,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    lineage: tuple[str, ...],
    transitions: tuple[str, ...],
    invalidation_chart_hash: str,
    composition_result_hash: str,
    metadata: Mapping[str, Any],
) -> str:
    return content_hash(
        {
            "schema": SIGNED_WORK_AUTHORIZED_ACQUISITION_SCHEMA,
            "campaign_binding_hash": binding.binding_hash,
            "generation_index": generation_index,
            "current_mask_hash": current_mask.mask_hash,
            "committed_mask_lineage_hashes": lineage,
            "committed_mask_transition_receipt_hashes": transitions,
            "invalidation_chart_hash": invalidation_chart_hash,
            "composition_result_hash": composition_result_hash,
            "metadata": metadata,
        },
        prefix="partition-free-v1-signed-work-authorized-acquisition-",
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkAcquisitionGenerationReceipt:
    campaign_contract_hash: str
    campaign_binding_hash: str
    registration_hash: str
    generation_index: int
    current_mask_hash: str
    route_axis_hash: str
    acquisition_hash: str
    invalidation_chart_hash: str
    composition_result_hash: str
    domain_hash: str
    certification_scope_hash: str
    six_role_authorization_hash: str
    six_role_certification_receipt_hash: str
    six_role_decision_hash: str
    six_role_decision_design_hash: str
    six_role_C_design_hash: str
    six_role_C_progress_receipt_hash: str
    selected_coordinate_hash: str
    ordered_candidate_ids: tuple[str, ...]
    ordered_coordinate_hashes: tuple[str, ...]
    ordered_branch_endpoint_coordinates: tuple[tuple[str, int], ...]
    ordered_packed_action_hashes: tuple[str, ...]
    ordered_six_role_interval_hashes: tuple[str, ...]
    refresh_reason: str
    logical_sequence: int
    proposal_only: bool = True
    complete_queue_frozen_before_K_O: bool = True
    paired_bce_oracle_required: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    candidate_freeze_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    claim_execution_certified: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SIGNED_WORK_ACQUISITION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "registration_hash",
            "current_mask_hash",
            "route_axis_hash",
            "acquisition_hash",
            "invalidation_chart_hash",
            "composition_result_hash",
            "domain_hash",
            "certification_scope_hash",
            "six_role_authorization_hash",
            "six_role_certification_receipt_hash",
            "six_role_decision_hash",
            "six_role_decision_design_hash",
            "six_role_C_design_hash",
            "six_role_C_progress_receipt_hash",
            "selected_coordinate_hash",
            "refresh_reason",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        object.__setattr__(
            self, "generation_index", _nonnegative_int(self.generation_index, "generation_index")
        )
        object.__setattr__(
            self, "logical_sequence", _nonnegative_int(self.logical_sequence, "logical_sequence")
        )
        candidate_ids = tuple(
            _identity(value, "ordered candidate id") for value in self.ordered_candidate_ids
        )
        coordinate_hashes = tuple(
            _identity(value, "ordered coordinate hash")
            for value in self.ordered_coordinate_hashes
        )
        packed_hashes = tuple(
            _identity(value, "ordered packed action hash")
            for value in self.ordered_packed_action_hashes
        )
        interval_hashes = tuple(
            _identity(value, "ordered six-role interval hash")
            for value in self.ordered_six_role_interval_hashes
        )
        coordinates = tuple(
            (_branch_value(branch, "ordered branch"), _positive_int(endpoint, "ordered endpoint"))
            for branch, endpoint in self.ordered_branch_endpoint_coordinates
        )
        if (
            len(candidate_ids) != 1
            or coordinate_hashes != (self.selected_coordinate_hash,)
            or len(packed_hashes) != 1
            or len(interval_hashes) != 1
            or len(coordinates) != 1
        ):
            raise ValueError("signed-work receipt must retain exactly one proposal")
        firewall = (
            self.proposal_only,
            self.complete_queue_frozen_before_K_O,
            self.paired_bce_oracle_required,
            not self.calibration_K_open_authority,
            not self.paired_oracle_O_open_authority,
            not self.candidate_freeze_authority,
            not self.irreversible_decision_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.claim_execution_certified,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("signed-work acquisition receipt crossed its authority firewall")
        if self.schema != SIGNED_WORK_ACQUISITION_RECEIPT_SCHEMA:
            raise ValueError("unsupported signed-work acquisition-receipt schema")
        object.__setattr__(self, "ordered_candidate_ids", candidate_ids)
        object.__setattr__(self, "ordered_coordinate_hashes", coordinate_hashes)
        object.__setattr__(self, "ordered_branch_endpoint_coordinates", coordinates)
        object.__setattr__(self, "ordered_packed_action_hashes", packed_hashes)
        object.__setattr__(self, "ordered_six_role_interval_hashes", interval_hashes)
        object.__setattr__(self, "metadata", _freeze(self.metadata, "signed-work receipt metadata"))

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self._payload(),
            prefix="partition-free-v1-signed-work-acquisition-generation-",
        )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "campaign_contract_hash",
                "campaign_binding_hash",
                "registration_hash",
                "generation_index",
                "current_mask_hash",
                "route_axis_hash",
                "acquisition_hash",
                "invalidation_chart_hash",
                "composition_result_hash",
                "domain_hash",
                "certification_scope_hash",
                "six_role_authorization_hash",
                "six_role_certification_receipt_hash",
                "six_role_decision_hash",
                "six_role_decision_design_hash",
                "six_role_C_design_hash",
                "six_role_C_progress_receipt_hash",
                "selected_coordinate_hash",
                "ordered_candidate_ids",
                "ordered_coordinate_hashes",
                "ordered_branch_endpoint_coordinates",
                "ordered_packed_action_hashes",
                "ordered_six_role_interval_hashes",
                "refresh_reason",
                "logical_sequence",
                "proposal_only",
                "complete_queue_frozen_before_K_O",
                "paired_bce_oracle_required",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "candidate_freeze_authority",
                "irreversible_decision_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
                "claim_execution_certified",
                "metadata",
                "schema",
            )
        }


@dataclass(frozen=True, slots=True)
class V1SignedWorkAuthorizedAcquisition:
    binding: V1OptionACampaignBinding
    generation_index: int
    current_mask: HardMask
    committed_mask_lineage_hashes: tuple[str, ...]
    committed_mask_transition_receipt_hashes: tuple[str, ...]
    composition: V1SignedWorkCandidateCompositionResult
    receipt: V1SignedWorkAcquisitionGenerationReceipt
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SIGNED_WORK_AUTHORIZED_ACQUISITION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.binding) is not V1OptionACampaignBinding:
            raise TypeError("signed-work acquisition requires an exact campaign binding")
        if type(self.current_mask) is not HardMask:
            raise TypeError("signed-work acquisition requires an exact current mask")
        if type(self.composition) is not V1SignedWorkCandidateCompositionResult:
            raise TypeError("signed-work acquisition requires its exact composition")
        if type(self.receipt) is not V1SignedWorkAcquisitionGenerationReceipt:
            raise TypeError("signed-work acquisition requires its exact receipt")
        self.composition.assert_integrity()
        generation = _nonnegative_int(self.generation_index, "generation_index")
        lineage = tuple(
            _identity(value, "mask lineage hash")
            for value in self.committed_mask_lineage_hashes
        )
        transitions = tuple(
            _identity(value, "mask transition hash")
            for value in self.committed_mask_transition_receipt_hashes
        )
        current_hash = _fresh_mask_hash(self.current_mask, "signed-work current mask")
        if (
            not lineage
            or lineage[-1] != current_hash
            or len(transitions) != len(lineage) - 1
            or len(set(lineage)) != len(lineage)
            or len(set(transitions)) != len(transitions)
        ):
            raise ValueError("signed-work acquisition has incomplete exact mask ancestry")
        composition = self.composition
        receipt = self.receipt
        contract = self.binding.contract
        if (
            contract.route_axis_hash != self.current_mask.route_axis_hash
            or composition.parent_mask_hash != self.current_mask.mask_hash
            or composition.route_axis_hash != self.current_mask.route_axis_hash
            or composition.route_count != self.current_mask.route_count
            or composition.action_numeric_states[0].generation_index != generation
        ):
            raise ValueError("signed-work acquisition differs from the current generation")
        if (
            receipt.campaign_contract_hash != contract.campaign_contract_hash
            or receipt.campaign_binding_hash != self.binding.binding_hash
            or receipt.registration_hash != self.binding.registration.registration_hash
            or receipt.generation_index != generation
            or receipt.current_mask_hash != self.current_mask.mask_hash
            or receipt.route_axis_hash != self.current_mask.route_axis_hash
            or receipt.composition_result_hash != composition.result_hash
            or receipt.domain_hash != composition.domain_hash
            or receipt.certification_scope_hash != composition.certification_scope_hash
            or receipt.six_role_authorization_hash != composition.six_role_authorization_hash
            or receipt.six_role_certification_receipt_hash
            != composition.six_role_certification_receipt_hash
            or receipt.six_role_decision_hash != composition.six_role_decision_hash
            or receipt.six_role_decision_design_hash
            != composition.six_role_decision_design_hash
            or receipt.six_role_C_design_hash != composition.six_role_C_design_hash
            or receipt.six_role_C_progress_receipt_hash
            != composition.six_role_C_progress_receipt_hash
            or receipt.selected_coordinate_hash != composition.queue_coordinate_hashes[0]
            or receipt.ordered_candidate_ids
            != tuple(value.candidate_id for value in composition.proposals)
            or receipt.ordered_coordinate_hashes != composition.queue_coordinate_hashes
            or receipt.ordered_branch_endpoint_coordinates
            != composition.queue_branch_endpoint_coordinates
            or receipt.ordered_packed_action_hashes
            != composition.queue_packed_action_hashes
            or receipt.ordered_six_role_interval_hashes
            != composition.queue_six_role_interval_hashes
            or receipt.invalidation_chart_hash != self.invalidation_chart_hash
            or receipt.acquisition_hash != self.acquisition_hash
        ):
            raise ValueError("signed-work acquisition receipt differs from its sources")
        if self.schema != SIGNED_WORK_AUTHORIZED_ACQUISITION_SCHEMA:
            raise ValueError("unsupported signed-work acquisition schema")
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "committed_mask_lineage_hashes", lineage)
        object.__setattr__(self, "committed_mask_transition_receipt_hashes", transitions)
        object.__setattr__(self, "metadata", _freeze(self.metadata, "signed-work acquisition metadata"))

    @property
    def invalidation_chart_hash(self) -> str:
        return _invalidation_chart_hash(
            generation_index=self.generation_index,
            current_mask=self.current_mask,
            composition=self.composition,
        )

    @property
    def acquisition_hash(self) -> str:
        return _acquisition_hash(
            binding=self.binding,
            generation_index=self.generation_index,
            current_mask=self.current_mask,
            lineage=self.committed_mask_lineage_hashes,
            transitions=self.committed_mask_transition_receipt_hashes,
            invalidation_chart_hash=self.invalidation_chart_hash,
            composition_result_hash=self.composition.result_hash,
            metadata=self.metadata,
        )

    @property
    def queue(self) -> tuple[V1SignedWorkCandidateProposal, ...]:
        return self.composition.proposals

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def mask_commit_authority(self) -> bool:
        return False


def _fallback_domain_hash(
    states: Sequence[V1SignedWorkActionNumericState],
    *,
    fallback_authorization_hash: str,
    physical_candidate_identity_hashes: tuple[str, ...],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_signed_work_fallback_physical_domain_v1",
            "six_role_fallback_authorization_hash": fallback_authorization_hash,
            "parent_mask_hash": states[0].parent_mask_hash,
            "route_axis_hash": states[0].route_axis_hash,
            "route_count": states[0].route_count,
            "selected_coordinate_hashes": tuple(
                value.selected_coordinate_hash for value in states
            ),
            "queue_packed_action_hashes": tuple(value.packed_action_hash for value in states),
            "queue_resulting_mask_hashes": tuple(value.resulting_mask_hash for value in states),
            "queued_physical_candidate_identity_hashes": physical_candidate_identity_hashes,
            "fallback_queue_proposal_authority": True,
            "filtering_performed": "source_authorized_refuted_duplicate_unresolved_filter",
            "reordering_performed": False,
        },
        prefix="partition-free-v1-signed-work-domain-",
    )


def _fallback_certification_scope_hash(
    states: Sequence[V1SignedWorkActionNumericState],
    *,
    fallback_authorization_hash: str,
    blocked_unresolved_coordinate_hash: str | None,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_signed_work_fallback_certification_scope_v1",
            "six_role_fallback_authorization_hash": fallback_authorization_hash,
            "six_role_certification_receipt_hash": states[0].six_role_certification_receipt_hash,
            "six_role_decision_hash": states[0].six_role_decision_hash,
            "six_role_decision_design_hash": states[0].six_role_decision_design_hash,
            "six_role_C_design_hash": states[0].six_role_C_design_hash,
            "six_role_C_progress_receipt_hash": states[0].six_role_C_progress_receipt_hash,
            "queue_coordinate_hashes": tuple(value.selected_coordinate_hash for value in states),
            "blocked_unresolved_coordinate_hash": blocked_unresolved_coordinate_hash,
            "direct_D_owns_finite_response_admission": True,
            "QR_transport_is_label_not_hidden_C_veto": True,
            "paired_bce_oracle_required": True,
        },
        prefix="partition-free-v1-signed-work-certification-scope-",
    )


@dataclass(frozen=True, slots=True, init=False)
class V1SignedWorkCandidateCompositionResultV2:
    """Controller-compatible pre-O fallback queue from six-role direct-D authority."""

    action_numeric_states: tuple[V1SignedWorkActionNumericState, ...]
    proposals: tuple[V1SignedWorkCandidateProposal, ...]
    domain_hash: str
    certification_scope_hash: str
    six_role_authorization_hash: str
    six_role_fallback_authorization_hash: str
    six_role_certification_receipt_hash: str
    six_role_decision_hash: str
    six_role_decision_design_hash: str
    six_role_C_design_hash: str
    six_role_C_progress_receipt_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    queue_coordinate_hashes: tuple[str, ...]
    queue_branch_endpoint_coordinates: tuple[tuple[str, int], ...]
    queue_packed_action_hashes: tuple[str, ...]
    queue_resulting_mask_hashes: tuple[str, ...]
    queue_six_role_interval_hashes: tuple[str, ...]
    queue_resource_receipt_hashes: tuple[str, ...]
    queued_physical_candidate_identity_hashes: tuple[str, ...]
    blocked_unresolved_coordinate_hash: str | None
    validated_action_count: int
    proposal_count: int
    composer_id: str
    selection_receipt_hash: str
    fallback_queue_materialized_exactly: bool
    complete_queue_frozen_before_K_O: bool
    resolved_prefix_queue_frozen_before_K_O: bool
    fallback_exhaustion_status: str
    same_generation_paired_O_fallback_without_recomposition_authority: bool
    source_authorized_filtering_performed: bool
    reordering_performed: bool
    claims_hard_global_optimum: bool
    omitted_mask_optimality_authority: bool
    proposal_only: bool
    proposal_witness_only: bool
    fresh_relinearization_required: bool
    paired_bce_oracle_required: bool
    candidate_freeze_authority: bool
    bce_oracle_authority: bool
    irreversible_decision_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    schema: str = SIGNED_WORK_CANDIDATE_COMPOSITION_V2_SCHEMA
    _result_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _COMPOSITION_TOKEN:
            raise TypeError("V1SignedWorkCandidateCompositionResultV2 is issuer-only")
        expected = tuple(
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        if set(values) != set(expected):
            missing = sorted(set(expected).difference(values))
            extra = sorted(set(values).difference(expected))
            raise TypeError(
                "signed-work fallback composition fields differ: "
                f"missing={missing}, extra={extra}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", SIGNED_WORK_CANDIDATE_COMPOSITION_V2_SCHEMA)
        object.__setattr__(self, "_result_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        states = tuple(self.action_numeric_states)
        proposals = tuple(self.proposals)
        if not states or any(
            type(item) is not V1SignedWorkActionNumericState for item in states
        ):
            raise TypeError("signed-work fallback composition requires issued states")
        if len(proposals) != len(states) or any(
            type(item) is not V1SignedWorkCandidateProposal for item in proposals
        ):
            raise TypeError("signed-work fallback composition requires issued proposals")
        for index, (state, proposal) in enumerate(zip(states, proposals, strict=True)):
            if state.queue_index != index:
                raise ValueError("signed-work fallback queue indices are not contiguous")
            if proposal.numeric_state is not state:
                raise ValueError("signed-work fallback proposal changed its state")
            if (
                proposal.domain_hash != self.domain_hash
                or proposal.certification_scope_hash != self.certification_scope_hash
            ):
                raise ValueError("signed-work fallback proposal names another composition")
        first = states[0]
        shared = (
            ("six_role_authorization_hash", first.six_role_authorization_hash),
            (
                "six_role_fallback_authorization_hash",
                first.six_role_authorization_hash,
            ),
            (
                "six_role_certification_receipt_hash",
                first.six_role_certification_receipt_hash,
            ),
            ("six_role_decision_hash", first.six_role_decision_hash),
            ("six_role_decision_design_hash", first.six_role_decision_design_hash),
            ("six_role_C_design_hash", first.six_role_C_design_hash),
            ("six_role_C_progress_receipt_hash", first.six_role_C_progress_receipt_hash),
            ("parent_mask_hash", first.parent_mask_hash),
            ("route_axis_hash", first.route_axis_hash),
        )
        if any(getattr(self, name) != expected for name, expected in shared):
            raise ValueError("signed-work fallback composition changed a source identity")
        if any(
            value.parent_mask_hash != first.parent_mask_hash
            or value.route_axis_hash != first.route_axis_hash
            or value.route_count != first.route_count
            or value.generation_index != first.generation_index
            or value.six_role_authorization_hash != first.six_role_authorization_hash
            for value in states
        ):
            raise ValueError("signed-work fallback states do not share one source frame")
        coordinate_hashes = tuple(value.selected_coordinate_hash for value in states)
        branch_endpoints = tuple(
            (value.representative_branch, value.representative_endpoint_index)
            for value in states
        )
        packed_hashes = tuple(value.packed_action_hash for value in states)
        mask_hashes = tuple(value.resulting_mask_hash for value in states)
        interval_hashes = tuple(value.six_role_interval_hash for value in states)
        resource_hashes = tuple(value.resource_receipt_hash for value in states)
        order_key = tuple(
            (-value.registered_resource, value.pre_C_priority) for value in states
        )
        physical = tuple(
            _identity(value, "queued physical candidate identity")
            for value in self.queued_physical_candidate_identity_hashes
        )
        if (
            self.route_count != first.route_count
            or self.queue_coordinate_hashes != coordinate_hashes
            or self.queue_branch_endpoint_coordinates != branch_endpoints
            or self.queue_packed_action_hashes != packed_hashes
            or self.queue_resulting_mask_hashes != mask_hashes
            or self.queue_six_role_interval_hashes != interval_hashes
            or self.queue_resource_receipt_hashes != resource_hashes
            or order_key != tuple(sorted(order_key))
            or len(set(mask_hashes)) != len(mask_hashes)
            or len(physical) != len(states)
            or len(set(physical)) != len(physical)
        ):
            raise ValueError("signed-work fallback queue axis is not reproducible")
        if self.domain_hash != _fallback_domain_hash(
            states,
            fallback_authorization_hash=self.six_role_fallback_authorization_hash,
            physical_candidate_identity_hashes=physical,
        ):
            raise ValueError("signed-work fallback domain hash is not reproducible")
        if self.certification_scope_hash != _fallback_certification_scope_hash(
            states,
            fallback_authorization_hash=self.six_role_fallback_authorization_hash,
            blocked_unresolved_coordinate_hash=self.blocked_unresolved_coordinate_hash,
        ):
            raise ValueError(
                "signed-work fallback certification scope hash is not reproducible"
            )
        if (
            _positive_int(self.validated_action_count, "validated_action_count")
            != len(states)
            or _positive_int(self.proposal_count, "proposal_count") != len(proposals)
            or self.composer_id != SIGNED_WORK_PRE_O_FALLBACK_COMPOSER_ID
        ):
            raise ValueError("signed-work fallback composition has the wrong queue size")
        if self.blocked_unresolved_coordinate_hash is not None:
            _identity(
                self.blocked_unresolved_coordinate_hash,
                "blocked unresolved coordinate hash",
            )
        if self.fallback_exhaustion_status not in {
            "blocked_on_more_C",
            "fully_resolved_no_more_candidate",
        }:
            raise ValueError("signed-work fallback exhaustion status is unsupported")
        if (
            self.blocked_unresolved_coordinate_hash is None
            and self.fallback_exhaustion_status != "fully_resolved_no_more_candidate"
        ) or (
            self.blocked_unresolved_coordinate_hash is not None
            and self.fallback_exhaustion_status != "blocked_on_more_C"
        ):
            raise ValueError("signed-work fallback exhaustion status contradicts blocker")
        firewall = (
            self.fallback_queue_materialized_exactly,
            self.complete_queue_frozen_before_K_O,
            self.resolved_prefix_queue_frozen_before_K_O,
            self.same_generation_paired_O_fallback_without_recomposition_authority,
            self.source_authorized_filtering_performed,
            not self.reordering_performed,
            not self.claims_hard_global_optimum,
            not self.omitted_mask_optimality_authority,
            self.proposal_only,
            self.proposal_witness_only,
            self.fresh_relinearization_required,
            self.paired_bce_oracle_required,
            not self.candidate_freeze_authority,
            not self.bce_oracle_authority,
            not self.irreversible_decision_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.execution_authority,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("signed-work fallback composition crossed its firewall")
        if self.schema != SIGNED_WORK_CANDIDATE_COMPOSITION_V2_SCHEMA:
            raise ValueError("unsupported signed-work fallback composition schema")
        expected_selection = content_hash(
            self._selection_payload(),
            prefix="partition-free-v1-signed-work-candidate-selection-",
        )
        if self.selection_receipt_hash != expected_selection:
            raise ValueError("signed-work fallback selection receipt is not reproducible")
        object.__setattr__(self, "action_numeric_states", states)
        object.__setattr__(self, "proposals", proposals)
        object.__setattr__(
            self,
            "queued_physical_candidate_identity_hashes",
            physical,
        )
        self.assert_integrity()

    @property
    def constraints_hash(self) -> str:
        return self.certification_scope_hash

    @property
    def ranked_proposals(self) -> tuple[V1SignedWorkCandidateProposal, ...]:
        return self.proposals

    @property
    def selected_proposal(self) -> V1SignedWorkCandidateProposal:
        return self.proposals[0]

    @property
    def result_hash(self) -> str:
        return self.assert_integrity()

    def _selection_payload(self) -> dict[str, object]:
        return {
            "schema": "partition_free_v1_signed_work_candidate_selection_v2",
            "composer_id": self.composer_id,
            "domain_hash": self.domain_hash,
            "certification_scope_hash": self.certification_scope_hash,
            "six_role_fallback_authorization_hash": (
                self.six_role_fallback_authorization_hash
            ),
            "queue_coordinate_hashes": self.queue_coordinate_hashes,
            "queue_candidate_ids": tuple(value.candidate_id for value in self.proposals),
            "queue_branch_endpoint_coordinates": self.queue_branch_endpoint_coordinates,
            "queue_packed_action_hashes": self.queue_packed_action_hashes,
            "queue_resulting_mask_hashes": self.queue_resulting_mask_hashes,
            "queued_physical_candidate_identity_hashes": (
                self.queued_physical_candidate_identity_hashes
            ),
            "blocked_unresolved_coordinate_hash": self.blocked_unresolved_coordinate_hash,
            "validated_action_count": self.validated_action_count,
            "proposal_count": self.proposal_count,
            "fallback_queue_materialized_exactly": True,
            "complete_queue_frozen_before_K_O": True,
            "resolved_prefix_queue_frozen_before_K_O": True,
            "fallback_exhaustion_status": self.fallback_exhaustion_status,
            "same_generation_paired_O_fallback_without_recomposition_authority": True,
            "source_authorized_filtering_performed": True,
            "reordering_performed": False,
            "claims_hard_global_optimum": False,
            "omitted_mask_optimality_authority": False,
            "proposal_only": True,
            "paired_bce_oracle_required": True,
        }

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "action_numeric_state_hashes": tuple(
                value.state_hash for value in self.action_numeric_states
            ),
            "candidate_ids": tuple(value.candidate_id for value in self.proposals),
            "domain_hash": self.domain_hash,
            "certification_scope_hash": self.certification_scope_hash,
            "six_role_authorization_hash": self.six_role_authorization_hash,
            "six_role_fallback_authorization_hash": (
                self.six_role_fallback_authorization_hash
            ),
            "six_role_certification_receipt_hash": self.six_role_certification_receipt_hash,
            "six_role_decision_hash": self.six_role_decision_hash,
            "six_role_decision_design_hash": self.six_role_decision_design_hash,
            "six_role_C_design_hash": self.six_role_C_design_hash,
            "six_role_C_progress_receipt_hash": self.six_role_C_progress_receipt_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "queue_coordinate_hashes": self.queue_coordinate_hashes,
            "queue_branch_endpoint_coordinates": self.queue_branch_endpoint_coordinates,
            "queue_packed_action_hashes": self.queue_packed_action_hashes,
            "queue_resulting_mask_hashes": self.queue_resulting_mask_hashes,
            "queue_six_role_interval_hashes": self.queue_six_role_interval_hashes,
            "queue_resource_receipt_hashes": self.queue_resource_receipt_hashes,
            "queued_physical_candidate_identity_hashes": (
                self.queued_physical_candidate_identity_hashes
            ),
            "blocked_unresolved_coordinate_hash": self.blocked_unresolved_coordinate_hash,
            "validated_action_count": self.validated_action_count,
            "proposal_count": self.proposal_count,
            "composer_id": self.composer_id,
            "selection_receipt_hash": self.selection_receipt_hash,
            "fallback_queue_materialized_exactly": (
                self.fallback_queue_materialized_exactly
            ),
            "complete_queue_frozen_before_K_O": self.complete_queue_frozen_before_K_O,
            "resolved_prefix_queue_frozen_before_K_O": (
                self.resolved_prefix_queue_frozen_before_K_O
            ),
            "fallback_exhaustion_status": self.fallback_exhaustion_status,
            "same_generation_paired_O_fallback_without_recomposition_authority": (
                self.same_generation_paired_O_fallback_without_recomposition_authority
            ),
            "source_authorized_filtering_performed": (
                self.source_authorized_filtering_performed
            ),
            "reordering_performed": self.reordering_performed,
            "claims_hard_global_optimum": self.claims_hard_global_optimum,
            "omitted_mask_optimality_authority": self.omitted_mask_optimality_authority,
            "proposal_only": self.proposal_only,
            "proposal_witness_only": self.proposal_witness_only,
            "fresh_relinearization_required": self.fresh_relinearization_required,
            "paired_bce_oracle_required": self.paired_bce_oracle_required,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "execution_authority": self.execution_authority,
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for state, proposal in zip(self.action_numeric_states, self.proposals, strict=True):
            state.assert_integrity()
            proposal.assert_integrity()
        observed = content_hash(
            self._payload(),
            prefix="partition-free-v1-signed-work-composition-",
        )
        if self._result_hash is not None and observed != self._result_hash:
            raise ValueError("signed-work fallback composition was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("signed-work fallback composition hash does not match")
        object.__setattr__(self, "_result_hash", observed)
        return observed


def _fallback_invalidation_chart_hash(
    *,
    generation_index: int,
    current_mask: HardMask,
    composition: V1SignedWorkCandidateCompositionResultV2,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_signed_work_fallback_invalidation_chart_v1",
            "generation_index": generation_index,
            "current_mask_hash": current_mask.mask_hash,
            "route_axis_hash": current_mask.route_axis_hash,
            "six_role_fallback_authorization_hash": (
                composition.six_role_fallback_authorization_hash
            ),
            "six_role_decision_design_hash": composition.six_role_decision_design_hash,
            "six_role_C_design_hash": composition.six_role_C_design_hash,
            "queue_coordinate_hashes": composition.queue_coordinate_hashes,
            "old_chart_validity_claims": tuple(
                value.old_chart_validity_claim
                for value in composition.action_numeric_states
            ),
        },
        prefix="partition-free-v1-signed-work-invalidation-chart-",
    )


def _fallback_acquisition_hash(
    *,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    lineage: tuple[str, ...],
    transitions: tuple[str, ...],
    invalidation_chart_hash: str,
    composition_result_hash: str,
    metadata: Mapping[str, Any],
) -> str:
    return content_hash(
        {
            "schema": SIGNED_WORK_AUTHORIZED_ACQUISITION_V2_SCHEMA,
            "campaign_binding_hash": binding.binding_hash,
            "generation_index": generation_index,
            "current_mask_hash": current_mask.mask_hash,
            "committed_mask_lineage_hashes": lineage,
            "committed_mask_transition_receipt_hashes": transitions,
            "invalidation_chart_hash": invalidation_chart_hash,
            "composition_result_hash": composition_result_hash,
            "metadata": metadata,
        },
        prefix="partition-free-v1-signed-work-authorized-acquisition-",
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkAcquisitionGenerationReceiptV2:
    campaign_contract_hash: str
    campaign_binding_hash: str
    registration_hash: str
    generation_index: int
    current_mask_hash: str
    route_axis_hash: str
    acquisition_hash: str
    invalidation_chart_hash: str
    composition_result_hash: str
    domain_hash: str
    certification_scope_hash: str
    six_role_authorization_hash: str
    six_role_fallback_authorization_hash: str
    six_role_certification_receipt_hash: str
    six_role_decision_hash: str
    six_role_decision_design_hash: str
    six_role_C_design_hash: str
    six_role_C_progress_receipt_hash: str
    selected_coordinate_hash: str
    ordered_candidate_ids: tuple[str, ...]
    ordered_coordinate_hashes: tuple[str, ...]
    ordered_branch_endpoint_coordinates: tuple[tuple[str, int], ...]
    ordered_packed_action_hashes: tuple[str, ...]
    ordered_six_role_interval_hashes: tuple[str, ...]
    queued_physical_candidate_identity_hashes: tuple[str, ...]
    blocked_unresolved_coordinate_hash: str | None
    refresh_reason: str
    logical_sequence: int
    proposal_only: bool = True
    complete_queue_frozen_before_K_O: bool = True
    resolved_prefix_queue_frozen_before_K_O: bool = True
    fallback_exhaustion_status: str = "fully_resolved_no_more_candidate"
    same_generation_paired_O_fallback_without_recomposition_authority: bool = True
    paired_bce_oracle_required: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    candidate_freeze_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    claim_execution_certified: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SIGNED_WORK_ACQUISITION_RECEIPT_V2_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "registration_hash",
            "current_mask_hash",
            "route_axis_hash",
            "acquisition_hash",
            "invalidation_chart_hash",
            "composition_result_hash",
            "domain_hash",
            "certification_scope_hash",
            "six_role_authorization_hash",
            "six_role_fallback_authorization_hash",
            "six_role_certification_receipt_hash",
            "six_role_decision_hash",
            "six_role_decision_design_hash",
            "six_role_C_design_hash",
            "six_role_C_progress_receipt_hash",
            "selected_coordinate_hash",
            "refresh_reason",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        object.__setattr__(
            self, "generation_index", _nonnegative_int(self.generation_index, "generation_index")
        )
        object.__setattr__(
            self, "logical_sequence", _nonnegative_int(self.logical_sequence, "logical_sequence")
        )
        candidate_ids = tuple(
            _identity(value, "ordered candidate id") for value in self.ordered_candidate_ids
        )
        coordinate_hashes = tuple(
            _identity(value, "ordered coordinate hash")
            for value in self.ordered_coordinate_hashes
        )
        packed_hashes = tuple(
            _identity(value, "ordered packed action hash")
            for value in self.ordered_packed_action_hashes
        )
        interval_hashes = tuple(
            _identity(value, "ordered six-role interval hash")
            for value in self.ordered_six_role_interval_hashes
        )
        coordinates = tuple(
            (_branch_value(branch, "ordered branch"), _positive_int(endpoint, "ordered endpoint"))
            for branch, endpoint in self.ordered_branch_endpoint_coordinates
        )
        physical = tuple(
            _identity(value, "queued physical candidate identity")
            for value in self.queued_physical_candidate_identity_hashes
        )
        if (
            not candidate_ids
            or coordinate_hashes[0] != self.selected_coordinate_hash
            or len(coordinate_hashes) != len(candidate_ids)
            or len(packed_hashes) != len(candidate_ids)
            or len(interval_hashes) != len(candidate_ids)
            or len(coordinates) != len(candidate_ids)
            or len(physical) != len(candidate_ids)
            or len(set(physical)) != len(physical)
        ):
            raise ValueError("signed-work fallback receipt queue is inconsistent")
        if self.blocked_unresolved_coordinate_hash is not None:
            _identity(
                self.blocked_unresolved_coordinate_hash,
                "blocked unresolved coordinate hash",
            )
        if self.fallback_exhaustion_status not in {
            "blocked_on_more_C",
            "fully_resolved_no_more_candidate",
        }:
            raise ValueError("signed-work fallback receipt exhaustion status is unsupported")
        if (
            self.blocked_unresolved_coordinate_hash is None
            and self.fallback_exhaustion_status != "fully_resolved_no_more_candidate"
        ) or (
            self.blocked_unresolved_coordinate_hash is not None
            and self.fallback_exhaustion_status != "blocked_on_more_C"
        ):
            raise ValueError("signed-work fallback receipt exhaustion status contradicts blocker")
        firewall = (
            self.proposal_only,
            self.complete_queue_frozen_before_K_O,
            self.resolved_prefix_queue_frozen_before_K_O,
            self.same_generation_paired_O_fallback_without_recomposition_authority,
            self.paired_bce_oracle_required,
            not self.calibration_K_open_authority,
            not self.paired_oracle_O_open_authority,
            not self.candidate_freeze_authority,
            not self.irreversible_decision_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.claim_execution_certified,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("signed-work fallback acquisition receipt crossed its firewall")
        if self.schema != SIGNED_WORK_ACQUISITION_RECEIPT_V2_SCHEMA:
            raise ValueError("unsupported signed-work fallback acquisition-receipt schema")
        object.__setattr__(self, "ordered_candidate_ids", candidate_ids)
        object.__setattr__(self, "ordered_coordinate_hashes", coordinate_hashes)
        object.__setattr__(self, "ordered_branch_endpoint_coordinates", coordinates)
        object.__setattr__(self, "ordered_packed_action_hashes", packed_hashes)
        object.__setattr__(self, "ordered_six_role_interval_hashes", interval_hashes)
        object.__setattr__(
            self,
            "queued_physical_candidate_identity_hashes",
            physical,
        )
        object.__setattr__(self, "metadata", _freeze(self.metadata, "signed-work fallback receipt metadata"))

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self._payload(),
            prefix="partition-free-v1-signed-work-acquisition-generation-",
        )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "campaign_contract_hash",
                "campaign_binding_hash",
                "registration_hash",
                "generation_index",
                "current_mask_hash",
                "route_axis_hash",
                "acquisition_hash",
                "invalidation_chart_hash",
                "composition_result_hash",
                "domain_hash",
                "certification_scope_hash",
                "six_role_authorization_hash",
                "six_role_fallback_authorization_hash",
                "six_role_certification_receipt_hash",
                "six_role_decision_hash",
                "six_role_decision_design_hash",
                "six_role_C_design_hash",
                "six_role_C_progress_receipt_hash",
                "selected_coordinate_hash",
                "ordered_candidate_ids",
                "ordered_coordinate_hashes",
                "ordered_branch_endpoint_coordinates",
                "ordered_packed_action_hashes",
                "ordered_six_role_interval_hashes",
                "queued_physical_candidate_identity_hashes",
                "blocked_unresolved_coordinate_hash",
                "refresh_reason",
                "logical_sequence",
                "proposal_only",
                "complete_queue_frozen_before_K_O",
                "resolved_prefix_queue_frozen_before_K_O",
                "fallback_exhaustion_status",
                "same_generation_paired_O_fallback_without_recomposition_authority",
                "paired_bce_oracle_required",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "candidate_freeze_authority",
                "irreversible_decision_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
                "claim_execution_certified",
                "metadata",
                "schema",
            )
        }


@dataclass(frozen=True, slots=True)
class V1SignedWorkAuthorizedAcquisitionV2:
    binding: V1OptionACampaignBinding
    generation_index: int
    current_mask: HardMask
    committed_mask_lineage_hashes: tuple[str, ...]
    committed_mask_transition_receipt_hashes: tuple[str, ...]
    composition: V1SignedWorkCandidateCompositionResultV2
    receipt: V1SignedWorkAcquisitionGenerationReceiptV2
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SIGNED_WORK_AUTHORIZED_ACQUISITION_V2_SCHEMA

    def __post_init__(self) -> None:
        if type(self.binding) is not V1OptionACampaignBinding:
            raise TypeError("signed-work fallback acquisition requires an exact binding")
        if type(self.current_mask) is not HardMask:
            raise TypeError("signed-work fallback acquisition requires an exact current mask")
        if type(self.composition) is not V1SignedWorkCandidateCompositionResultV2:
            raise TypeError("signed-work fallback acquisition requires its exact composition")
        if type(self.receipt) is not V1SignedWorkAcquisitionGenerationReceiptV2:
            raise TypeError("signed-work fallback acquisition requires its exact receipt")
        self.composition.assert_integrity()
        generation = _nonnegative_int(self.generation_index, "generation_index")
        lineage = tuple(
            _identity(value, "mask lineage hash")
            for value in self.committed_mask_lineage_hashes
        )
        transitions = tuple(
            _identity(value, "mask transition hash")
            for value in self.committed_mask_transition_receipt_hashes
        )
        current_hash = _fresh_mask_hash(self.current_mask, "signed-work current mask")
        if (
            not lineage
            or lineage[-1] != current_hash
            or len(transitions) != len(lineage) - 1
            or len(set(lineage)) != len(lineage)
            or len(set(transitions)) != len(transitions)
        ):
            raise ValueError("signed-work fallback acquisition has incomplete ancestry")
        composition = self.composition
        receipt = self.receipt
        contract = self.binding.contract
        if (
            contract.route_axis_hash != self.current_mask.route_axis_hash
            or composition.parent_mask_hash != self.current_mask.mask_hash
            or composition.route_axis_hash != self.current_mask.route_axis_hash
            or composition.route_count != self.current_mask.route_count
            or any(
                value.generation_index != generation
                for value in composition.action_numeric_states
            )
        ):
            raise ValueError("signed-work fallback acquisition differs from generation")
        if (
            receipt.campaign_contract_hash != contract.campaign_contract_hash
            or receipt.campaign_binding_hash != self.binding.binding_hash
            or receipt.registration_hash != self.binding.registration.registration_hash
            or receipt.generation_index != generation
            or receipt.current_mask_hash != self.current_mask.mask_hash
            or receipt.route_axis_hash != self.current_mask.route_axis_hash
            or receipt.composition_result_hash != composition.result_hash
            or receipt.domain_hash != composition.domain_hash
            or receipt.certification_scope_hash != composition.certification_scope_hash
            or receipt.six_role_authorization_hash != composition.six_role_authorization_hash
            or receipt.six_role_fallback_authorization_hash
            != composition.six_role_fallback_authorization_hash
            or receipt.six_role_certification_receipt_hash
            != composition.six_role_certification_receipt_hash
            or receipt.six_role_decision_hash != composition.six_role_decision_hash
            or receipt.six_role_decision_design_hash
            != composition.six_role_decision_design_hash
            or receipt.six_role_C_design_hash != composition.six_role_C_design_hash
            or receipt.six_role_C_progress_receipt_hash
            != composition.six_role_C_progress_receipt_hash
            or receipt.selected_coordinate_hash != composition.queue_coordinate_hashes[0]
            or receipt.ordered_candidate_ids
            != tuple(value.candidate_id for value in composition.proposals)
            or receipt.ordered_coordinate_hashes != composition.queue_coordinate_hashes
            or receipt.ordered_branch_endpoint_coordinates
            != composition.queue_branch_endpoint_coordinates
            or receipt.ordered_packed_action_hashes
            != composition.queue_packed_action_hashes
            or receipt.ordered_six_role_interval_hashes
            != composition.queue_six_role_interval_hashes
            or receipt.queued_physical_candidate_identity_hashes
            != composition.queued_physical_candidate_identity_hashes
            or receipt.blocked_unresolved_coordinate_hash
            != composition.blocked_unresolved_coordinate_hash
            or receipt.invalidation_chart_hash != self.invalidation_chart_hash
            or receipt.acquisition_hash != self.acquisition_hash
        ):
            raise ValueError("signed-work fallback acquisition receipt differs")
        if self.schema != SIGNED_WORK_AUTHORIZED_ACQUISITION_V2_SCHEMA:
            raise ValueError("unsupported signed-work fallback acquisition schema")
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "committed_mask_lineage_hashes", lineage)
        object.__setattr__(self, "committed_mask_transition_receipt_hashes", transitions)
        object.__setattr__(self, "metadata", _freeze(self.metadata, "signed-work fallback acquisition metadata"))

    @property
    def invalidation_chart_hash(self) -> str:
        return _fallback_invalidation_chart_hash(
            generation_index=self.generation_index,
            current_mask=self.current_mask,
            composition=self.composition,
        )

    @property
    def acquisition_hash(self) -> str:
        return _fallback_acquisition_hash(
            binding=self.binding,
            generation_index=self.generation_index,
            current_mask=self.current_mask,
            lineage=self.committed_mask_lineage_hashes,
            transitions=self.committed_mask_transition_receipt_hashes,
            invalidation_chart_hash=self.invalidation_chart_hash,
            composition_result_hash=self.composition.result_hash,
            metadata=self.metadata,
        )

    @property
    def queue(self) -> tuple[V1SignedWorkCandidateProposal, ...]:
        return self.composition.proposals

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def mask_commit_authority(self) -> bool:
        return False


def _selected_certificate(
    *,
    certification_receipt: V1SignedWorkSixRoleCertificationReceipt,
    coordinate: V1SignedWorkSixRoleEndpointCoordinate,
) -> V1SignedWorkSixRoleEndpointCertificate:
    for certificate in certification_receipt.endpoint_certificates:
        if certificate.coordinate.coordinate_hash == coordinate.coordinate_hash:
            return certificate
    raise ValueError("selected six-role coordinate lacks a matching certificate")


def _action_state_for_coordinate(
    *,
    queue_index: int,
    authorization_hash: str,
    six_role_C_design: V1SignedWorkSixRoleCDesign,
    progress_receipt: V1SignedWorkSixRoleProgressReceipt,
    certification_receipt: V1SignedWorkSixRoleCertificationReceipt,
    decision: V1SignedWorkSixRoleDecision,
    decision_design: V1SignedWorkSixRoleDecisionDesign,
    coordinate: V1SignedWorkSixRoleEndpointCoordinate,
    current_mask: HardMask,
    current_hash: str,
    generation: int,
    old_chart_validity_claim: bool,
) -> V1SignedWorkActionNumericState:
    branch_design = six_role_C_design.branch_design(coordinate.branch)
    branch_bundle = branch_design.bridge_branch
    ladder = branch_bundle.ladder
    if ladder is None:
        raise ValueError("selected six-role branch has no frozen C ladder")
    ladder.assert_integrity()
    endpoint_offset = coordinate.endpoint_index - 1
    if endpoint_offset < 0 or endpoint_offset >= len(ladder.actions):
        raise ValueError("selected six-role endpoint left the frozen ladder")
    action = ladder.actions[endpoint_offset]
    crosswalk = branch_bundle.crosswalks[endpoint_offset]
    certificate = _selected_certificate(
        certification_receipt=certification_receipt,
        coordinate=coordinate,
    )
    if (
        action.action_hash != coordinate.action_hash
        or action.action_hash != crosswalk.frozen_action_hash
        or crosswalk.receipt_hash != coordinate.crosswalk_receipt_hash
        or crosswalk.hard_replay_endpoint_receipt_hash
        != coordinate.hard_replay_endpoint_receipt_hash
        or crosswalk.canonical_candidate_identity_hash
        != coordinate.canonical_candidate_identity_hash
        or certificate.direct_D_status is not V1SignedWorkIntervalStatus.CERTIFIED
        or certificate.admission_basis is None
    ):
        raise ValueError("selected six-role source coordinates do not close")
    packed = action.packed_removal_action
    if type(packed) is not V1PackedRemovalAction:
        raise TypeError("selected six-role action must be a packed hard-mask action")
    child = packed.resulting_mask
    removed_count, removed_hash = child.removed_index_count_and_hash_from(current_mask)
    if (
        packed.action_hash != crosswalk.packed_action_hash
        or child.mask_hash != crosswalk.resulting_mask_hash
        or removed_count != packed.removed_count
        or removed_hash != packed.removed_route_index_hash
        or removed_count != coordinate.cumulative_removed_count
        or removed_count <= 0
    ):
        raise ValueError("selected six-role packed action differs from its bridge")
    values = {
        "six_role_authorization_hash": authorization_hash,
        "selected_coordinate_hash": coordinate.coordinate_hash,
        "packed_action_hash": packed.action_hash,
        "representative_branch": coordinate.branch.value,
        "representative_endpoint_index": coordinate.endpoint_index,
        "f_native_interval": certificate.f_native_interval,
        "q_j_interval": certificate.q_j_interval,
        "r_half_interval": certificate.r_half_interval,
        "r_one_interval": certificate.r_one_interval,
        "d_half_interval": certificate.d_half_interval,
        "d_one_interval": certificate.d_one_interval,
        "chart_half_interval": certificate.chart_half_interval,
        "chart_one_interval": certificate.chart_one_interval,
        "tau_l": certificate.tau_l,
        "direct_D_status": certificate.direct_D_status.value,
        "chart_transport_status": certificate.chart_transport_status.value,
        "admission_basis": certificate.admission_basis,
    }
    interval_hash = content_hash(
        _six_role_interval_payload(values=values),
        prefix="partition-free-v1-signed-work-six-role-interval-",
    )
    resource_value = float(coordinate.cumulative_removed_count)
    resource_units = "cumulative_removed_route_count"
    resource_source_hash = coordinate.coordinate_hash
    resource_hash = content_hash(
        _resource_payload(
            selected_coordinate_hash=coordinate.coordinate_hash,
            packed_action_hash=packed.action_hash,
            parent_mask_hash=current_hash,
            resulting_mask_hash=child.mask_hash,
            registered_resource=resource_value,
            resource_units=resource_units,
            resource_source_hash=resource_source_hash,
        ),
        prefix="partition-free-v1-signed-work-resource-",
    )
    return V1SignedWorkActionNumericState(
        _issuer_token=_STATE_TOKEN,
        queue_index=queue_index,
        representative_branch=coordinate.branch.value,
        representative_endpoint_index=coordinate.endpoint_index,
        representative_frozen_action_hash=action.action_hash,
        pre_C_priority=coordinate.pre_C_priority,
        endpoint_label=coordinate.endpoint_label,
        selected_coordinate_hash=coordinate.coordinate_hash,
        source_crosswalk_receipt_hash=coordinate.crosswalk_receipt_hash,
        source_hard_replay_endpoint_receipt_hash=(
            coordinate.hard_replay_endpoint_receipt_hash
        ),
        parent_mask=current_mask,
        packed_action=packed,
        parent_mask_hash=current_hash,
        resulting_mask_hash=child.mask_hash,
        packed_action_hash=packed.action_hash,
        route_axis_hash=current_mask.route_axis_hash,
        route_count=current_mask.route_count,
        removed_count=removed_count,
        removed_route_index_hash=removed_hash,
        registered_resource=resource_value,
        resource_units=resource_units,
        resource_source_hash=resource_source_hash,
        resource_receipt_hash=resource_hash,
        f_native_interval=certificate.f_native_interval,
        q_j_interval=certificate.q_j_interval,
        r_half_interval=certificate.r_half_interval,
        r_one_interval=certificate.r_one_interval,
        d_half_interval=certificate.d_half_interval,
        d_one_interval=certificate.d_one_interval,
        chart_half_interval=certificate.chart_half_interval,
        chart_one_interval=certificate.chart_one_interval,
        tau_l=certificate.tau_l,
        direct_D_status=certificate.direct_D_status.value,
        chart_transport_status=certificate.chart_transport_status.value,
        admission_basis=str(certificate.admission_basis),
        six_role_interval_hash=interval_hash,
        six_role_authorization_hash=authorization_hash,
        six_role_certification_receipt_hash=certification_receipt.receipt_hash,
        six_role_decision_hash=decision.decision_hash,
        six_role_decision_design_hash=decision_design.design_hash,
        six_role_C_design_hash=six_role_C_design.design_hash,
        six_role_C_progress_receipt_hash=progress_receipt.receipt_hash,
        bridge_family_bundle_hash=six_role_C_design.bridge_family.bundle_hash,
        native_fiber_axis_receipt_hash=six_role_C_design.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=six_role_C_design.native_fiber_measure_hash,
        prefrozen_ladder_hash=ladder.ladder_hash,
        generation_index=generation,
        single_endpoint_proposal_authority=True,
        source_replayed_six_role_authorization=True,
        direct_D_owns_finite_response_admission=True,
        QR_transport_is_label_not_hidden_C_veto=True,
        old_chart_validity_claim=old_chart_validity_claim,
        relinearization_required_after_oracle_accept=True,
        proposal_only=True,
        paired_bce_oracle_required=True,
        calibration_K_open_authority=False,
        paired_oracle_O_open_authority=False,
        candidate_freeze_authority=False,
        bce_oracle_authority=False,
        irreversible_decision_authority=False,
        mask_commit_authority=False,
        terminal_authority=False,
        pareto_authority=False,
        execution_authority=False,
    )


def issue_v1_signed_work_authorized_acquisition(
    *,
    six_role_C_design: V1SignedWorkSixRoleCDesign,
    progress_receipt: V1SignedWorkSixRoleProgressReceipt,
    certification_receipt: V1SignedWorkSixRoleCertificationReceipt,
    decision: V1SignedWorkSixRoleDecision,
    decision_design: V1SignedWorkSixRoleDecisionDesign,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
    live_chart_identity_hash: str,
    live_route_axis_hash: str,
    refresh_reason: str = "fresh_current_mask_signed_work_six_role_relinearization",
    logical_sequence: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> V1SignedWorkAuthorizedAcquisition:
    """Detach one source-replayed six-role proposal for controller execution."""

    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("signed-work acquisition requires an exact campaign binding")
    if type(current_mask) is not HardMask:
        raise TypeError("signed-work acquisition requires an exact current mask")
    generation = _nonnegative_int(generation_index, "generation_index")
    sequence = _nonnegative_int(logical_sequence, "logical_sequence")
    reason = _identity(refresh_reason, "refresh_reason")
    lineage = tuple(committed_mask_lineage_hashes)
    transitions = tuple(committed_mask_transition_receipt_hashes)
    current_hash = _fresh_mask_hash(current_mask, "signed-work current mask")
    if (
        not lineage
        or lineage[-1] != current_hash
        or len(transitions) != len(lineage) - 1
        or binding.contract.route_axis_hash != current_mask.route_axis_hash
    ):
        raise ValueError("signed-work acquisition coordinates are stale")
    authorization = authorize_v1_signed_work_six_role_proposal(
        six_role_C_design=six_role_C_design,
        progress_receipt=progress_receipt,
        certification_receipt=certification_receipt,
        decision=decision,
        decision_design=decision_design,
        live_chart_identity_hash=live_chart_identity_hash,
        live_current_mask_hash=current_hash,
        live_route_axis_hash=live_route_axis_hash,
        live_proposal_generation=generation,
    )
    if type(authorization) is not V1SignedWorkSixRoleProposalAuthorization:
        raise TypeError("six-role source replay returned the wrong authorization type")
    authorization_hash = authorization.authorization_hash
    if (
        authorization.current_mask_hash != current_hash
        or authorization.route_axis_hash != current_mask.route_axis_hash
        or authorization.route_axis_hash != live_route_axis_hash
        or authorization.route_count != current_mask.route_count
        or authorization.proposal_generation != generation
        or authorization.authority_scope != SIGNED_WORK_SIX_ROLE_AUTHORITY_SCOPE
        or not authorization.paired_bce_oracle_still_required
        or authorization.mask_commit_authority
    ):
        raise ValueError("signed-work six-role authorization is stale or overbroad")
    coordinate = authorization.selected_coordinate
    branch_design = six_role_C_design.branch_design(coordinate.branch)
    branch_bundle = branch_design.bridge_branch
    ladder = branch_bundle.ladder
    if ladder is None:
        raise ValueError("selected six-role branch has no frozen C ladder")
    ladder.assert_integrity()
    endpoint_offset = coordinate.endpoint_index - 1
    if endpoint_offset < 0 or endpoint_offset >= len(ladder.actions):
        raise ValueError("selected six-role endpoint left the frozen ladder")
    action = ladder.actions[endpoint_offset]
    crosswalk = branch_bundle.crosswalks[endpoint_offset]
    certificate = _selected_certificate(
        certification_receipt=certification_receipt,
        coordinate=coordinate,
    )
    if (
        action.action_hash != coordinate.action_hash
        or action.action_hash != crosswalk.frozen_action_hash
        or crosswalk.receipt_hash != coordinate.crosswalk_receipt_hash
        or crosswalk.hard_replay_endpoint_receipt_hash
        != coordinate.hard_replay_endpoint_receipt_hash
        or crosswalk.canonical_candidate_identity_hash
        != coordinate.canonical_candidate_identity_hash
        or certificate.direct_D_status is not V1SignedWorkIntervalStatus.CERTIFIED
        or certificate.admission_basis != authorization.admission_basis
        or certificate.chart_transport_status is not authorization.chart_transport_status
    ):
        raise ValueError("selected six-role source coordinates do not close")
    packed = action.packed_removal_action
    if type(packed) is not V1PackedRemovalAction:
        raise TypeError("selected six-role action must be a packed hard-mask action")
    child = packed.resulting_mask
    removed_count, removed_hash = child.removed_index_count_and_hash_from(current_mask)
    if (
        packed.action_hash != crosswalk.packed_action_hash
        or child.mask_hash != crosswalk.resulting_mask_hash
        or removed_count != packed.removed_count
        or removed_hash != packed.removed_route_index_hash
        or removed_count != coordinate.cumulative_removed_count
        or removed_count <= 0
    ):
        raise ValueError("selected six-role packed action differs from its bridge")
    values = {
        "six_role_authorization_hash": authorization_hash,
        "selected_coordinate_hash": coordinate.coordinate_hash,
        "packed_action_hash": packed.action_hash,
        "representative_branch": coordinate.branch.value,
        "representative_endpoint_index": coordinate.endpoint_index,
        "f_native_interval": certificate.f_native_interval,
        "q_j_interval": certificate.q_j_interval,
        "r_half_interval": certificate.r_half_interval,
        "r_one_interval": certificate.r_one_interval,
        "d_half_interval": certificate.d_half_interval,
        "d_one_interval": certificate.d_one_interval,
        "chart_half_interval": certificate.chart_half_interval,
        "chart_one_interval": certificate.chart_one_interval,
        "tau_l": certificate.tau_l,
        "direct_D_status": certificate.direct_D_status.value,
        "chart_transport_status": certificate.chart_transport_status.value,
        "admission_basis": certificate.admission_basis,
    }
    interval_hash = content_hash(
        _six_role_interval_payload(values=values),
        prefix="partition-free-v1-signed-work-six-role-interval-",
    )
    resource_value = float(coordinate.cumulative_removed_count)
    resource_units = "cumulative_removed_route_count"
    resource_source_hash = coordinate.coordinate_hash
    resource_hash = content_hash(
        _resource_payload(
            selected_coordinate_hash=coordinate.coordinate_hash,
            packed_action_hash=packed.action_hash,
            parent_mask_hash=current_hash,
            resulting_mask_hash=child.mask_hash,
            registered_resource=resource_value,
            resource_units=resource_units,
            resource_source_hash=resource_source_hash,
        ),
        prefix="partition-free-v1-signed-work-resource-",
    )
    state = V1SignedWorkActionNumericState(
        _issuer_token=_STATE_TOKEN,
        queue_index=0,
        representative_branch=coordinate.branch.value,
        representative_endpoint_index=coordinate.endpoint_index,
        representative_frozen_action_hash=action.action_hash,
        pre_C_priority=coordinate.pre_C_priority,
        endpoint_label=coordinate.endpoint_label,
        selected_coordinate_hash=coordinate.coordinate_hash,
        source_crosswalk_receipt_hash=coordinate.crosswalk_receipt_hash,
        source_hard_replay_endpoint_receipt_hash=(
            coordinate.hard_replay_endpoint_receipt_hash
        ),
        parent_mask=current_mask,
        packed_action=packed,
        parent_mask_hash=current_hash,
        resulting_mask_hash=child.mask_hash,
        packed_action_hash=packed.action_hash,
        route_axis_hash=current_mask.route_axis_hash,
        route_count=current_mask.route_count,
        removed_count=removed_count,
        removed_route_index_hash=removed_hash,
        registered_resource=resource_value,
        resource_units=resource_units,
        resource_source_hash=resource_source_hash,
        resource_receipt_hash=resource_hash,
        f_native_interval=certificate.f_native_interval,
        q_j_interval=certificate.q_j_interval,
        r_half_interval=certificate.r_half_interval,
        r_one_interval=certificate.r_one_interval,
        d_half_interval=certificate.d_half_interval,
        d_one_interval=certificate.d_one_interval,
        chart_half_interval=certificate.chart_half_interval,
        chart_one_interval=certificate.chart_one_interval,
        tau_l=certificate.tau_l,
        direct_D_status=certificate.direct_D_status.value,
        chart_transport_status=certificate.chart_transport_status.value,
        admission_basis=str(certificate.admission_basis),
        six_role_interval_hash=interval_hash,
        six_role_authorization_hash=authorization_hash,
        six_role_certification_receipt_hash=certification_receipt.receipt_hash,
        six_role_decision_hash=decision.decision_hash,
        six_role_decision_design_hash=decision_design.design_hash,
        six_role_C_design_hash=six_role_C_design.design_hash,
        six_role_C_progress_receipt_hash=progress_receipt.receipt_hash,
        bridge_family_bundle_hash=six_role_C_design.bridge_family.bundle_hash,
        native_fiber_axis_receipt_hash=six_role_C_design.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=six_role_C_design.native_fiber_measure_hash,
        prefrozen_ladder_hash=ladder.ladder_hash,
        generation_index=generation,
        single_endpoint_proposal_authority=True,
        source_replayed_six_role_authorization=True,
        direct_D_owns_finite_response_admission=True,
        QR_transport_is_label_not_hidden_C_veto=True,
        old_chart_validity_claim=authorization.old_chart_validity_claim,
        relinearization_required_after_oracle_accept=True,
        proposal_only=True,
        paired_bce_oracle_required=True,
        calibration_K_open_authority=False,
        paired_oracle_O_open_authority=False,
        candidate_freeze_authority=False,
        bce_oracle_authority=False,
        irreversible_decision_authority=False,
        mask_commit_authority=False,
        terminal_authority=False,
        pareto_authority=False,
        execution_authority=False,
    )
    states = (state,)
    domain_hash = _domain_hash(states)
    certification_scope_hash = _certification_scope_hash(state)
    proposal = _issue_proposal(
        state=state,
        domain_hash=domain_hash,
        certification_scope_hash=certification_scope_hash,
    )
    provisional = {
        "action_numeric_states": states,
        "proposals": (proposal,),
        "domain_hash": domain_hash,
        "certification_scope_hash": certification_scope_hash,
        "six_role_authorization_hash": authorization_hash,
        "six_role_certification_receipt_hash": certification_receipt.receipt_hash,
        "six_role_decision_hash": decision.decision_hash,
        "six_role_decision_design_hash": decision_design.design_hash,
        "six_role_C_design_hash": six_role_C_design.design_hash,
        "six_role_C_progress_receipt_hash": progress_receipt.receipt_hash,
        "parent_mask_hash": current_hash,
        "route_axis_hash": current_mask.route_axis_hash,
        "route_count": current_mask.route_count,
        "queue_coordinate_hashes": (coordinate.coordinate_hash,),
        "queue_branch_endpoint_coordinates": (
            (coordinate.branch.value, coordinate.endpoint_index),
        ),
        "queue_packed_action_hashes": (packed.action_hash,),
        "queue_resulting_mask_hashes": (child.mask_hash,),
        "queue_six_role_interval_hashes": (interval_hash,),
        "queue_resource_receipt_hashes": (resource_hash,),
        "validated_action_count": 1,
        "proposal_count": 1,
        "composer_id": SIGNED_WORK_SINGLETON_COMPOSER_ID,
        "single_endpoint_materialized_exactly": True,
        "complete_queue_frozen_before_K_O": True,
        "filtering_performed": False,
        "reordering_performed": False,
        "claims_hard_global_optimum": False,
        "omitted_mask_optimality_authority": False,
        "proposal_only": True,
        "proposal_witness_only": True,
        "fresh_relinearization_required": True,
        "paired_bce_oracle_required": True,
        "candidate_freeze_authority": False,
        "bce_oracle_authority": False,
        "irreversible_decision_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
        "execution_authority": False,
    }
    shell = object.__new__(V1SignedWorkCandidateCompositionResult)
    for name, value in provisional.items():
        object.__setattr__(shell, name, value)
    object.__setattr__(shell, "schema", SIGNED_WORK_CANDIDATE_COMPOSITION_SCHEMA)
    selection_hash = content_hash(
        shell._selection_payload(),
        prefix="partition-free-v1-signed-work-candidate-selection-",
    )
    composition = V1SignedWorkCandidateCompositionResult(
        _issuer_token=_COMPOSITION_TOKEN,
        **provisional,
        selection_receipt_hash=selection_hash,
    )
    required_metadata = {
        "refresh_reason": reason,
        "logical_sequence": sequence,
        "generation_index": generation,
        "authority_scope": SIGNED_WORK_RUNTIME_AUTHORITY_SCOPE,
        "candidate_composition_mode": "canonical_signed_work_six_role_direct_D_single_endpoint",
        "complete_queue_frozen_before_K_O": True,
        "full_six_role_C_design_embedded": False,
        "raw_C_records_embedded": False,
        "record_retaining_receipts_embedded": False,
        "full_route_numeric_state_embedded": False,
        "single_endpoint_proposal_authority": True,
        "direct_D_owns_finite_response_admission": True,
        "QR_transport_is_label_not_hidden_C_veto": True,
        "old_chart_validity_claim": authorization.old_chart_validity_claim,
        "proposal_only": True,
        "paired_bce_oracle_required": True,
        "calibration_K_open_authority": False,
        "paired_oracle_O_open_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
    }
    caller_metadata = dict(metadata or {})
    collisions = sorted(set(required_metadata).intersection(caller_metadata))
    if collisions:
        raise ValueError(
            f"caller metadata shadows reserved signed-work fields: {collisions}"
        )
    core_metadata = _freeze(
        {**caller_metadata, **required_metadata},
        "signed-work acquisition metadata",
    )
    invalidation_hash = _invalidation_chart_hash(
        generation_index=generation,
        current_mask=current_mask,
        composition=composition,
    )
    acquisition_hash = _acquisition_hash(
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        lineage=lineage,
        transitions=transitions,
        invalidation_chart_hash=invalidation_hash,
        composition_result_hash=composition.result_hash,
        metadata=core_metadata,
    )
    receipt = V1SignedWorkAcquisitionGenerationReceipt(
        campaign_contract_hash=binding.contract.campaign_contract_hash,
        campaign_binding_hash=binding.binding_hash,
        registration_hash=binding.registration.registration_hash,
        generation_index=generation,
        current_mask_hash=current_hash,
        route_axis_hash=current_mask.route_axis_hash,
        acquisition_hash=acquisition_hash,
        invalidation_chart_hash=invalidation_hash,
        composition_result_hash=composition.result_hash,
        domain_hash=composition.domain_hash,
        certification_scope_hash=composition.certification_scope_hash,
        six_role_authorization_hash=composition.six_role_authorization_hash,
        six_role_certification_receipt_hash=(
            composition.six_role_certification_receipt_hash
        ),
        six_role_decision_hash=composition.six_role_decision_hash,
        six_role_decision_design_hash=composition.six_role_decision_design_hash,
        six_role_C_design_hash=composition.six_role_C_design_hash,
        six_role_C_progress_receipt_hash=composition.six_role_C_progress_receipt_hash,
        selected_coordinate_hash=coordinate.coordinate_hash,
        ordered_candidate_ids=(proposal.candidate_id,),
        ordered_coordinate_hashes=composition.queue_coordinate_hashes,
        ordered_branch_endpoint_coordinates=composition.queue_branch_endpoint_coordinates,
        ordered_packed_action_hashes=composition.queue_packed_action_hashes,
        ordered_six_role_interval_hashes=composition.queue_six_role_interval_hashes,
        refresh_reason=reason,
        logical_sequence=sequence,
        metadata={
            "queue_size": 1,
            "queue_resource_values": (proposal.resource_gain,),
            "selected_coordinate_hash": coordinate.coordinate_hash,
            "admission_basis": state.admission_basis,
            "chart_transport_status": state.chart_transport_status,
            "authority_scope": SIGNED_WORK_RUNTIME_AUTHORITY_SCOPE,
        },
    )
    return V1SignedWorkAuthorizedAcquisition(
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        committed_mask_lineage_hashes=lineage,
        committed_mask_transition_receipt_hashes=transitions,
        composition=composition,
        receipt=receipt,
        metadata=core_metadata,
    )


def issue_v1_signed_work_deferred_fallback_plan(
    *,
    six_role_C_design: V1SignedWorkSixRoleCDesign,
    progress_receipt: V1SignedWorkSixRoleProgressReceipt,
    certification_receipt: V1SignedWorkSixRoleCertificationReceipt,
    decision: V1SignedWorkSixRoleDecision,
    decision_design: V1SignedWorkSixRoleDecisionDesign,
    branch_sources: Sequence[V1SignedWorkDeferredBranchSource],
    binding: V1OptionACampaignBinding,
    generation_index: int,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_route_axis_hash: str,
) -> V1SignedWorkDeferredFallbackPlan:
    """Issue an exact resolved-prefix plan without materializing child masks.

    The six-role authority is source-replayed first.  Each authorized
    coordinate is then joined to its exact C crosswalk and immutable
    label-native device-shell action receipt.  Only receipt-scale identities
    and intervals leave this function; a later one-endpoint materializer must
    replay a live provider against the copied commitments.
    """

    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("deferred fallback planning requires an exact binding")
    generation = _nonnegative_int(generation_index, "generation_index")
    chart_hash = _identity(live_chart_identity_hash, "live_chart_identity_hash")
    mask_hash = _identity(live_current_mask_hash, "live_current_mask_hash")
    route_axis = _identity(live_route_axis_hash, "live_route_axis_hash")
    authorization = authorize_v1_signed_work_six_role_pre_O_fallback_queue(
        six_role_C_design=six_role_C_design,
        progress_receipt=progress_receipt,
        certification_receipt=certification_receipt,
        decision=decision,
        decision_design=decision_design,
        live_chart_identity_hash=chart_hash,
        live_current_mask_hash=mask_hash,
        live_route_axis_hash=route_axis,
        live_proposal_generation=generation,
    )
    if type(authorization) is not V1SignedWorkSixRoleFallbackQueueAuthorization:
        raise TypeError("six-role replay returned a noncanonical fallback authorization")
    if (
        authorization.chart_identity_hash != chart_hash
        or authorization.current_mask_hash != mask_hash
        or authorization.route_axis_hash != route_axis
        or authorization.proposal_generation != generation
        or binding.contract.route_axis_hash != route_axis
        or authorization.mask_commit_authority
        or not authorization.paired_bce_oracle_still_required
    ):
        raise ValueError("deferred fallback authorization is stale or overbroad")
    sources = tuple(branch_sources)
    if not sources or any(
        type(value) is not V1SignedWorkDeferredBranchSource for value in sources
    ):
        raise TypeError("deferred fallback planning requires exact branch sources")
    source_by_branch: dict[V1RelaxedKKTBranch, V1SignedWorkDeferredBranchSource] = {}
    for source_binding in sources:
        if source_binding.branch in source_by_branch:
            raise ValueError("deferred fallback branch sources contain duplicates")
        source_by_branch[source_binding.branch] = source_binding
    if set(source_by_branch) != set(six_role_C_design.branches):
        raise ValueError("deferred fallback sources do not cover the frozen C branches")

    # Replay every branch source, including one that may disappear after exact
    # cross-branch physical deduplication.  This prevents an incomplete source
    # family from borrowing another branch's action identity later.
    for branch in six_role_C_design.branches:
        branch_design = six_role_C_design.branch_design(branch)
        bridge_branch = branch_design.bridge_branch
        source_binding = source_by_branch[branch]
        provider_receipt = source_binding.packed_direction_receipt
        source_ladder = provider_receipt.action_axis_registration.source_shell_ladder
        if (
            source_ladder.receipt_hash
            != bridge_branch.receipt.shell_ladder_receipt_hash
            or source_ladder.parent_mask_hash != authorization.current_mask_hash
            or source_ladder.route_axis_hash != authorization.route_axis_hash
            or source_ladder.route_count != authorization.route_count
            or len(source_ladder.actions)
            != bridge_branch.receipt.complete_source_endpoint_count
            or provider_receipt.parent_mask_hash != authorization.current_mask_hash
            or provider_receipt.route_axis_hash != authorization.route_axis_hash
            or provider_receipt.route_count != authorization.route_count
        ):
            raise ValueError("a deferred provider receipt names another source ladder")
        for crosswalk in bridge_branch.crosswalks:
            offset = crosswalk.endpoint_label - 1
            if offset < 0 or offset >= len(source_ladder.actions):
                raise ValueError("a C crosswalk left its device-shell action axis")
            action_receipt = source_ladder.actions[offset]
            if (
                action_receipt.receipt_hash != crosswalk.shell_action_receipt_hash
                or action_receipt.cumulative_removed_count
                != crosswalk.cumulative_removed_count
                or action_receipt.packed_action_hash != crosswalk.packed_action_hash
                or action_receipt.resulting_mask_hash != crosswalk.resulting_mask_hash
                or action_receipt.removed_route_index_hash
                != crosswalk.removed_route_index_hash
            ):
                raise ValueError("a deferred C crosswalk changed its shell action")

    authorization_hash = authorization.authorization_hash
    descriptors: list[V1SignedWorkCompactFallbackDescriptor] = []
    for queue_index, coordinate in enumerate(authorization.fallback_coordinates):
        branch_design = six_role_C_design.branch_design(coordinate.branch)
        bridge_branch = branch_design.bridge_branch
        crosswalk_offset = coordinate.endpoint_index - 1
        if crosswalk_offset < 0 or crosswalk_offset >= len(bridge_branch.crosswalks):
            raise ValueError("a deferred coordinate left its C crosswalk axis")
        crosswalk = bridge_branch.crosswalks[crosswalk_offset]
        source_binding = source_by_branch[coordinate.branch]
        provider_receipt = source_binding.packed_direction_receipt
        action_axis = provider_receipt.action_axis_registration
        source_ladder = action_axis.source_shell_ladder
        action_offset = coordinate.endpoint_label - 1
        if action_offset < 0 or action_offset >= len(source_ladder.actions):
            raise ValueError("a deferred coordinate left its source action axis")
        action_receipt = source_ladder.actions[action_offset]
        certificate = _selected_certificate(
            certification_receipt=certification_receipt,
            coordinate=coordinate,
        )
        if (
            coordinate.action_hash != crosswalk.frozen_action_hash
            or coordinate.crosswalk_receipt_hash != crosswalk.receipt_hash
            or coordinate.hard_replay_endpoint_receipt_hash
            != crosswalk.hard_replay_endpoint_receipt_hash
            or coordinate.canonical_candidate_identity_hash
            != crosswalk.canonical_candidate_identity_hash
            or coordinate.cumulative_removed_count
            != crosswalk.cumulative_removed_count
            or action_receipt.receipt_hash != crosswalk.shell_action_receipt_hash
            or action_receipt.packed_action_hash != crosswalk.packed_action_hash
            or certificate.certificate_hash
            != authorization.fallback_certificate_hashes[queue_index]
            or certificate.direct_D_status
            is not V1SignedWorkIntervalStatus.CERTIFIED
            or certificate.admission_basis
            != authorization.fallback_admission_bases[queue_index]
            or certificate.chart_transport_status
            is not authorization.fallback_chart_transport_statuses[queue_index]
            or coordinate.canonical_candidate_identity_hash
            != authorization.queued_physical_candidate_identity_hashes[queue_index]
        ):
            raise ValueError("a deferred fallback coordinate failed exact source replay")
        interval_values = {
            "six_role_authorization_hash": authorization_hash,
            "selected_coordinate_hash": coordinate.coordinate_hash,
            "packed_action_hash": action_receipt.packed_action_hash,
            "representative_branch": coordinate.branch.value,
            "representative_endpoint_index": coordinate.endpoint_index,
            "f_native_interval": certificate.f_native_interval,
            "q_j_interval": certificate.q_j_interval,
            "r_half_interval": certificate.r_half_interval,
            "r_one_interval": certificate.r_one_interval,
            "d_half_interval": certificate.d_half_interval,
            "d_one_interval": certificate.d_one_interval,
            "chart_half_interval": certificate.chart_half_interval,
            "chart_one_interval": certificate.chart_one_interval,
            "tau_l": certificate.tau_l,
            "direct_D_status": certificate.direct_D_status.value,
            "chart_transport_status": certificate.chart_transport_status.value,
            "admission_basis": certificate.admission_basis,
        }
        interval_hash = content_hash(
            _six_role_interval_payload(values=interval_values),
            prefix="partition-free-v1-signed-work-six-role-interval-",
        )
        bridge_receipt = bridge_branch.receipt
        descriptors.append(
            V1SignedWorkCompactFallbackDescriptor(
                _issuer_token=_COMPACT_DESCRIPTOR_TOKEN,
                queue_index=queue_index,
                coordinate_hash=coordinate.coordinate_hash,
                physical_candidate_identity_hash=(
                    coordinate.canonical_candidate_identity_hash
                ),
                pre_C_priority=coordinate.pre_C_priority,
                branch=coordinate.branch,
                endpoint_index=coordinate.endpoint_index,
                endpoint_label=coordinate.endpoint_label,
                certificate_hash=certificate.certificate_hash,
                direct_D_status=certificate.direct_D_status.value,
                admission_basis=str(certificate.admission_basis),
                chart_transport_status=certificate.chart_transport_status.value,
                old_chart_validity_claim=(
                    certificate.admission_basis == "chart_certified"
                ),
                f_native_interval=certificate.f_native_interval,
                q_j_interval=certificate.q_j_interval,
                r_half_interval=certificate.r_half_interval,
                r_one_interval=certificate.r_one_interval,
                d_half_interval=certificate.d_half_interval,
                d_one_interval=certificate.d_one_interval,
                chart_half_interval=certificate.chart_half_interval,
                chart_one_interval=certificate.chart_one_interval,
                tau_l=certificate.tau_l,
                six_role_interval_hash=interval_hash,
                frozen_action_hash=coordinate.action_hash,
                crosswalk_receipt_hash=crosswalk.receipt_hash,
                crosswalk_materialization_receipt_hash=(
                    crosswalk.materialization_receipt_hash
                ),
                canonical_candidate_occurrence=(
                    crosswalk.canonical_candidate_occurrence
                ),
                cross_branch_exact_deduplication=(
                    crosswalk.cross_branch_exact_deduplication
                ),
                hard_replay_endpoint_receipt_hash=(
                    coordinate.hard_replay_endpoint_receipt_hash
                ),
                source_action_receipt=action_receipt,
                source_action_receipt_hash=action_receipt.receipt_hash,
                shell_count=action_receipt.shell_count,
                cumulative_removed_count=action_receipt.cumulative_removed_count,
                packed_action_hash=action_receipt.packed_action_hash,
                resulting_mask_packed_sha256=(
                    action_receipt.resulting_mask_packed_sha256
                ),
                resulting_mask_hash=action_receipt.resulting_mask_hash,
                removed_route_index_hash=action_receipt.removed_route_index_hash,
                branch_bridge_receipt_hash=bridge_receipt.receipt_hash,
                bridge_shell_execution_hash=bridge_receipt.shell_execution_hash,
                bridge_shell_execution_receipt_hash=(
                    bridge_receipt.shell_execution_receipt_hash
                ),
                bridge_shell_branch_identity_receipt_hash=(
                    bridge_receipt.shell_branch_identity_receipt_hash
                ),
                bridge_hard_replay_branch_receipt_hash=(
                    bridge_receipt.hard_replay_branch_receipt_hash
                ),
                bridge_coordinates_hash=bridge_receipt.coordinates_hash,
                bridge_design_hash=bridge_receipt.design_hash,
                provider_source_commitment_hash=(
                    source_binding.source_commitment_hash
                ),
                provider_receipt_hash=provider_receipt.receipt_hash,
                provider_design_hash=provider_receipt.design_hash,
                provider_policy_hash=provider_receipt.policy_hash,
                provider_forecast_hash=provider_receipt.forecast_hash,
                provider_backend_kind=provider_receipt.backend_kind,
                provider_device_identity=provider_receipt.device_identity,
                action_axis_registration_hash=action_axis.registration_hash,
                prefix_rank_receipt_hash=provider_receipt.prefix_rank_receipt_hash,
                endpoint_query_hash=provider_receipt.endpoint_query_hash,
                resulting_label_commitment=(
                    provider_receipt.resulting_label_commitment
                ),
                source_shell_ladder_receipt_hash=source_ladder.receipt_hash,
                source_ladder_hash=source_ladder.source_ladder_hash,
                source_label_commitment=source_ladder.label_commitment,
                source_eligibility_commitment=(
                    source_ladder.eligibility_commitment
                ),
                source_block_capacity_hash=source_ladder.block_capacity_hash,
                chart_identity_hash=authorization.chart_identity_hash,
                current_mask_hash=authorization.current_mask_hash,
                route_axis_hash=authorization.route_axis_hash,
                route_count=authorization.route_count,
                proposal_generation=authorization.proposal_generation,
                six_role_fallback_authorization_hash=authorization_hash,
                six_role_certification_receipt_hash=(
                    certification_receipt.receipt_hash
                ),
                six_role_decision_hash=decision.decision_hash,
                six_role_decision_design_hash=decision_design.design_hash,
                six_role_C_design_hash=six_role_C_design.design_hash,
                six_role_C_progress_receipt_hash=progress_receipt.receipt_hash,
                bridge_family_bundle_hash=six_role_C_design.bridge_family.bundle_hash,
                deferred_one_endpoint_materialization_required=True,
                complete_ladder_mask_materialization_forbidden=True,
                paired_bce_oracle_required=True,
                calibration_K_open_authority=False,
                paired_oracle_O_open_authority=False,
                mask_commit_authority=False,
                terminal_authority=False,
                pareto_authority=False,
                execution_authority=False,
            )
        )
    if not descriptors:
        raise ValueError("deferred fallback authorization produced no descriptors")
    descriptor_tuple = tuple(descriptors)
    blocked = authorization.blocked_unresolved_coordinate_hash is not None
    return V1SignedWorkDeferredFallbackPlan(
        _issuer_token=_DEFERRED_PLAN_TOKEN,
        campaign_contract_hash=binding.contract.campaign_contract_hash,
        campaign_binding_hash=binding.binding_hash,
        registration_hash=binding.registration.registration_hash,
        six_role_fallback_authorization_hash=authorization_hash,
        six_role_certification_receipt_hash=certification_receipt.receipt_hash,
        six_role_decision_hash=decision.decision_hash,
        six_role_decision_design_hash=decision_design.design_hash,
        six_role_C_design_hash=six_role_C_design.design_hash,
        six_role_C_progress_receipt_hash=progress_receipt.receipt_hash,
        bridge_family_bundle_hash=six_role_C_design.bridge_family.bundle_hash,
        chart_identity_hash=authorization.chart_identity_hash,
        current_mask_hash=authorization.current_mask_hash,
        route_axis_hash=authorization.route_axis_hash,
        route_count=authorization.route_count,
        proposal_generation=authorization.proposal_generation,
        selected_coordinate_hash=authorization.selected_coordinate.coordinate_hash,
        descriptors=descriptor_tuple,
        ordered_descriptor_hashes=tuple(
            value.descriptor_hash for value in descriptor_tuple
        ),
        ordered_coordinate_hashes=tuple(
            value.coordinate_hash for value in descriptor_tuple
        ),
        ordered_physical_candidate_identity_hashes=(
            authorization.queued_physical_candidate_identity_hashes
        ),
        ordered_pre_O_rung_registration_hashes=tuple(
            value.pre_O_rung_registration_hash for value in descriptor_tuple
        ),
        source_branch_commitment_hashes=tuple(
            (
                branch.value,
                source_by_branch[branch].source_commitment_hash,
            )
            for branch in six_role_C_design.branches
        ),
        blocked_unresolved_coordinate_hash=(
            authorization.blocked_unresolved_coordinate_hash
        ),
        fallback_exhaustion_status=authorization.fallback_exhaustion_status,
        authorized_descriptor_count=len(descriptor_tuple),
        resolved_prefix_descriptors_frozen_before_K_O=True,
        source_ladder_decision_fully_resolved_before_K_O=not blocked,
        blocked_on_more_C=blocked,
        fully_resolved_no_more_candidate=not blocked,
        deferred_one_endpoint_materialization_required=True,
        complete_ladder_mask_materialization_forbidden=True,
        endpoint_hard_masks_embedded=False,
        packed_child_payloads_embedded=False,
        provider_or_label_handle_embedded=False,
        source_authorized_filtering_performed=True,
        reordering_performed=False,
        omitted_mask_optimality_authority=False,
        paired_bce_oracle_required=True,
        calibration_K_open_authority=False,
        paired_oracle_O_open_authority=False,
        irreversible_decision_authority=False,
        mask_commit_authority=False,
        terminal_authority=False,
        pareto_authority=False,
        execution_authority=False,
    )


def issue_v1_signed_work_fallback_authorized_acquisition(
    *,
    six_role_C_design: V1SignedWorkSixRoleCDesign,
    progress_receipt: V1SignedWorkSixRoleProgressReceipt,
    certification_receipt: V1SignedWorkSixRoleCertificationReceipt,
    decision: V1SignedWorkSixRoleDecision,
    decision_design: V1SignedWorkSixRoleDecisionDesign,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
    live_chart_identity_hash: str,
    live_route_axis_hash: str,
    refresh_reason: str = "fresh_current_mask_signed_work_six_role_relinearization",
    logical_sequence: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> V1SignedWorkAuthorizedAcquisitionV2:
    """Detach a source-replayed pre-O fallback queue for controller execution."""

    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("signed-work fallback acquisition requires an exact binding")
    if type(current_mask) is not HardMask:
        raise TypeError("signed-work fallback acquisition requires an exact current mask")
    generation = _nonnegative_int(generation_index, "generation_index")
    sequence = _nonnegative_int(logical_sequence, "logical_sequence")
    reason = _identity(refresh_reason, "refresh_reason")
    lineage = tuple(committed_mask_lineage_hashes)
    transitions = tuple(committed_mask_transition_receipt_hashes)
    current_hash = _fresh_mask_hash(current_mask, "signed-work current mask")
    if (
        not lineage
        or lineage[-1] != current_hash
        or len(transitions) != len(lineage) - 1
        or binding.contract.route_axis_hash != current_mask.route_axis_hash
    ):
        raise ValueError("signed-work fallback coordinates are stale")
    authorization = authorize_v1_signed_work_six_role_pre_O_fallback_queue(
        six_role_C_design=six_role_C_design,
        progress_receipt=progress_receipt,
        certification_receipt=certification_receipt,
        decision=decision,
        decision_design=decision_design,
        live_chart_identity_hash=live_chart_identity_hash,
        live_current_mask_hash=current_hash,
        live_route_axis_hash=live_route_axis_hash,
        live_proposal_generation=generation,
    )
    if type(authorization) is not V1SignedWorkSixRoleFallbackQueueAuthorization:
        raise TypeError("six-role source replay returned the wrong fallback type")
    authorization_hash = authorization.authorization_hash
    if (
        authorization.current_mask_hash != current_hash
        or authorization.route_axis_hash != current_mask.route_axis_hash
        or authorization.route_axis_hash != live_route_axis_hash
        or authorization.route_count != current_mask.route_count
        or authorization.proposal_generation != generation
        or not authorization.paired_bce_oracle_still_required
        or authorization.mask_commit_authority
    ):
        raise ValueError("signed-work fallback authorization is stale or overbroad")
    states = tuple(
        _action_state_for_coordinate(
            queue_index=index,
            authorization_hash=authorization_hash,
            six_role_C_design=six_role_C_design,
            progress_receipt=progress_receipt,
            certification_receipt=certification_receipt,
            decision=decision,
            decision_design=decision_design,
            coordinate=coordinate,
            current_mask=current_mask,
            current_hash=current_hash,
            generation=generation,
            old_chart_validity_claim=(
                authorization.fallback_admission_bases[index] == "chart_certified"
            ),
        )
        for index, coordinate in enumerate(authorization.fallback_coordinates)
    )
    if not states:
        raise ValueError("signed-work fallback authorization produced an empty queue")
    physical = authorization.queued_physical_candidate_identity_hashes
    domain_hash = _fallback_domain_hash(
        states,
        fallback_authorization_hash=authorization_hash,
        physical_candidate_identity_hashes=physical,
    )
    certification_scope_hash = _fallback_certification_scope_hash(
        states,
        fallback_authorization_hash=authorization_hash,
        blocked_unresolved_coordinate_hash=authorization.blocked_unresolved_coordinate_hash,
    )
    proposals = tuple(
        _issue_proposal(
            state=state,
            domain_hash=domain_hash,
            certification_scope_hash=certification_scope_hash,
        )
        for state in states
    )
    provisional = {
        "action_numeric_states": states,
        "proposals": proposals,
        "domain_hash": domain_hash,
        "certification_scope_hash": certification_scope_hash,
        "six_role_authorization_hash": authorization_hash,
        "six_role_fallback_authorization_hash": authorization_hash,
        "six_role_certification_receipt_hash": certification_receipt.receipt_hash,
        "six_role_decision_hash": decision.decision_hash,
        "six_role_decision_design_hash": decision_design.design_hash,
        "six_role_C_design_hash": six_role_C_design.design_hash,
        "six_role_C_progress_receipt_hash": progress_receipt.receipt_hash,
        "parent_mask_hash": current_hash,
        "route_axis_hash": current_mask.route_axis_hash,
        "route_count": current_mask.route_count,
        "queue_coordinate_hashes": tuple(
            value.selected_coordinate_hash for value in states
        ),
        "queue_branch_endpoint_coordinates": tuple(
            (value.representative_branch, value.representative_endpoint_index)
            for value in states
        ),
        "queue_packed_action_hashes": tuple(value.packed_action_hash for value in states),
        "queue_resulting_mask_hashes": tuple(value.resulting_mask_hash for value in states),
        "queue_six_role_interval_hashes": tuple(
            value.six_role_interval_hash for value in states
        ),
        "queue_resource_receipt_hashes": tuple(
            value.resource_receipt_hash for value in states
        ),
        "queued_physical_candidate_identity_hashes": physical,
        "blocked_unresolved_coordinate_hash": (
            authorization.blocked_unresolved_coordinate_hash
        ),
        "validated_action_count": len(states),
        "proposal_count": len(proposals),
        "composer_id": SIGNED_WORK_PRE_O_FALLBACK_COMPOSER_ID,
        "fallback_queue_materialized_exactly": True,
        "complete_queue_frozen_before_K_O": True,
        "resolved_prefix_queue_frozen_before_K_O": True,
        "fallback_exhaustion_status": authorization.fallback_exhaustion_status,
        "same_generation_paired_O_fallback_without_recomposition_authority": True,
        "source_authorized_filtering_performed": True,
        "reordering_performed": False,
        "claims_hard_global_optimum": False,
        "omitted_mask_optimality_authority": False,
        "proposal_only": True,
        "proposal_witness_only": True,
        "fresh_relinearization_required": True,
        "paired_bce_oracle_required": True,
        "candidate_freeze_authority": False,
        "bce_oracle_authority": False,
        "irreversible_decision_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
        "execution_authority": False,
    }
    shell = object.__new__(V1SignedWorkCandidateCompositionResultV2)
    for name, value in provisional.items():
        object.__setattr__(shell, name, value)
    object.__setattr__(shell, "schema", SIGNED_WORK_CANDIDATE_COMPOSITION_V2_SCHEMA)
    selection_hash = content_hash(
        shell._selection_payload(),
        prefix="partition-free-v1-signed-work-candidate-selection-",
    )
    composition = V1SignedWorkCandidateCompositionResultV2(
        _issuer_token=_COMPOSITION_TOKEN,
        **provisional,
        selection_receipt_hash=selection_hash,
    )
    required_metadata = {
        "refresh_reason": reason,
        "logical_sequence": sequence,
        "generation_index": generation,
        "authority_scope": SIGNED_WORK_RUNTIME_FALLBACK_AUTHORITY_SCOPE,
        "candidate_composition_mode": "canonical_signed_work_six_role_direct_D_pre_O_fallback_queue",
        "complete_queue_frozen_before_K_O": True,
        "resolved_prefix_queue_frozen_before_K_O": True,
        "fallback_exhaustion_status": authorization.fallback_exhaustion_status,
        "blocked_unresolved_coordinate_hash": (
            authorization.blocked_unresolved_coordinate_hash
        ),
        "full_six_role_C_design_embedded": False,
        "raw_C_records_embedded": False,
        "record_retaining_receipts_embedded": False,
        "full_route_numeric_state_embedded": False,
        "fallback_queue_proposal_authority": True,
        "same_generation_paired_O_fallback_without_recomposition_authority": True,
        "direct_D_owns_finite_response_admission": True,
        "QR_transport_is_label_not_hidden_C_veto": True,
        "proposal_only": True,
        "paired_bce_oracle_required": True,
        "calibration_K_open_authority": False,
        "paired_oracle_O_open_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
    }
    caller_metadata = dict(metadata or {})
    collisions = sorted(set(required_metadata).intersection(caller_metadata))
    if collisions:
        raise ValueError(
            f"caller metadata shadows reserved signed-work fallback fields: {collisions}"
        )
    core_metadata = _freeze(
        {**caller_metadata, **required_metadata},
        "signed-work fallback acquisition metadata",
    )
    invalidation_hash = _fallback_invalidation_chart_hash(
        generation_index=generation,
        current_mask=current_mask,
        composition=composition,
    )
    acquisition_hash = _fallback_acquisition_hash(
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        lineage=lineage,
        transitions=transitions,
        invalidation_chart_hash=invalidation_hash,
        composition_result_hash=composition.result_hash,
        metadata=core_metadata,
    )
    receipt = V1SignedWorkAcquisitionGenerationReceiptV2(
        campaign_contract_hash=binding.contract.campaign_contract_hash,
        campaign_binding_hash=binding.binding_hash,
        registration_hash=binding.registration.registration_hash,
        generation_index=generation,
        current_mask_hash=current_hash,
        route_axis_hash=current_mask.route_axis_hash,
        acquisition_hash=acquisition_hash,
        invalidation_chart_hash=invalidation_hash,
        composition_result_hash=composition.result_hash,
        domain_hash=composition.domain_hash,
        certification_scope_hash=composition.certification_scope_hash,
        six_role_authorization_hash=composition.six_role_authorization_hash,
        six_role_fallback_authorization_hash=(
            composition.six_role_fallback_authorization_hash
        ),
        six_role_certification_receipt_hash=(
            composition.six_role_certification_receipt_hash
        ),
        six_role_decision_hash=composition.six_role_decision_hash,
        six_role_decision_design_hash=composition.six_role_decision_design_hash,
        six_role_C_design_hash=composition.six_role_C_design_hash,
        six_role_C_progress_receipt_hash=composition.six_role_C_progress_receipt_hash,
        selected_coordinate_hash=states[0].selected_coordinate_hash,
        ordered_candidate_ids=tuple(value.candidate_id for value in proposals),
        ordered_coordinate_hashes=composition.queue_coordinate_hashes,
        ordered_branch_endpoint_coordinates=composition.queue_branch_endpoint_coordinates,
        ordered_packed_action_hashes=composition.queue_packed_action_hashes,
        ordered_six_role_interval_hashes=composition.queue_six_role_interval_hashes,
        queued_physical_candidate_identity_hashes=(
            composition.queued_physical_candidate_identity_hashes
        ),
        blocked_unresolved_coordinate_hash=composition.blocked_unresolved_coordinate_hash,
        refresh_reason=reason,
        logical_sequence=sequence,
        fallback_exhaustion_status=authorization.fallback_exhaustion_status,
        metadata={
            "queue_size": len(proposals),
            "queue_resource_values": tuple(value.resource_gain for value in proposals),
            "selected_coordinate_hash": states[0].selected_coordinate_hash,
            "blocked_unresolved_coordinate_hash": (
                authorization.blocked_unresolved_coordinate_hash
            ),
            "fallback_exhaustion_status": authorization.fallback_exhaustion_status,
            "authority_scope": SIGNED_WORK_RUNTIME_FALLBACK_AUTHORITY_SCOPE,
        },
    )
    return V1SignedWorkAuthorizedAcquisitionV2(
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        committed_mask_lineage_hashes=lineage,
        committed_mask_transition_receipt_hashes=transitions,
        composition=composition,
        receipt=receipt,
        metadata=core_metadata,
    )


__all__ = [
    "SIGNED_WORK_ACQUISITION_RECEIPT_SCHEMA",
    "SIGNED_WORK_ACQUISITION_RECEIPT_V2_SCHEMA",
    "SIGNED_WORK_ACTION_NUMERIC_STATE_SCHEMA",
    "SIGNED_WORK_AUTHORIZED_ACQUISITION_SCHEMA",
    "SIGNED_WORK_AUTHORIZED_ACQUISITION_V2_SCHEMA",
    "SIGNED_WORK_CANDIDATE_COMPOSITION_SCHEMA",
    "SIGNED_WORK_CANDIDATE_COMPOSITION_V2_SCHEMA",
    "SIGNED_WORK_CANDIDATE_PROPOSAL_SCHEMA",
    "SIGNED_WORK_COMPACT_FALLBACK_DESCRIPTOR_SCHEMA",
    "SIGNED_WORK_DEFERRED_BRANCH_SOURCE_SCHEMA",
    "SIGNED_WORK_DEFERRED_FALLBACK_PLAN_SCHEMA",
    "SIGNED_WORK_PRE_O_FALLBACK_COMPOSER_ID",
    "SIGNED_WORK_RUNTIME_FALLBACK_AUTHORITY_SCOPE",
    "SIGNED_WORK_RUNTIME_AUTHORITY_SCOPE",
    "SIGNED_WORK_SINGLETON_COMPOSER_ID",
    "V1SignedWorkAcquisitionGenerationReceipt",
    "V1SignedWorkAcquisitionGenerationReceiptV2",
    "V1SignedWorkActionNumericState",
    "V1SignedWorkAuthorizedAcquisition",
    "V1SignedWorkAuthorizedAcquisitionV2",
    "V1SignedWorkCandidateCompositionResult",
    "V1SignedWorkCandidateCompositionResultV2",
    "V1SignedWorkCandidateProposal",
    "V1SignedWorkCompactFallbackDescriptor",
    "V1SignedWorkDeferredBranchSource",
    "V1SignedWorkDeferredFallbackPlan",
    "issue_v1_signed_work_authorized_acquisition",
    "issue_v1_signed_work_deferred_fallback_plan",
    "issue_v1_signed_work_fallback_authorized_acquisition",
]
