"""Receipt-only pre-O runtime plans for label-native signed-work v1.

The C certifier authorizes a resolved prefix of frozen hard actions.  This
module turns that authorization into a compact runtime plan without retaining
the direction provider, its leased label handle, a mask payload, or a
route-scale array.  Every descriptor is joined through the exact qualified
provider lane, compact-C crosswalk, and immutable device-shell action receipt.

This seam is deliberately additive.  It grants no calibration, oracle,
commit, terminal, Pareto, or execution authority; a later runtime seam must
replay one descriptor against a live provider before materializing a carrier.
The legacy full-axis/provider-side filtering mode is valid for reference tests
but cannot issue this canonical production plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_label_native_c_bridge import (
    V1SignedWorkLabelNativeCActionCrosswalkReceipt,
    V1SignedWorkLabelNativeCBranchReceipt,
    V1SignedWorkLabelNativeCBridgeFamilyBundle,
)
from .modern_v1_signed_work_label_native_direction_provider import (
    LABEL_NATIVE_QUALIFIED_DIRECTION_RECEIPT_SCHEMA,
    V1LabelNativePackedDirectionReceipt,
    V1LabelNativeQualifiedLaneRegistration,
)
from .modern_v1_signed_work_label_native_six_role_c import (
    V1LabelNativeSixRoleCDesign,
    V1LabelNativeSixRoleProgressReceipt,
)
from .modern_v1_signed_work_label_native_six_role_certification import (
    authorize_v1_signed_work_label_native_six_role_pre_O_fallback_queue,
)
from .modern_v1_signed_work_six_role_certification import (
    SixRoleFallbackExhaustionStatus,
    V1SignedWorkIntervalStatus,
    V1SignedWorkSixRoleCertificationReceipt,
    V1SignedWorkSixRoleDecision,
    V1SignedWorkSixRoleDecisionDesign,
    V1SignedWorkSixRoleEndpointCertificate,
    V1SignedWorkSixRoleEndpointCoordinate,
    V1SignedWorkSixRoleFallbackQueueAuthorization,
)
from .modern_v1_device_shell_ladder import V1DeviceShellFrozenActionReceipt


LABEL_NATIVE_RUNTIME_DESCRIPTOR_SCHEMA: Final = (
    "partition_free_v1_label_native_pre_O_runtime_descriptor_v1"
)
LABEL_NATIVE_RUNTIME_PLAN_SCHEMA: Final = (
    "partition_free_v1_label_native_pre_O_runtime_plan_v1"
)
LABEL_NATIVE_RUNTIME_PLAN_CHECKPOINT_PAYLOAD_SCHEMA: Final = (
    "partition_free_v1_label_native_runtime_plan_checkpoint_payload_v1"
)
LABEL_NATIVE_RUNTIME_PLAN_AUTHORITY: Final = (
    "qualified_label_native_resolved_prefix_receipt_only_no_K_O_commit_"
    "terminal_pareto_or_execution_authority_v1"
)

_DESCRIPTOR_PREFIX: Final = "modern-v1-label-native-runtime-descriptor-"
_PLAN_PREFIX: Final = "modern-v1-label-native-runtime-plan-"
_DESCRIPTOR_ISSUER_TOKEN: Final = object()
_PLAN_ISSUER_TOKEN: Final = object()
_MAX_CHECKPOINT_DESCRIPTOR_COUNT: Final = 8

_DESCRIPTOR_CHECKPOINT_INIT_FIELDS: Final = (
    "queue_index",
    "coordinate",
    "certificate",
    "crosswalk",
    "source_action_receipt",
    "compact_provider_endpoint_label",
    "source_endpoint_label",
    "physical_duplicate_coordinate_hashes",
    "physical_duplicate_certificate_hashes",
    "provider_receipt_hash",
    "provider_design_hash",
    "provider_policy_hash",
    "provider_forecast_hash",
    "provider_backend_kind",
    "provider_device_identity",
    "prefix_rank_receipt_hash",
    "endpoint_query_hash",
    "resulting_label_commitment",
    "qualified_lane_registration_hash",
    "action_axis_registration_hash",
    "source_shell_ladder_receipt_hash",
    "eligible_subset_registration_hash",
    "branch_C_design_hash",
    "bridge_branch_receipt_hash",
    "fallback_authorization_hash",
    "six_role_C_design_hash",
    "six_role_C_progress_receipt_hash",
    "six_role_certification_receipt_hash",
    "six_role_decision_hash",
    "six_role_decision_design_hash",
    "bridge_family_bundle_hash",
    "campaign_identity_hash",
    "campaign_contract_hash",
    "campaign_registration_hash",
    "live_chart_identity_hash",
    "live_current_mask_hash",
    "live_route_axis_hash",
    "live_route_count",
    "live_proposal_generation",
    "native_qualified_subset_execution_replayed",
    "exact_source_replay_completed",
    "provider_or_handle_retained",
    "mask_or_route_payload_retained",
    "calibration_K_open_authority",
    "paired_oracle_O_open_authority",
    "mask_commit_authority",
    "terminal_authority",
    "pareto_authority",
    "execution_authority",
    "authority_scope",
)
_PLAN_CHECKPOINT_INIT_FIELDS: Final = (
    "campaign_identity_hash",
    "campaign_contract_hash",
    "campaign_registration_hash",
    "live_chart_identity_hash",
    "live_current_mask_hash",
    "live_route_axis_hash",
    "live_route_count",
    "live_proposal_generation",
    "six_role_C_design_hash",
    "six_role_C_progress_receipt_hash",
    "six_role_certification_receipt_hash",
    "six_role_decision_hash",
    "six_role_decision_design_hash",
    "bridge_family_bundle_hash",
    "fallback_authorization_hash",
    "selected_coordinate_hash",
    "ordered_descriptor_hashes",
    "ordered_coordinate_hashes",
    "ordered_physical_candidate_identity_hashes",
    "blocked_unresolved_coordinate_hash",
    "fallback_exhaustion_status",
    "authorized_descriptor_count",
    "resolved_prefix_frozen_before_K_O",
    "native_qualified_subset_execution_replayed",
    "source_authorized_duplicate_quotient_only",
    "stops_before_first_unresolved_decision_changing_coordinate",
    "no_provider_handle_mask_or_route_payload_retained",
    "paired_bce_oracle_still_required",
    "candidate_selection_authority",
    "calibration_K_open_authority",
    "paired_oracle_O_open_authority",
    "irreversible_decision_authority",
    "mask_commit_authority",
    "terminal_authority",
    "pareto_authority",
    "execution_authority",
    "authority_scope",
)
_CHECKPOINT_PAYLOAD_FIELDS: Final = frozenset(
    {
        "schema",
        "plan_schema",
        "descriptor_schema",
        "plan_fields",
        "descriptor_records",
        "plan_hash",
    }
)
_CHECKPOINT_DESCRIPTOR_RECORD_FIELDS: Final = frozenset(
    {"fields", "descriptor_hash"}
)


class V1LabelNativeRuntimePlanRefusalCode(str, Enum):
    """Typed fail-closed outcomes for plan issuance and replay."""

    INVALID_TYPE = "invalid_type"
    SOURCE_REPLAY_MISMATCH = "source_replay_mismatch"
    QUALIFIED_EXECUTION_REQUIRED = "qualified_execution_required"
    REFERENCE_BACKEND_FORBIDDEN = "reference_backend_forbidden"
    STALE_LIVE_SOURCE = "stale_live_source"
    ORDER_MISMATCH = "order_mismatch"
    DUPLICATE_REPLAY_MISMATCH = "duplicate_replay_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"
    TAMPERED = "tampered"
    CHECKPOINT_PAYLOAD_INVALID = "checkpoint_payload_invalid"


class V1LabelNativeRuntimePlanRefusal(ValueError):
    def __init__(
        self,
        code: V1LabelNativeRuntimePlanRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LabelNativeRuntimePlanRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LabelNativeRuntimePlanRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
            f"{name} must be one exact positive integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    return int(value)


def _branch_receipt(
    bridge: V1SignedWorkLabelNativeCBridgeFamilyBundle,
    branch: V1RelaxedKKTBranch,
) -> V1SignedWorkLabelNativeCBranchReceipt:
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        return bridge.spectral
    if branch is V1RelaxedKKTBranch.DIAGONAL_L1:
        return bridge.diagonal_l1
    _refuse(
        V1LabelNativeRuntimePlanRefusalCode.SOURCE_REPLAY_MISMATCH,
        "runtime coordinate names an unsupported KKT branch",
    )


def _certificate_signature(
    certificate: V1SignedWorkSixRoleEndpointCertificate,
) -> tuple[object, ...]:
    """Complete outcome signature used by the source authorization quotient."""

    return (
        certificate.f_native_interval,
        certificate.direct_D_status,
        certificate.d_half_interval,
        certificate.d_one_interval,
        certificate.admission_basis,
        certificate.chart_transport_status,
        certificate.q_j_interval,
        certificate.r_half_interval,
        certificate.r_one_interval,
        certificate.chart_half_interval,
        certificate.chart_one_interval,
    )


def _assert_exact_bool(value: object, *, expected: bool, name: str) -> None:
    if type(value) is not bool or value is not expected:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.AUTHORITY_VIOLATION,
            f"{name} crossed the receipt-only authority boundary",
        )


@dataclass(frozen=True, slots=True, init=False)
class V1LabelNativeRuntimeFallbackDescriptor:
    """One resolved fallback rung joined to its qualified provider lane."""

    queue_index: int
    coordinate: V1SignedWorkSixRoleEndpointCoordinate
    certificate: V1SignedWorkSixRoleEndpointCertificate = field(
        repr=False,
        compare=False,
    )
    crosswalk: V1SignedWorkLabelNativeCActionCrosswalkReceipt = field(
        repr=False,
        compare=False,
    )
    source_action_receipt: V1DeviceShellFrozenActionReceipt = field(
        repr=False,
        compare=False,
    )
    compact_provider_endpoint_label: int
    source_endpoint_label: int
    physical_duplicate_coordinate_hashes: tuple[str, ...]
    physical_duplicate_certificate_hashes: tuple[str, ...]
    provider_receipt_hash: str
    provider_design_hash: str
    provider_policy_hash: str
    provider_forecast_hash: str
    provider_backend_kind: str
    provider_device_identity: str
    prefix_rank_receipt_hash: str
    endpoint_query_hash: str
    resulting_label_commitment: str
    qualified_lane_registration_hash: str
    action_axis_registration_hash: str
    source_shell_ladder_receipt_hash: str
    eligible_subset_registration_hash: str
    branch_C_design_hash: str
    bridge_branch_receipt_hash: str
    fallback_authorization_hash: str
    six_role_C_design_hash: str
    six_role_C_progress_receipt_hash: str
    six_role_certification_receipt_hash: str
    six_role_decision_hash: str
    six_role_decision_design_hash: str
    bridge_family_bundle_hash: str
    campaign_identity_hash: str
    campaign_contract_hash: str
    campaign_registration_hash: str
    live_chart_identity_hash: str
    live_current_mask_hash: str
    live_route_axis_hash: str
    live_route_count: int
    live_proposal_generation: int
    native_qualified_subset_execution_replayed: bool
    exact_source_replay_completed: bool
    provider_or_handle_retained: bool
    mask_or_route_payload_retained: bool
    calibration_K_open_authority: bool
    paired_oracle_O_open_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    authority_scope: str
    schema: str = LABEL_NATIVE_RUNTIME_DESCRIPTOR_SCHEMA
    _descriptor_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _DESCRIPTOR_ISSUER_TOKEN:
            raise TypeError("label-native runtime descriptors are issuer-only")
        expected = {
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        }
        if set(values) != expected:
            raise TypeError(
                "label-native runtime descriptor fields differ: "
                f"missing={sorted(expected.difference(values))}, "
                f"extra={sorted(set(values).difference(expected))}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", LABEL_NATIVE_RUNTIME_DESCRIPTOR_SCHEMA)
        object.__setattr__(self, "_descriptor_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        queue_index = _nonnegative_int(self.queue_index, name="queue index")
        if type(self.coordinate) is not V1SignedWorkSixRoleEndpointCoordinate:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
                "descriptor requires an exact endpoint coordinate",
            )
        if type(self.certificate) is not V1SignedWorkSixRoleEndpointCertificate:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
                "descriptor requires an exact endpoint certificate",
            )
        if type(self.crosswalk) is not V1SignedWorkLabelNativeCActionCrosswalkReceipt:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
                "descriptor requires an exact compact-C crosswalk",
            )
        if type(self.source_action_receipt) is not V1DeviceShellFrozenActionReceipt:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
                "descriptor requires an exact shell action receipt",
            )
        compact_label = _positive_int(
            self.compact_provider_endpoint_label,
            name="compact provider endpoint label",
        )
        source_label = _positive_int(
            self.source_endpoint_label,
            name="source endpoint label",
        )
        duplicate_coordinates = tuple(
            _identity(value, name="physical duplicate coordinate hash")
            for value in self.physical_duplicate_coordinate_hashes
        )
        duplicate_certificates = tuple(
            _identity(value, name="physical duplicate certificate hash")
            for value in self.physical_duplicate_certificate_hashes
        )
        if (
            not duplicate_coordinates
            or len(duplicate_coordinates) != len(duplicate_certificates)
            or len(set(duplicate_coordinates)) != len(duplicate_coordinates)
            or self.coordinate.coordinate_hash not in duplicate_coordinates
            or self.certificate.certificate_hash not in duplicate_certificates
        ):
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.DUPLICATE_REPLAY_MISMATCH,
                "descriptor duplicate-equivalence class is incomplete or repeated",
            )
        for name in (
            "provider_receipt_hash",
            "provider_design_hash",
            "provider_policy_hash",
            "provider_forecast_hash",
            "provider_backend_kind",
            "provider_device_identity",
            "prefix_rank_receipt_hash",
            "endpoint_query_hash",
            "resulting_label_commitment",
            "qualified_lane_registration_hash",
            "action_axis_registration_hash",
            "source_shell_ladder_receipt_hash",
            "eligible_subset_registration_hash",
            "branch_C_design_hash",
            "bridge_branch_receipt_hash",
            "fallback_authorization_hash",
            "six_role_C_design_hash",
            "six_role_C_progress_receipt_hash",
            "six_role_certification_receipt_hash",
            "six_role_decision_hash",
            "six_role_decision_design_hash",
            "bridge_family_bundle_hash",
            "campaign_identity_hash",
            "campaign_contract_hash",
            "campaign_registration_hash",
            "live_chart_identity_hash",
            "live_current_mask_hash",
            "live_route_axis_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.provider_backend_kind != "torch_cuda":
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.REFERENCE_BACKEND_FORBIDDEN,
                "canonical runtime descriptors require qualified CUDA execution",
            )
        route_count = _positive_int(self.live_route_count, name="live route count")
        generation = _nonnegative_int(
            self.live_proposal_generation,
            name="live proposal generation",
        )
        if (
            self.certificate.coordinate != self.coordinate
            or self.crosswalk.branch is not self.coordinate.branch
            or self.crosswalk.compact_C_endpoint_index
            != self.coordinate.endpoint_index
            or self.crosswalk.source_endpoint_label != self.coordinate.endpoint_label
            or compact_label != self.coordinate.endpoint_index
            or source_label != self.coordinate.endpoint_label
            or self.source_action_receipt != self.crosswalk.source_action_receipt
            or self.source_action_receipt.endpoint_label != source_label
            or self.source_action_receipt.receipt_hash
            != self.crosswalk.source_action_receipt_hash
            or self.source_action_receipt.packed_action_hash
            != self.coordinate.action_hash
            or self.crosswalk.receipt_hash != self.coordinate.crosswalk_receipt_hash
            or self.crosswalk.hard_fq_endpoint_receipt_hash
            != self.coordinate.hard_replay_endpoint_receipt_hash
            or self.crosswalk.canonical_physical_identity_hash
            != self.coordinate.canonical_candidate_identity_hash
            or self.crosswalk.cumulative_removed_count
            != self.coordinate.cumulative_removed_count
            or self.crosswalk.route_axis_hash != self.live_route_axis_hash
            or self.crosswalk.route_count != route_count
            or self.crosswalk.parent_mask_hash != self.live_current_mask_hash
        ):
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.SOURCE_REPLAY_MISMATCH,
                "descriptor coordinate, certificate, crosswalk, and action differ",
            )
        required_true = (
            (self.native_qualified_subset_execution_replayed, "qualified replay"),
            (self.exact_source_replay_completed, "exact source replay"),
        )
        required_false = (
            (self.provider_or_handle_retained, "provider/handle retention"),
            (self.mask_or_route_payload_retained, "mask/route payload retention"),
            (self.calibration_K_open_authority, "K authority"),
            (self.paired_oracle_O_open_authority, "O authority"),
            (self.mask_commit_authority, "commit authority"),
            (self.terminal_authority, "terminal authority"),
            (self.pareto_authority, "Pareto authority"),
            (self.execution_authority, "execution authority"),
        )
        for value, name in required_true:
            _assert_exact_bool(value, expected=True, name=name)
        for value, name in required_false:
            _assert_exact_bool(value, expected=False, name=name)
        if (
            self.authority_scope != LABEL_NATIVE_RUNTIME_PLAN_AUTHORITY
            or self.schema != LABEL_NATIVE_RUNTIME_DESCRIPTOR_SCHEMA
        ):
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.AUTHORITY_VIOLATION,
                "descriptor authority or schema is unsupported",
            )
        object.__setattr__(self, "queue_index", queue_index)
        object.__setattr__(self, "compact_provider_endpoint_label", compact_label)
        object.__setattr__(self, "source_endpoint_label", source_label)
        object.__setattr__(
            self,
            "physical_duplicate_coordinate_hashes",
            duplicate_coordinates,
        )
        object.__setattr__(
            self,
            "physical_duplicate_certificate_hashes",
            duplicate_certificates,
        )
        object.__setattr__(self, "live_route_count", route_count)
        object.__setattr__(self, "live_proposal_generation", generation)

    @property
    def coordinate_hash(self) -> str:
        return self.coordinate.coordinate_hash

    @property
    def certificate_hash(self) -> str:
        return self.certificate.certificate_hash

    @property
    def branch(self) -> V1RelaxedKKTBranch:
        return self.coordinate.branch

    @property
    def cumulative_removed_count(self) -> int:
        return self.coordinate.cumulative_removed_count

    @property
    def pre_C_priority(self) -> int:
        return self.coordinate.pre_C_priority

    @property
    def canonical_physical_identity_hash(self) -> str:
        return self.coordinate.canonical_candidate_identity_hash

    @property
    def source_action_receipt_hash(self) -> str:
        return self.source_action_receipt.receipt_hash

    @property
    def f_native_interval(self) -> tuple[float, float]:
        return self.certificate.f_native_interval

    @property
    def q_j_interval(self) -> tuple[float, float]:
        return self.certificate.q_j_interval

    @property
    def r_half_interval(self) -> tuple[float, float]:
        return self.certificate.r_half_interval

    @property
    def r_one_interval(self) -> tuple[float, float]:
        return self.certificate.r_one_interval

    @property
    def d_half_interval(self) -> tuple[float, float]:
        return self.certificate.d_half_interval

    @property
    def d_one_interval(self) -> tuple[float, float]:
        return self.certificate.d_one_interval

    @property
    def chart_half_interval(self) -> tuple[float, float]:
        return self.certificate.chart_half_interval

    @property
    def chart_one_interval(self) -> tuple[float, float]:
        return self.certificate.chart_one_interval

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_descriptor_hash"
        }

    @property
    def descriptor_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        try:
            self.crosswalk.assert_integrity()
            self.__post_init__()
        except V1LabelNativeRuntimePlanRefusal:
            raise
        except (TypeError, ValueError) as error:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.TAMPERED,
                f"descriptor source receipt no longer replays: {error}",
            )
        observed = content_hash(self._payload(), prefix=_DESCRIPTOR_PREFIX)
        if self._descriptor_hash is not None and observed != self._descriptor_hash:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.TAMPERED,
                "runtime descriptor was mutated after issuance",
            )
        if expected_hash is not None and observed != _identity(
            expected_hash,
            name="expected descriptor hash",
        ):
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.TAMPERED,
                "runtime descriptor hash differs from the expected receipt",
            )
        object.__setattr__(self, "_descriptor_hash", observed)
        return observed

    def __reduce__(self) -> NoReturn:
        raise TypeError("label-native runtime descriptors are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("label-native runtime descriptors are nonserializable")


@dataclass(frozen=True, slots=True, init=False)
class V1LabelNativeRuntimeFallbackPlan:
    """Immutable resolved-prefix plan issued before K or paired O opens."""

    campaign_identity_hash: str
    campaign_contract_hash: str
    campaign_registration_hash: str
    live_chart_identity_hash: str
    live_current_mask_hash: str
    live_route_axis_hash: str
    live_route_count: int
    live_proposal_generation: int
    six_role_C_design_hash: str
    six_role_C_progress_receipt_hash: str
    six_role_certification_receipt_hash: str
    six_role_decision_hash: str
    six_role_decision_design_hash: str
    bridge_family_bundle_hash: str
    fallback_authorization_hash: str
    selected_coordinate_hash: str
    descriptors: tuple[V1LabelNativeRuntimeFallbackDescriptor, ...] = field(
        repr=False,
        compare=False,
    )
    ordered_descriptor_hashes: tuple[str, ...]
    ordered_coordinate_hashes: tuple[str, ...]
    ordered_physical_candidate_identity_hashes: tuple[str, ...]
    blocked_unresolved_coordinate_hash: str | None
    fallback_exhaustion_status: SixRoleFallbackExhaustionStatus
    authorized_descriptor_count: int
    resolved_prefix_frozen_before_K_O: bool
    native_qualified_subset_execution_replayed: bool
    source_authorized_duplicate_quotient_only: bool
    stops_before_first_unresolved_decision_changing_coordinate: bool
    no_provider_handle_mask_or_route_payload_retained: bool
    paired_bce_oracle_still_required: bool
    candidate_selection_authority: bool
    calibration_K_open_authority: bool
    paired_oracle_O_open_authority: bool
    irreversible_decision_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    authority_scope: str
    schema: str = LABEL_NATIVE_RUNTIME_PLAN_SCHEMA
    _plan_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _PLAN_ISSUER_TOKEN:
            raise TypeError("label-native runtime fallback plans are issuer-only")
        expected = {
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        }
        if set(values) != expected:
            raise TypeError(
                "label-native runtime plan fields differ: "
                f"missing={sorted(expected.difference(values))}, "
                f"extra={sorted(set(values).difference(expected))}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", LABEL_NATIVE_RUNTIME_PLAN_SCHEMA)
        object.__setattr__(self, "_plan_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "campaign_identity_hash",
            "campaign_contract_hash",
            "campaign_registration_hash",
            "live_chart_identity_hash",
            "live_current_mask_hash",
            "live_route_axis_hash",
            "six_role_C_design_hash",
            "six_role_C_progress_receipt_hash",
            "six_role_certification_receipt_hash",
            "six_role_decision_hash",
            "six_role_decision_design_hash",
            "bridge_family_bundle_hash",
            "fallback_authorization_hash",
            "selected_coordinate_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        route_count = _positive_int(self.live_route_count, name="live route count")
        generation = _nonnegative_int(
            self.live_proposal_generation,
            name="live proposal generation",
        )
        descriptors = tuple(self.descriptors)
        if not descriptors or any(
            type(value) is not V1LabelNativeRuntimeFallbackDescriptor
            for value in descriptors
        ):
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
                "runtime plan requires exact nonempty descriptors",
            )
        for index, descriptor in enumerate(descriptors):
            descriptor.assert_integrity()
            if (
                descriptor.queue_index != index
                or descriptor.campaign_identity_hash != self.campaign_identity_hash
                or descriptor.campaign_contract_hash != self.campaign_contract_hash
                or descriptor.campaign_registration_hash
                != self.campaign_registration_hash
                or descriptor.live_chart_identity_hash
                != self.live_chart_identity_hash
                or descriptor.live_current_mask_hash != self.live_current_mask_hash
                or descriptor.live_route_axis_hash != self.live_route_axis_hash
                or descriptor.live_route_count != route_count
                or descriptor.live_proposal_generation != generation
                or descriptor.six_role_C_design_hash
                != self.six_role_C_design_hash
                or descriptor.six_role_C_progress_receipt_hash
                != self.six_role_C_progress_receipt_hash
                or descriptor.six_role_certification_receipt_hash
                != self.six_role_certification_receipt_hash
                or descriptor.six_role_decision_hash != self.six_role_decision_hash
                or descriptor.six_role_decision_design_hash
                != self.six_role_decision_design_hash
                or descriptor.bridge_family_bundle_hash
                != self.bridge_family_bundle_hash
                or descriptor.fallback_authorization_hash
                != self.fallback_authorization_hash
            ):
                _refuse(
                    V1LabelNativeRuntimePlanRefusalCode.SOURCE_REPLAY_MISMATCH,
                    "a runtime descriptor left the plan source frame",
                )
        descriptor_hashes = tuple(value.descriptor_hash for value in descriptors)
        coordinates = tuple(value.coordinate_hash for value in descriptors)
        physical = tuple(
            value.canonical_physical_identity_hash for value in descriptors
        )
        order = tuple(
            (-value.cumulative_removed_count, value.pre_C_priority)
            for value in descriptors
        )
        if (
            self.selected_coordinate_hash != coordinates[0]
            or tuple(self.ordered_descriptor_hashes) != descriptor_hashes
            or tuple(self.ordered_coordinate_hashes) != coordinates
            or tuple(self.ordered_physical_candidate_identity_hashes) != physical
            or len(set(coordinates)) != len(coordinates)
            or len(set(physical)) != len(physical)
            or order != tuple(sorted(order))
        ):
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.ORDER_MISMATCH,
                "runtime descriptors differ from resource/hard-F prefix order",
            )
        blocked = self.blocked_unresolved_coordinate_hash
        if blocked is not None:
            blocked = _identity(blocked, name="blocked unresolved coordinate hash")
        expected_exhaustion: SixRoleFallbackExhaustionStatus = (
            "blocked_on_more_C"
            if blocked is not None
            else "fully_resolved_no_more_candidate"
        )
        if self.fallback_exhaustion_status != expected_exhaustion:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.ORDER_MISMATCH,
                "runtime plan exhaustion status contradicts its blocked coordinate",
            )
        if _positive_int(
            self.authorized_descriptor_count,
            name="authorized descriptor count",
        ) != len(descriptors):
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.ORDER_MISMATCH,
                "runtime descriptor count differs from its exact prefix",
            )
        required_true = (
            (self.resolved_prefix_frozen_before_K_O, "pre-K/O prefix freeze"),
            (
                self.native_qualified_subset_execution_replayed,
                "qualified execution replay",
            ),
            (
                self.source_authorized_duplicate_quotient_only,
                "source duplicate quotient",
            ),
            (
                self.stops_before_first_unresolved_decision_changing_coordinate,
                "unresolved stopping rule",
            ),
            (
                self.no_provider_handle_mask_or_route_payload_retained,
                "receipt-only retention",
            ),
            (self.paired_bce_oracle_still_required, "paired BCE requirement"),
        )
        required_false = (
            (self.candidate_selection_authority, "candidate-selection authority"),
            (self.calibration_K_open_authority, "K authority"),
            (self.paired_oracle_O_open_authority, "O authority"),
            (
                self.irreversible_decision_authority,
                "irreversible-decision authority",
            ),
            (self.mask_commit_authority, "commit authority"),
            (self.terminal_authority, "terminal authority"),
            (self.pareto_authority, "Pareto authority"),
            (self.execution_authority, "execution authority"),
        )
        for value, name in required_true:
            _assert_exact_bool(value, expected=True, name=name)
        for value, name in required_false:
            _assert_exact_bool(value, expected=False, name=name)
        if (
            self.authority_scope != LABEL_NATIVE_RUNTIME_PLAN_AUTHORITY
            or self.schema != LABEL_NATIVE_RUNTIME_PLAN_SCHEMA
        ):
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.AUTHORITY_VIOLATION,
                "runtime plan authority or schema is unsupported",
            )
        object.__setattr__(self, "live_route_count", route_count)
        object.__setattr__(self, "live_proposal_generation", generation)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "blocked_unresolved_coordinate_hash", blocked)

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
        try:
            self.__post_init__()
        except V1LabelNativeRuntimePlanRefusal:
            raise
        except (TypeError, ValueError) as error:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.TAMPERED,
                f"runtime plan no longer replays: {error}",
            )
        observed = content_hash(self._payload(), prefix=_PLAN_PREFIX)
        if self._plan_hash is not None and observed != self._plan_hash:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.TAMPERED,
                "runtime plan was mutated after issuance",
            )
        if expected_hash is not None and observed != _identity(
            expected_hash,
            name="expected runtime plan hash",
        ):
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.TAMPERED,
                "runtime plan hash differs from the expected receipt",
            )
        object.__setattr__(self, "_plan_hash", observed)
        return observed

    def __reduce__(self) -> NoReturn:
        raise TypeError("label-native runtime fallback plans are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("label-native runtime fallback plans are nonserializable")


def _assert_checkpoint_init_field_schema() -> None:
    descriptor_fields = tuple(
        name
        for name, declaration in (
            V1LabelNativeRuntimeFallbackDescriptor.__dataclass_fields__.items()
        )
        if declaration.init and name != "schema"
    )
    plan_fields = tuple(
        name
        for name, declaration in (
            V1LabelNativeRuntimeFallbackPlan.__dataclass_fields__.items()
        )
        if declaration.init and name not in {"descriptors", "schema"}
    )
    if (
        descriptor_fields != _DESCRIPTOR_CHECKPOINT_INIT_FIELDS
        or plan_fields != _PLAN_CHECKPOINT_INIT_FIELDS
    ):
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.CHECKPOINT_PAYLOAD_INVALID,
            "runtime plan checkpoint field schema changed without a payload version",
        )


def export_v1_label_native_runtime_plan_checkpoint_payload(
    plan: V1LabelNativeRuntimeFallbackPlan,
) -> Mapping[str, object]:
    """Export one bounded receipt-only plan without weakening pickle refusal."""

    if type(plan) is not V1LabelNativeRuntimeFallbackPlan:
        raise TypeError("checkpoint export requires an exact label-native runtime plan")
    _assert_checkpoint_init_field_schema()
    plan_hash = plan.assert_integrity()
    descriptors = tuple(plan.descriptors)
    if not descriptors or len(descriptors) > _MAX_CHECKPOINT_DESCRIPTOR_COUNT:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.CHECKPOINT_PAYLOAD_INVALID,
            "runtime plan checkpoint payload must contain between one and eight descriptors",
        )
    descriptor_records: list[Mapping[str, object]] = []
    for descriptor in descriptors:
        descriptor_hash = descriptor.assert_integrity()
        descriptor_records.append(
            MappingProxyType(
                {
                    "fields": MappingProxyType(
                        {
                            name: getattr(descriptor, name)
                            for name in _DESCRIPTOR_CHECKPOINT_INIT_FIELDS
                        }
                    ),
                    "descriptor_hash": descriptor_hash,
                }
            )
        )
    plan_fields = MappingProxyType(
        {
            name: getattr(plan, name)
            for name in _PLAN_CHECKPOINT_INIT_FIELDS
        }
    )
    return MappingProxyType(
        {
            "schema": LABEL_NATIVE_RUNTIME_PLAN_CHECKPOINT_PAYLOAD_SCHEMA,
            "plan_schema": LABEL_NATIVE_RUNTIME_PLAN_SCHEMA,
            "descriptor_schema": LABEL_NATIVE_RUNTIME_DESCRIPTOR_SCHEMA,
            "plan_fields": plan_fields,
            "descriptor_records": tuple(descriptor_records),
            "plan_hash": plan_hash,
        }
    )


def _checkpoint_mapping(
    value: object,
    *,
    expected_fields: frozenset[str] | set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(expected_fields):
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.CHECKPOINT_PAYLOAD_INVALID,
            f"{name} has stale, missing, or extra fields",
        )
    return value


def restore_v1_label_native_runtime_plan_checkpoint_payload(
    payload: Mapping[str, object],
) -> V1LabelNativeRuntimeFallbackPlan:
    """Reconstruct an exact issuer-owned plan and replay all source receipts."""

    _assert_checkpoint_init_field_schema()
    root = _checkpoint_mapping(
        payload,
        expected_fields=_CHECKPOINT_PAYLOAD_FIELDS,
        name="runtime plan checkpoint payload",
    )
    if (
        root["schema"] != LABEL_NATIVE_RUNTIME_PLAN_CHECKPOINT_PAYLOAD_SCHEMA
        or root["plan_schema"] != LABEL_NATIVE_RUNTIME_PLAN_SCHEMA
        or root["descriptor_schema"] != LABEL_NATIVE_RUNTIME_DESCRIPTOR_SCHEMA
    ):
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.CHECKPOINT_PAYLOAD_INVALID,
            "runtime plan checkpoint payload schema is unsupported",
        )
    raw_records = root["descriptor_records"]
    if (
        type(raw_records) is not tuple
        or not raw_records
        or len(raw_records) > _MAX_CHECKPOINT_DESCRIPTOR_COUNT
    ):
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.CHECKPOINT_PAYLOAD_INVALID,
            "runtime plan checkpoint descriptor axis is empty or unbounded",
        )
    descriptors: list[V1LabelNativeRuntimeFallbackDescriptor] = []
    for raw_record in raw_records:
        record = _checkpoint_mapping(
            raw_record,
            expected_fields=_CHECKPOINT_DESCRIPTOR_RECORD_FIELDS,
            name="runtime plan checkpoint descriptor record",
        )
        fields = _checkpoint_mapping(
            record["fields"],
            expected_fields=set(_DESCRIPTOR_CHECKPOINT_INIT_FIELDS),
            name="runtime plan checkpoint descriptor fields",
        )
        expected_descriptor_hash = _identity(
            record["descriptor_hash"],
            name="checkpoint descriptor hash",
        )
        try:
            descriptor = V1LabelNativeRuntimeFallbackDescriptor(
                _issuer_token=_DESCRIPTOR_ISSUER_TOKEN,
                **{name: fields[name] for name in _DESCRIPTOR_CHECKPOINT_INIT_FIELDS},
            )
            descriptor.assert_integrity(expected_descriptor_hash)
        except V1LabelNativeRuntimePlanRefusal:
            raise
        except (TypeError, ValueError) as error:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.CHECKPOINT_PAYLOAD_INVALID,
                f"runtime plan checkpoint descriptor failed replay: {error}",
            )
        descriptors.append(descriptor)
    plan_fields = _checkpoint_mapping(
        root["plan_fields"],
        expected_fields=set(_PLAN_CHECKPOINT_INIT_FIELDS),
        name="runtime plan checkpoint plan fields",
    )
    expected_plan_hash = _identity(
        root["plan_hash"],
        name="checkpoint runtime plan hash",
    )
    if (
        plan_fields["authorized_descriptor_count"] != len(descriptors)
        or tuple(plan_fields["ordered_descriptor_hashes"])
        != tuple(value.descriptor_hash for value in descriptors)
    ):
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.CHECKPOINT_PAYLOAD_INVALID,
            "runtime plan checkpoint descriptor ledger is stale",
        )
    try:
        plan = V1LabelNativeRuntimeFallbackPlan(
            _issuer_token=_PLAN_ISSUER_TOKEN,
            **{name: plan_fields[name] for name in _PLAN_CHECKPOINT_INIT_FIELDS},
            descriptors=tuple(descriptors),
        )
        plan.assert_integrity(expected_plan_hash)
    except V1LabelNativeRuntimePlanRefusal:
        raise
    except (TypeError, ValueError) as error:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.CHECKPOINT_PAYLOAD_INVALID,
            f"runtime plan checkpoint plan failed replay: {error}",
        )
    return plan


def _validate_qualified_branch(
    *,
    six_role_C_design: V1LabelNativeSixRoleCDesign,
    bridge_family: V1SignedWorkLabelNativeCBridgeFamilyBundle,
    branch: V1RelaxedKKTBranch,
) -> tuple[
    V1LabelNativePackedDirectionReceipt,
    V1LabelNativeQualifiedLaneRegistration,
]:
    branch_design = six_role_C_design.branch_design(branch)
    bridge = _branch_receipt(bridge_family, branch)
    receipt = branch_design.packed_direction_receipt
    subset = branch_design.eligible_action_subset_registration
    if type(receipt) is not V1LabelNativePackedDirectionReceipt:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
            "branch has no exact packed-direction receipt",
        )
    qualified = receipt.qualified_lane_registration
    if (
        type(qualified) is not V1LabelNativeQualifiedLaneRegistration
        or subset.qualified_lane_registration_hash is None
        or not subset.provider_native_subset_execution_available
        or subset.preoutcome_registered_column_filter_required
    ):
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.QUALIFIED_EXECUTION_REQUIRED,
            "canonical planning requires native pre-C qualified-lane execution",
        )
    if receipt.backend_kind != "torch_cuda":
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.REFERENCE_BACKEND_FORBIDDEN,
            "reference/full-axis execution cannot authorize a production plan",
        )
    try:
        qualified_hash = qualified.assert_integrity()
    except (TypeError, ValueError) as error:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.SOURCE_REPLAY_MISMATCH,
            f"qualified lane registration no longer replays: {error}",
        )
    crosswalks = bridge.eligible_action_crosswalks
    expected_source_labels = tuple(value.source_endpoint_label for value in crosswalks)
    expected_counts = tuple(value.cumulative_removed_count for value in crosswalks)
    expected_actions = tuple(value.packed_action_hash for value in crosswalks)
    if (
        receipt.schema != LABEL_NATIVE_QUALIFIED_DIRECTION_RECEIPT_SCHEMA
        or qualified_hash != subset.qualified_lane_registration_hash
        or qualified.compact_C_bridge_branch_receipt_hash != bridge.receipt_hash
        or qualified.action_axis_registration_hash
        != branch_design.action_axis_registration.registration_hash
        or qualified.source_endpoint_labels != expected_source_labels
        or qualified.endpoint_counts != expected_counts
        or qualified.endpoint_action_hashes != expected_actions
        or qualified.action_crosswalk_receipt_hashes
        != tuple(value.receipt_hash for value in crosswalks)
        or receipt.compact_provider_to_source_endpoint_labels
        != expected_source_labels
        or receipt.executed_source_endpoint_labels != expected_source_labels
        or receipt.executed_endpoint_counts != expected_counts
        or receipt.executed_endpoint_action_hashes != expected_actions
        or qualified.provider_route_scale_label_or_index_D2H_bytes != 0
    ):
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.SOURCE_REPLAY_MISMATCH,
            "qualified provider axis differs from the exact compact-C bridge",
        )
    return receipt, qualified


def issue_v1_signed_work_label_native_pre_O_runtime_fallback_plan(
    *,
    six_role_C_design: V1LabelNativeSixRoleCDesign,
    progress_receipt: V1LabelNativeSixRoleProgressReceipt,
    bridge_family: V1SignedWorkLabelNativeCBridgeFamilyBundle,
    certification_receipt: V1SignedWorkSixRoleCertificationReceipt,
    decision: V1SignedWorkSixRoleDecision,
    decision_design: V1SignedWorkSixRoleDecisionDesign,
    campaign_identity_hash: str,
    campaign_contract_hash: str,
    campaign_registration_hash: str,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_route_axis_hash: str,
    live_proposal_generation: int,
) -> V1LabelNativeRuntimeFallbackPlan:
    """Replay and issue a qualified, receipt-only resolved-prefix plan."""

    if type(six_role_C_design) is not V1LabelNativeSixRoleCDesign:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
            "plan issuance requires the exact label-native C design",
        )
    if type(progress_receipt) is not V1LabelNativeSixRoleProgressReceipt:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
            "plan issuance requires the exact label-native C progress receipt",
        )
    if type(bridge_family) is not V1SignedWorkLabelNativeCBridgeFamilyBundle:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
            "plan issuance requires the exact compact-C bridge family",
        )
    if type(certification_receipt) is not V1SignedWorkSixRoleCertificationReceipt:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
            "plan issuance requires the exact six-role certification receipt",
        )
    if type(decision) is not V1SignedWorkSixRoleDecision:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
            "plan issuance requires the exact six-role decision",
        )
    if type(decision_design) is not V1SignedWorkSixRoleDecisionDesign:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
            "plan issuance requires the exact six-role decision design",
        )
    campaign = _identity(campaign_identity_hash, name="campaign identity hash")
    campaign_contract = _identity(
        campaign_contract_hash,
        name="campaign contract hash",
    )
    campaign_registration = _identity(
        campaign_registration_hash,
        name="campaign registration hash",
    )
    live_chart = _identity(live_chart_identity_hash, name="live chart identity hash")
    live_mask = _identity(live_current_mask_hash, name="live current mask hash")
    live_route = _identity(live_route_axis_hash, name="live route axis hash")
    generation = _nonnegative_int(
        live_proposal_generation,
        name="live proposal generation",
    )
    if (
        live_chart != six_role_C_design.chart_identity_hash
        or live_mask != six_role_C_design.current_mask_hash
        or live_route != six_role_C_design.route_axis_hash
        or generation != six_role_C_design.proposal_generation
    ):
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.STALE_LIVE_SOURCE,
            "live chart/mask/route/generation differs from the frozen C campaign",
        )

    try:
        six_role_C_design.assert_integrity()
        progress_receipt.assert_replays(six_role_C_design)
        bridge_family.assert_integrity()
        authorization = (
            authorize_v1_signed_work_label_native_six_role_pre_O_fallback_queue(
                six_role_C_design=six_role_C_design,
                bridge_family=bridge_family,
                progress_receipt=progress_receipt,
                certification_receipt=certification_receipt,
                decision=decision,
                decision_design=decision_design,
                live_chart_identity_hash=live_chart,
                live_current_mask_hash=live_mask,
                live_route_axis_hash=live_route,
                live_proposal_generation=generation,
            )
        )
    except V1LabelNativeRuntimePlanRefusal:
        raise
    except (TypeError, ValueError) as error:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.SOURCE_REPLAY_MISMATCH,
            f"label-native fallback source replay failed: {error}",
        )
    if type(authorization) is not V1SignedWorkSixRoleFallbackQueueAuthorization:
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.INVALID_TYPE,
            "source replay returned a noncanonical fallback authorization",
        )
    if (
        authorization.chart_identity_hash != live_chart
        or authorization.current_mask_hash != live_mask
        or authorization.route_axis_hash != live_route
        or authorization.route_count != six_role_C_design.route_count
        or authorization.proposal_generation != generation
    ):
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.STALE_LIVE_SOURCE,
            "live chart/mask/route/generation differs from the frozen C campaign",
        )
    if (
        authorization.calibration_K_open_authority
        or authorization.paired_oracle_O_open_authority
        or authorization.mask_commit_authority
        or authorization.terminal_authority
        or authorization.pareto_authority
        or not authorization.paired_bce_oracle_still_required
    ):
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.AUTHORITY_VIOLATION,
            "source fallback authorization crossed its pre-O boundary",
        )

    qualified_by_branch: dict[
        V1RelaxedKKTBranch,
        tuple[V1LabelNativePackedDirectionReceipt, V1LabelNativeQualifiedLaneRegistration],
    ] = {}
    for branch in six_role_C_design.branches:
        qualified_by_branch[branch] = _validate_qualified_branch(
            six_role_C_design=six_role_C_design,
            bridge_family=bridge_family,
            branch=branch,
        )

    certificate_by_hash = {
        value.coordinate.coordinate_hash: value
        for value in certification_receipt.endpoint_certificates
    }
    if len(certificate_by_hash) != len(decision_design.endpoint_coordinates):
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.SOURCE_REPLAY_MISMATCH,
            "certification coordinate axis is incomplete or duplicated",
        )
    ordered = tuple(
        sorted(
            decision_design.endpoint_coordinates,
            key=lambda value: (
                -value.cumulative_removed_count,
                value.pre_C_priority,
            ),
        )
    )
    coordinate_by_physical: dict[str, list[V1SignedWorkSixRoleEndpointCoordinate]] = {}
    for coordinate in ordered:
        coordinate_by_physical.setdefault(
            coordinate.canonical_candidate_identity_hash,
            [],
        ).append(coordinate)
    for physical_coordinates in coordinate_by_physical.values():
        signatures = {
            _certificate_signature(certificate_by_hash[value.coordinate_hash])
            for value in physical_coordinates
        }
        if len(signatures) != 1:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.DUPLICATE_REPLAY_MISMATCH,
                "a physical duplicate class differs in F/D/Q/R replay",
            )

    authorization_hash = authorization.authorization_hash
    descriptors: list[V1LabelNativeRuntimeFallbackDescriptor] = []
    for queue_index, coordinate in enumerate(authorization.fallback_coordinates):
        certificate = certificate_by_hash.get(coordinate.coordinate_hash)
        if certificate is None:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.SOURCE_REPLAY_MISMATCH,
                "authorized coordinate has no exact endpoint certificate",
            )
        branch_design = six_role_C_design.branch_design(coordinate.branch)
        bridge = _branch_receipt(bridge_family, coordinate.branch)
        crosswalk_index = coordinate.endpoint_index - 1
        if not 0 <= crosswalk_index < len(bridge.eligible_action_crosswalks):
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.SOURCE_REPLAY_MISMATCH,
                "authorized coordinate left the compact-C crosswalk axis",
            )
        crosswalk = bridge.eligible_action_crosswalks[crosswalk_index]
        receipt, qualified = qualified_by_branch[coordinate.branch]
        provider_label = coordinate.endpoint_index
        try:
            mapped_source_label = qualified.source_label_for_provider_label(
                provider_label
            )
        except (TypeError, ValueError) as error:
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.SOURCE_REPLAY_MISMATCH,
                f"compact provider label does not replay: {error}",
            )
        source_ladder = receipt.action_axis_registration.source_shell_ladder
        source_index = coordinate.endpoint_label - 1
        if not 0 <= source_index < len(source_ladder.actions):
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.SOURCE_REPLAY_MISMATCH,
                "authorized source label left the complete shell ladder",
            )
        source_action = source_ladder.actions[source_index]
        action_identity = branch_design.action_identities[crosswalk_index]
        if (
            mapped_source_label != coordinate.endpoint_label
            or qualified.compact_provider_endpoint_labels[crosswalk_index]
            != provider_label
            or action_identity.endpoint_index != coordinate.endpoint_index
            or action_identity.source_endpoint_label != coordinate.endpoint_label
            or action_identity.source_action_receipt != source_action
            or action_identity.bridge_action_crosswalk_receipt_hash
            != crosswalk.receipt_hash
            or crosswalk.source_action_receipt != source_action
            or certificate.certificate_hash
            != authorization.fallback_certificate_hashes[queue_index]
            or certificate.direct_D_status is not V1SignedWorkIntervalStatus.CERTIFIED
            or certificate.admission_basis
            != authorization.fallback_admission_bases[queue_index]
            or certificate.chart_transport_status
            is not authorization.fallback_chart_transport_statuses[queue_index]
            or coordinate.canonical_candidate_identity_hash
            != authorization.queued_physical_candidate_identity_hashes[queue_index]
        ):
            _refuse(
                V1LabelNativeRuntimePlanRefusalCode.SOURCE_REPLAY_MISMATCH,
                "authorized coordinate failed provider/crosswalk/action replay",
            )
        duplicate_coordinates = tuple(
            coordinate_by_physical[coordinate.canonical_candidate_identity_hash]
        )
        descriptors.append(
            V1LabelNativeRuntimeFallbackDescriptor(
                _issuer_token=_DESCRIPTOR_ISSUER_TOKEN,
                queue_index=queue_index,
                coordinate=coordinate,
                certificate=certificate,
                crosswalk=crosswalk,
                source_action_receipt=source_action,
                compact_provider_endpoint_label=provider_label,
                source_endpoint_label=mapped_source_label,
                physical_duplicate_coordinate_hashes=tuple(
                    value.coordinate_hash for value in duplicate_coordinates
                ),
                physical_duplicate_certificate_hashes=tuple(
                    certificate_by_hash[value.coordinate_hash].certificate_hash
                    for value in duplicate_coordinates
                ),
                provider_receipt_hash=receipt.receipt_hash,
                provider_design_hash=receipt.design_hash,
                provider_policy_hash=receipt.policy_hash,
                provider_forecast_hash=receipt.forecast_hash,
                provider_backend_kind=receipt.backend_kind,
                provider_device_identity=receipt.device_identity,
                prefix_rank_receipt_hash=receipt.prefix_rank_receipt_hash,
                endpoint_query_hash=receipt.endpoint_query_hash,
                resulting_label_commitment=receipt.resulting_label_commitment,
                qualified_lane_registration_hash=qualified.registration_hash,
                action_axis_registration_hash=(
                    receipt.action_axis_registration.registration_hash
                ),
                source_shell_ladder_receipt_hash=source_ladder.receipt_hash,
                eligible_subset_registration_hash=(
                    branch_design.eligible_action_subset_registration.registration_hash
                ),
                branch_C_design_hash=branch_design.design_hash,
                bridge_branch_receipt_hash=bridge.receipt_hash,
                fallback_authorization_hash=authorization_hash,
                six_role_C_design_hash=six_role_C_design.design_hash,
                six_role_C_progress_receipt_hash=progress_receipt.receipt_hash,
                six_role_certification_receipt_hash=(
                    certification_receipt.receipt_hash
                ),
                six_role_decision_hash=decision.decision_hash,
                six_role_decision_design_hash=decision_design.design_hash,
                bridge_family_bundle_hash=bridge_family.bundle_hash,
                campaign_identity_hash=campaign,
                campaign_contract_hash=campaign_contract,
                campaign_registration_hash=campaign_registration,
                live_chart_identity_hash=live_chart,
                live_current_mask_hash=live_mask,
                live_route_axis_hash=live_route,
                live_route_count=six_role_C_design.route_count,
                live_proposal_generation=generation,
                native_qualified_subset_execution_replayed=True,
                exact_source_replay_completed=True,
                provider_or_handle_retained=False,
                mask_or_route_payload_retained=False,
                calibration_K_open_authority=False,
                paired_oracle_O_open_authority=False,
                mask_commit_authority=False,
                terminal_authority=False,
                pareto_authority=False,
                execution_authority=False,
                authority_scope=LABEL_NATIVE_RUNTIME_PLAN_AUTHORITY,
            )
        )

    descriptor_axis = tuple(descriptors)
    if tuple(value.coordinate_hash for value in descriptor_axis) != tuple(
        value.coordinate_hash for value in authorization.fallback_coordinates
    ):
        _refuse(
            V1LabelNativeRuntimePlanRefusalCode.ORDER_MISMATCH,
            "runtime issuer changed the source-authorized prefix",
        )
    return V1LabelNativeRuntimeFallbackPlan(
        _issuer_token=_PLAN_ISSUER_TOKEN,
        campaign_identity_hash=campaign,
        campaign_contract_hash=campaign_contract,
        campaign_registration_hash=campaign_registration,
        live_chart_identity_hash=live_chart,
        live_current_mask_hash=live_mask,
        live_route_axis_hash=live_route,
        live_route_count=six_role_C_design.route_count,
        live_proposal_generation=generation,
        six_role_C_design_hash=six_role_C_design.design_hash,
        six_role_C_progress_receipt_hash=progress_receipt.receipt_hash,
        six_role_certification_receipt_hash=certification_receipt.receipt_hash,
        six_role_decision_hash=decision.decision_hash,
        six_role_decision_design_hash=decision_design.design_hash,
        bridge_family_bundle_hash=bridge_family.bundle_hash,
        fallback_authorization_hash=authorization_hash,
        selected_coordinate_hash=authorization.selected_coordinate.coordinate_hash,
        descriptors=descriptor_axis,
        ordered_descriptor_hashes=tuple(
            value.descriptor_hash for value in descriptor_axis
        ),
        ordered_coordinate_hashes=tuple(
            value.coordinate_hash for value in descriptor_axis
        ),
        ordered_physical_candidate_identity_hashes=tuple(
            value.canonical_physical_identity_hash for value in descriptor_axis
        ),
        blocked_unresolved_coordinate_hash=(
            authorization.blocked_unresolved_coordinate_hash
        ),
        fallback_exhaustion_status=authorization.fallback_exhaustion_status,
        authorized_descriptor_count=len(descriptor_axis),
        resolved_prefix_frozen_before_K_O=True,
        native_qualified_subset_execution_replayed=True,
        source_authorized_duplicate_quotient_only=True,
        stops_before_first_unresolved_decision_changing_coordinate=True,
        no_provider_handle_mask_or_route_payload_retained=True,
        paired_bce_oracle_still_required=True,
        candidate_selection_authority=False,
        calibration_K_open_authority=False,
        paired_oracle_O_open_authority=False,
        irreversible_decision_authority=False,
        mask_commit_authority=False,
        terminal_authority=False,
        pareto_authority=False,
        execution_authority=False,
        authority_scope=LABEL_NATIVE_RUNTIME_PLAN_AUTHORITY,
    )


__all__ = [
    "LABEL_NATIVE_RUNTIME_DESCRIPTOR_SCHEMA",
    "LABEL_NATIVE_RUNTIME_PLAN_AUTHORITY",
    "LABEL_NATIVE_RUNTIME_PLAN_CHECKPOINT_PAYLOAD_SCHEMA",
    "LABEL_NATIVE_RUNTIME_PLAN_SCHEMA",
    "V1LabelNativeRuntimeFallbackDescriptor",
    "V1LabelNativeRuntimeFallbackPlan",
    "V1LabelNativeRuntimePlanRefusal",
    "V1LabelNativeRuntimePlanRefusalCode",
    "export_v1_label_native_runtime_plan_checkpoint_payload",
    "issue_v1_signed_work_label_native_pre_O_runtime_fallback_plan",
    "restore_v1_label_native_runtime_plan_checkpoint_payload",
]
