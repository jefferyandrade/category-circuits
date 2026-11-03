"""Branch-independent source issuance for sampled V7 ``D_W2``.

The shared measurement owner in this module is issued before the fresh
residual-population registration exists.  It owns the residual stream names,
alpha coordinates, capture policy, complete suffix schedule, and logical
packet request without seeing a preferred action or a branch-local compact
support receipt.  A population registration may subsequently attach only when
it repeats those source-issued coordinates exactly.

Branch-local compact supports are lowered into distinct append lineages after
that attachment.  They share the measurement lineage and schedule but retain
their own resource and CUDA-design custody.  This module deliberately exposes
no physical acquisition/fan-out execution, D decision, stability, C/K/O,
action, commit, terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from threading import RLock
from typing import Final, NoReturn, Sequence

from mechanistic_systems.circuits.irg.modern_v1_alexnet_fq_registration import (
    V1AlexNetFQCausalGenerationFrameReceipt,
    V1AlexNetFQCausalRoleSamplingStreamReceipt,
)
from mechanistic_systems.circuits.irg.modern_v1_campaign_contract import (
    V1OptionACampaignBinding,
)
from mechanistic_systems.circuits.irg.modern_v1_literal_j_d_fixed_residual_population_projection import (
    V1LiteralJDFixedResidualPopulationProjectionRegistration,
)
from mechanistic_systems.circuits.irg.modern_v1_neutral_runtime import (
    DeclaredRouteAxis,
    FixedAffineBCECalibration,
)
from mechanistic_systems.circuits.irg.modern_v1_native_fiber_axis import (
    V1NativeDestinationFiberAxisReceipt,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_residual_population_measure import (
    V1SampledResidualPopulationMeasureDesign,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_v7_resident_d1_literal_j_population import (
    V1SampledV7ResidentD1LiteralJPopulationResult,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_variable_shell_dw2_suffix import (
    V1SampledVariableShellDW2SuffixPacket,
    V1SampledVariableShellDW2SuffixRegistration,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_variable_shell_packet_registration import (
    SAMPLED_VARIABLE_SHELL_PACKET_REQUEST_VERSION,
    SAMPLED_VARIABLE_SHELL_TWO_BRANCH_KKT_ID,
    V1SampledVariableShellMeasurementSchedule,
    V1SampledVariableShellMeasurementSourceLineage,
    _issue_v1_sampled_variable_shell_measurement_schedule,
    _issue_v1_sampled_variable_shell_measurement_source_lineage,
    v1_sampled_variable_shell_common_measure_unit_hash,
    v1_sampled_variable_shell_packet_request_payload,
    v1_sampled_variable_shell_packet_request_state_hash,
    v1_sampled_variable_shell_stratum_measure_source_hash,
)
from mechanistic_systems.circuits.irg.modern_v1_signed_work_fixed_primal_shell_union import (
    V1SignedWorkFixedPrimalShellUnionReceipt,
)
from mechanistic_systems.circuits.irg.modern_v1_signed_work_variable_shell_source_append import (
    V1SignedWorkVariableShellSourceAppendLineage,
    V1SignedWorkVariableShellSourceAppendOwner,
)
from mechanistic_systems.circuits.irg.modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceCUDAExecutionDesign,
    V1SignedWorkVariableShellSourceDeviceHandle,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.contracts.measurement import (
    ContextSelector,
    MeasurementProvenance,
    MeasurementRequest,
)
from mechanistic_systems.contracts.science import QuerySpec
from mechanistic_systems.core.axis_identity import (
    ConstructiveOrdinalRouteAxis,
    ordered_string_axis_hash,
)
from mechanistic_systems.measurement.optimized_irg_providers import (
    OptimizedScoringResponseProvider,
)


SAMPLED_V7_D_ONLY_MEASUREMENT_OWNER_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_only_measurement_owner_v3"
)
SAMPLED_V7_D_ONLY_POPULATION_ATTACHMENT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_only_population_attachment_v3"
)
SAMPLED_V7_D_ONLY_BRANCH_SOURCE_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_only_branch_source_binding_v2"
)
SAMPLED_V7_D_ONLY_PACKET_ISSUANCE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_only_packet_measurement_issuance_v3"
)
SAMPLED_V7_D_ONLY_LOGICAL_LOOK_ISSUANCE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_only_logical_look_measurement_issuance_v1"
)
SAMPLED_V7_D_ONLY_SOURCE_AUTHORITY: Final = (
    "source_schedule_logical_request_and_branch_append_custody_only_no_physical_"
    "fanout_alpha_spend_outcomes_D_stability_C_K_O_action_commit_terminal_or_"
    "pareto_authority_v3"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-D-only-source-issuance-"
_MEASUREMENT_TOKEN: Final = object()
_ATTACHMENT_TOKEN: Final = object()
_BRANCH_TOKEN: Final = object()
_PACKET_TOKEN: Final = object()

_POPULATION_SOURCE_RESIDENT_D1: Final = "resident_d1"
_POPULATION_SOURCE_AFFINE_SHELL: Final = "affine_shell"


class V1SampledV7DOnlySourceIssuanceRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    AXIS_MISMATCH = "axis_mismatch"
    CALIBRATION_MISMATCH = "calibration_mismatch"
    POPULATION_MISMATCH = "population_mismatch"
    SUPPORT_MISMATCH = "support_mismatch"
    ORDER_MISMATCH = "order_mismatch"
    REPLAY = "replay"
    FANOUT_ABI_MISSING = "fanout_abi_missing"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7DOnlySourceIssuanceRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7DOnlySourceIssuanceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7DOnlySourceIssuanceRefusalCode:
            raise TypeError("source-issuance refusal code must have its exact enum type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7DOnlySourceIssuanceRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7DOnlySourceIssuanceRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
            f"{name} must be one nonempty source identity",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    if value < 0:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.AXIS_MISMATCH,
            f"{name} must be nonnegative",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.AXIS_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _count_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable target/complement count pair",
        )
    return (
        _nonnegative_int(value[0], name=f"{name} target"),
        _nonnegative_int(value[1], name=f"{name} complement"),
    )


class _PacketIssuanceLease:
    __slots__ = ("_issued", "_lock")

    def __init__(self) -> None:
        self._issued: set[str] = set()
        self._lock = RLock()

    def reserve(self, packet_hash: str) -> None:
        with self._lock:
            if packet_hash in self._issued:
                _refuse(
                    V1SampledV7DOnlySourceIssuanceRefusalCode.REPLAY,
                    "this frozen suffix packet already received its shared request",
                )
            self._issued.add(packet_hash)

    def reserve_many(self, packet_hashes: Sequence[str]) -> None:
        hashes = tuple(_identity(value, name="packet hash") for value in packet_hashes)
        if not hashes or len(set(hashes)) != len(hashes):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.REPLAY,
                "a logical look must reserve one nonrepeating transport packet axis",
            )
        with self._lock:
            already = tuple(value for value in hashes if value in self._issued)
            if already:
                _refuse(
                    V1SampledV7DOnlySourceIssuanceRefusalCode.REPLAY,
                    "one frozen suffix packet in the logical look already "
                    "received its shared request",
                )
            self._issued.update(hashes)

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("V7 measurement-owner leases are nonserializable")


def _scorer_source_snapshot(
    *,
    scorer: OptimizedScoringResponseProvider,
    current_mask: HardMask,
    calibration: FixedAffineBCECalibration,
    binding: V1OptionACampaignBinding,
    route_ids: ConstructiveOrdinalRouteAxis,
) -> dict[str, object]:
    if type(scorer) is not OptimizedScoringResponseProvider:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
            "production source issuance requires the exact optimized scorer",
        )
    scorer_routes = getattr(scorer, "route_ids", None)
    queries = tuple(getattr(scorer, "queries", ()))
    if (
        not isinstance(scorer_routes, Sequence)
        or isinstance(scorer_routes, (str, bytes))
        or len(scorer_routes) != len(route_ids)
        or ordered_string_axis_hash(
            scorer_routes, name="sampled V7 D-only scorer route axis"
        )
        != ordered_string_axis_hash(
            route_ids, name="sampled V7 D-only constructive route axis"
        )
        or len(queries) != 1
        or type(queries[0]) is not QuerySpec
    ):
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.AXIS_MISMATCH,
            "scorer route/query axes differ from the declared physical source",
        )
    query = queries[0]
    try:
        resolved_mask = scorer.mask_resolver(current_mask.mask_hash)
        resolved_calibration = scorer.calibration_resolver(resolved_mask)
    except Exception as error:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.CALIBRATION_MISMATCH,
            f"scorer could not resolve the current mask/calibration source: {error}",
        )
    if (
        getattr(resolved_mask, "mask_hash", None) != current_mask.mask_hash
        or getattr(resolved_mask, "route_axis_hash", None)
        != current_mask.route_axis_hash
        or type(resolved_calibration) is not FixedAffineBCECalibration
        or resolved_calibration.content_hash != calibration.content_hash
        or query.query_id != binding.contract.query_id
        or query.content_hash != calibration.query_hash
        or calibration.query_id != binding.contract.query_id
        or calibration.calibration_state_id != binding.contract.calibration_identity
    ):
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.CALIBRATION_MISMATCH,
            "scorer, query, mask fit, campaign, and calibration differ",
        )
    try:
        route_chunk_size = int(getattr(scorer, "response_route_chunk_size"))
        context_packet_size = int(getattr(scorer, "context_packet_size"))
    except Exception:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
            "scorer did not expose its registered packet geometry",
        )
    if route_chunk_size <= 0 or context_packet_size <= 0:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
            "scorer packet geometry must be positive",
        )
    return {
        "study_hash": _identity(
            getattr(scorer, "study_hash", None), name="scorer study hash"
        ),
        "query_hash": query.content_hash,
        "calibration_hash": calibration.content_hash,
        "destination_transport_hash": _identity(
            getattr(scorer, "local_destination_transport_hash", None),
            name="scorer destination transport hash",
        ),
        "response_route_chunk_size": route_chunk_size,
        "context_packet_size": context_packet_size,
    }


def _validate_common_sources(
    *,
    frame: V1AlexNetFQCausalGenerationFrameReceipt,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    residual_measure_design: V1SampledResidualPopulationMeasureDesign,
    resident_d1_population: V1SampledV7ResidentD1LiteralJPopulationResult,
    shell_union: V1SignedWorkFixedPrimalShellUnionReceipt,
    current_mask: HardMask,
    campaign_binding: V1OptionACampaignBinding,
    calibration: FixedAffineBCECalibration,
    scorer: OptimizedScoringResponseProvider,
    physical_route_axis: DeclaredRouteAxis,
    constructive_route_ids: ConstructiveOrdinalRouteAxis,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
) -> tuple[dict[str, object], str]:
    exact = (
        (frame, V1AlexNetFQCausalGenerationFrameReceipt),
        (discovery_stream, V1AlexNetFQCausalRoleSamplingStreamReceipt),
        (suffix_registration, V1SampledVariableShellDW2SuffixRegistration),
        (residual_measure_design, V1SampledResidualPopulationMeasureDesign),
        (resident_d1_population, V1SampledV7ResidentD1LiteralJPopulationResult),
        (shell_union, V1SignedWorkFixedPrimalShellUnionReceipt),
        (current_mask, HardMask),
        (campaign_binding, V1OptionACampaignBinding),
        (calibration, FixedAffineBCECalibration),
        (scorer, OptimizedScoringResponseProvider),
        (physical_route_axis, DeclaredRouteAxis),
        (constructive_route_ids, ConstructiveOrdinalRouteAxis),
        (native_fiber_axis_receipt, V1NativeDestinationFiberAxisReceipt),
    )
    if any(type(value) is not expected for value, expected in exact):
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
            "measurement ownership requires exact live production sources",
        )
    suffix_registration.assert_integrity()
    suffix_registration.assert_parent_graph(
        frame=frame, discovery_stream=discovery_stream
    )
    residual_measure_design.assert_integrity()
    native_fiber_axis_receipt.assert_integrity()
    resident_d1_population.device_handle.assert_live_identity()
    resident = resident_d1_population.receipt
    resident.assert_integrity()
    route_id_hash = ordered_string_axis_hash(
        constructive_route_ids,
        name="sampled V7 D-only constructive route axis",
    )
    shell_fibers = tuple(
        native_fiber_axis_receipt.route_to_fiber(index).global_fiber_index
        for index in shell_union.shell_indices
    )
    if (
        frame.current_mask_hash != current_mask.mask_hash
        or suffix_registration.current_mask_hash != current_mask.mask_hash
        or residual_measure_design.current_mask_hash != current_mask.mask_hash
        or shell_union.current_mask_hash != current_mask.mask_hash
        or resident.current_mask_hash != current_mask.mask_hash
        or suffix_registration.generation_index != frame.generation_index
        or resident.proposal_generation != frame.generation_index
        or residual_measure_design.generation_index != frame.generation_index
        or residual_measure_design.suffix_registration_hash
        != suffix_registration.receipt_hash
        or resident.d1_stratum_counts
        != suffix_registration.consumed_d1_stratum_counts
        or campaign_binding.contract.model_seal_hash != resident.model_identity_hash
        or campaign_binding.contract.route_axis_hash
        != physical_route_axis.route_axis_hash
        or current_mask.route_axis_hash != physical_route_axis.route_axis_hash
        or physical_route_axis.route_axis_hash != native_fiber_axis_receipt.route_axis_hash
        or physical_route_axis.route_id_axis_hash != route_id_hash
        or native_fiber_axis_receipt.route_id_axis_hash != route_id_hash
        or resident.route_axis_hash != physical_route_axis.route_axis_hash
        or resident.route_id_axis_hash != route_id_hash
        or resident.native_fiber_axis_receipt_hash
        != native_fiber_axis_receipt.receipt_hash
        or resident.route_count != physical_route_axis.route_count
        or shell_union.route_axis_hash != physical_route_axis.route_axis_hash
        or shell_union.route_id_axis_hash != route_id_hash
        or shell_union.shell_route_ids
        != tuple(constructive_route_ids[index] for index in shell_union.shell_indices)
        or len(shell_fibers) != len(shell_union.shell_indices)
    ):
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.AXIS_MISMATCH,
            "resident D1, suffix, shell, campaign, mask, route, and native-fiber axes diverged",
        )
    snapshot = _scorer_source_snapshot(
        scorer=scorer,
        current_mask=current_mask,
        calibration=calibration,
        binding=campaign_binding,
        route_ids=constructive_route_ids,
    )
    source_graph = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_only_common_source_graph_v2",
            "generation_frame_hash": frame.receipt_hash,
            "discovery_stream_hash": discovery_stream.receipt_hash,
            "suffix_registration_hash": suffix_registration.receipt_hash,
            "residual_measure_design_hash": residual_measure_design.design_hash,
            "resident_d1_receipt_hash": resident.receipt_hash,
            "resident_d1_live_identity_hash": (
                resident_d1_population.device_handle.assert_live_identity()
            ),
            "shell_union_receipt_hash": shell_union.receipt_hash,
            "current_mask_hash": current_mask.mask_hash,
            "campaign_binding_hash": campaign_binding.binding_hash,
            "calibration_hash": calibration.content_hash,
            "scorer_source": snapshot,
            "physical_route_axis_hash": physical_route_axis.content_hash,
            "constructive_route_id_axis_hash": route_id_hash,
            "native_fiber_axis_receipt_hash": native_fiber_axis_receipt.receipt_hash,
            "native_fiber_measure_hash": native_fiber_axis_receipt.measure_hash,
            "preferred_action_present": False,
            "branch_support_receipt_present": False,
        },
        prefix=_PREFIX + "common-source-graph-",
    )
    return snapshot, source_graph


@dataclass(frozen=True, slots=True)
class V1SampledV7DOnlyMeasurementOwner:
    source_graph_hash: str
    capture_policy_identity_hash: str
    residual_witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    residual_literal_j_stream_source_hash: str
    proposal_influencing_history_hash: str
    residual_design_lineage_hash: str
    alpha_ledger_hash: str
    alpha_coordinate_hash: str
    alpha_lineage_hash: str
    familywise_alpha: float
    population_registration_id: str
    schedule: V1SampledVariableShellMeasurementSchedule
    measurement_source_lineage: V1SampledVariableShellMeasurementSourceLineage
    source_outcomes_seen: bool = False
    branch_support_receipt_present: bool = False
    preferred_action_chosen: bool = False
    physical_acquisition_authority: bool = False
    alpha_spend_authority: bool = False
    trajectory_nonrecycling_alpha_closed: bool = False
    d_stability_authority: bool = False
    c_k_o_authority: bool = False
    action_or_commit_authority: bool = False
    authority_scope: str = SAMPLED_V7_D_ONLY_SOURCE_AUTHORITY
    schema: str = SAMPLED_V7_D_ONLY_MEASUREMENT_OWNER_SCHEMA
    _frame: V1AlexNetFQCausalGenerationFrameReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _suffix_registration: V1SampledVariableShellDW2SuffixRegistration = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _residual_measure_design: V1SampledResidualPopulationMeasureDesign = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _resident_d1_population: V1SampledV7ResidentD1LiteralJPopulationResult = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _shell_union: V1SignedWorkFixedPrimalShellUnionReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _current_mask: HardMask = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _campaign_binding: V1OptionACampaignBinding = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _calibration: FixedAffineBCECalibration = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _scorer: OptimizedScoringResponseProvider = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _physical_route_axis: DeclaredRouteAxis = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _constructive_route_ids: ConstructiveOrdinalRouteAxis = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _lease: _PacketIssuanceLease = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuance_token: InitVar[object | None] = None
    _owner_hash: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _MEASUREMENT_TOKEN:
            raise PermissionError("V7 measurement owners are issuer-only")
        exact = (
            (self.schedule, V1SampledVariableShellMeasurementSchedule),
            (
                self.measurement_source_lineage,
                V1SampledVariableShellMeasurementSourceLineage,
            ),
            (self._frame, V1AlexNetFQCausalGenerationFrameReceipt),
            (self._discovery_stream, V1AlexNetFQCausalRoleSamplingStreamReceipt),
            (self._suffix_registration, V1SampledVariableShellDW2SuffixRegistration),
            (self._residual_measure_design, V1SampledResidualPopulationMeasureDesign),
            (
                self._resident_d1_population,
                V1SampledV7ResidentD1LiteralJPopulationResult,
            ),
            (self._shell_union, V1SignedWorkFixedPrimalShellUnionReceipt),
            (self._current_mask, HardMask),
            (self._campaign_binding, V1OptionACampaignBinding),
            (self._calibration, FixedAffineBCECalibration),
            (self._scorer, OptimizedScoringResponseProvider),
            (self._physical_route_axis, DeclaredRouteAxis),
            (self._constructive_route_ids, ConstructiveOrdinalRouteAxis),
            (self._native_fiber_axis_receipt, V1NativeDestinationFiberAxisReceipt),
            (self._lease, _PacketIssuanceLease),
        )
        if any(type(value) is not expected for value, expected in exact):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
                "measurement owner lost an exact live source",
            )
        closed = (
            self.source_outcomes_seen,
            self.branch_support_receipt_present,
            self.preferred_action_chosen,
            self.physical_acquisition_authority,
            self.alpha_spend_authority,
            self.trajectory_nonrecycling_alpha_closed,
            self.d_stability_authority,
            self.c_k_o_authority,
            self.action_or_commit_authority,
        )
        if (
            any(type(value) is not bool or value is not False for value in closed)
            or self.authority_scope != SAMPLED_V7_D_ONLY_SOURCE_AUTHORITY
            or self.schema != SAMPLED_V7_D_ONLY_MEASUREMENT_OWNER_SCHEMA
        ):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.AUTHORITY_VIOLATION,
                "measurement owner crossed its branch-independent source boundary",
            )
        self.assert_live()
        object.__setattr__(self, "_owner_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "source_graph_hash": self.source_graph_hash,
                "capture_policy_identity_hash": self.capture_policy_identity_hash,
                "stream_axis": (
                    self.residual_witness_stream_source_hash,
                    self.same_capture_vjp_source_hash,
                    self.corrected_evidence_stream_source_hash,
                    self.residual_literal_j_stream_source_hash,
                ),
                "alpha_axis": (
                    self.alpha_ledger_hash,
                    self.alpha_coordinate_hash,
                    self.alpha_lineage_hash,
                    self.familywise_alpha,
                ),
                "population_registration_id": self.population_registration_id,
                "schedule_hash": self.schedule.schedule_hash,
                "measurement_source_lineage_hash": (
                    self.measurement_source_lineage.lineage_hash
                ),
                "branch_support_receipt_present": False,
                "preferred_action_chosen": False,
                "physical_acquisition_authority": False,
                "alpha_spend_authority": False,
                "trajectory_nonrecycling_alpha_closed": False,
                "action_or_commit_authority": False,
            },
            prefix=_PREFIX + "measurement-owner-",
        )

    def assert_live(self) -> str:
        _snapshot, observed_graph = _validate_common_sources(
            frame=self._frame,
            discovery_stream=self._discovery_stream,
            suffix_registration=self._suffix_registration,
            residual_measure_design=self._residual_measure_design,
            resident_d1_population=self._resident_d1_population,
            shell_union=self._shell_union,
            current_mask=self._current_mask,
            campaign_binding=self._campaign_binding,
            calibration=self._calibration,
            scorer=self._scorer,
            physical_route_axis=self._physical_route_axis,
            constructive_route_ids=self._constructive_route_ids,
            native_fiber_axis_receipt=self._native_fiber_axis_receipt,
        )
        self.schedule.assert_parent_graph(
            frame=self._frame,
            discovery_stream=self._discovery_stream,
            suffix_registration=self._suffix_registration,
        )
        self.measurement_source_lineage.assert_integrity()
        if (
            observed_graph != self.source_graph_hash
            or self.schedule.capture_policy_identity_hash
            != self.capture_policy_identity_hash
            or self.measurement_source_lineage.source_graph_hash
            != self.source_graph_hash
            or self.measurement_source_lineage.measurement_schedule_hash
            != self.schedule.schedule_hash
            or self.measurement_source_lineage.witness_stream_source_hash
            != self.residual_witness_stream_source_hash
            or self.measurement_source_lineage.same_capture_vjp_source_hash
            != self.same_capture_vjp_source_hash
            or self.measurement_source_lineage.corrected_evidence_stream_source_hash
            != self.corrected_evidence_stream_source_hash
            or self.measurement_source_lineage.literal_j_stream_source_hash
            != self.residual_literal_j_stream_source_hash
            or self.measurement_source_lineage.alpha_ledger_id
            != self.alpha_ledger_hash
            or self.measurement_source_lineage.alpha_coordinate_id
            != self.alpha_coordinate_hash
            or self.measurement_source_lineage.alpha_lineage_hash
            != self.alpha_lineage_hash
            or self.measurement_source_lineage.alpha_spend_authority is not False
            or self.measurement_source_lineage.trajectory_nonrecycling_alpha_closed
            is not False
        ):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
                "live branch-independent source graph changed after issuance",
            )
        observed = self._compute_hash()
        if self._owner_hash is not None and observed != self._owner_hash:
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
                "measurement-owner identity changed",
            )
        return observed

    @property
    def owner_hash(self) -> str:
        return self.assert_live()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live V7 measurement owners are nonserializable")


def _classify_population_registration_source(
    *,
    owner: V1SampledV7DOnlyMeasurementOwner,
    registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
) -> str:
    registration.assert_integrity()
    resident_hash = owner._resident_d1_population.receipt.receipt_hash
    common = (
        registration.registration_id == owner.population_registration_id
        and registration.population_measure_design_hash
        == owner._residual_measure_design.design_hash
        and registration.suffix_registration_hash
        == owner._suffix_registration.receipt_hash
        and registration.model_identity_hash
        == owner.measurement_source_lineage.model_identity_hash
        and registration.chart_identity_hash
        == owner.measurement_source_lineage.chart_identity_hash
        and registration.current_mask_hash
        == owner.measurement_source_lineage.current_mask_hash
        and registration.proposal_generation
        == owner.measurement_source_lineage.proposal_generation
        and registration.route_axis_hash
        == owner.measurement_source_lineage.route_axis_hash
        and registration.route_id_axis_hash
        == owner.measurement_source_lineage.route_id_axis_hash
        and registration.route_count == owner._physical_route_axis.route_count
        and registration.population_stratum_counts
        == owner._residual_measure_design.population_stratum_counts
        and registration.d_fixed_stratum_counts
        == owner._residual_measure_design.d1_stratum_counts
        and registration.residual_stratum_counts
        == owner._residual_measure_design.residual_stratum_counts
        and registration.population_count
        == owner._residual_measure_design.population_count
        and registration.d_fixed_count
        == sum(owner._residual_measure_design.d1_stratum_counts)
        and registration.d_fixed_pair_prefix_axis_hash
        == owner._residual_measure_design.d1_pair_prefix_axis_hash
        and registration.residual_pair_suffix_axis_hash
        == owner._residual_measure_design.residual_pair_suffix_axis_hash
        and registration.residual_sealed_population_order_hashes
        == (
            owner._suffix_registration.target_suffix_axis_hash,
            owner._suffix_registration.complement_suffix_axis_hash,
        )
        and registration.residual_witness_stream_source_hash
        == owner.residual_witness_stream_source_hash
        and registration.residual_literal_j_stream_source_hash
        == owner.residual_literal_j_stream_source_hash
        and registration.proposal_influencing_history_hash
        == owner.proposal_influencing_history_hash
        and registration.residual_design_lineage_hash
        == owner.residual_design_lineage_hash
        and registration.alpha_ledger_hash == owner.alpha_ledger_hash
        and registration.alpha_coordinate_hash == owner.alpha_coordinate_hash
        and registration.alpha_lineage_hash == owner.alpha_lineage_hash
        and registration.familywise_alpha == owner.familywise_alpha
    )
    if not common:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.POPULATION_MISMATCH,
            "population registration did not repeat the source-issued streams, alpha, population, and chart axes",
        )
    legacy_resident = (
        registration.affine_shell_shard_replay_receipt_hash is None
        and registration.d_fixed_registration_hash == resident_hash
        and registration.d_fixed_shard_replay_receipt_hash == resident_hash
    )
    structured_affine = (
        registration.affine_shell_shard_replay_receipt_hash is not None
        and registration.d_fixed_shard_replay_receipt_hash is not None
        and registration.directions_frozen_before_residual_outcomes is True
        and registration.registered_before_residual_outcomes is True
        and registration.residual_population_post_selected is False
        and registration.proposal_rows_reused_as_residual is False
    )
    if legacy_resident == structured_affine:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.POPULATION_MISMATCH,
            "population registration must name exactly one D-fixed source family",
        )
    if structured_affine and registration.d_fixed_registration_hash == resident_hash:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.POPULATION_MISMATCH,
            "affine-shell population registration cannot alias the resident-D1 receipt",
        )
    return (
        _POPULATION_SOURCE_AFFINE_SHELL
        if structured_affine
        else _POPULATION_SOURCE_RESIDENT_D1
    )


def issue_v1_sampled_v7_d_only_measurement_owner(
    *,
    frame: V1AlexNetFQCausalGenerationFrameReceipt,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    residual_measure_design: V1SampledResidualPopulationMeasureDesign,
    resident_d1_population: V1SampledV7ResidentD1LiteralJPopulationResult,
    shell_union: V1SignedWorkFixedPrimalShellUnionReceipt,
    current_mask: HardMask,
    campaign_binding: V1OptionACampaignBinding,
    calibration: FixedAffineBCECalibration,
    scorer: OptimizedScoringResponseProvider,
    physical_route_axis: DeclaredRouteAxis,
    constructive_route_ids: ConstructiveOrdinalRouteAxis,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
) -> V1SampledV7DOnlyMeasurementOwner:
    """Issue residual stream/alpha/schedule custody before population registration."""

    snapshot, source_graph = _validate_common_sources(
        frame=frame,
        discovery_stream=discovery_stream,
        suffix_registration=suffix_registration,
        residual_measure_design=residual_measure_design,
        resident_d1_population=resident_d1_population,
        shell_union=shell_union,
        current_mask=current_mask,
        campaign_binding=campaign_binding,
        calibration=calibration,
        scorer=scorer,
        physical_route_axis=physical_route_axis,
        constructive_route_ids=constructive_route_ids,
        native_fiber_axis_receipt=native_fiber_axis_receipt,
    )
    capture_policy = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_only_capture_policy_v3",
            "source_graph_hash": source_graph,
            "scorer_source": snapshot,
            "physical_calls_equal_nonempty_registered_strata": True,
            "logical_look_is_statistical_unit": True,
            "transport_packets_are_execution_only": True,
            "target_before_complement": True,
            "same_capture_corrected_A_and_literal_J": True,
            "request_identity_is_not_a_sampling_coordinate": True,
            "branch_support_receipt_present": False,
        },
        prefix=_PREFIX + "capture-policy-",
    )
    def derived(role: str) -> str:
        return content_hash(
            {
                "schema": "partition_free_v1_sampled_v7_D_only_residual_stream_v1",
                "source_graph_hash": source_graph,
                "capture_policy_identity_hash": capture_policy,
                "role": role,
            },
            prefix=_PREFIX + role + "-",
        )

    witness = derived("residual-witness-stream")
    same_capture = derived("same-capture-vjp")
    corrected = derived("corrected-A-stream")
    literal_j = derived("residual-literal-J-stream")
    resident = resident_d1_population.receipt
    if len({witness, same_capture, corrected, literal_j}) != 4 or any(
        value in (
            resident.witness_stream_source_hash,
            resident.corrected_evidence_stream_source_hash,
            resident.literal_j_stream_source_hash,
        )
        for value in (witness, corrected, literal_j)
    ):
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
            "fresh residual and resident D1 stream identities collapsed",
        )
    contract_hash = campaign_binding.contract.campaign_contract_hash
    alpha_ledger = content_hash(
        {"campaign_contract_hash": contract_hash, "role": "D-only-alpha-ledger"},
        prefix=_PREFIX + "alpha-ledger-",
    )
    alpha_coordinate = content_hash(
        {
            "alpha_ledger_hash": alpha_ledger,
            "source_graph_hash": source_graph,
            "generation_index": frame.generation_index,
            "familywise_alpha": campaign_binding.contract.per_step_alpha,
        },
        prefix=_PREFIX + "alpha-coordinate-",
    )
    alpha_lineage = content_hash(
        {
            "alpha_ledger_hash": alpha_ledger,
            "alpha_coordinate_hash": alpha_coordinate,
            "campaign_binding_hash": campaign_binding.binding_hash,
        },
        prefix=_PREFIX + "alpha-lineage-",
    )
    proposal_history = content_hash(
        {
            "resident_d1_receipt_hash": resident.receipt_hash,
            "source_graph_hash": source_graph,
            "preferred_action_present": False,
        },
        prefix=_PREFIX + "proposal-history-",
    )
    residual_lineage = content_hash(
        {
            "source_graph_hash": source_graph,
            "suffix_registration_hash": suffix_registration.receipt_hash,
            "capture_policy_identity_hash": capture_policy,
        },
        prefix=_PREFIX + "residual-design-lineage-",
    )
    population_registration_id = content_hash(
        {
            "source_graph_hash": source_graph,
            "residual_design_lineage_hash": residual_lineage,
        },
        prefix=_PREFIX + "population-registration-",
    )
    schedule = _issue_v1_sampled_variable_shell_measurement_schedule(
        frame=frame,
        discovery_stream=discovery_stream,
        suffix_registration=suffix_registration,
        capture_policy_identity_hash=capture_policy,
    )
    scorer_source_hash = content_hash(
        snapshot, prefix=_PREFIX + "scorer-source-"
    )
    measurement_lineage = _issue_v1_sampled_variable_shell_measurement_source_lineage(
        source_graph_hash=source_graph,
        capture_policy_identity_hash=capture_policy,
        measurement_schedule_hash=schedule.schedule_hash,
        suffix_registration_hash=suffix_registration.receipt_hash,
        residual_population_measure_design_hash=residual_measure_design.design_hash,
        shell_union_receipt_hash=shell_union.receipt_hash,
        campaign_binding_hash=campaign_binding.binding_hash,
        calibration_hash=calibration.content_hash,
        scorer_source_hash=scorer_source_hash,
        model_identity_hash=resident.model_identity_hash,
        chart_identity_hash=resident.chart_identity_hash,
        current_mask_hash=current_mask.mask_hash,
        proposal_generation=frame.generation_index,
        route_axis_hash=physical_route_axis.route_axis_hash,
        route_id_axis_hash=physical_route_axis.route_id_axis_hash,
        native_fiber_axis_receipt_hash=native_fiber_axis_receipt.receipt_hash,
        native_fiber_measure_hash=native_fiber_axis_receipt.measure_hash,
        witness_stream_source_hash=witness,
        same_capture_vjp_source_hash=same_capture,
        corrected_evidence_stream_source_hash=corrected,
        literal_j_stream_source_hash=literal_j,
        alpha_ledger_id=alpha_ledger,
        alpha_coordinate_id=alpha_coordinate,
        alpha_lineage_hash=alpha_lineage,
    )
    return V1SampledV7DOnlyMeasurementOwner(
        source_graph_hash=source_graph,
        capture_policy_identity_hash=capture_policy,
        residual_witness_stream_source_hash=witness,
        same_capture_vjp_source_hash=same_capture,
        corrected_evidence_stream_source_hash=corrected,
        residual_literal_j_stream_source_hash=literal_j,
        proposal_influencing_history_hash=proposal_history,
        residual_design_lineage_hash=residual_lineage,
        alpha_ledger_hash=alpha_ledger,
        alpha_coordinate_hash=alpha_coordinate,
        alpha_lineage_hash=alpha_lineage,
        familywise_alpha=campaign_binding.contract.per_step_alpha,
        population_registration_id=population_registration_id,
        schedule=schedule,
        measurement_source_lineage=measurement_lineage,
        _frame=frame,
        _discovery_stream=discovery_stream,
        _suffix_registration=suffix_registration,
        _residual_measure_design=residual_measure_design,
        _resident_d1_population=resident_d1_population,
        _shell_union=shell_union,
        _current_mask=current_mask,
        _campaign_binding=campaign_binding,
        _calibration=calibration,
        _scorer=scorer,
        _physical_route_axis=physical_route_axis,
        _constructive_route_ids=constructive_route_ids,
        _native_fiber_axis_receipt=native_fiber_axis_receipt,
        _lease=_PacketIssuanceLease(),
        _issuance_token=_MEASUREMENT_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7DOnlyPopulationAttachment:
    measurement_owner_hash: str
    population_registration_hash: str
    population_source_family: str
    attachment_hash: str
    source_outcomes_seen: bool = False
    physical_acquisition_authority: bool = False
    alpha_spend_authority: bool = False
    trajectory_nonrecycling_alpha_closed: bool = False
    action_or_commit_authority: bool = False
    schema: str = SAMPLED_V7_D_ONLY_POPULATION_ATTACHMENT_SCHEMA
    _owner: V1SampledV7DOnlyMeasurementOwner = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _registration: V1LiteralJDFixedResidualPopulationProjectionRegistration = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _ATTACHMENT_TOKEN:
            raise PermissionError("V7 population attachments are issuer-only")
        if (
            type(self._owner) is not V1SampledV7DOnlyMeasurementOwner
            or type(self._registration)
            is not V1LiteralJDFixedResidualPopulationProjectionRegistration
            or self.source_outcomes_seen is not False
            or self.physical_acquisition_authority is not False
            or self.alpha_spend_authority is not False
            or self.trajectory_nonrecycling_alpha_closed is not False
            or self.action_or_commit_authority is not False
            or self.schema != SAMPLED_V7_D_ONLY_POPULATION_ATTACHMENT_SCHEMA
        ):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.AUTHORITY_VIOLATION,
                "population attachment crossed its pre-outcome boundary",
            )
        self.assert_live()

    def assert_live(self) -> str:
        owner_hash = self._owner.assert_live()
        registration = self._registration
        source_family = _classify_population_registration_source(
            owner=self._owner,
            registration=registration,
        )
        if (
            owner_hash != self.measurement_owner_hash
            or registration.registration_hash != self.population_registration_hash
            or source_family != self.population_source_family
        ):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.POPULATION_MISMATCH,
                "population registration did not repeat the source-issued streams, alpha, population, and source family",
            )
        expected = content_hash(
            {
                "schema": self.schema,
                "measurement_owner_hash": owner_hash,
                "population_registration_hash": registration.registration_hash,
                "population_source_family": source_family,
                "source_outcomes_seen": False,
                "physical_acquisition_authority": False,
                "alpha_spend_authority": False,
                "trajectory_nonrecycling_alpha_closed": False,
            },
            prefix=_PREFIX + "population-attachment-",
        )
        if expected != self.attachment_hash:
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
                "population attachment identity changed",
            )
        return expected


def attach_v1_sampled_v7_d_only_population_registration(
    *,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
) -> V1SampledV7DOnlyPopulationAttachment:
    if (
        type(measurement_owner) is not V1SampledV7DOnlyMeasurementOwner
        or type(population_registration)
        is not V1LiteralJDFixedResidualPopulationProjectionRegistration
    ):
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
            "population attachment requires exact source-issued owners",
        )
    owner_hash = measurement_owner.assert_live()
    source_family = _classify_population_registration_source(
        owner=measurement_owner,
        registration=population_registration,
    )
    attachment_hash = content_hash(
        {
            "schema": SAMPLED_V7_D_ONLY_POPULATION_ATTACHMENT_SCHEMA,
            "measurement_owner_hash": owner_hash,
            "population_registration_hash": population_registration.registration_hash,
            "population_source_family": source_family,
            "source_outcomes_seen": False,
            "physical_acquisition_authority": False,
            "alpha_spend_authority": False,
            "trajectory_nonrecycling_alpha_closed": False,
        },
        prefix=_PREFIX + "population-attachment-",
    )
    return V1SampledV7DOnlyPopulationAttachment(
        measurement_owner_hash=owner_hash,
        population_registration_hash=population_registration.registration_hash,
        population_source_family=source_family,
        attachment_hash=attachment_hash,
        _owner=measurement_owner,
        _registration=population_registration,
        _issuance_token=_ATTACHMENT_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7DOnlyBranchSourceBinding:
    """Provisional support-source custody with no physical-action identity.

    Two source handles necessarily receive distinct support aliases, but this
    type cannot distinguish two physical actions that reuse one handle.  The
    alias therefore is not a production branch identity and cannot authorize
    packet append registration or physical shard fan-out.
    """

    measurement_owner_hash: str
    population_attachment_hash: str
    branch_source_receipt_hash: str
    support_source_alias_hash: str
    append_lineage: V1SignedWorkVariableShellSourceAppendLineage
    append_owner: V1SignedWorkVariableShellSourceAppendOwner = field(
        repr=False, compare=False
    )
    shared_measurement_schedule_hash: str = ""
    shared_measurement_source_lineage_hash: str = ""
    physical_acquisition_authority: bool = False
    raw_shard_fanout_execution_authority: bool = False
    packet_append_registration_authority: bool = False
    alpha_spend_authority: bool = False
    trajectory_nonrecycling_alpha_closed: bool = False
    physical_action_axis_bound: bool = False
    production_branch_identity_authority: bool = False
    preferred_action_chosen: bool = False
    action_or_commit_authority: bool = False
    schema: str = SAMPLED_V7_D_ONLY_BRANCH_SOURCE_BINDING_SCHEMA
    _measurement_owner: V1SampledV7DOnlyMeasurementOwner = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _population_attachment: V1SampledV7DOnlyPopulationAttachment = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _support_source: V1SignedWorkVariableShellSourceDeviceHandle = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _design: V1SignedWorkVariableShellSourceCUDAExecutionDesign = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuance_token: InitVar[object | None] = None
    _binding_hash: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _BRANCH_TOKEN:
            raise PermissionError("V7 branch-source bindings are issuer-only")
        exact = (
            (self.append_lineage, V1SignedWorkVariableShellSourceAppendLineage),
            (self.append_owner, V1SignedWorkVariableShellSourceAppendOwner),
            (self._measurement_owner, V1SampledV7DOnlyMeasurementOwner),
            (self._population_attachment, V1SampledV7DOnlyPopulationAttachment),
            (self._support_source, V1SignedWorkVariableShellSourceDeviceHandle),
            (self._design, V1SignedWorkVariableShellSourceCUDAExecutionDesign),
        )
        if any(type(value) is not expected for value, expected in exact):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
                "branch binding lost exact source custody",
            )
        if (
            self.physical_acquisition_authority is not False
            or self.raw_shard_fanout_execution_authority is not False
            or self.packet_append_registration_authority is not False
            or self.alpha_spend_authority is not False
            or self.trajectory_nonrecycling_alpha_closed is not False
            or self.physical_action_axis_bound is not False
            or self.production_branch_identity_authority is not False
            or self.preferred_action_chosen is not False
            or self.action_or_commit_authority is not False
            or self.schema != SAMPLED_V7_D_ONLY_BRANCH_SOURCE_BINDING_SCHEMA
        ):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.AUTHORITY_VIOLATION,
                "branch binding crossed the no-execution fan-out seam",
            )
        self.assert_live()
        object.__setattr__(self, "_binding_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "measurement_owner_hash": self.measurement_owner_hash,
                "population_attachment_hash": self.population_attachment_hash,
                "branch_source_receipt_hash": self.branch_source_receipt_hash,
                "support_source_alias_hash": self.support_source_alias_hash,
                "append_lineage_hash": self.append_lineage.lineage_hash,
                "shared_measurement_schedule_hash": self.shared_measurement_schedule_hash,
                "shared_measurement_source_lineage_hash": (
                    self.shared_measurement_source_lineage_hash
                ),
                "physical_acquisition_authority": False,
                "raw_shard_fanout_execution_authority": False,
                "packet_append_registration_authority": False,
                "alpha_spend_authority": False,
                "trajectory_nonrecycling_alpha_closed": False,
                "physical_action_axis_bound": False,
                "production_branch_identity_authority": False,
                "preferred_action_chosen": False,
            },
            prefix=_PREFIX + "branch-source-binding-",
        )

    def assert_live(self) -> str:
        owner_hash = self._measurement_owner.assert_live()
        attachment_hash = self._population_attachment.assert_live()
        support = self._support_source.receipt
        lineage = self.append_lineage
        if (
            owner_hash != self.measurement_owner_hash
            or attachment_hash != self.population_attachment_hash
            or self._population_attachment._owner is not self._measurement_owner
            or support.receipt_hash != self.branch_source_receipt_hash
            or support.design_hash != self._design.design_hash
            or self.append_owner.lineage is not lineage
            or self.shared_measurement_schedule_hash
            != self._measurement_owner.schedule.schedule_hash
            or self.shared_measurement_source_lineage_hash
            != self._measurement_owner.measurement_source_lineage.lineage_hash
            or lineage.predictable_packet_order_source_hash
            != self._measurement_owner.schedule.schedule_hash
            or lineage.population_context_axis_hash
            != self._measurement_owner.schedule.population_context_axis_hash
            or lineage.population_draw_axis_hash
            != self._measurement_owner.schedule.population_draw_axis_hash
            or lineage.resource_axis_hash != support.resource_axis_hash
            or lineage.resource_source_receipt_hash
            != support.resource_source_receipt_hash
        ):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
                "branch source, append owner, or shared measurement axes changed",
            )
        observed = self._compute_hash()
        if self._binding_hash is not None and observed != self._binding_hash:
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
                "branch-source binding identity changed",
            )
        return observed

    @property
    def binding_hash(self) -> str:
        return self.assert_live()

    def require_physical_fanout_execution(self) -> NoReturn:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.FANOUT_ABI_MISSING,
            "raw A/J shards are not yet separable from branch append/source registration; execution is fail-closed",
        )

    def require_packet_append_registration(self) -> NoReturn:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.FANOUT_ABI_MISSING,
            "branch-local support custody has no physical-action axis and cannot authorize packet append registration",
        )


def bind_v1_sampled_v7_d_only_branch_source(
    *,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    population_attachment: V1SampledV7DOnlyPopulationAttachment,
    support_bound_source: V1SignedWorkVariableShellSourceDeviceHandle,
    variable_shell_design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
) -> V1SampledV7DOnlyBranchSourceBinding:
    if (
        type(measurement_owner) is not V1SampledV7DOnlyMeasurementOwner
        or type(population_attachment) is not V1SampledV7DOnlyPopulationAttachment
        or type(support_bound_source) is not V1SignedWorkVariableShellSourceDeviceHandle
        or type(variable_shell_design)
        is not V1SignedWorkVariableShellSourceCUDAExecutionDesign
    ):
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
            "branch binding requires exact live population/support/design owners",
        )
    measurement_owner.assert_live()
    population_attachment.assert_live()
    if population_attachment._owner is not measurement_owner:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.POPULATION_MISMATCH,
            "population attachment belongs to another measurement owner",
        )
    support = support_bound_source.receipt
    source = measurement_owner.measurement_source_lineage
    shell = measurement_owner._shell_union
    native = measurement_owner._native_fiber_axis_receipt
    shell_fibers = tuple(
        native.route_to_fiber(index).global_fiber_index for index in shell.shell_indices
    )
    if (
        support.design_hash != variable_shell_design.design_hash
        or support.model_identity_hash != source.model_identity_hash
        or support.chart_identity_hash != source.chart_identity_hash
        or support.current_mask_hash != source.current_mask_hash
        or support.proposal_generation != source.proposal_generation
        or support.route_axis_hash != source.route_axis_hash
        or support.route_id_axis_hash != source.route_id_axis_hash
        or support.registered_d_w2_prefix_hash
        != measurement_owner._suffix_registration.d1_pair_prefix_axis_hash
        or support.shell_freeze_receipt_hash != shell.receipt_hash
        or support.shell_indices != shell.shell_indices
        or support.shell_route_ids != shell.shell_route_ids
        or support.shell_native_fiber_indices != shell_fibers
        or support.native_fiber_axis_receipt_hash != native.receipt_hash
        or support.native_fiber_measure_hash != native.measure_hash
        or support.route_count != native.route_count
        or support.fiber_count != native.fiber_count
        or support.cut_count != native.cut_count
    ):
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.SUPPORT_MISMATCH,
            "branch-local D1 support does not share the measurement chart/shell/native geometry",
        )
    lineage = V1SignedWorkVariableShellSourceAppendLineage(
        model_identity_hash=source.model_identity_hash,
        chart_identity_hash=source.chart_identity_hash,
        current_mask_hash=source.current_mask_hash,
        proposal_generation=source.proposal_generation,
        route_axis_hash=source.route_axis_hash,
        route_id_axis_hash=source.route_id_axis_hash,
        registered_d_prefix_hash=(
            measurement_owner._suffix_registration.d1_pair_prefix_axis_hash
        ),
        shell_freeze_receipt_hash=shell.receipt_hash,
        shell_indices=shell.shell_indices,
        shell_route_ids=shell.shell_route_ids,
        shell_native_fiber_indices=shell_fibers,
        native_fiber_axis_receipt_hash=native.receipt_hash,
        native_fiber_measure_hash=native.measure_hash,
        kkt_branch_id=SAMPLED_VARIABLE_SHELL_TWO_BRANCH_KKT_ID,
        kkt_branch_source_receipt_hash=support.receipt_hash,
        resource_axis_hash=support.resource_axis_hash,
        resource_source_receipt_hash=support.resource_source_receipt_hash,
        witness_stream_source_hash=source.witness_stream_source_hash,
        same_capture_vjp_source_hash=source.same_capture_vjp_source_hash,
        corrected_evidence_stream_source_hash=(
            source.corrected_evidence_stream_source_hash
        ),
        literal_j_stream_source_hash=source.literal_j_stream_source_hash,
        alpha_ledger_id=source.alpha_ledger_id,
        alpha_coordinate_id=source.alpha_coordinate_id,
        alpha_lineage_hash=source.alpha_lineage_hash,
        design_hash=variable_shell_design.design_hash,
        population_count=measurement_owner.schedule.population_count,
        population_context_axis_hash=(
            measurement_owner.schedule.population_context_axis_hash
        ),
        population_draw_axis_hash=measurement_owner.schedule.population_draw_axis_hash,
        route_count=native.route_count,
        fiber_count=native.fiber_count,
        cut_count=native.cut_count,
        predictable_packet_order_source_hash=measurement_owner.schedule.schedule_hash,
        stratum_measure_source_hash=(
            v1_sampled_variable_shell_stratum_measure_source_hash(
                measurement_owner._suffix_registration
            )
        ),
        common_integer_measure_unit_hash=(
            v1_sampled_variable_shell_common_measure_unit_hash(
                registration=measurement_owner._suffix_registration
            )
        ),
    )
    owner = V1SignedWorkVariableShellSourceAppendOwner(lineage=lineage)
    support_source_alias = content_hash(
        {
            "measurement_owner_hash": measurement_owner.owner_hash,
            "support_source_receipt_hash": support.receipt_hash,
            "resource_axis_hash": support.resource_axis_hash,
            "resource_source_receipt_hash": support.resource_source_receipt_hash,
        },
        prefix=_PREFIX + "support-source-alias-",
    )
    return V1SampledV7DOnlyBranchSourceBinding(
        measurement_owner_hash=measurement_owner.owner_hash,
        population_attachment_hash=population_attachment.attachment_hash,
        branch_source_receipt_hash=support.receipt_hash,
        support_source_alias_hash=support_source_alias,
        append_lineage=lineage,
        append_owner=owner,
        shared_measurement_schedule_hash=measurement_owner.schedule.schedule_hash,
        shared_measurement_source_lineage_hash=(
            measurement_owner.measurement_source_lineage.lineage_hash
        ),
        _measurement_owner=measurement_owner,
        _population_attachment=population_attachment,
        _support_source=support_bound_source,
        _design=variable_shell_design,
        _issuance_token=_BRANCH_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7DOnlyPacketMeasurementIssuance:
    measurement_owner_hash: str
    packet_hash: str
    measurement_schedule_hash: str
    measurement_source_lineage_hash: str
    logical_measurement_request: MeasurementRequest
    source_outcomes_seen: bool = False
    branch_support_receipt_present: bool = False
    physical_acquisition_authority: bool = False
    alpha_spend_authority: bool = False
    trajectory_nonrecycling_alpha_closed: bool = False
    action_or_commit_authority: bool = False
    schema: str = SAMPLED_V7_D_ONLY_PACKET_ISSUANCE_SCHEMA
    _owner: V1SampledV7DOnlyMeasurementOwner = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _packet: V1SampledVariableShellDW2SuffixPacket = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuance_token: InitVar[object | None] = None
    _issuance_hash: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PACKET_TOKEN:
            raise PermissionError("V7 packet measurement issuances are issuer-only")
        if (
            type(self._owner) is not V1SampledV7DOnlyMeasurementOwner
            or type(self._packet) is not V1SampledVariableShellDW2SuffixPacket
            or type(self.logical_measurement_request) is not MeasurementRequest
            or self.source_outcomes_seen is not False
            or self.branch_support_receipt_present is not False
            or self.physical_acquisition_authority is not False
            or self.alpha_spend_authority is not False
            or self.trajectory_nonrecycling_alpha_closed is not False
            or self.action_or_commit_authority is not False
            or self.schema != SAMPLED_V7_D_ONLY_PACKET_ISSUANCE_SCHEMA
        ):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.AUTHORITY_VIOLATION,
                "packet issuance crossed its shared pre-outcome boundary",
            )
        self.assert_live()
        object.__setattr__(self, "_issuance_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "measurement_owner_hash": self.measurement_owner_hash,
                "packet_hash": self.packet_hash,
                "measurement_schedule_hash": self.measurement_schedule_hash,
                "measurement_source_lineage_hash": self.measurement_source_lineage_hash,
                "logical_measurement_request_id": self.logical_measurement_request.request_id,
                "branch_support_receipt_present": False,
                "physical_acquisition_authority": False,
                "alpha_spend_authority": False,
                "trajectory_nonrecycling_alpha_closed": False,
            },
            prefix=_PREFIX + "packet-measurement-issuance-",
        )

    def assert_live(self) -> str:
        owner_hash = self._owner.assert_live()
        scheduled = self._owner.schedule.packet_for(self._packet)
        request = self.logical_measurement_request
        expected_payload = v1_sampled_variable_shell_packet_request_payload(
            registration=self._owner._suffix_registration,
            packet=self._packet,
            residual_measure_design=self._owner._residual_measure_design,
            shell_union=self._owner._shell_union,
            measurement_schedule=self._owner.schedule,
            measurement_source_lineage=self._owner.measurement_source_lineage,
        )
        expected_state = v1_sampled_variable_shell_packet_request_state_hash(
            registration=self._owner._suffix_registration,
            packet=self._packet,
            residual_measure_design=self._owner._residual_measure_design,
            shell_union=self._owner._shell_union,
            measurement_schedule=self._owner.schedule,
            measurement_source_lineage=self._owner.measurement_source_lineage,
        )
        observed_request_id = content_hash(
            {
                "kind": request.kind,
                "state_hash": request.state_hash,
                "version": request.version,
                "context_selector": request.context_selector,
                "candidate_ids": tuple(request.candidate_ids),
                "expected_cost": request.expected_cost,
                "reuse_keys": tuple(request.reuse_keys),
                "provenance": request.provenance,
                "payload": dict(request.payload),
            },
            prefix="irg-request-",
        )
        if (
            owner_hash != self.measurement_owner_hash
            or self.packet_hash != self._packet.receipt_hash
            or self.measurement_schedule_hash != self._owner.schedule.schedule_hash
            or self.measurement_source_lineage_hash
            != self._owner.measurement_source_lineage.lineage_hash
            or request.request_id != observed_request_id
            or request.state_hash != expected_state
            or dict(request.payload) != expected_payload
            or request.context_selector.context_ids
            != scheduled.registered_context_ids
        ):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
                "shared packet request identity changed after issuance",
            )
        observed = self._compute_hash()
        if self._issuance_hash is not None and observed != self._issuance_hash:
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
                "packet measurement issuance hash changed",
            )
        return observed

    @property
    def issuance_hash(self) -> str:
        return self.assert_live()


def _build_v1_sampled_v7_d_only_packet_measurement_request(
    *,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    packet: V1SampledVariableShellDW2SuffixPacket,
) -> MeasurementRequest:
    scheduled = measurement_owner.schedule.packet_for(packet)
    payload = v1_sampled_variable_shell_packet_request_payload(
        registration=measurement_owner._suffix_registration,
        packet=packet,
        residual_measure_design=measurement_owner._residual_measure_design,
        shell_union=measurement_owner._shell_union,
        measurement_schedule=measurement_owner.schedule,
        measurement_source_lineage=measurement_owner.measurement_source_lineage,
    )
    state_hash = v1_sampled_variable_shell_packet_request_state_hash(
        registration=measurement_owner._suffix_registration,
        packet=packet,
        residual_measure_design=measurement_owner._residual_measure_design,
        shell_union=measurement_owner._shell_union,
        measurement_schedule=measurement_owner.schedule,
        measurement_source_lineage=measurement_owner.measurement_source_lineage,
    )
    return MeasurementRequest.create(
        kind="response_jacobian",
        state_hash=state_hash,
        version=SAMPLED_VARIABLE_SHELL_PACKET_REQUEST_VERSION,
        context_selector=ContextSelector(
            selector_id=content_hash(
                {
                    "schema": "sampled-v7-D-only-shared-logical-selector-v1",
                    "schedule_packet_hash": scheduled.packet_schedule_hash,
                },
                prefix=_PREFIX + "selector-",
            ),
            requested_count=len(scheduled.registered_context_ids),
            strata=("target", "complement"),
            context_ids=scheduled.registered_context_ids,
        ),
        candidate_ids=measurement_owner._shell_union.shell_route_ids,
        provenance=MeasurementProvenance(
            producer_id="modern-v1-sampled-v7-d-only-shared-measurement-issuance",
            producer_version=SAMPLED_V7_D_ONLY_PACKET_ISSUANCE_SCHEMA,
            code_hash=content_hash(
                {
                    "measurement_owner_hash": measurement_owner.owner_hash,
                    "schedule_packet_hash": scheduled.packet_schedule_hash,
                    "role": "logical-request-provenance-only",
                },
                prefix=_PREFIX + "request-producer-",
            ),
        ),
        payload=payload,
    )


def _issue_v1_sampled_v7_d_only_packet_measurement_after_reserve(
    *,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    packet: V1SampledVariableShellDW2SuffixPacket,
    request: MeasurementRequest,
) -> V1SampledV7DOnlyPacketMeasurementIssuance:
    return V1SampledV7DOnlyPacketMeasurementIssuance(
        measurement_owner_hash=measurement_owner.owner_hash,
        packet_hash=packet.receipt_hash,
        measurement_schedule_hash=measurement_owner.schedule.schedule_hash,
        measurement_source_lineage_hash=(
            measurement_owner.measurement_source_lineage.lineage_hash
        ),
        logical_measurement_request=request,
        _owner=measurement_owner,
        _packet=packet,
        _issuance_token=_PACKET_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7DOnlyLogicalLookMeasurementIssuance:
    measurement_owner_hash: str
    measurement_schedule_hash: str
    measurement_source_lineage_hash: str
    logical_registered_look_index: int
    logical_registered_look_stratum_counts: tuple[int, int]
    logical_residual_before_stratum_counts: tuple[int, int]
    logical_residual_after_stratum_counts: tuple[int, int]
    transport_packet_hashes: tuple[str, ...]
    transport_schedule_packet_hashes: tuple[str, ...]
    transport_request_ids: tuple[str, ...]
    transport_packet_issuances: tuple[V1SampledV7DOnlyPacketMeasurementIssuance, ...]
    source_outcomes_seen: bool = False
    branch_support_receipt_present: bool = False
    physical_acquisition_authority: bool = False
    alpha_spend_authority: bool = False
    trajectory_nonrecycling_alpha_closed: bool = False
    action_or_commit_authority: bool = False
    logical_registered_look_is_statistical_unit: bool = True
    transport_packets_are_hardware_only: bool = True
    schema: str = SAMPLED_V7_D_ONLY_LOGICAL_LOOK_ISSUANCE_SCHEMA
    _owner: V1SampledV7DOnlyMeasurementOwner = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuance_token: InitVar[object | None] = None
    _issuance_hash: str | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PACKET_TOKEN:
            raise PermissionError("V7 logical-look issuances are issuer-only")
        if type(self._owner) is not V1SampledV7DOnlyMeasurementOwner:
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
                "logical-look issuance requires one exact measurement owner",
            )
        closed = (
            self.source_outcomes_seen,
            self.branch_support_receipt_present,
            self.physical_acquisition_authority,
            self.alpha_spend_authority,
            self.trajectory_nonrecycling_alpha_closed,
            self.action_or_commit_authority,
        )
        if (
            any(type(value) is not bool or value is not False for value in closed)
            or self.logical_registered_look_is_statistical_unit is not True
            or self.transport_packets_are_hardware_only is not True
            or self.schema != SAMPLED_V7_D_ONLY_LOGICAL_LOOK_ISSUANCE_SCHEMA
        ):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.AUTHORITY_VIOLATION,
                "logical-look issuance crossed its shared pre-outcome boundary",
            )
        self.assert_live()
        object.__setattr__(self, "_issuance_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "measurement_owner_hash": self.measurement_owner_hash,
                "measurement_schedule_hash": self.measurement_schedule_hash,
                "measurement_source_lineage_hash": (
                    self.measurement_source_lineage_hash
                ),
                "logical_registered_look_index": self.logical_registered_look_index,
                "logical_registered_look_stratum_counts": (
                    self.logical_registered_look_stratum_counts
                ),
                "logical_residual_before_stratum_counts": (
                    self.logical_residual_before_stratum_counts
                ),
                "logical_residual_after_stratum_counts": (
                    self.logical_residual_after_stratum_counts
                ),
                "transport_packet_hashes": self.transport_packet_hashes,
                "transport_schedule_packet_hashes": (
                    self.transport_schedule_packet_hashes
                ),
                "transport_request_ids": self.transport_request_ids,
                "logical_registered_look_is_statistical_unit": True,
                "transport_packets_are_hardware_only": True,
                "downstream_authority": False,
            },
            prefix=_PREFIX + "logical-look-measurement-issuance-",
        )

    def assert_live(self) -> str:
        owner_hash = self._owner.assert_live()
        look = _positive_int(
            self.logical_registered_look_index,
            name="logical registered look index",
        )
        counts = _count_pair(
            self.logical_registered_look_stratum_counts,
            name="logical registered look stratum counts",
        )
        before = _count_pair(
            self.logical_residual_before_stratum_counts,
            name="logical residual before stratum counts",
        )
        after = _count_pair(
            self.logical_residual_after_stratum_counts,
            name="logical residual after stratum counts",
        )
        d1 = self._owner._residual_measure_design.d1_stratum_counts
        expected_after = tuple(counts[index] - d1[index] for index in (0, 1))
        if any(value < 0 for value in expected_after):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.AXIS_MISMATCH,
                "logical registered look precedes the resident D1 prefix",
            )
        issuances = tuple(self.transport_packet_issuances)
        if (
            not issuances
            or any(
                type(value) is not V1SampledV7DOnlyPacketMeasurementIssuance
                for value in issuances
            )
        ):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
                "logical look requires one or more exact transport issuances",
            )
        packet_hashes: list[str] = []
        schedule_hashes: list[str] = []
        request_ids: list[str] = []
        packets: list[V1SampledVariableShellDW2SuffixPacket] = []
        for issuance in issuances:
            if issuance._owner is not self._owner:
                _refuse(
                    V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
                    "logical look mixed transport issuances from another owner",
                )
            issuance.assert_live()
            packet = issuance._packet
            scheduled = self._owner.schedule.packet_for(packet)
            packet_hashes.append(packet.receipt_hash)
            schedule_hashes.append(scheduled.packet_schedule_hash)
            request_ids.append(issuance.logical_measurement_request.request_id)
            packets.append(packet)
        if tuple(packet.packet_ordinal for packet in packets) != tuple(
            range(packets[0].packet_ordinal, packets[0].packet_ordinal + len(packets))
        ):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.ORDER_MISMATCH,
                "logical look transport packets are not one contiguous suffix axis",
            )
        for left, right in zip(packets, packets[1:], strict=False):
            if left.after_suffix_stratum_counts != right.before_suffix_stratum_counts:
                _refuse(
                    V1SampledV7DOnlySourceIssuanceRefusalCode.ORDER_MISMATCH,
                    "logical look transport packets leave a gap or overlap",
                )
        if (
            owner_hash != self.measurement_owner_hash
            or self.measurement_schedule_hash != self._owner.schedule.schedule_hash
            or self.measurement_source_lineage_hash
            != self._owner.measurement_source_lineage.lineage_hash
            or before != packets[0].before_suffix_stratum_counts
            or after != packets[-1].after_suffix_stratum_counts
            or after != expected_after
            or any(before[index] > after[index] for index in (0, 1))
            or tuple(packet_hashes) != self.transport_packet_hashes
            or tuple(schedule_hashes) != self.transport_schedule_packet_hashes
            or tuple(request_ids) != self.transport_request_ids
            or not look
        ):
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
                "logical-look issuance no longer matches its owner, endpoint, or packets",
            )
        observed = self._compute_hash()
        if self._issuance_hash is not None and observed != self._issuance_hash:
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
                "logical-look measurement issuance hash changed",
            )
        return observed

    @property
    def issuance_hash(self) -> str:
        return self.assert_live()


def v1_sampled_v7_d_only_transport_packet_issuances(
    issuance: object,
) -> tuple[V1SampledV7DOnlyPacketMeasurementIssuance, ...]:
    if type(issuance) is V1SampledV7DOnlyPacketMeasurementIssuance:
        issuance.assert_live()
        return (issuance,)
    if type(issuance) is V1SampledV7DOnlyLogicalLookMeasurementIssuance:
        issuance.assert_live()
        return issuance.transport_packet_issuances
    _refuse(
        V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
        "expected a single transport issuance or logical-look issuance",
    )


def v1_sampled_v7_d_only_registered_physical_call_count(
    issuance: object,
) -> int:
    """Return the pre-outcome scorer-call count implied by one D look.

    Transport packet size is an execution detail.  The statistical object is
    one registered logical look, while each nonempty target/complement stratum
    in its contiguous transport cover requires exactly one scorer call.  This
    count is therefore derived from the frozen source axis rather than supplied
    as a launcher constant.
    """

    transports = v1_sampled_v7_d_only_transport_packet_issuances(issuance)
    count = sum(
        int(bool(packet_issuance._packet.target_context_ids))
        + int(bool(packet_issuance._packet.complement_context_ids))
        for packet_issuance in transports
    )
    if count <= 0:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.AXIS_MISMATCH,
            "a registered D logical look has no nonempty physical stratum",
        )
    return count


def v1_sampled_v7_d_only_logical_registered_look(
    issuance: object,
    *,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
) -> tuple[int, tuple[int, int], tuple[int, int], tuple[int, int]]:
    if type(issuance) is V1SampledV7DOnlyLogicalLookMeasurementIssuance:
        if issuance._owner is not measurement_owner:
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
                "logical-look issuance belongs to another measurement owner",
            )
        issuance.assert_live()
        return (
            issuance.logical_registered_look_index,
            issuance.logical_registered_look_stratum_counts,
            issuance.logical_residual_before_stratum_counts,
            issuance.logical_residual_after_stratum_counts,
        )
    if type(issuance) is V1SampledV7DOnlyPacketMeasurementIssuance:
        if issuance._owner is not measurement_owner:
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.STALE_SOURCE,
                "transport packet issuance belongs to another measurement owner",
            )
        issuance.assert_live()
        packet = issuance._packet
        d1 = measurement_owner._residual_measure_design.d1_stratum_counts
        return (
            packet.packet_ordinal + 1,
            tuple(
                d1[index] + packet.after_suffix_stratum_counts[index]
                for index in (0, 1)
            ),  # type: ignore[return-value]
            packet.before_suffix_stratum_counts,
            packet.after_suffix_stratum_counts,
        )
    _refuse(
        V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
        "expected a single transport issuance or logical-look issuance",
    )


def issue_v1_sampled_v7_d_only_frozen_packet_measurement(
    *,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    packet: V1SampledVariableShellDW2SuffixPacket,
) -> V1SampledV7DOnlyPacketMeasurementIssuance:
    if (
        type(measurement_owner) is not V1SampledV7DOnlyMeasurementOwner
        or type(packet) is not V1SampledVariableShellDW2SuffixPacket
    ):
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
            "packet issuance requires an exact measurement owner and suffix packet",
        )
    measurement_owner.assert_live()
    request = _build_v1_sampled_v7_d_only_packet_measurement_request(
        measurement_owner=measurement_owner,
        packet=packet,
    )
    measurement_owner._lease.reserve(packet.receipt_hash)
    return _issue_v1_sampled_v7_d_only_packet_measurement_after_reserve(
        measurement_owner=measurement_owner,
        packet=packet,
        request=request,
    )


def issue_v1_sampled_v7_d_only_frozen_logical_look_measurement(
    *,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    packets: Sequence[V1SampledVariableShellDW2SuffixPacket],
    registered_look_index: int,
    registered_look_stratum_counts: tuple[int, int],
) -> V1SampledV7DOnlyLogicalLookMeasurementIssuance:
    if type(measurement_owner) is not V1SampledV7DOnlyMeasurementOwner:
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
            "logical-look issuance requires an exact measurement owner",
        )
    measurement_owner.assert_live()
    look = _positive_int(registered_look_index, name="registered look index")
    counts = _count_pair(
        registered_look_stratum_counts,
        name="registered look stratum counts",
    )
    packet_axis = tuple(packets)
    if (
        not packet_axis
        or any(
            type(packet) is not V1SampledVariableShellDW2SuffixPacket
            for packet in packet_axis
        )
    ):
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.INVALID_TYPE,
            "logical-look issuance requires one or more exact suffix packets",
        )
    for packet in packet_axis:
        measurement_owner.schedule.packet_for(packet)
    if tuple(packet.packet_ordinal for packet in packet_axis) != tuple(
        range(packet_axis[0].packet_ordinal, packet_axis[0].packet_ordinal + len(packet_axis))
    ):
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.ORDER_MISMATCH,
            "logical-look packets are not a contiguous transport axis",
        )
    for left, right in zip(packet_axis, packet_axis[1:], strict=False):
        if left.after_suffix_stratum_counts != right.before_suffix_stratum_counts:
            _refuse(
                V1SampledV7DOnlySourceIssuanceRefusalCode.ORDER_MISMATCH,
                "logical-look packets leave a gap or overlap",
            )
    d1 = measurement_owner._residual_measure_design.d1_stratum_counts
    residual_after = tuple(counts[index] - d1[index] for index in (0, 1))
    if (
        any(value < 0 for value in residual_after)
        or packet_axis[-1].after_suffix_stratum_counts != residual_after
        or any(
            packet_axis[0].before_suffix_stratum_counts[index] > residual_after[index]
            for index in (0, 1)
        )
    ):
        _refuse(
            V1SampledV7DOnlySourceIssuanceRefusalCode.AXIS_MISMATCH,
            "logical-look endpoint does not equal the transport packet prefix",
        )
    requests = tuple(
        _build_v1_sampled_v7_d_only_packet_measurement_request(
            measurement_owner=measurement_owner,
            packet=packet,
        )
        for packet in packet_axis
    )
    measurement_owner._lease.reserve_many(
        tuple(packet.receipt_hash for packet in packet_axis)
    )
    packet_issuances = tuple(
        _issue_v1_sampled_v7_d_only_packet_measurement_after_reserve(
            measurement_owner=measurement_owner,
            packet=packet,
            request=request,
        )
        for packet, request in zip(packet_axis, requests, strict=True)
    )
    scheduled = tuple(
        measurement_owner.schedule.packet_for(packet) for packet in packet_axis
    )
    return V1SampledV7DOnlyLogicalLookMeasurementIssuance(
        measurement_owner_hash=measurement_owner.owner_hash,
        measurement_schedule_hash=measurement_owner.schedule.schedule_hash,
        measurement_source_lineage_hash=(
            measurement_owner.measurement_source_lineage.lineage_hash
        ),
        logical_registered_look_index=look,
        logical_registered_look_stratum_counts=counts,
        logical_residual_before_stratum_counts=(
            packet_axis[0].before_suffix_stratum_counts
        ),
        logical_residual_after_stratum_counts=(
            packet_axis[-1].after_suffix_stratum_counts
        ),
        transport_packet_hashes=tuple(packet.receipt_hash for packet in packet_axis),
        transport_schedule_packet_hashes=tuple(
            packet.packet_schedule_hash for packet in scheduled
        ),
        transport_request_ids=tuple(request.request_id for request in requests),
        transport_packet_issuances=packet_issuances,
        _owner=measurement_owner,
        _issuance_token=_PACKET_TOKEN,
    )


def issue_v1_sampled_v7_d_only_source_owner_bundle(**_: object) -> NoReturn:
    """Fail closed for the retired branch-conflated construction."""

    _refuse(
        V1SampledV7DOnlySourceIssuanceRefusalCode.AUTHORITY_VIOLATION,
        "the conflated source-owner bundle is retired; issue shared measurement ownership then bind each branch support",
    )


def issue_v1_sampled_v7_d_only_frozen_packet_source(**_: object) -> NoReturn:
    """Fail closed for the retired branch-dependent logical request."""

    _refuse(
        V1SampledV7DOnlySourceIssuanceRefusalCode.AUTHORITY_VIOLATION,
        "branch-local packet issuance is forbidden; issue one shared packet measurement",
    )


__all__ = [
    "SAMPLED_V7_D_ONLY_BRANCH_SOURCE_BINDING_SCHEMA",
    "SAMPLED_V7_D_ONLY_LOGICAL_LOOK_ISSUANCE_SCHEMA",
    "SAMPLED_V7_D_ONLY_MEASUREMENT_OWNER_SCHEMA",
    "SAMPLED_V7_D_ONLY_PACKET_ISSUANCE_SCHEMA",
    "SAMPLED_V7_D_ONLY_POPULATION_ATTACHMENT_SCHEMA",
    "SAMPLED_V7_D_ONLY_SOURCE_AUTHORITY",
    "V1SampledV7DOnlyBranchSourceBinding",
    "V1SampledV7DOnlyLogicalLookMeasurementIssuance",
    "V1SampledV7DOnlyMeasurementOwner",
    "V1SampledV7DOnlyPacketMeasurementIssuance",
    "V1SampledV7DOnlyPopulationAttachment",
    "V1SampledV7DOnlySourceIssuanceRefusal",
    "V1SampledV7DOnlySourceIssuanceRefusalCode",
    "attach_v1_sampled_v7_d_only_population_registration",
    "bind_v1_sampled_v7_d_only_branch_source",
    "issue_v1_sampled_v7_d_only_frozen_logical_look_measurement",
    "issue_v1_sampled_v7_d_only_frozen_packet_measurement",
    "issue_v1_sampled_v7_d_only_frozen_packet_source",
    "issue_v1_sampled_v7_d_only_measurement_owner",
    "issue_v1_sampled_v7_d_only_source_owner_bundle",
    "v1_sampled_v7_d_only_logical_registered_look",
    "v1_sampled_v7_d_only_registered_physical_call_count",
    "v1_sampled_v7_d_only_transport_packet_issuances",
]
