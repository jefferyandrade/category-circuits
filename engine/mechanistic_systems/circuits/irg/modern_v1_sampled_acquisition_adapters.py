"""Verified sampled-summary sources for the modern-v1 numeric owner.

The sampled proposal path intentionally retains central moments and bounded
Frequent-Directions continuation state instead of the complete context-by-
route row stream.  Those summaries are sufficient for the registered score
order and selected-shell response calculation, but they are not the source
objects required by :class:`V1FDStreamAcquisition` or by an exact tangent-
price acquisition.

This module closes the truthful boundary.  It:

* verifies a complete sampled registered-look commit (live or externally
  restored) against the current campaign, mask, two route axes, calibration,
  raw-logit coordinate, and response-exactification replay;
* keeps the sampled point estimate distinct from the regret-certified order;
* restores and assembles the real landed stratified FD snapshot without
  reconstructing, retaining, or claiming the raw row stream; and
* issues replayable proposal-only receipts for a later explicit
  sampled-summary acquisition variant.

No object here certifies F/Q, opens the BCE oracle, creates a candidate or
mask transition, or observes terminal validation.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Final, Literal, NoReturn

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.geometry.trust import GlobalTrustGeometryState

from .contracts import content_hash
from .modern_v1_campaign_contract import V1OptionACampaignBinding
from .modern_v1_fqr_adaptive_rank import V1FQRAdaptiveRankPolicy
from .modern_v1_sampled_device_final_rank_source import (
    V1SampledDeviceFinalRankSourceReceipt,
)
from .modern_v1_proposal_statistics import (
    V1StratifiedCorrectedAttributionMoments,
)
from .modern_v1_response_exactification import (
    V1OrdinalPhysicalAxisBridgeReceipt,
    V1ResponseExactificationReplayReceipt,
)
from .modern_v1_sampled_external_state import (
    V1SampledRegisteredLookExternalDescriptor,
    V1SampledRegisteredLookExternalView,
)
from .modern_v1_sampled_look_transaction import (
    V1SampledDualChartRegisteredLookReceipt,
    V1SampledRegisteredLookCommit,
)
from .modern_v1_sampled_proposal_streaming import (
    RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE,
)
from .modern_v1_stratified_fd_gpu import (
    V1StratifiedDeviceFDTrustAccumulator,
    V1StratifiedFDAccumulatorSnapshot,
    V1StratifiedFDDesign,
    V1StratifiedFDGeometryResult,
)


SAMPLED_DISCOVERY_SOURCE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_discovery_acquisition_source_receipt_v1"
)
SAMPLED_DISCOVERY_ADMISSION_SCHEMA: Final = (
    "partition_free_v1_sampled_discovery_acquisition_admission_v1"
)
SAMPLED_SUMMARY_GEOMETRY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_summary_geometry_receipt_v1"
)
SAMPLED_SUMMARY_ACQUISITION_SOURCE_SCHEMA: Final = (
    "partition_free_v1_sampled_summary_acquisition_source_v1"
)

SAMPLED_SUMMARY_AUTHORITY_SCOPE: Final = (
    "proposal_numeric_source_only_no_fq_oracle_candidate_mask_or_terminal_authority_v1"
)
SAMPLED_SUMMARY_STANDARD_ACQUISITION_GAP_ID: Final = (
    "sampled_sufficient_state_is_noninjective_to_complete_ordered_raw_row_stream_"
    "and_exact_per_draw_source_hash_ledger_v1"
)
SAMPLED_POINT_ESTIMATE_STATUS: Final = (
    "observed_sealed_prefix_point_estimate_no_census_authority_v1"
)
REGRET_ORDER_STATUS: Final = (
    "registered_action_regret_certifies_prefix_order_not_point_value_exactness_v1"
)

_SOURCE_PREFIX: Final = "partition-free-v1-sampled-acquisition-source-"
_ADMISSION_PREFIX: Final = "partition-free-v1-sampled-acquisition-admission-"
_GEOMETRY_PREFIX: Final = "partition-free-v1-sampled-summary-geometry-"
_SUMMARY_PREFIX: Final = "partition-free-v1-sampled-summary-acquisition-source-"
_CONTEXT_MIXTURE_PREFIX: Final = "partition-free-v1-sampled-context-mixture-"
_POINT_PREFIX: Final = "partition-free-v1-sampled-corrected-point-"
_SIGNED_POINT_PREFIX: Final = "partition-free-v1-sampled-signed-point-"
_ABSOLUTE_POINT_PREFIX: Final = "partition-free-v1-sampled-absolute-point-"
_DRAW_AXIS_PREFIX: Final = "partition-free-v1-sampled-draw-axis-"
_SOURCE_CONTEXT_AXIS_PREFIX: Final = "partition-free-v1-sampled-source-context-axis-"
_SOURCE_MEASUREMENT_AXIS_PREFIX: Final = (
    "partition-free-v1-sampled-source-measurement-axis-"
)
_SAMPLE_WEIGHT_PREFIX: Final = "partition-free-v1-sampled-point-weight-"
_PERMUTATION_PREFIX: Final = "partition-free-v1-sampled-permutation-lineage-"
_REGRET_AXIS_PREFIX: Final = "partition-free-v1-sampled-regret-receipt-axis-"
_SNAPSHOT_NUMERICS_PREFIX: Final = "partition-free-v1-sampled-snapshot-numerics-"
_VALIDATION_FIREWALL_PREFIX: Final = "partition-free-v1-sampled-validation-firewall-"
_STANDARD_GAP_PREFIX: Final = "partition-free-v1-sampled-standard-acquisition-gap-"

SourceKind = Literal["registered_look_commit", "external_registered_look_commit"]
PointStatus = Literal["observed_sealed_prefix_point_estimate_no_census_authority_v1",]


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be one positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _pair_values(
    values: object,
    *,
    name: str,
) -> tuple[object, object]:
    if type(values) not in (tuple, list):
        raise TypeError(f"{name} must be an exact tuple or list")
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return values[0], values[1]


def _pair_of_identities(
    values: object,
    *,
    name: str,
) -> tuple[str, str]:
    pair = _pair_values(values, name=name)
    result = tuple(_identity(value, name=name) for value in pair)
    if len(set(result)) != 2:
        raise ValueError(f"{name} must contain two unique identities")
    return result


def _pair_of_positive_ints(
    values: object,
    *,
    name: str,
) -> tuple[int, int]:
    pair = _pair_values(values, name=name)
    return tuple(_positive_int(value, name=name) for value in pair)


def _pair_of_nonnegative_ints(
    values: object,
    *,
    name: str,
) -> tuple[int, int]:
    pair = _pair_values(values, name=name)
    return tuple(_nonnegative_int(value, name=name) for value in pair)


def _pair_of_class_balanced_weights(
    values: object,
    *,
    name: str,
) -> tuple[float, float]:
    pair = _pair_values(values, name=name)
    result: list[float] = []
    for value in pair:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(f"{name} must contain finite numeric values")
        weight = float(value)
        if not np.isfinite(weight):
            raise ValueError(f"{name} must contain finite numeric values")
        result.append(weight)
    weights = result[0], result[1]
    if weights != (0.5, 0.5):
        raise ValueError("sampled acquisition requires exact class-balanced weights")
    return weights


def _context_mixture_hash(
    *,
    campaign_contract_hash: str,
    context_population_hash: str,
    sampling_law_hash: str,
    raw_logit_stream_source_hash: str,
    current_mask_hash: str,
    proposal_generation: int,
    stratum_names: tuple[str, str],
    stratum_weights: tuple[float, float],
    population_stratum_counts: tuple[int, int],
    sealed_population_order_hashes: tuple[str, str],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_context_mixture_v1",
            "campaign_contract_hash": campaign_contract_hash,
            "context_population_hash": context_population_hash,
            "sampling_law_hash": sampling_law_hash,
            "raw_logit_stream_source_hash": raw_logit_stream_source_hash,
            "current_mask_hash": current_mask_hash,
            "proposal_generation": proposal_generation,
            "stratum_names": stratum_names,
            "stratum_weights": stratum_weights,
            "population_stratum_counts": population_stratum_counts,
            "sealed_population_order_hashes": sealed_population_order_hashes,
            "estimand_weighting": "equal_stratum_macro_mean_v1",
        },
        prefix=_CONTEXT_MIXTURE_PREFIX,
    )


def _direct_mask_hash(mask: HardMask) -> str:
    if type(mask) is not HardMask:
        raise TypeError("sampled acquisition requires the exact HardMask type")
    return content_hash(
        {
            "route_axis_hash": mask.route_axis_hash,
            "route_count": mask.route_count,
            "packed_bits": mask.packed_bits,
        },
        prefix="irg-mask-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledDiscoverySourceReceipt:
    """Common verified coordinates for a sampled-summary acquisition variant."""

    source_kind: SourceKind
    source_commit_hash: str
    source_parent_hash: str
    transaction_request_hash: str | None
    external_state_descriptor_hash: str | None
    campaign_contract_hash: str
    campaign_binding_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    context_population_hash: str
    sampling_law_hash: str
    current_mask_hash: str
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    route_count: int
    proposal_generation: int
    calibration_identity: str
    raw_logit_stream_source_hash: str
    raw_logit_coordinate: str
    context_mixture_hash: str
    stratum_names: tuple[str, str]
    stratum_weights: tuple[float, float]
    population_stratum_counts: tuple[int, int]
    sealed_population_order_hashes: tuple[str, str]
    observed_stratum_counts: tuple[int, int]
    sampled_look_index: int
    fd_packet_boundary_index: int
    observed_draw_axis_hash: str
    observed_source_context_axis_hash: str
    source_measurement_axis_hash: str
    sample_weight_hash: str
    corrected_moments_state_hash: str
    signed_point_hash: str
    absolute_point_hash: str
    corrected_point_estimate_hash: str
    point_status: PointStatus
    fd_design_hash: str
    fd_snapshot_hash: str
    snapshot_numerical_backend_hash: str
    proposal_order_design_hash: str
    score_order_registration_hash: str
    score_order_look_receipt_hash: str
    score_order_confidence_box_hash: str
    score_order_regret_receipt_axis_hash: str
    response_exactification_replay_receipt_hash: str
    axis_bridge_receipt_hash: str
    permutation_lineage_hash: str
    terminal_validation_population_hash: str
    terminal_validation_policy_hash: str
    validation_firewall_hash: str
    regret_order_status: str = REGRET_ORDER_STATUS
    census_authority: bool = False
    raw_context_route_rows_retained: bool = False
    full_per_draw_source_row_hash_ledger_retained: bool = False
    source_row_proof_authority: bool = False
    exact_tangent_price_acquisition_issued: bool = False
    fd_stream_acquisition_issued: bool = False
    proposal_only: bool = True
    score_order_authority: bool = False
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    validation_inputs_present: bool = False
    authority_scope: str = SAMPLED_SUMMARY_AUTHORITY_SCOPE
    schema: str = SAMPLED_DISCOVERY_SOURCE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "source_commit_hash",
            "source_parent_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "context_population_hash",
            "sampling_law_hash",
            "current_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "calibration_identity",
            "raw_logit_stream_source_hash",
            "context_mixture_hash",
            "observed_draw_axis_hash",
            "observed_source_context_axis_hash",
            "source_measurement_axis_hash",
            "sample_weight_hash",
            "corrected_moments_state_hash",
            "signed_point_hash",
            "absolute_point_hash",
            "corrected_point_estimate_hash",
            "fd_design_hash",
            "fd_snapshot_hash",
            "snapshot_numerical_backend_hash",
            "proposal_order_design_hash",
            "score_order_registration_hash",
            "score_order_look_receipt_hash",
            "score_order_confidence_box_hash",
            "score_order_regret_receipt_axis_hash",
            "response_exactification_replay_receipt_hash",
            "axis_bridge_receipt_hash",
            "permutation_lineage_hash",
            "terminal_validation_population_hash",
            "terminal_validation_policy_hash",
            "validation_firewall_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.transaction_request_hash is not None:
            object.__setattr__(
                self,
                "transaction_request_hash",
                _identity(
                    self.transaction_request_hash, name="transaction request hash"
                ),
            )
        if self.external_state_descriptor_hash is not None:
            object.__setattr__(
                self,
                "external_state_descriptor_hash",
                _identity(
                    self.external_state_descriptor_hash,
                    name="external state descriptor hash",
                ),
            )
        if self.source_kind == "registered_look_commit":
            if (
                self.transaction_request_hash is None
                or self.external_state_descriptor_hash is not None
            ):
                raise ValueError("live sampled source has an invalid origin lineage")
        elif self.source_kind == "external_registered_look_commit":
            if (
                self.external_state_descriptor_hash is None
                or self.transaction_request_hash is not None
            ):
                raise ValueError(
                    "external sampled source has an invalid origin lineage"
                )
        else:
            raise ValueError("unsupported sampled source kind")
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        object.__setattr__(
            self,
            "sampled_look_index",
            _positive_int(self.sampled_look_index, name="sampled look index"),
        )
        object.__setattr__(
            self,
            "fd_packet_boundary_index",
            _positive_int(
                self.fd_packet_boundary_index,
                name="FD packet boundary index",
            ),
        )
        names = _pair_of_identities(self.stratum_names, name="stratum names")
        populations = _pair_of_positive_ints(
            self.population_stratum_counts,
            name="population stratum counts",
        )
        counts = _pair_of_nonnegative_ints(
            self.observed_stratum_counts,
            name="observed stratum counts",
        )
        orders = _pair_of_identities(
            self.sealed_population_order_hashes,
            name="sealed population order hashes",
        )
        weights = _pair_of_class_balanced_weights(
            self.stratum_weights,
            name="stratum weights",
        )
        if any(
            count <= 0 or count > population
            for count, population in zip(counts, populations)
        ):
            raise ValueError(
                "sampled acquisition stratum counts are incomplete or overrun"
            )
        if self.point_status != SAMPLED_POINT_ESTIMATE_STATUS:
            raise ValueError("sampled source cannot promote a point estimate to census")
        if self.physical_route_axis_hash == self.ordered_route_axis_hash:
            raise ValueError(
                "sampled acquisition must preserve distinct physical and ordered axes"
            )
        if self.raw_logit_coordinate != RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE:
            raise ValueError(
                "sampled acquisition requires the raw-target-logit coordinate"
            )
        if self.regret_order_status != REGRET_ORDER_STATUS:
            raise ValueError("sampled acquisition changed regret-order semantics")
        expected_mixture = _context_mixture_hash(
            campaign_contract_hash=self.campaign_contract_hash,
            context_population_hash=self.context_population_hash,
            sampling_law_hash=self.sampling_law_hash,
            raw_logit_stream_source_hash=self.raw_logit_stream_source_hash,
            current_mask_hash=self.current_mask_hash,
            proposal_generation=self.proposal_generation,
            stratum_names=names,
            stratum_weights=weights,  # type: ignore[arg-type]
            population_stratum_counts=populations,
            sealed_population_order_hashes=orders,
        )
        if self.context_mixture_hash != expected_mixture:
            raise ValueError("sampled context-mixture hash does not replay")
        if (
            self.census_authority is not False
            or self.raw_context_route_rows_retained is not False
            or self.full_per_draw_source_row_hash_ledger_retained is not False
            or self.source_row_proof_authority is not False
            or self.exact_tangent_price_acquisition_issued is not False
            or self.fd_stream_acquisition_issued is not False
            or self.proposal_only is not True
            or self.score_order_authority is not False
            or self.fq_certification_authority is not False
            or self.bce_oracle_authority is not False
            or self.candidate_freeze_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
            or self.validation_inputs_present is not False
            or self.authority_scope != SAMPLED_SUMMARY_AUTHORITY_SCOPE
            or self.schema != SAMPLED_DISCOVERY_SOURCE_RECEIPT_SCHEMA
        ):
            raise PermissionError(
                "sampled discovery source crossed its authority boundary"
            )
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "observed_stratum_counts", counts)
        object.__setattr__(self, "sealed_population_order_hashes", orders)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_SOURCE_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SampledDiscoveryAdmission:
    """Exact sampled state graph admitted for proposal-numeric adaptation."""

    binding: V1OptionACampaignBinding = field(repr=False, compare=False)
    current_mask: HardMask = field(repr=False, compare=False)
    moments: V1StratifiedCorrectedAttributionMoments = field(
        repr=False,
        compare=False,
    )
    fd_snapshot: V1StratifiedFDAccumulatorSnapshot = field(
        repr=False,
        compare=False,
    )
    fd_design: V1StratifiedFDDesign = field(repr=False, compare=False)
    axis_bridge: V1OrdinalPhysicalAxisBridgeReceipt
    score_order_replay: V1ResponseExactificationReplayReceipt
    source_origin: (
        V1SampledRegisteredLookCommit | V1SampledRegisteredLookExternalDescriptor
    ) = field(repr=False, compare=False)
    receipt: V1SampledDiscoverySourceReceipt
    schema: str = SAMPLED_DISCOVERY_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.binding) is not V1OptionACampaignBinding:
            raise TypeError("sampled admission requires an exact campaign binding")
        if type(self.current_mask) is not HardMask:
            raise TypeError("sampled admission requires an exact current mask")
        if type(self.moments) is not V1StratifiedCorrectedAttributionMoments:
            raise TypeError("sampled admission requires exact corrected moments")
        if type(self.fd_snapshot) is not V1StratifiedFDAccumulatorSnapshot:
            raise TypeError("sampled admission requires an exact FD snapshot")
        if type(self.fd_design) is not V1StratifiedFDDesign:
            raise TypeError("sampled admission requires an exact FD design")
        if type(self.axis_bridge) is not V1OrdinalPhysicalAxisBridgeReceipt:
            raise TypeError("sampled admission requires an exact axis bridge")
        if type(self.score_order_replay) is not V1ResponseExactificationReplayReceipt:
            raise TypeError("sampled admission requires exact score-order replay")
        if type(self.receipt) is not V1SampledDiscoverySourceReceipt:
            raise TypeError("sampled admission requires its exact source receipt")
        moments_hash = self.moments.assert_integrity()
        snapshot_hash = self.fd_snapshot.assert_integrity()
        design_hash = self.fd_design.assert_integrity()
        bridge_hash = self.axis_bridge.assert_integrity()
        replay_hash = self.score_order_replay.assert_integrity()
        receipt = self.receipt
        contract = self.binding.contract
        if (
            receipt.corrected_moments_state_hash != moments_hash
            or receipt.fd_snapshot_hash != snapshot_hash
            or receipt.fd_design_hash != design_hash
            or receipt.axis_bridge_receipt_hash != bridge_hash
            or receipt.response_exactification_replay_receipt_hash != replay_hash
            or receipt.campaign_contract_hash != contract.campaign_contract_hash
            or receipt.campaign_binding_hash != self.binding.binding_hash
            or receipt.context_population_hash != contract.context_population_hash
            or receipt.sampling_law_hash != contract.sampling_law_hash
            or receipt.current_mask_hash != self.current_mask.mask_hash
            or receipt.physical_route_axis_hash != self.current_mask.route_axis_hash
            or receipt.terminal_validation_population_hash
            != contract.terminal_validation_population_hash
            or receipt.terminal_validation_policy_hash
            != contract.terminal_validation_policy_hash
            or self.score_order_replay.statistics_hash != moments_hash
            or self.score_order_replay.axis_bridge_receipt_hash != bridge_hash
            or self.axis_bridge.ordered_route_axis_hash != self.moments.route_axis_hash
            or self.axis_bridge.ordered_route_axis_hash
            != self.fd_design.route_axis_hash
            or self.axis_bridge.current_mask_hash != self.moments.current_mask_hash
            or self.fd_snapshot.design_hash != design_hash
            or self.fd_snapshot.route_axis_hash != self.fd_design.route_axis_hash
        ):
            raise ValueError("sampled admission source graph does not close")
        if _direct_mask_hash(self.current_mask) != self.current_mask.mask_hash:
            raise ValueError("sampled admission current mask is not canonical")

        source_measurements = self.fd_snapshot.source_measurement_hashes_by_stratum
        expected_draw_axis_hash = content_hash(
            self.moments.observed_draw_ids_by_stratum,
            prefix=_DRAW_AXIS_PREFIX,
        )
        expected_source_context_axis_hash = content_hash(
            self.moments.observed_context_ids_by_stratum,
            prefix=_SOURCE_CONTEXT_AXIS_PREFIX,
        )
        expected_source_measurement_axis_hash = content_hash(
            source_measurements,
            prefix=_SOURCE_MEASUREMENT_AXIS_PREFIX,
        )
        expected_sample_weight_hash = content_hash(
            {
                "stratum_weights": self.fd_design.stratum_weights,
                "observed_draw_ids_by_stratum": (
                    self.moments.observed_draw_ids_by_stratum
                ),
                "observed_stratum_counts": self.moments.stratum_counts,
                "weighting": "equal_stratum_mean_then_equal_macro_average_v1",
            },
            prefix=_SAMPLE_WEIGHT_PREFIX,
        )
        expected_signed_hash = content_hash(
            self.moments.class_balanced_signed_mean,
            prefix=_SIGNED_POINT_PREFIX,
        )
        expected_absolute_hash = content_hash(
            self.moments.class_balanced_absolute_mean,
            prefix=_ABSOLUTE_POINT_PREFIX,
        )
        expected_point_hash = content_hash(
            {
                "moments_state_hash": moments_hash,
                "signed_point_hash": expected_signed_hash,
                "absolute_point_hash": expected_absolute_hash,
                "sample_weight_hash": expected_sample_weight_hash,
                "point_status": SAMPLED_POINT_ESTIMATE_STATUS,
                "census_authority": False,
            },
            prefix=_POINT_PREFIX,
        )
        expected_snapshot_numerics_hash = content_hash(
            {
                "fd_design_hash": design_hash,
                "fd_snapshot_hash": snapshot_hash,
                "child_numerics": tuple(
                    (
                        name,
                        child.producer_algorithm_id,
                        child.producer_implementation,
                        child.dtype_identity,
                        child.sketch_rank,
                        child.snapshot_hash,
                    )
                    for name, child in self.fd_snapshot.stratum_snapshots
                ),
                "raw_context_route_rows_retained": False,
            },
            prefix=_SNAPSHOT_NUMERICS_PREFIX,
        )
        expected_regret_axis_hash = content_hash(
            self.score_order_replay.action_regret_receipt_hashes,
            prefix=_REGRET_AXIS_PREFIX,
        )
        expected_permutation_hash = content_hash(
            {
                "axis_bridge_receipt_hash": bridge_hash,
                "physical_route_axis_hash": self.current_mask.route_axis_hash,
                "ordered_route_axis_hash": self.moments.route_axis_hash,
                "sealed_population_order_hashes": (
                    self.moments.sealed_population_order_hashes
                ),
                "position_rule": self.axis_bridge.position_rule,
            },
            prefix=_PERMUTATION_PREFIX,
        )
        expected_validation_firewall_hash = content_hash(
            {
                "campaign_contract_hash": contract.campaign_contract_hash,
                "proposal_context_population_hash": contract.context_population_hash,
                "terminal_validation_population_hash": (
                    contract.terminal_validation_population_hash
                ),
                "terminal_validation_policy_hash": (
                    contract.terminal_validation_policy_hash
                ),
                "populations_are_distinct": True,
                "adapter_input_types_contain_validation_outcomes": False,
                "validation_inputs_present": False,
                "terminal_authority": False,
            },
            prefix=_VALIDATION_FIREWALL_PREFIX,
        )
        exact_receipt_values = (
            (receipt.model_identity_hash, self.moments.model_identity_hash),
            (receipt.query_contract_hash, self.moments.query_contract_hash),
            (receipt.preprocessing_hash, self.moments.preprocessing_hash),
            (receipt.ordered_route_axis_hash, self.moments.route_axis_hash),
            (receipt.route_count, self.moments.route_count),
            (receipt.proposal_generation, self.moments.proposal_generation),
            (receipt.calibration_identity, self.fd_design.calibration_state_id),
            (
                receipt.raw_logit_stream_source_hash,
                self.moments.raw_logit_stream_source_hash,
            ),
            (receipt.stratum_names, self.moments.stratum_names),
            (receipt.stratum_weights, self.fd_design.stratum_weights),
            (
                receipt.population_stratum_counts,
                self.moments.population_stratum_counts,
            ),
            (
                receipt.sealed_population_order_hashes,
                self.moments.sealed_population_order_hashes,
            ),
            (receipt.observed_stratum_counts, self.moments.stratum_counts),
            (receipt.sampled_look_index, self.moments.look_index),
            (
                receipt.fd_packet_boundary_index,
                self.fd_snapshot.packet_boundary_index,
            ),
            (receipt.observed_draw_axis_hash, expected_draw_axis_hash),
            (
                receipt.observed_source_context_axis_hash,
                expected_source_context_axis_hash,
            ),
            (
                receipt.source_measurement_axis_hash,
                expected_source_measurement_axis_hash,
            ),
            (receipt.sample_weight_hash, expected_sample_weight_hash),
            (receipt.signed_point_hash, expected_signed_hash),
            (receipt.absolute_point_hash, expected_absolute_hash),
            (receipt.corrected_point_estimate_hash, expected_point_hash),
            (
                receipt.snapshot_numerical_backend_hash,
                expected_snapshot_numerics_hash,
            ),
            (
                receipt.proposal_order_design_hash,
                self.score_order_replay.proposal_order_design_hash,
            ),
            (
                receipt.score_order_registration_hash,
                self.score_order_replay.score_order_registration_hash,
            ),
            (
                receipt.score_order_look_receipt_hash,
                self.score_order_replay.score_order_look_receipt_hash,
            ),
            (
                receipt.score_order_confidence_box_hash,
                self.score_order_replay.confidence_box_hash,
            ),
            (receipt.score_order_regret_receipt_axis_hash, expected_regret_axis_hash),
            (receipt.permutation_lineage_hash, expected_permutation_hash),
            (receipt.validation_firewall_hash, expected_validation_firewall_hash),
        )
        if any(left != right for left, right in exact_receipt_values):
            raise ValueError("sampled admission receipt fields do not replay")
        origin = self.source_origin
        if type(origin) is V1SampledRegisteredLookCommit:
            if (
                receipt.source_kind != "registered_look_commit"
                or receipt.source_commit_hash != origin.commit_hash
                or receipt.source_parent_hash != origin.request.request_hash
                or receipt.transaction_request_hash != origin.request.request_hash
                or origin.corrected_attribution_moments.state_hash != moments_hash
                or origin.fd_snapshot.snapshot_hash != snapshot_hash
            ):
                raise ValueError("sampled live origin differs from its receipt")
        elif type(origin) is V1SampledRegisteredLookExternalDescriptor:
            if (
                receipt.source_kind != "external_registered_look_commit"
                or receipt.source_commit_hash != origin.commit_hash
                or receipt.source_parent_hash != origin.external_state.descriptor_hash
                or receipt.external_state_descriptor_hash
                != origin.external_state.descriptor_hash
                or origin.moments_state_hash != moments_hash
                or origin.fd_snapshot_hash != snapshot_hash
                or origin.fd_design_hash != design_hash
            ):
                raise ValueError("sampled external origin differs from its receipt")
        else:
            raise TypeError("sampled admission has an unsupported source origin")
        if self.schema != SAMPLED_DISCOVERY_ADMISSION_SCHEMA:
            raise ValueError("unsupported sampled discovery-admission schema")

    @property
    def admission_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "campaign_binding_hash": self.binding.binding_hash,
                "current_mask_hash": self.current_mask.mask_hash,
                "source_receipt_hash": self.receipt.receipt_hash,
                "moments_state_hash": self.moments.assert_integrity(),
                "fd_snapshot_hash": self.fd_snapshot.assert_integrity(),
                "fd_design_hash": self.fd_design.assert_integrity(),
                "axis_bridge_receipt_hash": self.axis_bridge.assert_integrity(),
                "score_order_replay_receipt_hash": (
                    self.score_order_replay.assert_integrity()
                ),
            },
            prefix=_ADMISSION_PREFIX,
        )

    @property
    def signed_point(self) -> np.ndarray:
        return self.moments.class_balanced_signed_mean

    @property
    def absolute_point(self) -> np.ndarray:
        return self.moments.class_balanced_absolute_mean


@dataclass(frozen=True, slots=True)
class V1SampledSummaryGeometryReceipt:
    """Replay receipt for stratified FD assembly from sampled sufficient state."""

    source_receipt_hash: str
    source_admission_hash: str
    campaign_contract_hash: str
    campaign_binding_hash: str
    current_mask_hash: str
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    proposal_generation: int
    context_mixture_hash: str
    calibration_identity: str
    raw_logit_stream_source_hash: str
    raw_logit_coordinate: str
    corrected_point_estimate_hash: str
    score_order_regret_receipt_axis_hash: str
    permutation_lineage_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    geometry_state_hash: str
    stratified_assembly_receipt_hash: str
    stratified_geometry_result_hash: str
    source_measurement_axis_hash: str
    snapshot_numerical_backend_hash: str
    assembly_device_identity: str
    standard_acquisition_gap_proof_hash: str
    census_authority: bool = False
    raw_context_route_rows_retained: bool = False
    full_per_draw_source_row_hash_ledger_retained: bool = False
    source_row_proof_authority: bool = False
    standard_fd_stream_acquisition_compatible: bool = False
    exact_tangent_price_acquisition_compatible: bool = False
    proposal_only: bool = True
    score_order_authority: bool = False
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    validation_inputs_present: bool = False
    authority_scope: str = SAMPLED_SUMMARY_AUTHORITY_SCOPE
    schema: str = SAMPLED_SUMMARY_GEOMETRY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "source_receipt_hash",
            "source_admission_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "current_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "context_mixture_hash",
            "calibration_identity",
            "raw_logit_stream_source_hash",
            "corrected_point_estimate_hash",
            "score_order_regret_receipt_axis_hash",
            "permutation_lineage_hash",
            "fd_design_hash",
            "fd_snapshot_hash",
            "geometry_state_hash",
            "stratified_assembly_receipt_hash",
            "stratified_geometry_result_hash",
            "source_measurement_axis_hash",
            "snapshot_numerical_backend_hash",
            "assembly_device_identity",
            "standard_acquisition_gap_proof_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        if self.raw_logit_coordinate != RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE:
            raise ValueError("sampled summary changed the raw-logit coordinate")
        if (
            self.census_authority is not False
            or self.raw_context_route_rows_retained is not False
            or self.full_per_draw_source_row_hash_ledger_retained is not False
            or self.source_row_proof_authority is not False
            or self.standard_fd_stream_acquisition_compatible is not False
            or self.exact_tangent_price_acquisition_compatible is not False
            or self.proposal_only is not True
            or self.score_order_authority is not False
            or self.fq_certification_authority is not False
            or self.bce_oracle_authority is not False
            or self.candidate_freeze_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
            or self.validation_inputs_present is not False
            or self.authority_scope != SAMPLED_SUMMARY_AUTHORITY_SCOPE
            or self.schema != SAMPLED_SUMMARY_GEOMETRY_RECEIPT_SCHEMA
        ):
            raise PermissionError(
                "sampled summary geometry crossed its authority boundary"
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_GEOMETRY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SampledSummaryAcquisitionSource:
    """Assembled sampled geometry plus its exact proposal-only source graph."""

    discovery: V1SampledDiscoveryAdmission = field(repr=False, compare=False)
    geometry_result: V1StratifiedFDGeometryResult = field(repr=False, compare=False)
    receipt: V1SampledSummaryGeometryReceipt
    schema: str = SAMPLED_SUMMARY_ACQUISITION_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.discovery) is not V1SampledDiscoveryAdmission:
            raise TypeError("sampled summary requires an exact discovery admission")
        if type(self.geometry_result) is not V1StratifiedFDGeometryResult:
            raise TypeError("sampled summary requires an exact stratified geometry")
        if type(self.receipt) is not V1SampledSummaryGeometryReceipt:
            raise TypeError("sampled summary requires its exact geometry receipt")
        source = self.discovery.receipt
        result = self.geometry_result
        result.design.assert_integrity(source.fd_design_hash)
        result.receipt.assert_integrity(self.receipt.stratified_assembly_receipt_hash)
        receipt = self.receipt
        expected_gap_hash = content_hash(
            {
                "schema": (
                    "partition_free_v1_sampled_standard_acquisition_gap_proof_v1"
                ),
                "gap_id": SAMPLED_SUMMARY_STANDARD_ACQUISITION_GAP_ID,
                "source_receipt_hash": source.receipt_hash,
                "fd_snapshot_hash": source.fd_snapshot_hash,
                "geometry_state_hash": result.geometry.content_hash,
                "raw_context_route_rows_retained": False,
                "full_per_draw_source_row_hash_ledger_retained": False,
                "global_trust_derivation_receipt_present": False,
                "standard_fd_stream_acquisition_compatible": False,
                "exact_tangent_price_acquisition_compatible": False,
            },
            prefix=_STANDARD_GAP_PREFIX,
        )
        exact_receipt_values = (
            (receipt.source_receipt_hash, source.receipt_hash),
            (receipt.source_admission_hash, self.discovery.admission_hash),
            (receipt.campaign_contract_hash, source.campaign_contract_hash),
            (receipt.campaign_binding_hash, source.campaign_binding_hash),
            (receipt.current_mask_hash, source.current_mask_hash),
            (receipt.physical_route_axis_hash, source.physical_route_axis_hash),
            (receipt.ordered_route_axis_hash, source.ordered_route_axis_hash),
            (receipt.proposal_generation, source.proposal_generation),
            (receipt.context_mixture_hash, source.context_mixture_hash),
            (receipt.calibration_identity, source.calibration_identity),
            (
                receipt.raw_logit_stream_source_hash,
                source.raw_logit_stream_source_hash,
            ),
            (receipt.raw_logit_coordinate, source.raw_logit_coordinate),
            (
                receipt.corrected_point_estimate_hash,
                source.corrected_point_estimate_hash,
            ),
            (
                receipt.score_order_regret_receipt_axis_hash,
                source.score_order_regret_receipt_axis_hash,
            ),
            (receipt.permutation_lineage_hash, source.permutation_lineage_hash),
            (receipt.fd_design_hash, source.fd_design_hash),
            (receipt.fd_snapshot_hash, source.fd_snapshot_hash),
            (receipt.geometry_state_hash, result.geometry.content_hash),
            (
                receipt.stratified_assembly_receipt_hash,
                result.receipt.receipt_hash,
            ),
            (receipt.stratified_geometry_result_hash, result.result_hash),
            (
                receipt.source_measurement_axis_hash,
                source.source_measurement_axis_hash,
            ),
            (
                receipt.snapshot_numerical_backend_hash,
                source.snapshot_numerical_backend_hash,
            ),
            (receipt.assembly_device_identity, result.receipt.device_identity),
            (receipt.standard_acquisition_gap_proof_hash, expected_gap_hash),
            (receipt.census_authority, source.census_authority),
        )
        if (
            any(left != right for left, right in exact_receipt_values)
            or result.receipt.combined_geometry_hash != result.geometry.content_hash
            or result.geometry.derivation_receipt is not None
            or result.geometry.metadata.get("sampled_summary_source_receipt_hash")
            != source.receipt_hash
        ):
            raise ValueError("sampled summary geometry graph does not close")
        if self.schema != SAMPLED_SUMMARY_ACQUISITION_SOURCE_SCHEMA:
            raise ValueError("unsupported sampled-summary acquisition-source schema")

    @property
    def source_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "discovery_admission_hash": self.discovery.admission_hash,
                "geometry_result_hash": self.geometry_result.result_hash,
                "geometry_receipt_hash": self.receipt.receipt_hash,
            },
            prefix=_SUMMARY_PREFIX,
        )

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def mask_commit_authority(self) -> bool:
        return False


class V1SampledSummaryAcquisitionIncompatibility(ValueError):
    """Raised when a sampled summary is coerced into a raw-stream owner."""


def _dual_chart_stream_binding(
    source_origin: (
        V1SampledRegisteredLookCommit | V1SampledRegisteredLookExternalDescriptor
    ),
) -> tuple[str, str, str] | None:
    """Return the verified evidence/Jacobian/proof binding, or legacy ``None``."""

    if type(source_origin) is V1SampledRegisteredLookCommit:
        proof = source_origin.dual_chart_receipt
        if source_origin.canonical_q_input_authority is False:
            if proof is not None:
                raise ValueError(
                    "legacy sampled commit contains an unauthorized dual-chart proof"
                )
            return None
        if type(proof) is not V1SampledDualChartRegisteredLookReceipt:
            raise ValueError(
                "canonical sampled commit lacks its exact same-capture/VJP proof"
            )
        proof_hash = proof.receipt_hash
        if (
            proof.transaction_request_hash != source_origin.request.request_hash
            or proof.same_capture_and_vjp is not True
            or proof.corrected_rows_excluded_from_response_geometry is not True
            or proof.canonical_q_input_authority is not True
        ):
            raise ValueError("canonical sampled commit has a stale dual-chart proof")
        return (
            proof.corrected_evidence_stream_identity_hash,
            proof.literal_jacobian_stream_identity_hash,
            proof_hash,
        )
    if type(source_origin) is V1SampledRegisteredLookExternalDescriptor:
        if source_origin.canonical_q_input_authority is False:
            if (
                any(
                    value is not None
                    for value in (
                        source_origin.corrected_evidence_stream_identity_hash,
                        source_origin.literal_jacobian_stream_identity_hash,
                        source_origin.dual_chart_receipt_hash,
                    )
                )
                or source_origin.same_capture_and_vjp is not False
            ):
                raise ValueError(
                    "legacy sampled descriptor contains unauthorized dual-chart fields"
                )
            return None
        evidence = _identity(
            source_origin.corrected_evidence_stream_identity_hash,
            name="external corrected-evidence stream identity",
        )
        jacobian = _identity(
            source_origin.literal_jacobian_stream_identity_hash,
            name="external literal-Jacobian stream identity",
        )
        proof_hash = _identity(
            source_origin.dual_chart_receipt_hash,
            name="external dual-chart receipt hash",
        )
        if evidence == jacobian or source_origin.same_capture_and_vjp is not True:
            raise ValueError(
                "external dual-chart proof collapsed its stream identities"
            )
        return evidence, jacobian, proof_hash
    raise TypeError("sampled source origin has an unsupported exact type")


def _assert_sampled_dual_or_legacy_stream_coordinates(
    *,
    moments: V1StratifiedCorrectedAttributionMoments,
    fd_design: V1StratifiedFDDesign,
    source_origin: (
        V1SampledRegisteredLookCommit | V1SampledRegisteredLookExternalDescriptor
    ),
) -> tuple[str, str, str] | None:
    dual = _dual_chart_stream_binding(source_origin)
    if dual is None:
        if (
            moments.raw_logit_stream_source_hash
            != fd_design.raw_logit_stream_source_hash
        ):
            raise ValueError(
                "sampled state split evidence and FD identities without an exact "
                "same-capture/VJP dual receipt"
            )
        return None
    evidence, jacobian, proof_hash = dual
    if (
        evidence == jacobian
        or moments.raw_logit_stream_source_hash != evidence
        or fd_design.raw_logit_stream_source_hash != jacobian
    ):
        raise ValueError(
            "sampled dual receipt does not bind corrected moments and literal-J FD"
        )
    return evidence, jacobian, proof_hash


def _admit_sampled_state(
    *,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    moments: V1StratifiedCorrectedAttributionMoments,
    fd_snapshot: V1StratifiedFDAccumulatorSnapshot,
    fd_design: V1StratifiedFDDesign,
    axis_bridge: V1OrdinalPhysicalAxisBridgeReceipt,
    score_order_replay: V1ResponseExactificationReplayReceipt,
    source_origin: (
        V1SampledRegisteredLookCommit | V1SampledRegisteredLookExternalDescriptor
    ),
) -> V1SampledDiscoveryAdmission:
    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("sampled acquisition requires an exact campaign binding")
    if type(current_mask) is not HardMask:
        raise TypeError("sampled acquisition requires an exact current mask")
    if type(moments) is not V1StratifiedCorrectedAttributionMoments:
        raise TypeError("sampled acquisition requires exact corrected moments")
    if type(fd_snapshot) is not V1StratifiedFDAccumulatorSnapshot:
        raise TypeError("sampled acquisition requires an exact FD snapshot")
    if type(fd_design) is not V1StratifiedFDDesign:
        raise TypeError("sampled acquisition requires an exact FD design")
    if type(axis_bridge) is not V1OrdinalPhysicalAxisBridgeReceipt:
        raise TypeError("sampled acquisition requires an exact axis bridge")
    if type(score_order_replay) is not V1ResponseExactificationReplayReceipt:
        raise TypeError("sampled acquisition requires exact score-order replay")
    contract = binding.contract
    mask_hash = _direct_mask_hash(current_mask)
    moments_hash = moments.assert_integrity()
    snapshot_hash = fd_snapshot.assert_integrity()
    design_hash = fd_design.assert_integrity()
    bridge_hash = axis_bridge.assert_integrity()
    replay_hash = score_order_replay.assert_integrity()
    if mask_hash != current_mask.mask_hash:
        raise ValueError("sampled acquisition current mask is not canonical")
    if contract.context_population_hash == contract.terminal_validation_population_hash:
        raise PermissionError(
            "proposal and terminal-validation populations are not separated"
        )
    shared = (
        (moments.campaign_contract_hash, fd_design.campaign_contract_hash),
        (moments.model_identity_hash, fd_design.model_identity_hash),
        (moments.query_contract_hash, fd_design.query_contract_hash),
        (moments.preprocessing_hash, fd_design.preprocessing_hash),
        (moments.current_mask_hash, fd_design.current_mask_hash),
        (moments.current_mask_hash, fd_design.measured_mask_hash),
        (moments.sampling_law_hash, fd_design.sampling_law_hash),
        (moments.proposal_generation, fd_design.proposal_generation),
        (moments.route_count, len(fd_design.route_ids)),
        (moments.route_axis_hash, fd_design.route_axis_hash),
        (moments.stratum_names, fd_design.stratum_names),
    )
    if any(left != right for left, right in shared):
        raise ValueError(
            "sampled moments and FD state name different acquisition coordinates"
        )
    _assert_sampled_dual_or_legacy_stream_coordinates(
        moments=moments,
        fd_design=fd_design,
        source_origin=source_origin,
    )
    if (
        moments.campaign_contract_hash != contract.campaign_contract_hash
        or moments.model_identity_hash != contract.model_seal_hash
        or moments.sampling_law_hash != contract.sampling_law_hash
        or moments.current_mask_hash != current_mask.mask_hash
        or fd_design.calibration_state_id != contract.calibration_identity
        or fd_design.spec.initial_rank != contract.fd_rank
        or fd_design.spec.maximum_rank != contract.fd_rank
        or fd_snapshot.sketch_rank != contract.fd_rank
        or current_mask.route_axis_hash != contract.route_axis_hash
        or current_mask.route_count != moments.route_count
        or fd_snapshot.design_hash != design_hash
        or fd_snapshot.route_axis_hash != fd_design.route_axis_hash
        or fd_snapshot.design_stratum_names != fd_design.stratum_names
        or tuple(child.context_count for _name, child in fd_snapshot.stratum_snapshots)
        != moments.stratum_counts
    ):
        raise ValueError("sampled source differs from the current campaign or mask")
    if tuple(fd_design.stratum_weights) != (0.5, 0.5):
        raise ValueError("sampled source must use the class-balanced target mixture")
    if (
        axis_bridge.current_mask_hash != current_mask.mask_hash
        or axis_bridge.physical_route_axis_hash != current_mask.route_axis_hash
        or axis_bridge.ordered_route_axis_hash != moments.route_axis_hash
        or axis_bridge.route_count != moments.route_count
        or score_order_replay.axis_bridge_receipt_hash != bridge_hash
        or score_order_replay.statistics_hash != moments_hash
    ):
        raise ValueError("sampled score-order permutation or point lineage differs")

    snapshot_contexts = tuple(
        (name, tuple(child.context_ids))
        for name, child in fd_snapshot.stratum_snapshots
    )
    observed_contexts = tuple(
        (name, tuple(moments.observed_context_ids_by_stratum[index]))
        for index, name in enumerate(moments.stratum_names)
    )
    if snapshot_contexts != observed_contexts:
        raise ValueError("sampled FD and corrected moments observed different contexts")
    source_measurements = fd_snapshot.source_measurement_hashes_by_stratum
    if any(
        len(values) != moments.stratum_counts[index]
        for index, (_name, values) in enumerate(source_measurements)
    ):
        raise ValueError("sampled source-measurement ledger is incomplete")

    draw_axis_hash = content_hash(
        moments.observed_draw_ids_by_stratum,
        prefix=_DRAW_AXIS_PREFIX,
    )
    source_context_axis_hash = content_hash(
        moments.observed_context_ids_by_stratum,
        prefix=_SOURCE_CONTEXT_AXIS_PREFIX,
    )
    source_measurement_axis_hash = content_hash(
        source_measurements,
        prefix=_SOURCE_MEASUREMENT_AXIS_PREFIX,
    )
    sample_weight_hash = content_hash(
        {
            "stratum_weights": fd_design.stratum_weights,
            "observed_draw_ids_by_stratum": moments.observed_draw_ids_by_stratum,
            "observed_stratum_counts": moments.stratum_counts,
            "weighting": "equal_stratum_mean_then_equal_macro_average_v1",
        },
        prefix=_SAMPLE_WEIGHT_PREFIX,
    )
    # Hash the two point vectors one at a time.  Keeping both derived arrays
    # live would add two full route vectors to peak host memory at ImageNet
    # scale even though only their content identities belong in control state.
    signed_hash = content_hash(
        moments.class_balanced_signed_mean,
        prefix=_SIGNED_POINT_PREFIX,
    )
    absolute_hash = content_hash(
        moments.class_balanced_absolute_mean,
        prefix=_ABSOLUTE_POINT_PREFIX,
    )
    point_hash = content_hash(
        {
            "moments_state_hash": moments_hash,
            "signed_point_hash": signed_hash,
            "absolute_point_hash": absolute_hash,
            "sample_weight_hash": sample_weight_hash,
            "point_status": SAMPLED_POINT_ESTIMATE_STATUS,
            "census_authority": False,
        },
        prefix=_POINT_PREFIX,
    )
    context_mixture = _context_mixture_hash(
        campaign_contract_hash=contract.campaign_contract_hash,
        context_population_hash=contract.context_population_hash,
        sampling_law_hash=contract.sampling_law_hash,
        raw_logit_stream_source_hash=moments.raw_logit_stream_source_hash,
        current_mask_hash=current_mask.mask_hash,
        proposal_generation=moments.proposal_generation,
        stratum_names=moments.stratum_names,
        stratum_weights=fd_design.stratum_weights,  # type: ignore[arg-type]
        population_stratum_counts=moments.population_stratum_counts,
        sealed_population_order_hashes=moments.sealed_population_order_hashes,
    )
    child_numerics = tuple(
        (
            name,
            child.producer_algorithm_id,
            child.producer_implementation,
            child.dtype_identity,
            child.sketch_rank,
            child.snapshot_hash,
        )
        for name, child in fd_snapshot.stratum_snapshots
    )
    snapshot_numerics_hash = content_hash(
        {
            "fd_design_hash": design_hash,
            "fd_snapshot_hash": snapshot_hash,
            "child_numerics": child_numerics,
            "raw_context_route_rows_retained": False,
        },
        prefix=_SNAPSHOT_NUMERICS_PREFIX,
    )
    regret_axis_hash = content_hash(
        score_order_replay.action_regret_receipt_hashes,
        prefix=_REGRET_AXIS_PREFIX,
    )
    permutation_hash = content_hash(
        {
            "axis_bridge_receipt_hash": bridge_hash,
            "physical_route_axis_hash": current_mask.route_axis_hash,
            "ordered_route_axis_hash": moments.route_axis_hash,
            "sealed_population_order_hashes": moments.sealed_population_order_hashes,
            "position_rule": axis_bridge.position_rule,
        },
        prefix=_PERMUTATION_PREFIX,
    )
    validation_firewall_hash = content_hash(
        {
            "campaign_contract_hash": contract.campaign_contract_hash,
            "proposal_context_population_hash": contract.context_population_hash,
            "terminal_validation_population_hash": (
                contract.terminal_validation_population_hash
            ),
            "terminal_validation_policy_hash": contract.terminal_validation_policy_hash,
            "populations_are_distinct": True,
            "adapter_input_types_contain_validation_outcomes": False,
            "validation_inputs_present": False,
            "terminal_authority": False,
        },
        prefix=_VALIDATION_FIREWALL_PREFIX,
    )
    if type(source_origin) is V1SampledRegisteredLookCommit:
        source_origin.commit_hash
        source_kind: SourceKind = "registered_look_commit"
        source_commit_hash = source_origin.commit_hash
        source_parent_hash = source_origin.request.request_hash
        transaction_request_hash: str | None = source_origin.request.request_hash
        external_descriptor_hash: str | None = None
        if (
            source_origin.corrected_attribution_moments.state_hash != moments_hash
            or source_origin.fd_snapshot.snapshot_hash != snapshot_hash
            or source_origin.request.design.design_hash != design_hash
        ):
            raise ValueError("sampled live commit differs from the admitted state")
    elif type(source_origin) is V1SampledRegisteredLookExternalDescriptor:
        source_kind = "external_registered_look_commit"
        source_commit_hash = source_origin.commit_hash
        source_parent_hash = source_origin.external_state.descriptor_hash
        transaction_request_hash = None
        external_descriptor_hash = source_origin.external_state.descriptor_hash
        if (
            source_origin.moments_state_hash != moments_hash
            or source_origin.fd_snapshot_hash != snapshot_hash
            or source_origin.fd_design_hash != design_hash
        ):
            raise ValueError("sampled external commit differs from the admitted state")
    else:
        raise TypeError("sampled source origin has an unsupported exact type")

    receipt = V1SampledDiscoverySourceReceipt(
        source_kind=source_kind,
        source_commit_hash=source_commit_hash,
        source_parent_hash=source_parent_hash,
        transaction_request_hash=transaction_request_hash,
        external_state_descriptor_hash=external_descriptor_hash,
        campaign_contract_hash=contract.campaign_contract_hash,
        campaign_binding_hash=binding.binding_hash,
        model_identity_hash=moments.model_identity_hash,
        query_contract_hash=moments.query_contract_hash,
        preprocessing_hash=moments.preprocessing_hash,
        context_population_hash=contract.context_population_hash,
        sampling_law_hash=contract.sampling_law_hash,
        current_mask_hash=current_mask.mask_hash,
        physical_route_axis_hash=current_mask.route_axis_hash,
        ordered_route_axis_hash=moments.route_axis_hash,
        route_count=moments.route_count,
        proposal_generation=moments.proposal_generation,
        calibration_identity=fd_design.calibration_state_id,
        raw_logit_stream_source_hash=moments.raw_logit_stream_source_hash,
        raw_logit_coordinate=RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE,
        context_mixture_hash=context_mixture,
        stratum_names=moments.stratum_names,
        stratum_weights=fd_design.stratum_weights,  # type: ignore[arg-type]
        population_stratum_counts=moments.population_stratum_counts,
        sealed_population_order_hashes=moments.sealed_population_order_hashes,
        observed_stratum_counts=moments.stratum_counts,
        sampled_look_index=moments.look_index,
        fd_packet_boundary_index=fd_snapshot.packet_boundary_index,
        observed_draw_axis_hash=draw_axis_hash,
        observed_source_context_axis_hash=source_context_axis_hash,
        source_measurement_axis_hash=source_measurement_axis_hash,
        sample_weight_hash=sample_weight_hash,
        corrected_moments_state_hash=moments_hash,
        signed_point_hash=signed_hash,
        absolute_point_hash=absolute_hash,
        corrected_point_estimate_hash=point_hash,
        point_status=SAMPLED_POINT_ESTIMATE_STATUS,
        fd_design_hash=design_hash,
        fd_snapshot_hash=snapshot_hash,
        snapshot_numerical_backend_hash=snapshot_numerics_hash,
        proposal_order_design_hash=score_order_replay.proposal_order_design_hash,
        score_order_registration_hash=score_order_replay.score_order_registration_hash,
        score_order_look_receipt_hash=score_order_replay.score_order_look_receipt_hash,
        score_order_confidence_box_hash=score_order_replay.confidence_box_hash,
        score_order_regret_receipt_axis_hash=regret_axis_hash,
        response_exactification_replay_receipt_hash=replay_hash,
        axis_bridge_receipt_hash=bridge_hash,
        permutation_lineage_hash=permutation_hash,
        terminal_validation_population_hash=contract.terminal_validation_population_hash,
        terminal_validation_policy_hash=contract.terminal_validation_policy_hash,
        validation_firewall_hash=validation_firewall_hash,
    )
    return V1SampledDiscoveryAdmission(
        binding=binding,
        current_mask=current_mask,
        moments=moments,
        fd_snapshot=fd_snapshot,
        fd_design=fd_design,
        axis_bridge=axis_bridge,
        score_order_replay=score_order_replay,
        source_origin=source_origin,
        receipt=receipt,
    )


def admit_sampled_registered_look_commit(
    *,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    commit: V1SampledRegisteredLookCommit,
    axis_bridge: V1OrdinalPhysicalAxisBridgeReceipt,
    score_order_replay: V1ResponseExactificationReplayReceipt,
) -> V1SampledDiscoveryAdmission:
    """Admit one complete in-memory registered-look commit."""

    if type(commit) is not V1SampledRegisteredLookCommit:
        raise TypeError("sampled live admission requires its exact commit type")
    commit.commit_hash
    design = commit.request.design
    if type(design) is not V1StratifiedFDDesign:
        raise TypeError("sampled live commit exposes a noncanonical FD design")
    return _admit_sampled_state(
        binding=binding,
        current_mask=current_mask,
        moments=commit.corrected_attribution_moments,
        fd_snapshot=commit.fd_snapshot,
        fd_design=design,
        axis_bridge=axis_bridge,
        score_order_replay=score_order_replay,
        source_origin=commit,
    )


def admit_sampled_registered_look_external_view(
    *,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    view: V1SampledRegisteredLookExternalView,
    fd_design: V1StratifiedFDDesign,
    axis_bridge: V1OrdinalPhysicalAxisBridgeReceipt,
    score_order_replay: V1ResponseExactificationReplayReceipt,
) -> V1SampledDiscoveryAdmission:
    """Admit one verified externally restored registered-look continuation."""

    if type(view) is not V1SampledRegisteredLookExternalView:
        raise TypeError("sampled external admission requires its exact open-view type")
    moments = view.materialize_moments()
    snapshot = view.materialize_fd_snapshot(design=fd_design)
    return _admit_sampled_state(
        binding=binding,
        current_mask=current_mask,
        moments=moments,
        fd_snapshot=snapshot,
        fd_design=fd_design,
        axis_bridge=axis_bridge,
        score_order_replay=score_order_replay,
        source_origin=view.descriptor,
    )


def assemble_sampled_summary_acquisition_source(
    admission: V1SampledDiscoveryAdmission,
    *,
    device: object = "cpu",
) -> V1SampledSummaryAcquisitionSource:
    """Restore and assemble the sampled stratified FD sufficient state."""

    if type(admission) is not V1SampledDiscoveryAdmission:
        raise TypeError("sampled summary assembly requires an exact admission")
    source = admission.receipt
    before = admission.fd_snapshot.assert_integrity()
    result = V1StratifiedDeviceFDTrustAccumulator.finalize_snapshot_combined_consuming(
        admission.fd_snapshot,
        design=admission.fd_design,
        sketch_rank=admission.fd_snapshot.sketch_rank,
        device=device,
        metadata={
            "sampled_summary_source_receipt_hash": source.receipt_hash,
            "sampled_discovery_admission_hash": admission.admission_hash,
            "modern_v1_campaign_contract_hash": source.campaign_contract_hash,
            "modern_v1_campaign_binding_hash": source.campaign_binding_hash,
            "sampled_context_mixture_hash": source.context_mixture_hash,
            "sampled_proposal_generation": source.proposal_generation,
            "raw_logit_coordinate": source.raw_logit_coordinate,
            "raw_logit_stream_source_hash": source.raw_logit_stream_source_hash,
            "corrected_point_estimate_hash": source.corrected_point_estimate_hash,
            "score_order_regret_receipt_axis_hash": (
                source.score_order_regret_receipt_axis_hash
            ),
            "raw_context_route_rows_retained": False,
            "full_per_draw_source_row_hash_ledger_retained": False,
            "census_authority": False,
            "proposal_only": True,
        },
    )
    # The canonical continuation remains the caller-owned immutable snapshot.
    # Do not emit another full host snapshot merely to inspect the ephemeral
    # restored accumulator: `from_snapshot` already replay-checks restoration,
    # and the immutable caller-owned snapshot remains the continuation
    # authority. Consuming the ephemeral device copy avoids another complete
    # rank-by-route FD state. An extra check here would transfer all live state
    # again at production scale.
    admission.fd_snapshot.assert_integrity(before)
    if result.geometry.derivation_receipt is not None:
        raise RuntimeError(
            "sampled summary manufactured a raw-stream derivation receipt"
        )
    gap_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_standard_acquisition_gap_proof_v1",
            "gap_id": SAMPLED_SUMMARY_STANDARD_ACQUISITION_GAP_ID,
            "source_receipt_hash": source.receipt_hash,
            "fd_snapshot_hash": source.fd_snapshot_hash,
            "geometry_state_hash": result.geometry.content_hash,
            "raw_context_route_rows_retained": False,
            "full_per_draw_source_row_hash_ledger_retained": False,
            "global_trust_derivation_receipt_present": False,
            "standard_fd_stream_acquisition_compatible": False,
            "exact_tangent_price_acquisition_compatible": False,
        },
        prefix=_STANDARD_GAP_PREFIX,
    )
    receipt = V1SampledSummaryGeometryReceipt(
        source_receipt_hash=source.receipt_hash,
        source_admission_hash=admission.admission_hash,
        campaign_contract_hash=source.campaign_contract_hash,
        campaign_binding_hash=source.campaign_binding_hash,
        current_mask_hash=source.current_mask_hash,
        physical_route_axis_hash=source.physical_route_axis_hash,
        ordered_route_axis_hash=source.ordered_route_axis_hash,
        proposal_generation=source.proposal_generation,
        context_mixture_hash=source.context_mixture_hash,
        calibration_identity=source.calibration_identity,
        raw_logit_stream_source_hash=source.raw_logit_stream_source_hash,
        raw_logit_coordinate=source.raw_logit_coordinate,
        corrected_point_estimate_hash=source.corrected_point_estimate_hash,
        score_order_regret_receipt_axis_hash=(
            source.score_order_regret_receipt_axis_hash
        ),
        permutation_lineage_hash=source.permutation_lineage_hash,
        fd_design_hash=source.fd_design_hash,
        fd_snapshot_hash=source.fd_snapshot_hash,
        geometry_state_hash=result.geometry.content_hash,
        stratified_assembly_receipt_hash=result.receipt.receipt_hash,
        stratified_geometry_result_hash=result.result_hash,
        source_measurement_axis_hash=source.source_measurement_axis_hash,
        snapshot_numerical_backend_hash=source.snapshot_numerical_backend_hash,
        assembly_device_identity=result.receipt.device_identity,
        standard_acquisition_gap_proof_hash=gap_hash,
        census_authority=False,
    )
    return V1SampledSummaryAcquisitionSource(
        discovery=admission,
        geometry_result=result,
        receipt=receipt,
    )


# The original sampled-summary adapter above is a post-score-order compatibility
# surface.  Its schemas and hashes intentionally remain unchanged.  The
# following distinct ABI admits the same sufficient numerical state *before*
# an ordinal prefix or response-exactification replay exists, so sampled
# response geometry can participate in proposal composition without acquiring
# irreversible-action authority.
SAMPLED_PRE_FREEZE_GEOMETRY_ADMISSION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_pre_freeze_geometry_admission_receipt_v1"
)
SAMPLED_PRE_FREEZE_GEOMETRY_ADMISSION_SCHEMA: Final = (
    "partition_free_v1_sampled_pre_freeze_geometry_admission_v1"
)
SAMPLED_PRE_FREEZE_GEOMETRY_ASSEMBLY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_pre_freeze_geometry_assembly_receipt_v1"
)
SAMPLED_PRE_FREEZE_GEOMETRY_SOURCE_SCHEMA: Final = (
    "partition_free_v1_sampled_pre_freeze_geometry_source_v1"
)
SAMPLED_PRE_FREEZE_GEOMETRY_AUTHORITY_SCOPE: Final = (
    "pre_freeze_proposal_geometry_input_only_no_irreversible_order_fq_bce_"
    "candidate_mask_production_or_terminal_authority_v1"
)
SAMPLED_PHYSICAL_RANK_PRE_FREEZE_AUTHORIZATION_SCHEMA: Final = (
    "partition_free_v1_sampled_physical_rank_pre_freeze_authorization_v1"
)
SAMPLED_PHYSICAL_RANK_PRE_FREEZE_AUTHORITY_SCOPE: Final = (
    "registered_adaptive_rank_source_replay_for_pre_freeze_geometry_only_no_"
    "rank_stability_C_K_O_commit_or_terminal_authority_v1"
)
SAMPLED_DEVICE_FINAL_RANK_PRE_FREEZE_AUTHORIZATION_SCHEMA: Final = (
    "partition_free_v1_sampled_device_final_rank_pre_freeze_authorization_v1"
)
SAMPLED_DEVICE_FINAL_RANK_PRE_FREEZE_AUTHORITY_SCOPE: Final = (
    "source_native_device_final_rank_for_pre_freeze_geometry_only_no_rank_"
    "stability_C_K_O_commit_or_terminal_authority_v1"
)

_PRE_FREEZE_ADMISSION_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-pre-freeze-geometry-admission-receipt-"
)
_PRE_FREEZE_ADMISSION_PREFIX: Final = (
    "partition-free-v1-sampled-pre-freeze-geometry-admission-"
)
_PRE_FREEZE_ASSEMBLY_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-pre-freeze-geometry-assembly-receipt-"
)
_PRE_FREEZE_SOURCE_PREFIX: Final = (
    "partition-free-v1-sampled-pre-freeze-geometry-source-"
)
_PRE_FREEZE_DERIVATION_ABSENCE_PREFIX: Final = (
    "partition-free-v1-sampled-pre-freeze-raw-row-derivation-absence-"
)
_PHYSICAL_RANK_AUTHORIZATION_PREFIX: Final = (
    "partition-free-v1-sampled-physical-rank-pre-freeze-authorization-"
)
_PHYSICAL_RANK_AUTHORIZATION_TOKEN: Final = object()
_DEVICE_FINAL_RANK_AUTHORIZATION_PREFIX: Final = (
    "partition-free-v1-sampled-device-final-rank-pre-freeze-authorization-"
)
_DEVICE_FINAL_RANK_AUTHORIZATION_TOKEN: Final = object()


@dataclass(frozen=True, slots=True)
class V1SampledPhysicalRankPreFreezeAuthorization:
    """Narrow authority to admit one registered adaptive-rank source replay.

    The campaign contract's ``fd_rank`` is the genesis rank.  Adaptive D may
    replay the identical frozen raw prefix at a larger pre-outcome registered
    rank.  This receipt binds that distinct physical rank without upgrading it
    to rank-stability or C authority.
    """

    campaign_binding_hash: str
    current_mask_hash: str
    physical_route_axis_hash: str
    route_count: int
    proposal_generation: int
    external_commit_hash: str
    external_state_descriptor_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    physical_fd_rank: int
    adaptive_rank_policy_hash: str
    adaptive_rank_uncertainty_set_hash: str
    admitted_rank_ladder: tuple[int, ...]
    physical_rank_replay_receipt_hash: str
    rank_invariant_raw_prefix_hash: str
    _issuer_token: InitVar[object]
    registered_before_D_outcomes: bool = True
    empirical_outcomes_used_for_rank_filtering: bool = False
    source_replay_only: bool = True
    adaptive_rank_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = SAMPLED_PHYSICAL_RANK_PRE_FREEZE_AUTHORITY_SCOPE
    schema: str = SAMPLED_PHYSICAL_RANK_PRE_FREEZE_AUTHORIZATION_SCHEMA
    _authorization_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _PHYSICAL_RANK_AUTHORIZATION_TOKEN:
            raise TypeError("physical-rank pre-freeze authorizations are issuer-only")
        for name in (
            "campaign_binding_hash",
            "current_mask_hash",
            "physical_route_axis_hash",
            "external_commit_hash",
            "external_state_descriptor_hash",
            "fd_design_hash",
            "fd_snapshot_hash",
            "adaptive_rank_policy_hash",
            "adaptive_rank_uncertainty_set_hash",
            "physical_rank_replay_receipt_hash",
            "rank_invariant_raw_prefix_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        rank = _positive_int(self.physical_fd_rank, name="physical FD rank")
        ladder = tuple(
            _positive_int(value, name="admitted FD rank")
            for value in self.admitted_rank_ladder
        )
        if not ladder or tuple(sorted(set(ladder))) != ladder or rank not in ladder:
            raise ValueError(
                "physical-rank authorization does not bind one admitted rank ladder"
            )
        flags = (
            self.registered_before_D_outcomes,
            self.empirical_outcomes_used_for_rank_filtering,
            self.source_replay_only,
            self.adaptive_rank_stability_authority,
            self.C_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
        )
        if any(type(value) is not bool for value in flags) or flags != (
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            raise PermissionError(
                "physical-rank pre-freeze authorization crossed its source boundary"
            )
        if (
            self.authority_scope != SAMPLED_PHYSICAL_RANK_PRE_FREEZE_AUTHORITY_SCOPE
            or self.schema != SAMPLED_PHYSICAL_RANK_PRE_FREEZE_AUTHORIZATION_SCHEMA
        ):
            raise ValueError("physical-rank pre-freeze authorization schema changed")
        object.__setattr__(self, "physical_fd_rank", rank)
        object.__setattr__(self, "admitted_rank_ladder", ladder)
        object.__setattr__(self, "_authorization_hash", self.authorization_hash)

    @property
    def authorization_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PHYSICAL_RANK_AUTHORIZATION_PREFIX,
        )

    def assert_integrity(self) -> str:
        observed = self.authorization_hash
        if observed != self._authorization_hash:
            raise ValueError("physical-rank pre-freeze authorization was tampered")
        return observed


def authorize_sampled_physical_rank_replay_pre_freeze(
    *,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    source_origin: V1SampledRegisteredLookExternalDescriptor,
    fd_design: V1StratifiedFDDesign,
    adaptive_rank_policy: V1FQRAdaptiveRankPolicy,
    physical_rank_replay_receipt_hash: str,
    rank_invariant_raw_prefix_hash: str,
) -> V1SampledPhysicalRankPreFreezeAuthorization:
    """Bind one source-only physical-rank replay to the registered D policy."""

    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("physical-rank authorization requires an exact binding")
    if type(current_mask) is not HardMask:
        raise TypeError("physical-rank authorization requires an exact mask")
    if type(source_origin) is not V1SampledRegisteredLookExternalDescriptor:
        raise TypeError(
            "physical-rank authorization requires an exact external D source"
        )
    if type(fd_design) is not V1StratifiedFDDesign:
        raise TypeError("physical-rank authorization requires an exact FD design")
    if type(adaptive_rank_policy) is not V1FQRAdaptiveRankPolicy:
        raise TypeError("physical-rank authorization requires an exact rank policy")
    policy_hash = adaptive_rank_policy.assert_integrity()
    design_hash = fd_design.assert_integrity()
    snapshot_hash = source_origin.fd_snapshot_hash
    rank = fd_design.spec.initial_rank
    contract = binding.contract
    if (
        adaptive_rank_policy.route_count != current_mask.route_count
        or not adaptive_rank_policy.requested_rank_ladder
        or adaptive_rank_policy.requested_rank_ladder[0] != contract.fd_rank
        or rank not in adaptive_rank_policy.admitted_rank_ladder
        or fd_design.spec.maximum_rank != rank
        or source_origin.fd_design_hash != design_hash
        or source_origin.route_count != current_mask.route_count
        or source_origin.route_axis_hash != fd_design.route_axis_hash
        or fd_design.current_mask_hash != current_mask.mask_hash
        or fd_design.measured_mask_hash != current_mask.mask_hash
        or fd_design.campaign_contract_hash != contract.campaign_contract_hash
        or fd_design.proposal_generation != source_origin.proposal_generation
        or adaptive_rank_policy.registered_before_D_outcomes is not True
        or adaptive_rank_policy.empirical_outcomes_used_for_filtering is not False
    ):
        raise ValueError(
            "physical-rank source does not replay the registered adaptive D policy"
        )
    return V1SampledPhysicalRankPreFreezeAuthorization(
        campaign_binding_hash=binding.binding_hash,
        current_mask_hash=current_mask.mask_hash,
        physical_route_axis_hash=current_mask.route_axis_hash,
        route_count=current_mask.route_count,
        proposal_generation=source_origin.proposal_generation,
        external_commit_hash=source_origin.commit_hash,
        external_state_descriptor_hash=source_origin.external_state.descriptor_hash,
        fd_design_hash=design_hash,
        fd_snapshot_hash=snapshot_hash,
        physical_fd_rank=rank,
        adaptive_rank_policy_hash=policy_hash,
        adaptive_rank_uncertainty_set_hash=adaptive_rank_policy.uncertainty_set_hash,
        admitted_rank_ladder=adaptive_rank_policy.admitted_rank_ladder,
        physical_rank_replay_receipt_hash=physical_rank_replay_receipt_hash,
        rank_invariant_raw_prefix_hash=rank_invariant_raw_prefix_hash,
        _issuer_token=_PHYSICAL_RANK_AUTHORIZATION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledDeviceFinalRankPreFreezeAuthorization:
    """Pre-freeze authority for one exact source-native final-D rank.

    This is deliberately disjoint from
    :class:`V1SampledPhysicalRankPreFreezeAuthorization`: the latter names a
    host-restored physical-rank replay, while this receipt names the single
    portable endpoint emitted by the device-resident D session.  Neither
    receipt certifies adaptive-rank stability or authorizes C/K/O.
    """

    campaign_binding_hash: str
    current_mask_hash: str
    physical_route_axis_hash: str
    route_count: int
    proposal_generation: int
    external_commit_hash: str
    external_state_descriptor_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    corrected_moments_state_hash: str
    physical_fd_rank: int
    adaptive_rank_policy_hash: str
    adaptive_rank_uncertainty_set_hash: str
    admitted_rank_ladder: tuple[int, ...]
    device_final_rank_source_receipt_hash: str
    device_materialization_hash: str
    materialization_rank_receipt_hash: str
    registered_d_transaction_hash: str
    registered_d_prefix_binding_hash: str
    rank_invariant_raw_prefix_hash: str
    _issuer_token: InitVar[object]
    rank_policy_registered_before_D_outcomes: bool = True
    source_materialized_after_registered_D_prefix: bool = True
    empirical_outcomes_used_for_rank_filtering: bool = False
    source_native_device_materialization: bool = True
    adaptive_rank_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = SAMPLED_DEVICE_FINAL_RANK_PRE_FREEZE_AUTHORITY_SCOPE
    schema: str = SAMPLED_DEVICE_FINAL_RANK_PRE_FREEZE_AUTHORIZATION_SCHEMA
    _authorization_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _DEVICE_FINAL_RANK_AUTHORIZATION_TOKEN:
            raise TypeError(
                "device-final-rank pre-freeze authorizations are issuer-only"
            )
        for name in (
            "campaign_binding_hash",
            "current_mask_hash",
            "physical_route_axis_hash",
            "external_commit_hash",
            "external_state_descriptor_hash",
            "fd_design_hash",
            "fd_snapshot_hash",
            "corrected_moments_state_hash",
            "adaptive_rank_policy_hash",
            "adaptive_rank_uncertainty_set_hash",
            "device_final_rank_source_receipt_hash",
            "device_materialization_hash",
            "materialization_rank_receipt_hash",
            "registered_d_transaction_hash",
            "registered_d_prefix_binding_hash",
            "rank_invariant_raw_prefix_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        rank = _positive_int(self.physical_fd_rank, name="physical FD rank")
        ladder = tuple(
            _positive_int(value, name="admitted FD rank")
            for value in self.admitted_rank_ladder
        )
        if not ladder or tuple(sorted(set(ladder))) != ladder or rank not in ladder:
            raise ValueError(
                "device-final-rank authorization does not bind one admitted rank ladder"
            )
        flags = (
            self.rank_policy_registered_before_D_outcomes,
            self.source_materialized_after_registered_D_prefix,
            self.empirical_outcomes_used_for_rank_filtering,
            self.source_native_device_materialization,
            self.adaptive_rank_stability_authority,
            self.C_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
        )
        if any(type(value) is not bool for value in flags) or flags != (
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            raise PermissionError(
                "device-final-rank pre-freeze authorization crossed its source boundary"
            )
        if (
            self.authority_scope
            != SAMPLED_DEVICE_FINAL_RANK_PRE_FREEZE_AUTHORITY_SCOPE
            or self.schema
            != SAMPLED_DEVICE_FINAL_RANK_PRE_FREEZE_AUTHORIZATION_SCHEMA
        ):
            raise ValueError(
                "device-final-rank pre-freeze authorization schema changed"
            )
        object.__setattr__(self, "physical_fd_rank", rank)
        object.__setattr__(self, "admitted_rank_ladder", ladder)
        object.__setattr__(self, "_authorization_hash", self.authorization_hash)

    @property
    def authorization_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_DEVICE_FINAL_RANK_AUTHORIZATION_PREFIX,
        )

    def assert_integrity(self) -> str:
        observed = self.authorization_hash
        if observed != self._authorization_hash:
            raise ValueError(
                "device-final-rank pre-freeze authorization was tampered"
            )
        return observed


def authorize_sampled_device_final_rank_source_pre_freeze(
    *,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    source_origin: V1SampledRegisteredLookExternalDescriptor,
    fd_design: V1StratifiedFDDesign,
    device_final_rank_source: V1SampledDeviceFinalRankSourceReceipt,
    adaptive_rank_policy: V1FQRAdaptiveRankPolicy,
) -> V1SampledDeviceFinalRankPreFreezeAuthorization:
    """Bind one exact source-native final-D endpoint to pre-freeze geometry."""

    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("device-final-rank authorization requires an exact binding")
    if type(current_mask) is not HardMask:
        raise TypeError("device-final-rank authorization requires an exact mask")
    if type(source_origin) is not V1SampledRegisteredLookExternalDescriptor:
        raise TypeError(
            "device-final-rank authorization requires an exact external D source"
        )
    if type(fd_design) is not V1StratifiedFDDesign:
        raise TypeError("device-final-rank authorization requires an exact FD design")
    if type(device_final_rank_source) is not V1SampledDeviceFinalRankSourceReceipt:
        raise TypeError(
            "device-final-rank authorization requires its exact source receipt"
        )
    if type(adaptive_rank_policy) is not V1FQRAdaptiveRankPolicy:
        raise TypeError("device-final-rank authorization requires an exact rank policy")
    source_hash = device_final_rank_source.assert_integrity()
    policy_hash = adaptive_rank_policy.assert_integrity()
    design_hash = fd_design.assert_integrity()
    contract = binding.contract
    rank = device_final_rank_source.requested_physical_rank
    closure = {
        "policy_route_count": (
            adaptive_rank_policy.route_count == current_mask.route_count
        ),
        "requested_rank_ladder": bool(adaptive_rank_policy.requested_rank_ladder),
        "genesis_rank": (
            adaptive_rank_policy.requested_rank_ladder[0] == contract.fd_rank
        ),
        "rank_admitted": rank in adaptive_rank_policy.admitted_rank_ladder,
        "policy_preoutcome": (
            adaptive_rank_policy.registered_before_D_outcomes is True
            and adaptive_rank_policy.empirical_outcomes_used_for_filtering is False
        ),
        "design_rank": (
            fd_design.spec.initial_rank == rank
            and fd_design.spec.maximum_rank == rank
        ),
        "design_hash": (
            fd_design.design_hash == device_final_rank_source.fd_design_hash
            == source_origin.fd_design_hash
        ),
        "snapshot_hash": (
            source_origin.fd_snapshot_hash
            == device_final_rank_source.fd_snapshot_hash
        ),
        "moments_hash": (
            source_origin.moments_state_hash
            == device_final_rank_source.corrected_moments_state_hash
        ),
        "proposal_generation": (
            source_origin.proposal_generation
            == device_final_rank_source.proposal_generation
            == fd_design.proposal_generation
        ),
        "route_count": (
            source_origin.route_count
            == device_final_rank_source.route_count
            == current_mask.route_count
        ),
        # The external FD descriptor is ordered-route-axis native.  The
        # device-final receipt and mask are physical-route-axis native.
        "ordered_route_axis": (
            source_origin.route_axis_hash == fd_design.route_axis_hash
        ),
        "physical_route_axis": (
            current_mask.route_axis_hash == device_final_rank_source.route_axis_hash
        ),
        "registered_look": (
            source_origin.look_index
            == device_final_rank_source.registered_look_count
        ),
        "packet_boundary": (
            source_origin.packet_boundary_index
            == device_final_rank_source.packet_count
        ),
        "current_mask": (
            current_mask.mask_hash == device_final_rank_source.current_mask_hash
            == fd_design.current_mask_hash
            == fd_design.measured_mask_hash
        ),
        "campaign": (
            fd_design.campaign_contract_hash == contract.campaign_contract_hash
        ),
        "literal_j_stream": (
            source_origin.literal_jacobian_stream_identity_hash
            == device_final_rank_source.literal_j_stream_identity_hash
        ),
        "corrected_a_stream": (
            source_origin.corrected_evidence_stream_identity_hash
            == device_final_rank_source.corrected_a_stream_identity_hash
        ),
        "source_native": (
            device_final_rank_source.source_native_device_materialization is True
            and device_final_rank_source.host_custody_replay is False
        ),
        "authority_inert": (
            device_final_rank_source.rank_stability_authority is False
            and device_final_rank_source.d_decision_authority is False
        ),
    }
    if not all(closure.values()):
        raise ValueError(
            "device-final-rank source does not close the registered adaptive D "
            "endpoint; failed="
            f"{tuple(name for name, closed in closure.items() if not closed)}"
        )
    return V1SampledDeviceFinalRankPreFreezeAuthorization(
        campaign_binding_hash=binding.binding_hash,
        current_mask_hash=current_mask.mask_hash,
        physical_route_axis_hash=current_mask.route_axis_hash,
        route_count=current_mask.route_count,
        proposal_generation=device_final_rank_source.proposal_generation,
        external_commit_hash=source_origin.commit_hash,
        external_state_descriptor_hash=source_origin.external_state.descriptor_hash,
        fd_design_hash=design_hash,
        fd_snapshot_hash=device_final_rank_source.fd_snapshot_hash,
        corrected_moments_state_hash=(
            device_final_rank_source.corrected_moments_state_hash
        ),
        physical_fd_rank=rank,
        adaptive_rank_policy_hash=policy_hash,
        adaptive_rank_uncertainty_set_hash=adaptive_rank_policy.uncertainty_set_hash,
        admitted_rank_ladder=adaptive_rank_policy.admitted_rank_ladder,
        device_final_rank_source_receipt_hash=source_hash,
        device_materialization_hash=device_final_rank_source.device_materialization_hash,
        materialization_rank_receipt_hash=(
            device_final_rank_source.materialization_rank_receipt_hash
        ),
        registered_d_transaction_hash=(
            device_final_rank_source.registered_d_transaction_hash
        ),
        registered_d_prefix_binding_hash=(
            device_final_rank_source.registered_d_prefix_binding_hash
        ),
        rank_invariant_raw_prefix_hash=(
            device_final_rank_source.rank_invariant_raw_prefix_hash
        ),
        _issuer_token=_DEVICE_FINAL_RANK_AUTHORIZATION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledPreFreezeGeometryAdmissionReceipt:
    """Closed identity graph for pre-freeze sampled numerical geometry.

    This receipt deliberately contains neither an ordinal/physical axis bridge
    nor a response-exactification replay identity.  Distinct physical and
    ordered axes are bound, but this proposal-only object cannot apply an
    ordered action to the physical mask.
    """

    source_kind: SourceKind
    source_commit_hash: str
    source_parent_hash: str
    transaction_request_hash: str | None
    external_state_descriptor_hash: str | None
    campaign_contract_hash: str
    campaign_binding_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    context_population_hash: str
    sampling_law_hash: str
    current_mask_hash: str
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    route_count: int
    proposal_generation: int
    calibration_identity: str
    raw_logit_stream_source_hash: str
    raw_logit_coordinate: str
    context_mixture_hash: str
    stratum_names: tuple[str, str]
    stratum_weights: tuple[float, float]
    population_stratum_counts: tuple[int, int]
    sealed_population_order_hashes: tuple[str, str]
    observed_stratum_counts: tuple[int, int]
    sampled_look_index: int
    fd_packet_boundary_index: int
    observed_draw_axis_hash: str
    observed_source_context_axis_hash: str
    source_measurement_axis_hash: str
    sample_weight_hash: str
    corrected_moments_state_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    physical_fd_rank: int
    physical_rank_authorization_hash: str | None
    snapshot_numerical_backend_hash: str
    terminal_validation_population_hash: str
    terminal_validation_policy_hash: str
    validation_firewall_hash: str
    proposal_geometry_input_authority: bool = True
    axis_bridge_dependency: bool = False
    response_exactification_replay_dependency: bool = False
    ordered_to_physical_application_authority: bool = False
    raw_context_route_rows_retained: bool = False
    full_per_draw_source_row_hash_ledger_retained: bool = False
    raw_row_derivation_receipt_issued: bool = False
    proposal_only: bool = True
    irreversible_order_authority: bool = False
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    production_execution_authority: bool = False
    terminal_authority: bool = False
    validation_inputs_present: bool = False
    authority_scope: str = SAMPLED_PRE_FREEZE_GEOMETRY_AUTHORITY_SCOPE
    schema: str = SAMPLED_PRE_FREEZE_GEOMETRY_ADMISSION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "source_commit_hash",
            "source_parent_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "context_population_hash",
            "sampling_law_hash",
            "current_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "calibration_identity",
            "raw_logit_stream_source_hash",
            "context_mixture_hash",
            "observed_draw_axis_hash",
            "observed_source_context_axis_hash",
            "source_measurement_axis_hash",
            "sample_weight_hash",
            "corrected_moments_state_hash",
            "fd_design_hash",
            "fd_snapshot_hash",
            "snapshot_numerical_backend_hash",
            "terminal_validation_population_hash",
            "terminal_validation_policy_hash",
            "validation_firewall_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.transaction_request_hash is not None:
            object.__setattr__(
                self,
                "transaction_request_hash",
                _identity(
                    self.transaction_request_hash,
                    name="transaction request hash",
                ),
            )
        if self.external_state_descriptor_hash is not None:
            object.__setattr__(
                self,
                "external_state_descriptor_hash",
                _identity(
                    self.external_state_descriptor_hash,
                    name="external state descriptor hash",
                ),
            )
        if self.physical_rank_authorization_hash is not None:
            object.__setattr__(
                self,
                "physical_rank_authorization_hash",
                _identity(
                    self.physical_rank_authorization_hash,
                    name="physical rank authorization hash",
                ),
            )
        if self.source_kind == "registered_look_commit":
            if (
                self.transaction_request_hash is None
                or self.external_state_descriptor_hash is not None
            ):
                raise ValueError(
                    "pre-freeze live sampled source has an invalid origin lineage"
                )
        elif self.source_kind == "external_registered_look_commit":
            if (
                self.external_state_descriptor_hash is None
                or self.transaction_request_hash is not None
            ):
                raise ValueError(
                    "pre-freeze external sampled source has an invalid origin lineage"
                )
        else:
            raise ValueError("unsupported pre-freeze sampled source kind")
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        object.__setattr__(
            self,
            "physical_fd_rank",
            _positive_int(self.physical_fd_rank, name="physical FD rank"),
        )
        object.__setattr__(
            self,
            "sampled_look_index",
            _positive_int(self.sampled_look_index, name="sampled look index"),
        )
        object.__setattr__(
            self,
            "fd_packet_boundary_index",
            _positive_int(
                self.fd_packet_boundary_index,
                name="FD packet boundary index",
            ),
        )
        names = _pair_of_identities(self.stratum_names, name="stratum names")
        weights = _pair_of_class_balanced_weights(
            self.stratum_weights,
            name="stratum weights",
        )
        populations = _pair_of_positive_ints(
            self.population_stratum_counts,
            name="population stratum counts",
        )
        counts = _pair_of_nonnegative_ints(
            self.observed_stratum_counts,
            name="observed stratum counts",
        )
        orders = _pair_of_identities(
            self.sealed_population_order_hashes,
            name="sealed population order hashes",
        )
        if any(
            count <= 0 or count > population
            for count, population in zip(counts, populations)
        ):
            raise ValueError(
                "pre-freeze sampled stratum counts are incomplete or overrun"
            )
        if self.raw_logit_coordinate != RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE:
            raise ValueError(
                "pre-freeze sampled geometry requires the raw-target-logit coordinate"
            )
        expected_mixture = _context_mixture_hash(
            campaign_contract_hash=self.campaign_contract_hash,
            context_population_hash=self.context_population_hash,
            sampling_law_hash=self.sampling_law_hash,
            raw_logit_stream_source_hash=self.raw_logit_stream_source_hash,
            current_mask_hash=self.current_mask_hash,
            proposal_generation=self.proposal_generation,
            stratum_names=names,
            stratum_weights=weights,  # type: ignore[arg-type]
            population_stratum_counts=populations,
            sealed_population_order_hashes=orders,
        )
        if self.context_mixture_hash != expected_mixture:
            raise ValueError("pre-freeze sampled context-mixture hash does not replay")
        if (
            self.proposal_geometry_input_authority is not True
            or self.axis_bridge_dependency is not False
            or self.response_exactification_replay_dependency is not False
            or self.ordered_to_physical_application_authority is not False
            or self.raw_context_route_rows_retained is not False
            or self.full_per_draw_source_row_hash_ledger_retained is not False
            or self.raw_row_derivation_receipt_issued is not False
            or self.proposal_only is not True
            or self.irreversible_order_authority is not False
            or self.fq_certification_authority is not False
            or self.bce_oracle_authority is not False
            or self.candidate_freeze_authority is not False
            or self.mask_commit_authority is not False
            or self.production_execution_authority is not False
            or self.terminal_authority is not False
            or self.validation_inputs_present is not False
            or self.authority_scope != SAMPLED_PRE_FREEZE_GEOMETRY_AUTHORITY_SCOPE
            or self.schema != SAMPLED_PRE_FREEZE_GEOMETRY_ADMISSION_RECEIPT_SCHEMA
        ):
            raise PermissionError(
                "pre-freeze sampled geometry admission crossed its authority boundary"
            )
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "observed_stratum_counts", counts)
        object.__setattr__(self, "sealed_population_order_hashes", orders)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PRE_FREEZE_ADMISSION_RECEIPT_PREFIX)


def _build_pre_freeze_geometry_admission_receipt(
    *,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    moments: V1StratifiedCorrectedAttributionMoments,
    fd_snapshot: V1StratifiedFDAccumulatorSnapshot,
    fd_design: V1StratifiedFDDesign,
    source_origin: (
        V1SampledRegisteredLookCommit | V1SampledRegisteredLookExternalDescriptor
    ),
    physical_rank_authorization: (
        V1SampledPhysicalRankPreFreezeAuthorization
        | V1SampledDeviceFinalRankPreFreezeAuthorization
        | None
    ),
) -> V1SampledPreFreezeGeometryAdmissionReceipt:
    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("pre-freeze geometry requires an exact campaign binding")
    if type(current_mask) is not HardMask:
        raise TypeError("pre-freeze geometry requires an exact current mask")
    if type(moments) is not V1StratifiedCorrectedAttributionMoments:
        raise TypeError("pre-freeze geometry requires exact corrected moments")
    if type(fd_snapshot) is not V1StratifiedFDAccumulatorSnapshot:
        raise TypeError("pre-freeze geometry requires an exact FD snapshot")
    if type(fd_design) is not V1StratifiedFDDesign:
        raise TypeError("pre-freeze geometry requires an exact FD design")

    contract = binding.contract
    mask_hash = _direct_mask_hash(current_mask)
    moments_hash = moments.assert_integrity()
    snapshot_hash = fd_snapshot.assert_integrity()
    design_hash = fd_design.assert_integrity()
    physical_rank = fd_snapshot.sketch_rank
    if mask_hash != current_mask.mask_hash:
        raise ValueError("pre-freeze geometry current mask is not canonical")
    if contract.context_population_hash == contract.terminal_validation_population_hash:
        raise PermissionError(
            "proposal and terminal-validation populations are not separated"
        )
    shared = (
        (moments.campaign_contract_hash, fd_design.campaign_contract_hash),
        (moments.model_identity_hash, fd_design.model_identity_hash),
        (moments.query_contract_hash, fd_design.query_contract_hash),
        (moments.preprocessing_hash, fd_design.preprocessing_hash),
        (moments.current_mask_hash, fd_design.current_mask_hash),
        (moments.current_mask_hash, fd_design.measured_mask_hash),
        (moments.sampling_law_hash, fd_design.sampling_law_hash),
        (moments.proposal_generation, fd_design.proposal_generation),
        (moments.route_count, len(fd_design.route_ids)),
        (moments.route_axis_hash, fd_design.route_axis_hash),
        (moments.stratum_names, fd_design.stratum_names),
    )
    if any(left != right for left, right in shared):
        raise ValueError(
            "pre-freeze moments and FD state name different acquisition coordinates"
        )
    _assert_sampled_dual_or_legacy_stream_coordinates(
        moments=moments,
        fd_design=fd_design,
        source_origin=source_origin,
    )
    rank_consistent = (
        fd_design.spec.initial_rank == physical_rank
        and fd_design.spec.maximum_rank == physical_rank
    )
    if physical_rank_authorization is None:
        physical_rank_authorization_hash = None
        physical_rank_admitted = physical_rank == contract.fd_rank
    elif type(physical_rank_authorization) is (
        V1SampledPhysicalRankPreFreezeAuthorization
    ):
        physical_rank_authorization_hash = (
            physical_rank_authorization.assert_integrity()
        )
        if type(source_origin) is not V1SampledRegisteredLookExternalDescriptor:
            raise PermissionError(
                "physical-rank replay authorization is external-source only"
            )
        expected_rank_authorization = (
            binding.binding_hash,
            current_mask.mask_hash,
            current_mask.route_axis_hash,
            current_mask.route_count,
            moments.proposal_generation,
            source_origin.commit_hash,
            source_origin.external_state.descriptor_hash,
            design_hash,
            snapshot_hash,
            physical_rank,
        )
        observed_rank_authorization = (
            physical_rank_authorization.campaign_binding_hash,
            physical_rank_authorization.current_mask_hash,
            physical_rank_authorization.physical_route_axis_hash,
            physical_rank_authorization.route_count,
            physical_rank_authorization.proposal_generation,
            physical_rank_authorization.external_commit_hash,
            physical_rank_authorization.external_state_descriptor_hash,
            physical_rank_authorization.fd_design_hash,
            physical_rank_authorization.fd_snapshot_hash,
            physical_rank_authorization.physical_fd_rank,
        )
        if observed_rank_authorization != expected_rank_authorization:
            raise ValueError(
                "pre-freeze physical-rank authorization names another source"
            )
        physical_rank_admitted = True
    elif type(physical_rank_authorization) is (
        V1SampledDeviceFinalRankPreFreezeAuthorization
    ):
        physical_rank_authorization_hash = (
            physical_rank_authorization.assert_integrity()
        )
        if type(source_origin) is not V1SampledRegisteredLookExternalDescriptor:
            raise PermissionError(
                "device-final-rank authorization is external-source only"
            )
        expected_rank_authorization = (
            binding.binding_hash,
            current_mask.mask_hash,
            current_mask.route_axis_hash,
            current_mask.route_count,
            moments.proposal_generation,
            source_origin.commit_hash,
            source_origin.external_state.descriptor_hash,
            design_hash,
            snapshot_hash,
            moments_hash,
            physical_rank,
        )
        observed_rank_authorization = (
            physical_rank_authorization.campaign_binding_hash,
            physical_rank_authorization.current_mask_hash,
            physical_rank_authorization.physical_route_axis_hash,
            physical_rank_authorization.route_count,
            physical_rank_authorization.proposal_generation,
            physical_rank_authorization.external_commit_hash,
            physical_rank_authorization.external_state_descriptor_hash,
            physical_rank_authorization.fd_design_hash,
            physical_rank_authorization.fd_snapshot_hash,
            physical_rank_authorization.corrected_moments_state_hash,
            physical_rank_authorization.physical_fd_rank,
        )
        if observed_rank_authorization != expected_rank_authorization:
            raise ValueError(
                "pre-freeze device-final-rank authorization names another source"
            )
        physical_rank_admitted = True
    else:
        if physical_rank_authorization is not None:
            raise TypeError(
                "pre-freeze physical-rank authorization has the wrong exact type"
            )
    pre_freeze_mismatches = tuple(
        name
        for name, differs in (
            (
                "campaign_contract_hash",
                moments.campaign_contract_hash != contract.campaign_contract_hash,
            ),
            (
                "model_identity_hash",
                moments.model_identity_hash != contract.model_seal_hash,
            ),
            (
                "sampling_law_hash",
                moments.sampling_law_hash != contract.sampling_law_hash,
            ),
            ("current_mask_hash", moments.current_mask_hash != current_mask.mask_hash),
            (
                "calibration_state_id",
                fd_design.calibration_state_id != contract.calibration_identity,
            ),
            ("physical_rank_design", not rank_consistent),
            ("physical_rank_authorization", not physical_rank_admitted),
            (
                "physical_route_axis_hash",
                current_mask.route_axis_hash != contract.route_axis_hash,
            ),
            ("route_count", current_mask.route_count != moments.route_count),
            ("fd_snapshot_design_hash", fd_snapshot.design_hash != design_hash),
            (
                "fd_snapshot_route_axis_hash",
                fd_snapshot.route_axis_hash != fd_design.route_axis_hash,
            ),
            (
                "fd_snapshot_stratum_names",
                fd_snapshot.design_stratum_names != fd_design.stratum_names,
            ),
            (
                "fd_snapshot_stratum_counts",
                tuple(
                    child.context_count
                    for _name, child in fd_snapshot.stratum_snapshots
                )
                != moments.stratum_counts,
            ),
        )
        if differs
    )
    if pre_freeze_mismatches:
        raise ValueError(
            "pre-freeze sampled source differs from the current campaign or mask: "
            + ",".join(pre_freeze_mismatches)
        )
    if tuple(fd_design.stratum_weights) != (0.5, 0.5):
        raise ValueError(
            "pre-freeze sampled source must use the class-balanced target mixture"
        )

    snapshot_contexts = tuple(
        (name, tuple(child.context_ids))
        for name, child in fd_snapshot.stratum_snapshots
    )
    observed_contexts = tuple(
        (name, tuple(moments.observed_context_ids_by_stratum[index]))
        for index, name in enumerate(moments.stratum_names)
    )
    if snapshot_contexts != observed_contexts:
        raise ValueError(
            "pre-freeze FD and corrected moments observed different contexts"
        )
    source_measurements = fd_snapshot.source_measurement_hashes_by_stratum
    if any(
        len(values) != moments.stratum_counts[index]
        for index, (_name, values) in enumerate(source_measurements)
    ):
        raise ValueError("pre-freeze source-measurement ledger is incomplete")

    draw_axis_hash = content_hash(
        moments.observed_draw_ids_by_stratum,
        prefix=_DRAW_AXIS_PREFIX,
    )
    source_context_axis_hash = content_hash(
        moments.observed_context_ids_by_stratum,
        prefix=_SOURCE_CONTEXT_AXIS_PREFIX,
    )
    source_measurement_axis_hash = content_hash(
        source_measurements,
        prefix=_SOURCE_MEASUREMENT_AXIS_PREFIX,
    )
    sample_weight_hash = content_hash(
        {
            "stratum_weights": fd_design.stratum_weights,
            "observed_draw_ids_by_stratum": moments.observed_draw_ids_by_stratum,
            "observed_stratum_counts": moments.stratum_counts,
            "weighting": "equal_stratum_mean_then_equal_macro_average_v1",
        },
        prefix=_SAMPLE_WEIGHT_PREFIX,
    )
    context_mixture = _context_mixture_hash(
        campaign_contract_hash=contract.campaign_contract_hash,
        context_population_hash=contract.context_population_hash,
        sampling_law_hash=contract.sampling_law_hash,
        raw_logit_stream_source_hash=moments.raw_logit_stream_source_hash,
        current_mask_hash=current_mask.mask_hash,
        proposal_generation=moments.proposal_generation,
        stratum_names=moments.stratum_names,
        stratum_weights=fd_design.stratum_weights,  # type: ignore[arg-type]
        population_stratum_counts=moments.population_stratum_counts,
        sealed_population_order_hashes=moments.sealed_population_order_hashes,
    )
    snapshot_numerics_hash = content_hash(
        {
            "fd_design_hash": design_hash,
            "fd_snapshot_hash": snapshot_hash,
            "child_numerics": tuple(
                (
                    name,
                    child.producer_algorithm_id,
                    child.producer_implementation,
                    child.dtype_identity,
                    child.sketch_rank,
                    child.snapshot_hash,
                )
                for name, child in fd_snapshot.stratum_snapshots
            ),
            "raw_context_route_rows_retained": False,
        },
        prefix=_SNAPSHOT_NUMERICS_PREFIX,
    )
    validation_firewall_hash = content_hash(
        {
            "campaign_contract_hash": contract.campaign_contract_hash,
            "proposal_context_population_hash": contract.context_population_hash,
            "terminal_validation_population_hash": (
                contract.terminal_validation_population_hash
            ),
            "terminal_validation_policy_hash": (
                contract.terminal_validation_policy_hash
            ),
            "populations_are_distinct": True,
            "adapter_input_types_contain_validation_outcomes": False,
            "validation_inputs_present": False,
            "terminal_authority": False,
        },
        prefix=_VALIDATION_FIREWALL_PREFIX,
    )

    if type(source_origin) is V1SampledRegisteredLookCommit:
        source_commit_hash = source_origin.commit_hash
        request = source_origin.request
        if (
            source_origin.corrected_attribution_moments.state_hash != moments_hash
            or source_origin.fd_snapshot.snapshot_hash != snapshot_hash
            or request.design.design_hash != design_hash
            or request.registered_look_index != moments.look_index
            or tuple(request.registered_look_cumulative_stratum_counts)
            != moments.stratum_counts
        ):
            raise ValueError(
                "pre-freeze live commit differs from the admitted sampled state"
            )
        source_kind: SourceKind = "registered_look_commit"
        source_parent_hash = request.request_hash
        transaction_request_hash: str | None = request.request_hash
        external_descriptor_hash: str | None = None
    elif type(source_origin) is V1SampledRegisteredLookExternalDescriptor:
        if (
            source_origin.moments_state_hash != moments_hash
            or source_origin.fd_snapshot_hash != snapshot_hash
            or source_origin.fd_design_hash != design_hash
            or source_origin.proposal_generation != moments.proposal_generation
            or source_origin.look_index != moments.look_index
            or source_origin.packet_boundary_index != fd_snapshot.packet_boundary_index
            or source_origin.route_count != moments.route_count
            or source_origin.route_axis_hash != moments.route_axis_hash
            or source_origin.stratum_names != moments.stratum_names
        ):
            raise ValueError(
                "pre-freeze external commit differs from the admitted sampled state"
            )
        source_kind = "external_registered_look_commit"
        source_commit_hash = source_origin.commit_hash
        source_parent_hash = source_origin.external_state.descriptor_hash
        transaction_request_hash = None
        external_descriptor_hash = source_origin.external_state.descriptor_hash
    else:
        raise TypeError("pre-freeze sampled source has an unsupported exact type")

    return V1SampledPreFreezeGeometryAdmissionReceipt(
        source_kind=source_kind,
        source_commit_hash=source_commit_hash,
        source_parent_hash=source_parent_hash,
        transaction_request_hash=transaction_request_hash,
        external_state_descriptor_hash=external_descriptor_hash,
        campaign_contract_hash=contract.campaign_contract_hash,
        campaign_binding_hash=binding.binding_hash,
        model_identity_hash=moments.model_identity_hash,
        query_contract_hash=moments.query_contract_hash,
        preprocessing_hash=moments.preprocessing_hash,
        context_population_hash=contract.context_population_hash,
        sampling_law_hash=contract.sampling_law_hash,
        current_mask_hash=current_mask.mask_hash,
        physical_route_axis_hash=current_mask.route_axis_hash,
        ordered_route_axis_hash=moments.route_axis_hash,
        route_count=moments.route_count,
        proposal_generation=moments.proposal_generation,
        calibration_identity=fd_design.calibration_state_id,
        raw_logit_stream_source_hash=moments.raw_logit_stream_source_hash,
        raw_logit_coordinate=RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE,
        context_mixture_hash=context_mixture,
        stratum_names=moments.stratum_names,
        stratum_weights=fd_design.stratum_weights,  # type: ignore[arg-type]
        population_stratum_counts=moments.population_stratum_counts,
        sealed_population_order_hashes=moments.sealed_population_order_hashes,
        observed_stratum_counts=moments.stratum_counts,
        sampled_look_index=moments.look_index,
        fd_packet_boundary_index=fd_snapshot.packet_boundary_index,
        observed_draw_axis_hash=draw_axis_hash,
        observed_source_context_axis_hash=source_context_axis_hash,
        source_measurement_axis_hash=source_measurement_axis_hash,
        sample_weight_hash=sample_weight_hash,
        corrected_moments_state_hash=moments_hash,
        fd_design_hash=design_hash,
        fd_snapshot_hash=snapshot_hash,
        physical_fd_rank=physical_rank,
        physical_rank_authorization_hash=physical_rank_authorization_hash,
        snapshot_numerical_backend_hash=snapshot_numerics_hash,
        terminal_validation_population_hash=(
            contract.terminal_validation_population_hash
        ),
        terminal_validation_policy_hash=contract.terminal_validation_policy_hash,
        validation_firewall_hash=validation_firewall_hash,
    )


@dataclass(frozen=True, slots=True)
class V1SampledPreFreezeGeometryAdmission:
    """Exact sampled state admitted before any proposal ladder is frozen."""

    binding: V1OptionACampaignBinding = field(repr=False, compare=False)
    current_mask: HardMask = field(repr=False, compare=False)
    moments: V1StratifiedCorrectedAttributionMoments = field(
        repr=False,
        compare=False,
    )
    fd_snapshot: V1StratifiedFDAccumulatorSnapshot = field(
        repr=False,
        compare=False,
    )
    fd_design: V1StratifiedFDDesign = field(repr=False, compare=False)
    source_origin: (
        V1SampledRegisteredLookCommit | V1SampledRegisteredLookExternalDescriptor
    ) = field(repr=False, compare=False)
    physical_rank_authorization: (
        V1SampledPhysicalRankPreFreezeAuthorization
        | V1SampledDeviceFinalRankPreFreezeAuthorization
        | None
    ) = field(default=None, repr=False, compare=False)
    receipt: V1SampledPreFreezeGeometryAdmissionReceipt = field(init=False)
    schema: str = SAMPLED_PRE_FREEZE_GEOMETRY_ADMISSION_SCHEMA
    _admission_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema != SAMPLED_PRE_FREEZE_GEOMETRY_ADMISSION_SCHEMA:
            raise ValueError("unsupported pre-freeze geometry-admission schema")
        receipt = _build_pre_freeze_geometry_admission_receipt(
            binding=self.binding,
            current_mask=self.current_mask,
            moments=self.moments,
            fd_snapshot=self.fd_snapshot,
            fd_design=self.fd_design,
            source_origin=self.source_origin,
            physical_rank_authorization=self.physical_rank_authorization,
        )
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(
            self,
            "_admission_hash",
            content_hash(
                {
                    "schema": self.schema,
                    "campaign_binding_hash": self.binding.binding_hash,
                    "current_mask_hash": self.current_mask.mask_hash,
                    "source_receipt_hash": receipt.receipt_hash,
                    "moments_state_hash": receipt.corrected_moments_state_hash,
                    "fd_snapshot_hash": receipt.fd_snapshot_hash,
                    "fd_design_hash": receipt.fd_design_hash,
                    "physical_fd_rank": receipt.physical_fd_rank,
                    "physical_rank_authorization_hash": (
                        receipt.physical_rank_authorization_hash
                    ),
                    "axis_bridge_dependency": False,
                    "response_exactification_replay_dependency": False,
                },
                prefix=_PRE_FREEZE_ADMISSION_PREFIX,
            ),
        )

    @property
    def admission_hash(self) -> str:
        return self._admission_hash

    @property
    def signed_point(self) -> np.ndarray:
        return self.moments.class_balanced_signed_mean

    @property
    def absolute_point(self) -> np.ndarray:
        return self.moments.class_balanced_absolute_mean

    @property
    def dual_chart_stream_binding(self) -> tuple[str, str, str] | None:
        return _assert_sampled_dual_or_legacy_stream_coordinates(
            moments=self.moments,
            fd_design=self.fd_design,
            source_origin=self.source_origin,
        )

    @property
    def canonical_q_input_authority(self) -> bool:
        return self.dual_chart_stream_binding is not None


def admit_sampled_registered_look_commit_pre_freeze(
    *,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    commit: V1SampledRegisteredLookCommit,
) -> V1SampledPreFreezeGeometryAdmission:
    """Admit a complete live look without requiring post-order receipts."""

    if type(commit) is not V1SampledRegisteredLookCommit:
        raise TypeError("pre-freeze live admission requires its exact commit type")
    commit.commit_hash
    design = commit.request.design
    if type(design) is not V1StratifiedFDDesign:
        raise TypeError("pre-freeze live commit exposes a noncanonical FD design")
    return V1SampledPreFreezeGeometryAdmission(
        binding=binding,
        current_mask=current_mask,
        moments=commit.corrected_attribution_moments,
        fd_snapshot=commit.fd_snapshot,
        fd_design=design,
        source_origin=commit,
    )


def admit_sampled_registered_look_external_view_pre_freeze(
    *,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    view: V1SampledRegisteredLookExternalView,
    fd_design: V1StratifiedFDDesign,
    physical_rank_authorization: (
        V1SampledPhysicalRankPreFreezeAuthorization
        | V1SampledDeviceFinalRankPreFreezeAuthorization
        | None
    ) = None,
) -> V1SampledPreFreezeGeometryAdmission:
    """Admit a verified external look without any frozen-order dependency."""

    if type(view) is not V1SampledRegisteredLookExternalView:
        raise TypeError("pre-freeze external admission requires its exact open view")
    moments = view.materialize_moments()
    snapshot = view.materialize_fd_snapshot(design=fd_design)
    return V1SampledPreFreezeGeometryAdmission(
        binding=binding,
        current_mask=current_mask,
        moments=moments,
        fd_snapshot=snapshot,
        fd_design=fd_design,
        source_origin=view.descriptor,
        physical_rank_authorization=physical_rank_authorization,
    )


def admit_sampled_device_final_rank_materialization_pre_freeze(
    *,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    source_origin: V1SampledRegisteredLookExternalDescriptor,
    moments: V1StratifiedCorrectedAttributionMoments,
    fd_snapshot: V1StratifiedFDAccumulatorSnapshot,
    fd_design: V1StratifiedFDDesign,
    device_final_rank_source: V1SampledDeviceFinalRankSourceReceipt,
    adaptive_rank_policy: V1FQRAdaptiveRankPolicy,
) -> V1SampledPreFreezeGeometryAdmission:
    """Admit the one portable device-D endpoint without a host-replay fiction.

    The portable moment and FD objects are the session's registered final
    materialization.  This function performs no external-state reopen and
    grants no adaptive-rank, C, calibration, oracle, or commit authority.
    """

    if type(moments) is not V1StratifiedCorrectedAttributionMoments:
        raise TypeError(
            "device-final-rank admission requires exact corrected moments"
        )
    if type(fd_snapshot) is not V1StratifiedFDAccumulatorSnapshot:
        raise TypeError("device-final-rank admission requires an exact FD snapshot")
    authorization = authorize_sampled_device_final_rank_source_pre_freeze(
        binding=binding,
        current_mask=current_mask,
        source_origin=source_origin,
        fd_design=fd_design,
        device_final_rank_source=device_final_rank_source,
        adaptive_rank_policy=adaptive_rank_policy,
    )
    if (
        moments.assert_integrity()
        != device_final_rank_source.corrected_moments_state_hash
        or fd_snapshot.assert_integrity()
        != device_final_rank_source.fd_snapshot_hash
    ):
        raise ValueError(
            "device-final-rank portable materialization names another endpoint"
        )
    return V1SampledPreFreezeGeometryAdmission(
        binding=binding,
        current_mask=current_mask,
        moments=moments,
        fd_snapshot=fd_snapshot,
        fd_design=fd_design,
        source_origin=source_origin,
        physical_rank_authorization=authorization,
    )


@dataclass(frozen=True, slots=True)
class V1SampledPreFreezeGeometryAssemblyReceipt:
    """Numerical assembly receipt with an explicit no-raw-row boundary."""

    admission_receipt_hash: str
    admission_hash: str
    source_commit_hash: str
    external_state_descriptor_hash: str | None
    campaign_contract_hash: str
    campaign_binding_hash: str
    current_mask_hash: str
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    proposal_generation: int
    calibration_identity: str
    context_mixture_hash: str
    raw_logit_stream_source_hash: str
    raw_logit_coordinate: str
    source_measurement_axis_hash: str
    sample_weight_hash: str
    corrected_moments_state_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    snapshot_numerical_backend_hash: str
    geometry_state_hash: str
    numerical_assembly_receipt_hash: str
    geometry_result_hash: str
    assembly_device_identity: str
    raw_row_derivation_absence_hash: str
    numerical_assembly_receipt_present: bool = True
    global_trust_derivation_receipt_present: bool = False
    raw_context_route_rows_retained: bool = False
    full_per_draw_source_row_hash_ledger_retained: bool = False
    proposal_geometry_input_authority: bool = True
    proposal_only: bool = True
    irreversible_order_authority: bool = False
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    production_execution_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = SAMPLED_PRE_FREEZE_GEOMETRY_AUTHORITY_SCOPE
    schema: str = SAMPLED_PRE_FREEZE_GEOMETRY_ASSEMBLY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "admission_receipt_hash",
            "admission_hash",
            "source_commit_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "current_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "calibration_identity",
            "context_mixture_hash",
            "raw_logit_stream_source_hash",
            "source_measurement_axis_hash",
            "sample_weight_hash",
            "corrected_moments_state_hash",
            "fd_design_hash",
            "fd_snapshot_hash",
            "snapshot_numerical_backend_hash",
            "geometry_state_hash",
            "numerical_assembly_receipt_hash",
            "geometry_result_hash",
            "assembly_device_identity",
            "raw_row_derivation_absence_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.external_state_descriptor_hash is not None:
            object.__setattr__(
                self,
                "external_state_descriptor_hash",
                _identity(
                    self.external_state_descriptor_hash,
                    name="external state descriptor hash",
                ),
            )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        if self.raw_logit_coordinate != RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE:
            raise ValueError("pre-freeze assembly changed the raw-logit coordinate")
        if (
            self.numerical_assembly_receipt_present is not True
            or self.global_trust_derivation_receipt_present is not False
            or self.raw_context_route_rows_retained is not False
            or self.full_per_draw_source_row_hash_ledger_retained is not False
            or self.proposal_geometry_input_authority is not True
            or self.proposal_only is not True
            or self.irreversible_order_authority is not False
            or self.fq_certification_authority is not False
            or self.bce_oracle_authority is not False
            or self.candidate_freeze_authority is not False
            or self.mask_commit_authority is not False
            or self.production_execution_authority is not False
            or self.terminal_authority is not False
            or self.authority_scope != SAMPLED_PRE_FREEZE_GEOMETRY_AUTHORITY_SCOPE
            or self.schema != SAMPLED_PRE_FREEZE_GEOMETRY_ASSEMBLY_RECEIPT_SCHEMA
        ):
            raise PermissionError(
                "pre-freeze sampled geometry assembly crossed its authority boundary"
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PRE_FREEZE_ASSEMBLY_RECEIPT_PREFIX)


def _pre_freeze_geometry_assembly_receipt(
    admission: V1SampledPreFreezeGeometryAdmission,
    result: V1StratifiedFDGeometryResult,
) -> V1SampledPreFreezeGeometryAssemblyReceipt:
    source = admission.receipt
    result.design.assert_integrity(source.fd_design_hash)
    result.receipt.assert_integrity()
    if (
        result.receipt.combined_geometry_hash != result.geometry.content_hash
        or result.geometry.derivation_receipt is not None
        or result.geometry.measured_mask_hash != source.current_mask_hash
        or result.geometry.calibration_state_id != source.calibration_identity
        or result.geometry.route_ids != admission.fd_design.route_ids
        or result.geometry.metadata.get(
            "sampled_pre_freeze_geometry_admission_receipt_hash"
        )
        != source.receipt_hash
        or result.geometry.metadata.get("sampled_pre_freeze_geometry_admission_hash")
        != admission.admission_hash
        or result.geometry.metadata.get("sampled_pre_freeze_fd_design_hash")
        != source.fd_design_hash
        or result.geometry.metadata.get("sampled_pre_freeze_fd_snapshot_hash")
        != source.fd_snapshot_hash
    ):
        raise ValueError("pre-freeze sampled geometry result graph does not close")
    absence_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_pre_freeze_derivation_absence_v1",
            "admission_receipt_hash": source.receipt_hash,
            "fd_snapshot_hash": source.fd_snapshot_hash,
            "geometry_state_hash": result.geometry.content_hash,
            "global_trust_derivation_receipt_present": False,
            "raw_context_route_rows_retained": False,
            "full_per_draw_source_row_hash_ledger_retained": False,
        },
        prefix=_PRE_FREEZE_DERIVATION_ABSENCE_PREFIX,
    )
    return V1SampledPreFreezeGeometryAssemblyReceipt(
        admission_receipt_hash=source.receipt_hash,
        admission_hash=admission.admission_hash,
        source_commit_hash=source.source_commit_hash,
        external_state_descriptor_hash=source.external_state_descriptor_hash,
        campaign_contract_hash=source.campaign_contract_hash,
        campaign_binding_hash=source.campaign_binding_hash,
        current_mask_hash=source.current_mask_hash,
        physical_route_axis_hash=source.physical_route_axis_hash,
        ordered_route_axis_hash=source.ordered_route_axis_hash,
        proposal_generation=source.proposal_generation,
        calibration_identity=source.calibration_identity,
        context_mixture_hash=source.context_mixture_hash,
        raw_logit_stream_source_hash=source.raw_logit_stream_source_hash,
        raw_logit_coordinate=source.raw_logit_coordinate,
        source_measurement_axis_hash=source.source_measurement_axis_hash,
        sample_weight_hash=source.sample_weight_hash,
        corrected_moments_state_hash=source.corrected_moments_state_hash,
        fd_design_hash=source.fd_design_hash,
        fd_snapshot_hash=source.fd_snapshot_hash,
        snapshot_numerical_backend_hash=source.snapshot_numerical_backend_hash,
        geometry_state_hash=result.geometry.content_hash,
        numerical_assembly_receipt_hash=result.receipt.receipt_hash,
        geometry_result_hash=result.result_hash,
        assembly_device_identity=result.receipt.device_identity,
        raw_row_derivation_absence_hash=absence_hash,
    )


@dataclass(frozen=True, slots=True)
class V1SampledPreFreezeGeometrySource:
    """Proposal-only sampled ``GlobalTrustGeometryState`` and full lineage."""

    admission: V1SampledPreFreezeGeometryAdmission = field(
        repr=False,
        compare=False,
    )
    geometry_result: V1StratifiedFDGeometryResult = field(
        repr=False,
        compare=False,
    )
    receipt: V1SampledPreFreezeGeometryAssemblyReceipt = field(init=False)
    schema: str = SAMPLED_PRE_FREEZE_GEOMETRY_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.admission) is not V1SampledPreFreezeGeometryAdmission:
            raise TypeError("pre-freeze geometry source requires its exact admission")
        if type(self.geometry_result) is not V1StratifiedFDGeometryResult:
            raise TypeError("pre-freeze geometry source requires its exact result")
        if self.schema != SAMPLED_PRE_FREEZE_GEOMETRY_SOURCE_SCHEMA:
            raise ValueError("unsupported pre-freeze geometry-source schema")
        object.__setattr__(
            self,
            "receipt",
            _pre_freeze_geometry_assembly_receipt(
                self.admission,
                self.geometry_result,
            ),
        )

    @property
    def geometry_state(self) -> GlobalTrustGeometryState:
        return self.geometry_result.geometry

    @property
    def source_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "admission_hash": self.admission.admission_hash,
                "geometry_result_hash": self.geometry_result.result_hash,
                "assembly_receipt_hash": self.receipt.receipt_hash,
            },
            prefix=_PRE_FREEZE_SOURCE_PREFIX,
        )

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def mask_commit_authority(self) -> bool:
        return False


def assemble_pre_freeze_sampled_geometry_source(
    admission: V1SampledPreFreezeGeometryAdmission,
    *,
    device: object = "cpu",
) -> V1SampledPreFreezeGeometrySource:
    """Restore sampled FD state and assemble proposal-only response geometry."""

    if type(admission) is not V1SampledPreFreezeGeometryAdmission:
        raise TypeError("pre-freeze geometry assembly requires an exact admission")
    source = admission.receipt
    dual_binding = admission.dual_chart_stream_binding
    before = admission.fd_snapshot.assert_integrity()
    result = V1StratifiedDeviceFDTrustAccumulator.finalize_snapshot_combined_consuming(
        admission.fd_snapshot,
        design=admission.fd_design,
        sketch_rank=admission.fd_snapshot.sketch_rank,
        device=device,
        metadata={
            "sampled_pre_freeze_geometry_admission_receipt_hash": (source.receipt_hash),
            "sampled_pre_freeze_geometry_admission_hash": admission.admission_hash,
            "modern_v1_campaign_contract_hash": source.campaign_contract_hash,
            "modern_v1_campaign_binding_hash": source.campaign_binding_hash,
            "sampled_context_mixture_hash": source.context_mixture_hash,
            "sampled_proposal_generation": source.proposal_generation,
            "raw_logit_coordinate": source.raw_logit_coordinate,
            "raw_logit_stream_source_hash": source.raw_logit_stream_source_hash,
            "canonical_q_input_authority": dual_binding is not None,
            "corrected_evidence_stream_identity_hash": (
                None if dual_binding is None else dual_binding[0]
            ),
            "literal_jacobian_stream_identity_hash": (
                None if dual_binding is None else dual_binding[1]
            ),
            "same_capture_and_vjp_dual_receipt_hash": (
                None if dual_binding is None else dual_binding[2]
            ),
            "corrected_rows_excluded_from_response_geometry": (
                dual_binding is not None
            ),
            "corrected_moments_state_hash": source.corrected_moments_state_hash,
            "sampled_pre_freeze_fd_design_hash": source.fd_design_hash,
            "sampled_pre_freeze_fd_snapshot_hash": source.fd_snapshot_hash,
            "sample_weight_hash": source.sample_weight_hash,
            "axis_bridge_dependency": False,
            "response_exactification_replay_dependency": False,
            "raw_context_route_rows_retained": False,
            "full_per_draw_source_row_hash_ledger_retained": False,
            "proposal_geometry_input_authority": True,
            "proposal_only": True,
            "fq_certification_authority": False,
            "bce_oracle_authority": False,
            "candidate_freeze_authority": False,
            "mask_commit_authority": False,
            "production_execution_authority": False,
        },
    )
    admission.fd_snapshot.assert_integrity(before)
    if result.geometry.derivation_receipt is not None:
        raise RuntimeError(
            "pre-freeze sampled geometry manufactured a raw-row derivation receipt"
        )
    return V1SampledPreFreezeGeometrySource(
        admission=admission,
        geometry_result=result,
    )


def require_standard_fd_stream_acquisition(
    source: V1SampledSummaryAcquisitionSource,
) -> NoReturn:
    """Fail closed instead of inventing the raw ledger required by v1 FD."""

    if type(source) is not V1SampledSummaryAcquisitionSource:
        raise TypeError("standard FD coercion requires an exact sampled summary source")
    raise V1SampledSummaryAcquisitionIncompatibility(
        f"{SAMPLED_SUMMARY_STANDARD_ACQUISITION_GAP_ID}; "
        f"gap_proof={source.receipt.standard_acquisition_gap_proof_hash}"
    )


def require_exact_gate_correction_acquisition(
    admission: V1SampledDiscoveryAdmission,
) -> NoReturn:
    """Fail closed instead of upgrading a sampled point to an exact price."""

    if type(admission) is not V1SampledDiscoveryAdmission:
        raise TypeError("exact-gate coercion requires an exact sampled admission")
    status = admission.receipt.point_status
    raise V1SampledSummaryAcquisitionIncompatibility(
        "sampled corrected moments and regret-certified order do not issue the "
        "exact tangent-price lineage required by "
        "V1AnalyticExactGateCorrectionAcquisition; "
        f"point_status={status}; source_receipt={admission.receipt.receipt_hash}"
    )


__all__ = [
    "REGRET_ORDER_STATUS",
    "SAMPLED_POINT_ESTIMATE_STATUS",
    "SAMPLED_DEVICE_FINAL_RANK_PRE_FREEZE_AUTHORIZATION_SCHEMA",
    "SAMPLED_DEVICE_FINAL_RANK_PRE_FREEZE_AUTHORITY_SCOPE",
    "SAMPLED_PHYSICAL_RANK_PRE_FREEZE_AUTHORIZATION_SCHEMA",
    "SAMPLED_PHYSICAL_RANK_PRE_FREEZE_AUTHORITY_SCOPE",
    "SAMPLED_PRE_FREEZE_GEOMETRY_AUTHORITY_SCOPE",
    "SAMPLED_SUMMARY_AUTHORITY_SCOPE",
    "SAMPLED_SUMMARY_STANDARD_ACQUISITION_GAP_ID",
    "V1SampledDiscoveryAdmission",
    "V1SampledDiscoverySourceReceipt",
    "V1SampledDeviceFinalRankPreFreezeAuthorization",
    "V1SampledPreFreezeGeometryAdmission",
    "V1SampledPreFreezeGeometryAdmissionReceipt",
    "V1SampledPreFreezeGeometryAssemblyReceipt",
    "V1SampledPreFreezeGeometrySource",
    "V1SampledPhysicalRankPreFreezeAuthorization",
    "V1SampledSummaryAcquisitionIncompatibility",
    "V1SampledSummaryAcquisitionSource",
    "V1SampledSummaryGeometryReceipt",
    "admit_sampled_registered_look_commit_pre_freeze",
    "admit_sampled_device_final_rank_materialization_pre_freeze",
    "admit_sampled_registered_look_external_view_pre_freeze",
    "admit_sampled_registered_look_commit",
    "admit_sampled_registered_look_external_view",
    "assemble_pre_freeze_sampled_geometry_source",
    "assemble_sampled_summary_acquisition_source",
    "authorize_sampled_device_final_rank_source_pre_freeze",
    "authorize_sampled_physical_rank_replay_pre_freeze",
    "require_exact_gate_correction_acquisition",
    "require_standard_fd_stream_acquisition",
]
