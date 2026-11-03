"""Row-backed device-resident D live consumer: witness inputs and C-state gate.

This is the pre-eviction live consumer of the row-backed lane
(``sampled_device_resident_d_final_live_consumer``, called from the backend's
``_consume_sampled_device_resident_d_final_live_handoff_before_evict``).  It
produces, inside the live seam and from source-bound objects only, the six
input groups of the device-native adaptive D rank/sample KKT-face witness:

1. v2 endogenous-rank registrations (policy, sampled-KKT policy, D prefix,
   same-VJP fanout) from the backend's registered HBM numerals; the v2 D
   prefix names the device prefix identity
   (``build_v1_sampled_device_registered_d_prefix_binding``'s hash) as its
   ``registered_d_prefix_identity_hash`` and the device source was issued
   with the v2 ``prefix_hash`` (no adapter identity map);
2. the ceiling memory forecast from the same numerals;
3. the ceiling spectrum read from the assembled ceiling geometry before the
   route-scale state is released;
4. rank-local decision certificates for the exposed nested lattice as
   truncation views of the ceiling state (one rank-local KKT solve per
   evaluated rank, no D re-execution);
5. D-sampling owner radii: exact finite-population census (zero radius) when
   the registered D prefix is the full population; otherwise the typed
   absence ``D_sampling_owner_radius_absent`` (nothing is inferred and no
   unit is converted);
6. the live parents are the handoff and the final receipt themselves.

The consumer never evicts the session, never calls the measurement provider,
never opens C, and returns one compact typed result the backend retains for
the F/Q/R C-state gate.  A typed absence is a lawful outcome: the result
then carries the refusal and C stays closed on this chart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Final, NoReturn

import numpy as np

from .contracts import content_hash
from .modern_v1_fqr_adaptive_rank import (
    V1FQRAdaptiveRankPolicy,
    V1FQREndogenousRankDPrefixV2,
    V1FQREndogenousRankNoCRefusalV2,
    V1FQREndogenousRankPolicyV2,
    V1FQREndogenousRankSameVJPFanoutV2,
    register_v1_fqr_endogenous_rank_d_prefix_v2,
    register_v1_fqr_endogenous_rank_policy_v2,
    register_v1_fqr_same_vjp_rank_fanout_v2,
)
from .modern_v1_fqr_ceiling_rank_geometry import (
    V1FQRCeilingRankMemoryForecast,
    V1FQRCeilingRankNoCRefusal,
)
from .modern_v1_sampled_acquisition_adapters import (
    admit_sampled_device_final_rank_materialization_pre_freeze,
    assemble_pre_freeze_sampled_geometry_source,
)
from .modern_v1_sampled_composition_backend import (
    V1SampledCompositionRegisteredDPrefixBinding,
)
from .modern_v1_sampled_device_final_rank_source import (
    V1SampledDeviceFinalRankSourceReceipt,
)
from .modern_v1_sampled_device_resident_d_session import (
    V1SampledDeviceResidentDSession,
)
from .modern_v1_sampled_kkt_bridge import build_v1_sampled_kkt_branch_family_graph
from .modern_v1_sampled_kkt_device_shell_bridge import (
    V1SampledKKTEndogenousRankPolicyV2,
    register_v1_sampled_kkt_endogenous_rank_policy_v2,
)
from .modern_v1_sampled_v7_ceiling_rank_d_stability import (
    V1SampledV7CeilingRankDGeometryBinding,
    issue_v1_sampled_v7_ceiling_rank_geometry_from_device_final_rank_source,
)
from .modern_v1_sampled_v7_ceiling_truncation_view import (
    V1CeilingTruncationRankLocalDeriver,
    V1CeilingTruncationViewRefusal,
    derive_v1_ceiling_fd_eigenvalue_square_axis,
)
from .modern_v1_sampled_v7_device_native_adaptive_rank_sample_witness import (
    V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessResult,
    V1DeviceNativeAdaptiveRankSampleWitnessRefusal,
    V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode,
    V1DeviceNativeAdaptiveRankSelectionReceipt,
    forecast_v1_device_native_ceiling_rank_memory,
    issue_v1_device_native_adaptive_D_rank_sample_KKT_face_witness,
    select_v1_device_native_adaptive_physical_rank,
    summarize_v1_device_native_adaptive_rank_sample_witness,
)
from .modern_v1_signed_work_decision_enclosure import (
    V1SignedWorkDecisionOwnerRole,
    V1SignedWorkDecisionOwnerUncertaintyReceipt,
)
from .modern_v1_stratified_fd_gpu import (
    V1StratifiedDeviceFDTrustAccumulator,
    V1StratifiedFDDesign,
)
from .modern_v1_confidence_box_external_state import (
    open_confidence_box_v2_external_state,
)


ROW_BACKED_WITNESS_CONSUMER_SCHEMA: Final = (
    "partition_free_v1_alexnet_row_backed_device_resident_d_witness_consumer_v1"
)
ROW_BACKED_WITNESS_CONSUMER_RESULT_KEY: Final = "row_backed_witness_consumer_result"
ROW_BACKED_C_STATE_WITNESS_SIBLING_SCHEMA: Final = (
    "partition_free_v1_alexnet_row_backed_fqr_c_state_witness_sibling_v1"
)
ROW_BACKED_SAMPLING_OWNER_CENSUS_RULE: Final = (
    "registered D prefix equals the sealed finite population in every stratum: "
    "the sampling owners are an exact finite-population census with zero "
    "outward radius (no inference, no unit conversion); otherwise the radius "
    "in the certificate's decision-key units is a typed absence"
)
_PREFIX: Final = "partition-free-v1-alexnet-row-backed-witness-consumer-"
_DEVICE_RESIDENT_FINAL_RANK_PROVENANCE: Final = "device_resident_final_rank"


class V1RowBackedWitnessConsumerRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    CUSTODY_MISMATCH = "custody_mismatch"
    ENDOGENOUS_RANK_POLICY_ABSENT = "endogenous_rank_policy_absent"
    SOURCE_COORDINATE_MISMATCH = "source_coordinate_mismatch"
    SAME_VJP_FANOUT_ABSENT = "same_vjp_fanout_absent"
    PHYSICAL_HBM_REFUSED = "physical_hbm_refused"
    D_SAMPLING_OWNER_RADIUS_ABSENT = "D_sampling_owner_radius_absent"
    RANK_LOCAL_DECISION_IMAGE_ABSENT = "rank_local_decision_image_absent"
    WITNESS_REFUSED = "witness_refused"
    NO_DECISION_STABLE_RANK = "no_decision_stable_rank"


class V1RowBackedWitnessConsumerRefusal(RuntimeError):
    def __init__(
        self,
        code: V1RowBackedWitnessConsumerRefusalCode,
        message: str,
        *,
        missing_inputs: Sequence[str] = (),
        witness_code: str | None = None,
    ) -> None:
        if type(code) is not V1RowBackedWitnessConsumerRefusalCode:
            raise TypeError("row-backed witness consumer refusal code must be exact")
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.message = message
        self.missing_inputs = tuple(str(value) for value in missing_inputs)
        self.witness_code = witness_code

    def to_json_payload(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "message": self.message,
            "missing_inputs": self.missing_inputs,
            "witness_code": self.witness_code,
        }


def _refuse(
    code: V1RowBackedWitnessConsumerRefusalCode,
    message: str,
    *,
    missing_inputs: Sequence[str] = (),
    witness_code: str | None = None,
) -> NoReturn:
    raise V1RowBackedWitnessConsumerRefusal(
        code,
        message,
        missing_inputs=missing_inputs,
        witness_code=witness_code,
    )


def _exact(value: object, expected: type, label: str) -> None:
    if type(value) is not expected:
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.INVALID_TYPE,
            f"{label} must be the exact {expected.__name__}",
        )


# ---------------------------------------------------------------------------
# v2 endogenous-rank registrations naming the device prefix identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V1RowBackedEndogenousRankRegistrationsV2:
    """v2 policy, sampled-KKT policy, and D prefix bound to one device prefix."""

    memory_policy: V1FQREndogenousRankPolicyV2
    sampled_policy: V1SampledKKTEndogenousRankPolicyV2
    d_prefix: V1FQREndogenousRankDPrefixV2
    source_adaptive_rank_policy_hash: str
    registered_d_prefix_identity_hash: str

    def __post_init__(self) -> None:
        _exact(self.memory_policy, V1FQREndogenousRankPolicyV2, "v2 memory policy")
        _exact(self.sampled_policy, V1SampledKKTEndogenousRankPolicyV2, "v2 sampled policy")
        _exact(self.d_prefix, V1FQREndogenousRankDPrefixV2, "v2 D prefix")
        if (
            self.sampled_policy.source_memory_forecast_policy_hash
            != self.memory_policy.policy_hash
            or self.d_prefix.policy_hash != self.memory_policy.policy_hash
            or self.d_prefix.registered_d_prefix_identity_hash
            != self.registered_d_prefix_identity_hash
        ):
            _refuse(
                V1RowBackedWitnessConsumerRefusalCode.SOURCE_COORDINATE_MISMATCH,
                "v2 registrations do not name one policy and one device prefix identity",
            )

    def to_json_payload(self) -> dict[str, object]:
        return {
            "memory_policy_hash": self.memory_policy.policy_hash,
            "sampled_policy_hash": self.sampled_policy.policy_hash,
            "d_prefix_hash": self.d_prefix.prefix_hash,
            "source_adaptive_rank_policy_hash": self.source_adaptive_rank_policy_hash,
            "registered_d_prefix_identity_hash": self.registered_d_prefix_identity_hash,
            "admitted_rank_lattice": self.memory_policy.admitted_rank_lattice,
            "identifiable_admitted_ranks": self.d_prefix.identifiable_admitted_ranks,
            "full_D_census": self.d_prefix.full_D_census,
        }


def register_v1_row_backed_endogenous_rank_registrations_v2(
    *,
    adaptive_rank_policy: V1FQRAdaptiveRankPolicy,
    fd_design: V1StratifiedFDDesign,
    registered_d_prefix_binding: V1SampledCompositionRegisteredDPrefixBinding,
    population_stratum_counts: Sequence[int],
    observed_stratum_counts: Sequence[int],
    current_mask_hash: str,
    route_axis_hash: str,
    proposal_generation: int,
) -> V1RowBackedEndogenousRankRegistrationsV2:
    """Issue the v2 registrations from the registered HBM numerals.

    Every numeral is read from the pre-D registered adaptive-rank policy
    (route count, strata, response rows, available workspace below the frozen
    HBM ceiling, legality blocks, rounding tile, maximum hard actions, D packet
    staging); the D prefix identity is the device prefix binding hash and the
    realized context/draw axes are the binding's.  Deterministic: the same
    inputs re-register byte-identical objects at publication and at the live
    consumer.
    """

    _exact(adaptive_rank_policy, V1FQRAdaptiveRankPolicy, "adaptive rank policy")
    _exact(fd_design, V1StratifiedFDDesign, "FD design")
    _exact(
        registered_d_prefix_binding,
        V1SampledCompositionRegisteredDPrefixBinding,
        "registered D prefix binding",
    )
    adaptive_rank_policy.assert_integrity()
    forecasts = adaptive_rank_policy.memory_forecasts
    if not forecasts:
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.ENDOGENOUS_RANK_POLICY_ABSENT,
            "adaptive rank policy carries no registered memory forecast",
        )
    numerals = forecasts[0]
    strata = tuple(fd_design.stratum_names)
    if len(strata) != adaptive_rank_policy.stratum_count:
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.SOURCE_COORDINATE_MISMATCH,
            "FD design strata differ from the registered adaptive-rank stratum count",
        )
    population = tuple(int(value) for value in population_stratum_counts)
    observed = tuple(int(value) for value in observed_stratum_counts)
    if len(population) != len(strata) or len(observed) != len(strata):
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.SOURCE_COORDINATE_MISMATCH,
            "stratum counts do not cover the FD design strata",
        )
    if tuple(registered_d_prefix_binding.cumulative_stratum_counts) != observed:
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.SOURCE_COORDINATE_MISMATCH,
            "observed stratum counts differ from the device prefix binding",
        )
    memory_policy = register_v1_fqr_endogenous_rank_policy_v2(
        policy_id=content_hash(
            {
                "schema": ROW_BACKED_WITNESS_CONSUMER_SCHEMA + ":endogenous-policy-id",
                "source_adaptive_rank_policy_hash": adaptive_rank_policy.policy_hash,
                "fd_design_hash": fd_design.design_hash,
            },
            prefix=_PREFIX + "endogenous-policy-id-",
        ),
        route_count=adaptive_rank_policy.route_count,
        positive_weight_stratum_ids=strata,
        response_rows=adaptive_rank_policy.response_rows,
        physical_workspace_ceiling_bytes=numerals.available_rank_workspace_bytes,
        kkt_legality_block_count=numerals.kkt_legality_block_count,
        kkt_rounding_tile_size=numerals.kkt_rounding_tile_size,
        maximum_hard_action_count=numerals.maximum_hard_action_count,
        D_packet_staging_device_bytes=numerals.D_packet_staging_device_bytes,
    )
    if type(memory_policy) is V1FQREndogenousRankNoCRefusalV2:
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.ENDOGENOUS_RANK_POLICY_ABSENT,
            f"v2 endogenous rank policy refused physical liveness: {memory_policy.reason}",
        )
    sampled_policy = register_v1_sampled_kkt_endogenous_rank_policy_v2(policy=memory_policy)
    d_prefix = register_v1_fqr_endogenous_rank_d_prefix_v2(
        policy=memory_policy,
        current_mask_hash=current_mask_hash,
        route_axis_hash=route_axis_hash,
        proposal_generation=proposal_generation,
        registered_d_prefix_identity_hash=registered_d_prefix_binding.binding_hash,
        D_context_axis_hash=registered_d_prefix_binding.D_context_axis_hash,
        D_draw_axis_hash=registered_d_prefix_binding.D_draw_axis_hash,
        observed_stratum_counts=dict(zip(strata, observed, strict=True)),
        population_stratum_counts=dict(zip(strata, population, strict=True)),
    )
    return V1RowBackedEndogenousRankRegistrationsV2(
        memory_policy=memory_policy,
        sampled_policy=sampled_policy,
        d_prefix=d_prefix,
        source_adaptive_rank_policy_hash=adaptive_rank_policy.policy_hash,
        registered_d_prefix_identity_hash=registered_d_prefix_binding.binding_hash,
    )


def register_v1_row_backed_same_vjp_fanout_v2(
    *,
    registrations: V1RowBackedEndogenousRankRegistrationsV2,
    capture_vjp_packet_hashes: Sequence[str],
    nested_view_hashes_by_rank: Mapping[int, str],
    ceiling_rank: int,
) -> V1FQREndogenousRankSameVJPFanoutV2:
    """Bind the packet lineage to the nested-view FD state hashes (one ceiling state)."""

    _exact(registrations, V1RowBackedEndogenousRankRegistrationsV2, "v2 registrations")
    identifiable = registrations.d_prefix.identifiable_admitted_ranks
    above = tuple(rank for rank in identifiable if rank > ceiling_rank)
    if above:
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.SAME_VJP_FANOUT_ABSENT,
            f"identifiable admitted ranks {above} exceed the assembled ceiling rank "
            f"{ceiling_rank}; this session holds one ceiling FD state and no "
            "rank-local state above it",
            missing_inputs=tuple(f"rank_local_fd_state[{rank}]" for rank in above),
        )
    missing = tuple(rank for rank in identifiable if rank not in nested_view_hashes_by_rank)
    if missing:
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.SAME_VJP_FANOUT_ABSENT,
            f"no nested ceiling view exists for identifiable ranks {missing}",
            missing_inputs=tuple(f"nested_view[{rank}]" for rank in missing),
        )
    fanout = register_v1_fqr_same_vjp_rank_fanout_v2(
        policy=registrations.memory_policy,
        d_prefix=registrations.d_prefix,
        capture_vjp_packet_hashes=tuple(capture_vjp_packet_hashes),
        rank_local_fd_state_hashes={
            rank: nested_view_hashes_by_rank[rank] for rank in identifiable
        },
    )
    if type(fanout) is V1FQREndogenousRankNoCRefusalV2:
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.SAME_VJP_FANOUT_ABSENT,
            f"same-VJP fanout refused: {fanout.reason}",
            missing_inputs=("fanout",),
        )
    return fanout


# ---------------------------------------------------------------------------
# Sampling owner radii (census or typed absence)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V1RowBackedSamplingOwnerRadiiResult:
    receipts: tuple[V1SignedWorkDecisionOwnerUncertaintyReceipt, ...]
    radius_source: str
    per_step_alpha: float
    census: bool
    rule: str = ROW_BACKED_SAMPLING_OWNER_CENSUS_RULE

    def to_json_payload(self) -> dict[str, object]:
        return {
            "owner_receipt_hashes": tuple(value.receipt_hash for value in self.receipts),
            "radius_source": self.radius_source,
            "per_step_alpha": self.per_step_alpha,
            "census": self.census,
            "rule": self.rule,
        }


def derive_v1_row_backed_sampling_owner_radii(
    *,
    device_final_rank_source: V1SampledDeviceFinalRankSourceReceipt,
    d_prefix: V1FQREndogenousRankDPrefixV2,
    per_step_alpha: float,
    finite_population_statistics_hash: str,
) -> V1RowBackedSamplingOwnerRadiiResult:
    """Exact-census owner receipts, or the typed absence.

    When every stratum's registered prefix equals its sealed population the
    two sampling owners are an exact finite-population census: outward radius
    zero by definition (``exact_finite_population_census=True``), bound to the
    device source's registered prefix, route axis, and the session's own
    finite-population statistics hash.  Otherwise no radius is issued: mapping
    the session's per-stratum means and variances at the registered per-step
    alpha into the certificate's decision-key units requires a source-derived
    decision-key sensitivity this lane does not carry, and inferring or
    converting one is forbidden.
    """

    _exact(device_final_rank_source, V1SampledDeviceFinalRankSourceReceipt, "device source")
    _exact(d_prefix, V1FQREndogenousRankDPrefixV2, "v2 D prefix")
    alpha = float(per_step_alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.INVALID_TYPE,
            "registered per-step alpha must lie in (0, 1)",
        )
    if type(finite_population_statistics_hash) is not str or not finite_population_statistics_hash:
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.INVALID_TYPE,
            "finite-population statistics hash must be one nonempty string",
        )
    if d_prefix.prefix_hash != device_final_rank_source.registered_d_prefix_hash:
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.SOURCE_COORDINATE_MISMATCH,
            "v2 D prefix is not the device source's registered prefix",
        )
    if not d_prefix.full_D_census:
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.D_SAMPLING_OWNER_RADIUS_ABSENT,
            "registered D prefix is not the full finite population "
            f"(observed={d_prefix.observed_stratum_counts}, "
            f"population={d_prefix.population_stratum_counts}); an outward "
            "sampling radius in the certificate's decision-key units at the "
            f"registered per-step alpha {alpha} needs a source-derived "
            "decision-key sensitivity that this lane does not carry; nothing is "
            "inferred or converted",
            missing_inputs=("sampling_owner_receipts",),
        )
    uncertainty_set_hash = content_hash(
        {
            "schema": ROW_BACKED_WITNESS_CONSUMER_SCHEMA + ":census-uncertainty-set",
            "device_final_rank_source_receipt_hash": device_final_rank_source.receipt_hash,
            "registered_d_prefix_hash": device_final_rank_source.registered_d_prefix_hash,
            "finite_population_statistics_hash": finite_population_statistics_hash,
            "population_stratum_counts": d_prefix.population_stratum_counts,
        },
        prefix=_PREFIX + "census-uncertainty-set-",
    )
    receipts = []
    for role in (
        V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING,
        V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING,
    ):
        stream_hash = (
            device_final_rank_source.corrected_a_stream_identity_hash
            if role is V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING
            else device_final_rank_source.literal_j_stream_identity_hash
        )
        realization = content_hash(
            {
                "schema": ROW_BACKED_WITNESS_CONSUMER_SCHEMA + ":census-realization",
                "role": role.value,
                "label": "exact-full-train-census",
                "source_receipt_hash": device_final_rank_source.receipt_hash,
                "context_axis_hash": device_final_rank_source.context_axis_hash,
                "draw_axis_hash": device_final_rank_source.draw_axis_hash,
                "stream_source_hash": stream_hash,
                "per_step_alpha": alpha,
            },
            prefix=_PREFIX + "census-realization-",
        )
        receipts.append(
            V1SignedWorkDecisionOwnerUncertaintyReceipt(
                owner_role=role,
                uncertainty_set_hash=uncertainty_set_hash,
                owner_set_hash=content_hash(
                    {"role": role.value, "uncertainty_set_hash": uncertainty_set_hash},
                    prefix=_PREFIX + "census-owner-set-",
                ),
                registered_d_w_prefix_hash=device_final_rank_source.registered_d_prefix_hash,
                route_axis_hash=device_final_rank_source.route_axis_hash,
                realization_hashes=(realization,),
                source_derivation_hash=content_hash(
                    {
                        "rule": ROW_BACKED_SAMPLING_OWNER_CENSUS_RULE,
                        "role": role.value,
                        "realization": realization,
                    },
                    prefix=_PREFIX + "census-derivation-",
                ),
                outward_radius_upper_bound=0.0,
                statistical_uncertainty=False,
                deterministic_uncertainty=False,
                spends_alpha=False,
                exact_finite_population_census=True,
            )
        )
    return V1RowBackedSamplingOwnerRadiiResult(
        receipts=tuple(receipts),
        radius_source="exact_finite_population_census",
        per_step_alpha=alpha,
        census=True,
    )


# ---------------------------------------------------------------------------
# Consumer result and the C-state sibling receipt
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class V1RowBackedDeviceResidentDWitnessConsumerResult:
    """Compact typed product of the live consumer; retained by the backend."""

    current_mask_hash: str
    proposal_generation: int
    device_final_rank_source_receipt_hash: str
    registered_d_prefix_hash: str
    ceiling_rank: int
    hbm_admitted_fd_ranks: tuple[int, ...]
    stopped_at: str
    registrations: V1RowBackedEndogenousRankRegistrationsV2 | None = None
    fanout: V1FQREndogenousRankSameVJPFanoutV2 | None = None
    memory_forecast: V1FQRCeilingRankMemoryForecast | None = None
    eigenvalue_square_axis: tuple[float, ...] | None = None
    deterministic_fd_residual_spectral_upper_bound: float | None = None
    direction_l2_square_upper_bound: float | None = None
    direction_bound_source: str | None = None
    geometry_binding_hash: str | None = None
    radii: V1RowBackedSamplingOwnerRadiiResult | None = None
    truncation_summary: Mapping[str, Any] | None = None
    witness: V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessResult | None = field(
        default=None, repr=False
    )
    selection: V1DeviceNativeAdaptiveRankSelectionReceipt | None = None
    refusal: V1RowBackedWitnessConsumerRefusal | None = None
    provider_call_count_before: int = 0
    provider_call_count_after: int = 0
    session_evicted_by_consumer: bool = False
    C_stream_open_authority: bool = False
    schema: str = ROW_BACKED_WITNESS_CONSUMER_SCHEMA

    def __post_init__(self) -> None:
        if self.session_evicted_by_consumer is not False or self.C_stream_open_authority is not False:
            raise PermissionError("row-backed witness consumer crossed its authority")
        if (self.witness is None) == (self.refusal is None):
            raise ValueError("consumer result carries exactly one of witness or refusal")
        if self.witness is not None and type(self.witness) is not (
            V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessResult
        ):
            raise TypeError("consumer result requires the exact witness result")
        if self.refusal is not None and type(self.refusal) is not V1RowBackedWitnessConsumerRefusal:
            raise TypeError("consumer result requires the exact typed refusal")
        if self.provider_call_count_before != self.provider_call_count_after:
            raise PermissionError("the consumer observed a measurement-provider call")

    @property
    def freeze_for_C(self):
        return None if self.witness is None else self.witness.freeze_for_C

    @property
    def selected_physical_fd_rank(self) -> int | None:
        return None if self.witness is None else self.witness.receipt.selected_physical_fd_rank

    def to_json_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "device_final_rank_source_receipt_hash": self.device_final_rank_source_receipt_hash,
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "ceiling_rank": self.ceiling_rank,
            "hbm_admitted_fd_ranks": self.hbm_admitted_fd_ranks,
            "stopped_at": self.stopped_at,
            "registrations": None if self.registrations is None else self.registrations.to_json_payload(),
            "fanout_hash": None if self.fanout is None else self.fanout.fanout_hash,
            "memory_forecast_hash": None if self.memory_forecast is None else self.memory_forecast.forecast_hash,
            "eigenvalue_square_axis": self.eigenvalue_square_axis,
            "deterministic_fd_residual_spectral_upper_bound": (
                self.deterministic_fd_residual_spectral_upper_bound
            ),
            "direction_l2_square_upper_bound": self.direction_l2_square_upper_bound,
            "direction_bound_source": self.direction_bound_source,
            "geometry_binding_hash": self.geometry_binding_hash,
            "radii": None if self.radii is None else self.radii.to_json_payload(),
            "truncation_summary": None if self.truncation_summary is None else dict(self.truncation_summary),
            "witness": (
                None
                if self.witness is None
                else summarize_v1_device_native_adaptive_rank_sample_witness(self.witness)
            ),
            "freeze_for_C_hash": None if self.freeze_for_C is None else self.freeze_for_C.freeze_hash,
            "selection_receipt_hash": None if self.selection is None else self.selection.receipt_hash,
            "selected_physical_fd_rank": self.selected_physical_fd_rank,
            "refusal": None if self.refusal is None else self.refusal.to_json_payload(),
            "provider_call_count_before": self.provider_call_count_before,
            "provider_call_count_after": self.provider_call_count_after,
            "session_evicted_by_consumer": self.session_evicted_by_consumer,
            "C_stream_open_authority": self.C_stream_open_authority,
        }


@dataclass(frozen=True, slots=True)
class V1RowBackedCStateWitnessSiblingReceipt:
    """New sibling family keyed by the F/Q/R C-state payload hash."""

    fqr_C_state_payload_hash: str
    current_mask_hash: str
    proposal_generation: int
    graph_hash: str
    graph_kkt_fd_rank: int
    graph_fd_sketch_rank: int
    device_final_rank_source_receipt_hash: str
    witness_receipt_hash: str | None
    freeze_for_C_hash: str | None
    rank_selection_receipt_hash: str | None
    selected_physical_fd_rank: int | None
    witness_no_C_code: str | None
    consumer_refusal_code: str | None
    canonical_fqr_candidate_queue_authority: bool
    schema: str = ROW_BACKED_C_STATE_WITNESS_SIBLING_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "fqr_C_state_payload_hash",
            "current_mask_hash",
            "graph_hash",
            "device_final_rank_source_receipt_hash",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be one nonempty string")
        if type(self.canonical_fqr_candidate_queue_authority) is not bool:
            raise TypeError("queue authority must be an exact boolean")
        authority_replay = (
            self.witness_receipt_hash is not None
            and self.freeze_for_C_hash is not None
            and self.rank_selection_receipt_hash is not None
            and self.selected_physical_fd_rank == self.graph_fd_sketch_rank
            and self.consumer_refusal_code is None
        )
        if self.canonical_fqr_candidate_queue_authority and not authority_replay:
            raise PermissionError("C-state witness sibling grants authority without its witness")
        if self.schema != ROW_BACKED_C_STATE_WITNESS_SIBLING_SCHEMA:
            raise ValueError("C-state witness sibling schema is invalid")

    @property
    def receipt_hash(self) -> str:
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        return content_hash(payload, prefix=_PREFIX + "c-state-sibling-")

    def to_json_payload(self) -> dict[str, object]:
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        payload["receipt_hash"] = self.receipt_hash
        return payload


def evaluate_v1_row_backed_c_state_witness_authority(
    *,
    consumer_result: V1RowBackedDeviceResidentDWitnessConsumerResult | None,
    graph_fd_sketch_rank: int,
    device_final_rank_source_receipt_hash: str,
) -> tuple[bool, dict[str, Any]]:
    """Return (authority, sibling fields) for one row-backed C-state.

    Authority requires a retained witness result whose ``freeze_for_C`` is
    not ``None`` and whose selected physical FD rank equals the C graph's FD
    sketch rank, issued on the same device-final-rank source.
    """

    fields_out: dict[str, Any] = {
        "witness_receipt_hash": None,
        "freeze_for_C_hash": None,
        "rank_selection_receipt_hash": None,
        "selected_physical_fd_rank": None,
        "witness_no_C_code": None,
        "consumer_refusal_code": None,
    }
    if consumer_result is None:
        fields_out["consumer_refusal_code"] = "witness_consumer_result_absent"
        return False, fields_out
    if type(consumer_result) is not V1RowBackedDeviceResidentDWitnessConsumerResult:
        raise TypeError("C-state witness authority requires the exact consumer result")
    if (
        consumer_result.device_final_rank_source_receipt_hash
        != device_final_rank_source_receipt_hash
    ):
        fields_out["consumer_refusal_code"] = "witness_source_mismatch"
        return False, fields_out
    if consumer_result.refusal is not None:
        fields_out["consumer_refusal_code"] = consumer_result.refusal.code.value
        return False, fields_out
    witness = consumer_result.witness
    assert witness is not None
    receipt = witness.receipt
    receipt.assert_integrity()
    fields_out["witness_receipt_hash"] = receipt.receipt_hash
    fields_out["witness_no_C_code"] = receipt.no_C_code
    if witness.freeze_for_C is None:
        return False, fields_out
    fields_out["freeze_for_C_hash"] = witness.freeze_for_C.freeze_hash
    selection = consumer_result.selection
    if selection is None:
        fields_out["consumer_refusal_code"] = "rank_selection_absent"
        return False, fields_out
    fields_out["rank_selection_receipt_hash"] = selection.receipt_hash
    fields_out["selected_physical_fd_rank"] = receipt.selected_physical_fd_rank
    if receipt.selected_physical_fd_rank != int(graph_fd_sketch_rank):
        return False, fields_out
    return True, fields_out


# ---------------------------------------------------------------------------
# The live consumer
# ---------------------------------------------------------------------------


def _finite_population_statistics_hash(moments: Any) -> str:
    """Hash of the session's own per-stratum means/M2 (route-scale arrays hashed once)."""

    return content_hash(
        {
            "schema": ROW_BACKED_WITNESS_CONSUMER_SCHEMA + ":finite-population-statistics",
            "moments_state_hash": moments.state_hash,
            "stratum_counts": tuple(moments.stratum_counts),
            "population_stratum_counts": tuple(moments.population_stratum_counts),
            "signed_mean": np.asarray(moments.signed_mean_by_stratum),
            "signed_m2": np.asarray(moments.signed_m2_by_stratum),
        },
        prefix=_PREFIX + "finite-population-statistics-",
    )


def consume_v1_alexnet_row_backed_device_resident_d_final_live_handoff(
    *,
    owner: Any,
    request: Any,
    handoff: Any,
    final_receipt: Mapping[str, Any],
    outcome: Any,
    d_handoff_consumer_registration: Any,
) -> Mapping[str, Any]:
    """Registered pre-eviction consumer of the row-backed device-resident D lane.

    Custody discipline (after the compact lane's v2 consumer): the backend
    owns ``handoff`` and evicts it after this returns; every route-scale
    object created here (ceiling geometry source, ceiling and rank-local KKT
    graphs) is released before return; only compact typed products are
    returned.  This function never raises on a typed absence: the absence is
    recorded on the result and C remains closed.
    """

    from . import modern_v1_alexnet_backend as backend  # local: avoid import cycle

    if not isinstance(final_receipt, Mapping):
        _refuse(V1RowBackedWitnessConsumerRefusalCode.INVALID_TYPE, "final receipt must be a mapping")
    session = getattr(handoff, "session", None)
    _exact(session, V1SampledDeviceResidentDSession, "live D session")
    source = final_receipt.get("device_final_rank_source_receipt")
    _exact(source, V1SampledDeviceFinalRankSourceReceipt, "device-final-rank source")
    source.assert_integrity()
    fd_snapshot = final_receipt.get("fd_snapshot")
    fd_design = final_receipt.get("fd_design")
    moments = final_receipt.get("moments")
    external_descriptor = final_receipt.get("external_descriptor")
    prefix_binding = final_receipt.get("registered_d_prefix_binding")
    source_binding = getattr(handoff, "literal_j_source_binding", None)
    if (
        fd_snapshot is None
        or fd_design is None
        or moments is None
        or external_descriptor is None
        or prefix_binding is None
        or source_binding is None
    ):
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.CUSTODY_MISMATCH,
            "final receipt lacks its live typed custody objects",
        )
    _exact(prefix_binding, V1SampledCompositionRegisteredDPrefixBinding, "prefix binding")
    if (
        fd_snapshot.snapshot_hash != source.fd_snapshot_hash
        or fd_design.design_hash != source.fd_design_hash
        or moments.state_hash != source.corrected_moments_state_hash
        or prefix_binding.binding_hash != source.registered_d_prefix_binding_hash
        or source_binding.binding_hash != source.original_capture_binding_hash
        or request.current_mask_hash != source.current_mask_hash
        or int(request.acquisition_generation_index) != source.proposal_generation
    ):
        _refuse(
            V1RowBackedWitnessConsumerRefusalCode.CUSTODY_MISMATCH,
            "final receipt custody objects do not name the device-final-rank source",
        )
    scorer = getattr(owner, "scorer", None)
    call_count = (lambda: int(getattr(scorer, "call_count", 0)))
    calls_before = call_count()
    ceiling_rank = int(source.requested_physical_rank)
    admitted = tuple(int(value) for value in session.hbm_forecast.hbm_admitted_fd_ranks)
    result = V1RowBackedDeviceResidentDWitnessConsumerResult(
        current_mask_hash=source.current_mask_hash,
        proposal_generation=source.proposal_generation,
        device_final_rank_source_receipt_hash=source.receipt_hash,
        registered_d_prefix_hash=source.registered_d_prefix_hash,
        ceiling_rank=ceiling_rank,
        hbm_admitted_fd_ranks=admitted,
        stopped_at="start",
        refusal=V1RowBackedWitnessConsumerRefusal(
            V1RowBackedWitnessConsumerRefusalCode.ENDOGENOUS_RANK_POLICY_ABSENT,
            "consumer did not start",
        ),
    )

    def _stop(stage: str, refusal: V1RowBackedWitnessConsumerRefusal) -> Mapping[str, Any]:
        result.stopped_at = stage
        result.refusal = refusal
        result.witness = None
        result.selection = None
        result.provider_call_count_before = calls_before
        result.provider_call_count_after = call_count()
        return _package(result)

    ceiling_source = None
    ceiling_graph = None
    deriver = None
    try:
        # 1. v2 registrations naming the device prefix identity.
        policy = getattr(owner, "canonical_fqr_pre_c_policy", None)
        adaptive = getattr(policy, "adaptive_fd_rank_policy", None)
        if type(adaptive) is not V1FQRAdaptiveRankPolicy:
            return _stop(
                "endogenous_rank_registrations",
                V1RowBackedWitnessConsumerRefusal(
                    V1RowBackedWitnessConsumerRefusalCode.ENDOGENOUS_RANK_POLICY_ABSENT,
                    "no registered pre-D adaptive-rank policy supplies the HBM "
                    "numerals for the v2 endogenous-rank registrations",
                    missing_inputs=("canonical_fqr_pre_c_policy.adaptive_fd_rank_policy",),
                ),
            )
        try:
            registrations = register_v1_row_backed_endogenous_rank_registrations_v2(
                adaptive_rank_policy=adaptive,
                fd_design=fd_design,
                registered_d_prefix_binding=prefix_binding,
                population_stratum_counts=session.design.population_stratum_counts,
                observed_stratum_counts=moments.stratum_counts,
                current_mask_hash=source.current_mask_hash,
                route_axis_hash=source.route_axis_hash,
                proposal_generation=source.proposal_generation,
            )
        except V1RowBackedWitnessConsumerRefusal as error:
            return _stop("endogenous_rank_registrations", error)
        result.registrations = registrations
        if registrations.d_prefix.prefix_hash != source.registered_d_prefix_hash:
            return _stop(
                "endogenous_rank_registrations",
                V1RowBackedWitnessConsumerRefusal(
                    V1RowBackedWitnessConsumerRefusalCode.SOURCE_COORDINATE_MISMATCH,
                    "device source was not issued with the v2 D prefix hash; the "
                    "v2 prefix and the device source do not name one identity",
                ),
            )
        # 2. Memory forecast from the same numerals.
        numerals = adaptive.memory_forecasts[0]
        forecast = forecast_v1_device_native_ceiling_rank_memory(
            device_final_rank_source_receipt=source,
            positive_weight_stratum_count=len(fd_design.stratum_names),
            response_rows=adaptive.response_rows,
            physical_workspace_ceiling_bytes=numerals.available_rank_workspace_bytes,
            legality_block_count=numerals.kkt_legality_block_count,
            rounding_tile_size=numerals.kkt_rounding_tile_size,
            maximum_hard_action_count=numerals.maximum_hard_action_count,
            D_packet_staging_device_bytes=numerals.D_packet_staging_device_bytes,
        )
        result.memory_forecast = forecast
        # 3. Ceiling spectrum from the assembled ceiling geometry.  The
        #    admitted pre-freeze source is the KKT chart's input; when the
        #    legacy admission refuses (for example the ceiling rank is not on
        #    the legacy memory ladder) the spectrum is still read from the
        #    same deterministic assembly of the snapshot and the refusal is
        #    carried to the certificate stage as a typed absence.
        current_mask = owner.mask_registry[source.current_mask_hash]
        admission_refusal: str | None = None
        try:
            admission = admit_sampled_device_final_rank_materialization_pre_freeze(
                binding=owner.binding,
                current_mask=current_mask,
                source_origin=external_descriptor,
                moments=moments,
                fd_snapshot=fd_snapshot,
                fd_design=fd_design,
                device_final_rank_source=source,
                adaptive_rank_policy=adaptive,
            )
            ceiling_source = assemble_pre_freeze_sampled_geometry_source(
                admission,
                device=owner.device,
            )
            geometry_result = ceiling_source.geometry_result
        except (ValueError, PermissionError) as error:
            admission_refusal = f"{type(error).__name__}: {error}"
            ceiling_source = None
            geometry_result = (
                V1StratifiedDeviceFDTrustAccumulator.finalize_snapshot_combined_consuming(
                    fd_snapshot,
                    design=fd_design,
                    sketch_rank=fd_snapshot.sketch_rank,
                    device=owner.device,
                )
            )
        fd_receipt = geometry_result.receipt
        eigen_axis = derive_v1_ceiling_fd_eigenvalue_square_axis(
            geometry_result,
            ceiling_rank=ceiling_rank,
        )
        residual_bound = float(fd_receipt.combined_residual_spectral_bound)
        direction_bound = float(min(source.route_count, numerals.maximum_hard_action_count))
        result.eigenvalue_square_axis = eigen_axis
        result.deterministic_fd_residual_spectral_upper_bound = residual_bound
        result.direction_l2_square_upper_bound = direction_bound
        result.direction_bound_source = (
            "registered adaptive-rank policy maximum_hard_action_count clipped by "
            "route count (unit-entry hard actions: ||a||^2 <= action count)"
        )
        # Nested-view hashes (one ceiling state) for the same-VJP fanout.
        binding = issue_v1_sampled_v7_ceiling_rank_geometry_from_device_final_rank_source(
            device_final_rank_source_receipt=source,
            sampled_kkt_endogenous_rank_policy_hash=registrations.sampled_policy.policy_hash,
            eigenvalue_square_axis=eigen_axis,
            deterministic_fd_residual_spectral_upper_bound=residual_bound,
            direction_l2_square_upper_bound=direction_bound,
            memory_forecast=forecast,
            provider_call_count_before_nested_view_derivation=calls_before,
            provider_call_count_after_nested_view_derivation=call_count(),
        )
        if type(binding) is V1FQRCeilingRankNoCRefusal:
            return _stop(
                "ceiling_geometry",
                V1RowBackedWitnessConsumerRefusal(
                    V1RowBackedWitnessConsumerRefusalCode.PHYSICAL_HBM_REFUSED,
                    f"registered physical workspace refuses the ceiling FD state: {binding.reason}",
                ),
            )
        _exact(binding, V1SampledV7CeilingRankDGeometryBinding, "ceiling geometry binding")
        result.geometry_binding_hash = binding.binding_hash
        views = {view.rank: view.view_hash for view in binding.ceiling_geometry.nested_views()}
        try:
            fanout = register_v1_row_backed_same_vjp_fanout_v2(
                registrations=registrations,
                capture_vjp_packet_hashes=source_binding.packet_source_capture_identity_hashes,
                nested_view_hashes_by_rank=views,
                ceiling_rank=ceiling_rank,
            )
        except V1RowBackedWitnessConsumerRefusal as error:
            return _stop("same_vjp_fanout", error)
        result.fanout = fanout
        # 5. Sampling owner radii.
        try:
            radii = derive_v1_row_backed_sampling_owner_radii(
                device_final_rank_source=source,
                d_prefix=registrations.d_prefix,
                per_step_alpha=owner.binding.contract.per_step_alpha,
                finite_population_statistics_hash=_finite_population_statistics_hash(moments),
            )
        except V1RowBackedWitnessConsumerRefusal as error:
            return _stop("sampling_owner_radii", error)
        result.radii = radii
        # 4. Rank-local certificates as truncation views (needs the KKT chart).
        if ceiling_source is None:
            return _stop(
                "rank_local_decision_certificates",
                V1RowBackedWitnessConsumerRefusal(
                    V1RowBackedWitnessConsumerRefusalCode.RANK_LOCAL_DECISION_IMAGE_ABSENT,
                    "the registered adaptive D endpoint admission refused the "
                    f"ceiling state, so no rank-local KKT chart exists: {admission_refusal}",
                    missing_inputs=("sampled_device_final_rank_pre_freeze_admission",),
                ),
            )
        storage = outcome.updated_sufficient_state.storage_descriptor
        score_look = storage.get("proposal_order_score_look_receipt")
        if not isinstance(score_look, Mapping):
            return _stop(
                "rank_local_decision_certificates",
                V1RowBackedWitnessConsumerRefusal(
                    V1RowBackedWitnessConsumerRefusalCode.RANK_LOCAL_DECISION_IMAGE_ABSENT,
                    "no corrected-work confidence-box descriptor is registered on "
                    "this chart; the rank-local KKT source problem cannot be "
                    "prepared, so no rank-local decision certificate exists",
                    missing_inputs=("proposal_order_score_look_receipt",),
                ),
            )
        confidence_descriptor = owner._confidence_box_descriptor_from_score_look(score_look)
        calibration_value = final_receipt.get("mask_calibration")
        if type(calibration_value) is not backend.FixedAffineBCECalibration:
            return _stop(
                "rank_local_decision_certificates",
                V1RowBackedWitnessConsumerRefusal(
                    V1RowBackedWitnessConsumerRefusalCode.RANK_LOCAL_DECISION_IMAGE_ABSENT,
                    "final receipt carries no fixed affine calibration",
                    missing_inputs=("mask_calibration",),
                ),
            )
        calibration = backend._campaign_policy_calibration_view(
            calibration_value,
            calibration_identity=owner.binding.contract.calibration_identity,
        )
        chart_kkt_policy, _shell_design, _hard_fq_design = (
            owner._canonical_fqr_chart_execution_policies(current_mask)
        )
        budgets = owner.extraction.geometric_trust_budgets_for_mask(
            current_mask.mask_hash,
            calibration.scale,
        )
        eligible = np.ascontiguousarray(current_mask.to_bool_array(), dtype=np.bool_)
        eligible.setflags(write=False)
        legality_blocks = owner._sampled_geometric_legality_blocks()
        confidence_root = owner._confidence_box_external_state_root()

        def _build(geometry_source):
            with open_confidence_box_v2_external_state(
                confidence_root,
                confidence_descriptor,
            ) as confidence_view:
                return build_v1_sampled_kkt_branch_family_graph(
                    sampled_geometry_source=geometry_source,
                    confidence_box_view=confidence_view,
                    physical_route_axis=owner.route_axis,
                    fixed_affine_bce_calibration=calibration,
                    geometric_trust_budgets=budgets,
                    eligible_mask=eligible,
                    resource=policy.resource_per_route,
                    policy=chart_kkt_policy,
                    legality_blocks=legality_blocks,
                    minimum_live_per_block=policy.minimum_live_per_block,
                    production_route_scale_threshold=policy.production_route_scale_threshold,
                )

        ceiling_graph = _build(ceiling_source)
        deriver = V1CeilingTruncationRankLocalDeriver(
            ceiling_graph=ceiling_graph,
            endogenous_rank_policy=registrations.memory_policy,
            numerical_enclosure_policy_hash=policy.numerical_enclosure_policy_hash,
            graph_builder=_build,
            ceiling_fd_state_hash=source.fd_snapshot_hash,
            provider_call_counter=call_count,
        )
        # 6. The witness over the exposed lattice, then rank selection.
        try:
            witness = issue_v1_device_native_adaptive_D_rank_sample_KKT_face_witness(
                device_final_rank_source_receipt=source,
                memory_forecast=forecast,
                sampled_kkt_endogenous_rank_policy=registrations.sampled_policy,
                d_prefix=registrations.d_prefix,
                fanout=fanout,
                eigenvalue_square_axis=eigen_axis,
                deterministic_fd_residual_spectral_upper_bound=residual_bound,
                direction_l2_square_upper_bound=direction_bound,
                hbm_admitted_fd_ranks=admitted,
                rank_local_decision_deriver=deriver,
                sampling_owner_receipts=radii.receipts,
                provider_call_count_before=calls_before,
                provider_call_count_after=call_count(),
            )
        except V1DeviceNativeAdaptiveRankSampleWitnessRefusal as error:
            result.truncation_summary = MappingProxyType(deriver.summary())
            return _stop(
                "witness",
                V1RowBackedWitnessConsumerRefusal(
                    V1RowBackedWitnessConsumerRefusalCode.WITNESS_REFUSED,
                    error.message,
                    missing_inputs=error.missing_inputs,
                    witness_code=error.code.value,
                ),
            )
        except V1CeilingTruncationViewRefusal as error:
            result.truncation_summary = MappingProxyType(deriver.summary())
            return _stop(
                "rank_local_decision_certificates",
                V1RowBackedWitnessConsumerRefusal(
                    V1RowBackedWitnessConsumerRefusalCode.RANK_LOCAL_DECISION_IMAGE_ABSENT,
                    error.message,
                    witness_code=error.code.value,
                ),
            )
        result.truncation_summary = MappingProxyType(deriver.summary())
        result.witness = witness
        result.refusal = None
        result.stopped_at = "witness"
        if witness.freeze_for_C is None:
            result.stopped_at = "witness_no_C"
            result.selection = None
        else:
            try:
                result.selection = select_v1_device_native_adaptive_physical_rank(
                    witness=witness,
                    session_design_hash=session.design.design_hash,
                    hbm_forecast_hash=session.hbm_forecast.forecast_hash,
                    device_session=session,
                )
                result.stopped_at = "rank_selection"
            except V1DeviceNativeAdaptiveRankSampleWitnessRefusal as error:
                if error.code is not V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.NO_DECISION_STABLE_RANK:
                    raise
                result.selection = None
                result.stopped_at = "witness_no_admitted_rank"
        result.provider_call_count_before = calls_before
        result.provider_call_count_after = call_count()
        result.__post_init__()
        return _package(result)
    finally:
        # Release every route-scale owner created here; the backend evicts the
        # session after this consumer returns.
        deriver = None
        ceiling_graph = None
        ceiling_source = None


def _package(result: V1RowBackedDeviceResidentDWitnessConsumerResult) -> Mapping[str, Any]:
    payload = result.to_json_payload()
    return MappingProxyType(
        {
            "schema": ROW_BACKED_WITNESS_CONSUMER_SCHEMA,
            ROW_BACKED_WITNESS_CONSUMER_RESULT_KEY: result,
            "summary": payload,
            "consumer_result_hash": content_hash(payload, prefix=_PREFIX + "result-"),
        }
    )


__all__ = [
    "ROW_BACKED_C_STATE_WITNESS_SIBLING_SCHEMA",
    "ROW_BACKED_SAMPLING_OWNER_CENSUS_RULE",
    "ROW_BACKED_WITNESS_CONSUMER_RESULT_KEY",
    "ROW_BACKED_WITNESS_CONSUMER_SCHEMA",
    "V1RowBackedCStateWitnessSiblingReceipt",
    "V1RowBackedDeviceResidentDWitnessConsumerResult",
    "V1RowBackedEndogenousRankRegistrationsV2",
    "V1RowBackedSamplingOwnerRadiiResult",
    "V1RowBackedWitnessConsumerRefusal",
    "V1RowBackedWitnessConsumerRefusalCode",
    "consume_v1_alexnet_row_backed_device_resident_d_final_live_handoff",
    "derive_v1_row_backed_sampling_owner_radii",
    "evaluate_v1_row_backed_c_state_witness_authority",
    "register_v1_row_backed_endogenous_rank_registrations_v2",
    "register_v1_row_backed_same_vjp_fanout_v2",
]
