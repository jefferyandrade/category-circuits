"""Ceiling-rank nested geometry adapter for sampled-v7 D stability.

This module is the narrow live-path bridge between the device-resident D
source and the ceiling-rank F/Q geometry contract.  It deliberately does not
run CUDA, solve KKT, sample ImageNet, open C/K/O, or commit masks.  Its job is
only to verify that:

* one device-final-rank source receipt supplies the ceiling FD state for the
  registered D prefix;
* lower ranks are viewed as nested prefixes of that ceiling state, not as
  zero-padded or independently sketched lower-rank states;
* D-sampling, FD-tail, and numerical invariance are imported only from the
  existing source-issued robust KKT decision image/certificate; and
* C can open only through the minimum invariant ceiling-rank freeze.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from enum import Enum
import math
from typing import Final, NoReturn

from .contracts import content_hash
from .modern_v1_fqr_adaptive_rank import (
    FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA,
    V1FQRAdaptiveRankDecisionCertificate,
)
from .modern_v1_fqr_ceiling_rank_geometry import (
    V1FQRCeilingRankDecisionImage,
    V1FQRCeilingRankFreezeForC,
    V1FQRCeilingRankGeometry,
    V1FQRCeilingRankMemoryForecast,
    V1FQRCeilingRankNestedView,
    V1FQRCeilingRankNextAction,
    V1FQRCeilingRankNoCCode,
    V1FQRCeilingRankNoCRefusal,
    freeze_or_refuse_v1_fqr_ceiling_rank_for_C,
    issue_v1_fqr_ceiling_rank_decision_image,
    issue_v1_fqr_ceiling_rank_geometry,
)
from .modern_v1_sampled_device_final_rank_source import (
    V1SampledDeviceFinalRankSourceReceipt,
)
from .modern_v1_sampled_kkt_device_shell_bridge import (
    SAMPLED_KKT_SOURCE_DECISION_IMAGE_V2_SCHEMA,
    V1SampledKKTSourceDecisionImageV2,
)


SAMPLED_V7_CEILING_RANK_D_GEOMETRY_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_ceiling_rank_D_geometry_binding_v1"
)
SAMPLED_V7_CEILING_RANK_SOURCE_DECISION_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_ceiling_rank_source_decision_projection_v1"
)
SAMPLED_V7_CEILING_RANK_COMPONENT_STABILITY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_ceiling_rank_component_stability_receipt_v1"
)
SAMPLED_V7_CEILING_RANK_D_STABILITY_FREEZE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_ceiling_rank_D_stability_freeze_v1"
)
SAMPLED_V7_CEILING_RANK_D_STABILITY_AUTHORITY: Final = (
    "device_final_rank_to_ceiling_rank_D_stability_adapter_only_no_sampling_"
    "no_C_K_O_commit_terminal_runtime_or_pareto_authority_except_exact_C_"
    "freeze_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-ceiling-rank-D-stability-"
_GEOMETRY_BINDING_TOKEN: Final = object()
_DECISION_PROJECTION_TOKEN: Final = object()
_COMPONENT_RECEIPT_TOKEN: Final = object()
_FREEZE_TOKEN: Final = object()


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty exact string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise TypeError(f"{name} must be one exact integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _source_capture_vjp_lineage_hash(
    source: V1SampledDeviceFinalRankSourceReceipt,
) -> str:
    return content_hash(
        {
            "schema": (
                "partition_free_v1_sampled_v7_ceiling_rank_source_capture_"
                "vjp_lineage_v1"
            ),
            "device_final_rank_source_receipt_hash": source.receipt_hash,
            "registered_d_prefix_hash": source.registered_d_prefix_hash,
            "context_axis_hash": source.context_axis_hash,
            "draw_axis_hash": source.draw_axis_hash,
            "packet_axis_hash": source.packet_axis_hash,
            "source_measurement_axis_hash": source.source_measurement_axis_hash,
            "literal_j_stream_identity_hash": source.literal_j_stream_identity_hash,
            "corrected_a_stream_identity_hash": (
                source.corrected_a_stream_identity_hash
            ),
            "original_capture_batch_count": source.original_capture_batch_count,
            "original_vjp_count": source.original_vjp_count,
            "provider_reexecution_count": source.provider_reexecution_count,
            "capture_reexecution_count": source.capture_reexecution_count,
            "vjp_reexecution_count": source.vjp_reexecution_count,
        },
        prefix=_PREFIX + "capture-vjp-lineage-",
    )


def _source_selected_action_family_hash(
    certificate: V1FQRAdaptiveRankDecisionCertificate,
) -> str:
    robust = certificate.robust_next_decision_hash
    if robust is not None:
        return robust
    return content_hash(
        {
            "schema": (
                "partition_free_v1_sampled_v7_ceiling_rank_unstable_selected_"
                "action_family_v1"
            ),
            "source_decision_certificate_hash": certificate.certificate_hash,
            "rank": certificate.rank,
            "branch_selected_action_hashes": tuple(
                (branch.branch, branch.selected_packed_action_hash)
                for branch in certificate.branch_decisions
            ),
            "branch_endpoint_status_axis_hashes": tuple(
                branch.endpoint_status_axis_hash
                for branch in certificate.branch_decisions
            ),
        },
        prefix=_PREFIX + "selected-action-family-",
    )


class V1SampledV7CeilingRankComponentRole(str, Enum):
    """Source-stability components consumed by ceiling-rank D."""

    D_SAMPLING = "D_sampling"
    FD_TAIL = "FD_tail"
    KKT_NUMERICAL = "KKT_numerical"


_COMPONENT_ROLE_ORDER: Final = (
    V1SampledV7CeilingRankComponentRole.D_SAMPLING,
    V1SampledV7CeilingRankComponentRole.FD_TAIL,
    V1SampledV7CeilingRankComponentRole.KKT_NUMERICAL,
)


@dataclass(frozen=True, slots=True)
class V1SampledV7CeilingRankComponentStabilityReceipt:
    """Exact source component receipt whose invariant flag replays from a bound."""

    role: V1SampledV7CeilingRankComponentRole
    geometry_hash: str
    nested_view_hash: str
    source_decision_certificate_hash: str
    source_derivation_receipt_hash: str
    rank: int
    invariant_margin_lower_bound: float
    uncertainty_radius_upper_bound: float
    decision_invariant: bool
    _issuance_token: InitVar[object]
    source_derived_component_receipt: bool = True
    caller_boolean_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SAMPLED_V7_CEILING_RANK_COMPONENT_STABILITY_RECEIPT_SCHEMA
    authority_scope: str = SAMPLED_V7_CEILING_RANK_D_STABILITY_AUTHORITY
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _COMPONENT_RECEIPT_TOKEN:
            raise PermissionError("ceiling-rank component receipt is issuer-only")
        if type(self.role) is not V1SampledV7CeilingRankComponentRole:
            raise TypeError("component stability role must be exact")
        for name in (
            "geometry_hash",
            "nested_view_hash",
            "source_decision_certificate_hash",
            "source_derivation_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        rank = _nonnegative_int(self.rank, name="component stability rank")
        if rank == 0:
            raise ValueError("component stability rank must be positive")
        margin = float(self.invariant_margin_lower_bound)
        radius = float(self.uncertainty_radius_upper_bound)
        if (
            not math.isfinite(margin)
            or not math.isfinite(radius)
            or margin < 0.0
            or radius < 0.0
        ):
            raise ValueError(
                "component margin and uncertainty must be finite and nonnegative"
            )
        expected_invariant = margin > radius
        if type(self.decision_invariant) is not bool:
            raise TypeError("component decision invariant must be exact bool")
        if self.decision_invariant is not expected_invariant:
            raise ValueError("component decision invariant does not replay its bound")
        flags = (
            self.source_derived_component_receipt,
            self.caller_boolean_authority,
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("component stability flags must be exact")
        if flags != (True, False, False, False, False, False):
            raise PermissionError("component stability receipt crossed authority")
        if (
            self.schema != SAMPLED_V7_CEILING_RANK_COMPONENT_STABILITY_RECEIPT_SCHEMA
            or self.authority_scope != SAMPLED_V7_CEILING_RANK_D_STABILITY_AUTHORITY
        ):
            raise ValueError("component stability receipt schema is invalid")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "invariant_margin_lower_bound", margin)
        object.__setattr__(self, "uncertainty_radius_upper_bound", radius)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "authority_scope": self.authority_scope,
            "role": self.role.value,
            "geometry_hash": self.geometry_hash,
            "nested_view_hash": self.nested_view_hash,
            "source_decision_certificate_hash": (
                self.source_decision_certificate_hash
            ),
            "source_derivation_receipt_hash": self.source_derivation_receipt_hash,
            "rank": self.rank,
            "invariant_margin_lower_bound": self.invariant_margin_lower_bound,
            "uncertainty_radius_upper_bound": self.uncertainty_radius_upper_bound,
            "decision_invariant": self.decision_invariant,
            "source_derived_component_receipt": self.source_derived_component_receipt,
            "caller_boolean_authority": self.caller_boolean_authority,
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(),
            prefix=_PREFIX + "component-stability-receipt-",
        )
        cached = self._receipt_hash
        if cached is not None and cached != observed:
            raise ValueError("ceiling-rank component stability receipt was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("ceiling-rank component stability receipt hash mismatch")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


def issue_v1_sampled_v7_ceiling_rank_component_stability_receipt(
    *,
    geometry_binding: "V1SampledV7CeilingRankDGeometryBinding",
    nested_view: V1FQRCeilingRankNestedView,
    source_decision_certificate: V1FQRAdaptiveRankDecisionCertificate,
    role: V1SampledV7CeilingRankComponentRole,
    source_derivation_receipt_hash: str,
    invariant_margin_lower_bound: float,
    uncertainty_radius_upper_bound: float,
) -> V1SampledV7CeilingRankComponentStabilityReceipt:
    """Issue one component-stability receipt by replaying margin vs radius."""

    if type(geometry_binding) is not V1SampledV7CeilingRankDGeometryBinding:
        raise TypeError("component receipt requires exact geometry binding")
    if type(nested_view) is not V1FQRCeilingRankNestedView:
        raise TypeError("component receipt requires exact nested view")
    if type(source_decision_certificate) is not V1FQRAdaptiveRankDecisionCertificate:
        raise TypeError("component receipt requires exact source certificate")
    if type(role) is not V1SampledV7CeilingRankComponentRole:
        raise TypeError("component receipt role must be exact")
    geometry_hash = geometry_binding.ceiling_geometry.geometry_hash
    if (
        nested_view.geometry_hash != geometry_hash
        or source_decision_certificate.rank != nested_view.rank
        or source_decision_certificate.current_mask_hash
        != geometry_binding.ceiling_geometry.current_mask_hash
        or source_decision_certificate.route_axis_hash
        != geometry_binding.ceiling_geometry.route_axis_hash
        or source_decision_certificate.route_count
        != geometry_binding.ceiling_geometry.route_count
        or source_decision_certificate.proposal_generation
        != geometry_binding.ceiling_geometry.proposal_generation
    ):
        raise ValueError("component receipt source coordinates do not close")
    return V1SampledV7CeilingRankComponentStabilityReceipt(
        role=role,
        geometry_hash=geometry_hash,
        nested_view_hash=nested_view.view_hash,
        source_decision_certificate_hash=(
            source_decision_certificate.certificate_hash
        ),
        source_derivation_receipt_hash=source_derivation_receipt_hash,
        rank=nested_view.rank,
        invariant_margin_lower_bound=invariant_margin_lower_bound,
        uncertainty_radius_upper_bound=uncertainty_radius_upper_bound,
        decision_invariant=(
            float(invariant_margin_lower_bound) > float(uncertainty_radius_upper_bound)
        ),
        _issuance_token=_COMPONENT_RECEIPT_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7CeilingRankDGeometryBinding:
    """Device-final-rank source bound to one ceiling-rank geometry."""

    device_final_rank_source_receipt: V1SampledDeviceFinalRankSourceReceipt
    ceiling_geometry: V1FQRCeilingRankGeometry
    provider_call_count_before_nested_view_derivation: int
    provider_call_count_after_nested_view_derivation: int
    _issuance_token: InitVar[object]
    one_ceiling_fd_state_constructed: bool = True
    lower_rank_views_are_prefix_metadata: bool = True
    no_lower_rank_zero_padding: bool = True
    no_independent_lower_rank_sketches: bool = True
    no_extra_vjp_for_nested_views: bool = True
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_V7_CEILING_RANK_D_GEOMETRY_BINDING_SCHEMA
    authority_scope: str = SAMPLED_V7_CEILING_RANK_D_STABILITY_AUTHORITY
    _binding_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _GEOMETRY_BINDING_TOKEN:
            raise PermissionError("ceiling-rank D geometry binding is issuer-only")
        if type(self.device_final_rank_source_receipt) is not (
            V1SampledDeviceFinalRankSourceReceipt
        ):
            raise TypeError("geometry binding requires exact device-final-rank source")
        if type(self.ceiling_geometry) is not V1FQRCeilingRankGeometry:
            raise TypeError("geometry binding requires exact ceiling-rank geometry")
        source = self.device_final_rank_source_receipt
        geometry = self.ceiling_geometry
        source.assert_integrity()
        geometry_hash = geometry.assert_integrity()
        if (
            not geometry.memory_forecast.admitted
            or geometry.ceiling_rank != source.requested_physical_rank
            or geometry.ceiling_fd_state_hash != source.fd_snapshot_hash
            or geometry.registered_d_prefix_hash != source.registered_d_prefix_hash
            or geometry.route_axis_hash != source.route_axis_hash
            or geometry.current_mask_hash != source.current_mask_hash
            or geometry.proposal_generation != source.proposal_generation
            or geometry.route_count != source.route_count
            or geometry.source_capture_vjp_lineage_hash
            != _source_capture_vjp_lineage_hash(source)
        ):
            raise ValueError("device-final-rank source does not close ceiling geometry")
        before = _nonnegative_int(
            self.provider_call_count_before_nested_view_derivation,
            name="provider call count before nested-view derivation",
        )
        after = _nonnegative_int(
            self.provider_call_count_after_nested_view_derivation,
            name="provider call count after nested-view derivation",
        )
        if before != after:
            raise PermissionError(
                "nested lower-rank views cannot trigger provider calls or VJPs"
            )
        if (
            source.provider_reexecution_count != 0
            or source.capture_reexecution_count != 0
            or source.vjp_reexecution_count != 0
        ):
            raise PermissionError(
                "device-final-rank ceiling source cannot include reexecution"
            )
        flags = (
            self.one_ceiling_fd_state_constructed,
            self.lower_rank_views_are_prefix_metadata,
            self.no_lower_rank_zero_padding,
            self.no_independent_lower_rank_sketches,
            self.no_extra_vjp_for_nested_views,
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("ceiling-rank D geometry binding flags must be exact")
        if flags != (True, True, True, True, True, False, False, False, False, False, False):
            raise PermissionError("ceiling-rank D geometry binding crossed authority")
        if (
            self.schema != SAMPLED_V7_CEILING_RANK_D_GEOMETRY_BINDING_SCHEMA
            or self.authority_scope != SAMPLED_V7_CEILING_RANK_D_STABILITY_AUTHORITY
        ):
            raise ValueError("ceiling-rank D geometry binding schema is invalid")
        object.__setattr__(
            self,
            "provider_call_count_before_nested_view_derivation",
            before,
        )
        object.__setattr__(
            self,
            "provider_call_count_after_nested_view_derivation",
            after,
        )
        self.assert_integrity(expected_geometry_hash=geometry_hash)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "authority_scope": self.authority_scope,
            "device_final_rank_source_receipt_hash": (
                self.device_final_rank_source_receipt.receipt_hash
            ),
            "ceiling_geometry_hash": self.ceiling_geometry.geometry_hash,
            "memory_forecast_hash": (
                self.ceiling_geometry.memory_forecast.forecast_hash
            ),
            "provider_call_count_before_nested_view_derivation": (
                self.provider_call_count_before_nested_view_derivation
            ),
            "provider_call_count_after_nested_view_derivation": (
                self.provider_call_count_after_nested_view_derivation
            ),
            "one_ceiling_fd_state_constructed": (
                self.one_ceiling_fd_state_constructed
            ),
            "lower_rank_views_are_prefix_metadata": (
                self.lower_rank_views_are_prefix_metadata
            ),
            "no_lower_rank_zero_padding": self.no_lower_rank_zero_padding,
            "no_independent_lower_rank_sketches": (
                self.no_independent_lower_rank_sketches
            ),
            "no_extra_vjp_for_nested_views": self.no_extra_vjp_for_nested_views,
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }

    @property
    def binding_hash(self) -> str:
        cached = self._binding_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_geometry_hash: str | None = None) -> str:
        self.device_final_rank_source_receipt.assert_integrity()
        geometry_hash = self.ceiling_geometry.assert_integrity()
        if expected_geometry_hash is not None and geometry_hash != expected_geometry_hash:
            raise ValueError("ceiling-rank geometry hash does not match binding")
        observed = content_hash(self._payload(), prefix=_PREFIX + "geometry-binding-")
        cached = self._binding_hash
        if cached is not None and cached != observed:
            raise ValueError("ceiling-rank D geometry binding was tampered")
        object.__setattr__(self, "_binding_hash", observed)
        return observed


def issue_v1_sampled_v7_ceiling_rank_geometry_from_device_final_rank_source(
    *,
    device_final_rank_source_receipt: V1SampledDeviceFinalRankSourceReceipt,
    sampled_kkt_endogenous_rank_policy_hash: str,
    eigenvalue_square_axis: Sequence[float],
    deterministic_fd_residual_spectral_upper_bound: float,
    direction_l2_square_upper_bound: float,
    memory_forecast: V1FQRCeilingRankMemoryForecast,
    provider_call_count_before_nested_view_derivation: int = 0,
    provider_call_count_after_nested_view_derivation: int = 0,
) -> V1SampledV7CeilingRankDGeometryBinding | V1FQRCeilingRankNoCRefusal:
    """Issue the one-ceiling-state D geometry binding.

    Physical HBM refusal returns a typed no-C object.  Stale or mismatched
    source coordinates raise because those indicate a broken live-source seam,
    not an admissible uncertainty-resolution step.
    """

    if type(device_final_rank_source_receipt) is not (
        V1SampledDeviceFinalRankSourceReceipt
    ):
        raise TypeError("ceiling-rank geometry requires exact device-final source")
    if type(memory_forecast) is not V1FQRCeilingRankMemoryForecast:
        raise TypeError("ceiling-rank geometry requires exact memory forecast")
    source = device_final_rank_source_receipt
    source.assert_integrity()
    policy_hash = _identity(
        sampled_kkt_endogenous_rank_policy_hash,
        name="sampled KKT endogenous-rank policy hash",
    )
    if memory_forecast.ceiling_rank != source.requested_physical_rank:
        raise ValueError("memory forecast ceiling rank differs from device source")
    if memory_forecast.route_count != source.route_count:
        raise ValueError("memory forecast route count differs from device source")
    if not memory_forecast.admitted:
        return V1FQRCeilingRankNoCRefusal(
            code=V1FQRCeilingRankNoCCode.MAXIMUM_CEILING_RANK_AMBIGUOUS,
            reason="registered physical workspace refuses the ceiling FD state",
            required_action=V1FQRCeilingRankNextAction.PHYSICAL_NO_C_REFUSAL,
            geometry_hash=None,
            current_rank=source.requested_physical_rank,
        )
    lineage = _source_capture_vjp_lineage_hash(source)
    geometry = issue_v1_fqr_ceiling_rank_geometry(
        policy_hash=policy_hash,
        registered_d_prefix_hash=source.registered_d_prefix_hash,
        source_capture_vjp_lineage_hash=lineage,
        ceiling_fd_state_hash=source.fd_snapshot_hash,
        route_axis_hash=source.route_axis_hash,
        current_mask_hash=source.current_mask_hash,
        proposal_generation=source.proposal_generation,
        route_count=source.route_count,
        ceiling_rank=source.requested_physical_rank,
        eigenvalue_square_axis=eigenvalue_square_axis,
        deterministic_fd_residual_spectral_upper_bound=(
            deterministic_fd_residual_spectral_upper_bound
        ),
        direction_l2_square_upper_bound=direction_l2_square_upper_bound,
        memory_forecast=memory_forecast,
    )
    return V1SampledV7CeilingRankDGeometryBinding(
        device_final_rank_source_receipt=source,
        ceiling_geometry=geometry,
        provider_call_count_before_nested_view_derivation=(
            provider_call_count_before_nested_view_derivation
        ),
        provider_call_count_after_nested_view_derivation=(
            provider_call_count_after_nested_view_derivation
        ),
        _issuance_token=_GEOMETRY_BINDING_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7CeilingRankSourceDecisionProjection:
    """Source-issued KKT image projected onto one ceiling-rank nested view."""

    geometry_binding: V1SampledV7CeilingRankDGeometryBinding
    nested_view: V1FQRCeilingRankNestedView
    source_decision_image: V1SampledKKTSourceDecisionImageV2
    component_stability_receipts: tuple[
        V1SampledV7CeilingRankComponentStabilityReceipt, ...
    ]
    ceiling_decision_image: V1FQRCeilingRankDecisionImage
    _issuance_token: InitVar[object]
    caller_invariance_booleans_accepted: bool = False
    source_issued_robust_kkt_certificate_bound: bool = True
    source_component_receipts_bound: bool = True
    old_rank_local_fd_state_not_geometry_authority: bool = True
    no_independent_lower_rank_sketches: bool = True
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SAMPLED_V7_CEILING_RANK_SOURCE_DECISION_PROJECTION_SCHEMA
    authority_scope: str = SAMPLED_V7_CEILING_RANK_D_STABILITY_AUTHORITY
    _projection_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _DECISION_PROJECTION_TOKEN:
            raise PermissionError("ceiling-rank source projection is issuer-only")
        if type(self.geometry_binding) is not V1SampledV7CeilingRankDGeometryBinding:
            raise TypeError("source projection requires exact geometry binding")
        if type(self.nested_view) is not V1FQRCeilingRankNestedView:
            raise TypeError("source projection requires exact nested view")
        if type(self.source_decision_image) is not V1SampledKKTSourceDecisionImageV2:
            raise TypeError("source projection requires exact source decision image")
        if type(self.ceiling_decision_image) is not V1FQRCeilingRankDecisionImage:
            raise TypeError("source projection requires exact ceiling decision image")
        component_receipts = tuple(self.component_stability_receipts)
        if tuple(receipt.role for receipt in component_receipts) != _COMPONENT_ROLE_ORDER:
            raise TypeError(
                "source projection requires exact D, FD-tail, and KKT component receipts"
            )
        binding_hash = self.geometry_binding.assert_integrity()
        geometry = self.geometry_binding.ceiling_geometry
        geometry_hash = geometry.geometry_hash
        view_hash = self.nested_view.view_hash
        source_image_hash = self.source_decision_image.assert_integrity()
        ceiling_image_hash = self.ceiling_decision_image.assert_integrity()
        certificate = self.source_decision_image.source_decision_certificate
        certificate.assert_integrity()
        for receipt in component_receipts:
            receipt.assert_integrity()
        component_hashes = tuple(receipt.receipt_hash for receipt in component_receipts)
        component_invariance = tuple(
            receipt.decision_invariant for receipt in component_receipts
        )
        if (
            self.source_decision_image.schema
            != SAMPLED_KKT_SOURCE_DECISION_IMAGE_V2_SCHEMA
            or certificate.schema != FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA
            or self.nested_view.geometry_hash != geometry_hash
            or self.ceiling_decision_image.geometry_hash != geometry_hash
            or self.ceiling_decision_image.nested_view_hash != view_hash
            or self.source_decision_image.endogenous_rank_policy_hash
            != geometry.policy_hash
            or self.source_decision_image.registered_d_prefix_hash
            != geometry.registered_d_prefix_hash
            or self.source_decision_image.rank != self.nested_view.rank
            or self.ceiling_decision_image.rank != self.nested_view.rank
            or certificate.current_mask_hash != geometry.current_mask_hash
            or certificate.route_axis_hash != geometry.route_axis_hash
            or certificate.route_count != geometry.route_count
            or certificate.proposal_generation != geometry.proposal_generation
            or self.ceiling_decision_image.decision_receipt_hash
            != source_image_hash
            or self.ceiling_decision_image.selected_packed_action_hash
            != _source_selected_action_family_hash(certificate)
            or self.source_decision_image.source_component_receipt_hashes
            != component_hashes
            or tuple(
                receipt.geometry_hash for receipt in component_receipts
            )
            != (geometry_hash, geometry_hash, geometry_hash)
            or tuple(
                receipt.nested_view_hash for receipt in component_receipts
            )
            != (view_hash, view_hash, view_hash)
            or tuple(
                receipt.source_decision_certificate_hash
                for receipt in component_receipts
            )
            != (
                certificate.certificate_hash,
                certificate.certificate_hash,
                certificate.certificate_hash,
            )
            or tuple(receipt.rank for receipt in component_receipts)
            != (self.nested_view.rank, self.nested_view.rank, self.nested_view.rank)
            or component_invariance
            != (
                self.ceiling_decision_image.D_sampling_decision_invariant,
                self.ceiling_decision_image.FD_tail_decision_invariant,
                self.ceiling_decision_image.KKT_numerical_decision_invariant,
            )
            or component_invariance
            != (
                self.source_decision_image.D_sampling_decision_invariant,
                self.source_decision_image.FD_truncation_decision_invariant,
                self.source_decision_image.KKT_numerical_decision_invariant,
            )
        ):
            raise ValueError("source KKT decision image does not close ceiling view")
        flags = (
            self.caller_invariance_booleans_accepted,
            self.source_issued_robust_kkt_certificate_bound,
            self.source_component_receipts_bound,
            self.old_rank_local_fd_state_not_geometry_authority,
            self.no_independent_lower_rank_sketches,
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("ceiling-rank source projection flags must be exact")
        if flags != (False, True, True, True, True, False, False, False, False):
            raise PermissionError("ceiling-rank source projection crossed authority")
        if (
            self.schema != SAMPLED_V7_CEILING_RANK_SOURCE_DECISION_PROJECTION_SCHEMA
            or self.authority_scope != SAMPLED_V7_CEILING_RANK_D_STABILITY_AUTHORITY
        ):
            raise ValueError("ceiling-rank source projection schema is invalid")
        self.assert_integrity(
            expected_binding_hash=binding_hash,
            expected_source_image_hash=source_image_hash,
            expected_ceiling_image_hash=ceiling_image_hash,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "authority_scope": self.authority_scope,
            "geometry_binding_hash": self.geometry_binding.binding_hash,
            "geometry_hash": self.geometry_binding.ceiling_geometry.geometry_hash,
            "nested_view_hash": self.nested_view.view_hash,
            "rank": self.nested_view.rank,
            "source_decision_image_hash": self.source_decision_image.image_hash,
            "source_decision_certificate_hash": (
                self.source_decision_image.source_decision_certificate.certificate_hash
            ),
            "source_component_receipt_hashes": (
                self.source_decision_image.source_component_receipt_hashes
            ),
            "component_stability_receipt_hashes": tuple(
                receipt.receipt_hash for receipt in self.component_stability_receipts
            ),
            "component_decision_invariants": tuple(
                receipt.decision_invariant
                for receipt in self.component_stability_receipts
            ),
            "source_rank_local_fd_state_hash": (
                self.source_decision_image.rank_local_fd_state_hash
            ),
            "ceiling_decision_image_hash": self.ceiling_decision_image.image_hash,
            "caller_invariance_booleans_accepted": (
                self.caller_invariance_booleans_accepted
            ),
            "source_issued_robust_kkt_certificate_bound": (
                self.source_issued_robust_kkt_certificate_bound
            ),
            "source_component_receipts_bound": self.source_component_receipts_bound,
            "old_rank_local_fd_state_not_geometry_authority": (
                self.old_rank_local_fd_state_not_geometry_authority
            ),
            "no_independent_lower_rank_sketches": (
                self.no_independent_lower_rank_sketches
            ),
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }

    @property
    def projection_hash(self) -> str:
        cached = self._projection_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(
        self,
        *,
        expected_binding_hash: str | None = None,
        expected_source_image_hash: str | None = None,
        expected_ceiling_image_hash: str | None = None,
    ) -> str:
        binding_hash = self.geometry_binding.assert_integrity()
        source_hash = self.source_decision_image.assert_integrity()
        ceiling_hash = self.ceiling_decision_image.assert_integrity()
        if expected_binding_hash is not None and binding_hash != expected_binding_hash:
            raise ValueError("ceiling-rank source projection binding hash mismatch")
        if (
            expected_source_image_hash is not None
            and source_hash != expected_source_image_hash
        ):
            raise ValueError("ceiling-rank source projection source hash mismatch")
        if (
            expected_ceiling_image_hash is not None
            and ceiling_hash != expected_ceiling_image_hash
        ):
            raise ValueError("ceiling-rank source projection image hash mismatch")
        observed = content_hash(
            self._payload(),
            prefix=_PREFIX + "source-decision-projection-",
        )
        cached = self._projection_hash
        if cached is not None and cached != observed:
            raise ValueError("ceiling-rank source decision projection was tampered")
        object.__setattr__(self, "_projection_hash", observed)
        return observed


def issue_v1_sampled_v7_ceiling_rank_decision_projection_from_source_image(
    *,
    geometry_binding: V1SampledV7CeilingRankDGeometryBinding,
    source_decision_image: V1SampledKKTSourceDecisionImageV2,
    component_stability_receipts: Sequence[
        V1SampledV7CeilingRankComponentStabilityReceipt
    ],
) -> V1SampledV7CeilingRankSourceDecisionProjection:
    """Project one source-issued robust KKT image into ceiling-rank geometry."""

    if type(geometry_binding) is not V1SampledV7CeilingRankDGeometryBinding:
        raise TypeError("source projection requires exact geometry binding")
    if type(source_decision_image) is not V1SampledKKTSourceDecisionImageV2:
        raise TypeError("source projection requires exact source decision image")
    component_receipts = tuple(component_stability_receipts)
    if (
        tuple(receipt.role for receipt in component_receipts)
        != _COMPONENT_ROLE_ORDER
    ):
        raise TypeError(
            "source projection requires exact D, FD-tail, and KKT component receipts"
        )
    geometry_binding.assert_integrity()
    source_decision_image.assert_integrity()
    for receipt in component_receipts:
        receipt.assert_integrity()
    views = {
        view.rank: view
        for view in geometry_binding.ceiling_geometry.nested_views()
    }
    rank = source_decision_image.rank
    if rank not in views:
        raise ValueError("source decision rank is not exposed by ceiling geometry")
    nested_view = views[rank]
    certificate = source_decision_image.source_decision_certificate
    ceiling_image = issue_v1_fqr_ceiling_rank_decision_image(
        geometry=geometry_binding.ceiling_geometry,
        nested_view=nested_view,
        selected_packed_action_hash=_source_selected_action_family_hash(certificate),
        decision_receipt_hash=source_decision_image.image_hash,
        D_sampling_decision_invariant=(
            component_receipts[0].decision_invariant
        ),
        FD_tail_decision_invariant=(
            component_receipts[1].decision_invariant
        ),
        KKT_numerical_decision_invariant=(
            component_receipts[2].decision_invariant
        ),
    )
    return V1SampledV7CeilingRankSourceDecisionProjection(
        geometry_binding=geometry_binding,
        nested_view=nested_view,
        source_decision_image=source_decision_image,
        component_stability_receipts=component_receipts,
        ceiling_decision_image=ceiling_image,
        _issuance_token=_DECISION_PROJECTION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7CeilingRankDStabilityFreezeForC:
    """C-only freeze derived from source-issued KKT decisions."""

    geometry_binding: V1SampledV7CeilingRankDGeometryBinding
    source_decision_projections: tuple[V1SampledV7CeilingRankSourceDecisionProjection, ...]
    ceiling_freeze: V1FQRCeilingRankFreezeForC
    _issuance_token: InitVar[object]
    minimum_invariant_rank_selected: bool = True
    C_stream_open_authority: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_V7_CEILING_RANK_D_STABILITY_FREEZE_SCHEMA
    authority_scope: str = SAMPLED_V7_CEILING_RANK_D_STABILITY_AUTHORITY
    _freeze_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _FREEZE_TOKEN:
            raise PermissionError("ceiling-rank D-stability freeze is issuer-only")
        if type(self.geometry_binding) is not V1SampledV7CeilingRankDGeometryBinding:
            raise TypeError("D-stability freeze requires exact geometry binding")
        if type(self.ceiling_freeze) is not V1FQRCeilingRankFreezeForC:
            raise TypeError("D-stability freeze requires exact ceiling freeze")
        binding_hash = self.geometry_binding.assert_integrity()
        projections = tuple(self.source_decision_projections)
        if not projections or any(
            type(value) is not V1SampledV7CeilingRankSourceDecisionProjection
            for value in projections
        ):
            raise TypeError("D-stability freeze requires exact source projections")
        for value in projections:
            value.assert_integrity(expected_binding_hash=binding_hash)
        image_hashes = tuple(
            value.ceiling_decision_image.image_hash for value in projections
        )
        if (
            self.ceiling_freeze.geometry.geometry_hash
            != self.geometry_binding.ceiling_geometry.geometry_hash
            or tuple(
                image.image_hash for image in self.ceiling_freeze.evaluated_decision_images
            )
            != image_hashes
            or self.ceiling_freeze.frozen_rank != projections[-1].nested_view.rank
            or self.ceiling_freeze.frozen_decision_image_hash != image_hashes[-1]
        ):
            raise ValueError("ceiling-rank D-stability freeze does not close")
        flags = (
            self.minimum_invariant_rank_selected,
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("ceiling-rank D-stability freeze flags must be exact")
        if flags != (True, True, False, False, False, False, False):
            raise PermissionError("ceiling-rank D-stability freeze crossed authority")
        if (
            self.schema != SAMPLED_V7_CEILING_RANK_D_STABILITY_FREEZE_SCHEMA
            or self.authority_scope != SAMPLED_V7_CEILING_RANK_D_STABILITY_AUTHORITY
        ):
            raise ValueError("ceiling-rank D-stability freeze schema is invalid")
        self.assert_integrity()

    @property
    def frozen_rank(self) -> int:
        return self.ceiling_freeze.frozen_rank

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "authority_scope": self.authority_scope,
            "geometry_binding_hash": self.geometry_binding.binding_hash,
            "source_decision_projection_hashes": tuple(
                value.projection_hash for value in self.source_decision_projections
            ),
            "ceiling_freeze_hash": self.ceiling_freeze.freeze_hash,
            "frozen_rank": self.frozen_rank,
            "minimum_invariant_rank_selected": self.minimum_invariant_rank_selected,
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }

    @property
    def freeze_hash(self) -> str:
        cached = self._freeze_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self) -> str:
        self.geometry_binding.assert_integrity()
        for value in self.source_decision_projections:
            value.assert_integrity(
                expected_binding_hash=self.geometry_binding.binding_hash
            )
        self.ceiling_freeze.assert_integrity()
        observed = content_hash(self._payload(), prefix=_PREFIX + "freeze-")
        cached = self._freeze_hash
        if cached is not None and cached != observed:
            raise ValueError("ceiling-rank D-stability freeze was tampered")
        object.__setattr__(self, "_freeze_hash", observed)
        return observed


def freeze_or_refuse_v1_sampled_v7_ceiling_rank_for_C_from_source_images(
    *,
    geometry_binding: V1SampledV7CeilingRankDGeometryBinding,
    source_decision_images: Sequence[V1SampledKKTSourceDecisionImageV2],
    component_stability_receipts_by_rank: Mapping[
        int,
        Sequence[V1SampledV7CeilingRankComponentStabilityReceipt],
    ],
) -> V1SampledV7CeilingRankDStabilityFreezeForC | V1FQRCeilingRankNoCRefusal:
    """Freeze the first invariant rank or emit the dominant no-C action."""

    if type(geometry_binding) is not V1SampledV7CeilingRankDGeometryBinding:
        raise TypeError("ceiling-rank freeze requires exact geometry binding")
    if not isinstance(component_stability_receipts_by_rank, Mapping):
        raise TypeError("component receipts by rank must be one mapping")
    images = tuple(source_decision_images)
    if any(type(image) is not V1SampledKKTSourceDecisionImageV2 for image in images):
        raise TypeError("source projection requires exact source decision image")
    projections = tuple(
        issue_v1_sampled_v7_ceiling_rank_decision_projection_from_source_image(
            geometry_binding=geometry_binding,
            source_decision_image=image,
            component_stability_receipts=(
                component_stability_receipts_by_rank.get(image.rank, ())
            ),
        )
        for image in images
    )
    ceiling_result = freeze_or_refuse_v1_fqr_ceiling_rank_for_C(
        geometry=geometry_binding.ceiling_geometry,
        evaluated_decision_images=tuple(
            value.ceiling_decision_image for value in projections
        ),
    )
    if type(ceiling_result) is V1FQRCeilingRankNoCRefusal:
        return ceiling_result
    if type(ceiling_result) is not V1FQRCeilingRankFreezeForC:
        _assert_never_result(ceiling_result)
    return V1SampledV7CeilingRankDStabilityFreezeForC(
        geometry_binding=geometry_binding,
        source_decision_projections=projections,
        ceiling_freeze=ceiling_result,
        _issuance_token=_FREEZE_TOKEN,
    )


def _assert_never_result(value: object) -> NoReturn:
    raise TypeError(f"unexpected ceiling-rank freeze result: {type(value).__name__}")


__all__ = [
    "SAMPLED_V7_CEILING_RANK_D_GEOMETRY_BINDING_SCHEMA",
    "SAMPLED_V7_CEILING_RANK_D_STABILITY_AUTHORITY",
    "SAMPLED_V7_CEILING_RANK_D_STABILITY_FREEZE_SCHEMA",
    "SAMPLED_V7_CEILING_RANK_SOURCE_DECISION_PROJECTION_SCHEMA",
    "V1SampledV7CeilingRankDGeometryBinding",
    "V1SampledV7CeilingRankDStabilityFreezeForC",
    "V1SampledV7CeilingRankSourceDecisionProjection",
    "freeze_or_refuse_v1_sampled_v7_ceiling_rank_for_C_from_source_images",
    "issue_v1_sampled_v7_ceiling_rank_decision_projection_from_source_image",
    "issue_v1_sampled_v7_ceiling_rank_geometry_from_device_final_rank_source",
]
