"""Device-native adaptive D rank/sample KKT-face witness for the row-backed lane.

This module is the one missing caller on the sampled-V7 device-resident
(row-backed) lane.  It discharges the typed obligation
``device_native_adaptive_D_rank_sample_decision_stability_required``
(``missing_witness_type = "device_native_adaptive_D_rank_sample_KKT_face_witness"``)
raised by the qualification runner by *calling* the landed ceiling-rank
D-stability issuers rather than re-deriving any of their mathematics:

* ``issue_v1_sampled_v7_ceiling_rank_geometry_from_device_final_rank_source``
  binds the device-final-rank source to one ceiling FD state and its nested
  lower-rank prefix views (no extra VJPs, no zero padding);
* ``issue_v1_sampled_v7_ceiling_rank_component_stability_receipt`` replays
  margin-versus-radius for the D-sampling, FD-tail, and KKT-numerical
  components of every evaluated rank;
* ``issue_v1_sampled_kkt_source_decision_image_v2`` binds each rank-local
  source decision certificate to the registered prefix and same-VJP fanout;
* ``freeze_or_refuse_v1_sampled_v7_ceiling_rank_for_C_from_source_images``
  freezes the first invariant nested rank or returns the landed typed no-C
  action.

The rank-local decision certificates are the source-issued products of the
landed derivation ``derive_v1_fqr_adaptive_rank_decision`` (one rank-local
KKT solve on the same registered prefix per evaluated rank).  This witness
never synthesizes them: a missing certificate is a typed refusal naming the
rank the ceiling contract demanded next.  Likewise the D-sampling owner radii
come only from source-derived owner receipts (or an exact finite-population
census); an absent radius is the typed refusal ``D_SAMPLING_OWNER_RADIUS_ABSENT``.
No unit is ever converted: every margin and radius is read in the units the
landed certificate and owner receipt already declare.

Authority: every flag on the witness receipt is ``False``.  The only object
that may carry ``C_stream_open_authority=True`` is the landed
``V1SampledV7CeilingRankDStabilityFreezeForC`` returned beside the receipt.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from typing import Any, Final, NoReturn

from .contracts import content_hash
from .modern_v1_fqr_adaptive_rank import (
    FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA,
    V1FQRAdaptiveRankDecisionCertificate,
    V1FQREndogenousRankDPrefixV2,
    V1FQREndogenousRankSameVJPFanoutV2,
)
from .modern_v1_fqr_adaptive_rank_derivation import V1FQRAdaptiveRankDerivedDecision
from .modern_v1_fqr_ceiling_rank_geometry import (
    V1FQRCeilingRankMemoryForecast,
    V1FQRCeilingRankNextAction,
    V1FQRCeilingRankNoCRefusal,
    forecast_v1_fqr_ceiling_rank_memory,
)
from .modern_v1_sampled_device_final_rank_source import (
    V1SampledDeviceFinalRankSourceReceipt,
)
from .modern_v1_sampled_device_resident_d_session import (
    V1SampledDeviceResidentDRankSelectionReceipt,
    V1SampledDeviceResidentDRankStorageMode,
    V1SampledDeviceResidentDSession,
)
from .modern_v1_sampled_kkt_device_shell_bridge import (
    V1SampledKKTEndogenousRankPolicyV2,
    V1SampledKKTSourceDecisionImageV2,
    issue_v1_sampled_kkt_source_decision_image_v2,
)
from .modern_v1_sampled_v7_ceiling_rank_d_stability import (
    V1SampledV7CeilingRankComponentRole,
    V1SampledV7CeilingRankComponentStabilityReceipt,
    V1SampledV7CeilingRankDGeometryBinding,
    V1SampledV7CeilingRankDStabilityFreezeForC,
    freeze_or_refuse_v1_sampled_v7_ceiling_rank_for_C_from_source_images,
    issue_v1_sampled_v7_ceiling_rank_component_stability_receipt,
    issue_v1_sampled_v7_ceiling_rank_geometry_from_device_final_rank_source,
)
from .modern_v1_sampled_v7_decision_uncertainty_source import (
    V1SampledV7DDecisionUncertaintySourceResult,
)
from .modern_v1_signed_work_decision_enclosure import (
    V1SignedWorkDecisionOwnerRole,
    V1SignedWorkDecisionOwnerUncertaintyReceipt,
)


DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_SCHEMA: Final = (
    "partition_free_v1_device_native_adaptive_D_rank_sample_KKT_face_witness_v1"
)
DEVICE_NATIVE_ADAPTIVE_RANK_SELECTION_SCHEMA: Final = (
    "partition_free_v1_device_native_adaptive_rank_selection_v1"
)
DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_TYPE: Final = (
    "device_native_adaptive_D_rank_sample_KKT_face_witness"
)
DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_AUTHORITY: Final = (
    "device_native_adaptive_D_rank_sample_KKT_face_witness_calls_landed_ceiling_"
    "rank_issuers_only_no_sampling_no_solve_no_C_K_O_commit_terminal_or_pareto_"
    "authority_C_freeze_authority_stays_on_landed_freeze_v1"
)
DEVICE_NATIVE_COMPONENT_MARGIN_RULE: Final = (
    "D_sampling: margin=min_b(selected_vs_runner_up_margin_lower_bound), "
    "radius=max_b(selected_vs_runner_up_residual_perturbation_upper_bound)"
    "+sum(sampling owner outward radii, census=0); "
    "FD_tail: margin=min_b(winning_face_margin_lower_bound), "
    "radius=max_b(winning_face_residual_perturbation_upper_bound); "
    "KKT_numerical: margin=min_b(min(selected_vs_runner_up, next_larger_infeasible, "
    "hard_fq_feasibility margins)), radius=max_b(max(their perturbations)); "
    "all in the certificate's declared decision-key units; no conversion"
)
DEVICE_NATIVE_NO_C_FROZEN_RANK_NOT_HBM_ADMITTED: Final = (
    "frozen_rank_not_hbm_admitted"
)
DEVICE_NATIVE_RANK_SELECTION_FROM_FREEZE: Final = "ceiling_freeze_frozen_rank"
DEVICE_NATIVE_RANK_SELECTION_FROM_SESSION: Final = (
    "session_select_smallest_decision_stable_rank"
)

_PREFIX: Final = "partition-free-v1-device-native-adaptive-rank-sample-witness-"
_RECEIPT_TOKEN: Final = object()
_SELECTION_TOKEN: Final = object()
_SAMPLING_OWNER_ROLES: Final = (
    V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING,
    V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING,
)
_FALSE_AUTHORITY_FIELDS: Final = (
    "rank_stability_authority",
    "C_stream_open_authority",
    "calibration_K_open_authority",
    "paired_oracle_O_open_authority",
    "mask_commit_authority",
    "terminal_authority",
    "pareto_authority",
)


class V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode(str, Enum):
    """Typed reasons the witness cannot be issued (never a fabricated face)."""

    DEVICE_FINAL_RANK_SOURCE_ABSENT = "device_final_rank_source_absent"
    MEMORY_FORECAST_ABSENT = "memory_forecast_absent"
    ENDOGENOUS_RANK_POLICY_ABSENT = "endogenous_rank_policy_absent"
    REGISTERED_D_PREFIX_ABSENT = "registered_d_prefix_absent"
    SAME_VJP_FANOUT_ABSENT = "same_vjp_fanout_absent"
    CEILING_SPECTRUM_ABSENT = "ceiling_spectrum_absent"
    RANK_LOCAL_DECISION_IMAGE_ABSENT = "rank_local_decision_image_absent"
    D_SAMPLING_OWNER_RADIUS_ABSENT = "D_sampling_owner_radius_absent"
    HBM_ADMISSION_ABSENT = "hbm_admission_absent"
    SOURCE_COORDINATE_MISMATCH = "source_coordinate_mismatch"
    PHYSICAL_HBM_REFUSED = "physical_hbm_refused"
    PROVIDER_REEXECUTION = "provider_reexecution"
    INVALID_TYPE = "invalid_type"
    NO_DECISION_STABLE_RANK = "no_decision_stable_rank"
    SESSION_SELECTION_DISAGREES = "session_selection_disagrees"
    SOURCE_COMPONENT_DISAGREEMENT = "source_component_disagreement"


class V1DeviceNativeAdaptiveRankSampleWitnessRefusal(RuntimeError):
    """Fail-closed refusal carrying the typed code and the absent inputs."""

    def __init__(
        self,
        code: V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode,
        message: str,
        *,
        missing_inputs: Sequence[str] = (),
        required_rank: int | None = None,
    ) -> None:
        if type(code) is not V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode:
            raise TypeError("witness refusal code must be exact")
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.message = message
        self.missing_inputs = tuple(str(value) for value in missing_inputs)
        self.required_rank = required_rank

    def to_json_payload(self) -> dict[str, object]:
        return {
            "missing_witness_type": DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_TYPE,
            "code": self.code.value,
            "message": self.message,
            "missing_inputs": self.missing_inputs,
            "required_rank": self.required_rank,
        }


def _refuse(
    code: V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode,
    message: str,
    *,
    missing_inputs: Sequence[str] = (),
    required_rank: int | None = None,
) -> NoReturn:
    raise V1DeviceNativeAdaptiveRankSampleWitnessRefusal(
        code,
        message,
        missing_inputs=missing_inputs,
        required_rank=required_rank,
    )


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one finite number",
        )
    number = float(value)
    if not math.isfinite(number):
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be finite; typed None is the only lawful absence",
        )
    return number


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be an exact boolean",
        )
    return value


def _rank_axis(value: object, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one sequence of exact ranks",
        )
    ranks = tuple(_positive_int(rank, name=f"{name} entry") for rank in value)
    if len(set(ranks)) != len(ranks) or tuple(sorted(ranks)) != ranks:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be strictly increasing",
        )
    return ranks


# ---------------------------------------------------------------------------
# Component margins and radii (read from landed objects, never converted)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V1DeviceNativeComponentMarginPair:
    """One component's invariant margin lower bound and radius upper bound."""

    role: V1SampledV7CeilingRankComponentRole
    invariant_margin_lower_bound: float
    uncertainty_radius_upper_bound: float
    radius_source: str

    def __post_init__(self) -> None:
        if type(self.role) is not V1SampledV7CeilingRankComponentRole:
            raise TypeError("component margin pair role must be exact")
        _finite(self.invariant_margin_lower_bound, name="component margin")
        _finite(self.uncertainty_radius_upper_bound, name="component radius")
        _identity(self.radius_source, name="component radius source")


def _sampling_owner_radius(
    *,
    source: V1SampledDeviceFinalRankSourceReceipt,
    d_decision_uncertainty_source: V1SampledV7DDecisionUncertaintySourceResult | None,
    sampling_owner_receipts: Sequence[V1SignedWorkDecisionOwnerUncertaintyReceipt] | None,
) -> tuple[float, str, tuple[str, ...]]:
    """Return (radius, radius_source, owner receipt hashes) or refuse."""

    uncertainty = None
    if d_decision_uncertainty_source is not None:
        if type(d_decision_uncertainty_source) is not (
            V1SampledV7DDecisionUncertaintySourceResult
        ):
            _refuse(
                V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
                "D decision uncertainty source must be the exact landed result",
            )
        receipt = d_decision_uncertainty_source.receipt
        receipt.assert_integrity()
        uncertainty = d_decision_uncertainty_source.uncertainty
        if (
            receipt.registered_d_w_prefix_hash != source.registered_d_prefix_hash
            or receipt.route_axis_hash != source.route_axis_hash
            or receipt.fd_snapshot_hash != source.fd_snapshot_hash
            or receipt.fd_geometry_source_receipt_hash != source.receipt_hash
        ):
            _refuse(
                V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.SOURCE_COORDINATE_MISMATCH,
                "D decision uncertainty source is not bound to this device-final "
                "rank source (prefix, route axis, FD snapshot, or provenance differ)",
            )
    receipts = () if sampling_owner_receipts is None else tuple(sampling_owner_receipts)
    if not receipts:
        if uncertainty is not None and (
            uncertainty.signed_f_sampling.full_train_census
            and uncertainty.literal_j_sampling.full_train_census
        ):
            return 0.0, "exact_finite_population_census", ()
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.D_SAMPLING_OWNER_RADIUS_ABSENT,
            "D-sampling owner radii are absent: the landed uncertainty derivation "
            "issues owner sets without radii (owner_radii_not_inferred) and the "
            "landed owner-receipt derivation requires explicit source-derived "
            "radii; no radius is inferred or converted here",
            missing_inputs=("sampling_owner_receipts",),
        )
    by_role: dict[V1SignedWorkDecisionOwnerRole, V1SignedWorkDecisionOwnerUncertaintyReceipt] = {}
    for receipt in receipts:
        if type(receipt) is not V1SignedWorkDecisionOwnerUncertaintyReceipt:
            _refuse(
                V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
                "sampling owner receipts must be exact landed owner receipts",
            )
        if receipt.owner_role not in _SAMPLING_OWNER_ROLES:
            _refuse(
                V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
                "only signed-F and literal-J sampling owners bound the D-sampling radius",
            )
        if receipt.owner_role in by_role:
            _refuse(
                V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
                "duplicate sampling owner receipt role",
            )
        if (
            receipt.registered_d_w_prefix_hash != source.registered_d_prefix_hash
            or receipt.route_axis_hash != source.route_axis_hash
            or (
                uncertainty is not None
                and receipt.uncertainty_set_hash != uncertainty.uncertainty_hash
            )
        ):
            _refuse(
                V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.SOURCE_COORDINATE_MISMATCH,
                "sampling owner receipt is not bound to this registered D prefix, "
                "route axis, or uncertainty product",
            )
        by_role[receipt.owner_role] = receipt
    if set(by_role) != set(_SAMPLING_OWNER_ROLES):
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.D_SAMPLING_OWNER_RADIUS_ABSENT,
            "both signed-F and literal-J sampling owner receipts are required",
            missing_inputs=tuple(
                role.value for role in _SAMPLING_OWNER_ROLES if role not in by_role
            ),
        )
    ordered = tuple(by_role[role] for role in _SAMPLING_OWNER_ROLES)
    radius = math.fsum(
        _finite(receipt.outward_radius_upper_bound, name="owner outward radius")
        for receipt in ordered
    )
    return (
        radius,
        "sampling_owner_receipts_sum",
        tuple(receipt.receipt_hash for receipt in ordered),
    )


def derive_v1_device_native_component_margin_pairs(
    *,
    certificate: V1FQRAdaptiveRankDecisionCertificate,
    sampling_owner_radius_upper_bound: float,
    sampling_radius_source: str,
) -> tuple[V1DeviceNativeComponentMarginPair, ...]:
    """Read the three component margin/radius pairs off one landed certificate.

    The rule is ``DEVICE_NATIVE_COMPONENT_MARGIN_RULE``.  Only ``min``/``max``
    over the two convex branches and one outward sum are applied; no scale,
    unit, or norm conversion enters.
    """

    if type(certificate) is not V1FQRAdaptiveRankDecisionCertificate:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            "component margins require the exact rank-local decision certificate",
        )
    branches = tuple(certificate.branch_decisions)
    if not branches:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            "rank-local certificate carries no branch decisions",
        )
    owner_radius = _finite(
        sampling_owner_radius_upper_bound,
        name="sampling owner radius",
    )
    if owner_radius < 0.0:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            "sampling owner radius must be nonnegative",
        )
    sampling_margin = min(
        _finite(value.selected_vs_runner_up_margin_lower_bound, name="selected margin")
        for value in branches
    )
    sampling_key_perturbation = max(
        _finite(
            value.selected_vs_runner_up_residual_perturbation_upper_bound,
            name="selected perturbation",
        )
        for value in branches
    )
    fd_margin = min(
        _finite(value.winning_face_margin_lower_bound, name="winning-face margin")
        for value in branches
    )
    fd_radius = max(
        _finite(
            value.winning_face_residual_perturbation_upper_bound,
            name="winning-face perturbation",
        )
        for value in branches
    )
    kkt_margin = min(
        min(
            _finite(value.selected_vs_runner_up_margin_lower_bound, name="selected margin"),
            _finite(
                value.next_larger_infeasible_margin_lower_bound,
                name="next-larger margin",
            ),
            _finite(
                value.hard_fq_feasibility_margin_lower_bound,
                name="hard F/Q margin",
            ),
        )
        for value in branches
    )
    kkt_radius = max(
        max(
            _finite(
                value.selected_vs_runner_up_residual_perturbation_upper_bound,
                name="selected perturbation",
            ),
            _finite(
                value.next_larger_residual_perturbation_upper_bound,
                name="next-larger perturbation",
            ),
            _finite(
                value.hard_fq_residual_perturbation_upper_bound,
                name="hard F/Q perturbation",
            ),
        )
        for value in branches
    )
    return (
        V1DeviceNativeComponentMarginPair(
            role=V1SampledV7CeilingRankComponentRole.D_SAMPLING,
            invariant_margin_lower_bound=sampling_margin,
            uncertainty_radius_upper_bound=sampling_key_perturbation + owner_radius,
            radius_source=sampling_radius_source,
        ),
        V1DeviceNativeComponentMarginPair(
            role=V1SampledV7CeilingRankComponentRole.FD_TAIL,
            invariant_margin_lower_bound=fd_margin,
            uncertainty_radius_upper_bound=fd_radius,
            radius_source="certificate_winning_face_residual_perturbation",
        ),
        V1DeviceNativeComponentMarginPair(
            role=V1SampledV7CeilingRankComponentRole.KKT_NUMERICAL,
            invariant_margin_lower_bound=kkt_margin,
            uncertainty_radius_upper_bound=kkt_radius,
            radius_source="certificate_key_and_boundary_perturbations",
        ),
    )


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessReceipt:
    """Compact, JSON-safe witness receipt keyed by every parent hash.

    New family; it adds no field to any legacy hashed family.  All authority
    flags are ``False``.
    """

    device_final_rank_source_receipt_hash: str
    registered_d_prefix_hash: str
    ceiling_fd_state_hash: str
    route_axis_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_count: int
    ceiling_rank: int
    memory_forecast_hash: str
    endogenous_rank_policy_hash: str
    source_endogenous_rank_policy_hash: str
    d_prefix_hash: str
    same_vjp_fanout_hash: str
    geometry_binding_hash: str
    ceiling_geometry_hash: str
    exposed_rank_lattice: tuple[int, ...]
    hbm_admitted_fd_ranks: tuple[int, ...]
    evaluated_ranks: tuple[int, ...]
    rank_local_decision_certificate_hashes: tuple[tuple[int, str], ...]
    source_decision_image_hashes: tuple[tuple[int, str], ...]
    component_receipt_hashes: tuple[tuple[int, tuple[str, str, str]], ...]
    component_decision_invariants: tuple[tuple[int, tuple[bool, bool, bool]], ...]
    component_margin_lower_bounds: tuple[tuple[int, tuple[float, float, float]], ...]
    component_radius_upper_bounds: tuple[tuple[int, tuple[float, float, float]], ...]
    decision_stable_by_rank: tuple[tuple[int, bool], ...]
    hbm_admitted_decision_stable_by_rank: tuple[tuple[int, bool], ...]
    d_sampling_radius_source: str
    d_sampling_owner_radius_upper_bound: float
    d_decision_uncertainty_source_receipt_hash: str | None
    sampling_owner_receipt_hashes: tuple[str, ...]
    freeze_for_C_hash: str | None
    frozen_rank: int | None
    selected_physical_fd_rank: int | None
    no_C_code: str | None
    no_C_reason: str | None
    no_C_required_action: str | None
    no_C_next_rank: int | None
    provider_call_count_before: int
    provider_call_count_after: int
    _issuance_token: InitVar[object]
    rank_sample_decision_stability_established: bool = False
    units_converted: bool = False
    component_margin_rule: str = DEVICE_NATIVE_COMPONENT_MARGIN_RULE
    witness_type: str = DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_TYPE
    rank_stability_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_SCHEMA
    authority_scope: str = DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_AUTHORITY
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("device-native rank/sample witness receipts are issuer-only")
        for name in (
            "device_final_rank_source_receipt_hash",
            "registered_d_prefix_hash",
            "ceiling_fd_state_hash",
            "route_axis_hash",
            "current_mask_hash",
            "memory_forecast_hash",
            "endogenous_rank_policy_hash",
            "source_endogenous_rank_policy_hash",
            "d_prefix_hash",
            "same_vjp_fanout_hash",
            "geometry_binding_hash",
            "ceiling_geometry_hash",
            "d_sampling_radius_source",
        ):
            _identity(getattr(self, name), name=name)
        _nonnegative_int(self.proposal_generation, name="proposal generation")
        _positive_int(self.route_count, name="route count")
        _positive_int(self.ceiling_rank, name="ceiling rank")
        object.__setattr__(
            self, "exposed_rank_lattice", _rank_axis(self.exposed_rank_lattice, name="exposed rank lattice")
        )
        object.__setattr__(
            self, "hbm_admitted_fd_ranks", _rank_axis(self.hbm_admitted_fd_ranks, name="HBM-admitted ranks")
        )
        object.__setattr__(
            self, "evaluated_ranks", _rank_axis(self.evaluated_ranks, name="evaluated ranks")
        )
        if self.evaluated_ranks != self.exposed_rank_lattice[: len(self.evaluated_ranks)]:
            raise ValueError("evaluated ranks must be the exposed-lattice prefix")
        evaluated = self.evaluated_ranks

        def _axis(name: str, width: int | None, kind: type | None) -> tuple:
            rows = tuple(getattr(self, name))
            if tuple(rank for rank, _value in rows) != evaluated:
                raise ValueError(f"{name} must cover exactly the evaluated ranks in order")
            normalized = []
            for rank, value in rows:
                if width is None:
                    if kind is str:
                        _identity(value, name=name)
                    elif kind is bool:
                        _exact_bool(value, name=name)
                    normalized.append((rank, value))
                    continue
                entries = tuple(value)
                if len(entries) != width:
                    raise ValueError(f"{name} row must have {width} entries")
                if kind is str:
                    for entry in entries:
                        _identity(entry, name=name)
                elif kind is bool:
                    for entry in entries:
                        _exact_bool(entry, name=name)
                elif kind is float:
                    entries = tuple(_finite(entry, name=name) for entry in entries)
                normalized.append((rank, entries))
            return tuple(normalized)

        object.__setattr__(
            self,
            "rank_local_decision_certificate_hashes",
            _axis("rank_local_decision_certificate_hashes", None, str),
        )
        object.__setattr__(
            self, "source_decision_image_hashes", _axis("source_decision_image_hashes", None, str)
        )
        object.__setattr__(
            self, "component_receipt_hashes", _axis("component_receipt_hashes", 3, str)
        )
        object.__setattr__(
            self,
            "component_decision_invariants",
            _axis("component_decision_invariants", 3, bool),
        )
        object.__setattr__(
            self,
            "component_margin_lower_bounds",
            _axis("component_margin_lower_bounds", 3, float),
        )
        object.__setattr__(
            self,
            "component_radius_upper_bounds",
            _axis("component_radius_upper_bounds", 3, float),
        )
        object.__setattr__(
            self, "decision_stable_by_rank", _axis("decision_stable_by_rank", None, bool)
        )
        admitted_rows = tuple(self.hbm_admitted_decision_stable_by_rank)
        if tuple(rank for rank, _value in admitted_rows) != self.hbm_admitted_fd_ranks:
            raise ValueError("HBM-admitted stability rows must cover the admitted set in order")
        for _rank, value in admitted_rows:
            _exact_bool(value, name="HBM-admitted decision-stable flag")
        object.__setattr__(self, "hbm_admitted_decision_stable_by_rank", admitted_rows)
        radius = _finite(
            self.d_sampling_owner_radius_upper_bound,
            name="D-sampling owner radius",
        )
        if radius < 0.0:
            raise ValueError("D-sampling owner radius must be nonnegative")
        if self.d_decision_uncertainty_source_receipt_hash is not None:
            _identity(
                self.d_decision_uncertainty_source_receipt_hash,
                name="D uncertainty source receipt hash",
            )
        object.__setattr__(
            self,
            "sampling_owner_receipt_hashes",
            tuple(
                _identity(value, name="sampling owner receipt hash")
                for value in self.sampling_owner_receipt_hashes
            ),
        )
        _nonnegative_int(self.provider_call_count_before, name="provider calls before")
        _nonnegative_int(self.provider_call_count_after, name="provider calls after")
        if self.provider_call_count_before != self.provider_call_count_after:
            raise PermissionError("the witness cannot recall the measurement provider")
        stable_evaluated = tuple(
            rank for rank, value in self.decision_stable_by_rank if value
        )
        stable_admitted = tuple(
            rank for rank, value in self.hbm_admitted_decision_stable_by_rank if value
        )
        frozen = self.frozen_rank
        if self.freeze_for_C_hash is None:
            if (
                frozen is not None
                or self.selected_physical_fd_rank is not None
                or stable_evaluated
                or stable_admitted
                or self.no_C_code is None
                or self.rank_sample_decision_stability_established is not False
            ):
                raise ValueError("no-C witness must carry no frozen or stable rank")
            _identity(self.no_C_code, name="no-C code")
            _identity(self.no_C_reason, name="no-C reason")
            _identity(self.no_C_required_action, name="no-C required action")
        else:
            _identity(self.freeze_for_C_hash, name="freeze-for-C hash")
            frozen = _positive_int(frozen, name="frozen rank")
            if (
                stable_evaluated != (frozen,)
                or frozen != self.evaluated_ranks[-1]
                or self.rank_sample_decision_stability_established is not True
            ):
                raise ValueError("frozen rank must be the last, only, decision-stable evaluated rank")
            expected_admitted = tuple(
                rank for rank in self.hbm_admitted_fd_ranks if rank == frozen
            )
            if stable_admitted != expected_admitted:
                raise ValueError("HBM-admitted stability must name exactly the frozen rank")
            if stable_admitted:
                if (
                    self.selected_physical_fd_rank != min(stable_admitted)
                    or self.no_C_code is not None
                ):
                    raise ValueError("selected physical rank must be the smallest stable admitted rank")
            else:
                if (
                    self.selected_physical_fd_rank is not None
                    or self.no_C_code != DEVICE_NATIVE_NO_C_FROZEN_RANK_NOT_HBM_ADMITTED
                ):
                    raise ValueError("a frozen rank outside HBM admission cannot select a rank")
        if self.no_C_next_rank is not None:
            _positive_int(self.no_C_next_rank, name="no-C next rank")
        flags = tuple(getattr(self, name) for name in _FALSE_AUTHORITY_FIELDS)
        if any(type(value) is not bool for value in flags) or any(flags):
            raise PermissionError("device-native rank/sample witness crossed authority")
        if (
            _exact_bool(self.units_converted, name="units converted") is not False
            or self.component_margin_rule != DEVICE_NATIVE_COMPONENT_MARGIN_RULE
            or self.witness_type != DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_TYPE
            or self.schema != DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_SCHEMA
            or self.authority_scope != DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_AUTHORITY
        ):
            raise ValueError("device-native rank/sample witness schema is invalid")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        cached = self._receipt_hash
        if cached is not None and cached != observed:
            raise ValueError("device-native rank/sample witness receipt was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("device-native rank/sample witness receipt hash mismatch")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def to_json_payload(self) -> dict[str, object]:
        payload = self._payload()
        payload["receipt_hash"] = self.receipt_hash
        return payload

    @classmethod
    def from_json_payload(
        cls, payload: Mapping[str, object]
    ) -> "V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessReceipt":
        if not isinstance(payload, Mapping):
            raise TypeError("witness payload must be a mapping")
        values = dict(payload)
        declared = values.pop("receipt_hash", None)
        if type(declared) is not str or not declared:
            raise ValueError("witness payload lacks its receipt hash")
        names = {item.name for item in fields(cls) if not item.name.startswith("_")}
        if set(values) != names:
            raise ValueError("witness payload fields differ from the receipt family")

        def _pairs(rows: object, *, tuple_width: int | None) -> tuple:
            out = []
            for entry in rows:  # type: ignore[union-attr]
                rank, value = entry
                if tuple_width is not None:
                    value = tuple(value)
                out.append((int(rank), value))
            return tuple(out)

        for name in ("exposed_rank_lattice", "hbm_admitted_fd_ranks", "evaluated_ranks"):
            values[name] = tuple(int(value) for value in values[name])
        for name in (
            "rank_local_decision_certificate_hashes",
            "source_decision_image_hashes",
            "decision_stable_by_rank",
            "hbm_admitted_decision_stable_by_rank",
        ):
            values[name] = _pairs(values[name], tuple_width=None)
        for name in (
            "component_receipt_hashes",
            "component_decision_invariants",
            "component_margin_lower_bounds",
            "component_radius_upper_bounds",
        ):
            values[name] = _pairs(values[name], tuple_width=3)
        values["sampling_owner_receipt_hashes"] = tuple(values["sampling_owner_receipt_hashes"])
        receipt = cls(_issuance_token=_RECEIPT_TOKEN, **values)
        receipt.assert_integrity(expected_hash=declared)
        return receipt


@dataclass(frozen=True, slots=True)
class V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessResult:
    """Receipt plus the live landed objects it was issued from."""

    receipt: V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessReceipt
    geometry_binding: V1SampledV7CeilingRankDGeometryBinding = field(repr=False)
    source_decision_images: tuple[V1SampledKKTSourceDecisionImageV2, ...] = field(repr=False)
    component_receipts_by_rank: Mapping[
        int, tuple[V1SampledV7CeilingRankComponentStabilityReceipt, ...]
    ] = field(repr=False)
    freeze_for_C: V1SampledV7CeilingRankDStabilityFreezeForC | None = field(repr=False)
    no_C_refusal: V1FQRCeilingRankNoCRefusal | None = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessReceipt:
            raise TypeError("witness result requires the exact receipt")
        if type(self.geometry_binding) is not V1SampledV7CeilingRankDGeometryBinding:
            raise TypeError("witness result requires the exact geometry binding")
        if (self.freeze_for_C is None) == (self.no_C_refusal is None):
            raise ValueError("witness result carries exactly one of freeze or no-C refusal")
        if self.freeze_for_C is not None:
            if type(self.freeze_for_C) is not V1SampledV7CeilingRankDStabilityFreezeForC:
                raise TypeError("witness result requires the exact landed freeze")
            if self.freeze_for_C.freeze_hash != self.receipt.freeze_for_C_hash:
                raise ValueError("witness receipt does not name its freeze")
        elif type(self.no_C_refusal) is not V1FQRCeilingRankNoCRefusal:
            raise TypeError("witness result requires the exact landed no-C refusal")

    @property
    def decision_stable_by_rank(self) -> dict[int, bool]:
        return dict(self.receipt.hbm_admitted_decision_stable_by_rank)


# ---------------------------------------------------------------------------
# Issuer
# ---------------------------------------------------------------------------


def forecast_v1_device_native_ceiling_rank_memory(
    *,
    device_final_rank_source_receipt: V1SampledDeviceFinalRankSourceReceipt,
    positive_weight_stratum_count: int,
    response_rows: int,
    physical_workspace_ceiling_bytes: int,
    legality_block_count: int,
    rounding_tile_size: int,
    maximum_hard_action_count: int,
    D_packet_staging_device_bytes: int = 0,
) -> V1FQRCeilingRankMemoryForecast:
    """Forecast the ceiling state named by the device source (helper only)."""

    if type(device_final_rank_source_receipt) is not V1SampledDeviceFinalRankSourceReceipt:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.DEVICE_FINAL_RANK_SOURCE_ABSENT,
            "memory forecast requires the exact device-final-rank source",
            missing_inputs=("device_final_rank_source_receipt",),
        )
    source = device_final_rank_source_receipt
    return forecast_v1_fqr_ceiling_rank_memory(
        route_count=source.route_count,
        positive_weight_stratum_count=positive_weight_stratum_count,
        response_rows=response_rows,
        ceiling_rank=source.requested_physical_rank,
        physical_workspace_ceiling_bytes=physical_workspace_ceiling_bytes,
        legality_block_count=legality_block_count,
        rounding_tile_size=rounding_tile_size,
        maximum_hard_action_count=maximum_hard_action_count,
        D_packet_staging_device_bytes=D_packet_staging_device_bytes,
    )


RankLocalDecisionDeriver = Callable[
    [int],
    "V1FQRAdaptiveRankDecisionCertificate | V1FQRAdaptiveRankDerivedDecision | None",
]


def _certificate_for_rank(
    *,
    rank: int,
    certificates: Mapping[int, object] | None,
    deriver: RankLocalDecisionDeriver | None,
    source: V1SampledDeviceFinalRankSourceReceipt,
    source_policy_hash: str,
) -> V1FQRAdaptiveRankDecisionCertificate:
    value: object = None
    if certificates is not None and rank in certificates:
        value = certificates[rank]
    elif deriver is not None:
        value = deriver(rank)
    if value is None:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.RANK_LOCAL_DECISION_IMAGE_ABSENT,
            f"no rank-local source decision certificate exists for nested rank {rank} "
            "on the registered prefix; the landed derivation "
            "derive_v1_fqr_adaptive_rank_decision must be run at that rank before "
            "the witness can continue (nothing is synthesized)",
            missing_inputs=(f"rank_local_decision_certificates[{rank}]",),
            required_rank=rank,
        )
    if type(value) is V1FQRAdaptiveRankDerivedDecision:
        value = value.decision_certificate
    if type(value) is not V1FQRAdaptiveRankDecisionCertificate:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            f"rank {rank} decision must be the exact landed certificate or derived decision",
        )
    certificate: V1FQRAdaptiveRankDecisionCertificate = value
    certificate.assert_integrity()
    if (
        certificate.schema != FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA
        or certificate.rank != rank
        or certificate.policy_hash != source_policy_hash
        or certificate.current_mask_hash != source.current_mask_hash
        or certificate.route_axis_hash != source.route_axis_hash
        or certificate.route_count != source.route_count
        or certificate.proposal_generation != source.proposal_generation
    ):
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.SOURCE_COORDINATE_MISMATCH,
            f"rank {rank} certificate is not a v2 rank-local decision on this chart, "
            "policy, and rank",
        )
    return certificate


def issue_v1_device_native_adaptive_D_rank_sample_KKT_face_witness(
    *,
    device_final_rank_source_receipt: V1SampledDeviceFinalRankSourceReceipt | None,
    memory_forecast: V1FQRCeilingRankMemoryForecast | None,
    sampled_kkt_endogenous_rank_policy: V1SampledKKTEndogenousRankPolicyV2 | None,
    d_prefix: V1FQREndogenousRankDPrefixV2 | None,
    fanout: V1FQREndogenousRankSameVJPFanoutV2 | None,
    eigenvalue_square_axis: Sequence[float] | None,
    deterministic_fd_residual_spectral_upper_bound: float | None,
    direction_l2_square_upper_bound: float | None,
    hbm_admitted_fd_ranks: Sequence[int] | None,
    rank_local_decision_certificates: (
        Mapping[int, V1FQRAdaptiveRankDecisionCertificate | V1FQRAdaptiveRankDerivedDecision]
        | None
    ) = None,
    rank_local_decision_deriver: RankLocalDecisionDeriver | None = None,
    d_decision_uncertainty_source: V1SampledV7DDecisionUncertaintySourceResult | None = None,
    sampling_owner_receipts: Sequence[V1SignedWorkDecisionOwnerUncertaintyReceipt] | None = None,
    provider_call_count_before: int = 0,
    provider_call_count_after: int = 0,
) -> V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessResult:
    """Issue the device-native adaptive D rank/sample KKT-face witness.

    Every input is a landed typed object; every absence is a typed refusal.
    The witness evaluates nested ranks in the exposed-lattice prefix order and
    stops exactly where the landed ceiling contract stops: at the first
    invariant image (freeze) or at the first non-refinable ambiguity (no-C).
    """

    missing = tuple(
        name
        for name, value in (
            ("device_final_rank_source_receipt", device_final_rank_source_receipt),
            ("memory_forecast", memory_forecast),
            ("sampled_kkt_endogenous_rank_policy", sampled_kkt_endogenous_rank_policy),
            ("d_prefix", d_prefix),
            ("fanout", fanout),
            ("eigenvalue_square_axis", eigenvalue_square_axis),
            (
                "deterministic_fd_residual_spectral_upper_bound",
                deterministic_fd_residual_spectral_upper_bound,
            ),
            ("direction_l2_square_upper_bound", direction_l2_square_upper_bound),
            ("hbm_admitted_fd_ranks", hbm_admitted_fd_ranks),
        )
        if value is None
    )
    first_code = {
        "device_final_rank_source_receipt": (
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.DEVICE_FINAL_RANK_SOURCE_ABSENT
        ),
        "memory_forecast": V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.MEMORY_FORECAST_ABSENT,
        "sampled_kkt_endogenous_rank_policy": (
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.ENDOGENOUS_RANK_POLICY_ABSENT
        ),
        "d_prefix": V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.REGISTERED_D_PREFIX_ABSENT,
        "fanout": V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.SAME_VJP_FANOUT_ABSENT,
        "eigenvalue_square_axis": (
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.CEILING_SPECTRUM_ABSENT
        ),
        "deterministic_fd_residual_spectral_upper_bound": (
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.CEILING_SPECTRUM_ABSENT
        ),
        "direction_l2_square_upper_bound": (
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.CEILING_SPECTRUM_ABSENT
        ),
        "hbm_admitted_fd_ranks": (
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.HBM_ADMISSION_ABSENT
        ),
    }
    if missing:
        _refuse(
            first_code[missing[0]],
            "witness inputs are absent at this endpoint: " + ", ".join(missing),
            missing_inputs=missing,
        )
    assert device_final_rank_source_receipt is not None
    assert memory_forecast is not None
    assert sampled_kkt_endogenous_rank_policy is not None
    assert d_prefix is not None
    assert fanout is not None
    assert eigenvalue_square_axis is not None
    assert deterministic_fd_residual_spectral_upper_bound is not None
    assert direction_l2_square_upper_bound is not None
    assert hbm_admitted_fd_ranks is not None
    if type(device_final_rank_source_receipt) is not V1SampledDeviceFinalRankSourceReceipt:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            "device-final-rank source must be the exact landed receipt",
        )
    if type(memory_forecast) is not V1FQRCeilingRankMemoryForecast:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            "memory forecast must be the exact ceiling-rank forecast",
        )
    if type(sampled_kkt_endogenous_rank_policy) is not V1SampledKKTEndogenousRankPolicyV2:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            "endogenous rank policy must be the exact sampled-KKT v2 policy",
        )
    if type(d_prefix) is not V1FQREndogenousRankDPrefixV2:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            "registered D prefix must be the exact v2 prefix",
        )
    if type(fanout) is not V1FQREndogenousRankSameVJPFanoutV2:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            "same-VJP fanout must be the exact v2 fanout",
        )
    source = device_final_rank_source_receipt
    source_hash = source.assert_integrity()
    policy = sampled_kkt_endogenous_rank_policy
    policy_hash = policy.assert_integrity()
    source_policy_hash = policy.source_memory_forecast_policy_hash
    prefix_hash = d_prefix.assert_integrity()
    fanout_hash = fanout.assert_integrity()
    admitted = _rank_axis(hbm_admitted_fd_ranks, name="HBM-admitted FD ranks")
    before = _nonnegative_int(provider_call_count_before, name="provider calls before")
    after = _nonnegative_int(provider_call_count_after, name="provider calls after")
    if before != after:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.PROVIDER_REEXECUTION,
            "the witness observed a measurement-provider call; nested views and "
            "rank-local images cannot re-execute D",
        )
    if (
        d_prefix.policy_hash != source_policy_hash
        or fanout.policy_hash != source_policy_hash
        or fanout.registered_d_prefix_hash != prefix_hash
        or d_prefix.prefix_hash != source.registered_d_prefix_hash
        or d_prefix.current_mask_hash != source.current_mask_hash
        or d_prefix.route_axis_hash != source.route_axis_hash
        or d_prefix.proposal_generation != source.proposal_generation
        or policy.route_count != source.route_count
    ):
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.SOURCE_COORDINATE_MISMATCH,
            "device-final source, v2 policy, registered D prefix, and same-VJP "
            "fanout do not name one chart, prefix, and policy",
        )
    if source.requested_physical_rank not in admitted:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.SOURCE_COORDINATE_MISMATCH,
            "the device ceiling rank is not in the HBM-admitted rank set",
        )
    owner_radius, radius_source, owner_hashes = _sampling_owner_radius(
        source=source,
        d_decision_uncertainty_source=d_decision_uncertainty_source,
        sampling_owner_receipts=sampling_owner_receipts,
    )
    binding = issue_v1_sampled_v7_ceiling_rank_geometry_from_device_final_rank_source(
        device_final_rank_source_receipt=source,
        sampled_kkt_endogenous_rank_policy_hash=policy_hash,
        eigenvalue_square_axis=tuple(
            _finite(value, name="eigenvalue-square") for value in eigenvalue_square_axis
        ),
        deterministic_fd_residual_spectral_upper_bound=_finite(
            deterministic_fd_residual_spectral_upper_bound,
            name="deterministic FD residual spectral upper bound",
        ),
        direction_l2_square_upper_bound=_finite(
            direction_l2_square_upper_bound,
            name="direction l2-square upper bound",
        ),
        memory_forecast=memory_forecast,
        provider_call_count_before_nested_view_derivation=before,
        provider_call_count_after_nested_view_derivation=after,
    )
    if type(binding) is V1FQRCeilingRankNoCRefusal:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.PHYSICAL_HBM_REFUSED,
            f"registered physical workspace refuses the ceiling FD state: {binding.reason}",
        )
    if type(binding) is not V1SampledV7CeilingRankDGeometryBinding:
        raise TypeError("unexpected ceiling geometry issuance result")
    geometry = binding.ceiling_geometry
    views = {view.rank: view for view in geometry.nested_views()}
    lattice = geometry.exposed_rank_lattice

    images: list[V1SampledKKTSourceDecisionImageV2] = []
    receipts_by_rank: dict[int, tuple[V1SampledV7CeilingRankComponentStabilityReceipt, ...]] = {}
    certificate_hashes: list[tuple[int, str]] = []
    image_hashes: list[tuple[int, str]] = []
    component_hashes: list[tuple[int, tuple[str, str, str]]] = []
    component_invariants: list[tuple[int, tuple[bool, bool, bool]]] = []
    component_margins: list[tuple[int, tuple[float, float, float]]] = []
    component_radii: list[tuple[int, tuple[float, float, float]]] = []
    evaluated: list[int] = []
    freeze: V1SampledV7CeilingRankDStabilityFreezeForC | None = None
    refusal: V1FQRCeilingRankNoCRefusal | None = None
    next_rank: int | None = lattice[0]
    while next_rank is not None:
        rank = next_rank
        certificate = _certificate_for_rank(
            rank=rank,
            certificates=rank_local_decision_certificates,
            deriver=rank_local_decision_deriver,
            source=source,
            source_policy_hash=source_policy_hash,
        )
        pairs = derive_v1_device_native_component_margin_pairs(
            certificate=certificate,
            sampling_owner_radius_upper_bound=owner_radius,
            sampling_radius_source=radius_source,
        )
        component_invariant = all(
            pair.invariant_margin_lower_bound > pair.uncertainty_radius_upper_bound
            for pair in pairs
        )
        if component_invariant is not certificate.rank_local_residual_decision_stable:
            # The landed image issuer refuses an image whose component
            # conjunction disagrees with the source certificate's own verdict.
            # Either the source certificate did not price the D-sampling owner
            # radius, or its diagnostic scalars do not encode its ambiguity
            # codes.  Neither is this witness's to resolve.
            ambiguous = tuple(
                pair.role.value
                for pair in pairs
                if not pair.invariant_margin_lower_bound > pair.uncertainty_radius_upper_bound
            )
            codes = tuple(
                code
                for branch in certificate.branch_decisions
                for code in branch.source_derivation_ambiguity_codes
            )
            _refuse(
                V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.SOURCE_COMPONENT_DISAGREEMENT,
                f"rank {rank}: source certificate stable="
                f"{certificate.rank_local_residual_decision_stable} but component "
                f"margins say invariant={component_invariant} (ambiguous components "
                f"{ambiguous}, source ambiguity codes {codes}); the landed source "
                "decision image refuses this disagreement",
                required_rank=rank,
            )
        receipts = tuple(
            issue_v1_sampled_v7_ceiling_rank_component_stability_receipt(
                geometry_binding=binding,
                nested_view=views[rank],
                source_decision_certificate=certificate,
                role=pair.role,
                source_derivation_receipt_hash=content_hash(
                    {
                        "schema": _PREFIX + "component-derivation-v1",
                        "role": pair.role.value,
                        "rank": rank,
                        "certificate_hash": certificate.certificate_hash,
                        "radius_source": pair.radius_source,
                        "sampling_owner_receipt_hashes": owner_hashes,
                        "rule": DEVICE_NATIVE_COMPONENT_MARGIN_RULE,
                    },
                    prefix=_PREFIX + "component-derivation-",
                ),
                invariant_margin_lower_bound=pair.invariant_margin_lower_bound,
                uncertainty_radius_upper_bound=pair.uncertainty_radius_upper_bound,
            )
            for pair in pairs
        )
        image = issue_v1_sampled_kkt_source_decision_image_v2(
            policy=policy,
            d_prefix=d_prefix,
            fanout=fanout,
            source_decision_certificate=certificate,
            source_component_receipt_hashes=(
                receipts[0].receipt_hash,
                receipts[1].receipt_hash,
                receipts[2].receipt_hash,
            ),
            D_sampling_decision_invariant=receipts[0].decision_invariant,
            FD_truncation_decision_invariant=receipts[1].decision_invariant,
            KKT_numerical_decision_invariant=receipts[2].decision_invariant,
        )
        images.append(image)
        receipts_by_rank[rank] = receipts
        evaluated.append(rank)
        certificate_hashes.append((rank, certificate.certificate_hash))
        image_hashes.append((rank, image.image_hash))
        component_hashes.append((rank, tuple(value.receipt_hash for value in receipts)))
        component_invariants.append(
            (rank, tuple(value.decision_invariant for value in receipts))
        )
        component_margins.append(
            (rank, tuple(pair.invariant_margin_lower_bound for pair in pairs))
        )
        component_radii.append(
            (rank, tuple(pair.uncertainty_radius_upper_bound for pair in pairs))
        )
        result = freeze_or_refuse_v1_sampled_v7_ceiling_rank_for_C_from_source_images(
            geometry_binding=binding,
            source_decision_images=tuple(images),
            component_stability_receipts_by_rank=receipts_by_rank,
        )
        if type(result) is V1SampledV7CeilingRankDStabilityFreezeForC:
            freeze = result
            next_rank = None
        elif type(result) is V1FQRCeilingRankNoCRefusal:
            if (
                result.required_action
                is V1FQRCeilingRankNextAction.EVALUATE_NEXT_NESTED_RANK_FROM_CEILING_STATE
                and result.next_rank is not None
            ):
                next_rank = result.next_rank
            else:
                refusal = result
                next_rank = None
        else:
            raise TypeError("unexpected ceiling-rank freeze result")

    stable_by_rank = tuple(
        (rank, freeze is not None and rank == freeze.frozen_rank) for rank in evaluated
    )
    admitted_stable = tuple(
        (rank, freeze is not None and rank == freeze.frozen_rank) for rank in admitted
    )
    frozen_rank = None if freeze is None else freeze.frozen_rank
    selected = None
    no_c_code = no_c_reason = no_c_action = None
    no_c_next = None
    if freeze is not None:
        if frozen_rank in admitted:
            selected = frozen_rank
        else:
            no_c_code = DEVICE_NATIVE_NO_C_FROZEN_RANK_NOT_HBM_ADMITTED
            no_c_reason = (
                f"the first invariant nested rank {frozen_rank} is outside the "
                f"HBM-admitted set {admitted}; no admitted rank is decision stable"
            )
            no_c_action = V1FQRCeilingRankNextAction.PHYSICAL_NO_C_REFUSAL.value
    else:
        assert refusal is not None
        no_c_code = refusal.code.value
        no_c_reason = refusal.reason
        no_c_action = refusal.required_action.value
        no_c_next = refusal.next_rank
    receipt = V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessReceipt(
        device_final_rank_source_receipt_hash=source_hash,
        registered_d_prefix_hash=source.registered_d_prefix_hash,
        ceiling_fd_state_hash=source.fd_snapshot_hash,
        route_axis_hash=source.route_axis_hash,
        current_mask_hash=source.current_mask_hash,
        proposal_generation=source.proposal_generation,
        route_count=source.route_count,
        ceiling_rank=source.requested_physical_rank,
        memory_forecast_hash=memory_forecast.forecast_hash,
        endogenous_rank_policy_hash=policy_hash,
        source_endogenous_rank_policy_hash=source_policy_hash,
        d_prefix_hash=prefix_hash,
        same_vjp_fanout_hash=fanout_hash,
        geometry_binding_hash=binding.binding_hash,
        ceiling_geometry_hash=geometry.geometry_hash,
        exposed_rank_lattice=lattice,
        hbm_admitted_fd_ranks=admitted,
        evaluated_ranks=tuple(evaluated),
        rank_local_decision_certificate_hashes=tuple(certificate_hashes),
        source_decision_image_hashes=tuple(image_hashes),
        component_receipt_hashes=tuple(component_hashes),
        component_decision_invariants=tuple(component_invariants),
        component_margin_lower_bounds=tuple(component_margins),
        component_radius_upper_bounds=tuple(component_radii),
        decision_stable_by_rank=stable_by_rank,
        hbm_admitted_decision_stable_by_rank=admitted_stable,
        d_sampling_radius_source=radius_source,
        d_sampling_owner_radius_upper_bound=owner_radius,
        d_decision_uncertainty_source_receipt_hash=(
            None
            if d_decision_uncertainty_source is None
            else d_decision_uncertainty_source.receipt.receipt_hash
        ),
        sampling_owner_receipt_hashes=owner_hashes,
        freeze_for_C_hash=None if freeze is None else freeze.freeze_hash,
        frozen_rank=frozen_rank,
        selected_physical_fd_rank=selected,
        no_C_code=no_c_code,
        no_C_reason=no_c_reason,
        no_C_required_action=no_c_action,
        no_C_next_rank=no_c_next,
        provider_call_count_before=before,
        provider_call_count_after=after,
        rank_sample_decision_stability_established=freeze is not None,
        _issuance_token=_RECEIPT_TOKEN,
    )
    return V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessResult(
        receipt=receipt,
        geometry_binding=binding,
        source_decision_images=tuple(images),
        component_receipts_by_rank=dict(receipts_by_rank),
        freeze_for_C=freeze,
        no_C_refusal=refusal,
    )


# ---------------------------------------------------------------------------
# Rank selection (consumes exactly the witness booleans)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V1DeviceNativeAdaptiveRankSelectionReceipt:
    """Smallest decision-stable HBM-admitted rank, keyed by the witness hash."""

    witness_receipt_hash: str
    freeze_for_C_hash: str
    session_design_hash: str
    hbm_forecast_hash: str
    hbm_admitted_fd_ranks: tuple[int, ...]
    decision_stable_fd_ranks: tuple[int, ...]
    selected_physical_fd_rank: int
    selection_source: str
    session_rank_selection_receipt_hash: str | None
    _issuance_token: InitVar[object]
    rank_stability_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = DEVICE_NATIVE_ADAPTIVE_RANK_SELECTION_SCHEMA

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _SELECTION_TOKEN:
            raise PermissionError("device-native rank selection receipts are issuer-only")
        for name in (
            "witness_receipt_hash",
            "freeze_for_C_hash",
            "session_design_hash",
            "hbm_forecast_hash",
            "selection_source",
        ):
            _identity(getattr(self, name), name=name)
        admitted = _rank_axis(self.hbm_admitted_fd_ranks, name="HBM-admitted ranks")
        stable = _rank_axis(self.decision_stable_fd_ranks, name="decision-stable ranks")
        selected = _positive_int(self.selected_physical_fd_rank, name="selected rank")
        if not stable or selected != min(stable) or any(rank not in admitted for rank in stable):
            raise ValueError("selection must be the smallest decision-stable admitted rank")
        if self.session_rank_selection_receipt_hash is not None:
            _identity(
                self.session_rank_selection_receipt_hash,
                name="session rank selection receipt hash",
            )
        flags = tuple(getattr(self, name) for name in _FALSE_AUTHORITY_FIELDS)
        if any(type(value) is not bool for value in flags) or any(flags):
            raise PermissionError("device-native rank selection crossed authority")
        if self.schema != DEVICE_NATIVE_ADAPTIVE_RANK_SELECTION_SCHEMA:
            raise ValueError("device-native rank selection schema is invalid")
        object.__setattr__(self, "hbm_admitted_fd_ranks", admitted)
        object.__setattr__(self, "decision_stable_fd_ranks", stable)

    @property
    def receipt_hash(self) -> str:
        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }
        return content_hash(payload, prefix=_PREFIX + "rank-selection-")


def select_v1_device_native_adaptive_physical_rank(
    *,
    witness: V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessResult,
    session_design_hash: str,
    hbm_forecast_hash: str,
    device_session: V1SampledDeviceResidentDSession | None = None,
) -> V1DeviceNativeAdaptiveRankSelectionReceipt:
    """Select the smallest decision-stable admitted rank from the witness.

    On a live legacy independent-prefix session the landed
    ``select_smallest_decision_stable_rank`` consumes exactly the witness
    booleans and must agree.  On a canonical ceiling-only session that method
    refuses caller booleans by contract, so the selection is the landed
    freeze's frozen rank.
    """

    if type(witness) is not V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessResult:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
            "rank selection requires the exact witness result",
        )
    receipt = witness.receipt
    receipt.assert_integrity()
    if receipt.selected_physical_fd_rank is None or receipt.freeze_for_C_hash is None:
        _refuse(
            V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.NO_DECISION_STABLE_RANK,
            f"no HBM-admitted rank is decision stable ({receipt.no_C_code})",
        )
    stable = tuple(rank for rank, value in receipt.hbm_admitted_decision_stable_by_rank if value)
    selection_source = DEVICE_NATIVE_RANK_SELECTION_FROM_FREEZE
    session_hash: str | None = None
    if device_session is not None:
        if type(device_session) is not V1SampledDeviceResidentDSession:
            _refuse(
                V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.INVALID_TYPE,
                "device session must be the exact resident-D session",
            )
        if (
            device_session.rank_storage_mode
            is V1SampledDeviceResidentDRankStorageMode.LEGACY_INDEPENDENT_PREFIX
        ):
            session_receipt = device_session.select_smallest_decision_stable_rank(
                dict(receipt.hbm_admitted_decision_stable_by_rank)
            )
            if type(session_receipt) is not V1SampledDeviceResidentDRankSelectionReceipt:
                raise TypeError("session selection returned another receipt type")
            if (
                session_receipt.selected_physical_fd_rank != receipt.selected_physical_fd_rank
                or session_receipt.decision_stable_fd_ranks != stable
                or session_receipt.session_design_hash != session_design_hash
                or session_receipt.hbm_forecast_hash != hbm_forecast_hash
            ):
                _refuse(
                    V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode.SESSION_SELECTION_DISAGREES,
                    "session rank selection differs from the witness selection",
                )
            selection_source = DEVICE_NATIVE_RANK_SELECTION_FROM_SESSION
            session_hash = session_receipt.receipt_hash
    return V1DeviceNativeAdaptiveRankSelectionReceipt(
        witness_receipt_hash=receipt.receipt_hash,
        freeze_for_C_hash=receipt.freeze_for_C_hash,
        session_design_hash=_identity(session_design_hash, name="session design hash"),
        hbm_forecast_hash=_identity(hbm_forecast_hash, name="HBM forecast hash"),
        hbm_admitted_fd_ranks=receipt.hbm_admitted_fd_ranks,
        decision_stable_fd_ranks=stable,
        selected_physical_fd_rank=receipt.selected_physical_fd_rank,
        selection_source=selection_source,
        session_rank_selection_receipt_hash=session_hash,
        _issuance_token=_SELECTION_TOKEN,
    )


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize_v1_device_native_adaptive_rank_sample_witness(
    result: V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessResult
    | V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessReceipt,
) -> dict[str, Any]:
    """Compact durable mapping (hashes, ranks, booleans; no arrays)."""

    receipt = (
        result.receipt
        if type(result) is V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessResult
        else result
    )
    if type(receipt) is not V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessReceipt:
        raise TypeError("witness summary requires the exact receipt or result")
    receipt.assert_integrity()
    return {
        "schema": receipt.schema,
        "witness_type": receipt.witness_type,
        "witness_receipt_hash": receipt.receipt_hash,
        "device_final_rank_source_receipt_hash": (
            receipt.device_final_rank_source_receipt_hash
        ),
        "registered_d_prefix_hash": receipt.registered_d_prefix_hash,
        "ceiling_fd_state_hash": receipt.ceiling_fd_state_hash,
        "ceiling_rank": receipt.ceiling_rank,
        "memory_forecast_hash": receipt.memory_forecast_hash,
        "endogenous_rank_policy_hash": receipt.endogenous_rank_policy_hash,
        "d_prefix_hash": receipt.d_prefix_hash,
        "same_vjp_fanout_hash": receipt.same_vjp_fanout_hash,
        "geometry_binding_hash": receipt.geometry_binding_hash,
        "ceiling_geometry_hash": receipt.ceiling_geometry_hash,
        "exposed_rank_lattice": receipt.exposed_rank_lattice,
        "hbm_admitted_fd_ranks": receipt.hbm_admitted_fd_ranks,
        "evaluated_ranks": receipt.evaluated_ranks,
        "rank_local_decision_certificate_hashes": (
            receipt.rank_local_decision_certificate_hashes
        ),
        "source_decision_image_hashes": receipt.source_decision_image_hashes,
        "component_receipt_hashes": receipt.component_receipt_hashes,
        "component_decision_invariants": receipt.component_decision_invariants,
        "decision_stable_by_rank": receipt.decision_stable_by_rank,
        "hbm_admitted_decision_stable_by_rank": (
            receipt.hbm_admitted_decision_stable_by_rank
        ),
        "d_sampling_radius_source": receipt.d_sampling_radius_source,
        "d_sampling_owner_radius_upper_bound": (
            receipt.d_sampling_owner_radius_upper_bound
        ),
        "sampling_owner_receipt_hashes": receipt.sampling_owner_receipt_hashes,
        "freeze_for_C_hash": receipt.freeze_for_C_hash,
        "frozen_rank": receipt.frozen_rank,
        "selected_physical_fd_rank": receipt.selected_physical_fd_rank,
        "no_C_code": receipt.no_C_code,
        "no_C_required_action": receipt.no_C_required_action,
        "no_C_next_rank": receipt.no_C_next_rank,
        "rank_sample_decision_stability_established": (
            receipt.rank_sample_decision_stability_established
        ),
        "units_converted": receipt.units_converted,
        "provider_call_count_delta": (
            receipt.provider_call_count_after - receipt.provider_call_count_before
        ),
        "authority": {name: False for name in _FALSE_AUTHORITY_FIELDS},
    }


__all__ = [
    "DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_AUTHORITY",
    "DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_SCHEMA",
    "DEVICE_NATIVE_ADAPTIVE_RANK_SAMPLE_WITNESS_TYPE",
    "DEVICE_NATIVE_ADAPTIVE_RANK_SELECTION_SCHEMA",
    "DEVICE_NATIVE_COMPONENT_MARGIN_RULE",
    "DEVICE_NATIVE_NO_C_FROZEN_RANK_NOT_HBM_ADMITTED",
    "DEVICE_NATIVE_RANK_SELECTION_FROM_FREEZE",
    "DEVICE_NATIVE_RANK_SELECTION_FROM_SESSION",
    "V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessReceipt",
    "V1DeviceNativeAdaptiveRankSampleKKTFaceWitnessResult",
    "V1DeviceNativeAdaptiveRankSampleWitnessRefusal",
    "V1DeviceNativeAdaptiveRankSampleWitnessRefusalCode",
    "V1DeviceNativeAdaptiveRankSelectionReceipt",
    "V1DeviceNativeComponentMarginPair",
    "derive_v1_device_native_component_margin_pairs",
    "forecast_v1_device_native_ceiling_rank_memory",
    "issue_v1_device_native_adaptive_D_rank_sample_KKT_face_witness",
    "select_v1_device_native_adaptive_physical_rank",
    "summarize_v1_device_native_adaptive_rank_sample_witness",
]
