"""Consumers of the factored D rows.

Every consumer here replaces a route-scale row operation by a factored one and
refuses, with a typed :class:`V1FactoredRowRefusal`, when a required input is
absent:

* route moments in the exact shape the alpha observation family's
  ``V1StratifiedCorrectedAttributionMoments`` consumes, bound only when
  every identity the legacy type demands is supplied;
* the corrected-work projection ``F_x(s) = mean_cut Σ_fiber |Σ_{route∈fiber∩s}
  a_x(route)|`` by one masked matvec per action (the first-order part per
  fiber), against the native destination-fiber axis receipt;
* the exact own-kink set damage receipt;
* the Loewner-stream tensors of the literal-J joint affine shell
  (``shell Mv`` centers/radii and compact ``Q`` centers/radii, the exact inputs
  of ``_extract_component_bounds`` in the joint Loewner bridge) from factor
  bilinear forms with outward fp enclosures;
* the structural-liberation census.

Nothing here spends alpha, issues a confidence box, certifies F/Q/R, opens
C/K/O, authorizes a mask commit, or carries terminal, runtime, or Pareto
authority.  Binding the Loewner tensors into the bridge's typed result and the
moments into a live alpha observation are not done here.
"""

from __future__ import annotations

from fractions import Fraction

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_factored_row_capture import (
    V1FactoredOwnKinkSetDamage,
    V1FactoredRowPacket,
    V1FactoredRowRefusal,
    V1FactoredRowRefusalCode,
)
from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from .modern_v1_proposal_statistics import V1StratifiedCorrectedAttributionMoments
from .modern_v1_sampled_dual_chart_row_custody import V1SampledDualChartRowRole

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


FACTORED_ROW_ROUTE_MOMENTS_SCHEMA: Final = "partition_free_v1_factored_row_route_moments_v2"
FACTORED_ROW_ACTION_WORK_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_factored_row_action_work_projection_v2"
)
FACTORED_ROW_OWN_KINK_DAMAGE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_factored_row_own_kink_set_damage_receipt_v2"
)
FACTORED_ROW_LOEWNER_STREAM_SCHEMA: Final = (
    "partition_free_v1_factored_row_factor_gram_loewner_stream_v2"
)
FACTORED_ROW_LIBERATION_CENSUS_SCHEMA: Final = (
    "partition_free_v1_factored_row_structural_liberation_census_v2"
)
FACTORED_ROW_CONSUMER_AUTHORITY_SCOPE: Final = (
    "proposal_input_statistics_only_no_alpha_spend_confidence_box_D_stability_C_K_O_"
    "mask_commit_terminal_or_pareto_authority_v1"
)
ACTION_WORK_EQUATION_ID: Final = (
    "mean_cut_sum_fiber_abs_sum_route_in_fiber_and_action_a_x_route_by_masked_matvec_v1"
)
LOEWNER_STREAM_EQUATION_ID: Final = (
    "shell_Mv_center_equals_mean_x_inner_row_direction_times_row_at_shell_route_"
    "q_center_equals_mean_x_inner_row_direction_squared_v1"
)
LEGACY_LOEWNER_STREAM_ENCLOSURE_ID: Final = (
    "fp32_two_rounding_envelope_of_legacy_rows_plus_fp64_summation_gamma_outward_v1"
)
LOEWNER_STREAM_ENCLOSURE_ID: Final = (
    "scaled_legacy_row_projection_absolute_mass_with_second_order_observed_gram_enclosure_v2"
)
_MOMENTS_PREFIX: Final = "partition-free-v1-factored-row-route-moments-"
_WORK_PREFIX: Final = "partition-free-v1-factored-row-action-work-"
_KINK_RECEIPT_PREFIX: Final = "partition-free-v1-factored-row-own-kink-receipt-"
_LOEWNER_PREFIX: Final = "partition-free-v1-factored-row-loewner-stream-"
_LIBERATION_PREFIX: Final = "partition-free-v1-factored-row-liberation-"


def _refuse(code: V1FactoredRowRefusalCode, message: str) -> NoReturn:
    raise V1FactoredRowRefusal(code, message)


def _torch() -> Any:
    if torch is None:  # pragma: no cover
        _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "factored consumers require PyTorch")
    return torch


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, f"{name} must be a nonempty string")
    return value


def _packets(packets: Sequence[V1FactoredRowPacket]) -> tuple[V1FactoredRowPacket, ...]:
    result = tuple(packets)
    if not result or any(type(item) is not V1FactoredRowPacket for item in result):
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "consumers require factored row packets")
    first = result[0].receipt
    for packet in result:
        receipt = packet.receipt
        if (
            receipt.physical_route_axis_hash != first.physical_route_axis_hash
            or receipt.route_id_axis_hash != first.route_id_axis_hash
            or receipt.route_count != first.route_count
            or receipt.chart_identity_hash != first.chart_identity_hash
            or receipt.current_mask_hash != first.current_mask_hash
        ):
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "packets belong to different charts or axes")
    if len({p.receipt.receipt_hash for p in result}) != len(result):
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "duplicate packet")
    return result


def _role(role: object) -> V1SampledDualChartRowRole:
    if type(role) is not V1SampledDualChartRowRole:
        _refuse(V1FactoredRowRefusalCode.ROLE_MISMATCH, "row role must be exact")
    return role


def _closed(receipt: object) -> None:
    for name in (
        "D_stability_authority",
        "C_authority",
        "calibration_authority",
        "oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
    ):
        if getattr(receipt, name) is not False:
            _refuse(V1FactoredRowRefusalCode.AUTHORITY_VIOLATION, f"factored consumers carry no {name}")


# --- route moments ------------------------------------------------------------------------


@dataclass(frozen=True)
class V1FactoredRowRouteMoments:
    """Per-stratum route moments from factors; arrays are route-scale, not hashed."""

    role: str
    stratum_names: tuple[str, ...]
    stratum_counts: tuple[int, ...]
    route_count: int
    packet_receipt_hashes: tuple[str, ...]
    packet_stratum_labels: tuple[str, ...]
    signed_mean_by_stratum: np.ndarray = field(repr=False, compare=False)
    signed_m2_by_stratum: np.ndarray = field(repr=False, compare=False)
    absolute_mean_by_stratum: np.ndarray = field(repr=False, compare=False)
    absolute_m2_by_stratum: np.ndarray = field(repr=False, compare=False)
    m2_roundoff_clamped_count: int = 0
    signed_mean_l2_norms: tuple[float, ...] = ()
    absolute_mean_l2_norms: tuple[float, ...] = ()
    route_scale_d2h_bytes: int = 0
    D_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = FACTORED_ROW_CONSUMER_AUTHORITY_SCOPE
    schema: str = FACTORED_ROW_ROUTE_MOMENTS_SCHEMA

    def __post_init__(self) -> None:
        _closed(self)
        names = tuple(_identity(v, name="stratum name") for v in self.stratum_names)
        counts = tuple(int(v) for v in self.stratum_counts)
        if not names or len(names) != len(counts) or any(c < 0 for c in counts):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "stratum axes must align")
        shape = (len(names), int(self.route_count))
        arrays = {}
        for name in ("signed_mean_by_stratum", "signed_m2_by_stratum", "absolute_mean_by_stratum", "absolute_m2_by_stratum"):
            array = np.ascontiguousarray(np.asarray(getattr(self, name), dtype=np.float64))
            if array.shape != shape or not np.isfinite(array).all():
                _refuse(V1FactoredRowRefusalCode.NONFINITE, f"{name} must be finite [S, N]")
            if name != "signed_mean_by_stratum" and bool(np.any(array < 0.0)):
                _refuse(V1FactoredRowRefusalCode.NONFINITE, f"{name} must be nonnegative")
            array.setflags(write=False)
            arrays[name] = array
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_counts", counts)
        object.__setattr__(
            self,
            "signed_mean_l2_norms",
            tuple(float(np.linalg.norm(arrays["signed_mean_by_stratum"][s])) for s in range(len(names))),
        )
        object.__setattr__(
            self,
            "absolute_mean_l2_norms",
            tuple(float(np.linalg.norm(arrays["absolute_mean_by_stratum"][s])) for s in range(len(names))),
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "role": self.role,
                "stratum_names": self.stratum_names,
                "stratum_counts": self.stratum_counts,
                "route_count": self.route_count,
                "packet_receipt_hashes": self.packet_receipt_hashes,
                "packet_stratum_labels": self.packet_stratum_labels,
                "m2_roundoff_clamped_count": self.m2_roundoff_clamped_count,
                "signed_mean_l2_norms": self.signed_mean_l2_norms,
                "absolute_mean_l2_norms": self.absolute_mean_l2_norms,
                "route_scale_arrays_not_hashed": True,
                "authority_scope": self.authority_scope,
            },
            prefix=_MOMENTS_PREFIX,
        )


def compute_v1_factored_row_route_moments(
    *,
    packets: Sequence[V1FactoredRowPacket],
    packet_stratum_labels: Sequence[str],
    stratum_names: Sequence[str],
    role: V1SampledDualChartRowRole,
    gate: Any,
) -> V1FactoredRowRouteMoments:
    """Per-stratum signed/absolute mean and central M2 per route, from factors."""

    torch_module = _torch()
    packet_tuple = _packets(packets)
    role = _role(role)
    labels = tuple(_identity(v, name="packet stratum label") for v in packet_stratum_labels)
    names = tuple(_identity(v, name="stratum name") for v in stratum_names)
    if len(labels) != len(packet_tuple) or any(label not in names for label in labels) or len(set(names)) != len(names):
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "packet stratum labels must name the registered strata")
    gate64 = packet_tuple[0].gate64(gate)
    route_count = packet_tuple[0].route_count
    S = len(names)
    signed_mean = torch_module.zeros((S, route_count), dtype=torch_module.float64, device=gate64.device)
    signed_m2 = torch_module.zeros_like(signed_mean)
    absolute_mean = torch_module.zeros_like(signed_mean)
    absolute_m2 = torch_module.zeros_like(signed_mean)
    counts = [0] * S
    clamped = 0
    for s, name in enumerate(names):
        members = [(p, i) for p, i in zip(packet_tuple, range(len(packet_tuple))) if labels[i] == name]
        n_s = sum(p.context_count for p, _ in members)
        counts[s] = n_s
        if n_s == 0:
            continue
        for absolute, mean_bank, m2_bank in ((False, signed_mean, signed_m2), (True, absolute_mean, absolute_m2)):
            total = torch_module.zeros(route_count, dtype=torch_module.float64, device=gate64.device)
            for packet, _ in members:
                ones = torch_module.ones(packet.context_count, dtype=torch_module.float64, device=gate64.device)
                total += packet.moments(role, 1, ones, gate=gate64, absolute=absolute)
            mean = total / n_s
            mean_bank[s] = mean
            if n_s >= 2:
                # exact two-pass central moment as a pair-weighted Hadamard sum:
                # Σ_x (row_x - m)² = Σ_{a,b} (I - 11ᵀ/n)_{ab} row_a ∘ row_b, per packet block
                # pair (cross-packet blocks through combine_rows).
                m2 = torch_module.zeros_like(total)
                rows_cache: dict[int, Any] = {}

                def rows_of(packet: V1FactoredRowPacket, index: int) -> Any:
                    if index not in rows_cache:
                        eye = torch_module.eye(packet.context_count, dtype=torch_module.float64, device=gate64.device)
                        rows = packet.combine_rows(role, eye, gate=gate64)
                        rows_cache[index] = rows.abs() if absolute else rows
                    return rows_cache[index]

                if len(members) == 1 and not absolute:
                    packet, _ = members[0]
                    P = torch_module.eye(n_s, dtype=torch_module.float64, device=gate64.device) - 1.0 / n_s
                    m2 = packet.pair_weighted_square_diagonal(role, P, gate=gate64)
                else:
                    for packet, index in members:
                        rows = rows_of(packet, index)
                        m2 += ((rows - mean[None, :]) ** 2).sum(dim=0)
                negative = m2 < 0.0
                clamped += int(torch_module.count_nonzero(negative).item())
                m2_bank[s] = torch_module.clamp(m2, min=0.0)
    return V1FactoredRowRouteMoments(
        role=role.value,
        stratum_names=names,
        stratum_counts=tuple(counts),
        route_count=route_count,
        packet_receipt_hashes=tuple(p.receipt.receipt_hash for p in packet_tuple),
        packet_stratum_labels=labels,
        signed_mean_by_stratum=signed_mean.cpu().numpy(),
        signed_m2_by_stratum=signed_m2.cpu().numpy(),
        absolute_mean_by_stratum=absolute_mean.cpu().numpy(),
        absolute_m2_by_stratum=absolute_m2.cpu().numpy(),
        m2_roundoff_clamped_count=clamped,
    )


def bind_v1_factored_row_moments_to_corrected_attribution_moments(
    moments: V1FactoredRowRouteMoments,
    *,
    campaign_contract_hash: str | None,
    model_identity_hash: str | None,
    query_contract_hash: str | None,
    preprocessing_hash: str | None,
    current_mask_hash: str | None,
    raw_logit_stream_source_hash: str | None,
    sampling_law_hash: str | None,
    proposal_generation: int | None,
    route_axis_hash: str | None,
    population_stratum_counts: Sequence[int] | None,
    sealed_population_order_hashes: Sequence[str] | None,
    observed_context_ids_by_stratum: Sequence[Sequence[str]] | None,
    observed_draw_ids_by_stratum: Sequence[Sequence[str]] | None,
    look_index: int | None,
    prior_state_hash: str | None = None,
) -> V1StratifiedCorrectedAttributionMoments:
    """Bind factor moments into the legacy alpha-family moments type.

    Every identity the legacy type demands must be supplied; any ``None`` is a
    typed absence (``MISSING_INPUT``), never defaulted.  The corrected-A role
    with exactly two strata is required (the legacy type's contract).
    """

    if type(moments) is not V1FactoredRowRouteMoments:
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "binding requires factored route moments")
    if moments.role != V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY.value:
        _refuse(V1FactoredRowRefusalCode.ROLE_MISMATCH, "corrected-attribution moments bind the A role only")
    if len(moments.stratum_names) != 2:
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "the legacy moments type carries exactly two strata")
    required = {
        "campaign_contract_hash": campaign_contract_hash,
        "model_identity_hash": model_identity_hash,
        "query_contract_hash": query_contract_hash,
        "preprocessing_hash": preprocessing_hash,
        "current_mask_hash": current_mask_hash,
        "raw_logit_stream_source_hash": raw_logit_stream_source_hash,
        "sampling_law_hash": sampling_law_hash,
        "proposal_generation": proposal_generation,
        "route_axis_hash": route_axis_hash,
        "population_stratum_counts": population_stratum_counts,
        "sealed_population_order_hashes": sealed_population_order_hashes,
        "observed_context_ids_by_stratum": observed_context_ids_by_stratum,
        "observed_draw_ids_by_stratum": observed_draw_ids_by_stratum,
        "look_index": look_index,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "moments binding lacks identities: " + ", ".join(missing))
    contexts = tuple(tuple(str(v) for v in axis) for axis in observed_context_ids_by_stratum)  # type: ignore[union-attr]
    if tuple(len(axis) for axis in contexts) != moments.stratum_counts:
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "observed context ledgers differ from the factored stratum counts")
    return V1StratifiedCorrectedAttributionMoments(
        campaign_contract_hash=str(campaign_contract_hash),
        model_identity_hash=str(model_identity_hash),
        query_contract_hash=str(query_contract_hash),
        preprocessing_hash=str(preprocessing_hash),
        current_mask_hash=str(current_mask_hash),
        raw_logit_stream_source_hash=str(raw_logit_stream_source_hash),
        sampling_law_hash=str(sampling_law_hash),
        proposal_generation=int(proposal_generation),  # type: ignore[arg-type]
        route_count=moments.route_count,
        route_axis_hash=str(route_axis_hash),
        stratum_names=moments.stratum_names,  # type: ignore[arg-type]
        population_stratum_counts=tuple(int(v) for v in population_stratum_counts),  # type: ignore[union-attr]
        sealed_population_order_hashes=tuple(str(v) for v in sealed_population_order_hashes),  # type: ignore[union-attr]
        stratum_counts=moments.stratum_counts,  # type: ignore[arg-type]
        observed_context_ids_by_stratum=contexts,  # type: ignore[arg-type]
        observed_draw_ids_by_stratum=tuple(tuple(str(v) for v in axis) for axis in observed_draw_ids_by_stratum),  # type: ignore[union-attr,arg-type]
        signed_mean_by_stratum=np.array(moments.signed_mean_by_stratum, copy=True),
        signed_m2_by_stratum=np.array(moments.signed_m2_by_stratum, copy=True),
        absolute_mean_by_stratum=np.array(moments.absolute_mean_by_stratum, copy=True),
        absolute_m2_by_stratum=np.array(moments.absolute_m2_by_stratum, copy=True),
        look_index=int(look_index),  # type: ignore[arg-type]
        prior_state_hash=prior_state_hash,
    )


# --- first-order per fiber: corrected-work projection F_x(s) -------------------------------


@dataclass(frozen=True)
class V1FactoredRowActionWorkProjection:
    """``F_x(s)`` centers and singleton-triangle sums per context and action."""

    native_fiber_axis_receipt_hash: str
    packet_receipt_hashes: tuple[str, ...]
    action_count: int
    context_count: int
    cut_count: int
    f_center: np.ndarray = field(repr=False, compare=False)
    f_triangle: np.ndarray = field(repr=False, compare=False)
    equation_id: str = ACTION_WORK_EQUATION_ID
    route_scale_d2h_bytes: int = 0
    D_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = FACTORED_ROW_CONSUMER_AUTHORITY_SCOPE
    schema: str = FACTORED_ROW_ACTION_WORK_PROJECTION_SCHEMA

    def __post_init__(self) -> None:
        _closed(self)
        shape = (int(self.context_count), int(self.action_count))
        for name in ("f_center", "f_triangle"):
            array = np.ascontiguousarray(np.asarray(getattr(self, name), dtype=np.float64))
            if array.shape != shape or not np.isfinite(array).all() or bool(np.any(array < 0.0)):
                _refuse(V1FactoredRowRefusalCode.NONFINITE, f"{name} must be finite nonnegative [B, A]")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if bool(np.any(self.f_center > self.f_triangle * (1.0 + 1e-12) + 1e-300)):
            _refuse(V1FactoredRowRefusalCode.NONFINITE, "fibered work exceeds its singleton triangle bound")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "native_fiber_axis_receipt_hash": self.native_fiber_axis_receipt_hash,
                "packet_receipt_hashes": self.packet_receipt_hashes,
                "action_count": self.action_count,
                "context_count": self.context_count,
                "cut_count": self.cut_count,
                "f_center": self.f_center,
                "f_triangle": self.f_triangle,
                "equation_id": self.equation_id,
                "authority_scope": self.authority_scope,
            },
            prefix=_WORK_PREFIX,
        )


def compute_v1_factored_row_action_work_projection(
    *,
    packets: Sequence[V1FactoredRowPacket],
    removal_by_action_route: Any,
    native_fiber_axis: V1NativeDestinationFiberAxisReceipt | None,
    gate: Any,
) -> V1FactoredRowActionWorkProjection:
    """``F_x(s)`` for every context and action by masked matvecs (first order, per fiber).

    Refuses without the native-fiber axis receipt (``MISSING_INPUT``); each cut
    must coincide with one factored layer block (``AXIS_MISMATCH`` otherwise).
    """

    torch_module = _torch()
    packet_tuple = _packets(packets)
    if native_fiber_axis is None:
        _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "action work projection requires the native-fiber axis receipt")
    if type(native_fiber_axis) is not V1NativeDestinationFiberAxisReceipt:
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "native-fiber axis must be the exact receipt")
    first = packet_tuple[0].receipt
    if (
        native_fiber_axis.route_axis_hash != first.physical_route_axis_hash
        or native_fiber_axis.route_id_axis_hash != first.route_id_axis_hash
        or native_fiber_axis.route_count != first.route_count
    ):
        _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "native-fiber axis names another route axis")
    removal = np.asarray(removal_by_action_route, dtype=bool)
    if removal.ndim != 2 or removal.shape[1] != first.route_count or removal.shape[0] == 0:
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "removal indicators must be [A, N] with A >= 1")
    gate64 = packet_tuple[0].gate64(gate)
    device = gate64.device
    actions = int(removal.shape[0])
    removal64 = torch_module.as_tensor(removal, dtype=torch_module.float64, device=device) * gate64[None, :]
    role = V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY
    # cuts must coincide with layer blocks
    layer_by_offset = {layer.route_offset: layer for layer in packet_tuple[0].layers}
    plan = []
    for cut in native_fiber_axis.cuts:
        layer = layer_by_offset.get(cut.route_offset)
        if (
            layer is None
            or layer.route_stop != cut.route_stop
            or tuple(layer.route_block.shape) != tuple(cut.route_shape)
            or int(cut.routes_per_fiber) * (cut.fiber_stop - cut.fiber_offset) != layer.route_count
        ):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, f"native-fiber cut {cut.cut_id!r} does not coincide with a factored layer block")
        plan.append((cut, layer.layer_id))
    total_contexts = sum(p.context_count for p in packet_tuple)
    center = torch_module.zeros((total_contexts, actions), dtype=torch_module.float64, device=device)
    triangle = torch_module.zeros_like(center)
    start = 0
    for packet in packet_tuple:
        stop = start + packet.context_count
        for cut, layer_id in plan:
            layer = next(item for item in packet.layers if item.layer_id == layer_id)
            mask = removal64[:, layer.route_offset : layer.route_stop].reshape(actions, *layer.route_block.shape)
            cot = layer.role_cotangent(role).to(torch_module.float64)
            if layer.reducer_kind == "linear_weight":
                source = layer.input.to(torch_module.float64)
                weight = layer.weight.to(torch_module.float64)
                # fiber = output unit: Σ_{j∈s} A_ij(x) = -gtilde_i ((W ∘ M_s) x)_i  (masked matvec)
                drive = torch_module.einsum("aoi,bi->abo", weight[None] * mask, source) * layer.scale
                fiber_sums = -cot[:, None, :] * drive.transpose(0, 1)  # [B, A, out]
                # triangle: Σ_{j∈s} |A_ij| = |gtilde_i| Σ_j |W_ij| M_s |x_j|
                tri = torch_module.einsum("aoi,bi->abo", weight.abs()[None] * mask, source.abs()) * abs(layer.scale)
                triangle_terms = cot.abs()[:, None, :] * tri.transpose(0, 1)
            else:
                rows = -packet.materialize_rows(role, layer.route_offset, layer.route_stop, gate=gate64)
                rows = -rows  # legacy orientation already applied by materialize_rows
                fibers = cut.fiber_stop - cut.fiber_offset
                per_fiber = rows.reshape(packet.context_count, fibers, int(cut.routes_per_fiber))
                masked = per_fiber[:, None, :, :] * mask.reshape(actions, fibers, int(cut.routes_per_fiber))[None]
                fiber_sums = masked.sum(dim=3)  # [B, A, fibers]
                triangle_terms = masked.abs().sum(dim=3)
            center[start:stop] += fiber_sums.abs().sum(dim=2)
            triangle[start:stop] += triangle_terms.sum(dim=2)
        start = stop
    center /= native_fiber_axis.cut_count
    triangle /= native_fiber_axis.cut_count
    return V1FactoredRowActionWorkProjection(
        native_fiber_axis_receipt_hash=native_fiber_axis.receipt_hash,
        packet_receipt_hashes=tuple(p.receipt.receipt_hash for p in packet_tuple),
        action_count=actions,
        context_count=total_contexts,
        cut_count=int(native_fiber_axis.cut_count),
        f_center=center.cpu().numpy(),
        f_triangle=triangle.cpu().numpy(),
    )


# --- own-kink set damage receipt --------------------------------------------------------------


@dataclass(frozen=True)
class V1FactoredRowOwnKinkSetDamageReceipt:
    packet_receipt_hash: str
    context_ids: tuple[str, ...]
    damage_hash: str
    removed_route_count: int
    layer_ids: tuple[str, ...]
    total_change: tuple[float, ...]
    first_order_change: tuple[float, ...]
    kink_remainder: tuple[float, ...]
    flipped_gate_count: tuple[int, ...]
    in_layer_wall_dose: tuple[float | None, ...]
    claim_scope: str
    orientation: str
    price_or_disclosure: str = "disclosure"
    D_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = FACTORED_ROW_CONSUMER_AUTHORITY_SCOPE
    schema: str = FACTORED_ROW_OWN_KINK_DAMAGE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _closed(self)
        if self.price_or_disclosure != "disclosure":
            _refuse(
                V1FactoredRowRefusalCode.AUTHORITY_VIOLATION,
                "the own-kink set damage is a disclosure (document 51 P-L (c), document 52 F1 default)",
            )
        if any(dose is not None and not math.isfinite(dose) for dose in self.in_layer_wall_dose):
            _refuse(V1FactoredRowRefusalCode.NONFINITE, "no inf enters a receipt; absent walls are None")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in (
                    "schema",
                    "packet_receipt_hash",
                    "context_ids",
                    "damage_hash",
                    "removed_route_count",
                    "layer_ids",
                    "total_change",
                    "first_order_change",
                    "kink_remainder",
                    "flipped_gate_count",
                    "in_layer_wall_dose",
                    "claim_scope",
                    "orientation",
                    "price_or_disclosure",
                    "authority_scope",
                )
            },
            prefix=_KINK_RECEIPT_PREFIX,
        )


def compute_v1_factored_row_own_kink_set_damage(
    *,
    packet: V1FactoredRowPacket,
    removal: Any,
    gate: Any,
) -> tuple[V1FactoredOwnKinkSetDamage, V1FactoredRowOwnKinkSetDamageReceipt]:
    """Exact own-kink set damage with its disclosure receipt."""

    if type(packet) is not V1FactoredRowPacket:
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "own-kink damage requires a factored row packet")
    if removal is None:
        _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "own-kink damage requires a removal set")
    damage = packet.own_kink_set_damage(removal, gate=gate)
    receipt = V1FactoredRowOwnKinkSetDamageReceipt(
        packet_receipt_hash=packet.receipt.receipt_hash,
        context_ids=packet.receipt.context_ids,
        damage_hash=damage.receipt_hash,
        removed_route_count=damage.removed_route_count,
        layer_ids=damage.layer_ids,
        total_change=tuple(float(v) for v in damage.total_change),
        first_order_change=tuple(float(v) for v in damage.first_order_change),
        kink_remainder=tuple(float(v) for v in damage.kink_remainder),
        flipped_gate_count=tuple(int(v) for v in damage.flipped_gate_count),
        in_layer_wall_dose=damage.in_layer_wall_dose,
        claim_scope=damage.claim_scope,
        orientation=damage.orientation,
    )
    return damage, receipt


# --- Loewner stream tensors from factor bilinear forms ----------------------------------------


@dataclass(frozen=True)
class V1FactoredRowLoewnerStream:
    """``shell Mv`` and compact ``Q`` centers/radii for a direction family (bridge inputs)."""

    role: str
    packet_receipt_hashes: tuple[str, ...]
    action_count: int
    shell_size: int
    context_count: int
    shell_route_indices: tuple[int, ...]
    shell_m_center: Any = field(repr=False, compare=False)
    shell_m_radius: Any = field(repr=False, compare=False)
    q_center: Any = field(repr=False, compare=False)
    q_radius: Any = field(repr=False, compare=False)
    equation_id: str = LOEWNER_STREAM_EQUATION_ID
    enclosure_id: str = LOEWNER_STREAM_ENCLOSURE_ID
    bridge_result_bound: bool = False
    route_scale_d2h_bytes: int = 0
    D_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = FACTORED_ROW_CONSUMER_AUTHORITY_SCOPE
    schema: str = FACTORED_ROW_LOEWNER_STREAM_SCHEMA

    def __post_init__(self) -> None:
        _closed(self)
        torch_module = _torch()
        directions = int(self.action_count) + int(self.shell_size)
        for name, shape, nonneg in (
            ("shell_m_center", (directions, int(self.shell_size)), False),
            ("shell_m_radius", (directions, int(self.shell_size)), True),
            ("q_center", (directions,), False),
            ("q_radius", (directions,), True),
        ):
            value = getattr(self, name)
            if not isinstance(value, torch_module.Tensor) or value.dtype != torch_module.float64 or tuple(value.shape) != shape:
                _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, f"{name} must be fp64 with shape {shape}")
            if not bool(torch_module.isfinite(value).all().item()) or (nonneg and bool((value < 0).any().item())):
                _refuse(V1FactoredRowRefusalCode.NONFINITE, f"{name} must be finite (nonnegative radii)")
        if self.bridge_result_bound is not False:
            _refuse(V1FactoredRowRefusalCode.AUTHORITY_VIOLATION, "binding into the joint Loewner bridge result is a typed absence here")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "role": self.role,
                "packet_receipt_hashes": self.packet_receipt_hashes,
                "action_count": self.action_count,
                "shell_size": self.shell_size,
                "context_count": self.context_count,
                "shell_route_indices": self.shell_route_indices,
                "shell_m_center": self.shell_m_center.detach().cpu().numpy(),
                "shell_m_radius": self.shell_m_radius.detach().cpu().numpy(),
                "q_center": self.q_center.detach().cpu().numpy(),
                "q_radius": self.q_radius.detach().cpu().numpy(),
                "equation_id": self.equation_id,
                "enclosure_id": self.enclosure_id,
                "authority_scope": self.authority_scope,
            },
            prefix=_LOEWNER_PREFIX,
        )


def _factored_projection_numeric_interval(packet, directions, *, role, gate):
    """Observed projection enclosure; no population or sampling assertion.

    Linear absolute reduction mass is contracted directly from factors. No
    context-by-full-route bank or identity combination is formed. Conv keeps
    its existing exact legacy-row block, so only FP64 dot error is needed there.
    The FP32 error covers the two route products after the exact represented
    fp32 cotangent/scale. Normal-minimum absolute floors also cover FTZ.
    """
    from .modern_v1_literal_j_residual_population_projection_cuda import (
        _add_up, _mul_up, _div_up,
    )

    packet.assert_integrity()
    t = _torch()
    if not bool(t.isfinite(directions).all().item()):
        _refuse(V1FactoredRowRefusalCode.NONFINITE, "projection directions must be finite")
    center = packet.inner(role, directions, gate=gate)
    absolute_directions = directions.abs()
    operations = 8 * packet.route_count + 8 * len(packet.layers)
    error_fraction = Fraction(operations, 2**53)
    if error_fraction >= Fraction(1, 2):
        _refuse(V1FactoredRowRefusalCode.NONFINITE, "projection exceeds the finite FP64 gamma regime")
    gamma64 = math.nextafter(float(error_fraction / (1 - error_fraction)), math.inf)
    recovery = _div_up(1.0, 1.0 - gamma64)
    gamma32 = math.nextafter(float(Fraction(2, 2**24 - 2)), math.inf)
    eta32, eta64 = t.finfo(t.float32).tiny, t.finfo(t.float64).tiny
    absolute_mass = t.zeros_like(center)
    evaluation_floor = t.zeros_like(center)
    row_error = t.zeros_like(center)
    for layer in packet.layers:
        block = absolute_directions[:, layer.route_offset:layer.route_stop]
        if layer.reducer_kind == "linear_weight":
            cot = layer.row_cotangent(role).to(dtype=t.float64).abs()
            source = layer.input.to(dtype=t.float64).abs()
            weight = packet._linear_weight_gate(layer, gate).abs()
            out_units, in_units = tuple(layer.weight.shape)
            shaped = block.reshape(-1, out_units, in_units)
            outer_max = _mul_up(cot.amax(dim=1), source.amax(dim=1))
            rounded_outer_max = _add_up(_mul_up(1.0 + 2.0**-24, outer_max), eta32)
            row_max = _mul_up(rounded_outer_max, layer.weight.to(dtype=t.float64).abs().amax())
            limit = t.finfo(t.float32).max / (1.0 + 2.0**-24)
            if bool((rounded_outer_max > limit).any().item()) or bool((row_max > limit).any().item()):
                _refuse(V1FactoredRowRefusalCode.NONFINITE, "legacy fp32 route product may overflow")
            mass = t.einsum("bo,koi,bi->bk", cot, shaped * weight[None], source)
            # Any intermediate underflow can be amplified by later factors.
            # Start with eta to avoid overflow in the product of maxima.
            floor = _mul_up(float(operations), eta64)
            floor = _mul_up(floor, _add_up(1.0, cot.amax(dim=1)[:, None]))
            floor = _mul_up(floor, _add_up(1.0, source.amax(dim=1)[:, None]))
            floor = _mul_up(floor, _add_up(1.0, weight.amax()))
            floor = _mul_up(floor, _add_up(1.0, block.amax(dim=1)[None, :]))
            mass_upper = _mul_up(_add_up(mass, floor), recovery)
            # |fl(fl(c*x)*w)-c*x*w| <= gamma2 |c*x*w|
            # + 2 eta32 (1+|w|), including subnormal flushing at either product.
            row_floor_mass = (shaped * (1.0 + weight)[None]).sum(dim=(1, 2))
            row_floor_mass = _mul_up(_add_up(row_floor_mass, floor), recovery)
            row_error = _add_up(row_error, _add_up(
                _mul_up(gamma32, mass_upper), _mul_up(2.0 * eta32, row_floor_mass)))
        else:
            rows = packet._layer_rows64(layer, role, gate)
            mass = rows.abs() @ block.T
            floor = _mul_up(float(operations), eta64)
            mass_upper = _mul_up(_add_up(mass, floor), recovery)
        absolute_mass = _add_up(absolute_mass, mass_upper)
        evaluation_floor = _add_up(evaluation_floor, floor)
    radius = _add_up(row_error, _add_up(_mul_up(gamma64, absolute_mass), evaluation_floor))
    if not bool(t.isfinite(center).all().item()) or not bool(t.isfinite(radius).all().item()):
        _refuse(V1FactoredRowRefusalCode.NONFINITE, "factored projection enclosure is nonfinite")
    return center, radius


def compute_v1_factored_row_loewner_stream(
    *,
    packets: Sequence[V1FactoredRowPacket],
    action_directions: Any,
    shell_route_indices: Sequence[int] | None,
    role: V1SampledDualChartRowRole,
    gate: Any,
) -> V1FactoredRowLoewnerStream:
    """Loewner-stream tensors for the joint direction family ``(z_1..z_A, e_b1..e_bB)``.

    ``shell_m_center[d, i] = mean_x ⟨row(x), v_d⟩ row(x)[b_i]`` and
    ``q_center[d] = mean_x ⟨row(x), v_d⟩²`` (the literal-J response Gram
    ``M = mean_x J Jᵀ`` applied to the directions); radii are outward fp
    enclosures of the observed batch, including scaled fp32 row products,
    cancellation-safe absolute FP64 error, and second-order projection error.
    These are not finite-population sampling intervals.
    Refuses without shell indices or actions (``MISSING_INPUT``).
    """

    torch_module = _torch()
    packet_tuple = _packets(packets)
    role = _role(role)
    if shell_route_indices is None:
        _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "Loewner stream requires the shell route indices")
    shell = tuple(int(v) for v in shell_route_indices)
    route_count = packet_tuple[0].route_count
    if not shell or len(set(shell)) != len(shell) or any(v < 0 or v >= route_count for v in shell):
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "shell route indices must be unique and on the route axis")
    gate64 = packet_tuple[0].gate64(gate)
    device = gate64.device
    Z = torch_module.as_tensor(action_directions, dtype=torch_module.float64, device=device)
    if Z.ndim == 1:
        Z = Z.unsqueeze(0)
    if Z.ndim != 2 or int(Z.shape[1]) != route_count or int(Z.shape[0]) == 0:
        _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "Loewner stream requires action directions [A, N]")
    A = int(Z.shape[0])
    B = len(shell)
    basis = torch_module.zeros((B, route_count), dtype=torch_module.float64, device=device)
    for i, r in enumerate(shell):
        basis[i, r] = 1.0
    directions = torch_module.cat((Z, basis), dim=0)  # [A+B, N]
    n = sum(p.context_count for p in packet_tuple)
    p_parts = []
    env_parts = []
    for packet in packet_tuple:
        projection, projection_radius = _factored_projection_numeric_interval(
            packet, directions, role=role, gate=gate64)
        p_parts.append(projection)
        env_parts.append(projection_radius)
    p = torch_module.cat(p_parts, dim=0)
    env = torch_module.cat(env_parts, dim=0)
    from .modern_v1_literal_j_residual_population_projection_cuda import _quadratic_interval
    from .modern_v1_literal_j_joint_affine_shell_residual_population_stream_cuda import _compact_shell_m_interval

    # Reuse the established arithmetic kernels on this observed batch only.
    # n == population_count suppresses their sampling term; it does NOT declare
    # a census of the external discovery population. Support placeholders are
    # never used because this calculation makes no statistical assertion.
    support = torch_module.zeros(A + B, dtype=torch_module.float64, device=device)
    q_center, q_radius = _quadratic_interval(
        projection=p, projection_radius=env, projection_bound=support,
        population_count=n, alpha=0.0)
    shell_m_center, shell_m_radius = _compact_shell_m_interval(
        projection=p, projection_radius=env, projection_bound=support,
        literal_bound=torch_module.zeros(B, dtype=torch_module.float64, device=device),
        shell_indices=tuple(range(B)), action_count=A, population_count=n, alpha=0.0)
    return V1FactoredRowLoewnerStream(
        role=role.value,
        packet_receipt_hashes=tuple(pk.receipt.receipt_hash for pk in packet_tuple),
        action_count=A,
        shell_size=B,
        context_count=n,
        shell_route_indices=shell,
        shell_m_center=shell_m_center.contiguous(),
        shell_m_radius=shell_m_radius.contiguous(),
        q_center=q_center.contiguous(),
        q_radius=q_radius.contiguous(),
    )


# --- structural liberation census ---------------------------------------------------------------


@dataclass(frozen=True)
class V1FactoredRowLiberationCensus:
    """Routes whose rows are exactly zero by graph or sample fact."""

    packet_receipt_hash: str
    layer_ids: tuple[str, ...]
    graph_liberated_units_by_layer: tuple[tuple[int, ...], ...]
    graph_connectivity_known_by_layer: tuple[bool, ...]
    sample_inactive_units_by_layer: tuple[tuple[int, ...], ...]
    sample_inactive_with_nonpositive_baseline_by_layer: tuple[tuple[int, ...], ...]
    zero_route_count_literal_j: int
    zero_route_count_corrected_a: int
    literal_j_zero_routes: np.ndarray = field(repr=False, compare=False)
    corrected_a_zero_routes: np.ndarray = field(repr=False, compare=False)
    D_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = FACTORED_ROW_CONSUMER_AUTHORITY_SCOPE
    schema: str = FACTORED_ROW_LIBERATION_CENSUS_SCHEMA

    def __post_init__(self) -> None:
        _closed(self)
        for name in ("literal_j_zero_routes", "corrected_a_zero_routes"):
            array = np.ascontiguousarray(np.asarray(getattr(self, name), dtype=bool))
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if int(self.literal_j_zero_routes.sum()) != self.zero_route_count_literal_j or int(
            self.corrected_a_zero_routes.sum()
        ) != self.zero_route_count_corrected_a:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "liberation counts do not replay")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "packet_receipt_hash": self.packet_receipt_hash,
                "layer_ids": self.layer_ids,
                "graph_liberated_units_by_layer": self.graph_liberated_units_by_layer,
                "graph_connectivity_known_by_layer": self.graph_connectivity_known_by_layer,
                "sample_inactive_units_by_layer": self.sample_inactive_units_by_layer,
                "sample_inactive_with_nonpositive_baseline_by_layer": self.sample_inactive_with_nonpositive_baseline_by_layer,
                "zero_route_count_literal_j": self.zero_route_count_literal_j,
                "zero_route_count_corrected_a": self.zero_route_count_corrected_a,
                "authority_scope": self.authority_scope,
            },
            prefix=_LIBERATION_PREFIX,
        )


def structurally_liberated_routes(
    *,
    packet: V1FactoredRowPacket,
    gate: Any,
) -> V1FactoredRowLiberationCensus:
    """Read the exact-zero route sets off ``(M, a, b)`` and the layer chain.

    Graph fact: a unit whose outgoing routes (the next Linear layer's column,
    when the chain ``out_ℓ == in_{ℓ+1}`` identifies it) are all gated off has
    ``G = 0`` on every input, so its incoming routes are exactly zero for both
    roles.  Sample fact: a unit with ``a ≤ 0`` on every sampled context has
    zero literal-J rows; its corrected-A rows are zero only if also ``b ≤ 0``.
    Connectivity that cannot be inferred is recorded as unknown, never guessed.

    Precondition (production law): the packet's factors were captured on the
    chart of ``receipt.current_mask_hash`` -- the masked network -- so masked
    outgoing weights are zero in the forward and ``G_u`` vanishes exactly.  A
    gate that is not that chart's mask describes another chart; pass the
    ``HardMask`` so the identity is checked (``STALE_IDENTITY`` otherwise).
    """

    torch_module = _torch()
    if type(packet) is not V1FactoredRowPacket:
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "liberation census requires a factored row packet")
    gate64 = packet.gate64(gate)
    layers = packet.layers
    route_count = packet.route_count
    j_zero = np.zeros(route_count, dtype=bool)
    a_zero = np.zeros(route_count, dtype=bool)
    graph_units = []
    known = []
    inactive_units = []
    inactive_nonpos = []
    for index, layer in enumerate(layers):
        shape = tuple(layer.route_block.shape)
        gate_block = gate64[layer.route_offset : layer.route_stop].reshape(shape) > 0.5
        # graph fact via the next Linear layer's columns
        liberated: list[int] = []
        connectivity = False
        if index + 1 < len(layers):
            nxt = layers[index + 1]
            if (
                layer.reducer_kind == "linear_weight"
                and nxt.reducer_kind == "linear_weight"
                and int(nxt.weight.shape[1]) == int(layer.weight.shape[0])
            ):
                connectivity = True
                next_gate = gate64[nxt.route_offset : nxt.route_stop].reshape(tuple(nxt.route_block.shape)) > 0.5
                no_outgoing = ~next_gate.any(dim=0)  # per unit u of this layer
                liberated = [int(u) for u in torch_module.nonzero(no_outgoing).flatten().tolist()]
        graph_units.append(tuple(liberated))
        known.append(connectivity)
        # sample fact
        a = layer.preactivation
        if a.ndim == 2:
            inactive = ~(a > 0).any(dim=0)
        else:
            inactive = ~(a > 0).any(dim=(0, 2, 3))
        b = torch_module.broadcast_to(layer.bias_baseline, a.shape[1:]) if a.ndim == 2 else torch_module.broadcast_to(layer.bias_baseline, a.shape[1:])
        b_unit_nonpos = ~(b > 0).reshape(int(a.shape[1]), -1).any(dim=1)
        if layer.activation_kind != "relu":
            inactive = torch_module.zeros_like(inactive)
        inactive_idx = [int(u) for u in torch_module.nonzero(inactive).flatten().tolist()]
        nonpos_idx = [int(u) for u in torch_module.nonzero(inactive & b_unit_nonpos).flatten().tolist()]
        inactive_units.append(tuple(inactive_idx))
        inactive_nonpos.append(tuple(nonpos_idx))
        block_j = np.zeros(shape, dtype=bool)
        block_a = np.zeros(shape, dtype=bool)
        for u in liberated:
            block_j[u] = True
            block_a[u] = True
        for u in inactive_idx:
            block_j[u] = True
        for u in nonpos_idx:
            block_a[u] = True
        gated_off = ~gate_block.cpu().numpy()
        block_j |= gated_off
        block_a |= gated_off
        j_zero[layer.route_offset : layer.route_stop] = block_j.reshape(-1)
        a_zero[layer.route_offset : layer.route_stop] = block_a.reshape(-1)
    return V1FactoredRowLiberationCensus(
        packet_receipt_hash=packet.receipt.receipt_hash,
        layer_ids=tuple(layer.layer_id for layer in layers),
        graph_liberated_units_by_layer=tuple(graph_units),
        graph_connectivity_known_by_layer=tuple(known),
        sample_inactive_units_by_layer=tuple(inactive_units),
        sample_inactive_with_nonpositive_baseline_by_layer=tuple(inactive_nonpos),
        zero_route_count_literal_j=int(j_zero.sum()),
        zero_route_count_corrected_a=int(a_zero.sum()),
        literal_j_zero_routes=j_zero,
        corrected_a_zero_routes=a_zero,
    )



# --- row-free heterogeneous-mass population Loewner bounds ------------------------------------

FACTORED_ROW_JOINT_POPULATION_LOEWNER_SCHEMA: Final = (
    "partition_free_v1_factored_row_joint_population_loewner_bounds_v2"
)
_JOINT_POPULATION_LOEWNER_PREFIX: Final = (
    "partition-free-v1-factored-row-joint-population-loewner-bounds-"
)
JOINT_POPULATION_LOEWNER_MASS_LAW_ID: Final = (
    "heterogeneous_finite_population_masses_d1_over_N_and_residual_stratum_Nh_minus_dh_over_N_v1"
)


def _bridge_interval_law():
    """The joint affine-shell Loewner bridge's outward-rounding helpers (imported, not restated)."""

    from . import modern_v1_literal_j_joint_affine_shell_loewner_stream_bridge as bridge

    return bridge._extract_component_bounds, bridge._mix_population_bounds, bridge._loewner_from_bounds


@dataclass(frozen=True)
class V1FactoredRowJointPopulationLoewnerBounds:
    """Per-action ``(B+1)x(B+1)`` mixed-sample intervals from three factored components.

    The D1 (sampled) component and the two residual strata are mixed with the
    exact rational masses ``d/N`` and ``(N_h - d_h)/N`` by the bridge's own
    outward-rounded arithmetic. These masses do not supply finite-population
    sampling uncertainty: the component radii describe observed arithmetic
    only. No full-population enclosure or census custody is certified here.
    The historical type/schema name is retained for replay. Binding into the bridge result
    is not done: ``bridge_result_bound=False``.
    """

    role: str
    action_count: int
    shell_size: int
    shell_route_indices: tuple[int, ...]
    population_count: int
    d1_count: int
    d1_stratum_counts: tuple[int, int]
    residual_population_counts: tuple[int, int]
    residual_observed_counts: tuple[int, int]
    d1_mass_fraction: tuple[int, int]
    residual_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    component_receipt_hashes: tuple[str, str, str]
    center: Any = field(repr=False, compare=False)
    radius: Any = field(repr=False, compare=False)
    loewner_lower: Any = field(repr=False, compare=False)
    loewner_upper: Any = field(repr=False, compare=False)
    row_sum: Any = field(repr=False, compare=False)
    mass_law_id: str = JOINT_POPULATION_LOEWNER_MASS_LAW_ID
    equation_id: str = LOEWNER_STREAM_EQUATION_ID
    enclosure_id: str = LOEWNER_STREAM_ENCLOSURE_ID
    bridge_result_bound: bool = False
    route_scale_d2h_bytes: int = 0
    D_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = FACTORED_ROW_CONSUMER_AUTHORITY_SCOPE
    schema: str = FACTORED_ROW_JOINT_POPULATION_LOEWNER_SCHEMA

    def __post_init__(self) -> None:
        _closed(self)
        torch_module = _torch()
        dimension = int(self.shell_size) + 1
        for name, shape, nonneg in (
            ("center", (int(self.action_count), dimension, dimension), False),
            ("radius", (int(self.action_count), dimension, dimension), True),
            ("loewner_lower", (int(self.action_count), dimension, dimension), False),
            ("loewner_upper", (int(self.action_count), dimension, dimension), False),
            ("row_sum", (int(self.action_count), dimension), True),
        ):
            value = getattr(self, name)
            if not isinstance(value, torch_module.Tensor) or value.dtype != torch_module.float64 or tuple(value.shape) != shape:
                _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, f"{name} must be fp64 with shape {shape}")
            if not bool(torch_module.isfinite(value).all().item()) or (nonneg and bool((value < 0).any().item())):
                _refuse(V1FactoredRowRefusalCode.NONFINITE, f"{name} must be finite (nonnegative radii)")
        if self.bridge_result_bound is not False:
            _refuse(V1FactoredRowRefusalCode.AUTHORITY_VIOLATION, "binding into the joint Loewner bridge result is a typed absence here")
        total = Fraction(*self.d1_mass_fraction) + sum(Fraction(*value) for value in self.residual_mass_fractions)
        if total != 1:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "population masses must sum to one")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "role": self.role,
                "action_count": self.action_count,
                "shell_size": self.shell_size,
                "shell_route_indices": self.shell_route_indices,
                "population_count": self.population_count,
                "d1_count": self.d1_count,
                "d1_stratum_counts": self.d1_stratum_counts,
                "residual_population_counts": self.residual_population_counts,
                "residual_observed_counts": self.residual_observed_counts,
                "d1_mass_fraction": self.d1_mass_fraction,
                "residual_mass_fractions": self.residual_mass_fractions,
                "component_receipt_hashes": self.component_receipt_hashes,
                "center": self.center.detach().cpu().numpy(),
                "radius": self.radius.detach().cpu().numpy(),
                "loewner_lower": self.loewner_lower.detach().cpu().numpy(),
                "loewner_upper": self.loewner_upper.detach().cpu().numpy(),
                "row_sum": self.row_sum.detach().cpu().numpy(),
                "mass_law_id": self.mass_law_id,
                "equation_id": self.equation_id,
                "enclosure_id": self.enclosure_id,
                "authority_scope": self.authority_scope,
            },
            prefix=_JOINT_POPULATION_LOEWNER_PREFIX,
        )


def compute_v1_factored_row_joint_population_loewner_bounds(
    *,
    d1_stream: V1FactoredRowLoewnerStream,
    residual_streams: tuple[V1FactoredRowLoewnerStream, V1FactoredRowLoewnerStream],
    population_count: int,
    d1_stratum_counts: tuple[int, int],
    residual_population_counts: tuple[int, int],
) -> V1FactoredRowJointPopulationLoewnerBounds:
    """Mix observed D1 and residual-stratum geometry with population weights.

    ``d1_stream`` is the stream over the sampled D1 contexts (``d = d_t + d_c``
    contexts, masses ``d/N``); ``residual_streams`` are the target and
    complement residual streams over ``n_h`` observed residual contexts each,
    standing for the ``N_h - d_h`` unsampled contexts of stratum ``h`` (masses
    ``(N_h - d_h)/N``).  The mixture uses the bridge's exact rational-mass
    outward rounding; refuses on any axis, count, or role mismatch. Unobserved
    contexts have no statistical enclosure in this API. Even equal observed
    and remaining counts do not certify unique full-population coverage.
    """

    torch_module = _torch()
    extract, mix, loewner = _bridge_interval_law()
    streams = (d1_stream, *tuple(residual_streams))
    if len(streams) != 3 or any(type(value) is not V1FactoredRowLoewnerStream for value in streams):
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "population Loewner bounds require three factored Loewner streams")
    role = streams[0].role
    actions = int(streams[0].action_count)
    shell = int(streams[0].shell_size)
    shell_indices = tuple(int(v) for v in streams[0].shell_route_indices)
    for value in streams[1:]:
        if (
            value.role != role
            or int(value.action_count) != actions
            or int(value.shell_size) != shell
            or tuple(int(v) for v in value.shell_route_indices) != shell_indices
        ):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "population Loewner components must share role, actions, and shell")
    device = streams[0].shell_m_center.device
    if any(value.shell_m_center.device != device for value in streams[1:]):
        _refuse(V1FactoredRowRefusalCode.DEVICE_MISMATCH, "population Loewner components must share one device")
    N = int(population_count)
    d_counts = tuple(int(v) for v in d1_stratum_counts)
    N_counts = tuple(int(v) for v in residual_population_counts)
    if len(d_counts) != 2 or len(N_counts) != 2 or N <= 0 or any(v < 0 for v in d_counts) or any(v <= 0 for v in N_counts):
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "population counts must be two positive strata with nonnegative D1 counts")
    if sum(N_counts) != N:
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "stratum population counts must sum to the population count")
    if any(d > n for d, n in zip(d_counts, N_counts)):
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "a stratum's D1 count exceeds its population")
    d = sum(d_counts)
    if int(d1_stream.context_count) != d:
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "D1 stream context count differs from the D1 stratum counts")
    observed = tuple(int(value.context_count) for value in streams[1:])
    for h in (0, 1):
        remaining = N_counts[h] - d_counts[h]
        if observed[h] > remaining:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, f"residual stratum {h} observation count exceeds its remaining population")
        if remaining > 0 and observed[h] <= 0:
            _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, f"residual stratum {h} has unsampled contexts but no observed residual stream")
    d_weight = Fraction(d, N)
    residual_weights = tuple(Fraction(N_counts[h] - d_counts[h], N) for h in (0, 1))
    if d_weight + residual_weights[0] + residual_weights[1] != 1:
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "population masses do not close")
    components = tuple(
        extract(
            shell_m_center=value.shell_m_center,
            shell_m_radius=value.shell_m_radius,
            q_center=value.q_center,
            q_radius=value.q_radius,
            action_count=actions,
            shell_size=shell,
        )
        for value in streams
    )
    weights = (d_weight, *residual_weights)
    g_lower, g_upper = mix(tuple((c[0], c[1], w) for c, w in zip(components, weights)))
    h_lower, h_upper = mix(tuple((c[2], c[3], w) for c, w in zip(components, weights)))
    c_lower, c_upper = mix(tuple((c[4], c[5], w) for c, w in zip(components, weights)))
    center, radius, loewner_lower, loewner_upper, row_sum = loewner(
        g_lower=g_lower, g_upper=g_upper, h_lower=h_lower, h_upper=h_upper, c_lower=c_lower, c_upper=c_upper
    )
    del torch_module
    return V1FactoredRowJointPopulationLoewnerBounds(
        role=str(role),
        action_count=actions,
        shell_size=shell,
        shell_route_indices=shell_indices,
        population_count=N,
        d1_count=d,
        d1_stratum_counts=(d_counts[0], d_counts[1]),
        residual_population_counts=(N_counts[0], N_counts[1]),
        residual_observed_counts=(observed[0], observed[1]),
        d1_mass_fraction=(d_weight.numerator, d_weight.denominator),
        residual_mass_fractions=(
            (residual_weights[0].numerator, residual_weights[0].denominator),
            (residual_weights[1].numerator, residual_weights[1].denominator),
        ),
        component_receipt_hashes=tuple(value.receipt_hash for value in streams),  # type: ignore[arg-type]
        center=center,
        radius=radius,
        loewner_lower=loewner_lower,
        loewner_upper=loewner_upper,
        row_sum=row_sum,
    )


__all__ = [
    "FACTORED_ROW_JOINT_POPULATION_LOEWNER_SCHEMA",
    "JOINT_POPULATION_LOEWNER_MASS_LAW_ID",
    "V1FactoredRowJointPopulationLoewnerBounds",
    "compute_v1_factored_row_joint_population_loewner_bounds",
    "ACTION_WORK_EQUATION_ID",
    "FACTORED_ROW_ACTION_WORK_PROJECTION_SCHEMA",
    "FACTORED_ROW_CONSUMER_AUTHORITY_SCOPE",
    "FACTORED_ROW_LIBERATION_CENSUS_SCHEMA",
    "FACTORED_ROW_LOEWNER_STREAM_SCHEMA",
    "FACTORED_ROW_OWN_KINK_DAMAGE_RECEIPT_SCHEMA",
    "FACTORED_ROW_ROUTE_MOMENTS_SCHEMA",
    "LOEWNER_STREAM_ENCLOSURE_ID",
    "LOEWNER_STREAM_EQUATION_ID",
    "V1FactoredRowActionWorkProjection",
    "V1FactoredRowLiberationCensus",
    "V1FactoredRowLoewnerStream",
    "V1FactoredRowOwnKinkSetDamageReceipt",
    "V1FactoredRowRouteMoments",
    "bind_v1_factored_row_moments_to_corrected_attribution_moments",
    "compute_v1_factored_row_action_work_projection",
    "compute_v1_factored_row_loewner_stream",
    "compute_v1_factored_row_own_kink_set_damage",
    "compute_v1_factored_row_route_moments",
    "structurally_liberated_routes",
]
