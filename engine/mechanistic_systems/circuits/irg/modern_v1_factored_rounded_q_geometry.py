"""Native full observed second-moment Q from rounded legacy literal-J rows.

The device factor is Bhat[x,r] = fl(sqrt(fl(w_s/n_s))*L[x,r]). Its
Frobenius coefficient-error bound separates this stored tensor from the ideal
weighted rounded-row geometry. No old Gram/eigen/FD receipt is reinterpreted.
This numerical owner grants no solver, action or population authority.
"""
from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import dataclass, field
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Any, Mapping

import torch

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from .modern_v1_factored_alpha_source_binding import V1FactoredDAlphaSourceBindingReceipt
from .modern_v1_factored_row_capture import (
    V1FactoredDeviceResidentDLiveHandoff, V1FactoredDeviceResidentDSessionMode,
)
from .modern_v1_factored_signed_work_operator import (
    _custody, _error_constants, _mass_upper, _no_factor_autograd, _positive_int, _require, _up,
)
from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from .modern_v1_sampled_dual_chart_row_custody import V1SampledDualChartRowRole

_J = V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN
_PREFIX = "partition-free-v1-factored-rounded-Q-"
_LEASE_TOKEN = object()


def _float_up(value: Fraction) -> float:
    result = float(value)
    return math.nextafter(result, math.inf) if Fraction.from_float(result) < value else result


def _weight_scale(weight: float, count: int) -> tuple[float, float, float]:
    """Rounded context mass, sqrt scale, and outward absolute sqrt error."""
    _positive_int(count, "stratum count")
    _require(math.isfinite(weight) and weight >= 0, "invalid stratum weight")
    exact = Fraction.from_float(weight) / count
    if exact == 0:
        return 0.0, 0.0, 0.0
    q = float(exact)
    represented = Fraction.from_float(q)
    lo = math.nextafter(q, 0.0) if represented > exact else q
    hi = math.nextafter(q, math.inf) if represented < exact else q
    scale = math.sqrt(q)
    lower = math.nextafter(math.sqrt(lo), 0.0)
    upper = math.nextafter(math.sqrt(hi), math.inf)
    # The rational square comparisons establish the bracket; libm's result
    # is only a starting point, including when q underflows on conversion.
    while Fraction.from_float(lower)**2 > exact:
        lower = math.nextafter(lower, 0.0)
    while Fraction.from_float(upper)**2 < exact:
        upper = math.nextafter(upper, math.inf)
    error = max(abs(Fraction.from_float(scale) - Fraction.from_float(lower)),
                abs(Fraction.from_float(upper) - Fraction.from_float(scale)))
    return q, scale, _float_up(error)


def _rounding_radius(mass_upper: Any, operations: int) -> Any:
    gamma, denominator, floor = _error_constants(operations)
    return _up(_up(_up(mass_upper * gamma) + floor) / denominator)


@dataclass(frozen=True)
class V1FactoredRoundedQDesign:
    additional_memory_budget_bytes: int
    maximum_lanes: int = 2
    route_tile_size: int = 262144
    legacy_conv_row_budget_bytes: int = 256 << 20
    allow_private_cpu_reference: bool = False
    schema: str = "factored_rounded_Q_design_v1"

    def __post_init__(self):
        for name in ("additional_memory_budget_bytes", "maximum_lanes", "route_tile_size"):
            _positive_int(getattr(self, name), name)
        _require(type(self.legacy_conv_row_budget_bytes) is int and self.legacy_conv_row_budget_bytes >= 0,
                 "legacy conv row budget must be nonnegative")
        _require(type(self.allow_private_cpu_reference) is bool and self.schema == "factored_rounded_Q_design_v1",
                 "rounded Q design changed")

    @property
    def design_hash(self):
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True)
class V1FactoredRoundedQReceipt:
    coordinates: Mapping[str, Any]
    schema: str = "factored_rounded_Q_receipt_v1"
    solver_authority: bool = False
    kkt_authority: bool = False
    production_geometry_authority: bool = False
    action_authority: bool = False
    d_decision_authority: bool = False
    population_enclosure_authority: bool = False
    c_measurement_authority: bool = False
    calibration_k_authority: bool = False
    paired_oracle_o_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    production_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    _receipt_hash: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "coordinates", MappingProxyType(dict(self.coordinates)))
        object.__setattr__(self, "_receipt_hash", content_hash(self.to_json_payload(), prefix=_PREFIX+"receipt-"))
        self.assert_integrity()

    def to_json_payload(self):
        return {k:dict(v) if k == "coordinates" else v for k,v in vars(self).items() if not k.startswith("_")}

    @property
    def receipt_hash(self):
        return self._receipt_hash

    def assert_integrity(self):
        _require(self.schema == "factored_rounded_Q_receipt_v1" and all(
            v is False for k,v in vars(self).items() if k.endswith("_authority")), "rounded Q authority changed")
        _require(content_hash(self.to_json_payload(), prefix=_PREFIX+"receipt-") == self._receipt_hash,
                 "rounded Q receipt changed")
        return self._receipt_hash


def _assemble_rounded_factor(*, pairs, axis, gate, weight_scales, design):
    """Numerical kernel; public ownership requires the registered source join."""
    contexts = sum(p.context_count for _,p in pairs)
    factor = torch.empty((contexts, axis.route_count), dtype=torch.float64, device=gate.device)
    factor_squared = torch.zeros((), dtype=torch.float64, device=gate.device)
    error_squared = torch.zeros_like(factor_squared)
    offset, tiles = 0, 0
    for stratum, packet in pairs:
        _, scale, scale_error = weight_scales[stratum]
        for first,last,lo,hi,rows in packet._native_fiber_row_tiles(
                _J, axis, gate, design.route_tile_size, design.legacy_conv_row_budget_bytes):
            del first,last
            rows = rows.reshape(packet.context_count, hi-lo)
            stored = rows * scale
            factor[offset:offset+packet.context_count, lo:hi] = stored
            # For Bhat=fl(L*s), |Bhat-L*sqrt(q*)| is bounded by the
            # scale-interval contribution plus multiplication RN/FTZ error.
            error = _up(_up(rows.abs() * scale_error) + _rounding_radius(stored.abs(), 1))
            factor_squared.add_(stored.square().sum())
            error_squared.add_(error.square().sum())
            tiles += 1
            del rows, stored, error
        offset += packet.context_count
    operations = 2 * contexts * axis.route_count + 8 * tiles + 16
    norm = _up(torch.sqrt(_mass_upper(factor_squared, operations)))
    epsilon = _up(torch.sqrt(_mass_upper(error_squared, operations)))
    rho = _up(_up(_up(norm * epsilon) * 2.0) + _up(epsilon * epsilon))
    _require(bool(torch.isfinite(rho).item()) and bool(torch.isfinite(norm).item()),
             "rounded Q factor/error bound is nonfinite")
    return factor, float(norm.item()), float(epsilon.item()), float(rho.item()), tiles


class V1FactoredRoundedQDeviceHandle:
    """Full rounded-J second moment; one live spectral lease, no signed-F owner."""

    def __init__(self, *, source_binding: V1FactoredDAlphaSourceBindingReceipt,
                 native_fiber_axis: V1NativeDestinationFiberAxisReceipt,
                 current_mask: HardMask, design: V1FactoredRoundedQDesign):
        _require(type(source_binding) is V1FactoredDAlphaSourceBindingReceipt, "exact native source binding required")
        binding_hash = source_binding.assert_live()
        _require(type(design) is V1FactoredRoundedQDesign, "exact rounded Q design required")
        design.__post_init__()
        _require(type(native_fiber_axis) is V1NativeDestinationFiberAxisReceipt, "exact native axis required")
        native_fiber_axis.assert_integrity()
        _require(type(current_mask) is HardMask, "exact hard mask required")
        handoff = source_binding._parents[2]
        _require(type(handoff) is V1FactoredDeviceResidentDLiveHandoff, "exact finalized factored handoff required")
        handoff.assert_live()
        _require(handoff.session.is_finalized and handoff.session._active is None
                 and handoff.finalization_receipt is not None and handoff.replay_boundary.live_endpoint_registered,
                 "rounded Q requires the registered finalized endpoint")
        source = source_binding.coordinates
        _require(source["source_axis_kind"] == "native_declared_physical_coordinates"
                 and source["response_endpoint_family_bound"] is False,
                 "rounded Q requires the native source without a response endpoint")
        _require(current_mask.mask_hash == source["current_mask_hash"], "current mask changed")
        pairs, counts = handoff.session.packets, handoff.session.stratum_counts
        _require(bool(pairs) and all(n > 0 for n in counts)
                 and counts == source["registered_look_stratum_counts"], "source context counts changed")
        device = pairs[0][1].device
        if device.type == "cpu":
            _require(design.allow_private_cpu_reference and handoff.session.design.execution_mode
                     is V1FactoredDeviceResidentDSessionMode.CPU_REFERENCE_TEST_ONLY, "production rounded Q refuses CPU")
        else:
            _require(device.type == "cuda" and handoff.session.design.execution_mode
                     is V1FactoredDeviceResidentDSessionMode.CUDA_PRODUCTION, "CUDA source mode mismatch")
            _require(torch.are_deterministic_algorithms_enabled()
                     and not torch.is_deterministic_algorithms_warn_only_enabled(), "deterministic CUDA required")
        scales = tuple(_weight_scale(w,n) for w,n in zip(handoff.session.design.stratum_weights, counts, strict=True))
        weights, spans, offset = [], [], 0
        tile_bytes, conv_bytes, prepared_linear_cotangent_bytes = 0, 0, 0
        for stratum,packet in pairs:
            packet._native_fiber_axis(native_fiber_axis)
            _no_factor_autograd(packet)
            _require(packet.device == device, "packet device changed")
            weights.extend([scales[stratum][0]] * packet.context_count)
            spans.append((offset, offset+packet.context_count))
            offset += packet.context_count
            for cut,layer in zip(native_fiber_axis.cuts,packet.layers,strict=True):
                width = min(cut.route_count, max(1,design.route_tile_size//cut.routes_per_fiber)*cut.routes_per_fiber)
                tile_bytes = max(tile_bytes, packet.context_count*width*8*16)
                if layer.reducer_kind != "linear_weight":
                    conv_bytes = max(conv_bytes, packet.context_count*layer.route_count*8)
                else:
                    prepared_linear_cotangent_bytes = max(prepared_linear_cotangent_bytes,
                        layer.post_activation_cotangent.numel() * layer.post_activation_cotangent.element_size())
        _require(offset == sum(counts) and abs(math.fsum(weights)-1.0) <= 4*math.ulp(1.0), "context measure does not close")
        routes = native_fiber_axis.route_count
        factor_bytes = offset*routes*8
        # Full-destination ReLU fraction masks/FP32 intermediates and optional
        # scale division, in addition to the prepared result held by the tile loop.
        linear_cotangent_preparation_workspace_bytes = 8 * prepared_linear_cotangent_bytes
        # Diagnostic enclosure uses tile contractions, never abs(full factor).
        required = (factor_bytes + routes*8 + tile_bytes + conv_bytes + prepared_linear_cotangent_bytes
                    + linear_cotangent_preparation_workspace_bytes
                    + offset*design.maximum_lanes*8*16)
        required += 4*routes*design.maximum_lanes*8
        required += handoff.replay_boundary.provider_peak_workspace_bytes
        _require(conv_bytes <= design.legacy_conv_row_budget_bytes
                 and required <= design.additional_memory_budget_bytes, "rounded Q memory not admitted")
        if device.type == "cuda":
            _require(required <= cuda_reusable_memory_bytes(device, framework=torch), "insufficient CUDA memory")
        self._closed, self._lease_count = False, 0
        self._binding, self._handoff, self._mask = source_binding, handoff, current_mask
        self._axis, self._design, self._pairs = native_fiber_axis, design, pairs
        self._binding_hash, self._design_hash = binding_hash, design.design_hash
        self._spans = tuple(spans)
        self._gate = pairs[0][1].gate64(current_mask)
        self._factor, norm, epsilon, rho, tiles = _assemble_rounded_factor(
            pairs=pairs, axis=native_fiber_axis, gate=self._gate, weight_scales=scales, design=design)
        self._singleton = handoff.singleton_work
        _require(type(self._singleton) is torch.Tensor and self._singleton.dtype == torch.float64
                 and self._singleton.device == device and tuple(self._singleton.shape) == (routes,)
                 and not self._singleton.requires_grad and bool(torch.isfinite(self._singleton).all().item())
                 and bool((self._singleton >= 0).all().item()), "invalid native singleton work")
        self._owned_custody = tuple(_custody(t) for t in (self._factor,self._gate,self._singleton))
        self._receipt = V1FactoredRoundedQReceipt(dict(
            source_binding_hash=binding_hash, replay_boundary_hash=source["replay_boundary_hash"],
            session_design_hash=source["session_design_hash"],
            finalization_receipt_hash=source["finalization_receipt_hash"],
            upstream_factor_algebra_geometry_hash=source["exact_geometry_receipt_hash"],
            upstream_geometry_used_for_rounded_Q=False,
            packet_receipt_hashes=source["packet_receipt_hashes"], look_receipt_hashes=source["look_receipt_hashes"],
            physical_route_axis_hash=source["physical_route_axis_hash"], route_id_axis_hash=source["route_id_axis_hash"],
            current_mask_hash=current_mask.mask_hash, chart_identity_hash=source["chart_identity_hash"],
            proposal_generation=source["proposal_generation"], literal_j_stream_identity_hash=source["literal_j_stream_identity_hash"],
            native_fiber_axis_hash=native_fiber_axis.receipt_hash, native_fiber_measure_hash=native_fiber_axis.measure_hash,
            stratum_names=handoff.session.design.stratum_names, stratum_counts=counts,
            stratum_weights=handoff.session.design.stratum_weights, packet_strata=tuple(s for s,_ in pairs),
            packet_context_spans=self._spans, context_weights=tuple(weights), weight_scales=scales,
            factor_shape=(offset,routes), factor_row_count=offset, route_count=routes,
            factorization_kind="weighted_second_moment_full_observed_context_rounded_literal_J_rows_v1",
            centered_stack_used=False, eigendecomposition_used=False, fd_sketch_used=False,
            row_spool_receipt_hash=None, signed_f_operator_retained=False, route_scale_d2h_bytes=0,
            coefficient_error_frobenius_upper=epsilon, factor_frobenius_norm_upper=norm,
            roundoff_residual_spectral_bound=rho, diagonal_branch_admitted=False,
            numerical_scope="ideal_registered_weighted_rounded_J_Q_enclosed_by_stored_B_Gram_plus_rho_I",
            arithmetic_law="legacy_FP32_rows_FP64_RN_or_FTZ_with_sqrt_weight_interval_v1",
            factor_bytes=factor_bytes, additional_required_bytes=required, tile_workspace_bytes=tile_bytes,
            prepared_linear_cotangent_bytes=prepared_linear_cotangent_bytes,
            linear_cotangent_preparation_workspace_bytes=linear_cotangent_preparation_workspace_bytes,
            legacy_conv_payload_bytes=conv_bytes, norm_tile_count=tiles, design_hash=design.design_hash,
            device=str(device), private_cpu_reference_execution=device.type == "cpu"))
        self._receipt_hash = self._receipt.receipt_hash
        self.assert_live()

    @property
    def receipt(self):
        return self._receipt

    @property
    def receipt_hash(self):
        return self._receipt_hash

    def _assert_custody(self):
        _require(not self._closed, "rounded Q handle closed")
        _require(tuple(_custody(t) for t in (self._factor,self._gate,self._singleton)) == self._owned_custody,
                 "rounded Q tensor custody changed")
        _require(self._design.design_hash == self._design_hash, "rounded Q design changed")
        _require(self._handoff.session.packets == self._pairs
                 and self._handoff.singleton_work is self._singleton, "rounded Q parent changed")
        _require(self._handoff.session.is_finalized and self._handoff.session._active is None
                 and self._handoff.session.design.design_hash == self.receipt.coordinates["session_design_hash"]
                 and self._handoff.session.stratum_counts == self.receipt.coordinates["stratum_counts"]
                 and self._handoff.finalization_receipt is not None
                 and self._handoff.finalization_receipt.receipt_hash == self.receipt.coordinates["finalization_receipt_hash"]
                 and self._handoff.replay_boundary.boundary_hash == self.receipt.coordinates["replay_boundary_hash"],
                 "rounded Q finalized endpoint changed")
        self._axis.assert_integrity()
        _require(self._axis.receipt_hash == self.receipt.coordinates["native_fiber_axis_hash"], "native axis changed")
        for _,packet in self._pairs:
            packet.assert_integrity()
            _no_factor_autograd(packet)
        _require(self.receipt.assert_integrity() == self._receipt_hash, "rounded Q receipt replaced")

    def assert_live(self):
        self._assert_custody()
        _require(self._binding.assert_live() == self._binding_hash, "rounded Q source changed")
        _require(self._mask.mask_hash == self.receipt.coordinates["current_mask_hash"], "rounded Q mask changed")
        return self.receipt_hash

    def borrow_for_kkt(self, *, expected_receipt_hash, expected_source_binding_hash, expected_device, branch):
        self.assert_live()
        _require(branch == "spectral", "rounded Q admits only the spectral numerical envelope")
        _require(expected_receipt_hash == self.receipt_hash and expected_source_binding_hash == self._binding_hash
                 and torch.device(expected_device) == self._factor.device, "rounded Q lease identity mismatch")
        _require(self._lease_count == 0, "rounded Q already leased")
        self._lease_count = 1
        return V1FactoredRoundedQLease(self, _issuance_token=_LEASE_TOKEN)

    def quadratic_enclosure(self, z):
        """Diagnostic ideal-Q enclosure, including stored-factor matmul rounding."""
        self.assert_live()
        n,r = self._factor.shape
        _require(type(z) is torch.Tensor and z.dtype == torch.float64 and z.device == self._factor.device
                 and not z.requires_grad and z.is_contiguous() and z.ndim == 2 and z.shape[0] == r
                 and 0 < z.shape[1] <= self._design.maximum_lanes
                 and bool(torch.isfinite(z).all().item()), "rounded Q direction admission failed")
        projected = self._factor @ z
        mass = torch.zeros_like(projected)
        # Bound workspace independently of full factor size.
        for start,stop in self._spans:
            for cut in self._axis.cuts:
                width = max(1,self._design.route_tile_size//cut.routes_per_fiber)*cut.routes_per_fiber
                for lo in range(cut.route_offset,cut.route_offset+cut.route_count,width):
                    hi=min(lo+width,cut.route_offset+cut.route_count)
                    mass[start:stop].add_(self._factor[start:stop,lo:hi].abs() @ z[lo:hi].abs())
        # At most r tiles per row, plus dot products: this deliberately charges
        # every route as a separate accumulation, independent of tile size.
        mass = _mass_upper(mass, 4*r+16)
        radius = _rounding_radius(mass, 2*r+8)
        value = projected.square().sum(dim=0)
        evaluation_error = _mass_upper(_up(_up(projected.abs()*radius)*2)+_up(radius.square()), 8)
        error = _mass_upper(evaluation_error.sum(dim=0), n+8)
        error = _up(error + _rounding_radius(_mass_upper(value,2*n+8),2*n+8))
        norm_squared = _mass_upper(z.square().sum(dim=0),2*r+8)
        error = _up(error + _up(norm_squared*self.receipt.coordinates["roundoff_residual_spectral_bound"]))
        upper = _up(value+error)
        lower = torch.nextafter(value-error,torch.full_like(value,-math.inf)).clamp_min(0)
        _require(bool(torch.isfinite(upper).all().item()), "nonfinite rounded Q enclosure")
        self.assert_live()
        return value,lower,upper

    def close(self):
        if not self._closed:
            _require(self._lease_count == 0, "cannot close leased rounded Q handle")
            self._closed = True
            self._factor = self._gate = self._singleton = None
            self._binding = self._handoff = self._mask = None
            self._pairs = ()

    evict = close

    def __enter__(self):
        self.assert_live()
        return self

    def __exit__(self,*_exc):
        self.close()

    def __reduce_ex__(self,_protocol):
        raise TypeError("rounded Q handles are nonserializable")


class V1FactoredRoundedQLease:
    def __init__(self, owner, *, _issuance_token):
        _require(_issuance_token is _LEASE_TOKEN and type(owner) is V1FactoredRoundedQDeviceHandle,
                 "rounded Q leases are owner-issued only")
        self._owner, self._closed = owner, False

    def _live(self):
        _require(not self._closed, "rounded Q lease closed")
        self._owner.assert_live()
        return self._owner

    @property
    def factor(self):
        return self._live()._factor

    @property
    def singleton_work(self):
        return self._live()._singleton

    @property
    def receipt(self):
        return self._live().receipt

    def close(self):
        if not self._closed:
            self._closed = True
            self._owner._lease_count = 0
            self._owner = None

    def __enter__(self):
        self._live()
        return self

    def __exit__(self,*_exc):
        self.close()

    def __reduce_ex__(self,_protocol):
        raise TypeError("rounded Q leases are nonserializable")
