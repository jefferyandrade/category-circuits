"""Source-bound whole-session native-fiber corrected-A operator.

This sibling owns no U/B factorization or row spool and grants no decision
permission. It is consumed while the registered factored endpoint remains
live; closing it never evicts its parent session. Solver promotion requires
an explicit adapter rather than impersonating the compact-D operator type.
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
from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from .modern_v1_sampled_dual_chart_row_custody import V1SampledDualChartRowRole

_A = V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY
_PREFIX = "partition-free-v1-factored-signed-work-"


class V1FactoredSignedWorkRefusal(ValueError):
    """A native source, numerical or resource admission did not close."""


def _require(condition, message):
    if not condition:
        raise V1FactoredSignedWorkRefusal(message)


def _positive_int(value, name):
    _require(type(value) is int and value > 0, f"{name} must be a positive exact integer")
    return value


def _custody(t):
    return (id(t), str(t.device), t.dtype, tuple(t.shape), tuple(t.stride()),
            t.storage_offset(), t.data_ptr(), t.untyped_storage().data_ptr(), t._version, t.requires_grad)


def _no_factor_autograd(packet):
    _require(all(not getattr(layer, name).requires_grad for layer in packet.layers
                 for name in ("input", "post_activation_cotangent", "preactivation", "bias_baseline", "weight")),
             "factored numerical operator refuses autograd factor retention")


def _error_constants(operations):
    """Outward constants for binary64 RN/FTZ contractions on finite inputs."""
    _positive_int(operations, "rounding operation count")
    _require(operations < 2**52, "contraction is too long for numerical enclosure")
    gamma = math.nextafter(float(Fraction(operations, 2**53 - operations)), math.inf)
    _require(gamma < 0.5, "contraction gamma is not admitted")
    denominator = math.nextafter(1.0 - gamma, 0.0)
    floor = math.nextafter(operations * torch.finfo(torch.float64).tiny, math.inf)
    return gamma, denominator, floor


def _up(value):
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _mass_upper(value, operations):
    """Enclose exact nonnegative term mass from its rounded accumulation.

    The absolute floor precedes division so its propagated contribution is
    inflated too. Input/output subnormal loss is bounded by the minimum normal
    FP64 value per charged operation; this is deliberately conservative.
    """
    _, denominator, floor = _error_constants(operations)
    return _up(_up(value + floor) / denominator)


def _down(value):
    return torch.nextafter(value, torch.full_like(value, -math.inf))


def _mass_lower(value, operations):
    """Lower-enclose the same nonnegative RN/FTZ reduction as _mass_upper."""
    gamma, _, floor = _error_constants(operations)
    denominator = math.nextafter(1.0 + gamma, math.inf)
    return _down(_down(value - floor) / denominator).clamp_min(0.0)


def _objective_weight_interval(weight, count, cut_count):
    """Binary64 endpoints around the exact registered per-context objective mass."""
    _positive_int(count, "stratum count")
    _positive_int(cut_count, "cut count")
    _require(math.isfinite(weight) and weight >= 0.0, "invalid objective stratum weight")
    exact = Fraction.from_float(weight) / (count * cut_count)
    center = float(exact)
    represented = Fraction.from_float(center)
    lower = math.nextafter(center, 0.0) if represented > exact else center
    upper = math.nextafter(center, math.inf) if represented < exact else center
    _require(Fraction.from_float(lower) <= exact <= Fraction.from_float(upper),
             "objective coefficient interval failed")
    return lower, upper


@dataclass(frozen=True)
class V1FactoredSignedWorkDesign:
    maximum_lanes: int
    additional_memory_budget_bytes: int
    route_tile_size: int = 262144
    legacy_conv_row_budget_bytes: int = 256 << 20
    allow_private_cpu_reference: bool = False
    schema: str = "factored_native_signed_work_design_v1"

    def __post_init__(self):
        for name in ("maximum_lanes", "additional_memory_budget_bytes", "route_tile_size"):
            _positive_int(getattr(self, name), name)
        _require(type(self.legacy_conv_row_budget_bytes) is int and self.legacy_conv_row_budget_bytes >= 0,
                 "legacy conv row budget must be a nonnegative exact integer")
        _require(type(self.allow_private_cpu_reference) is bool, "exact CPU reference flag required")
        _require(self.schema == "factored_native_signed_work_design_v1", "design schema changed")

    @property
    def design_hash(self):
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True)
class V1FactoredSignedWorkReceipt:
    coordinates: Mapping[str, Any]
    schema: str = "factored_native_signed_work_receipt_v1"
    action_authority: bool = False
    d_decision_authority: bool = False
    c_measurement_authority: bool = False
    calibration_k_authority: bool = False
    paired_oracle_o_authority: bool = False
    population_enclosure_authority: bool = False
    production_geometry_authority: bool = False
    kkt_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    _receipt_hash: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "coordinates", MappingProxyType(dict(self.coordinates)))
        object.__setattr__(self, "_receipt_hash", self._hash())
        self.assert_integrity()

    def to_json_payload(self):
        return {k: dict(v) if k == "coordinates" else v for k, v in vars(self).items()
                if not k.startswith("_")}

    def _hash(self):
        return content_hash(self.to_json_payload(), prefix=_PREFIX + "receipt-")

    @property
    def receipt_hash(self):
        return self._receipt_hash

    def assert_integrity(self):
        _require(self._hash() == self._receipt_hash, "operator receipt changed")
        _require(self.schema == "factored_native_signed_work_receipt_v1" and all(
            v is False for k, v in vars(self).items() if k.endswith("_authority")),
            "numerical operator cannot grant downstream authority")
        return self.receipt_hash


_LEAST_WORK_LEASE_TOKEN = object()


class V1FactoredSignedWorkOperator:
    """Whole-session corrected-A linear pair with live native source custody."""

    def __init__(self, *, source_binding: V1FactoredDAlphaSourceBindingReceipt,
                 native_fiber_axis: V1NativeDestinationFiberAxisReceipt,
                 current_mask: HardMask, design: V1FactoredSignedWorkDesign,
                 linear_acceleration_design=None):
        _require(type(source_binding) is V1FactoredDAlphaSourceBindingReceipt, "exact native source binding required")
        binding_hash = source_binding.assert_live()
        _require(type(design) is V1FactoredSignedWorkDesign, "exact operator design required")
        design.__post_init__()
        _require(type(native_fiber_axis) is V1NativeDestinationFiberAxisReceipt, "exact native fiber axis required")
        native_fiber_axis.assert_integrity()
        _require(type(current_mask) is HardMask, "exact current hard mask required")
        handoff = source_binding._parents[2]
        _require(type(handoff) is V1FactoredDeviceResidentDLiveHandoff, "exact live factored endpoint required")
        handoff.assert_live()
        _require(handoff.session.is_finalized and handoff.finalization_receipt is not None
                 and handoff.replay_boundary.live_endpoint_registered,
                 "signed work requires the registered finalized endpoint")
        source = source_binding.coordinates
        _require(current_mask.mask_hash == source["current_mask_hash"], "current mask changed")
        pairs = handoff.session.packets
        _require(bool(pairs), "empty factored session")
        counts = handoff.session.stratum_counts
        _require(all(n > 0 for n in counts), "every objective stratum needs observed contexts")
        weights = handoff.session.design.stratum_weights
        _require(counts == source["registered_look_stratum_counts"]
                 and handoff.session._active is None, "final prefix/source count mismatch")
        device = pairs[0][1].device
        if device.type == "cpu":
            _require(design.allow_private_cpu_reference
                     and handoff.session.design.execution_mode is V1FactoredDeviceResidentDSessionMode.CPU_REFERENCE_TEST_ONLY,
                     "production signed work refuses CPU fallback")
        else:
            _require(device.type == "cuda" and torch.cuda.is_available()
                     and handoff.session.design.execution_mode is V1FactoredDeviceResidentDSessionMode.CUDA_PRODUCTION,
                     "CUDA source mode mismatch")
        context_weights, spans, offset = [], [], 0
        for stratum, packet in pairs:
            _require(packet.device == device, "packet devices differ")
            packet._native_fiber_axis(native_fiber_axis)
            _no_factor_autograd(packet)
            spans.append((offset, offset + packet.context_count))
            offset += packet.context_count
            context_weights.extend([float(Fraction.from_float(weights[stratum]) / counts[stratum])] * packet.context_count)
        _require(offset == sum(counts), "packet context counts differ from finalization")
        _require(abs(math.fsum(context_weights) - 1.0) <= 4 * math.ulp(1.0),
                 "context measure does not sum to one")
        R, F, L = native_fiber_axis.route_count, native_fiber_axis.fiber_count, design.maximum_lanes
        owned_bytes = (R + offset) * 8
        result_bytes = max(R * L, offset * F * L) * 8
        row_scratch, conv_payload, prepared_linear_cotangent_bytes = 0, 0, 0
        for _, packet in pairs:
            for cut, layer in zip(native_fiber_axis.cuts, packet.layers, strict=True):
                routes = min(cut.route_count, max(1, design.route_tile_size // cut.routes_per_fiber) * cut.routes_per_fiber)
                # FP32 tile plus FP64 tile and absolute/error-analysis copies;
                # FP64 contraction input/result scratch is charged separately.
                row_scratch = max(row_scratch, packet.context_count * routes * 40 + routes * L * 16)
                if layer.reducer_kind != "linear_weight":
                    conv_payload = max(conv_payload, packet.context_count * layer.route_count * 8)
                else:
                    prepared_linear_cotangent_bytes = max(prepared_linear_cotangent_bytes,
                        layer.post_activation_cotangent.numel() * layer.post_activation_cotangent.element_size())
        _require(conv_payload <= design.legacy_conv_row_budget_bytes, "legacy conv payload not admitted")
        provider_workspace = handoff.replay_boundary.provider_peak_workspace_bytes
        # ReLU preparation holds full-destination sign masks and successive
        # FP32 fraction expressions; scale division adds another FP32 output.
        # Eight FP32-sized slots cover that source-level temporary overlap.
        linear_cotangent_preparation_workspace_bytes = 8 * prepared_linear_cotangent_bytes
        # Signed-work replay retains cell centers, absolute mass, both interval
        # endpoints and directed-rounding temporaries. Charge their overlap.
        additional = (owned_bytes + 16 * result_bytes + row_scratch + conv_payload
                      + provider_workspace + prepared_linear_cotangent_bytes
                      + linear_cotangent_preparation_workspace_bytes)
        acceleration_forecast = None
        if linear_acceleration_design is not None:
            from .modern_v1_factored_signed_work_linear_acceleration import forecast_linear_acceleration
            acceleration_forecast = forecast_linear_acceleration(
                pairs=pairs, axis=native_fiber_axis, route_tile_size=design.route_tile_size,
                design=linear_acceleration_design, maximum_lanes=design.maximum_lanes)
            additional += acceleration_forecast["extra_required_bytes"]
        _require(additional <= design.additional_memory_budget_bytes, "additional operator memory not admitted")
        if device.type == "cuda":
            _require(additional <= cuda_reusable_memory_bytes(device, framework=torch), "insufficient currently free CUDA memory")
        self._closed = False
        self._least_work_lease_count = 0
        self._binding, self._handoff, self._mask = source_binding, handoff, current_mask
        self._binding_hash = binding_hash
        self._axis, self._design, self._device_obj = native_fiber_axis, design, device
        self._pairs, self._spans = pairs, tuple(spans)
        self._gate = pairs[0][1].gate64(current_mask)
        self._weights = torch.tensor(context_weights, dtype=torch.float64, device=device)
        self._owned_custody = (_custody(self._gate), _custody(self._weights))
        self._design_hash = design.design_hash
        self._axis_hash = native_fiber_axis.receipt_hash
        self._context_count = offset
        self._boundary_hash = handoff.replay_boundary.boundary_hash
        self._finalization_hash = handoff.finalization_receipt.receipt_hash
        self._session_design_hash = handoff.session.design.design_hash
        self._linear_acceleration = None
        if linear_acceleration_design is not None:
            from .modern_v1_factored_signed_work_linear_acceleration import _LinearAcceleration
            self._linear_acceleration = _LinearAcceleration(
                pairs=self._pairs, spans=self._spans, axis=self._axis, gate=self._gate,
                design=linear_acceleration_design, route_tile_size=design.route_tile_size,
                legacy_conv_row_budget_bytes=design.legacy_conv_row_budget_bytes,
                allow_private_cpu_reference=design.allow_private_cpu_reference,
                maximum_lanes=design.maximum_lanes)
        # One cold source pass. A rounded norm is never called an upper bound.
        squared = torch.zeros((), dtype=torch.float64, device=device)
        tile_count = 0
        for start, stop, first, last, lo, hi, rows in self._tiles():
            if self._linear_acceleration is not None:
                self._linear_acceleration.audit_rows(start, stop, first, last, lo, hi, rows)
            squared.add_(rows.square().sum())
            tile_count += 1
            del rows
        squared_upper = _mass_upper(squared, 2 * offset * R + 8 * tile_count + 16)
        norm_upper = _up(torch.sqrt(squared_upper))
        _require(bool(torch.isfinite(norm_upper).item()), "nonfinite corrected-A coefficients or Frobenius bound")
        self._frobenius_upper = float(norm_upper.item())
        acceleration_coordinates = {}
        if self._linear_acceleration is not None:
            self._linear_acceleration.finish_audit()
            acceleration_coordinates["linear_acceleration"] = self._linear_acceleration.coordinates
        self._receipt = V1FactoredSignedWorkReceipt(coordinates=dict(
            source_binding_hash=binding_hash, replay_boundary_hash=handoff.replay_boundary.boundary_hash,
            finalization_receipt_hash=handoff.finalization_receipt.receipt_hash,
            exact_geometry_receipt_hash=source["exact_geometry_receipt_hash"],
            packet_receipt_hashes=source["packet_receipt_hashes"], look_receipt_hashes=source["look_receipt_hashes"],
            packet_strata=tuple(s for s, _ in pairs), packet_context_spans=self._spans,
            context_weights=tuple(context_weights), stratum_counts=tuple(counts), stratum_weights=tuple(weights),
            current_mask_hash=current_mask.mask_hash, chart_identity_hash=source["chart_identity_hash"],
            proposal_generation=source["proposal_generation"],
            corrected_a_stream_identity_hash=source["corrected_a_stream_identity_hash"],
            native_fiber_axis_hash=self._axis_hash, native_fiber_measure_hash=native_fiber_axis.measure_hash,
            physical_route_axis_hash=source["physical_route_axis_hash"], route_id_axis_hash=source["route_id_axis_hash"],
            route_count=R, context_count=offset, fiber_count=F, design_hash=design.design_hash,
            factorization_kind="factored_packet_stream_corrected_A_legacy_rows", row_spool_receipt_hash=None,
            route_scale_fiber_id_array_materialized=False, logical_zero_residual=True,
            device=str(device), private_cpu_reference_execution=device.type == "cpu",
            arithmetic_law="legacy_FP32_rows_FP64_RN_or_FTZ_gamma_plus_absolute_floor_v1",
            solver_objective_weight_law="stored_binary64_context_weights_over_cut_count",
            public_signed_work_enclosure_weight_law="exact_registered_stratum_weight_fraction_over_count_over_cut_count",
            public_signed_work_enclosure_scope="candidate_primal_objective_only_no_dual_optimality",
            operator_frobenius_norm_upper=self._frobenius_upper, norm_tile_count=tile_count,
            owned_bytes=owned_bytes, maximum_result_bytes=result_bytes, row_scratch_bytes=row_scratch,
            prepared_linear_cotangent_bytes=prepared_linear_cotangent_bytes,
            linear_cotangent_preparation_workspace_bytes=linear_cotangent_preparation_workspace_bytes,
            maximum_result_buffer_reserve_count=16, result_buffer_reserve_bytes=16 * result_bytes,
            legacy_conv_payload_bytes=conv_payload, provider_workspace_bytes=provider_workspace,
            additional_required_bytes=additional, memory_forecast_scope="incremental_operator_plus_registered_provider_workspace_not_total_process_peak",
            **acceleration_coordinates))
        self._receipt_hash = self._receipt.receipt_hash
        self.assert_live()

    @property
    def receipt(self):
        return self._receipt

    @property
    def receipt_hash(self):
        return self._receipt_hash

    @property
    def route_count(self):
        return self._axis.route_count

    @property
    def context_count(self):
        return self._context_count

    @property
    def fiber_count(self):
        return self._axis.fiber_count

    @property
    def context_weights(self):
        self.assert_live()
        return self._weights

    @property
    def device(self):
        return str(self._device_obj)

    @property
    def operator_frobenius_norm_upper(self):
        self.assert_live()
        return self._frobenius_upper

    @property
    def native_fiber_axis_receipt(self):
        return self._axis

    @property
    def residual_is_logical_zero(self):
        return True

    def _assert_custody(self):
        _require(not self._closed, "factored operator closed")
        _require(self.receipt.assert_integrity() == self._receipt_hash, "operator receipt replaced")
        _require((_custody(self._gate), _custody(self._weights)) == self._owned_custody, "owned operator tensors changed")
        _require(self._design.design_hash == self._design_hash, "operator design changed")
        acceleration_receipt = self.receipt.coordinates.get("linear_acceleration")
        _require((self._linear_acceleration is None) == (acceleration_receipt is None),
                 "linear acceleration owner replaced")
        if self._linear_acceleration is not None:
            self._linear_acceleration.assert_live()
            _require(self._linear_acceleration.coordinates == acceleration_receipt,
                     "linear acceleration receipt changed")
        _require(self._handoff.session.is_finalized and self._handoff.session._active is None
                 and self._handoff.session.design.design_hash == self._session_design_hash
                 and self._handoff.finalization_receipt is not None
                 and self._handoff.finalization_receipt.receipt_hash == self._finalization_hash
                 and self._handoff.replay_boundary.boundary_hash == self._boundary_hash,
                 "finalized source endpoint changed")
        _require(tuple(self._handoff.session.stratum_counts) == self.receipt.coordinates["stratum_counts"]
                 and tuple(self._handoff.session.design.stratum_weights) == self.receipt.coordinates["stratum_weights"],
                 "registered objective counts or weights changed")
        self._axis.assert_integrity()
        _require(self._axis.receipt_hash == self._axis_hash, "native fiber axis changed")
        _require(self._handoff.session.packets == self._pairs, "packet owner/order changed")
        for _, packet in self._pairs:
            packet.assert_integrity()
            _no_factor_autograd(packet)

    def assert_live(self):
        self._assert_custody()
        _require(self._binding.assert_live() == self._binding_hash, "native source changed")
        self._axis.assert_integrity()
        _require(self._axis.receipt_hash == self._axis_hash, "native fiber axis changed")
        _require(self._mask.mask_hash == self.receipt.coordinates["current_mask_hash"], "mask source changed")
        return self.receipt.assert_integrity()

    def _validate(self, value, *, dual):
        self.assert_live()
        _require(torch.is_tensor(value) and value.dtype == torch.float64 and value.device == self._device_obj,
                 "operator inputs must already be FP64 tensors on the source device")
        _require(not value.requires_grad, "numerical operator refuses autograd input retention")
        prefix = (self.context_count, self.fiber_count) if dual else (self.route_count,)
        _require(value.ndim == len(prefix) + 1 and tuple(value.shape[:-1]) == prefix
                 and 0 < value.shape[-1] <= self._design.maximum_lanes, "operator lane/shape admission failed")
        _require(bool(torch.isfinite(value).all().item()), "nonfinite operator input")
        return value

    def _tiles(self, *, skip_accelerated_linear=False):
        if skip_accelerated_linear and self._linear_acceleration is not None:
            yield from self._linear_acceleration.convolution_tiles()
            return
        for (_, packet), (start, stop) in zip(self._pairs, self._spans, strict=True):
            for first, last, lo, hi, rows in packet._native_fiber_row_tiles(
                    _A, self._axis, self._gate, self._design.route_tile_size,
                    self._design.legacy_conv_row_budget_bytes):
                yield start, stop, first, last, lo, hi, rows
                del rows

    def _forward_prevalidated(self, z):
        """Solver-owned shape/finite tensors only; no source I/O here.

        Legacy nonunit scale normalization may synchronize a scalar finite
        check. That path must be profiled separately before hot-loop promotion.
        """
        self._assert_custody()
        _require(not z.requires_grad, "numerical operator refuses autograd input retention")
        out = torch.empty((self.context_count, self.fiber_count, z.shape[1]), dtype=torch.float64, device=self._device_obj)
        if self._linear_acceleration is not None:
            self._linear_acceleration.forward_into(z, out)
        for start, stop, first, last, lo, hi, rows in self._tiles(skip_accelerated_linear=True):
            out[start:stop, first:last] = torch.einsum("bfs,fsl->bfl", rows, z[lo:hi].reshape(last-first, -1, z.shape[1]))
            del rows
        return out

    def _adjoint_prevalidated(self, y):
        self._assert_custody()
        _require(not y.requires_grad, "numerical operator refuses autograd input retention")
        out = torch.zeros((self.route_count, y.shape[2]), dtype=torch.float64, device=self._device_obj)
        if self._linear_acceleration is not None:
            self._linear_acceleration.adjoint_into(y, out)
        for start, stop, first, last, lo, hi, rows in self._tiles(skip_accelerated_linear=True):
            out[lo:hi].add_(torch.einsum("bfs,bfl->fsl", rows, y[start:stop, first:last]).reshape(hi-lo, y.shape[2]))
            del rows
        return out

    def _absolute_forward_envelope_prevalidated(self, z):
        self._assert_custody()
        _require(not z.requires_grad, "numerical operator refuses autograd input retention")
        out = torch.empty((self.context_count, self.fiber_count, z.shape[1]), dtype=torch.float64, device=self._device_obj)
        if self._linear_acceleration is not None:
            self._linear_acceleration.forward_into(z, out, absolute=True)
            for cut in self._axis.cuts:
                if cut.route_kind == "linear_weight":
                    section = slice(cut.fiber_offset, cut.fiber_offset+cut.fiber_count)
                    out[:, section] = _mass_upper(out[:, section], 2*cut.routes_per_fiber+8)
        for start, stop, first, last, lo, hi, rows in self._tiles(skip_accelerated_linear=True):
            mass = torch.einsum("bfs,fsl->bfl", rows.abs(), z[lo:hi].abs().reshape(last-first, -1, z.shape[1]))
            out[start:stop, first:last] = _mass_upper(mass, 2 * rows.shape[2] + 8)
            del rows, mass
        return out

    def absolute_forward_envelope(self, z):
        result = self._absolute_forward_envelope_prevalidated(self._validate(z, dual=False))
        _require(bool(torch.isfinite(result).all().item()), "nonfinite forward mass upper")
        self.assert_live()
        return result

    def absolute_adjoint_envelope(self, y):
        y = self._validate(y, dual=True)
        mass = torch.zeros((self.route_count, y.shape[2]), dtype=torch.float64, device=self._device_obj)
        if self._linear_acceleration is not None:
            self._linear_acceleration.adjoint_into(y, mass, absolute=True)
        for start, stop, first, last, lo, hi, rows in self._tiles(skip_accelerated_linear=True):
            mass[lo:hi].add_(torch.einsum("bfs,bfl->fsl", rows.abs(), y[start:stop, first:last].abs()).reshape(hi-lo, y.shape[2]))
            del rows
        result = _mass_upper(mass, 2 * self.context_count + 8 * len(self._pairs) + 16)
        _require(bool(torch.isfinite(result).all().item()), "nonfinite adjoint mass upper")
        self.assert_live()
        return result

    def forward(self, z):
        result = self._forward_prevalidated(self._validate(z, dual=False))
        _require(bool(torch.isfinite(result).all().item()), "nonfinite native forward")
        self.assert_live()
        return result

    def adjoint(self, y):
        result = self._adjoint_prevalidated(self._validate(y, dual=True))
        _require(bool(torch.isfinite(result).all().item()), "nonfinite native adjoint")
        self.assert_live()
        return result

    def signed_work_enclosure(self, z):
        """Return stored-weight value and an exact registered-weight F interval.

        Cell centers preserve signed cancellation. Absolute row mass is used
        only to enclose FP64 RN/FTZ contraction error. The objective measure is
        exactly stratum_weight / (stratum_count * cut_count), whereas the
        numerical solver uses stored context weights divided in binary64.
        This primal interval does not certify the solver's raw dual lower.
        """
        _require(type(z) is torch.Tensor and z.is_contiguous(),
                 "signed-work enclosure needs an exact contiguous tensor")
        z = self._validate(z, dual=False)
        _require(self.residual_is_logical_zero is True, "signed-work enclosure requires logical zero residual")
        _require(self._device_obj.type != "cuda" or
                 (torch.are_deterministic_algorithms_enabled()
                  and not torch.is_deterministic_algorithms_warn_only_enabled()),
                 "signed-work CUDA replay requires deterministic algorithms")
        values = self._forward_prevalidated(z)
        mass = self._absolute_forward_envelope_prevalidated(z)
        lower_cells, upper_cells = torch.empty_like(values), torch.empty_like(values)
        for cut in self._axis.cuts:
            section = slice(cut.fiber_offset, cut.fiber_offset + cut.fiber_count)
            gamma, denominator, floor = _error_constants(2 * cut.routes_per_fiber + 8)
            radius = _up(_up(_up(mass[:, section] * gamma) + floor) / denominator)
            center = values[:, section].abs()
            lower_cells[:, section] = _down(center - radius).clamp_min(0.0)
            upper_cells[:, section] = _up(center + radius)
        counts = self.receipt.coordinates["stratum_counts"]
        weights = self.receipt.coordinates["stratum_weights"]
        coefficients = tuple(_objective_weight_interval(w, n, self._axis.cut_count)
                             for w, n in zip(weights, counts, strict=True))
        lower_weights, upper_weights = [], []
        for stratum, packet in self._pairs:
            lo, hi = coefficients[stratum]
            lower_weights.extend([lo] * packet.context_count)
            upper_weights.extend([hi] * packet.context_count)
        lo = torch.tensor(lower_weights, dtype=torch.float64, device=self._device_obj).reshape(-1, 1, 1)
        hi = torch.tensor(upper_weights, dtype=torch.float64, device=self._device_obj).reshape(-1, 1, 1)
        operations = 2 * self.context_count * self.fiber_count + 16
        lower = _mass_lower(_down(lower_cells * lo).clamp_min(0.0).sum(dim=(0, 1)), operations)
        upper = _mass_upper(_up(upper_cells * hi).sum(dim=(0, 1)), operations)
        represented = ((self._weights / float(self._axis.cut_count)).reshape(-1, 1, 1)
                       * values.abs()).sum(dim=(0, 1))
        _require(bool(torch.isfinite(represented).all().item())
                 and bool(torch.isfinite(lower).all().item())
                 and bool(torch.isfinite(upper).all().item())
                 and bool((lower <= upper).all().item()), "nonfinite or reversed signed-work interval")
        self.assert_live()
        return represented, lower, upper

    def borrow_for_least_work(self, *, expected_receipt_hash,
                             expected_source_binding_hash, expected_device):
        self.assert_live()
        _require(expected_receipt_hash == self.receipt_hash
                 and expected_source_binding_hash == self._binding_hash
                 and expected_device == self.device, "native least-work A source differs")
        _require(self._least_work_lease_count == 0, "native signed-work A already leased")
        lease = V1FactoredSignedWorkLeastWorkLease(self, token=_LEAST_WORK_LEASE_TOKEN)
        self._least_work_lease_count = 1
        return lease

    def close(self):
        if not self._closed:
            _require(self._least_work_lease_count == 0, "cannot close leased native signed-work A")
            self._closed = True
            if self._linear_acceleration is not None:
                self._linear_acceleration.close()
                self._linear_acceleration = None
            self._gate = self._weights = None
            self._pairs = ()
            self._binding = self._handoff = self._mask = None

    evict = close

    def __enter__(self):
        self.assert_live()
        return self

    def __exit__(self, *_exc):
        self.close()

    def __reduce_ex__(self, _protocol):
        raise TypeError("live factored signed-work operators are nonserializable")


class V1FactoredSignedWorkLeastWorkLease:
    """Exclusive live A ownership for native least-work numerical state."""
    def __init__(self, owner, *, token):
        _require(token is _LEAST_WORK_LEASE_TOKEN and type(owner) is V1FactoredSignedWorkOperator,
                 "native signed-work A leases are owner-issued only")
        self._owner = self._issued_owner = owner
        self._receipt_hash = owner.receipt_hash
        self._closed = False

    @property
    def operator(self):
        _require(not self._closed and self._owner is self._issued_owner,
                 "native signed-work A lease closed or owner changed")
        _require(self._owner.assert_live() == self._receipt_hash,
                 "native signed-work A lease identity changed")
        return self._owner

    def close(self):
        if not self._closed:
            self._closed = True
            self._issued_owner._least_work_lease_count = 0
            self._owner = self._issued_owner = None

    def __enter__(self):
        _ = self.operator
        return self

    def __exit__(self, *_exc):
        self.close()

    def __reduce_ex__(self, _protocol):
        raise TypeError("native signed-work A leases are nonserializable")
