"""Opt-in linear-only A contraction owner; default A remains byte-compatible.

Every enabled source is audited against every eager linear coefficient during
A's existing cold norm pass. CPU mode is an explicit mathematical reference,
never a substitute for Triton compile, performance or production qualification.
"""
from dataclasses import dataclass
import importlib

import torch

from mechanistic_systems.contracts.content_address import content_hash
from .modern_v1_factored_row_capture import V1FactoredRowPacket, _ROLE_QUERY_INDEX
from .modern_v1_sampled_dual_chart_row_custody import V1SampledDualChartRowRole

_A = V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY
_PREFIX = 'partition-free-v1-factored-A-linear-acceleration-'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _custody(t):
    return (id(t), str(t.device), t.dtype, tuple(t.shape), tuple(t.stride()),
            t.storage_offset(), t.data_ptr(), t.untyped_storage().data_ptr(),
            t.untyped_storage().nbytes(), t._version, t.requires_grad)


@dataclass(frozen=True)
class V1FactoredSignedWorkLinearAccelerationDesign:
    allow_private_cpu_reference: bool = False
    maximum_linear_inner: int = 16384
    maximum_packet_contexts: int = 128
    schema: str = 'experimental_factored_A_linear_fused_v1'

    def __post_init__(self):
        _require(type(self.allow_private_cpu_reference) is bool, 'exact private CPU flag required')
        _require(type(self.maximum_linear_inner) is int and 0 < self.maximum_linear_inner <= 16384,
                 'linear inner limit must be an exact integer at most 16384')
        _require(type(self.maximum_packet_contexts) is int and 0 < self.maximum_packet_contexts <= 128,
                 'packet context limit must be an exact integer at most 128')
        _require(self.schema == 'experimental_factored_A_linear_fused_v1', 'acceleration schema changed')

    @property
    def design_hash(self):
        return content_hash(self, prefix=_PREFIX + 'design-')


def forecast_linear_acceleration(*, pairs, axis, route_tile_size, design, maximum_lanes):
    _require(type(design) is V1FactoredSignedWorkLinearAccelerationDesign, 'exact acceleration design required')
    design.__post_init__()
    _require(type(maximum_lanes) is int and 0 < maximum_lanes <= 8, 'first linear kernel joint admits at most eight exact lanes')
    context_count = sum(packet.context_count for _,packet in pairs if type(packet) is V1FactoredRowPacket)
    _require(axis.route_count*maximum_lanes < 2**31
             and context_count*axis.fiber_count*maximum_lanes < 2**31,
             'first linear kernel index range is not admitted')
    cache, audit, cells, layers, cpu_workspace, partial_workspace = 0, 0, 0, 0, 0, 0
    for _, packet in pairs:
        _require(type(packet) is V1FactoredRowPacket, 'exact source packet required')
        packet._native_fiber_axis(axis)
        for cut, layer in zip(axis.cuts, packet.layers, strict=True):
            if layer.reducer_kind != 'linear_weight':
                continue
            _require(cut.routes_per_fiber <= design.maximum_linear_inner
                     and packet.context_count <= design.maximum_packet_contexts,
                     'linear CUDA kernel shape is outside the admitted first joint')
            _require(layer.input.dtype == layer.weight.dtype == torch.float32
                     and layer.input.ndim == layer.weight.ndim == 2,
                     'linear kernel requires actual FP32 matrix factors')
            _require(tuple(layer.input.shape) == (packet.context_count, cut.routes_per_fiber),
                     'captured input does not match the complete native fiber')
            for factor in (layer.input, layer.weight):
                _require(all(stride >= 0 for stride in factor.stride()) and
                         sum((size-1)*stride for size,stride in zip(factor.shape,factor.stride(),strict=True)) < 2**31,
                         'captured linear factor stride range is not admitted')
            cache += packet.context_count * cut.fiber_count * 4
            tile = min(cut.route_count, max(1, route_tile_size // cut.routes_per_fiber) * cut.routes_per_fiber)
            # Extra audit rows plus bitwise comparison mask; the existing A
            # forecast already owns the simultaneously live eager FP32/64 rows.
            _require(packet.context_count*tile < 2**31 and
                     packet.context_count*cut.fiber_count*maximum_lanes*((cut.routes_per_fiber+255)//256) < 2**31,
                     'cold audit or partial workspace index range is not admitted')
            audit = max(audit, 9 * packet.context_count * tile)
            cpu_workspace = max(cpu_workspace, packet.context_count * tile * (maximum_lanes*8+16))
            partial_workspace = max(partial_workspace, packet.context_count * cut.fiber_count
                                    * maximum_lanes * ((cut.routes_per_fiber+255)//256) * 8)
            cells += packet.context_count * cut.route_count
            layers += 1
    _require(layers > 0, 'no linear cut available for acceleration')
    cpu_workspace = cpu_workspace if pairs[0][1].device.type == 'cpu' else 0
    partial_workspace = partial_workspace if pairs[0][1].device.type == 'cuda' else 0
    return dict(prepared_cotangent_cache_bytes=cache, extra_cold_row_audit_bytes=audit,
                extra_private_cpu_reference_workspace_bytes=cpu_workspace,
                extra_required_bytes=cache + audit + cpu_workspace + partial_workspace, linear_packet_layer_count=layers,
                expected_cold_audited_coefficient_count=cells,
                hot_global_partial_workspace_bytes=partial_workspace,
                maximum_lanes=maximum_lanes, kernel_index_upper_exclusive=2**31,
                compiler_allocator_workspace_qualification_verified=False)


@dataclass(frozen=True)
class _Entry:
    start: int
    stop: int
    cut: object
    layer: object
    cotangent: object
    descriptor_hash: str
    cotangent_custody: tuple


class _LinearAcceleration:
    """Internal child of the exact native A owner; no independent leases."""

    def __init__(self, *, pairs, spans, axis, gate, design, route_tile_size,
                 legacy_conv_row_budget_bytes, allow_private_cpu_reference, maximum_lanes):
        self._closed = False
        self._pairs, self._spans, self._axis, self._gate = pairs, spans, axis, gate
        self._design, self._design_hash = design, design.design_hash
        self._route_tile_size, self._conv_budget = route_tile_size, legacy_conv_row_budget_bytes
        self._maximum_lanes = maximum_lanes
        self._forecast = forecast_linear_acceleration(pairs=pairs, axis=axis,
            route_tile_size=route_tile_size, design=design, maximum_lanes=maximum_lanes)
        _require(type(gate) is torch.Tensor and gate.dtype == torch.float64
                 and tuple(gate.shape) == (axis.route_count,) and gate.is_contiguous()
                 and not gate.requires_grad, 'linear acceleration requires the exact contiguous FP64 source gate')
        self._device = gate.device
        self._cpu = gate.device.type == 'cpu'
        _require(not self._cpu or (design.allow_private_cpu_reference and allow_private_cpu_reference),
                 'linear CUDA acceleration refuses CPU fallback')
        _require(self._cpu or (gate.device.type == 'cuda' and torch.cuda.is_available()),
                 'linear CUDA acceleration requires CUDA')
        if not self._cpu:
            _require(torch.are_deterministic_algorithms_enabled()
                     and not torch.is_deterministic_algorithms_warn_only_enabled(),
                     'linear CUDA acceleration requires strict deterministic execution')
            self._kernels = importlib.import_module(
                '.modern_v1_factored_signed_work_triton_kernels', __package__)
            self._triton = importlib.import_module('triton')
        else:
            self._kernels = self._triton = None
        entries = []
        for (_, packet), (start, stop) in zip(pairs, spans, strict=True):
            for cut, layer in zip(axis.cuts, packet.layers, strict=True):
                if layer.reducer_kind == 'linear_weight':
                    # Retain the exact eager correction/scale result; no
                    # analytical reassociation or reimplementation of ReLU.
                    cotangent = layer.row_cotangent(_A).detach().clone(memory_format=torch.contiguous_format)
                    _require(cotangent.dtype == torch.float32 and
                             tuple(cotangent.shape) == (packet.context_count, cut.fiber_count),
                             'prepared cotangent shape/type differs from source')
                    entries.append(_Entry(start, stop, cut, layer, cotangent,
                                          layer.descriptor_hash, _custody(cotangent)))
        self._entries = self._issued_entries = tuple(entries)
        self._gate_custody = _custody(gate)
        self._axis_hash = axis.receipt_hash
        self._audit_count = 0
        self._audit_complete = False
        self._audit_windows = set()
        self._issued_forecast = dict(self._forecast)
        self._signature = tuple((e.start, e.stop, id(e.cut), id(e.layer), id(e.cotangent),
                                 e.descriptor_hash, e.cotangent_custody, self._factor_custody(e)) for e in entries)

    @staticmethod
    def _factor_custody(e):
        return tuple(_custody(getattr(e.layer, name)) for name in (
            'input', 'weight', 'post_activation_cotangent', 'preactivation', 'bias_baseline'))

    def assert_live(self):
        _require(not self._closed, 'linear acceleration closed')
        _require(self._entries is self._issued_entries and self._forecast == self._issued_forecast
                 and self._design.design_hash == self._design_hash
                 and self._axis.receipt_hash == self._axis_hash
                 and _custody(self._gate) == self._gate_custody,
                 'linear acceleration owner changed')
        current = tuple((e.start, e.stop, id(e.cut), id(e.layer), id(e.cotangent),
                         e.layer.descriptor_hash, _custody(e.cotangent), self._factor_custody(e)) for e in self._entries)
        _require(current == self._signature, 'prepared cotangent or source descriptor changed')
        if not self._cpu:
            _require(torch.are_deterministic_algorithms_enabled()
                     and not torch.is_deterministic_algorithms_warn_only_enabled(),
                     'linear CUDA acceleration requires strict deterministic execution')

    @staticmethod
    def _strides(e):
        return dict(X0=e.layer.input.stride(0), X1=e.layer.input.stride(1),
                    W0=e.layer.weight.stride(0), W1=e.layer.weight.stride(1),
                    C0=e.cotangent.stride(0), C1=e.cotangent.stride(1))

    def _rows(self, e, first, last):
        s = e.cut.routes_per_fiber
        if self._cpu:
            # Eager, two-step source coefficients are the private CPU oracle.
            rows = e.layer._legacy_fp32_linear_row_window_from_cotangent(
                e.cotangent, first * s, last * s).to(torch.float64).contiguous()
            rows.mul_(self._gate[e.cut.route_offset + first*s:e.cut.route_offset + last*s][None])
            rows.neg_()
            return rows.reshape(e.stop-e.start, last-first, s)
        out = torch.empty((e.stop-e.start, last-first, s), dtype=torch.float64, device=self._device)
        self._kernels.rows_kernel[(self._triton.cdiv(out.numel(), 256),)](
            e.layer.input, e.layer.weight, e.cotangent, self._gate, out,
            B=e.stop-e.start, S=s, FIRST=first, FCOUNT=last-first,
            ROFF=e.cut.route_offset, **self._strides(e), BLOCK=256,
            num_warps=4, enable_fp_fusion=False)
        return out

    def audit_rows(self, start, stop, first, last, lo, hi, eager_rows):
        self.assert_live()
        _require(not self._audit_complete, 'cold row audit already completed')
        entry = next((e for e in self._entries if e.start == start and e.stop == stop
            and e.cut.fiber_offset <= first < e.cut.fiber_offset+e.cut.fiber_count), None)
        if entry is None:
            return
        e = entry
        a, b = first-e.cut.fiber_offset, last-e.cut.fiber_offset
        _require(0 <= a < b <= e.cut.fiber_count
                 and lo == e.cut.route_offset+a*e.cut.routes_per_fiber
                 and hi == e.cut.route_offset+b*e.cut.routes_per_fiber,
                 'cold row audit differs from complete native fiber window')
        key = (start, first, last)
        _require(key not in self._audit_windows, 'duplicate cold coefficient audit')
        fused_rows = self._rows(e, a, b)
        _require(tuple(eager_rows.shape) == tuple(fused_rows.shape)
                 and torch.equal(eager_rows.contiguous().view(torch.int64), fused_rows.view(torch.int64)),
                 'fused linear coefficients differ bitwise from eager source rows')
        self._audit_windows.add(key)
        self._audit_count += fused_rows.numel()

    def finish_audit(self):
        self.assert_live()
        _require(not self._audit_complete and self._audit_count ==
                 self._forecast['expected_cold_audited_coefficient_count'],
                 'every linear source coefficient must pass the cold bitwise audit')
        self._audit_complete = True
        self._audit_windows = None

    @property
    def coordinates(self):
        self.assert_live()
        return dict(design_hash=self._design_hash,
            backend='private_cpu_eager_linear_reference' if self._cpu else 'experimental_triton_linear_fused_v1',
            eager_linear_coefficient_bitwise_audit_complete=self._audit_complete,
            cold_audited_coefficient_count=self._audit_count,
            source_primitive_law='RN32(RN32(corrected_scaled_cotangent_times_input)_times_weight)',
            gate_law='FP64_mul_then_negate_same_eager_gate',
            contraction_law=('private_CPU_FP64_materialized_product_tree_sum' if self._cpu else
                             'FP64_fixed_tree_zero_preserving_no_atomics_no_fusion'),
            legacy_convolution_path_unchanged=True, contiguous_input_required=True,
            private_cpu_reference_execution=self._cpu,
            native_cuda_qualification_verified=False,
            performance_qualification_verified=False,
            route_vector_d2h_bytes=0, **self._forecast)

    def _validate_hot(self, value, *, dual):
        self.assert_live()
        _require(self._audit_complete, 'cold bitwise row audit required before fused contraction')
        prefix = (self._spans[-1][1], self._axis.fiber_count) if dual else (self._axis.route_count,)
        _require(type(value) is torch.Tensor and value.ndim == len(prefix)+1
                 and tuple(value.shape[:-1]) == prefix and 0 < value.shape[-1] <= self._maximum_lanes
                 and value.is_contiguous()
                 and value.dtype == torch.float64 and value.device == self._device
                 and not value.requires_grad, 'fused contraction needs contiguous source-device FP64 input')

    def _validate_output(self, out, shape):
        _require(type(out) is torch.Tensor and tuple(out.shape) == shape
                 and out.dtype == torch.float64 and out.device == self._device
                 and out.is_contiguous() and not out.requires_grad,
                 'fused contraction output buffer differs from exact shape/device/layout')

    def forward_into(self, z, out, *, absolute=False):
        self._validate_hot(z, dual=False)
        lanes = z.shape[-1]
        self._validate_output(out, (self._spans[-1][1], self._axis.fiber_count, lanes))
        for e in self._entries:
            if self._cpu:
                step = max(1, self._route_tile_size // e.cut.routes_per_fiber)
                for first in range(0, e.cut.fiber_count, step):
                    last = min(e.cut.fiber_count, first+step)
                    rows = self._rows(e, first, last)
                    lo = e.cut.route_offset+first*e.cut.routes_per_fiber
                    hi = e.cut.route_offset+last*e.cut.routes_per_fiber
                    directions = z[lo:hi].reshape(last-first, -1, lanes)
                    if absolute:
                        rows, directions = rows.abs(), directions.abs()
                    # Multiplication then tree sum; independent of BLAS einsum.
                    value = (rows[..., None] * directions[None]).sum(dim=2)
                    out[e.start:e.stop, e.cut.fiber_offset+first:e.cut.fiber_offset+last] = value
            else:
                chunks = self._triton.cdiv(e.cut.routes_per_fiber, 256)
                partial = torch.empty((e.stop-e.start, e.cut.fiber_count, lanes, chunks),
                                      dtype=torch.float64, device=self._device)
                self._kernels.forward_kernel[(self._triton.cdiv(e.cut.fiber_count,4),
                                               self._triton.cdiv(e.stop-e.start,4), chunks*lanes)](
                    e.layer.input, e.layer.weight, e.cotangent, self._gate, z, partial,
                    B=e.stop-e.start, S=e.cut.routes_per_fiber, F=e.cut.fiber_count,
                    LANES=lanes, CHUNKS=chunks, ROFF=e.cut.route_offset,
                    **self._strides(e), BLOCK_S=256, BLOCK_B=4, BLOCK_F=4,
                    ABSOLUTE=absolute, num_warps=8, enable_fp_fusion=False)
                self._kernels.forward_reduce_kernel[(self._triton.cdiv((e.stop-e.start)*e.cut.fiber_count*lanes,32),)](
                    partial, out, B=e.stop-e.start, F=e.cut.fiber_count, LANES=lanes,
                    CHUNKS=chunks, GLOBAL_F=self._axis.fiber_count,
                    CONTEXT_START=e.start, FOFF=e.cut.fiber_offset,
                    BLOCK_CELLS=32, BLOCK_CHUNKS=self._triton.next_power_of_2(chunks),
                    num_warps=4, enable_fp_fusion=False)
                del partial

    def adjoint_into(self, y, out, *, absolute=False):
        self._validate_hot(y, dual=True)
        lanes = y.shape[-1]
        self._validate_output(out, (self._axis.route_count, lanes))
        for e in self._entries:
            if self._cpu:
                step = max(1, self._route_tile_size // e.cut.routes_per_fiber)
                for first in range(0, e.cut.fiber_count, step):
                    last = min(e.cut.fiber_count, first+step)
                    rows = self._rows(e, first, last)
                    values = y[e.start:e.stop,e.cut.fiber_offset+first:e.cut.fiber_offset+last]
                    if absolute:
                        rows, values = rows.abs(), values.abs()
                    contracted = (rows[..., None] * values[:, :, None, :]).sum(dim=0)
                    lo = e.cut.route_offset+first*e.cut.routes_per_fiber
                    hi = e.cut.route_offset+last*e.cut.routes_per_fiber
                    out[lo:hi].add_(contracted.reshape(hi-lo, lanes))
            else:
                block_b = self._triton.next_power_of_2(e.stop-e.start)
                block_s = 256 if block_b <= 4 else 128 if block_b <= 8 else 64
                block_f = 4 if block_b <= 16 else 2 if block_b <= 32 else 1
                self._kernels.adjoint_kernel[(self._triton.cdiv(e.cut.fiber_count,block_f)*self._triton.cdiv(e.cut.routes_per_fiber,block_s), lanes)](
                    e.layer.input, e.layer.weight, e.cotangent, self._gate, y, out,
                    B=e.stop-e.start, S=e.cut.routes_per_fiber, F=e.cut.fiber_count, LANES=lanes,
                    GLOBAL_F=self._axis.fiber_count, CONTEXT_START=e.start,
                    FOFF=e.cut.fiber_offset, ROFF=e.cut.route_offset,
                    **self._strides(e), BLOCK_B=block_b, BLOCK_F=block_f,
                    BLOCK_S=block_s, ABSOLUTE=absolute, num_warps=8, enable_fp_fusion=False)

    def convolution_tiles(self):
        """Exact original convolution producer, same complete-fiber windows."""
        self.assert_live()
        for (_, packet), (start, stop) in zip(self._pairs, self._spans, strict=True):
            for cut, layer in zip(self._axis.cuts, packet.layers, strict=True):
                if layer.reducer_kind == 'linear_weight':
                    continue
                _require(2*packet.context_count*layer.route_count*4 <= self._conv_budget,
                         'legacy convolution payload not admitted')
                conv_rows = layer.legacy_fp32_rows()[:, _ROLE_QUERY_INDEX[_A], :]
                step = max(1, self._route_tile_size // cut.routes_per_fiber)
                for first in range(0, cut.fiber_count, step):
                    last = min(cut.fiber_count, first+step)
                    lo, hi = first*cut.routes_per_fiber, last*cut.routes_per_fiber
                    rows = conv_rows[:,lo:hi].to(torch.float64).contiguous()
                    rows.mul_(self._gate[layer.route_offset+lo:layer.route_offset+hi][None]).neg_()
                    yield start, stop, cut.fiber_offset+first, cut.fiber_offset+last, layer.route_offset+lo, layer.route_offset+hi, rows.reshape(packet.context_count,last-first,cut.routes_per_fiber)
                    del rows
                del conv_rows

    def close(self):
        self._closed = True
        self._entries = self._issued_entries = ()
        self._pairs = self._spans = ()
        self._gate = self._axis = None
