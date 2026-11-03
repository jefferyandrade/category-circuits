"""Batched phase reductions with input fault guards and run receipts.

Numeric work is tensor-only until a compact batch result is read. Python
assembles the existing products after that read. No decision rule is here.
"""

from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

import numpy as np
import torch
from .run_products import effective_settings

_active_state = ContextVar("irg_phase_device_state", default=None)


def active_state():
    return _active_state.get()


def call_with_state(state, function, **kwargs):
    token = _active_state.set(state)
    try:
        return function(**kwargs)
    finally:
        _active_state.reset(token)





@dataclass
class PhaseDeviceState:
    seed: int = 0
    calls: Counter = field(default_factory=Counter)
    reads: Counter = field(default_factory=Counter)
    _checked: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)

    def receipt(self):
        return {"schema": "irg_phase_device_orchestration_v2", "enabled": True,
                "effective_settings": effective_settings(resolved=self.settings),
                "calls": dict(self.calls), "batch_host_reads": dict(self.reads)}



def state_for(owner):
    state = getattr(owner, "_phase_device_state", None)
    if state is None:
        state = PhaseDeviceState(int(getattr(owner, "target_class_id", 0)))
        resolve = getattr(owner, "_effective_execution_settings", None)
        if not callable(resolve):
            resolve = getattr(owner, "_effective_delivery_settings", None)
        if callable(resolve):
            state.settings = resolve()
        owner._phase_device_state = state
    return state


def _join_anchor_packets(packets):
    return (tuple(context for ids, _, _ in packets for context in ids),
            np.concatenate([values for _, values, _ in packets], axis=0),
            np.concatenate([labels for _, _, labels in packets], axis=0))


def preload_anchor_packets(provider, mask_hash, packets):
    """Join input packets once, preserving their first-occurrence row order."""
    state, phase = state_for(provider), "dense_anchor_bulk_preload"
    packets = tuple(packets)
    if (not packets or any(values.ndim != 2 or labels.ndim != 1
            or values.shape[0] != len(ids) or labels.shape[0] != len(ids)
            for ids, values, labels in packets)
            or len({values.shape[1] for _, values, _ in packets}) != 1):
        raise ValueError("dense anchor packet axes differ")
    joined_ids, joined_values, joined_labels = _join_anchor_packets(packets)
    provider.preload_projected_readouts(mask_hash, joined_ids, joined_values, joined_labels)
    state.calls[phase] += 1
    return True



def _segment_sums(rows, lengths):
    return torch.segment_reduce(rows, "sum", lengths=lengths, axis=0)


def stratum_moments(rows, counts, state, *, phase="terminal_category_moments", second=True):
    """Reduce every stratum and head in float64 with one result transfer."""
    counts = tuple(int(n) for n in counts)
    if (not torch.is_tensor(rows) or rows.ndim != 2 or not counts
            or rows.shape[1] < 1 or min(counts) <= 0 or sum(counts) != rows.shape[0]):
        raise ValueError("stratum moment row axis differs")
    values = rows.to(torch.float64)
    lengths = torch.tensor(counts, device=values.device, dtype=torch.int64)
    total = _segment_sums(values, lengths)
    n = lengths.to(torch.float64).unsqueeze(1)
    mean = total / n
    fields = [mean.reshape(-1)]
    finite = torch.isfinite(values).all() & torch.isfinite(mean).all()
    if second:
        m2 = _segment_sums(values.square(), lengths) - total.square() / n
        fields.append(m2.reshape(-1))
        finite &= torch.isfinite(m2).all()
    fields.append(finite.to(torch.float64).reshape(1))
    packed = torch.cat(fields).detach().cpu().numpy()
    state.reads[phase] += 1
    if not packed[-1]:
        raise ValueError("stratum moments contain nonfinite values")
    size = len(counts) * rows.shape[1]
    result_mean = packed[:size].reshape(len(counts), rows.shape[1])
    result_m2 = packed[size:2 * size].reshape(result_mean.shape) if second else None
    state.calls[phase] += 1
    return result_mean, result_m2



def terminal_means(loss, accuracy, counts, state):
    return stratum_moments(torch.stack((loss, accuracy.to(loss.dtype)), dim=1), counts,
                           state, phase="terminal_validation", second=False)


def census_sign_batch(mean, error, activity, *, critical, decision_referenced, state, tolerance=None):
    """Reduce sign criteria and Gaussian telemetry in float64 in one read."""
    phase = "census_sign"
    mean, error, activity = (value.to(torch.float64) for value in (mean, error, activity))
    if mean.ndim != 2 or error.shape != mean.shape or activity.shape != mean.shape:
        raise ValueError("census sign axes differ")
    magnitude = mean.abs()
    mass = activity.sum(dim=1)
    consequence = (activity * magnitude).sum(dim=1)
    tiny = torch.finfo(torch.float64).tiny
    if decision_referenced:
        numerator = (2 * activity * torch.clamp(critical * error - magnitude, min=0)).sum(dim=1)
        denominator = consequence
    else:
        ambiguous = (magnitude <= critical * error) & (error > 0)
        numerator = torch.where(ambiguous, activity, 0).sum(dim=1)
        denominator = mass
    risks = torch.where(denominator <= tiny, 0, numerator / denominator.clamp_min(tiny))
    live = error > 0
    live_error = torch.where(live, error, 1.0)
    z = magnitude / live_error
    bracket = 1 / math.sqrt(2 * math.pi) - 0.5 * z * torch.special.erfcx(z / math.sqrt(2))
    wrong = live_error * torch.exp(-0.5 * z.square()) * bracket
    wrong = torch.where(torch.isinf(z) | ~live, 0, wrong).clamp_min(0)
    expected_numerator = (2 * activity * wrong).sum(dim=1)
    expected = torch.where(consequence <= tiny, 0, expected_numerator / consequence.clamp_min(tiny))
    finite = (torch.isfinite(mean).all() & torch.isfinite(error).all() & torch.isfinite(activity).all()
              & torch.isfinite(mass).all() & torch.isfinite(consequence).all()
              & torch.isfinite(numerator).all() & torch.isfinite(expected_numerator).all())
    packed = torch.cat((risks, expected, finite.to(torch.float64).reshape(1))).detach().cpu().numpy()
    state.reads[phase] += 1
    if not packed[-1] or not np.isfinite(packed[:-1]).all():
        raise ValueError("census sign reduction contains nonfinite values")
    state.calls[phase] += 1
    return packed[:mean.shape[0]], packed[mean.shape[0]:-1]



def finite_readout_batch(responses, width, rows, state):
    """Check device finiteness once per tensor generation and read the flags together."""
    phase = "readout_checks"
    values = tuple(r.device_values for r in responses)
    if (not values or rows < 0 or width < 1
            or any(v is None or tuple(v.shape) != (rows, width) for v in values)):
        raise ValueError("readout axes differ")
    if len({str(v.device) for v in values}) != 1:
        raise ValueError("readouts use different devices")
    signature = tuple((id(v), v.data_ptr(), v._version, tuple(v.shape), str(v.dtype), str(v.device)) for v in values)
    cached = state._checked.get(signature)
    if cached is not None and all(a is b for a, b in zip(cached, values, strict=True)):
        state.calls["readout_checks_reused"] += 1
        return True
    finite = torch.stack([torch.isfinite(v).all() for v in values]).all()
    if values[0].device.type == "cuda":
        torch._assert_async(finite, "readout contains nonfinite values")
    else:
        state.reads[phase] += 1
        if not bool(finite):
            raise ValueError("readout contains nonfinite values")
    if len(state._checked) >= 8:
        state._checked.clear()
    state._checked[signature] = values
    state.calls[phase] += 1
    return True



def install_scoring_gates(provider, mask):
    """Install a mask once while its route-gate tensor generations are stable."""
    state, phase = state_for(provider), "scoring_gates"
    graph = provider.model_adapter.gate_graph
    modules = tuple(graph.gates_by_layer[str(block.layer_id)] for block in provider._route_blocks)
    signature = tuple((id(m.route_gate), m.route_gate._version, tuple(m.route_gate.shape)) for m in modules)
    previous = getattr(provider, "_phase_installed_scoring_mask", None)
    if previous is not None and previous[0] is mask and previous[1] is provider._route_blocks and previous[2] == signature:
        state.calls["scoring_gates_reused"] += 1
        return True
    for layer, gate in provider._masks_by_layer(mask).items():
        graph.set_gate(layer, gate)
    signature = tuple((id(m.route_gate), m.route_gate._version, tuple(m.route_gate.shape)) for m in modules)
    provider._phase_installed_scoring_mask = (mask, provider._route_blocks, signature)
    state.calls["scoring_gates_installed"] += 1
    return True



def _model_signature(tensors):
    return tuple((name, value.data_ptr(), value._version, tuple(value.shape), str(value.dtype), str(value.device))
                 for name, value in sorted(tensors.items()))


def cached_model_state_hash(owner, tensors):
    """Reuse the model-state fingerprint while tensor generations are stable."""
    state = state_for(owner)
    if not tensors or any(not torch.is_tensor(v) for v in tensors.values()):
        raise TypeError("model state requires tensors")
    cached = getattr(owner, "_phase_model_fingerprint", None)
    if cached is None or cached[0] != _model_signature(tensors):
        return None
    state.calls["terminal_model_state_reused"] += 1
    return cached[1]



def remember_model_state_hash(owner, tensors, digest):
    owner._phase_model_fingerprint = (_model_signature(tensors), digest)
    state_for(owner).calls["terminal_model_state_hashed"] += 1



class DeferredPhaseValue:
    """Host product encoding, resolved only inside the completion boundary."""
    def __init__(self, produce):
        self.produce = produce
        self.ready = False
        self.value = None

    def resolve(self):
        if not self.ready:
            self.value = self.produce()
            self.produce = None
            self.ready = True
        return self.value


def resolve_products(value):
    if isinstance(value, DeferredPhaseValue):
        return resolve_products(value.resolve())
    if isinstance(value, dict):
        return {key: resolve_products(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(resolve_products(item) for item in value)
    if isinstance(value, list):
        return [resolve_products(item) for item in value]
    return value


def product_digest(value):
    if isinstance(value, DeferredPhaseValue):
        return DeferredPhaseValue(lambda: hashlib.sha256(value.resolve()).hexdigest())
    return hashlib.sha256(value).hexdigest()
