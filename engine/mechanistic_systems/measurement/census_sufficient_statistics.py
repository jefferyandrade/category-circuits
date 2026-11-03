"""Census producer ``sufficient_statistics_v1``: the census accumulator slabs from per-route sufficient statistics.

Same algorithm as the engine-rows producer (``OptimizedScoringResponseProvider.accumulate_device_tripartite``):
the same images, the same capture (``model_adapter.iter_captured_batches``), the same target-logit VJP
(``exact_batched_vjp``), the same frozen loss multiplier ``c = sigmoid(eta) - y``, the same heads

    l = ((x * w) * scale) * dest          g = (((grad / scale) * x) * w) * a          s = c * g

and the same accumulator slabs (``DeviceTripartiteMomentAccumulator``: sum_rows, sum_outer, sum_outer_squared,
sum_view_activity, sum_view_activity_squared; unit context weights; no sketch).  What changes is where the sums
over images are formed:

* ``linear_weight`` blocks (fc6/fc7: 96% of AlexNet's routes): every head is (image x output factor) times
  (image x input factor) times a route constant, so every slab entry is one fp64 GEMM over the image axis, e.g.
  ``sum_i |s| = a |w| / scale * (|c * grad|^T |x|)``.  No per-image route row is formed.  Products are exact fp64
  (the engine rounds each per-image fp32 product once before widening), so no quantity loses precision.
* ``conv_kernel_weight`` blocks: per-image rows from the engine's own route operator (fp32, the run's TF32 policy),
  reduced over images in fp64.
* The fc flip-margin observation (receipt-only) is fed the engine's own per-image local rows, unchanged.

The accumulator is written once per stratum addition (copy on its first addition, add afterwards), exactly the
arithmetic ``DeviceTripartiteMomentAccumulator.update`` performs, so the look snapshot, certificates, close,
moment hashes and receipts downstream run unchanged.
"""

from __future__ import annotations

import math
import numpy as np
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

CENSUS_PRODUCER_ENV = "IRG_CENSUS_PRODUCER"
ENGINE_ROWS = "engine_rows_v6"
SUFFICIENT_STATISTICS = "sufficient_statistics_v1"
CENSUS_PRODUCERS = (ENGINE_ROWS, SUFFICIENT_STATISTICS)
# The campaign census is the importance-batch design, whose session forms these sufficient statistics itself.
# For the sealed per-context census (legacy and class-stratified designs) the default stays the engine-rows producer,
# so those designs remain byte-identical to SEAL4 (check 3a) and their existing tests hold (35 of them assume the
# engine-rows path: mock providers, the census reference's shared-forward count); sufficient_statistics_v1 stays
# selectable.
DEFAULT_CENSUS_PRODUCER = ENGINE_ROWS

STAT_NAMES = (
    "l1", "g1", "s1",                      # sum_rows
    "ll", "lg", "ls", "gg", "gs", "ss",    # sum_outer (symmetric)
    "l4", "l2g2", "l2s2", "g4", "g2s2", "s4",  # sum_outer_squared (symmetric)
    "la", "ga", "sa",                      # sum_view_activity
)
# sum_view_activity_squared is ll, gg, ss (|v|^2 = v^2; one global head).


def census_producer(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    raw = str(source.get(CENSUS_PRODUCER_ENV, "")).strip()
    value = raw or DEFAULT_CENSUS_PRODUCER
    if value not in CENSUS_PRODUCERS:
        raise ValueError(f"{CENSUS_PRODUCER_ENV}={raw!r} is not a registered census producer {CENSUS_PRODUCERS}")
    return value


def _positive_scale(scale: Any) -> float:
    value = float(scale.detach().cpu().item()) if torch.is_tensor(scale) else float(scale)
    return 1.0 if abs(value) < 1e-12 else value


def linear_block_statistics(*, inputs: Any, grad: Any, weight: Any, scale: Any, c: Any, affine_scale: float,
                            dest: Any) -> dict[str, Any]:
    """All slab entries of one ``linear_weight`` block over a packet, as fp64 GEMMs ([O, I] each)."""
    b = int(inputs.shape[0])
    X = inputs.reshape(b, -1).to(torch.float64)
    sp = _positive_scale(scale)
    D = grad.reshape(b, -1).to(torch.float64)
    w = weight.to(torch.float64)
    d = dest.to(device=w.device, dtype=torch.float64).reshape(-1, 1)
    cc = c.to(device=X.device, dtype=torch.float64).reshape(-1, 1)
    a = float(affine_scale)
    k = a / sp  # g = k * w * grad * x
    X2 = X * X
    X4 = X2 * X2
    Xa = X.abs()
    D2 = D * D
    cD = cc * D
    c2 = cc * cc
    gw = k * w
    gw2 = gw * gw
    gw4 = gw2 * gw2
    # The local head multiplies by the raw producer scale (operators.py:213-218), not the guarded divisor.
    raw_scale = float(scale.detach().cpu().item()) if torch.is_tensor(scale) else float(scale)
    local_w = (raw_scale * w) * d  # l = x * (scale * w * dest)
    out: dict[str, Any] = {}
    out["g1"] = (D.T @ X).mul_(gw)
    out["s1"] = (cD.T @ X).mul_(gw)
    out["ga"] = (D.abs().T @ Xa).mul_(gw.abs())
    out["sa"] = ((cc.abs() * D.abs()).T @ Xa).mul_(gw.abs())
    out["gg"] = (D2.T @ X2).mul_(gw2)
    out["ss"] = ((c2 * D2).T @ X2).mul_(gw2)
    out["gs"] = ((cc * D2).T @ X2).mul_(gw2)
    out["g4"] = ((D2 * D2).T @ X4).mul_(gw4)
    out["s4"] = (((c2 * D2) ** 2).T @ X4).mul_(gw4)
    out["g2s2"] = ((c2 * D2 * D2).T @ X4).mul_(gw4)
    # l * g = (local_w x)(gw grad x): the raw producer scale (local head) and the guarded divisor (global head) do
    # not cancel when the scale is guarded (|scale| < 1e-12), so keep both factors explicitly
    lg_factor = local_w * gw
    out["lg"] = (D.T @ X2).mul_(lg_factor)
    out["ls"] = (cD.T @ X2).mul_(lg_factor)
    l2g2_factor = lg_factor * lg_factor
    out["l2g2"] = (D2.T @ X4).mul_(l2g2_factor)
    out["l2s2"] = ((c2 * D2).T @ X4).mul_(l2g2_factor)
    out["l1"] = local_w * X.sum(dim=0).unsqueeze(0)
    out["la"] = local_w.abs() * Xa.sum(dim=0).unsqueeze(0)
    lw2 = local_w * local_w
    out["ll"] = lw2 * X2.sum(dim=0).unsqueeze(0)
    out["l4"] = (lw2 * lw2) * X4.sum(dim=0).unsqueeze(0)
    return {name: value.reshape(-1) for name, value in out.items()}


def rows_block_statistics(*, local_rows: Any, global_rows: Any, c: Any, affine_scale: float,
                          dest_per_route: Any) -> dict[str, Any]:
    """All slab entries from per-image rows (conv blocks): fp64 reductions over the image axis ([R] each)."""
    b = int(global_rows.shape[0])
    g = global_rows.reshape(b, -1).to(torch.float64).mul_(float(affine_scale))
    l = local_rows.reshape(b, -1).to(torch.float64).mul_(dest_per_route.to(device=g.device, dtype=torch.float64)[None, :])
    cc = c.to(device=g.device, dtype=torch.float64).reshape(-1)
    c2 = cc * cc
    out: dict[str, Any] = {}
    out["l1"] = l.sum(0)
    out["g1"] = g.sum(0)
    out["s1"] = cc @ g
    out["la"] = l.abs().sum(0)
    out["ga"] = g.abs().sum(0)
    out["sa"] = cc.abs() @ g.abs()
    l2 = l * l
    g2 = g * g
    out["ll"] = l2.sum(0)
    out["gg"] = g2.sum(0)
    out["ss"] = c2 @ g2
    lg = l * g
    out["lg"] = lg.sum(0)
    out["ls"] = cc @ lg
    out["gs"] = cc @ g2
    del lg
    out["l4"] = (l2 * l2).sum(0)
    g4 = g2 * g2
    out["g4"] = g4.sum(0)
    out["s4"] = (c2 * c2) @ g4
    out["g2s2"] = c2 @ g4
    l2g2 = l2 * g2
    out["l2g2"] = l2g2.sum(0)
    out["l2s2"] = c2 @ l2g2
    return out


def write_accumulator(accumulator: Any, statistics: Sequence[tuple[int, Mapping[str, Any]]], *,
                      context_ids: Sequence[str], route_chunk: int = 1 << 21) -> None:
    """Add one stratum addition's sums into a DeviceTripartiteMomentAccumulator (update()'s arithmetic)."""
    ids = tuple(map(str, context_ids))
    route_count = len(accumulator.route_ids)
    if (not ids or len(set(ids)) != len(ids) or set(ids).intersection(accumulator.context_ids)
            or len(accumulator.head_ids) != 3 or int(accumulator.sketch_rank) != 0):
        raise ValueError("sufficient-statistic census rows/context axis is invalid")
    # every statistic of a block has the block's width; the blocks tile the route axis exactly once, in order
    # (a width/offset mismatch would otherwise overwrite one block and leave routes unwritten)
    cursor = 0
    for offset, stats in sorted(statistics, key=lambda item: int(item[0])):
        widths = {int(stats[name].numel()) for name in STAT_NAMES}
        if len(widths) != 1 or int(offset) != cursor:
            raise ValueError("sufficient-statistic census blocks do not tile the route axis exactly once")
        cursor += widths.pop()
        for name in STAT_NAMES:
            if not bool(torch.isfinite(stats[name]).all().item()):
                raise ValueError("device tripartite rows/context axis is invalid")
    if cursor != route_count:
        raise ValueError("sufficient-statistic census does not cover the whole route axis")
    first_packet = accumulator.sample_count == 0
    accumulator._ensure_storage()
    n = float(len(ids))
    for offset, stats in statistics:
        width = int(stats["g1"].numel())
        for start in range(0, width, int(route_chunk)):
            stop = min(width, start + int(route_chunk))
            a0, a1 = offset + start, offset + stop
            targets = accumulator.windowed_slabs(a0, a1)

            def part(name: str) -> Any:
                return stats[name][start:stop].to(device=targets["sum_rows"].device)

            rows = torch.stack((part("l1"), part("g1"), part("s1")), dim=0)
            outer = torch.stack((
                torch.stack((part("ll"), part("lg"), part("ls")), dim=1),
                torch.stack((part("lg"), part("gg"), part("gs")), dim=1),
                torch.stack((part("ls"), part("gs"), part("ss")), dim=1),
            ), dim=1)
            quad = torch.stack((
                torch.stack((part("l4"), part("l2g2"), part("l2s2")), dim=1),
                torch.stack((part("l2g2"), part("g4"), part("g2s2")), dim=1),
                torch.stack((part("l2s2"), part("g2s2"), part("s4")), dim=1),
            ), dim=1)
            activity = torch.stack((part("la"), part("ga"), part("sa")), dim=0)
            activity_sq = torch.stack((part("ll"), part("gg"), part("ss")), dim=0)
            write = "copy_" if first_packet else "add_"
            getattr(targets["sum_rows"], write)(rows)
            getattr(targets["sum_outer"], write)(outer)
            getattr(targets["sum_outer_squared"], write)(quad)
            getattr(targets["sum_view_activity"], write)(activity)
            getattr(targets["sum_view_activity_squared"], write)(activity_sq)
            if accumulator.host_primary:
                accumulator.commit_window(a0, a1)
    if first_packet:
        accumulator.sketch_sum.zero_()
        accumulator.sketch_constant.zero_()
    accumulator.total_weight += n
    accumulator.sum_weight_squared += n
    accumulator.sample_count += len(ids)
    accumulator._context_ids.extend(ids)


def accumulate_census_sufficient_statistics(provider: Any, request: Any, *, accumulator: Any,
                                            flip_margin_accumulator: Any | None = None) -> Mapping[str, object]:
    """Drop-in for ``provider.accumulate_device_tripartite`` on the dense census (P's v0 census)."""
    from mechanistic_systems.core.numerics import exact_batched_vjp
    from mechanistic_systems.core.operators import route_gate_operator
    from mechanistic_systems.measurement.queries import evaluate_queries
    from mechanistic_systems.measurement.pruning_loss import frozen_affine_parameters, frozen_pruning_loss_multiplier
    from mechanistic_systems.measurement.optimized_irg_providers import _optimized_response_context_block_id
    from mechanistic_systems.measurement.phase_device_orchestration import install_scoring_gates

    from mechanistic_systems.measurement.optimized_irg_providers import ordered_string_axis_hash

    if request.kind != "response_moments":
        raise ValueError("device tripartite accumulation requires response_moments")
    mask_hash = str(request.payload.get("mask_hash", ""))
    selected = tuple(provider._delegate._selected_contexts(request))
    route_indices, route_ids = provider._delegate._routes(request)
    # preflight: the canonical whole axis, matching route identities, a supported
    # accumulator, and a valid context addition, all before any capture
    canonical = (isinstance(route_indices, range) and route_indices.start == 0 and route_indices.step == 1
                 and route_indices.stop == len(provider.route_ids)) or (
        len(route_indices) == len(provider.route_ids)
        and bool(np.array_equal(np.asarray(route_indices, dtype=np.int64), np.arange(len(provider.route_ids)))))
    if not canonical:
        raise ValueError("device partition acquisition requires the whole physical route axis")
    accumulator_route_ids = getattr(accumulator, "route_ids", ())
    if accumulator_route_ids is not route_ids and (
            len(accumulator_route_ids) != len(route_ids)
            or ordered_string_axis_hash(accumulator_route_ids, name="device accumulator route axis")
            != ordered_string_axis_hash(route_ids, name="optimized IRG scoring route axis")):
        raise ValueError("device accumulator and optimized scorer route axes differ")
    if len(accumulator.head_ids) != 3 or int(accumulator.sketch_rank) != 0:
        raise ValueError("sufficient-statistic census supports three heads and no covariance sketch")
    if len(set(selected)) != len(selected) or set(selected).intersection(accumulator.context_ids):
        raise ValueError("sufficient-statistic census context addition is invalid")
    if not selected:  # an empty addition is a no-op
        return {"schema": "census_sufficient_statistics_panel_receipt_v1", "census_producer": SUFFICIENT_STATISTICS,
                "request_id": request.request_id, "context_count": 0, "context_packet_count": 0,
                "route_count": len(route_ids), "device_reduction": True, "host_context_route_rows_transferred": False}
    if len(provider.queries) != 1 or provider.local_destination_scales_by_layer is None \
            or not provider.apply_affine_calibration_to_response or provider.conserving_activation_bindings:
        raise ValueError("sufficient-statistic census requires the registered dense-loss census wiring")
    started = time.perf_counter()
    query = provider.queries[0]
    target = query.evaluator.get("class_id", query.evaluator.get("unit", query.evaluator.get("coordinate")))
    if target is None:
        raise ValueError("BCE loss pullback requires a scalar target-logit query")
    target = int(target)
    blocks = tuple(provider._route_blocks)
    offsets: dict[str, int] = {}
    cursor = 0
    for block in blocks:
        if str(block.layer_id) in offsets:
            raise ValueError("route blocks repeat a layer identity")
        offsets[str(block.layer_id)] = cursor
        cursor += int(block.route_count)
    if cursor != len(route_ids):
        raise ValueError("route blocks differ from the census route axis")
    expected_layers = set(offsets)
    device = torch.device(provider.evaluator.device)
    sums: dict[str, dict[str, Any]] = {}
    packets = 0
    fc_input_minimum: dict[str, float] = {}
    with provider._lock:
        mask = provider.mask_resolver(mask_hash)
        calibration = provider.calibration_resolver(mask)
        if not bool(mask.to_bool_array().all()):
            raise RuntimeError("census loss must be measured on the dense network")
        affine_scale, _, _ = frozen_affine_parameters(calibration)
        provider._active_mask = mask
        provider._active_calibration = calibration
        provider._active_request_id = request.request_id
        provider._census_guarded_request_id = ""
        try:
            provider._require_dense_census_calibration()
            size = int(provider.context_packet_size)
            for start in range(0, len(selected), size):
                ids = selected[start:start + size]
                packets += 1
                block_id = _optimized_response_context_block_id(
                    provider_plan_identity_hash=request.request_id, context_ids=ids)
                provider._batch_provider.activate(context_block_id=block_id, context_ids=ids)
                install_scoring_gates(provider, mask)
                plan = provider._measurement_plan(context_block_id=block_id)
                seen: list[str] = []
                for batch in provider.model_adapter.iter_captured_batches(
                        plan, context_block_id=block_id, queries=provider.queries):
                    seen.extend(map(str, batch.context_ids))
                    query_batch = evaluate_queries(batch.response_matrix, provider.queries, labels=batch.labels)
                    outputs = [blk.output for blk in batch.route_blocks]
                    if batch.census_route_vjp is not None:  # campaign census bank: the target VJP is cached
                        grads, _ = batch.census_route_vjp.for_query(batch.response_matrix, outputs, query_batch)
                    else:
                        grads, _ = exact_batched_vjp(batch.response_matrix, outputs, query_batch,
                                                     backend="is_grads_batched")
                    del outputs
                    c = frozen_pruning_loss_multiplier(
                        raw_responses=batch.response_matrix.detach(), labels=batch.labels, target_class_id=target,
                        fit=calibration, loss_id=provider.pruning_guiding_loss_id, device=device,
                        fixed_offsets=provider._decision_census_fixed_offsets(batch.context_ids))
                    if len(batch.route_blocks) != len(grads) or {
                            str(b.route_block.layer_id) for b in batch.route_blocks} != expected_layers:
                        raise RuntimeError("captured census batch does not cover every route block exactly once")
                    for blk, grad in zip(batch.route_blocks, grads):
                        layer_id = str(blk.route_block.layer_id)
                        dest = torch.as_tensor(provider.local_destination_scales_by_layer[layer_id],
                                               dtype=torch.float64, device=device).reshape(-1)
                        if blk.reducer_kind == "linear_weight":
                            stats = linear_block_statistics(
                                inputs=blk.input_source.detach(), grad=grad[0].detach(), weight=blk.weight.detach(),
                                scale=blk.scale, c=c, affine_scale=affine_scale, dest=dest)
                            fc_input_minimum[layer_id] = min(fc_input_minimum.get(layer_id, float("inf")),
                                                             float(blk.input_source.detach().min().item()))
                            if flip_margin_accumulator is not None:
                                operator = route_gate_operator(blk.reducer_kind)
                                local = operator.local_target_contribution_per_context(blk).reshape(
                                    int(grad.shape[1]), -1)
                                module = provider.evaluator.source_model.get_submodule(
                                    str(provider._mask_bindings[[str(b.layer_id) for b in blocks].index(layer_id)].module_path))
                                bias = module.bias
                                if bias is None:
                                    bias = torch.zeros(int(blk.weight.shape[0]), device=device)
                                flip_margin_accumulator.update_linear(
                                    layer_id=layer_id, local_context_rows=local.detach(), bias=bias.detach(),
                                    route_shape=tuple(blk.route_block.shape),
                                    flattening=str(blk.route_block.flattening),
                                    route_chunk_size=provider.response_route_chunk_size)
                                del local
                        else:
                            operator = route_gate_operator(blk.reducer_kind)
                            rows = operator.reduce_vjp_per_image(blk, grad.detach())
                            local = operator.local_target_contribution_per_context(blk)
                            reps = int(blk.route_block.route_count) // int(blk.route_block.shape[0])
                            stats = rows_block_statistics(
                                local_rows=local.detach(), global_rows=rows.detach()[:, 0], c=c,
                                affine_scale=affine_scale, dest_per_route=dest.repeat_interleave(reps))
                            if flip_margin_accumulator is not None:
                                flip_margin_accumulator.mark_not_captured(layer_id=layer_id,
                                                                          route_kind=str(blk.reducer_kind))
                            del rows, local
                        bucket = sums.setdefault(layer_id, {})
                        for name, value in stats.items():
                            if name in bucket:
                                bucket[name].add_(value)
                            else:
                                bucket[name] = value
                        del stats
                    del grads, batch, query_batch
                if tuple(seen) != tuple(ids):
                    raise RuntimeError("captured census batches changed the context order")
            write_accumulator(accumulator, [(offsets[str(b.layer_id)], sums[str(b.layer_id)]) for b in blocks],
                              context_ids=selected, route_chunk=int(provider.response_route_chunk_size))
        finally:
            provider._active_mask = None
            provider._active_calibration = None
            provider._active_request_id = ""
    del sums
    receipt = {
        "schema": "census_sufficient_statistics_panel_receipt_v1",
        "census_producer": SUFFICIENT_STATISTICS,
        "request_id": request.request_id,
        "context_count": len(selected),
        "context_packet_count": int(packets),
        "context_packet_size": int(provider.context_packet_size),
        "elapsed_seconds": float(time.perf_counter() - started),
        "route_count": len(route_ids),
        "head_ids": tuple(getattr(accumulator, "head_ids", ())),
        "device_reduction": True,
        "host_context_route_rows_transferred": False,
        "host_context_route_row_transfer_count": 0,
        "route_level_summaries_only": True,
        "linear_blocks": "fp64_gemm_sufficient_statistics",
        "conv_blocks": "engine_operator_per_image_rows_fp64_reduction",
        "fc_input_minimum": dict(fc_input_minimum),
        "flip_margin_geometry_observed": bool(flip_margin_accumulator is not None),
        "pruning_guiding_loss_id": provider.pruning_guiding_loss_id,
        "dense_calibration_state_hash": provider.census_dense_calibration_state_hash,
        "loss_pullback_field": "loss_gradient",
        "competitor_gradient": "zero",
    }
    if provider.decision_fixed_offset_resolver is not None:
        receipt["pruning_functional"] = "substituted_softmax"
    return receipt


# ================================================================================================================
# Weighted (Horvitz-Thompson) census: the census 6-tuple from per-image design weights.
#
# Design: every image i of the population carries a design weight a_i with
# sum_i a_i = 1 (on-target 0.5/N_on; off-target 0.5/(999 N_class(i))) and an inclusion probability
# pi_i = min(1, lam * a_i * u_i) (on-target pi = 1).  For any per-route head statistic f (|s|, s, s^2, ...):
#     HT field      F(f) = sum_sampled a_i f_i / pi_i
#     Poisson var   V(f) = sum_sampled (1 - pi_i) (a_i f_i / pi_i)^2
# Lambda-free sums keep both additive over nested looks (lam doubles between looks, so uncapped weights rescale):
#     capped (pi = 1)      : C1(f) = sum a_i f_i                       (no variance)
#     uncapped (pi < 1)    : U1(f) = sum f_i / u_i,  U2(f) = sum f_i^2 / u_i^2,  U3(f) = sum a_i f_i^2 / u_i
#     F(f) = C1(f) + U1(f) / lam          V(f) = U2(f) / lam^2 - U3(f) / lam
# An image that becomes capped at a later look is moved by adding it with negated uncapped weights and positive
# capped weight (one re-capture of that image).
# ================================================================================================================

WEIGHT_FAMILIES = ("cap_first", "unc_first", "unc_var2", "unc_var3")
FIRST_ORDER_FAMILIES = ("cap_first", "unc_first")
VARIANCE_FAMILIES = ("unc_var2", "unc_var3")
# statistic of the head row whose family weight multiplies it; variance families carry f^2 with f = |s| (= s^2), etc.
FULL_FIRST_STATS = ("sa", "s1", "s2", "ga", "g1", "g2", "l1")
FULL_VARIANCE_STATS = ("s2", "s4", "g2", "g4")   # row^2 for SEs of means and activities; row^4 for second moments
WALK_FIRST_STATS = ("sa", "s1", "g1", "l1")
WALK_VARIANCE_STATS = ("s2", "g2")


def _weighted_linear(stat: str, *, D0, X, cc, h, w, a, sp_raw, dest):
    """One fc statistic over a packet with per-image weight h: a single fp64 GEMM ([O, I])."""
    hc = h.reshape(-1, 1)
    if stat == "l1":
        return (sp_raw * w * dest.reshape(-1, 1)) * (h @ X).unsqueeze(0)
    if stat == "l2":
        return (sp_raw * w * dest.reshape(-1, 1)) ** 2 * (h @ (X * X)).unsqueeze(0)
    gw = a * w
    if stat == "g1":
        return ((hc * D0).T @ X).mul_(gw)
    if stat == "ga":
        return ((hc * D0.abs()).T @ X.abs()).mul_(gw.abs())
    if stat == "g2":
        return ((hc * D0 * D0).T @ (X * X)).mul_(gw * gw)
    if stat == "g4":
        return ((hc * (D0 * D0) ** 2).T @ (X * X) ** 2).mul_((gw * gw) ** 2)
    if stat == "s1":
        return ((hc * cc * D0).T @ X).mul_(gw)
    if stat == "sa":
        return ((hc * (cc * D0).abs()).T @ X.abs()).mul_(gw.abs())
    if stat == "s2":
        return ((hc * (cc * D0) ** 2).T @ (X * X)).mul_(gw * gw)
    if stat == "s4":
        return ((hc * (cc * D0) ** 4).T @ (X * X) ** 2).mul_((gw * gw) ** 2)
    raise ValueError(stat)


def _weighted_rows(stat: str, *, g, l, cc, h):
    """One statistic from per-image rows (conv) with per-image weight h ([R])."""
    if stat == "l1":
        return h @ l
    if stat == "l2":
        return h @ (l * l)
    if stat == "g1":
        return h @ g
    if stat == "ga":
        return h @ g.abs()
    if stat == "g2":
        return h @ (g * g)
    if stat == "g4":
        return h @ (g * g) ** 2
    if stat == "s1":
        return (h * cc) @ g
    if stat == "sa":
        return (h * cc.abs()) @ g.abs()
    if stat == "s2":
        return (h * cc * cc) @ (g * g)
    if stat == "s4":
        return (h * (cc * cc) ** 2) @ (g * g) ** 2
    raise ValueError(stat)


class WeightedCensus:
    """Lambda-free weighted census sums per route; fields at any lambda; additive over nested looks.

    ``add_blocks`` takes one captured packet (route blocks with input_source, output gradient, weight, scale,
    reducer kind), the loss multiplier c, the affine scale and per-family image weights; ``fields(lam)`` returns
    per-layer census fields (flat per block, fp64 host arrays)."""

    def __init__(self, *, layer_ids: Sequence[str], route_counts: Sequence[int], device: Any, mode: str = "full"):
        if mode not in ("full", "walk"):
            raise ValueError("weighted census mode must be full or walk")
        self.mode = mode
        self.layer_ids = tuple(map(str, layer_ids))
        if len(self.layer_ids) != len(route_counts) or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("census layers and route counts differ or repeat")
        self.route_counts = dict(zip(self.layer_ids, map(int, route_counts)))
        self.device = torch.device(device)
        self.first_stats = FULL_FIRST_STATS if mode == "full" else WALK_FIRST_STATS
        self.variance_stats = FULL_VARIANCE_STATS if mode == "full" else WALK_VARIANCE_STATS
        self.sums: dict[tuple[str, str], dict[str, Any]] = {}
        self.image_count = 0
        self.family_images = {family: 0 for family in WEIGHT_FAMILIES}

    def _add(self, family: str, stat: str, layer_id: str, value: Any) -> None:
        bucket = self.sums.setdefault((family, stat), {})
        flat = value.reshape(-1)
        if layer_id in bucket:
            bucket[layer_id].add_(flat)
        else:
            bucket[layer_id] = flat.clone()

    def add_blocks(self, blocks: Sequence[Any], grads: Sequence[Any], *, c: Any, affine_scale: float,
                   dest_by_layer: Mapping[str, Any], weights: Mapping[str, Any],
                   row_operator: Any | None = None) -> None:
        """Add one packet.  weights maps each family to a per-image weight vector (missing family = zeros)."""
        from mechanistic_systems.core.operators import route_gate_operator

        cc = torch.as_tensor(c, dtype=torch.float64, device=self.device).reshape(-1)
        b = int(cc.shape[0])
        fam = {}
        for family in WEIGHT_FAMILIES:
            if family in weights and weights[family] is not None:
                value = torch.as_tensor(weights[family], dtype=torch.float64, device=self.device).reshape(-1)
                if value.shape[0] != b or not bool(torch.isfinite(value).all()):
                    raise ValueError("weighted census image weights differ from the packet")
                if bool((value != 0).any()):
                    fam[family] = value
                    self.family_images[family] += int((value != 0).sum().item())
        a = float(affine_scale)
        if len(blocks) != len(grads):
            raise ValueError("captured blocks and gradients differ in number")
        for blk, grad in zip(blocks, grads):
            layer_id = str(blk.route_block.layer_id) if hasattr(blk, "route_block") and hasattr(blk.route_block, "layer_id") else str(blk.layer_id)
            dest = torch.as_tensor(dest_by_layer[layer_id], dtype=torch.float64, device=self.device).reshape(-1)
            gr = grad[0] if grad.dim() == blk.output.dim() + 1 else grad
            if blk.reducer_kind == "linear_weight":
                X = blk.input_source.detach().reshape(b, -1).to(device=self.device, dtype=torch.float64)
                raw_scale = float(blk.scale.detach().cpu().item()) if torch.is_tensor(blk.scale) else float(blk.scale)
                D0 = gr.detach().reshape(b, -1).to(device=self.device, dtype=torch.float64) / _positive_scale(blk.scale)
                w = blk.weight.detach().to(device=self.device, dtype=torch.float64)
                for family, h in fam.items():
                    stats = self.first_stats if family in FIRST_ORDER_FAMILIES else self.variance_stats
                    for stat in stats:
                        self._add(family, stat, layer_id, _weighted_linear(
                            stat, D0=D0, X=X, cc=cc.reshape(-1, 1), h=h, w=w, a=a, sp_raw=raw_scale, dest=dest))
            else:
                operator = row_operator or route_gate_operator(blk.reducer_kind)
                rows = operator.reduce_vjp_per_image(blk, gr.detach().unsqueeze(0))
                g = rows.reshape(b, -1).to(device=self.device, dtype=torch.float64).mul_(a)
                del rows
                need_local = any("l1" in (self.first_stats if f in FIRST_ORDER_FAMILIES else self.variance_stats)
                                 for f in fam)
                l = None
                if need_local:
                    reps = int(g.shape[1]) // int(dest.shape[0])
                    l = operator.local_target_contribution_per_context(blk).reshape(b, -1).to(
                        device=self.device, dtype=torch.float64).mul_(dest.repeat_interleave(reps)[None, :])
                for family, h in fam.items():
                    stats = self.first_stats if family in FIRST_ORDER_FAMILIES else self.variance_stats
                    for stat in stats:
                        self._add(family, stat, layer_id, _weighted_rows(stat, g=g, l=l, cc=cc, h=h))
                del g, l
        self.image_count += b

    def fields(self, lam: float) -> dict[str, dict[str, Any]]:
        """Census fields per layer at the given lambda (flat per block, host fp64).  Field names as the engine's."""
        lam = float(lam)
        if not (lam > 0 and math.isfinite(lam)):
            raise ValueError("lambda must be positive and finite")

        def get(family, stat, layer_id):
            return self.sums.get((family, stat), {}).get(layer_id)

        def first(stat, layer_id):
            cap = get("cap_first", stat, layer_id)
            unc = get("unc_first", stat, layer_id)
            if cap is None and unc is None:
                return None
            total = torch.zeros(self.route_counts[layer_id], dtype=torch.float64, device=self.device)
            if cap is not None:
                total = total + cap
            if unc is not None:
                total = total + unc / lam
            return total

        def variance(stat, layer_id):
            u2 = get("unc_var2", stat, layer_id)
            u3 = get("unc_var3", stat, layer_id)
            total = torch.zeros(self.route_counts[layer_id], dtype=torch.float64, device=self.device)
            if u2 is not None:
                total = total + u2 / (lam * lam)
            if u3 is not None:
                total = total - u3 / lam
            return torch.clamp(total, min=0.0)

        out: dict[str, dict[str, Any]] = {}
        nan = float("nan")
        for layer_id in self.layer_ids:
            r = self.route_counts[layer_id]
            missing = torch.full((r,), nan, dtype=torch.float64, device=self.device)
            f = {stat: first(stat, layer_id) for stat in FULL_FIRST_STATS}
            v = {stat: variance(stat, layer_id) for stat in FULL_VARIANCE_STATS}

            def have(x):
                return x if x is not None else missing

            fields = {
                "local_mean": have(f["l1"]),
                "global_mean": have(f["g1"]),
                "global_activity": have(f["ga"]),
                "global_mean_standard_error": torch.sqrt(v["g2"]),
                "global_activity_standard_error": torch.sqrt(v["g2"]) if f["ga"] is not None else missing,
                "global_second_moment": have(f["g2"]),
                "global_contextual_variance": (torch.clamp(f["g2"] - f["g1"] ** 2, min=0.0)
                                               if f["g2"] is not None and f["g1"] is not None else missing),
                "global_second_moment_standard_error": (torch.sqrt(v["g4"]) if "g4" in self.variance_stats else missing),
                "global_mean_estimator_variance": v["g2"],
                "loss_mean": have(f["s1"]),
                "loss_mean_standard_error": torch.sqrt(v["s2"]),
                "loss_activity": have(f["sa"]),
                "loss_activity_standard_error": torch.sqrt(v["s2"]),
                "loss_gradient_activity": have(f["sa"]),
                "loss_second_moment": have(f["s2"]),
                "loss_contextual_variance": (torch.clamp(f["s2"] - f["s1"] ** 2, min=0.0)
                                             if f["s2"] is not None and f["s1"] is not None else missing),
                "loss_second_moment_standard_error": (torch.sqrt(v["s4"]) if "s4" in self.variance_stats else missing),
                "loss_mean_estimator_variance": v["s2"],
            }
            out[layer_id] = {name: value.detach().cpu().numpy() for name, value in fields.items()}
        return out


def poisson_family_weights(*, a: Any, u: Any, capped: Any) -> dict[str, Any]:
    """Lambda-free per-image weights for one packet: capped images carry a_i; uncapped 1/u, 1/u^2, a/u."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    capped = np.asarray(capped, dtype=bool)
    if a.shape != u.shape or a.shape != capped.shape or np.any(a <= 0) or np.any((~capped) & ~(u > 0)):
        raise ValueError("Poisson census weights need positive a, and positive u for uncapped images")
    safe_u = np.where(capped, 1.0, u)
    return {
        "cap_first": np.where(capped, a, 0.0),
        "unc_first": np.where(capped, 0.0, 1.0 / safe_u),
        "unc_var2": np.where(capped, 0.0, 1.0 / (safe_u * safe_u)),
        "unc_var3": np.where(capped, 0.0, a / safe_u),
    }


# ================================================================================================================
# BatchCensusSession (census_batches.py): Hansen-Hurwitz batch sums with dispersion SEs,
# returning the census 6-tuple.  Factory: batch_census_session(...), called by _measure_dense_batch_census.
#
# Per route and statistic f the session keeps  exact_f = sum_{exact images} w_i f_i,  and over batches k = 1..K
#   T_kf = sum_{batch k unique images} w_i f_i,   sum_T_f = sum_k T_kf,   sum_T2_f = sum_k T_kf^2   (fp64)
# estimate_f = exact_f + sum_T_f / K;  se_f = sqrt(max(sum_T2_f - sum_T_f^2 / K, 0) / (K (K - 1))).
# Statistics: sa = |s| (the order field), s1 = s (signed damage), l1 = l (local head), g1 = g (global head); in
# "full" mode also ga = |g|, s2 = s^2, g2 = g^2 (second moments).  Every T_kf is one fp64 GEMM per fc layer
# (image weights on the image-side factor) and an fp64 weighted reduction of the engine's per-image rows elsewhere.
# ================================================================================================================

SESSION_WALK_STATS = ("sa", "s1", "l1", "g1")
# Walk split (default, no switch): every batch carries the walk's order field and the local head; the
# receipt-only heads (signed loss mean s1, global mean g1) are measured on exact images and on the first look's
# draws only, as their own Hansen-Hurwitz estimate with its own SE.
SESSION_WALK_EVERY_BATCH = ("sa", "l1")
SESSION_RECEIPT_ONLY = ("s1", "g1")
SESSION_RECEIPT_BATCH_INDICES = frozenset({0})
SESSION_FULL_STATS = ("sa", "s1", "l1", "g1", "ga", "s2", "g2")
# within-sample (per-draw) squared terms, weighted by second_i = m_i (a_i / q_i)^2: stat -> accumulator name.
# Walk mode keeps what the walk (order field SE) and the registered receipt fields (loss-mean SE, local-head SE)
# read; full mode adds the global head's.
SESSION_SQUARE_STATS = {"s2": "z2_s", "l2": "z2_l", "g2": "z2_g"}
SESSION_WALK_SQUARE_STATS = {"s2": "z2_s", "l2": "z2_l"}


class FastBatchCensusSession:
    """The weighted fast census driven by run_batch_census (add_exact / add_batch / scope_fields / result).

    Estimator: exact (take-all) images contribute
    sum_i w_i y_i; off-target draws are Hansen-Hurwitz, z_j = a_i y_i / q_i per draw.  A batch of b draws arrives as
    unique images with first_i = m_i a_i / q_i and second_i = m_i (a_i / q_i)^2, so S1 += sum_i first_i y_i
    (= sum_j z_j), S2 += sum_i second_i y_i^2 (= sum_j z_j^2), n += b; T = S1 increment / b.
        estimate = exact + S1 / n,
        within-sample SE (primary, nu = n - 1) = sqrt(max(sum z^2 / n - S_off^2, 0) / (n - 1)),
        batch-dispersion SE (diagnostic) = sqrt(max(sum T^2 - (sum T)^2 / K, 0) / (K (K - 1))).
    |s|^2 = s^2, so the order field and the signed damage share one squared sum."""

    def __init__(self, *, layer_ids: Sequence[str], route_counts: Sequence[int], layer_shapes: Mapping[str, Sequence[int]],
                 scopes: Mapping[str, Any], dest_by_layer: Mapping[str, Any], affine_scale: float, device: Any,
                 capture: Any, mode: str = "walk", scope_field: str = "sa", flip_margin_accumulator: Any = None,
                 row_reducer: Any = None, capture_rows: int | None = None):
        if mode not in ("walk", "full"):
            raise ValueError("batch census session mode must be walk or full")
        self.mode = mode
        self.capture_rows = None if capture_rows is None else int(capture_rows)
        self.stats = SESSION_WALK_STATS if mode == "walk" else SESSION_FULL_STATS
        self.square_map = dict(SESSION_WALK_SQUARE_STATS if mode == "walk" else SESSION_SQUARE_STATS)
        self.layer_ids = tuple(map(str, layer_ids))
        if len(self.layer_ids) != len(route_counts) or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("census layers and route counts differ or repeat")
        self.route_counts = dict(zip(self.layer_ids, map(int, route_counts)))
        self.layer_shapes = {str(k): tuple(map(int, v)) for k, v in layer_shapes.items()}
        self.offsets: dict[str, int] = {}
        cursor = 0
        for layer_id in self.layer_ids:
            self.offsets[layer_id] = cursor
            cursor += self.route_counts[layer_id]
        self.route_total = cursor
        self.scopes = dict(scopes)
        self.dest = {k: torch.as_tensor(v, dtype=torch.float64, device=device).reshape(-1) for k, v in dest_by_layer.items()}
        self.affine_scale = float(affine_scale)
        self.device = torch.device(device)
        self.capture = capture  # capture(context_ids) -> iterable of (blocks, grads, c, batch_context_ids)
        self.scope_field = scope_field
        self.flip_margin_accumulator = flip_margin_accumulator
        self.row_reducer = row_reducer  # fused conv reducer: (blk, grad, c, {name: weights}, {name: stat}, a, dest) -> dict
        zeros = lambda: torch.zeros(self.route_total, dtype=torch.float64, device=self.device)  # noqa: E731
        self.exact = {f: zeros() for f in self.stats}
        self.sum_bt = {f: zeros() for f in self.stats}          # sum_t b_t T_t
        self.sum_t = {"sa": zeros()}                              # dispersion diagnostic (order field)
        self.sum_t2 = {"sa": zeros()}
        self.sum_z2 = {name: zeros() for name in self.square_map.values()}
        self.split = mode == "walk"
        self.receipt_sum = {f: zeros() for f in SESSION_RECEIPT_ONLY} if self.split else {}
        self.receipt_z2_s = zeros() if self.split else None
        self.receipt_draws = 0
        self.batches = 0
        self.draws = 0
        self.batch_draws: list[int] = []
        self.exact_images = 0
        self.batch_images = 0
        self.seconds = {"capture_and_reduce": 0.0, "fold": 0.0, "scope_fields": 0.0}
        self.profile = os.environ.get("IRG_CENSUS_SESSION_PROFILE", "0").strip() == "1" and self.device.type == "cuda"
        self.stage_seconds = {"capture_forward_backward": 0.0, "fc_gemm": 0.0, "conv_reducer": 0.0,
                              "other_rows": 0.0}
        self.block_paths: dict[str, str] = {}
        self._scope_index: dict[str, Any] = {}
        for scope_id, spec in self.scopes.items():
            start, stop = int(spec.start), int(spec.stop)
            if getattr(spec, "partition", None) is None:
                self._scope_index[scope_id] = slice(start, stop)
            else:
                labels = torch.as_tensor(np.asarray(spec.labels).reshape(-1) == int(spec.partition), device=self.device)
                self._scope_index[scope_id] = (start + torch.nonzero(labels).reshape(-1)).to(torch.long)

    # -- per-packet sums ------------------------------------------------------------------------------------------
    def _packet_sums(self, blocks, grads, c, requests, out=None) -> dict[str, Any]:
        """requests: {output_name: (stat, per-image weight vector)}; ADDS each statistic's packet sum into
        out[output_name] ([R] fp64; allocated when not given) and returns out."""
        if out is None:
            out = {name: torch.zeros(self.route_total, dtype=torch.float64, device=self.device) for name in requests}
        cc = torch.as_tensor(c, dtype=torch.float64, device=self.device).reshape(-1)
        weights = {name: torch.as_tensor(h, dtype=torch.float64, device=self.device).reshape(-1)
                   for name, (_, h) in requests.items()}
        b = int(cc.shape[0])
        a = self.affine_scale
        if len(blocks) != len(grads) or sorted(str(b.route_block.layer_id) for b in blocks) != sorted(self.layer_ids):
            raise RuntimeError("captured census packet does not cover every route block exactly once")
        for blk, grad in zip(blocks, grads):
            if self.profile:
                torch.cuda.synchronize(self.device)
                t_block = time.perf_counter()
            self._timed_block(blk, grad, out, requests, weights, cc, b, a)
            if self.profile:
                torch.cuda.synchronize(self.device)
                kind = ("fc_gemm" if blk.reducer_kind == "linear_weight" else
                        "conv_reducer" if self.block_paths.get(str(blk.route_block.layer_id)) == "fused_conv_reducer"
                        else "other_rows")
                self.stage_seconds[kind] += time.perf_counter() - t_block
        return out

    def _timed_block(self, blk, grad, out, requests, weights, cc, b, a) -> None:
        from mechanistic_systems.core.operators import route_gate_operator

        if True:
            layer_id = str(blk.route_block.layer_id)
            start = self.offsets[layer_id]
            stop = start + self.route_counts[layer_id]
            gr = grad[0] if grad.dim() == blk.output.dim() + 1 else grad
            dest = self.dest[layer_id]
            if blk.reducer_kind == "linear_weight":
                self.block_paths[layer_id] = "fp64_gemm"
                X = blk.input_source.detach().reshape(b, -1).to(device=self.device, dtype=torch.float64)
                raw_scale = float(blk.scale.detach().cpu().item()) if torch.is_tensor(blk.scale) else float(blk.scale)
                D0 = gr.detach().reshape(b, -1).to(device=self.device, dtype=torch.float64) / _positive_scale(blk.scale)
                w = blk.weight.detach().to(device=self.device, dtype=torch.float64)
                for name, (stat, _) in requests.items():
                    out[name][start:stop] += _weighted_linear(stat, D0=D0, X=X, cc=cc.reshape(-1, 1), h=weights[name],
                                                              w=w, a=a, sp_raw=raw_scale, dest=dest).reshape(-1)
                del X, D0
                return
            if self.row_reducer is not None and blk.reducer_kind == "conv_kernel_weight":
                views = {name: out[name][start:stop] for name in requests}
                values = self.row_reducer(blk, gr.detach(), cc, weights, {n: st for n, (st, _) in requests.items()},
                                          a, dest, out=views)
                if values is not None:
                    for name in requests:
                        if values[name].data_ptr() != views[name].data_ptr():  # returned new tensors: add them
                            out[name][start:stop] += values[name].reshape(-1)
                    self.block_paths[layer_id] = "fused_conv_reducer"
                    return
            self.block_paths.setdefault(layer_id, "engine_operator_per_image_rows")
            operator = route_gate_operator(blk.reducer_kind)
            rows = operator.reduce_vjp_per_image(blk, gr.detach().unsqueeze(0))
            g = rows.reshape(b, -1).to(device=self.device, dtype=torch.float64).mul_(a)
            del rows
            l = None
            if any(stat in ("l1", "l2") for stat, _ in requests.values()):
                reps = int(g.shape[1]) // int(dest.shape[0])
                l = operator.local_target_contribution_per_context(blk).reshape(b, -1).to(
                    device=self.device, dtype=torch.float64).mul_(dest.repeat_interleave(reps)[None, :])
            for name, (stat, _) in requests.items():
                out[name][start:stop] += _weighted_rows(stat, g=g, l=l, cc=cc, h=weights[name])
            del g, l

    def _sums(self, context_ids, weight_vectors: Mapping[str, Any], stats_by_name: Mapping[str, str]) -> dict[str, Any]:
        ids = tuple(map(str, context_ids))
        vectors = {name: np.asarray(v, dtype=np.float64).reshape(-1) for name, v in weight_vectors.items()}
        for v in vectors.values():
            if len(ids) != v.size or not np.all(np.isfinite(v)):
                raise ValueError("batch census images and weights differ")
        if len(set(ids)) != len(ids):
            raise ValueError("batch census images repeat (duplicate draws arrive once with summed weights)")
        position = {cid: i for i, cid in enumerate(ids)}
        total = {name: torch.zeros(self.route_total, dtype=torch.float64, device=self.device) for name in stats_by_name}
        seen = []
        started = time.perf_counter()
        iterator = iter(self.capture(ids))
        while True:
            t0 = time.perf_counter()
            try:
                blocks, grads, c, batch_ids = next(iterator)
            except StopIteration:
                break
            if self.profile:
                torch.cuda.synchronize(self.device)
                self.stage_seconds["capture_forward_backward"] += time.perf_counter() - t0
            batch_ids = tuple(map(str, batch_ids))
            seen.extend(batch_ids)
            rows = [position[cid] for cid in batch_ids]
            requests = {name: (stat, vectors[weight_key][rows])
                        for name, (stat, weight_key) in stats_by_name.items()}
            # census reductions never need gradient history: without no_grad, a captured tensor that still carries
            # the capture graph (e.g. a residual-stream gate's input) would chain every packet's graph into the
            # running sums and keep all their activations alive
            with torch.no_grad():
                self._packet_sums(blocks, grads, c, requests, out=total)
            del blocks, grads
        if sorted(seen) != sorted(ids):
            raise RuntimeError("the census capture did not return exactly the requested images")
        # Fail closed: a nonfinite captured factor or gradient makes these sums nonfinite;
        # refuse them before any running sum is mutated, as the engine accumulator does.
        for name, value in total.items():
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"census packet sums for {name!r} are nonfinite (captured factors or gradients)")
        self.seconds["capture_and_reduce"] += time.perf_counter() - started
        return total

    # -- protocol -------------------------------------------------------------------------------------------------
    def add_exact(self, context_ids: Sequence[str], *, weights: Any) -> None:
        if len(context_ids) == 0:
            return
        total = self._sums(context_ids, {"w": weights}, {stat: (stat, "w") for stat in self.stats})
        for stat in self.stats:
            self.exact[stat] += total[stat]
        self.exact_images += len(context_ids)

    def add_batch(self, context_ids: Sequence[str], *, first_weights: Any, second_weights: Any, draws: int,
                  index: int) -> None:
        """One batch of unique images: S1 += first_i y_i (first_i = m_i a_i / q_i), S2 += second_i y_i^2
        (second_i = m_i (a_i / q_i)^2), n += draws.  T = (S1 increment) / draws feeds the dispersion diagnostic."""
        first = np.asarray(first_weights, dtype=np.float64).reshape(-1)
        second = np.asarray(second_weights, dtype=np.float64).reshape(-1)
        b = int(draws)
        if first.shape != second.shape or b < 1 or np.any(first < 0) or np.any(second < 0):
            raise ValueError("batch census weights or draws are invalid")
        receipt_batch = self.split and int(index) in SESSION_RECEIPT_BATCH_INDICES
        batch_stats = (self.stats if not self.split else
                       SESSION_WALK_EVERY_BATCH + (SESSION_RECEIPT_ONLY if receipt_batch else ()))
        names = {stat: (stat, "first") for stat in batch_stats}
        names.update({z: (stat, "second") for stat, z in self.square_map.items()})
        total = self._sums(context_ids, {"first": first, "second": second}, names)
        started = time.perf_counter()
        for stat in batch_stats:
            if self.split and stat in SESSION_RECEIPT_ONLY:
                self.receipt_sum[stat] += total[stat]
            else:
                self.sum_bt[stat] += total[stat]
        if receipt_batch:
            self.receipt_z2_s += total["z2_s"]
            self.receipt_draws += b
        t = total["sa"] / b
        self.sum_t["sa"] += t
        self.sum_t2["sa"] += t * t
        for z in self.square_map.values():
            self.sum_z2[z] += total[z]
        self.batches += 1
        self.draws += b
        self.batch_draws.append(b)
        self.batch_images += len(context_ids)
        self.seconds["fold"] += time.perf_counter() - started

    def drop_sampled(self) -> None:
        """Forget every sampled draw (the loop then enumerates the whole off-target population exactly)."""
        for bucket in (self.sum_bt, self.sum_t, self.sum_t2, self.sum_z2, self.receipt_sum):
            for value in bucket.values():
                value.zero_()
        if self.receipt_z2_s is not None:
            self.receipt_z2_s.zero_()
        self.receipt_draws = 0
        self.batches = 0
        self.draws = 0
        self.batch_draws = []
        self.batch_images = 0

    def _square_name(self, stat: str) -> str:
        return {"sa": "z2_s", "s1": "z2_s", "l1": "z2_l", "g1": "z2_g", "ga": "z2_g"}.get(stat, "")

    def _estimate_and_se(self, stat: str) -> tuple[Any, Any]:
        if self.split and stat in SESSION_RECEIPT_ONLY:
            # receipt-only heads: exact images plus the first look's draws (their own HH estimate and SE)
            n = self.receipt_draws
            off = self.receipt_sum[stat] / n if n else torch.zeros_like(self.exact[stat])
            estimate = self.exact[stat] + off
            if n == 0:
                return estimate, torch.zeros_like(estimate)
            if stat == "s1" and n >= 2:
                return estimate, torch.sqrt(torch.clamp(self.receipt_z2_s / n - off * off, min=0.0) / (n - 1))
            return estimate, torch.full_like(estimate, float("nan") if stat == "g1" else float("inf"))
        n = self.draws
        off = self.sum_bt[stat] / n if n else torch.zeros_like(self.exact[stat])
        estimate = self.exact[stat] + off
        z = self._square_name(stat)
        if z and z not in self.sum_z2:
            z = ""
        if n == 0:
            se = torch.zeros_like(estimate)  # take-all only: no sampling variance
        elif not z:
            se = torch.full_like(estimate, float("nan"))  # this head's squared sum is not kept in this mode
        elif n >= 2:
            se = torch.sqrt(torch.clamp(self.sum_z2[z] / n - off * off, min=0.0) / (n - 1))
        else:
            se = torch.full_like(estimate, float("inf"))
        return estimate, se

    def dispersion_se(self) -> Any:
        k = self.batches
        if k < 2:
            return torch.full_like(self.exact["sa"], float("inf"))
        return torch.sqrt(torch.clamp(self.sum_t2["sa"] - self.sum_t["sa"] ** 2 / k, min=0.0) / (k * (k - 1)))

    def scope_fields(self) -> Mapping[str, tuple[Any, Any]]:
        """Per scope (estimate, within-sample SE) of the order field as device fp64 tensors (no host copy)."""
        started = time.perf_counter()
        estimate, se = self._estimate_and_se(self.scope_field)
        out = {scope_id: (estimate[index], se[index]) for scope_id, index in self._scope_index.items()}
        self.seconds["scope_fields"] += time.perf_counter() - started
        return out

    def scope_fields_host(self) -> Mapping[str, tuple[Any, Any]]:
        return {scope_id: (e.cpu().numpy(), s.cpu().numpy()) for scope_id, (e, s) in self.scope_fields().items()}

    def result(self) -> tuple[Any, ...]:
        """The census 6-tuple: (local means, global means, global activity, ordering statistics, flip-margin
        geometry, session receipt), per layer in group shape, fp64 host arrays.

        Only the fields the walk and the registered receipts read are copied to the host (order field and its SE,
        signed loss mean and SE, local head mean and SE, global mean, and the two estimator variances); fields this
        mode does not estimate are zero-copy read-only NaN views.  The session's device buffers are released."""
        started = time.perf_counter()
        nan_scalar = np.float64("nan")
        fields = {stat: self._estimate_and_se(stat) for stat in self.stats}
        dispersion = self.dispersion_se()
        sa, sa_se = fields["sa"]
        within = torch.where(sa_se > 0, sa_se, torch.ones_like(sa_se))
        dispersion_ratio = {}
        for lid in self.layer_ids:
            sl = slice(self.offsets[lid], self.offsets[lid] + self.route_counts[lid])
            mask = (sa_se[sl] > 0) & torch.isfinite(dispersion[sl])
            dispersion_ratio[lid] = (float(torch.median(dispersion[sl][mask] / within[sl][mask]).item())
                                     if bool(mask.any()) else None)
        device_fields = {
            "loss_activity": sa, "loss_activity_standard_error": sa_se,
            "loss_mean": fields["s1"][0], "loss_mean_standard_error": fields["s1"][1],
            "loss_mean_estimator_variance": fields["s1"][1] ** 2,
            "local_mean": fields["l1"][0], "local_mean_standard_error": fields["l1"][1],
            "global_mean": fields["g1"][0], "global_mean_standard_error": fields["g1"][1],
            "global_mean_estimator_variance": fields["g1"][1] ** 2,
        }
        if "ga" in fields:
            device_fields.update(global_activity=fields["ga"][0], global_activity_standard_error=fields["ga"][1])
        if "s2" in fields:
            s1, g1 = fields["s1"][0], fields["g1"][0]
            device_fields.update(
                loss_second_moment=torch.clamp(fields["s2"][0], min=0.0),
                loss_contextual_variance=torch.clamp(fields["s2"][0] - s1 * s1, min=0.0),
                global_second_moment=torch.clamp(fields["g2"][0], min=0.0),
                global_contextual_variance=torch.clamp(fields["g2"][0] - g1 * g1, min=0.0))
        if self.mode == "walk" and torch.isnan(device_fields["global_mean_standard_error"]).all():
            device_fields.pop("global_mean_standard_error")
            device_fields.pop("global_mean_estimator_variance")
        host: dict[str, np.ndarray] = {}
        for name, tensor in device_fields.items():
            host[name] = tensor.detach().to("cpu", non_blocking=False).numpy()
        del device_fields, fields, dispersion, within

        def per_layer(flat):
            return {lid: flat[self.offsets[lid]:self.offsets[lid] + self.route_counts[lid]].reshape(self.layer_shapes[lid])
                    for lid in self.layer_ids}

        def missing():
            return {lid: np.broadcast_to(nan_scalar, self.layer_shapes[lid]) for lid in self.layer_ids}

        ordering_names = (
            "loss_activity", "loss_activity_standard_error", "loss_gradient_activity", "loss_mean",
            "loss_mean_standard_error", "loss_mean_estimator_variance", "global_mean_standard_error",
            "global_mean_estimator_variance", "global_activity_standard_error", "local_mean_standard_error",
            "loss_second_moment", "loss_second_moment_standard_error", "loss_contextual_variance",
            "global_second_moment", "global_second_moment_standard_error", "global_contextual_variance")
        ordering = {}
        for name in ordering_names:
            source = "loss_activity" if name == "loss_gradient_activity" else name
            ordering[name] = per_layer(host[source]) if source in host else missing()
        local_means = per_layer(host["local_mean"])
        global_means = per_layer(host["global_mean"])
        global_activity = per_layer(host["global_activity"]) if "global_activity" in host else missing()
        if self.flip_margin_accumulator is not None:
            flip = self.flip_margin_accumulator.finalize()
        else:
            flip = {"schema": "not_captured_under_importance_batch_census", "margin_percentiles": {},
                    "context_count_by_fc_layer": {}, "conv_margins": {}, "conv_flip_counters": "not_emitted"}
        not_estimated = sorted(name for name in ordering_names if (
            "loss_activity" if name == "loss_gradient_activity" else name) not in host)
        if "global_activity" not in host:
            not_estimated.append("global_activity")
        receipt = {
            "schema": "census_fast_batch_session_receipt_v3",
            "census_producer": "fast_batch_session_v3",
            "mode": self.mode, "statistics": list(self.stats),
            "estimator": "exact take-all plus Hansen-Hurwitz draws; within-sample SE (nu = n - 1) primary",
            "batch_dispersion_se_over_within_se_median_by_layer": dispersion_ratio,
            "batch_count": self.batches, "draws": self.draws, "batch_draws_equal": len(set(self.batch_draws)) <= 1,
            "exact_images": self.exact_images, "batch_unique_images": self.batch_images,
            "capture_rows": self.capture_rows,
            "gpu": (torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else str(self.device)),
            "tf32": {"cuda_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                     "cudnn": bool(torch.backends.cudnn.allow_tf32)},
            "linear_blocks": "fp64_gemm_sufficient_statistics",
            "block_paths": dict(self.block_paths),
            "stage_seconds_profiled": dict(self.stage_seconds) if self.profile else None,
            "fields_not_estimated_in_mode": not_estimated,
            "fields_on_host": sorted(host),
            "estimator_variance_fields": "squared within-sample SE",
            "receipt_only_heads": ({"heads": list(SESSION_RECEIPT_ONLY), "batches": sorted(SESSION_RECEIPT_BATCH_INDICES),
                                    "draws": self.receipt_draws,
                                    "source": "exact images (on-target and certainty set) plus the draws of batch "
                                              "index 0 (the first look); Hansen-Hurwitz estimate with its own "
                                              "within-sample SE; dropped with drop_sampled()"} if self.split else None),
            "seconds": dict(self.seconds),
        }
        if getattr(self, "pruning_functional", None) == "substituted_softmax":
            receipt["pruning_functional"] = "substituted_softmax"
        # release the device buffers (the walk reads only the host fields)
        self.exact = self.sum_bt = self.sum_t = self.sum_t2 = self.sum_z2 = self.receipt_sum = None
        self.receipt_z2_s = None
        receipt["result_seconds"] = time.perf_counter() - started
        return local_means, global_means, global_activity, ordering, flip, receipt


def provider_capture(provider: Any, *, mask: Any, calibration: Any, target: int, rows: int, rows_holder: dict):
    """capture(context_ids) over the engine's census forward: yields (blocks, grads, c, batch_context_ids)."""
    from mechanistic_systems.core.numerics import exact_batched_vjp
    from mechanistic_systems.measurement.queries import evaluate_queries
    from mechanistic_systems.measurement.pruning_loss import frozen_pruning_loss_multiplier
    from mechanistic_systems.measurement.optimized_irg_providers import _optimized_response_context_block_id
    from mechanistic_systems.measurement.phase_device_orchestration import install_scoring_gates

    device = torch.device(provider.evaluator.device)
    counter = {"n": 0}

    def capture(context_ids):
        rows_holder["rows"] = int(rows)
        counter["n"] += 1
        request = f"census-fast-batch-session-{counter['n']}-{len(context_ids)}-{context_ids[0]}"
        with provider._lock:
            provider._active_mask = mask
            provider._active_calibration = calibration
            provider._active_request_id = request
            try:
                block_id = _optimized_response_context_block_id(provider_plan_identity_hash=request,
                                                                context_ids=tuple(context_ids))
                provider._batch_provider.activate(context_block_id=block_id, context_ids=tuple(context_ids))
                install_scoring_gates(provider, mask)
                plan = provider._measurement_plan(context_block_id=block_id)
                for batch in provider.model_adapter.iter_captured_batches(plan, context_block_id=block_id,
                                                                         queries=provider.queries):
                    query_batch = evaluate_queries(batch.response_matrix, provider.queries, labels=batch.labels)
                    outputs = [blk.output for blk in batch.route_blocks]
                    grads, _ = exact_batched_vjp(batch.response_matrix, outputs, query_batch,
                                                 backend="is_grads_batched")
                    c = frozen_pruning_loss_multiplier(
                        raw_responses=batch.response_matrix.detach(), labels=batch.labels, target_class_id=target,
                        fit=calibration, loss_id=provider.pruning_guiding_loss_id, device=device,
                        fixed_offsets=provider._decision_census_fixed_offsets(batch.context_ids))
                    yield batch.route_blocks, grads, c, batch.context_ids
                    del batch, grads, outputs, query_batch
            finally:
                provider._active_mask = None
                provider._active_calibration = None
                provider._active_request_id = ""

    return capture


def batch_census_session(*, runtime: Any, optimization_axis: Any, query: Any, dense_mask: Any, route_ids: Any,
                         route_layer_ids: Any, target_class_id: int, dense_calibration_state: Any,
                         pruning_guiding_loss_id: str, scopes: Mapping[str, Any], mode: str | None = None,
                         capture_rows: int | None = None,
                         decision_fixed_offset_resolver: Any = None) -> FastBatchCensusSession:
    """Factory read by registered_resnet18_pilot._census_batch_session_factory."""
    from mechanistic_systems.contracts.content_address import content_hash
    from mechanistic_systems.measurement.optimized_irg_providers import (
        OptimizedScoringResponseProvider, resnet_local_destination_transport)
    from mechanistic_systems.measurement.pruning_loss import frozen_affine_parameters
    from mechanistic_systems.measurement.flip_margin_geometry import DenseFlipMarginAccumulator  # noqa: F401

    mode = mode or os.environ.get("IRG_CENSUS_SESSION_MODE", "walk")
    rows = int(capture_rows or os.environ.get("IRG_CENSUS_CAPTURE_ROWS", "256"))
    scales, transport = resnet_local_destination_transport(runtime.evaluator)
    sized = runtime.optimization_data_plane.sized_role_batch_factory("scoring")
    rows_holder = {"rows": rows}
    provider = OptimizedScoringResponseProvider(
        evaluator=runtime.evaluator, measurement_view=optimization_axis.data_view(),
        context_ids=optimization_axis.context_ids, batch_factory=lambda ids: sized(ids, rows_holder["rows"]),
        mask_resolver=lambda _: dense_mask, calibration_resolver=lambda _: dense_calibration_state,
        route_ids=route_ids, queries=(query,),
        response_spec_hash=content_hash({"query_id": query.query_id, "semantics": "census_fast_batch_session_v1"}),
        response_metric_id="census-fast-batch-session", study_hash=optimization_axis.content_hash,
        apply_affine_calibration_to_response=True, pruning_guiding_loss_id=pruning_guiding_loss_id,
        decision_fixed_offset_resolver=decision_fixed_offset_resolver,
        local_destination_scales_by_layer=scales, local_destination_transport_hash=transport,
        local_regime_specs_by_layer={}, conserving_activation_bindings=())
    blocks = tuple(provider._route_blocks)
    if (sum(int(b.route_count) for b in blocks) != len(route_ids)
            or len({str(b.layer_id) for b in blocks}) != len(blocks)
            or {str(b.layer_id) for b in blocks} != {str(g.layer_id) for g in runtime.route_groups}):
        raise ValueError("census route blocks do not tile the route axis exactly once")
    affine_scale, _, _ = frozen_affine_parameters(dense_calibration_state)
    shapes = {str(group.layer_id): tuple(int(v) for v in group.shape) for group in runtime.route_groups}
    row_reducer = None
    if (torch.device(runtime.evaluator.device).type == "cuda"
            and os.environ.get("IRG_CENSUS_FUSED_CONV", "1").strip() != "0"):
        try:
            from mechanistic_systems.measurement.fused_conv_census import row_reducer as _fused_row_reducer
            row_reducer = _fused_row_reducer
        except ImportError:  # the reducer module is not in this tree: the engine-operator path runs
            row_reducer = None
    session = FastBatchCensusSession(
        layer_ids=[str(b.layer_id) for b in blocks], route_counts=[int(b.route_count) for b in blocks],
        layer_shapes=shapes, scopes=scopes, dest_by_layer=scales, affine_scale=affine_scale,
        device=runtime.evaluator.device,
        capture=provider_capture(provider, mask=dense_mask, calibration=dense_calibration_state,
                                 target=int(target_class_id), rows=rows, rows_holder=rows_holder),
        mode=mode, row_reducer=row_reducer, capture_rows=rows)
    if decision_fixed_offset_resolver is not None:
        session.pruning_functional = "substituted_softmax"
    return session
