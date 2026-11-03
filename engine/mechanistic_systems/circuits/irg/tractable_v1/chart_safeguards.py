"""Finite chart safeguards, without a claim about unseen inputs or the full path.

The frozen-input in-layer drive is exact for that affine layer alone. It is
not the joint multilayer perturbation: upstream activations and max-pool
switches may change. Even unchanged ReLU gates leave multilayer mask products
nonlinear in dose. These observations never certify a population trust ball.
"""
from __future__ import annotations

import torch

from .operators import ChartOperators


@torch.no_grad()
def local_action_kinks(ops: ChartOperators, z: torch.Tensor, *, workspace_bytes: int = 256 << 20) -> dict:
    """Count frozen-input ReLU phase changes for the entire proposed action.

    Removing several individually safe routes can cross a wall collectively.
    A positive parent that reaches zero is counted; inactive units that remain
    exactly zero are unchanged. Nonfinite arithmetic refuses.
    """
    if type(workspace_bytes) is not int or workspace_bytes < 1024:
        raise ValueError("workspace_bytes must be an integer of at least 1024")
    if z.shape != (ops.route_count,) or not bool(torch.isfinite(z).all()):
        raise ValueError("finite route-aligned action required")
    if not bool(((z >= 0) & (z <= ops.eligible())).all()):
        raise ValueError("action must stay in the parent's irreversible cone")
    counts = []
    for layer, piece in zip(ops.factors.layers, ops.axis.split(z), strict=True):
        # Bound principal context temporaries (input layout copy, drive,
        # endpoint, comparisons); factor bank, route matrix and BLAS workspace
        # are separate. Accumulate counts on device until the layer is done.
        per_context = 2 * layer.X.element_size() * layer.X.shape[-1] * (
            layer.X.shape[1] + 4 * layer.a.shape[1])
        if per_context > workspace_bytes:
            raise ValueError("workspace cannot hold one local action context")
        chunk = max(1, workspace_bytes // per_context)
        matrix = layer.W * layer.mask * piece
        positions = torch.zeros((), device=ops.device, dtype=torch.int64)
        contexts = torch.zeros_like(positions)
        finite = torch.ones((), device=ops.device, dtype=torch.bool)
        for start in range(0, ops.context_count, chunk):
            parent = layer.a[start:start + chunk]
            drive = torch.einsum("fj,bjp->bfp", matrix, layer.X[start:start + chunk])
            endpoint = parent - drive
            finite &= torch.isfinite(endpoint).all()
            crossed = (parent > 0) != (endpoint > 0)
            positions += crossed.sum()
            contexts += crossed.flatten(1).any(1).sum()
            del drive, endpoint, crossed
        if not bool(finite):
            raise ValueError("nonfinite local action endpoint")
        counts.append(dict(layer=layer.geometry.name, crossed_positions=int(positions),
                           crossed_contexts=int(contexts)))
    return dict(schema="tractable_v1_frozen_input_action_kinks_v1", layers=counts,
                local_phase_unchanged=all(x["crossed_positions"] == 0 for x in counts),
                scope="observed_contexts_frozen_layer_inputs", full_path_certified=False,
                population_certified=False)


@torch.no_grad()
def singleton_kink_flags(ops: ChartOperators, *, workspace_bytes: int = 32 << 20) -> torch.Tensor:
    """Exact observed frozen-input singleton crossings, with bounded temporary tiles.

    This is an optional census, not a free operator. The budget bounds the
    principal gathered/product tensors, not total process memory. A sign-aware
    check avoids excluding large contributions that move away from the wall.
    """
    if type(workspace_bytes) is not int or workspace_bytes < 1024:
        raise ValueError("workspace_bytes must be an integer of at least 1024")
    flags = torch.zeros(ops.route_count, device=ops.device, dtype=torch.bool)
    for layer in ops.factors.layers:
        g = layer.geometry
        positions = layer.X.shape[-1]
        item = layer.X.element_size()
        # Gather X and a, multiplication and endpoint; one context at a time.
        chunk = max(1, workspace_bytes // (4 * item * positions))
        if 4 * item * positions > workspace_bytes:
            raise ValueError("workspace cannot hold one route-position tile")
        for start in range(0, g.route_count, chunk):
            idx = torch.arange(start, min(g.route_count, start + chunk), device=ops.device)
            f, j = idx // g.routes_per_fiber, idx % g.routes_per_fiber
            live = layer.mask[f, j] > 0
            hit = torch.zeros_like(live)
            for c in range(ops.context_count):
                a = layer.a[c, f]
                endpoint = a - (layer.W[f, j] * layer.mask[f, j])[:, None] * layer.X[c, j]
                if not bool(torch.isfinite(endpoint).all()):
                    raise ValueError("nonfinite singleton endpoint")
                hit |= ((a > 0) != (endpoint > 0)).any(dim=1) & live
            flags[g.route_offset + idx] = hit
    return flags
