"""Exact action decomposition for piecewise-linear sequential networks.

Let the network be a forward leaf sequence of three kinds: **affine** leaves
(``Linear``, ``Conv2d``; maskable), **piecewise-linear** leaves (``ReLU``,
``MaxPool2d``), and **linear or shape-only** leaves (``AdaptiveAvgPool2d``,
``AvgPool2d``, ``Flatten``, eval ``Dropout``, ``Identity``), with a linear
readout last.  Fix a context batch and a readout unit ``u`` (scalar ``s``).
Write ``h_k`` for the dense output of leaf ``k`` and ``G_k = ds/dh_k`` for the
dense cotangent at that output.  An action is a family of boolean removal masks
``M_k`` on hidden affine leaves.  Propagate the perturbation exactly,
``Delta_k = h'_k - h_k`` with ``h'`` the masked network's activations:

    affine:      Delta_k = -c_k + L_k^{W o not M_k}(Delta_{k-1}),  c_k = L_k^{W o M_k}(h_{k-1})  (no bias)
    piecewise:   Delta_k = J_k Delta_{k-1} + kappa_k,   J_k the frozen dense linearization
                 (ReLU: 1[h_{k-1} > 0] o . ;  MaxPool2d: gather at the dense argmax)
    linear:      Delta_k = L_k(Delta_{k-1})

Then, exactly,

    Delta s  =  sum_{affine k} ( -G_k . c_k )              first-order prices p_k
              + sum_{affine k} ( -G_k . L_k^{W o M_k}(Delta_{k-1}) )   removed-weight cross terms chi_k
              + sum_{piecewise k} G_k . kappa_k             pattern-change remainders (ReLU flips, pool switches)

*Proof.*  The readout is linear, so ``Delta s = G_{K-1} . Delta_{K-1}``.  For an
affine leaf, ``G_k . Delta_k = -G_k . c_k + G_k . W_k Delta_{k-1} - G_k . (W_k o M_k)
Delta_{k-1}`` and ``W_k^T G_k = G_{k-1}`` at the dense point; for a piecewise leaf
``G_k . Delta_k = G_k . J_k Delta_{k-1} + G_k . kappa_k`` with ``J_k^T G_k =
G_{k-1}``; for a linear leaf ``G_k . Delta_k = G_{k-1} . Delta_{k-1}``.
Telescoping from ``K-1`` to the input (``Delta_0 = 0``) gives the sum. ∎

Consequences.  (i) ``kappa_k`` is supported exactly on the gates that flip or
the pool windows whose argmax switches, so the remainder sum is the complete
pattern-change contribution and each ReLU term is the in-layer kink census of
``modern_v1_kink_census`` applied to the propagated drive.  (ii) ``chi_k``
vanishes for every leaf of a single-layer action.  (iii) For multi-layer actions
the cross terms are the exact multilinear interactions.  (iv) The
whole ledger costs one dense forward, one backward, and one propagated forward
per action, with no route-scale rows.

Claim scope: the leaf kinds above; anything else (``Tanh``, ``BatchNorm``,
``MaxPool1d``, training-mode ``Dropout``) refuses as a typed absence.  Float64.
No authority: a bounded receipt plus per-context terms; the paired oracle alone
commits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from mechanistic_systems.contracts.content_address import content_hash

try:  # pragma: no cover - torch is required at runtime, optional at import
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


EXACT_ACTION_DECOMPOSITION_SCHEMA: Final = (
    "partition_free_v1_exact_action_decomposition_piecewise_linear_sequential_v2"
)
EXACT_ACTION_DECOMPOSITION_THEOREM_ID: Final = (
    "delta_s_equals_first_order_plus_dense_cotangent_weighted_pattern_remainders_plus_removed_weight_cross_terms_v2"
)
EXACT_ACTION_DECOMPOSITION_AUTHORITY: Final = (
    "exact_bookkeeping_of_one_action_no_certificate_c_stream_or_commit_authority"
)
EXACT_ACTION_DECOMPOSITION_CLAIM_SCOPE: Final = (
    "linear_conv2d_relu_maxpool2d_avgpool_flatten_eval_dropout_identity_leaves_only_others_refuse"
)
_PREFIX: Final = "irg-v1-exact-action-decomposition-"
_AFFINE: Final = frozenset({"Linear", "Conv2d"})
_PIECEWISE: Final = frozenset({"ReLU", "MaxPool2d"})
_LINEAR_SHAPE: Final = frozenset({"AdaptiveAvgPool2d", "AvgPool2d", "Flatten", "Dropout", "Identity"})


class V1ExactActionDecompositionRefusal(RuntimeError):
    """Decomposition refused (leaf kind, mask placement, or shapes)."""


@dataclass(frozen=True, slots=True)
class V1ExactActionDecompositionReceipt:
    """Bounded summary of one decomposition over a context batch."""

    context_count: int
    hidden_layer_count: int  # maskable affine leaves (readout excluded)
    piecewise_leaf_count: int
    piecewise_leaf_kinds: tuple[str, ...]
    acted_layer_indices: tuple[int, ...]  # leaf indices
    removed_route_counts: tuple[int, ...]
    single_layer_action: bool
    mean_actual_change: float
    mean_first_order_total: float
    mean_kink_total: float
    mean_cross_total: float
    max_identity_residual: float
    flipped_gate_counts_by_layer: tuple[int, ...]  # per piecewise leaf (ReLU flips / pool switches)
    schema: str = EXACT_ACTION_DECOMPOSITION_SCHEMA
    theorem: str = EXACT_ACTION_DECOMPOSITION_THEOREM_ID
    claim_scope: str = EXACT_ACTION_DECOMPOSITION_CLAIM_SCOPE
    authority: str = EXACT_ACTION_DECOMPOSITION_AUTHORITY
    certificate_authority: bool = False
    commit_authority: bool = False

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "receipt-")


@dataclass(slots=True)
class V1ExactActionDecomposition:
    first_order: Any  # (N, A) per hidden affine leaf
    kink: Any  # (N, P) per piecewise leaf
    cross: Any  # (N, A)
    predicted_change: Any  # (N,)
    actual_change: Any  # (N,)
    identity_residual: Any  # (N,)
    flip_masks: tuple[Any, ...]  # per piecewise leaf, bool, shape of that leaf's output (pool) or input (relu)
    affine_leaf_indices: tuple[int, ...]
    piecewise_leaf_indices: tuple[int, ...]
    receipt: V1ExactActionDecompositionReceipt


def _kind(leaf: Any) -> str:
    kind = type(leaf).__name__
    if kind in _AFFINE or kind in _PIECEWISE or kind in _LINEAR_SHAPE:
        if kind == "Dropout" and getattr(leaf, "training", False):
            raise V1ExactActionDecompositionRefusal("Dropout must be in eval mode")
        return kind
    raise V1ExactActionDecompositionRefusal(f"unsupported leaf kind {kind!r} (typed absence)")


def _affine_apply(leaf: Any, kind: str, value: Any, weight: Any, bias: Any | None) -> Any:
    if kind == "Linear":
        return F.linear(value, weight, bias)
    return F.conv2d(value, weight, bias, stride=leaf.stride, padding=leaf.padding, dilation=leaf.dilation, groups=leaf.groups)


def _maxpool(leaf: Any, value: Any) -> tuple[Any, Any]:
    return F.max_pool2d(
        value,
        kernel_size=leaf.kernel_size,
        stride=leaf.stride,
        padding=leaf.padding,
        dilation=leaf.dilation,
        ceil_mode=leaf.ceil_mode,
        return_indices=True,
    )


def _gather_pool(delta: Any, indices: Any) -> Any:
    flat = delta.flatten(2)
    picked = flat.gather(2, indices.flatten(2))
    return picked.view_as(indices)


def _sum_dims(value: Any) -> Any:
    return value.reshape(value.shape[0], -1).sum(dim=1)


def decompose_exact_action(
    leaves: Sequence[Any],
    contexts: Any,
    *,
    removal_masks: Mapping[int, Any],
    unit: int,
) -> V1ExactActionDecomposition:
    """Exact decomposition of the change of readout ``unit`` under ``removal_masks``.

    ``removal_masks`` maps a leaf index (a hidden ``Linear``/``Conv2d`` leaf) to a
    boolean mask of that leaf's weight shape (``True`` = removed).
    """

    if torch is None:  # pragma: no cover
        raise V1ExactActionDecompositionRefusal("decomposition requires torch")
    leaves = tuple(leaves)
    kinds = tuple(_kind(leaf) for leaf in leaves)
    affine = [i for i, kind in enumerate(kinds) if kind in _AFFINE]
    if len(affine) < 2:
        raise V1ExactActionDecompositionRefusal("need at least one hidden affine leaf and a readout")
    readout = affine[-1]
    if kinds[readout] != "Linear" or readout != len(leaves) - 1:
        raise V1ExactActionDecompositionRefusal("the readout must be the last leaf and Linear")
    hidden = affine[:-1]
    piecewise = [i for i, kind in enumerate(kinds) if kind in _PIECEWISE]
    if not removal_masks:
        raise V1ExactActionDecompositionRefusal("an action needs at least one removal mask")
    masks: dict[int, Any] = {}
    for index, mask in removal_masks.items():
        if int(index) == readout:
            raise V1ExactActionDecompositionRefusal("the readout is outside the route axis")
        if int(index) not in hidden:
            raise V1ExactActionDecompositionRefusal(f"leaf {index} is not a hidden affine leaf")
        m = mask.to(torch.bool)
        if tuple(m.shape) != tuple(leaves[int(index)].weight.shape):
            raise V1ExactActionDecompositionRefusal("removal mask must match the weight shape")
        masks[int(index)] = m

    # Dense pass with the input as the differentiable leaf: every leaf output
    # is in the graph regardless of parameter grad flags.
    x = contexts.detach().to(torch.float64).clone().requires_grad_(True)
    outputs: list[Any] = []
    h = x
    for leaf in leaves:
        h = leaf(h)
        outputs.append(h)
    logits = outputs[-1]
    if logits.ndim != 2 or not 0 <= int(unit) < logits.shape[1]:
        raise V1ExactActionDecompositionRefusal("readout must be (N, units) and unit must index it")
    grads = torch.autograd.grad(logits[:, int(unit)].sum(), outputs[:-1], allow_unused=True)
    cot = [g.detach() if g is not None else None for g in grads]
    dense = [o.detach() for o in outputs]
    x0 = x.detach()

    n_ctx = int(x0.shape[0])
    first = torch.zeros(n_ctx, len(hidden), dtype=torch.float64)
    cross = torch.zeros(n_ctx, len(hidden), dtype=torch.float64)
    kink = torch.zeros(n_ctx, len(piecewise), dtype=torch.float64)
    flips: list[Any] = []
    flip_counts: list[int] = []
    delta = torch.zeros_like(x0)
    with torch.no_grad():
        for k, leaf in enumerate(leaves[:-1]):
            kind = kinds[k]
            h_prev = x0 if k == 0 else dense[k - 1]
            G = cot[k]
            if G is None:
                raise V1ExactActionDecompositionRefusal(f"leaf {k} is disconnected from the readout unit")
            if kind in _AFFINE:
                a_idx = hidden.index(k)
                W = leaf.weight.detach().to(torch.float64)
                M = masks.get(k)
                removed = W * M.to(torch.float64) if M is not None else torch.zeros_like(W)
                kept = W - removed
                c = _affine_apply(leaf, kind, h_prev, removed, None)
                cross_drive = _affine_apply(leaf, kind, delta, removed, None)
                delta = -c + _affine_apply(leaf, kind, delta, kept, None)
                first[:, a_idx] = -_sum_dims(G * c)
                cross[:, a_idx] = -_sum_dims(G * cross_drive)
            elif kind == "ReLU":
                p_idx = piecewise.index(k)
                gate = (h_prev > 0).to(torch.float64)
                new = torch.relu(h_prev + delta) - torch.relu(h_prev)
                kappa = new - gate * delta
                kink[:, p_idx] = _sum_dims(G * kappa)
                flip = (h_prev > 0) != ((h_prev + delta) > 0)
                flips.append(flip)
                flip_counts.append(int(flip.sum()))
                delta = new
            elif kind == "MaxPool2d":
                p_idx = piecewise.index(k)
                pooled, idx = _maxpool(leaf, h_prev)
                pooled_new, idx_new = _maxpool(leaf, h_prev + delta)
                new = pooled_new - pooled
                kappa = new - _gather_pool(delta, idx)
                kink[:, p_idx] = _sum_dims(G * kappa)
                flip = idx_new != idx
                flips.append(flip)
                flip_counts.append(int(flip.sum()))
                delta = new
            else:  # linear or shape-only leaf: exact difference
                delta = leaf(h_prev + delta) - leaf(h_prev)
        # Actual change: masked forward.
        hm = x0
        for k, leaf in enumerate(leaves):
            kind = kinds[k]
            if kind in _AFFINE and k in masks:
                W = leaf.weight.detach().to(torch.float64) * (~masks[k]).to(torch.float64)
                b = None if leaf.bias is None else leaf.bias.detach().to(torch.float64)
                hm = _affine_apply(leaf, kind, hm, W, b)
            else:
                hm = leaf(hm)
        actual = hm[:, int(unit)] - dense[-1][:, int(unit)]
    predicted = first.sum(dim=1) + kink.sum(dim=1) + cross.sum(dim=1)
    residual = (predicted - actual).abs()
    acted = tuple(sorted(masks))
    receipt = V1ExactActionDecompositionReceipt(
        context_count=n_ctx,
        hidden_layer_count=len(hidden),
        piecewise_leaf_count=len(piecewise),
        piecewise_leaf_kinds=tuple(kinds[i] for i in piecewise),
        acted_layer_indices=acted,
        removed_route_counts=tuple(int(masks[i].sum()) for i in acted),
        single_layer_action=len(acted) == 1,
        mean_actual_change=float(actual.mean()),
        mean_first_order_total=float(first.sum(dim=1).mean()),
        mean_kink_total=float(kink.sum(dim=1).mean()),
        mean_cross_total=float(cross.sum(dim=1).mean()),
        max_identity_residual=float(residual.max()),
        flipped_gate_counts_by_layer=tuple(flip_counts),
    )
    return V1ExactActionDecomposition(
        first_order=first,
        kink=kink,
        cross=cross,
        predicted_change=predicted,
        actual_change=actual,
        identity_residual=residual,
        flip_masks=tuple(flips),
        affine_leaf_indices=tuple(hidden),
        piecewise_leaf_indices=tuple(piecewise),
        receipt=receipt,
    )


__all__ = [
    "EXACT_ACTION_DECOMPOSITION_AUTHORITY",
    "EXACT_ACTION_DECOMPOSITION_CLAIM_SCOPE",
    "EXACT_ACTION_DECOMPOSITION_SCHEMA",
    "EXACT_ACTION_DECOMPOSITION_THEOREM_ID",
    "V1ExactActionDecomposition",
    "V1ExactActionDecompositionReceipt",
    "V1ExactActionDecompositionRefusal",
    "decompose_exact_action",
]
