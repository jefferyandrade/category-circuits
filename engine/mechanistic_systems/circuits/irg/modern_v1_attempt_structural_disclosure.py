"""Attempt-level structural disclosure: the exact action decomposition of one candidate action.

For one production attempt — parent hard mask ``m_p`` (the current chart), candidate
hard mask ``m_c`` (the proposal's resulting mask), target unit ``u``, and a batch of
contexts — this module derives the per-layer removal masks ``M_l = m_p & ~m_c`` on the
declared route blocks, applies the parent mask to a copy of the model's leaves (the
chart is the parent-masked network), runs the exact action decomposition
(``modern_v1_exact_action_decomposition``: first-order prices, removed-weight cross
terms, pattern-change remainders for ReLU flips and pool switches), and issues a
bounded, hashable receipt keyed by the candidate and parent mask hashes.

It is a **disclosure**: it prices nothing, certifies nothing, and cannot change an
accept or reject.  By default it stays a disclosure
on trajectory 1.  Everything it reports is exact bookkeeping of the in-network change
on the disclosed contexts; the estimand on held-out contexts remains the paired
oracle's alone.

Refusals are typed: a candidate that re-adds a route the parent had removed is not an
action (``NOT_AN_ACTION``); masks on another route axis (``AXIS_MISMATCH``); a route
block whose module is not an affine leaf of the forward sequence (``BINDING``);
unsupported leaves refuse inside the decomposition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.core.storage import RouteBlockSpec

from .modern_v1_exact_action_decomposition import (
    EXACT_ACTION_DECOMPOSITION_THEOREM_ID,
    V1ExactActionDecomposition,
    decompose_exact_action,
)

try:  # pragma: no cover - torch is required at runtime, optional at import
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ATTEMPT_STRUCTURAL_DISCLOSURE_SCHEMA: Final = (
    "partition_free_v1_attempt_structural_disclosure_v1"
)
ATTEMPT_STRUCTURAL_DISCLOSURE_REFUSAL_SCHEMA: Final = (
    "partition_free_v1_attempt_structural_disclosure_refusal_v1"
)
ATTEMPT_STRUCTURAL_DISCLOSURE_AUTHORITY: Final = (
    "disclosure_only_no_price_certificate_c_stream_or_commit_authority"
)
ATTEMPT_STRUCTURAL_DISCLOSURE_ROLE: Final = "disclosure"
_PREFIX: Final = "irg-v1-attempt-structural-disclosure-"


class V1AttemptStructuralDisclosureRefusalCode(str, Enum):
    NOT_AN_ACTION = "not_an_action"
    AXIS_MISMATCH = "axis_mismatch"
    BINDING = "binding"
    INVALID_TYPE = "invalid_type"
    MISSING_INPUT = "missing_input"


class V1AttemptStructuralDisclosureRefusal(RuntimeError):
    def __init__(self, code: V1AttemptStructuralDisclosureRefusalCode, message: str) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code


def _refuse(code: V1AttemptStructuralDisclosureRefusalCode, message: str) -> NoReturn:
    raise V1AttemptStructuralDisclosureRefusal(code, message)


# --------------------------------------------------------------------------- #
# Leaf sequences and bindings                                                  #
# --------------------------------------------------------------------------- #


def forward_leaves(model: Any) -> tuple[Any, ...]:
    """Ordered forward leaves of a torchvision-style model or a flat ``Sequential``.

    ``features`` / ``avgpool`` (+ an inserted ``Flatten``) / ``classifier`` when
    present (torchvision AlexNet flattens in ``forward``), else the model's
    children.  Containers are expanded one level.
    """

    if torch is None:  # pragma: no cover
        _refuse(V1AttemptStructuralDisclosureRefusalCode.MISSING_INPUT, "forward_leaves requires torch")
    stages: list[Any] = []
    for name in ("features", "avgpool", "classifier"):
        if hasattr(model, name):
            stages.append(getattr(model, name))
            if name == "avgpool":
                stages.append(torch.nn.Flatten())
    if not stages:
        stages = [model]
    leaves: list[Any] = []
    for stage in stages:
        children = list(stage.children())
        leaves.extend(children if children else [stage])
    return tuple(leaves)


def leaf_index_by_module_path(model: Any, leaves: Sequence[Any], module_paths: Mapping[str, str]) -> dict[str, int]:
    """Map ``layer_id -> leaf index`` from ``layer_id -> module path`` by object identity."""

    modules = dict(model.named_modules())
    result: dict[str, int] = {}
    for layer_id, path in module_paths.items():
        module = modules.get(str(path))
        if module is None:
            _refuse(V1AttemptStructuralDisclosureRefusalCode.BINDING, f"module path {path!r} not in the model")
        index = next((i for i, leaf in enumerate(leaves) if leaf is module), None)
        if index is None:
            _refuse(V1AttemptStructuralDisclosureRefusalCode.BINDING, f"module {path!r} is not a forward leaf")
        result[str(layer_id)] = index
    return result


def _block_slices(route_blocks: Sequence[RouteBlockSpec]) -> tuple[tuple[RouteBlockSpec, int, int], ...]:
    ordered = tuple(sorted(route_blocks, key=lambda block: int(block.offset)))
    spans = []
    cursor = 0
    for block in ordered:
        if int(block.offset) != cursor:
            _refuse(V1AttemptStructuralDisclosureRefusalCode.AXIS_MISMATCH, "route blocks are not contiguous on the axis")
        spans.append((block, cursor, cursor + int(block.route_count)))
        cursor += int(block.route_count)
    return tuple(spans)


def per_layer_masks_from_flat(
    flat: np.ndarray,
    route_blocks: Sequence[RouteBlockSpec],
    leaf_index_by_layer: Mapping[str, int],
) -> dict[int, Any]:
    """Reshape a flat boolean route vector into ``{leaf_index: bool tensor(weight shape)}``."""

    spans = _block_slices(route_blocks)
    if spans[-1][2] != int(flat.size):
        _refuse(V1AttemptStructuralDisclosureRefusalCode.AXIS_MISMATCH, "flat route vector differs from the block axis")
    out: dict[int, Any] = {}
    for block, start, stop in spans:
        # Both declared conventions are C-order over the block's own shape
        # ((target, source) for linear; (target, source, kh, kw) for conv taps),
        # so ``reshape(shape)`` recovers the weight layout in either case.
        if block.flattening not in {"c_order_target_then_source", "c_order_target_source_taps_v1"}:
            _refuse(V1AttemptStructuralDisclosureRefusalCode.AXIS_MISMATCH, f"unsupported block flattening {block.flattening!r}")
        index = leaf_index_by_layer.get(str(block.layer_id))
        if index is None:
            _refuse(V1AttemptStructuralDisclosureRefusalCode.BINDING, f"route block {block.layer_id!r} has no bound leaf")
        out[int(index)] = torch.as_tensor(np.asarray(flat[start:stop], dtype=bool).reshape(tuple(block.shape)))
    return out


def removal_masks_from_hard_masks(
    parent: HardMask,
    candidate: HardMask,
    route_blocks: Sequence[RouteBlockSpec],
    leaf_index_by_layer: Mapping[str, int],
) -> dict[int, Any]:
    """Per-leaf removal masks of the action ``parent -> candidate`` (non-empty layers only)."""

    if type(parent) is not HardMask or type(candidate) is not HardMask:
        _refuse(V1AttemptStructuralDisclosureRefusalCode.INVALID_TYPE, "parent and candidate must be exact HardMasks")
    if parent.route_axis_hash != candidate.route_axis_hash or parent.route_count != candidate.route_count:
        _refuse(V1AttemptStructuralDisclosureRefusalCode.AXIS_MISMATCH, "parent and candidate masks live on different axes")
    kept_parent = parent.to_bool_array()
    kept_candidate = candidate.to_bool_array()
    if bool(np.any(kept_candidate & ~kept_parent)):
        _refuse(V1AttemptStructuralDisclosureRefusalCode.NOT_AN_ACTION, "candidate re-adds routes the parent had removed")
    removed = kept_parent & ~kept_candidate
    if not bool(removed.any()):
        _refuse(V1AttemptStructuralDisclosureRefusalCode.NOT_AN_ACTION, "candidate removes nothing")
    masks = per_layer_masks_from_flat(removed, route_blocks, leaf_index_by_layer)
    return {index: mask for index, mask in masks.items() if bool(mask.any())}


def apply_hard_mask_to_leaves(
    leaves: Sequence[Any],
    mask: HardMask,
    route_blocks: Sequence[RouteBlockSpec],
    leaf_index_by_layer: Mapping[str, int],
) -> tuple[Any, ...]:
    """Deep-copy ``leaves`` with the retained routes of ``mask`` applied to the bound weights (the chart)."""

    kept = per_layer_masks_from_flat(mask.to_bool_array(), route_blocks, leaf_index_by_layer)
    # float64 copies: the decomposition is evaluated in float64 regardless of the
    # production model's dtype (an engineering cost per attempt; disclosure only).
    copies = tuple(
        copy.deepcopy(leaf).double() if hasattr(leaf, "double") else copy.deepcopy(leaf)
        for leaf in leaves
    )
    with torch.no_grad():
        for index, keep in kept.items():
            weight = copies[index].weight
            if tuple(weight.shape) != tuple(keep.shape):
                _refuse(V1AttemptStructuralDisclosureRefusalCode.AXIS_MISMATCH, "route block shape differs from the bound weight")
            weight.mul_(keep.to(device=weight.device, dtype=weight.dtype))
    for leaf in copies:
        if hasattr(leaf, "eval"):
            leaf.eval()
    return copies


# --------------------------------------------------------------------------- #
# Receipt                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class V1AttemptStructuralDisclosureReceipt:
    parent_mask_hash: str
    candidate_mask_hash: str
    dense_reference_hash: str
    route_axis_hash: str
    target_unit: int
    context_ids_hash: str
    context_count: int
    generation_index: int | None
    candidate_attempt_index: int | None
    acted_layer_ids: tuple[str, ...]
    removed_route_counts: tuple[int, ...]
    single_layer_action: bool
    hidden_layer_count: int
    piecewise_leaf_kinds: tuple[str, ...]
    mean_actual_change: float
    mean_first_order_total: float
    mean_kink_total: float
    mean_cross_total: float
    max_identity_residual: float
    flipped_gate_counts_by_layer: tuple[int, ...]
    context_fraction_with_any_flip: float
    decomposition_receipt_hash: str
    schema: str = ATTEMPT_STRUCTURAL_DISCLOSURE_SCHEMA
    theorem: str = EXACT_ACTION_DECOMPOSITION_THEOREM_ID
    role: str = ATTEMPT_STRUCTURAL_DISCLOSURE_ROLE
    authority: str = ATTEMPT_STRUCTURAL_DISCLOSURE_AUTHORITY
    price_authority: bool = False
    certificate_authority: bool = False
    commit_authority: bool = False

    def __post_init__(self) -> None:
        if self.price_authority or self.certificate_authority or self.commit_authority:
            _refuse(V1AttemptStructuralDisclosureRefusalCode.INVALID_TYPE, "disclosure receipts carry no authority")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "receipt-")

    def to_json_payload(self) -> dict[str, Any]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__  # type: ignore[attr-defined]
        }
        payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in payload.items()}
        payload["receipt_hash"] = self.receipt_hash
        return payload


def refusal_payload(
    *,
    parent_mask_hash: str,
    candidate_mask_hash: str,
    dense_reference_hash: str,
    generation_index: int | None,
    candidate_attempt_index: int | None,
    code: str,
    message: str,
) -> dict[str, Any]:
    """Durable typed-absence payload when the disclosure could not be issued."""

    body = {
        "schema": ATTEMPT_STRUCTURAL_DISCLOSURE_REFUSAL_SCHEMA,
        "parent_mask_hash": str(parent_mask_hash),
        "candidate_mask_hash": str(candidate_mask_hash),
        "dense_reference_hash": str(dense_reference_hash),
        "generation_index": generation_index,
        "candidate_attempt_index": candidate_attempt_index,
        "refusal_code": str(code),
        "message": str(message)[:2000],
        "role": ATTEMPT_STRUCTURAL_DISCLOSURE_ROLE,
        "authority": ATTEMPT_STRUCTURAL_DISCLOSURE_AUTHORITY,
        "price_authority": False,
        "certificate_authority": False,
        "commit_authority": False,
    }
    body["receipt_hash"] = content_hash(body, prefix=_PREFIX + "refusal-")
    return body


# --------------------------------------------------------------------------- #
# Issuance                                                                     #
# --------------------------------------------------------------------------- #


def issue_v1_attempt_structural_disclosure(
    *,
    leaves: Sequence[Any],
    inputs: Any,
    context_ids: Sequence[str],
    unit: int,
    parent: HardMask,
    candidate: HardMask,
    dense_reference_hash: str,
    route_blocks: Sequence[RouteBlockSpec],
    leaf_index_by_layer: Mapping[str, int],
    generation_index: int | None = None,
    candidate_attempt_index: int | None = None,
) -> tuple[V1AttemptStructuralDisclosureReceipt, V1ExactActionDecomposition]:
    """Decompose the action ``parent -> candidate`` over ``inputs`` (the chart is the parent-masked network)."""

    if torch is None:  # pragma: no cover
        _refuse(V1AttemptStructuralDisclosureRefusalCode.MISSING_INPUT, "disclosure requires torch")
    contexts = tuple(str(value) for value in context_ids)
    if not contexts or int(inputs.shape[0]) != len(contexts):
        _refuse(V1AttemptStructuralDisclosureRefusalCode.AXIS_MISMATCH, "inputs and context ids disagree")
    removal = removal_masks_from_hard_masks(parent, candidate, route_blocks, leaf_index_by_layer)
    chart = apply_hard_mask_to_leaves(leaves, parent, route_blocks, leaf_index_by_layer)
    result = decompose_exact_action(chart, inputs, removal_masks=removal, unit=int(unit))
    index_to_layer = {int(v): str(k) for k, v in leaf_index_by_layer.items()}
    acted = tuple(index_to_layer[i] for i in result.receipt.acted_layer_indices)
    any_flip = torch.stack([f.reshape(f.shape[0], -1).any(dim=1) for f in result.flip_masks]).any(dim=0) if result.flip_masks else torch.zeros(len(contexts), dtype=torch.bool)
    receipt = V1AttemptStructuralDisclosureReceipt(
        parent_mask_hash=parent.mask_hash,
        candidate_mask_hash=candidate.mask_hash,
        dense_reference_hash=str(dense_reference_hash),
        route_axis_hash=parent.route_axis_hash,
        target_unit=int(unit),
        context_ids_hash=content_hash(contexts, prefix=_PREFIX + "contexts-"),
        context_count=len(contexts),
        generation_index=None if generation_index is None else int(generation_index),
        candidate_attempt_index=None if candidate_attempt_index is None else int(candidate_attempt_index),
        acted_layer_ids=acted,
        removed_route_counts=result.receipt.removed_route_counts,
        single_layer_action=result.receipt.single_layer_action,
        hidden_layer_count=result.receipt.hidden_layer_count,
        piecewise_leaf_kinds=result.receipt.piecewise_leaf_kinds,
        mean_actual_change=result.receipt.mean_actual_change,
        mean_first_order_total=result.receipt.mean_first_order_total,
        mean_kink_total=result.receipt.mean_kink_total,
        mean_cross_total=result.receipt.mean_cross_total,
        max_identity_residual=result.receipt.max_identity_residual,
        flipped_gate_counts_by_layer=result.receipt.flipped_gate_counts_by_layer,
        context_fraction_with_any_flip=float(any_flip.to(torch.float64).mean()),
        decomposition_receipt_hash=result.receipt.receipt_hash,
    )
    return receipt, result


__all__ = [
    "ATTEMPT_STRUCTURAL_DISCLOSURE_AUTHORITY",
    "ATTEMPT_STRUCTURAL_DISCLOSURE_REFUSAL_SCHEMA",
    "ATTEMPT_STRUCTURAL_DISCLOSURE_ROLE",
    "ATTEMPT_STRUCTURAL_DISCLOSURE_SCHEMA",
    "V1AttemptStructuralDisclosureReceipt",
    "V1AttemptStructuralDisclosureRefusal",
    "V1AttemptStructuralDisclosureRefusalCode",
    "apply_hard_mask_to_leaves",
    "forward_leaves",
    "issue_v1_attempt_structural_disclosure",
    "leaf_index_by_module_path",
    "per_layer_masks_from_flat",
    "refusal_payload",
    "removal_masks_from_hard_masks",
]
