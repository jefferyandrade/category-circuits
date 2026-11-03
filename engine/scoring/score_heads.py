from __future__ import annotations

"""Canonical score-head names and compatibility aliases.

New durable artifacts use route-native names.  Legacy feedforward names remain
accepted as aliases so historical AlexNet artifacts and old pruning consumers
can still be read without making those names the active production contract.
"""

ROUTE_SCORE_HEAD = "route_scores"
ROUTE_ABS_SCORE_HEAD = "route_abs_scores"
UNIT_SCORE_HEAD = "unit_scores"
UNIT_ABS_SCORE_HEAD = "unit_abs_scores"
PASS_SPECIFIC_ROUTE_ABS_SCORE_HEAD = "pass_specific_route_abs_scores"
LEGACY_OLDFAST_ABS_SCORE_HEAD = "legacy_oldfast_abs_scores"

CORE_ROUTE_SCORE_HEADS: tuple[str, ...] = (
    ROUTE_SCORE_HEAD,
    ROUTE_ABS_SCORE_HEAD,
    UNIT_SCORE_HEAD,
    UNIT_ABS_SCORE_HEAD,
)

LEGACY_TO_ROUTE_SCORE_HEAD: dict[str, str] = {
    "all_scores": ROUTE_SCORE_HEAD,
    "all_abs_scores": ROUTE_ABS_SCORE_HEAD,
    "neuron_scores": UNIT_SCORE_HEAD,
    "neuron_abs_scores": UNIT_ABS_SCORE_HEAD,
    "pass_specific_all_abs_scores": PASS_SPECIFIC_ROUTE_ABS_SCORE_HEAD,
}
ROUTE_TO_LEGACY_SCORE_HEAD: dict[str, str] = {
    value: key for key, value in LEGACY_TO_ROUTE_SCORE_HEAD.items()
}

SCORESTORE_COMPATIBILITY_ALIASES: dict[str, str] = {
    **LEGACY_TO_ROUTE_SCORE_HEAD,
    "scores": ROUTE_ABS_SCORE_HEAD,
    "oldfast_all_abs_scores": LEGACY_OLDFAST_ABS_SCORE_HEAD,
}


def unit_reduction_from_route_scores(
    route_scores: dict[str, "torch.Tensor"],
    *,
    absolute: bool,
) -> dict[str, "torch.Tensor"]:
    """Reduce route tensors to their downstream unit analogue.

    The route axis convention is backend-neutral: the last axis is the incoming
    route/source dimension when one exists.  Summing that axis gives the natural
    downstream object for Conv/FC routes, residual branch gates, LRM pass routes,
    attention head/query routes, and MLP channels.  One-dimensional route
    objects are already unit-like and are copied unchanged.
    """

    import torch

    reduced: dict[str, torch.Tensor] = {}
    for layer, tensor in route_scores.items():
        value = tensor.detach().to(dtype=torch.float32)
        if absolute:
            value = value.abs()
        if value.ndim > 1:
            value = value.sum(dim=-1)
        reduced[str(layer)] = value.contiguous()
    return reduced


def canonical_score_head(head: str) -> str:
    """Return the modern durable score-head name for ``head``."""

    token = str(head)
    return LEGACY_TO_ROUTE_SCORE_HEAD.get(token, token)


def reducer_source_score_head(head: str) -> str:
    """Return the low-level reducer key that currently supplies ``head``."""

    token = str(head)
    return ROUTE_TO_LEGACY_SCORE_HEAD.get(token, token)


def canonical_score_heads(heads: list[str] | tuple[str, ...] | set[str]) -> tuple[str, ...]:
    """Canonicalize and de-duplicate score heads while preserving order."""

    out: list[str] = []
    for head in heads:
        canonical = canonical_score_head(str(head))
        if canonical not in out:
            out.append(canonical)
    return tuple(out)


__all__ = [
    "CORE_ROUTE_SCORE_HEADS",
    "LEGACY_TO_ROUTE_SCORE_HEAD",
    "LEGACY_OLDFAST_ABS_SCORE_HEAD",
    "PASS_SPECIFIC_ROUTE_ABS_SCORE_HEAD",
    "ROUTE_ABS_SCORE_HEAD",
    "ROUTE_SCORE_HEAD",
    "ROUTE_TO_LEGACY_SCORE_HEAD",
    "SCORESTORE_COMPATIBILITY_ALIASES",
    "UNIT_ABS_SCORE_HEAD",
    "UNIT_SCORE_HEAD",
    "canonical_score_head",
    "canonical_score_heads",
    "reducer_source_score_head",
    "unit_reduction_from_route_scores",
]
