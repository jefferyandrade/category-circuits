"""
Hessian-vector product utilities used for frontier salience analysis.

The helpers in this module provide:
    * Gauss–Legendre two-point integrated Hessian evaluation along a straight
      line path (for Integrated Gradients style probes).
    * Helpers to work directly in activation space (pre-activation ``z``) or to
      lift the integrated Hessian into linear weight space.
"""
from __future__ import annotations

from math import sqrt
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _line_point(
    baseline: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return the point along the straight line connecting ``baseline`` to ``target``."""

    return baseline + alpha * (target - baseline)


def _ensure_tensor(x: torch.Tensor) -> torch.Tensor:
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    return x


def gauss_legendre_two_point(
    scalar_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    vector: torch.Tensor,
    *,
    baseline: Optional[torch.Tensor] = None,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Approximate an integrated Hessian-vector product with the Gauss–Legendre rule.

    The approximation follows the two-node Gauss–Legendre quadrature on ``[0, 1]``.
    It is particularly handy when the scalar function represents an Integrated
    Gradients path from ``baseline`` to ``inputs``.

    Args:
        scalar_fn: Callable that consumes a tensor with the same shape as ``inputs``
            and returns a scalar tensor (e.g. a loss value).
        inputs: Target point ``x`` at which the integrated Hessian is evaluated.
        vector: Direction ``v`` for which ``H_integrated(x) @ v`` is desired.
        baseline: Optional baseline ``x0`` defining the integration path.
            Defaults to the zero tensor if omitted.
        retain_graph: Whether to retain the computation graph between the two
            quadrature evaluations (useful if the caller wants more grad work).

    Returns:
        Tensor containing the approximate Hessian-vector product with the same
        shape as ``inputs``.
    """

    if baseline is None:
        baseline = torch.zeros_like(inputs)

    alphas = (
        0.5 - 1.0 / (2.0 * sqrt(3.0)),
        0.5 + 1.0 / (2.0 * sqrt(3.0)),
    )
    weights = (0.5, 0.5)

    result: Optional[torch.Tensor] = None

    for weight, alpha in zip(weights, alphas):
        point = _line_point(baseline, inputs, alpha)
        point = _ensure_tensor(point)

        scalar = scalar_fn(point)
        if scalar.ndim != 0:
            scalar = scalar.sum()

        grad = torch.autograd.grad(scalar, point, create_graph=True)[0]
        hvp = torch.autograd.grad(grad, point, grad_outputs=vector, retain_graph=retain_graph)[0]

        result = weight * hvp if result is None else result + weight * hvp

    if result is None:
        raise RuntimeError("Gauss–Legendre evaluation produced no result. Check scalar_fn.")

    return result


def integrated_hvp_z(
    g: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    s: torch.Tensor,
    *,
    baseline: Optional[torch.Tensor] = None,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Approximate the integrated Hessian-vector product in activation space.

    Args:
        g: Callable that maps a pre-activation tensor to a scalar objective.
        z: Target pre-activation tensor (the endpoint of the IG path).
        s: Direction in activation space with the same shape as ``z``.
        baseline: Optional baseline tensor defining the start of the path. When
            omitted, a zero tensor with the same shape as ``z`` is used.
        retain_graph: Whether to retain the autograd graph after computing the
            HVP (useful when the caller needs additional higher-order grads).

    Returns:
        Tensor containing the Gauss–Legendre 2-point approximation of the
        integrated Hessian-vector product along the straight line path from
        ``baseline`` to ``z`` applied to the direction ``s``.
    """

    if not callable(g):
        raise TypeError("integrated_hvp_z expects a callable `g(z)` returning a scalar tensor.")
    if not isinstance(z, torch.Tensor) or not isinstance(s, torch.Tensor):
        raise TypeError("`z` and `s` must be torch.Tensors.")
    if z.shape != s.shape:
        raise ValueError(f"Shape mismatch: z.shape={z.shape} vs s.shape={s.shape}.")
    if baseline is None:
        baseline = torch.zeros_like(z)
    elif baseline.shape != z.shape:
        raise ValueError("`baseline` must match the shape of `z`.")

    device = z.device
    dtype = z.dtype
    direction = s.to(device=device, dtype=dtype)

    offset = 1.0 / (2.0 * sqrt(3.0))
    alphas = (
        0.5 - offset,
        0.5 + offset,
    )
    weights = (0.5, 0.5)

    result: Optional[torch.Tensor] = None

    for idx, (weight, alpha) in enumerate(zip(weights, alphas)):
        point = baseline + alpha * (z - baseline)
        point = point.detach().requires_grad_(True)

        scalar = g(point)
        if not isinstance(scalar, torch.Tensor):
            raise TypeError("The callable `g` must return a torch.Tensor.")
        if scalar.ndim != 0:
            scalar = scalar.sum()

        grad = torch.autograd.grad(
            scalar,
            point,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]

        hvp = torch.autograd.grad(
            grad,
            point,
            grad_outputs=direction,
            retain_graph=retain_graph or idx < len(alphas) - 1,
            allow_unused=False,
        )[0]

        contribution = hvp * weight
        result = contribution if result is None else result + contribution

    if result is None:
        raise RuntimeError("integrated_hvp_z failed to compute a Hessian-vector product.")

    return result


def integrated_hvp_linear(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    g: Callable[[torch.Tensor], torch.Tensor],
    vec: torch.Tensor,
    *,
    baseline: Optional[torch.Tensor] = None,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Lift an integrated activation-space HVP into linear weight space.

    Args:
        weight: The current weight matrix ``W`` with shape ``(out, in)``.
        inputs: Input activations ``X`` with shape ``(batch, in)``.
        g: Callable that consumes the pre-activation tensor ``Z = X Wᵀ`` and
            returns a scalar objective.
        vec: Direction in weight space matching the shape of ``weight``.
        baseline: Optional baseline weight tensor defining the start of the IG
            path. Defaults to zeros.
        retain_graph: Whether to retain autograd graphs after the computation.

    Returns:
        Tensor representing the integrated Hessian applied to ``vec`` in weight
        space via the Gauss–Legendre two-point rule.
    """

    if not callable(g):
        raise TypeError("integrated_hvp_linear expects a callable `g(z)` returning a scalar tensor.")
    if not isinstance(weight, torch.Tensor) or not isinstance(vec, torch.Tensor):
        raise TypeError("`weight` and `vec` must be torch.Tensors.")
    if weight.shape != vec.shape:
        raise ValueError(f"Shape mismatch: weight.shape={weight.shape} vs vec.shape={vec.shape}.")

    inputs_detached = inputs.detach()
    if inputs_detached.ndim != 2:
        raise ValueError("`inputs` must have shape (batch, in_features).")

    if baseline is None:
        baseline = torch.zeros_like(weight)
    elif baseline.shape != weight.shape:
        raise ValueError("`baseline` must match the shape of `weight`.")

    diff = weight - baseline
    direction = vec.to(device=weight.device, dtype=weight.dtype)

    offset = 1.0 / (2.0 * sqrt(3.0))
    alphas = (
        0.5 - offset,
        0.5 + offset,
    )
    weights = (0.5, 0.5)

    result: Optional[torch.Tensor] = None

    for idx, (quad_weight, alpha) in enumerate(zip(weights, alphas)):
        weight_point = baseline + alpha * diff
        weight_point = weight_point.detach().requires_grad_(True)

        z_point = F.linear(inputs_detached, weight_point)
        scalar = g(z_point)
        if not isinstance(scalar, torch.Tensor):
            raise TypeError("The callable `g` must return a torch.Tensor.")
        if scalar.ndim != 0:
            scalar = scalar.sum()

        grad_w = torch.autograd.grad(
            scalar,
            weight_point,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]

        hvp_w = torch.autograd.grad(
            grad_w,
            weight_point,
            grad_outputs=direction,
            retain_graph=retain_graph or idx < len(alphas) - 1,
            allow_unused=False,
        )[0]

        contribution = hvp_w * quad_weight
        result = contribution if result is None else result + contribution

    if result is None:
        raise RuntimeError("integrated_hvp_linear failed to compute a Hessian-vector product.")

    return result


def activation_block_hvp(
    scalar: torch.Tensor,
    pre_activation: torch.Tensor,
    vectors: Iterable[torch.Tensor],
    *,
    retain_graph: bool = True,
) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
    """Compute Hessian-vector products with respect to a pre-activation tensor.

    Args:
        scalar: Scalar objective whose Hessian is desired.
        pre_activation: Tensor ``z`` representing pre-activation outputs.
        vectors: Iterable of direction tensors (same shape as ``pre_activation``).
        retain_graph: Whether to retain the graph after computing the HVPs.

    Returns:
        A tuple containing the gradient ``∂scalar/∂z`` (or ``None`` if unavailable)
        and a list of Hessian-vector products, one per entry in ``vectors``.
    """

    directions = list(vectors)
    if not directions:
        return None, []

    grad = torch.autograd.grad(
        scalar,
        pre_activation,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        return None, [torch.zeros_like(direction) for direction in directions]

    hvps: List[torch.Tensor] = []
    for direction in directions:
        hvp = torch.autograd.grad(
            grad,
            pre_activation,
            grad_outputs=direction,
            retain_graph=retain_graph,
            allow_unused=True,
        )[0]
        if hvp is None:
            hvp = torch.zeros_like(direction)
        hvps.append(hvp)
    return grad, hvps


def batched_hvp(
    scalar_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    vectors: Iterable[torch.Tensor],
    *,
    baseline: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """Compute multiple HVPs using the same scalar function and input point.

    Args:
        scalar_fn: Callable returning a scalar tensor.
        inputs: Tensor where the HVP is evaluated.
        vectors: Iterable of direction tensors.
        baseline: Optional baseline for the integrated path.

    Returns:
        Tuple of tensors, each the same shape as ``inputs``.
    """

    outputs = []
    cached_grad = None

    for idx, vector in enumerate(vectors):
        if cached_grad is None:
            point = _ensure_tensor(inputs.clone())
            scalar = scalar_fn(point)
            if scalar.ndim != 0:
                scalar = scalar.sum()
            cached_grad = torch.autograd.grad(scalar, point, create_graph=True)[0]
        hvp = torch.autograd.grad(cached_grad, point, grad_outputs=vector, retain_graph=True)[0]
        outputs.append(hvp)

    # If we reused the cached grad we should detach it to avoid holding onto graphs.
    if cached_grad is not None:
        del cached_grad

    return tuple(outputs)


def simple_hvp(scalar: torch.Tensor, parameter: torch.Tensor, vector: torch.Tensor) -> Optional[torch.Tensor]:
    """Compute a Hessian-vector product using full autograd on ``scalar``.

    Args:
        scalar: Scalar tensor whose Hessian is of interest.
        parameter: Parameter tensor with respect to which the Hessian is taken.
        vector: Direction tensor (same shape as ``parameter``).

    Returns:
        The HVP tensor or ``None`` if the parameter does not influence ``scalar``.
    """

    if not parameter.requires_grad:
        return None

    grad = torch.autograd.grad(
        scalar,
        parameter,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        return None
    hvp = torch.autograd.grad(
        grad,
        parameter,
        grad_outputs=vector,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return hvp
