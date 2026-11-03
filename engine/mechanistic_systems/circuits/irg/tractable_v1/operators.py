"""Exact operator-form evaluation of the v1 row streams and response geometry.

Given the row factors (``X`` unfolded inputs, ``Gt`` corrected
cotangent, ``D`` literal-J cotangent, ``W`` weights, ``m`` mask), all
quantities of the chart are evaluated without forming context-by-route rows:

  mean price (work vector)        g[f,j]   = -W[f,j] m[f,j] sum_c w_c sum_p Gt[c,f,p] X[c,j,p]
  corrected forward (A z)         cell[c,f]= -sum_p Gt[c,f,p] ((W m Z) X_c)[f,p]
  corrected adjoint (A^T y)       out[f,j] = -W[f,j] m[f,j] sum_c y[c,f] sum_p Gt[c,f,p] X[c,j,p]
  literal-J forward (J z)         l[c]     = -sum_{f,p} D[c,f,p] ((W m Z) X_c)[f,p]
  literal-J adjoint (J^T y)       out[f,j] = -W[f,j] m[f,j] sum_c y[c] sum_p D[c,f,p] X[c,j,p]
  response Gram                   Gram = J J^T  over eligible routes (exact B x B)

Each is one batched GEMM (or unfolded GEMM) per layer in the factor dtype.
The registered response geometry is  Q(z) = sum_c w_c l_c(z)^2 + rho ||z||^2,
i.e. ||B z||^2 with B = diag(sqrt(w)) J; the exact Gram gives  B B^T  and hence
the exact spectral norm  ||B||_2^2 = lambda_max(B B^T)  used for step
certificates (Chambolle–Pock requires the spectral, not Frobenius, norm).

The signed native-fiber work is  F(z) = sum_c w_c sum_f pi_f |cell[c,f]|  with
the fiber counting measure  pi_f = 1 / (number of cuts)  (stored context
weights over cut count).

Sign conventions: positive ``z`` is removal, the child is
``m - z``; ``J_c[j] = -d r_c / d m_j`` so ``l_c(z) = sum_j J_c[j] z_j`` is the
first-order predicted change of the raw target logit; ``e_c[f,j] = -G phi c_j``
is the conserving positive-removal work of route ``j``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .alexnet_factors import Factors, LayerFactors, RouteAxis


def _masked_weight(layer: LayerFactors, z_layer: torch.Tensor | None) -> torch.Tensor:
    wm = layer.W * layer.mask
    return wm if z_layer is None else wm * z_layer


def _cot_x(layer: LayerFactors, cot: torch.Tensor) -> torch.Tensor:
    """``V[f,j] = sum_{c,p} cot[c,f,p] X[c,j,p]`` — the summed outer product."""

    return torch.einsum("bfp,bjp->fj", cot, layer.X)


def _contract_drive(layer: LayerFactors, matrix: torch.Tensor, cotangent: torch.Tensor,
                    *, workspace_bytes: int = 256 << 20) -> torch.Tensor:
    """Contract the drive before retaining it, using bounded context tiles.

    einsum may copy its permuted input. Budget for input, output and their
    contraction copies, rather than materializing a full-chart [B,F,P] drive.
    This bound excludes retained factors, the route matrix and backend GEMM
    workspace. It is not a bound on whole-process peak memory.
    """
    if type(workspace_bytes) is not int or workspace_bytes <= 0:
        raise ValueError("positive workspace_bytes required")
    per_context = 2 * layer.X.element_size() * layer.X.shape[-1] * (
        layer.X.shape[1] + matrix.shape[0])
    if per_context > workspace_bytes:
        raise ValueError("one context exceeds contraction workspace; increase the explicit budget")
    chunk = max(1, workspace_bytes // max(1, per_context))
    result = torch.empty((layer.X.shape[0], matrix.shape[0]), device=matrix.device, dtype=matrix.dtype)
    for start in range(0, layer.X.shape[0], chunk):
        stop = min(start + chunk, layer.X.shape[0])
        drive = torch.einsum("fj,bjp->bfp", matrix, layer.X[start:stop])
        result[start:stop] = torch.einsum("bfp,bfp->bf", cotangent[start:stop], drive)
    return result


@dataclass
class ChartOperators:
    """Operators of one chart (one mask, one set of contexts)."""

    factors: Factors
    axis: RouteAxis
    context_weights: torch.Tensor  # [B], sums to one over the balanced strata
    cut_count: int

    @classmethod
    def build(cls, factors: Factors, axis: RouteAxis, context_weights: torch.Tensor) -> "ChartOperators":
        w = context_weights.to(factors.device, factors.dtype).reshape(-1)
        if w.numel() != factors.context_count or not torch.isfinite(w).all() or bool((w < 0).any()):
            raise ValueError("context weights must be finite, nonnegative and one per context")
        # Nonnegative weights rounded to the factor dtype can differ in total
        # mass by half that dtype's epsilon. Accumulate in FP64 so reduction
        # rounding does not spuriously refuse valid balanced FP32 weights.
        # This validates their realization; it does not renormalize weights or
        # change the physical radius. Preserve the prior FP64 tolerance.
        mass_tolerance = max(1e-9, 0.5 * torch.finfo(w.dtype).eps)
        if abs(float(w.sum(dtype=torch.float64)) - 1.0) > mass_tolerance:
            raise ValueError("context weights must sum to one")
        return cls(factors=factors, axis=axis, context_weights=w, cut_count=len(factors.layers))

    # -- shapes ------------------------------------------------------------------
    @property
    def route_count(self) -> int:
        return self.axis.route_count

    @property
    def fiber_count(self) -> int:
        return self.axis.fiber_count

    @property
    def context_count(self) -> int:
        return self.factors.context_count

    @property
    def device(self) -> torch.device:
        return self.factors.device

    @property
    def dtype(self) -> torch.dtype:
        return self.factors.dtype

    def eligible(self) -> torch.Tensor:
        return self.axis.concat([layer.mask for layer in self.factors.layers])

    def fiber_measure(self) -> torch.Tensor:
        """``pi_f`` per fiber: 1/cut_count for every fiber of every cut."""

        return torch.full((self.fiber_count,), 1.0 / self.cut_count, device=self.device, dtype=self.dtype)

    def fiber_of_route(self) -> torch.Tensor:
        """Global fiber index of every route, ``[N]`` int64."""

        pieces, offset = [], 0
        for layer in self.axis.layers:
            idx = torch.arange(layer.fiber_count, device=self.device, dtype=torch.int64) + offset
            pieces.append(idx.repeat_interleave(layer.routes_per_fiber))
            offset += layer.fiber_count
        return torch.cat(pieces)

    # -- price vectors -------------------------------------------------------------
    def mean_price(self) -> torch.Tensor:
        """Balanced mean corrected row  g = sum_c w_c e_c  as a flat route vector ``[N]``."""

        pieces = []
        w = self.context_weights.reshape(-1, 1, 1)
        for layer in self.factors.layers:
            pieces.append(-(layer.W * layer.mask) * _cot_x(layer, w * layer.Gt))
        return self.axis.concat(pieces)

    def singleton_price_magnitude(self, *, context_chunk_bytes: int = 1 << 30) -> torch.Tensor:
        """``A_j = E|e_j|``, the safe singleton screen, chunked over contexts."""

        pieces = []
        w = self.context_weights
        for layer in self.factors.layers:
            per_context_bytes = layer.geometry.route_count * layer.X.element_size()
            chunk = max(1, context_chunk_bytes // max(1, per_context_bytes))
            acc = torch.zeros((layer.geometry.fiber_count, layer.geometry.routes_per_fiber),
                              device=self.device, dtype=self.dtype)
            for start in range(0, self.context_count, chunk):
                stop = min(self.context_count, start + chunk)
                rows = torch.einsum("bfp,bjp->bfj", layer.Gt[start:stop], layer.X[start:stop]).abs()
                acc += torch.einsum("b,bfj->fj", w[start:stop], rows)
                del rows
            pieces.append((layer.W * layer.mask).abs() * acc)
        return self.axis.concat(pieces)

    # -- corrected (A) operator -------------------------------------------------------
    def a_forward(self, z: torch.Tensor) -> torch.Tensor:
        """``cell[c,f] = sum_j e_c[f,j] z_j`` for flat ``z`` in ``[N]``; returns ``[B, F_total]``."""

        cells = []
        for layer, zl in zip(self.factors.layers, self.axis.split(z), strict=True):
            cells.append(-_contract_drive(layer, _masked_weight(layer, zl), layer.Gt))
        return torch.cat(cells, dim=1)

    def a_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """``out_j = sum_{c,f} y[c,f] e_c[f,j]`` for ``y`` in ``[B, F_total]``; returns ``[N]``."""

        pieces, offset = [], 0
        for layer in self.factors.layers:
            yl = y[:, offset:offset + layer.geometry.fiber_count]
            offset += layer.geometry.fiber_count
            pieces.append(-(layer.W * layer.mask) * _cot_x(layer, yl.unsqueeze(-1) * layer.Gt))
        return self.axis.concat(pieces)

    def a_fiber_block_norms(self) -> torch.Tensor:
        """Per-fiber Frobenius norms ``||A_f||_F`` over (contexts) x (routes of fiber f), ``[F_total]``.

        Routes are partitioned by destination fiber, so ``A`` is block-diagonal across fibers and
        ``||A||_2 = max_f ||A_f||_2 <= max_f ||A_f||_F``.  This is the spectral certificate for the
        work operator.
        """

        norms = []
        for layer in self.factors.layers:
            W2 = (layer.W * layer.mask) ** 2  # [F, S]
            if layer.X.shape[-1] == 1:
                Gt2 = layer.Gt[..., 0] ** 2                  # [B, F]
                X2 = layer.X[..., 0] ** 2                    # [B, S]
                norms.append(torch.einsum("bf,bf->f", Gt2, X2 @ W2.T).clamp_min(0).sqrt())
            else:
                V = torch.einsum("bfp,bjp->bfj", layer.Gt, layer.X)  # [B, F, S] (small for conv)
                norms.append(torch.einsum("bfj,fj->f", V ** 2, W2).clamp_min(0).sqrt())
        return torch.cat(norms)

    # -- literal-J (response) operator ---------------------------------------------
    def j_forward(self, z: torch.Tensor) -> torch.Tensor:
        """``l[c] = sum_j J_c[j] z_j`` = first-order predicted raw target-logit change, ``[B]``."""

        total = torch.zeros(self.context_count, device=self.device, dtype=self.dtype)
        for layer, zl in zip(self.factors.layers, self.axis.split(z), strict=True):
            total = total - _contract_drive(layer, _masked_weight(layer, zl), layer.D).sum(dim=1)
        return total

    def a_j_forward(self, z: torch.Tensor, *, workspace_bytes: int = 256 << 20) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay corrected cells and literal response from one drive per tile.

        This shares only ``(W m Z) X``. The two cotangent contractions and
        layer/fiber reductions retain the separate forwards' arithmetic order.
        The workspace bound has the same exclusions as ``_contract_drive``;
        retained outputs additionally include two context-by-fiber buffers for
        the current layer. No drive is retained between layers or actions.
        """
        if type(workspace_bytes) is not int or workspace_bytes <= 0:
            raise ValueError("positive workspace_bytes required")
        cells = []
        total = torch.zeros(self.context_count, device=self.device, dtype=self.dtype)
        for layer, zl in zip(self.factors.layers, self.axis.split(z), strict=True):
            matrix = _masked_weight(layer, zl)
            per_context = 2 * layer.X.element_size() * layer.X.shape[-1] * (
                layer.X.shape[1] + matrix.shape[0])
            if per_context > workspace_bytes:
                raise ValueError("one context exceeds contraction workspace; increase the explicit budget")
            chunk = max(1, workspace_bytes // max(1, per_context))
            corrected = torch.empty((self.context_count, matrix.shape[0]), device=self.device, dtype=self.dtype)
            literal = torch.empty_like(corrected)
            for start in range(0, self.context_count, chunk):
                stop = min(start + chunk, self.context_count)
                drive = torch.einsum("fj,bjp->bfp", matrix, layer.X[start:stop])
                corrected[start:stop] = torch.einsum("bfp,bfp->bf", layer.Gt[start:stop], drive)
                literal[start:stop] = torch.einsum("bfp,bfp->bf", layer.D[start:stop], drive)
                del drive
            cells.append(-corrected)
            total = total - literal.sum(dim=1)
        return torch.cat(cells, dim=1), total

    def j_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """``out_j = sum_c y_c J_c[j]`` for ``y`` in ``[B]``; returns ``[N]``."""

        pieces = []
        yb = y.reshape(-1, 1, 1)
        for layer in self.factors.layers:
            pieces.append(-(layer.W * layer.mask) * _cot_x(layer, yb * layer.D))
        return self.axis.concat(pieces)

    def b_forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.context_weights.sqrt() * self.j_forward(z)

    def b_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.j_adjoint(self.context_weights.sqrt() * y)

    def response_gram(self, *, workspace_bytes: int = 256 << 20) -> torch.Tensor:
        """Unweighted ``J J^T`` via bounded literal-J route tiles in the factor dtype.

        The explicit tensor budget includes the output and symmetrization;
        resident factors and vendor BLAS workspace are separate.
        """
        from .response_gram import tiled_response_gram
        return tiled_response_gram(self, workspace_bytes=workspace_bytes)

    def response_gram_adjoint_forward(self) -> torch.Tensor:
        """Original B adjoint/forward pairs; an explicit unbounded-memory reference."""

        B = self.context_count
        gram = torch.zeros((B, B), device=self.device, dtype=self.dtype)
        basis = torch.zeros(B, device=self.device, dtype=self.dtype)
        for c in range(B):
            basis.zero_(); basis[c] = 1.0
            row = self.j_adjoint(basis)          # J_c as a flat route vector (masked)
            gram[c] = self.j_forward(row)        # sum_j J_d[j] J_c[j] for every d
        return 0.5 * (gram + gram.T)

    def response_spectral_norm_squared(self, rho: float = 0.0, gram: torch.Tensor | None = None) -> float:
        """Exact ``||[B; sqrt(rho) I]||_2^2 = lambda_max(diag(sqrt w) J J^T diag(sqrt w)) + rho``."""

        g = self.response_gram() if gram is None else gram
        sw = self.context_weights.sqrt()
        M = sw[:, None] * g * sw[None, :]
        lam = torch.linalg.eigvalsh(M)[-1].item()
        return float(max(lam, 0.0) + rho)

    # -- signed native-fiber work ----------------------------------------------------
    def work_weights(self) -> torch.Tensor:
        """``omega[c,f] = w_c * pi_f`` — the weights of the L1 work objective, ``[B, F_total]``."""

        return self.context_weights[:, None] * self.fiber_measure()[None, :]

    def signed_work(self, z: torch.Tensor) -> torch.Tensor:
        cells = self.a_forward(z)
        return (self.work_weights() * cells.abs()).sum()

    def response_energy(self, z: torch.Tensor, rho: float = 0.0) -> torch.Tensor:
        l = self.j_forward(z)
        return (self.context_weights * l * l).sum() + rho * (z * z).sum()

    def predicted_target_logit_change(self, z: torch.Tensor) -> torch.Tensor:
        return self.j_forward(z)


def dense_reference_rows(factors: Factors, stream: str) -> torch.Tensor:
    """Materialize all rows ``[B, N]`` — for small layers / tests only."""

    return torch.cat([layer.dense_rows(stream).reshape(factors.context_count, -1)
                      for layer in factors.layers], dim=1)


# --------------------------------------------------------------------------------------------
# Loss-weighted (Fisher) quantities for the canonical homotopy controller
# --------------------------------------------------------------------------------------------

def expected_damage_price(ops: ChartOperators, g: torch.Tensor) -> torch.Tensor:
    """Signed first-order balanced-BCE change per unit removal of every route: ``d_j = sum_c w_c g_c sum_f e_c[f,j]``.

    Negative values mean removing the route is predicted to REDUCE the loss (sign-aware pruning).
    One adjoint application: ``y[c, f] = w_c g_c`` for every fiber ``f``.
    """

    y = (ops.context_weights * g.to(ops.dtype))[:, None].expand(ops.context_count, ops.fiber_count).contiguous()
    return ops.a_adjoint(y)


def response_diagonal(ops: ChartOperators, weights: torch.Tensor) -> torch.Tensor:
    """Per-route self term ``sum_c weights_c J_c[j]^2`` (the diagonal of the weighted response Gram), ``[N]``."""

    pieces = []
    wt = weights.to(ops.dtype)
    for layer in ops.factors.layers:
        W2 = (layer.W * layer.mask) ** 2
        if layer.X.shape[-1] == 1:
            D2 = layer.D[..., 0] ** 2 * wt[:, None]          # [B, F]
            X2 = layer.X[..., 0] ** 2                        # [B, S]
            pieces.append(W2 * torch.einsum("bf,bj->fj", D2, X2))
        else:
            V = torch.einsum("bfp,bjp->bfj", layer.D, layer.X)  # [B, F, S]
            pieces.append(W2 * torch.einsum("b,bfj->fj", wt, V ** 2))
    return ops.axis.concat(pieces)


def with_context_weights(ops: ChartOperators, weights: torch.Tensor) -> tuple[ChartOperators, float]:
    """A view of the same factors under new nonnegative context weights (normalized) and the total mass."""

    w = weights.to(ops.device, ops.dtype).clamp_min(0.0)
    total = float(w.sum().item())
    if total <= 0:
        raise ValueError("weights must have positive total mass")
    return ChartOperators(factors=ops.factors, axis=ops.axis, context_weights=w / total, cut_count=ops.cut_count), total
