"""Chart solvers of the tractable v1 lane: response geometry, Phase A, least work, rounding.

Response geometry.  The exact sample second moment of the literal-J rows is
``S = sum_c w_c J_c^T J_c`` (rank <= B).  A population estimate is required for
unseen contexts; the natural one is Ledoit–Wolf shrinkage toward the scaled
identity, ``S_hat = (1 - alpha) S + alpha mu I`` with ``mu = tr(S)/N``, whose
intensity ``alpha`` is a closed form of the B x B Gram (see
:func:`ledoit_wolf_response_geometry`).  The response energy of the chart is

    Q(z) = (1 - alpha) ||B z||^2 + alpha mu ||z||^2 = ||B' z||^2 + rho ||z||^2

with ``rho`` supplied by a
principled estimator instead of a sketch residual.

Phase A (resource frontier).  ``max 1^T z  s.t.  Q(z) <= tau_Q,  0 <= z <= 1``
on eligible routes.  Its Lagrange dual is (B + 1)-dimensional:

    D(nu, lam) = max_{z in box} [ 1^T z - nu^T B' z - lam rho ||z||^2 ]
                 + ||nu||^2 / (4 lam) + lam tau_Q,

jointly convex in ``(nu, lam > 0)``, with the separable maximizer
``z_j = clip((1 - (B'^T nu)_j) / (2 lam rho), 0, 1)``.  Each evaluation costs
one adjoint and one forward; L-BFGS-B converges in tens of evaluations.  In exact arithmetic, weak
duality makes ``D(nu, lam)`` an upper bound on the relaxed frontier. The current
floating-point calculation is a numerical bound estimate, not an outward
certificate for the registered operators (a 4096-iteration PDHG bound stays
at the trivial route count).

Least work.  ``min sum_{c,f} omega_cf |(A z)_cf|  s.t.  1^T z >= R,  Q(z) <= tau_Q,
0 <= z <= 1`` — the registered signed native-fiber work at a released-resource
level, solved by Chambolle–Pock PDHG with spectral-norm step
certificates (block-diagonal fiber bound for ``A``, exact ``lambda_max`` for
``B'``) and a floating-point primal–dual gap estimate. Current-problem domain and
outward numerical qualification are separate; the retained ``certified`` status
string alone does not establish them.

Rounding.  Relaxed solutions are rounded along the registered ladder key
``(-z, work per resource, index)`` with one-live-survivor legality per fiber;
replay of every rung is the controller's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

import numpy as np
import torch

from .operators import ChartOperators


# --------------------------------------------------------------------------------------
# response geometry
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ResponseGeometry:
    """``Q(z) = scale * ||B z||^2 + rho * ||z||^2`` with ``B = diag(sqrt w) J``."""

    scale: float
    rho: float
    alpha: float
    mu: float
    trace: float
    spectral_norm_squared: float  # ||[sqrt(scale) B ; sqrt(rho) I]||_2^2 = scale*lambda_max + rho
    gram: torch.Tensor            # exact J J^T, [B, B]
    law: str

    def energy(self, ops: ChartOperators, z: torch.Tensor) -> torch.Tensor:
        return self.scale * (ops.b_forward(z) ** 2).sum() + self.rho * (z * z).sum()

    def to_json(self) -> dict[str, Any]:
        return dict(schema="tractable_v1_response_geometry_v1", law=self.law, scale=self.scale, rho=self.rho,
                    alpha=self.alpha, mu=self.mu, trace=self.trace,
                    spectral_norm_squared=self.spectral_norm_squared, gram_shape=list(self.gram.shape))


def ledoit_wolf_response_geometry(ops: ChartOperators, *, route_count_for_mu: int | None = None,
                                  shrink: bool = True) -> ResponseGeometry:
    """Exact Ledoit–Wolf shrinkage of the weighted sample second moment from the B x B Gram.

    With ``x_c = sqrt(w_c) J_c`` rows (so ``S = sum_c x_c x_c^T`` has ``tr S = sum_c w_c G_cc``):
      mu      = tr(S) / N
      ||S||_F^2 = sum_{c,d} w_c w_d G_cd^2
      beta    = sum_c w_c^2 ( G_cc^2 - 2 sum_d w_d G_cd^2 + ||S||_F^2 )   (sum of ||x_c x_c^T - S||_F^2 weighted)
      delta   = ||S - mu I||_F^2 = ||S||_F^2 - 2 mu tr S + mu^2 N
      alpha   = clip(beta / delta, 0, 1)
    Then ``Q(z) = (1 - alpha) ||B z||^2 + alpha mu ||z||^2``.
    """

    gram = ops.response_gram()
    w = ops.context_weights
    eligible_count = float(ops.eligible().sum().item()) if route_count_for_mu is None else float(route_count_for_mu)
    N_eff = max(1.0, eligible_count)
    G2 = gram ** 2
    trace = float((w * torch.diagonal(gram)).sum().item())
    s_frob2 = float((w[:, None] * w[None, :] * G2).sum().item())
    mu = trace / N_eff
    if shrink:
        per_context = torch.diagonal(gram) ** 2 - 2.0 * (G2 * w[None, :]).sum(dim=1) + s_frob2
        beta = float((w * w * per_context).sum().item())
        delta = s_frob2 - 2.0 * mu * trace + mu * mu * N_eff
        alpha = 0.0 if delta <= 0 else min(1.0, max(0.0, beta / delta))
    else:
        alpha = 0.0
    scale = 1.0 - alpha
    rho = alpha * mu
    lam = ops.response_spectral_norm_squared(rho=0.0, gram=gram)
    return ResponseGeometry(scale=scale, rho=rho, alpha=alpha, mu=mu, trace=trace,
                            spectral_norm_squared=scale * lam + rho, gram=gram,
                            law="ledoit_wolf_shrunk_sample_second_moment_v1" if shrink else "raw_sample_second_moment_v1")


# --------------------------------------------------------------------------------------
# Phase A: exact low-dimensional dual
# --------------------------------------------------------------------------------------

@dataclass
class PhaseAResult:
    z: torch.Tensor
    resource: float
    resource_upper_bound: float
    gap: float
    relative_gap: float
    energy: float
    q_budget: float
    lam: float
    nu: torch.Tensor
    evaluations: int
    converged: bool
    seconds: float
    status: str

    def to_json(self) -> dict[str, Any]:
        return dict(schema="tractable_v1_phase_a_result_v1", resource=self.resource,
                    resource_upper_bound=self.resource_upper_bound, gap=self.gap, relative_gap=self.relative_gap,
                    energy=self.energy, q_budget=self.q_budget, lam=self.lam, evaluations=self.evaluations,
                    converged=self.converged, seconds=self.seconds, status=self.status)


def _phase_a_primal(ops: ChartOperators, geom: ResponseGeometry, eligible: torch.Tensor,
                    nu: torch.Tensor, lam: float) -> torch.Tensor:
    """Separable maximizer ``z(nu, lam)`` of the Phase-A Lagrangian."""

    bt_nu = math.sqrt(geom.scale) * ops.b_adjoint(nu) if geom.scale > 0 else torch.zeros_like(eligible)
    denom = 2.0 * lam * geom.rho
    if denom <= 0:
        # rho = 0: the Lagrangian is linear in z on the box -> bang-bang
        z = (1.0 - bt_nu > 0).to(eligible.dtype)
    else:
        z = ((1.0 - bt_nu) / denom).clamp_(0.0, 1.0)
    return z * eligible


def solve_phase_a(ops: ChartOperators, geom: ResponseGeometry, *, q_budget: float,
                  eligible: torch.Tensor | None = None, max_evaluations: int = 400,
                  gap_tolerance: float = 1e-6, nu0: torch.Tensor | None = None, lam0: float | None = None) -> PhaseAResult:
    """Maximize removed resource under the response trust radius via the exact dual."""

    from scipy.optimize import minimize

    started = time.perf_counter()
    elig = ops.eligible() if eligible is None else eligible.to(ops.device, ops.dtype)
    B = ops.context_count
    tau = float(q_budget)
    if tau <= 0:
        raise ValueError("q_budget must be positive")
    if geom.rho <= 0:
        raise ValueError("Phase A needs a positive rho (shrinkage); the raw sample geometry is degenerate on the null space")
    n_elig = float(elig.sum().item())
    if n_elig == 0:
        z = torch.zeros_like(elig)
        return PhaseAResult(z, 0.0, 0.0, 0.0, 0.0, 0.0, tau, 0.0, torch.zeros(B, device=ops.device, dtype=ops.dtype),
                            0, True, time.perf_counter() - started, "no_eligible_routes")

    evaluations = 0
    cache: dict[str, Any] = {}

    def dual(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        nu = torch.as_tensor(theta[:B], device=ops.device, dtype=ops.dtype)
        lam = float(theta[B])
        z = _phase_a_primal(ops, geom, elig, nu, lam)
        bz = math.sqrt(geom.scale) * ops.b_forward(z) if geom.scale > 0 else torch.zeros(B, device=ops.device, dtype=ops.dtype)
        zz = float((z * z).sum().item())
        ones = float(z.sum().item())
        value = ones - float((nu * bz).sum().item()) - lam * geom.rho * zz + float((nu * nu).sum().item()) / (4.0 * lam) + lam * tau
        g_nu = (-bz + nu / (2.0 * lam)).detach().cpu().numpy().astype(np.float64)
        g_lam = -geom.rho * zz - float((nu * nu).sum().item()) / (4.0 * lam * lam) + tau
        cache.update(z=z, value=value, ones=ones)
        return float(value), np.concatenate([g_nu, [g_lam]])

    # initial lambda: make the all-eligible removal hit the radius roughly, lam ~ 1 / (2 rho) * ...
    if lam0 is None:
        # at nu = 0, z = clip(1/(2 lam rho)) -> choose lam so that rho ||z||^2 ~ tau with z = 1 on eligible
        lam_init = max(1e-8, 1.0 / (2.0 * geom.rho)) if geom.rho * n_elig <= tau else max(1e-8, math.sqrt(n_elig / tau) / (2.0 * math.sqrt(geom.rho)))
    else:
        lam_init = float(lam0)
    theta0 = np.zeros(B + 1)
    if nu0 is not None:
        theta0[:B] = nu0.detach().cpu().numpy()
    theta0[B] = lam_init
    bounds = [(None, None)] * B + [(1e-12, None)]
    res = minimize(dual, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=max_evaluations, maxfun=max_evaluations * 2, ftol=1e-15, gtol=1e-10))
    nu = torch.as_tensor(res.x[:B], device=ops.device, dtype=ops.dtype)
    lam = float(res.x[B])
    z = _phase_a_primal(ops, geom, elig, nu, lam)
    energy = float(geom.energy(ops, z).item())
    # enforce feasibility exactly by scaling (keeps the box)
    if energy > tau:
        z = z * math.sqrt(tau / energy)
        energy = float(geom.energy(ops, z).item())
    resource = float(z.sum().item())
    upper = float(res.fun)
    gap = max(0.0, upper - resource)
    rel = gap / max(1.0, resource)
    converged = bool(res.success) and rel <= gap_tolerance * 10 or rel <= gap_tolerance
    return PhaseAResult(z=z, resource=resource, resource_upper_bound=upper, gap=gap, relative_gap=rel,
                        energy=energy, q_budget=tau, lam=lam, nu=nu, evaluations=evaluations,
                        converged=converged, seconds=time.perf_counter() - started,
                        status=("certified" if rel <= gap_tolerance else "bounded_not_converged") + ":" + str(res.message))


# --------------------------------------------------------------------------------------
# Least work: PDHG with spectral certificates
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PDHGPolicy:
    max_iterations: int = 3000
    check_interval: int = 50
    relative_gap_tolerance: float = 1e-4
    absolute_gap_tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-6
    step_safety: float = 0.99
    primal_dual_balance: float = 1.0   # tau_p = step * balance, sigma = step / balance
    over_relaxation: float = 1.0

    def to_json(self) -> dict[str, Any]:
        return dict(schema="tractable_v1_pdhg_policy_v1", **{k: getattr(self, k) for k in self.__dataclass_fields__})


@dataclass
class LeastWorkResult:
    z: torch.Tensor
    work: float
    work_lower_bound: float
    gap: float
    relative_gap: float
    resource: float
    resource_floor: float
    energy: float
    q_budget: float
    iterations: int
    converged: bool
    seconds: float
    operator_norm_squared_upper: float
    status: str
    history: list[dict[str, float]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return dict(schema="tractable_v1_least_work_result_v1", work=self.work, work_lower_bound=self.work_lower_bound,
                    gap=self.gap, relative_gap=self.relative_gap, resource=self.resource, resource_floor=self.resource_floor,
                    energy=self.energy, q_budget=self.q_budget, iterations=self.iterations, converged=self.converged,
                    seconds=self.seconds, operator_norm_squared_upper=self.operator_norm_squared_upper,
                    status=self.status, history=self.history)


def solve_least_work(ops: ChartOperators, geom: ResponseGeometry, *, q_budget: float, resource_floor: float,
                     z0: torch.Tensor, eligible: torch.Tensor | None = None,
                     policy: PDHGPolicy = PDHGPolicy()) -> LeastWorkResult:
    """Minimize registered signed native-fiber work at a released-resource floor inside the trust radius.

    PDHG on  min_z  iota_box(z) + h(K z),  K = [A; sqrt(scale) B; sqrt(rho) I; 1^T/sqrt(N)],
    h(a, b, r, s) = sum omega |a| + iota_{||(b, r)|| <= sqrt(tau)} + iota_{s >= R/sqrt(N)}.
    """

    started = time.perf_counter()
    elig = ops.eligible() if eligible is None else eligible.to(ops.device, ops.dtype)
    N = float(elig.sum().item())
    if N <= 0:
        raise ValueError("no eligible routes")
    tau = float(q_budget)
    sq_tau = math.sqrt(tau)
    R = float(resource_floor)
    scale_s = math.sqrt(max(geom.scale, 0.0))
    scale_r = math.sqrt(max(geom.rho, 0.0))
    inv_sqrt_n = 1.0 / math.sqrt(N)
    omega = ops.work_weights()                         # [B, F]
    a_norm2 = float(ops.a_fiber_block_norms().max().item() ** 2)
    k_norm2 = a_norm2 + geom.spectral_norm_squared + (N * inv_sqrt_n * inv_sqrt_n)  # last term = 1
    step = policy.step_safety / math.sqrt(k_norm2)
    tau_p = step * policy.primal_dual_balance
    sigma = step / policy.primal_dual_balance

    z = (z0.to(ops.device, ops.dtype) * elig).clamp(0.0, 1.0)
    z_bar = z.clone()
    y_a = torch.zeros_like(omega)
    y_b = torch.zeros(ops.context_count, device=ops.device, dtype=ops.dtype)
    y_r = torch.zeros_like(z) if scale_r > 0 else None
    y_s = torch.zeros((), device=ops.device, dtype=ops.dtype)
    best: dict[str, Any] = dict(work=math.inf, z=None, lower=-math.inf)
    history: list[dict[str, float]] = []
    iterations = 0
    converged = False

    def dual_bound(y_a_, y_b_, y_r_, y_s_) -> float:
        """Weak-duality lower bound on the work: -f*(-K^T y) - h*(y) = -sum_j max(-(K^T y)_j, 0) - sqrt(tau)||y_q|| - (R/sqrt N) y_s."""
        kty = ops.a_adjoint(y_a_) + scale_s * ops.b_adjoint(y_b_) + y_s_ * inv_sqrt_n
        if y_r_ is not None:
            kty = kty + scale_r * y_r_
        kty = kty * elig
        f_star = torch.clamp(-kty, min=0.0).sum().item()
        yq = float(torch.sqrt((y_b_ * y_b_).sum() + ((y_r_ * y_r_).sum() if y_r_ is not None else 0.0)).item())
        h_star = sq_tau * yq + (R * inv_sqrt_n) * float(y_s_.item())
        return -f_star - h_star

    for iteration in range(1, policy.max_iterations + 1):
        iterations = iteration
        # dual ascent on K z_bar
        cells = ops.a_forward(z_bar)
        y_a = torch.clamp(y_a + sigma * cells, min=-omega, max=omega)
        b_bar = scale_s * ops.b_forward(z_bar)
        yb_c = y_b + sigma * b_bar
        if y_r is not None:
            yr_c = y_r + sigma * scale_r * z_bar
            q_norm = torch.sqrt((yb_c * yb_c).sum() + (yr_c * yr_c).sum())
        else:
            yr_c = None
            q_norm = torch.sqrt((yb_c * yb_c).sum())
        shrink = torch.clamp(1.0 - sigma * sq_tau / torch.clamp(q_norm, min=1e-300), min=0.0)
        y_b = yb_c * shrink
        if yr_c is not None:
            y_r = yr_c * shrink
        y_s = torch.clamp(y_s + sigma * (z_bar.sum() * inv_sqrt_n - R * inv_sqrt_n), max=0.0)
        # primal descent
        grad = ops.a_adjoint(y_a) + scale_s * ops.b_adjoint(y_b) + y_s * inv_sqrt_n
        if y_r is not None:
            grad = grad + scale_r * y_r
        z_new = torch.clamp(z - tau_p * grad, 0.0, 1.0) * elig
        z_bar = z_new + policy.over_relaxation * (z_new - z)
        z = z_new
        if iteration % policy.check_interval == 0 or iteration == policy.max_iterations:
            energy = float(geom.energy(ops, z).item())
            resource = float(z.sum().item())
            work = float((omega * ops.a_forward(z).abs()).sum().item())
            lower = dual_bound(y_a, y_b, y_r, y_s)
            feasible = energy <= tau * (1.0 + policy.feasibility_tolerance) and resource >= R * (1.0 - policy.feasibility_tolerance) - 1e-9
            history.append(dict(iteration=iteration, work=work, lower=lower, energy=energy, resource=resource))
            if feasible and work < best["work"]:
                best.update(work=work, z=z.clone(), lower=lower, energy=energy, resource=resource)
            best["lower"] = max(best["lower"], lower)
            gap = best["work"] - best["lower"] if best["z"] is not None else math.inf
            if best["z"] is not None and gap <= policy.absolute_gap_tolerance + policy.relative_gap_tolerance * max(1.0, abs(best["work"])):
                converged = True
                break
    if best["z"] is None:
        # return the final iterate projected to feasibility by scaling toward the floor-feasible point
        z_final = z
        energy = float(geom.energy(ops, z_final).item())
        if energy > tau:
            z_final = z_final * math.sqrt(tau / energy)
            energy = float(geom.energy(ops, z_final).item())
        best.update(work=float((omega * ops.a_forward(z_final).abs()).sum().item()), z=z_final, energy=energy,
                    resource=float(z_final.sum().item()))
        status = "unresolved_infeasible_iterate"
    else:
        status = "certified" if converged else "bounded_not_converged"
    gap = max(0.0, best["work"] - best["lower"]) if math.isfinite(best["lower"]) else math.inf
    return LeastWorkResult(z=best["z"], work=best["work"], work_lower_bound=best["lower"], gap=gap,
                           relative_gap=gap / max(1.0, abs(best["work"])), resource=best["resource"],
                           resource_floor=R, energy=best["energy"], q_budget=tau, iterations=iterations,
                           converged=converged, seconds=time.perf_counter() - started,
                           operator_norm_squared_upper=k_norm2, status=status, history=history)


# --------------------------------------------------------------------------------------
# rounding ladder with one-live-survivor legality
# --------------------------------------------------------------------------------------

@dataclass
class RoundingLadder:
    order: torch.Tensor          # eligible routes ordered for removal (int64), legality already enforced
    resource_rungs: tuple[int, ...]

    def mask_for_rung(self, mask: torch.Tensor, resource: int) -> torch.Tensor:
        child = mask.clone()
        child[self.order[:resource]] = 0.0
        return child

    def removal_vector(self, resource: int, route_count: int) -> torch.Tensor:
        z = torch.zeros(route_count, device=self.order.device, dtype=torch.float64)
        z[self.order[:resource]] = 1.0
        return z


def build_rounding_ladder(z: torch.Tensor, *, price: torch.Tensor, eligible: torch.Tensor,
                          fiber_of_route: torch.Tensor, resource_target: int,
                          ladder_factor: float = 0.5, minimum_resource: int = 1,
                          z_quantum: float = 1e-6, signed_price: bool = False) -> RoundingLadder:
    """Order eligible routes by ``(-z, |price|, index)`` and enforce one live survivor per fiber.

    ``price`` is the mean corrected row (work per unit resource); cheaper routes round first
    among equal ``z``.  The rung list is ``resource_target, resource_target*ladder_factor, ...``
    down to ``minimum_resource`` (scanning continues past a Q violation).
    """

    device = z.device
    elig_idx = torch.nonzero(eligible > 0, as_tuple=False).reshape(-1)
    if elig_idx.numel() == 0:
        return RoundingLadder(order=elig_idx, resource_rungs=())
    zq = torch.round(z[elig_idx] / z_quantum)  # quantize so that ties resolve by price
    key_cost = price[elig_idx] if signed_price else price[elig_idx].abs()
    # lexicographic sort: primary -zq (descending z), secondary cost ascending, tertiary index
    order = torch.argsort(elig_idx, stable=True)
    order = order[torch.argsort(key_cost[order], stable=True)]
    order = order[torch.argsort(-zq[order], stable=True)]
    ordered_routes = elig_idx[order]
    # one-live-survivor legality: the last eligible route of each fiber in this order is protected
    fibers = fiber_of_route[ordered_routes]
    live_per_fiber = torch.bincount(fiber_of_route[elig_idx], minlength=int(fiber_of_route.max().item()) + 1)
    # count how many routes of the same fiber appear before each position; the route is removable iff
    # (position within fiber) < live_per_fiber[fiber] - 1
    fiber_sorted, perm = torch.sort(fibers, stable=True)
    ranks = torch.empty_like(perm)
    counts = torch.bincount(fiber_sorted, minlength=live_per_fiber.numel())
    starts = torch.cumsum(counts, 0) - counts
    ranks[perm] = torch.arange(perm.numel(), device=device) - starts[fiber_sorted]
    removable = ranks < (live_per_fiber[fibers] - 1)
    legal_order = ordered_routes[removable]
    rungs = []
    r = int(min(resource_target, legal_order.numel()))
    while r >= max(1, minimum_resource):
        rungs.append(r)
        nxt = int(math.floor(r * ladder_factor))
        if nxt == r:
            break
        r = nxt
    return RoundingLadder(order=legal_order, resource_rungs=tuple(rungs))


# --------------------------------------------------------------------------------------
# Greedy price-ordered prefix under the exact response radius (binary, no rounding)
# --------------------------------------------------------------------------------------

@dataclass
class GreedyPrefixResult:
    order: torch.Tensor            # legal removal order (int64), cheapest corrected price first
    resource: int                  # routes removed by the returned prefix
    z: torch.Tensor                # binary removal vector [N]
    energy: float                  # exact shrunk response energy of the prefix
    q_budget: float
    work: float                    # registered signed native-fiber work of the prefix
    evaluations: int
    seconds: float
    history: list[dict[str, float]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return dict(schema="tractable_v1_greedy_prefix_result_v1", resource=self.resource, energy=self.energy,
                    q_budget=self.q_budget, work=self.work, evaluations=self.evaluations, seconds=self.seconds,
                    history=self.history[-64:])


def legal_price_order(*, price: torch.Tensor, eligible: torch.Tensor, fiber_of_route: torch.Tensor) -> torch.Tensor:
    """Eligible routes ordered by ``|price|`` ascending (cheapest work per unit resource first), then index,
    with the last live route of every fiber protected (one-live-survivor legality)."""

    ladder = build_rounding_ladder(torch.zeros_like(price), price=price, eligible=eligible,
                                   fiber_of_route=fiber_of_route, resource_target=1)
    return ladder.order


def legal_signed_order(*, score: torch.Tensor, eligible: torch.Tensor, fiber_of_route: torch.Tensor) -> torch.Tensor:
    """Eligible routes ordered by SIGNED score ascending (most loss-reducing removals first), legality kept."""

    return build_rounding_ladder(torch.zeros_like(score), price=score, eligible=eligible, fiber_of_route=fiber_of_route,
                                 resource_target=1, signed_price=True).order


def greedy_prefix_action(ops: ChartOperators, geom: ResponseGeometry, *, order: torch.Tensor, q_budget: float,
                         initial_block: int | None = None, max_evaluations: int = 400) -> GreedyPrefixResult:
    """Largest prefix of ``order`` whose exact energy stays within ``q_budget``, found by doubling/bisection.

    The energy is not monotone along the prefix (cross-route cancellation), so a violated block is bisected
    rather than taken as a stopping proof; the scan stops when no further single route fits or the
    evaluation cap is reached.  Every energy is the exact shrunk response energy of a binary vector.
    """

    started = time.perf_counter()
    N = ops.route_count
    total = int(order.numel())
    z = torch.zeros(N, device=ops.device, dtype=ops.dtype)
    if total == 0:
        return GreedyPrefixResult(order, 0, z, 0.0, float(q_budget), 0.0, 0, time.perf_counter() - started)
    removed = 0
    block = int(initial_block) if initial_block else max(1, total // 8)
    block = max(1, min(block, total))
    evaluations = 0
    history: list[dict[str, float]] = []
    energy = 0.0
    while removed < total and block >= 1 and evaluations < max_evaluations:
        stop = min(total, removed + block)
        trial = z.clone()
        trial[order[removed:stop]] = 1.0
        q = float(geom.energy(ops, trial).item())
        evaluations += 1
        history.append(dict(removed=removed, block=stop - removed, energy=q, accepted=float(q <= q_budget)))
        if q <= q_budget:
            z, removed, energy = trial, stop, q
            block = min(block * 2, total - removed) if total - removed > 0 else 0
        else:
            if block == 1:
                break
            block //= 2
    work = float(ops.signed_work(z).item()) if removed else 0.0
    return GreedyPrefixResult(order=order, resource=removed, z=z, energy=energy, q_budget=float(q_budget), work=work,
                              evaluations=evaluations, seconds=time.perf_counter() - started, history=history)


# --------------------------------------------------------------------------------------
# Diagonal-target shrinkage geometry (canonical): Q(z) = (1-a) ||B z||^2 + a sum_j S_jj z_j^2
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagonalShrunkGeometry:
    """Weighted response second moment shrunk toward its own diagonal.

    ``S = sum_c w_c J_c^T J_c`` (rank <= B) is the observed interference structure; ``diag(S)`` is the
    per-route sensitivity. The estimate ``(1 - alpha) S + alpha diag(S)`` is a PSD
    shrinkage estimator in the registered route basis, not an unconditional
    law-of-total-variance identity or a population response upper bound.  ``alpha`` is the
    Schafer–Strimmer / Ledoit–Wolf intensity for the diagonal target, computed from the B x B Gram and
    the B x B fourth-moment matrix ``M_cd = sum_j x_cj^2 x_dj^2`` (estimated on a sealed route subsample).
    For binary removal vectors ``sum_j S_jj z_j^2 = sum_j S_jj z_j``.
    """

    alpha: float
    weights: torch.Tensor          # context weights used (normalized), [B]
    total_mass: float              # sum of the unnormalized weights (e.g. Fisher mass)
    diagonal: torch.Tensor         # S_jj under the normalized weights, [N]
    gram: torch.Tensor             # J J^T, [B, B] (unweighted)
    fourth: torch.Tensor           # estimated M, [B, B]
    subsample_routes: int
    law: str = "shrink_to_diagonal_schafer_strimmer_v1"
    shrinkage_variance_law: str = "uniform_unbiased_or_kish_diagnostic_v1"

    def energy(self, ops: ChartOperators, z: torch.Tensor) -> torch.Tensor:
        """Unnormalized energy (multiplied by the total mass so Fisher weights give damage units)."""

        return self.energy_from_response(z, ops.j_forward(z))

    def energy_from_response(self, z: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        """Same energy using a caller-owned replay of ``J z`` on this chart.

        This does not validate the provenance of a supplied response. Internal
        endpoint callers obtain it directly from the same frozen operator and
        action, avoiding a second literal forward without changing reductions.
        """
        if response.shape != self.weights.shape or z.shape != self.diagonal.shape:
            raise ValueError("energy response and action must match geometry axes")
        interference = (self.weights * response ** 2).sum()
        independent = (self.diagonal * z * z).sum()
        return self.total_mass * ((1.0 - self.alpha) * interference + self.alpha * independent)

    def cross_term(self, ops: ChartOperators, z: torch.Tensor) -> torch.Tensor:
        """``(1-alpha) * mass * (B^T B z)_j`` — the interference part of the marginal energy of adding route j."""

        return self.total_mass * (1.0 - self.alpha) * ops.j_adjoint(self.weights * ops.j_forward(z))

    def self_term(self) -> torch.Tensor:
        """``mass * S_jj`` — the energy of removing route j alone (both parts sum to S_jj on a singleton)."""

        return self.total_mass * self.diagonal

    def to_json(self) -> dict[str, Any]:
        return dict(schema="tractable_v1_diagonal_shrunk_geometry_v1", law=self.law, alpha=self.alpha,
                    shrinkage_variance_law=self.shrinkage_variance_law,
                    total_mass=self.total_mass, subsample_routes=self.subsample_routes,
                    trace=float((self.weights * torch.diagonal(self.gram)).sum().item()) * self.total_mass,
                    gram_shape=list(self.gram.shape))


def _route_subsample_rows(ops: ChartOperators, routes: torch.Tensor) -> torch.Tensor:
    """Exact literal-J rows restricted to a set of routes: ``x[c, s] = J_c[routes[s]]``, ``[B, n]``."""

    device, dtype = ops.device, ops.dtype
    fiber = ops.fiber_of_route()[routes]
    out = torch.empty((ops.context_count, routes.numel()), device=device, dtype=dtype)
    offset_f = 0
    for layer, (lo, hi) in zip(ops.factors.layers, [(l.route_offset, l.route_offset + l.route_count) for l in ops.axis.layers], strict=True):
        sel = (routes >= lo) & (routes < hi)
        if not bool(sel.any()):
            offset_f += layer.geometry.fiber_count
            continue
        r = routes[sel] - lo
        f_local = r // layer.geometry.routes_per_fiber
        j_local = r % layer.geometry.routes_per_fiber
        wm = (layer.W * layer.mask)[f_local, j_local]                          # [n_l]
        # J_c[f, j] = -wm * sum_p D[c, f, p] X[c, j, p]
        dx = torch.einsum("bnp,bnp->bn", layer.D[:, f_local, :], layer.X[:, j_local, :])
        out[:, sel] = -wm[None, :] * dx
        offset_f += layer.geometry.fiber_count
    del fiber
    return out


def diagonal_shrunk_geometry(ops: ChartOperators, *, weights: torch.Tensor, subsample: int = 1 << 20,
                             seed: int = 0,
                             unweighted_context_gram: torch.Tensor | None = None) -> DiagonalShrunkGeometry:
    """Diagonal-target shrinkage under arbitrary nonnegative context weights (e.g. Fisher weights)."""

    from .operators import response_diagonal, with_context_weights

    view, mass = with_context_weights(ops, weights)
    w = view.context_weights
    if unweighted_context_gram is None:
        gram = view.response_gram()
    else:
        gram = unweighted_context_gram.to(device=view.device, dtype=view.dtype)
        if gram.shape != (view.context_count, view.context_count):
            raise ValueError("unweighted_context_gram must be [context_count, context_count]")
        if not bool(torch.isfinite(gram).all()):
            raise ValueError("unweighted_context_gram must be finite")
    diagonal = response_diagonal(view, w)                                     # S_jj under normalized weights
    eligible = ops.eligible()
    elig_idx = torch.nonzero(eligible > 0, as_tuple=False).reshape(-1)
    n_sub = int(min(subsample, elig_idx.numel()))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    pick = elig_idx[torch.randperm(elig_idx.numel(), generator=gen)[:n_sub].to(elig_idx.device)]
    x = _route_subsample_rows(view, pick) * torch.sqrt(w)[:, None]           # weighted rows on the subsample
    scale = float(elig_idx.numel()) / float(max(1, n_sub))
    x2 = x * x
    fourth = (x2 @ x2.T) * scale                                              # M_cd ~ sum_j x_cj^2 x_dj^2
    # Schafer–Strimmer intensity for the diagonal target with these (already weighted) rows:
    #   alpha = sum_{j != k} Var(S_jk) / sum_{j != k} S_jk^2, with S = sum_c x_c x_c^T (rows already carry sqrt w)
    sw_gram = torch.sqrt(w)[:, None] * gram * torch.sqrt(w)[None, :]          # x_c . x_d
    s_frob2 = float((sw_gram ** 2).sum().item())                              # ||S||_F^2
    diag2 = float(fourth.sum().item())                                        # sum_j S_jj^2  (= sum_cd M_cd)
    off_s = s_frob2 - diag2                                                   # sum_{j != k} S_jk^2
    per_context = torch.diagonal(sw_gram) ** 2 - torch.diagonal(fourth)       # ||x_c||^4 - sum_j x_cj^4 = sum_{j!=k} x_cj^2 x_ck^2
    n_eff = 1.0 / float((w * w).sum().item())                                 # effective sample size (B for equal weights)
    raw_var = n_eff * float(per_context.sum().item()) - off_s
    var_sum = max(0.0, raw_var / max(1.0, n_eff - 1.0))
    uniform = bool(torch.allclose(w, torch.full_like(w, 1.0 / w.numel()), rtol=0.0, atol=1e-12))
    variance_law = "uniform_unbiased_offdiag_mean_variance_v1" if uniform else "kish_effective_sample_offdiag_variance_diagnostic_v1"
    alpha = 0.0 if off_s <= 0 else min(1.0, max(0.0, var_sum / off_s))
    return DiagonalShrunkGeometry(alpha=alpha, weights=w, total_mass=mass, diagonal=diagonal, gram=gram,
                                  fourth=fourth, subsample_routes=n_sub, shrinkage_variance_law=variance_law)
