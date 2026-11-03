"""Indifference-zone census stop with one confidence parameter.

Guarantee at the stop: with probability >= 1 - alpha (alpha = 1 - the walk's
registered confidence level), no route whose true census score differs from
a cut by more than delta = eps * tau_S (tau_S = tau x the run's positive loss
span) lies on the wrong side of that cut, at every executable cut of every
scope's walk ladder.  The stop requires the estimated expected number of such
misplacements,

    E = sum_scopes sum_cuts sum_r F_t,nu(-(|S_r - t_k| + delta) / se_r),     nu = n - 1,

to be at most alpha_j = alpha / ((j+1)(j+2)) at check j (the engine's
telescoping schedule), by Markov's inequality and a union over checks.  se_r
is the within-sample standard error over the n off-target draws; F_t,nu is
the Student-t distribution function.  eps is the one precision knob.
Diagnostics per cut: the score density rho at the cut, the median SE in the
window, the predicted identity and retest miss rates, the bound
(n_zone_below + alpha_j)/k.  (sig: the median SE over the nearest window
block.  A failing check stops summing once E exceeds alpha_j, so its E is a
lower bound; no draws forecast is made.)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

IZ_STOP_RULE = "indifference_zone_student_t_v2"
IZ_DEFAULT_EPS = 0.5
# A route whose standardized distance (|S - t| + delta)/se is at least this is
# skipped: its term F_t,nu(-(|S - t| + delta)/se) is below F_t,nu(-9) < 2e-18 at
# nu >= 2047 (the first check is at 2,048 draws), so all 57M routes together
# change E by less than 1.2e-10, far below any alpha_j the stop compares with.
IZ_WINDOW_SES = 9.0


def alpha_at_check(alpha: float, check: int) -> float:
    return float(alpha) / float((int(check) + 1) * (int(check) + 2))


def _tensor(values: Any) -> "torch.Tensor":
    if torch.is_tensor(values):
        return values.to(dtype=torch.float64).reshape(-1)
    return torch.as_tensor(np.asarray(values, dtype=np.float64)).reshape(-1)


def _cut_terms_outward(
    ascending: "torch.Tensor",
    se_ascending: "torch.Tensor",
    *,
    position: int,
    t: float,
    delta: float,
    nu: float,
    window_ses: float,
    reach: float,
    budget: float,
    block: int = 1 << 20,
) -> tuple[float, bool, int, float]:
    """Sum F_t,nu(-(|S-t|+delta)/se) over the cut's window, outward from the cut in bounded blocks.

    Routes are taken in blocks of at most ``block`` on each side, moving away
    from the cut along the sorted scores, so no block holds more than a few
    times ``block`` values; the sum stops as soon as it exceeds ``budget``
    (terms are nonnegative, so the decision "sum <= budget" is unchanged).
    Returns (sum, exceeded, window routes seen, median SE of the nearest block).
    """

    from scipy.special import stdtr

    n = int(ascending.numel())
    lo = int(torch.searchsorted(ascending, torch.tensor(t - reach, dtype=ascending.dtype, device=ascending.device),
                                right=True).item())
    hi = int(torch.searchsorted(ascending, torch.tensor(t + reach, dtype=ascending.dtype, device=ascending.device),
                                right=False).item())
    left, right = min(max(position, lo), hi), min(max(position, lo), hi)
    total, seen, sig = 0.0, 0, None
    while left > lo or right < hi:
        parts = []
        if right < hi:
            parts.append(slice(right, min(right + block, hi)))
            right = min(right + block, hi)
        if left > lo:
            parts.append(slice(max(left - block, lo), left))
            left = max(left - block, lo)
        for part in parts:
            g = (ascending[part] - t).abs()
            s_part = se_ascending[part]
            keep = ((g + delta) < (window_ses * s_part)) & (s_part > 0.0)
            if not bool(keep.any()):
                continue
            s_keep = s_part[keep]
            if sig is None:
                sig = float(torch.quantile(s_keep, 0.5).item()) if s_keep.numel() <= 16_000_000 else float(s_keep.median().item())
            z = ((g[keep] + delta) / s_keep).detach().cpu().numpy()
            seen += int(z.size)
            total += float(stdtr(float(nu), -z).sum())
            if total > budget:
                return total, (left > lo or right < hi), seen, (sig or 0.0)
    return total, False, seen, (sig or 0.0)


def census_iz_stop(
    scopes: Mapping[str, tuple[Any, Any]],
    cuts: Mapping[str, Sequence[int]],
    *,
    tau_S: float,
    alpha: float,
    look: int,
    n_draws: int,
    eps: float = IZ_DEFAULT_EPS,
    window_ses: float = IZ_WINDOW_SES,
    population_exhausted: bool = False,
    forecast: bool = True,
) -> dict[str, Any]:
    """Evaluate the one-parameter indifference-zone stop over every scope and cut."""

    if not float(tau_S) > 0.0 or not float(eps) >= 0.0 or not 0.0 < float(alpha) < 1.0:
        raise ValueError("census stop needs tau_S > 0, eps >= 0 and alpha in (0, 1)")
    delta = float(eps) * float(tau_S)
    nu = float(int(n_draws) - 1)
    threshold = alpha_at_check(float(alpha), int(look))
    per_scope: dict[str, dict[str, Any]] = {}
    total = 0.0
    # The stop needs only "E <= threshold": once the running sum exceeds it the check has failed, and the
    # remaining terms are not evaluated (early_exit; E is then a lower bound).  A passing check sums everything.
    early_exit = False
    exit_scope: str | None = None
    for scope in sorted(scopes):
        if early_exit:
            break
        est = _tensor(scopes[scope][0])
        se = _tensor(scopes[scope][1]).to(device=est.device)
        n = int(est.numel())
        # Fail closed: a nonfinite estimate or SE can never pass the stop.
        if not (bool(torch.isfinite(est).all()) and bool(torch.isfinite(se).all())) or bool((se < 0).any()):
            raise FloatingPointError(f"census scope {scope!r} has a nonfinite estimate or a nonfinite/negative SE")
        records: list[dict[str, Any]] = []
        e_scope = 0.0
        skipped = 0
        ks = sorted({int(v) for v in cuts.get(scope, ()) if 1 <= int(v) < n}, reverse=True)
        if ks:
            # One sort per scope per check; every cut reads its threshold and counts from it.
            ascending, order = torch.sort(est, stable=True)
            se_ascending = se[order]
            del order  # peak memory ~ 24 bytes per route of the scope during the sort, 16 after
            se_max = float(se.max().item())

        def count_between(low: float, high: float) -> int:
            left = torch.searchsorted(ascending, torch.tensor(low, dtype=ascending.dtype, device=ascending.device),
                                      right=True)
            right = torch.searchsorted(ascending, torch.tensor(high, dtype=ascending.dtype, device=ascending.device),
                                       right=False)
            return max(int(right.item()) - int(left.item()), 0)

        for k in ks:
            if early_exit:
                break
            t = float(ascending[n - k].item())  # k-th largest; the value does not depend on tie order
            if not t > 0.0:
                skipped += 1  # a block of zero scores: a deterministic tie
                continue
            if nu >= 1.0:
                # A route's term is F_t(-(g + delta)/se); it is kept when (g + delta) < W se, so the
                # candidates lie within W se_max - delta of the cut (none when delta >= W se_max).
                budget = math.inf if population_exhausted else max(threshold - total - e_scope, 0.0)
                e_cut, exceeded, window_routes, sig = _cut_terms_outward(
                    ascending, se_ascending, position=n - k, t=t, delta=delta, nu=nu, window_ses=window_ses,
                    reach=window_ses * se_max - delta, budget=budget,
                )
                if exceeded:
                    early_exit, exit_scope = True, scope
            else:
                window_routes, sig = 0, 0.0
                e_cut = math.inf if bool((se > 0).any()) else 0.0
            near = count_between(0.9 * t, 1.1 * t)
            rho = near / (0.2 * t)
            pred_truth = rho * sig / (math.sqrt(2.0 * math.pi) * k)
            zone_below = count_between(t - delta, t)
            records.append({
                "k": k, "f": k / n, "t": t, "E": e_cut, "window_routes": int(window_routes),
                "rho": rho, "sig": sig, "pred_1mO_truth": pred_truth,
                "pred_1mO_retest": math.sqrt(2.0) * pred_truth, "bound_1mO": (zone_below + threshold) / k,
            })
            e_scope += e_cut
            if not early_exit and total + e_scope > threshold and not population_exhausted:
                early_exit, exit_scope = True, scope
        worst = max(records, key=lambda c: c["E"]) if records else None
        per_scope[scope] = {
            "E": e_scope, "scope_size": n, "cuts_evaluated": len(records), "cuts_skipped_zero": skipped,
            "worst_cut": worst, "cuts": records,
        }
        total += e_scope
    binding = exit_scope if early_exit else (max(per_scope, key=lambda s: per_scope[s]["E"]) if per_scope else None)
    passed = bool(population_exhausted) or (not early_exit and total <= threshold)
    # eps = 0 is population-only: zero SEs on a sample do not establish coverage.
    if float(eps) == 0.0 and not population_exhausted:
        passed = False
    # Cuts are processed one at a time and no window is retained across cuts, so a check
    # holds at most one scope's sort and one cut's window; the draws forecast was dropped with the retention.
    b_forecast = None
    return {
        "rule": IZ_STOP_RULE,
        "passed": passed,
        "reason": ("population" if population_exhausted else ("alpha" if passed else None)),
        "statistic": total,
        "E_total": total,
        "early_exit": bool(early_exit),
        "E_total_is_lower_bound": bool(early_exit),
        "alpha": float(alpha),
        "alpha_check": threshold,
        "eps": float(eps),
        "delta": delta,
        "tau_S": float(tau_S),
        "nu": nu,
        "look": int(look),
        "n_draws": int(n_draws),
        "binding_scope": binding,
        "B_forecast": b_forecast,
        "scopes": per_scope,
    }


__all__ = ["IZ_DEFAULT_EPS", "IZ_STOP_RULE", "alpha_at_check", "census_iz_stop"]
