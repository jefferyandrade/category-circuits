"""Registered PPS census design ``pps_poisson_v1``: sampler, look loop, stop rule, receipt.

One PRN order for every scope, per-scope stops, and forecast-sized looks in whole
kernel batches.  The walk and its decision law
are unchanged; only the dense census's sample and its stopping change.

Estimand.  For route r the census score is ``S_r = sum_i a_i y_ri`` with
class-balanced weights: ``a_i = 0.5 / N_on`` on target and
``a_i = 0.5 / (C_off * N_class(i))`` off target (each off-target class carries
``0.5 / C_off``, as in the walk's own stratified loss).

Sample.  Every on-target image is taken (pi = 1).  Off-target images are drawn
by Poisson sampling with probability proportional to size,
``pi_i(lam) = min(1, lam * x_i)``, ``x_i = a_i * u_i``, where
``u_i = |sigmoid(eta_i) - y_i|`` comes from the dense target logits and the
run's dense calibration.  Image i is in the look with parameter ``lam`` iff
``U_i <= pi_i(lam)``; the permanent random numbers ``U_i`` are fixed by
``(run_id, random_seed)``, so looks are nested and one seed gives one census.

Estimator.  Horvitz-Thompson: first weight ``a_i / pi_i``; Poisson variance
``sum (1 - pi_i) (a_i y_ri / pi_i)**2``.  In the kernel's lambda-free form
(uncapped: S1 = sum y/u, S2 = sum y^2/u^2, S3 = sum a y^2/u; capped:
Sc = sum a y): field = Sc + S1/lam and variance = S2/lam^2 - S3/lam.

Stop.  A scope (layer x partition) stops at the first look where, at every cut
in its band, the predicted miss rate ``rho * e_UCB / (sqrt(pi) k)`` is at most
``(1 - O*) (1 - k/N_s)`` (predicted nestedness >= O*).  The census stops when
every scope has stopped, or the registered image cap of a scope's kind binds
(conv and fc scopes have separate caps), or pi = 1 everywhere (exact); the receipt
records each scope's reached state.  Wall time is recorded only: every stop
is a function of (seed, data, settings), never of node speed.

This module holds no engine state.  The weighted census itself is computed by
a :class:`CensusLookSession` (the kernel worker's fast census); the module
decides which images enter at each look and when each scope stops.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol

import numpy as np

CENSUS_SAMPLE_DESIGN_PPS = "pps_poisson_v1"
PPS_PRN_ROLE = "census_prn_v1"
STOP_TARGET = "target"
STOP_CAP = "cap"
STOP_EXACT = "exact"
SCOPE_KINDS = ("conv", "fc")

# Registered settings (environment names; defaults in PPSCensusSettings).
PPS_ENV = {
    "target_overlap": "IRG_CENSUS_TARGET_OVERLAP",
    "initial_expected_off_target": "IRG_CENSUS_INITIAL_OFF_TARGET",
    "cap": "IRG_CENSUS_IMAGE_CAP",
    "cap_conv": "IRG_CENSUS_IMAGE_CAP_CONV",
    "cap_fc": "IRG_CENSUS_IMAGE_CAP_FC",
    "overlap_alpha": "IRG_CENSUS_OVERLAP_ALPHA",
    "batch_size": "IRG_CENSUS_BATCH_SIZE",
}
PPS_BAND_TABLE_ENV = "IRG_CENSUS_STOP_BAND_TABLE"


@dataclass(frozen=True)
class PPSCensusSettings:
    """Registered parameters of the PPS census."""

    target_overlap: float = 0.99
    initial_expected_off_target: int = 2048
    growth_min: float = 1.5
    growth_max: float = 4.0
    forecast_inflation: float = 1.2
    # Hard cap on expected off-target images; None lets the census reach
    # pi = 1 (exact).  cap_conv / cap_fc are the registered per-network image
    # caps of each scope kind (None: the hard cap applies).
    cap: int | None = 262144
    cap_conv: int | None = None
    cap_fc: int | None = None
    overlap_alpha: float = 0.05
    ladder_ratio: float = 0.85
    density_halfwidth: float = 0.10
    density_halfwidth_steps: tuple[float, ...] = (0.10, 0.25, 0.50)
    minimum_band_count: int = 200
    se_quantile: float = 0.90
    # The kernel's saturation batch b* (per network and GPU); looks are whole
    # multiples of it.
    batch_size: int = 1
    # Used for a layer missing from the band table; flagged in the receipt.
    default_band: tuple[float, float] = (0.002, 0.5)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.target_overlap) < 1.0:
            raise ValueError("census target overlap must lie in (0, 1)")
        if int(self.initial_expected_off_target) < 1:
            raise ValueError("census initial look needs at least one expected image")
        if not 1.0 < float(self.growth_min) <= float(self.growth_max):
            raise ValueError("census look growth bounds are invalid")
        for cap in (self.cap, self.cap_conv, self.cap_fc):
            if cap is not None and int(cap) < int(self.initial_expected_off_target):
                raise ValueError("census image cap is below the first look")
        if not 0.0 < float(self.overlap_alpha) < 1.0:
            raise ValueError("census overlap alpha must lie in (0, 1)")
        if not 0.0 < float(self.ladder_ratio) < 1.0:
            raise ValueError("census ladder ratio must lie in (0, 1)")
        if int(self.batch_size) < 1:
            raise ValueError("census batch size must be positive")
        lo, hi = (float(v) for v in self.default_band)
        if not 0.0 < lo < hi < 1.0:
            raise ValueError("census default band must satisfy 0 < f_lo < f_hi < 1")

    def receipt(self) -> dict[str, Any]:
        value = asdict(self)
        value["default_band"] = [float(v) for v in self.default_band]
        value["density_halfwidth_steps"] = [float(v) for v in self.density_halfwidth_steps]
        return value

    def kind_cap(self, kind: str) -> int | None:
        specific = {"conv": self.cap_conv, "fc": self.cap_fc}.get(str(kind))
        if specific is None:
            return self.cap
        return int(specific) if self.cap is None else min(int(specific), int(self.cap))


def pps_settings_from_environment(environ: Mapping[str, str] | None = None) -> PPSCensusSettings:
    source = os.environ if environ is None else environ
    values: dict[str, Any] = {}
    for name, env in PPS_ENV.items():
        raw = source.get(env)
        if raw is None or not str(raw).strip():
            continue
        raw = str(raw).strip()
        if name in {"cap", "cap_conv", "cap_fc"}:
            values[name] = None if raw.lower() in {"none", "exact"} else int(raw)
        elif name in {"initial_expected_off_target", "batch_size"}:
            values[name] = int(raw)
        else:
            values[name] = float(raw)
    return PPSCensusSettings(**values)


def load_band_table(path: str | os.PathLike[str] | None) -> tuple[dict[str, tuple[float, float]], str | None]:
    """Per-layer stop band {layer_id: (f_lo, f_hi)} and the file's sha256."""

    if path is None or not str(path).strip():
        return {}, None
    payload = open(path, "rb").read()
    doc = json.loads(payload)
    bands = doc.get("bands", doc)
    table: dict[str, tuple[float, float]] = {}
    for layer, band in bands.items():
        lo, hi = float(band[0]), float(band[1])
        if not 0.0 < lo < hi <= 1.0:
            raise ValueError(f"census stop band for {layer} must satisfy 0 < f_lo < f_hi <= 1")
        table[str(layer)] = (lo, hi)
    return table, hashlib.sha256(payload).hexdigest()


def pps_design_receipt(
    settings: PPSCensusSettings,
    *,
    band_table_sha256: str | None,
) -> dict[str, Any]:
    """The design dict bound into the census cache key and receipts."""

    from mechanistic_systems.measurement.run_products import loss_weighting

    return {
        "schema": "irg_census_sample_design_v2",
        "design": CENSUS_SAMPLE_DESIGN_PPS,
        "on_target": "whole_stratum_pi_1",
        "off_target": "poisson_pps_pi_min_1_lam_a_u",
        "class_weights": ("one_over_training_population_size" if loss_weighting() == "per_image"
                          else "on_0.5_over_N_on__off_0.5_over_C_off_N_class"),
        **({"loss_weighting": "per_image"} if loss_weighting() == "per_image" else {}),
        "size_measure": "abs_sigmoid_scale_times_logit_minus_center_plus_offset_minus_y",
        "prn": f"{PPS_PRN_ROLE}:philox4x64_over_canonical_sorted_context_ids",
        "lambda_solver": "sorted_closed_form_v1",
        "estimator": "horvitz_thompson",
        "variance": "poisson_ht_v1",
        "look_sizes": "forecast_v1_image_caps_no_wall_clock",
        "stop_rule": "predicted_nestedness_ucb_v1",
        "settings": settings.receipt(),
        "stop_band_table_sha256": band_table_sha256,
    }


# ---------------------------------------------------------------------------
# Inputs: class weights, size measure, permanent random numbers, lambda.
# ---------------------------------------------------------------------------


def class_weights(labels: np.ndarray, *, target_class_id: int) -> np.ndarray:
    """Image mass: balanced binary/class weights, or 1/N under per_image."""

    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    target = int(target_class_id)
    classes, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    on = values == target
    if not on.any() or on.all():
        raise ValueError("census class weights need on-target and off-target images")
    from mechanistic_systems.measurement.run_products import loss_weighting

    if loss_weighting() == "per_image":
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    off_classes = int(np.count_nonzero(classes != target))
    weights = 0.5 / (off_classes * counts[inverse].astype(np.float64))
    weights[on] = 0.5 / float(np.count_nonzero(on))
    return weights


def size_measure(
    target_logits: np.ndarray,
    labels: np.ndarray,
    *,
    target_class_id: int,
    scale: float,
    offset: float,
    input_center: float,
    fixed_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Absolute dense residual in fp64, including fixed substituted offsets."""

    from scipy.special import expit

    z = np.asarray(target_logits, dtype=np.float64).reshape(-1)
    y = (np.asarray(labels, dtype=np.int64).reshape(-1) == int(target_class_id)).astype(np.float64)
    if z.shape != y.shape:
        raise ValueError("census size measure needs one logit per image")
    eta = float(scale) * (z - float(input_center)) + float(offset)
    if fixed_offsets is not None:
        row_offsets = np.asarray(fixed_offsets, dtype=np.float64).reshape(-1)
        if row_offsets.shape != eta.shape or not np.isfinite(row_offsets).all():
            raise ValueError("substituted census size measure requires one finite offset per image")
        eta = eta + row_offsets
    if not np.isfinite(eta).all():
        raise ValueError("census size measure has a nonfinite calibrated logit")
    # |p - y| = sigmoid(-eta) on target and sigmoid(eta) off target: no cancellation.
    return np.where(y > 0.5, expit(-eta), expit(eta))


def permanent_random_numbers(
    context_ids: Sequence[str],
    *,
    run_id: str,
    random_seed: int,
    content_hash: Callable[..., str],
) -> np.ndarray:
    """U_i in [0, 1), fixed by (run_id, random_seed) and the image's identity.

    Philox4x64 keyed by the content hash of (run_id, random_seed, role); the
    counter is the image's position in the canonical (sorted) id order, so the
    number belongs to the image, not to the order of the input.
    """

    ids = np.asarray([str(value) for value in context_ids])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("census PRN context ids must be distinct")
    digest = content_hash({"run_id": str(run_id), "random_seed": int(random_seed), "role": PPS_PRN_ROLE})
    hexdigest = "".join(ch for ch in str(digest) if ch in "0123456789abcdef")[-32:]
    key = np.array([int(hexdigest[:16], 16), int(hexdigest[16:], 16)], dtype=np.uint64)
    order = np.argsort(ids, kind="stable")
    uniforms = np.random.Generator(np.random.Philox(key=key)).random(len(ids))
    values = np.empty(len(ids), dtype=np.float64)
    values[order] = uniforms
    return values


def solve_lambda(x: np.ndarray, expected_count: float) -> float:
    """lam with sum_i min(1, lam * x_i) = expected_count (x_i > 0), exactly.

    Sorted closed form: with the m largest sizes capped,
    lam = (B - m) / sum_{i > m} x_(i); the first m with lam * x_(m+1) <= 1 is
    the solution.  ``expected_count >= len(x)`` returns the smallest lam that
    caps every image (the exact census).
    """

    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(values > 0.0) or not np.isfinite(values).all():
        raise ValueError("census sizes must be finite and positive")
    target = float(expected_count)
    if not target > 0.0:
        raise ValueError("census expected count must be positive")
    if target >= values.size:
        return float(1.0 / values.min())
    ordered = np.sort(values)[::-1]
    tail = np.concatenate([np.cumsum(ordered[::-1])[::-1], [0.0]])  # tail[m] = sum_{i>=m} ordered[i]
    limit = int(math.floor(target))
    m = np.arange(0, limit + 1)
    lam = (target - m) / tail[m]
    ok = lam * ordered[m] <= 1.0
    first = int(np.argmax(ok)) if ok.any() else limit
    return float(lam[first])


def inclusion_probability(x: np.ndarray, lam: float) -> np.ndarray:
    return np.minimum(1.0, float(lam) * np.asarray(x, dtype=np.float64))


def kish_effective_size(u_uncapped_sampled: np.ndarray) -> float:
    """Kish n of the sampled uncapped HT weights (proportional to 1/u)."""

    w = 1.0 / np.asarray(u_uncapped_sampled, dtype=np.float64)
    if w.size == 0:
        return math.inf
    return float(w.sum() ** 2 / np.square(w).sum())


# ---------------------------------------------------------------------------
# Stop rule.
# ---------------------------------------------------------------------------


def band_cuts(scope_size: int, band: tuple[float, float], ladder_ratio: float = 0.85) -> np.ndarray:
    """Rung ladder k_j = ceil(N_s * r**j) restricted to f = k/N_s in [f_lo, f_hi]."""

    n = int(scope_size)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    lo, hi = float(band[0]), float(band[1])
    cuts: list[int] = []
    j = 0
    while True:
        k = int(math.ceil(n * float(ladder_ratio) ** j))
        f = k / n
        if f < lo or k < 1:
            break
        if f <= hi and (not cuts or cuts[-1] != k):
            cuts.append(k)
        j += 1
        if k == 1:
            break
    return np.asarray(cuts, dtype=np.int64)


def alpha_at_look(alpha: float, look_index: int) -> float:
    return float(alpha) / float((int(look_index) + 1) * (int(look_index) + 2))


def se_upper_factor(nu: float, alpha_j: float) -> float:
    """sqrt(nu / chi2_{alpha_j}(nu)): one-sided UCB factor for an SE with nu dof."""

    if not math.isfinite(nu):
        return 1.0
    from scipy.stats import chi2

    nu = max(float(nu), 1.0)
    return float(math.sqrt(nu / chi2.ppf(float(alpha_j), nu)))


def scope_overlap_check(
    scores: np.ndarray,
    standard_errors: np.ndarray,
    *,
    cuts: np.ndarray,
    target_overlap: float,
    se_factor: float,
    density_halfwidth_steps: Sequence[float] = (0.10, 0.25, 0.50),
    minimum_band_count: int = 200,
    se_quantile: float = 0.90,
) -> dict[str, Any]:
    """Evaluate the predicted-overlap stop rule for one scope at one look.

    Per cut k (t = k-th largest score): the density band b = {0.9t < S < 1.1t}
    is widened to +-25% and +-50% until it holds ``minimum_band_count``
    routes (a cut still short is evaluated on what it has and counted in
    ``cuts_short_band``); rho = |b| / (2 h t); e = the se_quantile of the
    standard errors in b, times the UCB factor; miss = rho e / (sqrt(pi) k);
    the cut passes if miss <= (1 - O*)(1 - k/N).  Cuts at t = 0 (a block of
    zero scores) and at k = N are skipped.
    """

    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    e = np.asarray(standard_errors, dtype=np.float64).reshape(-1)
    n = s.size
    if e.size != n:
        raise ValueError("scope scores and standard errors differ in length")
    order = np.argsort(-s, kind="stable")
    descending = s[order]
    ascending = descending[::-1]
    se_ascending = e[order][::-1]
    worst: dict[str, Any] | None = None
    worst_ratio = -math.inf
    evaluated = skipped = failed = short = 0
    for k in np.asarray(cuts, dtype=np.int64).tolist():
        if not 1 <= k < n:
            continue
        t = float(descending[k - 1])
        if not t > 0.0:
            skipped += 1  # deterministic tie block of zero scores
            continue
        evaluated += 1
        for halfwidth in density_halfwidth_steps:
            lo = int(np.searchsorted(ascending, (1.0 - halfwidth) * t, side="right"))
            hi = int(np.searchsorted(ascending, (1.0 + halfwidth) * t, side="left"))
            if hi - lo >= int(minimum_band_count):
                break
        band_short = hi - lo < int(minimum_band_count)
        short += int(band_short)
        band = max(hi - lo, 1)
        rho = band / (2.0 * halfwidth * t)
        e_hat = float(np.quantile(se_ascending[lo:hi], se_quantile)) if hi > lo else 0.0
        e_ucb = e_hat * float(se_factor)
        miss = rho * e_ucb / (math.sqrt(math.pi) * k)
        threshold = (1.0 - float(target_overlap)) * (1.0 - k / n)
        ratio = 0.0 if miss == 0.0 else miss / threshold
        if ratio > 1.0:
            failed += 1
        if ratio > worst_ratio:
            worst_ratio = ratio
            f = k / n
            predicted = 1.0 - miss
            worst = {
                "k": int(k), "f": f, "t_hat": t, "band_count": int(band), "band_halfwidth": float(halfwidth),
                "band_short": bool(band_short), "rho_hat": rho, "se_hat": e_hat, "se_ucb": e_ucb,
                "miss_hat": miss, "threshold": threshold, "ratio": ratio, "predicted_overlap": predicted,
                "predicted_nestedness": (predicted - f) / (1.0 - f),
            }
    return {
        "passed": failed == 0,
        "cuts_evaluated": evaluated,
        "cuts_skipped_zero": skipped,
        "cuts_failed": failed,
        "cuts_short_band": short,
        "worst_ratio": (0.0 if worst is None else float(worst_ratio)),
        "worst_cut": worst,
    }


def next_expected_size(
    current: float,
    *,
    worst_ratio: float,
    settings: PPSCensusSettings,
    ceiling: int,
) -> int:
    """Forecast the next look's expected off-target size.

    B* = B_j * ratio**2 * inflation (ratio = worst predicted miss over its
    threshold among the active scopes), clipped to [growth_min, growth_max] *
    B_j and to ``ceiling`` (the largest image cap among the active scopes, or
    the population), rounded up to whole kernel batches.  No wall clock enters.
    """

    b = float(current)
    ratio = float(worst_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0:
        target = settings.growth_max * b
    else:
        target = b * ratio * ratio * settings.forecast_inflation
    target = min(max(target, settings.growth_min * b), settings.growth_max * b)
    batch = int(settings.batch_size)
    rounded = int(math.ceil(target / batch) * batch)
    return int(min(rounded, int(ceiling)))


# ---------------------------------------------------------------------------
# The look loop.
# ---------------------------------------------------------------------------


class CensusLookSession(Protocol):
    """The weighted fast census (kernel worker), as the look loop drives it.

    ``add_images`` accumulates images into the lambda-free sums of the active
    scopes only: capped images into Sc (+= a*y), uncapped ones into S1, S2,
    S3.  ``recap_images`` moves previously uncapped images into Sc for the
    active scopes.  ``scope_fields`` returns, per requested scope, the score
    field ``Sc + S1/lam`` and its standard error
    ``sqrt(max(S2/lam**2 - S3/lam, 0))`` in that scope's route order.
    ``freeze_scopes`` fixes those scopes at ``lam``: later calls never touch
    their sums, and the final fields use that ``lam``.
    """

    def add_images(self, context_ids: Sequence[str], *, u: np.ndarray, a: np.ndarray,
                   capped: np.ndarray, active_scopes: frozenset[str]) -> None: ...

    def recap_images(self, context_ids: Sequence[str], *, u: np.ndarray, a: np.ndarray,
                     active_scopes: frozenset[str]) -> None: ...

    def scope_fields(self, lam: float, scopes: Iterable[str]) -> Mapping[str, tuple[np.ndarray, np.ndarray]]: ...

    def freeze_scopes(self, scopes: Iterable[str], lam: float) -> None: ...


def _ids_hash(ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for value in ids:
        h.update(str(value).encode())
        h.update(b"\0")
    return h.hexdigest()


def run_pps_census_looks(
    session: CensusLookSession,
    *,
    on_target_ids: Sequence[str],
    on_target_u: np.ndarray,
    on_target_a: np.ndarray,
    off_target_ids: Sequence[str],
    off_target_u: np.ndarray,
    off_target_a: np.ndarray,
    off_target_prn: np.ndarray,
    scope_sizes: Mapping[str, int],
    scope_layers: Mapping[str, str],
    scope_kinds: Mapping[str, str],
    settings: PPSCensusSettings,
    band_table: Mapping[str, tuple[float, float]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run the nested PPS looks until every scope stops; return the receipt.

    One PRN order serves every scope.  A scope stops on its own prefix (its
    own lam) when the predicted-overlap rule passes (``target``), when the
    next look would exceed its kind's image cap (``cap``), or when every
    off-target image has pi = 1 (``exact``).  The session is frozen at each
    scope's lam; the receipt's ``scope_lambda`` records them.  ``clock`` is
    read for the receipt only.
    """

    off_ids = np.asarray([str(v) for v in off_target_ids])
    u_off = np.asarray(off_target_u, dtype=np.float64).reshape(-1)
    a_off = np.asarray(off_target_a, dtype=np.float64).reshape(-1)
    prn = np.asarray(off_target_prn, dtype=np.float64).reshape(-1)
    if not (off_ids.size == u_off.size == a_off.size == prn.size) or off_ids.size == 0:
        raise ValueError("census off-target inputs differ in length or are empty")
    if not (np.all(u_off > 0.0) and np.all(a_off > 0.0)):
        raise ValueError("census off-target sizes must be positive")
    x = a_off * u_off
    n_off = int(x.size)
    order = np.argsort(prn, kind="stable")  # one PRN order for every scope
    bands = dict(band_table or {})
    scopes = sorted(map(str, scope_sizes))
    cuts: dict[str, np.ndarray] = {}
    band_source: dict[str, str] = {}
    scope_cap: dict[str, int] = {}
    for scope in scopes:
        layer = str(scope_layers[scope])
        kind = str(scope_kinds[scope])
        if kind not in SCOPE_KINDS:
            raise ValueError(f"census scope kind must be one of {SCOPE_KINDS}")
        band = bands.get(layer)
        band_source[scope] = "table" if band is not None else "default"
        cuts[scope] = band_cuts(int(scope_sizes[scope]), band or settings.default_band, settings.ladder_ratio)
        cap = settings.kind_cap(kind)
        scope_cap[scope] = n_off if cap is None else min(int(cap), n_off)

    active = set(scopes)
    records: dict[str, dict[str, Any]] = {}
    looks: list[dict[str, Any]] = []
    included = np.zeros(n_off, dtype=bool)
    capped = np.zeros(n_off, dtype=bool)
    started = clock()
    processed_images = 0
    batch = int(settings.batch_size)
    expected = float(min(
        int(math.ceil(settings.initial_expected_off_target / batch) * batch),
        max(scope_cap.values()),
    ))
    look = 0
    while True:
        look_started = clock()
        lam = solve_lambda(x, expected)
        pi = inclusion_probability(x, lam)
        now_included = prn <= pi
        now_capped = (lam * x >= 1.0) & now_included
        new = now_included & ~included
        recap = included & ~capped & now_capped
        frozen_active = frozenset(active)
        if look == 0:
            session.add_images(
                tuple(map(str, on_target_ids)),
                u=np.asarray(on_target_u, dtype=np.float64),
                a=np.asarray(on_target_a, dtype=np.float64),
                capped=np.ones(len(on_target_ids), dtype=bool),
                active_scopes=frozen_active,
            )
            processed_images += len(on_target_ids)
        new_sorted = order[new[order]]
        if new_sorted.size:
            session.add_images(
                tuple(off_ids[new_sorted].tolist()), u=u_off[new_sorted], a=a_off[new_sorted],
                capped=now_capped[new_sorted], active_scopes=frozen_active,
            )
        recap_sorted = order[recap[order]]
        if recap_sorted.size:
            session.recap_images(
                tuple(off_ids[recap_sorted].tolist()), u=u_off[recap_sorted], a=a_off[recap_sorted],
                active_scopes=frozen_active,
            )
        processed_images += int(new_sorted.size + recap_sorted.size)
        included, capped = now_included, now_capped
        sampled_uncapped = included & ~capped
        nu = kish_effective_size(u_off[sampled_uncapped])
        alpha_j = alpha_at_look(settings.overlap_alpha, look)
        factor = se_upper_factor(nu, alpha_j)
        exact = bool(np.all(capped))
        fields = session.scope_fields(lam, sorted(active))
        stopped: list[str] = []
        worst_by_scope: dict[str, float] = {}
        for scope in sorted(active):
            scores, errors = fields[scope]
            check = scope_overlap_check(
                scores, errors, cuts=cuts[scope], target_overlap=settings.target_overlap,
                se_factor=factor, density_halfwidth_steps=settings.density_halfwidth_steps,
                minimum_band_count=settings.minimum_band_count, se_quantile=settings.se_quantile,
            )
            reason = STOP_EXACT if exact else (STOP_TARGET if check["passed"] else None)
            records[scope] = {
                "scope": scope, "layer": str(scope_layers[scope]), "kind": str(scope_kinds[scope]),
                "scope_size": int(scope_sizes[scope]), "band_source": band_source[scope],
                "cuts": int(cuts[scope].size), "image_cap": int(scope_cap[scope]),
                "stop_look": look, "lambda": lam, "expected_off_target": float(pi.sum()),
                "sampled_off_target": int(included.sum()), "sampled_uncapped": int(sampled_uncapped.sum()),
                "kish_nu": nu, "alpha_j": alpha_j, "se_ucb_factor": factor,
                "check": check, "stop_reason": reason,
            }
            if reason is None:
                worst_by_scope[scope] = float(check["worst_ratio"])
            else:
                stopped.append(scope)
        active.difference_update(stopped)
        next_size: int | None = None
        cap_stopped: list[str] = []
        if active:
            ceiling = max(scope_cap[scope] for scope in active)
            next_size = next_expected_size(
                expected, worst_ratio=max(worst_by_scope[scope] for scope in active),
                settings=settings, ceiling=ceiling,
            )
            for scope in sorted(active):
                # A scope whose kind cap the next look would pass freezes here.
                if next_size <= expected or next_size > scope_cap[scope]:
                    records[scope]["stop_reason"] = STOP_CAP
                    cap_stopped.append(scope)
            active.difference_update(cap_stopped)
        elapsed = clock() - started
        look_record = {
            "look": look, "lambda": lam, "expected_off_target": float(pi.sum()),
            "sampled_off_target": int(included.sum()), "new_images": int(new_sorted.size),
            "recapped_images": int(recap_sorted.size), "active_before": len(frozen_active),
            "stopped_target_or_exact": len(stopped), "stopped_cap": len(cap_stopped),
            "alpha_j": alpha_j, "kish_nu": nu, "se_ucb_factor": factor,
            "wall_seconds": clock() - look_started, "elapsed_wall_seconds": elapsed,
            "sampled_ids_hash": _ids_hash(off_ids[order[included[order]]].tolist()),
            "next_expected_off_target": (next_size if active else None),
        }
        looks.append(look_record)
        if stopped or cap_stopped:
            session.freeze_scopes(sorted(stopped + cap_stopped), lam)
        if not active:
            break
        expected = float(next_size)
        look += 1
    reasons: dict[str, int] = {}
    for record in records.values():
        reasons[record["stop_reason"]] = reasons.get(record["stop_reason"], 0) + 1
    worst = [r["check"]["worst_cut"] for r in records.values() if r["check"]["worst_cut"] is not None]
    return {
        "schema": "irg_pps_census_looks_receipt_v2",
        "settings": settings.receipt(),
        "off_target_population": n_off,
        "on_target_count": len(on_target_ids),
        "look_count": len(looks),
        "looks": looks,
        "scopes": records,
        "scope_lambda": {scope: records[scope]["lambda"] for scope in records},
        "stop_reasons": reasons,
        "elapsed_wall_seconds": clock() - started,
        "wall_clock_role": "recorded_only",
        "processed_images": processed_images,
        "minimum_predicted_overlap": min((w["predicted_overlap"] for w in worst), default=1.0),
        "minimum_predicted_nestedness": min((w["predicted_nestedness"] for w in worst), default=1.0),
    }


# ---------------------------------------------------------------------------
# A NumPy reference session (planted tests; the kernel worker's fast census
# implements the same protocol on the GPU).
# ---------------------------------------------------------------------------


@dataclass
class ReferenceLookSession:
    """Lambda-free sums over explicit per-image rows ``rows[context_id] -> y`` (>= 0)."""

    rows: Mapping[str, np.ndarray]
    scope_slices: Mapping[str, slice]
    frozen: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        width = len(next(iter(self.rows.values())))
        self.S1 = np.zeros(width)
        self.S2 = np.zeros(width)
        self.S3 = np.zeros(width)
        self.Sc = np.zeros(width)
        self.calls: list[tuple[str, int, frozenset[str]]] = []

    def _mask(self, scopes: frozenset[str]) -> np.ndarray:
        mask = np.zeros(self.S1.size, dtype=bool)
        for scope in scopes:
            if scope not in self.frozen:
                mask[self.scope_slices[scope]] = True
        return mask

    def add_images(self, context_ids, *, u, a, capped, active_scopes):
        mask = self._mask(active_scopes)
        self.calls.append(("add", len(context_ids), active_scopes))
        for cid, ui, ai, ci in zip(context_ids, u, a, capped, strict=True):
            y = np.asarray(self.rows[cid], dtype=np.float64)
            if ci:
                self.Sc[mask] += ai * y[mask]
            else:
                self.S1[mask] += y[mask] / ui
                self.S2[mask] += np.square(y[mask] / ui)
                self.S3[mask] += ai * np.square(y[mask]) / ui

    def recap_images(self, context_ids, *, u, a, active_scopes):
        mask = self._mask(active_scopes)
        self.calls.append(("recap", len(context_ids), active_scopes))
        for cid, ui, ai in zip(context_ids, u, a, strict=True):
            y = np.asarray(self.rows[cid], dtype=np.float64)
            self.S1[mask] -= y[mask] / ui
            self.S2[mask] -= np.square(y[mask] / ui)
            self.S3[mask] -= ai * np.square(y[mask]) / ui
            self.Sc[mask] += ai * y[mask]

    def _field(self, sl: slice, lam: float) -> tuple[np.ndarray, np.ndarray]:
        score = self.Sc[sl] + self.S1[sl] / lam
        variance = np.maximum(self.S2[sl] / lam**2 - self.S3[sl] / lam, 0.0)
        return score, np.sqrt(variance)

    def scope_fields(self, lam, scopes):
        return {scope: self._field(self.scope_slices[scope], float(lam)) for scope in scopes}

    def freeze_scopes(self, scopes, lam):
        for scope in scopes:
            self.frozen.setdefault(str(scope), float(lam))

    def final_fields(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {scope: self._field(sl, self.frozen[scope]) for scope, sl in self.scope_slices.items()}


__all__ = [
    "CENSUS_SAMPLE_DESIGN_PPS", "PPS_BAND_TABLE_ENV", "PPS_ENV", "PPSCensusSettings", "CensusLookSession",
    "ReferenceLookSession", "SCOPE_KINDS", "STOP_CAP", "STOP_EXACT", "STOP_TARGET", "alpha_at_look",
    "band_cuts", "class_weights", "inclusion_probability", "kish_effective_size", "load_band_table",
    "next_expected_size", "permanent_random_numbers", "pps_design_receipt", "pps_settings_from_environment",
    "run_pps_census_looks", "scope_overlap_check", "se_upper_factor", "size_measure", "solve_lambda",
]
