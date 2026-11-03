"""Calibration (K), paired balanced BCE (O), and terminal metrics for the tractable lane.

Definitions follow the v0 category protocol (``top1_category_baseline``) so that
v1 and v0 numbers are the same quantities:

* binary problem: target category vs everything else on the raw target logit ``r``;
* calibration: positive affine ``r_cal = a r + b`` fitted on a train-side
  calibration stream by minimizing the class-balanced BCE with
  ``pos_weight = n_neg / n_pos`` (candidate-specific);
* fidelity: ``binary_classification_loss`` = balanced BCE of ``r_cal`` on the
  evaluation stream (``dBCE = circuit - dense``);
* corrected accuracy = sensitivity + specificity - 1 at threshold 0.5.

The registered pathwise ceiling is a mean dense-relative BCE increment ``B``;
the trust radius in raw-logit units is ``tau_L = B / a``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class AffineCalibration:
    scale: float
    offset: float
    loss: float
    iterations: int
    positive_scale: bool

    def apply(self, raw_target_logit: torch.Tensor) -> torch.Tensor:
        return self.scale * raw_target_logit + self.offset

    def to_json(self) -> dict[str, Any]:
        return dict(schema="tractable_v1_affine_calibration_v1", scale=self.scale, offset=self.offset,
                    loss=self.loss, iterations=self.iterations, positive_scale=self.positive_scale)


def _balanced_weights(labels: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    y = labels.to(torch.float64)
    n_pos = float(y.sum().item()); n_neg = float(y.numel() - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        raise ValueError("balanced BCE needs both strata")
    pos_weight = n_neg / n_pos
    w = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.ones_like(y))
    return w / float(y.numel()), n_pos, n_neg


def balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """``BCEWithLogitsLoss(pos_weight=n_neg/n_pos, reduction='mean')`` exactly as v0 computes it."""

    x = logits.detach().to(torch.float64); y = labels.detach().to(x.device, torch.float64)
    w, n_pos, n_neg = _balanced_weights(labels)
    # mean over samples of  pos_weight*y*softplus(-x) + (1-y)*softplus(x)
    loss = torch.where(y > 0.5, (n_neg / n_pos) * torch.nn.functional.softplus(-x), torch.nn.functional.softplus(x))
    return float(loss.mean().item())


def fit_affine_calibration(raw_target_logit: torch.Tensor, labels: torch.Tensor, *,
                           max_iterations: int = 100, tolerance: float = 1e-10) -> AffineCalibration:
    """Newton's method on the convex balanced BCE in ``(a, b)``; refuses to a positive scale (G-CAL)."""

    # two-parameter convex fit: run on the CPU in fp64 (tiny problem; avoids per-iteration device syncs)
    r = raw_target_logit.detach().to("cpu", torch.float64).reshape(-1)
    y = labels.detach().to("cpu", torch.float64).reshape(-1)
    w, n_pos, n_neg = _balanced_weights(y)
    pw = n_neg / n_pos
    cw = torch.where(y > 0.5, torch.full_like(y, pw), torch.ones_like(y))  # per-sample class weight

    def loss_grad_hess(a: float, b: float):
        x = a * r + b
        p = torch.sigmoid(x)
        loss = (cw * torch.where(y > 0.5, torch.nn.functional.softplus(-x), torch.nn.functional.softplus(x))).mean()
        g = cw * (p - y)                       # d loss_i / d x_i
        h = cw * p * (1 - p)
        grad = torch.stack([(g * r).mean(), g.mean()])
        hess = torch.stack([torch.stack([(h * r * r).mean(), (h * r).mean()]),
                            torch.stack([(h * r).mean(), h.mean()])])
        return float(loss.item()), grad, hess

    a, b = 1.0, 0.0
    loss, grad, hess = loss_grad_hess(a, b)
    it = 0
    for it in range(1, max_iterations + 1):
        step = torch.linalg.solve(hess + 1e-12 * torch.eye(2, dtype=torch.float64), grad)
        t = 1.0
        while t > 1e-6:
            na, nb = a - t * float(step[0]), b - t * float(step[1])
            nloss, ngrad, nhess = loss_grad_hess(na, nb)
            if nloss <= loss - 1e-4 * t * float((grad * step).sum()):
                break
            t *= 0.5
        a, b, loss, grad, hess = na, nb, nloss, ngrad, nhess
        if float(grad.abs().max()) < tolerance:
            break
    positive = a > 0
    if not positive:
        # offset-only fallback at unit scale (disclosed); the positivity gate records the refusal
        a = 1.0
        b = float(torch.log(torch.tensor(n_pos / n_neg, dtype=torch.float64)) - r.mean())
        loss = balanced_bce(a * r + b, y)
    return AffineCalibration(scale=float(a), offset=float(b), loss=float(loss), iterations=it, positive_scale=bool(positive))


@dataclass(frozen=True)
class BinaryMetrics:
    bce: float
    corrected_accuracy: float
    sensitivity: float
    specificity: float
    n_positive: int
    n_negative: int

    def to_json(self) -> dict[str, Any]:
        return dict(schema="tractable_v1_binary_metrics_v1", **{k: getattr(self, k) for k in self.__dataclass_fields__})


def binary_metrics(raw_target_logit: torch.Tensor, labels: torch.Tensor, calibration: AffineCalibration) -> BinaryMetrics:
    x = calibration.apply(raw_target_logit.detach().to(torch.float64))
    y = labels.detach().to(x.device, torch.bool)
    pred = torch.sigmoid(x) > 0.5
    tp = float((pred & y).sum().item()); tn = float((~pred & ~y).sum().item())
    n_pos = int(y.sum().item()); n_neg = int(y.numel() - n_pos)
    sens = tp / n_pos if n_pos else math.nan
    spec = tn / n_neg if n_neg else math.nan
    return BinaryMetrics(bce=balanced_bce(x, y), corrected_accuracy=sens + spec - 1.0, sensitivity=sens,
                         specificity=spec, n_positive=n_pos, n_negative=n_neg)


@dataclass(frozen=True)
class FunctionPreservationMetrics:
    rms_logit_displacement: float
    rms_logit_displacement_target: float
    rms_logit_displacement_complement: float
    empirical_rms_logit_displacement: float
    logit_rank_correlation: float | None
    balanced_logit_rank_correlation: float | None
    logit_rank_correlation_target: float | None
    logit_rank_correlation_complement: float | None
    logit_rank_correlation_status: str
    balanced_logit_rank_correlation_status: str
    logit_rank_correlation_target_status: str
    logit_rank_correlation_complement_status: str
    n_positive: int
    n_negative: int

    def to_json(self) -> dict[str, Any]:
        return dict(
            schema="tractable_v1_function_preservation_metrics_v2",
            **{k: getattr(self, k) for k in self.__dataclass_fields__},
            rms_measure="equal_target_complement_mass_uniform_within_each_stratum",
            empirical_rms_measure="uniform_over_observed_contexts",
            logit_rank_correlation_law="ordinary_pooled_average_tie_Spearman_empirical_unbalanced",
            balanced_logit_rank_correlation_law="weighted_mid_distribution_ranks_weighted_Pearson_equal_stratum_mass",
            per_stratum_rank_law="ordinary_average_tie_Spearman_within_stratum",
            undefined_rank_value="null_with_explicit_status",
            authority="finite_function_preservation_diagnostic_only",
        )


def _binary_labels(labels: torch.Tensor, *, expected: int) -> torch.Tensor:
    y = labels.detach().to("cpu", torch.float64).reshape(-1)
    if y.numel() != expected or expected == 0:
        raise ValueError("labels and logits require the same nonempty context axis")
    if not bool(torch.isfinite(y).all()) or not bool(((y == 0) | (y == 1)).all()):
        raise ValueError("function metrics require finite binary labels")
    y = y.bool()
    if not bool(y.any()) or bool(y.all()):
        raise ValueError("balanced function metrics require both strata")
    return y


def _rms(x: torch.Tensor) -> float:
    scale = float(x.abs().max())
    return 0.0 if scale == 0.0 else scale * math.sqrt(float((x / scale).square().mean()))


def _mid_distribution_ranks(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted empirical CDF at each tie group's midpoint, in input order."""
    order = torch.argsort(values, stable=True)
    _, groups = torch.unique_consecutive(values[order], return_inverse=True)
    mass = torch.bincount(groups, weights=weights[order])
    midpoint = mass.cumsum(0) - 0.5 * mass
    ranks = torch.empty_like(values)
    ranks[order] = midpoint[groups]
    return ranks


def _rank_correlation(circuit: torch.Tensor, dense: torch.Tensor,
                      weights: torch.Tensor) -> tuple[float | None, str]:
    if circuit.numel() < 2:
        return None, "fewer_than_two_contexts"
    c_constant = bool((circuit == circuit[0]).all())
    d_constant = bool((dense == dense[0]).all())
    if c_constant or d_constant:
        return None, ("both_constant" if c_constant and d_constant else
                      "circuit_constant" if c_constant else "dense_constant")
    weights = weights / weights.sum()
    c = _mid_distribution_ranks(circuit, weights)
    d = _mid_distribution_ranks(dense, weights)
    c = c - (weights * c).sum()
    d = d - (weights * d).sum()
    denominator = torch.sqrt((weights * c.square()).sum() * (weights * d.square()).sum())
    value = float((weights * c * d).sum() / denominator)
    return max(-1.0, min(1.0, value)), "defined"


def function_preservation_metrics(circuit_logits: torch.Tensor, dense_logits: torch.Tensor,
                                  labels: torch.Tensor) -> FunctionPreservationMetrics:
    """Balanced raw-logit RMS and separately labeled rank diagnostics.

    Every observation has mass 0.5/n_h for the balanced metric. Conventional
    pooled Spearman still uses uniform context mass; its ties use average
    ranks. A separate weighted-rank correlation uses the balanced empirical
    measure for both ranks and covariance. These are finite diagnostics.
    """
    r = circuit_logits.detach().to("cpu", torch.float64).reshape(-1)
    d = dense_logits.detach().to("cpu", torch.float64).reshape(-1)
    if r.numel() != d.numel():
        raise ValueError("circuit and dense logits must share the same context axis")
    y = _binary_labels(labels, expected=r.numel())
    if not bool(torch.isfinite(r).all()) or not bool(torch.isfinite(d).all()):
        raise ValueError("function metrics require finite raw logits")
    displacement = r - d
    if not bool(torch.isfinite(displacement).all()):
        raise ValueError("raw-logit displacement exceeds the finite diagnostic range")
    n_pos = int(y.sum()); n_neg = int(y.numel() - n_pos)
    uniform = torch.full_like(r, 1.0 / r.numel())
    balanced = torch.where(y, torch.full_like(r, 0.5 / n_pos), torch.full_like(r, 0.5 / n_neg))
    pooled, pooled_status = _rank_correlation(r, d, uniform)
    weighted, weighted_status = _rank_correlation(r, d, balanced)
    target, target_status = _rank_correlation(r[y], d[y], torch.ones(n_pos, dtype=torch.float64))
    complement, complement_status = _rank_correlation(r[~y], d[~y], torch.ones(n_neg, dtype=torch.float64))
    target_rms, complement_rms = _rms(displacement[y]), _rms(displacement[~y])
    return FunctionPreservationMetrics(
        rms_logit_displacement=math.hypot(target_rms, complement_rms) / math.sqrt(2.0),
        rms_logit_displacement_target=target_rms,
        rms_logit_displacement_complement=complement_rms,
        empirical_rms_logit_displacement=_rms(displacement),
        logit_rank_correlation=pooled, balanced_logit_rank_correlation=weighted,
        logit_rank_correlation_target=target, logit_rank_correlation_complement=complement,
        logit_rank_correlation_status=pooled_status,
        balanced_logit_rank_correlation_status=weighted_status,
        logit_rank_correlation_target_status=target_status,
        logit_rank_correlation_complement_status=complement_status,
        n_positive=n_pos, n_negative=n_neg,
    )


@torch.no_grad()
def terminal_function_evaluation(net, *, target_class: int,
                                 calibration_images: torch.Tensor | None = None,
                                 calibration_labels: torch.Tensor | None = None,
                                 calibration_batches=None,
                                 evaluation_batches, forward_batch: int = 256) -> dict[str, Any]:
    """Score current circuit and original dense referent on identical batches.

    This helper does not choose when terminal data may be opened. Its caller
    must freeze the train-generated family first. Both candidate-specific
    calibrations are fitted on the same supplied K images before the first
    terminal batch is requested. The optional calibration_batches callable
    replaces the two tensor inputs and must preserve their physical forward
    partition. Only compact CPU logits/labels survive each K batch.
    Current circuit bias corrections and original
    dense biases are evaluated without any persistent mask/bias mutation.
    """
    from .alexnet_factors import raw_target_logits

    dense_mask = torch.ones_like(net.mask)
    if calibration_batches is None:
        if calibration_images is None or calibration_labels is None:
            raise ValueError("provide calibration tensors or calibration_batches")
        yk = _binary_labels(calibration_labels, expected=calibration_images.shape[0])
        circuit_k = raw_target_logits(net, calibration_images, target_class=target_class,
                                     batch_size=forward_batch)
        dense_k = raw_target_logits(net, calibration_images, target_class=target_class,
                                   batch_size=forward_batch, mask_override=dense_mask, use_dense_biases=True)
    else:
        if calibration_images is not None or calibration_labels is not None or not callable(calibration_batches):
            raise ValueError("calibration_batches is callable and mutually exclusive with calibration tensors")
        circuit_k_rows, dense_k_rows, k_labels = [], [], []
        iterator = iter(calibration_batches())
        try:
            for images, labels in iterator:
                labels = labels.detach().cpu().reshape(-1)
                if images.shape[0] != labels.numel() or not bool(torch.isfinite(labels).all()) or not bool(((labels == 0) | (labels == 1)).all()):
                    raise ValueError("K batch requires matching finite binary labels")
                circuit_k_rows.append(raw_target_logits(net, images, target_class=target_class, batch_size=forward_batch).double().cpu())
                dense_k_rows.append(raw_target_logits(net, images, target_class=target_class, batch_size=forward_batch,
                                                     mask_override=dense_mask, use_dense_biases=True).double().cpu())
                k_labels.append(labels)
                del images, labels
        finally:
            close = getattr(iterator, "close", None)
            if close is not None: close()
        if not circuit_k_rows: raise ValueError("calibration requires nonempty K batches")
        circuit_k = torch.cat(circuit_k_rows); dense_k = torch.cat(dense_k_rows)
        yk = _binary_labels(torch.cat(k_labels), expected=circuit_k.numel())
        del circuit_k_rows, dense_k_rows, k_labels
    circuit_cal = fit_affine_calibration(circuit_k, yk)
    dense_cal = fit_affine_calibration(dense_k, yk)
    circuit_rows, dense_rows, label_rows = [], [], []
    iterator = iter(evaluation_batches())
    try:
        for images, labels in iterator:
            circuit_rows.append(raw_target_logits(net, images, target_class=target_class,
                                                  batch_size=forward_batch).double().cpu())
            dense_rows.append(raw_target_logits(net, images, target_class=target_class,
                                                batch_size=forward_batch, mask_override=dense_mask,
                                                use_dense_biases=True).double().cpu())
            labels = labels.detach().cpu().reshape(-1)
            if labels.numel() != images.shape[0]:
                raise ValueError("terminal batch labels differ from its actual image axis")
            label_rows.append(labels)
            del images, labels
    finally:
        close = getattr(iterator, "close", None)
        if close is not None: close()
    if not circuit_rows:
        raise ValueError("terminal function evaluation needs at least one batch")
    circuit = torch.cat(circuit_rows); dense = torch.cat(dense_rows); y = torch.cat(label_rows)
    preservation = function_preservation_metrics(circuit, dense, y)
    def report(raw, calibration, density):
        metrics = binary_metrics(raw, y, calibration)
        return dict(schema="tractable_v1_terminal_evaluation_v2", density=density,
                    calibration=calibration.to_json(), metrics=metrics.to_json(),
                    bce=metrics.bce, corrected_accuracy=metrics.corrected_accuracy,
                    uncalibrated_bce=balanced_bce(raw, y))
    return dict(schema="tractable_v1_paired_terminal_function_evaluation_v1",
                circuit=report(circuit, circuit_cal, net.density),
                dense=report(dense, dense_cal, 1.0),
                function_preservation=preservation.to_json(),
                dense_reference="all_live_routes_original_loaded_layer_and_readout_biases",
                paired_same_evaluation_batches=True,
                separate_calibrations_on_identical_train_images=True,
                authority="finite_terminal_diagnostic_only")


def trust_radii(bce_ceiling: float, slope: float) -> dict[str, float]:
    """Trust radii ``tau_L = B / kappa`` and ``tau_Q = tau_L^2`` (raw-logit units)."""

    if slope <= 0:
        raise ValueError("calibration slope must be positive")
    tau_l = bce_ceiling / slope
    return dict(bce_ceiling=bce_ceiling, slope=slope, tau_L=tau_l, tau_Q=tau_l * tau_l)


# --------------------------------------------------------------------------------------------
# Fisher weights of the balanced BCE and an anytime-valid confidence sequence for the oracle
# --------------------------------------------------------------------------------------------

def balanced_bce_taylor_weights(raw_target_logit: torch.Tensor, labels: torch.Tensor,
                                calibration: AffineCalibration) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-context first- and second-order coefficients of the balanced BCE in the RAW target logit.

    With ``x = a r + b`` and class weight ``omega_c`` (``pos_weight`` on targets, 1 on complement),
    ``d BCE_c / d r = omega_c a (p_c - y_c)`` and ``d^2 BCE_c / d r^2 = omega_c a^2 p_c (1 - p_c)``.
    Returns ``(g, h, omega)`` so that ``dBCE ~ mean_c [ g_c dr_c + 0.5 h_c dr_c^2 ]`` over the
    evaluation measure (the mean over contexts is the balanced BCE's own reduction).
    """

    r = raw_target_logit.detach().to(torch.float64)
    y = labels.detach().to(r.device, torch.float64)
    n_pos = float(y.sum().item()); n_neg = float(y.numel() - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        raise ValueError("Fisher weights need both strata")
    omega = torch.where(y > 0.5, torch.full_like(y, n_neg / n_pos), torch.ones_like(y))
    p = torch.sigmoid(calibration.scale * r + calibration.offset)
    g = omega * calibration.scale * (p - y)
    h = omega * calibration.scale ** 2 * p * (1.0 - p)
    return g, h, omega


@dataclass(frozen=True)
class ConfidenceSequence:
    """Asymptotic anytime-valid confidence sequence for a mean (Waudby-Smith, Arbour, Sinha, Ramdas 2021).

    For running mean ``m_n`` and running variance ``s_n^2`` of i.i.d. observations, the interval
    ``m_n +/- s_n * sqrt( 2 (n rho^2 + 1) / (n^2 rho^2) * log( sqrt(n rho^2 + 1) / alpha ) )`` holds
    uniformly over ``n`` at level ``alpha`` asymptotically.  ``rho`` tunes where the sequence is
    tightest; the default follows the paper's recommendation for a planning size ``n_plan``.
    """

    alpha: float = 0.05
    rho: float | None = None
    n_plan: int = 400

    def _rho(self) -> float:
        if self.rho is not None:
            return self.rho
        la = -2.0 * math.log(self.alpha)
        return math.sqrt((la + math.log(la + 1.0)) / self.n_plan)

    def interval(self, samples: torch.Tensor) -> tuple[float, float, float, int]:
        x = samples.detach().to("cpu", torch.float64).reshape(-1)
        n = int(x.numel())
        if n < 2:
            return math.nan, -math.inf, math.inf, n
        m = float(x.mean()); s = float(x.std(unbiased=True))
        rho = self._rho()
        width = s * math.sqrt(2.0 * (n * rho * rho + 1.0) / (n * n * rho * rho) * math.log(math.sqrt(n * rho * rho + 1.0) / self.alpha))
        return m, m - width, m + width, n

    def to_json(self) -> dict[str, Any]:
        return dict(schema="tractable_v1_asymptotic_confidence_sequence_v1", alpha=self.alpha, rho=self._rho(), n_plan=self.n_plan)


def paired_balanced_bce_differences(child_logits: torch.Tensor, dense_logits: torch.Tensor, labels: torch.Tensor,
                                    child_calibration: AffineCalibration, dense_calibration: AffineCalibration) -> torch.Tensor:
    """Per-context contributions ``omega_c * (BCE_child,c - BCE_dense,c)`` whose mean is the dense-relative increment."""

    y = labels.detach().to(torch.float64)
    n_pos = float(y.sum().item()); n_neg = float(y.numel() - n_pos)
    omega = torch.where(y > 0.5, torch.full_like(y, n_neg / n_pos), torch.ones_like(y))
    xc = child_calibration.apply(child_logits.detach().to(y.device, torch.float64))
    xd = dense_calibration.apply(dense_logits.detach().to(y.device, torch.float64))
    sp = torch.nn.functional.softplus
    lc = torch.where(y > 0.5, sp(-xc), sp(xc)); ld = torch.where(y > 0.5, sp(-xd), sp(xd))
    return omega * (lc - ld)
