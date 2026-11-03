"""Float64 sufficient-statistic reducers for affine calibration Newton steps."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

try:  # Optional for import smoke; required by executable calibration.
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - environment dependent.
    torch = F = None  # type: ignore[assignment]


_TRITON_ENABLED = str(
    os.getenv("MECHANISTIC_SYSTEMS_TRITON_IMPORT_ENABLED", "1")
).strip().lower() not in {"0", "false", "no", "off"}
try:  # pragma: no cover - depends on CUDA/Triton runtime.
    if not _TRITON_ENABLED or torch is None or not torch.cuda.is_available():
        raise ImportError("Triton calibration statistics require an active CUDA driver")
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - expected on CPU hosts.
    triton = tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


@dataclass(frozen=True)
class CalibrationStatistics:
    objective: Any
    grad_scale: Any
    grad_offset: Any
    hessian_scale: Any
    hessian_cross: Any
    hessian_offset: Any
    backend: str
    fallback_reason: str = ""

    def values(self) -> tuple[Any, ...]:
        return (
            self.objective,
            self.grad_scale,
            self.grad_offset,
            self.hessian_scale,
            self.hessian_cross,
            self.hessian_offset,
        )


@dataclass(frozen=True)
class StratifiedAffineBCEUncertainty:
    """Device-resident loss-scale uncertainty for one affine BCE fit."""

    parameter_covariance: Any
    allocation_standard_deviations: Any
    risk_gradient: Any
    risk_hessian: Any
    maximum_prediction_radius: Any
    maximum_bce_excess_risk_bound: Any
    first_order_excess_risk_bound: Any
    second_order_excess_risk_bound: Any
    first_order_excess_risk_coefficient: Any
    second_order_excess_risk_coefficient: Any
    sample_counts: Any
    backend: str


def stratified_affine_bce_uncertainty(
    values: Any,
    labels: Any,
    stratum_ids: Any,
    stratum_weights: Any,
    population_counts: Any,
    scale: Any,
    offset: Any,
    *,
    confidence_level: float,
    hessian_ridge: float = 1e-10,
    finite_population_correction: bool = True,
    sample_weights: Any | None = None,
    fixed_offsets: Any | None = None,
) -> StratifiedAffineBCEUncertainty:
    """Reduce affine-fit uncertainty on the input tensor's current device.

    The declared risk is a fixed weighted mixture of finite strata. The
    sandwich covariance therefore uses within-stratum score covariance,
    squared estimand weights, and finite-population corrections. The returned
    loss bound projects the two-parameter confidence ellipsoid through the
    class-balanced BCE gradient and curvature. It is a calibration sample-size
    diagnostic, never pruning commit authority.

    ``fixed_offsets`` adds a known row offset to each fitted logit while
    leaving the affine parameter design unchanged. This covers a target
    affine evaluated against frozen dense softmax competitors.
    """

    if torch is None or F is None:
        raise RuntimeError("affine BCE uncertainty requires PyTorch")
    confidence = float(confidence_level)
    ridge = float(hessian_ridge)
    if not 0.5 < confidence < 1.0:
        raise ValueError("affine BCE uncertainty confidence must lie in (0.5,1)")
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("affine BCE uncertainty ridge must be positive")
    x = torch.as_tensor(values)
    if x.ndim != 1:
        raise ValueError("affine BCE uncertainty values must be one-dimensional")
    device = x.device
    x = x.to(device=device, dtype=torch.float64)
    y = torch.as_tensor(labels, device=device, dtype=torch.float64).reshape(-1)
    strata = torch.as_tensor(
        stratum_ids,
        device=device,
        dtype=torch.long,
    ).reshape(-1)
    weights = torch.as_tensor(
        stratum_weights,
        device=device,
        dtype=torch.float64,
    ).reshape(-1)
    populations = torch.as_tensor(
        population_counts,
        device=device,
        dtype=torch.float64,
    ).reshape(-1)
    if (
        int(x.numel()) != int(y.numel())
        or int(x.numel()) != int(strata.numel())
        or int(weights.numel()) == 0
        or tuple(weights.shape) != tuple(populations.shape)
    ):
        raise ValueError("affine BCE uncertainty axes are misaligned")
    stratum_count = int(weights.numel())
    if (
        bool((strata < 0).any().item())
        or bool((strata >= stratum_count).any().item())
        or not bool(torch.isfinite(x).all().item())
        or not bool(torch.isfinite(y).all().item())
        or not bool(torch.isfinite(weights).all().item())
        or not bool(torch.isfinite(populations).all().item())
        or bool((weights <= 0.0).any().item())
        or bool((populations <= 0.0).any().item())
        or not math.isclose(
            float(weights.sum().item()),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("affine BCE uncertainty design is invalid")
    membership = F.one_hot(
        strata,
        num_classes=stratum_count,
    ).T.to(dtype=torch.float64)
    sample_counts = membership.sum(dim=1)
    if bool((sample_counts < 2.0).any().item()) or (
        bool(finite_population_correction)
        and bool((sample_counts > populations).any().item())
    ):
        raise ValueError(
            "affine BCE uncertainty requires two rows per stratum within capacity"
        )
    fitted_scale = torch.as_tensor(
        scale,
        device=device,
        dtype=torch.float64,
    ).reshape(())
    fitted_offset = torch.as_tensor(
        offset,
        device=device,
        dtype=torch.float64,
    ).reshape(())
    design = torch.stack((x, torch.ones_like(x)), dim=1)
    eta = x * fitted_scale + fitted_offset
    if fixed_offsets is not None:
        fixed = torch.as_tensor(
            fixed_offsets, device=device, dtype=torch.float64
        ).reshape(-1)
        if fixed.shape != x.shape or not bool(torch.isfinite(fixed).all().item()):
            raise ValueError("affine BCE uncertainty fixed offsets are invalid")
        eta = eta + fixed
    probability = torch.sigmoid(eta)
    residual = probability - y
    curvature = probability * (1.0 - probability)
    score = design * residual.unsqueeze(1)
    if sample_weights is not None:
        metric = torch.as_tensor(sample_weights, device=device, dtype=torch.float64).reshape(-1)
        if metric.shape != x.shape or not bool(torch.isfinite(metric).all().item()) or bool((metric < 0.0).any().item()):
            raise ValueError("affine BCE uncertainty metric weights are invalid")
        metric = metric / metric.sum().clamp_min(1e-300)
        # The binary acquisition strata stay fixed. Within each stratum the
        # score and curvature carry the metric's per-class row masses, so
        # both the risk derivative and its sandwich covariance refer to the
        # same weighted BCE as the fit.
        importance = metric * sample_counts[strata] / weights[strata]
        score = score * importance.unsqueeze(1)
        curvature = curvature * importance
    stratum_score_means = membership @ score / sample_counts.unsqueeze(1)
    centered_score = (
        score.unsqueeze(0) - stratum_score_means.unsqueeze(1)
    )
    score_covariance = torch.einsum(
        "hn,hni,hnj->hij",
        membership,
        centered_score,
        centered_score,
    ) / (sample_counts - 1.0).reshape(-1, 1, 1)
    stratum_hessian = torch.einsum(
        "hn,n,ni,nj->hij",
        membership,
        curvature,
        design,
        design,
    ) / sample_counts.reshape(-1, 1, 1)
    risk_hessian = torch.einsum(
        "h,hij->ij",
        weights,
        stratum_hessian,
    ) + ridge * torch.eye(2, device=device, dtype=torch.float64)
    risk_gradient = torch.einsum(
        "h,hi->i",
        weights,
        stratum_score_means,
    )
    inverse_hessian = torch.linalg.pinv(risk_hessian, hermitian=True)
    fpc = (
        torch.clamp(1.0 - sample_counts / populations, min=0.0)
        if bool(finite_population_correction)
        else torch.ones_like(sample_counts)
    )
    gradient_covariance = torch.einsum(
        "h,hij->ij",
        weights.square() * fpc / sample_counts,
        score_covariance,
    )
    covariance = inverse_hessian @ gradient_covariance @ inverse_hessian
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    nonnegative_eigenvalues = torch.clamp(eigenvalues, min=0.0)
    covariance = (
        eigenvectors * nonnegative_eigenvalues.unsqueeze(0)
    ) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    covariance_square_root = (
        eigenvectors * torch.sqrt(nonnegative_eigenvalues).unsqueeze(0)
    ) @ eigenvectors.T
    influence_covariance = torch.einsum(
        "ij,hjk,lk->hil",
        inverse_hessian,
        score_covariance,
        inverse_hessian,
    )
    allocation_standard_deviations = torch.sqrt(
        torch.clamp(
            torch.diagonal(
                influence_covariance,
                dim1=1,
                dim2=2,
            ).sum(dim=1),
            min=0.0,
        )
    )
    ellipsoid_radius_squared = -2.0 * math.log(1.0 - confidence)
    first_order_coefficient = torch.sqrt(
        torch.clamp(
            risk_gradient @ covariance @ risk_gradient,
            min=0.0,
        )
    )
    loss_curvature = (
        covariance_square_root @ risk_hessian @ covariance_square_root
    )
    second_order_coefficient = (
        0.5
        * torch.clamp(torch.linalg.eigvalsh(loss_curvature).max(), min=0.0)
    )
    first_order = math.sqrt(ellipsoid_radius_squared) * first_order_coefficient
    second_order = ellipsoid_radius_squared * second_order_coefficient
    prediction_variance = torch.einsum(
        "ni,ij,nj->n",
        design,
        covariance,
        design,
    )
    normal_critical = NormalDist().inv_cdf(
        1.0 - (1.0 - confidence) / 4.0
    )
    return StratifiedAffineBCEUncertainty(
        parameter_covariance=covariance,
        allocation_standard_deviations=allocation_standard_deviations,
        risk_gradient=risk_gradient,
        risk_hessian=risk_hessian,
        maximum_prediction_radius=normal_critical
        * torch.sqrt(torch.clamp(prediction_variance.max(), min=0.0)),
        maximum_bce_excess_risk_bound=first_order + second_order,
        first_order_excess_risk_bound=first_order,
        second_order_excess_risk_bound=second_order,
        first_order_excess_risk_coefficient=first_order_coefficient,
        second_order_excess_risk_coefficient=second_order_coefficient,
        sample_counts=sample_counts,
        backend=(
            (
                "torch_float64_stratified_affine_bce_uncertainty_"
                f"{device.type}_v1"
            )
            if bool(finite_population_correction)
            else (
                "torch_float64_stratified_affine_bce_uncertainty_"
                f"{device.type}_iid_with_replacement_v1"
            )
        ),
    )


def triton_calibration_statistics_available() -> bool:
    return bool(_TRITON_AVAILABLE)


def torch_calibration_statistics(
    values: Any,
    labels: Any,
    weights: Any,
    scale: Any,
    offset: Any,
) -> CalibrationStatistics:
    """Authoritative vectorized float64 reducer."""

    eta = values * scale.unsqueeze(1) + offset.unsqueeze(1)
    y = (
        labels.unsqueeze(0).expand(int(eta.shape[0]), -1, -1)
        if labels.ndim == 2
        else labels
    )
    w = (
        weights.unsqueeze(0).expand(int(eta.shape[0]), -1, -1)
        if weights.ndim == 2
        else weights
    )
    if tuple(y.shape) != tuple(eta.shape) or tuple(w.shape) != tuple(eta.shape):
        raise ValueError(
            "calibration labels and weights must broadcast exactly over "
            "[mask,sample,query]"
        )
    # softplus(eta) - y*eta is the Bernoulli canonical proper loss for both
    # binary and soft targets.  Keeping this form in the shared reducer lets
    # identity-anchored calibration add predictive-KL pseudo-observations
    # without a second optimization kernel.
    loss = (F.softplus(eta) - y * eta) * w
    probability = torch.sigmoid(eta)
    residual = (probability - y) * w
    curvature = probability * (1.0 - probability) * w
    denominator = float(values.shape[1])
    return CalibrationStatistics(
        objective=loss.sum(dim=1) / denominator,
        grad_scale=(residual * values).sum(dim=1) / denominator,
        grad_offset=residual.sum(dim=1) / denominator,
        hessian_scale=(curvature * values.square()).sum(dim=1) / denominator,
        hessian_cross=(curvature * values).sum(dim=1) / denominator,
        hessian_offset=curvature.sum(dim=1) / denominator,
        backend="torch_float64_vectorized",
    )


if _TRITON_AVAILABLE:  # pragma: no cover - exercised in GPU canaries.

    @triton.jit
    def _affine_statistics_partial_kernel(
        values_ptr,
        labels_ptr,
        weights_ptr,
        scale_ptr,
        offset_ptr,
        partial_ptr,
        N,
        Q,
        BLOCK_N: tl.constexpr,
    ):
        slot = tl.program_id(0)
        tile = tl.program_id(1)
        mask_index = slot // Q
        query_index = slot - mask_index * Q
        scale = tl.load(scale_ptr + slot).to(tl.float64)
        offset = tl.load(offset_ptr + slot).to(tl.float64)
        sample = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = sample < N
        value_offset = mask_index * N * Q + sample * Q + query_index
        x = tl.load(values_ptr + value_offset, mask=valid, other=0.0).to(
            tl.float64
        )
        y = tl.load(labels_ptr + value_offset, mask=valid, other=0.0).to(
            tl.float64
        )
        weight = tl.load(
            weights_ptr + value_offset, mask=valid, other=0.0
        ).to(tl.float64)
        eta = x * scale + offset
        probability = tl.sigmoid(eta)
        curvature = probability * (1.0 - probability) * weight
        residual = (probability - y) * weight
        softplus = tl.where(
            eta >= 0.0,
            eta + tl.log(1.0 + tl.exp(-eta)),
            tl.log(1.0 + tl.exp(eta)),
        )
        row_loss = (softplus - y * eta) * weight
        tile_count = tl.cdiv(N, BLOCK_N)
        output_base = (slot * tile_count + tile) * 6
        tl.store(
            partial_ptr + output_base + 0,
            tl.sum(tl.where(valid, row_loss, 0.0), axis=0),
        )
        tl.store(
            partial_ptr + output_base + 1,
            tl.sum(tl.where(valid, residual * x, 0.0), axis=0),
        )
        tl.store(
            partial_ptr + output_base + 2,
            tl.sum(tl.where(valid, residual, 0.0), axis=0),
        )
        tl.store(
            partial_ptr + output_base + 3,
            tl.sum(tl.where(valid, curvature * x * x, 0.0), axis=0),
        )
        tl.store(
            partial_ptr + output_base + 4,
            tl.sum(tl.where(valid, curvature * x, 0.0), axis=0),
        )
        tl.store(
            partial_ptr + output_base + 5,
            tl.sum(tl.where(valid, curvature, 0.0), axis=0),
        )


def triton_calibration_statistics(
    values: Any,
    labels: Any,
    weights: Any,
    scale: Any,
    offset: Any,
) -> CalibrationStatistics:
    if not triton_calibration_statistics_available():
        raise RuntimeError("Triton calibration statistics are unavailable")
    if not values.is_cuda or values.dtype != torch.float64:
        raise ValueError("Triton calibration statistics require CUDA float64 values")
    mask_count, sample_count, query_count = map(int, values.shape)
    labels = (
        labels.unsqueeze(0).expand(mask_count, -1, -1) if labels.ndim == 2 else labels
    )
    weights = (
        weights.unsqueeze(0).expand(mask_count, -1, -1)
        if weights.ndim == 2
        else weights
    )
    # The kernel indexes mask-major contiguous payloads explicitly. Expanded
    # singleton axes have stride zero and must be materialized before launch.
    labels = labels.contiguous()
    weights = weights.contiguous()
    expected = (mask_count, sample_count, query_count)
    if tuple(labels.shape) != expected or tuple(weights.shape) != expected:
        raise ValueError(
            "Triton calibration labels and weights must match " "[mask,sample,query]"
        )
    block_n = 256
    tile_count = triton.cdiv(sample_count, block_n)
    partials = torch.empty(
        (mask_count * query_count, tile_count, 6),
        device=values.device,
        dtype=torch.float64,
    )
    _affine_statistics_partial_kernel[
        (mask_count * query_count, tile_count)
    ](
        values.contiguous(),
        labels.contiguous(),
        weights.contiguous(),
        scale.contiguous(),
        offset.contiguous(),
        partials,
        sample_count,
        query_count,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    # Every tile owns one float64 partial; the second stage is deterministic
    # and uses no atomic updates.  Prefix length changes only the launch grid,
    # never the compiled kernel identity.
    output = partials.sum(dim=1) / float(sample_count)
    shaped = output.reshape(mask_count, query_count, 6)
    return CalibrationStatistics(
        objective=shaped[:, :, 0],
        grad_scale=shaped[:, :, 1],
        grad_offset=shaped[:, :, 2],
        hessian_scale=shaped[:, :, 3],
        hessian_cross=shaped[:, :, 4],
        hessian_offset=shaped[:, :, 5],
        backend="triton_float64_runtime_length_tiled_statistics_v2",
    )


def calibration_statistics(
    values: Any,
    labels: Any,
    weights: Any,
    scale: Any,
    offset: Any,
    *,
    backend: str = "auto",
) -> CalibrationStatistics:
    token = str(backend).strip().lower().replace("-", "_")
    if token not in {"auto", "torch", "triton"}:
        raise ValueError(f"unsupported calibration statistics backend {backend!r}")
    use_triton = token == "triton" or (
        token == "auto"
        and bool(getattr(values, "is_cuda", False))
        and triton_calibration_statistics_available()
    )
    if use_triton:
        return triton_calibration_statistics(values, labels, weights, scale, offset)
    return torch_calibration_statistics(values, labels, weights, scale, offset)



__all__ = [
    "CalibrationStatistics",
    "StratifiedAffineBCEUncertainty",
    "calibration_statistics",
    "stratified_affine_bce_uncertainty",
    "torch_calibration_statistics",
    "triton_calibration_statistics",
    "triton_calibration_statistics_available",
]
