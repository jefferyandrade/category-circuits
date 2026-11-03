"""The calibrated pruning loss and its frozen target-only census pullback."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
import math
from typing import Any

from mechanistic_systems.contracts.content_address import content_hash


CALIBRATED_OVR_BCE = "calibrated_one_vs_all_bce"
CALIBRATED_SUBSTITUTED_SOFTMAX = "calibrated_substituted_binarized_softmax"
PRUNING_GUIDING_LOSSES = frozenset((CALIBRATED_OVR_BCE, CALIBRATED_SUBSTITUTED_SOFTMAX))


def validate_pruning_guiding_loss_id(loss_id: str) -> str:
    identifier = str(loss_id)
    if identifier not in PRUNING_GUIDING_LOSSES:
        raise ValueError(f"unsupported pruning-guiding census loss: {identifier!r}")
    return identifier


def resolve_pruning_guiding_loss_id(decision_law_id: str) -> str:
    """Read the loss named by the registered decision law; refuse unknown laws."""
    identifier = str(decision_law_id)
    if identifier in PRUNING_GUIDING_LOSSES:
        return identifier
    if identifier in {
        "sequential_calibrated_bce_resolution_v2",
        "sequential_calibrated_bce_resolution_v3",
        "sequential_calibrated_bce_resolution_v4",
        "sequential_calibrated_bce_resolution_v5",
    }:
        return CALIBRATED_OVR_BCE
    raise ValueError(f"unknown pruning-guiding decision loss: {identifier!r}")


def frozen_affine_parameters(fit: Any) -> tuple[float, float, float]:
    """Require an installed calibration before any census backward pass."""
    if fit is None:
        raise RuntimeError("dense calibration must be fitted before the census")
    scale = float(getattr(fit, "scale", float("nan")))
    offset = float(getattr(fit, "offset", float("nan")))
    center = float(getattr(fit, "input_center", 0.0))
    if not all(math.isfinite(value) for value in (scale, offset, center)) or scale <= 0:
        raise ValueError("census loss requires a finite positive frozen affine calibration")
    if not str(getattr(fit, "calibration_state_id", "")).strip():
        raise RuntimeError("dense calibration must have a fitted state identity before the census")
    return scale, offset, center


def dense_calibration_state_hash(fit: Any) -> str:
    """Hash the actual fitted state, including its scale, offset, and center."""
    scale, offset, center = frozen_affine_parameters(fit)
    materialize = getattr(fit, "materialize", None)
    state = materialize() if callable(materialize) else fit
    if is_dataclass(state) or isinstance(state, dict):
        payload = state
    else:
        # Lightweight installed scalar states need the same actual affine
        # binding as dataclass fits, rather than their opaque state ID alone.
        payload = {
            "scale": scale,
            "offset": offset,
            "input_center": center,
            "mask_hash": str(getattr(state, "mask_hash", "")),
            "calibration_state_id": str(state.calibration_state_id),
            "fit_receipt_hash": str(getattr(state, "fit_receipt_hash", "")),
        }
    return content_hash(payload, prefix="irg-dense-census-calibration-")


def census_loss_pullback_hash(
    *, query_id: str, target_class_id: int, fit: Any, loss_id: str
) -> str:
    if loss_id not in PRUNING_GUIDING_LOSSES:
        raise ValueError("unsupported pruning-guiding census loss")
    return content_hash(
        {
            "schema": "frozen_dense_pruning_loss_pullback_v2",
            "query_id": str(query_id),
            "target_class_id": int(target_class_id),
            "pruning_guiding_loss_id": loss_id,
            "dense_calibration_state_hash": dense_calibration_state_hash(fit),
            "competitor_gradient": "zero",
        },
        prefix="irg-loss-pullback-",
    )


def census_loss_path_hash(
    *, query_id: str, target_class_id: int, fit: Any, loss_id: str
) -> str:
    return content_hash(
        {
            "schema": "frozen_dense_pruning_loss_activation_path_v3",
            "endpoint_loss_pullback_hash": census_loss_pullback_hash(
                query_id=query_id, target_class_id=target_class_id, fit=fit, loss_id=loss_id,
            ),
            "path": "unitwise_bias_to_activation_frozen_target_cotangent_v1",
            "target_functional": "z_dense_plus_G_times_h_path_minus_h_dense",
            "loss_integration": "exact_softplus_divided_difference_v1",
        },
        prefix="irg-loss-path-",
    )


@dataclass(frozen=True, slots=True)
class FrozenPruningLossPath:
    """Dense calibrated loss composed with each registered unit activation path."""

    dense_eta: Any
    binary_label: Any
    affine_scale: float

    def __post_init__(self) -> None:
        import torch

        eta, label = self.dense_eta, self.binary_label
        if (
            not isinstance(eta, torch.Tensor) or not isinstance(label, torch.Tensor)
            or eta.ndim != 1 or eta.shape != label.shape
            or eta.dtype != torch.float64 or label.dtype != torch.float64
            or eta.device != label.device or eta.requires_grad or label.requires_grad
            or not math.isfinite(self.affine_scale) or self.affine_scale <= 0
        ):
            raise ValueError("frozen pruning-loss path has an invalid dense axis or affine")
        if not bool((torch.isfinite(eta) & ((label == 0) | (label == 1))).all().item()):
            raise ValueError("frozen pruning-loss path must have finite logits and binary labels")


def frozen_pruning_loss_path(
    *, raw_responses: Any, labels: Any, target_class_id: int,
    fit: Any, loss_id: str, device: Any = None, fixed_offsets: Any = None,
) -> FrozenPruningLossPath:
    """Freeze the dense loss logit, label, and affine derivative for either loss.

    Dense competitor logits have zero gradient and stay fixed along each
    registered unit path. The probability varies with its target functional.
    """
    import torch

    if loss_id not in PRUNING_GUIDING_LOSSES:
        raise ValueError("unsupported pruning-guiding census loss")
    scale, offset, center = frozen_affine_parameters(fit)
    if raw_responses is None or labels is None:
        raise ValueError("census loss pullback requires dense logits and image labels")
    responses = torch.as_tensor(raw_responses, device=device, dtype=torch.float64).detach()
    target = int(target_class_id)
    if responses.ndim != 2 or not 0 <= target < responses.shape[1]:
        raise ValueError("census target differs from the dense response axis")
    eta = (responses[:, target] - center).mul(scale).add(offset)
    if loss_id == CALIBRATED_SUBSTITUTED_SOFTMAX:
        if fixed_offsets is not None:
            row_offsets = torch.as_tensor(fixed_offsets, device=responses.device,
                                          dtype=torch.float64).detach()
            if row_offsets.shape != eta.shape or not bool(torch.isfinite(row_offsets).all().item()):
                raise ValueError("substituted census offsets must cover every decision row")
            eta = eta + row_offsets
        else:
            if responses.shape[1] < 2:
                raise ValueError("substituted loss requires dense competitor logits")
            competitors = torch.cat((responses[:, :target], responses[:, target + 1:]), dim=1)
            eta = eta - torch.logsumexp(competitors, dim=1)
    elif fixed_offsets is not None:
        raise ValueError("target-logit census does not use substituted offsets")
    binary_label = torch.as_tensor(labels, device=responses.device).reshape(-1).to(torch.int64).eq(target).to(torch.float64)
    if binary_label.shape != eta.shape:
        raise ValueError("census loss labels differ from the context axis")
    return FrozenPruningLossPath(eta, binary_label, scale)


def frozen_pruning_loss_multiplier(
    *, raw_responses: Any, labels: Any, target_class_id: int,
    fit: Any, loss_id: str, device: Any = None, fixed_offsets: Any = None,
) -> Any:
    """Return the detached dense endpoint p-y for ordinary census gradients."""
    path = frozen_pruning_loss_path(
        raw_responses=raw_responses, labels=labels, target_class_id=target_class_id,
        fit=fit, loss_id=loss_id, device=device, fixed_offsets=fixed_offsets,
    )
    return path.dense_eta.sigmoid().sub(path.binary_label)
