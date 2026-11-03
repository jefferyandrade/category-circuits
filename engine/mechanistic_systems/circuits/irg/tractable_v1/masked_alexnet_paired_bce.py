"""MaskedAlexNet row provider for the old paired-BCE owner.

This module opens no population/sampling/statistical authority. It computes
actual paired-BCE rows for a frozen old dynamic request from caller-issued
context tensors and caller-supplied old lineage identities.

The default live check is lease-based and cheap: tensor object, storage,
metadata, and PyTorch version counters are checked around source access and
forward execution. Full tensor byte hashing is an epoch-boundary audit only.
It catches ordinary PyTorch in-place updates and tensor/list rebinding, but it
is not a hostile-memory guarantee against raw storage writes that bypass
version counters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Callable

import numpy as np
import torch

from mechanistic_systems.circuits.irg import modern_v1_paired_bce_acquisition as old
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.circuits.irg.tractable_v1.alexnet_factors import (
    MaskedAlexNet,
    raw_target_logits,
)
from mechanistic_systems.circuits.irg.tractable_v1.evaluation import AffineCalibration


@dataclass(frozen=True)
class ContextTensorBatch:
    """Externally issued context tensors for one frozen request slice."""

    context_ids: tuple[str, ...]
    images: torch.Tensor
    labels: torch.Tensor
    source_batch_receipt_hash: str

    def __post_init__(self) -> None:
        context_ids = tuple(str(value) for value in self.context_ids)
        if not context_ids or len(context_ids) != len(set(context_ids)):
            raise ValueError("context tensor batch must contain unique context IDs")
        images = torch.as_tensor(self.images)
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError("context tensor batch images must be [B,3,224,224]")
        labels = torch.as_tensor(self.labels).reshape(-1).to(torch.float64)
        if labels.numel() != len(context_ids) or images.shape[0] != len(context_ids):
            raise ValueError("context tensor batch length mismatch")
        if not bool(torch.isfinite(labels).all()) or not bool(((labels == 0.0) | (labels == 1.0)).all()):
            raise ValueError("context tensor batch labels must be binary")
        if not self.source_batch_receipt_hash:
            raise ValueError("context tensor batch requires source receipt hash from the data owner")
        object.__setattr__(self, "context_ids", context_ids)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "source_batch_receipt_hash", str(self.source_batch_receipt_hash))


ContextSource = Callable[[str, tuple[str, ...]], ContextTensorBatch]


@dataclass(frozen=True)
class TensorLease:
    name: str
    object_id: int
    storage_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    version: int


def _finite_positive_calibration(calibration: AffineCalibration, *, name: str) -> AffineCalibration:
    if type(calibration) is not AffineCalibration:
        raise TypeError(f"{name} must be an exact tractable AffineCalibration")
    if type(calibration.positive_scale) is not bool:
        raise TypeError(f"{name}.positive_scale must be bool")
    if (
        not calibration.positive_scale
        or not math.isfinite(float(calibration.scale))
        or not math.isfinite(float(calibration.offset))
        or float(calibration.scale) <= 0.0
    ):
        raise ValueError(f"{name} must have finite positive scale and finite offset")
    if not math.isfinite(float(calibration.loss)):
        raise ValueError(f"{name}.loss must be finite")
    return calibration


def _storage_ptr(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except AttributeError:  # pragma: no cover - old torch fallback.
        return int(tensor.storage().data_ptr())


def _tensor_lease(tensor: torch.Tensor, *, name: str) -> TensorLease:
    if type(tensor) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    return TensorLease(
        name=name,
        object_id=id(tensor),
        storage_ptr=_storage_ptr(tensor),
        shape=tuple(int(v) for v in tensor.shape),
        stride=tuple(int(v) for v in tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        requires_grad=bool(tensor.requires_grad),
        version=int(tensor._version),
    )


def _check_tensor_lease(tensor: torch.Tensor, lease: TensorLease) -> None:
    current = _tensor_lease(tensor, name=lease.name)
    if current != lease:
        raise ValueError(f"{lease.name} changed after row epoch binding")


def _batch_tensor_leases(batch: "ContextTensorBatch") -> tuple[TensorLease, TensorLease]:
    return (
        _tensor_lease(batch.images, name="context_batch.images"),
        _tensor_lease(batch.labels, name="context_batch.labels"),
    )


def _check_batch_tensor_leases(
    batch: "ContextTensorBatch",
    leases: tuple[TensorLease, TensorLease],
) -> None:
    _check_tensor_lease(batch.images, leases[0])
    _check_tensor_lease(batch.labels, leases[1])


def _assert_structure_snapshot(net: MaskedAlexNet, structural_fingerprint: str) -> None:
    if _structural_fingerprint(net) != structural_fingerprint:
        raise ValueError("MaskedAlexNet row epoch structural metadata changed after binding")


def _assert_model_snapshot(
    net: MaskedAlexNet,
    structural_fingerprint: str,
    model_leases: tuple[TensorLease, ...],
) -> None:
    _assert_structure_snapshot(net, structural_fingerprint)
    _check_model_leases(net, model_leases)


def _tensor_content_digest(tensor: torch.Tensor, *, name: str) -> str:
    value = torch.as_tensor(tensor).detach().contiguous().cpu()
    if not bool(torch.isfinite(value.to(torch.float64)).all()):
        raise ValueError(f"{name} contains nonfinite values")
    digest = hashlib.sha256()
    digest.update(b"tractable-v1-row-provider-tensor-content-v1\0")
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(int(v) for v in value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _model_tensor_items(net: MaskedAlexNet) -> tuple[tuple[str, torch.Tensor], ...]:
    if type(net) is not MaskedAlexNet:
        raise TypeError("row provider requires a MaskedAlexNet")
    items: list[tuple[str, torch.Tensor]] = []
    items.extend((f"weight:{i}", t) for i, t in enumerate(net.weights))
    items.extend((f"bias:{i}", t) for i, t in enumerate(net.biases))
    items.extend((f"dense_bias:{i}", t) for i, t in enumerate(net._dense_biases))
    items.extend((
        ("readout_weight", net.readout_weight),
        ("readout_bias", net.readout_bias),
        ("dense_readout_bias", net._dense_readout_bias),
    ))
    return tuple(items)


def _model_content_digest_once(net: MaskedAlexNet) -> str:
    return content_hash(
        {
            "schema": "tractable_v1_masked_alexnet_row_provider_model_content_audit_v1",
            "axis": net.axis.to_json(),
            "tensor_digests": tuple((name, _tensor_content_digest(t, name=name)) for name, t in _model_tensor_items(net)),
            "device": str(net.device),
            "dtype": str(net.dtype),
        },
        prefix="tractable-v1-row-provider-model-content-audit-",
    )


def _model_leases(net: MaskedAlexNet) -> tuple[TensorLease, ...]:
    return tuple(_tensor_lease(t, name=name) for name, t in _model_tensor_items(net))


def _check_model_leases(net: MaskedAlexNet, leases: tuple[TensorLease, ...]) -> None:
    items = _model_tensor_items(net)
    if len(items) != len(leases):
        raise ValueError("MaskedAlexNet row epoch model/bias tensor set changed after binding")
    for (name, tensor), lease in zip(items, leases, strict=True):
        if name != lease.name:
            raise ValueError("MaskedAlexNet row epoch model/bias tensor order changed after binding")
        try:
            _check_tensor_lease(tensor, lease)
        except ValueError as exc:
            raise ValueError(
                f"MaskedAlexNet row epoch model/bias tensor lease changed: {lease.name}"
            ) from exc


def _route_axis_hash(net: MaskedAlexNet) -> str:
    return content_hash(net.axis.to_json(), prefix="tractable-v1-alexnet-route-axis-")


def _structural_fingerprint(net: MaskedAlexNet) -> str:
    if type(net) is not MaskedAlexNet:
        raise TypeError("row provider requires a MaskedAlexNet")
    return content_hash(
        {
            "schema": "tractable_v1_masked_alexnet_row_provider_structure_lease_v1",
            "axis": net.axis.to_json(),
            "pool_after": tuple(sorted(str(value) for value in net.pool_after)),
            "device": str(net.device),
            "dtype": str(net.dtype),
            "weight_count": len(net.weights),
            "bias_count": len(net.biases),
            "dense_bias_count": len(net._dense_biases),
        },
        prefix="tractable-v1-row-provider-structure-lease-",
    )


def _mask_content_digest_once(mask: torch.Tensor, *, axis_hash: str, name: str) -> str:
    value = torch.as_tensor(mask).detach().to(torch.float32).reshape(-1).cpu()
    if not bool(((value == 0.0) | (value == 1.0)).all()):
        raise ValueError(f"{name} must be a binary hard mask")
    packed = np.packbits(value.numpy().astype(np.uint8).reshape(-1)).tobytes()
    return content_hash(
        {
            "schema": "tractable_v1_route_vector_mask_row_provider_content_audit_v1",
            "axis_hash": axis_hash,
            "route_count": int(value.numel()),
            "packed_bits": packed,
        },
        prefix="tractable-v1-row-provider-mask-content-audit-",
    )


def _calibration_fingerprint(calibration: AffineCalibration, *, name: str) -> str:
    calibration = _finite_positive_calibration(calibration, name=name)
    return content_hash(
        {
            "schema": "tractable_v1_affine_calibration_row_provider_fingerprint_v1",
            "scale_hex": float(calibration.scale).hex(),
            "offset_hex": float(calibration.offset).hex(),
            "loss_hex": float(calibration.loss).hex(),
            "iterations": int(calibration.iterations),
            "positive_scale": calibration.positive_scale,
        },
        prefix="tractable-v1-row-provider-calibration-fingerprint-",
    )


def per_context_bce_increment_rows(
    *,
    candidate_logits: torch.Tensor,
    dense_logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_calibration: AffineCalibration,
    dense_calibration: AffineCalibration,
) -> torch.Tensor:
    """Unweighted per-context BCE increments; old owner applies stratum weights."""

    candidate_calibration = _finite_positive_calibration(candidate_calibration, name="candidate_calibration")
    dense_calibration = _finite_positive_calibration(dense_calibration, name="dense_calibration")
    child_raw = torch.as_tensor(candidate_logits)
    dense_raw = torch.as_tensor(dense_logits)
    if child_raw.device != dense_raw.device:
        raise ValueError("candidate and dense logits must be on the same device")
    device = child_raw.device
    y = torch.as_tensor(labels, dtype=torch.float64, device=device).reshape(-1)
    child = candidate_calibration.apply(child_raw.to(torch.float64).reshape(-1))
    dense = dense_calibration.apply(dense_raw.to(torch.float64).reshape(-1))
    if child.numel() != y.numel() or dense.numel() != y.numel():
        raise ValueError("paired-BCE row logits and labels are misaligned")
    if not bool(torch.isfinite(child).all()) or not bool(torch.isfinite(dense).all()):
        raise ValueError("paired-BCE row logits are nonfinite")
    if not bool(((y == 0.0) | (y == 1.0)).all()):
        raise ValueError("paired-BCE row labels must be binary")
    sp = torch.nn.functional.softplus
    candidate_loss = torch.where(y > 0.5, sp(-child), sp(child))
    dense_loss = torch.where(y > 0.5, sp(-dense), sp(dense))
    rows = candidate_loss - dense_loss
    if not bool(torch.isfinite(rows).all()):
        raise ValueError("paired-BCE rows are nonfinite")
    return rows.to(torch.float64)


@dataclass(frozen=True)
class MaskedAlexNetPairedBCEEpoch:
    """Row-only immutable epoch binding for one issued dynamic old registration."""

    registration: old.V1PairedBCEAcquisitionRegistration
    net: MaskedAlexNet
    candidate_mask: torch.Tensor
    dense_mask: torch.Tensor
    candidate_calibration: AffineCalibration
    dense_calibration: AffineCalibration
    row_provider_identity_hash: str
    paired_loss_evaluator_identity_hash: str
    dense_reference_hash: str
    candidate_mask_hash: str
    dense_affine_calibration_hash: str
    candidate_affine_calibration_hash: str
    target_class: int
    context_source_identity_hash: str
    preprocessing_identity_hash: str
    model_content_audit_hash: str = field(init=False)
    structural_fingerprint: str = field(init=False)
    model_leases: tuple[TensorLease, ...] = field(init=False, repr=False)
    candidate_mask_content_audit_hash: str = field(init=False)
    dense_mask_content_audit_hash: str = field(init=False)
    candidate_mask_lease: TensorLease = field(init=False, repr=False)
    dense_mask_lease: TensorLease = field(init=False, repr=False)
    candidate_calibration_fingerprint: str = field(init=False)
    dense_calibration_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.registration) is not old.V1PairedBCEAcquisitionRegistration:
            raise TypeError("row epoch requires the old paired-BCE registration")
        self.registration.assert_integrity()
        dynamic = self.registration.dynamic_oracle_receipt
        if dynamic is None:
            raise ValueError("MaskedAlexNet bridge requires a dynamic old registration with affine calibration hashes")
        dynamic.assert_integrity()
        expected = {
            "dense_reference_hash": self.registration.dense_reference_hash,
            "candidate_mask_hash": self.registration.candidate_mask_hash,
            "dense_affine_calibration_hash": dynamic.dense_affine_calibration_hash,
            "candidate_affine_calibration_hash": dynamic.candidate_affine_calibration_hash,
        }
        for name, value in expected.items():
            supplied = str(getattr(self, name))
            if supplied != value:
                raise ValueError(f"{name} differs from the issued old dynamic registration")
            object.__setattr__(self, name, supplied)
        for name in (
            "row_provider_identity_hash",
            "paired_loss_evaluator_identity_hash",
            "context_source_identity_hash",
            "preprocessing_identity_hash",
        ):
            supplied = str(getattr(self, name))
            if not supplied:
                raise ValueError(f"{name} must be supplied by its owning source")
            object.__setattr__(self, name, supplied)
        _finite_positive_calibration(self.candidate_calibration, name="candidate_calibration")
        _finite_positive_calibration(self.dense_calibration, name="dense_calibration")
        structural_fingerprint = _structural_fingerprint(self.net)
        model_leases = _model_leases(self.net)
        _assert_model_snapshot(self.net, structural_fingerprint, model_leases)
        axis_hash = _route_axis_hash(self.net)
        candidate = self.net._forward_mask(self.candidate_mask, require_binary=True).clone().detach()
        dense = self.net._forward_mask(self.dense_mask, require_binary=True).clone().detach()
        _assert_model_snapshot(self.net, structural_fingerprint, model_leases)
        if not bool((dense == 1.0).all()):
            raise ValueError("dense_reference_hash may bind only an all-live dense mask")
        if not bool((candidate <= dense).all()):
            raise ValueError("candidate mask must be a removal-only subset of the dense mask")
        candidate_lease = _tensor_lease(candidate, name="candidate_mask")
        dense_lease = _tensor_lease(dense, name="dense_mask")
        model_content_audit_hash = _model_content_digest_once(self.net)
        _assert_model_snapshot(self.net, structural_fingerprint, model_leases)
        candidate_mask_content_audit_hash = _mask_content_digest_once(candidate, axis_hash=axis_hash, name="candidate_mask")
        _check_tensor_lease(candidate, candidate_lease)
        dense_mask_content_audit_hash = _mask_content_digest_once(dense, axis_hash=axis_hash, name="dense_mask")
        _check_tensor_lease(dense, dense_lease)
        _assert_model_snapshot(self.net, structural_fingerprint, model_leases)
        _check_tensor_lease(candidate, candidate_lease)
        _check_tensor_lease(dense, dense_lease)
        object.__setattr__(self, "candidate_mask", candidate)
        object.__setattr__(self, "dense_mask", dense)
        object.__setattr__(self, "model_content_audit_hash", model_content_audit_hash)
        object.__setattr__(self, "structural_fingerprint", structural_fingerprint)
        object.__setattr__(self, "model_leases", model_leases)
        object.__setattr__(self, "candidate_mask_content_audit_hash", candidate_mask_content_audit_hash)
        object.__setattr__(self, "dense_mask_content_audit_hash", dense_mask_content_audit_hash)
        object.__setattr__(self, "candidate_mask_lease", candidate_lease)
        object.__setattr__(self, "dense_mask_lease", dense_lease)
        object.__setattr__(self, "candidate_calibration_fingerprint", _calibration_fingerprint(self.candidate_calibration, name="candidate_calibration"))
        object.__setattr__(self, "dense_calibration_fingerprint", _calibration_fingerprint(self.dense_calibration, name="dense_calibration"))
        if type(self.target_class) is not int or not 0 <= self.target_class < self.net.readout_bias.numel():
            raise ValueError("target_class is outside the readout axis")
        self.assert_live()

    def assert_live(self) -> None:
        _assert_model_snapshot(self.net, self.structural_fingerprint, self.model_leases)
        _check_tensor_lease(self.candidate_mask, self.candidate_mask_lease)
        _check_tensor_lease(self.dense_mask, self.dense_mask_lease)
        if _calibration_fingerprint(self.candidate_calibration, name="candidate_calibration") != self.candidate_calibration_fingerprint:
            raise ValueError("candidate calibration changed after row epoch binding")
        if _calibration_fingerprint(self.dense_calibration, name="dense_calibration") != self.dense_calibration_fingerprint:
            raise ValueError("dense calibration changed after row epoch binding")

    def audit_epoch_content(self) -> None:
        """Optional diagnostic full-byte audit for epoch boundaries, not per chunk."""

        axis_hash = _route_axis_hash(self.net)
        if _model_content_digest_once(self.net) != self.model_content_audit_hash:
            raise ValueError("MaskedAlexNet row epoch model content byte audit changed")
        if _mask_content_digest_once(self.candidate_mask, axis_hash=axis_hash, name="candidate_mask") != self.candidate_mask_content_audit_hash:
            raise ValueError("candidate mask content byte audit changed")
        if _mask_content_digest_once(self.dense_mask, axis_hash=axis_hash, name="dense_mask") != self.dense_mask_content_audit_hash:
            raise ValueError("dense mask content byte audit changed")


@dataclass(frozen=True)
class MaskedAlexNetPairedBCERowProvider:
    epoch: MaskedAlexNetPairedBCEEpoch
    context_source: ContextSource
    forward_batch: int = 64
    use_dense_biases_for_reference: bool = True

    def __post_init__(self) -> None:
        if type(self.epoch) is not MaskedAlexNetPairedBCEEpoch:
            raise TypeError("row provider requires a MaskedAlexNetPairedBCEEpoch")
        self.epoch.assert_live()
        if type(self.forward_batch) is not int or self.forward_batch <= 0:
            raise ValueError("forward_batch must be a positive integer")
        if type(self.use_dense_biases_for_reference) is not bool:
            raise TypeError("use_dense_biases_for_reference must be bool")
        if not self.use_dense_biases_for_reference:
            raise ValueError("dense_reference_hash requires original dense biases for the reference forward")

    @property
    def net(self) -> MaskedAlexNet:
        return self.epoch.net

    @property
    def target_class(self) -> int:
        return self.epoch.target_class

    def __call__(
        self,
        *,
        request: old.V1PairedBCEChunkRequest,
        stratum_id: str,
        prefix_start: int,
        prefix_stop: int,
        context_ids_in_registered_slice: tuple[str, ...],
    ) -> tuple[tuple[float, ...], str]:
        epoch = self.epoch
        if type(request) is not old.V1PairedBCEChunkRequest:
            raise TypeError("MaskedAlexNet paired-BCE bridge requires the old canonical request")
        request.assert_integrity()
        epoch.assert_live()
        if request.registration_hash != epoch.registration.registration_hash:
            raise ValueError("request names another old paired-BCE registration")
        if request.dynamic_oracle_receipt_hash != epoch.registration.dynamic_oracle_receipt_hash:
            raise ValueError("request names another dynamic oracle receipt")
        if request.row_provider_identity_hash != epoch.row_provider_identity_hash:
            raise ValueError("request names another row provider")
        if request.paired_loss_evaluator_identity_hash != epoch.paired_loss_evaluator_identity_hash:
            raise ValueError("request names another paired-loss evaluator")
        if stratum_id != request.stratum_id or prefix_start != request.prefix_start or prefix_stop != request.prefix_stop:
            raise ValueError("callback coordinates differ from the frozen request")
        contexts = tuple(str(value) for value in context_ids_in_registered_slice)
        if len(contexts) != request.row_count:
            raise ValueError("context slice length differs from request row count")
        observed_slice_hash = old._context_slice_hash(  # Preserve the old request-slice hash domain.
            registration_hash=request.registration_hash,
            stratum_id=request.stratum_id,
            prefix_start=request.prefix_start,
            context_ids=contexts,
        )
        if observed_slice_hash != request.context_slice_hash:
            raise ValueError("context IDs differ from the frozen request slice hash")
        batch = self.context_source(stratum_id, contexts)
        if type(batch) is not ContextTensorBatch:
            raise TypeError("context_source must return ContextTensorBatch")
        batch_leases = _batch_tensor_leases(batch)
        epoch.assert_live()
        _check_batch_tensor_leases(batch, batch_leases)
        if batch.context_ids != contexts:
            raise ValueError("context source returned a different ordered context axis")
        if stratum_id == old.TARGET_STRATUM:
            if not bool((batch.labels == 1.0).all()):
                raise ValueError("target stratum returned non-target labels")
        elif stratum_id == old.COMPLEMENT_STRATUM:
            if not bool((batch.labels == 0.0).all()):
                raise ValueError("complement stratum returned target labels")
        else:
            raise ValueError("unknown paired-BCE stratum")
        _check_batch_tensor_leases(batch, batch_leases)
        epoch.assert_live()
        candidate_logits = raw_target_logits(
            epoch.net,
            batch.images,
            target_class=epoch.target_class,
            batch_size=self.forward_batch,
            mask_override=epoch.candidate_mask,
        )
        _check_batch_tensor_leases(batch, batch_leases)
        epoch.assert_live()
        dense_logits = raw_target_logits(
            epoch.net,
            batch.images,
            target_class=epoch.target_class,
            batch_size=self.forward_batch,
            mask_override=epoch.dense_mask,
            use_dense_biases=self.use_dense_biases_for_reference,
        )
        _check_batch_tensor_leases(batch, batch_leases)
        epoch.assert_live()
        rows_tensor = per_context_bce_increment_rows(
            candidate_logits=candidate_logits,
            dense_logits=dense_logits,
            labels=batch.labels,
            candidate_calibration=epoch.candidate_calibration,
            dense_calibration=epoch.dense_calibration,
        )
        _check_batch_tensor_leases(batch, batch_leases)
        rows = tuple(float(value) for value in rows_tensor.detach().cpu().reshape(-1).tolist())
        _check_batch_tensor_leases(batch, batch_leases)
        receipt_hash = content_hash(
            {
                "schema": "tractable_v1_masked_alexnet_paired_bce_row_execution_receipt_v4",
                "request_hash": request.request_hash,
                "registration_hash": epoch.registration.registration_hash,
                "dynamic_oracle_receipt_hash": epoch.registration.dynamic_oracle_receipt_hash,
                "context_slice_hash": request.context_slice_hash,
                "source_batch_receipt_hash": batch.source_batch_receipt_hash,
                "context_source_identity_hash": epoch.context_source_identity_hash,
                "preprocessing_identity_hash": epoch.preprocessing_identity_hash,
                "row_provider_identity_hash": epoch.row_provider_identity_hash,
                "paired_loss_evaluator_identity_hash": epoch.paired_loss_evaluator_identity_hash,
                "dense_reference_hash": epoch.dense_reference_hash,
                "candidate_mask_hash": epoch.candidate_mask_hash,
                "dense_affine_calibration_hash": epoch.dense_affine_calibration_hash,
                "candidate_affine_calibration_hash": epoch.candidate_affine_calibration_hash,
                "model_content_audit_hash": epoch.model_content_audit_hash,
                "structural_fingerprint": epoch.structural_fingerprint,
                "candidate_mask_content_audit_hash": epoch.candidate_mask_content_audit_hash,
                "dense_mask_content_audit_hash": epoch.dense_mask_content_audit_hash,
                "candidate_calibration_fingerprint": epoch.candidate_calibration_fingerprint,
                "dense_calibration_fingerprint": epoch.dense_calibration_fingerprint,
                "live_check_scope": "epoch_and_batch_tensor_object_storage_metadata_version_checked_before_source_after_source_after_each_forward_after_rows_before_return",
                "hostile_raw_storage_write_scope": "not_detected_without_explicit_epoch_boundary_byte_audit",
                "target_class": int(epoch.target_class),
                "stratum_id": stratum_id,
                "prefix_start": prefix_start,
                "prefix_stop": prefix_stop,
                "context_ids_in_registered_slice": contexts,
                "paired_loss_rows_hex": tuple(value.hex() for value in rows),
                "forward_batch": self.forward_batch,
                "use_dense_biases_for_reference": self.use_dense_biases_for_reference,
                "authority": "row_execution_only",
            },
            prefix="tractable-v1-masked-alexnet-paired-bce-row-execution-",
        )
        _check_batch_tensor_leases(batch, batch_leases)
        epoch.assert_live()
        return rows, receipt_hash


__all__ = [
    "ContextTensorBatch",
    "ContextSource",
    "MaskedAlexNetPairedBCEEpoch",
    "MaskedAlexNetPairedBCERowProvider",
    "TensorLease",
    "per_context_bce_increment_rows",
]
