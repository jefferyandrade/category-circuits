"""Dense-reference statistics retained on the decision device."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np
import torch

from mechanistic_systems.inference import device_bce as db
from mechanistic_systems.circuits.irg.contracts import content_hash
from mechanistic_systems.inference.sequential_bce import (
    BaselineRiskCertificate, CalibrationImprovementProfile, STRATIFIED_T_METHOD,
)


def baseline_summary(counts, means, m2, weights, alpha, u, w):
    """The dense certificate's Student-t interval, without an FPC."""
    n = counts.clamp_min(1.)
    terms = weights.square()*(m2/(n-1.).clamp_min(1.))/n
    mean = (weights*means).sum(-1)
    variance = terms.sum(-1)
    denominator = (terms.square()/(n-1.).clamp_min(1.)).sum(-1)
    df = torch.where(variance > 0.,variance.square()/denominator.clamp_min(1e-300),torch.inf)
    critical = db.student_t_isf(alpha*.5,df,u,w)
    se = variance.clamp_min(0.).sqrt()
    radius = critical*se
    invalid = ((counts < 2).any() | (~torch.isfinite(counts)).any()
        | (~torch.isfinite(means)).any() | (~torch.isfinite(m2)).any() | (m2 < 0.).any()
        | (~torch.isfinite(weights)).any() | (~torch.isfinite(mean))
        | (~torch.isfinite(se)) | (~torch.isfinite(critical)) | (alpha <= 0.) | (alpha >= 1.))
    return torch.stack((mean,se,mean-radius,mean+radius,df,invalid.to(torch.float64)),-1)


SUMMARY = db.CapturedTensorFunction(baseline_summary)


def span_inputs(summary, allowance, empty, delta_tau):
    """Construct device anchors and enforce the loss-span domain."""
    mean,se,lower,upper = summary[:4].unbind()
    estimate = empty[0]-mean
    span_lower = empty[3]-upper
    span_upper = empty[4]-lower+allowance[0]
    invalid = ((summary[5] != 0.) | (~torch.isfinite(estimate))
        | (~torch.isfinite(span_lower)) | (~torch.isfinite(span_upper))
        | (~torch.isfinite(allowance[0])) | (allowance[0] < 0.)
        | (estimate <= 0.) | (span_lower <= 0.) | (span_upper < estimate)
        | ((empty[5]-empty[6]).abs() > 1e-15))
    anchors = torch.stack((mean,se,empty[0],empty[1],empty[2],delta_tau[0],delta_tau[1]))
    return anchors[None,:],invalid.reshape(1)


SPAN_INPUTS = db.CapturedTensorFunction(span_inputs)


def warm(device, stratum_count):
    device = torch.device(device)
    def filled(shape,value):
        return torch.full(shape,value,dtype=torch.float64,device=device)
    size = int(stratum_count)
    u,w = db.quadrature(device)
    SUMMARY(filled((size,),2.),filled((1,size),.25),filled((1,size),1.),
            filled((size,),1./size),filled((1,),.025),u,w)
    SPAN_INPUTS(filled((6,),.25),filled((1,),0.),
        filled((7,),.69),filled((2,),.01))


@dataclass(frozen=True,slots=True)
class DeviceBaselineRiskCertificate(BaselineRiskCertificate):
    """Completed host metadata plus the newly acquired device statistics.

    The inherited numerical fields describe the preceding completed
    certificate until ``materialize`` is requested by the receipt path.
    Decision arithmetic uses ``device_summary`` and ``device_profile``.
    """
    device_summary: object = field(default=None,compare=False,repr=False)
    _record: object = field(default=None,compare=False,repr=False)

    @property
    def device_profile(self):
        if (self.calibration_parameter_dimension != self._record.dimension
                or self.calibration_state_id != self._record.fit_state_id):
            return CalibrationImprovementProfile(
                first_order_coefficient=self.calibration_first_order_coefficient,
                second_order_coefficient=self.calibration_second_order_coefficient,
                parameter_dimension=self.calibration_parameter_dimension)
        return SimpleNamespace(first_order_coefficient=self._record.packet[7],
            second_order_coefficient=self._record.packet[8],
            parameter_dimension=self.calibration_parameter_dimension)

    @property
    def device_allowance(self):
        if (self.calibration_parameter_dimension == self._record.dimension
                and self.calibration_state_id == self._record.fit_state_id):
            return self._record.packet[6:7]
        return self.calibration_improvement_bound

    def materialize(self):
        record = self._record
        with record.lock:
            if not record.record_ready:
                raise RuntimeError("dense certificate requested before the look control completed")
            values = record.host_packet.numpy()
            if bool(values[5]):
                raise ValueError("invalid dense-reference device statistics")
            fields = {name:getattr(self,name) for name in BaselineRiskCertificate.__dataclass_fields__}
            fields.update(estimate=float(values[0]),standard_error=float(values[1]),
                lower=float(values[2]),upper=float(values[3]),
                degrees_of_freedom=None if math.isinf(float(values[4])) else float(values[4]))
            if (self.calibration_parameter_dimension == record.dimension
                    and self.calibration_state_id == record.fit_state_id):
                fields.update(calibration_improvement_bound=float(values[6]),
                    calibration_first_order_coefficient=float(values[7]),
                    calibration_second_order_coefficient=float(values[8]))
            fields["output_identity"] = content_hash({
                "mask_hash":self.mask_hash,"context_ids":record.metadata["context_ids"],
                "strata":record.metadata["strata"],"alpha":record.metadata["alpha"],
                "summary":{name:fields[name] for name in
                    ("estimate","lower","upper","standard_error","degrees_of_freedom")},
                "calibration_state_id":self.calibration_state_id,
                "oracle_rows_reexecuted":False},prefix="irg-dense-baseline-risk-output-")
            certificate = BaselineRiskCertificate(**fields)
            if not record.published:
                record.publish(certificate)
                record.published = True
            return certificate

    @property
    def content_hash(self):
        return self.materialize().content_hash


def materialize_span(span):
    if isinstance(span.dense,DeviceBaselineRiskCertificate):
        return replace(span,dense=span.dense.materialize())
    return span


def enqueue_certificate(provider, *, mask, fit, response, context_ids, strata,
                        alpha, confidence_level, confidence_scope, look_index,
                        sampling_design_id, data_role, method, previous=None,
                        target_loss_precision=None):
    """Enqueue dense rows, moments, interval and its asynchronous record copy."""
    from .device_bce_provider import _moment_lane
    names = tuple(provider._oracle_statistical_strata)
    position = {name:i for i,name in enumerate(names)}
    codes = np.fromiter((position[name] for name in strata),dtype=np.int64,count=len(strata))
    folds = (provider._calibration_fold_index_tensor(context_ids,role="proposal_adjudication")
             if mask.mask_hash in provider._calibration_fold_fits else None)
    rows = provider._binary_bce_device_rows_for_mask(response,mask_hash=mask.mask_hash,
        fit=fit,fold_index_per_row=folds,target_class_id=provider.target_class_id)
    transfers = getattr(provider,"_device_baseline_transfers",None)
    if transfers is None:
        transfers = provider._device_baseline_transfers = db.HostTransferBuffer(rows.device)
    n,mean,m2 = _moment_lane(provider,rows.device,len(names)).reduce(rows,
        transfers.copy("codes",codes,dtype=torch.int64))
    retained = getattr(provider, "_baseline_loss_moments_by_mask", None)
    if retained is None:
        retained = provider._baseline_loss_moments_by_mask = {}
    retained[mask.mask_hash] = {
        "calibration_state_id": fit.calibration_state_id, "names": names,
        "counts": n.detach(), "m2": m2.detach(),
    }
    u,w = db.quadrature(rows.device)
    summary = SUMMARY(n,mean,m2,
        transfers.copy("weights",[provider._oracle_statistical_weights[name] for name in names]),
        transfers.copy("alpha",[alpha]),u,w).reshape(-1)
    db.defer_device_check(summary[5] != 0.,lambda:ValueError("invalid dense-reference device statistics"))
    uncertainty = getattr(fit,"device_uncertainty",None)
    coefficient_names = ("maximum_bce_excess_risk_bound","first_order_excess_risk_coefficient",
                         "second_order_excess_risk_coefficient")
    coefficients = torch.stack(tuple(transfers.copy(f"coefficient_{i}",
        uncertainty[name] if uncertainty is not None else getattr(fit,name)).reshape(())
        for i,name in enumerate(coefficient_names)))
    packet = torch.cat((summary,coefficients))
    host_packet = torch.empty(packet.shape,dtype=packet.dtype,pin_memory=rows.device.type == "cuda")
    host_packet.copy_(packet.detach(),non_blocking=True)
    identity = f"{fit.calibration_state_id}:{method}:{look_index}"
    metadata = {"context_ids":tuple(context_ids),"strata":tuple(strata),"alpha":float(alpha),
        "confidence_level":float(confidence_level),"confidence_scope":str(confidence_scope),
        "sampling_design_id":str(sampling_design_id),"data_role":str(data_role),"look_index":int(look_index)}
    provider._baseline_risk_evidence_by_mask[mask.mask_hash] = metadata
    previous_by_mask = getattr(provider,"_device_baseline_previous_by_mask",None)
    if previous_by_mask is None:
        previous_by_mask = provider._device_baseline_previous_by_mask = {}
    def publish(certificate):
        if provider._baseline_risk_evidence_by_mask.get(mask.mask_hash) is metadata:
            previous_by_mask[mask.mask_hash] = certificate
        if method != "certify_baseline_risk":
            return
        refinement = {"schema":"dense_baseline_calibration_state_receipt_v2",
            "precision_policy":"q_free_attainable_improvement_bound_joint_allocation_v1",
            "target_loss_precision":target_loss_precision,"preliminary_calibration_state_id":fit.calibration_state_id,
            "final_calibration_state_id":certificate.calibration_state_id,
            "preliminary_bce_estimate":certificate.estimate,"final_bce_estimate":certificate.estimate,
            "calibration_refined":False,"oracle_contexts_reexecuted":False,"oracle_logits_reused_on_device":True}
        refinement["content_hash"] = content_hash(refinement,
            prefix="irg-dense-baseline-calibration-refinement-")
        provider._baseline_calibration_refinement_by_mask[mask.mask_hash] = refinement
        if provider.progress_callback is not None:
            provider.progress_callback("dense_baseline_calibration_refinement_completed",**refinement)
    record = SimpleNamespace(packet=packet,host_packet=host_packet,
        record_ready=rows.device.type != "cuda",lock=threading.Lock(),published=False,publish=publish,
        dimension=fit.uncertainty_parameter_dimension,fit_state_id=fit.calibration_state_id,
        metadata=metadata)
    pending = getattr(provider,"_pending_device_fit_records",None)
    if pending is None:
        pending = provider._pending_device_fit_records = []
    pending.append(record)
    fields = ({name:getattr(previous,name) for name in BaselineRiskCertificate.__dataclass_fields__}
              if previous is not None else dict(estimate=0.,lower=0.,upper=0.,standard_error=0.,
                context_count=len(context_ids),calibration_state_id=fit.calibration_state_id,
                sampling_design_id=sampling_design_id,data_role=data_role,code_identity=identity,
                output_identity=identity,mask_hash=mask.mask_hash))
    fields.update(context_count=len(context_ids),calibration_state_id=fit.calibration_state_id,
        sampling_design_id=sampling_design_id,data_role=data_role,
        code_identity=content_hash({"provider":type(provider).__name__,"method":method,
            "inference":STRATIFIED_T_METHOD},prefix="irg-dense-baseline-risk-code-"),output_identity=identity,
        confidence_level=float(confidence_level),confidence_scope=confidence_scope,look_index=int(look_index),
        mask_hash=mask.mask_hash,calibration_improvement_bound=0.,calibration_first_order_coefficient=0.,
        calibration_second_order_coefficient=0.,calibration_parameter_dimension=record.dimension)
    certificate = DeviceBaselineRiskCertificate(**fields,device_summary=summary,_record=record)
    previous_by_mask[mask.mask_hash] = certificate
    return certificate


__all__ = ["DeviceBaselineRiskCertificate","enqueue_certificate","materialize_span","warm","SPAN_INPUTS"]
