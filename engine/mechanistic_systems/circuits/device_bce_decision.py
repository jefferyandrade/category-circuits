"""v5 controller with device statistics and deferred receipt materialization."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import torch

from mechanistic_systems.inference import device_bce as db
from mechanistic_systems.inference.sequential_bce import (
    ScalarV0DecisionEnvelope, SCALAR_DECISION_FIELLER_METHOD,
    OptimizedCalibrationDecisionEnvelope, fieller_ratio_interval,
)


class DeviceBCEExecutionError(RuntimeError):
    """A failed device decision ends the run; its draws cannot be replayed."""

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(
            f"device BCE decision failed: query={receipt['query_id']!r}, "
            f"layer={receipt['layer_id']!r}, partition={receipt['partition_id']!r}, "
            f"proposal={receipt['proposal_index']}, look={receipt['look_index']} "
            f"during {receipt['phase']}: {receipt['cause_type']}: {receipt['cause']}"
        )


@dataclass
class _DeviceExecution:
    query_id: str
    layer_id: str
    partition_id: str
    proposal_index: int
    progress_callback: object
    look_index: int = 0
    phase: str = 'preparation'
    refused_error: object = None
    look_timer: object = None
    finish_host_timing: object = None

    def failure(self, cause):
        receipt = {
            'schema': 'device_bce_execution_failure_v1',
            'status': 'failed', 'reason': 'device_execution_failed',
            'query_id': str(self.query_id), 'layer_id': str(self.layer_id),
            'partition_id': str(self.partition_id),
            'proposal_index': int(self.proposal_index), 'look_index': self.look_index,
            'phase': self.phase, 'cause_type': type(cause).__name__, 'cause': str(cause),
        }
        failure = DeviceBCEExecutionError(receipt)
        if self.progress_callback is not None:
            try:
                self.progress_callback('bce_device_execution_failed', **receipt)
            except Exception as error:
                failure.add_note(f"failure receipt could not be recorded: {type(error).__name__}: {error}")
        return failure


class _LookTiming:
    """Host timing around one enqueued acquisition and its control event."""

    def __init__(self, execution, device, *, kind="oracle"):
        self.row = {"query_id":execution.query_id,"layer_id":execution.layer_id,
            "proposal_index":execution.proposal_index,"look_index":execution.look_index,
            "acquisition":kind}
        self.cuda = isinstance(device,torch.device) and device.type == "cuda"
        self.look_open = self.enqueue_open = True
        self.start_wall,self.start_cpu = time.perf_counter(),time.thread_time()
        self.queued_wall,self.queued_cpu = self.start_wall,self.start_cpu
        self.wait_start_wall,self.wait_start_cpu = self.start_wall,self.start_cpu
        if self.cuda:
            torch.cuda.nvtx.range_push("irg.i2.look")
            torch.cuda.nvtx.range_push("irg.i2.enqueue")

    def queued(self):
        self.queued_wall,self.queued_cpu = time.perf_counter(),time.thread_time()
        if self.enqueue_open:
            if self.cuda:
                torch.cuda.nvtx.range_pop()
            self.enqueue_open = False

    def before_wait(self):
        self.wait_start_wall,self.wait_start_cpu = time.perf_counter(),time.thread_time()

    def complete(self, readback, **metadata):
        finish_wall,finish_cpu = time.perf_counter(),time.thread_time()
        timing = readback.last_timing
        wait_wall,wait_cpu = timing["event_wait_wall_seconds"],timing["event_wait_thread_seconds"]
        self.row.update(metadata,**timing,
            enqueue_wall_seconds=self.queued_wall-self.start_wall,
            enqueue_thread_seconds=self.queued_cpu-self.start_cpu,
            planning_wall_seconds=self.wait_start_wall-self.queued_wall,
            planning_thread_seconds=self.wait_start_cpu-self.queued_cpu,
            host_pre_wait_wall_seconds=self.wait_start_wall-self.start_wall,
            host_post_wait_wall_seconds=finish_wall-self.wait_start_wall-wait_wall,
            host_thread_seconds=max(0.0,finish_cpu-self.start_cpu-wait_cpu),
            continuation_slot_wait_wall_seconds=readback.last_allocation_slot_wait["wall_seconds"],
            continuation_slot_wait_thread_seconds=readback.last_allocation_slot_wait["thread_seconds"],
            look_wall_seconds=finish_wall-self.start_wall,
            control_reads=1)
        self.close()
        return self.row

    def close(self):
        if self.enqueue_open:
            if self.cuda:
                torch.cuda.nvtx.range_pop()
            self.enqueue_open = False
        if self.look_open:
            if self.cuda:
                torch.cuda.nvtx.range_pop()
            self.look_open = False


def _record_look_timing(backend, row):
    owner = getattr(backend,"provider",backend)
    performance = getattr(owner,"_device_bce_loop_performance",None)
    if performance is None:
        performance = owner._device_bce_loop_performance = {
            "controls":0,"host_thread_seconds":0.,"event_wait_wall_seconds":0.,
            "maximum_host_thread_seconds":0.,"looks":[]}
    performance["looks"].append(row)
    performance["controls"] += 1
    performance["host_thread_seconds"] += row["host_thread_seconds"]
    performance["event_wait_wall_seconds"] += row["event_wait_wall_seconds"]
    performance["maximum_host_thread_seconds"] = max(
        performance["maximum_host_thread_seconds"],row["host_thread_seconds"])
    backend._device_bce_loop_performance = performance


def _control_window(backend):
    """Resolve the event-aware callback once, preserving zero-arg fixtures."""
    callback = getattr(backend,"device_control_ready",None)
    if not callable(callback):
        return None
    parameters = inspect.signature(callback).parameters
    accepts_event = "event" in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    return (lambda event: callback(event=event)) if accepts_event else (lambda event: callback())


def receipt_worker():
    """Keep receipt execution owned by the optional device controller."""
    from concurrent.futures import ThreadPoolExecutor
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="v0_receipts")


def evidence_tensors(evidence):
    if evidence.device_moments is not None:
        return evidence.device_moments
    # CPU reference fixtures and synthetic providers.  Production providers
    # supply device_moments; this conversion never reads an acquisition device.
    names = tuple(evidence.stratified_moments)
    if names:
        rows = [evidence.stratified_moments[name] for name in names]
        return (names, torch.tensor([r["count"] for r in rows], dtype=torch.float64),
                torch.tensor([[r["mean"] for r in rows]], dtype=torch.float64),
                torch.tensor([[r["m2"] for r in rows]], dtype=torch.float64))
    names = tuple(evidence.panel.allocation_metadata["oracle_stratum_weights"])
    positions = {s:i for i,s in enumerate(names)}
    codes = torch.tensor([positions[s] for s in evidence.panel.allocation_metadata["oracle_strata"]])
    values = torch.as_tensor(evidence.paired_bce_increment, dtype=torch.float64)
    n, means, m2 = db.paired_moments(values, torch.zeros_like(values), codes, len(names))
    return names, n, means, m2


def _deferred_span_inputs(span, delta, tau, transfers):
    from mechanistic_systems.measurement.device_baseline import SPAN_INPUTS
    empty = span.fully_pruned
    return SPAN_INPUTS(span.dense.device_summary,
        transfers.copy("dense_allowance",span.dense.device_allowance).reshape(1),
        transfers.copy("empty",[empty.estimate,getattr(empty,"standard_error",0.),
            max(empty.estimate-empty.lower,empty.upper-empty.estimate),empty.lower,empty.upper,
            empty.confidence_level,span.dense.confidence_level]),
        transfers.copy("delta_tau",[delta,tau]))


def calculate(evidence, *, policy, span, delta, tau, look, maximum_target, cache=None):
    names,counts,means,m2 = evidence_tensors(evidence)
    device = means.device
    cache = {} if cache is None else cache
    if "transfers" not in cache:
        cache["transfers"] = db.HostTransferBuffer(device)
    def tensor(name,value):
        return cache["transfers"].copy(name,value)
    sealed = policy.oracle_sampling_law != "independent_empirical_iid_with_replacement_v1"
    if cache.get("axis") != (names,device,sealed):
        metadata = evidence.panel.allocation_metadata
        cache.update(axis=(names,device,sealed),
            weights=tensor("weights",[metadata["oracle_stratum_weights"][s] for s in names]),
            populations=tensor("populations",[metadata["oracle_population_by_stratum"][s] for s in names])
                if sealed else torch.ones_like(counts),
            quadrature=db.quadrature(device),look=tensor("look",[float(look)]),
            error=tensor("error",[policy.rule.error_probability]),packet=tensor("packet",[policy.look_batch_images]))
    profiles = tuple(evidence.panel.allocation_metadata.get("device_calibration_profiles",
        (evidence.candidate_calibration_profile,evidence.committed_calibration_profile)))
    if len(profiles) == 2:
        profiles += (getattr(span.dense,"device_profile",None) or span.dense.calibration_profile,)
    dimensions = tuple(p.parameter_dimension for p in profiles)
    if any(d == 0 and not isinstance(p.second_order_coefficient, torch.Tensor)
           and p.second_order_coefficient > 0 for d,p in zip(dimensions,profiles)):
        raise ValueError("a zero-dimensional fit cannot have a quadratic allowance")
    empty = span.fully_pruned
    anchors = (span.dense.estimate,span.dense.standard_error,empty.estimate,
        getattr(empty,"standard_error",0.0),max(empty.estimate-empty.lower,empty.upper-empty.estimate),delta,tau)
    planned = None if policy.precision_only_iid else policy.maximum_planned_looks
    # Deferred profiles carry scalar tensors: equality would read their
    # values to Python. A fit's immutable profile identity is sufficient.
    fit_key = (tuple(id(p) for p in profiles),id(span),delta,tau,dimensions,planned)
    if cache.get("fit_key") != fit_key:
        first = torch.stack(tuple(tensor(f"first_{i}",p.first_order_coefficient).reshape(())
                                  for i,p in enumerate(profiles)))
        second = torch.stack(tuple(tensor(f"second_{i}",p.second_order_coefficient).reshape(())
                                   for i,p in enumerate(profiles))).reshape(1,-1)
        deferred_summary = getattr(span.dense,"device_summary",None)
        if deferred_summary is None:
            anchor_tensor = torch.stack(tuple(tensor(f"anchor_{i}",value).reshape(())
                for i,value in enumerate(anchors))).reshape(1,-1)
            span_invalid = torch.zeros((1,),dtype=torch.bool,device=device)
        else:
            anchor_tensor,span_invalid = _deferred_span_inputs(span,delta,tau,cache["transfers"])
        cache.update(fit_key=fit_key,profiles=profiles,span=span,
            covariance=torch.diag(first.square())[None,:,:],second=second,
            anchors=anchor_tensor,span_invalid=span_invalid,
            function=db.compiled_look(sealed=sealed,dimensions=dimensions,planned_looks=planned))
    maximum = math.inf if maximum_target is None else maximum_target
    if "maximum" not in cache:
        cache["maximum"] = tensor("maximum",[maximum])
    elif cache["maximum_value"] != maximum:
        cache["maximum"].fill_(maximum)
    cache["maximum_value"] = maximum
    u,w = cache["quadrature"]
    stats,code = cache["function"](counts,means,m2,cache["weights"],cache["populations"],
        cache["covariance"],cache["second"],cache["anchors"],cache["look"],cache["error"],
        cache["packet"],cache["maximum"],u,w)
    code = torch.where(cache["span_invalid"],6,code)
    cache["look"].add_(1.0)
    return names,counts,means,m2,stats,code


def _envelope(values, *, delta, span, look):
    r = dict(zip(db.FIELDS,values))
    side = r["alpha"]/2.0
    fieller = fieller_ratio_interval(numerator=r["raw_mean"],denominator=delta*span.estimate,
        numerator_variance=r["var_n"],denominator_variance=r["var_b"],covariance=r["cov_nb"],
        alpha=side,numerator_bias_bound=r["bias_n"],denominator_bias_bound=r["bias_b"],quantile=r["scalar_quantile"])
    bounds = {name:r[name] for name in (
        "paired_oracle_radius","dense_oracle_radius","parent_calibration_improvement",
        "candidate_calibration_improvement","dense_calibration_improvement","scalar_standard_error",
        "second_order_bias_bound","paired_oracle_variance","dense_oracle_variance_scaled",
        "fully_pruned_variance_scaled","fully_pruned_bias_scaled","candidate_calibration_variance",
        "parent_calibration_variance","dense_calibration_variance_scaled","candidate_second_order_bias",
        "parent_second_order_bias","dense_second_order_bias_scaled","scalar_quantile")}
    bounds["alpha_side"] = side
    return ScalarV0DecisionEnvelope(estimate=r["mean"],lower=r["lower"],upper=r["upper"],
        alpha=r["alpha"],joint_dimension=1,radius_squared=r["radius_squared"],look_index=look,
        component_bounds=bounds,statistic_estimate=r["statistic_estimate"],statistic_lower=r["statistic_lower"],
        statistic_upper=r["statistic_upper"],statistic_variance=r["statistic_variance"],
        fieller_interval=fieller,method=SCALAR_DECISION_FIELLER_METHOD)


def decide(**kwargs):
    """Stop on execution errors and release buffers without replaying draws."""
    execution = _DeviceExecution(
        query_id=kwargs['query_id'], layer_id=kwargs['layer_id'],
        partition_id=kwargs['partition_id'], proposal_index=kwargs['proposal_index'],
        progress_callback=(kwargs.get('progress_callback')
                          or getattr(getattr(kwargs['backend'], 'provider', None), 'progress_callback', None)),
    )
    try:
        try:
            return _decide(_execution=execution, **kwargs)
        except Exception as cause:
            # Packet validation and acquisition comparisons retain their
            # typed refusals. Errors executing device work gain context.
            if (cause is execution.refused_error
                    or (execution.phase == 'evidence_validation' and isinstance(cause, ValueError))):
                raise
            raise execution.failure(cause) from cause
    finally:
        active_error = sys.exception()
        if execution.look_timer is not None:
            execution.look_timer.close()
        cleanup = getattr(kwargs["backend"],"finish_device_decision",None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception as cause:
                if active_error is not None:
                    active_error.add_note(f"device decision cleanup failed: {type(cause).__name__}: {cause}")
                else:
                    execution.phase = 'cleanup'
                    raise execution.failure(cause) from cause
        if execution.finish_host_timing is not None:
            execution.finish_host_timing()


def _decide(*, backend, committed_masks, candidate_masks, query_id, layer_id, partition_id,
           previous_density, proposed_density, proposal_index, removed, threshold,
           connection_delta_receipt, loss_span_certificate, policy, class_count,
           sequential_spec, progress_callback=None, step_context_delta_sink=None,
           forced_action=None, v6_record=False, receipt_executor=None, _execution):
    from mechanistic_systems.circuits import evidence_bce_precision as ebp
    from mechanistic_systems.circuits.glocal_sampled_path import GlocalSampledComparisonRequest
    from mechanistic_systems.contracts.bce_evidence import (
        CalibrationPrecisionHorizonExhausted, OracleFinitePopulationCensusExhausted,
    )
    packet = int(policy.look_batch_images)
    horizon = None if policy.precision_only_iid else int(policy.futility_horizon_images)
    planned = policy.maximum_planned_looks
    target = packet if horizon is None else min(packet, horizon)
    actual = requested = looks = stagnant = refinements = 0
    prior_panel = None
    span = loss_span_certificate
    action, reason = "unresolved", "evidence_horizon_exhausted_unresolved"
    census = ebp._OracleCensusState.legacy()
    rows, layouts, panels, forecast_rows = [], [], [], []
    statistics_cache = {}
    started = time.perf_counter()
    started_thread = time.thread_time()
    rates = []
    envelope_receipt = None
    oracle_receipts = []
    refinement_receipts = []
    timing_rows = []
    continuation_plans = []
    control_window = _control_window(backend)
    from mechanistic_systems.inference.sequential_bce import LossSpanCertificate
    refreshes_span = callable(getattr(backend, "maybe_refine_joint_precision", None))
    while policy.precision_only_iid or (requested < horizon and looks < planned):
        _execution.look_index = looks
        _execution.phase = 'acquisition'
        if looks > 0 and refreshes_span:
            # An oracle continuation reads the backend's current loss span,
            # as the host loop does before each later look (its calibration
            # profile may have been frozen or refined since the first look).
            latest = getattr(backend, "current_loss_span_certificate", None)
            if isinstance(latest, LossSpanCertificate):
                if getattr(latest.dense,"device_summary",None) is None:
                    latest.validate_positive()
                span = latest
        if policy.oracle_acquisition_envelope_seconds is not None:
            declined = ebp._oracle_acquisition_envelope_exceeded(elapsed_seconds=time.perf_counter()-started,
                envelope_seconds=policy.oracle_acquisition_envelope_seconds,
                look_seconds_per_row=rates, requested_increment=max(1,target-actual))
            if declined is not None:
                envelope_receipt = {"schema":ebp.ORACLE_ACQUISITION_ENVELOPE_RECEIPT_V1,
                    **declined, "look_index":looks, "refinement_rounds":refinements}
                reason = ebp.ORACLE_ENVELOPE_UNRESOLVED_REASON
                break
        remaining = math.inf if policy.precision_only_iid or policy.image_budget is None else max(0,int(policy.image_budget)-int(backend.decision_images_consumed))
        maximum = None if horizon is None else min(horizon, actual+(horizon if math.isinf(remaining) else int(remaining)))
        target = max(target, requested+1)
        if maximum is not None:
            target = min(target, maximum)
        if target <= requested:
            reason = "image_budget_exhausted_unresolved"
            break
        if target < 2*int(policy.fold_count):
            reason = "image_budget_minimum_packet_unavailable"
            break
        requested = int(target)
        request = GlocalSampledComparisonRequest(query_id=str(query_id), layer_id=str(layer_id),
            partition_id=str(partition_id), previous_density=float(previous_density), proposed_density=float(proposed_density),
            stage_index=looks, stage_per_class=max(1,math.ceil(requested/max(class_count,1))),
            proposal_index=int(proposal_index), random_seed=int(policy.random_seed), fold_count=int(policy.fold_count),
            prior_panel=prior_panel, committed_warm_start=None, candidate_warm_start=None, allocation_evidence={},
            cumulative_target_images=requested, maximum_new_contexts=None if math.isinf(remaining) else int(remaining),
            maximum_cumulative_images=maximum, calibration_target_loss_precision=None,
            runtime_deadline_monotonic=None if policy.precision_only_iid else policy.runtime_deadline_monotonic)
        look_started = time.perf_counter()
        device = getattr(getattr(backend,"provider",None),"_device",torch.device("cpu"))
        timer = _execution.look_timer = _LookTiming(_execution,device)
        if progress_callback is not None:
            progress_callback("bce_oracle_panel_started", query_id=query_id, layer_id=layer_id,
                partition_id=partition_id, proposal_index=proposal_index, look_index=looks,
                requested_cumulative_images=requested, previous_panel_images=actual)
        try:
            with db.device_look_context() as checks:
                evidence = backend.evaluate_fixed_calibration_pair(committed_masks, candidate_masks, request)
        except OracleFinitePopulationCensusExhausted as error:
            timer.close()
            _execution.look_timer = None
            oracle_receipts.append(error.receipt)
            reason = ebp.CENSUS_EXHAUSTED_UNRESOLVED_REASON
            break
        except CalibrationPrecisionHorizonExhausted as error:
            timer.close()
            _execution.look_timer = None
            refinement_receipts.append(error.receipt)
            reason = "calibration_precision_horizon_exhausted_no_commit"
            break
        _execution.phase = 'evidence_validation'
        if evidence.device_moments is not None and 'transfers' not in statistics_cache:
            statistics_cache['transfers'] = db.HostTransferBuffer(evidence.device_moments[1].device)
        with db.device_look_context() as validation_checks:
            evidence.validate(fold_count=policy.fold_count,
                              device_count_transfers=statistics_cache.get('transfers'))
        checks.extend(validation_checks)
        if prior_panel is not None:
            evidence.panel.assert_nested_extension(prior_panel)
        previous_actual, actual = actual, len(evidence.panel.context_ids)
        replacement = bool(getattr(backend,"sampling_with_replacement",False)
                           or evidence.panel.allocation_metadata.get("sampling_with_replacement",False))
        census = ebp._oracle_census_state(evidence.panel.allocation_metadata, sampling_with_replacement=replacement)
        stagnant = stagnant+1 if actual <= previous_actual and not census.census_terminal_reached else 0
        _execution.phase = 'statistics'
        names, counts, means, m2, stats, code = calculate(evidence, policy=policy, span=span,
            delta=removed, tau=threshold, look=looks, maximum_target=maximum, cache=statistics_cache)
        if checks:
            failed_checks = torch.stack([flag.to(device=stats.device,dtype=torch.bool) for flag,_ in checks])
            code = torch.where(failed_checks.any(),8,code)
        option_factory = getattr(backend,"device_precision_options",None)
        options = option_factory(committed_masks,candidate_masks,actual,stats) if callable(option_factory) else None
        if options is not None:
            choice, reductions, utilities = db.compiled_precision_plan(stats, *options)
        else:
            choice = torch.full_like(code,4)
        # A single integral control word: low 5 bits = decision, next 4 =
        # acquisition role, remaining bits = the continuation's context target.
        control = code | (choice.to(torch.int64) << 5) | (stats[:,db.INDEX["next_target"]].clamp(max=2**53-1).to(torch.int64) << 9)
        _execution.phase = 'control'
        allocator=getattr(backend,"device_next_allocation",None)
        allocation = None
        if callable(allocator):
            allocation=allocator(evidence,stats,statistics_cache)
        readback = getattr(backend,"_device_bce_control_readback",None)
        if readback is None or readback.device != control.device:
            readback = backend._device_bce_control_readback = db.ControlReadback(control.device)
        # Every kernel and both transfers are enqueued before the planning
        # window. The main thread then waits on exactly one completion event.
        readback.enqueue(control, allocation)
        pending_allocation = None
        if allocation is not None:
            from mechanistic_systems.inference.device_allocation import plan_continuation
            pending_allocation = plan_continuation(backend,evidence,names,readback.allocation_packet)
            continuation_plans.append(pending_allocation)
        timer.queued()
        if control_window is not None:
            control_window(readback.event)
        timer.before_wait()
        word = readback.read()
        consumed = getattr(backend,"device_control_consumed",None)
        if callable(consumed):
            consumed()
        verdict, role, target = word & 31, (word >> 5) & 15, word >> 9
        if allocation is not None and verdict == 0:
            backend.publish_device_allocation(names,pending_allocation)
        timing_row = timer.complete(readback,requested_images=requested,actual_images=actual)
        if pending_allocation is not None:
            timing_row["continuation_planning"] = pending_allocation.timing
        timing_rows.append(timing_row)
        _record_look_timing(backend,timing_row)
        _execution.look_timer = None
        rows.append(torch.cat((stats[0], counts, means[0], m2[0])))
        layouts.append((names, requested, actual, evidence, span))
        looks += 1
        prior_panel = evidence.panel
        elapsed = time.perf_counter()-look_started
        panel = ebp._v6_panel_receipt(query_id=query_id,layer_id=layer_id,partition_id=partition_id,
            proposal_index=proposal_index,look_index=looks-1,target_images=requested,actual_images=actual,
            previous_actual=previous_actual,sampling_with_replacement=replacement,fpc_applied=not replacement,
            census=census,policy=policy,look_started=look_started,
            interval_method=ebp.STRATIFIED_T_FPC_ALPHA_SPENDING_METHOD if not replacement else ebp.STRATIFIED_T_ALPHA_SPENDING_METHOD)
        panels.append(panel)
        if actual > previous_actual:
            rates.append(elapsed/(actual-previous_actual))
        if progress_callback is not None:
            progress_callback("bce_oracle_panel_completed",**panel)
        if verdict == 8:
            # Additional details are read only after a failed acquisition;
            # successful looks still read the single control word.
            failed = failed_checks.detach().to(device="cpu").numpy()
            _execution.phase = 'acquisition_refusal'
            _execution.refused_error = checks[int(np.flatnonzero(failed)[0])][1]()
            raise _execution.refused_error
        if verdict == 6:
            _execution.refused_error = ValueError("invalid device BCE evidence or confidence bound")
            raise _execution.refused_error
        if verdict in {1,2,3}:
            action,reason = {1:("accept","confidence_interval_wholly_at_or_below_boundary"),
                2:("reject","confidence_interval_wholly_above_boundary"),
                3:("conservative_no_commit","point_estimate_at_or_above_boundary_no_commit")}[verdict]
            break
        if stagnant >= 2:
            reason = "panel_growth_exhausted_unresolved"
            break
        if verdict == 4:
            reason = "predicted_sample_horizon_exhausted_unresolved"
            break
        if verdict == 0:
            forecast_rows.append((looks-1, target))
        if target <= actual and not (options is not None and role != 4):
            action,reason = "conservative_no_commit",ebp.CENSUS_TERMINAL_NO_COMMIT_REASON
            break
        can_continue=policy.precision_only_iid or (requested < horizon and looks < planned)
        if options is not None and can_continue:
            _execution.phase = 'precision_refinement'
            backend._device_precision_oracle_target=target
            # Each calibration acquisition retains the reference planner's
            # old-look components and revalues options after the fit changes.
            executor = getattr(backend,"execute_device_precision",None)
            before_refinements=refinements
            while role != 4:
                if not callable(executor):
                    raise TypeError("device precision option lacks an executor")
                precision_timer = _execution.look_timer = _LookTiming(_execution,stats.device,kind="precision")
                with db.device_look_context() as precision_checks:
                    receipt = executor(role,committed_masks,candidate_masks,options,stats)
                    precision_span = backend.current_loss_span_certificate
                    if getattr(precision_span.dense,"device_summary",None) is not None:
                        _,span_invalid = _deferred_span_inputs(precision_span,removed,threshold,
                            statistics_cache["transfers"])
                        db.defer_device_check(span_invalid.any(),
                            lambda:ValueError("v0 requires a verified strictly positive dense-to-empty BCE span"))
                refinement_receipts.append(receipt)
                refinements += 1
                span = backend.current_loss_span_certificate
                options = option_factory(committed_masks,candidate_masks,actual,stats)
                choice, reductions, utilities = db.compiled_precision_plan(stats,*options)
                precision_control = choice.to(torch.int64)
                if precision_checks:
                    precision_failed = torch.stack(tuple(flag.to(device=stats.device,dtype=torch.bool)
                        for flag,_ in precision_checks))
                    precision_control = precision_control | (precision_failed.any().to(torch.int64) << 5)
                readback.enqueue(precision_control)
                precision_timer.queued()
                precision_timer.before_wait()
                precision_word = readback.read()  # one control read per calibration batch
                if callable(consumed):
                    consumed()
                timing_row = precision_timer.complete(readback,requested_images=requested,actual_images=actual)
                timing_rows.append(timing_row)
                _record_look_timing(backend,timing_row)
                _execution.look_timer = None
                if precision_word & 32:
                    failed = precision_failed.detach().to(device="cpu").numpy()
                    _execution.phase = 'acquisition_refusal'
                    _execution.refused_error = precision_checks[int(np.flatnonzero(failed)[0])][1]()
                    raise _execution.refused_error
                role = precision_word & 31
                target = max(target,requested+1)
            if census.census_complete and refinements == before_refinements:
                action,reason="conservative_no_commit",ebp.CENSUS_TERMINAL_NO_COMMIT_REASON
                break
    if planned is not None and looks >= planned and action == "unresolved":
        reason = "planned_look_horizon_exhausted_unresolved"
    _execution.look_index = max(0, looks - 1)
    _execution.phase = 'receipt'
    decision_statistics = {"path":"compiled_float64_device","control_reads":looks+refinements,
        "receipt_reads":int(bool(rows)),"calibration_control_reads":refinements,
        "look_performance":tuple(timing_rows),"receipt_off_acquisition_path":receipt_executor is not None
            and not policy.accept_replication_gate}
    def finish_host_timing():
        host_seconds = max(0.,time.thread_time()-started_thread-
            sum(row["event_wait_thread_seconds"] for row in timing_rows))
        decision_statistics["decision_host_thread_seconds"] = host_seconds
        performance = getattr(backend,"_device_bce_loop_performance",None)
        if performance is not None:
            performance.setdefault("decisions",[]).append({
                "query_id":query_id,"layer_id":layer_id,"proposal_index":proposal_index,
                "host_thread_seconds":host_seconds,"controls":len(timing_rows)})
            performance["decision_host_thread_seconds"] = performance.get("decision_host_thread_seconds",0.)+host_seconds
            performance["maximum_decision_host_thread_seconds"] = max(
                performance.get("maximum_decision_host_thread_seconds",0.),host_seconds)
            performance["average_decision_host_thread_seconds"] = (
                performance["decision_host_thread_seconds"]/len(performance["decisions"]))
    _execution.finish_host_timing = finish_host_timing
    retained_backend_span = getattr(backend,"current_loss_span_certificate",None)
    def finish_receipt():
        nonlocal action, reason
        for pending in continuation_plans:
            pending.complete()
        from mechanistic_systems.measurement.device_baseline import materialize_span
        for index,(names,wanted,delivered,evidence,look_span) in enumerate(layouts):
            layouts[index] = (names,wanted,delivered,evidence,materialize_span(look_span))
        completed_span = materialize_span(retained_backend_span) if retained_backend_span is not None else None
        if (completed_span is not retained_backend_span
                and getattr(backend,"current_loss_span_certificate",None) is retained_backend_span):
            backend._loss_span_certificate = completed_span
        # Receipt-only values cross once per decision, after the acquisition loop.
        # Width varies only across decisions; each decision has one fixed stratum axis.
        final_delta = None
        if rows:
            stacked = torch.stack(rows)
            observations = layouts[-1][3].panel.allocation_metadata.get("device_observation_packets",())
            if step_context_delta_sink is not None and observations:
                packed = torch.cat((stacked.reshape(-1),torch.cat(observations).to(torch.float64)))
                host_packet = packed.detach().to(device="cpu").numpy()
                host = host_packet[:stacked.numel()].reshape(stacked.shape)
                final_delta = host_packet[stacked.numel():].astype(np.float32)
            else:
                host = stacked.detach().to(device="cpu").numpy()
        else:
            host = np.zeros((0,len(db.FIELDS)))
        per_look = []
        latest = None
        for k,(row,(names,wanted,delivered,evidence,look_span)) in enumerate(zip(host,layouts)):
            S = len(names)
            count_values = row[len(db.FIELDS):len(db.FIELDS)+S].astype(np.int64)
            raw_means = row[len(db.FIELDS)+S:len(db.FIELDS)+2*S]
            raw_m2 = row[len(db.FIELDS)+2*S:]
            latest = dict(zip(db.FIELDS,row[:len(db.FIELDS)]))
            per_look.append((k,wanted,delivered,latest["mean"],latest["lower"],latest["upper"]))
        if progress_callback is not None:
            # Numerical forecast telemetry uses the same terminal bulk read.
            # The next acquisition only needs the packed integer target.
            for k, next_target in forecast_rows:
                values = dict(zip(db.FIELDS, host[k,:len(db.FIELDS)]))
                look_span = layouts[k][4]
                envelope = _envelope(host[k,:len(db.FIELDS)],delta=removed,span=look_span,look=k)
                progress_callback("v0_decision_precision_forecast",look_index=k,
                    forecast_cumulative_context_count=next_target,
                    forecast_growth_guard_applied=bool(values["growth_guard_applied"]),
                    analytic_forecast_cumulative_context_count=int(values["analytic_target"]),
                    forecast_population_clamp_applied=bool(values["population_clamp_applied"]),
                    raw_paired_bce_standard_error=values["raw_standard_error"],
                    joint_component_bounds=envelope.component_bounds)
        mean,lower,upper = (0.0,-math.inf,math.inf) if latest is None else (latest["mean"],latest["lower"],latest["upper"])
        replication_audit = None
        if policy.accept_replication_gate and action == "accept":
            confirm = getattr(backend,"confirm_accept_replication",None)
            if callable(confirm):
                audit = confirm(committed_masks,candidate_masks,query_id=query_id,layer_id=layer_id,
                    partition_id=partition_id,proposal_index=proposal_index,confidence_level=policy.rule.confidence_level)
                replicated = audit["confirm_mean_sensitivity"] <= threshold
                replication_audit = {**dict(audit),"select_action":"accept",
                    "replicated":bool(replicated),"accept_boundary":threshold}
                if not replicated:
                    action,reason = "unresolved","accept_failed_confirm_replication"
                if progress_callback is not None:
                    progress_callback("v0_accept_confirm_replication_completed",query_id=str(query_id),
                        layer_id=str(layer_id),partition_id=str(partition_id),proposal_index=int(proposal_index),
                        replicated=bool(replicated),confirm_mean_sensitivity=float(audit["confirm_mean_sensitivity"]),
                        accept_boundary=threshold,confirm_row_count=int(audit["confirm_row_count"]))
        natural_action = action
        if forced_action is not None:
            action,reason = str(forced_action.action), str(forced_action.reason or reason)
        if step_context_delta_sink is not None and layouts:
            final_evidence = layouts[-1][3]
            if final_delta is None:
                final_delta = np.asarray(final_evidence.paired_bce_increment,dtype=np.float32)
            contexts = tuple(final_evidence.panel.context_ids)
            available = final_delta.size == len(contexts)
            step_context_delta_sink(proposal_index=int(proposal_index),layer_id=str(layer_id),partition_id=str(partition_id),
                action=str(action),context_ids=contexts if available else (),paired_bce_delta=final_delta if available else np.zeros(0,np.float32),
                unavailable_context_count=0 if available else len(contexts))
        if v6_record:
            record = ebp._v6_decision_record(query_id=query_id,layer_id=layer_id,partition_id=partition_id,
                proposal_index=proposal_index,previous_density=previous_density,proposed_density=proposed_density,
                action=action,natural_action=natural_action,engine_reason=reason,engine_forecast_target=None,reason=reason,looks=looks,panel_images=actual,
                mean=mean,lower=lower,upper=upper,threshold=threshold,removed=removed,
                alpha_allocated=policy.rule.error_probability,suppressed=latest is None or not all(map(math.isfinite,(mean,lower,upper))),
                forced=forced_action is not None,replication=policy.accept_replication_gate,per_look=tuple(per_look),
                candidate_masks=candidate_masks)
            record["statistics_execution"] = decision_statistics
            return ebp._V6BCEStepEvidence(action=action,reason=reason,looks=looks,panel_images=actual,
                mean=mean,lower=lower,upper=upper,panel_hashes=(),panel_receipts=tuple(panels),
                calibration_precision_receipts=tuple(refinement_receipts),connection_delta_receipt=connection_delta_receipt.receipt,
                decision_receipt=ebp._v6_decision_reference(record),oracle_census_receipts=tuple(oracle_receipts),decision_record=record)
        if latest is None:
            receipt = {"schema":"sequential_bce_decision_receipt_suppressed_v1","suppressed":True,
                       "action":action,"reason":reason,"looks":looks,"failed_conditions":("at_least_one_usable_look",)}
        else:
            envelope = _envelope(host[-1,:len(db.FIELDS)],delta=removed,span=look_span,look=looks-1)
            raw_statistics = {name:{"count":int(n),"mean":float(m),"variance":float(v/(n-1))}
                for name,n,m,v in zip(names,count_values,raw_means,raw_m2)}
            optimized = OptimizedCalibrationDecisionEnvelope.paired_increment(oracle_estimate=latest["raw_mean"],
                oracle_lower=latest["raw_lower"],oracle_upper=latest["raw_upper"],
                parent_improvement_bound=evidence.committed_calibration_improvement_bound,
                candidate_improvement_bound=evidence.candidate_calibration_improvement_bound)
            receipt = ebp._build_decision_receipt(accept_replication_audit=replication_audit,action=action,allocation_receipts=(),
                calibration_state_ids=(evidence.committed_calibration_state_id,evidence.candidate_calibration_state_id),
                candidate_hash=str(evidence.candidate_mask_hash),census=census,connection_delta_receipt=connection_delta_receipt,
                decision_resource_usage=ebp._decision_resource_usage(backend),envelope_receipt=envelope_receipt,
                joint_decision_envelope=envelope,joint_precision_allocation_receipts=refinement_receipts,
                latest_step_observables={},layer_id=layer_id,loss_span_certificate=look_span,lower=lower,mean=mean,
                observables_enabled=False,optimized_calibration_envelope=optimized,partition_id=partition_id,
                proposal_index=proposal_index,query_id=query_id,raw_lower=latest["raw_lower"],raw_mean=latest["raw_mean"],
                raw_stratum_statistics=raw_statistics,raw_upper=latest["raw_upper"],refinement_rounds=refinements,
                removed=removed,resolved_threshold=threshold,sequential_spec=sequential_spec,upper=upper)
        return ebp._BCEStepEvidence(action=action,reason=reason,looks=looks,panel_images=actual,mean=mean,lower=lower,upper=upper,
            panel_hashes=(),panel_receipts=tuple(panels),connection_delta_receipt=connection_delta_receipt.receipt,
            decision_receipt=receipt,oracle_census_receipts=tuple(oracle_receipts))

    if (receipt_executor is not None and v6_record
            and not policy.accept_replication_gate):
        # The frontier needs only the action and a small record reference to
        # schedule its next acquisition.  Numerical fields are filled after
        # this future finishes, before the frontier result is returned.
        suppressed = not rows
        seed = ebp._v6_decision_record(query_id=query_id,layer_id=layer_id,partition_id=partition_id,
            proposal_index=proposal_index,previous_density=previous_density,proposed_density=proposed_density,
            action=action,natural_action=action,engine_reason=reason,engine_forecast_target=None,reason=reason,looks=looks,panel_images=actual,
            mean=0.,lower=0.,upper=0.,threshold=threshold,removed=removed,
            alpha_allocated=policy.rule.error_probability,suppressed=suppressed,forced=False,
            replication=False,per_look=(),candidate_masks=candidate_masks)
        device = rows[0].device if rows else torch.device("cpu")
        ready = None
        if device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
        def finish_in_worker():
            try:
                if ready is None:
                    return finish_receipt()
                # The copy and receipt work use a separate stream; the main
                # stream is free to execute the next proposal.
                with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device=device)):
                    stream = torch.cuda.current_stream(device)
                    stream.wait_event(ready)
                    for row in rows:
                        row.record_stream(stream)
                    observations = layouts[-1][3].panel.allocation_metadata.get("device_observation_packets",())
                    for packet in observations:
                        packet.record_stream(stream)
                    return finish_receipt()
            except Exception as cause:
                raise receipt_execution.failure(cause) from cause
        receipt_execution = replace(_execution)
        future = receipt_executor.submit(finish_in_worker)
        placeholder = ebp._V6BCEStepEvidence(action=action,reason=reason,looks=looks,panel_images=actual,
            mean=math.nan,lower=math.nan,upper=math.nan,panel_hashes=(),panel_receipts=tuple(panels),
            calibration_precision_receipts=tuple(refinement_receipts),
            connection_delta_receipt=connection_delta_receipt.receipt,
            decision_receipt=ebp._v6_decision_reference(seed),oracle_census_receipts=tuple(oracle_receipts),
            decision_record=seed)
        return SimpleNamespace(**vars(placeholder),receipt_future=future)
    return finish_receipt()
