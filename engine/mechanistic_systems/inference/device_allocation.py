"""V5 adaptive cost-aware Neyman counts for the next paired look.

The continuation planner decodes pinned allocation metadata after the stop
word's completion event. Sample means, variances and allocation SDs remain
on the acquisition device; the decision thread commits a prepared LookPlan.
"""
from __future__ import annotations
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import threading
import time
import numpy as np
import torch
from .device_bce import CapturedTensorFunction, neyman_counts


def allocation_sds(counts, means, m2):
    """Native _oracle_allocation_sds, with the same pooled fallback."""
    total = counts.sum(-1)
    pooled_mean = (counts*means).sum(-1)/total.clamp_min(1.0)
    pooled_m2 = (m2+counts*(means-pooled_mean[...,None]).square()).sum(-1)
    pooled_sd = (pooled_m2/(total-1.0).clamp_min(1.0)).clamp_min(0.0).sqrt()
    pooled_sd = torch.where(total > 1,pooled_sd,torch.ones_like(pooled_sd))
    sd = (m2/(counts-1.0).clamp_min(1.0)).clamp_min(0.0).sqrt()
    return torch.where(counts > 1,sd,pooled_sd[...,None]).amax(0)


def next_counts(counts, means, m2, weights, populations, costs, minimum,
                target, maximum, quantum, *, sealed):
    actual = counts.sum(-1)
    increment = (target-actual).clamp_min(0.0)
    increment = torch.ceil(increment/quantum)*quantum
    increment = torch.minimum(increment,(maximum-actual).clamp_min(0.0))
    if sealed:
        capacity = populations
        increment = torch.minimum(increment,(populations-counts).sum(-1))
    else:
        capacity = counts+increment[...,None]
    sds = allocation_sds(counts,means,m2)
    return neyman_counts(actual+increment,counts,minimum,capacity,weights,sds,costs)


@lru_cache(maxsize=2)
def _captured_next_counts(sealed):
    def function(*args):
        return next_counts(*args,sealed=sealed)
    function.__code__ = function.__code__.replace(co_name=f"bce_next_counts_{sealed}")
    return CapturedTensorFunction(function)


def compiled_next_counts(*args, sealed):
    return _captured_next_counts(bool(sealed))(*args)


_CONTINUATION_POOL = None
_CONTINUATION_LOCK = threading.Lock()


def _continuation_executor():
    global _CONTINUATION_POOL
    with _CONTINUATION_LOCK:
        if _CONTINUATION_POOL is None or _CONTINUATION_POOL[0] != os.getpid():
            _CONTINUATION_POOL = (os.getpid(),ThreadPoolExecutor(max_workers=1,
                thread_name_prefix="irg-continuation"))
        return _CONTINUATION_POOL[1]


def warm_continuation_planner():
    _continuation_executor().submit(lambda:None).result()


class PendingContinuation:
    """A complete next-look plan decoded outside the decision thread."""

    def __init__(self,names,timing):
        self.names,self.timing,self.total = names,timing,None
        self.requested_total = None
        self.future = None

    def result(self):
        started,cpu = time.perf_counter(),time.thread_time()
        pending = not self.future.done()
        try:
            return self.future.result()
        finally:
            self.timing["commit_wait_wall_seconds"] += time.perf_counter()-started
            self.timing["commit_wait_thread_seconds"] += time.thread_time()-cpu
            self.timing["commit_waits"] += int(pending)

    def complete(self):
        return self.future.result()


class DeviceAllocationError(ValueError):
    """Allocation refusal with host-readable counts from the completed packet."""

    def __init__(self, details):
        self.details = details
        super().__init__("device allocation differs from the requested nested prefix: "
            + json.dumps(details, separators=(",", ":")))


def allocation_failure(**details):
    """Report only a refused plan; never read device statistics or change draws."""
    details = {key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in details.items()}
    print("DEVICE_ALLOCATION_FAILURE " + json.dumps(details, separators=(",", ":")),
        file=sys.stderr, flush=True)
    return DeviceAllocationError(details)


def plan_continuation(backend,evidence,names,packet):
    """Decode pinned descriptors and build the exact native LookPlan.

    Only the control reader grants ``packet.ready``; this thread never
    waits on CUDA or reads the control word, and never advances draws.
    """
    from mechanistic_systems.measurement.stratum_plan import LookPlan,draw_triples,selection_codes
    provider = backend.provider
    axis = provider._oracle_stratum_axis()
    names = tuple(names)
    if names != axis.names:
        raise ValueError("device allocation stratum axis changed")
    declared = evidence.panel.allocation_metadata["delivered_stratum_counts"]
    existing = np.fromiter((declared[name] for name in names),dtype=np.int64,count=len(names))
    with_replacement = bool(provider.oracle_sampling_with_replacement)
    sealed = bool(provider.iid_design_mode and not with_replacement)
    timing = {"grant_wait_wall_seconds":0.,"worker_wall_seconds":0.,"worker_thread_seconds":0.,
        "worker_gil_cpu_upper_bound_seconds":0.,
        "commit_wait_wall_seconds":0.,"commit_wait_thread_seconds":0.,"commit_waits":0}
    pending = PendingContinuation(names,timing)
    def prepare():
        grant_started = time.perf_counter()
        packet.ready.wait()
        timing["grant_wait_wall_seconds"] = time.perf_counter()-grant_started
        started,cpu = time.perf_counter(),time.thread_time()
        try:
            targets = packet.copy_counts().reshape(-1)
            total = int(targets.sum())
            increment = total-int(existing.sum())
            capacities = existing+increment if with_replacement else axis.populations
            minima = np.minimum(axis.minimum,capacities)
            if (targets.shape != existing.shape or increment < 0
                    or np.any(targets < np.maximum(existing,minima)) or np.any(targets > capacities)):
                raise allocation_failure(
                    branch="continuation_bounds", host_requested_total=pending.requested_total,
                    device_plan_total=total, names=names, existing=existing,
                    minima=minima, capacities=capacities, targets=targets,
                    increment=increment,
                )
            counts = dict(zip(names,targets.tolist()))
            receipt = {"schema":"unified_neyman_allocation_receipt_v1",
                "policy_id":"class_stratified_binary_risk_neyman_v1","role":"paired_bce_oracle",
                "target_counts":counts,"additions":dict(zip(names,(targets-existing).tolist())),
                "requested_cumulative_count":total,"delivered_cumulative_count":total,
                "population_count":int(capacities.sum()),"full_census":bool(np.array_equal(targets,capacities)),
                "finite_population_correction":not with_replacement,
                "allocation_only_variance":True,"execution":"adaptive_device_neyman"}
            plan = LookPlan(targets,draw_triples(axis,existing,targets),selection_codes(existing,targets),receipt,
                provider._oracle_census_contract(drawn=targets,terminal_reached=False) if sealed else None)
            pending.total = total
            return plan
        finally:
            timing["worker_wall_seconds"] = time.perf_counter()-started
            timing["worker_thread_seconds"] = time.thread_time()-cpu
            timing["worker_gil_cpu_upper_bound_seconds"] = timing["worker_thread_seconds"]
    pending.future = _continuation_executor().submit(prepare)
    return pending
