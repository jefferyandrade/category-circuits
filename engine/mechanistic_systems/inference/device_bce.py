"""Float64 tensor transcription of the v5 look, on CPU or CUDA.

The functions below have no tensor-to-Python conversions.  Shapes and the
sampling law are static; counts, alpha, moments and decisions are tensors.
Receipt construction belongs to the caller, after the decision has finished.
"""

from __future__ import annotations

import functools
import math
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import torch

from mechanistic_systems.backends.cuda_graphs import initialize_cuda_graph_state
from mechanistic_systems.core.determinism import configure_deterministic_backend


DEVICE_LOOK_ACTIVE = ContextVar("device_bce_look_active", default=False)
DEVICE_LOOK_CHECKS = ContextVar("device_bce_look_checks", default=None)
DEVICE_LOOK_COPY_INDICES = ContextVar("device_bce_copy_indices", default=None)


@contextmanager
def device_look_context():
    token = DEVICE_LOOK_ACTIVE.set(True)
    checks = []
    check_token = DEVICE_LOOK_CHECKS.set(checks)
    copy_token = DEVICE_LOOK_COPY_INDICES.set({})
    try:
        yield checks
    finally:
        DEVICE_LOOK_CHECKS.reset(check_token)
        DEVICE_LOOK_COPY_INDICES.reset(copy_token)
        DEVICE_LOOK_ACTIVE.reset(token)


def defer_device_check(failed, error_factory):
    checks = DEVICE_LOOK_CHECKS.get()
    if checks is None:
        return False
    checks.append((failed, error_factory))
    return True


class HostTransferBuffer:
    """Reusable pinned staging and non-blocking H2D copies for look metadata.

    The preceding control read completes use of the previous staging values.
    CPU tests exercise the same copy protocol without allocating CUDA memory.
    """
    def __init__(self, device):
        self.device = torch.device(device)
        self.buffers = {}

    def copy(self, key, values, *, dtype=torch.float64):
        if isinstance(values, torch.Tensor) and values.device.type != "cpu":
            return values.to(device=self.device, dtype=dtype, non_blocking=True)
        source = torch.as_tensor(values, dtype=dtype, device="cpu")
        prior = self.buffers.get(key)
        if prior is None or prior[0].shape != source.shape or prior[0].dtype != dtype:
            staging = torch.empty(source.shape, dtype=dtype,
                                  pin_memory=self.device.type == "cuda")
            destination = torch.empty(source.shape, dtype=dtype, device=self.device)
            self.buffers[key] = staging, destination
        staging, destination = self.buffers[key]
        staging.copy_(source)
        destination.copy_(staging, non_blocking=True)
        return destination


class AllocationPacket:
    """Pinned metadata granted by the control event to the CPU planner."""

    def __init__(self, values, available):
        self.values,self.available = values,available
        self.ready = threading.Event()

    def copy_counts(self):
        self.ready.wait()
        try:
            return self.values.numpy().copy()
        finally:
            self.available.set()


class ControlReadback:
    """One pinned control word and one completion event per look.

    Allocation metadata is copied alongside the word and granted to the
    continuation planner after the same event completes. The decision
    thread reads only the word; the planner introduces no CUDA wait.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.word = torch.empty(1, dtype=torch.int64,
                                pin_memory=self.device.type == "cuda")
        self.allocation = None
        self.allocation_packet = None
        self._allocation_slots = []
        self.event = (torch.cuda.Event() if self.device.type == "cuda" else None)
        self.reads = 0
        self.last_timing = {}
        self.last_allocation_slot_wait = {"wall_seconds":0.,"thread_seconds":0.}

    def prepare_allocation(self, shape, dtype=torch.int64, slots=2):
        """Reserve pinned descriptor slots before the worker's first look."""
        key = (tuple(shape),dtype)
        matching = [slot for slot in self._allocation_slots if slot[0] == key]
        for _ in range(max(0,int(slots)-len(matching))):
            available = threading.Event()
            available.set()
            values = torch.empty(shape,dtype=dtype,pin_memory=self.device.type == "cuda")
            self._allocation_slots.append((key,values,available))

    def enqueue(self, word, allocation=None):
        if word.numel() != 1 or word.dtype != torch.int64:
            raise ValueError("a device look must produce one int64 control word")
        self.word.copy_(word.detach().reshape(1), non_blocking=True)
        self.allocation_packet = None
        self.last_allocation_slot_wait = {"wall_seconds":0.,"thread_seconds":0.}
        if allocation is not None:
            self.prepare_allocation(allocation.shape,allocation.dtype)
            key = (tuple(allocation.shape),allocation.dtype)
            matching = [slot for slot in self._allocation_slots if slot[0] == key]
            slot = next((slot for slot in matching if slot[2].is_set()),matching[0])
            # A full pool waits only for the CPU planner to copy metadata;
            # the GPU completion is still the single control event below.
            wait_started,wait_cpu = time.perf_counter(),time.thread_time()
            slot[2].wait()
            self.last_allocation_slot_wait = {"wall_seconds":time.perf_counter()-wait_started,
                "thread_seconds":time.thread_time()-wait_cpu}
            slot[2].clear()
            self.allocation = slot[1]
            self.allocation.copy_(allocation.detach(), non_blocking=True)
            self.allocation_packet = AllocationPacket(self.allocation,slot[2])
        if self.event is not None:
            self.event.record(torch.cuda.current_stream(self.device))

    def read(self):
        wait_wall,wait_cpu = time.perf_counter(),time.thread_time()
        if self.event is not None:
            self.event.synchronize()
        waited_wall,waited_cpu = time.perf_counter(),time.thread_time()
        self.reads += 1
        word = int(self.word.numpy()[0])
        if self.allocation_packet is not None:
            self.allocation_packet.ready.set()
        decoded_wall,decoded_cpu = time.perf_counter(),time.thread_time()
        self.last_timing = {
            "event_wait_wall_seconds": waited_wall-wait_wall,
            "event_wait_thread_seconds": waited_cpu-wait_cpu,
            "word_decode_wall_seconds": decoded_wall-waited_wall,
            "word_decode_thread_seconds": decoded_cpu-waited_cpu,
        }
        return word

    def allocation_counts(self):
        if self.allocation is None:
            raise RuntimeError("this look has no allocation descriptors")
        if self.allocation_packet is not None:
            return self.allocation_packet.copy_counts()
        return self.allocation.numpy().copy()


def _clone_outputs(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_outputs(item) for item in value)
    if isinstance(value, list):
        return [_clone_outputs(item) for item in value]
    if isinstance(value, dict):
        return {name: _clone_outputs(item) for name, item in value.items()}
    return value


class _CapturedTensorCall:
    """A static tensor function captured once for a worker's shape lane."""

    def __init__(self, function, arguments):
        initialize_cuda_graph_state(arguments[0].device)
        # Replay updates these private inputs even when its caller leaves
        # inference mode after capture. Their storage must stay mutable in
        # either context; the function itself still captures in caller mode.
        with torch.inference_mode(False):
            self.arguments = tuple(value.clone() for value in arguments)
        device = self.arguments[0].device
        stream = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                function(*self.arguments)
        current.wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream, capture_error_mode="thread_local"):
            self.outputs = function(*self.arguments)
        current.wait_stream(stream)

    def __call__(self, arguments):
        for resident, incoming in zip(self.arguments, arguments, strict=True):
            resident.copy_(incoming, non_blocking=True)
        self.graph.replay()
        # Look receipts retain these tensors while a later look replays the
        # same graph. Give each look its own output storage.
        return _clone_outputs(self.outputs)


class CapturedTensorFunction:
    """Compiled tensor lanes and static CUDA graphs, with worker lifetime."""

    def __init__(self, function):
        self.function = function
        self.compiled = None
        self._compiled_functions = {}
        self.graphs = {}
        self.captures = self.replays = 0

    def _compiled_lane(self, key):
        configure_deterministic_backend()
        runner = self._compiled_functions.get(key)
        if runner is None:
            import types
            function = self.function
            # Every physical layout owns an entry code object. A fixed-shape
            # compiler must not accumulate unrelated stride/shape guards in
            # one frame and exhaust Dynamo's finite recompilation bound.
            entry = types.FunctionType(
                function.__code__.replace(
                    co_name=f"{function.__name__}_{id(self)}_lane_{len(self._compiled_functions)}"),
                function.__globals__, function.__name__, function.__defaults__, function.__closure__,
            )
            entry.__kwdefaults__ = function.__kwdefaults__
            runner = torch.compile(entry, fullgraph=True, dynamic=False)
            self._compiled_functions[key] = runner
            if self.compiled is None:
                self.compiled = runner
        return runner

    def __call__(self, *arguments):
        layout = tuple((str(value.device), tuple(value.shape), tuple(value.stride()),
                        value.dtype, bool(value.requires_grad)) for value in arguments)
        # CPU compilation sees the ambient autograd mode. CUDA replay uses
        # its already captured kernels, including across inference contexts.
        key = (layout, torch.is_grad_enabled(), torch.is_inference_mode_enabled()) if arguments[0].device.type != "cuda" else layout
        compiled = self._compiled_lane(key)
        if arguments[0].device.type != "cuda":
            return compiled(*arguments)
        graph = self.graphs.get(key)
        if graph is None:
            graph = self.graphs[key] = _CapturedTensorCall(compiled, arguments)
            self.captures += 1
        self.replays += 1
        return graph(arguments)


# Stable receipt column order.  The statistic is N - tau * delta * D.
FIELDS = (
    "mean", "lower", "upper", "statistic_estimate", "statistic_lower",
    "statistic_upper", "statistic_variance", "raw_mean", "raw_lower",
    "raw_upper", "raw_standard_error", "degrees_of_freedom", "alpha",
    "scalar_quantile",
    "paired_oracle_variance", "dense_oracle_variance_scaled",
    "fully_pruned_variance_scaled", "candidate_calibration_variance",
    "parent_calibration_variance", "dense_calibration_variance_scaled",
    "candidate_second_order_bias", "parent_second_order_bias",
    "dense_second_order_bias_scaled", "fully_pruned_bias_scaled",
    "parent_calibration_improvement", "candidate_calibration_improvement",
    "dense_calibration_improvement", "paired_oracle_radius",
    "dense_oracle_radius", "scalar_standard_error", "second_order_bias_bound",
    "var_n", "var_b", "cov_nb", "bias_n", "bias_b", "next_target",
    "invalid", "radius_squared", "analytic_target", "growth_guard_applied",
    "population_clamp_applied", "current_standard_error",
)
INDEX = {name: index for index, name in enumerate(FIELDS)}

# Fixed Gaussian quadrature evaluates the survival probability directly,
# avoiding 1-CDF cancellation.  Its nodes are constants, not observation data.
_nodes, _weights = np.polynomial.legendre.leggauss(96)
_U = torch.from_numpy((_nodes + 1.0) * 0.5)
_W = torch.from_numpy(_weights * 0.5)


def _normal_isf(p):
    # The inverse lower normal tail avoids subtracting small alpha from one.
    return -torch.special.ndtri(p)


def _t_log_normalizer(df):
    a = df * 0.5
    exact = torch.lgamma(a + 0.5) - torch.lgamma(a) - 0.5 * torch.log(df * torch.pi)
    # log Gamma(a+1/2)/Gamma(a) = log(a)/2 - 1/(8a) + 1/(192a^3)
    # - 1/(640a^5) + ... .  Avoid subtracting large lgamma values.
    asymptotic = -0.5 * np.log(2.0 * np.pi) - 1.0 / (8.0 * a) + 1.0 / (192.0 * a**3) - 1.0 / (640.0 * a**5)
    return torch.where(df >= 100.0, asymptotic, exact)


def _t_sf(x, df, u, w):
    log_pdf = _t_log_normalizer(df)[..., None] - ((df + 1.0) * 0.5)[..., None] * torch.log1p(
        x[..., None].square() / (df[..., None] * u.square())
    )
    ordinary = x * (w * torch.exp(log_pdf) / u.square()).sum(dim=-1)
    # u = v^(4/df) smooths the fractional-power endpoint in the low-df
    # tail.  In the Satterthwaite domain df can approach one continuously.
    small_df = df.clamp(max=8.0)
    log_scale = _t_log_normalizer(small_df) + 0.5*(small_df-1.0)*torch.log(small_df) - small_df*torch.log(x)
    integrand = torch.exp(-0.5*(small_df+1.0)[...,None] * torch.log1p(
        small_df[...,None]/x[...,None].square() * u.pow(8.0/small_df[...,None])))
    low = torch.exp(log_scale)*(4.0*w*u**3*integrand).sum(dim=-1)
    return torch.where(df < 8.0, low, ordinary)


def student_t_isf(p, df, u, w):
    """Inverse Student-t tail by fixed quadrature and safeguarded Newton steps.

    Valid for df >= 1, the entire Satterthwaite domain when n_h >= 2.
    All iterations are tensor operations with fixed count and shape.
    """
    finite_df = torch.where(torch.isfinite(df), df, torch.full_like(df, 1e12))
    z = _normal_isf(p)
    z2 = z.square()
    x = z + (z**3 + z) / (4.0 * finite_df) + (5.0*z**5 + 16.0*z**3 + 3.0*z) / (96.0 * finite_df.square())
    x = x + (3.0*z**7 + 19.0*z**5 + 17.0*z**3 - 15.0*z) / (384.0 * finite_df**3)
    # A tail start for the low-degree cases where the normal expansion is poor.
    tail = torch.exp((_t_log_normalizer(finite_df) + 0.5 * (finite_df - 1.0) * torch.log(finite_df) - torch.log(p)) / finite_df)
    x = torch.where(finite_df < 5.0, torch.maximum(z, tail), x)
    for _ in range(8):
        sf = _t_sf(x, finite_df, u, w)
        pdf = torch.exp(_t_log_normalizer(finite_df) - 0.5*(finite_df + 1.0)*torch.log1p(x.square()/finite_df))
        step = (sf - p) / pdf.clamp_min(1e-300)
        x = (x + step.clamp(min=-0.5*x, max=x)).clamp_min(1e-15)
    return torch.where(torch.isfinite(df), x, z)


def _chi2_isf(p, dimension: int):
    if dimension == 0:
        return torch.zeros_like(p)
    if dimension == 2:
        return -2.0 * torch.log(p)
    if dimension == 1:
        return _normal_isf(p * 0.5).square()
    a = torch.full_like(p, dimension * 0.5)
    z = _normal_isf(p)
    x = dimension * (1.0 - 2.0/(9.0*dimension) + z * np.sqrt(2.0/(9.0*dimension)))**3
    for _ in range(12):
        sf = torch.special.gammaincc(a, x * 0.5)
        pdf = torch.exp((a - 1.0)*torch.log(x) - x*0.5 - a*np.log(2.0) - torch.lgamma(a))
        step = (sf - p) / pdf.clamp_min(1e-300)
        x = (x + step.clamp(min=-0.5*x, max=x)).clamp_min(1e-15)
    return x


def stratified_moments(values, codes, stratum_count: int):
    """Two-pass stratum reductions with fixed row order and no floating atomics.

    The compiler fuses the membership mask into each stratum's row reduction.
    Negative codes contribute nothing, including padded rows containing NaNs.
    Deterministic CUDA scatter uses an allocating sort that cannot be captured;
    these fixed-axis reductions keep the whole look inside its CUDA graph.
    """
    values = values.to(torch.float64)
    if values.ndim == 1:
        values = values[:, None]
    strata = torch.arange(stratum_count, device=values.device)
    members = codes[None, :] == strata[:, None]
    counts = members.sum(dim=1).to(torch.float64)
    sums = torch.where(members[:, :, None], values[None, :, :], 0.0).sum(dim=1)
    means = sums / counts.clamp_min(1.0)[:, None]
    deviations = values[None, :, :] - means[:, None, :]
    m2 = torch.where(members[:, :, None], deviations.square(), 0.0).sum(dim=1)
    # Canonical contiguous lane/stratum strides also when there is one lane.
    # Otherwise the first look's transpose has stride (1,1), while a Chan
    # merge has stride (S,1), causing a new statistics graph on look two.
    lanes=values.shape[1]
    mean_rows=means.T.contiguous().flatten().reshape(lanes,stratum_count)
    m2_rows=m2.T.contiguous().flatten().reshape(lanes,stratum_count)
    return counts,mean_rows,m2_rows


def paired_moments(candidate, parent, codes, stratum_count: int):
    """Two-pass segmented paired increments; rows [R,L], output [L,S]."""
    return stratified_moments(candidate.to(torch.float64) - parent.to(torch.float64),
                              codes, stratum_count)


def merge_moments(n1, mean1, m21, n2, mean2, m22):
    n = n1 + n2
    difference = mean2 - mean1
    mean = mean1 + difference * n2 / n.clamp_min(1.0)
    m2 = m21 + m22 + difference.square()*n1*n2/n.clamp_min(1.0)
    return n, torch.where(n1 == 0, mean2, mean), m2.clamp_min(0.0)


def look_statistics(counts, means, m2, weights, populations,
                    calibration_covariance, second_order, anchors, look_index,
                    error_probability, packet, maximum_target, u, w,
                    *, sealed: bool, dimensions: tuple[int, int, int],
                    planned_looks: int | None, oracle_covariance=None):
    """V5 stratified summary, scalar_v0_decision_interval, forecast and guard.

    anchors [L,7]: dense risk/SE, empty risk/SE/bias, connection delta, tau.
    Calibration covariance order is candidate, parent, dense. All observation
    arithmetic is float64 tensor arithmetic; static choices use Python metadata.
    """
    n = counts.clamp_min(1.0)
    sample_variance = m2.clamp_min(0.0)/(n-1.0).clamp_min(1.0)
    if sealed:
        terms = weights.square()*sample_variance*(1.0-n/populations)/n
    else:
        terms = weights.square()*sample_variance/n
    raw_mean = (weights*means).sum(-1)
    oracle_variance = terms.sum(-1)
    denominator = (terms.square()/(n-1.0).clamp_min(1.0)).sum(-1)
    df = torch.where(oracle_variance > 0,
        oracle_variance.square()/denominator.clamp_min(1e-300), torch.inf)
    alpha = (error_probability/((look_index+1.0)*(look_index+2.0))
        if planned_looks is None else error_probability/planned_looks)
    alpha = torch.broadcast_to(alpha,raw_mean.shape)
    side = alpha/2.0
    z = student_t_isf(side,df,u,w)
    raw_se = oracle_variance.clamp_min(0.0).sqrt()
    raw_lower,raw_upper = raw_mean-z*raw_se,raw_mean+z*raw_se
    dense,dense_se,empty,empty_se,empty_bias,delta,tau = anchors.unbind(-1)
    scale = delta*(empty-dense)
    c = calibration_covariance
    cross_oracle = torch.zeros_like(raw_mean) if oracle_covariance is None else oracle_covariance
    # Same gradients and operation order as the host's 3x3 quadratic forms.
    ng = torch.stack((torch.ones_like(delta),-torch.ones_like(delta),torch.zeros_like(delta)),-1)
    bg = torch.stack((torch.zeros_like(delta),torch.zeros_like(delta),-delta),-1)
    var_n = raw_se.square()+torch.einsum('li,lij,lj->l',ng,c,ng)
    var_b = delta.square()*(dense_se.square()+empty_se.square())+torch.einsum('li,lij,lj->l',bg,c,bg)
    cov_nb = -delta*cross_oracle+torch.einsum('li,lij,lj->l',ng,c,bg)
    var_t = (var_n-2.0*tau*cov_nb+tau.square()*var_b).clamp_min(0.0)
    chi = torch.stack(tuple(_chi2_isf(side,d) for d in dimensions),-1)
    bias = chi*second_order
    bias_n = bias[:,:2].sum(-1)
    bias_b = delta*(bias[:,2]+empty_bias)
    bias_t = bias_n+tau*bias_b
    statistic = raw_mean-tau*scale
    radius_t = z*var_t.sqrt()
    stat_lower,stat_upper = statistic-radius_t-bias_t,statistic+radius_t+bias_t
    mean = raw_mean/scale
    lower,upper = tau+stat_lower/scale,tau+stat_upper/scale
    current_se = var_t.sqrt()/scale
    actual = counts.sum(-1).clamp_min(1.0)
    radius = torch.maximum(mean-lower,upper-mean)
    gap = (tau-mean).abs()
    # forecast_cumulative_target, including its SE==0 and finite-cap branches.
    required = torch.ceil(actual*(radius.clamp_min(0.0)/gap.clamp_min(1e-15)).square())
    required = torch.maximum(actual+packet,required)
    analytic = torch.where((gap <= 0) | (current_se <= 0),actual+packet,required)
    analytic = torch.minimum(analytic,maximum_target)
    analytic = torch.where(actual >= maximum_target,actual,analytic)
    guarded = torch.maximum(actual+packet,torch.ceil(4.0*actual))
    target = torch.minimum(analytic,guarded)
    guard_applied = target < analytic
    if sealed:
        clamped = torch.minimum(target,populations.sum(-1))
        clamp_applied = clamped < target
        target = clamped
    else:
        clamp_applied = torch.zeros_like(target,dtype=torch.bool)
    diagonal = c.diagonal(dim1=-2,dim2=-1)
    improvements = z[:,None]*diagonal.clamp_min(0.0).sqrt()+bias
    invalid = ((counts < 2).any(-1) | (counts != counts.floor()).any(-1)
        | (~torch.isfinite(counts)).any(-1) | (~torch.isfinite(means)).any(-1)
        | (~torch.isfinite(m2)).any(-1) | (m2 < 0).any(-1)
        | (scale <= 0) | (~torch.isfinite(scale)) | (delta <= 0)
        | (tau < 0) | (~torch.isfinite(tau)) | (~torch.isfinite(var_t))
        | (~torch.isfinite(z)) | (z <= 0))
    if sealed:
        invalid = invalid | (counts > populations).any(-1)
    for index, dimension in enumerate(dimensions):
        if dimension == 0:
            invalid = invalid | (second_order[:,index] > 0.0)
    # Host order: strict accept, strict reject, point no-commit, no increment.
    # Codes 5/7 (Hall range/fixed-design forecast) do not belong to v5.
    code = torch.where(stat_upper < 0,1,torch.where(stat_lower > 0,2,
        torch.where(statistic >= 0,3,0)))
    code = torch.where((code == 0) & (analytic <= actual),4,code)
    code = torch.where(invalid,6,code).to(torch.int64)
    values = (mean,lower,upper,statistic,stat_lower,stat_upper,var_t,
        raw_mean,raw_lower,raw_upper,raw_se,df,alpha,z,
        raw_se.square(),tau.square()*delta.square()*dense_se.square(),
        tau.square()*delta.square()*empty_se.square(),diagonal[:,0],diagonal[:,1],
        (tau*delta).square()*diagonal[:,2],bias[:,0],bias[:,1],tau*delta*bias[:,2],
        tau*delta*empty_bias,improvements[:,1],improvements[:,0],improvements[:,2],
        z*raw_se,z*dense_se,var_t.sqrt(),bias_t,var_n,var_b,cov_nb,bias_n,bias_b,
        target,invalid.to(torch.float64),z.square(),analytic,guard_applied.to(torch.float64),
        clamp_applied.to(torch.float64),current_se)
    # In torch 2.7.1, stacking this wide set of computed columns lowers to
    # aten.cat and reaches Inductor's broken TensorBox input-buffer path.
    # Column copies keep the same receipt order without that concatenation.
    statistics = torch.empty((raw_mean.shape[0],len(FIELDS)),
        dtype=raw_mean.dtype,device=raw_mean.device)
    for index,value in enumerate(values):
        statistics[:,index] = value
    return statistics,code


@functools.lru_cache(maxsize=32)
def compiled_look(*, sealed: bool, dimensions=(2, 2, 2), planned_looks=None):
    def function(*args):
        return look_statistics(*args,sealed=sealed,dimensions=dimensions,planned_looks=planned_looks)
    # torch.compile(partial(...)) uses a shared variadic wrapper.  That
    # wrapper accumulates guards for unrelated kernels and sampling laws.
    # A code object per static configuration keeps their caches independent.
    function.__code__=function.__code__.replace(
        co_name=f"bce_look_{sealed}_{dimensions}_{planned_looks}")
    return CapturedTensorFunction(function)


def quadrature(device):
    return _U.to(device=device), _W.to(device=device)


def neyman_counts(target, existing, minimum, capacity, weights, sds, costs):
    """Exact discrete marginal-priority allocation, including declaration ties.

    Batch axis may enumerate independent targets.  A log-priority bisection
    finds the last purchased marginal row without enumerating all rows.
    At its boundary, a stable sort supplies declaration-order ties.
    """
    start = torch.maximum(existing, minimum)
    reachable = torch.minimum(target, capacity.sum(dim=-1))
    K = (reachable-start.sum(dim=-1)).clamp_min(0.0)
    coefficient = weights.square()*sds.square().clamp_min(1e-12)/costs
    first = coefficient/(start.clamp_min(1.0)*(start+1.0))
    last = coefficient/((capacity-1.0).clamp_min(1.0)*capacity.clamp_min(1.0))
    available = capacity > start
    hi = torch.log(torch.where(available, first, torch.zeros_like(first)).amax(dim=-1).clamp_min(1e-300))+1.0
    lo = torch.log(torch.where(available, last, torch.full_like(last, torch.inf)).amin(dim=-1).clamp_min(1e-300))-1.0
    lo = torch.where(torch.isfinite(lo), lo, hi-1.0)
    def purchases(log_threshold):
        threshold = torch.exp(log_threshold)[...,None]
        root = (torch.sqrt(1.0+4.0*coefficient/threshold)-1.0)*0.5
        end = torch.minimum(capacity, torch.maximum(start, torch.ceil(root)))
        # Correct a rounded integer root using the defining float64 priority.
        prior = coefficient/((end-1.0).clamp_min(1.0)*end.clamp_min(1.0))
        end = torch.where((end > start) & (prior <= threshold), end-1.0, end)
        next_priority = coefficient/(end.clamp_min(1.0)*(end+1.0))
        end = torch.where((end < capacity) & (next_priority > threshold), end+1.0, end)
        return end-start
    for _ in range(64):
        mid = (lo+hi)*0.5
        more = purchases(mid).sum(dim=-1) > K
        lo, hi = torch.where(more, mid, lo), torch.where(more, hi, mid)
    added = purchases(hi)
    end = start+added
    priorities = torch.where(end < capacity, coefficient/(end.clamp_min(1.0)*(end+1.0)), -torch.inf)
    order = torch.argsort(priorities, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    remaining = (K-added.sum(dim=-1)).clamp_min(0.0)
    result = end+((rank < remaining[...,None]) & (end < capacity)).to(torch.float64)
    return torch.where((reachable >= capacity.sum(dim=-1))[...,None], capacity, result).to(torch.int64)


compiled_neyman_counts = CapturedTensorFunction(neyman_counts)


def precision_plan(statistics, option_counts, option_targets, option_seconds, eligible):
    """The reference scalar_precision_forecasts and frozen role tie order.

    Options: candidate calibration, parent calibration, dense calibration,
    dense oracle, paired oracle.  The reduction is a scheduling quantity;
    it leaves the scalar interval unchanged.
    """
    variance = torch.stack(tuple(statistics[:,INDEX[k]] for k in (
        "candidate_calibration_variance", "parent_calibration_variance",
        "dense_calibration_variance_scaled", "dense_oracle_variance_scaled",
        "paired_oracle_variance")), dim=-1)
    bias = torch.stack((statistics[:,INDEX["candidate_second_order_bias"]],
                        statistics[:,INDEX["parent_second_order_bias"]],
                        statistics[:,INDEX["dense_second_order_bias_scaled"]],
                        torch.zeros_like(statistics[:,0]), torch.zeros_like(statistics[:,0])), dim=-1)
    total_v, total_b = variance.sum(-1), bias.sum(-1)
    z = statistics[:,INDEX["scalar_quantile"]]
    current = 2.0*(z*total_v.sqrt()+total_b)
    factor = option_counts/option_targets.clamp_min(1.0)
    projected = 2.0*(z[:,None]*(total_v[:,None]-variance*(1.0-factor)).clamp_min(0.0).sqrt()
                    + (total_b[:,None]-bias*(1.0-factor)).clamp_min(0.0))
    reduction = (current[:,None]-projected).clamp_min(0.0)
    utility = torch.where(eligible, reduction/option_seconds.clamp_min(1e-300), -torch.inf)
    best = utility.amax(-1, keepdim=True)
    tie = eligible & (utility == best)
    reduction = torch.where(tie, reduction, -torch.inf)
    tie = tie & (reduction == reduction.amax(-1, keepdim=True))
    choice = tie.to(torch.int64).argmax(-1)  # first = frozen role order
    return choice, reduction, utility


compiled_precision_plan = CapturedTensorFunction(precision_plan)


def warm_statistics(*, device, stratum_count, sealed, dimensions=(2, 2, 2),
                    planned_looks=None, packet_images=256):
    """Compile and capture the worker's fixed statistics and planning lanes."""
    device = torch.device(device)
    def filled(shape, value):
        return torch.full(shape, value, dtype=torch.float64, device=device)
    size = int(stratum_count)
    counts, means, m2 = filled((size,),2.),filled((1,size),0.),filled((1,size),1.)
    weights,populations = filled((size,),1./size),filled((size,),1000000.)
    covariance,second = filled((1,3,3),0.),filled((1,3),0.)
    anchors = torch.tensor([[.25,0.,.69,0.,0.,.05,.01]],dtype=torch.float64,device=device)
    look,error = filled((1,),0.),filled((1,),.05)
    packet,maximum = filled((1,),float(packet_images)),filled((1,),math.inf)
    u,w = quadrature(device)
    stats,_ = compiled_look(sealed=sealed,dimensions=tuple(dimensions),planned_looks=planned_looks)(
        counts,means,m2,weights,populations,covariance,second,anchors,look,error,packet,maximum,u,w)
    option_counts,targets,seconds = filled((1,5),2.),filled((1,5),4.),filled((1,5),1.)
    eligible = torch.ones((1,5),dtype=torch.bool,device=device)
    compiled_precision_plan(stats,option_counts,targets,seconds,eligible)
    from .device_allocation import compiled_next_counts,warm_continuation_planner
    compiled_next_counts(counts,means,m2,weights,populations,filled((size,),1.),
        filled((size,),2.),stats[:,INDEX["next_target"]],maximum,filled((1,),1.),sealed=sealed)
    warm_continuation_planner()
    from mechanistic_systems.measurement.device_baseline import warm as warm_baseline
    warm_baseline(device,size)


__all__ = ["FIELDS", "INDEX", "ControlReadback", "AllocationPacket", "HostTransferBuffer", "CapturedTensorFunction",
           "compiled_look", "look_statistics", "warm_statistics",
           "merge_moments", "paired_moments", "stratified_moments", "quadrature", "student_t_isf",
           "neyman_counts", "compiled_neyman_counts", "precision_plan", "compiled_precision_plan"]
