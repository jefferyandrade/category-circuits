"""Bounded, source-bound numerical C replay of an actual frozen binary action.

This is the observation producer, not a second statistical controller. Its
six-role envelopes enclose arithmetic on recorded floats; they do not enclose
model/operator rounding error. Separate raw-logit and displacement intervals
reuse the retained storage/execution error model. Exact population completion
bounds reuse the retained finite-population owner and preserve that conditional
numerical scope; they cannot open K/O or commit a mask.
The retained C sampler supplies the independent order. ReLU and pooling counts
describe all sampled half/full-dose switches, not a whole-path certificate.
"""
from dataclasses import dataclass, asdict, field
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tnf

from mechanistic_systems.contracts.content_address import content_hash
from ..modern_v1_physical_metric import (
    BALANCED_RMS, six_role_observation_bounds, squared_displacement_interval,
    balanced_role_bound, _outward_fraction,
)
from mechanistic_systems.evidence import finite_population as population_bounds
from ..modern_v1_signed_work_physical_action_six_role_certification import _single_interval_status
from . import masked_alexnet_paired_bce as leases
from .alexnet_factors import capture_factor_batches
from .discovery import FrozenOperatorProposal
from .genuine_issuer_bridge import torch_mask_from_hard_mask
from .operators import ChartOperators
from .logit_intervals import RawTargetLogitIntervalEvaluator
from .c_support import derive_operator_C_raw_support


@dataclass(frozen=True, slots=True)
class OperatorCReplayRequest:
    producer_hash: str
    stratum: str
    start: int
    context_ids: tuple[str, ...]

    @property
    def request_hash(self):
        return content_hash(asdict(self), prefix="v1-operator-C-replay-request-")


@dataclass(frozen=True, slots=True)
class OperatorCReplayBatch:
    producer_hash: str
    request_hash: str
    source_batch_hash: str
    stratum: str
    context_ids: tuple[str, ...]
    physical_batch_rows: int
    parent_logits: tuple[float, ...]
    literal_j: tuple[float, ...]
    dosed_logits: tuple[tuple[float, ...], ...]
    observations_lower: tuple[tuple[float, ...], ...]
    observations_upper: tuple[tuple[float, ...], ...]
    relu_switch_counts: tuple[tuple[int, ...], ...]
    pool_switch_counts: tuple[tuple[int, ...], ...]
    raw_logit_intervals_lower: tuple[tuple[float, ...], ...] | None = field(default=None, kw_only=True)
    raw_logit_intervals_upper: tuple[tuple[float, ...], ...] | None = field(default=None, kw_only=True)
    raw_logit_interval_evaluator_hash: str | None = field(default=None, kw_only=True)
    raw_logit_interval_profile_hash: str | None = field(default=None, kw_only=True)
    raw_logit_support_receipt_hash: str | None = field(default=None, kw_only=True)
    dense_logits: tuple[float, ...] = field(default=(), kw_only=True)
    dense_logit_intervals_lower: tuple[float, ...] | None = field(default=None, kw_only=True)
    dense_logit_intervals_upper: tuple[float, ...] | None = field(default=None, kw_only=True)

    def to_json(self):
        displacement_lower, displacement_upper = self.conditional_displacement_bounds()
        return dict(schema="v1_operator_numerical_C_replay_batch_v1", **asdict(self),
                    physical_measurement_profile=BALANCED_RMS, doses=(0.5, 1.0),
                    input_enclosure_scope="recorded_floats_only_no_model_rounding_bound",
                    raw_logit_interval_scope=(None if self.raw_logit_intervals_lower is None else
                        "conditional_retained_storage_execution_error_model_parent_half_full"),
                    conditional_displacement_squared_lower=displacement_lower,
                    conditional_displacement_squared_upper=displacement_upper,
                    dense_reference='all_routes_original_checkpoint_layer_and_readout_biases',
                    dense_relative_displacement_squared_bounds=self.conditional_dense_displacement_bounds(),
                    switch_scope="all_sampled_gates_no_relevance_or_whole_path_certificate",
                    population_C_certified=False, mask_commit_authority=False)

    def conditional_displacement_bounds(self):
        if self.raw_logit_intervals_lower is None:
            return None, None
        lower, upper = [], []
        for lo, hi in zip(self.raw_logit_intervals_lower, self.raw_logit_intervals_upper, strict=True):
            bounds = [squared_displacement_interval(parent_lower=lo[0], parent_upper=hi[0],
                      child_lower=lo[dose], child_upper=hi[dose]) for dose in (1,2)]
            lower.append(tuple(value[0] for value in bounds))
            upper.append(tuple(value[1] for value in bounds))
        return tuple(lower), tuple(upper)

    def conditional_dense_displacement_bounds(self):
        if self.dense_logit_intervals_lower is None:
            return None
        return tuple(squared_displacement_interval(parent_lower=dl, parent_upper=du,
            child_lower=lo[2], child_upper=hi[2]) for dl,du,lo,hi in zip(
                self.dense_logit_intervals_lower, self.dense_logit_intervals_upper,
                self.raw_logit_intervals_lower, self.raw_logit_intervals_upper, strict=True))

    @property
    def receipt_hash(self):
        return content_hash(self.to_json(), prefix="v1-operator-C-replay-batch-")


def _tuples(values):
    return tuple(tuple(row) for row in values.tolist())


@dataclass(frozen=True, slots=True)
class OperatorPhysicalCProgress:
    """Conditional population completion bounds, not K/O or commit admission."""
    producer_hash: str
    proposal_hash: str
    support_receipt_hash: str
    metric_budget_hash: str
    counts: tuple[int, int]
    population_sizes: tuple[int, int]
    moment_lower: tuple[tuple[float, float], tuple[float, float]]
    moment_upper: tuple[tuple[float, float], tuple[float, float]]
    rms_lower: tuple[float, float]
    rms_upper: tuple[float, float]
    raw_rms_radius: float
    conditional_direct_D_status: str
    observation_chain_hash: str
    dense_relative_moment_lower: tuple[float, float]
    dense_relative_moment_upper: tuple[float, float]
    dense_relative_rms_interval: tuple[float, float]
    dense_relative_rms_radius: float | None
    conditional_cumulative_dense_D_status: str

    def to_json(self):
        return dict(schema='v1_operator_C_population_completion_v1', **asdict(self),
            doses=(.5,1.), stratum_ids=('target','complement'),
            full_C_census=self.counts==self.population_sizes,
            interval_scope='conditional_retained_numeric_bounds_exact_finite_population_completion',
            alpha_spent=0., statistical_acceleration_used=False,
            decision_role='full_dose_parent_and_cumulative_dense_D_half_dose_diagnostic',
            cumulative_dense_gate_population_role='extraction_physical_C_not_terminal_validation',
            cumulative_dense_gate_radius_policy='registered_final_raw_radius_also_applied_to_each_extraction_candidate',
            F_J_R_population_evidence_available=False,
            raw_budget_redefined=False, calibration_K_open_authority=False,
            paired_oracle_O_open_authority=False, mask_commit_authority=False,
            final_dense_relative_gate_satisfied=False)

    @property
    def receipt_hash(self):
        return content_hash(self.to_json(), prefix='v1-operator-C-population-progress-')


class OperatorPhysicalReplay:
    """One process-local C cursor per frozen discovery, with one pending request.

    A post-opening failure poisons this producer. It cannot skip/retry revealed
    observations or issue another producer on that discovery. Portable restart
    and statistical row admission remain separate integration obligations.
    """
    def __init__(self, proposal: FrozenOperatorProposal, *, include_raw_logit_intervals: bool = True):
        if type(proposal) is not FrozenOperatorProposal:
            raise TypeError("physical replay requires the exact frozen operator proposal")
        if type(include_raw_logit_intervals) is not bool:
            raise TypeError("raw-logit interval policy must be frozen as an exact bool")
        proposal.assert_live()
        discovery = proposal.discovery
        if discovery._physical_replay_owner is not None:
            raise ValueError("frozen discovery already has a physical replay owner")
        stream = proposal.open_certification_stream()
        batch, limit, dtype, _seed = discovery._settings
        self.proposal = proposal
        self.stream = stream
        self._settings = (batch, limit, dtype)
        self._orders = (stream.target_assignment_receipt.generated_order,
                        stream.complement_assignment_receipt.generated_order)
        self._capacities = stream.sampling_capacity_stratum_counts
        source_owner=discovery.sources.context_source.source_owner
        if self._capacities!=(len(source_owner.target_context_ids),len(source_owner.complement_context_ids)):
            raise ValueError('physical C must cover the complete registered source population')
        self._cursor = [0, 0]
        self._pending = None
        self._pending_hash = None
        self._running = False
        self._poisoned = False
        self._admission_owner = None
        self._interval_evaluator = (RawTargetLogitIntervalEvaluator(discovery.sources.net)
                                   if include_raw_logit_intervals else None)
        discovery.release_factors(proposal)
        self.raw_support = (None if self._interval_evaluator is None else
                            derive_operator_C_raw_support(proposal, self._interval_evaluator))
        metric_path = Path(__file__).parent.parent / "modern_v1_physical_metric.py"
        decision_path = metric_path.with_name('modern_v1_signed_work_physical_action_six_role_certification.py')
        self.producer_hash = content_hash(dict(
            schema="v1_operator_numerical_C_replay_producer_v1",
            proposal_hash=proposal.proposal_hash, C_stream_hash=stream.receipt_hash,
            metric_budget_hash=discovery.metric_budget.budget_hash,
            source_opening_hash=discovery.sources.opening_hash,
            physical_batch_rows=batch, max_factor_bytes=limit, factor_dtype=str(dtype),
            physical_measurement_profile=BALANCED_RMS,
            raw_logit_interval_evaluator_hash=(None if self._interval_evaluator is None else
                                               self._interval_evaluator.evaluator_hash),
            raw_logit_support_receipt_hash=None if self.raw_support is None else self.raw_support.receipt_hash,
            last_batch_policy="repeat_last_image_to_fixed_batch_discard_padding_outputs",
            source_sha256={"physical_replay.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           metric_path.name: hashlib.sha256(metric_path.read_bytes()).hexdigest(),
                           decision_path.name: hashlib.sha256(decision_path.read_bytes()).hexdigest(),
                           'finite_population.py': hashlib.sha256(Path(population_bounds.__file__).read_bytes()).hexdigest()},
            population_C_certified=False, mask_commit_authority=False),
            prefix="v1-operator-C-replay-producer-")
        self._binding = self._coordinates()
        self._lower_sums=((Fraction(),Fraction(),Fraction()),(Fraction(),Fraction(),Fraction()))
        self._upper_sums=self._lower_sums
        self._row_chain=content_hash(dict(producer_hash=self.producer_hash,initial=True),
                                    prefix='v1-operator-C-population-chain-')
        self._progress=self._build_population_progress(tuple(self._cursor), self._lower_sums,
                                                      self._upper_sums, self._row_chain)
        self._population_binding=self._population_coordinates()
        discovery._physical_replay_owner = self
        self.assert_live()

    def _coordinates(self):
        return (self.proposal.proposal_hash, self.stream.receipt_hash,
                self._settings, self._orders, self._capacities, self.producer_hash,
                None if self._interval_evaluator is None else self._interval_evaluator.evaluator_hash,
                None if self.raw_support is None else self.raw_support.receipt_hash)

    @property
    def cumulative_counts(self):
        return tuple(self._cursor)

    def _population_coordinates(self):
        sums=tuple((v.numerator,v.denominator) for axis in (self._lower_sums,self._upper_sums)
                   for row in axis for v in row)
        return (tuple(self._cursor),sums,self._row_chain,
                None if self._progress is None else self._progress.receipt_hash)

    @property
    def population_progress(self):
        self.assert_live()
        if self._progress is None:
            raise ValueError('population completion requires conditional raw-logit intervals')
        return self._progress

    def _build_population_progress(self, counts, lower_sums, upper_sums, chain):
        if self.raw_support is None:
            return None
        lower=[];upper=[]
        supports = (*self.raw_support.squared_displacement_upper,
                    self.raw_support.dense_relative_squared_displacement_upper)
        for s in range(2):
            row=[population_bounds.finite_population_feasible_mean_bounds(
                population_size=self._capacities[s],sample_size=counts[s],
                observed_lower_sum=lower_sums[s][d],observed_upper_sum=upper_sums[s][d],
                lower_bound=Fraction(),
                upper_bound=Fraction.from_float(supports[d]))
                for d in range(3)]
            lower.append(tuple(_outward_fraction(v[0],direction='lower') for v in row))
            upper.append(tuple(_outward_fraction(v[1],direction='upper') for v in row))
        rms=[]
        for direction,values in (('lower',lower),('upper',upper)):
            rms.append(tuple(balanced_role_bound(stratum_ids=('target','complement'),
                values=tuple(values[s][d] for s in range(2)),take_root=True,direction=direction)
                for d in range(3)))
        budget=self.proposal.discovery.metric_budget
        radius=budget.final_dense_relative_requirement.rms_radius
        dense_interval=(rms[0][2],rms[1][2])
        return OperatorPhysicalCProgress(self.producer_hash,self.proposal.proposal_hash,
            self.raw_support.receipt_hash,budget.budget_hash,counts,self._capacities,
            tuple(v[:2] for v in lower),tuple(v[:2] for v in upper),rms[0][:2],rms[1][:2],budget.response_rms_radius,
            _single_interval_status((rms[0][1],rms[1][1]),budget=budget.response_rms_radius).value,
            chain,tuple(v[2] for v in lower),tuple(v[2] for v in upper),dense_interval,radius,
            'unregistered' if radius is None else _single_interval_status(dense_interval,budget=radius).value)

    def _prepare_population_update(self, row, index, stop):
        lower=[list(v) for v in self._lower_sums]
        upper=[list(v) for v in self._upper_sums]
        if self.raw_support is not None:
            lo,hi=row.conditional_displacement_bounds()
            if lo is None or len(lo)!=len(row.context_ids):
                raise ValueError('C population row lacks aligned conditional displacement bounds')
            dense=row.conditional_dense_displacement_bounds()
            if dense is None or len(dense)!=len(row.context_ids):
                raise ValueError('C population row lacks original-dense displacement bounds')
            supports=(*self.raw_support.squared_displacement_upper,
                      self.raw_support.dense_relative_squared_displacement_upper)
            for l,u,(dl,du) in zip(lo,hi,dense,strict=True):
                l,u=(*l,dl),(*u,du)
                for d in range(3):
                    # Two independently valid enclosures may be intersected.
                    # Raw global support need not contain a looser point upper.
                    point_lower=max(0.,l[d])
                    point_upper=min(supports[d],u[d])
                    if point_lower>point_upper:
                        raise ValueError('C point interval contradicts registered population support')
                    lower[index][d]+=Fraction.from_float(point_lower)
                    upper[index][d]+=Fraction.from_float(point_upper)
        lower=tuple(tuple(v) for v in lower);upper=tuple(tuple(v) for v in upper)
        counts=list(self._cursor);counts[index]=stop
        chain=content_hash(dict(previous=self._row_chain,row_hash=row.receipt_hash),
                           prefix='v1-operator-C-population-chain-')
        progress=self._build_population_progress(tuple(counts),lower,upper,chain)
        progress_hash=None if progress is None else progress.receipt_hash
        sums=tuple((v.numerator,v.denominator) for axis in (lower,upper) for values in axis for v in values)
        binding=(tuple(counts),sums,chain,progress_hash)
        return lower,upper,chain,progress,binding

    def assert_live(self):
        if self._poisoned:
            raise ValueError("physical C producer poisoned after an opened request failed")
        self.proposal.assert_live()
        self.stream.assert_integrity()
        if self._interval_evaluator is not None:
            if self._interval_evaluator.net is not self.proposal.discovery.sources.net:
                raise ValueError("point interval evaluator differs from actual C model source")
            self._interval_evaluator.assert_live()
        if (self.proposal.discovery._physical_replay_owner is not self
                or self._coordinates() != self._binding):
            raise ValueError("physical C replay owner or source coordinates changed")
        if self._population_coordinates()!=self._population_binding:
            raise ValueError('physical C population sufficient statistics changed')

    def next_request(self, stratum: str) -> OperatorCReplayRequest | None:
        self.assert_live()
        if self._admission_owner is not None:
            raise ValueError('physical C is sealed for native candidate admission')
        if stratum not in ("target", "complement"):
            raise ValueError("physical C requires target or complement")
        if self._pending is not None or self._running:
            raise ValueError("physical C already has a pending request")
        index = 0 if stratum == "target" else 1
        start = self._cursor[index]
        stop = min(start + self._settings[0], self._capacities[index])
        if stop == start:
            return None
        request = OperatorCReplayRequest(self.producer_hash, stratum, start,
                                        tuple(self._orders[index][start:stop]))
        self._pending = request
        self._pending_hash = request.request_hash
        return request

    def execute(self, request: OperatorCReplayRequest) -> OperatorCReplayBatch:
        # Validate the exact issued prefix before opening any images/outcomes.
        self.assert_live()
        if self._admission_owner is not None:
            raise ValueError('physical C is sealed for native candidate admission')
        if (type(request) is not OperatorCReplayRequest or request is not self._pending
                or request.request_hash != self._pending_hash or self._running):
            raise ValueError("physical C requires the exact pending unmodified request")
        index = 0 if request.stratum == "target" else 1
        start = self._cursor[index]
        stop = min(start + self._settings[0], self._capacities[index])
        if (request.producer_hash != self.producer_hash or request.start != start
                or request.context_ids != tuple(self._orders[index][start:stop])):
            raise ValueError("physical C request is not the next frozen sampling prefix")
        self._running = True
        try:
            result = self._replay(request)
            self.assert_live()
            if (type(result) is not OperatorCReplayBatch or result.producer_hash!=self.producer_hash
                    or result.request_hash!=request.request_hash or result.stratum!=request.stratum
                    or result.context_ids!=request.context_ids):
                raise ValueError('physical C row differs from its issued source request')
            # Hash publication must succeed before advancing the sampling cursor.
            result.receipt_hash
            population_update=self._prepare_population_update(result,index,stop)
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._running = False
        self._cursor[index] = stop
        (self._lower_sums,self._upper_sums,self._row_chain,
         self._progress,self._population_binding)=population_update
        self._pending = self._pending_hash = None
        return result

    def _replay(self, request):
        discovery = self.proposal.discovery
        sources = discovery.sources
        net = sources.net
        batch_rows, limit, dtype = self._settings
        batch = sources.context_source(request.stratum, request.context_ids)
        if self.raw_support is not None:
            self.raw_support.assert_images_in_domain(batch.images)
        batch_leases = leases._batch_tensor_leases(batch)
        images = batch.images
        count = len(request.context_ids)
        if len(images) != count:
            raise ValueError("actual C image count differs from requested prefix")
        if count < batch_rows:
            images = torch.cat((images, images[-1:].expand(batch_rows-count, -1, -1, -1)), dim=0)
        target = discovery.metric_budget.target_class
        factors = capture_factor_batches(net, (images,), context_count=batch_rows,
            target_class=target, factor_dtype=dtype, max_factor_bytes=limit)
        weights = torch.full((batch_rows,), 1 / batch_rows, device=factors.device, dtype=dtype)
        ops = ChartOperators.build(factors, net.axis, weights)
        child_mask = torch_mask_from_hard_mask(self.proposal.packed_action.resulting_mask,
                                              net=net, route_axis=sources.route_axis)
        z = (net.mask - child_mask).to(dtype)
        cells = ops.a_forward(z)[:count]
        work = cells.abs().sum(dim=1, dtype=torch.float64) / ops.cut_count
        j = ops.j_forward(z)[:count].to(torch.float64).cpu().numpy()
        parent = factors.target_logits()[:count].to(torch.float64).cpu().numpy()
        parent_activations = tuple((layer.geometry.name, layer.a) for layer in factors.layers)
        del cells, ops, factors, z
        dosed = np.empty((count, 2))
        relu_counts = np.zeros((count, 2), dtype=np.int64)
        pool_counts = np.zeros((count, 2), dtype=np.int64)
        # All three forwards use identical images, batch shape and circuit biases.
        # The parent capture also supplies the local linearization for this C row.
        with torch.no_grad():
            dense_mask = torch.ones_like(net.mask)
            dense_logits = net.forward(images, mask_override=dense_mask,
                use_dense_biases=True)['logits'][:count,target].to(torch.float64).cpu().numpy()
            for dose_index, dose in enumerate((0.5, 1.0)):
                out = net.forward(images, capture=True,
                                  mask_override=net.mask - dose * (net.mask-child_mask))
                dosed[:, dose_index] = out["logits"][:count, target].to(torch.float64).cpu().numpy()
                for (name, captured_a), child_a in zip(parent_activations, out["preacts"], strict=True):
                    parent_a = captured_a.reshape_as(child_a)[:count]
                    child_a = child_a[:count]
                    switches = ((parent_a > 0) != (child_a > 0)).flatten(1).sum(dim=1)
                    relu_counts[:, dose_index] += switches.cpu().numpy()
                    if name in net.pool_after:
                        _, pi = tnf.max_pool2d(parent_a.relu(), 3, 2, return_indices=True)
                        _, ci = tnf.max_pool2d(child_a.relu(), 3, 2, return_indices=True)
                        pool_counts[:, dose_index] += (pi != ci).flatten(1).sum(dim=1).cpu().numpy()
                del out
        f = work.cpu().numpy()
        lower, upper = np.empty((count, 6)), np.empty((count, 6))
        for row in range(count):
            lo, hi = six_role_observation_bounds(
                f_lower=f[row:row+1], f_upper=f[row:row+1],
                j_center=j[row:row+1], j_radius=np.zeros(1),
                child_center=dosed[row:row+1], child_radius=np.zeros((1, 2)),
                parent_center=float(parent[row]), parent_radius=0.,
                physical_measurement_profile=BALANCED_RMS)
            lower[row], upper[row] = lo[0], hi[0]
        leases._check_batch_tensor_leases(batch, batch_leases)
        raw_lower = raw_upper = interval_hash = profile_hash = None
        dense_lower = dense_upper = None
        if self._interval_evaluator is not None:
            raw_lower, raw_upper = self._interval_evaluator.evaluate(
                batch.images, masks=(net.mask, net.mask-.5*(net.mask-child_mask), child_mask, dense_mask),
                target_class=target, use_dense_biases=(False,False,False,True))
            dense_lower,dense_upper=raw_lower[:,3],raw_upper[:,3]
            if (not bool(np.isfinite(dense_logits).all())
                    or bool(np.any(dense_logits<dense_lower)) or bool(np.any(dense_logits>dense_upper))
                    or bool(np.any(dense_logits<self.raw_support.dense_logit_lower))
                    or bool(np.any(dense_logits>self.raw_support.dense_logit_upper))):
                raise ValueError('actual original-dense logits lie outside the retained numerical bounds/support')
            dense_lower,dense_upper=tuple(dense_lower.tolist()),tuple(dense_upper.tolist())
            raw_lower,raw_upper=raw_lower[:,:3],raw_upper[:,:3]
            recorded = np.column_stack((parent, dosed))
            if (not bool(np.isfinite(recorded).all())
                    or bool(np.any(recorded < raw_lower)) or bool(np.any(recorded > raw_upper))):
                raise ValueError("actual C logits lie outside the retained numerical error-model intervals")
            if (bool(np.any(recorded < self.raw_support.raw_logit_lower))
                    or bool(np.any(recorded > self.raw_support.raw_logit_upper))):
                raise ValueError("actual C logits exceed the registered raw-logit support")
            raw_lower, raw_upper = _tuples(raw_lower), _tuples(raw_upper)
            interval_hash = self._interval_evaluator.evaluator_hash
            profile_hash = self._interval_evaluator.profile.profile_hash
        leases._check_batch_tensor_leases(batch, batch_leases)
        self.assert_live()
        return OperatorCReplayBatch(self.producer_hash, request.request_hash,
            batch.source_batch_receipt_hash, request.stratum, request.context_ids, batch_rows,
            tuple(parent.tolist()), tuple(j.tolist()), _tuples(dosed), _tuples(lower), _tuples(upper),
            _tuples(relu_counts), _tuples(pool_counts),
            raw_logit_intervals_lower=raw_lower, raw_logit_intervals_upper=raw_upper,
            raw_logit_interval_evaluator_hash=interval_hash, raw_logit_interval_profile_hash=profile_hash,
            dense_logits=tuple(dense_logits.tolist()), dense_logit_intervals_lower=dense_lower,
            dense_logit_intervals_upper=dense_upper,
            raw_logit_support_receipt_hash=None if self.raw_support is None else self.raw_support.receipt_hash)
