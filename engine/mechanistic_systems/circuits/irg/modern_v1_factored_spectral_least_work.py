"""Native spectral least-work candidates; exact-F dual authority stays closed.

Phase A's issued resource lower bound and copied live primal are inseparable.
The numerical kernel optimizes stored binary64 weights and Bhat Gram + rho I.
Final physical/resource/Q/exact-weight F replay is a distinct primal gate. The
raw numerical dual diagnostics never become an exact registered-F lower bound.
"""
from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import dataclass, fields
from enum import Enum
from fractions import Fraction
import copy
import math
from types import MappingProxyType
import time

import torch

from mechanistic_systems.contracts.content_address import content_hash
from . import modern_v1_signed_work_kkt_cuda as core
from . import modern_v1_signed_work_resource_frontier_cuda as frontier
from .modern_v1_factored_post_d_device_proposal_source_join import V1FactoredPostDDeviceProposalSourceJoin
from .modern_v1_factored_spectral_resource_frontier import (
    V1FactoredSpectralResourceFrontierExecution, V1FactoredSpectralResourceFrontierReceipt,
)
from .modern_v1_factored_signed_work_operator import V1FactoredSignedWorkOperator

_PREFIX = 'irg-v1-native-factored-spectral-least-work-'
_TOKEN = object()
_SPECTRAL = core.V1RelaxedKKTBranch.SPECTRAL
_FALSE_AUTHORITY = dict(exact_F_dual_lower_authority=False, exact_F_optimality_authority=False,
    kkt_authority=False, action_authority=False, d_decision_authority=False,
    c_measurement_authority=False, calibration_k_authority=False, paired_oracle_o_authority=False,
    mask_commit_authority=False, terminal_authority=False, production_authority=False,
    performance_authority=False, pareto_authority=False)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _custody(t):
    _require(type(t) is torch.Tensor and not t.requires_grad, 'native tensor requires exact non-autograd custody')
    return core._state_tensor_storage_invariant(t)


def _up_fraction(value):
    result = float(value)
    _require(math.isfinite(result), 'native scale bound overflow')
    return math.nextafter(result, math.inf) if Fraction(result) < value else result


def _sqrt_up(value):
    _require(math.isfinite(value) and value >= 0, 'native norm argument is not finite nonnegative')
    result = math.sqrt(value)
    while Fraction(result)**2 < Fraction(value):
        result = math.nextafter(result, math.inf)
    return result


def _raw_payload(value):
    """Keep unresolved ±infinity distinct without asking the old hash to accept it."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return {'extended_real': 'positive_infinity' if value == math.inf else
                'negative_infinity' if value == -math.inf else 'nan'}
    if hasattr(value, '__dataclass_fields__'):
        return {f.name:_raw_payload(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_raw_payload(v) for v in value]
    if isinstance(value, dict):
        return {k:_raw_payload(v) for k,v in value.items()}
    return value


@dataclass(frozen=True)
class _PhysicalReplayInputs:
    block_starts_stops: tuple
    lower: object


def _primal_replay(*, geometry, resource, eligible, required, blocks, primal, resource_floor):
    """Use the same physical resource/compact-block owners as native Phase A."""
    n = resource.numel()
    z = primal.reshape(n)
    _require(z.dtype == torch.float64 and z.device == resource.device and not z.requires_grad,
             'native replay primal differs from source dtype/device')
    _require(bool((torch.isfinite(z) & (z >= 0) & (z <= 1)).all().item())
             and bool((z[~eligible] == 0).all().item())
             and bool((z[required] == 1).all().item()), 'native exact physical box failed')
    lower = required[0].to(torch.float64).expand(n) if required.stride() == (0,) else required.to(torch.float64)
    replay = _PhysicalReplayInputs(tuple((b.start,b.stop) for b in blocks), lower)
    # This helper only reads block_capacities from its first argument. No old
    # problem/receipt is impersonated; the explicit lightweight view is local.
    @dataclass(frozen=True)
    class _Blocks:
        block_capacities: tuple
    frontier._frontier_repair_and_certify_physical_blocks(_Blocks(blocks), replay, z, allow_inward_repair=False)
    center = float(torch.dot(resource, z).item())
    rlo,rhi = frontier._frontier_resource_interval(center, route_count=n, absolute_scale=abs(center))
    qcenter,qlo,qhi = geometry.quadratic_enclosure(z.reshape(n,1).contiguous())
    values = dict(resource_center=center, resource_lower=rlo, resource_upper=rhi,
        stored_Bhat_Q=float(qcenter.item()), ideal_Q_lower=float(qlo.item()), ideal_Q_upper=float(qhi.item()))
    _require(all(math.isfinite(v) and v >= 0 for v in values.values()), 'nonfinite native primal replay')
    values['resource_floor_verified'] = rlo >= resource_floor
    values['physical_box_and_blocks_verified'] = True
    return values


class V1FactoredPhaseALeastWorkSeed:
    """Copy the live Phase-A witness before releasing its exclusive Q lease."""
    def __init__(self, *, phase_a_execution, least_work_policy):
        _require(type(phase_a_execution) is V1FactoredSpectralResourceFrontierExecution,
                 'exact live native Phase-A execution required')
        phase_a_execution.assert_live()
        receipt = phase_a_execution.receipt
        _require(type(receipt) is V1FactoredSpectralResourceFrontierReceipt, 'native Phase-A receipt required')
        c = receipt.coordinates
        _require(c['status'] == 'OPTIMAL_CONSERVATIVE_SPECTRAL_COORDINATE'
                 and c['conservative_solver_status'] == 'optimal'
                 and c['ideal_q_primal_feasibility_verified'] is True
                 and c['conservative_spectral_coordinate_available'] is True,
                 'Phase A has no qualified native resource coordinate')
        problem = phase_a_execution.problem
        _require(type(least_work_policy) is core.V1SignedWorkLeastWorkCUDAPolicy,
                 'exact native least-work seed policy required')
        least_work_policy.__post_init__()
        _require(torch.are_deterministic_algorithms_enabled()
                 and not torch.is_deterministic_algorithms_warn_only_enabled(),
                 'native Phase-A seed replay requires strict deterministic algorithms')
        private = problem.device.type == 'cpu'
        _require(not private or (least_work_policy.allow_private_cpu_reference
                 and not least_work_policy.production_cuda_required
                 and problem.route_count <= least_work_policy.materialize_route_limit),
                 'private native seed policy not admitted')
        _require(private or (problem.device.type == 'cuda' and least_work_policy.production_cuda_required
                 and not least_work_policy.allow_private_cpu_reference), 'production CUDA seed policy required')
        requested = torch.device(least_work_policy.torch_device)
        _require(requested.type == problem.device.type and (requested.index is None or requested.index == problem.device.index),
                 'native seed policy device differs')
        self._policy_hash = least_work_policy.policy_hash
        self._join = problem._source_join
        self._join_hash = self._join.assert_live()
        _require(c['source_join_receipt_hash'] == self._join_hash, 'Phase-A source join differs')
        self._phase_a_receipt = receipt
        self._phase_a_hash = receipt.receipt_hash
        self._floor = c['achieved_resource_lower']
        _require(type(self._floor) is float and math.isfinite(self._floor)
                 and self._floor > max(0., c['nonempty_resource_threshold']), 'Phase-A resource floor is not positive')
        _require(c['q_budget'] == problem.quadratic_budget, 'Phase-A Q budget differs')
        # Admit the extra copy plus public replay temporaries while the Phase-A
        # primal, duals, Q and structural leases are still resident.
        n, rank = problem.route_count, problem.rank
        required = 13*n*8 + problem._q_handle.receipt.coordinates['tile_workspace_bytes'] + 16*rank*8
        seed_forecast = _memory_forecast(problem.device, required, least_work_policy.hbm_ceiling_fraction)
        _require(seed_forecast['admitted'], 'native Phase-A seed memory not admitted')
        self._primal = phase_a_execution.primal.reshape(n,1).clone().contiguous()
        replay = _primal_replay(geometry=problem._q_handle, resource=problem._resource,
            eligible=problem._eligible, required=problem._required, blocks=problem.block_capacities,
            primal=self._primal, resource_floor=self._floor)
        _require(replay['resource_floor_verified'] and replay['ideal_Q_upper'] <= problem.quadratic_budget,
                 'copied Phase-A seed does not replay exact issued resource floor and Q budget')
        self._primal_custody = _custody(self._primal)
        self._issued_parents = self._join, self._phase_a_receipt
        self._closed, self._borrow_count = False, 0
        self._payload = dict(schema='native_factored_phase_A_live_primal_seed_v1', phase_a_receipt_hash=self._phase_a_hash,
            source_join_receipt_hash=self._join_hash, source_geometry_hash=c['source_geometry_hash'],
            resource_floor=self._floor, q_budget=problem.quadratic_budget,
            route_count=problem.route_count, device=str(problem.device),
            private_cpu_reference_execution=problem.device.type == 'cpu', primal_copied_while_Phase_A_live=True,
            exact_Phase_A_floor_preserved=True, route_scale_d2h_bytes=0, primal_replay=replay,
            least_work_policy_hash=self._policy_hash, memory_forecast=seed_forecast,
            **_FALSE_AUTHORITY)
        self._hash = content_hash(self._payload, prefix=_PREFIX+'Phase-A-seed-')
        phase_a_execution.assert_live()

    @property
    def receipt_hash(self): return self._hash
    @property
    def resource_floor(self): return self._floor
    def assert_live(self):
        _require(not self._closed and _custody(self._primal) == self._primal_custody, 'Phase-A seed custody changed')
        _require(self._phase_a_receipt.assert_integrity() == self._phase_a_hash
                 and self._join.assert_live() == self._join_hash, 'Phase-A seed source changed')
        _require(all(a is b for a,b in zip(self._issued_parents,(self._join,self._phase_a_receipt),strict=True))
                 and self._floor == self._payload['resource_floor'] == self._phase_a_receipt.coordinates['achieved_resource_lower']
                 and self._phase_a_hash == self._payload['phase_a_receipt_hash']
                 and self._policy_hash == self._payload['least_work_policy_hash'], 'Phase-A seed issued coordinates changed')
        _require(content_hash(self._payload,prefix=_PREFIX+'Phase-A-seed-') == self._hash,
                 'Phase-A seed metadata changed')
        return self._hash
    def to_json_payload(self):
        self.assert_live()
        return copy.deepcopy(self._payload)
    def _borrow(self, join):
        self.assert_live()
        _require(join is self._join and self._borrow_count == 0, 'Phase-A seed join differs or is already borrowed')
        self._borrow_count = 1
        return self._primal
    def _release(self): self._borrow_count = 0
    def close(self):
        if self._closed: return
        _require(self._borrow_count == 0, 'Phase-A seed is borrowed')
        self._closed = True
        self._primal = self._join = None
        self._issued_parents = ()
    def __enter__(self): self.assert_live(); return self
    def __exit__(self,*_exc): self.close()
    def __reduce_ex__(self,_protocol): raise TypeError('native Phase-A seeds are nonserializable')


class V1FactoredSpectralLeastWorkProblem:
    """Native source join with exclusive Q, structure, A and seed custody."""
    def __init__(self, *, source_join, phase_a_seed, signed_work_operator, allow_private_cpu_reference=False):
        _require(type(source_join) is V1FactoredPostDDeviceProposalSourceJoin
                 and type(phase_a_seed) is V1FactoredPhaseALeastWorkSeed
                 and type(signed_work_operator) is V1FactoredSignedWorkOperator, 'exact native least-work parents required')
        _require(type(allow_private_cpu_reference) is bool, 'exact private CPU flag required')
        source_join.assert_live()
        phase_a_seed.assert_live()
        self._join, self._seed_owner, self.signed_work_operator = source_join, phase_a_seed, signed_work_operator
        c = source_join.receipt.coordinates
        _require(c['admitted_q_branch'] == 'spectral' and c['diagonal_branch_admitted'] is False,
                 'native least work is spectral only')
        _require(not c['private_cpu_reference_execution'] or allow_private_cpu_reference, 'private CPU mode not admitted')
        self._join_hash = source_join.receipt_hash
        self._a_join_hash = source_join.validate_signed_work(signed_work_operator)
        self._seed_hash = phase_a_seed.receipt_hash
        self._issued_parents = source_join, phase_a_seed, signed_work_operator
        self.resource_floor = phase_a_seed.resource_floor
        _, self._q_handle, self._structure, _, _, _ = source_join._parents
        self._q_lease = self._st_lease = self._a_lease = None
        self._seed = None
        self._closed, self._active_result = False, False
        self._solve_in_progress = False
        self._candidate_admission = None
        try:
            self._st_lease = self._structure.borrow_for_phase_b(
                expected_receipt_hash=c['structural_source_receipt_hash'],
                expected_route_axis_hash=c['physical_route_axis_hash'], expected_current_mask_hash=c['current_mask_hash'],
                expected_native_fiber_axis_receipt_hash=c['native_fiber_axis_hash'],
                expected_structural_registration_hash=c['structural_registration_hash'], expected_device=c['device'])
            self._q_lease = self._q_handle.borrow_for_kkt(expected_receipt_hash=c['rounded_q_receipt_hash'],
                expected_source_binding_hash=c['source_binding_hash'], expected_device=c['device'], branch='spectral')
            self._a_lease = signed_work_operator.borrow_for_least_work(expected_receipt_hash=signed_work_operator.receipt_hash,
                expected_source_binding_hash=c['source_binding_hash'], expected_device=c['device'])
            self._seed = phase_a_seed._borrow(source_join)
            self._resource = self._st_lease.resource_tensor
            self._eligible = self._st_lease.eligible_mask_tensor
            self._required = self._st_lease.required_hard_action_tensor
            self._factor, self._singleton = self._q_lease.factor, self._q_lease.singleton_work
            self._rank,self._route_count = self._factor.shape
            self.block_capacities = self._structure.block_capacities
            self.quadratic_budget = c['quadratic_budget']
            _require(self.quadratic_budget == phase_a_seed._payload['q_budget'], 'native seed/source Q budget differs')
            self._factor_frobenius_norm = self._q_handle.receipt.coordinates['factor_frobenius_norm_upper']
            self._residual_spectral_bound = c['roundoff_residual_spectral_bound']
            _require(signed_work_operator._design.maximum_lanes >= 2, 'native least-work candidate comparison needs two A lanes')
            _require(self._seed.shape == (self._route_count,1) and self._seed.device == self.device,
                     'Phase-A seed/source geometry differs')
            self._has_ineligible = not bool(self._eligible.all().item())
            self._has_required = bool(self._required.any().item())
            self._invariants = tuple(_custody(t) for t in self._tensors())
            self._problem_hash = content_hash(self._identity(), prefix=_PREFIX+'problem-')
            self.assert_integrity()
        except BaseException:
            self._release()
            raise

    def _tensors(self): return self._resource,self._eligible,self._required,self._factor,self._singleton,self._seed
    def _identity(self):
        return dict(schema='native_factored_spectral_least_work_problem_v1', source_join_hash=self._join_hash,
            signed_work_join_hash=self._a_join_hash, Phase_A_seed_hash=self._seed_hash,
            resource_floor=self.resource_floor, quadratic_budget=self.quadratic_budget,
            blocks=self.block_capacities, rank=self.rank, routes=self.route_count,
            factor_norm_upper=self._factor_frobenius_norm, residual_spectral_bound=self.residual_spectral_bound,
            branch='spectral', exact_Phase_A_floor_preserved=True)
    @property
    def source_problem(self): return self
    @property
    def device(self): return self._factor.device
    @property
    def problem_hash(self): return self._problem_hash
    @property
    def route_count(self): return self._route_count
    @property
    def rank(self): return self._rank
    @property
    def residual_spectral_bound(self): return self._residual_spectral_bound
    def assert_integrity(self):
        _require(not self._closed and tuple(_custody(t) for t in self._tensors()) == self._invariants,
                 'native least-work source tensor custody changed')
        _require(self._join.assert_live() == self._join_hash and self._seed_owner.assert_live() == self._seed_hash
                 and self._join.validate_signed_work(self.signed_work_operator) == self._a_join_hash,
                 'native least-work source identity changed')
        _require(all(a is b for a,b in zip(self._issued_parents,
                 (self._join,self._seed_owner,self.signed_work_operator),strict=True))
                 and self.resource_floor == self._seed_owner.resource_floor, 'native least-work parent or floor changed')
        _require(self._a_lease.operator is self.signed_work_operator and self._q_lease.factor is self._factor
                 and self._st_lease.resource_tensor is self._resource, 'native least-work lease alias changed')
        _require(content_hash(self._identity(),prefix=_PREFIX+'problem-') == self._problem_hash,
                 'native least-work problem metadata changed')
        return self._problem_hash
    def _native_source_tensors(self, *, device):
        self.assert_integrity()
        _require(torch.device(device) == self.device, 'native least-work requested another device')
        op = self.signed_work_operator
        return core._FusedSourceTensors(device=self.device,resource=self._resource,singleton_work=self._singleton,
            factor=self._factor,residual_diagonal=None,sqrt_residual=None,eligible=self._eligible,required=self._required,
            residual_work=None,work_dual_box=op.context_weights.reshape(-1,1,1)/float(op.native_fiber_axis_receipt.cut_count),
            block_bounds=torch.tensor([b.removal_capacity/math.sqrt(b.stop-b.start) for b in self.block_capacities],
                                      dtype=torch.float64,device=self.device),
            has_ineligible=self._has_ineligible,has_required=self._has_required)
    def _native_scales(self, *, active_branch):
        self.assert_integrity()
        _require(active_branch is _SPECTRAL, 'native least-work has no diagonal/fused Q branch')
        a = self.signed_work_operator.operator_frobenius_norm_upper
        if self._structure.receipt.resource_is_intrinsic_unit:
            resource_norm = _sqrt_up(self.route_count)
        else:
            from .modern_v1_factored_signed_work_operator import _mass_upper, _up
            upper = _mass_upper((self._resource*self._resource).sum(), 2*self.route_count+8)
            resource_norm = float(_up(torch.sqrt(upper)).item())
        work,resource = max(1.,a),max(1.,resource_norm)
        q = max(1.,self._factor_frobenius_norm,_sqrt_up(self.residual_spectral_bound))
        blocks = tuple(math.sqrt(b.stop-b.start) for b in self.block_capacities)
        block_bound = max((Fraction(b.stop-b.start)/Fraction(s)**2 for b,s in
                          zip(self.block_capacities,blocks,strict=True)), default=Fraction())
        tail_squared = max(Fraction(self.residual_spectral_bound), Fraction(math.sqrt(self.residual_spectral_bound))**2)
        exact_bound = (Fraction(a)**2/Fraction(work)**2 + Fraction(resource_norm)**2/Fraction(resource)**2
            + (Fraction(self._factor_frobenius_norm)**2+tail_squared)/Fraction(q)**2 + block_bound)
        bound = _up_fraction(exact_bound)
        step = math.nextafter(1/math.sqrt(bound),0.)
        while Fraction(step)**2*Fraction(bound) >= 1 or not step*step*bound < 1:
            step = math.nextafter(step,0.)
        _require(step > 0, 'native strict PDHG step unavailable')
        # q_diagonal=0 is unused storage in the shared numerical struct. The
        # exact native dispatch refuses every non-spectral branch above.
        return core._FusedScales(work,resource,q,0.,blocks,bound,step)
    def _validated_initial_primal(self, value):
        self.assert_integrity()
        _require(value is self._seed, 'native initial primal must be the issued live Phase-A copy')
        return self._seed
    def _admit_native_improving_candidate(self, value):
        """Protect the exact-floor incumbent; only called after raw improvement.

        This gate does not repair the point or certify an objective/dual bound.
        Source mutation remains fatal rather than a candidate rejection.
        """
        _require(self._solve_in_progress and self._candidate_admission is not None,
                 'native candidate admission requires an active numerical solve')
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        counts = self._candidate_admission
        counts['invocations'] += 1
        try:
            self.assert_integrity()
            try:
                # The core next copies this view into a contiguous incumbent.
                # Replay that same layout so reduction order does not change
                # merely because the two-lane candidate view has stride two.
                candidate = value.contiguous()
                replay = _primal_replay(geometry=self._q_handle,resource=self._resource,
                    eligible=self._eligible,required=self._required,blocks=self.block_capacities,
                    primal=candidate,resource_floor=self.resource_floor)
                accepted = replay['resource_floor_verified'] and replay['ideal_Q_upper'] <= self.quadratic_budget
                reason = None if accepted else 'resource_floor_or_outward_Q_replay_unresolved'
            except (ValueError,RuntimeError) as exc:
                accepted,reason = False,f'{type(exc).__name__}: {exc}'
            counts['accepted' if accepted else 'refused'] += 1
            if not accepted: counts['last_refusal'] = reason
            return accepted
        finally:
            if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
            counts['synchronized_seconds'] += time.perf_counter()-started
    def _release(self):
        try:
            if self._seed is not None: self._seed_owner._release()
            if self._a_lease is not None: self._a_lease.close()
        finally:
            try:
                if self._q_lease is not None: self._q_lease.close()
            finally:
                if self._st_lease is not None: self._st_lease.close()
                self._closed = True
    def close(self):
        if self._closed: return
        _require(not self._active_result and not self._solve_in_progress, 'close native least-work execution before its problem')
        self._release()
        self._q_lease=self._st_lease=self._a_lease=None
        self._resource=self._eligible=self._required=self._factor=self._singleton=self._seed=None
    def __enter__(self): self.assert_integrity(); return self
    def __exit__(self,*_exc): self.close()
    def __reduce_ex__(self,_protocol): raise TypeError('native least-work problems are nonserializable')


def _memory_forecast(device, incremental, ceiling_fraction):
    if device.type=='cuda':
        current=torch.cuda.memory_reserved(device)
        allocated=torch.cuda.memory_allocated(device)
        free,total=torch.cuda.mem_get_info(device)
        available=cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free)
    else:
        current,total=0,max(1<<30,4*incremental)
        available=total
    ceiling=math.floor(ceiling_fraction*total)
    return dict(current_reserved_bytes=current, incremental_bytes=incremental,
        peak_reserved_bytes=current+incremental,total_bytes=total,hbm_ceiling_bytes=ceiling,
        admitted=(allocated if device.type=='cuda' else current)+incremental<=ceiling and incremental<=available, private_cpu_forecast=device.type=='cpu')


def forecast_v1_factored_spectral_least_work(problem, policy):
    problem.assert_integrity()
    n,r=problem.route_count,problem.rank
    c,f=problem.signed_work_operator.context_count,problem.signed_work_operator.fiber_count
    b=len(problem.block_capacities)
    ac=problem.signed_work_operator.receipt.coordinates
    persistent=(c+b)*8
    state=(8*n+r+b+1+c*f)*8
    operator_workspace=ac['additional_required_bytes']-ac['owned_bytes']
    replay_workspace=max(operator_workspace+4*n*8,problem._q_handle.receipt.coordinates['tile_workspace_bytes']+6*n*8+16*c*8)
    # Improving-candidate replay occurs while z/average/candidate metrics and
    # the numerical scratch are live. Charge this overlap, not their maximum.
    numerical_transient=operator_workspace+(16*n+4*c*f+4*r)*8
    incremental=persistent+state+numerical_transient+replay_workspace
    return dict(schema='native_factored_spectral_least_work_hbm_forecast_v1',problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,**_memory_forecast(problem.device,incremental,policy.hbm_ceiling_fraction),
        candidate_admission_replay_workspace_bytes=replay_workspace,
        candidate_admission_workspace_overlaps_numerical_check=True,
        diagonal_residual_bytes=0, **_FALSE_AUTHORITY)


class V1FactoredSpectralLeastWorkExecution:
    """Actual retained solver state, with a separate outward primal gate."""
    def __init__(self, problem, raw, primal, reduced, tensors, duals, coordinates, *, token):
        _require(token is _TOKEN, 'native execution is solver-issued only')
        self._problem,self._raw,self._primal,self._reduced,self._source,self._duals=problem,raw,primal,reduced,tensors,duals
        self._issued_problem,self._issued_raw,self._issued_duals=problem,raw,duals
        self._receipt=MappingProxyType(copy.deepcopy(coordinates))
        self._hash=content_hash(dict(self._receipt),prefix=_PREFIX+'execution-')
        self._invariants=tuple(_custody(t) for t in self._state_tensors())
        self._closed,self._lease_count=False,0
        problem._active_result=True
    def _state_tensors(self):
        return (self._primal,self._reduced,self._source.resource,self._source.singleton_work,self._source.factor,
                self._source.eligible,self._source.required,self._source.work_dual_box,self._source.block_bounds,
                *(t for _,t in core._dual_state_tensor_items(self._duals)))
    @property
    def receipt_hash(self): return self._hash
    def assert_live(self):
        _require(not self._closed and tuple(_custody(t) for t in self._state_tensors())==self._invariants,
                 'native least-work retained tensor custody changed')
        self._problem.assert_integrity()
        _require(self._problem is self._issued_problem and self._raw is self._issued_raw
                 and self._duals is self._issued_duals
                 and self._problem.problem_hash == self._receipt['problem_hash']
                 and _raw_payload(self._raw) == self._receipt['raw_solver_diagnostics'], 'native least-work issued parent changed')
        _require(self._duals.branch_order==(_SPECTRAL,) and len(self._duals.q_tail_duals)==1,
                 'native least-work retained another branch')
        _require(content_hash(dict(self._receipt),prefix=_PREFIX+'execution-')==self._hash,
                 'native least-work result metadata changed')
        return self._hash
    def to_json_payload(self): self.assert_live(); return copy.deepcopy(dict(self._receipt))
    def borrow_for_witness(self):
        self.assert_live()
        _require(self._receipt['outward_primal_candidate_available'] and self._lease_count==0,
                 'native candidate unavailable or already borrowed')
        self._lease_count=1
        return V1FactoredSpectralLeastWorkWitnessLease(self,token=_TOKEN)
    def close(self):
        if self._closed:return
        _require(self._lease_count==0,'native least-work witness is borrowed')
        self._closed=True
        self._primal=self._reduced=self._source=self._duals=self._raw=None
        self._issued_raw=self._issued_duals=None
        self._issued_problem._active_result=False
    def __enter__(self):self.assert_live();return self
    def __exit__(self,*_exc):self.close()
    def __reduce_ex__(self,_protocol):raise TypeError('native least-work state is nonserializable')


class V1FactoredSpectralLeastWorkWitnessLease:
    def __init__(self, owner, *, token):
        _require(token is _TOKEN,'native witness leases are owner-issued')
        self._owner,self._closed=owner,False
        self._issued_owner=owner
    def _live(self):
        _require(not self._closed and self._owner is self._issued_owner,'native witness lease closed or owner changed')
        self._owner.assert_live()
        return self._owner
    @property
    def primal(self):return self._live()._primal
    @property
    def reduced_advantage(self):return self._live()._reduced
    @property
    def dual_tensors(self):return self._live()._duals
    @property
    def source_tensors(self):return self._live()._source
    @property
    def receipt_hash(self):return self._live().receipt_hash
    def close(self):
        if not self._closed:
            self._closed=True
            self._issued_owner._lease_count=0
            self._owner=self._issued_owner=None
    def __enter__(self):self._live();return self
    def __exit__(self,*_exc):self.close()
    def __reduce_ex__(self,_protocol):raise TypeError('native witness leases are nonserializable')


def solve_v1_factored_spectral_least_work(*, problem, policy, branch='spectral'):
    _require(type(problem) is V1FactoredSpectralLeastWorkProblem and type(policy) is core.V1SignedWorkLeastWorkCUDAPolicy,
             'exact native least-work problem/policy required')
    _require(branch=='spectral' and not problem._active_result and not problem._solve_in_progress,
             'native spectral branch unavailable or active')
    problem.assert_integrity()
    policy.__post_init__()
    _require(policy.policy_hash == problem._seed_owner._policy_hash, 'native least-work seed policy changed')
    private=problem.device.type=='cpu'
    _require(not private or (policy.allow_private_cpu_reference and not policy.production_cuda_required
             and problem.route_count<=policy.materialize_route_limit),'private native CPU policy not admitted')
    _require(private or (problem.device.type=='cuda' and policy.production_cuda_required and not policy.allow_private_cpu_reference),
             'native production CUDA policy required')
    requested=torch.device(policy.torch_device)
    _require(requested.type==problem.device.type and (requested.index is None or requested.index==problem.device.index),
             'native least-work policy device differs')
    _require(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled(),
             'native least-work requires strict deterministic algorithms')
    forecast=forecast_v1_factored_spectral_least_work(problem,policy)
    _require(forecast['admitted'],'native least-work workspace not admitted')
    if not private:torch.cuda.synchronize(problem.device)
    started=time.perf_counter()
    problem._candidate_admission = dict(invocations=0,accepted=0,refused=0,synchronized_seconds=0.,last_refusal=None)
    problem._solve_in_progress = True
    try:
        raw,primal,reduced,tensors,duals=core._solve_independent_signed_work_branch(problem=problem,policy=policy,
            branch=_SPECTRAL,device=problem.device,production=not private,private_cpu=private,initial_primal=problem._seed)
    finally:
        problem._solve_in_progress = False
    raw_payload=_raw_payload(raw)
    finite_raw=all(not isinstance(v,dict) for v in raw_payload.values())
    raw_hash=raw.receipt_hash if finite_raw else None
    replay,refusal={},None
    available=False
    try:
        _require(all(bool(torch.isfinite(t).all().item()) for t in
                     (primal,reduced,*(t for _,t in core._dual_state_tensor_items(duals)))), 'nonfinite retained numerical state')
        replay=_primal_replay(geometry=problem._q_handle,resource=problem._resource,eligible=problem._eligible,
            required=problem._required,blocks=problem.block_capacities,primal=primal,resource_floor=problem.resource_floor)
        value,lo,hi=problem.signed_work_operator.signed_work_enclosure(primal.contiguous())
        replay.update(stored_weight_F=float(value.item()),exact_weight_F_lower=float(lo.item()),exact_weight_F_upper=float(hi.item()))
        _require(all(math.isfinite(v) for v in (replay['stored_weight_F'],replay['exact_weight_F_lower'],replay['exact_weight_F_upper']))
                 and 0<=replay['exact_weight_F_lower']<=replay['exact_weight_F_upper'], 'exact-weight F replay is invalid')
        available=replay['resource_floor_verified'] and replay['ideal_Q_upper']<=problem.quadratic_budget
        if not available:refusal='resource_floor_or_outward_Q_replay_unresolved'
    except (ValueError,RuntimeError) as exc:
        refusal=f'{type(exc).__name__}: {exc}'
    problem.assert_integrity()
    if not private:torch.cuda.synchronize(problem.device)
    coordinates=dict(schema='native_factored_spectral_least_work_execution_v1',
        status='OUTWARD_PRIMAL_CANDIDATE_ONLY' if available else 'UNRESOLVED_NATIVE_PRIMAL_REPLAY',
        problem_hash=problem.problem_hash,policy_hash=policy.policy_hash,source_join_hash=problem._join_hash,
        Phase_A_seed_hash=problem._seed_hash,Phase_A_receipt_hash=problem._seed_owner._phase_a_hash,
        source_geometry_hash=problem._join.receipt.coordinates['source_geometry_hash'],
        signed_work_receipt_hash=problem.signed_work_operator.receipt_hash,rounded_Q_receipt_hash=problem._q_handle.receipt_hash,
        resource_floor=problem.resource_floor,quadratic_budget=problem.quadratic_budget,
        raw_solver_receipt_hash=raw_hash,raw_solver_diagnostics=raw_payload,
        raw_diagnostics_hash=content_hash(raw_payload,prefix=_PREFIX+'raw-stored-objective-diagnostics-'),
        old_receipt_hash_absence=None if raw_hash is not None else 'nonfinite_unresolved_sentinel_not_hashable_by_old_owner',
        stored_weight_numerical_convergence=raw.converged,raw_solver_dual_scope='diagnostic_stored_binary64_weights_only',
        exact_F_objective_weight_law=problem.signed_work_operator.receipt.coordinates['public_signed_work_enclosure_weight_law'],
        stored_F_objective_weight_law=problem.signed_work_operator.receipt.coordinates['solver_objective_weight_law'],
        exact_Phase_A_floor_preserved=True,validated_Phase_A_primal_seed_used=True,
        candidate_admission=copy.deepcopy(problem._candidate_admission),
        improving_raw_feasible_candidate_outward_replay_required=True,
        outward_primal_candidate_available=available,primal_replay=replay,replay_refusal=refusal,
        primal_or_duals_repaired_after_raw_solver=False,retained_actual_dual_tensor_names=[name for name,_ in core._dual_state_tensor_items(duals)],
        branch_order=('spectral',),diagonal_branch_admitted=False,raw_dual_promoted_to_exact_F=False,
        device=str(problem.device),private_cpu_reference_execution=private,route_scale_d2h_bytes=0,
        elapsed_seconds=time.perf_counter()-started,memory_forecast=forecast,**_FALSE_AUTHORITY)
    return V1FactoredSpectralLeastWorkExecution(problem,raw,primal,reduced,tensors,duals,coordinates,token=_TOKEN)
