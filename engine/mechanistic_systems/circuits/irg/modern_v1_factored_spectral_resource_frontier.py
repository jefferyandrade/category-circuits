"""Native spectral Phase A over the conservative rounded-J Q surrogate.

The numerical program is ||Bhat z||^2 + rho ||z||^2. Its resource bounds do not
exclude feasible points of ideal Q. A retained proposal witness additionally
passes physical replay and the rounded-Q owner's outward ideal-Q enclosure.
Only the spectral branch exists here; downstream action gates remain closed.
"""
from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field
import math
from types import MappingProxyType
import time

import torch

from mechanistic_systems.circuits.irg import modern_v1_signed_work_resource_frontier_cuda as core
from mechanistic_systems.circuits.irg.modern_v1_factored_post_d_device_proposal_source_join import (
    V1FactoredPostDDeviceProposalSourceJoin,
)
from mechanistic_systems.contracts.content_address import content_hash

_PREFIX = "irg-v1-factored-spectral-resource-frontier-"
_TOKEN = object()
_BRANCH = core.V1SignedWorkLeastWorkBranch.SPECTRAL
_NATIVE_RESOURCE_OBJECTIVE_SCALE_LAW = (
    "native_physical_unit_resource_objective_beta_1_v1"
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


class V1FactoredSignedWorkResourceFrontierCUDADeviceProblemV1:
    """One exclusive rounded-Q and structural lease; no signed-work owner."""
    def __init__(self, *, source_join, allow_private_cpu_reference=False):
        _require(type(source_join) is V1FactoredPostDDeviceProposalSourceJoin,
                 "exact native proposal source join required")
        _require(type(allow_private_cpu_reference) is bool, "exact private CPU flag required")
        source_join.assert_live()
        c = source_join.receipt.coordinates
        _require(c['admitted_q_branch'] == 'spectral' and c['diagonal_branch_admitted'] is False
                 and c['response_endpoint_family_bound'] is False and c['sampled_fd_or_row_bank_source'] is False,
                 "native spectral source required")
        private = c['private_cpu_reference_execution']
        _require(not private or allow_private_cpu_reference, "CPU execution requires explicit private reference")
        _, q, structural, _, _, _ = source_join._parents
        qc = q.receipt.coordinates
        _require(qc['numerical_scope'] == 'ideal_registered_weighted_rounded_J_Q_enclosed_by_stored_B_Gram_plus_rho_I'
                 and qc['arithmetic_law'] == 'legacy_FP32_rows_FP64_RN_or_FTZ_with_sqrt_weight_interval_v1',
                 "rounded Q arithmetic scope differs")
        self._source_join, self._q_handle, self._structural_source = source_join, q, structural
        self._source_join_hash = source_join.receipt_hash
        self._closed, self._active_result = False, False
        self._geometry_lease = self._structural_lease = None
        try:
            self._structural_lease = structural.borrow_for_phase_a(
                expected_receipt_hash=c['structural_source_receipt_hash'],
                expected_route_axis_hash=c['physical_route_axis_hash'],
                expected_current_mask_hash=c['current_mask_hash'],
                expected_native_fiber_axis_receipt_hash=c['native_fiber_axis_hash'],
                expected_structural_registration_hash=c['structural_registration_hash'],
                expected_device=c['device'])
            self._geometry_lease = q.borrow_for_kkt(expected_receipt_hash=c['rounded_q_receipt_hash'],
                expected_source_binding_hash=c['source_binding_hash'], expected_device=c['device'], branch='spectral')
            st = self._structural_lease
            self._resource, self._eligible, self._required = st.resource_tensor, st.eligible_mask_tensor, st.required_hard_action_tensor
            self._factor = self._geometry_lease.factor
            self._rank, self._route_count = qc['factor_shape']
            self._factor_frobenius_norm = qc['factor_frobenius_norm_upper']
            self._residual_spectral_bound = qc['roundoff_residual_spectral_bound']
            self.quadratic_budget = c['quadratic_budget']
            self.block_capacities = structural.block_capacities
            self.source_registration_hash = c['source_registration_hash']
            self.registered_d_coordinate_hash = c['registered_d_coordinate_hash']
            self.structural_source_receipt_hash = c['structural_source_receipt_hash']
            self.source_geometry_hash = c['source_geometry_hash']
            self.physical_route_axis_hash = c['physical_route_axis_hash']
            self.route_id_axis_hash = c['route_id_axis_hash']
            self.route_axis_hash = self.physical_route_axis_hash
            self.required_lower_bound_hash = structural.receipt.required_hard_action_identity_hash
            _require(all(math.isfinite(float(v)) and v >= 0 for v in
                         (self.quadratic_budget, self._residual_spectral_bound, self._factor_frobenius_norm)),
                     "nonfinite native spectral geometry/budget")
            # The exact structural owner already checks axis and box legality.
            _require(bool(torch.isfinite(self._resource).all().item())
                     and bool((self._resource > 0).all().item()), "resource must be positive finite")
            for block in self.block_capacities:
                _require(int(torch.count_nonzero(self._required[block.start:block.stop]).item()) <= block.removal_capacity,
                         "required action exceeds a block capacity")
            (self._resource_support, self._uniform_eligible_resource, self._literal_unit_resource,
             self._eligible_count, self._required_count, self.baseline_resource,
             self._block_eligible_counts) = core._device_phase_a_resource_summaries(
                resource=self._resource, eligible=self._eligible, required=self._required, blocks=self.block_capacities)
            self._tensor_invariants = tuple(core._frontier_state_tensor_storage_invariant(t) for t in self._tensors())
            self._problem_hash = content_hash(self._identity(), prefix=_PREFIX+'problem-')
            self.assert_integrity()
        except BaseException:
            self._release()
            raise

    def _tensors(self):
        return self._resource, self._eligible, self._required, self._factor

    def _identity(self):
        return dict(schema='native_factored_spectral_phase_A_problem_v1', source_join_receipt_hash=self._source_join_hash,
            source_registration_hash=self.source_registration_hash, registered_d_coordinate_hash=self.registered_d_coordinate_hash,
            source_geometry_hash=self.source_geometry_hash, structural_source_receipt_hash=self.structural_source_receipt_hash,
            route_axis_hash=self.route_axis_hash, physical_route_axis_hash=self.physical_route_axis_hash,
            route_id_axis_hash=self.route_id_axis_hash, quadratic_budget=self.quadratic_budget, blocks=self.block_capacities,
            required_lower_bound_hash=self.required_lower_bound_hash, baseline_resource=self.baseline_resource,
            route_count=self._route_count, rank=self._rank, factor_norm=self._factor_frobenius_norm,
            residual_spectral_bound=self._residual_spectral_bound, resource_support=self._resource_support,
            uniform_eligible_resource=self._uniform_eligible_resource, literal_unit_resource=self._literal_unit_resource,
            resource_objective_scale_law=_NATIVE_RESOURCE_OBJECTIVE_SCALE_LAW,
            resource_objective_beta=1.0, resource_dual_to_physical_scale=1.0,
            eligible_count=self._eligible_count, required_count=self._required_count,
            block_eligible_counts=self._block_eligible_counts, branch_order=('spectral',), no_signed_f_operator=True)

    @property
    def problem_hash(self): return self._problem_hash
    @property
    def device(self): return self._resource.device
    @property
    def route_count(self): return self._route_count
    @property
    def rank(self): return self._rank
    @property
    def response_rank(self): return self._rank
    @property
    def residual_spectral_bound(self): return self._residual_spectral_bound
    @property
    def resource_support(self): return self._resource_support
    @property
    def uniform_eligible_resource(self): return self._uniform_eligible_resource
    @property
    def literal_unit_resource(self): return self._literal_unit_resource
    @property
    def required_count(self): return self._required_count
    @property
    def eligible_count(self): return self._eligible_count
    @property
    def block_eligible_counts(self): return self._block_eligible_counts
    @property
    def resource(self):
        self.assert_structural_identity()
        return self._resource
    @property
    def low_rank_factor(self):
        self.assert_structural_identity()
        return self._factor
    @property
    def eligible_mask(self):
        self.assert_structural_identity()
        return self._eligible
    @property
    def required_hard_action(self):
        self.assert_structural_identity()
        return self._required

    def assert_structural_identity(self):
        _require(not self._closed, 'native spectral problem closed')
        _require(tuple(core._frontier_state_tensor_storage_invariant(t) for t in self._tensors()) == self._tensor_invariants,
                 'native spectral source tensor custody changed')
        _require(content_hash(self._identity(), prefix=_PREFIX+'problem-') == self._problem_hash,
                 'native spectral problem metadata changed')
        return self._problem_hash

    def assert_integrity(self):
        self.assert_structural_identity()
        _require(self._source_join.assert_live() == self._source_join_hash, 'native spectral source join changed')
        _require(self._geometry_lease.factor is self._factor
                 and self._structural_lease.resource_tensor is self._resource
                 and self._structural_lease.eligible_mask_tensor is self._eligible
                 and self._structural_lease.required_hard_action_tensor is self._required,
                 'native spectral lease aliases changed')
        return self._problem_hash

    def _release(self):
        try:
            if self._geometry_lease is not None:
                self._geometry_lease.close()
        finally:
            if self._structural_lease is not None:
                self._structural_lease.close()
            self._closed = True

    def close(self):
        if self._closed:
            return
        _require(not self._active_result, 'close retained spectral result before problem')
        self._release()
        self._geometry_lease = self._structural_lease = None
        self._factor = self._resource = self._eligible = self._required = None

    def __enter__(self):
        self.assert_integrity()
        return self
    def __exit__(self, *_exc): self.close()
    def __reduce_ex__(self, _protocol):
        raise TypeError('native spectral problems are nonserializable')


@dataclass(frozen=True)
class V1FactoredSpectralResourceFrontierReceipt:
    coordinates: object
    production_geometry_authority: bool = False
    response_action_authority: bool = False
    kkt_authority: bool = False
    ideal_frontier_exclusion_authority: bool = False
    action_authority: bool = False
    d_decision_authority: bool = False
    c_measurement_authority: bool = False
    calibration_k_authority: bool = False
    paired_oracle_o_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    _issuance_token: InitVar[object] = None
    _hash: str = field(init=False, repr=False)

    def __post_init__(self, _issuance_token):
        _require(_issuance_token is _TOKEN, 'spectral result receipts are owner-issued')
        object.__setattr__(self, 'coordinates', MappingProxyType(dict(self.coordinates)))
        object.__setattr__(self, '_hash', content_hash(self.to_json_payload(), prefix=_PREFIX+'result-'))
        self.assert_integrity()

    @property
    def receipt_hash(self): return self._hash
    def to_json_payload(self):
        return {k:dict(v) if k == 'coordinates' else v for k,v in vars(self).items() if not k.startswith('_')}
    def assert_integrity(self):
        c = self.coordinates
        _require(all(v is False for k,v in vars(self).items() if k.endswith('_authority')), 'spectral receipt authority changed')
        _require(c['schema'] == 'native_factored_spectral_phase_A_result_v1'
                 and tuple(c['branch_order']) == ('spectral',) and c['diagonal_branch_admitted'] is False
                 and c['native_cuda_qualification_verified'] is False
                 and all(v is False for k,v in c.items() if k.endswith('_authority')),
                 'native spectral result scope changed')
        _require(content_hash(self.to_json_payload(), prefix=_PREFIX+'result-') == self._hash, 'spectral result receipt changed')
        return self._hash


class V1FactoredSpectralResourceFrontierExecution:
    """Retains the one-branch state under its problem's exclusive leases."""
    def __init__(self, problem, outcome, tensors, receipt, *, _issuance_token):
        _require(_issuance_token is _TOKEN, 'native spectral results are owner-issued')
        self._problem, self.receipt = problem, receipt
        self._issued_problem = problem
        self._receipt_hash = receipt.receipt_hash
        self._outcome, self._tensors = outcome, tensors
        self._issued_outcome = outcome
        state = outcome.theorem_state
        self._state_tensors = () if state is None else tuple(t for t in
            (state.primal, state.low_q_dual, state.tail_q_dual, state.block_dual) if t is not None)
        self._state_custody = tuple(core._frontier_state_tensor_storage_invariant(t) for t in self._state_tensors)
        self._closed = False
        problem._active_result = True

    @property
    def problem(self): return self._problem

    def assert_live(self):
        _require(not self._closed, 'native spectral result closed')
        _require(self._problem is self._issued_problem, 'native spectral problem owner replaced')
        self.problem.assert_integrity()
        _require(self._outcome is self._issued_outcome, 'native spectral branch outcome replaced')
        _require(self.receipt.assert_integrity() == self._receipt_hash, 'native spectral receipt replaced')
        live_state = self._outcome.theorem_state
        live_tensors = () if live_state is None else tuple(t for t in
            (live_state.primal, live_state.low_q_dual, live_state.tail_q_dual, live_state.block_dual) if t is not None)
        _require(tuple(core._frontier_state_tensor_storage_invariant(t) for t in live_tensors) == self._state_custody,
                 'native spectral theorem state changed')
        return self._receipt_hash

    @property
    def primal(self):
        self.assert_live()
        _require(self.receipt.coordinates['ideal_q_primal_feasibility_verified'], 'spectral candidate not outward qualified')
        return self._outcome.theorem_state.primal

    def close(self):
        if not self._closed:
            self._closed = True
            self._state_tensors = ()
            self._outcome = self._issued_outcome = self._tensors = None
            self._issued_problem._active_result = False

    def __enter__(self):
        self.assert_live()
        return self
    def __exit__(self, *_exc): self.close()
    def __reduce_ex__(self, _protocol):
        raise TypeError('native spectral theorem states are nonserializable')


def forecast_v1_factored_spectral_resource_frontier(problem, policy):
    """Reuse conservative branch workspace counts, add outward-Q replay space."""
    problem.assert_integrity()
    n, r, b = problem.route_count, problem.rank, len(problem.block_capacities)
    def bound_bytes(t):
        return 8 if t.untyped_storage().nbytes() == t.element_size() and all(v == 0 for v in t.stride()) else 8*n
    state = (17*n + 2*r + 2 + 2*b)*8
    transient = (10*n + 4*r + 4*max(1,b))*8
    # Q replay runs while the frontier state is retained. The Q owner already
    # admits its tile workspace; charge that plus route-square/error temporaries.
    replay = problem._q_handle.receipt.coordinates['tile_workspace_bytes'] + 4*n*8 + 16*r*8
    compact, scalar = 2*b*8, max(1,b+8)*8
    derived = bound_bytes(problem._eligible) + bound_bytes(problem._required)
    incremental = state + transient + replay + compact + scalar + derived
    stores = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes() for t in problem._tensors()}
    if problem.device.type == 'cuda':
        current = torch.cuda.memory_reserved(problem.device)
        allocated = torch.cuda.memory_allocated(problem.device)
        free, total = torch.cuda.mem_get_info(problem.device)
        available = cuda_reusable_memory_bytes(problem.device, framework=torch, free_bytes=free)
    else:
        current = 0
        total = max(1 << 30, 4*incremental)
        available = total
    ceiling = math.floor(policy.hbm_ceiling_fraction*total)
    return dict(schema='native_factored_spectral_phase_A_hbm_forecast_v1',
        problem_hash=problem.problem_hash, policy_hash=policy.policy_hash,
        already_resident_aliased_source_bytes=sum(stores.values()), solver_state_bytes=state,
        transient_workspace_bytes=transient, outward_q_replay_workspace_bytes=replay,
        compact_block_coordinate_bytes=compact, scalar_workspace_bytes=scalar, derived_bound_bytes=derived,
        sqrt_residual_bytes=0, forecast_incremental_bytes=incremental, current_reserved_bytes=current,
        forecast_peak_reserved_bytes=current+incremental, device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling, admitted=(allocated if problem.device.type == 'cuda' else current)+incremental <= ceiling and incremental <= available,
        native_cuda_qualification_verified=False, private_cpu_forecast=problem.device.type == 'cpu')


def solve_v1_factored_spectral_resource_frontier(*, problem, policy, branch='spectral'):
    _require(type(problem) is V1FactoredSignedWorkResourceFrontierCUDADeviceProblemV1,
             'exact native spectral problem required')
    _require(type(policy) is core.V1SignedWorkResourceFrontierCUDAPolicy, 'exact frontier policy required')
    _require(branch == 'spectral', 'native rounded Q admits only the spectral branch')
    _require(not problem._active_result, 'native spectral result already active')
    problem.assert_integrity()
    policy.__post_init__()
    private = problem.device.type == 'cpu'
    _require(not private or (policy.allow_private_cpu_reference and not policy.production_cuda_required
             and policy.use_tensor_private_reference and problem.route_count <= policy.private_reference_route_limit),
             'private CPU spectral policy not admitted')
    requested_device = torch.device(policy.torch_device)
    _require(requested_device.type == problem.device.type and
             (requested_device.index is None or requested_device.index == problem.device.index),
             'policy/source device mismatch')
    _require(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled(),
             'native spectral solve requires deterministic algorithms')
    forecast = forecast_v1_factored_spectral_resource_frontier(problem, policy)
    _require(forecast['admitted'], 'native spectral workspace exceeds memory admission')
    if not private:
        torch.cuda.synchronize(problem.device)
    started = time.perf_counter()
    tensors = core._frontier_torch_tensors(problem, device=problem.device)
    outcome = core._solve_frontier_tensor_branch(resource_problem=problem, problem=problem,
        policy=policy, branch=_BRANCH, tensors=tensors)
    result = outcome.receipt
    replay = dict(stored_bhat_q=None, ideal_q_lower=None, achieved_q_upper=None)
    certified = False
    nonempty_threshold = (problem.baseline_resource + policy.absolute_gap_tolerance
        + policy.relative_gap_tolerance * max(1.0, abs(problem.baseline_resource)))
    status = ('UNRESOLVED_CONSERVATIVE_EMPTY' if result.status is core.V1SignedWorkResourceBranchStatus.CERTIFIED_INFEASIBLE
              else 'UNRESOLVED_CONSERVATIVE_SOLVE')
    if outcome.theorem_state is not None:
        z = outcome.theorem_state.primal
        # No repair after solver receipt issuance: changed coordinates would
        # need a fresh resource/dual receipt. The core already repaired them.
        _require(bool((torch.isfinite(z) & (z >= tensors.lower) & (z <= tensors.upper)).all().item()),
                 'native spectral witness violates its exact box')
        core._frontier_repair_and_certify_physical_blocks(problem, tensors, z, allow_inward_repair=False)
        value, lower, upper = problem._q_handle.quadratic_enclosure(z.reshape(problem.route_count, 1).contiguous())
        certified = bool((upper <= problem.quadratic_budget).all().item())
        replay = dict(stored_bhat_q=float(value.item()), ideal_q_lower=float(lower.item()), achieved_q_upper=float(upper.item()))
        status = ('OPTIMAL_CONSERVATIVE_SPECTRAL_COORDINATE' if certified and result.achieved_resource > nonempty_threshold
                  else 'UNRESOLVED_CONSERVATIVE_EMPTY' if certified else 'UNRESOLVED_Q_REPLAY')
    problem.assert_integrity()
    if not private:
        torch.cuda.synchronize(problem.device)
    coordinates = dict(schema='native_factored_spectral_phase_A_result_v1', status=status,
        memory_forecast=forecast, native_cuda_qualification_verified=False,
        source_join_receipt_hash=problem._source_join_hash, rounded_q_receipt_hash=problem._q_handle.receipt_hash,
        source_registration_hash=problem.source_registration_hash, registered_d_coordinate_hash=problem.registered_d_coordinate_hash,
        source_geometry_hash=problem.source_geometry_hash, structural_source_receipt_hash=problem.structural_source_receipt_hash,
        physical_route_axis_hash=problem.physical_route_axis_hash, route_id_axis_hash=problem.route_id_axis_hash,
        structural_capacity_receipt_hash=problem._source_join.receipt.coordinates['structural_capacity_receipt_hash'],
        problem_hash=problem.problem_hash, policy_hash=policy.policy_hash,
        conservative_spectral_branch_receipt_hash=result.branch_receipt_hash,
        conservative_solver_status=result.status.value, branch_order=('spectral',), diagonal_branch_admitted=False,
        optimization_target='stored_Bhat_Gram_plus_rho_I', resource_bound_scope='conservative_spectral_surrogate_only',
        resource_objective_scale_law=_NATIVE_RESOURCE_OBJECTIVE_SCALE_LAW,
        resource_objective_beta=1.0, resource_dual_to_physical_scale=1.0,
        ideal_frontier_exclusion_authority=False, ideal_q_primal_feasibility_verified=certified,
        conservative_spectral_coordinate_available=status == 'OPTIMAL_CONSERVATIVE_SPECTRAL_COORDINATE',
        achieved_resource_lower=result.achieved_resource if certified else None,
        conservative_resource_upper=result.resource_upper_bound,
        q_budget=problem.quadratic_budget, nonempty_resource_threshold=nonempty_threshold, elapsed_seconds=time.perf_counter()-started,
        device=str(problem.device), private_cpu_reference_execution=private, signed_f_operator_retained=False,
        route_scale_d2h_bytes=0, private_cpu_route_numpy_view=private and outcome.theorem_state is not None,
        action_authority=False, d_decision_authority=False, kkt_authority=False,
        c_measurement_authority=False, calibration_k_authority=False, paired_oracle_o_authority=False,
        mask_commit_authority=False, terminal_authority=False, production_authority=False,
        performance_authority=False, pareto_authority=False, **replay)
    receipt = V1FactoredSpectralResourceFrontierReceipt(coordinates, _issuance_token=_TOKEN)
    return V1FactoredSpectralResourceFrontierExecution(problem, outcome, tensors, receipt, _issuance_token=_TOKEN)
