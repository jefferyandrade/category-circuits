"""Actual operator acquisition for the retained modern-v1 runtime seam.

Owns D/C execution and process-local handoff only. The retained controller,
native calibration and paired oracle retain all decision authority.
"""
from fractions import Fraction
from threading import Lock
from pathlib import Path
import hashlib

from mechanistic_systems.contracts.content_address import content_hash
from .modern_v1_operator_acquisition import (
    issue_v1_operator_acquisition, issue_v1_operator_no_candidate_acquisition,
    validate_operator_metric_budget_lineage,
)
from .modern_v1_operator_lifecycle import synchronize_operator_committed_mask
from .modern_v1_operator_model import OperatorNativeModelEvaluator
from .tractable_v1.discovery import OperatorDiscovery, discovery_config_hash, _packed_parent
from .tractable_v1.physical_replay import OperatorPhysicalReplay
from .tractable_v1.metric_freeze import RawLogitMetricBudgetBinding


class OperatorAcquisitionFactory:
    """One source-bound factory, one consumed chart per controller generation.

    A completed unchanged call can be retrieved once again without images.
    Failures after chart entry poison the factory rather than retrying revealed
    data. Source reconstruction and crash recovery are separate obligations.
    """
    def __init__(self, *, native_evaluator, metric_budget, counts, capture_batch,
                 max_factor_bytes, policy, factor_dtype, seed, registration_hash=None):
        from .modern_v1_alexnet_backend import _DirectFixedCalibrationBCEEvaluator
        if type(native_evaluator) is not _DirectFixedCalibrationBCEEvaluator:
            raise TypeError('operator factory requires the retained exact native evaluator')
        if type(native_evaluator.evaluator) is not OperatorNativeModelEvaluator:
            raise TypeError('operator factory requires the shared actual functional model')
        if type(metric_budget) is not RawLogitMetricBudgetBinding:
            raise TypeError('operator factory requires a frozen raw metric budget')
        if type(counts) is not tuple or len(counts)!=2 or any(type(n) is not int or n<=0 for n in counts):
            raise ValueError('operator factory requires two positive discovery counts')
        self.native_evaluator=native_evaluator
        if registration_hash is not None and (type(registration_hash) is not str or not registration_hash.strip()):
            raise ValueError('operator factory registration identity must be nonempty')
        self.registration_hash=registration_hash
        self.metric_budget=metric_budget
        self.counts=counts
        self.settings=dict(capture_batch=capture_batch,max_factor_bytes=max_factor_bytes,
                           policy=policy,factor_dtype=factor_dtype,seed=seed)
        self._model=native_evaluator.evaluator
        self._model.sources.context_source.image_source.freeze_content_manifest()
        self._signature=self._coordinates()
        self._lock=Lock()
        self._poisoned=False
        self._last_controller_hash=None
        self._last_generation=None
        self._last_acquisition=None
        self._restored_between_generation_hash=None
        self.last_discovery=None
        self.last_physical=None
        if getattr(native_evaluator,'_operator_acquisition_factory',None) is not None:
            raise ValueError('native operator source already has an acquisition factory')
        native_evaluator._operator_acquisition_factory=self

    def _coordinates(self):
        model=self._model
        native=self.native_evaluator
        if (native.evaluator is not model or native._operator_native_model is not model
                or native.calibration_installer.evaluator is not model
                or native.data_plane is not model.data_plane
                or native.calibration_installer.data_plane is not model.data_plane
                or native.mask_registry is not native.calibration_installer.mask_registry):
            raise ValueError('operator factory shared execution owner changed')
        model.assert_live()
        config=discovery_config_hash(**self.settings)
        budget=self.metric_budget
        budget.__post_init__()
        owner=model.sources.context_source.source_owner
        if (budget.config_hash!=config or budget.population_identity_hash!=owner.source_population_hash
                or budget.route_axis_hash!=model.sources.route_axis.route_axis_hash
                or budget.target_class!=model.data_plane.target_class_id
                or self.settings['capture_batch']>model.sources.context_source.max_batch_rows
                or self.counts[0]>len(owner.target_context_ids)
                or self.counts[1]>len(owner.complement_context_ids)):
            raise ValueError('operator factory source, budget or discovery limits differ')
        payload=dict(schema='v1_actual_operator_acquisition_factory_v1',
            native_model=model.binding_hash,metric_budget=budget.budget_hash,counts=self.counts,
            image_content_manifest_hash=model.sources.context_source.image_source.content_manifest_hash,
            factory_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            config_hash=config,C_policy='balanced_prefix_until_both_certified_or_refuted_else_census',
            campaign_binding_hash=native.campaign_binding_hash)
        if self.registration_hash is not None:
            payload['operator_factory_registration_hash']=self.registration_hash
        return content_hash(payload,prefix='v1-operator-factory-')

    def __call__(self, controller):
        if not self._lock.acquire(blocking=False):
            raise ValueError('operator factory is already executing a chart')
        try:
            return self._acquire(controller)
        finally:
            self._lock.release()

    def adopt_restored_acquisition(self, acquisition):
        """Bind a verified closed acquisition before any resumed K/O outcomes.

        Restoration of an already active calibration/oracle epoch remains the
        retained backend's responsibility, not this acquisition-only adapter.
        """
        from .modern_v1_operator_acquisition import V1OperatorAcquisition
        if type(acquisition) is not V1OperatorAcquisition:
            raise TypeError('operator restore requires an exact reconstructed acquisition')
        if not self._lock.acquire(blocking=False):
            raise ValueError('operator factory is already executing a chart')
        try:
            if (self._poisoned or self._last_acquisition is not None
                    or self.native_evaluator._active_paired_bce_generation_frame is not None):
                raise ValueError('operator acquisition restore requires an unused native factory')
            if (self._coordinates()!=self._signature
                    or getattr(self.native_evaluator,'_operator_acquisition_factory',None) is not self):
                raise ValueError('operator factory configuration changed')
            acquisition.assert_integrity()
            physical=(acquisition._no_candidate_physical if acquisition.candidate is None
                      else acquisition.candidate._producer)
            proposal=(acquisition._no_candidate_proposal if physical is None else physical.proposal)
            d=proposal.discovery
            if d.target_resource is None:
                raise ValueError('restored canonical operator acquisition lacks its resource target')
            if (d.sources is not self._model.sources or d.metric_budget.budget_hash!=self.metric_budget.budget_hash
                    or d.counts!=self.counts or acquisition.binding.binding_hash!=self.native_evaluator.campaign_binding_hash):
                raise ValueError('restored operator acquisition differs from native factory source/configuration')
            if acquisition.candidate is not None:
                child=acquisition.candidate.resulting_mask
                try:
                    self.native_evaluator.mask_registry[child.mask_hash]=child
                    self.native_evaluator.activate_causal_certification_source(frame=d.frame,discovery_stream=d.stream,
                        discovery_freeze=proposal.discovery_freeze,certification_stream=physical.stream,
                        certification_cumulative_stratum_counts=physical.population_progress.counts,
                        certification_evidence_state_hash=physical.population_progress.receipt_hash)
                except Exception:
                    self._poisoned=True
                    raise
            self.last_discovery=d;self.last_physical=physical
            self._last_controller_hash=acquisition.receipt.source_controller_state_hash
            self._last_generation=acquisition.generation_index
            self._last_acquisition=acquisition
        finally:
            self._lock.release()

    def adopt_restored_runtime(self,state):
        """Hydrate a verified runtime on unused, already matching sources.

        Between generations the checkpoint's committed mask is already installed
        on the fresh model. No second parent-to-child commit is performed.
        """
        from .modern_v1_runtime import V1RuntimeState
        from .modern_v1_operator_run_binding import validate_bound_operator_sources
        if type(state) is not V1RuntimeState:
            raise TypeError('operator runtime adoption requires exact retained state')
        state.__post_init__()
        validate_bound_operator_sources(state.config,self._model.sources,factory=self,
            current_mask=state.controller.current_mask)
        if (self._poisoned or self._last_acquisition is not None or self._last_generation is not None
                or self._restored_between_generation_hash is not None
                or self._model.sources.context_source.image_source.stats()['decode_requests_submitted']):
            raise ValueError('operator runtime adoption requires an unused fresh factory')
        if state.controller.acquisition is not None:
            self.adopt_restored_acquisition(state.controller.acquisition)
            try:
                self.native_evaluator.restore_controller_state(state.controller)
            except Exception:
                self._poisoned=True
                raise
            return
        if not self._lock.acquire(blocking=False):
            raise ValueError('operator factory is already executing a chart')
        entered=False
        try:
            if (self._poisoned or self._last_acquisition is not None or self._last_generation is not None
                    or self._restored_between_generation_hash is not None
                    or self._model.sources.context_source.image_source.stats()['decode_requests_submitted']):
                raise ValueError('operator runtime adoption requires an unused fresh factory')
            if self.native_evaluator._active_paired_bce_generation_frame is not None:
                raise ValueError('between-generation operator restore has unrelated causal state')
            if state.controller.accepted_step_count:
                entered=True
                self.native_evaluator.restore_controller_state(state.controller)
            self._restored_between_generation_hash=state.controller.state_hash
            generation=state.controller.acquisition_generation_index
            self._last_generation=generation-1 if generation else None
        except Exception:
            if entered:self._poisoned=True
            raise
        finally:
            self._lock.release()

    def _acquire(self, controller):
        if self._poisoned:
            raise ValueError('operator factory poisoned after entered chart execution failed')
        if (self._coordinates()!=self._signature
                or getattr(self.native_evaluator,'_operator_acquisition_factory',None) is not self):
            raise ValueError('operator factory configuration or owner changed')
        validate_operator_metric_budget_lineage(controller=controller,metric_budget=self.metric_budget)
        if (controller.phase!='needs_acquisition'
                or controller.binding.binding_hash!=self.native_evaluator.campaign_binding_hash):
            raise ValueError('operator factory needs its bound controller awaiting acquisition')
        state_hash=controller.state_hash
        if state_hash==self._last_controller_hash:
            self._last_acquisition.assert_integrity()
            return self._last_acquisition
        if self._last_generation is not None and controller.acquisition_generation_index!=self._last_generation+1:
            raise ValueError('operator factory cannot reopen an alternate or skipped generation')
        if (controller.attempt_receipts and state_hash!=self._restored_between_generation_hash):
            synchronize_operator_committed_mask(controller=controller,native_evaluator=self.native_evaluator)
        elif self.native_evaluator._active_paired_bce_generation_frame is not None:
            raise ValueError('initial operator chart has unrelated native causal state')
        if _packed_parent(self._model.sources)!=controller.current_mask:
            raise ValueError('operator factory functional parent differs from the controller')
        remaining_resource=controller.binding.contract.maximum_removed_count-controller.cumulative_removed_count
        if remaining_resource<=0:
            raise ValueError('operator resource target already reached; controller must stop before discovery')
        try:
            self._restored_between_generation_hash=None
            discovery=OperatorDiscovery(sources=self._model.sources,metric_budget=self.metric_budget,
                generation=controller.acquisition_generation_index,counts=self.counts,
                target_resource=remaining_resource,**self.settings)
            self.last_discovery=discovery
            self.last_physical=None
            proposal=discovery.propose_and_freeze()
            if proposal.packed_action is None:
                acquisition=issue_v1_operator_no_candidate_acquisition(controller=controller,proposal=proposal)
            else:
                physical=OperatorPhysicalReplay(proposal)
                self.last_physical=physical
                while True:
                    p=physical.population_progress
                    statuses=(p.conditional_direct_D_status,p.conditional_cumulative_dense_D_status)
                    if ('refuted' in statuses or 'unregistered' in statuses
                            or (statuses==('certified','certified') and min(p.counts)>0)
                            or p.counts==p.population_sizes):
                        break
                    # Choice depends only on exposed counts, never unseen outcomes.
                    eligible=[s for s in range(2) if p.counts[s]<p.population_sizes[s]]
                    s=min(eligible,key=lambda i:Fraction(p.counts[i],p.population_sizes[i]))
                    request=physical.next_request(('target','complement')[s])
                    physical.execute(request)
                if statuses==('certified','certified'):
                    acquisition=issue_v1_operator_acquisition(controller=controller,producer=physical)
                    child=acquisition.candidate.resulting_mask
                    self.native_evaluator.mask_registry[child.mask_hash]=child
                    self.native_evaluator.activate_causal_certification_source(
                        frame=discovery.frame,discovery_stream=discovery.stream,
                        discovery_freeze=proposal.discovery_freeze,certification_stream=physical.stream,
                        certification_cumulative_stratum_counts=p.counts,
                        certification_evidence_state_hash=p.receipt_hash)
                else:
                    acquisition=issue_v1_operator_no_candidate_acquisition(
                        controller=controller,proposal=proposal,physical=physical)
            acquisition.assert_integrity()
        except BaseException:
            self._poisoned=True
            raise
        self._last_controller_hash=state_hash
        self._last_generation=controller.acquisition_generation_index
        self._last_acquisition=acquisition
        return acquisition
