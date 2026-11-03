"""Actual operator evidence carried into the retained modern-v1 controller.

This process-local acquisition preserves the source-owned C result. It does
not manufacture old solver/F/J/R witnesses, calibrate, query O, or commit.
Portable replay requires a separate source reconstruction protocol.
"""
from dataclasses import dataclass, field
from types import MappingProxyType

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from .modern_v1_campaign_contract import V1OptionACampaignBinding

_TOKEN=object()


@dataclass(frozen=True, slots=True)
class V1OperatorCandidate:
    _producer: object = field(repr=False,compare=False)
    _progress_hash: str
    _issuer_token: object = field(repr=False,compare=False)

    def __post_init__(self):
        self.assert_integrity()

    def assert_integrity(self):
        from .tractable_v1.physical_replay import OperatorPhysicalReplay
        if self._issuer_token is not _TOKEN or type(self._producer) is not OperatorPhysicalReplay:
            raise TypeError('operator candidate requires actual source-owned physical C issuance')
        self._producer.assert_live()
        progress=self._producer.population_progress
        if (progress.receipt_hash!=self._progress_hash
                or progress.conditional_direct_D_status!='certified'
                or progress.conditional_cumulative_dense_D_status!='certified'):
            raise ValueError('operator candidate lacks its frozen certified conditional direct-D bound')
        self._producer.proposal.assert_live()
        return self.candidate_id

    @property
    def candidate_id(self):
        return content_hash(dict(proposal_hash=self._producer.proposal.proposal_hash,
                                 physical_progress_hash=self._progress_hash),prefix='v1-operator-candidate-')

    @property
    def resulting_mask(self):
        return self._producer.proposal.packed_action.resulting_mask

    @property
    def parent_mask_hash(self):
        return self._producer.proposal.discovery.parent_mask.mask_hash

    @property
    def removed_count(self):
        return self._producer.proposal.packed_action.removed_count

    @property
    def removed_route_index_hash(self):
        return self._producer.proposal.packed_action.removed_route_index_hash

    compact_removed_indices_available=False

    @property
    def resource_gain(self):
        return float(self.removed_count)

    def definition_payload(self):
        self.assert_integrity()
        p=self._producer.proposal
        progress=self._producer.population_progress
        return dict(schema='v1_operator_native_candidate_definition_v1',candidate_id=self.candidate_id,
            proposal_hash=p.proposal_hash,physical_progress_hash=self._progress_hash,
            source_opening_hash=p.discovery.sources.opening_hash,
            parent_mask_hash=self.parent_mask_hash,candidate_mask_hash=self.resulting_mask.mask_hash,
            packed_action_hash=p.packed_action.action_hash,removed_route_count=self.removed_count,
            removed_route_index_hash=self.removed_route_index_hash,
            metric_budget_hash=p.discovery.metric_budget.budget_hash,
            physical_evidence_scope=progress.to_json()['interval_scope'],
            conditional_direct_D_status=progress.conditional_direct_D_status,
            parent_relative_full_dose_rms_interval=(progress.rms_lower[1],progress.rms_upper[1]),
            parent_relative_raw_rms_radius=progress.raw_rms_radius,
            cumulative_dense_relative_raw_rms_interval=progress.dense_relative_rms_interval,
            cumulative_dense_relative_raw_rms_radius=progress.dense_relative_rms_radius,
            conditional_cumulative_dense_D_status=progress.conditional_cumulative_dense_D_status,
            cumulative_gate_population_role='extraction_physical_C_not_terminal_validation',
            physical_population_measure=p.discovery.metric_budget.balanced_population_measure,
            old_solver_ancestry=False,F_J_R_population_evidence_available=False,
            paired_bce_oracle_required=True,mask_commit_authority=False)

    def __reduce_ex__(self, protocol):
        raise TypeError('operator acquisition needs explicit source reconstruction for portable restart')


@dataclass(frozen=True, slots=True)
class V1OperatorAcquisitionReceipt:
    campaign_binding_hash: str
    controller_run_id: str
    source_controller_state_hash: str
    generation_index: int
    current_mask_hash: str
    physical_progress_hash: str | None
    ordered_candidate_ids: tuple[str,...]
    committed_mask_lineage_hashes: tuple[str,...]
    committed_mask_transition_receipt_hashes: tuple[str,...]
    proposal_hash: str
    outcome: str

    @property
    def receipt_hash(self):
        from dataclasses import asdict
        return content_hash(asdict(self),prefix='v1-operator-controller-acquisition-')

    @property
    def acquisition_hash(self):
        return self.receipt_hash


@dataclass(frozen=True, slots=True)
class V1OperatorAcquisition:
    binding: V1OptionACampaignBinding
    current_mask: HardMask
    candidate: V1OperatorCandidate | None
    receipt: V1OperatorAcquisitionReceipt
    _issuer_token: object = field(repr=False,compare=False)
    _receipt_hash: str
    _no_candidate_proposal: object | None = field(default=None,kw_only=True,repr=False,compare=False)
    _no_candidate_physical: object | None = field(default=None,kw_only=True,repr=False,compare=False)

    def assert_integrity(self):
        if (self._issuer_token is not _TOKEN or type(self.binding) is not V1OptionACampaignBinding
                or type(self.current_mask) is not HardMask
                or type(self.receipt) is not V1OperatorAcquisitionReceipt):
            raise TypeError('operator acquisition must be issued from exact source/controller owners')
        if self.candidate is not None:
            if type(self.candidate) is not V1OperatorCandidate or self._no_candidate_proposal is not None or self._no_candidate_physical is not None:
                raise TypeError('operator candidate acquisition has alternate outcome owners')
            self.candidate.assert_integrity()
            proposal=self.candidate._producer.proposal
            expected_ids=(self.candidate.candidate_id,)
            progress_hash=self.candidate._progress_hash
            outcome='candidate_ready'
            custody=self.candidate._producer._admission_owner
        else:
            proposal=self._no_candidate_proposal
            outcome=_no_candidate_outcome(proposal,self._no_candidate_physical)
            expected_ids=()
            progress_hash=(None if self._no_candidate_physical is None else self._no_candidate_physical.population_progress.receipt_hash)
            custody=(getattr(proposal.discovery,'_operator_no_candidate_acquisition',None)
                     if self._no_candidate_physical is None else self._no_candidate_physical._admission_owner)
        source=proposal.discovery.sources
        if (self.receipt.receipt_hash!=self._receipt_hash
                or self.receipt.campaign_binding_hash!=self.binding.binding_hash
                or self.receipt.current_mask_hash!=self.current_mask.mask_hash
                or self.current_mask.mask_hash!=proposal.discovery.parent_mask.mask_hash
                or self.receipt.ordered_candidate_ids!=expected_ids
                or self.receipt.physical_progress_hash!=progress_hash
                or self.receipt.proposal_hash!=proposal.proposal_hash or self.receipt.outcome!=outcome
                or self.receipt.generation_index!=proposal.discovery.frame.generation_index
                or source.fq_registration.campaign_contract_hash!=self.binding.contract.campaign_contract_hash
                or source.context_source.source_owner.source_population_hash!=self.binding.contract.context_population_hash
                or self.current_mask.route_axis_hash!=self.binding.contract.route_axis_hash
                or custody is not self):
            raise ValueError('operator acquisition source/campaign/admission custody changed')
        return self.acquisition_hash

    @property
    def acquisition_hash(self):
        return self.receipt.receipt_hash

    @property
    def generation_index(self):
        return self.receipt.generation_index

    @property
    def queue(self):
        self.assert_integrity()
        return () if self.candidate is None else (self.candidate,)

    @property
    def committed_mask_lineage_hashes(self):
        return self.receipt.committed_mask_lineage_hashes

    @property
    def committed_mask_transition_receipt_hashes(self):
        return self.receipt.committed_mask_transition_receipt_hashes

    @property
    def invalidation_chart_hash(self):
        proposal=self._no_candidate_proposal if self.candidate is None else self.candidate._producer.proposal
        return proposal.discovery.capture_hash

    @property
    def metadata(self):
        return MappingProxyType(dict(operator_physical_progress_hash=self.receipt.physical_progress_hash,
                                     operator_outcome=self.receipt.outcome,
                                     paired_bce_oracle_required=self.candidate is not None,old_solver_ancestry=False))

    def __reduce_ex__(self, protocol):
        raise TypeError('operator acquisition needs explicit source reconstruction for portable restart')


def validate_operator_metric_budget_lineage(*, controller, metric_budget):
    """Keep every attempted operator action in one immutable physical budget.

    Call before a subsequent chart capture; issuance also enforces this check.
    Histories without the operator's explicit raw-function coordinates cannot
    silently acquire the new gate's authority.
    """
    from .modern_v1_controller import V1ControllerState
    from .tractable_v1.metric_freeze import RawLogitMetricBudgetBinding
    if type(controller) is not V1ControllerState or type(metric_budget) is not RawLogitMetricBudgetBinding:
        raise TypeError('operator budget lineage requires exact controller and metric binding')
    controller.__post_init__()
    metric_budget.__post_init__()
    for attempt in controller.attempt_receipts:
        definition=attempt.metadata.get('operator_candidate_definition', {})
        if (definition.get('metric_budget_hash')!=metric_budget.budget_hash
                or definition.get('conditional_cumulative_dense_D_status')!='certified'):
            raise ValueError('operator trajectory changed its physical budget or lacks prior dense-function gate evidence')


def _no_candidate_outcome(proposal, physical):
    from .tractable_v1.discovery import FrozenOperatorProposal
    from .tractable_v1.physical_replay import OperatorPhysicalReplay
    if type(proposal) is not FrozenOperatorProposal:
        raise TypeError('empty operator queue requires its actual frozen proposal')
    proposal.assert_live()
    if proposal.packed_action is None:
        if physical is not None:
            raise ValueError('empty binary action cannot carry physical C')
        return 'no_binary_action'
    if type(physical) is not OperatorPhysicalReplay or physical.proposal is not proposal:
        raise TypeError('physical refusal requires the exact actual C owner')
    physical.assert_live()
    if physical._pending is not None or physical._running:
        raise ValueError('physical refusal requires an idle C prefix')
    p=physical.population_progress
    if 'refuted' in (p.conditional_direct_D_status,p.conditional_cumulative_dense_D_status):
        return 'physical_refuted'
    if p.conditional_cumulative_dense_D_status=='unregistered':
        return 'dense_radius_unregistered'
    if (p.counts==p.population_sizes
            and 'unresolved' in (p.conditional_direct_D_status,p.conditional_cumulative_dense_D_status)):
        return 'physical_unresolved_at_census'
    raise ValueError('physical prefix has no terminal no-candidate outcome')


def validate_operator_resource_target(*, controller, discovery, required=False):
    """Check a pre-D target against this controller's initial-relative resource.

    Standalone discovery diagnostics may omit a target. Canonical factories
    supply one before capture; it is neither a per-step loss rate nor a new
    scientific distortion budget.
    """
    target=discovery.target_resource
    if target is None:
        if required:
            raise ValueError('canonical operator acquisition requires its campaign resource target')
        return
    remaining=controller.binding.contract.maximum_removed_count-controller.cumulative_removed_count
    if type(target) is not int or target<=0 or target!=remaining:
        raise ValueError('operator discovery resource target differs from remaining campaign resource')


def issue_v1_operator_no_candidate_acquisition(*, controller, proposal, physical=None):
    """Issue an empty queue from an actual empty action or terminal C refusal.

    This exhausts this bounded proposal attempt, not every possible binary action.
    It spends no oracle alpha and cannot authorize calibration or a mask change.
    """
    outcome=_no_candidate_outcome(proposal,physical)
    validate_operator_metric_budget_lineage(controller=controller,metric_budget=proposal.discovery.metric_budget)
    d=proposal.discovery
    validate_operator_resource_target(controller=controller,discovery=d)
    if (controller.phase!='needs_acquisition'
            or d.frame.generation_index!=controller.acquisition_generation_index
            or d.parent_mask!=controller.current_mask
            or d.sources.fq_registration.campaign_contract_hash!=controller.binding.contract.campaign_contract_hash
            or d.sources.context_source.source_owner.source_population_hash!=controller.binding.contract.context_population_hash):
        raise ValueError('no-candidate proposal differs from native campaign/current generation')
    if ((physical is not None and physical._admission_owner is not None)
            or getattr(d,'_operator_no_candidate_acquisition',None) is not None):
        raise ValueError('no-candidate outcome already has an admission owner')
    progress_hash=None if physical is None else physical.population_progress.receipt_hash
    receipt=V1OperatorAcquisitionReceipt(controller.binding.binding_hash,controller.controller_run_id,
        controller.state_hash,controller.acquisition_generation_index,d.parent_mask.mask_hash,progress_hash,
        (),controller.mask_lineage_hashes,controller.transition_receipt_hashes,proposal.proposal_hash,outcome)
    acquisition=V1OperatorAcquisition(controller.binding,d.parent_mask,None,receipt,_TOKEN,receipt.receipt_hash,
        _no_candidate_proposal=proposal,_no_candidate_physical=physical)
    d.release_factors(proposal)
    if physical is None:
        d._operator_no_candidate_acquisition=acquisition
    else:
        physical._admission_owner=acquisition
    acquisition.assert_integrity()
    return acquisition


def issue_v1_operator_acquisition(*, controller, producer):
    """Seal certified C evidence into one native queue entry, without opening K/O."""
    from .modern_v1_controller import V1ControllerState
    from .tractable_v1.physical_replay import OperatorPhysicalReplay
    if type(controller) is not V1ControllerState or type(producer) is not OperatorPhysicalReplay:
        raise TypeError('operator admission requires exact controller and actual C producer')
    validate_operator_metric_budget_lineage(controller=controller,
        metric_budget=producer.proposal.discovery.metric_budget)
    producer.assert_live()
    if controller.phase!='needs_acquisition' or producer._pending is not None or producer._running:
        raise ValueError('operator admission requires idle C and a controller awaiting acquisition')
    if producer._admission_owner is not None:
        raise ValueError('operator physical C already has an admission owner')
    proposal=producer.proposal
    validate_operator_resource_target(controller=controller,discovery=proposal.discovery)
    source=proposal.discovery.sources
    parent=proposal.discovery.parent_mask
    if (source.fq_registration.campaign_contract_hash!=controller.binding.contract.campaign_contract_hash
            or source.context_source.source_owner.source_population_hash!=controller.binding.contract.context_population_hash
            or parent.mask_hash!=controller.current_mask.mask_hash
            or parent.route_axis_hash!=controller.binding.contract.route_axis_hash
            or proposal.discovery.frame.generation_index!=controller.acquisition_generation_index):
        raise ValueError('operator proposal differs from native campaign/current generation')
    progress=producer.population_progress
    if progress.conditional_direct_D_status!='certified':
        raise ValueError('unresolved or refuted physical C cannot enter the native candidate queue')
    if progress.conditional_cumulative_dense_D_status!='certified':
        raise ValueError('unregistered, unresolved or refuted cumulative dense-relative C cannot enter the native candidate queue')
    candidate=V1OperatorCandidate(producer,progress.receipt_hash,_TOKEN)
    receipt=V1OperatorAcquisitionReceipt(controller.binding.binding_hash,
        controller.controller_run_id,controller.state_hash,
        controller.acquisition_generation_index,parent.mask_hash,progress.receipt_hash,
        (candidate.candidate_id,),controller.mask_lineage_hashes,controller.transition_receipt_hashes,
        proposal.proposal_hash,'candidate_ready')
    acquisition=V1OperatorAcquisition(controller.binding,parent,candidate,receipt,_TOKEN,receipt.receipt_hash)
    producer._admission_owner=acquisition
    acquisition.assert_integrity()
    return acquisition
