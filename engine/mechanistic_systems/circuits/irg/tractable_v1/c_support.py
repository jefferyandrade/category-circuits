"""Actual pre-outcome raw-logit/direct-D support for the operator C source.

The registered preprocessing box and retained numerical error model provide
conditional uniform supports. No old shell/true-Q/D-stability ancestry is
asserted. F/J/R supports and C statistical admission are separate obligations.
"""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import torch

from mechanistic_systems.contracts.content_address import content_hash
from ..modern_v1_alexnet_envelope import (
    V1RegisteredNormalizedInputBox, registered_normalized_input_box_from_preprocessing,
)
from ..modern_v1_physical_metric import BALANCED_RMS, squared_displacement_interval
from .discovery import FrozenOperatorProposal
from .genuine_issuer_bridge import torch_mask_from_hard_mask
from .logit_intervals import RawTargetLogitIntervalEvaluator, NATIVE_EXECUTION

_SUPPORT_TOKEN = object()


@dataclass(frozen=True, slots=True)
class OperatorPhysicalCRawSupport:
    proposal: FrozenOperatorProposal = field(repr=False, compare=False)
    evaluator: RawTargetLogitIntervalEvaluator = field(repr=False, compare=False)
    input_box: V1RegisteredNormalizedInputBox
    raw_logit_lower: tuple[float, float, float]
    raw_logit_upper: tuple[float, float, float]
    squared_displacement_upper: tuple[float, float]
    dense_logit_lower: float
    dense_logit_upper: float
    dense_relative_squared_displacement_upper: float
    _issuer_token: object | None = field(default=None, kw_only=True, repr=False, compare=False)
    _receipt_hash: str = field(init=False, repr=False)

    def __post_init__(self):
        if self._issuer_token is not _SUPPORT_TOKEN:
            raise ValueError('operator C support must be derived by its source-bound factory')
        object.__setattr__(self, '_receipt_hash', content_hash(self._payload(), prefix='v1-operator-C-raw-support-'))
        self.assert_live()

    def _payload(self):
        discovery = self.proposal.discovery
        return dict(schema='v1_operator_actual_C_raw_support_v1',
            derivation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            proposal_hash=self.proposal.proposal_hash,
            source_opening_hash=discovery.sources.opening_hash,
            parent_mask_hash=discovery.parent_mask.mask_hash,
            child_mask_hash=self.proposal.packed_action.resulting_mask.mask_hash,
            metric_budget_hash=discovery.metric_budget.budget_hash,
            population_hash=discovery.metric_budget.population_identity_hash,
            route_axis_hash=discovery.metric_budget.route_axis_hash,
            target_class=discovery.metric_budget.target_class,
            physical_measurement_profile=BALANCED_RMS,
            evaluator_hash=self.evaluator.evaluator_hash,
            arithmetic_profile_hash=self.evaluator.profile.profile_hash,
            input_box_hash=self.input_box.box_hash,
            preprocessing_hash=self.input_box.preprocessing_hash,
            doses=(0.,.5,1.), raw_logit_lower=self.raw_logit_lower,
            raw_logit_upper=self.raw_logit_upper,
            squared_displacement_upper=self.squared_displacement_upper,
            dense_logit_lower=self.dense_logit_lower, dense_logit_upper=self.dense_logit_upper,
            dense_relative_squared_displacement_upper=self.dense_relative_squared_displacement_upper,
            dense_reference='all_routes_original_checkpoint_layer_and_readout_biases',
            scope='uniform_input_box_conditional_retained_execution_error_model',
            F_J_R_support_available=False, old_shell_KKT_provenance_claimed=False,
            C_outcomes_used=False, statistical_C_admission=False, mask_commit_authority=False)

    @property
    def receipt_hash(self):
        self.assert_live()
        return self._receipt_hash

    def to_json(self):
        self.assert_live()
        return dict(receipt_hash=self._receipt_hash, **self._payload())

    def assert_live(self):
        self.proposal.assert_live()
        self.evaluator.assert_live()
        if self.evaluator.bound_scope != NATIVE_EXECUTION:
            raise ValueError('C execution support requires native execution interval scope')
        self.input_box.assert_integrity()
        source = self.proposal.discovery.sources
        if self.evaluator.net is not source.net:
            raise ValueError('C support evaluator differs from actual model source')
        if self.input_box.preprocessing_hash != source.context_source.preprocessing_hash:
            raise ValueError('C support input box differs from actual preprocessing')
        if content_hash(self._payload(), prefix='v1-operator-C-raw-support-') != self._receipt_hash:
            raise ValueError('frozen C support source/action/budget content changed')

    def assert_images_in_domain(self, images):
        self.assert_live()
        lower = torch.tensor(self.input_box.lower_by_channel, device=images.device,
                             dtype=torch.float64).reshape(1,3,1,1)
        upper = torch.tensor(self.input_box.upper_by_channel, device=images.device,
                             dtype=torch.float64).reshape(1,3,1,1)
        if (not bool(torch.isfinite(images).all()) or bool((images < lower).any())
                or bool((images > upper).any())):
            raise ValueError('actual C images exceed the registered preprocessing domain')


def derive_operator_C_raw_support(proposal: FrozenOperatorProposal,
                                  evaluator: RawTargetLogitIntervalEvaluator) -> OperatorPhysicalCRawSupport:
    if type(proposal) is not FrozenOperatorProposal or type(evaluator) is not RawTargetLogitIntervalEvaluator:
        raise TypeError('C support requires exact operator proposal and numerical evaluator')
    proposal.assert_live()
    evaluator.assert_live()
    if evaluator.bound_scope != NATIVE_EXECUTION:
        raise ValueError('C execution support requires native execution interval scope')
    discovery = proposal.discovery
    if proposal.packed_action is None:
        raise ValueError('empty operator proposal cannot register C support')
    if discovery._physical_replay_owner is not None:
        raise ValueError('C support must be derived before the replay producer is opened')
    if evaluator.net is not discovery.sources.net:
        raise ValueError('C support evaluator differs from actual model source')
    preprocessing = discovery.sources.context_source.source_owner.preprocessing
    if preprocessing is None:
        raise ValueError('C support requires the actual retained preprocessing specification')
    box = registered_normalized_input_box_from_preprocessing(preprocessing)
    net = discovery.sources.net
    child = torch_mask_from_hard_mask(proposal.packed_action.resulting_mask,
                                     net=net, route_axis=discovery.sources.route_axis)
    lo, hi = evaluator.evaluate_registered_box(box,
        masks=(net.mask, net.mask-.5*(net.mask-child), child, torch.ones_like(net.mask)),
        target_class=discovery.metric_budget.target_class,
        use_dense_biases=(False, False, False, True))
    lower, upper = tuple(lo[0,:3].tolist()), tuple(hi[0,:3].tolist())
    support = tuple(squared_displacement_interval(parent_lower=lower[0], parent_upper=upper[0],
                    child_lower=lower[dose], child_upper=upper[dose])[1] for dose in (1,2))
    dense_lower, dense_upper = float(lo[0,3]), float(hi[0,3])
    dense_support = squared_displacement_interval(parent_lower=dense_lower, parent_upper=dense_upper,
        child_lower=lower[2], child_upper=upper[2])[1]
    return OperatorPhysicalCRawSupport(proposal, evaluator, box, lower, upper, support,
                                      dense_lower, dense_upper, dense_support,
                                      _issuer_token=_SUPPORT_TOKEN)
