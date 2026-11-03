"""Apply a native controller's committed mask to its shared functional model.

This is a process-local execution handoff, not a decision oracle or a portable
checkpoint. Native generation retirement retains authority over K/O state.
"""
from dataclasses import dataclass, field

from mechanistic_systems.contracts.content_address import content_hash
from .modern_v1_controller import V1ControllerState
from .modern_v1_operator_model import OperatorNativeModelEvaluator
from .tractable_v1.discovery import _packed_parent
from .tractable_v1.genuine_issuer_bridge import torch_mask_from_hard_mask
from .tractable_v1.masked_alexnet_paired_bce import _tensor_lease, _check_tensor_lease


@dataclass(frozen=True, slots=True)
class OperatorCommittedMaskHandoff:
    controller_state_hash: str
    native_model_binding_hash: str
    parent_mask_hash: str
    child_mask_hash: str
    accepted_step_count: int
    acquisition_generation_index: int
    _mask_lease: object = field(repr=False, compare=False)

    @property
    def receipt_hash(self):
        return content_hash(dict(schema='v1_operator_committed_mask_handoff_v1',
            controller_state_hash=self.controller_state_hash,
            native_model_binding_hash=self.native_model_binding_hash,
            parent_mask_hash=self.parent_mask_hash, child_mask_hash=self.child_mask_hash,
            accepted_step_count=self.accepted_step_count,
            acquisition_generation_index=self.acquisition_generation_index),
            prefix='v1-operator-mask-handoff-')


def synchronize_operator_committed_mask(*, controller, native_evaluator):
    """Retire exact accepted K/O state and install its child, without images.

    Repeating this call for the same unchanged state is idempotent. All fallible
    mask validation/materialization precedes retirement. A process failure at
    the handoff still requires source reconstruction; this is not crash recovery.
    """
    from .modern_v1_alexnet_backend import _DirectFixedCalibrationBCEEvaluator
    if type(controller) is not V1ControllerState or type(native_evaluator) is not _DirectFixedCalibrationBCEEvaluator:
        raise TypeError('operator mask handoff requires exact controller and native evaluator')
    controller.__post_init__()
    model = native_evaluator.evaluator
    if (type(model) is not OperatorNativeModelEvaluator
            or native_evaluator._operator_native_model is not model
            or native_evaluator.calibration_installer.evaluator is not model
            or native_evaluator.data_plane is not model.data_plane
            or native_evaluator.campaign_binding_hash != controller.binding.binding_hash):
        raise ValueError('operator mask handoff differs from the shared native source')
    model.assert_live()
    if (controller.phase not in ('needs_acquisition', 'stopped')
            or controller.acquisition is not None or controller.active_candidate is not None
            or not controller.attempt_receipts or not controller.transition_receipts):
        raise PermissionError('operator mask handoff requires a completed accepted transition')
    terminal = controller.attempt_receipts[-1]
    transition = controller.transition_receipts[-1]
    if (terminal.action != 'accept_candidate' or terminal.mask_mutated is not True
            or terminal.candidate_mask_hash != controller.current_mask.mask_hash
            or transition.child_mask_hash != controller.current_mask.mask_hash
            or transition.parent_mask_hash != terminal.parent_mask_hash):
        raise PermissionError('operator mask handoff requires the last accepted child')
    state_hash = controller.state_hash
    binding_hash = model.binding_hash
    previous = getattr(native_evaluator, '_operator_committed_mask_handoff', None)
    if previous is not None and previous.controller_state_hash == state_hash:
        if previous.native_model_binding_hash != binding_hash:
            raise ValueError('operator mask handoff source changed')
        _check_tensor_lease(model.sources.net.mask, previous._mask_lease)
        return previous
    cached = native_evaluator._active_candidate_calibration_freeze
    if (cached is None or cached.metadata.get('operator_native_model_binding_hash') != binding_hash):
        raise ValueError('operator mask handoff lacks the exact active native calibration')
    native_evaluator.verify_candidate_calibration_freeze(cached)
    if _packed_parent(model.sources).mask_hash != transition.parent_mask_hash:
        raise ValueError('operator functional mask differs from the accepted parent')
    mask = torch_mask_from_hard_mask(controller.current_mask, net=model.sources.net,
        route_axis=model.sources.route_axis)
    receipt = OperatorCommittedMaskHandoff(state_hash, binding_hash,
        transition.parent_mask_hash, controller.current_mask.mask_hash,
        controller.accepted_step_count, controller.acquisition_generation_index,
        _tensor_lease(mask, name='operator committed mask'))
    # This retained method checks the complete attempt, transition, K and O
    # lineage before clearing any generation state. No alternate commit engine.
    native_evaluator.retire_committed_generation_state(controller)
    model.sources.net.mask = mask
    native_evaluator._operator_committed_mask_handoff = receipt
    return receipt
