"""Native K/O execution over the operator's verified functional model.

There is one forward owner: MaskedAlexNet. Native calibration, sampling,
statistical decisions and controller mutation retain their existing owners.
"""
from dataclasses import dataclass, field
from collections.abc import Mapping
from pathlib import Path
import hashlib

import numpy as np
import torch

from mechanistic_systems.backends.readout_projection import RuntimeReadoutProjectionSpec
from mechanistic_systems.circuits.torch_evaluator import ReadoutResponse
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.contracts.science import QuerySpec
from .modern_v1_operator_data import OperatorNativeImageDataPlane
from .tractable_v1.genuine_issuer_bridge import OpenedActualExecutionSources, torch_mask_from_hard_mask
from .tractable_v1.masked_alexnet_paired_bce import _tensor_lease, _check_tensor_lease


@dataclass(frozen=True, slots=True)
class _OperatorCompiledMask:
    binding_hash: str
    route_axis_hash: str
    route_count: int
    packed_bits: bytes

    @property
    def mask(self):
        return HardMask(self.route_axis_hash, self.route_count, self.packed_bits)


@dataclass(frozen=True, slots=True)
class OperatorNativeModelEvaluator:
    sources: OpenedActualExecutionSources
    data_plane: OperatorNativeImageDataPlane
    query: QuerySpec
    _settings: tuple = field(init=False, repr=False, compare=False)
    _source_sha256: str = field(init=False, repr=False, compare=False)
    _dense_mask: HardMask = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if type(self.sources) is not OpenedActualExecutionSources:
            raise TypeError('native operator evaluation requires opened actual execution sources')
        self._validate_data_plane()
        if type(self.query) is not QuerySpec:
            raise TypeError('native operator evaluation requires exact plane and query owners')
        object.__setattr__(self, '_settings', self._execution_settings())
        object.__setattr__(self, '_source_sha256', hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        object.__setattr__(self, '_dense_mask', HardMask.dense(self.sources.route_axis.route_count,
            route_axis_hash=self.sources.route_axis.route_axis_hash))
        self.assert_live()

    def _validate_data_plane(self):
        if type(self.data_plane) is not OperatorNativeImageDataPlane:
            raise TypeError('native operator requires the exact optimization data plane')

    def _execution_settings(self):
        return (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled(),
            torch.get_num_threads(), torch.is_autocast_enabled(self.sources.net.device.type))

    def _assert_model_and_query(self):
        self.sources.assert_live()
        if self._execution_settings() != self._settings or self._settings[-1]:
            raise ValueError('native operator numerical settings changed or autocast is enabled')
        if (self.query.content_hash != self.sources.fq_registration.query_contract_hash
                or self.query.query_kind != 'class_logit'
                or self.query.response_semantics != 'raw_logit'
                or type(self.query.evaluator.get('target_class_id')) is not int
                or self.query.evaluator.get('target_class_id') != self.data_plane.target_class_id
                or not 0 <= self.data_plane.target_class_id < self.sources.net.readout_bias.numel()):
            raise ValueError('native operator query differs from the opened raw target-logit source')

    def assert_live(self):
        self._assert_model_and_query()
        self.data_plane.assert_source(self.sources.context_source)

    @property
    def binding_hash(self):
        return content_hash(dict(schema='v1_operator_native_model_v1',
            source_opening_hash=self.sources.opening_hash,
            data_plane_binding_hash=self.data_plane.binding_hash,
            query_hash=self.query.content_hash,
            execution_settings=self._settings, source_sha256=self._source_sha256,
            bias_law='dense_mask_original_biases_other_masks_frozen_circuit_biases'),
            prefix='v1-operator-native-model-')

    @property
    def readout_bias(self):
        self.assert_live()
        # The native fitter may convert this to NumPy; do not expose a writable
        # view of the source tensor through that interface.
        return self.sources.net._dense_readout_bias.detach().clone()

    @property
    def route_blocks(self):
        return self.sources.route_axis.route_blocks

    def compile_hard_mask(self, mask):
        self.assert_live()
        if (type(mask) is not HardMask or mask.route_axis_hash != self.sources.route_axis.route_axis_hash
                or mask.route_count != self.sources.route_axis.route_count):
            raise ValueError('native operator mask differs from the verified route axis')
        # Snapshot packed bytes; no expanded device masks persist in this object.
        snapshot = HardMask(mask.route_axis_hash, mask.route_count, bytes(mask.packed_bits))
        return _OperatorCompiledMask(self.binding_hash, snapshot.route_axis_hash,
            snapshot.route_count, snapshot.packed_bits)

    def compile_mask(self, masks_by_layer):
        self.assert_live()
        if not isinstance(masks_by_layer, Mapping) or set(masks_by_layer) != {b.layer_id for b in self.route_blocks}:
            raise ValueError('native operator requires the complete declared mask blocks')
        values = np.empty(self.sources.route_axis.route_count, dtype=bool)
        for block in self.route_blocks:
            piece = np.asarray(masks_by_layer[block.layer_id])
            if piece.shape != block.shape or not np.all((piece == 0) | (piece == 1)):
                raise ValueError('native operator mask blocks must be binary with exact shapes')
            values[block.offset:block.stop] = piece.reshape(-1)
        return self.compile_hard_mask(HardMask.from_bool_array(values,
            route_axis_hash=self.sources.route_axis.route_axis_hash))

    def readout_responses_from_batches(self, batches, *, compiled_masks, lane_keys,
                                      readout_projection=None):
        self.assert_live()
        compiled = tuple(compiled_masks)
        if not compiled or len(compiled) != len(lane_keys):
            raise ValueError('native operator readout requires aligned nonempty mask lanes')
        target = self.data_plane.target_class_id
        if (type(readout_projection) is not RuntimeReadoutProjectionSpec
                or readout_projection.column_ids != (target,)):
            raise ValueError('native operator readout requires its registered target coordinate')
        for item in compiled:
            if type(item) is not _OperatorCompiledMask or item.binding_hash != self.binding_hash:
                raise ValueError('native operator compiled mask belongs to another execution source')
        masks = tuple(item.mask for item in compiled)
        net = self.sources.net
        tensors = tuple(torch_mask_from_hard_mask(mask, net=net,
            route_axis=self.sources.route_axis) for mask in masks)
        dense_flags = tuple(mask == self._dense_mask for mask in masks)
        leases = tuple(_tensor_lease(t, name=f'native-mask:{i}') for i, t in enumerate(tensors))
        rows = [[] for _ in compiled]
        labels = []
        contexts = []
        iterator = iter(batches)
        try:
            while True:
                # Check before advancing the source iterator (before outcomes).
                self.assert_live()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                batch.validate()
                if any(c not in self.data_plane.expected_labels for c in batch.context_ids):
                    raise ValueError('native operator batch contains an unknown context')
                actual_labels = torch.as_tensor(batch.labels).detach().cpu().reshape(-1)
                expected_labels = torch.tensor([self.data_plane.expected_labels[c] for c in batch.context_ids])
                if not torch.equal(actual_labels, expected_labels):
                    raise ValueError('native operator batch labels differ from the frozen source')
                input_lease = _tensor_lease(batch.inputs, name='native-images')
                label_lease = _tensor_lease(batch.labels, name='native-labels')
                with torch.no_grad():
                    for i, (mask_tensor, use_dense) in enumerate(zip(tensors, dense_flags, strict=True)):
                        self.assert_live()
                        _check_tensor_lease(mask_tensor, leases[i])
                        value = net.forward(batch.inputs, mask_override=mask_tensor,
                            use_dense_biases=use_dense)['logits'][:, target:target+1]
                        self.assert_live()
                        _check_tensor_lease(batch.inputs, input_lease)
                        _check_tensor_lease(batch.labels, label_lease)
                        _check_tensor_lease(mask_tensor, leases[i])
                        if not bool(torch.isfinite(value).all()):
                            raise ValueError('native operator forward produced nonfinite logits')
                        rows[i].append(value.detach().to(device='cpu', dtype=torch.float64).numpy().copy())
                contexts.extend(batch.context_ids)
                labels.append(actual_labels.numpy().copy())
                del batch, value
        finally:
            close = getattr(iterator, 'close', None)
            if callable(close):
                close()
        self.assert_live()
        if not contexts or len(contexts) != len(set(contexts)):
            raise ValueError('native operator readout requires nonempty unique contexts')
        label_array = np.concatenate(labels)
        responses = tuple(ReadoutResponse(context_ids=tuple(contexts),
            values=np.concatenate(lane), labels=label_array) for lane in rows)
        for response in responses:
            response.validate()
        return responses


def build_operator_native_bce_evaluator(*, sources, data_plane, query, binding, mask_registry):
    """Construct retained native K/O owners over one verified functional source.

    This factory retains the native census fallback. Analytic partial support,
    campaign source registration and complete production routing are separate.
    """
    from . import modern_v1_alexnet_backend as native
    from .modern_v1_campaign_contract import V1OptionACampaignBinding, DENSE_REFERENT_MASK_METADATA_KEY
    evaluator = OperatorNativeModelEvaluator(sources, data_plane, query)
    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError('operator native K/O requires an exact campaign binding')
    contract = binding.contract
    if (contract.campaign_contract_hash != sources.fq_registration.campaign_contract_hash
            or contract.context_population_hash != data_plane.population_hash
            or contract.route_axis_hash != sources.route_axis.route_axis_hash
            or contract.query_id != query.query_id):
        raise ValueError('operator native K/O campaign differs from actual source/query')
    dense = HardMask.dense(sources.route_axis.route_count, route_axis_hash=sources.route_axis.route_axis_hash)
    if (contract.metadata.get(DENSE_REFERENT_MASK_METADATA_KEY) != dense.mask_hash
            or mask_registry.get(dense.mask_hash) != dense):
        raise ValueError('operator native K/O lacks its original dense referent')
    owner = sources.context_source.source_owner
    counts = (len(owner.target_context_ids), len(owner.complement_context_ids))
    if min(counts) < 2:
        raise ValueError('native affine calibration needs K and residual O in each stratum')
    capacity = tuple(n-1 for n in counts)
    policy = dict(schema='partition_free_v1_alexnet_causal_calibration_prefix_policy_v1',
        full_population_stratum_counts=counts, calibration_sampling_capacity_stratum_counts=capacity,
        registered_look_cumulative_stratum_counts=native.balanced_dyadic_census_looks(*capacity))
    base = native.FixedAffineBCECalibration(query_id=query.query_id, query_hash=query.content_hash,
        target_class_id=data_plane.target_class_id, scale=1., offset=0.,
        calibration_state_id='operator-pre-K-proposal-only',
        calibration_source_hash=content_hash(dict(schema='v1_operator_pre_K_source',
            native_model_binding_hash=evaluator.binding_hash)),
        calibration_context_axis_hash=owner.source_context_axis_hash,
        source_label_map_hash=owner.source_label_map_hash,
        model_state_hash=sources.fq_registration.expected_model_state_hash)
    installer = native._OptimizationOnlyMaskCalibrationInstaller(evaluator=evaluator,
        data_plane=data_plane, route_axis=sources.route_axis,
        campaign_route_axis_hash=sources.route_axis.route_axis_hash, calibration_context_ids=(),
        calibration_context_selection_hash=content_hash(policy, prefix='v1-operator-native-K-policy-'),
        calibration_context_selection_policy=policy, expected_labels=data_plane.expected_labels,
        target_class_id=data_plane.target_class_id, base_calibration=base,
        mask_registry=mask_registry, calibration_registry={})
    return native._DirectFixedCalibrationBCEEvaluator(evaluator=evaluator, data_plane=data_plane,
        route_axis=sources.route_axis, campaign_binding_hash=binding.binding_hash,
        campaign_route_axis_hash=sources.route_axis.route_axis_hash,
        campaign_calibration_identity=contract.calibration_identity, dense_mask_hash=dense.mask_hash,
        calibration_installer=installer, mask_registry=mask_registry,
        expected_labels=data_plane.expected_labels, paired_bce_order_registration=sources.fq_registration)
