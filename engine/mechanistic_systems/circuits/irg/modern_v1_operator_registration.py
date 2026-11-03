"""Explicit pre-outcome settings for constructing the canonical operator factory.

This registration selects a realization and freezes raw function budgets. It
does not certify sampling, physical validity, acceptance or production quality.
"""
from dataclasses import dataclass, fields
from collections.abc import Mapping
import json
import math

import torch

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from .modern_v1_operator_checkpoint_wire import canonical_json
from .modern_v1_operator_factory import OperatorAcquisitionFactory
from .modern_v1_operator_model import OperatorNativeModelEvaluator, build_operator_native_bce_evaluator
from .tractable_v1.discovery import discovery_config_hash, _packed_parent
from .tractable_v1.metric_freeze import freeze_raw_logit_metric_budget, SOURCE_EXPLICIT_RAW
from .tractable_v1.selection_policy import SelectionPolicyConfig, SELECTION_POLICY_SCHEMA

SCHEMA = 'v1_operator_factory_registration_v1'
HASH_PREFIX = 'v1-operator-factory-registration-'
_FIELDS = frozenset(('schema', 'campaign_contract_hash', 'fq_registration_hash',
    'native_model_binding_hash', 'discovery_counts', 'capture_batch',
    'max_factor_bytes', 'factor_dtype', 'seed', 'selection_policy',
    'local_raw_logit_rms_radius', 'dense_relative_raw_logit_rms_radius',
    'bce_ceiling', 'registration_hash'))


def _json_value(value):
    # Admission/environment metadata may already be recursively frozen.
    if isinstance(value, Mapping):
        return {k:_json_value(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [_json_value(v) for v in value]
    return value


def _decode(raw):
    if type(raw) is not bytes:
        raise TypeError('operator registration requires immutable canonical bytes')
    value = json.loads(raw)
    if type(value) is not dict or set(value) != _FIELDS or value['schema'] != SCHEMA or canonical_json(value) != raw:
        raise ValueError('operator factory registration schema or canonical bytes differ')
    digest = content_hash({k:v for k,v in value.items() if k != 'registration_hash'}, prefix=HASH_PREFIX)
    if digest != value['registration_hash']:
        raise ValueError('operator factory registration hash differs')
    for name in ('campaign_contract_hash', 'fq_registration_hash', 'native_model_binding_hash'):
        if type(value[name]) is not str or not value[name].strip():
            raise ValueError(f'operator registration requires {name}')
    for name in ('local_raw_logit_rms_radius', 'dense_relative_raw_logit_rms_radius', 'bce_ceiling'):
        if type(value[name]) not in (int,float) or not math.isfinite(value[name]) or value[name] <= 0:
            raise ValueError(f'{name} must be a finite positive number')
    counts = value['discovery_counts']
    if type(counts) is not list or len(counts) != 2 or any(type(n) is not int or n < 2 for n in counts):
        raise ValueError('registered cross-fit discovery requires at least two contexts per stratum')
    policy_payload = value['selection_policy']
    policy_fields = {f.name for f in fields(SelectionPolicyConfig)}
    if (type(policy_payload) is not dict or set(policy_payload) != policy_fields | {'schema'}
            or policy_payload['schema'] != SELECTION_POLICY_SCHEMA
            or type(policy_payload['frontier_tolerance']) not in (int,float)):
        raise ValueError('operator registration requires the complete explicit selection policy')
    policy = SelectionPolicyConfig(**{k:v for k,v in policy_payload.items() if k != 'schema'})
    if value['factor_dtype'] not in ('torch.float32','torch.float64'):
        raise ValueError('operator registration requires an explicit FP32 or FP64 factor realization')
    if type(value['seed']) is not int or value['seed'] < 0:
        raise ValueError('operator registration seed must be a nonnegative integer')
    settings = dict(capture_batch=value['capture_batch'], max_factor_bytes=value['max_factor_bytes'],
        factor_dtype=getattr(torch,value['factor_dtype'].split('.')[1]), policy=policy, seed=value['seed'])
    discovery_config_hash(**settings)
    return value, settings


@dataclass(frozen=True, slots=True)
class OperatorFactoryRegistration:
    raw: bytes

    def __post_init__(self):
        _decode(self.raw)

    @classmethod
    def from_mapping(cls, payload, *, expected_registration_hash):
        if not isinstance(payload, Mapping):
            raise TypeError('operator registration requires a mapping')
        result = cls(canonical_json(_json_value(payload)))
        if result.registration_hash != expected_registration_hash:
            raise ValueError('operator registration differs from the admitted hash')
        return result

    @property
    def registration_hash(self):
        return _decode(self.raw)[0]['registration_hash']

    def to_json(self):
        return _decode(self.raw)[0]


def build_registered_operator_factory(*, registration, sources, data_plane, query, binding):
    """Construct from verified sources and explicit settings before D/K/O.

    The artifact factory must bind the expected registration hash to admission.
    This component constructor cannot replace full artifact/population admission.
    """
    if type(registration) is not OperatorFactoryRegistration:
        raise TypeError('operator factory requires its exact registration')
    from .modern_v1_campaign_contract import V1OptionACampaignBinding
    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError('registered operator factory requires an exact campaign binding')
    value, settings = _decode(registration.raw)
    model = OperatorNativeModelEvaluator(sources, data_plane, query)
    if sources.context_source.image_source.stats()['decode_requests_submitted']:
        raise ValueError('registered operator construction must precede image outcomes')
    if (value['native_model_binding_hash'] != model.binding_hash
            or value['fq_registration_hash'] != sources.fq_registration.registration_hash
            or value['campaign_contract_hash'] != binding.contract.campaign_contract_hash):
        raise ValueError('operator registration differs from native model, F/Q or campaign')
    dense = HardMask.dense(sources.route_axis.route_count, route_axis_hash=sources.route_axis.route_axis_hash)
    if _packed_parent(sources) != dense:
        raise ValueError('registered operator construction requires the original dense source')
    counts = tuple(value['discovery_counts'])
    owner = sources.context_source.source_owner
    if (counts[0] > len(owner.target_context_ids) or counts[1] > len(owner.complement_context_ids)
            or settings['capture_batch'] > sources.context_source.max_batch_rows):
        raise ValueError('registered discovery exceeds source population or batch capacity')
    budget = _registered_metric_budget(value, settings, model)
    native = build_operator_native_bce_evaluator(sources=sources, data_plane=data_plane,
        query=query, binding=binding, mask_registry={dense.mask_hash:dense})
    return OperatorAcquisitionFactory(native_evaluator=native, metric_budget=budget,
        counts=counts, registration_hash=registration.registration_hash, **settings)


def _registered_metric_budget(value, settings, model):
    sources, data_plane = model.sources, model.data_plane
    owner = sources.context_source.source_owner
    return freeze_raw_logit_metric_budget(target_class=data_plane.target_class_id,
        bce_ceiling=value['bce_ceiling'], radius_source_kind=SOURCE_EXPLICIT_RAW,
        raw_logit_rms_radius=value['local_raw_logit_rms_radius'], dense_calibration=None,
        normalization_counts=None, config_hash=discovery_config_hash(**settings),
        population_identity_hash=owner.source_population_hash,
        route_axis_hash=sources.route_axis.route_axis_hash,
        final_dense_relative_rms_radius=value['dense_relative_raw_logit_rms_radius'])


def admitted_operator_registration(*, admission, environment):
    """Read settings from the sealed environment using the admission's hash.

    Callers must verify environment artifact bytes. This joins declarations;
    it neither opens sources nor supplies population/sampling authority.
    """
    from .modern_v1_alexnet import V1AlexNetBackendAdmission
    if type(admission) is not V1AlexNetBackendAdmission or admission.proposal_protocol != 'operator_binary_d_c_v1':
        raise TypeError('operator settings require an exact operator admission')
    metadata = environment.get('metadata') if isinstance(environment, Mapping) else None
    if not isinstance(metadata, Mapping):
        raise ValueError('operator environment requires explicit registration metadata')
    registration = OperatorFactoryRegistration.from_mapping(
        metadata.get('operator_factory_registration'),
        expected_registration_hash=admission.metadata.get('operator_factory_registration_hash'))
    value = registration.to_json()
    if (value['campaign_contract_hash'] != admission.campaign_contract_hash
            or value['fq_registration_hash'] != admission.metadata.get('fq_acquisition_registration_hash')):
        raise ValueError('operator settings differ from the admitted campaign or F/Q registration')
    return registration


def validate_registered_operator_bundle(*, registration, bundle):
    """Require the returned executable factory and threshold to match admission."""
    from .modern_v1_production import _bound_operator_factory
    from .modern_v1_alexnet_backend import _ThresholdFactory
    if type(registration) is not OperatorFactoryRegistration:
        raise TypeError('operator bundle requires an exact settings registration')
    factory = _bound_operator_factory(bundle)
    if type(factory) is not OperatorAcquisitionFactory:
        raise TypeError('operator bundle requires its exact acquisition factory')
    value, settings = _decode(registration.raw)
    model = factory._model
    if (factory.registration_hash != registration.registration_hash
            or factory._coordinates() != factory._signature
            or factory.settings != settings
            or factory.counts != tuple(value['discovery_counts'])
            or factory.metric_budget != _registered_metric_budget(value, settings, model)
            or model.binding_hash != value['native_model_binding_hash']
            or model.sources.fq_registration.registration_hash != value['fq_registration_hash']
            or bundle.binding.contract.campaign_contract_hash != value['campaign_contract_hash']):
        raise ValueError('constructed operator differs from its admitted registration')
    dense = HardMask.dense(model.sources.route_axis.route_count,
        route_axis_hash=model.sources.route_axis.route_axis_hash)
    if (model.sources.context_source.image_source.stats()['decode_requests_submitted']
            or _packed_parent(model.sources) != dense):
        raise ValueError('operator admission requires an unopened dense execution source')
    threshold = bundle.threshold_factory.callable_value
    if (type(threshold) is not _ThresholdFactory
            or threshold.maximum_dense_relative_mean_damage != value['bce_ceiling']
            or threshold.source_hash != registration.registration_hash):
        raise ValueError('operator threshold differs from the admitted BCE budget')
