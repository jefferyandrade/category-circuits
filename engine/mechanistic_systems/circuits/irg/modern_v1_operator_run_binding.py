"""Persistent operator source identity across every retained journal state."""
from collections.abc import Mapping
from dataclasses import replace

from mechanistic_systems.contracts.content_address import content_hash

KEY='operator_run_source_binding'
SCHEMA='v1_operator_run_source_binding_v1'


def validate_config_binding(metadata,source_seal_hash):
    """Data-only check also used by the inert checkpoint inspector."""
    value=metadata.get(KEY)
    if value is None:
        if KEY in metadata or source_seal_hash.startswith('v1-operator-runtime-source-seal-'):
            raise ValueError('operator persistent source seal lost its run binding')
        return None
    if (not isinstance(value,Mapping) or set(value)!= {'schema','upstream_source_seal_hash',
            'source','factory_signature','binding_hash'} or value['schema']!=SCHEMA):
        raise ValueError('operator run binding schema differs')
    payload={k:v for k,v in value.items() if k!='binding_hash'}
    digest=content_hash(payload,prefix='v1-operator-run-binding-')
    if (digest!=value['binding_hash'] or source_seal_hash!=content_hash(dict(
            upstream_source_seal_hash=value['upstream_source_seal_hash'],operator_run_binding_hash=digest),
            prefix='v1-operator-runtime-source-seal-')):
        raise ValueError('operator run binding differs from persistent source seal')
    return value


def validate_history_binding(metadata,source_seal_hash,acquisition_receipt_hashes):
    value=validate_config_binding(metadata,source_seal_hash)
    if (value is None and any(type(h) is str and h.startswith('v1-operator-controller-acquisition-')
                             for h in acquisition_receipt_hashes)):
        raise ValueError('operator acquisition history lacks persistent run binding; no retroactive upgrade')
    return value


def _source_payload(sources):
    from .tractable_v1.genuine_issuer_bridge import OpenedActualExecutionSources
    from .modern_v1_operator_checkpoint import _circuit_bias_payload,_source_versions
    if type(sources) is not OpenedActualExecutionSources:
        raise TypeError('operator run requires exact verified execution sources')
    sources.assert_live()
    manifest=sources.context_source.image_source.content_manifest_hash
    if manifest is None:
        raise ValueError('operator run requires its pre-outcome image content manifest')
    return dict(source_opening_hash=sources.opening_hash,
        model_checkpoint_binding_hash=sources.model_checkpoint_binding_hash,
        actual_source_binding_hash=sources.actual_source_binding_hash,
        image_content_manifest_hash=manifest,circuit_biases=_circuit_bias_payload(sources),
        source_versions=_source_versions())


def bind_operator_runtime_config(config,factory):
    """Freeze before D/K/O and before the initial runtime/entropy checkpoint."""
    from .modern_v1_runtime import V1RuntimeExecutionConfig
    from .modern_v1_operator_factory import OperatorAcquisitionFactory
    if type(config) is not V1RuntimeExecutionConfig or type(factory) is not OperatorAcquisitionFactory:
        raise TypeError('operator binding requires exact runtime config and factory')
    sources=factory._model.sources
    if (factory.last_discovery is not None or sources.context_source.image_source.stats()['decode_requests_submitted']
            or factory.native_evaluator._active_candidate_calibration_freeze is not None):
        raise ValueError('operator run binding must precede all image outcomes')
    if KEY in config.metadata:
        validate_bound_operator_sources(config,sources,factory=factory)
        return config
    payload=dict(schema=SCHEMA,upstream_source_seal_hash=config.source_seal_hash,
        source=_source_payload(sources),factory_signature=factory._coordinates())
    payload['binding_hash']=content_hash(payload,prefix='v1-operator-run-binding-')
    seal=content_hash(dict(upstream_source_seal_hash=config.source_seal_hash,
        operator_run_binding_hash=payload['binding_hash']),prefix='v1-operator-runtime-source-seal-')
    return replace(config,source_seal_hash=seal,metadata={**config.metadata,KEY:payload})


def validate_bound_operator_sources(config,sources,*,factory=None,current_mask=None):
    value=validate_config_binding(config.metadata,config.source_seal_hash)
    if value is None:
        raise ValueError('operator runtime lacks its pre-outcome persistent source binding')
    if sources is None:
        raise ValueError('operator checkpoint requires freshly verified execution sources')
    if content_hash(value['source'])!=content_hash(_source_payload(sources)):
        raise ValueError('operator runtime source/model/bias/content/configuration changed')
    if factory is not None and value['factory_signature']!=factory._coordinates():
        raise ValueError('operator runtime factory configuration changed')
    if current_mask is not None:
        from .tractable_v1.discovery import _packed_parent
        if _packed_parent(sources)!=current_mask:
            raise ValueError('operator restored functional mask differs from checkpoint current mask')
    return value


def restore_operator_current_mask(sources,current_mask):
    """Prepare the verified journal mask only on an unopened fresh source."""
    from .tractable_v1.genuine_issuer_bridge import torch_mask_from_hard_mask
    from .tractable_v1.discovery import _packed_parent
    sources.assert_live()
    if sources.context_source.image_source.stats()['decode_requests_submitted']:
        raise ValueError('operator mask reconstruction requires unopened fresh sources')
    if _packed_parent(sources)==current_mask:return
    prepared=torch_mask_from_hard_mask(current_mask,net=sources.net,route_axis=sources.route_axis)
    sources.net.mask=prepared
