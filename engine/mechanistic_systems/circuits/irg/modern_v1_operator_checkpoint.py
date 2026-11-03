"""Closed operator acquisitions as inert JSON and verified source reconstruction.

The outer checkpoint must authenticate the payload hash and acquisition hash.
This module neither unpickles bytes nor restores a controller/calibration epoch.
It rebinds an already frozen D/C graph to freshly verified execution sources.
"""
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import hashlib
import json

import torch

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from . import modern_v1_operator_acquisition as admission
from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_campaign_contract import V1OptionACampaignBinding
from .tractable_v1 import discovery as discovery_module
from .tractable_v1 import physical_replay as physical_module
from .tractable_v1.genuine_issuer_bridge import OpenedActualExecutionSources
from .tractable_v1.metric_freeze import RawLogitMetricBudgetBinding
from .tractable_v1.selection_policy import SelectionPolicyConfig
from .tractable_v1.masked_alexnet_paired_bce import _tensor_lease
from .modern_v1_operator_checkpoint_wire import SCHEMA, ACTIVE_SCHEMA, inspect_wire_payload, validate_active_projection



def _source_versions():
    base=Path(__file__).parent
    names=('modern_v1_operator_checkpoint.py','modern_v1_operator_acquisition.py',
           'modern_v1_alexnet_fq_registration.py','tractable_v1/discovery.py',
           'tractable_v1/physical_replay.py','tractable_v1/c_support.py',
           'tractable_v1/logit_intervals.py','tractable_v1/genuine_issuer_bridge.py')
    names+=('tractable_v1/image_stream.py','modern_v1_operator_factory.py')
    names+=('modern_v1_alexnet_backend.py','modern_v1_operator_model.py','modern_v1_operator_data.py')
    names+=('modern_v1_operator_checkpoint_wire.py',)
    names+=('modern_v1_runtime.py','modern_v1_checkpoint_codec.py','modern_v1_controller.py')
    names+=('modern_v1_operator_run_binding.py',)
    names+=('modern_v1_production.py',)
    names+=('modern_v1_operator_bundle.py','modern_v1_alexnet.py',)
    names+=('modern_v1_operator_terminal_model.py',)
    names+=('modern_v1_operator_terminal_data.py',)
    names+=('modern_v1_operator_export.py',)
    names+=('modern_v1_operator_registration.py','modern_v1_operator_dataset.py')
    names+=('modern_v1_operator_production_inputs.py',)
    names+=('modern_v1_operator_terminal.py',)
    names+=('modern_v1_operator_production.py',)
    names+=('modern_v1_operator_sources.py','../../measurement/models.py',
            '../../measurement/model_sources.py','../../measurement/contexts.py','../../core/architectures.py')
    names+=tuple('tractable_v1/'+name for name in ('alexnet_factors.py','operators.py','solvers.py',
        'selection.py','selection_policy.py','selection_refinement.py','selection_pair_stream.py',
        'selection_exchange_order.py','selection_pair_batch.py','chart_safeguards.py'))
    return {name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in names}


def _json_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def _circuit_bias_payload(sources):
    biases={g.name:dict(dtype=str(t.dtype),values=t.detach().cpu().tolist())
            for g,t in zip(sources.net.axis.layers,sources.net.biases,strict=True)}
    biases['classifier.6']=dict(dtype=str(sources.net.readout_bias.dtype),
        values=sources.net.readout_bias.detach().cpu().tolist())
    return biases


def inspect_operator_acquisition_payload(payload, *, expected_payload_sha256):
    """Check inert canonical bytes against an independently retained digest."""
    if type(payload) is not bytes or type(expected_payload_sha256) is not str:
        raise TypeError('operator checkpoint requires exact bytes and a retained digest')
    if hashlib.sha256(payload).hexdigest()!=expected_payload_sha256:
        raise ValueError('operator checkpoint payload digest differs')
    value=inspect_wire_payload(payload)
    if value.get('source_versions')!=_source_versions():
        raise ValueError('operator checkpoint execution/reconstruction source version changed')
    return value


def export_operator_acquisition_payload(acquisition):
    """Snapshot closed, source-issued evidence; retain no tensor/lock/capability."""
    if type(acquisition) is not admission.V1OperatorAcquisition:
        raise TypeError('operator export requires an exact issued acquisition')
    acquisition.assert_integrity()
    if acquisition.candidate is None:
        proposal=acquisition._no_candidate_proposal
        physical=acquisition._no_candidate_physical
    else:
        physical=acquisition.candidate._producer
        proposal=physical.proposal
    d=proposal.discovery
    content_manifest=d.sources.context_source.image_source.content_manifest_hash
    if content_manifest is None:
        raise ValueError('operator export requires a pre-outcome image content manifest')
    if not d._factors_released or d.factors is not None:
        raise ValueError('operator export requires closed discovery factors')
    physical_state=None
    if physical is not None:
        if physical._pending is not None or physical._running or physical._admission_owner is not acquisition:
            raise ValueError('operator export requires sealed idle physical C')
        physical_state=dict(producer_hash=physical.producer_hash,
            stream_hash=physical.stream.receipt_hash,support_hash=physical.raw_support.receipt_hash,
            counts=physical.cumulative_counts,
            lower_sums=[[(v.numerator,v.denominator) for v in row] for row in physical._lower_sums],
            upper_sums=[[(v.numerator,v.denominator) for v in row] for row in physical._upper_sums],
            row_chain=physical._row_chain,progress_hash=physical.population_progress.receipt_hash)
    batch,limit,dtype,seed=d._settings
    value=dict(schema=SCHEMA,source_versions=_source_versions(),source_opening_hash=d.sources.opening_hash,
        image_content_manifest_hash=content_manifest,
        model_checkpoint_binding_hash=d.sources.model_checkpoint_binding_hash,
        actual_source_binding_hash=d.sources.actual_source_binding_hash,circuit_biases=_circuit_bias_payload(d.sources),
        acquisition_hash=acquisition.acquisition_hash,receipt=asdict(acquisition.receipt),
        candidate_definition=None if acquisition.candidate is None else acquisition.candidate.definition_payload(),
        parent_mask=dict(route_axis_hash=d.parent_mask.route_axis_hash,route_count=d.parent_mask.route_count,
            packed_bits=d.parent_mask.packed_bits.hex(),mask_hash=d.parent_mask.mask_hash),
        child_packed_bits=None if proposal.packed_action is None else proposal.packed_action.resulting_mask.packed_bits.hex(),
        budget=d.metric_budget.to_json(),policy=d.policy.to_json(),
        target_resource=d.target_resource,
        settings=dict(capture_batch=batch,max_factor_bytes=limit,factor_dtype=str(dtype),seed=seed),
        counts=d.counts,frame_hash=d.frame.receipt_hash,discovery_stream_hash=d.stream.receipt_hash,
        capture_receipt=d._capture_receipt_payload,capture_hash=d.capture_hash,
        proposal=dict(proposal_hash=proposal.proposal_hash,work=proposal.work,half_energy=proposal.half_energy,
            status=proposal.status,discovery_freeze_hash=proposal.discovery_freeze.receipt_hash),
        physical=physical_state)
    payload=_json_bytes(value)
    inspect_wire_payload(payload)
    acquisition.assert_integrity()
    return payload


def _fractions(value):
    if type(value) is not list or len(value)!=2:
        raise ValueError('operator checkpoint C sums require two strata')
    rows=[]
    for row in value:
        if type(row) is not list or len(row)!=3:
            raise ValueError('operator checkpoint C sums require three displacement lanes')
        result=[]
        for pair in row:
            if (type(pair) is not list or len(pair)!=2 or any(type(v) is not int for v in pair)
                    or pair[0]<0 or pair[1]<=0):
                raise ValueError('operator checkpoint C sums require nonnegative exact fractions')
            result.append(Fraction(*pair))
        rows.append(tuple(result))
    return tuple(rows)


def restore_operator_acquisition_payload(payload, *, expected_payload_sha256,
        expected_acquisition_hash, sources, binding, current_mask, initial_mask, trusted_local):
    """Rebuild closed evidence without D/C image reads or action reselection.

    `sources` must be freshly verified and already carry the checkpoint parent
    mask. The outer journal supplies both expected hashes and controller binding.
    Sampling receipts are reproduced from their retained deterministic KDF; no
    new observed contexts are acquired. New tensor leases bind this process.
    """
    if trusted_local is not True:
        raise PermissionError('operator reconstruction requires trusted local checkpoint provenance')
    value=inspect_operator_acquisition_payload(payload,expected_payload_sha256=expected_payload_sha256)
    if (type(sources) is not OpenedActualExecutionSources or type(binding) is not V1OptionACampaignBinding
            or type(current_mask) is not HardMask or type(initial_mask) is not HardMask):
        raise TypeError('operator reconstruction requires exact source/controller owners')
    target=value['target_resource']
    lineage=value['receipt']['committed_mask_lineage_hashes']
    if not lineage or initial_mask.mask_hash!=lineage[0]:
        raise ValueError('operator checkpoint initial mask differs from its committed lineage')
    if (initial_mask.route_axis_hash!=current_mask.route_axis_hash
            or initial_mask.route_count!=current_mask.route_count
            or any(c & ~p for p,c in zip(initial_mask.packed_bits,current_mask.packed_bits,strict=True))):
        raise ValueError('operator checkpoint current mask is outside its initial mask')
    remaining=binding.contract.maximum_removed_count-(initial_mask.retained_count-current_mask.retained_count)
    if target is not None and target!=remaining:
        raise ValueError('operator checkpoint resource target differs from the remaining campaign resource')
    sources.assert_live()
    content_manifest=sources.context_source.image_source.content_manifest_hash
    if content_manifest is None or content_manifest!=value['image_content_manifest_hash']:
        raise ValueError('operator checkpoint image content manifest differs or is missing')
    if value['circuit_biases']!=_circuit_bias_payload(sources):
        raise ValueError('operator checkpoint circuit biases differ from actual sources')
    if (value['acquisition_hash']!=expected_acquisition_hash
            or value['source_opening_hash']!=sources.opening_hash
            or value['model_checkpoint_binding_hash']!=sources.model_checkpoint_binding_hash
            or value['actual_source_binding_hash']!=sources.actual_source_binding_hash
            or value['receipt']['campaign_binding_hash']!=binding.binding_hash):
        raise ValueError('operator checkpoint source or controller binding differs')
    raw=value['parent_mask']
    parent=HardMask(raw['route_axis_hash'],raw['route_count'],bytes.fromhex(raw['packed_bits']))
    if (parent.mask_hash!=raw['mask_hash'] or parent!=current_mask
            or discovery_module._packed_parent(sources)!=parent):
        raise ValueError('operator checkpoint parent differs from current functional mask')
    budget=RawLogitMetricBudgetBinding.from_json(value['budget'])
    raw_policy=dict(value['policy']);schema=raw_policy.pop('schema')
    policy=SelectionPolicyConfig(**raw_policy)
    if schema!=policy.to_json()['schema']:
        raise ValueError('operator checkpoint policy schema differs')
    settings=dict(value['settings'])
    dtype={'torch.float32':torch.float32,'torch.float64':torch.float64}.get(settings.pop('factor_dtype'))
    if dtype is None:
        raise ValueError('operator checkpoint factor precision is unsupported')
    settings['factor_dtype']=dtype
    config_hash=discovery_module.discovery_config_hash(policy=policy,**settings)
    if (budget.config_hash!=config_hash or budget.population_identity_hash!=sources.context_source.source_owner.source_population_hash
            or budget.route_axis_hash!=sources.route_axis.route_axis_hash):
        raise ValueError('operator checkpoint budget/configuration differs from actual source')
    counts=tuple(value['counts'])
    if len(counts)!=2 or any(type(n) is not int or n<=0 for n in counts):
        raise ValueError('operator checkpoint requires two positive D counts')
    generation=value['receipt']['generation_index']
    owner=sources.context_source.source_owner
    population=dict(target_context_ids=owner.target_context_ids,complement_context_ids=owner.complement_context_ids)
    transform=discovery_module._discovery_transform_hash(sources,parent,budget.budget_hash,config_hash,target)
    frame=sources.fq_registration.open_causal_generation_frame(generation_index=generation,
        current_mask_hash=parent.mask_hash,frozen_transform_hash=transform,**population)
    stream=sources.fq_registration.open_discovery_stream(frame=frame,**population)
    if (frame.receipt_hash!=value['frame_hash'] or stream.receipt_hash!=value['discovery_stream_hash']
            or any(counts[i]>stream.sampling_capacity_stratum_counts[i] for i in range(2))):
        raise ValueError('operator checkpoint D frame or stream differs')
    contexts=(tuple(stream.target_assignment_receipt.generated_order[:counts[0]]),
              tuple(stream.complement_assignment_receipt.generated_order[:counts[1]]))
    capture=value['capture_receipt']
    expected_capture=dict(source_opening_hash=sources.opening_hash,generation_frame_hash=frame.receipt_hash,
        discovery_stream_hash=stream.receipt_hash,counts=list(counts),context_ids=[list(v) for v in contexts],
        metric_budget_hash=budget.budget_hash,target_resource=target)
    if (any(capture.get(k)!=v for k,v in expected_capture.items())
            or content_hash(capture,prefix='v1-operator-discovery-capture-')!=value['capture_hash']):
        raise ValueError('operator checkpoint capture receipt differs from its frozen source prefix')
    # No live object is accepted from the payload. Rebuild only the closed-D
    # representation whose factors were already released before checkpointing.
    d=object.__new__(discovery_module.OperatorDiscovery)
    d.sources=sources;d.metric_budget=budget;d.policy=policy;d.parent_mask=parent
    d._mask_lease=_tensor_lease(sources.net.mask,name='operator parent mask')
    d._budget_hash=budget.budget_hash;d._config_hash=config_hash
    d._settings=(settings['capture_batch'],settings['max_factor_bytes'],dtype,settings['seed'])
    d.target_resource=target;d._target_resource_binding=target
    d.frame=frame;d.stream=stream;d.counts=counts;d.contexts=contexts
    d._capture_receipt_payload=capture;d.capture_hash=value['capture_hash']
    d.factors=None;d._factor_leases=();d._factor_metadata=None
    d._consumed=True;d._factors_released=True;d._physical_replay_owner=None
    d._frozen_proposal_hash=value['proposal']['proposal_hash'];d._capture_binding=d._binding_coordinates()
    child_bits=value['child_packed_bits']
    action=(None if child_bits is None else V1PackedRemovalAction.from_resulting_mask(parent_mask=parent,
        resulting_mask=HardMask(parent.route_axis_hash,parent.route_count,bytes.fromhex(child_bits))))
    spec=value['proposal']
    freeze=sources.fq_registration.freeze_discovery_prefix(frame=frame,stream=stream,
        cumulative_stratum_counts=counts,discovery_evidence_state_hash=d.capture_hash,selected_shell_hash=spec['proposal_hash'])
    if freeze.receipt_hash!=spec['discovery_freeze_hash']:
        raise ValueError('operator checkpoint frozen D prefix differs')
    proposal=discovery_module.FrozenOperatorProposal(d,action,spec['proposal_hash'],freeze,
        spec['work'],spec['half_energy'],spec['status'])
    proposal.assert_live()
    physical=None
    saved=value['physical']
    if saved is not None:
        physical=physical_module.OperatorPhysicalReplay(proposal)
        if (physical.producer_hash!=saved['producer_hash'] or physical.stream.receipt_hash!=saved['stream_hash']
                or physical.raw_support.receipt_hash!=saved['support_hash']):
            raise ValueError('operator checkpoint physical C source/stream/support differs')
        physical._lower_sums=_fractions(saved['lower_sums']);physical._upper_sums=_fractions(saved['upper_sums'])
        physical._cursor=list(saved['counts']);physical._row_chain=saved['row_chain']
        physical._progress=physical._build_population_progress(tuple(physical._cursor),physical._lower_sums,
            physical._upper_sums,physical._row_chain)
        physical._population_binding=physical._population_coordinates()
        if physical.population_progress.receipt_hash!=saved['progress_hash']:
            raise ValueError('operator checkpoint C population state differs')
    raw_receipt=dict(value['receipt'])
    for key in ('ordered_candidate_ids','committed_mask_lineage_hashes','committed_mask_transition_receipt_hashes'):
        raw_receipt[key]=tuple(raw_receipt[key])
    receipt=admission.V1OperatorAcquisitionReceipt(**raw_receipt)
    if receipt.receipt_hash!=expected_acquisition_hash:
        raise ValueError('operator checkpoint acquisition receipt differs')
    candidate=(admission.V1OperatorCandidate(physical,physical.population_progress.receipt_hash,admission._TOKEN)
               if receipt.outcome=='candidate_ready' and physical is not None else None)
    acquisition=admission.V1OperatorAcquisition(binding,parent,candidate,receipt,admission._TOKEN,receipt.receipt_hash,
        _no_candidate_proposal=proposal if candidate is None else None,
        _no_candidate_physical=physical if candidate is None else None)
    if physical is None:
        d._operator_no_candidate_acquisition=acquisition
    else:
        physical._admission_owner=acquisition
    acquisition.assert_integrity()
    if _json_bytes(None if candidate is None else candidate.definition_payload())!=_json_bytes(value['candidate_definition']):
        raise ValueError('operator checkpoint candidate definition differs from reconstructed physics')
    return acquisition


def project_operator_active_candidate(active):
    from dataclasses import fields
    from .modern_v1_controller import V1ActiveCandidate
    if type(active) is not V1ActiveCandidate or type(active.proposal) is not admission.V1OperatorCandidate:
        return active
    active.__post_init__()
    values={f.name:getattr(active,f.name) for f in fields(active)}
    values['proposal']=active.proposal.definition_payload()
    return dict(schema=ACTIVE_SCHEMA,active=values)


def restore_operator_active_candidate(value,acquisition):
    from .modern_v1_controller import V1ActiveCandidate
    if type(value) is not dict:
        return value
    active=dict(validate_active_projection(value))
    if (type(acquisition) is not admission.V1OperatorAcquisition or acquisition.candidate is None
            or active['acquisition_receipt_hash']!=acquisition.receipt.receipt_hash
            or _json_bytes(active['proposal'])!=_json_bytes(acquisition.candidate.definition_payload())):
        raise ValueError('operator active checkpoint differs from the verified acquisition')
    active['proposal']=acquisition.candidate
    return V1ActiveCandidate(**active)
