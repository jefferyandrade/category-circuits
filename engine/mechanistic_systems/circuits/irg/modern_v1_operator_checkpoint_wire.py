"""Data-only operator checkpoint identities, usable before live reconstruction.

This module imports no model, sampler or runtime classes. Physical support and
population bounds must still be replayed on verified sources by the live reader.
"""
import json
import math

from mechanistic_systems.contracts.content_address import content_hash

SCHEMA='v1_closed_operator_acquisition_checkpoint_v3'
ACTIVE_SCHEMA='v1_operator_active_candidate_checkpoint_v1'
ACTIVE_FIELDS=frozenset(('acquisition_receipt_hash','queue_index','proposal','threshold',
    'sampling_law','step_freeze','ledger','paired_bce_registration','paired_bce_state','schema'))


def canonical_json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def validate_candidate_definition(value):
    if type(value) is not dict or value.get('schema')!='v1_operator_native_candidate_definition_v1':
        raise ValueError('operator checkpoint candidate definition schema differs')
    for name,expected in dict(old_solver_ancestry=False,F_J_R_population_evidence_available=False,
            paired_bce_oracle_required=True,mask_commit_authority=False).items():
        if value.get(name) is not expected:
            raise ValueError('operator checkpoint candidate changed authority')
    candidate_id=content_hash(dict(proposal_hash=value['proposal_hash'],
        physical_progress_hash=value['physical_progress_hash']),prefix='v1-operator-candidate-')
    if value['candidate_id']!=candidate_id:
        raise ValueError('operator checkpoint candidate identity differs')
    if type(value['removed_route_count']) is not int or value['removed_route_count']<=0:
        raise ValueError('operator checkpoint candidate requires removed routes')
    for status,interval,radius in (
        ('conditional_direct_D_status','parent_relative_full_dose_rms_interval','parent_relative_raw_rms_radius'),
        ('conditional_cumulative_dense_D_status','cumulative_dense_relative_raw_rms_interval','cumulative_dense_relative_raw_rms_radius')):
        pair=value[interval];limit=value[radius]
        if (value[status]!='certified' or type(pair) not in (list,tuple) or len(pair)!=2
                or any(type(v) not in (float,int) or not math.isfinite(v) for v in (*pair,limit))
                or not 0<=pair[0]<=pair[1]<=limit):
            raise ValueError('operator checkpoint candidate lacks both certified physical bounds')
    return value


def inspect_wire_payload(payload):
    if type(payload) is not bytes:
        raise TypeError('operator checkpoint wire payload must be exact bytes')
    value=json.loads(payload)
    if type(value) is not dict or value.get('schema')!=SCHEMA or canonical_json(value)!=payload:
        raise ValueError('operator checkpoint canonical wire schema differs')
    receipt=value['receipt'];parent=value['parent_mask']
    target=value['target_resource']
    if target is not None and (type(target) is not int or target<=0):
        raise ValueError('operator checkpoint resource target must be positive or None')
    if value['capture_receipt'].get('target_resource')!=target:
        raise ValueError('operator checkpoint capture resource target differs')
    if type(receipt) is not dict or type(parent) is not dict:
        raise ValueError('operator checkpoint requires explicit receipt and mask')
    bits=bytes.fromhex(parent['packed_bits']);count=parent['route_count']
    if type(count) is not int or count<=0 or len(bits)!=(count+7)//8:
        raise ValueError('operator checkpoint packed parent size differs')
    if count%8 and bits[-1]&((1<<(8-count%8))-1):
        raise ValueError('operator checkpoint packed parent padding differs')
    if value['child_packed_bits'] is not None and target is not None:
        child=bytes.fromhex(value['child_packed_bits'])
        if len(child)!=len(bits) or any(c & ~p for p,c in zip(bits,child,strict=True)):
            raise ValueError('operator checkpoint child is not a parent submask')
        if sum((p^c).bit_count() for p,c in zip(bits,child,strict=True))>target:
            raise ValueError('operator checkpoint action exceeds resource target')
    parent_hash=content_hash(dict(route_axis_hash=parent['route_axis_hash'],route_count=count,packed_bits=bits),prefix='irg-mask-')
    acquisition_hash=content_hash(receipt,prefix='v1-operator-controller-acquisition-')
    if (parent_hash!=parent['mask_hash'] or parent_hash!=receipt['current_mask_hash']
            or acquisition_hash!=value['acquisition_hash']
            or receipt['proposal_hash']!=value['proposal']['proposal_hash']
            or content_hash(value['capture_receipt'],prefix='v1-operator-discovery-capture-')!=value['capture_hash']):
        raise ValueError('operator checkpoint source receipt identities do not replay')
    candidate=value['candidate_definition']
    physical=value['physical']
    if candidate is not None:
        validate_candidate_definition(candidate)
        child=bytes.fromhex(value['child_packed_bits'])
        if len(child)!=len(bits) or any(c & ~p for p,c in zip(bits,child,strict=True)):
            raise ValueError('operator checkpoint child is not a parent submask')
        child_hash=content_hash(dict(route_axis_hash=parent['route_axis_hash'],route_count=count,packed_bits=child),prefix='irg-mask-')
        removed=sum((p^c).bit_count() for p,c in zip(bits,child,strict=True))
        if (receipt['outcome']!='candidate_ready' or receipt['ordered_candidate_ids']!=[candidate['candidate_id']]
                or candidate['parent_mask_hash']!=parent_hash or candidate['candidate_mask_hash']!=child_hash
                or candidate['removed_route_count']!=removed or candidate['proposal_hash']!=receipt['proposal_hash']
                or candidate['source_opening_hash']!=value['source_opening_hash']
                or candidate['metric_budget_hash']!=value['budget']['budget_hash']
                or physical is None or candidate['physical_progress_hash']!=physical['progress_hash']
                or receipt['physical_progress_hash']!=physical['progress_hash']):
            raise ValueError('operator checkpoint candidate differs from acquisition evidence')
    elif receipt['ordered_candidate_ids'] or receipt['outcome'] not in (
            'no_binary_action','physical_refuted','physical_unresolved_at_census','dense_radius_unregistered'):
        raise ValueError('operator checkpoint empty outcome differs')
    return value


def validate_active_projection(value):
    if (type(value) is not dict or set(value)!= {'schema','active'}
            or value['schema']!=ACTIVE_SCHEMA or type(value['active']) is not dict
            or set(value['active'])!=ACTIVE_FIELDS):
        raise ValueError('operator checkpoint active projection layout differs')
    active=value['active']
    validate_candidate_definition(active['proposal'])
    if (active['schema']!='partition_free_v1_active_candidate_v1'
            or type(active['queue_index']) is not int or active['queue_index']!=0):
        raise ValueError('operator checkpoint active candidate schema or queue differs')
    return active
