"""Source-bound operator discovery and binary freeze on retained causal D/C streams.

This owner executes D capture and selection. It does not issue old shell/KKT
witnesses, certify physical C, run an oracle, or commit. The retained sampler
opens an independent C order only after the actual binary proposal is frozen.
"""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import torch

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from ..modern_v1_candidate_composition import V1PackedRemovalAction
from ..modern_v1_physical_metric import BALANCED_RMS
from . import masked_alexnet_paired_bce as leases
from .alexnet_factors import capture_factor_batches
from .genuine_issuer_bridge import OpenedActualExecutionSources
from .metric_freeze import RawLogitMetricBudgetBinding
from .operators import ChartOperators
from .selection_policy import SelectionPolicyConfig, choose_action
from .solvers import diagonal_shrunk_geometry


def discovery_config_hash(*, capture_batch: int, max_factor_bytes: int,
                          policy: SelectionPolicyConfig, factor_dtype: torch.dtype,
                          seed: int) -> str:
    if type(policy) is not SelectionPolicyConfig:
        raise TypeError("operator discovery requires an exact selection policy")
    policy.__post_init__()
    if any(type(v) is not int or v <= 0 for v in (capture_batch, max_factor_bytes)):
        raise ValueError("positive capture batch and factor-bank budget required")
    if factor_dtype not in (torch.float32, torch.float64) or type(seed) is not int:
        raise ValueError("registered factor precision and integer seed required")
    return content_hash(dict(schema="v1_operator_discovery_configuration_v1",
        capture_batch=capture_batch, max_factor_bytes=max_factor_bytes,
        policy=policy.to_json(), factor_dtype=str(factor_dtype), seed=seed,
        selection_mode="logit_work", physical_C_profile=BALANCED_RMS),
        prefix="v1-operator-discovery-configuration-")


def _packed_parent(sources):
    mask = sources.net._forward_mask(sources.net.mask, require_binary=True)
    return HardMask.from_bool_array(mask.detach().cpu().numpy().astype(bool),
                                    route_axis_hash=sources.route_axis.route_axis_hash)


def _discovery_transform_hash(sources, parent_mask, budget_hash, config_hash, target_resource=None):
    return content_hash(dict(schema="v1_operator_discovery_transform_v2",
        target_resource=target_resource,
        execution_sources=sources.opening_hash,parent_mask=parent_mask.mask_hash,
        image_content_manifest_hash=sources.context_source.image_source.content_manifest_hash,
        metric_budget=budget_hash,configuration=config_hash,
        operator_source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("alexnet_factors.py", "discovery.py", "operators.py", "solvers.py",
                         "selection.py", "selection_policy.py", "selection_refinement.py", "chart_safeguards.py")}),
        prefix="v1-operator-discovery-transform-")


def _factor_items(factors):
    items=[("logits",factors.logits)]
    items += [(f"{i}:{name}",getattr(layer,name)) for i,layer in enumerate(factors.layers)
              for name in ("X","a","G","Gt","D","W","mask")]
    return tuple(items)


@dataclass(frozen=True, slots=True)
class FrozenOperatorProposal:
    discovery: "OperatorDiscovery"
    packed_action: V1PackedRemovalAction | None
    proposal_hash: str
    discovery_freeze: object
    work: float
    half_energy: float
    status: str
    _certification_stream: object | None = field(default=None,init=False,repr=False,compare=False)

    def assert_live(self):
        self.discovery.assert_live()
        self.discovery_freeze.assert_integrity()
        if self.discovery_freeze.selected_shell_hash != self.proposal_hash:
            raise ValueError("operator proposal differs from retained D freeze")
        expected=self.discovery._proposal_identity(self.packed_action,self.work,self.half_energy,self.status)
        if expected != self.proposal_hash:
            raise ValueError("operator proposal content changed after freeze")
        if self.packed_action is not None:
            if self.discovery.target_resource is not None and self.packed_action.removed_count>self.discovery.target_resource:
                raise ValueError("frozen action exceeds its resource target")
            resolved=V1PackedRemovalAction.from_resulting_mask(
                parent_mask=self.discovery.parent_mask,resulting_mask=self.packed_action.resulting_mask)
            if resolved.action_hash != self.packed_action.action_hash:
                raise ValueError("packed child no longer matches frozen removal")

    def open_certification_stream(self):
        """Open retained C order; no C row, certificate or action authority."""
        self.assert_live()
        if self.packed_action is None:
            raise ValueError("empty operator proposal cannot open physical C")
        if self._certification_stream is not None:
            stream=self._certification_stream
            stream.assert_integrity()
            if (stream.discovery_prefix_freeze_hash!=self.discovery_freeze.receipt_hash
                    or stream.selected_shell_hash!=self.proposal_hash):
                raise ValueError("cached C stream differs from the frozen operator action")
            return stream
        owner=self.discovery
        stream=owner.sources.fq_registration.open_certification_stream(
            frame=owner.frame,discovery_stream=owner.stream,discovery_freeze=self.discovery_freeze,
            target_context_ids=owner.sources.context_source.source_owner.target_context_ids,
            complement_context_ids=owner.sources.context_source.source_owner.complement_context_ids)
        object.__setattr__(self,"_certification_stream",stream)
        return stream


class OperatorDiscovery:
    """One actual, bounded D capture; one irreversible proposal-freeze attempt.

    Tensor leases guard this process-local evidence. Portable checkpoints and
    source-custodied C replay have separate owners; discovery grants neither
    statistical acceptance nor commit authority.
    """
    def __init__(self, *, sources: OpenedActualExecutionSources,
                 metric_budget: RawLogitMetricBudgetBinding, generation: int,
                 counts: tuple[int,int], capture_batch: int, max_factor_bytes: int,
                 policy: SelectionPolicyConfig = SelectionPolicyConfig(),
                 factor_dtype: torch.dtype = torch.float64, seed: int = 0,
                 target_resource: int | None = None):
        if target_resource is not None and (type(target_resource) is not int or target_resource <= 0):
            raise ValueError("target_resource must be None or a positive int")
        if type(sources) is not OpenedActualExecutionSources or type(metric_budget) is not RawLogitMetricBudgetBinding:
            raise TypeError("operator discovery requires verified sources and exact frozen metric budget")
        sources.assert_live(); metric_budget.__post_init__()
        config_hash=discovery_config_hash(capture_batch=capture_batch,max_factor_bytes=max_factor_bytes,
                                         policy=policy,factor_dtype=factor_dtype,seed=seed)
        if (metric_budget.config_hash != config_hash
            or metric_budget.population_identity_hash != sources.context_source.source_owner.source_population_hash
            or metric_budget.route_axis_hash != sources.route_axis.route_axis_hash):
            raise ValueError("operator metric budget differs from actual source/configuration/route owners")
        if type(counts) is not tuple or len(counts)!=2 or any(type(v) is not int or v<=0 for v in counts):
            raise ValueError("operator discovery requires two positive registered stratum counts")
        if capture_batch > sources.context_source.max_batch_rows:
            raise ValueError("capture batch exceeds bounded source request limit")
        if not 0 <= metric_budget.target_class < sources.net.readout_bias.numel():
            raise ValueError("metric target class is outside the actual readout")
        self.sources=sources; self.metric_budget=metric_budget; self.policy=policy
        self.parent_mask=_packed_parent(sources)
        self._mask_lease=leases._tensor_lease(sources.net.mask,name="operator parent mask")
        self._budget_hash=metric_budget.budget_hash; self._config_hash=config_hash
        self._settings=(capture_batch,max_factor_bytes,factor_dtype,seed)
        self.target_resource=target_resource
        self._target_resource_binding=target_resource
        transform_hash=_discovery_transform_hash(sources,self.parent_mask,self._budget_hash,config_hash,target_resource)
        source_owner=sources.context_source.source_owner
        population=dict(target_context_ids=source_owner.target_context_ids,
                        complement_context_ids=source_owner.complement_context_ids)
        self.frame=sources.fq_registration.open_causal_generation_frame(
            generation_index=generation,current_mask_hash=self.parent_mask.mask_hash,
            frozen_transform_hash=transform_hash,**population)
        self.stream=sources.fq_registration.open_discovery_stream(frame=self.frame,**population)
        if any(counts[i]>self.stream.sampling_capacity_stratum_counts[i] for i in (0,1)):
            raise ValueError("D capture exceeds registered sampling capacity")
        self.counts=counts
        self.contexts=(tuple(self.stream.target_assignment_receipt.generated_order[:counts[0]]),
                       tuple(self.stream.complement_assignment_receipt.generated_order[:counts[1]]))
        batch_hashes=[]
        def batches():
            for stratum,contexts in zip(("target","complement"),self.contexts,strict=True):
                for offset in range(0,len(contexts),capture_batch):
                    self._assert_sources()
                    batch=sources.context_source(stratum,contexts[offset:offset+capture_batch])
                    batch_hashes.append(batch.source_batch_receipt_hash)
                    yield batch.images
                    del batch
                    self._assert_sources()
        self.factors=capture_factor_batches(sources.net,batches(),context_count=sum(counts),
            target_class=metric_budget.target_class,factor_dtype=factor_dtype,max_factor_bytes=max_factor_bytes)
        self._factor_leases=tuple(leases._tensor_lease(t,name=name) for name,t in _factor_items(self.factors))
        self._factor_metadata=(self.factors.target_class,self.factors.context_count,
                               tuple(l.geometry for l in self.factors.layers))
        self._capture_receipt_payload=dict(schema="v1_operator_actual_discovery_capture_v1",
            source_opening_hash=sources.opening_hash,generation_frame_hash=self.frame.receipt_hash,
            discovery_stream_hash=self.stream.receipt_hash,counts=counts,context_ids=self.contexts,
            actual_batch_receipts=tuple(batch_hashes),metric_budget_hash=self._budget_hash,
            target_resource=self.target_resource,
            captured_target_logits=self.factors.target_logits().detach().cpu().tolist(),
            scope="actual_raw_logit_D_capture_no_population_C_or_commit_authority")
        self.capture_hash=content_hash(self._capture_receipt_payload, prefix="v1-operator-discovery-capture-")
        self._consumed=False
        self._frozen_proposal_hash=None
        self._factors_released=False
        self._physical_replay_owner=None
        self._capture_binding=self._binding_coordinates()
        self.assert_live()

    def _assert_sources(self):
        if self.target_resource != self._target_resource_binding or type(self.target_resource) is not type(self._target_resource_binding):
            raise ValueError("operator resource target changed after D opened")
        self.sources.assert_live()
        leases._check_tensor_lease(self.sources.net.mask,self._mask_lease)
        if self.metric_budget.budget_hash != self._budget_hash:
            raise ValueError("metric budget changed after operator discovery opened")
        batch,limit,dtype,seed=self._settings
        if discovery_config_hash(capture_batch=batch,max_factor_bytes=limit,policy=self.policy,
                                 factor_dtype=dtype,seed=seed)!=self._config_hash:
            raise ValueError("operator selection configuration changed after D opened")

    def _binding_coordinates(self):
        return (self.sources.opening_hash,self.frame.receipt_hash,self.stream.receipt_hash,
                self.parent_mask.mask_hash,self.counts,self.contexts,self.capture_hash,
                self._budget_hash,self._config_hash,self.target_resource,self.sources.context_source.image_source.content_manifest_hash)

    def assert_live(self):
        self._assert_sources(); self.frame.assert_integrity(); self.stream.assert_integrity()
        if (content_hash(self._capture_receipt_payload,prefix="v1-operator-discovery-capture-")!=self.capture_hash
                or self._binding_coordinates()!=self._capture_binding):
            raise ValueError("operator discovery source/prefix identity changed after capture")
        if self._factors_released:
            if self.factors is not None or not self._consumed or self._frozen_proposal_hash is None:
                raise ValueError("invalid frozen D factor-release state")
            return
        if (self.factors.target_class,self.factors.context_count,tuple(l.geometry for l in self.factors.layers))!=self._factor_metadata:
            raise ValueError("operator factor metadata changed after actual capture")
        items=_factor_items(self.factors)
        if len(items)!=len(self._factor_leases):
            raise ValueError("operator factor axis changed")
        for (name,tensor),lease in zip(items,self._factor_leases,strict=True):
            if name!=lease.name: raise ValueError("operator factor order changed")
            leases._check_tensor_lease(tensor,lease)

    def release_factors(self, proposal: FrozenOperatorProposal) -> None:
        """Release closed D payloads after the exact proposal has been verified.

        Capture receipts and the frozen action survive. This permits C replay
        without simultaneously retaining the complete D and C factor banks.
        Further proposal attempts are already forbidden.
        """
        if type(proposal) is not FrozenOperatorProposal or proposal.discovery is not self:
            raise TypeError("D factor release requires this owner's exact frozen proposal")
        proposal.assert_live()
        if proposal.proposal_hash!=self._frozen_proposal_hash:
            raise ValueError("D factor release names another frozen action")
        self.factors=None
        self._factors_released=True
        self.assert_live()

    def _proposal_identity(self, action, work, half_energy, status):
        return content_hash(dict(schema="v1_frozen_operator_binary_proposal_v1",
            capture_hash=self.capture_hash,parent_mask_hash=self.parent_mask.mask_hash,
            action_hash=None if action is None else action.action_hash,
            metric_budget_hash=self._budget_hash,physical_C_profile=BALANCED_RMS,
            target_resource=self.target_resource,
            observed_work=work,observed_half_energy=half_energy,selection_status=status,
            old_shell_or_KKT_witness=False,population_C_certified=False,mask_commit_authority=False),
            prefix="v1-frozen-operator-proposal-")

    def propose_and_freeze(self) -> FrozenOperatorProposal:
        self.assert_live()
        if self._consumed: raise ValueError("operator D capture already consumed a proposal attempt")
        self._consumed=True
        count=sum(self.counts); device=self.factors.device; dtype=self.factors.dtype
        w=torch.cat([torch.full((n,),.5/n,device=device,dtype=dtype) for n in self.counts])
        ops=ChartOperators.build(self.factors,self.sources.net.axis,w)
        geom=diagonal_shrunk_geometry(ops,weights=w,seed=self._settings[3])
        chart=dict(ops=ops,geom=geom,d=ops.singleton_price_magnitude(),g=torch.zeros_like(w),
                   h=torch.zeros_like(w),w=w,sel=torch.ones(count,device=device,dtype=torch.bool),
                   labels=torch.cat([torch.ones(self.counts[0],device=device),torch.zeros(self.counts[1],device=device)]),
                   eligible=ops.eligible())
        selected=choose_action(chart,self.metric_budget.response_half_energy_radius,mode="logit_work",config=self.policy,
                               target_resource=self.target_resource)
        self.assert_live()
        z=selected["z"]
        if z.shape!=(ops.route_count,) or not bool(((z==0)|(z==1)).all()):
            raise ValueError("operator selector returned a nonbinary action")
        indices=torch.nonzero(z,as_tuple=False).flatten()
        if len(indices)!=selected["resource"]:
            raise ValueError("operator selector resource differs from exact binary action")
        if self.target_resource is not None and len(indices)>self.target_resource:
            raise ValueError("operator selector exceeded the registered remaining resource")
        action=(None if not len(indices) else V1PackedRemovalAction.from_removed_indices(
            parent_mask=self.parent_mask,removed_indices=indices.detach().cpu().numpy()))
        work=float(selected["work"]); half_energy=float(selected["half_energy"]); status=str(selected["stage"])
        identity=self._proposal_identity(action,work,half_energy,status)
        freeze=self.sources.fq_registration.freeze_discovery_prefix(
            frame=self.frame,stream=self.stream,cumulative_stratum_counts=self.counts,
            discovery_evidence_state_hash=self.capture_hash,selected_shell_hash=identity)
        self._frozen_proposal_hash=identity
        result=FrozenOperatorProposal(self,action,identity,freeze,work,half_energy,status)
        result.assert_live()
        return result
