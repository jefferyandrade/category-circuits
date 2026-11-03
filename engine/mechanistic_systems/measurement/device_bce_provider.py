"""Device moments at the native fixed-pair acquisition seam.

String draw identities and CPU data-loader metadata remain host objects.
Numerical evidence stays on the acquisition device until the decision receipt.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from functools import cached_property, lru_cache
from types import SimpleNamespace

import numpy as np
import torch

from mechanistic_systems.inference import device_bce as db
from mechanistic_systems.inference.sequential_bce import CalibrationImprovementProfile
from mechanistic_systems.contracts.bce_evidence import FixedCalibrationPairedBCEEvidence
from mechanistic_systems.contracts.sampled_evidence import CrossfitPanel, PeriodicFolds, sampled_path_mask_hash, PANEL_AXIS_CHAIN_KEY
from mechanistic_systems.measurement import evidence_plane_v6 as v6


class PrefixIds(Sequence):
    """Immutable-length view of an append-only logical draw list."""
    def __init__(self,rows):
        self.rows,self.size = rows,len(rows)
    def __len__(self):
        return self.size
    def __getitem__(self,index):
        if isinstance(index,slice):
            return tuple(self.rows[i] for i in range(*index.indices(self.size)))
        index = index+self.size if index < 0 else index
        if index < 0 or index >= self.size:
            raise IndexError(index)
        return self.rows[index]


class StratumNames(Sequence):
    """Immutable-length names view over a panel's integer stratum codes."""

    def __init__(self, names, codes):
        self.names, self.codes = tuple(names), codes

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self.names[int(value)] for value in self.codes[index])
        return self.names[int(self.codes[index])]


class SourceIds(Sequence):
    """Physical sources of a fixed logical prefix, expanded on first read."""

    def __init__(self, provider, context_ids):
        self.provider, self.context_ids, self._values = provider, context_ids, None

    def __len__(self):
        return len(self.context_ids)

    def __getitem__(self, index):
        if self._values is None:
            self._values = tuple(self.provider._source_context_ids(
                tuple(self.context_ids), role="proposal_adjudication"))
        return self._values[index]


class StratumRowWeights(Sequence):
    """The panel's exact stratum weight/count quotient as a lazy row view."""

    def __init__(self, names, codes, weights, counts):
        self.codes = codes
        self.by_stratum = np.asarray([float(weights[name]) for name in names], np.float64)
        self.by_stratum /= np.asarray([counts[name] for name in names], np.float64)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self.by_stratum[self.codes[index]].tolist())
        return float(self.by_stratum[int(self.codes[index])])

    def __array__(self, dtype=None, copy=None):
        values = self.by_stratum[self.codes]
        return np.asarray(values, dtype=dtype)


class ContextsDrawId:
    """Original state/context draw identity, computed on an explicit read."""

    def __init__(self, context_ids, state_hash):
        self.context_ids, self.state_hash, self._value = context_ids, state_hash, None

    def __str__(self):
        if self._value is None:
            digest = v6.IncrementalContextsDigest()
            digest.extend(tuple(self.context_ids))
            self._value = digest.digest(self.state_hash)
        return self._value

    __repr__ = __str__

    def __len__(self):
        return 64

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


class DeviceMomentView(Mapping):
    """A look's sufficient statistics for explicit host evidence consumers."""

    def __init__(self, names, counts, means, m2):
        self.names, self.counts = tuple(names), dict(counts)
        self.means, self.m2, self._values = means, m2, None
        self.positions = {name: index for index, name in enumerate(self.names)}

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, name):
        position = self.positions[name]
        if self._values is None:
            self._values = torch.cat((self.means, self.m2), dim=0).detach().to("cpu").numpy()
        return {"count": int(self.counts[name]), "mean": float(self._values[0, position]),
                "m2": float(self._values[1, position])}


class DeviceOracleMomentHistory:
    """Acquisition moments on the device, with explicit lazy host views.

    The provider's allocation history retains each packet's acquisition fit.
    Calibration reprojection updates the separate evidence history.  The
    original ``V6OracleMomentDict`` surface remains readable without putting
    its arrays on the device decision's path.
    """

    def __init__(self, names, counts, moments, generation):
        self.names, self.counts = tuple(names), counts.copy()
        self.column_count = moments[1].shape[0]
        self.generation, self.moments, self._values = generation, moments, None

    def _host_values(self):
        if self._values is None:
            self._values = torch.cat(self.moments[1:], dim=0).detach().to("cpu").numpy()
        return self._values

    @property
    def means(self):
        return self._host_values()[:self.column_count].T

    @property
    def m2(self):
        return self._host_values()[self.column_count:].T


def _host_panel_value(value):
    if isinstance(value, Mapping):
        return {key: _host_panel_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_host_panel_value(item) for item in value)
    if isinstance(value, ContextsDrawId):
        return str(value)
    return value


class LazyDevicePanel(CrossfitPanel):
    """Panel identities remain available without constructing per-row tuples."""

    def assert_nested_extension(self, prior):
        if (isinstance(prior, LazyDevicePanel)
                and isinstance(self.context_ids, PrefixIds) and isinstance(prior.context_ids, PrefixIds)
                and self.context_ids.rows is prior.context_ids.rows
                and self.fold_ids.fold_count == prior.fold_ids.fold_count):
            if self.stage_index <= prior.stage_index:
                raise ValueError("sampled panel stages must advance monotonically")
            if len(self.context_ids) < len(prior.context_ids):
                raise ValueError("sampled panels must retain contexts and fold assignments")
            # Both views refer to the provider's append-only logical list.
            # Periodic folds preserve every earlier position by construction.
            return
        super().assert_nested_extension(prior)

    def validate(self, *, fold_count):
        size = len(self.context_ids)
        if size < fold_count or not self.__dict__.get("_v6_unique_contexts_verified", False):
            raise ValueError("sampled panel requires unique contexts covering every fold")
        if (len(self.fold_ids) != size or self.fold_ids.fold_count != fold_count):
            raise ValueError("sampled panel fold IDs must cover the declared cross-fit axis")
        weights = self.sampling_weights
        if len(weights) != size or not np.all(np.isfinite(weights.by_stratum) & (weights.by_stratum > 0.0)):
            raise ValueError("sampled panel weights must be finite and positive")
        if self.stage_index < 0:
            raise ValueError("sampled panel stage index must be nonnegative")

    @cached_property
    def panel_hash(self):
        metadata = {key: _host_panel_value(value) for key, value in self.allocation_metadata.items()
                    if not key.startswith("device_") and key != "oracle_strata_names"}
        return CrossfitPanel(context_ids=tuple(self.context_ids), fold_ids=tuple(self.fold_ids),
            sampling_weights=tuple(self.sampling_weights), labels=tuple(self.labels),
            draw_id=str(self.draw_id), stage_index=self.stage_index, allocation_metadata=metadata).panel_hash


class LazyMaskHash:
    """``sampled_path_mask_hash`` of a decision's masks, computed when read.

    v6 decision records never read the evidence's mask digests (the frontier
    keeps the candidate's ``HardMask`` hash), so the digest leaves the
    decision's critical path.  ``len`` is the digest length the evidence
    contract checks; ``str`` computes it from the masks the decision used,
    which do not change while the decision runs.
    """

    __slots__ = ("_masks", "_digest")

    def __init__(self, masks):
        self._masks, self._digest = masks, None

    def __str__(self):
        if self._digest is None:
            self._digest, self._masks = sampled_path_mask_hash(self._masks), None
        return self._digest

    __repr__ = __str__

    def __len__(self):
        return 64

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


def _padded_moments(values, codes, *, stratum_count):
    """Two-pass moments; code -1 denotes padding with zero contribution."""
    return db.stratified_moments(values, codes, stratum_count)


@lru_cache(maxsize=None)
def _paired_function(stratum_count):
    def function(values, codes):
        return _padded_moments(values, codes, stratum_count=stratum_count)
    return db.CapturedTensorFunction(function)


_merge = db.CapturedTensorFunction(db.merge_moments)


def _paired(values, codes, stratum_count, *, lane):
    """Execute one logical packet through its worker's captured chunks."""
    if lane.stratum_count != stratum_count:
        raise ValueError("moment packet differs from its stratum lane")
    return lane._reduce_chunks(values, codes)


class MomentLane:
    """One worker's padded reduction capacity, reused for every logical look."""

    def __init__(self, device, stratum_count, packet_rows):
        self.device = torch.device(device)
        self.stratum_count, self.packet_rows = int(stratum_count), int(packet_rows)
        if self.stratum_count <= 0 or self.packet_rows <= 0:
            raise ValueError("device moment lanes require positive strata and packet rows")
        self.values = torch.empty((self.packet_rows, 1), dtype=torch.float64, device=self.device)
        self.codes = torch.empty(self.packet_rows, dtype=torch.int64, device=self.device)
        self.reduction = _paired_function(self.stratum_count)

    def warm(self):
        self.values.zero_()
        self.codes.fill_(-1)
        moments = self.reduction(self.values, self.codes)
        _merge(*moments, *moments)
        return self

    def reduce(self, values, codes):
        """Chunk arbitrary logical prefixes into the same captured shape."""

        return _paired(values, codes, self.stratum_count, lane=self)

    def _reduce_chunks(self, values, codes):

        values = values.reshape(-1, 1)
        if len(values) != len(codes) or values.device != self.values.device or codes.device != self.codes.device:
            raise ValueError("device moment rows and codes must share their lane and axis")
        result = None
        for start in range(0, max(len(values), 1), self.packet_rows):
            stop = min(start + self.packet_rows, len(values))
            size = stop - start
            # Padding may contain old values. Its -1 codes mask both the
            # first-pass sum and the second-pass squared deviations.
            self.codes.fill_(-1)
            self.codes[:size].copy_(codes[start:stop], non_blocking=True)
            self.values[:size].copy_(values[start:stop], non_blocking=True)
            current = self.reduction(self.values, self.codes)
            result = current if result is None else _merge(*result, *current)
        return result


_MOMENT_LANES = {}


def prewarm_moments(device, stratum_count, packet_rows):
    """Compile and capture the paired and merge graph at worker startup."""

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (str(device), int(stratum_count), int(packet_rows))
    lane = _MOMENT_LANES.get(key)
    if lane is None:
        lane = _MOMENT_LANES[key] = MomentLane(device, stratum_count, packet_rows).warm()
    return lane


def _moment_lane(provider, device, stratum_count):
    lane = getattr(provider, "_device_bce_moment_lane", None)
    if lane is None:
        if torch.device(device).type == "cuda":
            raise RuntimeError("device moment graphs were not warmed before the walk")
        rows = int(provider.runtime_execution_profile.v0_paired_oracle_batch_rows)
        lane = provider._device_bce_moment_lane = prewarm_moments(device, stratum_count, rows)
    if lane.stratum_count != stratum_count:
        raise ValueError("device moment lane differs from the declared stratum axis")
    return lane


def _transfers(owner, device):
    buffers = getattr(owner, "_device_bce_transfers", None)
    if buffers is None:
        buffers = owner._device_bce_transfers = db.HostTransferBuffer(device)
    return buffers


def finish_packet(provider, *, request, key, selected, evaluation_contexts,
                  output_device, masks, calibration_state_ids,
                  oracle_allocation_receipt, request_fused_acquisition_receipts,
                  started, **unused):
    """Provider's outcome without moments, rows or finite checks read to CPU."""
    from mechanistic_systems.contracts.measurement import MeasurementOutcome
    from mechanistic_systems.contracts.measurement_costs import RealizedCost
    names = tuple(provider._oracle_statistical_strata)
    codes = provider._v6_stratum_codes(evaluation_contexts)
    host_counts = np.bincount(codes, minlength=len(names)).astype(np.int64)
    values = output_device.detach().to(torch.float64)
    index = _transfers(provider, values.device).copy("stratum_codes", codes, dtype=torch.int64)
    n, mean, m2 = _moment_lane(provider, values.device, len(names)).reduce(values, index)
    packet_moments = (n, mean, m2)
    states = getattr(provider, "_device_bce_histories", None)
    if states is None:
        states = provider._device_bce_histories = {}
    prior = states.get(key)
    if prior is not None:
        n, mean, m2 = _merge(*prior.moments, n, mean, m2)
        allocation_moments = _merge(*prior.allocation_moments, *packet_moments)
        known_counts = prior.counts + host_counts
    else:
        allocation_moments = packet_moments
        known_counts = host_counts.copy()
    states[key] = SimpleNamespace(moments=(n, mean, m2), counts=known_counts,
                                  allocation_moments=allocation_moments,
                                  generation=calibration_state_ids, names=names)
    retained = v6.V6OracleMomentDict(DeviceOracleMomentHistory(
        names, known_counts, allocation_moments, calibration_state_ids))
    retained.evidence_history = DeviceOracleMomentHistory(
        names, known_counts, (n, mean, m2), calibration_state_ids)
    provider._oracle_moments[key] = retained
    from mechanistic_systems.measurement.step_observables import step_observables_enabled
    if step_observables_enabled() and values.shape[1] == 1:
        states[key].observation_packets = ([] if prior is None else getattr(prior,"observation_packets",[])) + [values[:,0].to(torch.float32)]
    history = provider._oracle_context_history.setdefault(key, {s: v6.DrawnCount() for s in names})
    for h in np.flatnonzero(host_counts):
        history[names[h]].n += int(host_counts[h])
    provider._oracle_logical_history[key] = list(evaluation_contexts)
    provider._oracle_calibration_state_ids[key] = calibration_state_ids
    provider._post_speculative_adjudication_extension(
        oracle_key=key, history=history, selected=selected,
        mask_hashes=tuple(mask.mask_hash for mask in masks), stratum_codes=codes)
    return MeasurementOutcome.lightweight(
        request_id=request.request_id, state_hash=request.state_hash,
        context_ids=evaluation_contexts, candidate_ids=request.candidate_ids,
        values=np.zeros((0, len(request.candidate_ids)), dtype=np.float64),
        calibration_state_id=calibration_state_ids[0],
        realized_cost=RealizedCost(wall_seconds=time.perf_counter()-started, forward_passes=1,
                                  calibration_fits=provider._calibration_fit_count, packing_degree=len(masks)),
        metadata={"kind": "hard_mask_oracle", "summary_only": True,
                  "device_bce_state": states[key], "device_bce_codes": codes,
                  "oracle_allocation_receipt": oracle_allocation_receipt,
                  "new_oracle_logical_draw_count": len(selected),
                  "returned_oracle_logical_draw_count": len(evaluation_contexts),
                  "oracle_summary_scalar_transfer_count": 0,
                  "oracle_row_transfer_count": 0})


def _profile(backend, fit, *, parent):
    frozen = backend.calibration_decision_estimand == "frozen_extraction_calibration_risk_v1"
    frozen = frozen or (parent and backend.calibration_decision_estimand == "accepted_state_frozen_candidate_refinable_v1")
    return CalibrationImprovementProfile(
        first_order_coefficient=0. if frozen else fit.first_order_excess_risk_coefficient,
        second_order_coefficient=0. if frozen else fit.second_order_excess_risk_coefficient,
        parameter_dimension=0 if frozen else fit.uncertainty_parameter_dimension)


def _device_profile(backend, fit, *, parent):
    frozen = backend.calibration_decision_estimand == "frozen_extraction_calibration_risk_v1"
    frozen = frozen or (parent and backend.calibration_decision_estimand == "accepted_state_frozen_candidate_refinable_v1")
    if frozen:
        return CalibrationImprovementProfile(parameter_dimension=0)
    values = getattr(fit, "device_uncertainty", None)
    if values is None:
        return _profile(backend, fit, parent=parent)
    return SimpleNamespace(first_order_coefficient=values["first_order_excess_risk_coefficient"],
        second_order_coefficient=values["second_order_excess_risk_coefficient"],
        parameter_dimension=fit.uncertainty_parameter_dimension)


def assemble_evidence(backend, prep, outcome, request):
    provider = backend.provider
    state = outcome.metadata.get("device_bce_state")
    states = getattr(backend, "_device_bce_states", None)
    if states is None:
        states = backend._device_bce_states = {}
    prior = states.get(prep.key)
    if state is None:
        if prior is None:
            raise ValueError("census relabelling lacks device evidence")
        state = prior
        codes = np.zeros(0, np.int64)
    else:
        codes = outcome.metadata["device_bce_codes"]
    parent_fit = provider.calibration_fits[prep.parent.mask_hash]
    candidate_fit = provider.calibration_fits[prep.candidate_mask.mask_hash]
    generation = (parent_fit.calibration_state_id, candidate_fit.calibration_state_id)
    if prior is not None and prior.generation != generation:
        # Re-score the existing panel under a refined fit, just as the host
        # reference does.  Counts and logical draw identities are unchanged.
        complete = (*tuple(prep.context_history), *tuple(outcome.context_ids))
        axis = backend._decision_row_axis(prep.key, complete)
        readouts = provider._readouts((prep.parent, prep.candidate_mask), complete,
            role="proposal_adjudication", source_context_ids_override=tuple(axis.source_context_ids))
        folds = (_transfers(provider, provider._device).copy("reprojection_folds", axis.fold_indices, dtype=torch.long)
                 if any(mask.mask_hash in provider._calibration_fold_fits for mask in (prep.parent, prep.candidate_mask)) else None)
        candidate_rows = provider._binary_bce_device_rows_for_mask(readouts[1],
            mask_hash=prep.candidate_mask.mask_hash, fit=candidate_fit,
            fold_index_per_row=folds, target_class_id=provider.target_class_id)
        parent_rows = provider._binary_bce_device_rows_for_mask(readouts[0],
            mask_hash=prep.parent.mask_hash, fit=parent_fit,
            fold_index_per_row=folds, target_class_id=provider.target_class_id)
        positions = {s:i for i,s in enumerate(state.names)}
        all_codes = np.asarray([positions[s] for s in axis.strata], np.int64)
        # Preserve the provider's loss-row subtraction dtype.  Its ordinary
        # acquisition and reference re-projection subtract float32 BCE rows
        # before the float64 moment reduction.
        increment = candidate_rows-parent_rows
        n, mean, m2 = _moment_lane(provider, increment.device, len(state.names)).reduce(increment,
            _transfers(provider, provider._device).copy("reprojection_codes", all_codes, dtype=torch.int64))
        state.moments = (n, mean, m2)
        state.generation = generation
        if hasattr(state,"observation_packets"):
            state.observation_packets = [increment.to(torch.float32)]
        provider._device_bce_histories[provider._oracle_key(prep.measurement_request)] = state
        provider._oracle_moments[provider._oracle_key(prep.measurement_request)].evidence_history = (
            DeviceOracleMomentHistory(state.names, state.counts, state.moments, generation))
    states[prep.key] = state
    backend._device_bce_last_key = prep.key
    backend._device_bce_last_provider_key = provider._oracle_key(prep.measurement_request)
    layouts = getattr(backend,"_device_bce_layouts",None)
    if layouts is None:
        layouts = backend._device_bce_layouts = {}
    layout = layouts.setdefault(prep.key,SimpleNamespace(codes=v6.GrowingIntArray(),seen=set()))
    before = len(layout.seen)
    layout.seen.update(outcome.context_ids)
    if len(layout.seen) != before+len(outcome.context_ids):
        raise ValueError("device BCE panel has repeated logical draw identities")
    layout.codes.extend(codes)
    prep.context_history.extend(outcome.context_ids)
    prep.stratum_history.extend(v6.names_for_codes(state.names, codes))
    backend._history_calibration_state_ids[prep.key] = generation
    backend._decision_images_consumed += len(outcome.context_ids)
    backend._context_state_evaluations += 2*len(outcome.context_ids)
    counts = dict(zip(state.names, state.counts.tolist()))
    if any(count < 2 for count in counts.values()):
        raise ValueError("device BCE panel lacks minimum stratum support")
    weights = provider.stratum_weights
    context_ids = PrefixIds(prep.context_history)
    census = backend._oracle_census_contract(state_hash=prep.state_hash,
        parent_mask_hash=prep.parent.mask_hash, candidate_id=prep.candidate.candidate_id)
    chain = (None if census is None else backend._panel_axis_chain(prep.key,
        stage_index=int(request.stage_index), context_count=len(context_ids), appended=tuple(outcome.context_ids)))
    populations = provider.oracle_context_ids_by_stratum
    panel = LazyDevicePanel(context_ids=context_ids,
        fold_ids=PeriodicFolds(len(context_ids),int(request.fold_count)),
        sampling_weights=StratumRowWeights(state.names, layout.codes.view, weights, counts),
        draw_id=ContextsDrawId(context_ids, prep.state_hash),
        stage_index=int(request.stage_index),
        allocation_metadata={"oracle_strata_names": state.names,
            "allocation_rule": ("target_plus_off_target_class_stratified_cost_aware_neyman"
                                if len(weights) > 2 else "binary_pooled_off_target_cost_aware_neyman"),
            "positive_population": len(populations["on_target"]),
            "negative_population": sum(len(values) for name, values in populations.items() if name != "on_target"),
            "oracle_strata": StratumNames(state.names, layout.codes.view),
            "oracle_source_context_ids": SourceIds(provider, context_ids),
            "oracle_stratum_weights": dict(weights), "delivered_stratum_counts": counts,
            "sampling_with_replacement": backend.sampling_with_replacement,
            "finite_population_correction_permitted": not backend.sampling_with_replacement,
            "oracle_context_axis_role": ("fresh_proposal_empirical_iid_stream" if backend.sampling_with_replacement
                else "fresh_proposal_sealed_without_replacement_class_stratified_stream" if provider.iid_design_mode
                else "training_loss_oracle_disjoint_from_calibration"),
            "calibration_context_axis_role": ("independent_mask_specific_empirical_iid_stream"
                if provider.calibration_sampling_with_replacement else "training_calibration_fit_disjoint_from_loss_oracle"),
            "oracle_allocation_receipt": outcome.metadata.get("oracle_allocation_receipt", {}),
            "native_outcome_integrity_hash": outcome.integrity_hash,
            **dict(census or {}), "oracle_census_terminal_reached": bool(prep.census_no_extension),
            **({PANEL_AXIS_CHAIN_KEY: chain} if chain is not None else {})})
    if hasattr(state,"observation_packets"):
        panel.allocation_metadata["device_observation_packets"] = state.observation_packets
    v6.mark_panel_v6(panel, layout.codes.view, state.names, unique_verified=True)
    identities = getattr(backend,"_device_bce_evidence_mask_hashes",None)
    if identities is None:
        identities = backend._device_bce_evidence_mask_hashes = {}
    if prep.key not in identities:
        identities[prep.key] = (LazyMaskHash(prep.committed_masks),LazyMaskHash(prep.candidate_masks))
    parent_hash,candidate_hash = identities[prep.key]
    device_profiles = (_device_profile(backend, candidate_fit, parent=False),
                       _device_profile(backend, parent_fit, parent=True))
    panel.allocation_metadata["device_calibration_profiles"] = device_profiles
    panel.allocation_metadata["device_allocation_moments"] = state.allocation_moments
    return FixedCalibrationPairedBCEEvidence(panel=panel,
        paired_bce_increment=np.zeros(0, np.float64),
        committed_mask_hash=parent_hash, candidate_mask_hash=candidate_hash,
        committed_calibration_state_id=generation[0], candidate_calibration_state_id=generation[1],
        stratified_moments=DeviceMomentView(state.names, counts, state.moments[1], state.moments[2]),
        committed_calibration_profile=(CalibrationImprovementProfile(parameter_dimension=device_profiles[1].parameter_dimension)
            if getattr(parent_fit, "device_uncertainty", None) is not None else device_profiles[1]),
        candidate_calibration_profile=(CalibrationImprovementProfile(parameter_dimension=device_profiles[0].parameter_dimension)
            if getattr(candidate_fit, "device_uncertainty", None) is not None else device_profiles[0]),
        calibration_contexts_disjoint=not provider.iid_design_mode,
        device_moments=(state.names, *state.moments))


def precision_options(backend, committed_masks, candidate_masks, actual, stats):
    """Static fit/count/cost facts; all uncertainty-dependent arithmetic is tensor math."""
    provider = backend.provider
    prepared = getattr(backend,"_device_bce_prepared_masks",None)
    if prepared is not None and prepared[0] is committed_masks and prepared[1] is candidate_masks:
        parent,candidate = prepared[2:]
    else:
        parent,candidate = backend._hard_mask(committed_masks),backend._hard_mask(candidate_masks)
    dense = provider.known_mask(backend._loss_span_certificate.dense.mask_hash)
    roles = ("candidate_calibration", "parent_calibration", "dense_calibration")
    masks = (candidate,parent,dense)
    allowed = (() if backend.calibration_decision_estimand == "frozen_extraction_calibration_risk_v1"
        else (roles[:1] if backend.calibration_decision_estimand == "accepted_state_frozen_candidate_refinable_v1" else roles))
    counts,targets,seconds,eligible = [0.0]*5,[1.0]*5,[1.0]*5,[False]*5
    for i,(role,mask) in enumerate(zip(roles,masks)):
        if role not in allowed:
            continue
        fit = provider._calibration_fits.get(mask.mask_hash)
        if fit is None:
            continue
        current = sum(fit.sampled_stratum_counts.values()) or len(fit.context_ids)
        if current <= 0:
            continue
        quantum = provider._execution_tuner(role="calibration_fit",lane_count=1).occupancy_alignment_quantum(
            candidate_count=1,throughput_fraction=provider.alignment_throughput_fraction)
        if quantum <= 1 and provider._device.type == "cuda":
            quantum = provider.runtime_execution_profile.calibration_batch_rows
        quantum = max(1,int(quantum))
        increment = max(quantum,((current+quantum-1)//quantum)*quantum)
        counts[i],targets[i] = current,current+increment
        seconds[i] = provider._predicted_calibration_action_seconds(role=role,mask=mask,increment=increment)
        eligible[i] = True
    quantum = provider._execution_tuner(role="v0_parent_candidate_oracle",lane_count=2).occupancy_alignment_quantum(
        candidate_count=2,throughput_fraction=provider.alignment_throughput_fraction)
    if quantum <= 1 and provider._device.type == "cuda":
        quantum = provider.runtime_execution_profile.v0_paired_oracle_batch_rows
    quantum = max(1,int(quantum))
    rate = provider._predicted_role_seconds(role="v0_parent_candidate_oracle",lane_count=2,context_count=quantum)/quantum
    counts[4],eligible[4] = actual,True
    certificate=backend.current_loss_span_certificate.dense
    counts[3],eligible[3]=certificate.context_count,True
    dense_rate=provider._predicted_role_seconds(role="v1_candidate_oracle",lane_count=1,context_count=1)
    backend._device_precision_facts = (roles,masks,targets,seconds)
    device = stats.device
    buffers = _transfers(backend, device)
    facts = buffers.copy("precision_float_facts", [[counts], [targets], [seconds]])
    n, target, cost = facts.unbind(0)
    increment = (stats[:,db.INDEX["next_target"]]-actual).clamp(min=1.0,max=float(quantum))
    target[:,4] = actual+increment
    cost[:,4] = increment*rate
    dense_increment=(stats[:,db.INDEX["next_target"]]-actual).clamp_min(1.0)
    target[:,3]=certificate.context_count+dense_increment
    cost[:,3]=dense_increment*dense_rate
    backend._device_precision_actual=actual
    return n,target,cost,buffers.copy("precision_eligible", [eligible], dtype=torch.bool)


def execute_precision(backend, role, committed_masks, candidate_masks, options, stats):
    """Execute the selected suffix; no statistical scalar is read to the host."""
    provider = backend.provider
    roles,masks,targets,seconds = backend._device_precision_facts
    if role == 3:
        from dataclasses import replace
        from mechanistic_systems.inference.sequential_bce import LossSpanCertificate
        prior=backend.current_loss_span_certificate.dense
        increment=max(1,backend._device_precision_oracle_target-backend._device_precision_actual)
        refined=provider.certify_baseline_risk(provider.known_mask(prior.mask_hash),
            cumulative_context_count=int(prior.context_count)+increment,
            confidence_level=prior.confidence_level,maximum_planned_looks=None,precision_only_iid=True)
        if backend.calibration_decision_estimand in {"frozen_extraction_calibration_risk_v1",
                "accepted_state_frozen_candidate_refinable_v1"}:
            refined=replace(refined,calibration_state_id=prior.calibration_state_id,
                calibration_improvement_bound=prior.calibration_improvement_bound,
                calibration_first_order_coefficient=prior.calibration_first_order_coefficient,
                calibration_second_order_coefficient=prior.calibration_second_order_coefficient,
                calibration_parameter_dimension=prior.calibration_parameter_dimension)
        backend._loss_span_certificate=LossSpanCertificate(dense=refined,fully_pruned=backend._loss_span_certificate.fully_pruned)
        if not hasattr(refined, "device_summary"):
            backend._loss_span_certificate.validate_positive()
        return {"schema":"joint_calibration_oracle_allocation_receipt_v1","selected_role":"dense_oracle",
            "executed_precision_action":True,"executed_calibration":False}
    name,mask = roles[role],masks[role]
    prior = provider._calibration_fits[mask.mask_hash]
    started = time.perf_counter()
    refined = provider.ensure_calibration(mask,minimum_cumulative_count=int(targets[role]),warm_start_calibration=prior)
    elapsed = time.perf_counter()-started
    current = sum(prior.sampled_stratum_counts.values()) or len(prior.context_ids)
    observations = provider._joint_precision_action_cost_observations.setdefault(name,[])
    observations.append((max(1,int(targets[role])-current),elapsed))
    del observations[:-64]
    if role == 2:
        from mechanistic_systems.inference.sequential_bce import LossSpanCertificate
        dense = provider.refresh_baseline_calibration_certificate(backend._loss_span_certificate.dense)
        backend._loss_span_certificate = LossSpanCertificate(dense=dense,fully_pruned=backend._loss_span_certificate.fully_pruned)
        if not hasattr(dense, "device_summary"):
            backend._loss_span_certificate.validate_positive()
    return {"schema":"joint_calibration_oracle_allocation_receipt_v1","selected_role":name,
        "executed_calibration":True,"executed_precision_action":True,
        "current_count":current,"target_count":int(targets[role]),"predicted_device_seconds":seconds[role],
        "prior_calibration_state_id":prior.calibration_state_id,"refined_calibration_state_id":refined.calibration_state_id,
        "elapsed_seconds":elapsed}


def finish_decision(backend):
    """Drop device moment and planning buffers after their single receipt read."""
    key = getattr(backend,"_device_bce_last_key",None)
    provider_key = getattr(backend,"_device_bce_last_provider_key",None)
    for name in ("_device_bce_states","_device_bce_layouts","_device_bce_evidence_mask_hashes","_device_bce_proposals"):
        getattr(backend,name,{}).pop(key,None)
    for name in ("_device_bce_histories","_device_bce_plans"):
        getattr(backend.provider,name,{}).pop(provider_key,None)
    backend.provider._device_bce_next_allocation = None
    if hasattr(backend,"_device_bce_prepared_masks"):
        del backend._device_bce_prepared_masks


def next_allocation(backend,evidence,statistics,cache):
    from mechanistic_systems.inference.device_allocation import compiled_next_counts
    provider=backend.provider
    names = evidence.device_moments[0]
    n,means,m2 = evidence.panel.allocation_metadata["device_allocation_moments"]
    if "allocation_geometry" not in cache:
        buffers=cache["transfers"]
        quantum=provider._oracle_adjudication_alignment_quantum(lane_count=2)
        cache["allocation_geometry"]=(
            buffers.copy("allocation_costs",[provider._oracle_stratum_costs[h] for h in names]),
            buffers.copy("allocation_minimum",[provider.oracle_minimum_per_stratum for h in names]),
            buffers.copy("allocation_quantum",[max(1,int(quantum))]))
    costs,minimum,quantum=cache["allocation_geometry"]
    return compiled_next_counts(n,means,m2,cache["weights"],cache["populations"],costs,minimum,
        statistics[:,db.INDEX["next_target"]],cache["maximum"],quantum,
        sealed=not backend.sampling_with_replacement)
