"""Exact, observation-only materialization of the accepted modern-v0 path.

The continuous walker records statistical decisions separately from the
``ActionCandidate`` and ``HardMask`` registries owned by the native provider.
This module joins those surfaces only through their canonical identities.  It
never infers an action from density, list position, or a rejected proposal.

The resulting object is deliberately inert: it is an immutable receipt of
already-completed commits and grants no authority to commit, launch, promote,
validate, publish, or make a scientific claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import ActionCandidate, HardMask
from mechanistic_systems.contracts.sampled_evidence import sampled_path_mask_hash
from mechanistic_systems.inference.sequential_bce import (
    sequential_bce_decision_receipt_content_hash,
)
from mechanistic_systems.circuits.v6_records import (
    V6_LEDGER_SCHEMA,
    decision_key,
    decision_record_hash,
)

ACCEPTED_V0_TRAJECTORY_SCHEMA = "accepted_v0_trajectory_v1"
ACCEPTED_V0_TRAJECTORY_ENTRY_SCHEMA = "accepted_v0_trajectory_entry_v1"
ACCEPTED_V0_TRAJECTORY_RECEIPT_SCHEMA = "accepted_v0_trajectory_receipt_v1"

_PROVIDER_STATE_PREFIX = "irg-validation-bce-state-"
_DECISION_RECEIPT_HASH_PREFIX = "accepted-v0-decision-receipt-snapshot-"
_ENTRY_HASH_PREFIX = "accepted-v0-trajectory-entry-"
_RECEIPT_HASH_PREFIX = "accepted-v0-trajectory-receipt-"


@dataclass(frozen=True, slots=True)
class AcceptedV0TrajectoryAuthority:
    """Fail-closed declaration that this historical view grants no authority."""

    claim_bearing: bool = False
    campaign_launch: bool = False
    pruning_commit: bool = False
    promotion: bool = False
    publication: bool = False
    validation: bool = False

    def __post_init__(self) -> None:
        switches = (
            self.claim_bearing,
            self.campaign_launch,
            self.pruning_commit,
            self.promotion,
            self.publication,
            self.validation,
        )
        if any(type(value) is not bool for value in switches):
            raise TypeError("accepted-v0 authority switches must be exact booleans")
        if any(switches):
            raise ValueError("an accepted-v0 trajectory cannot grant authority")

    def to_dict(self) -> dict[str, bool]:
        return {
            "claim_bearing": self.claim_bearing,
            "campaign_launch": self.campaign_launch,
            "pruning_commit": self.pruning_commit,
            "promotion": self.promotion,
            "publication": self.publication,
            "validation": self.validation,
        }


def _hard_mask_hash_payload(mask: HardMask) -> Any:
    """The canonical content of ``mask`` for ``content_hash``.

    ``content_hash`` canonicalizes a HardMask dataclass as its three fields,
    with ``packed_bits`` as ``{"bytes_sha256", "length"}``.  A v6 HardMask
    carries the SHA-256 of exactly those packed bytes from when it was built,
    so the identical payload is formed without rehashing 7 MB.
    Any other HardMask is returned unchanged.
    """

    digest = getattr(mask, "__dict__", {}).get("_v6_packed_sha256")
    if not digest:
        return mask
    return {
        "route_axis_hash": mask.route_axis_hash,
        "route_count": mask.route_count,
        "packed_bits": {"bytes_sha256": str(digest), "length": len(mask.packed_bits)},
    }


@dataclass(frozen=True, slots=True)
class AcceptedV0TrajectoryEntry:
    """One exact accepted commit and its complete immutable mask transition."""

    trajectory_index: int
    run_index: int
    proposal_index: int
    query_id: str
    layer_id: str
    partition_id: str
    candidate_id: str
    parent_mask_hash: str
    resulting_mask_hash: str
    provider_state_hash: str
    parent_mask: HardMask
    resulting_mask: HardMask
    decision_proposal_id: str | None
    decision_receipt_hash: str | None
    decision_candidate_evidence_hash: str | None
    authority: AcceptedV0TrajectoryAuthority = field(default_factory=AcceptedV0TrajectoryAuthority)
    schema_name: str = ACCEPTED_V0_TRAJECTORY_ENTRY_SCHEMA
    entry_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_name != ACCEPTED_V0_TRAJECTORY_ENTRY_SCHEMA:
            raise ValueError("accepted-v0 entry schema identity is not canonical")
        if (
            type(self.trajectory_index) is not int
            or type(self.run_index) is not int
            or type(self.proposal_index) is not int
            or self.trajectory_index < 0
            or self.run_index < 0
            or self.proposal_index < 0
            or any(
                not isinstance(value, str) or not value
                for value in (
                    self.query_id,
                    self.layer_id,
                    self.partition_id,
                    self.candidate_id,
                    self.parent_mask_hash,
                    self.resulting_mask_hash,
                    self.provider_state_hash,
                )
            )
        ):
            raise ValueError("accepted-v0 trajectory entry has invalid identity")
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in (
                self.decision_proposal_id,
                self.decision_receipt_hash,
                self.decision_candidate_evidence_hash,
            )
        ):
            raise ValueError("accepted-v0 decision receipt identities are invalid")
        if not isinstance(self.parent_mask, HardMask) or not isinstance(
            self.resulting_mask, HardMask
        ):
            raise TypeError("accepted-v0 entries require exact HardMask objects")
        if not isinstance(self.authority, AcceptedV0TrajectoryAuthority):
            raise TypeError("accepted-v0 entry authority is not typed")
        if (
            type(self.parent_mask.packed_bits) is not bytes
            or type(self.resulting_mask.packed_bits) is not bytes
        ):
            raise TypeError("accepted-v0 HardMask storage must be immutable bytes")
        if self.parent_mask.mask_hash != self.parent_mask_hash:
            raise ValueError("entry parent HardMask differs from its bound hash")
        if self.resulting_mask.mask_hash != self.resulting_mask_hash:
            raise ValueError("entry resulting HardMask differs from its bound hash")
        self.parent_mask.assert_same_axis(self.resulting_mask)
        if (self.decision_proposal_id is None) != (self.decision_receipt_hash is None):
            raise ValueError("decision proposal and receipt identities must co-occur")
        if (self.decision_proposal_id is None) != (self.decision_candidate_evidence_hash is None):
            raise ValueError("decision candidate evidence identity must co-occur")
        object.__setattr__(
            self,
            "entry_hash",
            content_hash(self._hash_payload(), prefix=_ENTRY_HASH_PREFIX),
        )

    def _hash_payload(self) -> Mapping[str, Any]:
        return {
            "schema_name": self.schema_name,
            "trajectory_index": int(self.trajectory_index),
            "run_index": int(self.run_index),
            "proposal_index": int(self.proposal_index),
            "query_id": self.query_id,
            "layer_id": self.layer_id,
            "partition_id": self.partition_id,
            "candidate_id": self.candidate_id,
            "parent_mask_hash": self.parent_mask_hash,
            "resulting_mask_hash": self.resulting_mask_hash,
            "provider_state_hash": self.provider_state_hash,
            "parent_mask": _hard_mask_hash_payload(self.parent_mask),
            "resulting_mask": _hard_mask_hash_payload(self.resulting_mask),
            "decision_proposal_id": self.decision_proposal_id,
            "decision_receipt_hash": self.decision_receipt_hash,
            "decision_candidate_evidence_hash": (self.decision_candidate_evidence_hash),
            "authority": self.authority,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **dict(self._hash_payload()),
            "parent_mask": self.parent_mask.to_dict(),
            "resulting_mask": self.resulting_mask.to_dict(),
            "entry_hash": self.entry_hash,
        }
        payload["authority"] = self.authority.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class AcceptedV0TrajectoryReceipt:
    """Content-addressed proof of inclusion and non-commit decision counts."""

    adapter_id: str
    query_id: str
    root_mask_hash: str
    terminal_mask_hash: str
    run_count: int
    total_decision_count: int
    accepted_count: int
    rejected_count: int
    unresolved_count: int
    run_manifest_hash: str
    decision_manifest_hash: str
    nonaccepted_manifest_hash: str
    entry_hashes: tuple[str, ...]
    authority: AcceptedV0TrajectoryAuthority = field(default_factory=AcceptedV0TrajectoryAuthority)
    schema_name: str = ACCEPTED_V0_TRAJECTORY_RECEIPT_SCHEMA
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_name != ACCEPTED_V0_TRAJECTORY_RECEIPT_SCHEMA:
            raise ValueError("accepted-v0 receipt schema identity is not canonical")
        counts = (
            self.run_count,
            self.total_decision_count,
            self.accepted_count,
            self.rejected_count,
            self.unresolved_count,
        )
        if (
            any(type(value) is not int for value in counts)
            or any(value < 0 for value in counts)
            or any(
                not isinstance(value, str) or not value
                for value in (
                    self.adapter_id,
                    self.query_id,
                    self.root_mask_hash,
                    self.terminal_mask_hash,
                    self.run_manifest_hash,
                    self.decision_manifest_hash,
                    self.nonaccepted_manifest_hash,
                )
            )
        ):
            raise ValueError("accepted-v0 trajectory receipt is incomplete")
        if self.total_decision_count != (
            self.accepted_count + self.rejected_count + self.unresolved_count
        ):
            raise ValueError("accepted-v0 decision counts do not partition the walk")
        if self.accepted_count != len(self.entry_hashes):
            raise ValueError("accepted count differs from the entry manifest")
        if type(self.entry_hashes) is not tuple or any(
            not isinstance(value, str) for value in self.entry_hashes
        ):
            raise TypeError("accepted-v0 entry manifest is not deeply immutable")
        if not isinstance(self.authority, AcceptedV0TrajectoryAuthority):
            raise TypeError("accepted-v0 receipt authority is not typed")
        if len(set(self.entry_hashes)) != len(self.entry_hashes):
            raise ValueError("accepted-v0 entry hashes must be unique")
        if any(not value for value in self.entry_hashes):
            raise ValueError("accepted-v0 entry manifest contains an empty hash")
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(self._hash_payload(), prefix=_RECEIPT_HASH_PREFIX),
        )

    def _hash_payload(self) -> Mapping[str, Any]:
        return {
            "schema_name": self.schema_name,
            "adapter_id": self.adapter_id,
            "query_id": self.query_id,
            "root_mask_hash": self.root_mask_hash,
            "terminal_mask_hash": self.terminal_mask_hash,
            "run_count": int(self.run_count),
            "total_decision_count": int(self.total_decision_count),
            "accepted_count": int(self.accepted_count),
            "rejected_count": int(self.rejected_count),
            "unresolved_count": int(self.unresolved_count),
            "run_manifest_hash": self.run_manifest_hash,
            "decision_manifest_hash": self.decision_manifest_hash,
            "nonaccepted_manifest_hash": self.nonaccepted_manifest_hash,
            "entry_hashes": self.entry_hashes,
            "authority": self.authority,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {**dict(self._hash_payload()), "receipt_hash": self.receipt_hash}
        payload["authority"] = self.authority.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class AcceptedV0Trajectory:
    """Immutable accepted path plus its exact root, terminal, and receipt."""

    root_mask: HardMask
    terminal_mask: HardMask
    entries: tuple[AcceptedV0TrajectoryEntry, ...]
    receipt: AcceptedV0TrajectoryReceipt
    schema_name: str = ACCEPTED_V0_TRAJECTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_name != ACCEPTED_V0_TRAJECTORY_SCHEMA:
            raise ValueError("accepted-v0 trajectory schema identity is not canonical")
        if not isinstance(self.root_mask, HardMask) or not isinstance(self.terminal_mask, HardMask):
            raise TypeError("accepted-v0 trajectory endpoints must be HardMask objects")
        if type(self.entries) is not tuple or any(
            not isinstance(entry, AcceptedV0TrajectoryEntry) for entry in self.entries
        ):
            raise TypeError("accepted-v0 trajectory entries are not deeply immutable")
        if not isinstance(self.receipt, AcceptedV0TrajectoryReceipt):
            raise TypeError("accepted-v0 trajectory receipt is not typed")
        if (
            type(self.root_mask.packed_bits) is not bytes
            or type(self.terminal_mask.packed_bits) is not bytes
        ):
            raise TypeError("accepted-v0 endpoint storage must be immutable bytes")
        self.root_mask.assert_same_axis(self.terminal_mask)
        if self.root_mask.mask_hash != self.receipt.root_mask_hash:
            raise ValueError("trajectory root differs from its receipt")
        if self.terminal_mask.mask_hash != self.receipt.terminal_mask_hash:
            raise ValueError("trajectory terminal differs from its receipt")
        if tuple(entry.entry_hash for entry in self.entries) != (self.receipt.entry_hashes):
            raise ValueError("trajectory entries differ from their receipt manifest")
        current = self.root_mask
        for index, entry in enumerate(self.entries):
            if entry.trajectory_index != index:
                raise ValueError("accepted-v0 trajectory indices are not contiguous")
            if entry.parent_mask != current:
                raise ValueError("accepted-v0 trajectory parent chain is broken")
            current = entry.resulting_mask
        if current != self.terminal_mask:
            raise ValueError("accepted-v0 trajectory does not end at the terminal mask")

    @property
    def content_hash(self) -> str:
        return self.receipt.receipt_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "root_mask": self.root_mask.to_dict(),
            "terminal_mask": self.terminal_mask.to_dict(),
            "entries": tuple(entry.to_dict() for entry in self.entries),
            "receipt": self.receipt.to_dict(),
        }


def _registry_mask(
    registry: Mapping[str, HardMask],
    mask_hash: str,
    *,
    role: str,
) -> HardMask:
    if not isinstance(mask_hash, str) or not mask_hash:
        raise TypeError(f"{role} mask hash is not an exact string identity")
    try:
        mask = registry[mask_hash]
    except KeyError as exc:
        raise ValueError(f"{role} HardMask is absent from the v0 mask registry") from exc
    if not isinstance(mask, HardMask):
        raise TypeError(f"{role} mask-registry value is not a HardMask")
    if type(mask.packed_bits) is not bytes:
        raise TypeError(f"{role} HardMask storage is not immutable bytes")
    if mask.mask_hash != mask_hash:
        raise ValueError(f"{role} mask-registry key differs from HardMask content")
    return mask


def _candidate_registry_index(
    registry: Mapping[str, ActionCandidate],
) -> Mapping[tuple[str, str, str, str, int], tuple[ActionCandidate, ...]]:
    mutable: dict[tuple[str, str, str, str, int], list[ActionCandidate]] = {}
    for registry_id, candidate in registry.items():
        if not isinstance(candidate, ActionCandidate):
            raise TypeError("v0 candidate registry contains a non-ActionCandidate")
        if not isinstance(registry_id, str) or registry_id != candidate.candidate_id:
            raise ValueError("v0 candidate-registry key differs from candidate identity")
        if candidate.compute_candidate_id() != candidate.candidate_id:
            raise ValueError("v0 candidate content changed after registration")
        if candidate.generator_version != "v0":
            raise ValueError("accepted-v0 registry contains a non-v0 candidate")
        parameters = candidate.generator_parameters
        required = (
            "adapter",
            "query_id",
            "layer_id",
            "partition_id",
            "proposal_index",
        )
        if any(name not in parameters for name in required):
            raise ValueError("v0 candidate lacks a canonical generator binding")
        if any(
            not isinstance(parameters[name], str)
            for name in ("adapter", "query_id", "layer_id", "partition_id")
        ):
            raise TypeError("v0 candidate string generator bindings are not exact")
        proposal_index = parameters["proposal_index"]
        if type(proposal_index) is not int or proposal_index < 0:
            raise TypeError("v0 candidate proposal_index is not an exact nonnegative int")
        key = (
            parameters["adapter"],
            parameters["query_id"],
            parameters["layer_id"],
            parameters["partition_id"],
            proposal_index,
        )
        mutable.setdefault(key, []).append(candidate)
    return {key: tuple(values) for key, values in mutable.items()}


def _run_receipts(run: Any) -> tuple[Mapping[str, Any], ...]:
    ledger = getattr(getattr(run, "result", None), "ledger_receipt", None)
    if not isinstance(ledger, Mapping):
        raise TypeError("completed v0 run lacks a decision ledger mapping")
    raw = ledger.get("sequential_bce_decision_receipts", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("v0 sequential decision receipts are not a sequence")
    receipts: list[Mapping[str, Any]] = []
    proposal_ids: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("v0 sequential decision receipt is not a mapping")
        proposal_id = item.get("proposal_id")
        if not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError("v0 sequential decision receipt lacks a proposal_id")
        if proposal_id in proposal_ids:
            raise ValueError("v0 sequential decision receipts repeat a proposal_id")
        proposal_ids.add(proposal_id)
        receipts.append(item)
    return tuple(receipts)


def _v6_run_log(run: Any) -> Mapping[str, Any] | None:
    """The closed v6 walk log of a run whose ledger is v6, else None."""

    ledger = getattr(getattr(run, "result", None), "ledger_receipt", None)
    if not isinstance(ledger, Mapping) or ledger.get("schema_name") != V6_LEDGER_SCHEMA:
        return None
    handle = ledger.get("v6_walk_log")
    if handle is None or not callable(getattr(handle, "result", None)):
        raise TypeError("v6 decision ledger lacks its walk log")
    return handle.result()


def _validate_receipt_content_hash(receipt: Mapping[str, Any]) -> None:
    claimed = receipt.get("content_hash")
    if claimed is None:
        return
    if not isinstance(claimed, str) or not claimed:
        raise ValueError("v0 decision receipt content_hash is invalid")
    payload = dict(receipt)
    payload.pop("content_hash", None)
    expected = sequential_bce_decision_receipt_content_hash(payload)
    if claimed != expected:
        raise ValueError("v0 decision receipt content hash mismatch")


def _sampled_masks_from_hard_mask(
    mask: HardMask,
    *,
    sampled_template: Mapping[str, Any],
    hard_mask_to_layers: Callable[[HardMask], Mapping[str, Any]] | None,
    role: str,
) -> dict[str, np.ndarray]:
    """Project one complete HardMask onto a run's exact sampled-mask axis."""

    if not isinstance(mask, HardMask):
        raise TypeError(f"{role} is not a HardMask")
    if hard_mask_to_layers is not None:
        resolved = hard_mask_to_layers(mask)
        if not isinstance(resolved, Mapping):
            raise TypeError("hard-mask layer projector must return a mapping")
        if set(resolved) != set(sampled_template):
            raise ValueError(f"{role} differs from the sampled-mask layer axis")
        projected = {}
        for layer_id, raw_template in sampled_template.items():
            template = np.asarray(raw_template, dtype=bool)
            value = np.asarray(resolved[layer_id], dtype=bool)
            if value.shape != template.shape:
                raise ValueError(f"{role} differs from the sampled-mask shape axis")
            projected[str(layer_id)] = value
        return projected
    flat = mask.to_bool_array()
    cursor = 0
    projected: dict[str, np.ndarray] = {}
    for layer_id, raw_mask in sampled_template.items():
        if not isinstance(layer_id, str) or not layer_id:
            raise ValueError("completed v0 run has an invalid sampled-mask layer identity")
        sampled_mask = np.asarray(raw_mask, dtype=bool)
        if sampled_mask.size <= 0:
            raise ValueError("completed v0 run contains an empty sampled mask")
        stop = cursor + int(sampled_mask.size)
        if stop > mask.route_count:
            raise ValueError(f"{role} differs from the sampled-mask route axis")
        projected[layer_id] = flat[cursor:stop].reshape(sampled_mask.shape)
        cursor = stop
    if cursor != mask.route_count:
        raise ValueError(f"{role} differs from the sampled-mask route axis")
    return projected


def _accepted_decision_receipt(
    *,
    receipts: Sequence[Mapping[str, Any]],
    query_id: str,
    layer_id: str,
    partition_id: str,
    proposal_index: int,
    candidate: ActionCandidate,
    provider_state_hash: str,
    expected_candidate_evidence_hash: str,
) -> tuple[str | None, str | None, str | None]:
    prefix = f"{query_id}:{layer_id}:{partition_id}:{int(proposal_index)}:"
    matches = tuple(receipt for receipt in receipts if receipt["proposal_id"].startswith(prefix))
    if len(matches) > 1:
        raise ValueError("accepted v0 decision has ambiguous exact receipts")
    if receipts and not matches:
        raise ValueError("supplied v0 decision receipts omit an accepted proposal")
    if not matches:
        return None, None, None
    receipt = matches[0]
    proposal_id = str(receipt["proposal_id"])
    candidate_evidence_hash = proposal_id[len(prefix) :]
    if not candidate_evidence_hash:
        raise ValueError("accepted v0 decision proposal lacks candidate evidence identity")
    if candidate_evidence_hash != expected_candidate_evidence_hash:
        raise ValueError(
            "accepted decision proposal candidate evidence identity differs from "
            "its exact resulting HardMask"
        )
    if receipt.get("action") != "accept":
        raise ValueError("accepted trajectory disagrees with its decision receipt")
    for field_name in (
        "action_candidate_id",
        "candidate_id",
        "provider_candidate_id",
    ):
        if field_name in receipt and receipt[field_name] != candidate.candidate_id:
            raise ValueError("accepted decision receipt names a different candidate")
    expected_bindings = {
        "hard_mask_parent_hash": candidate.parent_mask_hash,
        "hard_mask_result_hash": candidate.resulting_mask.mask_hash,
        "provider_state_hash": provider_state_hash,
    }
    for field_name, expected in expected_bindings.items():
        if field_name in receipt and receipt[field_name] != expected:
            raise ValueError(f"accepted decision receipt has a mismatched {field_name}")
    _validate_receipt_content_hash(receipt)
    receipt_hash = receipt.get("content_hash")
    if receipt_hash is None:
        receipt_hash = content_hash(receipt, prefix=_DECISION_RECEIPT_HASH_PREFIX)
    return proposal_id, str(receipt_hash), candidate_evidence_hash


def materialize_accepted_v0_trajectory(
    runs: Sequence[Any],
    *,
    v0_candidate_registry: Mapping[str, ActionCandidate],
    v0_mask_registry: Mapping[str, HardMask],
    root_mask_hash: str,
    terminal_mask_hash: str,
    adapter_id: str,
    query_id: str,
    hard_mask_to_layers: Callable[[HardMask], Mapping[str, Any]] | None = None,
) -> AcceptedV0Trajectory:
    """Materialize only exact accepted commits from completed native-v0 runs.

    ``proposal_index`` is obtained solely by enumerating each run's completed
    ``result.decisions`` tuple, exactly as ``native_continuous`` materializes
    its canonical frontier rows.  The join to the candidate registry is the
    full provider generator key; no decision or registry sequence is zipped.
    """

    if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence):
        raise TypeError("completed v0 runs must be a sequence")
    if not isinstance(v0_candidate_registry, Mapping) or not isinstance(v0_mask_registry, Mapping):
        raise TypeError("accepted-v0 materialization requires registry mappings")
    if not isinstance(adapter_id, str) or not adapter_id:
        raise ValueError("accepted-v0 materialization requires an adapter_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("accepted-v0 materialization requires a query_id")

    run_snapshot = tuple(runs)
    root = _registry_mask(v0_mask_registry, root_mask_hash, role="root")
    terminal = _registry_mask(v0_mask_registry, terminal_mask_hash, role="terminal")
    root.assert_same_axis(terminal)
    candidate_index = _candidate_registry_index(v0_candidate_registry)

    entries: list[AcceptedV0TrajectoryEntry] = []
    run_manifest: list[Mapping[str, Any]] = []
    decision_manifest: list[Mapping[str, Any]] = []
    nonaccepted_manifest: list[Mapping[str, Any]] = []
    current = root
    rejected_count = 0
    unresolved_count = 0

    for run_index, run in enumerate(run_snapshot):
        result = getattr(run, "result", None)
        signature = getattr(run, "signature", None)
        if result is None or signature is None:
            raise TypeError("completed v0 run lacks result or signature")
        run_query_id = getattr(signature, "query_id", None)
        layer_id = getattr(signature, "layer_id", None)
        if run_query_id != query_id or not isinstance(layer_id, str) or not layer_id:
            raise ValueError("completed v0 run lies outside the requested query axis")
        decisions = getattr(result, "decisions", None)
        if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
            raise TypeError("completed v0 run decisions are not a sequence")
        decision_snapshot = tuple(decisions)
        declared_count = getattr(signature, "decision_count", None)
        if type(declared_count) is not int or declared_count != len(decision_snapshot):
            raise ValueError("completed v0 run signature has a bad decision count")
        committed_masks = getattr(result, "committed_masks", None)
        if not isinstance(committed_masks, Mapping):
            raise TypeError("completed v0 run lacks committed mask arrays")
        v6_log = _v6_run_log(run)
        if v6_log is None:
            sampled_terminal_hash = sampled_path_mask_hash(committed_masks)
            if getattr(signature, "final_mask_hash", None) != sampled_terminal_hash:
                raise ValueError("completed v0 run signature has a bad final mask hash")
            receipts = _run_receipts(run)
            v6_records: dict[int, Mapping[str, Any]] = {}
        else:
            # The run's own signature digest (computed from
            # these arrays in-process by the group runner) is not recomputed;
            # the decision records replace the JSON receipts.
            sampled_terminal_hash = getattr(signature, "final_mask_hash", None)
            if not isinstance(sampled_terminal_hash, str) or not sampled_terminal_hash:
                raise ValueError("completed v0 run signature has a bad final mask hash")
            receipts = ()
            v6_records = {
                int(record["proposal_index"]): record for record in v6_log["records"]
            }
            if len(v6_records) != len(decision_snapshot):
                raise ValueError("v6 decision records do not cover the run's decisions")
        run_manifest.append(
            {
                "run_index": int(run_index),
                "query_id": query_id,
                "layer_id": layer_id,
                "profile_hash": str(getattr(signature, "profile_hash", "")),
                "decision_count": len(decision_snapshot),
                "final_sampled_path_mask_hash": sampled_terminal_hash,
                "stopping_reason": str(getattr(result, "stopping_reason", "")),
                "algorithm_version": str(getattr(signature, "algorithm_version", "")),
            }
        )
        for proposal_index, decision in enumerate(decision_snapshot):
            partition_id = getattr(decision, "partition_id", None)
            action = getattr(decision, "action", None)
            if not isinstance(partition_id, str) or not partition_id:
                raise ValueError("v0 decision lacks a partition identity")
            if not isinstance(action, str) or not action:
                raise ValueError("v0 decision lacks an action")
            decision_record = {
                "run_index": int(run_index),
                "query_id": query_id,
                "layer_id": layer_id,
                "proposal_index": int(proposal_index),
                "partition_id": partition_id,
                "action": action,
                "reason": str(getattr(decision, "reason", "")),
            }
            decision_manifest.append(decision_record)
            if action != "accept":
                nonaccepted_manifest.append(decision_record)
                if action == "reject":
                    rejected_count += 1
                else:
                    unresolved_count += 1
                continue

            join_key = (
                adapter_id,
                query_id,
                layer_id,
                partition_id,
                int(proposal_index),
            )
            candidates = candidate_index.get(join_key, ())
            if len(candidates) != 1:
                raise ValueError("accepted v0 decision must bind exactly one ActionCandidate")
            candidate = candidates[0]
            parent = _registry_mask(
                v0_mask_registry,
                candidate.parent_mask_hash,
                role="accepted parent",
            )
            resulting = _registry_mask(
                v0_mask_registry,
                candidate.resulting_mask.mask_hash,
                role="accepted result",
            )
            if candidate.resulting_mask != resulting:
                raise ValueError("accepted candidate resulting HardMask differs from the registry")
            if parent != current:
                raise ValueError("accepted v0 candidate breaks the commit parent chain")
            v6_record = v6_records.get(int(proposal_index)) if v6_log is not None else None
            if v6_log is not None:
                # v6: the delta is exact by construction (an order slice); the
                # record's count and exact candidate mask_hash are compared
                # instead of re-scanning both complete masks (kind 4).
                if (
                    v6_record is None
                    or str(v6_record["action"]) != "accept"
                    or int(v6_record["removed_count"]) != candidate.removed_count
                    or candidate.removed_count <= 0
                    or (
                        v6_record.get("candidate_mask_hash") is not None
                        and v6_record["candidate_mask_hash"] != resulting.mask_hash
                    )
                ):
                    raise ValueError(
                        "accepted candidate route delta differs from its decision record"
                    )
            else:
                exact_removed = resulting.removed_index_set_from(parent)
                if exact_removed.count <= 0 or exact_removed.count != candidate.removed_count:
                    raise ValueError(
                        "accepted candidate route delta differs from its complete HardMasks"
                    )
                if candidate.removed_routes is not None and candidate.removed_routes != exact_removed:
                    raise ValueError(
                        "accepted candidate compact route delta differs from its HardMasks"
                    )
            expected_parent_state = content_hash(
                {
                    "query_id": query_id,
                    "parent_mask_hash": parent.mask_hash,
                },
                prefix=_PROVIDER_STATE_PREFIX,
            )
            if candidate.parent_state_hash != expected_parent_state:
                raise ValueError("accepted v0 candidate has a mismatched parent state")
            removed_route_count = getattr(decision, "removed_route_count", None)
            if removed_route_count is not None and (
                type(removed_route_count) is not int
                or removed_route_count != candidate.removed_count
            ):
                raise ValueError("accepted decision removed count differs from its ActionCandidate")
            provider_state_hash = content_hash(
                {
                    "adapter": adapter_id,
                    "query_id": query_id,
                    "proposal_index": int(proposal_index),
                    "parent_mask_hash": parent.mask_hash,
                    "candidate_mask_hash": resulting.mask_hash,
                },
                prefix=_PROVIDER_STATE_PREFIX,
            )
            if v6_log is not None:
                # The evidence identity is the exact candidate mask_hash
                # and the receipt hash is the decision-record join key.
                record_key = decision_key(query_id, layer_id, partition_id, proposal_index)
                decision_proposal_id = f"{record_key}:{resulting.mask_hash}"
                decision_receipt_hash = decision_record_hash(record_key)
                decision_candidate_evidence_hash = resulting.mask_hash
            else:
                expected_candidate_evidence_hash = sampled_path_mask_hash(
                    _sampled_masks_from_hard_mask(
                        resulting,
                        sampled_template=committed_masks,
                        hard_mask_to_layers=hard_mask_to_layers,
                        role="accepted candidate HardMask",
                    )
                )
                (
                    decision_proposal_id,
                    decision_receipt_hash,
                    decision_candidate_evidence_hash,
                ) = _accepted_decision_receipt(
                    receipts=receipts,
                    query_id=query_id,
                    layer_id=layer_id,
                    partition_id=partition_id,
                    proposal_index=proposal_index,
                    candidate=candidate,
                    provider_state_hash=provider_state_hash,
                    expected_candidate_evidence_hash=expected_candidate_evidence_hash,
                )
            entries.append(
                AcceptedV0TrajectoryEntry(
                    trajectory_index=len(entries),
                    run_index=run_index,
                    proposal_index=proposal_index,
                    query_id=query_id,
                    layer_id=layer_id,
                    partition_id=partition_id,
                    candidate_id=candidate.candidate_id,
                    parent_mask_hash=parent.mask_hash,
                    resulting_mask_hash=resulting.mask_hash,
                    provider_state_hash=provider_state_hash,
                    parent_mask=parent,
                    resulting_mask=resulting,
                    decision_proposal_id=decision_proposal_id,
                    decision_receipt_hash=decision_receipt_hash,
                    decision_candidate_evidence_hash=(decision_candidate_evidence_hash),
                )
            )
            current = resulting

        v6_terminal = (
            None
            if v6_log is None
            else v6_log["final_mask_hashes"].get(str(current.route_axis_hash))
        )
        if v6_terminal is not None:
            # v6: the run's final committed state's exact HardMask identity
            # (built on the declared axis during the walk) must be the chain's.
            if v6_terminal != current.mask_hash:
                raise ValueError(
                    "completed v0 run sampled terminal differs from the current "
                    "accepted-chain HardMask"
                )
            continue
        expected_sampled_terminal = _sampled_masks_from_hard_mask(
            current,
            sampled_template=committed_masks,
            hard_mask_to_layers=hard_mask_to_layers,
            role="current accepted-chain HardMask",
        )
        if any(
            not np.array_equal(
                expected_sampled_terminal[layer_id],
                np.asarray(committed_masks[layer_id], dtype=bool),
            )
            for layer_id in expected_sampled_terminal
        ):
            raise ValueError(
                "completed v0 run sampled terminal differs from the current accepted-chain HardMask"
            )
        if sampled_path_mask_hash(expected_sampled_terminal) != sampled_terminal_hash:
            raise ValueError(
                "completed v0 run sampled terminal identity differs from the current "
                "accepted-chain HardMask"
            )

    if current != terminal:
        raise ValueError("accepted v0 commit chain differs from the terminal HardMask")

    receipt = AcceptedV0TrajectoryReceipt(
        adapter_id=adapter_id,
        query_id=query_id,
        root_mask_hash=root.mask_hash,
        terminal_mask_hash=terminal.mask_hash,
        run_count=len(run_snapshot),
        total_decision_count=len(decision_manifest),
        accepted_count=len(entries),
        rejected_count=rejected_count,
        unresolved_count=unresolved_count,
        run_manifest_hash=content_hash(tuple(run_manifest), prefix="accepted-v0-run-manifest-"),
        decision_manifest_hash=content_hash(
            tuple(decision_manifest), prefix="accepted-v0-decision-manifest-"
        ),
        nonaccepted_manifest_hash=content_hash(
            tuple(nonaccepted_manifest),
            prefix="accepted-v0-noncommit-manifest-",
        ),
        entry_hashes=tuple(entry.entry_hash for entry in entries),
    )
    return AcceptedV0Trajectory(
        root_mask=root,
        terminal_mask=terminal,
        entries=tuple(entries),
        receipt=receipt,
    )


__all__ = [
    "ACCEPTED_V0_TRAJECTORY_ENTRY_SCHEMA",
    "ACCEPTED_V0_TRAJECTORY_RECEIPT_SCHEMA",
    "ACCEPTED_V0_TRAJECTORY_SCHEMA",
    "AcceptedV0Trajectory",
    "AcceptedV0TrajectoryAuthority",
    "AcceptedV0TrajectoryEntry",
    "AcceptedV0TrajectoryReceipt",
    "materialize_accepted_v0_trajectory",
]
