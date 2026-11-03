"""Authoritative IRG workflow identity registry.

This module is intentionally narrow: it names the current workflow surface and
refuses superseded numeric workflow forms.  It does not import implementation
modules, does not forward old versions to new ones, and does not make any
scientific completion claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping


CURRENT_AUTHORITY = "irg/v0"
NEXT_DESIGN = "irg/v1"


class WorkflowIdentityError(ValueError):
    """Base class for workflow-identity resolution failures."""


class UnknownWorkflowIdentityError(WorkflowIdentityError):
    """Raised when an identity is neither current, reserved, nor replay."""

    def __init__(self, requested_identity: str) -> None:
        self.requested_identity = str(requested_identity)
        super().__init__(
            f"unknown IRG workflow identity {self.requested_identity!r}; "
            "expected irg/v0, irg/v1, reserved irg/v2-irg/v5, or an "
            "explicit replay identity"
        )


class UnregisteredWorkflowError(WorkflowIdentityError):
    """Raised when a known developing workflow has no registered selector."""

    def __init__(self, record: "WorkflowIdentityRecord") -> None:
        self.requested_identity = record.canonical_identity
        self.status = record.status
        self.selector_status = record.selector_status
        self.current_authority = CURRENT_AUTHORITY
        self.next_design = NEXT_DESIGN
        self.registration_gate = record.registration_gate
        super().__init__(
            f"{record.canonical_identity} is a developing workflow design, "
            "but its selector is fail-closed until registration"
        )


class SupersededWorkflowError(WorkflowIdentityError):
    """Raised for reserved v2-v5 workflow names.

    The attributes are stable so callers can record a machine-readable refusal.
    """

    def __init__(self, record: "WorkflowIdentityRecord") -> None:
        self.requested_identity = record.canonical_identity
        self.status = "reserved_for_rederivation"
        self.archive_wave = record.archive_wave
        self.current_authority = CURRENT_AUTHORITY
        self.next_design = NEXT_DESIGN
        self.restoration_protocol = record.restoration_protocol
        super().__init__(
            f"{record.canonical_identity} is reserved for rederivation and has "
            "no live workflow implementation; restore historical material only "
            "through the archived-wave protocol"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_identity": self.requested_identity,
            "status": self.status,
            "archive_wave": self.archive_wave,
            "current_authority": self.current_authority,
            "next_design": self.next_design,
            "restoration_protocol": self.restoration_protocol,
        }


@dataclass(frozen=True, slots=True)
class WorkflowIdentityRecord:
    """Immutable machine-readable status record for one workflow identity."""

    canonical_identity: str
    aliases: tuple[str, ...]
    status: str
    selector_status: str
    implementation_identity: str | None = None
    design_identity: str | None = None
    implementation: str | None = None
    completion_status: str | None = None
    v0_complete: bool | None = None
    archive_wave: str | None = None
    current_authority: str = CURRENT_AUTHORITY
    next_design: str = NEXT_DESIGN
    registration_gate: str | None = None
    restoration_protocol: str | None = None
    runtime_role: str | None = None
    claim_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_identity": self.canonical_identity,
            "aliases": self.aliases,
            "status": self.status,
            "selector_status": self.selector_status,
            "implementation_identity": self.implementation_identity,
            "design_identity": self.design_identity,
            "implementation": self.implementation,
            "completion_status": self.completion_status,
            "v0_complete": self.v0_complete,
            "archive_wave": self.archive_wave,
            "current_authority": self.current_authority,
            "next_design": self.next_design,
            "registration_gate": self.registration_gate,
            "restoration_protocol": self.restoration_protocol,
            "runtime_role": self.runtime_role,
            "claim_status": self.claim_status,
        }


def _reserved_record(version: int) -> WorkflowIdentityRecord:
    canonical = f"irg/v{version}"
    archive_wave = f"20260809_irg_v{version}_workflow_form"
    return WorkflowIdentityRecord(
        canonical_identity=canonical,
        aliases=(canonical, f"v{version}"),
        status="reserved_for_rederivation",
        selector_status="fail_closed",
        implementation_identity=None,
        implementation=None,
        completion_status=None,
        v0_complete=None,
        archive_wave=archive_wave,
        registration_gate=None,
        restoration_protocol=(
            "restore archived workflow form only into an isolated scratch tree "
            "after validating archive manifest hashes with "
            f"tools/legacy_archive/restore.py --wave-id {archive_wave} --check; "
            "never import it as a current runtime workflow"
        ),
        runtime_role="reserved",
        claim_status=None,
    )


_RECORDS: dict[str, WorkflowIdentityRecord] = {
    "irg/v0": WorkflowIdentityRecord(
        canonical_identity="irg/v0",
        aliases=("irg/v0", "v0"),
        status="authoritative_current",
        selector_status="current",
        implementation_identity="irg_v0_layerwise_loss_span_per_connection_bce_precision_v3",
        implementation="modern_queue_v0",
        completion_status="not_v0_complete_by_identity",
        v0_complete=False,
        runtime_role="current_workflow",
        claim_status="authority_identity_only_not_scientific_promotion",
    ),
    "irg/v1": WorkflowIdentityRecord(
        canonical_identity="irg/v1",
        aliases=("irg/v1", "partition_free_v1"),
        status="unregistered_design",
        selector_status="fail_closed",
        implementation_identity=None,
        design_identity="irg_v1_partition_free_dynamic_relinearization_v1",
        implementation=None,
        completion_status=None,
        v0_complete=None,
        registration_gate=(
            "owner_signed_a1_a2_selected_family_proof_inversion_firewall_"
            "and_rank8_q60_q256_gpu_gates"
        ),
        runtime_role="developing_workflow",
        claim_status=None,
    ),
    **{f"irg/v{version}": _reserved_record(version) for version in range(2, 6)},
    "historical_irg_v0_hierarchical_marginal_replay_v1": WorkflowIdentityRecord(
        canonical_identity="historical_irg_v0_hierarchical_marginal_replay_v1",
        aliases=("historical_irg_v0_hierarchical_marginal_replay_v1",),
        status="historical_replay",
        selector_status="explicit_replay_only",
        implementation_identity="historical_irg_v0_hierarchical_marginal_replay_v1",
        implementation="legacy_v0_replay_adapter",
        runtime_role="historical_replay",
        claim_status="replay_only_not_current_authority",
    ),
    "retired_irg_v1_trust_walk_partition_alternating_v1": WorkflowIdentityRecord(
        canonical_identity="retired_irg_v1_trust_walk_partition_alternating_v1",
        aliases=("retired_irg_v1_trust_walk_partition_alternating_v1",),
        status="retired_nonclaim_replay",
        selector_status="explicit_replay_only",
        implementation_identity="irg_v1_trust_walk_partition_alternating_v1",
        implementation="trust_walk_partition_alternating_replay",
        runtime_role="retired_nonclaim_replay",
        claim_status="retired_nonclaim_replay_only",
    ),
}

WORKFLOW_STATUS_REGISTRY: Mapping[str, WorkflowIdentityRecord] = MappingProxyType(
    dict(_RECORDS)
)

_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        alias.lower(): canonical
        for canonical, record in WORKFLOW_STATUS_REGISTRY.items()
        for alias in record.aliases
    }
)


def normalize_workflow_identity(requested_identity: str) -> str:
    """Return the canonical spelling for a known workflow or replay identity."""

    key = str(requested_identity).strip().lower()
    if not key:
        raise UnknownWorkflowIdentityError(str(requested_identity))
    canonical = _ALIASES.get(key)
    if canonical is None:
        raise UnknownWorkflowIdentityError(str(requested_identity))
    return canonical


def lookup_workflow_identity_status(
    requested_identity: str,
) -> WorkflowIdentityRecord:
    """Look up a known identity without authorizing execution."""

    return WORKFLOW_STATUS_REGISTRY[normalize_workflow_identity(requested_identity)]


def resolve_current_workflow_identity(
    requested_identity: str,
) -> WorkflowIdentityRecord:
    """Resolve an executable current workflow identity or fail closed."""

    record = lookup_workflow_identity_status(requested_identity)
    if record.canonical_identity == CURRENT_AUTHORITY:
        return record
    if record.status == "unregistered_design":
        raise UnregisteredWorkflowError(record)
    if record.status == "reserved_for_rederivation":
        raise SupersededWorkflowError(record)
    raise WorkflowIdentityError(
        f"{record.canonical_identity!r} is {record.status!r}, not a current "
        "workflow selector"
    )


def iter_workflow_identity_records() -> Iterable[WorkflowIdentityRecord]:
    """Iterate immutable workflow records in canonical-key order."""

    for key in sorted(WORKFLOW_STATUS_REGISTRY):
        yield WORKFLOW_STATUS_REGISTRY[key]


def workflow_status_payload() -> dict[str, dict[str, Any]]:
    """Return a JSON/YAML-ready snapshot of the registry."""

    return {
        key: WORKFLOW_STATUS_REGISTRY[key].to_dict()
        for key in sorted(WORKFLOW_STATUS_REGISTRY)
    }


__all__ = [
    "CURRENT_AUTHORITY",
    "NEXT_DESIGN",
    "SupersededWorkflowError",
    "UnknownWorkflowIdentityError",
    "UnregisteredWorkflowError",
    "WORKFLOW_STATUS_REGISTRY",
    "WorkflowIdentityError",
    "WorkflowIdentityRecord",
    "iter_workflow_identity_records",
    "lookup_workflow_identity_status",
    "normalize_workflow_identity",
    "resolve_current_workflow_identity",
    "workflow_status_payload",
]
