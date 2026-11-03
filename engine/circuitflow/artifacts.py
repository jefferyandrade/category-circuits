"""Immutable, coordinate-aware CircuitFlow artifact bundles.

An artifact bundle is the only thing a downstream CircuitFlow stage may
consume.  Filesystem paths are transport locations, never semantic identity.
The manifest carries the parent hashes, axis declaration, coordinate schema,
and payload digests; certification and replica receipts live beside the bundle
without mutating its scientific content hash.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from contracts.hashing import sha256_file, sha256_path, stable_digest

SCHEMA_NAME = "circuitflow_artifact_manifest"
SCHEMA_VERSION = "circuitflow_artifact_manifest_v2"
CONTRACT_SCHEMA_NAME = "circuitflow_contract_record"
CONTRACT_SCHEMA_VERSION = "circuitflow_contract_record_v2"
LIFECYCLE_EVENT_SCHEMA_NAME = "circuitflow_lifecycle_event"
LIFECYCLE_EVENT_SCHEMA_VERSION = "circuitflow_lifecycle_event_v2"


class ArtifactType(StrEnum):
    MEASUREMENT = "measurement"
    CIRCUIT = "circuit"
    RESPONSE_BASIS = "response_basis"
    ACTION = "action"
    RESPONSE = "response"
    GEOMETRY = "geometry"
    INFERENCE = "inference"


class LifecycleState(StrEnum):
    PLANNED = "planned"
    SUBMITTED = "submitted"
    RUNNING = "running"
    STAGED = "staged"
    SEALED = "sealed"
    CERTIFIED = "certified"
    ANALYSIS_READY = "analysis_ready"
    CLAIM_READY = "claim_ready"
    FAILED = "failed"
    INVALID = "invalid"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"


# Lifecycle is operational metadata, deliberately outside an immutable
# scientific artifact's content hash.  The graph is conservative: an artifact
# cannot regress through the normal production path, while a failed or
# quarantined artifact may be recertified after its referenced external source
# has been repaired and revalidated.
_LIFECYCLE_TRANSITIONS: Mapping[LifecycleState | None, frozenset[LifecycleState]] = {
    # Direct certification is allowed only as a compatibility path for an
    # immutable bundle sealed by an earlier CircuitFlow build that did not yet
    # record lifecycle events.  New native writes always record ``sealed``.
    None: frozenset({LifecycleState.PLANNED, LifecycleState.STAGED, LifecycleState.SEALED, LifecycleState.CERTIFIED, LifecycleState.QUARANTINED}),
    LifecycleState.PLANNED: frozenset(
        {LifecycleState.SUBMITTED, LifecycleState.STAGED, LifecycleState.FAILED, LifecycleState.INVALID, LifecycleState.SUPERSEDED}
    ),
    LifecycleState.SUBMITTED: frozenset(
        {LifecycleState.RUNNING, LifecycleState.STAGED, LifecycleState.FAILED, LifecycleState.INVALID, LifecycleState.SUPERSEDED}
    ),
    LifecycleState.RUNNING: frozenset(
        {LifecycleState.STAGED, LifecycleState.FAILED, LifecycleState.INVALID, LifecycleState.SUPERSEDED}
    ),
    LifecycleState.STAGED: frozenset(
        {LifecycleState.SEALED, LifecycleState.FAILED, LifecycleState.INVALID, LifecycleState.QUARANTINED, LifecycleState.SUPERSEDED}
    ),
    LifecycleState.SEALED: frozenset(
        {LifecycleState.CERTIFIED, LifecycleState.QUARANTINED, LifecycleState.INVALID, LifecycleState.SUPERSEDED}
    ),
    LifecycleState.CERTIFIED: frozenset(
        {LifecycleState.ANALYSIS_READY, LifecycleState.CLAIM_READY, LifecycleState.QUARANTINED, LifecycleState.INVALID, LifecycleState.SUPERSEDED}
    ),
    LifecycleState.ANALYSIS_READY: frozenset(
        {LifecycleState.CLAIM_READY, LifecycleState.QUARANTINED, LifecycleState.INVALID, LifecycleState.SUPERSEDED}
    ),
    LifecycleState.CLAIM_READY: frozenset(
        {LifecycleState.QUARANTINED, LifecycleState.INVALID, LifecycleState.SUPERSEDED}
    ),
    LifecycleState.FAILED: frozenset({LifecycleState.STAGED, LifecycleState.SUPERSEDED}),
    LifecycleState.INVALID: frozenset({LifecycleState.STAGED, LifecycleState.SUPERSEDED}),
    LifecycleState.QUARANTINED: frozenset({LifecycleState.STAGED, LifecycleState.CERTIFIED, LifecycleState.INVALID, LifecycleState.SUPERSEDED}),
    LifecycleState.SUPERSEDED: frozenset(),
}


VALID_AXIS_ROLES = frozenset(
    {
        "query",
        "category",
        "stimulus",
        "query_stimulus_pair",
        "role",
        "route",
        "layer",
        "feature",
        "subject",
        "roi",
        "repeat",
        "action",
        "intervention",
        "readout",
        "outcome",
        "dose",
        "control",
    }
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=f".{path.name}.",
    ) as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


@dataclass(frozen=True)
class PayloadRef:
    """A content-verified payload inside or referenced by a bundle."""

    name: str
    sha256: str
    uri: str
    storage_format: str
    bytes: int = 0
    media_type: str = "application/octet-stream"
    external: bool = False
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "uri": self.uri,
            "storage_format": self.storage_format,
            "bytes": int(self.bytes),
            "media_type": self.media_type,
            "external": bool(self.external),
            "notes": _jsonable(self.notes),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PayloadRef:
        return cls(
            name=str(payload.get("name", "")),
            sha256=str(payload.get("sha256", "")),
            uri=str(payload.get("uri", "")),
            storage_format=str(payload.get("storage_format", "")),
            bytes=int(payload.get("bytes", 0) or 0),
            media_type=str(payload.get("media_type", "application/octet-stream")),
            external=bool(payload.get("external", False)),
            notes=dict(payload.get("notes", {}) or {}),
        )

    @classmethod
    def from_file(
        cls,
        *,
        name: str,
        path: str | Path,
        uri: str | None = None,
        storage_format: str = "binary",
        media_type: str = "application/octet-stream",
        external: bool = False,
        notes: Mapping[str, Any] | None = None,
    ) -> PayloadRef:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        return cls(
            name=str(name),
            sha256=sha256_file(source),
            uri=str(uri if uri is not None else source),
            storage_format=str(storage_format),
            bytes=int(source.stat().st_size),
            media_type=str(media_type),
            external=bool(external),
            notes=dict(notes or {}),
        )

    @classmethod
    def from_path(
        cls,
        *,
        name: str,
        path: str | Path,
        uri: str | None = None,
        storage_format: str = "binary",
        media_type: str = "application/octet-stream",
        external: bool = False,
        notes: Mapping[str, Any] | None = None,
    ) -> PayloadRef:
        """Build a payload reference for a file or immutable directory tree."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        is_directory = source.is_dir()
        if not is_directory and not source.is_file():
            raise ValueError(f"unsupported payload path type: {source}")
        byte_count = (
            sum(int(member.stat().st_size) for member in source.rglob("*") if member.is_file())
            if is_directory
            else int(source.stat().st_size)
        )
        payload_notes = {**dict(notes or {}), "is_directory": is_directory}
        return cls(
            name=str(name),
            sha256=sha256_path(source),
            uri=str(uri if uri is not None else source),
            storage_format=str(storage_format),
            bytes=byte_count,
            media_type=str(media_type),
            external=bool(external),
            notes=payload_notes,
        )

    def resolve(self, bundle_dir: str | Path) -> Path | None:
        """Resolve local payload locations; remote URIs remain intentionally lazy."""

        value = str(self.uri)
        if value.startswith("s3://") or "://" in value:
            return None
        path = Path(value)
        return path if path.is_absolute() else Path(bundle_dir) / path

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name:
            issues.append("payload_name_missing")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            issues.append(f"payload_sha256_invalid:{self.name}")
        if not self.uri:
            issues.append(f"payload_uri_missing:{self.name}")
        if not self.storage_format:
            issues.append(f"payload_storage_format_missing:{self.name}")
        if self.bytes < 0:
            issues.append(f"payload_bytes_negative:{self.name}")
        return tuple(issues)


@dataclass(frozen=True)
class ArtifactManifest:
    """Semantic envelope shared by every CircuitFlow artifact kind."""

    artifact_type: ArtifactType
    study_hash: str
    plan_hash: str
    parent_hashes: tuple[str, ...]
    axes: Mapping[str, tuple[str, ...]]
    coordinate_schema_hash: str
    semantics: Mapping[str, Any]
    payloads: tuple[PayloadRef, ...]
    unit_ids: tuple[str, ...] = tuple()
    source_kind: str = "native"
    valid_for_claims: bool = False
    schema_name: str = SCHEMA_NAME
    schema_version: str = SCHEMA_VERSION
    notes: Mapping[str, Any] = field(default_factory=dict)

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type.value,
            "study_hash": self.study_hash,
            "plan_hash": self.plan_hash,
            "parent_hashes": list(self.parent_hashes),
            "axes": {str(key): list(value) for key, value in self.axes.items()},
            "coordinate_schema_hash": self.coordinate_schema_hash,
            "semantics": _jsonable(self.semantics),
            "payloads": [payload.to_dict() for payload in self.payloads],
            "unit_ids": list(self.unit_ids),
            "source_kind": self.source_kind,
            "valid_for_claims": bool(self.valid_for_claims),
            "notes": _jsonable(self.notes),
        }

    @property
    def artifact_hash(self) -> str:
        return stable_digest(self.semantic_dict())

    @property
    def artifact_id(self) -> str:
        return f"{self.artifact_type.value}-{self.artifact_hash[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "artifact_hash": self.artifact_hash, "artifact_id": self.artifact_id}

    def validate(self, *, claim_bearing: bool = False) -> tuple[str, ...]:
        issues: list[str] = []
        if self.schema_name != SCHEMA_NAME or self.schema_version != SCHEMA_VERSION:
            issues.append("unsupported_artifact_schema")
        if not isinstance(self.artifact_type, ArtifactType):
            issues.append("unsupported_artifact_type")
        for name, value in (
            ("study_hash", self.study_hash),
            ("plan_hash", self.plan_hash),
            ("coordinate_schema_hash", self.coordinate_schema_hash),
            ("source_kind", self.source_kind),
        ):
            if not str(value).strip():
                issues.append(f"{name}_missing")
        if not self.payloads:
            issues.append("payloads_missing")
        payload_names: set[str] = set()
        for payload in self.payloads:
            issues.extend(payload.validate())
            if payload.name in payload_names:
                issues.append(f"duplicate_payload:{payload.name}")
            payload_names.add(payload.name)
        if not isinstance(self.axes, Mapping):
            issues.append("axes_not_mapping")
        else:
            for role, identifiers in self.axes.items():
                if str(role) not in VALID_AXIS_ROLES:
                    issues.append(f"unsupported_axis_role:{role}")
                if not isinstance(identifiers, (tuple, list)) or not identifiers:
                    issues.append(f"axis_empty:{role}")
                elif len({str(value) for value in identifiers}) != len(identifiers):
                    issues.append(f"axis_ids_not_unique:{role}")
        if not isinstance(self.semantics, Mapping) or not self.semantics:
            issues.append("semantics_missing")
        if claim_bearing and not self.valid_for_claims:
            issues.append("artifact_not_claim_valid")
        if claim_bearing and self.source_kind not in {"native", "certified_legacy_adapter"}:
            issues.append("claim_artifact_must_be_native_or_certified_legacy_adapter")
        return tuple(issues)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ArtifactManifest:
        raw_type = str(payload.get("artifact_type", ""))
        try:
            artifact_type = ArtifactType(raw_type)
        except ValueError as error:
            raise ValueError(f"unknown CircuitFlow artifact_type {raw_type!r}") from error
        raw_axes = payload.get("axes", {})
        if not isinstance(raw_axes, Mapping):
            raise ValueError("artifact axes must be a mapping")
        result = cls(
            artifact_type=artifact_type,
            study_hash=str(payload.get("study_hash", "")),
            plan_hash=str(payload.get("plan_hash", "")),
            parent_hashes=tuple(str(value) for value in payload.get("parent_hashes", ()) or ()),
            axes={str(key): tuple(str(value) for value in values) for key, values in raw_axes.items()},
            coordinate_schema_hash=str(payload.get("coordinate_schema_hash", "")),
            semantics=dict(payload.get("semantics", {}) or {}),
            payloads=tuple(PayloadRef.from_mapping(item) for item in payload.get("payloads", ()) or ()),
            unit_ids=tuple(str(value) for value in payload.get("unit_ids", ()) or ()),
            source_kind=str(payload.get("source_kind", "native")),
            valid_for_claims=bool(payload.get("valid_for_claims", False)),
            schema_name=str(payload.get("schema_name", SCHEMA_NAME)),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            notes=dict(payload.get("notes", {}) or {}),
        )
        declared_hash = str(payload.get("artifact_hash", ""))
        if declared_hash and declared_hash != result.artifact_hash:
            raise ValueError("artifact_hash does not match manifest semantic content")
        issues = result.validate()
        if issues:
            raise ValueError("invalid ArtifactManifest: " + "; ".join(issues))
        return result

    @classmethod
    def load(cls, path: str | Path) -> ArtifactManifest:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} did not contain an artifact object")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class CertificationReceipt:
    """A non-semantic validation event attached to an immutable artifact."""

    artifact_hash: str
    level: str
    passed: bool
    validator_id: str
    issues: tuple[str, ...] = tuple()
    details: Mapping[str, Any] = field(default_factory=dict)
    created_unix_seconds: float = field(default_factory=time.time)
    schema_name: str = "circuitflow_certification_receipt"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "artifact_hash": self.artifact_hash,
            "level": self.level,
            "passed": bool(self.passed),
            "validator_id": self.validator_id,
            "issues": list(self.issues),
            "details": _jsonable(self.details),
            "created_unix_seconds": float(self.created_unix_seconds),
        }

    @property
    def receipt_hash(self) -> str:
        return stable_digest(self.to_dict())


@dataclass(frozen=True)
class LifecycleEvent:
    """Append-only operational state transition for a sealed artifact.

    A lifecycle event never changes the artifact manifest or its semantic
    hash.  It records whether a specific immutable bundle is merely staged,
    certified for analysis, or has passed its declared claim gate.  This keeps
    runtime state auditable without polluting scientific identity.
    """

    artifact_hash: str
    state: LifecycleState
    previous_state: LifecycleState | None = None
    actor: str = "circuitflow"
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    created_unix_seconds: float = field(default_factory=time.time)
    schema_name: str = LIFECYCLE_EVENT_SCHEMA_NAME
    schema_version: str = LIFECYCLE_EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "artifact_hash": self.artifact_hash,
            "state": self.state.value,
            "previous_state": None if self.previous_state is None else self.previous_state.value,
            "actor": self.actor,
            "reason": self.reason,
            "details": _jsonable(self.details),
            "created_unix_seconds": float(self.created_unix_seconds),
        }

    @property
    def event_hash(self) -> str:
        return stable_digest(self.to_dict())

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if len(self.artifact_hash) != 64 or any(character not in "0123456789abcdef" for character in self.artifact_hash):
            issues.append("lifecycle_artifact_hash_invalid")
        if not isinstance(self.state, LifecycleState):
            issues.append("lifecycle_state_invalid")
        if self.previous_state is not None and not isinstance(self.previous_state, LifecycleState):
            issues.append("lifecycle_previous_state_invalid")
        if not str(self.actor).strip():
            issues.append("lifecycle_actor_missing")
        if self.schema_name != LIFECYCLE_EVENT_SCHEMA_NAME or self.schema_version != LIFECYCLE_EVENT_SCHEMA_VERSION:
            issues.append("unsupported_lifecycle_event_schema")
        return tuple(issues)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> LifecycleEvent:
        raw_previous = payload.get("previous_state")
        event = cls(
            artifact_hash=str(payload.get("artifact_hash", "")),
            state=LifecycleState(str(payload.get("state", ""))),
            previous_state=None if raw_previous in {None, ""} else LifecycleState(str(raw_previous)),
            actor=str(payload.get("actor", "circuitflow")),
            reason=str(payload.get("reason", "")),
            details=dict(payload.get("details", {}) or {}),
            created_unix_seconds=float(payload.get("created_unix_seconds", 0.0)),
            schema_name=str(payload.get("schema_name", LIFECYCLE_EVENT_SCHEMA_NAME)),
            schema_version=str(payload.get("schema_version", LIFECYCLE_EVENT_SCHEMA_VERSION)),
        )
        issues = event.validate()
        if issues:
            raise ValueError("invalid CircuitFlow lifecycle event: " + "; ".join(issues))
        return event


@dataclass(frozen=True)
class RuntimeReceipt:
    """Typed runtime/performance receipt associated with one artifact bundle.

    Runtime health is not scientific evidence, but it is essential for
    reproducibility and for deciding whether a packet can safely be skipped or
    escalated.  The receipt is therefore separately content-addressed and
    never participates in an artifact's semantic hash.
    """

    artifact_hash: str
    work_packet_hash: str
    stage: str
    partition: str
    wall_seconds: float
    throughput_items_per_second: float | None = None
    peak_hbm_gb: float | None = None
    peak_hbm_reserved_gb: float | None = None
    peak_rss_gb: float | None = None
    loader_seconds: float | None = None
    transfer_seconds: float | None = None
    kernel_seconds: float | None = None
    synchronization_seconds: float | None = None
    writer_seconds: float | None = None
    cpu_utilization: float | None = None
    gpu_utilization: float | None = None
    host_to_device_bytes: int | None = None
    device_to_host_bytes: int | None = None
    logical_forward_count: int | None = None
    physical_forward_count: int | None = None
    lane_occupancy: float | None = None
    cache_hits: int | None = None
    shard_id: str = ""
    retry_ordinal: int | None = None
    backend_id: str = ""
    precision: str = ""
    replay_fraction: float | None = None
    replay_parity_passed: bool | None = None
    cache_ready: bool | None = None
    fallback_reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    created_unix_seconds: float = field(default_factory=time.time)
    schema_name: str = "circuitflow_runtime_receipt"
    schema_version: str = "circuitflow_runtime_receipt_v2"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "artifact_hash": self.artifact_hash,
            "work_packet_hash": self.work_packet_hash,
            "stage": self.stage,
            "partition": self.partition,
            "wall_seconds": float(self.wall_seconds),
            "throughput_items_per_second": self.throughput_items_per_second,
            "peak_hbm_gb": self.peak_hbm_gb,
            "loader_seconds": self.loader_seconds,
            "replay_fraction": self.replay_fraction,
            "replay_parity_passed": self.replay_parity_passed,
            "cache_ready": self.cache_ready,
            "fallback_reason": self.fallback_reason,
            "details": _jsonable(self.details),
            "created_unix_seconds": float(self.created_unix_seconds),
        }
        optional = {
            "peak_hbm_reserved_gb": self.peak_hbm_reserved_gb,
            "peak_rss_gb": self.peak_rss_gb,
            "transfer_seconds": self.transfer_seconds,
            "kernel_seconds": self.kernel_seconds,
            "synchronization_seconds": self.synchronization_seconds,
            "writer_seconds": self.writer_seconds,
            "cpu_utilization": self.cpu_utilization,
            "gpu_utilization": self.gpu_utilization,
            "host_to_device_bytes": self.host_to_device_bytes,
            "device_to_host_bytes": self.device_to_host_bytes,
            "logical_forward_count": self.logical_forward_count,
            "physical_forward_count": self.physical_forward_count,
            "lane_occupancy": self.lane_occupancy,
            "cache_hits": self.cache_hits,
            "retry_ordinal": self.retry_ordinal,
        }
        payload.update(
            {
                key: value
                for key, value in optional.items()
                if value is not None
            }
        )
        for key, value in (
            ("shard_id", self.shard_id),
            ("backend_id", self.backend_id),
            ("precision", self.precision),
        ):
            if value:
                payload[key] = value
        return payload

    @property
    def receipt_hash(self) -> str:
        return stable_digest(self.to_dict())

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if len(self.artifact_hash) != 64 or any(character not in "0123456789abcdef" for character in self.artifact_hash):
            issues.append("runtime_artifact_hash_invalid")
        if len(self.work_packet_hash) != 64 or any(character not in "0123456789abcdef" for character in self.work_packet_hash):
            issues.append("runtime_work_packet_hash_invalid")
        if not str(self.stage).strip() or not str(self.partition).strip():
            issues.append("runtime_stage_or_partition_missing")
        for name, value in (
            ("wall_seconds", self.wall_seconds),
            ("throughput_items_per_second", self.throughput_items_per_second),
            ("peak_hbm_gb", self.peak_hbm_gb),
            ("peak_hbm_reserved_gb", self.peak_hbm_reserved_gb),
            ("peak_rss_gb", self.peak_rss_gb),
            ("loader_seconds", self.loader_seconds),
            ("transfer_seconds", self.transfer_seconds),
            ("kernel_seconds", self.kernel_seconds),
            ("synchronization_seconds", self.synchronization_seconds),
            ("writer_seconds", self.writer_seconds),
            ("replay_fraction", self.replay_fraction),
            ("cpu_utilization", self.cpu_utilization),
            ("gpu_utilization", self.gpu_utilization),
            ("lane_occupancy", self.lane_occupancy),
        ):
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                issues.append(f"runtime_{name}_invalid")
        if self.replay_fraction is not None and float(self.replay_fraction) > 1.0:
            issues.append("runtime_replay_fraction_invalid")
        for name, value in (
            ("cpu_utilization", self.cpu_utilization),
            ("gpu_utilization", self.gpu_utilization),
            ("lane_occupancy", self.lane_occupancy),
        ):
            if value is not None and float(value) > 1.0:
                issues.append(f"runtime_{name}_invalid")
        for name, value in (
            ("host_to_device_bytes", self.host_to_device_bytes),
            ("device_to_host_bytes", self.device_to_host_bytes),
            ("logical_forward_count", self.logical_forward_count),
            ("physical_forward_count", self.physical_forward_count),
            ("cache_hits", self.cache_hits),
            ("retry_ordinal", self.retry_ordinal),
        ):
            if value is not None and int(value) < 0:
                issues.append(f"runtime_{name}_invalid")
        if (
            self.logical_forward_count is not None
            and self.physical_forward_count is not None
            and int(self.physical_forward_count)
            > int(self.logical_forward_count)
        ):
            issues.append("runtime_physical_forward_count_exceeds_logical")
        if self.replay_fraction and self.replay_parity_passed is not True:
            issues.append("runtime_replay_requires_parity")
        if self.schema_name != "circuitflow_runtime_receipt" or self.schema_version != "circuitflow_runtime_receipt_v2":
            issues.append("unsupported_runtime_receipt_schema")
        return tuple(issues)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RuntimeReceipt:
        receipt = cls(
            artifact_hash=str(payload.get("artifact_hash", "")),
            work_packet_hash=str(payload.get("work_packet_hash", "")),
            stage=str(payload.get("stage", "")),
            partition=str(payload.get("partition", "")),
            wall_seconds=float(payload.get("wall_seconds", float("nan"))),
            throughput_items_per_second=None if payload.get("throughput_items_per_second") is None else float(payload["throughput_items_per_second"]),
            peak_hbm_gb=None if payload.get("peak_hbm_gb") is None else float(payload["peak_hbm_gb"]),
            peak_hbm_reserved_gb=None if payload.get("peak_hbm_reserved_gb") is None else float(payload["peak_hbm_reserved_gb"]),
            peak_rss_gb=None if payload.get("peak_rss_gb") is None else float(payload["peak_rss_gb"]),
            loader_seconds=None if payload.get("loader_seconds") is None else float(payload["loader_seconds"]),
            transfer_seconds=None if payload.get("transfer_seconds") is None else float(payload["transfer_seconds"]),
            kernel_seconds=None if payload.get("kernel_seconds") is None else float(payload["kernel_seconds"]),
            synchronization_seconds=None if payload.get("synchronization_seconds") is None else float(payload["synchronization_seconds"]),
            writer_seconds=None if payload.get("writer_seconds") is None else float(payload["writer_seconds"]),
            cpu_utilization=None if payload.get("cpu_utilization") is None else float(payload["cpu_utilization"]),
            gpu_utilization=None if payload.get("gpu_utilization") is None else float(payload["gpu_utilization"]),
            host_to_device_bytes=None if payload.get("host_to_device_bytes") is None else int(payload["host_to_device_bytes"]),
            device_to_host_bytes=None if payload.get("device_to_host_bytes") is None else int(payload["device_to_host_bytes"]),
            logical_forward_count=None if payload.get("logical_forward_count") is None else int(payload["logical_forward_count"]),
            physical_forward_count=None if payload.get("physical_forward_count") is None else int(payload["physical_forward_count"]),
            lane_occupancy=None if payload.get("lane_occupancy") is None else float(payload["lane_occupancy"]),
            cache_hits=None if payload.get("cache_hits") is None else int(payload["cache_hits"]),
            shard_id=str(payload.get("shard_id", "")),
            retry_ordinal=None if payload.get("retry_ordinal") is None else int(payload["retry_ordinal"]),
            backend_id=str(payload.get("backend_id", "")),
            precision=str(payload.get("precision", "")),
            replay_fraction=None if payload.get("replay_fraction") is None else float(payload["replay_fraction"]),
            replay_parity_passed=payload.get("replay_parity_passed"),
            cache_ready=payload.get("cache_ready"),
            fallback_reason=str(payload.get("fallback_reason", "")),
            details=dict(payload.get("details", {}) or {}),
            created_unix_seconds=float(payload.get("created_unix_seconds", 0.0)),
            schema_name=str(payload.get("schema_name", "circuitflow_runtime_receipt")),
            schema_version=str(payload.get("schema_version", "circuitflow_runtime_receipt_v2")),
        )
        issues = receipt.validate()
        if issues:
            raise ValueError("invalid CircuitFlow runtime receipt: " + "; ".join(issues))
        return receipt


@dataclass(frozen=True)
class ReplicaReceipt:
    """Verified mirror location; deliberately separate from artifact identity."""

    artifact_hash: str
    replica_uri: str
    replica_kind: str
    verified_payload_hashes: Mapping[str, str]
    created_unix_seconds: float = field(default_factory=time.time)
    schema_name: str = "circuitflow_replica_receipt"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "artifact_hash": self.artifact_hash,
            "replica_uri": self.replica_uri,
            "replica_kind": self.replica_kind,
            "verified_payload_hashes": dict(self.verified_payload_hashes),
            "created_unix_seconds": float(self.created_unix_seconds),
        }

    @property
    def receipt_hash(self) -> str:
        return stable_digest(self.to_dict())


class ArtifactStore:
    """File-native immutable store with atomic local writes and receipt streams."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    def bundle_dir(self, manifest_or_hash: ArtifactManifest | str, artifact_type: ArtifactType | str | None = None) -> Path:
        if isinstance(manifest_or_hash, ArtifactManifest):
            artifact_hash = manifest_or_hash.artifact_hash
            artifact_type_value = manifest_or_hash.artifact_type.value
        else:
            artifact_hash = str(manifest_or_hash)
            if artifact_type is None:
                raise ValueError("artifact_type is required when resolving from a hash")
            artifact_type_value = (
                artifact_type.value
                if isinstance(artifact_type, ArtifactType)
                else ArtifactType(str(artifact_type)).value
            )
        return self.root / "objects" / artifact_type_value / artifact_hash[:2] / artifact_hash

    def manifest_path(self, manifest_or_hash: ArtifactManifest | str, artifact_type: ArtifactType | str | None = None) -> Path:
        return self.bundle_dir(manifest_or_hash, artifact_type) / "manifest.json"

    def contract_path(self, contract_kind: str, content_hash: str) -> Path:
        """Return the immutable registry location for a plan or study spec.

        Plans are not scientific artifacts themselves, but they are first-class
        lineage nodes.  Storing them here lets the file-native registry resolve
        a study name or a frozen plan without consulting an arbitrary launch
        directory.
        """

        kind = str(contract_kind).strip()
        if not kind:
            raise ValueError("contract_kind must be nonempty")
        digest = str(content_hash).strip()
        if len(digest) != 64:
            raise ValueError("contract content hash must be a SHA-256 digest")
        return self.root / "contracts" / kind / digest[:2] / f"{digest}.json"

    def seal_contract(
        self,
        *,
        contract_kind: str,
        content_hash: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """Atomically seal an immutable StudySpec or plan record.

        The caller owns the content hash because contracts use typed hash
        semantics.  A repeated write is idempotent; a different payload at the
        same identity is rejected rather than overwritten.
        """

        target = self.contract_path(contract_kind, content_hash)
        record = {
            "schema_name": CONTRACT_SCHEMA_NAME,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "contract_kind": str(contract_kind),
            "content_hash": str(content_hash),
            "payload": _jsonable(payload),
        }
        if target.is_file():
            existing = json.loads(target.read_text())
            if existing != record:
                raise ValueError(f"immutable CircuitFlow contract conflict: {target}")
            return target
        _atomic_write_json(target, record)
        return target

    def load_contract(self, contract_kind: str, content_hash: str) -> Mapping[str, Any]:
        target = self.contract_path(contract_kind, content_hash)
        payload = json.loads(target.read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid CircuitFlow contract record: {target}")
        if (
            payload.get("schema_name") != CONTRACT_SCHEMA_NAME
            or payload.get("schema_version") != CONTRACT_SCHEMA_VERSION
            or payload.get("contract_kind") != str(contract_kind)
            or payload.get("content_hash") != str(content_hash)
            or not isinstance(payload.get("payload"), Mapping)
        ):
            raise ValueError(f"invalid CircuitFlow contract record: {target}")
        return dict(payload["payload"])

    def scan_contracts(self) -> tuple[dict[str, Any], ...]:
        """Read immutable plan/study records for status and provenance views."""

        root = self.root / "contracts"
        if not root.is_dir():
            return tuple()
        rows: list[dict[str, Any]] = []
        for path in sorted(root.glob("*/*/*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, Mapping):
                continue
            if payload.get("schema_name") != CONTRACT_SCHEMA_NAME:
                continue
            if payload.get("schema_version") != CONTRACT_SCHEMA_VERSION:
                continue
            if not isinstance(payload.get("payload"), Mapping):
                continue
            rows.append({**dict(payload), "path": str(path)})
        return tuple(rows)

    def write_bundle(
        self,
        manifest: ArtifactManifest,
        *,
        payload_files: Mapping[str, str | Path] | None = None,
        link_payloads: bool = False,
    ) -> Path:
        """Materialize a native immutable bundle using copy or hard-link payloads.

        External legacy payloads are represented in ``manifest.payloads`` and
        must not be passed here.  The final directory name is derived from the
        semantic hash, so repeated identical writes are naturally idempotent.
        """

        issues = manifest.validate()
        if issues:
            raise ValueError("cannot write invalid artifact: " + "; ".join(issues))
        destination = self.bundle_dir(manifest)
        manifest_path = destination / "manifest.json"
        if destination.exists():
            existing = ArtifactManifest.load(manifest_path)
            if existing.artifact_hash != manifest.artifact_hash:
                raise ValueError(f"existing artifact directory conflicts with {manifest.artifact_hash}")
            return destination

        expected_native = {payload.name: payload for payload in manifest.payloads if not payload.external}
        supplied = {str(name): Path(source) for name, source in (payload_files or {}).items()}
        unknown = sorted(set(supplied) - set(expected_native))
        missing = sorted(set(expected_native) - set(supplied))
        if unknown:
            raise ValueError("payload files not declared by manifest: " + ", ".join(unknown))
        if missing:
            raise ValueError("native manifest payloads missing from bundle write: " + ", ".join(missing))
        for name, source in supplied.items():
            if not source.exists() or (not source.is_file() and not source.is_dir()):
                raise FileNotFoundError(source)
            if sha256_path(source) != expected_native[name].sha256:
                raise ValueError(f"payload digest does not match manifest: {name}")

        stage_root = self.root / ".staging"
        stage_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f"{manifest.artifact_id}.", dir=stage_root))
        try:
            if supplied:
                for name, source in supplied.items():
                    target = temporary / "payloads" / str(name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source.is_dir():
                        if link_payloads:
                            try:
                                shutil.copytree(source, target, copy_function=os.link)
                            except OSError:
                                shutil.rmtree(target, ignore_errors=True)
                                shutil.copytree(source, target)
                        else:
                            shutil.copytree(source, target)
                    elif link_payloads:
                        try:
                            os.link(source, target)
                        except OSError:
                            # Scratch and quota roots may be different devices;
                            # preserve correctness if a hard-link is impossible.
                            shutil.copy2(source, target)
                    else:
                        shutil.copy2(source, target)
            _atomic_write_json(temporary / "manifest.json", manifest.to_dict())
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(temporary, destination)
            except FileExistsError:
                existing = ArtifactManifest.load(manifest_path)
                if existing.artifact_hash != manifest.artifact_hash:
                    raise
                shutil.rmtree(temporary, ignore_errors=True)
            self.record_lifecycle(
                manifest.artifact_hash,
                LifecycleState.SEALED,
                actor="circuitflow.bundle_writer",
                reason="immutable_bundle_written",
            )
            return destination
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def write_reference_bundle(self, manifest: ArtifactManifest) -> Path:
        """Seal an adapter bundle whose payloads remain verified external files."""

        if any(not payload.external for payload in manifest.payloads):
            raise ValueError("reference bundles require every payload to be external")
        return self.write_bundle(manifest)

    def load(self, artifact_hash: str, artifact_type: ArtifactType | str) -> ArtifactManifest:
        return ArtifactManifest.load(self.manifest_path(artifact_hash, artifact_type))

    def certify(
        self,
        manifest: ArtifactManifest,
        *,
        require_runtime_receipt: bool = False,
        semantic_issues: Sequence[str] = (),
        validator_id: str = "circuitflow.certify.v2",
    ) -> CertificationReceipt:
        bundle = self.bundle_dir(manifest)
        issues: list[str] = list(semantic_issues)
        for payload in manifest.payloads:
            path = payload.resolve(bundle)
            if path is None:
                if payload.external:
                    # Certified legacy adapters pin the original byte digest.
                    # They are verified in place at import and can be mirrored
                    # separately without making an immutable reference bundle
                    # fail solely because it points to an object store URI.
                    continue
                issues.append(f"remote_payload_unverified:{payload.name}")
                continue
            if not path.exists() or (not path.is_file() and not path.is_dir()):
                issues.append(f"missing_payload:{payload.name}")
                continue
            observed = sha256_path(path)
            if observed != payload.sha256:
                issues.append(f"payload_hash_mismatch:{payload.name}")
        if require_runtime_receipt:
            runtime_receipts = self.runtime_receipts(manifest.artifact_hash)
            legacy_runtime_payload = any(
                payload.name == "gpu_performance_receipt.json" for payload in manifest.payloads
            )
            if not runtime_receipts and not legacy_runtime_payload:
                issues.append("missing_runtime_receipt")
        receipt = CertificationReceipt(
            artifact_hash=manifest.artifact_hash,
            level="semantic" if not require_runtime_receipt else "semantic_and_runtime",
            passed=not issues,
            validator_id=validator_id,
            issues=tuple(sorted(set(issues))),
            details={"artifact_type": manifest.artifact_type.value, "source_kind": manifest.source_kind},
        )
        self.write_certification(receipt)
        current = infer_lifecycle_state(
            manifest,
            self.certification_receipts(manifest.artifact_hash),
            self.lifecycle_events(manifest.artifact_hash),
        )
        desired = LifecycleState.CERTIFIED if receipt.passed else LifecycleState.QUARANTINED
        # A repeat certification validates bytes again but should not regress an
        # artifact that has already advanced to analysis/claim readiness.
        if not (
            receipt.passed
            and current in {LifecycleState.CERTIFIED, LifecycleState.ANALYSIS_READY, LifecycleState.CLAIM_READY}
        ):
            self.record_lifecycle(
                manifest.artifact_hash,
                desired,
                actor=validator_id,
                reason="certification_passed" if receipt.passed else "certification_failed",
                details={"receipt_hash": receipt.receipt_hash, "issues": list(receipt.issues)},
            )
        return receipt

    def write_certification(self, receipt: CertificationReceipt) -> Path:
        target = self.root / "receipts" / "certifications" / receipt.artifact_hash / f"{receipt.receipt_hash}.json"
        _atomic_write_json(target, {**receipt.to_dict(), "receipt_hash": receipt.receipt_hash})
        return target

    def write_runtime_receipt(self, receipt: RuntimeReceipt) -> Path:
        """Persist a validated runtime receipt beside, never inside, a bundle."""

        issues = receipt.validate()
        if issues:
            raise ValueError("invalid CircuitFlow runtime receipt: " + "; ".join(issues))
        target = self.root / "receipts" / "runtime" / receipt.artifact_hash / f"{receipt.receipt_hash}.json"
        _atomic_write_json(target, {**receipt.to_dict(), "receipt_hash": receipt.receipt_hash})
        return target

    def runtime_receipts(self, artifact_hash: str) -> tuple[RuntimeReceipt, ...]:
        root = self.root / "receipts" / "runtime" / str(artifact_hash)
        if not root.is_dir():
            return tuple()
        receipts: list[RuntimeReceipt] = []
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
                if not isinstance(payload, Mapping):
                    continue
                receipt = RuntimeReceipt.from_mapping(payload)
                if receipt.receipt_hash != str(payload.get("receipt_hash", "")):
                    continue
                receipts.append(receipt)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return tuple(sorted(receipts, key=lambda receipt: (receipt.created_unix_seconds, receipt.receipt_hash)))

    def lifecycle_events(self, artifact_hash: str) -> tuple[LifecycleEvent, ...]:
        """Read valid append-only operational events for one artifact."""

        root = self.root / "receipts" / "lifecycle" / str(artifact_hash)
        if not root.is_dir():
            return tuple()
        events: list[LifecycleEvent] = []
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
                if not isinstance(payload, Mapping):
                    continue
                event = LifecycleEvent.from_mapping(payload)
                if event.event_hash != str(payload.get("event_hash", "")):
                    continue
                events.append(event)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return tuple(sorted(events, key=lambda event: (event.created_unix_seconds, event.event_hash)))

    def record_lifecycle(
        self,
        artifact_hash: str,
        state: LifecycleState,
        *,
        actor: str = "circuitflow",
        reason: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> LifecycleEvent:
        """Append a validated transition without changing artifact identity."""

        current_events = self.lifecycle_events(artifact_hash)
        previous = current_events[-1].state if current_events else None
        if previous == state:
            # Repeating the same operational fact is idempotent.  It avoids
            # duplicate event streams when a job retries an atomic writer.
            return current_events[-1]
        allowed = _LIFECYCLE_TRANSITIONS.get(previous, frozenset())
        if state not in allowed:
            previous_label = "none" if previous is None else previous.value
            raise ValueError(
                f"invalid CircuitFlow lifecycle transition {previous_label}->{state.value}"
            )
        event = LifecycleEvent(
            artifact_hash=str(artifact_hash),
            state=state,
            previous_state=previous,
            actor=str(actor),
            reason=str(reason),
            details=dict(details or {}),
        )
        issues = event.validate()
        if issues:
            raise ValueError("invalid CircuitFlow lifecycle event: " + "; ".join(issues))
        target = self.root / "receipts" / "lifecycle" / event.artifact_hash / f"{event.event_hash}.json"
        _atomic_write_json(target, {**event.to_dict(), "event_hash": event.event_hash})
        return event

    def mark_analysis_ready(self, manifest: ArtifactManifest, *, actor: str = "circuitflow") -> LifecycleEvent:
        """Mark a certified bundle usable by downstream canonical analysis."""

        if not any(receipt.passed for receipt in self.certification_receipts(manifest.artifact_hash)):
            raise ValueError("analysis-ready artifacts must first pass certification")
        current = infer_lifecycle_state(
            manifest,
            self.certification_receipts(manifest.artifact_hash),
            self.lifecycle_events(manifest.artifact_hash),
        )
        if current in {LifecycleState.ANALYSIS_READY, LifecycleState.CLAIM_READY}:
            events = self.lifecycle_events(manifest.artifact_hash)
            if events:
                return events[-1]
        if current != LifecycleState.CERTIFIED:
            raise ValueError("analysis-ready artifacts must be in the certified lifecycle state")
        return self.record_lifecycle(
            manifest.artifact_hash,
            LifecycleState.ANALYSIS_READY,
            actor=actor,
            reason="analysis_contract_validated",
        )

    def mark_claim_ready(self, manifest: ArtifactManifest, *, actor: str = "circuitflow") -> LifecycleEvent:
        """Mark a certified, claim-valid bundle after its declared gates pass."""

        if not manifest.valid_for_claims:
            raise ValueError("claim-ready lifecycle state requires manifest.valid_for_claims")
        current = infer_lifecycle_state(
            manifest,
            self.certification_receipts(manifest.artifact_hash),
            self.lifecycle_events(manifest.artifact_hash),
        )
        if current == LifecycleState.CLAIM_READY:
            events = self.lifecycle_events(manifest.artifact_hash)
            if events:
                return events[-1]
        if current == LifecycleState.CERTIFIED:
            self.mark_analysis_ready(manifest, actor=actor)
        elif current != LifecycleState.ANALYSIS_READY:
            raise ValueError("claim-ready artifacts must be certified and analysis-ready")
        return self.record_lifecycle(
            manifest.artifact_hash,
            LifecycleState.CLAIM_READY,
            actor=actor,
            reason="declared_claim_gates_passed",
        )

    def write_replica_receipt(self, receipt: ReplicaReceipt) -> Path:
        target = self.root / "receipts" / "replicas" / receipt.artifact_hash / f"{receipt.receipt_hash}.json"
        _atomic_write_json(target, {**receipt.to_dict(), "receipt_hash": receipt.receipt_hash})
        return target

    def certification_receipts(self, artifact_hash: str) -> tuple[CertificationReceipt, ...]:
        root = self.root / "receipts" / "certifications" / artifact_hash
        receipts: list[CertificationReceipt] = []
        if not root.is_dir():
            return tuple()
        for path in sorted(root.glob("*.json")):
            payload = json.loads(path.read_text())
            if not isinstance(payload, Mapping):
                continue
            receipts.append(
                CertificationReceipt(
                    artifact_hash=str(payload.get("artifact_hash", "")),
                    level=str(payload.get("level", "")),
                    passed=bool(payload.get("passed", False)),
                    validator_id=str(payload.get("validator_id", "")),
                    issues=tuple(str(value) for value in payload.get("issues", ()) or ()),
                    details=dict(payload.get("details", {}) or {}),
                    created_unix_seconds=float(payload.get("created_unix_seconds", 0.0)),
                    schema_name=str(payload.get("schema_name", "circuitflow_certification_receipt")),
                    schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
                )
            )
        return tuple(receipts)


def infer_lifecycle_state(
    manifest: ArtifactManifest,
    receipts: Sequence[CertificationReceipt],
    events: Sequence[LifecycleEvent] = (),
) -> LifecycleState:
    """Resolve current lifecycle from append-only events and receipt fallback.

    Receipt-only inference preserves compatibility with v2 bundles written
    before lifecycle events existed.  Once an event stream is present, its
    final state is authoritative because every transition was validated when
    appended.
    """

    del manifest  # Kept in the public signature for compatibility/readability.
    if events:
        return events[-1].state
    if any(not receipt.passed for receipt in receipts):
        return LifecycleState.QUARANTINED
    if any(receipt.passed for receipt in receipts):
        return LifecycleState.CERTIFIED
    return LifecycleState.SEALED


def payload_ref_for_external_file(
    *,
    name: str,
    path: str | Path,
    storage_format: str,
    media_type: str = "application/octet-stream",
    notes: Mapping[str, Any] | None = None,
) -> PayloadRef:
    return PayloadRef.from_file(
        name=name,
        path=path,
        uri=str(Path(path).resolve()),
        storage_format=storage_format,
        media_type=media_type,
        external=True,
        notes=notes,
    )


__all__ = [
    "ArtifactManifest",
    "ArtifactStore",
    "ArtifactType",
    "CertificationReceipt",
    "LifecycleEvent",
    "LifecycleState",
    "PayloadRef",
    "ReplicaReceipt",
    "RuntimeReceipt",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "infer_lifecycle_state",
    "payload_ref_for_external_file",
]
