"""Stable route coordinates and architecture-owned alias lowering."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from contracts.hashing import stable_digest


ROUTE_TOPOLOGY_COMPATIBILITY_SCHEMA = (
    "mechanistic_route_topology_compatibility_receipt_v1"
)


@dataclass(frozen=True)
class RouteCoordinate:
    """A module-path-independent coordinate for one executable route."""

    architecture_id: str
    topology_hash: str
    layer_id: str
    route_kind: str
    source_coordinate: tuple[int, ...]
    target_coordinate: tuple[int, ...]
    parameter_coordinate: tuple[int, ...] = tuple()
    schema_version: str = "route_coordinate_v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "architecture_id": self.architecture_id,
            "topology_hash": self.topology_hash,
            "layer_id": self.layer_id,
            "route_kind": self.route_kind,
            "source_coordinate": list(self.source_coordinate),
            "target_coordinate": list(self.target_coordinate),
            "parameter_coordinate": list(self.parameter_coordinate),
        }

    def validate(self) -> None:
        if not self.architecture_id or not self.layer_id or not self.route_kind:
            raise ValueError("route coordinate identity fields must be nonempty")
        if len(self.topology_hash) != 64:
            raise ValueError("route topology hash must be a SHA-256 digest")
        if not self.source_coordinate or not self.target_coordinate:
            raise ValueError("route endpoints must be nonempty coordinate tuples")
        if any(value < 0 for value in (*self.source_coordinate, *self.target_coordinate, *self.parameter_coordinate)):
            raise ValueError("route coordinate indices must be nonnegative")
        if self.schema_version != "route_coordinate_v1":
            raise ValueError("unsupported route coordinate schema")

    @property
    def route_id(self) -> str:
        self.validate()
        return f"route:v1:{stable_digest(self.to_dict())}"


@dataclass(frozen=True)
class RouteAlias:
    alias: str
    layer_id: str


class RouteAliasResolver:
    """Resolve framework module names only at an architecture boundary."""

    def __init__(self, aliases: Sequence[RouteAlias]):
        resolved: dict[str, str] = {}
        conflicts: set[str] = set()
        for row in aliases:
            alias = str(row.alias).strip()
            layer_id = str(row.layer_id).strip()
            if not alias or not layer_id:
                raise ValueError("route aliases and layer IDs must be nonempty")
            if alias in resolved and resolved[alias] != layer_id:
                conflicts.add(alias)
            resolved[alias] = layer_id
        if conflicts:
            raise ValueError(f"conflicting route aliases: {sorted(conflicts)}")
        self._aliases = resolved

    def resolve(self, alias: str) -> str:
        try:
            return self._aliases[str(alias)]
        except KeyError as error:
            raise ValueError(f"unresolved route layer alias: {alias!r}") from error

    @property
    def content_hash(self) -> str:
        return stable_digest({"schema_name": "route_alias_map_v1", "aliases": self._aliases})


class ArchitectureRouteAdapter(Protocol):
    """Architecture boundary for topology lowering and execution parity."""

    architecture_id: str
    topology_hash: str
    alias_resolver: RouteAliasResolver

    def lower_route(
        self,
        *,
        layer_alias: str,
        route_kind: str,
        source_coordinate: Sequence[int],
        target_coordinate: Sequence[int],
        parameter_coordinate: Sequence[int] = (),
    ) -> RouteCoordinate:
        ...

    def parity_metadata(self) -> Mapping[str, object]:
        ...


@dataclass(frozen=True)
class DeclarativeArchitectureAdapter:
    """Default adapter for architectures with a certified alias table."""

    architecture_id: str
    topology_hash: str
    alias_resolver: RouteAliasResolver
    parity_receipt_hash: str

    def __post_init__(self) -> None:
        if len(self.topology_hash) != 64 or len(self.parity_receipt_hash) != 64:
            raise ValueError("topology and parity receipts must be SHA-256 digests")

    def lower_route(
        self,
        *,
        layer_alias: str,
        route_kind: str,
        source_coordinate: Sequence[int],
        target_coordinate: Sequence[int],
        parameter_coordinate: Sequence[int] = (),
    ) -> RouteCoordinate:
        coordinate = RouteCoordinate(
            architecture_id=self.architecture_id,
            topology_hash=self.topology_hash,
            layer_id=self.alias_resolver.resolve(layer_alias),
            route_kind=str(route_kind),
            source_coordinate=tuple(int(value) for value in source_coordinate),
            target_coordinate=tuple(int(value) for value in target_coordinate),
            parameter_coordinate=tuple(int(value) for value in parameter_coordinate),
        )
        coordinate.validate()
        return coordinate

    def parity_metadata(self) -> Mapping[str, object]:
        return {
            "architecture_id": self.architecture_id,
            "topology_hash": self.topology_hash,
            "alias_map_hash": self.alias_resolver.content_hash,
            "parity_receipt_hash": self.parity_receipt_hash,
        }


def validate_route_axis(coordinates: Sequence[RouteCoordinate], *, topology_hash: str) -> tuple[str, ...]:
    if not coordinates:
        raise ValueError("route axis must be nonempty")
    for coordinate in coordinates:
        coordinate.validate()
        if coordinate.topology_hash != topology_hash:
            raise ValueError("route axis mixes incompatible topology hashes")
    route_ids = tuple(coordinate.route_id for coordinate in coordinates)
    if len(route_ids) != len(set(route_ids)):
        raise ValueError("route axis contains duplicate stable coordinates")
    return route_ids


@dataclass(frozen=True)
class RouteTopologyCompatibilityReceipt:
    """Certify one shared executable route universe across two model states."""

    reference_cell_id: str
    candidate_cell_id: str
    expected_prunable_route_blocks: int
    topology_hash: str
    route_schema_hash: str
    route_profile_id: str
    configured_parameter_layers: tuple[str, ...]
    fixed_readout_layers: tuple[str, ...]
    reference_model_state_hash: str
    candidate_model_state_hash: str
    checks: Mapping[str, bool]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures and all(bool(value) for value in self.checks.values())

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_name": ROUTE_TOPOLOGY_COMPATIBILITY_SCHEMA,
            "reference_cell_id": self.reference_cell_id,
            "candidate_cell_id": self.candidate_cell_id,
            "expected_prunable_route_blocks": self.expected_prunable_route_blocks,
            "topology_hash": self.topology_hash,
            "route_schema_hash": self.route_schema_hash,
            "route_profile_id": self.route_profile_id,
            "configured_parameter_layers": list(self.configured_parameter_layers),
            "fixed_readout_layers": list(self.fixed_readout_layers),
            "reference_model_state_hash": self.reference_model_state_hash,
            "candidate_model_state_hash": self.candidate_model_state_hash,
            "checks": dict(self.checks),
            "failure_count": len(self.failures),
            "failures": list(self.failures),
            "passed": self.passed,
        }
        payload["receipt_hash"] = stable_digest(payload)
        return payload

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


def certify_route_topology_compatibility(
    *,
    reference_report: Mapping[str, object],
    candidate_report: Mapping[str, object],
    expected_prunable_route_blocks: int,
) -> RouteTopologyCompatibilityReceipt:
    """Verify topology identity without requiring parameter-state identity."""

    expected = int(expected_prunable_route_blocks)
    if expected <= 0:
        raise ValueError("expected prunable route-block count must be positive")
    reference_profile = reference_report.get("profile_audit")
    candidate_profile = candidate_report.get("profile_audit")
    if not isinstance(reference_profile, Mapping) or not isinstance(
        candidate_profile, Mapping
    ):
        raise ValueError("topology certification requires both profile audits")

    reference_layers = tuple(
        str(value) for value in reference_profile.get("configured_parameter_layers", ())
    )
    candidate_layers = tuple(
        str(value) for value in candidate_profile.get("configured_parameter_layers", ())
    )
    reference_fixed = tuple(
        str(value) for value in reference_profile.get("fixed_readout_layers", ())
    )
    candidate_fixed = tuple(
        str(value) for value in candidate_profile.get("fixed_readout_layers", ())
    )
    reference_topology = str(reference_report.get("topology_hash", ""))
    candidate_topology = str(candidate_report.get("topology_hash", ""))
    reference_schema = str(reference_report.get("route_schema_hash", ""))
    candidate_schema = str(candidate_report.get("route_schema_hash", ""))
    reference_profile_id = str(reference_profile.get("route_profile_id", ""))
    candidate_profile_id = str(candidate_profile.get("route_profile_id", ""))
    reference_state = str(reference_report.get("model_state_hash", ""))
    candidate_state = str(candidate_report.get("model_state_hash", ""))
    reference_prunable = int(reference_profile.get("canonical_prunable_route_block_count", -1))
    candidate_prunable = int(candidate_profile.get("canonical_prunable_route_block_count", -1))
    reference_total = int(reference_profile.get("canonical_route_block_count", -1))
    candidate_total = int(candidate_profile.get("canonical_route_block_count", -1))

    checks = {
        "topology_hash_equal": (
            len(reference_topology) == 64 and reference_topology == candidate_topology
        ),
        "route_schema_hash_equal": (
            len(reference_schema) == 64 and reference_schema == candidate_schema
        ),
        "route_profile_equal": (
            bool(reference_profile_id) and reference_profile_id == candidate_profile_id
        ),
        "prunable_route_block_count_equal_expected": (
            reference_prunable == candidate_prunable == expected
        ),
        "total_route_block_count_equal": (
            reference_total == candidate_total == expected + len(reference_fixed)
        ),
        "ordered_parameter_layers_equal": (
            bool(reference_layers) and reference_layers == candidate_layers
        ),
        "fixed_readout_layers_equal": reference_fixed == candidate_fixed,
        "model_states_distinct": (
            len(reference_state) == 64
            and len(candidate_state) == 64
            and reference_state != candidate_state
        ),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return RouteTopologyCompatibilityReceipt(
        reference_cell_id=str(reference_report.get("cell_id", "")),
        candidate_cell_id=str(candidate_report.get("cell_id", "")),
        expected_prunable_route_blocks=expected,
        topology_hash=reference_topology,
        route_schema_hash=reference_schema,
        route_profile_id=reference_profile_id,
        configured_parameter_layers=reference_layers,
        fixed_readout_layers=reference_fixed,
        reference_model_state_hash=reference_state,
        candidate_model_state_hash=candidate_state,
        checks=checks,
        failures=failures,
    )


__all__ = [
    "ArchitectureRouteAdapter",
    "DeclarativeArchitectureAdapter",
    "RouteAlias",
    "RouteAliasResolver",
    "RouteCoordinate",
    "ROUTE_TOPOLOGY_COMPATIBILITY_SCHEMA",
    "RouteTopologyCompatibilityReceipt",
    "certify_route_topology_compatibility",
    "validate_route_axis",
]
