"""Parity-gated native authority for the frozen sampled-path algorithm.

The native tracer is deliberately not promoted merely because its component
oracles pass.  A strict receipt compares complete paths, endpoints, and mask
identities against the sealed production delegate before claim-bearing use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .final_validation import NativeSampledPathFinalValidationReceipt

from circuitflow.artifacts import ArtifactManifest, ArtifactStore
from contracts.hashing import stable_digest
from mechanistic_systems.contracts import (
    CircuitPlan,
    SAMPLED_PATH_GROUP_REFERENCE_POLICY,
)
from mechanistic_systems.evidence import GlocalDeploymentMaskCertificate
from mechanistic_systems.inference.sequential_bce import (
    IID_WITH_REPLACEMENT_SAMPLING_ID,
    SEQUENTIAL_CALIBRATED_BCE_V2,
    SEQUENTIAL_CALIBRATED_BCE_V3,
)

from ..core.architectures import historical_route_alias

# Panel-level algorithm identifiers. The grid engine remains the pinned
# contract for claim-bearing and frozen plans. The BCE-precision engine is the
# exploratory default; continuous-v2 remains available by explicit identity.
SAMPLED_PATH_PANEL_GRID_ALGORITHM = "statistical_sequential_symmetric_frontier"
EVIDENCE_CONTINUOUS_ALGORITHM = "evidence_foundation_continuous_v2"
BCE_PRECISION_ALGORITHM = "evidence_foundation_bce_precision_v1"
from .sampled_path import (
    SAMPLED_PATH_FRONTIER_V1,
    SampledComparisonRequest,
    SampledPathEvaluationBackend,
    SampledPathPolicy,
    SampledReferenceEvidence,
    SampledPathResult,
    fold_warm_start,
    sampled_path_mask_hash,
    trace_statistical_symmetric_sampled_path,
)
from ..core.storage import (
    BlockBitPackedCircuitSet,
    RouteBlockSpec,
    materialize_circuit_artifact,
    ordered_block_route_axis_hash,
)


NATIVE_SAMPLED_PATH_SIGNATURE_SCHEMA = "mechanistic_native_sampled_path_signature_v1"
NATIVE_SAMPLED_PATH_PARITY_SCHEMA = "mechanistic_native_sampled_path_parity_receipt_v1"
NATIVE_SAMPLED_PATH_PROMOTION_SCHEMA = (
    "mechanistic_native_sampled_path_panel_promotion_receipt_v1"
)
INITIAL_DENSE_ANCHOR_NUMERICAL_TOLERANCE = 1e-4


@dataclass(frozen=True)
class SampledPathRouteGroupBinding:
    """Bind one frozen profile group to one stable route-coordinate block."""

    profile_group_name: str
    layer_id: str
    module_path: str
    route_kind: str
    route_group_kind: str
    shape: tuple[int, ...]

    def validate(self) -> None:
        if not self.profile_group_name or not self.layer_id or not self.module_path:
            raise ValueError("sampled-path route-group identity must be nonempty")
        if self.route_kind not in {
            "conv_channel_pair",
            # Kernel-tap granularity walks the SAME registered
            # conv_like density-fraction ladder — fractions are
            # granularity-agnostic and no new numeral is introduced.
            "conv_kernel_weight",
            "linear_weight",
            "residual_stream_gate",
        }:
            raise ValueError("sampled-path route group has an unsupported route kind")
        expected_group = (
            "linear_like" if self.route_kind == "linear_weight" else "conv_like"
        )
        if self.route_group_kind != expected_group:
            raise ValueError("sampled-path route group has the wrong density family")
        if not self.shape or any(int(value) <= 0 for value in self.shape):
            raise ValueError("sampled-path route-group shape must be positive")


def compile_sampled_path_route_groups(
    *,
    profile: Any,
    bound_architecture: Any,
    measurement_route_blocks: Sequence[RouteBlockSpec],
) -> tuple[SampledPathRouteGroupBinding, ...]:
    """Resolve historical executable aliases onto stable canonical layer IDs.

    The accepted supervised v1 panel consists of singleton groups. Multi-route
    groups remain rejected until they receive their own parity evidence rather
    than being silently flattened into a different sequential experiment.
    """

    profile.validate()
    bound_architecture.validate()
    bindings = tuple(bound_architecture.circuit_bindings)
    blocks = tuple(bound_architecture.circuit_route_blocks)
    if len(bindings) != len(blocks):
        raise ValueError("bound architecture route bindings and blocks differ")
    by_alias = {
        historical_route_alias(binding): (binding, block)
        for binding, block in zip(bindings, blocks, strict=True)
    }
    names = tuple(str(value) for value in profile.layer_group_names)
    if set(names) != set(by_alias) or len(names) != len(by_alias):
        raise ValueError(
            "sampled_path_frontier_v1 profile groups do not form an exact "
            "singleton partition of the prunable route axis"
        )
    measured = {str(block.layer_id): block for block in measurement_route_blocks}
    required_layers = {str(block.layer_id) for block in blocks}
    allowed_layers = {str(block.layer_id) for block in bound_architecture.route_blocks}
    if not required_layers.issubset(measured) or not set(measured).issubset(
        allowed_layers
    ):
        raise ValueError("measurement route blocks differ from the bound route axis")
    groups: list[SampledPathRouteGroupBinding] = []
    for profile_name in names:
        binding, block = by_alias[profile_name]
        observed = measured[str(block.layer_id)]
        if observed.to_dict() != block.to_dict() or str(observed.route_kind) != str(
            binding.route_kind
        ):
            raise ValueError(
                f"measurement block differs from executable route {block.layer_id!r}"
            )
        group = SampledPathRouteGroupBinding(
            profile_group_name=profile_name,
            layer_id=str(block.layer_id),
            module_path=str(binding.module_path),
            route_kind=str(binding.route_kind),
            route_group_kind=(
                "linear_like"
                if str(binding.route_kind) == "linear_weight"
                else "conv_like"
            ),
            shape=tuple(int(value) for value in block.shape),
        )
        group.validate()
        groups.append(group)
    return tuple(groups)


def _normalized_action(value: str, *, accepted: bool) -> str:
    action = str(value or "").strip().lower()
    aliases = {
        "accept_commit": "accept",
        "commit_frontier_prefix": "accept",
        "reject_stop": "reject",
        "full_extract_resolve": "sample_more",
    }
    if action in aliases:
        return aliases[action]
    if action in {"accept", "reject", "sample_more"}:
        return action
    return "accept" if accepted else "reject"


def _schedule_index(schedule: Sequence[float], value: float) -> int:
    matches = [
        index
        for index, candidate in enumerate(schedule)
        if math.isclose(float(candidate), float(value), rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"density {value!r} is not unique on its declared schedule")
    return int(matches[0])


@dataclass(frozen=True)
class NativeSampledPathTraceRow:
    partition_id: str
    previous_density: float
    proposed_density: float
    accepted: bool
    action: str
    posterior_mean_sensitivity: float
    posterior_sd_sensitivity: float
    context_count: int
    stage_index: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "NativeSampledPathTraceRow":
        accepted = bool(payload["accepted"])
        return cls(
            partition_id=str(payload["partition_id"]),
            previous_density=float(payload["previous_density"]),
            proposed_density=float(payload["proposed_density"]),
            accepted=accepted,
            action=_normalized_action(
                str(payload.get("action", "")), accepted=accepted
            ),
            posterior_mean_sensitivity=float(payload["posterior_mean_sensitivity"]),
            posterior_sd_sensitivity=float(payload["posterior_sd_sensitivity"]),
            context_count=int(payload["context_count"]),
            stage_index=int(payload["stage_index"]),
        )


@dataclass(frozen=True)
class NativeSampledPathTraceSignature:
    profile_hash: str
    query_id: str
    layer_id: str
    positive_schedule: tuple[float, ...]
    negative_schedule: tuple[float, ...]
    final_positive_density: float
    final_negative_density: float
    final_mask_hash: str
    committed_pair_indices: tuple[tuple[int, int], ...]
    rows: tuple[NativeSampledPathTraceRow, ...]
    algorithm_version: str = SAMPLED_PATH_FRONTIER_V1
    schema_name: str = NATIVE_SAMPLED_PATH_SIGNATURE_SCHEMA

    def validate(self) -> None:
        if len(self.profile_hash) != 64:
            raise ValueError("native sampled-path signature requires a profile hash")
        if not self.query_id or not self.layer_id:
            raise ValueError(
                "native sampled-path signature requires query and layer IDs"
            )
        for schedule in (self.positive_schedule, self.negative_schedule):
            if (
                not schedule
                or schedule[0] != 1.0
                or any(left <= right for left, right in zip(schedule, schedule[1:]))
            ):
                raise ValueError(
                    "native sampled-path schedules must start at one and decrease"
                )
        if len(self.final_mask_hash) != 64:
            raise ValueError("native sampled-path signature requires a final mask hash")
        if not self.committed_pair_indices or self.committed_pair_indices[0] != (0, 0):
            raise ValueError("committed path must begin at the dense pair")
        if self.algorithm_version != SAMPLED_PATH_FRONTIER_V1:
            raise ValueError("native signature must identify sampled_path_frontier_v1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "profile_hash": self.profile_hash,
            "query_id": self.query_id,
            "layer_id": self.layer_id,
            "positive_schedule": list(self.positive_schedule),
            "negative_schedule": list(self.negative_schedule),
            "final_positive_density": self.final_positive_density,
            "final_negative_density": self.final_negative_density,
            "final_mask_hash": self.final_mask_hash,
            "committed_pair_indices": [
                list(pair) for pair in self.committed_pair_indices
            ],
            "rows": [asdict(row) for row in self.rows],
            "algorithm_version": self.algorithm_version,
        }

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "NativeSampledPathTraceSignature":
        if str(payload.get("schema_name", "")) != NATIVE_SAMPLED_PATH_SIGNATURE_SCHEMA:
            raise ValueError("unsupported native sampled-path signature schema")
        result = cls(
            profile_hash=str(payload["profile_hash"]),
            query_id=str(payload["query_id"]),
            layer_id=str(payload["layer_id"]),
            positive_schedule=tuple(
                float(value) for value in payload["positive_schedule"]
            ),
            negative_schedule=tuple(
                float(value) for value in payload["negative_schedule"]
            ),
            final_positive_density=float(payload["final_positive_density"]),
            final_negative_density=float(payload["final_negative_density"]),
            final_mask_hash=str(payload["final_mask_hash"]),
            committed_pair_indices=tuple(
                (int(pair[0]), int(pair[1]))
                for pair in payload["committed_pair_indices"]
            ),
            rows=tuple(
                NativeSampledPathTraceRow.from_mapping(row) for row in payload["rows"]
            ),
            algorithm_version=str(payload.get("algorithm_version", "")),
        )
        result.validate()
        return result


def signature_from_native_result(
    result: SampledPathResult,
    *,
    profile_hash: str,
    query_id: str,
    layer_id: str,
    positive_schedule: Sequence[float],
    negative_schedule: Sequence[float],
) -> NativeSampledPathTraceSignature:
    positive = tuple(float(value) for value in positive_schedule)
    negative = tuple(float(value) for value in negative_schedule)
    cursors = {"positive": 0, "negative": 0}
    committed: list[tuple[int, int]] = [(0, 0)]
    rows: list[NativeSampledPathTraceRow] = []
    for step in result.steps:
        partition = str(step.partition_id)
        rows.append(
            NativeSampledPathTraceRow(
                partition_id=partition,
                previous_density=float(step.previous_density),
                proposed_density=float(step.proposed_density),
                accepted=bool(step.accepted),
                action=_normalized_action(step.decision.action, accepted=step.accepted),
                posterior_mean_sensitivity=float(
                    step.decision.posterior_mean_sensitivity
                ),
                posterior_sd_sensitivity=float(step.decision.posterior_sd_sensitivity),
                context_count=int(step.decision.context_count),
                stage_index=int(step.decision.stage_index),
            )
        )
        if step.accepted:
            schedule = positive if partition == "positive" else negative
            cursors[partition] = _schedule_index(schedule, step.proposed_density)
            committed.append((cursors["positive"], cursors["negative"]))
    signature = NativeSampledPathTraceSignature(
        profile_hash=str(profile_hash),
        query_id=str(query_id),
        layer_id=str(layer_id),
        positive_schedule=positive,
        negative_schedule=negative,
        final_positive_density=float(result.positive_density),
        final_negative_density=float(result.negative_density),
        final_mask_hash=sampled_path_mask_hash(result.committed_masks),
        committed_pair_indices=tuple(committed),
        rows=tuple(rows),
    )
    signature.validate()
    return signature


def signature_from_historical_trace(
    *,
    profile_hash: str,
    query_id: str,
    layer_id: str,
    positive_schedule: Sequence[float],
    negative_schedule: Sequence[float],
    final_masks: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> NativeSampledPathTraceSignature:
    """Adapt a sealed live-session trace into the canonical comparison shape."""

    positive = tuple(float(value) for value in positive_schedule)
    negative = tuple(float(value) for value in negative_schedule)
    committed_pairs = tuple(
        (int(pair[0]), int(pair[1]))
        for pair in metadata[
            "live_target_preservation_frontier_committed_pair_sequence"
        ]
    )
    cursors = {"positive": 0, "negative": 0}
    rows: list[NativeSampledPathTraceRow] = []
    for raw in metadata["live_target_preservation_frontier_trace"]:
        turn = str(raw["turn"]).strip().lower()
        partition = "negative" if turn in {"neg", "negative"} else "positive"
        candidate_pair = raw["candidate_pair"]
        proposed = float(candidate_pair[1 if partition == "negative" else 0])
        schedule = negative if partition == "negative" else positive
        previous = float(schedule[cursors[partition]])
        accepted = bool(raw.get("committed", False))
        rows.append(
            NativeSampledPathTraceRow(
                partition_id=partition,
                previous_density=previous,
                proposed_density=proposed,
                accepted=accepted,
                action=_normalized_action(
                    str(raw.get("statistical_decision", "")),
                    accepted=accepted,
                ),
                posterior_mean_sensitivity=float(
                    raw.get(
                        "posterior_mean_step_gain",
                        raw.get("target_preservation_frontier_step_gain", math.nan),
                    )
                ),
                posterior_sd_sensitivity=float(
                    raw.get("posterior_sd_step_gain", math.nan)
                ),
                context_count=int(raw.get("statistical_sample_count", -1)),
                stage_index=int(raw.get("statistical_stage_index", -1)),
            )
        )
        if accepted:
            cursors[partition] = _schedule_index(schedule, proposed)
    selected = metadata["selected_pair"]
    signature = NativeSampledPathTraceSignature(
        profile_hash=str(profile_hash),
        query_id=str(query_id),
        layer_id=str(layer_id),
        positive_schedule=positive,
        negative_schedule=negative,
        final_positive_density=float(selected[0]),
        final_negative_density=float(selected[1]),
        final_mask_hash=sampled_path_mask_hash(
            {
                str(key): np.asarray(value, dtype=bool)
                for key, value in final_masks.items()
            }
        ),
        committed_pair_indices=committed_pairs,
        rows=tuple(rows),
    )
    signature.validate()
    return signature


@dataclass(frozen=True)
class NativeSampledPathParityReceipt:
    profile_hash: str
    reference_signature_hash: str
    candidate_signature_hash: str
    issues: tuple[str, ...]
    absolute_tolerance: float
    relative_tolerance: float
    comparison_scope: str
    schema_name: str = NATIVE_SAMPLED_PATH_PARITY_SCHEMA

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "profile_hash": self.profile_hash,
            "reference_signature_hash": self.reference_signature_hash,
            "candidate_signature_hash": self.candidate_signature_hash,
            "issues": list(self.issues),
            "passed": self.passed,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "comparison_scope": self.comparison_scope,
            "algorithm_version": SAMPLED_PATH_FRONTIER_V1,
        }

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())

    def require_passed(
        self,
        *,
        profile_hash: str | None = None,
        query_id: str | None = None,
    ) -> None:
        if not self.passed:
            raise ValueError(
                "native sampled-path parity failed: " + ";".join(self.issues)
            )
        if profile_hash is not None and self.profile_hash != str(profile_hash):
            raise ValueError("native sampled-path receipt profile differs from the run")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "NativeSampledPathParityReceipt":
        if str(payload.get("schema_name", "")) != NATIVE_SAMPLED_PATH_PARITY_SCHEMA:
            raise ValueError("unsupported native sampled-path parity receipt schema")
        receipt = cls(
            profile_hash=str(payload["profile_hash"]),
            reference_signature_hash=str(payload["reference_signature_hash"]),
            candidate_signature_hash=str(payload["candidate_signature_hash"]),
            issues=tuple(str(issue) for issue in payload.get("issues", ())),
            absolute_tolerance=float(payload["absolute_tolerance"]),
            relative_tolerance=float(payload["relative_tolerance"]),
            comparison_scope=str(payload["comparison_scope"]),
        )
        if bool(payload.get("passed", receipt.passed)) != receipt.passed:
            raise ValueError("native sampled-path receipt pass flag is inconsistent")
        return receipt


def certify_native_sampled_path_parity(
    reference: NativeSampledPathTraceSignature,
    candidate: NativeSampledPathTraceSignature,
    *,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-8,
    comparison_scope: str = "single_group_query",
) -> NativeSampledPathParityReceipt:
    """Compare complete native and sealed-delegate traces."""

    reference.validate()
    candidate.validate()
    issues: list[str] = []
    exact_fields = (
        "profile_hash",
        "query_id",
        "layer_id",
        "positive_schedule",
        "negative_schedule",
        "final_mask_hash",
        "committed_pair_indices",
        "algorithm_version",
    )
    for field_name in exact_fields:
        if getattr(reference, field_name) != getattr(candidate, field_name):
            issues.append(f"{field_name}_mismatch")
    for field_name in ("final_positive_density", "final_negative_density"):
        if not math.isclose(
            float(getattr(reference, field_name)),
            float(getattr(candidate, field_name)),
            abs_tol=float(absolute_tolerance),
            rel_tol=float(relative_tolerance),
        ):
            issues.append(f"{field_name}_mismatch")
    if len(reference.rows) != len(candidate.rows):
        issues.append("proposal_count_mismatch")
    for index, (expected, observed) in enumerate(zip(reference.rows, candidate.rows)):
        prefix = f"proposal_{index}"
        for field_name in (
            "partition_id",
            "accepted",
            "action",
            "context_count",
            "stage_index",
        ):
            if getattr(expected, field_name) != getattr(observed, field_name):
                issues.append(f"{prefix}_{field_name}_mismatch")
        for field_name in (
            "previous_density",
            "proposed_density",
            "posterior_mean_sensitivity",
            "posterior_sd_sensitivity",
        ):
            left = float(getattr(expected, field_name))
            right = float(getattr(observed, field_name))
            if not (
                math.isfinite(left)
                and math.isfinite(right)
                and math.isclose(
                    left,
                    right,
                    abs_tol=float(absolute_tolerance),
                    rel_tol=float(relative_tolerance),
                )
            ):
                issues.append(f"{prefix}_{field_name}_mismatch")
    return NativeSampledPathParityReceipt(
        profile_hash=reference.profile_hash,
        reference_signature_hash=reference.content_hash,
        candidate_signature_hash=candidate.content_hash,
        issues=tuple(dict.fromkeys(issues)),
        absolute_tolerance=float(absolute_tolerance),
        relative_tolerance=float(relative_tolerance),
        comparison_scope=str(comparison_scope),
    )


def write_native_sampled_path_parity_receipt(
    receipt: NativeSampledPathParityReceipt,
    path: str | Path,
) -> Path:
    """Atomically persist a parity gate receipt."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(receipt.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def load_native_sampled_path_parity_receipt(
    path: str | Path,
) -> NativeSampledPathParityReceipt:
    return NativeSampledPathParityReceipt.from_mapping(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


@dataclass(frozen=True)
class NativeSampledPathPromotionReceipt:
    """Panel-level authority for the immutable native sampled-path backend.

    Exact trace identity is sufficient but not necessary. A content-addressed
    numerical-tie adjudication may also promote the algorithm/configuration
    backend while explicitly declining to claim identical realized circuits.
    """

    profile_hash: str
    target_ids: tuple[str, ...]
    exact_or_tolerance_target_ids: tuple[str, ...]
    numerical_tie_target_ids: tuple[str, ...]
    evidence_hash: str
    strict_replay_passed: bool
    numerical_replay_passed: bool
    exact_circuit_identity: bool
    issues: tuple[str, ...] = tuple()
    acceptance_mode: str = "numerical_tie_adjudication"
    comparison_scope: str = "supervised_cluster_trace_panel"
    schema_name: str = NATIVE_SAMPLED_PATH_PROMOTION_SCHEMA

    @property
    def passed(self) -> bool:
        return bool(
            not self.issues
            and (self.strict_replay_passed or self.numerical_replay_passed)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "profile_hash": self.profile_hash,
            "target_ids": list(self.target_ids),
            "exact_or_tolerance_target_ids": list(self.exact_or_tolerance_target_ids),
            "numerical_tie_target_ids": list(self.numerical_tie_target_ids),
            "evidence_hash": self.evidence_hash,
            "strict_replay_passed": self.strict_replay_passed,
            "numerical_replay_passed": self.numerical_replay_passed,
            "exact_circuit_identity": self.exact_circuit_identity,
            "issues": list(self.issues),
            "acceptance_mode": self.acceptance_mode,
            "comparison_scope": self.comparison_scope,
            "algorithm_version": SAMPLED_PATH_FRONTIER_V1,
            "passed": self.passed,
        }

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())

    def require_passed(
        self,
        *,
        profile_hash: str | None = None,
        query_id: str | None = None,
    ) -> None:
        if not self.passed:
            raise ValueError(
                "native sampled-path panel promotion failed: " + ";".join(self.issues)
            )
        if profile_hash is not None and self.profile_hash != str(profile_hash):
            raise ValueError(
                "native sampled-path promotion profile differs from the run"
            )
        if query_id is not None and str(query_id) not in self.target_ids:
            raise ValueError("native sampled-path query is outside the promoted panel")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "NativeSampledPathPromotionReceipt":
        if str(payload.get("schema_name", "")) != NATIVE_SAMPLED_PATH_PROMOTION_SCHEMA:
            raise ValueError("unsupported native sampled-path promotion receipt schema")
        receipt = cls(
            profile_hash=str(payload.get("profile_hash", "")),
            target_ids=tuple(str(value) for value in payload.get("target_ids", ())),
            exact_or_tolerance_target_ids=tuple(
                str(value) for value in payload.get("exact_or_tolerance_target_ids", ())
            ),
            numerical_tie_target_ids=tuple(
                str(value) for value in payload.get("numerical_tie_target_ids", ())
            ),
            evidence_hash=str(payload.get("evidence_hash", "")),
            strict_replay_passed=bool(payload.get("strict_replay_passed", False)),
            numerical_replay_passed=bool(payload.get("numerical_replay_passed", False)),
            exact_circuit_identity=bool(payload.get("exact_circuit_identity", False)),
            issues=tuple(str(value) for value in payload.get("issues", ())),
            acceptance_mode=str(payload.get("acceptance_mode", "")),
            comparison_scope=str(payload.get("comparison_scope", "")),
        )
        if bool(payload.get("passed", receipt.passed)) != receipt.passed:
            raise ValueError("native sampled-path promotion pass flag is inconsistent")
        if (
            str(payload.get("algorithm_version", SAMPLED_PATH_FRONTIER_V1))
            != SAMPLED_PATH_FRONTIER_V1
        ):
            raise ValueError("native sampled-path promotion algorithm version differs")
        return receipt


def promotion_receipt_from_numerical_adjudication(
    adjudication: Mapping[str, Any],
    *,
    evidence_hash: str,
    expected_profile_hash: str,
    expected_target_ids: Sequence[str | int],
) -> NativeSampledPathPromotionReceipt:
    """Validate one panel-level numerical-tie adjudication without broadening it."""

    issues: list[str] = []
    expected = tuple(str(value) for value in expected_target_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("sampled-path promotion requires unique expected targets")
    if len(str(evidence_hash)) != 64:
        raise ValueError("sampled-path promotion evidence requires a SHA-256 hash")
    if str(adjudication.get("schema_name", "")) != (
        "sampled_path_frontier_v1_numerical_parity_adjudication_v1"
    ):
        issues.append("adjudication_schema_mismatch")
    if adjudication.get("passed") is not True:
        issues.append("adjudication_not_passed")
    if str(adjudication.get("status", "")) != "pass_with_numerical_tie":
        issues.append("adjudication_status_mismatch")
    if (
        str(adjudication.get("scope", ""))
        != "algorithm_and_configuration_backend_parity"
    ):
        issues.append("adjudication_scope_mismatch")

    scientific = adjudication.get("scientific_profile", {})
    if not isinstance(scientific, Mapping):
        scientific = {}
        issues.append("scientific_profile_missing")
    source_profile = str(scientific.get("source_profile_hash", ""))
    replay_profile = str(scientific.get("replay_profile_hash", ""))
    if source_profile != replay_profile or source_profile != str(expected_profile_hash):
        issues.append("scientific_profile_hash_mismatch")
    for field in (
        "sampled_path_semantics_equal",
        "calibration_and_extraction_roles_equal",
        "density_ladders_equal",
        "full_layer_slope_denominator_equal",
        "warm_start_and_final_validation_semantics_equal",
    ):
        if scientific.get(field) is not True:
            issues.append(f"scientific_profile_{field}_not_certified")

    summary = adjudication.get("target_summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
        issues.append("target_summary_missing")
    exact = tuple(
        str(value) for value in summary.get("exact_or_tolerance_pass_targets", ())
    )
    raw_ties = summary.get("numerical_tie_targets")
    if raw_ties is None:
        raw_ties = (summary.get("numerical_tie_target"),)
    elif isinstance(raw_ties, (str, int)):
        raw_ties = (raw_ties,)
    ties = tuple(str(value) for value in raw_ties if value is not None)
    if set(exact) & set(ties):
        issues.append("target_evidence_axes_overlap")
    if set((*exact, *ties)) != set(expected) or len((*exact, *ties)) != len(expected):
        issues.append("target_evidence_axis_mismatch")
    if not ties:
        issues.append("numerical_tie_target_missing")

    strict = adjudication.get("strict_mechanical_receipt", {})
    strict_passed = bool(
        strict.get("passed", False) if isinstance(strict, Mapping) else False
    )
    return NativeSampledPathPromotionReceipt(
        profile_hash=str(expected_profile_hash),
        target_ids=expected,
        exact_or_tolerance_target_ids=exact,
        numerical_tie_target_ids=ties,
        evidence_hash=str(evidence_hash),
        strict_replay_passed=strict_passed,
        numerical_replay_passed=not issues,
        exact_circuit_identity=bool(strict_passed and not ties),
        issues=tuple(dict.fromkeys(issues)),
    )


def load_numerical_adjudication_promotion_receipt(
    path: str | Path,
    *,
    expected_profile_hash: str,
    expected_target_ids: Sequence[str | int],
) -> NativeSampledPathPromotionReceipt:
    source = Path(path)
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("sampled-path numerical adjudication must be a JSON object")
    return promotion_receipt_from_numerical_adjudication(
        payload,
        evidence_hash=hashlib.sha256(raw).hexdigest(),
        expected_profile_hash=expected_profile_hash,
        expected_target_ids=expected_target_ids,
    )


def write_native_sampled_path_promotion_receipt(
    receipt: NativeSampledPathPromotionReceipt,
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(receipt.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def load_native_sampled_path_promotion_receipt(
    path: str | Path,
) -> NativeSampledPathPromotionReceipt:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("native sampled-path promotion receipt must be a JSON object")
    return NativeSampledPathPromotionReceipt.from_mapping(payload)


@dataclass(frozen=True)
class NativeSampledPathRun:
    result: SampledPathResult
    signature: NativeSampledPathTraceSignature
    maturity: str
    analysis_status: str
    parity_receipt_hash: str | None
    reference_anchor_loss: float | None = None
    reference_anchor_evidence_hash: str | None = None
    reference_anchor_mask_hash: str | None = None
    reference_anchor_panel_hash: str | None = None


def run_parity_gated_native_sampled_path(
    *,
    backend: SampledPathEvaluationBackend,
    committed_masks: Mapping[str, np.ndarray],
    build_candidate: Any,
    query_id: str,
    layer_id: str,
    positive_schedule: Sequence[float],
    negative_schedule: Sequence[float],
    total_layer_route_count: int,
    dense_anchor_loss: float,
    fully_pruned_anchor_loss: float,
    sensitivity_threshold: float,
    policy: SampledPathPolicy,
    profile_hash: str,
    parity_receipt: (
        NativeSampledPathParityReceipt | NativeSampledPathPromotionReceipt | None
    ) = None,
    claim_bearing: bool = False,
    initial_committed_warm_start: Mapping[str, Any] | None = None,
    reference_anchor_evidence_hash: str | None = None,
    reference_anchor_mask_hash: str | None = None,
    reference_anchor_panel_hash: str | None = None,
    initial_turn: str = "negative",
) -> NativeSampledPathRun:
    """Run native v1 while enforcing an explicit promotion boundary."""

    if claim_bearing:
        if parity_receipt is None:
            raise ValueError(
                "claim-bearing native sampled paths require a parity receipt"
            )
        parity_receipt.require_passed(
            profile_hash=profile_hash,
            query_id=query_id,
        )
    result = trace_statistical_symmetric_sampled_path(
        backend=backend,
        committed_masks=committed_masks,
        build_candidate=build_candidate,
        query_id=query_id,
        layer_id=layer_id,
        positive_schedule=positive_schedule,
        negative_schedule=negative_schedule,
        total_layer_route_count=total_layer_route_count,
        dense_anchor_loss=dense_anchor_loss,
        fully_pruned_anchor_loss=fully_pruned_anchor_loss,
        sensitivity_threshold=sensitivity_threshold,
        policy=policy,
        initial_committed_warm_start=initial_committed_warm_start,
        initial_turn=initial_turn,
    )
    signature = signature_from_native_result(
        result,
        profile_hash=profile_hash,
        query_id=query_id,
        layer_id=layer_id,
        positive_schedule=positive_schedule,
        negative_schedule=negative_schedule,
    )
    promoted = parity_receipt is not None and parity_receipt.passed
    return NativeSampledPathRun(
        result=result,
        signature=signature,
        maturity="parity_promoted" if promoted else "experimental_native",
        analysis_status="eligible" if promoted else "analysis_blocked_pending_parity",
        parity_receipt_hash=(parity_receipt.content_hash if parity_receipt else None),
        reference_anchor_loss=float(dense_anchor_loss),
        reference_anchor_evidence_hash=reference_anchor_evidence_hash,
        reference_anchor_mask_hash=reference_anchor_mask_hash,
        reference_anchor_panel_hash=reference_anchor_panel_hash,
    )


def sampled_path_policy_from_profile(profile: Any) -> SampledPathPolicy:
    """Compile the decision policy from a validated scientific profile."""

    profile.validate()
    return SampledPathPolicy(
        sample_stages_per_class=tuple(profile.sample_stages_per_class),
        fold_count=int(profile.crossfit_fold_count),
        posterior_accept_probability=float(profile.posterior_accept_probability),
        posterior_reject_probability=float(profile.posterior_reject_probability),
        variance_prior_strength=float(profile.variance_prior_strength),
        finite_population_correction=bool(profile.finite_population_correction),
        design_weighted=bool(profile.design_weighted_calibration),
        unresolved_final_action="reject",
        random_seed=int(profile.statistical_sampling_seed),
        precision_grid_units=float(profile.statistical_precision_grid_units),
    )


def neyman_policy_from_profile(profile: Any) -> Any:
    """Compile the cost-optimal panel policy without importing legacy code."""

    from .neyman import CostOptimalNeymanPolicy

    profile.validate()
    return CostOptimalNeymanPolicy(
        sampling_seed=int(profile.statistical_sampling_seed),
        fold_count=int(profile.crossfit_fold_count),
        panel_image_budget_sequence=tuple(profile.panel_image_budget_sequence),
        min_positive_images=int(profile.min_positive_images),
        min_negative_categories=int(profile.min_negative_categories),
        min_neg_images_per_category=int(profile.min_neg_images_per_category),
        min_neg_images_per_new_category=int(profile.min_neg_images_per_new_category),
        max_negative_categories=int(profile.max_negative_categories),
        panel_batch_multiple=int(profile.statistical_panel_batch_multiple),
        finite_population_correction=bool(profile.finite_population_correction),
        variance_prior_strength=float(profile.variance_prior_strength),
        storage_sort=True,
        on_target_context_cost=float(
            getattr(profile, "statistical_on_target_context_cost", 1.0)
        ),
        off_target_context_cost=float(
            getattr(profile, "statistical_off_target_context_cost", 1.0)
        ),
    )


def run_profiled_native_sampled_path_group(
    *,
    backend: SampledPathEvaluationBackend,
    profile: Any,
    scores: Mapping[str, Any],
    partition_labels: Mapping[str, Any],
    query_id: str,
    layer_id: str,
    route_group_kind: str,
    dense_anchor_loss: float,
    previous_group_masks: Mapping[str, Any] | None = None,
    parity_receipt: (
        NativeSampledPathParityReceipt | NativeSampledPathPromotionReceipt | None
    ) = None,
    claim_bearing: bool = False,
    initial_committed_warm_start: Mapping[str, Any] | None = None,
    reference_anchor_evidence_hash: str | None = None,
    reference_anchor_mask_hash: str | None = None,
    reference_anchor_panel_hash: str | None = None,
) -> NativeSampledPathRun:
    """Run one exact layer-group turn sequence from a frozen profile."""

    from .masking import (
        build_historical_partitioned_candidate_lowerer,
        compose_committed_masks,
        historical_density_axis,
        lower_historical_partitioned_masks,
    )

    profile.validate()
    schedule = historical_density_axis(
        route_group_kind,
        conv_decay=float(profile.target_preservation_conv_decay),
        conv_target_final_density=float(
            profile.target_preservation_conv_target_final_density
        ),
        linear_decay=float(profile.target_preservation_linear_decay),
        linear_target_final_density=float(
            profile.target_preservation_linear_target_final_density
        ),
    )
    dense_group = lower_historical_partitioned_masks(
        scores,
        partition_labels,
        positive_density=1.0,
        negative_density=1.0,
    )
    committed = compose_committed_masks(dense_group.masks, previous_group_masks)
    return run_parity_gated_native_sampled_path(
        backend=backend,
        committed_masks=committed,
        build_candidate=build_historical_partitioned_candidate_lowerer(
            scores,
            partition_labels,
            previous_group_masks=previous_group_masks,
        ),
        query_id=str(query_id),
        layer_id=str(layer_id),
        positive_schedule=schedule,
        negative_schedule=schedule,
        total_layer_route_count=int(dense_group.total_route_count),
        dense_anchor_loss=float(dense_anchor_loss),
        fully_pruned_anchor_loss=float(profile.fully_pruned_anchor_loss),
        sensitivity_threshold=float(profile.sensitivity_tau),
        policy=sampled_path_policy_from_profile(profile),
        profile_hash=profile.content_hash,
        parity_receipt=parity_receipt,
        claim_bearing=claim_bearing,
        initial_committed_warm_start=initial_committed_warm_start,
        reference_anchor_evidence_hash=reference_anchor_evidence_hash,
        reference_anchor_mask_hash=reference_anchor_mask_hash,
        reference_anchor_panel_hash=reference_anchor_panel_hash,
        initial_turn=str(profile.initial_partition),
    )


def run_profiled_native_sampled_path_panel(
    *,
    backend: SampledPathEvaluationBackend,
    profile: Any,
    measurement: Any,
    route_activity_scores_by_layer: Mapping[str, Any] | None = None,
    query_ids: Sequence[str],
    route_groups: Sequence[SampledPathRouteGroupBinding],
    partition_labels_by_layer: Mapping[str, Any],
    dense_anchor_loss_by_query: Mapping[str, float],
    promotion_receipt: NativeSampledPathPromotionReceipt | None = None,
    claim_bearing: bool = False,
    max_group_count: int | None = None,
    layer_execution_order: str = "forward",
    algorithm: str | None = None,
    image_budget_per_group: float | None = None,
    naturalization_facts: Any = None,
    maximum_commits: int | None = None,
    accept_replication_gate: bool = False,
    bce_decision_rule_by_query: Mapping[str, Any] | None = None,
    loss_span_certificate_by_query: Mapping[str, Any] | None = None,
    progress_callback: Any | None = None,
    step_context_delta_sink: Any | None = None,
    runtime_deadline_monotonic: float | None = None,
    group_transition_callback: Any | None = None,
    # v4 decision law.  Copied verbatim from ``IndependentSampleSizePolicy`` by
    # the caller and forwarded to the BCE precision group runner, which puts
    # them on ``BCEPrecisionPolicy``.  The frontier refuses a backend whose
    # ``oracle_sampling_id`` disagrees with the declared law, so this
    # pass-through is what makes the sealed law usable end to end.  The
    # defaults reproduce the pre-v4 policy exactly.
    oracle_sampling_law: str = IID_WITH_REPLACEMENT_SAMPLING_ID,
    oracle_acquisition_envelope_seconds: float | None = None,
    decision_law_id: str = "sequential_calibrated_bce_resolution_v5",
) -> dict[str, tuple[Any, ...]]:
    """Execute every query and layer group of one supervised panel.

    ``algorithm=None`` resolves through ``canonical_identification_default``
    (the BCE precision engine) for exploratory panels; claim-bearing panels stay
    contractually pinned to ``statistical_sequential_symmetric_frontier`` and
    resolve there unless a different algorithm is requested explicitly, which
    is refused.
    """

    from .native_continuous import (
        frontier_prior_from_history,
        record_frontier_history,
        run_profiled_native_bce_precision_group,
        run_profiled_native_continuous_group,
    )
    from ..core.versions import canonical_identification_default

    if algorithm is None:
        resolved_algorithm = (
            SAMPLED_PATH_PANEL_GRID_ALGORITHM
            if claim_bearing
            else canonical_identification_default()
        )
    else:
        resolved_algorithm = str(algorithm)
    if resolved_algorithm not in {
        SAMPLED_PATH_PANEL_GRID_ALGORITHM,
        EVIDENCE_CONTINUOUS_ALGORITHM,
        BCE_PRECISION_ALGORITHM,
    }:
        raise ValueError(
            f"unknown sampled-path panel algorithm {resolved_algorithm!r}"
        )
    if claim_bearing and resolved_algorithm != SAMPLED_PATH_PANEL_GRID_ALGORITHM:
        raise ValueError(
            "claim-bearing panels are contractually pinned to "
            "statistical_sequential_symmetric_frontier"
        )
    if maximum_commits is not None:
        if int(maximum_commits) <= 0:
            raise ValueError("sampled-path maximum_commits must be positive")
        if resolved_algorithm != BCE_PRECISION_ALGORITHM:
            raise ValueError(
                "campaign-wide commit limiting is currently defined for the "
                "BCE-precision v0 engine"
            )
    profile.validate()
    queries = tuple(str(value) for value in query_ids)
    if not queries or len(set(queries)) != len(queries):
        raise ValueError("sampled-path panel requires unique query IDs")
    measurement_queries = tuple(str(value) for value in measurement.query_ids)
    if not measurement_queries or len(set(measurement_queries)) != len(
        measurement_queries
    ):
        raise ValueError("sampled-path measurement requires unique query IDs")
    missing_queries = tuple(
        query_id for query_id in queries if query_id not in measurement_queries
    )
    if missing_queries:
        raise ValueError(
            "sampled-path panel query axis is outside its measurement: "
            + ",".join(missing_queries)
        )
    measurement_positions = tuple(
        measurement_queries.index(query_id) for query_id in queries
    )
    if tuple(sorted(measurement_positions)) != measurement_positions:
        raise ValueError(
            "sampled-path panel queries must preserve the measurement query order"
        )
    groups = tuple(route_groups)
    if not groups:
        raise ValueError("sampled-path panel requires route groups")
    if max_group_count is not None and not 1 <= int(max_group_count) <= len(groups):
        raise ValueError("sampled-path prefix group count is outside the route axis")
    # Layer EXECUTION order. The walk visits each route group exactly
    # once, carrying ``prior_masks`` forward, and every group re-anchors on
    # that accumulated state (only group_index 0 is checked against the sealed
    # dense anchor). So a layer's prices are measured against whatever earlier
    # layers already removed -- measured on real arms: with conv walled at cp
    # (100% retained) fc pruned to 0.44%, while kt removed 60% of conv first
    # and fc then halted at 0.71%, its retained set a strict SUPERSET of cp's.
    # Same routes, same score order, different prior state, different stopping
    # depth.
    #
    # "forward" is the declared route-block order (input side first; the
    # default). "reverse" visits the same groups in the
    # opposite order (output side first). ONLY the visit order changes: the
    # route axis, its offsets, route ids, the route-axis hash, the mask
    # layout, and the within-layer score ordering are all untouched, so
    # reverse-order masks remain directly comparable with forward-order masks
    # route-for-route. Reversal is applied BEFORE any max_group_count prefix,
    # so the prefix always means "the first N groups actually executed".
    if str(layer_execution_order) not in {"forward", "reverse"}:
        raise ValueError(
            "layer_execution_order must be 'forward' or 'reverse', got "
            f"{layer_execution_order!r}"
        )
    ordered_groups = (
        groups
        if str(layer_execution_order) == "forward"
        else tuple(reversed(groups))
    )
    execution_groups = (
        ordered_groups
        if max_group_count is None
        else ordered_groups[: int(max_group_count)]
    )
    for group in groups:
        group.validate()
    group_layers = tuple(group.layer_id for group in groups)
    if len(set(group_layers)) != len(group_layers):
        raise ValueError("sampled-path panel route groups repeat stable layer IDs")
    measured_blocks = {str(block.layer_id): block for block in measurement.route_blocks}
    if not set(group_layers).issubset(measured_blocks):
        raise ValueError(
            "sampled-path panel groups do not cover the prunable route axis"
        )
    for group in groups:
        if tuple(measured_blocks[group.layer_id].shape) != tuple(group.shape):
            raise ValueError(f"sampled-path group shape differs for {group.layer_id!r}")
    anchors = {
        str(key): float(value) for key, value in dense_anchor_loss_by_query.items()
    }
    bce_rules = (
        {}
        if bce_decision_rule_by_query is None
        else {
            str(key): value
            for key, value in bce_decision_rule_by_query.items()
        }
    )
    loss_span_certificates = {
        str(key): value
        for key, value in dict(
            loss_span_certificate_by_query or {}
        ).items()
    }
    from mechanistic_systems.circuits.evidence_bce_precision import (
        V0MarginalConnectionDecisionRule,
    )

    modern_v0_queries = tuple(
        sorted(
            query_id
            for query_id, rule in bce_rules.items()
            if isinstance(rule, V0MarginalConnectionDecisionRule)
        )
    )
    if modern_v0_queries and set(loss_span_certificates) != set(queries):
        # An empty certificate map used to short-circuit silently, dropping
        # the loss-span certificate and connection semantics from every
        # frontier call and downgrading modern v0 to the legacy estimand
        # with zero decision receipts and no error.
        raise ValueError(
            "modern v0 decision rules require a loss-span certificate for "
            "every query; missing "
            f"{sorted(set(queries) - set(loss_span_certificates))!r}"
        )
    if loss_span_certificates and set(loss_span_certificates) != set(queries):
        raise ValueError(
            "loss-span certificates must cover the exact query axis"
        )
    if loss_span_certificates:
        # Modern v0's cost coordinate is a circuit *connection*: FC scalar
        # weights and legal Conv channel pairs. Residual stream gates remain
        # physically present and dense but are not silently priced or pruned
        # as connections.
        execution_groups = tuple(
            group
            for group in execution_groups
            if group.route_kind in {"linear_weight", "conv_channel_pair", "conv_kernel_weight"}
        )
        if not execution_groups:
            raise ValueError(
                "modern v0 route axis contains no eligible circuit connections"
            )
    route_semantics_by_layer = {
        group.layer_id: (
            "fc_scalar_weight"
            if group.route_kind == "linear_weight"
            else (
                "conv_channel_pair"
                if group.route_kind == "conv_channel_pair"
                else (
                    "conv_kernel_tap"
                    if group.route_kind == "conv_kernel_weight"
                    else "protected_nonconnection_route"
                )
            )
        )
        for group in groups
    }
    if bce_rules and (
        resolved_algorithm != BCE_PRECISION_ALGORITHM
        or set(bce_rules) != set(queries)
    ):
        raise ValueError(
            "BCE decision-rule overrides require BCE precision execution and "
            "must cover the exact query axis"
        )
    if set(anchors) != set(queries) or any(
        not math.isfinite(value) for value in anchors.values()
    ):
        raise ValueError(
            "sampled-path dense anchors must be finite and cover the exact query axis"
        )
    if claim_bearing:
        if promotion_receipt is None:
            raise ValueError("claim-bearing sampled-path panels require promotion")
        for query_id in queries:
            promotion_receipt.require_passed(
                profile_hash=profile.content_hash,
                query_id=query_id,
            )

    query_index = {
        query_id: index for index, query_id in enumerate(measurement_queries)
    }
    if set(str(key) for key in partition_labels_by_layer) != set(group_layers):
        raise ValueError(
            "sampled-path structural partition labels must cover the exact route axis"
        )
    score_heads: dict[str, np.ndarray] = {}
    structural_labels: dict[str, np.ndarray] = {}
    scores_overridden = route_activity_scores_by_layer is not None
    if scores_overridden and set(
        map(str, route_activity_scores_by_layer)
    ) != set(group_layers):
        raise ValueError(
            "sampled-path activity-score override must cover the route groups"
        )
    # The measurement block carries one row per MEASUREMENT query, indexed
    # by measurement position.  A caller override carries one row per PANEL
    # query, indexed by panel position: the caller measured activity for
    # exactly the queries it asks this panel to run, and demanding the full
    # measurement query axis from it would require rows the caller never
    # measured.  (The registered AlexNet canaries died on exactly that
    # demand: a five-query sealed measurement axis against the pilot's
    # one-query override.)
    expected_head_rows = (
        len(queries) if scores_overridden else len(measurement_queries)
    )
    head_row_by_query = {
        query_id: (
            queries.index(query_id)
            if scores_overridden
            else query_index[query_id]
        )
        for query_id in queries
    }
    for group in groups:
        absolute = np.asarray(
            (
                route_activity_scores_by_layer[group.layer_id]
                if scores_overridden
                else measurement.source_image_aggregate_block(
                    group.layer_id,
                    score_head="route_abs_scores",
                )
            ),
            dtype=np.float64,
        )
        expected = (
            expected_head_rows,
            int(np.prod(group.shape, dtype=np.int64)),
        )
        if absolute.shape != expected:
            head_source = (
                "route_activity_scores_by_layer override "
                "(one row per panel query)"
                if scores_overridden
                else "measurement.source_image_aggregate_block "
                "(one row per measurement query)"
            )
            raise ValueError(
                "sampled-path source heads have the wrong shape for "
                f"{group.layer_id!r}: observed={absolute.shape!r} "
                f"expected={expected!r} from {head_source}"
            )
        if (
            not np.isfinite(absolute).all() or np.any(absolute < 0.0)
        ):
            raise ValueError("sampled-path source score heads must be finite")
        labels = np.asarray(partition_labels_by_layer[group.layer_id])
        if labels.shape != group.shape:
            raise ValueError(
                f"sampled-path partition labels have the wrong shape for {group.layer_id!r}"
            )
        if labels.dtype.kind not in {"i", "u", "b"} or not np.isin(
            labels, (0, 1)
        ).all():
            raise ValueError("sampled-path partition labels must be integer zero/one")
        score_heads[group.layer_id] = absolute
        structural_labels[group.layer_id] = labels.astype(np.int8, copy=False)

    runs_by_query: dict[str, tuple[Any, ...]] = {}
    dense_route_axis = {
        group.layer_id: np.ones(group.shape, dtype=bool) for group in groups
    }
    # Accepted-density history threads bisection priors across groups and
    # queries when the continuous engine runs; priors only seed the initial
    # step size and never affect validity.
    continuous_frontier_history: dict[str, list[float]] = {}
    for query_id in queries:
        index = head_row_by_query[query_id]
        # Historical execution treated every not-yet-visited group as dense.
        # The canonical evaluator makes that identity state explicit so every
        # candidate is a total function over the declared route axis.
        prior_masks = {
            layer_id: mask.copy() for layer_id, mask in dense_route_axis.items()
        }
        query_runs: list[Any] = []
        accepted_commit_count = 0
        aggregate_bce_spent = 0.0
        for group_index, group in enumerate(execution_groups):
            if (
                runtime_deadline_monotonic is not None
                and __import__("time").perf_counter()
                >= float(runtime_deadline_monotonic)
            ):
                break
            if (
                maximum_commits is not None
                and accepted_commit_count >= int(maximum_commits)
            ):
                break
            if group_transition_callback is not None:
                # Runtime-only cache retirement at a layer-group boundary.
                # The callback cannot alter contexts, statistics, or masks.
                group_transition_callback(
                    query_id=str(query_id),
                    group_index=int(group_index),
                    layer_id=str(group.layer_id),
                )
            absolute = score_heads[group.layer_id]
            reference_request = SampledComparisonRequest(
                query_id=query_id,
                layer_id=group.layer_id,
                partition_id="dense_reference",
                previous_density=1.0,
                proposed_density=1.0,
                stage_index=0,
                stage_per_class=int(profile.sample_stages_per_class[0]),
                proposal_index=0,
                random_seed=int(profile.statistical_sampling_seed),
                fold_count=int(profile.crossfit_fold_count),
            )
            reference: SampledReferenceEvidence = backend.evaluate_dense_reference(
                prior_masks,
                reference_request,
            )
            reference.validate(fold_count=int(profile.crossfit_fold_count))
            expected_mask_hash = sampled_path_mask_hash(prior_masks)
            if reference.query_id != query_id:
                raise ValueError("sampled-path group reference query identity differs")
            if reference.mask_hash != expected_mask_hash:
                raise ValueError("sampled-path group reference mask identity differs")
            group_anchor_loss = reference.canonical_loss
            if group_index == 0 and not math.isclose(
                group_anchor_loss,
                anchors[query_id],
                rel_tol=INITIAL_DENSE_ANCHOR_NUMERICAL_TOLERANCE,
                abs_tol=INITIAL_DENSE_ANCHOR_NUMERICAL_TOLERANCE,
            ):
                raise ValueError(
                    "native first-group reference differs from the sealed globally "
                    f"dense anchor for query {query_id!r}: "
                    f"observed={group_anchor_loss:.17g} expected={anchors[query_id]:.17g}"
                )
            if resolved_algorithm in {
                EVIDENCE_CONTINUOUS_ALGORITHM,
                BCE_PRECISION_ALGORITHM,
            }:
                continuous_runner = (
                    run_profiled_native_bce_precision_group
                    if resolved_algorithm == BCE_PRECISION_ALGORITHM
                    else run_profiled_native_continuous_group
                )
                run = continuous_runner(
                    backend=backend,
                    profile=profile,
                    scores={
                        group.layer_id: absolute[index].reshape(
                            group.shape, order="C"
                        )
                    },
                    partition_labels={
                        group.layer_id: structural_labels[group.layer_id]
                    },
                    query_id=query_id,
                    layer_id=group.layer_id,
                    route_group_kind=group.route_group_kind,
                    previous_group_masks=prior_masks,
                    frontier_prior=frontier_prior_from_history(
                        continuous_frontier_history
                    ),
                    image_budget=image_budget_per_group,
                    claim_bearing=claim_bearing,
                    naturalization_facts=naturalization_facts,
                    reference_anchor_loss=group_anchor_loss,
                    reference_anchor_evidence_hash=reference.content_hash,
                    reference_anchor_mask_hash=reference.mask_hash,
                    reference_anchor_panel_hash=reference.panel.panel_hash,
                    **(
                        {"dense_anchor_loss": group_anchor_loss}
                        if resolved_algorithm == EVIDENCE_CONTINUOUS_ALGORITHM
                        else {}
                    ),
                    **(
                        {
                            "maximum_accepted_decisions": (
                                int(maximum_commits) - accepted_commit_count
                            )
                        }
                        if (
                            resolved_algorithm == BCE_PRECISION_ALGORITHM
                            and maximum_commits is not None
                        )
                        else {}
                    ),
                    **(
                        {
                            "oracle_sampling_law": str(oracle_sampling_law),
                            "oracle_acquisition_envelope_seconds": (
                                oracle_acquisition_envelope_seconds
                            ),
                            "decision_law_id": str(decision_law_id),
                        }
                        if resolved_algorithm == BCE_PRECISION_ALGORITHM
                        else {}
                    ),
                    **(
                        {
                            "decision_rule": (
                                type(bce_rules[query_id])(
                                    confidence_level=float(
                                        bce_rules[query_id].confidence_level
                                    ),
                                    maximum_bce_increment=max(
                                        float(
                                            bce_rules[
                                                query_id
                                            ].maximum_bce_increment
                                        )
                                        - aggregate_bce_spent,
                                        0.0,
                                    ),
                                )
                            )
                        }
                        if (
                            resolved_algorithm == BCE_PRECISION_ALGORITHM
                            and query_id in bce_rules
                            and hasattr(
                                bce_rules[query_id],
                                "maximum_bce_increment",
                            )
                        )
                        else (
                            {"decision_rule": bce_rules[query_id]}
                            if (
                                resolved_algorithm == BCE_PRECISION_ALGORITHM
                                and query_id in bce_rules
                            )
                            else {}
                        )
                    ),
                    **(
                        {"progress_callback": progress_callback}
                        if resolved_algorithm == BCE_PRECISION_ALGORITHM
                        else {}
                    ),
                    **(
                        {"step_context_delta_sink": step_context_delta_sink}
                        if resolved_algorithm == BCE_PRECISION_ALGORITHM
                        else {}
                    ),
                    **(
                        {
                            "accept_replication_gate": (
                                accept_replication_gate
                            )
                        }
                        if resolved_algorithm == BCE_PRECISION_ALGORITHM
                        else {}
                    ),
                    **(
                        {
                            "runtime_deadline_monotonic": (
                                runtime_deadline_monotonic
                            )
                        }
                        if resolved_algorithm == BCE_PRECISION_ALGORITHM
                        else {}
                    ),
                    **(
                        {
                            "loss_span_certificate": (
                                loss_span_certificates[query_id]
                            ),
                            "route_semantics_by_layer": (
                                route_semantics_by_layer
                            ),
                        }
                        if query_id in loss_span_certificates
                        else {}
                    ),
                )
                record_frontier_history(
                    continuous_frontier_history, run.result
                )
            else:
                run = run_profiled_native_sampled_path_group(
                    backend=backend,
                    profile=profile,
                    scores={
                        group.layer_id: absolute[index].reshape(
                            group.shape, order="C"
                        )
                    },
                    partition_labels={
                        group.layer_id: structural_labels[group.layer_id]
                    },
                    query_id=query_id,
                    layer_id=group.layer_id,
                    route_group_kind=group.route_group_kind,
                    dense_anchor_loss=group_anchor_loss,
                    previous_group_masks=prior_masks,
                    parity_receipt=promotion_receipt,
                    claim_bearing=claim_bearing,
                    initial_committed_warm_start=fold_warm_start(
                        reference.calibration_by_fold
                    ),
                    reference_anchor_evidence_hash=reference.content_hash,
                    reference_anchor_mask_hash=reference.mask_hash,
                    reference_anchor_panel_hash=reference.panel.panel_hash,
                )
            prior_masks = {
                str(layer_id): np.asarray(mask, dtype=bool).copy()
                for layer_id, mask in run.result.committed_masks.items()
            }
            query_runs.append(run)
            if resolved_algorithm == BCE_PRECISION_ALGORITHM:
                accepted_commit_count += sum(
                    decision.action == "accept"
                    for decision in run.result.decisions
                )
                aggregate_bce_spent += float(run.result.delta_spent)
        runs_by_query[query_id] = tuple(query_runs)
    return runs_by_query


def materialize_native_sampled_path_circuits(
    *,
    plan: CircuitPlan,
    store: ArtifactStore,
    runs_by_query: Mapping[str, Sequence[NativeSampledPathRun]],
    route_blocks: Sequence[RouteBlockSpec],
    staging_dir: str | Path,
    promotion_receipt: NativeSampledPathPromotionReceipt | None = None,
    final_validation_receipt: "NativeSampledPathFinalValidationReceipt | None" = None,
    glocal_deployment_certificates: Mapping[
        str, GlocalDeploymentMaskCertificate
    ]
    | None = None,
    preservation_by_query: Mapping[str, float] | None = None,
    valid_for_claims: bool | None = None,
    partition_label_provider_id: str = "",
    partition_label_hash: str = "",
) -> tuple[ArtifactManifest, Path]:
    """Assemble promoted layer-group runs into one canonical circuit artifact."""

    query_ids = tuple(str(value) for value in runs_by_query)
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("native sampled-path materialization requires unique queries")
    blocks = tuple(route_blocks)
    route_axis_hash = ordered_block_route_axis_hash(blocks)
    if route_axis_hash != str(plan.route_universe_hash):
        raise ValueError(
            "native sampled-path blocks differ from the CircuitPlan route axis"
        )
    plan_algorithm = str(plan.search_policy.get("algorithm", ""))
    if plan_algorithm not in {
        SAMPLED_PATH_PANEL_GRID_ALGORITHM,
        EVIDENCE_CONTINUOUS_ALGORITHM,
        BCE_PRECISION_ALGORITHM,
    }:
        raise ValueError("native materialization requires a supported search plan")
    run_algorithms = {
        str(run.signature.algorithm_version)
        for runs in runs_by_query.values()
        for run in runs
    }
    expected_run_algorithm = {
        SAMPLED_PATH_PANEL_GRID_ALGORITHM: SAMPLED_PATH_FRONTIER_V1,
        EVIDENCE_CONTINUOUS_ALGORITHM: EVIDENCE_CONTINUOUS_ALGORITHM,
        BCE_PRECISION_ALGORITHM: BCE_PRECISION_ALGORITHM,
    }[plan_algorithm]
    allowed_run_algorithms = (
        {
            BCE_PRECISION_ALGORITHM,
            SEQUENTIAL_CALIBRATED_BCE_V2,
            SEQUENTIAL_CALIBRATED_BCE_V3,
        }
        if plan_algorithm == BCE_PRECISION_ALGORITHM
        else {expected_run_algorithm}
    )
    if (
        len(run_algorithms) != 1
        or not run_algorithms.issubset(allowed_run_algorithms)
    ):
        raise ValueError(
            "native runs were traced under a different algorithm than the plan "
            f"declares: plan={plan_algorithm!r} runs={sorted(run_algorithms)!r}"
        )
    resolved_claim_validity = (
        str(plan.status) == "frozen"
        if valid_for_claims is None
        else bool(valid_for_claims)
    )
    if resolved_claim_validity and plan_algorithm != (
        SAMPLED_PATH_PANEL_GRID_ALGORITHM
    ):
        raise ValueError(
            "claim-valid artifacts are contractually pinned to "
            "statistical_sequential_symmetric_frontier"
        )
    if resolved_claim_validity and glocal_deployment_certificates:
        raise ValueError(
            "shadow glocal-v3 deployment certificates cannot promote an "
            "artifact to claim-valid status"
        )
    if resolved_claim_validity and promotion_receipt is None:
        raise ValueError(
            "claim-valid native sampled-path artifacts require a panel promotion receipt"
        )
    if resolved_claim_validity and final_validation_receipt is None:
        raise ValueError(
            "claim-valid native sampled-path artifacts require untouched final validation"
        )

    expected_layers = tuple(str(block.layer_id) for block in blocks)
    expected_layer_set = set(expected_layers)
    masks_by_query: dict[str, dict[str, np.ndarray]] = {}
    frontier_rows: list[dict[str, object]] = []
    group_reference_rows: list[dict[str, object]] = []
    profile_hashes: set[str] = set()
    for query_id in query_ids:
        runs = tuple(runs_by_query[query_id])
        if not runs:
            raise ValueError(f"query {query_id!r} has no native sampled-path groups")
        layer_ids = tuple(run.signature.layer_id for run in runs)
        if (
            len(set(layer_ids)) != len(layer_ids)
            or set(layer_ids) != expected_layer_set
        ):
            raise ValueError(
                f"query {query_id!r} does not cover the declared layer groups"
            )
        prior_masks: dict[str, np.ndarray] = {}
        reference_masks = {
            str(block.layer_id): np.ones(block.shape, dtype=bool) for block in blocks
        }
        committed_layer_ids: set[str] = set()
        for group_index, run in enumerate(runs):
            signature = run.signature
            if signature.query_id != query_id:
                raise ValueError("native sampled-path run query identity mismatch")
            profile_hashes.add(signature.profile_hash)
            current_masks = {
                str(layer): np.asarray(mask, dtype=bool)
                for layer, mask in run.result.committed_masks.items()
            }
            expected_reference_mask_hash = sampled_path_mask_hash(reference_masks)
            reference_fields = (
                run.reference_anchor_loss,
                run.reference_anchor_evidence_hash,
                run.reference_anchor_mask_hash,
                run.reference_anchor_panel_hash,
            )
            if resolved_claim_validity and any(
                value is None for value in reference_fields
            ):
                raise ValueError(
                    "claim-valid sampled-path groups require committed-reference evidence"
                )
            if all(value is not None for value in reference_fields):
                if (
                    not math.isfinite(float(run.reference_anchor_loss))
                    or len(str(run.reference_anchor_evidence_hash)) != 64
                    or len(str(run.reference_anchor_mask_hash)) != 64
                    or len(str(run.reference_anchor_panel_hash)) != 64
                ):
                    raise ValueError("sampled-path group reference evidence is invalid")
                if str(run.reference_anchor_mask_hash) != expected_reference_mask_hash:
                    raise ValueError(
                        "sampled-path group reference was not measured at the "
                        "committed layer-group boundary"
                    )
                group_reference_rows.append(
                    {
                        "query_id": query_id,
                        "layer_id": signature.layer_id,
                        "group_index": int(group_index),
                        "canonical_loss": float(run.reference_anchor_loss),
                        "evidence_hash": str(run.reference_anchor_evidence_hash),
                        "mask_hash": str(run.reference_anchor_mask_hash),
                        "panel_hash": str(run.reference_anchor_panel_hash),
                    }
                )
            elif any(value is not None for value in reference_fields):
                raise ValueError(
                    "sampled-path group reference evidence must be complete or absent"
                )
            for prior_layer in committed_layer_ids:
                prior_mask = prior_masks[prior_layer]
                observed = current_masks.get(prior_layer)
                if observed is None or not np.array_equal(observed, prior_mask):
                    raise ValueError(
                        "native sampled-path cumulative masks changed a committed layer"
                    )
            prior_masks = current_masks
            reference_masks = {
                str(layer): np.asarray(mask, dtype=bool).copy()
                for layer, mask in current_masks.items()
            }
            committed_layer_ids.add(signature.layer_id)
            if not hasattr(run.result, "steps"):
                from .native_continuous import continuous_frontier_rows

                frontier_rows.extend(
                    continuous_frontier_rows(
                        query_id,
                        signature.layer_id,
                        group_index,
                        run,
                    )
                )
                continue
            for step in run.result.steps:
                frontier_rows.append(
                    {
                        "query_id": query_id,
                        "layer_id": signature.layer_id,
                        "group_index": group_index,
                        "proposal_index": int(step.proposal_index),
                        "partition_id": str(step.partition_id),
                        "previous_density": float(step.previous_density),
                        "proposed_density": float(step.proposed_density),
                        "accepted": bool(step.accepted),
                        "action": str(step.decision.action),
                        "posterior_mean_sensitivity": float(
                            step.decision.posterior_mean_sensitivity
                        ),
                        "posterior_sd_sensitivity": float(
                            step.decision.posterior_sd_sensitivity
                        ),
                        "context_count": int(step.decision.context_count),
                        "stage_index": int(step.decision.stage_index),
                        "panel_hash": str(step.decision.panel_hash),
                        "group_reference_anchor_loss": (
                            None
                            if run.reference_anchor_loss is None
                            else float(run.reference_anchor_loss)
                        ),
                        "group_reference_anchor_evidence_hash": (
                            run.reference_anchor_evidence_hash or ""
                        ),
                    }
                )
        if set(prior_masks) != expected_layer_set:
            raise ValueError(
                f"query {query_id!r} final masks differ from the declared route blocks"
            )
        ordered_masks: dict[str, np.ndarray] = {}
        for block in blocks:
            mask = np.asarray(prior_masks[block.layer_id], dtype=bool)
            if tuple(mask.shape) != tuple(block.shape):
                raise ValueError(
                    f"query {query_id!r} mask shape differs for {block.layer_id!r}"
                )
            ordered_masks[block.layer_id] = mask
        masks_by_query[query_id] = ordered_masks

    if len(profile_hashes) != 1:
        raise ValueError("native sampled-path groups do not share one profile hash")
    profile_hash = next(iter(profile_hashes))
    group_reference_anchor_hash = stable_digest(
        {
            "policy": SAMPLED_PATH_GROUP_REFERENCE_POLICY,
            "rows": group_reference_rows,
        }
    )
    if promotion_receipt is not None:
        for query_id in query_ids:
            promotion_receipt.require_passed(
                profile_hash=profile_hash,
                query_id=query_id,
            )
        if resolved_claim_validity:
            expected_receipt_hash = promotion_receipt.content_hash
            for runs in runs_by_query.values():
                for run in runs:
                    if run.parity_receipt_hash != expected_receipt_hash:
                        raise ValueError(
                            "claim-valid sampled-path run was not executed under the "
                            "panel promotion receipt"
                        )
    if final_validation_receipt is not None:
        from .final_validation import (
            GLOCAL_NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA,
            NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA,
        )

        if final_validation_receipt.schema_name == (
            GLOCAL_NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA
        ):
            final_validation_receipt.require_glocal_claim_ready(
                plan_hash=plan.content_hash,
                profile_hash=profile_hash,
                promotion_receipt_hash=(
                    promotion_receipt.content_hash if promotion_receipt else None
                ),
                query_ids=query_ids,
            )
        elif final_validation_receipt.schema_name == (
            NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA
        ):
            final_validation_receipt.require_passed(
                plan_hash=plan.content_hash,
                profile_hash=profile_hash,
                promotion_receipt_hash=(
                    promotion_receipt.content_hash if promotion_receipt else None
                ),
                query_ids=query_ids,
            )
        else:
            raise ValueError(
                "native materialization received an unknown final-validation schema"
            )
    deployment_certificates = dict(glocal_deployment_certificates or {})
    glocal_validation = bool(
        final_validation_receipt is not None
        and final_validation_receipt.schema_name
        == GLOCAL_NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA
    )
    if glocal_validation and not deployment_certificates:
        raise ValueError(
            "glocal terminal validation cannot be materialized without the "
            "campaign-global deployment certificate"
        )
    if (
        deployment_certificates
        and final_validation_receipt is not None
        and final_validation_receipt.schema_name
        == NATIVE_SAMPLED_PATH_FINAL_VALIDATION_SCHEMA
    ):
        raise ValueError(
            "glocal deployment certificates cannot be paired with a legacy "
            "terminal-validation receipt"
        )
    if deployment_certificates:
        if set(deployment_certificates) != set(query_ids):
            raise ValueError(
                "glocal deployment certificates must exactly cover the query axis"
            )
        for query_id in query_ids:
            certificate = deployment_certificates[query_id]
            if not isinstance(certificate, GlocalDeploymentMaskCertificate):
                raise TypeError(
                    "native materialization requires typed glocal deployment "
                    "certificates"
                )
            certificate.validate_mask_binding(
                native_route_axis_hash=route_axis_hash,
                native_mask_hash=sampled_path_mask_hash(
                    masks_by_query[query_id]
                ),
                query_id=query_id,
            )
            store.seal_contract(
                contract_kind="glocal_deployment_mask_certificate",
                content_hash=certificate.content_hash,
                payload=certificate.to_dict(),
            )

    circuits = BlockBitPackedCircuitSet.from_block_masks(
        masks_by_query,
        route_blocks=blocks,
    )
    return materialize_circuit_artifact(
        circuits,
        plan,
        store,
        staging_dir=staging_dir,
        preservation_by_query=preservation_by_query,
        frontier_rows=frontier_rows,
        source_kind="native",
        valid_for_claims=resolved_claim_validity,
        additional_semantics={
            "search_algorithm": plan_algorithm,
            "algorithm_version": expected_run_algorithm,
            "sampled_path_profile_hash": profile_hash,
            "promotion_receipt_hash": (
                promotion_receipt.content_hash if promotion_receipt else ""
            ),
            "promotion_acceptance_mode": (
                promotion_receipt.acceptance_mode if promotion_receipt else "unpromoted"
            ),
            "exact_circuit_identity": bool(
                promotion_receipt and promotion_receipt.exact_circuit_identity
            ),
            "final_validation_receipt_hash": (
                final_validation_receipt.content_hash
                if final_validation_receipt is not None
                else ""
            ),
            "final_validation_policy": (
                final_validation_receipt.evaluation_policy
                if final_validation_receipt is not None
                else "not_run"
            ),
            "statistical_foundation": (
                "evidence_foundation_glocal_v3"
                if deployment_certificates or glocal_validation
                else "legacy_sampled_path"
            ),
            "statistical_authority_status": (
                "shadow"
                if deployment_certificates or glocal_validation
                else "legacy"
            ),
            "glocal_deployment_certificate_hashes": {
                query_id: deployment_certificates[query_id].content_hash
                for query_id in query_ids
            }
            if deployment_certificates
            else {},
            "group_reference_anchor_policy": (SAMPLED_PATH_GROUP_REFERENCE_POLICY),
            "group_reference_anchor_hash": group_reference_anchor_hash,
            **(
                {
                    "partition_label_provider_id": str(
                        partition_label_provider_id
                    ),
                    "partition_label_hash": str(partition_label_hash),
                }
                if partition_label_provider_id and len(partition_label_hash) == 64
                else {}
            ),
        },
        notes={
            "sampled_path_group_count": sum(
                len(tuple(runs)) for runs in runs_by_query.values()
            ),
            "algorithm_configuration_parity": bool(
                promotion_receipt and promotion_receipt.passed
            ),
            "untouched_final_validation_passed": bool(
                final_validation_receipt and final_validation_receipt.passed
            ),
            "glocal_terminal_certificates_bound": bool(
                deployment_certificates
            ),
            "group_reference_anchor_count": len(group_reference_rows),
        },
    )


__all__ = [
    "BCE_PRECISION_ALGORITHM",
    "EVIDENCE_CONTINUOUS_ALGORITHM",
    "SAMPLED_PATH_PANEL_GRID_ALGORITHM",
    "NATIVE_SAMPLED_PATH_PARITY_SCHEMA",
    "NATIVE_SAMPLED_PATH_PROMOTION_SCHEMA",
    "NATIVE_SAMPLED_PATH_SIGNATURE_SCHEMA",
    "NativeSampledPathParityReceipt",
    "NativeSampledPathPromotionReceipt",
    "NativeSampledPathRun",
    "NativeSampledPathTraceRow",
    "NativeSampledPathTraceSignature",
    "SampledPathRouteGroupBinding",
    "certify_native_sampled_path_parity",
    "compile_sampled_path_route_groups",
    "load_native_sampled_path_parity_receipt",
    "load_native_sampled_path_promotion_receipt",
    "load_numerical_adjudication_promotion_receipt",
    "materialize_native_sampled_path_circuits",
    "neyman_policy_from_profile",
    "run_parity_gated_native_sampled_path",
    "run_profiled_native_sampled_path_group",
    "run_profiled_native_sampled_path_panel",
    "sampled_path_policy_from_profile",
    "signature_from_historical_trace",
    "signature_from_native_result",
    "promotion_receipt_from_numerical_adjudication",
    "write_native_sampled_path_parity_receipt",
    "write_native_sampled_path_promotion_receipt",
]
