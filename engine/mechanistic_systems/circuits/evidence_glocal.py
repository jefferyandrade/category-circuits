"""Shadow glocal-v3 execution over frozen Brier states and finite panels.

This module is the adapter that the statistical kernel deliberately does not
own.  It freezes one binary one-vs-rest risk frame, fits state-specific Brier
calibrations on calibration-only contexts, registers a predictable proposal,
and then reveals sealed within-stratum prefixes under a cost-aware Neyman
rule.  Only the bounded sampling-without-replacement e-process may authorize
the complete hard-mask transition.

The historical continuous-v2 controller is not imported or modified here.
Glocal v3 remains a shadow identity until its external coverage, matched
compute and retained-receipt gates pass.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from mechanistic_systems.evidence import (
    EVIDENCE_FOUNDATION_GLOCAL_V3,
    GLOCAL_IID_FIXED_N_RANDOMIZATION_METHOD_V1,
    GLOCAL_IID_RUNTIME_RANDOMIZATION_METHOD_V2,
    GLOCAL_SEALED_PERMUTATION_ALGORITHM_V1,
    CalibrationAlgorithmSpec,
    CalibrationFitState,
    CampaignEvidenceState,
    DecisionEvidenceReceipt,
    EvidenceRole,
    GlocalDeploymentAuthorityHandle,
    GlocalDeploymentMaskCertificate,
    IIDDrawRecord,
    IIDSamplingReceipt,
    IIDSupportDistribution,
    ObservationSetRef,
    PredictableAllocationLookReceipt,
    PredictionRuleSpec,
    ResourceLedger,
    ResourceUsage,
    RiskEstimandSpec,
    SamplingDesignSpec,
    SamplingPermutationReceipt,
    SamplingStratumSpec,
    StratifiedPairedEProcess,
    TerminalAuditEvidenceReceipt,
    TerminalAuditPrecommitment,
    TerminalParityEstimandSpec,
    TerminalParityCertificate,
    ValidationLock,
    build_glocal_deployment_mask_certificate,
    iid_physical_executor_identity_hash,
    iid_physical_provider_identity_hash,
    paired_brier_increment,
)
from mechanistic_systems.evidence.deployment import (
    _issue_physical_glocal_deployment_authority,
)
from mechanistic_systems.evidence.glocal_contracts import content_hash
from mechanistic_systems.measurement.contexts import ContextAxis
from mechanistic_systems.measurement.neyman_allocation import (
    NeymanAllocationRequest,
    NeymanStratumSpec,
    UNIFIED_NEYMAN_POLICY_ID,
    UnifiedNeymanAllocator,
)
from mechanistic_systems.measurement.request_providers import (
    ProviderRegistry,
    validate_outcome_for_request,
)
from mechanistic_systems.contracts.measurement import MeasurementOutcome
from mechanistic_systems.contracts.measurement_costs import CostEstimate, RealizedCost

from .irg.candidates import ActionCandidate, HardMask
from .irg.calibration import (
    CALIBRATION_CONTRASTS_CACHE_KEY,
    CALIBRATION_UNCERTAINTY_CACHE_KEY,
    CalibrationUncertaintyState,
)
from .irg.commit_authority import (
    ExactHardMaskCommitAuthorization,
    ExactHardMaskCommitDecision,
    _issue_physical_exact_mask_commit_authorization,
    _require_physical_glocal_commit_authority,
)
from .irg.response_geometry import RESPONSE_TRANSITION_CACHE_KEY
from .irg.runner import CommonIRGRunner, PruningRunResult
from .irg.state import PruningSessionState
from .irg.versions.base import ResponseVersion
from .glocal_sampled_path import (
    FrozenBrierCalibration,
    GlocalSampledComparisonRequest,
)
from .sampled_path import (
    CrossfitPanel,
    sampled_path_mask_hash,
)
from .sampled_torch import (
    GlocalCalibrationLogitRows,
    TorchSampledPathEvaluationBackend,
    fit_class_balanced_brier_affine,
)


GLOCAL_NEYMAN_ALLOCATION_RULE = (
    "predictable_packet_neyman_variance_reduction_per_realized_cost_v1"
)
GLOCAL_BINARY_CONTEXT_DESIGN_SCHEMA = "glocal_binary_context_design_v2"
GLOCAL_PROPOSAL_RESULT_SCHEMA = "glocal_proposal_execution_v1"
GLOCAL_IRG_MASK_BRIDGE_SCHEMA = "glocal_irg_native_mask_bridge_v1"
GLOCAL_CALIBRATION_TRANSITION_SCHEMA = (
    "glocal_campaign_calibration_transition_decision_v2"
)
GLOCAL_CALIBRATION_LOGIT_ACQUISITION_SCHEMA = (
    "glocal_campaign_calibration_logit_acquisition_v1"
)


def _require_sha256(value: str, label: str) -> str:
    identity = str(value)
    if len(identity) != 64 or any(
        character not in "0123456789abcdef" for character in identity
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return identity


def _require_prefixed_sha256(value: str, prefix: str, label: str) -> str:
    identity = str(value)
    if not identity.startswith(prefix):
        raise ValueError(f"{label} must use the {prefix!r} identity prefix")
    _require_sha256(identity[len(prefix) :], label)
    return identity


def _query_execution_state_hash(*, query_id: str, mask_hash: str) -> str:
    return content_hash(
        {
            "query_id": str(query_id),
            "mask_hash": str(mask_hash),
            "measurement": "raw_query_logits_before_frozen_brier_calibration",
        }
    )


def locked_brier_state_hash(
    *,
    query_id: str,
    mask_hash: str,
    calibration_fit_hash: str,
    risk_estimand_hash: str,
) -> str:
    """Identity of the official locked state ``(mask, calibration fit)``."""

    for value, label in (
        (mask_hash, "mask hash"),
        (calibration_fit_hash, "calibration fit hash"),
        (risk_estimand_hash, "risk estimand hash"),
    ):
        _require_sha256(value, label)
    return content_hash(
        {
            "foundation": EVIDENCE_FOUNDATION_GLOCAL_V3,
            "query_id": str(query_id),
            "mask_hash": str(mask_hash),
            "calibration_fit_hash": str(calibration_fit_hash),
            "risk_estimand_hash": str(risk_estimand_hash),
        }
    )


def _source_code_hash(function: Any) -> str:
    source_file = inspect.getsourcefile(function)
    if source_file is None:
        raise RuntimeError(
            "claim-bearing calibration requires an inspectable source file"
        )
    payload = Path(source_file).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _physical_live_object_binding(value: object, *, role: str) -> Mapping[str, Any]:
    """Bind one live execution object to its exact type, file, and instance."""

    source_file = inspect.getsourcefile(type(value))
    if source_file is None:
        raise RuntimeError(
            f"live deployment authority requires inspectable {role} source code"
        )
    source_path = Path(source_file).resolve()
    if not source_path.is_file():
        raise RuntimeError(f"live deployment authority cannot read {role} source code")
    declared: dict[str, Any] = {}
    for attribute in ("backend_id", "source_id", "axis_id"):
        observed = getattr(value, attribute, None)
        if observed is not None:
            declared[attribute] = str(observed)
    context_axis = getattr(value, "context_axis", None)
    context_axis_hash = getattr(context_axis, "content_hash", None)
    if isinstance(context_axis_hash, str):
        declared["context_axis_hash"] = context_axis_hash
    device = getattr(value, "device", None)
    if device is not None:
        declared["device"] = str(device)
    return MappingProxyType(
        {
            "role": str(role),
            "object_id": id(value),
            "type_module": type(value).__module__,
            "type_qualname": type(value).__qualname__,
            "source_path": str(source_path),
            "source_code_hash": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "declared_identity": declared,
        }
    )


def glocal_brier_calibration_algorithm_spec(
    *,
    maximum_iterations: int = 256,
    gradient_tolerance: float = 1e-9,
    identity_ridge: float = 1e-8,
) -> CalibrationAlgorithmSpec:
    """Content-address the exact positive-affine Brier implementation."""

    return CalibrationAlgorithmSpec(
        algorithm_id="class_balanced_brier_positive_affine",
        algorithm_version="1",
        code_hash=_source_code_hash(fit_class_balanced_brier_affine),
        objective="bounded_brier",
        maximum_iterations=int(maximum_iterations),
        convergence_tolerance=float(gradient_tolerance),
        supports_warm_start=True,
        options={
            "identity_ridge": float(identity_ridge),
            "scale_parameterization": "exp_log_scale",
            "covariance_role": "allocation_only_never_safety_bound",
        },
    )


@dataclass(frozen=True, slots=True)
class FrozenCalibrationBinding:
    """Native executable fit plus its stable statistical contract."""

    native_fit: FrozenBrierCalibration
    fit_state: CalibrationFitState
    observation_set: ObservationSetRef
    algorithm_spec: CalibrationAlgorithmSpec

    def __post_init__(self) -> None:
        self.native_fit.validate()
        self.fit_state.require_claim_bearing()
        if self.observation_set.role is not EvidenceRole.CALIBRATION_FIT:
            raise ValueError("calibration binding requires calibration-role rows")
        if self.fit_state.output_hash != self.native_fit.output_hash:
            raise ValueError("calibration contract and executable output differ")
        if self.fit_state.fit_observation_set_hash != self.observation_set.content_hash:
            raise ValueError("calibration fit and observation-set lineage differ")
        if self.fit_state.algorithm_spec_hash != self.algorithm_spec.content_hash:
            raise ValueError("calibration fit and algorithm identity differ")
        if (
            tuple(self.native_fit.calibration_context_ids)
            != self.observation_set.context_ids
        ):
            raise ValueError(
                "native calibration rows differ from the exact fit observation set"
            )


def calibration_binding_from_native(
    native_fit: FrozenBrierCalibration,
    *,
    fit_id: str,
    observation_set: ObservationSetRef,
    algorithm_spec: CalibrationAlgorithmSpec,
    warm_start_parent: CalibrationFitState | None = None,
    fit_input_evidence_hash: str | None = None,
) -> FrozenCalibrationBinding:
    native_fit.validate()
    if fit_input_evidence_hash is not None:
        _require_sha256(
            fit_input_evidence_hash,
            "calibration fit input-evidence hash",
        )
    fit_state = CalibrationFitState(
        fit_id=str(fit_id),
        algorithm_spec_hash=algorithm_spec.content_hash,
        fit_observation_set_hash=observation_set.content_hash,
        output_hash=native_fit.output_hash,
        parameters=(float(native_fit.scale), float(native_fit.offset)),
        covariance=native_fit.covariance,
        converged=bool(native_fit.converged),
        iterations=int(native_fit.iterations),
        warm_start_parent_hash=(
            None if warm_start_parent is None else warm_start_parent.content_hash
        ),
        diagnostics={
            **dict(native_fit.diagnostics),
            "input_center": float(native_fit.input_center),
            "native_algorithm_id": native_fit.algorithm_id,
            "native_algorithm_options": dict(native_fit.algorithm_options),
            "native_warm_start_output_hash": native_fit.warm_start_parent_hash,
            "mask_hash": native_fit.mask_hash,
            "query_id": native_fit.query_id,
            "role": native_fit.role,
            "fit_input_evidence_hash": fit_input_evidence_hash,
        },
    )
    return FrozenCalibrationBinding(
        native_fit=native_fit,
        fit_state=fit_state,
        observation_set=observation_set,
        algorithm_spec=algorithm_spec,
    )


@dataclass(frozen=True, slots=True)
class FrozenBinaryContextDesign:
    """Role-separated finite frame for one one-vs-rest query."""

    design_id: str
    query_id: str
    positive_class_id: int
    context_axis_hash: str
    risk_estimand: RiskEstimandSpec
    calibration_context_ids: tuple[str, ...]
    adjudication_context_ids: tuple[str, ...]
    terminal_context_ids: tuple[str, ...]
    label_by_context_id: tuple[tuple[str, int], ...]
    stratum_by_context_id: tuple[tuple[str, str], ...]
    calibration_observation_set: ObservationSetRef
    calibration_sampling_design: SamplingDesignSpec | None = None
    calibration_initial_targets_by_stratum: tuple[tuple[str, int], ...] = ()
    fold_count: int = 2
    schema_name: str = GLOCAL_BINARY_CONTEXT_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        if not self.design_id or not self.query_id:
            raise ValueError("glocal context design requires stable identity")
        _require_sha256(self.context_axis_hash, "context axis hash")
        if int(self.fold_count) < 2:
            raise ValueError("glocal context design requires at least two folds")
        calibration = tuple(str(value) for value in self.calibration_context_ids)
        adjudication = tuple(str(value) for value in self.adjudication_context_ids)
        terminal = tuple(str(value) for value in self.terminal_context_ids)
        if not calibration or not adjudication:
            raise ValueError(
                "glocal design requires calibration and adjudication populations"
            )
        combined = calibration + adjudication + terminal
        if len(combined) != len(set(combined)):
            raise ValueError(
                "calibration, adjudication, and terminal contexts must be disjoint"
            )
        labels = dict(self.label_by_context_id)
        strata = dict(self.stratum_by_context_id)
        if set(labels) != set(combined) or set(strata) != set(combined):
            raise ValueError("glocal context coordinates do not cover every role row")
        if self.calibration_observation_set.role is not EvidenceRole.CALIBRATION_FIT:
            raise ValueError("glocal calibration set has the wrong statistical role")
        calibration_design = self.calibration_sampling_design
        initial_targets = tuple(
            sorted(
                (str(stratum_id), int(target))
                for stratum_id, target in self.calibration_initial_targets_by_stratum
            )
        )
        if calibration_design is None:
            if initial_targets:
                raise ValueError(
                    "static calibration census cannot declare adaptive prefix targets"
                )
            if (
                self.calibration_observation_set.context_ids != calibration
                or self.calibration_observation_set.stratum_ids
                != tuple(strata[context_id] for context_id in calibration)
            ):
                raise ValueError("glocal calibration observation identity differs")
        else:
            if (
                calibration_design.design_kind
                != "sealed_uniform_without_replacement_stratified"
                or not calibration_design.claim_bearing
            ):
                raise ValueError(
                    "adaptive calibration support requires a claim-bearing sealed "
                    "finite-population design"
                )
            sealed_contexts = tuple(
                context_id
                for item in calibration_design.strata
                for context_id in item.context_ids_in_sealed_order
            )
            if set(sealed_contexts) != set(calibration) or len(sealed_contexts) != len(
                calibration
            ):
                raise ValueError(
                    "adaptive calibration design differs from its role-exclusive "
                    "support population"
                )
            target_map = dict(initial_targets)
            capacities = {
                item.stratum_id: item.population_size
                for item in calibration_design.strata
            }
            if set(target_map) != set(capacities) or any(
                target <= 0 or target > capacities[stratum_id]
                for stratum_id, target in target_map.items()
            ):
                raise ValueError(
                    "adaptive calibration initial targets must cover every "
                    "nonempty stratum within capacity"
                )
            expected_contexts = tuple(
                context_id
                for item in calibration_design.strata
                for context_id in item.context_ids_in_sealed_order[
                    : target_map[item.stratum_id]
                ]
            )
            expected_strata = tuple(
                item.stratum_id
                for item in calibration_design.strata
                for _ in range(target_map[item.stratum_id])
            )
            if (
                self.calibration_observation_set.context_ids != expected_contexts
                or self.calibration_observation_set.stratum_ids != expected_strata
            ):
                raise ValueError(
                    "initial calibration observation set is not the frozen "
                    "within-stratum prefix"
                )
            self.calibration_observation_set.validate_against(calibration_design)
        for contexts, label in (
            (calibration, "calibration"),
            (adjudication, "adjudication"),
        ):
            binary = {
                int(labels[context_id]) == int(self.positive_class_id)
                for context_id in contexts
            }
            if binary != {False, True}:
                raise ValueError(f"{label} frame must contain both binary sides")
        risk_strata = set(dict(dict(self.risk_estimand.strata_by_query)[self.query_id]))
        if {strata[context_id] for context_id in adjudication} != risk_strata:
            raise ValueError("adjudication frame does not cover the frozen risk strata")
        object.__setattr__(self, "calibration_context_ids", calibration)
        object.__setattr__(self, "adjudication_context_ids", adjudication)
        object.__setattr__(self, "terminal_context_ids", terminal)
        object.__setattr__(
            self,
            "calibration_initial_targets_by_stratum",
            initial_targets,
        )
        object.__setattr__(
            self,
            "label_by_context_id",
            tuple(sorted((str(key), int(value)) for key, value in labels.items())),
        )
        object.__setattr__(
            self,
            "stratum_by_context_id",
            tuple(sorted((str(key), str(value)) for key, value in strata.items())),
        )
        object.__setattr__(self, "fold_count", int(self.fold_count))

    @property
    def content_hash(self) -> str:
        return content_hash(self)

    @property
    def labels(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self.label_by_context_id))

    @property
    def strata(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.stratum_by_context_id))

    @property
    def adaptive_calibration(self) -> bool:
        return self.calibration_sampling_design is not None

    @property
    def calibration_initial_targets(self) -> Mapping[str, int]:
        if self.calibration_sampling_design is None:
            counts: dict[str, int] = {}
            for context_id in self.calibration_context_ids:
                stratum_id = self.strata[context_id]
                counts[stratum_id] = counts.get(stratum_id, 0) + 1
            return MappingProxyType(counts)
        return MappingProxyType(dict(self.calibration_initial_targets_by_stratum))

    def expand_calibration_targets(
        self,
        current_targets: Mapping[str, int],
        *,
        additional_contexts: int,
    ) -> dict[str, int]:
        """Choose the next sealed prefixes by predictable balanced information.

        The working unit-variance score is allocation-only.  It is the
        calibration analogue of packet Neyman allocation before any new
        outcomes are visible; fit covariance determines the requested total,
        while this rule maps that total onto the frozen class strata.
        """

        design = self.calibration_sampling_design
        current = {str(key): int(value) for key, value in current_targets.items()}
        increment = int(additional_contexts)
        if increment < 0:
            raise ValueError("calibration prefix increment must be nonnegative")
        if design is None:
            full = dict(self.calibration_initial_targets)
            if current != full or increment:
                raise RuntimeError(
                    "static calibration census has no adaptive support remaining"
                )
            return full
        capacities = {item.stratum_id: item.population_size for item in design.strata}
        weights = {item.stratum_id: item.risk_weight for item in design.strata}
        if set(current) != set(capacities) or any(
            value < 0 or value > capacities[key] for key, value in current.items()
        ):
            raise ValueError(
                "current calibration targets differ from the frozen design"
            )
        updated = dict(current)
        for _ in range(increment):
            available = [
                stratum_id
                for stratum_id in sorted(updated)
                if updated[stratum_id] < capacities[stratum_id]
            ]
            if not available:
                break
            selected = max(
                available,
                key=lambda stratum_id: (
                    weights[stratum_id] ** 2
                    / (max(updated[stratum_id], 1) * (updated[stratum_id] + 1)),
                    stratum_id,
                ),
            )
            updated[selected] += 1
        return updated

    def calibration_observation_set_for_targets(
        self,
        targets: Mapping[str, int] | None = None,
    ) -> ObservationSetRef:
        design = self.calibration_sampling_design
        if design is None:
            if targets is not None and dict(targets) != dict(
                self.calibration_initial_targets
            ):
                raise ValueError("static calibration census targets cannot change")
            return self.calibration_observation_set
        resolved = (
            dict(self.calibration_initial_targets)
            if targets is None
            else {str(key): int(value) for key, value in targets.items()}
        )
        capacities = {item.stratum_id: item.population_size for item in design.strata}
        if set(resolved) != set(capacities) or any(
            value <= 0 or value > capacities[key] for key, value in resolved.items()
        ):
            raise ValueError("calibration observation prefix targets are invalid")
        contexts = tuple(
            context_id
            for item in design.strata
            for context_id in item.context_ids_in_sealed_order[
                : resolved[item.stratum_id]
            ]
        )
        stratum_ids = tuple(
            item.stratum_id
            for item in design.strata
            for _ in range(resolved[item.stratum_id])
        )
        prefix_hash = content_hash(tuple(sorted(resolved.items())))
        observation_set = ObservationSetRef(
            observation_set_id=(f"{self.design_id}:calibration-prefix:{prefix_hash}"),
            role=EvidenceRole.CALIBRATION_FIT,
            sampling_design_hash=design.content_hash,
            context_ids=contexts,
            stratum_ids=stratum_ids,
        )
        observation_set.validate_against(design)
        return observation_set

    def calibration_panel(
        self,
        targets: Mapping[str, int] | None = None,
    ) -> CrossfitPanel:
        labels = self.labels
        strata = self.strata
        observation_set = self.calibration_observation_set_for_targets(targets)
        ordered = observation_set.context_ids
        if self.calibration_sampling_design is None:
            fold_ids = tuple(index % self.fold_count for index in range(len(ordered)))
        else:
            full_order = tuple(
                context_id
                for item in self.calibration_sampling_design.strata
                for context_id in item.context_ids_in_sealed_order
            )
            fold_by_context = {
                context_id: index % self.fold_count
                for index, context_id in enumerate(full_order)
            }
            fold_ids = tuple(fold_by_context[context_id] for context_id in ordered)
        resolved_targets: dict[str, int] = {}
        for stratum_id in observation_set.stratum_ids:
            resolved_targets[stratum_id] = resolved_targets.get(stratum_id, 0) + 1
        panel = CrossfitPanel(
            context_ids=ordered,
            fold_ids=fold_ids,
            labels=tuple(labels[context_id] for context_id in ordered),
            draw_id=(
                content_hash(
                    {
                        "context_design_hash": self.content_hash,
                        "role": EvidenceRole.CALIBRATION_FIT.value,
                    }
                )
                if self.calibration_sampling_design is None
                else self.calibration_sampling_design.randomization_commitment_hash
            ),
            stage_index=len(ordered),
            allocation_metadata={
                "foundation": EVIDENCE_FOUNDATION_GLOCAL_V3,
                "statistical_role": EvidenceRole.CALIBRATION_FIT.value,
                "stratum_counts": {
                    stratum_id: resolved_targets[stratum_id]
                    for stratum_id in sorted(resolved_targets)
                },
                "adaptive_calibration_support": self.adaptive_calibration,
                "sampling_design_hash": (
                    None
                    if self.calibration_sampling_design is None
                    else self.calibration_sampling_design.content_hash
                ),
                "observation_set_hash": observation_set.content_hash,
                "requested_per_stratum_target_images": resolved_targets,
                "requested_cumulative_target_images": len(ordered),
            },
        )
        panel.validate(fold_count=self.fold_count)
        return panel

    def freeze_proposal_design(
        self,
        *,
        proposal_id: str,
        candidate_definition_hash: str,
        random_seed: int,
    ) -> tuple[SamplingDesignSpec, ObservationSetRef]:
        """Draw and seal independent uniform permutations after candidate birth."""

        _require_sha256(candidate_definition_hash, "candidate definition hash")
        by_stratum: dict[str, list[str]] = {}
        strata_by_context = self.strata
        for context_id in self.adjudication_context_ids:
            by_stratum.setdefault(strata_by_context[context_id], []).append(context_id)
        risk_weights = dict(dict(self.risk_estimand.strata_by_query)[self.query_id])
        binary_mapping = dict(
            dict(self.risk_estimand.binary_class_by_stratum_by_query)[self.query_id]
        )
        birth_binding = {
            "role": EvidenceRole.PROPOSAL_ADJUDICATION.value,
            "context_design_hash": self.content_hash,
            "proposal_id": str(proposal_id),
            "candidate_definition_hash": str(candidate_definition_hash),
            "random_seed_namespace": str(int(random_seed)),
        }
        strata: list[SamplingStratumSpec] = []
        receipts: list[SamplingPermutationReceipt] = []
        for stratum_id in sorted(by_stratum):
            raw = tuple(by_stratum[stratum_id])
            receipt = SamplingPermutationReceipt.draw(
                stratum_id=stratum_id,
                population=raw,
                birth_binding=birth_binding,
            )
            sealed = receipt.generated_order
            receipts.append(receipt)
            strata.append(
                SamplingStratumSpec.from_context_ids(
                    stratum_id=stratum_id,
                    context_ids_in_sealed_order=sealed,
                    risk_weight=float(risk_weights[stratum_id]),
                )
            )
        design = SamplingDesignSpec(
            design_id=f"{self.design_id}:{proposal_id}",
            context_population_hash=self.risk_estimand.context_population_hash,
            randomization_commitment_hash=(
                SamplingDesignSpec.randomization_commitment_for(receipts)
            ),
            strata=tuple(strata),
            permutation_receipts=tuple(receipts),
            binary_class_by_stratum=tuple(binary_mapping.items()),
        )
        observation_set = ObservationSetRef(
            observation_set_id=f"{self.design_id}:{proposal_id}:adjudication",
            role=EvidenceRole.PROPOSAL_ADJUDICATION,
            sampling_design_hash=design.content_hash,
            context_ids=tuple(
                context_id
                for stratum in design.strata
                for context_id in stratum.context_ids_in_sealed_order
            ),
            stratum_ids=tuple(
                stratum.stratum_id
                for stratum in design.strata
                for _ in stratum.context_ids_in_sealed_order
            ),
        )
        observation_set.validate_against(design)
        return design, observation_set

    def proposal_panel(
        self,
        design: SamplingDesignSpec,
        *,
        targets: Mapping[str, int],
        stage_index: int,
        prior_panel: CrossfitPanel | None,
    ) -> CrossfitPanel:
        labels = self.labels
        context_ids: list[str] = []
        fold_ids: list[int] = []
        panel_labels: list[int] = []
        normalized_targets = {str(key): int(value) for key, value in targets.items()}
        if set(normalized_targets) != {item.stratum_id for item in design.strata}:
            raise ValueError("panel target axis differs from the sampling design")
        for stratum_index, stratum in enumerate(design.strata):
            count = normalized_targets[stratum.stratum_id]
            if not 0 <= count <= stratum.population_size:
                raise ValueError("panel target leaves its finite stratum")
            for rank, context_id in enumerate(
                stratum.context_ids_in_sealed_order[:count]
            ):
                context_ids.append(context_id)
                fold_ids.append((rank + stratum_index) % self.fold_count)
                panel_labels.append(labels[context_id])
        panel = CrossfitPanel(
            context_ids=tuple(context_ids),
            fold_ids=tuple(fold_ids),
            labels=tuple(panel_labels),
            draw_id=design.randomization_commitment_hash,
            stage_index=int(stage_index),
            allocation_metadata={
                "foundation": EVIDENCE_FOUNDATION_GLOCAL_V3,
                "allocation_rule": GLOCAL_NEYMAN_ALLOCATION_RULE,
                "sampling_design_hash": design.content_hash,
                "requested_per_stratum_target_images": dict(normalized_targets),
                "requested_cumulative_target_images": len(context_ids),
                "speculative_prefetch": False,
            },
        )
        panel.validate(fold_count=self.fold_count)
        if prior_panel is not None:
            panel.assert_nested_extension(prior_panel)
        return panel

    def terminal_panel(
        self,
        design: SamplingDesignSpec,
        *,
        targets: Mapping[str, int],
    ) -> CrossfitPanel:
        labels = self.labels
        source_by_draw = (
            {
                draw.draw_id: draw.source_unit_id
                for draw in design.iid_sampling_receipt.draws
            }
            if design.iid_sampling_receipt is not None
            else {}
        )
        normalized_targets = {str(key): int(value) for key, value in targets.items()}
        if set(normalized_targets) != {item.stratum_id for item in design.strata}:
            raise ValueError("terminal target axis differs from its sampling design")
        context_ids: list[str] = []
        fold_ids: list[int] = []
        panel_labels: list[int] = []
        for stratum_index, stratum in enumerate(design.strata):
            target = normalized_targets[stratum.stratum_id]
            if not 0 < target <= stratum.population_size:
                raise ValueError("terminal target leaves its finite stratum")
            for rank, context_id in enumerate(
                stratum.context_ids_in_sealed_order[:target]
            ):
                context_ids.append(context_id)
                fold_ids.append((rank + stratum_index) % self.fold_count)
                source_unit_id = source_by_draw.get(context_id, context_id)
                if source_unit_id not in labels:
                    raise ValueError(
                        "terminal draw references a source unit outside the frozen "
                        "context design"
                    )
                panel_labels.append(labels[source_unit_id])
        panel = CrossfitPanel(
            context_ids=tuple(context_ids),
            fold_ids=tuple(fold_ids),
            labels=tuple(panel_labels),
            draw_id=design.randomization_commitment_hash,
            stage_index=0,
            allocation_metadata={
                "foundation": EVIDENCE_FOUNDATION_GLOCAL_V3,
                "statistical_role": EvidenceRole.TERMINAL_AUDIT.value,
                "sampling_design_hash": design.content_hash,
                "iid_draw_mapping_hash": (
                    None
                    if design.iid_sampling_receipt is None
                    else design.iid_sampling_receipt.content_hash
                ),
                "requested_per_stratum_target_images": normalized_targets,
                "requested_cumulative_target_images": len(context_ids),
                "speculative_prefetch": False,
            },
        )
        panel.validate(fold_count=self.fold_count)
        return panel


def freeze_binary_context_design(
    axis: ContextAxis,
    *,
    design_id: str,
    query_id: str,
    positive_class_id: int,
    calibration_context_ids: Sequence[str],
    adjudication_context_ids: Sequence[str],
    terminal_context_ids: Sequence[str] = (),
    calibration_initial_per_stratum: int | Mapping[str, int] | None = None,
    fold_count: int = 2,
) -> FrozenBinaryContextDesign:
    """Freeze exact role membership and the class-balanced finite estimand."""

    axis.validate()
    by_id = {record.context_id: record for record in axis.records}
    calibration = tuple(str(value) for value in calibration_context_ids)
    adjudication = tuple(str(value) for value in adjudication_context_ids)
    terminal = tuple(str(value) for value in terminal_context_ids)
    requested = calibration + adjudication + terminal
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(
            "glocal design references unknown contexts: " + ",".join(unknown)
        )
    if len(requested) != len(set(requested)):
        raise ValueError("glocal evidence roles must use disjoint context sets")
    target = int(positive_class_id)

    def stratum(context_id: str) -> str:
        label = int(by_id[context_id].label)
        return "positive" if label == target else f"neg:{label}"

    adjudication_strata = tuple(stratum(value) for value in adjudication)
    positive_count = sum(value == "positive" for value in adjudication_strata)
    negative_count = len(adjudication_strata) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("adjudication population requires both binary sides")
    negative_counts = {
        stratum_id: adjudication_strata.count(stratum_id)
        for stratum_id in sorted(set(adjudication_strata) - {"positive"})
    }
    stratum_weights = {"positive": 0.5}
    stratum_weights.update(
        {
            stratum_id: 0.5 * count / negative_count
            for stratum_id, count in negative_counts.items()
        }
    )
    binary_mapping = {
        stratum_id: (1 if stratum_id == "positive" else 0)
        for stratum_id in stratum_weights
    }
    population_hash = content_hash(
        {
            "axis_hash": axis.content_hash,
            "query_id": str(query_id),
            "positive_class_id": target,
            "adjudication_contexts": tuple(
                (
                    context_id,
                    int(by_id[context_id].label),
                    stratum(context_id),
                )
                for context_id in adjudication
            ),
            "weighting": "binary_class_balanced_with_population_weighted_negative_substrata",
        }
    )
    risk = RiskEstimandSpec.from_mappings(
        estimand_id=f"{design_id}:bounded-brier",
        context_population_hash=population_hash,
        prediction_rule=PredictionRuleSpec(
            kind="binary",
            class_ids=(0, 1),
            positive_class_id=1,
        ),
        query_weights={str(query_id): 1.0},
        strata_by_query={str(query_id): stratum_weights},
        binary_class_by_stratum_by_query={str(query_id): binary_mapping},
    )
    all_contexts = requested
    labels = tuple(
        (context_id, int(by_id[context_id].label)) for context_id in all_contexts
    )
    strata = tuple((context_id, stratum(context_id)) for context_id in all_contexts)
    calibration_sampling_design = None
    calibration_initial_targets: tuple[tuple[str, int], ...] = ()
    if calibration_initial_per_stratum is None:
        calibration_design_hash = content_hash(
            {
                "context_design_id": str(design_id),
                "axis_hash": axis.content_hash,
                "role": EvidenceRole.CALIBRATION_FIT.value,
                "context_ids": calibration,
                "strata": tuple(stratum(value) for value in calibration),
            }
        )
        calibration_set = ObservationSetRef(
            observation_set_id=f"{design_id}:calibration",
            role=EvidenceRole.CALIBRATION_FIT,
            sampling_design_hash=calibration_design_hash,
            context_ids=calibration,
            stratum_ids=tuple(stratum(value) for value in calibration),
        )
    else:
        by_calibration_stratum: dict[str, list[str]] = {}
        for context_id in calibration:
            by_calibration_stratum.setdefault(stratum(context_id), []).append(
                context_id
            )
        if {
            1 if stratum_id == "positive" else 0
            for stratum_id in by_calibration_stratum
        } != {0, 1}:
            raise ValueError(
                "adaptive calibration support must contain both binary sides"
            )
        negative_total = sum(
            len(contexts)
            for stratum_id, contexts in by_calibration_stratum.items()
            if stratum_id != "positive"
        )
        calibration_weights = {"positive": 0.5}
        calibration_weights.update(
            {
                stratum_id: 0.5 * len(contexts) / negative_total
                for stratum_id, contexts in by_calibration_stratum.items()
                if stratum_id != "positive"
            }
        )
        calibration_population_hash = content_hash(
            {
                "axis_hash": axis.content_hash,
                "design_id": str(design_id),
                "query_id": str(query_id),
                "role": EvidenceRole.CALIBRATION_FIT.value,
                "support": tuple(
                    sorted(
                        (
                            context_id,
                            stratum(context_id),
                            int(by_id[context_id].label),
                        )
                        for context_id in calibration
                    )
                ),
            }
        )
        birth_binding = {
            "role": EvidenceRole.CALIBRATION_FIT.value,
            "design_id": str(design_id),
            "query_id": str(query_id),
            "context_population_hash": calibration_population_hash,
        }
        permutation_receipts = tuple(
            SamplingPermutationReceipt.draw(
                stratum_id=stratum_id,
                population=contexts,
                birth_binding=birth_binding,
            )
            for stratum_id, contexts in sorted(by_calibration_stratum.items())
        )
        calibration_strata = tuple(
            SamplingStratumSpec.from_context_ids(
                stratum_id=receipt.stratum_id,
                context_ids_in_sealed_order=receipt.generated_order,
                risk_weight=calibration_weights[receipt.stratum_id],
            )
            for receipt in permutation_receipts
        )
        calibration_sampling_design = SamplingDesignSpec(
            design_id=f"{design_id}:calibration-support",
            context_population_hash=calibration_population_hash,
            randomization_commitment_hash=(
                SamplingDesignSpec.randomization_commitment_for(permutation_receipts)
            ),
            strata=calibration_strata,
            permutation_receipts=permutation_receipts,
            binary_class_by_stratum=tuple(
                sorted(
                    (
                        stratum_id,
                        1 if stratum_id == "positive" else 0,
                    )
                    for stratum_id in by_calibration_stratum
                )
            ),
        )
        if isinstance(calibration_initial_per_stratum, Mapping):
            initial = {
                str(key): int(value)
                for key, value in calibration_initial_per_stratum.items()
            }
        else:
            initial = {
                item.stratum_id: min(
                    int(calibration_initial_per_stratum),
                    item.population_size,
                )
                for item in calibration_strata
            }
        capacities = {
            item.stratum_id: item.population_size for item in calibration_strata
        }
        if (
            set(initial) != set(capacities)
            or any(
                value <= 0 or value > capacities[stratum_id]
                for stratum_id, value in initial.items()
            )
            or sum(initial.values()) < max(4, int(fold_count))
        ):
            raise ValueError(
                "adaptive calibration initial prefix must cover every stratum, "
                "fit the declared folds, and contain at least four rows"
            )
        initial_contexts = tuple(
            context_id
            for item in calibration_strata
            for context_id in item.context_ids_in_sealed_order[
                : initial[item.stratum_id]
            ]
        )
        initial_strata = tuple(
            item.stratum_id
            for item in calibration_strata
            for _ in range(initial[item.stratum_id])
        )
        calibration_initial_targets = tuple(sorted(initial.items()))
        calibration_set = ObservationSetRef(
            observation_set_id=(
                f"{design_id}:calibration-prefix:"
                f"{content_hash(calibration_initial_targets)}"
            ),
            role=EvidenceRole.CALIBRATION_FIT,
            sampling_design_hash=calibration_sampling_design.content_hash,
            context_ids=initial_contexts,
            stratum_ids=initial_strata,
        )
        calibration_set.validate_against(calibration_sampling_design)
    return FrozenBinaryContextDesign(
        design_id=str(design_id),
        query_id=str(query_id),
        positive_class_id=target,
        context_axis_hash=axis.content_hash,
        risk_estimand=risk,
        calibration_context_ids=calibration,
        adjudication_context_ids=adjudication,
        terminal_context_ids=terminal,
        label_by_context_id=labels,
        stratum_by_context_id=strata,
        calibration_observation_set=calibration_set,
        calibration_sampling_design=calibration_sampling_design,
        calibration_initial_targets_by_stratum=calibration_initial_targets,
        fold_count=int(fold_count),
    )


@dataclass(frozen=True, slots=True)
class GlocalNeymanPolicy:
    """Predictable packet cadence; variance moderation is allocation-only."""

    initial_images_per_stratum: int = 2
    packet_images: int = 16
    maximum_adjudication_images: int | None = None
    maximum_looks: int | None = None
    allocation_variance_prior: float = 1.0
    variance_moderation_strength: float = 4.0
    initial_cost_per_context: float = 1.0
    minimum_information_value_per_cost: float = 0.0
    schema_name: str = "glocal_neyman_policy_v1"

    def __post_init__(self) -> None:
        if int(self.initial_images_per_stratum) <= 0 or int(self.packet_images) <= 0:
            raise ValueError("glocal Neyman packet counts must be positive")
        if (
            self.maximum_adjudication_images is not None
            and int(self.maximum_adjudication_images) <= 0
        ):
            raise ValueError("maximum adjudication images must be positive")
        if self.maximum_looks is not None and int(self.maximum_looks) <= 0:
            raise ValueError("maximum looks must be positive")
        for name in (
            "allocation_variance_prior",
            "variance_moderation_strength",
            "initial_cost_per_context",
            "minimum_information_value_per_cost",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.initial_cost_per_context <= 0.0:
            raise ValueError("initial cost per context must be positive")

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class GlocalPacketResourceEnvelope:
    """Frozen conservative maximum for one atomic backend call."""

    forward_passes: int
    backward_passes: int = 0
    calibration_fits: int = 0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    wall_seconds: float = 0.0
    device_seconds: float = 0.0
    peak_memory_bytes: int = 0

    def __post_init__(self) -> None:
        usage = ResourceUsage(
            forward_passes=self.forward_passes,
            backward_passes=self.backward_passes,
            calibration_fits=self.calibration_fits,
            host_to_device_bytes=self.host_to_device_bytes,
            device_to_host_bytes=self.device_to_host_bytes,
            wall_seconds=self.wall_seconds,
            device_seconds=self.device_seconds,
            peak_memory_bytes=self.peak_memory_bytes,
        )
        object.__setattr__(self, "forward_passes", usage.forward_passes)


@dataclass(frozen=True, slots=True)
class IRGNativeMaskLoweringSpec:
    """Content identity for the IRG route-axis to backend-mask adapter."""

    adapter_id: str
    adapter_version: str
    code_hash: str
    route_axis_hash: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    schema_name: str = "irg_native_mask_lowering_spec_v1"

    def __post_init__(self) -> None:
        if (
            not self.adapter_id
            or not self.adapter_version
            or not self.route_axis_hash
            or not self.schema_name
        ):
            raise ValueError("native-mask lowering spec has an empty identity field")
        _require_sha256(self.code_hash, "native-mask adapter code hash")
        object.__setattr__(self, "parameters", dict(self.parameters))
        content_hash(self.parameters)

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class GlocalIRGCommitPolicy:
    """Frozen Brier boundary and predictable seed schedule for IRG commits."""

    policy_id: str
    local_upper_boundary: float
    calibration_transition_upper_boundary: float = 0.0
    base_random_seed: int = 0
    group_policy: Literal["geometric_partition_or_global"] = (
        "geometric_partition_or_global"
    )
    schema_name: str = "glocal_irg_commit_policy_v2"

    def __post_init__(self) -> None:
        boundary = float(self.local_upper_boundary)
        calibration_boundary = float(self.calibration_transition_upper_boundary)
        if not self.policy_id or not self.schema_name:
            raise ValueError("glocal IRG commit policy requires stable identity")
        if not math.isfinite(boundary) or not math.isfinite(calibration_boundary):
            raise ValueError("glocal IRG Brier boundaries must be finite")
        if int(self.base_random_seed) < 0:
            raise ValueError("glocal IRG base random seed must be nonnegative")
        if self.group_policy != "geometric_partition_or_global":
            raise ValueError("unsupported glocal IRG grouping policy")
        object.__setattr__(self, "local_upper_boundary", boundary)
        object.__setattr__(
            self,
            "calibration_transition_upper_boundary",
            calibration_boundary,
        )
        object.__setattr__(self, "base_random_seed", int(self.base_random_seed))

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class GlocalTerminalAuditPolicy:
    """Frozen one-shot terminal parity design."""

    policy_id: str
    parity_tolerance: float
    sampling_design: Literal[
        "finite_census",
        "finite_population_wor",
        "iid_fixed_n",
    ] = "finite_census"
    random_seed: int = 0
    finite_population_sample_per_stratum: int | None = None
    iid_sample_per_stratum: int | None = None
    iid_sampling_receipt: IIDSamplingReceipt | None = None
    schema_name: str = "glocal_terminal_audit_policy_v2"

    def __post_init__(self) -> None:
        if not self.policy_id or not self.schema_name:
            raise ValueError("terminal audit policy requires stable identity")
        tolerance = float(self.parity_tolerance)
        if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
            raise ValueError("terminal parity tolerance must lie in [0,1]")
        if int(self.random_seed) < 0:
            raise ValueError("terminal audit random seed must be nonnegative")
        if self.sampling_design == "finite_population_wor":
            if (
                self.finite_population_sample_per_stratum is None
                or int(self.finite_population_sample_per_stratum) <= 0
            ):
                raise ValueError(
                    "finite-population audit requires a positive stratum sample size"
                )
            if self.iid_sampling_receipt is not None:
                raise ValueError(
                    "finite-population audit cannot carry an iid sampling receipt"
                )
            if self.iid_sample_per_stratum is not None:
                raise ValueError(
                    "finite-population audit cannot carry an IID sample target"
                )
        elif self.sampling_design == "iid_fixed_n":
            if self.finite_population_sample_per_stratum is not None:
                raise ValueError(
                    "fixed-n IID audit cannot carry a finite-population prefix"
                )
            has_target = (
                self.iid_sample_per_stratum is not None
                and int(self.iid_sample_per_stratum) > 0
            )
            has_replay_receipt = isinstance(
                self.iid_sampling_receipt,
                IIDSamplingReceipt,
            )
            if has_target == has_replay_receipt:
                raise ValueError(
                    "fixed-n IID audit requires exactly one positive runtime "
                    "sample target or one replay-only receipt"
                )
        elif self.sampling_design == "finite_census":
            if (
                self.finite_population_sample_per_stratum is not None
                or self.iid_sample_per_stratum is not None
                or self.iid_sampling_receipt is not None
            ):
                raise ValueError("terminal census cannot carry sampling-prefix options")
        else:
            raise ValueError("unsupported terminal audit sampling design")
        object.__setattr__(self, "parity_tolerance", tolerance)
        object.__setattr__(self, "random_seed", int(self.random_seed))
        if self.finite_population_sample_per_stratum is not None:
            object.__setattr__(
                self,
                "finite_population_sample_per_stratum",
                int(self.finite_population_sample_per_stratum),
            )
        if self.iid_sample_per_stratum is not None:
            object.__setattr__(
                self,
                "iid_sample_per_stratum",
                int(self.iid_sample_per_stratum),
            )

    @property
    def content_hash(self) -> str:
        return content_hash(self)


def _terminal_support_by_stratum(
    context_design: FrozenBinaryContextDesign,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not context_design.terminal_context_ids:
        raise ValueError("glocal terminal audit requires untouched terminal contexts")
    by_stratum: dict[str, list[str]] = {}
    context_strata = context_design.strata
    for context_id in context_design.terminal_context_ids:
        by_stratum.setdefault(context_strata[context_id], []).append(context_id)
    risk_weights = dict(
        dict(context_design.risk_estimand.strata_by_query)[context_design.query_id]
    )
    if set(by_stratum) != set(risk_weights):
        raise ValueError("terminal audit does not cover every frozen risk stratum")
    return tuple(
        (stratum_id, tuple(sorted(by_stratum[stratum_id])))
        for stratum_id in sorted(by_stratum)
    )


def _build_terminal_precommitment(
    *,
    context_design: FrozenBinaryContextDesign,
    policy: GlocalTerminalAuditPolicy,
) -> tuple[
    str,
    tuple[tuple[str, tuple[str, ...]], ...],
    dict[str, int],
    tuple[tuple[str, float], ...],
]:
    support = _terminal_support_by_stratum(context_design)
    risk_weights = tuple(
        sorted(
            dict(
                dict(context_design.risk_estimand.strata_by_query)[
                    context_design.query_id
                ]
            ).items()
        )
    )
    terminal_population_hash = content_hash(
        {
            "schema_name": "glocal_terminal_population_v1",
            "context_design_hash": context_design.content_hash,
            "query_id": context_design.query_id,
            "prediction_rule_hash": (
                context_design.risk_estimand.prediction_rule.content_hash
            ),
            "terminal_support": tuple(
                (
                    context_id,
                    context_design.labels[context_id],
                    context_design.strata[context_id],
                )
                for _, context_ids in support
                for context_id in context_ids
            ),
            "stratum_weights": risk_weights,
            "parity_event": "declared_decision_disagreement",
        }
    )
    if terminal_population_hash == context_design.risk_estimand.context_population_hash:
        raise ValueError(
            "terminal support must have an identity distinct from adjudication risk"
        )
    support_hash = TerminalAuditPrecommitment.support_distribution_hash_for(
        terminal_population_hash=terminal_population_hash,
        support_context_ids_by_stratum=support,
    )
    support_sizes = {
        stratum_id: len(context_ids) for stratum_id, context_ids in support
    }
    if policy.sampling_design == "finite_population_wor":
        assert policy.finite_population_sample_per_stratum is not None
        targets = {
            stratum_id: min(
                size,
                policy.finite_population_sample_per_stratum,
            )
            for stratum_id, size in support_sizes.items()
        }
    elif policy.sampling_design == "iid_fixed_n":
        if policy.iid_sampling_receipt is not None:
            raise ValueError(
                "caller-supplied IID draw receipts are replay-only; the live "
                "runtime requires iid_sample_per_stratum and draws after lock"
            )
        assert policy.iid_sample_per_stratum is not None
        targets = {
            stratum_id: policy.iid_sample_per_stratum for stratum_id in support_sizes
        }
    else:
        targets = dict(support_sizes)
    return terminal_population_hash, support, targets, risk_weights


def _terminal_precommitment(
    *,
    context_design: FrozenBinaryContextDesign,
    policy: GlocalTerminalAuditPolicy,
    initial_state_hash: str,
    initial_calibration_hash: str,
    backend: TorchSampledPathEvaluationBackend,
    terminal_envelope: GlocalPacketResourceEnvelope,
    terminal_alpha: float,
) -> TerminalAuditPrecommitment:
    population_hash, support, targets, stratum_weights = _build_terminal_precommitment(
        context_design=context_design,
        policy=policy,
    )
    provider = getattr(backend, "panel_source", None)
    if provider is None:
        raise ValueError(
            "terminal precommitment requires a physical panel-source identity"
        )
    support_hash = TerminalAuditPrecommitment.support_distribution_hash_for(
        terminal_population_hash=population_hash,
        support_context_ids_by_stratum=support,
    )
    estimand = TerminalParityEstimandSpec(
        estimand_id=f"{context_design.design_id}:{policy.policy_id}:parity",
        reference_state_hash=initial_state_hash,
        reference_calibration_hash=initial_calibration_hash,
        adjudication_population_hash=(
            context_design.risk_estimand.context_population_hash
        ),
        terminal_population_hash=population_hash,
        terminal_support_distribution_hash=support_hash,
        prediction_rule_hash=(
            context_design.risk_estimand.prediction_rule.content_hash
        ),
        query_weights=context_design.risk_estimand.query_weights,
        stratum_weights=stratum_weights,
        parity_tolerance=policy.parity_tolerance,
    )
    return TerminalAuditPrecommitment(
        precommitment_id=f"{context_design.design_id}:{policy.policy_id}:birth",
        estimand=estimand,
        design_kind=policy.sampling_design,
        support_context_ids_by_stratum=support,
        target_sample_sizes_by_stratum=tuple(sorted(targets.items())),
        terminal_alpha=float(terminal_alpha),
        policy_hash=policy.content_hash,
        randomization_method=(
            GLOCAL_IID_RUNTIME_RANDOMIZATION_METHOD_V2
            if policy.sampling_design == "iid_fixed_n"
            else GLOCAL_SEALED_PERMUTATION_ALGORITHM_V1
        ),
        provider_identity_hash=iid_physical_provider_identity_hash(provider),
        executor_identity_hash=iid_physical_executor_identity_hash(backend),
        resource_envelope_hash=content_hash(terminal_envelope),
    )


def _freeze_terminal_design(
    context_design: FrozenBinaryContextDesign,
    policy: GlocalTerminalAuditPolicy,
    precommitment: TerminalAuditPrecommitment,
    *,
    iid_sampling_receipt: IIDSamplingReceipt | None = None,
) -> tuple[SamplingDesignSpec, ObservationSetRef, dict[str, int]]:
    if (
        precommitment.policy_hash != policy.content_hash
        or precommitment.design_kind != policy.sampling_design
    ):
        raise ValueError("terminal design differs from its birth precommitment")
    support = dict(precommitment.support_context_ids_by_stratum)
    risk_weights = dict(precommitment.estimand.stratum_weights)
    targets = dict(precommitment.target_sample_sizes_by_stratum)
    birth_binding = {
        "role": EvidenceRole.TERMINAL_AUDIT.value,
        "context_design_hash": context_design.content_hash,
        "terminal_policy_hash": policy.content_hash,
        "terminal_precommitment_hash": precommitment.content_hash,
        "random_seed_namespace": str(policy.random_seed),
    }
    strata: list[SamplingStratumSpec] = []
    receipts: list[SamplingPermutationReceipt] = []
    if policy.sampling_design == "iid_fixed_n":
        iid_receipt = iid_sampling_receipt
        if iid_receipt is None:
            raise ValueError("live IID terminal design lacks runtime-generated draws")
        if iid_receipt.context_population_hash != (
            precommitment.estimand.terminal_population_hash
        ):
            raise ValueError("fixed-n IID receipt differs from the terminal population")
        frozen_support = {
            item.stratum_id: tuple(
                source_unit_id for source_unit_id, _ in item.source_unit_probabilities
            )
            for item in iid_receipt.support_distributions
        }
        if set(frozen_support) != set(support) or any(
            set(frozen_support[stratum_id]) != set(support[stratum_id])
            for stratum_id in support
        ):
            raise ValueError(
                "fixed-n IID support law differs from the untouched terminal "
                "source support"
            )
        draws_by_stratum: dict[str, list[IIDDrawRecord]] = {
            stratum_id: [] for stratum_id in support
        }
        for draw in iid_receipt.draws:
            draws_by_stratum[draw.stratum_id].append(draw)
        for stratum_id in sorted(draws_by_stratum):
            draw_ids = tuple(draw.draw_id for draw in draws_by_stratum[stratum_id])
            strata.append(
                SamplingStratumSpec.from_context_ids(
                    stratum_id=stratum_id,
                    context_ids_in_sealed_order=draw_ids,
                    risk_weight=float(risk_weights[stratum_id]),
                )
            )
            if len(draw_ids) != targets[stratum_id]:
                raise ValueError(
                    "runtime IID draws differ from the birth-frozen target"
                )
        randomization_commitment = iid_receipt.content_hash
        design_kind = "iid_fixed_n_stratified"
        randomization_method = GLOCAL_IID_RUNTIME_RANDOMIZATION_METHOD_V2
        sampling_unit = "draw"
    else:
        iid_receipt = None
        for stratum_id in sorted(support):
            raw = tuple(support[stratum_id])
            receipt = SamplingPermutationReceipt.draw(
                stratum_id=stratum_id,
                population=raw,
                birth_binding=birth_binding,
            )
            sealed = receipt.generated_order
            receipts.append(receipt)
            strata.append(
                SamplingStratumSpec.from_context_ids(
                    stratum_id=stratum_id,
                    context_ids_in_sealed_order=sealed,
                    risk_weight=float(risk_weights[stratum_id]),
                )
            )
        randomization_commitment = SamplingDesignSpec.randomization_commitment_for(
            receipts
        )
        design_kind = "sealed_uniform_without_replacement_stratified"
        randomization_method = GLOCAL_SEALED_PERMUTATION_ALGORITHM_V1
        sampling_unit = "context"
    design = SamplingDesignSpec(
        design_id=f"{context_design.design_id}:{policy.policy_id}:terminal",
        context_population_hash=precommitment.estimand.terminal_population_hash,
        randomization_commitment_hash=randomization_commitment,
        strata=tuple(strata),
        permutation_receipts=tuple(receipts),
        iid_sampling_receipt=iid_receipt,
        binary_class_by_stratum=tuple(
            dict(
                dict(context_design.risk_estimand.binary_class_by_stratum_by_query)[
                    context_design.query_id
                ]
            ).items()
        ),
        design_kind=design_kind,
        randomization_method=randomization_method,
        sampling_unit=sampling_unit,
    )
    observation_set = ObservationSetRef(
        observation_set_id=f"{context_design.design_id}:{policy.policy_id}:terminal",
        role=EvidenceRole.TERMINAL_AUDIT,
        sampling_design_hash=design.content_hash,
        context_ids=tuple(
            context_id
            for stratum in design.strata
            for context_id in stratum.context_ids_in_sealed_order[
                : targets[stratum.stratum_id]
            ]
        ),
        stratum_ids=tuple(
            stratum.stratum_id
            for stratum in design.strata
            for _ in stratum.context_ids_in_sealed_order[: targets[stratum.stratum_id]]
        ),
    )
    observation_set.validate_against(design)
    return design, observation_set, targets


def _iid_physical_terminal_panel(
    draw_panel: CrossfitPanel,
    receipt: IIDSamplingReceipt,
    *,
    fold_count: int,
) -> tuple[CrossfitPanel, np.ndarray, dict[str, int]]:
    """Lower unique statistical draws onto deduplicated physical source rows."""

    draw_by_id = {draw.draw_id: draw for draw in receipt.draws}
    source_position: dict[str, int] = {}
    source_ids: list[str] = []
    source_labels: list[int] = []
    source_strata: list[str] = []
    draw_take: list[int] = []
    for draw_id, draw_label in zip(
        draw_panel.context_ids,
        draw_panel.labels,
        strict=True,
    ):
        draw = draw_by_id.get(draw_id)
        if draw is None:
            raise ValueError("terminal draw panel contains an unreceipted IID draw")
        position = source_position.get(draw.source_unit_id)
        if position is None:
            position = len(source_ids)
            source_position[draw.source_unit_id] = position
            source_ids.append(draw.source_unit_id)
            source_labels.append(int(draw_label))
            source_strata.append(draw.stratum_id)
        elif (
            source_labels[position] != int(draw_label)
            or source_strata[position] != draw.stratum_id
        ):
            raise ValueError(
                "repeated IID source unit has inconsistent label or stratum"
            )
        draw_take.append(position)
    if len(source_ids) < int(fold_count):
        raise RuntimeError(
            "the current Torch terminal-pair interface requires at least one "
            "unique physical source unit per declared panel fold"
        )
    source_targets = {
        stratum_id: source_strata.count(stratum_id)
        for stratum_id in sorted(set(source_strata))
    }
    panel = CrossfitPanel(
        context_ids=tuple(source_ids),
        fold_ids=tuple(index % int(fold_count) for index in range(len(source_ids))),
        labels=tuple(source_labels),
        draw_id=receipt.content_hash,
        stage_index=draw_panel.stage_index,
        allocation_metadata={
            **dict(draw_panel.allocation_metadata),
            "requested_per_stratum_target_images": source_targets,
            "requested_cumulative_target_images": len(source_ids),
            "statistical_draw_per_stratum_targets": dict(
                draw_panel.allocation_metadata["requested_per_stratum_target_images"]
            ),
            "statistical_draw_count": len(draw_panel.context_ids),
            "statistical_draw_panel_hash": draw_panel.panel_hash,
            "physical_source_mapping_hash": receipt.content_hash,
            "physical_unique_source_count": len(source_ids),
        },
    )
    panel.validate(fold_count=int(fold_count))
    return panel, np.asarray(draw_take, dtype=np.int64), source_targets


@dataclass(frozen=True, slots=True)
class IRGNativeMaskBridge:
    """Hash bridge between immutable IRG masks and native layer masks."""

    candidate_id: str
    irg_parent_state_hash: str
    route_axis_hash: str
    irg_parent_mask_hash: str
    irg_candidate_mask_hash: str
    native_parent_mask_hash: str
    native_candidate_mask_hash: str
    native_layer_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    lowering_spec_hash: str
    schema_name: str = GLOCAL_IRG_MASK_BRIDGE_SCHEMA

    def __post_init__(self) -> None:
        if (
            not self.candidate_id
            or not self.irg_parent_state_hash
            or not self.route_axis_hash
            or not self.irg_parent_mask_hash
            or not self.irg_candidate_mask_hash
            or not self.schema_name
        ):
            raise ValueError("IRG/native mask bridge has an empty identity field")
        for value, label in (
            (self.native_parent_mask_hash, "native parent mask hash"),
            (self.native_candidate_mask_hash, "native candidate mask hash"),
            (self.lowering_spec_hash, "mask lowering spec hash"),
        ):
            _require_sha256(value, label)
        shapes = tuple(
            (str(layer_id), tuple(int(size) for size in shape))
            for layer_id, shape in self.native_layer_shapes
        )
        if (
            not shapes
            or len({layer_id for layer_id, _ in shapes}) != len(shapes)
            or any(
                not layer_id or not shape or any(size <= 0 for size in shape)
                for layer_id, shape in shapes
            )
        ):
            raise ValueError("IRG/native mask bridge has an invalid layer shape axis")
        object.__setattr__(self, "native_layer_shapes", tuple(sorted(shapes)))

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class GlocalProposalLook:
    look_index: int
    chosen_stratum: str
    cumulative_targets: tuple[tuple[str, int], ...]
    new_context_ids: tuple[str, ...]
    panel_hash: str
    paired_evidence_hash: str
    increment_hash: str
    evidence_snapshot_hash: str
    allocation_look_receipt_hash: str
    lower_bound: float
    upper_bound: float
    outcome: str
    allocation_scores: tuple[tuple[str, float], ...]
    realized_seconds: float
    resource_ledger_hash: str

    @property
    def content_hash(self) -> str:
        allocation_scores = tuple(
            (
                key,
                (
                    value
                    if math.isfinite(float(value))
                    else (
                        "positive_infinity"
                        if float(value) > 0.0
                        else "negative_infinity"
                    )
                ),
            )
            for key, value in self.allocation_scores
        )
        return content_hash(
            {
                "schema_name": "glocal_proposal_look_v1",
                "look_index": self.look_index,
                "chosen_stratum": self.chosen_stratum,
                "cumulative_targets": self.cumulative_targets,
                "new_context_ids": self.new_context_ids,
                "panel_hash": self.panel_hash,
                "paired_evidence_hash": self.paired_evidence_hash,
                "increment_hash": self.increment_hash,
                "evidence_snapshot_hash": self.evidence_snapshot_hash,
                "allocation_look_receipt_hash": (self.allocation_look_receipt_hash),
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
                "outcome": self.outcome,
                "allocation_scores": allocation_scores,
                "realized_seconds": self.realized_seconds,
                "resource_ledger_hash": self.resource_ledger_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class GlocalProposalResult:
    proposal_id: str
    outcome: Literal["accepted", "rejected", "unresolved"]
    reason: str
    committed: bool
    registration_hash: str | None
    candidate_state_hash: str | None
    candidate_calibration_hash: str | None
    decision_receipt: DecisionEvidenceReceipt | None
    looks: tuple[GlocalProposalLook, ...]
    campaign_receipt_hash: str
    schema_name: str = GLOCAL_PROPOSAL_RESULT_SCHEMA

    @property
    def content_hash(self) -> str:
        return content_hash(
            {
                "schema_name": self.schema_name,
                "proposal_id": self.proposal_id,
                "outcome": self.outcome,
                "reason": self.reason,
                "committed": self.committed,
                "registration_hash": self.registration_hash,
                "candidate_state_hash": self.candidate_state_hash,
                "candidate_calibration_hash": self.candidate_calibration_hash,
                "decision_receipt_hash": (
                    None
                    if self.decision_receipt is None
                    else self.decision_receipt.content_hash
                ),
                "look_hashes": tuple(look.content_hash for look in self.looks),
                "campaign_receipt_hash": self.campaign_receipt_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class CampaignCalibrationTransitionDecision:
    """Exact campaign verdict for a same-mask calibration state transition."""

    request_id: str
    request_cache_key_hash: str
    irg_parent_state_hash: str
    irg_mask_hash: str
    native_mask_hash: str
    candidate_native_mask_hash: str
    campaign_parent_state_hash: str
    campaign_candidate_state_hash: str | None
    parent_calibration_hash: str
    candidate_calibration_hash: str | None
    calibration_fit_observation_set_hash: str
    calibration_prefix_targets: tuple[tuple[str, int], ...]
    requested_context_increment: int
    candidate_definition_hash: str
    policy_hash: str
    proposal_id: str
    proposal_result_hash: str
    registration_hash: str | None
    decision_evidence_receipt_hash: str | None
    campaign_receipt_hash: str
    resource_ledger_before_hash: str
    resource_ledger_after_hash: str
    local_upper_boundary: float
    distortion_debit: float
    outcome: Literal["accepted", "rejected", "unresolved"]
    accepted: bool
    schema_name: str = GLOCAL_CALIBRATION_TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        if not self.request_id or not self.proposal_id or not self.schema_name:
            raise ValueError(
                "calibration transition decision requires stable identities"
            )
        _require_prefixed_sha256(
            self.request_id,
            "irg-request-",
            "calibration transition request ID",
        )
        _require_prefixed_sha256(
            self.irg_parent_state_hash,
            "irg-state-",
            "IRG parent state hash",
        )
        _require_prefixed_sha256(
            self.irg_mask_hash,
            "irg-mask-",
            "IRG mask hash",
        )
        for value, label in (
            (self.request_cache_key_hash, "request cache key hash"),
            (self.native_mask_hash, "native mask hash"),
            (self.candidate_native_mask_hash, "candidate native mask hash"),
            (self.campaign_parent_state_hash, "campaign parent state hash"),
            (self.parent_calibration_hash, "parent calibration hash"),
            (
                self.calibration_fit_observation_set_hash,
                "calibration fit observation-set hash",
            ),
            (self.candidate_definition_hash, "candidate definition hash"),
            (self.policy_hash, "commit policy hash"),
            (self.proposal_result_hash, "proposal result hash"),
            (self.campaign_receipt_hash, "campaign receipt hash"),
            (self.resource_ledger_before_hash, "resource ledger before hash"),
            (self.resource_ledger_after_hash, "resource ledger after hash"),
        ):
            _require_sha256(value, label)
        for value, label in (
            (self.campaign_candidate_state_hash, "campaign candidate state hash"),
            (self.candidate_calibration_hash, "candidate calibration hash"),
            (self.registration_hash, "proposal registration hash"),
            (
                self.decision_evidence_receipt_hash,
                "decision evidence receipt hash",
            ),
        ):
            if value is not None:
                _require_sha256(value, label)
        boundary = float(self.local_upper_boundary)
        debit = float(self.distortion_debit)
        prefix_targets = tuple(
            sorted(
                (str(stratum_id), int(target))
                for stratum_id, target in self.calibration_prefix_targets
            )
        )
        increment = int(self.requested_context_increment)
        if not math.isfinite(boundary) or not math.isfinite(debit) or debit < 0.0:
            raise ValueError(
                "calibration transition boundary must be finite and debit must "
                "be finite and nonnegative"
            )
        if (
            not prefix_targets
            or len({key for key, _ in prefix_targets}) != len(prefix_targets)
            or any(not key or target <= 0 for key, target in prefix_targets)
            or increment < 0
        ):
            raise ValueError(
                "calibration transition requires positive frozen prefix targets "
                "and a nonnegative requested increment"
            )
        if self.accepted != (self.outcome == "accepted"):
            raise ValueError(
                "calibration transition acceptance flag and outcome disagree"
            )
        if self.candidate_native_mask_hash != self.native_mask_hash:
            raise ValueError(
                "calibration-only transition must retain the exact native mask"
            )
        if self.accepted and (
            self.campaign_candidate_state_hash is None
            or self.candidate_calibration_hash is None
            or self.registration_hash is None
            or self.decision_evidence_receipt_hash is None
        ):
            raise ValueError(
                "accepted calibration transition lacks its exact campaign lineage"
            )
        if not self.accepted and debit != 0.0:
            raise ValueError(
                "nonaccepted calibration transition cannot debit distortion"
            )
        object.__setattr__(self, "local_upper_boundary", boundary)
        object.__setattr__(self, "distortion_debit", debit)
        object.__setattr__(self, "calibration_prefix_targets", prefix_targets)
        object.__setattr__(self, "requested_context_increment", increment)

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class CampaignCalibrationLogitAcquisitionReceipt:
    """Physical calibration-role prefix growth that does not change the fit."""

    request_id: str
    irg_state_hash: str
    irg_mask_hash: str
    native_mask_hash: str
    calibration_fit_state_hash: str
    calibration_sampling_design_hash: str
    before_targets: tuple[tuple[str, int], ...]
    after_targets: tuple[tuple[str, int], ...]
    observation_set_hash: str
    panel_hash: str
    row_evidence_hash: str
    new_context_ids: tuple[str, ...]
    backend_runtime_receipt_hash: str
    resource_ledger_before_hash: str
    resource_ledger_after_hash: str
    schema_name: str = GLOCAL_CALIBRATION_LOGIT_ACQUISITION_SCHEMA

    def __post_init__(self) -> None:
        _require_prefixed_sha256(
            self.request_id,
            "irg-request-",
            "calibration-logit request ID",
        )
        _require_prefixed_sha256(
            self.irg_state_hash,
            "irg-state-",
            "calibration-logit IRG state hash",
        )
        _require_prefixed_sha256(
            self.irg_mask_hash,
            "irg-mask-",
            "calibration-logit IRG mask hash",
        )
        for value, label in (
            (self.native_mask_hash, "native mask hash"),
            (self.calibration_fit_state_hash, "calibration fit-state hash"),
            (
                self.calibration_sampling_design_hash,
                "calibration sampling-design hash",
            ),
            (self.observation_set_hash, "calibration observation-set hash"),
            (self.panel_hash, "calibration panel hash"),
            (self.row_evidence_hash, "calibration row-evidence hash"),
            (self.backend_runtime_receipt_hash, "backend runtime receipt hash"),
            (self.resource_ledger_before_hash, "resource ledger before hash"),
            (self.resource_ledger_after_hash, "resource ledger after hash"),
        ):
            _require_sha256(value, label)
        before = tuple(
            sorted((str(key), int(value)) for key, value in self.before_targets)
        )
        after = tuple(
            sorted((str(key), int(value)) for key, value in self.after_targets)
        )
        if (
            not before
            or {key for key, _ in before} != {key for key, _ in after}
            or any(value <= 0 for _, value in before + after)
            or any(dict(after)[key] < value for key, value in before)
        ):
            raise ValueError(
                "calibration-logit receipt requires nested positive prefix targets"
            )
        new_contexts = tuple(str(value) for value in self.new_context_ids)
        if (
            not new_contexts
            or len(set(new_contexts)) != len(new_contexts)
            or sum(value for _, value in after) - sum(value for _, value in before)
            != len(new_contexts)
        ):
            raise ValueError(
                "calibration-logit receipt new-context axis differs from prefix growth"
            )
        object.__setattr__(self, "before_targets", before)
        object.__setattr__(self, "after_targets", after)
        object.__setattr__(self, "new_context_ids", new_contexts)

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class GlocalTerminalAuditResult:
    certificate: TerminalParityCertificate
    evidence_receipt: TerminalAuditEvidenceReceipt
    policy_hash: str
    sampling_design_hash: str
    terminal_observation_set_hash: str
    paired_evidence_hash: str
    physical_source_evidence_hash: str
    runtime_call_receipt_hash: str
    stratum_results: tuple[tuple[str, Mapping[str, int | float]], ...]
    resource_ledger_hash: str
    campaign_receipt_hash: str
    schema_name: str = "glocal_terminal_audit_result_v2"

    def __post_init__(self) -> None:
        if not isinstance(self.certificate, TerminalParityCertificate):
            raise TypeError("terminal audit result requires a parity certificate")
        if not isinstance(
            self.evidence_receipt,
            TerminalAuditEvidenceReceipt,
        ):
            raise TypeError(
                "terminal audit result requires typed physical audit evidence"
            )
        if (
            self.evidence_receipt.certificate.content_hash
            != self.certificate.content_hash
        ):
            raise ValueError(
                "terminal audit result and evidence receipt certificates differ"
            )
        for value, label in (
            (self.policy_hash, "terminal policy hash"),
            (self.sampling_design_hash, "terminal sampling design hash"),
            (
                self.terminal_observation_set_hash,
                "terminal observation set hash",
            ),
            (self.paired_evidence_hash, "terminal paired evidence hash"),
            (
                self.physical_source_evidence_hash,
                "terminal physical source evidence hash",
            ),
            (
                self.runtime_call_receipt_hash,
                "terminal runtime call receipt hash",
            ),
            (self.resource_ledger_hash, "terminal resource ledger hash"),
            (self.campaign_receipt_hash, "terminal campaign receipt hash"),
        ):
            _require_sha256(value, label)
        for observed, expected, label in (
            (
                self.policy_hash,
                self.evidence_receipt.policy_hash,
                "policy",
            ),
            (
                self.sampling_design_hash,
                self.evidence_receipt.sampling_design_hash,
                "sampling design",
            ),
            (
                self.paired_evidence_hash,
                self.evidence_receipt.paired_evidence_hash,
                "paired evidence",
            ),
            (
                self.resource_ledger_hash,
                self.evidence_receipt.resource_ledger_hash,
                "resource ledger",
            ),
        ):
            if observed != expected:
                raise ValueError(
                    f"terminal audit result and physical receipt {label} differ"
                )

    @property
    def content_hash(self) -> str:
        return content_hash(self)


class ResourceBudgetUnavailable(RuntimeError):
    """The next atomic packet cannot fit a frozen campaign resource cap."""


def _backend_resource_snapshot(backend: Any) -> dict[str, float]:
    payload = getattr(backend, "resource_usage", {})
    if not isinstance(payload, Mapping):
        return {}
    response_cache = dict(payload.get("response_cache", {}) or {})
    glocal = dict(payload.get("glocal", {}) or {})
    prefetch = dict(payload.get("prefetch", {}) or {})
    physical_observed = bool(payload.get("physical_forward_calls_observed", False))
    backward_observed = bool(payload.get("backward_calls_observed", False))
    return {
        "forward_passes": float(
            payload.get("physical_forward_calls", 0)
            if physical_observed
            else payload.get(
                "high_level_forward_calls", payload.get("forward_calls", 0)
            )
        ),
        "backward_passes": float(
            payload.get("backward_calls", 0) if backward_observed else 0
        ),
        "calibration_fits": float(payload.get("calibration_fit_calls", 0)),
        "host_to_device_bytes": float(payload.get("host_to_device_bytes", 0)),
        "device_to_host_bytes": float(payload.get("device_to_host_bytes", 0)),
        "wall_seconds": float(glocal.get("wall_seconds", 0.0)),
        "device_seconds": float(glocal.get("cuda_device_seconds", 0.0)),
        "peak_memory_bytes": float(payload.get("absolute_peak_device_memory_bytes", 0)),
        "prefetch_contexts": float(prefetch.get("unique_contexts_loaded", 0)),
        "cache_hits": float(response_cache.get("hits", 0)),
    }


def _resource_counter_delta(
    before: Mapping[str, float], after: Mapping[str, float]
) -> dict[str, int | float]:
    integer_additive = {
        "forward_passes",
        "backward_passes",
        "calibration_fits",
        "host_to_device_bytes",
        "device_to_host_bytes",
        "prefetch_contexts",
        "cache_hits",
    }
    delta: dict[str, int | float] = {
        key: max(
            0,
            int(round(float(after.get(key, 0)) - float(before.get(key, 0)))),
        )
        for key in integer_additive
    }
    for key in {"wall_seconds", "device_seconds"}:
        delta[key] = max(
            0.0,
            float(after.get(key, 0.0)) - float(before.get(key, 0.0)),
        )
    delta["peak_memory_bytes"] = max(
        0,
        int(round(float(after.get("peak_memory_bytes", 0)))),
    )
    return delta


class GlocalBrierCampaignRuntime:
    """One serial shadow campaign with a single global safety ledger."""

    def __init__(
        self,
        *,
        backend: TorchSampledPathEvaluationBackend,
        context_design: FrozenBinaryContextDesign,
        campaign: CampaignEvidenceState,
        initial_masks: Mapping[str, np.ndarray],
        initial_calibration: FrozenCalibrationBinding,
        current_masks: Mapping[str, np.ndarray],
        current_calibration: FrozenCalibrationBinding,
        policy: GlocalNeymanPolicy,
        calibration_envelope: GlocalPacketResourceEnvelope,
        decision_envelope: GlocalPacketResourceEnvelope,
        terminal_envelope: GlocalPacketResourceEnvelope,
        terminal_policy: GlocalTerminalAuditPolicy | None,
        terminal_precommitment: TerminalAuditPrecommitment | None,
        terminal_design: SamplingDesignSpec | None,
        terminal_observation_set: ObservationSetRef | None,
        terminal_targets: Mapping[str, int] | None,
        calibration_targets: Mapping[str, int],
        initial_calibration_logit_rows: GlocalCalibrationLogitRows,
    ) -> None:
        self.backend = backend
        self.context_design = context_design
        self.campaign = campaign
        self.initial_masks = {
            str(key): np.asarray(value, dtype=bool).copy()
            for key, value in initial_masks.items()
        }
        self.initial_calibration = initial_calibration
        self.current_masks = {
            str(key): np.asarray(value, dtype=bool).copy()
            for key, value in current_masks.items()
        }
        self.current_calibration = current_calibration
        self._available_calibration_targets = {
            str(key): int(value) for key, value in calibration_targets.items()
        }
        if self._available_calibration_targets != dict(
            self.context_design.calibration_initial_targets
        ):
            raise ValueError(
                "runtime calibration prefix differs from the campaign-birth design"
            )
        self.policy = policy
        self.calibration_envelope = calibration_envelope
        self.decision_envelope = decision_envelope
        self.terminal_envelope = terminal_envelope
        self._terminal_policy = terminal_policy
        self._terminal_precommitment = terminal_precommitment
        self._terminal_design = terminal_design
        self._terminal_observation_set = terminal_observation_set
        self._terminal_targets = (
            None if terminal_targets is None else dict(terminal_targets)
        )
        self._terminal_iid_sampling_receipt: IIDSamplingReceipt | None = None
        self._terminal_locked = False
        self._terminal_result: GlocalTerminalAuditResult | None = None
        self._terminal_paired_evidence: Any | None = None
        self._proposal_results: list[GlocalProposalResult] = []
        self._proposal_calibrations: dict[
            str, tuple[FrozenCalibrationBinding, FrozenCalibrationBinding]
        ] = {}
        self._proposal_paired_evidence: dict[str, list[Any]] = {}
        self._proposal_evidence_snapshots: dict[str, list[Any]] = {}
        if (
            initial_calibration_logit_rows.mask_hash
            != sampled_path_mask_hash(initial_masks)
            or initial_calibration_logit_rows.panel.context_ids
            != initial_calibration.observation_set.context_ids
        ):
            raise ValueError(
                "initial retained calibration logits differ from the fitted state"
            )
        self._calibration_logit_rows_by_hash: dict[str, GlocalCalibrationLogitRows] = {
            initial_calibration_logit_rows.evidence_hash: (
                initial_calibration_logit_rows
            )
        }
        self._live_capability_nonce = secrets.token_hex(32)
        self._calibration_growth_nonce = secrets.token_hex(32)
        self._deployment_certificate: GlocalDeploymentMaskCertificate | None = None
        self._deployment_authority_handle: GlocalDeploymentAuthorityHandle | None = None
        self._deployment_authority_request_hash: str | None = None

    @staticmethod
    def _require_envelope_for_caps(
        ledger: ResourceLedger,
        envelope: GlocalPacketResourceEnvelope,
    ) -> None:
        values = ResourceUsage(
            forward_passes=envelope.forward_passes,
            backward_passes=envelope.backward_passes,
            calibration_fits=envelope.calibration_fits,
            host_to_device_bytes=envelope.host_to_device_bytes,
            device_to_host_bytes=envelope.device_to_host_bytes,
            wall_seconds=envelope.wall_seconds,
            device_seconds=envelope.device_seconds,
            peak_memory_bytes=envelope.peak_memory_bytes,
        )
        for name in (
            "host_to_device_bytes",
            "device_to_host_bytes",
            "wall_seconds",
            "device_seconds",
            "peak_memory_bytes",
        ):
            if name in ledger.caps and float(getattr(values, name)) <= 0.0:
                raise ValueError(
                    f"claim-mode {name} cap requires a frozen conservative "
                    "packet envelope"
                )

    @classmethod
    def create(
        cls,
        *,
        backend: TorchSampledPathEvaluationBackend,
        context_design: FrozenBinaryContextDesign,
        initial_masks: Mapping[str, np.ndarray],
        campaign_id: str,
        distortion_budget: float,
        policy: GlocalNeymanPolicy = GlocalNeymanPolicy(),
        resource_caps: Mapping[str, int | float] | None = None,
        calibration_envelope: GlocalPacketResourceEnvelope = (
            GlocalPacketResourceEnvelope(
                forward_passes=1,
                calibration_fits=1,
            )
        ),
        decision_envelope: GlocalPacketResourceEnvelope = (
            GlocalPacketResourceEnvelope(forward_passes=2)
        ),
        terminal_envelope: GlocalPacketResourceEnvelope | None = None,
        terminal_audit_policy: GlocalTerminalAuditPolicy | None = None,
        total_epsilon: float = 0.05,
        sequential_alpha: float = 0.04,
        terminal_alpha: float = 0.01,
        maximum_proposals: int = 512,
        maximum_calibration_iterations: int = 256,
        calibration_gradient_tolerance: float = 1e-9,
        calibration_identity_ridge: float = 1e-8,
    ) -> "GlocalBrierCampaignRuntime":
        algorithm = glocal_brier_calibration_algorithm_spec(
            maximum_iterations=maximum_calibration_iterations,
            gradient_tolerance=calibration_gradient_tolerance,
            identity_ridge=calibration_identity_ridge,
        )
        temporary_ledger = ResourceLedger(caps=resource_caps)
        cls._require_envelope_for_caps(temporary_ledger, calibration_envelope)
        panel = context_design.calibration_panel()
        mask_hash = sampled_path_mask_hash(initial_masks)
        execution_state = _query_execution_state_hash(
            query_id=context_design.query_id,
            mask_hash=mask_hash,
        )
        preview, _, _ = temporary_ledger.preview_context_state_evaluations(
            context_ids=panel.context_ids,
            state_hashes=(execution_state,),
            forward_passes=calibration_envelope.forward_passes,
            backward_passes=calibration_envelope.backward_passes,
            calibration_fits=calibration_envelope.calibration_fits,
            host_to_device_bytes=calibration_envelope.host_to_device_bytes,
            device_to_host_bytes=calibration_envelope.device_to_host_bytes,
            wall_seconds=calibration_envelope.wall_seconds,
            device_seconds=calibration_envelope.device_seconds,
            peak_memory_bytes=calibration_envelope.peak_memory_bytes,
        )
        if not temporary_ledger.can_consume(preview):
            raise ResourceBudgetUnavailable(
                "initial calibration packet exceeds the frozen campaign caps"
            )
        request = cls._calibration_request(
            context_design=context_design,
            panel=panel,
            proposal_index=0,
            maximum_new_contexts=len(panel.context_ids),
        )
        before = _backend_resource_snapshot(backend)
        started = time.perf_counter()
        initial_rows = backend.evaluate_glocal_calibration_logits(
            initial_masks,
            request,
            panel=panel,
        )
        logit_receipt = backend.last_glocal_runtime_receipt
        if (
            logit_receipt is None
            or logit_receipt.operation != "evaluate_glocal_calibration_logits"
            or logit_receipt.panel_hash != panel.panel_hash
            or not logit_receipt.succeeded
        ):
            raise RuntimeError(
                "initial calibration logits lack their successful backend receipt"
            )
        native = backend.fit_glocal_brier_calibration(
            initial_masks,
            request,
            role=EvidenceRole.CALIBRATION_FIT.value,
            panel=panel,
            maximum_iterations=maximum_calibration_iterations,
            gradient_tolerance=calibration_gradient_tolerance,
            identity_ridge=calibration_identity_ridge,
        )
        fit_receipt = backend.last_glocal_runtime_receipt
        if (
            fit_receipt is None
            or fit_receipt.operation != "fit_glocal_brier_calibration"
            or fit_receipt.panel_hash != panel.panel_hash
            or not fit_receipt.succeeded
        ):
            raise RuntimeError(
                "initial calibration fit lacks its successful backend receipt"
            )
        elapsed = time.perf_counter() - started
        after = _backend_resource_snapshot(backend)
        delta = _resource_counter_delta(before, after)
        binding = calibration_binding_from_native(
            native,
            fit_id=f"{campaign_id}:initial-calibration",
            observation_set=context_design.calibration_observation_set,
            algorithm_spec=algorithm,
            fit_input_evidence_hash=initial_rows.evidence_hash,
        )
        initial_state_hash = locked_brier_state_hash(
            query_id=context_design.query_id,
            mask_hash=mask_hash,
            calibration_fit_hash=binding.fit_state.content_hash,
            risk_estimand_hash=context_design.risk_estimand.content_hash,
        )
        resolved_terminal_envelope = (
            decision_envelope if terminal_envelope is None else terminal_envelope
        )
        cls._require_envelope_for_caps(
            temporary_ledger,
            resolved_terminal_envelope,
        )
        terminal_precommitment = None
        frozen_terminal_design = None
        frozen_terminal_observation_set = None
        frozen_terminal_targets = None
        if terminal_audit_policy is not None:
            terminal_precommitment = _terminal_precommitment(
                context_design=context_design,
                policy=terminal_audit_policy,
                initial_state_hash=initial_state_hash,
                initial_calibration_hash=binding.fit_state.content_hash,
                backend=backend,
                terminal_envelope=resolved_terminal_envelope,
                terminal_alpha=float(terminal_alpha),
            )
            frozen_terminal_targets = dict(
                terminal_precommitment.target_sample_sizes_by_stratum
            )
            if terminal_audit_policy.sampling_design != "iid_fixed_n":
                (
                    frozen_terminal_design,
                    frozen_terminal_observation_set,
                    realized_targets,
                ) = _freeze_terminal_design(
                    context_design,
                    terminal_audit_policy,
                    terminal_precommitment,
                )
                if realized_targets != frozen_terminal_targets:
                    raise RuntimeError(
                        "terminal design realization changed its birth targets"
                    )
        campaign = CampaignEvidenceState(
            campaign_id=str(campaign_id),
            risk_estimand=context_design.risk_estimand,
            initial_state_hash=initial_state_hash,
            initial_calibration=binding.fit_state,
            distortion_budget=float(distortion_budget),
            total_epsilon=float(total_epsilon),
            sequential_alpha=float(sequential_alpha),
            terminal_alpha=float(terminal_alpha),
            maximum_proposals=int(maximum_proposals),
            resource_caps=resource_caps,
            observation_sets=(context_design.calibration_observation_set,),
            calibration_sampling_design=(context_design.calibration_sampling_design),
            terminal_precommitment=terminal_precommitment,
        )
        campaign.resource_ledger.record_context_state_evaluations(
            context_ids=panel.context_ids,
            state_hashes=(execution_state,),
            forward_passes=int(delta["forward_passes"]),
            backward_passes=int(delta["backward_passes"]),
            calibration_fits=max(1, int(delta["calibration_fits"])),
            host_to_device_bytes=int(delta["host_to_device_bytes"]),
            device_to_host_bytes=int(delta["device_to_host_bytes"]),
            wall_seconds=(
                float(delta["wall_seconds"])
                if float(delta["wall_seconds"]) > 0.0
                else elapsed
            ),
            device_seconds=float(delta["device_seconds"]),
            peak_memory_bytes=int(delta["peak_memory_bytes"]),
            cache_hits=int(delta["cache_hits"]),
        )
        cls._require_envelope_for_caps(campaign.resource_ledger, decision_envelope)
        cls._require_envelope_for_caps(
            campaign.resource_ledger,
            resolved_terminal_envelope,
        )
        return cls(
            backend=backend,
            context_design=context_design,
            campaign=campaign,
            initial_masks=initial_masks,
            initial_calibration=binding,
            current_masks=initial_masks,
            current_calibration=binding,
            policy=policy,
            calibration_envelope=calibration_envelope,
            decision_envelope=decision_envelope,
            terminal_envelope=resolved_terminal_envelope,
            terminal_policy=terminal_audit_policy,
            terminal_precommitment=terminal_precommitment,
            terminal_design=frozen_terminal_design,
            terminal_observation_set=frozen_terminal_observation_set,
            terminal_targets=frozen_terminal_targets,
            calibration_targets=context_design.calibration_initial_targets,
            initial_calibration_logit_rows=initial_rows,
        )

    @staticmethod
    def _calibration_request(
        *,
        context_design: FrozenBinaryContextDesign,
        panel: CrossfitPanel,
        proposal_index: int,
        maximum_new_contexts: int,
        prior_panel: CrossfitPanel | None = None,
        stage_index: int = 0,
    ) -> GlocalSampledComparisonRequest:
        labels = np.asarray(panel.labels, dtype=np.int64)
        counts = {
            "positive": int(
                np.count_nonzero(labels == context_design.positive_class_id)
            )
        }
        for class_id in np.unique(labels):
            if int(class_id) != context_design.positive_class_id:
                counts[f"neg:{int(class_id)}"] = int(
                    np.count_nonzero(labels == int(class_id))
                )
        return GlocalSampledComparisonRequest(
            query_id=context_design.query_id,
            layer_id="__campaign__",
            partition_id="calibration",
            previous_density=1.0,
            proposed_density=1.0,
            stage_index=int(stage_index),
            stage_per_class=max(1, len(panel.context_ids) // 2),
            proposal_index=int(proposal_index),
            random_seed=0,
            fold_count=context_design.fold_count,
            prior_panel=prior_panel,
            cumulative_target_images=len(panel.context_ids),
            per_stratum_target_images=counts,
            maximum_new_contexts=int(maximum_new_contexts),
        )

    def _preflight(
        self,
        *,
        context_ids: tuple[str, ...],
        execution_state_hashes: tuple[str, ...],
        envelope: GlocalPacketResourceEnvelope,
    ) -> ResourceUsage:
        self._require_envelope_for_caps(self.campaign.resource_ledger, envelope)
        preview, _, _ = self.campaign.resource_ledger.preview_context_state_evaluations(
            context_ids=context_ids,
            state_hashes=execution_state_hashes,
            forward_passes=envelope.forward_passes,
            backward_passes=envelope.backward_passes,
            calibration_fits=envelope.calibration_fits,
            host_to_device_bytes=envelope.host_to_device_bytes,
            device_to_host_bytes=envelope.device_to_host_bytes,
            wall_seconds=envelope.wall_seconds,
            device_seconds=envelope.device_seconds,
            peak_memory_bytes=envelope.peak_memory_bytes,
        )
        if not self.campaign.resource_ledger.can_consume(preview):
            raise ResourceBudgetUnavailable(
                "the next minimum packet does not fit the campaign resource caps"
            )
        return preview

    def _record_runtime_call(
        self,
        *,
        context_ids: tuple[str, ...],
        execution_state_hashes: tuple[str, ...],
        before: Mapping[str, float],
        elapsed: float,
        calibration_fit: bool,
        envelope: GlocalPacketResourceEnvelope,
    ) -> None:
        delta = _resource_counter_delta(
            before,
            _backend_resource_snapshot(self.backend),
        )
        actual = {
            "forward_passes": int(delta["forward_passes"]),
            "backward_passes": int(delta["backward_passes"]),
            "calibration_fits": (
                max(1, int(delta["calibration_fits"])) if calibration_fit else 0
            ),
            "host_to_device_bytes": int(delta["host_to_device_bytes"]),
            "device_to_host_bytes": int(delta["device_to_host_bytes"]),
            "wall_seconds": (
                float(delta["wall_seconds"])
                if float(delta["wall_seconds"]) > 0.0
                else float(elapsed)
            ),
            "device_seconds": float(delta["device_seconds"]),
            "peak_memory_bytes": int(delta["peak_memory_bytes"]),
        }
        underestimated = tuple(
            name
            for name, value in actual.items()
            if (
                name in self.campaign.resource_ledger.caps
                or float(getattr(envelope, name)) > 0.0
            )
            and float(value) > float(getattr(envelope, name)) + 1e-12
        )
        if underestimated:
            raise RuntimeError(
                "realized physical packet exceeded its frozen campaign "
                "envelope: " + ",".join(underestimated)
            )
        self.campaign.resource_ledger.record_context_state_evaluations(
            context_ids=context_ids,
            state_hashes=execution_state_hashes,
            forward_passes=actual["forward_passes"],
            backward_passes=actual["backward_passes"],
            calibration_fits=actual["calibration_fits"],
            host_to_device_bytes=actual["host_to_device_bytes"],
            device_to_host_bytes=actual["device_to_host_bytes"],
            wall_seconds=actual["wall_seconds"],
            device_seconds=actual["device_seconds"],
            peak_memory_bytes=actual["peak_memory_bytes"],
            cache_hits=int(delta["cache_hits"]),
        )

    def lock_terminal_audit(
        self,
        policy: GlocalTerminalAuditPolicy | None = None,
    ) -> ValidationLock:
        """Bind the final state to the campaign-birth terminal commitment."""

        if self._terminal_locked:
            raise RuntimeError("glocal terminal audit is already locked")
        frozen_policy = self._terminal_policy
        precommitment = self._terminal_precommitment
        targets = self._terminal_targets
        if frozen_policy is None or precommitment is None or targets is None:
            raise RuntimeError(
                "sequential-only runtime has no campaign-birth terminal policy"
            )
        if policy is not None and policy.content_hash != frozen_policy.content_hash:
            raise ValueError(
                "terminal policy differs from the campaign-birth precommitment"
            )
        provider = getattr(self.backend, "panel_source", None)
        if provider is None:
            raise RuntimeError(
                "terminal execution requires a physical panel-source identity"
            )
        if (
            iid_physical_provider_identity_hash(provider)
            != precommitment.provider_identity_hash
            or iid_physical_executor_identity_hash(self.backend)
            != precommitment.executor_identity_hash
            or content_hash(self.terminal_envelope)
            != precommitment.resource_envelope_hash
        ):
            raise ValueError(
                "terminal physical execution identity changed after campaign birth"
            )
        design = self._terminal_design
        observation_set = self._terminal_observation_set
        if frozen_policy.sampling_design == "iid_fixed_n":
            receipt = IIDSamplingReceipt.draw_uniform(
                receipt_id=(
                    f"{self.campaign.campaign_id}:"
                    f"{frozen_policy.policy_id}:terminal-iid"
                ),
                context_population_hash=(
                    precommitment.estimand.terminal_population_hash
                ),
                support_context_ids_by_stratum=dict(
                    precommitment.support_context_ids_by_stratum
                ),
                target_sample_sizes_by_stratum=targets,
                provider_identity_hash=precommitment.provider_identity_hash,
                executor_identity_hash=precommitment.executor_identity_hash,
                birth_binding={
                    "terminal_precommitment_hash": precommitment.content_hash,
                    "final_state_hash": self.campaign.current_state_hash,
                    "final_calibration_hash": (self.campaign.current_calibration_hash),
                    "policy_hash": frozen_policy.content_hash,
                },
            )
            design, observation_set, realized_targets = _freeze_terminal_design(
                self.context_design,
                frozen_policy,
                precommitment,
                iid_sampling_receipt=receipt,
            )
            if realized_targets != targets:
                raise RuntimeError(
                    "runtime IID realization changed its frozen target sizes"
                )
            self._terminal_iid_sampling_receipt = receipt
        if design is None or observation_set is None:
            raise RuntimeError("terminal sampling design was not realized")
        registration_count = len(self.campaign.receipt()["proposal_registrations"])
        lock = ValidationLock(
            lock_id=f"{self.campaign.campaign_id}:{frozen_policy.policy_id}",
            final_state_hash=self.campaign.current_state_hash,
            final_calibration_hash=self.campaign.current_calibration_hash,
            prediction_rule_hash=(
                self.context_design.risk_estimand.prediction_rule.content_hash
            ),
            terminal_precommitment_hash=precommitment.content_hash,
            terminal_population_hash=(precommitment.estimand.terminal_population_hash),
            sampling_design_hash=design.content_hash,
            audit_observation_set=observation_set,
            terminal_alpha=self.campaign.terminal_alpha,
            parity_tolerance=precommitment.estimand.parity_tolerance,
            locked_after_proposal_index=registration_count,
        )
        self.campaign.register_terminal_lock(lock)
        self._terminal_design = design
        self._terminal_observation_set = observation_set
        self._terminal_locked = True
        return lock

    @property
    def terminal_result(self) -> GlocalTerminalAuditResult | None:
        return self._terminal_result

    @property
    def terminal_iid_sampling_receipt(self) -> IIDSamplingReceipt | None:
        return self._terminal_iid_sampling_receipt

    @property
    def deployment_certificate(self) -> GlocalDeploymentMaskCertificate | None:
        return self._deployment_certificate

    @staticmethod
    def _clopper_pearson_upper(
        disagreements: int,
        count: int,
        alpha: float,
    ) -> float:
        errors = int(disagreements)
        size = int(count)
        stake = float(alpha)
        if size <= 0 or errors < 0 or errors > size or not 0.0 < stake < 1.0:
            raise ValueError("terminal Clopper-Pearson inputs are invalid")
        if errors == size:
            return 1.0
        from scipy.stats import beta

        return float(beta.ppf(1.0 - stake, errors + 1, size - errors))

    @staticmethod
    def _terminal_bounds(
        *,
        design: SamplingDesignSpec,
        targets: Mapping[str, int],
        disagreement_by_context: Mapping[str, bool],
        policy: GlocalTerminalAuditPolicy,
        alpha: float,
    ) -> tuple[
        float,
        int,
        tuple[tuple[str, Mapping[str, int | float]], ...],
        str,
    ]:
        weights = {item.stratum_id: float(item.risk_weight) for item in design.strata}
        results: dict[str, dict[str, int | float]] = {}
        total_disagreements = 0
        if policy.sampling_design == "finite_population_wor":
            process = StratifiedPairedEProcess(
                sampling_design=design,
                alpha=float(alpha),
                stratum_alpha={
                    stratum_id: float(alpha) * weight
                    for stratum_id, weight in weights.items()
                },
                lower_bound=0.0,
                upper_bound=1.0,
            )
            for stratum in design.strata:
                contexts = stratum.context_ids_in_sealed_order[
                    : int(targets[stratum.stratum_id])
                ]
                for context_id in contexts:
                    process.record(
                        stratum.stratum_id,
                        float(bool(disagreement_by_context[context_id])),
                        context_id=context_id,
                    )
            snapshot = process.interval()
            upper = float(snapshot.upper)
            snapshots = {item.stratum_id: item for item in snapshot.strata}
            for stratum in design.strata:
                contexts = stratum.context_ids_in_sealed_order[
                    : int(targets[stratum.stratum_id])
                ]
                errors = sum(
                    bool(disagreement_by_context[context_id]) for context_id in contexts
                )
                total_disagreements += int(errors)
                bound = snapshots[stratum.stratum_id]
                results[stratum.stratum_id] = {
                    "population_size": stratum.population_size,
                    "sample_size": len(contexts),
                    "disagreements": int(errors),
                    "rate": float(errors / len(contexts)),
                    "upper": float(bound.upper),
                    "weight": weights[stratum.stratum_id],
                    "alpha": float(bound.alpha),
                }
            method = "stratified_bounded_wor_betting_cs"
        else:
            upper = 0.0
            for stratum in design.strata:
                contexts = stratum.context_ids_in_sealed_order[
                    : int(targets[stratum.stratum_id])
                ]
                errors = sum(
                    bool(disagreement_by_context[context_id]) for context_id in contexts
                )
                total_disagreements += int(errors)
                rate = float(errors / len(contexts))
                stratum_alpha = float(alpha) * weights[stratum.stratum_id]
                stratum_upper = (
                    rate
                    if policy.sampling_design == "finite_census"
                    else GlocalBrierCampaignRuntime._clopper_pearson_upper(
                        int(errors),
                        len(contexts),
                        stratum_alpha,
                    )
                )
                upper += weights[stratum.stratum_id] * stratum_upper
                results[stratum.stratum_id] = {
                    "population_size": stratum.population_size,
                    "sample_size": len(contexts),
                    "disagreements": int(errors),
                    "rate": rate,
                    "upper": float(stratum_upper),
                    "weight": weights[stratum.stratum_id],
                    "alpha": stratum_alpha,
                }
            method = (
                "stratified_finite_population_census_exact"
                if policy.sampling_design == "finite_census"
                else "stratified_one_sided_clopper_pearson"
            )
        return (
            float(np.clip(upper, 0.0, 1.0)),
            total_disagreements,
            tuple((key, results[key]) for key in sorted(results)),
            method,
        )

    def execute_terminal_audit(self) -> GlocalTerminalAuditResult:
        """Consume the locked terminal rows once using frozen state calibrations."""

        if self._terminal_result is not None:
            raise RuntimeError("glocal terminal audit may execute only once")
        policy = self._terminal_policy
        precommitment = self._terminal_precommitment
        design = self._terminal_design
        targets = self._terminal_targets
        lock = self.campaign.terminal_lock
        if (
            policy is None
            or precommitment is None
            or design is None
            or targets is None
            or lock is None
        ):
            raise RuntimeError("glocal terminal audit must be locked before execution")
        draw_panel = self.context_design.terminal_panel(design, targets=targets)
        if policy.sampling_design == "iid_fixed_n":
            receipt = self._terminal_iid_sampling_receipt
            if receipt is None:
                raise RuntimeError(
                    "locked IID audit lacks its runtime-generated sampling receipt"
                )
            panel, draw_take, physical_targets = _iid_physical_terminal_panel(
                draw_panel,
                receipt,
                fold_count=self.context_design.fold_count,
            )
        else:
            panel = draw_panel
            draw_take = np.arange(len(panel.context_ids), dtype=np.int64)
            physical_targets = dict(targets)
        parent_mask_hash = sampled_path_mask_hash(self.initial_masks)
        candidate_mask_hash = sampled_path_mask_hash(self.current_masks)
        parent_execution_state = _query_execution_state_hash(
            query_id=self.context_design.query_id,
            mask_hash=parent_mask_hash,
        )
        candidate_execution_state = _query_execution_state_hash(
            query_id=self.context_design.query_id,
            mask_hash=candidate_mask_hash,
        )
        self._preflight(
            context_ids=panel.context_ids,
            execution_state_hashes=(
                parent_execution_state,
                candidate_execution_state,
            ),
            envelope=self.terminal_envelope,
        )
        request = GlocalSampledComparisonRequest(
            query_id=self.context_design.query_id,
            layer_id="__campaign__",
            partition_id="terminal-audit",
            previous_density=1.0,
            proposed_density=float(
                sum(np.count_nonzero(mask) for mask in self.current_masks.values())
                / max(sum(mask.size for mask in self.current_masks.values()), 1)
            ),
            stage_index=0,
            stage_per_class=max(1, len(panel.context_ids) // 2),
            proposal_index=len(self.campaign.receipt()["proposal_registrations"]),
            random_seed=policy.random_seed,
            fold_count=self.context_design.fold_count,
            cumulative_target_images=len(panel.context_ids),
            per_stratum_target_images=physical_targets,
            maximum_new_contexts=len(panel.context_ids),
        )
        captured: dict[str, Any] = {}

        def evaluate(
            opened_lock: ValidationLock,
            registered_lock_hash: str,
        ) -> TerminalAuditEvidenceReceipt:
            before = _backend_resource_snapshot(self.backend)
            started = time.perf_counter()
            method = getattr(self.backend, "evaluate_glocal_terminal_pair", None)
            if method is None:
                raise RuntimeError(
                    "Torch backend lacks the glocal terminal-audit adapter"
                )
            evidence = method(
                self.initial_masks,
                self.current_masks,
                request,
                panel=panel,
                parent_calibration=self.initial_calibration.native_fit,
                candidate_calibration=self.current_calibration.native_fit,
            )
            runtime_receipt = self.backend.last_glocal_runtime_receipt
            if (
                runtime_receipt is None
                or runtime_receipt.operation != "evaluate_glocal_terminal_pair"
                or runtime_receipt.panel_hash != panel.panel_hash
                or not runtime_receipt.succeeded
            ):
                raise RuntimeError(
                    "terminal physical evidence lacks its exact successful "
                    "backend call receipt"
                )
            elapsed = time.perf_counter() - started
            self._record_runtime_call(
                context_ids=panel.context_ids,
                execution_state_hashes=(
                    parent_execution_state,
                    candidate_execution_state,
                ),
                before=before,
                elapsed=elapsed,
                calibration_fit=False,
                envelope=self.terminal_envelope,
            )
            threshold = float(
                self.context_design.risk_estimand.prediction_rule.threshold
            )
            parent_probabilities = np.asarray(
                evidence.parent_probabilities,
                dtype=np.float64,
            )[draw_take]
            candidate_probabilities = np.asarray(
                evidence.candidate_probabilities,
                dtype=np.float64,
            )[draw_take]
            parent_decisions = parent_probabilities >= threshold
            candidate_decisions = candidate_probabilities >= threshold
            disagreement = parent_decisions != candidate_decisions
            disagreement_by_context = dict(
                zip(
                    draw_panel.context_ids,
                    disagreement.tolist(),
                    strict=True,
                )
            )
            upper, count, stratum_results, bound_method = self._terminal_bounds(
                design=design,
                targets=targets,
                disagreement_by_context=disagreement_by_context,
                policy=policy,
                alpha=opened_lock.terminal_alpha,
            )
            certificate = TerminalParityCertificate(
                validation_lock_hash=registered_lock_hash,
                final_state_hash=self.campaign.current_state_hash,
                audit_observation_set_hash=(
                    opened_lock.audit_observation_set.content_hash
                ),
                disagreement_count=count,
                sample_size=len(draw_panel.context_ids),
                upper_bound=upper,
                tolerance=opened_lock.parity_tolerance,
                certified=upper <= opened_lock.parity_tolerance,
                method=bound_method,
            )
            captured.update(
                {
                    "stratum_results": stratum_results,
                    "physical_evidence": evidence,
                    "runtime_call_receipt_hash": content_hash(
                        runtime_receipt.to_dict()
                    ),
                }
            )
            paired_evidence_hash = (
                evidence.evidence_hash
                if policy.sampling_design != "iid_fixed_n"
                else content_hash(
                    {
                        "schema_name": "iid_terminal_paired_evidence_v1",
                        "physical_source_evidence_hash": evidence.evidence_hash,
                        "iid_sampling_receipt_hash": receipt.content_hash,
                        "statistical_draw_panel_hash": draw_panel.panel_hash,
                        "parent_probabilities": parent_probabilities,
                        "candidate_probabilities": candidate_probabilities,
                        "binary_labels": np.asarray(
                            draw_panel.labels,
                            dtype=np.uint8,
                        ),
                    }
                )
            )
            return TerminalAuditEvidenceReceipt(
                certificate=certificate,
                terminal_precommitment_hash=precommitment.content_hash,
                policy_hash=policy.content_hash,
                sampling_design_hash=design.content_hash,
                paired_evidence_hash=paired_evidence_hash,
                resource_ledger_hash=str(
                    self.campaign.resource_ledger.receipt()["ledger_hash"]
                ),
                observed_context_ids_by_stratum=tuple(
                    (
                        stratum.stratum_id,
                        tuple(
                            stratum.context_ids_in_sealed_order[
                                : int(targets[stratum.stratum_id])
                            ]
                        ),
                    )
                    for stratum in design.strata
                ),
                disagreement_context_ids=tuple(
                    context_id
                    for context_id in draw_panel.context_ids
                    if bool(disagreement_by_context[context_id])
                ),
            )

        audit_receipt = self.campaign.execute_terminal_audit(evaluate)
        certificate = audit_receipt.certificate
        campaign_receipt = self.campaign.receipt()
        result = GlocalTerminalAuditResult(
            certificate=certificate,
            evidence_receipt=audit_receipt,
            policy_hash=policy.content_hash,
            sampling_design_hash=design.content_hash,
            terminal_observation_set_hash=(lock.audit_observation_set.content_hash),
            paired_evidence_hash=audit_receipt.paired_evidence_hash,
            physical_source_evidence_hash=captured["physical_evidence"].evidence_hash,
            runtime_call_receipt_hash=str(captured["runtime_call_receipt_hash"]),
            stratum_results=tuple(captured["stratum_results"]),
            resource_ledger_hash=str(
                self.campaign.resource_ledger.receipt()["ledger_hash"]
            ),
            campaign_receipt_hash=str(campaign_receipt["campaign_hash"]),
        )
        self._terminal_paired_evidence = captured["physical_evidence"]
        self._terminal_result = result
        return result

    @staticmethod
    def _revalidate_physical_authority(
        runtime_owner: object,
        backend_owner: object,
    ) -> Mapping[str, Any]:
        if type(runtime_owner) is not GlocalBrierCampaignRuntime:
            raise RuntimeError(
                "live deployment authority no longer owns its exact runtime"
            )
        if (
            type(backend_owner) is not TorchSampledPathEvaluationBackend
            or runtime_owner.backend is not backend_owner
        ):
            raise RuntimeError(
                "live deployment authority no longer owns its exact Torch backend"
            )
        return runtime_owner._physical_authority_binding_payload(backend_owner)

    def _physical_authority_binding_payload(
        self,
        backend_owner: TorchSampledPathEvaluationBackend,
    ) -> Mapping[str, Any]:
        """Rebuild the complete live execution graph or fail closed."""

        if self.backend is not backend_owner:
            raise RuntimeError("runtime backend identity changed after execution")
        terminal = self._terminal_result
        terminal_evidence = self._terminal_paired_evidence
        if terminal is None or terminal_evidence is None:
            raise RuntimeError(
                "live authority requires retained physical terminal evidence"
            )
        terminal_evidence.validate(fold_count=self.context_design.fold_count)
        initial_mask_hash = sampled_path_mask_hash(self.initial_masks)
        current_mask_hash = sampled_path_mask_hash(self.current_masks)
        risk_hash = self.context_design.risk_estimand.content_hash
        expected_initial_state = locked_brier_state_hash(
            query_id=self.context_design.query_id,
            mask_hash=initial_mask_hash,
            calibration_fit_hash=self.initial_calibration.fit_state.content_hash,
            risk_estimand_hash=risk_hash,
        )
        expected_current_state = locked_brier_state_hash(
            query_id=self.context_design.query_id,
            mask_hash=current_mask_hash,
            calibration_fit_hash=self.current_calibration.fit_state.content_hash,
            risk_estimand_hash=risk_hash,
        )
        if (
            expected_initial_state != self.campaign.initial_state_hash
            or expected_current_state != self.campaign.current_state_hash
            or self.campaign.initial_calibration_hash
            != self.initial_calibration.fit_state.content_hash
            or self.campaign.current_calibration_hash
            != self.current_calibration.fit_state.content_hash
            or self.initial_calibration.native_fit.mask_hash != initial_mask_hash
            or self.current_calibration.native_fit.mask_hash != current_mask_hash
        ):
            raise RuntimeError(
                "runtime masks, calibrations, and campaign states are no longer "
                "one locked state lineage"
            )
        if (
            terminal.physical_source_evidence_hash != terminal_evidence.evidence_hash
            or terminal_evidence.parent_mask_hash != initial_mask_hash
            or terminal_evidence.candidate_mask_hash != current_mask_hash
            or terminal_evidence.parent_calibration_output_hash
            != self.initial_calibration.native_fit.output_hash
            or terminal_evidence.candidate_calibration_output_hash
            != self.current_calibration.native_fit.output_hash
        ):
            raise RuntimeError(
                "retained terminal evidence differs from the locked physical states"
            )

        campaign_receipt = self.campaign.receipt()
        if (
            str(campaign_receipt["campaign_hash"]) != terminal.campaign_receipt_hash
            or str(self.campaign.resource_ledger.receipt()["ledger_hash"])
            != terminal.resource_ledger_hash
            or self.campaign.terminal_audit_evidence_receipt is None
            or self.campaign.terminal_audit_evidence_receipt.content_hash
            != terminal.evidence_receipt.content_hash
        ):
            raise RuntimeError(
                "terminal result differs from the finalized campaign/resource graph"
            )

        registrations = {
            str(item["registration"]["proposal_id"]): item
            for item in campaign_receipt["proposal_registrations"]
        }
        results = {item.proposal_id: item for item in self._proposal_results}
        if len(results) != len(self._proposal_results):
            raise RuntimeError("runtime proposal result identities are not unique")
        if set(registrations) != {
            proposal_id
            for proposal_id, result in results.items()
            if result.registration_hash is not None
        }:
            raise RuntimeError(
                "campaign registrations and retained physical proposal results differ"
            )
        if (
            set(self._proposal_calibrations) != set(registrations)
            or set(self._proposal_paired_evidence) != set(registrations)
            or set(self._proposal_evidence_snapshots) != set(registrations)
        ):
            raise RuntimeError(
                "registered proposal calibration/evidence lineage is incomplete"
            )
        decision_hashes = {receipt.content_hash for receipt in self.campaign.receipts}
        decision_by_look = {
            (receipt.proposal_id, receipt.look_index): receipt
            for receipt in self.campaign.receipts
        }
        proposal_bindings: list[Mapping[str, Any]] = []
        for proposal_id, registration_entry in registrations.items():
            result = results[proposal_id]
            registration = dict(registration_entry["registration"])
            if (
                result.registration_hash != str(registration_entry["registration_hash"])
                or result.candidate_state_hash
                != str(registration["candidate_state_hash"])
                or result.candidate_calibration_hash
                != str(registration["candidate_calibration_hash"])
                or (
                    result.decision_receipt is not None
                    and result.decision_receipt.content_hash not in decision_hashes
                )
            ):
                raise RuntimeError(
                    f"proposal {proposal_id!r} result differs from its campaign "
                    "registration/decision receipts"
                )
            expected_status = {
                "accepted": "accepted",
                "rejected": "rejected",
                "unresolved": "abandoned",
            }[result.outcome]
            if str(registration_entry["status"]) != expected_status:
                raise RuntimeError(
                    f"proposal {proposal_id!r} terminal status is inconsistent"
                )
            parent_calibration, candidate_calibration = self._proposal_calibrations[
                proposal_id
            ]
            parent_mask_hash = parent_calibration.native_fit.mask_hash
            candidate_mask_hash = candidate_calibration.native_fit.mask_hash
            if (
                parent_calibration.fit_state.content_hash
                != str(registration["parent_calibration_hash"])
                or candidate_calibration.fit_state.content_hash
                != str(registration["candidate_calibration_hash"])
                or locked_brier_state_hash(
                    query_id=self.context_design.query_id,
                    mask_hash=parent_mask_hash,
                    calibration_fit_hash=(parent_calibration.fit_state.content_hash),
                    risk_estimand_hash=risk_hash,
                )
                != str(registration["parent_state_hash"])
                or locked_brier_state_hash(
                    query_id=self.context_design.query_id,
                    mask_hash=candidate_mask_hash,
                    calibration_fit_hash=(candidate_calibration.fit_state.content_hash),
                    risk_estimand_hash=risk_hash,
                )
                != str(registration["candidate_state_hash"])
            ):
                raise RuntimeError(
                    f"proposal {proposal_id!r} state/calibration lineage changed"
                )
            evidences = tuple(self._proposal_paired_evidence[proposal_id])
            snapshots = tuple(self._proposal_evidence_snapshots[proposal_id])
            if len(evidences) != len(result.looks) or len(snapshots) != len(
                result.looks
            ):
                raise RuntimeError(
                    f"proposal {proposal_id!r} lost physical evidence looks"
                )
            evidence_bindings: list[Mapping[str, Any]] = []
            design = self.campaign._sampling_designs[proposal_id]
            for evidence, snapshot, look in zip(
                evidences,
                snapshots,
                result.looks,
                strict=True,
            ):
                evidence.validate(fold_count=self.context_design.fold_count)
                increments = paired_brier_increment(
                    evidence.candidate_probabilities,
                    evidence.parent_probabilities,
                    evidence.binary_labels,
                    prediction_rule=(self.context_design.risk_estimand.prediction_rule),
                )
                increment_by_context = dict(
                    zip(
                        evidence.panel.context_ids,
                        increments.tolist(),
                        strict=True,
                    )
                )
                raw_values = dict(snapshot.observed_values_by_stratum)
                raw_values_match = all(
                    len(raw_values[stratum.stratum_id])
                    == snapshot.strata[
                        tuple(item.stratum_id for item in snapshot.strata).index(
                            stratum.stratum_id
                        )
                    ].sample_size
                    and all(
                        math.isclose(
                            observed,
                            float(increment_by_context[context_id]),
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        )
                        for observed, context_id in zip(
                            raw_values[stratum.stratum_id],
                            stratum.context_ids_in_sealed_order[
                                : len(raw_values[stratum.stratum_id])
                            ],
                            strict=True,
                        )
                    )
                    for stratum in design.strata
                )
                decision_receipt = decision_by_look.get((proposal_id, look.look_index))
                if (
                    evidence.evidence_hash != look.paired_evidence_hash
                    or evidence.panel.panel_hash != look.panel_hash
                    or look.increment_hash
                    != content_hash(
                        {
                            "panel_hash": evidence.panel.panel_hash,
                            "increments": np.asarray(
                                increments,
                                dtype=np.float64,
                            ),
                        }
                    )
                    or snapshot.content_hash != look.evidence_snapshot_hash
                    or not raw_values_match
                    or decision_receipt is None
                    or decision_receipt.evidence_snapshot_hash != snapshot.content_hash
                    or decision_receipt.allocation_look_receipt_hash
                    != look.allocation_look_receipt_hash
                    or evidence.parent_mask_hash != parent_mask_hash
                    or evidence.candidate_mask_hash != candidate_mask_hash
                    or evidence.parent_calibration_output_hash
                    != parent_calibration.native_fit.output_hash
                    or evidence.candidate_calibration_output_hash
                    != candidate_calibration.native_fit.output_hash
                ):
                    raise RuntimeError(
                        f"proposal {proposal_id!r} retained evidence changed"
                    )
                evidence_bindings.append(
                    {
                        "object_id": id(evidence),
                        "evidence_hash": evidence.evidence_hash,
                        "panel_hash": evidence.panel.panel_hash,
                        "evidence_snapshot_object_id": id(snapshot),
                        "evidence_snapshot_hash": snapshot.content_hash,
                        "allocation_look_receipt_hash": (
                            look.allocation_look_receipt_hash
                        ),
                        "look_hash": look.content_hash,
                    }
                )
            proposal_bindings.append(
                {
                    "proposal_id": proposal_id,
                    "result_hash": result.content_hash,
                    "parent_calibration_hash": (
                        parent_calibration.fit_state.content_hash
                    ),
                    "candidate_calibration_hash": (
                        candidate_calibration.fit_state.content_hash
                    ),
                    "parent_calibration_output_hash": (
                        parent_calibration.native_fit.output_hash
                    ),
                    "candidate_calibration_output_hash": (
                        candidate_calibration.native_fit.output_hash
                    ),
                    "evidence": tuple(evidence_bindings),
                }
            )

        runtime_receipts = tuple(self.backend.glocal_runtime_receipts)
        runtime_receipt_hashes = tuple(
            content_hash(receipt.to_dict()) for receipt in runtime_receipts
        )
        if terminal.runtime_call_receipt_hash not in runtime_receipt_hashes:
            raise RuntimeError(
                "terminal runtime call receipt is absent from the owning backend"
            )
        query_bindings = tuple(
            sorted(
                (
                    str(query_id),
                    str(query.content_hash),
                )
                for query_id, query in self.backend._query_by_id.items()
            )
        )
        if self.context_design.query_id not in dict(query_bindings):
            raise RuntimeError("physical backend no longer owns the campaign query")
        panel_source_axis = getattr(self.backend.panel_source, "context_axis", None)
        if (
            getattr(panel_source_axis, "content_hash", None)
            != self.context_design.context_axis_hash
        ):
            raise RuntimeError(
                "physical panel source no longer owns the frozen context axis"
            )

        return MappingProxyType(
            {
                "schema_name": "glocal_physical_runtime_binding_v1",
                "live_capability_nonce": self._live_capability_nonce,
                "runtime": _physical_live_object_binding(self, role="runtime"),
                "backend": _physical_live_object_binding(
                    self.backend,
                    role="torch_backend",
                ),
                "evaluator": _physical_live_object_binding(
                    self.backend.evaluator,
                    role="torch_evaluator",
                ),
                "panel_source": _physical_live_object_binding(
                    self.backend.panel_source,
                    role="physical_panel_source",
                ),
                "backend_query_bindings": query_bindings,
                "backend_calibration_options_hash": content_hash(
                    dict(self.backend.calibration_options)
                ),
                "backend_runtime_profile_hash": content_hash(
                    self.backend.runtime_execution_profile
                ),
                "physical_provider_identity_hash": (
                    iid_physical_provider_identity_hash(self.backend.panel_source)
                ),
                "physical_executor_identity_hash": (
                    iid_physical_executor_identity_hash(self.backend)
                ),
                "backend_resource_usage_hash": content_hash(
                    self.backend.resource_usage
                ),
                "backend_runtime_receipt_hashes": runtime_receipt_hashes,
                "campaign_id": self.campaign.campaign_id,
                "campaign_receipt_hash": str(campaign_receipt["campaign_hash"]),
                "campaign_resource_ledger_hash": str(
                    self.campaign.resource_ledger.receipt()["ledger_hash"]
                ),
                "context_design_hash": self.context_design.content_hash,
                "context_population_hash": (
                    self.context_design.risk_estimand.context_population_hash
                ),
                "risk_estimand_hash": risk_hash,
                "initial_mask_hash": initial_mask_hash,
                "current_mask_hash": current_mask_hash,
                "initial_state_hash": expected_initial_state,
                "current_state_hash": expected_current_state,
                "initial_calibration_hash": (
                    self.initial_calibration.fit_state.content_hash
                ),
                "current_calibration_hash": (
                    self.current_calibration.fit_state.content_hash
                ),
                "proposal_bindings": tuple(proposal_bindings),
                "unregistered_result_hashes": tuple(
                    result.content_hash
                    for result in self._proposal_results
                    if result.registration_hash is None
                ),
                "terminal_result_hash": terminal.content_hash,
                "terminal_evidence_object_id": id(terminal_evidence),
                "terminal_physical_source_evidence_hash": (
                    terminal_evidence.evidence_hash
                ),
                "terminal_statistical_evidence_hash": (terminal.paired_evidence_hash),
                "terminal_panel_hash": terminal_evidence.panel.panel_hash,
                "terminal_runtime_call_receipt_hash": (
                    terminal.runtime_call_receipt_hash
                ),
                "terminal_audit_evidence_receipt_hash": (
                    terminal.evidence_receipt.content_hash
                ),
            }
        )

    def issue_deployment_certificate(
        self,
        *,
        native_route_axis_hash: str,
        native_mask_bridge_hashes: tuple[str, ...] = (),
        irg_final_state_hash: str | None = None,
        irg_final_mask_hash: str | None = None,
        external_scientific_identity_hash: str | None = None,
        external_query_measure_hash: str | None = None,
        certificate_id: str | None = None,
    ) -> GlocalDeploymentMaskCertificate:
        """Bind the actual locked hard mask to both glocal guarantees."""

        if self._deployment_certificate is not None:
            raise RuntimeError(
                "this terminally locked campaign already issued its deployment "
                "certificate"
            )
        terminal = self._terminal_result
        if terminal is None:
            raise RuntimeError(
                "deployment certification requires the executed terminal audit"
            )
        native_mask_hash = sampled_path_mask_hash(self.current_masks)
        certificate = build_glocal_deployment_mask_certificate(
            campaign=self.campaign,
            query_id=self.context_design.query_id,
            context_design_hash=self.context_design.content_hash,
            native_route_axis_hash=str(native_route_axis_hash),
            native_mask_hash=native_mask_hash,
            terminal_audit_result_hash=terminal.content_hash,
            terminal_sampling_design_hash=terminal.sampling_design_hash,
            native_mask_bridge_hashes=tuple(native_mask_bridge_hashes),
            irg_final_state_hash=irg_final_state_hash,
            irg_final_mask_hash=irg_final_mask_hash,
            external_scientific_identity_hash=(external_scientific_identity_hash),
            external_query_measure_hash=external_query_measure_hash,
            certificate_id=certificate_id,
        )
        if terminal.campaign_receipt_hash != certificate.campaign_receipt_hash:
            raise RuntimeError(
                "terminal audit result does not bind the finalized campaign receipt"
            )
        self._deployment_certificate = certificate
        return certificate

    def issue_deployment_authority(
        self,
        *,
        native_route_axis_hash: str,
        native_mask_bridge_hashes: tuple[str, ...] = (),
        irg_final_state_hash: str | None = None,
        irg_final_mask_hash: str | None = None,
        external_scientific_identity_hash: str | None = None,
        external_query_measure_hash: str | None = None,
        certificate_id: str | None = None,
    ) -> GlocalDeploymentAuthorityHandle:
        """Return live campaign authority; detached certificates are not enough."""

        request_hash = content_hash(
            {
                "native_route_axis_hash": str(native_route_axis_hash),
                "native_mask_bridge_hashes": tuple(native_mask_bridge_hashes),
                "irg_final_state_hash": irg_final_state_hash,
                "irg_final_mask_hash": irg_final_mask_hash,
                "external_scientific_identity_hash": (
                    external_scientific_identity_hash
                ),
                "external_query_measure_hash": external_query_measure_hash,
                "certificate_id": certificate_id,
            }
        )
        if self._deployment_authority_handle is not None:
            if request_hash != self._deployment_authority_request_hash:
                raise RuntimeError(
                    "conflicting deployment-authority identities were supplied "
                    "after one live capability was issued"
                )
            self._deployment_authority_handle.require_current()
            return self._deployment_authority_handle
        terminal = self._terminal_result
        if terminal is None:
            raise RuntimeError(
                "deployment authority requires the executed terminal audit"
            )
        native_mask_hash = sampled_path_mask_hash(self.current_masks)
        handle = _issue_physical_glocal_deployment_authority(
            runtime_owner=self,
            backend_owner=self.backend,
            campaign=self.campaign,
            query_id=self.context_design.query_id,
            context_design_hash=self.context_design.content_hash,
            native_route_axis_hash=str(native_route_axis_hash),
            native_mask_hash=native_mask_hash,
            terminal_audit_result_hash=terminal.content_hash,
            terminal_sampling_design_hash=terminal.sampling_design_hash,
            native_mask_bridge_hashes=tuple(native_mask_bridge_hashes),
            irg_final_state_hash=irg_final_state_hash,
            irg_final_mask_hash=irg_final_mask_hash,
            external_scientific_identity_hash=(external_scientific_identity_hash),
            external_query_measure_hash=external_query_measure_hash,
            certificate_id=certificate_id,
        )
        if (
            self._deployment_certificate is not None
            and self._deployment_certificate.content_hash
            != handle.certificate.content_hash
        ):
            raise RuntimeError(
                "existing deployment certificate differs from live authority"
            )
        self._deployment_certificate = handle.certificate
        self._deployment_authority_handle = handle
        self._deployment_authority_request_hash = request_hash
        return handle

    @property
    def calibration_targets(self) -> Mapping[str, int]:
        """Largest role-safe calibration prefix physically admitted so far."""

        return MappingProxyType(dict(self._available_calibration_targets))

    @property
    def calibration_logit_rows(
        self,
    ) -> tuple[GlocalCalibrationLogitRows, ...]:
        """Retained cumulative physical rows for exact calibration replay."""

        return tuple(self._calibration_logit_rows_by_hash.values())

    @property
    def calibration_support_exhausted(self) -> bool:
        design = self.context_design.calibration_sampling_design
        if design is None:
            return True
        return all(
            self._available_calibration_targets[item.stratum_id] >= item.population_size
            for item in design.strata
        )

    def planned_calibration_targets(
        self,
        *,
        additional_contexts: int,
    ) -> dict[str, int]:
        if not self.context_design.adaptive_calibration:
            return dict(self._available_calibration_targets)
        return self.context_design.expand_calibration_targets(
            self._available_calibration_targets,
            additional_contexts=int(additional_contexts),
        )

    def calibration_targets_for_total(
        self,
        target_context_count: int,
    ) -> dict[str, int]:
        """Map a requested total onto the birth-frozen balanced prefixes."""

        target = int(target_context_count)
        if target <= 0:
            raise ValueError("calibration target context count must be positive")
        if not self.context_design.adaptive_calibration:
            return dict(self._available_calibration_targets)
        initial = dict(self.context_design.calibration_initial_targets)
        minimum = sum(initial.values())
        desired = max(
            target, minimum, sum(self._available_calibration_targets.values())
        )
        resolved = self.context_design.expand_calibration_targets(
            initial,
            additional_contexts=desired - minimum,
        )
        if any(
            resolved[key] < self._available_calibration_targets[key] for key in resolved
        ):
            raise RuntimeError(
                "calibration total target would regress an admitted prefix"
            )
        return resolved

    def acquire_calibration_logits(
        self,
        *,
        target_context_count: int,
    ) -> tuple[
        GlocalCalibrationLogitRows,
        ObservationSetRef,
        tuple[str, ...],
        tuple[tuple[str, int], ...],
        tuple[tuple[str, int], ...],
        str,
        str,
        str,
    ]:
        """Physically admit the next sealed calibration-role prefix."""

        design = self.context_design.calibration_sampling_design
        if design is None:
            raise RuntimeError(
                "static calibration census has no separate logit-prefix acquisition"
            )
        before_targets = dict(self._available_calibration_targets)
        after_targets = self.calibration_targets_for_total(target_context_count)
        if after_targets == before_targets:
            raise RuntimeError(
                "calibration-logit request adds no new sealed support contexts"
            )
        before_panel = self.context_design.calibration_panel(before_targets)
        panel = self.context_design.calibration_panel(after_targets)
        panel.assert_nested_extension(before_panel)
        prior_contexts = set(before_panel.context_ids)
        new_contexts = tuple(
            context_id
            for context_id in panel.context_ids
            if context_id not in prior_contexts
        )
        if not new_contexts:
            raise RuntimeError(
                "calibration-logit request changed targets without adding support rows"
            )
        observation_set = self._own_calibration_observation_set(after_targets)
        mask_hash = sampled_path_mask_hash(self.current_masks)
        execution_state = _query_execution_state_hash(
            query_id=self.context_design.query_id,
            mask_hash=mask_hash,
        )
        logit_envelope = replace(self.calibration_envelope, calibration_fits=0)
        self._preflight(
            context_ids=panel.context_ids,
            execution_state_hashes=(execution_state,),
            envelope=logit_envelope,
        )
        request = self._calibration_request(
            context_design=self.context_design,
            panel=panel,
            proposal_index=(len(self.campaign.receipt()["proposal_registrations"]) + 1),
            maximum_new_contexts=len(new_contexts),
            prior_panel=before_panel,
            stage_index=len(self._proposal_results) + 1,
        )
        resource_before_hash = str(
            self.campaign.resource_ledger.receipt()["ledger_hash"]
        )
        before = _backend_resource_snapshot(self.backend)
        started = time.perf_counter()
        rows = self.backend.evaluate_glocal_calibration_logits(
            self.current_masks,
            request,
            panel=panel,
        )
        elapsed = time.perf_counter() - started
        runtime_receipt = self.backend.last_glocal_runtime_receipt
        if (
            runtime_receipt is None
            or runtime_receipt.operation != "evaluate_glocal_calibration_logits"
            or runtime_receipt.panel_hash != panel.panel_hash
            or not runtime_receipt.succeeded
        ):
            raise RuntimeError(
                "calibration-logit acquisition lacks its successful exact "
                "backend call receipt"
            )
        self._record_runtime_call(
            context_ids=panel.context_ids,
            execution_state_hashes=(execution_state,),
            before=before,
            elapsed=elapsed,
            calibration_fit=False,
            envelope=logit_envelope,
        )
        prior_rows = self._calibration_logit_rows_by_hash.get(rows.evidence_hash)
        if prior_rows is not None and (
            prior_rows.panel.context_ids != rows.panel.context_ids
            or not np.array_equal(prior_rows.logits, rows.logits)
            or not np.array_equal(prior_rows.binary_labels, rows.binary_labels)
        ):
            raise RuntimeError(
                "calibration row-evidence hash collided with different physical rows"
            )
        self._calibration_logit_rows_by_hash[rows.evidence_hash] = rows
        self._available_calibration_targets = dict(after_targets)
        resource_after_hash = str(
            self.campaign.resource_ledger.receipt()["ledger_hash"]
        )
        return (
            rows,
            observation_set,
            new_contexts,
            tuple(sorted(before_targets.items())),
            tuple(sorted(after_targets.items())),
            content_hash(runtime_receipt.to_dict()),
            resource_before_hash,
            resource_after_hash,
        )

    def _own_calibration_observation_set(
        self,
        targets: Mapping[str, int],
    ) -> ObservationSetRef:
        observation_set = self.context_design.calibration_observation_set_for_targets(
            targets
        )
        for existing in self.campaign.observation_sets:
            if existing.content_hash == observation_set.content_hash:
                return existing
            if existing.observation_set_id == observation_set.observation_set_id:
                raise RuntimeError(
                    "calibration prefix observation-set ID was reused for "
                    "different physical rows"
                )
        return self.campaign.register_observation_set(observation_set)

    def _fit_candidate_calibration(
        self,
        candidate_masks: Mapping[str, np.ndarray],
        *,
        proposal_index: int,
        proposal_id: str,
        calibration_targets: Mapping[str, int] | None = None,
    ) -> FrozenCalibrationBinding:
        resolved_targets = (
            dict(self._available_calibration_targets)
            if calibration_targets is None
            else {str(key): int(value) for key, value in calibration_targets.items()}
        )
        if set(resolved_targets) != set(self._available_calibration_targets) or any(
            resolved_targets[key] < self._available_calibration_targets[key]
            for key in resolved_targets
        ):
            raise ValueError(
                "candidate calibration must use a nested extension of the "
                "admitted fit-role prefix"
            )
        observation_set = self._own_calibration_observation_set(resolved_targets)
        panel = self.context_design.calibration_panel(resolved_targets)
        if panel.context_ids != observation_set.context_ids:
            raise RuntimeError(
                "candidate calibration panel differs from its registered prefix"
            )
        mask_hash = sampled_path_mask_hash(candidate_masks)
        execution_state = _query_execution_state_hash(
            query_id=self.context_design.query_id,
            mask_hash=mask_hash,
        )
        self._preflight(
            context_ids=panel.context_ids,
            execution_state_hashes=(execution_state,),
            envelope=self.calibration_envelope,
        )
        request = self._calibration_request(
            context_design=self.context_design,
            panel=panel,
            proposal_index=proposal_index,
            maximum_new_contexts=len(panel.context_ids),
        )
        before = _backend_resource_snapshot(self.backend)
        started = time.perf_counter()
        input_rows = self.backend.evaluate_glocal_calibration_logits(
            candidate_masks,
            request,
            panel=panel,
        )
        logit_receipt = self.backend.last_glocal_runtime_receipt
        if (
            logit_receipt is None
            or logit_receipt.operation != "evaluate_glocal_calibration_logits"
            or logit_receipt.panel_hash != panel.panel_hash
            or not logit_receipt.succeeded
        ):
            raise RuntimeError(
                "candidate calibration logits lack their successful backend receipt"
            )
        native = self.backend.fit_glocal_brier_calibration(
            candidate_masks,
            request,
            role=EvidenceRole.CALIBRATION_FIT.value,
            panel=panel,
            warm_start=self.current_calibration.native_fit,
            maximum_iterations=self.current_calibration.algorithm_spec.maximum_iterations,
            gradient_tolerance=(
                self.current_calibration.algorithm_spec.convergence_tolerance
            ),
            identity_ridge=float(
                self.current_calibration.algorithm_spec.options["identity_ridge"]
            ),
        )
        fit_receipt = self.backend.last_glocal_runtime_receipt
        if (
            fit_receipt is None
            or fit_receipt.operation != "fit_glocal_brier_calibration"
            or fit_receipt.panel_hash != panel.panel_hash
            or not fit_receipt.succeeded
        ):
            raise RuntimeError(
                "candidate calibration fit lacks its successful backend receipt"
            )
        elapsed = time.perf_counter() - started
        self._record_runtime_call(
            context_ids=panel.context_ids,
            execution_state_hashes=(execution_state,),
            before=before,
            elapsed=elapsed,
            calibration_fit=True,
            envelope=self.calibration_envelope,
        )
        binding = calibration_binding_from_native(
            native,
            fit_id=f"{self.campaign.campaign_id}:{proposal_id}:candidate-calibration",
            observation_set=observation_set,
            algorithm_spec=self.current_calibration.algorithm_spec,
            warm_start_parent=self.current_calibration.fit_state,
            fit_input_evidence_hash=input_rows.evidence_hash,
        )
        prior_rows = self._calibration_logit_rows_by_hash.get(input_rows.evidence_hash)
        if prior_rows is not None and (
            prior_rows.panel.context_ids != input_rows.panel.context_ids
            or not np.array_equal(prior_rows.logits, input_rows.logits)
            or not np.array_equal(
                prior_rows.binary_labels,
                input_rows.binary_labels,
            )
        ):
            raise RuntimeError(
                "candidate calibration input hash collided with different rows"
            )
        self._calibration_logit_rows_by_hash[input_rows.evidence_hash] = input_rows
        self._available_calibration_targets = dict(resolved_targets)
        return binding

    def _allocation_variance(self, values: Sequence[float]) -> float:
        count = len(values)
        prior = float(self.policy.allocation_variance_prior)
        strength = float(self.policy.variance_moderation_strength)
        if count < 2:
            return prior
        raw = float(np.var(np.asarray(values, dtype=np.float64), ddof=1))
        dof = float(count - 1)
        return max(1e-12, (dof * raw + strength * prior) / (dof + strength))

    def _next_targets(
        self,
        *,
        design: SamplingDesignSpec,
        targets: Mapping[str, int],
        observed_values: Mapping[str, Sequence[float]],
        cost_per_context: Mapping[str, float],
    ) -> (
        tuple[
            dict[str, int],
            str,
            dict[str, float],
            Mapping[str, Any],
        ]
        | None
    ):
        current = {str(key): int(value) for key, value in targets.items()}
        allocator = UnifiedNeymanAllocator()
        weights = {item.stratum_id: item.risk_weight for item in design.strata}
        capacities = {item.stratum_id: item.population_size for item in design.strata}
        standard_deviations = {
            stratum_id: math.sqrt(
                self._allocation_variance(observed_values[stratum_id])
            )
            for stratum_id in current
        }
        specs = tuple(
            NeymanStratumSpec(
                stratum_id=item.stratum_id,
                capacity=item.population_size,
                estimand_weight=item.risk_weight,
                allocation_standard_deviation=standard_deviations[item.stratum_id],
                unit_cost=max(
                    float(cost_per_context[item.stratum_id]),
                    1e-12,
                ),
                minimum_count=(
                    min(
                        item.population_size,
                        int(self.policy.initial_images_per_stratum),
                    )
                    if not any(current.values())
                    else 0
                ),
            )
            for item in design.strata
        )
        if not any(current.values()):
            initial_target = sum(item.minimum_count for item in specs)
            result = allocator.allocate(
                NeymanAllocationRequest(
                    strata=specs,
                    existing_counts=current,
                    cumulative_count_target=initial_target,
                    require_normalized_estimand_weights=True,
                    role="glocal_brier_adjudication",
                    policy_id=UNIFIED_NEYMAN_POLICY_ID,
                )
            )
            initial = dict(result.target_counts)
            if sum(initial.values()) < self.context_design.fold_count:
                raise ValueError(
                    "initial glocal packet cannot cover every declared fold"
                )
            return (
                initial,
                "__initial_all_strata__",
                {item.stratum_id: math.inf for item in design.strata},
                result.to_receipt(),
            )
        scores = allocator.marginal_priorities(specs, current)
        if not scores:
            return None
        best = max(scores, key=lambda key: (scores[key], key))
        if scores[best] < float(self.policy.minimum_information_value_per_cost):
            return None
        target_total = min(
            sum(capacities.values()),
            sum(current.values()) + int(self.policy.packet_images),
        )
        maximum = self.policy.maximum_adjudication_images
        if maximum is not None:
            target_total = min(target_total, int(maximum))
        if target_total <= sum(current.values()):
            return None
        result = allocator.allocate(
            NeymanAllocationRequest(
                strata=specs,
                existing_counts=current,
                cumulative_count_target=target_total,
                require_normalized_estimand_weights=True,
                role="glocal_brier_adjudication",
                policy_id=UNIFIED_NEYMAN_POLICY_ID,
            )
        )
        updated = dict(result.target_counts)
        if updated == current:
            return None
        selected = tuple(
            stratum_id
            for stratum_id, addition in result.additions.items()
            if int(addition) > 0
        )
        chosen = (
            selected[0] if len(selected) == 1 else "__multi_stratum_neyman_packet__"
        )
        return updated, chosen, scores, result.to_receipt()

    def _record_proposal_result(
        self,
        result: GlocalProposalResult,
    ) -> GlocalProposalResult:
        if any(
            existing.proposal_id == result.proposal_id
            for existing in self._proposal_results
        ):
            raise RuntimeError(
                f"proposal {result.proposal_id!r} already has a runtime result"
            )
        self._proposal_results.append(result)
        return result

    def run_calibration_transition(
        self,
        *,
        proposal_id: str,
        candidate_definition_hash: str,
        local_upper_boundary: float,
        group_id: str,
        random_seed: int,
        requested_context_increment: int = 0,
    ) -> GlocalProposalResult:
        """Certify a calibration-only state change under the unchanged mask.

        The candidate calibration is fitted inside ``run_proposal`` and the
        paired Brier increment is measured against the currently committed
        calibration on the exact same physical mask.  This is intentionally a
        campaign proposal: it reserves alpha, consumes the global resource
        ledger, and can advance the parent calibration only after acceptance.
        """

        masks = {
            str(layer_id): np.asarray(mask, dtype=bool).copy()
            for layer_id, mask in self.current_masks.items()
        }
        retained = sum(int(np.count_nonzero(mask)) for mask in masks.values())
        total = sum(int(mask.size) for mask in masks.values())
        density = retained / max(total, 1)
        calibration_targets = self.planned_calibration_targets(
            additional_contexts=int(requested_context_increment)
        )
        growth_authorization = content_hash(
            {
                "nonce": self._calibration_growth_nonce,
                "proposal_id": str(proposal_id),
                "candidate_definition_hash": str(candidate_definition_hash),
                "calibration_targets": tuple(sorted(calibration_targets.items())),
            }
        )
        return self.run_proposal(
            proposal_id=str(proposal_id),
            candidate_masks=masks,
            candidate_definition_hash=str(candidate_definition_hash),
            local_upper_boundary=float(local_upper_boundary),
            group_id=str(group_id),
            random_seed=int(random_seed),
            layer_id="__calibration_state__",
            partition_id="__calibration_state__",
            previous_density=density,
            proposed_density=density,
            candidate_calibration_targets=calibration_targets,
            _calibration_growth_authorization=growth_authorization,
        )

    def run_proposal(
        self,
        *,
        proposal_id: str,
        candidate_masks: Mapping[str, np.ndarray],
        candidate_definition_hash: str,
        local_upper_boundary: float,
        group_id: str,
        random_seed: int,
        layer_id: str = "__campaign__",
        partition_id: str = "__candidate__",
        previous_density: float = 1.0,
        proposed_density: float = 0.0,
        candidate_calibration_targets: Mapping[str, int] | None = None,
        _calibration_growth_authorization: str | None = None,
    ) -> GlocalProposalResult:
        """Fit, register, sequentially adjudicate, and possibly commit one state."""

        if any(
            existing.proposal_id == str(proposal_id)
            for existing in self._proposal_results
        ):
            raise ValueError(
                f"proposal {proposal_id!r} already executed in this runtime"
            )
        if self.campaign.terminal_lock is not None:
            raise RuntimeError(
                "campaign is terminally locked; no further proposals are legal"
            )
        _require_sha256(candidate_definition_hash, "candidate definition hash")
        if candidate_calibration_targets is not None:
            normalized_calibration_targets = {
                str(key): int(value)
                for key, value in candidate_calibration_targets.items()
            }
            if normalized_calibration_targets != dict(
                self._available_calibration_targets
            ):
                expected_growth_authorization = content_hash(
                    {
                        "nonce": self._calibration_growth_nonce,
                        "proposal_id": str(proposal_id),
                        "candidate_definition_hash": str(candidate_definition_hash),
                        "calibration_targets": tuple(
                            sorted(normalized_calibration_targets.items())
                        ),
                    }
                )
                if (
                    _calibration_growth_authorization is None
                    or not secrets.compare_digest(
                        str(_calibration_growth_authorization),
                        expected_growth_authorization,
                    )
                ):
                    raise RuntimeError(
                        "only the calibration-transition path may grow the "
                        "fit-role prefix"
                    )
        proposal_index = len(self.campaign.receipt()["proposal_registrations"]) + 1
        try:
            candidate_calibration = self._fit_candidate_calibration(
                candidate_masks,
                proposal_index=proposal_index,
                proposal_id=proposal_id,
                calibration_targets=candidate_calibration_targets,
            )
        except ResourceBudgetUnavailable as error:
            campaign_receipt = self.campaign.receipt()
            return self._record_proposal_result(
                GlocalProposalResult(
                    proposal_id=str(proposal_id),
                    outcome="unresolved",
                    reason=str(error),
                    committed=False,
                    registration_hash=None,
                    candidate_state_hash=None,
                    candidate_calibration_hash=None,
                    decision_receipt=None,
                    looks=(),
                    campaign_receipt_hash=str(campaign_receipt["campaign_hash"]),
                )
            )
        candidate_mask_hash = sampled_path_mask_hash(candidate_masks)
        candidate_state_hash = locked_brier_state_hash(
            query_id=self.context_design.query_id,
            mask_hash=candidate_mask_hash,
            calibration_fit_hash=candidate_calibration.fit_state.content_hash,
            risk_estimand_hash=self.context_design.risk_estimand.content_hash,
        )
        design, adjudication_set = self.context_design.freeze_proposal_design(
            proposal_id=proposal_id,
            candidate_definition_hash=candidate_definition_hash,
            random_seed=int(random_seed),
        )
        registration = self.campaign.register_proposal(
            proposal_id=str(proposal_id),
            query_id=self.context_design.query_id,
            group_id=str(group_id),
            parent_state_hash=self.campaign.current_state_hash,
            candidate_state_hash=candidate_state_hash,
            parent_calibration=self.current_calibration.fit_state,
            candidate_calibration=candidate_calibration.fit_state,
            sampling_design=design,
            adjudication_observation_set=adjudication_set,
            local_upper_boundary=float(local_upper_boundary),
            allocation_rule_id=GLOCAL_NEYMAN_ALLOCATION_RULE,
            allocation_rule_hash=self.policy.content_hash,
            candidate_definition_hash=str(candidate_definition_hash),
        )
        self._proposal_calibrations[str(proposal_id)] = (
            self.current_calibration,
            candidate_calibration,
        )
        self._proposal_paired_evidence[str(proposal_id)] = []
        self._proposal_evidence_snapshots[str(proposal_id)] = []
        process = StratifiedPairedEProcess(
            sampling_design=design,
            alpha=registration.alpha_reserved,
            stratum_alpha=dict(registration.stratum_alpha),
        )
        targets = {item.stratum_id: 0 for item in design.strata}
        observed: dict[str, list[float]] = {
            item.stratum_id: [] for item in design.strata
        }
        costs = {
            item.stratum_id: float(self.policy.initial_cost_per_context)
            for item in design.strata
        }
        prior_panel: CrossfitPanel | None = None
        prior_snapshot = None
        prior_allocation_receipt: PredictableAllocationLookReceipt | None = None
        panel_hashes: list[str] = []
        looks: list[GlocalProposalLook] = []
        final_receipt: DecisionEvidenceReceipt | None = None
        outcome: Literal["accepted", "rejected", "unresolved"] = "unresolved"
        committed = False
        reason = "finite population exhausted without an admissible commit"
        parent_mask_hash = sampled_path_mask_hash(self.current_masks)
        parent_execution_state = _query_execution_state_hash(
            query_id=self.context_design.query_id,
            mask_hash=parent_mask_hash,
        )
        candidate_execution_state = _query_execution_state_hash(
            query_id=self.context_design.query_id,
            mask_hash=candidate_mask_hash,
        )
        while True:
            if self.policy.maximum_looks is not None and len(looks) >= int(
                self.policy.maximum_looks
            ):
                reason = "maximum predictable evidence looks exhausted"
                break
            next_allocation = self._next_targets(
                design=design,
                targets=targets,
                observed_values=observed,
                cost_per_context=costs,
            )
            if next_allocation is None:
                reason = "futility_or_value_of_information_stopped_unresolved"
                break
            (
                next_targets,
                selected_stratum,
                allocation_scores,
                unified_allocator_receipt,
            ) = next_allocation
            panel = self.context_design.proposal_panel(
                design,
                targets=next_targets,
                stage_index=len(looks),
                prior_panel=prior_panel,
            )
            previous_contexts = (
                set() if prior_panel is None else set(prior_panel.context_ids)
            )
            new_contexts = tuple(
                value for value in panel.context_ids if value not in previous_contexts
            )
            ledger_before_hash = str(
                self.campaign.resource_ledger.receipt()["ledger_hash"]
            )
            try:
                resource_preview = self._preflight(
                    context_ids=panel.context_ids,
                    execution_state_hashes=(
                        parent_execution_state,
                        candidate_execution_state,
                    ),
                    envelope=self.decision_envelope,
                )
            except ResourceBudgetUnavailable as error:
                reason = str(error)
                break
            allocation_score_receipt = tuple(
                (
                    key,
                    (
                        value
                        if math.isfinite(float(value))
                        else (
                            "positive_infinity"
                            if float(value) > 0.0
                            else "negative_infinity"
                        )
                    ),
                )
                for key, value in sorted(allocation_scores.items())
            )
            allocation_receipt = PredictableAllocationLookReceipt(
                proposal_id=str(proposal_id),
                proposal_registration_hash=registration.content_hash,
                look_index=len(looks) + 1,
                sampling_design_hash=design.content_hash,
                allocation_rule_hash=registration.allocation_rule_hash,
                previous_allocation_receipt_hash=(
                    None
                    if prior_allocation_receipt is None
                    else prior_allocation_receipt.content_hash
                ),
                previous_context_prefix_hash=(
                    self.campaign._prefix_hash(design, ())
                    if prior_snapshot is None
                    else prior_snapshot.context_prefix_hash
                ),
                cumulative_targets=tuple(sorted(next_targets.items())),
                selected_stratum=selected_stratum,
                planned_panel_hash=panel.panel_hash,
                planned_new_context_ids=new_contexts,
                allocation_inputs_hash=content_hash(
                    {
                        "prior_targets": tuple(sorted(targets.items())),
                        "prior_observed_values": tuple(
                            (key, tuple(values))
                            for key, values in sorted(observed.items())
                        ),
                        "realized_cost_per_context": tuple(sorted(costs.items())),
                        "allocation_scores": allocation_score_receipt,
                        "unified_allocator_receipt": (unified_allocator_receipt),
                    }
                ),
                resource_preflight_hash=content_hash(
                    {
                        "ledger_before_hash": ledger_before_hash,
                        "resource_preview": resource_preview,
                        "resource_envelope": self.decision_envelope,
                        "panel_context_ids": panel.context_ids,
                        "execution_state_hashes": (
                            parent_execution_state,
                            candidate_execution_state,
                        ),
                    }
                ),
            )
            self.campaign.register_allocation_look(
                proposal_id,
                allocation_receipt,
            )
            request = GlocalSampledComparisonRequest(
                query_id=self.context_design.query_id,
                layer_id=str(layer_id),
                partition_id=str(partition_id),
                previous_density=float(previous_density),
                proposed_density=float(proposed_density),
                stage_index=len(looks),
                stage_per_class=max(1, len(panel.context_ids) // 2),
                proposal_index=proposal_index,
                random_seed=int(random_seed),
                fold_count=self.context_design.fold_count,
                prior_panel=prior_panel,
                cumulative_target_images=len(panel.context_ids),
                per_stratum_target_images=next_targets,
                maximum_new_contexts=len(new_contexts),
            )
            before = _backend_resource_snapshot(self.backend)
            started = time.perf_counter()
            evidence = self.backend.evaluate_glocal_brier_pair(
                self.current_masks,
                candidate_masks,
                request,
                panel=panel,
                parent_calibration=self.current_calibration.native_fit,
                candidate_calibration=candidate_calibration.native_fit,
            )
            runtime_receipt = self.backend.last_glocal_runtime_receipt
            if (
                runtime_receipt is None
                or runtime_receipt.operation != "evaluate_glocal_brier_pair"
                or runtime_receipt.panel_hash != panel.panel_hash
                or not runtime_receipt.succeeded
            ):
                raise RuntimeError(
                    "proposal physical evidence lacks its exact successful "
                    "backend call receipt"
                )
            self._proposal_paired_evidence[str(proposal_id)].append(evidence)
            elapsed = time.perf_counter() - started
            self._record_runtime_call(
                context_ids=panel.context_ids,
                execution_state_hashes=(
                    parent_execution_state,
                    candidate_execution_state,
                ),
                before=before,
                elapsed=elapsed,
                calibration_fit=False,
                envelope=self.decision_envelope,
            )
            increments = paired_brier_increment(
                evidence.candidate_probabilities,
                evidence.parent_probabilities,
                evidence.binary_labels,
                prediction_rule=self.context_design.risk_estimand.prediction_rule,
            )
            increment_by_context = dict(
                zip(panel.context_ids, increments.tolist(), strict=True)
            )
            for stratum in design.strata:
                prior_count = targets[stratum.stratum_id]
                new_count = next_targets[stratum.stratum_id]
                for context_id in stratum.context_ids_in_sealed_order[
                    prior_count:new_count
                ]:
                    value = float(increment_by_context[context_id])
                    process.update(
                        stratum.stratum_id,
                        value,
                        context_id=context_id,
                    )
                    observed[stratum.stratum_id].append(value)
            if new_contexts:
                per_context = elapsed / len(new_contexts)
                affected = {
                    self.context_design.strata[context_id]
                    for context_id in new_contexts
                }
                for stratum_id in affected:
                    old_count = max(targets[stratum_id], 0)
                    new_count = next_targets[stratum_id] - old_count
                    old_cost = costs[stratum_id]
                    costs[stratum_id] = (
                        old_count * old_cost + new_count * per_context
                    ) / max(next_targets[stratum_id], 1)
            targets = next_targets
            prior_panel = panel
            panel_hashes.append(panel.panel_hash)
            snapshot = process.interval()
            self._proposal_evidence_snapshots[str(proposal_id)].append(snapshot)
            final_receipt = self.campaign.adjudicate(
                proposal_id,
                snapshot,
                panel_prefix_hashes=tuple(panel_hashes),
            )
            looks.append(
                GlocalProposalLook(
                    look_index=len(looks) + 1,
                    chosen_stratum=selected_stratum,
                    cumulative_targets=tuple(sorted(targets.items())),
                    new_context_ids=new_contexts,
                    panel_hash=panel.panel_hash,
                    paired_evidence_hash=evidence.evidence_hash,
                    increment_hash=content_hash(
                        {
                            "panel_hash": panel.panel_hash,
                            "increments": np.asarray(increments, dtype=np.float64),
                        }
                    ),
                    evidence_snapshot_hash=snapshot.content_hash,
                    allocation_look_receipt_hash=(allocation_receipt.content_hash),
                    lower_bound=final_receipt.lower_bound,
                    upper_bound=final_receipt.upper_bound,
                    outcome=final_receipt.outcome,
                    allocation_scores=tuple(sorted(allocation_scores.items())),
                    realized_seconds=float(elapsed),
                    resource_ledger_hash=str(
                        self.campaign.resource_ledger.receipt()["ledger_hash"]
                    ),
                )
            )
            prior_snapshot = snapshot
            prior_allocation_receipt = allocation_receipt
            if final_receipt.outcome == "accepted":
                self.current_masks = {
                    str(key): np.asarray(value, dtype=bool).copy()
                    for key, value in candidate_masks.items()
                }
                self.current_calibration = candidate_calibration
                reason = "bounded Brier upper sequence passed both frozen boundaries"
                outcome = "accepted"
                committed = True
                break
            if final_receipt.outcome == "rejected":
                reason = "bounded Brier lower sequence crossed the frozen boundary"
                outcome = "rejected"
                committed = False
                break
            if snapshot.full_census:
                reason = (
                    "full census could not satisfy the local or global "
                    "distortion boundary"
                )
                outcome = "unresolved"
                committed = False
                break
        if (
            outcome == "unresolved"
            and self.campaign.proposal_status(proposal_id) == "registered"
        ):
            self.campaign.retire_proposal(proposal_id, outcome="abandoned")
        campaign_receipt = self.campaign.receipt()
        return self._record_proposal_result(
            GlocalProposalResult(
                proposal_id=str(proposal_id),
                outcome=outcome,
                reason=reason,
                committed=committed,
                registration_hash=registration.content_hash,
                candidate_state_hash=candidate_state_hash,
                candidate_calibration_hash=(
                    candidate_calibration.fit_state.content_hash
                ),
                decision_receipt=final_receipt,
                looks=tuple(looks),
                campaign_receipt_hash=str(campaign_receipt["campaign_hash"]),
            )
        )


def _normalize_lowered_masks(
    masks: Mapping[str, np.ndarray],
    *,
    route_count: int,
) -> dict[str, np.ndarray]:
    normalized = {
        str(layer_id): np.asarray(mask, dtype=bool).copy()
        for layer_id, mask in masks.items()
    }
    if (
        not normalized
        or any(not layer_id or mask.size <= 0 for layer_id, mask in normalized.items())
        or sum(int(mask.size) for mask in normalized.values()) != int(route_count)
    ):
        raise ValueError("native mask lowering must cover every IRG route exactly once")
    return normalized


class GlocalIRGCommitAuthority:
    """Bridge selected IRG actions into the shadow glocal-v3 campaign."""

    authority_id = EVIDENCE_FOUNDATION_GLOCAL_V3

    def __init__(
        self,
        *,
        runtime: GlocalBrierCampaignRuntime,
        mask_lowerer: Callable[[HardMask], Mapping[str, np.ndarray]],
        lowering_spec: IRGNativeMaskLoweringSpec,
        policy: GlocalIRGCommitPolicy,
        guidance_resource_envelopes: (
            Mapping[str, GlocalPacketResourceEnvelope] | None
        ) = None,
    ) -> None:
        if lowering_spec.route_axis_hash == "":
            raise ValueError("glocal IRG mask lowering requires a route axis")
        self.runtime = runtime
        self.mask_lowerer = mask_lowerer
        self.lowering_spec = lowering_spec
        self.policy = policy
        self.guidance_resource_envelopes = {
            str(kind): envelope
            for kind, envelope in dict(guidance_resource_envelopes or {}).items()
        }
        self._accepted_mask_bridges: list[IRGNativeMaskBridge] = []
        self._calibration_transition_executions: dict[
            str,
            tuple[
                CampaignCalibrationTransitionDecision,
                CalibrationUncertaintyState | None,
                object | None,
                str,
            ],
        ] = {}
        self._consumed_calibration_transition_requests: set[str] = set()
        self._calibration_logit_executions: dict[
            str,
            tuple[
                CampaignCalibrationLogitAcquisitionReceipt,
                str,
            ],
        ] = {}
        self._consumed_calibration_logit_requests: set[str] = set()
        self._issued_commit_records: dict[str, tuple[str, str, str]] = {}
        _require_physical_glocal_commit_authority(self)

    @property
    def calibration_transition_decisions(
        self,
    ) -> tuple[CampaignCalibrationTransitionDecision, ...]:
        return tuple(
            value[0] for value in self._calibration_transition_executions.values()
        )

    @property
    def calibration_logit_receipts(
        self,
    ) -> tuple[CampaignCalibrationLogitAcquisitionReceipt, ...]:
        return tuple(value[0] for value in self._calibration_logit_executions.values())

    def _require_session_calibration_lineage(
        self,
        session: PruningSessionState,
    ) -> CalibrationUncertaintyState:
        value = session.cache.get(CALIBRATION_UNCERTAINTY_CACHE_KEY)
        if not isinstance(value, CalibrationUncertaintyState):
            raise RuntimeError(
                "shadow IRG session lacks typed calibration uncertainty lineage"
            )
        expected = self.runtime.campaign.current_calibration_hash
        fit = self.runtime.current_calibration.fit_state
        support_design = self.runtime.context_design.calibration_sampling_design
        if (
            value.calibration_fit_state_hash != expected
            or fit.content_hash != expected
            or value.calibration_state_id != fit.fit_id
            or not np.array_equal(
                value.parameter_mean,
                np.asarray(fit.parameters, dtype=np.float64),
            )
            or not np.array_equal(
                value.parameter_covariance,
                np.asarray(fit.covariance, dtype=np.float64),
            )
            or value.fitted_context_count
            != len(self.runtime.current_calibration.observation_set.context_ids)
            or value.planning_context_count
            != max(
                value.fitted_context_count,
                sum(self.runtime.calibration_targets.values()),
            )
            or value.numerical_converged != fit.converged
            or (
                support_design is not None
                and (
                    value.calibration_support_design_hash != support_design.content_hash
                    or value.calibration_support_capacity
                    != len(self.runtime.context_design.calibration_context_ids)
                )
            )
        ):
            raise RuntimeError(
                "IRG calibration state differs from the campaign's committed "
                "CalibrationFitState"
            )
        active = session.version_state
        intrinsic = getattr(active, "intrinsic_state", None)
        identities = {
            str(item)
            for item in (
                getattr(active, "calibration_state_id", ""),
                getattr(intrinsic, "calibration_state_id", ""),
            )
            if str(item)
        }
        if len(identities) != 1 or value.calibration_state_id not in identities:
            raise RuntimeError(
                "IRG response chart and committed calibration fit have different "
                "lineage"
            )
        return value

    def synchronize_committed_calibration(
        self,
        *,
        session: PruningSessionState,
        decision: ExactHardMaskCommitDecision,
    ) -> None:
        """Carry an accepted candidate's exact fit into the next parent state."""

        if not decision.accepted or decision.candidate_calibration_hash is None:
            raise ValueError(
                "only an accepted exact-mask decision can advance calibration"
            )
        binding = self.runtime.current_calibration
        fit = binding.fit_state
        fit.require_claim_bearing()
        if (
            decision.candidate_calibration_hash != fit.content_hash
            or fit.content_hash != self.runtime.campaign.current_calibration_hash
        ):
            raise RuntimeError(
                "accepted calibration decision differs from the committed runtime fit"
            )
        replacement = self._calibration_uncertainty_from_current_binding()
        session.cache[CALIBRATION_UNCERTAINTY_CACHE_KEY] = replacement
        for key in tuple(session.cache):
            if str(key).startswith(CALIBRATION_CONTRASTS_CACHE_KEY):
                session.cache.pop(key, None)
        session.decision.calibration_sample_count = replacement.fitted_context_count
        session.decision.calibration_uncertainty = float(
            math.sqrt(
                max(
                    0.0,
                    float(np.max(np.diag(replacement.parameter_covariance))),
                )
            )
        )
        session.decision.record_transition(
            "carry_committed_calibration",
            "accepted exact-mask evidence advanced the next parent to its exact "
            "claim-bearing calibration fit",
            calibration_fit_state_hash=fit.content_hash,
            calibration_state_id=fit.fit_id,
            fit_observation_set_hash=fit.fit_observation_set_hash,
            algorithm_spec_hash=fit.algorithm_spec_hash,
            warm_start_parent_hash=fit.warm_start_parent_hash,
        )

    def _calibration_uncertainty_from_current_binding(
        self,
    ) -> CalibrationUncertaintyState:
        """Materialize the exact committed fit as typed allocation state."""

        binding = self.runtime.current_calibration
        fit = binding.fit_state
        fit.require_claim_bearing()
        numeric_diagnostics = {
            str(key): float(value)
            for key, value in fit.diagnostics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(float(value))
        }
        return CalibrationUncertaintyState(
            calibration_state_id=fit.fit_id,
            parameter_mean=np.asarray(fit.parameters, dtype=np.float64),
            parameter_covariance=np.asarray(fit.covariance, dtype=np.float64),
            fitted_context_count=len(binding.observation_set.context_ids),
            numerical_converged=fit.converged,
            calibration_fit_state_hash=fit.content_hash,
            cached_logit_keys=(),
            available_context_count=max(
                len(binding.observation_set.context_ids),
                sum(self.runtime.calibration_targets.values()),
            ),
            calibration_support_design_hash=(
                ""
                if self.runtime.context_design.calibration_sampling_design is None
                else self.runtime.context_design.calibration_sampling_design.content_hash
            ),
            calibration_support_capacity=len(
                self.runtime.context_design.calibration_context_ids
            ),
            diagnostics=numeric_diagnostics,
        )

    def current_calibration_uncertainty_state(
        self,
    ) -> CalibrationUncertaintyState:
        """Return the exact campaign fit/support state for IRG session seeding.

        The returned covariance is allocation-only.  Claim-bearing safety
        remains owned by the paired bounded-risk campaign.
        """

        return self._calibration_uncertainty_from_current_binding()

    @staticmethod
    def _guidance_execution_state_hashes(request: Any) -> tuple[str, ...]:
        states = [str(request.state_hash)]
        payload = dict(getattr(request, "payload", {}))
        for value in tuple(payload.get("candidate_mask_hashes", ())):
            states.append(
                content_hash(
                    {
                        "role": EvidenceRole.GUIDANCE_GEOMETRY.value,
                        "candidate_mask_hash": str(value),
                        "request_id": str(request.request_id),
                    }
                )
            )
        candidate_mask = payload.get("candidate_mask_hash")
        if candidate_mask:
            states.append(
                content_hash(
                    {
                        "role": EvidenceRole.GUIDANCE_GEOMETRY.value,
                        "candidate_mask_hash": str(candidate_mask),
                        "request_id": str(request.request_id),
                    }
                )
            )
        return tuple(dict.fromkeys(states))

    def _guidance_envelope(
        self,
        request: Any,
    ) -> GlocalPacketResourceEnvelope:
        envelope = self.guidance_resource_envelopes.get(str(request.kind))
        if envelope is not None:
            return envelope
        if self.runtime.campaign.resource_ledger.caps:
            raise ResourceBudgetUnavailable(
                "shadow IRG guidance has campaign resource caps but no frozen "
                f"conservative envelope for measurement kind {request.kind!r}"
            )
        estimate = request.expected_cost
        inflation = 1.0 + float(estimate.uncertainty_fraction)
        return GlocalPacketResourceEnvelope(
            forward_passes=int(math.ceil(inflation * estimate.forward_passes)),
            backward_passes=int(math.ceil(inflation * estimate.backward_passes)),
            calibration_fits=int(math.ceil(inflation * estimate.calibration_fits)),
            host_to_device_bytes=int(math.ceil(inflation * estimate.transferred_bytes)),
            wall_seconds=inflation * float(estimate.wall_seconds),
            device_seconds=inflation * float(estimate.device_seconds),
            peak_memory_bytes=int(math.ceil(inflation * estimate.peak_memory_bytes)),
        )

    @staticmethod
    def _preview_context_ids(request: Any) -> tuple[str, ...]:
        explicit = tuple(request.context_selector.context_ids)
        if explicit:
            return explicit
        return tuple(
            f"preflight:{request.request_id}:{index}"
            for index in range(max(1, int(request.context_selector.requested_count)))
        )

    def supports_guidance_measurement(self, request: Any) -> bool:
        """Whether the campaign authority, rather than a provider, owns a kind."""

        return str(request.kind) in {"calibration_logits", "calibration_refit"}

    def guidance_cost_envelope(self, request: Any) -> CostEstimate:
        """Return a conservative runner-local cap for an authority-owned call."""

        kind = str(request.kind)
        if kind not in {"calibration_logits", "calibration_refit"}:
            return request.expected_cost
        if kind == "calibration_logits":
            calibration = self.runtime.calibration_envelope
            return CostEstimate(
                wall_seconds=calibration.wall_seconds,
                device_seconds=calibration.device_seconds,
                forward_passes=float(calibration.forward_passes),
                backward_passes=float(calibration.backward_passes),
                calibration_fits=0.0,
                transferred_bytes=float(
                    calibration.host_to_device_bytes + calibration.device_to_host_bytes
                ),
                peak_memory_bytes=float(calibration.peak_memory_bytes),
            )
        population_size = len(self.runtime.context_design.adjudication_context_ids)
        maximum_looks = (
            population_size
            if self.runtime.policy.maximum_looks is None
            else min(population_size, int(self.runtime.policy.maximum_looks))
        )
        calibration = self.runtime.calibration_envelope
        decision = self.runtime.decision_envelope
        return CostEstimate(
            wall_seconds=(
                calibration.wall_seconds + maximum_looks * decision.wall_seconds
            ),
            device_seconds=(
                calibration.device_seconds + maximum_looks * decision.device_seconds
            ),
            forward_passes=float(
                calibration.forward_passes + maximum_looks * decision.forward_passes
            ),
            backward_passes=float(
                calibration.backward_passes + maximum_looks * decision.backward_passes
            ),
            calibration_fits=float(
                calibration.calibration_fits + maximum_looks * decision.calibration_fits
            ),
            transferred_bytes=float(
                calibration.host_to_device_bytes
                + calibration.device_to_host_bytes
                + maximum_looks
                * (decision.host_to_device_bytes + decision.device_to_host_bytes)
            ),
            peak_memory_bytes=float(
                max(
                    calibration.peak_memory_bytes,
                    decision.peak_memory_bytes,
                )
            ),
        )

    def guidance_measurement_eligibility(
        self,
        request: Any,
    ) -> tuple[bool, str]:
        """Check the live campaign-owned calibration transition before selection."""

        kind = str(request.kind)
        if kind not in {"calibration_logits", "calibration_refit"}:
            return True, ""
        if self.runtime.campaign.terminal_lock is not None:
            return (
                False,
                "campaign is terminally locked; calibration state cannot change",
            )
        selector = getattr(request, "context_selector", None)
        if selector is not None and tuple(getattr(selector, "context_ids", ())):
            return (
                False,
                "campaign calibration transition owns its frozen fit and "
                "adjudication populations; caller-selected contexts are illegal",
            )
        request_id = str(getattr(request, "request_id", ""))
        executions = (
            self._calibration_logit_executions
            if kind == "calibration_logits"
            else self._calibration_transition_executions
        )
        if request_id in executions:
            return (
                False,
                "this predictable calibration request was already physically "
                "executed",
            )
        payload = dict(getattr(request, "payload", {}))
        current_fit_count = len(
            self.runtime.current_calibration.observation_set.context_ids
        )
        available_count = sum(self.runtime.calibration_targets.values())
        target_count = int(
            payload.get(
                "calibration_target_context_count",
                current_fit_count + int(getattr(selector, "requested_count", 0)),
            )
        )
        declared_fit_count = int(
            payload.get("calibration_current_fit_context_count", current_fit_count)
        )
        declared_available = int(
            payload.get("calibration_available_context_count", available_count)
        )
        design = self.runtime.context_design.calibration_sampling_design
        declared_design_hash = str(payload.get("calibration_support_design_hash", ""))
        declared_capacity = int(
            payload.get(
                "calibration_support_capacity",
                len(self.runtime.context_design.calibration_context_ids),
            )
        )
        if selector is not None:
            selector_parameters = dict(getattr(selector, "parameters", {}))
            selector_target = int(
                selector_parameters.get(
                    "calibration_target_context_count",
                    target_count,
                )
            )
            expected_increment = (
                target_count - available_count
                if kind == "calibration_logits"
                else target_count - current_fit_count
            )
            if (
                str(getattr(selector, "selector_id", ""))
                != "irg_calibration_active_contrast_selector_v1"
                or selector_target != target_count
                or int(getattr(selector, "requested_count", -1))
                != max(0, expected_increment)
                or selector_parameters.get("fit_fold_only") is not True
                or selector_parameters.get("terminal_audit_visible") is not False
            ):
                return (
                    False,
                    "calibration request selector differs from the predictable "
                    "active-contrast acquisition contract",
                )
        provenance = getattr(request, "provenance", None)
        if provenance is not None and (
            str(getattr(provenance, "producer_id", ""))
            != "irg_calibration_measurement_advisor"
            or not str(getattr(provenance, "producer_version", ""))
            or not str(getattr(provenance, "code_hash", ""))
        ):
            return (
                False,
                "calibration request provenance is not owned by the canonical "
                "measurement advisor",
            )
        if (
            declared_fit_count != current_fit_count
            or declared_available != available_count
            or declared_capacity
            != len(self.runtime.context_design.calibration_context_ids)
            or (design is not None and declared_design_hash != design.content_hash)
        ):
            return (
                False,
                "calibration request differs from the live frozen fit/support "
                "prefix",
            )
        if target_count < max(current_fit_count, available_count):
            return (
                False,
                "calibration request would regress the admitted fit-role prefix",
            )
        if design is not None and target_count > len(
            self.runtime.context_design.calibration_context_ids
        ):
            return (
                False,
                "calibration request exceeds the frozen support capacity",
            )
        if kind == "calibration_logits":
            if design is None:
                return (
                    False,
                    "static calibration census has no separate logit-prefix "
                    "acquisition",
                )
            planned = self.runtime.calibration_targets_for_total(target_count)
            if planned == dict(self.runtime.calibration_targets):
                return (
                    False,
                    "calibration support is exhausted or the request adds no "
                    "new prefix rows",
                )
        elif design is not None:
            planned = self.runtime.calibration_targets_for_total(target_count)
            if planned != dict(self.runtime.calibration_targets):
                return (
                    False,
                    "adaptive calibration refit requires the exact sealed logit "
                    "prefix to be physically admitted first",
                )
        return True, ""

    def preflight_guidance_measurement(self, request: Any) -> None:
        """Charge every IRG guidance call against the campaign before launch."""

        if str(request.kind) in {"calibration_logits", "calibration_refit"}:
            allowed, reason = self.guidance_measurement_eligibility(request)
            if not allowed:
                raise RuntimeError(reason)
            # The campaign runtime preflights the calibration packet and every
            # paired adjudication packet against its own frozen envelopes.
            # Treating those contexts as generic guidance would both
            # double-charge them and assign the wrong statistical role.
            return
        envelope = self._guidance_envelope(request)
        preview, _, _ = (
            self.runtime.campaign.resource_ledger.preview_context_state_evaluations(
                context_ids=self._preview_context_ids(request),
                state_hashes=self._guidance_execution_state_hashes(request),
                forward_passes=envelope.forward_passes,
                backward_passes=envelope.backward_passes,
                calibration_fits=envelope.calibration_fits,
                host_to_device_bytes=envelope.host_to_device_bytes,
                device_to_host_bytes=envelope.device_to_host_bytes,
                wall_seconds=envelope.wall_seconds,
                device_seconds=envelope.device_seconds,
                peak_memory_bytes=envelope.peak_memory_bytes,
            )
        )
        if not self.runtime.campaign.resource_ledger.can_consume(preview):
            raise ResourceBudgetUnavailable(
                "shadow IRG guidance would exceed the campaign-global resource cap"
            )

    @staticmethod
    def _realized_cost_between(
        before: ResourceUsage,
        after: ResourceUsage,
    ) -> RealizedCost:
        """Project one campaign-ledger delta onto the runner-local cost type."""

        return RealizedCost(
            wall_seconds=max(0.0, after.wall_seconds - before.wall_seconds),
            device_seconds=max(0.0, after.device_seconds - before.device_seconds),
            forward_passes=max(0, after.forward_passes - before.forward_passes),
            backward_passes=max(0, after.backward_passes - before.backward_passes),
            calibration_fits=max(
                0,
                after.calibration_fits - before.calibration_fits,
            ),
            transferred_bytes=max(
                0,
                (
                    after.host_to_device_bytes
                    + after.device_to_host_bytes
                    - before.host_to_device_bytes
                    - before.device_to_host_bytes
                ),
            ),
            peak_memory_bytes=max(
                0,
                after.peak_memory_bytes - before.peak_memory_bytes,
            ),
        )

    def execute_guidance_measurement(
        self,
        *,
        session: PruningSessionState,
        request: Any,
    ) -> MeasurementOutcome | None:
        """Execute a campaign-certified state transition owned by this authority."""

        if str(request.kind) not in {"calibration_logits", "calibration_refit"}:
            return None
        allowed, reason = self.guidance_measurement_eligibility(request)
        if not allowed:
            raise RuntimeError(reason)
        session.assert_request_state(request.state_hash)
        self._require_session_calibration_lineage(session)
        local_remaining = session.compute_ledger.remaining_wall_seconds
        local_envelope = self.guidance_cost_envelope(request)
        if local_remaining is not None and local_envelope.wall_seconds <= 0.0:
            raise RuntimeError(
                "finite runner wall budget requires a positive frozen wall-time "
                "envelope for campaign calibration transitions"
            )
        if sampled_path_mask_hash(self.runtime.current_masks) != sampled_path_mask_hash(
            _normalize_lowered_masks(
                self.mask_lowerer(session.current_mask),
                route_count=session.current_mask.route_count,
            )
        ):
            raise RuntimeError(
                "calibration transition IRG and campaign masks are not synchronized"
            )

        if str(request.kind) == "calibration_logits":
            target_context_count = int(
                request.payload["calibration_target_context_count"]
            )
            resource_before = self.runtime.campaign.resource_ledger.usage
            (
                rows,
                observation_set,
                new_context_ids,
                before_targets,
                after_targets,
                runtime_receipt_hash,
                resource_before_hash,
                resource_after_hash,
            ) = self.runtime.acquire_calibration_logits(
                target_context_count=target_context_count
            )
            resource_after = self.runtime.campaign.resource_ledger.usage
            design = self.runtime.context_design.calibration_sampling_design
            if design is None:
                raise RuntimeError(
                    "campaign logit acquisition lost its adaptive support design"
                )
            receipt = CampaignCalibrationLogitAcquisitionReceipt(
                request_id=str(request.request_id),
                irg_state_hash=session.state_hash,
                irg_mask_hash=session.current_mask.mask_hash,
                native_mask_hash=sampled_path_mask_hash(self.runtime.current_masks),
                calibration_fit_state_hash=(
                    self.runtime.current_calibration.fit_state.content_hash
                ),
                calibration_sampling_design_hash=design.content_hash,
                before_targets=before_targets,
                after_targets=after_targets,
                observation_set_hash=observation_set.content_hash,
                panel_hash=rows.panel.panel_hash,
                row_evidence_hash=rows.evidence_hash,
                new_context_ids=new_context_ids,
                backend_runtime_receipt_hash=runtime_receipt_hash,
                resource_ledger_before_hash=resource_before_hash,
                resource_ledger_after_hash=resource_after_hash,
            )
            row_index = {
                context_id: index
                for index, context_id in enumerate(rows.panel.context_ids)
            }
            values = np.asarray(
                [
                    (
                        float(rows.logits[row_index[context_id]]),
                        float(rows.binary_labels[row_index[context_id]]),
                    )
                    for context_id in new_context_ids
                ],
                dtype=np.float64,
            )
            outcome = MeasurementOutcome.create(
                request_id=request.request_id,
                state_hash=request.state_hash,
                context_ids=new_context_ids,
                candidate_ids=request.candidate_ids,
                values=values,
                calibration_state_id=(
                    self.runtime.current_calibration.fit_state.fit_id
                ),
                realized_cost=self._realized_cost_between(
                    resource_before,
                    resource_after,
                ),
                metadata={
                    "kind": "calibration_logits",
                    "execution_role": "campaign_calibration_fit_prefix",
                    "campaign_calibration_logit_receipt_hash": (receipt.content_hash),
                    "calibration_sampling_design_hash": design.content_hash,
                    "calibration_observation_set_hash": (observation_set.content_hash),
                    "calibration_row_evidence_hash": rows.evidence_hash,
                    "calibration_prefix_targets_before": before_targets,
                    "calibration_prefix_targets_after": after_targets,
                    "claim_bound_narrowing": False,
                },
            )
            validate_outcome_for_request(request, outcome)
            self._calibration_logit_executions[str(request.request_id)] = (
                receipt,
                outcome.integrity_hash,
            )
            return outcome

        # Validate the fail-closed response-state replacement before a campaign
        # acceptance can mutate the statistical parent.
        prepared_version_state = CommonIRGRunner._invalidate_response_state_for_refresh(
            session.version_state,
            current_mask_hash=session.current_mask.mask_hash,
        )
        parent_session_hash = session.state_hash
        parent_mask_hash = session.current_mask.mask_hash
        native_mask_hash = sampled_path_mask_hash(self.runtime.current_masks)
        campaign_parent_hash = self.runtime.campaign.current_state_hash
        parent_calibration_hash = self.runtime.campaign.current_calibration_hash
        current_fit_count = len(
            self.runtime.current_calibration.observation_set.context_ids
        )
        target_context_count = int(
            request.payload.get(
                "calibration_target_context_count",
                current_fit_count
                + max(0, int(request.context_selector.requested_count)),
            )
        )
        requested_context_increment = max(
            0,
            target_context_count - current_fit_count,
        )
        planned_calibration_targets = self.runtime.calibration_targets_for_total(
            target_context_count
        )
        planned_calibration_set = (
            self.runtime.context_design.calibration_observation_set_for_targets(
                planned_calibration_targets
            )
        )
        candidate_definition_hash = content_hash(
            {
                "transition_kind": "calibration_only",
                "request_id": request.request_id,
                "request_cache_key": request.cache_key,
                "request_state_hash": request.state_hash,
                "contrast_snapshot_hash": request.payload.get(
                    "calibration_contrast_snapshot_hash"
                ),
                "irg_mask_hash": parent_mask_hash,
                "native_mask_hash": native_mask_hash,
                "campaign_parent_state_hash": campaign_parent_hash,
                "parent_calibration_hash": parent_calibration_hash,
                "planned_calibration_targets": tuple(
                    sorted(planned_calibration_targets.items())
                ),
                "planned_calibration_observation_set_hash": (
                    planned_calibration_set.content_hash
                ),
                "requested_calibration_context_increment": (
                    requested_context_increment
                ),
                "commit_policy_hash": self.policy.content_hash,
            }
        )
        proposal_id = f"irg-calibration-transition:{request.request_id}"
        seed_digest = content_hash(
            {
                "purpose": "calibration_transition_sampling_seed",
                "request_cache_key": request.cache_key,
                "commit_policy_hash": self.policy.content_hash,
            }
        )
        random_seed = int(self.policy.base_random_seed) + (
            int(seed_digest[:16], 16) % (2**31 - 1)
        )
        resource_before = self.runtime.campaign.resource_ledger.usage
        resource_before_hash = str(
            self.runtime.campaign.resource_ledger.receipt()["ledger_hash"]
        )
        result = self.runtime.run_calibration_transition(
            proposal_id=proposal_id,
            candidate_definition_hash=candidate_definition_hash,
            local_upper_boundary=(self.policy.calibration_transition_upper_boundary),
            group_id=f"{self.runtime.context_design.query_id}:calibration-state",
            random_seed=random_seed,
            requested_context_increment=max(
                0,
                sum(planned_calibration_targets.values())
                - sum(self.runtime.calibration_targets.values()),
            ),
        )
        fitted_pair = self.runtime._proposal_calibrations.get(proposal_id)
        if result.candidate_calibration_hash is not None:
            if fitted_pair is None:
                raise RuntimeError(
                    "calibration transition lost its candidate fit binding"
                )
            fitted_candidate = fitted_pair[1]
            if (
                fitted_candidate.observation_set.content_hash
                != planned_calibration_set.content_hash
                or fitted_candidate.observation_set.context_ids
                != planned_calibration_set.context_ids
            ):
                raise RuntimeError(
                    "calibration transition fitted a different physical prefix "
                    "than the predictable request"
                )
        resource_after = self.runtime.campaign.resource_ledger.usage
        resource_after_hash = str(
            self.runtime.campaign.resource_ledger.receipt()["ledger_hash"]
        )
        evidence_receipt = result.decision_receipt
        decision = CampaignCalibrationTransitionDecision(
            request_id=str(request.request_id),
            request_cache_key_hash=content_hash(
                {"request_cache_key": str(request.cache_key)}
            ),
            irg_parent_state_hash=parent_session_hash,
            irg_mask_hash=parent_mask_hash,
            native_mask_hash=native_mask_hash,
            candidate_native_mask_hash=sampled_path_mask_hash(
                self.runtime.current_masks
            ),
            campaign_parent_state_hash=campaign_parent_hash,
            campaign_candidate_state_hash=result.candidate_state_hash,
            parent_calibration_hash=parent_calibration_hash,
            candidate_calibration_hash=result.candidate_calibration_hash,
            calibration_fit_observation_set_hash=(planned_calibration_set.content_hash),
            calibration_prefix_targets=tuple(
                sorted(planned_calibration_targets.items())
            ),
            requested_context_increment=requested_context_increment,
            candidate_definition_hash=candidate_definition_hash,
            policy_hash=self.policy.content_hash,
            proposal_id=proposal_id,
            proposal_result_hash=result.content_hash,
            registration_hash=result.registration_hash,
            decision_evidence_receipt_hash=(
                None if evidence_receipt is None else evidence_receipt.content_hash
            ),
            campaign_receipt_hash=result.campaign_receipt_hash,
            resource_ledger_before_hash=resource_before_hash,
            resource_ledger_after_hash=resource_after_hash,
            local_upper_boundary=(self.policy.calibration_transition_upper_boundary),
            distortion_debit=(
                0.0 if evidence_receipt is None else evidence_receipt.distortion_debit
            ),
            outcome=result.outcome,
            accepted=result.committed,
        )
        replacement = (
            self._calibration_uncertainty_from_current_binding()
            if decision.accepted
            else None
        )
        if sampled_path_mask_hash(self.runtime.current_masks) != native_mask_hash:
            raise RuntimeError(
                "calibration-only campaign proposal changed the physical mask"
            )
        if decision.accepted and (
            self.runtime.campaign.current_state_hash
            != decision.campaign_candidate_state_hash
            or self.runtime.campaign.current_calibration_hash
            != decision.candidate_calibration_hash
        ):
            raise RuntimeError(
                "accepted calibration transition did not become the campaign parent"
            )
        if not decision.accepted and (
            self.runtime.campaign.current_state_hash != campaign_parent_hash
            or self.runtime.campaign.current_calibration_hash != parent_calibration_hash
        ):
            raise RuntimeError(
                "nonaccepted calibration transition mutated the campaign parent"
            )
        lower = 0.0 if evidence_receipt is None else evidence_receipt.lower_bound
        upper = 0.0 if evidence_receipt is None else evidence_receipt.upper_bound
        calibration_state_id = self.runtime.current_calibration.fit_state.fit_id
        physical_context_ids = tuple(
            dict.fromkeys(
                (
                    *planned_calibration_set.context_ids,
                    *(
                        context_id
                        for look in result.looks
                        for context_id in look.new_context_ids
                    ),
                )
            )
        )
        if not physical_context_ids:
            raise RuntimeError(
                "campaign calibration transition retained no physical contexts"
            )
        summary_row = np.asarray(
            (
                float(decision.accepted),
                float(lower),
                float(upper),
                float(decision.distortion_debit),
            ),
            dtype=np.float64,
        )
        outcome = MeasurementOutcome.create(
            request_id=request.request_id,
            state_hash=request.state_hash,
            context_ids=physical_context_ids,
            candidate_ids=request.candidate_ids,
            values=np.repeat(
                summary_row[None, :],
                len(physical_context_ids),
                axis=0,
            ),
            calibration_state_id=calibration_state_id,
            realized_cost=self._realized_cost_between(
                resource_before,
                resource_after,
            ),
            metadata={
                "kind": "calibration_refit",
                "execution_role": "campaign_calibration_state_transition",
                "campaign_calibration_transition_decision_hash": (
                    decision.content_hash
                ),
                "campaign_proposal_result_hash": result.content_hash,
                "campaign_outcome": decision.outcome,
                "campaign_accepted": decision.accepted,
                "native_mask_hash": decision.native_mask_hash,
                "candidate_native_mask_hash": (decision.candidate_native_mask_hash),
                "candidate_calibration_hash": (decision.candidate_calibration_hash),
                "calibration_prefix_targets": tuple(
                    sorted(planned_calibration_targets.items())
                ),
                "calibration_fit_observation_set_hash": (
                    planned_calibration_set.content_hash
                ),
                "campaign_receipt_hash": decision.campaign_receipt_hash,
            },
        )
        validate_outcome_for_request(request, outcome)
        self._calibration_transition_executions[str(request.request_id)] = (
            decision,
            replacement,
            prepared_version_state if decision.accepted else None,
            outcome.integrity_hash,
        )
        return outcome

    def consume_certified_guidance_transition(
        self,
        *,
        session: PruningSessionState,
        request: Any,
        outcome: Any,
    ) -> bool:
        """Atomically install only an accepted campaign calibration verdict."""

        kind = str(request.kind)
        if kind == "calibration_logits":
            validate_outcome_for_request(request, outcome)
            request_id = str(request.request_id)
            if request_id in self._consumed_calibration_logit_requests:
                raise RuntimeError(
                    "campaign calibration-logit outcome cannot be consumed twice"
                )
            try:
                receipt, issued_outcome_hash = self._calibration_logit_executions[
                    request_id
                ]
            except KeyError as error:
                raise RuntimeError(
                    "shadow calibration logits lack authority-owned physical "
                    "execution"
                ) from error
            previous = session.cache.get(CALIBRATION_UNCERTAINTY_CACHE_KEY)
            if not isinstance(previous, CalibrationUncertaintyState):
                raise RuntimeError(
                    "calibration-logit outcome lost its typed parent fit"
                )
            if (
                outcome.integrity_hash != issued_outcome_hash
                or outcome.metadata.get("campaign_calibration_logit_receipt_hash")
                != receipt.content_hash
                or session.state_hash != receipt.irg_state_hash
                or session.current_mask.mask_hash != receipt.irg_mask_hash
                or previous.calibration_fit_state_hash
                != receipt.calibration_fit_state_hash
                or tuple(outcome.context_ids) != receipt.new_context_ids
                or tuple(sorted(self.runtime.calibration_targets.items()))
                != receipt.after_targets
                or not any(
                    rows.evidence_hash == receipt.row_evidence_hash
                    for rows in self.runtime.calibration_logit_rows
                )
            ):
                raise RuntimeError(
                    "calibration-logit outcome differs from its retained "
                    "campaign evidence"
                )
            updated = replace(
                previous,
                cached_logit_keys=tuple(
                    dict.fromkeys(
                        (
                            *previous.cached_logit_keys,
                            receipt.row_evidence_hash,
                            receipt.content_hash,
                        )
                    )
                ),
                available_context_count=sum(
                    target for _, target in receipt.after_targets
                ),
                calibration_support_design_hash=(
                    receipt.calibration_sampling_design_hash
                ),
                calibration_support_capacity=len(
                    self.runtime.context_design.calibration_context_ids
                ),
            )
            if updated.inferential_hash != previous.inferential_hash:
                raise RuntimeError(
                    "calibration-logit prefix changed the fitted calibration "
                    "distribution before a certified refit"
                )
            session.cache[CALIBRATION_UNCERTAINTY_CACHE_KEY] = updated
            session.cache[f"irg_campaign_calibration_logits:{receipt.content_hash}"] = (
                receipt
            )
            self._consumed_calibration_logit_requests.add(request_id)
            session.decision.record_transition(
                "campaign_calibration_prefix_growth",
                "campaign-owned sealed calibration-role rows became available "
                "to the next exact refit without changing the active fit",
                request_id=request_id,
                calibration_logit_receipt_hash=receipt.content_hash,
                row_evidence_hash=receipt.row_evidence_hash,
                before_targets=receipt.before_targets,
                after_targets=receipt.after_targets,
                observation_set_hash=receipt.observation_set_hash,
            )
            return True
        if kind != "calibration_refit":
            return False
        validate_outcome_for_request(request, outcome)
        request_id = str(request.request_id)
        if request_id in self._consumed_calibration_transition_requests:
            raise RuntimeError(
                "campaign calibration transition outcome cannot be consumed twice"
            )
        try:
            decision, replacement, prepared_version_state, issued_outcome_hash = (
                self._calibration_transition_executions[request_id]
            )
        except KeyError as error:
            raise RuntimeError(
                "shadow calibration refit lacks authority-owned campaign execution"
            ) from error
        if (
            outcome.integrity_hash != issued_outcome_hash
            or outcome.calibration_state_id
            != (
                replacement.calibration_state_id
                if replacement is not None
                else self.runtime.current_calibration.fit_state.fit_id
            )
            or outcome.metadata.get("campaign_calibration_transition_decision_hash")
            != decision.content_hash
            or outcome.metadata.get("campaign_proposal_result_hash")
            != decision.proposal_result_hash
        ):
            raise RuntimeError(
                "calibration transition outcome differs from its campaign verdict"
            )
        if session.state_hash != decision.irg_parent_state_hash:
            raise RuntimeError(
                "calibration transition cannot install into a changed IRG parent"
            )
        if session.current_mask.mask_hash != decision.irg_mask_hash:
            raise RuntimeError(
                "calibration transition cannot alter or retarget the IRG mask"
            )
        cache_key = "irg_campaign_calibration_transition:" f"{decision.content_hash}"
        if not decision.accepted:
            if (
                self.runtime.campaign.current_state_hash
                != decision.campaign_parent_state_hash
                or self.runtime.campaign.current_calibration_hash
                != decision.parent_calibration_hash
            ):
                raise RuntimeError(
                    "nonaccepted calibration verdict no longer has its parent"
                )
            session.cache[cache_key] = decision
            self._consumed_calibration_transition_requests.add(request_id)
            session.decision.record_transition(
                "calibration_transition_not_committed",
                "paired campaign evidence did not authorize the proposed "
                "calibration-only state change",
                request_id=request_id,
                outcome=decision.outcome,
                calibration_transition_decision_hash=decision.content_hash,
                proposal_result_hash=decision.proposal_result_hash,
                campaign_receipt_hash=decision.campaign_receipt_hash,
            )
            return True

        if replacement is None or prepared_version_state is None:
            raise RuntimeError(
                "accepted calibration transition lacks its prepared IRG replacement"
            )
        if (
            self.runtime.campaign.current_state_hash
            != decision.campaign_candidate_state_hash
            or self.runtime.campaign.current_calibration_hash
            != decision.candidate_calibration_hash
            or replacement.calibration_fit_state_hash
            != decision.candidate_calibration_hash
        ):
            raise RuntimeError(
                "accepted calibration transition is stale against the campaign"
            )
        prior_session_hash = session.state_hash
        session.version_state = prepared_version_state
        session.cache[CALIBRATION_UNCERTAINTY_CACHE_KEY] = replacement
        for key in tuple(session.cache):
            if str(key).startswith(CALIBRATION_CONTRASTS_CACHE_KEY):
                session.cache.pop(key, None)
        session.cache.pop(RESPONSE_TRANSITION_CACHE_KEY, None)
        session.decision.active_candidate_ids.clear()
        session.decision.practical_top_set.clear()
        session.decision.active_contrasts.clear()
        session.decision.acquisition_queue.clear()
        session.decision.pending_model_validity_action = "refresh"
        session.decision.recommended_action_scale = 0.0
        session.decision.requested_capability_transition = None
        session.decision.geometry_anchor_mask_hash = None
        session.decision.geometry_current_mask_hash = None
        session.decision.cumulative_geometry_removed_indices.clear()
        session.decision.geometry_transition_receipt_hashes.clear()
        session.decision.calibration_sample_count = replacement.fitted_context_count
        session.decision.calibration_uncertainty = float(
            math.sqrt(
                max(
                    0.0,
                    float(np.max(np.diag(replacement.parameter_covariance))),
                )
            )
        )
        replacement_session_hash = session.state_hash
        if replacement_session_hash == prior_session_hash:
            raise RuntimeError(
                "accepted calibration transition failed to retire the parent chart"
            )
        session.cache[cache_key] = decision
        self._consumed_calibration_transition_requests.add(request_id)
        session.decision.record_transition(
            "campaign_certified_calibration_transition",
            "accepted paired Brier evidence installed the same-mask calibration "
            "fit and forced response-geometry refresh",
            request_id=request_id,
            prior_session_state_hash=prior_session_hash,
            replacement_session_state_hash=replacement_session_hash,
            irg_mask_hash=decision.irg_mask_hash,
            native_mask_hash=decision.native_mask_hash,
            parent_calibration_hash=decision.parent_calibration_hash,
            candidate_calibration_hash=decision.candidate_calibration_hash,
            distortion_debit=decision.distortion_debit,
            calibration_transition_decision_hash=decision.content_hash,
            proposal_result_hash=decision.proposal_result_hash,
            campaign_receipt_hash=decision.campaign_receipt_hash,
        )
        return True

    def register_guidance_measurement(
        self,
        *,
        request: Any,
        outcome: Any,
    ) -> ObservationSetRef:
        """Register realized guidance role and compute in the global campaign."""

        if str(request.kind) in {"calibration_logits", "calibration_refit"}:
            raise RuntimeError(
                "campaign calibration evidence cannot be relabeled or "
                "double-charged as generic guidance"
            )
        envelope = self._guidance_envelope(request)
        realized = outcome.realized_cost
        if not request.context_selector.context_ids and len(outcome.context_ids) > max(
            1, int(request.context_selector.requested_count)
        ):
            raise RuntimeError(
                "IRG guidance provider exceeded its preflight context count"
            )
        comparisons = {
            "forward_passes": (
                int(realized.forward_passes),
                envelope.forward_passes,
            ),
            "backward_passes": (
                int(realized.backward_passes),
                envelope.backward_passes,
            ),
            "calibration_fits": (
                int(realized.calibration_fits),
                envelope.calibration_fits,
            ),
            "host_to_device_bytes": (
                int(realized.transferred_bytes),
                envelope.host_to_device_bytes,
            ),
            "wall_seconds": (
                float(realized.wall_seconds),
                envelope.wall_seconds,
            ),
            "device_seconds": (
                float(realized.device_seconds),
                envelope.device_seconds,
            ),
            "peak_memory_bytes": (
                int(realized.peak_memory_bytes),
                envelope.peak_memory_bytes,
            ),
        }
        capped = self.runtime.campaign.resource_ledger.caps
        underestimated = tuple(
            name
            for name, (actual, maximum) in comparisons.items()
            if name in capped and float(actual) > float(maximum) + 1e-12
        )
        if underestimated:
            raise RuntimeError(
                "realized IRG guidance exceeded its frozen campaign envelope: "
                + ",".join(underestimated)
            )
        observation = self.register_guidance_observation(
            observation_id=f"irg-guidance-request:{request.request_id}",
            context_ids=tuple(outcome.context_ids),
            sampling_design_hash=content_hash(
                {
                    "role": EvidenceRole.GUIDANCE_GEOMETRY.value,
                    "request_id": request.request_id,
                    "request_kind": request.kind,
                    "context_selector": request.context_selector,
                }
            ),
        )
        self.runtime.campaign.resource_ledger.record_context_state_evaluations(
            context_ids=tuple(outcome.context_ids),
            state_hashes=self._guidance_execution_state_hashes(request),
            forward_passes=int(realized.forward_passes),
            backward_passes=int(realized.backward_passes),
            calibration_fits=int(realized.calibration_fits),
            host_to_device_bytes=int(realized.transferred_bytes),
            wall_seconds=float(realized.wall_seconds),
            device_seconds=float(realized.device_seconds),
            peak_memory_bytes=int(realized.peak_memory_bytes),
            cache_hits=(len(outcome.context_ids) if realized.cache_hit else 0),
        )
        return observation

    def register_guidance_observation(
        self,
        *,
        observation_id: str,
        context_ids: Sequence[str],
        sampling_design_hash: str,
    ) -> ObservationSetRef:
        """Register pre-proposal guidance so claim/terminal overlap fails closed."""

        _require_sha256(sampling_design_hash, "guidance sampling design hash")
        contexts = tuple(str(value) for value in context_ids)
        forbidden = set(self.runtime.context_design.adjudication_context_ids)
        overlap = forbidden.intersection(contexts)
        if overlap:
            raise ValueError(
                "IRG guidance contexts overlap the glocal proposal "
                "adjudication population: " + ",".join(sorted(overlap))
            )
        identity = str(observation_id)
        existing = {
            item.observation_set_id: item
            for item in self.runtime.campaign.observation_sets
        }
        stratum_by_context = self.runtime.context_design.strata
        guidance = ObservationSetRef(
            observation_set_id=identity,
            role=EvidenceRole.GUIDANCE_GEOMETRY,
            sampling_design_hash=sampling_design_hash,
            context_ids=contexts,
            stratum_ids=tuple(
                stratum_by_context.get(context_id, "external-guidance")
                for context_id in contexts
            ),
        )
        prior = existing.get(identity)
        if prior is not None:
            if prior.content_hash != guidance.content_hash:
                raise ValueError(
                    "guidance observation ID was replayed with different content"
                )
            return prior
        return self.runtime.campaign.register_observation_set(
            guidance,
            allow_nonclaim_guidance_reuse=True,
        )

    def _register_irg_guidance_lineage(
        self,
        *,
        session: PruningSessionState,
        candidate: ActionCandidate,
    ) -> None:
        entry = session.frontier.entry(candidate.candidate_id)
        for receipt in entry.oracle_receipts:
            observation_set_id = (
                f"irg-guidance:{candidate.candidate_id}:{receipt.receipt_hash}"
            )
            self.register_guidance_observation(
                observation_id=observation_set_id,
                context_ids=receipt.context_ids,
                sampling_design_hash=content_hash(
                    {
                        "role": EvidenceRole.GUIDANCE_GEOMETRY.value,
                        "irg_oracle_receipt_hash": receipt.receipt_hash,
                        "candidate_id": candidate.candidate_id,
                    }
                ),
            )

    def _revalidate_commit_authorization(
        self,
        *,
        runtime_owner: object,
        campaign_owner: object,
        session: PruningSessionState,
        candidate: ActionCandidate,
        decision: ExactHardMaskCommitDecision,
    ) -> None:
        """Bind one detached verdict back to its retained live campaign graph."""

        if (
            runtime_owner is not self.runtime
            or campaign_owner is not self.runtime.campaign
        ):
            raise RuntimeError(
                "exact-mask commit authorization changed runtime or campaign owner"
            )
        if session.commit_authority_mode != "glocal_v3_shadow":
            raise ValueError(
                "exact-mask commit authorization requires a shadow glocal-v3 session"
            )
        decision.validate_transition(
            candidate_id=candidate.candidate_id,
            irg_parent_state_hash=candidate.parent_state_hash,
            parent_mask_hash=session.current_mask.mask_hash,
            candidate_mask_hash=candidate.resulting_mask.mask_hash,
        )
        if (
            decision.authority_id != self.authority_id
            or session.current_mask.mask_hash != candidate.parent_mask_hash
            or session.state_hash != candidate.parent_state_hash
        ):
            raise ValueError(
                "exact-mask commit decision is stale or belongs to another authority"
            )
        issued = self._issued_commit_records.get(decision.content_hash)
        if issued is None:
            raise RuntimeError(
                "exact-mask commit decision is absent from the issuing authority"
            )
        retained_results = tuple(
            result
            for result in self.runtime._proposal_results
            if result.proposal_id == decision.proposal_id
        )
        if len(retained_results) != 1:
            raise RuntimeError(
                "exact-mask commit decision lacks one unique retained proposal result"
            )
        result = retained_results[0]
        bridge_hash = str(decision.metadata.get("native_mask_bridge_hash", ""))
        if issued != (
            result.content_hash,
            bridge_hash,
            decision.campaign_receipt_hash,
        ):
            raise RuntimeError(
                "exact-mask commit decision differs from its live issuance record"
            )
        evidence_hash = (
            None
            if result.decision_receipt is None
            else result.decision_receipt.content_hash
        )
        if (
            result.outcome != decision.outcome
            or result.reason != decision.reason
            or result.registration_hash != decision.registration_hash
            or result.candidate_state_hash != decision.statistical_candidate_state_hash
            or result.candidate_calibration_hash != decision.candidate_calibration_hash
            or evidence_hash != decision.decision_evidence_receipt_hash
            or result.campaign_receipt_hash != decision.campaign_receipt_hash
        ):
            raise RuntimeError(
                "exact-mask commit decision differs from its retained physical result"
            )
        campaign_receipt = self.runtime.campaign.receipt()
        if str(campaign_receipt["campaign_hash"]) != decision.campaign_receipt_hash:
            raise RuntimeError(
                "exact-mask commit decision no longer matches the live campaign receipt"
            )
        registrations = {
            str(item["registration"]["proposal_id"]): item
            for item in campaign_receipt["proposal_registrations"]
        }
        registration_entry = registrations.get(decision.proposal_id)
        if decision.registration_hash is None:
            if registration_entry is not None or decision.outcome != "unresolved":
                raise RuntimeError(
                    "unregistered exact-mask decision has inconsistent terminal status"
                )
        else:
            if registration_entry is None:
                raise RuntimeError(
                    "exact-mask commit registration is absent from the campaign"
                )
            registration = dict(registration_entry["registration"])
            campaign_status = str(registration_entry["status"])
            status_matches = (
                campaign_status == decision.outcome
                if decision.outcome in {"accepted", "rejected"}
                else campaign_status in {"abandoned", "superseded"}
            )
            if (
                str(registration_entry["registration_hash"])
                != decision.registration_hash
                or str(registration["candidate_definition_hash"])
                != decision.candidate_definition_hash
                or str(registration["parent_state_hash"])
                != decision.statistical_parent_state_hash
                or str(registration["candidate_state_hash"])
                != decision.statistical_candidate_state_hash
                or str(registration["parent_calibration_hash"])
                != decision.parent_calibration_hash
                or str(registration["candidate_calibration_hash"])
                != decision.candidate_calibration_hash
                or not status_matches
            ):
                raise RuntimeError(
                    "exact-mask commit decision differs from its campaign registration"
                )
        native_parent_hash = str(decision.metadata.get("native_parent_mask_hash", ""))
        native_candidate_hash = str(
            decision.metadata.get("native_candidate_mask_hash", "")
        )
        current_native_hash = sampled_path_mask_hash(self.runtime.current_masks)
        expected_native_hash = (
            native_candidate_hash if decision.accepted else native_parent_hash
        )
        if current_native_hash != expected_native_hash:
            raise RuntimeError(
                "live physical mask differs from the exact-mask authority verdict"
            )

    def authorize(
        self,
        *,
        session: PruningSessionState,
        candidate: ActionCandidate,
    ) -> ExactHardMaskCommitDecision:
        if session.commit_authority_mode != "glocal_v3_shadow":
            raise ValueError(
                "glocal IRG authority may only serve a shadow glocal-v3 session"
            )
        if session.current_mask.mask_hash != candidate.parent_mask_hash:
            raise ValueError(
                "glocal IRG candidate does not descend from the current mask"
            )
        if session.state_hash != candidate.parent_state_hash:
            raise ValueError("glocal IRG candidate is stale at proposal birth")
        frontier_entry = session.frontier.preflight_commit(candidate.candidate_id)
        if frontier_entry.candidate is not candidate:
            raise ValueError(
                "glocal IRG authority requires the exact registered candidate "
                "object selected by the frontier"
            )
        self._require_session_calibration_lineage(session)
        if (
            session.current_mask.route_axis_hash != self.lowering_spec.route_axis_hash
            or candidate.resulting_mask.route_axis_hash
            != self.lowering_spec.route_axis_hash
        ):
            raise ValueError("native mask lowering spec belongs to another route axis")
        candidate.resulting_mask.removed_index_set_from(session.current_mask)
        parent_masks = _normalize_lowered_masks(
            self.mask_lowerer(session.current_mask),
            route_count=session.current_mask.route_count,
        )
        candidate_masks = _normalize_lowered_masks(
            self.mask_lowerer(candidate.resulting_mask),
            route_count=candidate.resulting_mask.route_count,
        )
        if set(parent_masks) != set(candidate_masks) or any(
            parent_masks[layer_id].shape != candidate_masks[layer_id].shape
            for layer_id in parent_masks
        ):
            raise ValueError(
                "native parent and candidate masks have different layer axes"
            )
        if any(
            np.any(candidate_masks[layer_id] & ~parent_masks[layer_id])
            for layer_id in parent_masks
        ):
            raise ValueError("native candidate mask reactivates a pruned route")
        native_removed = sum(
            int(np.count_nonzero(parent_masks[layer_id] & ~candidate_masks[layer_id]))
            for layer_id in parent_masks
        )
        if native_removed != candidate.removed_count:
            raise ValueError(
                "native mask lowering changes a different number of physical routes"
            )
        native_parent_hash = sampled_path_mask_hash(parent_masks)
        native_candidate_hash = sampled_path_mask_hash(candidate_masks)
        if native_parent_hash != sampled_path_mask_hash(self.runtime.current_masks):
            raise ValueError(
                "glocal campaign parent is not synchronized with the IRG hard mask"
            )
        bridge = IRGNativeMaskBridge(
            candidate_id=candidate.candidate_id,
            irg_parent_state_hash=candidate.parent_state_hash,
            route_axis_hash=session.current_mask.route_axis_hash,
            irg_parent_mask_hash=session.current_mask.mask_hash,
            irg_candidate_mask_hash=candidate.resulting_mask.mask_hash,
            native_parent_mask_hash=native_parent_hash,
            native_candidate_mask_hash=native_candidate_hash,
            native_layer_shapes=tuple(
                (layer_id, tuple(mask.shape)) for layer_id, mask in parent_masks.items()
            ),
            lowering_spec_hash=self.lowering_spec.content_hash,
        )
        self._register_irg_guidance_lineage(
            session=session,
            candidate=candidate,
        )
        campaign_before = self.runtime.campaign.receipt()
        registration_index = len(campaign_before["proposal_registrations"]) + 1
        proposal_id = (
            f"{session.run_id}:{candidate.candidate_id}:" f"claim-{registration_index}"
        )
        group_id = str(
            candidate.generator_parameters.get(
                "geometric_partition_id",
                "__global__",
            )
        )
        candidate_definition_hash = content_hash(
            {
                "candidate": candidate.to_dict(),
                "native_mask_bridge_hash": bridge.content_hash,
                "commit_policy_hash": self.policy.content_hash,
            }
        )
        statistical_parent_state_hash = self.runtime.campaign.current_state_hash
        parent_calibration_hash = self.runtime.campaign.current_calibration_hash
        result = self.runtime.run_proposal(
            proposal_id=proposal_id,
            candidate_masks=candidate_masks,
            candidate_definition_hash=candidate_definition_hash,
            local_upper_boundary=self.policy.local_upper_boundary,
            group_id=group_id,
            random_seed=self.policy.base_random_seed + registration_index,
            partition_id=group_id,
            previous_density=(
                session.current_mask.retained_count / session.current_mask.route_count
            ),
            proposed_density=(
                candidate.resulting_mask.retained_count
                / candidate.resulting_mask.route_count
            ),
        )
        evidence = result.decision_receipt
        if result.outcome == "accepted":
            if (
                not result.committed
                or evidence is None
                or evidence.outcome != "accepted"
                or result.candidate_state_hash
                != self.runtime.campaign.current_state_hash
                or result.candidate_calibration_hash
                != self.runtime.campaign.current_calibration_hash
                or sampled_path_mask_hash(self.runtime.current_masks)
                != native_candidate_hash
            ):
                raise RuntimeError(
                    "accepted glocal result did not atomically advance its locked state"
                )
            self._accepted_mask_bridges.append(bridge)
        elif result.committed:
            raise RuntimeError("nonaccepted glocal result claims a committed state")
        decision = ExactHardMaskCommitDecision(
            authority_id=self.authority_id,
            outcome=result.outcome,
            reason=result.reason,
            proposal_id=result.proposal_id,
            candidate_id=candidate.candidate_id,
            irg_parent_state_hash=candidate.parent_state_hash,
            parent_mask_hash=session.current_mask.mask_hash,
            candidate_mask_hash=candidate.resulting_mask.mask_hash,
            candidate_definition_hash=candidate_definition_hash,
            campaign_receipt_hash=result.campaign_receipt_hash,
            registration_hash=result.registration_hash,
            decision_evidence_receipt_hash=(
                None if evidence is None else evidence.content_hash
            ),
            statistical_parent_state_hash=statistical_parent_state_hash,
            statistical_candidate_state_hash=result.candidate_state_hash,
            parent_calibration_hash=parent_calibration_hash,
            candidate_calibration_hash=result.candidate_calibration_hash,
            lower_bound=None if evidence is None else evidence.lower_bound,
            upper_bound=None if evidence is None else evidence.upper_bound,
            local_upper_boundary=(
                None if evidence is None else evidence.local_upper_boundary
            ),
            distortion_debit=(None if evidence is None else evidence.distortion_debit),
            distortion_spent_after=(
                None if evidence is None else evidence.distortion_spent_after
            ),
            distortion_budget=self.runtime.campaign.distortion_budget,
            metadata={
                "authority_status": "shadow",
                "native_mask_bridge_hash": bridge.content_hash,
                "native_parent_mask_hash": native_parent_hash,
                "native_candidate_mask_hash": native_candidate_hash,
                "lowering_spec_hash": self.lowering_spec.content_hash,
                "commit_policy_hash": self.policy.content_hash,
                "route_axis_hash": session.current_mask.route_axis_hash,
            },
        )
        decision_hash = decision.content_hash
        if decision_hash in self._issued_commit_records:
            raise RuntimeError(
                "exact-mask commit decision identity was issued more than once"
            )
        self._issued_commit_records[decision_hash] = (
            result.content_hash,
            bridge.content_hash,
            result.campaign_receipt_hash,
        )
        session.decision.record_transition(
            "exact_hard_mask_commit_authority",
            decision.reason,
            authority_id=decision.authority_id,
            authority_outcome=decision.outcome,
            authority_decision_hash=decision.content_hash,
            proposal_id=decision.proposal_id,
            registration_hash=decision.registration_hash,
            decision_evidence_receipt_hash=(decision.decision_evidence_receipt_hash),
            campaign_receipt_hash=decision.campaign_receipt_hash,
            candidate_id=candidate.candidate_id,
        )
        if decision.accepted:
            authorization = _issue_physical_exact_mask_commit_authorization(
                authority_owner=self,
                runtime_owner=self.runtime,
                campaign_owner=self.runtime.campaign,
                session_owner=session,
                candidate_owner=candidate,
                decision=decision,
            )
            session.commit_candidate(
                candidate.candidate_id,
                authorization=authorization,
            )
            self.synchronize_committed_calibration(
                session=session,
                decision=decision,
            )
        else:
            self._revalidate_commit_authorization(
                runtime_owner=self.runtime,
                campaign_owner=self.runtime.campaign,
                session=session,
                candidate=candidate,
                decision=decision,
            )
            session.frontier.refuse(
                candidate.candidate_id,
                reason=(
                    "shadow glocal-v3 exact hard-mask authority did not "
                    "authorize a commit"
                ),
                authority_outcome=decision.outcome,
                authority_decision_hash=decision.content_hash,
            )
        return decision

    def issue_deployment_certificate(
        self,
        *,
        session: PruningSessionState,
        external_scientific_identity_hash: str | None = None,
        external_query_measure_hash: str | None = None,
        certificate_id: str | None = None,
    ) -> GlocalDeploymentMaskCertificate:
        """Finalize the exact current IRG/native mask bridge after terminal audit."""

        if session.commit_authority_mode != "glocal_v3_shadow":
            raise ValueError(
                "glocal deployment authority requires a shadow glocal-v3 session"
            )
        native_masks = _normalize_lowered_masks(
            self.mask_lowerer(session.current_mask),
            route_count=session.current_mask.route_count,
        )
        if sampled_path_mask_hash(native_masks) != sampled_path_mask_hash(
            self.runtime.current_masks
        ):
            raise ValueError(
                "deployment session mask is not synchronized with the glocal runtime"
            )
        if session.current_mask.route_axis_hash != self.lowering_spec.route_axis_hash:
            raise ValueError(
                "deployment session belongs to another scientific route axis"
            )
        return self.runtime.issue_deployment_certificate(
            native_route_axis_hash=self.lowering_spec.route_axis_hash,
            native_mask_bridge_hashes=tuple(
                item.content_hash for item in self._accepted_mask_bridges
            ),
            irg_final_state_hash=session.state_hash,
            irg_final_mask_hash=session.current_mask.mask_hash,
            external_scientific_identity_hash=external_scientific_identity_hash,
            external_query_measure_hash=external_query_measure_hash,
            certificate_id=certificate_id,
        )

    def issue_deployment_authority(
        self,
        *,
        session: PruningSessionState,
        external_scientific_identity_hash: str | None = None,
        external_query_measure_hash: str | None = None,
        certificate_id: str | None = None,
    ) -> GlocalDeploymentAuthorityHandle:
        """Mint the nonserializable live authority from the physical runtime."""

        if session.commit_authority_mode != "glocal_v3_shadow":
            raise ValueError(
                "glocal deployment authority requires a shadow glocal-v3 session"
            )
        native_masks = _normalize_lowered_masks(
            self.mask_lowerer(session.current_mask),
            route_count=session.current_mask.route_count,
        )
        if sampled_path_mask_hash(native_masks) != sampled_path_mask_hash(
            self.runtime.current_masks
        ):
            raise ValueError(
                "deployment session mask is not synchronized with the glocal runtime"
            )
        if session.current_mask.route_axis_hash != self.lowering_spec.route_axis_hash:
            raise ValueError(
                "deployment session belongs to another scientific route axis"
            )
        return self.runtime.issue_deployment_authority(
            native_route_axis_hash=self.lowering_spec.route_axis_hash,
            native_mask_bridge_hashes=tuple(
                item.content_hash for item in self._accepted_mask_bridges
            ),
            irg_final_state_hash=session.state_hash,
            irg_final_mask_hash=session.current_mask.mask_hash,
            external_scientific_identity_hash=external_scientific_identity_hash,
            external_query_measure_hash=external_query_measure_hash,
            certificate_id=certificate_id,
        )


class GlocalShadowIRGExecution:
    """Canonical public composition for the still-shadow glocal-v3 path.

    Low-level runners and authorities remain available for local validation,
    but this composition is the supported integration boundary: one object
    owns guidance acquisition, the exact-mask commit authority, and the
    campaign-global statistical/resource ledgers.  It deliberately cannot
    advertise claim eligibility before the external promotion gates pass.
    """

    foundation_identity = EVIDENCE_FOUNDATION_GLOCAL_V3
    authority_status: Literal["shadow"] = "shadow"
    claim_eligible = False

    def __init__(
        self,
        *,
        runtime: GlocalBrierCampaignRuntime,
        version: ResponseVersion,
        providers: ProviderRegistry,
        mask_lowerer: Callable[[HardMask], Mapping[str, np.ndarray]],
        lowering_spec: IRGNativeMaskLoweringSpec,
        commit_policy: GlocalIRGCommitPolicy,
        response_capability_admission: Callable[[ResponseVersion], bool],
        guidance_resource_envelopes: (
            Mapping[str, GlocalPacketResourceEnvelope] | None
        ) = None,
        runner_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(response_capability_admission) or not bool(
            response_capability_admission(version)
        ):
            raise ValueError(
                "the public glocal composition rejected the supplied response capability"
            )
        options = dict(runner_options or {})
        forbidden = {
            "version",
            "providers",
            "commit_authority",
            "commit_authority_mode",
        }.intersection(options)
        if forbidden:
            raise ValueError(
                "runner_options may not override the integrated shadow "
                "authority boundary: " + ",".join(sorted(forbidden))
            )
        adapters = tuple(options.get("measurement_adapters", ()))
        adapter_by_kind = {
            str(kind): adapter
            for adapter in adapters
            for kind in getattr(adapter, "supported_kinds", ())
        }
        required_adapter_ids = {
            "response_moments": "irg_adaptive_whole_route_response_acquisition_v1",
            "calibration_logits": "irg_calibration_active_contrast_adapter_v1",
            "calibration_refit": "irg_calibration_active_contrast_adapter_v1",
        }
        missing_kinds = set(required_adapter_ids).difference(adapter_by_kind)
        if missing_kinds:
            raise ValueError(
                "the canonical glocal composition requires unified response "
                "and calibration acquisition adapters; missing kinds: "
                + ",".join(sorted(missing_kinds))
            )
        wrong_adapters = {
            kind: str(getattr(adapter_by_kind[kind], "adapter_id", ""))
            for kind, expected in required_adapter_ids.items()
            if str(getattr(adapter_by_kind[kind], "adapter_id", "")) != expected
        }
        if wrong_adapters:
            raise ValueError(
                "the canonical glocal composition requires canonical scientific "
                "effect adapters, not kind-label aliases: "
                + ",".join(
                    f"{kind}={adapter_id}"
                    for kind, adapter_id in sorted(wrong_adapters.items())
                )
            )
        self.runtime = runtime
        self.version = version
        self.providers = providers
        self.authority = GlocalIRGCommitAuthority(
            runtime=runtime,
            mask_lowerer=mask_lowerer,
            lowering_spec=lowering_spec,
            policy=commit_policy,
            guidance_resource_envelopes=guidance_resource_envelopes,
        )
        self.runner = CommonIRGRunner(
            version=version,
            providers=providers,
            commit_authority_mode="glocal_v3_shadow",
            commit_authority=self.authority,
            **options,
        )

    @property
    def campaign(self) -> CampaignEvidenceState:
        return self.runtime.campaign

    def _validate_session(
        self,
        session: PruningSessionState,
        *,
        require_acquisition_ready: bool = True,
    ) -> None:
        active_version = str(self.runner.version.name)
        if session.commit_authority_mode != "glocal_v3_shadow":
            raise ValueError(
                "the public glocal composition requires a shadow glocal-v3 session"
            )
        if str(session.active_version) != active_version:
            raise ValueError(
                "the session and integrated response-version runner differ"
            )
        session.identity.assert_canonical_response_for_version(session.active_version)
        if session.identity.response_constraint is None:
            raise ValueError(
                "the canonical glocal path requires a frozen response safe region"
            )
        partitioning = session.identity.geometric_partitioning
        if partitioning is None:
            raise ValueError(
                "the canonical glocal path requires complete disjoint geometric "
                "partitions"
            )
        if partitioning.route_count != session.current_mask.route_count:
            raise ValueError(
                "geometric partitioning and the current hard mask differ in size"
            )
        if (
            session.identity.context.context_axis_hash
            != self.runtime.context_design.context_axis_hash
        ):
            raise ValueError(
                "IRG guidance and glocal adjudication use different frozen "
                "context populations"
            )
        if (
            session.current_mask.route_axis_hash
            != self.authority.lowering_spec.route_axis_hash
        ):
            raise ValueError(
                "IRG session and native mask lowering use different route axes"
            )
        calibration_state = session.cache.get(CALIBRATION_UNCERTAINTY_CACHE_KEY)
        if not isinstance(calibration_state, CalibrationUncertaintyState):
            raise ValueError(
                "the canonical glocal path requires typed calibration uncertainty"
            )
        expected_calibration_hash = self.runtime.campaign.current_calibration_hash
        committed_fit = self.runtime.current_calibration.fit_state
        support_design = self.runtime.context_design.calibration_sampling_design
        fitted_context_count = len(
            self.runtime.current_calibration.observation_set.context_ids
        )
        if (
            calibration_state.calibration_fit_state_hash != expected_calibration_hash
            or committed_fit.content_hash != expected_calibration_hash
            or calibration_state.calibration_state_id != committed_fit.fit_id
            or not np.array_equal(
                calibration_state.parameter_mean,
                np.asarray(committed_fit.parameters, dtype=np.float64),
            )
            or not np.array_equal(
                calibration_state.parameter_covariance,
                np.asarray(committed_fit.covariance, dtype=np.float64),
            )
            or calibration_state.fitted_context_count != fitted_context_count
            or calibration_state.planning_context_count
            != max(
                fitted_context_count,
                sum(self.runtime.calibration_targets.values()),
            )
            or calibration_state.numerical_converged != committed_fit.converged
            or (
                support_design is not None
                and (
                    calibration_state.calibration_support_design_hash
                    != support_design.content_hash
                    or calibration_state.calibration_support_capacity
                    != len(self.runtime.context_design.calibration_context_ids)
                )
            )
        ):
            raise ValueError(
                "IRG calibration uncertainty and the campaign's committed "
                "CalibrationFitState differ"
            )
        if require_acquisition_ready:
            checked: set[str] = set()
            calibration_refresh_pending = (
                str(session.decision.pending_model_validity_action) == "refresh"
                and str(
                    getattr(
                        getattr(
                            session.version_state,
                            "intrinsic_state",
                            session.version_state,
                        ),
                        "calibration_state_id",
                        "",
                    )
                )
                != calibration_state.calibration_state_id
            )
            for kind in (
                "response_moments",
                "calibration_logits",
                "calibration_refit",
            ):
                adapter = self.runner._adapter_by_kind[kind]
                if adapter.adapter_id in checked:
                    continue
                if (
                    calibration_refresh_pending
                    and adapter.adapter_id
                    == "irg_calibration_active_contrast_adapter_v1"
                ):
                    checked.add(adapter.adapter_id)
                    continue
                adapter.assert_ready(session)
                checked.add(adapter.adapter_id)
        native_masks = _normalize_lowered_masks(
            self.authority.mask_lowerer(session.current_mask),
            route_count=session.current_mask.route_count,
        )
        if sampled_path_mask_hash(native_masks) != sampled_path_mask_hash(
            self.runtime.current_masks
        ):
            raise ValueError(
                "IRG session and glocal campaign do not share the same locked "
                "parent mask"
            )

    def run(
        self,
        session: PruningSessionState,
        *,
        maximum_commits: int = 1,
        budget_targets: Sequence[float] = (),
    ) -> PruningRunResult:
        self._validate_session(session)
        return self.runner.run(
            session,
            maximum_commits=maximum_commits,
            budget_targets=budget_targets,
        )

    def lock_terminal_audit(
        self,
        policy: GlocalTerminalAuditPolicy | None = None,
    ) -> ValidationLock:
        return self.runtime.lock_terminal_audit(policy)

    def execute_terminal_audit(self) -> GlocalTerminalAuditResult:
        return self.runtime.execute_terminal_audit()

    def issue_deployment_certificate(
        self,
        *,
        session: PruningSessionState,
        external_scientific_identity_hash: str | None = None,
        external_query_measure_hash: str | None = None,
        certificate_id: str | None = None,
    ) -> GlocalDeploymentMaskCertificate:
        self._validate_session(session, require_acquisition_ready=False)
        return self.authority.issue_deployment_certificate(
            session=session,
            external_scientific_identity_hash=(external_scientific_identity_hash),
            external_query_measure_hash=external_query_measure_hash,
            certificate_id=certificate_id,
        )

    def issue_deployment_authority(
        self,
        *,
        session: PruningSessionState,
        external_scientific_identity_hash: str | None = None,
        external_query_measure_hash: str | None = None,
        certificate_id: str | None = None,
    ) -> GlocalDeploymentAuthorityHandle:
        self._validate_session(session, require_acquisition_ready=False)
        return self.authority.issue_deployment_authority(
            session=session,
            external_scientific_identity_hash=(external_scientific_identity_hash),
            external_query_measure_hash=external_query_measure_hash,
            certificate_id=certificate_id,
        )


__all__ = [
    "CampaignCalibrationLogitAcquisitionReceipt",
    "CampaignCalibrationTransitionDecision",
    "EVIDENCE_FOUNDATION_GLOCAL_V3",
    "GLOCAL_NEYMAN_ALLOCATION_RULE",
    "FrozenBinaryContextDesign",
    "FrozenCalibrationBinding",
    "GlocalBrierCampaignRuntime",
    "GlocalDeploymentMaskCertificate",
    "GlocalIRGCommitAuthority",
    "GlocalIRGCommitPolicy",
    "GlocalShadowIRGExecution",
    "GlocalNeymanPolicy",
    "GlocalPacketResourceEnvelope",
    "GlocalProposalLook",
    "GlocalProposalResult",
    "GlocalTerminalAuditPolicy",
    "GlocalTerminalAuditResult",
    "IRGNativeMaskBridge",
    "IRGNativeMaskLoweringSpec",
    "ResourceBudgetUnavailable",
    "calibration_binding_from_native",
    "freeze_binary_context_design",
    "glocal_brier_calibration_algorithm_spec",
    "locked_brier_state_hash",
]
