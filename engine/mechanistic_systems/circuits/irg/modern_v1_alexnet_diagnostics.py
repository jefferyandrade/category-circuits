"""Authority-inert AlexNet diagnostic acquisition boundary.

Modern-v1 production uses the staged proposal protocol and therefore must not
carry the former one-shot ``acquisition_factory``.  A few frozen-chart
diagnostics still need the admitted dense scorer, train contexts, and (for the
registered C2 ablations) one generation-zero chart.  This module keeps that
capability outside :class:`V1ProductionRuntimeBundle` and binds it to a narrow,
explicitly non-production protocol.

Neither type below proxies production-bundle attributes.  Callers must name
the production owners and the diagnostic acquisition owner separately; this
makes accidentally rebranding the diagnostic capability as proposal or mask
authority fail closed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_alexnet import (
    V1AlexNetBackendAdmission,
    V1AlexNetDependencyUnavailable,
    load_v1_alexnet_backend_factory,
)
from .modern_v1_acquisition import V1RelinearizedAcquisition
from .modern_v1_campaign_contract import V1OptionACampaignBinding
from .modern_v1_controller import V1ControllerState
from .modern_v1_neutral_runtime import ClassBalancedSamplingPopulation
from .modern_v1_production import (
    V1ProductionCallableSeal,
    V1ProductionRuntimeBundle,
)


ALEXNET_DIAGNOSTIC_PROTOCOL: Final = "alexnet_diagnostic_one_shot_v1"
ALEXNET_DIAGNOSTIC_PURPOSE: Final = "train_only_frozen_chart_diagnostic"
ALEXNET_DIAGNOSTIC_CARRIER_SCHEMA: Final = (
    "partition_free_v1_alexnet_diagnostic_acquisition_carrier_v1"
)
ALEXNET_DIAGNOSTIC_RUNTIME_VIEW_SCHEMA: Final = (
    "partition_free_v1_alexnet_diagnostic_runtime_view_v1"
)
ALEXNET_PRODUCTION_FACTORY_SPEC: Final = (
    "mechanistic_systems.circuits.irg.modern_v1_alexnet_backend:"
    "build_option_a_alexnet_bundle"
)
ALEXNET_SAMPLED_DISCOVERY_DIAGNOSTIC_FACTORY_SPEC: Final = (
    "mechanistic_systems.circuits.irg.modern_v1_alexnet_backend:"
    "build_option_a_alexnet_static_score_bundle"
)
ALEXNET_SAMPLED_DISCOVERY_ADMISSION_SCHEMA: Final = (
    "partition_free_v1_alexnet_sampled_discovery_diagnostic_admission_sibling_v1"
)
ALEXNET_SAMPLED_DISCOVERY_ADMISSION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_alexnet_sampled_discovery_diagnostic_admission_receipt_v1"
)
_CARRIER_PREFIX: Final = "partition-free-v1-alexnet-diagnostic-acquisition-"
_VIEW_PREFIX: Final = "partition-free-v1-alexnet-diagnostic-runtime-view-"
_SAMPLED_DISCOVERY_ADMISSION_ID_PREFIX: Final = (
    "partition-free-v1-alexnet-sampled-discovery-diagnostic-admission-id-"
)
_SAMPLED_DISCOVERY_ADMISSION_RECEIPT_PREFIX: Final = (
    "partition-free-v1-alexnet-sampled-discovery-diagnostic-admission-receipt-"
)
_SOURCE_MANIFEST_SCHEMA: Final = "partition_free_v1_source_manifest_v1"
_SOURCE_SEAL_PREFIX: Final = "partition-free-v1-alexnet-source-seal-"
_ARTIFACT_TUPLE_PREFIX: Final = (
    "partition-free-v1-alexnet-sampled-discovery-parent-artifact-tuple-"
)
_SAMPLED_DISCOVERY_AUTHORITY_METADATA: Final = MappingProxyType(
    {
        "sampled_D_discovery_authority": True,
        "C_FQR_certification_authority": False,
        "K_calibration_authority": False,
        "O_paired_BCE_oracle_authority": False,
        "candidate_freeze_authority": False,
        "calibration_authority": False,
        "bce_oracle_authority": False,
        "mask_commit_authority": False,
        "terminal_validation_authority": False,
        "pareto_claim_authority": False,
        "production_authority": False,
        "scientific_trajectory_admission_eligible": False,
        "dynamic_trajectory_runtime_comparison_eligible": False,
        "exploratory_development_only": True,
        "canonical_fqr_candidate_masks_are_proposals_only": True,
        "canonical_fqr_mask_commit_authority": False,
    }
)


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _factory_source_identity(spec: str) -> tuple[Path, str]:
    factory = load_v1_alexnet_backend_factory(spec)
    raw_source = inspect.getsourcefile(factory)
    if raw_source is None:
        raise V1AlexNetDependencyUnavailable(
            f"AlexNet backend factory source is unavailable for {spec}"
        )
    source = Path(raw_source).resolve()
    if not source.is_file():
        raise V1AlexNetDependencyUnavailable(
            f"AlexNet backend factory source is unavailable for {spec}: {source}"
        )
    return source, _sha256_file(source)


def _artifact_tuple_hash(admission: V1AlexNetBackendAdmission) -> str:
    return content_hash(admission.artifacts, prefix=_ARTIFACT_TUPLE_PREFIX)


def _verify_parent_source_manifest(
    *,
    admission: V1AlexNetBackendAdmission,
    verified_artifacts: Mapping[str, Path],
) -> tuple[str, str]:
    """Replay the production source seal without deriving another manifest."""

    source_path = verified_artifacts.get("source_manifest")
    if not isinstance(source_path, Path):
        raise TypeError("AlexNet admission lacks an exact source-manifest path")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("AlexNet source manifest must contain an object")
    required = {
        "schema",
        "source_files",
        "terminal_outcomes_opened",
        "source_seal_hash",
    }
    extra = set(payload).difference(required | {"metadata"})
    missing = required.difference(payload)
    if missing or extra:
        raise ValueError(
            "AlexNet source-manifest fields differ; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if payload.get("schema") != _SOURCE_MANIFEST_SCHEMA:
        raise ValueError("unsupported AlexNet source-manifest schema")
    if payload.get("terminal_outcomes_opened") is not False:
        raise PermissionError("sampled discovery parent opened terminal outcomes")
    source_files = payload.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise TypeError("AlexNet source manifest requires a nonempty source_files map")
    expected_source_files = {
        item.role: item.sha256
        for item in admission.artifacts
        if item.role != "source_manifest"
    }
    expected_source_files["backend_factory"] = admission.backend_factory_source_hash
    if admission.production_geometric_selector_source_hash is not None:
        expected_source_files["production_geometric_selector"] = (
            admission.production_geometric_selector_source_hash
        )
    if source_files != expected_source_files:
        raise ValueError(
            "AlexNet source manifest differs from the unchanged parent artifact tuple"
        )
    declared_seal = payload.get("source_seal_hash")
    if type(declared_seal) is not str or not declared_seal:
        raise ValueError("AlexNet source manifest lacks a source seal")
    seal_payload = dict(payload)
    seal_payload.pop("source_seal_hash")
    observed_seal = content_hash(seal_payload, prefix=_SOURCE_SEAL_PREFIX)
    if declared_seal != observed_seal:
        raise ValueError("AlexNet parent source-manifest seal is not reproducible")
    artifact = next(
        item for item in admission.artifacts if item.role == "source_manifest"
    )
    if artifact.sha256 != _sha256_file(source_path):
        raise ValueError("AlexNet parent source-manifest artifact SHA-256 drifted")
    return artifact.sha256, declared_seal


def _validate_production_parent(
    *,
    parent_admission: V1AlexNetBackendAdmission,
    binding: V1OptionACampaignBinding,
    artifact_root: str | Path,
) -> tuple[str, str, str]:
    if type(parent_admission) is not V1AlexNetBackendAdmission:
        raise TypeError(
            "sampled discovery requires an exact production parent admission"
        )
    if parent_admission.backend_factory_spec != ALEXNET_PRODUCTION_FACTORY_SPEC:
        raise ValueError(
            "sampled discovery parent must use the exact production AlexNet factory"
        )
    parent_admission.validate_campaign(binding)
    verified = parent_admission.verify_artifacts(artifact_root)
    production_source, production_sha = _factory_source_identity(
        parent_admission.backend_factory_spec
    )
    diagnostic_source, diagnostic_sha = _factory_source_identity(
        ALEXNET_SAMPLED_DISCOVERY_DIAGNOSTIC_FACTORY_SPEC
    )
    if production_source != diagnostic_source or production_sha != diagnostic_sha:
        raise ValueError(
            "production and sampled-discovery factories are not in the same "
            "backend file"
        )
    if production_sha != parent_admission.backend_factory_source_hash:
        raise ValueError(
            "production parent backend source differs from its admitted SHA-256"
        )
    source_manifest_sha, source_seal_hash = _verify_parent_source_manifest(
        admission=parent_admission,
        verified_artifacts=verified,
    )
    return source_manifest_sha, source_seal_hash, production_sha


def derive_v1_alexnet_sampled_discovery_admission(
    *,
    parent_admission: V1AlexNetBackendAdmission,
    binding: V1OptionACampaignBinding,
    artifact_root: str | Path,
) -> V1AlexNetBackendAdmission:
    """Derive a D-only diagnostic sibling without changing any sealed artifact."""

    source_manifest_sha, source_seal_hash, backend_sha = (
        _validate_production_parent(
            parent_admission=parent_admission,
            binding=binding,
            artifact_root=artifact_root,
        )
    )
    artifact_tuple_hash = _artifact_tuple_hash(parent_admission)
    metadata = dict(parent_admission.metadata)
    # A production parent may record downstream eligibility that is valid for
    # the production runtime but is categorically invalid for this D-only
    # sibling.  Preserve configuration facts while closing every top-level
    # eligibility/authority boolean before installing the one D permission.
    for name, value in tuple(metadata.items()):
        if (
            type(value) is bool
            and ("authority" in name.lower() or "eligible" in name.lower())
        ):
            metadata[name] = False
    metadata.update(
        {
            "sampled_discovery_admission_schema": (
                ALEXNET_SAMPLED_DISCOVERY_ADMISSION_SCHEMA
            ),
            "parent_production_admission_hash": parent_admission.admission_hash,
            "parent_production_admission_id": parent_admission.admission_id,
            "parent_production_factory_spec": ALEXNET_PRODUCTION_FACTORY_SPEC,
            "parent_backend_factory_source_hash": backend_sha,
            "parent_artifact_tuple_hash": artifact_tuple_hash,
            "parent_source_manifest_sha256": source_manifest_sha,
            "parent_source_seal_hash": source_seal_hash,
            "artifacts_unchanged_from_parent": True,
            "source_manifest_unchanged_from_parent": True,
            "source_manifest_derived_or_rewritten": False,
            "backend_source_file_shared_with_parent": True,
            "initial_dynamic_acquisition_deferred": True,
            "sampled_registered_look_discovery_only": True,
            "fd_q_geometry_constructed": False,
            "candidate_or_mask_created": False,
            "validation_outcomes_opened": False,
            "terminal_outcomes_used_for_selection": False,
            **dict(_SAMPLED_DISCOVERY_AUTHORITY_METADATA),
        }
    )
    admission_id = content_hash(
        {
            "schema": ALEXNET_SAMPLED_DISCOVERY_ADMISSION_SCHEMA,
            "parent_production_admission_hash": parent_admission.admission_hash,
            "campaign_binding_hash": binding.binding_hash,
            "parent_artifact_tuple_hash": artifact_tuple_hash,
            "parent_source_manifest_sha256": source_manifest_sha,
            "parent_source_seal_hash": source_seal_hash,
            "diagnostic_factory_spec": (
                ALEXNET_SAMPLED_DISCOVERY_DIAGNOSTIC_FACTORY_SPEC
            ),
            "backend_factory_source_hash": backend_sha,
            "authority": dict(_SAMPLED_DISCOVERY_AUTHORITY_METADATA),
        },
        prefix=_SAMPLED_DISCOVERY_ADMISSION_ID_PREFIX,
    )
    sibling = replace(
        parent_admission,
        admission_id=admission_id,
        backend_factory_spec=ALEXNET_SAMPLED_DISCOVERY_DIAGNOSTIC_FACTORY_SPEC,
        metadata=metadata,
    )
    validate_v1_alexnet_sampled_discovery_admission(
        admission=sibling,
        parent_admission=parent_admission,
        binding=binding,
        artifact_root=artifact_root,
    )
    return sibling


def validate_v1_alexnet_sampled_discovery_admission(
    *,
    admission: V1AlexNetBackendAdmission,
    parent_admission: V1AlexNetBackendAdmission,
    binding: V1OptionACampaignBinding,
    artifact_root: str | Path,
) -> Mapping[str, Any]:
    """Validate exact lineage and return an authority-inert identity receipt."""

    if type(admission) is not V1AlexNetBackendAdmission:
        raise TypeError("sampled discovery requires an exact diagnostic admission")
    source_manifest_sha, source_seal_hash, backend_sha = (
        _validate_production_parent(
            parent_admission=parent_admission,
            binding=binding,
            artifact_root=artifact_root,
        )
    )
    admission.validate_campaign(binding)
    if admission.backend_factory_spec != (
        ALEXNET_SAMPLED_DISCOVERY_DIAGNOSTIC_FACTORY_SPEC
    ):
        raise ValueError(
            "sampled discovery requires the exact deferred diagnostic factory"
        )
    _diagnostic_source, diagnostic_sha = _factory_source_identity(
        admission.backend_factory_spec
    )
    if (
        admission.backend_factory_source_hash != backend_sha
        or diagnostic_sha != backend_sha
        or admission.backend_factory_source_hash
        != parent_admission.backend_factory_source_hash
    ):
        raise ValueError(
            "sampled-discovery backend source SHA differs from the production parent"
        )
    if admission.artifacts != parent_admission.artifacts:
        raise ValueError(
            "sampled-discovery artifact tuple differs from the production parent"
        )
    if admission.production_geometric_selector_source_hash != (
        parent_admission.production_geometric_selector_source_hash
    ):
        raise ValueError(
            "sampled-discovery selector source SHA differs from the production parent"
        )
    lineage_fields = (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "model_seal_hash",
        "query_id",
        "referent_id",
        "route_axis_hash",
        "context_population_hash",
        "dense_reference_hash",
        "calibration_identity",
        "terminal_validation_population_hash",
        "architecture_family",
        "controller_identity",
    )
    for name in lineage_fields:
        if getattr(admission, name) != getattr(parent_admission, name):
            raise ValueError(
                f"sampled-discovery admission differs from parent field {name}"
            )
    artifact_tuple_hash = _artifact_tuple_hash(parent_admission)
    expected_metadata = {
        "sampled_discovery_admission_schema": (
            ALEXNET_SAMPLED_DISCOVERY_ADMISSION_SCHEMA
        ),
        "parent_production_admission_hash": parent_admission.admission_hash,
        "parent_production_admission_id": parent_admission.admission_id,
        "parent_production_factory_spec": ALEXNET_PRODUCTION_FACTORY_SPEC,
        "parent_backend_factory_source_hash": backend_sha,
        "parent_artifact_tuple_hash": artifact_tuple_hash,
        "parent_source_manifest_sha256": source_manifest_sha,
        "parent_source_seal_hash": source_seal_hash,
        "artifacts_unchanged_from_parent": True,
        "source_manifest_unchanged_from_parent": True,
        "source_manifest_derived_or_rewritten": False,
        "backend_source_file_shared_with_parent": True,
        "initial_dynamic_acquisition_deferred": True,
        "sampled_registered_look_discovery_only": True,
        "fd_q_geometry_constructed": False,
        "candidate_or_mask_created": False,
        "validation_outcomes_opened": False,
        "terminal_outcomes_used_for_selection": False,
        **dict(_SAMPLED_DISCOVERY_AUTHORITY_METADATA),
    }
    metadata = dict(admission.metadata)
    allowed_true_control_flags = {
        "sampled_D_discovery_authority",
    }
    leaked_controls = sorted(
        name
        for name, value in metadata.items()
        if type(value) is bool
        and value
        and ("authority" in name.lower() or "eligible" in name.lower())
        and name not in allowed_true_control_flags
    )
    if leaked_controls:
        raise PermissionError(
            "sampled-discovery admission inherited production eligibility or "
            f"authority: {leaked_controls}"
        )
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise PermissionError(
                f"sampled-discovery admission crossed metadata boundary {name}"
            )
    expected_id = content_hash(
        {
            "schema": ALEXNET_SAMPLED_DISCOVERY_ADMISSION_SCHEMA,
            "parent_production_admission_hash": parent_admission.admission_hash,
            "campaign_binding_hash": binding.binding_hash,
            "parent_artifact_tuple_hash": artifact_tuple_hash,
            "parent_source_manifest_sha256": source_manifest_sha,
            "parent_source_seal_hash": source_seal_hash,
            "diagnostic_factory_spec": (
                ALEXNET_SAMPLED_DISCOVERY_DIAGNOSTIC_FACTORY_SPEC
            ),
            "backend_factory_source_hash": backend_sha,
            "authority": dict(_SAMPLED_DISCOVERY_AUTHORITY_METADATA),
        },
        prefix=_SAMPLED_DISCOVERY_ADMISSION_ID_PREFIX,
    )
    if admission.admission_id != expected_id:
        raise ValueError("sampled-discovery admission ID is not reproducible")
    if admission.admission_hash == parent_admission.admission_hash:
        raise ValueError("sampled-discovery admission aliases its production parent")
    verified = admission.verify_artifacts(artifact_root)
    sibling_manifest_sha, sibling_source_seal = _verify_parent_source_manifest(
        admission=admission,
        verified_artifacts=verified,
    )
    if (
        sibling_manifest_sha != source_manifest_sha
        or sibling_source_seal != source_seal_hash
    ):
        raise ValueError(
            "sampled-discovery source manifest differs from the production parent"
        )
    receipt: dict[str, Any] = {
        "schema": ALEXNET_SAMPLED_DISCOVERY_ADMISSION_RECEIPT_SCHEMA,
        "campaign_binding_hash": binding.binding_hash,
        "parent_production_admission_hash": parent_admission.admission_hash,
        "diagnostic_admission_hash": admission.admission_hash,
        "parent_artifact_tuple_hash": artifact_tuple_hash,
        "source_manifest_sha256": source_manifest_sha,
        "source_seal_hash": source_seal_hash,
        "backend_factory_source_hash": backend_sha,
        "production_factory_spec": ALEXNET_PRODUCTION_FACTORY_SPEC,
        "diagnostic_factory_spec": (
            ALEXNET_SAMPLED_DISCOVERY_DIAGNOSTIC_FACTORY_SPEC
        ),
        "artifacts_unchanged_from_parent": True,
        "source_manifest_unchanged_from_parent": True,
        "authority": dict(_SAMPLED_DISCOVERY_AUTHORITY_METADATA),
    }
    receipt["receipt_hash"] = content_hash(
        receipt,
        prefix=_SAMPLED_DISCOVERY_ADMISSION_RECEIPT_PREFIX,
    )
    return MappingProxyType(receipt)


@dataclass(frozen=True, slots=True)
class V1AlexNetDiagnosticAcquisitionCarrier:
    """Sealed train-only one-shot capability with no production authority."""

    _sealed_factory: V1ProductionCallableSeal = field(repr=False, compare=False)
    backend_manifest_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    model_data_seal_hash: str
    runtime_logical_config_hash: str
    production_bundle_hash: str
    initial_mask_hash: str
    route_axis_hash: str
    train_population_hash: str
    production_proposal_backend_identity_hash: str
    protocol: str = ALEXNET_DIAGNOSTIC_PROTOCOL
    purpose: str = ALEXNET_DIAGNOSTIC_PURPOSE
    proposal_only: bool = True
    production_authority: bool = False
    train_optimization_data_only: bool = True
    terminal_outcomes_observed: bool = False
    authority_inert: bool = True
    candidate_or_mask_authority: bool = False
    schema: str = ALEXNET_DIAGNOSTIC_CARRIER_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self._sealed_factory) is not V1ProductionCallableSeal
            or self._sealed_factory.role != "acquisition_factory"
        ):
            raise TypeError(
                "diagnostic acquisition requires an exact acquisition-factory seal"
            )
        for name in (
            "backend_manifest_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "model_data_seal_hash",
            "runtime_logical_config_hash",
            "production_bundle_hash",
            "initial_mask_hash",
            "route_axis_hash",
            "train_population_hash",
            "production_proposal_backend_identity_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if not self._sealed_factory.callable_spec.endswith(
            ":_DynamicAcquisitionFactory"
        ):
            raise TypeError(
                "AlexNet diagnostic carrier requires the explicit deferred "
                "dynamic-acquisition owner"
            )
        factory = self._sealed_factory.callable_value
        cache = getattr(factory, "_cache", None)
        if type(cache) is not dict or cache:
            raise PermissionError(
                "diagnostic acquisition carrier must begin with a cold FD/Q cache"
            )
        required_factory_attributes = (
            "binding",
            "route_axis",
            "scorer",
            "fd_contexts",
            "population",
            "extraction",
        )
        if any(not hasattr(factory, name) for name in required_factory_attributes):
            raise TypeError("diagnostic acquisition factory lacks a required owner")
        if type(factory.population) is not ClassBalancedSamplingPopulation:
            raise TypeError(
                "diagnostic acquisition requires the exact train sampling population"
            )
        contexts = tuple(factory.fd_contexts)
        if (
            not contexts
            or len(set(contexts)) != len(contexts)
            or not set(contexts).issubset(factory.population.population_context_ids)
        ):
            raise ValueError(
                "diagnostic acquisition contexts differ from the train population"
            )
        if factory.binding.binding_hash != self.campaign_binding_hash:
            raise ValueError("diagnostic acquisition factory names another campaign")
        if factory.route_axis.route_axis_hash != self.route_axis_hash:
            raise ValueError("diagnostic acquisition factory names another route axis")
        if factory.population.population_hash != self.train_population_hash:
            raise ValueError(
                "diagnostic acquisition factory names another train population"
            )
        if getattr(factory.population, "terminal_outcomes_opened", None) is not False:
            raise PermissionError("diagnostic acquisition observed terminal outcomes")
        if (
            self.protocol != ALEXNET_DIAGNOSTIC_PROTOCOL
            or self.purpose != ALEXNET_DIAGNOSTIC_PURPOSE
            or self.proposal_only is not True
            or self.production_authority is not False
            or self.train_optimization_data_only is not True
            or self.terminal_outcomes_observed is not False
            or self.authority_inert is not True
            or self.candidate_or_mask_authority is not False
            or self.schema != ALEXNET_DIAGNOSTIC_CARRIER_SCHEMA
        ):
            raise ValueError(
                "AlexNet diagnostic carrier crossed its authority boundary"
            )
        if self.identity_hash == self.production_proposal_backend_identity_hash:
            raise ValueError(
                "diagnostic identity aliases production proposal authority"
            )

    @property
    def factory_identity_hash(self) -> str:
        return self._sealed_factory.identity_hash

    @property
    def factory_callable_spec(self) -> str:
        return self._sealed_factory.callable_spec

    @property
    def factory_source_file_sha256(self) -> str:
        return self._sealed_factory.source_file_sha256

    @property
    def factory_configuration_hash(self) -> str:
        return self._sealed_factory.configuration_hash

    @property
    def scorer(self) -> Any:
        return self._sealed_factory.callable_value.scorer

    @property
    def initial_context_ids(self) -> tuple[str, ...]:
        return tuple(self._sealed_factory.callable_value.fd_contexts)

    @property
    def train_population(self) -> ClassBalancedSamplingPopulation:
        return self._sealed_factory.callable_value.population

    @property
    def minimum_retained_route_count(self) -> int:
        return int(
            self._sealed_factory.callable_value.extraction.minimum_retained_route_count
        )

    @property
    def is_cold(self) -> bool:
        return not bool(self._sealed_factory.callable_value._cache)

    @property
    def identity_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "protocol": self.protocol,
                "purpose": self.purpose,
                "sealed_factory_identity_hash": self._sealed_factory.identity_hash,
                "backend_manifest_hash": self.backend_manifest_hash,
                "campaign_binding_hash": self.campaign_binding_hash,
                "controller_run_id": self.controller_run_id,
                "model_data_seal_hash": self.model_data_seal_hash,
                "runtime_logical_config_hash": self.runtime_logical_config_hash,
                "production_bundle_hash": self.production_bundle_hash,
                "initial_mask_hash": self.initial_mask_hash,
                "route_axis_hash": self.route_axis_hash,
                "train_population_hash": self.train_population_hash,
                "production_proposal_backend_identity_hash": (
                    self.production_proposal_backend_identity_hash
                ),
                "proposal_only": self.proposal_only,
                "production_authority": self.production_authority,
                "train_optimization_data_only": self.train_optimization_data_only,
                "terminal_outcomes_observed": self.terminal_outcomes_observed,
                "authority_inert": self.authority_inert,
                "candidate_or_mask_authority": self.candidate_or_mask_authority,
            },
            prefix=_CARRIER_PREFIX,
        )

    def materialize_generation_zero_chart(self, controller: Any) -> Any:
        """Build one detached dense chart without mutating controller authority."""

        if type(controller) is not V1ControllerState:
            raise TypeError(
                "diagnostic acquisition requires an exact modern-v1 controller"
            )
        trajectory = getattr(controller, "accepted_trajectory", None)
        if (
            getattr(controller, "phase", None) != "needs_acquisition"
            or getattr(controller, "acquisition", None) is not None
            or getattr(controller, "acquisition_generation_index", None) != 0
            or getattr(controller, "controller_run_id", None) != self.controller_run_id
            or controller.binding.binding_hash != self.campaign_binding_hash
            or controller.initial_mask.mask_hash != self.initial_mask_hash
            or controller.current_mask.mask_hash != self.initial_mask_hash
            or controller.current_mask.retained_count
            != controller.current_mask.route_count
            or getattr(controller, "queue_cursor", None) != 0
            or getattr(controller, "active_candidate", None) is not None
            or trajectory is None
            or tuple(trajectory.step_ledgers)
            or tuple(trajectory.step_decisions)
            or tuple(getattr(controller, "attempt_receipts", ()))
            or tuple(getattr(controller, "transition_receipts", ()))
            or tuple(getattr(controller, "acquisition_receipt_hashes", ()))
        ):
            raise PermissionError(
                "diagnostic acquisition requires the pristine generation-zero "
                "controller"
            )
        before_state_hash = controller.state_hash
        acquisition = self._sealed_factory.callable_value(controller)
        if type(acquisition) is not V1RelinearizedAcquisition:
            raise TypeError(
                "diagnostic acquisition returned a noncanonical chart type"
            )
        if controller.state_hash != before_state_hash:
            raise RuntimeError("diagnostic acquisition mutated controller state")
        if (
            acquisition.generation_index != 0
            or acquisition.current_mask.mask_hash != self.initial_mask_hash
        ):
            raise RuntimeError("diagnostic acquisition returned another chart")
        return acquisition


@dataclass(frozen=True, slots=True)
class V1AlexNetDiagnosticRuntimeView:
    """Explicit composition of staged runtime owners and inert diagnostics.

    This class deliberately has no attribute forwarding.  It is neither a
    subtype nor a structural replacement for :class:`V1ProductionRuntimeBundle`.
    """

    production_bundle: V1ProductionRuntimeBundle = field(repr=False, compare=False)
    acquisition: V1AlexNetDiagnosticAcquisitionCarrier
    protocol: str = ALEXNET_DIAGNOSTIC_PROTOCOL
    production_authority: bool = False
    schema: str = ALEXNET_DIAGNOSTIC_RUNTIME_VIEW_SCHEMA

    def __post_init__(self) -> None:
        if type(self.production_bundle) is not V1ProductionRuntimeBundle:
            raise TypeError(
                "diagnostic runtime view requires an exact production bundle"
            )
        if type(self.acquisition) is not V1AlexNetDiagnosticAcquisitionCarrier:
            raise TypeError(
                "diagnostic runtime view requires an exact acquisition carrier"
            )
        bundle = self.production_bundle
        proposal = bundle.proposal_backend_seal
        if (
            bundle.manifest.backend_kind != "alexnet_production"
            or bundle.manifest.proposal_protocol != "staged_train_only_proposal_v1"
            or bundle.acquisition_factory is not None
            or proposal is None
            or not bundle.has_staged_proposal_protocol
        ):
            raise PermissionError(
                "diagnostic runtime view requires a staged AlexNet production bundle"
            )
        metadata = dict(bundle.manifest.metadata)
        if metadata.get("initial_dynamic_acquisition_deferred") is not True:
            raise PermissionError(
                "diagnostic runtime view requires explicitly deferred FD/Q acquisition"
            )
        expected = {
            "backend_manifest_hash": bundle.manifest.backend_manifest_hash,
            "campaign_binding_hash": bundle.binding.binding_hash,
            "controller_run_id": bundle.manifest.controller_run_id,
            "model_data_seal_hash": bundle.manifest.model_data_seal_hash,
            "runtime_logical_config_hash": bundle.runtime_config.logical_identity_hash,
            "production_bundle_hash": bundle.bundle_hash,
            "initial_mask_hash": bundle.initial_mask.mask_hash,
            "route_axis_hash": bundle.initial_mask.route_axis_hash,
            "production_proposal_backend_identity_hash": proposal.identity_hash,
            "train_population_hash": proposal.train_population_hash,
        }
        for name, value in expected.items():
            if getattr(self.acquisition, name) != value:
                raise ValueError(f"diagnostic acquisition differs on {name}")
        terminal_owner = bundle.terminal_validator.callable_value
        if getattr(terminal_owner, "_observation", None) is not None:
            raise PermissionError(
                "diagnostic runtime view found opened terminal outcomes"
            )
        if (
            self.protocol != ALEXNET_DIAGNOSTIC_PROTOCOL
            or self.production_authority is not False
            or self.schema != ALEXNET_DIAGNOSTIC_RUNTIME_VIEW_SCHEMA
        ):
            raise ValueError("diagnostic runtime view crossed its authority boundary")

    @property
    def view_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "protocol": self.protocol,
                "production_bundle_hash": self.production_bundle.bundle_hash,
                "diagnostic_acquisition_identity_hash": self.acquisition.identity_hash,
                "production_authority": self.production_authority,
            },
            prefix=_VIEW_PREFIX,
        )

    @property
    def identity_receipt(self) -> MappingProxyType[str, Any]:
        """Return the immutable authority and source identity for artifacts."""

        acquisition = self.acquisition
        return MappingProxyType(
            {
                "schema": self.schema,
                "protocol": self.protocol,
                "purpose": acquisition.purpose,
                "diagnostic_runtime_view_hash": self.view_hash,
                "diagnostic_acquisition_identity_hash": acquisition.identity_hash,
                "diagnostic_factory_identity_hash": (
                    acquisition.factory_identity_hash
                ),
                "diagnostic_factory_callable_spec": (
                    acquisition.factory_callable_spec
                ),
                "diagnostic_factory_source_file_sha256": (
                    acquisition.factory_source_file_sha256
                ),
                "diagnostic_factory_configuration_hash": (
                    acquisition.factory_configuration_hash
                ),
                "production_bundle_hash": self.production_bundle.bundle_hash,
                "production_proposal_backend_identity_hash": (
                    acquisition.production_proposal_backend_identity_hash
                ),
                "train_population_hash": acquisition.train_population_hash,
                "proposal_only": acquisition.proposal_only,
                "production_authority": acquisition.production_authority,
                "train_optimization_data_only": (
                    acquisition.train_optimization_data_only
                ),
                "terminal_outcomes_observed": (
                    acquisition.terminal_outcomes_observed
                ),
                "authority_inert": acquisition.authority_inert,
                "candidate_or_mask_authority": (
                    acquisition.candidate_or_mask_authority
                ),
            }
        )


def bind_v1_alexnet_diagnostic_acquisition(
    *,
    production_bundle: V1ProductionRuntimeBundle,
    sealed_factory: V1ProductionCallableSeal,
) -> V1AlexNetDiagnosticAcquisitionCarrier:
    """Derive the diagnostic carrier only from exact staged runtime owners."""

    if type(production_bundle) is not V1ProductionRuntimeBundle:
        raise TypeError("diagnostic binding requires an exact production bundle")
    if (
        type(sealed_factory) is not V1ProductionCallableSeal
        or sealed_factory.role != "acquisition_factory"
    ):
        raise TypeError(
            "diagnostic binding requires an exact acquisition-factory seal"
        )
    proposal = production_bundle.proposal_backend_seal
    if proposal is None:
        raise PermissionError("diagnostic binding requires a staged proposal backend")
    factory = sealed_factory.callable_value
    population = getattr(factory, "population", None)
    population_hash = getattr(population, "population_hash", None)
    return V1AlexNetDiagnosticAcquisitionCarrier(
        _sealed_factory=sealed_factory,
        backend_manifest_hash=production_bundle.manifest.backend_manifest_hash,
        campaign_binding_hash=production_bundle.binding.binding_hash,
        controller_run_id=production_bundle.manifest.controller_run_id,
        model_data_seal_hash=production_bundle.manifest.model_data_seal_hash,
        runtime_logical_config_hash=(
            production_bundle.runtime_config.logical_identity_hash
        ),
        production_bundle_hash=production_bundle.bundle_hash,
        initial_mask_hash=production_bundle.initial_mask.mask_hash,
        route_axis_hash=production_bundle.initial_mask.route_axis_hash,
        train_population_hash=_nonempty(population_hash, "train_population_hash"),
        production_proposal_backend_identity_hash=proposal.identity_hash,
    )


def compose_v1_alexnet_diagnostic_runtime_view(
    *,
    production_bundle: V1ProductionRuntimeBundle,
    sealed_factory: V1ProductionCallableSeal,
) -> V1AlexNetDiagnosticRuntimeView:
    """Compose the exact, non-convertible diagnostic runtime boundary."""

    return V1AlexNetDiagnosticRuntimeView(
        production_bundle=production_bundle,
        acquisition=bind_v1_alexnet_diagnostic_acquisition(
            production_bundle=production_bundle,
            sealed_factory=sealed_factory,
        ),
    )


def materialize_v1_alexnet_diagnostic_runtime(
    *,
    admission: V1AlexNetBackendAdmission,
    binding: V1OptionACampaignBinding,
    artifact_root: str | Path,
) -> V1AlexNetDiagnosticRuntimeView:
    """Verify an explicitly admitted diagnostic factory and its staged owners."""

    if type(admission) is not V1AlexNetBackendAdmission:
        raise TypeError(
            "AlexNet diagnostic materialization requires an exact admission"
        )
    admission.validate_campaign(binding)
    verified = admission.verify_artifacts(artifact_root)
    factory = load_v1_alexnet_backend_factory(admission.backend_factory_spec)
    source_file = inspect.getsourcefile(factory)
    if source_file is None or not Path(source_file).is_file():
        raise V1AlexNetDependencyUnavailable(
            "AlexNet diagnostic backend factory source is unavailable for sealing"
        )
    if _sha256_file(Path(source_file)) != admission.backend_factory_source_hash:
        raise ValueError(
            "AlexNet diagnostic backend factory source differs from admission SHA-256"
        )
    view = factory(
        admission=admission,
        binding=binding,
        verified_artifacts=verified,
    )
    if type(view) is not V1AlexNetDiagnosticRuntimeView:
        raise TypeError(
            "AlexNet diagnostic factory did not return an exact diagnostic runtime view"
        )
    bundle = view.production_bundle
    manifest = bundle.manifest
    checks = {
        "backend_kind": "alexnet_production",
        "backend_manifest_hash": admission.admission_hash,
        "study_contract_hash": next(
            item.sha256 for item in admission.artifacts if item.role == "study_contract"
        ),
        "measurement_contract_hash": next(
            item.sha256
            for item in admission.artifacts
            if item.role == "measurement_plan"
        ),
        "model_data_seal_hash": admission.model_seal_hash,
        "controller_run_id": admission.controller_run_id,
    }
    for name, expected in checks.items():
        if getattr(manifest, name) != expected:
            raise ValueError(
                f"AlexNet diagnostic runtime differs from admission field {name}"
            )
    if bundle.binding.binding_hash != binding.binding_hash:
        raise ValueError("AlexNet diagnostic runtime returned another campaign binding")
    if (
        manifest.direct_rule_over_v0_adapter
        or manifest.retired_trust_walk
        or not manifest.genuine_partition_free_controller
        or not manifest.selector_fail_closed
    ):
        raise ValueError("AlexNet diagnostic runtime crossed production invariants")
    return view


__all__ = [
    "ALEXNET_DIAGNOSTIC_CARRIER_SCHEMA",
    "ALEXNET_DIAGNOSTIC_PROTOCOL",
    "ALEXNET_DIAGNOSTIC_PURPOSE",
    "ALEXNET_DIAGNOSTIC_RUNTIME_VIEW_SCHEMA",
    "ALEXNET_PRODUCTION_FACTORY_SPEC",
    "ALEXNET_SAMPLED_DISCOVERY_ADMISSION_RECEIPT_SCHEMA",
    "ALEXNET_SAMPLED_DISCOVERY_ADMISSION_SCHEMA",
    "ALEXNET_SAMPLED_DISCOVERY_DIAGNOSTIC_FACTORY_SPEC",
    "V1AlexNetDiagnosticAcquisitionCarrier",
    "V1AlexNetDiagnosticRuntimeView",
    "bind_v1_alexnet_diagnostic_acquisition",
    "compose_v1_alexnet_diagnostic_runtime_view",
    "derive_v1_alexnet_sampled_discovery_admission",
    "materialize_v1_alexnet_diagnostic_runtime",
    "validate_v1_alexnet_sampled_discovery_admission",
]
