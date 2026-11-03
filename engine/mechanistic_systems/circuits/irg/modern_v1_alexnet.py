"""Sealed AlexNet backend admission for genuine modern-v1.

The production AlexNet substrate is intentionally separated from the v0
layerwise walker.  This module verifies a complete source/model/data/
calibration artifact set, loads one explicitly named backend factory, and
admits only a :class:`V1ProductionRuntimeBundle` whose manifest identifies the
genuine partition-free controller.

It does not make missing cluster artifacts appear, infer paths from environment
variables, or turn an unrun launcher into evidence.  Dependency and artifact
failures occur before an output directory or CUDA process is created.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from mechanistic_systems.core.determinism import (
    configure_deterministic_backend,
    deterministic_backend_record,
)
from .contracts import content_hash
from .modern_v1_campaign_contract import V1OptionACampaignBinding
from .modern_v1_production import (
    V1ProductionProposalBackendSeal,
    V1ProductionRuntimeBundle,
)

_ARTIFACT_SCHEMA: Final = "partition_free_v1_alexnet_sealed_artifact_v1"
_ADMISSION_SCHEMA: Final = "partition_free_v1_alexnet_backend_admission_v1"
_CONSTRUCTION_SCHEMA: Final = "partition_free_v1_alexnet_construction_receipt_v1"
_REQUIRED_ARTIFACT_ROLES: Final = frozenset(
    {
        "study_contract",
        "measurement_plan",
        "model_load_receipt",
        "model_source_replica",
        "bound_circuit_plan",
        "precision_profile",
        "optimized_context_bank",
        "dense_reference",
        "extraction_calibration",
        "fq_acquisition_registration",
        "terminal_validation_bank",
        "environment_lock",
        "source_manifest",
    }
)
_FORBIDDEN_FACTORY_FRAGMENTS: Final = (
    "registered_resnet18_pilot",
    "trust_walk",
    "native_sampled",
    "native_continuous",
    "modern_v1_validation_20260813",
)
_FACTORY_PATTERN: Final = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
)

AlexNetBackendFactory = Callable[..., V1ProductionRuntimeBundle]


class V1AlexNetDependencyUnavailable(RuntimeError):
    """A sealed backend factory or one of its dependencies cannot be imported."""


class V1AlexNetArtifactUnavailable(FileNotFoundError):
    """A required sealed AlexNet artifact is absent from the admitted root."""


@dataclass(frozen=True, slots=True)
class V1AlexNetConstructionReceipt:
    """Real torchvision construction smoke; never checkpoint or launch evidence."""

    torch_version: str
    torchvision_version: str
    device: str
    parameter_layout_hash: str
    state_dict_hash: str
    input_hash: str
    output_hash: str
    output_shape: tuple[int, ...]
    output_finite: bool
    terminal_outcomes_opened: bool = False
    claim_execution_certified: bool = False
    schema: str = _CONSTRUCTION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "torch_version",
            "torchvision_version",
            "device",
            "parameter_layout_hash",
            "state_dict_hash",
            "input_hash",
            "output_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        shape = tuple(int(item) for item in self.output_shape)
        if not shape or any(item <= 0 for item in shape):
            raise ValueError("AlexNet construction output shape is invalid")
        object.__setattr__(self, "output_shape", shape)
        if self.output_finite is not True:
            raise ValueError("AlexNet construction output is nonfinite")
        if self.terminal_outcomes_opened is not False or self.claim_execution_certified is not False:
            raise ValueError("AlexNet construction smoke crossed an authority boundary")
        if self.schema != _CONSTRUCTION_SCHEMA:
            raise ValueError("unsupported AlexNet construction schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-alexnet-construction-")

    def to_dict(self) -> dict[str, Any]:
        return {
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "device": self.device,
            "parameter_layout_hash": self.parameter_layout_hash,
            "state_dict_hash": self.state_dict_hash,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "output_shape": list(self.output_shape),
            "output_finite": self.output_finite,
            "terminal_outcomes_opened": self.terminal_outcomes_opened,
            "claim_execution_certified": self.claim_execution_certified,
            "schema": self.schema,
            "receipt_hash": self.receipt_hash,
            "determinism": deterministic_backend_record(),
        }


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha256(value: object, label: str) -> str:
    result = _nonempty(value, label).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _freeze(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{label} keys must be nonempty strings")
            frozen[key] = _freeze(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, f"{label}[{index}]") for index, item in enumerate(value))
    if isinstance(value, float):
        if not (-float("inf") < value < float("inf")):
            raise ValueError(f"{label} contains a nonfinite float")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{label} contains unsupported value {type(value)!r}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_bytes_hash(items: Sequence[tuple[str, Any]], *, prefix: str) -> str:
    digest = hashlib.sha256()
    descriptor: list[dict[str, Any]] = []
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        array = value.numpy()
        payload = memoryview(array).cast("B")
        digest.update(name.encode("utf-8"))
        digest.update(payload)
        descriptor.append(
            {
                "name": name,
                "shape": tuple(int(item) for item in value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return content_hash(
        {"descriptor": descriptor, "combined_sha256": digest.hexdigest()},
        prefix=prefix,
    )


def run_alexnet_construction_smoke(
    *,
    device: str = "cpu",
    seed: int = 0,
    batch_size: int = 1,
) -> V1AlexNetConstructionReceipt:
    """Construct and execute an actual torchvision AlexNet with no downloads."""

    configure_deterministic_backend()
    try:
        import torch
        import torchvision
        from torchvision.models import alexnet
    except (ImportError, ModuleNotFoundError) as exc:
        raise V1AlexNetDependencyUnavailable(
            f"torchvision AlexNet construction dependency is unavailable: {exc}"
        ) from exc
    if type(seed) is not int:
        raise TypeError("AlexNet construction seed must be int")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("AlexNet construction batch_size must be positive")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested AlexNet CUDA construction but CUDA is unavailable")
    torch.manual_seed(seed)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = alexnet(weights=None).eval().to(target)
    state_items = tuple(sorted(model.state_dict().items()))
    layout_hash = content_hash(
        tuple(
            (name, tuple(int(item) for item in tensor.shape), str(tensor.dtype))
            for name, tensor in state_items
        ),
        prefix="partition-free-v1-alexnet-parameter-layout-",
    )
    state_hash = _tensor_bytes_hash(
        state_items,
        prefix="partition-free-v1-alexnet-state-",
    )
    element_count = batch_size * 3 * 224 * 224
    inputs = torch.linspace(-1.0, 1.0, steps=element_count, dtype=torch.float32)
    inputs = inputs.reshape(batch_size, 3, 224, 224).to(target)
    input_hash = _tensor_bytes_hash(
        (("input", inputs),),
        prefix="partition-free-v1-alexnet-input-",
    )
    with torch.inference_mode():
        outputs = model(inputs)
    output_hash = _tensor_bytes_hash(
        (("output", outputs),),
        prefix="partition-free-v1-alexnet-output-",
    )
    return V1AlexNetConstructionReceipt(
        torch_version=str(torch.__version__),
        torchvision_version=str(torchvision.__version__),
        device=str(target),
        parameter_layout_hash=layout_hash,
        state_dict_hash=state_hash,
        input_hash=input_hash,
        output_hash=output_hash,
        output_shape=tuple(int(item) for item in outputs.shape),
        output_finite=bool(torch.isfinite(outputs).all().item()),
    )


@dataclass(frozen=True, slots=True)
class V1AlexNetSealedArtifact:
    role: str
    relative_path: str
    sha256: str
    schema: str = _ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _nonempty(self.role, "artifact role"))
        raw = _nonempty(self.relative_path, "artifact relative_path")
        normalized = PurePosixPath(raw)
        if normalized.is_absolute() or ".." in normalized.parts or str(normalized) != raw:
            raise ValueError("AlexNet artifact paths must be canonical relative POSIX paths")
        object.__setattr__(self, "relative_path", raw)
        object.__setattr__(self, "sha256", _sha256(self.sha256, "artifact sha256"))
        if self.schema != _ARTIFACT_SCHEMA:
            raise ValueError("unsupported AlexNet sealed-artifact schema")

    @property
    def artifact_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-alexnet-artifact-")

    def verify(self, root: Path) -> Path:
        source = (root / self.relative_path).resolve()
        resolved_root = root.resolve()
        try:
            source.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("sealed artifact escapes the admitted root") from exc
        if not source.is_file():
            raise V1AlexNetArtifactUnavailable(
                f"required AlexNet artifact {self.role!r} is unavailable at {source}"
            )
        observed = _file_sha256(source)
        if observed != self.sha256:
            raise ValueError(
                f"AlexNet artifact {self.role!r} SHA-256 mismatch: "
                f"expected {self.sha256}, observed {observed}"
            )
        return source


@dataclass(frozen=True, slots=True)
class V1AlexNetBackendAdmission:
    """Complete pre-import identity for one real AlexNet development run."""

    admission_id: str
    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    model_seal_hash: str
    query_id: str
    referent_id: str
    route_axis_hash: str
    context_population_hash: str
    dense_reference_hash: str
    calibration_identity: str
    terminal_validation_population_hash: str
    backend_factory_spec: str
    backend_factory_source_hash: str
    artifacts: tuple[V1AlexNetSealedArtifact, ...]
    production_geometric_selector_source_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    architecture_family: str = "alexnet"
    controller_identity: str = "partition_free_v1_production_bundle_v1"
    global_queue: bool = True
    refresh_after_every_accept: bool = True
    direct_rule_over_v0_adapter: bool = False
    retired_trust_walk: bool = False
    launch_authority: bool = False
    claim_execution_certified: bool = False
    schema: str = _ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "admission_id",
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
            "backend_factory_source_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.production_geometric_selector_source_hash is not None:
            object.__setattr__(
                self,
                "production_geometric_selector_source_hash",
                _sha256(
                    self.production_geometric_selector_source_hash,
                    "production_geometric_selector_source_hash",
                ),
            )
        spec = _nonempty(self.backend_factory_spec, "backend_factory_spec")
        if _FACTORY_PATTERN.fullmatch(spec) is None:
            raise ValueError("backend_factory_spec must be canonical module:function")
        lowered = spec.lower()
        if any(fragment in lowered for fragment in _FORBIDDEN_FACTORY_FRAGMENTS):
            raise ValueError("AlexNet admission cannot route through a retired/v0 walker surface")
        object.__setattr__(self, "backend_factory_spec", spec)
        artifacts = tuple(self.artifacts)
        if any(type(item) is not V1AlexNetSealedArtifact for item in artifacts):
            raise TypeError("AlexNet admission requires exact sealed-artifact objects")
        roles = tuple(item.role for item in artifacts)
        if len(set(roles)) != len(roles):
            raise ValueError("AlexNet sealed-artifact roles must be unique")
        missing = _REQUIRED_ARTIFACT_ROLES.difference(roles)
        extra = set(roles).difference(_REQUIRED_ARTIFACT_ROLES)
        if missing or extra:
            raise ValueError(
                f"AlexNet artifact role set differs; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        object.__setattr__(self, "artifacts", tuple(sorted(artifacts, key=lambda item: item.role)))
        if (
            self.schema != _ADMISSION_SCHEMA
            or self.architecture_family != "alexnet"
            or self.controller_identity != "partition_free_v1_production_bundle_v1"
            or self.global_queue is not True
            or self.refresh_after_every_accept is not True
            or self.direct_rule_over_v0_adapter is not False
            or self.retired_trust_walk is not False
            or self.launch_authority is not False
            or self.claim_execution_certified is not False
        ):
            raise ValueError("AlexNet admission crossed its partition-free authority boundary")
        object.__setattr__(self, "metadata", _freeze(self.metadata, "AlexNet admission metadata"))
        if self.proposal_protocol not in {'staged_train_only_proposal_v1', 'operator_binary_d_c_v1'}:
            raise ValueError('AlexNet admission has an unsupported proposal protocol')
        if self.proposal_protocol == 'operator_binary_d_c_v1':
            _nonempty(self.metadata.get('operator_factory_registration_hash'),
                'operator_factory_registration_hash')

    @property
    def proposal_protocol(self) -> str:
        # Absent metadata preserves historical admission bytes and hashes.
        return self.metadata.get('proposal_protocol', 'staged_train_only_proposal_v1')

    @property
    def admission_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-alexnet-admission-")

    def validate_campaign(self, binding: V1OptionACampaignBinding) -> None:
        if type(binding) is not V1OptionACampaignBinding:
            raise TypeError("AlexNet admission requires an exact campaign binding")
        contract = binding.contract
        checks = {
            "campaign_contract_hash": contract.campaign_contract_hash,
            "campaign_binding_hash": binding.binding_hash,
            "model_seal_hash": contract.model_seal_hash,
            "query_id": contract.query_id,
            "referent_id": contract.referent_id,
            "route_axis_hash": contract.route_axis_hash,
            "context_population_hash": contract.context_population_hash,
            "dense_reference_hash": contract.dense_reference_hash,
            "calibration_identity": contract.calibration_identity,
            "terminal_validation_population_hash": (
                contract.terminal_validation_population_hash
            ),
        }
        for name, expected in checks.items():
            if getattr(self, name) != expected:
                raise ValueError(f"AlexNet admission differs from campaign field {name}")

    def verify_artifacts(self, root: str | Path) -> Mapping[str, Path]:
        base = Path(root)
        if not base.is_dir():
            raise V1AlexNetArtifactUnavailable(
                f"AlexNet admitted artifact root is unavailable: {base}"
            )
        verified = {item.role: item.verify(base) for item in self.artifacts}
        return MappingProxyType(verified)

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "admission_id": self.admission_id,
            "admission_hash": self.admission_hash,
            "campaign_contract_hash": self.campaign_contract_hash,
            "campaign_binding_hash": self.campaign_binding_hash,
            "controller_run_id": self.controller_run_id,
            "model_seal_hash": self.model_seal_hash,
            "query_id": self.query_id,
            "referent_id": self.referent_id,
            "route_axis_hash": self.route_axis_hash,
            "context_population_hash": self.context_population_hash,
            "dense_reference_hash": self.dense_reference_hash,
            "calibration_identity": self.calibration_identity,
            "terminal_validation_population_hash": self.terminal_validation_population_hash,
            "backend_factory_spec": self.backend_factory_spec,
            "backend_factory_source_hash": self.backend_factory_source_hash,
            "production_geometric_selector_source_hash": (
                self.production_geometric_selector_source_hash
            ),
            "artifacts": [
                {
                    "schema": item.schema,
                    "role": item.role,
                    "relative_path": item.relative_path,
                    "sha256": item.sha256,
                    "artifact_hash": item.artifact_hash,
                }
                for item in self.artifacts
            ],
            "metadata": dict(self.metadata),
            "architecture_family": self.architecture_family,
            "controller_identity": self.controller_identity,
            "global_queue": True,
            "refresh_after_every_accept": True,
            "direct_rule_over_v0_adapter": False,
            "retired_trust_walk": False,
            "launch_authority": False,
            "claim_execution_certified": False,
        }


def load_v1_alexnet_admission(path: str | Path) -> V1AlexNetBackendAdmission:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("AlexNet admission JSON must contain an object")
    artifacts_raw = payload.pop("artifacts", None)
    payload.pop("admission_hash", None)
    if not isinstance(artifacts_raw, list):
        raise TypeError("AlexNet admission JSON requires an artifacts array")
    artifacts: list[V1AlexNetSealedArtifact] = []
    for item in artifacts_raw:
        if not isinstance(item, dict):
            raise TypeError("AlexNet artifact entries must be objects")
        values = dict(item)
        values.pop("artifact_hash", None)
        artifacts.append(V1AlexNetSealedArtifact(**values))
    admission = V1AlexNetBackendAdmission(artifacts=tuple(artifacts), **payload)
    declared = json.loads(source.read_text(encoding="utf-8")).get("admission_hash")
    if declared is not None and declared != admission.admission_hash:
        raise ValueError("AlexNet admission JSON content differs from its declared hash")
    return admission


def load_v1_alexnet_backend_factory(spec: str) -> AlexNetBackendFactory:
    canonical = _nonempty(spec, "backend factory spec")
    if _FACTORY_PATTERN.fullmatch(canonical) is None:
        raise ValueError("backend factory spec must be canonical module:function")
    lowered = canonical.lower()
    if any(fragment in lowered for fragment in _FORBIDDEN_FACTORY_FRAGMENTS):
        raise ValueError("backend factory cannot use a v0/retired control-flow module")
    module_name, function_name = canonical.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise V1AlexNetDependencyUnavailable(
            f"AlexNet backend factory dependency is unavailable: {module_name}: {exc}"
        ) from exc
    value = getattr(module, function_name, None)
    if not callable(value):
        raise V1AlexNetDependencyUnavailable(
            f"AlexNet backend factory is not callable: {canonical}"
        )
    return value


def materialize_v1_alexnet_bundle(
    *,
    admission: V1AlexNetBackendAdmission,
    binding: V1OptionACampaignBinding,
    artifact_root: str | Path,
) -> V1ProductionRuntimeBundle:
    """Verify sealed files, import the explicit factory, and validate its bundle."""

    if type(admission) is not V1AlexNetBackendAdmission:
        raise TypeError("AlexNet materialization requires an exact admission")
    admission.validate_campaign(binding)
    verified = admission.verify_artifacts(artifact_root)
    operator_registration = None
    if admission.proposal_protocol == 'operator_binary_d_c_v1':
        from .modern_v1_operator_registration import admitted_operator_registration
        raw_environment = Path(verified['environment_lock']).read_bytes()
        expected_environment = next(a.sha256 for a in admission.artifacts if a.role == 'environment_lock')
        if hashlib.sha256(raw_environment).hexdigest() != expected_environment:
            raise ValueError('operator environment bytes changed after artifact verification')
        operator_registration = admitted_operator_registration(admission=admission,
            environment=json.loads(raw_environment))
    factory = load_v1_alexnet_backend_factory(admission.backend_factory_spec)
    source_file = inspect.getsourcefile(factory)
    if source_file is None or not Path(source_file).is_file():
        raise V1AlexNetDependencyUnavailable(
            "AlexNet backend factory source file is unavailable for sealing"
        )
    observed_factory_hash = _file_sha256(Path(source_file))
    if observed_factory_hash != admission.backend_factory_source_hash:
        raise ValueError(
            "AlexNet backend factory source differs from its admitted SHA-256"
        )
    bundle = factory(
        admission=admission,
        binding=binding,
        verified_artifacts=verified,
    )
    if type(bundle) is not V1ProductionRuntimeBundle:
        raise TypeError("AlexNet backend factory did not return an exact production bundle")
    manifest = bundle.manifest
    checks = {
        "backend_kind": "alexnet_production",
        "backend_manifest_hash": admission.admission_hash,
        "study_contract_hash": next(
            item.sha256 for item in admission.artifacts if item.role == "study_contract"
        ),
        "measurement_contract_hash": next(
            item.sha256 for item in admission.artifacts if item.role == "measurement_plan"
        ),
        "model_data_seal_hash": admission.model_seal_hash,
        "controller_run_id": admission.controller_run_id,
    }
    for name, expected in checks.items():
        if getattr(manifest, name) != expected:
            raise ValueError(f"AlexNet backend bundle differs from admission field {name}")
    if bundle.binding.binding_hash != binding.binding_hash:
        raise ValueError("AlexNet backend factory returned another campaign binding")
    if manifest.proposal_protocol != admission.proposal_protocol:
        raise ValueError('AlexNet bundle differs from the admitted proposal protocol')
    proposal = bundle.proposal_backend_seal
    if admission.proposal_protocol == 'operator_binary_d_c_v1':
        from .modern_v1_production import _bound_operator_factory
        operator = _bound_operator_factory(bundle)
        if (operator is None or not bundle.has_paired_bce_oracle_protocol
                or bundle.has_staged_proposal_protocol or proposal is not None):
            raise ValueError('AlexNet operator admission requires the exact source-bound native producer')
        registration = operator._model.sources.fq_registration
        if (registration.expected_model_identity_hash != admission.model_seal_hash
                or registration.campaign_contract_hash != admission.campaign_contract_hash):
            raise ValueError('AlexNet operator source registration differs from model/campaign admission')
        if (registration.maximum_proposal_generations != binding.contract.maximum_step_count
                or registration.maximum_oracle_attempts != binding.contract.maximum_candidate_count
                or registration.oracle_familywise_alpha != binding.contract.campaign_wide_alpha):
            raise ValueError('AlexNet operator F/Q horizons or oracle reserve differ from the campaign')
        from .modern_v1_operator_registration import validate_registered_operator_bundle
        validate_registered_operator_bundle(registration=operator_registration, bundle=bundle)
        from .modern_v1_operator_dataset import register_operator_dataset, validate_operator_fq_dataset
        dataset = register_operator_dataset(context_source=operator._model.sources.context_source,
            context_axis=operator._model.data_plane.context_axis,
            target_class_id=operator._model.data_plane.target_class_id)
        validate_operator_fq_dataset(dataset=dataset, fq_registration=registration)
    elif (
        manifest.proposal_protocol != "staged_train_only_proposal_v1"
        or bundle.acquisition_factory is not None
        or type(proposal) is not V1ProductionProposalBackendSeal
        or not bundle.has_staged_proposal_protocol
        or proposal.backend_kind != "alexnet_production"
        or manifest.proposal_backend_identity_hash != proposal.identity_hash
    ):
        raise ValueError(
            "AlexNet admission requires one exact staged train-only proposal "
            "backend and forbids one-shot acquisition authority"
        )
    if (
        manifest.direct_rule_over_v0_adapter
        or manifest.retired_trust_walk
        or not manifest.genuine_partition_free_controller
        or not manifest.selector_fail_closed
    ):
        raise ValueError("AlexNet backend factory returned a prohibited control-flow identity")
    return bundle


__all__ = [
    "AlexNetBackendFactory",
    "V1AlexNetArtifactUnavailable",
    "V1AlexNetBackendAdmission",
    "V1AlexNetConstructionReceipt",
    "V1AlexNetDependencyUnavailable",
    "V1AlexNetSealedArtifact",
    "load_v1_alexnet_admission",
    "load_v1_alexnet_backend_factory",
    "materialize_v1_alexnet_bundle",
    "run_alexnet_construction_smoke",
]
