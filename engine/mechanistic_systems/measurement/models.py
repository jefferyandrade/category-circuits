"""Registered PyTorch model loading with exact checkpoint and topology identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mechanistic_systems.core.determinism import (
    configure_deterministic_backend,
    deterministic_backend_record,
)
from circuitflow.contracts import RouteCoordinateSchema
from contracts.hashing import sha256_file, stable_digest
from mechanistic_systems.core.architectures import (
    BoundTorchArchitecture,
    canonical_torch_architecture_spec,
)
from mechanistic_systems.contracts import StudySpec
from mechanistic_systems.input_operators import (
    canonical_imagenet_input_operator_contract,
)

from .contexts import ImagePreprocessingSpec

try:  # Optional for import smoke; required by executable model loading.
    import torch
    import torchvision.models as tv_models
except Exception:  # pragma: no cover - environment dependent.
    torch = tv_models = None  # type: ignore[assignment]


MODEL_LOAD_SCHEMA = "mechanistic_registered_torch_model_v1"
CHECKPOINT_QUALIFIED_SOURCE_SCHEMA = (
    "mechanistic_checkpoint_qualified_model_source_v1"
)
ROBUSTBENCH_SALMAN2020DO_R50_MODEL_ID = (
    "robustbench.salman2020do_r50.conv_kernel_weight.imagenet_linf_4_255_v1"
)


def _is_sha256(value: str) -> bool:
    normalized = str(value)
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


def _require_torchvision() -> None:
    if torch is None or tv_models is None:
        raise RuntimeError("registered model loading requires PyTorch and torchvision")


@dataclass(frozen=True)
class CheckpointQualifiedModelSource:
    """Immutable release, artifact, and input-operator identity for one source."""

    upstream_model_id: str
    release_id: str
    registry_uri: str
    registry_sha256: str
    metadata_uri: str
    metadata_sha256: str
    artifact_uri: str
    artifact_id: str
    artifact_filename: str
    artifact_size_bytes: int
    artifact_last_modified_utc: str
    checkpoint_sha256: str
    checkpoint_sha256_authority: str
    upstream_checksum_published: bool
    state_dict_sha256: str
    dataset_id: str
    threat_model: str
    threat_epsilon: str
    upstream_preprocessing_id: str
    execution_preprocessing_sha256: str
    normalization_placement: str

    def validate(self) -> None:
        required = (
            self.upstream_model_id,
            self.release_id,
            self.registry_uri,
            self.metadata_uri,
            self.artifact_uri,
            self.artifact_id,
            self.artifact_filename,
            self.artifact_last_modified_utc,
            self.checkpoint_sha256_authority,
            self.dataset_id,
            self.threat_model,
            self.threat_epsilon,
            self.upstream_preprocessing_id,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("checkpoint-qualified source identity is incomplete")
        for name, value in (
            ("registry_sha256", self.registry_sha256),
            ("metadata_sha256", self.metadata_sha256),
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("state_dict_sha256", self.state_dict_sha256),
            (
                "execution_preprocessing_sha256",
                self.execution_preprocessing_sha256,
            ),
        ):
            if not _is_sha256(value):
                raise ValueError(
                    f"checkpoint-qualified source {name} is not a SHA-256 digest"
                )
        if int(self.artifact_size_bytes) <= 0:
            raise ValueError("checkpoint-qualified artifact size must be positive")
        if self.normalization_placement != "external_input_operator_only":
            raise ValueError(
                "checkpoint-qualified normalization must have one external owner"
            )
        preprocessing = ImagePreprocessingSpec.from_mapping(
            canonical_imagenet_input_operator_contract()
        )
        if preprocessing.content_hash != self.execution_preprocessing_sha256:
            raise ValueError(
                "checkpoint-qualified execution preprocessing identity drifted"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_name": CHECKPOINT_QUALIFIED_SOURCE_SCHEMA,
            "upstream_model_id": self.upstream_model_id,
            "release_id": self.release_id,
            "registry_uri": self.registry_uri,
            "registry_sha256": self.registry_sha256,
            "metadata_uri": self.metadata_uri,
            "metadata_sha256": self.metadata_sha256,
            "artifact_uri": self.artifact_uri,
            "artifact_id": self.artifact_id,
            "artifact_filename": self.artifact_filename,
            "artifact_size_bytes": int(self.artifact_size_bytes),
            "artifact_last_modified_utc": self.artifact_last_modified_utc,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_sha256_authority": self.checkpoint_sha256_authority,
            "upstream_checksum_published": bool(
                self.upstream_checksum_published
            ),
            "state_dict_sha256": self.state_dict_sha256,
            "dataset_id": self.dataset_id,
            "threat_model": self.threat_model,
            "threat_epsilon": self.threat_epsilon,
            "upstream_preprocessing_id": self.upstream_preprocessing_id,
            "execution_preprocessing": (
                canonical_imagenet_input_operator_contract()
            ),
            "execution_preprocessing_sha256": (
                self.execution_preprocessing_sha256
            ),
            "normalization_placement": self.normalization_placement,
        }

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
    ) -> CheckpointQualifiedModelSource:
        if str(payload.get("schema_name", "")) != (
            CHECKPOINT_QUALIFIED_SOURCE_SCHEMA
        ):
            raise ValueError("unsupported checkpoint-qualified source schema")
        declared_preprocessing = ImagePreprocessingSpec.from_mapping(
            dict(payload.get("execution_preprocessing", {}) or {})
        )
        expected_preprocessing = ImagePreprocessingSpec.from_mapping(
            canonical_imagenet_input_operator_contract()
        )
        if declared_preprocessing.to_dict() != expected_preprocessing.to_dict():
            raise ValueError(
                "checkpoint-qualified source carries a different execution operator"
            )
        result = cls(
            upstream_model_id=str(payload.get("upstream_model_id", "")),
            release_id=str(payload.get("release_id", "")),
            registry_uri=str(payload.get("registry_uri", "")),
            registry_sha256=str(payload.get("registry_sha256", "")),
            metadata_uri=str(payload.get("metadata_uri", "")),
            metadata_sha256=str(payload.get("metadata_sha256", "")),
            artifact_uri=str(payload.get("artifact_uri", "")),
            artifact_id=str(payload.get("artifact_id", "")),
            artifact_filename=str(payload.get("artifact_filename", "")),
            artifact_size_bytes=int(payload.get("artifact_size_bytes", 0)),
            artifact_last_modified_utc=str(
                payload.get("artifact_last_modified_utc", "")
            ),
            checkpoint_sha256=str(payload.get("checkpoint_sha256", "")),
            checkpoint_sha256_authority=str(
                payload.get("checkpoint_sha256_authority", "")
            ),
            upstream_checksum_published=bool(
                payload.get("upstream_checksum_published", False)
            ),
            state_dict_sha256=str(payload.get("state_dict_sha256", "")),
            dataset_id=str(payload.get("dataset_id", "")),
            threat_model=str(payload.get("threat_model", "")),
            threat_epsilon=str(payload.get("threat_epsilon", "")),
            upstream_preprocessing_id=str(
                payload.get("upstream_preprocessing_id", "")
            ),
            execution_preprocessing_sha256=str(
                payload.get("execution_preprocessing_sha256", "")
            ),
            normalization_placement=str(
                payload.get("normalization_placement", "")
            ),
        )
        result.validate()
        if (
            declared_preprocessing.content_hash
            != result.execution_preprocessing_sha256
        ):
            raise ValueError(
                "checkpoint-qualified execution operator hash differs"
            )
        return result

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())


ROBUSTBENCH_SALMAN2020DO_R50_SOURCE = CheckpointQualifiedModelSource(
    upstream_model_id="Salman2020Do_R50",
    release_id=(
        "robustbench/v1.1@37bdaf46628bf66658e40bf88a46fe9bd48b8e98"
    ),
    registry_uri=(
        "https://github.com/RobustBench/robustbench/blob/"
        "37bdaf46628bf66658e40bf88a46fe9bd48b8e98/"
        "robustbench/model_zoo/imagenet.py"
    ),
    registry_sha256=(
        "8397555c0f69a648b5f8ffb8946100424479e4cfbd2ad10af44dd6e8e0bd5d67"
    ),
    metadata_uri=(
        "https://github.com/RobustBench/robustbench/blob/"
        "37bdaf46628bf66658e40bf88a46fe9bd48b8e98/"
        "model_info/imagenet/Linf/Salman2020Do_R50.json"
    ),
    metadata_sha256=(
        "1be77d458c86aaa698170aeb63d8b84170467defa93b305c6d1d8b78150ee4d4"
    ),
    artifact_uri=(
        "https://drive.usercontent.google.com/download?"
        "id=1TmT5oGa1UvVjM3d-XeSj_XmKqBNRUg8r&export=download"
    ),
    artifact_id="1TmT5oGa1UvVjM3d-XeSj_XmKqBNRUg8r",
    artifact_filename="Salman2020Do_R50.pt",
    artifact_size_bytes=102_546_991,
    artifact_last_modified_utc="2021-08-23T17:01:02Z",
    checkpoint_sha256=(
        "1835a5bbe5e31ef35915889dc949b96528bed513205bb9ec1cac80c32c84ce46"
    ),
    checkpoint_sha256_authority=(
        "local_full_byte_sha256_bound_to_official_object_id_filename_size_"
        "and_last_modified"
    ),
    upstream_checksum_published=False,
    state_dict_sha256=(
        "6b91c4f06576fe354277a5f36fe6ba3f999bf45418f404422cc6f3001465aea2"
    ),
    dataset_id="imagenet1k",
    threat_model="linf",
    threat_epsilon="4/255",
    upstream_preprocessing_id=(
        "robustbench.Res256Crop224+imagenet_mean_std_normalization.v1"
    ),
    execution_preprocessing_sha256=(
        "cd97b713422eb8f2f5622f31efa53b70eee8a3b339edc9ffc1250ee4048bb09e"
    ),
    normalization_placement="external_input_operator_only",
)


@dataclass(frozen=True)
class RegisteredModelDefinition:
    model_id: str
    architecture_family: str
    builder_name: str
    source_kind: str
    weights_enum_class: str = ""
    weights_member: str = ""
    num_classes: int = 1000
    checkpoint_source: CheckpointQualifiedModelSource | None = None
    official_checkpoint_sha256: str = ""

    def validate(self) -> None:
        if not self.model_id or not self.architecture_family or not self.builder_name:
            raise ValueError("registered model identity must be nonempty")
        if self.source_kind not in {
            "torchvision_weights",
            "explicit_state_dict",
            "checkpoint_qualified_state_dict",
            "seeded_untrained",
        }:
            raise ValueError("unsupported registered model source kind")
        if self.source_kind == "torchvision_weights" and (
            not self.weights_enum_class or not self.weights_member
        ):
            raise ValueError("torchvision-weight model requires an explicit enum member")
        if self.official_checkpoint_sha256 and not _is_sha256(
            self.official_checkpoint_sha256
        ):
            raise ValueError(
                "registered torchvision checkpoint SHA-256 is invalid"
            )
        if (
            self.source_kind != "torchvision_weights"
            and self.official_checkpoint_sha256
        ):
            raise ValueError(
                "only torchvision-weight models may preregister an official checkpoint"
            )
        if self.source_kind == "checkpoint_qualified_state_dict":
            if self.checkpoint_source is None:
                raise ValueError(
                    "checkpoint-qualified model requires a source contract"
                )
            self.checkpoint_source.validate()
        elif self.checkpoint_source is not None:
            raise ValueError(
                "only checkpoint-qualified models may carry a source contract"
            )
        if int(self.num_classes) <= 0:
            raise ValueError("registered model class count must be positive")


MODEL_REGISTRY: dict[str, RegisteredModelDefinition] = {
    item.model_id: item
    for item in (
        RegisteredModelDefinition(
            "torchvision.alexnet.imagenet1k_v1",
            "alexnet",
            "alexnet",
            "torchvision_weights",
            "AlexNet_Weights",
            "IMAGENET1K_V1",
            official_checkpoint_sha256=(
                "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
            ),
        ),
        # Same official weights, kernel-tap route decomposition.
        # Identity separation across granularities is mechanical (one
        # model id per (weights x route family)), so contracts, caches,
        # and receipts can never pool with the channel-pair family.
        RegisteredModelDefinition(
            "torchvision.alexnet.conv_kernel_weight.imagenet1k_v1",
            "alexnet_conv_kernel_weight",
            "alexnet",
            "torchvision_weights",
            "AlexNet_Weights",
            "IMAGENET1K_V1",
            official_checkpoint_sha256=(
                "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
            ),
        ),
        RegisteredModelDefinition(
            "torchvision.resnet18.imagenet1k_v1",
            "resnet18",
            "resnet18",
            "torchvision_weights",
            "ResNet18_Weights",
            "IMAGENET1K_V1",
        ),
        RegisteredModelDefinition(
            "torchvision.resnet18.conv_kernel_weight.imagenet1k_v1",
            "resnet18_conv_kernel_weight",
            "resnet18",
            "torchvision_weights",
            "ResNet18_Weights",
            "IMAGENET1K_V1",
        ),
        RegisteredModelDefinition(
            "torchvision.resnet50.imagenet1k_v2",
            "resnet50",
            "resnet50",
            "torchvision_weights",
            "ResNet50_Weights",
            "IMAGENET1K_V2",
        ),
        RegisteredModelDefinition(
            "torchvision.resnet50.conv_kernel_weight.imagenet1k_v2",
            "resnet50_conv_kernel_weight",
            "resnet50",
            "torchvision_weights",
            "ResNet50_Weights",
            "IMAGENET1K_V2",
        ),
        # The IMAGENET1K_V1 checkpoint on the SAME registered
        # kernel-weight axis (topology, route universe and admission are keyed
        # by family, not checkpoint).  The preregistered full digest admits
        # only the exact official bytes, including a verified replica.  Every
        # pre-existing entry is unchanged.
        RegisteredModelDefinition(
            "torchvision.resnet50.conv_kernel_weight.imagenet1k_v1",
            "resnet50_conv_kernel_weight",
            "resnet50",
            "torchvision_weights",
            "ResNet50_Weights",
            "IMAGENET1K_V1",
            official_checkpoint_sha256=(
                "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
            ),
        ),
        RegisteredModelDefinition(
            ROBUSTBENCH_SALMAN2020DO_R50_MODEL_ID,
            "resnet50_conv_kernel_weight",
            "resnet50",
            "checkpoint_qualified_state_dict",
            checkpoint_source=ROBUSTBENCH_SALMAN2020DO_R50_SOURCE,
        ),
        RegisteredModelDefinition(
            "torchvision.vgg11.imagenet1k_v1",
            "vgg11",
            "vgg11",
            "torchvision_weights",
            "VGG11_Weights",
            "IMAGENET1K_V1",
        ),
        RegisteredModelDefinition(
            "torchvision.vgg11.conv_kernel_weight.imagenet1k_v1",
            "vgg11_conv_kernel_weight",
            "vgg11",
            "torchvision_weights",
            "VGG11_Weights",
            "IMAGENET1K_V1",
        ),
        RegisteredModelDefinition(
            "torchvision.vgg11_bn.imagenet1k_v1",
            "vgg11_bn",
            "vgg11_bn",
            "torchvision_weights",
            "VGG11_BN_Weights",
            "IMAGENET1K_V1",
        ),
        RegisteredModelDefinition(
            "torchvision.resnet18.explicit_state_dict_v1",
            "resnet18",
            "resnet18",
            "explicit_state_dict",
        ),
        RegisteredModelDefinition(
            "torchvision.resnet50.explicit_state_dict_v1",
            "resnet50",
            "resnet50",
            "explicit_state_dict",
        ),
        RegisteredModelDefinition(
            "torchvision.resnet18.seeded_untrained_v1",
            "resnet18",
            "resnet18",
            "seeded_untrained",
        ),
        RegisteredModelDefinition(
            "torchvision.resnet50.seeded_untrained_v1",
            "resnet50",
            "resnet50",
            "seeded_untrained",
        ),
    )
}


def state_dict_digest(model_or_state_dict: Any) -> str:
    """Hash an ordered tensor state without constructing one giant byte buffer."""

    _require_torchvision()
    state = model_or_state_dict.state_dict() if hasattr(model_or_state_dict, "state_dict") else model_or_state_dict
    if not isinstance(state, dict) or not state:
        raise ValueError("state-dict digest requires a nonempty raw tensor mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise ValueError(f"state-dict entry {name!r} is not a tensor")
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(int(item) for item in value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelLoadReceipt:
    model_id: str
    architecture_family: str
    source_kind: str
    checkpoint_hash: str
    state_dict_hash: str
    model_definition_hash: str
    route_schema_hash: str
    source_uri: str
    strict_load: bool
    claim_eligible_source: bool
    source_contract_hash: str = ""
    # The content hash of the registered load-time model
    # transform, emitted only when non-empty so every existing receipt keeps
    # its exact bytes and ``content_hash``.
    model_transform_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_name": MODEL_LOAD_SCHEMA,
            "model_id": self.model_id,
            "architecture_family": self.architecture_family,
            "source_kind": self.source_kind,
            "checkpoint_hash": self.checkpoint_hash,
            "state_dict_hash": self.state_dict_hash,
            "model_definition_hash": self.model_definition_hash,
            "route_schema_hash": self.route_schema_hash,
            "source_uri": self.source_uri,
            "strict_load": bool(self.strict_load),
            "claim_eligible_source": bool(self.claim_eligible_source),
            "determinism": deterministic_backend_record(),
        }
        if self.source_contract_hash:
            payload["source_contract_hash"] = self.source_contract_hash
        if self.model_transform_hash:
            payload["model_transform_hash"] = self.model_transform_hash
        return payload

    def validate(self) -> None:
        if not self.model_id or not self.architecture_family or not self.source_kind:
            raise ValueError("model-load receipt identity is incomplete")
        for name, value in (
            ("checkpoint_hash", self.checkpoint_hash),
            ("state_dict_hash", self.state_dict_hash),
            ("model_definition_hash", self.model_definition_hash),
            ("route_schema_hash", self.route_schema_hash),
        ):
            if len(str(value)) != 64:
                raise ValueError(f"model-load receipt {name} is not a SHA-256 digest")
        if not self.strict_load:
            raise ValueError("canonical model-load receipts require strict parameter loading")
        if self.source_contract_hash and not _is_sha256(
            self.source_contract_hash
        ):
            raise ValueError(
                "model-load receipt source_contract_hash is not a SHA-256 digest"
            )
        if self.model_transform_hash and not _is_sha256(
            self.model_transform_hash
        ):
            raise ValueError(
                "model-load receipt model_transform_hash is not a SHA-256 digest"
            )

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ModelLoadReceipt:
        if str(payload.get("schema_name", "")) != MODEL_LOAD_SCHEMA:
            raise ValueError("unsupported model-load receipt schema")
        result = cls(
            model_id=str(payload.get("model_id", "")),
            architecture_family=str(payload.get("architecture_family", "")),
            source_kind=str(payload.get("source_kind", "")),
            checkpoint_hash=str(payload.get("checkpoint_hash", "")),
            state_dict_hash=str(payload.get("state_dict_hash", "")),
            model_definition_hash=str(payload.get("model_definition_hash", "")),
            route_schema_hash=str(payload.get("route_schema_hash", "")),
            source_uri=str(payload.get("source_uri", "")),
            strict_load=bool(payload.get("strict_load", False)),
            claim_eligible_source=bool(payload.get("claim_eligible_source", False)),
            source_contract_hash=str(payload.get("source_contract_hash", "")),
            model_transform_hash=str(payload.get("model_transform_hash", "")),
        )
        result.validate()
        return result

    @property
    def content_hash(self) -> str:
        self.validate()
        payload = self.to_dict()
        payload.pop("source_uri")
        payload.pop("determinism")
        return stable_digest(payload)


@dataclass(frozen=True)
class LoadedRegisteredModel:
    model: Any
    architecture: BoundTorchArchitecture
    receipt: ModelLoadReceipt

    def validate_against_study(self, study: StudySpec, *, claim_bearing: bool) -> None:
        study.validate().require_valid()
        issues: list[str] = []
        if study.model_id != self.receipt.model_id:
            issues.append("model_id")
        if study.checkpoint_hash != self.receipt.checkpoint_hash:
            issues.append("checkpoint_hash")
        if study.model_definition_hash != self.receipt.model_definition_hash:
            issues.append("model_definition_hash")
        if study.route_schema.architecture_family != self.receipt.architecture_family:
            issues.append("architecture_family")
        if study.route_schema.coordinates_hash != self.receipt.route_schema_hash:
            issues.append("route_schema_hash")
        if claim_bearing and not self.receipt.claim_eligible_source:
            issues.append("claim_eligible_source")
        if issues:
            raise ValueError("loaded model differs from StudySpec: " + ",".join(issues))


@dataclass(frozen=True)
class LoadedCheckpointQualifiedModel(LoadedRegisteredModel):
    """A bare model whose only input normalization owner is the Study operator."""

    source_contract: CheckpointQualifiedModelSource
    preprocessing_spec: ImagePreprocessingSpec

    def validate_against_study(self, study: StudySpec, *, claim_bearing: bool) -> None:
        super().validate_against_study(study, claim_bearing=claim_bearing)
        if self.receipt.source_contract_hash != self.source_contract.content_hash:
            raise ValueError("loaded model source contract differs from its receipt")
        if dict(study.preprocessing) != self.preprocessing_spec.to_dict():
            raise ValueError(
                "checkpoint-qualified model requires its exact external preprocessing"
            )


def route_coordinate_schema_from_architecture(
    architecture: BoundTorchArchitecture,
) -> RouteCoordinateSchema:
    architecture.validate()
    return RouteCoordinateSchema(
        schema_id=f"{architecture.architecture_id}:{architecture.spec.route_profile_id}",
        coordinates_hash=architecture.route_schema_hash,
        coordinate_table_uri=(
            "implicit://mechanistic-systems/"
            f"{architecture.architecture_id}/{architecture.spec.route_profile_id}"
        ),
        architecture_family=architecture.spec.family,
        route_kinds=tuple(dict.fromkeys(block.route_kind for block in architecture.route_blocks)),
        alias_map_hash=stable_digest(
            {
                "aliases": [
                    {"module_path": binding.module_path, "layer_id": binding.layer_id}
                    for binding in architecture.bindings
                ]
            }
        ),
        lowering_version="v1",
        notes={
            "measurement_route_block_count": len(architecture.route_blocks),
            "circuit_route_block_count": len(architecture.circuit_route_blocks),
            "circuit_route_schema_hash": architecture.circuit_route_schema_hash,
            "fixed_measurement_layer_ids": list(
                architecture.fixed_measurement_layer_ids
            ),
        },
    )


def _registered_definition(model_id: str) -> RegisteredModelDefinition:
    try:
        definition = MODEL_REGISTRY[str(model_id)]
    except KeyError as error:
        raise ValueError(f"unregistered canonical model ID {model_id!r}") from error
    definition.validate()
    return definition


def _builder(definition: RegisteredModelDefinition) -> Any:
    builder = getattr(tv_models, definition.builder_name, None)
    if not callable(builder):
        raise RuntimeError(f"torchvision has no builder {definition.builder_name!r}")
    return builder


def _official_checkpoint(
    definition: RegisteredModelDefinition,
    *,
    allow_download: bool,
    checkpoint_path: str | Path | None = None,
    checkpoint_file_hash: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Resolve official torchvision bytes or one exactly qualified replica.

    A caller-supplied path is never interpreted as an interchangeable state
    dict for a torchvision model ID.  It is accepted only when the caller
    declares the full digest and that digest is preregistered on the canonical
    model definition.  This keeps a replica-location override separate from a
    scientific weight-identity override.
    """

    declared_hash = str(checkpoint_file_hash or "")
    if declared_hash and not _is_sha256(declared_hash):
        raise ValueError(
            "registered torchvision checkpoint replica requires a full lowercase SHA-256"
        )
    if (
        declared_hash
        and definition.official_checkpoint_sha256
        and declared_hash != definition.official_checkpoint_sha256
    ):
        raise ValueError(
            "registered torchvision checkpoint replica differs from the "
            "preregistered official digest"
        )
    if checkpoint_path is not None:
        if not declared_hash:
            raise ValueError(
                "registered torchvision weights do not accept an unqualified "
                "alternate checkpoint path"
            )
        if not definition.official_checkpoint_sha256:
            raise ValueError(
                "registered torchvision model has no full checkpoint digest for replicas"
            )
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"registered torchvision checkpoint replica is absent: {checkpoint}"
            )
        observed_hash = sha256_file(checkpoint)
        if observed_hash != declared_hash:
            raise ValueError(
                "registered torchvision checkpoint replica SHA-256 mismatch"
            )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or not all(
            torch.is_tensor(value) for value in state.values()
        ):
            raise ValueError(
                "registered torchvision checkpoint is not a raw tensor state dict"
            )
        return dict(state), checkpoint

    enum_class = getattr(tv_models, definition.weights_enum_class, None)
    if enum_class is None or not hasattr(enum_class, definition.weights_member):
        raise RuntimeError("installed torchvision lacks the registered weight enum")
    weights = getattr(enum_class, definition.weights_member)
    filename = Path(urlparse(str(weights.url)).path).name
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not checkpoint.is_file():
        if not allow_download:
            raise FileNotFoundError(
                f"registered torchvision checkpoint is absent from the local cache: {checkpoint}"
            )
        state = torch.hub.load_state_dict_from_url(
            str(weights.url),
            model_dir=str(checkpoint.parent),
            map_location="cpu",
            progress=False,
            check_hash=True,
        )
        if definition.official_checkpoint_sha256 or declared_hash:
            observed_hash = sha256_file(checkpoint)
            if (
                definition.official_checkpoint_sha256
                and observed_hash != definition.official_checkpoint_sha256
            ):
                raise ValueError(
                    "downloaded registered torchvision checkpoint SHA-256 mismatch"
                )
            if declared_hash and observed_hash != declared_hash:
                raise ValueError(
                    "downloaded registered torchvision checkpoint differs from its replica binding"
                )
    else:
        if definition.official_checkpoint_sha256 or declared_hash:
            observed_hash = sha256_file(checkpoint)
            if (
                definition.official_checkpoint_sha256
                and observed_hash != definition.official_checkpoint_sha256
            ):
                raise ValueError(
                    "cached registered torchvision checkpoint SHA-256 mismatch"
                )
            if declared_hash and observed_hash != declared_hash:
                raise ValueError(
                    "cached registered torchvision checkpoint differs from its replica binding"
                )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("registered torchvision checkpoint is not a raw tensor state dict")
    return dict(state), checkpoint


def _apply_registered_model_transform(
    definition: RegisteredModelDefinition,
    *,
    state: dict[str, Any],
    source: Path,
    model_transform: Any,
) -> dict[str, Any]:
    """Verify the base bytes and state, then derive the transformed state."""

    from .model_transforms import apply_model_transform

    if sha256_file(source) != model_transform.base_checkpoint_sha256:
        raise ValueError(
            "model transform base checkpoint differs from the loaded checkpoint"
        )
    if state_dict_digest(state) != model_transform.base_state_dict_sha256:
        raise ValueError(
            "model transform base state digest differs from the loaded checkpoint"
        )
    with torch.device("meta"):
        shell = _builder(definition)(weights=None, num_classes=definition.num_classes)
    bound = canonical_torch_architecture_spec(definition.architecture_family).bind(shell)
    bound.validate()
    return apply_model_transform(state, model_transform, bound)


def load_registered_model_source(
    model_id: str,
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_file_hash: str | None = None,
    random_seed: int = 0,
    allow_download: bool = False,
    device: Any = "cpu",
    model_transform: Any | None = None,
) -> LoadedRegisteredModel:
    """Load one registered model and compute all scientific identities.

    ``model_transform`` (a ``ModelTransformSpec``) derives a
    registered model from the verified base checkpoint at load time.  The
    base file and base state digests are checked before the transform, the
    transform is applied before the strict load, and the loaded state digest
    must equal the spec's derived digest.  ``checkpoint_hash`` stays the base
    file's hash (so the base Study validates); ``state_dict_hash`` is the
    derived digest; the receipt carries the transform's content hash.  With
    ``model_transform=None`` the path is unchanged.
    """

    configure_deterministic_backend()
    _require_torchvision()
    definition = _registered_definition(model_id)
    builder = _builder(definition)
    if model_transform is not None:
        from .model_transforms import ModelTransformSpec

        if not isinstance(model_transform, ModelTransformSpec):
            raise TypeError("model_transform must be a ModelTransformSpec")
        model_transform.validate()
        if definition.source_kind != "torchvision_weights":
            raise ValueError(
                "registered model transforms admit only torchvision-weight sources"
            )
        if model_transform.base_model_id != definition.model_id:
            raise ValueError(
                "model transform base model differs from the loaded model id"
            )
    strict_load = True
    source_uri = ""
    source_contract_hash = ""
    checkpoint_source = definition.checkpoint_source
    claim_eligible = definition.source_kind != "seeded_untrained"
    if definition.source_kind == "seeded_untrained":
        if checkpoint_path is not None or checkpoint_file_hash:
            raise ValueError("seeded-untrained model does not accept a checkpoint")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(random_seed))
            model = builder(weights=None, num_classes=definition.num_classes)
        checkpoint_hash = state_dict_digest(model)
        source_uri = f"seeded-untrained://{definition.builder_name}/{int(random_seed)}"
    else:
        if definition.source_kind == "torchvision_weights":
            state, source = _official_checkpoint(
                definition,
                allow_download=allow_download,
                checkpoint_path=checkpoint_path,
                checkpoint_file_hash=checkpoint_file_hash,
            )
            if model_transform is not None:
                state = _apply_registered_model_transform(
                    definition,
                    state=state,
                    source=source,
                    model_transform=model_transform,
                )
        else:
            if checkpoint_path is None:
                raise ValueError(
                    "explicit-state-dict model requires checkpoint_path"
                )
            source = Path(checkpoint_path)
            if not source.is_file():
                raise FileNotFoundError(source)
            if checkpoint_file_hash:
                if not _is_sha256(checkpoint_file_hash):
                    raise ValueError(
                        "registered checkpoint replica hash is not a full lowercase SHA-256"
                    )
                if sha256_file(source) != checkpoint_file_hash:
                    raise ValueError("registered checkpoint replica SHA-256 mismatch")
            if checkpoint_source is not None:
                checkpoint_source.validate()
                observed_size = source.stat().st_size
                if observed_size != checkpoint_source.artifact_size_bytes:
                    raise ValueError(
                        "checkpoint-qualified artifact size mismatch: "
                        f"expected={checkpoint_source.artifact_size_bytes}, "
                        f"observed={observed_size}"
                    )
                checkpoint_hash = sha256_file(source)
                if checkpoint_hash != checkpoint_source.checkpoint_sha256:
                    raise ValueError(
                        "checkpoint-qualified artifact SHA-256 mismatch"
                    )
            state = torch.load(source, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or not state or not all(
                torch.is_tensor(value) for value in state.values()
            ):
                raise ValueError(
                    "canonical explicit checkpoints must be raw tensor state dicts; "
                    "convert wrapped legacy checkpoints through import-legacy"
                )
            state = dict(state)
            if checkpoint_source is not None:
                source_state_hash = state_dict_digest(state)
                if source_state_hash != checkpoint_source.state_dict_sha256:
                    raise ValueError(
                        "checkpoint-qualified canonical state digest mismatch"
                    )
        model = builder(weights=None, num_classes=definition.num_classes)
        load_result = model.load_state_dict(state, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise ValueError("strict canonical checkpoint load reported missing or unexpected keys")
        if checkpoint_source is None:
            checkpoint_hash = sha256_file(source)
            source_uri = str(source)
        else:
            source_uri = checkpoint_source.artifact_uri
            source_contract_hash = checkpoint_source.content_hash
    state_hash = state_dict_digest(model)
    if (
        model_transform is not None
        and state_hash != model_transform.derived_state_dict_sha256
    ):
        raise ValueError(
            "strict load of the transformed model differs from the declared "
            "derived state digest"
        )
    if (
        checkpoint_source is not None
        and state_hash != checkpoint_source.state_dict_sha256
    ):
        raise RuntimeError(
            "strict checkpoint-qualified load changed the canonical state digest"
        )
    architecture = canonical_torch_architecture_spec(definition.architecture_family).bind(model)
    architecture.validate()
    from mechanistic_systems.backends.deterministic_pooling import install_deterministic_adaptive_avg_pool2d
    install_deterministic_adaptive_avg_pool2d(model)
    model.to(torch.device(device))
    model.eval()
    receipt = ModelLoadReceipt(
        model_id=definition.model_id,
        architecture_family=definition.architecture_family,
        source_kind=definition.source_kind,
        checkpoint_hash=checkpoint_hash,
        state_dict_hash=state_hash,
        model_definition_hash=architecture.topology_hash,
        route_schema_hash=architecture.route_schema_hash,
        source_uri=source_uri,
        strict_load=strict_load,
        claim_eligible_source=claim_eligible,
        source_contract_hash=source_contract_hash,
        model_transform_hash=(
            "" if model_transform is None else model_transform.content_hash
        ),
    )
    if checkpoint_source is not None:
        preprocessing = ImagePreprocessingSpec.from_mapping(
            canonical_imagenet_input_operator_contract()
        )
        if preprocessing.content_hash != (
            checkpoint_source.execution_preprocessing_sha256
        ):
            raise RuntimeError(
                "checkpoint-qualified preprocessing changed after source validation"
            )
        return LoadedCheckpointQualifiedModel(
            model=model,
            architecture=architecture,
            receipt=receipt,
            source_contract=checkpoint_source,
            preprocessing_spec=preprocessing,
        )
    return LoadedRegisteredModel(
        model=model,
        architecture=architecture,
        receipt=receipt,
    )


def load_registered_model_for_study(
    study: StudySpec,
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_file_hash: str | None = None,
    allow_download: bool = False,
    device: Any = "cpu",
    claim_bearing: bool = True,
    model_transform: Any | None = None,
) -> LoadedRegisteredModel:
    loaded = load_registered_model_source(
        study.model_id,
        checkpoint_path=checkpoint_path,
        checkpoint_file_hash=checkpoint_file_hash,
        random_seed=int(study.seeds.get("model_init", 0)),
        allow_download=allow_download,
        device=device,
        **({} if model_transform is None else {"model_transform": model_transform}),
    )
    loaded.validate_against_study(study, claim_bearing=claim_bearing)
    return loaded


__all__ = [
    "CHECKPOINT_QUALIFIED_SOURCE_SCHEMA",
    "MODEL_LOAD_SCHEMA",
    "MODEL_REGISTRY",
    "ROBUSTBENCH_SALMAN2020DO_R50_MODEL_ID",
    "ROBUSTBENCH_SALMAN2020DO_R50_SOURCE",
    "CheckpointQualifiedModelSource",
    "LoadedCheckpointQualifiedModel",
    "LoadedRegisteredModel",
    "ModelLoadReceipt",
    "RegisteredModelDefinition",
    "load_registered_model_for_study",
    "load_registered_model_source",
    "route_coordinate_schema_from_architecture",
    "state_dict_digest",
]
