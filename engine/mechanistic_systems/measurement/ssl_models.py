"""Canonical SSL CNN backbones lowered to fixed-query executable models.

The active SSL ResNet workflows differ from supervised models in two inputs:
the frozen backbone checkpoint and the frozen target-direction bank.  Once the
bank is installed as the bias-free readout, the ordinary route-gate scorer and
sampled-path circuit implementation apply without SSL-specific mathematics.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts.hashing import sha256_file, stable_digest
from mechanistic_systems.core.architectures import (
    canonical_torch_architecture_spec,
)
from mechanistic_systems.measurement.models import (
    LoadedRegisteredModel,
    ModelLoadReceipt,
    state_dict_digest,
)
from mechanistic_systems.measurement.contexts import (
    ImagePreprocessingSpec,
)
from mechanistic_systems.measurement.ssl_adapter import (
    SSLDirectionBank,
    install_ssl_direction_readout,
)

try:  # Optional for import smoke; required for executable loading.
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
except Exception:  # pragma: no cover - environment dependent.
    torch = nn = tv_models = None  # type: ignore[assignment]

try:  # Exact SMP fixtures are distributed as safetensors.
    from safetensors.torch import load_file as _load_safetensors_file
except Exception:  # pragma: no cover - optional import surface.
    _load_safetensors_file = None


SSL_QUERY_MODEL_SCHEMA = "mechanistic_ssl_query_model_v1"
_CHECKPOINT_PREFIXES = ("attacker.model.", "module.", "model.", "net.", "_orig_mod.")
_VISIONLAB_ALEXNET_CHECKPOINT_TO_CANONICAL = {
    "base_arch.conv_block_1.0.weight": "features.0.weight",
    "base_arch.conv_block_2.0.weight": "features.3.weight",
    "base_arch.conv_block_3.0.weight": "features.6.weight",
    "base_arch.conv_block_4.0.weight": "features.8.weight",
    "base_arch.conv_block_5.0.weight": "features.10.weight",
    "base_arch.head.layers.0.linear.weight": "classifier.1.weight",
    "base_arch.head.layers.0.linear.bias": "classifier.1.bias",
    "base_arch.head.layers.0.norm.weight": "classifier.2.weight",
    "base_arch.head.layers.0.norm.bias": "classifier.2.bias",
    "base_arch.head.layers.0.norm.running_mean": "classifier.2.running_mean",
    "base_arch.head.layers.0.norm.running_var": "classifier.2.running_var",
    "base_arch.head.layers.0.norm.num_batches_tracked": (
        "classifier.2.num_batches_tracked"
    ),
    "base_arch.head.layers.1.linear.weight": "classifier.4.weight",
    "base_arch.head.layers.1.linear.bias": "classifier.4.bias",
    "base_arch.head.layers.1.norm.weight": "classifier.5.weight",
    "base_arch.head.layers.1.norm.bias": "classifier.5.bias",
    "base_arch.head.layers.1.norm.running_mean": "classifier.5.running_mean",
    "base_arch.head.layers.1.norm.running_var": "classifier.5.running_var",
    "base_arch.head.layers.1.norm.num_batches_tracked": (
        "classifier.5.num_batches_tracked"
    ),
    "base_arch.head.layers.2.linear.weight": "classifier.6.weight",
}
_VISIONLAB_ALEXNET_IGNORED_CHECKPOINT_KEYS = frozenset({"loss_fn.temperature"})


class VisionLabAlexNetW1MLP4096(nn.Module if nn is not None else object):
    """Exact executable topology of the active Vision Lab SimCLR checkpoint."""

    def __init__(self) -> None:
        _require_torchvision()
        super().__init__()
        self.features = nn.Sequential(
            OrderedDict(
                (
                    ("0", nn.Conv2d(3, 96, 11, stride=4, padding=2, bias=False)),
                    ("1", nn.ReLU(inplace=False)),
                    ("2", nn.MaxPool2d(3, stride=2)),
                    ("3", nn.Conv2d(96, 256, 5, padding=2, bias=False)),
                    ("4", nn.ReLU(inplace=False)),
                    ("5", nn.MaxPool2d(3, stride=2)),
                    ("6", nn.Conv2d(256, 384, 3, padding=1, bias=False)),
                    ("7", nn.ReLU(inplace=False)),
                    ("8", nn.Conv2d(384, 384, 3, padding=1, bias=False)),
                    ("9", nn.ReLU(inplace=False)),
                    ("10", nn.Conv2d(384, 256, 3, padding=1, bias=False)),
                    ("11", nn.ReLU(inplace=False)),
                    ("12", nn.MaxPool2d(3, stride=2)),
                )
            )
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            OrderedDict(
                (
                    ("0", nn.Identity()),
                    ("1", nn.Linear(256 * 6 * 6, 4096, bias=True)),
                    ("2", nn.BatchNorm1d(4096)),
                    ("3", nn.ReLU(inplace=False)),
                    ("4", nn.Linear(4096, 4096, bias=True)),
                    ("5", nn.BatchNorm1d(4096)),
                    ("6", nn.Linear(4096, 4096, bias=False)),
                )
            )
        )

    def forward(self, inputs: Any) -> Any:
        values = self.features(inputs)
        values = self.avgpool(values)
        values = torch.flatten(values, 1)
        return self.classifier(values)


def build_visionlab_alexnet_ssl_backbone() -> VisionLabAlexNetW1MLP4096:
    """Build the checkpoint-shaped model without importing a legacy adapter."""

    model = VisionLabAlexNetW1MLP4096()
    model.eval()
    return model


def _require_torchvision() -> None:
    if torch is None or nn is None or tv_models is None:
        raise RuntimeError("SSL query-model loading requires PyTorch and torchvision")


@dataclass(frozen=True)
class SSLBackboneDefinition:
    model_id: str
    architecture_family: str
    builder_name: str
    training_method: str
    feature_width: int
    fixed_readout_layer: str
    checkpoint_contract: str
    checkpoint_serialization: str
    checkpoint_buffer_policy: str
    default_preprocessing_policy: str

    def validate(self) -> None:
        if not all(
            (
                self.model_id,
                self.architecture_family,
                self.builder_name,
                self.training_method,
                self.fixed_readout_layer,
                self.checkpoint_contract,
                self.checkpoint_serialization,
                self.checkpoint_buffer_policy,
                self.default_preprocessing_policy,
            )
        ):
            raise ValueError("SSL backbone definition requires complete identity")
        if self.architecture_family not in {
            "alexnet_ssl_w1_mlp4096",
            "resnet18",
            "resnet50",
        }:
            raise ValueError("unsupported canonical SSL CNN architecture")
        if int(self.feature_width) <= 0:
            raise ValueError("SSL backbone feature width must be positive")
        if self.checkpoint_serialization not in {
            "safetensors",
            "torch_weights_only",
            "trusted_legacy_torch_pickle",
        }:
            raise ValueError("unsupported SSL checkpoint serialization policy")
        if self.checkpoint_buffer_policy not in {
            "strict",
            "zero_missing_batch_norm_counters",
        }:
            raise ValueError("unsupported SSL checkpoint buffer policy")


SSL_BACKBONE_REGISTRY: dict[str, SSLBackboneDefinition] = {
    item.model_id: item
    for item in (
        SSLBackboneDefinition(
            model_id="ssl.alexnet.visionlab_simclr_imagenet100_v1",
            architecture_family="alexnet_ssl_w1_mlp4096",
            builder_name="visionlab_alexnet_w1_mlp4096",
            training_method="visionlab_simclr_imagenet100_ffcv",
            feature_width=4096,
            fixed_readout_layer="classifier.6",
            checkpoint_contract="visionlab_alexnet_w1_mlp4096_state_v1",
            checkpoint_serialization="trusted_legacy_torch_pickle",
            checkpoint_buffer_policy="strict",
            default_preprocessing_policy="visionlab_alexnet_config_native",
        ),
        SSLBackboneDefinition(
            model_id="ssl.resnet18.smp_imagenet_ssl_v1",
            architecture_family="resnet18",
            builder_name="resnet18",
            training_method="segmentation_models_pytorch_imagenet_ssl",
            feature_width=512,
            fixed_readout_layer="fc",
            checkpoint_contract="torchvision_resnet18_trunk_state_v1",
            checkpoint_serialization="safetensors",
            checkpoint_buffer_policy="zero_missing_batch_norm_counters",
            default_preprocessing_policy="legacy_resnet18_config_native",
        ),
        SSLBackboneDefinition(
            model_id="ssl.resnet50.dino_imagenet1k_v1",
            architecture_family="resnet50",
            builder_name="resnet50",
            training_method="dino_imagenet1k",
            feature_width=2048,
            fixed_readout_layer="fc",
            checkpoint_contract="torchvision_resnet50_trunk_state_v1",
            checkpoint_serialization="torch_weights_only",
            checkpoint_buffer_policy="zero_missing_batch_norm_counters",
            default_preprocessing_policy="legacy_resnet50_config_native",
        ),
        SSLBackboneDefinition(
            model_id="ssl.resnet50.simclrv1_imagenet1k_1x_v1",
            architecture_family="resnet50",
            builder_name="resnet50",
            training_method="simclrv1_imagenet1k_1x",
            feature_width=2048,
            fixed_readout_layer="fc",
            checkpoint_contract="torchvision_resnet50_trunk_state_v1",
            checkpoint_serialization="torch_weights_only",
            checkpoint_buffer_policy="zero_missing_batch_norm_counters",
            default_preprocessing_policy="simclr_converter_linear_eval",
        ),
    )
}


def _definition(model_id: str) -> SSLBackboneDefinition:
    try:
        result = SSL_BACKBONE_REGISTRY[str(model_id)]
    except KeyError as error:
        raise ValueError(f"unregistered canonical SSL backbone {model_id!r}") from error
    result.validate()
    return result


def _unwrap_tensor_state(payload: Any) -> dict[str, Any]:
    value = payload
    if isinstance(value, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                value = nested
                break
    if not isinstance(value, Mapping) or not value:
        raise ValueError("SSL checkpoint is not a nonempty tensor state mapping")
    state = {str(key): tensor for key, tensor in value.items()}
    if not all(torch.is_tensor(tensor) for tensor in state.values()):
        raise ValueError("SSL checkpoint state contains non-tensor values")
    changed = True
    while changed:
        changed = False
        for prefix in _CHECKPOINT_PREFIXES:
            if state and all(key.startswith(prefix) for key in state):
                state = {key[len(prefix) :]: tensor for key, tensor in state.items()}
                changed = True
                break
    return state


def _load_checkpoint(
    path: Path,
    *,
    serialization_policy: str | None = None,
) -> dict[str, Any]:
    policy = serialization_policy
    if policy is None:
        policy = (
            "safetensors"
            if path.suffix.lower() == ".safetensors"
            else "torch_weights_only"
        )
    if policy == "safetensors":
        if _load_safetensors_file is None:
            raise RuntimeError(
                "loading an SSL safetensors checkpoint requires the safetensors package"
            )
        return _unwrap_tensor_state(_load_safetensors_file(str(path), device="cpu"))
    if policy == "torch_weights_only":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older supported Torch versions.
            payload = torch.load(path, map_location="cpu")
    elif policy == "trusted_legacy_torch_pickle":
        # This is intentionally registry-gated. It exists only for sealed,
        # content-hashed lab checkpoints whose old pickle protocol cannot be
        # decoded by PyTorch's restricted weights-only unpickler.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"unsupported checkpoint serialization policy {policy!r}")
    return _unwrap_tensor_state(payload)


def _translate_visionlab_alexnet_state(state: Mapping[str, Any]) -> dict[str, Any]:
    translated: dict[str, Any] = {}
    unexpected: list[str] = []
    for raw_key, tensor in state.items():
        key = str(raw_key)
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        target = _VISIONLAB_ALEXNET_CHECKPOINT_TO_CANONICAL.get(key)
        if target is not None:
            translated[target] = tensor
        elif key not in _VISIONLAB_ALEXNET_IGNORED_CHECKPOINT_KEYS:
            unexpected.append(key)
    if unexpected:
        raise ValueError(
            "Vision Lab AlexNet SSL checkpoint has undeclared keys: "
            + ", ".join(sorted(unexpected)[:20])
        )
    return translated


def _load_strict_complete_state(model: Any, state: Mapping[str, Any]) -> None:
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = sorted(
        key
        for key in set(expected).intersection(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise ValueError(
            "SSL checkpoint differs from its canonical topology: "
            f"missing={missing}; unexpected={unexpected}; "
            f"shape_mismatch={shape_mismatch}"
        )
    model.load_state_dict(dict(state), strict=True)


def ssl_preprocessing_spec(
    model_id: str,
    direction_bank: SSLDirectionBank,
) -> ImagePreprocessingSpec:
    """Resolve the exact historical image contract for one SSL query model."""

    definition = _definition(model_id)
    direction_bank.validate()
    if direction_bank.architecture_family != definition.architecture_family:
        raise ValueError("SSL direction bank and preprocessing architecture differ")
    declared = str(direction_bank.preprocessing_policy).strip()
    if definition.training_method == "simclrv1_imagenet1k_1x":
        required = "simclr_converter_linear_eval"
    elif direction_bank.target_family == "linear_probe_category_direction":
        required = "imagenet_linear_eval"
    else:
        required = definition.default_preprocessing_policy
    if declared and declared != required:
        raise ValueError(
            "SSL target manifest preprocessing differs from the backbone/query contract: "
            f"declared={declared!r}, required={required!r}"
        )
    common_notes = {
        "ssl_backbone_id": definition.model_id,
        "ssl_target_family": direction_bank.target_family,
        "target_manifest_hash": direction_bank.manifest_hash,
    }
    if required == "imagenet_linear_eval":
        result = ImagePreprocessingSpec(
            preprocessing_id=required,
            resize_size=(256,),
            center_crop_size=(224,),
            interpolation="bicubic",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            notes=common_notes,
        )
    elif required == "simclr_converter_linear_eval":
        result = ImagePreprocessingSpec(
            preprocessing_id=required,
            resize_size=(256,),
            center_crop_size=(224,),
            interpolation="bicubic",
            mean=tuple(),
            std=tuple(),
            notes=common_notes,
        )
    elif required in {
        "legacy_resnet18_config_native",
        "legacy_resnet50_config_native",
    }:
        result = ImagePreprocessingSpec(
            preprocessing_id=required,
            resize_size=(224, 224),
            interpolation="bilinear",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            notes=common_notes,
        )
    elif required == "visionlab_alexnet_config_native":
        result = ImagePreprocessingSpec(
            preprocessing_id=required,
            resize_size=(256,),
            center_crop_size=(224,),
            interpolation="bilinear",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            notes=common_notes,
        )
    else:  # pragma: no cover - registry validation owns known policies.
        raise ValueError(f"unsupported SSL preprocessing policy {required!r}")
    result.validate()
    return result


@dataclass(frozen=True)
class LoadedSSLQueryModel(LoadedRegisteredModel):
    direction_bank_hash: str
    preprocessing_spec: ImagePreprocessingSpec

    def validate_against_study(self, study: Any, *, claim_bearing: bool) -> None:
        super().validate_against_study(study, claim_bearing=claim_bearing)
        if dict(study.preprocessing) != self.preprocessing_spec.to_dict():
            raise ValueError("loaded SSL query model preprocessing differs from StudySpec")


def _strict_trunk_state(
    model: Any,
    state: Mapping[str, Any],
    *,
    missing_buffer_policy: str = "strict",
) -> dict[str, Any]:
    expected = {
        key: tensor for key, tensor in model.state_dict().items() if not key.startswith("fc.")
    }
    supplied = {key: tensor for key, tensor in state.items() if not key.startswith("fc.")}
    if missing_buffer_policy == "zero_missing_batch_norm_counters":
        for key in sorted(set(expected) - set(supplied)):
            if key.endswith(".num_batches_tracked") or key == "num_batches_tracked":
                supplied[key] = torch.zeros_like(expected[key])
    elif missing_buffer_policy != "strict":
        raise ValueError(
            f"unsupported SSL missing-buffer policy {missing_buffer_policy!r}"
        )
    discarded_head_keys = sorted(set(state) - set(supplied))
    undeclared_head_keys = sorted(
        set(discarded_head_keys) - {"fc.weight", "fc.bias"}
    )
    missing = sorted(set(expected) - set(supplied))
    unexpected = sorted(set(supplied) - set(expected))
    shape_mismatch = sorted(
        key
        for key in set(expected).intersection(supplied)
        if tuple(expected[key].shape) != tuple(supplied[key].shape)
    )
    if missing or unexpected or shape_mismatch or undeclared_head_keys:
        raise ValueError(
            "SSL backbone checkpoint differs from its canonical trunk: "
            f"missing={missing}; unexpected={unexpected}; shape_mismatch={shape_mismatch}; "
            f"undeclared_head_keys={undeclared_head_keys}"
        )
    return supplied


def load_ssl_query_model(
    model_id: str,
    *,
    checkpoint_path: str | Path,
    direction_bank: SSLDirectionBank,
    device: Any = "cpu",
) -> LoadedSSLQueryModel:
    """Load a sealed SSL trunk and direction bank as one canonical model.

    The checkpoint may be a raw tensor mapping or a standard wrapped state
    mapping.  Prefix and wrapper conversion is part of this explicit adapter;
    every backbone tensor is then checked strictly before execution.
    """

    _require_torchvision()
    definition = _definition(model_id)
    direction_bank.validate()
    if direction_bank.architecture_family != definition.architecture_family:
        raise ValueError("SSL direction bank and backbone architecture differ")
    if direction_bank.feature_width != definition.feature_width:
        raise ValueError("SSL direction bank width differs from backbone feature width")
    if direction_bank.target_layer != definition.fixed_readout_layer:
        raise ValueError(
            "SSL direction bank target layer differs from the canonical fixed readout"
        )
    preprocessing = ssl_preprocessing_spec(model_id, direction_bank)

    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    state = _load_checkpoint(
        source,
        serialization_policy=definition.checkpoint_serialization,
    )
    if definition.builder_name == "visionlab_alexnet_w1_mlp4096":
        model = build_visionlab_alexnet_ssl_backbone()
        translated = _translate_visionlab_alexnet_state(state)
        _load_strict_complete_state(model, translated)
        # Historical target banks are defined on classifier.4 before the
        # second head BatchNorm. Replace that downstream normalization so the
        # fixed classifier.6 query reads the declared representation exactly.
        model.classifier[5] = nn.Identity()
    else:
        builder = getattr(tv_models, definition.builder_name, None)
        if not callable(builder):
            raise RuntimeError(
                f"torchvision lacks SSL backbone builder {definition.builder_name!r}"
            )
        model = builder(weights=None)
        conv1 = state.get("conv1.weight")
        if definition.architecture_family == "resnet50" and (
            not torch.is_tensor(conv1) or tuple(conv1.shape[:2]) != (64, 3)
        ):
            shape = tuple(conv1.shape) if torch.is_tensor(conv1) else None
            raise ValueError(
                "canonical SSL ResNet50 requires the true 1x trunk; "
                f"conv1.weight shape is {shape}"
            )
        trunk_state = _strict_trunk_state(
            model,
            state,
            missing_buffer_policy=definition.checkpoint_buffer_policy,
        )
        load_result = model.load_state_dict(trunk_state, strict=False)
        allowed_missing = {"fc.weight", "fc.bias"}
        if set(load_result.missing_keys) != allowed_missing or load_result.unexpected_keys:
            raise ValueError(
                "strict SSL trunk load produced an undeclared key difference: "
                f"missing={sorted(load_result.missing_keys)}; "
                f"unexpected={sorted(load_result.unexpected_keys)}"
            )
    install_ssl_direction_readout(model, direction_bank)
    installed_readout = model.get_submodule(definition.fixed_readout_layer)
    if not isinstance(installed_readout, nn.Linear) or int(
        installed_readout.in_features
    ) != definition.feature_width:
        raise RuntimeError("SSL direction readout did not produce the declared fixed head")

    architecture = canonical_torch_architecture_spec(
        definition.architecture_family
    ).bind(model)
    architecture.validate()
    model.to(torch.device(device))
    model.eval()
    backbone_hash = sha256_file(source)
    executable_hash = stable_digest(
        {
            "schema_name": SSL_QUERY_MODEL_SCHEMA,
            "backbone_checkpoint_hash": backbone_hash,
            "direction_bank_hash": direction_bank.content_hash,
            "checkpoint_contract": definition.checkpoint_contract,
            "checkpoint_serialization": definition.checkpoint_serialization,
            "checkpoint_buffer_policy": definition.checkpoint_buffer_policy,
        }
    )
    receipt = ModelLoadReceipt(
        model_id=definition.model_id,
        architecture_family=definition.architecture_family,
        source_kind="ssl_fixed_query_readout",
        checkpoint_hash=executable_hash,
        state_dict_hash=state_dict_digest(model),
        model_definition_hash=architecture.topology_hash,
        route_schema_hash=architecture.route_schema_hash,
        source_uri=f"{source}|{direction_bank.manifest_path}",
        strict_load=True,
        claim_eligible_source=True,
    )
    return LoadedSSLQueryModel(
        model=model,
        architecture=architecture,
        receipt=receipt,
        direction_bank_hash=direction_bank.content_hash,
        preprocessing_spec=preprocessing,
    )


__all__ = [
    "SSL_BACKBONE_REGISTRY",
    "SSL_QUERY_MODEL_SCHEMA",
    "SSLBackboneDefinition",
    "LoadedSSLQueryModel",
    "VisionLabAlexNetW1MLP4096",
    "build_visionlab_alexnet_ssl_backbone",
    "load_ssl_query_model",
    "ssl_preprocessing_spec",
]
