"""Checkpoint-qualified bindings for analytic IRG empty-mask proofs.

The structural cutset argument lives with the registered pilot.  This module
owns the stronger source gate needed when two scientific model sources share
one operator topology: topology alone may establish input independence, but a
campaign receipt must additionally identify the exact loaded function and its
single preprocessing owner.
"""

from __future__ import annotations

from typing import Any, Mapping

from mechanistic_systems.input_operators import (
    canonical_imagenet_input_operator_contract,
)
from mechanistic_systems.measurement.contexts import ImagePreprocessingSpec
from mechanistic_systems.measurement.models import (
    ROBUSTBENCH_SALMAN2020DO_R50_MODEL_ID,
    ROBUSTBENCH_SALMAN2020DO_R50_SOURCE,
    LoadedCheckpointQualifiedModel,
    state_dict_digest,
)


_ROBUST_RESNET50_ARCHITECTURE_FAMILY = "resnet50_conv_kernel_weight"
_ROBUST_RESNET50_ARCHITECTURE_ID = (
    "torchvision.resnet50.conv_kernel_weight.v1"
)
_ROBUST_RESNET50_EVALUATOR_ID = "torch-explicit-gate-graph-evaluator-v2"


def _axis_signature(items: Any) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    return tuple(
        (
            str(item.layer_id),
            str(item.route_kind),
            tuple(int(value) for value in item.shape),
        )
        for item in items
    )


def _binding_signature(items: Any) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            str(item.layer_id),
            str(item.module_path),
            str(item.route_kind),
        )
        for item in items
    )


def robust_resnet50_checkpoint_proof_binding(
    runtime: Any,
) -> Mapping[str, Any]:
    """Validate and receipt the exact Salman2020Do_R50 runtime binding.

    The returned payload is safe to incorporate into an analytic empty-mask
    proof.  Every mismatch refuses before the structural proof is emitted.
    In particular, a generic ResNet50, a lookalike state dict, a topology-only
    fixture, or an internally normalized wrapper cannot inherit this source's
    proof identity.
    """

    loaded = getattr(runtime, "loaded", None)
    if not isinstance(loaded, LoadedCheckpointQualifiedModel):
        raise ValueError(
            "robust ResNet50 analytic proof requires a checkpoint-qualified "
            "loaded model"
        )
    loaded.architecture.validate()
    loaded.receipt.validate()
    loaded.source_contract.validate()

    architecture = loaded.architecture
    receipt = loaded.receipt
    source = loaded.source_contract
    preprocessing = loaded.preprocessing_spec
    expected_source = ROBUSTBENCH_SALMAN2020DO_R50_SOURCE
    expected_preprocessing = ImagePreprocessingSpec.from_mapping(
        canonical_imagenet_input_operator_contract()
    )

    source_identity_valid = (
        receipt.model_id == ROBUSTBENCH_SALMAN2020DO_R50_MODEL_ID
        and receipt.architecture_family == _ROBUST_RESNET50_ARCHITECTURE_FAMILY
        and receipt.source_kind == "checkpoint_qualified_state_dict"
        and bool(receipt.strict_load)
        and bool(receipt.claim_eligible_source)
        and source.content_hash == expected_source.content_hash
        and receipt.source_contract_hash == source.content_hash
        and receipt.checkpoint_hash == source.checkpoint_sha256
        and receipt.state_dict_hash == source.state_dict_sha256
        and receipt.source_uri == source.artifact_uri
    )
    if not source_identity_valid:
        raise ValueError(
            "robust ResNet50 analytic proof source/checkpoint identity differs "
            "from the registered Salman2020Do_R50 contract"
        )

    topology_identity_valid = (
        str(architecture.spec.family).lower()
        == _ROBUST_RESNET50_ARCHITECTURE_FAMILY
        and architecture.architecture_id == _ROBUST_RESNET50_ARCHITECTURE_ID
        and receipt.model_definition_hash == architecture.topology_hash
        and receipt.route_schema_hash == architecture.route_schema_hash
    )
    if not topology_identity_valid:
        raise ValueError(
            "robust ResNet50 analytic proof topology differs from its "
            "checkpoint-qualified model-load receipt"
        )

    if (
        not isinstance(preprocessing, ImagePreprocessingSpec)
        or preprocessing.to_dict() != expected_preprocessing.to_dict()
        or preprocessing.content_hash != source.execution_preprocessing_sha256
        or source.normalization_placement != "external_input_operator_only"
    ):
        raise ValueError(
            "robust ResNet50 analytic proof requires the exact single-owner "
            "external ImageNet preprocessing contract"
        )

    model = loaded.model
    model_type = type(model)
    if (
        model_type.__module__ != "torchvision.models.resnet"
        or model_type.__name__ != "ResNet"
        or bool(getattr(model, "training", True))
    ):
        raise ValueError(
            "robust ResNet50 analytic proof requires the bare torchvision "
            "ResNet function in eval mode"
        )
    forbidden_input_wrappers = {
        str(name)
        for name, _module in model.named_modules()
        if name
        and str(name).split(".")[-1].lower()
        in {"normalize", "normalizer", "preprocess", "preprocessor"}
    }
    if forbidden_input_wrappers:
        raise ValueError(
            "robust ResNet50 model contains an internal preprocessing owner"
        )
    if state_dict_digest(model) != source.state_dict_sha256:
        raise ValueError(
            "robust ResNet50 loaded function state differs from the "
            "checkpoint-qualified state digest"
        )

    evaluator = getattr(runtime, "evaluator", None)
    gate_graph = getattr(evaluator, "gate_graph", None)
    compiled_model = getattr(evaluator, "model", None)
    evaluator_valid = (
        evaluator is not None
        and getattr(evaluator, "source_model", None) is model
        and str(getattr(evaluator, "evaluator_id", ""))
        == _ROBUST_RESNET50_EVALUATOR_ID
        and bool(getattr(evaluator, "claim_execution_certified", False))
        and str(getattr(evaluator, "topology_hash", ""))
        == architecture.topology_hash
        and str(getattr(evaluator, "route_schema_hash", ""))
        == architecture.circuit_route_schema_hash
        and _axis_signature(getattr(evaluator, "route_blocks", ()))
        == _axis_signature(architecture.circuit_route_blocks)
        and _binding_signature(getattr(evaluator, "mask_bindings", ()))
        == _binding_signature(architecture.circuit_bindings)
        and gate_graph is not None
        and str(getattr(gate_graph, "architecture_id", ""))
        == architecture.architecture_id
        and str(getattr(gate_graph, "topology_hash", ""))
        == architecture.topology_hash
        and str(getattr(gate_graph, "route_schema_hash", ""))
        == architecture.circuit_route_schema_hash
        and compiled_model is getattr(gate_graph, "model", None)
        and not bool(getattr(compiled_model, "training", True))
    )
    if not evaluator_valid:
        raise ValueError(
            "robust ResNet50 analytic proof evaluator is not the certified "
            "gate graph for the loaded checkpoint-qualified function"
        )

    return {
        "schema": "salman2020do_r50_checkpoint_proof_binding_v1",
        "model_id": receipt.model_id,
        "model_load_receipt_hash": receipt.content_hash,
        "model_state_sha256": receipt.state_dict_hash,
        "checkpoint_sha256": receipt.checkpoint_hash,
        "source_contract_sha256": source.content_hash,
        "source_release_id": source.release_id,
        "source_artifact_id": source.artifact_id,
        "source_checksum_authority": source.checkpoint_sha256_authority,
        "topology_hash": architecture.topology_hash,
        "route_schema_hash": architecture.route_schema_hash,
        "circuit_route_schema_hash": architecture.circuit_route_schema_hash,
        "preprocessing_sha256": preprocessing.content_hash,
        "normalization_placement": source.normalization_placement,
        "normalization_owner_count": 1,
        "evaluator_id": evaluator.evaluator_id,
        "evaluator_binding": (
            "loaded_source_model_identity+compiled_gate_graph_identity"
        ),
    }


__all__ = ["robust_resnet50_checkpoint_proof_binding"]
