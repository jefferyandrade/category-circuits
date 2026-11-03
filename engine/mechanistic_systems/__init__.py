"""Intrinsic mechanism inference and response-system identification.

The package root is a lazy compatibility facade.  Importing a neutral owner
such as ``mechanistic_systems.contracts`` or ``mechanistic_systems.geometry``
must not transitively load circuit workflows, engines, or artifact execution
stacks.  Existing root-level public names retain object identity and are
resolved only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final


SCIENTIFIC_CHAIN = (
    "represented_state",
    "physical_route_differential",
    "behavior_preserving_circuit_family",
    "frozen_hierarchical_mechanism",
    "external_response_prediction",
    "prospective_adjudication",
)
NEGATIVE_RESULT_CLASSES = (
    "activation_dominant",
    "dense_attribution_equivalent",
    "route_coordinate_equivalent",
    "circuit_denoising_only",
    "mechanism_gain_absent",
    "mapping_dependent",
    "structural_null_equivalent",
    "unstable_signed_measure",
    "observationally_equivalent",
    "passive_data_insufficient",
)

_LAZY_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "ActionArtifact": (".artifacts", "ActionArtifact"),
    "CircuitArtifact": (".artifacts", "CircuitArtifact"),
    "GeometryArtifact": (".artifacts", "GeometryArtifact"),
    "InferenceArtifact": (".artifacts", "InferenceArtifact"),
    "MeasurementArtifact": (".artifacts", "MeasurementArtifact"),
    "MechanismResponseArtifact": (".artifacts", "MechanismResponseArtifact"),
    "ResponseArtifact": (".artifacts", "ResponseArtifact"),
    "ResponseBasisArtifact": (".artifacts", "ResponseBasisArtifact"),
    "StateResponseArtifact": (".artifacts", "StateResponseArtifact"),
    "HierarchicalMechanismRealization": (
        ".mechanisms",
        "HierarchicalMechanismRealization",
    ),
    "AnalysisPlan": (".contracts", "AnalysisPlan"),
    "CircuitPlan": (".contracts", "CircuitPlan"),
    "DataView": (".contracts", "DataView"),
    "GeometryPlan": (".contracts", "GeometryPlan"),
    "InterventionPlan": (".contracts", "InterventionPlan"),
    "KernelContractionSpec": (".contracts", "KernelContractionSpec"),
    "MeasurementPlan": (".contracts", "MeasurementPlan"),
    "MechanismPlan": (".contracts", "MechanismPlan"),
    "QuerySpec": (".contracts", "QuerySpec"),
    "RouteCoordinateSchema": (".contracts", "RouteCoordinateSchema"),
    "RuntimePlan": (".contracts", "RuntimePlan"),
    "StudySpec": (".contracts", "StudySpec"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = [
    "ActionArtifact",
    "AnalysisPlan",
    "CircuitArtifact",
    "CircuitPlan",
    "DataView",
    "GeometryArtifact",
    "GeometryPlan",
    "HierarchicalMechanismRealization",
    "InferenceArtifact",
    "InterventionPlan",
    "KernelContractionSpec",
    "MeasurementArtifact",
    "MeasurementPlan",
    "MechanismPlan",
    "MechanismResponseArtifact",
    "NEGATIVE_RESULT_CLASSES",
    "QuerySpec",
    "ResponseArtifact",
    "ResponseBasisArtifact",
    "RouteCoordinateSchema",
    "RuntimePlan",
    "SCIENTIFIC_CHAIN",
    "StateResponseArtifact",
    "StudySpec",
]
