"""CircuitFlow v2: typed, content-addressed circuit experiment workflow.

CircuitFlow is the active orchestration boundary for new circuit-geometry work.
It separates scientific identity from score, circuit, analysis, and runtime
plans, then connects those plans through immutable artifact bundles.

Legacy ScoreStores, pruning roots, and report directories remain readable only
through :mod:`circuitflow.legacy` import adapters. Public names resolve lazily
so artifact and contract consumers do not import workflow engines as a package
initialization side effect.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_GROUPS = {
    ".analysis": (
        "AnalysisInputValidation",
        "build_inference_artifact",
        "validate_analysis_inputs",
    ),
    ".artifacts": (
        "ArtifactManifest",
        "ArtifactStore",
        "ArtifactType",
        "CertificationReceipt",
        "LifecycleState",
        "RuntimeReceipt",
    ),
    ".contracts": (
        "ActionPlan",
        "AnalysisPlan",
        "CircuitPlan",
        "DataView",
        "GeometryPlan",
        "KernelContractionSpec",
        "MechanismPlan",
        "RouteCoordinateSchema",
        "RuntimePlan",
        "ScorePlan",
        "StudySpec",
    ),
    ".engines": (
        "THERMOBAYES_PROGRAM_CLAIM_BINDING_KEYS",
        "ClassicFrontierEngine",
        "ThermobayesProgramEngine",
        "validate_thermobayes_program_claim_bindings",
    ),
    ".registry": ("ArtifactRegistry",),
}

_LAZY_EXPORTS = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}

_PUBLIC_NAMES = (
    "ActionPlan",
    "AnalysisInputValidation",
    "AnalysisPlan",
    "ArtifactManifest",
    "ArtifactRegistry",
    "ArtifactStore",
    "ArtifactType",
    "CertificationReceipt",
    "CircuitPlan",
    "ClassicFrontierEngine",
    "DataView",
    "GeometryPlan",
    "KernelContractionSpec",
    "LifecycleState",
    "MechanismPlan",
    "RouteCoordinateSchema",
    "RuntimePlan",
    "RuntimeReceipt",
    "ScorePlan",
    "StudySpec",
    "THERMOBAYES_PROGRAM_CLAIM_BINDING_KEYS",
    "ThermobayesProgramEngine",
    "build_inference_artifact",
    "validate_analysis_inputs",
    "validate_thermobayes_program_claim_bindings",
)


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = list(_PUBLIC_NAMES)
