"""Lazy facade for architecture-neutral numerical backends.

Importing a leaf backend must not initialize every optional Torch/Triton or
workflow-facing backend.  Public objects retain their defining-module identity
and are loaded only when requested from this compatibility facade.
"""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType


_EXPORT_GROUPS = {
    ".contracts": (
        "BackendProfile", "BackendPromotionDecision", "BackendRegistry",
        "CalibrationBackend", "CandidateExecutionBackend", "ContractionBackend",
        "GateGraphCompiler", "RouteReducerBackend", "RuntimeTile",
        "RuntimeTuningSpace", "assess_backend_promotion",
        "select_feasible_runtime_tile",
    ),
    ".calibration_backend": (
        "AffineCalibrationFit", "CanonicalCalibrationBackend",
        "ReadoutAffineCalibrationFit",
    ),
    ".calibration_problem": (
        "CalibrationComponent", "CalibrationFactorization",
        "CalibrationProblemCompiler",
    ),
    ".calibration_runtime": (
        "CalibrationExecutionContext", "CalibrationExecutionTopology",
        "CalibrationSolverCapability", "CalibrationSolverCapabilityProtocol",
        "CrossFitLayout", "CrossFitSupport", "PreparedCalibrationState",
        "RuntimeExecutionProfile", "RuntimeExecutionSession",
        "calibration_solver_capability",
    ),
    ".calibration_statistics": (
        "CalibrationStatistics", "StratifiedAffineBCEUncertainty",
        "calibration_statistics", "stratified_affine_bce_uncertainty",
        "torch_calibration_statistics", "triton_calibration_statistics",
        "triton_calibration_statistics_available",
    ),
    ".factored_calibration": (
        "CalibrationConvergenceError", "CalibrationFitReceipt",
        "FactoredCalibrationFit", "fit_batched_post_query_affine",
        "fit_shared_temperature_multiclass",
    ),
    ".sampled_path_v1_calibration": (
        "SAMPLED_PATH_V1_CALIBRATION_ALGORITHM", "SampledPathV1CalibrationFit",
        "SampledPathV1CalibrationReceipt", "fit_sampled_path_v1_coordinate_affine",
    ),
    ".proper_score_shrinkage": (
        "DEFAULT_IDENTITY_ANCHOR_STRENGTHS", "IDENTITY_CANDIDATE",
        "ProperScoreShrinkageFit", "ProperScoreShrinkageReceipt",
        "deterministic_fold_local_calibration_blocks",
        "deterministic_stratified_calibration_blocks",
        "fit_proper_score_shrinkage_affine",
    ),
    ".profiling": ("PhaseProfiler", "nvtx_range"),
    ".torch_gate_graph": (
        "CompiledGateGraph", "GatedConv2d", "GatedLinear", "GatedResidualBlock",
        "TorchGateGraphCompiler",
    ),
    ".torch_runtime": (
        "DirectTorchConvRouteReducerBackend", "TorchCandidateExecutionBackend",
        "TorchContractionBackend", "TorchRouteReducerBackend",
        "conv_channel_route_measurement_target_batched",
        "conv_channel_route_measurement_direct_target_batched",
        "linear_route_measurement_target_batched",
    ),
    ".structural_overlap": (
        "PackedIntersectionResult", "StructuralOverlapBackendUnavailable",
        "compute_packed_intersection_counts", "structural_triton_available",
    ),
    ".defaults": ("build_default_backend_registry",),
    ".conformance": (
        "PHYSICAL_BACKEND_CONFORMANCE_SCHEMA",
        "PhysicalBackendConformanceReceipt",
    ),
    ".hardware_cost": (
        "HARDWARE_COST_AUTHORITY_CEILING", "ComputeLedger", "HardwareCostFeatures",
        "HardwareCostObservation", "HardwareCostPrediction", "OnlineHardwareCostModel",
    ),
    ".hardware_memory": (
        "MEMORY_ESTIMATE_AUTHORITY_CEILING", "DeviceMemoryBudget",
        "ResponseLandscapeMemoryEstimate", "declared_h200_141gb_budget",
        "estimate_response_landscape_memory",
    ),
    ".torch_response_landscape": (
        "TORCH_FANTASY_SELECTION_SCHEMA", "TORCH_RELAXED_PROPOSAL_OPTIMIZATION_SCHEMA",
        "TORCH_RESPONSE_LANDSCAPE_AUTHORITY_CEILING",
        "TORCH_RESPONSE_LANDSCAPE_MIRROR_SCHEMA", "TorchFantasyCoordinatedSelection",
        "TorchMechanismFeatureMirror", "TorchRelaxedProposalOptimization",
        "TorchRelaxedProposalOptimizationReceipt", "TorchResponseChartMirror",
        "TorchResponseLandscapeMirrorReceipt", "TorchResponsePosteriorScorer",
        "optimize_relaxed_information_proposal_torch",
    ),
}

_LAZY_EXPORTS = {
    name: module_name
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = list(_LAZY_EXPORTS)


class _LazyFacadeModule(ModuleType):
    """Resolve a public symbol even when its leaf module has the same name.

    Python installs an imported submodule on its parent package.  Consequently,
    importing ``backends.calibration_statistics`` used to replace this
    facade's callable ``calibration_statistics`` export with the module object,
    making the public API depend on import order.  Intercept only that general
    module/symbol collision and cache the defining object, preserving lazy
    loading for every untouched backend.
    """

    def __getattribute__(self, name: str) -> object:
        value = ModuleType.__getattribute__(self, name)
        exports = ModuleType.__getattribute__(self, "_LAZY_EXPORTS")
        module_name = exports.get(name)
        if module_name is not None and isinstance(value, ModuleType):
            package_name = ModuleType.__getattribute__(self, "__name__")
            if value.__name__ == package_name + module_name:
                resolved = getattr(value, name)
                ModuleType.__setattr__(self, name, resolved)
                return resolved
        return value


sys.modules[__name__].__class__ = _LazyFacadeModule
