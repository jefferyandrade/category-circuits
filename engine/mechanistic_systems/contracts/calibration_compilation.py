"""Capability-neutral calibration contract compilation."""

from __future__ import annotations

from mechanistic_systems.contracts import (
    CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON,
    CANONICAL_CALIBRATION_HESSIAN_RIDGE,
    CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS,
    CANONICAL_CALIBRATION_KKT_TOLERANCE,
    CANONICAL_CALIBRATION_MAX_ITERATIONS,
    CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE,
    CANONICAL_CALIBRATION_REFERENCE_TOLERANCE,
    CANONICAL_CALIBRATION_SELECTION_BLOCKS,
    CalibrationProblemSpec,
    PROPER_SCORE_SHRINKAGE_AFFINE_V2,
    QueryOperatorSpec,
    QuerySpec,
    ReadoutProjectionSpec,
    SAMPLED_PATH_V1_CALIBRATION_ALGORITHM,
)


def coordinate_calibration_contracts(
    query: QuerySpec,
    *,
    response_width: int,
    class_id: int,
    weighting: str,
    minimum_scale: float,
    fold_ids: tuple[str, ...],
    optimization_contract: str,
    column_ids: tuple[int, ...] | None = None,
    calibration_penalty: str = "identity_anchor",
    identity_anchor_strengths: tuple[float, ...] = (
        CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS
    ),
    calibration_selection_blocks: int = CANONICAL_CALIBRATION_SELECTION_BLOCKS,
    anchor_probability_epsilon: float = (
        CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON
    ),
    max_iterations: int = CANONICAL_CALIBRATION_MAX_ITERATIONS,
    kkt_tolerance: float = CANONICAL_CALIBRATION_KKT_TOLERANCE,
    hessian_ridge: float = CANONICAL_CALIBRATION_HESSIAN_RIDGE,
    objective_tolerance: float = CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE,
    reference_tolerance: float = CANONICAL_CALIBRATION_REFERENCE_TOLERANCE,
) -> tuple[ReadoutProjectionSpec, QueryOperatorSpec, CalibrationProblemSpec]:
    output_axes = tuple(f"readout_output:{index}" for index in range(response_width))
    readout = ReadoutProjectionSpec(
        projection_id=f"{query.query_id}:fixed_readout",
        projection_kind="identity",
        source_axis_ids=output_axes,
        primitive_axis_ids=output_axes,
        source_hash=query.source_hash,
        notes={"fixed_readout_constructed_before_calibration": True},
    )
    if column_ids is None:
        column_ids = (int(class_id),)
    if len(column_ids) == 1:
        query_axis_ids = (query.query_id,)
    else:
        query_axis_ids = tuple(
            f"{query.query_id}:col:{int(column)}" for column in column_ids
        )
    operator = QueryOperatorSpec.coordinate(
        operator_id=f"{query.query_id}:coordinate",
        primitive_axis_ids=output_axes,
        query_axis_ids=query_axis_ids,
        query_primitive_axis_ids=tuple(
            output_axes[int(column)] for column in column_ids
        ),
        source_hash=readout.content_hash,
    )
    problem = CalibrationProblemSpec(
        problem_id=f"{query.query_id}:sampled_path_output_affine",
        readout_projection_hash=readout.content_hash,
        query_operator_hash=operator.content_hash,
        parameterization="post_query_positive_affine",
        loss_family="independent_binary_bce",
        weighting=weighting,
        fold_ids=fold_ids,
        minimum_scale=minimum_scale,
        notes={
            "sampled_path_frontier_version": "sampled_path_frontier_v1",
            "optimization_contract": optimization_contract,
            "optimization_estimand": (
                "finite_horizon_algorithmic_fit"
                if optimization_contract == SAMPLED_PATH_V1_CALIBRATION_ALGORITHM
                else "calibration_only_selected_identity_anchored_proper_score"
            ),
            "fixed_readout_centering": (
                "subtract_bias_before_affine"
                if optimization_contract == SAMPLED_PATH_V1_CALIBRATION_ALGORITHM
                else "post_head_query_output_identity_map"
            ),
            **(
                {
                    "identity_anchor_strengths": list(identity_anchor_strengths),
                    "calibration_selection_blocks": int(calibration_selection_blocks),
                    "anchor_probability_epsilon": float(anchor_probability_epsilon),
                    "regularization_geometry": ("bernoulli_predictive_kl_to_identity"),
                    "selection_rule": ("largest_identity_anchor_within_one_se_of_best"),
                    "max_iterations": int(max_iterations),
                    "kkt_tolerance": float(kkt_tolerance),
                    "hessian_ridge": float(hessian_ridge),
                    "objective_tolerance": float(objective_tolerance),
                    "reference_tolerance": float(reference_tolerance),
                }
                if optimization_contract == PROPER_SCORE_SHRINKAGE_AFFINE_V2 and calibration_penalty != "firth"
                else {}
            ),
            **({
                "calibration_penalty": "firth",
                "optimization_estimand": "firth_bias_reduced_metric_weighted_likelihood",
                "regularization_geometry": "jeffreys_information_log_determinant",
                "selection_rule": "single_full_training_fit",
                "max_iterations": int(max_iterations), "kkt_tolerance": float(kkt_tolerance),
                "hessian_ridge": float(hessian_ridge), "objective_tolerance": float(objective_tolerance),
                "reference_tolerance": float(reference_tolerance),
            } if calibration_penalty == "firth" else {}),
        },
    )
    return readout, operator, problem


__all__ = ["coordinate_calibration_contracts"]
