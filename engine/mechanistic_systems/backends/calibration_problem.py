"""Compile exact calibration independence from declared dependency support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contracts.hashing import stable_digest
from mechanistic_systems.contracts import (
    CalibrationProblemSpec,
    QueryOperatorSpec,
    ReadoutProjectionSpec,
)


@dataclass(frozen=True)
class CalibrationComponent:
    component_id: str
    factor_ids: tuple[str, ...]
    query_axis_ids: tuple[str, ...]
    parameter_axis_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "factor_ids": list(self.factor_ids),
            "query_axis_ids": list(self.query_axis_ids),
            "parameter_axis_ids": list(self.parameter_axis_ids),
        }


@dataclass(frozen=True)
class CalibrationFactorization:
    problem_hash: str
    dependency_hash: str
    components: tuple[CalibrationComponent, ...]
    dependency_rows: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def factorization_hash(self) -> str:
        return stable_digest(self.to_dict())

    @property
    def is_fully_factorized(self) -> bool:
        return all(
            len(component.query_axis_ids) == 1
            and len(component.parameter_axis_ids) == 2
            for component in self.components
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": "mechanistic_calibration_factorization",
            "schema_version": "1",
            "problem_hash": self.problem_hash,
            "dependency_hash": self.dependency_hash,
            "components": [component.to_dict() for component in self.components],
            "dependency_rows": [
                {"factor_id": factor_id, "parameter_axis_ids": list(parameters)}
                for factor_id, parameters in self.dependency_rows
            ],
        }


class CalibrationProblemCompiler:
    """Derive independent optimization components without model-type dispatch."""

    compiler_id = "mechanistic_systems.calibration_problem_compiler.v1"

    @staticmethod
    def _parameter_groups(
        problem: CalibrationProblemSpec,
        query: QueryOperatorSpec,
    ) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
        if problem.parameterization == "post_query_positive_affine":
            calibrated = (
                problem.calibration_axis_ids
                if problem.calibration_axis_ids
                else query.query_axis_ids
            )
            if set(calibrated) != set(query.query_axis_ids):
                raise ValueError(
                    "post-query affine calibration must cover every declared query"
                )
            by_query = {
                axis: (f"query:{axis}:scale", f"query:{axis}:offset")
                for axis in query.query_axis_ids
            }
            parameters = tuple(
                value
                for axis in query.query_axis_ids
                for value in by_query[axis]
            )
            return parameters, by_query
        if problem.parameterization == "primitive_diagonal_affine":
            supported = tuple(
                dict.fromkeys(axis for row in query.support for axis in row)
            )
            calibrated = (
                problem.calibration_axis_ids
                if problem.calibration_axis_ids
                else supported
            )
            if not set(calibrated).issubset(set(query.primitive_axis_ids)):
                raise ValueError("primitive calibration names unknown readout coordinates")
            if not set(supported).issubset(set(calibrated)):
                raise ValueError(
                    "primitive calibration omits coordinates used by a declared query"
                )
            by_primitive = {
                axis: (f"primitive:{axis}:scale", f"primitive:{axis}:offset")
                for axis in calibrated
            }
            by_query = {
                query_axis: tuple(
                    parameter
                    for primitive_axis in support
                    for parameter in by_primitive[primitive_axis]
                )
                for query_axis, support in zip(
                    query.query_axis_ids, query.support, strict=True
                )
            }
            parameters = tuple(
                value
                for axis in calibrated
                for value in by_primitive[axis]
            )
            return parameters, by_query
        if problem.parameterization == "shared_temperature":
            parameters = ("shared:temperature",)
            return parameters, {
                axis: parameters for axis in query.query_axis_ids
            }
        raise ValueError(f"unsupported calibration parameterization {problem.parameterization!r}")

    def compile(
        self,
        *,
        readout: ReadoutProjectionSpec,
        query: QueryOperatorSpec,
        problem: CalibrationProblemSpec,
    ) -> CalibrationFactorization:
        readout.validate()
        query.validate()
        problem.validate()
        if readout.primitive_axis_ids != query.primitive_axis_ids:
            raise ValueError("readout and query primitive axes differ")
        if problem.readout_projection_hash != readout.content_hash:
            raise ValueError("calibration problem readout parent hash differs")
        if problem.query_operator_hash != query.content_hash:
            raise ValueError("calibration problem query parent hash differs")

        all_parameters, parameters_by_query = self._parameter_groups(problem, query)
        queries_by_factor: dict[str, tuple[str, ...]] = {}
        if problem.loss_family == "independent_binary_bce":
            for axis in query.query_axis_ids:
                queries_by_factor[f"binary_bce:{axis}"] = (axis,)
        elif problem.loss_family == "multiclass_cross_entropy":
            queries_by_factor["multiclass_cross_entropy"] = query.query_axis_ids
        else:  # validated above; retained as a fail-closed guard.
            raise ValueError(f"unsupported calibration loss family {problem.loss_family!r}")

        dependency_rows = tuple(
            (
                factor_id,
                tuple(
                    dict.fromkeys(
                        parameter
                        for query_axis in factor_queries
                        for parameter in parameters_by_query[query_axis]
                    )
                ),
            )
            for factor_id, factor_queries in queries_by_factor.items()
        )
        dependency_hash = stable_digest(
            {
                "compiler_id": self.compiler_id,
                "query_support": [list(row) for row in query.support],
                "dependency_rows": [
                    [factor_id, list(parameters)]
                    for factor_id, parameters in dependency_rows
                ],
            }
        )

        # Components in the factor-parameter bipartite graph are the exact
        # independent optimization problems. Query labels are annotations on
        # factor nodes and cannot introduce architecture-dependent dispatch.
        factor_neighbors = {factor_id: set(parameters) for factor_id, parameters in dependency_rows}
        parameter_neighbors = {parameter: set() for parameter in all_parameters}
        for factor_id, parameters in dependency_rows:
            for parameter in parameters:
                parameter_neighbors.setdefault(parameter, set()).add(factor_id)

        visited_factors: set[str] = set()
        visited_parameters: set[str] = set()
        components: list[CalibrationComponent] = []
        query_order = {axis: index for index, axis in enumerate(query.query_axis_ids)}
        parameter_order = {axis: index for index, axis in enumerate(all_parameters)}
        for root_factor in queries_by_factor:
            if root_factor in visited_factors:
                continue
            pending: list[tuple[str, str]] = [("factor", root_factor)]
            factors: set[str] = set()
            parameters: set[str] = set()
            while pending:
                kind, node = pending.pop()
                if kind == "factor":
                    if node in visited_factors:
                        continue
                    visited_factors.add(node)
                    factors.add(node)
                    pending.extend(("parameter", value) for value in factor_neighbors[node])
                else:
                    if node in visited_parameters:
                        continue
                    visited_parameters.add(node)
                    parameters.add(node)
                    pending.extend(("factor", value) for value in parameter_neighbors[node])
            component_queries = tuple(
                sorted(
                    {
                        query_axis
                        for factor_id in factors
                        for query_axis in queries_by_factor[factor_id]
                    },
                    key=query_order.__getitem__,
                )
            )
            ordered_parameters = tuple(sorted(parameters, key=parameter_order.__getitem__))
            component_id = stable_digest(
                {
                    "problem_hash": problem.content_hash,
                    "factors": sorted(factors),
                    "queries": list(component_queries),
                    "parameters": list(ordered_parameters),
                }
            )[:16]
            components.append(
                CalibrationComponent(
                    component_id=component_id,
                    factor_ids=tuple(sorted(factors)),
                    query_axis_ids=component_queries,
                    parameter_axis_ids=ordered_parameters,
                )
            )

        components.sort(key=lambda item: query_order[item.query_axis_ids[0]])
        if set(visited_parameters) != set(all_parameters):
            raise ValueError("calibration parameterization contains unidentifiable axes")
        return CalibrationFactorization(
            problem_hash=problem.content_hash,
            dependency_hash=dependency_hash,
            components=tuple(components),
            dependency_rows=dependency_rows,
        )


__all__ = [
    "CalibrationComponent",
    "CalibrationFactorization",
    "CalibrationProblemCompiler",
]
