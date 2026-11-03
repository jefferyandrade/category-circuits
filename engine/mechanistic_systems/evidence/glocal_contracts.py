"""Stable contracts for the shadow glocal v3 evidence foundation.

The v3 foundation separates exploratory estimators from the objects allowed to
carry a pruning-safety claim.  Claim-bearing losses are bounded Brier losses;
the sampled unit, population, sealed order, evidence role, calibration state,
and proposal are all content addressed before adjudication outcomes are
observed.

This module deliberately has no dependency on the circuit runtime.  Runtime
adapters may consume these contracts, while historical v1/v2 evidence remains
unchanged and replayable.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np


EVIDENCE_FOUNDATION_GLOCAL_V3 = "evidence_foundation_glocal_v3"
GLOCAL_CONTRACT_SCHEMA = "glocal_statistical_contracts_v1"
JOINT_RISK_QUERY_ID = "__joint_risk__"


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _canonical(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(
                np.ascontiguousarray(array).view(np.uint8)
            ).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        items = list(value)
        if isinstance(value, (set, frozenset)):
            items.sort(
                key=lambda item: json.dumps(
                    _canonical(item), sort_keys=True, separators=(",", ":")
                )
            )
        return [_canonical(item) for item in items]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("content-addressed statistical values must be finite")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported statistical contract value: {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_hash(value: str, label: str) -> str:
    identity = str(value)
    if len(identity) != 64 or any(
        character not in "0123456789abcdef" for character in identity
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return identity


def _is_sha256(value: str) -> bool:
    identity = str(value)
    return len(identity) == 64 and all(
        character in "0123456789abcdef" for character in identity
    )


def _normalized_weights(
    values: Sequence[tuple[str, float]],
    *,
    label: str,
) -> tuple[tuple[str, float], ...]:
    weights = tuple((str(key), float(weight)) for key, weight in values)
    if not weights or len({key for key, _ in weights}) != len(weights):
        raise ValueError(f"{label} must have nonempty unique identities")
    if any(not key for key, _ in weights):
        raise ValueError(f"{label} identities must be nonempty")
    if any(not math.isfinite(weight) or weight <= 0.0 for _, weight in weights):
        raise ValueError(f"{label} weights must be finite and positive")
    total = math.fsum(weight for _, weight in weights)
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise ValueError(f"{label} weights must sum to one")
    return tuple(sorted((key, weight / total) for key, weight in weights))


PredictionKind = Literal["binary", "multiclass"]


@dataclass(frozen=True, slots=True)
class PredictionRuleSpec:
    """Frozen conversion from calibrated probabilities to declared decisions."""

    kind: PredictionKind
    class_ids: tuple[int, ...]
    threshold: float | None = None
    positive_class_id: int | None = None
    tie_break: str = "lowest_class_id"
    schema_name: str = "prediction_rule_v1"

    def __post_init__(self) -> None:
        classes = tuple(int(value) for value in self.class_ids)
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("prediction rule requires at least two unique class IDs")
        if self.kind == "binary":
            if len(classes) != 2:
                raise ValueError("binary prediction rule requires exactly two classes")
            threshold = 0.5 if self.threshold is None else float(self.threshold)
            positive = (
                classes[-1]
                if self.positive_class_id is None
                else int(self.positive_class_id)
            )
            if positive not in classes:
                raise ValueError("binary positive class must appear in class_ids")
            if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
                raise ValueError("binary threshold must lie strictly inside (0,1)")
            object.__setattr__(self, "threshold", threshold)
            object.__setattr__(self, "positive_class_id", positive)
        elif self.kind == "multiclass":
            if self.threshold is not None or self.positive_class_id is not None:
                raise ValueError(
                    "multiclass argmax rules do not accept a binary threshold"
                )
        else:
            raise ValueError(f"unknown prediction kind {self.kind!r}")
        if self.tie_break != "lowest_class_id":
            raise ValueError(
                "v3 supports only the deterministic lowest-class-ID tie break"
            )
        object.__setattr__(self, "class_ids", classes)

    @property
    def content_hash(self) -> str:
        return content_hash(self)


def binary_brier_loss(probabilities: object, labels: object) -> np.ndarray:
    """Binary Brier loss in ``[0,1]`` with strict probability/label checks."""

    probability = np.asarray(probabilities, dtype=np.float64)
    label = np.asarray(labels)
    try:
        probability, label = np.broadcast_arrays(probability, label)
    except ValueError as error:
        raise ValueError(
            "binary Brier probabilities and labels must broadcast"
        ) from error
    if (
        not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise ValueError("binary Brier probabilities must lie in [0,1]")
    if not np.all(np.isin(label, (0, 1, False, True))):
        raise ValueError("binary Brier labels must be zero or one")
    result = np.square(probability - label.astype(np.float64, copy=False))
    return np.clip(result, 0.0, 1.0)


def normalized_multiclass_brier_loss(
    probabilities: object, labels: object
) -> np.ndarray:
    """Normalized multiclass Brier loss ``0.5 ||p-e_y||²`` in ``[0,1]``."""

    probability = np.asarray(probabilities, dtype=np.float64)
    label = np.asarray(labels)
    if probability.ndim < 2 or probability.shape[-1] < 2:
        raise ValueError("multiclass Brier probabilities require a class axis")
    if label.shape != probability.shape[:-1]:
        raise ValueError("multiclass Brier labels must match non-class axes")
    if (
        not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise ValueError("multiclass probabilities must lie in [0,1]")
    if not np.allclose(probability.sum(axis=-1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("multiclass probability rows must sum to one")
    if not np.issubdtype(label.dtype, np.integer):
        if not np.all(np.equal(label, np.floor(label))):
            raise ValueError("multiclass labels must be integer class indices")
        label = label.astype(np.int64)
    label = label.astype(np.int64, copy=False)
    class_count = probability.shape[-1]
    if np.any(label < 0) or np.any(label >= class_count):
        raise ValueError("multiclass label left the probability class axis")
    one_hot = np.zeros_like(probability)
    np.put_along_axis(one_hot, np.expand_dims(label, axis=-1), 1.0, axis=-1)
    return np.clip(0.5 * np.square(probability - one_hot).sum(axis=-1), 0.0, 1.0)


def paired_brier_increment(
    candidate_probabilities: object,
    parent_probabilities: object,
    labels: object,
    *,
    prediction_rule: PredictionRuleSpec,
) -> np.ndarray:
    """Candidate-minus-parent paired Brier increment, always in ``[-1,1]``."""

    declared_labels = np.asarray(labels)
    class_ids = prediction_rule.class_ids
    if not np.all(np.isin(declared_labels, class_ids)):
        raise ValueError("paired Brier label left the declared prediction classes")
    if prediction_rule.kind == "binary":
        binary_labels = declared_labels == int(prediction_rule.positive_class_id)
        candidate = binary_brier_loss(candidate_probabilities, binary_labels)
        parent = binary_brier_loss(parent_probabilities, binary_labels)
    else:
        class_index = {class_id: index for index, class_id in enumerate(class_ids)}
        indexed_labels = np.asarray(
            [class_index[int(value)] for value in declared_labels.reshape(-1)],
            dtype=np.int64,
        ).reshape(declared_labels.shape)
        candidate = normalized_multiclass_brier_loss(
            candidate_probabilities, indexed_labels
        )
        parent = normalized_multiclass_brier_loss(parent_probabilities, indexed_labels)
    if candidate.shape != parent.shape:
        raise ValueError("paired Brier states must share an observation axis")
    result = candidate - parent
    if np.any(result < -1.0 - 1e-12) or np.any(result > 1.0 + 1e-12):
        raise RuntimeError("bounded Brier increment escaped [-1,1]")
    return np.clip(result, -1.0, 1.0)


def class_balanced_observation_weights(
    labels: object,
    *,
    class_ids: Sequence[int],
) -> np.ndarray:
    """Weights observations so each declared class contributes equal total mass."""

    label = np.asarray(labels)
    classes = tuple(int(value) for value in class_ids)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("class-balanced weights require unique declared classes")
    if not np.all(np.isin(label, classes)):
        raise ValueError("class-balanced label left the declared class axis")
    weights = np.zeros(label.shape, dtype=np.float64)
    class_mass = 1.0 / len(classes)
    for class_id in classes:
        selected = label == class_id
        count = int(np.count_nonzero(selected))
        if count == 0:
            raise ValueError("class-balanced risk requires every declared class")
        weights[selected] = class_mass / count
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12):
        raise RuntimeError("class-balanced observation weights failed to normalize")
    return weights


@dataclass(frozen=True, slots=True)
class RiskEstimandSpec:
    """Frozen globally weighted bounded-Brier risk."""

    estimand_id: str
    context_population_hash: str
    prediction_rule: PredictionRuleSpec
    query_weights: tuple[tuple[str, float], ...]
    strata_by_query: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    binary_class_by_stratum_by_query: tuple[
        tuple[str, tuple[tuple[str, int], ...]], ...
    ] = ()
    weighting_policy: str = "binary_class_balanced"
    nondefault_weighting_rationale: str | None = None
    loss_id: str = "bounded_brier_v1"
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    schema_name: str = "risk_estimand_v1"

    def __post_init__(self) -> None:
        if not self.estimand_id:
            raise ValueError("risk estimand requires estimand_id")
        _require_hash(self.context_population_hash, "context population hash")
        if self.loss_id != "bounded_brier_v1":
            raise ValueError("glocal v3 claim-bearing risk must use bounded Brier loss")
        if self.lower_bound != 0.0 or self.upper_bound != 1.0:
            raise ValueError("bounded Brier risk has immutable bounds [0,1]")
        query_weights = _normalized_weights(self.query_weights, label="query")
        strata = tuple(
            (
                str(query_id),
                _normalized_weights(weights, label=f"strata for {query_id!r}"),
            )
            for query_id, weights in self.strata_by_query
        )
        if len({query_id for query_id, _ in strata}) != len(strata):
            raise ValueError("risk estimand query strata must be unique")
        if {query_id for query_id, _ in query_weights} != {
            query_id for query_id, _ in strata
        }:
            raise ValueError("risk estimand query and stratum axes differ")
        binary_mapping = tuple(
            (
                str(query_id),
                tuple(
                    (str(stratum_id), int(class_id)) for stratum_id, class_id in rows
                ),
            )
            for query_id, rows in self.binary_class_by_stratum_by_query
        )
        if self.prediction_rule.kind == "binary":
            if not binary_mapping:
                canonical_strata = {
                    f"class:{class_id}": class_id
                    for class_id in self.prediction_rule.class_ids
                }
                if all(
                    set(dict(weights)) == set(canonical_strata) for _, weights in strata
                ):
                    binary_mapping = tuple(
                        (
                            query_id,
                            tuple(sorted(canonical_strata.items())),
                        )
                        for query_id, _ in strata
                    )
                else:
                    raise ValueError(
                        "binary risk requires explicit frozen stratum-to-class mapping"
                    )
            if len({query_id for query_id, _ in binary_mapping}) != len(binary_mapping):
                raise ValueError("binary risk mapping query IDs must be unique")
            if {query_id for query_id, _ in binary_mapping} != {
                query_id for query_id, _ in strata
            }:
                raise ValueError(
                    "binary risk mapping query axis differs from risk strata"
                )
            mapping_by_query: dict[str, tuple[tuple[str, int], ...]] = {}
            for query_id, rows in binary_mapping:
                if len({stratum_id for stratum_id, _ in rows}) != len(rows):
                    raise ValueError(
                        "binary risk contains duplicate stratum-to-class mapping"
                    )
                expected_strata = {
                    stratum_id for stratum_id, _ in dict(strata)[query_id]
                }
                mapped_strata = {stratum_id for stratum_id, _ in rows}
                if mapped_strata != expected_strata:
                    raise ValueError(
                        "binary risk stratum-to-class mapping is missing or foreign"
                    )
                mapped_classes = {class_id for _, class_id in rows}
                if mapped_classes != set(self.prediction_rule.class_ids):
                    raise ValueError(
                        "binary risk mapping must cover both declared classes"
                    )
                mapping_by_query[query_id] = tuple(sorted(rows))
            binary_mapping = tuple(sorted(mapping_by_query.items()))
        elif binary_mapping:
            raise ValueError("multiclass risk cannot carry a binary stratum mapping")

        if self.weighting_policy == "binary_class_balanced":
            if self.prediction_rule.kind != "binary":
                raise ValueError(
                    "binary_class_balanced weighting requires a binary prediction rule"
                )
            normalized_strata: list[tuple[str, tuple[tuple[str, float], ...]]] = []
            mapping_lookup = dict(binary_mapping)
            for query_id, weights in strata:
                class_by_stratum = dict(mapping_lookup[query_id])
                side_mass = {
                    class_id: math.fsum(
                        weight
                        for stratum_id, weight in weights
                        if class_by_stratum[stratum_id] == class_id
                    )
                    for class_id in self.prediction_rule.class_ids
                }
                if any(
                    not math.isclose(mass, 0.5, abs_tol=1e-12)
                    for mass in side_mass.values()
                ):
                    raise ValueError(
                        "default binary Brier risk requires exactly 0.5 mass "
                        "on each binary side"
                    )
                normalized_strata.append(
                    (
                        query_id,
                        tuple(
                            (
                                stratum_id,
                                weight * 0.5 / side_mass[class_by_stratum[stratum_id]],
                            )
                            for stratum_id, weight in weights
                        ),
                    )
                )
            strata = tuple(normalized_strata)
            if self.nondefault_weighting_rationale is not None:
                raise ValueError(
                    "default class-balanced risk cannot carry a nondefault rationale"
                )
        elif self.weighting_policy == "explicit_nondefault":
            rationale = str(self.nondefault_weighting_rationale or "").strip()
            if not rationale:
                raise ValueError(
                    "nondefault Brier weights require an explicit frozen rationale"
                )
            object.__setattr__(self, "nondefault_weighting_rationale", rationale)
        else:
            raise ValueError("unknown Brier weighting policy")
        total_weight = math.fsum(
            query_weight * stratum_weight
            for query_id, query_weight in query_weights
            for _, stratum_weight in dict(strata)[query_id]
        )
        if not math.isclose(total_weight, 1.0, abs_tol=1e-12):
            raise ValueError("claim-bearing Brier weights must have unit total mass")
        object.__setattr__(self, "query_weights", query_weights)
        object.__setattr__(self, "strata_by_query", tuple(sorted(strata)))
        object.__setattr__(
            self,
            "binary_class_by_stratum_by_query",
            binary_mapping,
        )

    @classmethod
    def from_mappings(
        cls,
        *,
        estimand_id: str,
        context_population_hash: str,
        prediction_rule: PredictionRuleSpec,
        query_weights: Mapping[str, float],
        strata_by_query: Mapping[str, Mapping[str, float]] | None = None,
        binary_class_by_stratum_by_query: Mapping[str, Mapping[str, int]] | None = None,
        weighting_policy: str | None = None,
        nondefault_weighting_rationale: str | None = None,
    ) -> "RiskEstimandSpec":
        if strata_by_query is None:
            if prediction_rule.kind != "binary":
                raise ValueError(
                    "multiclass risk requires explicit frozen stratum weights"
                )
            balanced = {
                f"class:{class_id}": 0.5 for class_id in prediction_rule.class_ids
            }
            strata_by_query = {
                str(query_id): dict(balanced) for query_id in query_weights
            }
            binary_class_by_stratum_by_query = {
                str(query_id): {
                    f"class:{class_id}": class_id
                    for class_id in prediction_rule.class_ids
                }
                for query_id in query_weights
            }
            selected_weighting_policy = "binary_class_balanced"
        else:
            if (
                prediction_rule.kind == "binary"
                and binary_class_by_stratum_by_query is None
            ):
                canonical_mapping = {
                    f"class:{class_id}": class_id
                    for class_id in prediction_rule.class_ids
                }
                if all(
                    set(weights) == set(canonical_mapping)
                    for weights in strata_by_query.values()
                ):
                    binary_class_by_stratum_by_query = {
                        str(query_id): dict(canonical_mapping)
                        for query_id in strata_by_query
                    }
                else:
                    raise ValueError(
                        "hierarchical binary strata require explicit frozen "
                        "stratum-to-class mapping"
                    )
            selected_weighting_policy = (
                "binary_class_balanced"
                if prediction_rule.kind == "binary" and weighting_policy is None
                else str(weighting_policy or "")
            )
            if not selected_weighting_policy:
                raise ValueError(
                    "nonbinary or arbitrary Brier weights require weighting_policy="
                    "'explicit_nondefault'"
                )
        return cls(
            estimand_id=estimand_id,
            context_population_hash=context_population_hash,
            prediction_rule=prediction_rule,
            query_weights=tuple(query_weights.items()),
            strata_by_query=tuple(
                (query_id, tuple(weights.items()))
                for query_id, weights in strata_by_query.items()
            ),
            binary_class_by_stratum_by_query=tuple(
                (
                    query_id,
                    tuple(mapping.items()),
                )
                for query_id, mapping in (
                    binary_class_by_stratum_by_query or {}
                ).items()
            ),
            weighting_policy=selected_weighting_policy,
            nondefault_weighting_rationale=nondefault_weighting_rationale,
        )

    @property
    def content_hash(self) -> str:
        return content_hash(self)


GLOCAL_SEALED_PERMUTATION_ALGORITHM_V1 = "uniform_256bit_rejection_fisher_yates_v2"
GLOCAL_SYSTEM_ENTROPY_SOURCE_V1 = "python_secrets_system_random_256bit_v1"
GLOCAL_EXTERNAL_ENTROPY_SOURCE_V1 = "externally_frozen_uniform_256bit_v1"
GLOCAL_IID_FIXED_N_RANDOMIZATION_METHOD_V1 = "declared_iid_fixed_n_draw_law_v1"
GLOCAL_IID_RUNTIME_RANDOMIZATION_METHOD_V2 = (
    "uniform_within_stratum_256bit_rejection_iid_v2"
)


def _replay_sampling_permutation(
    *,
    canonical_population: tuple[str, ...],
    entropy_blocks_by_step: tuple[tuple[str, ...], ...],
) -> tuple[str, ...]:
    values = list(canonical_population)
    expected_steps = max(0, len(values) - 1)
    if len(entropy_blocks_by_step) != expected_steps:
        raise ValueError(
            "sampling permutation entropy trace has the wrong number of steps"
        )
    range_size = 1 << 256
    for trace_index, step in enumerate(range(len(values) - 1, 0, -1)):
        blocks = tuple(str(value) for value in entropy_blocks_by_step[trace_index])
        if not blocks:
            raise ValueError(
                "sampling permutation entropy trace contains an empty step"
            )
        span = step + 1
        acceptance_limit = range_size - (range_size % span)
        values_by_attempt: list[int] = []
        for block in blocks:
            _require_hash(block, "sampling permutation entropy block")
            values_by_attempt.append(
                int.from_bytes(bytes.fromhex(block), byteorder="big", signed=False)
            )
        if any(value < acceptance_limit for value in values_by_attempt[:-1]):
            raise ValueError(
                "sampling permutation trace ignored an earlier accepted draw"
            )
        accepted = values_by_attempt[-1]
        if accepted >= acceptance_limit:
            raise ValueError("sampling permutation trace ends with a rejected draw")
        swap_index = accepted % span
        values[step], values[swap_index] = values[swap_index], values[step]
    return tuple(values)


def _draw_sampling_permutation(
    canonical_population: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    values = list(canonical_population)
    trace: list[tuple[str, ...]] = []
    range_size = 1 << 256
    for step in range(len(values) - 1, 0, -1):
        span = step + 1
        acceptance_limit = range_size - (range_size % span)
        blocks: list[str] = []
        while True:
            block = secrets.token_bytes(32)
            blocks.append(block.hex())
            value = int.from_bytes(block, byteorder="big", signed=False)
            if value < acceptance_limit:
                swap_index = value % span
                break
        values[step], values[swap_index] = values[swap_index], values[step]
        trace.append(tuple(blocks))
    return tuple(values), tuple(trace)


def _replay_iid_source_indices(
    *,
    support_sizes: tuple[int, ...],
    entropy_blocks_by_draw: tuple[tuple[str, ...], ...],
) -> tuple[int, ...]:
    if len(support_sizes) != len(entropy_blocks_by_draw):
        raise ValueError("IID entropy trace has the wrong number of draws")
    range_size = 1 << 256
    indices: list[int] = []
    for support_size, blocks in zip(
        support_sizes,
        entropy_blocks_by_draw,
        strict=True,
    ):
        if int(support_size) <= 0 or not blocks:
            raise ValueError("IID entropy trace has an empty support or draw")
        acceptance_limit = range_size - (range_size % int(support_size))
        attempts: list[int] = []
        for block in blocks:
            _require_hash(block, "IID entropy block")
            attempts.append(
                int.from_bytes(bytes.fromhex(block), byteorder="big", signed=False)
            )
        if any(value < acceptance_limit for value in attempts[:-1]):
            raise ValueError("IID entropy trace ignored an earlier accepted draw")
        accepted = attempts[-1]
        if accepted >= acceptance_limit:
            raise ValueError("IID entropy trace ends with a rejected draw")
        indices.append(int(accepted % int(support_size)))
    return tuple(indices)


def _draw_iid_source_indices(
    support_sizes: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[tuple[str, ...], ...]]:
    range_size = 1 << 256
    indices: list[int] = []
    traces: list[tuple[str, ...]] = []
    for support_size in support_sizes:
        size = int(support_size)
        if size <= 0:
            raise ValueError("IID draw requires nonempty support")
        acceptance_limit = range_size - (range_size % size)
        blocks: list[str] = []
        while True:
            block = secrets.token_bytes(32)
            blocks.append(block.hex())
            value = int.from_bytes(block, byteorder="big", signed=False)
            if value < acceptance_limit:
                indices.append(int(value % size))
                break
        traces.append(tuple(blocks))
    return tuple(indices), tuple(traces)


@dataclass(frozen=True, slots=True)
class SamplingPermutationReceipt:
    """Auditable entropy reveal and exact within-stratum permutation."""

    stratum_id: str
    canonical_population: tuple[str, ...]
    birth_binding: tuple[tuple[str, str], ...]
    entropy_blocks_by_step: tuple[tuple[str, ...], ...]
    generated_order: tuple[str, ...]
    entropy_source_id: str = GLOCAL_EXTERNAL_ENTROPY_SOURCE_V1
    algorithm_id: str = GLOCAL_SEALED_PERMUTATION_ALGORITHM_V1
    schema_name: str = "glocal_sampling_permutation_receipt_v1"

    def __post_init__(self) -> None:
        stratum_id = str(self.stratum_id)
        population = tuple(str(value) for value in self.canonical_population)
        birth_binding = tuple(
            (str(key), str(value)) for key, value in self.birth_binding
        )
        entropy_trace = tuple(
            tuple(str(block) for block in blocks)
            for blocks in self.entropy_blocks_by_step
        )
        order = tuple(str(value) for value in self.generated_order)
        if (
            not stratum_id
            or not population
            or population != tuple(sorted(population))
            or len(set(population)) != len(population)
        ):
            raise ValueError(
                "sampling permutation requires a canonical unique population"
            )
        if (
            not birth_binding
            or len({key for key, _ in birth_binding}) != len(birth_binding)
            or any(not key or not value for key, value in birth_binding)
        ):
            raise ValueError(
                "sampling permutation requires a unique nonempty birth binding"
            )
        source = str(self.entropy_source_id)
        if source not in {
            GLOCAL_SYSTEM_ENTROPY_SOURCE_V1,
            GLOCAL_EXTERNAL_ENTROPY_SOURCE_V1,
        }:
            raise ValueError("unsupported sampling entropy source")
        if (
            len(order) != len(population)
            or len(set(order)) != len(order)
            or set(order) != set(population)
        ):
            raise ValueError("sampling permutation order must enumerate its population")
        if self.algorithm_id != GLOCAL_SEALED_PERMUTATION_ALGORITHM_V1:
            raise ValueError("unsupported sampling permutation algorithm")
        object.__setattr__(self, "stratum_id", stratum_id)
        object.__setattr__(self, "canonical_population", population)
        object.__setattr__(self, "birth_binding", tuple(sorted(birth_binding)))
        object.__setattr__(self, "entropy_blocks_by_step", entropy_trace)
        object.__setattr__(self, "generated_order", order)
        object.__setattr__(self, "entropy_source_id", source)
        self.verify()

    @classmethod
    def create(
        cls,
        *,
        stratum_id: str,
        population: Sequence[str],
        birth_binding: Mapping[str, str],
        entropy_blocks_by_step: Sequence[Sequence[str]],
        entropy_source_id: str = GLOCAL_EXTERNAL_ENTROPY_SOURCE_V1,
    ) -> "SamplingPermutationReceipt":
        canonical = tuple(sorted(str(value) for value in population))
        trace = tuple(
            tuple(str(block) for block in blocks) for blocks in entropy_blocks_by_step
        )
        return cls(
            stratum_id=str(stratum_id),
            canonical_population=canonical,
            birth_binding=tuple(
                sorted((str(key), str(value)) for key, value in birth_binding.items())
            ),
            entropy_blocks_by_step=trace,
            generated_order=_replay_sampling_permutation(
                canonical_population=canonical,
                entropy_blocks_by_step=trace,
            ),
            entropy_source_id=str(entropy_source_id),
        )

    @classmethod
    def draw(
        cls,
        *,
        stratum_id: str,
        population: Sequence[str],
        birth_binding: Mapping[str, str],
    ) -> "SamplingPermutationReceipt":
        """Draw fresh system entropy after the bound decision object exists."""

        canonical = tuple(sorted(str(value) for value in population))
        order, trace = _draw_sampling_permutation(canonical)
        return cls(
            stratum_id=stratum_id,
            canonical_population=canonical,
            birth_binding=tuple(
                sorted((str(key), str(value)) for key, value in birth_binding.items())
            ),
            entropy_blocks_by_step=trace,
            generated_order=order,
            entropy_source_id=GLOCAL_SYSTEM_ENTROPY_SOURCE_V1,
        )

    def verify(self) -> None:
        expected = _replay_sampling_permutation(
            canonical_population=self.canonical_population,
            entropy_blocks_by_step=self.entropy_blocks_by_step,
        )
        if self.generated_order != expected:
            raise ValueError("sampling permutation does not match its entropy reveal")

    @property
    def content_hash(self) -> str:
        return content_hash(self)

    @property
    def birth_binding_hash(self) -> str:
        return content_hash({"birth_binding": self.birth_binding})


@dataclass(frozen=True, slots=True)
class IIDSupportDistribution:
    """Frozen conditional source-unit law for one terminal-audit stratum."""

    stratum_id: str
    source_unit_probabilities: tuple[tuple[str, float], ...]
    schema_name: str = "iid_support_distribution_v1"

    def __post_init__(self) -> None:
        stratum_id = str(self.stratum_id)
        rows = tuple(
            (str(source_unit_id), float(probability))
            for source_unit_id, probability in self.source_unit_probabilities
        )
        if (
            not stratum_id
            or not rows
            or len({source_unit_id for source_unit_id, _ in rows}) != len(rows)
            or any(not source_unit_id for source_unit_id, _ in rows)
            or any(
                not math.isfinite(probability) or probability <= 0.0
                for _, probability in rows
            )
        ):
            raise ValueError(
                "IID support distribution requires unique nonempty source units "
                "with positive finite probabilities"
            )
        if not math.isclose(
            math.fsum(probability for _, probability in rows),
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "IID conditional source probabilities must sum to one per stratum"
            )
        if self.schema_name != "iid_support_distribution_v1":
            raise ValueError("unsupported IID support-distribution schema")
        object.__setattr__(self, "stratum_id", stratum_id)
        object.__setattr__(self, "source_unit_probabilities", tuple(sorted(rows)))

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class IIDDrawRecord:
    """One realized IID draw, uniquely named but physically source-bound."""

    draw_id: str
    source_unit_id: str
    stratum_id: str
    conditional_draw_probability: float
    draw_position: int
    schema_name: str = "iid_draw_record_v1"

    def __post_init__(self) -> None:
        probability = float(self.conditional_draw_probability)
        position = int(self.draw_position)
        if (
            not str(self.draw_id)
            or not str(self.source_unit_id)
            or not str(self.stratum_id)
            or not math.isfinite(probability)
            or not 0.0 < probability <= 1.0
            or position < 0
        ):
            raise ValueError("IID draw record has invalid physical coordinates")
        if self.schema_name != "iid_draw_record_v1":
            raise ValueError("unsupported IID draw-record schema")
        object.__setattr__(self, "draw_id", str(self.draw_id))
        object.__setattr__(self, "source_unit_id", str(self.source_unit_id))
        object.__setattr__(self, "stratum_id", str(self.stratum_id))
        object.__setattr__(self, "conditional_draw_probability", probability)
        object.__setattr__(self, "draw_position", position)

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class IIDSamplingReceipt:
    """Physical pre-outcome contract for a fixed-n stratified IID audit.

    Statistical observations are globally unique ``draw_id`` values.  Each is
    mapped, in realized order, to the source/support unit actually evaluated.
    The complete conditional support law is retained so a claimed draw
    probability can be checked rather than merely asserted.  Repeated source
    units are therefore legal, while repeated draw IDs are not.
    """

    receipt_id: str
    context_population_hash: str
    support_distribution_hash: str
    sampling_algorithm_id: str
    sampling_code_hash: str
    provider_identity_hash: str
    executor_identity_hash: str
    support_distributions: tuple[IIDSupportDistribution, ...]
    draws: tuple[IIDDrawRecord, ...]
    birth_binding: tuple[tuple[str, str], ...] = ()
    entropy_blocks_by_draw: tuple[tuple[str, ...], ...] = ()
    entropy_source_id: str = GLOCAL_EXTERNAL_ENTROPY_SOURCE_V1
    frozen_before_outcomes: bool = True
    independent_draws: bool = True
    with_replacement: bool = True
    schema_name: str = "iid_sampling_receipt_v2"

    def __post_init__(self) -> None:
        if not self.receipt_id or not self.sampling_algorithm_id:
            raise ValueError("IID sampling receipt requires stable identity")
        for value, label in (
            (self.context_population_hash, "IID context population hash"),
            (self.support_distribution_hash, "IID support-distribution hash"),
            (self.sampling_code_hash, "IID sampling code hash"),
            (self.provider_identity_hash, "IID provider identity hash"),
            (self.executor_identity_hash, "IID executor identity hash"),
        ):
            _require_hash(value, label)
        support = tuple(self.support_distributions)
        draws = tuple(self.draws)
        birth_binding = tuple(
            sorted((str(key), str(value)) for key, value in self.birth_binding)
        )
        entropy_trace = tuple(
            tuple(str(block) for block in blocks)
            for blocks in self.entropy_blocks_by_draw
        )
        if (
            not support
            or len({item.stratum_id for item in support}) != len(support)
            or not draws
            or any(not isinstance(item, IIDSupportDistribution) for item in support)
            or any(not isinstance(item, IIDDrawRecord) for item in draws)
        ):
            raise ValueError(
                "IID sampling receipt requires typed support laws and realized draws"
            )
        support = tuple(sorted(support, key=lambda item: item.stratum_id))
        source_membership: dict[str, str] = {}
        probability_by_coordinate: dict[tuple[str, str], float] = {}
        for distribution in support:
            for source_unit_id, probability in distribution.source_unit_probabilities:
                previous = source_membership.setdefault(
                    source_unit_id,
                    distribution.stratum_id,
                )
                if previous != distribution.stratum_id:
                    raise ValueError(
                        "one IID source unit cannot belong to multiple strata"
                    )
                probability_by_coordinate[(distribution.stratum_id, source_unit_id)] = (
                    probability
                )
        expected_support_hash = self.support_hash_for(
            context_population_hash=self.context_population_hash,
            support_distributions=support,
        )
        if self.support_distribution_hash != expected_support_hash:
            raise ValueError(
                "IID support/distribution hash differs from its retained law"
            )
        if len({item.draw_id for item in draws}) != len(draws) or tuple(
            item.draw_position for item in draws
        ) != tuple(range(len(draws))):
            raise ValueError(
                "IID draws require globally unique IDs in contiguous realized order"
            )
        for draw in draws:
            expected_probability = probability_by_coordinate.get(
                (draw.stratum_id, draw.source_unit_id)
            )
            if expected_probability is None:
                raise ValueError(
                    "IID realized draw lies outside its frozen support stratum"
                )
            if not math.isclose(
                draw.conditional_draw_probability,
                expected_probability,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "IID realized draw probability differs from its frozen law"
                )
        if {item.stratum_id for item in draws} != {item.stratum_id for item in support}:
            raise ValueError("IID realized sample must cover every frozen stratum")
        if not (
            self.frozen_before_outcomes
            and self.independent_draws
            and self.with_replacement
        ):
            raise ValueError(
                "claim-bearing IID audit requires pre-outcome independent draws "
                "with replacement"
            )
        if self.schema_name not in {
            "iid_sampling_receipt_v2",
            "iid_sampling_receipt_v3",
        }:
            raise ValueError(
                "legacy or unsupported IID sampling receipts are not claim-bearing"
            )
        if self.schema_name == "iid_sampling_receipt_v3":
            if (
                self.sampling_algorithm_id != GLOCAL_IID_RUNTIME_RANDOMIZATION_METHOD_V2
                or self.entropy_source_id != GLOCAL_SYSTEM_ENTROPY_SOURCE_V1
                or not birth_binding
                or len({key for key, _ in birth_binding}) != len(birth_binding)
            ):
                raise ValueError(
                    "runtime IID receipt lacks its system-entropy birth binding"
                )
            support_by_stratum = {
                item.stratum_id: tuple(
                    source_unit_id
                    for source_unit_id, _ in item.source_unit_probabilities
                )
                for item in support
            }
            support_sizes = tuple(
                len(support_by_stratum[draw.stratum_id]) for draw in draws
            )
            replayed = _replay_iid_source_indices(
                support_sizes=support_sizes,
                entropy_blocks_by_draw=entropy_trace,
            )
            replayed_sources = tuple(
                support_by_stratum[draw.stratum_id][index]
                for draw, index in zip(draws, replayed, strict=True)
            )
            if replayed_sources != tuple(draw.source_unit_id for draw in draws):
                raise ValueError(
                    "runtime IID draws differ from their retained entropy trace"
                )
        elif birth_binding or entropy_trace:
            raise ValueError(
                "replay-only IID v2 receipts cannot assert runtime entropy issuance"
            )
        object.__setattr__(self, "support_distributions", support)
        object.__setattr__(self, "draws", draws)
        object.__setattr__(self, "birth_binding", birth_binding)
        object.__setattr__(self, "entropy_blocks_by_draw", entropy_trace)

    @classmethod
    def draw_uniform(
        cls,
        *,
        receipt_id: str,
        context_population_hash: str,
        support_context_ids_by_stratum: Mapping[str, Sequence[str]],
        target_sample_sizes_by_stratum: Mapping[str, int],
        provider_identity_hash: str,
        executor_identity_hash: str,
        birth_binding: Mapping[str, str],
    ) -> "IIDSamplingReceipt":
        """Draw runtime-owned IID rows with an exactly replayable entropy trace."""

        support_axis = {
            str(stratum_id): tuple(sorted(str(source_id) for source_id in source_ids))
            for stratum_id, source_ids in support_context_ids_by_stratum.items()
        }
        targets = {
            str(stratum_id): int(target)
            for stratum_id, target in target_sample_sizes_by_stratum.items()
        }
        if set(support_axis) != set(targets) or any(
            not source_ids or targets[stratum_id] <= 0
            for stratum_id, source_ids in support_axis.items()
        ):
            raise ValueError("runtime IID support and target axes must agree")
        support = tuple(
            IIDSupportDistribution(
                stratum_id=stratum_id,
                source_unit_probabilities=tuple(
                    (source_id, 1.0 / len(source_ids)) for source_id in source_ids
                ),
            )
            for stratum_id, source_ids in sorted(support_axis.items())
        )
        draw_strata = tuple(
            stratum_id
            for stratum_id in sorted(targets)
            for _ in range(targets[stratum_id])
        )
        support_sizes = tuple(
            len(support_axis[stratum_id]) for stratum_id in draw_strata
        )
        source_indices, entropy_trace = _draw_iid_source_indices(support_sizes)
        draws = tuple(
            IIDDrawRecord(
                draw_id=(
                    f"{receipt_id}:draw:{position}:"
                    f"{content_hash((stratum_id, position, entropy_trace[position]))[:16]}"
                ),
                source_unit_id=support_axis[stratum_id][source_index],
                stratum_id=stratum_id,
                conditional_draw_probability=(1.0 / len(support_axis[stratum_id])),
                draw_position=position,
            )
            for position, (stratum_id, source_index) in enumerate(
                zip(draw_strata, source_indices, strict=True)
            )
        )
        return cls(
            receipt_id=str(receipt_id),
            context_population_hash=str(context_population_hash),
            support_distribution_hash=cls.support_hash_for(
                context_population_hash=context_population_hash,
                support_distributions=support,
            ),
            sampling_algorithm_id=GLOCAL_IID_RUNTIME_RANDOMIZATION_METHOD_V2,
            sampling_code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            provider_identity_hash=str(provider_identity_hash),
            executor_identity_hash=str(executor_identity_hash),
            support_distributions=support,
            draws=draws,
            birth_binding=tuple(
                sorted((str(key), str(value)) for key, value in birth_binding.items())
            ),
            entropy_blocks_by_draw=entropy_trace,
            entropy_source_id=GLOCAL_SYSTEM_ENTROPY_SOURCE_V1,
            schema_name="iid_sampling_receipt_v3",
        )

    @property
    def runtime_generated(self) -> bool:
        return (
            self.schema_name == "iid_sampling_receipt_v3"
            and self.sampling_algorithm_id == GLOCAL_IID_RUNTIME_RANDOMIZATION_METHOD_V2
            and self.entropy_source_id == GLOCAL_SYSTEM_ENTROPY_SOURCE_V1
        )

    @staticmethod
    def support_hash_for(
        *,
        context_population_hash: str,
        support_distributions: Sequence[IIDSupportDistribution],
    ) -> str:
        _require_hash(context_population_hash, "IID context population hash")
        support = tuple(
            sorted(
                tuple(support_distributions),
                key=lambda item: item.stratum_id,
            )
        )
        if not support or any(
            not isinstance(item, IIDSupportDistribution) for item in support
        ):
            raise ValueError("IID support hash requires typed support distributions")
        return content_hash(
            {
                "schema_name": "iid_frozen_support_distribution_v1",
                "context_population_hash": str(context_population_hash),
                "support_distributions": support,
            }
        )

    @property
    def draw_ids(self) -> tuple[str, ...]:
        return tuple(item.draw_id for item in self.draws)

    @property
    def source_unit_ids(self) -> tuple[str, ...]:
        return tuple(item.source_unit_id for item in self.draws)

    @property
    def stratum_ids(self) -> tuple[str, ...]:
        return tuple(item.stratum_id for item in self.draws)

    @property
    def unique_source_unit_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.source_unit_ids))

    @property
    def support_source_unit_ids(self) -> tuple[str, ...]:
        return tuple(
            source_unit_id
            for distribution in self.support_distributions
            for source_unit_id, _ in distribution.source_unit_probabilities
        )

    @property
    def content_hash(self) -> str:
        return content_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "receipt_id": self.receipt_id,
            "context_population_hash": self.context_population_hash,
            "support_distribution_hash": self.support_distribution_hash,
            "sampling_algorithm_id": self.sampling_algorithm_id,
            "sampling_code_hash": self.sampling_code_hash,
            "provider_identity_hash": self.provider_identity_hash,
            "executor_identity_hash": self.executor_identity_hash,
            "support_distributions": [
                {
                    "schema_name": item.schema_name,
                    "stratum_id": item.stratum_id,
                    "source_unit_probabilities": [
                        [source_unit_id, probability]
                        for source_unit_id, probability in (
                            item.source_unit_probabilities
                        )
                    ],
                }
                for item in self.support_distributions
            ],
            "draws": [
                {
                    "schema_name": item.schema_name,
                    "draw_id": item.draw_id,
                    "source_unit_id": item.source_unit_id,
                    "stratum_id": item.stratum_id,
                    "conditional_draw_probability": (item.conditional_draw_probability),
                    "draw_position": item.draw_position,
                }
                for item in self.draws
            ],
            "birth_binding": [list(item) for item in self.birth_binding],
            "entropy_blocks_by_draw": [
                list(item) for item in self.entropy_blocks_by_draw
            ],
            "entropy_source_id": self.entropy_source_id,
            "frozen_before_outcomes": self.frozen_before_outcomes,
            "independent_draws": self.independent_draws,
            "with_replacement": self.with_replacement,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "IIDSamplingReceipt":
        if str(payload.get("schema_name", "")) != "iid_sampling_receipt_v2":
            raise ValueError(
                "legacy or unsupported IID sampling receipts are not claim-bearing"
            )
        receipt = cls(
            receipt_id=str(payload.get("receipt_id", "")),
            context_population_hash=str(payload.get("context_population_hash", "")),
            support_distribution_hash=str(payload.get("support_distribution_hash", "")),
            sampling_algorithm_id=str(payload.get("sampling_algorithm_id", "")),
            sampling_code_hash=str(payload.get("sampling_code_hash", "")),
            provider_identity_hash=str(payload.get("provider_identity_hash", "")),
            executor_identity_hash=str(payload.get("executor_identity_hash", "")),
            support_distributions=tuple(
                IIDSupportDistribution(
                    stratum_id=str(item.get("stratum_id", "")),
                    source_unit_probabilities=tuple(
                        (str(source_unit_id), float(probability))
                        for source_unit_id, probability in item.get(
                            "source_unit_probabilities",
                            (),
                        )
                    ),
                    schema_name=str(item.get("schema_name", "")),
                )
                for item in payload.get("support_distributions", ())
            ),
            draws=tuple(
                IIDDrawRecord(
                    draw_id=str(item.get("draw_id", "")),
                    source_unit_id=str(item.get("source_unit_id", "")),
                    stratum_id=str(item.get("stratum_id", "")),
                    conditional_draw_probability=float(
                        item.get("conditional_draw_probability", float("nan"))
                    ),
                    draw_position=int(item.get("draw_position", -1)),
                    schema_name=str(item.get("schema_name", "")),
                )
                for item in payload.get("draws", ())
            ),
            frozen_before_outcomes=bool(payload.get("frozen_before_outcomes", False)),
            independent_draws=bool(payload.get("independent_draws", False)),
            with_replacement=bool(payload.get("with_replacement", False)),
            schema_name=str(payload.get("schema_name", "")),
        )
        expected = str(payload.get("content_hash", ""))
        if not expected or expected != receipt.content_hash:
            raise ValueError("IID sampling receipt content hash mismatch")
        return receipt


def iid_physical_provider_identity_hash(provider: Any) -> str:
    """Derive the provider identity checked at physical IID execution.

    A plain object type is not an adequate identity: the provider must expose
    a stable ``source_id`` and the exact source ``context_axis`` (or ``axis``)
    whose support units it resolves.
    """

    source_id = str(getattr(provider, "source_id", ""))
    axis = getattr(provider, "context_axis", None)
    if axis is None:
        axis = getattr(provider, "axis", None)
    axis_hash = str(getattr(axis, "content_hash", ""))
    if not source_id or not axis_hash:
        raise ValueError(
            "IID physical execution requires provider source_id and source axis identity"
        )
    _require_hash(axis_hash, "IID provider source-axis hash")
    provider_type = type(provider)
    provider_source = inspect.getsourcefile(provider_type)
    if provider_source is None:
        raise ValueError(
            "IID physical provider requires inspectable executable source code"
        )
    provider_code_hash = hashlib.sha256(Path(provider_source).read_bytes()).hexdigest()
    return content_hash(
        {
            "schema_name": "iid_physical_provider_identity_v2",
            "source_id": source_id,
            "source_axis_hash": axis_hash,
            "provider_type": (
                f"{provider_type.__module__}.{provider_type.__qualname__}"
            ),
            "provider_code_hash": provider_code_hash,
        }
    )


def _physical_tensor_state_hash(model: Any) -> str:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - physical runtime dependency.
        raise ValueError(
            "physical model-state identity requires the Torch runtime"
        ) from error
    state = model.state_dict() if hasattr(model, "state_dict") else None
    if not isinstance(state, Mapping) or not state:
        raise ValueError("IID physical executor model requires a nonempty tensor state")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise ValueError(f"IID physical model state entry {name!r} is not a tensor")
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(int(item) for item in value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _physical_evaluator_model_state_hash(evaluator: Any) -> str:
    declared = str(getattr(evaluator, "model_state_hash", ""))
    source_model = getattr(evaluator, "source_model", None)
    if source_model is not None:
        computed = _physical_tensor_state_hash(source_model)
        if declared and declared != computed:
            raise ValueError(
                "IID evaluator declared model state differs from executable weights"
            )
        return computed
    if not _is_sha256(declared):
        raise ValueError(
            "IID physical evaluator without an inspectable Torch model must "
            "declare a SHA-256 model_state_hash"
        )
    return declared


def iid_physical_executor_identity_hash(executor: Any) -> str:
    """Derive the backend/evaluator identity checked at physical IID execution."""

    evaluator = getattr(executor, "evaluator", None)
    if evaluator is None:
        raise ValueError(
            "IID physical execution requires an executor with an evaluator identity"
        )
    executor_type = type(executor)
    evaluator_type = type(evaluator)
    device = str(getattr(evaluator, "device", ""))
    if not device:
        raise ValueError("IID physical executor is missing its device identity")
    executor_source = inspect.getsourcefile(executor_type)
    evaluator_source = inspect.getsourcefile(evaluator_type)
    if executor_source is None or evaluator_source is None:
        raise ValueError(
            "IID physical executor/evaluator requires inspectable source code"
        )
    query_by_id = getattr(executor, "_query_by_id", {})
    if not isinstance(query_by_id, Mapping) or not query_by_id:
        raise ValueError("IID physical executor requires frozen query identities")
    query_hashes = tuple(
        sorted(
            (
                str(query_id),
                str(getattr(query, "content_hash", "")),
            )
            for query_id, query in query_by_id.items()
        )
    )
    if any(not _is_sha256(query_hash) for _, query_hash in query_hashes):
        raise ValueError("IID physical executor query identity is incomplete")
    evaluator_contract = {
        key: str(getattr(evaluator, key))
        for key in (
            "evaluator_id",
            "topology_hash",
            "route_schema_hash",
            "parity_receipt_hash",
        )
        if getattr(evaluator, key, None) is not None
    }
    return content_hash(
        {
            "schema_name": "iid_physical_executor_identity_v2",
            "executor_type": (
                f"{executor_type.__module__}.{executor_type.__qualname__}"
            ),
            "evaluator_type": (
                f"{evaluator_type.__module__}.{evaluator_type.__qualname__}"
            ),
            "executor_code_hash": hashlib.sha256(
                Path(executor_source).read_bytes()
            ).hexdigest(),
            "evaluator_code_hash": hashlib.sha256(
                Path(evaluator_source).read_bytes()
            ).hexdigest(),
            "device": device,
            "model_state_hash": _physical_evaluator_model_state_hash(evaluator),
            "query_hashes": query_hashes,
            "calibration_options_hash": content_hash(
                dict(getattr(executor, "calibration_options", {}))
            ),
            "runtime_execution_profile_hash": content_hash(
                getattr(executor, "runtime_execution_profile", {})
            ),
            "evaluator_contract": evaluator_contract,
        }
    )


@dataclass(frozen=True, slots=True)
class SamplingStratumSpec:
    stratum_id: str
    population_size: int
    risk_weight: float
    sealed_order_hash: str
    context_ids_in_sealed_order: tuple[str, ...] = ()
    claim_bearing: bool = True

    def __post_init__(self) -> None:
        if not self.stratum_id:
            raise ValueError("sampling stratum requires stratum_id")
        size = int(self.population_size)
        if size <= 0:
            raise ValueError("sampling stratum population_size must be positive")
        weight = float(self.risk_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("sampling stratum risk_weight must be finite and positive")
        _require_hash(self.sealed_order_hash, "sealed order hash")
        contexts = tuple(str(value) for value in self.context_ids_in_sealed_order)
        if self.claim_bearing and not contexts:
            raise ValueError(
                "claim-bearing sampling strata must enumerate every context in sealed order"
            )
        if contexts:
            if len(contexts) != size or len(set(contexts)) != size:
                raise ValueError(
                    "sealed context order must contain each population unit once"
                )
            expected = content_hash(
                {"stratum_id": self.stratum_id, "context_ids": contexts}
            )
            if self.sealed_order_hash != expected:
                raise ValueError(
                    "sealed context order differs from its commitment hash"
                )
        object.__setattr__(self, "population_size", size)
        object.__setattr__(self, "risk_weight", weight)
        object.__setattr__(self, "context_ids_in_sealed_order", contexts)

    @classmethod
    def from_context_ids(
        cls,
        *,
        stratum_id: str,
        context_ids_in_sealed_order: Sequence[str],
        risk_weight: float,
    ) -> "SamplingStratumSpec":
        contexts = tuple(str(value) for value in context_ids_in_sealed_order)
        return cls(
            stratum_id=stratum_id,
            population_size=len(contexts),
            risk_weight=risk_weight,
            sealed_order_hash=content_hash(
                {"stratum_id": str(stratum_id), "context_ids": contexts}
            ),
            context_ids_in_sealed_order=contexts,
            claim_bearing=True,
        )

    @classmethod
    def exploratory_opaque(
        cls,
        *,
        stratum_id: str,
        population_size: int,
        risk_weight: float,
        sealed_order_hash: str,
    ) -> "SamplingStratumSpec":
        """Explicit nonclaim compatibility for legacy opaque finite panels."""

        return cls(
            stratum_id=stratum_id,
            population_size=population_size,
            risk_weight=risk_weight,
            sealed_order_hash=sealed_order_hash,
            context_ids_in_sealed_order=(),
            claim_bearing=False,
        )


@dataclass(frozen=True, slots=True)
class SamplingDesignSpec:
    """Predictable stratified WoR design or fixed-n terminal IID design."""

    design_id: str
    context_population_hash: str
    randomization_commitment_hash: str
    strata: tuple[SamplingStratumSpec, ...]
    permutation_receipts: tuple[SamplingPermutationReceipt, ...] = ()
    iid_sampling_receipt: IIDSamplingReceipt | None = None
    binary_class_by_stratum: tuple[tuple[str, int], ...] = ()
    risk_component_by_stratum: tuple[tuple[str, str, str], ...] = ()
    sampling_unit: str = "context"
    design_kind: str = "sealed_uniform_without_replacement_stratified"
    randomization_method: str = GLOCAL_SEALED_PERMUTATION_ALGORITHM_V1
    claim_bearing: bool = True
    schema_name: str = "sampling_design_v1"

    def __post_init__(self) -> None:
        if not self.design_id:
            raise ValueError("sampling design requires design_id")
        _require_hash(self.context_population_hash, "context population hash")
        _require_hash(
            self.randomization_commitment_hash,
            "sampling randomization commitment hash",
        )
        if self.design_kind not in {
            "sealed_uniform_without_replacement_stratified",
            "iid_fixed_n_stratified",
        }:
            raise ValueError("unsupported claim-bearing sampling design")
        if self.design_kind == "sealed_uniform_without_replacement_stratified":
            if self.sampling_unit != "context":
                raise ValueError(
                    "finite-population glocal sampling requires context units"
                )
            if self.randomization_method != GLOCAL_SEALED_PERMUTATION_ALGORITHM_V1:
                raise ValueError("unsupported claim-bearing randomization method")
        else:
            if self.sampling_unit != "draw":
                raise ValueError(
                    "fixed-n IID designs require unique statistical draw units"
                )
            if self.randomization_method not in {
                GLOCAL_IID_FIXED_N_RANDOMIZATION_METHOD_V1,
                GLOCAL_IID_RUNTIME_RANDOMIZATION_METHOD_V2,
            }:
                raise ValueError("unsupported fixed-n IID randomization method")
        strata = tuple(self.strata)
        if not strata or len({item.stratum_id for item in strata}) != len(strata):
            raise ValueError("sampling design requires unique nonempty strata")
        if self.claim_bearing and any(not item.claim_bearing for item in strata):
            raise ValueError(
                "claim-bearing design cannot contain an opaque nonclaim stratum"
            )
        total_risk_weight = math.fsum(item.risk_weight for item in strata)
        if not math.isclose(total_risk_weight, 1.0, abs_tol=1e-12):
            raise ValueError("sampling stratum risk weights must sum to one")
        strata = tuple(
            replace(item, risk_weight=item.risk_weight / total_risk_weight)
            for item in strata
        )
        has_explicit_order = [bool(item.context_ids_in_sealed_order) for item in strata]
        if self.claim_bearing and not all(has_explicit_order):
            raise ValueError(
                "claim-bearing sampling design requires every sealed context order"
            )
        if any(has_explicit_order) and not all(has_explicit_order):
            raise ValueError(
                "sampling design must seal every within-stratum order or none"
            )
        if all(has_explicit_order):
            context_ids = tuple(
                context_id
                for item in strata
                for context_id in item.context_ids_in_sealed_order
            )
            if len(context_ids) != len(set(context_ids)):
                raise ValueError("sampling design context IDs must be globally unique")
        receipts = tuple(self.permutation_receipts)
        iid_receipt = self.iid_sampling_receipt
        if (
            self.design_kind == "sealed_uniform_without_replacement_stratified"
            and iid_receipt is not None
        ):
            raise ValueError("finite-population design cannot carry an IID receipt")
        if self.design_kind == "iid_fixed_n_stratified" and receipts:
            raise ValueError("fixed-n IID design cannot carry permutation receipts")
        if (
            self.claim_bearing
            and self.design_kind == "sealed_uniform_without_replacement_stratified"
        ):
            if len(receipts) != len(strata) or len(
                {item.stratum_id for item in receipts}
            ) != len(receipts):
                raise ValueError(
                    "claim-bearing sampling design requires one typed permutation "
                    "receipt per stratum"
                )
            if any(
                item.entropy_source_id != GLOCAL_SYSTEM_ENTROPY_SOURCE_V1
                for item in receipts
            ):
                raise ValueError(
                    "claim-bearing sampling requires fresh system entropy; "
                    "externally supplied traces are replay-only"
                )
            receipt_by_stratum = {item.stratum_id: item for item in receipts}
            if set(receipt_by_stratum) != {item.stratum_id for item in strata}:
                raise ValueError(
                    "sampling permutation receipt axis differs from the design"
                )
            for stratum in strata:
                receipt = receipt_by_stratum[stratum.stratum_id]
                receipt.verify()
                if receipt.generated_order != stratum.context_ids_in_sealed_order:
                    raise ValueError(
                        "sampling stratum order differs from its permutation receipt"
                    )
            if len({item.birth_binding_hash for item in receipts}) != 1:
                raise ValueError(
                    "sampling permutation receipts do not share one frozen birth "
                    "binding"
                )
            expected_commitment = self.randomization_commitment_for(receipts)
            if self.randomization_commitment_hash != expected_commitment:
                raise ValueError(
                    "sampling randomization commitment differs from its receipts"
                )
        elif self.claim_bearing:
            if not isinstance(iid_receipt, IIDSamplingReceipt):
                raise ValueError(
                    "claim-bearing fixed-n IID design requires a typed draw receipt"
                )
            if (
                self.randomization_method != GLOCAL_IID_RUNTIME_RANDOMIZATION_METHOD_V2
                or not iid_receipt.runtime_generated
            ):
                raise ValueError(
                    "claim-bearing fixed-n IID design requires runtime-issued "
                    "system-entropy draws; v2/manual receipts are replay-only"
                )
            context_ids = tuple(
                context_id
                for item in strata
                for context_id in item.context_ids_in_sealed_order
            )
            stratum_ids = tuple(
                item.stratum_id
                for item in strata
                for _ in item.context_ids_in_sealed_order
            )
            receipt_membership = dict(
                zip(
                    iid_receipt.draw_ids,
                    iid_receipt.stratum_ids,
                    strict=True,
                )
            )
            design_membership = dict(zip(context_ids, stratum_ids, strict=True))
            if receipt_membership != design_membership:
                raise ValueError(
                    "fixed-n IID design differs from its realized sample receipt"
                )
            if iid_receipt.context_population_hash != self.context_population_hash:
                raise ValueError(
                    "fixed-n IID receipt and design population identities differ"
                )
            if self.randomization_commitment_hash != iid_receipt.content_hash:
                raise ValueError(
                    "fixed-n IID randomization commitment differs from its receipt"
                )
        elif receipts or iid_receipt is not None:
            raise ValueError(
                "opaque nonclaim sampling designs cannot carry claim receipts"
            )
        binary_mapping = tuple(
            (str(stratum_id), int(class_id))
            for stratum_id, class_id in self.binary_class_by_stratum
        )
        if not binary_mapping:
            canonical_rows: list[tuple[str, int]] = []
            for item in strata:
                prefix, separator, suffix = item.stratum_id.partition(":")
                if prefix != "class" or separator != ":":
                    canonical_rows = []
                    break
                try:
                    canonical_rows.append((item.stratum_id, int(suffix)))
                except ValueError:
                    canonical_rows = []
                    break
            binary_mapping = tuple(canonical_rows)
        if binary_mapping:
            if len({stratum_id for stratum_id, _ in binary_mapping}) != len(
                binary_mapping
            ):
                raise ValueError(
                    "sampling design contains duplicate stratum-to-class mapping"
                )
            if {stratum_id for stratum_id, _ in binary_mapping} != {
                item.stratum_id for item in strata
            }:
                raise ValueError(
                    "sampling design binary mapping must cover every stratum exactly"
                )
        risk_components = tuple(
            (
                str(sampling_stratum_id),
                str(query_id),
                str(risk_stratum_id),
            )
            for sampling_stratum_id, query_id, risk_stratum_id in (
                self.risk_component_by_stratum
            )
        )
        if risk_components:
            if (
                len(
                    {
                        sampling_stratum_id
                        for sampling_stratum_id, _, _ in risk_components
                    }
                )
                != len(risk_components)
                or {
                    sampling_stratum_id for sampling_stratum_id, _, _ in risk_components
                }
                != {item.stratum_id for item in strata}
                or any(
                    not query_id or not risk_stratum_id
                    for _, query_id, risk_stratum_id in risk_components
                )
            ):
                raise ValueError(
                    "sampling risk-component mapping must cover every stratum "
                    "exactly once"
                )
            if len(
                {
                    (query_id, risk_stratum_id)
                    for _, query_id, risk_stratum_id in risk_components
                }
            ) != len(risk_components):
                raise ValueError(
                    "sampling design repeats a query-by-risk-stratum component"
                )
        object.__setattr__(
            self, "strata", tuple(sorted(strata, key=lambda item: item.stratum_id))
        )
        object.__setattr__(
            self,
            "permutation_receipts",
            tuple(sorted(receipts, key=lambda item: item.stratum_id)),
        )
        object.__setattr__(
            self,
            "binary_class_by_stratum",
            tuple(sorted(binary_mapping)),
        )
        object.__setattr__(
            self,
            "risk_component_by_stratum",
            tuple(sorted(risk_components)),
        )

    @property
    def content_hash(self) -> str:
        return content_hash(self)

    @staticmethod
    def randomization_commitment_for(
        receipts: Sequence[SamplingPermutationReceipt],
    ) -> str:
        rows = tuple(
            sorted(
                (
                    str(receipt.stratum_id),
                    str(receipt.content_hash),
                )
                for receipt in receipts
            )
        )
        if not rows or len({stratum_id for stratum_id, _ in rows}) != len(rows):
            raise ValueError(
                "randomization commitment requires unique permutation receipts"
            )
        return content_hash(
            {
                "randomization_method": (GLOCAL_SEALED_PERMUTATION_ALGORITHM_V1),
                "permutation_receipts": rows,
            }
        )


class EvidenceRole(str, Enum):
    GUIDANCE_GEOMETRY = "guidance_geometry"
    CALIBRATION_FIT = "calibration_fit"
    PROPOSAL_ADJUDICATION = "proposal_adjudication"
    TERMINAL_AUDIT = "terminal_audit"


@dataclass(frozen=True, slots=True)
class ObservationSetRef:
    """Exact context membership for one statistical role."""

    observation_set_id: str
    role: EvidenceRole
    sampling_design_hash: str
    context_ids: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    schema_name: str = "observation_set_ref_v1"

    def __post_init__(self) -> None:
        if not self.observation_set_id:
            raise ValueError("observation set requires observation_set_id")
        role = (
            self.role
            if isinstance(self.role, EvidenceRole)
            else EvidenceRole(self.role)
        )
        _require_hash(self.sampling_design_hash, "sampling design hash")
        contexts = tuple(str(value) for value in self.context_ids)
        strata = tuple(str(value) for value in self.stratum_ids)
        if not contexts or len(contexts) != len(strata):
            raise ValueError(
                "observation set requires aligned context and stratum axes"
            )
        if len(set(contexts)) != len(contexts):
            raise ValueError("observation set context IDs must be unique")
        if any(not value for value in contexts + strata):
            raise ValueError("observation set identities must be nonempty")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "stratum_ids", strata)

    @property
    def content_hash(self) -> str:
        return content_hash(self)

    def overlap(self, other: "ObservationSetRef") -> frozenset[str]:
        return frozenset(self.context_ids).intersection(other.context_ids)

    def validate_against(self, sampling_design: SamplingDesignSpec) -> None:
        if self.sampling_design_hash != sampling_design.content_hash:
            raise ValueError("observation set differs from its sampling design")
        sealed_membership: dict[str, str] = {}
        for stratum in sampling_design.strata:
            for context_id in stratum.context_ids_in_sealed_order:
                sealed_membership[context_id] = stratum.stratum_id
        known_strata = {item.stratum_id for item in sampling_design.strata}
        for context_id, stratum_id in zip(
            self.context_ids, self.stratum_ids, strict=True
        ):
            if stratum_id not in known_strata:
                raise ValueError("observation set contains an unknown stratum")
            if sealed_membership and sealed_membership.get(context_id) != stratum_id:
                raise ValueError(
                    "observation context/stratum membership differs from the sealed design"
                )


def validate_role_separation(observation_sets: Sequence[ObservationSetRef]) -> None:
    """Enforce the v3 calibration/adjudication and terminal isolation rules."""

    sets = tuple(observation_sets)
    if len({item.observation_set_id for item in sets}) != len(sets):
        raise ValueError("observation set IDs must be unique")
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            overlap = left.overlap(right)
            if not overlap:
                continue
            roles = {left.role, right.role}
            if EvidenceRole.TERMINAL_AUDIT in roles:
                raise ValueError(
                    "terminal audit contexts overlap an earlier statistical role: "
                    + ",".join(sorted(overlap))
                )
            if (
                EvidenceRole.PROPOSAL_ADJUDICATION in roles
                and left.role is not right.role
            ):
                raise ValueError(
                    "proposal-adjudication contexts overlap a different "
                    "statistical role: " + ",".join(sorted(overlap))
                )


@dataclass(frozen=True, slots=True)
class CalibrationAlgorithmSpec:
    algorithm_id: str
    algorithm_version: str
    code_hash: str
    objective: str = "bounded_brier"
    maximum_iterations: int = 1_000
    convergence_tolerance: float = 1e-8
    supports_warm_start: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)
    schema_name: str = "calibration_algorithm_v1"

    def __post_init__(self) -> None:
        if not self.algorithm_id or not self.algorithm_version:
            raise ValueError(
                "calibration algorithm requires stable identity and version"
            )
        _require_hash(self.code_hash, "calibration code hash")
        if self.objective != "bounded_brier":
            raise ValueError("claim-bearing calibration must target bounded Brier risk")
        if int(self.maximum_iterations) <= 0:
            raise ValueError("calibration maximum_iterations must be positive")
        tolerance = float(self.convergence_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("calibration convergence_tolerance must be positive")
        canonical_json(self.options)
        object.__setattr__(self, "maximum_iterations", int(self.maximum_iterations))
        object.__setattr__(self, "convergence_tolerance", tolerance)
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class CalibrationFitState:
    """Content-addressed fitted calibration state and its exact lineage."""

    fit_id: str
    algorithm_spec_hash: str
    fit_observation_set_hash: str
    output_hash: str
    parameters: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    converged: bool
    iterations: int
    warm_start_parent_hash: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_name: str = "calibration_fit_state_v1"

    def __post_init__(self) -> None:
        if not self.fit_id:
            raise ValueError("calibration fit requires fit_id")
        _require_hash(self.algorithm_spec_hash, "calibration algorithm spec hash")
        _require_hash(self.fit_observation_set_hash, "calibration observation set hash")
        _require_hash(self.output_hash, "calibration output hash")
        if self.warm_start_parent_hash is not None:
            _require_hash(
                self.warm_start_parent_hash, "calibration warm-start parent hash"
            )
        parameters = tuple(float(value) for value in self.parameters)
        if not parameters or any(not math.isfinite(value) for value in parameters):
            raise ValueError("calibration parameters must be finite and nonempty")
        covariance = np.asarray(self.covariance, dtype=np.float64)
        dimension = len(parameters)
        if (
            covariance.shape != (dimension, dimension)
            or not np.isfinite(covariance).all()
        ):
            raise ValueError("calibration covariance must match the parameter axis")
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        scale = (
            max(1.0, float(np.max(np.abs(eigenvalues)))) if eigenvalues.size else 1.0
        )
        if eigenvalues.size and float(np.min(eigenvalues)) < -1e-10 * scale:
            raise ValueError("calibration covariance must be positive semidefinite")
        if int(self.iterations) < 0:
            raise ValueError("calibration iteration count cannot be negative")
        canonical_json(self.diagnostics)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(
            self,
            "covariance",
            tuple(tuple(float(value) for value in row) for row in covariance),
        )
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    def require_claim_bearing(self) -> None:
        if not self.converged:
            raise ValueError("nonconverged calibration fit cannot carry a v3 claim")
        if self.iterations <= 0:
            raise ValueError("claim-bearing calibration fit lacks executed iterations")

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class PredictableProposalRegistration:
    """Everything frozen at candidate birth, before adjudication outcomes."""

    proposal_id: str
    query_id: str
    group_id: str
    registration_index: int
    parent_state_hash: str
    candidate_state_hash: str
    parent_calibration_hash: str
    candidate_calibration_hash: str
    risk_estimand_hash: str
    sampling_design_hash: str
    adjudication_observation_set_hash: str
    local_upper_boundary: float
    alpha_reserved: float
    stratum_alpha: tuple[tuple[str, float], ...]
    allocation_rule_id: str
    allocation_rule_hash: str
    candidate_definition_hash: str
    schema_name: str = "predictable_proposal_registration_v1"

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.query_id or not self.group_id:
            raise ValueError(
                "proposal registration requires proposal/query/group identity"
            )
        if int(self.registration_index) <= 0:
            raise ValueError("proposal registration index must be positive")
        for value, label in (
            (self.parent_state_hash, "parent state hash"),
            (self.candidate_state_hash, "candidate state hash"),
            (self.parent_calibration_hash, "parent calibration hash"),
            (self.candidate_calibration_hash, "candidate calibration hash"),
            (self.risk_estimand_hash, "risk estimand hash"),
            (self.sampling_design_hash, "sampling design hash"),
            (
                self.adjudication_observation_set_hash,
                "adjudication observation set hash",
            ),
            (self.allocation_rule_hash, "allocation rule hash"),
            (self.candidate_definition_hash, "candidate definition hash"),
        ):
            _require_hash(value, label)
        if self.parent_state_hash == self.candidate_state_hash:
            raise ValueError("proposal candidate state must differ from its parent")
        boundary = float(self.local_upper_boundary)
        alpha = float(self.alpha_reserved)
        if not math.isfinite(boundary):
            raise ValueError("proposal local boundary must be finite")
        if not 0.0 < alpha < 1.0:
            raise ValueError("proposal alpha reservation must lie in (0,1)")
        stratum_alpha = tuple(
            (str(stratum), float(value)) for stratum, value in self.stratum_alpha
        )
        if not stratum_alpha or len({item[0] for item in stratum_alpha}) != len(
            stratum_alpha
        ):
            raise ValueError("proposal stratum alpha axis must be unique and nonempty")
        if any(not 0.0 < value < 1.0 for _, value in stratum_alpha):
            raise ValueError("proposal stratum alpha values must lie in (0,1)")
        if sum(value for _, value in stratum_alpha) > alpha + 1e-15:
            raise ValueError("stratum alpha allocation exceeds proposal reservation")
        if not self.allocation_rule_id:
            raise ValueError(
                "proposal registration requires a predictable allocation rule"
            )
        object.__setattr__(self, "registration_index", int(self.registration_index))
        object.__setattr__(self, "local_upper_boundary", boundary)
        object.__setattr__(self, "alpha_reserved", alpha)
        object.__setattr__(self, "stratum_alpha", tuple(sorted(stratum_alpha)))

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class PredictableAllocationLookReceipt:
    """One pre-outcome allocation commitment for a proposal evidence look."""

    proposal_id: str
    proposal_registration_hash: str
    look_index: int
    sampling_design_hash: str
    allocation_rule_hash: str
    previous_allocation_receipt_hash: str | None
    previous_context_prefix_hash: str
    cumulative_targets: tuple[tuple[str, int], ...]
    selected_stratum: str
    planned_panel_hash: str
    planned_new_context_ids: tuple[str, ...]
    allocation_inputs_hash: str
    resource_preflight_hash: str
    created_before_outcomes: bool = True
    schema_name: str = "predictable_allocation_look_receipt_v1"

    def __post_init__(self) -> None:
        if (
            not self.proposal_id
            or int(self.look_index) <= 0
            or not self.selected_stratum
        ):
            raise ValueError("predictable allocation receipt lacks identity")
        for value, label in (
            (self.proposal_registration_hash, "proposal registration hash"),
            (self.sampling_design_hash, "sampling design hash"),
            (self.allocation_rule_hash, "allocation rule hash"),
            (self.previous_context_prefix_hash, "previous context prefix hash"),
            (self.planned_panel_hash, "planned panel hash"),
            (self.allocation_inputs_hash, "allocation inputs hash"),
            (self.resource_preflight_hash, "resource preflight hash"),
        ):
            _require_hash(value, label)
        if self.previous_allocation_receipt_hash is not None:
            _require_hash(
                self.previous_allocation_receipt_hash,
                "previous allocation receipt hash",
            )
        targets = tuple(
            (str(stratum_id), int(target))
            for stratum_id, target in self.cumulative_targets
        )
        new_contexts = tuple(str(value) for value in self.planned_new_context_ids)
        if (
            not targets
            or len({stratum_id for stratum_id, _ in targets}) != len(targets)
            or any(target < 0 for _, target in targets)
            or sum(target for _, target in targets) <= 0
            or not new_contexts
            or len(new_contexts) != len(set(new_contexts))
        ):
            raise ValueError(
                "predictable allocation receipt has invalid targets/contexts"
            )
        if not self.created_before_outcomes:
            raise ValueError(
                "allocation look must be committed before backend outcomes"
            )
        if self.schema_name != "predictable_allocation_look_receipt_v1":
            raise ValueError("unsupported predictable allocation receipt schema")
        object.__setattr__(self, "look_index", int(self.look_index))
        object.__setattr__(self, "cumulative_targets", tuple(sorted(targets)))
        object.__setattr__(self, "planned_new_context_ids", new_contexts)

    @property
    def content_hash(self) -> str:
        return content_hash(self)


TerminalAuditDesignKind = Literal[
    "finite_census",
    "finite_population_wor",
    "iid_fixed_n",
]


@dataclass(frozen=True, slots=True)
class TerminalParityEstimandSpec:
    """Birth-frozen faithfulness estimand, distinct from sequential Brier risk."""

    estimand_id: str
    reference_state_hash: str
    reference_calibration_hash: str
    adjudication_population_hash: str
    terminal_population_hash: str
    terminal_support_distribution_hash: str
    prediction_rule_hash: str
    query_weights: tuple[tuple[str, float], ...]
    stratum_weights: tuple[tuple[str, float], ...]
    parity_tolerance: float
    parity_event: Literal["declared_decision_disagreement"] = (
        "declared_decision_disagreement"
    )
    schema_name: str = "terminal_parity_estimand_v1"

    def __post_init__(self) -> None:
        if not self.estimand_id:
            raise ValueError("terminal parity estimand requires stable identity")
        for value, label in (
            (self.reference_state_hash, "terminal reference state hash"),
            (
                self.reference_calibration_hash,
                "terminal reference calibration hash",
            ),
            (
                self.adjudication_population_hash,
                "sequential adjudication population hash",
            ),
            (self.terminal_population_hash, "terminal population hash"),
            (
                self.terminal_support_distribution_hash,
                "terminal support-distribution hash",
            ),
            (self.prediction_rule_hash, "terminal prediction-rule hash"),
        ):
            _require_hash(value, label)
        if self.terminal_population_hash == self.adjudication_population_hash:
            raise ValueError(
                "terminal and sequential adjudication populations require "
                "distinct identities"
            )
        tolerance = float(self.parity_tolerance)
        if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
            raise ValueError("terminal parity tolerance must lie in [0,1]")
        if self.parity_event != "declared_decision_disagreement":
            raise ValueError("unsupported terminal parity event")
        if self.schema_name != "terminal_parity_estimand_v1":
            raise ValueError("unsupported terminal parity estimand schema")
        object.__setattr__(
            self,
            "query_weights",
            _normalized_weights(self.query_weights, label="terminal query weights"),
        )
        object.__setattr__(
            self,
            "stratum_weights",
            _normalized_weights(
                self.stratum_weights,
                label="terminal stratum weights",
            ),
        )
        object.__setattr__(self, "parity_tolerance", tolerance)

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class TerminalAuditPrecommitment:
    """Campaign-birth support, sample-size, and execution commitment.

    The realized finite permutation or IID draws may be revealed later, but
    neither their support law nor their exact target sizes may change after
    this object is installed.
    """

    precommitment_id: str
    estimand: TerminalParityEstimandSpec
    design_kind: TerminalAuditDesignKind
    support_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    target_sample_sizes_by_stratum: tuple[tuple[str, int], ...]
    terminal_alpha: float
    policy_hash: str
    randomization_method: str
    provider_identity_hash: str
    executor_identity_hash: str
    resource_envelope_hash: str
    support_law: Literal["uniform_within_stratum"] = "uniform_within_stratum"
    schema_name: str = "terminal_audit_precommitment_v1"

    def __post_init__(self) -> None:
        if not self.precommitment_id or not self.randomization_method:
            raise ValueError("terminal precommitment requires stable identity")
        if not isinstance(self.estimand, TerminalParityEstimandSpec):
            raise TypeError("terminal precommitment requires a typed parity estimand")
        for value, label in (
            (self.policy_hash, "terminal policy hash"),
            (self.provider_identity_hash, "terminal provider identity hash"),
            (self.executor_identity_hash, "terminal executor identity hash"),
            (self.resource_envelope_hash, "terminal resource-envelope hash"),
        ):
            _require_hash(value, label)
        support = tuple(
            (
                str(stratum_id),
                tuple(sorted(str(context_id) for context_id in context_ids)),
            )
            for stratum_id, context_ids in self.support_context_ids_by_stratum
        )
        targets = tuple(
            (str(stratum_id), int(target))
            for stratum_id, target in self.target_sample_sizes_by_stratum
        )
        if (
            not support
            or len({stratum_id for stratum_id, _ in support}) != len(support)
            or any(
                not stratum_id
                or not context_ids
                or len(context_ids) != len(set(context_ids))
                for stratum_id, context_ids in support
            )
        ):
            raise ValueError(
                "terminal support requires unique nonempty strata and contexts"
            )
        all_contexts = tuple(
            context_id for _, context_ids in support for context_id in context_ids
        )
        if len(all_contexts) != len(set(all_contexts)):
            raise ValueError(
                "terminal support contexts cannot belong to multiple strata"
            )
        support_axis = {stratum_id for stratum_id, _ in support}
        target_axis = {stratum_id for stratum_id, _ in targets}
        weight_axis = {stratum_id for stratum_id, _ in self.estimand.stratum_weights}
        if (
            not targets
            or len(target_axis) != len(targets)
            or support_axis != target_axis
            or support_axis != weight_axis
        ):
            raise ValueError(
                "terminal support, target, and estimand stratum axes differ"
            )
        support_sizes = {
            stratum_id: len(context_ids) for stratum_id, context_ids in support
        }
        if any(target <= 0 for _, target in targets):
            raise ValueError("terminal target sample sizes must be positive")
        if self.design_kind == "finite_census":
            if any(
                target != support_sizes[stratum_id] for stratum_id, target in targets
            ):
                raise ValueError(
                    "terminal census targets must equal the complete support"
                )
        elif self.design_kind == "finite_population_wor":
            if any(
                target > support_sizes[stratum_id] for stratum_id, target in targets
            ):
                raise ValueError("finite terminal targets cannot exceed their support")
        elif self.design_kind != "iid_fixed_n":
            raise ValueError("unsupported terminal audit design kind")
        if self.support_law != "uniform_within_stratum":
            raise ValueError("unsupported terminal support law")
        expected_support_hash = self.support_distribution_hash_for(
            terminal_population_hash=self.estimand.terminal_population_hash,
            support_context_ids_by_stratum=support,
            support_law=self.support_law,
        )
        if self.estimand.terminal_support_distribution_hash != expected_support_hash:
            raise ValueError(
                "terminal support-distribution hash differs from its frozen law"
            )
        alpha = float(self.terminal_alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("terminal alpha must lie in (0,1)")
        if self.schema_name != "terminal_audit_precommitment_v1":
            raise ValueError("unsupported terminal precommitment schema")
        object.__setattr__(
            self,
            "support_context_ids_by_stratum",
            tuple(sorted(support)),
        )
        object.__setattr__(
            self,
            "target_sample_sizes_by_stratum",
            tuple(sorted(targets)),
        )
        object.__setattr__(self, "terminal_alpha", alpha)

    @staticmethod
    def support_distribution_hash_for(
        *,
        terminal_population_hash: str,
        support_context_ids_by_stratum: Sequence[tuple[str, Sequence[str]]],
        support_law: str = "uniform_within_stratum",
    ) -> str:
        _require_hash(terminal_population_hash, "terminal population hash")
        support = tuple(
            sorted(
                (
                    str(stratum_id),
                    tuple(sorted(str(context_id) for context_id in context_ids)),
                )
                for stratum_id, context_ids in support_context_ids_by_stratum
            )
        )
        if support_law != "uniform_within_stratum":
            raise ValueError("unsupported terminal support law")
        return content_hash(
            {
                "schema_name": "terminal_support_distribution_v1",
                "terminal_population_hash": terminal_population_hash,
                "support_law": support_law,
                "support_context_ids_by_stratum": support,
            }
        )

    @property
    def support_context_ids(self) -> tuple[str, ...]:
        return tuple(
            context_id
            for _, context_ids in self.support_context_ids_by_stratum
            for context_id in context_ids
        )

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class ValidationLock:
    """Final state lock created before the untouched audit is opened."""

    lock_id: str
    final_state_hash: str
    final_calibration_hash: str
    prediction_rule_hash: str
    terminal_precommitment_hash: str
    terminal_population_hash: str
    sampling_design_hash: str
    audit_observation_set: ObservationSetRef
    terminal_alpha: float
    parity_tolerance: float
    locked_after_proposal_index: int
    audit_opened: bool = False
    schema_name: str = "terminal_validation_lock_v1"

    def __post_init__(self) -> None:
        if not self.lock_id:
            raise ValueError("validation lock requires lock_id")
        for value, label in (
            (self.final_state_hash, "final state hash"),
            (self.final_calibration_hash, "final calibration hash"),
            (self.prediction_rule_hash, "prediction rule hash"),
            (
                self.terminal_precommitment_hash,
                "terminal precommitment hash",
            ),
            (self.terminal_population_hash, "terminal population hash"),
            (self.sampling_design_hash, "terminal sampling design hash"),
        ):
            _require_hash(value, label)
        if self.audit_observation_set.role is not EvidenceRole.TERMINAL_AUDIT:
            raise ValueError(
                "validation lock requires a terminal-audit observation set"
            )
        if not 0.0 < float(self.terminal_alpha) < 1.0:
            raise ValueError("terminal alpha must lie in (0,1)")
        tolerance = float(self.parity_tolerance)
        if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
            raise ValueError("terminal parity tolerance must lie in [0,1]")
        if int(self.locked_after_proposal_index) < 0:
            raise ValueError("validation lock proposal index cannot be negative")
        if self.audit_observation_set.sampling_design_hash != self.sampling_design_hash:
            raise ValueError(
                "terminal lock observation set differs from its sampling design"
            )
        object.__setattr__(self, "terminal_alpha", float(self.terminal_alpha))
        object.__setattr__(self, "parity_tolerance", tolerance)
        object.__setattr__(
            self, "locked_after_proposal_index", int(self.locked_after_proposal_index)
        )

    def open_audit(self) -> "ValidationLock":
        if self.audit_opened:
            raise RuntimeError("terminal audit may be opened only once")
        return replace(self, audit_opened=True)

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class TerminalParityCertificate:
    validation_lock_hash: str
    final_state_hash: str
    audit_observation_set_hash: str
    disagreement_count: int
    sample_size: int
    upper_bound: float
    tolerance: float
    certified: bool
    method: str
    schema_name: str = "terminal_parity_certificate_v1"

    def __post_init__(self) -> None:
        for value, label in (
            (self.validation_lock_hash, "validation lock hash"),
            (self.final_state_hash, "final state hash"),
            (self.audit_observation_set_hash, "audit observation set hash"),
        ):
            _require_hash(value, label)
        disagreements = int(self.disagreement_count)
        size = int(self.sample_size)
        if size <= 0 or disagreements < 0 or disagreements > size:
            raise ValueError("terminal parity counts are invalid")
        upper = float(self.upper_bound)
        tolerance = float(self.tolerance)
        if not 0.0 <= upper <= 1.0 or not 0.0 <= tolerance <= 1.0:
            raise ValueError("terminal parity bounds must lie in [0,1]")
        if bool(self.certified) != (upper <= tolerance):
            raise ValueError("terminal parity verdict disagrees with its bound")
        if not self.method:
            raise ValueError("terminal parity certificate requires a method")
        object.__setattr__(self, "disagreement_count", disagreements)
        object.__setattr__(self, "sample_size", size)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class TerminalAuditEvidenceReceipt:
    """Physical one-shot audit evidence underlying a parity certificate.

    A terminal certificate is not accepted from free-form counts.  This
    receipt binds it to the exact opened context axis, the frozen sampling
    design, paired backend output, disagreement identities, and the resource
    ledger after execution.
    """

    certificate: TerminalParityCertificate
    terminal_precommitment_hash: str
    policy_hash: str
    sampling_design_hash: str
    paired_evidence_hash: str
    resource_ledger_hash: str
    observed_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    disagreement_context_ids: tuple[str, ...]
    schema_name: str = "terminal_audit_evidence_receipt_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.certificate, TerminalParityCertificate):
            raise TypeError(
                "terminal audit evidence requires a typed parity certificate"
            )
        for value, label in (
            (
                self.terminal_precommitment_hash,
                "terminal precommitment hash",
            ),
            (self.policy_hash, "terminal policy hash"),
            (self.sampling_design_hash, "terminal sampling design hash"),
            (self.paired_evidence_hash, "terminal paired evidence hash"),
            (self.resource_ledger_hash, "terminal resource ledger hash"),
        ):
            _require_hash(value, label)
        rows = tuple(
            (
                str(stratum_id),
                tuple(str(context_id) for context_id in context_ids),
            )
            for stratum_id, context_ids in self.observed_context_ids_by_stratum
        )
        if (
            not rows
            or len({stratum_id for stratum_id, _ in rows}) != len(rows)
            or any(
                not stratum_id
                or not context_ids
                or len(set(context_ids)) != len(context_ids)
                or any(not context_id for context_id in context_ids)
                for stratum_id, context_ids in rows
            )
        ):
            raise ValueError(
                "terminal audit evidence has an invalid stratum/context axis"
            )
        observed = tuple(
            context_id for _, context_ids in rows for context_id in context_ids
        )
        if len(set(observed)) != len(observed):
            raise ValueError("terminal audit evidence repeated a context")
        disagreements = tuple(str(value) for value in self.disagreement_context_ids)
        if (
            len(set(disagreements)) != len(disagreements)
            or any(not value for value in disagreements)
            or not set(disagreements).issubset(observed)
        ):
            raise ValueError("terminal audit disagreement identities are invalid")
        if len(observed) != self.certificate.sample_size:
            raise ValueError(
                "terminal audit evidence sample count differs from its certificate"
            )
        if len(disagreements) != self.certificate.disagreement_count:
            raise ValueError(
                "terminal audit evidence disagreement count differs from its "
                "certificate"
            )
        object.__setattr__(self, "observed_context_ids_by_stratum", rows)
        object.__setattr__(self, "disagreement_context_ids", disagreements)

    @property
    def observed_context_ids(self) -> tuple[str, ...]:
        return tuple(
            context_id
            for _, context_ids in self.observed_context_ids_by_stratum
            for context_id in context_ids
        )

    @property
    def content_hash(self) -> str:
        return content_hash(self)


__all__ = [
    "EVIDENCE_FOUNDATION_GLOCAL_V3",
    "GLOCAL_CONTRACT_SCHEMA",
    "GLOCAL_EXTERNAL_ENTROPY_SOURCE_V1",
    "GLOCAL_IID_FIXED_N_RANDOMIZATION_METHOD_V1",
    "GLOCAL_IID_RUNTIME_RANDOMIZATION_METHOD_V2",
    "GLOCAL_SEALED_PERMUTATION_ALGORITHM_V1",
    "GLOCAL_SYSTEM_ENTROPY_SOURCE_V1",
    "IIDDrawRecord",
    "IIDSamplingReceipt",
    "IIDSupportDistribution",
    "JOINT_RISK_QUERY_ID",
    "CalibrationAlgorithmSpec",
    "CalibrationFitState",
    "EvidenceRole",
    "ObservationSetRef",
    "PredictableAllocationLookReceipt",
    "PredictableProposalRegistration",
    "PredictionRuleSpec",
    "RiskEstimandSpec",
    "SamplingDesignSpec",
    "SamplingPermutationReceipt",
    "SamplingStratumSpec",
    "TerminalAuditDesignKind",
    "TerminalAuditEvidenceReceipt",
    "TerminalAuditPrecommitment",
    "TerminalParityEstimandSpec",
    "TerminalParityCertificate",
    "ValidationLock",
    "binary_brier_loss",
    "canonical_json",
    "class_balanced_observation_weights",
    "content_hash",
    "iid_physical_executor_identity_hash",
    "iid_physical_provider_identity_hash",
    "normalized_multiclass_brier_loss",
    "paired_brier_increment",
    "validate_role_separation",
]
