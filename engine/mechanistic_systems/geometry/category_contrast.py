"""Workflow-neutral vector-valued category-contrast response geometry.

v0 remains the immutable historical target-logit control.  v1-v4 operate on a
shift-invariant category-contrast response law.  Scalar target-writing,
log-odds margin, cross-entropy risk, probability, and decision parity are
explicit derived views rather than silently interchangeable objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.response_geometry import (
    HistoricalResponseProfileName,
    ResponseFunctionalSpec,
    ResponseMetricSpec,
)

ResponseRepresentation = Literal[
    "legacy_target_logit",
    "full_category_contrast",
    "target_competitor_aggregate",
    "smooth_log_odds_margin",
    "top1_margin",
    "cross_entropy_risk",
]
ResponseMetricKind = Literal[
    "contrast_euclidean",
    "margin_euclidean",
    "softmax_fisher",
]
CompetitorPolicy = Literal["all", "declared", "top_k"]

_SCIENTIFIC_RESPONSE_PAYLOAD = "scientific_response_payload"
_NUMERICAL_RESPONSE_PAYLOAD = "numerical_response_payload"
_NUMERICAL_RESPONSE_HASH = "numerical_response_hash"
_NUMERICAL_RESPONSE_HASH_PREFIX = "category-contrast-numerical-binding-"


class _FrozenList(tuple[Any, ...]):
    """Immutable list representation that preserves round-trip container type."""


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return _FrozenList(_freeze_metadata(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_metadata(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
            array.shape
        )
        frozen.setflags(write=False)
        return frozen
    return value


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [_thaw_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_thaw_metadata(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_thaw_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True, order="C")
        array.setflags(write=False)
        return array
    return value


def _functional_payload(functional: ResponseFunctionalSpec) -> dict[str, Any]:
    return {
        "response_id": functional.response_id,
        "kind": functional.kind,
        "target_index": int(functional.target_index),
        "class_count": int(functional.class_count),
        "competitor_policy": functional.competitor_policy,
        "competitor_indices": list(functional.competitor_indices),
        "top_k": functional.top_k,
        "calibrated": bool(functional.calibrated),
        "parameters": _thaw_metadata(functional.parameters),
        "schema": functional.schema,
    }


def _functional_from_payload(payload: object) -> ResponseFunctionalSpec:
    if not isinstance(payload, Mapping):
        raise ValueError("scientific response payload must be a mapping")
    expected = {
        "response_id",
        "kind",
        "target_index",
        "class_count",
        "competitor_policy",
        "competitor_indices",
        "top_k",
        "calibrated",
        "parameters",
        "schema",
    }
    if set(payload) != expected:
        raise ValueError("scientific response payload has an invalid field set")
    return ResponseFunctionalSpec(
        response_id=str(payload["response_id"]),
        kind=payload["kind"],  # type: ignore[arg-type]
        target_index=int(payload["target_index"]),
        class_count=int(payload["class_count"]),
        competitor_policy=payload["competitor_policy"],  # type: ignore[arg-type]
        competitor_indices=tuple(payload["competitor_indices"]),  # type: ignore[arg-type]
        top_k=None if payload["top_k"] is None else int(payload["top_k"]),
        calibrated=bool(payload["calibrated"]),
        parameters=_thaw_metadata(payload["parameters"]),  # type: ignore[arg-type]
        schema=str(payload["schema"]),
    )


def _readonly(value: np.ndarray, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    array = np.array(array, copy=True, order="C")
    array.setflags(write=False)
    return array


def _logsumexp(value: np.ndarray, *, axis: int = -1) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.sum(np.exp(array - maximum), axis=axis)
    )


def _softmax(value: np.ndarray, *, axis: int = -1) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=axis, keepdims=True)


@dataclass(frozen=True, slots=True)
class CategoryContrastResponseSpec:
    """Frozen scientific identity of a target-category response law."""

    target_index: int
    class_ids: tuple[str, ...]
    representation: ResponseRepresentation = "target_competitor_aggregate"
    metric_kind: ResponseMetricKind = "margin_euclidean"
    competitor_policy: CompetitorPolicy = "all"
    competitor_indices: tuple[int, ...] = ()
    top_k: int | None = None
    temperature: float = 1.0
    metric_regularization: float = 1e-10
    schema: str = "category_contrast_response_v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        classes = tuple(map(str, self.class_ids))
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError(
                "category-contrast response requires at least two unique classes"
            )
        if not 0 <= int(self.target_index) < len(classes):
            raise ValueError("target_index lies outside the class axis")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("response temperature must be finite and positive")
        if (
            not np.isfinite(self.metric_regularization)
            or self.metric_regularization < 0
        ):
            raise ValueError("metric_regularization must be finite and nonnegative")
        competitors = tuple(int(index) for index in self.competitor_indices)
        if competitors:
            if len(set(competitors)) != len(competitors):
                raise ValueError("competitor_indices must be unique")
            if self.target_index in competitors or any(
                index < 0 or index >= len(classes) for index in competitors
            ):
                raise ValueError(
                    "competitor_indices must name valid non-target classes"
                )
        if self.competitor_policy == "declared" and not competitors:
            raise ValueError("declared competitor policy requires competitor_indices")
        if self.competitor_policy == "top_k":
            if self.top_k is None or not 1 <= int(self.top_k) <= len(classes) - 1:
                raise ValueError(
                    "top_k competitor policy requires 1 <= top_k < class_count"
                )
        if (
            self.representation == "legacy_target_logit"
            and self.metric_kind != "contrast_euclidean"
        ):
            raise ValueError(
                "legacy target-logit control uses its scalar Euclidean metric"
            )
        object.__setattr__(self, "class_ids", classes)
        object.__setattr__(self, "competitor_indices", competitors)
        object.__setattr__(self, "metadata", _freeze_metadata(dict(self.metadata)))

    def _numerical_contract_payload(self) -> dict[str, Any]:
        return {
            "target_index": int(self.target_index),
            "class_ids": list(self.class_ids),
            "representation": self.representation,
            "metric_kind": self.metric_kind,
            "competitor_policy": self.competitor_policy,
            "competitor_indices": list(self.competitor_indices),
            "top_k": self.top_k,
            "temperature": float(self.temperature),
            "metric_regularization": float(self.metric_regularization),
            "schema": self.schema,
        }

    @property
    def target_class_id(self) -> str:
        return self.class_ids[self.target_index]

    @property
    def content_hash(self) -> str:
        return content_hash(self.to_dict(include_hash=False), prefix="irg-response-")

    @property
    def response_id(self) -> str:
        return f"category-contrast:{self.target_class_id}:{self.content_hash[-16:]}"

    @property
    def output_dimension(self) -> int:
        if self.representation in {
            "legacy_target_logit",
            "smooth_log_odds_margin",
            "top1_margin",
            "cross_entropy_risk",
        }:
            return 1
        if self.representation == "target_competitor_aggregate":
            return 2
        if self.competitor_policy == "declared":
            return len(self.competitor_indices)
        if self.competitor_policy == "top_k":
            assert self.top_k is not None
            return int(self.top_k)
        return len(self.class_ids) - 1

    @property
    def is_target_rest_complete(self) -> bool:
        """Whether target probability/risk versus aggregate rest is exact."""
        return self.representation in {
            "target_competitor_aggregate",
            "full_category_contrast",
        } and (
            self.representation == "target_competitor_aggregate"
            or self.is_competitor_complete
        )

    @property
    def is_competitor_complete(self) -> bool:
        """Whether individual non-target competitor identities are all retained."""
        if self.representation != "full_category_contrast":
            return False
        required = {
            index for index in range(len(self.class_ids)) if index != self.target_index
        }
        if self.competitor_policy == "all":
            return True
        if self.competitor_policy == "declared":
            return set(self.competitor_indices) == required
        return self.top_k == len(required)

    @property
    def is_contrast_complete(self) -> bool:
        """Whether the response retains the complete target-vs-alternatives law.

        The two-channel target/rest chart is complete because the rest channel
        is the exact log-sum-exp aggregate.  A pairwise chart is complete only
        when every non-target class is present.  This guard prevents a top-k
        diagnostic from silently becoming the scientific response of v1-v4.
        """

        return self.is_target_rest_complete

    def assert_version_compatible(
        self,
        version: HistoricalResponseProfileName,
        *,
        require_competitor_identity: bool = False,
    ) -> None:
        if version == "v0":
            if self.representation != "legacy_target_logit":
                raise ValueError(
                    "immutable irg/v0 requires the historical target-logit response"
                )
            return
        if not self.is_target_rest_complete:
            raise ValueError(
                "irg/v1-v4 require an exact category-contrast response: target-versus-rest or "
                "competitor-resolved; scalar margin/risk views are diagnostics"
            )
        if require_competitor_identity and not self.is_competitor_complete:
            raise ValueError(
                "this analysis requires competitor-resolved category contrasts; the two-channel "
                "target/rest chart does not retain competitor identity"
            )

    @classmethod
    def from_functional_spec(
        cls,
        functional: ResponseFunctionalSpec,
        *,
        class_ids: Sequence[str] | None = None,
        metric_kind: ResponseMetricKind = "margin_euclidean",
    ) -> "CategoryContrastResponseSpec":
        """Construct the numerical response contract from the scientific spec."""

        classes = tuple(
            class_ids or (str(index) for index in range(functional.class_count))
        )
        if len(classes) != functional.class_count:
            raise ValueError(
                "class_ids must align with ResponseFunctionalSpec.class_count"
            )
        representation: ResponseRepresentation
        if functional.kind == "target_logit":
            representation = "legacy_target_logit"
            metric_kind = "contrast_euclidean"
        elif functional.kind == "category_contrast":
            representation = "full_category_contrast"
        elif functional.kind == "target_vs_rest_channels":
            representation = "target_competitor_aggregate"
        elif functional.kind == "target_vs_rest_margin":
            representation = "smooth_log_odds_margin"
        elif functional.kind == "top1_margin":
            representation = "top1_margin"
        elif functional.kind == "cross_entropy":
            representation = "cross_entropy_risk"
        else:
            raise ValueError(
                f"unsupported response functional kind: {functional.kind!r}"
            )
        policy = {
            "explicit": "declared",
            "top_k": "top_k",
            "all": "all",
            "aggregate_rest": "all",
        }[functional.competitor_policy]
        temperature = float(functional.parameters.get("temperature", 1.0))
        metric_regularization = float(
            functional.parameters.get("metric_regularization", 1e-10)
        )
        numerical_payload = {
            "target_index": int(functional.target_index),
            "class_ids": list(classes),
            "representation": representation,
            "metric_kind": metric_kind,
            "competitor_policy": policy,
            "competitor_indices": list(functional.competitor_indices),
            "top_k": functional.top_k,
            "temperature": temperature,
            "metric_regularization": metric_regularization,
            "schema": "category_contrast_response_v1",
        }
        return cls(
            target_index=functional.target_index,
            class_ids=classes,
            representation=representation,
            metric_kind=metric_kind,
            competitor_policy=policy,  # type: ignore[arg-type]
            competitor_indices=functional.competitor_indices,
            top_k=functional.top_k,
            temperature=temperature,
            metric_regularization=metric_regularization,
            metadata={
                **dict(functional.parameters),
                "scientific_response_spec_hash": functional.content_hash,
                "scientific_response_id": functional.response_id,
                _SCIENTIFIC_RESPONSE_PAYLOAD: _functional_payload(functional),
                _NUMERICAL_RESPONSE_PAYLOAD: numerical_payload,
                _NUMERICAL_RESPONSE_HASH: content_hash(
                    numerical_payload,
                    prefix=_NUMERICAL_RESPONSE_HASH_PREFIX,
                ),
            },
        )

    def _source_functional_spec(self) -> ResponseFunctionalSpec | None:
        payload = self.metadata.get(_SCIENTIFIC_RESPONSE_PAYLOAD)
        if payload is None:
            return None
        functional = _functional_from_payload(payload)
        if functional.content_hash != self.metadata.get(
            "scientific_response_spec_hash"
        ):
            raise ValueError(
                "scientific response payload hash does not match its binding"
            )
        if functional.response_id != self.metadata.get("scientific_response_id"):
            raise ValueError(
                "scientific response payload ID does not match its binding"
            )
        numerical_payload = self.metadata.get(_NUMERICAL_RESPONSE_PAYLOAD)
        numerical_hash = self.metadata.get(_NUMERICAL_RESPONSE_HASH)
        numerical_fields = {
            "target_index",
            "class_ids",
            "representation",
            "metric_kind",
            "competitor_policy",
            "competitor_indices",
            "top_k",
            "temperature",
            "metric_regularization",
            "schema",
        }
        if (
            not isinstance(numerical_payload, Mapping)
            or set(numerical_payload) != numerical_fields
        ):
            raise ValueError("numerical response binding has an invalid field set")
        stored_hash = content_hash(
            numerical_payload,
            prefix=_NUMERICAL_RESPONSE_HASH_PREFIX,
        )
        current_hash = content_hash(
            self._numerical_contract_payload(),
            prefix=_NUMERICAL_RESPONSE_HASH_PREFIX,
        )
        if numerical_hash != stored_hash or current_hash != stored_hash:
            raise ValueError(
                "scientific response payload does not match the numerical response contract"
            )
        return functional

    def functional_spec(
        self, *, calibrated: bool | None = None
    ) -> ResponseFunctionalSpec:
        """Return the frozen scientific response identity used by IRG sessions."""

        source = self._source_functional_spec()
        if source is not None:
            if calibrated is None or calibrated == source.calibrated:
                return source
            return replace(source, calibrated=calibrated)

        if self.representation == "legacy_target_logit":
            kind = "target_logit"
        elif self.representation == "full_category_contrast":
            kind = "category_contrast"
        elif self.representation == "target_competitor_aggregate":
            kind = "target_vs_rest_channels"
        elif self.representation == "smooth_log_odds_margin":
            kind = "target_vs_rest_margin"
        elif self.representation == "top1_margin":
            kind = "top1_margin"
        else:
            kind = "cross_entropy"
        policy = {"declared": "explicit", "top_k": "top_k", "all": "all"}[
            self.competitor_policy
        ]
        return ResponseFunctionalSpec(
            response_id=self.response_id,
            kind=kind,  # type: ignore[arg-type]
            target_index=self.target_index,
            class_count=len(self.class_ids),
            competitor_policy=policy,  # type: ignore[arg-type]
            competitor_indices=self.competitor_indices,
            top_k=self.top_k,
            calibrated=True if calibrated is None else calibrated,
            parameters={
                "category_contrast_response_hash": self.content_hash,
                "temperature": self.temperature,
                "metric_regularization": self.metric_regularization,
                "metric_kind": self.metric_kind,
                "representation": self.representation,
            },
        )

    def metric_spec(self) -> ResponseMetricSpec:
        geometry = {
            "contrast_euclidean": "contrast_euclidean",
            "margin_euclidean": "margin",
            "softmax_fisher": "softmax_fisher",
        }[self.metric_kind]
        return ResponseMetricSpec(
            metric_id=f"{self.metric_kind}:{self.content_hash[-16:]}",
            output_dimension=self.output_dimension,
            geometry=geometry,  # type: ignore[arg-type]
            parameters={
                "category_contrast_response": self.to_dict(),
                "response_contract_hash": self.content_hash,
            },
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "target_index": int(self.target_index),
            "target_class_id": self.target_class_id,
            "class_ids": list(self.class_ids),
            "representation": self.representation,
            "metric_kind": self.metric_kind,
            "competitor_policy": self.competitor_policy,
            "competitor_indices": list(self.competitor_indices),
            "top_k": self.top_k,
            "temperature": float(self.temperature),
            "metric_regularization": float(self.metric_regularization),
            "schema": self.schema,
            "metadata": _thaw_metadata(self.metadata),
        }
        if include_hash:
            payload["content_hash"] = self.content_hash
        return payload


@dataclass(frozen=True, slots=True)
class CategoryContrastResponse:
    spec: CategoryContrastResponseSpec
    class_axis: tuple[int, ...]
    competitor_axis: tuple[int, ...]
    values: np.ndarray = field(repr=False, compare=False)
    target_writing: np.ndarray = field(repr=False, compare=False)
    competitor_aggregate: np.ndarray = field(repr=False, compare=False)
    log_odds_margin: np.ndarray = field(repr=False, compare=False)
    top1_margin: np.ndarray = field(repr=False, compare=False)
    target_probability: np.ndarray = field(repr=False, compare=False)
    cross_entropy: np.ndarray = field(repr=False, compare=False)
    target_decision: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        rows = np.asarray(self.values).shape[0]
        object.__setattr__(
            self, "values", _readonly(self.values, ndim=2, name="response values")
        )
        for name in (
            "target_writing",
            "competitor_aggregate",
            "log_odds_margin",
            "top1_margin",
            "target_probability",
            "cross_entropy",
        ):
            value = _readonly(np.asarray(getattr(self, name)), ndim=1, name=name)
            if value.shape[0] != rows:
                raise ValueError(f"{name} must align with the context axis")
            object.__setattr__(self, name, value)
        decision = np.asarray(self.target_decision, dtype=bool)
        if decision.shape != (rows,):
            raise ValueError("target_decision must align with the context axis")
        decision = np.array(decision, copy=True)
        decision.setflags(write=False)
        object.__setattr__(self, "target_decision", decision)


@dataclass(frozen=True, slots=True)
class CategoryContrastJacobian:
    spec: CategoryContrastResponseSpec
    response: CategoryContrastResponse
    jacobian: np.ndarray = field(repr=False, compare=False)  # [context,response,route]
    metric: np.ndarray | None = field(
        repr=False, compare=False
    )  # [context,response,response]
    metric_factor: np.ndarray = field(
        repr=False, compare=False
    )  # [context,factor,route]
    oriented_margin_rows: np.ndarray = field(
        repr=False, compare=False
    )  # [context,route]

    def __post_init__(self) -> None:
        jacobian = _readonly(self.jacobian, ndim=3, name="category-contrast Jacobian")
        metric = (
            None
            if self.metric is None
            else _readonly(self.metric, ndim=3, name="category-contrast metric")
        )
        factor = _readonly(
            self.metric_factor, ndim=3, name="category-contrast metric factor"
        )
        oriented = _readonly(
            self.oriented_margin_rows, ndim=2, name="oriented margin rows"
        )
        n, d, _ = jacobian.shape
        if (
            factor.shape[0] != n
            or factor.shape[2] != jacobian.shape[2]
            or oriented.shape[0] != n
        ):
            raise ValueError(
                "response Jacobian, metric, and margin rows have inconsistent axes"
            )
        if metric is not None:
            if metric.shape != (n, d, d):
                raise ValueError("dense response metric has inconsistent axes")
            for index in range(n):
                if not np.allclose(
                    metric[index], metric[index].T, atol=1e-9, rtol=1e-9
                ):
                    raise ValueError("response metric must be symmetric")
                if float(np.min(np.linalg.eigvalsh(metric[index]))) < -1e-8:
                    raise ValueError("response metric must be positive semidefinite")
                explicit = jacobian[index].T @ metric[index] @ jacobian[index]
                factored = factor[index].T @ factor[index]
                if not np.allclose(explicit, factored, atol=2e-8, rtol=2e-8):
                    raise ValueError(
                        "response metric factor does not realize the declared dense metric"
                    )
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "metric_factor", factor)
        object.__setattr__(self, "oriented_margin_rows", oriented)

    @property
    def route_energy_rows(self) -> np.ndarray:
        return np.sum(self.metric_factor * self.metric_factor, axis=1)

    @property
    def flattened_metric_factor(self) -> np.ndarray:
        """Return factor rows F with ``F.T F / n = E[J.T H J]``.

        The response-coordinate rows are flattened.  They are scaled by
        ``sqrt(d)`` so legacy matrix consumers that divide by the flattened row
        count recover the physical-context average rather than an average over
        response coordinates.
        """

        n, factor_dimension, routes = self.metric_factor.shape
        return np.ascontiguousarray(
            self.metric_factor.reshape(n * factor_dimension, routes)
            * np.sqrt(float(factor_dimension))
        )

    def marginal_statistics(self) -> np.ndarray:
        n = self.jacobian.shape[0]
        energy_rows = self.route_energy_rows
        mean = np.mean(self.oriented_margin_rows, axis=0)
        energy = np.mean(energy_rows, axis=0)
        mean_se = (
            np.std(self.oriented_margin_rows, axis=0, ddof=1) / np.sqrt(n)
            if n >= 2
            else np.zeros(self.jacobian.shape[2])
        )
        energy_se = (
            np.std(energy_rows, axis=0, ddof=1) / np.sqrt(n)
            if n >= 2
            else np.zeros(self.jacobian.shape[2])
        )
        return np.vstack((mean, energy, mean_se, energy_se))


@dataclass(frozen=True, slots=True)
class ExactMaskResponseComparison:
    spec_hash: str
    target_logit_change: np.ndarray = field(repr=False, compare=False)
    competitor_aggregate_change: np.ndarray = field(repr=False, compare=False)
    log_odds_margin_change: np.ndarray = field(repr=False, compare=False)
    top1_margin_change: np.ndarray = field(repr=False, compare=False)
    target_probability_change: np.ndarray = field(repr=False, compare=False)
    cross_entropy_change: np.ndarray = field(repr=False, compare=False)
    decision_disagreement: np.ndarray = field(repr=False, compare=False)
    distribution_kl: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        names = (
            "target_logit_change",
            "competitor_aggregate_change",
            "log_odds_margin_change",
            "top1_margin_change",
            "target_probability_change",
            "cross_entropy_change",
            "distribution_kl",
        )
        size = None
        for name in names:
            value = _readonly(np.asarray(getattr(self, name)), ndim=1, name=name)
            size = value.size if size is None else size
            if value.size != size:
                raise ValueError("exact response views must share one context axis")
            object.__setattr__(self, name, value)
        disagreement = np.asarray(self.decision_disagreement, dtype=bool)
        if disagreement.shape != (size,):
            raise ValueError("decision_disagreement must share the context axis")
        disagreement = np.array(disagreement, copy=True)
        disagreement.setflags(write=False)
        object.__setattr__(self, "decision_disagreement", disagreement)

    def summary(self) -> dict[str, float]:
        return {
            "mean_absolute_target_logit_change": float(
                np.mean(np.abs(self.target_logit_change))
            ),
            "mean_absolute_competitor_aggregate_change": float(
                np.mean(np.abs(self.competitor_aggregate_change))
            ),
            "mean_absolute_log_odds_margin_change": float(
                np.mean(np.abs(self.log_odds_margin_change))
            ),
            "mean_absolute_top1_margin_change": float(
                np.mean(np.abs(self.top1_margin_change))
            ),
            "mean_absolute_probability_change": float(
                np.mean(np.abs(self.target_probability_change))
            ),
            "mean_cross_entropy_change": float(np.mean(self.cross_entropy_change)),
            "decision_disagreement_rate": float(np.mean(self.decision_disagreement)),
            "mean_distribution_kl": float(np.mean(self.distribution_kl)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_hash": self.spec_hash,
            "summary": self.summary(),
            "array_hashes": {
                "target_logit_change": content_hash(
                    self.target_logit_change, prefix="exact-response-array-"
                ),
                "competitor_aggregate_change": content_hash(
                    self.competitor_aggregate_change, prefix="exact-response-array-"
                ),
                "log_odds_margin_change": content_hash(
                    self.log_odds_margin_change, prefix="exact-response-array-"
                ),
                "top1_margin_change": content_hash(
                    self.top1_margin_change, prefix="exact-response-array-"
                ),
                "target_probability_change": content_hash(
                    self.target_probability_change, prefix="exact-response-array-"
                ),
                "cross_entropy_change": content_hash(
                    self.cross_entropy_change, prefix="exact-response-array-"
                ),
                "decision_disagreement": content_hash(
                    self.decision_disagreement.astype(np.uint8),
                    prefix="exact-response-array-",
                ),
                "distribution_kl": content_hash(
                    self.distribution_kl, prefix="exact-response-array-"
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class ExactMaskResponseReceipt:
    """Hash-bound multiview response receipt for one complete hard mask.

    The receipt does not replace the pruning objective.  It prevents target
    writing, competitor integration, risk, probability, and final decision
    views from being silently interchanged after the candidate has been seen.
    """

    receipt_id: str
    response_spec_hash: str
    context_ids: tuple[str, ...]
    reference_mask_hash: str
    candidate_mask_hash: str
    reference_logits_hash: str
    candidate_logits_hash: str
    comparison: ExactMaskResponseComparison
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(
            (
                self.receipt_id,
                self.response_spec_hash,
                self.reference_mask_hash,
                self.candidate_mask_hash,
                self.reference_logits_hash,
                self.candidate_logits_hash,
            )
        ):
            raise ValueError(
                "exact-mask response receipt requires complete scientific identity"
            )
        axis = tuple(map(str, self.context_ids))
        if not axis or len(set(axis)) != len(axis):
            raise ValueError(
                "exact-mask response receipt requires a unique ordered context axis"
            )
        if len(axis) != self.comparison.target_logit_change.size:
            raise ValueError(
                "exact-mask response receipt context axis does not match comparison arrays"
            )
        if self.comparison.spec_hash != self.response_spec_hash:
            raise ValueError(
                "exact-mask response receipt uses a different response contract"
            )
        object.__setattr__(self, "context_ids", axis)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def content_hash(self) -> str:
        return content_hash(
            self.to_dict(include_hash=False), prefix="exact-mask-response-receipt-"
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "receipt_id": self.receipt_id,
            "response_spec_hash": self.response_spec_hash,
            "context_ids": list(self.context_ids),
            "reference_mask_hash": self.reference_mask_hash,
            "candidate_mask_hash": self.candidate_mask_hash,
            "reference_logits_hash": self.reference_logits_hash,
            "candidate_logits_hash": self.candidate_logits_hash,
            "comparison": self.comparison.to_dict(),
            "metadata": dict(self.metadata),
        }
        if include_hash:
            payload["content_hash"] = self.content_hash
        return payload


def _competitor_indices(
    logits: np.ndarray, spec: CategoryContrastResponseSpec
) -> np.ndarray:
    count, classes = logits.shape
    all_competitors = np.asarray(
        [index for index in range(classes) if index != spec.target_index],
        dtype=np.int64,
    )
    if spec.competitor_policy == "declared":
        return np.tile(np.asarray(spec.competitor_indices, dtype=np.int64), (count, 1))
    if spec.competitor_policy == "top_k":
        assert spec.top_k is not None
        competitor_values = logits[:, all_competitors]
        selected = np.argpartition(competitor_values, -int(spec.top_k), axis=1)[
            :, -int(spec.top_k) :
        ]
        # Stable order by descending logit, then class index.
        rows = []
        for row, local in zip(competitor_values, selected):
            classes_selected = all_competitors[local]
            order = np.lexsort((classes_selected, -row[local]))
            rows.append(classes_selected[order])
        return np.asarray(rows, dtype=np.int64)
    return np.tile(all_competitors, (count, 1))


def category_contrast_response(
    logits: np.ndarray | Sequence[Sequence[float]],
    spec: CategoryContrastResponseSpec,
) -> CategoryContrastResponse:
    values = np.asarray(logits, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != len(spec.class_ids)
        or not np.isfinite(values).all()
    ):
        raise ValueError("logits must be finite with the declared class axis")
    scaled = values / float(spec.temperature)
    competitors = _competitor_indices(scaled, spec)
    target = scaled[:, spec.target_index]
    competitor_values = np.take_along_axis(scaled, competitors, axis=1)
    aggregate = _logsumexp(competitor_values, axis=1)
    margin = target - aggregate
    top1 = target - np.max(competitor_values, axis=1)
    probability = _softmax(scaled, axis=1)[:, spec.target_index]
    risk = -np.log(np.maximum(probability, np.finfo(np.float64).tiny))
    decision = np.argmax(scaled, axis=1) == spec.target_index
    if spec.representation == "legacy_target_logit":
        response_values = target[:, None]
    elif spec.representation == "target_competitor_aggregate":
        response_values = np.column_stack((target, aggregate))
    elif spec.representation == "smooth_log_odds_margin":
        response_values = margin[:, None]
    elif spec.representation == "top1_margin":
        response_values = top1[:, None]
    elif spec.representation == "cross_entropy_risk":
        response_values = risk[:, None]
    else:
        response_values = target[:, None] - competitor_values
    return CategoryContrastResponse(
        spec=spec,
        class_axis=tuple(range(values.shape[1])),
        competitor_axis=tuple(
            int(index) for index in sorted(set(competitors.reshape(-1).tolist()))
        ),
        values=response_values,
        target_writing=target,
        competitor_aggregate=aggregate,
        log_odds_margin=margin,
        top1_margin=top1,
        target_probability=probability,
        cross_entropy=risk,
        target_decision=decision,
    )


def category_contrast_jacobian(
    logits: np.ndarray | Sequence[Sequence[float]],
    logit_jacobian: np.ndarray,
    spec: CategoryContrastResponseSpec,
) -> CategoryContrastJacobian:
    raw_logits = np.asarray(logits, dtype=np.float64)
    jacobian = np.asarray(logit_jacobian, dtype=np.float64)
    if (
        jacobian.ndim != 3
        or jacobian.shape[:2] != raw_logits.shape
        or not np.isfinite(jacobian).all()
    ):
        raise ValueError(
            "logit_jacobian must have [context,class,route] axes matching logits"
        )
    response = category_contrast_response(raw_logits, spec)
    if (
        spec.metric_kind == "softmax_fisher"
        and spec.representation == "full_category_contrast"
        and not spec.is_contrast_complete
    ):
        raise ValueError(
            "softmax-Fisher geometry on pairwise category contrasts requires the complete competitor axis"
        )
    scaled_logits = raw_logits / float(spec.temperature)
    scaled_jacobian = jacobian / float(spec.temperature)
    competitors = _competitor_indices(scaled_logits, spec)
    rows = np.arange(raw_logits.shape[0])[:, None]
    target_j = scaled_jacobian[:, spec.target_index, :]
    competitor_j = scaled_jacobian[rows, competitors, :]
    competitor_values = scaled_logits[rows, competitors]
    weights = _softmax(competitor_values, axis=1)
    aggregate_j = np.einsum("nk,nkr->nr", weights, competitor_j)
    margin_j = target_j - aggregate_j
    if spec.representation == "legacy_target_logit":
        response_j = target_j[:, None, :]
    elif spec.representation == "target_competitor_aggregate":
        response_j = np.stack((target_j, aggregate_j), axis=1)
    elif spec.representation == "smooth_log_odds_margin":
        response_j = margin_j[:, None, :]
    elif spec.representation == "top1_margin":
        active = np.argmax(competitor_values, axis=1)
        active_j = competitor_j[np.arange(raw_logits.shape[0]), active]
        response_j = (target_j - active_j)[:, None, :]
    elif spec.representation == "cross_entropy_risk":
        target_probability = _softmax(scaled_logits, axis=1)[:, spec.target_index]
        response_j = (-(1.0 - target_probability)[:, None] * margin_j)[:, None, :]
    else:
        response_j = target_j[:, None, :] - competitor_j

    n, d, _ = response_j.shape
    metric: np.ndarray | None = np.zeros((n, d, d), dtype=np.float64)
    metric_factor: np.ndarray
    if spec.metric_kind == "contrast_euclidean":
        assert metric is not None
        metric[:] = np.eye(d, dtype=np.float64)
        metric_factor = response_j
    elif spec.metric_kind == "margin_euclidean":
        if spec.representation == "target_competitor_aggregate":
            direction = np.array([1.0, -1.0], dtype=np.float64)
            assert metric is not None
            metric[:] = np.outer(direction, direction)
            metric_factor = margin_j[:, None, :]
        elif spec.representation == "full_category_contrast":
            # m = -logsumexp(-r); dm/dr is the conditional competitor
            # distribution.  This is the exact rank-one metric induced by the
            # smooth target-vs-rest margin on the complete contrast chart.
            assert metric is not None
            metric = weights[:, :, None] * weights[:, None, :]
            metric_factor = margin_j[:, None, :]
        else:
            assert metric is not None
            metric[:] = np.eye(d, dtype=np.float64)
            metric_factor = response_j
    else:  # softmax Fisher on the common-shift quotient
        probabilities = _softmax(scaled_logits, axis=1)
        if spec.representation == "target_competitor_aggregate":
            # The two-channel chart is scalar margin geometry.  Cross-entropy
            # curvature is p_c(1-p_c) in this coordinate.
            curvature = probabilities[:, spec.target_index] * (
                1.0 - probabilities[:, spec.target_index]
            )
            direction = np.array([1.0, -1.0], dtype=np.float64)
            metric = (
                curvature[:, None, None] * np.outer(direction, direction)[None, :, :]
            )
            metric_factor = np.sqrt(curvature)[:, None, None] * margin_j[:, None, :]
        elif spec.representation in {"smooth_log_odds_margin", "top1_margin"}:
            assert metric is not None
            curvature = probabilities[:, spec.target_index] * (
                1.0 - probabilities[:, spec.target_index]
            )
            metric[:, 0, 0] = curvature
            metric_factor = np.sqrt(curvature)[:, None, None] * response_j
        elif spec.representation == "cross_entropy_risk":
            # The response coordinate is already the nonlinear risk itself.
            assert metric is not None
            metric[:, 0, 0] = 1.0
            metric_factor = response_j
        elif spec.representation == "legacy_target_logit":
            assert metric is not None
            metric[:, 0, 0] = 1.0
            metric_factor = response_j
        else:
            # For any logit perturbation dz,
            #   dz^T (diag(p)-pp^T) dz
            # = ||diag(sqrt(p))(dz-E_p[dz])||^2.
            # This direct class-space factor avoids constructing or
            # diagonalizing a per-context (K-1)x(K-1) contrast metric.
            weighted_mean_j = np.einsum("nk,nkr->nr", probabilities, scaled_jacobian)
            metric_factor = np.sqrt(probabilities)[:, :, None] * (
                scaled_jacobian - weighted_mean_j[:, None, :]
            )
            if spec.metric_regularization > 0:
                metric_factor = np.concatenate(
                    (
                        metric_factor,
                        np.sqrt(spec.metric_regularization) * response_j,
                    ),
                    axis=1,
                )

            # The dense chart metric is diagnostic only.  Materialize it for
            # bounded response dimensions; native ImageNet-scale execution
            # remains factor-only.
            if d <= 128:
                dense_metric = np.zeros((n, d, d), dtype=np.float64)
                for index in range(n):
                    competitor_index = competitors[index]
                    b = np.zeros((d, raw_logits.shape[1]), dtype=np.float64)
                    b[:, spec.target_index] = 1.0
                    b[np.arange(d), competitor_index] = -1.0
                    right_inverse = b.T @ np.linalg.pinv(b @ b.T, rcond=1e-12)
                    p = probabilities[index]
                    fisher = np.diag(p) - np.outer(p, p)
                    h = right_inverse.T @ fisher @ right_inverse
                    dense_metric[index] = 0.5 * (
                        h + h.T
                    ) + spec.metric_regularization * np.eye(d)
                metric = dense_metric
            else:
                metric = None
    return CategoryContrastJacobian(
        spec=spec,
        response=response,
        jacobian=response_j,
        metric=metric,
        metric_factor=metric_factor,
        oriented_margin_rows=margin_j,
    )


def compare_exact_mask_logits(
    dense_logits: np.ndarray,
    candidate_logits: np.ndarray,
    spec: CategoryContrastResponseSpec,
) -> ExactMaskResponseComparison:
    dense = np.asarray(dense_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    if dense.shape != candidate.shape:
        raise ValueError("dense and candidate logits must share context and class axes")
    dense_view = category_contrast_response(dense, spec)
    candidate_view = category_contrast_response(candidate, spec)
    dense_probability = _softmax(dense / spec.temperature, axis=1)
    candidate_probability = _softmax(candidate / spec.temperature, axis=1)
    kl = np.sum(
        dense_probability
        * (
            np.log(np.maximum(dense_probability, np.finfo(np.float64).tiny))
            - np.log(np.maximum(candidate_probability, np.finfo(np.float64).tiny))
        ),
        axis=1,
    )
    return ExactMaskResponseComparison(
        spec_hash=spec.content_hash,
        target_logit_change=candidate_view.target_writing - dense_view.target_writing,
        competitor_aggregate_change=candidate_view.competitor_aggregate
        - dense_view.competitor_aggregate,
        log_odds_margin_change=candidate_view.log_odds_margin
        - dense_view.log_odds_margin,
        top1_margin_change=candidate_view.top1_margin - dense_view.top1_margin,
        target_probability_change=candidate_view.target_probability
        - dense_view.target_probability,
        cross_entropy_change=candidate_view.cross_entropy - dense_view.cross_entropy,
        decision_disagreement=candidate_view.target_decision
        != dense_view.target_decision,
        distribution_kl=kl,
    )


def build_exact_mask_response_receipt(
    *,
    receipt_id: str,
    context_ids: Sequence[str],
    reference_mask_hash: str,
    candidate_mask_hash: str,
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    response_spec: CategoryContrastResponseSpec,
    metadata: Mapping[str, Any] | None = None,
) -> ExactMaskResponseReceipt:
    reference = np.asarray(reference_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    return ExactMaskResponseReceipt(
        receipt_id=str(receipt_id),
        response_spec_hash=response_spec.content_hash,
        context_ids=tuple(map(str, context_ids)),
        reference_mask_hash=str(reference_mask_hash),
        candidate_mask_hash=str(candidate_mask_hash),
        reference_logits_hash=content_hash(reference, prefix="exact-mask-logits-"),
        candidate_logits_hash=content_hash(candidate, prefix="exact-mask-logits-"),
        comparison=compare_exact_mask_logits(reference, candidate, response_spec),
        metadata={
            "all_declared_response_views_retained": True,
            "response_view_selected_after_candidate_observation": False,
            **dict(metadata or {}),
        },
    )


__all__ = [
    "CategoryContrastJacobian",
    "CategoryContrastResponse",
    "CategoryContrastResponseSpec",
    "ExactMaskResponseComparison",
    "ExactMaskResponseReceipt",
    "build_exact_mask_response_receipt",
    "category_contrast_jacobian",
    "category_contrast_response",
    "compare_exact_mask_logits",
]
