"""Frozen, training-only category-calibration bank contracts.

This module deliberately owns no fitting algorithm.  It accepts already fitted
affines, freezes them behind a content-addressed identity, and applies them to a
validation partition only after proving that the declared training and
validation memberships are disjoint.

Two functionals are intentionally distinct:

``ORDINARY_OVR``
    ``a_q * (raw_q - center_q) + b_q``.

``SUBSTITUTED_TARGET_AFFINE_DENSE_EXCLUSION``
    ``a_q * (mask_raw_q - center_q) + b_q
    - logsumexp(dense_reference_-q)``.  The frozen affine changes only the mask
    target coordinate; every dense-reference competitor is left unchanged.
    Affine-on-full-log-odds, ``a_q * (raw_q - logsumexp(raw_-q)) + b_q``, is a
    distinct model and is not represented by this functional identifier.

Claim-bearing banks have exactly the ordered class axis ``0..999`` and a ready
state for both functionals at every class.  Smaller axes exist only behind the
explicit ``PARTIAL_TEST_ONLY`` mode and never acquire claim-bearing authority.

One narrowly scoped exception exists: a bank frozen with
``tolerate_unconverged=True`` carries its unready
``SUBSTITUTED_TARGET_AFFINE_DENSE_EXCLUSION`` classes as NaN columns with
explicit ``unready_class_ids`` through evaluation and stratified reduction.
``ORDINARY_OVR`` is never carried: its readers stay strict and raise exactly
as before on any unready state.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
import threading
from typing import Any, ClassVar, Mapping

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

IMAGENET_1K_CATEGORY_COUNT = 1000

# v6 setup/data lane: content hashes of the frozen identity and bank objects,
# memoized per object.  Both are frozen dataclasses whose fields are immutable
# (tuples, frozen states, bytes-backed read-only arrays), so the hash is a
# property of the object; the step setup, the terminal fit and the terminal
# captures recomputed it dozens of times per category (a 1,000-class identity
# canonicalizes its ~256k training ids on every call).  Keyed by ``id`` with
# an identity re-check and a strong reference, so a recycled id cannot hit;
# bounded, so retired objects are released.  Same pattern as the terminal
# lane's fit-bundle memo.
_CONTENT_HASH_MEMO: "OrderedDict[int, tuple[Any, str]]" = OrderedDict()
_CONTENT_HASH_MEMO_ENTRIES = 16
_CONTENT_HASH_MEMO_LOCK = threading.Lock()


def _memoized_content_hash(owner: Any, compute: Callable[[], str]) -> str:
    key = id(owner)
    with _CONTENT_HASH_MEMO_LOCK:
        cached = _CONTENT_HASH_MEMO.get(key)
        if cached is not None and cached[0] is owner:
            _CONTENT_HASH_MEMO.move_to_end(key)
            return cached[1]
    digest = compute()
    with _CONTENT_HASH_MEMO_LOCK:
        _CONTENT_HASH_MEMO[key] = (owner, digest)
        _CONTENT_HASH_MEMO.move_to_end(key)
        while len(_CONTENT_HASH_MEMO) > _CONTENT_HASH_MEMO_ENTRIES:
            _CONTENT_HASH_MEMO.popitem(last=False)
    return digest

FROZEN_CATEGORY_CALIBRATION_BANK_SCHEMA = "mechanistic_systems.frozen_category_calibration_bank.v1"
FROZEN_CATEGORY_CALIBRATION_EVALUATION_SCHEMA = (
    "mechanistic_systems.frozen_category_calibration_evaluation.v1"
)
FROZEN_CATEGORY_CALIBRATION_AUTHORITY_CEILING = (
    "training_fitted_affine_freeze_and_disjoint_validation_evaluation_only_"
    "no_fitting_selection_pruning_commit_stopping_or_promotion_authority"
)


class CalibrationFunctional(StrEnum):
    """The two non-interchangeable binary category functionals."""

    ORDINARY_OVR = "ordinary_ovr"
    SUBSTITUTED_TARGET_AFFINE_DENSE_EXCLUSION = "substituted_target_affine_dense_exclusion"


class CalibrationBankMode(StrEnum):
    """Structural scope and authority of a frozen bank."""

    CLAIM_BEARING_1000 = "claim_bearing_1000"
    PARTIAL_TEST_ONLY = "partial_test_only"


_FUNCTIONAL_ORDER = (
    CalibrationFunctional.ORDINARY_OVR,
    CalibrationFunctional.SUBSTITUTED_TARGET_AFFINE_DENSE_EXCLUSION,
)


class CalibrationContractError(ValueError):
    """A category-calibration contract failed a structural invariant."""


class CalibrationBankNotReadyError(RuntimeError):
    """A requested frozen functional has unavailable or unconverged states."""


class CalibrationBankUnreadyStatesError(CalibrationContractError):
    """A claim-bearing bank was frozen with unavailable/unconverged states.

    Carries the offending class ids per functional so a caller that refuses
    to tolerate them can still receipt exactly which categories failed.
    """

    def __init__(
        self,
        message: str,
        *,
        unready_class_ids_by_functional: Mapping[str, tuple[int, ...]],
    ) -> None:
        super().__init__(message)
        self.unready_class_ids_by_functional = {
            str(functional): tuple(int(class_id) for class_id in class_ids)
            for functional, class_ids in unready_class_ids_by_functional.items()
        }


class CalibrationPartitionOverlapError(CalibrationContractError):
    """Training membership C intersects validation membership V."""


class CalibrationBuilderFrozenError(RuntimeError):
    """A one-way calibration-bank builder was used after freezing."""


def _nonempty(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise CalibrationContractError(f"{name} must be an exact string identity")
    checked = value.strip()
    if not checked:
        raise CalibrationContractError(f"{name} must be non-empty")
    return checked


def _strict_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise CalibrationContractError(f"{name} must be an integer")
    return int(value)


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise CalibrationContractError(f"{name} must be boolean")
    return bool(value)


def _unique_nonempty_ids(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    checked = tuple(_nonempty(value, name=f"{name} item") for value in values)
    if not checked:
        raise CalibrationContractError(f"{name} must be non-empty")
    if len(set(checked)) != len(checked):
        raise CalibrationContractError(f"{name} must contain unique identities")
    return checked


def _bytes_backed_array(
    value: Any,
    *,
    dtype: str,
    name: str,
    ndim: int | None = None,
    finite: bool = False,
) -> np.ndarray:
    """Return a canonical immutable ndarray whose ultimate owner is ``bytes``.

    ``setflags(write=False)`` alone is not an immutability boundary: an owning
    ndarray can generally be made writable again.  ``np.frombuffer(bytes, ...)``
    makes the read-only property originate in the immutable backing object, so
    callers cannot re-enable writes through NumPy's flag API.
    """

    canonical_dtype = np.dtype(dtype)
    array = np.asarray(value, dtype=canonical_dtype)
    if ndim is not None and array.ndim != ndim:
        raise CalibrationContractError(f"{name} must have rank {ndim}")
    if finite and not np.isfinite(array).all():
        raise CalibrationContractError(f"{name} must contain only finite values")
    contiguous = np.ascontiguousarray(array, dtype=canonical_dtype)
    raw = contiguous.tobytes(order="C")
    frozen = np.frombuffer(raw, dtype=canonical_dtype).reshape(contiguous.shape)
    if frozen.flags.writeable:  # pragma: no cover - defensive NumPy invariant.
        raise RuntimeError(f"{name} did not freeze on an immutable byte buffer")
    return frozen


def _carried_unready_class_ids(
    value: Any,
    *,
    class_ids: np.ndarray,
    functional: CalibrationFunctional,
    name: str,
) -> tuple[int, ...]:
    """Canonicalize explicit NaN-carried class ids.

    The ids must be unique members of the ordered class axis, and only the
    substituted functional may carry any: an ordinary matrix with a carried
    column is refused, so the ordinary functional can never degrade silently.
    """

    checked = tuple(
        _strict_integer(class_id, name=f"{name} item") for class_id in tuple(value)
    )
    if not checked:
        return ()
    if len(set(checked)) != len(checked):
        raise CalibrationContractError(f"{name} must be unique")
    if checked != tuple(sorted(checked)):
        raise CalibrationContractError(f"{name} must follow the ordered class axis")
    if not np.isin(np.asarray(checked, dtype=np.int64), class_ids).all():
        raise CalibrationContractError(f"{name} lie outside the ordered class axis")
    if (
        CalibrationFunctional(functional)
        is not CalibrationFunctional.SUBSTITUTED_TARGET_AFFINE_DENSE_EXCLUSION
    ):
        raise CalibrationContractError(
            f"{name} are admitted only for the substituted functional; the "
            "ordinary functional is strict"
        )
    return checked


def _ready_columns(class_ids: np.ndarray, unready: tuple[int, ...]) -> np.ndarray:
    return ~np.isin(class_ids, np.asarray(unready, dtype=np.int64))


def _require_carried_columns(
    matrix: np.ndarray,
    *,
    ready_columns: np.ndarray,
    name: str,
    cells: np.ndarray | None = None,
) -> None:
    """Finite on ready classifier columns, NaN on every carried column cell.

    ``cells`` optionally restricts the check to the structurally defined
    cells (sensitivity/specificity); undefined cells keep their own rule.
    """

    columns = np.broadcast_to(ready_columns, matrix.shape)
    scope = np.ones(matrix.shape, dtype=np.bool_) if cells is None else cells
    if not np.isfinite(matrix[scope & columns]).all():
        raise CalibrationContractError(f"{name} must be finite on ready classes")
    if not np.isnan(matrix[scope & ~columns]).all():
        raise CalibrationContractError(
            f"{name} must be NaN exactly on the carried unready classes"
        )


@dataclass(frozen=True, slots=True)
class CategoryCalibrationIdentity:
    """Shared scientific and training identity for every state in one bank.

    ``training_member_ids`` is the exact calibration/training membership C,
    retained specifically so that validation membership V can be checked for
    disjointness.  The accompanying opaque identities bind the authoritative
    external axis, membership, order, labels, weights, and solver receipts.
    """

    mask_identity: str
    reference_identity: str
    model_identity: str
    ontology_identity: str
    ordered_class_identity: str
    training_axis_identity: str
    training_membership_identity: str
    training_order_identity: str
    training_label_identity: str
    training_weight_identity: str
    solver_identity: str
    ordered_class_ids: tuple[int, ...]
    training_member_ids: tuple[str, ...]
    mode: CalibrationBankMode = CalibrationBankMode.CLAIM_BEARING_1000

    schema: ClassVar[str] = "mechanistic_systems.category_calibration_identity.v1"

    def __post_init__(self) -> None:
        for name in (
            "mask_identity",
            "reference_identity",
            "model_identity",
            "ontology_identity",
            "ordered_class_identity",
            "training_axis_identity",
            "training_membership_identity",
            "training_order_identity",
            "training_label_identity",
            "training_weight_identity",
            "solver_identity",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))

        try:
            mode = CalibrationBankMode(self.mode)
        except ValueError as error:
            raise CalibrationContractError("unsupported calibration-bank mode") from error
        class_ids = tuple(
            _strict_integer(value, name="ordered class ID") for value in self.ordered_class_ids
        )
        if len(class_ids) < 2:
            raise CalibrationContractError(
                "a category bank requires at least two classes for dense exclusion"
            )
        if len(set(class_ids)) != len(class_ids):
            raise CalibrationContractError("ordered class IDs must be unique")
        expected_axis = tuple(range(len(class_ids)))
        if class_ids != expected_axis:
            raise CalibrationContractError(
                "ordered class IDs must be the exact contiguous axis 0..K-1"
            )
        if mode is CalibrationBankMode.CLAIM_BEARING_1000 and class_ids != tuple(
            range(IMAGENET_1K_CATEGORY_COUNT)
        ):
            raise CalibrationContractError(
                "a claim-bearing category bank requires exactly class IDs 0..999"
            )
        member_ids = _unique_nonempty_ids(
            self.training_member_ids,
            name="training membership C",
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "ordered_class_ids", class_ids)
        object.__setattr__(self, "training_member_ids", member_ids)

    @property
    def is_claim_bearing(self) -> bool:
        return self.mode is CalibrationBankMode.CLAIM_BEARING_1000

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mode": self.mode.value,
            "mask_identity": self.mask_identity,
            "reference_identity": self.reference_identity,
            "model_identity": self.model_identity,
            "ontology_identity": self.ontology_identity,
            "ordered_class_identity": self.ordered_class_identity,
            "ordered_class_ids": self.ordered_class_ids,
            "training_axis_identity": self.training_axis_identity,
            "training_membership_identity": self.training_membership_identity,
            "training_member_ids": self.training_member_ids,
            "training_order_identity": self.training_order_identity,
            "training_label_identity": self.training_label_identity,
            "training_weight_identity": self.training_weight_identity,
            "solver_identity": self.solver_identity,
        }

    @property
    def content_hash(self) -> str:
        return _memoized_content_hash(
            self,
            lambda: content_hash(
                self.to_payload(),
                prefix="category-calibration-identity-",
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenAffineCalibrationState:
    """One already fitted affine for one class and one functional."""

    state_id: str
    bank_identity_hash: str
    class_id: int
    functional: CalibrationFunctional
    scale: float
    offset: float
    input_center: float
    available: bool
    converged: bool

    schema: ClassVar[str] = "mechanistic_systems.frozen_affine_calibration_state.v1"

    def __post_init__(self) -> None:
        state_id = _nonempty(self.state_id, name="calibration state ID")
        bank_identity_hash = _nonempty(
            self.bank_identity_hash,
            name="calibration bank identity hash",
        )
        class_id = _strict_integer(self.class_id, name="calibration class ID")
        if class_id < 0:
            raise CalibrationContractError("calibration class ID must be nonnegative")
        try:
            functional = CalibrationFunctional(self.functional)
        except ValueError as error:
            raise CalibrationContractError("unsupported calibration functional") from error
        scale = float(self.scale)
        offset = float(self.offset)
        input_center = float(self.input_center)
        if not np.isfinite((scale, offset, input_center)).all():
            raise CalibrationContractError("calibration affine parameters must be finite")
        available = _strict_bool(self.available, name="calibration availability")
        converged = _strict_bool(self.converged, name="calibration convergence")
        object.__setattr__(self, "state_id", state_id)
        object.__setattr__(self, "bank_identity_hash", bank_identity_hash)
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "functional", functional)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "input_center", input_center)
        object.__setattr__(self, "available", available)
        object.__setattr__(self, "converged", converged)

    @property
    def ready(self) -> bool:
        return self.available and self.converged

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "state_id": self.state_id,
            "bank_identity_hash": self.bank_identity_hash,
            "class_id": self.class_id,
            "functional": self.functional.value,
            "scale": self.scale,
            "offset": self.offset,
            "input_center": self.input_center,
            "available": self.available,
            "converged": self.converged,
        }

    @property
    def content_hash(self) -> str:
        return content_hash(
            self.to_payload(),
            prefix="frozen-affine-calibration-state-",
        )


@dataclass(frozen=True, slots=True)
class ValidationPartition:
    """Exact validation membership V and its external axis identities."""

    axis_identity: str
    membership_identity: str
    order_identity: str
    label_identity: str
    member_ids: tuple[str, ...]

    schema: ClassVar[str] = "mechanistic_systems.category_validation_partition.v1"

    def __post_init__(self) -> None:
        for name in (
            "axis_identity",
            "membership_identity",
            "order_identity",
            "label_identity",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "member_ids",
            _unique_nonempty_ids(self.member_ids, name="validation membership V"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "axis_identity": self.axis_identity,
            "membership_identity": self.membership_identity,
            "order_identity": self.order_identity,
            "label_identity": self.label_identity,
            "member_ids": self.member_ids,
        }

    @property
    def content_hash(self) -> str:
        return content_hash(
            self.to_payload(),
            prefix="category-validation-partition-",
        )


@dataclass(frozen=True, slots=True)
class SensitivitySpecificityComponents:
    """Per-class confusion components without undefined-rate conventions."""

    true_positives: np.ndarray = field(repr=False, compare=False)
    false_negatives: np.ndarray = field(repr=False, compare=False)
    true_negatives: np.ndarray = field(repr=False, compare=False)
    false_positives: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        arrays = {}
        expected_shape: tuple[int, ...] | None = None
        for name in (
            "true_positives",
            "false_negatives",
            "true_negatives",
            "false_positives",
        ):
            frozen = _bytes_backed_array(
                getattr(self, name),
                dtype="<i8",
                name=name,
                ndim=1,
            )
            if np.any(frozen < 0):
                raise CalibrationContractError(
                    "sensitivity/specificity components must be nonnegative"
                )
            if expected_shape is None:
                expected_shape = frozen.shape
            elif frozen.shape != expected_shape:
                raise CalibrationContractError(
                    "sensitivity/specificity component axes are misaligned"
                )
            arrays[name] = frozen
        for name, frozen in arrays.items():
            object.__setattr__(self, name, frozen)

    @property
    def sensitivity_denominator(self) -> np.ndarray:
        return _bytes_backed_array(
            self.true_positives + self.false_negatives,
            dtype="<i8",
            name="sensitivity denominator",
            ndim=1,
        )

    @property
    def specificity_denominator(self) -> np.ndarray:
        return _bytes_backed_array(
            self.true_negatives + self.false_positives,
            dtype="<i8",
            name="specificity denominator",
            ndim=1,
        )

    def to_payload(self) -> dict[str, np.ndarray]:
        return {
            "true_positives": self.true_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
        }


@dataclass(frozen=True, slots=True)
class FrozenCategoryCalibrationEvaluation:
    """Immutable result of applying one frozen functional without fitting.

    ``unready_class_ids`` is empty except for the
    substituted functional of a bank frozen with ``tolerate_unconverged``.
    For every carried class, the calibrated logits, probabilities, BCE and
    binary accuracy of its classifier column are NaN and its confusion
    components are zero (no row was classified by an unconverged affine);
    every other column is finite and computed exactly as on the default path.
    """

    bank_hash: str
    validation_partition_hash: str
    source_values_hash: str
    functional: CalibrationFunctional
    class_ids: np.ndarray = field(repr=False, compare=False)
    labels: np.ndarray = field(repr=False, compare=False)
    functional_inputs: np.ndarray = field(repr=False, compare=False)
    calibrated_logits: np.ndarray = field(repr=False, compare=False)
    probabilities: np.ndarray = field(repr=False, compare=False)
    bce: np.ndarray = field(repr=False, compare=False)
    binary_accuracy: np.ndarray = field(repr=False, compare=False)
    sensitivity_specificity: SensitivitySpecificityComponents
    unready_class_ids: tuple[int, ...] = ()

    schema: ClassVar[str] = FROZEN_CATEGORY_CALIBRATION_EVALUATION_SCHEMA
    grants_fitting_authority: ClassVar[bool] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bank_hash", _nonempty(self.bank_hash, name="bank hash"))
        object.__setattr__(
            self,
            "validation_partition_hash",
            _nonempty(
                self.validation_partition_hash,
                name="validation partition hash",
            ),
        )
        object.__setattr__(
            self,
            "source_values_hash",
            _nonempty(self.source_values_hash, name="source values hash"),
        )
        try:
            functional = CalibrationFunctional(self.functional)
        except ValueError as error:
            raise CalibrationContractError("unsupported evaluation functional") from error
        object.__setattr__(self, "functional", functional)

        class_ids = _bytes_backed_array(
            self.class_ids,
            dtype="<i8",
            name="evaluation class IDs",
            ndim=1,
        )
        labels = _bytes_backed_array(
            self.labels,
            dtype="<i8",
            name="evaluation labels",
            ndim=1,
        )
        row_count = int(labels.shape[0])
        class_count = int(class_ids.shape[0])
        if row_count <= 0 or class_count < 2:
            raise CalibrationContractError("evaluation axes must be non-empty")
        unready = _carried_unready_class_ids(
            self.unready_class_ids,
            class_ids=class_ids,
            functional=functional,
            name="evaluation unready class IDs",
        )
        ready_columns = _ready_columns(class_ids, unready)
        matrix_shape = (row_count, class_count)
        matrices: dict[str, np.ndarray] = {}
        for name in (
            "functional_inputs",
            "calibrated_logits",
            "probabilities",
            "bce",
        ):
            # Functional inputs are raw readouts and never depend on a state.
            carried = bool(unready) and name != "functional_inputs"
            matrix = _bytes_backed_array(
                getattr(self, name),
                dtype="<f8",
                name=name,
                ndim=2,
                finite=not carried,
            )
            if matrix.shape != matrix_shape:
                raise CalibrationContractError(f"{name} has the wrong evaluation shape")
            if carried:
                _require_carried_columns(matrix, ready_columns=ready_columns, name=name)
            matrices[name] = matrix
        accuracy = _bytes_backed_array(
            self.binary_accuracy,
            dtype="<f8",
            name="binary accuracy",
            ndim=1,
            finite=not unready,
        )
        if accuracy.shape != (class_count,) or np.any((accuracy < 0.0) | (accuracy > 1.0)):
            raise CalibrationContractError("binary accuracy is outside [0, 1]")
        if unready:
            _require_carried_columns(
                accuracy, ready_columns=ready_columns, name="binary accuracy"
            )
        probability = matrices["probabilities"]
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise CalibrationContractError("calibrated probabilities are outside [0, 1]")
        if np.any(matrices["bce"] < 0.0):
            raise CalibrationContractError("binary cross-entropy must be nonnegative")
        if not isinstance(
            self.sensitivity_specificity,
            SensitivitySpecificityComponents,
        ):
            raise TypeError("evaluation requires typed sensitivity/specificity components")
        component_shape = self.sensitivity_specificity.true_positives.shape
        if component_shape != (class_count,):
            raise CalibrationContractError(
                "sensitivity/specificity components differ from the class axis"
            )
        total = (
            self.sensitivity_specificity.true_positives
            + self.sensitivity_specificity.false_negatives
            + self.sensitivity_specificity.true_negatives
            + self.sensitivity_specificity.false_positives
        )
        # A carried classifier classified no row: its components are zero.
        expected_total = np.where(ready_columns, row_count, 0).astype(np.int64)
        if not np.array_equal(total, expected_total):
            raise CalibrationContractError(
                "sensitivity/specificity components do not partition validation rows"
            )
        implied_accuracy = (
            self.sensitivity_specificity.true_positives
            + self.sensitivity_specificity.true_negatives
        ) / float(row_count)
        if not np.array_equal(accuracy[ready_columns], implied_accuracy[ready_columns]):
            raise CalibrationContractError("binary accuracy differs from its confusion components")

        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "labels", labels)
        for name, matrix in matrices.items():
            object.__setattr__(self, name, matrix)
        object.__setattr__(self, "binary_accuracy", accuracy)
        object.__setattr__(self, "unready_class_ids", unready)

    @property
    def mean_bce(self) -> np.ndarray:
        """Per-classifier mean BCE; NaN exactly on carried unready classes."""

        return _bytes_backed_array(
            np.mean(self.bce, axis=0, dtype=np.float64),
            dtype="<f8",
            name="mean BCE",
            ndim=1,
            finite=not self.unready_class_ids,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "bank_hash": self.bank_hash,
            "validation_partition_hash": self.validation_partition_hash,
            "source_values_hash": self.source_values_hash,
            "functional": self.functional.value,
            "class_ids": self.class_ids,
            "labels": self.labels,
            "functional_inputs": self.functional_inputs,
            "calibrated_logits": self.calibrated_logits,
            "probabilities": self.probabilities,
            "bce": self.bce,
            "binary_accuracy": self.binary_accuracy,
            "sensitivity_specificity": self.sensitivity_specificity.to_payload(),
        }
        # Present only when non-empty, so every default-path hash is unchanged.
        if self.unready_class_ids:
            payload["unready_class_ids"] = self.unready_class_ids
        return payload

    @property
    def content_hash(self) -> str:
        return content_hash(
            self.to_payload(),
            prefix="frozen-category-calibration-evaluation-",
        )

    def stratified_matrices(self) -> CategoryStratifiedEvaluationMatrices:
        """Reduce rows to true-label-stratum by classifier matrices."""

        return reduce_category_stratified_evaluation(self)


@dataclass(frozen=True, slots=True)
class CategoryStratifiedEvaluationMatrices:
    """True-label-stratum by binary-classifier evaluation matrices.

    Rows and columns use the exact same ordered ``class_ids`` axis.  A row
    selects validation examples whose multiclass label equals that class ID;
    a column selects the corresponding one-vs-rest binary classifier.

    Sensitivity is meaningful only where the row is the classifier's positive
    class (the diagonal).  Specificity is meaningful only where the row is a
    negative class (off-diagonal).  To keep every stored numeric array finite,
    structurally undefined cells are zero-filled and accompanied by explicit
    authoritative ``*_defined`` masks.  Consumers must never interpret values
    where the corresponding mask is false.

    The single exception to finiteness is explicit: the
    substituted functional of a tolerated bank lists its unready classes in
    ``unready_class_ids``, and exactly those classifier COLUMNS are NaN in
    every calibration-dependent matrix (and in the defined sensitivity and
    specificity cells).  ``mean_functional_input`` is raw and stays finite;
    undefined cells keep the canonical zero fill; ordinary matrices never
    carry a column.
    """

    evaluation_hash: str
    functional: CalibrationFunctional
    class_ids: np.ndarray = field(repr=False, compare=False)
    labels: np.ndarray = field(repr=False, compare=False)
    stratum_counts: np.ndarray = field(repr=False, compare=False)
    mean_functional_input: np.ndarray = field(repr=False, compare=False)
    mean_calibrated_logit: np.ndarray = field(repr=False, compare=False)
    mean_probability: np.ndarray = field(repr=False, compare=False)
    mean_bce: np.ndarray = field(repr=False, compare=False)
    binary_accuracy: np.ndarray = field(repr=False, compare=False)
    sensitivity_component: np.ndarray = field(repr=False, compare=False)
    sensitivity_defined: np.ndarray = field(repr=False, compare=False)
    specificity_component: np.ndarray = field(repr=False, compare=False)
    specificity_defined: np.ndarray = field(repr=False, compare=False)
    unready_class_ids: tuple[int, ...] = ()

    schema: ClassVar[str] = "mechanistic_systems.category_stratified_calibration_evaluation.v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_hash",
            _nonempty(self.evaluation_hash, name="evaluation hash"),
        )
        try:
            functional = CalibrationFunctional(self.functional)
        except ValueError as error:
            raise CalibrationContractError(
                "unsupported stratified evaluation functional"
            ) from error
        object.__setattr__(self, "functional", functional)
        class_ids = _bytes_backed_array(
            self.class_ids,
            dtype="<i8",
            name="stratified class IDs",
            ndim=1,
        )
        labels = _bytes_backed_array(
            self.labels,
            dtype="<i8",
            name="stratified source labels",
            ndim=1,
        )
        counts = _bytes_backed_array(
            self.stratum_counts,
            dtype="<i8",
            name="stratum counts",
            ndim=1,
        )
        class_count = int(class_ids.shape[0])
        if class_count < 2 or labels.size == 0 or counts.shape != (class_count,):
            raise CalibrationContractError("stratified evaluation axes are invalid")
        if np.any(counts <= 0) or int(np.sum(counts, dtype=np.int64)) != labels.size:
            raise CalibrationContractError(
                "every class stratum must be occupied and counts must cover labels"
            )
        if not np.array_equal(
            counts,
            np.asarray(
                [np.count_nonzero(labels == class_id) for class_id in class_ids],
                dtype=np.int64,
            ),
        ):
            raise CalibrationContractError("stratum counts differ from the exact source labels")
        unready = _carried_unready_class_ids(
            self.unready_class_ids,
            class_ids=class_ids,
            functional=functional,
            name="stratified unready class IDs",
        )
        ready_columns = _ready_columns(class_ids, unready)

        shape = (class_count, class_count)
        numeric: dict[str, np.ndarray] = {}
        for name in (
            "mean_functional_input",
            "mean_calibrated_logit",
            "mean_probability",
            "mean_bce",
            "binary_accuracy",
            "sensitivity_component",
            "specificity_component",
        ):
            matrix = _bytes_backed_array(
                getattr(self, name),
                dtype="<f8",
                name=name,
                ndim=2,
                finite=not unready or name == "mean_functional_input",
            )
            if matrix.shape != shape:
                raise CalibrationContractError(f"{name} differs from the stratum/classifier axes")
            numeric[name] = matrix
        masks: dict[str, np.ndarray] = {}
        for name in ("sensitivity_defined", "specificity_defined"):
            mask = _bytes_backed_array(
                getattr(self, name),
                dtype="?",
                name=name,
                ndim=2,
            )
            if mask.shape != shape:
                raise CalibrationContractError(f"{name} differs from the stratum/classifier axes")
            masks[name] = mask

        diagonal = np.eye(class_count, dtype=np.bool_)
        if not np.array_equal(masks["sensitivity_defined"], diagonal):
            raise CalibrationContractError("sensitivity must be defined on the diagonal only")
        if not np.array_equal(masks["specificity_defined"], ~diagonal):
            raise CalibrationContractError("specificity must be defined off the diagonal only")
        if unready:
            for name in (
                "mean_calibrated_logit",
                "mean_probability",
                "mean_bce",
                "binary_accuracy",
            ):
                _require_carried_columns(
                    numeric[name], ready_columns=ready_columns, name=name
                )
        for value_name, mask_name in (
            ("sensitivity_component", "sensitivity_defined"),
            ("specificity_component", "specificity_defined"),
        ):
            values = numeric[value_name]
            defined = masks[mask_name]
            if np.any(values[~defined] != 0.0):
                raise CalibrationContractError(
                    f"undefined {value_name} cells must use the canonical zero fill"
                )
            if unready:
                _require_carried_columns(
                    values,
                    ready_columns=ready_columns,
                    name=f"defined {value_name}",
                    cells=defined,
                )
            if np.any((values[defined] < 0.0) | (values[defined] > 1.0)):
                raise CalibrationContractError(f"defined {value_name} is outside [0, 1]")
        for name in ("mean_probability", "binary_accuracy"):
            values = numeric[name]
            if np.any((values < 0.0) | (values > 1.0)):
                raise CalibrationContractError(f"{name} is outside [0, 1]")
        if np.any(numeric["mean_bce"] < 0.0):
            raise CalibrationContractError("stratified mean BCE must be nonnegative")
        # NaN placement is fixed above, so NaN-equal comparison stays exact.
        if not np.array_equal(
            numeric["sensitivity_component"][diagonal],
            numeric["binary_accuracy"][diagonal],
            equal_nan=bool(unready),
        ):
            raise CalibrationContractError(
                "diagonal sensitivity differs from positive-stratum accuracy"
            )
        if not np.array_equal(
            numeric["specificity_component"][~diagonal],
            numeric["binary_accuracy"][~diagonal],
            equal_nan=bool(unready),
        ):
            raise CalibrationContractError(
                "off-diagonal specificity differs from negative-stratum accuracy"
            )

        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "stratum_counts", counts)
        for name, matrix in numeric.items():
            object.__setattr__(self, name, matrix)
        for name, mask in masks.items():
            object.__setattr__(self, name, mask)
        object.__setattr__(self, "unready_class_ids", unready)

    @property
    def ready_class_mask(self) -> np.ndarray:
        """Boolean classifier-column mask: False exactly on carried classes."""

        return _bytes_backed_array(
            _ready_columns(self.class_ids, self.unready_class_ids),
            dtype="?",
            name="ready class mask",
            ndim=1,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "evaluation_hash": self.evaluation_hash,
            "functional": self.functional.value,
            "class_ids": self.class_ids,
            "labels": self.labels,
            "stratum_counts": self.stratum_counts,
            "mean_functional_input": self.mean_functional_input,
            "mean_calibrated_logit": self.mean_calibrated_logit,
            "mean_probability": self.mean_probability,
            "mean_bce": self.mean_bce,
            "binary_accuracy": self.binary_accuracy,
            "sensitivity_component": self.sensitivity_component,
            "sensitivity_defined": self.sensitivity_defined,
            "specificity_component": self.specificity_component,
            "specificity_defined": self.specificity_defined,
        }
        # Present only when non-empty, so every default-path hash is unchanged.
        if self.unready_class_ids:
            payload["unready_class_ids"] = self.unready_class_ids
        return payload

    @property
    def content_hash(self) -> str:
        return content_hash(
            self.to_payload(),
            prefix="category-stratified-calibration-evaluation-",
        )


@dataclass(frozen=True, slots=True)
class FrozenCategoryCalibrationBank:
    """Content-addressed immutable bank produced by the one-way builder."""

    identity: CategoryCalibrationIdentity
    states: tuple[FrozenAffineCalibrationState, ...]
    # Terminal category structure ON: a claim-bearing bank may be frozen
    # with unready states when the builder tolerates them.  The unready class
    # ids are recorded per functional, and only ``affine_arrays(...,
    # allow_unready=True)`` can read such a functional -- as NaN affines for
    # the unready classes -- so no consumer can apply an unconverged affine
    # by accident.  The content hash is unchanged: the states themselves
    # already carry ``available``/``converged``.  The
    # matrix readers go through ``evaluation_affine_arrays``, which carries
    # NaN columns for the SUBSTITUTED functional only; ORDINARY stays strict.
    tolerate_unconverged: bool = False
    # Bank-level guard: the functionals whose unready
    # states a tolerant bank may hold. A claim-bearing bank that declares
    # them refuses at freeze, with CalibrationBankUnreadyStatesError, any
    # unready state of a functional outside the set, so "every ordinary state
    # ready" is a bank invariant and not only the provider's decision. None
    # keeps the unrestricted meaning of ``tolerate_unconverged`` (every
    # functional). Not part of the payload: it constrains which states may
    # be frozen and never changes a state, so the content hash is unchanged.
    tolerate_unconverged_functionals: tuple[str, ...] | None = field(
        default=None, compare=False
    )
    _class_ids: np.ndarray = field(init=False, repr=False, compare=False)
    _scales: np.ndarray = field(init=False, repr=False, compare=False)
    _offsets: np.ndarray = field(init=False, repr=False, compare=False)
    _input_centers: np.ndarray = field(init=False, repr=False, compare=False)
    _availability: np.ndarray = field(init=False, repr=False, compare=False)
    _convergence: np.ndarray = field(init=False, repr=False, compare=False)
    _unready_class_ids_by_functional: Mapping[str, tuple[int, ...]] = field(
        init=False, repr=False, compare=False
    )

    schema: ClassVar[str] = FROZEN_CATEGORY_CALIBRATION_BANK_SCHEMA
    authority_ceiling: ClassVar[str] = FROZEN_CATEGORY_CALIBRATION_AUTHORITY_CEILING
    grants_fitting_authority: ClassVar[bool] = False
    grants_selection_authority: ClassVar[bool] = False
    grants_pruning_commit_authority: ClassVar[bool] = False
    grants_promotion_authority: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CategoryCalibrationIdentity):
            raise TypeError("frozen category bank requires a typed identity")
        supplied = tuple(self.states)
        if any(not isinstance(state, FrozenAffineCalibrationState) for state in supplied):
            raise TypeError("frozen category bank states have the wrong type")
        expected_keys = {
            (functional, class_id)
            for functional in _FUNCTIONAL_ORDER
            for class_id in self.identity.ordered_class_ids
        }
        by_key: dict[tuple[CalibrationFunctional, int], FrozenAffineCalibrationState] = {}
        state_ids: set[str] = set()
        identity_hash = self.identity.content_hash
        for state in supplied:
            if state.bank_identity_hash != identity_hash:
                raise CalibrationContractError(
                    "calibration state is bound to a different bank identity"
                )
            key = (state.functional, state.class_id)
            if key in by_key:
                raise CalibrationContractError(
                    "calibration bank has a duplicate class-functional state"
                )
            if state.state_id in state_ids:
                raise CalibrationContractError(
                    "calibration state IDs must be globally unique within a bank"
                )
            by_key[key] = state
            state_ids.add(state.state_id)
        observed_keys = set(by_key)
        if observed_keys != expected_keys:
            missing = sorted(
                (functional.value, class_id)
                for functional, class_id in expected_keys - observed_keys
            )
            unexpected = sorted(
                (functional.value, class_id)
                for functional, class_id in observed_keys - expected_keys
            )
            raise CalibrationContractError(
                "calibration bank must contain exactly one state per functional and "
                f"ordered class; missing={missing[:8]!r}, unexpected={unexpected[:8]!r}"
            )
        canonical_states = tuple(
            by_key[(functional, class_id)]
            for functional in _FUNCTIONAL_ORDER
            for class_id in self.identity.ordered_class_ids
        )
        tolerate_unconverged = _strict_bool(
            self.tolerate_unconverged, name="calibration unconverged tolerance"
        )
        unready_by_functional = {
            functional.value: tuple(
                int(class_id)
                for class_id in self.identity.ordered_class_ids
                if not by_key[(functional, class_id)].ready
            )
            for functional in _FUNCTIONAL_ORDER
        }
        if (
            self.identity.is_claim_bearing
            and any(unready_by_functional.values())
            and not tolerate_unconverged
        ):
            raise CalibrationBankUnreadyStatesError(
                "every claim-bearing calibration state must be available and converged",
                unready_class_ids_by_functional=unready_by_functional,
            )
        tolerated_functionals: tuple[str, ...] | None = None
        if self.tolerate_unconverged_functionals is not None:
            if not tolerate_unconverged:
                raise CalibrationContractError(
                    "tolerated calibration functionals require tolerate_unconverged"
                )
            declared = {
                CalibrationFunctional(value).value
                for value in self.tolerate_unconverged_functionals
            }
            if not declared:
                raise CalibrationContractError(
                    "tolerated calibration functionals must be non-empty"
                )
            tolerated_functionals = tuple(
                functional.value
                for functional in _FUNCTIONAL_ORDER
                if functional.value in declared
            )
            if self.identity.is_claim_bearing and any(
                class_ids and functional not in tolerated_functionals
                for functional, class_ids in unready_by_functional.items()
            ):
                raise CalibrationBankUnreadyStatesError(
                    "every claim-bearing calibration state outside the tolerated "
                    f"functionals {list(tolerated_functionals)!r} must be available "
                    "and converged",
                    unready_class_ids_by_functional=unready_by_functional,
                )

        width = len(self.identity.ordered_class_ids)
        scales = np.empty((len(_FUNCTIONAL_ORDER), width), dtype=np.float64)
        offsets = np.empty_like(scales)
        input_centers = np.empty_like(scales)
        availability = np.empty(scales.shape, dtype=np.bool_)
        convergence = np.empty(scales.shape, dtype=np.bool_)
        for functional_index, functional in enumerate(_FUNCTIONAL_ORDER):
            for class_index, class_id in enumerate(self.identity.ordered_class_ids):
                state = by_key[(functional, class_id)]
                scales[functional_index, class_index] = state.scale
                offsets[functional_index, class_index] = state.offset
                input_centers[functional_index, class_index] = state.input_center
                availability[functional_index, class_index] = state.available
                convergence[functional_index, class_index] = state.converged
        if not (
            np.isfinite(scales).all()
            and np.isfinite(offsets).all()
            and np.isfinite(input_centers).all()
        ):
            raise CalibrationContractError("calibration bank affine arrays must be finite")

        object.__setattr__(self, "states", canonical_states)
        object.__setattr__(self, "tolerate_unconverged", tolerate_unconverged)
        object.__setattr__(
            self, "tolerate_unconverged_functionals", tolerated_functionals
        )
        object.__setattr__(
            self, "_unready_class_ids_by_functional", unready_by_functional
        )
        object.__setattr__(
            self,
            "_class_ids",
            _bytes_backed_array(
                self.identity.ordered_class_ids,
                dtype="<i8",
                name="bank class IDs",
                ndim=1,
            ),
        )
        object.__setattr__(
            self,
            "_scales",
            _bytes_backed_array(
                scales,
                dtype="<f8",
                name="bank scales",
                ndim=2,
                finite=True,
            ),
        )
        object.__setattr__(
            self,
            "_offsets",
            _bytes_backed_array(
                offsets,
                dtype="<f8",
                name="bank offsets",
                ndim=2,
                finite=True,
            ),
        )
        object.__setattr__(
            self,
            "_input_centers",
            _bytes_backed_array(
                input_centers,
                dtype="<f8",
                name="bank input centers",
                ndim=2,
                finite=True,
            ),
        )
        object.__setattr__(
            self,
            "_availability",
            _bytes_backed_array(
                availability,
                dtype="?",
                name="bank availability",
                ndim=2,
            ),
        )
        object.__setattr__(
            self,
            "_convergence",
            _bytes_backed_array(
                convergence,
                dtype="?",
                name="bank convergence",
                ndim=2,
            ),
        )

    @property
    def is_claim_bearing(self) -> bool:
        return self.identity.is_claim_bearing

    @property
    def class_ids(self) -> np.ndarray:
        return self._class_ids

    @property
    def scales(self) -> np.ndarray:
        return self._scales

    @property
    def offsets(self) -> np.ndarray:
        return self._offsets

    @property
    def input_centers(self) -> np.ndarray:
        return self._input_centers

    @property
    def availability(self) -> np.ndarray:
        return self._availability

    @property
    def convergence(self) -> np.ndarray:
        return self._convergence

    @property
    def unready_class_ids_by_functional(self) -> Mapping[str, tuple[int, ...]]:
        """Class ids whose state is unavailable or unconverged, per functional."""

        return dict(self._unready_class_ids_by_functional)

    def unready_class_ids(self, functional: CalibrationFunctional) -> tuple[int, ...]:
        checked = CalibrationFunctional(functional)
        return tuple(self._unready_class_ids_by_functional[checked.value])

    def _functional_index(self, functional: CalibrationFunctional) -> int:
        checked = CalibrationFunctional(functional)
        return _FUNCTIONAL_ORDER.index(checked)

    def require_ready(self, functional: CalibrationFunctional) -> None:
        """Fail closed unless every state for ``functional`` can be applied."""

        checked = CalibrationFunctional(functional)
        index = self._functional_index(checked)
        ready = self._availability[index] & self._convergence[index]
        if not bool(np.all(ready)):
            failed_classes = self._class_ids[~ready].tolist()
            raise CalibrationBankNotReadyError(
                f"frozen {checked.value} states are unavailable or unconverged for "
                f"classes {failed_classes[:16]!r}"
            )

    def affine_arrays(
        self,
        functional: CalibrationFunctional,
        *,
        allow_unready: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return immutable scale, offset, and input-center vectors.

        With ``allow_unready=True`` the ready check is skipped and every
        unavailable-or-unconverged class reads as NaN scale/offset/center, so
        a consumer that masks NaN columns (the ``fitted_mask`` path, the
        sidecar's ``calibration_converged`` flag) degrades per category
        instead of failing the whole bank.  Ready classes are byte-identical
        to the default path.
        """

        checked = CalibrationFunctional(functional)
        index = self._functional_index(checked)
        if not allow_unready:
            self.require_ready(checked)
            return (
                self._scales[index],
                self._offsets[index],
                self._input_centers[index],
            )
        ready = self._availability[index] & self._convergence[index]
        arrays = tuple(
            np.where(ready, source[index], np.nan)
            for source in (self._scales, self._offsets, self._input_centers)
        )
        for array in arrays:
            array.setflags(write=False)
        return arrays  # type: ignore[return-value]

    def carried_unready_class_ids(
        self,
        functional: CalibrationFunctional,
    ) -> tuple[int, ...]:
        """Class ids a matrix reader must carry as NaN columns.

        Non-empty only for ``SUBSTITUTED_TARGET_AFFINE_DENSE_EXCLUSION`` on a
        bank frozen with ``tolerate_unconverged`` (and, when the bank declares
        ``tolerate_unconverged_functionals``, only if the substituted
        functional is among them).  ``ORDINARY_OVR`` always returns ``()``, so
        an ordinary reader applies the strict ready check and raises exactly
        as before on any unready state.
        """

        checked = CalibrationFunctional(functional)
        if (
            checked is not CalibrationFunctional.SUBSTITUTED_TARGET_AFFINE_DENSE_EXCLUSION
            or not self.tolerate_unconverged
            or (
                self.tolerate_unconverged_functionals is not None
                and checked.value not in self.tolerate_unconverged_functionals
            )
        ):
            return ()
        return self.unready_class_ids(checked)

    def evaluation_affine_arrays(
        self,
        functional: CalibrationFunctional,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
        """Affine vectors plus the class ids their consumer must carry as NaN.

        With no carried ids this is exactly the strict ``affine_arrays`` read
        (raising :class:`CalibrationBankNotReadyError` on any unready class).
        Otherwise the carried classes read as NaN and are returned explicitly,
        and every other class is byte-identical to the strict read.
        """

        checked = CalibrationFunctional(functional)
        carried = self.carried_unready_class_ids(checked)
        if not carried:
            return (*self.affine_arrays(checked), ())
        return (*self.affine_arrays(checked, allow_unready=True), carried)

    def validate_partition(self, validation: ValidationPartition) -> None:
        """Validate V only after this bank has completed its freeze lifecycle."""

        if not isinstance(validation, ValidationPartition):
            raise TypeError("validation requires a typed ValidationPartition")
        overlap = set(self.identity.training_member_ids).intersection(validation.member_ids)
        if overlap:
            raise CalibrationPartitionOverlapError(
                "calibration training membership C and validation membership V "
                f"overlap at {sorted(overlap)[:8]!r}"
            )

    def evaluate_ordinary_ovr(
        self,
        *,
        raw_query_logits: Any,
        labels: Any,
        validation: ValidationPartition,
    ) -> FrozenCategoryCalibrationEvaluation:
        return evaluate_frozen_ordinary_ovr(
            self,
            raw_query_logits=raw_query_logits,
            labels=labels,
            validation=validation,
        )

    def evaluate_substituted_target_affine_dense_exclusion(
        self,
        *,
        mask_raw_logits: Any,
        dense_reference_logits: Any,
        labels: Any,
        validation: ValidationPartition,
    ) -> FrozenCategoryCalibrationEvaluation:
        return evaluate_frozen_substituted_target_affine_dense_exclusion(
            self,
            mask_raw_logits=mask_raw_logits,
            dense_reference_logits=dense_reference_logits,
            labels=labels,
            validation=validation,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "authority_ceiling": self.authority_ceiling,
            "identity_hash": self.identity.content_hash,
            "state_hashes": tuple(state.content_hash for state in self.states),
            "class_ids": self._class_ids,
            "scales": self._scales,
            "offsets": self._offsets,
            "input_centers": self._input_centers,
            "availability": self._availability,
            "convergence": self._convergence,
        }

    @property
    def content_hash(self) -> str:
        return _memoized_content_hash(
            self,
            lambda: content_hash(
                self.to_payload(),
                prefix="frozen-category-calibration-bank-",
            ),
        )


class CategoryCalibrationBankBuilder:
    """One-way assembly lifecycle for prefit states; never a fitting API."""

    __slots__ = (
        "_frozen",
        "_identity",
        "_identity_hash",
        "_states",
        "_state_ids",
        "_tolerate_unconverged",
        "_tolerate_unconverged_functionals",
    )

    def __init__(
        self,
        identity: CategoryCalibrationIdentity,
        *,
        tolerate_unconverged: bool = False,
        tolerate_unconverged_functionals: (
            Iterable[CalibrationFunctional | str] | None
        ) = None,
    ) -> None:
        if not isinstance(identity, CategoryCalibrationIdentity):
            raise TypeError("calibration-bank builder requires a typed identity")
        self._identity = identity
        self._identity_hash = identity.content_hash
        self._states: dict[tuple[CalibrationFunctional, int], FrozenAffineCalibrationState] = {}
        self._state_ids: set[str] = set()
        self._frozen = False
        self._tolerate_unconverged = _strict_bool(
            tolerate_unconverged, name="calibration unconverged tolerance"
        )
        # Optional per-functional scope of the tolerance;
        # validated here and enforced again by the frozen bank.
        self._tolerate_unconverged_functionals: tuple[str, ...] | None = None
        if tolerate_unconverged_functionals is not None:
            if not self._tolerate_unconverged:
                raise CalibrationContractError(
                    "tolerated calibration functionals require tolerate_unconverged"
                )
            self._tolerate_unconverged_functionals = tuple(
                CalibrationFunctional(value).value
                for value in tolerate_unconverged_functionals
            )
            if not self._tolerate_unconverged_functionals:
                raise CalibrationContractError(
                    "tolerated calibration functionals must be non-empty"
                )

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def tolerate_unconverged(self) -> bool:
        return self._tolerate_unconverged

    @property
    def tolerate_unconverged_functionals(self) -> tuple[str, ...] | None:
        return self._tolerate_unconverged_functionals

    @property
    def identity(self) -> CategoryCalibrationIdentity:
        return self._identity

    def _require_open(self) -> None:
        if self._frozen:
            raise CalibrationBuilderFrozenError("calibration-bank builder is already frozen")

    def add_state(
        self,
        state: FrozenAffineCalibrationState,
    ) -> CategoryCalibrationBankBuilder:
        self._require_open()
        if not isinstance(state, FrozenAffineCalibrationState):
            raise TypeError("calibration-bank builder requires typed frozen states")
        if state.bank_identity_hash != self._identity_hash:
            raise CalibrationContractError(
                "calibration state is bound to a different bank identity"
            )
        key = (state.functional, state.class_id)
        if key in self._states:
            raise CalibrationContractError(
                "calibration builder already contains this class-functional state"
            )
        if state.state_id in self._state_ids:
            raise CalibrationContractError(
                "calibration state IDs must be globally unique within a bank"
            )
        self._states[key] = state
        self._state_ids.add(state.state_id)
        return self

    def extend(
        self,
        states: Iterable[FrozenAffineCalibrationState],
    ) -> CategoryCalibrationBankBuilder:
        for state in states:
            self.add_state(state)
        return self

    def freeze(self) -> FrozenCategoryCalibrationBank:
        """Validate completeness, materialize immutable arrays, and seal forever.

        A claim-bearing bank with unready states is refused unless the builder
        was constructed with ``tolerate_unconverged=True``, in which case the
        bank records ``unready_class_ids_by_functional`` instead of raising.
        With ``tolerate_unconverged_functionals`` the tolerance covers only
        those functionals; an unready state of any other one still refuses.
        """

        self._require_open()
        bank = FrozenCategoryCalibrationBank(
            identity=self._identity,
            states=tuple(self._states.values()),
            tolerate_unconverged=self._tolerate_unconverged,
            tolerate_unconverged_functionals=self._tolerate_unconverged_functionals,
        )
        self._frozen = True
        self._states.clear()
        self._state_ids.clear()
        return bank


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise CalibrationContractError(f"{name} must be a finite rank-2 array")
    return matrix


def _integer_labels(value: Any) -> np.ndarray:
    labels = np.asarray(value)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or labels.dtype.kind == "b":
        raise CalibrationContractError("validation labels must be a rank-1 integer array")
    return np.asarray(labels, dtype=np.int64)


def _stable_dense_exclusion_lse(dense_logits: np.ndarray) -> np.ndarray:
    """Compute every class-exclusion LSE in O(N*K), including dominant maxima.

    The unique/deterministically selected row maximum is recomputed against a
    fresh shift after exclusion.  This avoids the catastrophic cancellation in
    ``log(exp(logsumexp(row)) - exp(row[q]))`` when ``q`` dominates the row.
    """

    values = _finite_matrix(dense_logits, name="dense logits")
    row_count, width = values.shape
    if row_count <= 0 or width < 2:
        raise CalibrationContractError(
            "dense exclusion requires non-empty rows and at least two classes"
        )
    row_index = np.arange(row_count)
    maximum = np.max(values, axis=1)
    argmax = np.argmax(values, axis=1)
    shifted = np.exp(values - maximum[:, None])
    shifted_sum = np.sum(shifted, axis=1, dtype=np.float64)
    remaining = shifted_sum[:, None] - shifted
    # The selected maximum is the only subtraction that can catastrophically
    # cancel.  Install a harmless placeholder before log; it is replaced below.
    remaining[row_index, argmax] = 1.0
    if np.any(remaining <= 0.0) or not np.isfinite(remaining).all():
        raise CalibrationContractError("dense exclusion produced an invalid shifted competitor sum")
    exclusion_lse = maximum[:, None] + np.log(remaining)

    masked = np.array(values, copy=True, order="C")
    masked[row_index, argmax] = -np.inf
    second_maximum = np.max(masked, axis=1)
    with np.errstate(under="ignore"):
        second_shifted_sum = np.sum(
            np.exp(masked - second_maximum[:, None]),
            axis=1,
            dtype=np.float64,
        )
    second_lse = second_maximum + np.log(second_shifted_sum)
    exclusion_lse[row_index, argmax] = second_lse
    if not np.isfinite(exclusion_lse).all():
        raise CalibrationContractError("dense exclusion logsumexp produced non-finite values")
    return exclusion_lse


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    probabilities = np.empty_like(logits, dtype=np.float64)
    nonnegative = logits >= 0.0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exponent = np.exp(logits[~nonnegative])
    probabilities[~nonnegative] = exponent / (1.0 + exponent)
    return probabilities


def _prepare_evaluation(
    bank: FrozenCategoryCalibrationBank,
    *,
    source_values: Any,
    labels: Any,
    validation: ValidationPartition,
    source_name: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if not isinstance(bank, FrozenCategoryCalibrationBank):
        raise TypeError("evaluation requires a frozen category calibration bank")
    bank.validate_partition(validation)
    values = _finite_matrix(source_values, name=source_name)
    label_array = _integer_labels(labels)
    row_count = len(validation.member_ids)
    expected_shape = (row_count, len(bank.identity.ordered_class_ids))
    if values.shape != expected_shape:
        raise CalibrationContractError(
            f"{source_name} shape {values.shape!r} differs from validation/class "
            f"axes {expected_shape!r}"
        )
    if label_array.shape != (row_count,):
        raise CalibrationContractError("validation labels differ from validation membership order")
    if not np.isin(label_array, bank.class_ids).all():
        raise CalibrationContractError(
            "validation labels contain IDs outside the ordered class axis"
        )
    source_hash = content_hash(
        {
            "source_name": source_name,
            "values": values,
            "validation_partition_hash": validation.content_hash,
        },
        prefix="category-calibration-source-values-",
    )
    return values, label_array, source_hash


def _evaluate_functional_inputs(
    bank: FrozenCategoryCalibrationBank,
    *,
    functional: CalibrationFunctional,
    functional_inputs: np.ndarray,
    labels: np.ndarray,
    validation: ValidationPartition,
    source_values_hash: str,
    subtracted_logit_offsets: np.ndarray | None = None,
) -> FrozenCategoryCalibrationEvaluation:
    # ``unready`` is empty (strict read, raising exactly as before) except for
    # the substituted functional of a tolerated bank.
    scale, offset, input_center, unready = bank.evaluation_affine_arrays(functional)
    fixed_offsets = (
        np.zeros_like(functional_inputs, dtype=np.float64)
        if subtracted_logit_offsets is None
        else _finite_matrix(
            subtracted_logit_offsets,
            name="subtracted fixed logit offsets",
        )
    )
    if fixed_offsets.shape != functional_inputs.shape:
        raise CalibrationContractError(
            "subtracted fixed logit offsets differ from the functional-input shape"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        calibrated_logits = (
            scale[None, :] * (functional_inputs - input_center[None, :])
            + offset[None, :]
            - fixed_offsets
        )
    carried: np.ndarray | None = None
    if unready:
        carried = ~_ready_columns(bank.class_ids, unready)
        # Per-column finiteness: only a READY column may refuse, and no
        # census-sized boolean-indexed copy is made.
        if not np.isfinite(calibrated_logits).all(axis=0)[~carried].all():
            raise CalibrationContractError(
                "frozen affine application produced non-finite calibrated logits"
            )
        # Carried columns are never evaluated.  A finite placeholder keeps NaN
        # out of the arithmetic below; every operation is elementwise or a
        # per-column reduction, so ready columns are byte-identical to the
        # strict path, and each carried column is overwritten with NaN (zero
        # confusion counts) before anything is stored.
        calibrated_logits[:, carried] = 0.0
    elif not np.isfinite(calibrated_logits).all():
        raise CalibrationContractError(
            "frozen affine application produced non-finite calibrated logits"
        )
    probabilities = _stable_sigmoid(calibrated_logits)
    targets = labels[:, None] == bank.class_ids[None, :]
    target_float = targets.astype(np.float64, copy=False)
    # max(x, 0) - x*y + log1p(exp(-abs(x))) is the stable Bernoulli BCE.
    bce = (
        np.maximum(calibrated_logits, 0.0)
        - calibrated_logits * target_float
        + np.log1p(np.exp(-np.abs(calibrated_logits)))
    )
    predicted_positive = calibrated_logits >= 0.0
    true_positives = np.sum(predicted_positive & targets, axis=0, dtype=np.int64)
    false_negatives = np.sum(~predicted_positive & targets, axis=0, dtype=np.int64)
    true_negatives = np.sum(~predicted_positive & ~targets, axis=0, dtype=np.int64)
    false_positives = np.sum(predicted_positive & ~targets, axis=0, dtype=np.int64)
    if carried is not None:
        calibrated_logits[:, carried] = np.nan
        probabilities[:, carried] = np.nan
        bce[:, carried] = np.nan
        for component in (
            true_positives,
            false_negatives,
            true_negatives,
            false_positives,
        ):
            component[carried] = 0
    components = SensitivitySpecificityComponents(
        true_positives=true_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        false_positives=false_positives,
    )
    binary_accuracy = (true_positives + true_negatives) / float(labels.shape[0])
    if carried is not None:
        binary_accuracy[carried] = np.nan
    return FrozenCategoryCalibrationEvaluation(
        bank_hash=bank.content_hash,
        validation_partition_hash=validation.content_hash,
        source_values_hash=source_values_hash,
        functional=functional,
        class_ids=bank.class_ids,
        labels=labels,
        functional_inputs=functional_inputs,
        calibrated_logits=calibrated_logits,
        probabilities=probabilities,
        bce=bce,
        binary_accuracy=binary_accuracy,
        sensitivity_specificity=components,
        unready_class_ids=unready,
    )


def evaluate_frozen_ordinary_ovr(
    bank: FrozenCategoryCalibrationBank,
    *,
    raw_query_logits: Any,
    labels: Any,
    validation: ValidationPartition,
) -> FrozenCategoryCalibrationEvaluation:
    """Apply only the frozen ordinary one-vs-rest affines."""

    values, label_array, source_hash = _prepare_evaluation(
        bank,
        source_values=raw_query_logits,
        labels=labels,
        validation=validation,
        source_name="ordinary OVR raw query logits",
    )
    return _evaluate_functional_inputs(
        bank,
        functional=CalibrationFunctional.ORDINARY_OVR,
        functional_inputs=values,
        labels=label_array,
        validation=validation,
        source_values_hash=source_hash,
    )


def evaluate_frozen_substituted_target_affine_dense_exclusion(
    bank: FrozenCategoryCalibrationBank,
    *,
    mask_raw_logits: Any,
    dense_reference_logits: Any,
    labels: Any,
    validation: ValidationPartition,
) -> FrozenCategoryCalibrationEvaluation:
    """Affine each mask target, then subtract its fixed dense competitor LSE.

    The dense-reference matrix supplies competitors only.  Its target column is
    excluded independently for each binary classifier, and no dense-reference
    coordinate is calibrated or otherwise changed.
    """

    mask_values, label_array, _ = _prepare_evaluation(
        bank,
        source_values=mask_raw_logits,
        labels=labels,
        validation=validation,
        source_name="substituted mask raw logits",
    )
    dense_reference = _finite_matrix(
        dense_reference_logits,
        name="substituted dense-reference logits",
    )
    if dense_reference.shape != mask_values.shape:
        raise CalibrationContractError(
            "dense-reference logits differ from the validation/class axes"
        )
    competitor_logsumexp = _stable_dense_exclusion_lse(dense_reference)
    source_hash = content_hash(
        {
            "source_name": "substituted target-affine dense exclusion",
            "mask_raw_logits": mask_values,
            "dense_reference_logits": dense_reference,
            "validation_partition_hash": validation.content_hash,
        },
        prefix="category-calibration-source-values-",
    )
    return _evaluate_functional_inputs(
        bank,
        functional=CalibrationFunctional.SUBSTITUTED_TARGET_AFFINE_DENSE_EXCLUSION,
        functional_inputs=mask_values,
        labels=label_array,
        validation=validation,
        source_values_hash=source_hash,
        subtracted_logit_offsets=competitor_logsumexp,
    )


def reduce_category_stratified_evaluation(
    evaluation: FrozenCategoryCalibrationEvaluation,
) -> CategoryStratifiedEvaluationMatrices:
    """Reduce a row-level result to true-label-stratum by classifier matrices.

    Every declared class must occur at least once.  Silently emitting an empty
    class row would make its diagonal sensitivity and off-diagonal specificity
    denominators undefined, so incomplete validation axes fail closed.

    Carried unready classifier columns are NaN in every
    calibration-dependent matrix and keep their ids; they are never reduced.
    """

    if not isinstance(evaluation, FrozenCategoryCalibrationEvaluation):
        raise TypeError("stratified reduction requires a frozen category evaluation")
    class_ids = evaluation.class_ids
    labels = evaluation.labels
    class_count = int(class_ids.shape[0])
    shape = (class_count, class_count)
    counts = np.asarray(
        [np.count_nonzero(labels == class_id) for class_id in class_ids],
        dtype=np.int64,
    )
    if np.any(counts <= 0):
        missing = class_ids[counts == 0].tolist()
        raise CalibrationContractError(
            "stratified calibration evaluation requires every class to be present; "
            f"missing class IDs {missing[:16]!r}"
        )

    mean_functional_input = np.empty(shape, dtype=np.float64)
    mean_calibrated_logit = np.empty(shape, dtype=np.float64)
    mean_probability = np.empty(shape, dtype=np.float64)
    mean_bce = np.empty(shape, dtype=np.float64)
    binary_accuracy = np.empty(shape, dtype=np.float64)
    targets = labels[:, None] == class_ids[None, :]
    predictions = evaluation.calibrated_logits >= 0.0
    for stratum_index, class_id in enumerate(class_ids):
        selected = labels == class_id
        mean_functional_input[stratum_index] = np.mean(
            evaluation.functional_inputs[selected],
            axis=0,
            dtype=np.float64,
        )
        mean_calibrated_logit[stratum_index] = np.mean(
            evaluation.calibrated_logits[selected],
            axis=0,
            dtype=np.float64,
        )
        mean_probability[stratum_index] = np.mean(
            evaluation.probabilities[selected],
            axis=0,
            dtype=np.float64,
        )
        mean_bce[stratum_index] = np.mean(
            evaluation.bce[selected],
            axis=0,
            dtype=np.float64,
        )
        binary_accuracy[stratum_index] = np.mean(
            predictions[selected] == targets[selected],
            axis=0,
            dtype=np.float64,
        )

    unready = evaluation.unready_class_ids
    if unready:
        carried = ~_ready_columns(class_ids, unready)
        for matrix in (
            mean_calibrated_logit,
            mean_probability,
            mean_bce,
            binary_accuracy,
        ):
            matrix[:, carried] = np.nan

    sensitivity_defined = np.eye(class_count, dtype=np.bool_)
    specificity_defined = ~sensitivity_defined
    sensitivity_component = np.zeros(shape, dtype=np.float64)
    specificity_component = np.zeros(shape, dtype=np.float64)
    sensitivity_component[sensitivity_defined] = binary_accuracy[sensitivity_defined]
    specificity_component[specificity_defined] = binary_accuracy[specificity_defined]
    return CategoryStratifiedEvaluationMatrices(
        evaluation_hash=evaluation.content_hash,
        functional=evaluation.functional,
        class_ids=class_ids,
        labels=labels,
        stratum_counts=counts,
        mean_functional_input=mean_functional_input,
        mean_calibrated_logit=mean_calibrated_logit,
        mean_probability=mean_probability,
        mean_bce=mean_bce,
        binary_accuracy=binary_accuracy,
        sensitivity_component=sensitivity_component,
        sensitivity_defined=sensitivity_defined,
        specificity_component=specificity_component,
        specificity_defined=specificity_defined,
        unready_class_ids=unready,
    )


__all__ = [
    "FROZEN_CATEGORY_CALIBRATION_AUTHORITY_CEILING",
    "FROZEN_CATEGORY_CALIBRATION_BANK_SCHEMA",
    "FROZEN_CATEGORY_CALIBRATION_EVALUATION_SCHEMA",
    "IMAGENET_1K_CATEGORY_COUNT",
    "CalibrationBankMode",
    "CalibrationBankNotReadyError",
    "CalibrationBankUnreadyStatesError",
    "CalibrationBuilderFrozenError",
    "CalibrationContractError",
    "CalibrationFunctional",
    "CalibrationPartitionOverlapError",
    "CategoryStratifiedEvaluationMatrices",
    "CategoryCalibrationBankBuilder",
    "CategoryCalibrationIdentity",
    "FrozenAffineCalibrationState",
    "FrozenCategoryCalibrationBank",
    "FrozenCategoryCalibrationEvaluation",
    "SensitivitySpecificityComponents",
    "ValidationPartition",
    "evaluate_frozen_ordinary_ovr",
    "evaluate_frozen_substituted_target_affine_dense_exclusion",
    "reduce_category_stratified_evaluation",
]
