"""Canonical complete-ImageNet population identity for modern v1.

This module certifies the scientific population pair without opening an image
or reading a tensor bank.  The two :class:`ContextAxis` objects already bind
every ordered context coordinate; optional bank receipts add the identity of
the certified execution bytes while deliberately excluding replica paths.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts.hashing import stable_digest
from mechanistic_systems.circuits.irg.modern_v1_alexnet_train_bank import (
    certified_uint8_bank_payload_identity_hash,
)
from mechanistic_systems.contracts import StudySpec
from mechanistic_systems.contracts.content_address import canonical_json, content_hash
from mechanistic_systems.measurement.contexts import (
    ContextAxis,
    ImagePreprocessingSpec,
    PreprocessedUint8BankReplicaReceipt,
    ordered_context_ids_hash,
)


COMPLETE_IMAGENET_POPULATION_PAIR_SCHEMA = (
    "modern_v1_complete_imagenet1k_population_pair_v1"
)
IMAGENET1K_DATASET_ID = "imagenet1k_ilsvrc2012"
IMAGENET1K_TRAIN_SPLIT = "train"
IMAGENET1K_VALIDATION_SPLIT = "val"
IMAGENET1K_TRAIN_ROLE = "statistical_reservoir"
IMAGENET1K_VALIDATION_ROLE = "final_validation"
IMAGENET1K_TRAIN_CONTEXT_COUNT = 1_281_167
IMAGENET1K_VALIDATION_CONTEXT_COUNT = 50_000
IMAGENET1K_CLASS_COUNT = 1_000
IMAGENET1K_VALIDATION_CONTEXTS_PER_CLASS = 50


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be one lowercase SHA-256 string")
    result = str(value)
    if (
        len(result) != 64
        or result != result.lower()
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 string")
    return result


def _optional_sha256(value: object | None, *, name: str) -> str | None:
    return None if value is None else _sha256(value, name=name)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return int(value)


def _selector_role(axis: ContextAxis, *, expected: str, name: str) -> None:
    if axis.selector.get("role") != expected:
        raise ValueError(f"{name} axis must have selector role {expected!r}")
    if axis.selector.get("complete_split") is not True:
        raise ValueError(f"{name} axis must declare complete_split=true")


def _class_order(axis: ContextAxis, *, name: str) -> tuple[str, ...]:
    raw = axis.selector.get("class_order")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{name} axis must carry its exact class_order")
    result = tuple(str(value) for value in raw)
    if (
        len(result) != IMAGENET1K_CLASS_COUNT
        or len(set(result)) != len(result)
        or any(not value for value in result)
    ):
        raise ValueError(
            f"{name} axis must carry exactly {IMAGENET1K_CLASS_COUNT} unique classes"
        )
    if axis.selector.get("class_count") != IMAGENET1K_CLASS_COUNT:
        raise ValueError(
            f"{name} axis selector must declare {IMAGENET1K_CLASS_COUNT} classes"
        )
    return result


def _observed_class_map_and_counts(
    axis: ContextAxis,
    *,
    class_order: tuple[str, ...],
    name: str,
) -> tuple[dict[str, int], dict[str, int]]:
    class_to_label: dict[str, int] = {}
    label_to_class: dict[int, str] = {}
    counts = {class_id: 0 for class_id in class_order}
    for record in axis.records:
        class_id = str(record.class_id)
        label = int(record.label)
        if class_id not in counts:
            raise ValueError(f"{name} axis contains a class outside class_order")
        previous_label = class_to_label.setdefault(class_id, label)
        previous_class = label_to_class.setdefault(label, class_id)
        if previous_label != label or previous_class != class_id:
            raise ValueError(f"{name} axis does not encode a one-to-one class map")
        counts[class_id] += 1
    if (
        set(class_to_label) != set(class_order)
        or set(label_to_class) != set(range(IMAGENET1K_CLASS_COUNT))
    ):
        raise ValueError(
            f"{name} axis does not realize labels 0..{IMAGENET1K_CLASS_COUNT - 1}"
        )
    return class_to_label, counts


@dataclass(frozen=True, slots=True)
class CertifiedImageNetBankIdentity:
    """Path-free identity of one certified preprocessed uint8 bank."""

    replica_receipt_hash: str
    payload_identity_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "replica_receipt_hash",
            _sha256(self.replica_receipt_hash, name="replica_receipt_hash"),
        )
        object.__setattr__(
            self,
            "payload_identity_hash",
            _sha256(self.payload_identity_hash, name="payload_identity_hash"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "replica_receipt_hash": self.replica_receipt_hash,
            "payload_identity_hash": self.payload_identity_hash,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CertifiedImageNetBankIdentity":
        return cls(
            replica_receipt_hash=payload.get("replica_receipt_hash", ""),
            payload_identity_hash=payload.get("payload_identity_hash", ""),
        )


@dataclass(frozen=True, slots=True)
class CompleteImageNetPopulationPairReceipt:
    """Content-addressed proof of the one admissible v1 train/val population."""

    train_axis_hash: str
    train_ordered_context_ids_hash: str
    validation_axis_hash: str
    validation_ordered_context_ids_hash: str
    terminal_ordered_context_ids_hash: str
    preprocessing_hash: str
    label_map_hash: str
    study_hash: str | None = None
    train_bank_identity: CertifiedImageNetBankIdentity | None = None
    validation_bank_identity: CertifiedImageNetBankIdentity | None = None
    schema_name: str = COMPLETE_IMAGENET_POPULATION_PAIR_SCHEMA
    dataset_id: str = IMAGENET1K_DATASET_ID
    train_split: str = IMAGENET1K_TRAIN_SPLIT
    validation_split: str = IMAGENET1K_VALIDATION_SPLIT
    train_role: str = IMAGENET1K_TRAIN_ROLE
    validation_role: str = IMAGENET1K_VALIDATION_ROLE
    train_context_count: int = IMAGENET1K_TRAIN_CONTEXT_COUNT
    validation_context_count: int = IMAGENET1K_VALIDATION_CONTEXT_COUNT
    class_count: int = IMAGENET1K_CLASS_COUNT
    validation_contexts_per_class: int = IMAGENET1K_VALIDATION_CONTEXTS_PER_CLASS

    def __post_init__(self) -> None:
        expected_coordinates = {
            "schema_name": COMPLETE_IMAGENET_POPULATION_PAIR_SCHEMA,
            "dataset_id": IMAGENET1K_DATASET_ID,
            "train_split": IMAGENET1K_TRAIN_SPLIT,
            "validation_split": IMAGENET1K_VALIDATION_SPLIT,
            "train_role": IMAGENET1K_TRAIN_ROLE,
            "validation_role": IMAGENET1K_VALIDATION_ROLE,
            "train_context_count": IMAGENET1K_TRAIN_CONTEXT_COUNT,
            "validation_context_count": IMAGENET1K_VALIDATION_CONTEXT_COUNT,
            "class_count": IMAGENET1K_CLASS_COUNT,
            "validation_contexts_per_class": (
                IMAGENET1K_VALIDATION_CONTEXTS_PER_CLASS
            ),
        }
        for name, expected in expected_coordinates.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"complete ImageNet pair has noncanonical {name}: "
                    f"expected {expected!r}"
                )
        for name in (
            "train_axis_hash",
            "train_ordered_context_ids_hash",
            "validation_axis_hash",
            "validation_ordered_context_ids_hash",
            "terminal_ordered_context_ids_hash",
            "preprocessing_hash",
            "label_map_hash",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "study_hash",
            _optional_sha256(self.study_hash, name="study_hash"),
        )
        if self.train_axis_hash == self.validation_axis_hash:
            raise ValueError("train and validation axes must be distinct")
        if (self.train_bank_identity is None) != (
            self.validation_bank_identity is None
        ):
            raise ValueError("certified train and validation bank identities are paired")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "dataset_id": self.dataset_id,
            "train_split": self.train_split,
            "validation_split": self.validation_split,
            "train_role": self.train_role,
            "validation_role": self.validation_role,
            "train_context_count": self.train_context_count,
            "validation_context_count": self.validation_context_count,
            "class_count": self.class_count,
            "validation_contexts_per_class": self.validation_contexts_per_class,
            "train_axis_hash": self.train_axis_hash,
            "train_ordered_context_ids_hash": self.train_ordered_context_ids_hash,
            "validation_axis_hash": self.validation_axis_hash,
            "validation_ordered_context_ids_hash": (
                self.validation_ordered_context_ids_hash
            ),
            "terminal_ordered_context_ids_hash": (
                self.terminal_ordered_context_ids_hash
            ),
            "preprocessing_hash": self.preprocessing_hash,
            "label_map_hash": self.label_map_hash,
            "study_hash": self.study_hash,
            "train_bank_identity": (
                None
                if self.train_bank_identity is None
                else self.train_bank_identity.to_dict()
            ),
            "validation_bank_identity": (
                None
                if self.validation_bank_identity is None
                else self.validation_bank_identity.to_dict()
            ),
        }

    @property
    def pair_hash(self) -> str:
        return content_hash(self.to_dict())

    @property
    def content_hash(self) -> str:
        return self.pair_hash

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {**self.to_dict(), "content_hash": self.content_hash},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "CompleteImageNetPopulationPairReceipt":
        train_bank_raw = payload.get("train_bank_identity")
        validation_bank_raw = payload.get("validation_bank_identity")
        if train_bank_raw is not None and not isinstance(train_bank_raw, Mapping):
            raise TypeError("train_bank_identity must be a mapping or null")
        if validation_bank_raw is not None and not isinstance(
            validation_bank_raw, Mapping
        ):
            raise TypeError("validation_bank_identity must be a mapping or null")
        result = cls(
            train_axis_hash=payload.get("train_axis_hash", ""),
            train_ordered_context_ids_hash=payload.get(
                "train_ordered_context_ids_hash", ""
            ),
            validation_axis_hash=payload.get("validation_axis_hash", ""),
            validation_ordered_context_ids_hash=payload.get(
                "validation_ordered_context_ids_hash", ""
            ),
            terminal_ordered_context_ids_hash=payload.get(
                "terminal_ordered_context_ids_hash", ""
            ),
            preprocessing_hash=payload.get("preprocessing_hash", ""),
            label_map_hash=payload.get("label_map_hash", ""),
            study_hash=payload.get("study_hash"),
            train_bank_identity=(
                None
                if train_bank_raw is None
                else CertifiedImageNetBankIdentity.from_mapping(train_bank_raw)
            ),
            validation_bank_identity=(
                None
                if validation_bank_raw is None
                else CertifiedImageNetBankIdentity.from_mapping(validation_bank_raw)
            ),
            schema_name=payload.get("schema_name", ""),
            dataset_id=payload.get("dataset_id", ""),
            train_split=payload.get("train_split", ""),
            validation_split=payload.get("validation_split", ""),
            train_role=payload.get("train_role", ""),
            validation_role=payload.get("validation_role", ""),
            train_context_count=payload.get("train_context_count", 0),
            validation_context_count=payload.get("validation_context_count", 0),
            class_count=payload.get("class_count", 0),
            validation_contexts_per_class=payload.get(
                "validation_contexts_per_class", 0
            ),
        )
        declared = payload.get("content_hash")
        if declared is not None and declared != result.content_hash:
            raise ValueError("complete ImageNet population receipt hash differs")
        return result

    @classmethod
    def load(cls, path: str | Path) -> "CompleteImageNetPopulationPairReceipt":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("complete ImageNet population receipt must be a mapping")
        return cls.from_mapping(payload)


def _bank_identity(
    *,
    axis: ContextAxis,
    receipt: PreprocessedUint8BankReplicaReceipt | None,
    payload_identity_hash: str | None,
    name: str,
) -> CertifiedImageNetBankIdentity | None:
    if receipt is None:
        if payload_identity_hash is not None:
            raise ValueError(f"{name} payload identity requires its certified receipt")
        return None
    if receipt.passed is not True or receipt.failures:
        raise ValueError(f"{name} bank receipt is not a passing certification")
    if receipt.axis_hash != axis.content_hash:
        raise ValueError(f"{name} bank receipt names another context axis")
    if receipt.preprocessing_hash != axis.preprocessing.content_hash:
        raise ValueError(f"{name} bank receipt names another preprocessing operator")
    if (
        not receipt.shape
        or int(receipt.shape[0]) != len(axis.records)
        or receipt.dtype != "uint8"
        or receipt.layout.lower() != "nchw"
    ):
        raise ValueError(f"{name} bank receipt does not cover the complete uint8 axis")
    for field_name in (
        "manifest_sha256",
        "inputs_sha256",
        "labels_sha256",
        "pixel_probe_hash",
    ):
        _sha256(getattr(receipt, field_name), name=f"{name}.{field_name}")
    _positive_int(receipt.inputs_size_bytes, name=f"{name}.inputs_size_bytes")
    _positive_int(receipt.labels_size_bytes, name=f"{name}.labels_size_bytes")
    _positive_int(receipt.pixel_probe_count, name=f"{name}.pixel_probe_count")
    expected_payload_identity = certified_uint8_bank_payload_identity_hash(
        inputs_sha256=receipt.inputs_sha256,
        labels_sha256=receipt.labels_sha256,
        inputs_size_bytes=receipt.inputs_size_bytes,
        labels_size_bytes=receipt.labels_size_bytes,
    )
    if (
        payload_identity_hash is not None
        and _sha256(payload_identity_hash, name=f"{name}_bank_payload_identity_hash")
        != expected_payload_identity
    ):
        raise ValueError(f"{name} bank payload identity differs from its receipt")
    return CertifiedImageNetBankIdentity(
        replica_receipt_hash=receipt.content_hash,
        payload_identity_hash=expected_payload_identity,
    )


def certify_complete_imagenet_population_pair(
    *,
    train_axis: ContextAxis,
    validation_axis: ContextAxis,
    terminal_context_ids: Sequence[str],
    study: StudySpec | None = None,
    train_bank_receipt: PreprocessedUint8BankReplicaReceipt | None = None,
    validation_bank_receipt: PreprocessedUint8BankReplicaReceipt | None = None,
    train_bank_payload_identity_hash: str | None = None,
    validation_bank_payload_identity_hash: str | None = None,
) -> CompleteImageNetPopulationPairReceipt:
    """Validate and receipt the exact complete ImageNet-1k train/val pair.

    ``terminal_context_ids`` must be a duplicate-free permutation of the
    complete validation axis.  Its scientific order may be stratified (the
    matched v0/v1 BCE design is), so the receipt binds that order separately
    from the canonical axis order.  Adaptive development may reveal only
    train-axis prefixes, but it cannot substitute a smaller population.
    Optional bank arguments are validated from their small receipts only; this
    function never opens bank arrays or source images.
    """

    # Computing the content hashes invokes ContextAxis.validate(), including
    # its contiguous-order and within-axis context-ID uniqueness checks.
    train_axis_hash = train_axis.content_hash
    validation_axis_hash = validation_axis.content_hash
    if train_axis.dataset_id != IMAGENET1K_DATASET_ID:
        raise ValueError("train axis is not the canonical ImageNet-1k dataset")
    if validation_axis.dataset_id != IMAGENET1K_DATASET_ID:
        raise ValueError("validation axis is not the canonical ImageNet-1k dataset")
    if train_axis.split != IMAGENET1K_TRAIN_SPLIT:
        raise ValueError("train axis must bind split='train'")
    if validation_axis.split != IMAGENET1K_VALIDATION_SPLIT:
        raise ValueError("validation axis must bind split='val'")
    _selector_role(train_axis, expected=IMAGENET1K_TRAIN_ROLE, name="train")
    _selector_role(
        validation_axis,
        expected=IMAGENET1K_VALIDATION_ROLE,
        name="validation",
    )
    if len(train_axis.records) != IMAGENET1K_TRAIN_CONTEXT_COUNT:
        raise ValueError(
            "train axis must contain the complete 1,281,167-context population"
        )
    if len(validation_axis.records) != IMAGENET1K_VALIDATION_CONTEXT_COUNT:
        raise ValueError(
            "validation axis must contain the complete 50,000-context population"
        )

    train_class_order = _class_order(train_axis, name="train")
    validation_class_order = _class_order(validation_axis, name="validation")
    if train_class_order != validation_class_order:
        raise ValueError("train and validation class orders differ")
    train_class_map, _train_counts = _observed_class_map_and_counts(
        train_axis,
        class_order=train_class_order,
        name="train",
    )
    validation_class_map, validation_counts = _observed_class_map_and_counts(
        validation_axis,
        class_order=validation_class_order,
        name="validation",
    )
    if train_class_map != validation_class_map:
        raise ValueError("train and validation class maps differ")
    if any(
        count != IMAGENET1K_VALIDATION_CONTEXTS_PER_CLASS
        for count in validation_counts.values()
    ):
        raise ValueError(
            "validation axis must contain exactly 50 contexts in every class"
        )
    if (
        validation_axis.selector.get("expected_images_per_class")
        != IMAGENET1K_VALIDATION_CONTEXTS_PER_CLASS
    ):
        raise ValueError("validation selector must declare 50 images per class")

    if train_axis.label_map_hash != validation_axis.label_map_hash:
        raise ValueError("train and validation label-map hashes differ")
    label_map_hash = _sha256(train_axis.label_map_hash, name="label_map_hash")
    if label_map_hash != stable_digest({"class_map": train_class_map}):
        raise ValueError("shared label-map hash does not encode the observed class map")
    train_preprocessing_payload = train_axis.preprocessing.to_dict()
    validation_preprocessing_payload = validation_axis.preprocessing.to_dict()
    if (
        train_axis.preprocessing != validation_axis.preprocessing
        or train_preprocessing_payload != validation_preprocessing_payload
        or canonical_json(train_preprocessing_payload)
        != canonical_json(validation_preprocessing_payload)
        or train_axis.preprocessing.content_hash
        != validation_axis.preprocessing.content_hash
    ):
        raise ValueError(
            "train and validation preprocessing must be byte/equality-identical"
        )
    preprocessing_hash = _sha256(
        train_axis.preprocessing.content_hash,
        name="preprocessing_hash",
    )

    # A ContextAxis validates uniqueness within each split.  Holding only the
    # 50k validation IDs lets us additionally prove pairwise disjointness
    # without a second 1.28M-entry train-ID structure.
    validation_ids = {record.context_id for record in validation_axis.records}
    if any(record.context_id in validation_ids for record in train_axis.records):
        raise ValueError("train and validation context IDs must be globally unique")

    if isinstance(terminal_context_ids, (str, bytes)) or not isinstance(
        terminal_context_ids, Sequence
    ):
        raise TypeError("terminal_context_ids must be an ordered sequence")
    terminal_ids = tuple(terminal_context_ids)
    if any(type(terminal_id) is not str for terminal_id in terminal_ids):
        raise TypeError("terminal IDs must be strings")
    if (
        len(terminal_ids) != len(validation_axis.records)
        or len(terminal_ids) != len(set(terminal_ids))
        or set(terminal_ids) != validation_ids
    ):
        raise ValueError("terminal IDs must exhaust the complete validation axis")

    study_hash: str | None = None
    if study is not None:
        study.validate().require_valid()
        study_preprocessing = ImagePreprocessingSpec.from_mapping(study.preprocessing)
        if study.label_map_hash != label_map_hash:
            raise ValueError("Study label map differs from the ImageNet population")
        if (
            study_preprocessing != train_axis.preprocessing
            or canonical_json(study_preprocessing.to_dict())
            != canonical_json(train_preprocessing_payload)
            or study_preprocessing.content_hash != preprocessing_hash
        ):
            raise ValueError("Study preprocessing differs from the ImageNet population")
        study_hash = _sha256(study.content_hash, name="study_hash")

    train_bank_identity = _bank_identity(
        axis=train_axis,
        receipt=train_bank_receipt,
        payload_identity_hash=train_bank_payload_identity_hash,
        name="train",
    )
    validation_bank_identity = _bank_identity(
        axis=validation_axis,
        receipt=validation_bank_receipt,
        payload_identity_hash=validation_bank_payload_identity_hash,
        name="validation",
    )
    if (train_bank_identity is None) != (validation_bank_identity is None):
        raise ValueError("certified train and validation bank receipts are paired")

    return CompleteImageNetPopulationPairReceipt(
        train_axis_hash=train_axis_hash,
        train_ordered_context_ids_hash=train_axis.ids_hash,
        validation_axis_hash=validation_axis_hash,
        validation_ordered_context_ids_hash=validation_axis.ids_hash,
        terminal_ordered_context_ids_hash=ordered_context_ids_hash(terminal_ids),
        preprocessing_hash=preprocessing_hash,
        label_map_hash=label_map_hash,
        study_hash=study_hash,
        train_bank_identity=train_bank_identity,
        validation_bank_identity=validation_bank_identity,
        train_context_count=IMAGENET1K_TRAIN_CONTEXT_COUNT,
        validation_context_count=IMAGENET1K_VALIDATION_CONTEXT_COUNT,
        class_count=IMAGENET1K_CLASS_COUNT,
        validation_contexts_per_class=(
            IMAGENET1K_VALIDATION_CONTEXTS_PER_CLASS
        ),
    )


__all__ = [
    "COMPLETE_IMAGENET_POPULATION_PAIR_SCHEMA",
    "CertifiedImageNetBankIdentity",
    "CompleteImageNetPopulationPairReceipt",
    "IMAGENET1K_CLASS_COUNT",
    "IMAGENET1K_DATASET_ID",
    "IMAGENET1K_TRAIN_CONTEXT_COUNT",
    "IMAGENET1K_TRAIN_ROLE",
    "IMAGENET1K_TRAIN_SPLIT",
    "IMAGENET1K_VALIDATION_CONTEXT_COUNT",
    "IMAGENET1K_VALIDATION_CONTEXTS_PER_CLASS",
    "IMAGENET1K_VALIDATION_ROLE",
    "IMAGENET1K_VALIDATION_SPLIT",
    "certify_complete_imagenet_population_pair",
]
