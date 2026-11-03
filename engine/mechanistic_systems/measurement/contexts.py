"""Content-addressed image contexts for measurement and circuit evaluation.

The scientific axis is an ordered table of relative image IDs, labels, and
frozen context blocks. Storage roots and worker counts are runtime concerns;
they may change between replicas without changing the DataView.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from json.encoder import encode_basestring_ascii as _json_ascii_string
from pathlib import Path
from typing import Any

import numpy as np

from circuitflow.contracts import DataView
from contracts.hashing import sha256_file, stable_digest
from mechanistic_systems.contracts import (
    MeasurementPlan,
    ordered_axis_hash,
)
from mechanistic_systems.input_operators import (
    CANONICAL_IMAGENET_INPUT_OPERATOR_V1,
    canonical_imagenet_input_operator_contract,
)
from mechanistic_systems.measurement.bank_staging import resolve_staged_bank
from mechanistic_systems.measurement.lazy_attribute import LazyAttribute

MeasurementInputBatch = LazyAttribute("mechanistic_systems.measurement.torch_adapter", "MeasurementInputBatch")
from mechanistic_systems.storage import (
    iter_parquet_rows,
    write_parquet_row_batches,
)

try:  # Optional for import smoke; required by executable image providers.
    import torch
    from torch.utils.data import DataLoader, Dataset
    ImageReadMode = read_image = transforms_v2 = None
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]
    DataLoader = Dataset = None  # type: ignore[assignment]
    ImageReadMode = read_image = transforms_v2 = None  # type: ignore[assignment]

_DatasetBase = Dataset if Dataset is not None else object


CONTEXT_AXIS_SCHEMA = "mechanistic_context_axis_v2"
PREPROCESSING_SCHEMA = "mechanistic_image_preprocessing_v1"
SCRATCH_FOLDER_LOADER_CONTRACT = "scratch_folder_uint8_prefix_online_normalize_v1"
PREPROCESSED_UINT8_BANK_LOADER_CONTRACT = (
    "content_verified_preprocessed_uint8_bank_online_normalize_v1"
)
_IMAGE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".webp"})
UINT8_BANK_RECEIPT_SCHEMA = "preprocessed_uint8_bank_replica_receipt_v1"
UINT8_BANK_RUNTIME_VERIFICATION_MODES = (
    "deep_sha256",
    "certified_probes",
)


def _require_torchvision() -> None:
    global ImageReadMode, read_image, transforms_v2
    if transforms_v2 is None:
        try:
            from torchvision.io import ImageReadMode, read_image
            from torchvision.transforms import v2 as transforms_v2
        except Exception:
            pass  # Retain the existing executable-provider error below.
    if (
        torch is None
        or DataLoader is None
        or read_image is None
        or transforms_v2 is None
    ):
        raise RuntimeError("image context execution requires PyTorch and torchvision")


def ordered_context_ids_hash(context_ids: Sequence[str]) -> str:
    values = tuple(str(value) for value in context_ids)
    if not values or len(set(values)) != len(values):
        raise ValueError("context IDs must be nonempty and unique")
    return ordered_axis_hash(values)


@dataclass(frozen=True)
class PreprocessedUint8BankReplicaReceipt:
    """Content and semantic certification for one runtime uint8 bank replica."""

    axis_hash: str
    preprocessing_hash: str
    manifest_sha256: str
    inputs_sha256: str
    labels_sha256: str
    inputs_size_bytes: int
    labels_size_bytes: int
    shape: tuple[int, ...]
    dtype: str
    layout: str
    ordering_contract: str
    pixel_probe_count: int
    pixel_probe_hash: str
    passed: bool
    failures: tuple[str, ...] = tuple()
    schema_name: str = UINT8_BANK_RECEIPT_SCHEMA

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "axis_hash": self.axis_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "manifest_sha256": self.manifest_sha256,
            "inputs_sha256": self.inputs_sha256,
            "labels_sha256": self.labels_sha256,
            "inputs_size_bytes": int(self.inputs_size_bytes),
            "labels_size_bytes": int(self.labels_size_bytes),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "ordering_contract": self.ordering_contract,
            "pixel_probe_count": int(self.pixel_probe_count),
            "pixel_probe_hash": self.pixel_probe_hash,
            "passed": bool(self.passed),
            "failures": list(self.failures),
        }

    @property
    def content_hash(self) -> str:
        return stable_digest(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_hash": self.content_hash}

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessedUint8BankReplicaReceipt":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_name") != UINT8_BANK_RECEIPT_SCHEMA:
            raise ValueError("unsupported uint8-bank replica receipt schema")
        result = cls(
            axis_hash=str(payload.get("axis_hash", "")),
            preprocessing_hash=str(payload.get("preprocessing_hash", "")),
            manifest_sha256=str(payload.get("manifest_sha256", "")),
            inputs_sha256=str(payload.get("inputs_sha256", "")),
            labels_sha256=str(payload.get("labels_sha256", "")),
            inputs_size_bytes=int(payload.get("inputs_size_bytes", -1)),
            labels_size_bytes=int(payload.get("labels_size_bytes", -1)),
            shape=tuple(int(value) for value in payload.get("shape", ()) or ()),
            dtype=str(payload.get("dtype", "")),
            layout=str(payload.get("layout", "")),
            ordering_contract=str(payload.get("ordering_contract", "")),
            pixel_probe_count=int(payload.get("pixel_probe_count", 0)),
            pixel_probe_hash=str(payload.get("pixel_probe_hash", "")),
            passed=bool(payload.get("passed", False)),
            failures=tuple(str(value) for value in payload.get("failures", ()) or ()),
        )
        if result.content_hash != str(payload.get("content_hash", "")):
            raise ValueError("uint8-bank replica receipt content hash differs")
        return result

    def require_passed(self, *, axis: "ContextAxis", manifest_path: str | Path) -> None:
        failures = list(self.failures)
        if not self.passed:
            failures.append("receipt_not_passed")
        if self.axis_hash != axis.content_hash:
            failures.append("axis_hash_mismatch")
        if self.preprocessing_hash != axis.preprocessing.content_hash:
            failures.append("preprocessing_hash_mismatch")
        if self.manifest_sha256 != sha256_file(Path(manifest_path)):
            failures.append("manifest_sha256_mismatch")
        if failures:
            raise ValueError(
                "uint8-bank replica is not certified: "
                + ",".join(sorted(set(failures)))
            )


@dataclass(frozen=True)
class ImagePreprocessingSpec:
    """Serializable tensor preprocessing with no hidden model defaults."""

    preprocessing_id: str
    resize_size: tuple[int, ...]
    center_crop_size: tuple[int, ...] = tuple()
    interpolation: str = "bilinear"
    antialias: bool = True
    scale_uint8_to_unit: bool = True
    mean: tuple[float, ...] = tuple()
    std: tuple[float, ...] = tuple()
    output_dtype: str = "float32"
    notes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.preprocessing_id:
            raise ValueError("preprocessing ID must be nonempty")
        if len(self.resize_size) not in {1, 2} or any(
            int(value) <= 0 for value in self.resize_size
        ):
            raise ValueError("resize_size must contain one or two positive dimensions")
        if self.center_crop_size and (
            len(self.center_crop_size) not in {1, 2}
            or any(int(value) <= 0 for value in self.center_crop_size)
        ):
            raise ValueError(
                "center_crop_size must contain one or two positive dimensions"
            )
        if self.interpolation not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError("unsupported image interpolation policy")
        if bool(self.mean) != bool(self.std) or (
            self.mean and len(self.mean) != len(self.std)
        ):
            raise ValueError(
                "normalization mean and standard deviation must have matching axes"
            )
        if self.mean and (
            len(self.mean) != 3 or any(float(value) <= 0.0 for value in self.std)
        ):
            raise ValueError(
                "image normalization requires three channels and positive standard deviations"
            )
        if self.output_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("unsupported preprocessing output dtype")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_name": PREPROCESSING_SCHEMA,
            "preprocessing_id": self.preprocessing_id,
            "resize_size": list(self.resize_size),
            "center_crop_size": list(self.center_crop_size),
            "interpolation": self.interpolation,
            "antialias": bool(self.antialias),
            "scale_uint8_to_unit": bool(self.scale_uint8_to_unit),
            "mean": list(self.mean),
            "std": list(self.std),
            "output_dtype": self.output_dtype,
            "notes": dict(self.notes),
        }

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ImagePreprocessingSpec:
        result = cls(
            preprocessing_id=str(payload.get("preprocessing_id", "")),
            resize_size=tuple(
                int(value) for value in payload.get("resize_size", ()) or ()
            ),
            center_crop_size=tuple(
                int(value) for value in payload.get("center_crop_size", ()) or ()
            ),
            interpolation=str(payload.get("interpolation", "bilinear")),
            antialias=bool(payload.get("antialias", True)),
            scale_uint8_to_unit=bool(payload.get("scale_uint8_to_unit", True)),
            mean=tuple(float(value) for value in payload.get("mean", ()) or ()),
            std=tuple(float(value) for value in payload.get("std", ()) or ()),
            output_dtype=str(payload.get("output_dtype", "float32")),
            notes=dict(payload.get("notes", {}) or {}),
        )
        result.validate()
        return result

    def build_transform(self) -> Any:
        _require_torchvision()
        if self.notes.get("physical_decoder") == "pil_rgb":
            from torchvision import transforms

            interpolation = {
                "nearest": transforms.InterpolationMode.NEAREST,
                "bilinear": transforms.InterpolationMode.BILINEAR,
                "bicubic": transforms.InterpolationMode.BICUBIC,
            }[self.interpolation]
            steps: list[Any] = [
                transforms.Resize(
                    (
                        self.resize_size[0]
                        if len(self.resize_size) == 1
                        else self.resize_size
                    ),
                    interpolation=interpolation,
                    antialias=self.antialias,
                )
            ]
            if self.center_crop_size:
                steps.append(
                    transforms.CenterCrop(
                        self.center_crop_size[0]
                        if len(self.center_crop_size) == 1
                        else self.center_crop_size
                    )
                )
            steps.extend(
                [
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float32),
                ]
            )
            if self.mean:
                steps.append(transforms.Normalize(mean=self.mean, std=self.std))
            if self.output_dtype != "float32":
                dtype = {
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                }[self.output_dtype]
                steps.append(transforms.ConvertImageDtype(dtype))
            return transforms.Compose(steps)
        interpolation = {
            "nearest": transforms_v2.InterpolationMode.NEAREST,
            "bilinear": transforms_v2.InterpolationMode.BILINEAR,
            "bicubic": transforms_v2.InterpolationMode.BICUBIC,
        }[self.interpolation]
        steps: list[Any] = [
            transforms_v2.ToImage(),
            transforms_v2.Resize(
                self.resize_size[0] if len(self.resize_size) == 1 else self.resize_size,
                interpolation=interpolation,
                antialias=self.antialias,
            ),
        ]
        if self.center_crop_size:
            steps.append(
                transforms_v2.CenterCrop(
                    self.center_crop_size[0]
                    if len(self.center_crop_size) == 1
                    else self.center_crop_size
                )
            )
        steps.append(
            transforms_v2.ToDtype(torch.float32, scale=bool(self.scale_uint8_to_unit))
        )
        if self.mean:
            steps.append(transforms_v2.Normalize(mean=self.mean, std=self.std))
        if self.output_dtype != "float32":
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
                self.output_dtype
            ]
            steps.append(transforms_v2.ToDtype(dtype, scale=False))
        return transforms_v2.Compose(steps)

    def build_uint8_prefix_transform(self) -> Any:
        """Build the historical scratch-cache transform before dtype/normalization."""

        _require_torchvision()
        if self.notes.get("physical_decoder") == "pil_rgb":
            from torchvision import transforms

            interpolation = {
                "nearest": transforms.InterpolationMode.NEAREST,
                "bilinear": transforms.InterpolationMode.BILINEAR,
                "bicubic": transforms.InterpolationMode.BICUBIC,
            }[self.interpolation]
            steps: list[Any] = [
                transforms.Resize(
                    (
                        self.resize_size[0]
                        if len(self.resize_size) == 1
                        else self.resize_size
                    ),
                    interpolation=interpolation,
                    antialias=self.antialias,
                )
            ]
            if self.center_crop_size:
                steps.append(
                    transforms.CenterCrop(
                        self.center_crop_size[0]
                        if len(self.center_crop_size) == 1
                        else self.center_crop_size
                    )
                )
            steps.append(transforms.PILToTensor())
            return transforms.Compose(steps)
        interpolation = {
            "nearest": transforms_v2.InterpolationMode.NEAREST,
            "bilinear": transforms_v2.InterpolationMode.BILINEAR,
            "bicubic": transforms_v2.InterpolationMode.BICUBIC,
        }[self.interpolation]
        steps: list[Any] = [
            transforms_v2.ToImage(),
            transforms_v2.Resize(
                self.resize_size[0] if len(self.resize_size) == 1 else self.resize_size,
                interpolation=interpolation,
                antialias=self.antialias,
            ),
        ]
        if self.center_crop_size:
            steps.append(
                transforms_v2.CenterCrop(
                    self.center_crop_size[0]
                    if len(self.center_crop_size) == 1
                    else self.center_crop_size
                )
            )
        return transforms_v2.Compose(steps)

    def uint8_online_decode_parameters(self, *, num_channels: int) -> tuple[Any, Any]:
        """Return the exact affine decode used after scratch uint8 materialization."""

        _require_torchvision()
        if int(num_channels) <= 0:
            raise ValueError("online uint8 decoding requires a positive channel count")
        if not self.scale_uint8_to_unit:
            raise ValueError(
                "scratch-folder parity requires scale_uint8_to_unit=True before normalization"
            )
        if self.mean:
            if len(self.mean) != int(num_channels) or len(self.std) != int(
                num_channels
            ):
                raise ValueError(
                    "normalization coordinates differ from the image channels"
                )
            mean = torch.as_tensor(self.mean, dtype=torch.float32).view(
                1, int(num_channels), 1, 1
            )
            std = torch.as_tensor(self.std, dtype=torch.float32).view(
                1, int(num_channels), 1, 1
            )
            scale = 1.0 / (255.0 * std.clamp_min(1e-8))
            bias = -mean / std.clamp_min(1e-8)
        else:
            scale = torch.full(
                (1, int(num_channels), 1, 1),
                1.0 / 255.0,
                dtype=torch.float32,
            )
            bias = torch.zeros(
                (1, int(num_channels), 1, 1),
                dtype=torch.float32,
            )
        return scale, bias


def imagenet_preprocessing_spec(policy: str) -> ImagePreprocessingSpec:
    """Return one explicit ImageNet policy used by certified model lanes."""

    normalized = str(policy).strip().lower()
    imagenet_normalization = {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    }
    if normalized == CANONICAL_IMAGENET_INPUT_OPERATOR_V1:
        return ImagePreprocessingSpec.from_mapping(
            canonical_imagenet_input_operator_contract()
        )
    if normalized == "legacy_resnet50_config_native":
        return ImagePreprocessingSpec(
            preprocessing_id=normalized,
            resize_size=(224, 224),
            interpolation="bilinear",
            antialias=True,
            output_dtype="float32",
            notes={
                "historical_lane": "supervised_resnet50",
                "source": "configs.resnet50_config",
            },
            **imagenet_normalization,
        )
    if normalized == "legacy_pruning_uint8_bilinear_center_crop224_imagenet":
        return ImagePreprocessingSpec(
            preprocessing_id=normalized,
            resize_size=(256,),
            center_crop_size=(224, 224),
            interpolation="bilinear",
            antialias=True,
            output_dtype="float32",
            notes={
                "historical_lane": "sampled_path_frontier_v1",
                "physical_decoder": "torchvision_imagefolder_pil_rgb",
                "storage_quantization": "round_clamp_uint8_v1",
                "online_decode": "uint8_to_float32_imagenet_normalize",
            },
            **imagenet_normalization,
        )
    if normalized == "resize256_center_crop224_imagenet":
        return ImagePreprocessingSpec(
            preprocessing_id=normalized,
            resize_size=(256,),
            center_crop_size=(224, 224),
            interpolation="bicubic",
            antialias=True,
            output_dtype="float32",
            **imagenet_normalization,
        )
    if normalized == "resize256_center_crop224_unit_float":
        return ImagePreprocessingSpec(
            preprocessing_id=normalized,
            resize_size=(256,),
            center_crop_size=(224, 224),
            interpolation="bicubic",
            output_dtype="float32",
            notes={"normalization": "none"},
        )
    raise ValueError(f"unsupported canonical ImageNet preprocessing policy {policy!r}")


@dataclass(frozen=True)
class ContextRecord:
    ordinal: int
    context_id: str
    context_block_id: str
    relative_path: str
    label: int
    class_id: str
    source_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": int(self.ordinal),
            "context_id": self.context_id,
            "context_block_id": self.context_block_id,
            "relative_path": self.relative_path,
            "label": int(self.label),
            "class_id": self.class_id,
            "source_size_bytes": int(self.source_size_bytes),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ContextRecord:
        return cls(
            ordinal=int(payload.get("ordinal", -1)),
            context_id=str(payload.get("context_id", "")),
            context_block_id=str(payload.get("context_block_id", "")),
            relative_path=str(payload.get("relative_path", "")),
            label=int(payload.get("label", -1)),
            class_id=str(payload.get("class_id", "")),
            source_size_bytes=int(payload.get("source_size_bytes", -1)),
        )


# Every category process re-derives the ordered-id and
# content hashes of the 1.28M-record train axis when it loads it (the load
# compares them with the manifest).  The per-record canonical JSON of these
# fixed-schema records is assembled directly with the very string encoder
# ``json.dumps(..., ensure_ascii=True)`` uses, and hashed in large chunks.  The
# bytes fed to SHA-256 are json.dumps's, so the digests agree; a probe of
# records (first, last, powers of two, a stride) is checked against json.dumps
# on every call and any difference, or any field of an unexpected type, takes
# the per-record json.dumps path.  ContextAxis.load still fails closed on a
# content-hash mismatch against its manifest.
_CANONICAL_JSON_CHUNK = 65_536


def _context_record_json_slow(record: "ContextRecord") -> str:
    return json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))


def _context_record_json_fast(record: "ContextRecord") -> str | None:
    context_id = record.context_id
    block_id = record.context_block_id
    relative_path = record.relative_path
    class_id = record.class_id
    if not (
        isinstance(context_id, str)
        and isinstance(block_id, str)
        and isinstance(relative_path, str)
        and isinstance(class_id, str)
    ):
        return None
    return (
        '{"class_id":'
        + _json_ascii_string(class_id)
        + ',"context_block_id":'
        + _json_ascii_string(block_id)
        + ',"context_id":'
        + _json_ascii_string(context_id)
        + ',"label":'
        + int.__repr__(int(record.label))
        + ',"ordinal":'
        + int.__repr__(int(record.ordinal))
        + ',"relative_path":'
        + _json_ascii_string(relative_path)
        + ',"source_size_bytes":'
        + int.__repr__(int(record.source_size_bytes))
        + "}"
    )


def _context_id_json_fast(record: "ContextRecord") -> str | None:
    context_id = record.context_id
    return _json_ascii_string(context_id) if isinstance(context_id, str) else None


def _canonical_json_probe_indices(count: int) -> list[int]:
    probes = {0, count - 1, *range(0, count, 4096)}
    power = 1
    while power < count:
        probes.add(power - 1)
        power *= 2
    return sorted(index for index in probes if 0 <= index < count)


def _update_canonical_json_sequence(
    digest: Any,
    records: Sequence[Any],
    *,
    fast: Any,
    slow: Any,
) -> None:
    """``digest.update`` the comma-joined canonical JSON of ``records``."""

    count = len(records)
    if not count:
        return
    use_fast = all(
        fast(records[index]) == slow(records[index])
        for index in _canonical_json_probe_indices(count)
    )
    if not use_fast:
        for index, record in enumerate(records):
            if index:
                digest.update(b",")
            digest.update(slow(record).encode("utf-8"))
        return
    for start in range(0, count, _CANONICAL_JSON_CHUNK):
        parts = []
        for record in records[start : start + _CANONICAL_JSON_CHUNK]:
            text = fast(record)
            parts.append(slow(record) if text is None else text)
        if start:
            digest.update(b",")
        digest.update(",".join(parts).encode("utf-8"))


@dataclass(frozen=True)
class ContextAxis:
    axis_id: str
    dataset_id: str
    split: str
    preprocessing: ImagePreprocessingSpec
    label_map_hash: str
    records: tuple[ContextRecord, ...]
    selector: Mapping[str, Any]

    def validate(self) -> None:
        if not self.axis_id or not self.dataset_id or not self.split:
            raise ValueError("context axis identity must be nonempty")
        self.preprocessing.validate()
        if len(self.label_map_hash) != 64 or not self.records:
            raise ValueError("context axis requires a label-map hash and records")
        seen_context_ids: set[str] = set()
        for expected_ordinal, record in enumerate(self.records):
            if record.ordinal != expected_ordinal:
                raise ValueError(
                    "context records must use a contiguous canonical order"
                )
            if record.context_id in seen_context_ids:
                raise ValueError("context records may not reuse an image across blocks")
            seen_context_ids.add(record.context_id)
            if (
                not record.context_id
                or not record.context_block_id
                or not record.relative_path
                or not record.class_id
                or record.label < 0
                or record.source_size_bytes < 0
            ):
                raise ValueError("context record contains invalid coordinates")

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(record.context_id for record in self.records)

    @property
    def context_block_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(record.context_block_id for record in self.records))

    @cached_property
    def ids_hash(self) -> str:
        # DataView axes use one role-independent ordered-axis identity. Stream
        # the canonical ``ordered_axis_hash`` envelope so million-image axes do
        # not require a second tuple/list of IDs in memory.
        digest = hashlib.sha256()
        digest.update(b'{"ordered_axis_ids":[')
        # The same bytes, emitted in chunks (see
        # _update_canonical_json_sequence); json.dumps is the fallback.
        _update_canonical_json_sequence(
            digest,
            self.records,
            fast=_context_id_json_fast,
            slow=lambda record: json.dumps(
                record.context_id, separators=(",", ":")
            ),
        )
        digest.update(b"]}")
        return digest.hexdigest()

    def to_dict(self, *, include_records: bool = True) -> dict[str, Any]:
        self.validate()
        payload = {
            "schema_name": CONTEXT_AXIS_SCHEMA,
            "axis_id": self.axis_id,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "preprocessing": self.preprocessing.to_dict(),
            "label_map_hash": self.label_map_hash,
            "selector": dict(self.selector),
            "context_block_ids": list(self.context_block_ids),
            "context_count": len(self.records),
            "ids_hash": self.ids_hash,
        }
        if include_records:
            payload["records"] = [record.to_dict() for record in self.records]
        return payload

    @cached_property
    def content_hash(self) -> str:
        # Match stable_digest(self.to_dict(include_records=True)) exactly while
        # never constructing a second million-row Python object graph.
        base = self.to_dict(include_records=False)
        keys = sorted((*base.keys(), "records"))
        digest = hashlib.sha256()
        digest.update(b"{")
        for key_index, key in enumerate(keys):
            if key_index:
                digest.update(b",")
            digest.update(json.dumps(key, separators=(",", ":")).encode("utf-8"))
            digest.update(b":")
            if key == "records":
                digest.update(b"[")
                _update_canonical_json_sequence(
                    digest,
                    self.records,
                    fast=_context_record_json_fast,
                    slow=_context_record_json_slow,
                )
                digest.update(b"]")
            else:
                digest.update(
                    json.dumps(
                        base[key],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
        digest.update(b"}")
        return digest.hexdigest()

    def data_view(self) -> DataView:
        return DataView(
            view_id=self.axis_id,
            dataset_id=self.dataset_id,
            ids_hash=self.ids_hash,
            selector={
                **dict(self.selector),
                "schema_name": CONTEXT_AXIS_SCHEMA,
                "context_axis_hash": self.content_hash,
                "context_block_ids": list(self.context_block_ids),
                "split": self.split,
            },
            preprocessing_id=self.preprocessing.content_hash,
            label_map_hash=self.label_map_hash,
            axis_role="stimulus",
        )

    def subset(self, *, axis_id: str, context_block_ids: Sequence[str]) -> ContextAxis:
        requested = tuple(str(value) for value in context_block_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("context-axis subset requires unique block IDs")
        unknown = sorted(set(requested) - set(self.context_block_ids))
        if unknown:
            raise ValueError(
                "context-axis subset has unknown blocks: " + ",".join(unknown)
            )
        selected = [
            record
            for block_id in requested
            for record in self.records
            if record.context_block_id == block_id
        ]
        return ContextAxis(
            axis_id=str(axis_id),
            dataset_id=self.dataset_id,
            split=self.split,
            preprocessing=self.preprocessing,
            label_map_hash=self.label_map_hash,
            records=tuple(
                ContextRecord(
                    ordinal=index,
                    **{
                        key: value
                        for key, value in record.to_dict().items()
                        if key != "ordinal"
                    },
                )
                for index, record in enumerate(selected)
            ),
            selector={
                **dict(self.selector),
                "parent_context_axis_hash": self.content_hash,
            },
        )

    def select_contexts(
        self,
        *,
        axis_id: str,
        context_ids: Sequence[str],
        selector: Mapping[str, Any] | None = None,
    ) -> ContextAxis:
        """Freeze an explicitly ordered, provenance-linked subset of this axis."""

        requested = tuple(str(value) for value in context_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("context selection requires unique context IDs")
        by_id = {record.context_id: record for record in self.records}
        unknown = tuple(
            context_id for context_id in requested if context_id not in by_id
        )
        if unknown:
            raise ValueError(
                "context selection has unknown IDs: " + ",".join(unknown[:8])
            )
        selected = tuple(by_id[context_id] for context_id in requested)
        return ContextAxis(
            axis_id=str(axis_id),
            dataset_id=self.dataset_id,
            split=self.split,
            preprocessing=self.preprocessing,
            label_map_hash=self.label_map_hash,
            records=tuple(
                ContextRecord(
                    ordinal=index,
                    **{
                        key: value
                        for key, value in record.to_dict().items()
                        if key != "ordinal"
                    },
                )
                for index, record in enumerate(selected)
            ),
            selector={
                **dict(selector or {}),
                "selection": "explicit_ordered_context_subset_v1",
                "parent_context_axis_hash": self.content_hash,
                "parent_context_count": len(self.records),
            },
        )

    def write(self, directory: str | Path) -> tuple[Path, Path]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        table = write_parquet_row_batches(
            target / "context_axis.parquet",
            (record.to_dict() for record in self.records),
        )
        manifest = target / "context_axis_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    **self.to_dict(include_records=False),
                    "content_hash": self.content_hash,
                    "coordinate_table": table.name,
                    "coordinate_table_sha256": sha256_file(table),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return manifest, table

    @classmethod
    def load(cls, manifest_path: str | Path) -> ContextAxis:
        source = Path(manifest_path)
        payload = json.loads(source.read_text())
        if payload.get("schema_name") != CONTEXT_AXIS_SCHEMA:
            raise ValueError("unsupported context-axis schema")
        table = source.parent / str(payload.get("coordinate_table", ""))
        if sha256_file(table) != str(payload.get("coordinate_table_sha256", "")):
            raise ValueError("context-axis coordinate checksum mismatch")
        records = tuple(
            ContextRecord.from_mapping(row) for row in iter_parquet_rows(table)
        )
        axis = cls(
            axis_id=str(payload.get("axis_id", "")),
            dataset_id=str(payload.get("dataset_id", "")),
            split=str(payload.get("split", "")),
            preprocessing=ImagePreprocessingSpec.from_mapping(
                dict(payload.get("preprocessing", {}) or {})
            ),
            label_map_hash=str(payload.get("label_map_hash", "")),
            records=records,
            selector=dict(payload.get("selector", {}) or {}),
        )
        axis.validate()
        if axis.content_hash != str(payload.get("content_hash", "")):
            raise ValueError("context-axis content hash mismatch")
        return axis


def rebind_context_axis_preprocessing(
    source_manifest: str | Path,
    *,
    preprocessing: ImagePreprocessingSpec,
    axis_id: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a new scientific axis over unchanged ordered image coordinates.

    Cache location is operational, but preprocessing is scientific identity.
    A compatible physical bank therefore receives a new ContextAxis rather
    than mutating the source axis or weakening replica certification.
    """

    source_path = Path(source_manifest).resolve()
    source = ContextAxis.load(source_path)
    preprocessing.validate()
    rebound = ContextAxis(
        axis_id=str(axis_id),
        dataset_id=source.dataset_id,
        split=source.split,
        preprocessing=preprocessing,
        label_map_hash=source.label_map_hash,
        records=source.records,
        selector={
            **dict(source.selector),
            "axis_transform": "ordered_coordinates_preserved_preprocessing_rebind_v1",
            "parent_context_axis_hash": source.content_hash,
            "parent_preprocessing_hash": source.preprocessing.content_hash,
        },
    )
    rebound.validate()
    if rebound.ids_hash != source.ids_hash:
        raise RuntimeError("context-axis preprocessing rebind changed ordered IDs")
    manifest, table = rebound.write(output_dir)
    return {
        "schema_name": "mechanistic_context_axis_preprocessing_rebind_v1",
        "source_manifest": str(source_path),
        "source_axis_hash": source.content_hash,
        "source_preprocessing_hash": source.preprocessing.content_hash,
        "axis_manifest": str(manifest),
        "axis_table": str(table),
        "axis_hash": rebound.content_hash,
        "preprocessing_hash": rebound.preprocessing.content_hash,
        "ordered_ids_hash": rebound.ids_hash,
        "ordered_coordinates_preserved": True,
        "context_count": len(rebound.records),
    }


def build_balanced_imagefolder_context_axis(
    root: str | Path,
    *,
    axis_id: str,
    dataset_id: str,
    split: str,
    class_map: Mapping[str, int],
    block_windows: Mapping[str, tuple[int, int]],
    preprocessing: ImagePreprocessingSpec,
) -> ContextAxis:
    """Freeze sorted, disjoint per-class windows from one ImageFolder replica."""

    source = Path(root)
    if not source.is_dir():
        raise NotADirectoryError(source)
    normalized_map = {str(key): int(value) for key, value in class_map.items()}
    if not normalized_map or len(set(normalized_map.values())) != len(normalized_map):
        raise ValueError("class map must be nonempty with unique nonnegative labels")
    if any(value < 0 for value in normalized_map.values()):
        raise ValueError("class map labels must be nonnegative")
    windows = tuple(
        (str(block_id), int(offset), int(count))
        for block_id, (offset, count) in block_windows.items()
    )
    if not windows or len({item[0] for item in windows}) != len(windows):
        raise ValueError("context block windows must be nonempty and unique")
    if any(offset < 0 or count <= 0 for _block, offset, count in windows):
        raise ValueError(
            "context block offsets must be nonnegative and counts positive"
        )
    records: list[ContextRecord] = []
    classes = sorted(normalized_map.items(), key=lambda item: (item[1], item[0]))
    for block_id, offset, count in windows:
        for class_id, label in classes:
            class_root = source / class_id
            if not class_root.is_dir():
                raise FileNotFoundError(
                    f"ImageFolder class directory is missing: {class_root}"
                )
            members = sorted(
                (
                    path
                    for path in class_root.iterdir()
                    if path.is_file()
                    and not path.name.startswith(".")
                    and path.suffix.lower() in _IMAGE_EXTENSIONS
                ),
                key=lambda path: path.name,
            )
            selected = members[offset : offset + count]
            if len(selected) != count:
                raise ValueError(
                    f"class {class_id!r} has {len(members)} images but block {block_id!r} "
                    f"requires indices [{offset},{offset + count})"
                )
            for member in selected:
                relative = member.relative_to(source).as_posix()
                records.append(
                    ContextRecord(
                        ordinal=len(records),
                        context_id=f"{dataset_id}:{split}:{relative}",
                        context_block_id=block_id,
                        relative_path=relative,
                        label=label,
                        class_id=class_id,
                        source_size_bytes=int(member.stat().st_size),
                    )
                )
    axis = ContextAxis(
        axis_id=str(axis_id),
        dataset_id=str(dataset_id),
        split=str(split),
        preprocessing=preprocessing,
        label_map_hash=stable_digest({"class_map": normalized_map}),
        records=tuple(records),
        selector={
            "selection": "sorted_balanced_imagefolder_windows",
            "block_windows": [
                {"context_block_id": block_id, "offset": offset, "count": count}
                for block_id, offset, count in windows
            ],
            "class_count": len(normalized_map),
        },
    )
    axis.validate()
    return axis


def build_scratch_imagefolder_context_axis(
    root: str | Path,
    *,
    axis_id: str,
    dataset_id: str,
    split: str,
    class_map: Mapping[str, int],
    block_windows: Mapping[str, tuple[int, int]],
    preprocessing: ImagePreprocessingSpec,
    reverse_order: bool = False,
    record_order: str = "block_major",
) -> ContextAxis:
    """Freeze the exact ordering used by ``rank_image_data_id_folder``.

    Historical score production retained filesystem class-directory order,
    sorted filenames within each class, selected a per-class shifted window,
    and then materialized the resized/cropped tensors as uint8.  The resulting
    records make that otherwise implicit order immutable.
    """

    source = Path(root)
    if not source.is_dir():
        raise NotADirectoryError(source)
    preprocessing.validate()
    normalized_map = {str(key): int(value) for key, value in class_map.items()}
    if not normalized_map or len(set(normalized_map.values())) != len(normalized_map):
        raise ValueError("class map must be nonempty with unique nonnegative labels")
    if any(value < 0 for value in normalized_map.values()):
        raise ValueError("class map labels must be nonnegative")
    windows = tuple(
        (str(block_id), int(offset), int(count))
        for block_id, (offset, count) in block_windows.items()
    )
    if not windows or len({item[0] for item in windows}) != len(windows):
        raise ValueError("context block windows must be nonempty and unique")
    if any(offset < 0 or count <= 0 for _block, offset, count in windows):
        raise ValueError(
            "context block offsets must be nonnegative and counts positive"
        )

    filesystem_classes = [
        class_id
        for class_id in os.listdir(source)
        if (source / class_id).is_dir() and class_id in normalized_map
    ]
    missing_classes = sorted(set(normalized_map) - set(filesystem_classes))
    if missing_classes:
        raise FileNotFoundError(
            "ImageFolder class directories are missing: "
            + ",".join(missing_classes[:8])
        )
    if record_order not in {"block_major", "source_major"}:
        raise ValueError(
            "scratch context record order must be block_major or source_major"
        )
    records: list[ContextRecord] = []

    def append_record(*, block_id: str, class_id: str, name: str) -> None:
        member = source / class_id / name
        if not member.is_file():
            raise ValueError(
                f"historical scratch selection resolved a non-file: {member}"
            )
        relative = member.relative_to(source).as_posix()
        records.append(
            ContextRecord(
                ordinal=len(records),
                context_id=f"{dataset_id}:{split}:{relative}",
                context_block_id=block_id,
                relative_path=relative,
                label=normalized_map[class_id],
                class_id=class_id,
                source_size_bytes=int(member.stat().st_size),
            )
        )

    if record_order == "block_major":
        iterator = (
            (block_id, class_id, offset, count)
            for block_id, offset, count in windows
            for class_id in filesystem_classes
        )
        for block_id, class_id, offset, count in iterator:
            class_root = source / class_id
            members = [
                name for name in os.listdir(class_root) if not name.startswith(".")
            ]
            members.sort(reverse=bool(reverse_order))
            selected = members[offset : offset + count]
            if len(selected) != count:
                raise ValueError(
                    f"class {class_id!r} has {len(members)} images but block {block_id!r} "
                    f"requires indices [{offset},{offset + count})"
                )
            for name in selected:
                append_record(block_id=block_id, class_id=class_id, name=name)
    else:
        for class_id in filesystem_classes:
            class_root = source / class_id
            members = [
                name for name in os.listdir(class_root) if not name.startswith(".")
            ]
            members.sort(reverse=bool(reverse_order))
            selected: list[tuple[int, int, str, str]] = []
            for block_index, (block_id, offset, count) in enumerate(windows):
                window = members[offset : offset + count]
                if len(window) != count:
                    raise ValueError(
                        f"class {class_id!r} has {len(members)} images but block "
                        f"{block_id!r} requires indices [{offset},{offset + count})"
                    )
                selected.extend(
                    (offset + local_index, block_index, block_id, name)
                    for local_index, name in enumerate(window)
                )
            ranks = [rank for rank, _block_index, _block_id, _name in selected]
            if len(set(ranks)) != len(ranks):
                raise ValueError(
                    "source-major scratch axes require disjoint context-block windows"
                )
            for _rank, _block_index, block_id, name in sorted(selected):
                append_record(block_id=block_id, class_id=class_id, name=name)
    selector = {
        "selection": "rank_image_data_id_folder_order_v1",
        "runtime_loader_contract": SCRATCH_FOLDER_LOADER_CONTRACT,
        "class_order": list(filesystem_classes),
        "class_order_hash": stable_digest({"class_order": filesystem_classes}),
        "reverse_order": bool(reverse_order),
        "block_windows": [
            {"context_block_id": block_id, "offset": offset, "count": count}
            for block_id, offset, count in windows
        ],
        "class_count": len(normalized_map),
    }
    if record_order == "source_major":
        selector["record_order"] = "historical_class_major_source_order_v1"
    axis = ContextAxis(
        axis_id=str(axis_id),
        dataset_id=str(dataset_id),
        split=str(split),
        preprocessing=preprocessing,
        label_map_hash=stable_digest({"class_map": normalized_map}),
        records=tuple(records),
        selector=selector,
    )
    axis.validate()
    return axis


def scratch_context_axis_order_evidence(
    axis: ContextAxis,
    *,
    image_root: str | Path,
) -> dict[str, str]:
    """Certify source-major identity and block-major execution separately.

    The scratch scorer discovers a class-major source axis, then evaluates
    deterministic context blocks in block-major order. Both orderings contain
    the same coordinates but have distinct scientific roles and hashes.
    """

    class_order = tuple(
        str(value) for value in axis.selector.get("class_order", ()) or ()
    )
    if not class_order or len(set(class_order)) != len(class_order):
        raise ValueError("scratch context axis omits a unique source class order")
    records_by_class = {class_id: [] for class_id in class_order}
    for record in axis.records:
        if record.class_id not in records_by_class:
            raise ValueError("scratch context axis contains an undeclared class")
        records_by_class[record.class_id].append(record)
    root = Path(image_root).resolve()

    def row(record: ContextRecord) -> dict[str, Any]:
        return {
            "path": str(root / record.relative_path),
            "label": int(record.label),
        }

    source_records = tuple(
        row(record)
        for class_id in class_order
        for record in sorted(
            records_by_class[class_id],
            key=lambda item: Path(item.relative_path).name,
            reverse=bool(axis.selector.get("reverse_order", False)),
        )
    )
    execution_records = tuple(row(record) for record in axis.records)
    source_coordinates = tuple((item["path"], item["label"]) for item in source_records)
    execution_coordinates = tuple(
        (item["path"], item["label"]) for item in execution_records
    )
    if len(set(source_coordinates)) != len(source_coordinates):
        raise ValueError("scratch context source coordinates must be unique")
    if set(source_coordinates) != set(execution_coordinates):
        raise ValueError("scratch source and execution axes cover different images")
    source_index = {
        coordinate: index for index, coordinate in enumerate(source_coordinates)
    }
    block_identity = {
        block_id: [
            {
                "index": source_index[(item["path"], item["label"])],
                **item,
            }
            for record, item in zip(axis.records, execution_records, strict=True)
            if record.context_block_id == block_id
        ]
        for block_id in axis.context_block_ids
    }
    if sum(len(rows) for rows in block_identity.values()) != len(axis.records):
        raise RuntimeError("scratch context blocks do not partition the execution axis")
    return {
        "source_records_hash": stable_digest(source_records),
        "execution_records_hash": stable_digest(execution_records),
        "context_block_identity_hash": stable_digest(block_identity),
    }


def build_complete_imagefolder_context_axis(
    root: str | Path,
    *,
    axis_id: str,
    dataset_id: str,
    split: str,
    class_map: Mapping[str, int],
    context_block_id: str,
    role: str,
    preprocessing: ImagePreprocessingSpec,
    reverse_order: bool = False,
    expected_images_per_class: int | None = None,
    ordering_contract: str = "rank_image_data_id_folder_v1",
    runtime_loader_contract: str = SCRATCH_FOLDER_LOADER_CONTRACT,
) -> ContextAxis:
    """Freeze an entire class-folder split under one explicit ordering contract.

    This is the executable counterpart of a complete statistical reservoir or
    untouched validation declaration. Runtime roots remain outside the
    scientific identity; class/member order, every selected relative path, and
    source size are frozen in the resulting axis.
    """

    source = Path(root)
    if not source.is_dir():
        raise NotADirectoryError(source)
    preprocessing.validate()
    normalized_map = {str(key): int(value) for key, value in class_map.items()}
    if not normalized_map or len(set(normalized_map.values())) != len(normalized_map):
        raise ValueError("class map must be nonempty with unique nonnegative labels")
    if any(value < 0 for value in normalized_map.values()):
        raise ValueError("class map labels must be nonnegative")
    block_id = str(context_block_id)
    scientific_role = str(role)
    if not block_id or not scientific_role:
        raise ValueError("complete context axis requires block and scientific role IDs")
    expected = (
        None if expected_images_per_class is None else int(expected_images_per_class)
    )
    if expected is not None and expected <= 0:
        raise ValueError("expected images per class must be positive")

    # ``DirEntry`` caches file metadata on the common local and network
    # filesystems.  A single scan therefore freezes type and byte size without
    # the separate ``is_file`` and ``stat`` calls that made a full ImageNet
    # axis perform more than 2.5 million metadata round trips.
    filesystem_classes = [
        entry.name
        for entry in os.scandir(source)
        if not entry.name.startswith(".")
        and entry.name in normalized_map
        and entry.is_dir(follow_symlinks=True)
    ]
    missing_classes = sorted(set(normalized_map) - set(filesystem_classes))
    if missing_classes:
        raise FileNotFoundError(
            "ImageFolder class directories are missing: "
            + ",".join(missing_classes[:8])
        )
    normalized_ordering = str(ordering_contract).strip().lower()
    if normalized_ordering == "torchvision_imagefolder_v1":
        filesystem_classes = sorted(filesystem_classes)
    elif normalized_ordering != "rank_image_data_id_folder_v1":
        raise ValueError(
            f"unsupported complete-axis ordering contract {ordering_contract!r}"
        )
    loader_contract = str(runtime_loader_contract).strip()
    if loader_contract not in {
        SCRATCH_FOLDER_LOADER_CONTRACT,
        PREPROCESSED_UINT8_BANK_LOADER_CONTRACT,
    }:
        raise ValueError(
            f"unsupported complete-axis loader contract {loader_contract!r}"
        )

    records: list[ContextRecord] = []
    counts: dict[str, int] = {}
    for class_id in filesystem_classes:
        class_root = source / class_id
        members = [
            entry for entry in os.scandir(class_root) if not entry.name.startswith(".")
        ]
        members.sort(key=lambda entry: entry.name, reverse=bool(reverse_order))
        nonfile = next(
            (entry for entry in members if not entry.is_file(follow_symlinks=True)),
            None,
        )
        if nonfile is not None:
            raise ValueError(
                "complete context axis resolved a non-file: "
                f"{class_root / nonfile.name}"
            )
        if expected is not None and len(members) != expected:
            raise ValueError(
                f"class {class_id!r} contains {len(members)} images; expected exactly {expected}"
            )
        counts[class_id] = len(members)
        for member in members:
            relative = f"{class_id}/{member.name}"
            records.append(
                ContextRecord(
                    ordinal=len(records),
                    context_id=f"{dataset_id}:{split}:{relative}",
                    context_block_id=block_id,
                    relative_path=relative,
                    label=normalized_map[class_id],
                    class_id=class_id,
                    source_size_bytes=int(member.stat(follow_symlinks=True).st_size),
                )
            )
    axis = ContextAxis(
        axis_id=str(axis_id),
        dataset_id=str(dataset_id),
        split=str(split),
        preprocessing=preprocessing,
        label_map_hash=stable_digest({"class_map": normalized_map}),
        records=tuple(records),
        selector={
            "selection": (
                "complete_torchvision_imagefolder_order_v1"
                if normalized_ordering == "torchvision_imagefolder_v1"
                else "complete_rank_image_data_id_folder_order_v1"
            ),
            "runtime_loader_contract": loader_contract,
            "ordering_contract": normalized_ordering,
            "context_block_id": block_id,
            "role": scientific_role,
            "class_order": list(filesystem_classes),
            "class_order_hash": stable_digest({"class_order": filesystem_classes}),
            "class_count": len(filesystem_classes),
            "per_class_count_hash": stable_digest({"counts": counts}),
            "expected_images_per_class": expected,
            "reverse_order": bool(reverse_order),
            "complete_split": True,
        },
    )
    axis.validate()
    return axis


class _ContextDataset(_DatasetBase):  # type: ignore[misc]
    def __init__(
        self,
        root: Path,
        records: Sequence[ContextRecord],
        preprocessing: ImagePreprocessingSpec,
    ):
        _require_torchvision()
        self.root = root
        self.records = tuple(records)
        self.pil_rgb = preprocessing.notes.get("physical_decoder") == "pil_rgb"
        self.transform = preprocessing.build_transform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        record = self.records[int(index)]
        source = self.root / record.relative_path
        if (
            not source.is_file()
            or int(source.stat().st_size) != record.source_size_bytes
        ):
            raise ValueError(
                f"context image replica differs from its frozen coordinate: {source}"
            )
        if self.pil_rgb:
            from PIL import Image

            with Image.open(source) as decoded:
                image = self.transform(decoded.convert("RGB"))
            return image, int(record.label), record.context_id
        image = read_image(str(source), mode=ImageReadMode.RGB)
        return self.transform(image), int(record.label), record.context_id


class ImageFolderContextSource:
    """Runtime image replica for one immutable context axis."""

    source_id = "canonical-imagefolder-context-source-v1"

    def __init__(
        self,
        axis: ContextAxis,
        root: str | Path,
        *,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        _require_torchvision()
        axis.validate()
        self.axis = axis
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        if int(batch_size) <= 0 or int(num_workers) < 0:
            raise ValueError("context source batch size/workers are invalid")
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

    def _batches(
        self, records: Sequence[ContextRecord]
    ) -> Iterable[MeasurementInputBatch]:
        dataset = _ContextDataset(self.root, records, self.axis.preprocessing)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # The panel source constructs a new loader for every sealed
            # context prefix.  Persistent workers cannot be reused across
            # those loader instances and instead retain one process pool per
            # live generator, multiplying route-state memory during adaptive
            # evaluation.
            persistent_workers=False,
        )
        for images, labels, context_ids in loader:
            yield MeasurementInputBatch(
                context_ids=tuple(str(value) for value in context_ids),
                inputs=images,
                labels=labels,
            )

    def measurement_batches(
        self,
        plan: MeasurementPlan,
        *,
        context_block_id: str,
    ) -> Iterable[MeasurementInputBatch]:
        if plan.measurement_view.view_hash != self.axis.data_view().view_hash:
            raise ValueError("MeasurementPlan DataView differs from the context axis")
        if context_block_id not in self.axis.context_block_ids:
            raise ValueError(f"unknown measurement context block {context_block_id!r}")
        records = tuple(
            record
            for record in self.axis.records
            if record.context_block_id == context_block_id
        )
        return self._batches(records)

    def circuit_batches(
        self, view: DataView, *, query_id: str
    ) -> Iterable[MeasurementInputBatch]:
        del query_id
        if view.view_hash != self.axis.data_view().view_hash:
            raise ValueError("CircuitPlan DataView differs from the context axis")
        return self._batches(self.axis.records)

    def panel_batches(
        self, context_ids: Sequence[str]
    ) -> Iterable[MeasurementInputBatch]:
        """Read an ordered sampled panel from the frozen context axis."""

        requested = tuple(str(value) for value in context_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("sampled panel requires unique context IDs")
        by_id = {record.context_id: record for record in self.axis.records}
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise ValueError(
                "sampled panel contains unknown contexts: " + ",".join(missing)
            )
        return self._batches(tuple(by_id[context_id] for context_id in requested))


def _to_uint8_chw(images: Any) -> Any:
    tensor = torch.as_tensor(images)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"expected rank-4 image batch, got {tuple(tensor.shape)}")
    if tensor.shape[-1] in (1, 3) and tensor.shape[1] not in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.dtype != torch.uint8:
        if tensor.is_floating_point():
            tensor = tensor.mul(255.0)
        tensor = tensor.round().clamp_(0, 255).to(dtype=torch.uint8)
    return tensor.contiguous()


class _ScratchUint8Dataset(_DatasetBase):  # type: ignore[misc]
    def __init__(
        self,
        root: Path,
        records: Sequence[ContextRecord],
        preprocessing: ImagePreprocessingSpec,
    ) -> None:
        _require_torchvision()
        self.root = root
        self.records = tuple(records)
        self.pil_rgb = preprocessing.notes.get("physical_decoder") == "pil_rgb"
        self.transform = preprocessing.build_uint8_prefix_transform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        record = self.records[int(index)]
        source = self.root / record.relative_path
        if (
            not source.is_file()
            or int(source.stat().st_size) != record.source_size_bytes
        ):
            raise ValueError(
                f"context image replica differs from its frozen coordinate: {source}"
            )
        if self.pil_rgb:
            from PIL import Image

            with Image.open(source) as decoded:
                image = self.transform(decoded.convert("RGB"))
            return image, int(record.label), record.context_id
        image = read_image(str(source), mode=ImageReadMode.RGB)
        return self.transform(image), int(record.label), record.context_id


class ScratchFolderContextSource:
    """Historical scratch-materialized uint8 loader with online device decode."""

    source_id = "canonical-scratch-folder-context-source-v1"

    def __init__(
        self,
        axis: ContextAxis,
        root: str | Path,
        *,
        batch_size: int,
        device: Any = "cpu",
        storage_device: Any = "cpu",
        num_workers: int = 0,
        pin_memory: bool = True,
        _require_scratch_contract: bool = True,
    ) -> None:
        _require_torchvision()
        axis.validate()
        if (
            _require_scratch_contract
            and str(axis.selector.get("runtime_loader_contract", ""))
            != SCRATCH_FOLDER_LOADER_CONTRACT
        ):
            raise ValueError(
                "context axis does not declare the scratch-folder loader contract"
            )
        self.axis = axis
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        if int(batch_size) <= 0 or int(num_workers) < 0:
            raise ValueError("context source batch size/workers are invalid")
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self._records_by_context_id = {
            record.context_id: record for record in self.axis.records
        }
        self._bundle_cache: OrderedDict[tuple[str, ...], dict[str, Any]] = OrderedDict()
        # One entry is sufficient for paired committed/candidate reuse.  A
        # second entry retains the previous nested stage without allowing the
        # stochastic path to accumulate an unbounded uint8 image bank.
        self._bundle_cache_limit = 2

    def _materialize(self, records: Sequence[ContextRecord]) -> dict[str, Any]:
        key = tuple(record.context_id for record in records)
        cached = self._bundle_cache.get(key)
        if cached is not None:
            self._bundle_cache.move_to_end(key)
            return cached
        dataset = _ScratchUint8Dataset(self.root, records, self.axis.preprocessing)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # Materialization is a one-shot pass; keeping this loader's
            # workers alive provides no reuse and can overlap worker pools
            # across nested-prefix requests.
            persistent_workers=False,
        )
        batches = iter(loader)
        try:
            first_images, first_labels, first_ids = next(batches)
        except StopIteration as error:
            raise ValueError(
                "scratch-folder context source selected no images"
            ) from error
        first_images = _to_uint8_chw(first_images)
        total = len(records)
        channels, height, width = map(int, first_images.shape[1:])
        inputs = torch.empty(
            (total, channels, height, width),
            dtype=torch.uint8,
            device=self.storage_device,
        ).contiguous(memory_format=torch.channels_last)
        labels = torch.empty((total,), dtype=torch.long, device=self.storage_device)
        context_ids: list[str] = []
        cursor = 0

        def copy_batch(
            batch_images: Any, batch_labels: Any, batch_ids: Sequence[str]
        ) -> None:
            nonlocal cursor
            image_tensor = _to_uint8_chw(batch_images)
            label_tensor = torch.as_tensor(batch_labels, dtype=torch.long).reshape(-1)
            count = int(label_tensor.numel())
            end = cursor + count
            if end > total or len(batch_ids) != count:
                raise ValueError(
                    "scratch-folder materialization changed its frozen batch axis"
                )
            moved = image_tensor.to(
                device=self.storage_device,
                dtype=torch.uint8,
                memory_format=torch.channels_last,
                non_blocking=True,
            )
            inputs[cursor:end].copy_(moved)
            labels[cursor:end].copy_(
                label_tensor.to(device=self.storage_device, non_blocking=True)
            )
            context_ids.extend(str(value) for value in batch_ids)
            cursor = end

        copy_batch(first_images, first_labels, first_ids)
        for images, batch_labels, batch_ids in batches:
            copy_batch(images, batch_labels, batch_ids)
        if cursor != total or tuple(context_ids) != key:
            raise ValueError(
                "scratch-folder materialization differs from the frozen context axis"
            )
        decode_scale, decode_bias = (
            self.axis.preprocessing.uint8_online_decode_parameters(
                num_channels=channels
            )
        )
        result = {
            "inputs": inputs,
            "labels": labels,
            "context_ids": key,
            "decode_scale": decode_scale,
            "decode_bias": decode_bias,
        }
        self._bundle_cache[key] = result
        self._bundle_cache.move_to_end(key)
        while len(self._bundle_cache) > self._bundle_cache_limit:
            self._bundle_cache.popitem(last=False)
        return result

    def _batches(
        self, records: Sequence[ContextRecord]
    ) -> Iterable[MeasurementInputBatch]:
        bundle = self._materialize(records)
        total = len(bundle["context_ids"])
        scale = bundle["decode_scale"].to(device=self.device, dtype=torch.float32)
        bias = bundle["decode_bias"].to(device=self.device, dtype=torch.float32)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            images = bundle["inputs"][start:end].to(
                device=self.device,
                dtype=torch.float32,
                memory_format=torch.channels_last,
                non_blocking=True,
            )
            images.mul_(scale).add_(bias)
            yield MeasurementInputBatch(
                context_ids=tuple(bundle["context_ids"][start:end]),
                inputs=images,
                labels=bundle["labels"][start:end],
            )

    def measurement_batches(
        self,
        plan: MeasurementPlan,
        *,
        context_block_id: str,
    ) -> Iterable[MeasurementInputBatch]:
        if plan.measurement_view.view_hash != self.axis.data_view().view_hash:
            raise ValueError("MeasurementPlan DataView differs from the context axis")
        if context_block_id not in self.axis.context_block_ids:
            raise ValueError(f"unknown measurement context block {context_block_id!r}")
        records = tuple(
            record
            for record in self.axis.records
            if record.context_block_id == context_block_id
        )
        return self._batches(records)

    def circuit_batches(
        self, view: DataView, *, query_id: str
    ) -> Iterable[MeasurementInputBatch]:
        del query_id
        if view.view_hash != self.axis.data_view().view_hash:
            raise ValueError("CircuitPlan DataView differs from the context axis")
        return self._batches(self.axis.records)

    def panel_batches(
        self, context_ids: Sequence[str]
    ) -> Iterable[MeasurementInputBatch]:
        """Read an ordered sampled panel through the uint8 scoring path."""

        requested = tuple(str(value) for value in context_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("sampled panel requires unique context IDs")
        missing = sorted(set(requested) - set(self._records_by_context_id))
        if missing:
            raise ValueError(
                "sampled panel contains unknown contexts: " + ",".join(missing)
            )
        return self._batches(
            tuple(self._records_by_context_id[context_id] for context_id in requested)
        )


class ClassIndexedUint8ContextSource(ScratchFolderContextSource):
    """Production pruning reservoir with uint8 storage and online float decode."""

    source_id = "canonical-class-indexed-uint8-context-source-v1"

    def __init__(
        self,
        axis: ContextAxis,
        root: str | Path,
        *,
        batch_size: int,
        device: Any = "cpu",
        storage_device: Any = "cpu",
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__(
            axis,
            root,
            batch_size=batch_size,
            device=device,
            storage_device=storage_device,
            num_workers=num_workers,
            pin_memory=pin_memory,
            _require_scratch_contract=False,
        )


def _uint8_bank_payload_paths(manifest_path: str | Path) -> tuple[Path, Path, Path]:
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    # Cache manifests may retain an adoption-time absolute path. The payloads
    # beside the immutable manifest are the authoritative runtime replica.
    inputs = manifest.parent / "inputs.npy"
    labels = manifest.parent / "labels.npy"
    if not inputs.is_file() or not labels.is_file():
        raise FileNotFoundError(f"uint8-bank payloads are incomplete beside {manifest}")
    return _stage_bank_locally(manifest, inputs, labels)


def _stage_bank_locally(
    manifest: Path, inputs: Path, labels: Path
) -> tuple[Path, Path, Path]:
    """Optional node-local staging of the uint8 bank.

    With ``CIRCUIT_NVME_STAGE_DIR`` set, the three payloads are copied
    once per node (single writer via an atomic lock directory; readers
    wait on the done marker) and reads mmap the local replica. Bytes are
    verified by size against the source; any mismatch, timeout, or copy
    failure falls back to the source paths — staging is runtime-only
    and can never change content (same bytes or not used).
    """

    stage_root = os.environ.get("CIRCUIT_NVME_STAGE_DIR", "").strip()
    if not stage_root:
        return manifest, inputs, labels
    import hashlib
    import shutil
    import time as _time

    key = hashlib.sha256(str(manifest.resolve()).encode()).hexdigest()[:16]
    target = Path(stage_root) / f"uint8_bank_{manifest.parent.name}_{key}"
    done = target / ".staged_done"
    sources = {"manifest.json": manifest, "inputs.npy": inputs, "labels.npy": labels}

    def _verified() -> tuple[Path, Path, Path] | None:
        if not done.is_file():
            return None
        for name, src in sources.items():
            replica = target / name
            if not replica.is_file() or replica.stat().st_size != src.stat().st_size:
                return None
        return target / "manifest.json", target / "inputs.npy", target / "labels.npy"

    staged = _verified()
    if staged is not None:
        return staged
    lock = target.parent / (target.name + ".lock")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        lock.mkdir()  # atomic: exactly one stager per node
    except FileExistsError:
        deadline = _time.monotonic() + float(
            os.environ.get("CIRCUIT_NVME_STAGE_TIMEOUT", "1800")
        )
        while _time.monotonic() < deadline:
            staged = _verified()
            if staged is not None:
                return staged
            if not lock.exists():  # stager died or finished badly
                break
            _time.sleep(2.0)
        return manifest, inputs, labels
    except OSError:
        return manifest, inputs, labels
    try:
        target.mkdir(parents=True, exist_ok=True)
        for name, src in sources.items():
            tmp = target / (name + ".part")
            shutil.copyfile(src, tmp)
            os.replace(tmp, target / name)
        done.touch()
        staged = _verified()
        return staged if staged is not None else (manifest, inputs, labels)
    except OSError:
        return manifest, inputs, labels
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


def _uint8_bank_probe_hash(
    *,
    axis: ContextAxis,
    inputs: np.ndarray,
    probe_count: int,
) -> tuple[int, str]:
    """Hash deterministic certified rows without scanning the complete bank."""

    requested = max(1, int(probe_count))
    probe_indices = np.unique(
        np.linspace(0, max(0, len(axis.records) - 1), requested, dtype=np.int64)
    )
    rows = []
    for raw_index in probe_indices.tolist():
        record = axis.records[int(raw_index)]
        cached = np.asarray(inputs[int(raw_index)])
        rows.append(
            {
                "ordinal": int(raw_index),
                "context_id": record.context_id,
                "pixel_sha256": hashlib.sha256(cached.tobytes()).hexdigest(),
            }
        )
    return len(rows), stable_digest({"pixel_probes": rows})


def _preprocessing_manifest_failures(
    preprocessing: ImagePreprocessingSpec,
    manifest: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    expected_resize = int(preprocessing.resize_size[0])
    expected_crop = int(
        preprocessing.center_crop_size[0]
        if preprocessing.center_crop_size
        else preprocessing.resize_size[-1]
    )
    if int(manifest.get("resize", -1)) != expected_resize:
        failures.append("resize_mismatch")
    if int(manifest.get("crop", -1)) != expected_crop:
        failures.append("crop_mismatch")
    # Version-1 cache manifests predate explicit interpolation fields. Their
    # materializer's declared default was bilinear/antialias=True.
    interpolation = str(manifest.get("interpolation", "bilinear"))
    antialias = bool(manifest.get("antialias", True))
    if interpolation != preprocessing.interpolation:
        failures.append("interpolation_mismatch")
    if antialias != preprocessing.antialias:
        failures.append("antialias_mismatch")
    if str(manifest.get("dtype", "")) != "uint8":
        failures.append("dtype_mismatch")
    if str(manifest.get("layout", "")) != "nchw":
        failures.append("layout_mismatch")
    mean = np.asarray(manifest.get("mean", ()) or (), dtype=np.float64)
    std = np.asarray(manifest.get("std", ()) or (), dtype=np.float64)
    if mean.shape != (3,) or not np.allclose(
        mean, np.asarray(preprocessing.mean), atol=1e-7, rtol=0.0
    ):
        failures.append("mean_mismatch")
    if std.shape != (3,) or not np.allclose(
        std, np.asarray(preprocessing.std), atol=1e-7, rtol=0.0
    ):
        failures.append("std_mismatch")
    return failures


def certify_preprocessed_uint8_bank_replica(
    *,
    axis: ContextAxis,
    image_root: str | Path,
    cache_manifest_path: str | Path,
    pixel_probe_count: int = 16,
) -> PreprocessedUint8BankReplicaReceipt:
    """Deep-certify one cache as an exact replica of a frozen context axis."""

    _require_torchvision()
    axis.validate()
    manifest_path, inputs_path, labels_path = _uint8_bank_payload_paths(
        cache_manifest_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = _preprocessing_manifest_failures(axis.preprocessing, manifest)
    if str(manifest.get("schema_name", "")) != "canonical_imagefolder_tensor_cache":
        failures.append("manifest_schema_mismatch")
    if axis.selector.get("ordering_contract") != "torchvision_imagefolder_v1":
        failures.append("axis_ordering_contract_mismatch")
    if axis.selector.get("runtime_loader_contract") != (
        PREPROCESSED_UINT8_BANK_LOADER_CONTRACT
    ):
        failures.append("axis_loader_contract_mismatch")

    inputs = np.load(inputs_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    crop = axis.preprocessing.center_crop_size or axis.preprocessing.resize_size
    crop_height = int(crop[0])
    crop_width = int(crop[-1])
    expected_shape = (len(axis.records), 3, crop_height, crop_width)
    if tuple(inputs.shape) != expected_shape:
        failures.append("input_shape_mismatch")
    if inputs.dtype != np.uint8:
        failures.append("input_dtype_mismatch")
    if tuple(labels.shape) != (len(axis.records),):
        failures.append("label_shape_mismatch")
    axis_labels = np.fromiter(
        (int(record.label) for record in axis.records),
        dtype=np.int64,
        count=len(axis.records),
    )
    if labels.dtype.kind not in {"i", "u"} or not np.array_equal(labels, axis_labels):
        failures.append("ordered_label_axis_mismatch")

    requested_probes = max(1, int(pixel_probe_count))
    probe_indices = np.unique(
        np.linspace(0, max(0, len(axis.records) - 1), requested_probes, dtype=np.int64)
    )
    probe_rows: list[dict[str, Any]] = []
    root = Path(image_root)
    if not root.is_dir():
        failures.append("image_root_missing")
    elif not failures or set(failures).issubset({"mean_mismatch", "std_mismatch"}):
        from PIL import Image
        from torchvision import transforms

        interpolation = {
            "nearest": transforms.InterpolationMode.NEAREST,
            "bilinear": transforms.InterpolationMode.BILINEAR,
            "bicubic": transforms.InterpolationMode.BICUBIC,
        }[axis.preprocessing.interpolation]
        transform = transforms.Compose(
            [
                transforms.Resize(
                    axis.preprocessing.resize_size[0],
                    interpolation=interpolation,
                    antialias=axis.preprocessing.antialias,
                ),
                transforms.CenterCrop(axis.preprocessing.center_crop_size[0]),
                transforms.ToTensor(),
            ]
        )
        for raw_index in probe_indices.tolist():
            record = axis.records[int(raw_index)]
            source = root / record.relative_path
            if (
                not source.is_file()
                or int(source.stat().st_size) != record.source_size_bytes
            ):
                failures.append(f"pixel_probe_source_mismatch:{raw_index}")
                continue
            with Image.open(source) as image:
                tensor = transform(image.convert("RGB"))
            quantized = (
                tensor.mul(255.0).round().clamp_(0, 255).to(dtype=torch.uint8).numpy()
            )
            cached = np.asarray(inputs[int(raw_index)])
            if not np.array_equal(quantized, cached):
                failures.append(f"pixel_probe_mismatch:{raw_index}")
            probe_rows.append(
                {
                    "ordinal": int(raw_index),
                    "context_id": record.context_id,
                    "pixel_sha256": hashlib.sha256(cached.tobytes()).hexdigest(),
                }
            )

    return PreprocessedUint8BankReplicaReceipt(
        axis_hash=axis.content_hash,
        preprocessing_hash=axis.preprocessing.content_hash,
        manifest_sha256=sha256_file(manifest_path),
        inputs_sha256=sha256_file(inputs_path),
        labels_sha256=sha256_file(labels_path),
        inputs_size_bytes=int(inputs_path.stat().st_size),
        labels_size_bytes=int(labels_path.stat().st_size),
        shape=tuple(int(value) for value in inputs.shape),
        dtype=str(inputs.dtype),
        layout=str(manifest.get("layout", "")),
        ordering_contract=str(axis.selector.get("ordering_contract", "")),
        pixel_probe_count=len(probe_rows),
        pixel_probe_hash=stable_digest({"pixel_probes": probe_rows}),
        passed=not failures,
        failures=tuple(sorted(set(failures))),
    )


def rebind_preprocessed_uint8_bank_receipt(
    *,
    source_axis_manifest: str | Path,
    target_axis_manifest: str | Path,
    cache_manifest_path: str | Path,
    source_receipt_path: str | Path,
    target_receipt_path: str | Path,
    evidence_path: str | Path,
) -> dict[str, Any]:
    """Rebind a certified physical bank to an exactly equal child axis.

    Context-axis IDs and selector provenance are scientific metadata. They can
    change while the ordered physical coordinates, labels, and preprocessing
    remain exactly equal. In that case rehashing the immutable image payload is
    unnecessary: the source receipt remains the byte authority and this proof
    binds it to the child axis without weakening any content check.
    """

    source = ContextAxis.load(source_axis_manifest)
    target = ContextAxis.load(target_axis_manifest)
    receipt = PreprocessedUint8BankReplicaReceipt.load(source_receipt_path)
    receipt.require_passed(axis=source, manifest_path=cache_manifest_path)

    issues: list[str] = []
    if target.selector.get("axis_transform") != (
        "ordered_coordinates_preserved_preprocessing_rebind_v1"
    ):
        issues.append("target_axis_transform_not_declared")
    if target.selector.get("parent_context_axis_hash") != source.content_hash:
        issues.append("target_parent_axis_hash_mismatch")
    if target.selector.get("parent_preprocessing_hash") != (
        source.preprocessing.content_hash
    ):
        issues.append("target_parent_preprocessing_hash_mismatch")
    for field_name in ("dataset_id", "split", "label_map_hash"):
        if getattr(source, field_name) != getattr(target, field_name):
            issues.append(f"axis_{field_name}_mismatch")
    if source.preprocessing != target.preprocessing:
        issues.append("axis_preprocessing_mismatch")
    if source.records != target.records:
        issues.append("ordered_context_records_mismatch")
    if source.ids_hash != target.ids_hash:
        issues.append("ordered_context_ids_mismatch")
    if issues:
        raise ValueError(
            "uint8-bank axis receipt rebind failed: " + ",".join(sorted(set(issues)))
        )

    target_receipt = replace(
        receipt,
        axis_hash=target.content_hash,
        preprocessing_hash=target.preprocessing.content_hash,
    )
    target_receipt.write(target_receipt_path)
    core = {
        "schema_name": "preprocessed_uint8_bank_axis_rebind_v1",
        "source_axis_hash": source.content_hash,
        "target_axis_hash": target.content_hash,
        "ordered_context_ids_hash": target.ids_hash,
        "context_count": len(target.records),
        "preprocessing_hash": target.preprocessing.content_hash,
        "source_receipt_hash": receipt.content_hash,
        "target_receipt_hash": target_receipt.content_hash,
        "cache_manifest_sha256": receipt.manifest_sha256,
        "ordered_coordinates_equal": True,
        "preprocessing_equal": True,
        "physical_payload_rehashed": False,
        "passed": True,
    }
    evidence = {**core, "content_hash": stable_digest(core)}
    destination = Path(evidence_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **evidence,
        "target_receipt": str(target_receipt_path),
        "evidence_path": str(destination),
    }


class PreprocessedUint8BankContextSource:
    """Class-indexed uint8 bank with a separately certified scientific axis."""

    source_id = "canonical-preprocessed-uint8-bank-context-source-v1"

    def __init__(
        self,
        axis: ContextAxis,
        cache_manifest_path: str | Path,
        replica_receipt_path: str | Path,
        *,
        batch_size: int,
        device: Any = "cpu",
        preload_to_ram: bool = True,
        pin_memory: bool = True,
        verification_mode: str = "deep_sha256",
    ) -> None:
        _require_torchvision()
        axis.validate()
        source_manifest_path = Path(cache_manifest_path).resolve()
        staging_started = time.perf_counter()
        manifest_path, inputs_path, labels_path = _uint8_bank_payload_paths(
            cache_manifest_path
        )
        receipt = PreprocessedUint8BankReplicaReceipt.load(replica_receipt_path)
        receipt.require_passed(axis=axis, manifest_path=manifest_path)

        def _verify_replica(staged_inputs_path: Path, staged_labels_path: Path) -> None:
            if sha256_file(staged_labels_path) != receipt.labels_sha256:
                raise ValueError("staged uint8-bank label checksum changed")
            staged_inputs = np.load(staged_inputs_path, mmap_mode="r")
            probe_count, probe_hash = _uint8_bank_probe_hash(
                axis=axis,
                inputs=staged_inputs,
                probe_count=receipt.pixel_probe_count,
            )
            if probe_count != receipt.pixel_probe_count:
                raise ValueError("staged uint8-bank probe count changed")
            if probe_hash != receipt.pixel_probe_hash:
                raise ValueError("staged uint8-bank pixel probes changed")

        # Opt-in: with
        # IRG_BANK_ROW_STREAMING=on|required and a per-row digest table bound
        # to this bank's certified whole-file SHA-256, no replica is copied.
        # Rows are read in place from the certified file, only when requested,
        # and each is verified against its digest at read time (see
        # bank_row_digests.py).  ``on`` falls back to node-local staging when
        # the table is unusable; ``required`` refuses.  The default is off and
        # leaves this constructor byte-for-byte on the staging path.
        from mechanistic_systems.measurement.bank_row_digests import (
            VerifiedRowBank,
            load_row_digest_table,
            resolve_row_streaming_request,
            row_digest_table_dir,
        )

        streaming_request = resolve_row_streaming_request()
        streaming_bank: VerifiedRowBank | None = None
        streaming_refusal: str | None = None
        if streaming_request != "off":
            try:
                if str(verification_mode) == "deep_sha256":
                    raise ValueError("deep_sha256 verification reads the whole bank")
                if bool(preload_to_ram):
                    raise ValueError("preload_to_ram requests a full in-memory bank")
                if int(Path(inputs_path).stat().st_size) != int(
                    receipt.inputs_size_bytes
                ):
                    raise ValueError("certified uint8-bank payload size changed")
                table_dir = row_digest_table_dir(inputs_path)
                digests, table_receipt = load_row_digest_table(
                    table_dir,
                    inputs_path=inputs_path,
                    inputs_sha256=receipt.inputs_sha256,
                )
                streaming_bank = VerifiedRowBank(
                    inputs_path, digests=digests, receipt=table_receipt
                )
            except Exception as exc:
                if streaming_request == "required":
                    raise
                streaming_refusal = f"{type(exc).__name__}:{exc}"
        if streaming_bank is None:
            inputs_path, labels_path, node_local_bank_staging = resolve_staged_bank(
                inputs_path,
                labels_path,
                verify_replica=_verify_replica,
                source_identity={
                    "inputs_sha256": receipt.inputs_sha256,
                    "labels_sha256": receipt.labels_sha256,
                    "inputs_size_bytes": receipt.inputs_size_bytes,
                    "labels_size_bytes": receipt.labels_size_bytes,
                },
            )
            if streaming_request != "off":
                node_local_bank_staging = {
                    **node_local_bank_staging,
                    "row_streaming": {
                        "requested": streaming_request,
                        "active": False,
                        "refusal": streaming_refusal,
                    },
                }
        else:
            node_local_bank_staging = {
                "schema_name": "irg_node_local_bank_staging_v2",
                "staging_mode": "row_streamed_verified",
                "required_mode": False,
                "staged_path": None,
                "replica_reused": False,
                "copy_seconds": 0.0,
                "bank_bytes": int(receipt.inputs_size_bytes)
                + int(receipt.labels_size_bytes),
                "fallback_reason": None,
                "row_streaming": {
                    "requested": streaming_request,
                    "active": True,
                    "table_dir": str(row_digest_table_dir(inputs_path)),
                    "table_receipt_content_sha256": str(
                        streaming_bank.receipt.get("content_sha256")
                    ),
                    "table_sha256": str(streaming_bank.receipt.get("table_sha256")),
                    "inputs_sha256": str(streaming_bank.receipt.get("inputs_sha256")),
                    "row_integrity": "sha256_per_row_at_every_read_v1",
                },
            }
        staging_elapsed_seconds = time.perf_counter() - staging_started
        for path, expected_size in (
            (inputs_path, receipt.inputs_size_bytes),
            (labels_path, receipt.labels_size_bytes),
        ):
            if int(path.stat().st_size) != int(expected_size):
                raise ValueError(f"certified uint8-bank payload size changed: {path}")
        resolved_verification_mode = str(verification_mode)
        if resolved_verification_mode not in UINT8_BANK_RUNTIME_VERIFICATION_MODES:
            raise ValueError(
                "unsupported uint8-bank runtime verification mode: "
                f"{resolved_verification_mode}"
            )
        mmap_mode = None if bool(preload_to_ram) else "r"
        inputs = (
            np.load(inputs_path, mmap_mode=mmap_mode)
            if streaming_bank is None
            else streaming_bank
        )
        labels = np.load(labels_path, mmap_mode=mmap_mode)
        if tuple(inputs.shape) != receipt.shape or str(inputs.dtype) != receipt.dtype:
            raise ValueError(
                "uint8-bank payload shape or dtype differs from its receipt"
            )
        if resolved_verification_mode == "deep_sha256":
            if sha256_file(inputs_path) != receipt.inputs_sha256:
                raise ValueError("certified uint8-bank input checksum changed")
            if sha256_file(labels_path) != receipt.labels_sha256:
                raise ValueError("certified uint8-bank label checksum changed")
            runtime_probe_hash = receipt.pixel_probe_hash
        else:
            if sha256_file(labels_path) != receipt.labels_sha256:
                raise ValueError("certified uint8-bank label checksum changed")
            runtime_probe_count, runtime_probe_hash = _uint8_bank_probe_hash(
                axis=axis,
                inputs=inputs,
                probe_count=receipt.pixel_probe_count,
            )
            if runtime_probe_count != receipt.pixel_probe_count:
                raise ValueError("certified uint8-bank probe count changed")
            if runtime_probe_hash != receipt.pixel_probe_hash:
                raise ValueError("certified uint8-bank pixel probes changed")
        if int(batch_size) <= 0:
            raise ValueError("uint8-bank batch size must be positive")
        self.axis = axis
        self.source_cache_manifest_path = source_manifest_path
        self.cache_manifest_path = manifest_path
        self.node_local_staged = (
            manifest_path.resolve() != source_manifest_path
            or node_local_bank_staging.get("staging_mode") == "staged"
        )
        self.staging_elapsed_seconds = float(staging_elapsed_seconds)
        self.node_local_bank_staging: dict[str, Any] = dict(node_local_bank_staging)
        # One sequential page-granular pass over the staged payload turns
        # later random 224x224 row faults into already-resident pages. Only
        # meaningful after node-local staging; opt-in because a single-unit
        # canary should not read the whole bank to warm rows it never
        # touches (the working-set prefetch covers that case).
        self.page_warm_performed = False
        self.page_warm_elapsed_seconds = 0.0
        if (
            self.node_local_staged
            and mmap_mode == "r"
            and os.environ.get("CIRCUIT_NVME_PAGE_WARM", "")
            not in {"", "0", "false", "False"}
        ):
            warm_started = time.perf_counter()
            flat_view = inputs.reshape(-1)
            _ = np.asarray(flat_view[:: 4096 // max(1, inputs.dtype.itemsize)]).sum(
                dtype=np.int64
            )
            self.page_warm_elapsed_seconds = float(time.perf_counter() - warm_started)
            self.page_warm_performed = True
        self.inputs_payload_path = inputs_path
        self.labels_payload_path = labels_path
        self.replica_receipt = receipt
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.preload_to_ram = bool(preload_to_ram)
        self.pin_memory = bool(pin_memory)
        self.verification_mode = resolved_verification_mode
        self.runtime_probe_hash = runtime_probe_hash
        self._inputs = inputs
        self._labels = labels
        axis_labels = np.fromiter(
            (int(record.label) for record in axis.records),
            dtype=np.int64,
            count=len(axis.records),
        )
        if not np.array_equal(self._labels, axis_labels):
            raise ValueError("uint8-bank ordered labels differ from its ContextAxis")
        self._records_by_context_id = {
            record.context_id: record for record in self.axis.records
        }
        self._bundle_cache: OrderedDict[tuple[str, ...], dict[str, Any]] = OrderedDict()
        self._bundle_cache_limit = 2

    def _materialize(self, records: Sequence[ContextRecord]) -> dict[str, Any]:
        key = tuple(record.context_id for record in records)
        cached = self._bundle_cache.get(key)
        if cached is not None:
            self._bundle_cache.move_to_end(key)
            return cached
        indices = np.fromiter(
            (int(record.ordinal) for record in records),
            dtype=np.int64,
            count=len(records),
        )
        inputs = torch.from_numpy(np.asarray(self._inputs[indices]).copy()).contiguous(
            memory_format=torch.channels_last
        )
        labels = torch.from_numpy(
            np.asarray(self._labels[indices]).astype(np.int64, copy=True)
        )
        if self.pin_memory and self.device.type == "cuda":
            try:
                inputs = inputs.pin_memory()
                labels = labels.pin_memory()
            except RuntimeError:
                pass
        scale, bias = self.axis.preprocessing.uint8_online_decode_parameters(
            num_channels=int(inputs.shape[1])
        )
        result = {
            "inputs": inputs,
            "labels": labels,
            "context_ids": key,
            "decode_scale": scale,
            "decode_bias": bias,
        }
        self._bundle_cache[key] = result
        self._bundle_cache.move_to_end(key)
        while len(self._bundle_cache) > self._bundle_cache_limit:
            self._bundle_cache.popitem(last=False)
        return result

    def enter_raw_panel_delivery(self) -> bool:
        """Raw-space uint8 panels (the folded-stem transaction; see
        ``bind_uint8_folded_stem``): the normalize is elided and the
        folded first conv owns the decode algebra.  Must flip before any
        serving; the bundle cache is dropped defensively."""

        self._raw_panel_delivery = True
        self._bundle_cache.clear()
        return True

    def exit_raw_panel_delivery(self) -> None:
        self._raw_panel_delivery = False
        self._bundle_cache.clear()

    def _batches(
        self, records: Sequence[ContextRecord]
    ) -> Iterable[MeasurementInputBatch]:
        bundle = self._materialize(records)
        total = len(bundle["context_ids"])
        scale = bundle["decode_scale"].to(device=self.device, dtype=torch.float32)
        bias = bundle["decode_bias"].to(device=self.device, dtype=torch.float32)
        raw_delivery = bool(getattr(self, "_raw_panel_delivery", False))
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            images = bundle["inputs"][start:end]
            images = images.to(
                device=self.device,
                dtype=torch.float32,
                memory_format=torch.channels_last,
                non_blocking=True,
            )
            if not raw_delivery:
                images.mul_(scale).add_(bias)
            yield MeasurementInputBatch(
                context_ids=tuple(bundle["context_ids"][start:end]),
                inputs=images,
                labels=bundle["labels"][start:end],
            )

    def circuit_batches(
        self, view: DataView, *, query_id: str
    ) -> Iterable[MeasurementInputBatch]:
        del query_id
        if view.view_hash != self.axis.data_view().view_hash:
            raise ValueError("CircuitPlan DataView differs from the uint8 bank axis")
        return self._batches(self.axis.records)

    def panel_batches(
        self, context_ids: Sequence[str]
    ) -> Iterable[MeasurementInputBatch]:
        requested = tuple(str(value) for value in context_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("sampled panel requires unique context IDs")
        missing = sorted(set(requested) - set(self._records_by_context_id))
        if missing:
            raise ValueError(
                "sampled panel contains unknown contexts: " + ",".join(missing)
            )
        return self._batches(
            tuple(self._records_by_context_id[context_id] for context_id in requested)
        )


class FixedSubsetContextSource:
    """Expose one immutable ordered subset of a larger certified source.

    The parent remains responsible for byte/receipt verification and physical
    delivery.  This adapter changes only the requested coordinate view and
    refuses any batch outside the frozen subset axis.
    """

    source_id = "fixed-subset-of-certified-context-source-v1"

    def __init__(self, axis: ContextAxis, parent: Any) -> None:
        axis.validate()
        parent_axis = getattr(parent, "axis", None)
        if parent_axis is None:
            raise ValueError("fixed subset source requires an axis-bound parent")
        parent_ids = set(parent_axis.context_ids)
        missing = tuple(
            context_id
            for context_id in axis.context_ids
            if context_id not in parent_ids
        )
        if missing:
            raise ValueError(
                "fixed subset source has contexts absent from its certified parent: "
                + ",".join(missing[:8])
            )
        if axis.preprocessing.content_hash != parent_axis.preprocessing.content_hash:
            raise ValueError(
                "fixed subset source preprocessing differs from its certified parent"
            )
        self.axis = axis
        self.parent = parent

    def circuit_batches(
        self, view: DataView, *, query_id: str
    ) -> Iterable[MeasurementInputBatch]:
        del query_id
        if view.view_hash != self.axis.data_view().view_hash:
            raise ValueError("CircuitPlan DataView differs from the fixed subset axis")
        return self.parent.panel_batches(self.axis.context_ids)

    def panel_batches(
        self, context_ids: Sequence[str]
    ) -> Iterable[MeasurementInputBatch]:
        requested = tuple(str(value) for value in context_ids)
        if requested != self.axis.context_ids:
            raise ValueError(
                "fixed subset source requires its complete exact ordered axis"
            )
        return self.parent.panel_batches(requested)


def build_context_source(
    axis: ContextAxis,
    root: str | Path,
    *,
    batch_size: int,
    device: Any = "cpu",
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Any:
    """Construct the runtime source declared by an immutable context axis."""

    if (
        str(axis.selector.get("runtime_loader_contract", ""))
        == SCRATCH_FOLDER_LOADER_CONTRACT
    ):
        return ScratchFolderContextSource(
            axis,
            root,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return ImageFolderContextSource(
        axis,
        root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class ImageFolderMeasurementProvider:
    provider_id = "canonical-imagefolder-measurement-provider-v1"

    def __init__(self, source: Any) -> None:
        self.source = source

    def iter_batches(
        self,
        plan: MeasurementPlan,
        *,
        context_block_id: str,
    ) -> Iterable[MeasurementInputBatch]:
        return self.source.measurement_batches(plan, context_block_id=context_block_id)


class ImageFolderCircuitProvider:
    provider_id = "canonical-imagefolder-circuit-provider-v1"

    def __init__(self, sources_by_view_hash: Mapping[str, Any]) -> None:
        self.sources_by_view_hash = dict(sources_by_view_hash)
        if not self.sources_by_view_hash:
            raise ValueError("circuit provider requires at least one frozen DataView")
        for view_hash, source in self.sources_by_view_hash.items():
            if view_hash != source.axis.data_view().view_hash:
                raise ValueError(
                    "circuit provider view key differs from its context axis"
                )

    def iter_batches(
        self, view: DataView, *, query_id: str
    ) -> Iterable[MeasurementInputBatch]:
        source = self.sources_by_view_hash.get(view.view_hash)
        if source is None:
            raise ValueError(
                "circuit provider has no exact source for the requested DataView"
            )
        return source.circuit_batches(view, query_id=query_id)


__all__ = [
    "CANONICAL_IMAGENET_INPUT_OPERATOR_V1",
    "CONTEXT_AXIS_SCHEMA",
    "PREPROCESSING_SCHEMA",
    "PREPROCESSED_UINT8_BANK_LOADER_CONTRACT",
    "SCRATCH_FOLDER_LOADER_CONTRACT",
    "UINT8_BANK_RECEIPT_SCHEMA",
    "ClassIndexedUint8ContextSource",
    "ContextAxis",
    "ContextRecord",
    "FixedSubsetContextSource",
    "ImageFolderCircuitProvider",
    "ImageFolderContextSource",
    "ImageFolderMeasurementProvider",
    "ImagePreprocessingSpec",
    "PreprocessedUint8BankContextSource",
    "PreprocessedUint8BankReplicaReceipt",
    "ScratchFolderContextSource",
    "build_balanced_imagefolder_context_axis",
    "build_complete_imagefolder_context_axis",
    "build_context_source",
    "build_scratch_imagefolder_context_axis",
    "certify_preprocessed_uint8_bank_replica",
    "imagenet_preprocessing_spec",
    "ordered_context_ids_hash",
    "rebind_context_axis_preprocessing",
]
