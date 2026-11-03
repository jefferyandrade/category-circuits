"""Canonical SSL direction-bank lowering into an executable fixed readout.

Existing SSL CNN workflows identify target computations as rows of a frozen
direction bank over the backbone feature space. Canonical scoring does not
need a separate SSL estimator: installing the frozen affine probe as a fixed
Linear readout makes each target an ordinary declared query while preserving
the exact scientific scalar objective.
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contracts.hashing import sha256_file, stable_digest
from mechanistic_systems.contracts import (
    QueryOperatorSpec,
    QuerySpec,
    ReadoutProjectionSpec,
)

try:  # Optional for import smoke; required for executable lowering.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - environment dependent.
    torch = nn = F = None  # type: ignore[assignment]


SSL_DIRECTION_BANK_SCHEMA = "mechanistic_ssl_direction_bank_v1"
SSL_READOUT_RESTRICTION_SCHEMA = "mechanistic_ssl_readout_restriction_v1"
_BANK_KEYS = (
    "target_vectors",
    "target_directions",
    "directions",
    "prototype_directions",
    "prototypes",
)
_INTERCEPT_KEYS = (
    "target_intercepts",
    "target_biases",
    "intercepts",
    "biases",
)
_STATE_CONTAINER_KEYS = (
    "state_dict",
    "linear_state_dict",
)
_OUTPUT_KEYS = (
    "target_vectors_pt",
    "target_vectors_pth",
    "target_vectors_npy",
    "target_vectors_npz",
    "target_bank_path",
)
_NEGATED_TARGET_FAMILIES = frozenset(
    {"prototype_direction", "adaptive_cluster_direction"}
)
_NORMALIZED_TARGET_FAMILIES = frozenset(
    {"prototype_direction", "adaptive_cluster_direction"}
)


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("SSL direction-bank lowering requires PyTorch")


def _manifest_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("SSL target manifest must be a JSON object")
    return {str(key): value for key, value in payload.items()}


def _target_bank_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    outputs = manifest.get("outputs")
    outputs = outputs if isinstance(outputs, Mapping) else {}
    candidates: list[Path] = []
    for key in _OUTPUT_KEYS:
        raw = outputs.get(key, manifest.get(key))
        if not raw:
            continue
        candidate = Path(str(raw)).expanduser()
        candidates.append(
            candidate if candidate.is_absolute() else manifest_path.parent / candidate
        )
    target_family = str(
        manifest.get("target_mode", manifest.get("target_type", ""))
    ).strip()
    if target_family:
        candidates.extend(
            manifest_path.parent / f"{target_family}_target_bank{suffix}"
            for suffix in (".pt", ".pth", ".npy", ".npz")
        )
    candidates.append(
        manifest_path.parent / "linear_probe_category_direction_target_bank.pt"
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "SSL target bank was not found; tried "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _tensor_bank_payload(
    path: Path,
    *,
    declared_weight_key: str = "",
    declared_bias_key: str = "",
) -> tuple[Any, Any | None]:
    _require_torch()
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if torch.is_tensor(payload):
            value = payload
            intercepts = None
        elif isinstance(payload, Mapping):
            value = next(
                (
                    payload[key]
                    for key in _BANK_KEYS
                    if key in payload and torch.is_tensor(payload[key])
                ),
                None,
            )
            intercepts = next(
                (
                    payload[key]
                    for key in _INTERCEPT_KEYS
                    if key in payload and torch.is_tensor(payload[key])
                ),
                None,
            )
            if value is None:
                state_candidates = [
                    (key, payload[key])
                    for key in _STATE_CONTAINER_KEYS
                    if key in payload and isinstance(payload[key], Mapping)
                ]
                if not state_candidates:
                    state_candidates = [("<root>", payload)]
                resolved: list[tuple[str, str, Any, Any | None]] = []
                for container_key, state in state_candidates:
                    if declared_weight_key:
                        weight = state.get(declared_weight_key)
                        if not torch.is_tensor(weight):
                            continue
                        bias = state.get(declared_bias_key) if declared_bias_key else None
                        if bias is not None and not torch.is_tensor(bias):
                            raise ValueError(
                                "declared full-head bias key is not a tensor: "
                                f"{declared_bias_key!r}"
                            )
                        resolved.append(
                            (container_key, declared_weight_key, weight, bias)
                        )
                        continue

                    affine_weights = [
                        (str(key), tensor)
                        for key, tensor in state.items()
                        if torch.is_tensor(tensor)
                        and tensor.ndim == 2
                        and (str(key) == "weight" or str(key).endswith(".weight"))
                    ]
                    for weight_key, weight in affine_weights:
                        prefix = weight_key[: -len("weight")]
                        bias_key = f"{prefix}bias"
                        bias = state.get(bias_key)
                        if bias is not None and (
                            not torch.is_tensor(bias)
                            or bias.ndim != 1
                            or int(bias.numel()) != int(weight.shape[0])
                        ):
                            raise ValueError(
                                "inferred full-head bias is incompatible with its weight: "
                                f"{container_key}.{bias_key}"
                            )
                        resolved.append((container_key, weight_key, weight, bias))
                if len(resolved) != 1:
                    names = [
                        f"{container}.{key}"
                        for container, key, _weight, _bias in resolved
                    ]
                    raise ValueError(
                        "SSL target-bank checkpoint must expose exactly one affine "
                        f"readout; resolved {len(resolved)} candidates: {names}"
                    )
                _container_key, _weight_key, value, intercepts = resolved[0]
        else:
            raise ValueError("SSL target-bank payload is not a tensor or tensor mapping")
        vectors = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        bias = (
            None
            if intercepts is None
            else intercepts.detach().to(device="cpu", dtype=torch.float32).reshape(-1).contiguous()
        )
        return vectors, bias
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
        intercept_value = None
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            key = next((name for name in _BANK_KEYS if name in payload), None)
            if key is None and len(payload.files) == 1:
                key = payload.files[0]
            if key is None:
                raise ValueError("SSL NPZ target bank has no recognized array")
            value = np.asarray(payload[key])
            intercept_key = next((name for name in _INTERCEPT_KEYS if name in payload), None)
            intercept_value = None if intercept_key is None else np.asarray(payload[intercept_key])
    else:
        raise ValueError(f"unsupported SSL target-bank format: {path}")
    vectors = torch.as_tensor(
        np.asarray(value, dtype=np.float32), dtype=torch.float32
    ).contiguous()
    intercepts = (
        None
        if intercept_value is None
        else torch.as_tensor(
            np.asarray(intercept_value, dtype=np.float32), dtype=torch.float32
        ).reshape(-1).contiguous()
    )
    return vectors, intercepts


def _tensor_content_hash(value: Any | None) -> str:
    if value is None:
        return ""
    array = value.detach().to(device="cpu").contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _source_class_ids(manifest: Mapping[str, Any], row_count: int) -> tuple[int, ...]:
    raw = manifest.get("exported_target_class_ids")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        result = tuple(int(value) for value in raw)
        if len(result) == row_count:
            return result
        raise ValueError("exported target class IDs do not match direction-bank rows")
    targets = manifest.get("targets")
    if isinstance(targets, Sequence) and not isinstance(targets, (str, bytes)):
        result = tuple(
            int(item.get("source_class_id", item.get("target_id")))
            for item in targets
            if isinstance(item, Mapping)
            and item.get("source_class_id", item.get("target_id")) is not None
        )
        if len(result) == row_count:
            return result
    return tuple(range(row_count))


@dataclass(frozen=True)
class SSLDirectionBank:
    cell_id: str
    architecture_family: str
    target_family: str
    target_layer: str
    vectors: Any
    intercepts: Any | None
    source_class_ids: tuple[int, ...]
    bank_path: Path
    bank_hash: str
    manifest_path: Path
    manifest_hash: str
    preprocessing_policy: str
    validation_geometry: str
    direction_normalization: str
    direction_multiplier: float

    def validate(self) -> None:
        _require_torch()
        if not self.cell_id or not self.architecture_family or not self.target_family:
            raise ValueError("SSL direction bank requires cell, architecture, and target identity")
        if not self.target_layer:
            raise ValueError("SSL direction bank requires a target layer")
        if not torch.is_tensor(self.vectors) or self.vectors.ndim != 2:
            raise ValueError("SSL direction bank must be a rank-2 tensor")
        if not bool(torch.isfinite(self.vectors).all().item()):
            raise ValueError("SSL direction bank contains non-finite values")
        if self.intercepts is not None:
            if (
                not torch.is_tensor(self.intercepts)
                or self.intercepts.ndim != 1
                or int(self.intercepts.numel()) != int(self.vectors.shape[0])
            ):
                raise ValueError("SSL probe intercepts must match direction-bank rows")
            if not bool(torch.isfinite(self.intercepts).all().item()):
                raise ValueError("SSL probe intercepts contain non-finite values")
        if len(self.source_class_ids) != int(self.vectors.shape[0]):
            raise ValueError("SSL source-class axis differs from direction-bank rows")
        if len(set(self.source_class_ids)) != len(self.source_class_ids):
            raise ValueError("SSL source-class IDs must be unique")
        if len(self.bank_hash) != 64 or len(self.manifest_hash) != 64:
            raise ValueError("SSL manifest and bank require content hashes")
        if self.direction_normalization not in {"none", "l2"}:
            raise ValueError("SSL direction normalization must be none or l2")
        if not np.isfinite(float(self.direction_multiplier)) or float(
            self.direction_multiplier
        ) == 0.0:
            raise ValueError("SSL direction multiplier must be finite and nonzero")

    @property
    def row_count(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def feature_width(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def content_hash(self) -> str:
        self.validate()
        return stable_digest(
            {
                "schema_name": SSL_DIRECTION_BANK_SCHEMA,
                "cell_id": self.cell_id,
                "architecture_family": self.architecture_family,
                "target_family": self.target_family,
                "target_layer": self.target_layer,
                "source_class_ids": list(self.source_class_ids),
                "bank_hash": self.bank_hash,
                "vector_payload_hash": _tensor_content_hash(self.vectors),
                "intercept_payload_hash": _tensor_content_hash(self.intercepts),
                "manifest_hash": self.manifest_hash,
                "preprocessing_policy": self.preprocessing_policy,
                "validation_geometry": self.validation_geometry,
                "direction_normalization": self.direction_normalization,
                "direction_multiplier": float(self.direction_multiplier),
            }
        )

    def executable_vectors(self) -> Any:
        self.validate()
        values = self.vectors.to(dtype=torch.float32)
        if self.direction_normalization == "l2":
            values = F.normalize(values, p=2, dim=1)
        return values * float(self.direction_multiplier)

    def executable_intercepts(self) -> Any | None:
        self.validate()
        if self.intercepts is None:
            return None
        values = self.intercepts.to(dtype=torch.float32)
        if self.direction_normalization == "l2":
            norms = self.vectors.to(dtype=torch.float32).norm(p=2, dim=1).clamp_min(
                torch.finfo(torch.float32).eps
            )
            values = values / norms
        return values * float(self.direction_multiplier)

    def readout_projection_spec(self) -> ReadoutProjectionSpec:
        """Declare the fixed affine probe as readout construction, not repair."""

        self.validate()
        primitive_axes = tuple(
            f"probe_logit:{source_id}" for source_id in self.source_class_ids
        )
        executable_intercepts = self.executable_intercepts()
        return ReadoutProjectionSpec(
            projection_id=f"{self.cell_id}:fixed_probe",
            projection_kind="fixed_linear",
            source_axis_ids=tuple(
                f"backbone_feature:{index}" for index in range(self.feature_width)
            ),
            primitive_axis_ids=primitive_axes,
            source_hash=self.content_hash,
            weight_hash=_tensor_content_hash(self.executable_vectors()),
            intercept_hash=_tensor_content_hash(executable_intercepts),
            notes={
                "target_family": self.target_family,
                "probe_intercepts_present": executable_intercepts is not None,
            },
        )

    def query_operator_spec(
        self,
        rows: Sequence[int] | None = None,
        *,
        query_track: str = "probe_logit_v1",
    ) -> QueryOperatorSpec:
        self.validate()
        selected = (
            tuple(range(self.row_count))
            if rows is None
            else tuple(int(row) for row in rows)
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("SSL query rows must be nonempty and unique")
        if min(selected) < 0 or max(selected) >= self.row_count:
            raise ValueError("SSL query row is outside the direction bank")
        readout = self.readout_projection_spec()
        query_axes = tuple(
            f"category:{self.source_class_ids[row]}" for row in selected
        )
        if query_track == "probe_logit_v1":
            return QueryOperatorSpec.coordinate(
                operator_id=f"{self.cell_id}:probe_logit_v1",
                primitive_axis_ids=readout.primitive_axis_ids,
                query_axis_ids=query_axes,
                query_primitive_axis_ids=tuple(
                    readout.primitive_axis_ids[row] for row in selected
                ),
                source_hash=readout.content_hash,
            )
        if query_track == "probe_margin_v2":
            return QueryOperatorSpec(
                operator_id=f"{self.cell_id}:probe_margin_v2",
                operator_kind="one_vs_rest_log_odds",
                primitive_axis_ids=readout.primitive_axis_ids,
                query_axis_ids=query_axes,
                support=tuple(readout.primitive_axis_ids for _row in selected),
                source_hash=readout.content_hash,
                parameters={
                    "target_primitive_axis_ids": [
                        readout.primitive_axis_ids[row] for row in selected
                    ]
                },
            )
        raise ValueError(f"unsupported SSL query track {query_track!r}")

    def query_specs(
        self,
        rows: Sequence[int] | None = None,
        *,
        query_track: str = "probe_logit_v1",
    ) -> tuple[QuerySpec, ...]:
        self.validate()
        selected = (
            tuple(range(self.row_count))
            if rows is None
            else tuple(int(row) for row in rows)
        )
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("SSL query rows must be nonempty and unique")
        if min(selected) < 0 or max(selected) >= self.row_count:
            raise ValueError("SSL query row is outside the direction bank")
        if query_track not in {"probe_logit_v1", "probe_margin_v2"}:
            raise ValueError(f"unsupported SSL query track {query_track!r}")
        objective = (
            "class_logit"
            if query_track == "probe_logit_v1"
            else "one_vs_rest_log_odds"
        )
        return tuple(
            QuerySpec(
                query_id=f"{self.cell_id}:{self.target_family}:{self.source_class_ids[row]}",
                query_kind="learned_head",
                ontology_id=f"{self.target_family}:{self.cell_id}",
                axis_ids=(str(self.source_class_ids[row]),),
                evaluator={
                    "objective": objective,
                    "class_id": row,
                    "direction_bank_row": row,
                    "query_track": query_track,
                },
                response_semantics="fixed_ssl_direction_readout",
                source_hash=self.content_hash,
                output_metric={
                    "axis_ids": [str(self.source_class_ids[row])],
                    "metric_kind": "identity",
                },
                directional_functional={
                    "axis_ids": [str(self.source_class_ids[row])],
                    "weights": [1.0],
                },
                notes={
                    "source_class_id": self.source_class_ids[row],
                    "direction_bank_row": row,
                    "target_layer": self.target_layer,
                    "query_track": query_track,
                },
            )
            for row in selected
        )


@dataclass(frozen=True)
class SSLReadoutRestrictionReceipt:
    selected_readout_projection_hash: str
    full_readout_projection_hash: str
    selected_source_class_ids: tuple[int, ...]
    full_row_indices: tuple[int, ...]
    weight_max_abs: float
    intercept_max_abs: float
    tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_name": SSL_READOUT_RESTRICTION_SCHEMA,
            "schema_version": "1",
            "selected_readout_projection_hash": self.selected_readout_projection_hash,
            "full_readout_projection_hash": self.full_readout_projection_hash,
            "selected_source_class_ids": list(self.selected_source_class_ids),
            "full_row_indices": list(self.full_row_indices),
            "weight_max_abs": float(self.weight_max_abs),
            "intercept_max_abs": float(self.intercept_max_abs),
            "tolerance": float(self.tolerance),
            "passed": bool(self.passed),
            "scientific_meaning": (
                "selected fixed readout is a row restriction of the full head; "
                "route derivatives for retained coordinates are unchanged"
            ),
        }
        payload["receipt_hash"] = stable_digest(payload)
        return payload

    @property
    def content_hash(self) -> str:
        return str(self.to_dict()["receipt_hash"])


def certify_ssl_readout_restriction(
    selected: SSLDirectionBank,
    full: SSLDirectionBank,
    *,
    tolerance: float = 1e-7,
) -> SSLReadoutRestrictionReceipt:
    """Prove that a selected probe bank is an exact full-head row restriction."""

    _require_torch()
    selected.validate()
    full.validate()
    if selected.feature_width != full.feature_width:
        raise ValueError("selected and full probe heads have different feature widths")
    full_index = {
        int(source_id): index for index, source_id in enumerate(full.source_class_ids)
    }
    missing = tuple(
        source_id
        for source_id in selected.source_class_ids
        if int(source_id) not in full_index
    )
    if missing:
        raise ValueError(f"selected probe classes are absent from full head: {missing}")
    rows = tuple(full_index[int(source_id)] for source_id in selected.source_class_ids)
    row_index = torch.as_tensor(rows, dtype=torch.long)
    selected_vectors = selected.executable_vectors().to(dtype=torch.float64)
    full_vectors = full.executable_vectors().index_select(0, row_index).to(
        dtype=torch.float64
    )
    weight_max_abs = float((selected_vectors - full_vectors).abs().max().item())
    selected_intercepts = selected.executable_intercepts()
    full_intercepts = full.executable_intercepts()
    if selected_intercepts is None and full_intercepts is None:
        intercept_max_abs = 0.0
    elif selected_intercepts is None or full_intercepts is None:
        intercept_max_abs = float("inf")
    else:
        intercept_max_abs = float(
            (
                selected_intercepts.to(dtype=torch.float64)
                - full_intercepts.index_select(0, row_index).to(dtype=torch.float64)
            )
            .abs()
            .max()
            .item()
        )
    passed = bool(
        np.isfinite(weight_max_abs)
        and np.isfinite(intercept_max_abs)
        and weight_max_abs <= float(tolerance)
        and intercept_max_abs <= float(tolerance)
    )
    receipt = SSLReadoutRestrictionReceipt(
        selected_readout_projection_hash=selected.readout_projection_spec().content_hash,
        full_readout_projection_hash=full.readout_projection_spec().content_hash,
        selected_source_class_ids=selected.source_class_ids,
        full_row_indices=rows,
        weight_max_abs=weight_max_abs,
        intercept_max_abs=intercept_max_abs,
        tolerance=float(tolerance),
        passed=passed,
    )
    if not passed:
        raise ValueError(
            "selected SSL readout is not numerically equivalent to the declared "
            "full-head row restriction"
        )
    return receipt


def load_ssl_direction_bank(
    manifest_path: str | Path,
    *,
    cell_id: str,
    architecture_family: str,
    target_family: str | None = None,
    target_layer: str = "fc",
    bank_path: str | Path | None = None,
    direction_normalization: str | None = None,
    direction_multiplier: float | None = None,
    source_class_ids: Sequence[int] | None = None,
) -> SSLDirectionBank:
    """Load and hash an existing SSL target bank without legacy imports."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    manifest = _manifest_mapping(manifest_file)
    family = str(
        target_family
        or manifest.get("target_mode")
        or manifest.get("target_type")
        or "linear_probe_category_direction"
    ).strip()
    resolved_bank = (
        Path(bank_path).expanduser().resolve()
        if bank_path is not None
        else _target_bank_path(manifest_file, manifest)
    )
    if not resolved_bank.is_file():
        raise FileNotFoundError(resolved_bank)
    checkpoint_contract = manifest.get("checkpoint_sanity", {})
    if not isinstance(checkpoint_contract, Mapping):
        checkpoint_contract = {}
    vectors, intercepts = _tensor_bank_payload(
        resolved_bank,
        declared_weight_key=str(checkpoint_contract.get("weight_key", "")),
        declared_bias_key=str(checkpoint_contract.get("bias_key", "")),
    )
    normalization = str(
        direction_normalization
        if direction_normalization is not None
        else ("l2" if family in _NORMALIZED_TARGET_FAMILIES else "none")
    )
    multiplier = float(
        direction_multiplier
        if direction_multiplier is not None
        else (-1.0 if family in _NEGATED_TARGET_FAMILIES else 1.0)
    )
    result = SSLDirectionBank(
        cell_id=str(cell_id),
        architecture_family=str(architecture_family),
        target_family=family,
        target_layer=str(target_layer),
        vectors=vectors,
        intercepts=intercepts,
        source_class_ids=(
            _source_class_ids(manifest, int(vectors.shape[0]))
            if source_class_ids is None
            else tuple(int(value) for value in source_class_ids)
        ),
        bank_path=resolved_bank,
        bank_hash=sha256_file(resolved_bank),
        manifest_path=manifest_file,
        manifest_hash=sha256_file(manifest_file),
        preprocessing_policy=str(
            manifest.get("linear_eval_preprocessing_policy", "")
        ),
        validation_geometry=str(
            manifest.get("linear_eval_validation_preprocessing_policy", "")
        ),
        direction_normalization=normalization,
        direction_multiplier=multiplier,
    )
    result.validate()
    return result


def install_ssl_direction_readout(model: Any, bank: SSLDirectionBank) -> Any:
    """Install the frozen affine bank exactly once at the readout boundary."""

    _require_torch()
    bank.validate()
    if not isinstance(model, nn.Module):
        raise ValueError("SSL readout installation requires an nn.Module")
    try:
        original = model.get_submodule(bank.target_layer)
    except (AttributeError, KeyError) as error:
        raise ValueError(f"unresolved SSL target layer {bank.target_layer!r}") from error
    if not isinstance(original, nn.Linear):
        raise TypeError("SSL direction readout target must be nn.Linear")
    if int(original.in_features) != bank.feature_width:
        raise ValueError(
            "SSL direction width differs from the declared backbone feature width"
        )
    executable_intercepts = bank.executable_intercepts()
    replacement = nn.Linear(
        bank.feature_width,
        bank.row_count,
        bias=executable_intercepts is not None,
    )
    replacement.to(device=original.weight.device, dtype=original.weight.dtype)
    with torch.no_grad():
        replacement.weight.copy_(bank.executable_vectors().to(replacement.weight))
        if replacement.bias is not None:
            replacement.bias.copy_(executable_intercepts.to(replacement.bias))
    parent_path, _, child_name = bank.target_layer.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)
    replacement.eval()
    return model


__all__ = [
    "SSL_DIRECTION_BANK_SCHEMA",
    "SSL_READOUT_RESTRICTION_SCHEMA",
    "SSLDirectionBank",
    "SSLReadoutRestrictionReceipt",
    "certify_ssl_readout_restriction",
    "install_ssl_direction_readout",
    "load_ssl_direction_bank",
]
