"""Registered, hash-bound load-time model transforms.

A derived model is the pair (registered base checkpoint, transform payload).
The sealed loader re-derives it on every load and verifies it against the
digests the transform spec declares, so no builder or caller-supplied state
dict is ever an authority.  Two kinds are registered:

``zero_routes_v1`` (R03, theta-squared)
    Zero the weight scalars selected by a HardMask ``final_mask.packbits``
    payload (numpy ``packbits``, ``bitorder='big'``) on the declared circuit
    route universe.  Only the declared blocks are touched; within a block the
    bits in ``[offset, stop)`` are reshaped C-order to the block shape (target
    first, then source) and the selected weights become exactly ``0.0``.
    Biases and every other tensor are unchanged.

``positive_unit_rescale_v1`` (GAUGE)
    Rescale every hidden unit j of layer l by a strictly positive ``c_j``:
    ``W_l[j] *= c_j``, ``b_l[j] *= c_j``, and the successor's input slice for
    unit j is divided by ``c_j``.  For ``features.10 -> classifier.1`` that
    slice is the columns ``j*36:(j+1)*36`` of the 256x6x6 flatten.  This is
    function-preserving in exact arithmetic for torchvision AlexNet: ReLU,
    max-pool and adaptive average pooling are positively homogeneous per
    channel, dropout is the identity in eval, and there is no BatchNorm or
    LRN.  A folded uint8 input stem acts on conv1's input side and commutes.

Each modified tensor is computed once in float64 from its float32 source and
rounded to float32 once, so the derived state is a deterministic pure CPU
function of (base state, payload bytes, parameters).  ``apply_model_transform``
never mutates its input mapping.

The identity of a transform is the ``stable_digest`` of every spec field
except ``payload_path``, which is only a runtime locator.  The transforms are
admitted only on ``alexnet_conv_kernel_weight``.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from contracts.hashing import sha256_file, stable_digest

try:  # Optional for import smoke; required to apply a transform.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


MODEL_TRANSFORM_SCHEMA = "v0_model_transform_v1"
MODEL_TRANSFORM_RECEIPT_SCHEMA = "v0_model_transform_application_v1"
ZERO_ROUTES_V1 = "zero_routes_v1"
POSITIVE_UNIT_RESCALE_V1 = "positive_unit_rescale_v1"
MODEL_TRANSFORM_KINDS = frozenset({ZERO_ROUTES_V1, POSITIVE_UNIT_RESCALE_V1})
MODEL_TRANSFORM_ADMITTED_FAMILIES = frozenset({"alexnet_conv_kernel_weight"})
ZERO_ROUTES_PAYLOAD_ENCODING = "numpy_packbits_big_endian_zero_padding_v1"
ZERO_ROUTES_ADMISSIBLE_ROUTE_KINDS = frozenset({"linear_weight", "conv_kernel_weight"})
POSITIVE_UNIT_RESCALE_PAYLOAD_ENCODING = "numpy_npz_float32_c_module_path_v1"
# The loader enforces the semantics that
# the spec identity hashes, not only its digests.  Parameter keys are
# whitelisted per kind; a gauge scale must lie inside a registered hard bound
# (and inside its declared draw range when one is declared).
ZERO_ROUTES_REQUIRED_PARAMETERS = frozenset(
    {
        "payload_encoding",
        "route_universe_hash",
        "route_count",
        "layer_ids",
        "admitted_route_kinds",
    }
)
ZERO_ROUTES_OPTIONAL_PARAMETERS = frozenset({"zeroed_bit_value", "source_circuit"})
ZERO_ROUTES_SOURCE_CIRCUIT_FIELDS = frozenset(
    {
        "p_receipt_hash",
        "query_id",
        "target_class_id",
        "partition_family",
        "mask_hash",
        "mask_packed_sha256",
        "retained_route_count",
    }
)
POSITIVE_UNIT_RESCALE_REQUIRED_PARAMETERS = frozenset({"payload_encoding", "module_paths"})
POSITIVE_UNIT_RESCALE_OPTIONAL_PARAMETERS = frozenset({"draw"})
POSITIVE_UNIT_RESCALE_DRAW_FIELDS = frozenset({"draw_id", "seed", "log_uniform_range"})
# Registered hard bound on every gauge scale, whatever the declared range:
# [2^-8, 2^8] keeps every derived AlexNet weight a normal float32.
POSITIVE_UNIT_RESCALE_HARD_BOUNDS = (2.0**-8, 2.0**8)

# torchvision AlexNet hidden-unit chain: (module, successor, successor input
# group width).  A group width of 36 is the 256x6x6 flatten into classifier.1.
ALEXNET_GAUGE_CHAIN: tuple[tuple[str, str, int], ...] = (
    ("features.0", "features.3", 1),
    ("features.3", "features.6", 1),
    ("features.6", "features.8", 1),
    ("features.8", "features.10", 1),
    ("features.10", "classifier.1", 36),
    ("classifier.1", "classifier.4", 1),
    ("classifier.4", "classifier.6", 1),
)
ALEXNET_GAUGE_MODULE_PATHS: tuple[str, ...] = tuple(
    module for module, _successor, _width in ALEXNET_GAUGE_CHAIN
)
_ALEXNET_STATE_SHAPES: Mapping[str, tuple[int, ...]] = {
    "features.0.weight": (64, 3, 11, 11),
    "features.0.bias": (64,),
    "features.3.weight": (192, 64, 5, 5),
    "features.3.bias": (192,),
    "features.6.weight": (384, 192, 3, 3),
    "features.6.bias": (384,),
    "features.8.weight": (256, 384, 3, 3),
    "features.8.bias": (256,),
    "features.10.weight": (256, 256, 3, 3),
    "features.10.bias": (256,),
    "classifier.1.weight": (4096, 9216),
    "classifier.1.bias": (4096,),
    "classifier.4.weight": (4096, 4096),
    "classifier.4.bias": (4096,),
    "classifier.6.weight": (1000, 4096),
    "classifier.6.bias": (1000,),
}


class ModelTransformError(ValueError):
    """A model transform spec, payload or application failed closed."""


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _canonical_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(parameters, Mapping):
        raise ModelTransformError("model transform parameters must be a mapping")
    try:
        canonical = json.loads(json.dumps(dict(parameters), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ModelTransformError(
            "model transform parameters must be finite JSON values"
        ) from error
    return canonical


def _declared_rescale_range(draw: Mapping[str, Any]) -> tuple[float, float]:
    """The declared log-uniform draw range, validated against the hard bound."""

    value = draw.get("log_uniform_range")
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(bound, (int, float)) and not isinstance(bound, bool) for bound in value)
    ):
        raise ModelTransformError(
            "positive_unit_rescale_v1 draw.log_uniform_range must be [low, high]"
        )
    low, high = float(value[0]), float(value[1])
    hard_low, hard_high = POSITIVE_UNIT_RESCALE_HARD_BOUNDS
    if not (np.isfinite(low) and np.isfinite(high) and hard_low <= low < high <= hard_high):
        raise ModelTransformError(
            "positive_unit_rescale_v1 draw range must satisfy "
            f"{hard_low:g} <= low < high <= {hard_high:g}; got [{low!r}, {high!r}]"
        )
    return low, high


def _validate_parameters(kind: str, parameters: Mapping[str, Any]) -> None:
    if kind not in MODEL_TRANSFORM_KINDS:
        raise ModelTransformError(f"unregistered model transform kind {kind!r}")
    if kind == ZERO_ROUTES_V1:
        required = set(ZERO_ROUTES_REQUIRED_PARAMETERS)
        missing = sorted(required - set(parameters))
        if missing:
            raise ModelTransformError(f"zero_routes_v1 parameters lack {missing}")
        unexpected = sorted(
            set(parameters) - required - set(ZERO_ROUTES_OPTIONAL_PARAMETERS)
        )
        if unexpected:
            raise ModelTransformError(
                f"zero_routes_v1 parameters carry unregistered keys {unexpected}"
            )
        if "zeroed_bit_value" in parameters and (
            type(parameters["zeroed_bit_value"]) is not int
            or parameters["zeroed_bit_value"] != 1
        ):
            raise ModelTransformError(
                "zero_routes_v1 zeroes the payload's 1-bits; zeroed_bit_value "
                "must be 1 when it is declared"
            )
        if "source_circuit" in parameters:
            source = parameters["source_circuit"]
            if not isinstance(source, Mapping):
                raise ModelTransformError("zero_routes_v1 source_circuit must be a mapping")
            extra = sorted(set(source) - set(ZERO_ROUTES_SOURCE_CIRCUIT_FIELDS))
            if extra:
                raise ModelTransformError(
                    f"zero_routes_v1 source_circuit carries unregistered keys {extra}"
                )
            if not _is_sha256(source.get("mask_packed_sha256", "")) or not str(
                source.get("mask_hash", "")
            ):
                raise ModelTransformError(
                    "zero_routes_v1 source_circuit must bind mask_hash and a "
                    "mask_packed_sha256 digest"
                )
        if parameters["payload_encoding"] != ZERO_ROUTES_PAYLOAD_ENCODING:
            raise ModelTransformError("zero_routes_v1 payload encoding differs")
        if not _is_sha256(parameters["route_universe_hash"]):
            raise ModelTransformError("zero_routes_v1 route universe is not a digest")
        if type(parameters["route_count"]) is not int or parameters["route_count"] <= 0:
            raise ModelTransformError("zero_routes_v1 route count must be positive")
        layer_ids = parameters["layer_ids"]
        if (
            not isinstance(layer_ids, list)
            or not layer_ids
            or len(set(layer_ids)) != len(layer_ids)
            or not all(isinstance(value, str) and value for value in layer_ids)
        ):
            raise ModelTransformError(
                "zero_routes_v1 layer_ids must be a nonempty unique list"
            )
        kinds = parameters["admitted_route_kinds"]
        if (
            not isinstance(kinds, list)
            or not kinds
            or len(set(kinds)) != len(kinds)
            or not set(kinds) <= ZERO_ROUTES_ADMISSIBLE_ROUTE_KINDS
        ):
            raise ModelTransformError(
                "zero_routes_v1 admitted_route_kinds must be a nonempty subset "
                f"of {sorted(ZERO_ROUTES_ADMISSIBLE_ROUTE_KINDS)!r}"
            )
    else:
        required = set(POSITIVE_UNIT_RESCALE_REQUIRED_PARAMETERS)
        missing = sorted(required - set(parameters))
        if missing:
            raise ModelTransformError(
                f"positive_unit_rescale_v1 parameters lack {missing}"
            )
        unexpected = sorted(
            set(parameters) - required - set(POSITIVE_UNIT_RESCALE_OPTIONAL_PARAMETERS)
        )
        if unexpected:
            raise ModelTransformError(
                f"positive_unit_rescale_v1 parameters carry unregistered keys {unexpected}"
            )
        if "draw" in parameters:
            draw = parameters["draw"]
            if not isinstance(draw, Mapping):
                raise ModelTransformError("positive_unit_rescale_v1 draw must be a mapping")
            extra = sorted(set(draw) - set(POSITIVE_UNIT_RESCALE_DRAW_FIELDS))
            if extra:
                raise ModelTransformError(
                    f"positive_unit_rescale_v1 draw carries unregistered keys {extra}"
                )
            if "log_uniform_range" in draw:
                _declared_rescale_range(draw)
        if parameters["payload_encoding"] != POSITIVE_UNIT_RESCALE_PAYLOAD_ENCODING:
            raise ModelTransformError(
                "positive_unit_rescale_v1 payload encoding differs"
            )
        if list(parameters["module_paths"]) != list(ALEXNET_GAUGE_MODULE_PATHS):
            raise ModelTransformError(
                "positive_unit_rescale_v1 must rescale exactly the AlexNet "
                f"hidden chain {list(ALEXNET_GAUGE_MODULE_PATHS)!r}"
            )


@dataclass(frozen=True)
class ModelTransformSpec:
    """One registered derived model: (base checkpoint, transform payload)."""

    kind: str
    base_model_id: str
    base_checkpoint_sha256: str
    base_state_dict_sha256: str
    payload_sha256: str
    parameters: Mapping[str, Any]
    derived_state_dict_sha256: str
    payload_path: str = ""
    schema: str = MODEL_TRANSFORM_SCHEMA
    # Directory that resolves a relative ``payload_path``; never identity.
    locator_root: str = field(default="", compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _canonical_parameters(self.parameters))

    def validate(self) -> "ModelTransformSpec":
        if self.schema != MODEL_TRANSFORM_SCHEMA:
            raise ModelTransformError("unsupported model transform schema")
        if self.kind not in MODEL_TRANSFORM_KINDS:
            raise ModelTransformError(
                f"unregistered model transform kind {self.kind!r}; registered "
                f"kinds are {sorted(MODEL_TRANSFORM_KINDS)!r}"
            )
        if not str(self.base_model_id).strip():
            raise ModelTransformError("model transform requires a base model id")
        for name in (
            "base_checkpoint_sha256",
            "base_state_dict_sha256",
            "payload_sha256",
            "derived_state_dict_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ModelTransformError(
                    f"model transform {name} is not a lowercase SHA-256 digest"
                )
        if self.derived_state_dict_sha256 == self.base_state_dict_sha256:
            raise ModelTransformError(
                "a registered model transform must change the model state"
            )
        _validate_parameters(self.kind, self.parameters)
        return self

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind,
            "base_model_id": self.base_model_id,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "base_state_dict_sha256": self.base_state_dict_sha256,
            "payload_sha256": self.payload_sha256,
            "parameters": dict(self.parameters),
            "derived_state_dict_sha256": self.derived_state_dict_sha256,
        }

    @property
    def content_hash(self) -> str:
        self.validate()
        return stable_digest(self.identity_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_dict()
        payload["payload_path"] = self.payload_path
        payload["content_hash"] = self.content_hash
        return payload

    def receipt_block(self) -> dict[str, Any]:
        """The receipt form: identity, content hash and payload digest."""

        return {
            "schema": "v0_model_transform_binding_v1",
            "spec": self.identity_dict(),
            "content_hash": self.content_hash,
            "payload_sha256": self.payload_sha256,
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        locator_root: str | Path | None = None,
    ) -> "ModelTransformSpec":
        expected = {
            "schema",
            "kind",
            "base_model_id",
            "base_checkpoint_sha256",
            "base_state_dict_sha256",
            "payload_sha256",
            "parameters",
            "derived_state_dict_sha256",
            "payload_path",
            "content_hash",
        }
        if set(payload) != expected:
            raise ModelTransformError(
                "model transform spec fields differ: "
                f"missing={sorted(expected - set(payload))}, "
                f"unexpected={sorted(set(payload) - expected)}"
            )
        spec = cls(
            kind=str(payload["kind"]),
            base_model_id=str(payload["base_model_id"]),
            base_checkpoint_sha256=str(payload["base_checkpoint_sha256"]),
            base_state_dict_sha256=str(payload["base_state_dict_sha256"]),
            payload_sha256=str(payload["payload_sha256"]),
            parameters=dict(payload["parameters"]),
            derived_state_dict_sha256=str(payload["derived_state_dict_sha256"]),
            payload_path=str(payload["payload_path"]),
            schema=str(payload["schema"]),
            locator_root="" if locator_root is None else str(locator_root),
        )
        spec.validate()
        if spec.content_hash != str(payload["content_hash"]):
            raise ModelTransformError("model transform spec content hash mismatch")
        return spec

    @classmethod
    def load(cls, path: str | Path) -> "ModelTransformSpec":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ModelTransformError("model transform spec must be a JSON object")
        return cls.from_mapping(payload, locator_root=source.resolve().parent)

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"model transform spec exists: {destination}")
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    def resolved_payload_path(self) -> Path:
        if not self.payload_path:
            raise ModelTransformError("model transform spec has no payload locator")
        candidate = Path(self.payload_path)
        if not candidate.is_absolute() and self.locator_root:
            candidate = Path(self.locator_root) / candidate
        return candidate


def load_model_transform(path: str | Path) -> ModelTransformSpec:
    return ModelTransformSpec.load(path)


def read_model_transform_payload(spec: ModelTransformSpec) -> bytes:
    """Read the payload bytes and verify their declared SHA-256."""

    spec.validate()
    path = spec.resolved_payload_path()
    if not path.is_file():
        raise FileNotFoundError(f"model transform payload is absent: {path}")
    if sha256_file(path) != spec.payload_sha256:
        raise ModelTransformError("model transform payload SHA-256 mismatch")
    return path.read_bytes()


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("model transforms require PyTorch")


def _require_admitted_architecture(bound_architecture: Any) -> None:
    family = str(bound_architecture.spec.family).lower()
    if family not in MODEL_TRANSFORM_ADMITTED_FAMILIES:
        raise ModelTransformError(
            f"model transforms are admitted only for {sorted(MODEL_TRANSFORM_ADMITTED_FAMILIES)!r}; "
            f"got {family!r}"
        )


def _require_alexnet_state(state: Mapping[str, Any]) -> None:
    if set(state) != set(_ALEXNET_STATE_SHAPES):
        raise ModelTransformError("model transform state is not the torchvision AlexNet state")
    for name, shape in _ALEXNET_STATE_SHAPES.items():
        tensor = state[name]
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape:
            raise ModelTransformError(f"AlexNet state entry {name!r} has an unexpected shape")
        if tensor.dtype != torch.float32:
            raise ModelTransformError("model transforms admit only float32 base states")


def _zero_routes(
    state: Mapping[str, Any],
    parameters: Mapping[str, Any],
    bound_architecture: Any,
    payload: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if str(bound_architecture.circuit_route_schema_hash) != parameters["route_universe_hash"]:
        raise ModelTransformError(
            "zero_routes_v1 route universe differs from the bound architecture"
        )
    blocks = tuple(bound_architecture.circuit_route_blocks)
    route_count = sum(int(block.route_count) for block in blocks)
    if route_count != int(parameters["route_count"]):
        raise ModelTransformError("zero_routes_v1 route count differs from the bound axis")
    expected_bytes = (route_count + 7) // 8
    if len(payload) != expected_bytes:
        raise ModelTransformError("zero_routes_v1 payload length differs from the route axis")
    padding = expected_bytes * 8 - route_count
    if padding and payload[-1] & ((1 << padding) - 1):
        raise ModelTransformError("zero_routes_v1 payload has nonzero padding bits")
    blocks_by_layer = {str(block.layer_id): block for block in blocks}
    module_by_layer = {
        str(binding.layer_id): str(binding.module_path)
        for binding in bound_architecture.bindings
    }
    source = parameters.get("source_circuit")
    if source is not None:
        # A spec that names circuit A must
        # zero circuit A -- the declared source circuit binds the payload.
        if str(source["mask_packed_sha256"]) != hashlib.sha256(payload).hexdigest():
            raise ModelTransformError(
                "zero_routes_v1 source_circuit.mask_packed_sha256 differs from the payload"
            )
        from mechanistic_systems.circuits.irg.candidates import HardMask

        payload_mask = HardMask(
            route_axis_hash=str(parameters["route_universe_hash"]),
            route_count=int(parameters["route_count"]),
            packed_bits=bytes(payload),
        )
        if payload_mask.mask_hash != str(source["mask_hash"]):
            raise ModelTransformError(
                "zero_routes_v1 source_circuit.mask_hash differs from the payload's HardMask"
            )
        if "retained_route_count" in source and int(source["retained_route_count"]) != int(
            payload_mask.retained_count
        ):
            raise ModelTransformError(
                "zero_routes_v1 source_circuit.retained_route_count differs from the payload"
            )
    admitted_kinds = set(parameters["admitted_route_kinds"])
    packed = np.frombuffer(payload, dtype=np.uint8)
    derived = dict(state)
    zeroed_by_layer: dict[str, int] = {}
    for layer_id in parameters["layer_ids"]:
        block = blocks_by_layer.get(str(layer_id))
        if block is None:
            raise ModelTransformError(
                f"zero_routes_v1 layer {layer_id!r} is not a circuit route block"
            )
        route_kind = str(block.route_kind)
        if route_kind not in ZERO_ROUTES_ADMISSIBLE_ROUTE_KINDS:
            raise ModelTransformError(
                f"zero_routes_v1 refuses layer {layer_id!r} of route kind {route_kind!r}"
            )
        if route_kind not in admitted_kinds:
            raise ModelTransformError(
                f"zero_routes_v1 layer {layer_id!r} ({route_kind}) is outside the "
                f"admitted route kinds {sorted(admitted_kinds)!r}"
            )
        weight_name = f"{module_by_layer[str(layer_id)]}.weight"
        weight = state.get(weight_name)
        if weight is None or tuple(weight.shape) != tuple(int(v) for v in block.shape):
            raise ModelTransformError(
                f"zero_routes_v1 block {layer_id!r} differs from {weight_name!r}"
            )
        start, stop = int(block.offset), int(block.stop)
        byte_start, byte_stop = start // 8, (stop + 7) // 8
        bits = np.unpackbits(packed[byte_start:byte_stop], bitorder="big")
        head = start - byte_start * 8
        selected = bits[head : head + (stop - start)].astype(bool)
        if selected.size != stop - start:
            raise ModelTransformError("zero_routes_v1 payload does not cover the block")
        selector = torch.from_numpy(
            np.ascontiguousarray(selected.reshape(tuple(int(v) for v in block.shape), order="C"))
        )
        updated = weight.detach().to(device="cpu").clone()
        updated[selector] = 0.0
        derived[weight_name] = updated.contiguous()
        zeroed_by_layer[str(layer_id)] = int(selected.sum(dtype=np.int64))
    details = {
        "zeroed_route_count_by_layer": zeroed_by_layer,
        "zeroed_route_count": int(sum(zeroed_by_layer.values())),
    }
    return derived, details


def _load_rescale_payload(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        names = set(archive.files)
        expected = {f"c_{module}" for module in ALEXNET_GAUGE_MODULE_PATHS}
        if names != expected:
            raise ModelTransformError(
                "positive_unit_rescale_v1 payload keys differ: "
                f"missing={sorted(expected - names)}, unexpected={sorted(names - expected)}"
            )
        vectors = {module: np.asarray(archive[f"c_{module}"]) for module in ALEXNET_GAUGE_MODULE_PATHS}
    for module, vector in vectors.items():
        if vector.dtype != np.float32 or vector.ndim != 1:
            raise ModelTransformError(
                f"positive_unit_rescale_v1 c_{module} must be a float32 vector"
            )
        if not bool(np.all(np.isfinite(vector))) or not bool(np.all(vector > 0.0)):
            raise ModelTransformError(
                f"positive_unit_rescale_v1 c_{module} must be strictly positive and finite"
            )
    return vectors


def _require_rescale_bounds(
    vectors: Mapping[str, np.ndarray], parameters: Mapping[str, Any]
) -> None:
    """Every scale inside the hard bound and the declared draw range.

    The draw is log-uniform in float64 and rounded once to float32, so the
    declared bounds are compared after the same float32 rounding with a
    one-ulp allowance on each side.
    """

    low, high = POSITIVE_UNIT_RESCALE_HARD_BOUNDS
    draw = parameters.get("draw")
    if isinstance(draw, Mapping) and "log_uniform_range" in draw:
        low, high = _declared_rescale_range(draw)
    low32 = np.nextafter(np.float32(low), np.float32(0.0))
    high32 = np.nextafter(np.float32(high), np.float32(np.inf))
    for module, vector in vectors.items():
        if not bool(np.all((vector >= low32) & (vector <= high32))):
            raise ModelTransformError(
                f"positive_unit_rescale_v1 c_{module} leaves the registered scale "
                f"range [{low:g}, {high:g}] (observed [{float(vector.min())!r}, "
                f"{float(vector.max())!r}])"
            )


def _require_representable_rescale(
    module: str, name: str, base: Any, derived: Any
) -> None:
    """The rescaled tensor is finite and loses no nonzero value."""

    derived64 = derived.to(torch.float64)
    if not bool(torch.isfinite(derived64).all()):
        raise ModelTransformError(
            f"positive_unit_rescale_v1 {module}.{name} overflowed to a non-finite value"
        )
    tiny = float(torch.finfo(torch.float32).tiny)
    base_nonzero = base != 0
    underflow = base_nonzero & (derived64.abs() < tiny) & (base.abs() >= tiny)
    if bool((base_nonzero & (derived == 0)).any()) or bool(underflow.any()):
        raise ModelTransformError(
            f"positive_unit_rescale_v1 {module}.{name} underflowed a nonzero value"
        )


def _positive_unit_rescale(
    state: Mapping[str, Any],
    payload: bytes,
    parameters: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vectors = _load_rescale_payload(payload)
    _require_rescale_bounds(vectors, {} if parameters is None else parameters)
    row_scale: dict[str, np.ndarray] = {}
    column_scale: dict[str, tuple[np.ndarray, int]] = {}
    for module, successor, width in ALEXNET_GAUGE_CHAIN:
        vector = vectors[module]
        out_features = int(state[f"{module}.weight"].shape[0])
        if int(vector.shape[0]) != out_features:
            raise ModelTransformError(
                f"positive_unit_rescale_v1 c_{module} has {vector.shape[0]} entries; "
                f"{module} has {out_features} units"
            )
        in_features = int(state[f"{successor}.weight"].shape[1])
        if in_features != out_features * int(width):
            raise ModelTransformError(
                f"{successor} input width {in_features} is not {out_features}x{width}"
            )
        row_scale[module] = vector.astype(np.float64)
        column_scale[successor] = (vector.astype(np.float64), int(width))
    derived = dict(state)
    touched = sorted({*row_scale, *column_scale})
    for module in touched:
        weight = state[f"{module}.weight"].detach().to(device="cpu").to(torch.float64)
        bias = state[f"{module}.bias"].detach().to(device="cpu").to(torch.float64)
        if module in row_scale:
            c = torch.from_numpy(row_scale[module])
            weight = weight * c.reshape((-1,) + (1,) * (weight.ndim - 1))
            bias = bias * c
        if module in column_scale:
            c, width = column_scale[module]
            divisor = torch.from_numpy(np.repeat(c, width))
            weight = weight / divisor.reshape((1, -1) + (1,) * (weight.ndim - 2))
        derived_weight = weight.to(torch.float32).contiguous()
        derived_bias = bias.to(torch.float32).contiguous()
        _require_representable_rescale(
            module, "weight", state[f"{module}.weight"].detach().to(device="cpu"), derived_weight
        )
        _require_representable_rescale(
            module, "bias", state[f"{module}.bias"].detach().to(device="cpu"), derived_bias
        )
        derived[f"{module}.weight"] = derived_weight
        derived[f"{module}.bias"] = derived_bias
    details = {
        "rescaled_unit_count_by_module": {
            module: int(vectors[module].shape[0]) for module in ALEXNET_GAUGE_MODULE_PATHS
        },
        "log_c_range": [
            float(min(np.log(vectors[m].astype(np.float64)).min() for m in vectors)),
            float(max(np.log(vectors[m].astype(np.float64)).max() for m in vectors)),
        ],
        "arithmetic": "float64_once_then_float32_round_once_v1",
    }
    return derived, details


def derive_model_state(
    state: Mapping[str, Any],
    *,
    kind: str,
    parameters: Mapping[str, Any],
    payload: bytes,
    bound_architecture: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The transform itself, before a derived digest exists (builder use).

    ``apply_model_transform`` is this function plus the spec's identity and
    payload-digest checks; the builder uses it to compute the derived digest
    that the spec then declares, and the sealed loader re-derives it.
    """

    _require_torch()
    canonical = _canonical_parameters(parameters)
    _validate_parameters(str(kind), canonical)
    _require_admitted_architecture(bound_architecture)
    if not isinstance(state, Mapping) or not state:
        raise ModelTransformError("model transform requires a nonempty state mapping")
    _require_alexnet_state(state)
    if kind == ZERO_ROUTES_V1:
        return _zero_routes(state, canonical, bound_architecture, bytes(payload))
    return _positive_unit_rescale(state, bytes(payload), canonical)


def apply_model_transform_with_receipt(
    state: Mapping[str, Any],
    spec: ModelTransformSpec,
    bound_architecture: Any,
    *,
    payload: bytes | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply ``spec`` to a CPU state dict; return (derived state, receipt)."""

    _require_torch()
    spec.validate()
    data = read_model_transform_payload(spec) if payload is None else bytes(payload)
    if hashlib.sha256(data).hexdigest() != spec.payload_sha256:
        raise ModelTransformError("model transform payload SHA-256 mismatch")
    derived, details = derive_model_state(
        state,
        kind=spec.kind,
        parameters=spec.parameters,
        payload=data,
        bound_architecture=bound_architecture,
    )
    receipt = {
        "schema": MODEL_TRANSFORM_RECEIPT_SCHEMA,
        "kind": spec.kind,
        "content_hash": spec.content_hash,
        "payload_sha256": spec.payload_sha256,
        **details,
    }
    return derived, receipt


def apply_model_transform(
    state: Mapping[str, Any],
    spec: ModelTransformSpec,
    bound_architecture: Any,
    *,
    payload: bytes | None = None,
) -> dict[str, Any]:
    """Pure CPU float32 transform of ``state``; the input is never mutated."""

    derived, _receipt = apply_model_transform_with_receipt(
        state, spec, bound_architecture, payload=payload
    )
    return derived


__all__ = [
    "ALEXNET_GAUGE_CHAIN",
    "ALEXNET_GAUGE_MODULE_PATHS",
    "MODEL_TRANSFORM_ADMITTED_FAMILIES",
    "MODEL_TRANSFORM_KINDS",
    "MODEL_TRANSFORM_RECEIPT_SCHEMA",
    "MODEL_TRANSFORM_SCHEMA",
    "POSITIVE_UNIT_RESCALE_PAYLOAD_ENCODING",
    "POSITIVE_UNIT_RESCALE_V1",
    "ZERO_ROUTES_ADMISSIBLE_ROUTE_KINDS",
    "ZERO_ROUTES_PAYLOAD_ENCODING",
    "ZERO_ROUTES_V1",
    "ModelTransformError",
    "ModelTransformSpec",
    "apply_model_transform",
    "apply_model_transform_with_receipt",
    "derive_model_state",
    "load_model_transform",
    "read_model_transform_payload",
]
