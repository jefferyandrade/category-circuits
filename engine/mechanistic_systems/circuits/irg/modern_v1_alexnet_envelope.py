"""Pre-outcome analytic envelope for modern-v1 AlexNet proposal scores.

The proposal statistic is the positive-removal orientation of the conserving
raw target-logit attribution ``-G * phi * c_r``.  This module certifies its
absolute value without looking at a train outcome:

* forward interval/absolute propagation bounds every possible registered
  normalized image;
* reverse absolute propagation bounds the cotangent of one frozen target
  logit, including exact max-pool incidence multiplicity;
* ``0 <= phi <= 1`` removes the need to estimate the ReLU path fraction; and
* each route is bounded by its gated weight, source bound, and destination
  cotangent bound.  Kernel-tap routes sum only over their valid spatial
  incidences; and
* an already-frozen action can instead aggregate its signed selected weights
  into one destination-unit drive before taking absolute values.  This keeps
  the canonical ``G * phi * c`` allocation while retaining sound within-unit
  cancellation that singleton-route sums necessarily discard.

The supported graph is deliberately narrow: the exact torchvision AlexNet
module sequence, with kernel-tap convolution routes and linear-edge hidden
routes.  Unsupported modules, altered graph order, non-eval execution,
mixed/nonfinite parameters, axis drift, identity drift, or numerically
unusable bounds refuse.  No context, image, response, attribution row, or
oracle outcome is accepted by the public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_neutral_runtime import (
    DeclaredRouteAxis,
)
from mechanistic_systems.circuits.irg.modern_v1_ladder_statistics import (
    ACTION_SPECIFIC_ENDPOINT_ENVELOPE_KIND,
    V1FrozenCandidateLadder,
    V1FrozenLadderEndpointEnvelopeCertificate,
    build_v1_frozen_ladder_route_sum_endpoint_envelope,
)
from mechanistic_systems.circuits.irg.modern_v1_proposal_statistics import (
    V1CorrectedAttributionEnvelopeCertificate,
)
from mechanistic_systems.circuits.irg.modern_v1_response_exactification import (
    V1CancellationAwareEndpointEnvelopeBundle,
    bind_v1_cancellation_aware_endpoint_envelopes,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.contracts.science import QuerySpec
from mechanistic_systems.core.architectures import (
    canonical_torch_architecture_spec,
)
from mechanistic_systems.measurement.contexts import ImagePreprocessingSpec
from mechanistic_systems.measurement.models import state_dict_digest

try:  # Keep package imports usable where torch is intentionally unavailable.
    import torch
    import torch.fx as fx
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]
    fx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


REGISTERED_NORMALIZED_INPUT_BOX_SCHEMA: Final = (
    "partition_free_v1_registered_normalized_input_box_v1"
)
ALEXNET_ENVELOPE_LAYOUT_SCHEMA: Final = (
    "partition_free_v1_alexnet_envelope_layout_admission_v1"
)
ALEXNET_ENVELOPE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_alexnet_corrected_attribution_envelope_receipt_v1"
)
ALEXNET_ENVELOPE_RESULT_SCHEMA: Final = (
    "partition_free_v1_alexnet_corrected_attribution_envelope_result_v1"
)
ALEXNET_ENVELOPE_ALGORITHM_ID: Final = (
    "alexnet_box_forward_interval_cotangent_reverse_spatial_incidence_v1"
)
ALEXNET_ENVELOPE_PHI_BOUND_ID: Final = (
    "conserving_relu_path_fraction_closed_interval_zero_one_v1"
)
ALEXNET_ENVELOPE_ARITHMETIC_ID: Final = (
    "float64_bound_engine_with_storage_execution_roundoff_and_reverse_absolute_underflow_v2"
)
ALEXNET_ACTION_DRIVE_ENDPOINT_ALGORITHM_ID: Final = (
    "alexnet_frozen_action_shared_destination_drive_interval_v1"
)
ALEXNET_ACTION_DRIVE_ENDPOINT_PROOF_SCHEMA: Final = (
    "partition_free_v1_alexnet_action_drive_endpoint_envelope_proof_v1"
)
ALEXNET_RAW_TARGET_LOGIT_INTERVAL_SCHEMA: Final = (
    "partition_free_v1_alexnet_raw_target_logit_interval_certificate_v1"
)
ALEXNET_RAW_TARGET_LOGIT_INTERVAL_ALGORITHM_ID: Final = (
    "alexnet_mask_specific_box_forward_final_relu_target_readout_interval_v1"
)
ALEXNET_RAW_TARGET_LOGIT_OUTWARD_ROUNDING_ID: Final = (
    "binary64_nextafter_with_storage_execution_roundoff_majorant_v1"
)

_INPUT_BOX_PREFIX: Final = "partition-free-v1-normalized-input-box-"
_LAYOUT_PREFIX: Final = "partition-free-v1-alexnet-envelope-layout-"
_RECEIPT_PREFIX: Final = "partition-free-v1-alexnet-envelope-proof-"
_ENVELOPE_VECTOR_PREFIX: Final = "partition-free-v1-alexnet-envelope-vector-"
_ROUTE_AXIS_PREFIX: Final = "partition-free-v1-alexnet-envelope-route-axis-"
_ACTION_DRIVE_ENDPOINT_PROOF_PREFIX: Final = (
    "partition-free-v1-alexnet-action-drive-endpoint-envelope-proof-"
)
_RAW_TARGET_LOGIT_INTERVAL_PREFIX: Final = (
    "partition-free-v1-alexnet-raw-target-logit-interval-"
)

_FEATURE_TYPES: Final = (
    "Conv2d",
    "ReLU",
    "MaxPool2d",
    "Conv2d",
    "ReLU",
    "MaxPool2d",
    "Conv2d",
    "ReLU",
    "Conv2d",
    "ReLU",
    "Conv2d",
    "ReLU",
    "MaxPool2d",
)
_CLASSIFIER_TYPES: Final = (
    "Dropout",
    "Linear",
    "ReLU",
    "Dropout",
    "Linear",
    "ReLU",
    "Linear",
)
_AFFINE_PATH_BY_LAYER: Final = {
    "features.stage1.conv": "features.0",
    "features.stage2.conv": "features.3",
    "features.stage3.conv": "features.6",
    "features.stage4.conv": "features.8",
    "features.stage5.conv": "features.10",
    "readout.hidden1": "classifier.1",
    "readout.hidden2": "classifier.4",
}
_LAYER_BY_AFFINE_PATH: Final = {
    value: key for key, value in _AFFINE_PATH_BY_LAYER.items()
}
_AFFINE_BOUND_OUTPUT_ROW_CHUNK: Final = 256


def _require_torch() -> None:
    if torch is None or nn is None or F is None or fx is None:
        raise RuntimeError("AlexNet analytic envelopes require PyTorch")


def _identity(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be one positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _direct_hard_mask_hash(mask: HardMask, *, name: str) -> str:
    """Validate and hash live packed bytes without trusting cached identity."""

    if type(mask) is not HardMask:
        raise TypeError(f"{name} must use the exact HardMask type")
    route_count = _positive_int(mask.route_count, f"{name} route count")
    route_axis_hash = _identity(mask.route_axis_hash, f"{name} route axis hash")
    packed = mask.packed_bits
    if type(packed) is not bytes or len(packed) != (route_count + 7) // 8:
        raise ValueError(f"{name} packed byte length differs from its route count")
    padding = len(packed) * 8 - route_count
    if padding and packed[-1] & ((1 << padding) - 1):
        raise ValueError(f"{name} contains nonzero padding bits")
    return content_hash(
        {
            "route_axis_hash": route_axis_hash,
            "route_count": route_count,
            "packed_bits": packed,
        },
        prefix="irg-mask-",
    )


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be one finite scalar")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(
            f"{name} must be finite" + (" and nonnegative" if nonnegative else "")
        )
    return result


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _immutable_f64_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(_finite(value, f"{name} item") for value in values)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _pair_names(values: Sequence[str], name: str) -> tuple[str, str]:
    result = tuple(_identity(value, f"{name} item") for value in values)
    if len(result) != 2 or result[0] == result[1]:
        raise ValueError(f"{name} must contain exactly two distinct names")
    return result  # type: ignore[return-value]


def _pair_positive_ints(values: Sequence[int], name: str) -> tuple[int, int]:
    result = tuple(_positive_int(value, f"{name} item") for value in values)
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return result  # type: ignore[return-value]


def _pair_identities(values: Sequence[str], name: str) -> tuple[str, str]:
    result = tuple(_identity(value, f"{name} item") for value in values)
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two identities")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class V1RegisteredNormalizedInputBox:
    """Outcome-independent box for every tensor emitted by one preprocessor."""

    preprocessing_hash: str
    spatial_shape: tuple[int, int]
    lower_by_channel: tuple[float, ...]
    upper_by_channel: tuple[float, ...]
    registration_source_hash: str
    raw_pixel_support: tuple[float, float] = (0.0, 1.0)
    issued_before_outcomes: bool = True
    schema: str = REGISTERED_NORMALIZED_INPUT_BOX_SCHEMA
    _box_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        preprocessing = _identity(self.preprocessing_hash, "preprocessing_hash")
        source = _identity(self.registration_source_hash, "registration_source_hash")
        spatial = tuple(
            _positive_int(value, "input spatial extent") for value in self.spatial_shape
        )
        if len(spatial) != 2:
            raise ValueError("normalized input box requires height and width")
        lower = _immutable_f64_tuple(self.lower_by_channel, name="input lower bound")
        upper = _immutable_f64_tuple(self.upper_by_channel, name="input upper bound")
        if len(lower) != len(upper) or any(
            lo > hi for lo, hi in zip(lower, upper, strict=True)
        ):
            raise ValueError("normalized input lower/upper channel bounds are invalid")
        raw = tuple(
            _finite(value, "raw pixel support") for value in self.raw_pixel_support
        )
        if len(raw) != 2 or raw[0] > raw[1]:
            raise ValueError("raw pixel support must be an ordered pair")
        if self.issued_before_outcomes is not True:
            raise ValueError("registered input support must be issued before outcomes")
        if self.schema != REGISTERED_NORMALIZED_INPUT_BOX_SCHEMA:
            raise ValueError("unsupported registered normalized input-box schema")
        object.__setattr__(self, "preprocessing_hash", preprocessing)
        object.__setattr__(self, "registration_source_hash", source)
        object.__setattr__(self, "spatial_shape", spatial)
        object.__setattr__(self, "lower_by_channel", lower)
        object.__setattr__(self, "upper_by_channel", upper)
        object.__setattr__(self, "raw_pixel_support", raw)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "preprocessing_hash": self.preprocessing_hash,
            "spatial_shape": self.spatial_shape,
            "lower_by_channel": self.lower_by_channel,
            "upper_by_channel": self.upper_by_channel,
            "registration_source_hash": self.registration_source_hash,
            "raw_pixel_support": self.raw_pixel_support,
            "issued_before_outcomes": self.issued_before_outcomes,
        }

    @property
    def box_hash(self) -> str:
        cached = self._box_hash
        if cached is None:
            cached = content_hash(self._payload(), prefix=_INPUT_BOX_PREFIX)
            object.__setattr__(self, "_box_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_INPUT_BOX_PREFIX)
        if self._box_hash is not None and observed != self._box_hash:
            raise ValueError("registered normalized input box was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("registered normalized input-box hash does not match")
        object.__setattr__(self, "_box_hash", observed)
        return observed


def registered_normalized_input_box_from_preprocessing(
    preprocessing: ImagePreprocessingSpec,
) -> V1RegisteredNormalizedInputBox:
    """Derive the exact normalized support of a uint8-to-unit image operator."""

    _require_torch()
    if type(preprocessing) is not ImagePreprocessingSpec:
        raise TypeError("preprocessing must be an exact ImagePreprocessingSpec")
    preprocessing.validate()
    if (
        preprocessing.scale_uint8_to_unit is not True
        or preprocessing.output_dtype != "float32"
        or len(preprocessing.mean) != 3
        or len(preprocessing.std) != 3
        or not preprocessing.center_crop_size
    ):
        raise ValueError(
            "registered AlexNet input support requires float32 uint8-to-unit "
            "three-channel normalization with a frozen center crop"
        )
    spatial = tuple(int(value) for value in preprocessing.center_crop_size)
    if len(spatial) == 1:
        spatial = (spatial[0], spatial[0])
    if len(spatial) != 2:
        raise ValueError("registered AlexNet preprocessing has no exact output shape")

    # Match Normalize's float32 endpoint arithmetic, then expand by one
    # representable float32 value in each direction.  Every uint8-derived
    # pre-normalization pixel lies in [0, 1], so monotonicity closes the box.
    mean = torch.tensor(preprocessing.mean, dtype=torch.float32)
    std = torch.tensor(preprocessing.std, dtype=torch.float32)
    lower = (torch.zeros_like(mean) - mean) / std
    upper = (torch.ones_like(mean) - mean) / std
    lower = torch.nextafter(lower, torch.full_like(lower, -torch.inf))
    upper = torch.nextafter(upper, torch.full_like(upper, torch.inf))
    registration_source_hash = content_hash(
        {
            "schema": "uint8_unit_interval_float32_normalize_endpoint_derivation_v1",
            "preprocessing": preprocessing.to_dict(),
            "raw_pixel_support": (0.0, 1.0),
            "outward_rounding": "one_float32_nextafter_each_endpoint",
        },
        prefix="partition-free-v1-input-box-derivation-",
    )
    return V1RegisteredNormalizedInputBox(
        preprocessing_hash=preprocessing.content_hash,
        spatial_shape=spatial,
        lower_by_channel=tuple(float(value) for value in lower.tolist()),
        upper_by_channel=tuple(float(value) for value in upper.tolist()),
        registration_source_hash=registration_source_hash,
    )


@dataclass(frozen=True, slots=True)
class V1AlexNetEnvelopeArithmeticProfile:
    """Conservative execution and bound-engine rounding model."""

    device_type: str
    storage_dtype: str
    multiplicand_unit_roundoff: float
    accumulation_unit_roundoff: float
    bound_engine_unit_roundoff: float
    numerical_ceiling: float
    tf32_multiplicand_rounding_admitted: bool
    arithmetic_id: str = ALEXNET_ENVELOPE_ARITHMETIC_ID

    def __post_init__(self) -> None:
        if self.device_type not in {"cpu", "cuda"}:
            raise ValueError("AlexNet envelope arithmetic supports CPU or CUDA only")
        if self.storage_dtype not in {"torch.float32", "torch.float64"}:
            raise ValueError("AlexNet envelope arithmetic requires float32/float64")
        for name in (
            "multiplicand_unit_roundoff",
            "accumulation_unit_roundoff",
            "bound_engine_unit_roundoff",
            "numerical_ceiling",
        ):
            value = _finite(getattr(self, name), name, nonnegative=True)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if type(self.tf32_multiplicand_rounding_admitted) is not bool:
            raise TypeError("tf32_multiplicand_rounding_admitted must be bool")
        if self.arithmetic_id != ALEXNET_ENVELOPE_ARITHMETIC_ID:
            raise ValueError("unsupported AlexNet envelope arithmetic identity")

    @property
    def profile_hash(self) -> str:
        return content_hash(
            {
                "arithmetic_id": self.arithmetic_id,
                "device_type": self.device_type,
                "storage_dtype": self.storage_dtype,
                "multiplicand_unit_roundoff": self.multiplicand_unit_roundoff,
                "accumulation_unit_roundoff": self.accumulation_unit_roundoff,
                "bound_engine_unit_roundoff": self.bound_engine_unit_roundoff,
                "numerical_ceiling": self.numerical_ceiling,
                "tf32_multiplicand_rounding_admitted": (
                    self.tf32_multiplicand_rounding_admitted
                ),
            },
            prefix="partition-free-v1-alexnet-envelope-arithmetic-",
        )


def _arithmetic_profile(model: Any) -> V1AlexNetEnvelopeArithmeticProfile:
    _require_torch()
    state = tuple(model.state_dict().values())
    if not state or any(not isinstance(value, torch.Tensor) for value in state):
        raise ValueError("AlexNet envelope requires a nonempty tensor state")
    floating = tuple(value for value in state if value.is_floating_point())
    if not floating:
        raise ValueError("AlexNet envelope requires floating parameters")
    devices = {value.device for value in floating}
    dtypes = {value.dtype for value in floating}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("AlexNet envelope refuses mixed dtype or device state")
    device = next(iter(devices))
    dtype = next(iter(dtypes))
    if device.type not in {"cpu", "cuda"} or dtype not in {
        torch.float32,
        torch.float64,
    }:
        raise ValueError("AlexNet envelope supports CPU/CUDA float32/float64 only")
    for value in floating:
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError("AlexNet envelope model state is nonfinite")
    if dtype == torch.float64:
        multiplicand_u = 2.0**-53
        accumulation_u = 2.0**-53
        tf32 = False
    elif device.type == "cuda":
        # CUDA float32 convolution/GEMM may use TF32 regardless of the local
        # process default inherited by a future executor.  Certify the weaker
        # multiplicand precision while retaining float32 accumulation.
        multiplicand_u = 2.0**-11
        accumulation_u = 2.0**-24
        tf32 = True
    else:
        multiplicand_u = 2.0**-24
        accumulation_u = 2.0**-24
        tf32 = False
    numerical_ceiling = float(torch.finfo(dtype).max) / 16.0
    return V1AlexNetEnvelopeArithmeticProfile(
        device_type=device.type,
        storage_dtype=str(dtype),
        multiplicand_unit_roundoff=multiplicand_u,
        accumulation_unit_roundoff=accumulation_u,
        bound_engine_unit_roundoff=2.0**-53,
        numerical_ceiling=numerical_ceiling,
        tf32_multiplicand_rounding_admitted=tf32,
    )


def _rounding_factor(
    profile: V1AlexNetEnvelopeArithmeticProfile,
    *,
    term_count: int,
    multiplication_count: int,
) -> float:
    terms = _positive_int(term_count, "rounding term count")
    multiplies = _nonnegative_int(multiplication_count, "multiplication count")
    profile.__post_init__()
    return _exact_rounding_factor_upper(
        terms, multiplies, float(profile.multiplicand_unit_roundoff),
        float(profile.accumulation_unit_roundoff), float(profile.bound_engine_unit_roundoff),
    )


@lru_cache(maxsize=256)
def _exact_rounding_factor_upper(
    terms: int, multiplies: int, multiplicand_u: float,
    accumulation_u: float, bound_engine_u: float,
) -> float:
    """Outward evaluation of the registered error-model formula.

    Computing ``1 + 2**-53`` in binary64 loses the increment; one nextafter
    after several rounded products cannot repair that loss. Evaluate the
    scalar formula exactly, then choose an enclosing binary64 value. Cache
    only scalar factors, never tensors or source/model state. This establishes
    the formula's numerical enclosure, not the underlying error-model theorem.
    """
    multiplication_unit = Fraction.from_float(multiplicand_u)
    accumulation_load = (terms + 1) * Fraction.from_float(accumulation_u)
    bound_unit = Fraction.from_float(bound_engine_u)
    bound_load = (terms + 1) * bound_unit
    if accumulation_load >= Fraction(1, 4) or bound_load >= Fraction(1, 4):
        raise FloatingPointError(
            "rounding-error majorant is outside its admitted regime"
        )
    exact = ((1 + multiplication_unit) ** (2 * multiplies)
             * (1 + bound_unit) ** (multiplies + 1)) / (
        (1 - accumulation_load) * (1 - bound_load)
    )
    try:
        result = float(exact)
    except OverflowError as error:
        raise FloatingPointError("rounding-error factor is invalid") from error
    if math.isfinite(result) and Fraction.from_float(result) < exact:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result) or result < 1.0:
        raise FloatingPointError("rounding-error factor is invalid")
    return result


def _assert_bound_tensor(
    value: Any,
    *,
    profile: V1AlexNetEnvelopeArithmeticProfile,
    name: str,
) -> Any:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float64
        or not bool(torch.isfinite(value).all().item())
        or bool(torch.any(value < 0.0).item())
    ):
        raise FloatingPointError(f"{name} is not a finite nonnegative float64 bound")
    maximum = float(value.max().item()) if value.numel() else 0.0
    if maximum > profile.numerical_ceiling:
        raise FloatingPointError(
            f"{name} exceeds the storage-arithmetic numerical admission ceiling"
        )
    return value


@dataclass(frozen=True, slots=True)
class V1AlexNetEnvelopeLayoutAdmission:
    architecture_id: str
    topology_hash: str
    route_axis_hash: str
    route_axis_content_hash: str
    route_count: int
    model_identity_hash: str
    model_state_hash: str
    query_contract_hash: str
    target_class_id: int
    preprocessing_hash: str
    input_box_hash: str
    graph_semantics_hash: str
    arithmetic_profile_hash: str
    device_type: str
    storage_dtype: str
    schema: str = ALEXNET_ENVELOPE_LAYOUT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "architecture_id",
            "topology_hash",
            "route_axis_hash",
            "route_axis_content_hash",
            "model_identity_hash",
            "model_state_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "input_box_hash",
            "graph_semantics_hash",
            "arithmetic_profile_hash",
            "device_type",
            "storage_dtype",
        ):
            _identity(getattr(self, name), name)
        _positive_int(self.route_count, "route_count")
        _nonnegative_int(self.target_class_id, "target_class_id")
        if self.schema != ALEXNET_ENVELOPE_LAYOUT_SCHEMA:
            raise ValueError("unsupported AlexNet envelope layout schema")

    @property
    def layout_hash(self) -> str:
        return content_hash(self, prefix=_LAYOUT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1AlexNetRawTargetLogitIntervalCertificate:
    """Pre-outcome raw target-logit enclosure for one exact hard mask.

    The bounds are in the model's uncalibrated raw-logit units.  They cover
    every image in the registered normalized input box and include the
    admitted storage/execution arithmetic error.  They do not consume a
    context, label, calibration row, or oracle outcome.
    """

    architecture_id: str
    topology_hash: str
    route_axis_hash: str
    route_axis_content_hash: str
    route_count: int
    current_mask_hash: str
    model_identity_hash: str
    model_state_hash: str
    query_contract_hash: str
    target_class_id: int
    preprocessing_hash: str
    input_box_hash: str
    graph_semantics_hash: str
    arithmetic_profile_hash: str
    layout_hash: str
    raw_target_logit_lower: float
    raw_target_logit_upper: float
    readout_rounding_factor: float
    verifier_source_hash: str
    algorithm_id: str = ALEXNET_RAW_TARGET_LOGIT_INTERVAL_ALGORITHM_ID
    outward_rounding_id: str = ALEXNET_RAW_TARGET_LOGIT_OUTWARD_ROUNDING_ID
    mask_specific: bool = True
    observed_data_consumed: bool = False
    terminal_outcomes_opened: bool = False
    schema: str = ALEXNET_RAW_TARGET_LOGIT_INTERVAL_SCHEMA
    _certificate_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "architecture_id",
            "topology_hash",
            "route_axis_hash",
            "route_axis_content_hash",
            "current_mask_hash",
            "model_identity_hash",
            "model_state_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "input_box_hash",
            "graph_semantics_hash",
            "arithmetic_profile_hash",
            "layout_hash",
            "verifier_source_hash",
        ):
            _identity(getattr(self, name), name)
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, "route_count")
        )
        object.__setattr__(
            self,
            "target_class_id",
            _nonnegative_int(self.target_class_id, "target_class_id"),
        )
        lower = _finite(self.raw_target_logit_lower, "raw_target_logit_lower")
        upper = _finite(self.raw_target_logit_upper, "raw_target_logit_upper")
        if lower > upper:
            raise ValueError("raw target-logit interval is not ordered")
        if _finite(self.readout_rounding_factor, "readout_rounding_factor") < 1.0:
            raise ValueError("readout_rounding_factor must be at least one")
        if (
            self.algorithm_id != ALEXNET_RAW_TARGET_LOGIT_INTERVAL_ALGORITHM_ID
            or self.outward_rounding_id
            != ALEXNET_RAW_TARGET_LOGIT_OUTWARD_ROUNDING_ID
        ):
            raise ValueError("unsupported raw target-logit interval identity")
        if (
            self.mask_specific is not True
            or self.observed_data_consumed is not False
            or self.terminal_outcomes_opened is not False
        ):
            raise PermissionError(
                "raw target-logit interval crossed its pre-outcome authority boundary"
            )
        if self.schema != ALEXNET_RAW_TARGET_LOGIT_INTERVAL_SCHEMA:
            raise ValueError("unsupported raw target-logit interval schema")

    def _payload(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "architecture_id",
                "topology_hash",
                "route_axis_hash",
                "route_axis_content_hash",
                "route_count",
                "current_mask_hash",
                "model_identity_hash",
                "model_state_hash",
                "query_contract_hash",
                "target_class_id",
                "preprocessing_hash",
                "input_box_hash",
                "graph_semantics_hash",
                "arithmetic_profile_hash",
                "layout_hash",
                "raw_target_logit_lower",
                "raw_target_logit_upper",
                "readout_rounding_factor",
                "verifier_source_hash",
                "algorithm_id",
                "outward_rounding_id",
                "mask_specific",
                "observed_data_consumed",
                "terminal_outcomes_opened",
            )
        }

    @property
    def certificate_hash(self) -> str:
        cached = self._certificate_hash
        if cached is None:
            cached = content_hash(
                self._payload(), prefix=_RAW_TARGET_LOGIT_INTERVAL_PREFIX
            )
            object.__setattr__(self, "_certificate_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(), prefix=_RAW_TARGET_LOGIT_INTERVAL_PREFIX
        )
        if self._certificate_hash is not None and observed != self._certificate_hash:
            raise ValueError("raw target-logit interval certificate was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError(
                "raw target-logit interval certificate hash does not match"
            )
        object.__setattr__(self, "_certificate_hash", observed)
        return observed


def _validate_alexnet_graph(model: Any) -> str:
    _require_torch()
    if not isinstance(model, nn.Module):
        raise TypeError("AlexNet envelope requires one torch.nn.Module")
    if bool(model.training):
        raise ValueError("AlexNet envelope requires the frozen model in eval mode")
    if not isinstance(getattr(model, "features", None), nn.Sequential):
        raise ValueError("AlexNet envelope requires the exact features Sequential")
    if not isinstance(getattr(model, "classifier", None), nn.Sequential):
        raise ValueError("AlexNet envelope requires the exact classifier Sequential")
    if type(getattr(model, "avgpool", None)) is not nn.AdaptiveAvgPool2d:
        raise ValueError("AlexNet envelope requires exact AdaptiveAvgPool2d")
    feature_modules = tuple(model.features)
    classifier_modules = tuple(model.classifier)
    if tuple(type(value).__name__ for value in feature_modules) != _FEATURE_TYPES:
        raise ValueError("AlexNet envelope refuses an altered features module sequence")
    if tuple(type(value).__name__ for value in classifier_modules) != _CLASSIFIER_TYPES:
        raise ValueError(
            "AlexNet envelope refuses an altered classifier module sequence"
        )
    expected_types = {
        "Conv2d": nn.Conv2d,
        "ReLU": nn.ReLU,
        "MaxPool2d": nn.MaxPool2d,
        "Dropout": nn.Dropout,
        "Linear": nn.Linear,
    }
    for module, expected_name in zip(feature_modules, _FEATURE_TYPES, strict=True):
        if type(module) is not expected_types[expected_name]:
            raise ValueError("AlexNet envelope refuses module subclasses or wrappers")
    for module, expected_name in zip(
        classifier_modules, _CLASSIFIER_TYPES, strict=True
    ):
        if type(module) is not expected_types[expected_name]:
            raise ValueError("AlexNet envelope refuses module subclasses or wrappers")
    if any(
        "forward" in module.__dict__
        for module in (
            model.features,
            model.avgpool,
            model.classifier,
            *feature_modules,
            *classifier_modules,
        )
    ):
        raise ValueError("AlexNet envelope refuses instance-level forward overrides")
    # torchvision AlexNet declares in-place ReLUs.  Bound propagation never
    # executes those modules and uses only the mathematical ReLU map, so the
    # mutation flag cannot alter the derivation; the exact flag is nevertheless
    # captured by the topology hash.
    for module in feature_modules:
        if type(module) is nn.Conv2d and (
            int(module.groups) != 1 or str(module.padding_mode) != "zeros"
        ):
            raise ValueError(
                "AlexNet envelope currently certifies ungrouped zero-padded "
                "convolutions only"
            )
        if type(module) is nn.MaxPool2d and bool(module.ceil_mode):
            raise ValueError("AlexNet envelope refuses ceil-mode max pooling")
        if type(module) is nn.MaxPool2d and bool(module.return_indices):
            raise ValueError("AlexNet envelope refuses max-pool index outputs")

    graph = fx.symbolic_trace(model).graph
    nodes = tuple(graph.nodes)
    expected_module_targets = tuple(
        [f"features.{index}" for index in range(len(feature_modules))]
        + ["avgpool"]
        + [f"classifier.{index}" for index in range(len(classifier_modules))]
    )
    module_targets = tuple(
        str(node.target) for node in nodes if node.op == "call_module"
    )
    call_functions = tuple(node for node in nodes if node.op == "call_function")
    if (
        module_targets != expected_module_targets
        or len(call_functions) != 1
        or call_functions[0].target is not torch.flatten
        or tuple(call_functions[0].args[1:]) != (1,)
        or call_functions[0].kwargs
        or sum(node.op == "placeholder" for node in nodes) != 1
        or sum(node.op == "output" for node in nodes) != 1
        or any(
            node.op not in {"placeholder", "call_module", "call_function", "output"}
            for node in nodes
        )
    ):
        raise ValueError("AlexNet envelope refuses an unsupported forward graph")
    module_semantics: list[tuple[str, str, tuple[tuple[str, Any], ...]]] = []
    for path in expected_module_targets:
        module = model.get_submodule(path)
        fields: dict[str, Any] = {}
        if type(module) is nn.Conv2d:
            fields = {
                "in_channels": int(module.in_channels),
                "out_channels": int(module.out_channels),
                "kernel_size": _pair(module.kernel_size),
                "stride": _pair(module.stride),
                "padding": _pair(module.padding),
                "dilation": _pair(module.dilation),
                "groups": int(module.groups),
                "padding_mode": str(module.padding_mode),
                "bias": module.bias is not None,
            }
        elif type(module) is nn.MaxPool2d:
            fields = {
                "kernel_size": _pair(module.kernel_size),
                "stride": _pair(
                    module.stride if module.stride is not None else module.kernel_size
                ),
                "padding": _pair(module.padding),
                "dilation": _pair(module.dilation),
                "ceil_mode": bool(module.ceil_mode),
                "return_indices": bool(module.return_indices),
            }
        elif type(module) is nn.AdaptiveAvgPool2d:
            fields = {"output_size": _pair(module.output_size)}
        elif type(module) is nn.Linear:
            fields = {
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "bias": module.bias is not None,
            }
        elif type(module) is nn.ReLU:
            fields = {"inplace": bool(module.inplace)}
        elif type(module) is nn.Dropout:
            fields = {"p": float(module.p), "inplace": bool(module.inplace)}
        module_semantics.append(
            (
                path,
                f"{type(module).__module__}.{type(module).__qualname__}",
                tuple(sorted(fields.items())),
            )
        )
    graph_payload = {
        "node_program": tuple(
            (
                node.op,
                "torch.flatten" if node.op == "call_function" else str(node.target),
                (1,) if node.op == "call_function" else (),
            )
            for node in nodes
        ),
        "module_semantics": tuple(module_semantics),
    }
    return content_hash(
        graph_payload,
        prefix="partition-free-v1-alexnet-forward-graph-",
    )


def _target_class(query: QuerySpec, output_count: int) -> int:
    if type(query) is not QuerySpec:
        raise TypeError("AlexNet envelope requires an exact QuerySpec")
    query.validate().require_valid()
    objective = str(query.evaluator.get("objective", query.evaluator.get("kind", "")))
    target = query.evaluator.get("class_id")
    if (
        query.query_kind != "class_logit"
        or objective != "class_logit"
        or isinstance(target, bool)
        or not isinstance(target, (int, np.integer))
        or int(target) < 0
        or int(target) >= int(output_count)
    ):
        raise ValueError(
            "AlexNet envelope requires one in-range raw class-logit target"
        )
    return int(target)


def admit_v1_alexnet_envelope_layout(
    *,
    model: Any,
    route_axis: DeclaredRouteAxis,
    current_mask: HardMask,
    query: QuerySpec,
    model_identity_hash: str,
    expected_model_state_hash: str,
    preprocessing_hash: str,
    input_box: V1RegisteredNormalizedInputBox,
) -> V1AlexNetEnvelopeLayoutAdmission:
    """Close graph, state, route, mask, query, and preprocessing identities."""

    _require_torch()
    if type(route_axis) is not DeclaredRouteAxis:
        raise TypeError("AlexNet envelope requires an exact DeclaredRouteAxis")
    if type(current_mask) is not HardMask:
        raise TypeError("AlexNet envelope requires an exact HardMask")
    if type(input_box) is not V1RegisteredNormalizedInputBox:
        raise TypeError("AlexNet envelope requires an exact registered input box")
    _direct_hard_mask_hash(current_mask, name="AlexNet envelope current mask")
    model_identity = _identity(model_identity_hash, "model_identity_hash")
    expected_state = _identity(expected_model_state_hash, "expected_model_state_hash")
    preprocessing = _identity(preprocessing_hash, "preprocessing_hash")
    input_box.assert_integrity()
    if input_box.preprocessing_hash != preprocessing:
        raise ValueError("registered input box differs from the preprocessing identity")
    graph_hash = _validate_alexnet_graph(model)
    profile = _arithmetic_profile(model)
    observed_state = state_dict_digest(model)
    if observed_state != expected_state:
        raise ValueError(
            "live AlexNet state differs from the expected model state hash"
        )

    bound = canonical_torch_architecture_spec("alexnet_conv_kernel_weight").bind(model)
    bound.validate()
    expected_blocks = bound.circuit_route_blocks
    if (
        route_axis.route_axis_hash != bound.circuit_route_schema_hash
        or tuple(block.to_dict() for block in route_axis.route_blocks)
        != tuple(block.to_dict() for block in expected_blocks)
        or route_axis.route_count != expected_blocks[-1].stop
    ):
        raise ValueError("AlexNet envelope route blocks differ from the live model")
    if (
        current_mask.route_axis_hash != route_axis.route_axis_hash
        or current_mask.route_count != route_axis.route_count
    ):
        raise ValueError("AlexNet envelope hard mask differs from the route axis")
    expected_layers = tuple(_AFFINE_PATH_BY_LAYER)
    if tuple(block.layer_id for block in route_axis.route_blocks) != expected_layers:
        raise ValueError("AlexNet envelope route blocks use an unsupported layer order")
    if (
        tuple(block.route_kind for block in route_axis.route_blocks[:5])
        != ("conv_kernel_weight",) * 5
        or tuple(block.route_kind for block in route_axis.route_blocks[5:])
        != ("linear_weight",) * 2
    ):
        raise ValueError(
            "AlexNet envelope requires tap-granularity conv and linear routes"
        )
    first = model.features[0]
    if len(input_box.lower_by_channel) != int(first.in_channels):
        raise ValueError("registered input box channel count differs from AlexNet")
    target = _target_class(query, int(model.classifier[6].out_features))
    route_axis_content_hash = content_hash(
        tuple(block.to_dict() for block in route_axis.route_blocks),
        prefix=_ROUTE_AXIS_PREFIX,
    )
    return V1AlexNetEnvelopeLayoutAdmission(
        architecture_id=bound.architecture_id,
        topology_hash=bound.topology_hash,
        route_axis_hash=route_axis.route_axis_hash,
        route_axis_content_hash=route_axis_content_hash,
        route_count=route_axis.route_count,
        model_identity_hash=model_identity,
        model_state_hash=observed_state,
        query_contract_hash=query.content_hash,
        target_class_id=target,
        preprocessing_hash=preprocessing,
        input_box_hash=input_box.box_hash,
        graph_semantics_hash=graph_hash,
        arithmetic_profile_hash=profile.profile_hash,
        device_type=profile.device_type,
        storage_dtype=profile.storage_dtype,
    )


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("operator parameter must be scalar or pair")
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def _effective_gate_tensor(gate: np.ndarray, *, module: Any) -> Any:
    return torch.as_tensor(
        gate,
        dtype=torch.float64,
        device=module.weight.device,
    )


def _assert_interval_tensors(
    lower: Any,
    upper: Any,
    *,
    profile: V1AlexNetEnvelopeArithmeticProfile,
    name: str,
) -> tuple[Any, Any]:
    if (
        not isinstance(lower, torch.Tensor)
        or not isinstance(upper, torch.Tensor)
        or lower.dtype != torch.float64
        or upper.dtype != torch.float64
        or lower.shape != upper.shape
        or not bool(torch.isfinite(lower).all().item())
        or not bool(torch.isfinite(upper).all().item())
        or bool(torch.any(lower > upper).item())
    ):
        raise FloatingPointError(f"{name} is not one finite ordered float64 interval")
    maximum = (
        float(torch.maximum(lower.abs(), upper.abs()).max().item())
        if lower.numel()
        else 0.0
    )
    if maximum > profile.numerical_ceiling:
        raise FloatingPointError(
            f"{name} exceeds the storage-arithmetic numerical admission ceiling"
        )
    return lower, upper


def _interval_absolute_bound(lower: Any, upper: Any) -> Any:
    return torch.maximum(lower.abs(), upper.abs()).contiguous()


def _execution_underflow_radius(
    profile: V1AlexNetEnvelopeArithmeticProfile,
    *,
    operation_count: int,
) -> float:
    """Conservative absolute radius for gradual-underflow or FTZ execution.

    Relative ``gamma_n`` bounds do not cover subnormal arithmetic.  The
    storage dtype's smallest *normal* value is an intentionally conservative
    per-operation radius: it covers gradual underflow and also a possible
    flush-to-zero execution mode.  This is negligible on ordinary AlexNet
    scales, but prevents a nonzero action drive from acquiring a false exact
    zero certificate in adversarial arithmetic regimes.
    """

    operations = _positive_int(operation_count, "underflow operation count")
    if profile.storage_dtype == "torch.float32":
        minimum_normal = float(np.finfo(np.float32).tiny)
    elif profile.storage_dtype == "torch.float64":
        minimum_normal = float(np.finfo(np.float64).tiny)
    else:  # pragma: no cover - the arithmetic profile already refuses this.
        raise ValueError("unsupported storage dtype for underflow majorant")
    radius = operations * minimum_normal
    if not math.isfinite(radius) or radius <= 0.0:
        raise FloatingPointError("underflow-error radius is invalid")
    return math.nextafter(radius, math.inf)


def _outward_interval(
    nominal_lower: Any,
    nominal_upper: Any,
    absolute_sum: Any,
    factor: float,
    *,
    profile: V1AlexNetEnvelopeArithmeticProfile,
    name: str,
    nonzero_support: Any | None = None,
    absolute_underflow_radius: float = 0.0,
) -> tuple[Any, Any]:
    """Add a roundoff radius to real-arithmetic interval endpoints.

    ``absolute_sum`` is the positive contraction majorizing every signed
    summand.  The same combined storage/execution/bound-engine factor used by
    the absolute propagator therefore bounds both the native operator error
    and the float64 endpoint-construction error.  Multiplying the already
    outward absolute sum by ``factor - 1`` is deliberately conservative.
    """

    radius = _finite(
        absolute_underflow_radius,
        f"{name} absolute underflow radius",
        nonnegative=True,
    )
    if nonzero_support is None:
        support = absolute_sum != 0.0
    else:
        if (
            not isinstance(nonzero_support, torch.Tensor)
            or nonzero_support.dtype != torch.bool
            or nonzero_support.shape != absolute_sum.shape
        ):
            raise TypeError(f"{name} nonzero support must be one aligned bool tensor")
        support = nonzero_support
    exact_zero = torch.logical_not(support)
    radius_tensor = torch.where(
        support,
        torch.full_like(absolute_sum, radius),
        torch.zeros_like(absolute_sum),
    )
    absolute_upper = torch.where(
        exact_zero,
        torch.zeros_like(absolute_sum),
        torch.nextafter(
            absolute_sum * factor + radius_tensor,
            torch.full_like(absolute_sum, torch.inf),
        ),
    )
    _assert_bound_tensor(
        absolute_upper, profile=profile, name=f"{name} absolute summand bound"
    )
    error = torch.where(
        exact_zero,
        torch.zeros_like(absolute_upper),
        torch.nextafter(
            absolute_upper * (factor - 1.0) + radius_tensor,
            torch.full_like(absolute_upper, torch.inf),
        ),
    )
    lower = torch.nextafter(
        nominal_lower - error, torch.full_like(nominal_lower, -torch.inf)
    )
    upper = torch.nextafter(
        nominal_upper + error, torch.full_like(nominal_upper, torch.inf)
    )
    # A zero absolute summand proves that the affine output is the exact
    # floating-point zero; outward nextafter must not manufacture a path.
    lower = torch.where(exact_zero, torch.zeros_like(lower), lower)
    upper = torch.where(exact_zero, torch.zeros_like(upper), upper)
    return _assert_interval_tensors(
        lower, upper, profile=profile, name=f"{name} forward interval"
    )


def _conv_forward_interval_bound(
    source_lower: Any,
    source_upper: Any,
    module: Any,
    gate: np.ndarray,
    profile: V1AlexNetEnvelopeArithmeticProfile,
) -> tuple[Any, Any, float]:
    weight = module.weight.detach().to(torch.float64)
    weight = weight * _effective_gate_tensor(gate, module=module)
    positive = weight.clamp_min(0.0)
    negative = weight.clamp_max(0.0)
    common = (
        module.stride,
        module.padding,
        module.dilation,
        module.groups,
    )
    nominal_lower = (
        F.conv2d(source_lower.unsqueeze(0), positive, None, *common)[0]
        + F.conv2d(source_upper.unsqueeze(0), negative, None, *common)[0]
    )
    nominal_upper = (
        F.conv2d(source_upper.unsqueeze(0), positive, None, *common)[0]
        + F.conv2d(source_lower.unsqueeze(0), negative, None, *common)[0]
    )
    source_absolute = _interval_absolute_bound(source_lower, source_upper)
    absolute_sum = F.conv2d(
        source_absolute.unsqueeze(0),
        weight.abs(),
        None,
        *common,
    )[0]
    support = (
        F.conv2d(
            (source_absolute != 0.0).to(torch.float64).unsqueeze(0),
            (weight != 0.0).to(torch.float64),
            None,
            *common,
        )[0]
        > 0.0
    )
    if module.bias is not None:
        bias = module.bias.detach().to(torch.float64).reshape(-1, 1, 1)
        nominal_lower = nominal_lower + bias
        nominal_upper = nominal_upper + bias
        absolute_sum = absolute_sum + bias.abs()
        support = torch.logical_or(support, bias != 0.0)
    terms = (
        int(module.in_channels // module.groups)
        * int(module.kernel_size[0])
        * int(module.kernel_size[1])
    )
    factor = _rounding_factor(
        profile,
        # Positive and negative terms are evaluated in separate contractions
        # and then joined.  Twice the logical term count covers both engines.
        term_count=2 * (terms + 1),
        multiplication_count=1,
    )
    lower, upper = _outward_interval(
        nominal_lower,
        nominal_upper,
        absolute_sum,
        factor,
        profile=profile,
        name="conv",
        nonzero_support=support,
        absolute_underflow_radius=_execution_underflow_radius(
            profile, operation_count=2 * (terms + 1)
        ),
    )
    return lower, upper, factor


def _linear_forward_interval_bound(
    source_lower: Any,
    source_upper: Any,
    module: Any,
    gate: np.ndarray | None,
    profile: V1AlexNetEnvelopeArithmeticProfile,
    *,
    output_row_chunk: int,
    output_row_start: int = 0,
    output_row_stop: int | None = None,
) -> tuple[Any, Any, float]:
    rows = _positive_int(output_row_chunk, "linear output row chunk")
    row_start = _nonnegative_int(output_row_start, "linear output row start")
    row_stop = (
        int(module.out_features)
        if output_row_stop is None
        else _positive_int(output_row_stop, "linear output row stop")
    )
    if row_start >= row_stop or row_stop > int(module.out_features):
        raise ValueError("linear output row interval is outside the module axis")
    output_count = row_stop - row_start
    lower = torch.empty(
        output_count, dtype=torch.float64, device=module.weight.device
    )
    upper = torch.empty(
        output_count, dtype=torch.float64, device=module.weight.device
    )
    factor = _rounding_factor(
        profile,
        term_count=2 * (int(module.in_features) + 1),
        multiplication_count=1,
    )
    source_absolute = _interval_absolute_bound(source_lower, source_upper)
    for start in range(row_start, row_stop, rows):
        stop = min(row_stop, start + rows)
        destination_start = start - row_start
        destination_stop = stop - row_start
        weight = module.weight.detach()[start:stop].to(torch.float64)
        if gate is not None:
            weight = weight * _effective_gate_tensor(gate[start:stop], module=module)
        positive = weight.clamp_min(0.0)
        negative = weight.clamp_max(0.0)
        nominal_lower = torch.mv(positive, source_lower) + torch.mv(
            negative, source_upper
        )
        nominal_upper = torch.mv(positive, source_upper) + torch.mv(
            negative, source_lower
        )
        absolute_sum = torch.mv(weight.abs(), source_absolute)
        support = (
            torch.mv(
                (weight != 0.0).to(torch.float64),
                (source_absolute != 0.0).to(torch.float64),
            )
            > 0.0
        )
        if module.bias is not None:
            bias = module.bias.detach()[start:stop].to(torch.float64)
            nominal_lower = nominal_lower + bias
            nominal_upper = nominal_upper + bias
            absolute_sum = absolute_sum + bias.abs()
            support = torch.logical_or(support, bias != 0.0)
        chunk_lower, chunk_upper = _outward_interval(
            nominal_lower,
            nominal_upper,
            absolute_sum,
            factor,
            profile=profile,
            name="linear",
            nonzero_support=support,
            absolute_underflow_radius=_execution_underflow_radius(
                profile, operation_count=2 * (int(module.in_features) + 1)
            ),
        )
        lower[destination_start:destination_stop] = chunk_lower
        upper[destination_start:destination_stop] = chunk_upper
    lower, upper = _assert_interval_tensors(
        lower, upper, profile=profile, name="linear forward interval"
    )
    return lower, upper, factor


def _outward_reverse_absolute_bound(
    base: Any, *, factor: float, support: Any,
    profile: V1AlexNetEnvelopeArithmeticProfile, operation_count: int, name: str,
) -> Any:
    """Reverse contraction with local absolute underflow/FTZ error.

    Support is formed from operand incidence, never from a rounded product.
    Local floors propagate through the next operator and are therefore
    amplified by its actual weights. The factor also covers accumulation of
    local absolute errors. Exact absence of every path remains zero.
    """
    floor = _execution_underflow_radius(profile, operation_count=operation_count)
    positive_inf = torch.full_like(base, torch.inf)
    rounded_base = torch.nextafter(base * factor, positive_inf)
    rounded_floor = torch.nextafter(torch.full_like(base, floor) * factor, positive_inf)
    upper = torch.nextafter(rounded_base + rounded_floor, positive_inf)
    result = torch.where(support, upper, torch.zeros_like(base))
    return _assert_bound_tensor(result, profile=profile, name=name)


def _conv_transpose_absolute_bound(
    destination: Any,
    module: Any,
    gate: np.ndarray,
    input_shape: tuple[int, int, int],
    profile: V1AlexNetEnvelopeArithmeticProfile,
) -> tuple[Any, float]:
    weight = module.weight.detach().to(torch.float64).abs()
    weight = weight * _effective_gate_tensor(gate, module=module)
    stride = _pair(module.stride)
    padding = _pair(module.padding)
    dilation = _pair(module.dilation)
    kernel = _pair(module.kernel_size)
    output_padding = tuple(
        int(input_shape[index + 1])
        - (
            (int(destination.shape[index + 1]) - 1) * stride[index]
            - 2 * padding[index]
            + dilation[index] * (kernel[index] - 1)
            + 1
        )
        for index in range(2)
    )
    if any(
        value < 0 or value >= stride[index]
        for index, value in enumerate(output_padding)
    ):
        raise ValueError(
            "convolution transpose cannot recover the registered source shape"
        )
    base = F.conv_transpose2d(
        destination.unsqueeze(0),
        weight,
        None,
        module.stride,
        module.padding,
        output_padding,
        module.groups,
        module.dilation,
    )[0]
    if tuple(int(value) for value in base.shape) != input_shape:
        raise RuntimeError("convolution cotangent bound changed the source shape")
    terms = int(module.out_channels // module.groups) * kernel[0] * kernel[1]
    factor = _rounding_factor(profile, term_count=terms, multiplication_count=1)
    support = F.conv_transpose2d(
        (destination != 0).to(torch.float64).unsqueeze(0),
        (weight != 0).to(torch.float64), None, module.stride, module.padding,
        output_padding, module.groups, module.dilation,
    )[0] > 0
    result = _outward_reverse_absolute_bound(
        base, factor=factor, support=support, profile=profile,
        operation_count=2 * terms + 1, name="conv cotangent bound")
    return (
        _assert_bound_tensor(result, profile=profile, name="conv cotangent bound"),
        factor,
    )


def _linear_transpose_absolute_bound(
    destination: Any,
    module: Any,
    gate: np.ndarray,
    profile: V1AlexNetEnvelopeArithmeticProfile,
    *,
    output_row_chunk: int,
) -> tuple[Any, float]:
    result = torch.zeros(
        int(module.in_features), dtype=torch.float64, device=module.weight.device
    )
    rows = _positive_int(output_row_chunk, "linear transpose row chunk")
    support = torch.zeros_like(result, dtype=torch.bool)
    for start in range(0, int(module.out_features), rows):
        stop = min(int(module.out_features), start + rows)
        weight = module.weight.detach()[start:stop].to(torch.float64).abs()
        weight = weight * _effective_gate_tensor(gate[start:stop], module=module)
        result.add_(torch.mv(weight.T, destination[start:stop]))
        support |= torch.mv((weight != 0).to(torch.float64).T,
                            (destination[start:stop] != 0).to(torch.float64)) > 0
    factor = _rounding_factor(
        profile,
        # The bound engine performs a dot-product reduction inside each row
        # packet and then adds packet totals.  Twice the logical term count is
        # a conservative operation-count majorant for both stages.
        term_count=2 * int(module.out_features),
        multiplication_count=1,
    )
    result = _outward_reverse_absolute_bound(
        result, factor=factor, support=support, profile=profile,
        operation_count=2 * int(module.out_features) + math.ceil(int(module.out_features) / rows) + 1,
        name="linear cotangent bound")
    return (
        _assert_bound_tensor(result, profile=profile, name="linear cotangent bound"),
        factor,
    )


def _maxpool_output_shape(module: Any, input_shape: tuple[int, int]) -> tuple[int, int]:
    kernel = _pair(module.kernel_size)
    stride = _pair(module.stride if module.stride is not None else module.kernel_size)
    padding = _pair(module.padding)
    dilation = _pair(module.dilation)
    return tuple(
        math.floor(
            (
                input_shape[index]
                + 2 * padding[index]
                - dilation[index] * (kernel[index] - 1)
                - 1
            )
            / stride[index]
            + 1
        )
        for index in range(2)
    )  # type: ignore[return-value]


def _reverse_maxpool_absolute_bound(
    destination: Any,
    module: Any,
    input_shape: tuple[int, int, int],
    profile: V1AlexNetEnvelopeArithmeticProfile,
) -> tuple[Any, float]:
    """Exact incidence upper bound for an unknown max-pool argmax pattern."""

    channels, input_h, input_w = input_shape
    if int(destination.shape[0]) != channels or tuple(
        destination.shape[1:]
    ) != _maxpool_output_shape(module, (input_h, input_w)):
        raise ValueError("max-pool cotangent bound has incompatible geometry")
    kernel = _pair(module.kernel_size)
    stride = _pair(module.stride if module.stride is not None else module.kernel_size)
    padding = _pair(module.padding)
    dilation = _pair(module.dilation)
    result = torch.zeros(input_shape, dtype=torch.float64, device=destination.device)
    support = torch.zeros_like(result, dtype=torch.bool)
    incidence = torch.zeros(
        (input_h, input_w), dtype=torch.int32, device=destination.device
    )
    for output_h in range(int(destination.shape[1])):
        for output_w in range(int(destination.shape[2])):
            value = destination[:, output_h, output_w]
            for kh in range(kernel[0]):
                input_row = output_h * stride[0] - padding[0] + kh * dilation[0]
                if input_row < 0 or input_row >= input_h:
                    continue
                for kw in range(kernel[1]):
                    input_col = output_w * stride[1] - padding[1] + kw * dilation[1]
                    if input_col < 0 or input_col >= input_w:
                        continue
                    result[:, input_row, input_col].add_(value)
                    support[:, input_row, input_col] |= value != 0
                    incidence[input_row, input_col] += 1
    maximum_incidence = max(1, int(incidence.max().item()))
    factor = _rounding_factor(
        profile, term_count=maximum_incidence, multiplication_count=0
    )
    result = _outward_reverse_absolute_bound(
        result, factor=factor, support=support, profile=profile,
        operation_count=maximum_incidence + 1, name="max-pool cotangent bound")
    return (
        _assert_bound_tensor(result, profile=profile, name="max-pool cotangent bound"),
        factor,
    )


def _adaptive_region(
    index: int, input_extent: int, output_extent: int
) -> tuple[int, int]:
    start = math.floor(index * input_extent / output_extent)
    stop = math.ceil((index + 1) * input_extent / output_extent)
    return start, stop


def _adaptive_avg_forward_interval_bound(
    source_lower: Any,
    source_upper: Any,
    output_size: tuple[int, int],
    profile: V1AlexNetEnvelopeArithmeticProfile,
) -> tuple[Any, Any, float]:
    nominal_lower = F.adaptive_avg_pool2d(source_lower.unsqueeze(0), output_size)[0]
    nominal_upper = F.adaptive_avg_pool2d(source_upper.unsqueeze(0), output_size)[0]
    source_absolute = _interval_absolute_bound(source_lower, source_upper)
    absolute_sum = F.adaptive_avg_pool2d(source_absolute.unsqueeze(0), output_size)[0]
    support = (
        F.adaptive_avg_pool2d(
            (source_absolute != 0.0).to(torch.float64).unsqueeze(0), output_size
        )[0]
        > 0.0
    )
    input_h, input_w = (int(value) for value in source_lower.shape[-2:])
    maximum_terms = 1
    for oh in range(output_size[0]):
        hs, he = _adaptive_region(oh, input_h, output_size[0])
        for ow in range(output_size[1]):
            ws, we = _adaptive_region(ow, input_w, output_size[1])
            maximum_terms = max(maximum_terms, (he - hs) * (we - ws))
    factor = _rounding_factor(profile, term_count=maximum_terms, multiplication_count=1)
    lower, upper = _outward_interval(
        nominal_lower,
        nominal_upper,
        absolute_sum,
        factor,
        profile=profile,
        name="adaptive-average",
        nonzero_support=support,
        absolute_underflow_radius=_execution_underflow_radius(
            profile, operation_count=maximum_terms + 1
        ),
    )
    return lower, upper, factor


def _reverse_adaptive_avg_absolute_bound(
    destination: Any,
    input_shape: tuple[int, int, int],
    profile: V1AlexNetEnvelopeArithmeticProfile,
) -> tuple[Any, float]:
    channels, input_h, input_w = input_shape
    if int(destination.shape[0]) != channels:
        raise ValueError("adaptive-average cotangent channel axis differs")
    output_h, output_w = (int(value) for value in destination.shape[-2:])
    result = torch.zeros(input_shape, dtype=torch.float64, device=destination.device)
    support = torch.zeros_like(result, dtype=torch.bool)
    incidence = torch.zeros(
        (input_h, input_w), dtype=torch.int32, device=destination.device
    )
    for oh in range(output_h):
        hs, he = _adaptive_region(oh, input_h, output_h)
        for ow in range(output_w):
            ws, we = _adaptive_region(ow, input_w, output_w)
            area = (he - hs) * (we - ws)
            result[:, hs:he, ws:we].add_(
                destination[:, oh, ow].reshape(-1, 1, 1) / area
            )
            support[:, hs:he, ws:we] |= destination[:, oh, ow].reshape(-1, 1, 1) != 0
            incidence[hs:he, ws:we] += 1
    maximum_incidence = max(1, int(incidence.max().item()))
    factor = _rounding_factor(
        profile,
        term_count=maximum_incidence,
        multiplication_count=1,
    )
    result = _outward_reverse_absolute_bound(
        result, factor=factor, support=support, profile=profile,
        operation_count=2 * maximum_incidence + 1, name="adaptive-average cotangent bound")
    return (
        _assert_bound_tensor(
            result, profile=profile, name="adaptive-average cotangent bound"
        ),
        factor,
    )


@dataclass(frozen=True, slots=True)
class V1AlexNetEnvelopeLayerReceipt:
    layer_id: str
    module_path: str
    route_kind: str
    source_shape: tuple[int, ...]
    destination_shape: tuple[int, ...]
    route_count: int
    live_route_count: int
    source_absolute_max: float
    destination_cotangent_absolute_max: float
    route_envelope_max: float
    forward_rounding_factor: float
    backward_rounding_factor: float
    route_rounding_factor: float

    def __post_init__(self) -> None:
        for name in ("layer_id", "module_path", "route_kind"):
            _identity(getattr(self, name), name)
        if not self.source_shape or not self.destination_shape:
            raise ValueError("layer envelope receipt shapes must be nonempty")
        if any(
            int(value) <= 0 for value in (*self.source_shape, *self.destination_shape)
        ):
            raise ValueError("layer envelope receipt shapes must be positive")
        route_count = _positive_int(self.route_count, "route_count")
        live = _nonnegative_int(self.live_route_count, "live_route_count")
        if live > route_count:
            raise ValueError("live route count exceeds route count")
        for name in (
            "source_absolute_max",
            "destination_cotangent_absolute_max",
            "route_envelope_max",
        ):
            _finite(getattr(self, name), name, nonnegative=True)
        for name in (
            "forward_rounding_factor",
            "backward_rounding_factor",
            "route_rounding_factor",
        ):
            if _finite(getattr(self, name), name) < 1.0:
                raise ValueError(f"{name} must be at least one")


@dataclass(frozen=True, slots=True)
class V1AlexNetAttributionEnvelopeDerivationReceipt:
    campaign_contract_hash: str
    model_identity_hash: str
    model_state_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    route_axis_content_hash: str
    stratum_names: tuple[str, str]
    population_stratum_counts: tuple[int, int]
    sealed_population_order_hashes: tuple[str, str]
    target_class_id: int
    architecture_id: str
    topology_hash: str
    graph_semantics_hash: str
    input_box_hash: str
    arithmetic_profile_hash: str
    absolute_envelope_vector_hash: str
    verifier_source_hash: str
    route_chunk_size: int
    layer_receipts: tuple[V1AlexNetEnvelopeLayerReceipt, ...]
    maximum_absolute_envelope: float
    algorithm_id: str = ALEXNET_ENVELOPE_ALGORITHM_ID
    phi_bound_id: str = ALEXNET_ENVELOPE_PHI_BOUND_ID
    observed_data_consumed: bool = False
    terminal_outcomes_opened: bool = False
    stratum_invariant_registered_support: bool = True
    schema: str = ALEXNET_ENVELOPE_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "model_identity_hash",
            "model_state_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "route_axis_hash",
            "route_axis_content_hash",
            "architecture_id",
            "topology_hash",
            "graph_semantics_hash",
            "input_box_hash",
            "arithmetic_profile_hash",
            "absolute_envelope_vector_hash",
            "verifier_source_hash",
        ):
            _identity(getattr(self, name), name)
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, "proposal_generation"),
        )
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, "route_count")
        )
        object.__setattr__(
            self,
            "target_class_id",
            _nonnegative_int(self.target_class_id, "target_class_id"),
        )
        object.__setattr__(
            self,
            "route_chunk_size",
            _positive_int(self.route_chunk_size, "route_chunk_size"),
        )
        object.__setattr__(
            self, "stratum_names", _pair_names(self.stratum_names, "stratum_names")
        )
        object.__setattr__(
            self,
            "population_stratum_counts",
            _pair_positive_ints(
                self.population_stratum_counts, "population_stratum_counts"
            ),
        )
        object.__setattr__(
            self,
            "sealed_population_order_hashes",
            _pair_identities(
                self.sealed_population_order_hashes, "sealed_population_order_hashes"
            ),
        )
        layers = tuple(self.layer_receipts)
        if len(layers) != len(_AFFINE_PATH_BY_LAYER) or any(
            type(value) is not V1AlexNetEnvelopeLayerReceipt for value in layers
        ):
            raise ValueError(
                "AlexNet envelope proof must cover all seven prunable layers"
            )
        if tuple(value.layer_id for value in layers) != tuple(_AFFINE_PATH_BY_LAYER):
            raise ValueError("AlexNet envelope proof layer order is invalid")
        if sum(value.route_count for value in layers) != self.route_count:
            raise ValueError("AlexNet envelope proof does not close the route axis")
        object.__setattr__(self, "layer_receipts", layers)
        _finite(
            self.maximum_absolute_envelope,
            "maximum_absolute_envelope",
            nonnegative=True,
        )
        if (
            self.algorithm_id != ALEXNET_ENVELOPE_ALGORITHM_ID
            or self.phi_bound_id != ALEXNET_ENVELOPE_PHI_BOUND_ID
        ):
            raise ValueError("unsupported AlexNet envelope derivation identity")
        if (
            self.observed_data_consumed is not False
            or self.terminal_outcomes_opened is not False
            or self.stratum_invariant_registered_support is not True
        ):
            raise PermissionError(
                "AlexNet analytic envelope crossed its pre-outcome authority boundary"
            )
        if self.schema != ALEXNET_ENVELOPE_RECEIPT_SCHEMA:
            raise ValueError("unsupported AlexNet envelope proof schema")

    def _payload(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "campaign_contract_hash",
                "model_identity_hash",
                "model_state_hash",
                "query_contract_hash",
                "preprocessing_hash",
                "current_mask_hash",
                "raw_logit_stream_source_hash",
                "sampling_law_hash",
                "proposal_generation",
                "route_count",
                "route_axis_hash",
                "route_axis_content_hash",
                "stratum_names",
                "population_stratum_counts",
                "sealed_population_order_hashes",
                "target_class_id",
                "architecture_id",
                "topology_hash",
                "graph_semantics_hash",
                "input_box_hash",
                "arithmetic_profile_hash",
                "absolute_envelope_vector_hash",
                "verifier_source_hash",
                "route_chunk_size",
                "layer_receipts",
                "maximum_absolute_envelope",
                "algorithm_id",
                "phi_bound_id",
                "observed_data_consumed",
                "terminal_outcomes_opened",
                "stratum_invariant_registered_support",
            )
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        if cached is None:
            cached = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
            object.__setattr__(self, "_receipt_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("AlexNet envelope proof receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("AlexNet envelope proof receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1AlexNetCorrectedAttributionEnvelopeResult:
    certificate: V1CorrectedAttributionEnvelopeCertificate
    derivation_receipt: V1AlexNetAttributionEnvelopeDerivationReceipt
    schema: str = ALEXNET_ENVELOPE_RESULT_SCHEMA

    def __post_init__(self) -> None:
        self.assert_integrity()
        if self.schema != ALEXNET_ENVELOPE_RESULT_SCHEMA:
            raise ValueError("unsupported AlexNet envelope result schema")

    def assert_integrity(self) -> tuple[str, str]:
        """Revalidate the certificate and bind it completely to its proof.

        The receipt proves one stratum-invariant route vector.  The generic
        certificate stores that vector once per stratum, so every stored row
        must reproduce the receipt's vector hash.  Binding only the receipt
        identity would otherwise allow a separately valid certificate to
        substitute an unproved (and potentially smaller) envelope.
        """

        if type(self.certificate) is not V1CorrectedAttributionEnvelopeCertificate:
            raise TypeError(
                "AlexNet envelope result requires the canonical certificate"
            )
        if (
            type(self.derivation_receipt)
            is not V1AlexNetAttributionEnvelopeDerivationReceipt
        ):
            raise TypeError(
                "AlexNet envelope result requires the canonical proof receipt"
            )
        receipt_hash = self.derivation_receipt.assert_integrity()
        certificate_hash = self.certificate.assert_integrity()
        shared_coordinates = (
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "proposal_generation",
            "route_count",
            "route_axis_hash",
            "stratum_names",
            "population_stratum_counts",
            "sealed_population_order_hashes",
            "verifier_source_hash",
        )
        if self.certificate.proof_artifact_hash != receipt_hash or any(
            getattr(self.certificate, name)
            != getattr(self.derivation_receipt, name)
            for name in shared_coordinates
        ):
            raise ValueError(
                "AlexNet envelope certificate differs from its proof receipt"
            )
        envelope_row_hashes = tuple(
            content_hash(row, prefix=_ENVELOPE_VECTOR_PREFIX)
            for row in self.certificate.absolute_upper_envelope_by_stratum
        )
        if not envelope_row_hashes or any(
            row_hash != self.derivation_receipt.absolute_envelope_vector_hash
            for row_hash in envelope_row_hashes
        ):
            raise ValueError(
                "AlexNet envelope certificate vector differs from its proof receipt"
            )
        if not certificate_hash:
            raise RuntimeError("AlexNet envelope certificate hash did not close")
        return receipt_hash, certificate_hash


def _split_gates(
    current_mask: HardMask,
    route_axis: DeclaredRouteAxis,
) -> dict[str, np.ndarray]:
    values = current_mask.to_bool_array()
    return {
        str(block.layer_id): values[int(block.offset) : int(block.stop)].reshape(
            block.shape
        )
        for block in route_axis.route_blocks
    }


def _forward_bounds(
    *,
    model: Any,
    gates: dict[str, np.ndarray],
    input_box: V1RegisteredNormalizedInputBox,
    profile: V1AlexNetEnvelopeArithmeticProfile,
    output_row_chunk: int,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[Any, Any]],
    dict[str, tuple[Any, Any]],
    dict[str, tuple[int, ...]],
    dict[str, float],
    dict[int, tuple[int, int, int]],
    tuple[int, int, int],
    tuple[Any, Any],
]:
    device = next(model.parameters()).device
    lower = torch.tensor(
        input_box.lower_by_channel, dtype=torch.float64, device=device
    ).reshape(-1, 1, 1)
    upper = torch.tensor(
        input_box.upper_by_channel, dtype=torch.float64, device=device
    ).reshape(-1, 1, 1)
    lower = lower.expand(-1, *input_box.spatial_shape)
    upper = upper.expand(-1, *input_box.spatial_shape)
    source_by_layer: dict[str, Any] = {}
    source_interval_by_layer: dict[str, tuple[Any, Any]] = {}
    preactivation_interval_by_layer: dict[str, tuple[Any, Any]] = {}
    destination_shapes: dict[str, tuple[int, ...]] = {}
    forward_factors: dict[str, float] = {}
    pool_input_shapes: dict[int, tuple[int, int, int]] = {}
    for index, module in enumerate(model.features):
        path = f"features.{index}"
        if type(module) is nn.Conv2d:
            layer = _LAYER_BY_AFFINE_PATH[path]
            source_interval_by_layer[layer] = (lower, upper)
            source_by_layer[layer] = _interval_absolute_bound(lower, upper)
            lower, upper, factor = _conv_forward_interval_bound(
                lower, upper, module, gates[layer], profile
            )
            preactivation_interval_by_layer[layer] = (lower, upper)
            destination_shapes[layer] = tuple(int(value) for value in lower.shape)
            forward_factors[layer] = factor
        elif type(module) is nn.ReLU:
            lower = lower.clamp_min(0.0)
            upper = upper.clamp_min(0.0)
            lower, upper = _assert_interval_tensors(
                lower, upper, profile=profile, name="ReLU forward interval"
            )
        elif type(module) is nn.MaxPool2d:
            pool_input_shapes[index] = tuple(int(value) for value in lower.shape)
            lower = F.max_pool2d(
                lower.unsqueeze(0),
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.ceil_mode,
                module.return_indices,
            )[0]
            upper = F.max_pool2d(
                upper.unsqueeze(0),
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.ceil_mode,
                module.return_indices,
            )[0]
            lower, upper = _assert_interval_tensors(
                lower, upper, profile=profile, name="max-pool forward interval"
            )
        else:  # pragma: no cover - graph admission already makes this unreachable.
            raise RuntimeError("unsupported admitted AlexNet feature module")
    avgpool_input_shape = tuple(int(value) for value in lower.shape)
    output_size = _pair(model.avgpool.output_size)
    lower, upper, _ = _adaptive_avg_forward_interval_bound(
        lower, upper, output_size, profile
    )
    lower = lower.reshape(-1)
    upper = upper.reshape(-1)
    for index, module in enumerate(model.classifier):
        path = f"classifier.{index}"
        if type(module) is nn.Dropout:
            if bool(module.training):
                raise RuntimeError("eval-mode Dropout changed during bound propagation")
        elif type(module) is nn.Linear:
            if index == 6:
                if int(module.in_features) != int(lower.numel()):
                    raise ValueError("fixed AlexNet readout input axis differs")
                break
            layer = _LAYER_BY_AFFINE_PATH[path]
            if int(module.in_features) != int(lower.numel()):
                raise ValueError(f"AlexNet affine {path} input axis differs")
            source_interval_by_layer[layer] = (lower, upper)
            source_by_layer[layer] = _interval_absolute_bound(lower, upper)
            lower, upper, factor = _linear_forward_interval_bound(
                lower,
                upper,
                module,
                gates[layer],
                profile,
                output_row_chunk=output_row_chunk,
            )
            preactivation_interval_by_layer[layer] = (lower, upper)
            destination_shapes[layer] = tuple(int(value) for value in lower.shape)
            forward_factors[layer] = factor
        elif type(module) is nn.ReLU:
            lower = lower.clamp_min(0.0)
            upper = upper.clamp_min(0.0)
            lower, upper = _assert_interval_tensors(
                lower, upper, profile=profile, name="ReLU forward interval"
            )
        else:  # pragma: no cover
            raise RuntimeError("unsupported admitted AlexNet classifier module")
    if (
        set(source_by_layer) != set(_AFFINE_PATH_BY_LAYER)
        or set(source_interval_by_layer) != set(_AFFINE_PATH_BY_LAYER)
        or set(preactivation_interval_by_layer) != set(_AFFINE_PATH_BY_LAYER)
    ):
        raise RuntimeError("AlexNet forward bounds did not cover every prunable affine")
    return (
        source_by_layer,
        source_interval_by_layer,
        preactivation_interval_by_layer,
        destination_shapes,
        forward_factors,
        pool_input_shapes,
        avgpool_input_shape,
        (lower, upper),
    )


def build_v1_alexnet_raw_target_logit_interval_certificate(
    *,
    model: Any,
    route_axis: DeclaredRouteAxis,
    current_mask: HardMask,
    query: QuerySpec,
    input_box: V1RegisteredNormalizedInputBox,
    model_identity_hash: str,
    expected_model_state_hash: str,
    preprocessing_hash: str,
) -> V1AlexNetRawTargetLogitIntervalCertificate:
    """Certify one mask-specific raw target-logit interval analytically.

    The target is fixed by ``query`` before propagation.  Only the selected
    target row of the unpruned final classifier is evaluated after the
    mask-specific forward interval reaches the final hidden ReLU.
    """

    current_mask_hash = _direct_hard_mask_hash(
        current_mask, name="raw target-logit interval current mask"
    )
    layout = admit_v1_alexnet_envelope_layout(
        model=model,
        route_axis=route_axis,
        current_mask=current_mask,
        query=query,
        model_identity_hash=model_identity_hash,
        expected_model_state_hash=expected_model_state_hash,
        preprocessing_hash=preprocessing_hash,
        input_box=input_box,
    )
    profile = _arithmetic_profile(model)
    gates = _split_gates(current_mask, route_axis)
    with torch.no_grad():
        (
            _source_by_layer,
            _source_interval_by_layer,
            _preactivation_interval_by_layer,
            _destination_shapes,
            _forward_factors,
            _pool_input_shapes,
            _avgpool_input_shape,
            final_hidden2_interval,
        ) = _forward_bounds(
            model=model,
            gates=gates,
            input_box=input_box,
            profile=profile,
            output_row_chunk=_AFFINE_BOUND_OUTPUT_ROW_CHUNK,
        )
        hidden_lower, hidden_upper = final_hidden2_interval
        readout_lower, readout_upper, readout_factor = (
            _linear_forward_interval_bound(
                hidden_lower,
                hidden_upper,
                model.classifier[6],
                None,
                profile,
                output_row_chunk=1,
                output_row_start=layout.target_class_id,
                output_row_stop=layout.target_class_id + 1,
            )
        )
    if readout_lower.numel() != 1 or readout_upper.numel() != 1:
        raise RuntimeError("target-row interval propagation changed scalar shape")

    # Close every live identity after the contractions so a concurrent graph,
    # parameter, mask, query, route-axis, or input-box change cannot inherit
    # the pre-propagation admission receipt.
    closing_layout = admit_v1_alexnet_envelope_layout(
        model=model,
        route_axis=route_axis,
        current_mask=current_mask,
        query=query,
        model_identity_hash=model_identity_hash,
        expected_model_state_hash=expected_model_state_hash,
        preprocessing_hash=preprocessing_hash,
        input_box=input_box,
    )
    if closing_layout != layout:
        raise RuntimeError("AlexNet layout changed during target-logit derivation")
    if (
        _direct_hard_mask_hash(
            current_mask, name="raw target-logit interval current mask"
        )
        != current_mask_hash
    ):
        raise RuntimeError("hard mask changed during target-logit derivation")

    certificate = V1AlexNetRawTargetLogitIntervalCertificate(
        architecture_id=layout.architecture_id,
        topology_hash=layout.topology_hash,
        route_axis_hash=layout.route_axis_hash,
        route_axis_content_hash=layout.route_axis_content_hash,
        route_count=layout.route_count,
        current_mask_hash=current_mask_hash,
        model_identity_hash=layout.model_identity_hash,
        model_state_hash=layout.model_state_hash,
        query_contract_hash=layout.query_contract_hash,
        target_class_id=layout.target_class_id,
        preprocessing_hash=layout.preprocessing_hash,
        input_box_hash=layout.input_box_hash,
        graph_semantics_hash=layout.graph_semantics_hash,
        arithmetic_profile_hash=layout.arithmetic_profile_hash,
        layout_hash=layout.layout_hash,
        raw_target_logit_lower=float(readout_lower[0].item()),
        raw_target_logit_upper=float(readout_upper[0].item()),
        readout_rounding_factor=readout_factor,
        verifier_source_hash=_source_sha256(),
    )
    certificate.assert_integrity()
    return certificate


def _backward_bounds(
    *,
    model: Any,
    gates: dict[str, np.ndarray],
    target_class_id: int,
    source_by_layer: dict[str, Any],
    pool_input_shapes: dict[int, tuple[int, int, int]],
    avgpool_input_shape: tuple[int, int, int],
    profile: V1AlexNetEnvelopeArithmeticProfile,
    output_row_chunk: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    readout = model.classifier[6]
    destination = readout.weight.detach()[target_class_id].to(torch.float64).abs()
    readout_factor = _rounding_factor(profile, term_count=1, multiplication_count=1)
    destination = torch.nextafter(
        destination * readout_factor, torch.full_like(destination, torch.inf)
    )
    _assert_bound_tensor(
        destination, profile=profile, name="fixed readout cotangent bound"
    )

    cotangent_by_layer: dict[str, Any] = {"readout.hidden2": destination}
    backward_factors: dict[str, float] = {"readout.hidden2": readout_factor}
    destination, factor = _linear_transpose_absolute_bound(
        destination,
        model.classifier[4],
        gates["readout.hidden2"],
        profile,
        output_row_chunk=output_row_chunk,
    )
    cotangent_by_layer["readout.hidden1"] = destination
    backward_factors["readout.hidden1"] = factor
    destination, _ = _linear_transpose_absolute_bound(
        destination,
        model.classifier[1],
        gates["readout.hidden1"],
        profile,
        output_row_chunk=output_row_chunk,
    )
    avg_output_shape = tuple(int(value) for value in model.avgpool.output_size)
    if len(avg_output_shape) == 1:
        avg_output_shape = (avg_output_shape[0], avg_output_shape[0])
    expected_flat = (
        int(avgpool_input_shape[0])
        * int(avg_output_shape[0])
        * int(avg_output_shape[1])
    )
    if int(destination.numel()) != expected_flat:
        raise ValueError("classifier cotangent cannot reshape to adaptive-pool output")
    destination = destination.reshape(avgpool_input_shape[0], *avg_output_shape)
    destination, _ = _reverse_adaptive_avg_absolute_bound(
        destination, avgpool_input_shape, profile
    )

    # features.12 is the third MaxPool, immediately after stage 5.
    destination, pool_factor = _reverse_maxpool_absolute_bound(
        destination, model.features[12], pool_input_shapes[12], profile
    )
    cotangent_by_layer["features.stage5.conv"] = destination
    backward_factors["features.stage5.conv"] = pool_factor
    destination, factor = _conv_transpose_absolute_bound(
        destination,
        model.features[10],
        gates["features.stage5.conv"],
        tuple(int(value) for value in source_by_layer["features.stage5.conv"].shape),
        profile,
    )
    cotangent_by_layer["features.stage4.conv"] = destination
    backward_factors["features.stage4.conv"] = factor
    destination, factor = _conv_transpose_absolute_bound(
        destination,
        model.features[8],
        gates["features.stage4.conv"],
        tuple(int(value) for value in source_by_layer["features.stage4.conv"].shape),
        profile,
    )
    cotangent_by_layer["features.stage3.conv"] = destination
    backward_factors["features.stage3.conv"] = factor
    destination, _ = _conv_transpose_absolute_bound(
        destination,
        model.features[6],
        gates["features.stage3.conv"],
        tuple(int(value) for value in source_by_layer["features.stage3.conv"].shape),
        profile,
    )
    destination, pool_factor = _reverse_maxpool_absolute_bound(
        destination, model.features[5], pool_input_shapes[5], profile
    )
    cotangent_by_layer["features.stage2.conv"] = destination
    backward_factors["features.stage2.conv"] = pool_factor
    destination, _ = _conv_transpose_absolute_bound(
        destination,
        model.features[3],
        gates["features.stage2.conv"],
        tuple(int(value) for value in source_by_layer["features.stage2.conv"].shape),
        profile,
    )
    destination, pool_factor = _reverse_maxpool_absolute_bound(
        destination, model.features[2], pool_input_shapes[2], profile
    )
    cotangent_by_layer["features.stage1.conv"] = destination
    backward_factors["features.stage1.conv"] = pool_factor
    if set(cotangent_by_layer) != set(_AFFINE_PATH_BY_LAYER):
        raise RuntimeError(
            "AlexNet cotangent bounds did not cover every prunable affine"
        )
    return cotangent_by_layer, backward_factors


def _conv_route_envelope(
    *,
    source: Any,
    destination: Any,
    module: Any,
    gate: np.ndarray,
    profile: V1AlexNetEnvelopeArithmeticProfile,
    route_chunk_size: int,
) -> tuple[np.ndarray, float]:
    kernel_h, kernel_w = _pair(module.kernel_size)
    output_count = int(module.out_channels)
    input_count = int(module.in_channels)
    routes_per_output = input_count * kernel_h * kernel_w
    output_rows = max(1, int(route_chunk_size) // routes_per_output)
    patches = F.unfold(
        source.unsqueeze(0),
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )[0]
    spatial = int(patches.shape[1])
    if (
        destination.ndim != 3
        or int(destination.shape[0]) != output_count
        or int(destination.shape[1] * destination.shape[2]) != spatial
    ):
        raise ValueError(
            "convolution route envelope source/destination incidence differs"
        )
    factor = _rounding_factor(
        profile,
        term_count=spatial,
        multiplication_count=3,
    )
    result = np.empty(
        tuple(int(value) for value in module.weight.shape), dtype=np.float64
    )
    for start in range(0, output_count, output_rows):
        stop = min(output_count, start + output_rows)
        destination_chunk = destination[start:stop].reshape(stop - start, spatial)
        incidence = torch.mm(destination_chunk, patches.T)
        weight = (
            module.weight.detach()[start:stop]
            .to(torch.float64)
            .abs()
            .reshape(stop - start, -1)
        )
        gate_chunk = _effective_gate_tensor(gate[start:stop], module=module).reshape(
            stop - start, -1
        )
        base = incidence * weight * gate_chunk
        values = torch.where(
            base == 0.0,
            torch.zeros_like(base),
            torch.nextafter(
                base * factor,
                torch.full_like(base, torch.inf),
            ),
        )
        _assert_bound_tensor(values, profile=profile, name="conv route envelope")
        result[start:stop] = (
            values.reshape(stop - start, input_count, kernel_h, kernel_w)
            .detach()
            .cpu()
            .numpy()
        )
    # nextafter(0,+inf) would violate the exact-zero pruned-route contract.
    result[np.logical_not(gate)] = 0.0
    return result, factor


def _linear_route_envelope(
    *,
    source: Any,
    destination: Any,
    module: Any,
    gate: np.ndarray,
    profile: V1AlexNetEnvelopeArithmeticProfile,
    route_chunk_size: int,
) -> tuple[np.ndarray, float]:
    input_count = int(module.in_features)
    output_rows = max(1, int(route_chunk_size) // input_count)
    factor = _rounding_factor(profile, term_count=1, multiplication_count=3)
    result = np.empty(
        tuple(int(value) for value in module.weight.shape), dtype=np.float64
    )
    for start in range(0, int(module.out_features), output_rows):
        stop = min(int(module.out_features), start + output_rows)
        weight = module.weight.detach()[start:stop].to(torch.float64).abs()
        gate_chunk = _effective_gate_tensor(gate[start:stop], module=module)
        base = (
            weight
            * source.reshape(1, -1)
            * destination[start:stop].reshape(-1, 1)
            * gate_chunk
        )
        values = torch.where(
            base == 0.0,
            torch.zeros_like(base),
            torch.nextafter(
                base * factor,
                torch.full_like(base, torch.inf),
            ),
        )
        _assert_bound_tensor(values, profile=profile, name="linear route envelope")
        result[start:stop] = values.detach().cpu().numpy()
    result[np.logical_not(gate)] = 0.0
    return result, factor


def _conv_action_drive_interval_bound(
    *,
    source_lower: Any,
    source_upper: Any,
    module: Any,
    selected_gate: np.ndarray,
    profile: V1AlexNetEnvelopeArithmeticProfile,
) -> tuple[Any, Any]:
    """Bound the signed no-bias drive of one selected convolution action."""

    if selected_gate.shape != tuple(int(value) for value in module.weight.shape):
        raise ValueError("convolution action gate differs from the weight axis")
    weight = module.weight.detach().to(torch.float64)
    weight = weight * _effective_gate_tensor(selected_gate, module=module)
    positive = weight.clamp_min(0.0)
    negative = weight.clamp_max(0.0)
    common = (
        module.stride,
        module.padding,
        module.dilation,
        module.groups,
    )
    nominal_lower = (
        F.conv2d(source_lower.unsqueeze(0), positive, None, *common)[0]
        + F.conv2d(source_upper.unsqueeze(0), negative, None, *common)[0]
    )
    nominal_upper = (
        F.conv2d(source_upper.unsqueeze(0), positive, None, *common)[0]
        + F.conv2d(source_lower.unsqueeze(0), negative, None, *common)[0]
    )
    source_absolute = _interval_absolute_bound(source_lower, source_upper)
    absolute_sum = F.conv2d(source_absolute.unsqueeze(0), weight.abs(), None, *common)[
        0
    ]
    support = (
        F.conv2d(
            (source_absolute != 0.0).to(torch.float64).unsqueeze(0),
            (weight != 0.0).to(torch.float64),
            None,
            *common,
        )[0]
        > 0.0
    )
    terms = (
        int(module.in_channels // module.groups)
        * int(module.kernel_size[0])
        * int(module.kernel_size[1])
    )
    factor = _rounding_factor(
        profile,
        term_count=2 * terms,
        multiplication_count=1,
    )
    return _outward_interval(
        nominal_lower,
        nominal_upper,
        absolute_sum,
        factor,
        profile=profile,
        name="convolution action drive",
        nonzero_support=support,
        absolute_underflow_radius=_execution_underflow_radius(
            profile, operation_count=2 * (terms + 1)
        ),
    )


def _linear_action_drive_interval_bound(
    *,
    source_lower: Any,
    source_upper: Any,
    module: Any,
    selected_gate: np.ndarray,
    profile: V1AlexNetEnvelopeArithmeticProfile,
    output_row_chunk: int,
) -> tuple[Any, Any]:
    """Bound the signed no-bias drive of one selected linear action."""

    if selected_gate.shape != tuple(int(value) for value in module.weight.shape):
        raise ValueError("linear action gate differs from the weight axis")
    rows = _positive_int(output_row_chunk, "linear action output row chunk")
    lower = torch.zeros(
        int(module.out_features), dtype=torch.float64, device=module.weight.device
    )
    upper = torch.zeros_like(lower)
    source_absolute = _interval_absolute_bound(source_lower, source_upper)
    terms = int(module.in_features)
    factor = _rounding_factor(
        profile,
        term_count=2 * terms,
        multiplication_count=1,
    )
    underflow_radius = _execution_underflow_radius(
        profile, operation_count=2 * (terms + 1)
    )
    active_rows = np.flatnonzero(
        np.any(selected_gate.reshape(int(module.out_features), -1), axis=1)
    )
    for start in range(0, int(active_rows.size), rows):
        row_index_numpy = active_rows[start : start + rows]
        row_index = torch.as_tensor(
            row_index_numpy,
            dtype=torch.long,
            device=module.weight.device,
        )
        weight = module.weight.detach().index_select(0, row_index).to(torch.float64)
        weight = weight * _effective_gate_tensor(
            selected_gate[row_index_numpy], module=module
        )
        positive = weight.clamp_min(0.0)
        negative = weight.clamp_max(0.0)
        nominal_lower = torch.mv(positive, source_lower) + torch.mv(
            negative, source_upper
        )
        nominal_upper = torch.mv(positive, source_upper) + torch.mv(
            negative, source_lower
        )
        absolute_sum = torch.mv(weight.abs(), source_absolute)
        support = (
            torch.mv(
                (weight != 0.0).to(torch.float64),
                (source_absolute != 0.0).to(torch.float64),
            )
            > 0.0
        )
        chunk_lower, chunk_upper = _outward_interval(
            nominal_lower,
            nominal_upper,
            absolute_sum,
            factor,
            profile=profile,
            name="linear action drive",
            nonzero_support=support,
            absolute_underflow_radius=underflow_radius,
        )
        lower.index_copy_(0, row_index, chunk_lower)
        upper.index_copy_(0, row_index, chunk_upper)
    return _assert_interval_tensors(
        lower,
        upper,
        profile=profile,
        name="linear action-drive interval",
    )


def _outward_positive_ratio(numerator: Any, denominator: Any) -> Any:
    """Return an upward binary64 enclosure of ``numerator / denominator``."""

    active = numerator > 0.0
    if bool(torch.any(torch.logical_and(active, denominator <= 0.0)).item()):
        raise FloatingPointError("positive-ratio denominator is not positive")
    numerator_upper = torch.nextafter(numerator, torch.full_like(numerator, torch.inf))
    denominator_lower = torch.nextafter(denominator, torch.zeros_like(denominator))
    nominal = torch.where(
        active,
        numerator_upper / denominator_lower,
        torch.zeros_like(numerator),
    )
    return torch.where(
        active,
        torch.nextafter(nominal, torch.full_like(nominal, torch.inf)),
        torch.zeros_like(nominal),
    )


def _relu_path_fraction_upper_bound(
    *,
    preactivation_lower: Any,
    preactivation_upper: Any,
    module: Any,
    profile: V1AlexNetEnvelopeArithmeticProfile,
) -> Any:
    """Supremum of canonical ``phi(a,b)`` over an admitted ``a`` interval.

    For fixed bias baseline ``b``, the ReLU path-open fraction is monotone in
    the current preactivation ``a``.  Its supremum is therefore attained at
    the admitted upper endpoint.  This is the Aumann--Shapley path fraction;
    it is deliberately not the finite-removal ``Delta h`` response.
    """

    lower, upper = _assert_interval_tensors(
        preactivation_lower,
        preactivation_upper,
        profile=profile,
        name="action phi preactivation interval",
    )
    if module.bias is None:
        bias = torch.zeros(
            int(
                module.out_channels
                if type(module) is nn.Conv2d
                else module.out_features
            ),
            dtype=torch.float64,
            device=upper.device,
        )
    else:
        bias = module.bias.detach().to(torch.float64)
    if type(module) is nn.Conv2d:
        bias = bias.reshape(-1, 1, 1)
    elif type(module) is not nn.Linear:
        raise TypeError("path-fraction bounds require a convolution or linear module")
    bias = torch.broadcast_to(bias, upper.shape)

    result = torch.zeros_like(upper)
    positive_bias = bias > 0.0
    nonpositive_bias = torch.logical_not(positive_bias)
    result = torch.where(
        torch.logical_and(positive_bias, upper >= 0.0),
        torch.ones_like(result),
        result,
    )
    leaving = torch.logical_and(positive_bias, upper < 0.0)
    leaving_ratio = _outward_positive_ratio(
        torch.where(leaving, bias, torch.zeros_like(bias)),
        torch.where(leaving, bias - upper, torch.ones_like(bias)),
    )
    result = torch.where(leaving, leaving_ratio, result)

    entering = torch.logical_and(nonpositive_bias, upper > 0.0)
    zero_bias = bias == 0.0
    result = torch.where(
        torch.logical_and(entering, zero_bias),
        torch.ones_like(result),
        result,
    )
    negative_bias_entering = torch.logical_and(entering, bias < 0.0)
    entering_ratio = _outward_positive_ratio(
        torch.where(negative_bias_entering, upper, torch.zeros_like(upper)),
        torch.where(negative_bias_entering, upper - bias, torch.ones_like(upper)),
    )
    result = torch.where(negative_bias_entering, entering_ratio, result)
    result = result.clamp(min=0.0, max=1.0)
    # ``lower`` is intentionally consumed by interval validation above.  The
    # monotonicity proof needs only the upper endpoint after that validation.
    del lower
    return _assert_bound_tensor(
        result, profile=profile, name="canonical ReLU path-fraction upper bound"
    )


def _outward_nonnegative_product_sum(
    factors: Sequence[Any],
    *,
    profile: V1AlexNetEnvelopeArithmeticProfile,
    name: str,
) -> float:
    """Outward product followed by an outward, cancellation-free ``fsum``."""

    tensors = tuple(factors)
    if not tensors:
        raise ValueError(f"{name} requires at least one factor")
    shape = tensors[0].shape
    for index, value in enumerate(tensors):
        _assert_bound_tensor(value, profile=profile, name=f"{name} factor {index}")
        if value.shape != shape:
            raise ValueError(f"{name} factors must share one destination-unit axis")
    product = tensors[0]
    support = product != 0.0
    for value in tensors[1:]:
        support = torch.logical_and(support, value != 0.0)
        nominal = product * value
        product = torch.where(
            support,
            torch.nextafter(nominal, torch.full_like(nominal, torch.inf)),
            torch.zeros_like(nominal),
        )
        _assert_bound_tensor(product, profile=profile, name=f"{name} product")
    flattened = product.detach().reshape(-1).cpu().numpy()
    try:
        total = math.fsum(flattened.tolist())
    except OverflowError as error:
        raise FloatingPointError(f"{name} sum overflowed binary64") from error
    if not math.isfinite(total) or total < 0.0:
        raise FloatingPointError(f"{name} sum is nonfinite or negative")
    if total == 0.0:
        return 0.0
    upper = math.nextafter(total, math.inf)
    if not math.isfinite(upper) or upper > profile.numerical_ceiling:
        raise FloatingPointError(f"{name} exceeds the numerical admission ceiling")
    return upper


def _outward_nonnegative_scalar_sum(
    values: Sequence[float],
    *,
    profile: V1AlexNetEnvelopeArithmeticProfile,
    name: str,
) -> float:
    checked = tuple(_finite(value, name, nonnegative=True) for value in values)
    try:
        total = math.fsum(checked)
    except OverflowError as error:
        raise FloatingPointError(f"{name} overflowed binary64") from error
    if total == 0.0:
        return 0.0
    upper = math.nextafter(total, math.inf)
    if not math.isfinite(upper) or upper > profile.numerical_ceiling:
        raise FloatingPointError(f"{name} exceeds the numerical admission ceiling")
    return upper


def build_v1_alexnet_corrected_attribution_envelope(
    *,
    model: Any,
    route_axis: DeclaredRouteAxis,
    current_mask: HardMask,
    query: QuerySpec,
    input_box: V1RegisteredNormalizedInputBox,
    campaign_contract_hash: str,
    model_identity_hash: str,
    expected_model_state_hash: str,
    preprocessing_hash: str,
    raw_logit_stream_source_hash: str,
    sampling_law_hash: str,
    proposal_generation: int,
    stratum_names: tuple[str, str],
    population_stratum_counts: tuple[int, int],
    sealed_population_order_hashes: tuple[str, str],
    route_chunk_size: int = 1_048_576,
) -> V1AlexNetCorrectedAttributionEnvelopeResult:
    """Build a proposal-statistics-compatible analytic envelope certificate."""

    route_chunk = _positive_int(route_chunk_size, "route_chunk_size")
    current_mask_hash = _direct_hard_mask_hash(
        current_mask,
        name="AlexNet envelope current mask",
    )
    campaign = _identity(campaign_contract_hash, "campaign_contract_hash")
    raw_stream = _identity(raw_logit_stream_source_hash, "raw_logit_stream_source_hash")
    sampling = _identity(sampling_law_hash, "sampling_law_hash")
    generation = _nonnegative_int(proposal_generation, "proposal_generation")
    names = _pair_names(stratum_names, "stratum_names")
    populations = _pair_positive_ints(
        population_stratum_counts, "population_stratum_counts"
    )
    orders = _pair_identities(
        sealed_population_order_hashes, "sealed_population_order_hashes"
    )
    layout = admit_v1_alexnet_envelope_layout(
        model=model,
        route_axis=route_axis,
        current_mask=current_mask,
        query=query,
        model_identity_hash=model_identity_hash,
        expected_model_state_hash=expected_model_state_hash,
        preprocessing_hash=preprocessing_hash,
        input_box=input_box,
    )
    profile = _arithmetic_profile(model)
    gates = _split_gates(current_mask, route_axis)
    # Bound propagation is a matrix-vector workload, not a route-output
    # workload.  This keeps the largest float64 AlexNet slab below ~19 MiB
    # while avoiding hundreds of tiny GEMV launches.
    output_row_chunk = _AFFINE_BOUND_OUTPUT_ROW_CHUNK
    with torch.no_grad():
        (
            source_by_layer,
            _source_interval_by_layer,
            _preactivation_interval_by_layer,
            destination_shapes,
            forward_factors,
            pool_input_shapes,
            avgpool_input_shape,
            _final_hidden2_interval,
        ) = _forward_bounds(
            model=model,
            gates=gates,
            input_box=input_box,
            profile=profile,
            output_row_chunk=output_row_chunk,
        )
        cotangent_by_layer, backward_factors = _backward_bounds(
            model=model,
            gates=gates,
            target_class_id=layout.target_class_id,
            source_by_layer=source_by_layer,
            pool_input_shapes=pool_input_shapes,
            avgpool_input_shape=avgpool_input_shape,
            profile=profile,
            output_row_chunk=output_row_chunk,
        )

        envelope = np.empty(route_axis.route_count, dtype=np.float64)
        layer_receipts: list[V1AlexNetEnvelopeLayerReceipt] = []
        for block in route_axis.route_blocks:
            layer = str(block.layer_id)
            path = _AFFINE_PATH_BY_LAYER[layer]
            module = model.get_submodule(path)
            source = source_by_layer[layer]
            destination = cotangent_by_layer[layer]
            gate = gates[layer]
            if block.route_kind == "conv_kernel_weight":
                block_envelope, route_factor = _conv_route_envelope(
                    source=source,
                    destination=destination,
                    module=module,
                    gate=gate,
                    profile=profile,
                    route_chunk_size=route_chunk,
                )
            elif block.route_kind == "linear_weight":
                block_envelope, route_factor = _linear_route_envelope(
                    source=source,
                    destination=destination,
                    module=module,
                    gate=gate,
                    profile=profile,
                    route_chunk_size=route_chunk,
                )
            else:  # pragma: no cover - layout admission rejects this.
                raise RuntimeError("unsupported admitted AlexNet route kind")
            if block_envelope.shape != block.shape:
                raise RuntimeError("AlexNet route envelope changed the block shape")
            flat = block_envelope.reshape(-1)
            envelope[int(block.offset) : int(block.stop)] = flat
            layer_receipts.append(
                V1AlexNetEnvelopeLayerReceipt(
                    layer_id=layer,
                    module_path=path,
                    route_kind=str(block.route_kind),
                    source_shape=tuple(int(value) for value in source.shape),
                    destination_shape=destination_shapes[layer],
                    route_count=block.route_count,
                    live_route_count=int(np.count_nonzero(gate)),
                    source_absolute_max=float(source.max().item()),
                    destination_cotangent_absolute_max=float(destination.max().item()),
                    route_envelope_max=float(flat.max()) if flat.size else 0.0,
                    forward_rounding_factor=forward_factors[layer],
                    backward_rounding_factor=backward_factors[layer],
                    route_rounding_factor=route_factor,
                )
            )
    if not np.isfinite(envelope).all() or np.any(envelope < 0.0):
        raise FloatingPointError("AlexNet route envelope is nonfinite or negative")
    if float(envelope.max()) > profile.numerical_ceiling:
        raise FloatingPointError("AlexNet route envelope exceeds the numerical ceiling")
    mask_values = current_mask.to_bool_array()
    if np.any(envelope[np.logical_not(mask_values)] != 0.0):
        raise RuntimeError("AlexNet route envelope failed exact pruned-route zeroing")
    envelope[envelope == 0.0] = 0.0
    envelope.setflags(write=False)

    # Re-read the tensor state after all bound construction.  This closes the
    # otherwise possible race in which a concurrent writer changes weights
    # between layout admission and a later layer contraction.
    if state_dict_digest(model) != layout.model_state_hash:
        raise RuntimeError("AlexNet model state changed during envelope derivation")
    if (
        _direct_hard_mask_hash(
            current_mask,
            name="AlexNet envelope current mask",
        )
        != current_mask_hash
    ):
        raise RuntimeError("AlexNet hard mask changed during envelope derivation")
    verifier_source_hash = _source_sha256()
    envelope_hash = content_hash(envelope, prefix=_ENVELOPE_VECTOR_PREFIX)
    receipt = V1AlexNetAttributionEnvelopeDerivationReceipt(
        campaign_contract_hash=campaign,
        model_identity_hash=layout.model_identity_hash,
        model_state_hash=layout.model_state_hash,
        query_contract_hash=layout.query_contract_hash,
        preprocessing_hash=layout.preprocessing_hash,
        current_mask_hash=current_mask_hash,
        raw_logit_stream_source_hash=raw_stream,
        sampling_law_hash=sampling,
        proposal_generation=generation,
        route_count=route_axis.route_count,
        route_axis_hash=route_axis.route_axis_hash,
        route_axis_content_hash=layout.route_axis_content_hash,
        stratum_names=names,
        population_stratum_counts=populations,
        sealed_population_order_hashes=orders,
        target_class_id=layout.target_class_id,
        architecture_id=layout.architecture_id,
        topology_hash=layout.topology_hash,
        graph_semantics_hash=layout.graph_semantics_hash,
        input_box_hash=layout.input_box_hash,
        arithmetic_profile_hash=layout.arithmetic_profile_hash,
        absolute_envelope_vector_hash=envelope_hash,
        verifier_source_hash=verifier_source_hash,
        route_chunk_size=route_chunk,
        layer_receipts=tuple(layer_receipts),
        maximum_absolute_envelope=float(envelope.max()),
    )
    certificate = V1CorrectedAttributionEnvelopeCertificate(
        campaign_contract_hash=campaign,
        model_identity_hash=layout.model_identity_hash,
        query_contract_hash=layout.query_contract_hash,
        preprocessing_hash=layout.preprocessing_hash,
        current_mask_hash=current_mask_hash,
        raw_logit_stream_source_hash=raw_stream,
        sampling_law_hash=sampling,
        proposal_generation=generation,
        route_count=route_axis.route_count,
        route_axis_hash=route_axis.route_axis_hash,
        stratum_names=names,
        population_stratum_counts=populations,
        sealed_population_order_hashes=orders,
        # Both strata share the same registered pixel support.  Broadcast
        # avoids allocating a second input copy before the canonical
        # certificate constructor freezes its own (2, route) storage.
        absolute_upper_envelope_by_stratum=np.broadcast_to(
            envelope, (2, route_axis.route_count)
        ),
        verifier_source_hash=verifier_source_hash,
        proof_artifact_hash=receipt.receipt_hash,
    )
    return V1AlexNetCorrectedAttributionEnvelopeResult(
        certificate=certificate,
        derivation_receipt=receipt,
    )


def build_v1_alexnet_route_sum_endpoint_envelope(
    *,
    envelope_result: V1AlexNetCorrectedAttributionEnvelopeResult,
    ladder: V1FrozenCandidateLadder,
) -> V1FrozenLadderEndpointEnvelopeCertificate:
    """Build a frozen-ladder endpoint certificate from an AlexNet route envelope.

    This wrapper verifies that the AlexNet analytic route envelope result is
    still internally sealed, then delegates the endpoint aggregation to the
    generic ladder-statistics producer.
    """

    if type(envelope_result) is not V1AlexNetCorrectedAttributionEnvelopeResult:
        raise TypeError("AlexNet endpoint envelope requires the AlexNet result type")
    envelope_result.assert_integrity()
    return build_v1_frozen_ladder_route_sum_endpoint_envelope(
        ladder=ladder,
        envelope_certificate=envelope_result.certificate,
    )


def build_v1_alexnet_action_drive_endpoint_envelope(
    *,
    model: Any,
    route_axis: DeclaredRouteAxis,
    current_mask: HardMask,
    query: QuerySpec,
    input_box: V1RegisteredNormalizedInputBox,
    envelope_result: V1AlexNetCorrectedAttributionEnvelopeResult,
    ladder: V1FrozenCandidateLadder,
) -> V1FrozenLadderEndpointEnvelopeCertificate:
    """Certify frozen endpoints after shared-unit signed-drive aggregation.

    For one endpoint ``S`` and destination unit ``u`` in layer ``ell``, this
    builder first encloses the signed selected-route drive

    ``D_{ell,u}(S) = sum_{r in S intersect ell,u} c_r``.

    It then applies the canonical Aumann--Shapley envelope

    ``sum_{ell,u} |G_{ell,u}| sup(phi_{ell,u}) |D_{ell,u}(S)|``.

    Thus signed selected weights can cancel before the absolute value, but
    ``phi`` remains the current-state path-open fraction and is never replaced
    by finite-removal ``Delta h``.  The final coordinate is the minimum of
    this bound and the independently sealed singleton-route sum, after exact
    coordinate/action-axis alignment is asserted and bound into the proof.
    The result has proposal-envelope authority only.
    """

    if type(envelope_result) is not V1AlexNetCorrectedAttributionEnvelopeResult:
        raise TypeError("action-drive envelope requires the AlexNet result type")
    if type(ladder) is not V1FrozenCandidateLadder:
        raise TypeError("action-drive envelope requires a typed frozen ladder")
    current_mask_hash = _direct_hard_mask_hash(
        current_mask,
        name="AlexNet action-drive current mask",
    )
    receipt_hash, envelope_hash = envelope_result.assert_integrity()
    ladder_hash = ladder.assert_integrity()

    receipt = envelope_result.derivation_receipt
    layout = admit_v1_alexnet_envelope_layout(
        model=model,
        route_axis=route_axis,
        current_mask=current_mask,
        query=query,
        model_identity_hash=receipt.model_identity_hash,
        expected_model_state_hash=receipt.model_state_hash,
        preprocessing_hash=receipt.preprocessing_hash,
        input_box=input_box,
    )
    if (
        current_mask_hash != receipt.current_mask_hash
        or current_mask_hash != envelope_result.certificate.current_mask_hash
        or current_mask_hash != ladder.current_mask_hash
        or layout.route_axis_hash != receipt.route_axis_hash
        or layout.route_axis_content_hash != receipt.route_axis_content_hash
        or layout.query_contract_hash != receipt.query_contract_hash
        or layout.input_box_hash != receipt.input_box_hash
        or layout.graph_semantics_hash != receipt.graph_semantics_hash
        or layout.arithmetic_profile_hash != receipt.arithmetic_profile_hash
        or layout.model_state_hash != receipt.model_state_hash
    ):
        raise ValueError(
            "live AlexNet action-drive coordinates differ from the route proof"
        )

    route_sum = build_v1_alexnet_route_sum_endpoint_envelope(
        envelope_result=envelope_result,
        ladder=ladder,
    )
    route_sum_hash = route_sum.assert_integrity()
    if (
        route_sum.ladder_hash != ladder_hash
        or route_sum.endpoint_action_hashes != ladder.action_hashes
        or route_sum.stratum_names != envelope_result.certificate.stratum_names
        or route_sum.route_count != ladder.route_count
        or route_sum.route_axis_hash != ladder.route_axis_hash
        or route_sum.current_mask_hash != ladder.current_mask_hash
        or route_sum.current_mask_hash != current_mask_hash
    ):
        raise ValueError(
            "route-sum and action-drive endpoint certificates are not aligned"
        )

    profile = _arithmetic_profile(model)
    gates = _split_gates(current_mask, route_axis)
    output_row_chunk = _AFFINE_BOUND_OUTPUT_ROW_CHUNK
    with torch.no_grad():
        (
            source_by_layer,
            source_interval_by_layer,
            preactivation_interval_by_layer,
            _destination_shapes,
            _forward_factors,
            pool_input_shapes,
            avgpool_input_shape,
            _final_hidden2_interval,
        ) = _forward_bounds(
            model=model,
            gates=gates,
            input_box=input_box,
            profile=profile,
            output_row_chunk=output_row_chunk,
        )
        cotangent_by_layer, _backward_factors = _backward_bounds(
            model=model,
            gates=gates,
            target_class_id=layout.target_class_id,
            source_by_layer=source_by_layer,
            pool_input_shapes=pool_input_shapes,
            avgpool_input_shape=avgpool_input_shape,
            profile=profile,
            output_row_chunk=output_row_chunk,
        )

        selected_routes = np.zeros(route_axis.route_count, dtype=np.bool_)
        action_bounds: list[float] = []
        layer_bounds_by_endpoint: list[tuple[float, ...]] = []
        selected_live_routes_by_endpoint: list[tuple[int, ...]] = []
        for action in ladder.actions:
            for new_route_index in ladder.iter_shell_index_chunks(
                endpoint_index=action.endpoint_index
            ):
                selected_routes[new_route_index] = True

            layer_bounds: list[float] = []
            selected_live_counts: list[int] = []
            for block in route_axis.route_blocks:
                layer = str(block.layer_id)
                start = int(block.offset)
                stop = int(block.stop)
                selected_gate = np.logical_and(
                    selected_routes[start:stop].reshape(block.shape),
                    gates[layer],
                )
                selected_live_count = int(np.count_nonzero(selected_gate))
                selected_live_counts.append(selected_live_count)
                if selected_live_count == 0:
                    layer_bounds.append(0.0)
                    continue

                path = _AFFINE_PATH_BY_LAYER[layer]
                module = model.get_submodule(path)
                source_lower, source_upper = source_interval_by_layer[layer]
                if block.route_kind == "conv_kernel_weight":
                    drive_lower, drive_upper = _conv_action_drive_interval_bound(
                        source_lower=source_lower,
                        source_upper=source_upper,
                        module=module,
                        selected_gate=selected_gate,
                        profile=profile,
                    )
                elif block.route_kind == "linear_weight":
                    drive_lower, drive_upper = _linear_action_drive_interval_bound(
                        source_lower=source_lower,
                        source_upper=source_upper,
                        module=module,
                        selected_gate=selected_gate,
                        profile=profile,
                        output_row_chunk=output_row_chunk,
                    )
                else:  # pragma: no cover - layout admission rejects this.
                    raise RuntimeError("unsupported admitted AlexNet route kind")
                drive_absolute = _interval_absolute_bound(drive_lower, drive_upper)
                preactivation_lower, preactivation_upper = (
                    preactivation_interval_by_layer[layer]
                )
                phi_upper = _relu_path_fraction_upper_bound(
                    preactivation_lower=preactivation_lower,
                    preactivation_upper=preactivation_upper,
                    module=module,
                    profile=profile,
                )
                cotangent_upper = cotangent_by_layer[layer]
                layer_bounds.append(
                    _outward_nonnegative_product_sum(
                        (cotangent_upper, phi_upper, drive_absolute),
                        profile=profile,
                        name=f"{layer} action-drive endpoint envelope",
                    )
                )
            layer_bounds_tuple = tuple(layer_bounds)
            layer_bounds_by_endpoint.append(layer_bounds_tuple)
            selected_live_routes_by_endpoint.append(tuple(selected_live_counts))
            action_bounds.append(
                _outward_nonnegative_scalar_sum(
                    layer_bounds_tuple,
                    profile=profile,
                    name="cross-layer action-drive endpoint envelope",
                )
            )

    # Close the same concurrent-mutation race as the singleton route builder.
    if state_dict_digest(model) != layout.model_state_hash:
        raise RuntimeError(
            "AlexNet model state changed during action-drive envelope derivation"
        )
    if (
        _direct_hard_mask_hash(
            current_mask,
            name="AlexNet action-drive current mask",
        )
        != current_mask_hash
    ):
        raise RuntimeError(
            "AlexNet hard mask changed during action-drive envelope derivation"
        )

    action_bound_axis = tuple(action_bounds)
    if len(action_bound_axis) != len(ladder.actions):
        raise RuntimeError("action-drive endpoint bound axis did not close")
    route_sum_rows = route_sum.endpoint_absolute_upper_bounds_by_stratum
    if any(len(row) != len(action_bound_axis) for row in route_sum_rows):
        raise RuntimeError("route-sum endpoint bound axis changed after alignment")
    hybrid_rows = tuple(
        tuple(
            min(action_bound, route_bound)
            for action_bound, route_bound in zip(
                action_bound_axis, route_row, strict=True
            )
        )
        for route_row in route_sum_rows
    )
    if any(
        hybrid > action or hybrid > route
        for route_row, hybrid_row in zip(route_sum_rows, hybrid_rows, strict=True)
        for action, route, hybrid in zip(
            action_bound_axis, route_row, hybrid_row, strict=True
        )
    ):
        raise RuntimeError("hybrid endpoint envelope failed component domination")

    verifier_source_hash = _source_sha256()
    alignment_payload = {
        "ladder_hash": ladder_hash,
        "endpoint_action_hashes": ladder.action_hashes,
        "route_count": ladder.route_count,
        "route_axis_hash": ladder.route_axis_hash,
        "current_mask_hash": ladder.current_mask_hash,
        "stratum_names": route_sum.stratum_names,
        "route_sum_endpoint_certificate_hash": route_sum_hash,
        "route_envelope_certificate_hash": envelope_hash,
        "route_envelope_proof_artifact_hash": receipt_hash,
    }
    alignment_hash = content_hash(
        alignment_payload,
        prefix="partition-free-v1-action-drive-endpoint-alignment-",
    )
    proof_payload = {
        "schema": ALEXNET_ACTION_DRIVE_ENDPOINT_PROOF_SCHEMA,
        "algorithm_id": ALEXNET_ACTION_DRIVE_ENDPOINT_ALGORITHM_ID,
        "phi_bound_id": ALEXNET_ENVELOPE_PHI_BOUND_ID,
        "arithmetic_profile_hash": profile.profile_hash,
        "model_state_hash": layout.model_state_hash,
        "graph_semantics_hash": layout.graph_semantics_hash,
        "input_box_hash": layout.input_box_hash,
        "alignment_hash": alignment_hash,
        "alignment": alignment_payload,
        "action_drive_bounds": action_bound_axis,
        "layer_order": tuple(_AFFINE_PATH_BY_LAYER),
        "layer_bounds_by_endpoint": tuple(layer_bounds_by_endpoint),
        "selected_live_routes_by_endpoint": tuple(selected_live_routes_by_endpoint),
        "singleton_route_sum_bounds_by_stratum": route_sum_rows,
        "hybrid_endpoint_bounds_by_stratum": hybrid_rows,
        "signed_selected_weights_aggregate_before_absolute_value": True,
        "canonical_g_phi_c_not_finite_removal_delta_h": True,
        "within_destination_unit_cancellation_only": True,
        "layerwise_then_global_outward_sum": True,
        "verifier_source_hash": verifier_source_hash,
        "issued_before_certification_outcomes": True,
        "score_order_authority": False,
        "bce_mask_commit_authority": False,
    }
    proof_artifact_hash = content_hash(
        proof_payload,
        prefix=_ACTION_DRIVE_ENDPOINT_PROOF_PREFIX,
    )
    certificate = V1FrozenLadderEndpointEnvelopeCertificate(
        campaign_contract_hash=ladder.campaign_contract_hash,
        model_identity_hash=ladder.model_identity_hash,
        query_contract_hash=ladder.query_contract_hash,
        preprocessing_hash=ladder.preprocessing_hash,
        current_mask_hash=ladder.current_mask_hash,
        raw_logit_stream_source_hash=ladder.raw_logit_stream_source_hash,
        sampling_law_hash=ladder.sampling_law_hash,
        proposal_generation=ladder.proposal_generation,
        route_count=ladder.route_count,
        route_axis_hash=ladder.route_axis_hash,
        ladder_hash=ladder_hash,
        endpoint_action_hashes=ladder.action_hashes,
        stratum_names=route_sum.stratum_names,
        endpoint_absolute_upper_bounds_by_stratum=hybrid_rows,
        verifier_source_hash=verifier_source_hash,
        proof_artifact_hash=proof_artifact_hash,
        derivation_kind=ACTION_SPECIFIC_ENDPOINT_ENVELOPE_KIND,
    )
    certificate.assert_integrity()
    return certificate


def build_v1_alexnet_action_drive_endpoint_envelope_bundle(
    *,
    model: Any,
    route_axis: DeclaredRouteAxis,
    current_mask: HardMask,
    query: QuerySpec,
    input_box: V1RegisteredNormalizedInputBox,
    envelope_result: V1AlexNetCorrectedAttributionEnvelopeResult,
    ladder: V1FrozenCandidateLadder,
) -> V1CancellationAwareEndpointEnvelopeBundle:
    """Return the cancellation-aware envelope with its explicit fallback.

    The route-sum certificate is retained as a first-class artifact rather
    than merely used transiently inside the sharper interpreter.  Downstream
    response exactification can therefore prove both pointwise domination and
    the continued availability of a sound, nonempty-endpoint fallback before
    any fresh F/Q row is observed.
    """

    route_sum = build_v1_alexnet_route_sum_endpoint_envelope(
        envelope_result=envelope_result,
        ladder=ladder,
    )
    action_drive = build_v1_alexnet_action_drive_endpoint_envelope(
        model=model,
        route_axis=route_axis,
        current_mask=current_mask,
        query=query,
        input_box=input_box,
        envelope_result=envelope_result,
        ladder=ladder,
    )
    return bind_v1_cancellation_aware_endpoint_envelopes(
        ladder=ladder,
        action_drive_certificate=action_drive,
        route_sum_fallback_certificate=route_sum,
    )


__all__ = [
    "ALEXNET_ACTION_DRIVE_ENDPOINT_ALGORITHM_ID",
    "ALEXNET_ACTION_DRIVE_ENDPOINT_PROOF_SCHEMA",
    "ALEXNET_ENVELOPE_ALGORITHM_ID",
    "ALEXNET_ENVELOPE_ARITHMETIC_ID",
    "ALEXNET_ENVELOPE_PHI_BOUND_ID",
    "V1AlexNetAttributionEnvelopeDerivationReceipt",
    "V1AlexNetCorrectedAttributionEnvelopeResult",
    "V1AlexNetEnvelopeArithmeticProfile",
    "V1AlexNetEnvelopeLayerReceipt",
    "V1AlexNetEnvelopeLayoutAdmission",
    "V1RegisteredNormalizedInputBox",
    "admit_v1_alexnet_envelope_layout",
    "build_v1_alexnet_action_drive_endpoint_envelope",
    "build_v1_alexnet_action_drive_endpoint_envelope_bundle",
    "build_v1_alexnet_corrected_attribution_envelope",
    "build_v1_alexnet_route_sum_endpoint_envelope",
    "registered_normalized_input_box_from_preprocessing",
]
