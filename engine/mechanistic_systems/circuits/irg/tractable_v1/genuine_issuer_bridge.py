"""Join retained model/source/FQ owners to bounded functional K/O row execution.

The comprehensive production factory retains support-bound, sampling, alpha,
and decision ownership. This adapter supplies source-verified numerical rows;
it grants no population or mask-commit authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from io import BytesIO
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import hashlib
import math

import torch

from mechanistic_systems.contracts import StudySpec
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.circuits.irg import modern_v1_paired_bce_acquisition as old
from mechanistic_systems.circuits.irg.modern_v1_alexnet_fq_registration import (
    V1AlexNetFQAcquisitionRegistration,
    V1AlexNetFQOracleSamplingFrameReceipt,
)
from mechanistic_systems.circuits.irg.modern_v1_controller import (
    V1CandidateCalibrationFreeze,
    V1FrozenAffineCalibration,
)
from mechanistic_systems.circuits.irg.tractable_v1.alexnet_factors import (
    READOUT_PATH,
    ROUTE_LAYER_SPECS,
    MaskedAlexNet,
    raw_target_logits,
)
from mechanistic_systems.circuits.irg.tractable_v1.evaluation import AffineCalibration
from mechanistic_systems.circuits.irg.tractable_v1 import masked_alexnet_paired_bce as bridge
from mechanistic_systems.circuits.irg.tractable_v1.masked_alexnet_paired_bce import (
    ContextSource,
    ContextTensorBatch,
    MaskedAlexNetPairedBCEEpoch,
    MaskedAlexNetPairedBCERowProvider,
    per_context_bce_increment_rows,
)
from mechanistic_systems.circuits.irg.modern_v1_neutral_runtime import (
    ClassBalancedSamplingPopulation,
    DeclaredRouteAxis,
    build_class_balanced_sampling_population,
)
from mechanistic_systems.measurement.contexts import ContextAxis, ImagePreprocessingSpec
from mechanistic_systems.measurement.model_sources import ModelSourceReplica
from mechanistic_systems.measurement.models import ModelLoadReceipt, state_dict_digest


OLD_ALEXNET_ROUTE_LAYER_ALIASES = MappingProxyType({
    "features.stage1.conv": "features.0",
    "features.stage2.conv": "features.3",
    "features.stage3.conv": "features.6",
    "features.stage4.conv": "features.8",
    "features.stage5.conv": "features.10",
    "readout.hidden1": "classifier.1",
    "readout.hidden2": "classifier.4",
    **{name: name for name, _relu, _kind in ROUTE_LAYER_SPECS},
})


def _nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty source-owned identity")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _bounded_source_identity(image_source: Any) -> tuple[str, Mapping[str, Any]]:
    source_identity = _nonempty(getattr(image_source, "source_identity_hash", ""), "bounded image source identity")
    identity_fn = getattr(image_source, "identity", None)
    if not callable(identity_fn):
        raise TypeError("bounded image source requires an identity() owner")
    identity = identity_fn()
    if not isinstance(identity, Mapping) or identity.get("schema") != "tractable_v1_bounded_image_source_identity_v1":
        raise ValueError("bounded image source identity has an unsupported schema")
    if content_hash(identity, prefix="tractable-v1-image-source-") != source_identity:
        raise ValueError("bounded image source identity hash is not reproducible")
    return source_identity, identity


def _bounded_preprocessing_hash(identity: Mapping[str, Any]) -> str:
    preprocessing = identity.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ValueError("bounded image source identity lacks preprocessing")
    return content_hash(preprocessing, prefix="tractable-v1-bounded-image-source-preprocessing-")


def _context_axis_matches_bounded_preprocessing(axis: ContextAxis, bounded_identity: Mapping[str, Any]) -> None:
    bounded = bounded_identity.get("preprocessing")
    if not isinstance(bounded, Mapping):
        raise ValueError("bounded image source identity lacks preprocessing")
    spec = axis.preprocessing
    spec.validate()
    if (
        tuple(spec.resize_size) != (256,)
        or tuple(spec.center_crop_size) != (224, 224)
        or spec.interpolation != "bilinear"
        or not bool(spec.antialias)
        or not bool(spec.scale_uint8_to_unit)
        or tuple(float(v) for v in spec.mean) != (0.485, 0.456, 0.406)
        or tuple(float(v) for v in spec.std) != (0.229, 0.224, 0.225)
        or spec.output_dtype != "float32"
        or bounded.get("normalization") != "separate_fp32_div255_submean_divstd"
    ):
        raise ValueError("ContextAxis preprocessing is not compatible with bounded image source preprocessing")


def _bounded_source_paths(image_source: Any) -> Mapping[str, tuple[str, ...]]:
    paths = getattr(image_source, "_paths", None)
    if not isinstance(paths, Mapping):
        raise TypeError("bounded image source must expose its sealed path listing for ContextAxis joining")
    cleaned: dict[str, tuple[str, ...]] = {}
    for stratum in old.PAIRED_BCE_STRATA:
        raw = paths.get(stratum)
        if raw is None:
            raise ValueError("bounded image source path listing lacks paired-BCE stratum")
        values = tuple(str(value) for value in raw)
        if not values:
            raise ValueError("bounded image source path listing is empty")
        cleaned[stratum] = values
    return MappingProxyType(cleaned)


def _record_path(root: Path, relative_path: str) -> str:
    rel = Path(str(relative_path))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("ContextAxis record relative_path must be a safe relative path")
    return str((root / rel).resolve())


def _study_preprocessing_content_hash(study: StudySpec) -> str:
    preprocessing = dict(study.preprocessing)
    keys = tuple(str(key).strip().lower() for key in preprocessing)
    if len(keys) != len(set(keys)):
        raise ValueError("StudySpec preprocessing contains ambiguous normalization key collisions")
    if "content_hash" not in preprocessing:
        raise ValueError("StudySpec preprocessing must expose the retained preprocessing content_hash")
    if any(key in preprocessing for key in ("normalization_hash", "preprocessing_hash")):
        raise ValueError("StudySpec preprocessing must not carry competing normalization/preprocessing hashes")
    return _nonempty(preprocessing["content_hash"], "StudySpec preprocessing content_hash")


def _population_equals(a: ClassBalancedSamplingPopulation, b: ClassBalancedSamplingPopulation) -> bool:
    return (
        a.source_context_axis_hash == b.source_context_axis_hash
        and a.source_label_map_hash == b.source_label_map_hash
        and int(a.target_class_id) == int(b.target_class_id)
        and a.source_role == b.source_role
        and a.population_context_ids == b.population_context_ids
        and a.population_labels == b.population_labels
        and a.on_target_context_ids == b.on_target_context_ids
        and a.off_target_context_ids == b.off_target_context_ids
        and a.target_population_probabilities == b.target_population_probabilities
        and a.sampling_population_probabilities == b.sampling_population_probabilities
        and a.population_hash == b.population_hash
        and a.sampling_law_hash == b.sampling_law_hash
    )


def _require_population_matches_context_axis(
    population: ClassBalancedSamplingPopulation,
    *,
    context_axis: ContextAxis,
    target_class_id: int,
    source_role: str,
) -> ClassBalancedSamplingPopulation:
    if type(population) is not ClassBalancedSamplingPopulation:
        raise TypeError("sampling_population must be an exact ClassBalancedSamplingPopulation")
    expected = build_class_balanced_sampling_population(
        context_axis,
        target_class_id=target_class_id,
        source_role=source_role,
    )
    if not _population_equals(population, expected):
        raise ValueError("ClassBalancedSamplingPopulation is not the exact ContextAxis-derived owner")
    return expected


def _normalize_checkpoint_state_dict(loaded: object) -> Mapping[str, torch.Tensor]:
    if isinstance(loaded, Mapping) and "state_dict" in loaded and isinstance(loaded["state_dict"], Mapping):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, Mapping) or not loaded:
        raise ValueError("checkpoint does not contain a nonempty state_dict mapping")
    state: dict[str, torch.Tensor] = {}
    source_key_by_normalized: dict[str, str] = {}
    for key, value in loaded.items():
        raw_key = str(key)
        normalized_key = raw_key.removeprefix("module.")
        if normalized_key in state:
            raise ValueError(
                "checkpoint state_dict has colliding normalized keys: "
                f"{source_key_by_normalized[normalized_key]!r} and {raw_key!r}"
            )
        if not torch.is_tensor(value):
            raise ValueError(f"checkpoint state_dict entry {raw_key!r} is not a tensor")
        state[normalized_key] = value.detach().cpu().contiguous()
        source_key_by_normalized[normalized_key] = raw_key
    return MappingProxyType(state)


def _open_checkpoint_state_dict(path: Path, *, expected_checkpoint_hash: str) -> tuple[str, Mapping[str, torch.Tensor]]:
    payload = path.read_bytes()
    observed_hash = hashlib.sha256(payload).hexdigest()
    if observed_hash != expected_checkpoint_hash:
        raise ValueError("model checkpoint bytes differ from ModelSourceReplica/ModelLoadReceipt")
    try:
        loaded = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch without weights_only.
        loaded = torch.load(BytesIO(payload), map_location="cpu")
    return observed_hash, _normalize_checkpoint_state_dict(loaded)


def _state_tensor(state: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    try:
        value = state[key]
    except KeyError as exc:
        raise ValueError(f"checkpoint state_dict is missing {key}") from exc
    if not torch.is_tensor(value):
        raise ValueError(f"checkpoint state_dict entry {key} is not a tensor")
    return value


def _compare_realized_tensor(*, observed: torch.Tensor, expected: torch.Tensor, name: str, device: torch.device, dtype: torch.dtype) -> None:
    if observed.device != device or observed.dtype != dtype:
        raise ValueError(f"actual MaskedAlexNet tensor has unexpected device or dtype: {name}")
    realized = expected.detach().to(device=device, dtype=dtype).contiguous()
    actual = observed.detach().to(device=device, dtype=dtype).contiguous()
    if actual.shape != realized.shape or actual.dtype != realized.dtype or actual.device != realized.device or not bool(torch.equal(actual, realized)):
        raise ValueError(f"actual MaskedAlexNet tensor differs from checkpoint state_dict: {name}")


def verify_masked_alexnet_matches_checkpoint_state(
    *,
    net: MaskedAlexNet,
    checkpoint_state: Mapping[str, torch.Tensor],
    structural_fingerprint: str | None = None,
    model_leases: tuple[bridge.TensorLease, ...] | None = None,
) -> tuple[tuple[str, str, str, tuple[int, ...]], ...]:
    if type(net) is not MaskedAlexNet:
        raise TypeError("checkpoint verification requires an exact MaskedAlexNet")
    structural = bridge._structural_fingerprint(net) if structural_fingerprint is None else structural_fingerprint
    leases = bridge._model_leases(net) if model_leases is None else model_leases
    bridge._assert_model_snapshot(net, structural, leases)
    realized: list[tuple[str, str, str, tuple[int, ...]]] = []
    for index, (layer_name, _relu, _kind) in enumerate(ROUTE_LAYER_SPECS):
        for suffix, observed in (
            ("weight", net.weights[index]),
            ("bias", net._dense_biases[index]),
        ):
            key = f"{layer_name}.{suffix}"
            expected = _state_tensor(checkpoint_state, key)
            _compare_realized_tensor(
                observed=observed,
                expected=expected,
                name=key,
                device=net.device,
                dtype=net.dtype,
            )
            realized.append((key, str(net.device), str(net.dtype), tuple(int(v) for v in observed.shape)))
    for key, observed in (
        (f"{READOUT_PATH}.weight", net.readout_weight),
        (f"{READOUT_PATH}.bias", net._dense_readout_bias),
    ):
        expected = _state_tensor(checkpoint_state, key)
        _compare_realized_tensor(
            observed=observed,
            expected=expected,
            name=key,
            device=net.device,
            dtype=net.dtype,
        )
        realized.append((key, str(net.device), str(net.dtype), tuple(int(v) for v in observed.shape)))
    bridge._assert_model_snapshot(net, structural, leases)
    return tuple(realized)


def _route_kind_for_masked_layer(kind: str) -> tuple[str, str]:
    if kind == "conv":
        return "conv_kernel_weight", "c_order_target_source_taps_v1"
    if kind == "linear":
        return "linear_weight", "c_order_target_then_source"
    raise ValueError("unsupported MaskedAlexNet route layer kind")


def verify_declared_route_axis_matches_masked_alexnet(
    route_axis: DeclaredRouteAxis,
    *,
    net: MaskedAlexNet,
) -> tuple[tuple[str, str, str, tuple[int, ...], int], ...]:
    if type(route_axis) is not DeclaredRouteAxis:
        raise TypeError("old route-axis mapping requires an exact DeclaredRouteAxis")
    if type(net) is not MaskedAlexNet:
        raise TypeError("old route-axis mapping requires an exact MaskedAlexNet")
    route_axis.__post_init__()
    if len(route_axis.route_blocks) != len(net.axis.layers):
        raise ValueError("old DeclaredRouteAxis block count differs from MaskedAlexNet")
    mapped_layer_order = tuple(
        OLD_ALEXNET_ROUTE_LAYER_ALIASES.get(str(block.layer_id))
        for block in route_axis.route_blocks
    )
    expected_layer_order = tuple(name for name, _relu, _kind in ROUTE_LAYER_SPECS)
    if mapped_layer_order != expected_layer_order:
        raise ValueError("old DeclaredRouteAxis layer order differs from registered AlexNet route specs")
    summary: list[tuple[str, str, str, tuple[int, ...], int]] = []
    architecture_ids = {str(block.architecture_id) for block in route_axis.route_blocks}
    topology_hashes = {str(block.topology_hash) for block in route_axis.route_blocks}
    if len(architecture_ids) != 1 or len(topology_hashes) != 1:
        raise ValueError("old DeclaredRouteAxis blocks must share one architecture/topology owner")
    for block, layer in zip(route_axis.route_blocks, net.axis.layers, strict=True):
        expected_kind, expected_flattening = _route_kind_for_masked_layer(layer.kind)
        mapped_layer = OLD_ALEXNET_ROUTE_LAYER_ALIASES.get(str(block.layer_id))
        if (
            mapped_layer != layer.name
            or str(block.route_kind) != expected_kind
            or tuple(int(value) for value in block.shape) != tuple(int(value) for value in layer.weight_shape)
            or int(block.offset) != int(layer.route_offset)
            or str(block.flattening) != expected_flattening
            or int(block.stop) != int(layer.route_offset + layer.route_count)
        ):
            raise ValueError("old DeclaredRouteAxis physical layer mapping differs from MaskedAlexNet")
        summary.append((str(mapped_layer), str(block.route_kind), str(block.flattening), tuple(int(v) for v in block.shape), int(block.offset)))
    if route_axis.route_count != int(net.mask.numel()):
        raise ValueError("old DeclaredRouteAxis route count differs from MaskedAlexNet")
    return tuple(summary)


def torch_mask_from_hard_mask(mask: HardMask, *, net: MaskedAlexNet, route_axis: DeclaredRouteAxis) -> torch.Tensor:
    if type(mask) is not HardMask:
        raise TypeError("old mask mapping requires an exact HardMask")
    if type(net) is not MaskedAlexNet:
        raise TypeError("old mask mapping requires an exact MaskedAlexNet")
    verify_declared_route_axis_matches_masked_alexnet(route_axis, net=net)
    if mask.route_axis_hash != route_axis.route_axis_hash:
        raise ValueError("old HardMask route axis differs from the supplied old DeclaredRouteAxis")
    if mask.route_count != int(net.mask.numel()) or mask.route_count != route_axis.route_count:
        raise ValueError("old HardMask route count differs from the supplied old DeclaredRouteAxis")
    values = mask.to_bool_array().astype("float32")
    return torch.from_numpy(values).to(device=net.mask.device, dtype=net.mask.dtype)


def affine_from_frozen(
    frozen: V1FrozenAffineCalibration,
    *,
    expected_mask_hash: str,
    target_class: int,
) -> AffineCalibration:
    if type(frozen) is not V1FrozenAffineCalibration:
        raise TypeError("affine mapping requires an exact old frozen affine calibration")
    frozen.__post_init__()
    if frozen.mask_hash != _nonempty(expected_mask_hash, "expected_mask_hash"):
        raise ValueError("old affine calibration names another mask")
    if frozen.target_class_id != int(target_class):
        raise ValueError("old affine calibration names another target class")
    # The old owner carries the fit receipt and fit metadata. The tractable
    # runtime only needs the frozen coefficients, so loss/iterations are marked
    # unavailable in this coefficient adapter rather than reconstructed.
    return AffineCalibration(
        scale=float(frozen.scale),
        offset=float(frozen.offset),
        loss=0.0,
        iterations=0,
        positive_scale=True,
    )


@dataclass(frozen=True, slots=True)
class ActualRowIssuerJoin:
    """Explicit old-owner to actual-row-owner identity join.

    This join must be issued by the data/model/calibration owner. It is not a
    statistical certificate. The bridge verifies the cheap actual model content
    audit against `actual_model_content_audit_hash` so same-axis/different-weight
    accidental rebinding is caught at row-admission time.
    """

    model_identity_hash: str
    model_state_hash: str
    actual_model_content_audit_hash: str
    route_axis_hash: str
    source_context_axis_hash: str
    source_label_map_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    calibration_source_hash: str
    dense_calibration_query_hash: str
    candidate_calibration_query_hash: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class PairedBCEBridgeIdentities:
    """Caller-owned executable/source identities for the actual row path.

    These identities are admission coordinates, not statistical authority. They
    must be supplied by their owners and are intentionally not derived from a
    PRNG seed, a sorted context list, or a human-readable label.
    """

    row_provider_identity_hash: str
    paired_loss_evaluator_identity_hash: str
    context_source_identity_hash: str
    preprocessing_identity_hash: str
    bound_certificate_hash_by_stratum: Mapping[str, str]
    sampling_law_identity_hash: str
    sealed_order_slice_verifier_identity_hash: str
    candidate_attempt_alpha_axis_hash: str
    alpha_committed_before_attempt: float
    oracle_familywise_alpha: float
    attempt_alpha_allocation: float
    maximum_candidate_attempts: int

    def __post_init__(self) -> None:
        for name in (
            "row_provider_identity_hash",
            "paired_loss_evaluator_identity_hash",
            "context_source_identity_hash",
            "preprocessing_identity_hash",
            "sampling_law_identity_hash",
            "sealed_order_slice_verifier_identity_hash",
            "candidate_attempt_alpha_axis_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        bounds = dict(self.bound_certificate_hash_by_stratum)
        if set(bounds) != set(old.PAIRED_BCE_STRATA):
            raise ValueError("bound certificates must name exactly target and complement strata")
        for stratum in old.PAIRED_BCE_STRATA:
            bounds[stratum] = _nonempty(bounds[stratum], f"{stratum} bound_certificate_hash")
        object.__setattr__(self, "bound_certificate_hash_by_stratum", MappingProxyType(bounds))
        if type(self.maximum_candidate_attempts) is not int or self.maximum_candidate_attempts <= 0:
            raise ValueError("maximum_candidate_attempts must be a positive integer")
        object.__setattr__(self, "alpha_committed_before_attempt", _finite_nonnegative(self.alpha_committed_before_attempt, "alpha_committed_before_attempt"))
        object.__setattr__(self, "oracle_familywise_alpha", _finite_nonnegative(self.oracle_familywise_alpha, "oracle_familywise_alpha"))
        object.__setattr__(self, "attempt_alpha_allocation", _finite_nonnegative(self.attempt_alpha_allocation, "attempt_alpha_allocation"))



@dataclass(frozen=True, slots=True)
class OpenedBoundedImageSourceOwner:
    """Verified relation between retained ContextAxis coordinates and a bounded image source."""

    source_context_axis_hash: str
    source_label_map_hash: str
    preprocessing_hash: str
    bounded_image_source_identity_hash: str
    bounded_preprocessing_hash: str
    target_context_ids: tuple[str, ...]
    complement_context_ids: tuple[str, ...]
    target_listing_indices: tuple[int, ...]
    complement_listing_indices: tuple[int, ...]
    source_population_hash: str
    sampling_law_hash: str
    context_id_crosswalk_hash: str
    source_axis_binding_hash: str
    schema: str = "tractable_v1_opened_context_axis_bounded_image_source_owner_v2"
    preprocessing: ImagePreprocessingSpec | None = field(default=None, kw_only=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.preprocessing is not None:
            if type(self.preprocessing) is not ImagePreprocessingSpec:
                raise TypeError("source preprocessing must retain its exact specification")
            self.preprocessing.validate()
            if self.preprocessing.content_hash != self.preprocessing_hash:
                raise ValueError("retained preprocessing differs from verified source hash")
        for name in (
            "source_context_axis_hash",
            "source_label_map_hash",
            "preprocessing_hash",
            "bounded_image_source_identity_hash",
            "bounded_preprocessing_hash",
            "source_population_hash",
            "sampling_law_hash",
            "context_id_crosswalk_hash",
            "source_axis_binding_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        rows = []
        for stratum, raw, raw_indices in (
            (old.TARGET_STRATUM, self.target_context_ids, self.target_listing_indices),
            (old.COMPLEMENT_STRATUM, self.complement_context_ids, self.complement_listing_indices),
        ):
            contexts = tuple(str(value) for value in raw)
            if not contexts or len(contexts) != len(set(contexts)):
                raise ValueError("bounded source owner context orders must be nonempty and unique")
            indices = tuple(int(value) for value in raw_indices)
            if len(indices) != len(contexts) or any(value < 0 for value in indices):
                raise ValueError("bounded source owner listing indices must align with context orders")
            if len(indices) != len(set(indices)):
                raise ValueError("bounded source owner listing indices must be unique")
            object.__setattr__(self, f"{stratum}_context_ids", contexts)
            object.__setattr__(self, f"{stratum}_listing_indices", indices)
            rows.append((stratum, contexts, indices))
        crosswalk_hash = content_hash(
            {
                "schema": "tractable_v1_context_axis_to_bounded_listing_crosswalk_v1",
                "source_context_axis_hash": self.source_context_axis_hash,
                "bounded_image_source_identity_hash": self.bounded_image_source_identity_hash,
                "context_orders": tuple(rows),
            },
            prefix="tractable-v1-context-axis-bounded-listing-crosswalk-",
        )
        if crosswalk_hash != self.context_id_crosswalk_hash:
            raise ValueError("bounded source owner crosswalk hash is not reproducible")
        expected = content_hash(
            {
                "schema": self.schema,
                "source_context_axis_hash": self.source_context_axis_hash,
                "source_label_map_hash": self.source_label_map_hash,
                "preprocessing_hash": self.preprocessing_hash,
                "bounded_image_source_identity_hash": self.bounded_image_source_identity_hash,
                "bounded_preprocessing_hash": self.bounded_preprocessing_hash,
                "source_population_hash": self.source_population_hash,
                "sampling_law_hash": self.sampling_law_hash,
                "context_id_crosswalk_hash": self.context_id_crosswalk_hash,
                "context_orders": tuple(rows),
                "context_id_format": "retained_ContextAxis_context_id",
                "bounded_listing_coordinate_format": "target:N/complement:N_listing_index",
                "sampling_authority": False,
                "production_commit_authority": False,
            },
            prefix="tractable-v1-opened-bounded-source-owner-",
        )
        if expected != self.source_axis_binding_hash:
            raise ValueError("bounded source owner binding hash is not reproducible")

    @property
    def context_ids_by_stratum(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType({
            old.TARGET_STRATUM: self.target_context_ids,
            old.COMPLEMENT_STRATUM: self.complement_context_ids,
        })

    @property
    def listing_indices_by_stratum(self) -> Mapping[str, tuple[int, ...]]:
        return MappingProxyType({
            old.TARGET_STRATUM: self.target_listing_indices,
            old.COMPLEMENT_STRATUM: self.complement_listing_indices,
        })

    @property
    def context_id_sets_by_stratum(self) -> Mapping[str, frozenset[str]]:
        return MappingProxyType({
            old.TARGET_STRATUM: frozenset(self.target_context_ids),
            old.COMPLEMENT_STRATUM: frozenset(self.complement_context_ids),
        })


def open_bounded_image_source_owner(
    *,
    image_source: Any,
    context_axis: ContextAxis,
    image_root: str | Path,
    target_class_id: int,
    source_role: str = "optimization",
    sampling_population: ClassBalancedSamplingPopulation | None = None,
) -> OpenedBoundedImageSourceOwner:
    source_identity, identity = _bounded_source_identity(image_source)
    bounded_preprocess = _bounded_preprocessing_hash(identity)
    if type(context_axis) is not ContextAxis:
        raise TypeError("bounded source owner requires an exact retained ContextAxis")
    context_axis.validate()
    _context_axis_matches_bounded_preprocessing(context_axis, identity)
    expected_population = build_class_balanced_sampling_population(
        context_axis,
        target_class_id=target_class_id,
        source_role=source_role,
    )
    population = expected_population if sampling_population is None else _require_population_matches_context_axis(
        sampling_population,
        context_axis=context_axis,
        target_class_id=target_class_id,
        source_role=source_role,
    )
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths_by_stratum = _bounded_source_paths(image_source)
    validate = getattr(image_source, "_validate_ids", None)
    if not callable(validate):
        raise TypeError("bounded image source requires an ID validator")
    record_by_id = {record.context_id: record for record in context_axis.records}
    if len(record_by_id) != len(context_axis.records):
        raise ValueError("ContextAxis records must be uniquely addressable by context_id")
    rows = []
    cleaned: dict[str, tuple[str, ...]] = {}
    indices_by_stratum: dict[str, tuple[int, ...]] = {}
    for stratum, contexts in (
        (old.TARGET_STRATUM, population.on_target_context_ids),
        (old.COMPLEMENT_STRATUM, population.off_target_context_ids),
    ):
        if not contexts:
            raise ValueError("ContextAxis-derived paired-BCE stratum is empty")
        path_to_index = {str(Path(path).resolve()): index for index, path in enumerate(paths_by_stratum[stratum])}
        if len(path_to_index) != len(paths_by_stratum[stratum]):
            raise ValueError("bounded image source listing paths must be unique")
        indices: list[int] = []
        for context_id in contexts:
            record = record_by_id[context_id]
            expected_path = _record_path(root, record.relative_path)
            try:
                listing_index = path_to_index[expected_path]
            except KeyError as exc:
                raise ValueError("ContextAxis record path is absent from bounded source listing") from exc
            if stratum == old.TARGET_STRATUM and int(record.label) != int(target_class_id):
                raise ValueError("ContextAxis target stratum label differs from target_class_id")
            if stratum == old.COMPLEMENT_STRATUM and int(record.label) == int(target_class_id):
                raise ValueError("ContextAxis complement stratum contains the target label")
            indices.append(int(listing_index))
        frozen_indices = tuple(indices)
        validate(stratum, frozen_indices)
        if len(frozen_indices) != len(set(frozen_indices)):
            raise ValueError("ContextAxis to bounded listing crosswalk repeats a listing index")
        cleaned[stratum] = tuple(contexts)
        indices_by_stratum[stratum] = frozen_indices
        rows.append((stratum, tuple(contexts), frozen_indices))
    crosswalk_hash = content_hash(
        {
            "schema": "tractable_v1_context_axis_to_bounded_listing_crosswalk_v1",
            "source_context_axis_hash": context_axis.content_hash,
            "bounded_image_source_identity_hash": source_identity,
            "context_orders": tuple(rows),
        },
        prefix="tractable-v1-context-axis-bounded-listing-crosswalk-",
    )
    binding_hash = content_hash(
        {
            "schema": "tractable_v1_opened_context_axis_bounded_image_source_owner_v2",
            "source_context_axis_hash": context_axis.content_hash,
            "source_label_map_hash": context_axis.label_map_hash,
            "preprocessing_hash": context_axis.preprocessing.content_hash,
            "bounded_image_source_identity_hash": source_identity,
            "bounded_preprocessing_hash": bounded_preprocess,
            "source_population_hash": population.population_hash,
            "sampling_law_hash": population.sampling_law_hash,
            "context_id_crosswalk_hash": crosswalk_hash,
            "context_orders": tuple(rows),
            "context_id_format": "retained_ContextAxis_context_id",
            "bounded_listing_coordinate_format": "target:N/complement:N_listing_index",
            "sampling_authority": False,
            "production_commit_authority": False,
        },
        prefix="tractable-v1-opened-bounded-source-owner-",
    )
    return OpenedBoundedImageSourceOwner(
        preprocessing=context_axis.preprocessing,
        source_context_axis_hash=context_axis.content_hash,
        source_label_map_hash=context_axis.label_map_hash,
        preprocessing_hash=context_axis.preprocessing.content_hash,
        bounded_image_source_identity_hash=source_identity,
        bounded_preprocessing_hash=bounded_preprocess,
        target_context_ids=cleaned[old.TARGET_STRATUM],
        complement_context_ids=cleaned[old.COMPLEMENT_STRATUM],
        target_listing_indices=indices_by_stratum[old.TARGET_STRATUM],
        complement_listing_indices=indices_by_stratum[old.COMPLEMENT_STRATUM],
        source_population_hash=population.population_hash,
        sampling_law_hash=population.sampling_law_hash,
        context_id_crosswalk_hash=crosswalk_hash,
        source_axis_binding_hash=binding_hash,
    )


@dataclass(frozen=True, slots=True)
class SourceOwnedActualRowIssuance:
    """Receipt-like carrier for a verified old-domain to actual-row join."""

    join: ActualRowIssuerJoin
    model_checkpoint_binding_hash: str
    actual_source_binding_hash: str
    actual_model_content_audit_hash: str
    old_route_axis_hash: str
    old_route_axis_content_hash: str
    route_axis_mapping_hash: str
    model_structural_fingerprint: str
    model_leases: tuple[bridge.TensorLease, ...]
    issuance_hash: str

    def __post_init__(self) -> None:
        if type(self.join) is not ActualRowIssuerJoin:
            raise TypeError("source-owned issuance requires an ActualRowIssuerJoin")
        leases = tuple(self.model_leases)
        if not leases or any(type(value) is not bridge.TensorLease for value in leases):
            raise TypeError("source-owned issuance requires frozen MaskedAlexNet model leases")
        object.__setattr__(self, "model_leases", leases)
        for name in (
            "model_checkpoint_binding_hash",
            "actual_source_binding_hash",
            "actual_model_content_audit_hash",
            "old_route_axis_hash",
            "old_route_axis_content_hash",
            "route_axis_mapping_hash",
            "model_structural_fingerprint",
            "issuance_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class BoundedImageContextSource:
    """Request-addressed adapter from `BoundedImageSource` batches to row batches.

    The adapter verifies the bounded source's existing receipt for each exact
    requested ID list. It does not create sampling authority.
    """

    image_source: Any
    source_owner: OpenedBoundedImageSourceOwner
    max_batch_rows: int = 64
    source_identity_hash: str = field(init=False)
    source_context_axis_hash: str = field(init=False)
    source_label_map_hash: str = field(init=False)
    preprocessing_hash: str = field(init=False)
    actual_source_binding_hash: str = field(init=False)
    _context_set_by_stratum: Mapping[str, frozenset[str]] = field(init=False, repr=False)
    _listing_index_by_context_id: Mapping[str, Mapping[str, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.source_owner) is not OpenedBoundedImageSourceOwner:
            raise TypeError("bounded image context source requires an opened source owner")
        if type(self.max_batch_rows) is not int or self.max_batch_rows <= 0:
            raise ValueError("max_batch_rows must be a positive integer")
        if not callable(getattr(self.image_source, "iter_indices", None)):
            raise TypeError("bounded image context source requires an image_source.iter_indices owner")
        source_identity, identity = _bounded_source_identity(self.image_source)
        if identity.get("sampling_authority") is not False or identity.get("production_authority") is not False:
            raise ValueError("bounded image source identity must not claim sampling or production authority")
        if source_identity != self.source_owner.bounded_image_source_identity_hash:
            raise ValueError("bounded image source identity differs from opened source owner")
        if _bounded_preprocessing_hash(identity) != self.source_owner.bounded_preprocessing_hash:
            raise ValueError("bounded image source preprocessing differs from opened source owner")
        binding_hash = content_hash(
            {
                "schema": "tractable_v1_actual_row_bounded_image_context_source_binding_v2",
                "opened_source_owner_hash": self.source_owner.source_axis_binding_hash,
                "bounded_image_source_identity_hash": source_identity,
                "source_context_axis_hash": self.source_owner.source_context_axis_hash,
                "source_label_map_hash": self.source_owner.source_label_map_hash,
                "preprocessing_hash": self.source_owner.preprocessing_hash,
                "max_batch_rows": self.max_batch_rows,
                "unique_request_context_ids_required": True,
                "request_context_id_format": "retained_ContextAxis_context_id",
                "bounded_listing_coordinate_format": "target:N/complement:N_listing_index",
                "sampling_authority": False,
                "production_commit_authority": False,
            },
            prefix="tractable-v1-actual-row-source-binding-",
        )
        context_sets = {
            stratum: frozenset(self.source_owner.context_ids_by_stratum[stratum])
            for stratum in old.PAIRED_BCE_STRATA
        }
        index_maps = {
            stratum: MappingProxyType(dict(zip(
                self.source_owner.context_ids_by_stratum[stratum],
                self.source_owner.listing_indices_by_stratum[stratum],
                strict=True,
            )))
            for stratum in old.PAIRED_BCE_STRATA
        }
        object.__setattr__(self, "source_identity_hash", source_identity)
        object.__setattr__(self, "source_context_axis_hash", self.source_owner.source_context_axis_hash)
        object.__setattr__(self, "source_label_map_hash", self.source_owner.source_label_map_hash)
        object.__setattr__(self, "preprocessing_hash", self.source_owner.preprocessing_hash)
        object.__setattr__(self, "actual_source_binding_hash", binding_hash)
        object.__setattr__(self, "_context_set_by_stratum", MappingProxyType(context_sets))
        object.__setattr__(self, "_listing_index_by_context_id", MappingProxyType(index_maps))

    def __call__(self, stratum: str, context_ids: tuple[str, ...]) -> ContextTensorBatch:
        if stratum not in old.PAIRED_BCE_STRATA:
            raise ValueError("bounded image request names an unknown paired-BCE stratum")
        requested = tuple(str(value) for value in context_ids)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("bounded image request requires nonempty unique context IDs")
        if len(requested) > self.max_batch_rows:
            raise ValueError("bounded image request exceeds max_batch_rows; reduce the old physical request size")
        owner_set = self._context_set_by_stratum[stratum]
        if any(value not in owner_set for value in requested):
            raise ValueError("bounded image request contains a context outside the opened source owner")
        index_map = self._listing_index_by_context_id[stratum]
        indices = tuple(int(index_map[value]) for value in requested)
        expected_bounded_ids = tuple(f"{stratum}:{index}" for index in indices)
        iterator = self.image_source.iter_indices(stratum, indices, batch_size=len(indices))
        batches = []
        try:
            for batch in iterator:
                batches.append(batch)
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        if not batches:
            raise ValueError("bounded image source returned no batch for a nonempty request")
        if len(batches) != 1:
            raise ValueError("bounded image source split a request larger than max_batch_rows")
        ordered: list[str] = []
        images: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        receipts: list[str] = []
        for batch in batches:
            if getattr(batch, "source_identity_hash", None) != self.source_identity_hash:
                raise ValueError("bounded image source batch identity changed")
            payload = getattr(batch, "source_batch_receipt_payload", None)
            if not isinstance(payload, Mapping):
                raise ValueError("bounded image source batch lacks a receipt payload")
            expected_hash = content_hash(payload, prefix="tractable-v1-image-batch-")
            if expected_hash != getattr(batch, "source_batch_receipt_hash", None):
                raise ValueError("bounded image source batch receipt is not reproducible")
            ids = tuple(str(value) for value in getattr(batch, "ordered_context_ids", ()))
            if not ids:
                raise ValueError("bounded image source batch lacks ordered context IDs")
            ordered.extend(ids)
            images.append(torch.as_tensor(batch.images))
            labels.append(torch.as_tensor(batch.labels, dtype=torch.float64).reshape(-1))
            receipts.append(str(batch.source_batch_receipt_hash))
        if tuple(ordered) != expected_bounded_ids:
            raise ValueError("bounded image source returned a different ordered bounded listing axis")
        image_tensor = torch.cat(images, dim=0)
        label_tensor = torch.cat(labels, dim=0)
        expected_label = 1.0 if stratum == old.TARGET_STRATUM else 0.0
        if not bool((label_tensor == expected_label).all()):
            raise ValueError("bounded image source labels differ from requested stratum")
        request_receipt = content_hash(
            {
                "schema": "tractable_v1_actual_row_context_tensor_batch_from_bounded_image_source_v1",
                "actual_source_binding_hash": self.actual_source_binding_hash,
                "source_identity_hash": self.source_identity_hash,
                "stratum": stratum,
                "ordered_context_ids": requested,
                "ordered_bounded_listing_ids": expected_bounded_ids,
                "source_batch_receipt_hashes": tuple(receipts),
                "sampling_authority": False,
                "production_commit_authority": False,
            },
            prefix="tractable-v1-actual-row-context-batch-",
        )
        return ContextTensorBatch(
            context_ids=requested,
            images=image_tensor,
            labels=label_tensor,
            source_batch_receipt_hash=request_receipt,
        )


_EXECUTION_SOURCE_OPEN_TOKEN = object()


@dataclass(frozen=True, slots=True)
class OpenedActualExecutionSources:
    """Live checkpoint/source join, available before action calibration exists.

    The owner binds weights, original dense biases, current circuit biases,
    route interpretation and source identities. It deliberately does not bind
    an action mask or authorize a draw. A caller must separately freeze the
    action and obtain the appropriate D/C/K/O sampling admission. Tensor leases
    are process-local guards; this is not a portable restart certificate.
    """

    fq_registration: V1AlexNetFQAcquisitionRegistration
    net: MaskedAlexNet = field(repr=False, compare=False)
    context_source: BoundedImageContextSource = field(repr=False, compare=False)
    route_axis: DeclaredRouteAxis
    model_checkpoint_binding_hash: str
    actual_source_binding_hash: str
    actual_model_content_audit_hash: str
    route_axis_mapping_hash: str
    model_structural_fingerprint: str
    model_leases: tuple[bridge.TensorLease, ...] = field(repr=False)
    opening_hash: str
    _issuer_token: object = field(repr=False, compare=False)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "tractable_v1_opened_actual_execution_sources_v1",
            "fq_registration_hash": self.fq_registration.registration_hash,
            "model_checkpoint_binding_hash": self.model_checkpoint_binding_hash,
            "actual_source_binding_hash": self.actual_source_binding_hash,
            "actual_model_content_audit_hash": self.actual_model_content_audit_hash,
            "old_route_axis_hash": self.route_axis.route_axis_hash,
            "old_route_axis_content_hash": self.route_axis.content_hash,
            "route_axis_mapping_hash": self.route_axis_mapping_hash,
            "model_structural_fingerprint": self.model_structural_fingerprint,
            "source_identity_hash": self.context_source.source_identity_hash,
            "sampling_authority": False,
            "physical_c_authority": False,
            "production_commit_authority": False,
        }

    def __post_init__(self) -> None:
        if self._issuer_token is not _EXECUTION_SOURCE_OPEN_TOKEN:
            raise ValueError("execution sources must be opened by the checkpoint/source verifier")
        if (type(self.fq_registration) is not V1AlexNetFQAcquisitionRegistration
                or type(self.net) is not MaskedAlexNet
                or type(self.context_source) is not BoundedImageContextSource
                or type(self.route_axis) is not DeclaredRouteAxis):
            raise TypeError("execution sources require exact retained source owners")
        self.assert_live()

    def assert_live(self) -> None:
        """Refuse stale executable/source bindings without opening image outcomes."""
        if self._issuer_token is not _EXECUTION_SOURCE_OPEN_TOKEN:
            raise ValueError("execution sources lack verifier issuance")
        self.fq_registration.assert_integrity()
        self.route_axis.__post_init__()
        if content_hash(self._payload(), prefix="tractable-v1-opened-execution-sources-") != self.opening_hash:
            raise ValueError("execution source opening identity changed")
        if self.context_source.actual_source_binding_hash != self.actual_source_binding_hash:
            raise ValueError("execution source adapter changed after opening")
        source_identity, _identity = _bounded_source_identity(self.context_source.image_source)
        if source_identity != self.context_source.source_identity_hash:
            raise ValueError("bounded image source changed after execution source opening")
        bridge._assert_model_snapshot(self.net, self.model_structural_fingerprint, self.model_leases)


def open_actual_execution_sources(
    *,
    fq_registration: V1AlexNetFQAcquisitionRegistration,
    net: MaskedAlexNet,
    model_load_receipt: ModelLoadReceipt,
    model_source_replica: ModelSourceReplica,
    context_source: BoundedImageContextSource,
    route_axis: DeclaredRouteAxis,
    query_contract_hash: str,
    study: StudySpec | None = None,
    model_architecture=None,
) -> OpenedActualExecutionSources:
    """Verify actual checkpoint and source owners before any calibration join.

    This performs the same checkpoint-byte, state-dict, route and live tensor
    verification used by the retained paired-BCE issuer. No image is decoded
    and no fitted calibration is needed. It does not issue an action, physical
    certificate, sampling request, or commitment permission.
    """
    if type(fq_registration) is not V1AlexNetFQAcquisitionRegistration:
        raise TypeError("actual-row issuance requires an exact old FQ registration")
    if type(net) is not MaskedAlexNet:
        raise TypeError("actual-row issuance requires an exact MaskedAlexNet")
    if type(model_load_receipt) is not ModelLoadReceipt:
        raise TypeError("actual-row issuance requires an exact ModelLoadReceipt")
    if type(model_source_replica) is not ModelSourceReplica:
        raise TypeError("actual-row issuance requires an exact ModelSourceReplica")
    if type(context_source) is not BoundedImageContextSource:
        raise TypeError("actual-row issuance requires the bounded image context-source adapter")
    if type(route_axis) is not DeclaredRouteAxis:
        raise TypeError("actual-row issuance requires the exact old DeclaredRouteAxis")
    if study is not None and type(study) is not StudySpec:
        raise TypeError("actual-row issuance study must be an exact StudySpec when supplied")
    structural = bridge._structural_fingerprint(net)
    leases = bridge._model_leases(net)
    bridge._assert_model_snapshot(net, structural, leases)
    fq_registration.assert_integrity()
    model_load_receipt.validate()
    model_source_replica.validate(claim_bearing=True)
    if model_source_replica.source_kind != "registered_model":
        raise ValueError("actual-row issuance currently supports only registered-model replicas")
    checkpoint_path = Path(_nonempty(model_source_replica.checkpoint_path, "model checkpoint path"))
    if not checkpoint_path.is_absolute():
        raise ValueError("model checkpoint path must be absolute")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"model checkpoint is unavailable: {checkpoint_path}")
    checkpoint_hash = _sha256(model_source_replica.checkpoint_file_hash, "model source replica checkpoint hash")
    if model_load_receipt.checkpoint_hash != checkpoint_hash:
        raise ValueError("model checkpoint bytes differ from ModelSourceReplica/ModelLoadReceipt")
    observed_checkpoint_hash, checkpoint_state = _open_checkpoint_state_dict(
        checkpoint_path,
        expected_checkpoint_hash=checkpoint_hash,
    )
    if study is not None:
        study.validate().require_valid()
        if (
            study.model_id != model_load_receipt.model_id
            or study.checkpoint_hash != model_load_receipt.checkpoint_hash
            or study.model_definition_hash != model_load_receipt.model_definition_hash
            or study.label_map_hash != context_source.source_label_map_hash
            or _study_preprocessing_content_hash(study) != context_source.preprocessing_hash
        ):
            raise ValueError("StudySpec differs from ModelLoadReceipt or bounded source registration")
    route_mapping = verify_declared_route_axis_matches_masked_alexnet(route_axis, net=net)
    architecture_projection = None
    if model_architecture is not None:
        from mechanistic_systems.core.architectures import BoundTorchArchitecture
        if type(model_architecture) is not BoundTorchArchitecture:
            raise TypeError('model architecture projection requires an exact BoundTorchArchitecture')
        model_architecture.validate()
        if (model_architecture.topology_hash != model_load_receipt.model_definition_hash
                or model_architecture.route_schema_hash != model_load_receipt.route_schema_hash
                or model_architecture.circuit_route_schema_hash != route_axis.route_axis_hash
                or model_architecture.circuit_route_blocks != route_axis.route_blocks):
            raise ValueError('registered full-model architecture differs from its prunable route projection')
        architecture_projection = dict(
            full_route_schema_hash=model_architecture.route_schema_hash,
            prunable_route_schema_hash=model_architecture.circuit_route_schema_hash,
            model_definition_hash=model_architecture.topology_hash,
            fixed_layer_ids=model_architecture.fixed_measurement_layer_ids,
        )
    elif model_load_receipt.route_schema_hash != route_axis.route_axis_hash:
        raise ValueError("ModelLoadReceipt route_schema_hash differs from supplied old DeclaredRouteAxis")
    bridge._assert_model_snapshot(net, structural, leases)
    if state_dict_digest(dict(checkpoint_state)) != model_load_receipt.state_dict_hash:
        raise ValueError("checkpoint state_dict digest differs from ModelLoadReceipt")
    realized_tensor_map = verify_masked_alexnet_matches_checkpoint_state(
        net=net,
        checkpoint_state=checkpoint_state,
        structural_fingerprint=structural,
        model_leases=leases,
    )
    bridge._assert_model_snapshot(net, structural, leases)
    if model_load_receipt.state_dict_hash != fq_registration.expected_model_state_hash:
        raise ValueError("ModelLoadReceipt state_dict_hash differs from FQ expected model state")
    if query_contract_hash != fq_registration.query_contract_hash:
        raise ValueError("query contract differs from FQ registration")
    if (
        context_source.source_context_axis_hash != fq_registration.source_context_axis_hash
        or context_source.source_label_map_hash != fq_registration.source_label_map_hash
        or context_source.preprocessing_hash != fq_registration.preprocessing_hash
    ):
        raise ValueError("bounded image source adapter differs from FQ source/preprocess registration")
    actual_model_content = bridge._model_content_digest_once(net)
    bridge._assert_model_snapshot(net, structural, leases)
    model_checkpoint_binding_hash = content_hash(
        {
            "schema": "tractable_v1_actual_row_model_checkpoint_join_v1",
            "study_hash": None if study is None else study.content_hash,
            "model_load_receipt_hash": model_load_receipt.content_hash,
            "model_id": model_load_receipt.model_id,
            "model_identity_hash": fq_registration.expected_model_identity_hash,
            "model_state_hash": model_load_receipt.state_dict_hash,
            "checkpoint_hash": checkpoint_hash,
            "observed_checkpoint_hash": observed_checkpoint_hash,
            "replica_binding_hash": model_source_replica.binding_hash,
            "old_route_axis_hash": route_axis.route_axis_hash,
            "old_route_axis_content_hash": route_axis.content_hash,
            "model_load_route_schema_hash": model_load_receipt.route_schema_hash,
            "actual_model_content_audit_hash": actual_model_content,
            "realized_tensor_map": realized_tensor_map,
            "route_mapping": route_mapping,
            "dtype_realization": str(net.dtype),
            "device_realization": str(net.device),
            "old_state_hash_not_redefined": True,
            **({} if architecture_projection is None else
               {"registered_architecture_projection": architecture_projection}),
        },
        prefix="tractable-v1-actual-row-model-join-",
    )
    fields = dict(
        fq_registration=fq_registration,
        net=net,
        context_source=context_source,
        route_axis=route_axis,
        model_checkpoint_binding_hash=model_checkpoint_binding_hash,
        actual_source_binding_hash=context_source.actual_source_binding_hash,
        actual_model_content_audit_hash=actual_model_content,
        route_axis_mapping_hash=content_hash(route_mapping, prefix="tractable-v1-old-route-axis-mapping-"),
        model_structural_fingerprint=structural,
        model_leases=leases,
    )
    payload = {
        "schema": "tractable_v1_opened_actual_execution_sources_v1",
        "fq_registration_hash": fq_registration.registration_hash,
        "model_checkpoint_binding_hash": model_checkpoint_binding_hash,
        "actual_source_binding_hash": context_source.actual_source_binding_hash,
        "actual_model_content_audit_hash": actual_model_content,
        "old_route_axis_hash": route_axis.route_axis_hash,
        "old_route_axis_content_hash": route_axis.content_hash,
        "route_axis_mapping_hash": fields["route_axis_mapping_hash"],
        "model_structural_fingerprint": structural,
        "source_identity_hash": context_source.source_identity_hash,
        "sampling_authority": False,
        "physical_c_authority": False,
        "production_commit_authority": False,
    }
    return OpenedActualExecutionSources(
        **fields,
        opening_hash=content_hash(payload, prefix="tractable-v1-opened-execution-sources-"),
        _issuer_token=_EXECUTION_SOURCE_OPEN_TOKEN,
    )


def issue_actual_row_join_from_opened_sources(
    *,
    fq_registration: V1AlexNetFQAcquisitionRegistration,
    calibration_freeze: V1CandidateCalibrationFreeze,
    net: MaskedAlexNet,
    model_load_receipt: ModelLoadReceipt,
    model_source_replica: ModelSourceReplica,
    context_source: BoundedImageContextSource,
    route_axis: DeclaredRouteAxis,
    query_contract_hash: str,
    study: StudySpec | None = None,
) -> SourceOwnedActualRowIssuance:
    """Construct the source-owned join from existing old model/source owners.

    The old domain remains `ModelLoadReceipt.state_dict_hash` and the F/Q
    registration's source/query/preprocess fields. The tractable model content
    audit is checked as a live executable-model admission coordinate; it is not
    substituted for the old state-dict hash.
    """

    if type(calibration_freeze) is not V1CandidateCalibrationFreeze:
        raise TypeError("actual-row issuance requires an exact candidate calibration freeze")
    calibration_freeze.__post_init__()
    sources = open_actual_execution_sources(
        fq_registration=fq_registration,
        net=net,
        model_load_receipt=model_load_receipt,
        model_source_replica=model_source_replica,
        context_source=context_source,
        route_axis=route_axis,
        query_contract_hash=query_contract_hash,
        study=study,
    )
    structural = sources.model_structural_fingerprint
    leases = sources.model_leases
    actual_model_content = sources.actual_model_content_audit_hash
    model_checkpoint_binding_hash = sources.model_checkpoint_binding_hash
    freeze = calibration_freeze
    dense_cal = freeze.dense_calibration
    cand_cal = freeze.candidate_calibration
    if (
        dense_cal.model_state_hash != model_load_receipt.state_dict_hash
        or cand_cal.model_state_hash != model_load_receipt.state_dict_hash
        or dense_cal.calibration_context_axis_hash != fq_registration.source_context_axis_hash
        or cand_cal.calibration_context_axis_hash != fq_registration.source_context_axis_hash
        or dense_cal.source_label_map_hash != fq_registration.source_label_map_hash
        or cand_cal.source_label_map_hash != fq_registration.source_label_map_hash
        or dense_cal.query_hash != query_contract_hash
        or cand_cal.query_hash != query_contract_hash
    ):
        raise ValueError("frozen affine calibrations differ from opened FQ/model/source/query owners")
    join = ActualRowIssuerJoin(
        model_identity_hash=fq_registration.expected_model_identity_hash,
        model_state_hash=model_load_receipt.state_dict_hash,
        actual_model_content_audit_hash=actual_model_content,
        route_axis_hash=route_axis.route_axis_hash,
        source_context_axis_hash=fq_registration.source_context_axis_hash,
        source_label_map_hash=fq_registration.source_label_map_hash,
        query_contract_hash=fq_registration.query_contract_hash,
        preprocessing_hash=fq_registration.preprocessing_hash,
        calibration_source_hash=dense_cal.calibration_source_hash,
        dense_calibration_query_hash=dense_cal.query_hash,
        candidate_calibration_query_hash=cand_cal.query_hash,
    )
    issuance_hash = content_hash(
        {
            "schema": "tractable_v1_actual_row_issuer_join_receipt_v1",
            "fq_registration_hash": fq_registration.registration_hash,
            "candidate_calibration_freeze_hash": freeze.freeze_hash,
            "model_checkpoint_binding_hash": model_checkpoint_binding_hash,
            "actual_source_binding_hash": context_source.actual_source_binding_hash,
            "actual_model_content_audit_hash": actual_model_content,
            "old_route_axis_hash": route_axis.route_axis_hash,
            "old_route_axis_content_hash": route_axis.content_hash,
            "route_axis_mapping_hash": sources.route_axis_mapping_hash,
            "model_structural_fingerprint": structural,
            "join": {name: getattr(join, name) for name in join.__dataclass_fields__},
            "sampling_authority": False,
            "production_commit_authority": False,
        },
        prefix="tractable-v1-actual-row-issuer-join-",
    )
    sources.assert_live()
    return SourceOwnedActualRowIssuance(
        join=join,
        model_checkpoint_binding_hash=model_checkpoint_binding_hash,
        actual_source_binding_hash=context_source.actual_source_binding_hash,
        actual_model_content_audit_hash=actual_model_content,
        old_route_axis_hash=route_axis.route_axis_hash,
        old_route_axis_content_hash=route_axis.content_hash,
        route_axis_mapping_hash=sources.route_axis_mapping_hash,
        model_structural_fingerprint=structural,
        model_leases=leases,
        issuance_hash=issuance_hash,
    )


@dataclass(frozen=True, slots=True)
class OldFQToTractablePairedBCEBridge:
    """Bind old FQ/mask/calibration owners to the actual MaskedAlexNet row law."""

    fq_registration: V1AlexNetFQAcquisitionRegistration
    net: MaskedAlexNet
    dense_mask: HardMask
    candidate_mask: HardMask
    route_axis: DeclaredRouteAxis
    calibration_freeze: V1CandidateCalibrationFreeze
    oracle_sampling_frame: V1AlexNetFQOracleSamplingFrameReceipt
    target_class: int
    identities: PairedBCEBridgeIdentities
    issuer_join: ActualRowIssuerJoin
    campaign_contract_hash: str
    registration_id: str
    source_issuance: SourceOwnedActualRowIssuance | None = None
    diagnostic_without_source_issuance: bool = False
    _structural_fingerprint: str = field(init=False, repr=False, compare=False)
    _model_leases: tuple[bridge.TensorLease, ...] = field(init=False, repr=False, compare=False)
    _candidate_mask_tensor: torch.Tensor = field(init=False, repr=False, compare=False)
    _dense_mask_tensor: torch.Tensor = field(init=False, repr=False, compare=False)
    _candidate_mask_lease: bridge.TensorLease = field(init=False, repr=False, compare=False)
    _dense_mask_lease: bridge.TensorLease = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.fq_registration) is not V1AlexNetFQAcquisitionRegistration:
            raise TypeError("bridge requires the exact old FQ acquisition registration")
        if type(self.net) is not MaskedAlexNet:
            raise TypeError("bridge requires an exact MaskedAlexNet")
        if type(self.route_axis) is not DeclaredRouteAxis:
            raise TypeError("bridge requires the exact old DeclaredRouteAxis")
        if type(self.calibration_freeze) is not V1CandidateCalibrationFreeze:
            raise TypeError("bridge requires an exact old candidate calibration freeze")
        if type(self.oracle_sampling_frame) is not V1AlexNetFQOracleSamplingFrameReceipt:
            raise TypeError("bridge requires an exact old FQ oracle sampling frame")
        if type(self.identities) is not PairedBCEBridgeIdentities:
            raise TypeError("bridge requires explicit source-owned identities")
        if type(self.issuer_join) is not ActualRowIssuerJoin:
            raise TypeError("bridge requires an explicit actual-row issuer join")
        if self.source_issuance is None:
            if self.diagnostic_without_source_issuance is not True:
                raise ValueError("claim-bearing bridge construction requires SourceOwnedActualRowIssuance")
        else:
            if type(self.source_issuance) is not SourceOwnedActualRowIssuance:
                raise TypeError("source_issuance must be a SourceOwnedActualRowIssuance")
            if self.source_issuance.join != self.issuer_join:
                raise ValueError("bridge issuer_join differs from source-owned issuance")
            if (
                self.source_issuance.old_route_axis_hash != self.route_axis.route_axis_hash
                or self.source_issuance.old_route_axis_content_hash != self.route_axis.content_hash
            ):
                raise ValueError("bridge route_axis differs from source-owned issuance")
        self.fq_registration.assert_integrity()
        self.calibration_freeze.__post_init__()
        self.oracle_sampling_frame.assert_integrity()
        object.__setattr__(self, "campaign_contract_hash", _nonempty(self.campaign_contract_hash, "campaign_contract_hash"))
        object.__setattr__(self, "registration_id", _nonempty(self.registration_id, "registration_id"))
        if type(self.target_class) is not int or self.target_class < 0:
            raise ValueError("target_class must be a nonnegative integer")
        if self.campaign_contract_hash != self.fq_registration.campaign_contract_hash:
            raise ValueError("bridge campaign hash differs from old FQ registration")
        if self.oracle_sampling_frame.registration_hash != self.fq_registration.registration_hash:
            raise ValueError("old FQ oracle frame belongs to another FQ registration")
        join = self.issuer_join
        fq = self.fq_registration
        if (
            join.model_identity_hash != fq.expected_model_identity_hash
            or join.model_state_hash != fq.expected_model_state_hash
            or join.route_axis_hash != self.route_axis.route_axis_hash
            or join.source_context_axis_hash != fq.source_context_axis_hash
            or join.source_label_map_hash != fq.source_label_map_hash
            or join.query_contract_hash != fq.query_contract_hash
            or join.preprocessing_hash != fq.preprocessing_hash
            or self.identities.preprocessing_identity_hash != fq.preprocessing_hash
        ):
            raise ValueError("actual-row issuer join differs from old FQ model/source/query/preprocess registration")
        if self.source_issuance is None:
            structural_fingerprint = bridge._structural_fingerprint(self.net)
            model_leases = bridge._model_leases(self.net)
        else:
            structural_fingerprint = self.source_issuance.model_structural_fingerprint
            model_leases = self.source_issuance.model_leases
        bridge._assert_model_snapshot(self.net, structural_fingerprint, model_leases)
        actual_content_hash = bridge._model_content_digest_once(self.net)
        bridge._assert_model_snapshot(self.net, structural_fingerprint, model_leases)
        if actual_content_hash != join.actual_model_content_audit_hash:
            raise ValueError("actual MaskedAlexNet content differs from issuer join")
        route_mapping = verify_declared_route_axis_matches_masked_alexnet(self.route_axis, net=self.net)
        if self.source_issuance is not None and self.source_issuance.route_axis_mapping_hash != content_hash(route_mapping, prefix="tractable-v1-old-route-axis-mapping-"):
            raise ValueError("bridge route-axis mapping differs from source-owned issuance")
        dense_tensor = torch_mask_from_hard_mask(self.dense_mask, net=self.net, route_axis=self.route_axis)
        candidate_tensor = torch_mask_from_hard_mask(self.candidate_mask, net=self.net, route_axis=self.route_axis)
        if not bool((dense_tensor == 1).all()):
            raise ValueError("dense HardMask must be all-live for dense reference binding")
        if not bool((candidate_tensor <= dense_tensor).all()):
            raise ValueError("candidate HardMask must be a removal-only subset of dense")
        candidate_lease = bridge._tensor_lease(candidate_tensor, name="candidate_mask")
        dense_lease = bridge._tensor_lease(dense_tensor, name="dense_mask")
        bridge._assert_model_snapshot(self.net, structural_fingerprint, model_leases)
        bridge._check_tensor_lease(candidate_tensor, candidate_lease)
        bridge._check_tensor_lease(dense_tensor, dense_lease)
        freeze = self.calibration_freeze
        frame = self.oracle_sampling_frame
        dense_cal = freeze.dense_calibration
        cand_cal = freeze.candidate_calibration
        if (
            dense_cal.model_state_hash != join.model_state_hash
            or cand_cal.model_state_hash != join.model_state_hash
            or dense_cal.source_label_map_hash != join.source_label_map_hash
            or cand_cal.source_label_map_hash != join.source_label_map_hash
            or dense_cal.calibration_context_axis_hash != join.source_context_axis_hash
            or cand_cal.calibration_context_axis_hash != join.source_context_axis_hash
            or dense_cal.calibration_source_hash != join.calibration_source_hash
            or cand_cal.calibration_source_hash != join.calibration_source_hash
            or dense_cal.query_hash != join.dense_calibration_query_hash
            or cand_cal.query_hash != join.candidate_calibration_query_hash
        ):
            raise ValueError("old affine calibration identities differ from actual-row issuer join")
        if (
            self.dense_mask.mask_hash != freeze.dense_referent_mask_hash
            or self.candidate_mask.mask_hash != freeze.candidate_mask_hash
            or frame.dense_referent_mask_hash != self.dense_mask.mask_hash
            or frame.candidate_mask_hash != self.candidate_mask.mask_hash
            # C freezes the binary action before K is acquired. The final
            # affine-calibration freeze is issued after K and therefore has a
            # different identity; equating them creates a circular dependency.
            or frame.candidate_freeze_hash != freeze.metadata.get("candidate_mask_freeze_hash")
            or frame.dense_calibration_hash != dense_cal.frozen_calibration_hash
            or frame.candidate_calibration_hash != cand_cal.frozen_calibration_hash
        ):
            raise ValueError("old FQ/mask/calibration coordinates do not agree")
        expected_causal_metadata = {
            "causal_generation_frame_receipt_hash": frame.generation_frame_hash,
            "certification_prefix_freeze_hash": frame.certification_prefix_freeze_hash,
            "calibration_prefix_freeze_hash": frame.calibration_prefix_freeze_hash,
            "oracle_sampling_frame_receipt_hash": frame.receipt_hash,
            "candidate_attempt_index": frame.candidate_attempt_index,
        }
        if any(freeze.metadata.get(name) != value for name, value in expected_causal_metadata.items()):
            raise ValueError("calibration freeze differs from its causal K/O frame")
        k_contexts = tuple(
            context for _, contexts in frame.calibration_prefix_context_ids_by_stratum
            for context in contexts
        )
        if freeze.calibration_context_ids != k_contexts:
            raise ValueError("calibration freeze contexts differ from the exact K prefix")
        if dense_cal.target_class_id != self.target_class or cand_cal.target_class_id != self.target_class:
            raise ValueError("calibration freeze target class differs from bridge target_class")
        object.__setattr__(self, "_structural_fingerprint", structural_fingerprint)
        object.__setattr__(self, "_model_leases", model_leases)
        object.__setattr__(self, "_candidate_mask_tensor", candidate_tensor)
        object.__setattr__(self, "_dense_mask_tensor", dense_tensor)
        object.__setattr__(self, "_candidate_mask_lease", candidate_lease)
        object.__setattr__(self, "_dense_mask_lease", dense_lease)
        if self.dense_mask.route_axis_hash != self.route_axis.route_axis_hash or self.candidate_mask.route_axis_hash != self.route_axis.route_axis_hash:
            raise ValueError("old HardMask route axis does not match supplied old DeclaredRouteAxis")
        self.assert_live()

    def assert_live(self) -> None:
        bridge._assert_model_snapshot(self.net, self._structural_fingerprint, self._model_leases)
        bridge._check_tensor_lease(self._candidate_mask_tensor, self._candidate_mask_lease)
        bridge._check_tensor_lease(self._dense_mask_tensor, self._dense_mask_lease)

    @property
    def dense_mask_tensor(self) -> torch.Tensor:
        self.assert_live()
        return self._dense_mask_tensor

    @property
    def candidate_mask_tensor(self) -> torch.Tensor:
        self.assert_live()
        return self._candidate_mask_tensor

    @property
    def dense_calibration(self) -> AffineCalibration:
        return affine_from_frozen(
            self.calibration_freeze.dense_calibration,
            expected_mask_hash=self.dense_mask.mask_hash,
            target_class=self.target_class,
        )

    @property
    def candidate_calibration(self) -> AffineCalibration:
        return affine_from_frozen(
            self.calibration_freeze.candidate_calibration,
            expected_mask_hash=self.candidate_mask.mask_hash,
            target_class=self.target_class,
        )

    def _require_source_owned_context_source(self, context_source: ContextSource) -> ContextSource:
        """Return the exact issued bounded source adapter or refuse before outcomes open.

        The old FQ/source issuance join names the bounded source owner that is
        allowed to supply actual rows for this bridge.  A caller-provided source
        object is therefore data, not authority: K evaluation and O row-provider
        construction must re-bind it to the issued owner before any source
        callback, disclosure, or evaluator path can observe rows.
        """
        if self.source_issuance is None:
            if self.diagnostic_without_source_issuance is True:
                return context_source
            raise ValueError("claim-bearing context rows require source-owned actual-row issuance")
        if type(context_source) is not BoundedImageContextSource:
            raise TypeError("claim-bearing context rows require the issued BoundedImageContextSource")
        if context_source.actual_source_binding_hash != self.source_issuance.actual_source_binding_hash:
            raise ValueError("context source binding differs from source-owned actual-row issuance")
        if context_source.source_identity_hash != self.identities.context_source_identity_hash:
            raise ValueError("bridge context-source identity differs from the supplied context source")
        if context_source.source_context_axis_hash != self.issuer_join.source_context_axis_hash:
            raise ValueError("context source ContextAxis differs from actual-row issuer join")
        if context_source.source_label_map_hash != self.issuer_join.source_label_map_hash:
            raise ValueError("context source label map differs from actual-row issuer join")
        if context_source.preprocessing_hash != self.issuer_join.preprocessing_hash:
            raise ValueError("context source preprocessing differs from actual-row issuer join")
        return context_source

    def evaluate_calibration_prefix_rows(
        self,
        *,
        context_source: ContextSource,
        forward_batch: int = 64,
        k_source_chunk_rows: int = 64,
        k_forward_batch_rows: int | None = None,
    ) -> tuple[tuple[tuple[str, tuple[float, ...]], ...], str]:
        if type(forward_batch) is not int or forward_batch <= 0:
            raise ValueError("forward_batch must be a positive integer")
        if type(k_source_chunk_rows) is not int or k_source_chunk_rows <= 0:
            raise ValueError("k_source_chunk_rows must be a positive integer")
        forward_rows = forward_batch if k_forward_batch_rows is None else k_forward_batch_rows
        if type(forward_rows) is not int or forward_rows <= 0:
            raise ValueError("k_forward_batch_rows must be a positive integer")
        self.assert_live()
        context_source = self._require_source_owned_context_source(context_source)
        candidate_mask = self.candidate_mask_tensor
        dense_mask = self.dense_mask_tensor
        candidate_calibration = self.candidate_calibration
        dense_calibration = self.dense_calibration
        self.assert_live()
        rows_by_stratum: list[tuple[str, tuple[float, ...]]] = []
        chunk_receipts: list[tuple[str, int, int, str]] = []
        forward_partitions: list[tuple[str, int, int]] = []

        def evaluate_partition(
            *,
            stratum_id: str,
            contexts: tuple[str, ...],
            images: torch.Tensor,
            labels: torch.Tensor,
            stratum_rows: list[float],
        ) -> None:
            partition_receipt = content_hash(
                {
                    "schema": "tractable_v1_bridge_owned_k_forward_partition_v1",
                    "oracle_sampling_frame_receipt_hash": self.oracle_sampling_frame.receipt_hash,
                    "stratum": stratum_id,
                    "ordered_context_ids": contexts,
                    "bridge_owned_clone_scope": True,
                },
                prefix="tractable-v1-k-forward-partition-",
            )
            partition_batch = ContextTensorBatch(
                context_ids=contexts,
                images=images,
                labels=labels,
                source_batch_receipt_hash=partition_receipt,
            )
            partition_leases = bridge._batch_tensor_leases(partition_batch)
            self.assert_live()
            bridge._check_batch_tensor_leases(partition_batch, partition_leases)
            candidate_logits = raw_target_logits(
                self.net,
                partition_batch.images,
                target_class=self.target_class,
                batch_size=forward_batch,
                mask_override=candidate_mask,
            )
            self.assert_live()
            bridge._check_batch_tensor_leases(partition_batch, partition_leases)
            dense_logits = raw_target_logits(
                self.net,
                partition_batch.images,
                target_class=self.target_class,
                batch_size=forward_batch,
                mask_override=dense_mask,
                use_dense_biases=True,
            )
            self.assert_live()
            bridge._check_batch_tensor_leases(partition_batch, partition_leases)
            row_tensor = per_context_bce_increment_rows(
                candidate_logits=candidate_logits,
                dense_logits=dense_logits,
                labels=partition_batch.labels,
                candidate_calibration=candidate_calibration,
                dense_calibration=dense_calibration,
            )
            self.assert_live()
            bridge._check_batch_tensor_leases(partition_batch, partition_leases)
            rows = tuple(float(value) for value in row_tensor.detach().cpu().reshape(-1).tolist())
            bridge._check_batch_tensor_leases(partition_batch, partition_leases)
            if len(rows) != len(contexts):
                raise ValueError("K row count differs from its fixed forward partition")
            stratum_rows.extend(rows)

        for stratum_id, contexts in self.oracle_sampling_frame.calibration_prefix_context_ids_by_stratum:
            contexts = tuple(str(value) for value in contexts)
            stratum_rows: list[float] = []
            if stratum_id not in old.PAIRED_BCE_STRATA:
                raise ValueError("unknown K paired-BCE stratum")
            pending_contexts: list[str] = []
            pending_images: list[torch.Tensor] = []
            pending_labels: list[torch.Tensor] = []
            pending_start = 0

            def flush() -> None:
                nonlocal pending_contexts, pending_images, pending_labels, pending_start
                if not pending_contexts:
                    return
                part_contexts = tuple(pending_contexts)
                images = torch.cat(pending_images, dim=0)
                labels = torch.cat(pending_labels, dim=0)
                stop = pending_start + len(part_contexts)
                forward_partitions.append((stratum_id, pending_start, stop))
                evaluate_partition(
                    stratum_id=stratum_id,
                    contexts=part_contexts,
                    images=images,
                    labels=labels,
                    stratum_rows=stratum_rows,
                )
                pending_contexts = []
                pending_images = []
                pending_labels = []
                pending_start = stop

            for start in range(0, len(contexts), k_source_chunk_rows):
                stop = min(len(contexts), start + k_source_chunk_rows)
                chunk = contexts[start:stop]
                self.assert_live()
                batch = context_source(stratum_id, chunk)
                if type(batch) is not bridge.ContextTensorBatch:
                    raise TypeError("context_source must return ContextTensorBatch")
                batch_leases = bridge._batch_tensor_leases(batch)
                self.assert_live()
                bridge._check_batch_tensor_leases(batch, batch_leases)
                if batch.context_ids != chunk:
                    raise ValueError("K context source returned a different ordered context axis")
                if stratum_id == old.TARGET_STRATUM:
                    if not bool((batch.labels == 1.0).all()):
                        raise ValueError("K target stratum returned non-target labels")
                else:
                    if not bool((batch.labels == 0.0).all()):
                        raise ValueError("K complement stratum returned target labels")
                bridge._check_batch_tensor_leases(batch, batch_leases)
                self.assert_live()
                cursor = 0
                while cursor < len(chunk):
                    remaining = forward_rows - len(pending_contexts)
                    take = min(remaining, len(chunk) - cursor)
                    pending_contexts.extend(chunk[cursor:cursor + take])
                    pending_images.append(batch.images[cursor:cursor + take].clone())
                    pending_labels.append(batch.labels[cursor:cursor + take].clone())
                    cursor += take
                    bridge._check_batch_tensor_leases(batch, batch_leases)
                    self.assert_live()
                    if len(pending_contexts) == forward_rows:
                        flush()
                chunk_receipts.append((stratum_id, start, stop, batch.source_batch_receipt_hash))
            flush()
            rows_by_stratum.append((stratum_id, tuple(stratum_rows)))
        self.assert_live()
        receipt_hash = content_hash(
            {
                "schema": "tractable_v1_source_owned_k_row_evaluation_receipt_v3",
                "oracle_sampling_frame_receipt_hash": self.oracle_sampling_frame.receipt_hash,
                "candidate_mask_freeze_hash": self.oracle_sampling_frame.candidate_freeze_hash,
                "candidate_calibration_freeze_hash": self.calibration_freeze.freeze_hash,
                "dense_mask_hash": self.dense_mask.mask_hash,
                "candidate_mask_hash": self.candidate_mask.mask_hash,
                "dense_calibration_hash": self.calibration_freeze.dense_calibration.frozen_calibration_hash,
                "candidate_calibration_hash": self.calibration_freeze.candidate_calibration.frozen_calibration_hash,
                "actual_model_content_audit_hash": self.issuer_join.actual_model_content_audit_hash,
                "context_source_identity_hash": self.identities.context_source_identity_hash,
                "preprocessing_identity_hash": self.identities.preprocessing_identity_hash,
                "source_chunk_receipts": tuple(chunk_receipts),
                "forward_partitions": tuple(forward_partitions),
                "k_source_chunk_rows": k_source_chunk_rows,
                "k_forward_batch_rows": forward_rows,
                "row_counts": tuple((name, len(rows)) for name, rows in rows_by_stratum),
            },
            prefix="tractable-v1-source-owned-k-paired-bce-",
        )
        return tuple(rows_by_stratum), receipt_hash

    def build_registration_from_k_rows(
        self,
        *,
        loss_ceiling: float,
        calibration_rows_by_stratum: Sequence[tuple[str, Sequence[float]]],
        calibration_paired_loss_evaluator_receipt_hash: str,
        initial_look_count: int = 1,
        look_growth_factor: int = 2,
    ) -> old.V1PairedBCEAcquisitionRegistration:
        """Build a diagnostic census-only registration from the supplied K rows.

        Production bounded sequential use must retain the comprehensive
        production registration factory and its support-certificate recovery.
        This helper supplies neither partial-confidence nor commit authority.
        """
        self.assert_live()
        identities = self.identities
        loss_kind = old.CENSUS_ONLY_PAIRED_BCE_KIND
        ceiling_hash = old.paired_bce_loss_ceiling_identity_hash(
            dense_reference_hash=self.dense_mask.mask_hash,
            candidate_calibration_freeze_hash=self.calibration_freeze.freeze_hash,
            paired_loss_kind=loss_kind,
            loss_ceiling=float(loss_ceiling),
        )
        alpha_coordinate_hash = old.paired_bce_attempt_alpha_coordinate_hash(
            candidate_attempt_alpha_axis_hash=identities.candidate_attempt_alpha_axis_hash,
            candidate_attempt_index=self.oracle_sampling_frame.candidate_attempt_index,
            maximum_candidate_attempts=identities.maximum_candidate_attempts,
            attempt_alpha_allocation=identities.attempt_alpha_allocation,
        )
        dynamic = old.register_paired_bce_dynamic_oracle(
            oracle_sampling_frame=self.oracle_sampling_frame,
            candidate_calibration_freeze_hash=self.calibration_freeze.freeze_hash,
            loss_ceiling_identity_hash=ceiling_hash,
            candidate_attempt_alpha_coordinate_hash=alpha_coordinate_hash,
            calibration_paired_loss_rows_by_stratum=calibration_rows_by_stratum,
            calibration_paired_loss_evaluator_receipt_hash=_nonempty(
                calibration_paired_loss_evaluator_receipt_hash,
                "calibration_paired_loss_evaluator_receipt_hash",
            ),
        )
        strata = tuple(
            old.V1PairedBCEStratumRegistration(
                stratum_id=stratum_id,
                stratum_weight=0.5,
                population_size=dynamic.oracle_population_stratum_counts[index],
                population_identity_hash=dynamic.oracle_population_identity_hashes[index],
                sealed_order_hash=dynamic.oracle_sealed_order_hashes[index],
                sealed_permutation_receipt_hash=dynamic.oracle_permutation_receipt_hashes[index],
                bound_certificate_hash=identities.bound_certificate_hash_by_stratum[stratum_id],
                paired_loss_lower_bound=None,
                paired_loss_upper_bound=None,
            )
            for index, stratum_id in enumerate(old.PAIRED_BCE_STRATA)
        )
        self.assert_live()
        return old.build_paired_bce_acquisition_registration(
            registration_id=self.registration_id,
            campaign_contract_hash=self.campaign_contract_hash,
            candidate_mask_hash=self.candidate_mask.mask_hash,
            dense_reference_hash=self.dense_mask.mask_hash,
            candidate_calibration_freeze_hash=self.calibration_freeze.freeze_hash,
            paired_loss_kind=loss_kind,
            loss_ceiling=float(loss_ceiling),
            loss_ceiling_identity_hash=ceiling_hash,
            source_population_identity_hash=dynamic.source_population_identity_hash,
            oracle_fold_identity_hash=dynamic.oracle_sampling_frame_receipt_hash,
            oracle_role_separation_receipt=None,
            strata=strata,
            oracle_order_entropy_commitment_hash=dynamic.oracle_order_entropy_commitment_hash,
            sampling_law_identity_hash=identities.sampling_law_identity_hash,
            sealed_order_slice_verifier_identity_hash=identities.sealed_order_slice_verifier_identity_hash,
            candidate_attempt_alpha_axis_hash=identities.candidate_attempt_alpha_axis_hash,
            candidate_attempt_alpha_coordinate_hash=alpha_coordinate_hash,
            candidate_attempt_index=dynamic.candidate_attempt_index,
            maximum_candidate_attempts=identities.maximum_candidate_attempts,
            oracle_familywise_alpha=identities.oracle_familywise_alpha,
            alpha_committed_before_attempt=identities.alpha_committed_before_attempt,
            attempt_alpha_allocation=identities.attempt_alpha_allocation,
            dynamic_oracle_receipt=dynamic,
            initial_look_count=initial_look_count,
            look_growth_factor=look_growth_factor,
        )

    def build_epoch(self, *, registration: old.V1PairedBCEAcquisitionRegistration) -> MaskedAlexNetPairedBCEEpoch:
        self.assert_live()
        if type(registration) is not old.V1PairedBCEAcquisitionRegistration:
            raise TypeError("epoch bridge requires an exact old paired-BCE registration")
        dynamic = registration.dynamic_oracle_receipt
        if dynamic is None:
            raise ValueError("epoch bridge requires dynamic K/O registration")
        registration.assert_integrity()
        if (
            registration.campaign_contract_hash != self.campaign_contract_hash
            or registration.candidate_mask_hash != self.candidate_mask.mask_hash
            or registration.dense_reference_hash != self.dense_mask.mask_hash
            or registration.candidate_calibration_freeze_hash != self.calibration_freeze.freeze_hash
            or dynamic.oracle_sampling_frame_receipt_hash != self.oracle_sampling_frame.receipt_hash
            or dynamic.dense_affine_calibration_hash != self.calibration_freeze.dense_calibration.frozen_calibration_hash
            or dynamic.candidate_affine_calibration_hash != self.calibration_freeze.candidate_calibration.frozen_calibration_hash
        ):
            raise ValueError("paired-BCE registration is stale for the old FQ tractable bridge")
        self.assert_live()
        return MaskedAlexNetPairedBCEEpoch(
            registration=registration,
            net=self.net,
            candidate_mask=self.candidate_mask_tensor,
            dense_mask=self.dense_mask_tensor,
            candidate_calibration=self.candidate_calibration,
            dense_calibration=self.dense_calibration,
            row_provider_identity_hash=self.identities.row_provider_identity_hash,
            paired_loss_evaluator_identity_hash=self.identities.paired_loss_evaluator_identity_hash,
            dense_reference_hash=self.dense_mask.mask_hash,
            candidate_mask_hash=self.candidate_mask.mask_hash,
            dense_affine_calibration_hash=dynamic.dense_affine_calibration_hash,
            candidate_affine_calibration_hash=dynamic.candidate_affine_calibration_hash,
            target_class=self.target_class,
            context_source_identity_hash=self.identities.context_source_identity_hash,
            preprocessing_identity_hash=self.identities.preprocessing_identity_hash,
        )

    def build_row_provider(
        self,
        *,
        registration: old.V1PairedBCEAcquisitionRegistration,
        context_source: ContextSource,
        forward_batch: int = 64,
    ) -> MaskedAlexNetPairedBCERowProvider:
        context_source = self._require_source_owned_context_source(context_source)
        return MaskedAlexNetPairedBCERowProvider(
            epoch=self.build_epoch(registration=registration),
            context_source=context_source,
            forward_batch=forward_batch,
        )


def oracle_context_orders(frame: V1AlexNetFQOracleSamplingFrameReceipt) -> Mapping[str, tuple[str, ...]]:
    if type(frame) is not V1AlexNetFQOracleSamplingFrameReceipt:
        raise TypeError("oracle orders require exact old FQ oracle frame")
    frame.assert_integrity()
    return MappingProxyType({name: tuple(values) for name, values in frame.oracle_context_ids_by_stratum})
