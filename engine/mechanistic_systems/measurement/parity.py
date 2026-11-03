"""Exact migration parity between canonical measurements and legacy ScoreStores."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from contracts.hashing import stable_digest
from mechanistic_systems.core.architectures import (
    ExecutableRouteBinding,
    canonical_torch_architecture_spec,
    historical_route_alias,
)
from mechanistic_systems.measurement.streaming import (
    SufficientStatisticsMeasurementReader,
)
from mechanistic_systems.measurement.scorestore_format import (
    load_scorestore_target_payload,
)

LEGACY_SCORE_PARITY_SCHEMA = "mechanistic_legacy_scorestore_parity_v1"


def legacy_scorestore_layer_id(binding: ExecutableRouteBinding) -> str:
    """Lower one stable measurement binding to its historical ScoreStore key."""

    return historical_route_alias(binding)


def legacy_layer_map(architecture_family: str) -> dict[str, str]:
    spec = canonical_torch_architecture_spec(architecture_family)
    return {
        binding.layer_id: legacy_scorestore_layer_id(binding)
        for binding in spec.bindings
    }


def _correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 1e-15 else 1.0


def _compare_head(
    *,
    reader: SufficientStatisticsMeasurementReader,
    query_index: int,
    reference_by_layer: Mapping[str, Any],
    stable_to_legacy: Mapping[str, str],
    score_head: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    maximum_error = 0.0
    residual_squared = 0.0
    reference_squared = 0.0
    element_count = 0
    all_close = True
    rows: list[dict[str, Any]] = []
    for block in reader.route_blocks:
        legacy_layer = stable_to_legacy[block.layer_id]
        reference = (
            reference_by_layer[legacy_layer]
            .detach()
            .to(device="cpu")
            .numpy()
            .astype(np.float64, copy=False)
            .reshape(-1)
        )
        candidate = reader.source_image_aggregate_block(
            block.layer_id,
            score_head=score_head,
        )[query_index].astype(np.float64, copy=False).reshape(-1)
        if reference.shape != candidate.shape:
            raise ValueError(
                f"score shape differs for {block.layer_id!r}: "
                f"legacy={reference.shape}, canonical={candidate.shape}"
            )
        residual = candidate - reference
        block_maximum = float(np.max(np.abs(residual)))
        block_reference_squared = float(reference @ reference)
        block_residual_squared = float(residual @ residual)
        block_relative = float(
            np.sqrt(block_residual_squared) /
            max(np.sqrt(block_reference_squared), 1e-15)
        )
        block_close = bool(
            np.allclose(
                candidate,
                reference,
                atol=float(absolute_tolerance),
                rtol=float(relative_tolerance),
            )
        )
        rows.append(
            {
                "stable_layer_id": block.layer_id,
                "legacy_layer_id": legacy_layer,
                "route_count": int(reference.size),
                "maximum_absolute_error": block_maximum,
                "relative_l2_error": block_relative,
                "pearson": _correlation(reference, candidate),
                "sign_agreement": float(
                    np.mean(np.signbit(reference) == np.signbit(candidate))
                ),
                "reference_l2": float(np.sqrt(block_reference_squared)),
                "candidate_l2": float(np.linalg.norm(candidate)),
                "passed": block_close,
            }
        )
        maximum_error = max(maximum_error, block_maximum)
        residual_squared += block_residual_squared
        reference_squared += block_reference_squared
        element_count += int(reference.size)
        all_close = all_close and block_close
    return {
        "score_head": score_head,
        "passed": bool(all_close),
        "element_count": element_count,
        "maximum_absolute_error": maximum_error,
        "relative_l2_error": float(
            np.sqrt(residual_squared) / max(np.sqrt(reference_squared), 1e-15)
        ),
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "blocks": rows,
    }


def build_legacy_scorestore_parity_report(
    *,
    candidate_measurement_zarr: str | Path,
    legacy_scorestore_root: str | Path,
    architecture_family: str,
    query_id: str,
    expected_source_context_count: int,
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare exact image-sum score heads over one coordinate-identical query."""

    if int(expected_source_context_count) <= 0:
        raise ValueError("score parity requires a positive source context count")
    if float(absolute_tolerance) < 0.0 or float(relative_tolerance) < 0.0:
        raise ValueError("score parity tolerances must be nonnegative")
    reader = SufficientStatisticsMeasurementReader(candidate_measurement_zarr)
    try:
        query_index = reader.query_ids.index(str(query_id))
    except ValueError as error:
        raise ValueError(f"candidate measurement lacks query {query_id!r}") from error
    stable_to_legacy = legacy_layer_map(architecture_family)
    observed_stable = tuple(block.layer_id for block in reader.route_blocks)
    expected_stable = tuple(stable_to_legacy)
    if observed_stable != expected_stable:
        raise ValueError(
            "candidate measurement route blocks differ from the canonical architecture profile"
        )

    root = Path(legacy_scorestore_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if manifest.get("canonical_exact") is not True:
        raise ValueError("legacy score parity requires a canonical exact ScoreStore")
    if str(query_id) not in index.get("targets", {}):
        raise ValueError(f"legacy ScoreStore lacks target {query_id!r}")
    reference = {
        head: load_scorestore_target_payload(
            root,
            target=str(query_id),
            payload=head,
            verify=True,
        )
        for head in ("route_scores", "route_abs_scores")
    }
    expected_legacy = tuple(stable_to_legacy.values())
    for head, values in reference.items():
        if set(values) != set(expected_legacy):
            missing = sorted(set(expected_legacy) - set(values))
            extra = sorted(set(values) - set(expected_legacy))
            raise ValueError(
                f"legacy {head} route blocks differ; missing={missing[:8]}, extra={extra[:8]}"
            )

    context_counts = {
        str(key): int(value)
        for key, value in dict(
            reader.root.attrs.get("source_context_count_by_block", {}) or {}
        ).items()
    }
    source_context_count = int(sum(context_counts.values()))
    contract_checks = {
        "candidate_query_present": True,
        "candidate_route_profile_exact": True,
        "legacy_store_canonical_exact": True,
        "legacy_target_present": True,
        "source_context_count_matches": (
            source_context_count == int(expected_source_context_count)
        ),
        "candidate_estimator_exact_actgrad": (
            str(reader.root.attrs.get("estimator_id", "")) == "exact_actgrad_vjp"
        ),
    }
    heads = {
        head: _compare_head(
            reader=reader,
            query_index=query_index,
            reference_by_layer=reference[head],
            stable_to_legacy=stable_to_legacy,
            score_head=head,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for head in ("route_scores", "route_abs_scores")
    }
    passed = bool(all(contract_checks.values()) and all(row["passed"] for row in heads.values()))
    identity = {
        "schema_name": LEGACY_SCORE_PARITY_SCHEMA,
        "architecture_family": str(architecture_family),
        "query_id": str(query_id),
        "candidate_route_axis_hash": reader.route_axis_hash,
        "candidate_query_ids": list(reader.query_ids),
        "candidate_context_block_ids": list(reader.context_block_ids),
        "source_context_count": source_context_count,
        "expected_source_context_count": int(expected_source_context_count),
        "contract_checks": contract_checks,
        "heads": heads,
        "passed": passed,
        "outcome": (
            "passed_exact_score_migration_parity"
            if passed
            else "failed_score_migration_parity"
        ),
    }
    return {
        **identity,
        "parity_receipt_hash": stable_digest(identity),
        "candidate_measurement_zarr": str(Path(candidate_measurement_zarr)),
        "legacy_scorestore_root": str(root),
    }


__all__ = [
    "LEGACY_SCORE_PARITY_SCHEMA",
    "build_legacy_scorestore_parity_report",
    "legacy_layer_map",
    "legacy_scorestore_layer_id",
]
