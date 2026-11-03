"""Lower canonical route-gate measurements to exact pruning ScoreStores.

The Zarr measurement is the scientific artifact.  ScoreStore is an exact,
lossless execution view for the existing sampled-path pruning backend.  This
module is the only active place where stable route coordinates are lowered to
historical module-path keys.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mechanistic_systems.measurement.parity import legacy_layer_map
from mechanistic_systems.measurement.streaming import (
    SufficientStatisticsMeasurementReader,
)
from mechanistic_systems.measurement.scorestore_format import (
    ScoreStoreWriteReport,
    canonical_target_scorestore_dir,
    write_exact_score_store,
)
from mechanistic_systems.measurement.heads import (
    CORE_ROUTE_SCORE_HEADS,
    ROUTE_ABS_SCORE_HEAD,
    ROUTE_SCORE_HEAD,
    UNIT_ABS_SCORE_HEAD,
    UNIT_SCORE_HEAD,
    unit_reduction_from_route_scores,
)


UNIFIED_GATE_SCORESTORE_SCHEMA = "mechanistic_unified_route_gate_scorestore_v1"


_SOURCE_AGGREGATE_STATISTICS = {
    ROUTE_SCORE_HEAD: "source_score_sum",
    ROUTE_ABS_SCORE_HEAD: "source_absolute_score_sum",
}


def _source_aggregates_by_layer(
    reader: SufficientStatisticsMeasurementReader,
) -> dict[str, dict[str, np.ndarray]]:
    """One Zarr read per measured block, serving every query and head.

    A per-(query, head) path would re-read and re-decompress every
    block's full statistics group ``queries x heads`` times — the dominant
    lowering cost after compression itself. Values are the exact image-sum
    heads ``source_image_aggregate_block`` returns, byte-identically.
    """

    aggregates: dict[str, dict[str, np.ndarray]] = {}
    for block in reader.route_blocks:
        raw = reader.read_block(block.layer_id)
        aggregates[block.layer_id] = {
            score_head: raw[statistic].astype(np.float64)
            for score_head, statistic in _SOURCE_AGGREGATE_STATISTICS.items()
        }
    return aggregates


def _query_layer_map(
    reader: SufficientStatisticsMeasurementReader,
    *,
    query_index: int,
    score_head: str,
    stable_to_executable: Mapping[str, str],
    source_aggregates: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for block in reader.route_blocks:
        try:
            executable_layer = stable_to_executable[block.layer_id]
        except KeyError as error:
            raise ValueError(
                f"measurement route block {block.layer_id!r} has no executable lowering"
            ) from error
        if source_aggregates is not None:
            values = source_aggregates[block.layer_id][score_head][
                int(query_index)
            ]
        else:
            values = reader.source_image_aggregate_block(
                block.layer_id,
                score_head=score_head,
            )[int(query_index)]
        array = np.asarray(values, dtype=np.float32).reshape(block.shape, order="C")
        if not np.isfinite(array).all():
            raise ValueError(
                f"measurement contains non-finite {score_head} values at {block.layer_id!r}"
            )
        result[executable_layer] = torch.from_numpy(array.copy())
    return result


def lower_measurement_to_scorestores(
    *,
    measurement_zarr: str | Path,
    scorestore_root: str | Path,
    architecture_family: str,
    cell_id: str,
    target_family: str = "class_logit",
    source_metadata: Mapping[str, Any] | None = None,
    scorestore_query_ids: Mapping[str, str] | None = None,
) -> tuple[ScoreStoreWriteReport, ...]:
    """Write one exact ScoreStore per query from a canonical measurement.

    No score is recomputed here.  Signed image sums and per-image absolute
    image sums are copied from the canonical sufficient-statistics artifact;
    downstream-unit heads are deterministic reductions of those route heads.
    """

    reader = SufficientStatisticsMeasurementReader(measurement_zarr)
    canonical_query_ids = tuple(str(value) for value in reader.query_ids)
    lowered_query_ids = {
        query_id: str(
            query_id
            if scorestore_query_ids is None
            else scorestore_query_ids.get(query_id, "")
        )
        for query_id in canonical_query_ids
    }
    if scorestore_query_ids is not None and set(scorestore_query_ids) != set(
        canonical_query_ids
    ):
        raise ValueError(
            "ScoreStore query lowering must cover exactly the canonical query axis"
        )
    if any(not value for value in lowered_query_ids.values()) or len(
        set(lowered_query_ids.values())
    ) != len(lowered_query_ids):
        raise ValueError("lowered ScoreStore query IDs must be nonempty and unique")
    stable_to_executable = legacy_layer_map(architecture_family)
    observed = tuple(block.layer_id for block in reader.route_blocks)
    expected = tuple(stable_to_executable)
    if observed != expected:
        raise ValueError(
            "measurement route profile differs from the canonical architecture: "
            f"observed={observed[:8]!r}, expected={expected[:8]!r}"
        )
    root = Path(scorestore_root)
    root.mkdir(parents=True, exist_ok=True)
    aliases: dict[str, str] = {}
    route_schema = {
        "schema_name": UNIFIED_GATE_SCORESTORE_SCHEMA,
        "architecture_family": str(architecture_family),
        "route_axis_hash": reader.route_axis_hash,
        "route_blocks": [block.to_dict() for block in reader.route_blocks],
        "stable_to_executable_layer": dict(stable_to_executable),
        "gate_estimand": (
            "d_sum_query_response_d_alpha_at_alpha_1_for_executable_route_gate"
        ),
        "route_kinds": sorted({block.route_kind for block in reader.route_blocks}),
    }
    reports: list[ScoreStoreWriteReport] = []
    source_aggregates = _source_aggregates_by_layer(reader)
    for query_index, query_id in enumerate(canonical_query_ids):
        scorestore_query_id = lowered_query_ids[query_id]
        route_scores = _query_layer_map(
            reader,
            query_index=query_index,
            score_head=ROUTE_SCORE_HEAD,
            stable_to_executable=stable_to_executable,
            source_aggregates=source_aggregates,
        )
        route_abs_scores = _query_layer_map(
            reader,
            query_index=query_index,
            score_head=ROUTE_ABS_SCORE_HEAD,
            stable_to_executable=stable_to_executable,
            source_aggregates=source_aggregates,
        )
        payload = {
            ROUTE_SCORE_HEAD: route_scores,
            ROUTE_ABS_SCORE_HEAD: route_abs_scores,
            UNIT_SCORE_HEAD: unit_reduction_from_route_scores(
                route_scores,
                absolute=False,
            ),
            UNIT_ABS_SCORE_HEAD: unit_reduction_from_route_scores(
                route_abs_scores,
                absolute=False,
            ),
        }
        reports.append(
            write_exact_score_store(
                output_dir=canonical_target_scorestore_dir(root, scorestore_query_id),
                target_payloads={scorestore_query_id: payload},
                cell_id=str(cell_id),
                target_family=str(target_family),
                route_schema=route_schema,
                source_score_config={
                    "schema_name": UNIFIED_GATE_SCORESTORE_SCHEMA,
                    "measurement_zarr": str(Path(measurement_zarr)),
                    "estimator_id": str(reader.root.attrs.get("estimator_id", "")),
                    "source_context_count_by_block": dict(
                        reader.root.attrs.get("source_context_count_by_block", {}) or {}
                    ),
                    "signed_aggregation": "float64_host_sum_then_exact_float32_storage",
                    "absolute_aggregation": "sum_abs_per_context_route_gate_derivative",
                    "canonical_query_id": query_id,
                    "scorestore_query_id": scorestore_query_id,
                    "query_coordinate_lowering": (
                        "identity"
                        if query_id == scorestore_query_id
                        else "explicit_legacy_direction_bank_row"
                    ),
                    "source_metadata": dict(source_metadata or {}),
                },
                include_payload_keys=CORE_ROUTE_SCORE_HEADS,
                aliases=aliases,
                # Runtime-only: shard zlib is 52% of lowering wall (measured)
                # and is losslessly optional - arrays and manifest
                # sha256 semantics are identical either way. Cluster jobs
                # opt out via CIRCUIT_SCORESTORE_COMPRESSION=0.
                compression=os.environ.get(
                    "CIRCUIT_SCORESTORE_COMPRESSION", "1"
                ).strip().lower() not in {"0", "false", "no", "off"},
            )
        )
    return tuple(reports)


__all__ = [
    "UNIFIED_GATE_SCORESTORE_SCHEMA",
    "lower_measurement_to_scorestores",
]
