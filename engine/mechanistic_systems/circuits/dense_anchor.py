"""Native dense-reference receipts for sampled-path circuit identification."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.hashing import stable_digest
from mechanistic_systems.contracts import QuerySpec

from .sampled_path import SampledComparisonRequest, SampledReferenceEvidence


NATIVE_DENSE_ANCHOR_SCHEMA = "mechanistic_native_dense_anchor_receipt_v1"


def _float64_vector_hash(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8").reshape(-1)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _query_class_id(query: QuerySpec) -> int:
    value = query.evaluator.get(
        "class_id",
        query.evaluator.get("unit", query.evaluator.get("coordinate")),
    )
    if value is None:
        raise ValueError("native dense anchors require coordinate class queries")
    return int(value)


@dataclass(frozen=True)
class NativeDenseAnchorReceipt:
    """Proof that dense losses were measured inside the canonical v1 runtime."""

    plan_hash: str
    profile_hash: str
    optimization_axis_hash: str
    model_load_receipt_hash: str
    query_ids: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]
    backend_id: str
    estimator: str = "stage0_crossfit_heldout_row_bce_mean"
    schema_name: str = NATIVE_DENSE_ANCHOR_SCHEMA

    @property
    def dense_anchor_loss_by_query(self) -> dict[str, float]:
        return {
            str(row["query_id"]): float(row["canonical_loss"])
            for row in self.rows
        }

    def validate(self) -> None:
        hashes = (
            self.plan_hash,
            self.profile_hash,
            self.optimization_axis_hash,
            self.model_load_receipt_hash,
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("native dense anchors require complete parent hashes")
        if not self.backend_id:
            raise ValueError("native dense anchors require a backend identity")
        if not self.query_ids or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("native dense-anchor query axis is invalid")
        if tuple(str(row.get("query_id", "")) for row in self.rows) != self.query_ids:
            raise ValueError("native dense-anchor rows differ from the query axis")
        for row in self.rows:
            if int(row.get("context_count", 0)) <= 1:
                raise ValueError("native dense-anchor panel is empty")
            if int(row.get("positive_count", 0)) <= 0 or int(
                row.get("negative_count", 0)
            ) <= 0:
                raise ValueError("native dense anchors require both binary classes")
            loss = float(row.get("canonical_loss", float("nan")))
            if not math.isfinite(loss) or loss < 0.0:
                raise ValueError("native dense-anchor loss must be finite and nonnegative")
            for key in (
                "panel_hash",
                "panel_context_ids_hash",
                "row_loss_hash",
                "mask_hash",
                "calibration_evidence_hash",
            ):
                if len(str(row.get(key, ""))) != 64:
                    raise ValueError(f"native dense-anchor {key} is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "plan_hash": self.plan_hash,
            "profile_hash": self.profile_hash,
            "optimization_axis_hash": self.optimization_axis_hash,
            "model_load_receipt_hash": self.model_load_receipt_hash,
            "query_ids": list(self.query_ids),
            "rows": [dict(row) for row in self.rows],
            "backend_id": self.backend_id,
            "estimator": self.estimator,
            "dense_anchor_loss_by_query": self.dense_anchor_loss_by_query,
            "content_hash": self.content_hash,
        }

    @property
    def content_hash(self) -> str:
        payload = {
            "schema_name": self.schema_name,
            "plan_hash": self.plan_hash,
            "profile_hash": self.profile_hash,
            "optimization_axis_hash": self.optimization_axis_hash,
            "model_load_receipt_hash": self.model_load_receipt_hash,
            "query_ids": list(self.query_ids),
            "rows": [dict(row) for row in self.rows],
            "backend_id": self.backend_id,
            "estimator": self.estimator,
        }
        return stable_digest(payload)


def derive_native_dense_anchor_receipt(
    *,
    backend: Any,
    profile: Any,
    queries: Sequence[QuerySpec],
    dense_masks: Mapping[str, np.ndarray],
    first_layer_id: str,
    plan_hash: str,
    optimization_axis_hash: str,
    model_load_receipt_hash: str,
) -> NativeDenseAnchorReceipt:
    """Measure the exact production-v1 dense reference without legacy files."""

    profile.validate()
    query_axis = tuple(queries)
    if not query_axis or len({query.query_id for query in query_axis}) != len(query_axis):
        raise ValueError("native dense anchors require unique queries")
    if not dense_masks or not first_layer_id:
        raise ValueError("native dense anchors require the complete dense route axis")
    rows: list[dict[str, Any]] = []
    for query in query_axis:
        request = SampledComparisonRequest(
            query_id=query.query_id,
            layer_id=str(first_layer_id),
            partition_id="dense_reference",
            previous_density=1.0,
            proposed_density=1.0,
            stage_index=0,
            stage_per_class=int(profile.sample_stages_per_class[0]),
            proposal_index=0,
            random_seed=int(profile.statistical_sampling_seed),
            fold_count=int(profile.crossfit_fold_count),
        )
        evidence: SampledReferenceEvidence = backend.evaluate_dense_reference(
            dense_masks,
            request,
        )
        evidence.validate(fold_count=int(profile.crossfit_fold_count))
        values = np.asarray(evidence.losses, dtype=np.float64).reshape(-1)
        labels = np.asarray(evidence.panel.labels, dtype=np.int64).reshape(-1)
        class_id = _query_class_id(query)
        positive_count = int(np.count_nonzero(labels == class_id))
        calibration_hash = stable_digest(
            [dict(item) for item in evidence.calibration_by_fold]
        )
        rows.append(
            {
                "query_id": query.query_id,
                "class_id": class_id,
                "layer_id": str(first_layer_id),
                "stage_index": 0,
                "stage_per_class": int(profile.sample_stages_per_class[0]),
                "fold_count": int(profile.crossfit_fold_count),
                "context_count": int(values.size),
                "positive_count": positive_count,
                "negative_count": int(values.size) - positive_count,
                "canonical_loss": float(values.mean()),
                "canonical_loss_metric": (
                    "adaptive_label_weighted_binary_cross_entropy"
                ),
                "panel_hash": evidence.panel.panel_hash,
                "panel_context_ids_hash": hashlib.sha256(
                    "\x1f".join(evidence.panel.context_ids).encode("utf-8")
                ).hexdigest(),
                "row_loss_hash": _float64_vector_hash(values),
                "mask_hash": evidence.mask_hash,
                "calibration_evidence_hash": calibration_hash,
                "calibration_fold_count": len(evidence.calibration_by_fold),
                "final_validation_accessed": False,
            }
        )
    receipt = NativeDenseAnchorReceipt(
        plan_hash=str(plan_hash),
        profile_hash=profile.content_hash,
        optimization_axis_hash=str(optimization_axis_hash),
        model_load_receipt_hash=str(model_load_receipt_hash),
        query_ids=tuple(query.query_id for query in query_axis),
        rows=tuple(rows),
        backend_id=str(backend.backend_id),
    )
    receipt.validate()
    return receipt


__all__ = [
    "NATIVE_DENSE_ANCHOR_SCHEMA",
    "NativeDenseAnchorReceipt",
    "derive_native_dense_anchor_receipt",
]
