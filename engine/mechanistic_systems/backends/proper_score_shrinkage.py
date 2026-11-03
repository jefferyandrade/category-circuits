"""Identity-anchored proper-score calibration for finite affine repair.

The calibrated natural parameter is a positive affine transform of one fixed
query output.  A Bernoulli predictive-KL term anchors that transform to the
identity map.  Unlike finite-horizon optimization, the declared objective has
a unique finite solution under separation whenever the query coordinate is
identifiable.  Regularization strength is selected only inside the supplied
calibration rows; extraction and validation outcomes are never inspected.
"""

from __future__ import annotations

import math
import threading
import time
from mechanistic_systems.measurement.run_products import loss_weighting

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from contracts.hashing import stable_digest
from mechanistic_systems.contracts import (
    CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON,
    CANONICAL_CALIBRATION_HESSIAN_RIDGE,
    CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS,
    CANONICAL_CALIBRATION_KKT_TOLERANCE,
    CANONICAL_CALIBRATION_MAX_ITERATIONS,
    CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE,
    CANONICAL_CALIBRATION_REFERENCE_TOLERANCE,
    CANONICAL_CALIBRATION_SELECTION_BLOCKS,
    PROPER_SCORE_SHRINKAGE_AFFINE_V2,
)

from .calibration_problem import CalibrationFactorization
from .calibration_runtime import CrossFitLayout
from .contracts import BackendProfile
from .factored_calibration import (
    GPU_HARD_GATES_AUDIT_POLICY,
    V6_MISSING_CELL_RULE,
    CalibrationFailClosedError,
    PositiveAffineDifferentialOwnerReceipt,
    calibration_tensor_content_hash,
    fit_batched_post_query_affine,
    fit_batched_post_query_affine_with_differential_owner,
    v6_missing_cell_positions,
)

try:  # Optional for import smoke; required by executable calibration.
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - environment dependent.
    torch = F = None  # type: ignore[assignment]


DEFAULT_IDENTITY_ANCHOR_STRENGTHS = (
    CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS
)
IDENTITY_CANDIDATE = "identity"
# The selected-strength token of a MISSING column, one
# whose selection lanes or deployment refit failed the device hard gates
# after the device fallback.  Its deployed parameters are NaN (unavailable);
# nothing is substituted.
MISSING_SELECTION_TOKEN = "missing_gpu_hard_gate_fail_closed"


def scientific_fit_receipt_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Fit receipt content used for identity, excluding measured runtime data.

    Both the proper-score receipt and its nested Newton receipt carry a
    backend profile with wall time. The full receipts remain available for
    diagnostics; this projection is the one already used by selection-face
    identity.
    """

    payload = dict(receipt)
    payload.pop("backend_profile", None)
    if isinstance(payload.get("fit_receipt"), Mapping):
        payload["fit_receipt"] = dict(payload["fit_receipt"])
        payload["fit_receipt"].pop("backend_profile", None)
    return payload
# How a v6 fit hands a missing column to its consumer.  ``carry``: the column
# is returned MISSING (NaN affine, the token above, receipted) and the fit
# continues; only a consumer that can carry an unavailable column requests
# it (the terminal bank provider, which applies the registered bank-level
# threshold).  ``fail_closed`` (default): after the WHOLE fit the typed
# ``CalibrationFailClosedError`` names every missing column, so a consumer
# that cannot carry NaN never receives one.
MISSING_COLUMN_POLICY_CARRY = "carry"
MISSING_COLUMN_POLICY_FAIL_CLOSED = "fail_closed"
MISSING_COLUMN_POLICIES = frozenset(
    {MISSING_COLUMN_POLICY_CARRY, MISSING_COLUMN_POLICY_FAIL_CLOSED}
)


def _strength_token(value: float) -> str:
    return format(float(value), ".12g")


# A CrossFitLayout is a frozen dataclass of tuples, so its validation,
# its context-ID agreement with an immutable ID tuple, its all-rows outer
# membership and its device tensors are pure functions of the object.  The
# terminal bank passes ONE layout and ONE ID tuple to all 125 chunk fits, so
# the layout is validated, its 256k IDs stringified and both tensors built
# from 256k-element tuples once, not in every chunk.  Entries hold strong
# references (so ``id`` keys stay valid) and a cached tensor is reused only
# while its mutation version is unchanged.
_LAYOUT_MEMO_LOCK = threading.Lock()
_LAYOUT_MEMO: "OrderedDict[int, dict[str, Any]]" = OrderedDict()
_LAYOUT_MEMO_ENTRIES = 4
_LAYOUT_MEMO_ID_TUPLES = 4


def _layout_memo_entry(layout: CrossFitLayout) -> dict[str, Any]:
    key = id(layout)
    with _LAYOUT_MEMO_LOCK:
        entry = _LAYOUT_MEMO.get(key)
        if (
            entry is not None
            and entry["layout"] is layout
            and entry["digest"] == layout.content_digest
        ):
            _LAYOUT_MEMO.move_to_end(key)
            return entry
    layout.validate()
    entry = {
        "layout": layout,
        "digest": layout.content_digest,
        "context_ids": [],
        "all_fit_rows": None,
        "tensors": {},
    }
    with _LAYOUT_MEMO_LOCK:
        _LAYOUT_MEMO[key] = entry
        _LAYOUT_MEMO.move_to_end(key)
        while len(_LAYOUT_MEMO) > _LAYOUT_MEMO_ENTRIES:
            _LAYOUT_MEMO.popitem(last=False)
    return entry


def _layout_matches_context_ids(
    entry: dict[str, Any], context_ids: Sequence[Any]
) -> bool:
    layout = entry["layout"]
    with _LAYOUT_MEMO_LOCK:
        if any(known is context_ids for known in entry["context_ids"]):
            return True
    matches = layout.context_ids == tuple(str(value) for value in context_ids)
    if matches and type(context_ids) is tuple:
        with _LAYOUT_MEMO_LOCK:
            known = entry["context_ids"]
            known.append(context_ids)
            del known[:-_LAYOUT_MEMO_ID_TUPLES]
    return matches


def _layout_tensor(entry: dict[str, Any], kind: str, device: Any) -> Any:
    layout = entry["layout"]
    token = (kind, str(torch.device(device)))
    with _LAYOUT_MEMO_LOCK:
        cached = entry["tensors"].get(token)
    if cached is not None and int(cached[0]._version) == cached[1]:
        return cached[0]
    tensor = (
        layout.outer_fit_tensor(device=device)
        if kind == "outer"
        else layout.inner_block_tensor(device=device)
    )
    with _LAYOUT_MEMO_LOCK:
        entry["tensors"][token] = (tensor, int(tensor._version))
    return tensor


def _layout_all_fit_rows(entry: dict[str, Any], expected_membership: Any) -> bool:
    with _LAYOUT_MEMO_LOCK:
        known = entry["all_fit_rows"]
    if known is None:
        known = bool(entry["layout"].fold_count == 1) and bool(
            (expected_membership == 1.0).all().item()
        )
        with _LAYOUT_MEMO_LOCK:
            entry["all_fit_rows"] = known
    return bool(known)


def _v6_gate_record(
    fitted: Any,
    *,
    lane_mask_index: Sequence[int],
    query_count: int,
    stage: str,
) -> dict[str, Any] | None:
    """One core call's device hard-gate outcome, per (mask, query).

    Replaces the round-1 whole-fit fail-closed rule.  Every uncertified
    cell of the call (it failed its device hard gates after the device
    fallback, or is separated without an anchor) is mapped from its lane to
    the ``(mask_index, query_index)`` column it belongs to; for the
    selection and deployment stages the caller marks that column MISSING
    (an identity-selected column needs no deployment cell).  The cell
    counters are the core receipt's own and count cells, not columns; the
    fit's missing COLUMNS are ``missing_components`` of the summary.
    """

    receipt = fitted.receipt
    metadata = dict(receipt.backend_profile.get("metadata", {}))
    missing_cells = []
    for position in v6_missing_cell_positions(receipt):
        lane, query_index = divmod(int(position), int(query_count))
        missing_cells.append(
            {
                "stage": str(stage),
                "lane": int(lane),
                "mask_index": int(lane_mask_index[lane]),
                "query_index": int(query_index),
                "reason": str(receipt.termination_reason[position]),
            }
        )
    fail_closed_cells = []
    for cell in metadata.get("gpu_fallback_fail_closed_cells", ()):
        lane = int(cell["position"]) // int(query_count)
        fail_closed_cells.append(
            {
                **dict(cell),
                "stage": str(stage),
                "lane": lane,
                "mask_index": int(lane_mask_index[lane]),
            }
        )
    by_query = {
        name: [0] * int(query_count)
        for name in (
            "gpu_hard_gate_failed_cell_count",
            "gpu_fallback_rescued_cell_count",
            "gpu_fallback_fail_closed_cell_count",
            "gpu_hard_gate_missing_cell_count",
        )
    }
    for position, reason in enumerate(receipt.termination_reason):
        query_index = int(position) % int(query_count)
        text = str(reason)
        if text.startswith("gpu_fallback_"):
            by_query["gpu_hard_gate_failed_cell_count"][query_index] += 1
            if text.startswith("gpu_fallback_failed_"):
                by_query["gpu_fallback_fail_closed_cell_count"][query_index] += 1
            else:
                by_query["gpu_fallback_rescued_cell_count"][query_index] += 1
    for cell in missing_cells:
        by_query["gpu_hard_gate_missing_cell_count"][int(cell["query_index"])] += 1
    return {
        "stage": str(stage),
        "by_query": by_query,
        "cells": int(len(receipt.converged)),
        "gpu_hard_gate_failed_cell_count": int(
            metadata.get("gpu_hard_gate_failed_cell_count", 0)
        ),
        "gpu_fallback_rescued_cell_count": int(
            metadata.get("gpu_fallback_rescued_cell_count", 0)
        ),
        "gpu_fallback_fail_closed_cell_count": int(
            metadata.get("gpu_fallback_fail_closed_cell_count", 0)
        ),
        "gpu_hard_gate_missing_cell_count": int(len(missing_cells)),
        "gpu_hard_gate_separated_missing_cell_count": int(
            metadata.get("gpu_hard_gate_separated_missing_cell_count", 0)
        ),
        "failed_gate_counts": {
            str(name): int(count)
            for name, count in dict(
                metadata.get("gpu_hard_gate_failed_gate_counts", {})
            ).items()
        },
        "missing_cells": missing_cells,
        "fail_closed_cells": fail_closed_cells,
    }


_V6_GATE_COUNTERS = (
    "cells",
    "gpu_hard_gate_failed_cell_count",
    "gpu_fallback_rescued_cell_count",
    "gpu_fallback_fail_closed_cell_count",
    "gpu_hard_gate_missing_cell_count",
    "gpu_hard_gate_separated_missing_cell_count",
)


def _v6_gate_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    missing: Any,
    missing_stage: Mapping[tuple[int, int], set[str]],
    query_axis_ids: Sequence[str],
) -> dict[str, Any]:
    """The proper-score receipt's v6 block (persisted gate counters)."""

    by_stage: dict[str, dict[str, Any]] = {}
    by_stage_by_query: dict[str, dict[str, list[int]]] = {}
    for record in records:
        stage = by_stage.setdefault(
            str(record["stage"]),
            {
                "calls": 0,
                **{name: 0 for name in _V6_GATE_COUNTERS},
                "failed_gate_counts": {},
            },
        )
        stage["calls"] += 1
        for name in _V6_GATE_COUNTERS:
            stage[name] += int(record[name])
        for gate, count in dict(record.get("failed_gate_counts", {})).items():
            stage["failed_gate_counts"][str(gate)] = int(
                stage["failed_gate_counts"].get(str(gate), 0)
            ) + int(count)
        per_query = by_stage_by_query.setdefault(
            str(record["stage"]),
            {name: [0] * len(query_axis_ids) for name in record["by_query"]},
        )
        for name, values in record["by_query"].items():
            for query_index, value in enumerate(values):
                per_query[name][query_index] += int(value)
    totals = {
        name: int(sum(int(record[name]) for record in records))
        for name in _V6_GATE_COUNTERS
    }
    reasons: dict[tuple[int, int], list[str]] = {}
    for record in records:
        if str(record["stage"]) == "deployment_unregularized":
            # Informational stage: its uncertified cells fall back to their
            # anchored parameters (the registered rule), never missing.
            continue
        for cell in record["missing_cells"]:
            reasons.setdefault(
                (int(cell["mask_index"]), int(cell["query_index"])), []
            ).append(f"{cell['stage']}:{cell['reason']}")
    missing_components = []
    missing_flags = missing.tolist()
    for mask_index, row in enumerate(missing_flags):
        for query_index, is_missing in enumerate(row):
            if not is_missing:
                continue
            key = (mask_index, query_index)
            missing_components.append(
                {
                    "mask_index": int(mask_index),
                    "query_index": int(query_index),
                    "query_axis_id": str(query_axis_ids[query_index]),
                    "stages": sorted(missing_stage.get(key, set())),
                    "reasons": sorted(set(reasons.get(key, ()))),
                }
            )
    missing_ids: list[str] = []
    for component in missing_components:
        if component["query_axis_id"] not in missing_ids:
            missing_ids.append(component["query_axis_id"])
    return {
        "policy": GPU_HARD_GATES_AUDIT_POLICY,
        "missing_cell_rule": V6_MISSING_CELL_RULE,
        **totals,
        "by_stage": by_stage,
        "by_stage_by_query": by_stage_by_query,
        "missing_component_count": int(len(missing_components)),
        "missing_components": missing_components,
        "missing_query_axis_ids": missing_ids,
        "missing_cells": [
            cell for record in records for cell in record["missing_cells"]
        ],
        "fail_closed_cells": [
            cell for record in records for cell in record["fail_closed_cells"]
        ],
    }


def _receipt_safe_losses(values: Sequence[Any]) -> list[Any]:
    """A missing row's losses: finite floats, non-finite recorded as None."""

    safe = []
    for value in values:
        if isinstance(value, (list, tuple)):
            safe.append(_receipt_safe_losses(value))
        else:
            number = float(value)
            safe.append(number if math.isfinite(number) else None)
    return safe


@dataclass(frozen=True)
class ProperScoreShrinkageReceipt:
    dependency_hash: str
    factorization_hash: str
    algorithm_id: str
    candidate_strengths: tuple[str, ...]
    selected_strengths: tuple[str, ...]
    component_rows: tuple[Mapping[str, Any], ...]
    block_assignment_hash: str
    block_count: int
    selection_rule: str
    fit_receipt: Mapping[str, Any]
    backend_profile: Mapping[str, Any]

    def validate(self) -> None:
        if len(self.dependency_hash) != 64 or len(self.factorization_hash) != 64:
            raise ValueError("proper-score calibration hashes are invalid")
        if self.algorithm_id == "firth_positive_affine_v1":
            if self.candidate_strengths or self.block_count != 0 or self.block_assignment_hash:
                raise ValueError("Firth calibration must not contain selection blocks or strengths")
            if self.selection_rule != "single_firth_fit" or not self.selected_strengths:
                raise ValueError("Firth calibration selection metadata is invalid")
            if len(self.component_rows) != len(self.selected_strengths) or any(
                token != "firth" or row.get("selected_strength") != "firth"
                or not math.isfinite(float(row.get("effective_sample_size", 0.0)))
                or float(row.get("effective_sample_size", 0.0)) <= 0.0
                for token, row in zip(self.selected_strengths, self.component_rows)
            ):
                raise ValueError("Firth calibration component metadata is invalid")
            required_flag = self.fit_receipt.get("required_passed")
            if not (bool(self.fit_receipt.get("passed")) if required_flag is None else bool(required_flag)):
                raise ValueError("Firth calibration final fit did not pass")
            return
        if self.algorithm_id != PROPER_SCORE_SHRINKAGE_AFFINE_V2:
            raise ValueError("proper-score calibration algorithm ID is invalid")
        if len(self.block_assignment_hash) != 64 or self.block_count < 2:
            raise ValueError("proper-score calibration blocks are invalid")
        if (
            not self.candidate_strengths
            or self.candidate_strengths[-1] != IDENTITY_CANDIDATE
        ):
            raise ValueError("proper-score calibration must include identity last")
        if len(self.component_rows) != len(self.selected_strengths):
            raise ValueError("proper-score component and selection axes differ")
        if not self.selected_strengths or any(
            value not in self.candidate_strengths
            and not (
                value == MISSING_SELECTION_TOKEN
                and isinstance(row, Mapping)
                and row.get("missing") is not None
            )
            for value, row in zip(
                self.selected_strengths, self.component_rows, strict=True
            )
        ):
            raise ValueError("proper-score selected strength is outside the grid")
        for row, selected in zip(
            self.component_rows, self.selected_strengths, strict=True
        ):
            if str(row.get("selected_strength")) != selected:
                raise ValueError("proper-score component selection is inconsistent")
            # A MISSING row records its losses with non-finite
            # values as None; every other row stays strictly finite.
            missing_row = row.get("missing") is not None
            if missing_row != (selected == MISSING_SELECTION_TOKEN):
                raise ValueError("proper-score missing component is inconsistent")
            row_block_count = int(row.get("block_count", self.block_count))

            def _loss(value: Any) -> float:
                if value is None and missing_row:
                    return 0.0
                return float(value)

            means = tuple(_loss(value) for value in row.get("mean_losses", ()))
            errors = tuple(_loss(value) for value in row.get("standard_errors", ()))
            losses = tuple(
                tuple(_loss(value) for value in fold)
                for fold in row.get("fold_losses", ())
            )
            if (
                row_block_count < 2
                or row_block_count > self.block_count
                or len(means) != len(self.candidate_strengths)
                or len(errors) != len(means)
                or len(losses) != len(means)
                or any(len(fold) != row_block_count for fold in losses)
                or any(
                    not math.isfinite(value)
                    for value in (
                        *means,
                        *errors,
                        *(v for fold in losses for v in fold),
                    )
                )
            ):
                raise ValueError("proper-score selection losses are invalid")
        required_flag = self.fit_receipt.get("required_passed")
        fit_ok = (
            bool(required_flag)
            if required_flag is not None
            else bool(self.fit_receipt.get("passed", False))
        )
        if not fit_ok:
            raise ValueError("proper-score final fit did not pass")

    @property
    def passed(self) -> bool:
        self.validate()
        return True

    @property
    def content_hash(self) -> str:
        return stable_digest(scientific_fit_receipt_payload(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_name": "mechanistic_proper_score_shrinkage_receipt",
            **({"loss_weighting": "per_image"} if self.backend_profile.get("metadata", {}).get("loss_weighting") == "per_image" else {}),
            "schema_version": "1",
            "dependency_hash": self.dependency_hash,
            "factorization_hash": self.factorization_hash,
            "algorithm_id": self.algorithm_id,
            "candidate_strengths": list(self.candidate_strengths),
            "selected_strengths": list(self.selected_strengths),
            "component_rows": [dict(row) for row in self.component_rows],
            "block_assignment_hash": self.block_assignment_hash,
            "block_count": int(self.block_count),
            "selection_rule": self.selection_rule,
            "fit_receipt": dict(self.fit_receipt),
            "backend_profile": dict(self.backend_profile),
            "passed": True,
            **({"calibration_penalty": "firth", "effective_sample_size": [
                float(row["effective_sample_size"]) for row in self.component_rows]}
               if self.algorithm_id == "firth_positive_affine_v1" else {}),
        }


@dataclass(frozen=True)
class ProperScoreShrinkageFit:
    scale: Any
    offset: Any
    factorization: CalibrationFactorization
    receipt: ProperScoreShrinkageReceipt
    input_was_batched: bool
    family: str = "proper_score_identity_shrinkage_affine_v2"

    @property
    def backend(self) -> str:
        if self.receipt.algorithm_id == "firth_positive_affine_v1":
            return "batched_float64_firth_fisher_scoring"
        return "batched_projected_newton_predictive_kl"

    @property
    def calibration_backend(self) -> str:
        return self.backend

    @property
    def readout_scale(self) -> Any:
        return self.scale

    @property
    def readout_offset(self) -> Any:
        return self.offset

    def apply(self, values: Any) -> Any:
        tensor = torch.as_tensor(values)
        tensor = tensor.to(dtype=torch.promote_types(tensor.dtype, self.scale.dtype))
        if tensor.ndim == 2:
            return tensor * self.scale[0].to(tensor).unsqueeze(0) + self.offset[0].to(
                tensor
            ).unsqueeze(0)
        if tensor.ndim == 3:
            return tensor * self.scale.to(tensor).unsqueeze(1) + self.offset.to(
                tensor
            ).unsqueeze(1)
        raise ValueError(
            "proper-score calibration values must be [sample,query] or "
            "[mask,sample,query]"
        )


@dataclass(frozen=True)
class CalibrationSelectionFaceComponent:
    mask_index: int; best_token: str; selected_token: str; best_threshold: float
    mean_losses: tuple[float, ...]; standard_errors: tuple[float, ...]; fold_losses: tuple[tuple[float, ...], ...]
    eligible_tokens: tuple[str, ...]; unique_best_margins: tuple[tuple[str, float], ...]
    eligibility_margins: tuple[tuple[str, float], ...]
    raw_face_margin: float; minimizing_member: str

    def validate(self, tokens: tuple[str, ...], tau: float, block_count: int) -> None:
        count = len(tokens)
        if not (len(self.mean_losses) == len(self.standard_errors) == len(self.fold_losses) == count): raise ValueError("calibration selection candidate axes are misaligned")
        if any(len(row) != block_count for row in self.fold_losses): raise ValueError("calibration selection fold axes are misaligned")
        means = tuple(math.fsum(row) / len(row) for row in self.fold_losses)
        errors = tuple(math.sqrt(math.fsum((value - mean) ** 2 for value in row) / (len(row) - 1) / len(row)) for row, mean in zip(self.fold_losses, means, strict=True))
        if any(abs(left - right) > tau for left, right in zip((*means, *errors), (*self.mean_losses, *self.standard_errors), strict=True)): raise ValueError("calibration selection reductions differ from fold losses")
        best = min(range(count), key=lambda index: (self.mean_losses[index], index))
        threshold = self.mean_losses[best] + self.standard_errors[best]
        eligible = tuple(tokens[index] for index in range(count) if threshold - self.mean_losses[index] >= 0.0)
        selected = max((index for index in range(count) if tokens[index] in eligible), default=-1)
        if self.best_token != tokens[best] or self.selected_token != tokens[selected] or self.eligible_tokens != eligible or abs(self.best_threshold - threshold) > tau: raise ValueError("calibration selection branch does not match its reductions")
        unique = tuple((tokens[index], self.mean_losses[index] - self.mean_losses[best]) for index in range(count) if index != best)
        eligibility = tuple((tokens[index], threshold - self.mean_losses[index]) for index in range(count))
        if self.unique_best_margins != unique or self.eligibility_margins != eligibility: raise ValueError("calibration selection margins differ from their reductions")
        raw = tuple((f"unique_best:{token}", value) for token, value in unique) + tuple((f"{'eligible' if token in eligible else 'ineligible'}:{token}", value if token in eligible else -value) for token, value in eligibility)
        minimizing = min(raw, key=lambda value: (value[1], value[0]))
        if minimizing != (self.minimizing_member, self.raw_face_margin) or self.raw_face_margin <= tau: raise ValueError("CALIBRATION_SELECTION_FACE_UNRESOLVED")


@dataclass(frozen=True)
class CalibrationSelectionFaceReceipt:
    revision_identity: str; selection_owner_identity: str; factorization_hash: str
    logits_hash: str; targets_hash: str
    candidate_tokens: tuple[str, ...]
    block_assignment_hash: str; context_ids_hash: str; row_measure_hash: str
    block_count: int; tau_strength: float; tie_rule: str; reduction_policy: str
    components: tuple[CalibrationSelectionFaceComponent, ...]
    selected_fit_hash: str

    def validate(self) -> None:
        hash_names = ("revision_identity", "selection_owner_identity", "factorization_hash", "logits_hash", "targets_hash", "block_assignment_hash", "context_ids_hash", "row_measure_hash", "selected_fit_hash")
        if any(len(getattr(self, name)) != 64 or any(character not in "0123456789abcdef" for character in getattr(self, name)) for name in hash_names) or not self.components: raise ValueError("calibration selection-face identity is invalid")
        if self.block_count < 2 or not math.isfinite(self.tau_strength) or self.tau_strength <= 0.0: raise ValueError("calibration selection-face tolerance is invalid")
        finite = self.candidate_tokens[:-1]
        if not finite or self.candidate_tokens[-1] != IDENTITY_CANDIDATE or tuple(sorted(set(float(value) for value in finite))) != tuple(float(value) for value in finite) or self.tie_rule != "token_order_argmin_then_maximum_complexity_eligible_v1" or self.reduction_policy != "float64_mean_and_unbiased_se_over_stratified_blocks_v1": raise ValueError("calibration selection-face contract is invalid")
        for value in self.components: value.validate(self.candidate_tokens, self.tau_strength, self.block_count)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "components"}
        return {"schema_name": "mechanistic_calibration_selection_face_receipt_v1", **payload, "components": [{name: getattr(component, name) for name in component.__dataclass_fields__} for component in self.components]}

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())


@dataclass(frozen=True)
class ProperScoreShrinkageOwnedFit:
    fit: ProperScoreShrinkageFit; selection_face: CalibrationSelectionFaceReceipt
    positive_affine_owner: PositiveAffineDifferentialOwnerReceipt | None


def deterministic_stratified_calibration_blocks(
    binary_targets: Any,
    context_ids: Sequence[Any],
    *,
    requested_blocks: int = 4,
) -> tuple[Any, str]:
    """Assign stable class-balanced inner blocks without outcome inspection."""
    layout = CrossFitLayout.from_binary_targets(
        binary_targets,
        context_ids,
        requested_blocks=requested_blocks,
    )
    return layout.inner_block_tensor(device="cpu")[0], layout.block_assignment_hash


def deterministic_fold_local_calibration_blocks(
    binary_targets: Any,
    context_ids: Sequence[Any],
    *,
    fold_validity: Any,
    requested_blocks: int = 4,
) -> tuple[Any, tuple[int, ...], str]:
    """Compile each outer fold's inner selection blocks on its own fit rows.

    A fold-validity mask cannot be applied *after* a global inner-block split:
    global stratification need not remain class-balanced after restriction to an
    outer calibration fold.  This helper is the batched equivalent of
    per-fold subsetting.  It preserves per-fold subsetting's deterministic
    local block IDs, while returning a padded [outer_fold, sample] assignment
    suitable for one vectorized solve per inner block.
    """

    layout = CrossFitLayout.from_binary_targets(
        binary_targets,
        context_ids,
        fold_validity=fold_validity,
        requested_blocks=requested_blocks,
    )
    return (
        layout.inner_block_tensor(device="cpu"),
        layout.inner_block_counts,
        layout.block_assignment_hash,
    )


def _materialize_selection_rows(
    *,
    tokens: tuple[str, ...],
    selected_flat: Any,
    best_flat: Any,
    means_cpu: Any,
    errors_cpu: Any,
    fold_losses_cpu: Any,
    eligible_cpu: Any,
    block_counts: Sequence[int],
    block_count: int,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Selection receipt rows via one host transfer per tensor.

    Byte-identical to per-cell slicing (same floats, same ordering, same
    optional ragged block_count field).
    """

    mask_count, _, query_count = map(int, means_cpu.shape)
    selected_lists = selected_flat.tolist()
    best_lists = best_flat.tolist()
    means_lists = means_cpu.permute(0, 2, 1).contiguous().tolist()
    errors_lists = errors_cpu.permute(0, 2, 1).contiguous().tolist()
    fold_lists = fold_losses_cpu.permute(0, 3, 1, 2).contiguous().tolist()
    eligible_lists = eligible_cpu.permute(0, 2, 1).contiguous().tolist()
    selected_tokens = tuple(
        tokens[int(value)] for row in selected_lists for value in row
    )
    component_rows: list[dict[str, Any]] = []
    for mask_index in range(mask_count):
        local_block_count = int(block_counts[mask_index])
        for query_index in range(query_count):
            row = {
                "mask_index": mask_index,
                "query_index": query_index,
                "selected_strength": tokens[
                    int(selected_lists[mask_index][query_index])
                ],
                "best_strength": tokens[
                    int(best_lists[mask_index][query_index])
                ],
                "mean_losses": means_lists[mask_index][query_index],
                "standard_errors": errors_lists[mask_index][query_index],
                "fold_losses": [
                    candidate[:local_block_count]
                    for candidate in fold_lists[mask_index][query_index]
                ],
                "eligible_strengths": [
                    tokens[index]
                    for index, value in enumerate(
                        eligible_lists[mask_index][query_index]
                    )
                    if bool(value)
                ],
            }
            if local_block_count != block_count:
                row["block_count"] = local_block_count
            component_rows.append(row)
    return selected_tokens, component_rows


def _class_balanced_binary_loss(
    logits: Any, targets: Any, *, row_validity: Any | None = None, sample_weights: Any | None = None
) -> Any:
    y = torch.as_tensor(targets, device=logits.device, dtype=torch.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if loss_weighting() == "per_image":
        valid = (torch.ones(logits.shape[:2], device=logits.device, dtype=torch.float64)
                 if row_validity is None else torch.as_tensor(row_validity, device=logits.device, dtype=torch.float64))
        if tuple(valid.shape) != tuple(logits.shape[:2]):
            raise ValueError("selection-loss row validity differs from logits axes")
        weights = valid.unsqueeze(-1)
        if sample_weights is not None:
            supplied = torch.as_tensor(sample_weights, device=logits.device, dtype=torch.float64).reshape(1, -1, 1)
            weights = weights * supplied
        rows = F.softplus(logits.to(torch.float64)) - y.unsqueeze(0) * logits.to(torch.float64)
        return (rows * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-300)
    if row_validity is not None:
        # Masked reduction: class balance and the mean are computed within
        # the valid rows of each leading-axis group (e.g. one crossfit fold
        # per group), so zero-validity rows contribute nothing.
        valid = torch.as_tensor(
            row_validity, device=logits.device, dtype=torch.float64
        )
        if valid.ndim != 2 or tuple(valid.shape) != tuple(logits.shape[:2]):
            raise ValueError("selection-loss row validity differs from logits axes")
        membership = valid.unsqueeze(2)
        positives = (membership * y.unsqueeze(0)).sum(dim=1)
        negatives = membership.sum(dim=1) - positives
        if bool(((positives <= 0.0) | (negatives <= 0.0)).any().item()):
            raise ValueError("held-out calibration selection requires both classes")
        weights = torch.where(
            y.unsqueeze(0) > 0.5,
            (negatives / positives).unsqueeze(1),
            torch.ones_like(y.unsqueeze(0).expand(int(valid.shape[0]), -1, -1)),
        ) * membership
        rows = F.softplus(logits.to(torch.float64)) - y.unsqueeze(0) * logits.to(
            torch.float64
        )
        return (rows * weights).sum(dim=1) / membership.sum(dim=1).clamp_min(1e-30)
    positives = y.sum(dim=0)
    negatives = float(y.shape[0]) - positives
    if bool(((positives <= 0.0) | (negatives <= 0.0)).any().item()):
        raise ValueError("held-out calibration selection requires both classes")
    weights = torch.where(
        y > 0.5,
        (negatives / positives).reshape(1, -1),
        torch.ones_like(y),
    )
    rows = F.softplus(logits.to(torch.float64)) - y.unsqueeze(0) * logits.to(
        torch.float64
    )
    return (rows * weights.unsqueeze(0)).mean(dim=1)


def _selection_face(
    *, factorization: CalibrationFactorization, tokens: tuple[str, ...],
    means: Any, errors: Any, best_index: Any, selected_index: Any,
    eligible: Any, fold_losses: Any, block_hash: str, context_ids: Sequence[Any],
    values: Any, targets: Any, weighting: str, fit: ProperScoreShrinkageFit,
) -> CalibrationSelectionFaceReceipt:
    scale = max(1.0, float(means.abs().max().item()), float(errors.abs().max().item()))
    tau = max(1e-12, 64.0 * float(torch.finfo(torch.float64).eps) * scale)
    components = []
    for mask_index in range(int(means.shape[0])):
        best = int(best_index[mask_index].item())
        selected = int(selected_index[mask_index].item())
        threshold = float((means[mask_index, best] + errors[mask_index, best]).item())
        unique = tuple(
            (tokens[index], float((means[mask_index, index] - means[mask_index, best]).item()))
            for index in range(len(tokens)) if index != best
        )
        signed_eligibility = tuple(
            (tokens[index], threshold - float(means[mask_index, index].item()))
            for index in range(len(tokens))
        )
        raw_members = [
            (f"unique_best:{token}", value) for token, value in unique
        ] + [
            (
                f"{'eligible' if bool(eligible[mask_index, index].item()) else 'ineligible'}:{token}",
                value if bool(eligible[mask_index, index].item()) else -value,
            )
            for index, (token, value) in enumerate(signed_eligibility)
        ]
        minimizing_member, raw_margin = min(raw_members, key=lambda value: (value[1], value[0]))
        components.append(CalibrationSelectionFaceComponent(
            mask_index=mask_index, best_token=tokens[best], selected_token=tokens[selected],
            best_threshold=threshold,
            mean_losses=tuple(float(value) for value in means[mask_index].tolist()),
            standard_errors=tuple(float(value) for value in errors[mask_index].tolist()),
            fold_losses=tuple(tuple(float(value) for value in row) for row in fold_losses[mask_index].tolist()),
            eligible_tokens=tuple(tokens[index] for index in range(len(tokens)) if bool(eligible[mask_index, index].item())),
            unique_best_margins=unique, eligibility_margins=signed_eligibility,
            raw_face_margin=float(raw_margin), minimizing_member=minimizing_member,
        ))
    fit_payload = scientific_fit_receipt_payload(fit.receipt.to_dict())
    result = CalibrationSelectionFaceReceipt(
        revision_identity=stable_digest("scalar_v2r_revision4_strength_selection_face"),
        selection_owner_identity=stable_digest("proper_score_shrinkage.selection_owner.v1"),
        factorization_hash=factorization.factorization_hash,
        logits_hash=calibration_tensor_content_hash(values),
        targets_hash=calibration_tensor_content_hash(targets),
        candidate_tokens=tokens, block_assignment_hash=block_hash,
        context_ids_hash=stable_digest(tuple(str(value) for value in context_ids)),
        row_measure_hash=stable_digest({"targets": torch.as_tensor(targets, device="cpu", dtype=torch.float64).tolist(), "weighting": weighting, "selection_loss": "per_image_binary_bce" if loss_weighting() == "per_image" else "class_balanced_binary_bce"}),
        block_count=int(fold_losses.shape[2]), tau_strength=tau,
        tie_rule="token_order_argmin_then_maximum_complexity_eligible_v1",
        reduction_policy="float64_mean_and_unbiased_se_over_stratified_blocks_v1",
        components=tuple(components), selected_fit_hash=stable_digest(fit_payload),
    )
    result.validate()
    return result


def _fit_proper_score_shrinkage_affine_core(
    values: Any,
    binary_targets: Any,
    *,
    factorization: CalibrationFactorization,
    context_ids: Sequence[Any],
    weighting: str = "pos_weighted",
    sample_weights: Any | None = None,
    calibration_penalty: str = "identity_anchor",
    class_ids: Any | None = None,
    fixed_offsets: Any | None = None,
    firth_convergence: str = "absolute",
    metric_loss_weighting: str | None = None,
    minimum_scale: float = 1e-8,
    initial_scale: Any | None = None,
    initial_offset: Any | None = None,
    candidate_strengths: Sequence[float] = DEFAULT_IDENTITY_ANCHOR_STRENGTHS,
    selection_block_count: int = CANONICAL_CALIBRATION_SELECTION_BLOCKS,
    max_iterations: int = CANONICAL_CALIBRATION_MAX_ITERATIONS,
    tolerance: float = CANONICAL_CALIBRATION_KKT_TOLERANCE,
    hessian_ridge: float = CANONICAL_CALIBRATION_HESSIAN_RIDGE,
    objective_tolerance: float = CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE,
    reference_tolerance: float = CANONICAL_CALIBRATION_REFERENCE_TOLERANCE,
    statistics_backend: str = "auto",
    source_logit_dtype: Any | None = None,
    anchor_probability_epsilon: float = (
        CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON
    ),
    fold_validity: Any | None = None,
    required_query_mask: Any | None = None,
    reference_audit_policy: str = GPU_HARD_GATES_AUDIT_POLICY,
    audit_seed: int | None = None,
    deployment_refit_anchor: str = "selected",
    missing_column_policy: str = MISSING_COLUMN_POLICY_FAIL_CLOSED,
    _emit_differential_owner: bool = False,
    _crossfit_layout: CrossFitLayout | None = None,
) -> tuple[
    ProperScoreShrinkageFit,
    CalibrationSelectionFaceReceipt | None,
    PositiveAffineDifferentialOwnerReceipt | None,
]:
    """Select and fit identity-anchored positive-affine calibration.

    Two batching axes are available: the query axis carries K independent
    one-vs-rest columns of a shared readout, and ``fold_validity`` (a binary
    ``[fold, sample]`` membership matrix) multiplexes crossfit folds onto the
    mask axis — every fold's selection and final refit run in the same batched
    solves, with zero-validity rows contributing nothing (the row-validity
    mechanism). Owner-receipt emission remains single-column, single-fold.
    """

    if torch is None or F is None:
        raise RuntimeError("proper-score calibration requires PyTorch")
    if calibration_penalty not in {"identity_anchor", "firth"}:
        raise ValueError(f"unsupported calibration penalty {calibration_penalty!r}")
    if calibration_penalty == "firth":
        if _emit_differential_owner:
            raise ValueError("Firth calibration does not emit the predictive-KL selection differential owner")
        final_fit = fit_batched_post_query_affine(
            values, binary_targets, factorization=factorization,
            weighting="design_weighted", sample_weights=sample_weights, class_ids=class_ids,
            fixed_offsets=fixed_offsets, calibration_penalty="firth", row_validity=fold_validity,
            firth_convergence=firth_convergence,
            metric_loss_weighting=metric_loss_weighting,
            minimum_scale=minimum_scale, initial_scale=initial_scale, initial_offset=initial_offset,
            max_iterations=max_iterations, tolerance=tolerance, hessian_ridge=hessian_ridge,
            objective_tolerance=objective_tolerance, reference_tolerance=reference_tolerance,
            statistics_backend=statistics_backend, require_convergence=True,
            source_logit_dtype=source_logit_dtype, required_query_mask=required_query_mask)
        masks, queries = map(int, final_fit.scale.shape)
        nested = final_fit.receipt.to_dict()
        sizes = tuple(float(value) for value in nested["effective_sample_size"])
        rows = tuple({"mask_index": mask, "query_index": query,
                      "selected_strength": "firth",
                      "effective_sample_size": sizes[mask * queries + query]}
                     for mask in range(masks) for query in range(queries))
        receipt = ProperScoreShrinkageReceipt(
            dependency_hash=factorization.dependency_hash,
            factorization_hash=factorization.factorization_hash,
            algorithm_id="firth_positive_affine_v1", candidate_strengths=(),
            selected_strengths=("firth",) * (masks * queries), component_rows=rows,
            block_assignment_hash="", block_count=0, selection_rule="single_firth_fit",
            fit_receipt=nested, backend_profile=final_fit.receipt.backend_profile)
        receipt.validate()
        return ProperScoreShrinkageFit(final_fit.scale, final_fit.offset, factorization,
            receipt, final_fit.input_was_batched, family="firth_positive_affine_v1"), None, None
    if reference_audit_policy not in {"full", "required_plus_sampled", GPU_HARD_GATES_AUDIT_POLICY}:
        raise ValueError(f"unsupported reference audit policy {reference_audit_policy!r}")
    reference_audit_policy = GPU_HARD_GATES_AUDIT_POLICY
    if str(missing_column_policy) not in MISSING_COLUMN_POLICIES:
        raise ValueError(
            f"unsupported missing-column policy {missing_column_policy!r}"
        )
    started = time.perf_counter()
    x = torch.as_tensor(values)
    input_was_batched = x.ndim == 3
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(
            "proper-score calibration values must have shape "
            "[sample, query] or [mask, sample, query]"
        )
    if not x.is_floating_point():
        x = x.to(torch.float32)
    y = torch.as_tensor(binary_targets, device=x.device, dtype=torch.float64)
    query_count = int(x.shape[2])
    if tuple(y.shape) != (int(x.shape[1]), query_count):
        raise ValueError("proper-score targets differ from the sample/query axes")
    fold_membership = None
    if fold_validity is not None:
        fold_membership = torch.as_tensor(
            fold_validity, device=x.device, dtype=torch.float64
        )
        if fold_membership.ndim != 2 or int(fold_membership.shape[1]) != int(
            x.shape[1]
        ):
            raise ValueError("fold validity must have shape [fold, sample]")
        if int(x.shape[0]) == 1:
            x = x.expand(int(fold_membership.shape[0]), -1, -1)
        elif int(x.shape[0]) != int(fold_membership.shape[0]):
            raise ValueError("fold validity differs from the batched mask axis")
    if _emit_differential_owner and (query_count != 1 or fold_membership is not None):
        raise ValueError(
            "differential-owner emission is single-column, single-fold only"
        )
    strengths = tuple(float(value) for value in candidate_strengths)
    if (
        not strengths
        or tuple(sorted(set(strengths))) != strengths
        or any(not math.isfinite(value) or value <= 0.0 for value in strengths)
    ):
        raise ValueError(
            "proper-score anchor strengths must be positive, finite, unique, and increasing"
        )
    mask_count = int(x.shape[0])
    if _crossfit_layout is not None:
        # Validation, the ID check and the tensors run once per
        # layout object (``_layout_memo_entry``), not once per chunk.
        layout_entry = _layout_memo_entry(_crossfit_layout)
        if not _layout_matches_context_ids(layout_entry, context_ids):
            raise ValueError("cross-fit layout context IDs differ from calibration inputs")
        if _crossfit_layout.sample_count != int(x.shape[1]):
            raise ValueError("cross-fit layout sample axis differs from calibration inputs")
        expected_membership = _layout_tensor(layout_entry, "outer", x.device)
        if fold_membership is None:
            if not _layout_all_fit_rows(layout_entry, expected_membership):
                raise ValueError("cross-fit layout requires declared outer fold validity")
        elif not bool(torch.equal(expected_membership, fold_membership)):
            raise ValueError("cross-fit layout outer membership differs from calibration inputs")
        block_ids = _layout_tensor(layout_entry, "inner", x.device)
        if fold_membership is None:
            block_ids = block_ids[0]
            block_counts = (_crossfit_layout.inner_block_counts[0],) * mask_count
        else:
            block_counts = _crossfit_layout.inner_block_counts
        block_hash = _crossfit_layout.block_assignment_hash
    elif fold_membership is None:
        block_ids, block_hash = deterministic_stratified_calibration_blocks(
            y,
            context_ids,
            requested_blocks=selection_block_count,
        )
        block_ids = block_ids.to(device=x.device)
        block_counts = (int(block_ids.max().item()) + 1,) * mask_count
    else:
        block_ids, block_counts, block_hash = (
            deterministic_fold_local_calibration_blocks(
                y,
                context_ids,
                fold_validity=fold_membership,
                requested_blocks=selection_block_count,
            )
        )
        block_ids = block_ids.to(device=x.device)
    block_count = max(block_counts)
    candidate_count = len(strengths) + 1
    strength_count = len(strengths)
    fold_losses = torch.full(
        (mask_count, candidate_count, block_count, query_count),
        float("nan"),
        device=x.device,
        dtype=torch.float64,
    )
    finite_strengths = torch.tensor(strengths, device=x.device, dtype=torch.float64)
    required_bool = None
    if required_query_mask is not None:
        required_bool = torch.as_tensor(
            required_query_mask, dtype=torch.bool
        ).reshape(-1)
        if int(required_bool.shape[0]) != query_count:
            raise ValueError("required query mask differs from the query axis")
    # Per-column missing semantics instead of whole-fit fail-closed.
    gate_records: list[dict[str, Any]] = []
    selection_missing_host = [[False] * query_count for _ in range(mask_count)]
    missing_stage: dict[tuple[int, int], set[str]] = {}
    last_gated_fit: list[Any] = []

    def _record_gates(fitted_call: Any, lane_mask_index: Sequence[int], stage: str) -> None:
        last_gated_fit[:] = [fitted_call]
        record = _v6_gate_record(
            fitted_call,
            lane_mask_index=lane_mask_index,
            query_count=query_count,
            stage=stage,
        )
        if record is None:
            return
        gate_records.append(record)
        if stage == "selection":
            for cell in record["missing_cells"]:
                key = (int(cell["mask_index"]), int(cell["query_index"]))
                selection_missing_host[key[0]][key[1]] = True
                missing_stage.setdefault(key, set()).add(stage)

    if loss_weighting() == "per_image" and weighting == "pos_weighted":
        weighting = "design_weighted" if sample_weights is not None else "unweighted"
    inner_fit_options = dict(
        factorization=factorization,
        weighting=weighting,
        minimum_scale=minimum_scale,
        initial_scale=initial_scale,
        initial_offset=initial_offset,
        max_iterations=max_iterations,
        tolerance=tolerance,
        hessian_ridge=hessian_ridge,
        objective_tolerance=objective_tolerance,
        reference_tolerance=reference_tolerance,
        statistics_backend=statistics_backend,
        require_convergence=True,
        source_logit_dtype=source_logit_dtype or x.dtype,
        anchor_probability_epsilon=anchor_probability_epsilon,
        required_query_mask=required_query_mask,
        reference_audit_policy=reference_audit_policy,
        audit_seed=audit_seed,
    )
    source_weights = None
    if sample_weights is not None:
        source_weights = torch.as_tensor(sample_weights, device=x.device).reshape(-1)
        if int(source_weights.numel()) != int(x.shape[1]):
            raise ValueError("proper-score sample weights differ from samples")

    # Calibration-residual reduction:
    # the calibration solve is launch-bound (the
    # Newton loop converges in 2-4 iters and dispatch/sync dominate FLOP by
    # 18-18000x). For the uniform-block crossfit case (the production path),
    # the ``block_count`` selection solves share one ``x`` and differ only by
    # the ``(block_ids == block)`` row-validity mask, so stack
    # block x fold x strength onto the mask axis and run ONE Newton solve
    # instead of ``block_count`` serial solves. Per-lane bit-identical by the
    # Masked-no-op invariant (each lane's fit depends only on its own
    # row_validity); collapses the selection-phase dispatch ~block_count x.
    # Both the ordinary one-fit path and uniform outer-fold crossfit are
    # rectangular when every lane has the same number of deterministic inner
    # blocks.  Express held-out subsetting as row-validity masks and solve the
    # complete block x lane x strength rectangle in one backend call.  This
    # replaces eight serial selection launches without changing
    # which rows belong to any fit or held-out loss.
    rectangular_selection = len(set(int(c) for c in block_counts)) == 1
    if rectangular_selection:
        sample_axis = int(x.shape[1])
        lanes = block_count * mask_count * strength_count
        blocks = torch.arange(block_count, device=x.device)
        lane_membership = (
            torch.ones(
                (mask_count, sample_axis),
                device=x.device,
                dtype=torch.float64,
            )
            if fold_membership is None
            else fold_membership
        )
        lane_block_ids = (
            block_ids.unsqueeze(0).expand(mask_count, -1)
            if block_ids.ndim == 1
            else block_ids
        )
        not_held = (
            lane_block_ids.unsqueeze(0)
            != blocks.view(block_count, 1, 1)
        )
        train_validity = (
            lane_membership.unsqueeze(0) * not_held.to(torch.float64)
        )
        heldout_validity = (
            lane_membership.unsqueeze(0) * (~not_held).to(torch.float64)
        )
        expanded = (
            x.unsqueeze(0)
            .unsqueeze(2)
            .expand(block_count, mask_count, strength_count, -1, -1)
            .reshape(lanes, sample_axis, query_count)
        )
        expanded_validity = (
            train_validity.unsqueeze(2)
            .expand(block_count, mask_count, strength_count, -1)
            .reshape(lanes, sample_axis)
        )
        strength_grid = (
            finite_strengths.view(1, 1, strength_count, 1)
            .expand(block_count, mask_count, strength_count, query_count)
            .reshape(lanes, query_count)
        )
        fitted = fit_batched_post_query_affine(
            expanded,
            y,
            sample_weights=source_weights,
            row_validity=expanded_validity,
            identity_anchor_strength=strength_grid,
            **inner_fit_options,
        )
        _record_gates(
            fitted,
            tuple(
                (lane // strength_count) % mask_count for lane in range(lanes)
            ),
            "selection",
        )
        heldout_expanded_validity = (
            heldout_validity.unsqueeze(2)
            .expand(block_count, mask_count, strength_count, -1)
            .reshape(lanes, sample_axis)
        )
        finite_loss = _class_balanced_binary_loss(
            fitted.apply(expanded),
            y,
            row_validity=heldout_expanded_validity, sample_weights=source_weights,
        ).reshape(block_count, mask_count, strength_count, query_count)
        identity_x = (
            x.unsqueeze(0)
            .expand(block_count, mask_count, -1, -1)
            .reshape(block_count * mask_count, sample_axis, query_count)
        )
        identity_loss = _class_balanced_binary_loss(
            identity_x,
            y,
            row_validity=heldout_validity.reshape(block_count * mask_count, sample_axis), sample_weights=source_weights,
        ).reshape(block_count, mask_count, query_count)
        fold_losses[:, :strength_count, :, :] = finite_loss.permute(1, 2, 0, 3)
        fold_losses[:, -1, :, :] = identity_loss.permute(1, 0, 2)
    for block in range(block_count) if not rectangular_selection else ():
        if fold_membership is None:
            training = block_ids != block
            heldout = ~training
            strength_grid = (
                finite_strengths.reshape(1, -1)
                .expand(mask_count, -1)
                .reshape(-1, 1)
                .expand(-1, query_count)
            )
            # Historical subsetting path (bit-preserving for existing
            # single-fold callers), generalized to K query columns.
            train_count = int(training.sum().item())
            heldout_count = int(heldout.sum().item())
            expanded = (
                x[:, training, :]
                .unsqueeze(1)
                .expand(mask_count, strength_count, -1, -1)
                .reshape(mask_count * strength_count, train_count, query_count)
            )
            fitted = fit_batched_post_query_affine(
                expanded,
                y[training],
                sample_weights=(
                    None if source_weights is None else source_weights[training]
                ),
                identity_anchor_strength=strength_grid,
                **inner_fit_options,
            )
            _record_gates(
                fitted,
                tuple(
                    lane // strength_count
                    for lane in range(mask_count * strength_count)
                ),
                "selection",
            )
            heldout_expanded = (
                x[:, heldout, :]
                .unsqueeze(1)
                .expand(mask_count, strength_count, -1, -1)
                .reshape(mask_count * strength_count, heldout_count, query_count)
            )
            finite_loss = _class_balanced_binary_loss(
                fitted.apply(heldout_expanded),
                y[heldout], sample_weights=None if source_weights is None else source_weights[heldout],
            ).reshape(mask_count, strength_count, query_count)
            identity_loss = _class_balanced_binary_loss(x[:, heldout, :], y[heldout], sample_weights=None if source_weights is None else source_weights[heldout])
            fold_losses[:, :strength_count, block, :] = finite_loss
            fold_losses[:, -1, block, :] = identity_loss
        else:
            # Each outer fold receives its own deterministic inner split. The
            # active folds for this block still share one batched solve, so
            # local selection semantics do not sacrifice batched throughput.
            active_folds = [
                fold for fold, local_count in enumerate(block_counts)
                if block < local_count
            ]
            active_index = torch.as_tensor(
                active_folds, device=x.device, dtype=torch.long
            )
            active_count = int(active_index.numel())
            active_x = x.index_select(0, active_index)
            active_membership = fold_membership.index_select(0, active_index)
            active_block_ids = block_ids.index_select(0, active_index)
            train_validity = active_membership * (active_block_ids != block).to(
                dtype=torch.float64
            )
            heldout_validity = active_membership * (active_block_ids == block).to(
                dtype=torch.float64
            )
            strength_grid = (
                finite_strengths.reshape(1, -1)
                .expand(active_count, -1)
                .reshape(-1, 1)
                .expand(-1, query_count)
            )
            expanded_validity = (
                train_validity.unsqueeze(1)
                .expand(active_count, strength_count, -1)
                .reshape(active_count * strength_count, int(x.shape[1]))
            )
            expanded = (
                active_x.unsqueeze(1)
                .expand(active_count, strength_count, -1, -1)
                .reshape(active_count * strength_count, int(x.shape[1]), query_count)
            )
            fitted = fit_batched_post_query_affine(
                expanded,
                y,
                sample_weights=source_weights,
                row_validity=expanded_validity,
                identity_anchor_strength=strength_grid,
                **inner_fit_options,
            )
            _record_gates(
                fitted,
                tuple(
                    active_folds[lane // strength_count]
                    for lane in range(active_count * strength_count)
                ),
                "selection",
            )
            heldout_expanded_validity = (
                heldout_validity.unsqueeze(1)
                .expand(active_count, strength_count, -1)
                .reshape(active_count * strength_count, int(x.shape[1]))
            )
            finite_loss = _class_balanced_binary_loss(
                fitted.apply(expanded),
                y,
                row_validity=heldout_expanded_validity, sample_weights=source_weights,
            ).reshape(active_count, strength_count, query_count)
            identity_loss = _class_balanced_binary_loss(
                active_x, y, row_validity=heldout_validity, sample_weights=source_weights
            )
            fold_losses[active_index, :strength_count, block, :] = finite_loss
            fold_losses[active_index, -1, block, :] = identity_loss

    if fold_membership is None:
        means = fold_losses.mean(dim=2)
        errors = fold_losses.std(dim=2, unbiased=True) / math.sqrt(
            float(block_count)
        )
    else:
        local_counts = torch.as_tensor(
            block_counts, device=x.device, dtype=torch.float64
        ).reshape(mask_count, 1, 1)
        populated = ~torch.isnan(fold_losses)
        sums = torch.where(
            populated, fold_losses, torch.zeros_like(fold_losses)
        ).sum(dim=2)
        means = sums / local_counts
        deviations = torch.where(
            populated,
            fold_losses - means.unsqueeze(2),
            torch.zeros_like(fold_losses),
        )
        errors = torch.sqrt(
            deviations.square().sum(dim=2)
            / (local_counts - 1.0)
            / local_counts
        )
    best_index = means.argmin(dim=1)
    best_threshold = torch.take_along_dim(
        means, best_index.unsqueeze(1), dim=1
    ) + torch.take_along_dim(errors, best_index.unsqueeze(1), dim=1)
    eligible = means <= best_threshold
    complexity = torch.arange(
        candidate_count, device=x.device, dtype=torch.int64
    ).reshape(1, -1, 1)
    selected_index = torch.where(
        eligible,
        complexity,
        torch.full_like(complexity, -1),
    ).amax(dim=1)
    selection_missing = torch.as_tensor(
        selection_missing_host, dtype=torch.bool
    ).to(device=x.device)
    # A missing column's one-SE rule ran on uncertified lanes; its index
    # is pinned to a valid grid slot (its output is NaN below), so no
    # undefined index reaches the deployment refit.
    selected_index = torch.where(
        selection_missing, torch.zeros_like(selected_index), selected_index
    )
    identity_selected = selected_index == strength_count
    selected_finite = finite_strengths[
        selected_index.clamp_max(strength_count - 1)
    ].reshape(mask_count, query_count)
    identity_or_missing = identity_selected | selection_missing

    if deployment_refit_anchor not in {"selected", "unregularized"}:
        raise ValueError(
            "deployment_refit_anchor must be 'selected' or 'unregularized'"
        )
    if deployment_refit_anchor == "unregularized" and _emit_differential_owner:
        raise ValueError(
            "the differential-owner path certifies the anchored deployment "
            "fit; combine it with deployment_refit_anchor='selected'"
        )
    deployment_refit_receipt: dict[str, Any] | None = None
    final_fit_receipt: Mapping[str, Any]
    positive_affine_owner = None
    if bool(identity_or_missing.all().item()):
        scale = torch.ones(
            (mask_count, query_count), device=x.device, dtype=x.dtype
        )
        offset = torch.zeros_like(scale)
        final_fit_receipt = {
            "schema_name": "mechanistic_identity_calibration_receipt",
            "schema_version": "1",
            "passed": True,
            "reason": "identity_selected_within_one_standard_error",
        }
        solver_calls = 0
        solver_backend = "identity"
    else:
        fit_options = dict(
            weighting=weighting, sample_weights=sample_weights,
            minimum_scale=minimum_scale,
            initial_scale=initial_scale,
            initial_offset=initial_offset,
            max_iterations=max_iterations,
            tolerance=tolerance, hessian_ridge=hessian_ridge,
            objective_tolerance=objective_tolerance,
            reference_tolerance=reference_tolerance,
            statistics_backend=statistics_backend, require_convergence=True,
            source_logit_dtype=source_logit_dtype or x.dtype,
            identity_anchor_strength=selected_finite,
            anchor_probability_epsilon=anchor_probability_epsilon,
            required_query_mask=required_query_mask,
            reference_audit_policy=reference_audit_policy,
            audit_seed=audit_seed,
        )
        if fold_membership is not None:
            fit_options["row_validity"] = fold_membership
        if _emit_differential_owner:
            owned_fit = fit_batched_post_query_affine_with_differential_owner(
                x, y, factorization=factorization, context_ids=context_ids, **fit_options
            )
            final_fit = owned_fit.fit
            positive_affine_owner = owned_fit.differential_owner
        else:
            final_fit = fit_batched_post_query_affine(
                x, y, factorization=factorization, **fit_options
            )
        scale = final_fit.scale.detach().clone()
        offset = final_fit.offset.detach().clone()
        _record_gates(final_fit, tuple(range(mask_count)), "deployment")
        final_fit_receipt = final_fit.receipt.to_dict()
        solver_calls = sum(int(value) for value in final_fit.receipt.iterations)
        solver_backend = final_fit.backend
        if deployment_refit_anchor == "unregularized":
            # Crossfit selection keeps its variance-control role, but the
            # DEPLOYED parameters come from one additional lambda=0 solve so
            # the raw BCE gradient vanishes at interior optima and the
            # attainable-improvement certificate loses its anchored
            # first-order floor. Cells that fail unregularized convergence
            # (complete/quasi separation arms only at lambda=0) fall back
            # per-cell to the anchored parameters, receipted.
            refit_options = dict(fit_options)
            refit_options.update(
                identity_anchor_strength=0.0,
                require_convergence=False,
            )
            unregularized_fit = fit_batched_post_query_affine(
                x, y, factorization=factorization, **refit_options
            )
            # Informational: an unregularized cell that fails falls back to
            # its anchored parameters (the registered rule), never missing.
            _record_gates(
                unregularized_fit,
                tuple(range(mask_count)),
                "deployment_unregularized",
            )
            refit_converged = torch.as_tensor(
                unregularized_fit.receipt.converged,
                device=scale.device,
                dtype=torch.bool,
            ).reshape(scale.shape)
            adopt = refit_converged & ~identity_selected
            scale = torch.where(
                adopt, unregularized_fit.scale.detach(), scale
            )
            offset = torch.where(
                adopt, unregularized_fit.offset.detach(), offset
            )
            fallback_cells = int(
                ((~refit_converged) & ~identity_selected).sum().item()
            )
            separated_cells = sum(
                1
                for converged, reason in zip(
                    unregularized_fit.receipt.converged,
                    unregularized_fit.receipt.termination_reason,
                )
                if not converged and "separation" in str(reason)
            )
            solver_calls += sum(
                int(value)
                for value in unregularized_fit.receipt.iterations
            )
            deployment_refit_receipt = {
                "deployment_refit_anchor": "unregularized",
                "adopted_cell_count": int(adopt.sum().item()),
                "fallback_to_anchored_cell_count": fallback_cells,
                "separated_cell_count": int(separated_cells),
                "identity_pinned_cell_count": int(
                    identity_selected.sum().item()
                ),
                "refit_termination_reasons": tuple(
                    str(reason)
                    for reason in
                    unregularized_fit.receipt.termination_reason
                ),
                "refit_gradient_kkt_residual": tuple(
                    float(value)
                    for value in
                    unregularized_fit.receipt.gradient_kkt_residual
                ),
            }
        scale[identity_selected] = 1.0
        offset[identity_selected] = 0.0

    missing_host = [row[:] for row in selection_missing_host]
    identity_host = identity_selected.detach().to(device="cpu").tolist()
    for record in gate_records:
        if record["stage"] != "deployment":
            continue
        for cell in record["missing_cells"]:
            key = (int(cell["mask_index"]), int(cell["query_index"]))
            # An identity column needs no refit; a column already
            # missing at selection stays missing for that reason.
            if identity_host[key[0]][key[1]] or missing_host[key[0]][key[1]]:
                continue
            missing_host[key[0]][key[1]] = True
            missing_stage.setdefault(key, set()).add("deployment")
    missing = torch.as_tensor(missing_host, dtype=torch.bool)
    if bool(missing.any().item()):
        summary = _v6_gate_summary(
            gate_records,
            missing=missing,
            missing_stage=missing_stage,
            query_axis_ids=tuple(
                str(query_id)
                for component in factorization.components
                for query_id in component.query_axis_ids
            ),
        )
        if (
            _emit_differential_owner
            or str(missing_column_policy) != MISSING_COLUMN_POLICY_CARRY
        ):
            failed_ids = tuple(summary["missing_query_axis_ids"])
            error = CalibrationFailClosedError(
                "proper-score columns are missing after the device fallback "
                f"(fail closed; missing_column_policy={missing_column_policy!r}"
                f"{', differential owner' if _emit_differential_owner else ''}): "
                f"query_axis_ids={failed_ids!r} "
                f"reasons={sorted({r for c in summary['missing_components'] for r in c['reasons']})!r}",
                last_gated_fit[0].receipt,
                failed_query_axis_ids=failed_ids,
            )
            error.gate_summary = summary
            raise error
        device_missing = missing.to(device=scale.device)
        # MISSING, never substituted: the deployed affine is NaN.
        scale = torch.where(
            device_missing, torch.full_like(scale, float("nan")), scale
        )
        offset = torch.where(
            device_missing, torch.full_like(offset, float("nan")), offset
        )

    tokens = tuple(_strength_token(value) for value in strengths) + (
        IDENTITY_CANDIDATE,
    )
    # One component row per (mask, query) cell; at K=1 this is exactly the
    # per-mask structure. Materialized with one host transfer per
    # tensor, without per-cell slicing.
    selected_tokens, component_rows = _materialize_selection_rows(
        tokens=tokens,
        selected_flat=selected_index.detach().cpu(),
        best_flat=best_index.detach().cpu(),
        means_cpu=means.detach().cpu(),
        errors_cpu=errors.detach().cpu(),
        fold_losses_cpu=fold_losses.detach().cpu(),
        eligible_cpu=eligible.detach().cpu(),
        block_counts=block_counts,
        block_count=block_count,
    )
    query_axis_ids = tuple(
        str(query_id)
        for component in factorization.components
        for query_id in component.query_axis_ids
    )
    gate_summary = _v6_gate_summary(
        gate_records,
        missing=missing,
        missing_stage=missing_stage,
        query_axis_ids=query_axis_ids,
    )
    if gate_summary["missing_component_count"]:
        reasons_by_key = {
            (int(item["mask_index"]), int(item["query_index"])): item
            for item in gate_summary["missing_components"]
        }
        selected_list = list(selected_tokens)
        for flat, row in enumerate(component_rows):
            key = (int(row["mask_index"]), int(row["query_index"]))
            item = reasons_by_key.get(key)
            if item is None:
                continue
            row["selected_strength"] = MISSING_SELECTION_TOKEN
            row["missing"] = {
                "rule": V6_MISSING_CELL_RULE,
                "stages": list(item["stages"]),
                "reasons": list(item["reasons"]),
            }
            for name in ("mean_losses", "standard_errors", "fold_losses"):
                row[name] = _receipt_safe_losses(row[name])
            selected_list[flat] = MISSING_SELECTION_TOKEN
        selected_tokens = tuple(selected_list)
    profile = BackendProfile(
        backend_id="mechanistic_systems.proper_score_shrinkage.v2",
        capability="calibration_only_selected_identity_anchored_affine",
        implementation="batched_float64_projected_newton_predictive_kl",
        device_type=str(x.device.type),
        numerical_mode="direct_positive_affine_with_proper_score_anchor",
        calls=int(solver_calls),
        wall_time_s=max(0.0, time.perf_counter() - started),
        peak_hbm_bytes=(
            int(torch.cuda.max_memory_allocated(x.device))
            if x.device.type == "cuda"
            else None
        ),
        processed_contexts=int(mask_count * x.shape[1]),
        processed_queries=mask_count,
        metadata={
            **({"loss_weighting": "per_image", "weighting": "per_image_population_mean"} if loss_weighting() == "per_image" else {}),
            "solver_backend": solver_backend,
            "selection_block_count": block_count,
            "selection_block_count_by_mask": list(block_counts),
            "candidate_strengths": tokens,
            "selection_rule": "largest_identity_anchor_within_one_se_of_best",
            "regularization_geometry": "bernoulli_predictive_kl_to_identity",
            "anchor_probability_epsilon": float(anchor_probability_epsilon),
            "initialization": (
                "identity"
                if initial_scale is None and initial_offset is None
                else "declared_warm_start"
            ),
            **(
                {"deployment_refit": deployment_refit_receipt}
                if deployment_refit_receipt is not None
                else {}
            ),
            "gpu_hard_gates_v6": gate_summary,
        },
    )
    receipt = ProperScoreShrinkageReceipt(
        dependency_hash=factorization.dependency_hash,
        factorization_hash=factorization.factorization_hash,
        algorithm_id=PROPER_SCORE_SHRINKAGE_AFFINE_V2,
        candidate_strengths=tokens,
        selected_strengths=selected_tokens,
        component_rows=tuple(component_rows),
        block_assignment_hash=block_hash,
        block_count=block_count,
        selection_rule="largest_identity_anchor_within_one_se_of_best",
        fit_receipt=final_fit_receipt,
        backend_profile=profile.to_runtime_dict(),
    )
    receipt.validate()
    fit = ProperScoreShrinkageFit(
        scale=scale,
        offset=offset,
        factorization=factorization,
        receipt=receipt,
        input_was_batched=input_was_batched,
    )
    selection = None
    if _emit_differential_owner:
        # Owner emission is single-column, single-fold (guarded above);
        # project the query axis away for the historical face shapes.
        selection = _selection_face(
            factorization=factorization, tokens=tokens,
            means=means[:, :, 0], errors=errors[:, :, 0],
            best_index=best_index[:, 0], selected_index=selected_index[:, 0],
            eligible=eligible[:, :, 0],
            fold_losses=fold_losses[:, :, :, 0], block_hash=block_hash,
            context_ids=context_ids,
            values=x, targets=y, weighting=weighting, fit=fit,
        )
    return fit, selection, positive_affine_owner


def fit_proper_score_shrinkage_affine(
    values: Any, binary_targets: Any, *, factorization: CalibrationFactorization,
    context_ids: Sequence[Any], weighting: str = "pos_weighted",
    sample_weights: Any | None = None, minimum_scale: float = 1e-8,
    calibration_penalty: str = "identity_anchor", class_ids: Any | None = None,
    fixed_offsets: Any | None = None,
    firth_convergence: str = "absolute",
    metric_loss_weighting: str | None = None,
    candidate_strengths: Sequence[float] = DEFAULT_IDENTITY_ANCHOR_STRENGTHS,
    selection_block_count: int = CANONICAL_CALIBRATION_SELECTION_BLOCKS,
    max_iterations: int = CANONICAL_CALIBRATION_MAX_ITERATIONS,
    tolerance: float = CANONICAL_CALIBRATION_KKT_TOLERANCE,
    hessian_ridge: float = CANONICAL_CALIBRATION_HESSIAN_RIDGE,
    objective_tolerance: float = CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE,
    reference_tolerance: float = CANONICAL_CALIBRATION_REFERENCE_TOLERANCE,
    statistics_backend: str = "auto", source_logit_dtype: Any | None = None,
    anchor_probability_epsilon: float = CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON,
    fold_validity: Any | None = None,
    required_query_mask: Any | None = None,
    reference_audit_policy: str = GPU_HARD_GATES_AUDIT_POLICY,
    audit_seed: int | None = None,
    deployment_refit_anchor: str = "selected",
    missing_column_policy: str = MISSING_COLUMN_POLICY_FAIL_CLOSED,
) -> ProperScoreShrinkageFit:
    return _fit_proper_score_shrinkage_affine_core(
        values, binary_targets, factorization=factorization, context_ids=context_ids,
        weighting=weighting, sample_weights=sample_weights, minimum_scale=minimum_scale,
        calibration_penalty=calibration_penalty, class_ids=class_ids, fixed_offsets=fixed_offsets,
        firth_convergence=firth_convergence,
        metric_loss_weighting=metric_loss_weighting,
        candidate_strengths=candidate_strengths, selection_block_count=selection_block_count,
        max_iterations=max_iterations, tolerance=tolerance, hessian_ridge=hessian_ridge,
        objective_tolerance=objective_tolerance, reference_tolerance=reference_tolerance,
        statistics_backend=statistics_backend, source_logit_dtype=source_logit_dtype,
        anchor_probability_epsilon=anchor_probability_epsilon,
        fold_validity=fold_validity, required_query_mask=required_query_mask,
        reference_audit_policy=reference_audit_policy, audit_seed=audit_seed,
        deployment_refit_anchor=deployment_refit_anchor,
        missing_column_policy=missing_column_policy,
    )[0]


def fit_proper_score_shrinkage_affine_with_owner_receipts(
    values: Any, binary_targets: Any, *, factorization: CalibrationFactorization,
    context_ids: Sequence[Any], **fit_options: Any,
) -> ProperScoreShrinkageOwnedFit:
    fit, selection, positive = _fit_proper_score_shrinkage_affine_core(
        values, binary_targets, factorization=factorization, context_ids=context_ids,
        _emit_differential_owner=True, **fit_options,
    )
    if selection is None:  # pragma: no cover - construction invariant.
        raise RuntimeError("selection owner receipt was not emitted")
    return ProperScoreShrinkageOwnedFit(fit, selection, positive)


__all__ = [
    "CalibrationSelectionFaceComponent",
    "CalibrationSelectionFaceReceipt",
    "DEFAULT_IDENTITY_ANCHOR_STRENGTHS",
    "IDENTITY_CANDIDATE",
    "MISSING_COLUMN_POLICIES",
    "MISSING_COLUMN_POLICY_CARRY",
    "MISSING_COLUMN_POLICY_FAIL_CLOSED",
    "MISSING_SELECTION_TOKEN",
    "ProperScoreShrinkageFit",
    "ProperScoreShrinkageOwnedFit",
    "ProperScoreShrinkageReceipt",
    "deterministic_fold_local_calibration_blocks",
    "deterministic_stratified_calibration_blocks",
    "fit_proper_score_shrinkage_affine",
    "fit_proper_score_shrinkage_affine_with_owner_receipts",
]
