"""Run-scoped calibration execution topology.

This module deliberately separates runtime scheduling from the scientific
calibration contracts.  ``CalibrationFactorization`` remains the one source of
truth for mathematical coupling; the classes here only derive legal batching,
prepare immutable run-local inputs, and record operational provenance.
"""

from __future__ import annotations

import copy
import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from contracts.hashing import stable_digest
from mechanistic_systems.contracts import (
    CalibrationProblemSpec,
    PROPER_SCORE_SHRINKAGE_AFFINE_V2,
    SAMPLED_PATH_V1_CALIBRATION_ALGORITHM,
)

from .calibration_problem import CalibrationFactorization
from .profiling import PhaseProfiler

try:  # Calibration is executable only when PyTorch is available.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


_LEGAL_BATCH_AXES = frozenset(
    {
        "candidate_mask",
        "outer_fold",
        "inner_block",
        "anchor_strength",
        "independent_query_component",
    }
)


# Compile alone was bit-exact in the four-arm H200 A/B.  It is
# therefore the default execution topology; any compile failure falls back to
# the exact uncompiled path.  Forward tier 2 did change the panel trajectory,
# so bf16 autocast remains off unless explicitly requested.
_EXACT_RUNTIME_DEFAULT_VMAP_COMPILE = True
_EXACT_RUNTIME_DEFAULT_FORWARD_TIER = 0
_EXACT_RUNTIME_DEFAULT_PROVENANCE = "compiled_exact_runtime_default_v3"


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("calibration execution topology requires PyTorch")


def _environment_bool(
    environ: Mapping[str, str], name: str, default: bool
) -> bool:
    value = environ.get(name)
    if value is None or not str(value).strip():
        return bool(default)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean environment value")


def _environment_int(
    environ: Mapping[str, str], name: str, default: int | None
) -> int | None:
    value = environ.get(name)
    if value is None or not str(value).strip():
        return default
    try:
        result = int(str(value).strip())
    except ValueError as error:
        raise ValueError(f"{name} must be an integer environment value") from error
    return result


@dataclass(frozen=True)
class RuntimeExecutionProfile:
    """Immutable runtime-only choices resolved once at the process boundary."""

    vmap_compile: bool = _EXACT_RUNTIME_DEFAULT_VMAP_COMPILE
    k1_compile: bool = True
    # IID source deduplication and fused-role cache reuse can produce a
    # different physical tail length for every mask.  Mark only the leading
    # context dimension dynamic so one structural K=1/K=2 graph carries those
    # runtime lengths instead of recompiling on every prefix.
    compile_dynamic_batch: bool = True
    forward_tier: int = _EXACT_RUNTIME_DEFAULT_FORWARD_TIER
    execution_batch_rows: int | None = None
    # Native v0/v1 physical panel sizes are intentionally independent of
    # statistical look sizes.  The controller chooses how much evidence is
    # required; these runtime-only values choose how those frozen rows fill
    # the GPU.  K=1 lanes admit larger batches than v0's paired K=2 lane.
    scoring_batch_rows: int = 32
    calibration_batch_rows: int = 512
    v0_paired_oracle_batch_rows: int = 256
    v1_candidate_oracle_batch_rows: int = 512
    terminal_batch_rows: int = 512
    native_panel_device_cache_bytes: int = 2 << 30
    activation_prefix_cache_bytes: int = 2 << 30
    require_optimized_native_panel: bool = False
    unit_packing_degree: int = 1
    query_tile_size: int = 1
    # NVIDIA MPS is an inter-process campaign-throughput layer.  It is
    # orthogonal to in-process K-lane vmap execution: a packed unit may use
    # both.  The daemon and pipe lifecycle are owned by the Slurm launcher;
    # the process records only the observed, runtime-only contract.
    multiprocess_service: bool = False
    multiprocess_active_thread_percentage: int | None = None
    multiprocess_pipe_configured: bool = False
    incremental_responses: bool = True
    sampled_fit_cache: bool = True
    sampled_row_store_bytes: int = 4 << 30
    # Measured at production panel scale, the default 512MB cap
    # thrashes (17 hits / 7524 misses / 7470 evictions, working set ~74GB);
    # the default is kept for small hosts and cluster jobs opt in via
    # CIRCUIT_SAMPLED_RESPONSE_CACHE_BYTES. Cache hits return stored exact
    # responses, so the size can never change a decision.
    sampled_response_cache_bytes: int = 512 << 20
    # Measured: candidate.batch_wait (producer starvation) is 550s
    # of the unit - prefetch depth bounds how far the background producer
    # runs ahead of the consuming forwards. Runtime-only buffering.
    sampled_prefetch_depth: int = 2
    # Staging workers parallelize panel spans and preserve their row order.
    sampled_producer_workers: int = 1
    # Cold-path prefetch: fault the declared working set's pages on daemon
    # threads (bank-ordinal order, one strided byte per page) ahead of the
    # serial gathers, so a single unit on a cold shared filesystem never
    # pays the full-bank staging copy just to warm rows it will not touch.
    # Runtime-only page faults; no logical draw, order, or value changes.
    working_set_prefetch: bool = False
    # Within-session memo of drawn panels keyed by the FULL
    # draw identity (request scalars + prior-panel hash + warm-start and
    # evidence digests). Panels are frozen and draws deterministic in
    # that identity, so a hit is bit-identical to redrawing (the
    # panel_draw_repeats telemetry priced this at ~413s/unit). 0 = off.
    panel_draw_memo_entries: int = 0
    # Defer the per-readout D2H + float64 cast (the ~245s
    # response_materialize span) — `values` materializes once on first
    # host access from the retained device tensor, identical bytes.
    # Consumers that only touch device_values never pay the transfer.
    lazy_readout: bool = False
    sampled_coalesce_rows: int = 0
    prepared_view_cache_entries: int = 16
    # CUDA-graph capture of the K=1 single-lane forward. Capture and replay
    # errors propagate from the selected execution path.
    cuda_graphs: bool = False
    cuda_events: bool = False
    # Runtime-only TF32 is DEFAULT-ON (validated against a
    # byte-reproduction reference: 51/51 identical decisions, terminal
    # delta shift 1.4e-5). CIRCUIT_ALLOW_TF32=0 opts out; the applied
    # value is disclosed in every receipt.
    allow_tf32: bool = True
    provenance: str = _EXACT_RUNTIME_DEFAULT_PROVENANCE

    def validate(self) -> None:
        if self.forward_tier not in {0, 1, 2}:
            raise ValueError("runtime forward tier must be 0, 1, or 2")
        if self.execution_batch_rows is not None and self.execution_batch_rows < 1:
            raise ValueError("runtime execution batch must be positive")
        for name in (
            "scoring_batch_rows",
            "calibration_batch_rows",
            "v0_paired_oracle_batch_rows",
            "v1_candidate_oracle_batch_rows",
            "terminal_batch_rows",
            "unit_packing_degree",
            "query_tile_size",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"runtime {name} must be positive")
        if int(self.native_panel_device_cache_bytes) < 0:
            raise ValueError(
                "runtime native panel device cache bytes cannot be negative"
            )
        if int(self.activation_prefix_cache_bytes) < 1:
            raise ValueError(
                "runtime activation-prefix cache bytes must be positive"
            )
        if (
            self.multiprocess_active_thread_percentage is not None
            and not 1
            <= int(self.multiprocess_active_thread_percentage)
            <= 100
        ):
            raise ValueError(
                "runtime MPS active-thread percentage must lie in [1,100]"
            )
        if self.multiprocess_service and not self.multiprocess_pipe_configured:
            raise ValueError(
                "runtime MPS mode requires a configured CUDA MPS pipe"
            )
        if self.sampled_row_store_bytes < 0:
            raise ValueError("runtime row-store bytes cannot be negative")
        if self.sampled_response_cache_bytes < 0:
            raise ValueError("runtime response-cache bytes cannot be negative")
        if self.sampled_prefetch_depth < 1:
            raise ValueError("runtime prefetch depth must be positive")
        if self.sampled_producer_workers < 1:
            raise ValueError("runtime producer workers must be positive")
        if self.panel_draw_memo_entries < 0:
            raise ValueError("runtime panel-draw memo entries cannot be negative")
        if self.sampled_coalesce_rows < 0:
            raise ValueError("runtime coalesce rows cannot be negative")
        if self.prepared_view_cache_entries < 0:
            raise ValueError("runtime prepared-view cache entries cannot be negative")
        if not self.provenance:
            raise ValueError("runtime profile provenance must be nonempty")

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "RuntimeExecutionProfile":
        """Bridge legacy ``CIRCUIT_*`` settings once into typed state."""

        values = os.environ if environ is None else environ
        execution_batch = _environment_int(values, "CIRCUIT_EXECUTION_BATCH", None)
        # TF32 intentionally bypasses _environment_bool: only the exact
        # opt-out "0" disables; unset or malformed means the default-on
        # policy, and the realized value is receipt-disclosed either way.
        allow_tf32 = str(values.get("CIRCUIT_ALLOW_TF32", "1")).strip() != "0"
        profile = cls(
            vmap_compile=_environment_bool(
                values,
                "CIRCUIT_VMAP_COMPILE",
                _EXACT_RUNTIME_DEFAULT_VMAP_COMPILE,
            ),
            k1_compile=_environment_bool(
                values,
                "CIRCUIT_K1_COMPILE",
                True,
            ),
            compile_dynamic_batch=_environment_bool(
                values,
                "CIRCUIT_COMPILE_DYNAMIC_BATCH",
                True,
            ),
            forward_tier=int(
                _environment_int(
                    values,
                    "CIRCUIT_FORWARD_TIER",
                    _EXACT_RUNTIME_DEFAULT_FORWARD_TIER,
                )
                or 0
            ),
            execution_batch_rows=execution_batch,
            scoring_batch_rows=int(
                _environment_int(values, "CIRCUIT_SCORING_BATCH_ROWS", 32)
                or 32
            ),
            calibration_batch_rows=int(
                _environment_int(
                    values, "CIRCUIT_CALIBRATION_BATCH_ROWS", 512
                )
                or 512
            ),
            v0_paired_oracle_batch_rows=int(
                _environment_int(
                    values, "CIRCUIT_V0_PAIRED_ORACLE_BATCH_ROWS", 256
                )
                or 256
            ),
            v1_candidate_oracle_batch_rows=int(
                _environment_int(
                    values, "CIRCUIT_V1_CANDIDATE_ORACLE_BATCH_ROWS", 512
                )
                or 512
            ),
            terminal_batch_rows=int(
                _environment_int(
                    values, "CIRCUIT_TERMINAL_BATCH_ROWS", 512
                )
                or 512
            ),
            native_panel_device_cache_bytes=int(
                _environment_int(
                    values,
                    "CIRCUIT_NATIVE_PANEL_DEVICE_CACHE_BYTES",
                    2 << 30,
                )
                or 0
            ),
            activation_prefix_cache_bytes=int(
                _environment_int(
                    values,
                    "CIRCUIT_ACTIVATION_PREFIX_CACHE_BYTES",
                    2 << 30,
                )
                or 2 << 30
            ),
            require_optimized_native_panel=_environment_bool(
                values, "CIRCUIT_REQUIRE_OPTIMIZED_NATIVE_PANEL", False
            ),
            unit_packing_degree=int(
                _environment_int(values, "CIRCUIT_UNIT_PACKING_DEGREE", 1)
                or 1
            ),
            query_tile_size=int(
                _environment_int(values, "CIRCUIT_QUERY_TILE_SIZE", 1) or 1
            ),
            multiprocess_service=_environment_bool(
                values, "CIRCUIT_MPS_ENABLED", False
            ),
            multiprocess_active_thread_percentage=_environment_int(
                values,
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                None,
            ),
            multiprocess_pipe_configured=bool(
                str(values.get("CUDA_MPS_PIPE_DIRECTORY", "")).strip()
            ),
            incremental_responses=_environment_bool(
                values, "CIRCUIT_SAMPLED_INCREMENTAL", True
            ),
            sampled_fit_cache=_environment_bool(
                values, "CIRCUIT_SAMPLED_FIT_CACHE", True
            ),
            sampled_row_store_bytes=int(
                _environment_int(
                    values, "CIRCUIT_SAMPLED_ROW_STORE_BYTES", 4 << 30
                )
                or 0
            ),
            sampled_response_cache_bytes=int(
                _environment_int(
                    values, "CIRCUIT_SAMPLED_RESPONSE_CACHE_BYTES", 512 << 20
                )
                or 0
            ),
            sampled_prefetch_depth=int(
                _environment_int(values, "CIRCUIT_SAMPLED_PREFETCH_DEPTH", 2)
                or 2
            ),
            sampled_producer_workers=int(
                _environment_int(values, "CIRCUIT_PRODUCER_WORKERS", 1) or 1
            ),
            working_set_prefetch=_environment_bool(
                values, "CIRCUIT_WORKING_SET_PREFETCH", False
            ),
            panel_draw_memo_entries=int(
                _environment_int(values, "CIRCUIT_PANEL_DRAW_MEMO", 0) or 0
            ),
            lazy_readout=_environment_bool(
                values, "CIRCUIT_LAZY_READOUT", False
            ),
            sampled_coalesce_rows=int(
                _environment_int(values, "CIRCUIT_SAMPLED_COALESCE", 0) or 0
            ),
            prepared_view_cache_entries=int(
                _environment_int(
                    values, "CIRCUIT_CALIBRATION_PREPARED_VIEWS", 16
                )
                or 0
            ),
            cuda_graphs=_environment_bool(
                values, "CIRCUIT_CUDA_GRAPHS", False
            ),
            cuda_events=_environment_bool(
                values, "MECHANISTIC_PROFILE_CUDA_EVENTS", False
            ),
            allow_tf32=allow_tf32,
            provenance="environment_bridge_compiled_exact_default_v3",
        )
        profile.validate()
        return profile

    @classmethod
    def resolve(
        cls,
        explicit: "RuntimeExecutionProfile | None" = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "RuntimeExecutionProfile":
        """Give an explicit typed profile precedence over ambient settings."""

        if explicit is not None:
            explicit.validate()
            return explicit
        return cls.from_environment(environ)

    @property
    def content_digest(self) -> str:
        return stable_digest(self.to_runtime_dict())

    def to_runtime_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_name": "mechanistic_runtime_execution_profile_v1",
            "vmap_compile": bool(self.vmap_compile),
            "k1_compile": bool(self.k1_compile),
            "compile_dynamic_batch": bool(self.compile_dynamic_batch),
            "forward_tier": int(self.forward_tier),
            "execution_batch_rows": self.execution_batch_rows,
            "scoring_batch_rows": int(self.scoring_batch_rows),
            "calibration_batch_rows": int(self.calibration_batch_rows),
            "v0_paired_oracle_batch_rows": int(
                self.v0_paired_oracle_batch_rows
            ),
            "v1_candidate_oracle_batch_rows": int(
                self.v1_candidate_oracle_batch_rows
            ),
            "terminal_batch_rows": int(self.terminal_batch_rows),
            "native_panel_device_cache_bytes": int(
                self.native_panel_device_cache_bytes
            ),
            "activation_prefix_cache_bytes": int(
                self.activation_prefix_cache_bytes
            ),
            "require_optimized_native_panel": bool(
                self.require_optimized_native_panel
            ),
            "unit_packing_degree": int(self.unit_packing_degree),
            "query_tile_size": int(self.query_tile_size),
            "multiprocess_service": bool(self.multiprocess_service),
            "multiprocess_active_thread_percentage": (
                None
                if self.multiprocess_active_thread_percentage is None
                else int(self.multiprocess_active_thread_percentage)
            ),
            "multiprocess_pipe_configured": bool(
                self.multiprocess_pipe_configured
            ),
            "incremental_responses": bool(self.incremental_responses),
            "sampled_fit_cache": bool(self.sampled_fit_cache),
            "sampled_row_store_bytes": int(self.sampled_row_store_bytes),
            "sampled_response_cache_bytes": int(self.sampled_response_cache_bytes),
            "sampled_prefetch_depth": int(self.sampled_prefetch_depth),
            "sampled_producer_workers": int(self.sampled_producer_workers),
            "panel_draw_memo_entries": int(self.panel_draw_memo_entries),
            "lazy_readout": bool(self.lazy_readout),
            "sampled_coalesce_rows": int(self.sampled_coalesce_rows),
            "prepared_view_cache_entries": int(self.prepared_view_cache_entries),
            "cuda_graphs": bool(self.cuda_graphs),
            "cuda_events": bool(self.cuda_events),
            "allow_tf32": bool(self.allow_tf32),
            "provenance": self.provenance,
            "scientific_identity": "excluded",
        }


@dataclass(frozen=True)
class CrossFitSupport:
    """Support facts for one outer fit, recorded without outcome selection."""

    outer_fold_index: int
    fit_row_count: int
    positive_counts: tuple[int, ...]
    negative_counts: tuple[int, ...]
    block_count: int

    def validate(self) -> None:
        if self.outer_fold_index < 0 or self.fit_row_count <= 0:
            raise ValueError("cross-fit support fold or row count is invalid")
        if len(self.positive_counts) != len(self.negative_counts):
            raise ValueError("cross-fit support query axes differ")
        if any(value < 0 for value in (*self.positive_counts, *self.negative_counts)):
            raise ValueError("cross-fit support counts cannot be negative")
        if self.block_count < 0:
            raise ValueError("cross-fit support block count cannot be negative")

    @property
    def binary_supported(self) -> bool:
        return bool(self.positive_counts) and all(
            positive > 0 and negative > 0
            for positive, negative in zip(
                self.positive_counts, self.negative_counts, strict=True
            )
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "outer_fold_index": int(self.outer_fold_index),
            "fit_row_count": int(self.fit_row_count),
            "positive_counts": list(self.positive_counts),
            "negative_counts": list(self.negative_counts),
            "block_count": int(self.block_count),
            "binary_supported": bool(self.binary_supported),
        }


def _binary_target_matrix(binary_targets: Any) -> Any:
    _require_torch()
    target = torch.as_tensor(binary_targets, device="cpu", dtype=torch.float64)
    if target.ndim != 2 or int(target.shape[0]) < 1 or int(target.shape[1]) < 1:
        raise ValueError("cross-fit targets must have shape [sample, query]")
    if not bool(((target == 0.0) | (target == 1.0)).all().item()):
        raise ValueError("cross-fit binary targets must contain only zero and one")
    return target


def _ordered_context_ids(context_ids: Sequence[Any], count: int) -> tuple[str, ...]:
    ids = tuple(str(value) for value in context_ids)
    if len(ids) != int(count) or len(set(ids)) != len(ids):
        raise ValueError("cross-fit context IDs must be unique and ordered")
    return ids


def _binary_stratified_block_assignment(
    target: Any,
    context_ids: tuple[str, ...],
    *,
    requested_blocks: int,
) -> tuple[Any, int, str, str]:
    """The deterministic block rule, factored for reuse."""

    query_count = int(target.shape[1])
    if query_count == 1:
        strata = target[:, 0].to(torch.int64).tolist()
        rule = "sha256_ordered_within_binary_class_round_robin_v1"
    else:
        strata = torch.where(
            target.sum(dim=1) > 0.5,
            target.argmax(dim=1),
            torch.full((int(target.shape[0]),), -1, dtype=torch.int64),
        ).tolist()
        rule = "sha256_ordered_within_multiclass_stratum_round_robin_v1"
    groups: dict[int, list[int]] = {0: [], 1: []} if query_count == 1 else {}
    for index, stratum in enumerate(strata):
        groups.setdefault(int(stratum), []).append(index)
    block_count = min(
        max(2, int(requested_blocks)), *(len(indices) for indices in groups.values())
    )
    if block_count < 2:
        raise ValueError(
            "cross-fit selection requires at least two examples of each stratum"
        )
    assignment = [-1] * len(context_ids)
    for stratum in sorted(groups):
        ordered = sorted(
            groups[stratum],
            key=lambda index: hashlib.sha256(
                f"{stratum}\0{context_ids[index]}".encode("utf-8")
            ).hexdigest(),
        )
        for rank, index in enumerate(ordered):
            assignment[index] = int(rank % block_count)
    assignments = torch.tensor(assignment, dtype=torch.int64)
    # Validated 0.0/1.0 targets convert losslessly to int64, so the batched
    # counts are exact independent of accumulation order.  Materialize only
    # the single existential result needed by the exception contract.
    integer_target = target.to(torch.int64)
    heldout_positive = torch.zeros(
        (block_count, query_count), dtype=torch.int64
    )
    heldout_positive.index_add_(0, assignments, integer_target)
    total_positive = integer_target.sum(dim=0)
    training_positive = total_positive.unsqueeze(0) - heldout_positive
    heldout_row_count = torch.bincount(
        assignments, minlength=block_count
    )
    training_row_count = int(target.shape[0]) - heldout_row_count
    degenerate = (
        (heldout_positive == 0.0)
        | (heldout_positive == heldout_row_count.unsqueeze(1))
        | (training_positive == 0.0)
        | (training_positive == training_row_count.unsqueeze(1))
    )
    if bool(degenerate.any().item()):
        raise ValueError(
            "cross-fit inner blocks must retain both classes in train and holdout"
        )
    assignment_hash = stable_digest(
        {
            "rule": rule,
            "context_ids": context_ids,
            "block_ids": assignment,
            "block_count": block_count,
        }
    )
    return assignments, int(block_count), rule, assignment_hash


@dataclass(frozen=True)
class CrossFitLayout:
    """Frozen outer and fold-local-inner membership for one calibration run."""

    context_ids: tuple[str, ...]
    outer_fit_membership: tuple[tuple[bool, ...], ...]
    inner_block_ids: tuple[tuple[int, ...], ...]
    inner_block_counts: tuple[int, ...]
    support: tuple[CrossFitSupport, ...]
    requested_block_count: int
    selection_enabled: bool
    stratification_rule: str
    block_assignment_hash: str
    context_digest: str
    content_digest: str

    def validate(self) -> None:
        # A layout whose containers are all tuples is immutable, so a
        # passed validation is a property of the object, and the terminal
        # bank validates its one 256k-row layout once, not in every chunk.
        if self.__dict__.get("_validated_immutable", False):
            return
        self._validate_fields()
        if self._immutable_containers():
            object.__setattr__(self, "_validated_immutable", True)

    def _immutable_containers(self) -> bool:
        return (
            type(self.context_ids) is tuple
            and type(self.outer_fit_membership) is tuple
            and all(type(row) is tuple for row in self.outer_fit_membership)
            and type(self.inner_block_ids) is tuple
            and all(type(row) is tuple for row in self.inner_block_ids)
            and type(self.inner_block_counts) is tuple
            and type(self.support) is tuple
            and all(
                type(row) is CrossFitSupport
                and type(row.positive_counts) is tuple
                and type(row.negative_counts) is tuple
                for row in self.support
            )
        )

    def _validate_fields(self) -> None:
        count = len(self.context_ids)
        if count == 0 or len(set(self.context_ids)) != count:
            raise ValueError("cross-fit layout context IDs are invalid")
        fold_count = len(self.outer_fit_membership)
        if fold_count == 0:
            raise ValueError("cross-fit layout requires at least one outer fold")
        aligned = (
            self.inner_block_ids,
            self.inner_block_counts,
            self.support,
        )
        if any(len(value) != fold_count for value in aligned):
            raise ValueError("cross-fit layout fold axes are misaligned")
        if self.requested_block_count < 0:
            raise ValueError("cross-fit requested block count cannot be negative")
        for fold, (membership, blocks, block_count, support) in enumerate(
            zip(
                self.outer_fit_membership,
                self.inner_block_ids,
                self.inner_block_counts,
                self.support,
                strict=True,
            )
        ):
            if len(membership) != count or len(blocks) != count:
                raise ValueError("cross-fit layout sample axes are misaligned")
            if not any(membership):
                raise ValueError("every cross-fit outer fold needs fit rows")
            support.validate()
            if support.outer_fold_index != fold or support.fit_row_count != sum(membership):
                raise ValueError("cross-fit support differs from outer membership")
            if self.selection_enabled:
                if block_count < 2 or not support.binary_supported:
                    raise ValueError("cross-fit selection support is insufficient")
                for member, block in zip(membership, blocks, strict=True):
                    if member != (0 <= int(block) < block_count):
                        raise ValueError("cross-fit inner blocks differ from outer membership")
            elif block_count != 0 or any(int(block) != -1 for block in blocks):
                raise ValueError("non-selection cross-fit layouts cannot carry blocks")
        for value in (
            self.block_assignment_hash,
            self.context_digest,
            self.content_digest,
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("cross-fit layout digest is invalid")
        if not self.stratification_rule:
            raise ValueError("cross-fit layout rule must be nonempty")

    @property
    def fold_count(self) -> int:
        return len(self.outer_fit_membership)

    @property
    def sample_count(self) -> int:
        return len(self.context_ids)

    @property
    def maximum_block_count(self) -> int:
        return max(self.inner_block_counts, default=0)

    def outer_fit_tensor(self, *, device: Any | None = None) -> Any:
        _require_torch()
        return torch.as_tensor(
            self.outer_fit_membership, device=device, dtype=torch.float64
        )

    def inner_block_tensor(self, *, device: Any | None = None) -> Any:
        _require_torch()
        return torch.as_tensor(self.inner_block_ids, device=device, dtype=torch.int64)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_name": "mechanistic_cross_fit_layout_v1",
            "context_ids": list(self.context_ids),
            "outer_fit_membership": [
                [bool(value) for value in row] for row in self.outer_fit_membership
            ],
            "inner_block_ids": [list(row) for row in self.inner_block_ids],
            "inner_block_counts": list(self.inner_block_counts),
            "support": [row.to_dict() for row in self.support],
            "requested_block_count": int(self.requested_block_count),
            "selection_enabled": bool(self.selection_enabled),
            "stratification_rule": self.stratification_rule,
            "block_assignment_hash": self.block_assignment_hash,
            "context_digest": self.context_digest,
            "content_digest": self.content_digest,
            "scientific_identity": "excluded",
        }

    @classmethod
    def from_binary_targets(
        cls,
        binary_targets: Any,
        context_ids: Sequence[Any],
        *,
        fold_validity: Any | None = None,
        requested_blocks: int = 4,
    ) -> "CrossFitLayout":
        """Compile fold-local blocks with the existing deterministic rule."""

        target = _binary_target_matrix(binary_targets)
        ids = _ordered_context_ids(context_ids, int(target.shape[0]))
        if int(requested_blocks) < 2:
            raise ValueError("cross-fit selection requires at least two requested blocks")
        if fold_validity is None:
            memberships = torch.ones((1, int(target.shape[0])), dtype=torch.bool)
            assignment, count, rule, block_hash = _binary_stratified_block_assignment(
                target, ids, requested_blocks=int(requested_blocks)
            )
            assignments = assignment.reshape(1, -1)
            counts = (count,)
        else:
            memberships = torch.as_tensor(fold_validity, device="cpu")
            if memberships.ndim != 2 or int(memberships.shape[1]) != len(ids):
                raise ValueError("fold validity must have shape [fold, sample]")
            if not bool(((memberships == 0) | (memberships == 1)).all().item()):
                raise ValueError("fold validity must be binary")
            memberships = memberships.to(torch.bool)
            assignments = torch.full_like(memberships, -1, dtype=torch.int64)
            local_counts: list[int] = []
            fold_rows: list[dict[str, Any]] = []
            rules: list[str] = []
            for fold in range(int(memberships.shape[0])):
                indices = torch.nonzero(memberships[fold], as_tuple=False).reshape(-1)
                if int(indices.numel()) == 0:
                    raise ValueError("every outer calibration fold needs fit contexts")
                local_indices = tuple(int(index) for index in indices.tolist())
                local_ids = tuple(ids[index] for index in local_indices)
                local, local_count, local_rule, local_hash = (
                    _binary_stratified_block_assignment(
                        target.index_select(0, indices),
                        local_ids,
                        requested_blocks=int(requested_blocks),
                    )
                )
                assignments[fold, indices] = local
                local_counts.append(local_count)
                rules.append(local_rule)
                fold_rows.append(
                    {
                        "fold": int(fold),
                        "context_ids": list(local_ids),
                        "local_block_hash": local_hash,
                        "local_block_count": local_count,
                    }
                )
            counts = tuple(local_counts)
            rule = "fold_local_sha256_stratified_calibration_blocks_v1"
            block_hash = stable_digest(
                {
                    "rule": rule,
                    "requested_blocks": int(requested_blocks),
                    "fold_rows": fold_rows,
                    "assignment": assignments.tolist(),
                }
            )
            if not all(value for value in rules):  # pragma: no cover - invariant.
                raise RuntimeError("cross-fit block construction lost a local rule")
        support: list[CrossFitSupport] = []
        for fold in range(int(memberships.shape[0])):
            selected = memberships[fold]
            local_target = target[selected]
            positive = tuple(
                int(value) for value in local_target.sum(dim=0).to(torch.int64).tolist()
            )
            negative = tuple(int(local_target.shape[0]) - value for value in positive)
            support.append(
                CrossFitSupport(
                    outer_fold_index=fold,
                    fit_row_count=int(selected.sum().item()),
                    positive_counts=positive,
                    negative_counts=negative,
                    block_count=int(counts[fold]),
                )
            )
        membership_rows = tuple(
            tuple(bool(value) for value in row.tolist()) for row in memberships
        )
        block_rows = tuple(
            tuple(int(value) for value in row.tolist()) for row in assignments
        )
        context_digest = stable_digest(ids)
        payload = {
            "schema_name": "mechanistic_cross_fit_layout_v1",
            "context_digest": context_digest,
            "target_digest": stable_digest(target.tolist()),
            "outer_fit_membership": [list(row) for row in membership_rows],
            "inner_block_ids": [list(row) for row in block_rows],
            "inner_block_counts": list(counts),
            "support": [row.to_dict() for row in support],
            "requested_block_count": int(requested_blocks),
            "stratification_rule": rule,
            "block_assignment_hash": block_hash,
        }
        layout = cls(
            context_ids=ids,
            outer_fit_membership=membership_rows,
            inner_block_ids=block_rows,
            inner_block_counts=tuple(int(value) for value in counts),
            support=tuple(support),
            requested_block_count=int(requested_blocks),
            selection_enabled=True,
            stratification_rule=rule,
            block_assignment_hash=block_hash,
            context_digest=context_digest,
            content_digest=stable_digest(payload),
        )
        layout.validate()
        return layout

    @classmethod
    def from_outer_fit_membership(
        cls,
        context_ids: Sequence[Any],
        fold_validity: Any,
        *,
        binary_targets: Any | None = None,
    ) -> "CrossFitLayout":
        """Compile a non-selection row-validity layout for joint solvers."""

        _require_torch()
        membership = torch.as_tensor(fold_validity, device="cpu")
        if membership.ndim != 2:
            raise ValueError("fold validity must have shape [fold, sample]")
        ids = _ordered_context_ids(context_ids, int(membership.shape[1]))
        if not bool(((membership == 0) | (membership == 1)).all().item()):
            raise ValueError("fold validity must be binary")
        membership = membership.to(torch.bool)
        target = None if binary_targets is None else _binary_target_matrix(binary_targets)
        if target is not None and int(target.shape[0]) != len(ids):
            raise ValueError("binary targets differ from the outer fit sample axis")
        fit_row_counts = membership.sum(dim=1)
        if bool((fit_row_counts == 0).any().item()):
            raise ValueError("every outer calibration fold needs fit contexts")
        fit_row_count_rows = [int(value) for value in fit_row_counts.tolist()]
        if target is None:
            positive_count_rows = [
                [] for _ in range(int(membership.shape[0]))
            ]
        else:
            # Both inputs convert losslessly to int64, making every batched
            # multiply/add an exact integer count on this CPU path.
            positive_count_rows = (
                membership.to(torch.int64)
                .matmul(target.to(torch.int64))
                .tolist()
            )
        supports: list[CrossFitSupport] = []
        for fold in range(int(membership.shape[0])):
            fit_row_count = fit_row_count_rows[fold]
            positive = tuple(
                int(value) for value in positive_count_rows[fold]
            )
            negative = tuple(
                fit_row_count - value for value in positive
            )
            supports.append(
                CrossFitSupport(
                    outer_fold_index=fold,
                    fit_row_count=fit_row_count,
                    positive_counts=positive,
                    negative_counts=negative,
                    block_count=0,
                )
            )
        membership_values = membership.tolist()
        membership_rows = tuple(
            tuple(bool(value) for value in row) for row in membership_values
        )
        block_rows = tuple(tuple(-1 for _ in ids) for _ in membership_rows)
        context_digest = stable_digest(ids)
        block_hash = stable_digest(
            {
                "rule": "outer_fit_membership_only_v1",
                "context_ids": ids,
                "membership": [list(row) for row in membership_rows],
            }
        )
        payload = {
            "context_digest": context_digest,
            "membership": [list(row) for row in membership_rows],
            "support": [row.to_dict() for row in supports],
            "block_assignment_hash": block_hash,
        }
        layout = cls(
            context_ids=ids,
            outer_fit_membership=membership_rows,
            inner_block_ids=block_rows,
            inner_block_counts=tuple(0 for _ in membership_rows),
            support=tuple(supports),
            requested_block_count=0,
            selection_enabled=False,
            stratification_rule="outer_fit_membership_only_v1",
            block_assignment_hash=block_hash,
            context_digest=context_digest,
            content_digest=stable_digest(payload),
        )
        layout.validate()
        return layout


@runtime_checkable
class CalibrationSolverCapabilityProtocol(Protocol):
    """A math-derived calibration scheduler capability, never model dispatch."""

    capability_id: str
    legal_batch_axes: tuple[str, ...]
    coupled_parameter_axes: bool
    reference_mode: str


@dataclass(frozen=True)
class CalibrationSolverCapability:
    capability_id: str
    legal_batch_axes: tuple[str, ...]
    coupled_parameter_axes: bool
    reference_mode: str

    def validate(self) -> None:
        if not self.capability_id or not self.reference_mode:
            raise ValueError("calibration solver capability identity is incomplete")
        if len(set(self.legal_batch_axes)) != len(self.legal_batch_axes) or any(
            value not in _LEGAL_BATCH_AXES for value in self.legal_batch_axes
        ):
            raise ValueError("calibration solver capability has an invalid batch axis")


def calibration_solver_capability(
    *,
    problem: CalibrationProblemSpec,
    factorization: CalibrationFactorization,
) -> CalibrationSolverCapability:
    """Derive legal execution axes from the declared problem and factor graph."""

    optimization_contract = str(problem.notes.get("optimization_contract", "finite_mle"))
    independent_queries = bool(factorization.is_fully_factorized)
    if optimization_contract == SAMPLED_PATH_V1_CALIBRATION_ALGORITHM:
        result = CalibrationSolverCapability(
            capability_id="historical_c2_serial_compatibility",
            legal_batch_axes=(),
            coupled_parameter_axes=False,
            reference_mode="sealed_serial_compatibility",
        )
    elif optimization_contract == PROPER_SCORE_SHRINKAGE_AFFINE_V2:
        result = CalibrationSolverCapability(
            capability_id="proper_score_positive_affine",
            legal_batch_axes=tuple(
                value
                for value in (
                    "candidate_mask",
                    "outer_fold",
                    "inner_block",
                    "anchor_strength",
                    "independent_query_component" if independent_queries else None,
                )
                if value is not None
            ),
            coupled_parameter_axes=not independent_queries,
            reference_mode="serial_subset_and_scalar_component_reference",
        )
    elif (
        problem.parameterization == "post_query_positive_affine"
        and problem.loss_family == "independent_binary_bce"
    ):
        result = CalibrationSolverCapability(
            capability_id="independent_post_query_positive_affine",
            legal_batch_axes=tuple(
                value
                for value in (
                    "candidate_mask",
                    "outer_fold",
                    "independent_query_component" if independent_queries else None,
                )
                if value is not None
            ),
            coupled_parameter_axes=not independent_queries,
            reference_mode="scalar_component_reference",
        )
    elif (
        problem.parameterization == "shared_temperature"
        and problem.loss_family == "multiclass_cross_entropy"
    ):
        result = CalibrationSolverCapability(
            capability_id="shared_temperature_multiclass",
            legal_batch_axes=("candidate_mask", "outer_fold"),
            coupled_parameter_axes=True,
            reference_mode="serial_subset_temperature_reference",
        )
    elif problem.parameterization == "primitive_diagonal_affine":
        result = CalibrationSolverCapability(
            capability_id="primitive_diagonal_affine_reference_adapter",
            legal_batch_axes=("candidate_mask",),
            coupled_parameter_axes=True,
            reference_mode="serial_outer_fold_reference_pending_parity",
        )
    else:
        result = CalibrationSolverCapability(
            capability_id="unrecognized_calibration_reference_adapter",
            legal_batch_axes=(),
            coupled_parameter_axes=True,
            reference_mode="serial_reference_only",
        )
    result.validate()
    return result


@dataclass(frozen=True)
class CalibrationExecutionTopology:
    """A derived schedule; it cannot introduce a second coupling theory."""

    factorization_hash: str
    layout_digest: str | None
    capability: CalibrationSolverCapability
    coupled_parameter_axis_groups: tuple[tuple[str, ...], ...]
    content_digest: str

    @classmethod
    def derive(
        cls,
        *,
        problem: CalibrationProblemSpec,
        factorization: CalibrationFactorization,
        layout: CrossFitLayout | None = None,
    ) -> "CalibrationExecutionTopology":
        capability = calibration_solver_capability(
            problem=problem, factorization=factorization
        )
        # Each compiler component is an atomic parameter group for scheduling:
        # an independent affine column may batch beside another column, but
        # its own scale and offset never split. A shared-temperature component
        # remains atomic even though its parameter tuple has length one.
        coupled = tuple(
            component.parameter_axis_ids for component in factorization.components
        )
        payload = {
            "schema_name": "mechanistic_calibration_execution_topology_v1",
            "factorization_hash": factorization.factorization_hash,
            "layout_digest": None if layout is None else layout.content_digest,
            "capability_id": capability.capability_id,
            "legal_batch_axes": list(capability.legal_batch_axes),
            "coupled_parameter_axis_groups": [list(row) for row in coupled],
            "reference_mode": capability.reference_mode,
        }
        result = cls(
            factorization_hash=factorization.factorization_hash,
            layout_digest=None if layout is None else layout.content_digest,
            capability=capability,
            coupled_parameter_axis_groups=coupled,
            content_digest=stable_digest(payload),
        )
        result.validate()
        return result

    def validate(self) -> None:
        self.capability.validate()
        if len(self.factorization_hash) != 64 or len(self.content_digest) != 64:
            raise ValueError("calibration execution topology hashes are invalid")
        if self.layout_digest is not None and len(self.layout_digest) != 64:
            raise ValueError("calibration execution layout digest is invalid")
        if any(not row for row in self.coupled_parameter_axis_groups):
            raise ValueError("calibration coupled parameter group cannot be empty")

    def can_batch(self, axis: str) -> bool:
        return str(axis) in self.capability.legal_batch_axes

    def to_runtime_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_name": "mechanistic_calibration_execution_topology_v1",
            "factorization_hash": self.factorization_hash,
            "layout_digest": self.layout_digest,
            "capability_id": self.capability.capability_id,
            "legal_batch_axes": list(self.capability.legal_batch_axes),
            "coupled_parameter_axes": bool(self.capability.coupled_parameter_axes),
            "coupled_parameter_axis_groups": [
                list(row) for row in self.coupled_parameter_axis_groups
            ],
            "reference_mode": self.capability.reference_mode,
            "content_digest": self.content_digest,
            "scientific_identity": "excluded",
        }


def _runtime_tensor_identity(value: Any) -> dict[str, Any]:
    """Identify an in-process tensor without materializing it on the host.

    The identity guards reuse within one execution context only. It is never
    scientific provenance and deliberately includes a storage pointer and
    PyTorch mutation version instead of synchronizing or copying tensor
    contents from an accelerator.
    """

    _require_torch()
    tensor = torch.as_tensor(value)
    return {
        "shape": tuple(int(item) for item in tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": int(tensor.data_ptr()),
        "storage_offset": int(tensor.storage_offset()),
        "version": int(getattr(tensor, "_version", 0)),
    }


def _tensor_structure(value: Any) -> dict[str, Any]:
    """Stable, transfer-free tensor description for a runtime receipt."""

    _require_torch()
    tensor = torch.as_tensor(value)
    return {
        "shape": tuple(int(item) for item in tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _declared_input_provenance(
    declared: Mapping[str, Any] | str | None,
    *,
    query_values: Any,
    targets: Any,
    sample_weights: Any | None,
) -> Mapping[str, Any] | str:
    """Return a receipt-safe provenance declaration without reading tensor data."""

    if declared is not None:
        return declared
    return {
        "kind": "unlabeled_prepared_calibration_input_v1",
        "values": _tensor_structure(query_values),
        "targets": _tensor_structure(targets),
        "weights": None
        if sample_weights is None
        else _tensor_structure(sample_weights),
    }


def _tensor_nbytes(value: Any) -> int:
    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())
    nbytes = getattr(value, "nbytes", None)
    return int(nbytes) if nbytes is not None else 0


@dataclass
class PreparedCalibrationState:
    """One run's device-resident calibration inputs and bounded tensor views."""

    context_ids: tuple[str, ...]
    query_values: Any
    targets: Any
    sample_weights: Any | None
    factorization: CalibrationFactorization
    runtime_profile: RuntimeExecutionProfile
    layout: CrossFitLayout | None = None
    input_provenance: Mapping[str, Any] | str | None = None
    profiler: PhaseProfiler | None = None
    _views: OrderedDict[str, Any] = field(default_factory=OrderedDict, init=False, repr=False)
    _view_bytes: int = field(default=0, init=False, repr=False)
    _provenance_digest: str = field(default="", init=False, repr=False)
    _query_values_identity: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _targets_identity: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _weights_identity: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _input_provenance_digest: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_torch()
        self.runtime_profile.validate()
        if self.profiler is None:
            self.profiler = PhaseProfiler(
                cuda_events_enabled=self.runtime_profile.cuda_events
            )
        values = torch.as_tensor(self.query_values)
        targets = torch.as_tensor(self.targets, device=values.device)
        if values.ndim not in {2, 3}:
            raise ValueError("prepared calibration values must be [sample, query] or [mask, sample, query]")
        sample_count = int(values.shape[-2])
        if int(targets.shape[0]) != sample_count:
            raise ValueError("prepared calibration targets differ from values")
        if self.context_ids and len(self.context_ids) != sample_count:
            raise ValueError("prepared calibration context IDs differ from values")
        if self.layout is not None:
            self.layout.validate()
            if self.layout.context_ids != self.context_ids:
                raise ValueError("prepared calibration layout context IDs differ")
            if self.layout.sample_count != sample_count:
                raise ValueError("prepared calibration layout sample axis differs")
        if self.sample_weights is not None:
            weights = torch.as_tensor(self.sample_weights, device=values.device)
            if int(weights.shape[-1]) != sample_count:
                raise ValueError("prepared calibration weights differ from values")
        declared_provenance = _declared_input_provenance(
            self.input_provenance,
            query_values=values,
            targets=targets,
            sample_weights=self.sample_weights,
        )
        object.__setattr__(self, "query_values", values)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "_query_values_identity", _runtime_tensor_identity(values))
        object.__setattr__(self, "_targets_identity", _runtime_tensor_identity(targets))
        object.__setattr__(
            self,
            "_weights_identity",
            None
            if self.sample_weights is None
            else _runtime_tensor_identity(self.sample_weights),
        )
        # The provenance digest serializes every context ID (256k per
        # terminal chunk) and is read only by runtime receipts, which the
        # terminal bank's per-chunk contexts never emit.  Its inputs are
        # frozen here and it is computed on first read, with the same value.
        object.__setattr__(
            self,
            "_provenance_inputs",
            (
                tuple(self.context_ids),
                copy.deepcopy(declared_provenance)
                if self.input_provenance is not None
                else declared_provenance,
                self.factorization.factorization_hash,
                None if self.layout is None else self.layout.content_digest,
            ),
        )
        object.__setattr__(
            self,
            "_input_provenance_digest",
            stable_digest(declared_provenance),
        )

    @property
    def provenance_digest(self) -> str:
        if not self._provenance_digest:
            context_ids, provenance, factorization_hash, layout_digest = (
                self._provenance_inputs
            )
            object.__setattr__(
                self,
                "_provenance_digest",
                stable_digest(
                    {
                        "context_ids": list(context_ids),
                        "input_provenance": provenance,
                        "factorization_hash": factorization_hash,
                        "layout_digest": layout_digest,
                    }
                ),
            )
        return self._provenance_digest

    @property
    def input_provenance_digest(self) -> str:
        return self._input_provenance_digest

    def matches_inputs(
        self,
        *,
        query_values: Any,
        targets: Any,
        sample_weights: Any | None,
        input_provenance: Mapping[str, Any] | str | None = None,
    ) -> bool:
        """Check same in-process inputs without device-to-host copies."""

        return (
            self._query_values_identity == _runtime_tensor_identity(query_values)
            and self._targets_identity == _runtime_tensor_identity(targets)
            and (
                (self._weights_identity is None and sample_weights is None)
                or (
                    self._weights_identity is not None
                    and sample_weights is not None
                    and self._weights_identity
                    == _runtime_tensor_identity(sample_weights)
                )
            )
            and self.input_provenance_digest
            == stable_digest(
                _declared_input_provenance(
                    input_provenance,
                    query_values=query_values,
                    targets=targets,
                    sample_weights=sample_weights,
                )
            )
        )

    def cached_view(self, key: str, factory: Any) -> Any:
        """Return a bounded run-local view; never cache across sessions."""

        token = str(key)
        if not token:
            raise ValueError("prepared calibration view key must be nonempty")
        existing = self._views.pop(token, None)
        if existing is not None:
            self._views[token] = existing
            return existing
        assert self.profiler is not None
        with self.profiler.phase("calibration.prepared_view"):
            value = factory()
        limit = int(self.runtime_profile.prepared_view_cache_entries)
        if limit <= 0:
            return value
        while len(self._views) >= limit:
            _, evicted = self._views.popitem(last=False)
            self._view_bytes -= _tensor_nbytes(evicted)
        self._views[token] = value
        self._view_bytes += _tensor_nbytes(value)
        return value

    def outer_fold_validity(self) -> Any | None:
        if self.layout is None:
            return None
        return self.cached_view(
            "outer_fold_validity",
            lambda: self.layout.outer_fit_tensor(device=self.query_values.device),
        )

    def inner_block_assignments(self) -> Any | None:
        if self.layout is None or not self.layout.selection_enabled:
            return None
        return self.cached_view(
            "inner_block_assignments",
            lambda: self.layout.inner_block_tensor(device=self.query_values.device),
        )

    def close(self) -> None:
        self._views.clear()
        self._view_bytes = 0

    def to_runtime_dict(self) -> dict[str, Any]:
        assert self.profiler is not None
        return {
            "provenance_digest": self.provenance_digest,
            "input_provenance_digest": self.input_provenance_digest,
            "context_count": len(self.context_ids),
            "device_type": str(self.query_values.device.type),
            "cached_view_count": len(self._views),
            "cached_view_bytes": int(self._view_bytes),
            "layout_digest": None if self.layout is None else self.layout.content_digest,
            "profile": self.runtime_profile.to_runtime_dict(),
            "profile_phases": self.profiler.snapshot(),
            "scientific_identity": "excluded",
        }


@dataclass
class CalibrationExecutionContext:
    """Run-scoped coordinator for one calibration backend invocation family."""

    runtime_profile: RuntimeExecutionProfile
    prepared_state: PreparedCalibrationState | None = None
    topology: CalibrationExecutionTopology | None = None
    profiler: PhaseProfiler | None = None
    solver_calls: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.runtime_profile.validate()
        if self.profiler is None:
            self.profiler = PhaseProfiler(
                cuda_events_enabled=self.runtime_profile.cuda_events
            )

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "CalibrationExecutionContext":
        return cls(RuntimeExecutionProfile.from_environment(environ))

    def bind(
        self,
        *,
        factorization: CalibrationFactorization,
        query_values: Any,
        targets: Any,
        sample_weights: Any | None,
        context_ids: Sequence[Any] | None,
        layout: CrossFitLayout | None,
        problem: CalibrationProblemSpec,
        input_provenance: Mapping[str, Any] | str | None = None,
    ) -> PreparedCalibrationState:
        ids = tuple(str(value) for value in (context_ids or ()))
        if self.prepared_state is None:
            assert self.profiler is not None
            with self.profiler.phase("calibration.prepare_state"):
                self.prepared_state = PreparedCalibrationState(
                    context_ids=ids,
                    query_values=query_values,
                    targets=targets,
                    sample_weights=sample_weights,
                    factorization=factorization,
                    runtime_profile=self.runtime_profile,
                    layout=layout,
                    input_provenance=input_provenance,
                )
            self.topology = CalibrationExecutionTopology.derive(
                problem=problem, factorization=factorization, layout=layout
            )
            return self.prepared_state
        state = self.prepared_state
        if state.factorization.factorization_hash != factorization.factorization_hash:
            raise ValueError("execution context factorization differs from its prepared state")
        if tuple(state.context_ids) != ids:
            raise ValueError("execution context context IDs differ from its prepared state")
        if not state.matches_inputs(
            query_values=query_values,
            targets=targets,
            sample_weights=sample_weights,
            input_provenance=input_provenance,
        ):
            raise ValueError(
                "execution context inputs or provenance differ from its prepared state"
            )
        if (state.layout is None) != (layout is None) or (
            state.layout is not None
            and layout is not None
            and state.layout.content_digest != layout.content_digest
        ):
            raise ValueError("execution context cross-fit layout differs from its prepared state")
        return state

    def record_solver_call(self, solver_id: str) -> None:
        token = str(solver_id).strip()
        if not token:
            raise ValueError("calibration solver ID must be nonempty")
        self.solver_calls[token] = int(self.solver_calls.get(token, 0)) + 1

    def record_phase(self, name: str, elapsed_seconds: float) -> None:
        assert self.profiler is not None
        self.profiler.record(name, elapsed_seconds)

    def runtime_receipt(self) -> dict[str, Any]:
        assert self.profiler is not None
        return {
            "schema_name": "mechanistic_calibration_execution_receipt_v1",
            "profile": self.runtime_profile.to_runtime_dict(),
            "prepared_state": None
            if self.prepared_state is None
            else self.prepared_state.to_runtime_dict(),
            "topology": None
            if self.topology is None
            else self.topology.to_runtime_dict(),
            "solver_calls": {
                key: int(value) for key, value in sorted(self.solver_calls.items())
            },
            "phase_profile": self.profiler.snapshot(),
            "scientific_identity": "excluded",
        }


@dataclass
class RuntimeExecutionSession:
    """Coordinate separate candidate and calibration runtimes for one run."""

    runtime_profile: RuntimeExecutionProfile
    candidate_backend: Any | None = None
    calibration_context_count: int = field(default=0, init=False)
    calibration_receipt_digests: list[str] = field(default_factory=list, init=False)
    calibration_solver_calls: dict[str, int] = field(default_factory=dict, init=False)
    calibration_profiler: PhaseProfiler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.runtime_profile.validate()
        self.calibration_profiler = PhaseProfiler(
            cuda_events_enabled=self.runtime_profile.cuda_events
        )

    def new_calibration_context(self) -> CalibrationExecutionContext:
        """Create one exact-input context under this run's immutable profile.

        A prepared state is intentionally scoped to one calibration input
        tensor. Candidate masks may therefore obtain separate contexts while
        still sharing one typed runtime profile and one aggregate receipt.
        """

        return CalibrationExecutionContext(self.runtime_profile)

    def record_calibration_context(self, context: CalibrationExecutionContext) -> None:
        """Retain bounded operational evidence without a cross-run cache."""

        receipt = context.runtime_receipt()
        self.calibration_context_count += 1
        for solver_id, count in receipt["solver_calls"].items():
            self.calibration_solver_calls[str(solver_id)] = (
                int(self.calibration_solver_calls.get(str(solver_id), 0))
                + int(count)
            )
        for phase_snapshot in (
            receipt["phase_profile"],
            (
                {} if receipt["prepared_state"] is None
                else receipt["prepared_state"]["profile_phases"]
            ),
        ):
            phase_times = phase_snapshot.get("phase_time_s", {})
            phase_calls = phase_snapshot.get("phase_calls", {})
            for name, elapsed in phase_times.items():
                self.calibration_profiler.record(
                    str(name),
                    float(elapsed),
                    calls=int(phase_calls.get(name, 1)),
                )
        self.calibration_receipt_digests.append(stable_digest(receipt))
        if len(self.calibration_receipt_digests) > 32:
            del self.calibration_receipt_digests[:-32]

    def runtime_receipt(self) -> dict[str, Any]:
        candidate_profile = getattr(self.candidate_backend, "runtime_profile", None)
        candidate = (
            candidate_profile().to_runtime_dict()
            if callable(candidate_profile)
            else None
        )
        return {
            "schema_name": "mechanistic_runtime_execution_session_v1",
            "profile": self.runtime_profile.to_runtime_dict(),
            "calibration": {
                "context_count": int(self.calibration_context_count),
                "recent_receipt_digests": list(self.calibration_receipt_digests),
                "solver_calls": {
                    key: int(value)
                    for key, value in sorted(self.calibration_solver_calls.items())
                },
                "phase_profile": self.calibration_profiler.snapshot(),
            },
            "candidate_execution": candidate,
            "scientific_identity": "excluded",
        }


__all__ = [
    "CalibrationExecutionContext",
    "CalibrationExecutionTopology",
    "CalibrationSolverCapability",
    "CalibrationSolverCapabilityProtocol",
    "CrossFitLayout",
    "CrossFitSupport",
    "PreparedCalibrationState",
    "RuntimeExecutionProfile",
    "RuntimeExecutionSession",
    "calibration_solver_capability",
]
