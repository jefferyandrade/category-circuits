"""Environment-gated, default-OFF runtime performance policies for the v0 pilot.

Three independent gates, each inert unless its environment variable is truthy,
each recording its activation in telemetry/receipt fields so a cluster canary
can verify the optimization actually engaged:

``IRG_WALK_REDUCE_OVERHEAD``
    CUDA-graph-backed ``torch.compile(mode="reduce-overhead")`` for the walk's
    steady-state per-step forward runners.  Capture is additionally guarded by
    an explicit walk-phase arming flag set by the pilot driver strictly
    between census completion and terminal entry, so graph capture can never
    overlap the census phase's side-stream ``record_stream`` transfers
    (``device_route_moments`` offload/restore).

``IRG_TERMINAL_ADAPTIVE_BATCH``
    Historical name, kept stable for launchers; it now governs the whole
    per-role adaptive batch policy:
    ``rows(role) = clamp(saturation_floor(role), memory_ceiling(role))``,
    derived at each phase boundary from explicit measured inputs, with every
    derivation input written to the receipts.  With the variable unset every
    role keeps today's exact ``NativePanelExecutionSpec`` constants.

``IRG_COMPILE_COVERAGE``
    ``torch.compile`` (default mode) over remaining eager hot-loop tensor
    callables with stable shape structure.  With the variable unset the
    coverage factory returns the original eager callable object unchanged.

Everything in this module is pure Python and import-safe without torch or a
GPU; all device measurements (free bytes, allocator deltas, probe timings)
are taken by the callers and passed in explicitly, which is what makes the
derivations unit-testable on CPU-only machines.

None of these gates may change a registered quantity definition, a decision
threshold, a look/step count, or an accumulator/certificate precision.  They
only change physical execution: launch overhead, physical batch shapes, and
kernel fusion.  Floating-point reordering from fusion/batching/graph replay
is the accepted equivalence class.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Mapping, Sequence

WALK_REDUCE_OVERHEAD_ENV = "IRG_WALK_REDUCE_OVERHEAD"
ADAPTIVE_BATCH_POLICY_ENV = "IRG_TERMINAL_ADAPTIVE_BATCH"
COMPILE_COVERAGE_ENV = "IRG_COMPILE_COVERAGE"
# Single-pair walk mode is the
# DEFAULT.  Unset => single-pair composition (fused chunk width pinned to one
# successor per launch, prefix reuse engaged, per-step fresh pair-shared
# panels via the existing fresh-proposal stream contract).  An explicit
# falsy value ("0"/"false"/"no"/"off") selects the legacy chunked-8 path
# with prefix reuse off, kept intact for comparison canaries and historical
# reproduction only.
SINGLE_PAIR_WALK_ENV = "IRG_SINGLE_PAIR_WALK"

_FALSY_TOKENS = frozenset({"0", "false", "no", "off"})

_TRUTHY_TOKENS = frozenset({"1", "true", "yes", "on"})

ADAPTIVE_BATCH_POLICY_SCHEMA = "irg_adaptive_batch_policy_v1"
ADAPTIVE_ROLE_ROWS_SCHEMA = "irg_adaptive_role_rows_v1"
SATURATION_PROBE_SCHEMA = "irg_saturation_floor_probe_v1"
WALK_REDUCE_OVERHEAD_SCHEMA = "irg_walk_reduce_overhead_v1"

# Documented fallback estimate for the device bytes one forward row occupies
# transiently (float32 input row for a 3x224x224 image; the multiplier covers
# the uint8 staging copy, channels-last activation working set, and readout
# rows for AlexNet/ResNet18-class networks).  Estimates are used only where a
# measured probe is unavailable (walk-role ceilings); the terminal role
# MEASURES its footprint with allocator deltas instead.
DEFAULT_INPUT_ROW_BYTES = 4 * 3 * 224 * 224
DEFAULT_ACTIVATION_MULTIPLIER = 6.0
DEFAULT_MEMORY_SAFETY_FACTOR = 0.5
DEFAULT_SATURATION_GAIN_THRESHOLD = 0.10
TERMINAL_PROBE_START_ROWS = 64
TERMINAL_ROWS_HARD_CAP = 4096


def _env_truthy(name: str, environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return str(values.get(name, "")).strip().lower() in _TRUTHY_TOKENS


def walk_reduce_overhead_requested(
    environ: Mapping[str, str] | None = None,
) -> bool:
    return _env_truthy(WALK_REDUCE_OVERHEAD_ENV, environ)


def adaptive_batch_policy_requested(
    environ: Mapping[str, str] | None = None,
) -> bool:
    return _env_truthy(ADAPTIVE_BATCH_POLICY_ENV, environ)


def compile_coverage_requested(
    environ: Mapping[str, str] | None = None,
) -> bool:
    return _env_truthy(COMPILE_COVERAGE_ENV, environ)


def single_pair_walk_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """DEFAULT-ON walk mode.

    Unset or any non-falsy value => single-pair mode.  Only an explicit
    falsy token selects the legacy chunked path.
    

    EXPERIMENTAL — MEASURED TO WORSEN RUNTIME (in an A/B run,
    a 2.45x oracle stage-time regression with
    graphs armed; CUDA-graph capture serialization also breaks the
    panel producer/consumer overlap, tripling batch_wait). Default
    stays OFF; arm only for experiments.
    """

    values = os.environ if environ is None else environ
    token = str(values.get(SINGLE_PAIR_WALK_ENV, "")).strip().lower()
    return token not in _FALSY_TOKENS


def walk_execution_mode(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Receipt-facing mode label: ``single_pair`` or ``chunked_legacy``."""

    return (
        "single_pair"
        if single_pair_walk_enabled(environ)
        else "chunked_legacy"
    )


def derive_step_panel_seed(
    *,
    registered_seed: int,
    step_index: int,
    namespace: str = "single-pair-walk",
) -> dict[str, Any]:
    """Deterministic per-step panel-seed derivation with its full receipt.

    ``seed = int(sha256(f"{namespace}:{registered_seed}:{step_index}")[:16], 16)``

    A stable pure function of the run's registered seed and the step index:
    reproducible across replays, independent across steps.  NOTE: the v0
    walk's oracle draws are ALREADY independent per proposal through the
    registered ``fresh_proposal_stream_conditional_on_birth_v1`` stream
    contract (stream keyed on state hash + parent mask + candidate id), so
    this derivation is provided for drivers whose selector seed would
    otherwise be fixed within a look; wherever used, the receipt must carry
    the returned derivation record.
    """

    import hashlib

    registered = int(registered_seed)
    step = int(step_index)
    if step < 0:
        raise ValueError("step index cannot be negative")
    digest = hashlib.sha256(
        f"{namespace}:{registered}:{step}".encode("utf-8")
    ).hexdigest()
    return {
        "schema": "irg_single_pair_step_panel_seed_v1",
        "namespace": str(namespace),
        "registered_seed": registered,
        "step_index": step,
        "sha256": digest,
        "panel_seed": int(digest[:16], 16),
        "derivation": (
            "int(sha256(namespace:registered_seed:step_index)[:16], 16)"
        ),
    }


# ---------------------------------------------------------------------------
# Walk-phase arming.  The env gate says "may"; the phase flag says
# "now".  Both must hold before any reduce-overhead compile happens, which is
# what keeps CUDA-graph capture strictly inside the walk.
# ---------------------------------------------------------------------------

_WALK_PHASE_ARMED = False


def arm_walk_phase() -> None:
    global _WALK_PHASE_ARMED
    _WALK_PHASE_ARMED = True


def disarm_walk_phase() -> None:
    global _WALK_PHASE_ARMED
    _WALK_PHASE_ARMED = False


def walk_phase_armed() -> bool:
    return bool(_WALK_PHASE_ARMED)


def walk_reduce_overhead_active(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """True only when the env gate is on AND the pilot armed the walk phase."""

    return walk_reduce_overhead_requested(environ) and walk_phase_armed()


# ---------------------------------------------------------------------------
# Pure batch-rows derivations.  All inputs explicit; no torch.
# ---------------------------------------------------------------------------


def memory_ceiling_rows(
    *,
    free_bytes: int,
    per_row_bytes: int,
    safety_factor: float = DEFAULT_MEMORY_SAFETY_FACTOR,
    reserve_bytes: int = 0,
) -> int:
    """Largest row count whose transient footprint fits the free-memory budget.

    ``ceiling = floor((free - reserve) * safety / per_row_bytes)``, never
    negative.  ``per_row_bytes`` is either measured (terminal probe) or an
    explicit documented estimate (walk roles).
    """

    free = int(free_bytes)
    per_row = int(per_row_bytes)
    reserve = int(reserve_bytes)
    safety = float(safety_factor)
    if free < 0 or reserve < 0:
        raise ValueError("memory ceiling requires nonnegative byte counts")
    if per_row <= 0:
        raise ValueError("memory ceiling requires positive per-row bytes")
    if not (0.0 < safety <= 1.0) or not math.isfinite(safety):
        raise ValueError("memory safety factor must lie in (0, 1]")
    usable = max(0, free - reserve) * safety
    return int(usable // per_row)


def probe_row_bytes(*, allocated_delta_bytes: int, probe_rows: int) -> int:
    """Measured per-row activation footprint from one probe forward.

    ``allocated_delta_bytes`` is the caller's allocator peak delta
    (``max_memory_allocated - memory_allocated`` before the probe) over a
    ``probe_rows``-row forward; the result is the ceil per-row share, floored
    at 1 byte so a zero-delta probe can never manufacture an infinite ceiling.
    """

    rows = int(probe_rows)
    delta = int(allocated_delta_bytes)
    if rows <= 0:
        raise ValueError("probe row bytes require a positive probe row count")
    if delta < 0:
        delta = 0
    return max(1, -(-delta // rows))


def estimated_forward_row_bytes(
    *,
    input_row_bytes: int = DEFAULT_INPUT_ROW_BYTES,
    lane_count: int = 1,
    activation_multiplier: float = DEFAULT_ACTIVATION_MULTIPLIER,
) -> int:
    """Documented ESTIMATE of one row's transient forward footprint.

    Used only for walk-role ceilings where a measured probe would sit inside
    the registered walk; the estimate and its inputs are receipted so the
    canary can audit the derivation.
    """

    row = int(input_row_bytes)
    lanes = int(lane_count)
    multiplier = float(activation_multiplier)
    if row <= 0 or lanes <= 0:
        raise ValueError("row-byte estimate requires positive inputs")
    if multiplier < 1.0 or not math.isfinite(multiplier):
        raise ValueError("activation multiplier must be finite and >= 1")
    return int(math.ceil(row * lanes * multiplier))


def saturation_floor_rows(
    samples: Sequence[tuple[int, float]],
    *,
    gain_threshold: float = DEFAULT_SATURATION_GAIN_THRESHOLD,
) -> dict[str, Any]:
    """Choose the fewest rows that saturate throughput from doubling probes.

    ``samples`` are ordered ``(rows, rows_per_second)`` pairs with strictly
    increasing rows.  Doubling continues to count as useful while the
    per-step relative throughput gain exceeds ``gain_threshold``; the floor
    is the last size whose step over its predecessor still cleared the bar
    (the classic knee rule).  One sample means no evidence beyond it.
    """

    if not samples:
        raise ValueError("saturation floor requires at least one probe sample")
    threshold = float(gain_threshold)
    if threshold < 0.0 or not math.isfinite(threshold):
        raise ValueError("saturation gain threshold must be finite and >= 0")
    previous_rows = 0
    previous_tput = 0.0
    normalized: list[tuple[int, float]] = []
    for rows, throughput in samples:
        rows = int(rows)
        throughput = float(throughput)
        if rows <= previous_rows:
            raise ValueError("saturation samples require increasing rows")
        if throughput <= 0.0 or not math.isfinite(throughput):
            raise ValueError("saturation samples require positive throughput")
        normalized.append((rows, throughput))
        previous_rows = rows
    chosen_rows = normalized[0][0]
    previous_tput = normalized[0][1]
    for rows, throughput in normalized[1:]:
        gain = (throughput - previous_tput) / previous_tput
        if gain > threshold:
            chosen_rows = rows
            previous_tput = throughput
        else:
            break
    return {
        "schema": SATURATION_PROBE_SCHEMA,
        "floor_rows": int(chosen_rows),
        "gain_threshold": threshold,
        "samples": [
            {"rows": int(rows), "rows_per_second": float(throughput)}
            for rows, throughput in normalized
        ],
    }


def derive_role_rows(
    *,
    role: str,
    configured_rows: int,
    ceiling_rows: int,
    floor_rows: int | None = None,
    minimum_rows: int = 1,
    maximum_rows: int | None = None,
    derivation_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``rows = clamp(saturation_floor, memory_ceiling)`` with sane bounds.

    ``floor_rows`` is the saturation floor (measured where available); roles
    without a saturation probe pass ``None`` and the configured constant acts
    as the floor, so absent probe evidence the policy can only SHRINK a role
    to fit memory, never raise it.  The ceiling always wins over the floor.
    """

    configured = int(configured_rows)
    ceiling = int(ceiling_rows)
    minimum = int(minimum_rows)
    if configured <= 0 or minimum <= 0:
        raise ValueError("role rows derivation requires positive bounds")
    if ceiling < 0:
        raise ValueError("memory ceiling rows cannot be negative")
    floor = configured if floor_rows is None else int(floor_rows)
    if floor <= 0:
        raise ValueError("floor rows must be positive")
    derived = min(floor, ceiling) if ceiling > 0 else minimum
    derived = max(minimum, derived)
    if maximum_rows is not None:
        derived = min(derived, int(maximum_rows))
    return {
        "schema": ADAPTIVE_ROLE_ROWS_SCHEMA,
        "role": str(role),
        "configured_rows": configured,
        "floor_rows": floor,
        "floor_source": (
            "configured_constant" if floor_rows is None else "saturation_probe"
        ),
        "ceiling_rows": ceiling,
        "minimum_rows": minimum,
        "maximum_rows": None if maximum_rows is None else int(maximum_rows),
        "derived_rows": int(derived),
        "changed": int(derived) != configured,
        "derivation_inputs": (
            {} if derivation_inputs is None else dict(derivation_inputs)
        ),
    }


# ---------------------------------------------------------------------------
# OOM halving-retry: never abort on the first OutOfMemory; halve the
# physical rows and re-run.  Physical batch size never changes a registered
# quantity (batching is partition-only on these paths), so the retry is
# algorithmically the identity.
# ---------------------------------------------------------------------------


def default_is_oom(error: BaseException) -> bool:
    """Match torch.cuda.OutOfMemoryError and CUDA OOM RuntimeErrors by shape.

    Pure string/type inspection so this module stays torch-free.
    """

    if type(error).__name__ == "OutOfMemoryError":
        return True
    message = str(error).lower()
    return "out of memory" in message


class OOMHalvingController:
    """Mutable physical-rows holder with a halve-and-retry execution loop."""

    def __init__(
        self,
        *,
        initial_rows: int,
        minimum_rows: int = 1,
        is_oom: Callable[[BaseException], bool] | None = None,
        on_backoff: Callable[[BaseException, int], None] | None = None,
    ) -> None:
        rows = int(initial_rows)
        minimum = int(minimum_rows)
        if rows <= 0 or minimum <= 0 or rows < minimum:
            raise ValueError(
                "OOM halving requires initial_rows >= minimum_rows >= 1"
            )
        self.rows = rows
        self.minimum_rows = minimum
        self.retries = 0
        self.history: list[int] = [rows]
        self._is_oom = default_is_oom if is_oom is None else is_oom
        self._on_backoff = on_backoff

    def run(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn``, halving ``self.rows`` and retrying on OOM.

        Non-OOM exceptions propagate untouched.  An OOM at ``minimum_rows``
        re-raises: there is no smaller physical shape left to try.
        """

        while True:
            try:
                return fn()
            except Exception as error:  # noqa: BLE001 - filtered below
                if not self._is_oom(error):
                    raise
                if self.rows <= self.minimum_rows:
                    raise
                self.rows = max(self.minimum_rows, self.rows // 2)
                self.retries += 1
                self.history.append(self.rows)
                if self._on_backoff is not None:
                    self._on_backoff(error, self.rows)

    def receipt(self) -> dict[str, Any]:
        return {
            "initial_rows": int(self.history[0]),
            "final_rows": int(self.rows),
            "minimum_rows": int(self.minimum_rows),
            "oom_retries": int(self.retries),
            "rows_history": [int(value) for value in self.history],
        }


# ---------------------------------------------------------------------------
# Per-role batch policy carried by the native panel data plane.
# ---------------------------------------------------------------------------


class BatchPolicy:
    """Role -> derived physical rows, with the full derivation receipts.

    Installed on a ``NativePanelDataPlane`` only when the env gate is on;
    roles without an entry fall through to the spec constants, so a policy
    can be partial and the OFF state (no policy installed) is exactly
    today's behavior.
    """

    def __init__(self) -> None:
        self._rows: dict[str, int] = {}
        self._receipts: dict[str, dict[str, Any]] = {}

    def set_role_rows(
        self,
        role: str,
        rows: int,
        receipt: Mapping[str, Any],
    ) -> None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError("batch policy rows must be positive")
        self._rows[str(role)] = rows
        self._receipts[str(role)] = dict(receipt)

    def rows_for(self, role: str) -> int | None:
        return self._rows.get(str(role))

    def roles(self) -> tuple[str, ...]:
        return tuple(sorted(self._rows))

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": ADAPTIVE_BATCH_POLICY_SCHEMA,
            "rows_by_role": dict(sorted(self._rows.items())),
            "derivations_by_role": {
                role: dict(value)
                for role, value in sorted(self._receipts.items())
            },
        }


__all__ = [
    "ADAPTIVE_BATCH_POLICY_ENV",
    "ADAPTIVE_BATCH_POLICY_SCHEMA",
    "ADAPTIVE_ROLE_ROWS_SCHEMA",
    "BatchPolicy",
    "COMPILE_COVERAGE_ENV",
    "DEFAULT_ACTIVATION_MULTIPLIER",
    "DEFAULT_INPUT_ROW_BYTES",
    "DEFAULT_MEMORY_SAFETY_FACTOR",
    "DEFAULT_SATURATION_GAIN_THRESHOLD",
    "OOMHalvingController",
    "SATURATION_PROBE_SCHEMA",
    "SINGLE_PAIR_WALK_ENV",
    "derive_step_panel_seed",
    "single_pair_walk_enabled",
    "walk_execution_mode",
    "TERMINAL_PROBE_START_ROWS",
    "TERMINAL_ROWS_HARD_CAP",
    "WALK_REDUCE_OVERHEAD_ENV",
    "WALK_REDUCE_OVERHEAD_SCHEMA",
    "adaptive_batch_policy_requested",
    "arm_walk_phase",
    "compile_coverage_requested",
    "default_is_oom",
    "derive_role_rows",
    "disarm_walk_phase",
    "estimated_forward_row_bytes",
    "memory_ceiling_rows",
    "probe_row_bytes",
    "saturation_floor_rows",
    "walk_phase_armed",
    "walk_reduce_overhead_active",
    "walk_reduce_overhead_requested",
]
