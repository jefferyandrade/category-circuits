"""Integer rung geometry shared by census cuts and frontier decision bounds.

Use the lowerer's retained counts when a rung rule specifies counts directly;
use ``retained_counts_for_density_schedule`` for ordinary density lowering.
The boundary fractions recover precisely the nontrivial count-changing cuts
under ceil(fraction * partition size). Empty partitions have no cuts. Invalid
or increasing counts raise rather than produce a partially audited schedule.
CPU fixtures check direct ceil arithmetic, no-ops, empty partitions, and caps.
This is engine arithmetic, not a standalone research-toolkit instrument.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .masking import _keep_count


def retained_counts_for_density_schedule(
    partition_route_counts: Mapping[str, int],
    density_schedule: Sequence[float],
) -> dict[str, tuple[int, ...]]:
    """Apply the walk's exact retained-count rule to each rung and partition."""

    return {
        str(partition): tuple(_keep_count(int(size), float(density)) for density in density_schedule)
        for partition, size in partition_route_counts.items()
    }


def _validated_retained_counts(values: Sequence[int]) -> tuple[int, ...]:
    counts = tuple(int(value) for value in values)
    if not counts or any(value < 0 for value in counts) or any(
        later > earlier for earlier, later in zip(counts, counts[1:])
    ):
        raise ValueError("rung retained counts must be nonempty, nonnegative and nested")
    return counts


def exact_ceil_fraction(count: int, size: int) -> float:
    """Translate an interior retained count without rounding it up a route."""

    retained, subset = int(count), int(size)
    if not 1 <= retained <= subset - 1:
        raise ValueError("exact ceil fraction requires an interior count")
    fraction = retained / subset
    if 0.0 < fraction < 1.0 and math.ceil(fraction * subset) == retained:
        return float(fraction)
    fraction = (retained - 0.5) / subset
    if math.ceil(fraction * subset) != retained:
        raise ValueError("ceil-exact fraction translation failed")
    return float(fraction)


def boundary_fractions_from_retained_counts(
    retained_counts_by_partition: Mapping[str, Sequence[int]],
) -> dict[str, tuple[float, ...]]:
    """Every executable score-dependent cut, with density-one/no-op cuts removed."""

    result = {}
    for partition, values in retained_counts_by_partition.items():
        counts = _validated_retained_counts(values)
        size = counts[0]
        result[str(partition)] = tuple(
            exact_ceil_fraction(later, size)
            for earlier, later in zip(counts, counts[1:])
            if later < earlier and 0 < later < size
        )
    return result


def maximum_decisions_from_retained_counts(
    retained_counts_by_partition: Mapping[str, Sequence[int]],
) -> int:
    """Decision cap: twice the sum of executable partition steps, plus two."""

    executable_steps = 0
    for values in retained_counts_by_partition.values():
        counts = _validated_retained_counts(values)
        executable_steps += sum(
            later < earlier for earlier, later in zip(counts, counts[1:])
        )
    return 2 * int(executable_steps) + 2
