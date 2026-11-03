from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch


@dataclass(frozen=True)
class EffectPlan:
    layer: str
    cluster_id: int
    mask_path: Path


def build_effect_masks(
    layer: str,
    assignment_path: str | Path,
    param_shape: torch.Size,
    output_dir: str | Path,
) -> list[EffectPlan]:
    assignments = torch.load(assignment_path, map_location="cpu").view(-1).to(torch.long)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plans: list[EffectPlan] = []
    for cid in sorted(int(v) for v in assignments.unique().tolist()):
        mask = torch.zeros(int(assignments.numel()), dtype=torch.bool)
        mask[assignments == cid] = True
        mask = mask.reshape(param_shape)
        mask_path = out / f"cluster_{cid:04d}" / "mask.pt"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mask, mask_path)
        plans.append(EffectPlan(layer=layer, cluster_id=cid, mask_path=mask_path))
    return plans


def execute_effect_plans(
    plans: Iterable[EffectPlan],
    evaluator: Callable[[EffectPlan], object],
    *,
    workers: int = 1,
) -> None:
    plan_list = list(plans)
    if workers <= 1:
        for plan in plan_list:
            evaluator(plan)
        return
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        list(pool.map(evaluator, plan_list))


__all__ = ["EffectPlan", "build_effect_masks", "execute_effect_plans"]
