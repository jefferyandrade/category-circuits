"""Utilities that turn streamed statistics into clustering/pruning signals."""
from __future__ import annotations

import os
import math
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F

EPS = 1e-8


def build_alignment(summary: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Construct a correlation-style alignment matrix from row summaries."""

    outer_sum = summary.get("outer_sum")
    sum_vec = summary.get("sum")
    count = float(summary.get("count", torch.tensor(0.0)))
    if outer_sum is None or sum_vec is None or count <= 1:
        return torch.zeros(0, 0)

    outer = outer_sum.to(torch.float64)
    mean = sum_vec.to(torch.float64) / count
    cov = outer - count * mean.unsqueeze(1) * mean.unsqueeze(0)
    diag = cov.diagonal().clamp(min=EPS)
    denom = torch.sqrt(torch.outer(diag, diag)) + EPS
    alignment = (cov / denom).to(torch.float32)
    return alignment.clamp(-1.0, 1.0)


def build_synergy(sketch: Dict[str, object], rows: int) -> torch.Tensor:
    """Approximate a row-level synergy matrix from HVP sketches."""

    if not sketch or rows == 0:
        return torch.zeros(rows, rows)
    vectors: Sequence[torch.Tensor] = (
        sketch.get("weight_vectors")
        or sketch.get("vectors", [])
    )
    responses: Sequence[torch.Tensor] = (
        sketch.get("weight_responses")
        or sketch.get("responses", [])
    )
    if not vectors or not responses:
        return torch.zeros(rows, rows)

    synergy = torch.zeros((rows, rows), dtype=torch.float32)
    for vec, resp in zip(vectors, responses):
        v = vec.to(torch.float32).view(rows, -1)
        r = resp.to(torch.float32).view(rows, -1)
        synergy += r @ v.t()
    synergy = 0.5 * (synergy + synergy.t())
    diag = synergy.diagonal().abs().clamp(min=EPS)
    denom = torch.sqrt(torch.outer(diag, diag)) + EPS
    normalized = synergy / denom
    return normalized.clamp(-1.0, 1.0)


def build_commensurability(alignment: torch.Tensor, synergy: torch.Tensor) -> torch.Tensor:
    if alignment.numel() == 0:
        return alignment
    return alignment.abs() * torch.exp(-synergy.abs())


def _gather_low_rank_features(summary: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """
    Collect a compact feature set for low-rank clustering. These features are
    simple per-unit statistics that already exist in the streaming summary and
    therefore scale to tens of millions of rows without materialising dense
    pairwise matrices.
    """

    candidate_keys = [
        "mean_abs",
        "S1",
        "Ns",
        "sum",
        "sum_sq",
        "target_sum",
        "target_abs_sum",
        "diff_abs",
    ]
    features: List[torch.Tensor] = []
    rows = None
    for key in candidate_keys:
        tensor = summary.get(key)
        if tensor is None:
            continue
        if tensor.ndim != 1:
            continue
        tensor = tensor.to(torch.float32)
        if rows is None:
            rows = tensor.numel()
        elif tensor.numel() != rows:
            continue
        features.append(tensor)
    return features


def _build_low_rank_embedding(
    features: List[torch.Tensor],
    rank: int,
    *,
    seed: int,
) -> torch.Tensor:
    device = features[0].device
    rows = features[0].numel()
    components = len(features)
    rank = max(1, min(rank, components))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    projection = torch.randn((components, rank), generator=generator, device=device)
    embedding = torch.zeros((rows, rank), dtype=torch.float32, device=device)

    for idx, feature in enumerate(features):
        mean = feature.mean()
        std = feature.std(unbiased=False)
        if std <= 1e-6:
            continue
        normalized = (feature - mean) / (std + 1e-6)
        embedding += normalized.unsqueeze(1) * projection[idx].unsqueeze(0)

    return embedding


def _cluster_low_rank(
    summary: Dict[str, torch.Tensor],
    rank: int,
) -> Tuple[List[List[int]], List[List[float]], List[List[int]]]:
    features = _gather_low_rank_features(summary)
    if not features:
        return [], [], []

    seed = int(os.environ.get("SCORING_CLUSTER_LOW_RANK_SEED", "0"))
    embedding = _build_low_rank_embedding(features, rank, seed=seed)

    rows, components = embedding.shape
    max_members = int(os.environ.get("SCORING_CLUSTER_MAX_MEMBERS", "2048"))
    max_members = max(1, min(max_members, rows))
    guard_select = torch.zeros(rows, dtype=torch.bool, device=embedding.device)

    clusters: List[List[int]] = []
    alphas: List[List[float]] = []

    for comp in range(components):
        scores = embedding[:, comp].abs().clone()
        scores[guard_select] = 0.0
        if torch.all(scores <= 0):
            continue
        take = min(max_members, rows - guard_select.sum().item())
        values, indices = torch.topk(scores, take)
        mask = values > 0
        values = values[mask]
        indices = indices[mask]
        if indices.numel() == 0:
            continue
        clusters.append(indices.cpu().tolist())
        weights = values / (values.sum().item() + EPS)
        alphas.append(weights.cpu().tolist())
        guard_select[indices] = True
        if guard_select.all():
            break

    prune_order = [cluster[:] for cluster in clusters]
    return clusters, alphas, prune_order


def _gaussian_mutual_information(summary: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Approximate the mutual information between each frontier row and the target
    response under a Gaussian assumption.

    The streaming statistics expose sufficient moments to evaluate
    ``I(phi_i; y) = -0.5 * log(1 - rho_i^2)`` where ``rho_i`` is the Pearson
    correlation between feature ``phi_i`` and the target.  When target moments
    are unavailable, falls back to the mean absolute activation magnitude.
    """

    Ns = summary.get("Ns")
    sum_phi = summary.get("sum")
    sum_phi_sq = summary.get("sum_sq")
    sum_target = summary.get("target_sum")
    sum_target_sq = summary.get("target_sum_sq")
    cross_sum = summary.get("cross_sum")

    if (
        Ns is None
        or sum_phi is None
        or sum_phi_sq is None
        or sum_target is None
        or sum_target_sq is None
        or cross_sum is None
    ):
        proxy = summary.get("mean_abs")
        if proxy is None:
            return torch.zeros(0)
        return proxy.to(torch.float64)

    Ns = Ns.to(torch.float64).clamp(min=1.0)
    sum_phi = sum_phi.to(torch.float64)
    sum_phi_sq = sum_phi_sq.to(torch.float64)
    sum_target = sum_target.to(torch.float64)
    sum_target_sq = sum_target_sq.to(torch.float64)
    cross_sum = cross_sum.to(torch.float64)

    mean_phi = sum_phi / Ns
    mean_target = sum_target / Ns

    var_phi = (sum_phi_sq / Ns) - mean_phi.pow(2)
    var_target = (sum_target_sq / Ns) - mean_target.pow(2)
    var_phi = var_phi.clamp(min=EPS)
    var_target = var_target.clamp(min=EPS)

    cov = (cross_sum / Ns) - mean_phi * mean_target
    denom = torch.sqrt(var_phi * var_target) + EPS
    rho = (cov / denom).clamp(-0.9999, 0.9999)
    mi = -0.5 * torch.log1p(-rho.pow(2))
    mi = mi.clamp(min=0.0)
    return mi


def _information_bottleneck_clusters(
    information: torch.Tensor,
    redundancy: torch.Tensor,
    commensurability: torch.Tensor,
) -> List[List[int]]:
    """
    Greedy submodular selection that builds clusters by maximising a mutual
    information objective with a redundancy penalty.

    Each step picks the unassigned unit with the highest residual gain
    ``g(i) = I_i + gamma * mean(commensurability) - lambda * mean(redundancy)``.
    The penalty discourages selecting highly redundant units, while the
    commensurability bonus encourages compatibility within a cluster.  The
    procedure is a standard greedy maximiser for the quadratic information
    bottleneck surrogate ``I(S) - lambda * R(S)`` which enjoys (1-1/e)
    guarantees for submodular objectives.
    """

    rows = information.numel()
    if rows == 0:
        return []

    info = information.to(torch.float64)
    redundancy = redundancy.to(torch.float64)
    comm = commensurability.to(torch.float64)

    info = torch.relu(info)
    info_max = info.max().item()
    if info_max > 0:
        info = info / info_max

    redundancy = redundancy.abs().clamp(0.0, 1.0)
    comm = comm.clamp(min=0.0, max=1.0)

    lambda_reg = float(max(info.mean().item(), EPS))
    gamma = float(comm.mean().item())

    remaining = set(range(rows))
    clusters: List[List[int]] = []

    while remaining:
        root = max(remaining, key=lambda idx: float(info[idx]))
        remaining.remove(root)
        cluster = [root]
        cluster_indices = torch.tensor(cluster, dtype=torch.long, device=redundancy.device)

        while remaining:
            best_gain = 0.0
            best_idx = None
            base_penalty = float(info[cluster_indices].mean().item())

            for candidate in remaining:
                redundancy_penalty = float(redundancy[candidate, cluster_indices].mean().item())
                comm_bonus = float(comm[candidate, cluster_indices].mean().item())
                gain = float(info[candidate].item()) + gamma * comm_bonus - lambda_reg * redundancy_penalty
                gain -= 0.1 * base_penalty * redundancy_penalty
                if gain > best_gain + 1e-6:
                    best_gain = gain
                    best_idx = candidate

            if best_idx is None or best_gain <= 0.0:
                break
            cluster.append(best_idx)
            remaining.remove(best_idx)
            cluster_indices = torch.tensor(cluster, dtype=torch.long, device=redundancy.device)

        clusters.append(sorted(cluster))

    return clusters


def _build_hvp_diag_features(hvp_data: Dict[str, object]) -> Optional[torch.Tensor]:
    vectors: Sequence[torch.Tensor] = hvp_data.get("weight_vectors") or hvp_data.get("vectors") or []
    responses: Sequence[torch.Tensor] = hvp_data.get("weight_responses") or hvp_data.get("responses") or []
    if not vectors or not responses:
        return None
    count = min(len(vectors), len(responses))
    if count == 0:
        return None
    rows = int(hvp_data.get("rows") or 0)
    if rows <= 0:
        rows = vectors[0].numel()
    device = vectors[0].device
    matrices: List[torch.Tensor] = []
    for vec, resp in zip(vectors[:count], responses[:count]):
        v = vec.to(device=device, dtype=torch.float32).reshape(rows, -1)
        r = resp.to(device=device, dtype=torch.float32).reshape(rows, -1)
        diag = (v * r).sum(dim=1, keepdim=True)
        matrices.append(diag)
    if not matrices:
        return None
    feature_matrix = torch.cat(matrices, dim=1)
    mask = torch.isnan(feature_matrix) | torch.isinf(feature_matrix)
    if mask.any():
        feature_matrix = torch.where(mask, torch.zeros_like(feature_matrix), feature_matrix)
    mean = feature_matrix.mean(dim=0, keepdim=True)
    std = feature_matrix.std(dim=0, keepdim=True).clamp_min(1e-6)
    feature_matrix = (feature_matrix - mean) / std
    return feature_matrix


def _standardize_feature_matrix(features: List[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(features, dim=1).to(torch.float32)
    mean = stacked.mean(dim=0, keepdim=True)
    std = stacked.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (stacked - mean) / std


def _info_submodular_clusters(
    summary: Dict[str, torch.Tensor],
    *,
    max_members: int,
    max_clusters: int,
    redundancy_penalty: float,
    min_gain: float,
    projection_rank: Optional[int],
    seed: int,
    hvp_data: Optional[Dict[str, object]] = None,
) -> Tuple[List[List[int]], List[List[float]], List[List[int]]]:
    features = _gather_low_rank_features(summary)
    if not features:
        return [], [], []

    feature_matrix = _standardize_feature_matrix(features)
    rows, dims = feature_matrix.shape
    if rows == 0:
        return [], [], []

    if projection_rank is not None and projection_rank > 0 and projection_rank < dims:
        generator = torch.Generator(device=feature_matrix.device)
        generator.manual_seed(seed)
        projection = torch.randn((dims, projection_rank), generator=generator, device=feature_matrix.device)
        feature_matrix = feature_matrix.matmul(projection)

    embedding = F.normalize(feature_matrix, p=2, dim=1)

    info = _gaussian_mutual_information(summary)
    if info.numel() == 0:
        info = torch.ones(rows, dtype=torch.float64)
    elif info.numel() != rows:
        info = F.pad(info, (0, max(0, rows - info.numel())))[:rows]

    info = info.to(torch.float32)
    info_weight = float(os.environ.get("SCORING_CLUSTER_INFO_WEIGHT", "1.0"))

    if hvp_data:
        diag_features = _build_hvp_diag_features(hvp_data)
        if diag_features is not None and diag_features.size(0) == rows:
            curvature_penalty = float(os.environ.get("SCORING_CLUSTER_INFO_CURV_PENALTY", "0.0"))
            if curvature_penalty > 0.0:
                curv = diag_features.abs().mean(dim=1)
                info = info / (1.0 + curvature_penalty * curv.to(info.device).to(info.dtype))
        else:
            diag_features = None
    else:
        diag_features = None

    curv_weight = float(os.environ.get("SCORING_CLUSTER_CURV_WEIGHT", "0.0"))
    curv_redundancy_penalty = float(os.environ.get("SCORING_CLUSTER_CURV_REDUNDANCY", "0.0"))
    curv_rank_env = os.environ.get("SCORING_CLUSTER_CURV_RANK")
    curv_rank = int(curv_rank_env) if curv_rank_env is not None else None
    curv_ridge = float(os.environ.get("SCORING_CLUSTER_CURV_RIDGE", "1e-6"))

    use_curvature = curv_weight > 0.0 and diag_features is not None
    if use_curvature:
        curvature_matrix = diag_features.abs().to(dtype=torch.float32, device=embedding.device)
        if curvature_matrix.ndim == 1:
            curvature_matrix = curvature_matrix.unsqueeze(1)
        if curvature_matrix.size(1) == 0:
            use_curvature = False
        else:
            if curv_rank is not None and 0 < curv_rank < curvature_matrix.size(1):
                gram = curvature_matrix.t().matmul(curvature_matrix)
                eigvals, eigvecs = torch.linalg.eigh(gram)
                order = eigvals.argsort(descending=True)[:curv_rank]
                eigvals = eigvals[order].clamp_min(1e-8)
                eigvecs = eigvecs[:, order]
                curvature_matrix = curvature_matrix.matmul(eigvecs)
            curv_lambda = max(curv_ridge, 1e-6)
            curvature_matrix = curvature_matrix.contiguous()
            U = curvature_matrix / math.sqrt(curv_lambda)
            curv_rank = U.size(1)
            curv_identity = torch.eye(curv_rank, dtype=torch.float32, device=embedding.device)
            curv_A_inv = curv_identity.clone()
            curv_cluster_vec = torch.zeros(curv_rank, dtype=torch.float32, device=embedding.device)
            curv_ready = True
            if curv_redundancy_penalty > 0.0:
                curv_embedding = F.normalize(curvature_matrix, p=2, dim=1)
            else:
                curv_embedding = None
    else:
        curv_ready = False
        U = None
        curv_A_inv = None
        curv_cluster_vec = None
        curv_embedding = None
        curv_identity = None

    mask = torch.zeros(rows, dtype=torch.bool, device=embedding.device)
    clusters: List[List[int]] = []
    alphas: List[List[float]] = []
    prune_order: List[List[int]] = []

    max_members = max(1, min(max_members, rows))
    max_clusters = max(1, max_clusters)
    redundancy_penalty = max(0.0, redundancy_penalty)
    min_gain = max(0.0, min_gain)

    for _ in range(max_clusters):
        cluster_indices: List[int] = []
        cluster_vector = torch.zeros(embedding.size(1), dtype=torch.float32, device=embedding.device)
        if curv_ready:
            curv_cluster_vec.zero_()
            curv_A_inv.copy_(curv_identity)
        selections = 0
        while selections < max_members:
            gains = info_weight * info
            if selections > 0:
                redundancy = embedding.matmul(cluster_vector)
                gains = gains - redundancy_penalty * redundancy
                if curv_ready and curv_redundancy_penalty > 0.0 and curv_embedding is not None:
                    curv_redund = curv_embedding.matmul(curv_cluster_vec)
                    gains = gains - curv_redundancy_penalty * curv_redund

            if curv_ready:
                quad = torch.einsum("ij,jk,ik->i", U, curv_A_inv, U).clamp_min(0.0)
                gains = gains + curv_weight * torch.log1p(quad)

            gains = gains.masked_fill(mask, float("-inf"))
            best_gain, best_idx = gains.max(dim=0)
            if not torch.isfinite(best_gain) or best_gain.item() <= min_gain:
                break
            idx = int(best_idx.item())
            cluster_indices.append(idx)
            mask[idx] = True
            cluster_vector.add_(embedding[idx])
            if curv_ready:
                u = U[idx]
                tmp_vec = curv_A_inv.matmul(u)
                denom = 1.0 + float(torch.dot(u, tmp_vec))
                if denom <= 1e-8:
                    # skip curvature update to avoid instability
                    pass
                else:
                    update = torch.outer(tmp_vec, tmp_vec) / denom
                    curv_A_inv.sub_(update)
                if curv_redundancy_penalty > 0.0 and curv_embedding is not None:
                    curv_cluster_vec.add_(curv_embedding[idx])
            selections += 1

        if not cluster_indices:
            break

        cluster_indices_sorted = sorted(cluster_indices)
        weights = info[cluster_indices_sorted].clamp(min=EPS)
        weight_sum = float(weights.sum().item()) + EPS
        alphas.append([float(w.item() / weight_sum) for w in weights])
        clusters.append(cluster_indices_sorted)
        prune_order.append(cluster_indices_sorted[:])

        if mask.all():
            break

    return clusters, alphas, prune_order


def cluster_frontier(
    summary: Dict[str, torch.Tensor],
    alignment: torch.Tensor,
    synergy: torch.Tensor,
    commensurability: torch.Tensor,
    hvp_data: Optional[Dict[str, object]] = None,
) -> Tuple[List[List[int]], List[List[float]], List[List[int]]]:
    mode = os.environ.get("SCORING_CLUSTER_MODE", "").lower().strip()
    if mode == "info_submodular":
        projection_rank_env = os.environ.get("SCORING_CLUSTER_INFO_RANK")
        projection_rank = int(projection_rank_env) if projection_rank_env is not None else None
        max_members = int(os.environ.get("SCORING_CLUSTER_MAX_MEMBERS", "2048"))
        max_clusters = int(os.environ.get("SCORING_CLUSTER_MAX_CLUSTERS", "1"))
        redundancy_penalty = float(os.environ.get("SCORING_CLUSTER_INFO_REDUNDANCY", "0.1"))
        min_gain = float(os.environ.get("SCORING_CLUSTER_INFO_MIN_GAIN", "1e-4"))
        seed = int(os.environ.get("SCORING_CLUSTER_LOW_RANK_SEED", "0"))
        clusters, alphas, prune = _info_submodular_clusters(
            summary,
            max_members=max_members,
            max_clusters=max_clusters,
            redundancy_penalty=redundancy_penalty,
            min_gain=min_gain,
            projection_rank=projection_rank,
            seed=seed,
            hvp_data=hvp_data,
        )
        if clusters:
            return clusters, alphas, prune

    low_rank_requested = int(os.environ.get("SCORING_CLUSTER_LOW_RANK_K", "0"))
    if low_rank_requested > 0:
        return _cluster_low_rank(summary, low_rank_requested)

    rows = alignment.size(0)
    if rows == 0:
        return [], [], []

    information = _gaussian_mutual_information(summary)
    if information.numel() == 0:
        information = torch.ones(rows, dtype=torch.float64)
    elif information.numel() != rows:
        information = F.pad(information, (0, max(0, rows - information.numel())))
        information = information[:rows]

    redundancy = alignment.abs() * torch.exp(synergy.abs())
    clusters = _information_bottleneck_clusters(information, redundancy, commensurability)

    information = information.to(torch.float64)
    alphas: List[List[float]] = []
    for cluster in clusters:
        values = information[cluster]
        total = float(values.sum().item()) + EPS
        alphas.append([float(v.item() / total) for v in values])

    cluster_scores = [float(information[cluster].sum().item()) for cluster in clusters]
    prune_order = [cluster for _, cluster in sorted(zip(cluster_scores, clusters), reverse=True)]

    return clusters, alphas, prune_order
