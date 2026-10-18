"""Build per-person decoder tokens from expanded instance points."""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from torch import Tensor

from lib.utils.features import pixel_to_feature_grid, project_visible_points

def sample_per_view(
    feature_bv: Tensor, u: Tensor, v: Tensor,
    original_hw: Tuple[int, int], input_size: int,
) -> Tensor:
    """Return sampled features as [B,N,V,C] without averaging viewpoints."""
    batch, views, count = u.shape
    grid = pixel_to_feature_grid(
        u, v, original_hw, input_size, feature_bv.shape[-2:]
    ).reshape(batch * views, count, 1, 2)
    sampled = F.grid_sample(
        feature_bv.float(), grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).squeeze(-1)
    return sampled.reshape(
        batch, views, feature_bv.shape[1], count
    ).permute(0, 3, 1, 2)


@torch.no_grad()
def select_variable_person_points(
    encoder: Any, args: argparse.Namespace,
) -> Tuple[List[Dict[str, Tensor]], Dict[str, float]]:
    """Find a precise core, then recover a spatially adjacent expanded body."""
    selected_batches: List[Dict[str, Tensor]] = []
    candidates, components, core_counts, expanded_counts = [], [], [], []
    points = encoder.gaussians.means.float()
    confidence = encoder.human_masks[..., 0].float()
    opacity = encoder.gaussians.opacities.reshape(points.shape[:2]).float()
    valid = encoder.infos["valid_point_mask"].bool()
    for batch_index in range(points.shape[0]):
        keep = (
            valid[batch_index]
            & (confidence[batch_index] >= args.human_conf_threshold)
            & (opacity[batch_index] >= args.human_opacity_threshold)
            & (encoder.infos["view_support"][batch_index] >= args.min_view_support)
            & (encoder.infos["pixel_support"][batch_index] >= args.min_pixel_support)
        )
        indices = torch.where(keep)[0]
        candidates.append(int(indices.numel()))
        if indices.numel() < args.cluster_min_samples:
            selected_batches.append({"indices": indices[:0], "core_mask": indices[:0].bool()})
            components.append(0)
            core_counts.append(0)
            expanded_counts.append(0)
            continue
        cluster_indices = indices
        if cluster_indices.numel() > args.cluster_max_points:
            cluster_indices = cluster_indices[torch.randperm(
                cluster_indices.numel(), device=points.device
            )[:args.cluster_max_points]]
        labels = DBSCAN(
            eps=1.0, min_samples=args.cluster_min_samples, n_jobs=-1
        ).fit_predict(
            (points[batch_index, cluster_indices] / args.cluster_eps_xyz)
            .cpu().numpy()
        )
        labels = torch.from_numpy(labels).to(points.device)
        ids, counts = torch.unique(labels[labels >= 0], return_counts=True)
        accepted = counts >= args.cluster_min_points
        ids, counts = ids[accepted], counts[accepted]
        components.append(int(ids.numel()))
        if ids.numel() == 0:
            selected_batches.append({"indices": indices[:0], "core_mask": indices[:0].bool()})
            core_counts.append(0)
            expanded_counts.append(0)
            continue
        core = cluster_indices[labels == ids[counts.argmax()]]
        loose = (
            valid[batch_index]
            & (confidence[batch_index] >= args.expanded_human_conf_threshold)
            & (opacity[batch_index] >= args.expanded_opacity_threshold)
            & (encoder.infos["view_support"][batch_index]
               >= args.expanded_min_view_support)
            & (encoder.infos["pixel_support"][batch_index]
               >= args.expanded_min_pixel_support)
        )
        loose_indices = torch.where(loose)[0]
        is_core = torch.zeros(points.shape[1], device=points.device, dtype=torch.bool)
        is_core[core] = True
        query = loose_indices[~is_core[loose_indices]]
        reference = core
        if reference.numel() > args.expansion_reference_points:
            take = torch.linspace(
                0, reference.numel() - 1, args.expansion_reference_points,
                device=reference.device,
            ).long()
            reference = reference[take]
        recovered_parts: List[Tensor] = []
        for start in range(0, query.numel(), args.expansion_query_chunk):
            part = query[start:start + args.expansion_query_chunk]
            distance = torch.cdist(
                points[batch_index, part], points[batch_index, reference]
            ).amin(dim=1)
            recovered_parts.append(part[distance <= args.expansion_max_distance])
        recovered = (
            torch.cat(recovered_parts) if recovered_parts else query[:0]
        )
        expanded = torch.cat([core, recovered])
        selected_batches.append({
            "indices": expanded,
            "core_mask": torch.cat([
                torch.ones_like(core, dtype=torch.bool),
                torch.zeros_like(recovered, dtype=torch.bool),
            ]),
        })
        core_counts.append(int(core.numel()))
        expanded_counts.append(int(expanded.numel()))
    return selected_batches, {
        "cluster_candidates": float(np.mean(candidates)),
        "cluster_count": float(np.mean(components)),
        "core_points": float(np.mean(core_counts)),
        "expanded_points": float(np.mean(expanded_counts)),
    }


@torch.no_grad()
def robust_expanded_frame(
    points: Tensor, reliability: Tensor, trim_quantile: float,
) -> Tuple[Tensor, Tensor]:
    weights = reliability.float().clamp_min(1e-6)
    weights = weights / weights.sum()
    initial = (points * weights[:, None]).sum(0)
    radius = torch.linalg.vector_norm(points - initial, dim=-1)
    threshold = torch.quantile(radius, trim_quantile)
    keep = radius <= threshold
    weights = reliability[keep].float().clamp_min(1e-6)
    weights = weights / weights.sum()
    centre = (points[keep] * weights[:, None]).sum(0)
    radius = torch.linalg.vector_norm(points[keep] - centre, dim=-1)
    scale = torch.sqrt((radius.square() * weights).sum()).clamp_min(1e-3)
    return centre, scale


@torch.no_grad()
def adaptive_voxel_representatives(
    xyz_local: Tensor, core_mask: Tensor, reliability: Tensor,
    voxel_size: float, maximum_tokens: int,
) -> Tuple[Tensor, Tensor, float]:
    """Choose one reliable representative per normalized spatial voxel."""
    size = float(voxel_size)
    device = xyz_local.device
    while True:
        coordinate = torch.floor(xyz_local / size).to(torch.int64).cpu().numpy()
        # Core points are ordered first and reliability breaks ties within each
        # group.  np.unique(return_index) then favours a core representative.
        order = torch.argsort(
            core_mask.float() * 2.0 + reliability.clamp(0, 1), descending=True
        )
        ordered_coordinate = coordinate[order.cpu().numpy()]
        _, first, counts = np.unique(
            ordered_coordinate, axis=0, return_index=True, return_counts=True
        )
        chosen = order[torch.from_numpy(first).to(device=device, dtype=torch.long)]
        population = torch.from_numpy(counts).to(
            device=device, dtype=torch.float32
        )
        if chosen.numel() <= maximum_tokens:
            return chosen, population, size
        size *= 1.20


@torch.no_grad()
def build_person_inputs(
    encoder: Any, images: Tensor, selected: List[Dict[str, Tensor]],
    romp_feature: Tensor, dino_feature: Tensor,
    semantic_args: argparse.Namespace, args: argparse.Namespace,
) -> Tuple[List[Dict[str, Tensor]], List[int]]:
    batch, views = images.shape[:2]
    height, width = images.shape[-2:]
    romp_feature = romp_feature.reshape(
        batch, views, *romp_feature.shape[1:]
    )
    dino_feature = dino_feature.reshape(
        batch, views, *dino_feature.shape[1:]
    )
    persons, kept_batches = [], []
    for b, selection in enumerate(selected):
        indices = selection["indices"]
        core_mask = selection["core_mask"]
        if indices.numel() < args.min_person_points:
            continue
        confidence_all = encoder.human_masks[b, indices, 0].float()
        opacity_all = encoder.gaussians.opacities[b, indices].reshape(-1).float()
        support_all = encoder.infos["view_support"][b, indices].float()
        reliability_all = (
            confidence_all.pow(2.0) * opacity_all
            * (0.5 + 0.5 * (support_all / max(views, 1)).clamp(0, 1))
        )
        reliability_all = torch.where(
            core_mask, reliability_all.clamp_min(0.75), reliability_all
        )
        all_points = encoder.gaussians.means[b, indices].float()
        center, observed_scale = robust_expanded_frame(
            all_points, reliability_all, args.frame_trim_quantile
        )
        core_center, core_scale = robust_expanded_frame(
            all_points[core_mask], reliability_all[core_mask],
            args.frame_trim_quantile,
        )
        xyz_local_all = (all_points - center) / observed_scale
        representative, voxel_population, actual_voxel_size = (
            adaptive_voxel_representatives(
                xyz_local_all, core_mask, reliability_all,
                args.normalized_voxel_size, args.maximum_voxel_tokens,
            )
        )
        expanded_count = torch.tensor(
            indices.numel(), device=indices.device, dtype=torch.float32
        )
        core_fraction = core_mask.float().mean()
        indices = indices[representative]
        core_mask = core_mask[representative]
        reliability = reliability_all[representative]
        person_points = encoder.gaussians.means[b:b + 1, indices].float()
        u, v, inside, visible, weights, depth_error = project_visible_points(
            person_points,
            encoder.pred_context_pose["extrinsic"][b:b + 1],
            encoder.pred_context_pose["intrinsic"][b:b + 1],
            encoder.depth_dict["depth"][b:b + 1],
            encoder.depth_dict["depth_conf"][b:b + 1],
            (height, width), semantic_args.visibility_abs,
            semantic_args.visibility_rel,
        )
        visible_points = visible.any(dim=1)[0]
        if int(visible_points.sum()) < args.min_person_points:
            continue
        indices = indices[visible_points]
        core_mask = core_mask[visible_points]
        reliability = reliability[visible_points]
        voxel_population = voxel_population[visible_points]
        person_points = person_points[:, visible_points]
        u, v = u[:, :, visible_points], v[:, :, visible_points]
        visible, weights = visible[:, :, visible_points], weights[:, :, visible_points]
        depth_error = depth_error[:, :, visible_points]
        weights *= visible.float()
        weights /= weights.sum(1, keepdim=True).clamp_min(1e-6)

        romp_observation = sample_per_view(
            romp_feature[b].reshape(views, *romp_feature.shape[2:]),
            u, v, (height, width), semantic_args.romp_input_size,
        )[0]
        dino_observation = sample_per_view(
            dino_feature[b].reshape(views, *dino_feature.shape[2:]),
            u, v, (height, width), semantic_args.dino_input_size,
        )[0]
        c2w = encoder.pred_context_pose["extrinsic"][b:b + 1].float()
        camera_center = c2w[:, :, :3, 3]
        ray = F.normalize(
            person_points[:, None] - camera_center[:, :, None], dim=-1
        )[0].permute(1, 0, 2)
        uv = torch.stack([
            u[0] / max(width - 1, 1), v[0] / max(height - 1, 1)
        ], dim=-1).permute(1, 0, 2)
        relative_depth = depth_error[0].permute(1, 0).unsqueeze(-1).clamp_max(1.0)
        confidence_view = weights[0].permute(1, 0).unsqueeze(-1)
        view_geometry = torch.cat([ray, uv, relative_depth, confidence_view], dim=-1)

        conf = encoder.human_masks[b, indices, 0].float()
        opacity = encoder.gaussians.opacities[b, indices].reshape(-1).float()
        persons.append({
            "vggt": encoder.infos["feature_3d"][b, indices].float(),
            "romp": romp_observation,
            "dino": dino_observation,
            "view_geometry": view_geometry,
            "visible": visible[0].permute(1, 0),
            "view_weights": weights[0].permute(1, 0),
            "xyz": person_points[0],
            "xyz_local": (person_points[0] - center)
                / observed_scale.clamp_min(1e-6),
            "confidence": conf,
            "opacity": opacity,
            "reliability": reliability,
            "core_mask": core_mask,
            "voxel_population": voxel_population,
            "center": center,
            "observed_scale": observed_scale,
            "core_scale": core_scale,
            "core_fraction": core_fraction,
            "expanded_count": expanded_count,
            "mean_view_coverage": visible[0].float().sum(0).mean() / views,
            "actual_voxel_size": torch.tensor(
                actual_voxel_size, device=indices.device
            ),
        })
        kept_batches.append(b)
    return persons, kept_batches
