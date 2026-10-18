"""XYZ-first semantic instance clustering for multi-person scenes."""
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from torch import Tensor

INSTANCE_COLORS = np.asarray(
    [
        [230, 25, 75],
        [60, 180, 75],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 212],
        [0, 128, 128],
        [220, 190, 255],
        [170, 110, 40],
    ],
    dtype=np.uint8,
)


def _dbscan(values: Tensor, eps: float, min_samples: int) -> Tensor:
    labels = DBSCAN(
        eps=eps, min_samples=min_samples, n_jobs=-1
    ).fit_predict(values.detach().float().cpu().numpy())
    return torch.from_numpy(labels).to(values.device, dtype=torch.long)


@torch.no_grad()
def infer_semantic_instances(
    encoder: Any, embedding: Tensor, semantic_visible: Tensor,
    args: Dict[str, Any],
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    points = encoder.gaussians.means[0].float()
    confidence = encoder.human_masks[0, :, 0].float()
    opacity = encoder.gaussians.opacities[0].float().reshape(-1)
    valid = encoder.infos["valid_point_mask"][0].bool()
    candidate = (
        valid
        & semantic_visible[0]
        & (confidence >= args["human_threshold"])
        & (opacity >= args["human_opacity_threshold"])
        & (encoder.infos["view_support"][0] >= args["min_view_support"])
        & (encoder.infos["pixel_support"][0] >= args["min_pixel_support"])
    )
    candidate_indices = torch.where(candidate)[0]
    raw_labels = torch.zeros(points.shape[0], device=points.device, dtype=torch.long)
    decoder_labels = torch.zeros_like(raw_labels)
    if candidate_indices.numel() < args["semantic_min_samples"]:
        return decoder_labels[None], raw_labels[None], {
            "candidate_points": int(candidate_indices.numel()),
            "xyz_cluster_count": 0,
            "semantic_cluster_count": 0,
            "decoded_cluster_count": 0,
            "raw_cluster_counts": [],
            "decoder_cluster_counts": [],
        }

    candidate_points = points[candidate_indices]
    candidate_embedding = F.normalize(embedding[0, candidate_indices].float(), dim=-1)
    xyz_labels = _dbscan(
        candidate_points / args["xyz_cleanup_eps"],
        1.0,
        args["xyz_cleanup_min_samples"],
    )
    xyz_ids, xyz_counts = torch.unique(xyz_labels[xyz_labels >= 0], return_counts=True)
    xyz_keep = xyz_counts >= args["min_person_points"]
    xyz_ids, xyz_counts = xyz_ids[xyz_keep], xyz_counts[xyz_keep]

    person_candidates = []
    split_scores: Dict[str, float] = {}
    semantic_subclusters: Dict[str, int] = {}
    for xyz_id, xyz_count in zip(xyz_ids.tolist(), xyz_counts.tolist()):
        local = torch.where(xyz_labels == xyz_id)[0]
        local_semantic = _dbscan(
            candidate_embedding[local],
            args["semantic_split_eps"],
            args["semantic_min_samples"],
        )
        semantic_ids, semantic_counts = torch.unique(
            local_semantic[local_semantic >= 0], return_counts=True
        )
        semantic_keep = semantic_counts >= args["min_person_points"]
        semantic_ids = semantic_ids[semantic_keep]
        semantic_counts = semantic_counts[semantic_keep]
        use_split = False
        score = float("nan")
        if semantic_ids.numel() >= 2:
            labelled = torch.zeros_like(local_semantic, dtype=torch.bool)
            for semantic_id in semantic_ids.tolist():
                labelled |= local_semantic == semantic_id
            metric_index = torch.where(labelled)[0]
            if metric_index.numel() > args["semantic_score_points"]:
                take = torch.linspace(
                    0, metric_index.numel() - 1,
                    args["semantic_score_points"], device=metric_index.device,
                ).long()
                metric_index = metric_index[take]
            score = float(silhouette_score(
                candidate_embedding[local][metric_index].cpu().numpy(),
                local_semantic[metric_index].cpu().numpy(),
                metric="cosine",
            ))
            use_split = score >= args["semantic_split_min_silhouette"]
        split_scores[str(xyz_id)] = score
        semantic_subclusters[str(xyz_id)] = int(semantic_ids.numel())
        if use_split:
            for semantic_id in semantic_ids.tolist():
                sub_local = local[local_semantic == semantic_id]
                person_candidates.append(candidate_indices[sub_local])
        else:
            person_candidates.append(candidate_indices[local])

    person_candidates.sort(key=lambda indices: int(indices.numel()), reverse=True)
    person_candidates = person_candidates[:args["max_persons"]]
    raw_counts = [int(indices.numel()) for indices in person_candidates]
    decoder_counts = []
    generator = torch.Generator(device=points.device)
    generator.manual_seed(args["seed"])
    for person_id, identity_indices in enumerate(person_candidates, start=1):
        raw_labels[identity_indices] = person_id
        selected = identity_indices
        if selected.numel() > args["person_point_count"]:
            permutation = torch.randperm(
                selected.numel(), generator=generator, device=selected.device
            )
            selected = selected[permutation[:args["person_point_count"]]]
        decoder_labels[selected] = person_id
        decoder_counts.append(int(selected.numel()))

    return decoder_labels[None], raw_labels[None], {
        "candidate_points": int(candidate_indices.numel()),
        "xyz_cluster_count": int(xyz_ids.numel()),
        "semantic_cluster_count": len(person_candidates),
        "decoded_cluster_count": len(person_candidates),
        "raw_cluster_counts": raw_counts,
        "decoder_cluster_counts": decoder_counts,
        "xyz_noise_ratio": float((xyz_labels < 0).float().mean()),
        "semantic_split_scores": split_scores,
        "semantic_subclusters_per_xyz": semantic_subclusters,
    }


def colour_instances(base_rgb: Tensor, labels: Tensor) -> np.ndarray:
    colours = (base_rgb.clamp(0, 1) * 255.0).byte().cpu().numpy()
    labels_numpy = labels.cpu().numpy()
    for person_id in torch.unique(labels[labels > 0]).tolist():
        colours[labels_numpy == person_id] = INSTANCE_COLORS[
            (person_id - 1) % len(INSTANCE_COLORS)
        ]
    return colours


@torch.no_grad()
def expand_scene_instances(
    encoder: Any, core_labels: Tensor, semantic_visible: Tensor,
    args: Dict[str, Any],
) -> Tuple[List[Dict[str, Tensor]], List[int], Dict[str, Any]]:
    points = encoder.gaussians.means[0].float()
    confidence = encoder.human_masks[0, :, 0].float()
    opacity = encoder.gaussians.opacities[0].reshape(-1).float()
    valid = encoder.infos["valid_point_mask"][0].bool()
    labels = core_labels[0].long()
    quality = (
        valid & semantic_visible[0]
        & (confidence >= args["expanded_human_conf_threshold"])
        & (opacity >= args["expanded_opacity_threshold"])
        & (encoder.infos["view_support"][0] >= args["expanded_min_view_support"])
        & (encoder.infos["pixel_support"][0] >= args["expanded_min_pixel_support"])
    )
    core = labels > 0
    query = torch.where(quality & ~core)[0]
    reference_parts: List[Tensor] = []
    reference_label_parts: List[Tensor] = []
    person_ids = [int(value) for value in torch.unique(labels[core]).tolist()]
    for person_id in person_ids:
        indices = torch.where(labels == person_id)[0]
        if indices.numel() > args["expansion_reference_points"]:
            take = torch.linspace(
                0, indices.numel() - 1, args["expansion_reference_points"],
                device=indices.device,
            ).long()
            indices = indices[take]
        reference_parts.append(indices)
        reference_label_parts.append(torch.full_like(indices, person_id))

    expanded_labels = labels.clone()
    if reference_parts and query.numel() > 0:
        reference = torch.cat(reference_parts)
        reference_labels = torch.cat(reference_label_parts)
        for start in range(0, query.numel(), args["expansion_query_chunk"]):
            part = query[start:start + args["expansion_query_chunk"]]
            distance = torch.cdist(points[part], points[reference])
            nearest_distance, nearest = distance.min(1)
            accepted = nearest_distance <= args["expansion_max_distance"]
            expanded_labels[part[accepted]] = reference_labels[nearest[accepted]]

    selections: List[Dict[str, Tensor]] = []
    retained_labels: List[int] = []
    counts: Dict[str, Any] = {}
    for person_id in person_ids:
        core_indices = torch.where(labels == person_id)[0]
        recovered = torch.where(
            (expanded_labels == person_id) & (labels != person_id)
        )[0]
        indices = torch.cat([core_indices, recovered])
        if indices.numel() < args["min_person_points"]:
            continue
        selections.append({
            "indices": indices,
            "core_mask": torch.cat([
                torch.ones_like(core_indices, dtype=torch.bool),
                torch.zeros_like(recovered, dtype=torch.bool),
            ]),
        })
        retained_labels.append(person_id)
        counts[str(person_id)] = {
            "core": int(core_indices.numel()),
            "recovered": int(recovered.numel()),
            "expanded": int(indices.numel()),
        }
    return selections, retained_labels, {
        "expanded_instance_counts": counts,
        "expanded_total_points": int((expanded_labels > 0).sum()),
    }
