"""Camera projection helpers used by colouring and visualization."""
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

SH_C0 = 0.28209479177387814


def gaussian_rgb(encoder: Any) -> Tensor:
    harmonics = encoder.gaussians.harmonics
    if harmonics.ndim != 4 or harmonics.shape[2] != 3:
        raise ValueError(f"Expected harmonics [B,N,3,D], got {harmonics.shape}")
    return (0.5 + SH_C0 * harmonics[..., 0]).clamp(0.0, 1.0)


@torch.no_grad()
def colors_from_input_images(images: Tensor, encoder: Any) -> Tuple[Tensor, Tensor]:
    """Project VGGT points into predicted cameras and average observed RGB."""
    points = encoder.gaussians.means.float()
    c2w = encoder.pred_context_pose["extrinsic"].float()
    intrinsics = encoder.pred_context_pose["intrinsic"].float()
    world_to_camera = torch.linalg.inv(c2w)
    rotation = world_to_camera[..., :3, :3]
    translation = world_to_camera[..., :3, 3]
    camera_points = torch.einsum("bvij,bnj->bvni", rotation, points) + translation[:, :, None]
    depth = camera_points[..., 2]
    x = camera_points[..., 0] / depth.clamp_min(1e-6)
    y = camera_points[..., 1] / depth.clamp_min(1e-6)
    u = intrinsics[..., 0, 0, None] * x + intrinsics[..., 0, 2, None]
    v = intrinsics[..., 1, 1, None] * y + intrinsics[..., 1, 2, None]
    visible = (depth > 1e-6) & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    grid = torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1).unsqueeze(-2)
    batch, views, points_count = visible.shape
    sampled = F.grid_sample(
        images.reshape(batch * views, 3, *images.shape[-2:]),
        grid.reshape(batch * views, points_count, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch, views, 3, points_count).permute(0, 1, 3, 2)
    weights = visible.float().unsqueeze(-1)
    color_sum = (sampled * weights).sum(1)
    observations = weights.sum(1)
    observed_rgb = color_sum / observations.clamp_min(1.0)
    fallback_rgb = gaussian_rgb(encoder)
    rgb = torch.where(observations > 0, observed_rgb, fallback_rgb)
    return rgb.clamp(0.0, 1.0), observations.squeeze(-1).long()


def project_world_points(
    points_world: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    homogeneous = np.concatenate([
        points_world.astype(np.float64),
        np.ones((points_world.shape[0], 1), dtype=np.float64),
    ], axis=-1)
    camera = homogeneous @ np.linalg.inv(c2w).T
    depth = camera[:, 2]
    uvw = camera[:, :3] @ intrinsic.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)
    return uv, depth
