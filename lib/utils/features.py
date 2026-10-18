"""ROMP / DINOv3 sampling and multi-view feature fusion."""
import torch
import torch.nn.functional as F

@torch.no_grad()
def extract_romp_features(romp, images, input_size, expected_dim):
    # Direct resize preserves a one-to-one normalized coordinate mapping.  ROMP
    # receives the same BHWC [0,255] convention used in the earlier experiments.
    batch, views, channels, _, _ = images.shape
    resized = F.interpolate(
        images.reshape(batch * views, channels, *images.shape[-2:]).float(),
        size=(input_size, input_size), mode="bilinear", align_corners=False,
    ).permute(0, 2, 3, 1) * 255.0
    feature = romp.model.module.backbone(resized)
    if feature.ndim != 4 or feature.shape[1] != expected_dim:
        raise RuntimeError(f"Unexpected ROMP feature shape {tuple(feature.shape)}")
    return feature.float()


@torch.no_grad()
def extract_dino_features(model, images, input_size, expected_dim):
    batch, views, channels, _, _ = images.shape
    resized = F.interpolate(
        images.reshape(batch * views, channels, *images.shape[-2:]).float(),
        size=(input_size, input_size), mode="bicubic", align_corners=False,
    )
    mean = resized.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = resized.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = model.forward_features((resized - mean) / std)
    tokens = output["x_norm_patchtokens"]
    if tokens.shape[-1] != expected_dim:
        raise RuntimeError(f"Unexpected DINO feature shape {tuple(tokens.shape)}")
    patch = int(getattr(model, "patch_size", 16))
    height = input_size // patch
    width = input_size // patch
    if tokens.shape[1] != height * width:
        raise RuntimeError(f"DINO token count {tokens.shape[1]} != {height}x{width}")
    return tokens.transpose(1, 2).reshape(batch * views, expected_dim, height, width).float()


@torch.no_grad()
def project_visible_points(points, c2w, intrinsics, depth, depth_conf, image_hw,
                           tau_abs, tau_rel):
    batch, count = points.shape[:2]
    views = c2w.shape[1]
    height, width = image_hw
    ones = torch.ones((batch, count, 1), device=points.device, dtype=torch.float32)
    homogeneous = torch.cat([points.float(), ones], dim=-1)
    camera = torch.einsum("bvij,bnj->bvni", torch.linalg.inv(c2w.float()), homogeneous)
    z = camera[..., 2]
    k = intrinsics.float().clone()
    k[..., 0, :] *= width
    k[..., 1, :] *= height
    u = k[..., 0, 0, None] * camera[..., 0] / z.clamp_min(1e-8) + k[..., 0, 2, None]
    v = k[..., 1, 1, None] * camera[..., 1] / z.clamp_min(1e-8) + k[..., 1, 2, None]
    inside = (z > 0) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    gx = 2.0 * u / max(width - 1, 1) - 1.0
    gy = 2.0 * v / max(height - 1, 1) - 1.0
    image_grid = torch.stack([gx, gy], dim=-1).reshape(batch * views, count, 1, 2)

    if depth.ndim == 5:
        depth = depth.squeeze(-1)
    if depth_conf.ndim == 5:
        depth_conf = depth_conf.squeeze(-1)
    if depth.ndim != 4 or depth_conf.ndim != 4:
        raise RuntimeError(
            f"Unexpected depth/conf shapes {tuple(depth.shape)}, {tuple(depth_conf.shape)}"
        )
    depth_bv = depth.float().reshape(batch * views, 1, *depth.shape[-2:])
    confidence_bv = depth_conf.float().reshape(batch * views, 1, *depth_conf.shape[-2:])
    sampled_depth = F.grid_sample(
        depth_bv, image_grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    ).reshape(batch, views, count)
    sampled_conf = F.grid_sample(
        confidence_bv, image_grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    ).reshape(batch, views, count).clamp_min(0)
    tolerance = tau_abs + tau_rel * sampled_depth.abs()
    visible = inside & (sampled_depth > 0) & ((z - sampled_depth).abs() <= tolerance)
    weights = sampled_conf * visible.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    relative_depth_error = (z - sampled_depth).abs() / sampled_depth.abs().clamp_min(1e-6)
    return u, v, inside, visible, weights, relative_depth_error


@torch.no_grad()
def raw_reprojection_diagnostics(points_all, c2w, intrinsics, depth, stride=16):
    """Verify camera math on the unvoxelized points at their source pixels."""
    if depth.ndim == 5:
        depth = depth.squeeze(-1)
    batch, views, height, width = depth.shape
    sampled_points = points_all[:, :, ::stride, ::stride].float()
    sample_h, sample_w = sampled_points.shape[2:4]
    points = sampled_points.reshape(batch * views, sample_h * sample_w, 3)
    cameras = c2w.reshape(batch * views, 4, 4)
    world_to_camera = torch.linalg.inv(cameras)
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    camera = torch.einsum("bij,bnj->bni", world_to_camera, homogeneous)
    k = intrinsics.reshape(batch * views, 3, 3).float().clone()
    k[:, 0, :] *= width
    k[:, 1, :] *= height
    u = k[:, 0, 0, None] * camera[..., 0] / camera[..., 2].clamp_min(1e-8) + k[:, 0, 2, None]
    v = k[:, 1, 1, None] * camera[..., 1] / camera[..., 2].clamp_min(1e-8) + k[:, 1, 2, None]
    expected_v, expected_u = torch.meshgrid(
        torch.arange(0, height, stride, device=points.device)[:sample_h],
        torch.arange(0, width, stride, device=points.device)[:sample_w],
        indexing="ij",
    )
    expected_u = expected_u.reshape(1, -1).float()
    expected_v = expected_v.reshape(1, -1).float()
    pixel_error = ((u - expected_u).square() + (v - expected_v).square()).sqrt()
    source_depth = depth[:, :, ::stride, ::stride].reshape(batch * views, -1).float()
    depth_relative = (camera[..., 2] - source_depth).abs() / source_depth.abs().clamp_min(1e-6)
    valid = torch.isfinite(pixel_error) & torch.isfinite(depth_relative) & (source_depth > 0)
    return pixel_error[valid].mean(), pixel_error[valid].max(), depth_relative[valid].mean()


def pixel_to_feature_grid(u, v, original_hw, input_size, feature_hw):
    """Map original pixel centres to resized feature-cell centres."""
    height, width = original_hw
    feature_height, feature_width = feature_hw
    u_resized = (u + 0.5) * input_size / width - 0.5
    v_resized = (v + 0.5) * input_size / height - 0.5
    feature_x = (u_resized + 0.5) * feature_width / input_size - 0.5
    feature_y = (v_resized + 0.5) * feature_height / input_size - 0.5
    gx = 2.0 * feature_x / max(feature_width - 1, 1) - 1.0
    gy = 2.0 * feature_y / max(feature_height - 1, 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


@torch.no_grad()
def sample_and_fuse(feature_bv, u, v, visible, weights, original_hw, input_size):
    batch, views, count = visible.shape
    grid = pixel_to_feature_grid(
        u, v, original_hw, input_size, feature_bv.shape[-2:]
    ).reshape(batch * views, count, 1, 2)
    sampled = F.grid_sample(
        feature_bv.float(), grid, mode="bilinear", padding_mode="zeros",
        align_corners=True,
    ).squeeze(-1).reshape(batch, views, feature_bv.shape[1], count)
    sampled = sampled.permute(0, 3, 1, 2)
    fused = (sampled * weights.permute(0, 2, 1).unsqueeze(-1)).sum(dim=2)
    return fused


@torch.no_grad()
def extract_scene_semantics(encoder, images, fusion, romp, dino, semantic_args):
    """Extract feature maps once and produce the identity embedding."""
    height, width = images.shape[-2:]
    romp_feature = extract_romp_features(
        romp, images, semantic_args.romp_input_size,
        semantic_args.romp_feature_dim,
    )
    dino_feature = extract_dino_features(
        dino, images, semantic_args.dino_input_size,
        semantic_args.dino_feature_dim,
    )
    u, v, inside, visible, weights, depth_error = project_visible_points(
        encoder.gaussians.means,
        encoder.pred_context_pose["extrinsic"],
        encoder.pred_context_pose["intrinsic"],
        encoder.depth_dict["depth"], encoder.depth_dict["depth_conf"],
        (height, width), semantic_args.visibility_abs,
        semantic_args.visibility_rel,
    )
    visible &= encoder.infos["valid_point_mask"][:, None]
    weights *= visible.float()
    weights /= weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    romp_3d = sample_and_fuse(
        romp_feature, u, v, visible, weights,
        (height, width), semantic_args.romp_input_size,
    )
    dino_3d = sample_and_fuse(
        dino_feature, u, v, visible, weights,
        (height, width), semantic_args.dino_input_size,
    )
    embedding = fusion(encoder.infos["feature_3d"], romp_3d, dino_3d)
    visible_error = depth_error[visible]
    diagnostics = {
        "projection_inside_ratio": float(inside.float().mean()),
        "projection_visible_ratio": float(visible.float().mean()),
        "any_view_visible_ratio": float(visible.any(dim=1).float().mean()),
        "visible_depth_relative_error": (
            float(visible_error.mean()) if visible_error.numel() else 0.0
        ),
    }
    return embedding, visible.any(dim=1), romp_feature, dino_feature, diagnostics
