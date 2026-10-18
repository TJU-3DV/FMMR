"""Trainable-depth human recognition pipeline."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization
from torch import Tensor
from torch_scatter import scatter_add, scatter_max

from lib.network.pipeline import EncoderOutput, Pipeline3MLP, PipelineConfig, pad_tensor_list
from vggt.utils.geometry import batchify_unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

SH_C0 = 0.28209479177387814


class HumanGaussianHeads(nn.Module):
    """Shape/opacity, binary-human logits, and open-set person embeddings."""

    def __init__(self, feature_dim: int = 128, embed_dim: int = 16):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(inplace=True),
        )
        # density + scale(3) + quaternion(4); colour is not predicted.
        self.head_gs = nn.Linear(64, 8)
        self.head_human = nn.Linear(64, 1)
        self.head_embed = nn.Linear(64, embed_dim)
        nn.init.constant_(self.head_human.bias, -2.0)
        nn.init.constant_(self.head_gs.bias[0], -1.0)

    def forward(self, feature_3d: Tensor):
        shape = feature_3d.shape[:2]
        feat = self.trunk(feature_3d.reshape(-1, feature_3d.shape[-1]))
        feat = feat.reshape(*shape, -1)
        gs = self.head_gs(feat)
        human_logits = self.head_human(feat)
        # Keep instance refinement from moving the shared trunk and degrading
        # the already trained binary-human classifier.  The embedding head
        # itself remains trainable and checkpoint-compatible.
        embedding = F.normalize(self.head_embed(feat.detach()), dim=-1)
        return gs, human_logits, embedding


class HumanRecognitionPipeline(Pipeline3MLP):
    """Pipeline3MLP geometry with sampled RGB and explicit point-quality data."""

    def __init__(self, cfg: PipelineConfig, training: bool = True):
        super().__init__(cfg, training=training)
        del self.color_head
        self.gaussian_head = HumanGaussianHeads(cfg.feature_dim, cfg.embed_dim)

    def render_3d_mask(
        self, gaussians, human_conf, extrinsics, intrinsics, image_shape,
        detach_gs=True, opacity_tau=0.02, opacity_sharpness=120.0,
    ):
        """Render human confidence while retaining full-scene occlusion.

        Non-human Gaussians have semantic colour zero but keep their scene
        opacity, so a chair or another foreground object correctly occludes a
        human Gaussian behind it.  Gradients flow only through human_conf when
        detach_gs is true.
        """
        batch, views = extrinsics.shape[:2]
        height, width = image_shape
        outputs = []
        for b in range(batch):
            xyz = gaussians.means[b]
            rotations = gaussians.rotations[b]
            scales = gaussians.scales[b]
            covariances = gaussians.covariances[b]
            opacity = gaussians.opacities[b].reshape(-1)
            if detach_gs:
                xyz = xyz.detach()
                rotations = rotations.detach()
                scales = scales.detach()
                covariances = covariances.detach()
                opacity = opacity.detach()
            xyz = xyz.float()
            rotations = rotations.float()
            scales = scales.float()
            covariances = covariances.float()
            opacity = opacity.float()
            scene_opacity = opacity * torch.sigmoid(
                (opacity - opacity_tau) * opacity_sharpness
            )
            semantic_colour = human_conf[b].float().reshape(-1, 1)

            w2c = extrinsics[b].float().inverse()
            k_pixels = intrinsics[b].float().clone()
            k_pixels[:, 0] *= width
            k_pixels[:, 1] *= height
            view_outputs = []
            for view in range(views):
                rendering, _, _ = rasterization(
                    xyz, rotations, scales, scene_opacity, semantic_colour,
                    w2c[view:view + 1], k_pixels[view:view + 1], width, height,
                    sh_degree=None, render_mode="RGB", packed=False,
                    near_plane=self.cfg.near_plane,
                    backgrounds=torch.zeros((1, 1), device=xyz.device),
                    radius_clip=self.cfg.radius_clip, covars=covariances,
                    rasterize_mode=self.cfg.rasterize_mode,
                )
                if rendering.dim() == 4:
                    rendering = rendering[0]
                view_outputs.append(rendering.permute(2, 0, 1))
            outputs.append(torch.stack(view_outputs, dim=0))
        return torch.stack(outputs, dim=0).clamp(0.0, 1.0)

    @staticmethod
    def _voxelize_with_rgb_and_support(
        feature: Tensor, rgb: Tensor, points: Tensor, confidence: Tensor,
        voxel_size: float,
    ):
        # feature: (V,C,H,W), rgb: (V,3,H,W), points: (V,3,H,W)
        views, _, height, width = feature.shape
        xyz = points.permute(0, 2, 3, 1).flatten(0, 2)
        feat = torch.cat([feature, rgb], dim=1).permute(0, 2, 3, 1).flatten(0, 2)
        conf = confidence.flatten().float()
        voxel_index = (xyz / voxel_size).round().int()
        _, inverse, counts = torch.unique(
            voxel_index, dim=0, return_inverse=True, return_counts=True
        )

        conf_max, _ = scatter_max(conf, inverse, dim=0)
        weights = torch.exp(conf - conf_max[inverse])
        denom = scatter_add(weights, inverse, dim=0).clamp_min(1e-6)
        weights = (weights / denom[inverse]).unsqueeze(-1)
        voxel_xyz = scatter_add(xyz * weights, inverse, dim=0)
        voxel_feat = scatter_add(feat * weights, inverse, dim=0)

        view_id = torch.arange(views, device=xyz.device).repeat_interleave(height * width)
        pair = torch.unique(inverse.long() * views + view_id.long())
        support = scatter_add(
            torch.ones_like(pair, dtype=torch.float32), pair // views,
            dim=0, dim_size=voxel_xyz.shape[0],
        )
        return voxel_xyz, voxel_feat, support, counts.float()

    def forward(
        self,
        images: Tensor,
        gt_mask: Optional[Tensor] = None,
        global_step: int = 0,
        debug: bool = False,
        render_output: bool = True,
    ):
        del gt_mask
        if images.dim() == 4:
            images = images.unsqueeze(0)
        batch, views, _, height, width = images.shape
        device = images.device

        # Frozen VGGT branch supplies camera/depth/geometry and detached tokens.
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                tokens = self.distill_aggregator(images.to(torch.bfloat16))
            with torch.amp.autocast("cuda", enabled=False):
                pose_list = self.distill_camera_head(tokens)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    pose_list[-1], images.shape[-2:]
                )
                depth, depth_conf, _ = self.distill_depth_head(
                    tokens, images=images,
                    patch_start_idx=self.cfg.distill_ps_idx,
                    with_feature=True,
                )
                points_all = batchify_unproject_depth_map_to_point_map(
                    depth, extrinsic, intrinsic
                )

        # Match the successful 12-30/1-13 lineage: adapt a separate depth head
        # for semantic features while leaving all reconstructed geometry frozen.
        # ``tokens`` is detached by the no_grad block above, hence gradients stop
        # at this head and cannot update the frozen aggregator.
        with torch.amp.autocast("cuda", enabled=False):
            _, _, feature_2d = self.depth_head(
                tokens,
                images=images,
                patch_start_idx=self.cfg.distill_ps_idx,
                with_feature=True,
            )

        point_lists, feature_lists, rgb_lists = [], [], []
        support_lists, count_lists = [], []
        for b in range(batch):
            xyz, fused, support, counts = self._voxelize_with_rgb_and_support(
                feature_2d[b].float(), images[b].float(),
                points_all[b].permute(0, 3, 1, 2).contiguous().float(),
                depth_conf[b].float(), self.cfg.voxel_size,
            )
            point_lists.append(xyz)
            feature_lists.append(fused[:, :self.cfg.feature_dim])
            rgb_lists.append(fused[:, self.cfg.feature_dim:self.cfg.feature_dim + 3].clamp(0, 1))
            support_lists.append(support)
            count_lists.append(counts)

        max_points = max(x.shape[0] for x in point_lists)
        voxel_counts = torch.tensor([x.shape[0] for x in point_lists], device=device)
        valid = torch.arange(max_points, device=device)[None] < voxel_counts[:, None]
        points = pad_tensor_list(point_lists, max_points, -1e4)
        feature_3d = pad_tensor_list(feature_lists, max_points, 0.0)
        sampled_rgb = pad_tensor_list(rgb_lists, max_points, 0.0)
        view_support = pad_tensor_list([x[:, None] for x in support_lists], max_points, 0.0).squeeze(-1)
        pixel_support = pad_tensor_list([x[:, None] for x in count_lists], max_points, 0.0).squeeze(-1)

        gs_pred, human_logits, embedding = self.gaussian_head(feature_3d)
        opacity = self.map_pdf_to_opacity(torch.sigmoid(gs_pred[..., 0]), global_step)
        # Padded points must never enter gsplat or any regularizer as real Gaussians.
        opacity = opacity * valid.to(opacity.dtype)

        # Fixed sampled colour as degree-0 SH; all view-dependent coefficients are zero.
        harmonics = torch.zeros(
            (*sampled_rgb.shape[:2], 3, (self.cfg.sh_degree + 1) ** 2),
            device=device, dtype=sampled_rgb.dtype,
        )
        harmonics[..., 0] = (sampled_rgb - 0.5) / SH_C0
        raw_harmonics = harmonics.reshape(batch, max_points, -1)
        raw_gaussians = torch.cat([gs_pred[..., 1:], raw_harmonics], dim=-1)
        gaussians = self.gaussian_adapter(
            points, points[..., -1:], opacity, raw_gaussians
        )

        extr_pad = torch.tensor([0, 0, 0, 1], device=device, dtype=extrinsic.dtype)
        extr_pad = extr_pad.view(1, 1, 1, 4).repeat(batch, views, 1, 1)
        c2w = torch.cat([extrinsic, extr_pad], dim=2).inverse()
        intr_norm = torch.stack(
            [intrinsic[:, :, 0] / width, intrinsic[:, :, 1] / height, intrinsic[:, :, 2]],
            dim=2,
        )
        human_conf = torch.sigmoid(human_logits)
        # This mask is for decoding/export. Low-support floaters remain differentiable
        # during training, but are never treated as valid inference points.
        inference_valid = (
            valid & (view_support >= 2) & (pixel_support >= 2)
            & (opacity.detach() >= 0.02)
        )

        encoder = EncoderOutput(
            gaussians=gaussians,
            pred_pose_enc_list=pose_list,
            pred_context_pose={"extrinsic": c2w, "intrinsic": intr_norm},
            depth_dict={
                "depth": depth, "depth_conf": depth_conf,
                "neural_pts": points, "pts_all": points_all,
            },
            human_masks=human_conf,
            infos={
                "human_logits": human_logits,
                "inst_embed": embedding,
                "feature_3d": feature_3d,
                "sampled_rgb": sampled_rgb,
                "valid_point_mask": valid,
                "inference_valid_point_mask": inference_valid,
                "view_support": view_support,
                "pixel_support": pixel_support,
                "voxel_counts": voxel_counts,
            },
            distill_infos={
                "extrinsic": extrinsic, "intrinsic": intrinsic,
                "depth_map": depth, "depth_conf": depth_conf,
                "pts_all": points_all,
            },
        )
        decoder = self.render_gs(gaussians, c2w, intr_norm, (height, width)) if render_output else None
        if debug:
            print(
                f"points={max_points} opacity={opacity[valid].mean().item():.4f} "
                f"human={human_conf[valid].mean().item():.4f} "
                f"support2={(view_support[valid] >= 2).float().mean().item():.4f}"
            )
        return encoder, decoder
