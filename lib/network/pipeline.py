"""VGGT Gaussian pipeline backbone used by human recognition."""
import copy
from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization
from torch import Tensor
from torch_scatter import scatter_add, scatter_max

from vggt.dependency.gaussian_adapter import UnifiedGaussianAdapter
from vggt.models.vggt import VGGT
from vggt.utils.geometry import batchify_unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

@dataclass
class Gaussians:
    means: Tensor
    delta: Tensor
    covariances: Tensor
    harmonics: Tensor
    opacities: Tensor
    scales: Tensor
    rotations: Tensor


@dataclass
class EncoderOutput:
    gaussians: Gaussians
    pred_pose_enc_list: list[Tensor] | None
    pred_context_pose: dict | None
    depth_dict: dict | None
    human_masks: Tensor | None     # keep as Tensor for backward compatibility (store human_logit)
    infos: dict | None
    distill_infos: dict | None


@dataclass
class DecoderOutput:
    color: Tensor
    depth: Tensor | None
    alpha: Tensor | None
    lod_rendering: dict | None


# =========================
# Config
# =========================

@dataclass
class PipelineConfig:
    ckpt_path: Optional[str] = "ckpt/VGGT-1B/model.pt"

    voxel_size: float = 0.004
    feature_dim: int = 128
    sh_degree: int = 4

    # patch start idx for vggt
    ps_idx: int = 5
    distill_ps_idx: int = 5

    # if True, uses a frozen distill branch to produce pose/depth/pts for stability (like your current code)
    use_distill: bool = True

    # open-set option (OFF by default)
    enable_open_set: bool = True
    embed_dim: int = 16

    # rasterization options
    near_plane: float = 1e-10
    radius_clip: float = 0.1
    rasterize_mode: str = "classic"


# =========================
# Small utils
# =========================

def pad_tensor_list(tensor_list: list[Tensor], pad_len: int, value: float) -> Tensor:
    """Pad a list of [Ni, ...] tensors to [pad_len, ...] then stack -> [B, pad_len, ...]."""
    padded = []
    for t in tensor_list:
        if t.shape[0] < pad_len:
            pad = torch.full(
                (pad_len - t.shape[0], *t.shape[1:]),
                value,
                device=t.device,
                dtype=t.dtype,
            )
            t = torch.cat([t, pad], dim=0)
        padded.append(t)
    return torch.stack(padded, dim=0)

class StableOpenSetHeads(nn.Module):
    """
    Outputs compatible with UnifiedGaussianAdapter:
      raw_gaussians = [scales(3), rotations(4), sh(3*d_sh)] => d_in = 7 + 3*d_sh
    Also outputs:
      human_logit: (B,N,1)
      inst_embed:  (B,N,D) if enable_open_set else None
    """
    def __init__(
        self,
        feature_dim: int = 128,
        sh_degree: int = 4,
        embed_dim: int = 16,
        enable_open_set: bool = False,
        hidden=(256, 128, 64),
        dropout=0.1,
    ):
        super().__init__()
        self.enable_open_set = enable_open_set
        self.d_sh = (sh_degree + 1) ** 2
        self.d_in = 7 + 3 * self.d_sh  # 82 when sh_degree=4

        trunk = []
        d = feature_dim
        for h in hidden:
            trunk += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*trunk)

        self.head_gs = nn.Linear(d, 1+7)
        self.head_human = nn.Linear(d, 1)

        self.head_embed = None
        if enable_open_set:
            self.head_embed = nn.Linear(d, embed_dim)

        nn.init.constant_(self.head_human.bias, -2.0)
        # nn.init.constant_(self.head_gs.bias[0], -1.0)

    def forward(self, feat_3d: Tensor):
        B, N, C = feat_3d.shape
        f = self.trunk(feat_3d.reshape(B * N, C)).reshape(B, N, -1)

        gs_feat = self.head_gs(f)
        human_logit = self.head_human(f).sigmoid()                            # (B,N,1)

        inst_embed = None
        if self.enable_open_set:
            inst_embed = F.normalize(self.head_embed(f), dim=-1)

        return gs_feat, human_logit, inst_embed


# =========================
# Pipeline
# =========================

class Pipeline3MLP(nn.Module):
    def __init__(self, cfg: PipelineConfig, training: bool = True):
        super().__init__()
        self.cfg = cfg
        self.training = training

        # --- VGGT backbone ---
        model_vggt = VGGT()
        if cfg.ckpt_path is not None:
            state = torch.load(cfg.ckpt_path, map_location="cpu")
            model_vggt.load_state_dict(state)

        # self.aggregator = model_vggt.aggregator
        # self.camera_head = model_vggt.camera_head
        self.depth_head = model_vggt.depth_head

        # Optional distill (frozen copy)
        if self.training and cfg.use_distill:
            self.distill_aggregator = copy.deepcopy(model_vggt.aggregator).eval()
            self.distill_camera_head = copy.deepcopy(model_vggt.camera_head).eval()
            self.distill_depth_head = copy.deepcopy(model_vggt.depth_head).eval()
            for m in [self.distill_aggregator, self.distill_camera_head, self.distill_depth_head]:
                for p in m.parameters():
                    p.requires_grad = False

        # --- Gaussian adapter (your definition) ---
        self.gaussian_adapter = UnifiedGaussianAdapter()

        # Sanity: d_in should match adapter
        expected_d_sh = (cfg.sh_degree + 1) ** 2
        expected_d_in = 7 + 3 * expected_d_sh
        assert self.gaussian_adapter.d_in == expected_d_in, f"Adapter d_in={self.gaussian_adapter.d_in} != expected {expected_d_in}"

        # --- Heads ---
        self.color_head = nn.Sequential(
            nn.Conv2d(cfg.feature_dim, cfg.feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.feature_dim, 3*expected_d_sh, kernel_size=1, stride=1, padding=0),
        )
        self.gaussian_head = StableOpenSetHeads(
            feature_dim=cfg.feature_dim,
            sh_degree=cfg.sh_degree,
            embed_dim=cfg.embed_dim,
            enable_open_set=cfg.enable_open_set,
        )

        self.register_buffer(
            "background_color",
            torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            persistent=False,
        )

    # -------------------------
    # opacity mapping (your function)
    # -------------------------
    def map_pdf_to_opacity(self, pdf: Tensor, global_step: int) -> Tensor:
        initial, final, warm_up = 0.0, 0.0, 1
        x = initial + min(global_step / warm_up, 1) * (final - initial)
        exponent = 2 ** x
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    # -------------------------
    # voxelize + fuse
    # -------------------------
    def voxelizaton_with_fusion(self, img_feat: Tensor, pts3d: Tensor, voxel_size: float, conf: Optional[Tensor] = None):
        """
        img_feat: (V,C,H,W)
        pts3d:    (V,3,H,W)
        conf:     (V,H,W)
        returns:
          voxel_pts:  (Nv,3)
          voxel_feat: (Nv,C)
        """
        V, C, H, W = img_feat.shape
        pts_flat = pts3d.permute(0, 2, 3, 1).flatten(0, 2)      # (V*H*W,3)
        vox_idx = (pts_flat / voxel_size).round().int()         # (V*H*W,3)
        _, inv_idx, _ = torch.unique(vox_idx, dim=0, return_inverse=True, return_counts=True)

        feat_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # (V*H*W,C)

        if conf is None:
            # uniform average
            ones = torch.ones((feat_flat.shape[0],), device=feat_flat.device, dtype=feat_flat.dtype)
            denom = scatter_add(ones, inv_idx, dim=0).clamp_min(1.0).unsqueeze(-1)
            voxel_pts = scatter_add(pts_flat, inv_idx, dim=0) / denom
            voxel_feat = scatter_add(feat_flat, inv_idx, dim=0) / denom
            return voxel_pts, voxel_feat

        conf_flat = conf.flatten()  # (V*H*W,)
        conf_max, _ = scatter_max(conf_flat, inv_idx, dim=0)
        conf_exp = torch.exp(conf_flat - conf_max[inv_idx])
        denom = scatter_add(conf_exp, inv_idx, dim=0).clamp_min(1e-6)
        w = (conf_exp / denom[inv_idx]).unsqueeze(-1)  # (V*H*W,1)

        voxel_pts = scatter_add(pts_flat * w, inv_idx, dim=0)    # (Nv,3)
        voxel_feat = scatter_add(feat_flat * w, inv_idx, dim=0)  # (Nv,C)
        return voxel_pts, voxel_feat
    
    # -------------------------
    # render gs (RGB + D) like your original, but cleaned and strictly shaped
    # -------------------------
    def render_gs(
        self,
        gaussians: Gaussians,
        extrinsics: Tensor,   # (B,V,4,4) c2w
        intrinsics: Tensor,   # (B,V,3,3) normalized
        image_shape: Tuple[int, int],
    ) -> DecoderOutput:
        B, V = intrinsics.shape[:2]
        H, W = image_shape

        rendered_colors = []
        rendered_depths = []
        rendered_alphas = []

        # harmonics: (B,N,3,d_sh) -> (B,N,d_sh,3)
        sh_all = gaussians.harmonics.permute(0, 1, 3, 2).contiguous()

        for b in range(B):
            # IMPORTANT: gsplat wants (N,3) not (1,N,3)
            xyz = gaussians.means[b].float()              # (N,3)
            rot = gaussians.rotations[b].float()          # (N,4)
            scale = gaussians.scales[b].float()           # (N,3)
            opa = gaussians.opacities[b].float()          # (N,) or (N,1)
            cov = gaussians.covariances[b].float()        # (N,3,3)

            # colors for SH: expect (C,N,d_sh,3). We'll render per-view so C=1.
            cols = sh_all[b].float().unsqueeze(0)         # (1,N,d_sh,3)
            sh_degree = int(sqrt(cols.shape[2])) - 1      # d_sh=(deg+1)^2

            # w2c, K per view
            w2c = extrinsics[b].float().inverse()         # (V,4,4)

            K_norm = intrinsics[b].float()
            K = K_norm.clone()
            K[:, 0] = K_norm[:, 0] * W
            K[:, 1] = K_norm[:, 1] * H

            view_colors, view_depths, view_alphas = [], [], []

            for v in range(V):
                rendering, alpha, _ = rasterization(
                    xyz, rot, scale, opa, cols,
                    w2c[v:v+1], K[v:v+1],
                    W, H,
                    sh_degree=sh_degree,
                    render_mode="RGB+D",
                    packed=False,
                    near_plane=self.cfg.near_plane,
                    backgrounds=self.background_color.unsqueeze(0),  # (1,3)
                    radius_clip=self.cfg.radius_clip,
                    covars=cov,
                    rasterize_mode=self.cfg.rasterize_mode,
                )
                # rendering expected: (1,H,W,4) or (H,W,4) depending on version.
                if rendering.dim() == 4:
                    rendering = rendering[0]
                if alpha.dim() == 4:
                    alpha = alpha[0]

                rgb, dep = torch.split(rendering, [3, 1], dim=-1)  # (H,W,3), (H,W,1)
                rgb = rgb.clamp(0, 1)

                view_colors.append(rgb.permute(2, 0, 1).unsqueeze(0))   # (1,3,H,W)
                view_depths.append(dep[..., 0].unsqueeze(0))            # (1,H,W)
                view_alphas.append(alpha[..., 0].unsqueeze(0))          # (1,H,W)

            rendered_colors.append(torch.cat(view_colors, dim=0))  # (V,3,H,W)
            rendered_depths.append(torch.cat(view_depths, dim=0))  # (V,H,W)
            rendered_alphas.append(torch.cat(view_alphas, dim=0))  # (V,H,W)
        

        return DecoderOutput(
            color=torch.stack(rendered_colors, dim=0),   # (B,V,3,H,W)
            depth=torch.stack(rendered_depths, dim=0),   # (B,V,H,W)
            alpha=torch.stack(rendered_alphas, dim=0),   # (B,V,H,W)
            lod_rendering=None,
        )

    def render_3d_mask(
        self,
        gaussians: Gaussians,
        human_conf: Tensor,      # (B,N,1) in [0,1]
        extrinsics: Tensor,      # (B,V,4,4) c2w
        intrinsics: Tensor,      # (B,V,3,3) normalized
        image_shape: Tuple[int, int],
        detach_gs: bool = True,
        opacity_tau: float = 0.03,
        opacity_sharpness: float = 120.0,
    ) -> Tensor:
        """
        Render human alpha mask using:
        mask_opacity = gaussians.opacities * human_conf

        Returns:
        (B,V,1,H,W) float32 in [0,1]
        """
        B, V = extrinsics.shape[:2]
        H, W = image_shape
        device = gaussians.means.device

        out = torch.zeros((B, V, 1, H, W), device=device, dtype=torch.float32)

        for b in range(B):
            # ---- camera matrices ----
            w2c = extrinsics[b].float().inverse()  # (V,4,4)

            K_norm = intrinsics[b].float()
            K = K_norm.clone()
            K[:, 0] = K_norm[:, 0] * W
            K[:, 1] = K_norm[:, 1] * H

            # ---- gaussians (NO batch dim) ----
            if detach_gs:
                xyz = gaussians.means[b].detach().float()            # (N,3)
                rot = gaussians.rotations[b].detach().float()        # (N,4)
                scl = gaussians.scales[b].detach().float()           # (N,3)
                cov = gaussians.covariances[b].detach().float()      # (N,3,3)
                opa = gaussians.opacities[b].detach().float()        # (N,) or (N,1)
            else:
                xyz = gaussians.means[b].float()            # (N,3)
                rot = gaussians.rotations[b].float()        # (N,4)
                scl = gaussians.scales[b].float()           # (N,3)
                cov = gaussians.covariances[b].float()      # (N,3,3)
                opa = gaussians.opacities[b].float()        # (N,) or (N,1)
            if opa.dim() == 2:
                opa = opa.squeeze(-1)                   # (N,)

            hc = human_conf[b].float()                  # (N,1) or (N,)
            if hc.dim() == 2:
                hc = hc.squeeze(-1)                     # (N,)
            # soft gate to suppress low-opacity noisy points
            gate = torch.sigmoid((opa - opacity_tau) * opacity_sharpness)  # (N,)
            mask_opa = opa * gate * hc                                     # (N,)

            # colors for sh_degree=0: (C,N,1,3), here C=1
            colors = torch.ones((1, xyz.shape[0], 1, 3), device=device, dtype=torch.float32)

            for v in range(V):
                _, alpha, _ = rasterization(
                    xyz, rot, scl, mask_opa, colors,
                    w2c[v:v+1], K[v:v+1],
                    W, H,
                    sh_degree=0,
                    render_mode="D",
                    packed=False,
                    near_plane=self.cfg.near_plane,
                    backgrounds=self.background_color.unsqueeze(0),
                    radius_clip=self.cfg.radius_clip,
                    covars=cov,
                    rasterize_mode=self.cfg.rasterize_mode,
                )

                # alpha could be (1,H,W,1) or (H,W,1) depending on gsplat
                if alpha.dim() == 4:
                    alpha = alpha[0]        # (H,W,1)
                out[b, v, 0] = alpha[..., 0]

        return out.clamp(0, 1)


    # -------------------------
    # forward
    # -------------------------
    def forward(
        self,
        images: Tensor,                 # (B,V,3,H,W) or (V,3,H,W)
        gt_mask: Optional[Tensor] = None,
        global_step: int = 0,
        debug: bool = False,
        render_output: bool = True,
    ) -> Tuple[EncoderOutput, Optional[DecoderOutput]]:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        device = images.device
        B, V, _, H, W = images.shape

        distill_infos: Dict[str, Any] = {}

        # -------- distill branch (pose/depth/pts) --------
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                tok_d = self.distill_aggregator(images.to(dtype=torch.bfloat16))
            with torch.amp.autocast("cuda", enabled=False):
                pose_list = self.distill_camera_head(tok_d)
                pose = pose_list[-1]
                extr, intr = pose_encoding_to_extri_intri(pose, images.shape[-2:])

                depth, depth_conf, feat2d = self.distill_depth_head(
                    tok_d, images=images, patch_start_idx=self.cfg.distill_ps_idx, with_feature=True
                )
                pts = batchify_unproject_depth_map_to_point_map(depth, extr, intr)

        with torch.amp.autocast("cuda", enabled=False):
            _, _, feat2d = self.depth_head(
                tok_d,
                images=images,
                patch_start_idx=self.cfg.distill_ps_idx,
                with_feature=True
            )

        distill_infos.update(
            pred_pose_enc_list=pose_list,
            extrinsic=extr,
            intrinsic=intr,
            depth_map=depth,
            depth_conf=depth_conf,
            pts_all=pts,
            # feat2d=feat2d,
        )        
        pred_pose_enc_list = distill_infos["pred_pose_enc_list"]
        extrinsic = distill_infos["extrinsic"]    # (B,V,3,4) or similar from vggt util
        intrinsic = distill_infos["intrinsic"]    # (B,V,3,3)
        depth_map = distill_infos["depth_map"]    # (B,V,H,W)
        depth_conf = distill_infos["depth_conf"]  # (B,V,H,W)
        pts_all = distill_infos["pts_all"]        # (B,V,H,W,3)
        # feat2d = distill_infos["feat2d"]          # (B,V,C,H,W)

        # -------- voxelize per batch --------
        colors = self.color_head(feat2d.view(B*V,*feat2d.shape[2:]))
        colors = colors.view(B,V,-1,H,W)
        feat2d = torch.cat([feat2d, colors], dim=2)
        neural_feats_list, neural_pts_list = [], []
        for b in range(B):
            # inputs: (V,C,H,W) and (V,3,H,W)
            voxel_pts, voxel_feat = self.voxelizaton_with_fusion(
                feat2d[b],
                pts_all[b].permute(0, 3, 1, 2).contiguous(),
                self.cfg.voxel_size,
                conf=depth_conf[b],
            )
            neural_feats_list.append(voxel_feat)
            neural_pts_list.append(voxel_pts)

        max_vox = max(t.shape[0] for t in neural_feats_list)
        voxel_counts = torch.tensor(
            [t.shape[0] for t in neural_feats_list],
            device=device,
            dtype=torch.long,
        )
        valid_point_mask = torch.arange(max_vox, device=device).unsqueeze(0) < voxel_counts.unsqueeze(1)
        feature_3d = pad_tensor_list(neural_feats_list, max_vox, value=-1e10)  # (B,N,C)
        neural_pts = pad_tensor_list(neural_pts_list, max_vox, value=-1e4)     # (B,N,3)

        # -------- heads -> opacity + gaussians + human --------
        feature_3d, gs_colors = feature_3d[...,:self.cfg.feature_dim], feature_3d[...,self.cfg.feature_dim:]
        gs_feat, human_logit, inst_embed = self.gaussian_head(feature_3d)
        gs_feat = torch.cat([gs_feat, gs_colors], dim=2)

        densities = gs_feat[...,0].sigmoid()              # (B,N)
        opacity = self.map_pdf_to_opacity(densities, global_step)    # (B,N) or (B,N,1)
        opacity = opacity.squeeze(-1) if opacity.dim() == 3 else opacity
        depths = neural_pts[..., -1].unsqueeze(-1)                   # (B,N,1)

        gaussians = self.gaussian_adapter.forward(
            neural_pts, depths, opacity, gs_feat[...,1:]
        )

        # -------- prepare context pose (same normalization as your old code) --------
        # Build 4x4 extrinsic: vggt extrinsic often is (B,V,3,4). We add last row then inverse to c2w like your code.
        extr_pad = torch.tensor([0, 0, 0, 1], device=device, dtype=extrinsic.dtype).view(1, 1, 1, 4).repeat(B, V, 1, 1)
        c2w = torch.cat([extrinsic, extr_pad], dim=2).inverse()  # (B,V,4,4)

        # Normalize intrinsics as you did: fx/W, fy/H
        intr_norm = intrinsic.clone()
        intr_norm = torch.stack([intr_norm[:, :, 0] / W, intr_norm[:, :, 1] / H, intr_norm[:, :, 2]], dim=2)  # (B,V,3,3)

        # -------- outputs --------
        encoder = EncoderOutput(
            gaussians=gaussians,
            pred_pose_enc_list=pred_pose_enc_list,
            pred_context_pose=dict(extrinsic=c2w, intrinsic=intr_norm),
            depth_dict=dict(depth=depth_map, depth_conf=depth_conf, neural_pts=neural_pts, pts_all=pts_all),
            human_masks=human_logit,  # store logit (Tensor) for single-person training
            infos=dict(
                scene_scale=neural_pts.reshape(B, -1, 3).norm(dim=-1).mean().clamp_min(1e-8).detach(),
                voxel_num=max_vox,
                enable_open_set=self.cfg.enable_open_set,
                inst_embed=inst_embed,  # optional, for future multi-person extension
                # Read-only interface consumed by the separately trained SMPL decoder.
                feature_3d=feature_3d,
                valid_point_mask=valid_point_mask,
                voxel_counts=voxel_counts,
            ),
            distill_infos=distill_infos,
        )

        # Stage3 SMPL training consumes only EncoderOutput.  Skipping gsplat
        # avoids unnecessary rasterization, camera inversion and GPU memory.
        decoder = self.render_gs(gaussians, c2w, intr_norm, (H, W)) if render_output else None

        if debug:
            print(f"[Pipeline3MLP] B={B} V={V} H={H} W={W} vox={max_vox}")
            print(f"  opacity: mean={opacity.mean().item():.4f} min={opacity.min().item():.4f} max={opacity.max().item():.4f}")
            # print(f"  human_conf: mean={hc.mean().item():.4f} min={hc.min().item():.4f} max={hc.max().item():.4f}")

        return encoder, decoder
