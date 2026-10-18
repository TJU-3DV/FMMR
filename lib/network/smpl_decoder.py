"""Spatial-expanded variable-length SMPL decoder."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import smplx
import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
from torch import Tensor

@dataclass
class MultiFeatureDecoderConfig:
    vggt_dim: int = 128
    romp_dim: int = 32
    dino_dim: int = 384
    view_dim: int = 128
    point_dim: int = 256
    hidden_dim: int = 256
    num_joints: int = 24
    n_betas: int = 10
    num_heads: int = 8

    # NEW: spatial-skeleton decoder depth
    num_decoder_layers: int = 3

    min_person_points: int = 32
    human_conf_power: float = 2.0
    scale_residual_limit: float = 2.0
    initial_radius_to_smpl_scale: float = 2.0
    smpl_model_path: str = "smpl/models"
    smpl_gender: str = "NEUTRAL"
    normalized_voxel_size: float = 0.035
    maximum_voxel_tokens: int = 12000
    frame_trim_quantile: float = 0.95


class ViewAwarePointEncoder(nn.Module):
    """Fuse a small variable set of visible views for every 3-D point."""

    def __init__(self, cfg: MultiFeatureDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vggt = nn.Sequential(
            nn.Linear(cfg.vggt_dim, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.romp = nn.Sequential(
            nn.Linear(cfg.romp_dim, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.dino = nn.Sequential(
            nn.Linear(cfg.dino_dim, 64), nn.LayerNorm(64), nn.GELU()
        )
        # Geometry: world-space camera ray (3), normalized uv (2), relative
        # depth error (1), sampled view confidence (1).
        self.view_value = nn.Sequential(
            nn.Linear(64 + 64 + 7, cfg.view_dim),
            nn.LayerNorm(cfg.view_dim), nn.GELU(),
            nn.Linear(cfg.view_dim, cfg.view_dim), nn.GELU(),
        )
        self.view_score = nn.Sequential(
            nn.Linear(cfg.view_dim + 128, 128), nn.GELU(), nn.Linear(128, 1)
        )
        # Additional fields: reliability, core flag and log voxel population.
        self.point = nn.Sequential(
            nn.Linear(128 + cfg.view_dim + 3 + 1 + 1 + 3, cfg.point_dim),
            nn.LayerNorm(cfg.point_dim), nn.GELU(),
            nn.Linear(cfg.point_dim, cfg.point_dim),
            nn.LayerNorm(cfg.point_dim), nn.GELU(),
        )

    def forward(
        self,
        vggt: Tensor,
        romp: Tensor,
        dino: Tensor,
        view_geometry: Tensor,
        visible: Tensor,
        view_weights: Tensor,
        xyz_local: Tensor,
        confidence: Tensor,
        opacity: Tensor,
        reliability: Tensor,
        core_mask: Tensor,
        voxel_population: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Point-major tensors: ROMP/DINO/geometry [N,V,C], visible [N,V].
        vggt_token = self.vggt(vggt.float())
        observation = self.view_value(torch.cat([
            self.romp(romp.float()), self.dino(dino.float()),
            view_geometry.float(),
        ], dim=-1))
        query = vggt_token[:, None].expand(-1, observation.shape[1], -1)
        logits = self.view_score(torch.cat([observation, query], dim=-1)).squeeze(-1)
        logits = logits + view_weights.clamp_min(1e-8).log()
        logits = logits.masked_fill(~visible, -1e4)
        attention = torch.softmax(logits.float(), dim=1).to(observation.dtype)
        attention = attention * visible.float()
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-6)
        fused_view = (observation * attention.unsqueeze(-1)).sum(dim=1)
        token = self.point(torch.cat([
            vggt_token, fused_view, xyz_local.float(),
            confidence.float().unsqueeze(-1), opacity.float().unsqueeze(-1),
            reliability.float().unsqueeze(-1),
            core_mask.float().unsqueeze(-1),
            voxel_population.float().clamp_min(1.0).log1p().unsqueeze(-1),
        ], dim=-1))
        return token, attention

class SpatialSkeletonDecoderLayer(nn.Module):
    """
    One refinement layer:

        joint/global queries
                |
                v
        cross-attention to spatial point tokens
                |
                v
        self-attention among joint/global queries
                |
                v
               FFN

    The point positional embedding is explicitly added to K/V,
    so spatial coordinates are not only buried inside the point MLP.

    We deliberately use full joint self-attention here instead of a
    hard skeleton adjacency mask. With only 25 queries, the cost is
    negligible, while distant but semantically related joints can
    still communicate.
    """

    def __init__(self, dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()

        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.cross_norm_out = nn.LayerNorm(dim)

        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.self_norm_out = nn.LayerNorm(dim)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.ffn_out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        queries: Tensor,
        point_tokens: Tensor,
        point_position: Tensor,
    ) -> Tensor:
        """
        queries: [1, 25, C]
        point_tokens: [1, N, C]
        point_position: [1, N, C]
        """

        # ----------------------------------------------------
        # 1. Spatial cross-attention
        # ----------------------------------------------------
        # Position is explicitly injected into point K/V.
        # The original point encoder already contains xyz_local,
        # but here xyz is exposed again at the attention level.
        spatial_points = point_tokens + point_position

        q = self.cross_norm_q(queries)
        kv = self.cross_norm_kv(spatial_points)
        attended, _ = self.cross_attention(q, kv, kv, need_weights=False)
        queries = self.cross_norm_out(queries + attended)

        # ----------------------------------------------------
        # 2. Joint self-attention
        # ----------------------------------------------------
        # All 24 SMPL joint tokens and the global token can communicate.
        q = self.self_norm(queries)
        attended, _ = self.self_attention(q, q, q, need_weights=False)
        queries = self.self_norm_out(queries + attended)

        # ----------------------------------------------------
        # 3. Feed-forward refinement
        # ----------------------------------------------------
        refined = self.ffn(self.ffn_norm(queries))
        queries = self.ffn_out_norm(queries + refined)

        return queries


class VariablePointSMPLDecoder(nn.Module):
    """
    Spatial-skeleton SMPL decoder.

    Compared with the previous decoder:

        old:
            point tokens
                -> one cross-attention
                -> pose

        new:
            point tokens + explicit xyz embedding
                -> cross-attention
                -> joint self-attention
                -> FFN
                -> repeated 3 times
                -> pose

    The external input/output interface is kept unchanged.
    """

    def __init__(self, cfg: MultiFeatureDecoderConfig) -> None:
        super().__init__()

        self.cfg = cfg

        # ----------------------------------------------------
        # Existing point encoder
        # ----------------------------------------------------
        self.point_encoder = ViewAwarePointEncoder(cfg)

        # ----------------------------------------------------
        # 24 SMPL joint queries + 1 global person query
        # ----------------------------------------------------
        self.queries = nn.Parameter(
            torch.randn(cfg.num_joints + 1, cfg.point_dim) * 0.02
        )

        # ----------------------------------------------------
        # NEW: Explicit spatial embedding for xyz_local.
        #
        # xyz_local is already normalized by:
        #   (xyz - person_center) / observed_scale
        #
        # so this MLP operates in a relatively stable space.
        # ----------------------------------------------------
        self.position_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, cfg.point_dim),
            nn.LayerNorm(cfg.point_dim),
        )

        # ----------------------------------------------------
        # NEW: Multi-layer spatial-skeleton decoder
        # ----------------------------------------------------
        self.decoder_layers = nn.ModuleList([
            SpatialSkeletonDecoderLayer(
                dim=cfg.point_dim,
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
            )
            for _ in range(cfg.num_decoder_layers)
        ])

        # ----------------------------------------------------
        # Pose / shape heads
        # ----------------------------------------------------
        self.pose_head = nn.Linear(cfg.point_dim, 6)
        self.betas_head = nn.Linear(cfg.point_dim, cfg.n_betas)

        # ----------------------------------------------------
        # Placement branch
        # ----------------------------------------------------
        self.placement_context = nn.Sequential(
            nn.Linear(5, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.translation_head = nn.Linear(cfg.point_dim + 32, 3)
        self.scale_head = nn.Linear(cfg.point_dim + 32, 1)

        self.log_radius_to_scale = nn.Parameter(
            torch.tensor(
                math.log(cfg.initial_radius_to_smpl_scale),
                dtype=torch.float32,
            )
        )

        # ----------------------------------------------------
        # SMPL
        # ----------------------------------------------------
        self.body_model = smplx.SMPL(
            cfg.smpl_model_path,
            gender=cfg.smpl_gender,
            batch_size=1,
            create_transl=False,
        )
        self.body_model.requires_grad_(False)

        # ----------------------------------------------------
        # Initialization
        #
        # Keep the original identity-pose initialization.
        # This is intentional: training starts from a valid
        # neutral SMPL rather than arbitrary rotations.
        # ----------------------------------------------------
        nn.init.zeros_(self.pose_head.weight)

        identity_6d = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            dtype=torch.float32,
        )

        with torch.no_grad():
            self.pose_head.bias.copy_(identity_6d)

        nn.init.zeros_(self.betas_head.weight)
        nn.init.zeros_(self.betas_head.bias)

        nn.init.zeros_(self.translation_head.weight)
        nn.init.zeros_(self.translation_head.bias)

        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(
        self,
        persons: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:

        results: Dict[str, List[Tensor]] = {}

        for person in persons:
            # ==================================================
            # 1. Build point tokens
            # ==================================================
            point_token, view_attention = self.point_encoder(
                person["vggt"],
                person["romp"],
                person["dino"],
                person["view_geometry"],
                person["visible"],
                person["view_weights"],
                person["xyz_local"],
                person["confidence"],
                person["opacity"],
                person["reliability"],
                person["core_mask"],
                person["voxel_population"],
            )

            # point_token: [N, 256]
            # xyz_local: [N, 3]

            # ==================================================
            # 2. Explicit spatial embedding
            # ==================================================
            point_position = self.position_encoder(person["xyz_local"].float())

            point_token_batched = point_token.unsqueeze(0)
            point_position_batched = point_position.unsqueeze(0)

            # ==================================================
            # 3. Initialize SMPL joint queries
            # ==================================================
            queries = self.queries.unsqueeze(0)

            # ==================================================
            # 4. Multi-stage spatial-skeleton decoding
            # ==================================================
            for layer in self.decoder_layers:
                queries = layer(
                    queries,
                    point_token_batched,
                    point_position_batched,
                )

            query_token = queries[0]
            joint_token = query_token[:self.cfg.num_joints]
            global_token = query_token[self.cfg.num_joints]

            # ==================================================
            # 5. Decode local SMPL rotations
            # ==================================================
            pose_6d = self.pose_head(joint_token)
            rotation = rotation_6d_to_matrix(pose_6d)
            pose = matrix_to_axis_angle(rotation)

            # ==================================================
            # 6. Shape
            # ==================================================
            betas = self.betas_head(global_token)

            # ==================================================
            # 7. Placement
            # Keep original implementation unchanged.
            # ==================================================
            placement_values = torch.stack([
                person["observed_scale"].clamp_min(1e-8).log(),
                (
                    person["core_scale"]
                    / person["observed_scale"].clamp_min(1e-8)
                ).clamp_min(1e-6).log(),
                person["core_fraction"],
                person["expanded_count"].float().clamp_min(1.0).log() / 12.0,
                person["mean_view_coverage"],
            ])

            placement_token = torch.cat([
                global_token,
                self.placement_context(placement_values),
            ])

            translation_residual = torch.tanh(
                self.translation_head(placement_token)
            )

            translation = (
                person["center"]
                + translation_residual * person["observed_scale"]
            )

            log_residual = self.scale_head(placement_token).clamp(
                -self.cfg.scale_residual_limit,
                self.cfg.scale_residual_limit,
            )

            scale = (
                person["observed_scale"]
                * torch.exp(self.log_radius_to_scale + log_residual)
            )

            reliability = (
                person["confidence"].pow(self.cfg.human_conf_power)
                * person["opacity"]
            )

            # ==================================================
            # 8. Preserve original output contract
            # ==================================================
            result = {
                "person_token": global_token,
                "pose_6d": pose_6d,
                "rotation_matrices": rotation,
                "pose_axis_angle": pose,
                "global_orient": pose[:1],
                "body_pose": pose[1:],
                "betas": betas,
                "translation": translation,
                "translation_residual": translation_residual,
                "scale": scale,
                "log_scale": scale.clamp_min(1e-8).log(),
                "person_center": person["center"],
                "person_scale": person["observed_scale"].reshape(1),
                "core_scale": person["core_scale"].reshape(1),
                "core_fraction": person["core_fraction"].reshape(1),
                "expanded_count": person["expanded_count"].reshape(1),
                "actual_voxel_size": person["actual_voxel_size"].reshape(1),
                "mean_view_coverage": person["mean_view_coverage"].reshape(1),
                "person_valid": torch.tensor(True, device=pose.device),
                "point_count": torch.tensor(
                    point_token.shape[0],
                    device=pose.device,
                    dtype=torch.long,
                ),
                "weight_sum": reliability.sum().reshape(1),
                "mean_view_entropy": (
                    -(view_attention.clamp_min(1e-8).log() * view_attention).sum(-1)
                ).mean(),
            }

            for key, value in result.items():
                results.setdefault(key, []).append(value)

        # ======================================================
        # Preserve original [B, P, ...] output contract
        # ======================================================
        output = {
            key: torch.stack(values, dim=0).unsqueeze(1)
            for key, values in results.items()
        }

        batch = output["betas"].shape[0]

        # ======================================================
        # SMPL forward
        # ======================================================
        body = self.body_model(
            global_orient=output["global_orient"].reshape(batch, 1, 3),
            body_pose=output["body_pose"].reshape(batch, 23, 3),
            betas=output["betas"].reshape(batch, self.cfg.n_betas),
            transl=None,
        )

        output["vertices_local"] = body.vertices[:, None]
        output["joints_local"] = body.joints[:, None]

        # ======================================================
        # World placement
        # ======================================================
        output["vertices_world"] = (
            output["vertices_local"] * output["scale"].unsqueeze(-1)
            + output["translation"].unsqueeze(-2)
        )

        output["joints_world"] = (
            output["joints_local"] * output["scale"].unsqueeze(-1)
            + output["translation"].unsqueeze(-2)
        )

        return output
