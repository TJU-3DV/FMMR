"""Semantic instance embedding fusion and frozen appearance backbones."""
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticEmbeddingFusion(nn.Module):
    def __init__(self, vggt_dim=128, romp_dim=32, dino_dim=384, hidden=64, embed_dim=16):
        super().__init__()
        self.vggt = nn.Sequential(nn.Linear(vggt_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.romp = nn.Sequential(nn.Linear(romp_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.dino = nn.Sequential(nn.Linear(dino_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, vggt, romp, dino):
        fused = torch.cat([
            self.vggt(vggt.float()), self.romp(romp.float()), self.dino(dino.float())
        ], dim=-1)
        return F.normalize(self.fusion(fused), dim=-1)


def load_dinov3(args, device):
    from dinov3.hub import backbones
    builder = getattr(backbones, args.dino_model)
    model = builder(pretrained=True, weights=args.dino_weights).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_romp(device):
    from romp import ROMP
    model_dir = Path(__file__).resolve().parents[2] / "ckpt" / "romp"
    options = Namespace(
        mode="image", input=None, save_path="output/romp", GPU=0, onnx=False,
        temporal_optimize=False, center_thresh=0.25, show_largest=False,
        smooth_coeff=3.0, calc_smpl=True, render_mesh=False, renderer="sim3dr",
        show=False, show_items="mesh", save_video=False, frame_rate=24,
        smpl_path=str(model_dir / "SMPL_NEUTRAL.pth"),
        model_path=str(model_dir / "ROMP.pkl"),
        model_onnx_path=str(model_dir / "ROMP.onnx"), root_align=False,
        webcam_id=0,
    )
    model = ROMP(options).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
