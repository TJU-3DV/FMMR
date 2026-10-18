"""Checkpoint loading for the demo mainline stack."""
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from lib.network.human_recognition import HumanRecognitionPipeline
from lib.network.pipeline import PipelineConfig
from lib.network.semantic import SemanticEmbeddingFusion, load_dinov3, load_romp
from lib.network.smpl_decoder import MultiFeatureDecoderConfig, VariablePointSMPLDecoder

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def checkpoint_reference_matches(reference: str, requested: Path) -> bool:
    referenced = repo_path(reference).resolve()
    return referenced == requested.resolve()


def load_mainline_stack(
    semantic_path: Path, human_path: Path, vggt_path: Path,
    dino_path: Path, device: torch.device,
):
    checkpoint = torch.load(semantic_path, map_location="cpu")
    if checkpoint.get("embedding_architecture") != "vggt_romp_dinov3_fusion_v1":
        raise ValueError(
            f"Unexpected semantic architecture: {checkpoint.get('embedding_architecture')!r}"
        )
    semantic_args = Namespace(**checkpoint["args"])
    semantic_args.dino_weights = str(dino_path)
    human_state = torch.load(human_path, map_location="cpu")
    saved_args = human_state.get("args", {})
    config = PipelineConfig(
        ckpt_path=str(vggt_path),
        voxel_size=saved_args.get("voxel_size", 0.004),
        enable_open_set=True,
        use_distill=True,
    )
    pipeline = HumanRecognitionPipeline(config, training=True)
    pipeline.load_state_dict(human_state["net"], strict=True)
    pipeline.eval().requires_grad_(False)
    pipeline = pipeline.to(device)
    pipeline_step = int(human_state.get("global_step", 0))
    fusion = SemanticEmbeddingFusion(
        vggt_dim=128,
        romp_dim=semantic_args.romp_feature_dim,
        dino_dim=semantic_args.dino_feature_dim,
        embed_dim=16,
    ).to(device)
    fusion.load_state_dict(checkpoint["semantic_fusion"], strict=True)
    fusion.eval().requires_grad_(False)
    romp = load_romp(device)
    dino = load_dinov3(semantic_args, device)
    return pipeline, pipeline_step, fusion, romp, dino, semantic_args, checkpoint


def load_multifeature_decoder(
    path: Path, device: torch.device, smpl_model_path: str,
) -> Tuple[VariablePointSMPLDecoder, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    architecture = checkpoint.get("decoder_architecture")
    if architecture != "spatial_expanded_multiview_smpl_v3":
        raise ValueError(
            "Expected spatial_expanded_multiview_smpl_v3, "
            f"got {architecture!r}"
        )
    config = dict(checkpoint["config"])
    config["smpl_model_path"] = smpl_model_path
    decoder = VariablePointSMPLDecoder(MultiFeatureDecoderConfig(**config)).to(device)
    missing, unexpected = decoder.load_state_dict(
        checkpoint["decoder"], strict=False
    )
    missing = [key for key in missing if not key.startswith("body_model.")]
    if missing or unexpected:
        raise RuntimeError(
            f"Decoder checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    decoder.eval().requires_grad_(False)
    return decoder, checkpoint


def decoder_checkpoint_architecture(decoder: VariablePointSMPLDecoder) -> str:
    return "spatial_expanded_multiview_smpl_v3"
