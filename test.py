"""Run the FMMR demo on arbitrary multi-view images.

Loads the merged FMMR pipeline and its frozen external resources,
then writes the background point cloud, SMPL meshes, camera
parameters and mesh reprojection overlays.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch

from lib.datasets.images import collect_images
from lib.network.fmmr import FMMRPipeline
from lib.utils.checkpoint import REPO_ROOT, repo_path
from lib.utils.infer import infer_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="input/logo")
    parser.add_argument("--output-dir", default="output/egohumans_logo")
    parser.add_argument("--checkpoint", default="ckpt/fmmr_pipeline.pth")
    parser.add_argument("--smpl-model-path", default="ckpt/smpl/models")
    parser.add_argument(
        "--dino-weights",
        default="ckpt/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument("--max-persons", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    os.chdir(REPO_ROOT)
    paths = collect_images([cli.inputs])
    checkpoint = repo_path(cli.checkpoint)
    dino_weights = repo_path(cli.dino_weights)
    for path in [checkpoint, dino_weights]:
        if not path.is_file():
            raise FileNotFoundError(path)
    smpl_model_path = repo_path(cli.smpl_model_path)
    if not smpl_model_path.exists():
        raise FileNotFoundError(smpl_model_path)
    device = torch.device(cli.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Inference requires CUDA")
    torch.cuda.set_device(device)
    torch.manual_seed(cli.seed)
    pipeline = FMMRPipeline.from_pretrained(
        checkpoint, smpl_model_path=smpl_model_path,
        dino_weights=dino_weights, device=device,
    )
    args: Dict[str, Any] = {
        "output_dir": str(repo_path(cli.output_dir)),
        "checkpoint": cli.checkpoint,
        "smpl_model_path": str(smpl_model_path),
        "device": cli.device,
        "seed": cli.seed,
        "min_views": cli.min_views,
        "max_views": cli.max_views,
        "max_persons": cli.max_persons,
        "human_threshold": 0.50,
        "human_opacity_threshold": 0.01,
        "min_view_support": 2,
        "min_pixel_support": 2,
        "semantic_split_eps": 0.35,
        "semantic_min_samples": 15,
        "semantic_split_min_silhouette": 0.15,
        "semantic_score_points": 4096,
        "min_person_points": 100,
        "xyz_cleanup_eps": 0.08,
        "xyz_cleanup_min_samples": 30,
        "person_point_count": 8192,
        "expanded_human_conf_threshold": 0.15,
        "expanded_opacity_threshold": 0.01,
        "expanded_min_view_support": 1,
        "expanded_min_pixel_support": 1,
        "expansion_max_distance": 0.08,
        "expansion_reference_points": 1024,
        "expansion_query_chunk": 8192,
        "normalized_voxel_size": 0.035,
        "maximum_voxel_tokens": 12000,
        "frame_trim_quantile": 0.95,
        "romp_match_max_distance": 0.25,
        "scene_opacity_threshold": 0.01,
        "reprojection_overlay_alpha": 0.75,
    }
    infer_scene(args, pipeline, paths, device)


if __name__ == "__main__":
    main()
