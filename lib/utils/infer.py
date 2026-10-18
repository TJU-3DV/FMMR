"""End-to-end multi-view SMPL inference used by the demo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from lib.datasets.images import load_and_preprocess_images, select_views
from lib.network.fmmr import FMMRPipeline
from lib.utils.geometry import colors_from_input_images
from lib.utils.io import (
    save_parameters,
    save_smpl_reprojections,
    write_colored_ply,
    write_combined_obj,
    write_obj,
)


@torch.no_grad()
def infer_scene(
    args: Dict[str, Any], pipeline: FMMRPipeline,
    paths: Sequence[Path], device: torch.device,
) -> None:
    paths = select_views(paths, args["max_views"])
    if len(paths) < args["min_views"]:
        raise ValueError(f"Need at least {args['min_views']} views, found {len(paths)}")
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    images = load_and_preprocess_images([str(path) for path in paths]).unsqueeze(0).to(device)
    result = pipeline(images, args)
    encoder, prediction = result["encoder"], result["prediction"]
    raw_labels, selections = result["raw_labels"], result["selections"]
    person_labels, retained_labels = result["person_labels"], result["retained_labels"]
    decoder = pipeline.decoder

    colors, observations = colors_from_input_images(images.float(), encoder)
    points = encoder.gaussians.means[0].float()
    opacity = encoder.gaussians.opacities[0].reshape(-1).float()
    valid_scene = encoder.infos["valid_point_mask"][0] & (opacity >= args["scene_opacity_threshold"])
    expanded_labels = torch.zeros_like(raw_labels[0])
    for selection, person_label in zip(selections, person_labels):
        expanded_labels[selection["indices"]] = person_label
    human = expanded_labels > 0
    background = valid_scene & ~human
    write_colored_ply(
        output_dir / "scene_without_humans_colored.ply",
        points[background].cpu().numpy(),
        (colors[0, background] * 255.0).cpu().numpy(),
    )
    core_human = raw_labels[0] > 0

    faces = decoder.body_model.faces
    world_meshes = []
    for person_index, person_label in enumerate(retained_labels):
        vertices = prediction["vertices_world"][person_index, 0].float().cpu().numpy()
        world_meshes.append(vertices)
        write_obj(output_dir / f"person_{person_label - 1:02d}_smpl_world.obj", vertices, faces)
    write_combined_obj(output_dir / "smpl_all_world.obj", world_meshes, faces)
    save_parameters(output_dir / "smpl_parameters.npz", prediction, retained_labels)

    c2w = encoder.pred_context_pose["extrinsic"][0].float().cpu().numpy()
    intrinsic = encoder.pred_context_pose["intrinsic"][0].float().cpu().numpy()
    height, width = images.shape[-2:]
    intrinsic_pixel = intrinsic.copy()
    intrinsic_pixel[:, 0, :] *= width
    intrinsic_pixel[:, 1, :] *= height
    np.savez_compressed(
        output_dir / "cameras.npz", c2w=c2w, w2c=np.linalg.inv(c2w),
        intrinsics_normalized=intrinsic, intrinsics_pixel=intrinsic_pixel,
        image_paths=np.asarray([str(path) for path in paths]),
        image_size=np.asarray([height, width], dtype=np.int32),
    )
    smpl_reprojections = save_smpl_reprojections(
        output_dir, images.float(), paths, prediction, c2w, intrinsic_pixel,
        faces, args["reprojection_overlay_alpha"],
    )
    metadata = {
        "images": [str(path) for path in paths],
        "image_size": [height, width],
        "background_scene_points": int(background.sum()),
        "recognized_human_points": int(human.sum()),
        "recognized_human_core_points": int(core_human.sum()),
        "points_with_observed_color": int((observations[0, valid_scene] > 0).sum()),
        "decoded_person_count": len(retained_labels),
        "decoder_person_labels": retained_labels,
        "pipeline_checkpoint": args["checkpoint"],
        "pipeline_sources": pipeline.metadata["sources"],
        "decoder_architecture": pipeline.metadata["decoder_architecture"],
        "decoder_training_pipeline_checkpoint": pipeline.metadata.get(
            "decoder_training_pipeline_checkpoint"
        ),
        "smpl_reprojections": smpl_reprojections,
        **result["expansion"], **result["clustering"], **result["projection"],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
