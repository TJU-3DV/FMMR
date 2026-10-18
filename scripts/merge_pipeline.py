"""Merge FMMR learned stages without optimizer or external body/backbone files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch


def merge_pipeline(
    human_path: Path, semantic_path: Path, decoder_path: Path,
    output: Path, root: Path,
) -> dict:
    """Save one tensor-exact checkpoint; preserve all original stage files."""
    if output.exists():
        raise FileExistsError(output)
    human = torch.load(str(human_path), map_location="cpu", mmap=True, weights_only=False)
    semantic = torch.load(str(semantic_path), map_location="cpu", mmap=True, weights_only=False)
    decoder = torch.load(str(decoder_path), map_location="cpu", mmap=True, weights_only=False)
    if semantic.get("embedding_architecture") != "vggt_romp_dinov3_fusion_v1":
        raise ValueError("Unexpected semantic architecture")
    if decoder.get("decoder_architecture") != "spatial_expanded_multiview_smpl_v3":
        raise ValueError("Unexpected decoder architecture")

    def resolve(reference: str) -> Path:
        path = Path(reference).expanduser()
        return (path if path.is_absolute() else root / path).resolve()

    if resolve(semantic["base_checkpoint"]) != human_path.resolve():
        raise ValueError("Semantic checkpoint does not reference the requested Human checkpoint")
    for key, requested in [("pipeline_checkpoint", human_path), ("semantic_checkpoint", semantic_path)]:
        reference = decoder.get("args", {}).get(key)
        if reference and resolve(reference) != requested.resolve():
            raise ValueError(f"Decoder {key} does not match the requested checkpoint")
    prefixes = {"depth_head", "distill_aggregator", "distill_camera_head", "distill_depth_head"}
    if not prefixes.issubset({key.split(".")[0] for key in human["net"]}):
        raise ValueError("Human checkpoint does not contain all required VGGT branches")
    state = {}
    for prefix, source in [
        ("human", human["net"]), ("semantic_fusion", semantic["semantic_fusion"]),
        ("decoder", decoder["decoder"]),
    ]:
        for name, tensor in source.items():
            if prefix == "decoder" and name.startswith("body_model."):
                continue
            if not torch.is_tensor(tensor):
                raise TypeError(f"Expected a tensor for {prefix}.{name}")
            state[f"{prefix}.{name}"] = tensor
    semantic_args = semantic["args"]
    metadata = {
        "human_config": {"voxel_size": human.get("args", {}).get("voxel_size", 0.004),
                         "enable_open_set": True, "use_distill": True},
        "human_global_step": int(human.get("global_step", 0)),
        "fusion_config": {"vggt_dim": 128, "romp_dim": semantic_args["romp_feature_dim"],
                          "dino_dim": semantic_args["dino_feature_dim"], "embed_dim": 16},
        "semantic_args": {key: semantic_args[key] for key in [
            "romp_input_size", "romp_feature_dim", "dino_input_size", "dino_feature_dim",
            "dino_model", "visibility_abs", "visibility_rel",
        ]},
        "decoder_config": {key: value for key, value in decoder["config"].items()
                           if key != "smpl_model_path"},
        "decoder_architecture": decoder["decoder_architecture"],
        "decoder_training_pipeline_checkpoint": decoder.get("args", {}).get("pipeline_checkpoint"),
        "sources": {"human": str(human_path.relative_to(root)),
                    "semantic": str(semantic_path.relative_to(root)),
                    "decoder": str(decoder_path.relative_to(root))},
        "includes_vggt": True,
        "excluded": ["ROMP", "DINOv3", "SMPL body model", "optimizer", "scaler"],
    }
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < total_bytes + 128 * 1024**2:
        raise OSError(f"Not enough free space to write {total_bytes} bytes to {output.parent}")
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save({"format": "fmmr_pipeline_v1", "metadata": metadata, "state_dict": state}, temporary)
    saved = torch.load(str(temporary), map_location="cpu", mmap=True, weights_only=True)
    if set(saved["state_dict"]) != set(state):
        raise RuntimeError("Merged checkpoint key mismatch")
    for name, tensor in state.items():
        restored = saved["state_dict"][name]
        if restored.dtype != tensor.dtype or not torch.equal(tensor, restored):
            raise RuntimeError(f"Merged checkpoint tensor mismatch: {name}")
    del saved
    temporary.rename(output)
    report = {"format": "fmmr_pipeline_v1", "tensor_count": len(state),
              "tensor_bytes": total_bytes, "file_bytes": output.stat().st_size,
              "verified_tensor_exact": True, **metadata}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", default="ckpt/pipeline4_human_trainable_depth_full/008.pth")
    parser.add_argument("--semantic", default="ckpt/pipeline4_embedding_semantic_full/010.pth")
    parser.add_argument("--decoder", default=(
        "ckpt/pipeline4_smpl_decoder_spatial_expanded_multi_supervised_2d_full/latest.pth"
    ))
    parser.add_argument("--output", default="ckpt/fmmr_pipeline.pth")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def path(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else root / candidate

    report = merge_pipeline(path(args.human), path(args.semantic), path(args.decoder), path(args.output), root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
