"""Unified Human Recognition, Semantic Embedding and SMPL Decoder pipeline."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lib.network.human_recognition import HumanRecognitionPipeline
from lib.network.pipeline import PipelineConfig
from lib.network.semantic import SemanticEmbeddingFusion, load_dinov3, load_romp
from lib.network.smpl_decoder import MultiFeatureDecoderConfig, VariablePointSMPLDecoder


class FMMRPipeline(nn.Module):
    """All three learned stages, with independently loaded frozen resources.

    The merged checkpoint includes VGGT, but excludes ROMP, DINOv3 and the
    SMPL body-model tensors. Those resources retain their own files in ckpt/.
    """

    def __init__(self, metadata: dict[str, Any], smpl_model_path: Path) -> None:
        super().__init__()
        self.metadata = metadata
        self.pipeline_step = int(metadata["human_global_step"])
        self.semantic_args = Namespace(**metadata["semantic_args"])
        human_config = dict(metadata["human_config"])
        human_config["ckpt_path"] = None
        self.human = HumanRecognitionPipeline(PipelineConfig(**human_config), training=True)
        self.semantic_fusion = SemanticEmbeddingFusion(**metadata["fusion_config"])
        decoder_config = dict(metadata["decoder_config"])
        decoder_config["smpl_model_path"] = str(smpl_model_path)
        self.decoder = VariablePointSMPLDecoder(MultiFeatureDecoderConfig(**decoder_config))

    @classmethod
    def from_pretrained(
        cls, checkpoint: str | Path = "ckpt/fmmr_pipeline.pth", *,
        smpl_model_path: str | Path = "ckpt/smpl/models",
        dino_weights: str | Path = (
            "ckpt/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        ),
        device: str | torch.device = "cuda:0",
    ) -> FMMRPipeline:
        """Load one merged learned checkpoint, then the external frozen models."""
        root = Path(__file__).resolve().parents[2]

        def resolve(value: str | Path) -> Path:
            path = Path(value).expanduser()
            return path if path.is_absolute() else root / path

        checkpoint_path = resolve(checkpoint)
        smpl_path = resolve(smpl_model_path)
        dino_path = resolve(dino_weights)
        if not smpl_path.is_dir():
            raise FileNotFoundError(smpl_path)
        for path in [checkpoint_path, dino_path]:
            if not path.is_file():
                raise FileNotFoundError(path)
        bundle = torch.load(
            str(checkpoint_path), map_location="cpu", mmap=True, weights_only=True,
        )
        if bundle.get("format") != "fmmr_pipeline_v1":
            raise ValueError("Expected a fmmr_pipeline_v1 merged checkpoint")
        model = cls(bundle["metadata"], smpl_path)
        missing, unexpected = model.load_state_dict(bundle["state_dict"], strict=False)
        missing = [key for key in missing if not key.startswith("decoder.body_model.")]
        if missing or unexpected:
            raise RuntimeError(f"Pipeline checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        del bundle
        model.to(device).eval().requires_grad_(False)
        model.semantic_args.dino_weights = str(dino_path)
        model.romp = load_romp(torch.device(device))
        model.dino = load_dinov3(model.semantic_args, torch.device(device))
        return model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: Tensor, args: dict[str, Any]) -> dict[str, Any]:
        """RGB [1,V,3,H,W], float [0,1] -> world-space SMPL and diagnostics.

        Images must already be on the pipeline device. Camera/geometry and
        grouping behavior match the original three-stage inference path.
        """
        from lib.utils.clustering import expand_scene_instances, infer_semantic_instances
        from lib.utils.features import extract_scene_semantics
        from lib.utils.person import build_person_inputs

        if images.ndim != 5 or images.shape[0] != 1 or images.shape[2] != 3:
            raise ValueError(f"Expected images [1,V,3,H,W], got {tuple(images.shape)}")
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            encoder, _ = self.human(
                images.float(), global_step=self.pipeline_step, render_output=False,
            )
        embedding, visible, romp_feature, dino_feature, projection = extract_scene_semantics(
            encoder, images.float(), self.semantic_fusion,
            self.romp, self.dino, self.semantic_args,
        )
        _, raw_labels, clustering = infer_semantic_instances(encoder, embedding, visible, args)
        selections, person_labels, expansion = expand_scene_instances(
            encoder, raw_labels, visible, args,
        )
        persons, retained_labels = [], []
        spatial_args = Namespace(**args)
        for selection, person_label in zip(selections, person_labels):
            built, _ = build_person_inputs(
                encoder, images, [selection], romp_feature, dino_feature,
                self.semantic_args, spatial_args,
            )
            if built:
                persons.append(built[0])
                retained_labels.append(person_label)
        if not persons:
            raise RuntimeError("No valid human instance was found for SMPL decoding")
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            prediction = self.decoder(persons)
        return {
            "encoder": encoder, "prediction": prediction,
            "raw_labels": raw_labels, "selections": selections,
            "person_labels": person_labels, "retained_labels": retained_labels,
            "projection": projection, "clustering": clustering, "expansion": expansion,
        }
