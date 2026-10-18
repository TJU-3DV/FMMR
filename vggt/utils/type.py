from dataclasses import dataclass
from torch import Tensor, nn


@dataclass
class Gaussians:
    means: Tensor # batch gaussian dim
    # delta: Tensor
    covariances: Tensor # batch gaussian dim dim
    harmonics: Tensor # batch gaussian 3 d_sh
    opacities: Tensor # batch gaussian
    scales: Tensor # batch gaussian 3
    rotations: Tensor # batch gaussian 4
    # levels:  batch gaussian"]

@dataclass
class EncoderOutput:
    gaussians: Gaussians
    pred_pose_enc_list: list[Tensor] | None #"batch view 6"
    pred_context_pose: dict | None
    depth_dict: dict | None
    human_masks: dict | None
    infos: dict | None
    distill_infos: dict | None


@dataclass
class DecoderOutput:
    color: Tensor
    depth: Tensor | None
    alpha: Tensor | None
    lod_rendering: dict | None