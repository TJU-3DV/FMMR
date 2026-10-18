from typing import Optional
import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn

from vggt.utils.type import Gaussians
from vggt.utils.geometry import get_world_rays
from einops import rearrange

# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Tensor, # *batch 4
    eps: float = 1e-8,
) -> Tensor: # *batch 3 3
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Tensor, # *#batch 3
    rotation_xyzw: Tensor, # *#batch 4
) -> Tensor: # *batch 3 3"
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )

class GaussianAdapter(nn.Module):
    def __init__(self, gaussian_scale_min=0.5, gaussian_scale_max=15.0, sh_degree=4):
        super().__init__()
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max
        self.sh_degree = sh_degree

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Tensor,# *#batch 4 4
        intrinsics: Tensor,# *#batch 3 3
        coordinates: Tensor,# *#batch 2
        depths: Tensor,# *#batch
        opacities: Tensor,# *#batch
        raw_gaussians: Tensor,# *#batch _
        image_shape: tuple[int, int],
        eps: float = 1e-8,
    ) -> Gaussians:
        device = extrinsics.device
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        
        # Map scale features to valid scale range.
        scale_min = self.gaussian_scale_min
        scale_max = self.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        h, w = image_shape
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        scales = scales * depths[..., None] * multiplier[..., None]

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        return Gaussians(
            means=means,
            covariances=covariances,
            # harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            harmonics=sh,
            opacities=opacities,
            # Note: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )
        
    def get_scale_multiplier(
        self,
        intrinsics,#:  *#batch 3 3
        pixel_size,#:  *#batch 2
        multiplier: float = 0.1,
    ) -> Tensor:#  *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh


class UnifiedGaussianAdapter(GaussianAdapter):
    def forward(
        self,
        means: Tensor,# *#batch 3
        # levels:  *#batch
        depths: Tensor,# *#batch
        opacities: Tensor,# *#batch
        raw_gaussians: Tensor,# *#batch _
        eps: float = 1e-8,
        intrinsics: Optional[Tensor] = None,# *#batch 3 3"]
        coordinates: Optional[Tensor] = None,# *#batch 2"]
    ) -> Gaussians:
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        
        scales = 0.001 * F.softplus(scales)
        scales = scales.clamp_max(0.3)
        
        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh))
        # dc = sh[..., :1]
        # rest = 0.5 * torch.tanh(sh[..., 1:])
        # sh = torch.cat([dc, rest], dim=-1)
        # print(scales.max())
        covariances = build_covariance(scales, rotations)
        
        return Gaussians(
            means=means.float(),
            # levels=levels.int(),
            covariances=covariances.float(),
            harmonics=sh.float(),
            opacities=opacities.float(),
            scales=scales.float(),
            rotations=rotations.float(),
        )


