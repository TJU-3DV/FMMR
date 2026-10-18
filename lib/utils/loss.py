"""Training losses shared with the original pipeline scripts."""
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add

def mask_in_out_loss(alpha: Tensor, gt_mask: Tensor, w_out: float = 3.0, eps: float = 1e-6) -> Tensor:
    """
    alpha:   (B,V,1,H,W) in [0,1]
    gt_mask: (B,V,1,H,W) noisy mask
    """
    a = alpha.clamp(eps, 1 - eps)
    m = gt_mask.float()
    inside = (m > 0.5)
    outside = ~inside
    loss_in = F.binary_cross_entropy(a[inside], torch.ones_like(a[inside])) if inside.any() else a.sum() * 0.0
    loss_out = F.binary_cross_entropy(a[outside], torch.zeros_like(a[outside])) if outside.any() else a.sum() * 0.0
    return loss_in + w_out * loss_out


def voxel_smooth_loss(points, conf, voxel_size=0.01):
    """
    points: (B,N,3)
    conf:   (B,N,1)  [0,1]
    """
    B, N, _ = points.shape
    loss = points.new_zeros(())

    for b in range(B):
        p = points[b]
        c = conf[b].squeeze(-1)  # (N,)

        # voxel index
        vox = torch.floor(p / voxel_size).to(torch.int32)      # (N,3)
        # hash to 1D (simple but works; reduce collision by offset)
        # You can also use torch.unique on vox directly; hash is faster.
        minv = vox.min(dim=0).values
        vox0 = vox - minv
        # large strides
        key = vox0[:, 0] * 73856093 + vox0[:, 1] * 19349663 + vox0[:, 2] * 83492791  # (N,)

        uniq, inv = torch.unique(key, return_inverse=True)

        sum_c = scatter_add(c, inv, dim=0)                     # (M,)
        cnt   = scatter_add(torch.ones_like(c), inv, dim=0)    # (M,)
        mean_c = sum_c / (cnt + 1e-6)                          # (M,)

        # per-point voxel mean
        c_vox = mean_c[inv]                                    # (N,)

        loss = loss + (c - c_vox).pow(2).mean()

    return loss / B
