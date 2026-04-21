"""Metrics used for FlowRefiner evaluation."""

import numpy as np
import torch
import torch.nn.functional as F


def compute_ssim_3d(pred, gt, win_size=7):
    """Compute SSIM for 3D volumes, averaged over timesteps and channels.

    Args:
        pred : (T, C, Z, H, W) denormalized prediction
        gt   : (T, C, Z, H, W) denormalized ground truth
        win_size : window size for local statistics

    Uses circular padding (periodic domain [0, 2pi]^3).

    Returns:
        float : mean SSIM across (T, C)
    """
    T, C = pred.shape[:2]
    vals = []
    for t in range(T):
        for c in range(C):
            p = pred[t, c]
            g = gt[t, c]
            data_range = max((g.max() - g.min()).item(),
                             (p.max() - p.min()).item())
            if data_range < 1e-12:
                vals.append(1.0)
                continue
            C1 = (0.01 * data_range) ** 2
            C2 = (0.03 * data_range) ** 2
            p4 = p.unsqueeze(0).unsqueeze(0)
            g4 = g.unsqueeze(0).unsqueeze(0)
            pad = win_size // 2
            p_pad = F.pad(p4, [pad]*6, mode='circular')
            g_pad = F.pad(g4, [pad]*6, mode='circular')
            kernel = torch.ones(1, 1, win_size, win_size, win_size,
                                device=p.device, dtype=p.dtype) / (win_size ** 3)
            mu_p = F.conv3d(p_pad, kernel)
            mu_g = F.conv3d(g_pad, kernel)
            mu_p2 = mu_p ** 2
            mu_g2 = mu_g ** 2
            mu_pg = mu_p * mu_g
            var_p = (F.conv3d(p_pad ** 2, kernel) - mu_p2).clamp(min=0)
            var_g = (F.conv3d(g_pad ** 2, kernel) - mu_g2).clamp(min=0)
            cov_pg = F.conv3d(p_pad * g_pad, kernel) - mu_pg
            num = (2 * mu_pg + C1) * (2 * cov_pg + C2)
            den = (mu_p2 + mu_g2 + C1) * (var_p + var_g + C2)
            vals.append((num / den).mean().item())
    return float(np.mean(vals))
