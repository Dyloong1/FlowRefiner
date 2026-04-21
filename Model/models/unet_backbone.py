"""
3D U-Net backbone used by FlowRefiner.

Pre-norm residual blocks with zero-init on conv2 and the final output conv,
learned strided convolutions for down/up-sampling, and Fourier step
embeddings. Skip connections use concatenation.

The architectural pattern follows the "modern" 2D U-Net lineage
(Lippe et al. 2023; pdearena), lifted to 3D.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def fourier_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embedding (cos first, then sin)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock3D(nn.Module):
    """Pre-norm residual block with step conditioning.

    Pattern: norm -> act -> conv1 -> (add step emb) -> norm -> act -> zero_init(conv2).
    """

    def __init__(self, in_channels, out_channels, cond_channels, n_groups=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = zero_module(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
        self.act = nn.GELU()
        self.cond_emb = nn.Linear(cond_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


class Downsample3D(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv3d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class MiddleBlock3D(nn.Module):
    def __init__(self, n_channels, cond_channels, n_groups=1):
        super().__init__()
        self.res1 = ResidualBlock3D(n_channels, n_channels, cond_channels, n_groups)
        self.res2 = ResidualBlock3D(n_channels, n_channels, cond_channels, n_groups)

    def forward(self, x, emb):
        return self.res2(self.res1(x, emb), emb)


class RefinerUNet3D(nn.Module):
    """3D U-Net used as the refinement/base network in FlowRefiner.

    Structure:
        image_proj(3x3x3)
        -> [ResBlock x n_blocks + Downsample] x n_levels
        -> MiddleBlock
        -> [Upsample + ResBlock x (n_blocks+1)] x n_levels
        -> norm -> act -> zero_init conv(3x3x3)

    Skip connections via concatenation. Gradient checkpointing supported.
    """

    def __init__(self, in_channels, cond_channels, out_channels,
                 hidden_channels=64, ch_mults=(1, 2, 2, 4), n_blocks=2,
                 n_groups=1, gradient_checkpointing=False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gradient_checkpointing = gradient_checkpointing
        n_resolutions = len(ch_mults)

        # Step/time embedding
        time_embed_dim = hidden_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_channels, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.fourier_dim = hidden_channels

        total_in = in_channels + cond_channels
        self.image_proj = nn.Conv3d(total_in, hidden_channels, kernel_size=3, padding=1)

        # ---------------- Encoder ----------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [hidden_channels]
        ch_in = hidden_channels
        for i in range(n_resolutions):
            ch_out = hidden_channels * ch_mults[i]
            blocks = nn.ModuleList()
            for _ in range(n_blocks):
                blocks.append(ResidualBlock3D(ch_in, ch_out, time_embed_dim, n_groups))
                ch_in = ch_out
                channels.append(ch_out)
            self.down_blocks.append(blocks)
            if i < n_resolutions - 1:
                self.downsamples.append(Downsample3D(ch_out))
                channels.append(ch_out)
            else:
                self.downsamples.append(None)

        # ---------------- Middle ----------------
        self.middle = MiddleBlock3D(ch_in, time_embed_dim, n_groups)

        # ---------------- Decoder ----------------
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(n_resolutions)):
            ch_out = hidden_channels * ch_mults[i]
            if i < n_resolutions - 1:
                self.upsamples.append(Upsample3D(ch_in))
            else:
                self.upsamples.append(None)
            blocks = nn.ModuleList()
            for _ in range(n_blocks + 1):
                skip_ch = channels.pop()
                blocks.append(ResidualBlock3D(ch_in + skip_ch, ch_out, time_embed_dim, n_groups))
                ch_in = ch_out
            self.up_blocks.append(blocks)

        self.final_norm = nn.GroupNorm(n_groups, ch_in)
        self.final_act = nn.GELU()
        self.final_conv = zero_module(nn.Conv3d(ch_in, out_channels, kernel_size=3, padding=1))

    def forward(self, y_noised, condition, timestep):
        t_emb = fourier_embedding(timestep, self.fourier_dim)
        t_emb = self.time_embed(t_emb)

        h = torch.cat([y_noised, condition], dim=1)
        h = self.image_proj(h)

        use_ckpt = self.gradient_checkpointing and self.training

        skips = [h]
        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                if use_ckpt:
                    h = torch_checkpoint(block, h, t_emb, use_reentrant=False)
                else:
                    h = block(h, t_emb)
                skips.append(h)
            if downsample is not None:
                h = downsample(h)
                skips.append(h)

        if use_ckpt:
            h = torch_checkpoint(self.middle, h, t_emb, use_reentrant=False)
        else:
            h = self.middle(h, t_emb)

        for blocks, upsample in zip(self.up_blocks, self.upsamples):
            if upsample is not None:
                h = upsample(h)
            for block in blocks:
                skip = skips.pop()
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:],
                                      mode='trilinear', align_corners=False)
                h = torch.cat([h, skip], dim=1)
                if use_ckpt:
                    h = torch_checkpoint(block, h, t_emb, use_reentrant=False)
                else:
                    h = block(h, t_emb)

        return self.final_conv(self.final_act(self.final_norm(h)))
