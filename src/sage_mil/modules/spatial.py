from __future__ import annotations

import torch
from torch import nn

try:
    from nystrom_attention import NystromAttention
except Exception:  # pragma: no cover
    NystromAttention = None


class TransLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        if NystromAttention is None:
            raise ImportError(
                "nystrom-attention is required for the spatial MIL encoder. "
                "Install the dependency listed in requirements.txt."
            )
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=max(16, dim // heads),
            heads=heads,
            num_landmarks=64,
            pinv_iterations=6,
            residual=True,
            dropout=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class SRU(nn.Module):
    """Spatial reconstruction unit from SCConv."""

    def __init__(self, channels: int, groups: int = 16, gate_threshold: float = 0.5):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SCConv SRU requires an even number of channels")
        groups = min(int(groups), int(channels))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.gate_threshold = float(gate_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gn_x = self.norm(x)
        gamma = self.norm.weight / self.norm.weight.sum().clamp_min(1e-6)
        gamma = gamma.view(1, -1, 1, 1)
        weights = torch.sigmoid(gn_x * gamma)

        informative = weights >= self.gate_threshold
        x_info = informative * x
        x_noninfo = (~informative) * x

        x11, x12 = torch.chunk(x_info, 2, dim=1)
        x21, x22 = torch.chunk(x_noninfo, 2, dim=1)
        return torch.cat([x11 + x22, x12 + x21], dim=1)


class CRU(nn.Module):
    """Channel reconstruction unit from SCConv."""

    def __init__(
        self,
        channels: int,
        alpha: float = 0.5,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ):
        super().__init__()
        upper = int(round(float(alpha) * channels))
        lower = channels - upper
        if upper <= 0 or lower <= 0:
            raise ValueError("SCConv CRU requires non-empty upper and lower channel groups")

        upper_squeezed = max(1, upper // int(squeeze_ratio))
        lower_squeezed = max(1, lower // int(squeeze_ratio))
        groups = min(int(group_size), upper_squeezed)
        while upper_squeezed % groups != 0 and groups > 1:
            groups -= 1

        self.upper = upper
        self.lower = lower
        self.squeeze_upper = nn.Conv2d(upper, upper_squeezed, kernel_size=1, bias=False)
        self.squeeze_lower = nn.Conv2d(lower, lower_squeezed, kernel_size=1, bias=False)
        self.group_conv = nn.Conv2d(
            upper_squeezed,
            channels,
            kernel_size=group_kernel_size,
            padding=group_kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.point_upper = nn.Conv2d(upper_squeezed, channels, kernel_size=1, bias=False)
        self.point_lower = nn.Conv2d(
            lower_squeezed,
            channels - lower_squeezed,
            kernel_size=1,
            bias=False,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upper, lower = torch.split(x, [self.upper, self.lower], dim=1)
        upper = self.squeeze_upper(upper)
        lower = self.squeeze_lower(lower)

        upper_out = self.group_conv(upper) + self.point_upper(upper)
        lower_out = torch.cat([self.point_lower(lower), lower], dim=1)
        merged = torch.cat([upper_out, lower_out], dim=1)

        weights = torch.softmax(self.pool(merged), dim=1)
        merged = merged * weights
        left, right = torch.chunk(merged, 2, dim=1)
        return left + right


class SCConvBlock(nn.Module):
    """Spatial and channel reconstruction convolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.sru = SRU(dim, groups=16, gate_threshold=0.5)
        self.cru = CRU(
            dim,
            alpha=0.5,
            squeeze_ratio=2,
            group_size=2,
            group_kernel_size=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cru(self.sru(x))


class PPEGSemanticSC(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = SCConvBlock(dim)

    def forward(self, seq: torch.Tensor, h: int, w: int, special: int = 4):
        special_tokens = seq[:, :special]
        patch_tokens = seq[:, special:]
        b, n, d = patch_tokens.shape
        expected = h * w
        if n < expected:
            pad = torch.zeros(b, expected - n, d, device=seq.device, dtype=seq.dtype)
            patch_tokens = torch.cat([patch_tokens, pad], dim=1)
        elif n > expected:
            patch_tokens = patch_tokens[:, :expected]
        grid = patch_tokens.transpose(1, 2).reshape(b, d, h, w)
        grid = self.block(grid)
        patch_tokens = grid.flatten(2).transpose(1, 2)
        return torch.cat([special_tokens, patch_tokens], dim=1)
