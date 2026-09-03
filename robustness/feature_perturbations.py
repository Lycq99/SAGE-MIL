from __future__ import annotations

import math
import torch


def _wsi_global_std(x: torch.Tensor) -> torch.Tensor:
    """Per-WSI standard deviation over all patch-feature elements."""
    if x.ndim != 3:
        raise ValueError(f"expected x with shape [B,N,D], got {tuple(x.shape)}")
    b = x.shape[0]
    return x.detach().float().reshape(b, -1).std(dim=1, correction=0).view(b, 1, 1).to(x)


def _dictionary_global_std(dictionary: torch.Tensor) -> torch.Tensor:
    """Global standard deviation of the complete prototype bank."""
    if dictionary.ndim != 2:
        raise ValueError(f"expected dictionary with shape [M,D], got {tuple(dictionary.shape)}")
    return dictionary.detach().float().std(correction=0).to(dictionary)


def gaussian_noise(x: torch.Tensor, severity: float = 0.1, generator=None) -> torch.Tensor:
    """X' = X + alpha * sigma(X) * Z, with sigma(X) computed per WSI."""
    alpha = float(severity)
    if alpha < 0:
        raise ValueError("Gaussian-noise severity must be non-negative")
    if alpha == 0:
        return x.clone()
    sigma = _wsi_global_std(x)
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return x + alpha * sigma * noise


def dictionary_noise(dictionary: torch.Tensor, severity: float = 0.1, generator=None) -> torch.Tensor:
    """D' = D + alpha * sigma(D) * Z_D over the full prototype bank."""
    alpha = float(severity)
    if alpha < 0:
        raise ValueError("dictionary-noise severity must be non-negative")
    if alpha == 0:
        return dictionary.clone()
    sigma = _dictionary_global_std(dictionary)
    noise = torch.randn(
        dictionary.shape,
        device=dictionary.device,
        dtype=dictionary.dtype,
        generator=generator,
    )
    return dictionary + alpha * sigma * noise


def channel_affine(x: torch.Tensor, severity: float = 0.1, generator=None) -> torch.Tensor:
    """Per-WSI, per-channel affine shift used in the robustness analysis."""
    if x.ndim != 3:
        raise ValueError(f"expected x with shape [B,N,D], got {tuple(x.shape)}")
    alpha = float(severity)
    if alpha < 0:
        raise ValueError("channel-affine severity must be non-negative")
    if alpha == 0:
        return x.clone()

    b, _, d = x.shape
    u = torch.randn((b, 1, d), device=x.device, dtype=x.dtype, generator=generator).clamp_(-2.0, 2.0)
    v = torch.randn((b, 1, d), device=x.device, dtype=x.dtype, generator=generator).clamp_(-2.0, 2.0)
    sigma_c = x.detach().float().std(dim=1, correction=0, keepdim=True).to(x)

    return x * (1.0 + 0.25 * alpha * u) + 0.25 * alpha * v * sigma_c


def background_mixing(
    x: torch.Tensor,
    reference: torch.Tensor,
    attention_score: torch.Tensor,
    severity: float = 0.1,
    generator=None,
    low_fraction: float = 0.5,
) -> torch.Tensor:
    """Mix the lowest-attention half of patches with randomly sampled proxies."""
    if x.ndim != 3:
        raise ValueError(f"expected x with shape [B,N,D], got {tuple(x.shape)}")
    if reference.ndim != 2 or reference.shape[-1] != x.shape[-1]:
        raise ValueError("reference must have shape [M,D] with D matching x")
    if attention_score.ndim != 2 or attention_score.shape != x.shape[:2]:
        raise ValueError("attention_score must have shape [B,N]")

    alpha = float(severity)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("background-mixing severity must be in [0,1]")
    if not 0.0 < float(low_fraction) <= 1.0:
        raise ValueError("low_fraction must be in (0,1]")
    if alpha == 0:
        return x.clone()

    b, n, d = x.shape
    k = max(1, int(math.floor(n * float(low_fraction))))
    low_idx = torch.topk(attention_score, k=k, dim=1, largest=False).indices
    gather = low_idx.unsqueeze(-1).expand(-1, -1, d)
    selected = torch.gather(x, 1, gather)

    proxy_idx = torch.randint(
        low=0,
        high=reference.shape[0],
        size=(b, k),
        device=x.device,
        generator=generator,
    )
    targets = reference.to(x)[proxy_idx]
    mixed = (1.0 - alpha) * selected + alpha * targets

    out = x.clone()
    out.scatter_(1, gather, mixed)
    return out
