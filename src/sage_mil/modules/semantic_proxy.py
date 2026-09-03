from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class SemanticReferenceAttention(nn.Module):
    """Semantic response routing with a fixed normal-tissue reference dictionary."""

    def __init__(
        self,
        dim: int,
        lambda_env: float = 0.05,
        topk_ratio: float = 0.30,
        correction_mask=(1.0, 1.0, 0.0),
        num_heads: int = 1,
        proxy_temperature: float = 0.10,
        logit_scale_init: float = 1.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.num_heads < 1 or self.dim % self.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.head_dim = self.dim // self.num_heads
        self.lambda_env = float(lambda_env)
        self.topk_ratio = float(topk_ratio)
        self.proxy_temperature = float(proxy_temperature)
        if self.proxy_temperature <= 0:
            raise ValueError("proxy_temperature must be positive")

        gate = torch.tensor(correction_mask, dtype=torch.float32).view(1, 1, -1, 1)
        self.register_buffer("correction_mask", gate, persistent=True)

        # Learnable anchor projection, initialized as identity.
        self.anchor_projector = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.eye_(self.anchor_projector.weight)

        init = max(float(logit_scale_init), 1e-6)
        self.eta = nn.Parameter(torch.tensor(math.log(init), dtype=torch.float32))

    def _split_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # [B,N,D] -> [B,H,N,D_h]
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _split_anchors(self, x: torch.Tensor) -> torch.Tensor:
        # [B,3,D] -> [B,H,3,D_h]
        b, r, _ = x.shape
        return x.view(b, r, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, anchors, semantic_key, visual_value, proxy_key):
        if anchors.ndim != 3 or anchors.shape[1] != 3:
            raise ValueError("anchors must have shape [B,3,D]")

        q = self._split_anchors(self.anchor_projector(anchors))
        k = self._split_tokens(semantic_key)
        d = self._split_tokens(proxy_key)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        d = F.normalize(d, dim=-1)

        scale = torch.exp(self.eta).clamp(max=100.0)
        raw = scale * torch.einsum("bhrd,bhnd->bhrn", q, k)

        alloc_logits = torch.einsum("bhnd,bhmd->bhnm", k, d) / self.proxy_temperature
        allocation = torch.softmax(alloc_logits, dim=-1)
        reference = torch.einsum("bhnm,bhmd->bhnd", allocation, d)
        ref_response = scale * torch.einsum("bhrd,bhnd->bhrn", q, reference)

        adjusted = raw - self.lambda_env * self.correction_mask * ref_response

        n = adjusted.shape[-1]
        keep = max(1, min(n, int(math.ceil(self.topk_ratio * n))))
        idx = torch.topk(adjusted, keep, dim=-1).indices
        masked = torch.full_like(adjusted, float("-inf"))
        masked.scatter_(-1, idx, torch.gather(adjusted, -1, idx))
        head_attention = torch.softmax(masked, dim=-1)

        # Average head-wise attention before semantic pooling.
        attention = head_attention.mean(dim=1)
        summaries = torch.einsum("brn,bnd->brd", attention, visual_value)
        return summaries, attention
