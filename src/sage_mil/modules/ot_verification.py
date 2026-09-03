"""Continuous Sinkhorn-based feature replacement utilities."""
from __future__ import annotations

import math
import torch


class OTVerifier:
    def __init__(self, epsilon=0.05, iterations=30, norm_matching=True, clip_min=0.5, clip_max=2.0):
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)
        self.norm_matching = bool(norm_matching)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def barycenters(self, candidates: torch.Tensor, prototypes: torch.Tensor):
        b, k, _ = candidates.shape
        p = prototypes.detach().float().unsqueeze(0).expand(b, -1, -1)
        cost = torch.cdist(candidates.float(), p, p=2).pow(2)
        m = p.shape[1]
        log_a = torch.full((b, k), -math.log(float(k)), device=candidates.device)
        log_b = torch.full((b, m), -math.log(float(m)), device=candidates.device)
        log_kernel = -cost / max(self.epsilon, 1e-6)
        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)
        for _ in range(self.iterations):
            u = log_a - torch.logsumexp(log_kernel + v.unsqueeze(1), dim=-1)
            v = log_b - torch.logsumexp(log_kernel + u.unsqueeze(-1), dim=-2)
        coupling = torch.exp(log_kernel + u.unsqueeze(-1) + v.unsqueeze(-2))
        bary = torch.bmm(coupling, p) / coupling.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return bary.detach().to(candidates.dtype)

    def effective_targets(self, bag, candidates, barycenters):
        if not self.norm_matching:
            return barycenters
        bag_mean = bag.detach().mean(dim=1, keepdim=True)
        target_dist = (bag_mean - candidates).norm(dim=-1, keepdim=True)
        delta = barycenters - candidates
        scale = (target_dist / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(self.clip_min, self.clip_max)
        return candidates + scale * delta

    def replace(self, features, attention, prototypes, k_train=5, tau=None):
        b, n, d = features.shape
        score = attention.detach()[:, : min(2, attention.shape[1])].mean(dim=1)
        k = min(int(k_train), n)
        indices = torch.topk(score, k, dim=1).indices
        gather = indices.unsqueeze(-1).expand(-1, -1, d)
        candidates = torch.gather(features.detach(), 1, gather)
        targets = self.effective_targets(features, candidates, self.barycenters(candidates, prototypes))
        if tau is None:
            tau = torch.rand(b, 1, 1, device=features.device, dtype=features.dtype)
        elif tau.ndim == 1:
            tau = tau.view(b, 1, 1).to(features)
        blended = (1 - tau) * candidates + tau * targets
        out = features.clone()
        out.scatter_(1, gather, blended)
        return out, tau.view(b), indices, targets
