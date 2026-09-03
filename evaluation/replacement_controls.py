from __future__ import annotations

import torch


def zero_targets(candidates):
    return torch.zeros_like(candidates)


def bag_mean_targets(bag, candidates):
    return bag.detach().mean(dim=1, keepdim=True).expand_as(candidates)


def random_indices(n: int, k: int, device=None, generator=None):
    return torch.randperm(n, device=device, generator=generator)[: min(k, n)]
