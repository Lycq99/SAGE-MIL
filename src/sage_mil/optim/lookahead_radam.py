from __future__ import annotations

import torch


class RAdam(torch.optim.RAdam):
    """Thin alias used by the training configuration."""


class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, alpha: float = 0.5, k: int = 6):
        self.optimizer = optimizer
        self.alpha = float(alpha)
        self.k = int(k)
        self._step_count_lookahead = 0
        defaults = dict(lookahead_alpha=self.alpha, lookahead_k=self.k)
        super().__init__(optimizer.param_groups, defaults)
        self.slow_weights = []
        for group in self.optimizer.param_groups:
            slow_group = []
            for p in group["params"]:
                q = p.detach().clone()
                q.requires_grad = False
                slow_group.append(q)
            self.slow_weights.append(slow_group)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        if hasattr(self, "optimizer"):
            self.optimizer.param_groups = value

    def zero_grad(self, set_to_none: bool = False):
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step_count_lookahead += 1
        if self._step_count_lookahead % self.k == 0:
            for group, slow_group in zip(self.optimizer.param_groups, self.slow_weights):
                for p, q in zip(group["params"], slow_group):
                    if p.grad is None:
                        continue
                    q.add_(p.data - q, alpha=self.alpha)
                    p.data.copy_(q)
        return loss

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "slow_weights": [[q.cpu() for q in group] for group in self.slow_weights],
            "lookahead_step": self._step_count_lookahead,
            "alpha": self.alpha,
            "k": self.k,
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._step_count_lookahead = int(state_dict.get("lookahead_step", 0))
        for group, saved in zip(self.slow_weights, state_dict.get("slow_weights", [])):
            for q, src in zip(group, saved):
                q.copy_(src.to(q.device))
