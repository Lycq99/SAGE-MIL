from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(a, b, n_boot=10000, seed=2022, alpha=0.05):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("paired arrays must have the same length")
    rng = np.random.default_rng(seed)
    d = a - b
    samples = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def paired_randomization_p(a, b, n_perm=20000, seed=2022):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    count = 1
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(d))
        count += abs((d * signs).mean()) >= observed
    return float(count / (n_perm + 1))
