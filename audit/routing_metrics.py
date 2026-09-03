from __future__ import annotations

import numpy as np


def topk_overlap(a, b, k):
    ia = set(np.argsort(np.asarray(a))[-k:])
    ib = set(np.argsort(np.asarray(b))[-k:])
    return len(ia & ib) / max(1, k)


def jensen_shannon(p, q, eps=1e-12):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / max(p.sum(), eps); q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)
    kl1 = np.sum(p * np.log((p + eps) / (m + eps)))
    kl2 = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * (kl1 + kl2))
