import numpy as np
import pandas as pd

from evaluation.statistics import paired_bootstrap_ci
from audit.summarize_replacement import summarize_slides, method_summary
from audit.replacement_protocol import _true_class_probability


def test_bootstrap_order():
    mean, lo, hi = paired_bootstrap_ci(np.array([2, 3, 4]), np.array([1, 1, 1]), n_boot=100, seed=1)
    assert lo <= mean <= hi


def test_true_class_probability_binary():
    import torch

    p_pos = torch.tensor([0.8, 0.8])
    y = torch.tensor([1, 0])
    p_true = _true_class_probability(p_pos, y, 2)
    assert torch.allclose(p_true, torch.tensor([0.8, 0.2]))


def test_aupc_aopc_definition():
    rows = []
    for method, end in [("OT-Key", 0.6), ("OT-Random", 0.8)]:
        for tau, p in [(0.0, 1.0), (1.0, end)]:
            rows.append(
                {
                    "slide_id": "s1",
                    "branch": "bag",
                    "method": method,
                    "tau": tau,
                    "factual_probability": 1.0,
                    "counterfactual_probability": p,
                    "margin_drop": tau,
                }
            )
    out = summarize_slides(pd.DataFrame(rows))
    key = out[out.method == "OT-Key"].iloc[0]
    rnd = out[out.method == "OT-Random"].iloc[0]
    assert np.isclose(key.aupc, 0.8)
    assert np.isclose(key.aopc, 0.2)
    assert np.isclose(rnd.aopc, 0.1)
    assert np.isclose(key.aopcr, 2.0)
    pooled = method_summary(out)
    key_pooled = pooled[pooled.method == "OT-Key"].iloc[0]
    assert np.isclose(key_pooled.pooled_aopcr, 2.0)
