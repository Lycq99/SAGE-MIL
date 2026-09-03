import torch
from sage_mil.modules.ot_verification import OTVerifier


def test_ot_shapes():
    ot = OTVerifier(epsilon=0.05, iterations=3)
    x = torch.randn(1, 8, 4)
    a = torch.softmax(torch.randn(1, 3, 8), -1)
    p = torch.randn(5, 4)
    out, tau, idx, target = ot.replace(x, a, p, k_train=3, tau=torch.tensor([0.5]))
    assert out.shape == x.shape
    assert idx.shape == (1, 3)
    assert target.shape == (1, 3, 4)
    assert tau.shape == (1,)
