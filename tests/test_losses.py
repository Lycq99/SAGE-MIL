import torch
from sage_mil.losses import true_class_margin, verification_hinge


def test_binary_margin():
    z = torch.tensor([2.0, -1.0])
    y = torch.tensor([1, 0])
    m = true_class_margin(z, y, 2)
    assert torch.allclose(m, torch.tensor([2.0, 1.0]))


def test_verification_hinge_nonnegative():
    x = verification_hinge(torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([0.5]), 0.25)
    assert x.item() >= 0.0
