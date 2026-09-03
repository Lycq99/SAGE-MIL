import torch

from sage_mil.modules.spatial import SCConvBlock


def test_scconv_shape():
    block = SCConvBlock(32)
    x = torch.randn(2, 32, 5, 5)
    y = block(x)
    assert y.shape == x.shape
