import torch
from sage_mil.modules.semantic_proxy import SemanticReferenceAttention


def test_semantic_response_shapes():
    layer = SemanticReferenceAttention(
        dim=8,
        lambda_env=0.05,
        topk_ratio=0.5,
        correction_mask=(1.0, 1.0, 0.0),
        num_heads=2,
        proxy_temperature=0.1,
    )
    anchors = torch.randn(2, 3, 8)
    keys = torch.randn(2, 6, 8)
    values = torch.randn(2, 6, 8)
    proxies = torch.randn(2, 5, 8)
    summaries, attention = layer(anchors, keys, values, proxies)
    assert summaries.shape == (2, 3, 8)
    assert attention.shape == (2, 3, 6)
    assert torch.allclose(attention.sum(-1), torch.ones(2, 3), atol=1e-5)


def test_microenvironment_mask_is_zero():
    layer = SemanticReferenceAttention(dim=4, correction_mask=(1.0, 1.0, 0.0))
    assert layer.correction_mask[0, 0, 2, 0].item() == 0.0
