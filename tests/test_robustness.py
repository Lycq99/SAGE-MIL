import math
import torch

from robustness.feature_perturbations import (
    gaussian_noise,
    channel_affine,
    background_mixing,
    dictionary_noise,
)


def test_gaussian_noise_uses_per_wsi_global_std():
    x = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[0.0, 2.0], [4.0, 6.0]]])
    alpha = 0.2
    g1 = torch.Generator().manual_seed(7)
    out = gaussian_noise(x, alpha, g1)

    g2 = torch.Generator().manual_seed(7)
    z = torch.randn(x.shape, generator=g2)
    sigma = x.reshape(2, -1).std(dim=1, correction=0).view(2, 1, 1)
    expected = x + alpha * sigma * z
    assert torch.allclose(out, expected)


def test_dictionary_noise_uses_full_bank_std():
    d = torch.tensor([[0.0, 1.0], [2.0, 5.0], [3.0, 7.0]])
    alpha = 0.3
    g1 = torch.Generator().manual_seed(11)
    out = dictionary_noise(d, alpha, g1)

    g2 = torch.Generator().manual_seed(11)
    z = torch.randn(d.shape, generator=g2)
    expected = d + alpha * d.std(correction=0) * z
    assert torch.allclose(out, expected)


def test_channel_affine_matches_truncated_gaussian_definition():
    x = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]])
    alpha = 0.4
    g1 = torch.Generator().manual_seed(19)
    out = channel_affine(x, alpha, g1)

    g2 = torch.Generator().manual_seed(19)
    u = torch.randn((1, 1, 2), generator=g2).clamp(-2.0, 2.0)
    v = torch.randn((1, 1, 2), generator=g2).clamp(-2.0, 2.0)
    sigma_c = x.std(dim=1, correction=0, keepdim=True)
    expected = x * (1.0 + 0.25 * alpha * u) + 0.25 * alpha * v * sigma_c
    assert torch.allclose(out, expected)


def test_background_mixing_changes_only_lowest_half_attention_patches():
    x = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])
    reference = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    score = torch.tensor([[0.10, 0.20, 0.80, 0.90]])
    alpha = 0.5

    g1 = torch.Generator().manual_seed(23)
    out = background_mixing(x, reference, score, alpha, g1, low_fraction=0.5)

    g2 = torch.Generator().manual_seed(23)
    k = math.floor(x.shape[1] * 0.5)
    proxy_idx = torch.randint(0, reference.shape[0], (1, k), generator=g2)
    expected = x.clone()
    expected[:, 0] = (1.0 - alpha) * x[:, 0] + alpha * reference[proxy_idx[:, 0]]
    expected[:, 1] = (1.0 - alpha) * x[:, 1] + alpha * reference[proxy_idx[:, 1]]

    assert torch.allclose(out, expected)
    assert torch.allclose(out[:, 2:], x[:, 2:])


def test_zero_severity_is_identity():
    x = torch.randn(2, 5, 4)
    ref = torch.randn(7, 4)
    score = torch.rand(2, 5)
    assert torch.allclose(gaussian_noise(x, 0.0), x)
    assert torch.allclose(channel_affine(x, 0.0), x)
    assert torch.allclose(background_mixing(x, ref, score, 0.0), x)
    assert torch.allclose(dictionary_noise(ref, 0.0), ref)
