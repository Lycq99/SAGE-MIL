import torch
from sage_mil.losses import gram_orthogonality


def test_orthogonality_matches_frobenius_form():
    attention = torch.eye(3).unsqueeze(0)
    assert torch.allclose(gram_orthogonality(attention), torch.tensor(0.0))
