from __future__ import annotations

import torch
import torch.nn.functional as F


def average_concept_embeddings(groups):
    """Average prompt embeddings within each concept and L2-normalize the result."""
    anchors = []
    for x in groups:
        x = F.normalize(torch.as_tensor(x).float(), dim=-1)
        anchors.append(F.normalize(x.mean(dim=0), dim=-1))
    return torch.stack(anchors, dim=0)
