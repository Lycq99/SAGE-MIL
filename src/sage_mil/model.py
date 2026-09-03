"""SAGE-MIL model."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import math
import torch
from torch import nn

from .modules.semantic_proxy import SemanticReferenceAttention
from .modules.spatial import TransLayer, PPEGSemanticSC
from .modules.ot_verification import OTVerifier
from .losses import gram_orthogonality


def _load_tensor(path: Union[str, Path], name: str) -> torch.Tensor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")
    obj = torch.load(p, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("prototypes", "features", "anchors", "embeddings", "tensor"):
            if key in obj:
                obj = obj[key]
                break
    if not torch.is_tensor(obj):
        raise TypeError(f"{name} must resolve to a torch.Tensor, got {type(obj)!r}")
    return obj.float()


class SemanticTransMIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        model_dim: int = 512,
        num_classes: int = 2,
        anchors_path: str = "",
        proxy_path: str = "",
        proxy_num_prototypes: Optional[int] = None,
        train_key_patches: int = 5,
        lambda_env: float = 0.05,
        correction_mask=(1.0, 1.0, 0.0),
        topk_ratio: float = 0.30,
        semantic_heads: int = 1,
        proxy_temperature: float = 0.10,
        semantic_logit_scale_init: float = 1.0,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 30,
        target_norm_matching: bool = True,
        norm_match_clip=(0.5, 2.0),
    ):
        super().__init__()
        self.dim = int(model_dim)
        self.num_classes = int(num_classes)
        self.train_key_patches = int(train_key_patches)
        self.image_projector = nn.Sequential(nn.Linear(input_dim, model_dim), nn.LayerNorm(model_dim))
        self.semantic_adapter = nn.Sequential(
            nn.Linear(input_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim), nn.LayerNorm(model_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        anchors = _load_tensor(anchors_path, "semantic anchors")
        if anchors.ndim == 2:
            anchors = anchors.unsqueeze(0)
        if anchors.ndim != 3 or anchors.shape[1] != 3:
            raise ValueError(f"expected anchors shaped [3,d] or [1,3,d], got {tuple(anchors.shape)}")
        if anchors.shape[-1] != model_dim:
            raise ValueError("stored anchors must match model_dim")
        self.register_buffer("anchors", anchors, persistent=True)

        proxies = _load_tensor(proxy_path, "proxy dictionary")
        if proxies.ndim != 2 or proxies.shape[-1] != input_dim:
            raise ValueError(f"expected proxy dictionary [M,{input_dim}], got {tuple(proxies.shape)}")
        if proxy_num_prototypes is not None and proxies.shape[0] != int(proxy_num_prototypes):
            raise ValueError(
                f"expected {int(proxy_num_prototypes)} proxy prototypes, got {proxies.shape[0]}"
            )
        self.register_buffer("proxy_dictionary", proxies, persistent=True)

        self.semantic_query_attn = SemanticReferenceAttention(
            model_dim,
            lambda_env=lambda_env,
            topk_ratio=topk_ratio,
            correction_mask=correction_mask,
            num_heads=semantic_heads,
            proxy_temperature=proxy_temperature,
            logit_scale_init=semantic_logit_scale_init,
        )
        self.semantic_norm = nn.LayerNorm(model_dim)
        self.layer1 = TransLayer(model_dim)
        self.pos_layer = PPEGSemanticSC(model_dim)
        self.layer2 = TransLayer(model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.feature_fusion = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(model_dim, model_dim),
        )
        out_dim = 1 if self.num_classes <= 2 else self.num_classes
        self.final_classifier = nn.Linear(model_dim, out_dim)
        self.instance_classifier = nn.Linear(model_dim, out_dim)
        self.ot = OTVerifier(sinkhorn_epsilon, sinkhorn_iterations, target_norm_matching, *norm_match_clip)

    def _encode_semantics(self, features: torch.Tensor):
        b = features.shape[0]
        key_d = self.semantic_adapter(self.proxy_dictionary.unsqueeze(0).expand(b, -1, -1))
        visual = self.image_projector(features)
        semantic_key = self.semantic_adapter(features)
        anchors = self.anchors.expand(b, -1, -1)
        summaries, attention = self.semantic_query_attn(anchors, semantic_key, visual, key_d)
        return visual, self.semantic_norm(summaries), attention

    def _spatial(self, visual: torch.Tensor, summaries: torch.Tensor, n: int):
        b = visual.shape[0]
        h = int(math.ceil(math.sqrt(n)))
        w = h
        pad = h * w - n
        if pad > 0:
            visual = torch.cat(
                [visual, torch.zeros(b, pad, self.dim, device=visual.device, dtype=visual.dtype)],
                dim=1,
            )
        seq = torch.cat([self.cls_token.expand(b, -1, -1), summaries, visual], dim=1)
        special = 4
        seq = self.layer1(seq)
        seq = self.pos_layer(seq, h, w, special)
        seq = self.layer2(seq)
        seq = self.norm(seq)
        cls = seq[:, 0]
        sem = seq[:, 1:special].mean(dim=1)
        fused = self.feature_fusion(torch.cat([cls, sem], dim=1)) + cls
        bag = self.final_classifier(fused)
        inst_raw = self.instance_classifier(seq[:, special : special + n])
        if self.num_classes <= 2:
            bag = bag.squeeze(-1)
            inst_raw = inst_raw.squeeze(-1)
        return bag, inst_raw

    @staticmethod
    def _topk_instance_logits(inst_raw: torch.Tensor, k: int) -> torch.Tensor:
        k = min(inst_raw.shape[1], max(1, int(k)))
        vals = torch.topk(inst_raw, k, dim=1).values
        return vals.mean(dim=1) if k > 1 else vals[:, 0]

    @staticmethod
    def instance_topk(current_epoch: int, max_epochs: int, training: bool) -> int:
        if not training:
            return 6
        progress = min(1.0, max(0.0, current_epoch / max(1, max_epochs)))
        return max(1, int(10 - 9 * progress))

    def forward(
        self,
        features: torch.Tensor,
        coords=None,
        training: bool = False,
        current_epoch: int = 0,
        max_epochs: int = 25,
    ):
        if features.ndim != 3:
            raise ValueError(f"features must have shape [B,N,D], got {tuple(features.shape)}")
        if training:
            features = features + torch.randn_like(features) * 0.005

        b, n, _ = features.shape
        visual, summaries, attention = self._encode_semantics(features)
        bag, inst_raw = self._spatial(visual, summaries, n)
        inst = self._topk_instance_logits(inst_raw, self.instance_topk(current_epoch, max_epochs, training))

        cf_bag = None
        tau = torch.zeros(b, device=features.device, dtype=features.dtype)
        if training:
            features_cf, tau, _, _ = self.ot.replace(
                features,
                attention,
                self.proxy_dictionary,
                self.train_key_patches,
            )
            visual_cf, summaries_cf, _ = self._encode_semantics(features_cf)
            cf_bag, _ = self._spatial(visual_cf, summaries_cf, n)

        orth = gram_orthogonality(attention) if training else torch.zeros((), device=features.device)
        return bag, inst, orth, cf_bag, tau, attention
