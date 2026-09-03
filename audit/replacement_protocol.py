from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

from sage_mil.model import SemanticTransMIL
from sage_mil.data import FeatureBagDataset
from sage_mil.losses import true_class_margin


def load_model(cfg, checkpoint, device="cpu"):
    m, d, task = cfg["model"], cfg["data"], cfg["task"]
    model = SemanticTransMIL(
        input_dim=m["input_dim"],
        model_dim=m["model_dim"],
        num_classes=task["num_classes"],
        anchors_path=d["anchors_path"],
        proxy_path=d["proxy_path"],
        proxy_num_prototypes=d.get("proxy_num_prototypes"),
        train_key_patches=m["train_key_patches"],
        lambda_env=m["lambda_env"],
        correction_mask=m["correction_mask"],
        topk_ratio=m["topk_ratio"],
        semantic_heads=m.get("semantic_heads", 1),
        proxy_temperature=m.get("proxy_temperature", 0.10),
        semantic_logit_scale_init=m.get("semantic_logit_scale_init", 1.0),
        sinkhorn_epsilon=m["sinkhorn_epsilon"],
        sinkhorn_iterations=m["sinkhorn_iterations"],
        target_norm_matching=m["target_norm_matching"],
        norm_match_clip=tuple(m["norm_match_clip"]),
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
    # Lightning checkpoints can contain non-model entries under state_dict prefixes;
    # missing keys are reported because analysis must use a compatible checkpoint.
    if missing:
        raise RuntimeError(f"checkpoint is missing model keys: {missing}")
    return model.to(device).eval(), ckpt


def _score(attention: torch.Tensor) -> torch.Tensor:
    return attention[:, : min(2, attention.shape[1])].mean(dim=1)


def _indices(score: torch.Tensor, method: str, k: int, rng: np.random.Generator) -> torch.Tensor:
    n = score.shape[-1]
    k = min(k, n)
    if method in {"OT-Key", "Mean-Key", "Zero-Key"}:
        return torch.topk(score, k, dim=1).indices
    if method == "OT-Low":
        return torch.topk(-score, k, dim=1).indices
    if method == "OT-Random":
        idx = rng.choice(n, size=k, replace=False)
        return torch.as_tensor(idx, device=score.device, dtype=torch.long).view(1, -1)
    raise ValueError(method)


def _positive_probability(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes <= 2:
        return torch.sigmoid(logits.reshape(-1))
    return torch.softmax(logits, dim=-1)


def _true_class_probability(probability: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    target = target.long().reshape(-1)
    if num_classes <= 2:
        p_pos = probability.reshape(-1)
        return torch.where(target > 0, p_pos, 1.0 - p_pos)
    return probability.gather(1, target[:, None]).squeeze(1)


def _fused_probability(
    bag: torch.Tensor,
    inst: torch.Tensor,
    num_classes: int,
    bag_weight: float,
    temperature: float,
) -> torch.Tensor:
    if num_classes <= 2:
        pb = torch.sigmoid(bag.reshape(-1) / temperature)
        pi = torch.sigmoid(inst.reshape(-1) / temperature)
    else:
        pb = torch.softmax(bag / temperature, dim=-1)
        pi = torch.softmax(inst / temperature, dim=-1)
    return float(bag_weight) * pb + (1.0 - float(bag_weight)) * pi


def _probability_margin(probability: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Margin induced by a fused probability distribution."""
    eps = torch.finfo(probability.dtype).eps
    target = target.long().reshape(-1)
    if num_classes <= 2:
        p = probability.reshape(-1).clamp(eps, 1.0 - eps)
        logit = torch.log(p) - torch.log1p(-p)
        return torch.where(target > 0, logit, -logit)
    p = probability.clamp_min(eps)
    logp = torch.log(p)
    true = logp.gather(1, target[:, None]).squeeze(1)
    mask = torch.nn.functional.one_hot(target, num_classes=num_classes).bool()
    other = logp.masked_fill(mask, float("-inf")).max(dim=1).values
    return true - other


def _instance_logit(model: SemanticTransMIL, inst_raw: torch.Tensor) -> torch.Tensor:
    return model._topk_instance_logits(inst_raw, model.instance_topk(0, 1, training=False))


def _branch_values(model, bag, inst, y, bag_weight, temperature):
    bag_prob_all = _positive_probability(bag / temperature, model.num_classes)
    bag_true = _true_class_probability(bag_prob_all, y, model.num_classes)
    bag_margin = true_class_margin(bag / temperature, y, model.num_classes)

    fused_prob_all = _fused_probability(bag, inst, model.num_classes, bag_weight, temperature)
    fused_true = _true_class_probability(fused_prob_all, y, model.num_classes)
    fused_margin = _probability_margin(fused_prob_all, y, model.num_classes)
    return {
        "bag": (bag_margin, bag_true),
        "fused": (fused_margin, fused_true),
    }


def run_one(model, x, y, budget, rng, taus, bag_weight=0.5, temperature=1.0):
    with torch.no_grad():
        visual, summaries, attention = model._encode_semantics(x)
        bag, inst_raw = model._spatial(visual, summaries, x.shape[1])
        inst = _instance_logit(model, inst_raw)
        factual = _branch_values(model, bag, inst, y, bag_weight, temperature)
        score = _score(attention)

        rows = []
        for method in ["OT-Key", "OT-Random", "OT-Low", "Mean-Key", "Zero-Key"]:
            idx = _indices(score, method, budget, rng)
            gather = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            candidates = torch.gather(x.detach(), 1, gather)
            if method.startswith("OT-"):
                bary = model.ot.barycenters(candidates, model.proxy_dictionary)
                targets = model.ot.effective_targets(x, candidates, bary)
            elif method == "Mean-Key":
                targets = x.detach().mean(dim=1, keepdim=True).expand_as(candidates)
            else:
                targets = torch.zeros_like(candidates)

            for tau in taus:
                out = x.clone()
                blended = (1.0 - float(tau)) * candidates + float(tau) * targets
                out.scatter_(1, gather, blended)
                visual_cf, summaries_cf, _ = model._encode_semantics(out)
                bag_cf, inst_raw_cf = model._spatial(visual_cf, summaries_cf, x.shape[1])
                inst_cf = _instance_logit(model, inst_raw_cf)
                counterfactual = _branch_values(model, bag_cf, inst_cf, y, bag_weight, temperature)

                for branch in ("bag", "fused"):
                    m0, p0 = factual[branch]
                    m1, p1 = counterfactual[branch]
                    rows.append(
                        {
                            "branch": branch,
                            "method": method,
                            "tau": float(tau),
                            "factual_margin": float(m0.item()),
                            "counterfactual_margin": float(m1.item()),
                            "margin_drop": float((m0 - m1).item()),
                            "factual_probability": float(p0.item()),
                            "counterfactual_probability": float(p1.item()),
                            "probability_drop": float((p0 - p1).item()),
                        }
                    )
        return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--tau-steps", type=int, default=11)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model, _ = load_model(cfg, args.checkpoint, args.device)
    manifest = pd.read_csv(cfg["data"]["manifest"])
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    ds = FeatureBagDataset(test, cfg["data"].get("feature_root"), cfg["data"].get("max_patches", -1))
    rng = np.random.default_rng(args.seed)
    taus = np.linspace(0.0, 1.0, args.tau_steps)
    bag_weight = cfg["model"].get("eval_bag_weight", 0.5)
    temperature = cfg["model"].get("eval_temperature", 1.0)

    rows = []
    for i in range(len(ds)):
        x, _, y = ds[i]
        x = x.unsqueeze(0).to(args.device)
        y = y.view(1).to(args.device)
        slide_id = str(test.iloc[i].get("slide_id", i))
        label = int(y.item())
        for row in run_one(model, x, y, args.budget, rng, taus, bag_weight, temperature):
            row["slide_id"] = slide_id
            row["label"] = label
            rows.append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
