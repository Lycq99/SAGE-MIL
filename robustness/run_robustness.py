from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score, f1_score

from audit.replacement_protocol import load_model
from sage_mil.data import FeatureBagDataset
from robustness.feature_perturbations import (
    background_mixing,
    channel_affine,
    dictionary_noise,
    gaussian_noise,
)


@contextmanager
def temporary_dictionary(model, value: torch.Tensor):
    original = model.proxy_dictionary.detach().clone()
    model.proxy_dictionary.copy_(value)
    try:
        yield
    finally:
        model.proxy_dictionary.copy_(original)


def _fused_probability(model, x, coords, bag_weight: float, temperature: float):
    bag, inst, *_ = model(x, coords, training=False)
    if model.num_classes <= 2:
        pb = torch.sigmoid(bag / temperature)
        pi = torch.sigmoid(inst / temperature)
    else:
        pb = torch.softmax(bag / temperature, dim=-1)
        pi = torch.softmax(inst / temperature, dim=-1)
    return bag_weight * pb + (1.0 - bag_weight) * pi


def _clean_attention_score(model, x, coords):
    """Clean morphology/texture attention used to define background patches."""
    _, _, _, _, _, attention = model(x, coords, training=False)
    return attention[:, : min(2, attention.shape[1])].mean(dim=1)


def _metrics(labels, probabilities, threshold: float, num_classes: int):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities)
    if num_classes <= 2:
        p = p.reshape(-1)
        pred = (p >= threshold).astype(int)
        return {
            "auc": float(roc_auc_score(y, p)),
            "f1": float(f1_score(y, pred, zero_division=0)),
        }
    pred = p.argmax(axis=1)
    return {
        "auc": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def _predict_condition(model, ds, perturbation, severity, seed, device, bag_weight, temperature):
    labels = []
    probabilities = []
    generator = torch.Generator(device=device if str(device).startswith("cuda") else "cpu")
    generator.manual_seed(int(seed))

    dictionary_override = None
    if perturbation == "dictionary_noise" and severity > 0:
        dictionary_override = dictionary_noise(model.proxy_dictionary, severity, generator)

    manager = temporary_dictionary(model, dictionary_override) if dictionary_override is not None else _nullcontext()
    with manager, torch.no_grad():
        for i in range(len(ds)):
            x, coords, y = ds[i]
            x = x.unsqueeze(0).to(device)
            coords = coords.unsqueeze(0).to(device)

            if severity > 0:
                if perturbation == "channel_affine":
                    x = channel_affine(x, severity, generator)
                elif perturbation == "gaussian_noise":
                    x = gaussian_noise(x, severity, generator)
                elif perturbation == "background_mix":
                    clean_score = _clean_attention_score(model, x, coords)
                    x = background_mixing(
                        x,
                        model.proxy_dictionary,
                        clean_score,
                        severity,
                        generator=generator,
                        low_fraction=0.5,
                    )
                elif perturbation == "dictionary_noise":
                    pass
                else:
                    raise ValueError(perturbation)

            p = _fused_probability(model, x, coords, bag_weight, temperature)
            labels.append(int(y.item()))
            if model.num_classes <= 2:
                probabilities.append(float(p.reshape(-1)[0].item()))
            else:
                probabilities.append(p.reshape(-1).detach().cpu().numpy())

    return labels, probabilities


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--severities", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model, ckpt = load_model(cfg, args.checkpoint, args.device)
    manifest = pd.read_csv(cfg["data"]["manifest"])
    frame = manifest[manifest["split"] == args.split].reset_index(drop=True)
    ds = FeatureBagDataset(frame, cfg["data"].get("feature_root"), cfg["data"].get("max_patches", -1))

    threshold = float(ckpt.get("best_val_threshold", 0.5))
    bag_weight = float(cfg["model"].get("eval_bag_weight", 0.5))
    temperature = float(cfg["model"].get("eval_temperature", 1.0))

    perturbations = ["channel_affine", "dictionary_noise", "gaussian_noise", "background_mix"]
    display = {
        "channel_affine": "Affine",
        "dictionary_noise": "Dictionary noise",
        "gaussian_noise": "Gaussian noise",
        "background_mix": "Background mix",
    }

    rows = []
    for perturbation in perturbations:
        for severity in args.severities:
            labels, probs = _predict_condition(
                model,
                ds,
                perturbation,
                float(severity),
                args.seed,
                args.device,
                bag_weight,
                temperature,
            )
            m = _metrics(labels, probs, threshold, model.num_classes)
            rows.append(
                {
                    "perturbation": display[perturbation],
                    "severity": float(severity),
                    "auc": m["auc"],
                    "f1": m["f1"],
                    "threshold": threshold if model.num_classes <= 2 else np.nan,
                    "n_slides": len(labels),
                }
            )

    out = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    summary_rows = []
    integrate = getattr(np, "trapezoid", np.trapz)
    for name, group in out.groupby("perturbation"):
        group = group.sort_values("severity")
        x = group["severity"].to_numpy(dtype=float)
        span = float(x.max() - x.min()) if len(x) else 0.0
        auc_area = float(integrate(group["auc"].to_numpy(dtype=float), x) / span) if span > 0 else np.nan
        f1_area = float(integrate(group["f1"].to_numpy(dtype=float), x) / span) if span > 0 else np.nan
        summary_rows.append(
            {
                "perturbation": name,
                "auc_severity_area": auc_area,
                "f1_severity_area": f1_area,
                "max_severity": float(x.max()) if len(x) else np.nan,
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output_path.with_name(output_path.stem + "_summary.csv"), index=False
    )


if __name__ == "__main__":
    main()
