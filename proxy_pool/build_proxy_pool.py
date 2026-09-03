from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans


def load_features(manifest: str, max_features: int = -1, seed: int = 2022):
    df = pd.read_csv(manifest)
    if "split" in df.columns and not (df["split"].astype(str) == "train").all():
        raise ValueError("proxy manifest must contain training-side rows only")
    if "sample_type" in df.columns:
        allowed = df["sample_type"].astype(str).str.lower().isin({"solid tissue normal", "normal"})
        if not allowed.all():
            raise ValueError("proxy manifest contains non-normal sample types")
    arrays = []
    for p in df["feature_path"].astype(str):
        obj = torch.load(Path(p), map_location="cpu")
        if isinstance(obj, dict):
            obj = obj.get("features", obj.get("embeddings", obj.get("tensor")))
        if not torch.is_tensor(obj):
            raise TypeError(f"feature file does not contain a tensor: {p}")
        arrays.append(obj.float().reshape(-1, obj.shape[-1]).numpy())
    x = np.concatenate(arrays, axis=0)
    if max_features > 0 and len(x) > max_features:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(len(x), size=max_features, replace=False)]
    return x, len(df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-prototypes", type=int, default=100)
    p.add_argument("--max-features", type=int, default=-1)
    p.add_argument("--seed", type=int, default=2022)
    args = p.parse_args()

    x, n_slides = load_features(args.manifest, args.max_features, args.seed)
    km = MiniBatchKMeans(
        n_clusters=args.num_prototypes,
        random_state=args.seed,
        batch_size=min(4096, max(args.num_prototypes * 4, 256)),
        n_init=10,
    )
    km.fit(x)
    obj = {
        "prototypes": torch.from_numpy(km.cluster_centers_).float(),
        "metadata": {
            "manifest": args.manifest,
            "num_prototypes": int(args.num_prototypes),
            "num_slides": int(n_slides),
            "num_features": int(len(x)),
            "seed": int(args.seed),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, args.output)
    print({"shape": tuple(obj["prototypes"].shape), **obj["metadata"]})


if __name__ == "__main__":
    main()
