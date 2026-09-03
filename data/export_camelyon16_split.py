from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from data.manifest_metadata import write_manifest_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("output_csv")
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--val-fraction", type=float, default=0.25)
    args = p.parse_args()
    df = pd.read_csv(args.input_csv)
    dev = df[df["split"].isin(["development", "dev", "train"])].copy()
    test = df[df["split"].isin(["official_test", "test"])].copy()
    rng = np.random.default_rng(args.seed)
    ids = np.arange(len(dev)); rng.shuffle(ids)
    n_val = int(round(args.val_fraction * len(dev)))
    val_idx = set(ids[:n_val].tolist())
    dev["split"] = ["val" if i in val_idx else "train" for i in range(len(dev))]
    test["split"] = "test"
    out = pd.concat([dev, test], ignore_index=True)
    if "source_project" not in out.columns:
        out["source_project"] = "CAMELYON16"
    out.to_csv(args.output_csv, index=False)
    write_manifest_metadata(args.output_csv, source_project="CAMELYON16")


if __name__ == "__main__":
    main()
