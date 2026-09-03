from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metric", choices=["auc", "f1"], default="auc")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name, group in df.groupby("perturbation"):
        group = group.sort_values("severity")
        ax.plot(group["severity"], group[args.metric], marker="o", label=name)
    ax.set_xlabel("Perturbation severity")
    ax.set_ylabel(args.metric.upper())
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
