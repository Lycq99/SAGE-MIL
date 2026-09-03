from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--y", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input)[[args.x, args.y]].dropna()
    rho, _ = spearmanr(df[args.x], df[args.y])
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    ax.scatter(df[args.x], df[args.y], alpha=0.65)
    ax.set_xlabel(args.x.replace("_", " ").title())
    ax.set_ylabel(args.y.replace("_", " ").title())
    ax.text(0.04, 0.95, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
