from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--x", default="budget")
    p.add_argument("--y", default="terminal_margin_drop")
    p.add_argument("--method", default="method")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for method, group in df.groupby(args.method):
        group = group.sort_values(args.x)
        ax.plot(group[args.x], group[args.y], marker="o", label=method)
    ax.axvline(200, linestyle="--", linewidth=1)
    ax.set_xlabel("Selected patches K")
    ax.set_ylabel(args.y.replace("_", " ").title())
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
