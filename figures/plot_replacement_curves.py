from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--value", choices=["margin_drop", "counterfactual_probability"], default="margin_drop")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if "branch" not in df.columns:
        df["branch"] = "bag"
    summary = df.groupby(["branch", "method", "tau"], as_index=False)[args.value].mean()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for (branch, method), group in summary.groupby(["branch", "method"]):
        group = group.sort_values("tau")
        linestyle = "-" if branch == "bag" else "--"
        ax.plot(
            group["tau"],
            group[args.value],
            linestyle=linestyle,
            marker="o" if branch == "bag" else None,
            label=f"{method}, {branch}",
        )
    ax.set_xlabel("Replacement intensity")
    if args.value == "margin_drop":
        ax.set_ylabel("True-class margin drop")
    else:
        ax.set_ylabel("True-class probability")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
