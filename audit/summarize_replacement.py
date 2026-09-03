from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.statistics import paired_bootstrap_ci, paired_randomization_p


def trapz(g: pd.DataFrame, column: str) -> float:
    order = np.argsort(g["tau"].to_numpy())
    x = g["tau"].to_numpy(dtype=float)[order]
    y = g[column].to_numpy(dtype=float)[order]
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x))


def summarize_slides(df: pd.DataFrame) -> pd.DataFrame:
    if "branch" not in df.columns:
        df = df.assign(branch="bag")

    rows = []
    for (slide, branch, method), g in df.groupby(["slide_id", "branch", "method"]):
        factual = float(g["factual_probability"].iloc[0])
        aupc = trapz(g, "counterfactual_probability")
        aopc = factual - aupc
        rows.append(
            {
                "slide_id": slide,
                "branch": branch,
                "method": method,
                "aurd": trapz(g, "margin_drop"),
                "aupc": aupc,
                "aopc": aopc,
            }
        )

    out = pd.DataFrame(rows)
    denom = (
        out[out["method"] == "OT-Random"]
        .set_index(["slide_id", "branch"])["aopc"]
        .rename("random_aopc")
    )
    out = out.join(denom, on=["slide_id", "branch"])
    out["aopcr"] = np.where(
        out["random_aopc"].abs() > 1e-12,
        out["aopc"] / out["random_aopc"],
        np.nan,
    )
    return out


def method_summary(slides: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        slides.groupby(["branch", "method"], as_index=False)
        .agg(
            n_slides=("slide_id", "nunique"),
            mean_aurd=("aurd", "mean"),
            mean_aupc=("aupc", "mean"),
            mean_aopc=("aopc", "mean"),
        )
    )
    random = (
        grouped[grouped["method"] == "OT-Random"]
        .set_index("branch")["mean_aopc"]
        .rename("random_mean_aopc")
    )
    grouped = grouped.join(random, on="branch")
    grouped["pooled_aopcr"] = np.where(
        grouped["random_mean_aopc"].abs() > 1e-12,
        grouped["mean_aopc"] / grouped["random_mean_aopc"],
        np.nan,
    )
    return grouped


def paired_output(slides: pd.DataFrame, value: str, branch: str) -> dict | None:
    sub = slides[slides["branch"] == branch]
    pivot = sub.pivot(index="slide_id", columns="method", values=value)
    if not {"OT-Key", "OT-Random"}.issubset(pivot.columns):
        return None
    pair = pivot[["OT-Key", "OT-Random"]].dropna()
    if pair.empty:
        return None
    mean, lo, hi = paired_bootstrap_ci(pair["OT-Key"], pair["OT-Random"], n_boot=10000)
    pval = paired_randomization_p(pair["OT-Key"], pair["OT-Random"], n_perm=20000)
    return {
        "branch": branch,
        "metric": value,
        "mean_difference": mean,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": pval,
        "n_slides": int(len(pair)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slides = summarize_slides(df)
    slides.to_csv(out_dir / "slide_summary.csv", index=False)
    method_summary(slides).to_csv(out_dir / "method_summary.csv", index=False)

    stats = []
    for branch in sorted(slides["branch"].unique()):
        for value in ("aurd", "aopc"):
            row = paired_output(slides, value, branch)
            if row is not None:
                stats.append(row)
    pd.DataFrame(stats).to_csv(out_dir / "ot_key_vs_random.csv", index=False)


if __name__ == "__main__":
    main()
