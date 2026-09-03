from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def area_under_response_drop(tau, drop):
    order = np.argsort(tau)
    return float(np.trapz(np.asarray(drop)[order], np.asarray(tau)[order]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.input)
    keys = [c for c in ["slide_id", "method"] if c in df.columns]
    rows = []
    for group, g in df.groupby(keys) if keys else [((), df)]:
        row = dict(zip(keys, group if isinstance(group, tuple) else (group,)))
        row["aurd"] = area_under_response_drop(g["tau"], g["margin_drop"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
