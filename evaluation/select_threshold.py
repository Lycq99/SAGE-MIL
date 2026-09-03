from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def select_f1_threshold(y, p, start=0.10, stop=0.90, step=0.05):
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    best_f1 = -1.0
    best_threshold = float(start)
    thresholds = np.arange(start, stop + step / 2.0, step)
    for threshold in thresholds:
        score = float(f1_score(y, p >= threshold, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return {"threshold": best_threshold, "f1": best_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    print(json.dumps(select_f1_threshold(df["label"], df["probability"]), indent=2))


if __name__ == "__main__":
    main()
