from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def evaluate(df: pd.DataFrame, task: str, threshold: float = 0.5, average: str = "macro"):
    y = df["label"].to_numpy().astype(int)
    if task == "binary":
        p = df["probability"].to_numpy(float)
        pred = (p >= threshold).astype(int)
        return {
            "auc": float(roc_auc_score(y, p)),
            "acc": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "threshold": float(threshold),
        }
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    p = df[prob_cols].to_numpy(float)
    pred = p.argmax(1)
    return {
        "auc": float(roc_auc_score(y, p, multi_class="ovr", average=average)),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, average=average, zero_division=0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--task", choices=["binary", "multiclass"], required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--average", default="macro")
    args = p.parse_args()
    result = evaluate(pd.read_csv(args.input), args.task, args.threshold, args.average)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
