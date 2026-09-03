from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)


def binary_calibration(y, p, bins: int = 15):
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    if y.shape[0] != p.shape[0]:
        raise ValueError("labels and probabilities must have the same length")
    brier = float(np.mean((p - y) ** 2))
    pred = (p >= 0.5).astype(int)
    confidence = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == y).astype(float)
    return {"brier": brier, "ece": _ece(confidence, correct, bins), "bins": int(bins)}


def multiclass_calibration(y, p, bins: int = 15):
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[0] != y.shape[0]:
        raise ValueError("multiclass probabilities must have shape [N,C]")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probabilities must lie in [0,1]")

    row_sum = p.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("each probability row must have positive mass")
    p = p / row_sum

    one_hot = np.eye(p.shape[1], dtype=float)[y]
    brier = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
    pred = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = (pred == y).astype(float)
    return {"brier": brier, "ece": _ece(confidence, correct, bins), "bins": int(bins)}


def _probability_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("prob_")]
    if not cols:
        raise ValueError("multiclass input must contain prob_0, prob_1, ... columns")
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--task", choices=["binary", "multiclass"], required=True)
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    y = df["label"].to_numpy(int)
    if args.task == "binary":
        result = binary_calibration(y, df["probability"].to_numpy(float), args.bins)
    else:
        cols = _probability_columns(df)
        result = multiclass_calibration(y, df[cols].to_numpy(float), args.bins)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
