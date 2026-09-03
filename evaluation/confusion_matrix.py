from __future__ import annotations

import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    df = pd.read_csv(args.input)
    y = df["label"].astype(int).to_numpy()
    if "probability" in df.columns:
        pred = (df["probability"].to_numpy(float) >= args.threshold).astype(int)
    else:
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        pred = df[prob_cols].to_numpy(float).argmax(1)
    print(confusion_matrix(y, pred))


if __name__ == "__main__":
    main()
