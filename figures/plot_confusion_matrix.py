from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", choices=["binary", "multiclass"], required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--class-names", nargs="+")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    y = df["label"].to_numpy(int)
    if args.task == "binary":
        pred = (df["probability"].to_numpy(float) >= args.threshold).astype(int)
    else:
        cols = sorted([c for c in df.columns if c.startswith("prob_")], key=lambda x: int(x.split("_")[1]))
        pred = df[cols].to_numpy(float).argmax(axis=1)

    labels = np.arange(max(y.max(), pred.max()) + 1)
    cm = confusion_matrix(y, pred, labels=labels)
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    pct = 100.0 * cm / row_sum

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(pct, vmin=0, vmax=100)
    names = args.class_names or [str(x) for x in labels]
    ax.set_xticks(labels, names)
    ax.set_yticks(labels, names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}\n{pct[i,j]:.1f}%", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Row-normalized percentage")
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
