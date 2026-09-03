from __future__ import annotations

import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    args = p.parse_args()
    frames = [pd.read_csv(x) for x in args.inputs]
    df = pd.concat(frames, ignore_index=True)
    numeric = df.select_dtypes("number")
    out = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=1)})
    print(out.to_string())


if __name__ == "__main__":
    main()
