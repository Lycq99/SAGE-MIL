from __future__ import annotations

import argparse
import pandas as pd


def validate(path: str) -> None:
    df = pd.read_csv(path)
    required = {"slide_id", "feature_path", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    allowed = {"train", "val", "test"}
    bad = sorted(set(df["split"].astype(str)) - allowed)
    if bad:
        raise ValueError(f"unexpected split values: {bad}")
    if "patient_id" in df.columns:
        by_patient = df.groupby("patient_id")["split"].nunique()
        leaking = by_patient[by_patient > 1]
        if len(leaking):
            raise ValueError(f"patient leakage across splits: {len(leaking)} patients")
    print({"rows": len(df), "splits": df["split"].value_counts().to_dict()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    args = p.parse_args()
    validate(args.manifest)
