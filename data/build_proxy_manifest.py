from __future__ import annotations

import argparse
import pandas as pd
from data.manifest_metadata import write_manifest_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("output_csv")
    p.add_argument("--sample-type", default="Solid Tissue Normal")
    args = p.parse_args()
    df = pd.read_csv(args.input_csv)
    if "sample_type" in df.columns:
        df = df[df["sample_type"] == args.sample_type]
    if "split" in df.columns:
        df = df[df["split"] == "train"]
    cols = [c for c in ["patient_id", "slide_id", "feature_path", "split", "sample_type", "source_project"] if c in df.columns]
    out = df[cols].copy()
    out.to_csv(args.output_csv, index=False)
    project = str(out["source_project"].iloc[0]) if "source_project" in out.columns and len(out) else None
    write_manifest_metadata(args.output_csv, source_project=project)


if __name__ == "__main__":
    main()
