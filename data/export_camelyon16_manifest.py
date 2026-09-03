from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from data.manifest_metadata import write_manifest_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("feature_dir")
    p.add_argument("labels_csv")
    p.add_argument("output_csv")
    args = p.parse_args()
    labels = pd.read_csv(args.labels_csv)
    if "slide_id" not in labels.columns or "label" not in labels.columns:
        raise ValueError("labels CSV must contain slide_id and label")
    root = Path(args.feature_dir)
    labels["feature_path"] = labels["slide_id"].map(lambda s: str(root / f"{s}.pt"))
    if "patient_id" not in labels.columns:
        labels["patient_id"] = labels["slide_id"]
    labels["source_project"] = "CAMELYON16"
    labels.to_csv(args.output_csv, index=False)
    write_manifest_metadata(args.output_csv, source_project="CAMELYON16")


if __name__ == "__main__":
    main()
