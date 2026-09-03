from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from data.manifest_metadata import write_manifest_metadata


def split_manifest(input_csv: str, output_csv: str, seed: int = 2022):
    df = pd.read_csv(input_csv)
    if "patient_id" not in df.columns:
        raise ValueError("patient_id is required for patient-level TCGA splitting")
    patients = df["patient_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(round(0.60 * n))
    n_val = int(round(0.15 * n))
    train = set(patients[:n_train])
    val = set(patients[n_train:n_train + n_val])
    test = set(patients[n_train + n_val:])
    df["split"] = df["patient_id"].map(lambda x: "train" if x in train else ("val" if x in val else "test"))
    df.to_csv(output_csv, index=False)
    project = str(df["source_project"].iloc[0]) if "source_project" in df.columns and len(df) else None
    write_manifest_metadata(output_csv, source_project=project)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("output_csv")
    p.add_argument("--seed", type=int, default=2022)
    a = p.parse_args()
    split_manifest(a.input_csv, a.output_csv, a.seed)
