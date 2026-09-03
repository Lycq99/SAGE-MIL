from __future__ import annotations

import pandas as pd


def load_camelyon16_manifest(path: str) -> pd.DataFrame:
    """Load a CAMELYON16 manifest with train/val/test split labels."""
    df = pd.read_csv(path)
    required = {"slide_id", "feature_path", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    return df
