from __future__ import annotations

import argparse
import csv
from pathlib import Path

from feature_extraction.wsi_preprocess import iter_tissue_patches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wsi", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-magnification", type=float, default=20.0)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--tissue-threshold", type=float, default=0.10)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, rec in enumerate(iter_tissue_patches(
        args.wsi,
        target_magnification=args.target_magnification,
        patch_size=args.patch_size,
        min_tissue_fraction=args.tissue_threshold,
    )):
        name = f"{idx:07d}_x{rec.x}_y{rec.y}.png"
        rec.image.save(out / name)
        rows.append((idx, rec.x, rec.y, name))

    with (out / "coordinates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y", "file"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
