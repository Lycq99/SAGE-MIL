from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from feature_extraction.wsi_preprocess import iter_tissue_patches


def encode_slide(model, preprocess, wsi_path, batch_size=128, target_magnification=20.0, patch_size=256, tissue_threshold=0.10):
    features = []
    coords = []
    batch = []
    batch_coords = []

    def flush():
        if not batch:
            return
        x = torch.stack([preprocess(im) for im in batch]).cuda(non_blocking=True)
        with torch.no_grad():
            z = model.encode_image(x, proj_contrast=True, normalize=True)
        features.append(z.cpu())
        coords.extend(batch_coords)
        batch.clear()
        batch_coords.clear()

    for rec in iter_tissue_patches(
        wsi_path,
        target_magnification=target_magnification,
        patch_size=patch_size,
        min_tissue_fraction=tissue_threshold,
    ):
        batch.append(rec.image)
        batch_coords.append((rec.x, rec.y))
        if len(batch) >= batch_size:
            flush()
    flush()

    if not features:
        raise RuntimeError(f"no tissue patches retained for {wsi_path}")
    return torch.cat(features, dim=0), torch.tensor(coords, dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--conch-weight", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--temp-dir")
    p.add_argument("--wsi-column", default="wsi_path")
    p.add_argument("--slide-id-column", default="slide_id")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--target-magnification", type=float, default=20.0)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--tissue-threshold", type=float, default=0.10)
    args = p.parse_args()

    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as e:
        raise SystemExit("Install CONCH from its official repository before running this script.") from e

    frame = pd.read_csv(args.manifest, sep=None, engine="python")
    if args.wsi_column not in frame or args.slide_id_column not in frame:
        raise ValueError(f"manifest must contain {args.wsi_column!r} and {args.slide_id_column!r}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.temp_dir:
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)

    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=args.conch_weight)
    model = model.eval().cuda()

    rows = []
    for _, row in frame.iterrows():
        slide_id = str(row[args.slide_id_column])
        features, coords = encode_slide(
            model,
            preprocess,
            row[args.wsi_column],
            batch_size=args.batch_size,
            target_magnification=args.target_magnification,
            patch_size=args.patch_size,
            tissue_threshold=args.tissue_threshold,
        )
        output = out_dir / f"{slide_id}.pt"
        torch.save({"features": features, "coords": coords}, output)
        rows.append({"slide_id": slide_id, "feature_path": str(output), "num_patches": int(features.shape[0])})

    pd.DataFrame(rows).to_csv(out_dir / "feature_index.csv", index=False)


if __name__ == "__main__":
    main()
