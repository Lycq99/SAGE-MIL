from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-dir", required=True)
    p.add_argument("--weight", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as e:
        raise SystemExit("Install CONCH from its official repository before running this script.") from e

    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=args.weight)
    model = model.eval().cuda()
    paths = sorted([x for x in Path(args.patch_dir).iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    feats = []
    for i in range(0, len(paths), args.batch_size):
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + args.batch_size]]).cuda()
        with torch.no_grad():
            z = model.encode_image(x, proj_contrast=True, normalize=True)
        feats.append(z.cpu())
    features = torch.cat(feats, dim=0)
    coords_path = Path(args.patch_dir) / "coordinates.csv"
    payload = {"features": features}
    if coords_path.exists():
        coords = pd.read_csv(coords_path)[["x", "y"]].to_numpy()
        payload["coords"] = torch.as_tensor(coords, dtype=torch.long)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)


if __name__ == "__main__":
    main()
