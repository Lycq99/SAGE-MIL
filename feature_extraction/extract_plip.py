from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="vinid/plip")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    model = CLIPModel.from_pretrained(args.model).eval().cuda()
    proc = CLIPProcessor.from_pretrained(args.model)
    paths = sorted([x for x in Path(args.patch_dir).iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    feats = []
    for i in range(0, len(paths), args.batch_size):
        images = [Image.open(x).convert("RGB") for x in paths[i:i + args.batch_size]]
        batch = proc(images=images, return_tensors="pt").to("cuda")
        with torch.no_grad():
            z = model.get_image_features(**batch)
            z = torch.nn.functional.normalize(z, dim=-1)
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
