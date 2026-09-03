from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import torch
import yaml

from sage_mil.data import FeatureBagDataset
from train.train import LitSAGEMIL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    module = LitSAGEMIL.load_from_checkpoint(args.checkpoint, cfg=cfg, map_location="cpu")
    module.eval()

    manifest = pd.read_csv(cfg["data"]["manifest"])
    frame = manifest[manifest["split"] == args.split].reset_index(drop=True)
    ds = FeatureBagDataset(frame, cfg["data"].get("feature_root"), cfg["data"].get("max_patches", -1))
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, coords, y = ds[i]
            x = x.unsqueeze(0); coords = coords.unsqueeze(0)
            bag, inst, *_ = module.model(x, coords, training=False)
            p = module._prob(bag, inst)
            row = {
                "slide_id": str(frame.iloc[i].get("slide_id", i)),
                "label": int(y.item()),
            }
            if module.num_classes <= 2:
                row["probability"] = float(p.reshape(-1)[0])
            else:
                for c, value in enumerate(p.reshape(-1).tolist()):
                    row[f"prob_{c}"] = float(value)
            rows.append(row)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
