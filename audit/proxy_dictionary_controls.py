from __future__ import annotations

import argparse
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--collapsed-output", required=True)
    p.add_argument("--random-output", required=True)
    p.add_argument("--seed", type=int, default=2022)
    args = p.parse_args()
    obj = torch.load(args.input, map_location="cpu")
    proto = obj["prototypes"] if isinstance(obj, dict) else obj
    collapsed = proto.mean(0, keepdim=True).expand_as(proto).clone()
    gen = torch.Generator().manual_seed(args.seed)
    rnd = torch.randn(proto.shape, generator=gen)
    rnd = rnd / rnd.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    norms = proto.norm(dim=-1, keepdim=True)
    rnd = rnd * norms
    Path(args.collapsed_output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prototypes": collapsed}, args.collapsed_output)
    torch.save({"prototypes": rnd}, args.random_output)


if __name__ == "__main__":
    main()
