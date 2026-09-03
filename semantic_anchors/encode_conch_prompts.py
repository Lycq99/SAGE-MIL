from __future__ import annotations

import argparse
from pathlib import Path
import torch
import yaml
from semantic_anchors.build_anchors import average_concept_embeddings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--weight", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    try:
        from conch.open_clip_custom import create_model_from_pretrained, tokenize
    except ImportError as e:
        raise SystemExit("Install CONCH from its official repository before running this script.") from e

    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=args.weight)
    model = model.eval()
    cfg = yaml.safe_load(Path(args.prompts).read_text(encoding="utf-8"))["concepts"]
    groups = []
    with torch.no_grad():
        for name in ["morphology", "texture", "microenvironment"]:
            tokens = tokenize(cfg[name])
            groups.append(model.encode_text(tokens, normalize=True))
    anchors = average_concept_embeddings(groups)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"anchors": anchors, "concepts": ["Morphology", "Texture", "Microenvironment"]}, args.output)


if __name__ == "__main__":
    main()
