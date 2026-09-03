from __future__ import annotations

import argparse
from pathlib import Path
import torch
import yaml
from transformers import CLIPModel, CLIPProcessor
from semantic_anchors.build_anchors import average_concept_embeddings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="vinid/plip")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.prompts).read_text(encoding="utf-8"))["concepts"]
    model = CLIPModel.from_pretrained(args.model).eval()
    proc = CLIPProcessor.from_pretrained(args.model)
    groups = []
    with torch.no_grad():
        for name in ["morphology", "texture", "microenvironment"]:
            batch = proc(text=cfg[name], return_tensors="pt", padding=True)
            z = model.get_text_features(**batch)
            groups.append(z)
    anchors = average_concept_embeddings(groups)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"anchors": anchors, "concepts": ["Morphology", "Texture", "Microenvironment"]}, args.output)


if __name__ == "__main__":
    main()
