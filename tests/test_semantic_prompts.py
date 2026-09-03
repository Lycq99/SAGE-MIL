from pathlib import Path
import yaml


def test_three_semantic_concepts_and_one_prompt_each():
    path = Path(__file__).resolve().parents[1] / "configs" / "semantic_prompts.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(cfg["concepts"].keys()) == ["morphology", "texture", "microenvironment"]
    assert all(len(cfg["concepts"][name]) == 1 for name in cfg["concepts"])
