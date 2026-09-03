# Semantic anchors

The model uses three fixed pathology concepts: Morphology, Texture, and Microenvironment. Their text prompts are defined in `configs/semantic_prompts.yaml` and encoded with the text encoder corresponding to the frozen patch-feature model.

The stored anchor tensor has shape `[3,d]`. During MIL training the anchor tensor remains fixed; the model applies the learnable anchor projection `E_A` before head-wise cosine response calculation.

CONCH and PLIP anchor-generation scripts are provided under `semantic_anchors/`.
