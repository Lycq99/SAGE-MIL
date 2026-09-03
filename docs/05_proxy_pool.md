# Normal-tissue reference dictionary

The reference dictionary is built from training-side normal-tissue patch features only. TCGA `Solid Tissue Normal` slides are excluded from subtype-classification cohorts and used only as reference sources.

Normal-patch features are pooled, optionally subsampled, and clustered in the pretrained feature space. Cluster centroids define the fixed dictionary `D_proxy`. The reported configuration uses 100 prototypes. The dictionary remains fixed during MIL training and retrospective feature-replacement analysis.

```bash
python proxy_pool/build_proxy_pool.py \
  --manifest data/proxy/train_normal_manifest.csv \
  --output assets/reference_pool.pt \
  --num-prototypes 100 \
  --seed 2022
```
