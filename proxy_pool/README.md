# Normal-tissue reference dictionary

`build_proxy_pool.py` builds the fixed reference dictionary used by SAGE-MIL from training-side normal-tissue patch features.

Required manifest column:

```text
feature_path
```

Optional columns:

```text
split,sample_type,patient_id,slide_id
```

If `split` is present, all rows must belong to the training split. For TCGA, `sample_type` can be used to restrict the source to `Solid Tissue Normal` samples.

## CAMELYON16

The CAMELYON16 reference dictionary used in the detailed analyses contains `M=100` prototypes. The CONCH tensor has shape `100 × 512`.

```bash
python proxy_pool/build_proxy_pool.py \
  --manifest data/proxy/train_normal_manifest.csv \
  --output assets/camelyon16_proxy_pool.pt \
  --num-prototypes 100 \
  --seed 2022
```

The saved file contains the prototype tensor and construction metadata. Keep the prototype count, source manifest, sampling settings, clustering settings, seed, and encoder consistent with the experiment being reproduced.
