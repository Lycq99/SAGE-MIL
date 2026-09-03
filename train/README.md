# Training

The main entry point is `train.py`.

```bash
PYTHONPATH=src python train/train.py \
  --config configs/experiments/camelyon16_conch.yaml \
  --seed 2022
```

`train_camelyon16.py` is a convenience wrapper for the CAMELYON16-CONCH configuration.
