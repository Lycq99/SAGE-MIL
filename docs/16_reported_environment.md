# Experimental environment

| Component | Version |
|---|---|
| Python | 3.9.23 |
| PyTorch | 2.5.1 |
| CUDA | 12.1 |
| GPU | 2 × NVIDIA GeForce RTX 4090 |
| NVIDIA driver | 535.161.08 |

Install the CUDA-enabled PyTorch build separately before installing the remaining packages in `requirements.txt`.

## Seeds

- split seed: 2022
- model seeds: 2021, 2022, 2023, 2024

`train/train.py` calls `pl.seed_everything(seed)` and runs Lightning with `deterministic=True`.
