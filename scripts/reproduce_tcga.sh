#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?config path required}
for SEED in 2021 2022 2023 2024; do
  PYTHONPATH=src python train/train.py --config "$CONFIG" --seed "$SEED"
done
