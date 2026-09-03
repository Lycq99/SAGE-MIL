#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python train/train.py --config configs/experiments/camelyon16_conch.yaml --seed "${1:-2022}"
