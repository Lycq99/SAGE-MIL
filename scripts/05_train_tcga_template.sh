#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?config path required}
SEED=${2:-2022}
PYTHONPATH=src python train/train.py --config "$CONFIG" --seed "$SEED"
