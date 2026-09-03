#!/usr/bin/env bash
set -euo pipefail
python evaluation/classification_metrics.py "$@"
