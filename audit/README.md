# Feature-replacement analysis

The retrospective analysis uses a fixed trained checkpoint and the test split defined by the experiment manifest.

```bash
PYTHONPATH=src:. python audit/replacement_protocol.py \
  --config configs/experiments/camelyon16_conch.yaml \
  --checkpoint /path/to/best.ckpt \
  --budget 200 \
  --output outputs/audit/replacement_curves.csv \
  --device cuda
```

The output contains bag and fused branches for OT-Key, OT-Random, OT-Low, Mean-Key, and Zero-Key across the replacement-intensity grid. For binary tasks, true-class probability is `p` for positive slides and `1-p` for negative slides.

Summary statistics are generated with:

```bash
PYTHONPATH=src:. python audit/summarize_replacement.py \
  outputs/audit/replacement_curves.csv \
  --output-dir outputs/audit/summary
```

`slide_summary.csv` reports AURD, AUPC, AOPC, and AOPCR. AUPC is the integral of true-class probability over replacement intensity, AOPC is `p(0) - AUPC`, and AOPCR uses OT-Random as the reference denominator. Paired OT-Key versus OT-Random comparisons use 10,000 bootstrap resamples and 20,000 sign-randomization draws.
