# Evaluation

`predict.py` exports fused bag/instance probabilities for a fixed checkpoint. Binary outputs use a `probability` column; multiclass outputs use `prob_0`, `prob_1`, ... columns.

Classification:

```bash
python evaluation/classification_metrics.py --input predictions.csv --task binary --threshold 0.5
```

Calibration:

```bash
python evaluation/calibration_metrics.py predictions.csv --task binary
python evaluation/calibration_metrics.py predictions.csv --task multiclass
```

Calibration uses 15 equal-width confidence bins. The multiclass Brier score is computed against one-hot targets and ECE uses maximum predicted probability and correctness.

`statistics.py` provides the slide-level paired bootstrap confidence interval (10,000 resamples) and paired sign-randomization test (20,000 draws) used by the replacement analyses.
