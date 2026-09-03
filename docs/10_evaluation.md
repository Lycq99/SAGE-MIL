# Evaluation

Classification metrics are computed from the fused probability

```text
p = 0.5 * p_bag + 0.5 * p_inst.
```

For binary tasks, `p_bag` and `p_inst` are sigmoid probabilities. For RCC, softmax class probabilities are fused element-wise. AUC is one-vs-rest macro-averaged for RCC and F1 is macro-averaged.

Binary operating thresholds are selected on validation by maximizing F1 over 0.10–0.90 in steps of 0.05 and are frozen before test evaluation. RCC uses the class with maximum predicted probability.

Calibration uses Brier score and expected calibration error with 15 equal-width confidence bins. Multiclass Brier score is calculated against one-hot targets, and multiclass ECE uses maximum predicted probability and correctness.
