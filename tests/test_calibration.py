import numpy as np
from evaluation.calibration_metrics import binary_calibration, multiclass_calibration


def test_binary_calibration_uses_15_bins_by_default():
    out = binary_calibration(np.array([0, 1]), np.array([0.1, 0.9]))
    assert out["bins"] == 15
    assert out["brier"] >= 0.0
    assert out["ece"] >= 0.0


def test_multiclass_calibration():
    y = np.array([0, 1, 2])
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
    out = multiclass_calibration(y, p)
    assert out["bins"] == 15
    assert out["brier"] >= 0.0
    assert out["ece"] >= 0.0
