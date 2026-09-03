# Training protocol

The main training entry point is `train/train.py`.

```text
L_train = L_bag + L_inst
          + lambda_orth * L_orth
          + lambda_verify(epoch) * L_verify
```

The orthogonality term is the squared Frobenius norm of the normalized three-channel attention Gram matrix relative to the identity matrix. The replacement-loss weight is warmed linearly over the first 20 epochs.

Training adds zero-mean Gaussian feature noise with standard deviation `0.005`. The auxiliary instance branch uses a top-k schedule from 10 to 1 during training and `k=6` at inference. The four model seeds are 2021, 2022, 2023, and 2024; the split seed is fixed at 2022.

The checkpoint with minimum validation loss is used for held-out evaluation.
