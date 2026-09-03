# Continuous OT feature replacement

Candidate patches are ranked by the mean Morphology/Texture semantic attention. Training uses `K_train=5` selected patches. Their features are matched to the fixed normal-tissue reference dictionary with entropy-regularized Sinkhorn OT (`epsilon=0.05`, 30 iterations).

The OT barycenter is converted to a norm-matched replacement target. For replacement intensity `tau`:

```text
x_i(tau) = (1-tau) x_i + tau p_i*
```

for selected patches, while all other features remain unchanged. The complete classifier is then re-run on the modified bag.

The feature-replacement loss uses the true-class margin change with `gamma_margin=0.25`. Its weight is warmed linearly to `lambda_verify=0.10` over 20 epochs. The retrospective analysis uses a larger fixed selection budget (`K=200`) and does not update model parameters.
