# Robustness analyses

The robustness runner evaluates a fixed checkpoint over the severity grid `0.0, 0.1, 0.2, 0.3, 0.5`.

```bash
PYTHONPATH=src:. python robustness/run_robustness.py \
  --config configs/experiments/camelyon16_conch.yaml \
  --checkpoint /path/to/best.ckpt \
  --output outputs/robustness.csv \
  --device cuda
```

Four perturbations are used.

**Gaussian noise.** For each WSI,

\[
X' = X + \alpha\,\sigma(X)Z, \qquad Z\sim\mathcal N(0,1),
\]

where `sigma(X)` is the standard deviation over all `N x 512` feature elements of that WSI.

**Dictionary noise.** For the current prototype bank,

\[
D' = D + \alpha\,\sigma(D)Z_D, \qquad Z_D\sim\mathcal N(0,1),
\]

where `sigma(D)` is computed over all elements of the `100 x 512` dictionary. Patch features are not directly perturbed in this condition.

**Background proxy mixing.** Clean morphology/texture attention is computed first. The lowest-attention 50% of patches are selected, and each selected patch is assigned a randomly sampled prototype `d_j` from the normal-tissue dictionary:

\[
x_i' = (1-\alpha)x_i + \alpha d_j.
\]

Patches outside the selected half are unchanged.

**Channel-affine shift.** For every WSI and feature channel, independent directions are drawn as

\[
u_c,v_c \sim \operatorname{clip}(\mathcal N(0,1),-2,2),
\]

and

\[
x'_{i,c}=x_{i,c}(1+0.25\alpha u_c)+0.25\alpha v_c\sigma_c(X),
\]

where `sigma_c(X)` is the standard deviation of channel `c` across patches in that WSI.

The runner reports AUC and F1 at each severity and also writes normalized areas under the metric-versus-severity curves.
