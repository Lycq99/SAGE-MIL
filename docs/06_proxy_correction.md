# Normal-tissue-reference adjustment

For each attention head, patch and reference features are projected with `E_s` and L2-normalized. Patch-specific reference allocation is

```text
pi_i,m = softmax((k_i^T d_m) / T_e)
```

and the corresponding reference estimate is the weighted prototype average. Raw and reference semantic responses are scaled by

```text
s = min(100, exp(eta)).
```

The adjusted response is

```text
M_adj = M_raw - lambda_env * g * B_patch
```

with correction mask `g=[1,1,0]`. Morphology and Texture are adjusted; Microenvironment is retained.
