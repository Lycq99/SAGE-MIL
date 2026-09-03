# Analysis workflows

Retrospective analyses use a fixed trained checkpoint and the test split specified by the experiment manifest. The main feature-replacement protocol uses 200 selected patches and compares OT-Key with random, low-response, bag-mean, and zero-vector controls.

The replacement output includes both the bag branch and the fused bag/instance branch. True-class probability is label-aware for binary tasks. The summary script reports AURD, AUPC, AOPC, AOPCR, paired bootstrap confidence intervals, and paired sign-randomization tests.

The robustness analysis uses four feature-space perturbations over the severity grid shown in the sensitivity figure. Gaussian feature noise is scaled by the global feature standard deviation of each WSI. Dictionary noise is scaled by the global standard deviation of the current prototype bank. Background proxy mixing is applied only to the lowest clean morphology/texture-attention 50% of patches, with an independently sampled proxy target for each selected patch. Channel-affine perturbation uses clipped Gaussian scale and shift directions per WSI and feature channel, with the shift term scaled by the within-WSI channel standard deviation. The exact equations are listed in `robustness/README.md`.
