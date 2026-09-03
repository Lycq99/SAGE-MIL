# SAGE-MIL component flow

## Patch representation

```text
WSI
 -> 20× tissue fields
 -> non-overlapping 256×256 model inputs
 -> frozen CONCH or PLIP encoder
 -> N×512 patch-feature bag
```

## Semantic guidance

```text
Morphology / Texture / Microenvironment prompts
 -> frozen text encoder
 -> fixed anchors A
 -> anchor projection E_A

patch features X
 -> semantic projection E_s
 -> value projection E_v
```

For each attention head, the projected anchors and semantic features are L2-normalized. Raw semantic responses use a learned scale `s=min(100, exp(eta))`. Patch-specific normal-tissue reference estimates are obtained with soft allocation over the fixed reference dictionary using temperature `T_e`. The correction mask `[1,1,0]` is applied to Morphology and Texture responses only.

## Sparse semantic routing

```text
adjusted semantic responses
 -> top ceil(rho*N) positions per concept/head
 -> softmax over patches
 -> average across heads
 -> three semantic summaries
```

## Spatial MIL

```text
semantic summaries + visual patch tokens
 -> padded raster-order grid
 -> Nyström Transformer
 -> SCConv
 -> Nyström Transformer
 -> bag prediction + auxiliary instance prediction
```

## Training objective

```text
L_train = L_bag + L_inst
          + lambda_orth * L_orth
          + lambda_verify(epoch) * L_verify
```

`L_orth` uses the squared Frobenius norm of the three-channel attention Gram matrix relative to `I_3`.

## Continuous feature replacement

```text
mean Morphology/Texture semantic attention
 -> top K_train=5 patches
 -> Sinkhorn OT to the normal-tissue reference dictionary
 -> norm-matched replacement targets
 -> x_i(tau)=(1-tau)x_i+tau p_i*
 -> shared classifier re-forward
 -> true-class margin constraint
```

The retrospective protocol uses `K=200` for the main mechanism analysis and is separate from the training replacement budget.
