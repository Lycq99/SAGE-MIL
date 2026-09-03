# Sparse semantic routing and spatial MIL

For each semantic concept and attention head, the largest

```text
k_attn = max(1, ceil(rho*N))
```

adjusted responses are retained before softmax normalization over patches. Head-wise maps are averaged to obtain the three semantic attention channels, and the corresponding semantic summaries are formed from the value projection `E_v(X)`.

The visual tokens are arranged on the padded raster-order square grid used by the model. Local processing uses SCConv with a spatial reconstruction unit followed by a channel reconstruction unit. Long-range interactions are modeled with Nyström attention. The Nyström dependency is required; the implementation does not silently substitute standard full self-attention when it is unavailable.
