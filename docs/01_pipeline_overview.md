# Pipeline overview

SAGE-MIL separates routine classification from the feature-replacement path used during training and retrospective analysis.

For a feature bag `X`, patch and normal-tissue reference features are projected into a shared semantic space. Fixed Morphology, Texture, and Microenvironment anchors are projected by `E_A`; patch and reference features use `E_s`, while patch values use `E_v`. Head-wise cosine responses are scaled by `s=min(100, exp(eta))`. Reference allocation uses temperature `T_e`, and the correction mask `[1,1,0]` leaves the Microenvironment response unchanged.

The largest adjusted responses are retained within each semantic concept and attention head. The resulting attention maps are normalized over patches, averaged across heads, and used to form three semantic summaries. These summaries are combined with raster-order visual tokens by the spatial MIL backbone.

During training, the mean Morphology/Texture semantic attention ranks candidate patches for continuous Sinkhorn-based feature replacement. The shared classifier is re-evaluated on the modified bag and the true-class margin change contributes to the training loss. Routine inference does not require this replacement path.
