# Method-to-code map

| Method component | Implementation |
|---|---|
| WSI preprocessing | `feature_extraction/wsi_preprocess.py` |
| CONCH / PLIP patch features | `feature_extraction/` |
| Morphology / Texture / Microenvironment prompts | `configs/semantic_prompts.yaml` |
| Fixed semantic anchors | `semantic_anchors/` |
| `E_A`, `E_s`, `E_v` semantic/value projections | `src/sage_mil/model.py`, `src/sage_mil/modules/semantic_proxy.py` |
| Head-wise scaled semantic responses | `src/sage_mil/modules/semantic_proxy.py` |
| Reference allocation with `T_e` | `src/sage_mil/modules/semantic_proxy.py` |
| Correction mask `[1,1,0]` | `src/sage_mil/modules/semantic_proxy.py` |
| Sparse semantic routing and head averaging | `src/sage_mil/modules/semantic_proxy.py` |
| Nyström Transformer + SCConv | `src/sage_mil/modules/spatial.py` |
| Bag and auxiliary instance heads | `src/sage_mil/model.py` |
| Focal classification loss | `src/sage_mil/losses.py`, `train/train.py` |
| Gram-matrix orthogonality loss | `src/sage_mil/losses.py` |
| Sinkhorn replacement targets | `src/sage_mil/modules/ot_verification.py` |
| Continuous replacement path | `src/sage_mil/modules/ot_verification.py`, `audit/replacement_protocol.py` |
| True-class margin and replacement loss | `src/sage_mil/losses.py` |
| Fused prediction probability | `train/train.py` |
| Calibration with 15 bins | `evaluation/calibration_metrics.py` |
| OT-Key / Random / Low / Mean / Zero | `audit/replacement_protocol.py` |
| AURD / AUPC / AOPC / AOPCR | `audit/`, `evaluation/` |
| Bootstrap and sign-randomization tests | `evaluation/statistics.py`, `audit/summarize_replacement.py` |
| Quantitative figure utilities | `figures/` |
