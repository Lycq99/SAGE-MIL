# Feature extraction

The WSI preprocessing code follows the 20× protocol used in the study. A 256×256 model input is mapped to the corresponding level-0 read size from the slide objective magnification. Slides scanned at 40× therefore use a 512×512 level-0 read before bicubic resizing to 256×256. If objective-power metadata are unavailable, 40× is used as the fallback value.

Tissue masking is computed at OpenSlide level 2 when available, otherwise at the closest lower-resolution level. Otsu thresholding is applied to the HSV saturation channel. Patches are sampled without overlap in raster order, incomplete boundary regions are discarded, and patches with less than 10% tissue are skipped.

Patch extraction:

```bash
python feature_extraction/extract_wsi_patches.py \
  --wsi /path/to/slide.svs \
  --output-dir /path/to/patches
```

CONCH patch features:

```bash
python feature_extraction/extract_conch.py \
  --patch-dir /path/to/patches \
  --weight /path/to/conch_checkpoint.bin \
  --output /path/to/features.pt
```

The CONCH extractor uses normalized embeddings and a default batch size of 128. `extract_conch_tcga.py` performs the same patching and encoding steps directly from a manifest containing `slide_id` and `wsi_path` columns. PLIP extraction is provided by `extract_plip.py`.
