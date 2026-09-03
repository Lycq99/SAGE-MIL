# Feature extraction

WSIs are processed at a 20× field of view. The level-0 read size is derived from objective magnification so that the encoder always receives a 256×256 image. A 40× slide therefore uses a 512×512 level-0 read followed by bicubic resizing; 40× is used as the fallback objective when slide metadata do not provide magnification.

The tissue mask is computed at OpenSlide level 2 when available, otherwise at the closest lower-resolution level. Otsu thresholding is applied to the HSV saturation channel. Patches are sampled without overlap in raster order, incomplete boundary patches are discarded, and a 10% tissue threshold is used.

CONCH uses the frozen ViT-B/16 encoder in evaluation mode with normalized embeddings and a default batch size of 128. PLIP features are also L2-normalized. Feature files contain the patch-feature tensor and level-0 patch coordinates; the current MIL forward path preserves raster order but does not consume physical coordinates.
