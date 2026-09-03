from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

import cv2
import numpy as np
from PIL import Image

try:
    import openslide
except ImportError:  # pragma: no cover
    openslide = None


@dataclass(frozen=True)
class PatchRecord:
    x: int
    y: int
    image: Image.Image


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


def keep_patch(rgb: np.ndarray, min_tissue_fraction: float = 0.10) -> bool:
    return float(tissue_mask(rgb).mean()) >= float(min_tissue_fraction)


def objective_power(slide, fallback: float = 40.0) -> float:
    keys = [
        getattr(openslide, "PROPERTY_NAME_OBJECTIVE_POWER", "openslide.objective-power") if openslide else "openslide.objective-power",
        "aperio.AppMag",
    ]
    for key in keys:
        value = slide.properties.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return float(fallback)


def mask_level(slide) -> int:
    if slide.level_count <= 1:
        return 0
    return min(2, slide.level_count - 1)


def build_slide_tissue_mask(slide) -> tuple[np.ndarray, float]:
    level = mask_level(slide)
    width, height = slide.level_dimensions[level]
    rgb = np.asarray(slide.read_region((0, 0), level, (width, height)).convert("RGB"))
    return tissue_mask(rgb), float(slide.level_downsamples[level])


def _mask_fraction(mask: np.ndarray, downsample: float, x: int, y: int, read_size: int) -> float:
    x0 = max(0, int(np.floor(x / downsample)))
    y0 = max(0, int(np.floor(y / downsample)))
    x1 = min(mask.shape[1], int(np.ceil((x + read_size) / downsample)))
    y1 = min(mask.shape[0], int(np.ceil((y + read_size) / downsample)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def iter_tissue_patches(
    slide_path: Union[str, Path],
    target_magnification: float = 20.0,
    patch_size: int = 256,
    min_tissue_fraction: float = 0.10,
    fallback_objective: float = 40.0,
) -> Iterator[PatchRecord]:
    if openslide is None:
        raise ImportError("openslide-python is required for WSI patch extraction")

    slide = openslide.OpenSlide(str(slide_path))
    try:
        objective = objective_power(slide, fallback=fallback_objective)
        scale = objective / float(target_magnification)
        read_size = max(1, int(round(patch_size * scale)))
        width, height = slide.dimensions
        mask, downsample = build_slide_tissue_mask(slide)

        for y in range(0, height - read_size + 1, read_size):
            for x in range(0, width - read_size + 1, read_size):
                if _mask_fraction(mask, downsample, x, y, read_size) < min_tissue_fraction:
                    continue
                patch = slide.read_region((x, y), 0, (read_size, read_size)).convert("RGB")
                if read_size != patch_size:
                    patch = patch.resize((patch_size, patch_size), Image.Resampling.BICUBIC)
                yield PatchRecord(x=x, y=y, image=patch)
    finally:
        slide.close()
