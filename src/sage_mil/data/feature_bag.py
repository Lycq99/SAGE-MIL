from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class FeatureBagDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_root: Optional[str] = None, max_patches: int = -1):
        self.frame = frame.reset_index(drop=True)
        self.feature_root = Path(feature_root) if feature_root else None
        self.max_patches = int(max_patches)

    def __len__(self):
        return len(self.frame)

    def _path(self, value: str) -> Path:
        p = Path(value)
        if not p.is_absolute() and self.feature_root is not None:
            p = self.feature_root / p
        return p

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        obj = torch.load(self._path(str(row.feature_path)), map_location="cpu")
        coords = None
        if isinstance(obj, dict):
            features = obj.get("features", obj.get("embeddings", obj.get("tensor")))
            coords = obj.get("coords")
        else:
            features = obj
        if not torch.is_tensor(features):
            raise TypeError("feature file must contain a tensor or a dict with features")
        features = features.float()
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        if features.ndim != 2:
            raise ValueError(f"expected [N,D] features, got {tuple(features.shape)}")
        if self.max_patches > 0 and features.shape[0] > self.max_patches:
            features = features[: self.max_patches]
            if torch.is_tensor(coords):
                coords = coords[: self.max_patches]
        if coords is None:
            coords = torch.zeros(features.shape[0], 2, dtype=torch.long)
        else:
            coords = torch.as_tensor(coords).long()
        return features, coords, torch.tensor(int(row.label), dtype=torch.long)


def _collate_single(batch):
    # WSI bags have variable numbers of patches; the study uses batch size one.
    if len(batch) != 1:
        raise ValueError("FeatureDataModule expects batch_size=1 for variable-length bags")
    x, c, y = batch[0]
    return x.unsqueeze(0), c.unsqueeze(0), y.view(1)


class FeatureDataModule(pl.LightningDataModule):
    def __init__(self, manifest, max_patches=-1, num_workers=8, feature_root=None):
        super().__init__()
        self.manifest = str(manifest)
        self.max_patches = int(max_patches)
        self.num_workers = int(num_workers)
        self.feature_root = feature_root

    def setup(self, stage=None):
        df = pd.read_csv(self.manifest)
        required = {"feature_path", "label", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        self.train_ds = FeatureBagDataset(df[df.split == "train"], self.feature_root, self.max_patches)
        self.val_ds = FeatureBagDataset(df[df.split == "val"], self.feature_root, self.max_patches)
        self.test_ds = FeatureBagDataset(df[df.split == "test"], self.feature_root, self.max_patches)

    def _loader(self, ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size=1,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate_single,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds)

    def test_dataloader(self):
        return self._loader(self.test_ds)
