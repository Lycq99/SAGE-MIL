from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Union
import pandas as pd


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest_metadata(csv_path: Union[str, Path], source_project: Optional[str] = None) -> Path:
    path = Path(csv_path)
    df = pd.read_csv(path)
    meta = {
        'manifest': path.name,
        'sha256': sha256_file(path),
        'rows': int(len(df)),
        'class_counts': {str(k): int(v) for k, v in df['label'].value_counts().sort_index().items()} if 'label' in df else {},
        'split_counts': {str(k): int(v) for k, v in df['split'].value_counts().items()} if 'split' in df else {},
    }
    if source_project:
        meta['source_project'] = source_project
    out = path.with_suffix(path.suffix + '.meta.json')
    out.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return out
