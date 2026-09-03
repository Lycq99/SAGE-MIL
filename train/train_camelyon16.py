"""CAMELYON16-CONCH training wrapper.

Usage:
    python train/train_camelyon16.py [seed]
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "experiments" / "camelyon16_conch.yaml"
    seed = sys.argv[1] if len(sys.argv) > 1 else "2022"

    if not config.exists():
        raise FileNotFoundError(f"CAMELYON16-CONCH config not found: {config}")

    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(root / "train" / "train.py"),
                "--config",
                str(config),
                "--seed",
                seed,
            ],
            cwd=root,
            env=env,
        )
    )
