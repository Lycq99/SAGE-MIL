from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 200, 500])
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for budget in args.budgets:
        subprocess.check_call([
            "python", "audit/replacement_protocol.py", "--config", args.config,
            "--checkpoint", args.checkpoint, "--budget", str(budget),
            "--output", str(out / f"budget_{budget}.csv")
        ])


if __name__ == "__main__":
    main()
