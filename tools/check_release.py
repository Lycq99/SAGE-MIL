from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    re.compile(r"/home/[^\s]+"),
    re.compile(r"[A-Za-z]:\\\\Users\\\\"),
]


def main():
    problems = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".txt", ".toml", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in PATTERNS:
            if pattern.search(text):
                problems.append(f"local path in {path.relative_to(ROOT)}")
    if problems:
        print("\n".join(problems))
        return 1
    print("release check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
