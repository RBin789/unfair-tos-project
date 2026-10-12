#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unfair_tos.data_ops import convert_parquet_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert parquet files to CSV recursively")
    parser.add_argument("--root", default=".", help="Directory to scan for .parquet files")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    converted = convert_parquet_tree(root)
    if not converted:
        print(f"No .parquet files found under {root}")
        return

    for path in converted:
        print(f"Converted: {path}")
    print(f"Finished converting {len(converted)} file(s).")


if __name__ == "__main__":
    main()
