#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unfair_tos.data_ops import read_dataset_csv, targeted_undersample, write_dataset_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted undersampling on CSV file")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fair-per-unfair", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-output", action="store_true")
    args = parser.parse_args()

    rows = read_dataset_csv(Path(args.input))
    output_rows, stats = targeted_undersample(
        rows=rows,
        fair_per_unfair=args.fair_per_unfair,
        seed=args.seed,
        shuffle_output=args.shuffle_output,
    )
    write_dataset_csv(Path(args.output), output_rows)

    print(f"Input rows: {stats['input_rows']}")
    print(f"Unfair rows kept: {stats['unfair_rows_kept']}")
    print(f"Fair rows available: {stats['fair_rows_available']}")
    print(f"Fair rows sampled: {stats['fair_rows_sampled']}")
    print(f"Output rows: {stats['output_rows']}")
    print(
        f"Achieved unfair:fair = 1:{stats['actual_fair_per_unfair']:.2f} "
        f"(requested 1:{args.fair_per_unfair})"
    )
    print(f"Wrote undersampled CSV to: {args.output}")


if __name__ == "__main__":
    main()
