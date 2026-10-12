#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unfair_tos.data_ops import read_dataset_csv
from unfair_tos.jsonl_ops import csv_to_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CSV to JSONL file for OpenAI or Gemini")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["openai", "gemini"], default="openai", help="Target fine-tuning format (openai or gemini)")
    parser.add_argument("--system-content", default="You are a legal clause classifier.")
    parser.add_argument("--system-file", default=None, help="Optional path to system prompt file")
    parser.add_argument("--user-template", default="{text}")
    parser.add_argument("--assistant-template", default="{label}")
    args = parser.parse_args()

    if "{text}" not in args.user_template:
        raise SystemExit("--user-template must include {text}")
    if "{label}" not in args.assistant_template:
        raise SystemExit("--assistant-template must include {label}")

    system_content = args.system_content
    if args.system_file:
        system_content = Path(args.system_file).read_text(encoding="utf-8").strip()

    rows = read_dataset_csv(Path(args.input))
    converted = csv_to_jsonl(
        rows=rows,
        output_path=Path(args.output),
        system_content=system_content,
        user_template=args.user_template,
        assistant_template=args.assistant_template,
        dataset_format=args.format,
    )
    print(f"Converted {converted} row(s) from {args.input} to {args.output} ({args.format} format)")


if __name__ == "__main__":
    main()
