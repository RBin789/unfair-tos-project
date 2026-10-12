#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unfair_tos.eval_ops import evaluate_finetuned_model
from unfair_tos.llm_providers import SUPPORTED_PROVIDERS


FINETUNED_PROVIDERS = tuple(provider for provider in SUPPORTED_PROVIDERS if provider != "gemini")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on test JSONL")
    parser.add_argument("--provider", choices=FINETUNED_PROVIDERS, default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-file", default="jsonl-version/test.jsonl")
    parser.add_argument("--output-dir", default="results/finetuned-eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    args = parser.parse_args()

    evaluate_finetuned_model(
        provider_name=args.provider,
        model=args.model,
        test_file=Path(args.test_file),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        temperature=args.temperature,
        api_key=args.api_key,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )


if __name__ == "__main__":
    main()
