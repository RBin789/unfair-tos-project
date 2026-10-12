#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unfair_tos.eval_ops import evaluate_prompt_engineering
from unfair_tos.llm_providers import SUPPORTED_PROVIDERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a single prompt on test JSONL")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--test-file", required=True, help="Path to test JSONL dataset")
    parser.add_argument("--prompt-file", required=True, help="Path to system prompt text file")
    parser.add_argument("--output-root", default="results/prompt-runs", help="Directory where run outputs are saved")
    parser.add_argument("--run-name", default=None, help="Optional run folder name; defaults to UTC timestamp")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on evaluated examples")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--gemini-batch-mode",
        action="store_true",
        help="Use Gemini Batch API instead of per-example online inference (Gemini provider only)",
    )
    parser.add_argument(
        "--gemini-batch-job-name",
        default=None,
        help="Existing Gemini batch job name (e.g., batches/123) to poll/fetch instead of submitting a new one",
    )
    parser.add_argument(
        "--gemini-batch-no-wait",
        action="store_true",
        help="Submit batch job and exit without waiting for completion",
    )
    parser.add_argument(
        "--gemini-batch-poll-seconds",
        type=int,
        default=30,
        help="Polling interval in seconds when waiting on Gemini batch jobs",
    )
    parser.add_argument(
        "--gemini-batch-chunk-size",
        type=int,
        default=None,
        help="Optional chunk size for Gemini batch mode; submits one batch job per chunk and merges outputs",
    )
    args = parser.parse_args()

    if args.gemini_batch_job_name and not args.gemini_batch_mode:
        parser.error("--gemini-batch-job-name requires --gemini-batch-mode")
    if args.gemini_batch_no_wait and not args.gemini_batch_mode:
        parser.error("--gemini-batch-no-wait requires --gemini-batch-mode")
    if args.gemini_batch_no_wait and args.gemini_batch_job_name:
        parser.error("--gemini-batch-no-wait cannot be used with --gemini-batch-job-name")
    if args.gemini_batch_poll_seconds < 1:
        parser.error("--gemini-batch-poll-seconds must be >= 1")
    if args.gemini_batch_chunk_size is not None and args.gemini_batch_chunk_size < 1:
        parser.error("--gemini-batch-chunk-size must be >= 1")
    if args.gemini_batch_chunk_size is not None and not args.gemini_batch_mode:
        parser.error("--gemini-batch-chunk-size requires --gemini-batch-mode")
    if args.gemini_batch_chunk_size is not None and args.gemini_batch_job_name:
        parser.error("--gemini-batch-chunk-size cannot be used with --gemini-batch-job-name")
    if args.gemini_batch_chunk_size is not None and args.gemini_batch_no_wait:
        parser.error("--gemini-batch-chunk-size requires waiting; remove --gemini-batch-no-wait")

    evaluate_prompt_engineering(
        provider_name=args.provider,
        model=args.model,
        test_file=Path(args.test_file),
        prompt_file=Path(args.prompt_file),
        output_root=Path(args.output_root),
        run_name=args.run_name,
        max_samples=args.max_samples,
        api_key=args.api_key,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        temperature=args.temperature,
        gemini_batch_mode=args.gemini_batch_mode,
        gemini_batch_wait=not args.gemini_batch_no_wait,
        gemini_batch_poll_seconds=args.gemini_batch_poll_seconds,
        gemini_batch_job_name=args.gemini_batch_job_name,
        gemini_batch_chunk_size=args.gemini_batch_chunk_size,
    )


if __name__ == "__main__":
    main()