#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from unfair_tos.eval_ops import extract_predicted_labels
from unfair_tos.jsonl_ops import load_test_examples
from unfair_tos.metrics import evaluate, safe_div


def _extract_text_from_response_payload(payload: dict[str, Any]) -> str:
	candidates = payload.get("candidates", [])
	if not isinstance(candidates, list) or not candidates:
		return ""

	first = candidates[0]
	if not isinstance(first, dict):
		return ""

	content = first.get("content", {})
	if not isinstance(content, dict):
		return ""

	parts = content.get("parts", [])
	if not isinstance(parts, list):
		return ""

	chunks: list[str] = []
	for part in parts:
		if isinstance(part, dict):
			piece = part.get("text", "")
			if isinstance(piece, str):
				chunks.append(piece)

	return "".join(chunks).strip()


def _coerce_label_list(value: Any) -> set[int]:
	if not isinstance(value, list):
		return set()
	labels: set[int] = set()
	for item in value:
		try:
			labels.add(int(item))
		except (TypeError, ValueError):
			continue
	return labels


def _extract_prediction_labels(record: dict[str, Any]) -> set[int]:
	pred = record.get("pred")
	if isinstance(pred, list):
		return _coerce_label_list(pred)

	raw_prediction = record.get("raw_prediction")
	if isinstance(raw_prediction, str):
		return extract_predicted_labels(raw_prediction)

	prediction = record.get("prediction")
	if isinstance(prediction, str):
		return extract_predicted_labels(prediction)

	response = record.get("response")
	if isinstance(response, dict):
		text = _extract_text_from_response_payload(response)
		if text:
			return extract_predicted_labels(text)

	if "candidates" in record:
		text = _extract_text_from_response_payload(record)
		if text:
			return extract_predicted_labels(text)

	return set()


def load_prediction_labels(path: Path) -> tuple[list[set[int]], int]:
	preds: list[set[int]] = []
	invalid_prediction_count = 0

	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue

			record = json.loads(line)
			if not isinstance(record, dict):
				preds.append(set())
				invalid_prediction_count += 1
				continue

			labels = _extract_prediction_labels(record)
			if not labels:
				invalid_prediction_count += 1
			preds.append(labels)

	return preds, invalid_prediction_count


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Evaluate Gemini predictions against test JSONL and write metrics.jsonl"
	)
	parser.add_argument(
		"--test-file",
		default="jsonl-version/test-gemini.jsonl",
		help="Path to the Gemini test JSONL with gold labels",
	)
	parser.add_argument(
		"--predictions-file",
		default="results/finetuned-eval/gemini-flash-2.5/results_prediction_rearranged.jsonl",
		help="Path to model predictions JSONL",
	)
	parser.add_argument(
		"--output",
		default=None,
		help="Optional output path for metrics.jsonl (defaults to predictions file directory)",
	)
	parser.add_argument(
		"--max-samples",
		type=int,
		default=None,
		help="Optionally limit to the first N examples",
	)
	args = parser.parse_args()

	test_path = Path(args.test_file)
	predictions_path = Path(args.predictions_file)

	if not test_path.exists():
		raise SystemExit(f"Test file not found: {test_path}")
	if not predictions_path.exists():
		raise SystemExit(f"Predictions file not found: {predictions_path}")

	examples = load_test_examples(test_path, max_samples=args.max_samples)
	if not examples:
		raise SystemExit("No valid examples found in the test file.")

	preds, invalid_prediction_count = load_prediction_labels(predictions_path)

	golds = [example.gold_labels for example in examples]
	if args.max_samples is not None:
		preds = preds[: args.max_samples]

	if len(golds) != len(preds):
		raise SystemExit(
			"Mismatch between golds and predictions: "
			f"{len(golds)} gold labels vs {len(preds)} predictions. "
			"Ensure the files are aligned or use --max-samples."
		)

	metrics = evaluate(golds=golds, preds=preds)
	metrics["invalid_prediction_count"] = invalid_prediction_count
	metrics["invalid_prediction_rate"] = safe_div(invalid_prediction_count, len(golds))
	metrics["per_label"] = {str(label): value for label, value in metrics["per_label"].items()}

	output_path = Path(args.output) if args.output else predictions_path.parent / "metrics.jsonl"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")

	print(f"Saved metrics: {output_path}")


if __name__ == "__main__":
	main()
