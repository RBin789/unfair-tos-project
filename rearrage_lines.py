import json
from pathlib import Path


def _extract_request_text(obj: dict) -> str:
	return obj["request"]["contents"][0]["parts"][0]["text"]


def _extract_prediction_label(obj: dict) -> str:
	return obj["response"]["candidates"][0]["content"]["parts"][0]["text"]


def rearrange_predictions(
	original_path: Path,
	predictions_path: Path,
	output_path: Path,
) -> None:
	original_texts = []
	with original_path.open("r", encoding="utf-8") as original_file:
		for line in original_file:
			line = line.strip()
			if not line:
				continue
			original_texts.append(_extract_request_text(json.loads(line)))

	prediction_map = {}
	with predictions_path.open("r", encoding="utf-8") as predictions_file:
		for line in predictions_file:
			line = line.strip()
			if not line:
				continue
			data = json.loads(line)
			prediction_map[_extract_request_text(data)] = data

	missing_texts = []
	with output_path.open("w", encoding="utf-8") as output_file:
		for text in original_texts:
			prediction = prediction_map.get(text)
			if prediction is None:
				missing_texts.append(text)
				continue
			output_file.write(json.dumps(prediction, ensure_ascii=True) + "\n")

	if missing_texts:
		missing_preview = " | ".join(missing_texts[:5])
		raise ValueError(
			"Missing predictions for one or more texts. "
			f"Count: {len(missing_texts)}. Preview: {missing_preview}"
		)


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	original_path = base_dir / "jsonl-version" / "test-gemini-batch-format.jsonl"
	predictions_path = (
		base_dir
		/ "results"
		/ "finetuned-eval"
		/ "gemini-flash-2.5"
		/ "results_prediction-unfairtos-gemini-2.5-flashv2.jsonl"
	)
	output_path = (
		base_dir
		/ "results"
		/ "finetuned-eval"
		/ "gemini-flash-2.5"
		/ "results_prediction-unfairtos-gemini-2.5-flashv2.rearranged.jsonl"
	)

	rearrange_predictions(original_path, predictions_path, output_path)
	print(f"Wrote reordered predictions to: {output_path}")


if __name__ == "__main__":
	main()
