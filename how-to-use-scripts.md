# Unfair ToS Pipeline

This project is now organized as one script per task.

JSON/JSONL convention used across this workspace:
- `.jsonl` for datasets, predictions, metrics, and run metadata.
- `.json` only for editor/workspace settings.

## Structure

- `scripts/convert_parquet_to_csv.py` - convert parquet files to CSV recursively
- `scripts/undersample_dataset.py` - targeted undersampling for class balance
- `scripts/convert_csv_to_jsonl.py` - convert dataset CSV into chat fine-tuning JSONL
- `scripts/evaluate_finetuned_model.py` - evaluate a fine-tuned model on JSONL test data
- `scripts/evaluate_prompt_engineering.py` - evaluate a single prompt on a base model
- `src/unfair_tos/` - shared core logic

## Install

```bash
pip install openai pandas pyarrow
```

For Gemini batch mode support:

```bash
pip install google-genai
```

## API key

Evaluation scripts require `--api-key` explicitly.
They now also accept `--provider` with one of: `openai`, `gemini`, `claude`.

## Commands

### 1) Parquet -> CSV

```bash
python scripts/convert_parquet_to_csv.py --root .
```

### 2) Targeted undersampling

```bash
python scripts/undersample_dataset.py \
  --input csv-version/train-00000-of-00001.csv \
  --output csv-version/train_undersampled.csv \
  --fair-per-unfair 4 \
  --seed 42 \
  --shuffle-output
```

### 3) CSV -> JSONL

```bash
python scripts/convert_csv_to_jsonl.py \
  --input csv-version/train_undersampled.csv \
  --output jsonl-version/train.jsonl \
  --system-content "You are a legal clause classifier." \
  --user-template "{text}" \
  --assistant-template "{label}"
```

### 4) Evaluate fine-tuned model

OpenAI

```bash
python scripts/evaluate_finetuned_model.py \
  --provider openai \
  --model ft:gpt-4.1-mini:YOUR_MODEL_ID \
  --test-file jsonl-version/test.jsonl \
  --output-dir results/finetuned-eval \
  --api-key YOUR_OPENAI_API_KEY
```

Gemini:

```bash
python scripts/evaluate_finetuned_model.py \
  --provider gemini \
  --model gemini-2.0-flash \
  --test-file jsonl-version/test.jsonl \
  --output-dir results/finetuned-eval \
  --api-key YOUR_GEMINI_API_KEY
```

### 5) Evaluate prompt engineering

OpenAI
```bash
python scripts/evaluate_prompt_engineering.py \
  --provider openai \
  --model gpt-4.1-mini \
  --test-file jsonl-version/test.jsonl \
  --prompt-file prompts/zero-shot-classifier.txt \
  --output-root results/prompt-runs \
  --run-name full-test \
  --temperature 0.0 \
  --api-key YOUR_OPENAI_API_KEY
```

Gemini:

```bash
python scripts/evaluate_prompt_engineering.py \
  --provider gemini \
  --model gemini-2.0-flash \
  --test-file jsonl-version/test.jsonl \
  --prompt-file prompts/zero-shot-classifier.txt \
  --output-root results/prompt-runs \
  --run-name gemini-full-test \
  --temperature 0.0 \
  --api-key YOUR_GEMINI_API_KEY
```

Gemini batch mode (submit, wait, and evaluate when done):

```bash
python scripts/evaluate_prompt_engineering.py \
  --provider gemini \
  --model gemini-2.0-flash \
  --test-file jsonl-version/test.jsonl \
  --prompt-file prompts/zero-shot-classifier.txt \
  --output-root results/prompt-runs \
  --run-name gemini-batch-full-test \
  --temperature 0.0 \
  --api-key YOUR_GEMINI_API_KEY \
  --gemini-batch-mode
```

Gemini batch mode (submit now, fetch later):

```bash
python scripts/evaluate_prompt_engineering.py \
  --provider gemini \
  --model gemini-2.0-flash \
  --test-file jsonl-version/test.jsonl \
  --prompt-file prompts/zero-shot-classifier.txt \
  --output-root results/prompt-runs \
  --run-name gemini-batch-submit-only \
  --temperature 0.0 \
  --api-key YOUR_GEMINI_API_KEY \
  --gemini-batch-mode \
  --gemini-batch-no-wait
```

Notes:
- `--gemini-batch-mode` requires `--provider gemini`.
- `--gemini-batch-no-wait` and `--gemini-batch-job-name` are mutually exclusive.
- `--gemini-batch-job-name` is for polling/fetching an existing job instead of creating a new one.

Later, fetch results from an existing job:

```bash
python scripts/evaluate_prompt_engineering.py \
  --provider gemini \
  --model gemini-2.0-flash \
  --test-file jsonl-version/test.jsonl \
  --prompt-file prompts/zero-shot-classifier.txt \
  --output-root results/prompt-runs \
  --run-name gemini-batch-fetch \
  --temperature 0.0 \
  --api-key YOUR_GEMINI_API_KEY \
  --gemini-batch-mode \
  --gemini-batch-job-name batches/YOUR_BATCH_ID
```

## Outputs

- `results/finetuned-eval/metrics.jsonl`
- `results/finetuned-eval/predictions.jsonl`
- `results/prompt-runs/<run-name>/metrics.jsonl`
- `results/prompt-runs/<run-name>/predictions.jsonl`
- `results/prompt-runs/<run-name>/run_manifest.jsonl`
