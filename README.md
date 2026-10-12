# Unfair ToS Analyzer

This Project contians the code for my thesis which is made to tests LLM-based classification using zero-shot/few-shot prompting and fine-tuned models to detect significant imbalances between consumer rights and obligations.

## Overview

This project enables testing of unfair legal provisions in digital contracts by classifying TOS clauses against 8 standardized categories defined by EU consumer protection law:

- **[0-7]**: Unfair category labels (e.g., unilateral termination, arbitration clauses)
- **[8]**: Fair/balanced clause

Supported LLM providers: **OpenAI** and **Google Gemini**.

---

## Legal Categories

| Label | Category                  | Description                                                                 |
|-------|---------------------------|-----------------------------------------------------------------------------|
| 0     | Limitation of Liability   | Excluding liability for non-performance or injury                          |
| 1     | Unilateral Termination     | Provider can terminate service at will                                      |
| 2     | Unilateral Change         | Provider can change terms/prices without consent                            |
| 3     | Content Removal           | Provider can delete/restrict user content                                   |
| 4     | Contract by Using         | "Browse-wrap" acceptance (mere use = agreement)                             |
| 5     | Choice of Law             | Foreign law designation depriving local consumer protections                |
| 6     | Jurisdiction              | Forcing disputes in inconvenient forum                                      |
| 7     | Arbitration               | Mandatory arbitration preventing court access                               |
| 8     | Fair                      | Balanced, legally sound clause                                              |

---

## Quick Start

### Installation

```bash
# Basic dependencies
pip install openai pandas pyarrow

# For Gemini support
pip install google-genai
```

### Usage Overview

1. **Prepare your dataset**: Convert Parquet → CSV → JSONL
2. **Evaluate with prompt engineering** or **fine-tuned models**
3. **Analyze metrics**: Accuracy, precision, recall, F1 scores per category

---

## Project Structure

```
unfair-tos-project/
├── src/unfair_tos/         # Core logic modules
│   ├── __init__.py
│   ├── data_ops.py         # Dataset handling (CSV/Parquet)
│   ├── eval_ops.py         # Evaluation utilities
│   ├── gemini_batch_ops.py # Google Gemini batch API support
│   ├── jsonl_ops.py        # JSONL file operations
│   ├── labels.py           # Label parsing and validation
│   ├── llm_providers.py    # LLM provider implementations
│   ├── metrics.py          # Evaluation metrics calculation
│   └── prompt_variants.py  # Prompt template variants
├── scripts/                # Standalone evaluation scripts
│   ├── convert_parquet_to_csv.py
│   ├── convert_csv_to_jsonl.py
│   ├── undersample_dataset.py
│   ├── evaluate_finetuned_model.py
│   └── evaluate_prompt_engineering.py
├── prompts/                 # Prompt templates (zero-shot, few-shot)
├── configs/                 # Configuration files
├── results/                 # Evaluation outputs
├── csv-version/             # CSV datasets
└── jsonl-version/           # JSONL datasets for fine-tuning
```

---

## Scripts Reference

Check the `how-to-use-scripts.md` file for using the scripts

---

## Input Format

The input files for this project come from this hugging face project: https://huggingface.co/datasets/coastalcph/lex_glue/tree/main/unfair_tos  

Which are based on the orginal data from this paper: https://link.springer.com/article/10.1007/s10506-019-09243-2

---

## Output Format

Evaluation outputs are stored in JSONL format:

- **predictions.jsonl**: Individual predictions with labels and rationales
- **metrics.jsonl**: Per-run evaluation metrics
- **run_manifest.jsonl**: Metadata about the evaluation run

### Metrics Report Structure

| Field                  | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `num_examples`         | Total number of examples evaluated                           |
| `exact_match_accuracy` | Percentage of exactly correct predictions                    |
| `macro_f1`             | Macro-averaged F1 score across all categories                |
| `micro_f1`             | Micro-averaged F1 score                                      |
| `per_label`            | Detailed metrics for each category (0-8)                     |
| `unfair_binary`        | Binary classification metrics (unfair vs. fair)              |

---


## Metrics Explanation

### Category-Level Evaluation

For each category (0-8), the system computes:
- **Precision**: Of all predicted labels, what fraction were correct?
- **Recall**: Of all true labels, what fraction did we predict correctly?
- **F1 Score**: Harmonic mean of precision and recall
- **Support**: Number of examples containing this category

### Binary Unfair/Fair Classification

Aggregates all categories 0-7 as "unfair" and label 8 as "fair":
- `accuracy`: Overall correct unfair/fair classification
- `precision`, `recall`, `f1`: Standard binary classification metrics
- `tp`, `tn`, `fp`, `fn`: Confusion matrix counts

---

## Few-Shot vs Zero-Shot

The project supports two prompting strategies:

### Zero-Shot (`prompts/zero-shot-classifier.txt`)

Direct classification without examples.

### Few-Shot (`prompts/few-shot-classifier.txt`)

Direct classification with examples.


---

## Result Format

### JSONL Prediction Record

```json
{
  "text": "<original TOS clause>",
  "label": "[8]",
  "legal_rationale": "The clause preserves mandatory statutory rights."
}
```

### Metrics Summary (JSON)

```json
{
  "num_examples": 10000,
  "exact_match_accuracy": 0.7845,
  "macro_f1": 0.7234,
  "micro_f1": 0.8123,
  "per_label": {
    "0": {"precision": 0.89, "recall": 0.85, "f1": 0.87},
    ...
    "8": {"precision": 0.92, "recall": 0.94, "f1": 0.93}
  }
}
```

---

## API Key Configuration

All evaluation scripts require an API key via `--api-key`:

```bash
python scripts/evaluate_finetuned_model.py --api-key sk-your-api-key-...
```

Supported providers: `openai` and `gemini`

---

## License

This project is provided as-is for research purposes.

---

**Author**: Dexter  
**Last Updated**: 2026



