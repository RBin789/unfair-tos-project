#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


TABLE_COLUMNS = [
    "Category",
    "Precision",
    "Recall",
    "F1 (Macro)",
    "F1 (Micro)",
    "Support",
    "TP",
    "FP",
    "FN",
    "TN",
    "Accuracy",
    "invalid_prediction_count",
    "invalid_prediction_rate",
]

DEFAULT_CLASS_LABELS = [str(i) for i in range(9)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError("Each JSONL line must be a JSON object")
        records.append(obj)
    if not records:
        raise ValueError("Input JSONL contains no JSON objects")
    return records


def _blank_row() -> list[Any]:
    return ["" for _ in TABLE_COLUMNS]


def _format_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return value
        return f"{value:.5f}"
    return value


def _sorted_label_keys(labels: list[str]) -> list[str]:
    numeric_labels = [k for k in labels if k.isdigit()]
    non_numeric_labels = [k for k in labels if not k.isdigit()]
    return sorted(numeric_labels, key=int) + sorted(non_numeric_labels)


def _table_row_from_metrics(
    category: str,
    metrics: dict[str, Any] | None,
    *,
    invalid_prediction_count: Any = "",
    invalid_prediction_rate: Any = "",
) -> list[Any]:
    m = metrics if isinstance(metrics, dict) else {}
    return [
        category,
        _format_value(m.get("precision", "")),
        _format_value(m.get("recall", "")),
        _format_value(m.get("f1", "")),
        "",
        _format_value(m.get("support", "")),
        _format_value(m.get("tp", "")),
        _format_value(m.get("fp", "")),
        _format_value(m.get("fn", "")),
        _format_value(m.get("tn", "")),
        _format_value(m.get("accuracy", "")),
        _format_value(invalid_prediction_count),
        _format_value(invalid_prediction_rate),
    ]


def _all_row(rec: dict[str, Any]) -> list[Any]:
    return [
        "All",
        _format_value(rec.get("micro_precision", "")),
        _format_value(rec.get("micro_recall", "")),
        _format_value(rec.get("macro_f1", "")),
        _format_value(rec.get("micro_f1", "")),
        _format_value(rec.get("num_examples", "")),
        "",
        "",
        "",
        "",
        _format_value(rec.get("exact_match_accuracy", "")),
        _format_value(rec.get("invalid_prediction_count", "")),
        _format_value(rec.get("invalid_prediction_rate", "")),
    ]


def _record_to_template_rows(
    rec: dict[str, Any],
    run_index: int,
    include_run_index_row: bool,
) -> list[list[Any]]:
    rows: list[list[Any]] = []

    if include_run_index_row:
        run_row = _blank_row()
        run_row[0] = "run_index"
        run_row[1] = run_index
        rows.append(run_row)

    rows.append(TABLE_COLUMNS.copy())
    rows.append(_all_row(rec))

    per_label = rec.get("per_label")
    per_label_metrics = per_label if isinstance(per_label, dict) else {}

    seen_labels: set[str] = set()
    for label in DEFAULT_CLASS_LABELS:
        rows.append(_table_row_from_metrics(label, per_label_metrics.get(label)))
        seen_labels.add(label)

    unfair_binary = rec.get("unfair_binary")
    rows.append(_table_row_from_metrics("unfair_binary", unfair_binary if isinstance(unfair_binary, dict) else None))

    # Append unexpected per-label categories after template rows so data is not dropped.
    extra_label_keys = []
    if isinstance(per_label, dict):
        for label in _sorted_label_keys(list(per_label.keys())):
            if label in seen_labels:
                continue
            extra_label_keys.append(label)
    for label in extra_label_keys:
        rows.append(_table_row_from_metrics(label, per_label_metrics.get(label)))

    return rows


def _write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    all_rows: list[list[Any]] = []

    include_run_index_row = len(records) > 1
    for i, rec in enumerate(records, start=1):
        rows = _record_to_template_rows(
            rec=rec,
            run_index=i,
            include_run_index_row=include_run_index_row,
        )
        all_rows.extend(rows)
        if i < len(records):
            all_rows.append(_blank_row())

    if not all_rows:
        raise ValueError("No data available to write")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert metrics.jsonl into a category-wise CSV"
    )
    parser.add_argument("--input", required=True, help="Path to metrics.jsonl")
    parser.add_argument(
        "--output",
        help="Path to output .csv (default: metrics.csv in the input file's directory)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name("metrics.csv")

    records = _read_jsonl(input_path)
    _write_csv(records, output_path)

    print(f"Wrote CSV in template format to {output_path}")


if __name__ == "__main__":
    main()
