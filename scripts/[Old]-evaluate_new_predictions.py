import json
import argparse
from pathlib import Path
import sys

# Add src to Python path so we can import unfair_tos modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from unfair_tos.metrics import evaluate

def evaluate_predictions(jsonl_file: str):
    golds = []
    preds = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            gold = data.get("gold", [])
            
            # The raw_prediction contains the new format JSON
            raw_pred_str = data.get("raw_prediction", "{}")
            pred_labels = []
            try:
                raw_pred = json.loads(raw_pred_str)
                original_labels = raw_pred.get("label", [])
                is_fair = raw_pred.get("is_fair", True)
                
                # Mapping logic to compare new 2-step against old 1-step gold dataset:
                # Old system: 0-7 means UNFAIR of that category. 8 means FAIR (any category).
                #
                # A clause is considered FAIR if `is_fair` is true AND/OR the LLM assigned no unfair categories ([]).
                # If the LLM returns `[]` but leaves `is_fair: false`, it failed to assign a topic, 
                # meaning no specific unfairness was triggered, thus it reverts to fair [8].
                if is_fair or not original_labels:
                    pred_labels = [8]
                else:
                    pred_labels = original_labels
                    
            except json.JSONDecodeError:
                # Fallback to the 'pred' field or [8] if parsing fails
                pred_labels = data.get("pred", [])
                if not pred_labels:
                    pred_labels = [8]

            golds.append(set(gold))
            preds.append(set(pred_labels))

    metrics_result = evaluate(golds, preds)
    
    print(f"Reviewed {metrics_result['num_examples']} examples.")
    print(f"Exact Match Accuracy: {metrics_result['exact_match_accuracy']:.4f}")
    print(f"Micro F1: {metrics_result['micro_f1']:.4f}")
    print(f"Macro F1: {metrics_result['macro_f1']:.4f}")
    
    print("\n--- Unfair Binary Metrics ---")
    for k, v in metrics_result["unfair_binary"].items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n--- Per Label Metrics ---")
    for label, mets in metrics_result["per_label"].items():
        print(f"Label {label}: F1: {mets['f1']:.4f} | Precision: {mets['precision']:.4f} | Recall: {mets['recall']:.4f} | Support: {mets['support']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate new prediction format.")
    parser.add_argument("predictions_file", type=str, help="Path to predictions.jsonl")
    args = parser.parse_args()
    
    evaluate_predictions(args.predictions_file)
