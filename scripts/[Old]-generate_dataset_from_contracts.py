#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unfair_tos.llm_providers import create_provider
from unfair_tos.labels import normalize_for_training

SYSTEM_PROMPT = """You are a legal document analyst. Extract clauses from the following contract text verbatim. For each clause, categorize it into one of these labels:
0 = Limitation of Liability
1 = Unilateral Termination
2 = Unilateral Change
3 = Content Removal
4 = Contract by Using
5 = Choice of Law
6 = Jurisdiction
7 = Arbitration

If a clause is perfectly fair, or does not represent any of the unfair categories, categorize it with label [8].

Format your output STRICTLY as a JSON array of objects, with NO markdown formatting, NO backticks:
[
  {
    "text": "The extracted clause verbatim...",
    "label": [0]
  }
]
"""

# The system prompt to store in the dataset jsonl for downstream evaluation
DATASET_SYSTEM_CONTENT = "You are a legal clause classifier."

def chunk_text(text: str, max_chars: int = 3000) -> list[str]:
    """Splits text into chunks of paragraphs up to max_chars in length."""
    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk: list[str] = []
    current_len = 0
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if current_len + len(p) > max_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_len = len(p)
        else:
            current_chunk.append(p)
            current_len += len(p)
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks

def extract_json_array(text: str) -> list[dict]:
    """Attempt to parse a JSON array from textual LLM output."""
    text = text.strip()
    # Strip markdown if present
    if text.startswith("```"):
        text = re.sub(r"^```.*?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        # Fallback to regex finding brackets if simple load fails
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
            except Exception:
                pass
    return []

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL dataset from unstructured contract text.")
    parser.add_argument("--input", required=True, help="Path to input text file or directory of text files.")
    parser.add_argument("--output", required=True, help="Path to output JSONL file.")
    parser.add_argument("--api-key", required=True, help="OpenAI API key.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI Model to use for extraction.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    args = parser.parse_args()

    provider = create_provider("openai", args.api_key)
    
    in_path = Path(args.input)
    if in_path.is_file():
        files = [in_path]
    elif in_path.is_dir():
        files = list(in_path.glob("*.txt"))
    else:
        print(f"Input path {args.input} does not exist.")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_file = out_path.open("w", encoding="utf-8")
    
    total_extracted = 0
    for txt_file in files:
        print(f"Processing {txt_file}...")
        raw_text = txt_file.read_text(encoding="utf-8")
        chunks = chunk_text(raw_text)
        
        for i, chunk in enumerate(chunks):
            print(f"  - chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...", end="", flush=True)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chunk}
            ]
            response = provider.predict(
                model=args.model,
                messages=messages,
                temperature=0.0,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay
            )
            
            items = extract_json_array(response)
            print(f" extracted {len(items)} clauses.")
            for item in items:
                text = item.get("text", "").strip()
                labels_raw = item.get("label", [])
                if not text:
                    continue
                
                # convert labels_raw to string "[0]"
                raw_label_str = str(labels_raw)
                normalized_label = normalize_for_training(raw_label_str)
                
                record = {
                    "messages": [
                        {"role": "system", "content": DATASET_SYSTEM_CONTENT},
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": normalized_label},
                    ]
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_extracted += 1

    out_file.close()
    print(f"Done. Extracted {total_extracted} total clauses to {args.output}.")

if __name__ == "__main__":
    main()
