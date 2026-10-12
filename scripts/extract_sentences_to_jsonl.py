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

def extract_sentences(text: str) -> list[str]:
    """Extract sentences from text using regex."""
    sentences = []
    # Split text into paragraphs
    paragraphs = re.split(r'\n+', text)
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # Naive sentence boundary detection: punctuation followed by space and an uppercase letter
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9”"\'(-])', p)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sentences from ToS text and output as JSONL dataset.")
    parser.add_argument("--input", required=True, help="Path to input .txt file or directory")
    parser.add_argument("--output", required=True, help="Path to output .jsonl file")
    parser.add_argument("--system-content", default="You are a legal clause classifier.", help="System prompt content")
    parser.add_argument("--system-file", default=None, help="Optional path to system prompt file")
    parser.add_argument("--placeholder-label", default="[8]", help="Label to use for the assistant message (defaults to [8] for standard/fair)")
    args = parser.parse_args()

    system_content = args.system_content
    if args.system_file:
        system_content = Path(args.system_file).read_text(encoding="utf-8").strip()

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
    
    total_sentences = 0
    with out_path.open("w", encoding="utf-8") as out_file:
        for txt_file in files:
            print(f"Processing {txt_file}...")
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            sentences = extract_sentences(text)
            
            for sentence in sentences:
                record = {
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": sentence},
                        {"role": "assistant", "content": args.placeholder_label},
                    ]
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_sentences += 1
                
    print(f"Extracted {total_sentences} sentences and saved to {args.output}")

if __name__ == "__main__":
    main()
