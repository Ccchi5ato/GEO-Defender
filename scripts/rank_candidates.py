from __future__ import annotations

import argparse
import json
from pathlib import Path

from geo_defender.reranker import ShieldReranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Select top-k evidence with Shield Reranker.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-reranker", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    reranker = ShieldReranker(
        adapter_path=args.adapter,
        base_model=args.base_reranker,
        load_in_4bit=not args.no_4bit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            selected = reranker.rank(item["query"], item["documents"], top_k=args.top_k)
            output = {
                "sample_id": item.get("sample_id", str(line_number)),
                "query": item["query"],
                "top5_docs": [document.as_dict() for document in selected],
            }
            target.write(json.dumps(output, ensure_ascii=False) + "\n")
            target.flush()


if __name__ == "__main__":
    main()

