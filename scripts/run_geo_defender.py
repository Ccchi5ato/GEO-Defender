from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from geo_defender.pipeline import GEODefender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete GEO Defender pipeline.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL with query and documents.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--experience-library", type=Path, required=True)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--base-url", default=os.getenv("GENERATOR_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="GENERATOR_API_KEY")
    parser.add_argument("--base-reranker", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {args.api_key_env}.")
    defender = GEODefender(
        adapter_path=args.adapter,
        experience_library=args.experience_library,
        generator_model=args.generator_model,
        api_key=api_key,
        base_url=args.base_url or None,
        base_reranker=args.base_reranker,
        top_k=args.top_k,
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
            if not isinstance(item.get("query"), str) or not isinstance(item.get("documents"), list):
                raise ValueError(f"Invalid input schema on line {line_number}.")
            result = defender.defend(item["query"], item["documents"])
            result["sample_id"] = item.get("sample_id", str(line_number))
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()


if __name__ == "__main__":
    main()

