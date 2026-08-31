from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_user_content(instruct: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {doc}"


def load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any]:
    trust_remote_code = bool(config.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=bool(config.get("load_in_4bit", True)),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def score_batch(
    model: Any,
    tokenizer: Any,
    prefix_tokens: list[int],
    suffix_tokens: list[int],
    true_token_id: int,
    false_token_id: int,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> list[float]:
    margins: list[float] = []
    device = next(model.parameters()).device
    body_max_length = max_length - len(prefix_tokens) - len(suffix_tokens)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            add_special_tokens=False,
            max_length=body_max_length,
        )
        input_ids = [prefix_tokens + ids + suffix_tokens for ids in encoded["input_ids"]]
        batch = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            final_logits = outputs.logits[:, -1, :]
            true_logits = final_logits[:, true_token_id]
            false_logits = final_logits[:, false_token_id]
            batch_margins = true_logits - false_logits
        margins.extend(float(x) for x in batch_margins.detach().cpu().tolist())
    return margins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = load_jsonl(Path(args.input_path))
    model, tokenizer = load_model_and_tokenizer(config)

    prefix_tokens = tokenizer.encode(config["prefix"], add_special_tokens=False)
    suffix_tokens = tokenizer.encode(config["suffix"], add_special_tokens=False)
    true_token_id = tokenizer.convert_tokens_to_ids(config["true_token"])
    false_token_id = tokenizer.convert_tokens_to_ids(config["false_token"])
    instruct_key = str(config["instruct_key"])
    max_length = int(config["max_length"])

    texts: list[str] = []
    refs: list[tuple[int, str]] = []
    for row_idx, row in enumerate(rows):
        instruct = row[instruct_key]
        query = row["query"]
        for text_key, score_key in [
            ("pos1_text", "ref_clean_score"),
            ("neg_text", "ref_attack_score"),
            ("pos2_text", "ref_benign_score"),
        ]:
            texts.append(format_user_content(instruct, query, row[text_key]))
            refs.append((row_idx, score_key))

    margins = score_batch(
        model=model,
        tokenizer=tokenizer,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        true_token_id=true_token_id,
        false_token_id=false_token_id,
        texts=texts,
        max_length=max_length,
        batch_size=int(args.batch_size),
    )
    for (row_idx, score_key), margin in zip(refs, margins):
        rows[row_idx][score_key] = margin

    write_jsonl(Path(args.output_path), rows)
    print(json.dumps({"input_rows": len(rows), "scores_written": len(margins)}, indent=2))


if __name__ == "__main__":
    main()
