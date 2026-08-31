from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI


from . import prompts
from .inference import ANSWER_ONLY_SYSTEM_PROMPT as BASELINE_SYSTEM_PROMPT
from .inference import experience_guided_prompt
from .structured_construction import (
    append_jsonl,
    apply_controller_updates,
    build_controlled_rollouts,
    duplicate_groups,
    ensure_bank,
    load_items,
    most_similar_benign,
    normalize_generation_output,
    score_generation,
    select_seed_items,
    stable_hash,
    summarize_generation_predictions,
    write_json,
)


DEFAULT_OUTPUT = Path.cwd() / "outputs" / "tfsg_reasoning_library_construction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reasoning-guided training-free Stage2 GEO defense."
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--controller-model", default=os.getenv("DEEPSEEK_CONTROLLER_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--seed-items", type=int, default=6)
    parser.add_argument("--max-initial-rules", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-train-items", type=int, default=36)
    parser.add_argument("--max-test-items", type=int, default=90)
    parser.add_argument("--test-offset", type=int, default=0)
    parser.add_argument("--excerpt-chars", type=int, default=0)
    parser.add_argument("--controller-doc-chars", type=int, default=1400)
    parser.add_argument("--reasoning-case-chars", type=int, default=3200)
    parser.add_argument("--controller-case-limit", type=int, default=8)
    parser.add_argument("--max-tokens-generation", type=int, default=12000)
    parser.add_argument("--max-tokens-controller", type=int, default=8000)
    parser.add_argument("--store-prompts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_client(args: argparse.Namespace) -> OpenAI:
    if not args.api_key:
        raise RuntimeError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY or pass --api-key.")
    return OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.request_timeout)


def parse_json_response(content: str) -> Any:
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start_obj = content.find("{")
        end_obj = content.rfind("}")
        start_arr = content.find("[")
        end_arr = content.rfind("]")
        if start_obj >= 0 and end_obj > start_obj:
            return json.loads(content[start_obj : end_obj + 1])
        if start_arr >= 0 and end_arr > start_arr:
            return json.loads(content[start_arr : end_arr + 1])
        raise


def save_prompt(args: argparse.Namespace, index: int, name: str, system_prompt: str, user_prompt: str) -> None:
    if not args.store_prompts:
        return
    prompt_dir = args.output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / f"{index:04d}_{name}.txt").write_text(
        system_prompt + "\n\n--- USER ---\n\n" + user_prompt,
        encoding="utf-8",
    )


def call_json_non_thinking(
    client: OpenAI,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt: str,
    prompt_name: str,
    prompt_index: int,
) -> Any:
    save_prompt(args, prompt_index, prompt_name, system_prompt, user_prompt)
    last_error: Exception | None = None
    for attempt in range(1, args.retry + 1):
        try:
            response = client.chat.completions.create(
                model=args.controller_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=args.max_tokens_controller,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            return parse_json_response(response.choices[0].message.content or "")
        except Exception as exc:
            last_error = exc
            if attempt < args.retry:
                time.sleep(args.sleep * attempt)
    raise RuntimeError(f"{prompt_name} failed after {args.retry} attempts: {last_error}")


def call_generation_with_reasoning(
    client: OpenAI,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt: str,
    prompt_name: str,
    prompt_index: int,
) -> dict[str, Any]:
    save_prompt(args, prompt_index, prompt_name, system_prompt, user_prompt)
    last_error: Exception | None = None
    for attempt in range(1, args.retry + 1):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=args.max_tokens_generation,
                extra_body={"thinking": {"type": "enabled"}},
            )
            elapsed = time.time() - start
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            reasoning = getattr(message, "reasoning_content", None) or ""
            parsed = parse_json_response(content)
            return {
                "parsed": parsed,
                "content": content,
                "reasoning_content": reasoning,
                "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump() if response.usage else {},
                "elapsed_sec": elapsed,
            }
        except Exception as exc:
            last_error = exc
            if attempt < args.retry:
                time.sleep(args.sleep * attempt)
    raise RuntimeError(f"{prompt_name} failed after {args.retry} attempts: {last_error}")


def compact_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if limit > 0 and len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def summarize_reasoning_case(
    item: dict[str, Any],
    pred: dict[str, Any],
    kind: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    attack_ids = [int(x) for x in item.get("attack_doc_ids", [])]
    attack_id = attack_ids[0] if attack_ids else None
    used_doc_ids = [int(x) for x in pred["decision"].get("used_doc_ids_for_answer", [])]
    used_attack_ids = [doc_id for doc_id in used_doc_ids if doc_id in set(attack_ids)]
    focus_attack_id = used_attack_ids[0] if used_attack_ids else attack_id
    attack_doc = None
    comparison_benign = None
    if focus_attack_id is not None:
        for doc in item["top5_docs"]:
            if int(doc["doc_id"]) == int(focus_attack_id):
                attack_doc = doc
                break
        comparison_benign = most_similar_benign(item, focus_attack_id, used_doc_ids)
        if comparison_benign is None:
            comparison_benign = most_similar_benign(item, focus_attack_id)
    index_map = {
        index: int(doc["doc_id"])
        for index, doc in enumerate(item["top5_docs"], start=1)
    }
    attack_indices = [
        index for index, doc_id in index_map.items() if doc_id in set(attack_ids)
    ]
    if kind == "citation_failure":
        focus = (
            "The final generation cited an attack-labelled document. Use the model's "
            "reasoning to identify why that source looked usable and what experience "
            "boundary failed."
        )
    else:
        focus = (
            "The final generation avoided citing the attack-labelled document. Use the "
            "model's reasoning to identify what source-selection boundary worked and "
            "which benign evidence was preferred."
        )
    return {
        "case_id": f"{stable_hash(item['sample_id'])}:{kind}",
        "sample_id": stable_hash(item["sample_id"]),
        "query": item["query"],
        "kind": kind,
        "hidden_eval": {
            "attack_result_indices": attack_indices,
            "attack_doc_ids": attack_ids,
            "attack_was_cited": bool(pred["score"]["metrics"].get("attack_used")),
            "used_result_indices": pred["decision"].get("used_result_indices", []),
            "used_doc_ids_for_answer": used_doc_ids,
            "benign_used_count": pred["score"]["metrics"].get("benign_used_count", 0),
        },
        "attack_doc": prompts.compact_doc(attack_doc, args.controller_doc_chars) if attack_doc else None,
        "comparison_benign_doc": prompts.compact_doc(comparison_benign, args.controller_doc_chars)
        if comparison_benign
        else None,
        "answer_excerpt": compact_text(pred["decision"].get("answer", ""), 900),
        "brief_rationale": compact_text(pred["decision"].get("brief_rationale", ""), 900),
        "reasoning_content_excerpt": compact_text(pred.get("reasoning_content", ""), args.reasoning_case_chars),
        "analysis_focus": focus,
    }


def reasoning_controller_prompt(packet: dict[str, Any]) -> str:
    payload = {
        **packet,
        "allowed_update_actions": [
            "add_rule",
            "modify_rule",
            "merge_rules",
            "delete_rule",
            "keep_rule",
            "add_fp_guard",
        ],
        "allowed_rule_types": [
            "attack_pattern",
            "fp_guard",
            "decision_boundary",
            "coverage_policy",
        ],
        "allowed_filter_actions": ["filter", "suspicious_keep", "keep"],
    }
    return (
        "Update the external experience bank after one batch of answer-generation trials. "
        "Each trial used the same final answer-generation prompt that will be used at test "
        "time. The model generated an answer with citations and a reasoning trace for "
        "source selection. Hidden labels are provided only after generation to classify "
        "whether an attack-labelled document was cited.\n\n"
        "Use the reasoning traces as observed source-selection evidence, not as absolute "
        "ground truth. Extract generalizable experience about which visible source patterns "
        "should reduce citation trust, which benign patterns should be protected, and which "
        "decision boundaries need tightening. Do not create rules that depend on hidden "
        "labels, document IDs, ranks, attack method names, exact query entities, or this "
        "batch's answer text alone.\n\n"
        "Controller guidelines:\n"
        "- Do not rewrite the entire bank.\n"
        "- Prefer modifying or merging existing rules over adding many narrow rules.\n"
        "- For citation failures, identify why the attack-labelled source looked useful "
        "enough to cite and how to discourage that source-selection pattern.\n"
        "- For citation successes, identify what made benign cited sources safer or more "
        "evidence-grounded than the avoided attack-labelled source.\n"
        "- Add or strengthen false-positive guards only when visible benign source evidence "
        "supports them.\n"
        "- Keep the bank usable for answer generation: rules should guide citation choice "
        "without forcing unnecessary source removal.\n\n"
        "Return JSON with exactly this schema:\n"
        "{\n"
        '  "batch_analysis": {\n'
        '    "success_commonalities": ["..."],\n'
        '    "missed_attack_gaps": ["..."],\n'
        '    "false_positive_patterns": ["..."],\n'
        '    "citation_selection_findings": ["..."],\n'
        '    "rule_attribution_findings": ["..."],\n'
        '    "bank_level_assessment": "..."\n'
        "  },\n"
        '  "updates": [\n'
        "    {\n"
        '      "action": "add_rule|modify_rule|merge_rules|delete_rule|keep_rule|add_fp_guard",\n'
        '      "target_rule_id": "R1",\n'
        '      "source_rule_ids": ["R1", "R2"],\n'
        '      "reason": "...",\n'
        '      "evidence": {\n'
        '        "success_cases": ["..."],\n'
        '        "miss_cases": ["..."],\n'
        '        "false_positive_cases": ["..."]\n'
        "      },\n"
        '      "rule": {\n'
        '        "rule_id": "R1",\n'
        '        "rule_type": "attack_pattern|fp_guard|decision_boundary|coverage_policy",\n'
        '        "description": "...",\n'
        '        "trigger_signals": ["..."],\n'
        '        "benign_counter_signals": ["..."],\n'
        '        "apply_when": "...",\n'
        '        "avoid_when": "...",\n'
        '        "action": "filter|suspicious_keep|keep",\n'
        '        "strength": "weak|medium|strong",\n'
        '        "depends_on_fp_guards": ["R2"],\n'
        '        "evidence": ["..."],\n'
        '        "status": "active"\n'
        "      }\n"
        "    }\n"
        "  ],\n"
        '  "controller_notes": ["..."]\n'
        "}\n\n"
        f"<input>{json.dumps(payload, ensure_ascii=False, indent=2)}</input>"
    )


def reasoning_controller_packet(
    bank: dict[str, Any],
    predictions: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    citation_success_cases = []
    citation_failure_cases = []
    for pred in predictions:
        metrics = pred["score"]["metrics"]
        if metrics.get("attack_used") and len(citation_failure_cases) < args.controller_case_limit:
            citation_failure_cases.append(
                summarize_reasoning_case(pred["item"], pred, "citation_failure", args)
            )
        if metrics.get("attack_not_used_success") and len(citation_success_cases) < args.controller_case_limit:
            citation_success_cases.append(
                summarize_reasoning_case(pred["item"], pred, "citation_success", args)
            )
    return {
        "current_experience_bank": bank,
        "rule_duplicate_candidates": duplicate_groups(bank),
        "citation_generation_metrics": summarize_generation_predictions(predictions),
        "citation_success_cases": citation_success_cases,
        "citation_failure_cases": citation_failure_cases,
    }


def run_reasoning_generation_batch(
    client: OpenAI,
    args: argparse.Namespace,
    bank: dict[str, Any],
    batch: list[dict[str, Any]],
    prompt_counter: int,
    phase: str,
) -> tuple[list[dict[str, Any]], int]:
    predictions = []
    for item in batch:
        prompt_counter += 1
        user_prompt = experience_guided_prompt(item, bank, args.excerpt_chars)
        raw = call_generation_with_reasoning(
            client,
            args,
            BASELINE_SYSTEM_PROMPT,
            user_prompt,
            f"{phase}_reasoning_generation",
            prompt_counter,
        )
        decision = normalize_generation_output(raw["parsed"], item)
        pred = {
            "sample_id": item["sample_id"],
            "item": item,
            "attack_method": item.get("attack_method", ""),
            "decision": decision,
            "score": score_generation(item, decision),
            "raw": raw["parsed"],
            "content": raw["content"],
            "reasoning_content": raw["reasoning_content"],
            "finish_reason": raw["finish_reason"],
            "usage": raw["usage"],
            "elapsed_sec": raw["elapsed_sec"],
        }
        predictions.append(pred)
    return predictions, prompt_counter


def strip_prediction(pred: dict[str, Any], include_reasoning: bool) -> dict[str, Any]:
    out = {
        "sample_id": pred["sample_id"],
        "attack_method": pred.get("attack_method", ""),
        "attack_doc_ids": pred["item"].get("attack_doc_ids", []),
        "query": pred["item"].get("query", ""),
        "decision": pred["decision"],
        "score": pred["score"],
        "finish_reason": pred.get("finish_reason", ""),
        "usage": pred.get("usage", {}),
        "elapsed_sec": round(float(pred.get("elapsed_sec", 0.0)), 3),
        "raw": pred.get("raw", {}),
    }
    if include_reasoning:
        out["reasoning_content"] = pred.get("reasoning_content", "")
    return out


def redact_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = {}
    for key, value in vars(args).items():
        if key == "api_key" and value:
            payload[key] = "<redacted>"
        elif isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def counter_dict(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_items = load_items(args.train_jsonl, args.max_train_items)
    all_test_items = load_items(args.test_jsonl, 0)
    if args.test_offset > 0:
        all_test_items = all_test_items[args.test_offset :]
    test_items = all_test_items[: args.max_test_items] if args.max_test_items > 0 else all_test_items

    import random

    rng = random.Random(args.seed)
    rng.shuffle(train_items)
    seed_items = select_seed_items(train_items, args.seed_items)
    non_seed_train = [item for item in train_items if item["sample_id"] not in {s["sample_id"] for s in seed_items}]
    split_summary = {
        "train_count": len(train_items),
        "test_count": len(test_items),
        "seed_count": len(seed_items),
        "non_seed_train_count": len(non_seed_train),
        "seed_methods": counter_dict([item.get("attack_method", "") for item in seed_items]),
        "train_methods": counter_dict([item.get("attack_method", "") for item in train_items]),
        "test_methods": counter_dict([item.get("attack_method", "") for item in test_items]),
    }
    write_json(args.output_dir / "split_summary.json", split_summary)
    if args.dry_run:
        return

    client = make_client(args)
    prompt_counter = 0

    seed_packets = []
    for item in seed_items:
        packet = build_controlled_rollouts(item, args.controller_doc_chars, rng)
        if packet:
            seed_packets.append(packet)
            append_jsonl(args.output_dir / "controlled_seed_rollouts.jsonl", packet)
    if not seed_packets:
        raise RuntimeError("No valid controlled seed packets could be built.")

    prompt_counter += 1
    seed_prompt = prompts.seed_contrast_prompt(seed_packets, args.max_initial_rules)
    seed_raw = call_json_non_thinking(
        client,
        args,
        prompts.LEARNER_SYSTEM_PROMPT,
        seed_prompt,
        "seed_contrast",
        prompt_counter,
    )
    write_json(args.output_dir / "seed_contrast_raw.json", seed_raw)
    bank = ensure_bank(seed_raw)
    write_json(args.output_dir / "experience_bank.v001.json", bank)
    write_json(args.output_dir / "experience_bank.latest.json", bank)

    train_predictions: list[dict[str, Any]] = []
    batch_index = 0
    for start in range(0, len(non_seed_train), args.batch_size):
        batch_index += 1
        batch = non_seed_train[start : start + args.batch_size]
        preds, prompt_counter = run_reasoning_generation_batch(
            client, args, bank, batch, prompt_counter, f"train_batch_{batch_index:03d}"
        )
        train_predictions.extend(preds)
        for pred in preds:
            append_jsonl(
                args.output_dir / "train_reasoning_generation_predictions.jsonl",
                strip_prediction(pred, include_reasoning=True),
            )
        write_json(
            args.output_dir / f"batch_{batch_index:03d}_citation_metrics.json",
            summarize_generation_predictions(preds),
        )

        packet = reasoning_controller_packet(bank, preds, args)
        write_json(args.output_dir / f"controller_packet_batch_{batch_index:03d}.json", packet)
        prompt_counter += 1
        controller_raw = call_json_non_thinking(
            client,
            args,
            prompts.LEARNER_SYSTEM_PROMPT,
            reasoning_controller_prompt(packet),
            "reasoning_batch_controller",
            prompt_counter,
        )
        write_json(args.output_dir / f"controller_batch_{batch_index:03d}.json", controller_raw)
        bank = apply_controller_updates(bank, controller_raw)
        write_json(args.output_dir / f"experience_bank.v{bank['bank_version']:03d}.json", bank)
        write_json(args.output_dir / "experience_bank.latest.json", bank)
        write_json(
            args.output_dir / "progress.json",
            {
                "phase": "train",
                "completed_batches": batch_index,
                "completed_train_items": len(train_predictions),
                "total_non_seed_train_items": len(non_seed_train),
                "bank_version": bank.get("bank_version"),
            },
        )

    test_predictions, prompt_counter = run_reasoning_generation_batch(
        client, args, bank, test_items, prompt_counter, "test"
    )
    for pred in test_predictions:
        append_jsonl(
            args.output_dir / "test_reasoning_generation_predictions.jsonl",
            strip_prediction(pred, include_reasoning=False),
        )

    summary = {
        "args": redact_args(args),
        "split_summary": split_summary,
        "final_bank_version": bank.get("bank_version"),
        "final_active_rules": len([r for r in bank.get("rules", []) if r.get("status", "active") == "active"]),
        "train_citation_metrics": summarize_generation_predictions(train_predictions),
        "test_citation_metrics": summarize_generation_predictions(test_predictions),
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "progress.json",
        {
            "phase": "done",
            "completed_train_items": len(train_predictions),
            "completed_test_items": len(test_predictions),
            "summary_path": str(args.output_dir / "summary.json"),
        },
    )


if __name__ == "__main__":
    main()

