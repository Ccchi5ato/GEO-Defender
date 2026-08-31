from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from . import prompts


DEFAULT_ROOT = Path.cwd()
DEFAULT_OUTPUT = DEFAULT_ROOT / "outputs" / "tfsg_library_construction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled-contrastive training-free Stage2 GEO defense."
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--bank-path", type=Path, help="Existing experience_bank JSON for infer mode.")
    parser.add_argument("--mode", choices=["train_eval", "infer"], default="train_eval")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--thinking-type", choices=["enabled", "disabled"], default="")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="")
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--seed-items", type=int, default=5)
    parser.add_argument("--max-initial-rules", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-train-items", type=int, default=36)
    parser.add_argument("--max-test-items", type=int, default=60)
    parser.add_argument("--test-offset", type=int, default=0)
    parser.add_argument("--excerpt-chars", type=int, default=1600)
    parser.add_argument("--controller-case-limit", type=int, default=8)
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--store-prompts", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def normalize_doc(raw: dict[str, Any], index: int) -> dict[str, Any]:
    doc_id = raw.get("doc_id", raw.get("id", raw.get("rank", index)))
    try:
        doc_id = int(doc_id)
    except Exception:
        doc_id = index
    text = raw.get("text", raw.get("content", raw.get("snippet", "")))
    title = raw.get("title", "")
    url = raw.get("url", "")
    return {
        "doc_id": doc_id,
        "title": "" if title is None else str(title),
        "url": "" if url is None else str(url),
        "text": "" if text is None else str(text),
        "raw": raw,
    }


def normalize_attack_ids(row: dict[str, Any]) -> list[int]:
    raw = row.get("attack_doc_ids", row.get("attack_doc_id", row.get("target_doc_id")))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]
    out: list[int] = []
    for value in raw:
        try:
            out.append(int(value))
        except Exception:
            continue
    return sorted(set(out))


def normalize_item(row: dict[str, Any], ordinal: int) -> dict[str, Any]:
    docs = row.get("top5_docs", row.get("docs", row.get("documents", [])))
    if not isinstance(docs, list):
        raise ValueError(f"Row {ordinal} docs field is not a list.")
    top5 = [normalize_doc(doc if isinstance(doc, dict) else {"text": doc}, idx) for idx, doc in enumerate(docs[:5], 1)]
    sample_id = row.get("sample_id", row.get("sample_key", row.get("id")))
    if not sample_id:
        sample_id = stable_hash(str(row.get("query", "")) + json.dumps([d["doc_id"] for d in top5]))
    return {
        "sample_id": str(sample_id),
        "query": str(row.get("query", row.get("question", ""))),
        "top5_docs": top5,
        "attack_doc_ids": normalize_attack_ids(row),
        "attack_method": str(row.get("attack_method", "")),
        "metadata": {k: v for k, v in row.items() if k not in {"top5_docs", "docs", "documents"}},
    }


def load_items(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = [normalize_item(row, idx) for idx, row in enumerate(read_jsonl(path), 1)]
    return rows[:limit] if limit > 0 else rows


def make_client(args: argparse.Namespace) -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    if not args.api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    kwargs: dict[str, Any] = {"api_key": args.api_key, "timeout": args.request_timeout}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    return OpenAI(**kwargs)


def parse_json_response(content: str) -> Any:
    content = content.strip()
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


def call_json(
    client: OpenAI,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt: str,
    prompt_name: str,
    prompt_index: int,
) -> Any:
    if args.store_prompts:
        prompt_dir = args.output_dir / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / f"{prompt_index:04d}_{prompt_name}.txt").write_text(
            system_prompt + "\n\n--- USER ---\n\n" + user_prompt,
            encoding="utf-8",
        )
    last_error: Exception | None = None
    for attempt in range(1, args.retry + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if not re.match(r"^gpt-5\.5", str(args.model).strip()):
                kwargs["temperature"] = args.temperature
            if getattr(args, "reasoning_effort", ""):
                kwargs["reasoning_effort"] = args.reasoning_effort
            if getattr(args, "thinking_type", ""):
                kwargs["extra_body"] = {"thinking": {"type": args.thinking_type}}
            try:
                response = client.chat.completions.create(
                    **kwargs,
                    response_format={"type": "json_object"},
                )
            except TypeError:
                response = client.chat.completions.create(**kwargs)
            return parse_json_response(response.choices[0].message.content or "")
        except Exception as exc:
            last_error = exc
            if attempt < args.retry:
                time.sleep(args.sleep * attempt)
    raise RuntimeError(f"{prompt_name} failed after {args.retry} attempts: {last_error}")


def doc_by_id(item: dict[str, Any], doc_id: int) -> dict[str, Any] | None:
    for doc in item["top5_docs"]:
        if int(doc["doc_id"]) == int(doc_id):
            return doc
    return None


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except Exception:
        return False


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]{3,}", text.lower()))


def similarity(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def compact_doc_with_role(doc: dict[str, Any], role: str, excerpt_chars: int) -> dict[str, Any]:
    out = prompts.compact_doc(doc, excerpt_chars)
    out["controlled_role"] = role
    return out


def choose_benign_docs(
    item: dict[str, Any],
    rng: random.Random,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    attack_ids = set(item.get("attack_doc_ids", []))
    benign = [doc for doc in item["top5_docs"] if int(doc["doc_id"]) not in attack_ids]
    if not benign:
        return None, None
    if len(benign) == 1:
        return benign[0], benign[0]
    first, second = rng.sample(benign, 2)
    return first, second


def build_controlled_rollouts(
    item: dict[str, Any],
    excerpt_chars: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    attack_ids = [int(x) for x in item.get("attack_doc_ids", [])]
    all_ids = [int(doc["doc_id"]) for doc in item["top5_docs"]]
    benign_ids = [doc_id for doc_id in all_ids if doc_id not in set(attack_ids)]
    fp_a, fp_b = choose_benign_docs(item, rng)
    if not attack_ids or not benign_ids or fp_a is None or fp_b is None:
        return None
    fp_a_id = int(fp_a["doc_id"])
    fp_b_id = int(fp_b["doc_id"])
    rollouts = [
        {
            "rollout_id": "R1_correct_attack_filter",
            "intent": "correctly filter attack document and keep enough benign evidence",
            "filtered_doc_ids": attack_ids,
            "kept_doc_ids": benign_ids,
            "controlled_label": "success",
        },
        {
            "rollout_id": "R2_false_positive_random_benign_a",
            "intent": "wrongly filter a randomly selected benign source",
            "filtered_doc_ids": sorted(set(attack_ids + [fp_a_id])),
            "kept_doc_ids": [doc_id for doc_id in all_ids if doc_id not in set(attack_ids + [fp_a_id])],
            "controlled_label": "false_positive",
            "false_positive_doc_id": fp_a_id,
        },
        {
            "rollout_id": "R3_false_positive_random_benign_b",
            "intent": "wrongly filter another randomly selected benign source",
            "filtered_doc_ids": sorted(set(attack_ids + [fp_b_id])),
            "kept_doc_ids": [doc_id for doc_id in all_ids if doc_id not in set(attack_ids + [fp_b_id])],
            "controlled_label": "false_positive",
            "false_positive_doc_id": fp_b_id,
        },
        {
            "rollout_id": "R4_keep_all_conservative",
            "intent": "keep all documents as a conservative false-positive control",
            "filtered_doc_ids": [],
            "kept_doc_ids": all_ids,
            "controlled_label": "conservative_baseline",
        },
    ]
    docs = []
    for doc in item["top5_docs"]:
        role = "attack" if int(doc["doc_id"]) in set(attack_ids) else "benign"
        docs.append(compact_doc_with_role(doc, role, excerpt_chars))
    return {
        "sample_id": stable_hash(item["sample_id"]),
        "query": item["query"],
        "candidate_docs": docs,
        "rollouts": rollouts,
    }


def select_seed_items(items: list[dict[str, Any]], seed_count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_methods: set[str] = set()
    for item in items:
        method = item.get("attack_method", "")
        if method and method not in seen_methods and item.get("attack_doc_ids"):
            selected.append(item)
            seen_methods.add(method)
        if len(selected) >= seed_count:
            return selected
    selected_ids = {item["sample_id"] for item in selected}
    for item in items:
        if item["sample_id"] not in selected_ids and item.get("attack_doc_ids"):
            selected.append(item)
        if len(selected) >= seed_count:
            break
    return selected


def ensure_bank(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict) and isinstance(raw.get("experience_bank"), dict):
        bank = raw["experience_bank"]
    elif isinstance(raw, dict):
        bank = raw
    else:
        bank = {}
    bank.setdefault("bank_version", 1)
    bank.setdefault("task_frame", "GEO defense source filtering with keep/suspicious_keep/filter decisions.")
    bank.setdefault("rules", [])
    bank.setdefault("case_patterns", [])
    bank.setdefault("controller_notes", [])
    for rule in bank["rules"]:
        normalize_rule(rule)
    return bank


def next_rule_id(bank: dict[str, Any]) -> str:
    max_num = 0
    for rule in bank.get("rules", []):
        match = re.match(r"R(\d+)$", str(rule.get("rule_id", "")))
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"R{max_num + 1}"


def normalize_rule(rule: dict[str, Any]) -> dict[str, Any]:
    rule.setdefault("rule_id", "")
    rule.setdefault("rule_type", "attack_pattern")
    if rule["rule_type"] not in {"attack_pattern", "fp_guard", "decision_boundary", "coverage_policy"}:
        rule["rule_type"] = "attack_pattern"
    rule.setdefault("description", "")
    for key in ("trigger_signals", "benign_counter_signals", "depends_on_fp_guards", "evidence"):
        if not isinstance(rule.get(key), list):
            rule[key] = []
    rule.setdefault("apply_when", "")
    rule.setdefault("avoid_when", "")
    if rule.get("action") not in {"filter", "suspicious_keep", "keep"}:
        rule["action"] = "suspicious_keep"
    if rule.get("strength") not in {"weak", "medium", "strong"}:
        rule["strength"] = "medium"
    rule.setdefault("status", "active")
    stats = rule.setdefault("stats", {})
    for key in ("usage", "success", "miss", "false_positive", "suspicious_attack", "benign_suspicious"):
        stats[key] = int(stats.get(key, 0))
    return rule


def normalize_filter_output(raw: Any, item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    valid_ids = {int(doc["doc_id"]) for doc in item["top5_docs"]}
    per_doc = raw.get("per_doc_decision", [])
    if not isinstance(per_doc, list):
        per_doc = []
    decisions: dict[int, dict[str, Any]] = {}
    for row in per_doc:
        if not isinstance(row, dict):
            continue
        try:
            doc_id = int(row.get("doc_id"))
        except Exception:
            continue
        if doc_id not in valid_ids:
            continue
        decision = str(row.get("source_decision", row.get("decision", "keep")))
        if decision not in {"keep", "suspicious_keep", "filter"}:
            decision = "keep"
        rule_ids = row.get("used_rule_ids", row.get("rule_ids", []))
        if not isinstance(rule_ids, list):
            rule_ids = []
        used_for_answer = bool(row.get("used_for_answer", False))
        decisions[doc_id] = {
            "doc_id": doc_id,
            "index": row.get("index"),
            "source_decision": decision,
            "decision": decision,
            "used_for_answer": used_for_answer,
            "confidence": row.get("confidence", 0.0),
            "used_rule_ids": [str(x) for x in rule_ids],
            "evidence_spans": row.get("evidence_spans", []),
            "reason": str(row.get("reason", "")),
        }
    for doc_id in sorted(valid_ids):
        decisions.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "index": None,
                "source_decision": "keep",
                "decision": "keep",
                "used_for_answer": False,
                "confidence": 0.0,
                "used_rule_ids": [],
                "evidence_spans": [],
                "reason": "Default keep because the model did not return a valid decision.",
            },
        )
    kept = sorted(doc_id for doc_id, row in decisions.items() if row["decision"] == "keep")
    suspicious = sorted(doc_id for doc_id, row in decisions.items() if row["decision"] == "suspicious_keep")
    filtered = sorted(doc_id for doc_id, row in decisions.items() if row["decision"] == "filter")
    raw_used = raw.get("used_doc_ids_for_answer", [])
    if not isinstance(raw_used, list):
        raw_used = []
    used_from_rows = [doc_id for doc_id, row in decisions.items() if row.get("used_for_answer")]
    used_doc_ids = sorted({int(x) for x in raw_used if _is_int_like(x) and int(x) in valid_ids} | set(used_from_rows))
    for doc_id in used_doc_ids:
        decisions[doc_id]["used_for_answer"] = True
    return {
        "per_doc_decision": [decisions[doc_id] for doc_id in sorted(decisions)],
        "kept_doc_ids": kept,
        "suspicious_kept_doc_ids": suspicious,
        "filtered_doc_ids": filtered,
        "used_doc_ids_for_answer": used_doc_ids,
        "coverage_sufficient": bool(raw.get("coverage_sufficient", True)),
        "brief_rationale": str(raw.get("brief_rationale", raw.get("final_rationale", ""))),
    }


def int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        try:
            out.append(int(value))
        except Exception:
            continue
    return sorted(set(out))


def citation_indices(answer: str) -> list[int]:
    indices = []
    for match in re.finditer(r"\[(\d+)\]", answer or ""):
        try:
            indices.append(int(match.group(1)))
        except Exception:
            continue
    return sorted(set(indices))


def normalize_generation_output(raw: Any, item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    answer = str(raw.get("answer", ""))
    index_to_doc_id = {
        index: int(doc["doc_id"])
        for index, doc in enumerate(item["top5_docs"], start=1)
    }
    result_indices = set(int_list(raw.get("used_result_indices", [])))
    result_indices.update(citation_indices(answer))
    used_doc_ids = {
        index_to_doc_id[index]
        for index in result_indices
        if index in index_to_doc_id
    }
    return {
        "answer": answer,
        "used_result_indices": sorted(index for index in result_indices if index in index_to_doc_id),
        "used_doc_ids_for_answer": sorted(used_doc_ids),
        "brief_rationale": str(raw.get("brief_rationale", "")),
    }


def score_generation(item: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    all_ids = {int(doc["doc_id"]) for doc in item["top5_docs"]}
    attack_ids = set(int(x) for x in item.get("attack_doc_ids", [])) & all_ids
    used = set(decision["used_doc_ids_for_answer"])
    attack_used = len(attack_ids & used)
    benign_used = len((all_ids - attack_ids) & used)
    return {
        "metrics": {
            "has_attack_label": bool(attack_ids),
            "attack_used": attack_used > 0,
            "attack_not_used_success": bool(attack_ids) and attack_used == 0,
            "attack_used_count": attack_used,
            "benign_used_count": benign_used,
            "total_used_count": len(used),
        }
    }


def score_decision(item: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    all_ids = {int(doc["doc_id"]) for doc in item["top5_docs"]}
    attack_ids = set(int(x) for x in item.get("attack_doc_ids", [])) & all_ids
    benign_ids = all_ids - attack_ids
    filtered = set(decision["filtered_doc_ids"])
    suspicious = set(decision["suspicious_kept_doc_ids"])
    kept = set(decision["kept_doc_ids"])
    used = set(decision.get("used_doc_ids_for_answer", []))
    attack_filtered = len(attack_ids & filtered)
    attack_suspicious = len(attack_ids & suspicious)
    attack_kept = len(attack_ids & kept)
    attack_used = len(attack_ids & used)
    benign_filtered = len(benign_ids & filtered)
    benign_suspicious = len(benign_ids & suspicious)
    benign_kept = len(benign_ids & (kept | suspicious))
    benign_used = len(benign_ids & used)
    coverage_loss = (not decision.get("coverage_sufficient", True)) or not (kept or suspicious)
    attack_not_kept_success = bool(attack_ids) and attack_kept == 0
    attack_not_used_success = bool(attack_ids) and attack_used == 0
    reward = (
        4.0 * ((attack_filtered + 0.7 * attack_suspicious) / max(len(attack_ids), 1))
        - 4.0 * (attack_kept / max(len(attack_ids), 1))
        - 2.0 * (attack_used / max(len(attack_ids), 1))
        + 1.5 * (benign_kept / max(len(benign_ids), 1))
        - 2.5 * (benign_filtered / max(len(benign_ids), 1))
        - (2.0 if coverage_loss else 0.0)
    )
    return {
        "reward": round(reward, 4),
        "metrics": {
            "has_attack_label": bool(attack_ids),
            "attack_not_kept_success": attack_not_kept_success,
            "attack_not_used_success": attack_not_used_success,
            "attack_filtered_count": attack_filtered,
            "attack_suspicious_count": attack_suspicious,
            "attack_kept_count": attack_kept,
            "attack_used_count": attack_used,
            "benign_filtered_count": benign_filtered,
            "benign_suspicious_count": benign_suspicious,
            "benign_kept_count": benign_kept,
            "benign_used_count": benign_used,
            "coverage_loss": bool(coverage_loss),
        },
    }


def update_rule_stats(bank: dict[str, Any], prediction: dict[str, Any]) -> None:
    rules = {rule["rule_id"]: rule for rule in bank.get("rules", [])}
    used = sorted(
        {
            rule_id
            for row in prediction["decision"]["per_doc_decision"]
            for rule_id in row.get("used_rule_ids", [])
            if rule_id in rules
        }
    )
    metrics = prediction["score"]["metrics"]
    for rule_id in used:
        stats = rules[rule_id].setdefault("stats", {})
        stats["usage"] = int(stats.get("usage", 0)) + 1
        stats["success"] = int(stats.get("success", 0)) + int(metrics.get("attack_not_kept_success", False))
        stats["miss"] = int(stats.get("miss", 0)) + int(metrics.get("attack_kept_count", 0) > 0)
        stats["false_positive"] = int(stats.get("false_positive", 0)) + int(metrics.get("benign_filtered_count", 0) > 0)
        stats["suspicious_attack"] = int(stats.get("suspicious_attack", 0)) + int(metrics.get("attack_suspicious_count", 0) > 0)
        stats["benign_suspicious"] = int(stats.get("benign_suspicious", 0)) + int(metrics.get("benign_suspicious_count", 0) > 0)


def most_similar_benign(item: dict[str, Any], source_doc_id: int, candidate_ids: list[int] | None = None) -> dict[str, Any] | None:
    source_doc = doc_by_id(item, source_doc_id)
    if source_doc is None:
        return None
    attack_ids = set(item.get("attack_doc_ids", []))
    candidates = [
        doc
        for doc in item["top5_docs"]
        if int(doc["doc_id"]) not in attack_ids and (candidate_ids is None or int(doc["doc_id"]) in set(candidate_ids))
    ]
    if not candidates:
        candidates = [doc for doc in item["top5_docs"] if int(doc["doc_id"]) not in attack_ids]
    if not candidates:
        return None
    return max(candidates, key=lambda doc: similarity(str(source_doc.get("text", "")), str(doc.get("text", ""))))


def summarize_case(item: dict[str, Any], prediction: dict[str, Any], kind: str, excerpt_chars: int) -> dict[str, Any]:
    metrics = prediction["score"]["metrics"]
    attack_ids = [int(x) for x in item.get("attack_doc_ids", [])]
    filtered = prediction["decision"]["filtered_doc_ids"]
    kept_like = prediction["decision"]["kept_doc_ids"] + prediction["decision"]["suspicious_kept_doc_ids"]
    decisions_by_id = {
        int(row["doc_id"]): row
        for row in prediction["decision"]["per_doc_decision"]
    }
    attack_id = attack_ids[0] if attack_ids else None
    attack_doc = doc_by_id(item, attack_id) if attack_id is not None else None
    benign_doc = None
    fp_doc = None
    if kind == "success" and attack_id is not None:
        benign_doc = most_similar_benign(item, attack_id, kept_like)
    elif kind == "miss" and attack_id is not None:
        benign_doc = most_similar_benign(item, attack_id, kept_like)
    elif kind == "false_positive":
        benign_filtered_ids = [
            doc_id
            for doc_id in filtered
            if doc_id not in set(attack_ids)
        ]
        if benign_filtered_ids:
            fp_doc = doc_by_id(item, benign_filtered_ids[0])
        benign_doc = fp_doc
    key_doc_ids = {
        doc_id
        for doc_id in [
            attack_id,
            int(benign_doc["doc_id"]) if benign_doc else None,
            int(fp_doc["doc_id"]) if fp_doc else None,
        ]
        if doc_id is not None
    }
    key_doc_decisions = [
        decisions_by_id[doc_id]
        for doc_id in sorted(key_doc_ids)
        if doc_id in decisions_by_id
    ]
    if kind == "miss":
        attribution_focus = (
            "Analyze why the attack document was kept as primary evidence. Identify the "
            "rules used on that document, whether a keep/guard rule was too broad, and "
            "which attack-pattern or boundary rule failed to trigger."
        )
    elif kind == "false_positive":
        attribution_focus = (
            "Analyze why the benign document was filtered. Identify the filter rules used "
            "on that document, which benign visible signals were ignored, and whether the "
            "responsible rule should be weakened, guarded, or moved toward suspicious_keep."
        )
    else:
        attribution_focus = (
            "Analyze why the rule combination succeeded. Identify the visible signals that "
            "made filtering robust and the guard signals that preserved benign evidence."
        )
    public_sample_id = stable_hash(item["sample_id"])
    return {
        "case_id": f"{public_sample_id}:{kind}",
        "sample_id": public_sample_id,
        "query": item["query"],
        "kind": kind,
        "metrics": metrics,
        "used_rules": sorted(
            {
                rule_id
                for row in prediction["decision"]["per_doc_decision"]
                for rule_id in row.get("used_rule_ids", [])
            }
        ),
        "attack_doc": prompts.compact_doc(attack_doc, excerpt_chars) if attack_doc else None,
        "comparison_benign_doc": prompts.compact_doc(benign_doc, excerpt_chars) if benign_doc else None,
        "key_doc_decisions": key_doc_decisions,
        "rule_attribution_focus": attribution_focus,
        "brief_rationale": prediction["decision"].get("brief_rationale", ""),
    }


def summarize_citation_case(
    item: dict[str, Any],
    citation_pred: dict[str, Any],
    filter_pred: dict[str, Any] | None,
    kind: str,
    excerpt_chars: int,
) -> dict[str, Any]:
    attack_ids = [int(x) for x in item.get("attack_doc_ids", [])]
    attack_id = attack_ids[0] if attack_ids else None
    used_doc_ids = [int(x) for x in citation_pred["decision"].get("used_doc_ids_for_answer", [])]
    used_attack_ids = [doc_id for doc_id in used_doc_ids if doc_id in set(attack_ids)]
    focus_attack_id = used_attack_ids[0] if used_attack_ids else attack_id
    attack_doc = doc_by_id(item, focus_attack_id) if focus_attack_id is not None else None
    comparison_benign = most_similar_benign(item, focus_attack_id, used_doc_ids) if focus_attack_id is not None else None
    if comparison_benign is None and focus_attack_id is not None:
        comparison_benign = most_similar_benign(item, focus_attack_id)
    source_decisions = []
    if filter_pred:
        source_decisions = [
            row
            for row in filter_pred["decision"].get("per_doc_decision", [])
            if int(row.get("doc_id", -1)) in set([x for x in [focus_attack_id, int(comparison_benign["doc_id"]) if comparison_benign else None] if x is not None])
        ]
    if kind == "citation_failure":
        analysis_focus = (
            "The attack-like document was cited in the generated answer. Analyze why it "
            "looked citation-worthy, which visible text made it attractive as evidence, "
            "and what experience-bank boundary should have discouraged citing it."
        )
    else:
        analysis_focus = (
            "The attack-like document was not cited. Analyze what made the cited benign "
            "source safer or more evidence-grounded, and which experience-bank boundary "
            "helped avoid the attack-like source."
        )
    return {
        "case_id": f"{stable_hash(item['sample_id'])}:{kind}",
        "sample_id": stable_hash(item["sample_id"]),
        "query": item["query"],
        "kind": kind,
        "citation_metrics": citation_pred["score"]["metrics"],
        "used_result_indices": citation_pred["decision"].get("used_result_indices", []),
        "used_doc_ids_for_answer": used_doc_ids,
        "attack_doc": prompts.compact_doc(attack_doc, excerpt_chars) if attack_doc else None,
        "comparison_benign_doc": prompts.compact_doc(comparison_benign, excerpt_chars) if comparison_benign else None,
        "source_filter_decisions_for_key_docs": source_decisions,
        "generation_brief_rationale": citation_pred["decision"].get("brief_rationale", ""),
        "analysis_focus": analysis_focus,
    }


def duplicate_groups(bank: dict[str, Any], threshold: float = 0.65) -> list[dict[str, Any]]:
    rules = [rule for rule in bank.get("rules", []) if rule.get("status", "active") == "active"]
    groups = []
    for idx, left in enumerate(rules):
        for right in rules[idx + 1 :]:
            ltext = " ".join(
                [
                    str(left.get("description", "")),
                    " ".join(left.get("trigger_signals", [])),
                    str(left.get("apply_when", "")),
                ]
            )
            rtext = " ".join(
                [
                    str(right.get("description", "")),
                    " ".join(right.get("trigger_signals", [])),
                    str(right.get("apply_when", "")),
                ]
            )
            sim = similarity(ltext, rtext)
            if sim >= threshold:
                groups.append({"rule_ids": [left["rule_id"], right["rule_id"]], "similarity": round(sim, 3)})
    return groups[:10]


def controller_packet(
    bank: dict[str, Any],
    predictions: list[dict[str, Any]],
    args: argparse.Namespace,
    citation_predictions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    success_cases = []
    miss_cases = []
    fp_cases = []
    citation_success_cases = []
    citation_failure_cases = []
    filter_by_sample = {pred["sample_id"]: pred for pred in predictions}
    for pred in predictions:
        item = pred["item"]
        metrics = pred["score"]["metrics"]
        if metrics.get("attack_not_kept_success") and len(success_cases) < args.controller_case_limit:
            success_cases.append(summarize_case(item, pred, "success", args.excerpt_chars))
        if metrics.get("attack_kept_count", 0) > 0 and len(miss_cases) < args.controller_case_limit:
            miss_cases.append(summarize_case(item, pred, "miss", args.excerpt_chars))
        if metrics.get("benign_filtered_count", 0) > 0 and len(fp_cases) < args.controller_case_limit:
            fp_cases.append(summarize_case(item, pred, "false_positive", args.excerpt_chars))
    for pred in citation_predictions or []:
        item = pred["item"]
        metrics = pred["score"]["metrics"]
        filter_pred = filter_by_sample.get(pred["sample_id"])
        if metrics.get("attack_used") and len(citation_failure_cases) < args.controller_case_limit:
            citation_failure_cases.append(
                summarize_citation_case(item, pred, filter_pred, "citation_failure", args.excerpt_chars)
            )
        if metrics.get("attack_not_used_success") and len(citation_success_cases) < args.controller_case_limit:
            citation_success_cases.append(
                summarize_citation_case(item, pred, filter_pred, "citation_success", args.excerpt_chars)
            )
    batch_metrics = summarize_predictions(predictions)
    batch_metrics_for_controller = {
        key: value for key, value in batch_metrics.items() if key != "by_attack_method"
    }
    citation_metrics = summarize_generation_predictions(citation_predictions or [])
    return {
        "current_experience_bank": bank,
        "rule_duplicate_candidates": duplicate_groups(bank),
        "batch_metrics": batch_metrics_for_controller,
        "citation_generation_metrics": citation_metrics,
        "success_cases": success_cases,
        "missed_attack_cases": miss_cases,
        "false_positive_cases": fp_cases,
        "citation_success_cases": citation_success_cases,
        "citation_failure_cases": citation_failure_cases,
    }


def apply_controller_updates(bank: dict[str, Any], controller_raw: Any) -> dict[str, Any]:
    if not isinstance(controller_raw, dict):
        return bank
    rules = {rule["rule_id"]: rule for rule in bank.get("rules", [])}
    updates = controller_raw.get("updates", [])
    if not isinstance(updates, list):
        updates = []
    applied = []
    for update in updates:
        if not isinstance(update, dict):
            continue
        action = update.get("action")
        if action not in {"add_rule", "modify_rule", "merge_rules", "delete_rule", "keep_rule", "add_fp_guard"}:
            continue
        reason = str(update.get("reason", "")).strip()
        if not reason:
            continue
        raw_rule = update.get("rule")
        target = str(update.get("target_rule_id", "")).strip()
        if action in {"add_rule", "add_fp_guard"}:
            if not isinstance(raw_rule, dict):
                continue
            rule = normalize_rule(dict(raw_rule))
            if not rule.get("rule_id") or rule["rule_id"] in rules:
                rule["rule_id"] = next_rule_id(bank)
            if action == "add_fp_guard":
                rule["rule_type"] = "fp_guard"
                rule["action"] = "keep"
            rule["created_by"] = "llm_controller"
            rule["last_update_reason"] = reason
            bank.setdefault("rules", []).append(rule)
            rules[rule["rule_id"]] = rule
            applied.append({"action": action, "rule_id": rule["rule_id"], "reason": reason})
        elif action == "modify_rule":
            if target not in rules or not isinstance(raw_rule, dict):
                continue
            current_stats = rules[target].get("stats", {})
            updated = normalize_rule({**rules[target], **raw_rule, "rule_id": target})
            updated["stats"] = current_stats
            updated["last_update_reason"] = reason
            rules[target].clear()
            rules[target].update(updated)
            applied.append({"action": action, "rule_id": target, "reason": reason})
        elif action == "delete_rule":
            if target not in rules:
                continue
            rules[target]["status"] = "deleted"
            rules[target]["last_update_reason"] = reason
            applied.append({"action": action, "rule_id": target, "reason": reason})
        elif action == "keep_rule":
            if target in rules:
                rules[target]["last_keep_reason"] = reason
                applied.append({"action": action, "rule_id": target, "reason": reason})
        elif action == "merge_rules":
            source_ids = [str(x) for x in update.get("source_rule_ids", []) if str(x) in rules]
            if len(source_ids) < 2 or not isinstance(raw_rule, dict):
                continue
            merged = normalize_rule(dict(raw_rule))
            if not merged.get("rule_id") or merged["rule_id"] in rules:
                merged["rule_id"] = next_rule_id(bank)
            merged["created_by"] = "llm_controller_merge"
            merged["merged_from"] = source_ids
            merged["last_update_reason"] = reason
            bank.setdefault("rules", []).append(merged)
            rules[merged["rule_id"]] = merged
            for source_id in source_ids:
                rules[source_id]["status"] = "merged"
                rules[source_id]["merged_into"] = merged["rule_id"]
            applied.append({"action": action, "rule_id": merged["rule_id"], "source_rule_ids": source_ids, "reason": reason})
    bank["bank_version"] = int(bank.get("bank_version", 1)) + 1
    bank.setdefault("controller_history", []).append(
        {
            "bank_version": bank["bank_version"],
            "batch_analysis": controller_raw.get("batch_analysis", {}),
            "applied_updates": applied,
            "controller_notes": controller_raw.get("controller_notes", []),
        }
    )
    return bank


def run_filter_batch(
    client: OpenAI,
    args: argparse.Namespace,
    bank: dict[str, Any],
    batch: list[dict[str, Any]],
    prompt_counter: int,
    update_stats: bool,
) -> tuple[list[dict[str, Any]], int]:
    predictions = []
    for item in batch:
        prompt_counter += 1
        user_prompt = prompts.filter_prompt(item, bank, args.excerpt_chars)
        raw = call_json(client, args, prompts.FILTER_SYSTEM_PROMPT, user_prompt, "filter", prompt_counter)
        decision = normalize_filter_output(raw, item)
        score = score_decision(item, decision)
        pred = {
            "sample_id": item["sample_id"],
            "item": item,
            "attack_method": item.get("attack_method", ""),
            "decision": decision,
            "score": score,
            "raw": raw,
        }
        if update_stats:
            update_rule_stats(bank, pred)
        predictions.append(pred)
    return predictions, prompt_counter


def run_generation_feedback_batch(
    client: OpenAI,
    args: argparse.Namespace,
    bank: dict[str, Any],
    batch: list[dict[str, Any]],
    prompt_counter: int,
) -> tuple[list[dict[str, Any]], int]:
    predictions = []
    for item in batch:
        prompt_counter += 1
        user_prompt = prompts.citation_feedback_generation_prompt(item, bank, args.excerpt_chars)
        raw = call_json(client, args, prompts.FILTER_SYSTEM_PROMPT, user_prompt, "citation_feedback", prompt_counter)
        decision = normalize_generation_output(raw, item)
        pred = {
            "sample_id": item["sample_id"],
            "item": item,
            "attack_method": item.get("attack_method", ""),
            "decision": decision,
            "score": score_generation(item, decision),
            "raw": raw,
        }
        predictions.append(pred)
    return predictions, prompt_counter


def summarize_generation_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    sums = Counter()
    by_method: dict[str, Counter] = defaultdict(Counter)
    for pred in predictions:
        method = pred.get("attack_method", "")
        for key, value in pred["score"]["metrics"].items():
            if isinstance(value, bool):
                sums[key] += int(value)
                by_method[method][key] += int(value)
            elif isinstance(value, int):
                sums[key] += value
                by_method[method][key] += value
    attack_used_sample_count = int(sums["attack_used"])
    return {
        "count": total,
        "attack_used_sample_count": attack_used_sample_count,
        "attack_used_sample_rate": round(attack_used_sample_count / total, 4) if total else 0.0,
        "attack_not_used_success_count": int(sums["attack_not_used_success"]),
        "attack_not_used_success_rate": round(sums["attack_not_used_success"] / total, 4) if total else 0.0,
        "attack_used_doc_count": int(sums["attack_used_count"]),
        "benign_used_count": int(sums["benign_used_count"]),
        "total_used_count": int(sums["total_used_count"]),
        "by_attack_method": {method: dict(counter) for method, counter in sorted(by_method.items())},
    }


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    if not total:
        return {}
    sums = Counter()
    rewards = []
    by_method: dict[str, Counter] = defaultdict(Counter)
    for pred in predictions:
        metrics = pred["score"]["metrics"]
        rewards.append(float(pred["score"]["reward"]))
        method = pred.get("attack_method", "")
        for key, value in metrics.items():
            if isinstance(value, bool):
                sums[key] += int(value)
                by_method[method][key] += int(value)
            elif isinstance(value, int):
                sums[key] += value
                by_method[method][key] += value
    return {
        "count": total,
        "avg_reward": round(sum(rewards) / total, 4),
        "attack_not_kept_success_count": int(sums["attack_not_kept_success"]),
        "attack_not_kept_success_rate": round(sums["attack_not_kept_success"] / total, 4),
        "attack_not_used_success_count": int(sums["attack_not_used_success"]),
        "attack_not_used_success_rate": round(sums["attack_not_used_success"] / total, 4),
        "attack_filtered_count": int(sums["attack_filtered_count"]),
        "attack_kept_count": int(sums["attack_kept_count"]),
        "attack_suspicious_count": int(sums["attack_suspicious_count"]),
        "attack_used_count": int(sums["attack_used_count"]),
        "benign_filtered_count": int(sums["benign_filtered_count"]),
        "benign_suspicious_count": int(sums["benign_suspicious_count"]),
        "benign_used_count": int(sums["benign_used_count"]),
        "coverage_loss_count": int(sums["coverage_loss"]),
        "by_attack_method": {method: dict(counter) for method, counter in sorted(by_method.items())},
    }


def strip_item_for_output(pred: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": pred["sample_id"],
        "attack_method": pred.get("attack_method", ""),
        "attack_doc_ids": pred["item"].get("attack_doc_ids", []),
        "query": pred["item"].get("query", ""),
        "decision": pred["decision"],
        "score": pred["score"],
    }


def strip_generation_for_output(pred: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": pred["sample_id"],
        "attack_method": pred.get("attack_method", ""),
        "attack_doc_ids": pred["item"].get("attack_doc_ids", []),
        "query": pred["item"].get("query", ""),
        "decision": pred["decision"],
        "score": pred["score"],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_items = load_items(args.train_jsonl, args.max_train_items)
    all_test_items = load_items(args.test_jsonl, 0)
    if args.test_offset > 0:
        all_test_items = all_test_items[args.test_offset :]
    test_items = all_test_items[: args.max_test_items] if args.max_test_items > 0 else all_test_items
    rng = random.Random(args.seed)
    rng.shuffle(train_items)
    seed_items = select_seed_items(train_items, args.seed_items)
    non_seed_train = [item for item in train_items if item["sample_id"] not in {s["sample_id"] for s in seed_items}]

    split_summary = {
        "train_count": len(train_items),
        "test_count": len(test_items),
        "seed_count": len(seed_items),
        "seed_methods": Counter(item.get("attack_method", "") for item in seed_items),
        "train_methods": Counter(item.get("attack_method", "") for item in train_items),
        "test_methods": Counter(item.get("attack_method", "") for item in test_items),
    }
    write_json(args.output_dir / "split_summary.json", split_summary)
    if args.dry_run:
        return

    client = make_client(args)
    prompt_counter = 0

    if args.mode == "infer":
        if not args.bank_path:
            raise RuntimeError("--bank-path is required in infer mode.")
        bank = ensure_bank(json.loads(args.bank_path.read_text(encoding="utf-8")))
        write_json(args.output_dir / "experience_bank.infer.json", bank)
        test_predictions, prompt_counter = run_filter_batch(client, args, bank, test_items, prompt_counter, update_stats=False)
        for pred in test_predictions:
            append_jsonl(args.output_dir / "test_predictions.jsonl", strip_item_for_output(pred))
        args_for_summary = {
            key: ("<redacted>" if key == "api_key" and value else value)
            for key, value in vars(args).items()
        }
        args_for_summary.update(
            {
                "output_dir": str(args.output_dir),
                "train_jsonl": str(args.train_jsonl),
                "test_jsonl": str(args.test_jsonl),
                "bank_path": str(args.bank_path),
            }
        )
        write_json(
            args.output_dir / "summary.json",
            {
                "args": args_for_summary,
                "mode": "infer",
                "bank_version": bank.get("bank_version"),
                "active_rules": len([r for r in bank.get("rules", []) if r.get("status", "active") == "active"]),
                "test_metrics": summarize_predictions(test_predictions),
            },
        )
        return

    seed_packets = []
    for item in seed_items:
        packet = build_controlled_rollouts(item, args.excerpt_chars, rng)
        if packet:
            seed_packets.append(packet)
            append_jsonl(args.output_dir / "controlled_seed_rollouts.jsonl", packet)
    if not seed_packets:
        raise RuntimeError("No valid controlled seed packets could be built.")

    prompt_counter += 1
    seed_prompt = prompts.seed_contrast_prompt(seed_packets, args.max_initial_rules)
    seed_raw = call_json(
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
    train_citation_predictions: list[dict[str, Any]] = []
    batch_index = 0
    for start in range(0, len(non_seed_train), args.batch_size):
        batch_index += 1
        batch = non_seed_train[start : start + args.batch_size]
        preds, prompt_counter = run_filter_batch(client, args, bank, batch, prompt_counter, update_stats=True)
        train_predictions.extend(preds)
        for pred in preds:
            append_jsonl(args.output_dir / "train_predictions.jsonl", strip_item_for_output(pred))
        write_json(args.output_dir / f"batch_{batch_index:03d}_metrics.json", summarize_predictions(preds))

        citation_preds, prompt_counter = run_generation_feedback_batch(client, args, bank, batch, prompt_counter)
        train_citation_predictions.extend(citation_preds)
        for pred in citation_preds:
            append_jsonl(args.output_dir / "train_citation_feedback_predictions.jsonl", strip_generation_for_output(pred))
        write_json(
            args.output_dir / f"batch_{batch_index:03d}_citation_metrics.json",
            summarize_generation_predictions(citation_preds),
        )

        packet = controller_packet(bank, preds, args, citation_preds)
        prompt_counter += 1
        controller_raw = call_json(
            client,
            args,
            prompts.LEARNER_SYSTEM_PROMPT,
            prompts.batch_controller_prompt(packet),
            "batch_controller",
            prompt_counter,
        )
        write_json(args.output_dir / f"controller_batch_{batch_index:03d}.json", controller_raw)
        bank = apply_controller_updates(bank, controller_raw)
        write_json(args.output_dir / f"experience_bank.v{bank['bank_version']:03d}.json", bank)
        write_json(args.output_dir / "experience_bank.latest.json", bank)

    test_predictions = []
    if not args.skip_final_test:
        test_predictions, prompt_counter = run_filter_batch(
            client, args, bank, test_items, prompt_counter, update_stats=False
        )
        for pred in test_predictions:
            append_jsonl(args.output_dir / "test_predictions.jsonl", strip_item_for_output(pred))

    args_for_summary = {
        key: ("<redacted>" if key == "api_key" and value else value)
        for key, value in vars(args).items()
    }
    args_for_summary.update(
        {
            "output_dir": str(args.output_dir),
            "train_jsonl": str(args.train_jsonl),
            "test_jsonl": str(args.test_jsonl),
        }
    )
    summary = {
        "args": args_for_summary,
        "seed_count": len(seed_packets),
        "final_bank_version": bank.get("bank_version"),
        "final_active_rules": len([r for r in bank.get("rules", []) if r.get("status", "active") == "active"]),
        "train_metrics": summarize_predictions(train_predictions),
        "train_citation_metrics": summarize_generation_predictions(train_citation_predictions),
        "test_metrics": None if args.skip_final_test else summarize_predictions(test_predictions),
        "final_test_skipped": bool(args.skip_final_test),
    }
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()



