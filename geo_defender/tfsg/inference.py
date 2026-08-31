from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .prompts import FILTER_SYSTEM_PROMPT, citation_feedback_generation_prompt


# The construction-time reasoning mode and final defense execution use the
# same answer-only prompt. The reasoning mode additionally reads the provider's
# reasoning_content; final inference ignores it.
SYSTEM_PROMPT = FILTER_SYSTEM_PROMPT
ANSWER_ONLY_SYSTEM_PROMPT = FILTER_SYSTEM_PROMPT


def build_generation_prompt(
    query: str, documents: list[dict[str, Any]], library: dict[str, Any]
) -> str:
    item = {"query": query, "top5_docs": documents}
    return citation_feedback_generation_prompt(item, library, excerpt_chars=0)


def experience_guided_prompt(
    item: dict[str, Any], bank: dict[str, Any], excerpt_chars: int
) -> str:
    return citation_feedback_generation_prompt(item, bank, excerpt_chars)


def parse_json_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        result = json.loads(text[start : end + 1])
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
        raise ValueError("The target LLM did not return the required answer object.")
    return result


def generate_answer(
    *,
    query: str,
    documents: list[dict[str, Any]],
    library: dict[str, Any] | str | Path,
    model: str,
    api_key: str,
    base_url: str | None = None,
    timeout: float = 240.0,
    retries: int = 3,
) -> dict[str, Any]:
    if not isinstance(library, dict):
        library = json.loads(Path(library).read_text(encoding="utf-8"))

    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    prompt = build_generation_prompt(query, documents, library)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            result = parse_json_response(raw)
            result["raw_response"] = raw
            return result
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Generation failed after {retries} attempts") from last_error
