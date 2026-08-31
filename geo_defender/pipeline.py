from __future__ import annotations

from pathlib import Path
from typing import Any

from .reranker import ShieldReranker
from .tfsg.inference import generate_answer


class GEODefender:
    """End-to-end Shield Reranker + TFSG inference pipeline."""

    def __init__(
        self,
        *,
        adapter_path: str | Path,
        experience_library: str | Path,
        generator_model: str,
        api_key: str,
        base_url: str | None = None,
        base_reranker: str = "Qwen/Qwen3-Reranker-0.6B",
        top_k: int = 5,
        load_in_4bit: bool = True,
    ) -> None:
        self.reranker = ShieldReranker(
            adapter_path=adapter_path,
            base_model=base_reranker,
            load_in_4bit=load_in_4bit,
        )
        self.experience_library = Path(experience_library)
        self.generator_model = generator_model
        self.api_key = api_key
        self.base_url = base_url
        self.top_k = top_k

    def defend(self, query: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        selected = self.reranker.rank(query, documents, top_k=self.top_k)
        evidence = [item.as_dict() for item in selected]
        generation = generate_answer(
            query=query,
            documents=evidence,
            library=self.experience_library,
            model=self.generator_model,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        return {
            "query": query,
            "selected_evidence": evidence,
            "answer": generation["answer"],
            "raw_response": generation["raw_response"],
        }
