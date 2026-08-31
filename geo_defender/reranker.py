from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_INSTRUCTION = "Given a web search query, determine whether the document is relevant to the query."
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \\\"yes\\\" or \\\"no\\\"."
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@dataclass
class RankedDocument:
    index: int
    title: str
    text: str
    url: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "score": self.score,
        }


class ShieldReranker:
    """Qwen3 reranker with the released GEO Defender LoRA adapter."""

    def __init__(
        self,
        adapter_path: str | Path,
        base_model: str = "Qwen/Qwen3-Reranker-0.6B",
        max_length: int = 2048,
        load_in_4bit: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, padding_side="left", trust_remote_code=trust_remote_code
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=trust_remote_code,
        )
        self.model = PeftModel.from_pretrained(model, str(adapter_path))
        self.model.eval()
        self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")

    def score(self, query: str, document: str, instruction: str = DEFAULT_INSTRUCTION) -> float:
        user_text = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
        budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        encoded = self.tokenizer(
            user_text, add_special_tokens=False, truncation=True, max_length=budget
        )["input_ids"]
        input_ids = self.prefix_tokens + encoded + self.suffix_tokens
        batch = self.tokenizer.pad(
            {"input_ids": [input_ids]}, padding=True, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            logits = self.model(**batch).logits[:, -1, :]
        return float((logits[0, self.yes_token_id] - logits[0, self.no_token_id]).item())

    def rank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> list[RankedDocument]:
        ranked = []
        for original_index, document in enumerate(documents, start=1):
            ranked.append(
                RankedDocument(
                    index=original_index,
                    title=str(document.get("title", "")),
                    text=str(document.get("text", "")),
                    url=str(document.get("url", "")),
                    score=self.score(query, str(document.get("text", "")), instruction),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

