import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo_defender.data_utils import load_records, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_rows(rows: list[dict], train_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    grouped = list(rows)
    rng = random.Random(seed)
    rng.shuffle(grouped)
    cut = int(len(grouped) * train_ratio)
    return grouped[:cut], grouped[cut:]


def format_user_content(instruct: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {doc}"


class TripletDataset(Dataset):
    def __init__(self, rows: list[dict], instruct_key: str):
        self.rows = rows
        self.instruct_key = instruct_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        instruct = row[self.instruct_key]
        return {
            "query": row["query"],
            "instruct": instruct,
            "clean_doc": row["pos1_text"],
            "attack_doc": row["neg_text"],
            "benign_doc": row["pos2_text"],
            "ref_clean_score": row.get("ref_clean_score"),
            "ref_attack_score": row.get("ref_attack_score"),
            "ref_benign_score": row.get("ref_benign_score"),
            "source_row_index": row["source_row_index"],
        }


@dataclass
class Collator:
    tokenizer: Any
    prefix_tokens: list[int]
    suffix_tokens: list[int]
    max_length: int

    def _encode_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            add_special_tokens=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        input_ids = []
        for ids in inputs["input_ids"]:
            input_ids.append(self.prefix_tokens + ids + self.suffix_tokens)
        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return batch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        clean_texts = [format_user_content(item["instruct"], item["query"], item["clean_doc"]) for item in features]
        attack_texts = [format_user_content(item["instruct"], item["query"], item["attack_doc"]) for item in features]
        benign_texts = [format_user_content(item["instruct"], item["query"], item["benign_doc"]) for item in features]
        return {
            "clean_inputs": self._encode_texts(clean_texts),
            "attack_inputs": self._encode_texts(attack_texts),
            "benign_inputs": self._encode_texts(benign_texts),
            "ref_clean_score": torch.tensor(
                [float(item["ref_clean_score"]) if item["ref_clean_score"] is not None else float("nan") for item in features],
                dtype=torch.float32,
            ),
            "ref_attack_score": torch.tensor(
                [float(item["ref_attack_score"]) if item["ref_attack_score"] is not None else float("nan") for item in features],
                dtype=torch.float32,
            ),
            "ref_benign_score": torch.tensor(
                [float(item["ref_benign_score"]) if item["ref_benign_score"] is not None else float("nan") for item in features],
                dtype=torch.float32,
            ),
            "source_row_index": [item["source_row_index"] for item in features],
        }


def move_batch(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def compute_yes_scores(
    model: AutoModelForCausalLM,
    inputs: dict[str, torch.Tensor],
    true_token_id: int,
    false_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(**inputs)
    final_logits = outputs.logits[:, -1, :]
    true_logits = final_logits[:, true_token_id]
    false_logits = final_logits[:, false_token_id]
    margin = true_logits - false_logits
    pair_logits = torch.stack([false_logits, true_logits], dim=1)
    log_probs = torch.nn.functional.log_softmax(pair_logits, dim=1)
    return margin, log_probs[:, 1]


def dpo_pair_loss(win_residual: torch.Tensor, lose_residual: torch.Tensor, beta: float) -> torch.Tensor:
    return -torch.nn.functional.logsigmoid(beta * (win_residual - lose_residual)).mean()


def consistency_loss(clean_residual: torch.Tensor, benign_residual: torch.Tensor) -> torch.Tensor:
    return torch.square(benign_residual - clean_residual).mean()


@torch.no_grad()
def detached_yes_scores(
    model: AutoModelForCausalLM,
    inputs: dict[str, torch.Tensor],
    true_token_id: int,
    false_token_id: int,
) -> torch.Tensor:
    scores, _ = compute_yes_scores(model, inputs, true_token_id, false_token_id)
    return scores.detach()


def dpo_grad_coefficients(
    win_residual: torch.Tensor,
    lose_residual: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff = win_residual - lose_residual
    loss = -torch.nn.functional.logsigmoid(beta * diff).mean()
    scale = torch.sigmoid(-beta * diff) / max(1, diff.numel())
    win_coeff = -beta * scale
    lose_coeff = beta * scale
    return loss.detach(), win_coeff.detach(), lose_coeff.detach()


def consistency_grad_coefficients(
    clean_residual: torch.Tensor,
    benign_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff = benign_residual - clean_residual
    loss = torch.square(diff).mean()
    scale = 2.0 / max(1, diff.numel())
    clean_coeff = -scale * diff
    benign_coeff = scale * diff
    return loss.detach(), clean_coeff.detach(), benign_coeff.detach()


def backward_weighted_scores(
    model: AutoModelForCausalLM,
    inputs: dict[str, torch.Tensor],
    coeff: torch.Tensor,
    multiplier: float,
    scaler: torch.cuda.amp.GradScaler,
    true_token_id: int,
    false_token_id: int,
) -> None:
    with torch.cuda.amp.autocast(dtype=torch.float16):
        scores, _ = compute_yes_scores(model, inputs, true_token_id, false_token_id)
        objective = (scores * coeff.to(scores.device)).sum() * multiplier
    scaler.scale(objective).backward()
    del scores, objective


def evaluate(
    model: AutoModelForCausalLM,
    loader: DataLoader,
    device: torch.device,
    true_token_id: int,
    false_token_id: int,
    beta: float,
    lambdas: tuple[float, float, float],
    regularizer_mode: str,
    consistency_delta: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    def_ok = 0
    util_ok = 0
    reg_ok = 0
    normal_gap_total = 0.0
    with torch.no_grad():
        for batch in loader:
            clean_inputs = move_batch(batch["clean_inputs"], device)
            attack_inputs = move_batch(batch["attack_inputs"], device)
            benign_inputs = move_batch(batch["benign_inputs"], device)
            ref_clean = batch["ref_clean_score"].to(device)
            ref_attack = batch["ref_attack_score"].to(device)
            ref_benign = batch["ref_benign_score"].to(device)

            clean_scores, _ = compute_yes_scores(model, clean_inputs, true_token_id, false_token_id)
            attack_scores, _ = compute_yes_scores(model, attack_inputs, true_token_id, false_token_id)
            benign_scores, _ = compute_yes_scores(model, benign_inputs, true_token_id, false_token_id)

            res_clean = clean_scores - ref_clean
            res_attack = attack_scores - ref_attack
            res_benign = benign_scores - ref_benign

            loss_def = dpo_pair_loss(res_clean, res_attack, beta)
            loss_util = dpo_pair_loss(res_benign, res_attack, beta)
            if regularizer_mode == "consistency":
                loss_reg = consistency_loss(res_clean, res_benign)
            else:
                loss_reg = dpo_pair_loss(res_benign, res_clean, beta)
            loss = lambdas[0] * loss_def + lambdas[1] * loss_util + lambdas[2] * loss_reg

            bs = clean_scores.size(0)
            total_loss += float(loss.item()) * bs
            total += bs
            def_ok += int((res_clean > res_attack).sum().item())
            util_ok += int((res_benign > res_attack).sum().item())
            if regularizer_mode == "consistency":
                reg_ok += int((torch.abs(res_benign - res_clean) <= consistency_delta).sum().item())
                normal_gap_total += float(torch.abs(res_benign - res_clean).sum().item())
            else:
                reg_ok += int((res_benign > res_clean).sum().item())
                normal_gap_total += float(torch.abs(res_benign - res_clean).sum().item())

    if total == 0:
        return {
            "val_loss": 0.0,
            "def_acc": 0.0,
            "util_acc": 0.0,
            "reg_acc": 0.0,
            "avg_pair_acc": 0.0,
            "normal_gap_mean": 0.0,
        }
    metrics = {
        "val_loss": total_loss / total,
        "def_acc": def_ok / total,
        "util_acc": util_ok / total,
        "reg_acc": reg_ok / total,
        "normal_gap_mean": normal_gap_total / total,
    }
    metrics["avg_pair_acc"] = (metrics["def_acc"] + metrics["util_acc"] + metrics["reg_acc"]) / 3.0
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows = load_records(config["data_path"])
    train_rows, val_rows = split_rows(rows, train_ratio=float(config["train_ratio"]), seed=int(config["seed"]))

    trust_remote_code = bool(config.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token
    prefix_tokens = tokenizer.encode(config["prefix"], add_special_tokens=False)
    suffix_tokens = tokenizer.encode(config["suffix"], add_special_tokens=False)
    true_token_id = tokenizer.convert_tokens_to_ids(config["true_token"])
    false_token_id = tokenizer.convert_tokens_to_ids(config["false_token"])

    quant_config = BitsAndBytesConfig(
        load_in_4bit=bool(config["load_in_4bit"]),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    if bool(config.get("gradient_checkpointing", True)):
        base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config["target_modules"]),
    )
    model = get_peft_model(base_model, lora_config)

    train_dataset = TripletDataset(train_rows, instruct_key=config["instruct_key"])
    val_dataset = TripletDataset(val_rows, instruct_key=config["instruct_key"])
    collator = Collator(
        tokenizer=tokenizer,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        max_length=int(config["max_length"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["per_device_train_batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["per_device_eval_batch_size"]),
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    accumulation = int(config["gradient_accumulation_steps"])
    beta = float(config["beta"])
    lambdas = (float(config["lambda_def"]), float(config["lambda_util"]), float(config["lambda_reg"]))
    regularizer_mode = str(config.get("regularizer_mode", "ranking"))
    consistency_delta = float(config.get("consistency_delta", 0.15))
    max_steps = int(config["max_steps"])
    eval_steps = int(config["eval_steps"])
    save_steps = int(config["save_steps"])

    history = []
    best_metric = -1.0
    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        for batch in train_loader:
            clean_inputs = move_batch(batch["clean_inputs"], device)
            attack_inputs = move_batch(batch["attack_inputs"], device)
            benign_inputs = move_batch(batch["benign_inputs"], device)
            ref_clean = batch["ref_clean_score"].to(device)
            ref_attack = batch["ref_attack_score"].to(device)
            ref_benign = batch["ref_benign_score"].to(device)

            losses = []
            multiplier_def = lambdas[0] / accumulation
            multiplier_util = lambdas[1] / accumulation
            multiplier_reg = lambdas[2] / accumulation

            # Memory-efficient DPO: compute scalar gradient coefficients without
            # retaining two long-sequence graphs, then backprop each document
            # independently. This keeps the first-order DPO gradients while
            # avoiding simultaneous clean/attack activation storage.
            with torch.cuda.amp.autocast(dtype=torch.float16):
                clean_det = detached_yes_scores(model, clean_inputs, true_token_id, false_token_id)
                attack_det = detached_yes_scores(model, attack_inputs, true_token_id, false_token_id)
            loss_def, clean_coeff, attack_coeff = dpo_grad_coefficients(
                clean_det - ref_clean, attack_det - ref_attack, beta
            )
            backward_weighted_scores(model, clean_inputs, clean_coeff, multiplier_def, scaler, true_token_id, false_token_id)
            backward_weighted_scores(model, attack_inputs, attack_coeff, multiplier_def, scaler, true_token_id, false_token_id)
            losses.append(loss_def)
            del clean_det, attack_det, clean_coeff, attack_coeff

            with torch.cuda.amp.autocast(dtype=torch.float16):
                benign_det = detached_yes_scores(model, benign_inputs, true_token_id, false_token_id)
                attack_det = detached_yes_scores(model, attack_inputs, true_token_id, false_token_id)
            loss_util, benign_coeff, attack_coeff = dpo_grad_coefficients(
                benign_det - ref_benign, attack_det - ref_attack, beta
            )
            backward_weighted_scores(model, benign_inputs, benign_coeff, multiplier_util, scaler, true_token_id, false_token_id)
            backward_weighted_scores(model, attack_inputs, attack_coeff, multiplier_util, scaler, true_token_id, false_token_id)
            losses.append(loss_util)
            del benign_det, attack_det, benign_coeff, attack_coeff

            with torch.cuda.amp.autocast(dtype=torch.float16):
                benign_det = detached_yes_scores(model, benign_inputs, true_token_id, false_token_id)
                clean_det = detached_yes_scores(model, clean_inputs, true_token_id, false_token_id)
            if regularizer_mode == "consistency":
                loss_reg, clean_coeff, benign_coeff = consistency_grad_coefficients(
                    clean_det - ref_clean, benign_det - ref_benign
                )
            else:
                loss_reg, benign_coeff, clean_coeff = dpo_grad_coefficients(
                    benign_det - ref_benign, clean_det - ref_clean, beta
                )
            backward_weighted_scores(model, benign_inputs, benign_coeff, multiplier_reg, scaler, true_token_id, false_token_id)
            backward_weighted_scores(model, clean_inputs, clean_coeff, multiplier_reg, scaler, true_token_id, false_token_id)
            losses.append(loss_reg)
            del benign_det, clean_det, benign_coeff, clean_coeff

            if (global_step + 1) % accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % eval_steps == 0 or global_step == 1:
                metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    true_token_id=true_token_id,
                    false_token_id=false_token_id,
                    beta=beta,
                    lambdas=lambdas,
                    regularizer_mode=regularizer_mode,
                    consistency_delta=consistency_delta,
                )
                metrics["step"] = global_step
                history.append(metrics)
                if metrics["avg_pair_acc"] > best_metric:
                    best_metric = metrics["avg_pair_acc"]
                    model.save_pretrained(checkpoint_dir / "best")
                    tokenizer.save_pretrained(checkpoint_dir / "best")
                write_json(output_dir / "history.json", {"history": history, "best_metric": best_metric})

            if global_step % save_steps == 0:
                model.save_pretrained(checkpoint_dir / f"step_{global_step}")
                tokenizer.save_pretrained(checkpoint_dir / f"step_{global_step}")

            if global_step >= max_steps:
                break

    final_metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        true_token_id=true_token_id,
        false_token_id=false_token_id,
        beta=beta,
        lambdas=lambdas,
        regularizer_mode=regularizer_mode,
        consistency_delta=consistency_delta,
    )
    write_json(
        output_dir / "final_metrics.json",
        {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "best_metric": best_metric,
            "final_metrics": final_metrics,
            "config": config,
        },
    )
    write_markdown(
        output_dir / "report.md",
        "\n".join(
            [
                "# Qwen3-Reranker DPO",
                "",
                f"- model_name: `{config['model_name']}`",
                f"- instruct_key: `{config['instruct_key']}`",
                f"- train_size: `{len(train_dataset)}`",
                f"- val_size: `{len(val_dataset)}`",
                f"- regularizer_mode: `{regularizer_mode}`",
                f"- best_avg_pair_acc: `{best_metric:.6f}`",
                f"- final_val_loss: `{final_metrics['val_loss']:.6f}`",
                f"- final_def_acc: `{final_metrics['def_acc']:.6f}`",
                f"- final_util_acc: `{final_metrics['util_acc']:.6f}`",
                f"- final_reg_acc: `{final_metrics['reg_acc']:.6f}`",
                f"- final_avg_pair_acc: `{final_metrics['avg_pair_acc']:.6f}`",
                f"- final_normal_gap_mean: `{final_metrics['normal_gap_mean']:.6f}`",
                "",
            ]
        ),
    )


if __name__ == "__main__":
    main()

