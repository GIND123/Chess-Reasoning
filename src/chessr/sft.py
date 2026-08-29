"""Supervised fine-tuning on verified traces.

The stored pair is always (student prompt -> trace). The student prompt contains no
engine data and no answer, so the model cannot learn to read the answer off its input --
which is the defect measured in the legacy corpus (100% of prompts leaked the target).
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class SFTSettings:
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    out_dir: str = "runs/sft"
    epochs: float = 2.0
    lr: float = 1e-5
    per_device_bs: int = 4
    grad_accum: int = 8
    max_len: int = 1280
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 0
    packing: bool = False


def load_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def build_dataset(records: list[dict], system: str):
    """Conversational format: TRL applies the chat template itself."""
    from datasets import Dataset
    rows = [{
        "prompt": [{"role": "system", "content": system},
                   {"role": "user", "content": r["prompt"]}],
        "completion": [{"role": "assistant", "content": r["completion"].strip()}],
        "fen": r["fen"],
    } for r in records]
    return Dataset.from_list(rows)


def train(records: list[dict], system: str, cfg: SFTSettings):
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    ds = build_dataset(records, system)
    peft_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=cfg.out_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.per_device_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        max_length=cfg.max_len,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        packing=cfg.packing,
        logging_steps=20,
        include_num_input_tokens_seen=True,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to="none",
        # Train on the completion only: the prompt is fixed scaffolding.
        completion_only_loss=True,
    )
    trainer = SFTTrainer(model=cfg.model, args=args, train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(cfg.out_dir)
    return trainer
