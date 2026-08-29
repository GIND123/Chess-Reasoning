"""Process-verified GRPO.

Reward composition is a *config* choice, not a code change, so every arm of the
comparison and ablation programme is one YAML file:

    M6 (ours)   move + precision + coverage + format + penalty
    M3          move only                        (dense action-value RLVR)
    M4          move only, sparse=True           (binary outcome RLVR)
    M2          sparse + format                  (Master-Distillation-style)
    A1/A3       any subset, e.g. precision with no coverage (the degenerate arm)

Defaults follow current TRL: loss_type="dapo" (token-level loss + clip-higher) and
beta=0.0, which skips loading a reference model entirely and frees the memory for vLLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GRPOSettings:
    model: str = "runs/sft"                  # SFT checkpoint, or a base model id
    out_dir: str = "runs/grpo"
    tables: str = "data/final/tables.jsonl"
    train_jsonl: str = "data/final/rl_positions.jsonl"

    # which reward terms are active, and their weights
    terms: list[str] = field(default_factory=lambda:
                             ["move", "precision", "coverage", "format", "penalty"])
    weights: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.3, 0.2, 1.0])
    sparse_move: bool = False

    # optimisation
    steps: int = 400
    num_generations: int = 8                 # group size
    per_device_bs: int = 24
    grad_accum: int = 1
    lr: float = 1e-6
    max_completion_length: int = 400
    max_prompt_length: int = 768
    temperature: float = 1.0
    loss_type: str = "dapo"
    beta: float = 0.0                        # 0.0 -> no reference model
    epsilon: float = 0.2
    epsilon_high: float = 0.28               # clip-higher
    scale_rewards: str = "group"
    mask_truncated_completions: bool = True

    # vLLM
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.28

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    seed: int = 0
    # GRPO is tighter than SFT: vLLM colocate reserves part of the device and
    # each step holds num_generations completions per prompt.
    gradient_checkpointing: bool = True


def build_dataset(path: str, system: str):
    import json
    from datasets import Dataset
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "prompt": [{"role": "system", "content": system},
                           {"role": "user", "content": r["prompt"]}],
                "fen": r["fen"],           # reaches reward fns as kwargs["fen"]
            })
    return Dataset.from_list(rows)


def train(cfg: GRPOSettings):
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from chessr.engine import TableStore
    from chessr.prompts import SYS_STUDENT
    from chessr.reward import make_reward_fns

    import os
    store = TableStore()
    if os.path.isdir(cfg.tables):
        for f in sorted(os.listdir(cfg.tables)):
            if f.endswith(".jsonl"):
                store.load(os.path.join(cfg.tables, f))
    elif os.path.exists(cfg.tables):
        store.load(cfg.tables)
    if not len(store):
        raise SystemExit(f"no engine tables at {cfg.tables}; run scripts/01_engine_tables.py")

    all_fns = make_reward_fns(store, sparse=cfg.sparse_move)
    missing = [t for t in cfg.terms if t not in all_fns]
    if missing:
        raise SystemExit(f"unknown reward terms: {missing}")
    reward_funcs = [all_fns[t] for t in cfg.terms]
    weights = cfg.weights[:len(reward_funcs)]

    ds = build_dataset(cfg.train_jsonl, SYS_STUDENT)

    args = GRPOConfig(
        output_dir=cfg.out_dir,
        max_steps=cfg.steps,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.per_device_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        max_prompt_length=cfg.max_prompt_length,
        temperature=cfg.temperature,
        loss_type=cfg.loss_type,
        beta=cfg.beta,
        epsilon=cfg.epsilon,
        epsilon_high=cfg.epsilon_high,
        scale_rewards=cfg.scale_rewards,
        mask_truncated_completions=cfg.mask_truncated_completions,
        use_vllm=cfg.use_vllm,
        vllm_mode=cfg.vllm_mode,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        reward_weights=weights,
        bf16=True,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=5,
        save_steps=100,
        seed=cfg.seed,
        report_to="none",
    )
    peft_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                          target_modules="all-linear", task_type="CAUSAL_LM")

    trainer = GRPOTrainer(
        model=cfg.model,
        args=args,
        reward_funcs=reward_funcs,
        train_dataset=ds,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(cfg.out_dir)
    return trainer
