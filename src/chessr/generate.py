"""Offline batched generation with vLLM.

No HTTP server, no client, no service discovery -- `LLM.generate()` batches a list of
prompts in-process. Every previous failure in this project came from assuming otherwise.

Throughput comes almost entirely from `max_model_len`. vLLM allocates KV blocks against
it, so leaving it at the model default starves concurrency. For Qwen3-14B-AWQ on a 24 GB
card (160 KB KV/token, ~13 GB free for KV):

    max_model_len=8192, kv=bf16 ->   9 concurrent sequences
    max_model_len=1280, kv=fp8  -> 124 concurrent sequences

A 14x difference from two flags. Our prompts are ~450 tokens and traces are capped at
400, so 1280 is ample.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Sequence


@dataclass
class GenConfig:
    model: str = "Qwen/Qwen3-14B-AWQ"
    quantization: str | None = "awq"
    dtype: str = "auto"
    # The dominant throughput lever -- but it caps *prompt + output*, not output alone.
    # Set at 1280 with the piece list and PV continuations in the prompt, vLLM silently
    # truncated 249/300 completions at ~122 tokens because only ~130 remained. Size it as
    # prompt + max_tokens with headroom, then check finish_reason.
    max_model_len: int = 2048
    # fp8 KV doubles concurrency and is the difference between 62 and 124 concurrent
    # sequences on a 24 GB card. On a 48 GB device there is already ample KV memory
    # (measured: 29.45 GiB free for KV with 14B-AWQ), so "auto" avoids the fp8 code
    # paths that route through FlashInfer.
    kv_cache_dtype: str = "auto"
    gpu_memory_utilization: float = 0.92
    enable_prefix_caching: bool = True
    tensor_parallel_size: int = 1

    # sampling
    n: int = 1
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 600
    seed: int = 0


def build_llm(cfg: GenConfig):
    from vllm import LLM
    kw = dict(
        model=cfg.model,
        dtype=cfg.dtype,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enable_prefix_caching=cfg.enable_prefix_caching,
        tensor_parallel_size=cfg.tensor_parallel_size,
        kv_cache_dtype=cfg.kv_cache_dtype,
    )
    if cfg.quantization:
        kw["quantization"] = cfg.quantization
    return LLM(**kw)


def sampling_params(cfg: GenConfig, *, legal_moves: Sequence[str] | None = None):
    """`n>1` shares prefill across the group, which matters for both rejection sampling
    and GRPO rollouts. `legal_moves` enables guided decoding of the <move> tag."""
    from vllm import SamplingParams
    kw = dict(n=cfg.n, temperature=cfg.temperature, top_p=cfg.top_p,
              max_tokens=cfg.max_tokens, seed=cfg.seed)
    if legal_moves:
        # vLLM 0.27 names this StructuredOutputsParams / `structured_outputs`. Do not wrap
        # this in a bare except: silently falling back to unconstrained decoding produced
        # a "constrained" evaluation in which none of the answers were constrained.
        from vllm.sampling_params import StructuredOutputsParams
        kw["structured_outputs"] = StructuredOutputsParams(choice=list(legal_moves))
    return SamplingParams(**kw)


def chat_prompts(tokenizer, system: str, users: Sequence[str],
                 enable_thinking: bool = False) -> list[str]:
    """`enable_thinking=False` matters: Qwen3 is a hybrid reasoning model with thinking on
    by default, and it will spend the whole token budget inside <think> without ever
    emitting the structured trace. Measured: 100/100 completions truncated at the cap."""
    kw = {}
    try:
        return [tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking) for u in users]
    except TypeError:
        return [tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True) for u in users]


def generate_shard(records: list[dict], cfg: GenConfig, system: str,
                   out_path: str, *, resume: bool = True) -> int:
    """`records` need `fen` and `prompt`. Writes JSONL; idempotent at shard granularity.

    A crash costs one shard, not the run.
    """
    if resume and os.path.exists(out_path):
        print(f"[skip] {out_path} exists")
        return 0

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model)
    llm = build_llm(cfg)
    prompts = chat_prompts(tok, system, [r["prompt"] for r in records])
    lens = [len(tok(p).input_ids) for p in prompts[:64]]
    print(f"[gen] prompt tokens: mean {sum(lens)/len(lens):.0f} max {max(lens)} | "
          f"max_model_len {cfg.max_model_len} | budget for output "
          f"{cfg.max_model_len - max(lens)}", flush=True)
    outs = llm.generate(prompts, sampling_params(cfg))

    tmp = out_path + ".tmp"
    n = 0
    with open(tmp, "w") as fh:
        for rec, o in zip(records, outs):
            for cand in o.outputs:
                fh.write(json.dumps({
                    "fen": rec["fen"],
                    "prompt": rec["prompt"],
                    "completion": cand.text,
                    "finish_reason": cand.finish_reason,
                    "n_tokens": len(cand.token_ids),
                    "gen": asdict(cfg),
                }) + "\n")
                n += 1
    os.replace(tmp, out_path)   # atomic: a partial file is never mistaken for a done shard
    return n
