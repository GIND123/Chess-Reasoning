"""Evaluation harness: generate once, save everything, recompute metrics forever.

The design constraint is that a reviewer's question must be answerable without new GPU
time. So the harness writes an *immutable raw record* per prompt containing every
completion (not just the first), the decode settings, the gold data and the engine table
slice. Every metric in `metrics.py` is a pure function of those records, which means a
new metric, a regraded rule, or a different n for best-of-n costs nothing to produce.

Variants are generated in the same pass, because each needs its own forward pass and
going back for them later would mean re-running everything:

    base          the normal prompt
    perturbed     one piece relocated so a stated justification becomes false
    corrupted     one verified-true claim in the model's own trace flipped, then re-asked
    no_reasoning  answer-only, to establish what the reasoning is actually buying
    constrained   legal-move-constrained decoding, the control for illegal-move rate
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field

VARIANTS = ("base", "perturbed", "no_reasoning", "constrained")


@dataclass
class EvalRecord:
    """Everything needed to recompute any metric, offline, forever."""
    run_id: str
    model: str
    adapter: str | None
    variant: str
    benchmark: str
    item_id: str
    fen: str
    prompt: str
    system: str
    completions: list[str] = field(default_factory=list)
    n_tokens: list[int] = field(default_factory=list)
    finish_reason: list[str] = field(default_factory=list)
    gold_moves: list[str] = field(default_factory=list)
    solution: list[str] = field(default_factory=list)
    rating: int | None = None
    themes: list[str] = field(default_factory=list)
    engine_table: dict | None = None      # slice for this fen: recomputes any move metric
    meta: dict = field(default_factory=dict)
    decode: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


def run_id_for(model: str, adapter: str | None, tag: str) -> str:
    h = hashlib.sha1(f"{model}|{adapter}|{tag}".encode()).hexdigest()[:8]
    safe = model.split("/")[-1].replace(".", "_")
    return f"{safe}__{tag}__{h}"


SYS_NO_REASONING = ("You are a chess analyst. Given a position, reply with the best move "
                    "and nothing else, in the form <move>e2e4</move>. Do not explain.")


def build_variant_prompts(items, variant: str, seed: int = 0):
    """Return [(item, prompt, system, extra_meta)] for one variant."""
    from chessr.evaluate import perturb_position
    from chessr.prompts import SYS_STUDENT, student_prompt

    rng = random.Random(seed)
    out = []
    for it in items:
        if variant == "no_reasoning":
            out.append((it, student_prompt(it.fen), SYS_NO_REASONING, {}))
        elif variant == "perturbed":
            pf = perturb_position(it.fen, rng) if it.fen else None
            if not pf:
                continue
            out.append((it, student_prompt(pf), SYS_STUDENT, {"perturbed_fen": pf}))
        else:                                  # base, constrained
            out.append((it, student_prompt(it.fen), SYS_STUDENT, {}))
    return out


def run_eval(items, model: str, adapter: str | None, out_path: str,
             *, tables=None, variants=("base",), n_samples: int = 1,
             temperature: float = 0.0, max_tokens: int = 700, seed: int = 0,
             max_model_len: int = 2048, tag: str = "eval") -> str:
    """Generate every variant and write raw records. Resumable at file granularity."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    import chess

    rid = run_id_for(model, adapter, tag)
    if os.path.exists(out_path):
        print(f"[skip] {out_path} exists")
        return out_path

    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=0.90, enable_prefix_caching=True,
              enable_lora=bool(adapter), max_lora_rank=64)
    lora = LoRARequest("adapter", 1, adapter) if adapter else None

    tmp = out_path + ".tmp"
    written = 0
    with open(tmp, "w") as fh:
        for variant in variants:
            triples = build_variant_prompts(items, variant, seed)
            if not triples:
                continue
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": sysmsg},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for _, user, sysmsg, _ in triples]

            n = 1 if variant in ("no_reasoning", "constrained") else n_samples
            temp = 0.0 if n == 1 else max(temperature, 0.7)
            sp_kw = dict(n=n, temperature=temp, top_p=0.95,
                         max_tokens=200 if variant == "no_reasoning" else max_tokens,
                         seed=seed)
            outs = llm.generate(prompts, SamplingParams(**sp_kw),
                                lora_request=lora) if lora else \
                   llm.generate(prompts, SamplingParams(**sp_kw))

            for (it, user, sysmsg, extra), o in zip(triples, outs):
                fen = extra.get("perturbed_fen", it.fen)
                tbl = tables.get(fen) if tables else None
                if tbl is None and tables is not None and it.fen:
                    tbl = tables.get(it.fen)
                rec = EvalRecord(
                    run_id=rid, model=model, adapter=adapter, variant=variant,
                    benchmark=it.benchmark, item_id=it.id, fen=fen, prompt=user,
                    system=sysmsg,
                    completions=[c.text for c in o.outputs],
                    n_tokens=[len(c.token_ids) for c in o.outputs],
                    finish_reason=[c.finish_reason for c in o.outputs],
                    gold_moves=it.gold_moves, solution=it.solution,
                    rating=it.rating, themes=it.themes,
                    engine_table=tbl,
                    meta={**it.meta, **extra},
                    decode={**sp_kw, "max_model_len": max_model_len,
                            "variant": variant},
                )
                fh.write(json.dumps(asdict(rec)) + "\n")
                written += 1
            print(f"[eval] {variant}: {len(triples)} prompts", flush=True)

    os.replace(tmp, out_path)
    print(f"[eval] wrote {written} records -> {out_path}", flush=True)
    return out_path
