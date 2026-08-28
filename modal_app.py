"""Govind-LLM — process-verified chess reasoning on Modal.

Everything runs inside one Modal app. Offline batched generation only: `LLM.generate()`
batches in-process, so there is no HTTP server, no client and no service discovery. Shards
are idempotent, so a crash costs one shard rather than the run. No region is pinned
anywhere — region selection applies a 1.5-1.75x rate multiplier.

    modal run modal_app.py::smoke                     # gate 1: generation works at all
    modal run modal_app.py::engine_tables             # Stockfish tables, CPU fan-out
    modal run modal_app.py::generate_all
    modal run modal_app.py::grpo --config configs/grpo_m6.yaml
"""
from __future__ import annotations

import modal

APP_NAME = "Govind-LLM"
app = modal.App(APP_NAME)

cache = modal.Volume.from_name("govind-llm-cache", create_if_missing=True)   # HF weights
data = modal.Volume.from_name("govind-llm-data", create_if_missing=True)     # fens, tables
shards = modal.Volume.from_name("govind-llm-shards", create_if_missing=True) # generations
runs = modal.Volume.from_name("govind-llm-runs", create_if_missing=True)     # checkpoints

VOLUMES = {"/cache": cache, "/data": data, "/shards": shards, "/runs": runs}

# CPU-only image: the verifier and the engine need nothing else.
CPU_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("stockfish")
    .pip_install("chess==1.11.2")
    .add_local_dir("src/chessr", remote_path="/root/chessr")
)

GPU_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "stockfish")
    .pip_install(
        "torch==2.6.0",
        "vllm==0.8.5.post1",
        "transformers>=4.51.0",
        "trl>=0.17.0",
        "peft>=0.15.0",
        "accelerate>=1.6.0",
        "datasets>=3.5.0",
        "chess==1.11.2",
        "pyyaml",
    )
    .env({"HF_HOME": "/cache/hf", "VLLM_USE_V1": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/chessr", remote_path="/root/chessr")
)


# --------------------------------------------------------------------------- #
# Stage 1 — engine tables (CPU)
# --------------------------------------------------------------------------- #

@app.function(image=CPU_IMAGE, cpu=4, volumes=VOLUMES, timeout=6 * 3600, retries=2,
              max_containers=60)
def engine_shard(shard_id: int, n_shards: int, nodes: int = 400_000,
                 limit: int | None = None) -> int:
    """Score every legal move of every position in this shard.

    Node limits, not time limits: a time limit makes the corpus depend on machine load.
    """
    import json
    import os
    import sys
    sys.path.insert(0, "/root")

    from chessr.engine import EngineConfig, engine_session, move_table

    out = f"/data/tables/tables_{shard_id:04d}.jsonl"
    os.makedirs("/data/tables", exist_ok=True)
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["fen"])
                except Exception:
                    pass

    fens = open("/data/fens.txt").read().split("\n")
    if limit:
        fens = fens[:limit]
    mine = [f for f in fens[shard_id::n_shards] if f.strip() and f not in done]
    if not mine:
        return 0

    cfg = EngineConfig(path="stockfish", nodes=nodes, threads=1, hash_mb=256)
    n = 0
    with engine_session(cfg) as eng, open(out, "a") as fh:
        for i, fen in enumerate(mine):
            try:
                tbl = move_table(eng, fen, cfg)
            except Exception:
                continue
            if tbl:
                fh.write(json.dumps({"fen": fen, "table": tbl}) + "\n")
                n += 1
            if n and n % 200 == 0:
                fh.flush()
                data.commit()
                print(f"[shard {shard_id}] {n}/{len(mine)}", flush=True)
    data.commit()
    print(f"[shard {shard_id}] done: {n}")
    return n


# --------------------------------------------------------------------------- #
# Stage 2 — teacher generation (GPU)
# --------------------------------------------------------------------------- #

@app.function(image=GPU_IMAGE, gpu="L40S", volumes=VOLUMES, timeout=6 * 3600, retries=2,
              max_containers=10)
def generate_shard(shard_id: int, n_shards: int, model: str, n: int = 1,
                   temperature: float = 0.7, limit: int | None = None,
                   max_tokens: int = 400) -> int:
    """Teacher traces for one shard. The teacher sees the engine table; the stored
    prompt is the student prompt, which contains neither the answer nor engine data."""
    import json
    import os
    import sys
    sys.path.insert(0, "/root")

    from chessr.engine import TableStore
    from chessr.generate import GenConfig, build_llm, chat_prompts, sampling_params
    from chessr.prompts import SYS_TEACHER, student_prompt, teacher_prompt

    tag = model.split("/")[-1].replace(".", "_")
    out = f"/shards/{tag}_n{n}_{shard_id:04d}.jsonl"
    os.makedirs("/shards", exist_ok=True)
    if os.path.exists(out):
        print(f"[skip] shard {shard_id}")
        return 0

    store = TableStore()
    for f in sorted(os.listdir("/data/tables")):
        store.load(f"/data/tables/{f}")
    print(f"[shard {shard_id}] {len(store)} engine tables loaded")

    fens = [f for f in store.fens() if f.strip()]
    if limit:
        fens = fens[:limit]
    mine = fens[shard_id::n_shards]
    if not mine:
        return 0

    cfg = GenConfig(model=model, n=n, temperature=temperature, max_tokens=max_tokens)
    if "AWQ" not in model and "GPTQ" not in model:
        cfg.quantization = None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    llm = build_llm(cfg)

    users = [teacher_prompt(f, store.get(f)) for f in mine]
    prompts = chat_prompts(tok, SYS_TEACHER, users)
    outs = llm.generate(prompts, sampling_params(cfg))

    tmp = out + ".tmp"
    written = 0
    with open(tmp, "w") as fh:
        for fen, o in zip(mine, outs):
            for cand in o.outputs:
                fh.write(json.dumps({
                    "fen": fen,
                    "prompt": student_prompt(fen),      # what the policy will ever see
                    "completion": cand.text,
                    "finish_reason": cand.finish_reason,
                    "n_tokens": len(cand.token_ids),
                }) + "\n")
                written += 1
    os.replace(tmp, out)     # atomic: a partial file is never mistaken for a done shard
    shards.commit()
    print(f"[shard {shard_id}] wrote {written}")
    return written


# --------------------------------------------------------------------------- #
# Stage 3 — training
# --------------------------------------------------------------------------- #

@app.function(image=GPU_IMAGE, gpu="H100", volumes=VOLUMES, timeout=10 * 3600, retries=1)
def sft(config_yaml: str, data_path: str = "/data/sft.jsonl") -> str:
    import sys
    import yaml
    sys.path.insert(0, "/root")
    from chessr.prompts import SYS_STUDENT
    from chessr.sft import SFTSettings, load_jsonl, train

    cfg = SFTSettings(**yaml.safe_load(config_yaml))
    recs = load_jsonl(data_path)
    print(f"{len(recs)} verified traces -> {cfg.out_dir}")
    train(recs, SYS_STUDENT, cfg)
    runs.commit()
    return cfg.out_dir


@app.function(image=GPU_IMAGE, gpu="H100", volumes=VOLUMES, timeout=10 * 3600, retries=1)
def grpo(config_yaml: str) -> str:
    """One arm. Reward composition comes entirely from the config, so M2/M3/M4/A1/A3
    differ from M6 by YAML alone."""
    import sys
    import yaml
    sys.path.insert(0, "/root")
    from chessr.grpo import GRPOSettings, train

    cfg = GRPOSettings(**yaml.safe_load(config_yaml))
    print(f"terms={cfg.terms} weights={cfg.weights} sparse={cfg.sparse_move}")
    train(cfg)
    runs.commit()
    return cfg.out_dir


# --------------------------------------------------------------------------- #
# Entrypoints
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def smoke(n: int = 200, model: str = "Qwen/Qwen3-14B-AWQ"):
    """Gate 1. Engine tables then generation for `n` positions, end to end, no HTTP."""
    print(f"[gate 1] engine tables for {n} positions ...")
    got = sum(engine_shard.starmap([(i, 8, 400_000, n) for i in range(8)]))
    print(f"[gate 1] {got} tables written")
    print("[gate 1] generating ...")
    w = generate_shard.remote(0, 1, model, 1, 0.7, n)
    print(f"[gate 1] PASS - {w} completions generated in one offline batch, no HTTP.")


@app.local_entrypoint()
def engine_tables(n_shards: int = 50, nodes: int = 400_000, limit: int = 0):
    total = sum(engine_shard.starmap(
        [(i, n_shards, nodes, limit or None) for i in range(n_shards)]))
    print(f"engine tables: {total} positions scored")


@app.local_entrypoint()
def generate_all(n_shards: int = 8, model: str = "Qwen/Qwen3-14B-AWQ", n: int = 1,
                 limit: int = 0, temperature: float = 0.7):
    total = sum(generate_shard.starmap(
        [(i, n_shards, model, n, temperature, limit or None) for i in range(n_shards)]))
    print(f"generated {total} completions across {n_shards} shards")
