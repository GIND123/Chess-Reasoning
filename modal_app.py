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

# Authenticated Hub access: unauthenticated shards hit rate limits when they fan out.
HF_SECRET = [modal.Secret.from_name("huggingface")]

HUB_REPO = "GOVINDFROM/chess-process-verified"


def _hub_push(local_path: str, remote_path: str) -> bool:
    """Push one finished shard to the Hub. Never fatal: a sync failure must not lose the
    shard that is already safely on the Volume."""
    import os
    token = os.environ.get("HF_TOKEN")
    if not token or not os.path.exists(local_path):
        return False
    try:
        from huggingface_hub import HfApi, create_repo
        create_repo(HUB_REPO, repo_type="dataset", private=True, exist_ok=True,
                    token=token)
        HfApi(token=token).upload_file(
            path_or_fileobj=local_path, path_in_repo=remote_path,
            repo_id=HUB_REPO, repo_type="dataset",
            commit_message=f"shard {remote_path}")
        print(f"[hub] pushed {remote_path}", flush=True)
        return True
    except Exception as e:                      # noqa: BLE001
        print(f"[hub] push failed for {remote_path}: {e}", flush=True)
        return False

# Stockfish is pinned to an exact released binary rather than an apt package: the
# corpus must be reproducible, and the paper commits to naming the engine version.
SF_URL = ("https://github.com/official-stockfish/Stockfish/releases/download/"
          "sf_17.1/stockfish-ubuntu-x86-64-avx2.tar")
SF_INSTALL = (
    f"curl -fsSL {SF_URL} -o /tmp/sf.tar "
    "&& tar -xf /tmp/sf.tar -C /tmp "
    "&& mv /tmp/stockfish/stockfish-ubuntu-x86-64-avx2 /usr/local/bin/stockfish "
    "&& chmod +x /usr/local/bin/stockfish "
    "&& rm -rf /tmp/sf.tar /tmp/stockfish "
    "&& stockfish compiler 2>/dev/null | head -2 || true"
)

# CPU-only image: the verifier and the engine need nothing else.
CPU_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "tar")
    .run_commands(SF_INSTALL)
    # benchmark loaders (Lichess, ChessQA, MATE) run on CPU too
    .pip_install("chess==1.11.2", "huggingface_hub>=0.28",
                 "datasets>=3.6.0", "pandas", "pyarrow")
    .add_local_dir("src/chessr", remote_path="/root/chessr")
)

# A CUDA *devel* base, not debian_slim: vLLM's FlashInfer path JIT-compiles kernels at
# engine start and needs nvcc. Setting VLLM_ATTENTION_BACKEND alone does not avoid it.
GPU_IMAGE = (
    # Python 3.12, not 3.11: the FlashInfer build that vLLM 0.27.1 pulls annotates with
    # `array.array[int]`, and array.array only became subscriptable in 3.12.
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "tar")
    .run_commands(SF_INSTALL)
    .pip_install(
        # TRL declares vllm<=0.27.1; 0.28.0 removed a symbol its GRPO trainer
        # imports (NCCLTrainerSendWeightsArgs), so "latest + latest" does not
        # work here. Pin to the pair TRL actually supports.
        "vllm==0.27.1",
        "trl==1.12.0",           # GRPOConfig(loss_type="dapo", vllm_mode="colocate")
        "peft>=0.17.0",
        "accelerate>=1.10.0",
        "transformers>=4.57.0",
        "datasets>=3.6.0",
        "chess==1.11.2",
        "huggingface_hub>=0.28",
        "pyyaml",
    )
    .env({
        "HF_HOME": "/cache/hf",
        "VLLM_USE_V1": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # FlashInfer JIT-compiles kernels and needs nvcc, which debian_slim has no
        # CUDA toolkit for. FlashAttention is prebuilt and needs no compiler.
        "CUDA_HOME": "/usr/local/cuda",
    })
    .add_local_dir("src/chessr", remote_path="/root/chessr")
)


def _push_dir(local_dir: str, remote_prefix: str) -> bool:
    """Push a finished checkpoint directory. LoRA adapters are small enough to keep."""
    import os
    token = os.environ.get("HF_TOKEN")
    if not token or not os.path.isdir(local_dir):
        return False
    try:
        from huggingface_hub import HfApi, create_repo
        create_repo(HUB_REPO, repo_type="dataset", private=True, exist_ok=True,
                    token=token)
        HfApi(token=token).upload_folder(
            folder_path=local_dir, path_in_repo=remote_prefix,
            repo_id=HUB_REPO, repo_type="dataset",
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.bin"],
            commit_message=f"checkpoint {remote_prefix}")
        print(f"[hub] pushed {remote_prefix}", flush=True)
        return True
    except Exception as e:                      # noqa: BLE001
        print(f"[hub] checkpoint push failed: {e}", flush=True)
        return False


# --------------------------------------------------------------------------- #
# Stage 1 — engine tables (CPU)
# --------------------------------------------------------------------------- #

@app.function(image=CPU_IMAGE, cpu=4, volumes=VOLUMES, secrets=HF_SECRET,
              timeout=6 * 3600, retries=2, max_containers=60)
def engine_shard(shard_id: int, n_shards: int, nodes: int = 400_000,
                 limit: int | None = None) -> int:
    """Score every legal move of every position in this shard.

    Node limits, not time limits: a time limit makes the corpus depend on machine load.
    """
    import json
    import os
    import sys
    sys.path.insert(0, "/root")

    from chessr.engine import EngineConfig, engine_session, move_table_with_pvs

    out = f"/data/tables_pv/tables_{shard_id:04d}.jsonl"
    os.makedirs("/data/tables_pv", exist_ok=True)
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
                tbl, pvs = move_table_with_pvs(eng, fen, cfg)
            except Exception:
                continue
            if tbl:
                fh.write(json.dumps({"fen": fen, "table": tbl, "pvs": pvs}) + "\n")
                n += 1
            if n and n % 200 == 0:
                fh.flush()
                data.commit()
                print(f"[shard {shard_id}] {n}/{len(mine)}", flush=True)
    data.commit()
    _hub_push(out, f"tables/tables_{shard_id:04d}.jsonl")
    print(f"[shard {shard_id}] done: {n}")
    return n


# --------------------------------------------------------------------------- #
# Stage 2 — teacher generation (GPU)
# --------------------------------------------------------------------------- #

@app.function(image=GPU_IMAGE, gpu="L40S", volumes=VOLUMES, secrets=HF_SECRET,
              timeout=6 * 3600, retries=2, max_containers=10)
def generate_shard(shard_id: int, n_shards: int, model: str, n: int = 1,
                   temperature: float = 0.7, limit: int | None = None,
                   max_tokens: int = 700) -> int:
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
    tdir = "/data/tables_pv" if os.path.isdir("/data/tables_pv") else "/data/tables"
    for f in sorted(os.listdir(tdir)):
        store.load(f"{tdir}/{f}")
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

    users = [teacher_prompt(f, store.get(f), store.pvs(f)) for f in mine]
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
    _hub_push(out, f"generations/{os.path.basename(out)}")
    print(f"[shard {shard_id}] wrote {written}")
    return written



# --------------------------------------------------------------------------- #
# Stage 2b — filtering (CPU; the data is already in the cloud)
# --------------------------------------------------------------------------- #

@app.function(image=CPU_IMAGE, cpu=8, memory=32768, volumes=VOLUMES,
              secrets=HF_SECRET, timeout=4 * 3600)
def filter_all(tol_wp: float = 0.10, min_precision: float = 0.90,
               graded: bool = False, out_name: str = "sft.jsonl") -> dict:
    """Apply the acceptance gates across every generation shard.

    Gates are the method, not a tuning knob: if acceptance is low the teacher changes,
    never these thresholds. The rejection breakdown is returned so that decision is made
    on evidence.
    """
    import collections
    import json
    import os
    import sys
    sys.path.insert(0, "/root")

    from chessr.engine import TableStore
    from chessr.filtering import Gates, judge
    from chessr.prompts import student_prompt

    store = TableStore()
    tdir = "/data/tables_pv" if os.path.isdir("/data/tables_pv") else "/data/tables"
    for f in sorted(os.listdir(tdir)):
        store.load(f"{tdir}/{f}")
    print(f"{len(store):,} engine tables", flush=True)

    gates = Gates(tol_wp=tol_wp, min_precision=min_precision, graded=graded)
    stats = collections.Counter()
    kept, seen = [], set()
    rejected_fens = []

    for fname in sorted(os.listdir("/shards")):
        if not fname.endswith(".jsonl"):
            continue
        with open(f"/shards/{fname}") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                fen = r["fen"]
                tbl = store.get(fen)
                if not tbl:
                    stats["no_table"] += 1
                    continue
                stats["total"] += 1
                d = judge(fen, r["completion"], tbl, gates)
                if d.accept and fen not in seen:
                    seen.add(fen)
                    stats["accepted"] += 1
                    kept.append({"fen": fen, "prompt": student_prompt(fen),
                                 "completion": r["completion"].strip(), "move": d.move,
                                 "wp_loss": d.wp_loss, "precision": d.precision,
                                 "n_root_claims": d.n_root_claims})
                elif not d.accept:
                    rejected_fens.append(fen)
                    for x in d.reasons:
                        stats[x.split(" ")[0] if " " in x else x] += 1
        print(f"  {fname}: kept {len(kept):,}", flush=True)

    out = f"/data/{out_name}"
    with open(out, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    with open("/data/rejected_fens.jsonl", "w") as fh:
        for f in dict.fromkeys(rejected_fens):
            fh.write(json.dumps({"fen": f}) + "\n")
    data.commit()
    _hub_push(out, f"final/{out_name}")

    total = stats["total"] or 1
    summary = {"total": stats["total"], "accepted": stats["accepted"],
               "unique_positions": len(kept),
               "acceptance": stats["accepted"] / total,
               "reasons": {k: v for k, v in stats.most_common()
                           if k not in ("total", "accepted")}}
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def filter_corpus(tol_wp: float = 0.10, min_precision: float = 0.90):
    s = filter_all.remote(tol_wp, min_precision)
    print(f"\nACCEPTED {s['accepted']:,}/{s['total']:,} = {s['acceptance']:.1%}")
    print(f"unique positions in the SFT corpus: {s['unique_positions']:,}")



@app.function(image=CPU_IMAGE, cpu=4, volumes=VOLUMES, secrets=HF_SECRET, timeout=3600)
def build_splits(n_rl: int = 20000, n_test: int = 6000, seed: int = 0) -> dict:
    """Banded RL and held-out splits.

    The source corpus is ~86% tactical, so a uniform sample would train and evaluate
    almost entirely in the regime where a single best move is obvious. We take every
    near-tie and moderate position available and cap the tactical band, which is the
    closest approach to a balanced mix this corpus supports. Results are still reported
    per band -- the mix improves the gradient, it does not license pooling.
    """
    import json, os, random, sys, collections
    sys.path.insert(0, "/root")
    from chessr.boards import band_for_gap
    from chessr.engine import TableStore, gap_cp
    from chessr.prompts import student_prompt

    store = TableStore()
    tdir = "/data/tables_pv" if os.path.isdir("/data/tables_pv") else "/data/tables"
    for f in sorted(os.listdir(tdir)):
        store.load(f"{tdir}/{f}")

    by_band = collections.defaultdict(list)
    for fen in store.fens():
        t = store.get(fen)
        if t and len(t) > 1:
            by_band[band_for_gap(gap_cp(t))].append(fen)
    print({k: len(v) for k, v in by_band.items()}, flush=True)

    rng = random.Random(seed)
    for v in by_band.values():
        rng.shuffle(v)

    # Hold out first, so nothing evaluated on is ever trained on.
    test, pool = [], collections.defaultdict(list)
    per_band_test = max(1, n_test // 4)
    for band, fens in by_band.items():
        take = min(per_band_test, len(fens) // 3)
        test += [(f, band) for f in fens[:take]]
        pool[band] = fens[take:]

    caps = {"near_tie": n_rl, "moderate": n_rl, "decisive": int(n_rl * 0.4),
            "tactical": int(n_rl * 0.35)}
    rl = []
    for band, fens in pool.items():
        rl += [(f, band) for f in fens[:caps.get(band, n_rl)]]
    rng.shuffle(rl)

    def dump(rows, path):
        with open(path, "w") as fh:
            for fen, band in rows:
                fh.write(json.dumps({"fen": fen, "prompt": student_prompt(fen),
                                     "band": band}) + "\n")
        return len(rows)

    n1 = dump(rl, "/data/rl_positions.jsonl")
    n2 = dump(test, "/data/test_positions.jsonl")
    data.commit()
    _hub_push("/data/rl_positions.jsonl", "final/rl_positions.jsonl")
    _hub_push("/data/test_positions.jsonl", "final/test_positions.jsonl")
    out = {"rl": n1, "test": n2,
           "rl_bands": dict(collections.Counter(b for _, b in rl)),
           "test_bands": dict(collections.Counter(b for _, b in test))}
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def run_sft(config: str = "configs/sft.yaml", data: str = "/data/sft.jsonl"):
    print(sft.remote(open(config).read(), data))


@app.local_entrypoint()
def run_grpo(config: str = "configs/grpo_m6.yaml", seed: int = 0):
    y = open(config).read()
    if seed:
        y += f"\nseed: {seed}\n"
    print(grpo.remote(y))


@app.local_entrypoint()
def splits(n_rl: int = 20000, n_test: int = 6000):
    print(build_splits.remote(n_rl, n_test))


# --------------------------------------------------------------------------- #
# Stage 3 — training
# --------------------------------------------------------------------------- #

@app.function(image=GPU_IMAGE, gpu="H100", volumes=VOLUMES, secrets=HF_SECRET,
              timeout=10 * 3600, retries=1)
def sft(config_yaml: str, data_path: str = "/data/sft.jsonl") -> str:
    import os
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
    _push_dir(cfg.out_dir, f"checkpoints/{os.path.basename(cfg.out_dir)}")
    return cfg.out_dir


@app.function(image=GPU_IMAGE, gpu="H100", volumes=VOLUMES, secrets=HF_SECRET,
              timeout=10 * 3600, retries=1)
def grpo(config_yaml: str) -> str:
    """One arm. Reward composition comes entirely from the config, so M2/M3/M4/A1/A3
    differ from M6 by YAML alone."""
    import os
    import sys
    import yaml
    sys.path.insert(0, "/root")
    from chessr.grpo import GRPOSettings, train

    cfg = GRPOSettings(**yaml.safe_load(config_yaml))
    print(f"terms={cfg.terms} weights={cfg.weights} sparse={cfg.sparse_move}")
    train(cfg)
    runs.commit()
    _push_dir(cfg.out_dir, f"checkpoints/{os.path.basename(cfg.out_dir)}")
    return cfg.out_dir




@app.function(image=GPU_IMAGE, gpu="L40S", volumes=VOLUMES, secrets=HF_SECRET,
              timeout=2 * 3600)
def merge_adapter(base: str = "Qwen/Qwen3-4B-Instruct-2507",
                  adapter: str = "/runs/sft", out_dir: str = "/runs/sft_merged") -> str:
    """Fold the SFT LoRA into the base weights.

    GRPO needs a model directory, not an adapter: every arm then starts from identical
    merged weights and applies its own fresh LoRA, so the only difference between arms is
    the reward. Keeping SFT as an adapter on top of the base would confound that.
    """
    import os
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if os.path.exists(os.path.join(out_dir, "config.json")):
        print(f"[skip] {out_dir} exists")
        return out_dir

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                                 device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    runs.commit()
    print("merged ->", out_dir, os.listdir(out_dir))
    return out_dir


@app.local_entrypoint()
def merge(base: str = "Qwen/Qwen3-4B-Instruct-2507", adapter: str = "/runs/sft"):
    print(merge_adapter.remote(base, adapter))



# --------------------------------------------------------------------------- #
# Stage 3b — freeze the evaluation set
# --------------------------------------------------------------------------- #

@app.function(image=CPU_IMAGE, cpu=4, memory=16384, volumes=VOLUMES,
              secrets=HF_SECRET, timeout=3600)
def build_eval_items(holdout_limit: int = 3615, lichess_per_band: int = 150,
                     n_per_theme: int = 30, chessqa_per_config: int = 100,
                     mate_n: int = 400, seed: int = 0) -> dict:
    """Write the evaluation item set once, to a file.

    Freezing it matters for the comparison: every system is then scored on byte-identical
    items, guaranteed by construction rather than by hoping a sampling seed reproduces.
    It also gives us the exact FEN list to compute engine tables for.
    """
    import json
    import sys
    from collections import Counter
    sys.path.insert(0, "/root")
    from chessr.benchmarks import load_all

    items = load_all({
        "holdout": "/data/test_positions.jsonl", "holdout_limit": holdout_limit,
        "lichess": True, "lichess_args": {"n_per_band": lichess_per_band,
                                          "n_per_theme": n_per_theme, "seed": seed},
        "chessqa": True, "chessqa_args": {"n_per_config": chessqa_per_config,
                                          "seed": seed},
        "mate": bool(mate_n), "mate_args": {"n": mate_n, "seed": seed},
    })
    with open("/data/eval_items.jsonl", "w") as fh:
        for it in items:
            fh.write(json.dumps(it.to_json()) + "\n")
    fens = sorted({i.fen for i in items if i.fen})
    with open("/data/eval_fens.txt", "w") as fh:
        fh.write("\n".join(fens))
    data.commit()
    _hub_push("/data/eval_items.jsonl", "final/eval_items.jsonl")
    out = {"items": len(items), "unique_fens": len(fens),
           "mix": dict(Counter(i.benchmark for i in items))}
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.function(image=CPU_IMAGE, cpu=4, volumes=VOLUMES, secrets=HF_SECRET,
              timeout=6 * 3600, retries=2, max_containers=40)
def eval_tables_shard(shard_id: int, n_shards: int, nodes: int = 400_000) -> int:
    """Engine tables for the frozen evaluation positions.

    External benchmarks bring positions our corpus never saw, so without this every
    engine-grounded metric (win-probability loss, top-3, decision band) is undefined on
    Lichess and ChessQA -- the smoke run resolved only 44 of 705 positions.
    """
    import json
    import os
    import sys
    sys.path.insert(0, "/root")
    from chessr.engine import EngineConfig, engine_session, move_table_with_pvs

    out = f"/data/eval_tables/tables_{shard_id:04d}.jsonl"
    os.makedirs("/data/eval_tables", exist_ok=True)
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["fen"])
                except Exception:
                    pass
    fens = [f for f in open("/data/eval_fens.txt").read().split("\n") if f.strip()]
    mine = [f for f in fens[shard_id::n_shards] if f not in done]
    if not mine:
        return 0
    cfg = EngineConfig(path="stockfish", nodes=nodes, threads=1, hash_mb=256)
    n = 0
    with engine_session(cfg) as eng, open(out, "a") as fh:
        for fen in mine:
            try:
                tbl, pvs = move_table_with_pvs(eng, fen, cfg)
            except Exception:
                continue
            if tbl:
                fh.write(json.dumps({"fen": fen, "table": tbl, "pvs": pvs}) + "\n")
                n += 1
    data.commit()
    _hub_push(out, f"eval_tables/tables_{shard_id:04d}.jsonl")
    print(f"[eval-tables {shard_id}] {n}")
    return n


@app.local_entrypoint()
def prepare_eval(n_shards: int = 40):
    info = build_eval_items.remote()
    print(info)
    total = sum(eval_tables_shard.starmap([(i, n_shards) for i in range(n_shards)]))
    print(f"engine tables for evaluation positions: {total}")


# --------------------------------------------------------------------------- #
# Stage 4 — evaluation (generate once, save everything)
# --------------------------------------------------------------------------- #

@app.function(image=GPU_IMAGE, gpu="L40S", volumes=VOLUMES, secrets=HF_SECRET,
              memory=32768, timeout=8 * 3600, retries=1)
def evaluate(model: str, adapter: str | None = None, tag: str = "eval",
             n_samples: int = 8, holdout_limit: int = 3615,
             lichess_per_band: int = 250, chessqa_per_config: int = 200,
             mate_n: int = 600, variants: str = "base,perturbed,no_reasoning") -> str:
    """One pass per model over every benchmark and variant, writing raw records.

    Raw records are the deliverable, not the numbers: any metric, any regrading, any n
    for best-of-n is recomputable from them without new GPU time.
    """
    import os
    import sys
    sys.path.insert(0, "/root")

    from chessr.engine import TableStore
    from chessr.evalsuite import run_eval, run_id_for

    tdir = "/data/tables_pv" if os.path.isdir("/data/tables_pv") else "/data/tables"

    # Load the benchmark items first so we can restrict the table store to the positions
    # they actually reference. Loading all ~150k tables next to a vLLM engine exhausts
    # container memory and the process is killed without a Python traceback.
    import json as _json
    from chessr.benchmarks import Item
    with open("/data/eval_items.jsonl") as fh:
        items = [Item(**_json.loads(l)) for l in fh if l.strip()]
    from collections import Counter
    print("benchmark mix:", dict(Counter(i.benchmark for i in items)), flush=True)

    wanted = {i.fen for i in items if i.fen}
    store = TableStore().load_dir(tdir, keep=wanted)
    if os.path.isdir("/data/eval_tables"):
        store.load_dir("/data/eval_tables", keep=wanted)
    print(f"{len(store):,} engine tables kept of {len(wanted):,} requested", flush=True)

    # gold moves for the holdout split come from the engine tables
    for it in items:
        if it.benchmark == "holdout" and not it.gold_moves:
            t = store.get(it.fen)
            if t:
                best = max(t.values())
                it.gold_moves = [u for u, cp in t.items() if cp == best]

    rid = run_id_for(model, adapter, tag)
    out = f"/runs/eval/{rid}.jsonl"
    os.makedirs("/runs/eval", exist_ok=True)
    try:
        run_eval(items, model, adapter, out, tables=store,
                 variants=tuple(variants.split(",")), n_samples=n_samples)
    except Exception:
        import traceback
        print("=== EVAL FAILED ===", flush=True)
        traceback.print_exc()
        raise
    runs.commit()
    _hub_push(out, f"eval/{rid}.jsonl")
    return out


@app.function(image=CPU_IMAGE, cpu=8, volumes=VOLUMES, secrets=HF_SECRET, timeout=3600)
def report(records_path: str) -> dict:
    """Metrics from saved records. CPU only -- rerun freely as questions come up."""
    import json
    import os
    import sys
    sys.path.insert(0, "/root")
    from chessr.metrics import full_report

    rep = full_report(records_path)
    out = records_path.replace(".jsonl", "_report.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    runs.commit()
    _hub_push(out, f"eval/{os.path.basename(out)}")
    print(json.dumps(rep["overall"], indent=2, default=str)[:2500], flush=True)
    return rep



@app.local_entrypoint()
def eval_smoke(model: str = "Qwen/Qwen3-4B-Instruct-2507", adapter: str = ""):
    """Exercise every benchmark loader and variant on a handful of items."""
    print(evaluate.remote(model, adapter or None, "smoke", 2, 20, 5, 5, 10,
                          "base,perturbed,no_reasoning"))


@app.function(image=GPU_IMAGE, gpu="L40S", volumes=VOLUMES, secrets=HF_SECRET,
              memory=32768, timeout=8 * 3600, retries=1)
def evaluate_constrained(model: str, adapter: str | None = None,
                         tag: str = "constrained", holdout_limit: int = 1500) -> str:
    import json as _json
    import os
    import sys
    sys.path.insert(0, "/root")
    from chessr.benchmarks import Item
    from chessr.engine import TableStore
    from chessr.evalsuite import run_constrained, run_id_for

    with open("/data/eval_items.jsonl") as fh:
        items = [Item(**_json.loads(l)) for l in fh if l.strip()]
    items = [i for i in items if i.fen and not i.question][:holdout_limit]
    wanted = {i.fen for i in items}
    tdir = "/data/tables_pv" if os.path.isdir("/data/tables_pv") else "/data/tables"
    store = TableStore().load_dir(tdir, keep=wanted)
    if os.path.isdir("/data/eval_tables"):
        store.load_dir("/data/eval_tables", keep=wanted)
    for it in items:
        if not it.gold_moves and (t := store.get(it.fen)):
            b = max(t.values())
            it.gold_moves = [u for u, cp in t.items() if cp == b]

    rid = run_id_for(model, adapter, tag)
    out = f"/runs/eval/{rid}.jsonl"
    os.makedirs("/runs/eval", exist_ok=True)
    run_constrained(items, model, adapter, out, tables=store)
    runs.commit()
    _hub_push(out, f"eval/{rid}.jsonl")
    return out


@app.local_entrypoint()
def eval_constrained(adapter: str = "/runs/grpo_m6v2", tag: str = "m6v2_constrained",
                     n: int = 1500):
    print(evaluate_constrained.remote("/runs/sft_merged", adapter or None, tag, n))


@app.local_entrypoint()
def eval_v2(n_samples: int = 4):
    """Evaluate the dense-reward arms on the same frozen item set as v1."""
    merged = "/runs/sft_merged"
    systems = [(merged, "/runs/grpo_m6v2", "m6v2"),
               (merged, "/runs/grpo_m3v2", "m3v2"),
               (merged, "/runs/grpo_a3v2", "a3v2")]
    args = [(m, a, t, n_samples, 3615, 150, 100, 400, "base,perturbed,no_reasoning")
            for m, a, t in systems]
    for p in evaluate.starmap(args):
        print("records:", p)


@app.local_entrypoint()
def eval_sweep(n_samples: int = 4, holdout_limit: int = 3615,
               lichess_per_band: int = 150, chessqa_per_config: int = 100,
               mate_n: int = 400):
    """Every system on identical items, in parallel.

    All arms share one base (/runs/sft_merged) and differ only by LoRA adapter, so the
    comparison is matched by construction. n_samples=4 still supports the reranking
    analysis at n in {1,2,4}; raw records make any larger n a re-run of metrics, not of
    the model.
    """
    base = "Qwen/Qwen3-4B-Instruct-2507"
    merged = "/runs/sft_merged"
    systems = [
        (base,   None,              "base_model"),
        (merged, None,              "sft"),
        (merged, "/runs/grpo_m6",   "m6_composite"),
        (merged, "/runs/grpo_m3",   "m3_move_only"),
        (merged, "/runs/grpo_m4",   "m4_sparse"),
        (merged, "/runs/grpo_a3",   "a3_no_coverage"),
    ]
    args = [(m, a, t, n_samples, holdout_limit, lichess_per_band,
             chessqa_per_config, mate_n, "base,perturbed,no_reasoning")
            for m, a, t in systems]
    paths = list(evaluate.starmap(args))
    print("records written:")
    for p in paths:
        print("  ", p)
    for p in paths:
        report.spawn(p)
    return paths


@app.local_entrypoint()
def run_eval_suite(model: str = "Qwen/Qwen3-4B-Instruct-2507",
                   adapter: str = "", tag: str = "base_model", n_samples: int = 8):
    path = evaluate.remote(model, adapter or None, tag, n_samples)
    print("records:", path)
    report.remote(path)



# --------------------------------------------------------------------------- #
# Stage 5 — playing strength
# --------------------------------------------------------------------------- #

@app.function(image=GPU_IMAGE, gpu="L40S", volumes=VOLUMES, secrets=HF_SECRET,
              memory=32768, timeout=8 * 3600, retries=1)
def play_matches(model: str, adapter: str | None, tag: str,
                 games: int = 120, skill: int = 0, nodes: int = 2000,
                 max_plies: int = 120) -> dict:
    """Full games against Stockfish at a fixed, weak node limit.

    Puzzle accuracy does not tell you whether a model can hold a position together over a
    game; only playing does. Stockfish is held at a low node budget so the match is
    informative rather than a foregone conclusion.
    """
    import json
    import os
    import sys
    sys.path.insert(0, "/root")
    import chess.engine
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    from chessr.play import GameLog, dump_logs, play_game, summarise

    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.90, enable_prefix_caching=True,
              enable_lora=bool(adapter), max_lora_rank=64)
    lora = LoRARequest("adapter", 1, adapter) if adapter else None

    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    eng.configure({"Threads": 1, "Hash": 64, "Skill Level": skill})
    limit = chess.engine.Limit(nodes=nodes)

    logs = []
    for g in range(games):
        logs.append(play_game(llm, tok, eng, limit, g, model_is_white=(g % 2 == 0),
                              lora=lora, max_plies=max_plies))
        if (g + 1) % 20 == 0:
            print(f"  {g + 1}/{games} games", flush=True)
    eng.quit()

    os.makedirs("/runs/play", exist_ok=True)
    path = f"/runs/play/{tag}.jsonl"
    dump_logs(logs, path)
    summary = summarise(logs)
    summary["tag"] = tag
    summary["opponent"] = f"stockfish skill={skill} nodes={nodes}"
    with open(f"/runs/play/{tag}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    runs.commit()
    _hub_push(path, f"play/{tag}.jsonl")
    _hub_push(f"/runs/play/{tag}_summary.json", f"play/{tag}_summary.json")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def play(games: int = 120, skill: int = 0, nodes: int = 2000):
    systems = [("Qwen/Qwen3-4B-Instruct-2507", None, "base"),
               ("/runs/sft_merged", None, "sft"),
               ("/runs/sft_merged", "/runs/grpo_m6v2", "m6v2")]
    for r in play_matches.starmap([(m, a, t, games, skill, nodes) for m, a, t in systems]):
        print(f"{r['tag']:<8} score={r['score']:.3f} elo={r['elo_diff']:+.0f} "
              f"[{r['elo_lo']:+.0f},{r['elo_hi']:+.0f}] blunders/100={r['blunders_per_100']:.1f} "
              f"illegal_fallback={r['illegal_fallback_rate']:.3f}")


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
                 limit: int = 0, temperature: float = 0.7, max_tokens: int = 700):
    total = sum(generate_shard.starmap(
        [(i, n_shards, model, n, temperature, limit or None, max_tokens)
         for i in range(n_shards)]))
    print(f"generated {total} completions across {n_shards} shards")
