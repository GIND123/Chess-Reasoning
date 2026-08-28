#!/usr/bin/env python3
"""Teacher generation. Runs locally on one 24 GB card, or on Modal via modal_app.py.

Pass 1 uses n=1 across everything; pass 2 re-runs only the rejects at n=4 and a higher
temperature. Cheaper than n=4 everywhere, and the acceptance curve is a reportable result.
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessr.datasets import load_set_a
from chessr.engine import TableStore
from chessr.generate import GenConfig, generate_shard
from chessr.prompts import SYS_TEACHER, teacher_prompt

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tables", default="data/final/tables.jsonl")
    p.add_argument("--out-dir", default="data/interim/gen")
    p.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--only-fens", help="JSONL of fens to redo (pass 2)")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    store = TableStore(a.tables)
    recs = load_set_a(limit=a.limit)

    if a.only_fens:
        want = {json.loads(l)["fen"] for l in open(a.only_fens) if l.strip()}
        recs = [r for r in recs if r["fen"] in want]

    recs = [r for r in recs if r["fen"] in store][a.shard::a.num_shards]
    for r in recs:
        r["prompt"] = teacher_prompt(r["fen"], store.get(r["fen"]))

    cfg = GenConfig(model=a.model, n=a.n, temperature=a.temperature)
    out = os.path.join(a.out_dir, f"gen_n{a.n}_{a.shard:04d}.jsonl")
    print("wrote", generate_shard(recs, cfg, SYS_TEACHER, out), "->", out)
