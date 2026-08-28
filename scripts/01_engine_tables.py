#!/usr/bin/env python3
"""Score every legal move in every position, once, offline.

Node limits (not time limits) so the corpus is reproducible across machines. Parallelise
over positions with --workers; each worker holds one long-lived engine process.
"""
import argparse, json, os, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessr.datasets import load_set_a
from chessr.engine import EngineConfig, build_tables


def worker(args):
    shard, fens, cfg, out_dir = args
    return build_tables(fens, cfg, os.path.join(out_dir, f"tables_{shard:03d}.jsonl"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="GRPO_GM_dataset.csv")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--nodes", type=int, default=400_000)
    p.add_argument("--out-dir", default="data/interim/tables")
    p.add_argument("--merge-to", default="data/final/tables.jsonl")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.merge_to), exist_ok=True)
    fens = [r["fen"] for r in load_set_a(a.csv, a.limit)]
    print(f"{len(fens)} positions, {a.workers} workers, {a.nodes} nodes/position")

    cfg = EngineConfig(nodes=a.nodes)
    jobs = [(i, fens[i::a.workers], cfg, a.out_dir) for i in range(a.workers)]
    with Pool(a.workers) as pool:
        print("wrote", sum(pool.map(worker, jobs)), "tables")

    with open(a.merge_to, "w") as out:
        for f in sorted(Path(a.out_dir).glob("tables_*.jsonl")):
            out.write(f.read_text())
    print("merged ->", a.merge_to)
