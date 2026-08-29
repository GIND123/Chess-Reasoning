#!/usr/bin/env python3
"""Compute every number the figures are built from, into data/final/metrics.json.

Nothing here is hand-entered except published baselines, which carry their source.
Re-run after new results land and the figures regenerate from it.
"""
import argparse, csv, glob, json, random, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
csv.field_size_limit(2**31 - 1)

import chess
from chessr.boards import band_for_gap
from chessr.claims import ClaimVerdict, verify_trace
from chessr.engine import TableStore, gap_cp


def position_stats(store: TableStore) -> dict:
    evals, bands, mates = [], collections.Counter(), 0
    for fen in store.fens():
        t = store.get(fen)
        if not t:
            continue
        best = max(t.values())
        evals.append(best)
        if best >= 9000:
            mates += 1
        bands[band_for_gap(gap_cp(t))] += 1
    lo, hi, nb = -400, 1200, 32
    step = (hi - lo) / nb
    edges = [lo + i * step for i in range(nb + 1)]
    counts = [0] * nb
    for v in evals:
        i = int((min(max(v, lo), hi - 1e-9) - lo) / step)
        counts[min(max(i, 0), nb - 1)] += 1
    ev = sorted(evals)
    return {
        "n": len(evals),
        "eval_hist": {"edges": edges, "counts": counts},
        "bands": dict(bands),
        "median_cp": ev[len(ev) // 2] if ev else 0,
        "pct_winning": 100 * sum(v > 0 for v in evals) / max(len(evals), 1),
        "pct_mate": 100 * mates / max(len(evals), 1),
    }


def claim_precision(csv_path: str, n: int = 3000, seed: int = 0) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    random.seed(seed)
    per = collections.Counter()
    for r in random.sample(rows, min(n, len(rows))):
        for c in verify_trace(r["FEN"], str(r["final_answer"])).claims:
            if c.verdict is ClaimVerdict.TRUE:
                per[(c.type.value, "t")] += 1
            elif c.verdict is ClaimVerdict.FALSE:
                per[(c.type.value, "f")] += 1
    out = []
    for t in sorted({k[0] for k in per}):
        tr, fa = per.get((t, "t"), 0), per.get((t, "f"), 0)
        if tr + fa >= 5:
            out.append({"type": t, "n": tr + fa, "precision": tr / (tr + fa)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables-glob", default="data/interim/tables_pv/*.jsonl")
    p.add_argument("--legacy-csv", default="10k_chunk_1.csv")
    p.add_argument("--acceptance", default="data/final/acceptance.json")
    p.add_argument("--out", default="data/final/metrics.json")
    a = p.parse_args()

    m = {}
    files = sorted(glob.glob(a.tables_glob))
    if files:
        store = TableStore()
        for f in files:
            store.load(f)
        m["position_stats"] = position_stats(store)
        print(f"position_stats over {len(store):,} positions from {len(files)} shards")

    if Path(a.legacy_csv).exists():
        m["claim_precision"] = claim_precision(a.legacy_csv)
        print("claim_precision:", [(r["type"], round(r["precision"], 3))
                                   for r in m["claim_precision"]])

    if Path(a.acceptance).exists():
        m["acceptance"] = json.loads(Path(a.acceptance).read_text())

    # Published baselines, each with its source; ours is filled from the run.
    m["token_efficiency"] = [
        {"name": "GPT-5", "tokens": 12193, "src": "Tang et al. 2026"},
        {"name": "DeepSeek-V3.1", "tokens": 11249, "src": "Tang et al. 2026"},
        {"name": "Gemini-3-Flash", "tokens": 6418, "src": "Tang et al. 2026"},
        {"name": "C1-4B", "tokens": 178, "src": "Tang et al. 2026"},
    ]
    if "ours_mean_tokens" in m.get("acceptance", {}):
        m["token_efficiency"].append({"name": "Ours (teacher)",
                                      "tokens": int(m["acceptance"]["ours_mean_tokens"]),
                                      "ours": True, "src": "this run"})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(m, indent=2, sort_keys=True))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
