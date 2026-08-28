#!/usr/bin/env python3
"""Tier 1-3 evaluation with per-band reporting and bootstrap intervals.

Never pool across bands: the legacy distribution is 99.3% already-winning positions, so a
pooled number is dominated by the easiest regime.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessr.engine import TableStore
from chessr.evaluate import bootstrap_ci, evaluate, paired_bootstrap

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True, help="JSONL with fen + completion")
    p.add_argument("--tables", default="data/final/tables.jsonl")
    p.add_argument("--compare", help="second predictions file for a paired test")
    a = p.parse_args()

    store = TableStore(a.tables)
    recs = [json.loads(l) for l in open(a.preds) if l.strip()]
    m = evaluate(recs, store)

    print(f"n={m.n}")
    print(f"  top-1 agreement    {m.top1:.3f}")
    print(f"  win-prob loss      {m.wp_loss:.4f}")
    print(f"  illegal move rate  {m.illegal:.3f}")
    print(f"  claim precision    {m.claim_precision:.3f}")
    print(f"  claims / trace     {m.claims_per_trace:.1f}  (false {m.false_per_trace:.2f})")
    print(f"  tokens / trace     {m.tokens:.0f}")
    print("\nby decision band:")
    for band in ("near_tie", "moderate", "decisive", "tactical"):
        if band in m.by_band:
            b = m.by_band[band]
            print(f"  {band:<10} n={b['n']:<6} top1={b['top1']:.3f} "
                  f"wp_loss={b['wp_loss']:.4f} prec={b['prec']:.3f}")

    if a.compare:
        other = [json.loads(l) for l in open(a.compare) if l.strip()]
        by_fen = {r["fen"]: r for r in other}
        shared = [r for r in recs if r["fen"] in by_fen]
        m1 = evaluate(shared, store)
        m2 = evaluate([by_fen[r["fen"]] for r in shared], store)
        print(f"\npaired on {len(shared)} shared positions")
        print(f"  A top1={m1.top1:.3f}   B top1={m2.top1:.3f}   diff={m1.top1-m2.top1:+.3f}")
