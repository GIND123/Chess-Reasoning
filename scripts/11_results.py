#!/usr/bin/env python3
"""Compute every reported number from saved evaluation records and write results.json."""
import argparse, glob, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessr.metrics import (aggregate, faithfulness, holm_bonferroni, load_records,
                            paired_test, score_record)

NAMES = {"sft": "SFT", "m6_composite": "M6 composite", "m3_move_only": "M3 move-only",
         "m4_sparse": "M4 sparse", "a3_no_coverage": "A3 no-coverage",
         "base_model": "Base"}
HEAD = ["top1_engine", "top3_engine", "wp_loss", "illegal", "no_move",
        "claim_precision", "n_claims", "n_false", "hard_violation",
        "well_formed", "tokens"]


def tag_of(path):
    """Records are named {model}__{tag}__{hash}. Substring matching is wrong here:
    every arm's file starts with `sft_merged__`, so a naive scan labels them all SFT."""
    parts = Path(path).stem.split("__")
    return parts[1] if len(parts) >= 3 else Path(path).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/eval/*.jsonl")
    ap.add_argument("--out", default="data/final/results.json")
    ap.add_argument("--rerank-n", type=int, default=4)
    a = ap.parse_args()

    systems, rows_by = {}, {}
    for path in sorted(glob.glob(a.glob)):
        tag = tag_of(path)
        if tag == "smoke":
            continue
        recs = load_records(path)
        base = [r for r in recs if r["variant"] == "base"]
        rows = [score_record(r, rerank_n=a.rerank_n) for r in base]
        rows_by[tag] = rows
        systems[tag] = {
            "name": NAMES.get(tag, tag),
            "n_records": len(recs),
            "overall": aggregate(rows, HEAD),
            "by_band": aggregate(rows, HEAD, "gap_band", ci=False),
            "by_benchmark": aggregate(rows, HEAD, "benchmark", ci=False),
            "by_rating": aggregate(rows, HEAD, "rating_bucket", ci=False),
            "faithfulness": faithfulness(recs),
        }
        # rerank columns only appear on rows that had an engine table, so scan all rows
        # and keep the numeric ones (rerankN_move is a UCI string, not a metric).
        rr = sorted({k for r in rows for k in r
                     if (k.startswith("rerank") or k.startswith("vote")
                         or k.startswith("oracle")) and not k.endswith("_move")})
        if rr:
            systems[tag]["rerank"] = aggregate(rows, rr, ci=False)
        print(f"{NAMES.get(tag,tag):<18} n={len(rows):<6} "
              f"top1={systems[tag]['overall']['top1_engine']['mean']:.4f} "
              f"prec={systems[tag]['overall']['claim_precision']['mean']:.4f}")

    # paired comparisons against the SFT initialisation and between arms
    comps, pvals = {}, {}
    pairs = [("m6_composite", "sft"), ("m3_move_only", "sft"), ("m4_sparse", "sft"),
             ("a3_no_coverage", "sft"), ("m6_composite", "m3_move_only"),
             ("m6_composite", "a3_no_coverage"), ("m3_move_only", "m4_sparse")]
    for x, y in pairs:
        if x not in rows_by or y not in rows_by:
            continue
        for key in ("top1_engine", "wp_loss", "claim_precision", "n_false", "illegal"):
            t = paired_test(rows_by[x], rows_by[y], key)
            if t.get("n"):
                name = f"{x} vs {y} :: {key}"
                comps[name] = t
                if key in ("top1_engine", "claim_precision"):
                    pvals[name] = t["p"]
    corrected = holm_bonferroni(pvals) if pvals else {}

    out = {"systems": systems, "comparisons": comps, "holm_bonferroni": corrected}
    os.makedirs(Path(a.out).parent, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
