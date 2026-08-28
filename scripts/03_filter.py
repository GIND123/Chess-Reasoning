#!/usr/bin/env python3
"""Apply acceptance gates, write the SFT corpus, and report the rejection breakdown.

The acceptance rate is the week-5 gate: below ~20%, change the teacher, never the filter.
"""
import argparse, glob, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessr.engine import TableStore
from chessr.filtering import Gates, filter_stream
from chessr.prompts import student_prompt

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gen-glob", default="data/interim/gen/*.jsonl")
    p.add_argument("--tables", default="data/final/tables.jsonl")
    p.add_argument("--out", default="data/final/sft.jsonl")
    p.add_argument("--rejects", default="data/interim/rejects.jsonl")
    p.add_argument("--graded", action="store_true", help="Set B: keep with a graded label")
    p.add_argument("--tol-wp", type=float, default=0.10)
    a = p.parse_args()

    store = TableStore(a.tables)
    recs = []
    for f in sorted(glob.glob(a.gen_glob)):
        with open(f) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    print(f"{len(recs)} generations from {len(glob.glob(a.gen_glob))} shards")

    gates = Gates(graded=a.graded, tol_wp=a.tol_wp)
    stream = filter_stream(recs, store, gates)
    kept, seen = [], set()
    for r in stream:
        if r["fen"] in seen:      # one trace per position
            continue
        seen.add(r["fen"])
        kept.append({"fen": r["fen"], "prompt": student_prompt(r["fen"]),
                     "completion": r["completion"], "move": r["move"],
                     "wp_loss": r["wp_loss"], "precision": r["precision"]})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")

    accepted_fens = {r["fen"] for r in kept}
    with open(a.rejects, "w") as fh:
        for f in {r["fen"] for r in recs} - accepted_fens:
            fh.write(json.dumps({"fen": f}) + "\n")

    st = stream.stats
    total = st["total"] or 1
    print(f"\naccepted {len(kept)} unique positions ({st['accepted']}/{total} = "
          f"{st['accepted']/total:.1%} of generations)")
    print("rejection reasons:")
    for k, v in st.most_common():
        if k not in ("total", "accepted"):
            print(f"  {k:<24} {v:>7} ({v/total:.1%})")
    rate = st["accepted"] / total
    print(f"\nGATE (week 5): {'PASS' if rate >= 0.20 else 'FAIL'} - acceptance {rate:.1%}")
    if rate < 0.20:
        print("  -> change the teacher, not the filter.")
