#!/usr/bin/env python3
"""Gate 2 (week 3): the extractor must agree with hand labels before it feeds a gradient.

Two modes:
  --audit    reproduce the corpus audit numbers against the legacy CSV (a regression test
             on the verifier itself: occupancy error should come out at ~8.3%)
  --sample   emit N claims to a TSV for hand-labelling
  --score    score a hand-labelled TSV and report extractor precision
"""
import argparse, csv, json, random, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
csv.field_size_limit(2**31 - 1)

from chessr.claims import verify_trace, ClaimVerdict


def rows(path, n, seed=0):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        all_rows = list(csv.DictReader(fh))
    random.seed(seed)
    return random.sample(all_rows, min(n, len(all_rows)))


def audit(path, n):
    per = collections.Counter(); hard = 0
    for r in rows(path, n):
        rep = verify_trace(r["FEN"], str(r["final_answer"]))
        hard += rep.has_hard_violation
        for c in rep.claims:
            per[(c.type.value, c.verdict.value)] += 1
    print(f"answers with a hard violation: {hard}/{n} = {hard/n:.1%}")
    for t in sorted({k[0] for k in per}):
        tr, fa = per.get((t, "true"), 0), per.get((t, "false"), 0)
        if tr + fa:
            print(f"  {t:<14} true={tr:<6} false={fa:<5} precision={tr/(tr+fa):.1%}")
    occ_t, occ_f = per.get(("occupancy", "true"), 0), per.get(("occupancy", "false"), 0)
    rate = occ_f / (occ_t + occ_f) if occ_t + occ_f else 0
    print(f"\noccupancy error rate {rate:.1%} (audit reference: 8.3%)")
    assert 0.06 < rate < 0.11, "verifier regression: occupancy rate moved off the audit value"


def sample(path, n, out):
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["fen", "claim_type", "claim_text", "auto_verdict", "human_verdict", "note"])
        for r in rows(path, n):
            rep = verify_trace(r["FEN"], str(r["final_answer"]))
            for c in rep.claims:
                if c.verdict in (ClaimVerdict.TRUE, ClaimVerdict.FALSE):
                    w.writerow([r["FEN"], c.type.value, c.text, c.verdict.value, "", c.detail])
    print(f"wrote {out} - fill in human_verdict (true/false/notaclaim) and re-run with --score")


def score(path):
    tot = agree = notaclaim = 0
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            h = (row.get("human_verdict") or "").strip().lower()
            if not h:
                continue
            tot += 1
            if h == "notaclaim":
                notaclaim += 1
            elif h == row["auto_verdict"]:
                agree += 1
    if not tot:
        raise SystemExit("no labels found")
    print(f"labelled {tot} | extraction precision {(tot-notaclaim)/tot:.1%} "
          f"| verdict agreement {agree/max(tot-notaclaim,1):.1%}")
    print("GATE 2:", "PASS" if agree / max(tot - notaclaim, 1) >= 0.90 else "FAIL (<90%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="10k_chunk_1.csv")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--sample", metavar="OUT")
    p.add_argument("--score", metavar="TSV")
    a = p.parse_args()
    if a.audit:  audit(a.csv, a.n)
    elif a.sample: sample(a.csv, 200, a.sample)
    elif a.score:  score(a.score)
    else: p.print_help()
