#!/usr/bin/env python3
"""Board figures from real model output: what the engine wants against what the model played.

Cases are selected by the verifier, not by hand, so the figure shows the behaviour the
method targets rather than a curated best case.
"""
import argparse, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chess
from chessr.boards import win_prob
from chessr.boardviz import BoardPanel, compose
from chessr.claims import ClaimType, ClaimVerdict, verify_structured_trace
from chessr.engine import TableStore
from chessr.prompts import parse_trace


def pick(records, store, n_each=2):
    agree, disagree, hallucinated = [], [], []
    for r in records:
        fen = r["fen"]
        tbl = store.get(fen)
        if not tbl:
            continue
        best = max(tbl, key=tbl.get)
        tr = parse_trace(r["completion"], fen)
        if not tr.move:
            continue
        rep = verify_structured_trace(fen, r["completion"], tbl)
        false_occ = [c for c in rep.claims
                     if c.type is ClaimType.OCCUPANCY and c.verdict is ClaimVerdict.FALSE]
        cp_loss = tbl[best] - tbl.get(tr.move, min(tbl.values()))
        rec = (fen, best, tr.move, cp_loss, rep, false_occ)
        if false_occ and len(hallucinated) < n_each:
            hallucinated.append(rec)
        elif tr.move == best and not false_occ and len(agree) < n_each:
            agree.append(rec)
        elif (tr.move != best and 150 < cp_loss < 3000
              and not false_occ and len(disagree) < n_each):
            disagree.append(rec)
        if len(agree) >= n_each and len(disagree) >= n_each and len(hallucinated) >= n_each:
            break
    return agree, disagree, hallucinated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="JSONL with fen + completion")
    ap.add_argument("--tables-glob", required=True)
    ap.add_argument("--out", default="figures/fig5_board_examples.png")
    a = ap.parse_args()

    store = TableStore()
    for f in sorted(glob.glob(a.tables_glob)):
        store.load(f)
    recs = [json.loads(l) for l in open(a.records) if l.strip()]

    agree, disagree, hall = pick(recs, store)
    panels = []

    for fen, best, mv, cp, rep, _ in agree[:1]:
        panels.append(BoardPanel(
            fen, best, mv, "Engine move found",
            f"model {mv}  ·  claim precision {rep.precision:.0%}\n"
            f"{rep.n_scored} verifiable claims, {rep.n_false} false"))

    for fen, best, mv, cp, rep, _ in disagree[:1]:
        tbl = store.get(fen)
        wl = win_prob(tbl[best]) - win_prob(tbl.get(mv, min(tbl.values())))
        # Mates are stored near +/-10000; printing that as a centipawn delta is nonsense.
        cost = "misses a forced mate" if tbl[best] >= 9000 else f"{cp:.0f} cp worse"
        panels.append(BoardPanel(
            fen, best, mv, "Weaker move chosen",
            f"engine {best}  ·  model {mv}\n"
            f"{cost}  ·  win probability −{wl:.2f}"))

    for fen, best, mv, cp, rep, bad in hall[:1]:
        claim = bad[0]
        panels.append(BoardPanel(
            fen, best, mv, "False claim caught",
            f"asserted \"{claim.text}\" — {claim.detail}\n"
            f"claim precision {rep.precision:.0%}, {rep.n_false} of {rep.n_scored} false"))

    if not panels:
        raise SystemExit("no cases found")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    print(compose(panels, a.out, cols=len(panels)))
    for p in panels:
        print(f"  {p.title}: {p.fen}")


if __name__ == "__main__":
    main()
