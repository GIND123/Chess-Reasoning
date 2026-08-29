"""Every metric, computed from saved EvalRecords. No model, no GPU, no engine calls.

This is the answer to "a reviewer asked for X": add a function here and re-run over the
existing records. Nothing in this module regenerates text.

Grouping is available on every metric (band, rating bucket, theme, benchmark), because
pooling across decision difficulty is exactly how a tactics-heavy test set flatters a
model -- and our source distribution is ~86% tactical.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict

import chess

from chessr.boards import band_for_gap, win_prob
from chessr.claims import ClaimType, ClaimVerdict, verify_structured_trace
from chessr.engine import gap_cp
from chessr.prompts import parse_trace
from chessr.rerank import rerank

RATING_BUCKETS = ((0, 1200), "<1200"), ((1200, 1600), "1200-1600"), \
                 ((1600, 2000), "1600-2000"), ((2000, 2400), "2000-2400"), \
                 ((2400, 9999), "2400+")


def load_records(path: str, variant: str | None = None) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if variant is None or r["variant"] == variant:
                out.append(r)
    return out


def _rating_bucket(r):
    if r is None:
        return "unrated"
    for (lo, hi), name in RATING_BUCKETS:
        if lo <= r < hi:
            return name
    return "unrated"


def _first_move(rec, idx=0):
    if idx >= len(rec["completions"]):
        return None
    return parse_trace(rec["completions"][idx], rec["fen"]).move


# --------------------------------------------------------------------------- #
# Per-record scoring -- one row per record, then aggregate however you like
# --------------------------------------------------------------------------- #

def score_record(rec: dict, *, rerank_n: int | None = None) -> dict:
    """All per-position quantities. Aggregation and grouping happen afterwards."""
    from chessr.boards import is_fen
    fen, tbl = rec["fen"], rec.get("engine_table")
    if not is_fen(fen):
        fen = ""
    board = chess.Board(fen) if fen else None
    row = {
        "item_id": rec["item_id"], "benchmark": rec["benchmark"],
        "variant": rec["variant"], "band": rec.get("meta", {}).get("band"),
        "rating": rec.get("rating"), "rating_bucket": _rating_bucket(rec.get("rating")),
        "themes": rec.get("themes") or [],
        "tokens": rec["n_tokens"][0] if rec["n_tokens"] else 0,
        "truncated": float(bool(rec["finish_reason"]) and rec["finish_reason"][0] == "length"),
    }

    move = _first_move(rec) if fen else None
    row["move"] = move
    row["no_move"] = float(move is None)
    legal = bool(move and board and chess.Move.from_uci(move) in board.legal_moves)
    row["illegal"] = float(move is not None and not legal)

    gold = set(rec.get("gold_moves") or [])
    row["top1"] = float(bool(move) and move in gold) if gold else float("nan")

    if tbl:
        best = max(tbl.values())
        ranked = sorted(tbl, key=lambda u: -tbl[u])
        row["top1_engine"] = float(move in ranked[:1]) if move else 0.0
        row["top3_engine"] = float(move in ranked[:3]) if move else 0.0
        row["wp_loss"] = (max(0.0, win_prob(best) - win_prob(tbl[move]))
                          if move in tbl else 1.0)
        row["gap_band"] = band_for_gap(gap_cp(tbl))
        row["cp_loss"] = float(best - tbl[move]) if move in tbl else float("nan")
    else:
        row.update(top1_engine=float("nan"), top3_engine=float("nan"),
                   wp_loss=float("nan"), gap_band=None, cp_loss=float("nan"))

    # full-line accuracy, for puzzle sets that ship a solution
    sol = rec.get("solution") or []
    row["solution_first"] = float(bool(move) and bool(sol) and move == sol[0]) if sol else float("nan")

    # grounding
    if fen and rec["completions"]:
        rep = verify_structured_trace(fen, rec["completions"][0], tbl)
        row["claim_precision"] = rep.precision if rep.n_scored else float("nan")
        row["n_claims"] = rep.n_scored
        row["n_false"] = rep.n_false
        row["hard_violation"] = float(rep.has_hard_violation)
        for t in (ClaimType.OCCUPANCY, ClaimType.ATTACK, ClaimType.PIN,
                  ClaimType.MOVE_LEGAL, ClaimType.STRUCTURE):
            cs = [c for c in rep.claims if c.type is t and c.scored]
            row[f"prec_{t.value}"] = (sum(c.verdict is ClaimVerdict.TRUE for c in cs) / len(cs)
                                      if cs else float("nan"))
        tr = parse_trace(rec["completions"][0], fen)
        row["n_candidates"] = len(tr.candidates)
        row["well_formed"] = float(tr.well_formed)
    else:
        row.update(claim_precision=float("nan"), n_claims=0, n_false=0,
                   hard_violation=float("nan"), n_candidates=0, well_formed=0.0)

    # verified reranking -- engine-free, so it is a legitimate test-time method
    # Self-consistency control. Verified reranking samples n traces and picks one, so the
    # gain has to be separated from the gain of merely sampling n times and voting.
    if fen and len(rec["completions"]) > 1:
        from collections import Counter
        kk = min(rerank_n or len(rec["completions"]), len(rec["completions"]))
        moves_k = [_first_move(rec, i) for i in range(kk)]
        votes = Counter(m for m in moves_k if m)
        if votes and tbl:
            maj = votes.most_common(1)[0][0]
            best_u = max(tbl, key=tbl.get)
            row[f"vote{kk}_top1"] = float(maj == best_u)
            row[f"vote{kk}_wp_loss"] = (max(0.0, win_prob(tbl[best_u]) - win_prob(tbl[maj]))
                                        if maj in tbl else 1.0)
            cand = [m for m in moves_k if m in tbl]
            if cand:
                row[f"oracle{kk}_top1"] = float(max(cand, key=lambda m: tbl[m]) == best_u)

    # QA benchmarks carry no position, so board-grounded reranking does not apply.
    if rerank_n and fen and len(rec["completions"]) > 1:
        k = min(rerank_n, len(rec["completions"]))
        rr = rerank(fen, rec["completions"][:k])
        row[f"rerank{k}_move"] = rr.move
        if tbl and rr.move:
            best = max(tbl.values())
            row[f"rerank{k}_top1"] = float(rr.move == max(tbl, key=tbl.get))
            row[f"rerank{k}_wp_loss"] = (max(0.0, win_prob(best) - win_prob(tbl[rr.move]))
                                         if rr.move in tbl else 1.0)
        if gold and rr.move:
            row[f"rerank{k}_gold"] = float(rr.move in gold)
    return row


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap. Vectorised: the metrics are recomputed often enough that a
    pure-Python resample loop dominates the runtime of a whole results pass."""
    import numpy as np
    vals = np.asarray([v for v in values
                       if isinstance(v, (int, float, bool)) and not (
                           isinstance(v, float) and math.isnan(v))], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = np.sort(vals[idx].mean(axis=1))
    return (float(vals.mean()),
            float(means[int(n_boot * alpha / 2)]),
            float(means[int(n_boot * (1 - alpha / 2))]))


def aggregate(rows: list[dict], keys: list[str], group_by: str | None = None,
              ci: bool = True) -> dict:
    """Mean (and bootstrap CI) for `keys`, optionally split by a grouping column."""
    def block(rs):
        out = {"n": len(rs)}
        for k in keys:
            vals = [r.get(k) for r in rs]
            if ci:
                m, lo, hi = bootstrap_ci(vals)
                out[k] = {"mean": m, "lo": lo, "hi": hi}
            else:
                out[k] = _mean(vals)
        return out

    if not group_by:
        return block(rows)
    groups = defaultdict(list)
    for r in rows:
        g = r.get(group_by)
        if isinstance(g, list):
            for t in g:
                groups[t].append(r)
        else:
            groups[g].append(r)
    return {str(g): block(rs) for g, rs in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def paired_test(rows_a: list[dict], rows_b: list[dict], key: str,
                n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap on the items both systems answered. Same positions, always."""
    a = {r["item_id"]: r.get(key) for r in rows_a}
    b = {r["item_id"]: r.get(key) for r in rows_b}
    ids = [i for i in a if i in b
           and isinstance(a[i], (int, float)) and isinstance(b[i], (int, float))
           and not math.isnan(a[i]) and not math.isnan(b[i])]
    if not ids:
        return {"n": 0}
    da = [a[i] for i in ids]
    db = [b[i] for i in ids]
    import numpy as np
    A = np.asarray(da, dtype=float)
    B = np.asarray(db, dtype=float)
    d0 = A - B
    obs = float(d0.mean())
    rng = np.random.default_rng(seed)
    n = len(ids)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d0[idx].mean(axis=1)
    # two-sided: how often a resample lands on the other side of zero
    cnt = int((boot <= 0).sum()) if obs > 0 else int((boot >= 0).sum())
    return {"n": n, "a": float(A.mean()), "b": float(B.mean()), "diff": obs,
            "p": min(1.0, 2.0 * cnt / n_boot)}


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Family-wise correction over the primary comparison family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (name, p) in enumerate(items):
        thr = alpha / (m - i)
        prev = max(prev, p)
        out[name] = {"p": p, "threshold": thr, "significant": prev <= thr}
    return out


# --------------------------------------------------------------------------- #
# Cross-variant metrics (faithfulness) -- need more than one variant present
# --------------------------------------------------------------------------- #

def faithfulness(records: list[dict]) -> dict:
    """Two probes, both requiring variants generated in the same run.

    perturbation_sensitivity: relocate a piece so the justification no longer holds. A
        model whose answer never moves was not using the board.
    reasoning_necessity: accuracy with the trace minus accuracy answering directly. If
        this is ~0, the reasoning is decoration.
    """
    by = defaultdict(dict)
    for r in records:
        by[r["item_id"]][r["variant"]] = r

    changed = tot = 0
    for _, v in by.items():
        if "base" in v and "perturbed" in v:
            m1, m2 = _first_move(v["base"]), _first_move(v["perturbed"])
            if m1 is not None:
                tot += 1
                changed += int(m1 != m2)

    base_rows = [score_record(r) for r in records if r["variant"] == "base"]
    nr_rows = [score_record(r) for r in records if r["variant"] == "no_reasoning"]
    key = "top1_engine"
    return {
        "perturbation_sensitivity": (changed / tot) if tot else float("nan"),
        "perturbation_n": tot,
        "reasoning_necessity": (_mean([r[key] for r in base_rows])
                                - _mean([r[key] for r in nr_rows])) if nr_rows else float("nan"),
        "base_acc": _mean([r[key] for r in base_rows]),
        "no_reasoning_acc": _mean([r[key] for r in nr_rows]) if nr_rows else float("nan"),
    }


HEADLINE = ["top1_engine", "top3_engine", "wp_loss", "illegal", "no_move",
            "claim_precision", "n_claims", "n_false", "hard_violation",
            "well_formed", "tokens", "top1"]


def full_report(path: str, rerank_n: int = 8) -> dict:
    """Everything, from one records file."""
    recs = load_records(path)
    base = [r for r in recs if r["variant"] == "base"]
    rows = [score_record(r, rerank_n=rerank_n) for r in base]
    rr_keys = [k for k in (rows[0] if rows else {}) if k.startswith("rerank")]
    keys = HEADLINE + [k for k in rr_keys if k.endswith(("top1", "wp_loss", "gold"))]
    return {
        "overall": aggregate(rows, keys),
        "by_benchmark": aggregate(rows, keys, "benchmark", ci=False),
        "by_band": aggregate(rows, keys, "gap_band", ci=False),
        "by_rating": aggregate(rows, keys, "rating_bucket", ci=False),
        "by_theme": aggregate(rows, ["top1_engine", "claim_precision"], "themes", ci=False),
        "faithfulness": faithfulness(recs),
        "n_records": len(recs),
    }
