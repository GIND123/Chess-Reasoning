"""Acceptance gates for generated traces.

Reject and resample; never repair with a second model pass, which would reintroduce
exactly the unverified content the filter exists to remove.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from chessr.boards import win_prob
from chessr.claims import ClaimType, verify_structured_trace
from chessr.prompts import parse_trace

LEAK_RX = re.compile(
    r"\b(engine|centipawn|evaluation|eval score|the score|suggests that the best|"
    r"cp\(|mate\(|private analysis|as given|provided analysis)\b", re.I)


@dataclass
class Gates:
    tol_wp: float = 0.10          # win-probability tolerance on the chosen move
    min_precision: float = 0.90   # over scorable claims
    min_root_claims: int = 2      # <read> must ground itself
    require_three_candidates: bool = True
    forbid_false_occupancy: bool = True
    forbid_illegal_move: bool = True
    graded: bool = False          # Set B: keep with a graded label instead of rejecting


@dataclass
class Decision:
    accept: bool
    reasons: list[str]
    wp_loss: float
    precision: float
    n_root_claims: int
    move: str | None


def judge(fen: str, completion: str, table: dict[str, int] | None,
          gates: Gates = Gates()) -> Decision:
    trace = parse_trace(completion, fen)
    report = verify_structured_trace(fen, completion, table)
    reasons: list[str] = []

    if not trace.move:
        reasons.append("no <move>")
    if LEAK_RX.search(completion):
        reasons.append("engine leak in trace")
    if "<think>" in completion:
        reasons.append("thinking block not disabled")
    if gates.require_three_candidates and len(trace.candidates) < 3:
        reasons.append(f"only {len(trace.candidates)} candidates")

    wp_loss = 1.0
    if table and trace.move:
        cp = table.get(trace.move)
        if cp is None:
            reasons.append("chosen move illegal")
        else:
            wp_loss = max(0.0, win_prob(max(table.values())) - win_prob(cp))
            if not gates.graded and wp_loss > gates.tol_wp:
                reasons.append(f"wp_loss {wp_loss:.3f} > {gates.tol_wp}")

    if gates.forbid_false_occupancy and report.violations(ClaimType.OCCUPANCY):
        reasons.append("false occupancy claim")
    if gates.forbid_illegal_move and report.violations(ClaimType.MOVE_LEGAL):
        reasons.append("illegal move reference")

    prec = report.precision
    if report.n_scored and prec < gates.min_precision:
        reasons.append(f"precision {prec:.2f} < {gates.min_precision}")

    n_root = sum(1 for c in report.claims
                 if c.type in (ClaimType.OCCUPANCY, ClaimType.ATTACK, ClaimType.PIN,
                               ClaimType.MOBILITY, ClaimType.STRUCTURE)
                 and c.verdict.value == "true")
    if n_root < gates.min_root_claims:
        reasons.append(f"only {n_root} grounded claims")

    return Decision(not reasons, reasons, wp_loss, prec, n_root, trace.move)


def filter_stream(records, store, gates: Gates = Gates()):
    """Yield accepted records; return a Counter of rejection reasons via `.stats`."""
    stats: Counter = Counter()

    def _gen():
        for rec in records:
            d = judge(rec["fen"], rec["completion"], store.get(rec["fen"]), gates)
            stats["total"] += 1
            if d.accept:
                stats["accepted"] += 1
                rec = dict(rec)
                rec.update(wp_loss=d.wp_loss, precision=d.precision,
                           n_root_claims=d.n_root_claims, move=d.move)
                yield rec
            else:
                for r in d.reasons:
                    stats[r.split(" ")[0] if " " in r else r] += 1

    g = _gen()
    g.stats = stats  # type: ignore[attr-defined]
    return g
