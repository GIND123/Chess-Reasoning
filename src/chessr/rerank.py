"""Verified reranking: choose among sampled traces using board verification alone.

Claim types 1-8 need only python-chess. No engine, no ground truth, nothing that is
unavailable at test time. So sampling n traces and returning the move from the
best-verified one is a legitimate inference-time method, not an oracle.

If this works, it is also the cleanest evidence that grounding is causal rather than
cosmetic: a trace that is more true about the board leads to a better move.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from chessr.claims import ClaimType, ClaimVerdict, verify_structured_trace
from chessr.prompts import parse_trace
from chessr.reward import completion_text, r_coverage, r_format


@dataclass
class RerankResult:
    move: str | None
    score: float
    index: int
    scores: list[float]
    votes: Counter


def trace_score(fen: str, completion: str, *, w_prec: float = 1.0, w_cov: float = 0.5,
                w_fmt: float = 0.25, penalty_false: float = 1.0) -> float:
    """Engine-free quality score for a single trace."""
    completion = completion_text(completion)
    report = verify_structured_trace(fen, completion, engine_table=None)
    trace = parse_trace(completion, fen)
    n_false = sum(c.verdict is ClaimVerdict.FALSE for c in report.claims)
    return (w_prec * report.precision
            + w_cov * r_coverage(report)
            + w_fmt * r_format(trace)
            - penalty_false * n_false)


def rerank(fen: str, completions: list[str], *, vote: bool = False,
           **kw) -> RerankResult:
    """Pick the move from the highest-scoring trace.

    `vote=True` aggregates verification score per distinct move instead of taking the
    single best trace -- a verification-weighted variant of self-consistency.
    """
    scores = [trace_score(fen, c, **kw) for c in completions]
    moves = [parse_trace(c, fen).move for c in completions]

    votes: Counter = Counter()
    for m, s in zip(moves, scores):
        if m:
            votes[m] += max(s, 0.0)

    if vote and votes:
        best_move, best_score = votes.most_common(1)[0]
        idx = next(i for i, m in enumerate(moves) if m == best_move)
        return RerankResult(best_move, best_score, idx, scores, votes)

    if not scores:
        return RerankResult(None, 0.0, -1, [], votes)
    idx = max(range(len(scores)), key=lambda i: scores[i])
    return RerankResult(moves[idx], scores[idx], idx, scores, votes)
