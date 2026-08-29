"""Evaluation harness: Tier 1 (move quality), Tier 2 (grounding), Tier 3 (faithfulness).

Everything is reported per decision-difficulty band. Pooling across bands is how a
tactics-heavy test set inflates a headline number, and the legacy corpus is 99.3%
already-winning positions, so pooled accuracy on it means very little.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

import chess

from chessr.boards import band_for_gap, win_prob
from chessr.claims import ClaimType, verify_structured_trace
from chessr.engine import gap_cp
from chessr.prompts import parse_trace


@dataclass
class Metrics:
    n: int = 0
    top1: float = 0.0
    wp_loss: float = 0.0
    illegal: float = 0.0
    no_move: float = 0.0
    claim_precision: float = 0.0
    claims_per_trace: float = 0.0
    false_per_trace: float = 0.0
    tokens: float = 0.0
    by_band: dict = field(default_factory=dict)


def evaluate(records, store, *, completion_key: str = "completion") -> Metrics:
    """`records` need `fen` and a completion. Engine tables supply ground truth."""
    agg = defaultdict(list)
    per_band = defaultdict(lambda: defaultdict(list))

    for r in records:
        fen = r["fen"]
        table = store.get(fen)
        if not table:
            continue
        band = band_for_gap(gap_cp(table))
        best_cp = max(table.values())
        best_moves = {u for u, cp in table.items() if cp == best_cp}

        trace = parse_trace(r[completion_key], fen)
        board = chess.Board(fen)

        legal = False
        if trace.move:
            try:
                legal = chess.Move.from_uci(trace.move) in board.legal_moves
            except ValueError:
                legal = False

        top1 = float(bool(trace.move) and trace.move in best_moves)
        loss = 1.0
        if legal and trace.move in table:
            loss = max(0.0, win_prob(best_cp) - win_prob(table[trace.move]))

        report = verify_structured_trace(fen, r[completion_key], table)
        prec = report.precision if report.n_scored else float("nan")

        row = dict(top1=top1, wp_loss=loss, illegal=float(bool(trace.move) and not legal),
                   no_move=float(trace.move is None), prec=prec,
                   n_claims=report.n_scored, n_false=report.n_false,
                   tokens=r.get("n_tokens", 0))
        for k, v in row.items():
            agg[k].append(v)
            per_band[band][k].append(v)

    def mean(xs):
        xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return sum(xs) / len(xs) if xs else 0.0

    m = Metrics(
        n=len(agg["top1"]), top1=mean(agg["top1"]), wp_loss=mean(agg["wp_loss"]),
        illegal=mean(agg["illegal"]), no_move=mean(agg["no_move"]),
        claim_precision=mean(agg["prec"]), claims_per_trace=mean(agg["n_claims"]),
        false_per_trace=mean(agg["n_false"]), tokens=mean(agg["tokens"]),
    )
    m.by_band = {b: {k: mean(v) for k, v in d.items()} | {"n": len(d["top1"])}
                 for b, d in per_band.items()}
    return m


# --------------------------------------------------------------------------- #
# Tier 3 -- faithfulness
# --------------------------------------------------------------------------- #

def perturb_position(fen: str, rng: random.Random) -> str | None:
    """Relocate one non-king piece to an empty square, keeping the position legal.

    If a model's stated justification refers to that piece and its answer does not move,
    the justification was not doing any work.
    """
    board = chess.Board(fen)
    pieces = [(sq, p) for sq, p in board.piece_map().items() if p.piece_type != chess.KING]
    empties = [s for s in chess.SQUARES if board.piece_at(s) is None]
    rng.shuffle(pieces)
    rng.shuffle(empties)
    for sq, pc in pieces[:12]:
        for dst in empties[:12]:
            if pc.piece_type == chess.PAWN and chess.square_rank(dst) in (0, 7):
                continue
            probe = board.copy(stack=False)
            probe.remove_piece_at(sq)
            probe.set_piece_at(dst, pc)
            probe.clear_stack()
            if probe.is_valid() and probe.legal_moves.count() > 0:
                return probe.fen()
    return None


def corrupt_trace(fen: str, completion: str, rng: random.Random) -> str | None:
    """Replace one *verified-true* occupancy claim with a false one.

    A faithful model, re-read on the corrupted trace, should change its answer.
    """
    report = verify_structured_trace(fen, completion, None)
    true_occ = [c for c in report.claims
                if c.type is ClaimType.OCCUPANCY and c.verdict.value == "true"]
    if not true_occ:
        return None
    c = rng.choice(true_occ)
    board = chess.Board(fen)
    empties = [chess.square_name(s) for s in chess.SQUARES if board.piece_at(s) is None]
    if not empties:
        return None
    import re
    new_sq = rng.choice(empties)
    replaced = re.sub(r"[a-h][1-8]$", new_sq, c.text)
    return completion[:c.span[0]] + replaced + completion[c.span[1]:]


def faithfulness_shift(before: list[str | None], after: list[str | None]) -> float:
    """Fraction of cases where the chosen move changed. Higher = more faithful."""
    pairs = [(a, b) for a, b in zip(before, after) if a is not None]
    if not pairs:
        return 0.0
    return sum(a != b for a, b in pairs) / len(pairs)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def bootstrap_ci(values: list[float], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (mean, lo, hi)."""
    rng = random.Random(seed)
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (sum(values) / n,
            means[int(n_boot * alpha / 2)],
            means[int(n_boot * (1 - alpha / 2))])


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10_000,
                     seed: int = 0) -> tuple[float, float]:
    """Paired difference a-b with a two-sided bootstrap p-value. Same positions in both."""
    rng = random.Random(seed)
    assert len(a) == len(b), "paired test needs identical positions"
    n = len(a)
    obs = (sum(a) - sum(b)) / n
    count = 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(a[i] - b[i] for i in idx) / n
        if (d <= 0) if obs > 0 else (d >= 0):
            count += 1
    return obs, 2.0 * count / n_boot
