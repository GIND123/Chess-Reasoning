"""Reward components and TRL-compatible reward functions.

    R = w1*R_move + w2*R_precision + w3*R_coverage + w4*R_format
        - hard penalties (illegal chosen move, false claim about the root position)

Every component is a pure function of (fen, trace, engine table). No model, no engine
call at training time.

R_coverage is not optional. A policy maximising precision alone learns to assert one
trivially-true fact and stop; coverage is what closes that exit. Ablation A3 runs the
degenerate configuration on purpose so the failure is documented rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from chessr.boards import win_prob
from chessr.claims import ClaimType, ClaimVerdict, verify_structured_trace
from chessr.prompts import parse_trace

TOL_WP = 0.10            # win-probability loss that maps to zero move reward
MIN_ROOT_CLAIMS = 2      # <read> must ground itself at least this much
TARGET_CLAIMS = 6        # coverage saturates here
PENALTY_ILLEGAL = 1.0
PENALTY_FALSE_CLAIM = 0.5


@dataclass
class RewardBreakdown:
    move: float = 0.0
    precision: float = 0.0
    coverage: float = 0.0
    fmt: float = 0.0
    penalty: float = 0.0
    total: float = 0.0
    # diagnostics, logged during training
    chosen: str | None = None
    legal: bool = False
    wp_loss: float = 1.0
    n_scored: int = 0
    n_false: int = 0
    info: dict = field(default_factory=dict)


def r_move(fen: str, chosen: str | None, table: dict[str, int] | None,
           tol: float = TOL_WP) -> tuple[float, float, bool]:
    """Graded move quality in win probability. Returns (reward, wp_loss, legal).

    Win probability rather than centipawns: 50 cp is decisive at equality and irrelevant
    at +900, so a cp-linear reward would be nearly flat exactly where this corpus lives.
    """
    if not chosen or not table:
        return 0.0, 1.0, False
    board = chess.Board(fen)
    try:
        mv = chess.Move.from_uci(chosen)
    except ValueError:
        return 0.0, 1.0, False
    if mv not in board.legal_moves:
        return 0.0, 1.0, False
    cp_played = table.get(chosen)
    if cp_played is None:
        return 0.0, 1.0, True
    loss = max(0.0, win_prob(max(table.values())) - win_prob(cp_played))
    return max(0.0, 1.0 - loss / tol), loss, True


def r_precision(report) -> float:
    """Verified claims / scorable claims. 1.0 when nothing was asserted -- see module docstring."""
    return report.precision


def r_coverage(report, target: int = TARGET_CLAIMS) -> float:
    """How much verifiable grounding the trace actually offers, saturating at `target`.

    Only *true* claims count, so coverage cannot be farmed by asserting false facts.
    """
    n_true_root = sum(
        1 for c in report.claims
        if c.verdict is ClaimVerdict.TRUE
        and c.type in (ClaimType.OCCUPANCY, ClaimType.ATTACK, ClaimType.PIN,
                       ClaimType.MOBILITY, ClaimType.STRUCTURE, ClaimType.MOVE_LEGAL)
    )
    return min(1.0, n_true_root / target)


def r_format(trace) -> float:
    score = 0.0
    if trace.read:
        score += 0.25
    if len(trace.candidates) >= 3:
        score += 0.35
    elif trace.candidates:
        score += 0.15
    if trace.choice:
        score += 0.15
    if trace.move:
        score += 0.25
    return min(1.0, score)


def score_completion(fen: str, completion: str, table: dict[str, int] | None,
                     *, weights: tuple[float, float, float, float] = (1.0, 0.5, 0.3, 0.2),
                     use_penalties: bool = True) -> RewardBreakdown:
    """The full composite reward for one rollout."""
    trace = parse_trace(completion)
    report = verify_structured_trace(fen, completion, table)

    rm, wp_loss, legal = r_move(fen, trace.move, table)
    rp = r_precision(report)
    rc = r_coverage(report)
    rf = r_format(trace)

    w1, w2, w3, w4 = weights
    b = RewardBreakdown(
        move=rm, precision=rp, coverage=rc, fmt=rf,
        chosen=trace.move, legal=legal, wp_loss=wp_loss,
        n_scored=report.n_scored, n_false=report.n_false,
    )
    if use_penalties:
        if trace.move and not legal:
            b.penalty += PENALTY_ILLEGAL
        if report.violations(ClaimType.OCCUPANCY):
            b.penalty += PENALTY_FALSE_CLAIM
    b.total = w1 * rm + w2 * rp + w3 * rc + w4 * rf - b.penalty
    b.info = report.counts_by_type()
    return b


# --------------------------------------------------------------------------- #
# TRL reward functions
#
# TRL calls each reward function with keyword arguments: `prompts`, `completions`,
# `completion_ids`, `trainer_state`, `log_metric`, plus every remaining dataset column.
# Our dataset carries an `fen` column, so it arrives as `fen=[...]`.
#
# Weighting is done by GRPOConfig.reward_weights, so each function here returns its own
# unweighted component. That way A1 (reward composition) is a config change, not a code
# change.
# --------------------------------------------------------------------------- #

def make_reward_fns(store, *, sparse: bool = False, tol: float = TOL_WP):
    """Build the reward callables. `store` is a chessr.engine.TableStore.

    sparse=True gives the binary outcome reward used by comparisons M2 and M4.
    """

    def move_reward(completions, fen, **kwargs):
        log = kwargs.get("log_metric")
        out, losses, legals = [], [], []
        for c, f in zip(completions, fen):
            tr = parse_trace(c)
            tbl = store.get(f)
            if sparse:
                best = max(tbl, key=tbl.get) if tbl else None
                r = 1.0 if (tr.move and best and tr.move == best) else 0.0
                loss, legal = (0.0 if r else 1.0), bool(tr.move)
            else:
                r, loss, legal = r_move(f, tr.move, tbl, tol)
            out.append(r); losses.append(loss); legals.append(float(legal))
        if log:
            log("wp_loss", sum(losses) / max(len(losses), 1))
            log("legal_rate", sum(legals) / max(len(legals), 1))
        return out

    def precision_reward(completions, fen, **kwargs):
        log = kwargs.get("log_metric")
        out, nfalse = [], []
        for c, f in zip(completions, fen):
            rep = verify_structured_trace(f, c, store.get(f))
            out.append(rep.precision); nfalse.append(rep.n_false)
        if log:
            log("false_claims", sum(nfalse) / max(len(nfalse), 1))
        return out

    def coverage_reward(completions, fen, **kwargs):
        log = kwargs.get("log_metric")
        out = []
        for c, f in zip(completions, fen):
            out.append(r_coverage(verify_structured_trace(f, c, store.get(f))))
        if log:
            log("coverage", sum(out) / max(len(out), 1))
        return out

    def format_reward(completions, **kwargs):
        return [r_format(parse_trace(c)) for c in completions]

    def penalty_reward(completions, fen, **kwargs):
        """Negative. Hard constraints are penalties, not weighted preferences."""
        out = []
        for c, f in zip(completions, fen):
            tr = parse_trace(c)
            rep = verify_structured_trace(f, c, store.get(f))
            p = 0.0
            if tr.move:
                b = chess.Board(f)
                try:
                    if chess.Move.from_uci(tr.move) not in b.legal_moves:
                        p += PENALTY_ILLEGAL
                except ValueError:
                    p += PENALTY_ILLEGAL
            else:
                p += PENALTY_ILLEGAL
            if rep.violations(ClaimType.OCCUPANCY):
                p += PENALTY_FALSE_CLAIM
            out.append(-p)
        return out

    move_reward.__name__ = "move_reward"
    precision_reward.__name__ = "precision_reward"
    coverage_reward.__name__ = "coverage_reward"
    format_reward.__name__ = "format_reward"
    penalty_reward.__name__ = "penalty_reward"
    return {
        "move": move_reward,
        "precision": precision_reward,
        "coverage": coverage_reward,
        "format": format_reward,
        "penalty": penalty_reward,
    }
