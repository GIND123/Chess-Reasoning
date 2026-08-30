"""Deterministic grading for ChessQA, from saved evaluation records.

The harness stores raw answers, so grading happens offline and a change of rule costs no
GPU time. Each task family has its own scoring:

  multi   set F1 against the reference (legal moves, controlled squares, motifs)
  single  exact match after normalisation (state tracking, tactics, multiple choice)
  judge   position evaluation, scored both exactly and with a one-bucket tolerance
"""
from __future__ import annotations

import re
from collections import defaultdict

UCI = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")
SQUARE = re.compile(r"\b[a-h][1-8]\b")
MOTIF = re.compile(r"\b[a-h][1-8]\s*[>\-]\s*[a-h][1-8](?:\s*[>\-]\s*[a-h][1-8])?\b")
FEN_RX = re.compile(r"\b[rnbqkpRNBQKP1-8/]{15,}\s+[wb]\s+\S+\s+\S+\s+\d+\s+\d+")
CHOICE = re.compile(r"\b([A-D])\b")
NUMBER = re.compile(r"-?\d+")


def _norm_set(text: str, pattern: re.Pattern) -> set[str]:
    return {m.group(0).replace(" ", "").lower() for m in pattern.finditer(text or "")}


def _f1(pred: set[str], gold: set[str]) -> float:
    if not gold:
        return float("nan")
    if not pred:
        return 0.0
    tp = len(pred & gold)
    if not tp:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def _last(pattern: re.Pattern, text: str) -> str | None:
    hits = pattern.findall(text or "")
    return hits[-1] if hits else None


def grade(task_type: str, reference: str, answer: str) -> dict:
    """Return {metric: value} for one item. Metrics differ by family, so aggregation
    is done per family rather than pooled."""
    t, ref, ans = task_type or "", reference or "", answer or ""

    if t.startswith("structural_state_tracking"):
        m = FEN_RX.search(ans)
        got = m.group(0) if m else ""
        # board placement only: side-to-move and clocks are a separate skill
        return {"exact": float(got.split()[0] == ref.split()[0]) if got and ref else 0.0}

    if t == "structural_piece_arrangement":
        gold = _norm_set(ref, SQUARE)
        return {"f1": _f1(_norm_set(ans, SQUARE), gold)}

    if t.startswith("structural_legal_move") or t == "structural_check_in_1":
        return {"f1": _f1(_norm_set(ans, UCI), _norm_set(ref, UCI))}

    if t in ("structural_capture_squares", "structural_control_squares",
             "structural_protect_squares"):
        return {"f1": _f1(_norm_set(ans, SQUARE), _norm_set(ref, SQUARE))}

    if t == "structural_check_detection":
        gold = _norm_set(ref, SQUARE)
        return {"f1": _f1(_norm_set(ans, SQUARE), gold)}

    if t.startswith("motifs_"):
        gold = _norm_set(ref, MOTIF) or _norm_set(ref, UCI)
        pred = _norm_set(ans, MOTIF) or _norm_set(ans, UCI)
        return {"f1": _f1(pred, gold)}

    if t.startswith("short_tactics_"):
        got = _last(UCI, ans)
        return {"exact": float(bool(got) and got.lower() == ref.strip().lower())}

    if t.startswith("position_judgement"):
        got = _last(NUMBER, ans)
        if got is None:
            return {"exact": 0.0, "within_one_bucket": 0.0}
        try:
            g, r = int(got), int(ref)
        except ValueError:
            return {"exact": 0.0, "within_one_bucket": 0.0}
        return {"exact": float(g == r), "within_one_bucket": float(abs(g - r) <= 200)}

    if t.startswith("semantic_"):
        # the answer is a letter choice; take the last standalone A-D
        hits = CHOICE.findall(ans.upper())
        got = hits[-1] if hits else None
        return {"exact": float(bool(got) and got == ref.strip().upper())}

    return {}


def grade_records(records) -> dict:
    """Aggregate by ChessQA category and by task type."""
    by_cat, by_task = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for r in records:
        if not r["benchmark"].startswith("chessqa"):
            continue
        tt = (r.get("meta") or {}).get("task_type", "")
        ans = r["completions"][0] if r["completions"] else ""
        for k, v in grade(tt, r.get("meta_answer", ""), ans).items():
            by_cat[r["benchmark"]][k].append(v)
            by_task[tt][k].append(v)

    def agg(d):
        out = {}
        for name, metrics in d.items():
            out[name] = {k: sum(v) / len(v) for k, v in metrics.items() if v}
            out[name]["n"] = max(len(v) for v in metrics.values()) if metrics else 0
        return out

    return {"by_category": agg(by_cat), "by_task": agg(by_task)}
