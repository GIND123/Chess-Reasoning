"""Prompt construction and structured-trace parsing.

The teacher sees the engine table; the student never does. The stored SFT pair is
always (student prompt -> trace), so nothing that leaks the answer can reach the policy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from chessr.boards import ascii_board, side_to_move

SYS_TEACHER = """You write chess analysis for a training corpus. You are given a position and \
engine analysis of the candidate moves. The engine data is PRIVATE CONTEXT: it tells you what is \
true, but it must never appear in your output, and your output must read as analysis produced by a \
strong player seeing this position for the first time.

Hard rules:
1. Never mention the engine, evaluations, centipawns, scores, "the best move is", or the fact that \
you were given an answer.
2. Every factual statement about the board must be true of the position exactly as given. Name \
pieces by square. If you are not certain a piece is on a square, do not mention it.
3. Consider exactly three candidate moves. At least one must be a plausible mistake that a strong \
human would seriously consider, and which the engine data shows is inferior.
4. For each candidate give the concrete continuation as a sequence of UCI moves, and the concrete \
reason it succeeds or fails: a specific recapture, a specific square, a specific material or \
mating consequence. Never "this is passive" or "improves coordination" without a follow-up.
5. Write 120-220 words total. Output the structure below and nothing else.

<read>
Two or three sentences of concrete position facts: material, king squares, immediate threats, key \
squares. Verifiable statements only.
</read>
<candidates>
1. <uci> | <continuation as UCI moves> | <verdict>
2. <uci> | <continuation as UCI moves> | <verdict>
3. <uci> | <continuation as UCI moves> | <verdict>
</candidates>
<choice>
One sentence: the chosen move and the single decisive reason.
</choice>
<move><uci></move>"""

SYS_STUDENT = """You are a chess analyst. Given a position, analyse it and choose the best move.
Name pieces by the square they stand on, and give concrete continuations as UCI moves.
Reply using exactly this structure and nothing else:

<read>...</read>
<candidates>
1. <uci> | <continuation> | <verdict>
2. <uci> | <continuation> | <verdict>
3. <uci> | <continuation> | <verdict>
</candidates>
<choice>...</choice>
<move><uci></move>"""


def student_prompt(fen: str) -> str:
    """What the policy sees at training and inference. No engine data, no answer."""
    return (f"{ascii_board(fen)}\n\nFEN: {fen}\n{side_to_move(fen)} to move.\n\n"
            f"Analyse the position and choose the best move.")


def teacher_prompt(fen: str, move_table: dict[str, int], top_k: int = 12) -> str:
    """Generation-time only. `move_table` maps UCI -> centipawns for every legal move."""
    ranked = sorted(move_table.items(), key=lambda kv: -kv[1])[:top_k]
    lines = "\n".join(f"  {u}: {cp:+d}" for u, cp in ranked)
    return (f"{ascii_board(fen)}\n\nFEN: {fen}\n{side_to_move(fen)} to move.\n\n"
            f"PRIVATE engine analysis (never reference):\n{lines}\n\nWrite the analysis.")


# --------------------------------------------------------------------------- #
# Trace parsing
# --------------------------------------------------------------------------- #

_TAG = {k: re.compile(rf"<{k}>(.*?)</{k}>", re.S | re.I)
        for k in ("read", "candidates", "choice", "move")}
_CAND_LINE = re.compile(r"^\s*\d+\s*[.)]\s*(?P<move>\S+)\s*\|(?P<rest>.*)$", re.M)
_UCI = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


@dataclass
class Candidate:
    move: str
    line: list[str] = field(default_factory=list)   # continuation, UCI, replayed in order
    verdict: str = ""
    raw: str = ""


@dataclass
class Trace:
    read: str = ""
    choice: str = ""
    move: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    raw: str = ""
    well_formed: bool = False

    @property
    def prose(self) -> str:
        """Everything that makes claims about the *root* position."""
        return f"{self.read}\n{self.choice}"


def parse_trace(text: str) -> Trace:
    """Parse the structured trace. Tolerant: a missing section yields an empty field
    rather than an exception, and `well_formed` records whether everything was present."""
    t = Trace(raw=text)
    got = {}
    for k, rx in _TAG.items():
        m = rx.search(text)
        got[k] = m.group(1).strip() if m else ""

    t.read = got["read"]
    t.choice = got["choice"]
    mv = _UCI.search(got["move"]) if got["move"] else None
    t.move = mv.group(1).lower() if mv else None

    for m in _CAND_LINE.finditer(got["candidates"]):
        parts = [p.strip() for p in m.group("rest").split("|")]
        cont = parts[0] if parts else ""
        verdict = parts[1] if len(parts) > 1 else ""
        head = _UCI.search(m.group("move"))
        t.candidates.append(Candidate(
            move=head.group(1).lower() if head else m.group("move").strip().lower(),
            line=[u.lower() for u in _UCI.findall(cont)],
            verdict=verdict,
            raw=m.group(0),
        ))

    t.well_formed = bool(t.read and t.candidates and t.move)
    return t


def extract_move(text: str) -> str | None:
    """Last-resort move extraction for models that ignore the format."""
    m = _TAG["move"].search(text)
    if m:
        u = _UCI.search(m.group(1))
        if u:
            return u.group(1).lower()
    hits = _UCI.findall(text)
    return hits[-1].lower() if hits else None
