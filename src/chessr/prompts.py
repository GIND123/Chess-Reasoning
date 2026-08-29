"""Prompt construction and structured-trace parsing.

The teacher sees the engine table; the student never does. The stored SFT pair is
always (student prompt -> trace), so nothing that leaks the answer can reach the policy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import chess

from chessr.boards import ascii_board, piece_list, side_to_move

SYS_TEACHER = """You write chess analysis for a training corpus. You are given a position and \
private analysis of the candidate moves. That analysis tells you what is true, but it must never \
appear in your output: write as though you worked the position out yourself.

Hard rules:
1. Never write the words engine, evaluation, score, centipawn, or suggests, and never write a \
number in parentheses. Never say "the best move is" or refer to being given an answer.
2. Every statement about the board must be true of the position exactly as given. Name pieces by \
the square they stand on. If you are not certain, do not write it.
3. Do NOT restate the piece inventory -- it is already in the prompt. The <read> section is for \
threats, weaknesses, loose pieces and key squares, in at most two sentences, and every one of them \
must name the piece and the square it stands on (for example "the knight on f6 is pinned", "h7 is \
defended only by the king"). At least two such statements.
4. Give exactly three candidate moves. At least one must be a plausible mistake a strong human \
would consider, and which the private analysis shows is inferior.
5. The continuation field must be copied EXACTLY from the continuation given for that move in the \
private analysis. Do not invent moves and do not repeat the candidate move itself. If the analysis \
says there is no forcing reply, leave the field empty.
6. Each verdict is at most twelve words and must be concrete: name a square, a recapture, or a \
material or mating consequence. Never "passive" or "improves coordination" alone.
7. Be brief. Begin your reply with <read> immediately. Output this structure and nothing else:

<read>
At most two sentences on threats and weaknesses.
</read>
<candidates>
1. <uci> | <opponent replies as UCI> | <verdict, <=12 words>
2. <uci> | <opponent replies as UCI> | <verdict, <=12 words>
3. <uci> | <opponent replies as UCI> | <verdict, <=12 words>
</candidates>
<choice>
One sentence naming the chosen move and the decisive reason.
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
    return (f"{ascii_board(fen)}\n\nFEN: {fen}\n\n{piece_list(fen)}\n\n"
            f"{side_to_move(fen)} to move.\n\n"
            f"Analyse the position and choose the best move.")


def teacher_prompt(fen: str, move_table: dict[str, int],
                   pvs: dict[str, list[str]] | None = None, top_k: int = 8) -> str:
    """Generation-time only. `move_table` maps UCI -> centipawns for every legal move;
    `pvs` gives the true continuation for the strongest moves, so the teacher describes
    real lines rather than inventing them."""
    pvs = pvs or {}
    ranked = sorted(move_table.items(), key=lambda kv: -kv[1])[:top_k]
    rows = []
    for u, cp in ranked:
        line = pvs.get(u, [])
        cont = " ".join(line[1:]) if len(line) > 1 else "(no forcing reply)"
        rows.append(f"  {u}: {cp:+d} | continuation: {cont}")
    lines = "\n".join(rows)
    return (f"{ascii_board(fen)}\n\nFEN: {fen}\n\n{piece_list(fen)}\n\n"
            f"{side_to_move(fen)} to move.\n\n"
            f"PRIVATE analysis (never reference, never quote):\n{lines}\n\n"
            f"Write the analysis.")


# --------------------------------------------------------------------------- #
# Trace parsing
# --------------------------------------------------------------------------- #

_TAG = {k: re.compile(rf"<{k}>(.*?)</{k}>", re.S | re.I)
        for k in ("read", "candidates", "choice", "move")}
_CAND_LINE = re.compile(r"^\s*\d+\s*[.)]\s*(?P<move>\S+)\s*\|(?P<rest>.*)$", re.M)
# Models routinely write "g8xh7", mixing SAN's capture marker into UCI. That is a
# notation slip, not a claim about the board, so it is normalised rather than failed.
_UCI = re.compile(r"\b([a-h][1-8])x?([a-h][1-8])([qrbn]?)\b")


def _uci(m: "re.Match") -> str:
    return (m.group(1) + m.group(2) + m.group(3)).lower()


_BARE_SQ = re.compile(r"^[a-h][1-8]$")


def resolve_move(board: "chess.Board", token: str) -> str | None:
    """Normalise a move written in whatever notation the model reached for.

    Models mix UCI, SAN and bare destination squares freely. Notation is not a claim
    about the board, so it is normalised rather than failed -- but a token that cannot be
    resolved to a *legal* move returns None instead of being passed through, which is what
    previously produced cascades of phantom violations.
    """
    if not token:
        return None
    tok = token.strip().rstrip(".,;:")

    # UCI, tolerating an interpolated capture marker ("g8xh7").
    m = re.fullmatch(r"([a-h][1-8])x?([a-h][1-8])([qrbn]?)", tok.lower())
    if m:
        uci = m.group(1) + m.group(2) + m.group(3)
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            mv = None
        if mv is not None:
            if mv in board.legal_moves:
                return mv.uci()
            if len(uci) == 4:
                promo = chess.Move.from_uci(uci + "q")
                if promo in board.legal_moves:
                    return promo.uci()
        return None

    # SAN, including castling and long algebraic.
    for cand in (tok, tok.replace("-", "")):
        try:
            return board.parse_san(cand).uci()
        except ValueError:
            pass

    # A bare destination square, but only when it is unambiguous.
    if _BARE_SQ.match(tok.lower()):
        dst = chess.parse_square(tok.lower())
        hits = [mv for mv in board.legal_moves if mv.to_square == dst]
        if len(hits) == 1:
            return hits[0].uci()
    return None


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
    move: str | None = None          # resolved to legal UCI, or None
    move_raw: str = ""               # what the model actually wrote
    candidates: list[Candidate] = field(default_factory=list)
    raw: str = ""
    well_formed: bool = False

    @property
    def prose(self) -> str:
        """Everything that makes claims about the *root* position."""
        return f"{self.read}\n{self.choice}"


def parse_trace(text: str, fen: str | None = None) -> Trace:
    """Parse the structured trace. Tolerant: a missing section yields an empty field
    rather than an exception, and `well_formed` records whether everything was present.

    When `fen` is supplied, move tokens are resolved against the board so that SAN, long
    algebraic and bare destination squares all normalise to legal UCI. A token that
    cannot be resolved is dropped rather than passed through as a phantom move.
    """
    board = chess.Board(fen) if fen else None
    t = Trace(raw=text)
    got = {}
    for k, rx in _TAG.items():
        m = rx.search(text)
        got[k] = m.group(1).strip() if m else ""

    t.read = got["read"]
    t.choice = got["choice"]
    t.move_raw = got["move"].strip() if got["move"] else ""
    if got["move"]:
        if board is not None:
            t.move = resolve_move(board, got["move"].strip())
            if t.move is None:
                mv = _UCI.search(got["move"])
                t.move = resolve_move(board, _uci(mv)) if mv else None
        else:
            mv = _UCI.search(got["move"])
            t.move = _uci(mv) if mv else None
    else:
        t.move = None

    for m in _CAND_LINE.finditer(got["candidates"]):
        parts = [p.strip() for p in m.group("rest").split("|")]
        cont = parts[0] if parts else ""
        verdict = parts[1] if len(parts) > 1 else ""
        raw_head = m.group("move").strip()
        if board is not None:
            head_uci = resolve_move(board, raw_head)
            if head_uci is None:
                hm = _UCI.search(raw_head)
                head_uci = resolve_move(board, _uci(hm)) if hm else None
        else:
            hm = _UCI.search(raw_head)
            head_uci = _uci(hm) if hm else raw_head.lower()
        if head_uci is None:
            continue                      # unresolvable notation: not a claim, just noise
        t.candidates.append(Candidate(
            move=head_uci,
            line=[_uci(mm) for mm in _UCI.finditer(cont)],
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
            return _uci(u)
    hits = [_uci(m) for m in _UCI.finditer(text)]
    return hits[-1] if hits else None
