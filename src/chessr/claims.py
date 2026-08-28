"""
Deterministic claim verification for chess reasoning traces.

A reasoning trace is decomposed into *atomic claims* by regular grammar (no model in
the loop), and each claim is checked against the root position with python-chess, or
against a precomputed engine table for the three claim types that need search.

Design rules, all of which matter for using this as a reward:

1.  **No LLM anywhere.** Extraction is parsing; checking is a function call. The reward is
    reproducible bit-for-bit and costs microseconds.
2.  **Conservative extraction.** A pattern that could plausibly match non-claims is not
    included. We would rather under-extract (lower recall, honest precision) than invent
    violations the model cannot fix. Bare algebraic squares are *never* read as moves.
3.  **An explicit UNVERIFIABLE class.** Vague strategic prose is counted but not scored, so
    the reward neither rewards nor punishes it as though it were false.
4.  **Every verdict carries its evidence** so failures can be audited and hand-labelled.

Claim types 1-8 need only the board. Types 9-11 need the engine table and are skipped
when it is absent -- which is what makes verified reranking (rerank.py) possible at test
time with no engine and no ground truth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import chess

__all__ = ["ClaimType", "ClaimVerdict", "Claim", "VerificationReport", "verify_trace",
           "extract_claims"]


class ClaimType(str, Enum):
    OCCUPANCY = "occupancy"          # 1  a piece stands on a square
    MOVE_LEGAL = "move_legal"        # 2  a referenced move is legal
    MOVE_PROPERTY = "move_property"  # 3  a move captures / gives check
    ATTACK = "attack"                # 4  a square is attacked / defended
    PIN = "pin"                      # 5  a piece is pinned
    MATERIAL = "material"            # 6  material balance after a line
    MOBILITY = "mobility"            # 7  a piece has no legal moves
    STRUCTURE = "structure"          # 8  doubled / isolated / passed pawns
    FORCED = "forced"                # 9  the opponent has no real alternative   [engine]
    CONSEQUENCE = "consequence"      # 10 evaluation after a line                [engine]
    HANGS = "hangs"                  # 11 a move loses material                  [engine]
    UNVERIFIABLE = "unverifiable"    # 12 counted, never scored


ENGINE_TYPES = {ClaimType.FORCED, ClaimType.CONSEQUENCE, ClaimType.HANGS}


class ClaimVerdict(str, Enum):
    TRUE = "true"
    FALSE = "false"
    SKIPPED = "skipped"     # engine claim with no table available
    UNSCORED = "unscored"   # UNVERIFIABLE class


@dataclass
class Claim:
    type: ClaimType
    text: str                 # the exact span that produced the claim
    verdict: ClaimVerdict = ClaimVerdict.SKIPPED
    detail: str = ""          # why it failed, for auditing
    span: tuple[int, int] = (0, 0)

    @property
    def scored(self) -> bool:
        return self.verdict in (ClaimVerdict.TRUE, ClaimVerdict.FALSE)


@dataclass
class VerificationReport:
    fen: str
    claims: list[Claim] = field(default_factory=list)

    # --- aggregate views used by the reward and the metrics -----------------
    @property
    def scored_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.scored]

    @property
    def n_scored(self) -> int:
        return len(self.scored_claims)

    @property
    def n_true(self) -> int:
        return sum(c.verdict is ClaimVerdict.TRUE for c in self.claims)

    @property
    def n_false(self) -> int:
        return sum(c.verdict is ClaimVerdict.FALSE for c in self.claims)

    @property
    def precision(self) -> float:
        """Fraction of scorable claims that are true. 1.0 when nothing was asserted --
        which is exactly why the reward must pair this with a coverage term."""
        return self.n_true / self.n_scored if self.n_scored else 1.0

    def violations(self, *types: ClaimType) -> list[Claim]:
        want = set(types) if types else None
        return [c for c in self.claims
                if c.verdict is ClaimVerdict.FALSE and (want is None or c.type in want)]

    @property
    def has_hard_violation(self) -> bool:
        """Occupancy and legality errors are hard failures: the model asserted something
        about the board that is simply not so."""
        return bool(self.violations(ClaimType.OCCUPANCY, ClaimType.MOVE_LEGAL))

    def counts_by_type(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for c in self.claims:
            t, f = out.get(c.type.value, (0, 0))
            out[c.type.value] = (t + (c.verdict is ClaimVerdict.TRUE),
                                 f + (c.verdict is ClaimVerdict.FALSE))
        return out


# --------------------------------------------------------------------------- #
# Grammar
# --------------------------------------------------------------------------- #

_P = r"(?P<piece>king|queen|rook|bishop|knight|pawn)"
_SQ = r"[a-h][1-8]"
_COL = r"(?:(?P<colour>white|black)(?:'s)?\s+)?"

PIECE_TYPES = {
    "king": chess.KING, "queen": chess.QUEEN, "rook": chess.ROOK,
    "bishop": chess.BISHOP, "knight": chess.KNIGHT, "pawn": chess.PAWN,
}

RX = {
    # 1. "the knight on f6", "White's rook on e1", "black king at g8"
    "occupancy": re.compile(rf"\b{_COL}{_P}s?\s+(?:on|at)\s+(?P<sq>{_SQ})\b", re.I),

    # 2. UCI only. Bare squares are never treated as moves.
    "uci": re.compile(rf"\b(?P<uci>{_SQ}{_SQ}[qrbn]?)\b"),

    # 2a. Long algebraic: "Rd8-d5", "d2-d4". Matched *before* SAN so the head is not
    #     mistaken for a SAN move to the origin square.
    "long_alg": re.compile(rf"\b[KQRBN]?(?P<from>{_SQ})[-x](?P<to>{_SQ})(?P<promo>=?[QRBN])?\b"),

    # 2b. SAN, only when unambiguous: needs a piece letter, a capture, castling,
    #     a promotion or a check marker. "e4" alone stays a square reference.
    "san": re.compile(
        r"\b(?P<san>O-O(?:-O)?|"
        r"(?:[KQRBN][a-h1-8]?x?[a-h][1-8]|[a-h]x[a-h][1-8])(?:=[QRBN])?[+#]?|"
        r"[a-h][1-8](?:=[QRBN])[+#]?)\b"),

    # 4. "the bishop on d3 attacks h7"
    "attacks_from": re.compile(
        rf"\b{_COL}{_P}\s+on\s+(?P<from>{_SQ})\s+"
        rf"(?:attacks|hits|eyes|targets|bears\s+down\s+on)\s+(?P<to>{_SQ})\b", re.I),

    # 4b. "h7 is defended by the king", "e5 is attacked"
    "square_rel": re.compile(
        rf"\b(?P<sq>{_SQ})\s+is\s+(?P<rel>defended|protected|guarded|attacked|"
        rf"undefended|unprotected|hanging|loose)\b", re.I),

    # 5. "the knight on f6 is pinned"
    "pin": re.compile(rf"\b{_COL}{_P}\s+on\s+(?P<sq>{_SQ})\s+is\s+(?:absolutely\s+)?pinned\b",
                      re.I),

    # 7. "the knight on a4 has no squares"
    "mobility": re.compile(
        rf"\b{_COL}{_P}\s+on\s+(?P<sq>{_SQ})\s+has\s+no\s+"
        rf"(?:legal\s+)?(?:moves|squares|retreat)\b", re.I),

    # 8. structure
    "doubled": re.compile(r"\bdoubled\s+(?:(?P<colour>white|black)\s+)?pawns?\s+"
                          r"on\s+the\s+(?P<file>[a-h])[- ]file\b", re.I),
    "isolated": re.compile(rf"\bisolated\s+pawn\s+on\s+(?P<sq>{_SQ})\b", re.I),
    "passed": re.compile(rf"\bpassed\s+pawn\s+on\s+(?P<sq>{_SQ})\b", re.I),

    # 12. vague strategic prose -- counted, never scored
    "vague": re.compile(
        r"\b(?:better\s+coordination|more\s+active|improves?\s+the\s+position|"
        r"greater\s+control|more\s+influence|strategic(?:ally)?\s+\w+|"
        r"long[- ]term\s+(?:pressure|advantage|plan)|positional(?:ly)?\s+\w+|"
        r"initiative|harmonious|dynamic\s+potential)\b", re.I),
}


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #

def _sq(name: str) -> int:
    return chess.parse_square(name.lower())


def _colour(name: str | None) -> bool | None:
    if not name:
        return None
    return name.lower() == "white"


def _check_occupancy(b: chess.Board, m: re.Match) -> tuple[bool, str]:
    want_type = PIECE_TYPES[m.group("piece").lower()]
    want_col = _colour(m.group("colour"))
    pc = b.piece_at(_sq(m.group("sq")))
    if pc is None:
        return False, f"{m.group('sq')} is empty"
    if pc.piece_type != want_type:
        return False, f"{m.group('sq')} holds {chess.piece_name(pc.piece_type)}"
    if want_col is not None and pc.color != want_col:
        return False, f"{m.group('sq')} piece is the other colour"
    return True, ""


def _check_uci(b: chess.Board, uci: str) -> tuple[bool, str]:
    try:
        mv = chess.Move.from_uci(uci)
    except ValueError:
        return False, "unparseable"
    if mv in b.legal_moves:
        return True, ""
    if len(uci) == 4:
        try:
            if chess.Move.from_uci(uci + "q") in b.legal_moves:
                return True, ""
        except ValueError:
            pass
    return False, "illegal in this position"


def _check_san(b: chess.Board, san: str) -> tuple[bool, str]:
    try:
        b.parse_san(san)
        return True, ""
    except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError) as e:
        return False, type(e).__name__
    except ValueError as e:
        return False, str(e)[:40]


def _check_attacks_from(b: chess.Board, m: re.Match) -> tuple[bool, str]:
    frm, to = _sq(m.group("from")), _sq(m.group("to"))
    pc = b.piece_at(frm)
    if pc is None:
        return False, f"{m.group('from')} is empty"
    if pc.piece_type != PIECE_TYPES[m.group("piece").lower()]:
        return False, f"{m.group('from')} holds {chess.piece_name(pc.piece_type)}"
    if to not in b.attacks(frm):
        return False, f"{m.group('from')} does not attack {m.group('to')}"
    return True, ""


def _check_square_rel(b: chess.Board, m: re.Match) -> tuple[bool, str]:
    """Convention, documented and applied consistently:
       - "defended/protected/covered/guarded": >=1 attacker of the *owner's* colour.
       - "attacked": >=1 attacker of the colour opposing the occupant.
       - "undefended/hanging/loose": occupied and zero defenders.
       For an empty square, the occupant's colour is taken to be the side to move.
    """
    sq = _sq(m.group("sq"))
    rel = m.group("rel").lower()
    pc = b.piece_at(sq)
    owner = pc.color if pc is not None else b.turn

    defenders = len(b.attackers(owner, sq))
    attackers = len(b.attackers(not owner, sq))

    if rel in ("defended", "protected", "guarded"):
        return (defenders > 0), ("" if defenders else f"{m.group('sq')} has no defenders")
    if rel == "attacked":
        return (attackers > 0), ("" if attackers else f"{m.group('sq')} is not attacked")
    # undefended / unprotected / hanging / loose
    if pc is None:
        return False, f"{m.group('sq')} is empty, cannot hang"
    return (defenders == 0), ("" if defenders == 0 else f"{m.group('sq')} has {defenders} defender(s)")


def _check_pin(b: chess.Board, m: re.Match) -> tuple[bool, str]:
    sq = _sq(m.group("sq"))
    pc = b.piece_at(sq)
    if pc is None:
        return False, f"{m.group('sq')} is empty"
    if pc.piece_type != PIECE_TYPES[m.group("piece").lower()]:
        return False, f"{m.group('sq')} holds {chess.piece_name(pc.piece_type)}"
    return (b.is_pinned(pc.color, sq)), ("" if b.is_pinned(pc.color, sq) else "not pinned")


def _check_mobility(b: chess.Board, m: re.Match) -> tuple[bool, str]:
    sq = _sq(m.group("sq"))
    pc = b.piece_at(sq)
    if pc is None:
        return False, f"{m.group('sq')} is empty"
    if pc.color != b.turn:
        # Mobility of the side not to move: count pseudo-legal moves on a null-moved board.
        probe = b.copy(stack=False)
        probe.turn = pc.color
        n = sum(1 for mv in probe.legal_moves if mv.from_square == sq)
    else:
        n = sum(1 for mv in b.legal_moves if mv.from_square == sq)
    return (n == 0), ("" if n == 0 else f"{n} legal move(s) available")


def _check_doubled(b: chess.Board, m: re.Match) -> tuple[bool, str]:
    f = ord(m.group("file").lower()) - ord("a")
    col = _colour(m.group("colour"))
    cols = [chess.WHITE, chess.BLACK] if col is None else [col]
    for c in cols:
        n = sum(1 for sq in chess.SquareSet(chess.BB_FILES[f])
                if (p := b.piece_at(sq)) and p.piece_type == chess.PAWN and p.color == c)
        if n >= 2:
            return True, ""
    return False, "no side has two pawns on that file"


def _check_isolated(b: chess.Board, m: re.Match) -> tuple[bool, str]:
    sq = _sq(m.group("sq"))
    pc = b.piece_at(sq)
    if pc is None or pc.piece_type != chess.PAWN:
        return False, "no pawn there"
    f = chess.square_file(sq)
    for nf in (f - 1, f + 1):
        if 0 <= nf < 8:
            for s in chess.SquareSet(chess.BB_FILES[nf]):
                p = b.piece_at(s)
                if p and p.piece_type == chess.PAWN and p.color == pc.color:
                    return False, "friendly pawn on an adjacent file"
    return True, ""


def _check_passed(b: chess.Board, m: re.Match) -> tuple[bool, str]:
    sq = _sq(m.group("sq"))
    pc = b.piece_at(sq)
    if pc is None or pc.piece_type != chess.PAWN:
        return False, "no pawn there"
    f, r = chess.square_file(sq), chess.square_rank(sq)
    ahead = range(r + 1, 8) if pc.color == chess.WHITE else range(0, r)
    for nf in (f - 1, f, f + 1):
        if not 0 <= nf < 8:
            continue
        for nr in ahead:
            p = b.piece_at(chess.square(nf, nr))
            if p and p.piece_type == chess.PAWN and p.color != pc.color:
                return False, f"enemy pawn on {chess.square_name(chess.square(nf, nr))}"
    return True, ""


# --------------------------------------------------------------------------- #
# Extraction + verification
# --------------------------------------------------------------------------- #

_SIMPLE = [
    ("occupancy", ClaimType.OCCUPANCY, _check_occupancy),
    ("attacks_from", ClaimType.ATTACK, _check_attacks_from),
    ("square_rel", ClaimType.ATTACK, _check_square_rel),
    ("pin", ClaimType.PIN, _check_pin),
    ("mobility", ClaimType.MOBILITY, _check_mobility),
    ("doubled", ClaimType.STRUCTURE, _check_doubled),
    ("isolated", ClaimType.STRUCTURE, _check_isolated),
    ("passed", ClaimType.STRUCTURE, _check_passed),
]


def extract_claims(text: str) -> list[Claim]:
    """Parse a trace into claims without checking them (useful for extractor validation)."""
    claims: list[Claim] = []
    for key, ctype, _ in _SIMPLE:
        for m in RX[key].finditer(text):
            claims.append(Claim(ctype, m.group(0), span=m.span()))
    for m in RX["uci"].finditer(text):
        claims.append(Claim(ClaimType.MOVE_LEGAL, m.group("uci"), span=m.span()))
    for m in RX["san"].finditer(text):
        claims.append(Claim(ClaimType.MOVE_LEGAL, m.group("san"), span=m.span()))
    for m in RX["vague"].finditer(text):
        claims.append(Claim(ClaimType.UNVERIFIABLE, m.group(0),
                            verdict=ClaimVerdict.UNSCORED, span=m.span()))
    claims.sort(key=lambda c: c.span)
    return claims


def verify_trace(fen: str, text: str, engine_table: dict[str, int] | None = None,
                 *, verify_engine_claims: bool = True, strict_moves: bool = False,
                 root_spans: list[tuple[int, int]] | None = None) -> VerificationReport:
    """Verify every extractable claim in `text` against the position `fen`.

    `engine_table` maps UCI -> centipawns from the mover's point of view for *every*
    legal move (see engine.py). When absent, engine-backed claim types are recorded as
    SKIPPED rather than guessed at -- this is the mode used by test-time reranking.

    Move handling is deliberately conservative. A move written in prose is usually a move
    *inside a variation* ("if White defends with Re1, Black plays Qxe1"), and checking it
    against the root position would punish correct analysis. So:

      * moves inside `root_spans` (the <candidates> heads and the <move> tag) are checked
        against the root position;
      * every other move reference is extracted and left UNSCORED unless
        `strict_moves=True`.

    Continuations are checked by replay instead -- see `verify_line`.
    """
    board = chess.Board(fen)
    report = VerificationReport(fen=fen)

    for key, ctype, checker in _SIMPLE:
        for m in RX[key].finditer(text):
            ok, why = checker(board, m)
            report.claims.append(Claim(
                ctype, m.group(0),
                ClaimVerdict.TRUE if ok else ClaimVerdict.FALSE, why, m.span()))

    report.claims.extend(_verify_moves(board, text, strict_moves=strict_moves,
                                       root_spans=root_spans))

    for m in RX["vague"].finditer(text):
        report.claims.append(Claim(ClaimType.UNVERIFIABLE, m.group(0),
                                   ClaimVerdict.UNSCORED, "", m.span()))

    if engine_table and verify_engine_claims:
        report.claims.extend(_verify_engine_claims(board, text, engine_table))

    report.claims.sort(key=lambda c: c.span)
    return report


_HANGS = re.compile(r"\b(?P<uci>[a-h][1-8][a-h][1-8][qrbn]?)\b[^.]{0,60}?"
                    r"\b(?:hangs|loses\s+(?:a\s+)?(?:piece|material|the\s+\w+)|"
                    r"drops\s+(?:a\s+)?piece|blunders)\b", re.I)
_WINNING = re.compile(r"\b(?P<uci>[a-h][1-8][a-h][1-8][qrbn]?)\b[^.]{0,60}?"
                      r"\b(?:is\s+winning|wins|is\s+decisive|is\s+crushing)\b", re.I)

HANG_THRESHOLD_CP = 200   # a move this much worse than best is fairly called a blunder
WIN_THRESHOLD_CP = 300    # an evaluation this high is fairly called winning


def _verify_engine_claims(board: chess.Board, text: str,
                          table: dict[str, int]) -> list[Claim]:
    """Types 9-11. Cheap: every check is a dict lookup against the precomputed table."""
    out: list[Claim] = []
    if not table:
        return out
    best = max(table.values())

    for rx, ctype, want in ((_HANGS, ClaimType.HANGS, "hangs"),
                            (_WINNING, ClaimType.CONSEQUENCE, "winning")):
        for m in rx.finditer(text):
            u = m.group("uci").lower()
            cp = table.get(u)
            if cp is None:
                out.append(Claim(ctype, m.group(0), ClaimVerdict.FALSE,
                                 "move not legal here", m.span()))
                continue
            if want == "hangs":
                ok = (best - cp) >= HANG_THRESHOLD_CP
                why = "" if ok else f"only {best - cp} cp worse than best"
            else:
                ok = cp >= WIN_THRESHOLD_CP
                why = "" if ok else f"evaluation is {cp} cp"
            out.append(Claim(ctype, m.group(0),
                             ClaimVerdict.TRUE if ok else ClaimVerdict.FALSE,
                             why, m.span()))
    return out


# --------------------------------------------------------------------------- #
# Move references and line replay
# --------------------------------------------------------------------------- #

def _in_spans(span: tuple[int, int], spans: list[tuple[int, int]] | None) -> bool:
    if spans is None:
        return False
    return any(a <= span[0] and span[1] <= b for a, b in spans)


def _verify_moves(board: chess.Board, text: str, *, strict_moves: bool,
                  root_spans: list[tuple[int, int]] | None) -> list[Claim]:
    """Extract move references. Only those asserted about the root position are scored.

    Rationale is in `verify_trace`: a move in prose is usually a move deeper in a line,
    and scoring it against the root produces false violations. Measured on the legacy
    corpus, 221 of 255 apparent SAN violations were of exactly this kind.
    """
    out: list[Claim] = []
    consumed: list[tuple[int, int]] = []

    def emit(ctype: ClaimType, text_: str, span: tuple[int, int], scorable: bool,
             checker) -> None:
        if scorable:
            ok, why = checker()
            out.append(Claim(ctype, text_,
                             ClaimVerdict.TRUE if ok else ClaimVerdict.FALSE, why, span))
        else:
            out.append(Claim(ctype, text_, ClaimVerdict.UNSCORED,
                             "variation move; not asserted of the root position", span))

    # Long algebraic first, so its head is not re-read as SAN.
    for m in RX["long_alg"].finditer(text):
        consumed.append(m.span())
        uci = (m.group("from") + m.group("to")).lower()
        scorable = strict_moves or _in_spans(m.span(), root_spans)
        emit(ClaimType.MOVE_LEGAL, m.group(0), m.span(), scorable,
             lambda u=uci: _check_uci(board, u))

    seen: set[str] = set()
    for m in RX["uci"].finditer(text):
        if any(a <= m.start() < b for a, b in consumed):
            continue
        u = m.group("uci").lower()
        scorable = strict_moves or _in_spans(m.span(), root_spans)
        if scorable and u in seen:
            continue
        if scorable:
            seen.add(u)
        emit(ClaimType.MOVE_LEGAL, u, m.span(), scorable, lambda u=u: _check_uci(board, u))

    for m in RX["san"].finditer(text):
        if any(a <= m.start() < b for a, b in consumed):
            continue
        s_ = m.group("san")
        scorable = strict_moves or _in_spans(m.span(), root_spans)
        emit(ClaimType.MOVE_LEGAL, s_, m.span(), scorable, lambda s_=s_: _check_san(board, s_))

    return out


def verify_line(fen: str, moves: list[str]) -> tuple[int, str]:
    """Replay a continuation. Returns (n_legal_plies, reason_for_stopping).

    This is the correct way to check a variation: push each move in turn rather than
    testing it against the root. A line is fully legal iff n_legal_plies == len(moves).
    """
    board = chess.Board(fen)
    for i, u in enumerate(moves):
        try:
            mv = chess.Move.from_uci(u.lower())
        except ValueError:
            return i, f"ply {i + 1} ({u}) is unparseable"
        if mv not in board.legal_moves:
            if len(u) == 4:
                try:
                    alt = chess.Move.from_uci(u.lower() + "q")
                except ValueError:
                    alt = None
                if alt is not None and alt in board.legal_moves:
                    board.push(alt)
                    continue
            return i, f"ply {i + 1} ({u}) is illegal"
        board.push(mv)
    return len(moves), ""


def verify_structured_trace(fen: str, text: str,
                            engine_table: dict[str, int] | None = None) -> VerificationReport:
    """Verify a trace in the structured format defined in prompts.py.

    Section-aware, which is what makes the move checks sound:
      * `<read>` and `<choice>` make claims about the root position;
      * `<candidates>` heads are moves playable now, so they are root-scored;
      * each candidate's continuation is verified by replay.
    """
    from chessr.prompts import parse_trace  # local import: prompts imports boards only

    tr = parse_trace(text)
    root_spans: list[tuple[int, int]] = []
    for c in tr.candidates:
        i = text.find(c.raw)
        if i >= 0:
            head = text.find(c.move, i)
            if head >= 0:
                root_spans.append((head, head + len(c.move)))
    m = re.search(r"<move>(.*?)</move>", text, re.S | re.I)
    if m:
        root_spans.append(m.span(1))

    report = verify_trace(fen, text, engine_table, root_spans=root_spans)

    for c in tr.candidates:
        if not c.line:
            continue
        n_ok, why = verify_line(fen, [c.move] + c.line)
        report.claims.append(Claim(
            ClaimType.MOVE_LEGAL, f"line {c.move}: {' '.join(c.line)}",
            ClaimVerdict.TRUE if not why else ClaimVerdict.FALSE, why, (0, 0)))
    return report
