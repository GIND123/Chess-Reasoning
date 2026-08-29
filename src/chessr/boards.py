"""FEN / board utilities and the win-probability scale used by every reward."""
from __future__ import annotations

import math
from dataclasses import dataclass

import chess

# Logistic used throughout. 400 is the Elo-style scale; Lichess uses ~ -0.00368208*cp
# in a sigmoid, which is within a couple of points of this over the range we care about.
WP_SCALE = 400.0
MATE_CP = 10_000  # centipawn value assigned to a forced mate before conversion


def is_fen(fen: str) -> bool:
    """Cheap structural check. Benchmarks carry rows whose position field is prose, a
    task id, or empty, and those must be filtered rather than crash the harness."""
    if not fen or not isinstance(fen, str):
        return False
    parts = fen.split()
    if not parts or parts[0].count("/") != 7:
        return False
    try:
        chess.Board(fen)
        return True
    except Exception:
        return False


def ascii_board(fen: str) -> str:
    """The 8x8 grid used in every prompt. Rank 8 at the top, matching the FEN order."""
    parts = fen.split() if isinstance(fen, str) else []
    if not parts:
        raise ValueError(f"not a FEN: {fen!r}")
    placement = parts[0]
    sep = "+---" * 8 + "+"
    rows = []
    for rank in placement.split("/"):
        cells = "".join("." * int(c) if c.isdigit() else c for c in rank)
        rows.append(f"| {' | '.join(cells)} |")
    out = [sep]
    for r in rows:
        out += [r, sep]
    return "\n".join(out)


def piece_list(fen: str) -> str:
    """Explicit piece inventory. Derived from the FEN, so it adds no information the
    prompt did not already carry -- but it removes the parsing step that every tested
    model fails at (0.0% board-state accuracy in the published literature)."""
    board = chess.Board(fen)
    out = []
    for colour, name in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        parts = []
        for pt in (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP,
                   chess.KNIGHT, chess.PAWN):
            sqs = sorted(chess.square_name(s) for s in board.pieces(pt, colour))
            if sqs:
                parts.append(f"{chess.piece_name(pt)}s: {', '.join(sqs)}"
                             if len(sqs) > 1 else
                             f"{chess.piece_name(pt)}: {sqs[0]}")
        out.append(f"{name} -- " + "; ".join(parts))
    return "\n".join(out)


def side_to_move(fen: str) -> str:
    return "White" if fen.split()[1] == "w" else "Black"


def score_to_cp(score, mate_cp: int = MATE_CP) -> int:
    """python-chess PovScore/Score -> signed centipawns from the mover's point of view."""
    if hasattr(score, "relative"):
        score = score.relative
    if score.is_mate():
        m = score.mate()
        # Closer mates are worth more; sign carries who is mating.
        return (mate_cp - abs(m) * 10) * (1 if m > 0 else -1)
    return int(score.score())


def win_prob(cp: float, scale: float = WP_SCALE) -> float:
    """Centipawns -> win probability in [0, 1], from the mover's point of view."""
    return 1.0 / (1.0 + math.pow(10.0, -float(cp) / scale))


def cp_to_wp_loss(cp_best: float, cp_played: float) -> float:
    """Win-probability lost by playing `cp_played` instead of `cp_best`. Always >= 0."""
    return max(0.0, win_prob(cp_best) - win_prob(cp_played))


@dataclass(frozen=True)
class Band:
    name: str
    lo: float  # inclusive, centipawns
    hi: float  # exclusive


#: Decision-difficulty bands. The gap is |cp(best) - cp(second best)|.
BANDS = (
    Band("near_tie", 0, 30),
    Band("moderate", 30, 100),
    Band("decisive", 100, 300),
    Band("tactical", 300, float("inf")),
)


def band_for_gap(gap_cp: float) -> str:
    for b in BANDS:
        if b.lo <= gap_cp < b.hi:
            return b.name
    return BANDS[-1].name


def phase_of(board: chess.Board) -> str:
    n = len(board.piece_map())
    return "opening" if n >= 28 else ("middlegame" if n >= 15 else "endgame")


def legal_uci(fen: str) -> list[str]:
    return [m.uci() for m in chess.Board(fen).legal_moves]


def normalise_uci(fen: str, uci: str) -> str | None:
    """Return a legal UCI string, adding a queen promotion if that is what was meant."""
    board = chess.Board(fen)
    uci = uci.strip().lower()
    try:
        mv = chess.Move.from_uci(uci)
    except ValueError:
        return None
    if mv in board.legal_moves:
        return mv.uci()
    if len(uci) == 4:
        try:
            mv2 = chess.Move.from_uci(uci + "q")
        except ValueError:
            return None
        if mv2 in board.legal_moves:
            return mv2.uci()
    return None
