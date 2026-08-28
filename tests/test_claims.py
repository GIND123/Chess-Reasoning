"""The verifier is the instrument the whole project rests on, so it gets real tests
with hand-checked positions rather than smoke tests."""
import pytest
import chess

from chessr.claims import (ClaimType, ClaimVerdict, verify_trace, verify_line,
                           verify_structured_trace, extract_claims)

START = chess.STARTING_FEN
# White: Ke1 Qd1 Rf1 Bc4 Nf3 pawns e4 d3; Black: Ke8 Qd8 Nf6 Bc5 pawns e5 d6
ITALIAN = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 5"


def _of(report, ctype):
    """Verdicts for one claim type. A sentence often asserts several checkable facts, so
    total counts are the wrong thing to assert on."""
    return [c.verdict for c in report.claims if c.type is ctype]


class TestOccupancy:
    def test_true_claim(self):
        r = verify_trace(START, "the knight on g1 supports the centre")
        assert r.n_true == 1 and r.n_false == 0

    def test_false_claim_empty_square(self):
        r = verify_trace(START, "the knight on e4 is strong")
        assert r.n_false == 1
        assert "empty" in r.claims[0].detail

    def test_wrong_piece_type(self):
        r = verify_trace(START, "the queen on g1 dominates")
        assert r.n_false == 1

    def test_colour_is_checked(self):
        assert verify_trace(START, "the white knight on g1").n_true == 1
        assert verify_trace(START, "the black knight on g1").n_false == 1

    def test_at_is_a_synonym_for_on(self):
        assert verify_trace(START, "the rook at a1").n_true == 1


class TestMoves:
    def test_prose_moves_are_not_scored_against_the_root(self):
        """A move inside a variation must not be checked against the root position.
        Measured on the legacy corpus, 221 of 255 apparent violations were of this kind."""
        r = verify_trace(START, "if Black replies Nf6, White plays Qxe1 next")
        assert all(c.verdict is ClaimVerdict.UNSCORED
                   for c in r.claims if c.type is ClaimType.MOVE_LEGAL)

    def test_strict_mode_scores_them(self):
        r = verify_trace(START, "White plays Qxe1", strict_moves=True)
        assert any(c.verdict is ClaimVerdict.FALSE
                   for c in r.claims if c.type is ClaimType.MOVE_LEGAL)

    def test_long_algebraic_is_one_move_not_a_san_head(self):
        """'Rd8-d5' must not be read as SAN 'Rd8'."""
        r = verify_trace(START, "the move e2-e4 opens the centre", strict_moves=True)
        legal = [c for c in r.claims if c.type is ClaimType.MOVE_LEGAL]
        assert len(legal) == 1
        assert legal[0].verdict is ClaimVerdict.TRUE

    def test_bare_squares_are_never_moves(self):
        r = verify_trace(START, "the e4 square and the d5 square matter")
        assert not [c for c in r.claims if c.type is ClaimType.MOVE_LEGAL]

    def test_uci_promotion_defaults_to_queen(self):
        fen = "8/P7/8/8/8/8/8/K6k w - - 0 1"
        r = verify_trace(fen, "a7a8 promotes", strict_moves=True)
        assert r.n_false == 0


class TestLineReplay:
    def test_legal_line(self):
        assert verify_line(START, ["e2e4", "e7e5", "g1f3"]) == (3, "")

    def test_illegal_at_ply_three(self):
        n, why = verify_line(START, ["e2e4", "e7e5", "e2e4"])
        assert n == 2 and "ply 3" in why

    def test_unparseable(self):
        n, why = verify_line(START, ["zz99"])
        assert n == 0 and "unparseable" in why


class TestRelations:
    def test_defended(self):
        # e5 is defended by the black knight on c6 and the d6 pawn.
        assert verify_trace(ITALIAN, "e5 is defended").n_true == 1

    def test_attacked_false_when_nothing_attacks(self):
        # a6 *is* attacked in the start position (by the b7 pawn), so use e4, which
        # nothing black attacks.
        r = verify_trace(START, "e4 is attacked")
        assert _of(r, ClaimType.ATTACK) == [ClaimVerdict.FALSE]

    def test_attacked_true_when_something_does(self):
        assert _of(verify_trace(START, "a6 is attacked"), ClaimType.ATTACK) == [ClaimVerdict.TRUE]

    def test_pin_detection(self):
        # Black knight on f6 pinned against the king by a bishop on g5.
        fen = "rnbqkb1r/pppp1ppp/5n2/4p1B1/4P3/8/PPPP1PPP/RN1QKBNR b KQkq - 0 3"
        assert verify_trace(fen, "the knight on f6 is pinned").n_true == 1
        assert verify_trace(START, "the knight on g1 is pinned").n_false == 1

    def test_attacks_from(self):
        # A compound sentence asserts two checkable things: that a bishop stands on c4,
        # and that it attacks f7. Both are counted, so assert per claim type.
        assert _of(verify_trace(ITALIAN, "the bishop on c4 attacks f7"),
                   ClaimType.ATTACK) == [ClaimVerdict.TRUE]
        assert _of(verify_trace(ITALIAN, "the bishop on c4 attacks h7"),
                   ClaimType.ATTACK) == [ClaimVerdict.FALSE]


class TestStructure:
    def test_doubled_pawns(self):
        fen = "4k3/8/8/8/8/2P5/2P5/4K3 w - - 0 1"
        assert verify_trace(fen, "doubled pawns on the c-file").n_true == 1
        assert verify_trace(START, "doubled pawns on the c-file").n_false == 1

    def test_isolated_pawn(self):
        fen = "4k3/8/8/8/8/8/3P4/4K3 w - - 0 1"
        r = verify_trace(fen, "isolated pawn on d2")
        assert _of(r, ClaimType.STRUCTURE) == [ClaimVerdict.TRUE]
        assert _of(r, ClaimType.OCCUPANCY) == [ClaimVerdict.TRUE]   # compound claim

    def test_passed_pawn(self):
        fen = "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1"
        assert _of(verify_trace(fen, "passed pawn on d5"),
                   ClaimType.STRUCTURE) == [ClaimVerdict.TRUE]

    def test_not_passed_when_an_enemy_pawn_blocks(self):
        fen = "4k3/3p4/8/3P4/8/8/8/4K3 w - - 0 1"
        assert _of(verify_trace(fen, "passed pawn on d5"),
                   ClaimType.STRUCTURE) == [ClaimVerdict.FALSE]


class TestPrecisionSemantics:
    def test_silence_scores_one(self):
        """Precision alone is maximised by saying nothing -- which is exactly why the
        reward pairs it with a coverage term."""
        assert verify_trace(START, "This looks promising.").precision == 1.0

    def test_vague_prose_is_counted_not_scored(self):
        r = verify_trace(START, "White has better coordination and the initiative")
        vague = [c for c in r.claims if c.type is ClaimType.UNVERIFIABLE]
        assert vague and all(c.verdict is ClaimVerdict.UNSCORED for c in vague)
        assert r.n_scored == 0

    def test_hard_violation_flag(self):
        assert verify_trace(START, "the knight on e4").has_hard_violation
        assert not verify_trace(START, "the knight on g1").has_hard_violation


class TestStructured:
    TRACE = """<read>
The knight on g1 and the bishop on f1 are undeveloped. The king on e1 has not castled.
</read>
<candidates>
1. e2e4 | e7e5 g1f3 | takes the centre
2. d2d4 | d7d5 | solid
3. g1f3 | g8f6 | flexible
</candidates>
<choice>
e2e4 is best because it opens lines fastest.
</choice>
<move>e2e4</move>"""

    def test_candidate_heads_are_root_scored(self):
        r = verify_structured_trace(START, self.TRACE)
        mv = [c for c in r.claims if c.type is ClaimType.MOVE_LEGAL and c.scored]
        assert mv, "candidate heads and <move> must be scored against the root"
        assert all(c.verdict is ClaimVerdict.TRUE for c in mv)

    def test_illegal_continuation_is_caught_by_replay(self):
        bad = self.TRACE.replace("e7e5 g1f3", "e7e5 e7e5")
        r = verify_structured_trace(START, bad)
        assert any(c.text.startswith("line ") and c.verdict is ClaimVerdict.FALSE
                   for c in r.claims)

    def test_false_read_is_caught(self):
        bad = self.TRACE.replace("knight on g1", "knight on e4")
        assert verify_structured_trace(START, bad).has_hard_violation


def test_extract_without_checking():
    claims = extract_claims("the knight on g1 and the rook on a1")
    assert len(claims) == 2
    assert all(c.type is ClaimType.OCCUPANCY for c in claims)
