import chess
import pytest

from chessr.boards import win_prob, cp_to_wp_loss, band_for_gap, ascii_board
from chessr.prompts import parse_trace, student_prompt
from chessr.reward import r_move, r_coverage, r_format, score_completion
from chessr.rerank import rerank

START = chess.STARTING_FEN
TABLE = {"e2e4": 30, "d2d4": 25, "g1f3": 20, "a2a3": -10, "h2h4": -40}

GOOD = """<read>
The knight on g1 is undeveloped and the king on e1 has not castled.
</read>
<candidates>
1. e2e4 | e7e5 | takes the centre
2. d2d4 | d7d5 | solid
3. a2a3 | e7e5 | slow
</candidates>
<choice>e2e4 is best.</choice>
<move>e2e4</move>"""

TERSE = """<read>The king on e1 is safe.</read>
<candidates>
1. e2e4 | e7e5 | fine
</candidates>
<choice>e2e4.</choice>
<move>e2e4</move>"""


class TestWinProb:
    def test_monotonic_and_centred(self):
        assert win_prob(0) == pytest.approx(0.5)
        assert win_prob(400) > win_prob(0) > win_prob(-400)

    def test_loss_is_non_negative(self):
        assert cp_to_wp_loss(100, 500) == 0.0
        assert cp_to_wp_loss(500, 100) > 0.0

    def test_same_cp_gap_costs_less_when_already_winning(self):
        """The reason the reward is in win probability and not centipawns: this corpus is
        99.3% already-winning positions, where a cp-linear reward is nearly flat."""
        near_equal = cp_to_wp_loss(20, -20)
        already_won = cp_to_wp_loss(920, 880)
        assert near_equal > already_won * 3


class TestMoveReward:
    def test_best_move_scores_one(self):
        r, loss, legal = r_move(START, "e2e4", TABLE)
        assert r == pytest.approx(1.0) and legal and loss == 0.0

    def test_illegal_move_scores_zero(self):
        assert r_move(START, "e2e5", TABLE) == (0.0, 1.0, False)

    def test_missing_move_scores_zero(self):
        assert r_move(START, None, TABLE)[0] == 0.0

    def test_slightly_worse_move_is_partially_rewarded(self):
        r, _, _ = r_move(START, "g1f3", TABLE)
        assert 0.0 < r < 1.0


class TestTraceRewards:
    def test_format_reward_full_marks(self):
        assert r_format(parse_trace(GOOD)) == pytest.approx(1.0)

    def test_format_reward_penalises_missing_candidates(self):
        assert r_format(parse_trace(TERSE)) < 1.0

    def test_coverage_beats_terseness(self):
        """The whole point of the coverage term: a fuller grounded trace must win."""
        assert (score_completion(START, GOOD, TABLE).coverage
                > score_completion(START, TERSE, TABLE).coverage)

    def test_false_claim_incurs_a_penalty(self):
        bad = GOOD.replace("knight on g1", "knight on e4")
        assert score_completion(START, bad, TABLE).penalty > 0
        assert (score_completion(START, bad, TABLE).total
                < score_completion(START, GOOD, TABLE).total)

    def test_illegal_choice_is_penalised(self):
        """An illegal token normalises to no move at all; both are the same failure."""
        bad = GOOD.replace("<move>e2e4</move>", "<move>e2e5</move>")
        b = score_completion(START, bad, TABLE)
        assert b.penalty >= 1.0 and not b.legal and b.chosen is None

    def test_san_and_bare_squares_are_accepted(self):
        """Notation is not a claim about the board, so it is normalised, not punished."""
        for token in ("Nf3", "g1f3", "g1xf3"):
            t = parse_trace(GOOD.replace("<move>e2e4</move>", f"<move>{token}</move>"), START)
            assert t.move == "g1f3", token

    def test_unresolvable_token_is_dropped_not_faked(self):
        t = parse_trace(GOOD.replace("<move>e2e4</move>", "<move>zz99</move>"), START)
        assert t.move is None and t.move_raw == "zz99"


class TestParsing:
    def test_parses_all_sections(self):
        t = parse_trace(GOOD)
        assert t.well_formed
        assert t.move == "e2e4"
        assert len(t.candidates) == 3
        assert t.candidates[0].line == ["e7e5"]

    def test_missing_sections_do_not_raise(self):
        t = parse_trace("nothing structured here")
        assert not t.well_formed and t.move is None

    def test_student_prompt_leaks_nothing(self):
        """The instruction may say "choose the best move"; what it must never carry is
        *which* move, or any engine data. 100% of the legacy prompts carried both."""
        import re
        p = student_prompt(START)
        assert "the best move is" not in p.lower()
        assert not re.search(r"\b[a-h][1-8][a-h][1-8]\b", p), "a move leaked into the prompt"
        for token in ("cp(", "mate(", "engine", "annotation", "line 1"):
            assert token not in p.lower(), f"{token!r} leaked into the prompt"


class TestRerank:
    def test_prefers_the_better_grounded_trace(self):
        bad = GOOD.replace("knight on g1", "knight on e4")
        res = rerank(START, [bad, GOOD])
        assert res.index == 1

    def test_works_without_an_engine_table(self):
        """Reranking must need nothing that is unavailable at test time."""
        assert rerank(START, [GOOD, TERSE]).move == "e2e4"


def test_bands():
    assert band_for_gap(10) == "near_tie"
    assert band_for_gap(50) == "moderate"
    assert band_for_gap(200) == "decisive"
    assert band_for_gap(900) == "tactical"


def test_ascii_board_shape():
    b = ascii_board(START)
    assert b.count("\n") == 16
    assert b.splitlines()[1].startswith("| r |")
