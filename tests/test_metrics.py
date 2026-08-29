"""Metrics are the only thing standing between saved records and the paper's tables,
so the record->metric path is pinned."""
import json, math, tempfile, os
import chess
import pytest

from chessr.metrics import (score_record, aggregate, paired_test, holm_bonferroni,
                            faithfulness, load_records)

START = chess.STARTING_FEN
TABLE = {"e2e4": 30, "d2d4": 25, "g1f3": 20, "a2a3": -10, "h2h4": -40}

TRACE = """<read>
The knight on g1 is undeveloped and the king on e1 has not castled.
</read>
<candidates>
1. e2e4 | e7e5 | takes the centre
2. d2d4 | d7d5 | solid
3. a2a3 | e7e5 | slow
</candidates>
<choice>e2e4 is best.</choice>
<move>e2e4</move>"""


def rec(variant="base", completions=None, move="e2e4", **kw):
    c = completions or [TRACE.replace("<move>e2e4</move>", f"<move>{move}</move>")]
    base = dict(run_id="r", model="m", adapter=None, variant=variant,
                benchmark="holdout", item_id="i1", fen=START, prompt="p", system="s",
                completions=c, n_tokens=[120] * len(c), finish_reason=["stop"] * len(c),
                gold_moves=["e2e4"], solution=[], rating=1500, themes=["fork"],
                engine_table=TABLE, meta={"band": "near_tie"}, decode={}, ts=0.0)
    base.update(kw)
    return base


class TestScoring:
    def test_correct_move_scores_top1(self):
        r = score_record(rec())
        assert r["top1"] == 1.0 and r["top1_engine"] == 1.0
        assert r["wp_loss"] == pytest.approx(0.0)
        assert r["illegal"] == 0.0 and r["no_move"] == 0.0

    def test_suboptimal_move_is_graded_not_binary(self):
        r = score_record(rec(move="g1f3"))
        assert r["top1_engine"] == 0.0
        assert r["top3_engine"] == 1.0          # still a top-3 move
        assert 0.0 < r["wp_loss"] < 0.1         # graded, not a cliff

    def test_illegal_move_flagged(self):
        """An illegal move is an illegal move, not a missing one. The earlier assertion
        here encoded the bug: normalisation maps both to move=None, which reported a 0%
        illegal rate for a model that was proposing illegal moves."""
        r = score_record(rec(move="e2e5"))
        assert r["illegal"] == 1.0 and r["no_move"] == 0.0

    def test_grounding_recorded(self):
        r = score_record(rec())
        assert r["n_claims"] > 0
        assert r["claim_precision"] == 1.0
        assert r["hard_violation"] == 0.0

    def test_false_claim_detected(self):
        bad = TRACE.replace("knight on g1", "knight on e4")
        r = score_record(rec(completions=[bad]))
        assert r["n_false"] >= 1 and r["hard_violation"] == 1.0

    def test_rating_bucket(self):
        assert score_record(rec(rating=2500))["rating_bucket"] == "2400+"
        assert score_record(rec(rating=None))["rating_bucket"] == "unrated"

    def test_rerank_columns_appear(self):
        good = TRACE
        bad = TRACE.replace("knight on g1", "knight on e4").replace("<move>e2e4</move>",
                                                                    "<move>a2a3</move>")
        r = score_record(rec(completions=[bad, good]), rerank_n=2)
        assert r["rerank2_move"] == "e2e4"      # picks the grounded trace
        assert r["rerank2_top1"] == 1.0


class TestAggregation:
    def test_grouping_by_band(self):
        rows = [score_record(rec()), score_record(rec(meta={"band": "tactical"}))]
        out = aggregate(rows, ["top1_engine"], "gap_band", ci=False)
        assert out and all("n" in v for v in out.values())

    def test_themes_group_expands_lists(self):
        rows = [score_record(rec(themes=["fork", "pin"]))]
        out = aggregate(rows, ["top1_engine"], "themes", ci=False)
        assert set(out) == {"fork", "pin"}

    def test_bootstrap_ci_brackets_the_mean(self):
        rows = [score_record(rec()) for _ in range(20)]
        out = aggregate(rows, ["top1_engine"])
        m = out["top1_engine"]
        assert m["lo"] <= m["mean"] <= m["hi"]

    def test_nan_columns_do_not_poison_means(self):
        rows = [score_record(rec(engine_table=None)), score_record(rec())]
        out = aggregate(rows, ["top1_engine"], ci=False)
        assert out["top1_engine"] == 1.0


class TestStatistics:
    def test_paired_test_is_paired_on_item_id(self):
        a = [score_record(rec(item_id=f"i{i}")) for i in range(30)]
        b = [score_record(rec(item_id=f"i{i}", move="h2h4")) for i in range(30)]
        out = paired_test(a, b, "top1_engine")
        assert out["n"] == 30 and out["diff"] > 0 and out["p"] < 0.05

    def test_paired_test_ignores_unmatched_items(self):
        a = [score_record(rec(item_id="x"))]
        b = [score_record(rec(item_id="y"))]
        assert paired_test(a, b, "top1_engine")["n"] == 0

    def test_holm_bonferroni_is_more_conservative_than_raw(self):
        out = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.9})
        assert out["a"]["significant"]
        assert not out["c"]["significant"]
        assert out["b"]["threshold"] < 0.05


class TestFaithfulness:
    def test_move_change_under_perturbation_is_counted(self):
        recs = [rec(variant="base", move="e2e4"),
                rec(variant="perturbed", move="d2d4")]
        f = faithfulness(recs)
        assert f["perturbation_n"] == 1
        assert f["perturbation_sensitivity"] == 1.0

    def test_unmoved_answer_reads_as_unfaithful(self):
        recs = [rec(variant="base", move="e2e4"),
                rec(variant="perturbed", move="e2e4")]
        assert faithfulness(recs)["perturbation_sensitivity"] == 0.0

    def test_reasoning_necessity_is_a_delta(self):
        recs = [rec(variant="base", move="e2e4"),
                rec(variant="no_reasoning", move="h2h4")]
        f = faithfulness(recs)
        assert f["base_acc"] == 1.0 and f["no_reasoning_acc"] == 0.0
        assert f["reasoning_necessity"] == 1.0


def test_records_roundtrip_through_disk():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps(rec()) + "\n")
        fh.write(json.dumps(rec(variant="perturbed")) + "\n")
        path = fh.name
    try:
        assert len(load_records(path)) == 2
        assert len(load_records(path, "base")) == 1
    finally:
        os.unlink(path)


class TestBenchmarkRouting:
    """Benchmarks carry rows whose position field is prose or empty. Those must be
    filtered, not crash the harness -- this killed the first eval sweep."""

    def test_qa_items_use_their_own_prompt_and_only_base(self):
        from chessr.benchmarks import Item
        from chessr.evalsuite import build_variant_prompts
        qa = Item(id="q1", benchmark="chessqa_structural", fen="",
                  question="List the pieces.", answer="White King: ['g1']")
        base = build_variant_prompts([qa], "base")
        assert len(base) == 1 and base[0][1] == "List the pieces."
        for v in ("perturbed", "no_reasoning"):
            assert build_variant_prompts([qa], v) == []

    def test_unusable_fen_is_dropped(self):
        from chessr.benchmarks import Item
        from chessr.evalsuite import build_variant_prompts
        junk = Item(id="j", benchmark="x", fen="structural_piece_0000")
        assert build_variant_prompts([junk], "base") == []

    def test_valid_position_still_routes_normally(self):
        from chessr.benchmarks import Item
        from chessr.evalsuite import build_variant_prompts
        it = Item(id="ok", benchmark="holdout", fen=START)
        out = build_variant_prompts([it], "base")
        assert len(out) == 1 and "FEN:" in out[0][1]


class TestIllegalVsNoMove:
    """An illegal move and no move at all are different failures. Normalisation maps both
    to move=None, so they must be separated by whether the model wrote anything."""

    def test_illegal_move_counts_as_illegal_not_missing(self):
        r = rec(completions=[TRACE.replace("<move>e2e4</move>", "<move>e2e5</move>")])
        row = score_record(r)
        assert row["illegal"] == 1.0 and row["no_move"] == 0.0
        assert row["produced_a_move"] == 1.0

    def test_absent_move_tag_counts_as_missing(self):
        r = rec(completions=[TRACE.replace("<move>e2e4</move>", "")])
        row = score_record(r)
        assert row["no_move"] == 1.0 and row["illegal"] == 0.0

    def test_legal_move_is_neither(self):
        row = score_record(rec())
        assert row["illegal"] == 0.0 and row["no_move"] == 0.0
