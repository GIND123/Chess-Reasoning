"""Benchmark loaders.

Every benchmark is normalised to one record shape so the harness and the metrics never
branch on which benchmark a row came from:

    {id, benchmark, fen, gold_moves[], solution[], rating, themes[], meta{}}

Externally-published benchmarks are included deliberately: results on our own held-out
split alone are not checkable by a reader, and the standard sets are how this work is
placed against Tang et al. (Lichess puzzles), Wang et al. (MATE) and ChessQA.
"""
from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass, field, asdict


@dataclass
class Item:
    id: str
    benchmark: str
    fen: str
    gold_moves: list[str] = field(default_factory=list)   # acceptable best moves (UCI)
    solution: list[str] = field(default_factory=list)     # full line, UCI
    rating: int | None = None
    themes: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Our held-out split
# --------------------------------------------------------------------------- #

def load_holdout(path: str, tables=None, limit: int | None = None) -> list[Item]:
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            r = json.loads(line)
            gold = []
            if tables and (t := tables.get(r["fen"])):
                best = max(t.values())
                gold = [u for u, cp in t.items() if cp == best]
            out.append(Item(id=f"holdout-{i}", benchmark="holdout", fen=r["fen"],
                            gold_moves=gold, meta={"band": r.get("band")}))
    return out


# --------------------------------------------------------------------------- #
# Lichess puzzles -- rating- and theme-stratified
# --------------------------------------------------------------------------- #

def load_lichess_puzzles(n_per_band: int = 250, seed: int = 0,
                         rating_bands=((400, 1200), (1200, 1600), (1600, 2000),
                                       (2000, 2400), (2400, 3200)),
                         themes_wanted: int = 20,
                         n_per_theme: int = 40) -> list[Item]:
    """Two stratifications from one pull: by rating (difficulty curve) and by theme
    (which tactical motifs a model actually knows).

    Lichess convention: `FEN` is the position *before* the opponent's setup move, so the
    puzzle position is after pushing Moves[0], and the solution begins at Moves[1].
    """
    from datasets import load_dataset
    import chess

    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    rng = random.Random(seed)
    by_band: dict[tuple, list] = {b: [] for b in rating_bands}
    by_theme: dict[str, list] = {}
    seen = 0
    for row in ds:
        seen += 1
        if seen > 400_000:
            break
        try:
            rating = int(row["Rating"])
            moves = row["Moves"].split()
            if len(moves) < 2:
                continue
            board = chess.Board(row["FEN"])
            board.push(chess.Move.from_uci(moves[0]))     # setup move
            fen = board.fen()
            themes = row["Themes"]
            if isinstance(themes, str):
                themes = ast.literal_eval(themes) if themes.startswith("[") else themes.split()
        except Exception:
            continue
        item = Item(id=f"lichess-{row['PuzzleId']}", benchmark="lichess_puzzles",
                    fen=fen, gold_moves=[moves[1]], solution=moves[1:],
                    rating=rating, themes=list(themes),
                    meta={"popularity": row.get("Popularity")})
        for b in rating_bands:
            if b[0] <= rating < b[1] and len(by_band[b]) < n_per_band * 4:
                by_band[b].append(item)
        for th in themes:
            by_theme.setdefault(th, [])
            if len(by_theme[th]) < n_per_theme * 3:
                by_theme[th].append(item)

    out, ids = [], set()
    for b, pool in by_band.items():
        rng.shuffle(pool)
        for it in pool[:n_per_band]:
            if it.id not in ids:
                ids.add(it.id); out.append(it)
    top_themes = sorted(by_theme, key=lambda t: -len(by_theme[t]))[:themes_wanted]
    for th in top_themes:
        pool = by_theme[th]; rng.shuffle(pool)
        for it in pool[:n_per_theme]:
            if it.id not in ids:
                ids.add(it.id); out.append(it)
    return out


# --------------------------------------------------------------------------- #
# ChessQA (CSSLab) -- five categories of chess understanding
# --------------------------------------------------------------------------- #

CHESSQA_CONFIGS = ("structural", "motifs", "short_tactics",
                   "position_judgement", "semantic")


def load_chessqa(n_per_config: int = 200, seed: int = 0) -> list[Item]:
    """Loaded as free-form QA: the harness stores the raw answer and metrics grade it,
    so a change of grading rule never needs a re-run."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    rng = random.Random(seed)
    out = []
    for cfg in CHESSQA_CONFIGS:
        try:
            p = hf_hub_download("wieeii/ChessQA-Benchmark",
                                f"data/chessqa_{cfg}.parquet", repo_type="dataset")
            df = pd.read_parquet(p)
        except Exception as e:            # noqa: BLE001
            print(f"[chessqa] skipped {cfg}: {e}")
            continue
        idx = list(range(len(df)))
        rng.shuffle(idx)
        for i in idx[:n_per_config]:
            row = df.iloc[i].to_dict()
            fen = next((row[k] for k in ("fen", "FEN", "board_fen") if k in row and row[k]), "")
            out.append(Item(id=f"chessqa-{cfg}-{i}", benchmark=f"chessqa_{cfg}",
                            fen=str(fen),
                            meta={k: (v.tolist() if hasattr(v, "tolist") else v)
                                  for k, v in row.items()}))
    return out


# --------------------------------------------------------------------------- #
# MATE -- the benchmark our source corpus came from
# --------------------------------------------------------------------------- #

def load_mate(n: int = 1000, seed: int = 0) -> list[Item]:
    """Direct comparability with Wang et al. (NAACL 2025), whose headline is a two-way
    move choice. We record the raw answer; the two-way grading happens in metrics."""
    from datasets import load_dataset
    try:
        ds = load_dataset("OutFlankShu/MATE_DATASET", split="train", streaming=True)
    except Exception as e:                # noqa: BLE001
        print(f"[mate] skipped: {e}")
        return []
    rng = random.Random(seed)
    rows = []
    for i, row in enumerate(ds):
        if i > n * 20:
            break
        rows.append(row)
    rng.shuffle(rows)
    out = []
    for i, row in enumerate(rows[:n]):
        text = json.dumps(row)[:2000]
        fen = ""
        import re
        m = re.search(r'"([rnbqkpRNBQKP1-8/]+ [wb] [KQkq-]+ [a-h1-8-]+ \d+ \d+)"', text)
        if m:
            fen = m.group(1)
        out.append(Item(id=f"mate-{i}", benchmark="mate", fen=fen, meta=dict(row)))
    return [it for it in out if it.fen]


def load_all(cfg: dict, tables=None) -> list[Item]:
    items: list[Item] = []
    if cfg.get("holdout"):
        items += load_holdout(cfg["holdout"], tables, cfg.get("holdout_limit"))
    if cfg.get("lichess"):
        items += load_lichess_puzzles(**cfg.get("lichess_args", {}))
    if cfg.get("chessqa"):
        items += load_chessqa(**cfg.get("chessqa_args", {}))
    if cfg.get("mate"):
        items += load_mate(**cfg.get("mate_args", {}))
    return items
