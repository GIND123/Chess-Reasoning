"""Building Sets A-D.

Set A comes from the existing 150k prompts. Its engine lines are already embedded in the
prompt text and can be parsed out free -- only the full move table is new work.

Set B is sourced fresh and banded by decision difficulty, because Set A has essentially
no close decisions (0.2% of gaps below 200 cp) and that is the regime the work is about.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict

import chess

from chessr.boards import band_for_gap
from chessr.prompts import student_prompt

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

LINE_RX = re.compile(r"Line (\d); (?:Cp\((-?\d+)\)|Mate\((-?\d+)\)): ([^\n]*)")
ANN_RX = re.compile(r"Annotation : (.*?)\n\nUse the below", re.S)
MATE_CP = 10_000


def parse_legacy_prompt(prompt: str) -> dict:
    """Pull the embedded engine lines out of a legacy prompt.

    Note what this shows: Line 1 always begins with the correct move, so stripping the
    trailing 'The best move is' sentence is *not* sufficient to close the leak.
    """
    lines = []
    for _, cp, mate, seq in LINE_RX.findall(prompt):
        moves = seq.split()
        val = int(cp) if cp else (MATE_CP - abs(int(mate)) * 10) * (1 if int(mate) > 0 else -1)
        lines.append({"cp": val, "moves": moves})
    ann = ANN_RX.search(prompt)
    return {"lines": lines, "annotation": ann.group(1).strip().replace("\n", " ") if ann else None}


def load_set_a(path: str = "GRPO_GM_dataset.csv", limit: int | None = None) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit and i >= limit:
                break
            parsed = parse_legacy_prompt(row["Prompt"])
            out.append({
                "fen": row["FEN"],
                "best_move": row["Best_move"],
                "prompt": student_prompt(row["FEN"]),   # rebuilt: no answer, no hint
                "legacy_lines": parsed["lines"],
                "legacy_annotation": parsed["annotation"],
            })
    return out


def band_and_sample(records: list[dict], store, targets: dict[str, float],
                    total: int, seed: int = 0) -> list[dict]:
    """Stratified sample to a target band mix."""
    from chessr.engine import gap_cp
    by_band = defaultdict(list)
    for r in records:
        t = store.get(r["fen"])
        if not t:
            continue
        r = dict(r, band=band_for_gap(gap_cp(t)))
        by_band[r["band"]].append(r)

    rng = random.Random(seed)
    out = []
    for band, frac in targets.items():
        pool = by_band.get(band, [])
        want = int(total * frac)
        if len(pool) <= want:
            out.extend(pool)
        else:
            out.extend(rng.sample(pool, want))
    rng.shuffle(out)
    return out


def split(records: list[dict], test_frac: float = 0.04, seed: int = 0):
    """Position-level split. Defensible here because there is no game-level leakage:
    149,897 distinct pawn skeletons across 150,000 positions."""
    rng = random.Random(seed)
    recs = list(records)
    rng.shuffle(recs)
    n_test = int(len(recs) * test_frac)
    return recs[n_test:], recs[:n_test]


def write_jsonl(records: list[dict], path: str) -> int:
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return len(records)


def read_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def describe(records: list[dict], store=None) -> dict:
    """Corpus statistics for the datacard."""
    from chessr.engine import gap_cp
    stats: Counter = Counter()
    for r in records:
        b = chess.Board(r["fen"])
        stats["n"] += 1
        stats["white_to_move"] += b.turn
        n = len(b.piece_map())
        stats["opening" if n >= 28 else ("middlegame" if n >= 15 else "endgame")] += 1
        if store and (t := store.get(r["fen"])):
            stats[f"band_{band_for_gap(gap_cp(t))}"] += 1
    return dict(stats)
