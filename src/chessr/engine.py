"""Stockfish move tables.

For every position we score *every* legal move once, offline, and store the result. RL
then never touches an engine: reward is a dict lookup. This is what makes graded rewards
affordable and reproducible.

Node limits, not time limits: time limits make the corpus depend on machine load and are
not reproducible.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator

import chess
import chess.engine

from chessr.boards import score_to_cp

DEFAULT_NODES = 400_000       # ~ depth 16-18 in a middlegame; reproducible across machines
DEFAULT_THREADS = 1           # one thread per worker; parallelise over positions instead
DEFAULT_HASH_MB = 128


@dataclass
class EngineConfig:
    path: str = os.environ.get("STOCKFISH_PATH", "stockfish")
    nodes: int = DEFAULT_NODES
    threads: int = DEFAULT_THREADS
    hash_mb: int = DEFAULT_HASH_MB


@contextmanager
def engine_session(cfg: EngineConfig) -> Iterator[chess.engine.SimpleEngine]:
    """One long-lived engine per worker.

    The original annotator spawned a process per position, which dominated its runtime.
    """
    eng = chess.engine.SimpleEngine.popen_uci(cfg.path)
    try:
        eng.configure({"Threads": cfg.threads, "Hash": cfg.hash_mb})
        yield eng
    finally:
        eng.quit()


def move_table(eng: chess.engine.SimpleEngine, fen: str, cfg: EngineConfig) -> dict[str, int]:
    """UCI -> centipawns, from the mover's point of view, for every legal move.

    MultiPV is set to the number of legal moves so the whole table comes from one search.
    """
    board = chess.Board(fen)
    n = board.legal_moves.count()
    if n == 0:
        return {}
    infos = eng.analyse(board, chess.engine.Limit(nodes=cfg.nodes), multipv=n)
    if isinstance(infos, dict):
        infos = [infos]
    out: dict[str, int] = {}
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        out[pv[0].uci()] = score_to_cp(info["score"])
    # Any move the engine did not report (rare, on truncated MultiPV) is scored as the
    # worst reported value so it is never accidentally rewarded.
    if out:
        floor = min(out.values())
        for mv in board.legal_moves:
            out.setdefault(mv.uci(), floor)
    return out


def best_cp(table: dict[str, int]) -> int:
    return max(table.values()) if table else 0


def gap_cp(table: dict[str, int]) -> float:
    """|best - second best|: the decision-difficulty axis used for banding."""
    if len(table) < 2:
        return float("inf")
    vals = sorted(table.values(), reverse=True)
    return float(vals[0] - vals[1])


def build_tables(fens: Iterable[str], cfg: EngineConfig, out_path: str,
                 *, resume: bool = True, log_every: int = 500) -> int:
    """Write one JSON line per position: {"fen":..., "table": {...}}. Resumable."""
    done: set[str] = set()
    if resume and os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["fen"])
                except Exception:
                    continue
    todo = [f for f in fens if f not in done]
    n = 0
    with engine_session(cfg) as eng, open(out_path, "a") as fh:
        for fen in todo:
            try:
                tbl = move_table(eng, fen, cfg)
            except chess.engine.EngineError:
                continue
            if not tbl:
                continue
            fh.write(json.dumps({"fen": fen, "table": tbl}) + "\n")
            n += 1
            if n % log_every == 0:
                fh.flush()
                print(f"  {n}/{len(todo)} positions", flush=True)
    return n


class TableStore:
    """In-memory FEN -> move table. ~40k positions is a few hundred MB; fine for RL."""

    def __init__(self, path: str | None = None):
        self._d: dict[str, dict[str, int]] = {}
        if path:
            self.load(path)

    def load(self, path: str) -> "TableStore":
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self._d[rec["fen"]] = rec["table"]
        return self

    def __contains__(self, fen: str) -> bool:
        return fen in self._d

    def __len__(self) -> int:
        return len(self._d)

    def get(self, fen: str) -> dict[str, int] | None:
        return self._d.get(fen)

    def fens(self) -> list[str]:
        return list(self._d)
