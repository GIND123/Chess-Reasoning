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


def move_table_with_pvs(eng: chess.engine.SimpleEngine, fen: str, cfg: EngineConfig,
                        pv_top_k: int = 8, pv_plies: int = 6
                        ) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Scores for every legal move, plus the principal variation for the top-k.

    The MultiPV search already computes these lines; discarding them and then asking a
    language model to invent continuations was the largest single source of false claims
    (illegal replies) in the first generation run. Storing them makes the teacher's
    continuations correct by construction.
    """
    board = chess.Board(fen)
    n = board.legal_moves.count()
    if n == 0:
        return {}, {}
    infos = eng.analyse(board, chess.engine.Limit(nodes=cfg.nodes), multipv=n)
    if isinstance(infos, dict):
        infos = [infos]
    table: dict[str, int] = {}
    pvs: dict[str, list[str]] = {}
    ranked = []
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        uci = pv[0].uci()
        cp = score_to_cp(info["score"])
        table[uci] = cp
        ranked.append((cp, uci, [m.uci() for m in pv[:pv_plies]]))
    ranked.sort(key=lambda t: -t[0])
    for _, uci, line in ranked[:pv_top_k]:
        pvs[uci] = line
    if table:
        floor = min(table.values())
        for mv in board.legal_moves:
            table.setdefault(mv.uci(), floor)
    return table, pvs


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
        self._pv: dict[str, dict[str, list[str]]] = {}
        if path:
            self.load(path)

    def load(self, path: str) -> "TableStore":
        """Tolerant of a partially-written trailing line: shards are appended to while
        other stages read them, so a torn last record must not kill the reader."""
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._d[rec["fen"]] = rec["table"]
                if rec.get("pvs"):
                    self._pv[rec["fen"]] = rec["pvs"]
        return self

    def __contains__(self, fen: str) -> bool:
        return fen in self._d

    def __len__(self) -> int:
        return len(self._d)

    def get(self, fen: str) -> dict[str, int] | None:
        return self._d.get(fen)

    def pvs(self, fen: str) -> dict[str, list[str]]:
        return self._pv.get(fen, {})

    def fens(self) -> list[str]:
        return list(self._d)
