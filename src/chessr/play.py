"""Full-game play against Stockfish, for playing strength.

This is the one evaluation that cannot be recomputed from saved single-position records:
a game is a sequence of dependent decisions, so it has to be played. Every move is still
logged with its position and the engine's view, so blunder rate, ACPL and per-phase
breakdowns are recomputable from the game logs without replaying.

Illegal output is handled by *resampling*, then by falling back to a legal move, with
both events counted -- a model that cannot produce a legal move should be visible in the
results rather than silently rescued.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field

import chess
import chess.engine

from chessr.boards import win_prob
from chessr.prompts import SYS_STUDENT, student_prompt
from chessr.prompts import parse_trace


@dataclass
class MoveLog:
    ply: int
    fen: str
    by: str                       # "model" | "engine"
    move: str | None
    legal: bool
    resampled: int = 0
    fell_back: bool = False
    cp_before: int | None = None
    cp_after_best: int | None = None
    cp_played: int | None = None
    wp_loss: float | None = None
    tokens: int = 0
    completion: str = ""


@dataclass
class GameLog:
    game_id: int
    model_is_white: bool
    opponent: str
    result: str = "*"
    plies: list[MoveLog] = field(default_factory=list)
    termination: str = ""


def _model_move(llm, tok, sampling, board: chess.Board, lora=None,
                tries: int = 3) -> tuple[str | None, str, int, int]:
    """Sample until a legal move appears. Returns (uci, text, tokens, n_resampled)."""
    from vllm import SamplingParams
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYS_STUDENT},
         {"role": "user", "content": student_prompt(board.fen())}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text, ntok = "", 0
    for attempt in range(tries):
        sp = SamplingParams(n=1, temperature=0.0 if attempt == 0 else 0.8,
                            top_p=0.95, max_tokens=sampling.get("max_tokens", 600),
                            seed=attempt)
        outs = llm.generate([prompt], sp, lora_request=lora) if lora else \
               llm.generate([prompt], sp)
        cand = outs[0].outputs[0]
        text, ntok = cand.text, len(cand.token_ids)
        mv = parse_trace(text, board.fen()).move
        if mv:
            return mv, text, ntok, attempt
    return None, text, ntok, tries - 1


def play_game(llm, tok, engine, board_limit, game_id: int, model_is_white: bool,
              *, lora=None, max_plies: int = 200, sampling: dict | None = None,
              eval_nodes: int = 200_000) -> GameLog:
    sampling = sampling or {}
    board = chess.Board()
    log = GameLog(game_id=game_id, model_is_white=model_is_white,
                  opponent=str(board_limit))
    rng = random.Random(game_id)

    while not board.is_game_over(claim_draw=True) and len(log.plies) < max_plies:
        model_turn = (board.turn == chess.WHITE) == model_is_white
        fen = board.fen()
        info = engine.analyse(board, chess.engine.Limit(nodes=eval_nodes))
        cp_before = info["score"].relative.score(mate_score=10000)

        if model_turn:
            uci, text, ntok, resampled = _model_move(llm, tok, sampling, board, lora)
            fell_back = False
            if uci is None:
                uci = rng.choice([m.uci() for m in board.legal_moves])
                fell_back = True
            mv = chess.Move.from_uci(uci)
            board.push(mv)
            after = engine.analyse(board, chess.engine.Limit(nodes=eval_nodes))
            cp_played = -after["score"].relative.score(mate_score=10000)
            log.plies.append(MoveLog(
                ply=len(log.plies), fen=fen, by="model", move=uci, legal=not fell_back,
                resampled=resampled, fell_back=fell_back, cp_before=cp_before,
                cp_played=cp_played,
                wp_loss=max(0.0, win_prob(cp_before) - win_prob(cp_played)),
                tokens=ntok, completion=text[:1200]))
        else:
            res = engine.play(board, board_limit)
            board.push(res.move)
            log.plies.append(MoveLog(ply=len(log.plies), fen=fen, by="engine",
                                     move=res.move.uci(), legal=True,
                                     cp_before=cp_before))

    log.result = board.result(claim_draw=True)
    log.termination = ("checkmate" if board.is_checkmate() else
                       "stalemate" if board.is_stalemate() else
                       "insufficient" if board.is_insufficient_material() else
                       "max_plies" if len(log.plies) >= max_plies else "other")
    return log


def score_for_model(log: GameLog) -> float:
    if log.result == "1/2-1/2":
        return 0.5
    if log.result == "1-0":
        return 1.0 if log.model_is_white else 0.0
    if log.result == "0-1":
        return 0.0 if log.model_is_white else 1.0
    return 0.5


def elo_diff(score: float, n: int) -> tuple[float, float, float]:
    """Elo difference from a match score, with a normal-approximation 95% interval.
    Saturating scores are clamped so the estimate stays finite and is reported as a bound.
    """
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    eps = 1.0 / (2 * n)
    p = min(max(score, eps), 1 - eps)
    d = -400.0 * math.log10(1 / p - 1)
    se = math.sqrt(max(p * (1 - p), 1e-9) / n)
    lo_p = min(max(p - 1.96 * se, eps), 1 - eps)
    hi_p = min(max(p + 1.96 * se, eps), 1 - eps)
    return (d, -400.0 * math.log10(1 / lo_p - 1), -400.0 * math.log10(1 / hi_p - 1))


def summarise(logs: list[GameLog]) -> dict:
    """Recomputable from the saved logs; no replay needed."""
    n = len(logs)
    pts = sum(score_for_model(g) for g in logs)
    mp = [p for g in logs for p in g.plies if p.by == "model"]
    losses = [p.wp_loss for p in mp if p.wp_loss is not None]
    cp_losses = [max(0, (p.cp_before or 0) - (p.cp_played or 0)) for p in mp
                 if p.cp_before is not None and p.cp_played is not None]
    blunders = sum(1 for c in cp_losses if c >= 300)
    d, lo, hi = elo_diff(pts / n if n else 0.0, n)
    return {
        "games": n, "score": pts / n if n else float("nan"),
        "wins": sum(1 for g in logs if score_for_model(g) == 1.0),
        "draws": sum(1 for g in logs if score_for_model(g) == 0.5),
        "losses": sum(1 for g in logs if score_for_model(g) == 0.0),
        "elo_diff": d, "elo_lo": lo, "elo_hi": hi,
        "mean_wp_loss": sum(losses) / len(losses) if losses else float("nan"),
        "mean_cp_loss": sum(cp_losses) / len(cp_losses) if cp_losses else float("nan"),
        "blunders_per_100": 100 * blunders / len(mp) if mp else float("nan"),
        "illegal_fallback_rate": sum(p.fell_back for p in mp) / len(mp) if mp else float("nan"),
        "resample_rate": sum(p.resampled > 0 for p in mp) / len(mp) if mp else float("nan"),
        "mean_tokens": sum(p.tokens for p in mp) / len(mp) if mp else float("nan"),
    }


def dump_logs(logs: list[GameLog], path: str) -> str:
    with open(path, "w") as fh:
        for g in logs:
            fh.write(json.dumps(asdict(g)) + "\n")
    return path
