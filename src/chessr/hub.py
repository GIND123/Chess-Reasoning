"""Hugging Face Hub sync — a durable fallback for everything the run produces.

Modal Volumes are the working store; the Hub is the copy that survives a lost app, an
expired container, or a machine change. Sync is incremental and idempotent: every call
uploads what changed and nothing else, so it is safe to call on a timer mid-run.

Layout in the dataset repo:

    tables/          engine tables (scores + principal variations) per shard
    generations/     raw teacher traces per shard
    final/           filtered SFT corpus, RL positions, held-out splits
    figures/         publication figures (pdf + png)
    metrics.json     every measured number the figures are built from
    README.md        datacard
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REPO = os.environ.get("CHESSR_HUB_REPO", "GOVINDFROM/chess-process-verified")


@dataclass
class HubSync:
    repo_id: str = DEFAULT_REPO
    repo_type: str = "dataset"
    token: str | None = field(default=None, repr=False)
    private: bool = True
    _ensured: bool = field(default=False, repr=False)

    def __post_init__(self):
        self.token = self.token or os.environ.get("HF_TOKEN")

    # --- internals ---------------------------------------------------------
    def _api(self):
        from huggingface_hub import HfApi
        return HfApi(token=self.token)

    def ensure(self):
        if self._ensured:
            return
        from huggingface_hub import create_repo
        create_repo(self.repo_id, repo_type=self.repo_type, private=self.private,
                    exist_ok=True, token=self.token)
        self._ensured = True

    # --- uploads -----------------------------------------------------------
    def put_file(self, local: str | Path, remote: str, msg: str | None = None) -> str:
        """Upload one file. Overwrites in place, so re-running is safe."""
        self.ensure()
        local = Path(local)
        if not local.exists():
            raise FileNotFoundError(local)
        self._api().upload_file(
            path_or_fileobj=str(local), path_in_repo=remote,
            repo_id=self.repo_id, repo_type=self.repo_type,
            commit_message=msg or f"sync {remote}",
        )
        return remote

    def put_dir(self, local: str | Path, remote_prefix: str,
                patterns: tuple[str, ...] = ("*",), msg: str | None = None) -> int:
        """Upload a directory as one commit. Returns the number of files matched."""
        self.ensure()
        local = Path(local)
        if not local.is_dir():
            return 0
        n = sum(1 for pat in patterns for _ in local.rglob(pat) if _.is_file())
        if not n:
            return 0
        self._api().upload_folder(
            folder_path=str(local), path_in_repo=remote_prefix,
            repo_id=self.repo_id, repo_type=self.repo_type,
            allow_patterns=list(patterns),
            commit_message=msg or f"sync {remote_prefix}",
        )
        return n

    def put_json(self, obj, remote: str, msg: str | None = None) -> str:
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            tmp = fh.name
        try:
            return self.put_file(tmp, remote, msg)
        finally:
            os.unlink(tmp)

    def exists(self, remote: str) -> bool:
        from huggingface_hub import file_exists
        try:
            return file_exists(self.repo_id, remote, repo_type=self.repo_type,
                               token=self.token)
        except Exception:
            return False


def datacard(stats: dict) -> str:
    """Datacard for the dataset repo. States the corpus defects explicitly, because the
    leaked-prompt split exists to be a negative control and must not be mistaken for
    training data."""
    return f"""---
license: cc-by-4.0
task_categories: [text-generation]
tags: [chess, reasoning, process-supervision, verifiable-rewards]
---

# Process-Verified Chess Reasoning

Reasoning traces whose every factual claim about the board has been checked
deterministically against the position, plus the engine tables that make the checking
possible.

## Contents

| Path | What it is |
|---|---|
| `tables/` | Stockfish {stats.get('engine', 'SF 17.1')} scores for **every legal move** of every position, plus principal variations for the top 8. Node-limited ({stats.get('nodes', 400000):,} nodes), so it is reproducible. |
| `generations/` | Raw teacher traces, before filtering. |
| `final/sft.jsonl` | Accepted traces. Prompt contains the position only — no answer, no engine data. |
| `final/leaked_control.jsonl` | The answer-conditioned corpus, retained as a **negative control**. Do not train on it expecting good results; that is the point. |
| `figures/` | Publication figures. |
| `metrics.json` | Every measured number behind the figures. |

## Key measured properties

- Positions: **{stats.get('n_positions', 0):,}**, each with a complete legal-move table.
- The source distribution is **not representative chess**: in {stats.get('pct_winning', 0):.1f}% of
  positions the side to move is already winning (median {stats.get('median_cp', 0):+.0f} cp), and
  {stats.get('pct_mate', 0):.1f}% have a forced mate as the best line. Report results per
  decision-difficulty band; pooled accuracy on this distribution is misleading.
- Acceptance rate under the verifier: **{stats.get('acceptance', 0):.0%}**.
- Accepted traces average **{stats.get('mean_tokens', 0):.0f} tokens**.

## Verification

Claims are extracted by regular grammar and checked with `python-chess` and the cached
engine tables — no model in the loop, so the labels are reproducible bit-for-bit. Twelve
claim types; nine are hard-verifiable, and one class is explicitly *unverifiable* and is
counted but never scored.

Code: https://github.com/ (see `src/chessr/claims.py`)
"""
