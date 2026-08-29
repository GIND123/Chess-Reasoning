#!/usr/bin/env python3
"""Push everything durable to the Hub. Incremental and idempotent — safe on a timer.

Modal Volumes are the working store; the Hub is what survives a lost app or an expired
container. Run after any stage, or on a loop during a long one.
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessr.hub import HubSync, datacard


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=os.environ.get("CHESSR_HUB_REPO",
                                                    "GOVINDFROM/chess-process-verified"))
    p.add_argument("--public", action="store_true")
    p.add_argument("--figures", default="figures")
    p.add_argument("--metrics", default="data/final/metrics.json")
    p.add_argument("--final", default="data/final")
    p.add_argument("--tables", default="data/interim/tables_pv")
    p.add_argument("--generations", default="data/interim/gen")
    p.add_argument("--code", action="store_true", help="also push src/ and configs/")
    a = p.parse_args()

    hub = HubSync(repo_id=a.repo, private=not a.public)
    if not hub.token:
        raise SystemExit("no HF_TOKEN in the environment")
    hub.ensure()
    print(f"repo: {a.repo} ({'public' if a.public else 'private'})")

    n = hub.put_dir(a.figures, "figures", ("*.pdf", "*.png"), "figures")
    print(f"  figures        {n}")

    stats = {}
    if Path(a.metrics).exists():
        m = json.loads(Path(a.metrics).read_text())
        hub.put_file(a.metrics, "metrics.json", "metrics")
        ps = m.get("position_stats", {})
        stats = {
            "n_positions": ps.get("n", 0), "median_cp": ps.get("median_cp", 0),
            "pct_winning": ps.get("pct_winning", 0), "pct_mate": ps.get("pct_mate", 0),
            "acceptance": (m.get("acceptance", {}).get("stages") or [{}])[-1].get("accept", 0),
            "mean_tokens": m.get("acceptance", {}).get("ours_mean_tokens", 0),
            "engine": "Stockfish 17.1", "nodes": 400_000,
        }
        print("  metrics.json   1")

    for local, remote, pats in ((a.final, "final", ("*.jsonl", "*.json")),
                                (a.tables, "tables", ("*.jsonl",)),
                                (a.generations, "generations", ("*.jsonl",))):
        n = hub.put_dir(local, remote, pats, f"sync {remote}")
        print(f"  {remote:<14} {n}")

    if a.code:
        print("  src            ", hub.put_dir("src/chessr", "code/chessr", ("*.py",), "code"))
        print("  configs        ", hub.put_dir("configs", "code/configs", ("*.yaml",), "configs"))

    hub.put_json({"note": "datacard is README.md"}, ".sync_ok", "sync marker")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(datacard(stats)); tmp = fh.name
    hub.put_file(tmp, "README.md", "datacard")
    os.unlink(tmp)
    print(f"\nhttps://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
