#!/usr/bin/env python3
"""SFT on verified traces."""
import argparse, sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessr.prompts import SYS_STUDENT
from chessr.sft import SFTSettings, load_jsonl, train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sft.yaml")
    p.add_argument("--data", default="data/final/sft.jsonl")
    p.add_argument("--out-dir")
    a = p.parse_args()
    cfg = SFTSettings(**yaml.safe_load(open(a.config)))
    if a.out_dir:
        cfg.out_dir = a.out_dir
    recs = load_jsonl(a.data)
    print(f"{len(recs)} verified traces -> {cfg.out_dir}")
    train(recs, SYS_STUDENT, cfg)
