#!/usr/bin/env python3
"""One GRPO arm. Reward composition comes entirely from the config file."""
import argparse, sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessr.grpo import GRPOSettings, train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    cfg = GRPOSettings(**yaml.safe_load(open(a.config)))
    if a.seed is not None:
        cfg.seed = a.seed
        cfg.out_dir = f"{cfg.out_dir}_s{a.seed}"
    print(f"terms={cfg.terms} weights={cfg.weights} sparse={cfg.sparse_move} -> {cfg.out_dir}")
    train(cfg)
