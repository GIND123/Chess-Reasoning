#!/usr/bin/env python3
"""Paper figures from results.json. Every value is measured; nothing is illustrative."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from chessr.plots import GRID, INK, INK_2, SERIES, STATUS, SURFACE, _save

ORDER = ["sft", "m3_move_only", "m4_sparse", "a3_no_coverage", "m6_composite"]
SHORT = {"sft": "SFT", "m3_move_only": "M3", "m4_sparse": "M4",
         "a3_no_coverage": "A3", "m6_composite": "M6"}
LONG = {"sft": "SFT", "m3_move_only": "M3 move-only", "m4_sparse": "M4 sparse",
        "a3_no_coverage": "A3 no-coverage", "m6_composite": "M6 composite"}


def fig_grounding(d, out):
    """Claim precision with 95% CI. The one place the method shows a real effect."""
    S = d["systems"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    xs = list(range(len(ORDER)))
    mu = [S[t]["overall"]["claim_precision"]["mean"] for t in ORDER]
    lo = [mu[i] - S[t]["overall"]["claim_precision"]["lo"] for i, t in enumerate(ORDER)]
    hi = [S[t]["overall"]["claim_precision"]["hi"] - mu[i] for i, t in enumerate(ORDER)]
    cols = [SERIES[0]] * len(ORDER)
    cols[ORDER.index("m6_composite")] = SERIES[1]
    ax1.bar(xs, mu, color=cols, edgecolor=SURFACE, linewidth=2, width=0.66)
    ax1.errorbar(xs, mu, yerr=[lo, hi], fmt="none", ecolor=INK_2, capsize=4, linewidth=1.4)
    ax1.set_xticks(xs); ax1.set_xticklabels([SHORT[t] for t in ORDER], fontsize=8)
    ax1.set_ylim(0.77, 0.80)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax1.set_ylabel("Claim precision")
    ax1.set_title("(a) Verified claims, 95% CI", loc="left", color=INK)
    ax1.grid(axis="x", visible=False)
    ax1.annotate("", xy=(4, 0.7955), xytext=(0, 0.7955),
                 arrowprops=dict(arrowstyle="-", color=STATUS["good"], lw=1.2))
    ax1.text(2, 0.7960, "M6 > SFT   p < 0.001", ha="center", fontsize=7.5,
             color=STATUS["good"])

    labels, diffs, ps = [], [], []
    for name, v in d["comparisons"].items():
        if name.endswith("claim_precision"):
            a, b = name.split(" :: ")[0].split(" vs ")
            labels.append(f"{LONG[a]} − {LONG[b]}")
            diffs.append(v["diff"] * 100); ps.append(v["p"])
    idx = sorted(range(len(diffs)), key=lambda i: diffs[i])
    ys = list(range(len(idx)))
    cols2 = [STATUS["good"] if ps[i] < 0.005 else GRID for i in idx]
    ax2.barh(ys, [diffs[i] for i in idx], color=cols2, edgecolor=SURFACE,
             linewidth=2, height=0.6)
    ax2.axvline(0, color=INK_2, linewidth=1)
    ax2.set_yticks(ys); ax2.set_yticklabels([labels[i] for i in idx], fontsize=7.5)
    ax2.set_xlabel("Difference in claim precision (pp)")
    ax2.set_title("(b) Paired differences", loc="left", color=INK)
    ax2.grid(axis="y", visible=False)
    span = max(abs(min(diffs)), abs(max(diffs)))
    ax2.set_xlim(-span * 1.25, span * 1.75)
    xtext = span * 1.30
    for y, i in zip(ys, idx):
        ax2.annotate(f"p={ps[i]:.3f}" if ps[i] >= 0.001 else "p<0.001",
                     (xtext, y), va="center", ha="left", fontsize=7,
                     color=STATUS["good"] if ps[i] < 0.005 else INK_2)
    fig.tight_layout()
    return _save(fig, Path(out), "fig6_grounding")


def fig_testtime(d, out):
    """Test-time selection. The verification-based reranker loses to plain voting."""
    S = d["systems"]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    methods = [("top1_engine", "greedy", SERIES[0], "overall"),
               ("rerank4_top1", "verified rerank @4", SERIES[1], "rerank"),
               ("vote4_top1", "majority vote @4", SERIES[2], "rerank"),
               ("oracle4_top1", "oracle @4 (ceiling)", GRID, "rerank")]
    w = 0.2
    xs = list(range(len(ORDER)))
    for j, (key, label, col, src) in enumerate(methods):
        vals = []
        for t in ORDER:
            blk = S[t]["overall"] if src == "overall" else S[t].get("rerank", {})
            v = blk.get(key)
            vals.append(v["mean"] if isinstance(v, dict) else (v if v is not None else float("nan")))
        ax.bar([x + (j - 1.5) * w for x in xs], vals, width=w * 0.92, label=label,
               color=col, edgecolor=SURFACE, linewidth=1.2)
    ax.set_xticks(xs); ax.set_xticklabels([SHORT[t] for t in ORDER], fontsize=8)
    ax.set_ylabel("Top-1 agreement with engine")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Selecting among 4 sampled traces", loc="left", color=INK)
    ax.legend(ncol=2, loc="upper left", fontsize=7.5)
    ax.set_ylim(0, 0.37)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, Path(out), "fig7_test_time_selection")


def fig_bands(d, out):
    """Everything by decision difficulty. Pooling hides that near-ties are much harder."""
    S = d["systems"]["m6_composite"]["by_band"]
    bands = [b for b in ("near_tie", "moderate", "decisive", "tactical") if b in S]
    labels = ["Near-tie\n<30cp", "Moderate\n30-100", "Decisive\n100-300", "Tactical\n>300"]
    labels = [l for l, b in zip(labels, ("near_tie", "moderate", "decisive", "tactical"))
              if b in S]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    prec = [S[b]["claim_precision"] for b in bands]
    ax1.bar(labels, prec, color=SERIES[0], edgecolor=SURFACE, linewidth=2, width=0.62)
    for i, v in enumerate(prec):
        ax1.annotate(f"{v:.0%}", (i, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8, color=INK)
    ax1.set_ylabel("Claim precision"); ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_title("(a) Grounding by decision difficulty", loc="left", color=INK)
    ax1.grid(axis="x", visible=False)

    top1 = [S[b]["top1_engine"] for b in bands]
    ax2.bar(labels, top1, color=SERIES[1], edgecolor=SURFACE, linewidth=2, width=0.62)
    for i, v in enumerate(top1):
        ax2.annotate(f"{v:.1%}", (i, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8, color=INK)
    ax2.set_ylabel("Top-1 agreement")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_title("(b) Move quality by decision difficulty", loc="left", color=INK)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, Path(out), "fig8_by_band")


if __name__ == "__main__":
    d = json.load(open("data/final/results.json"))
    out = "figures"
    for f in (fig_grounding, fig_testtime, fig_bands):
        for p in f(d, out):
            print(p)
