"""Publication figures.

Palette is the validated categorical set (slots assigned in fixed order, never cycled);
slots 1-3 are the ones that clear all-pairs separation, so any figure whose marks are
compared pairwise rather than adjacently uses at most three. Status colours are reserved
for pass/fail state and never stand in for a series.

Every figure is regenerated from the artefacts on disk, so re-running after new results
land refreshes the paper's plots without hand editing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

# --- design tokens -------------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
          "critical": "#d03b3b"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 1.0,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 2.0, "lines.markersize": 5,
    "figure.dpi": 160, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def _bar_labels(ax, bars, fmt="{:.0f}", pad=2, colour=INK):
    """Direct labels, selectively -- never a number on every mark of a dense chart."""
    for b in bars:
        ax.annotate(fmt.format(b.get_height()),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, pad),
                    ha="center", va="bottom", fontsize=8, color=colour)


# --- Figure 1: the corpus is not chess ----------------------------------------

def fig_position_distribution(stats: dict, out_dir: Path) -> list[Path]:
    """The F3 finding: the source corpus is overwhelmingly already-won positions.

    Two panels, one measure each -- never a second y-scale on one axes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))

    edges = stats["eval_hist"]["edges"]
    counts = stats["eval_hist"]["counts"]
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    widths = [(edges[i + 1] - edges[i]) * 0.92 for i in range(len(counts))]
    total = sum(counts) or 1
    ax1.bar(centres, [c / total for c in counts], width=widths,
            color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    ax1.axvline(0, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    top = ax1.get_ylim()[1]
    ax1.annotate("equal", (0, top * 0.93), xytext=(4, 0),
                 textcoords="offset points", fontsize=8, color=INK_2)
    if stats.get("pct_mate"):
        ax1.annotate(f"forced mate\n{stats['pct_mate']:.0f}% (clipped)",
                     (edges[-1], counts[-1] / total), xytext=(-6, -4),
                     textcoords="offset points", ha="right", va="top",
                     fontsize=7.5, color=INK_2)
    ax1.set_xlabel("Evaluation of the best move (centipawns, mover's view)")
    ax1.set_ylabel("Share of positions")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_title("(a) The side to move is already winning", loc="left", color=INK)

    bands = ["near_tie", "moderate", "decisive", "tactical"]
    labels = ["Near-tie\n<30cp", "Moderate\n30-100", "Decisive\n100-300", "Tactical\n>300"]
    vals = [stats["bands"].get(b, 0) for b in bands]
    tot = sum(vals) or 1
    shares = [v / tot for v in vals]
    bars = ax2.bar(labels, shares, color=[SERIES[2], SERIES[2], SERIES[2], SERIES[1]],
                   edgecolor=SURFACE, linewidth=2)
    _bar_labels(ax2, bars, fmt="{:.1%}")
    ax2.set_ylabel("Share of positions")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylim(0, max(shares) * 1.25 if shares else 1)
    ax2.set_title("(b) Almost no close decisions", loc="left", color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "fig1_position_distribution")


# --- Figure 2: what the verifier finds ----------------------------------------

def fig_claim_precision(rows: list[dict], out_dir: Path) -> list[Path]:
    """Per-claim-type precision. Identity is carried by the axis labels, not by hue,
    so a single series colour is correct here and no legend is needed."""
    rows = sorted(rows, key=lambda r: r["precision"])
    fig, ax = plt.subplots(figsize=(4.6, 0.42 * len(rows) + 1.1))
    ypos = range(len(rows))
    bars = ax.barh(list(ypos), [r["precision"] for r in rows],
                   color=SERIES[0], height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([f"{r['type']}  (n={r['n']:,})" for r in rows])
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Claims verified true")
    ax.axvline(0.9, color=STATUS["critical"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate("acceptance\nthreshold", (0.9, len(rows) - 0.35), xytext=(5, 0),
                textcoords="offset points", fontsize=7.5, color=STATUS["critical"],
                va="center")
    for b, r in zip(bars, rows):
        ax.annotate(f"{r['precision']:.1%}", (b.get_width(), b.get_y() + b.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_title("Claim verification on answer-conditioned traces", loc="left", color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "fig2_claim_precision")


# --- Figure 3: the acceptance gate --------------------------------------------

def fig_acceptance(stages: list[dict], reasons: list[dict], out_dir: Path) -> list[Path]:
    """Left: acceptance across pipeline states. Right: why traces are rejected.

    The left panel is an ordered progression, so it uses one ordinal ramp rather than
    categorical hues -- rank is the meaning, and colour follows it deliberately.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    ramp = ["#86b6ef", "#5598e7", "#2a78d6"]
    names = [s["name"] for s in stages]
    vals = [s["accept"] for s in stages]
    bars = ax1.bar(names, vals, color=ramp[-len(vals):], edgecolor=SURFACE, linewidth=2)
    _bar_labels(ax1, bars, fmt="{:.0%}")
    ax1.axhline(0.20, color=STATUS["critical"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax1.annotate("acceptance gate", (-0.45, 0.20), xytext=(0, 5),
                 textcoords="offset points", ha="left", fontsize=7.5,
                 color=STATUS["critical"])
    ax1.set_ylabel("Traces accepted")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylim(0, max(max(vals) * 1.35, 0.28))
    ax1.set_title("(a) Acceptance rate", loc="left", color=INK)
    ax1.grid(axis="x", visible=False)

    reasons = sorted(reasons, key=lambda r: r["share"])
    ypos = range(len(reasons))
    ax2.barh(list(ypos), [r["share"] for r in reasons], color=SERIES[1],
             height=0.62, edgecolor=SURFACE, linewidth=2)
    ax2.set_yticks(list(ypos))
    ax2.set_yticklabels([r["reason"] for r in reasons])
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_xlabel("Share of generated traces")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("(b) Why traces are rejected", loc="left", color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_acceptance")


# --- Figure 4: token efficiency ------------------------------------------------

def fig_token_efficiency(systems: list[dict], out_dir: Path) -> list[Path]:
    """Tokens per solution. Log axis because the range spans two orders of magnitude;
    ours is highlighted, everything else recedes to one neutral series colour."""
    systems = sorted(systems, key=lambda s: s["tokens"])
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    cols = [SERIES[1] if s.get("ours") else SERIES[0] for s in systems]
    bars = ax.bar([s["name"] for s in systems], [s["tokens"] for s in systems],
                  color=cols, edgecolor=SURFACE, linewidth=2)
    ax.set_yscale("log")
    ax.set_ylabel("Tokens per solution (log)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    for b, s in zip(bars, systems):
        ax.annotate(f"{s['tokens']:,}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8, color=INK)
    ax.grid(axis="x", visible=False)
    ax.set_title("Reasoning length per solved position", loc="left", color=INK)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_token_efficiency")


def build_all(metrics_path: str = "data/final/metrics.json",
              out_dir: str = "figures") -> list[Path]:
    """Regenerate every figure whose inputs are present in metrics.json."""
    m = json.loads(Path(metrics_path).read_text())
    out = Path(out_dir)
    made: list[Path] = []
    if "position_stats" in m:
        made += fig_position_distribution(m["position_stats"], out)
    if "claim_precision" in m:
        made += fig_claim_precision(m["claim_precision"], out)
    if "acceptance" in m:
        made += fig_acceptance(m["acceptance"]["stages"], m["acceptance"]["reasons"], out)
    if "token_efficiency" in m:
        made += fig_token_efficiency(m["token_efficiency"], out)
    return made
