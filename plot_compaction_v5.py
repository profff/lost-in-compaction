#!/usr/bin/env python
"""
Publication-quality figures for the compaction benchmark paper.

4 figures:
  1. Recall vs Density (Part A baseline) — curves per batch_size
  2. Recall by Category — grouped bars at d80
  3. Recall vs Compaction Level (Part B) — curves per density
  4. Spatial Recall Density — cumulative recalled facts by position in original context

Usage:
  ./plot_compaction_v5.py                  # show all 4 figures
  ./plot_compaction_v5.py --save           # save as PNG
  ./plot_compaction_v5.py --fig 4          # show only figure 4
  ./plot_compaction_v5.py --save --dpi 300 # high-res PNGs
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ============================================================================
# PATHS
# ============================================================================
BASE = Path(__file__).parent
RECALL_DIR = BASE / "recall_v5_R4_20260223_1415"
COMPACT_DIR = BASE / "compaction_v5_R4_20260224_1353"
CONTEXT_META_DIR = BASE / "data" / "contexts" / "v5_R4"

# ============================================================================
# STYLE
# ============================================================================
COLORS = {
    "bs1": "#e74c3c",   # red
    "bs5": "#3498db",   # blue
    "bs10": "#2ecc71",  # green
    "d40": "#e67e22",   # orange
    "d60": "#9b59b6",   # purple
    "d80": "#2c3e50",   # dark blue
    "C0": "#2ecc71",
    "C1": "#3498db",
    "C2": "#f39c12",
    "C3": "#e74c3c",
    "C4": "#8e44ad",
}

CATEGORY_LABELS = {
    "single-session-user": "User statements",
    "single-session-assistant": "Assistant responses",
    "knowledge-update": "Knowledge updates",
    "single-session-preference": "Preferences",
}

CATEGORY_ORDER = [
    "single-session-user",
    "single-session-assistant",
    "knowledge-update",
    "single-session-preference",
]

CATEGORY_COLORS = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]


def loadJson(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# FIGURE 1: Recall vs Density
# ============================================================================
def fig1_recall_vs_density(ax):
    """Part A: recall as function of fact density, one curve per batch size."""
    summary = loadJson(RECALL_DIR / "summary.json")
    results = summary["results"]

    densities = sorted(int(k[1:]) for k in results.keys())
    densitiesKtok = [d / 190 for d in densities]  # facts per kTok (190K context)

    for bs, color, marker in [("bs1", COLORS["bs1"], "o"),
                               ("bs5", COLORS["bs5"], "s"),
                               ("bs10", COLORS["bs10"], "^")]:
        recalls = []
        for d in densities:
            r = results[f"d{d}"].get(bs, {}).get("recall", None)
            recalls.append(r * 100 if r is not None else None)
        ax.plot(densitiesKtok, recalls, color=color, marker=marker, markersize=5,
                linewidth=1.8, label=f"bs={bs[2:]}", zorder=3)

    # Grep upper bound
    grepRecalls = []
    for d in densities:
        g = results[f"d{d}"].get("grep", {}).get("recall_upper_bound", None)
        grepRecalls.append(g * 100 if g is not None else None)
    ax.plot(densitiesKtok, grepRecalls, color="#95a5a6", linestyle="--",
            linewidth=1.2, label="grep (upper bound)", zorder=2)

    ax.set_xlabel("Fact density (facts / kTok)")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall vs Fact Density (R4, 190K context)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))

    # Secondary x-axis: d** labels at key densities
    ax2 = ax.twiny()
    keyDensities = [20, 40, 60, 80, 120, 150]
    keyKtok = [d / 190 for d in keyDensities]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(keyKtok)
    ax2.set_xticklabels([f"d{d}" for d in keyDensities], fontsize=7, color="#666666")
    ax2.tick_params(axis="x", length=3, pad=1)


# ============================================================================
# FIGURE 2: Recall by Category
# ============================================================================
def fig2_recall_by_category(ax):
    """Part A: category breakdown at d80 for 3 batch sizes."""
    summary = loadJson(RECALL_DIR / "summary.json")
    results = summary["results"]["d80"]

    batchSizes = ["bs1", "bs5", "bs10"]
    bsLabels = ["bs=1", "bs=5", "bs=10"]
    bsColors = [COLORS["bs1"], COLORS["bs5"], COLORS["bs10"]]

    x = np.arange(len(CATEGORY_ORDER))
    width = 0.25

    for i, (bs, label, color) in enumerate(zip(batchSizes, bsLabels, bsColors)):
        vals = []
        for cat in CATEGORY_ORDER:
            r = results[bs]["by_category"].get(cat, {}).get("recall", 0)
            vals.append(r * 100)
        ax.bar(x + i * width - width, vals, width, label=label, color=color,
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORY_ORDER],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall by Category (d80, 190K context)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))


# ============================================================================
# FIGURE 3: Recall vs Compaction Level
# ============================================================================
def fig3_recall_vs_compaction(ax):
    """Part B: recall degradation by compaction level, one curve per density."""
    compSummary = loadJson(COMPACT_DIR / "summary.json")
    recallSummary = loadJson(RECALL_DIR / "summary.json")

    levels = ["C0", "C1", "C2", "C3", "C4"]
    levelPct = [0, 5, 25, 50, 98]
    bs = "bs5"

    for d, color, marker in [(40, COLORS["d40"], "o"),
                              (60, COLORS["d60"], "s"),
                              (80, COLORS["d80"], "^")]:
        recalls = []
        for lvl in levels:
            if lvl == "C0":
                r = recallSummary["results"][f"d{d}"][bs]["recall"]
            else:
                r = compSummary["results"][f"d{d}_{lvl}"][bs]["recall"]
            recalls.append(r * 100)
        ax.plot(levelPct, recalls, color=color, marker=marker, markersize=6,
                linewidth=2, label=f"d{d}", zorder=3)

    # Grep upper bound for d80
    grepVals = []
    for lvl in levels:
        if lvl == "C0":
            g = recallSummary["results"]["d80"]["grep"]["recall_upper_bound"]
        else:
            g = compSummary["results"][f"d80_{lvl}"]["grep"]["recall_upper_bound"]
        grepVals.append(g * 100)
    ax.plot(levelPct, grepVals, color="#95a5a6", linestyle="--",
            linewidth=1.2, label="grep d80", zorder=2)

    ax.set_xlabel("Context compacted (%)")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall vs Compaction Level (bs=5)")
    ax.set_xticks(levelPct)
    ax.set_xticklabels(["0%\n(C0)", "5%\n(C1)", "25%\n(C2)", "50%\n(C3)", "98%\n(C4)"])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))


# ============================================================================
# FIGURE 4: Spatial Recall Density
# ============================================================================
def fig4_spatial_recall(ax):
    """Part B: cumulative recalled facts by position in original context.

    For each compaction level, walk through the original context and
    plot cumulative number of recalled facts vs position (kTok).
    """
    d = 80
    bs = "bs5"

    # --- C0 baseline ---
    c0Meta = loadJson(CONTEXT_META_DIR / f"d{d}_seed42_meta.json")
    c0Judgments = loadJson(RECALL_DIR / f"judgments/d{d}_{bs}.json")

    # Build recall map: fact_id -> recalled
    c0RecallMap = {}
    for batch in c0Judgments["batches"]:
        for v in batch["verdicts"]:
            c0RecallMap[v["fact_id"]] = v["recalled"]

    # Build (position_kTok, recalled) pairs
    totalTokC0 = c0Meta.get("est_tokens", 190000)
    c0Facts = []
    for fm in c0Meta["facts"]:
        posTok = fm["position_pct"] / 100.0 * totalTokC0 / 1000  # kTok
        recalled = c0RecallMap.get(fm["fact_id"], False)
        c0Facts.append((posTok, recalled))
    c0Facts.sort(key=lambda x: x[0])

    # Plot cumulative
    c0Positions = [p for p, _ in c0Facts]
    c0Cumul = np.cumsum([1 if r else 0 for _, r in c0Facts])
    ax.step(c0Positions, c0Cumul, where="post", color=COLORS["C0"],
            linewidth=2, label="C0 (baseline)", zorder=5)

    # --- C1 to C4 ---
    for lvl, color, lw in [("C1", COLORS["C1"], 1.8),
                            ("C2", COLORS["C2"], 1.8),
                            ("C3", COLORS["C3"], 1.8),
                            ("C4", COLORS["C4"], 1.8)]:
        metaFile = COMPACT_DIR / f"contexts/d{d}_{lvl}_meta.json"
        judgFile = COMPACT_DIR / f"judgments/d{d}_{lvl}_{bs}.json"
        if not metaFile.exists() or not judgFile.exists():
            continue

        meta = loadJson(metaFile)
        judgments = loadJson(judgFile)

        recallMap = {}
        for batch in judgments["batches"]:
            for v in batch["verdicts"]:
                recallMap[v["fact_id"]] = v["recalled"]

        # Use ORIGINAL positions (from the v5 context meta, not the compacted meta)
        # The compacted meta has same facts with same positions
        totalTok = c0Meta.get("est_tokens", 190000)
        facts = []
        for fm in meta["facts"]:
            posTok = fm["position_pct"] / 100.0 * totalTok / 1000
            recalled = recallMap.get(fm["fact_id"], False)
            facts.append((posTok, recalled))
        facts.sort(key=lambda x: x[0])

        positions = [p for p, _ in facts]
        cumul = np.cumsum([1 if r else 0 for _, r in facts])

        # Add compaction boundary line
        fraction = meta["compaction"]["fraction"]
        boundaryKtok = fraction * totalTok / 1000

        ax.step(positions, cumul, where="post", color=color, linewidth=lw,
                label=f"{lvl} ({fraction:.0%})", zorder=4)

    # Compaction boundary annotations
    for lvl, frac, color in [("C1", 0.05, COLORS["C1"]),
                              ("C2", 0.25, COLORS["C2"]),
                              ("C3", 0.50, COLORS["C3"])]:
        bndKtok = frac * totalTokC0 / 1000
        ax.axvline(bndKtok, color=color, linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xlabel("Position in original context (kTok)")
    ax.set_ylabel("Cumulative facts recalled")
    ax.set_title(f"Spatial Recall Density (d{d}, bs=5)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, totalTokC0 / 1000 * 1.02)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Publication figures for compaction paper")
    parser.add_argument("--save", action="store_true", help="Save PNGs instead of showing")
    parser.add_argument("--fig", type=int, default=0, help="Show only this figure (1-4)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved PNGs")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Directory for saved PNGs")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
    })

    outDir = Path(args.output_dir)

    figures = {
        1: ("fig1_recall_vs_density", fig1_recall_vs_density),
        2: ("fig2_recall_by_category", fig2_recall_by_category),
        3: ("fig3_recall_vs_compaction", fig3_recall_vs_compaction),
        4: ("fig4_spatial_recall", fig4_spatial_recall),
    }

    toPlot = [args.fig] if args.fig else [1, 2, 3, 4]

    for figNum in toPlot:
        name, func = figures[figNum]
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        func(ax)
        fig.tight_layout()

        if args.save:
            outDir.mkdir(exist_ok=True)
            outPath = outDir / f"{name}.png"
            fig.savefig(outPath, dpi=args.dpi, bbox_inches="tight")
            print(f"  Saved: {outPath}")
        else:
            plt.show()

    if args.save:
        print(f"\nAll figures saved to {outDir}/")


if __name__ == "__main__":
    main()
