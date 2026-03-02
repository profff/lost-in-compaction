#!python3
"""
Plot iterative compaction benchmark results (Phase 3 / Part C).

Auto-discovers all iterative_v6_R4_*/summary.json files and produces
comparison figures across conversation sizes and strategies.

Usage:
    python plot_iterative_v6.py
    python plot_iterative_v6.py --output-dir figures_v6
"""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ============================================================================
# STYLE
# ============================================================================

STRATEGY_COLORS = {
    "S1": "#e74c3c",   # red — brutal
    "S2": "#f39c12",   # orange — incremental
    "S3": "#3498db",   # blue — frozen
    "S4": "#2ecc71",   # green — frozen ranked
}

STRATEGY_LABELS = {
    "S1": "S1 Brutal",
    "S2": "S2 Incremental",
    "S3": "S3 Frozen",
    "S4": "S4 FrozenRanked",
}

STRATEGY_MARKERS = {
    "S1": "x",
    "S2": "s",
    "S3": "^",
    "S4": "D",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})


# ============================================================================
# DATA LOADING
# ============================================================================

def discover_runs(baseDir="."):
    """Find all iterative_v6 summary files, return sorted by conversation size."""
    runs = []
    for p in sorted(Path(baseDir).glob("iterative_v6_R4_*/summary.json")):
        with open(p) as f:
            data = json.load(f)
        runs.append({
            "path": p.parent,
            "summary": data,
            "tokens": data["config"]["conversation_tokens"],
            "label": format_tokens(data["config"]["conversation_tokens"]),
        })
    runs.sort(key=lambda r: r["tokens"])
    return runs


def format_tokens(tok):
    """Round to nearest nice label: 500K, 1M, 5M, 10M, etc."""
    if tok >= 1_000_000:
        m = tok / 1_000_000
        if m >= 10:
            return f"{round(m):.0f}M"
        label = f"{m:.1f}".rstrip("0").rstrip(".")
        return f"{label}M"
    k = tok / 1_000
    rounded = round(k / 100) * 100
    if rounded >= 1000:
        return f"{rounded / 1000:.0f}M"
    return f"{rounded:.0f}K"


# ============================================================================
# FIGURE 1 — Recall vs conversation size (main result)
# ============================================================================

def plot_recall_vs_size(runs, outputDir):
    """Line plot: recall (y) vs conversation size (x) per strategy."""
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = ["S1", "S2", "S3", "S4"]
    xTokens = [r["tokens"] for r in runs]

    for strat in strategies:
        yVals = []
        for r in runs:
            if strat in r["summary"]["results"]:
                yVals.append(r["summary"]["results"][strat]["recall"] * 100)
            else:
                yVals.append(None)

        ax.plot(xTokens, yVals,
                color=STRATEGY_COLORS[strat],
                marker=STRATEGY_MARKERS[strat],
                markersize=8,
                linewidth=2,
                label=STRATEGY_LABELS[strat])

        # Annotate last point
        if yVals and yVals[-1] is not None:
            ax.annotate(f"{yVals[-1]:.1f}%",
                        (xTokens[-1], yVals[-1]),
                        textcoords="offset points",
                        xytext=(8, 0),
                        fontsize=8,
                        color=STRATEGY_COLORS[strat])

    ax.set_xlabel("Conversation size (tokens)")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall vs Conversation Size by Strategy\n(d80, bs=5, 190K context window)")

    ax.set_xscale("log")
    ax.set_xticks(xTokens)
    ax.set_xticklabels([r["label"] for r in runs])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([r["label"] for r in runs]))

    ax.set_ylim(-2, max(45, max(
        r["summary"]["results"]["S4"]["recall"] * 100 for r in runs) + 5))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    outPath = outputDir / "fig_c1_recall_vs_size.png"
    fig.savefig(outPath)
    plt.close(fig)
    print(f"  {outPath}")
    return outPath


# ============================================================================
# FIGURE 2 — Grep vs conversation size
# ============================================================================

def plot_grep_vs_size(runs, outputDir):
    """Line plot: grep keyword survival vs conversation size."""
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = ["S1", "S2", "S3", "S4"]
    xTokens = [r["tokens"] for r in runs]

    for strat in strategies:
        yVals = []
        for r in runs:
            if strat in r["summary"]["results"]:
                yVals.append(r["summary"]["results"][strat]["grep"]["recall_upper_bound"] * 100)
            else:
                yVals.append(None)

        ax.plot(xTokens, yVals,
                color=STRATEGY_COLORS[strat],
                marker=STRATEGY_MARKERS[strat],
                markersize=8,
                linewidth=2,
                label=STRATEGY_LABELS[strat])

    ax.set_xlabel("Conversation size (tokens)")
    ax.set_ylabel("Keywords found (%)")
    ax.set_title("Keyword Grep Survival vs Conversation Size\n(all keywords present in final context)")

    ax.set_xscale("log")
    ax.set_xticks(xTokens)
    ax.set_xticklabels([r["label"] for r in runs])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    outPath = outputDir / "fig_c2_grep_vs_size.png"
    fig.savefig(outPath)
    plt.close(fig)
    print(f"  {outPath}")
    return outPath


# ============================================================================
# FIGURE 3 — Quintile heatmap (position recall by strategy and size)
# ============================================================================

def plot_quintile_heatmap(runs, outputDir):
    """Grouped bar chart: recall per quintile, one subplot per conv size."""
    strategies = ["S1", "S2", "S3", "S4"]
    quintiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    nRuns = len(runs)
    fig, axes = plt.subplots(1, nRuns, figsize=(4 * nRuns, 5), sharey=True)
    if nRuns == 1:
        axes = [axes]

    barWidth = 0.18
    xPos = np.arange(len(quintiles))

    for idx, run in enumerate(runs):
        ax = axes[idx]
        for si, strat in enumerate(strategies):
            res = run["summary"]["results"].get(strat, {})
            vals = [res.get(f"recall_q{q+1}", 0) * 100 for q in range(5)]

            ax.bar(xPos + si * barWidth, vals,
                   width=barWidth,
                   color=STRATEGY_COLORS[strat],
                   label=STRATEGY_LABELS[strat] if idx == 0 else None,
                   edgecolor="white",
                   linewidth=0.5)

        ax.set_xticks(xPos + barWidth * 1.5)
        ax.set_xticklabels(quintiles)
        ax.set_xlabel("Position quintile (Q1=earliest)")
        ax.set_title(f"{run['label']} tokens\n({run['summary']['results']['S2']['compaction_cycles']} cycles)")
        ax.grid(True, alpha=0.2, axis="y")

    axes[0].set_ylabel("Recall (%)")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter())
    fig.legend(*axes[0].get_legend_handles_labels(),
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Recall by Original Position in Conversation",
                 fontsize=14, y=1.08)

    outPath = outputDir / "fig_c3_quintile_position.png"
    fig.savefig(outPath)
    plt.close(fig)
    print(f"  {outPath}")
    return outPath


# ============================================================================
# FIGURE 4 — Category breakdown (latest run only for clarity)
# ============================================================================

def plot_category_breakdown(runs, outputDir):
    """Grouped bar chart: recall per fact category, latest/largest run."""
    run = runs[-1]  # largest conversation
    strategies = ["S1", "S2", "S3", "S4"]

    # Get categories from first strategy
    firstRes = run["summary"]["results"]["S1"]
    categories = sorted(firstRes["by_category"].keys())
    catLabels = [c.replace("single-session-", "ss-") for c in categories]

    fig, ax = plt.subplots(figsize=(8, 5))
    barWidth = 0.18
    xPos = np.arange(len(categories))

    for si, strat in enumerate(strategies):
        res = run["summary"]["results"][strat]
        vals = [res["by_category"].get(cat, {}).get("recall", 0) * 100
                for cat in categories]

        bars = ax.bar(xPos + si * barWidth, vals,
                      width=barWidth,
                      color=STRATEGY_COLORS[strat],
                      label=STRATEGY_LABELS[strat],
                      edgecolor="white",
                      linewidth=0.5)

        # Count labels on bars
        for bar, cat in zip(bars, categories):
            n = res["by_category"].get(cat, {}).get("total", 0)
            if si == 0:  # only label count once
                ax.text(bar.get_x() + barWidth * 2, -4,
                        f"n={n}", fontsize=7, ha="center", color="#666")

    ax.set_xticks(xPos + barWidth * 1.5)
    ax.set_xticklabels(catLabels, fontsize=9)
    ax.set_ylabel("Recall (%)")
    ax.set_title(f"Recall by Fact Category — {run['label']} tokens")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")

    outPath = outputDir / "fig_c4_category_breakdown.png"
    fig.savefig(outPath)
    plt.close(fig)
    print(f"  {outPath}")
    return outPath


# ============================================================================
# FIGURE 5 — Compaction cycles vs conversation size
# ============================================================================

def plot_cycles_vs_size(runs, outputDir):
    """Dual-axis: compaction cycles (bars) and recall (lines) vs size."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    strategies = ["S1", "S2", "S3", "S4"]
    xTokens = [r["tokens"] for r in runs]
    xIdx = np.arange(len(runs))

    barWidth = 0.18
    for si, strat in enumerate(strategies):
        cycles = [r["summary"]["results"][strat]["compaction_cycles"] for r in runs]
        ax1.bar(xIdx + si * barWidth, cycles,
                width=barWidth,
                color=STRATEGY_COLORS[strat],
                alpha=0.4,
                label=f"{STRATEGY_LABELS[strat]} cycles")

    ax1.set_ylabel("Compaction cycles")
    ax1.set_xticks(xIdx + barWidth * 1.5)
    ax1.set_xticklabels([r["label"] for r in runs])
    ax1.set_xlabel("Conversation size")

    ax2 = ax1.twinx()
    for strat in strategies:
        recalls = [r["summary"]["results"][strat]["recall"] * 100 for r in runs]
        ax2.plot(xIdx + barWidth * 1.5, recalls,
                 color=STRATEGY_COLORS[strat],
                 marker=STRATEGY_MARKERS[strat],
                 markersize=8,
                 linewidth=2,
                 label=f"{STRATEGY_LABELS[strat]} recall")

    ax2.set_ylabel("Recall (%)")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())

    # Combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h2, l2, loc="upper right", fontsize=8)

    ax1.set_title("Compaction Cycles and Recall vs Conversation Size")
    ax1.grid(True, alpha=0.2, axis="y")

    outPath = outputDir / "fig_c5_cycles_and_recall.png"
    fig.savefig(outPath)
    plt.close(fig)
    print(f"  {outPath}")
    return outPath


# ============================================================================
# FIGURE 6 — Spatial recall: cumulative recalled facts by position
# ============================================================================

def load_meta_for_run(run, baseDir="."):
    """Load the conversation meta file matching this run's config.
    Handles mismatch between actual tokens and target-tokens label."""
    cfg = run["summary"]["config"]
    tok = cfg["conversation_tokens"]
    density = cfg["density"]
    seed = cfg.get("seed", 42)
    convDir = Path(baseDir) / "data/conversations/v6_R4"

    # Try exact formatted label first
    tokLabel = format_tokens(tok)
    metaFile = convDir / f"d{density}_{tokLabel}_seed{seed}_meta.json"
    if metaFile.exists():
        with open(metaFile) as f:
            return json.load(f)

    # Fallback: glob all meta files and pick closest by token count
    candidates = list(convDir.glob(f"d{density}_*_seed{seed}_meta.json"))
    if not candidates:
        return None
    best = None
    bestDelta = float("inf")
    for c in candidates:
        with open(c) as f:
            m = json.load(f)
        delta = abs(m["est_tokens"] - tok)
        if delta < bestDelta:
            bestDelta = delta
            best = m
    return best


def load_judgments(run, strategy):
    """Load judgment verdicts for a strategy from a run directory."""
    judgFile = run["path"] / f"judgments/{strategy}_bs5.json"
    if not judgFile.exists():
        return {}
    with open(judgFile) as f:
        data = json.load(f)
    verdicts = data.get("verdicts", data) if isinstance(data, dict) else data
    if isinstance(verdicts, list):
        return {v["fact_id"]: v["recalled"] for v in verdicts}
    return {}


def plot_spatial_recall(runs, outputDir, baseDir="."):
    """Step plot: cumulative recalled facts by position in original conversation.
    One subplot per conversation size, one curve per strategy."""
    strategies = ["S1", "S2", "S3", "S4"]
    nRuns = len(runs)

    fig, axes = plt.subplots(1, nRuns, figsize=(5 * nRuns, 5), sharey=True)
    if nRuns == 1:
        axes = [axes]

    for idx, run in enumerate(runs):
        ax = axes[idx]
        meta = load_meta_for_run(run, baseDir)
        if meta is None:
            ax.set_title(f"{run['label']} — meta not found")
            continue

        totalTokens = run["tokens"]

        # Build fact_id → position_tokens mapping
        factPositions = {}
        for fact in meta["facts"]:
            posTok = fact["position_pct"] / 100.0 * totalTokens
            factPositions[fact["fact_id"]] = posTok

        for strat in strategies:
            recallMap = load_judgments(run, strat)
            if not recallMap:
                continue

            # Sort facts by position
            points = []
            for factId, posTok in factPositions.items():
                recalled = recallMap.get(factId, False)
                points.append((posTok, 1 if recalled else 0))
            points.sort(key=lambda p: p[0])

            # Cumulative sum
            xVals = [0]
            yVals = [0]
            cumul = 0
            for posTok, rec in points:
                cumul += rec
                xVals.append(posTok)
                yVals.append(cumul)

            ax.step(xVals, yVals, where="post",
                    color=STRATEGY_COLORS[strat],
                    linewidth=2,
                    label=STRATEGY_LABELS[strat] if idx == 0 else None)

        nFacts = len(meta["facts"])
        nRecalledMax = max(
            sum(1 for v in load_judgments(run, s).values() if v)
            for s in strategies if load_judgments(run, s)
        )

        ax.set_xlabel(f"Position in original conversation")
        ax.set_title(f"{run['label']} tokens\n({run['summary']['results']['S2']['compaction_cycles']} cycles, {nFacts} facts)")
        ax.set_xlim(0, totalTokens)

        # Format x-axis as kTok or MTok
        if totalTokens >= 2_000_000:
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, p: f"{x/1e6:.0f}M"))
        else:
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, p: f"{x/1e3:.0f}K"))

        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Cumulative facts recalled")
    fig.legend(*axes[0].get_legend_handles_labels(),
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Spatial Recall: Where Are Facts Recovered From?",
                 fontsize=14, y=1.08)

    outPath = outputDir / "fig_c6_spatial_recall.png"
    fig.savefig(outPath)
    plt.close(fig)
    print(f"  {outPath}")
    return outPath


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot iterative compaction results")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--base-dir", type=str, default=".")
    args = parser.parse_args()

    outputDir = Path(args.output_dir)
    outputDir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  PLOT ITERATIVE COMPACTION RESULTS (Part C)")
    print("=" * 60)

    runs = discover_runs(args.base_dir)
    if not runs:
        print("  No iterative_v6_R4_*/summary.json files found!")
        return

    print(f"\n  Found {len(runs)} runs:")
    for r in runs:
        print(f"    {r['path'].name}: {r['label']} tokens")

    print(f"\n  Generating figures in {outputDir}/...")

    plot_recall_vs_size(runs, outputDir)
    plot_grep_vs_size(runs, outputDir)
    plot_quintile_heatmap(runs, outputDir)
    plot_category_breakdown(runs, outputDir)
    plot_cycles_vs_size(runs, outputDir)
    plot_spatial_recall(runs, outputDir, args.base_dir)

    print(f"\n  Done! {len(runs)} data points, 6 figures.")


if __name__ == "__main__":
    main()
