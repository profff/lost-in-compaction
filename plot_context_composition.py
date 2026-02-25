#!python
"""
Visualize the final context window composition for each compaction strategy.
Shows the "glass" view: what's inside the 200K context after all messages are fed.

Usage:
    ./plot_context_composition.py
    ./plot_context_composition.py --scale 1.5M   # use 1.5M results instead
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np


def load_strategy_data(files):
    """Load compaction stats from checkpoint files."""
    strategies = {}
    for fname, label in files:
        try:
            with open(fname, encoding="utf-8") as f:
                data = json.load(f)
            strategies[label] = data
        except FileNotFoundError:
            print(f"  Warning: {fname} not found, skipping {label}")
    return strategies


def estimate_final_state(data, label, total_msgs=18220, max_tokens=200_000):
    """Estimate the final context composition for a strategy."""
    stats = data["compaction_stats"]
    last = stats[-1]
    nCycles = len(stats)

    # After last compaction
    msgsAfterCompact = last["messagesRemaining"]
    tokensAfterCompact = last["tokensAfterEst"]

    # Estimate tail (messages fed after last compaction)
    if nCycles >= 2:
        avgFreed = sum(s["tokensFreed"] for s in stats) / nCycles
        avgCompacted = sum(s["messagesCompacted"] for s in stats) / nCycles
        tokensPerMsg = avgFreed / avgCompacted if avgCompacted > 0 else 150
    else:
        tokensPerMsg = 150

    # Known final states from benchmark output logs
    finalStates = {
        "Frozen": (932, 157_346),
        "Incremental": (974, 159_211),
    }

    if label in finalStates:
        finalMsgs, finalTokens = finalStates[label]
    else:
        # Estimate for brutal
        tailMsgs = total_msgs - sum(s["messagesCompacted"] for s in stats)
        tailTokens = int(tailMsgs * tokensPerMsg)
        finalMsgs = tailMsgs
        finalTokens = tokensAfterCompact + tailTokens

    result = {
        "label": label,
        "finalMsgs": finalMsgs,
        "finalTokens": min(finalTokens, max_tokens),
        "maxTokens": max_tokens,
        "cycles": nCycles,
        "zones": [],
    }

    # Compute zones based on strategy
    if label == "Frozen":
        nFrozen = last.get("frozenSummaries", 0)
        tailMsgs = finalMsgs - msgsAfterCompact
        tailTokens = finalTokens - tokensAfterCompact
        if tailMsgs > 0:
            tokensPerRawMsg = tailTokens / tailMsgs
        else:
            tokensPerRawMsg = tokensPerMsg
        rawMsgsAtCompact = msgsAfterCompact - (nFrozen * 2)
        rawTokensAtCompact = rawMsgsAtCompact * tokensPerRawMsg
        frozenTokens = max(tokensAfterCompact - rawTokensAtCompact, 5000)
        rawTokens = finalTokens - frozenTokens
        emptyTokens = max_tokens - finalTokens

        result["zones"] = [
            {"name": f"{nFrozen} frozen summaries", "tokens": int(frozenTokens),
             "color": "#1a5276", "order": 0},
            {"name": "Raw messages", "tokens": int(rawTokens),
             "color": "#5dade2", "order": 1},
            {"name": "Available", "tokens": int(emptyTokens),
             "color": "#eaecee", "order": 2, "hatch": "///"},
        ]
        result["frozenCount"] = nFrozen

    elif label == "Incremental":
        summaryTokens = 2000
        rawTokens = finalTokens - summaryTokens
        emptyTokens = max_tokens - finalTokens

        result["zones"] = [
            {"name": f"Summary (re-summarized {nCycles}\u00d7)", "tokens": summaryTokens,
             "color": "#d35400", "order": 0},
            {"name": "Raw messages", "tokens": int(rawTokens),
             "color": "#f0b27a", "order": 1},
            {"name": "Available", "tokens": int(emptyTokens),
             "color": "#eaecee", "order": 2, "hatch": "///"},
        ]

    elif label == "Brutal":
        summaryTokens = 500
        rawTokens = finalTokens - summaryTokens
        emptyTokens = max_tokens - finalTokens

        result["zones"] = [
            {"name": f"Summary ({nCycles} cycles)", "tokens": summaryTokens,
             "color": "#922b21", "order": 0},
            {"name": "Raw messages (tail)", "tokens": int(rawTokens),
             "color": "#f1948a", "order": 1},
            {"name": "Available", "tokens": int(emptyTokens),
             "color": "#eaecee", "order": 2, "hatch": "///"},
        ]

    return result


def draw_context_glasses(strategies_data, max_tokens=200_000, output="context_composition.png",
                         title_suffix=""):
    """Draw side-by-side context window composition."""
    nStrats = len(strategies_data)
    fig, axes = plt.subplots(1, nStrats, figsize=(4.5 * nStrats, 11),
                             gridspec_kw={"wspace": 0.35})
    if nStrats == 1:
        axes = [axes]

    fig.suptitle(f"Context Window Composition \u2014 Final State{title_suffix}",
                 fontsize=15, fontweight="bold", y=0.97)
    fig.text(0.5, 0.94,
             "200K token context window after all 3M tokens fed & compacted",
             ha="center", fontsize=10, color="#666", style="italic")

    barWidth = 0.55

    for idx, (ax, sdata) in enumerate(zip(axes, strategies_data)):
        label = sdata["label"]
        zones = sdata["zones"]

        # Draw zones from bottom up
        bottom = 0
        for zone in zones:
            height = zone["tokens"]
            pct = zone["tokens"] / max_tokens * 100
            hatch = zone.get("hatch")

            ax.bar(0, height, bottom=bottom, width=barWidth,
                   color=zone["color"], edgecolor="#555", linewidth=0.6,
                   hatch=hatch, zorder=2)

            # Labels — only for zones big enough
            if pct > 8:
                textColor = "white" if zone["order"] < 2 else "#888"
                fontSize = 9 if pct > 20 else 8
                ax.text(0, bottom + height / 2,
                        f"{zone['name']}\n{zone['tokens']//1000}K tok ({pct:.0f}%)",
                        ha="center", va="center", fontsize=fontSize,
                        fontweight="bold", color=textColor, zorder=5)
            elif pct > 2.5:
                textColor = "white" if zone["order"] < 2 else "#888"
                ax.text(0, bottom + height / 2,
                        f"{zone['name']}  {zone['tokens']//1000}K ({pct:.0f}%)",
                        ha="center", va="center", fontsize=7,
                        fontweight="bold", color=textColor, zorder=5)

            bottom += height

        # Draw frozen summary bands with gradient
        if label == "Frozen" and "frozenCount" in sdata:
            nFrozen = sdata["frozenCount"]
            frozenTokens = zones[0]["tokens"]
            bandHeight = frozenTokens / nFrozen

            for i in range(nFrozen):
                y = i * bandHeight
                t = i / max(nFrozen - 1, 1)
                r = int(20 + t * 60)
                g = int(60 + t * 100)
                b = int(100 + t * 60)
                color = f"#{r:02x}{g:02x}{b:02x}"
                ax.bar(0, bandHeight, bottom=y, width=barWidth,
                       color=color, edgecolor="#1a3a5a", linewidth=0.2,
                       zorder=3)

            # Overlay label with semi-transparent background
            ax.text(0, frozenTokens / 2,
                    f"{nFrozen} frozen\nsummaries\n{frozenTokens//1000}K tok ({frozenTokens/max_tokens*100:.0f}%)",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground="#1a5276")])

        # Annotation arrows for key insights
        if label == "Brutal":
            emptyPct = zones[2]["tokens"] / max_tokens * 100
            fillLevel = sdata["finalTokens"]
            ax.annotate(
                f"Only {100-emptyPct:.0f}% used!\nBrutal discards\naggressively",
                xy=(0, fillLevel), xytext=(0.6, max_tokens * 0.5),
                fontsize=8, color="#922b21", fontweight="bold",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#922b21", lw=1.2),
                zorder=10)

        elif label == "Frozen":
            frozenTokens = zones[0]["tokens"]
            ax.annotate(
                f"48 immutable blocks\npreserve history\n\u2192 12% early recall",
                xy=(0, frozenTokens / 2), xytext=(0.55, max_tokens * 0.15),
                fontsize=8, color="#1a5276", fontweight="bold",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#1a5276", lw=1.2),
                zorder=10)

        elif label == "Incremental":
            ax.annotate(
                f"Summary re-compressed\n48\u00d7 (JPEG cascade)\n\u2192 2% early recall",
                xy=(0, 1000), xytext=(0.55, max_tokens * 0.12),
                fontsize=8, color="#d35400", fontweight="bold",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#d35400", lw=1.2),
                zorder=10)

        # Watermark lines (across all columns)
        highWM = int(max_tokens * 0.9)
        lowWM = int(max_tokens * 0.6)
        ax.axhline(y=highWM, color="#e74c3c", linestyle="--",
                   linewidth=1, alpha=0.5, zorder=1)
        ax.axhline(y=lowWM, color="#e67e22", linestyle="--",
                   linewidth=1, alpha=0.5, zorder=1)

        # Watermark labels only on first column
        if idx == 0:
            ax.text(-barWidth / 2 - 0.03, highWM, "HIGH\n90%",
                    fontsize=7, color="#e74c3c", va="center", ha="right",
                    fontweight="bold")
            ax.text(-barWidth / 2 - 0.03, lowWM, "LOW\n60%",
                    fontsize=7, color="#e67e22", va="center", ha="right",
                    fontweight="bold")

        # Fill level line
        ax.axhline(y=sdata["finalTokens"], color="#27ae60", linestyle="-",
                   linewidth=1.5, alpha=0.7, zorder=4)

        # Strategy title + cycles
        ax.set_title(f"{label}\n{sdata['cycles']} cycles",
                     fontsize=14, fontweight="bold", pad=12)

        # Recall at the bottom
        recall = sdata.get("recall", "?")
        mainColor = zones[0]["color"]
        ax.text(0, -max_tokens * 0.05,
                f"Recall: {recall}",
                ha="center", va="top", fontsize=12, fontweight="bold",
                color=mainColor)

        # Axes styling
        ax.set_xlim(-0.7, 1.1)
        ax.set_ylim(-max_tokens * 0.08, max_tokens * 1.03)
        ax.set_xticks([])
        yticks = [0, 40_000, 80_000, 120_000, 160_000, 200_000]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y // 1000}K" for y in yticks], fontsize=8)
        if idx == 0:
            ax.set_ylabel("Tokens", fontsize=11)

        ax.grid(axis="y", alpha=0.1, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {output}")
    plt.close()


def main():
    scale = "3M"
    if "--scale" in sys.argv:
        idx = sys.argv.index("--scale")
        if idx + 1 < len(sys.argv):
            scale = sys.argv[idx + 1]

    if scale == "3M":
        files = [
            ("results_3M_200k_brutal_checkpoint.json", "Brutal"),
            ("results_3M_200k_v2_incremental_checkpoint.json", "Incremental"),
            ("results_3M_200k_v2_frozen_checkpoint.json", "Frozen"),
        ]
        totalMsgs = 18220
        titleSuffix = " \u2014 3M tokens, 300 facts, 200K context"
    else:
        files = [
            ("results_full_200k_brutal_checkpoint.json", "Brutal"),
            ("results_full_200k_incremental_checkpoint.json", "Incremental"),
            ("results_full_200k_frozen_checkpoint.json", "Frozen"),
        ]
        totalMsgs = 8996
        titleSuffix = " \u2014 1.5M tokens, 150 facts, 200K context"

    print(f"Loading {scale} results...")
    strategies = load_strategy_data(files)

    if not strategies:
        print("No data found!")
        return

    # Compute final states
    finalStates = []
    for label in ["Brutal", "Incremental", "Frozen"]:
        if label not in strategies:
            continue
        state = estimate_final_state(strategies[label], label,
                                     total_msgs=totalMsgs)
        metrics = strategies[label].get("metrics", {})
        recall = metrics.get("recall_global", 0)
        state["recall"] = f"{recall * 100:.1f}%"
        finalStates.append(state)
        print(f"  {label}: {state['finalTokens']:,} tokens, "
              f"{state['finalMsgs']} msgs, recall={state['recall']}")
        for z in state["zones"]:
            print(f"    {z['name']}: {z['tokens']:,} tokens "
                  f"({z['tokens'] / 200_000 * 100:.1f}%)")

    output = f"context_composition_{scale.lower()}.png"
    draw_context_glasses(finalStates, output=output, title_suffix=titleSuffix)
    print(f"\nDone! Open {output}")


if __name__ == "__main__":
    main()
