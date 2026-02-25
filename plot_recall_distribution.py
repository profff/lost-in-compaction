#!python3
"""
Plot per-fact recall distribution across 3 compaction strategies.

Generates:
1. ASCII heatmap (per-fact recall, one line per strategy)
2. ASCII cumulative recall curve
3. Binned histogram (15 bins of 10 facts)
4. Optional matplotlib charts if available
"""

import json
import sys


def load_results():
    """Load per-fact recall data from all 3 result files."""
    strategies = {}

    # Brutal
    with open("results_full_200k_brutal.json") as f:
        data = json.load(f)
    strategies["Brutal"] = [d["recalled"] for d in data["details"]]

    # Incremental
    with open("results_full_200k_incremental_checkpoint.json") as f:
        data = json.load(f)
    strategies["Incremental"] = [d["recalled"] for d in data["details"]]

    # Frozen
    with open("results_full_200k_frozen.json") as f:
        data = json.load(f)
    strategies["Frozen"] = [d["recalled"] for d in data["frozen_details"]]

    return strategies


def print_heatmap(strategies):
    """Print per-fact recall heatmap. Each char = 1 fact."""
    nFacts = len(next(iter(strategies.values())))

    print("\n" + "=" * 80)
    print("PER-FACT RECALL HEATMAP")
    print("Each character = 1 fact. █ = recalled, · = lost")
    print("X axis: fact position in conversation (0 = earliest, 149 = latest)")
    print("=" * 80)

    # Scale markers
    print(f"\n{'':14s}", end="")
    for i in range(0, nFacts, 10):
        print(f"{i:<10d}", end="")
    print()
    print(f"{'':14s}", end="")
    for i in range(nFacts):
        if i % 10 == 0:
            print("|", end="")
        else:
            print(" ", end="")
    print()

    for name, recalls in strategies.items():
        line = ""
        for r in recalls:
            line += "█" if r else "·"
        recalled = sum(recalls)
        print(f"{name:14s}{line}  ({recalled}/{nFacts})")

    # Zone markers
    third = nFacts // 3
    print(f"{'':14s}", end="")
    print(f"{'◄── early ──►':^{third}s}", end="")
    print(f"{'◄── mid ──►':^{third}s}", end="")
    print(f"{'◄── late ──►':^{nFacts - 2*third}s}")


def print_cumulative(strategies):
    """Print cumulative recall curve in ASCII."""
    nFacts = len(next(iter(strategies.values())))
    maxRecall = max(sum(r) for r in strategies.values())

    print("\n" + "=" * 80)
    print("CUMULATIVE RECALL CURVE")
    print("Y axis: cumulative facts recalled. X axis: fact position.")
    print("=" * 80)

    # Build cumulative data
    cumulatives = {}
    for name, recalls in strategies.items():
        cum = []
        total = 0
        for r in recalls:
            total += int(r)
            cum.append(total)
        cumulatives[name] = cum

    # ASCII chart: 20 rows high, nFacts wide (compressed to 75 chars)
    chartHeight = 20
    chartWidth = 75
    symbols = {"Frozen": "F", "Incremental": "I", "Brutal": "B"}
    colors = {"Frozen": "\033[36m", "Incremental": "\033[33m", "Brutal": "\033[31m"}
    reset = "\033[0m"

    # Build the chart grid
    grid = [[" " for _ in range(chartWidth)] for _ in range(chartHeight)]

    for name, cum in cumulatives.items():
        sym = symbols[name]
        for x in range(chartWidth):
            # Map x to fact index
            factIdx = int(x * (nFacts - 1) / (chartWidth - 1))
            # Map cumulative value to row
            row = int(cum[factIdx] * (chartHeight - 1) / max(maxRecall, 1))
            row = chartHeight - 1 - row  # flip Y axis
            if grid[row][x] == " ":
                grid[row][x] = sym
            else:
                grid[row][x] = "*"  # overlap

    # Print with Y axis labels
    for row in range(chartHeight):
        yVal = int((chartHeight - 1 - row) * maxRecall / (chartHeight - 1))
        print(f"  {yVal:3d} │", end="")
        line = "".join(grid[row])
        # Colorize symbols
        for char in line:
            if char in symbols.values():
                for name, sym in symbols.items():
                    if char == sym:
                        print(f"{colors[name]}{char}{reset}", end="")
                        break
            elif char == "*":
                print(f"\033[35m*{reset}", end="")
            else:
                print(char, end="")
        print()

    # X axis
    print(f"      └{'─' * chartWidth}")
    print(f"       0", end="")
    for i in range(1, 6):
        pos = int(i * chartWidth / 5)
        label = str(int(i * nFacts / 5))
        padding = pos - len(f"       0") if i == 1 else pos - prevPos - len(label)
        prevPos = pos
        print(f"{label:>{pos if i == 1 else len(label) + padding}s}", end="")
    print(f"\n       {'Fact position in conversation':^{chartWidth}s}")

    # Legend
    print(f"\n  Legend: {colors['Frozen']}F{reset} = Frozen  "
          f"{colors['Incremental']}I{reset} = Incremental  "
          f"{colors['Brutal']}B{reset} = Brutal  "
          f"\033[35m*{reset} = overlap")


def print_histogram(strategies):
    """Print binned histogram (15 bins of 10 facts each)."""
    nFacts = len(next(iter(strategies.values())))
    binSize = 10
    nBins = nFacts // binSize

    print("\n" + "=" * 80)
    print(f"RECALL HISTOGRAM — {nBins} bins of {binSize} facts each")
    print("=" * 80)

    # Calculate per-bin recall counts
    binData = {}
    for name, recalls in strategies.items():
        bins = []
        for b in range(nBins):
            start = b * binSize
            end = start + binSize
            bins.append(sum(recalls[start:end]))
        binData[name] = bins

    # Bar chart: horizontal bars for each bin
    symbols = {"Frozen": "█", "Incremental": "▓", "Brutal": "░"}
    colors = {"Frozen": "\033[36m", "Incremental": "\033[33m", "Brutal": "\033[31m"}
    reset = "\033[0m"

    maxVal = max(max(bins) for bins in binData.values())
    barScale = 30  # max bar width

    for b in range(nBins):
        startFact = b * binSize
        endFact = startFact + binSize - 1
        # Map to approximate token position
        startTok = int(startFact * 1500000 / nFacts / 1000)
        endTok = int((endFact + 1) * 1500000 / nFacts / 1000)

        print(f"\n  F{startFact:03d}-F{endFact:03d} ({startTok:4d}K-{endTok:4d}K tokens)")
        for name in ["Frozen", "Incremental", "Brutal"]:
            count = binData[name][b]
            barLen = int(count * barScale / max(maxVal, 1))
            bar = symbols[name] * barLen
            print(f"    {name:14s} {colors[name]}{bar}{reset} {count}/{binSize}")

    # Summary table
    print(f"\n  {'Bin':>16s}", end="")
    for name in ["Frozen", "Incremental", "Brutal"]:
        print(f"  {name:>12s}", end="")
    print()
    print(f"  {'─' * 16}", end="")
    for _ in range(3):
        print(f"  {'─' * 12}", end="")
    print()

    for b in range(nBins):
        startFact = b * binSize
        endFact = startFact + binSize - 1
        print(f"  F{startFact:03d}-F{endFact:03d}     ", end="")
        for name in ["Frozen", "Incremental", "Brutal"]:
            count = binData[name][b]
            pct = count * 100 // binSize
            print(f"  {count:>5d} ({pct:2d}%)", end="")
        print()


def try_matplotlib(strategies):
    """Generate matplotlib charts if available."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("\n[matplotlib not available — ASCII output only]")
        return

    nFacts = len(next(iter(strategies.values())))
    colors = {"Frozen": "#00BCD4", "Incremental": "#FF9800", "Brutal": "#F44336"}

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 2, 2]})
    fig.suptitle("Fact Recall Distribution — 3 Compaction Strategies\n(150 facts across 1.5M-token conversation, 200K context window)",
                 fontsize=12, fontweight="bold")

    # 1. Heatmap
    ax = axes[0]
    yPos = {"Frozen": 2, "Incremental": 1, "Brutal": 0}
    for name, recalls in strategies.items():
        for i, r in enumerate(recalls):
            if r:
                ax.barh(yPos[name], 1, left=i, height=0.8, color=colors[name], edgecolor="none")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Brutal", "Incremental", "Frozen"])
    ax.set_xlim(0, nFacts)
    ax.set_xlabel("Fact index")
    ax.set_title("Per-fact recall (colored = recalled)")
    # Add zone separators
    for x in [nFacts // 3, 2 * nFacts // 3]:
        ax.axvline(x, color="gray", linestyle="--", alpha=0.5)

    # 2. Cumulative recall
    ax = axes[1]
    for name, recalls in strategies.items():
        cum = []
        total = 0
        for r in recalls:
            total += int(r)
            cum.append(total)
        ax.plot(range(nFacts), cum, color=colors[name], linewidth=2, label=f"{name} ({total})")
    ax.set_xlabel("Fact index")
    ax.set_ylabel("Cumulative facts recalled")
    ax.set_title("Cumulative recall curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for x in [nFacts // 3, 2 * nFacts // 3]:
        ax.axvline(x, color="gray", linestyle="--", alpha=0.5)

    # 3. Binned histogram
    ax = axes[2]
    binSize = 10
    nBins = nFacts // binSize
    x = range(nBins)
    width = 0.25
    offsets = {"Frozen": -width, "Incremental": 0, "Brutal": width}

    for name, recalls in strategies.items():
        bins = []
        for b in range(nBins):
            start = b * binSize
            bins.append(sum(recalls[start:start + binSize]))
        ax.bar([i + offsets[name] for i in x], bins, width, color=colors[name], label=name)

    ax.set_xlabel("Fact bin (10 facts each)")
    ax.set_ylabel("Facts recalled")
    ax.set_title(f"Recall histogram — {nBins} bins of {binSize} facts")
    ax.set_xticks(range(nBins))
    ax.set_xticklabels([f"F{i*10:03d}" for i in range(nBins)], rotation=45, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for x_pos in [nBins // 3, 2 * nBins // 3]:
        ax.axvline(x_pos - 0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    outFile = "recall_distribution.png"
    plt.savefig(outFile, dpi=150, bbox_inches="tight")
    print(f"\n[Saved matplotlib chart to {outFile}]")
    plt.close()


def main():
    strategies = load_results()

    print_heatmap(strategies)
    print_cumulative(strategies)
    print_histogram(strategies)

    if "--no-plot" not in sys.argv:
        try_matplotlib(strategies)


if __name__ == "__main__":
    main()
