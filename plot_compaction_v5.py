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
def _spatial_recall(ax, bs="bs5", compactDir=None, recallDir=None):
    """Cumulative recalled facts by position in original context.

    For each compaction level, walk through the original context and
    plot cumulative number of recalled facts vs position (kTok).
    """
    d = 80
    if compactDir is None:
        compactDir = COMPACT_DIR
    if recallDir is None:
        recallDir = RECALL_DIR
    qLabel = bs.replace("bs", "Q=")

    # --- C0 baseline ---
    c0Meta = loadJson(CONTEXT_META_DIR / f"d{d}_seed42_meta.json")
    c0JudgFile = recallDir / f"judgments/d{d}_{bs}.json"
    if not c0JudgFile.exists():
        print(f"  WARNING: C0 judgments not found: {c0JudgFile}")
        return
    c0Judgments = loadJson(c0JudgFile)

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
        metaFile = compactDir / f"contexts/d{d}_{lvl}_meta.json"
        judgFile = compactDir / f"judgments/d{d}_{lvl}_{bs}.json"
        if not metaFile.exists() or not judgFile.exists():
            continue

        meta = loadJson(metaFile)
        judgments = loadJson(judgFile)

        recallMap = {}
        for batch in judgments["batches"]:
            for v in batch["verdicts"]:
                recallMap[v["fact_id"]] = v["recalled"]

        # Use ORIGINAL positions from c0Meta (compacted meta has positions
        # relative to the compacted context, not the original 190K)
        facts = []
        for fm in c0Meta["facts"]:
            posTok = fm["position_pct"] / 100.0 * totalTokC0 / 1000
            recalled = recallMap.get(fm["fact_id"], False)
            facts.append((posTok, recalled))
        facts.sort(key=lambda x: x[0])

        positions = [p for p, _ in facts]
        cumul = np.cumsum([1 if r else 0 for _, r in facts])

        # Add compaction boundary line
        fraction = meta["compaction"]["fraction"]

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
    ax.set_title(f"Spatial Recall Density (d{d}, {qLabel})")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, totalTokC0 / 1000 * 1.02)


def fig4_spatial_recall(ax):
    """Fig 4: spatial recall Q=5."""
    _spatial_recall(ax, bs="bs5")


def fig5_spatial_recall_q1(ax):
    """Fig 5: spatial recall Q=1."""
    _spatial_recall(ax, bs="bs1",
                    compactDir=BASE / "compaction_v5_R4_20260314_0856")


def _binned_recall_data(bs="bs5", compactDir=None, recallDir=None, nBins=10):
    """Compute per-bin recall rate for each compaction level."""
    d = 80
    if compactDir is None:
        compactDir = COMPACT_DIR
    if recallDir is None:
        recallDir = RECALL_DIR

    c0Meta = loadJson(CONTEXT_META_DIR / f"d{d}_seed42_meta.json")
    totalTokC0 = c0Meta.get("est_tokens", 190000)
    binEdges = np.linspace(0, totalTokC0 / 1000, nBins + 1)

    # Map facts to bins
    factBins = {}
    for fm in c0Meta["facts"]:
        posTok = fm["position_pct"] / 100.0 * totalTokC0 / 1000
        binIdx = min(int(posTok / (totalTokC0 / 1000) * nBins), nBins - 1)
        factBins[fm["fact_id"]] = binIdx

    results = {}

    # C0
    c0JudgFile = recallDir / f"judgments/d{d}_{bs}.json"
    if c0JudgFile.exists():
        c0Judgments = loadJson(c0JudgFile)
        recallMap = {}
        for batch in c0Judgments["batches"]:
            for v in batch["verdicts"]:
                recallMap[v["fact_id"]] = v["recalled"]
        binRecalled = [0] * nBins
        binTotal = [0] * nBins
        for fid, bIdx in factBins.items():
            binTotal[bIdx] += 1
            if recallMap.get(fid, False):
                binRecalled[bIdx] += 1
        results["C0"] = [binRecalled[i] / max(binTotal[i], 1) * 100
                         for i in range(nBins)]

    # C1-C4
    for lvl in ["C1", "C2", "C3", "C4"]:
        judgFile = compactDir / f"judgments/d{d}_{lvl}_{bs}.json"
        if not judgFile.exists():
            continue
        judgments = loadJson(judgFile)
        recallMap = {}
        for batch in judgments["batches"]:
            for v in batch["verdicts"]:
                recallMap[v["fact_id"]] = v["recalled"]
        binRecalled = [0] * nBins
        binTotal = [0] * nBins
        for fid, bIdx in factBins.items():
            binTotal[bIdx] += 1
            if recallMap.get(fid, False):
                binRecalled[bIdx] += 1
        results[lvl] = [binRecalled[i] / max(binTotal[i], 1) * 100
                        for i in range(nBins)]

    return results, binEdges


def _recall_density_data(bs="bs5", compactDir=None, recallDir=None):
    """Get per-fact (position_kTok, recalled) for each compaction level."""
    d = 80
    if compactDir is None:
        compactDir = COMPACT_DIR
    if recallDir is None:
        recallDir = RECALL_DIR

    c0Meta = loadJson(CONTEXT_META_DIR / f"d{d}_seed42_meta.json")
    totalTokC0 = c0Meta.get("est_tokens", 190000)

    # Map fact_id -> position in kTok (original context)
    factPos = {}
    for fm in c0Meta["facts"]:
        factPos[fm["fact_id"]] = fm["position_pct"] / 100.0 * totalTokC0 / 1000

    results = {}

    # C0
    c0JudgFile = recallDir / f"judgments/d{d}_{bs}.json"
    if c0JudgFile.exists():
        c0J = loadJson(c0JudgFile)
        recallMap = {}
        for batch in c0J["batches"]:
            for v in batch["verdicts"]:
                recallMap[v["fact_id"]] = v["recalled"]
        results["C0"] = [(factPos[fid], recallMap.get(fid, False))
                         for fid in factPos]

    # C1-C4
    for lvl in ["C1", "C2", "C3", "C4"]:
        judgFile = compactDir / f"judgments/d{d}_{lvl}_{bs}.json"
        if not judgFile.exists():
            continue
        j = loadJson(judgFile)
        recallMap = {}
        for batch in j["batches"]:
            for v in batch["verdicts"]:
                recallMap[v["fact_id"]] = v["recalled"]
        results[lvl] = [(factPos[fid], recallMap.get(fid, False))
                        for fid in factPos]

    return results, totalTokC0 / 1000


def _plot_recall_derivative(ax, bs, compactDir, recallDir, bandwidth=15.0):
    """Plot smoothed recall density (dRecall/dkTok) using Gaussian KDE."""
    from scipy.ndimage import gaussian_filter1d

    data, maxKtok = _recall_density_data(bs=bs, compactDir=compactDir,
                                          recallDir=recallDir)
    qLabel = bs.replace("bs", "Q=")

    # Evaluation grid
    xGrid = np.linspace(0, maxKtok, 500)
    dx = xGrid[1] - xGrid[0]

    for lvl in ["C0", "C1", "C2", "C3", "C4"]:
        if lvl not in data:
            continue
        pairs = sorted(data[lvl], key=lambda x: x[0])
        positions = np.array([p for p, _ in pairs])
        recalled = np.array([1.0 if r else 0.0 for _, r in pairs])
        totalFacts = np.array([1.0] * len(pairs))

        # Place each fact as a Gaussian blob on the grid, accumulate recalled and total
        recGrid = np.zeros_like(xGrid)
        totGrid = np.zeros_like(xGrid)
        for pos, rec, tot in zip(positions, recalled, totalFacts):
            idx = np.argmin(np.abs(xGrid - pos))
            recGrid[idx] += rec
            totGrid[idx] += tot

        # Smooth both with same bandwidth
        recSmooth = gaussian_filter1d(recGrid, sigma=bandwidth / dx)
        totSmooth = gaussian_filter1d(totGrid, sigma=bandwidth / dx)

        # Recall rate = smoothed recalled / smoothed total (avoid div by 0)
        rate = np.where(totSmooth > 1e-6, recSmooth / totSmooth * 100, np.nan)

        ax.plot(xGrid, rate, color=COLORS[lvl], linewidth=1.8, label=lvl)

    ax.set_xlabel("Position in original context (kTok)")
    ax.set_ylabel("Recall density (%)")
    ax.set_title(f"Smoothed Recall Density (d80, {qLabel}, bw={bandwidth:.0f}kTok)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_xlim(0, maxKtok * 1.02)


def _plot_normalized_recall(ax, bs, compactDir, recallDir, mode="binned",
                            nBins=10, bandwidth=15.0):
    """Plot Cx/C0 recall ratio by position, cancelling fact density bias.

    mode='binned': per-bin ratio
    mode='smoothed': ratio of Gaussian-smoothed curves
    """
    if mode == "binned":
        data, binEdges = _binned_recall_data(bs=bs, compactDir=compactDir,
                                              recallDir=recallDir, nBins=nBins)
        if "C0" not in data:
            return
        binCenters = (binEdges[:-1] + binEdges[1:]) / 2
        c0 = np.array(data["C0"])

        for lvl in ["C1", "C2", "C3", "C4"]:
            if lvl not in data:
                continue
            cx = np.array(data[lvl])
            # ratio where C0 > 0, else NaN
            ratio = np.where(c0 > 0, cx / c0, np.nan)
            ax.plot(binCenters, ratio, color=COLORS[lvl], marker="o",
                    markersize=4, linewidth=1.8, label=lvl)

        ax.axhline(1.0, color="#95a5a6", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Position in original context (kTok)")
        ax.set_ylabel("Recall ratio (Cx / C0)")
        ax.set_xlim(0, binEdges[-1] * 1.02)

    else:  # smoothed
        from scipy.ndimage import gaussian_filter1d
        rawData, maxKtok = _recall_density_data(bs=bs, compactDir=compactDir,
                                                 recallDir=recallDir)
        if "C0" not in rawData:
            return
        xGrid = np.linspace(0, maxKtok, 500)
        dx = xGrid[1] - xGrid[0]

        def _smooth_rate(lvlData):
            pairs = sorted(lvlData, key=lambda x: x[0])
            positions = np.array([p for p, _ in pairs])
            recalled = np.array([1.0 if r else 0.0 for _, r in pairs])
            recGrid = np.zeros_like(xGrid)
            totGrid = np.zeros_like(xGrid)
            for pos, rec in zip(positions, recalled):
                idx = np.argmin(np.abs(xGrid - pos))
                recGrid[idx] += rec
                totGrid[idx] += 1.0
            recSmooth = gaussian_filter1d(recGrid, sigma=bandwidth / dx)
            totSmooth = gaussian_filter1d(totGrid, sigma=bandwidth / dx)
            return np.where(totSmooth > 1e-6, recSmooth / totSmooth, np.nan)

        c0Rate = _smooth_rate(rawData["C0"])

        for lvl in ["C1", "C2", "C3", "C4"]:
            if lvl not in rawData:
                continue
            cxRate = _smooth_rate(rawData[lvl])
            ratio = np.where(c0Rate > 0.01, cxRate / c0Rate, np.nan)
            ax.plot(xGrid, ratio, color=COLORS[lvl], linewidth=1.8, label=lvl)

        ax.axhline(1.0, color="#95a5a6", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Position in original context (kTok)")
        ax.set_ylabel("Recall ratio (Cx / C0)")
        ax.set_xlim(0, maxKtok * 1.02)

    qLabel = bs.replace("bs", "Q=")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.5)


def _spatial_recall_normalized(ax, bs="bs5", compactDir=None, recallDir=None):
    """Cumulative recall ratio Cx/C0 by position. C0 = flat 1.0 line."""
    d = 80
    if compactDir is None:
        compactDir = COMPACT_DIR
    if recallDir is None:
        recallDir = RECALL_DIR
    qLabel = bs.replace("bs", "Q=")

    c0Meta = loadJson(CONTEXT_META_DIR / f"d{d}_seed42_meta.json")
    totalTokC0 = c0Meta.get("est_tokens", 190000)

    # C0 recall map
    c0JudgFile = recallDir / f"judgments/d{d}_{bs}.json"
    if not c0JudgFile.exists():
        return
    c0Judgments = loadJson(c0JudgFile)
    c0RecallMap = {}
    for batch in c0Judgments["batches"]:
        for v in batch["verdicts"]:
            c0RecallMap[v["fact_id"]] = v["recalled"]

    # Build sorted fact list with positions
    factList = []
    for fm in c0Meta["facts"]:
        posTok = fm["position_pct"] / 100.0 * totalTokC0 / 1000
        factList.append((posTok, fm["fact_id"]))
    factList.sort(key=lambda x: x[0])

    positions = [p for p, _ in factList]
    factIds = [fid for _, fid in factList]

    # C0 cumulative
    c0Cumul = np.cumsum([1.0 if c0RecallMap.get(fid, False) else 0.0
                         for fid in factIds])

    # C0 line at 1.0
    ax.axhline(1.0, color=COLORS["C0"], linewidth=2, label="C0 (baseline)",
               zorder=5)

    # C1-C4
    for lvl, color, lw in [("C1", COLORS["C1"], 1.8),
                            ("C2", COLORS["C2"], 1.8),
                            ("C3", COLORS["C3"], 1.8),
                            ("C4", COLORS["C4"], 1.8)]:
        judgFile = compactDir / f"judgments/d{d}_{lvl}_{bs}.json"
        metaFile = compactDir / f"contexts/d{d}_{lvl}_meta.json"
        if not judgFile.exists():
            continue

        judgments = loadJson(judgFile)
        recallMap = {}
        for batch in judgments["batches"]:
            for v in batch["verdicts"]:
                recallMap[v["fact_id"]] = v["recalled"]

        cxCumul = np.cumsum([1.0 if recallMap.get(fid, False) else 0.0
                             for fid in factIds])

        # Ratio — avoid div by 0 (use NaN where C0 cumul is 0)
        ratio = np.where(c0Cumul > 0, cxCumul / c0Cumul, np.nan)

        # Label with compaction fraction if available
        label = lvl
        if metaFile.exists():
            meta = loadJson(metaFile)
            fraction = meta["compaction"]["fraction"]
            label = f"{lvl} ({fraction:.0%})"

        ax.step(positions, ratio, where="post", color=color, linewidth=lw,
                label=label, zorder=4)

    # Compaction boundaries
    for lvl, frac, color in [("C1", 0.05, COLORS["C1"]),
                              ("C2", 0.25, COLORS["C2"]),
                              ("C3", 0.50, COLORS["C3"])]:
        bndKtok = frac * totalTokC0 / 1000
        ax.axvline(bndKtok, color=color, linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xlabel("Position in original context (kTok)")
    ax.set_ylabel("Cumulative recall ratio (Cx / C0)")
    ax.set_title(f"Normalized Spatial Recall (d80, {qLabel})")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, totalTokC0 / 1000 * 1.02)
    ax.set_ylim(-0.05, 1.3)


def fig6_binned_recall_comparison(fig_unused):
    """Fig 6: side-by-side binned recall rate Q=5 vs Q=1."""
    pass


def _plot_binned_recall(ax, bs, compactDir, recallDir, nBins=10):
    """Plot binned recall rate on given axes."""
    data, binEdges = _binned_recall_data(bs=bs, compactDir=compactDir,
                                          recallDir=recallDir, nBins=nBins)
    binCenters = (binEdges[:-1] + binEdges[1:]) / 2
    qLabel = bs.replace("bs", "Q=")

    for lvl in ["C0", "C1", "C2", "C3", "C4"]:
        if lvl not in data:
            continue
        ax.plot(binCenters, data[lvl], color=COLORS[lvl], marker="o",
                markersize=4, linewidth=1.8, label=lvl)

    ax.set_xlabel("Position in original context (kTok)")
    ax.set_ylabel("Recall rate (%)")
    ax.set_title(f"Local Recall Rate by Position (d80, {qLabel})")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_xlim(0, binEdges[-1] * 1.02)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Publication figures for compaction paper")
    parser.add_argument("--save", action="store_true", help="Save PNGs instead of showing")
    parser.add_argument("--fig", type=int, default=0, help="Show only this figure (1-5)")
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
        5: ("fig5_spatial_recall_q1", fig5_spatial_recall_q1),
        6: ("fig6_binned_recall_comparison", None),  # special case: 2 subplots
        7: ("fig7_spatial_recall_sonnet", None),  # special case: Sonnet C0 + C1-C4
        8: ("fig8_recall_density", None),  # smoothed derivative: Haiku Q=5, Q=1, Sonnet Q=1
        9: ("fig9_normalized_recall", None),  # Cx/C0 ratio: cancels fact density bias
    }

    toPlot = [args.fig] if args.fig else list(figures.keys())

    for figNum in toPlot:
        name, func = figures[figNum]

        if figNum == 6:
            # Side-by-side binned recall Q=5 vs Q=1
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            _plot_binned_recall(ax1, bs="bs5",
                                compactDir=COMPACT_DIR,
                                recallDir=RECALL_DIR, nBins=6)
            _plot_binned_recall(ax2, bs="bs1",
                                compactDir=BASE / "compaction_v5_R4_20260314_0856",
                                recallDir=RECALL_DIR, nBins=6)
        elif figNum == 8:
            # Smoothed recall density: 3 panels
            sonnetCompDir = BASE / "compaction_v5_R4_20260314_0957"
            sonnetRecDir = BASE / "recall_v5_R4_20260314_0950"
            haikuQ1CompDir = BASE / "compaction_v5_R4_20260314_0856"
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            _plot_recall_derivative(ax1, bs="bs5",
                                    compactDir=COMPACT_DIR,
                                    recallDir=RECALL_DIR)
            _plot_recall_derivative(ax2, bs="bs1",
                                    compactDir=haikuQ1CompDir,
                                    recallDir=RECALL_DIR)
            _plot_recall_derivative(ax3, bs="bs1",
                                    compactDir=sonnetCompDir,
                                    recallDir=sonnetRecDir)
            ax1.set_title("Haiku — Q=5")
            ax2.set_title("Haiku — Q=1")
            ax3.set_title("Sonnet 4.6 — Q=1")
            fig.suptitle("Smoothed Recall Density by Position (d80, bw=15kTok)",
                         fontsize=13, fontweight="bold", y=1.02)
        elif figNum == 9:
            # Normalized spatial recall Cx/C0: Q=5 and Q=1
            haikuQ1CompDir = BASE / "compaction_v5_R4_20260314_0856"
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            _spatial_recall_normalized(ax1, bs="bs5",
                                       compactDir=COMPACT_DIR,
                                       recallDir=RECALL_DIR)
            ax1.set_title("Haiku Q=5 — Normalized Spatial Recall")
            _spatial_recall_normalized(ax2, bs="bs1",
                                       compactDir=haikuQ1CompDir,
                                       recallDir=RECALL_DIR)
            ax2.set_title("Haiku Q=1 — Normalized Spatial Recall")
            fig.suptitle("Cumulative Recall Ratio (Cx / C0) — Fact Density Bias Removed",
                         fontsize=13, fontweight="bold", y=1.02)
        elif figNum == 7:
            # Sonnet: side-by-side cumulative + binned
            sonnetCompDir = BASE / "compaction_v5_R4_20260314_0957"
            sonnetRecDir = BASE / "recall_v5_R4_20260314_0950"
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            _spatial_recall(ax1, bs="bs1",
                            compactDir=sonnetCompDir,
                            recallDir=sonnetRecDir)
            ax1.set_title("Sonnet 4.6 — Cumulative Recall (d80, Q=1)")
            _plot_binned_recall(ax2, bs="bs1",
                                compactDir=sonnetCompDir,
                                recallDir=sonnetRecDir, nBins=6)
            ax2.set_title("Sonnet 4.6 — Local Recall Rate (d80, Q=1)")
        elif figNum == 4:
            # Combined 3-panel local recall rate: Haiku Q=5, Haiku Q=1, Sonnet Q=1.
            # Replaces standalone fig5 (Haiku Q=1) and fig7 (Sonnet Q=1).
            haikuQ1CompDir = BASE / "compaction_v5_R4_20260314_0856"
            sonnetCompDir = BASE / "compaction_v5_R4_20260314_0957"
            sonnetRecDir = BASE / "recall_v5_R4_20260314_0950"
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5),
                                                 sharey=True)
            _plot_binned_recall(ax1, bs="bs5",
                                compactDir=COMPACT_DIR,
                                recallDir=RECALL_DIR, nBins=6)
            ax1.set_title("Haiku 4.5 — Q=5")
            _plot_binned_recall(ax2, bs="bs1",
                                compactDir=haikuQ1CompDir,
                                recallDir=RECALL_DIR, nBins=6)
            ax2.set_title("Haiku 4.5 — Q=1")
            ax2.set_ylabel("")
            _plot_binned_recall(ax3, bs="bs1",
                                compactDir=sonnetCompDir,
                                recallDir=sonnetRecDir, nBins=6)
            ax3.set_title("Sonnet 4.6 — Q=1")
            ax3.set_ylabel("")
            fig.suptitle("Local Recall Rate by Position — Haiku vs Sonnet (d80)",
                         fontsize=13, fontweight="bold", y=1.02)
        else:
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
