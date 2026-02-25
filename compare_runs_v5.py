#!python3
"""
Compare v5 recall runs -- 2x2 factorial design.

                Complete    Filtered
  Full evid.    R1          R3
  Chopped evid. R2          R4

Analyses:
  1. Per-run recall overview (density x batch_size)
  2. Truncation effect: R1↔R2 (all cats) and R3↔R4 (filtered cats)
  3. Filtering effect: R1↔R3 (full evid.) and R2↔R4 (chopped evid.)
  4. Interaction: does truncation delta differ between complete/filtered?
  5. Per-category density curves across all modes
  6. Verdict & recommendations

Usage:
    python compare_runs_v5.py R1_DIR R2_DIR R3_DIR R4_DIR
    python compare_runs_v5.py --auto
    python compare_runs_v5.py R1_DIR R2_DIR   (subset OK)
"""

import json
import sys
import math
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations


FILTERED_OUT_TYPES = {"temporal-reasoning", "multi-session"}
ALL_MODES = ["R1", "R2", "R3", "R4"]

# 2x2 factorial pairs
TRUNCATION_PAIRS = [("R1", "R2"), ("R3", "R4")]  # full->chopped
FILTERING_PAIRS = [("R1", "R3"), ("R2", "R4")]   # complete->filtered


# ============================================================================
# DATA LOADING
# ============================================================================

def load_summary(runDir):
    path = Path(runDir) / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(runDir):
    path = Path(runDir) / "config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_verdicts(runDir, density, batchSize):
    """Load all verdicts from a v5 judgment file."""
    path = Path(runDir) / "judgments" / f"d{density}_bs{batchSize}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    verdicts = []
    for batch in data["batches"]:
        verdicts.extend(batch["verdicts"])
    return verdicts


def load_recalled_set(runDir, density, batchSize):
    verdicts = load_verdicts(runDir, density, batchSize)
    if verdicts is None:
        return None
    return set(v["fact_id"] for v in verdicts if v.get("recalled"))


def load_verdicts_map(runDir, density, batchSize):
    verdicts = load_verdicts(runDir, density, batchSize)
    if verdicts is None:
        return None
    return {v["fact_id"]: v for v in verdicts}


def load_context_meta(runDir, density, seed=42):
    config = load_config(runDir)
    contextsDir = config.get("contexts_dir", "")
    if contextsDir:
        metaFile = Path(contextsDir) / f"d{density}_seed{seed}_meta.json"
        if metaFile.exists():
            return json.loads(metaFile.read_text(encoding="utf-8"))
    return None


def discover_densities(runDir):
    jdir = Path(runDir) / "judgments"
    if not jdir.exists():
        return []
    densities = set()
    for f in jdir.glob("d*_bs*.json"):
        densities.add(int(f.stem.split("_")[0][1:]))
    return sorted(densities)


def discover_batch_sizes(runDir):
    jdir = Path(runDir) / "judgments"
    if not jdir.exists():
        return []
    batchSizes = set()
    for f in jdir.glob("d*_bs*.json"):
        batchSizes.add(int(f.stem.split("_")[1][2:]))
    return sorted(batchSizes)


# ============================================================================
# HELPERS
# ============================================================================

def mean(vals):
    return sum(vals) / len(vals) if vals else 0


def stdev(vals):
    if len(vals) < 2:
        return 0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def recall_from_verdicts(verdicts):
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v.get("recalled")) / len(verdicts)


def compute_category_recall(verdicts):
    byCategory = defaultdict(lambda: {"total": 0, "recalled": 0, "accurate": 0})
    for v in verdicts:
        cat = v.get("question_type", "unknown")
        byCategory[cat]["total"] += 1
        if v.get("recalled"):
            byCategory[cat]["recalled"] += 1
        if v.get("recalled") and v.get("accurate"):
            byCategory[cat]["accurate"] += 1

    result = {}
    for cat, stats in sorted(byCategory.items()):
        result[cat] = {
            "total": stats["total"],
            "recalled": stats["recalled"],
            "recall": stats["recalled"] / stats["total"] if stats["total"] else 0,
            "accuracy": stats["accurate"] / stats["total"] if stats["total"] else 0,
        }
    return result


# ============================================================================
# AUTO-DETECT
# ============================================================================

def detect_runs(args):
    runDirs = {}

    if args.runs:
        for p in args.runs:
            d = Path(p)
            if not d.is_dir():
                print(f"  WARNING: {p} not a directory, skipping")
                continue
            config = load_config(d)
            mode = config.get("run_mode", "")
            if not mode:
                for m in ALL_MODES:
                    if m in d.name:
                        mode = m
                        break
            if mode:
                runDirs[mode] = d
            else:
                print(f"  WARNING: can't determine mode for {d.name}")
    else:
        for mode in ALL_MODES:
            candidates = sorted(Path(".").glob(f"recall_v5_{mode}_*"))
            candidates = [c for c in candidates if (c / "summary.json").exists()]
            if candidates:
                runDirs[mode] = candidates[-1]

    return runDirs


# ============================================================================
# SECTION 1 -- Per-run recall overview
# ============================================================================

def section_overview(modes, runDirs, modeDensities, batchSizes):
    print(f"\n{'-' * 80}")
    print(f"  1. PER-RUN RECALL OVERVIEW")
    print(f"{'-' * 80}")

    for mode in modes:
        densities = modeDensities[mode]
        summary = load_summary(runDirs[mode])

        print(f"\n  --- {mode} ({runDirs[mode].name}) ---")

        header = f"  {'Density':<10} {'Grep':>8}"
        for bs in batchSizes:
            header += f" {'bs=' + str(bs):>10}"
        print(header)
        print(f"  {'-' * (10 + 8 + 10 * len(batchSizes))}")

        for d in densities:
            dKey = f"d{d}"
            data = summary.get("results", {}).get(dKey, {})
            grepRecall = data.get("grep", {}).get("recall_upper_bound", 0)
            row = f"  {dKey:<10} {grepRecall:>7.0%}"

            for bs in batchSizes:
                bsKey = f"bs{bs}"
                if bsKey in data:
                    row += f" {data[bsKey]['recall']:>9.1%}"
                else:
                    row += f" {'--':>10}"
            print(row)

        # Category summary for highest density
        if densities:
            dKey = f"d{densities[-1]}"
            data = summary.get("results", {}).get(dKey, {})
            bsKey = f"bs{batchSizes[0]}"
            if bsKey in data and "by_category" in data[bsKey]:
                print(f"\n    Category breakdown ({dKey}, bs={batchSizes[0]}):")
                for cat, m in sorted(data[bsKey]["by_category"].items()):
                    print(f"      {cat:35s}: {m['recalled']}/{m['total']} "
                          f"({m['recall']:.0%})")


# ============================================================================
# GENERIC PAIR COMPARISON
# ============================================================================

def compare_pair(modeA, modeB, runDirs, modeDensities, batchSizes,
                 label="", showFacts=True):
    """
    Compare two modes at overlapping densities.
    Returns (deltas, jaccards) lists for verdict aggregation.
    """
    if modeA not in runDirs or modeB not in runDirs:
        return None, None

    overlap = sorted(set(modeDensities[modeA]) & set(modeDensities[modeB]))
    if not overlap:
        print(f"    No overlapping densities between {modeA} and {modeB}.")
        return None, None

    print(f"\n    {modeA} vs {modeB} -- {label}")
    print(f"    Overlap: {overlap}")

    # Global recall delta
    header = f"    {'d':<8} {modeA:>8} {modeB:>8} {'Delta':>8} {'Jacc':>8}"
    print(header)
    print(f"    {'-' * 42}")

    deltas, jaccards = [], []

    for d in overlap:
        mARecalls, mBRecalls, jacs = [], [], []

        for bs in batchSizes:
            vA = load_verdicts(runDirs[modeA], d, bs)
            vB = load_verdicts(runDirs[modeB], d, bs)
            if vA is None or vB is None:
                continue

            setA = set(v["fact_id"] for v in vA if v.get("recalled"))
            setB = set(v["fact_id"] for v in vB if v.get("recalled"))

            mARecalls.append(len(setA) / len(vA) if vA else 0)
            mBRecalls.append(len(setB) / len(vB) if vB else 0)
            jacs.append(jaccard(setA, setB))

        if mARecalls:
            mA, mB = mean(mARecalls), mean(mBRecalls)
            delta = mB - mA
            j = mean(jacs)
            deltas.append(delta)
            jaccards.append(j)
            print(f"    d{d:<7} {mA:>7.1%} {mB:>7.1%} {delta:>+7.1%} {j:>8.3f}")

    if deltas:
        print(f"    Mean delta: {mean(deltas):+.1%}  |  Mean Jaccard: {mean(jaccards):.3f}")

    # Per-category breakdown
    catPairs = defaultdict(list)
    for d in overlap:
        for bs in batchSizes:
            vA = load_verdicts(runDirs[modeA], d, bs)
            vB = load_verdicts(runDirs[modeB], d, bs)
            if vA is None or vB is None:
                continue
            catA = compute_category_recall(vA)
            catB = compute_category_recall(vB)
            for cat in set(catA.keys()) | set(catB.keys()):
                rA = catA.get(cat, {}).get("recall", 0)
                rB = catB.get(cat, {}).get("recall", 0)
                nA = catA.get(cat, {}).get("total", 0)
                nB = catB.get(cat, {}).get("total", 0)
                if nA > 0 or nB > 0:
                    catPairs[cat].append((rA, rB, nA, nB))

    if catPairs:
        print(f"\n    Per-category (mean across d x bs):")
        header = f"    {'Category':<35s} {modeA:>6} {modeB:>6} {'Delta':>7}"
        print(header)
        print(f"    {'-' * 56}")
        for cat in sorted(catPairs.keys()):
            pairs = catPairs[cat]
            rAm = mean([p[0] for p in pairs if p[2] > 0])
            rBm = mean([p[1] for p in pairs if p[3] > 0])
            delta = rBm - rAm
            print(f"    {cat:<35s} {rAm:>5.0%} {rBm:>5.0%} {delta:>+6.1%}")

    # Per-fact differential (bs=first, representative density)
    if showFacts and overlap:
        bs0 = batchSizes[0]
        for d in overlap:
            mapA = load_verdicts_map(runDirs[modeA], d, bs0)
            mapB = load_verdicts_map(runDirs[modeB], d, bs0)
            if mapA is None or mapB is None:
                continue

            setA = set(fid for fid, v in mapA.items() if v.get("recalled"))
            setB = set(fid for fid, v in mapB.items() if v.get("recalled"))
            # Only show for shared fact_ids (R1↔R2 share facts; R1↔R3 don't)
            sharedFids = set(mapA.keys()) & set(mapB.keys())
            onlyA = (setA & sharedFids) - setB
            onlyB = (setB & sharedFids) - setA

            if onlyA or onlyB:
                print(f"\n    d{d} (bs={bs0}): {modeA}-only={len(onlyA)}, {modeB}-only={len(onlyB)}")
                meta = load_context_meta(runDirs[modeA], d)
                positions = {}
                if meta:
                    positions = {fm["fact_id"]: fm.get("position_pct", -1)
                                 for fm in meta["facts"]}

                for fid in sorted(onlyA)[:10]:
                    cat = mapA[fid].get("question_type", "?")
                    pos = positions.get(fid, -1)
                    posStr = f"@{pos:.0f}%" if pos >= 0 else ""
                    print(f"      {modeA}-only: {fid} [{cat:25s}] {posStr}")
                for fid in sorted(onlyB)[:10]:
                    cat = mapB[fid].get("question_type", "?")
                    pos = positions.get(fid, -1)
                    posStr = f"@{pos:.0f}%" if pos >= 0 else ""
                    print(f"      {modeB}-only: {fid} [{cat:25s}] {posStr}")

    return deltas, jaccards


# ============================================================================
# SECTION 2 -- Truncation effect
# ============================================================================

def section_truncation(modes, runDirs, modeDensities, batchSizes):
    print(f"\n{'-' * 80}")
    print(f"  2. TRUNCATION EFFECT (full -> chopped evidence)")
    print(f"{'-' * 80}")

    allDeltas = {}
    allJaccards = {}

    for modeA, modeB in TRUNCATION_PAIRS:
        if modeA in modes and modeB in modes:
            label = "all categories" if modeA == "R1" else "filtered categories"
            deltas, jaccards = compare_pair(
                modeA, modeB, runDirs, modeDensities, batchSizes,
                label=f"truncation ({label})", showFacts=True)
            if deltas:
                allDeltas[(modeA, modeB)] = deltas
                allJaccards[(modeA, modeB)] = jaccards

    return allDeltas, allJaccards


# ============================================================================
# SECTION 3 -- Filtering effect
# ============================================================================

def section_filtering(modes, runDirs, modeDensities, batchSizes):
    print(f"\n{'-' * 80}")
    print(f"  3. FILTERING EFFECT (all categories -> filtered)")
    print(f"{'-' * 80}")

    allDeltas = {}

    for modeA, modeB in FILTERING_PAIRS:
        if modeA in modes and modeB in modes:
            label = "full evidence" if modeA == "R1" else "chopped evidence"
            deltas, _ = compare_pair(
                modeA, modeB, runDirs, modeDensities, batchSizes,
                label=f"filtering ({label})", showFacts=False)
            if deltas:
                allDeltas[(modeA, modeB)] = deltas

    # Post-hoc filtering comparison (R2 filtered verdicts vs R4)
    if "R2" in modes and "R4" in modes:
        overlap = sorted(set(modeDensities["R2"]) & set(modeDensities["R4"]))
        if overlap:
            print(f"\n    R2 post-hoc filtered vs R4 (context composition effect)")
            print(f"    {'d':<8} {'R2filt':>8} {'R4':>8} {'Delta':>8}")
            print(f"    {'-' * 34}")

            compDeltas = []
            for d in overlap:
                r2f, r4g = [], []
                for bs in batchSizes:
                    v2 = load_verdicts(runDirs["R2"], d, bs)
                    v4 = load_verdicts(runDirs["R4"], d, bs)
                    if v2:
                        v2f = [v for v in v2
                               if v.get("question_type") not in FILTERED_OUT_TYPES]
                        r2f.append(recall_from_verdicts(v2f))
                    if v4:
                        r4g.append(recall_from_verdicts(v4))

                if r2f and r4g:
                    mR2f, mR4 = mean(r2f), mean(r4g)
                    delta = mR4 - mR2f
                    compDeltas.append(delta)
                    print(f"    d{d:<7} {mR2f:>7.1%} {mR4:>7.1%} {delta:>+7.1%}")

            if compDeltas:
                avgDelta = mean(compDeltas)
                print(f"    Mean composition delta: {avgDelta:+.1%}")
                print(f"      >0 = removing hard cats from CONTEXT helps")
                print(f"      ~0 = hard cats don't hurt easy-cat recall")

    return allDeltas


# ============================================================================
# SECTION 4 -- Interaction
# ============================================================================

def section_interaction(truncDeltas, filtDeltas, modes):
    print(f"\n{'-' * 80}")
    print(f"  4. INTERACTION CHECK (2x2 factorial)")
    print(f"{'-' * 80}")

    # Truncation deltas: R1->R2 vs R3->R4
    t12 = truncDeltas.get(("R1", "R2"))
    t34 = truncDeltas.get(("R3", "R4"))

    if t12 and t34:
        m12 = mean(t12)
        m34 = mean(t34)
        interaction = m34 - m12
        print(f"\n  Truncation effect on all categories (R1->R2):  {m12:+.1%}")
        print(f"  Truncation effect on filtered cats  (R3->R4):  {m34:+.1%}")
        print(f"  Interaction (difference of deltas):            {interaction:+.1%}")
        if abs(interaction) < 0.03:
            print(f"  -> Truncation effect is CONSISTENT across category sets")
        else:
            direction = "stronger" if interaction < 0 else "weaker"
            print(f"  -> Truncation is {direction} on filtered categories")

    # Filtering deltas: R1->R3 vs R2->R4
    f13 = filtDeltas.get(("R1", "R3"))
    f24 = filtDeltas.get(("R2", "R4"))

    if f13 and f24:
        m13 = mean(f13)
        m24 = mean(f24)
        interaction = m24 - m13
        print(f"\n  Filtering effect with full evidence  (R1->R3):  {m13:+.1%}")
        print(f"  Filtering effect with chopped evid.  (R2->R4):  {m24:+.1%}")
        print(f"  Interaction (difference of deltas):             {interaction:+.1%}")
        if abs(interaction) < 0.03:
            print(f"  -> Filtering effect is CONSISTENT across evidence formats")
        else:
            direction = "stronger" if interaction < 0 else "weaker"
            print(f"  -> Filtering is {direction} with chopped evidence")

    if not t12 and not t34 and not f13 and not f24:
        print(f"  Not enough modes to compute interaction (need 3+ of R1/R2/R3/R4).")


# ============================================================================
# SECTION 5 -- Per-category density curves
# ============================================================================

def section_category_curves(modes, runDirs, modeDensities, batchSizes):
    print(f"\n{'-' * 80}")
    print(f"  5. PER-CATEGORY DENSITY CURVES (mean across bs)")
    print(f"{'-' * 80}")

    # Collect: categoryData[cat][mode][d] = mean_recall
    categoryData = defaultdict(lambda: defaultdict(dict))

    for mode in modes:
        for d in modeDensities[mode]:
            catRecalls = defaultdict(list)
            for bs in batchSizes:
                verdicts = load_verdicts(runDirs[mode], d, bs)
                if verdicts is None:
                    continue
                for cat, stats in compute_category_recall(verdicts).items():
                    catRecalls[cat].append(stats["recall"])

            for cat, recalls in catRecalls.items():
                categoryData[cat][mode][d] = mean(recalls)

    for cat in sorted(categoryData.keys()):
        modeData = categoryData[cat]
        modesWithCat = [m for m in modes if m in modeData]
        if not modesWithCat:
            continue

        allDens = sorted(set(d for m in modesWithCat for d in modeData[m]))

        print(f"\n  {cat}")
        header = f"  {'d':<8}"
        for mode in modesWithCat:
            header += f" {mode:>8}"
        print(header)
        print(f"  {'-' * (8 + 8 * len(modesWithCat))}")

        for d in allDens:
            row = f"  d{d:<7}"
            anyData = False
            for mode in modesWithCat:
                if d in modeData[mode]:
                    row += f" {modeData[mode][d]:>7.0%}"
                    anyData = True
                else:
                    row += f" {'--':>8}"
            if anyData:
                print(row)


# ============================================================================
# SECTION 6 -- Verdict
# ============================================================================

def section_verdict(modes, runDirs, modeDensities, batchSizes,
                    truncDeltas, filtDeltas):
    print(f"\n{'=' * 80}")
    print(f"  6. VERDICT & RECOMMENDATIONS")
    print(f"{'=' * 80}")

    # Factorial summary
    print(f"\n  2x2 FACTORIAL SUMMARY")
    print(f"  {'':20s} {'Complete':>12} {'Filtered':>12}")
    print(f"  {'-' * 46}")

    for (evLabel, modeComplete, modeFiltered) in [
        ("Full evidence", "R1", "R3"),
        ("Chopped evidence", "R2", "R4"),
    ]:
        row = f"  {evLabel:20s}"
        for mode in [modeComplete, modeFiltered]:
            if mode in modes:
                densities = modeDensities[mode]
                if densities:
                    maxD = densities[-1]
                    recalls = []
                    for bs in batchSizes:
                        v = load_verdicts(runDirs[mode], maxD, bs)
                        if v:
                            recalls.append(recall_from_verdicts(v))
                    if recalls:
                        row += f" {mode} d{maxD}={mean(recalls):.0%}"
                    else:
                        row += f" {mode} d{maxD}=?"
                else:
                    row += f" {mode} (empty)"
            else:
                row += f" {'--':>12}"
        print(row)

    # Effect sizes
    print(f"\n  EFFECT SIZES:")

    for (mA, mB), deltas in truncDeltas.items():
        label = "all cats" if mA == "R1" else "filtered"
        print(f"    Truncation ({mA}->{mB}, {label}): {mean(deltas):+.1%}")

    for (mA, mB), deltas in filtDeltas.items():
        label = "full evid." if mA == "R1" else "chopped"
        print(f"    Filtering  ({mA}->{mB}, {label}): {mean(deltas):+.1%}")

    # Usable densities per mode (recall >= 15%)
    print(f"\n  USABLE DENSITIES (recall >= 15%, mean across bs):")
    for mode in modes:
        usable = []
        for d in modeDensities[mode]:
            recalls = []
            for bs in batchSizes:
                v = load_verdicts(runDirs[mode], d, bs)
                if v:
                    recalls.append(recall_from_verdicts(v))
            if recalls and mean(recalls) >= 0.15:
                usable.append(f"d{d}")
        if usable:
            print(f"    {mode}: {', '.join(usable)}")
        else:
            print(f"    {mode}: none reach 15%")

    # Recommendation
    print(f"\n  RECOMMENDATION FOR COMPACTION TESTING:")
    if "R4" in modes:
        d4 = modeDensities.get("R4", [])
        print(f"    Use R4 (chopped + filtered) -- max range d{d4[0] if d4 else '?'}->d{d4[-1] if d4 else '?'}")
        print(f"    Cleanest signal: no hard categories, compact evidence")
    elif "R2" in modes:
        print(f"    Use R2 (chopped + complete) -- widest density range")
    print(f"    Always report per-category breakdown for transparency")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare v5 recall runs -- 2x2 factorial (R1/R2/R3/R4)")
    parser.add_argument("runs", nargs="*",
                        help="Run directories (auto-detected if omitted)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect latest v5 runs")
    parser.add_argument("--batch-sizes", type=str, default=None)
    args = parser.parse_args()

    runDirs = detect_runs(args)

    if not runDirs:
        print("No v5 run directories found.")
        print("Usage: python compare_runs_v5.py R1_DIR R2_DIR R3_DIR R4_DIR")
        sys.exit(1)

    modes = sorted(m for m in ALL_MODES if m in runDirs)

    modeDensities = {m: discover_densities(runDirs[m]) for m in modes}

    if args.batch_sizes:
        batchSizes = [int(b) for b in args.batch_sizes.split(",")]
    else:
        allBs = set()
        for m in modes:
            allBs.update(discover_batch_sizes(runDirs[m]))
        batchSizes = sorted(allBs)

    if not batchSizes:
        print("No batch sizes found.")
        sys.exit(1)

    # Header
    print("=" * 80)
    print(f"  RECALL v5 -- 2x2 FACTORIAL ANALYSIS")
    print(f"  {'':20s} {'Complete':>12} {'Filtered':>12}")
    print(f"  {'Full evidence':20s} "
          f"{'R1' if 'R1' in modes else '--':>12} "
          f"{'R3' if 'R3' in modes else '--':>12}")
    print(f"  {'Chopped evidence':20s} "
          f"{'R2' if 'R2' in modes else '--':>12} "
          f"{'R4' if 'R4' in modes else '--':>12}")
    for m in modes:
        print(f"  {m}: {runDirs[m].name}  densities={modeDensities[m]}")
    print(f"  Batch sizes: {batchSizes}")
    print("=" * 80)

    # Sections
    section_overview(modes, runDirs, modeDensities, batchSizes)

    truncDeltas, truncJaccards = section_truncation(
        modes, runDirs, modeDensities, batchSizes)

    filtDeltas = section_filtering(
        modes, runDirs, modeDensities, batchSizes)

    section_interaction(truncDeltas or {}, filtDeltas or {}, modes)

    section_category_curves(modes, runDirs, modeDensities, batchSizes)

    section_verdict(modes, runDirs, modeDensities, batchSizes,
                    truncDeltas or {}, filtDeltas or {})

    print()


if __name__ == "__main__":
    main()
