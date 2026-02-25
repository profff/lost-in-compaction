#!python3
"""
Compare N v4 recall runs on the same LongMemEval contexts.

- Cross-run Jaccard (repeatability)
- Recall mean +/- stdev across runs
- Breakdown by LongMemEval question type
- Stable/unstable/never classification across ALL runs

Usage:
    python compare_runs_v4.py recall_v4_20260210_1703 recall_v4_run2 recall_v4_run3
"""

import json
import sys
import math
from pathlib import Path
from itertools import combinations
from collections import defaultdict


DENSITIES = [4, 8, 19]
BATCH_SIZES = [1, 5, 10]
DATA_DIR = Path("data")


def load_verdicts_v4(runDir, density, batchSize):
    """Load recalled fact IDs from a v4 judgment file."""
    path = Path(runDir) / "judgments" / f"d{density}_bs{batchSize}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    recalled = set()
    for batch in data["batches"]:
        for v in batch["verdicts"]:
            if v.get("recalled"):
                recalled.add(v["fact_id"])
    return recalled


def load_verdicts_detail(runDir, density, batchSize):
    """Load full verdict details from a v4 judgment file."""
    path = Path(runDir) / "judgments" / f"d{density}_bs{batchSize}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for batch in data["batches"]:
        for v in batch["verdicts"]:
            result[v["fact_id"]] = {
                "recalled": v.get("recalled", False),
                "accurate": v.get("accurate", False),
                "notes": v.get("notes", ""),
            }
    return result


def load_question_types():
    """Load question types from evidence_longmemeval.json."""
    path = DATA_DIR / "evidence_longmemeval.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    types = {}
    for entry in data:
        types[entry["fact_id"]] = entry.get("question_type", "unknown")
    return types


def load_fact_meta(density):
    """Load fact metadata from context meta files."""
    path = DATA_DIR / "contexts" / "recall_190K" / f"d{density}_seed42_meta.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {fm["fact_id"]: fm for fm in data["facts"]}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def mean(vals):
    return sum(vals) / len(vals) if vals else 0


def stdev(vals):
    if len(vals) < 2:
        return 0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def main():
    if len(sys.argv) > 1:
        runs = [Path(p) for p in sys.argv[1:] if Path(p).is_dir()]
    else:
        runs = sorted(Path(".").glob("recall_v4_*"))
        runs = [r for r in runs if (r / "summary.json").exists()]

    if len(runs) < 2:
        print(f"Need at least 2 runs. Found: {[str(r) for r in runs]}")
        sys.exit(1)

    qTypes = load_question_types()

    print("=" * 80)
    print(f"  RECALL v4 — CROSS-RUN ANALYSIS ({len(runs)} runs)")
    print(f"  Runs: {', '.join(r.name for r in runs)}")
    print("=" * 80)

    # ===== 1. Recall mean +/- stdev =====
    print(f"\n{'-' * 80}")
    print(f"  1. RECALL: mean +/- stdev across {len(runs)} runs")
    print(f"{'-' * 80}")

    header = f"  {'Density':<10}"
    for bs in BATCH_SIZES:
        header += f" {'bs=' + str(bs):>16}"
    print(header)
    print(f"  {'-' * 58}")

    for density in DENSITIES:
        row = f"  d{density:<9}"
        for bs in BATCH_SIZES:
            recalls = []
            for runDir in runs:
                r = load_verdicts_v4(runDir, density, bs)
                if r is not None:
                    meta = load_fact_meta(density)
                    nFacts = len(meta)
                    recalls.append(len(r) / nFacts)
            if recalls:
                m = mean(recalls)
                s = stdev(recalls)
                row += f" {m*100:>6.1f}+/-{s*100:4.1f}pp"
            else:
                row += f" {'N/A':>16}"
        print(row)

    # ===== 2. Inter-run Jaccard =====
    print(f"\n{'-' * 80}")
    print(f"  2. INTER-RUN JACCARD (same bs, different run = repeatability)")
    print(f"{'-' * 80}")

    header = f"  {'Density':<10}"
    for bs in BATCH_SIZES:
        header += f" {'bs=' + str(bs):>10}"
    header += f" {'Mean':>10}"
    print(header)
    print(f"  {'-' * 50}")

    allInterRun = []

    for density in DENSITIES:
        row = f"  d{density:<9}"
        rowJaccards = []
        for bs in BATCH_SIZES:
            pairJaccards = []
            for i, j in combinations(range(len(runs)), 2):
                r1 = load_verdicts_v4(runs[i], density, bs)
                r2 = load_verdicts_v4(runs[j], density, bs)
                if r1 is not None and r2 is not None:
                    pairJaccards.append(jaccard(r1, r2))

            if pairJaccards:
                m = mean(pairJaccards)
                rowJaccards.append(m)
                allInterRun.append(m)
                row += f" {m:>10.3f}"
            else:
                row += f" {'N/A':>10}"

        rowMean = mean(rowJaccards) if rowJaccards else 0
        row += f" {rowMean:>10.3f}"
        print(row)

    overallInterRun = mean(allInterRun)
    print(f"\n  Overall mean inter-run Jaccard: {overallInterRun:.3f}")

    # ===== 3. Inter-bs Jaccard =====
    print(f"\n{'-' * 80}")
    print(f"  3. INTER-BS JACCARD (same run, different bs = batch size effect)")
    print(f"{'-' * 80}")

    allInterBs = []
    for density in DENSITIES:
        runJaccards = []
        for runDir in runs:
            sets = {}
            for bs in BATCH_SIZES:
                r = load_verdicts_v4(runDir, density, bs)
                if r is not None:
                    sets[bs] = r
            if len(sets) >= 2:
                pairJ = [jaccard(s1, s2) for (_, s1), (_, s2) in combinations(sets.items(), 2)]
                runJaccards.append(mean(pairJ))

        if runJaccards:
            m = mean(runJaccards)
            allInterBs.append(m)
            print(f"  d{density}: mean={m:.3f} (across {len(runJaccards)} runs)")

    overallInterBs = mean(allInterBs)
    print(f"\n  Overall mean inter-bs Jaccard: {overallInterBs:.3f}")

    # ===== 4. Per-fact cross-run analysis =====
    print(f"\n{'-' * 80}")
    print(f"  4. PER-FACT ANALYSIS (across all runs & batch sizes)")
    print(f"{'-' * 80}")

    for density in DENSITIES:
        meta = load_fact_meta(density)
        nFacts = len(meta)
        nCells = len(runs) * len(BATCH_SIZES)  # total run×bs combinations

        # Count how many times each fact is recalled
        recallCounts = defaultdict(int)
        for runDir in runs:
            for bs in BATCH_SIZES:
                r = load_verdicts_v4(runDir, density, bs)
                if r is not None:
                    for fid in r:
                        recallCounts[fid] += 1

        # Classify facts
        stableFacts = []   # recalled in ALL cells
        frequentFacts = [] # recalled in >50% of cells
        unstableFacts = [] # recalled in 1-50% of cells
        neverFacts = []    # recalled in 0 cells

        for fid in sorted(meta.keys()):
            count = recallCounts.get(fid, 0)
            qType = qTypes.get(fid, "unknown")
            pos = meta[fid]["position_pct"]
            info = (fid, qType, pos, count, nCells)

            if count == nCells:
                stableFacts.append(info)
            elif count > nCells / 2:
                frequentFacts.append(info)
            elif count > 0:
                unstableFacts.append(info)
            else:
                neverFacts.append(info)

        print(f"\n  d{density} ({nFacts} facts, {nCells} cells = {len(runs)} runs x {len(BATCH_SIZES)} bs)")
        print(f"  {'-' * 70}")

        def printGroup(label, facts):
            if not facts:
                print(f"    {label}: (none)")
                return
            print(f"    {label} ({len(facts)}):")
            for fid, qt, pos, count, total in facts:
                print(f"      {fid} [{qt:28s}] @{pos:5.1f}%  recalled {count}/{total}")

        printGroup("STABLE (all cells)", stableFacts)
        printGroup("FREQUENT (>50%)", frequentFacts)
        printGroup("UNSTABLE (1-50%)", unstableFacts)
        printGroup("NEVER (0%)", neverFacts)

    # ===== 5. Breakdown by question type =====
    print(f"\n{'-' * 80}")
    print(f"  5. RECALL BY QUESTION TYPE (across all densities, all runs)")
    print(f"{'-' * 80}")

    # Collect all facts across all densities
    typeStats = defaultdict(lambda: {"total": 0, "recalled": 0, "accurate": 0,
                                      "cells": 0, "facts": []})

    for density in DENSITIES:
        meta = load_fact_meta(density)
        for fid, fm in meta.items():
            qType = qTypes.get(fid, "unknown")
            typeStats[qType]["facts"].append(fid)

            for runDir in runs:
                for bs in BATCH_SIZES:
                    typeStats[qType]["cells"] += 1
                    details = load_verdicts_detail(runDir, density, bs)
                    v = details.get(fid, {})
                    if v.get("recalled"):
                        typeStats[qType]["recalled"] += 1
                    if v.get("accurate"):
                        typeStats[qType]["accurate"] += 1

    print(f"\n  {'Question Type':<32s} {'Facts':>5} {'Cells':>6}  {'Recall':>8} {'Accuracy':>8}")
    print(f"  {'-' * 65}")

    for qType in sorted(typeStats.keys()):
        st = typeStats[qType]
        nFacts = len(set(st["facts"]))  # unique facts
        cells = st["cells"]
        recallPct = st["recalled"] / cells * 100 if cells else 0
        accPct = st["accurate"] / cells * 100 if cells else 0
        print(f"  {qType:<32s} {nFacts:>5} {cells:>6}  {recallPct:>7.1f}% {accPct:>7.1f}%")

    # ===== 6. Stable core & union =====
    print(f"\n{'-' * 80}")
    print(f"  6. STABLE CORE & UNION")
    print(f"{'-' * 80}")

    for density in DENSITIES:
        stableCore = None
        union = set()
        for runDir in runs:
            for bs in BATCH_SIZES:
                r = load_verdicts_v4(runDir, density, bs)
                if r is None:
                    continue
                union |= r
                if stableCore is None:
                    stableCore = r.copy()
                else:
                    stableCore &= r

        meta = load_fact_meta(density)
        nFacts = len(meta)

        if stableCore:
            coreTypes = [qTypes.get(fid, "?") for fid in sorted(stableCore)]
            print(f"  d{density} core: {len(stableCore)}/{nFacts} "
                  f"({len(stableCore)/nFacts*100:.0f}%) {sorted(stableCore)}")
            print(f"         types: {coreTypes}")
        else:
            print(f"  d{density} core: 0/{nFacts}")

        unionTypes = defaultdict(int)
        for fid in union:
            unionTypes[qTypes.get(fid, "?")] += 1
        print(f"  d{density} union: {len(union)}/{nFacts} "
              f"({len(union)/nFacts*100:.0f}%) types={dict(unionTypes)}")

    # ===== Verdict =====
    print(f"\n{'=' * 80}")
    print(f"  VERDICT")
    print(f"{'=' * 80}")

    ratio = overallInterRun / overallInterBs if overallInterBs > 0 else float('inf')
    print(f"  Inter-run Jaccard (repeatability): {overallInterRun:.3f}")
    print(f"  Inter-bs  Jaccard (batch effect):  {overallInterBs:.3f}")
    print(f"  Ratio: {ratio:.1f}x "
          f"(signal {'>' if overallInterRun > overallInterBs else '<'} noise)")

    # Recommendations
    print(f"\n  QUESTION TYPE RECOMMENDATIONS:")
    for qType in sorted(typeStats.keys()):
        st = typeStats[qType]
        cells = st["cells"]
        recallPct = st["recalled"] / cells * 100 if cells else 0
        nFacts = len(set(st["facts"]))
        if recallPct >= 20:
            print(f"    OK  {qType}: {recallPct:.0f}% recall, {nFacts} facts")
        else:
            print(f"    SKIP {qType}: {recallPct:.0f}% recall (too low for compaction testing)")


if __name__ == "__main__":
    main()
