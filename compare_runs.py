#!python3
"""
Compare N recall test runs on the same contexts.

Computes inter-run Jaccard (repeatability) vs inter-batch-size Jaccard (batch effect).
Aggregates recall across runs (mean + stdev).

Usage:
    python compare_runs.py recall_test_stage1_20260210_0954 recall_test_stage1_20260210_run2
    python compare_runs.py recall_test_*   (glob)
"""

import json
import sys
import math
from pathlib import Path
from itertools import combinations


DENSITIES = [4, 8, 19, 50, 100]
BATCH_SIZES = [1, 5, 10, 15]


def load_verdicts(runDir, density, batchSize):
    """Load recalled fact IDs from a judgment file."""
    dKey = f"d{density}"
    path = Path(runDir) / "judgments" / f"{dKey}_C0_bs{batchSize}_r0.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    recalled = set()
    for batch in data["batches"]:
        for v in batch["verdicts"]:
            if v.get("recalled"):
                recalled.add(v["fact_id"])
    return recalled


def load_recall(runDir, density, batchSize):
    """Load recall percentage from summary."""
    path = Path(runDir) / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    dKey = f"d{density}"
    bsKey = f"batch_{batchSize}"
    try:
        return data["stage1"][dKey][bsKey]["recall"]
    except KeyError:
        return None


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


def find_runs():
    """Find all recall_test_stage1* directories."""
    base = Path(".")
    runs = sorted(base.glob("recall_test_stage1_*"))
    # Only keep dirs with a summary.json
    return [r for r in runs if (r / "summary.json").exists()]


def main():
    if len(sys.argv) > 1:
        runs = [Path(p) for p in sys.argv[1:] if Path(p).is_dir()]
    else:
        runs = find_runs()

    if len(runs) < 2:
        print(f"Need at least 2 runs. Found: {[str(r) for r in runs]}")
        sys.exit(1)

    runLabels = [r.name.split("_")[-1] if "run" in r.name else r.name[-4:] for r in runs]
    print("=" * 80)
    print(f"  REPEATABILITY ANALYSIS -- {len(runs)} runs")
    print(f"  Runs: {', '.join(r.name for r in runs)}")
    print("=" * 80)

    # ===== 1. Aggregated recall (mean +/- stdev across runs) =====
    print(f"\n{'-' * 80}")
    print(f"  1. RECALL: mean +/- stdev across {len(runs)} runs")
    print(f"{'-' * 80}")

    header = f"  {'Density':<10}"
    for bs in BATCH_SIZES:
        header += f" {'bs=' + str(bs):>16}"
    print(header)
    print(f"  {'-' * 74}")

    for density in DENSITIES:
        row = f"  d{density:<9}"
        for bs in BATCH_SIZES:
            recalls = []
            for runDir in runs:
                r = load_recall(runDir, density, bs)
                if r is not None:
                    recalls.append(r)
            if recalls:
                m = mean(recalls)
                s = stdev(recalls)
                row += f" {m*100:>6.1f}+/-{s*100:4.1f}pp"
            else:
                row += f" {'N/A':>16}"
        print(row)

    # ===== 2. Inter-run Jaccard (pairwise) =====
    print(f"\n{'-' * 80}")
    print(f"  2. INTER-RUN JACCARD (same bs, different run = repeatability)")
    print(f"{'-' * 80}")

    header = f"  {'Density':<10}"
    for bs in BATCH_SIZES:
        header += f" {'bs=' + str(bs):>10}"
    header += f" {'Mean':>10}"
    print(header)
    print(f"  {'-' * 60}")

    allInterRun = []
    interRunByDensity = {}

    for density in DENSITIES:
        row = f"  d{density:<9}"
        rowJaccards = []
        for bs in BATCH_SIZES:
            pairJaccards = []
            for i, j in combinations(range(len(runs)), 2):
                r1 = load_verdicts(runs[i], density, bs)
                r2 = load_verdicts(runs[j], density, bs)
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
        interRunByDensity[density] = rowMean
        row += f" {rowMean:>10.3f}"
        print(row)

    overallMean = mean(allInterRun)
    print(f"\n  Overall mean inter-run Jaccard: {overallMean:.3f}")

    # ===== 3. Inter-bs Jaccard (averaged across runs) =====
    print(f"\n{'-' * 80}")
    print(f"  3. INTER-BS JACCARD (same run, different bs = batch size effect)")
    print(f"{'-' * 80}")

    allInterBs = []
    for density in DENSITIES:
        runJaccards = []
        for runDir in runs:
            sets = {}
            for bs in BATCH_SIZES:
                r = load_verdicts(runDir, density, bs)
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

    # ===== 4. Stable core =====
    print(f"\n{'-' * 80}")
    print(f"  4. STABLE CORE (recalled in ALL runs AND ALL batch sizes)")
    print(f"{'-' * 80}")

    for density in DENSITIES:
        allRecalled = None
        for runDir in runs:
            for bs in BATCH_SIZES:
                r = load_verdicts(runDir, density, bs)
                if r is None:
                    continue
                if allRecalled is None:
                    allRecalled = r.copy()
                else:
                    allRecalled = allRecalled & r

        if allRecalled:
            print(f"  d{density}: {len(allRecalled)} facts: {sorted(allRecalled)}")
        else:
            print(f"  d{density}: 0 facts")

    # ===== 5. Union recall (recalled at least once across all runs & bs) =====
    print(f"\n{'-' * 80}")
    print(f"  5. UNION (recalled at least once across all runs & batch sizes)")
    print(f"{'-' * 80}")

    for density in DENSITIES:
        anyRecalled = set()
        for runDir in runs:
            for bs in BATCH_SIZES:
                r = load_verdicts(runDir, density, bs)
                if r is not None:
                    anyRecalled |= r
        print(f"  d{density}: {len(anyRecalled)}/{density} facts "
              f"({len(anyRecalled)/density*100:.0f}%)")

    # ===== Verdict =====
    print(f"\n{'=' * 80}")
    print(f"  VERDICT")
    print(f"{'=' * 80}")

    goodCells = sum(1 for j in allInterRun if j >= 0.7)
    okCells = sum(1 for j in allInterRun if 0.4 <= j < 0.7)
    badCells = sum(1 for j in allInterRun if j < 0.4)
    total = len(allInterRun)

    print(f"  Inter-run Jaccard (repeatability): {overallMean:.3f}")
    print(f"  Inter-bs  Jaccard (batch effect):  {overallInterBs:.3f}")
    print(f"  Ratio: {overallMean/overallInterBs:.1f}x "
          f"(signal {'>' if overallMean > overallInterBs else '<'} noise)")
    print(f"  Good (>=0.7): {goodCells}/{total}  "
          f"OK (0.4-0.7): {okCells}/{total}  "
          f"Bad (<0.4): {badCells}/{total}")

    if overallMean >= 0.7:
        print(f"\n  => Measurement is REPEATABLE. Batch size effect is real.")
    elif overallMean >= 0.4:
        print(f"\n  => Measurement is NOISY but SIGNAL > NOISE.")
        print(f"     Aggregate across 3+ runs for reliable results.")
    else:
        print(f"\n  => Measurement is UNRELIABLE.")
        print(f"     Fundamental methodology change needed.")


if __name__ == "__main__":
    main()
