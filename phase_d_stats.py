#!python3
"""
Compute paired statistical tests for the Phase D strategy comparison.

For each (checkpoint, strategy_pair), test whether one strategy's recall is
systematically different from another using:
- Wilcoxon signed-rank test on paired replicates (when ≥4 paired replicates)
- A simple sign test as fallback

Replicates are paired across runs that contain both strategies.

Usage:
    python phase_d_stats.py
"""
import json, os, sys
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
RUNS = [
    ("run_2136", "iterative_v6_R4_20260416_2136", ["S1","S2","S3","S4"]),
    ("run_2249", "iterative_v6_R4_20260427_2249", ["S3","S4"]),
    ("run_2249_v2", "iterative_v6_R4_20260427_2249_qa_v2", ["S3","S4"]),
    ("run_2333", "iterative_v6_R4_20260428_2333", ["S1","S2"]),
    ("run_2333_v2", "iterative_v6_R4_20260428_2333_qa_v2", ["S1","S2"]),
    ("run_29",   "iterative_v6_R4_20260429_2249", ["S1","S2","S3","S4"]),
    ("run_29_v2","iterative_v6_R4_20260429_2249_qa_v2", ["S1","S2","S3","S4"]),
    ("run_30",   "iterative_v6_R4_20260430_1033", ["S1","S2","S3","S4"]),
]
CPS = [("500K","500K"),("1M","1001K"),("2M","2001K"),("3.5M","3501K"),("5M","final_reeval")]
STRATS = ["S1","S2","S3","S4"]


def load_strict(run_dir, cp_dir, sk):
    cp_path = ROOT / run_dir / (f"checkpoint_{cp_dir}" if cp_dir != "final_reeval" else cp_dir)
    sf = cp_path / "summary_strict.json"
    if not sf.exists():
        # rerun_qa_only saves directly as summary.json with strict-judge already
        if run_dir.endswith("_qa_v2"):
            sf = cp_path / "summary.json"
    if not sf.exists():
        return None
    with open(sf) as f:
        d = json.load(f)
    if sk in d.get("results", {}):
        r = d["results"][sk]
        if r.get("facts_total", 0) > 0:
            return r["recall"] * 100
    return None


def collect():
    """Return data[cp][sk] = {run_id: recall_pct}"""
    out = {cp_label: {sk: {} for sk in STRATS} for cp_label, _ in CPS}
    for run_id, run_dir, strats in RUNS:
        for cp_label, cp_dir in CPS:
            for sk in strats:
                v = load_strict(run_dir, cp_dir, sk)
                if v is not None:
                    out[cp_label][sk][run_id] = v
    return out


def paired_test(d_cp, sa, sb):
    """Return (n_paired, mean_a, mean_b, mean_diff, wilcoxon_p, sign_p)."""
    common = sorted(set(d_cp[sa].keys()) & set(d_cp[sb].keys()))
    if len(common) < 2:
        return None
    a = np.array([d_cp[sa][r] for r in common])
    b = np.array([d_cp[sb][r] for r in common])
    diff = a - b
    res = {
        "n": len(common),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)) if len(common) > 1 else 0.0,
        "wins_a": int(np.sum(diff > 0)),
        "wins_b": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
    }
    # Wilcoxon (needs ≥3 non-zero diffs ideally)
    if (diff != 0).sum() >= 2:
        try:
            w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="greater")
            res["wilcoxon_p_a_gt_b"] = float(w_p)
        except Exception:
            res["wilcoxon_p_a_gt_b"] = None
    else:
        res["wilcoxon_p_a_gt_b"] = None
    # Sign test (binomial, alternative=greater)
    n_eff = res["wins_a"] + res["wins_b"]
    if n_eff > 0:
        res["sign_p_a_gt_b"] = float(stats.binomtest(res["wins_a"], n_eff, p=0.5,
                                                      alternative="greater").pvalue)
    else:
        res["sign_p_a_gt_b"] = None
    return res


def fmt_p(p):
    if p is None:
        return "n/a"
    if p < 0.001: return "<0.001"
    if p < 0.01:  return f"{p:.3f}"
    return f"{p:.2f}"


def main():
    data = collect()
    pairs = [("S4","S3"), ("S4","S2"), ("S4","S1"), ("S3","S2"), ("S3","S1"), ("S2","S1")]
    print(f"{'CP':<6} {'A':<3} {'B':<3} {'n':>3} {'mean_A':>7} {'mean_B':>7} "
          f"{'Δ(A−B)':>8} {'σ(Δ)':>6} {'A>B':>4} {'B>A':>4} "
          f"{'Wilcoxon p (A>B)':>17} {'Sign p (A>B)':>13}")
    print("-" * 100)
    rows = []
    for cp_label, _ in CPS:
        for sa, sb in pairs:
            res = paired_test(data[cp_label], sa, sb)
            if not res:
                continue
            row = {"cp": cp_label, "a": sa, "b": sb, **res}
            rows.append(row)
            print(f"{cp_label:<6} {sa:<3} {sb:<3} {res['n']:>3} "
                  f"{res['mean_a']:>6.2f}% {res['mean_b']:>6.2f}% "
                  f"{res['mean_diff']:>+7.2f}pp {res['std_diff']:>5.2f}pp "
                  f"{res['wins_a']:>4} {res['wins_b']:>4} "
                  f"{fmt_p(res['wilcoxon_p_a_gt_b']):>17} "
                  f"{fmt_p(res['sign_p_a_gt_b']):>13}")
        print()
    # Save JSON
    with open(ROOT / "phase_d_stats.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("Saved: phase_d_stats.json")


if __name__ == "__main__":
    main()
