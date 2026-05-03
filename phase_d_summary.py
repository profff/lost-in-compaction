#!python3
"""
Consolidate Phase D results across all runs into a single JSON for paper figures.

Output: phase_d_consolidated.json
"""

import json
import os
from pathlib import Path
from datetime import datetime

CHECKPOINTS = [
    ("500K", 500_000),
    ("1001K", 1_000_000),
    ("2001K", 2_000_000),
    ("3501K", 3_500_000),
    ("5M", 5_000_000),
]

RUNS = [
    {
        "id": "run_2136",
        "dir": "iterative_v6_R4_20260416_2136",
        "label": "gateway, Sonnet QA + Sonnet judge",
        "qa_model": "claude-sonnet-4-20250514",
        "judge_model": "claude-sonnet-4-20250514",
        "strategies": ["S1", "S2", "S3", "S4"],
        "infra": "gateway",
    },
    {
        "id": "run_2249",
        "dir": "iterative_v6_R4_20260427_2249",
        "label": "gateway, Sonnet QA + Haiku judge (S3+S4)",
        "qa_model": "claude-sonnet-4-20250514",
        "judge_model": "claude-haiku-4-5-20251001",
        "strategies": ["S3", "S4"],
        "infra": "gateway",
    },
    {
        "id": "run_2333",
        "dir": "iterative_v6_R4_20260428_2333",
        "label": "gateway, Sonnet QA + Haiku judge (S1+S2)",
        "qa_model": "claude-sonnet-4-20250514",
        "judge_model": "claude-haiku-4-5-20251001",
        "strategies": ["S1", "S2"],
        "infra": "gateway",
    },
    {
        "id": "run_29_2249",
        "dir": "iterative_v6_R4_20260429_2249",
        "label": "batch API, Sonnet QA + Haiku judge (full S1-S4)",
        "qa_model": "claude-sonnet-4-20250514",
        "judge_model": "claude-haiku-4-5-20251001",
        "strategies": ["S1", "S2", "S3", "S4"],
        "infra": "batch_api",
    },
]


def classify_status(answers, verdicts):
    """Determine if a checkpoint result is valid, qa_fail, judge_fail, etc."""
    total = len(answers)
    no_ans = sum(1 for v in answers.values() if v == "[no answer]")
    real_ans = total - no_ans
    n_verdicts = len(verdicts)

    if total == 0:
        return "missing"
    if real_ans == 0:
        return "qa_total_fail"
    if real_ans < total * 0.5:
        return "qa_partial_fail"
    if n_verdicts == 0:
        return "judge_total_fail"
    if n_verdicts < real_ans * 0.5:
        return "judge_partial_fail"
    return "valid"


def load_data(cp_dir, sk, batch_size=5):
    """Load answers + verdicts for a strategy at a checkpoint."""
    ans_file = cp_dir / "answers" / f"{sk}_bs{batch_size}.json"
    rejudged_file = cp_dir / "judgments" / f"{sk}_bs{batch_size}_rejudged.json"
    judge_file = cp_dir / "judgments" / f"{sk}_bs{batch_size}.json"

    if not ans_file.exists():
        return None, None
    with open(ans_file, encoding="utf-8") as f:
        answers = json.load(f)

    # Prefer rejudged if exists
    chosen_judge = rejudged_file if rejudged_file.exists() else judge_file
    if not chosen_judge.exists():
        return answers, None
    with open(chosen_judge, encoding="utf-8") as f:
        jdata = json.load(f)
    verdicts = jdata.get("verdicts", [])
    return answers, verdicts


def get_recall_from_summary(cp_dir, sk, prefer_strict=False):
    """Get recall from summary files. If prefer_strict, look for *_strict variants first."""
    if prefer_strict:
        candidates = ("summary_strict.json", "summary_recomputed.json",
                      "summary_rejudged.json", "summary_reeval.json", "summary.json")
    else:
        candidates = ("summary_recomputed.json", "summary_rejudged.json",
                      "summary_reeval.json", "summary.json")
    for fname in candidates:
        sf = cp_dir / fname
        if sf.exists():
            with open(sf) as f:
                d = json.load(f)
            if "results" in d and sk in d["results"]:
                r = d["results"][sk]
                if r.get("facts_total", 0) > 0:
                    return r
    return None


def main():
    out = {
        "generated": datetime.now().isoformat(),
        "checkpoints": [{"label": l, "tokens": t} for l, t in CHECKPOINTS],
        "runs": {},
    }

    for run_meta in RUNS:
        run_dir = Path(run_meta["dir"])
        if not run_dir.exists():
            continue
        run_data = {
            "meta": run_meta,
            "results": {},  # results[strategy][checkpoint] = {recall, status, ...}
        }

        for sk in run_meta["strategies"]:
            run_data["results"][sk] = {}
            for cp_label, cp_tokens in CHECKPOINTS:
                if cp_label == "5M":
                    cp_dir = run_dir / "final_reeval"
                else:
                    cp_dir = run_dir / f"checkpoint_{cp_label}"

                if not cp_dir.exists():
                    run_data["results"][sk][cp_label] = {"status": "missing"}
                    continue

                # Determine raw status from answers/verdicts
                answers, verdicts = load_data(cp_dir, sk)
                if answers is None:
                    status = "missing"
                else:
                    status = classify_status(answers, verdicts or [])

                # Get recall from best available summary
                metrics = get_recall_from_summary(cp_dir, sk)

                entry = {"status": status}
                if metrics:
                    entry["recall"] = metrics["recall"]
                    entry["facts_recalled"] = metrics["facts_recalled"]
                    entry["facts_total"] = metrics["facts_total"]
                    if "grep" in metrics:
                        entry["grep_upper_bound"] = metrics["grep"].get("recall_upper_bound", 0)
                run_data["results"][sk][cp_label] = entry

        out["runs"][run_meta["id"]] = run_data

    out_path = Path("phase_d_consolidated.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Print human-readable summary
    print(f"\n{'='*90}")
    print("PHASE D — CONSOLIDATED RESULTS")
    print('='*90)
    for run_id, rd in out["runs"].items():
        print(f"\n{run_id}: {rd['meta']['label']}")
        print(f"  {'CP':<8} {'S1':>14} {'S2':>14} {'S3':>14} {'S4':>14}")
        print('  ' + '-'*72)
        for cp_label, _ in CHECKPOINTS:
            row = [cp_label]
            for sk in ["S1", "S2", "S3", "S4"]:
                e = rd["results"].get(sk, {}).get(cp_label, {"status": "missing"})
                if "recall" in e:
                    row.append(f"{e['recall']*100:>5.1f}% ({e['facts_total']:>3})")
                elif e["status"] in ("qa_total_fail", "qa_partial_fail"):
                    row.append("    QA_FAIL")
                elif e["status"] in ("judge_total_fail", "judge_partial_fail"):
                    row.append(" JUDGE_FAIL")
                else:
                    row.append("           -")
            print(f"  {row[0]:<8} {row[1]:>14} {row[2]:>14} {row[3]:>14} {row[4]:>14}")

    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
