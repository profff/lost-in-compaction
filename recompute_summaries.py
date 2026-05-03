#!python3
"""
Recompute checkpoint summaries from existing answers + judgments files.

Use case: when judgments were re-run (rerun, rejudge_only) but the summary.json
still reflects the original failed run.

Scans all iterative_v6_R4_*/checkpoint_*/ directories and updates summary.json
from the actual content of answers/ + judgments/ subdirs.

Usage:
    ./recompute_summaries.py iterative_v6_R4_20260427_2249
    ./recompute_summaries.py iterative_v6_R4_20260427_2249 --use-rejudged
    ./recompute_summaries.py --all  # scan all runs
"""

import json
import argparse
import sys
import glob
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

from benchmark_iterative_v6 import (
    STRATEGIES, compute_metrics, extract_facts,
    grep_keywords, summarize_grep,
)


def find_meta_for_run(run_dir):
    """Auto-detect conversation meta from run config."""
    configFile = run_dir / "config.json"
    if not configFile.exists():
        return None
    with open(configFile) as f:
        config = json.load(f)
    convDensity = config["density"]
    convSeed = config["seed"]
    matches = glob.glob(f"data/conversations/v6_R4/d{convDensity}_*_seed{convSeed}_meta.json")
    if matches:
        return Path(matches[0])
    return None


def recompute_checkpoint(cp_dir, factMeta, factById, batch_size=5,
                         use_rejudged=False, save_suffix=""):
    """Recompute summary for one checkpoint dir from disk content."""
    answers_dir = cp_dir / "answers"
    judgments_dir = cp_dir / "judgments"
    if not answers_dir.exists() or not judgments_dir.exists():
        return None

    suffix = "_rejudged" if use_rejudged else ""
    results = {}
    for sk in ["S1", "S2", "S3", "S4"]:
        ansFile = answers_dir / f"{sk}_bs{batch_size}.json"
        judgeFile = judgments_dir / f"{sk}_bs{batch_size}{suffix}.json"
        if not judgeFile.exists() and use_rejudged:
            # Fallback to non-rejudged file
            judgeFile = judgments_dir / f"{sk}_bs{batch_size}.json"
        if not ansFile.exists() or not judgeFile.exists():
            continue

        with open(ansFile, encoding="utf-8") as f:
            answers = json.load(f)
        with open(judgeFile, encoding="utf-8") as f:
            jdata = json.load(f)
        verdicts = jdata.get("verdicts", [])

        if not verdicts:
            continue  # no verdicts, skip (will leave existing summary as-is)

        # Filter factMeta to only those present in verdicts (fed at this checkpoint)
        verdict_ids = {v["fact_id"] for v in verdicts}
        cpFactMeta = [fm for fm in factMeta if fm["fact_id"] in verdict_ids]

        metrics = compute_metrics(verdicts, cpFactMeta)

        # Try to add grep info if grep file exists
        grepFile = cp_dir / "grep" / f"{sk}.json"
        if grepFile.exists():
            with open(grepFile, encoding="utf-8") as f:
                grep_results = json.load(f)
            grep_summary = summarize_grep(grep_results)
            metrics["grep"] = grep_summary

        results[sk] = metrics

    if not results:
        return None

    out = {
        "results": results,
        "recompute_timestamp": datetime.now().isoformat(),
        "source": "recompute_summaries.py",
    }
    if use_rejudged:
        out["used_rejudged"] = True

    out_path = cp_dir / f"summary_recomputed{save_suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return results, out_path


def process_run(run_dir, batch_size=5, use_rejudged=False):
    """Process all checkpoints in a run directory."""
    print(f"\n=== {run_dir} ===")

    # Find conversation meta
    metaFile = find_meta_for_run(run_dir)
    if not metaFile:
        print(f"  ERROR: cannot find conversation meta")
        return

    with open(metaFile, encoding="utf-8") as f:
        convMeta = json.load(f)
    facts = extract_facts(convMeta)
    factMeta = convMeta["facts"]
    factById = {f[0]: f for f in facts}

    # Process each checkpoint
    for cp_dir in sorted(run_dir.glob("checkpoint_*")):
        out = recompute_checkpoint(cp_dir, factMeta, factById, batch_size, use_rejudged)
        if out is None:
            print(f"  {cp_dir.name}: no data to recompute")
            continue
        results, out_path = out
        print(f"  {cp_dir.name}:")
        for sk, m in results.items():
            print(f"    {sk}: recall={m['recall']*100:.1f}% ({m['facts_recalled']}/{m['facts_total']})")

    # Also process final eval if exists
    for final_dir_name in ["final_reeval"]:
        final_dir = run_dir / final_dir_name
        if final_dir.exists():
            out = recompute_checkpoint(final_dir, factMeta, factById, batch_size, use_rejudged)
            if out:
                results, out_path = out
                print(f"  {final_dir_name}:")
                for sk, m in results.items():
                    print(f"    {sk}: recall={m['recall']*100:.1f}% ({m['facts_recalled']}/{m['facts_total']})")


def main():
    parser = argparse.ArgumentParser(description="Recompute summaries from disk content")
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--all", action="store_true", help="Process all iterative_v6_R4_* dirs")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--use-rejudged", action="store_true",
                        help="Prefer *_rejudged.json files over originals")
    args = parser.parse_args()

    if args.all:
        for run_dir in sorted(glob.glob("iterative_v6_R4_*")):
            process_run(Path(run_dir), args.batch_size, args.use_rejudged)
    elif args.run_dir:
        process_run(Path(args.run_dir), args.batch_size, args.use_rejudged)
    else:
        print("Usage: recompute_summaries.py <run_dir> | --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
