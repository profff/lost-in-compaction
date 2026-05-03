#!python3
"""
Re-run JUST the judge phase on existing QA answers.

Useful when judge phase failed (rate limit, gateway error) but the QA answers
were saved successfully. Cheap: only ~6-15 calls per checkpoint.

Usage:
    ./rejudge_only.py iterative_v6_R4_20260428_2333/checkpoint_2001K \
        --strategies S1 \
        --judge-model claude-haiku-4-5-20251001
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

from benchmark_iterative_v6 import (
    BATCH_JUDGE_PROMPT, BATCH_JUDGE_PROMPT_STRICT, JUDGE_SYSTEM, STRATEGIES,
    parse_llm_json, save_json, compute_metrics, extract_facts,
)


def main():
    parser = argparse.ArgumentParser(description="Re-run judge on existing QA answers")
    parser.add_argument("checkpoint_dir",
                        help="Path to checkpoint directory (e.g. iterative_v6_R4_*/checkpoint_2001K)")
    parser.add_argument("--strategies", default="S1,S2,S3,S4")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--judge-batch-size", type=int, default=15)
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--judge-backend", default="wrapper")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--conversation-meta", default=None,
                        help="Path to conversation meta JSON (auto-detected if omitted)")
    parser.add_argument("--save-suffix", default="_rejudged",
                        help="Suffix for saved files (avoids overwriting originals)")
    parser.add_argument("--strict", action="store_true",
                        help="Use BATCH_JUDGE_PROMPT_STRICT (more conservative recalled criterion)")
    args = parser.parse_args()

    judge_prompt = BATCH_JUDGE_PROMPT_STRICT if args.strict else BATCH_JUDGE_PROMPT
    print(f"  Using {'STRICT' if args.strict else 'STANDARD'} judge prompt")

    cpDir = Path(args.checkpoint_dir)
    if not cpDir.exists():
        print(f"ERROR: checkpoint dir not found: {cpDir}")
        sys.exit(1)

    runDir = cpDir.parent
    configFile = runDir / "config.json"
    with open(configFile) as f:
        config = json.load(f)

    # Auto-detect conversation meta
    convDensity = config["density"]
    convSeed = config["seed"]
    metaFile = None
    if args.conversation_meta:
        metaFile = Path(args.conversation_meta)
    else:
        from glob import glob
        matches = glob(f"data/conversations/v6_R4/d{convDensity}_*_seed{convSeed}_meta.json")
        if matches:
            metaFile = Path(matches[0])
    if not metaFile or not metaFile.exists():
        print(f"ERROR: cannot find conversation meta for d{convDensity} seed{convSeed}")
        sys.exit(1)

    with open(metaFile, encoding="utf-8") as f:
        convMeta = json.load(f)
    facts = extract_facts(convMeta)
    factMeta = convMeta["facts"]
    factById = {f[0]: f for f in facts}

    strategyKeys = [s.strip() for s in args.strategies.split(",")]

    print(f"  Checkpoint: {cpDir}")
    print(f"  Strategies: {strategyKeys}")
    print(f"  Judge model: {args.judge_model}")

    from llm_backend import LLM_CreateBackend
    judgeBackend = LLM_CreateBackend(
        args.judge_backend, model=args.judge_model,
        base_url=args.base_url, workers=args.workers,
    )

    bs = args.batch_size

    # Build judge requests for each strategy
    judgeRequests = []
    answersByStrategy = {}
    for sk in strategyKeys:
        ansFile = cpDir / "answers" / f"{sk}_bs{bs}.json"
        if not ansFile.exists():
            print(f"  [{sk}] SKIP: no answers file")
            continue
        with open(ansFile, encoding="utf-8") as f:
            answers = json.load(f)
        # Filter only facts with real answers (skip [no answer])
        real_answers = {fid: ans for fid, ans in answers.items() if ans != "[no answer]"}
        if not real_answers:
            print(f"  [{sk}] SKIP: all answers are [no answer]")
            continue
        print(f"  [{sk}] {len(real_answers)}/{len(answers)} real answers to judge")
        answersByStrategy[sk] = answers  # keep all for later mapping

        # Build entries for judge
        entries = []
        for fid, ans in real_answers.items():
            if fid not in factById:
                continue
            _, q, expected, kw, _ = factById[fid]
            entries.append({
                "id": fid,
                "question": q,
                "expected_answer": expected,
                "expected_keywords": kw,
                "llm_answer": ans,
            })

        judgeBatchSize = args.judge_batch_size
        for jIdx in range(0, len(entries), judgeBatchSize):
            batch = entries[jIdx:jIdx + judgeBatchSize]
            judgeRequests.append({
                "custom_id": f"judge_{sk}_b{jIdx // judgeBatchSize}",
                "params": {
                    "model": args.judge_model,
                    "max_tokens": 4096,
                    "system": JUDGE_SYSTEM,
                    "messages": [{"role": "user", "content":
                        judge_prompt.format(entries=json.dumps(batch, indent=2))}],
                },
            })

    if not judgeRequests:
        print("Nothing to judge.")
        return

    print(f"\n  Total judge requests: {len(judgeRequests)}")
    judgeResults = judgeBackend.run_requests(judgeRequests)

    # Parse verdicts
    verdictsByStrategy = {sk: [] for sk in answersByStrategy}
    for cid, result in judgeResults.items():
        parts = cid.split("_")
        sk = parts[1]
        if result["status"] != "succeeded":
            print(f"  WARNING: {cid} failed")
            continue
        try:
            verdicts = parse_llm_json(result["text"])
            for v in verdicts:
                v["fact_id"] = v.pop("id", v.get("fact_id", "?"))
            verdictsByStrategy[sk].extend(verdicts)
        except Exception as e:
            print(f"  WARNING: {cid} parse error: {e}")

    # Add [no answer] entries as not recalled (for completeness)
    for sk in answersByStrategy:
        judged_ids = {v["fact_id"] for v in verdictsByStrategy[sk]}
        for fid, ans in answersByStrategy[sk].items():
            if fid not in judged_ids:
                verdictsByStrategy[sk].append({
                    "fact_id": fid,
                    "recalled": False,
                    "accurate": False,
                    "notes": "No answer provided" if ans == "[no answer]" else "Not judged",
                })

    # Save judgments + metrics
    suffix = args.save_suffix
    results_dict = {}
    for sk in answersByStrategy:
        # Save judgments with suffix
        save_json(
            {"verdicts": verdictsByStrategy[sk], "judge_model": args.judge_model,
             "rejudge_timestamp": datetime.now().isoformat()},
            cpDir / "judgments" / f"{sk}_bs{bs}{suffix}.json"
        )
        # Compute metrics
        # Filter factMeta to only those in verdicts
        verdict_ids = {v["fact_id"] for v in verdictsByStrategy[sk]}
        cpFactMeta = [fm for fm in factMeta if fm["fact_id"] in verdict_ids]

        metrics = compute_metrics(verdictsByStrategy[sk], cpFactMeta)
        results_dict[sk] = metrics
        recalled = metrics["facts_recalled"]
        total = metrics["facts_total"]
        print(f"  [{sk}] Recall: {metrics['recall']*100:.1f}% ({recalled}/{total})")

    # Save summary
    save_json(
        {"results": results_dict, "rejudge_timestamp": datetime.now().isoformat(),
         "judge_model": args.judge_model},
        cpDir / f"summary{suffix}.json"
    )

    print(f"\n  Done! Files saved with suffix '{suffix}'")


if __name__ == "__main__":
    main()
