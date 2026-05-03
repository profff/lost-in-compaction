#!python3
"""
Re-run QA + Judge on existing snapshots from a previous run, saving results
to a new output directory.

Use case: measure QA variance by re-asking the same questions on the same
contexts. Compaction phase is not redone — we use the saved snapshot contexts.

Usage:
    ./rerun_qa_only.py iterative_v6_R4_20260429_2249 \
        --output-dir iterative_v6_R4_qa_replicate_v2 \
        --strategies S1,S2,S3,S4 \
        --judge-model claude-haiku-4-5-20251001 \
        --backend anthropic_batch \
        --judge-backend anthropic_batch
"""

import json
import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

from benchmark_iterative_v6 import (
    BATCH_QUESTION_PROMPT, BATCH_JUDGE_PROMPT, BATCH_JUDGE_PROMPT_STRICT,
    JUDGE_SYSTEM, SYSTEM_PROMPT, STRATEGIES,
    parse_llm_json, save_json, compute_metrics,
    grep_keywords, summarize_grep, extract_facts,
)


def get_fed_facts(tracking):
    fed = set()
    for fid, info in tracking.get("fact_status", {}).items():
        if info["status"] != "not_yet_fed":
            fed.add(fid)
    return fed


def main():
    parser = argparse.ArgumentParser(description="Rerun QA+Judge from existing snapshots")
    parser.add_argument("source_run", help="Source run dir with snapshots")
    parser.add_argument("--output-dir", default=None,
                        help="New output dir (default: source_run_qa_v<N>)")
    parser.add_argument("--strategies", default="S1,S2,S3,S4")
    parser.add_argument("--checkpoints", default="500K,1001K,2001K,3501K,5M",
                        help="Checkpoints to evaluate (5M = final)")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--backend", default="anthropic_batch")
    parser.add_argument("--judge-backend", default="anthropic_batch")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--strict-judge", action="store_true",
                        help="Use BATCH_JUDGE_PROMPT_STRICT")
    args = parser.parse_args()

    src = Path(args.source_run)
    if not src.exists():
        print(f"ERROR: source run dir not found: {src}")
        sys.exit(1)

    # Determine output dir
    if args.output_dir:
        outDir = Path(args.output_dir)
    else:
        # Auto: source name + _qa_v<N>
        i = 2
        while True:
            candidate = Path(f"{src.name}_qa_v{i}")
            if not candidate.exists():
                outDir = candidate
                break
            i += 1
    outDir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {outDir}")

    # Copy strategies/ subdir (snapshots + final_context) to new dir
    print(f"  Copying snapshots from {src}/strategies/ ...")
    src_strats = src / "strategies"
    dst_strats = outDir / "strategies"
    if not dst_strats.exists():
        shutil.copytree(src_strats, dst_strats)

    # Copy config and update timestamp
    with open(src / "config.json") as f:
        config = json.load(f)
    config["replicate_of"] = str(src)
    config["replicate_timestamp"] = datetime.now().isoformat()
    config["judge_prompt"] = "STRICT" if args.strict_judge else "STANDARD"
    save_json(config, outDir / "config.json")

    # Load conversation meta
    convDensity = config["density"]
    convSeed = config["seed"]
    from glob import glob
    matches = glob(f"data/conversations/v6_R4/d{convDensity}_*_seed{convSeed}_meta.json")
    if not matches:
        print("ERROR: cannot find conversation meta")
        sys.exit(1)
    metaFile = Path(matches[0])
    with open(metaFile, encoding="utf-8") as f:
        convMeta = json.load(f)
    facts = extract_facts(convMeta)
    factMeta = convMeta["facts"]
    factById = {f[0]: f for f in facts}
    factMetaById = {fm["fact_id"]: fm for fm in factMeta}

    strategyKeys = [s.strip() for s in args.strategies.split(",")]
    checkpoints = [c.strip() for c in args.checkpoints.split(",")]

    # Setup backends
    from llm_backend import LLM_CreateBackend
    qaBackend = LLM_CreateBackend(
        args.backend, model=args.model,
        base_url=args.base_url, workers=args.workers,
    )
    judgeBackend = LLM_CreateBackend(
        args.judge_backend, model=args.judge_model,
        base_url=args.base_url, workers=args.workers,
    )

    judge_prompt = BATCH_JUDGE_PROMPT_STRICT if args.strict_judge else BATCH_JUDGE_PROMPT
    print(f"  Using {'STRICT' if args.strict_judge else 'STANDARD'} judge prompt")

    # Map checkpoint name to dir
    cpNameMap = {"500K": "500K", "1001K": "1001K", "2001K": "2001K", "3501K": "3501K",
                 "1M": "1001K", "2M": "2001K", "3.5M": "3501K", "5M": "5M", "final": "5M"}

    bs = args.batch_size

    for cp in checkpoints:
        cpKey = cpNameMap.get(cp, cp)

        snapshotContexts = {}
        snapshotTracking = {}
        for sk in strategyKeys:
            sName = STRATEGIES[sk][0]
            stratDir = outDir / "strategies" / f"{sk}_{sName}"
            if cpKey == "5M":
                ctxFile = stratDir / "final_context.json"
                trackFile = stratDir / "fact_tracking.json"
            else:
                snapDir = stratDir / f"snapshot_{cpKey}"
                ctxFile = snapDir / "context.json"
                trackFile = snapDir / "tracking.json"
            if not ctxFile.exists():
                print(f"  [{sk}@{cp}] SKIP: snapshot not found")
                continue
            with open(ctxFile, encoding="utf-8") as f:
                msgs = json.load(f)
            with open(trackFile, encoding="utf-8") as f:
                tracking = json.load(f)
            snapshotContexts[sk] = msgs
            snapshotTracking[sk] = tracking

        if not snapshotContexts:
            continue

        cpDir = outDir / (f"checkpoint_{cpKey}" if cpKey != "5M" else "final_reeval")
        print(f"\n  === EVAL @{cp} ===")

        fedFactIds = set()
        for sk in snapshotContexts:
            fedFactIds |= get_fed_facts(snapshotTracking[sk])
        cpFacts = [factById[fid] for fid in fedFactIds if fid in factById]
        cpFactMeta = [factMetaById[fid] for fid in fedFactIds if fid in factMetaById]
        print(f"  Facts fed: {len(cpFacts)}/{len(facts)}")

        # Grep
        for sk in snapshotContexts:
            gResult = grep_keywords(snapshotContexts[sk], cpFacts)
            gSummary = summarize_grep(gResult)
            save_json(gResult, cpDir / "grep" / f"{sk}.json")
            print(f"  [{sk}] grep: {gSummary['recall_upper_bound']:.1%}")

        # QA
        qaRequests = []
        for sk in snapshotContexts:
            ctx = snapshotContexts[sk]
            for bIdx in range(0, len(cpFacts), bs):
                batch = cpFacts[bIdx:bIdx + bs]
                questionsText = "\n".join(
                    f"- [{fid}] {q}" for fid, q, _, _, _ in batch
                )
                prompt = BATCH_QUESTION_PROMPT.format(questions=questionsText)
                reqMessages = ctx + [{"role": "user", "content": prompt}]
                qaRequests.append({
                    "custom_id": f"qa_{sk}_bs{bs}_b{bIdx // bs}",
                    "params": {
                        "model": args.model,
                        "max_tokens": 4096,
                        "system": SYSTEM_PROMPT,
                        "messages": reqMessages,
                    },
                })
        print(f"  QA: {len(qaRequests)} requests")
        qaResults = qaBackend.run_requests(qaRequests)

        answersByStrategy = {sk: {} for sk in snapshotContexts}
        parse_failures = 0
        for cid, result in qaResults.items():
            parts = cid.split("_")
            sk = parts[1]
            batchIdx = int(parts[3][1:])
            if result["status"] != "succeeded":
                continue
            try:
                answers = parse_llm_json(result["text"])
                answerMap = {a["id"]: a.get("answer", "[parse error]") for a in answers}
            except Exception as e:
                parse_failures += 1
                answerMap = {}
            batchStart = batchIdx * bs
            batchFacts = cpFacts[batchStart:batchStart + bs]
            for fid, _, _, _, _ in batchFacts:
                answersByStrategy[sk][fid] = answerMap.get(fid, "[no answer]")
        if parse_failures:
            print(f"  WARNING: {parse_failures} QA parse failures")

        for sk in answersByStrategy:
            save_json(answersByStrategy[sk], cpDir / "answers" / f"{sk}_bs{bs}.json")

        # Judge
        judgeBatchSize = 15
        judgeRequests = []
        for sk in snapshotContexts:
            entries = []
            for fid, q, ans, kw, qt in cpFacts:
                llmAns = answersByStrategy[sk].get(fid, "[no answer]")
                entries.append({
                    "id": fid, "question": q,
                    "expected_answer": ans, "expected_keywords": kw,
                    "llm_answer": llmAns,
                })
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
        print(f"  Judge: {len(judgeRequests)} requests")
        judgeResults = judgeBackend.run_requests(judgeRequests)

        verdictsByStrategy = {sk: [] for sk in snapshotContexts}
        for cid, result in judgeResults.items():
            parts = cid.split("_")
            sk = parts[1]
            if result["status"] != "succeeded":
                continue
            try:
                verdicts = parse_llm_json(result["text"])
                for v in verdicts:
                    v["fact_id"] = v.pop("id", v.get("fact_id", "?"))
                verdictsByStrategy[sk].extend(verdicts)
            except Exception:
                pass

        for sk in verdictsByStrategy:
            save_json({"verdicts": verdictsByStrategy[sk]},
                      cpDir / "judgments" / f"{sk}_bs{bs}.json")

        evalResults = {}
        for sk in snapshotContexts:
            metrics = compute_metrics(
                verdictsByStrategy[sk], cpFactMeta,
                tracking=snapshotTracking[sk],
            )
            evalResults[sk] = metrics
            print(f"  [{sk}] Recall: {metrics['recall']:.1%} ({metrics['facts_recalled']}/{metrics['facts_total']})")

        save_json({"results": evalResults, "label": cp,
                   "judge_model": args.judge_model,
                   "qa_model": args.model,
                   "rerun_timestamp": datetime.now().isoformat()},
                  cpDir / "summary.json")

    print(f"\n  Done! Results in: {outDir}")


if __name__ == "__main__":
    main()
