#!python3
"""
Re-evaluate specific checkpoints from saved snapshots.

Use case: when QA/judge phase failed (rate limit, gateway error) but the
feed+compact phase succeeded and snapshots are saved on disk.

Usage:
    ./reeval_checkpoints.py iterative_v6_R4_20260427_2249 \
        --strategies S3,S4 \
        --checkpoints 2001K,3501K,5M \
        --backend wrapper --workers 1 \
        --judge-model claude-haiku-4-5-20251001
"""

import json
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


# Reuse code from main bench
from benchmark_iterative_v6 import (
    BATCH_QUESTION_PROMPT, BATCH_JUDGE_PROMPT, JUDGE_SYSTEM, SYSTEM_PROMPT,
    STRATEGIES, parse_llm_json, save_json,
    grep_keywords, summarize_grep, compute_metrics, extract_facts,
)


def load_snapshot(stratDir, cpName):
    """Load a checkpoint snapshot. cpName is '500K', '1001K', '5M' (final), etc."""
    if cpName in ("5M", "final"):
        ctxFile = stratDir / "final_context.json"
        trackFile = stratDir / "fact_tracking.json"
        with open(ctxFile, encoding="utf-8") as f:
            messages = json.load(f)
        with open(trackFile, encoding="utf-8") as f:
            tracking = json.load(f)
        return messages, tracking, None  # fed_tokens unknown for final
    else:
        snapDir = stratDir / f"snapshot_{cpName}"
        ctxFile = snapDir / "context.json"
        trackFile = snapDir / "tracking.json"
        metaFile = snapDir / "meta.json"
        with open(ctxFile, encoding="utf-8") as f:
            messages = json.load(f)
        with open(trackFile, encoding="utf-8") as f:
            tracking = json.load(f)
        meta = {}
        if metaFile.exists():
            with open(metaFile, encoding="utf-8") as f:
                meta = json.load(f)
        return messages, tracking, meta.get("fed_tokens")


def get_fed_facts(tracking):
    """Return set of fact_ids that have been fed (status != not_yet_fed)."""
    fed = set()
    for fid, info in tracking.get("fact_status", {}).items():
        if info["status"] != "not_yet_fed":
            fed.add(fid)
    return fed


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate failed checkpoints from saved snapshots")
    parser.add_argument("run_dir", help="Path to run directory (e.g. iterative_v6_R4_20260427_2249)")
    parser.add_argument("--strategies", default="S3,S4")
    parser.add_argument("--checkpoints", default="500K,1001K,2001K,3501K,5M",
                        help="Checkpoints to evaluate (5M = final)")
    parser.add_argument("--batch-size", type=int, default=5, help="QA questions per batch")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--backend", default="wrapper")
    parser.add_argument("--judge-backend", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only-failed", action="store_true",
                        help="Only re-eval checkpoints that have FAIL status (>50% no_answer)")
    parser.add_argument("--conversation", default=None,
                        help="Path to conversation JSON (auto-detected if omitted)")
    args = parser.parse_args()

    if args.judge_backend is None:
        args.judge_backend = args.backend

    runDir = Path(args.run_dir)
    if not runDir.exists():
        print(f"ERROR: run dir not found: {runDir}")
        sys.exit(1)

    strategyKeys = [s.strip() for s in args.strategies.split(",")]
    checkpoints = [c.strip() for c in args.checkpoints.split(",")]

    # Load original conversation metadata to get fact list
    configFile = runDir / "config.json"
    with open(configFile) as f:
        config = json.load(f)

    convDensity = config["density"]
    convSeed = config["seed"]

    if args.conversation:
        convFile = Path(args.conversation)
        metaFile = convFile.parent / (convFile.stem + "_meta.json")
    else:
        # Auto-detect: round conversation_tokens up to nearest 500K, then format
        convTokens = config.get("conversation_tokens", 5_000_000)
        # Round up to nearest 500K
        rounded = ((convTokens + 499_999) // 500_000) * 500_000
        candidates = []
        for tot in (rounded, rounded + 500_000, convTokens):
            label = (f"{tot // 1_000_000}M" if tot >= 1_000_000 else f"{tot // 1_000}K")
            candidates.append(label)
        convFile = None
        for label in candidates:
            cf = Path(f"data/conversations/v6_R4/d{convDensity}_{label}_seed{convSeed}.json")
            if cf.exists():
                convFile = cf
                break
        if convFile is None:
            # Fallback: glob for matching density+seed
            from glob import glob
            matches = glob(f"data/conversations/v6_R4/d{convDensity}_*_seed{convSeed}.json")
            if matches:
                convFile = Path(matches[0])
            else:
                print(f"ERROR: cannot find conversation file for d{convDensity} seed{convSeed}")
                sys.exit(1)
        metaFile = convFile.parent / (convFile.stem + "_meta.json")
    print(f"  Conv: {convFile}")

    with open(metaFile, encoding="utf-8") as f:
        convMeta = json.load(f)
    facts = extract_facts(convMeta)
    factMeta = convMeta["facts"]
    factById = {f[0]: f for f in facts}
    factMetaById = {fm["fact_id"]: fm for fm in factMeta}

    print(f"  Run: {runDir.name}")
    print(f"  Strategies: {strategyKeys}")
    print(f"  Checkpoints: {checkpoints}")
    print(f"  Total facts: {len(facts)}")

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

    # Map checkpoint name to label/dir
    cpNameMap = {"500K": "500K", "1001K": "1001K", "2001K": "2001K", "3501K": "3501K",
                 "1M": "1001K", "2M": "2001K", "3.5M": "3501K", "5M": "5M", "final": "5M"}

    # For each checkpoint × strategy, evaluate if needed
    for cp in checkpoints:
        cpKey = cpNameMap.get(cp, cp)
        cpDir = runDir / (f"checkpoint_{cpKey}" if cpKey != "5M" else "final")
        # 5M has no checkpoint_5M dir — use main run dir for "final" results
        if cpKey == "5M":
            cpDir = runDir / "final_reeval"

        # Load snapshots for all strategies
        snapshotContexts = {}
        snapshotTracking = {}
        for sk in strategyKeys:
            sName = STRATEGIES[sk][0]
            stratDir = runDir / "strategies" / f"{sk}_{sName}"
            try:
                msgs, tracking, fedTok = load_snapshot(stratDir, cpKey)
                snapshotContexts[sk] = msgs
                snapshotTracking[sk] = tracking
                print(f"  [{sk}@{cp}] loaded: {len(msgs)} msgs, fedTok={fedTok}")
            except FileNotFoundError as e:
                print(f"  [{sk}@{cp}] SKIP: snapshot not found ({e})")
                continue

        if not snapshotContexts:
            continue

        # If --only-failed, filter strategies per checkpoint:
        # only re-eval strategies whose answers file is missing OR has >50% no_answer
        if args.only_failed:
            toEval = {}
            for sk in list(snapshotContexts.keys()):
                ansFile = cpDir / "answers" / f"{sk}_bs{args.batch_size}.json"
                needs_eval = True
                if ansFile.exists():
                    with open(ansFile) as f:
                        a = json.load(f)
                    no_ans = sum(1 for v in a.values() if v == "[no answer]")
                    # Need eval if missing answers OR more than 50% [no answer]
                    if len(a) > 0 and no_ans <= len(a) * 0.5:
                        needs_eval = False
                if needs_eval:
                    toEval[sk] = (snapshotContexts[sk], snapshotTracking[sk])
                    print(f"  [{sk}@{cp}] needs re-eval")
                else:
                    print(f"  [{sk}@{cp}] OK, skipping")
            if not toEval:
                print(f"  Skipping {cp} (all strategies OK)")
                continue
            # Restrict snapshot dicts to strategies needing re-eval
            snapshotContexts = {sk: ctx for sk, (ctx, _) in toEval.items()}
            snapshotTracking = {sk: tr for sk, (_, tr) in toEval.items()}

        # Determine fed facts at this checkpoint
        fedFactIds = set()
        for sk in snapshotContexts:
            fedFactIds |= get_fed_facts(snapshotTracking[sk])
        cpFacts = [factById[fid] for fid in fedFactIds if fid in factById]
        cpFactMeta = [factMetaById[fid] for fid in fedFactIds if fid in factMetaById]
        print(f"\n  === EVAL @{cp} === ({len(cpFacts)}/{len(facts)} facts fed)")

        # Phase: Grep
        for sk in snapshotContexts:
            gResult = grep_keywords(snapshotContexts[sk], cpFacts)
            gSummary = summarize_grep(gResult)
            save_json(gResult, cpDir / "grep" / f"{sk}.json")
            print(f"  [{sk}] grep: {gSummary['recall_upper_bound']:.1%}")

        # Phase: QA
        bs = args.batch_size
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

        # Parse answers
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
                print(f"  WARNING: {cid} parse error: {e}")
                print(f"    raw text head: {result['text'][:200]!r}")
                answerMap = {}
            batchStart = batchIdx * bs
            batchFacts = cpFacts[batchStart:batchStart + bs]
            for fid, _, _, _, _ in batchFacts:
                answersByStrategy[sk][fid] = answerMap.get(fid, "[no answer]")
        if parse_failures:
            print(f"  WARNING: {parse_failures} QA responses failed JSON parse")

        for sk in answersByStrategy:
            save_json(answersByStrategy[sk], cpDir / "answers" / f"{sk}_bs{bs}.json")

        # Phase: Judge
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
                            BATCH_JUDGE_PROMPT.format(entries=json.dumps(batch, indent=2))}],
                    },
                })
        print(f"  Judge: {len(judgeRequests)} requests")
        judgeResults = judgeBackend.run_requests(judgeRequests)

        # Parse verdicts
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

        # Phase: Metrics
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
                   "reeval_timestamp": datetime.now().isoformat()},
                  cpDir / "summary_reeval.json")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
